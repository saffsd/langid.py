from hydrat.common.distance_metrics import distance_metric
import numpy

class dm_skew_new(distance_metric):
  # TODO: Implement dimensionality reduction as per cosine.
  __name__ = "skew_new"

  def __init__(self, alpha = 0.99):
    distance_metric.__init__(self)
    self.alpha = alpha

  def _params(self):
    p = distance_metric._params(self)
    p['alpha'] = self.alpha
    return p

  def vector_distances(self, v1, v2, thresh=10):
    self.logger.debug('Creating dense representation')
    orig_feats = v1.shape[1]
    k = numpy.log(1 / (1 - self.alpha))

    # Calculate instance sums
    sum1 = numpy.array(v1.sum(1))
    sum2 = numpy.array(v2.sum(1))

    # Determine shared features
    f1 = numpy.asarray( v1.sum(0) ) [0] > 0
    f2 = numpy.asarray( v2.sum(0) ) [0] > 0
    nb = numpy.flatnonzero(numpy.logical_and(f1, f2)) # both nonzero
    n1 = numpy.flatnonzero(numpy.logical_and(f1, numpy.logical_not(f2))) # only f1 nonzero
      
    if len(nb) == 0:
      # No overlapping features.
      return numpy.empty((v1.shape[0],v2.shape[0]), dtype='float')

    # Calculate term for where v2 is always zero but v1 is not
    #   This is a 1-dimensional vector, one scalar value per v1 instance
    penalty = numpy.array(v1[:,n1].sum(1))

    # Select only shared features from both matrices
    v1 = v1.transpose()[nb].transpose().toarray()
    v2 = v2.transpose()[nb].transpose().toarray()
    self.logger.debug('Reduced matrices from %d to %d features', orig_feats, v1.shape[1] )

    # Replace empty distributions with uniform distributions
    for i in numpy.flatnonzero(sum1 == 0):
      v1[i] = numpy.ones_like(v1[i])
      sum1[i] = v1[i].sum()
    for i in numpy.flatnonzero(sum2 == 0):
      v2[i] = numpy.ones_like(v1[i])
      sum2[i] = v2[i].sum()

    # Calculate term for shared features
    acc_a  = numpy.zeros((v1.shape[0], v2.shape[0]))
    a = self.alpha
    sum_ratio = (sum1.astype(float) / sum2.transpose())[...,numpy.newaxis]

    # Alias to a familiar notation for computing skew divergence
    p = v1[:,numpy.newaxis]
    q = v2

    with numpy.errstate(invalid='ignore', divide='ignore'):
      r = p * numpy.log( p / (sum_ratio * a * q + (1-a) * p) )

    # We must ignore values where p and q are not both nonzero
    acc_a = numpy.nansum(r * numpy.logical_and(p>0, q>0),axis=2)

    # This is just the sum of p where q is zero. 
    acc_b  = numpy.dot(v1, (v2==0).transpose())

    retval = ( acc_a + k * (acc_b + penalty) ) / sum1

    self.logger.debug('Returning Results')
    return retval

class dm_skew_newer(distance_metric):
  # This version attempts to avoid wasteful use of memory in the computation of acc_a
  # by returning some of the outer looping to python space
  __name__ = "skew_newer"

  def __init__(self, alpha = 0.99):
    distance_metric.__init__(self)
    self.alpha = alpha

  def _params(self):
    p = distance_metric._params(self)
    p['alpha'] = self.alpha
    return p

  def vector_distances(self, v1, v2, thresh=10):
    self.logger.debug('Creating dense representation')
    orig_feats = v1.shape[1]
    k = numpy.log(1 / (1 - self.alpha))

    # Calculate instance sums
    sum1 = numpy.array(v1.sum(1))
    sum2 = numpy.array(v2.sum(1))

    # Determine shared features
    f1 = numpy.asarray( v1.sum(0) ) [0] > 0
    f2 = numpy.asarray( v2.sum(0) ) [0] > 0
    nb = numpy.flatnonzero(numpy.logical_and(f1, f2)) # both nonzero
    n1 = numpy.flatnonzero(numpy.logical_and(f1, numpy.logical_not(f2))) # only f1 nonzero
      
    if len(nb) == 0:
      # No overlapping features.
      return numpy.empty((v1.shape[0],v2.shape[0]), dtype='float')

    # Calculate term for where v2 is always zero but v1 is not
    #   This is a 1-dimensional vector, one scalar value per v1 instance
    penalty = numpy.array(v1[:,n1].sum(1))

    # Select only shared features from both matrices
    v1 = v1.transpose()[nb].transpose().toarray()
    v2 = v2.transpose()[nb].transpose().toarray()
    self.logger.debug('Reduced matrices from %d to %d features', orig_feats, v1.shape[1] )

    # Replace empty distributions with uniform distributions
    for i in numpy.flatnonzero(sum1 == 0):
      v1[i] = numpy.ones_like(v1[i])
      sum1[i] = v1[i].sum()
    for i in numpy.flatnonzero(sum2 == 0):
      v2[i] = numpy.ones_like(v1[i])
      sum2[i] = v2[i].sum()

    # Calculate term for shared features
    # TODO: Possibly roll one loop back into numpy for speed
    a = self.alpha
    sum_ratio = (sum1.astype(float) / sum2.transpose())
    acc_a = numpy.empty((v1.shape[0], v2.shape[0]))
    for ip, _p in enumerate(v1):
      pg = _p > 0
      for iq, _q in enumerate(v2):
        feats = numpy.logical_and(pg, _q>0)
        p = _p[feats]
        q = _q[feats]
        v = p * numpy.log( p / (sum_ratio[ip,iq] * a * q + (1-a) * p) )
        acc_a[ip, iq] = numpy.sum(v)

    # This is just the sum of p where q is zero. 
    acc_b  = numpy.dot(v1, (v2==0).transpose())

    retval = ( acc_a + k * (acc_b + penalty) ) / sum1

    self.logger.debug('Returning Results')
    return retval
