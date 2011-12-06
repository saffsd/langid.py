#!/usr/bin/env python
"""
LDfeatureselect.py - 
LD (Lang-Domain) feature extractor
Marco Lui November 2011

Based on research by Marco Lui and Tim Baldwin.

Copyright 2011 Marco Lui <saffsd@gmail.com>. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of the copyright holder.
"""

######
# Default values
# Can be overriden with command-line options
######
MAX_NGRAM_ORDER = 4 # largest order of n-grams to consider
TOP_DOC_FREQ = 15000 # number of tokens to consider for each order
FEATURES_PER_LANG = 300 # number of features to select for each language

import os, sys, optparse
import collections
import csv
import shutil
import tempfile
import marshal
import numpy
import cPickle
import multiprocessing as mp
from itertools import tee, izip, repeat
from collections import defaultdict
from datetime import datetime

class disklist(collections.Iterable, collections.Sized):
  """
  Disk-backed queue. Not to be used for object persistence.
  Items can be added to the queue, and the queue can be iterated,
  """
  def __init__(self, temp_dir=None):
    self.fileh = tempfile.TemporaryFile(dir=temp_dir)
    self.count = 0

  def __iter__(self):
    self.fileh.seek(0)
    while True:
      try:
        yield marshal.load(self.fileh)
      except (EOFError, ValueError, TypeError):
        break

  def __len__(self):
    return self.count

  def append(self, value):
    marshal.dump(value, self.fileh)
    self.count += 1


class Tokenizer(object):
  def __init__(self, max_order):
    self.max_order = max_order

  def __call__(self, seq):
    max_order = self.max_order
    t = tee(seq, max_order)
    for i in xrange(max_order):
      for j in xrange(i):
        # advance iterators, ignoring result
        t[i].next()
    while True:
      token = tuple(tn.next() for tn in t)
      if len(token) < max_order: break
      for n in xrange(max_order):
        yield token[:n+1]
    for a in xrange(max_order-1):
      for b in xrange(1, max_order-a):
        yield token[a:a+b]

class Enumerator(object):
  """
  Enumerator object. Returns a larger number each call. 
  Can be used with defaultdict to enumerate a sequence of items.
  """
  def __init__(self, start=0):
    self.n = start

  def __call__(self):
    retval = self.n
    self.n += 1
    return retval

def entropy(v, axis=0):
  """
  Optimized implementation of entropy. This version is faster than that in 
  scipy.stats.distributions, particularly over long vectors.
  """
  v = numpy.array(v, dtype='float')
  s = numpy.sum(v, axis=axis)
  with numpy.errstate(divide='ignore', invalid='ignore'):
    r = numpy.log(s) - numpy.nansum(v * numpy.log(v), axis=axis) / s
  return r


def split_info(arg):
  """
  Helper for the infogain class. This lives as its own top-level function
  to allow it to work with multiprocessing.
  """
  f_masks, class_map = arg
  num_inst = f_masks.shape[1]
  f_count = f_masks.sum(1) # sum across instances
  f_weight = f_count / float(num_inst) 
  f_entropy = numpy.empty((f_masks.shape[0], f_masks.shape[2]), dtype=float)
  # TODO: This is the main cost. See if this can be made faster. 
  for i, band in enumerate(f_masks):
    f_entropy[i] = entropy((class_map[:,None,:] * band[...,None]).sum(0), axis=-1)
  # nans are introduced by features that are entirely in a single band
  # We must redefine this to 0 as otherwise we may lose information about other bands.
  # TODO: Push this back into the definition of entropy?
  f_entropy[numpy.isnan(f_entropy)] = 0
  return (f_weight * f_entropy).sum(0) #sum across discrete bands



class InfoGain(object):
  def __init__(self, chunksize=50, num_process=None):
    self.chunksize = chunksize
    self.num_process = num_process if num_process else mp.cpu_count()
 
  def weight(self, feature_map, class_map):
    # Feature map should be a boolean map
    num_inst, num_feat = feature_map.shape

    # We can eliminate unused classes as they do not contribute to entropy
    class_map = class_map[:,class_map.sum(0) > 0]
    
    # Calculate  the entropy of the class distribution over all instances 
    H_P = entropy(class_map.sum(0))
      
    # unused features have 0 information gain, so we skip them
    nz_index = numpy.array(feature_map.sum(0).nonzero())[0]
    nz_fm = feature_map[:, nz_index]
    nz_num = len(nz_index)

    # compute the information gain of nonzero features
    pool = mp.Pool(self.num_process)
    def chunks():
      for chunkstart in range(0, nz_num, self.chunksize):
        chunkend = min(nz_num, chunkstart+self.chunksize)
        v = nz_fm[:,chunkstart:chunkend]
        nonzero = numpy.zeros(v.shape, dtype=bool)
        nonzero[v.nonzero()] = True
        zero = numpy.logical_not(nonzero)
        retval = numpy.concatenate((zero[None], nonzero[None]))
        yield (retval, class_map)
    x = pool.imap(split_info, chunks())
    nz_fw = H_P - numpy.hstack(x)

    # return 0 for unused features
    feature_weights = numpy.zeros(num_feat, dtype=float)
    feature_weights[nz_index] = nz_fw
    return feature_weights

def set_maxorder(arg):
  global __maxorder
  __maxorder = arg

def pass1(pathschunk):
  global __maxorder
  extractor = Tokenizer(__maxorder)
  doc_count = defaultdict(int)
  doc_reprs = []
  token_index = {}
  token_count = 0

  for path in pathschunk:
    with open(path) as f:
      tokenset = set(extractor(f.read()))
      doc_repr = set()
      for token in tokenset:
        doc_count[token] += 1
        if token not in token_index:
          token_index[token] = token_count
          token_count += 1
        doc_repr.add(token_index[token])
      doc_reprs.append(doc_repr)
  return doc_count, token_index, doc_reprs

def write_weights(path, weights):
  w = dict(weights)
  with open(path, 'w') as f:
    writer = csv.writer(f, delimiter = '\t')
    for k in sorted(w, key=w.get, reverse=True):
      writer.writerow((repr(k), w[k]))

class ClassIndexer(object):
  def __init__(self, paths):
    self.lang_index = defaultdict(Enumerator())
    self.domain_index = defaultdict(Enumerator())
    self.doc_keys = []
    self.index_paths(paths)

  def index_paths(self, paths):
    for path in paths:
      # split the path into identifying components
      path, docname = os.path.split(path)
      path, lang = os.path.split(path)
      path, domain = os.path.split(path)

      # obtain a unique key for the file
      key = domain,lang,docname
      self.doc_keys.append(key)

      # index the language and the domain
      lang_id = self.lang_index[lang]
      domain_id = self.domain_index[domain]

  def get_class_maps(self):
    num_instances = len(self.doc_keys)
    cm_domain = numpy.zeros((num_instances, len(self.domain_index)), dtype='bool')
    cm_lang = numpy.zeros((num_instances, len(self.lang_index)), dtype='bool')

    # Populate the class maps
    for docid, (domain, lang, docname) in enumerate(self.doc_keys):
      cm_domain[docid, self.domain_index[domain]] = True
      cm_lang[docid, self.lang_index[lang]] = True
    return cm_domain, cm_lang

def select_LD_features(features, lang_index, feature_map, cm_domain, cm_lang, options):
  print "computing information gain"
  # Compute the information gain WRT domains and binary for each language
  ig = InfoGain(num_process=options.job_count)
  w_domain = ig.weight(feature_map, cm_domain)
  if options.weights:
    write_weights(os.path.join(options.weights, 'domain'), zip(features, w_domain))

  final_feature_set = set()
  for lang in lang_index:
    print "computing infogain: ", lang
    start_t = datetime.now()
    pos = cm_lang[:, lang_index[lang]]
    neg = numpy.logical_not(pos)
    cm = numpy.hstack((neg[:,None], pos[:,None]))
    w_lang = ig.weight(feature_map, cm)
    items = zip(features, w_lang - w_domain)
    ld_w = dict(items)
    final_feature_set |= set(sorted(ld_w, key=ld_w.get, reverse=True)[:options.feats_per_lang])
    print "  done! duration: %ss" % str(datetime.now() - start_t)
    if options.weights:
      path = os.path.join(options.weights, lang)
      write_weights(path, items)
      print '  output %s weights to: "%s"' % (lang, path)
  return final_feature_set

def get_classmaps(paths):
  indexer = ClassIndexer(paths)
  cm_domain, cm_lang = indexer.get_class_maps()
  print "langs:", indexer.lang_index.keys()
  print "domains:", indexer.domain_index.keys()
  return cm_domain, cm_lang, indexer.lang_index 

def get_featuremap(paths, options):

  # First pass: Construct candidate set of types
  def chunk(seq, chunksize):
    seq_iter = iter(seq)
    while True:
      chunk = tuple(seq_iter.next() for i in range(chunksize))
      if len(chunk) == 0:
        break
      yield chunk


  pool = mp.Pool(options.job_count, set_maxorder, (options.max_order,))

  # Tokenize documents into sets of terms
  # TODO: Dynamic selection of chunk size depending on input count
  chunk_size = min(len(paths) / (options.job_count*2), 1000)
  chunk_out = pool.imap_unordered(pass1, chunk(paths, chunk_size), chunksize=1)
  pool.close()
  doc_count = defaultdict(int)
  doc_reprs = disklist() # list of lists of termid-count pairs
  processed = 0
  total = len(paths)/chunk_size + (0 if len(paths)%chunk_size else 1)

  print "chunk size: %d (%d chunks)" % (chunk_size, total)
  for count, term_index, reprs in chunk_out:
    for term in count:
      doc_count[term] += count[term]
    doc_reprs.append((term_index, reprs))
    processed += 1
    print "Processed %d of %d chunks" % (processed, total)
  print "unique features: ", len(doc_count)

  # Work out the set of features to compute IG
  features = set()
  for i in range(1, options.max_order+1):
    d = dict( (k, doc_count[k]) for k in doc_count if len(k) == i)
    features |= set(sorted(d, key=d.get, reverse=True)[:options.df_tokens])
  features = sorted(features)
  print "candidate features: ", len(features)

  # Initialize feature and class maps 
  num_instances = len(paths)
  feature_map = numpy.zeros((num_instances, len(features)), dtype='bool')
  
  # Populate the feature map
  docid = 0
  for term_index, reprs in doc_reprs:
    index_feats = dict((term_index[f],i) for i,f in enumerate(features) if f in term_index)
    index_terms = set(index_feats)
    for doc_repr in reprs:
      for ind in set(doc_repr) & index_terms:
        featid = index_feats[ind]
        feature_map[docid, featid] = True
      docid += 1
  print "feature map sum:", feature_map.sum()

  # Convert features from character tuples to strings
  features = [ ''.join(f) for f in features ]
  return feature_map, features

if __name__ == "__main__":
  parser = optparse.OptionParser()
  parser.add_option("-o","--output", dest="outfile", help="output features to FILE", metavar="FILE")
  parser.add_option("-c","--corpus", dest="corpus", help="read corpus from DIR", metavar="DIR")
  parser.add_option("-j","--jobs", dest="job_count", type="int", help="number of processes to use", default=mp.cpu_count())
  parser.add_option("-w","--weights",dest="weights", help="output weights to DIR (optional)", metavar="DIR")
  parser.add_option("-s","--save",dest="save", help="pickle an intermediate model to FILE", metavar="FILE")
  parser.add_option("-l","--load",dest="load", help="load an intermediate model from FILE", metavar="FILE")
  parser.add_option("--max_order", dest="max_order", type="int", help="highest n-gram order to use", default=MAX_NGRAM_ORDER)
  parser.add_option("--feats_per_lang", dest="feats_per_lang", type="int", help="number of features to retain for each language", default=FEATURES_PER_LANG)
  parser.add_option("--df_tokens", dest="df_tokens", type="int", help="number of tokens to consider for each n-gram order", 
  default=TOP_DOC_FREQ)

  options, args = parser.parse_args()

  # check options
  if not (options.corpus or options.load):
    parser.error("corpus(-c) or intermediate model(-l) must be specified")

  if options.save and options.load:
    parser.error("can only specify one of -l or -s")

  if options.weights:
    if not os.path.exists(options.weights):
      os.mkdir(options.weights)

  # work out output path
  if options.outfile:
    output_path = options.outfile 
  elif options.corpus:
    if os.path.basename(options.corpus):
      output_path = os.path.basename(options.corpus+'.LDfeatures')
    else:
      output_path = options.corpus+'.LDfeatures'
  else:
    output_path = 'a.LDfeatures'

  # display paths
  print "output path:", output_path
  if options.corpus:
    print "corpus path:", options.corpus
  if options.weights:
    print "weights path:", options.weights
  if options.load:
    print "load intermediate:", options.load
  if options.save:
    print "save intermediate:", options.save



  # Obtain tokenized representation
  if options.load:
    with open(options.load) as f:
      (features, lang_index, fm, cm_domain, cm_lang) = cPickle.load(f)
    print 'loaded intermediate model: "%s"' % options.load
  else:
    # build a list of paths
    paths = []
    for dirpath, dirnames, filenames in os.walk(options.corpus):
      for f in filenames:
        paths.append(os.path.join(dirpath, f))
    print "will tokenize %d files" % len(paths)

    cm_domain, cm_lang, lang_index = get_classmaps(paths)
    fm, features = get_featuremap(paths, options)

  # Save tokenized representaion
  if options.save:
    with open(options.save, 'w') as f:
      cPickle.dump((features, lang_index, fm, cm_domain, cm_lang), f)
      print 'wrote intermediate model: "%s"' % options.save

  #features = [ 'f%5d' % f for f in range(0,45256)]
  #fm = numpy.empty((cm_domain.shape[0], len(features)), dtype='bool')
  # Compute LD from tokenized representation
  try:
    final_feature_set = select_LD_features(features, lang_index, fm, cm_domain, cm_lang, options)
  except OSError, e:
    print e
    import pdb;pdb.pm()
 
        
  # Output
  print "selected %d features" % len(final_feature_set)

  with open(output_path,'w') as f:
    for feat in final_feature_set:
      print >>f, repr(feat)
    print 'wrote features to "%s"' % output_path 
    

