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

def pass1(path):
  """
  Read a file on disk and return the set of types it contains.
  """
  global __maxorder
  extractor = Tokenizer(__maxorder)
  with open(path) as f:
    retval = set(extractor(f.read()))
  return retval

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

def compute_infogain(features, lang_index, feature_map, cm_domain, cm_lang, options):
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
  def get_paths():
    for i,p in enumerate(paths):
      if i % 100 == 0:
        print "%d..." % i
      yield p
    print "Done Producing!"

  pool = mp.Pool(options.job_count, set_maxorder, (options.max_order,))

  # Tokenize documents into sets of terms
  termsets = pool.imap(pass1, get_paths(), chunksize=1)
  pool.close()
  doc_count = defaultdict(int)
  term_index = defaultdict(Enumerator())
  doc_reprs = disklist() # list of lists of termid-count pairs
  count = 0
  for termset in termsets:
    doc_repr = []
    for term in termset:
      doc_count[term] += 1
      doc_repr.append(term_index[term])
    doc_reprs.append(set(doc_repr))
    count += 1
    if count % 100 == 0:
      print "Consumed %d..." % count
  print "Done Consuming"

  # Work out the set of features to compute IG
  candidate_features = set()
  for i in range(1, options.max_order+1):
    d = dict( (k, doc_count[k]) for k in doc_count if len(k) == i)
    candidate_features |= set(sorted(d, key=d.get, reverse=True)[:options.df_tokens])
  candidate_features = sorted(candidate_features)
  print "candidate features: ", len(candidate_features)


  # Compute indices of features to retain
  feats = tuple(term_index[f] for f in candidate_features)

  # Initialize feature and class maps 
  num_instances = len(paths)
  feature_map = numpy.zeros((num_instances, len(candidate_features)), dtype='bool')
  
  print "doc_reprs:", len(doc_reprs)
  print "feats:", len(feats)

  # Populate the feature map
  for docid, doc_repr in enumerate(doc_reprs):
    for featid, termid in enumerate(feats):
      feature_map[docid, featid] = termid in doc_repr
  print "feature map sum:", feature_map.sum()

  # Convert features from character tuples to strings
  features = [ ''.join(f) for f in candidate_features ]
  return feature_map, features

if __name__ == "__main__":
  parser = optparse.OptionParser()
  parser.add_option("-o","--output", dest="outfile", help="output features to FILE", metavar="FILE")
  parser.add_option("-c","--corpus", dest="corpus", help="read corpus from DIR", metavar="DIR")
  parser.add_option("-j","--jobs", dest="job_count", type="int", help="number of processes to use", default=mp.cpu_count())
  parser.add_option("-w","--weights",dest="weights", help="output weights to DIR (optional)", metavar="DIR")
  parser.add_option("--max_order", dest="max_order", type="int", help="highest n-gram order to use", default=MAX_NGRAM_ORDER)
  parser.add_option("--feats_per_lang", dest="feats_per_lang", type="int", help="number of features to retain for each language", default=FEATURES_PER_LANG)
  parser.add_option("--df_tokens", dest="df_tokens", type="int", help="number of tokens to consider for each n-gram order", 
  default=TOP_DOC_FREQ)

  options, args = parser.parse_args()

  # check options
  if not options.corpus:
    parser.error("corpus must be specified")
    
  if options.weights:
    if not os.path.exists(options.weights):
      os.mkdir(options.weights)

  # work out output path
  if options.outfile:
    output_path = options.outfile 
  elif os.path.basename(options.corpus):
    output_path = os.path.basename(options.corpus+'.LDfeatures')
  else:
    output_path = options.corpus+'.LDfeatures'
  print "output path:", output_path

  # build a list of paths
  paths = []
  for dirpath, dirnames, filenames in os.walk(options.corpus):
    for f in filenames:
      paths.append(os.path.join(dirpath, f))
  print "corpus path:", options.corpus
  print "found %d files" % len(paths)

  fm, features = get_featuremap(paths, options)
  cm_domain, cm_lang, lang_index = get_classmaps(paths)
  final_feature_set = compute_infogain(features, lang_index, fm, cm_domain, cm_lang, options)
        
  # Output
  print "selected %d features" % len(final_feature_set)

  with open(output_path,'w') as f:
    for feat in final_feature_set:
      print >>f, repr(feat)
    print 'wrote features to "%s"' % output_path 
    

