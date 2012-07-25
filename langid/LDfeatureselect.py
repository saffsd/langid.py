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
NUM_BUCKETS = 64 # number of buckets to use in k-v pair generation

import os, sys, optparse
import collections
import csv
import shutil
import tempfile
import marshal
import numpy
import cPickle
import multiprocessing as mp
from itertools import tee, imap
from collections import defaultdict
from datetime import datetime
from contextlib import closing

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

def chunk(seq, chunksize):
  """
  Break a sequence into chunks not exceeeding a predetermined size
  """
  seq_iter = iter(seq)
  while True:
    chunk = tuple(seq_iter.next() for i in range(chunksize))
    if len(chunk) == 0:
      break
    yield chunk

def unmarshal_iter(f):
  """
  Iterator over a file object, which unmarshals
  item by item.
  """
  while True:
    try:
      yield marshal.load(f)
    except EOFError:
      break

def split_info(f_masks, class_map):
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

def infogain(nonzero, class_map):
  if nonzero.dtype != bool:
    raise TypeError, "expected a boolean feature map"

  # Feature map should be a boolean map
  num_inst, num_feat = nonzero.shape
  
  # Calculate  the entropy of the class distribution over all instances 
  H_P = entropy(class_map.sum(0))
    
  # compute information gain
  zero = numpy.logical_not(nonzero)
  x = numpy.concatenate((zero[None], nonzero[None]))
  feature_weights = H_P - split_info(x, class_map)
  return feature_weights

def setup_pass1(maxorder, b_freq, b_list, locks):
  global __maxorder, __b_freq, __b_list, __locks
  __maxorder = maxorder 
  __locks = locks
  __b_freq = b_freq
  __b_list = b_list


def pass1(arg):
  """
  Chunk files into a doc->term mapping,
  and simultaneously build a term->df count.
  The term->df counts are redistributed to
  buckets via python's in-built hash function.
  This is basically an inversion step, so that 
  now we are chunked on the term axis rather
  than the document axis.
  """
  global __maxorder, __b_freq, __b_list, __locks
  chunk_id, chunk_paths = arg
  
  extractor = Tokenizer(__maxorder)
  term_doc_freq = defaultdict(int)
  term_doc_list = defaultdict(list)

  for doc_index, path in enumerate(chunk_paths):
    with open(path) as f:
      tokenset = set(extractor(f.read()))
      for token in tokenset:
        term_doc_freq[token] += 1
        term_doc_list[token].append(doc_index)

  for key in term_doc_freq:
    bucket_index = hash(key) % len(__locks)
    with __locks[bucket_index]:
      os.write(__b_freq[bucket_index], marshal.dumps((key, term_doc_freq[key])))
      os.write(__b_list[bucket_index], marshal.dumps((key, chunk_id, term_doc_list[key])))

  return len(term_doc_freq)

def setup_pass2(maxorder, doc_freq_fd, lock):
  global __maxorder, __doc_freq, __doc_freq_lock
  __maxorder = maxorder
  __doc_freq = doc_freq_fd
  __doc_freq_lock = lock

def pass2(bucket):
  """
  Take a term->df count bucket and sum it up
  """
  global __maxorder, __doc_freq
  fileh = os.fdopen(bucket)
  doc_count = defaultdict(int)
  count = 0

  for key, value in unmarshal_iter(fileh):
    doc_count[key] += value
    count += 1
    
  with __doc_freq_lock:
    for item in doc_count.iteritems():
      os.write(__doc_freq, marshal.dumps(item))
  return count

def setup_pass3(features, chunk_offsets, cm_domain, cm_lang):
  global __features, __chunk_offsets, __cm_domain, __cm_lang
  __features = features
  __chunk_offsets = chunk_offsets
  __cm_domain = cm_domain
  __cm_lang = cm_lang

def pass3(chunk_path):
  """
  In this pass we actually compute information gain.
  For each chunk, we need to load up the corresponding feature map.
  This includes the filtering of top-DF features as identified in
  the previous pass.
  Then we compute information gain with respect to the domain
  class map and the binarized language class maps.
  """
  global __features, __chunk_offsets, __cm_domain, __cm_lang
   
  # Select only our listed features
  term_doc_map = defaultdict(list)
  with open(chunk_path) as f:
    for key, chunk_id, docids in unmarshal_iter(f):
      if key in __features:
        offset = __chunk_offsets[chunk_id]
        for docid in docids:
          term_doc_map[key].append(docid+offset)
  num_inst = __chunk_offsets[-1]
  num_feat = len(term_doc_map)

  # Build the feature map for the chunk
  feature_map = numpy.zeros((num_inst, num_feat), dtype=bool)
  terms = []
  for termid, term in enumerate(term_doc_map):
    terms.append(term)
    for docid in term_doc_map[term]:
      feature_map[docid, termid] = True

  # Compute information gain over all domains as well as binarized per-language
  w_lang = []
  w_domain = infogain(feature_map, __cm_domain)
  for langid in range(__cm_lang.shape[1]):
    pos = cm_lang[:, langid]
    neg = numpy.logical_not(pos)
    cm = numpy.hstack((neg[:,None], pos[:,None]))
    w = infogain(feature_map, cm)
    w_lang.append( w - w_domain )

  w_lang = numpy.vstack(w_lang)
  return terms, w_lang, w_domain

    


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

def select_LD_features(features, lang_index, chunk_paths, chunk_offsets, cm_domain, cm_lang, options):
  print "computing information gain"
  # Instead of receiving a single feature map, we now receive a list of paths,
  # each corresponding to a chunk containing a portion of the final feature set
  # for each of these chunks we need to compute the IG of each feature WRT to
  # domain as well as WRT to each language.
  # The parallelism should come at the feature chunk level,
  # so we can collapse IG into a non-parallelized function.

  #setup_pass3(features, chunk_offsets, cm_domain, cm_lang)
  #pass3_out = map(pass3, chunk_paths)

  with closing( mp.Pool(options.job_count, setup_pass3, 
                (features, chunk_offsets, cm_domain, cm_lang))
              ) as pool:
    pass3_out = pool.imap_unordered(pass3, chunk_paths, chunksize=1)

  num_chunk = len(chunk_paths)
  w_lang = []
  w_domain = []
  terms = []
  for i, (t, w_l, w_d) in enumerate(pass3_out):
    w_lang.append(w_l)
    w_domain.append(w_d)
    terms.extend(t)
    print "processed chunk (%d/%d)" % (i+1, num_chunk)
  w_lang = numpy.hstack(w_lang)
  w_domain = numpy.concatenate(w_domain)
  terms = ["".join(t) for t in terms]

  if options.weights:
    write_weights(os.path.join(options.weights, 'domain'), zip(terms, w_domain))

  # compile the final feature set
  final_feature_set = set()
  for lang in lang_index:
    lang_weights = w_lang[lang_index[lang]]
    term_inds = numpy.argsort(lang_weights)[-options.feats_per_lang:]
    for t in term_inds:
      final_feature_set.add(terms[t])
    if options.weights:
      path = os.path.join(options.weights, lang)
      write_weights(path, zip(terms,lang_weights))
      print '  output %s weights to: "%s"' % (lang, path)

  return final_feature_set
    

def get_classmaps(paths):
  indexer = ClassIndexer(paths)
  cm_domain, cm_lang = indexer.get_class_maps()
  print "langs:", indexer.lang_index.keys()
  print "domains:", indexer.domain_index.keys()
  return cm_domain, cm_lang, indexer.lang_index 

def build_inverted_index(paths, options):
  b_f = []
  b_l = []
  b_f_paths = []
  b_l_paths = []
  b_locks = []

  for i in range(options.buckets):
    handle, path = tempfile.mkstemp(prefix="bucket_freq-")
    b_f.append(handle)
    b_f_paths.append(path)

    handle, path = tempfile.mkstemp(prefix="bucket_list-")
    b_l.append(handle)
    b_l_paths.append(path)

    b_locks.append(mp.Lock())

  chunk_size = min(len(paths) / (options.job_count*2), 100)
  path_chunks = list(chunk(paths, chunk_size))
  # PASS 1: Tokenize documents into sets of terms
  with closing( mp.Pool(options.job_count, setup_pass1, 
                (options.max_order, b_f, b_l, b_locks))
              ) as pool:
    pass1_out = pool.imap_unordered(pass1, enumerate(path_chunks), chunksize=1)

  doc_count = defaultdict(int)
  total = len(paths)/chunk_size + (0 if len(paths)%chunk_size else 1)
  print "chunk size: %d (%d chunks)" % (chunk_size, total)

  wrotekeys = 0
  for i, keycount in enumerate(pass1_out):
    print "tokenized chunk (%d/%d)" % (i+1,total)
    wrotekeys += keycount

  print "wrote a total of %d keys" % wrotekeys 

  # rewind all the file descriptors
  for bucket in b_l + b_f:
    os.lseek(bucket, 0, os.SEEK_SET)


  # PASS 2: Compile document frequency counts
  doc_freq_fd, doc_count_path = tempfile.mkstemp(prefix="doccount-")
  with closing( mp.Pool(options.job_count, setup_pass2, 
                (options.max_order, doc_freq_fd, mp.Lock()))
              ) as pool:
    pass2_out = pool.imap_unordered(pass2, b_f, chunksize=1)

  readkeys = 0
  for i, keycount in enumerate(pass2_out):
    readkeys += keycount 
    print "processed bucket (%d/%d)" % (i+1, NUM_BUCKETS)

  print "read back a total of %d keys (%d short)" % ( readkeys, wrotekeys-readkeys)

  # close all file descriptors
  for fd in b_l + b_f:
    os.close(fd)

  # delete the b_f files
  for path in b_f_paths:
    os.remove(path)

  doc_count = {}
  with os.fdopen(doc_freq_fd) as f:
    f.seek(0)
    for key, value in unmarshal_iter(f):
      doc_count[key] = value

  print "unique features:", len(doc_count)
  os.remove(doc_count_path)

  # Work out the set of features to compute IG
  features = set()
  for i in range(1, options.max_order+1):
    d = dict( (k, doc_count[k]) for k in doc_count if len(k) == i)
    features |= set(sorted(d, key=d.get, reverse=True)[:options.df_tokens])
  features = sorted(features)
  print "candidate features: ", len(features)

  # Work out the path chunk start offsets
  chunk_offsets = [0]
  for c in path_chunks:
    chunk_offsets.append(chunk_offsets[-1] + len(c))

  return b_l_paths, features, chunk_offsets

if __name__ == "__main__":
  parser = optparse.OptionParser()
  parser.add_option("-o","--output", dest="outfile", help="output features to FILE", metavar="FILE")
  parser.add_option("-c","--corpus", dest="corpus", help="read corpus from DIR", metavar="DIR")
  parser.add_option("-j","--jobs", dest="job_count", type="int", help="number of processes to use", default=mp.cpu_count()+4)
  parser.add_option("-w","--weights",dest="weights", help="output weights to DIR (optional)", metavar="DIR")
  parser.add_option("-t","--temp",dest="temp", help="store temporary files in DIR", metavar="DIR", default=tempfile.gettempdir())
  parser.add_option("--max_order", dest="max_order", type="int", help="highest n-gram order to use", default=MAX_NGRAM_ORDER)
  parser.add_option("--feats_per_lang", dest="feats_per_lang", type="int", help="number of features to retain for each language", default=FEATURES_PER_LANG)
  parser.add_option("--df_tokens", dest="df_tokens", type="int", help="number of tokens to consider for each n-gram order", default=TOP_DOC_FREQ)
  parser.add_option("--buckets", dest="buckets", type="int", help="numer of buckets to use in k-v pair generation", default=NUM_BUCKETS)

  options, args = parser.parse_args()

  # check options
  if not options.corpus:
    parser.error("corpus(-c) must be specified")

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

  # set tempdir
  tempfile.tempdir = options.temp

  # display paths
  print "output path:", output_path
  print "temp path:", options.temp
  if options.corpus:
    print "corpus path:", options.corpus
  if options.weights:
    print "weights path:", options.weights

  # build a list of paths
  paths = []
  for dirpath, dirnames, filenames in os.walk(options.corpus, followlinks=True):
    for f in filenames:
      paths.append(os.path.join(dirpath, f))
  print "will tokenize %d files" % len(paths)

  # Tokenize
  cm_domain, cm_lang, lang_index = get_classmaps(paths)
  chunk_paths, features, chunk_offsets = build_inverted_index(paths, options)

  # Convert features from character tuples to strings
  #features = [ ''.join(f) for f in features ]

  # Compute LD from inverted index
  try:
    final_feature_set = select_LD_features(features, lang_index, chunk_paths, chunk_offsets, cm_domain, cm_lang, options)
  except OSError, e:
    print e
    import pdb;pdb.pm()
 
        
  # Output
  print "selected %d features" % len(final_feature_set)

  with open(output_path,'w') as f:
    for feat in final_feature_set:
      print >>f, repr(feat)
    print 'wrote features to "%s"' % output_path 
    

