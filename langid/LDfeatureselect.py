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
TRAIN_PROP = 1.0 # proportion of training data available to use
MIN_DOMAIN = 1 # minimum number of domains a language must be present in to be included
CHUNKSIZE = 50 # maximum size of chunk (number of files tokenized - less = less memory use)

import os, sys, optparse
import collections
import csv
import shutil
import tempfile
import marshal
import random
import numpy
import cPickle
import multiprocessing as mp
import atexit
from itertools import tee, imap, islice
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
    chunk = tuple(islice(seq_iter, chunksize))
    if not chunk: break
    yield chunk

def unmarshal_iter(path):
  """
  Open a given path and yield an iterator over items unmarshalled from it.
  """
  with open(path, 'rb') as f:
    while True:
      try:
        yield marshal.load(f)
      except EOFError:
        break

@atexit.register
def cleanup():
  global b_dirs
  try:
    for d in b_dirs:
      shutil.rmtree(d)
  except NameError:
    # Failed before b_dirs is defined, nothing to clean
    pass

def setup_pass1(maxorder, b_dirs):
  global __maxorder, __b_dirs
  __maxorder = maxorder 
  __b_dirs = b_dirs


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
  global __maxorder, __b_dirs
  __procname = mp.current_process().name
  __b_freq = [tempfile.mkstemp(prefix=__procname, suffix='.freq', dir=p)[0] for p in __b_dirs]
  __b_list = [tempfile.mkstemp(prefix=__procname, suffix='.list', dir=p)[0] for p in __b_dirs]
  chunk_id, chunk_items = arg
  
  extractor = Tokenizer(__maxorder)
  term_doc_freq = defaultdict(int)
  term_lng_freq = defaultdict(lambda: defaultdict(int))
  term_dom_freq = defaultdict(lambda: defaultdict(int))

  for doc_index, (domain_id, lang_id, path) in enumerate(chunk_items):
    with open(path) as f:
      # TODO: integrate this into the tokenizer. It is probably safe to assume
      #       we will only ever apply this tokenizer to strings.
      tokenset = set(map(''.join,extractor(f.read())))
      for token in tokenset:
        term_doc_freq[token] += 1
        term_lng_freq[token][lang_id] += 1
        term_dom_freq[token][domain_id] += 1

  for key in term_doc_freq:
    bucket_index = hash(key) % len(__b_freq)
    os.write(__b_freq[bucket_index], marshal.dumps((key, term_doc_freq[key])))
    os.write(__b_list[bucket_index], marshal.dumps((key, chunk_id, (term_lng_freq[key].items(),term_dom_freq[key].items()))))

  # Close all the open files
  for f in __b_freq + __b_list:
    os.close(f)

  return len(term_doc_freq)

def pass2(bucket):
  """
  Take a term->df count bucket and sum it up
  """
  doc_count = defaultdict(int)
  count = 0
  with open(os.path.join(bucket, "docfreq"),'wb') as docfreq:
    for path in os.listdir(bucket):
      if path.endswith('.freq'):
        for key, value in unmarshal_iter(os.path.join(bucket,path)):
          doc_count[key] += value
          count += 1
    
    for item in doc_count.iteritems():
      docfreq.write(marshal.dumps(item))
  return count

def setup_pass3(features, dist_lang, dist_domain):
  """
  @param features a list of the features selected in pass2
  @param dist_lang a vector of the background distribution over languages
  @param dist_domain a vector of the background distribution over domains
  """
  global __features, __dist_lang, __dist_domain
  __features = features
  __dist_lang = dist_lang
  __dist_domain = dist_domain

def pass3(bucket):
  """
  In this pass we compute the information gain for each feature, binarized 
  with respect to each language as well as unified over the set of all 
  classes. 

  @global __features the list of features selected by pass2
  @global __dist_lang the global language distribution
  @param bucket the bucketdir to process
  """
  global __features, __dist_lang, __dist_domain
   
  term_lng_freq = defaultdict(lambda: defaultdict(int))
  term_dom_freq = defaultdict(lambda: defaultdict(int))
  term_index = defaultdict(Enumerator())

  for path in os.listdir(bucket):
    if path.endswith('.list'):
      for key, chunk_id, (lng_freq, dom_freq) in unmarshal_iter(os.path.join(bucket,path)):
        # Select only our listed features
        if key in __features:
          term_index[key]
          for lang, count in lng_freq:
            term_lng_freq[key][lang] += count
          for dom, count in dom_freq:
            term_dom_freq[key][dom] += count

  num_term = len(term_index)
  num_domain = len(__dist_domain)
  num_lang = len(__dist_lang)

  domain_cm_pos = numpy.zeros((num_term, num_domain), dtype='int')
  lang_cm_pos = numpy.zeros((num_term, num_lang), dtype='int')

  for term,term_id in term_index.iteritems():
    # update domain matrix
    dom_freq = term_dom_freq[term]
    for domain_id, count in dom_freq.iteritems():
      domain_cm_pos[term_id, domain_id] = count

    # update language matrix
    lng_freq = term_lng_freq[term]
    for lang_id, count in lng_freq.iteritems():
      lang_cm_pos[term_id, lang_id] = count

  domain_cm_neg = __dist_domain - domain_cm_pos
  lang_cm_neg = __dist_lang - lang_cm_pos

  domain_cm = numpy.dstack((domain_cm_neg, domain_cm_pos))
  lang_cm = numpy.dstack((lang_cm_neg, lang_cm_pos))

  x = domain_cm.sum(axis=1)
  term_w = x / x.sum(axis=1)[:, None].astype(float)

  # Entropy of the term-present/term-absent events
  domain_e = entropy(domain_cm, axis=1)

  # Information Gain with respect to the set of domains
  ig_domain = entropy(__dist_domain) - (term_w * domain_e).sum(axis=1)

  # Compute IG binarized with respect to each language
  ig_lang = list()
  for lang_id in xrange(num_lang):
    num_doc = __dist_lang.sum()
    prior = numpy.array((num_doc - __dist_lang[lang_id], __dist_lang[lang_id]), dtype=float) / num_doc

    lang_cm_bin = numpy.zeros((num_term, 2, 2), dtype=int) # (term, p(term), p(lang|term))
    lang_cm_bin[:,0,:] = lang_cm.sum(axis=1) - lang_cm[:,lang_id,:]
    lang_cm_bin[:,1,:] = lang_cm[:,lang_id,:]

    lang_e = entropy(lang_cm_bin, axis=1)
    x = lang_cm_bin.sum(axis=1)
    term_w = x / x.sum(axis=1)[:, None].astype(float)

    ig_lang.append( entropy(prior) - (term_w * lang_e).sum(axis=1) )
  # This is the actual LD weight being computed here.
  w_lang = numpy.vstack(ig_lang) - ig_domain

  terms = sorted(term_index, key=term_index.get)
  return terms, w_lang, ig_domain

def write_weights(path, weights):
  w = dict(weights)
  with open(path, 'w') as f:
    writer = csv.writer(f, delimiter = '\t')
    for k in sorted(w, key=w.get, reverse=True):
      writer.writerow((repr(k), w[k]))

class CorpusIndexer(object):
  """
  Class to index the contents of a corpus
  """
  def __init__(self, root, min_domain=MIN_DOMAIN, proportion=TRAIN_PROP):
    self.root = root
    self.min_domain = min_domain
    self.proportion = proportion 

    self.lang_index = defaultdict(Enumerator())
    self.domain_index = defaultdict(Enumerator())
    self.coverage_index = defaultdict(set)
    self.items = list()

    self.index(root)
    self.prune_min_domain(self.min_domain)

  def index(self, root):
    # build a list of paths
    paths = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
      for docname in filenames:
        if random.random() < self.proportion:
          # Each file has 'proportion' chance of being selected.
          path = os.path.join(dirpath, docname)

          # split the dirpath into identifying components
          d, lang = os.path.split(dirpath)
          d, domain = os.path.split(d)

          # index the language and the domain
          lang_id = self.lang_index[lang]
          domain_id = self.domain_index[domain]

          # add the domain-lang relation to the coverage index
          self.coverage_index[domain].add(lang)

          # add the item to our list
          self.items.append((domain_id,lang_id,docname,path))

  def prune_min_domain(self, min_domain):
    # prune files for all languages that do not occur in at least min_domain 
     
    # Work out which languages to reject as they are not present in at least 
    # the required number of domains
    lang_domain_count = defaultdict(int)
    for langs in self.coverage_index.values():
      for lang in langs:
        lang_domain_count[lang] += 1
    reject_langs = set( l for l in lang_domain_count if lang_domain_count[l] < min_domain)

    # Remove the languages from the indexer
    if reject_langs:
      #print "reject (<{0} domains): {1}".format(min_domain, sorted(reject_langs))
      reject_ids = set(self.lang_index[l] for l in reject_langs)
    
      new_lang_index = defaultdict(Enumerator())
      lm = dict()
      for k,v in self.lang_index.items():
        if v not in reject_ids:
          new_id = new_lang_index[k]
          lm[v] = new_id

      # Eliminate all entries for the languages
      self.items = [ (d, lm[l], n, p) for (d, l, n, p) in self.items if l in lm]

      self.lang_index = new_lang_index


  @property
  def dist_lang(self):
    """
    @returns A vector over frequency counts for each language
    """
    retval = numpy.zeros((len(self.lang_index),), dtype='int')
    for d, l, n, p in self.items:
      retval[l] += 1
    return retval

  @property
  def dist_domain(self):
    """
    @returns A vector over frequency counts for each domain 
    """
    retval = numpy.zeros((len(self.domain_index),), dtype='int')
    for d, l, n, p in self.items:
      retval[d] += 1
    return retval

  # TODO: Remove this as it should no longer be needed
  @property 
  def classmaps(self):
    num_instances = len(self.items)
    if num_instances == 0:
      raise ValueError("no items indexed!")
    cm_domain = numpy.zeros((num_instances, len(self.domain_index)), dtype='bool')
    cm_lang = numpy.zeros((num_instances, len(self.lang_index)), dtype='bool')

    # Populate the class maps
    for docid, (domain_id, lang_id, docname, path) in enumerate(self.items):
      cm_domain[docid, domain_id] = True
      cm_lang[docid, lang_id] = True
    return cm_domain, cm_lang

  @property
  def paths(self):
    return [ p for (d,l,n,p) in self.items ]



    
    

def select_LD_features(features, lang_index, b_dirs, dist_lang, dist_domain, options):
  print "computing information gain"
  # Instead of receiving a single feature map, we now receive a list of paths,
  # each corresponding to a chunk containing a portion of the final feature set
  # for each of these chunks we need to compute the IG of each feature WRT to
  # domain as well as WRT to each language.
  # The parallelism should come at the feature chunk level,
  # so we can collapse IG into a non-parallelized function.

  pass3_args = (features, dist_lang, dist_domain)
  #setup_pass3(*pass3_args)
  #pass3_out = map(pass3, b_dirs)

  with closing( mp.Pool(options.job_count, setup_pass3, pass3_args) ) as pool:
    pass3_out = pool.imap_unordered(pass3, b_dirs, chunksize=1)

  num_chunk = len(b_dirs)
  w_lang = []
  w_domain = []
  terms = []
  for i, (t, w_l, w_d) in enumerate(pass3_out):
    w_lang.append(w_l)
    w_domain.append(w_d)
    terms.extend(t)
    print "processed chunk (%d/%d) [%d terms]" % (i+1, num_chunk, len(t))
  pool.join()

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
    
def build_index(items, options):
  """
  @param items a list of (language, path) pairs
  """
  global b_dirs
  b_dirs = [ tempfile.mkdtemp(prefix="LDfeatureselect-",suffix='-bucket') for i in range(options.buckets) ]

  chunk_size = max(1,min(len(items) / (options.job_count*2), options.chunksize))
  item_chunks = list(chunk(items, chunk_size))
  pass1_arg = enumerate(item_chunks)
  # PASS 1: Tokenize documents into sets of terms
  with closing( mp.Pool(options.job_count, setup_pass1, 
                (options.max_order, b_dirs))
              ) as pool:
    pass1_out = pool.imap_unordered(pass1, pass1_arg, chunksize=1)

  doc_count = defaultdict(int)
  #print "CHUNK SIZE", chunk_size
  chunk_count = len(item_chunks)
  print "chunk size: %d (%d chunks)" % (chunk_size, chunk_count)

  wrotekeys = 0
  for i, keycount in enumerate(pass1_out):
    print "tokenized chunk (%d/%d) [%d keys]" % (i+1,chunk_count, keycount)
    wrotekeys += keycount
  pool.join()

  print "wrote a total of %d keys" % wrotekeys 

  # PASS 2: Compile document frequency counts
  with closing( mp.Pool(options.job_count) ) as pool:
    pass2_out = pool.imap_unordered(pass2, b_dirs, chunksize=1)

  readkeys = 0
  for i, keycount in enumerate(pass2_out):
    readkeys += keycount 
    print "processed bucket (%d/%d) [%d keys]" % (i+1, options.buckets, keycount)
  pool.join()

  print "read back a total of %d keys (%d short)" % ( readkeys, wrotekeys-readkeys)

  # build the global term->df mapping
  doc_count = {}
  for bucket in b_dirs:
    for key, value in unmarshal_iter(os.path.join(bucket, 'docfreq')):
      doc_count[key] = value

  print "unique features:", len(doc_count)

  # Work out the set of features to compute IG
  features = set()
  for i in range(1, options.max_order+1):
    d = dict( (k, doc_count[k]) for k in doc_count if len(k) == i)
    features |= set(sorted(d, key=d.get, reverse=True)[:options.df_tokens])
  features = sorted(features)
  print "candidate features: ", len(features)

  return b_dirs, features

if __name__ == "__main__":
  parser = optparse.OptionParser()
  parser.add_option("-o","--output", dest="outfile", help="output features to FILE", metavar="FILE")
  parser.add_option("-c","--corpus", dest="corpus", help="read corpus from DIR", metavar="DIR")
  parser.add_option("-j","--jobs", dest="job_count", type="int", help="number of processes to use", default=mp.cpu_count()+4)
  parser.add_option("-w","--weights",dest="weights", help="output weights to DIR (optional)", metavar="DIR")
  parser.add_option("-t","--temp",dest="temp", help="store temporary files in DIR", metavar="DIR", default=tempfile.gettempdir())
  parser.add_option("-p","--proportion", help="proportion of training data to use", type="float", default=TRAIN_PROP)
  parser.add_option("--max_order", dest="max_order", type="int", help="highest n-gram order to use", default=MAX_NGRAM_ORDER)
  parser.add_option("--feats_per_lang", dest="feats_per_lang", type="int", help="number of features to retain for each language", default=FEATURES_PER_LANG)
  parser.add_option("--df_tokens", dest="df_tokens", type="int", help="number of tokens to consider for each n-gram order", default=TOP_DOC_FREQ)
  parser.add_option("--buckets", dest="buckets", type="int", help="number of buckets to use in k-v pair generation", default=NUM_BUCKETS)
  parser.add_option("--min_domain", dest="min_domain", type="int", help="minimum number of domains a language must be present in", default=MIN_DOMAIN)
  parser.add_option("--chunksize", dest="chunksize", type="int", help="max chunk size (number of files to tokenize at a time - smaller should reduce memory use)", default=CHUNKSIZE)
  parser.add_option("-l","--langs",dest="langs", help="output list of languages to DIR", metavar="DIR")

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
  else:
    output_path = os.path.basename(options.corpus)+'.LDfeatures'

  # work out where to write langs to
  if options.langs:
    langs_path = options.langs
  else:
    langs_path = os.path.basename(options.corpus)+'.langs'

  # set tempdir
  if not os.path.exists(options.temp) and os.path.isdir(options.temp):
    parser.error("tempdir %s does not exist" % options.temp)
  tempfile.tempdir = options.temp

  # display paths
  print "output path:", output_path
  print "temp path:", options.temp
  if options.corpus:
    print "corpus path:", options.corpus
  if options.weights:
    print "weights path:", options.weights

  indexer = CorpusIndexer(options.corpus, options.min_domain, options.proportion)

  # Compute mappings between files, languages and domains
  dist_lang = indexer.dist_lang
  dist_domain = indexer.dist_domain
  lang_index = indexer.lang_index
  lang_info = ' '.join(("{0}({1})".format(k, dist_lang[v]) for k,v in lang_index.items()))
  print "langs({0}): {1}".format(len(dist_lang), lang_info)
  print "domains({0}): {1}".format(len(dist_domain), sorted(indexer.domain_index.keys()))

  # output the list of languages selected - this is particularly important when using
  # min_domains
  with open(langs_path,'w') as f:
    for lang in sorted(lang_index.keys()):
      f.write(lang+'\n')

  # Tokenize
  items = [ (d,l,p) for (d,l,n,p) in indexer.items ]
  print "will tokenize %d files" % len(items)
  b_dirs, features = build_index(items, options)


  # Compute LD from inverted index
  try:
    final_feature_set = select_LD_features(features, lang_index, b_dirs, dist_lang, dist_domain, options)
  except OSError, e:
    print e
    import pdb;pdb.pm()
 
  # Output
  print "selected %d features" % len(final_feature_set)

  with open(output_path,'w') as f:
    for feat in final_feature_set:
      print >>f, repr(feat)
    print 'wrote features to "%s"' % output_path 
    

