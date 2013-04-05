#!/usr/bin/env python
"""
tokenize.py - 
Tokenizer for langid.py training system. This takes a list of files and tokenizes them
in parallel.

Marco Lui, January 2013

Copyright 2013 Marco Lui <saffsd@gmail.com>. All rights reserved.

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
NUM_BUCKETS = 64 # number of buckets to use in k-v pair generation
CHUNKSIZE = 50 # maximum size of chunk (number of files tokenized - less = less memory use)

import os, sys, argparse
import csv
import shutil
import tempfile
import marshal
import multiprocessing as mp
import random
import atexit

from itertools import tee 
from collections import defaultdict

from common import makedir, chunk, MapPool

class NGramTokenizer(object):
  def __init__(self, min_order=1, max_order=3):
    self.min_order = min_order
    self.max_order = max_order

  def __call__(self, seq):
    min_order = self.min_order
    max_order = self.max_order
    t = tee(seq, max_order)
    for i in xrange(max_order):
      for j in xrange(i):
        # advance iterators, ignoring result
        t[i].next()
    while True:
      token = ''.join(tn.next() for tn in t)
      if len(token) < max_order: break
      for n in xrange(min_order-1, max_order):
        yield token[:n+1]
    for a in xrange(max_order-1):
      for b in xrange(min_order, max_order-a):
        yield token[a:a+b]

@atexit.register
def cleanup():
  global b_dirs, complete
  try:
    if not complete:
      for d in b_dirs:
        shutil.rmtree(d)
  except NameError:
    # Failed before globals defined, nothing to clean
    pass

def setup_pass_tokenize(tokenizer, b_dirs, sample_count, sample_size):
  global __tokenizer, __b_dirs, __sample_count, __sample_size
  __tokenizer = tokenizer
  __b_dirs = b_dirs
  __sample_count = sample_count
  __sample_size = sample_size

def pass_tokenize(chunk_items):
  """
  Chunk files into a doc->term mapping,
  and simultaneously build a term->df count.
  The term->df counts are redistributed to
  buckets via python's in-built hash function.
  This is basically an inversion step, so that 
  now we are chunked on the term axis rather
  than the document axis.
  """
  global __maxorder, __b_dirs, __extractor, __sample_count, __sample_size
  __procname = mp.current_process().name
  b_freq_lang = [tempfile.mkstemp(prefix=__procname+'-', suffix='.lang', dir=p)[0] for p in __b_dirs]
  b_freq_domain = [tempfile.mkstemp(prefix=__procname+'-', suffix='.domain', dir=p)[0] for p in __b_dirs]
  
  extractor = __tokenizer
  term_lng_freq = defaultdict(lambda: defaultdict(int))
  term_dom_freq = defaultdict(lambda: defaultdict(int))

  for domain_id, lang_id, path in chunk_items:
    with open(path) as f:
      if __sample_count:
        # sampling tokenization
        text = f.read()
        poss = max(1,len(text) - __sample_size) # possibe start locations
        count = min(poss, __sample_count) # reduce number of samples if document is too short
        offsets = random.sample(xrange(poss), count)
        for offset in offsets:
          tokenset = set(extractor(text[offset: offset+__sample_size]))
          for token in tokenset:
            term_lng_freq[token][lang_id] += 1
            term_dom_freq[token][domain_id] += 1
          
      else:
        # whole-document tokenization
        tokenset = set(extractor(f.read()))
        for token in tokenset:
          term_lng_freq[token][lang_id] += 1
          term_dom_freq[token][domain_id] += 1

  for term in term_lng_freq:
    bucket_index = hash(term) % len(b_freq_lang)
    for lang, count in term_lng_freq[term].iteritems():
      os.write(b_freq_lang[bucket_index], marshal.dumps((term, lang, count)))
    for domain, count in term_dom_freq[term].iteritems():
      os.write(b_freq_domain[bucket_index], marshal.dumps((term, domain, count)))

  # Close all the open files
  for f in b_freq_lang + b_freq_domain:
    os.close(f)

  return len(term_lng_freq)

def build_index(items, tokenizer, outdir, buckets=NUM_BUCKETS, jobs=None, chunksize=CHUNKSIZE, sample_count=None, sample_size=None):
  """
  @param items a list of (domain, language, path) tuples
  """
  global b_dirs, complete

  # Our exitfunc uses this to know whether to delete the tokenized files
  complete = False 

  if jobs is None:
    jobs = mp.cpu_count() + 4

  b_dirs = [ tempfile.mkdtemp(prefix="tokenize-",suffix='-{0}'.format(tokenizer.__class__.__name__), dir=outdir) for i in range(buckets) ]

  # PASS 1: Tokenize documents into sets of terms
   
  # If there are few items, make the chunk size such that each job
  # will have 2 chunks
  chunk_size = max(1,min(len(items) / (jobs * 2), chunksize))
  item_chunks = list(chunk(items, chunk_size))
  pass_tokenize_globals = (tokenizer, b_dirs, sample_count, sample_size)

  with MapPool(jobs, setup_pass_tokenize, pass_tokenize_globals) as f:
    pass_tokenize_out = f(pass_tokenize, item_chunks)


    doc_count = defaultdict(int)
    chunk_count = len(item_chunks)
    print "chunk size: {0} ({1} chunks)".format(chunk_size, chunk_count)
    print "job count: {0}".format(jobs)

    if sample_count:
      print "sampling-based tokenization: size {0} count {1}".format(sample_size, sample_count)
    else:
      print "whole-document tokenization"

    for i, keycount in enumerate(pass_tokenize_out):
      print "tokenized chunk (%d/%d) [%d keys]" % (i+1,chunk_count, keycount)

  complete = True

  return b_dirs

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-j","--jobs", type=int, metavar='N', help="spawn N processes (set to 1 for no paralleization)")
  parser.add_argument("-s", "--scanner", metavar='SCANNER', help="use SCANNER for tokenizing")
  parser.add_argument("--buckets", type=int, metavar='N', help="distribute features into N buckets", default=NUM_BUCKETS)
  parser.add_argument("--max_order", type=int, help="highest n-gram order to use")
  parser.add_argument("--word", action='store_true', default=False, help="use 'word' tokenization (currently str.split)")
  parser.add_argument("--chunksize", type=int, help="max chunk size (number of files to tokenize at a time - smaller should reduce memory use)", default=CHUNKSIZE)
  parser.add_argument("-t", "--temp", metavar='TEMP_DIR', help="store buckets in TEMP_DIR instead of in MODEL_DIR/buckets")
  parser.add_argument("model", metavar='MODEL_DIR', help="read index and produce output in MODEL_DIR")

  group = parser.add_argument_group('sampling')
  group.add_argument("--sample_size", type=int, help="size of sample for sampling-based tokenization", default=140)
  group.add_argument("--sample_count", type=int, help="number of samples for sampling-based tokenization", default=None)
  
  args = parser.parse_args()
  

  if args.temp:
    buckets_dir = args.temp
  else:
    buckets_dir = os.path.join(args.model, 'buckets')
  makedir(buckets_dir)

  bucketlist_path = os.path.join(args.model, 'bucketlist')
  index_path = os.path.join(args.model, 'paths')

  # display paths
  print "index path:", index_path
  print "bucketlist path:", bucketlist_path
  print "buckets path:", buckets_dir

  with open(index_path) as f:
    reader = csv.reader(f)
    items = list(reader)

  if sum(map(bool,(args.scanner, args.max_order, args.word))) > 1:
    parser.error('can only specify one of --word, --scanner and --max_order')

  # Tokenize
  print "will tokenize %d files" % len(items)
  if args.scanner:
    from scanner import Scanner
    tokenizer = Scanner.from_file(args.scanner)
    print "using provided scanner: ", args.scanner
  elif args.word:
    tokenizer = str.split
    print "using str.split to tokenize"
  else:
    max_order = args.max_order if args.max_order else MAX_NGRAM_ORDER
    tokenizer = NGramTokenizer(1,max_order)
    print "using n-gram tokenizer: max_order({0})".format(max_order)
  b_dirs = build_index(items, tokenizer, buckets_dir, args.buckets, args.jobs, args.chunksize, args.sample_count, args.sample_size)

  # output the paths to the buckets
  with open(bucketlist_path,'w') as f:
    for d in b_dirs:
      f.write(d+'\n')

