#!/usr/bin/env python
"""
NBtrain.py - 
Model generator for langid.py

Marco Lui, January 2013

Based on research by Marco Lui and Tim Baldwin.

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
MAX_CHUNK_SIZE = 100 # maximum number of files to tokenize at once
NUM_BUCKETS = 64 # number of buckets to use in k-v pair generation

import base64, bz2, cPickle
import os, sys, argparse, csv
import array
import numpy as np
import tempfile
import marshal
import atexit, shutil
import multiprocessing as mp
from collections import deque, defaultdict
from contextlib import closing

from common import chunk, unmarshal_iter, read_features, index, MapPool

def offsets(chunks):
  # Work out the path chunk start offsets
  chunk_offsets = [0]
  for c in chunks:
    chunk_offsets.append(chunk_offsets[-1] + len(c))
  return chunk_offsets

def state_trace(path):
  """
  Returns counts of how often each state was entered
  """
  global __nm_arr
  c = defaultdict(int)
  state = 0
  with open(path) as f:
    text = f.read()
    for letter in map(ord,text):
      state = __nm_arr[(state << 8) + letter]
      c[state] += 1
  return c

def setup_pass_tokenize(nm_arr, output_states, tk_output, b_dirs):
  """
  Set the global next-move array used by the aho-corasick scanner
  """
  global __nm_arr, __output_states, __tk_output, __b_dirs
  __nm_arr = nm_arr
  __output_states = output_states
  __tk_output = tk_output
  __b_dirs = b_dirs

def pass_tokenize(arg):
  """
  Tokenize documents and do counts for each feature
  Split this into buckets chunked over features rather than documents
  """
  global __output_states, __tk_output, __b_dirs
  chunk_offset, chunk_paths = arg
  term_freq = defaultdict(int)
  __procname = mp.current_process().name
  __buckets = [tempfile.mkstemp(prefix=__procname, suffix='.index', dir=p)[0] for p in __b_dirs]

  # Tokenize each document and add to a count of (doc_id, f_id) frequencies
  for doc_count, path in enumerate(chunk_paths):
    doc_id = doc_count + chunk_offset
    count = state_trace(path)
    for state in (set(count) & __output_states):
      for f_id in __tk_output[state]:
        term_freq[doc_id, f_id] += count[state]

  # Distribute the aggregated counts into buckets
  bucket_count = len(__buckets)
  for doc_id, f_id in term_freq:
    bucket_index = hash(f_id) % bucket_count
    count = term_freq[doc_id, f_id]
    item = ( f_id, doc_id, count )
    os.write(__buckets[bucket_index], marshal.dumps(item))

  for f in __buckets:
    os.close(f)

  return len(term_freq)

def setup_pass_ptc(cm, num_instances):
  global __cm, __num_instances
  __cm = cm
  __num_instances = num_instances

def pass_ptc(b_dir):
  """
  Take a bucket, form a feature map, compute the count of
  each feature in each class.
  @param b_dir path to the bucket directory
  @returns (read_count, f_ids, prod) 
  """
  global __cm, __num_instances

  terms = defaultdict(lambda : np.zeros((__num_instances,), dtype='int'))

  read_count = 0
  for path in os.listdir(b_dir):
    if path.endswith('.index'):
      for f_id, doc_id, count in unmarshal_iter(os.path.join(b_dir, path)):
        terms[f_id][doc_id] = count
        read_count += 1

  f_ids, f_vs = zip(*terms.items())
  fm = np.vstack(f_vs)
  prod = np.dot(fm, __cm)
  return read_count, f_ids, prod


def learn_pc(cm):
  """
  @param cm class map
  @returns nb_pc: log(P(C))
  """
  pc = np.log(cm.sum(0))
  nb_pc = array.array('d', pc)
  return nb_pc

def generate_cm(items, num_classes):
  """
  @param items (class id, path) pairs
  @param num_classes The number of classes present
  """
  num_instances = len(items)

  # Generate the class map
  cm = np.zeros((num_instances, num_classes), dtype='bool')
  for docid, (lang_id, path) in enumerate(items):
    cm[docid, lang_id] = True

  return cm

def learn_ptc(paths, tk_nextmove, tk_output, cm, temp_path, args):
  global b_dirs
  num_instances = len(paths)
  num_features = max( i for v in tk_output.values() for i in v) + 1

  # Generate the feature map
  nm_arr = mp.Array('i', tk_nextmove, lock=False)

  if args.jobs:
    chunksize = min(len(paths) / (args.jobs*2), args.chunksize)
  else:
    chunksize = min(len(paths) / (mp.cpu_count()*2), args.chunksize)

  # TODO: Set the output dir
  b_dirs = [ tempfile.mkdtemp(prefix="train-",suffix='-bucket', dir=temp_path) for i in range(args.buckets) ]

  output_states = set(tk_output)
  
  path_chunks = list(chunk(paths, chunksize))
  pass_tokenize_arg = zip(offsets(path_chunks), path_chunks)
  
  pass_tokenize_params = (nm_arr, output_states, tk_output, b_dirs) 
  with MapPool(args.jobs, setup_pass_tokenize, pass_tokenize_params) as f:
    pass_tokenize_out = f(pass_tokenize, pass_tokenize_arg)

  write_count = sum(pass_tokenize_out)
  print "wrote a total of %d keys" % write_count

  pass_ptc_params = (cm, num_instances)
  with MapPool(args.jobs, setup_pass_ptc, pass_ptc_params) as f:
    pass_ptc_out = f(pass_ptc, b_dirs)

  reads, ids, prods = zip(*pass_ptc_out)
  read_count = sum(reads)
  print "read a total of %d keys (%d short)" % (read_count, write_count - read_count)

  prod = np.zeros((num_features, cm.shape[1]), dtype=int)
  prod[np.concatenate(ids)] = np.vstack(prods)
  ptc = np.log(1 + prod) - np.log(num_features + prod.sum(0))

  nb_ptc = array.array('d')
  for term_dist in ptc.tolist():
    nb_ptc.extend(term_dist)
  return nb_ptc

@atexit.register
def cleanup():
  global b_dirs
  try:
    for d in b_dirs:
      shutil.rmtree(d)
  except NameError:
    # Failed before b_dirs is defined, nothing to clean
    pass

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-j","--jobs", type=int, metavar='N', help="spawn N processes (set to 1 for no paralleization)")
  parser.add_argument("-t", "--temp", metavar='TEMP_DIR', help="store buckets in TEMP_DIR instead of in MODEL_DIR/buckets")
  parser.add_argument("-s", "--scanner", metavar='SCANNER', help="use SCANNER for feature counting")
  parser.add_argument("-o", "--output", metavar='OUTPUT', help="output langid.py-compatible model to OUTPUT")
  #parser.add_argument("-i","--index",metavar='INDEX',help="read list of training document paths from INDEX")
  parser.add_argument("model", metavar='MODEL_DIR', help="read index and produce output in MODEL_DIR")
  parser.add_argument("--chunksize", type=int, help='maximum chunk size (number of files)', default=MAX_CHUNK_SIZE)
  parser.add_argument("--buckets", type=int, metavar='N', help="distribute features into N buckets", default=NUM_BUCKETS)
  args = parser.parse_args()

  if args.temp:
    temp_path = args.temp
  else:
    temp_path = os.path.join(args.model, 'buckets')

  if args.scanner:
    scanner_path = args.scanner
  else:
    scanner_path = os.path.join(args.model, 'LDfeats.scanner')

  if args.output:
    output_path = args.output
  else:
    output_path = os.path.join(args.model, 'model')

  index_path = os.path.join(args.model, 'paths')
  lang_path = os.path.join(args.model, 'lang_index')

  # display paths
  print "model path:", args.model
  print "temp path:", temp_path
  print "scanner path:", scanner_path
  #print "index path:", index_path
  print "output path:", output_path

  # read list of training files
  with open(index_path) as f:
    reader = csv.reader(f)
    items = [ (l,p) for _,l,p in reader ]

  # read scanner
  with open(scanner_path) as f:
    tk_nextmove, tk_output, _ = cPickle.load(f)

  # read list of languages in order
  with open(lang_path) as f:
    reader = csv.reader(f)
    langs = zip(*reader)[0]
    
  cm = generate_cm(items, len(langs))
  paths = zip(*items)[1]

  nb_classes = langs
  nb_pc = learn_pc(cm)
  nb_ptc = learn_ptc(paths, tk_nextmove, tk_output, cm, temp_path, args)

  # output the model
  model = nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output
  string = base64.b64encode(bz2.compress(cPickle.dumps(model)))
  with open(output_path, 'w') as f:
    f.write(string)
  print "wrote model to %s (%d bytes)" % (output_path, len(string))
