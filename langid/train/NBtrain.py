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
FEATS_PER_CHUNK = 100

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

def setup_pass1(nm_arr, output_states, state2feat, b_dirs, bucket_map):
  """
  Set the global next-move array used by the aho-corasick scanner
  """
  global __nm_arr, __output_states, __state2feat, __b_dirs, __bucket_map
  __nm_arr = nm_arr
  __output_states = output_states
  __state2feat = state2feat
  __b_dirs = b_dirs
  __bucket_map = bucket_map

def pass1(arg):
  """
  Tokenize documents and do counts for each feature
  Split this into buckets chunked over features rather than documents
  """
  global __output_states, __state2feat, __b_dirs, __bucket_map
  chunk_id, chunk_paths = arg
  term_freq = defaultdict(int)
  __procname = mp.current_process().name
  __buckets = [tempfile.mkstemp(prefix=__procname, suffix='.index', dir=p)[0] for p in __b_dirs]

  for doc_id, path in enumerate(chunk_paths):
    count = state_trace(path)
    for state in (set(count) & __output_states):
      for f_id in __state2feat[state]:
        term_freq[doc_id, f_id] += count[state]

  for doc_id, f_id in term_freq:
    bucket_index = __bucket_map[f_id]
    count = term_freq[doc_id, f_id]
    item = ( f_id, chunk_id, doc_id, count )
    os.write(__buckets[bucket_index], marshal.dumps(item))

  for f in __buckets:
    os.close(f)

  return len(term_freq)

def setup_pass2(cm, chunk_offsets, num_instances):
  global __cm, __chunk_offsets, __num_instances
  __cm = cm
  __chunk_offsets = chunk_offsets
  __num_instances = num_instances

def pass2(arg):
  """
  Take a bucket, form a feature map, learn the nb_ptc for it.
  """
  global __cm, __chunk_offsets, __num_instances
  num_feats, base_f_id, b_dir = arg
  fm = np.zeros((__num_instances, num_feats), dtype='int')

  read_count = 0
  for path in os.listdir(b_dir):
    if path.endswith('.index'):
      for f_id, chunk_id, doc_id, count in unmarshal_iter(os.path.join(b_dir, path)):
        doc_index = __chunk_offsets[chunk_id] + doc_id
        f_index = f_id - base_f_id
        fm[doc_index, f_index] = count
        read_count += 1

  prod = np.dot(fm.T, __cm)
  return read_count, prod


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

def learn_ptc(paths, nb_features, tk_nextmove, state2feat, cm, job_count=None):
  global b_dirs
  num_instances = len(paths)
  num_features = len(nb_features)

  # Generate the feature map
  nm_arr = mp.Array('i', tk_nextmove, lock=False)

  # TODO: Magic const
  if job_count:
    chunk_size = min(len(paths) / (job_count*2), 100)
  else:
    chunk_size = 100
  path_chunks = list(chunk(paths, chunk_size))
  feat_chunks = list(chunk(nb_features, FEATS_PER_CHUNK))

  feat_index = index(nb_features)

  bucket_map = {}
  b_dirs = []
  for chunk_id, feat_chunk in enumerate(feat_chunks):
    for feat in feat_chunk:
      bucket_map[feat_index[feat]] = chunk_id

    b_dirs.append(tempfile.mkdtemp(prefix="train-",suffix="-bucket"))


  output_states = set(state2feat)
  
  pass1_args = (nm_arr, output_states, state2feat, b_dirs, bucket_map)
  with MapPool(job_count, setup_pass1, pass1_args) as f:
    pass1_out = f(pass1, enumerate(path_chunks))

  write_count = sum(pass1_out)
  print "wrote a total of %d keys" % write_count

  f_chunk_sizes = map(len, feat_chunks)
  f_chunk_offsets = offsets(feat_chunks)
  pass2_args = (cm, offsets(path_chunks), num_instances)
  with MapPool(job_count, setup_pass2, pass2_args) as f:
    pass2_out = f(pass2, zip(f_chunk_sizes, f_chunk_offsets, b_dirs))

  reads, pass2_out = zip(*pass2_out)
  read_count = sum(reads)

  print "read a total of %d keys (%d short)" % (read_count, write_count - read_count)
  prod = np.vstack(pass2_out)
  ptc = np.log(1 + prod) - np.log(num_features + prod.sum(0))

  nb_ptc = array.array('d')
  for term_dist in ptc.tolist():
    nb_ptc.extend(term_dist)
  return nb_ptc

def read_corpus(path, wanted_langs = None):
  print "data directory: ", path

  if wanted_langs is not None:
    wanted_langs = set(wanted_langs)

  langs = set()
  paths = []
  for dirpath, dirnames, filenames in os.walk(path, followlinks=True):
    for f in filenames:
      lang = os.path.basename(dirpath)
      if wanted_langs is None or lang in wanted_langs:
        # None actually represents we want all langs
        paths.append(os.path.join(dirpath, f))
        langs.add(lang)

  print "found %d files" % len(paths)
  print "langs(%d): %s" % (len(langs), sorted(langs))
  return paths, langs

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
  args = parser.parse_args()

  if args.temp:
    temp_path = args.temp
  else:
    raise NotImplementedError

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
  features_path = os.path.join(args.model, 'LDfeats')

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
    tk_nextmove, tk_output, state2feat = cPickle.load(f)

  # read list of languages in order
  with open(lang_path) as f:
    reader = csv.reader(f)
    langs = zip(*reader)[0]
    
  cm = generate_cm(items, len(langs))
  paths = zip(*items)[1]

  nb_features = read_features(features_path)
  nb_classes = langs
  nb_pc = learn_pc(cm)
  nb_ptc = learn_ptc(paths, nb_features, tk_nextmove, state2feat, cm, args.jobs)

  # output the model
  model = nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output
  string = base64.b64encode(bz2.compress(cPickle.dumps(model)))
  with open(output_path, 'w') as f:
    f.write(string)
  print "wrote model to %s (%d bytes)" % (output_path, len(string))
