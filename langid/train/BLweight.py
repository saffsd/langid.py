"""
Implementing the "blacklist" feature weighting metric proposed by
Tiedemann & Ljubesic.

Marco Lui, February 2013
"""

NUM_BUCKETS = 64 # number of buckets to use in k-v pair generation
CHUNKSIZE = 50 # maximum size of chunk (number of files tokenized - less = less memory use)

import os
import argparse
import numpy as np

from common import read_features, makedir, write_weights
from scanner import build_scanner
from index import CorpusIndexer
from NBtrain import generate_cm, learn_pc, learn_ptc


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("-o","--output", metavar="PATH", help = "write weights to PATH")
  parser.add_argument('-f','--features', metavar="FILE", help = 'only output features from FILE')
  parser.add_argument("-t", "--temp", metavar='TEMP_DIR', help="store buckets in TEMP_DIR instead of in MODEL_DIR/buckets")
  parser.add_argument("-j","--jobs", type=int, metavar='N', help="spawn N processes (set to 1 for no paralleization)")
  parser.add_argument("-m","--model", help="save output to MODEL_DIR", metavar="MODEL_DIR")
  parser.add_argument("--buckets", type=int, metavar='N', help="distribute features into N buckets", default=NUM_BUCKETS)
  parser.add_argument("--chunksize", type=int, help="max chunk size (number of files to tokenize at a time - smaller should reduce memory use)", default=CHUNKSIZE)
  parser.add_argument("lang1", metavar='LANG', help="first language")
  parser.add_argument("lang2", metavar='LANG', help="second language")
  parser.add_argument("corpus", help="read corpus from CORPUS_DIR", metavar="CORPUS_DIR")
  args = parser.parse_args()

  # Work out where our model directory is
  corpus_name = os.path.basename(args.corpus)
  if args.model:
    model_dir = args.model
  else:
    model_dir = os.path.join('.', corpus_name+'.model')

  def m_path(name):
    return os.path.join(model_dir, name)

  # Try to determine the set of features to consider
  if args.features:
    # Use a pre-determined feature list
    feat_path = args.features
  elif os.path.exists(m_path('DFfeats')):
    # Use LDfeats
    feat_path = m_path('DFfeats')
  else:
    raise ValueError("no suitable feature list")

  # Where to do output
  if args.output:
    weights_path = args.output
  else:
    weights_path = m_path('BLfeats.{0}.{1}'.format(args.lang1, args.lang2))

  # Where temp files go
  if args.temp:
    buckets_dir = args.temp
  else:
    buckets_dir = m_path('buckets')
  makedir(buckets_dir)

  # display paths
  print "languages: {0} {1}".format(args.lang1, args.lang2)
  print "model path:", model_dir
  print "feature path:", feat_path
  print "weights path:", weights_path
  print "temp (buckets) path:", buckets_dir

  feats = read_features(feat_path)

  indexer = CorpusIndexer(args.corpus, langs = [args.lang1, args.lang2])
  items = [ (d,l,p) for (d,l,n,p) in indexer.items ]
  if len(items) == 0:
    raise ValueError("found no files!")

  print "will process {0} features across {1} paths".format(len(feats), len(items))

  langs = [args.lang1, args.lang2]
  cm = generate_cm([ (l,p) for d,l,p in items], len(langs))
  paths = zip(*items)[2]


  nb_classes = langs
  nb_pc = learn_pc(cm)

  tk_nextmove, tk_output = build_scanner(feats)
  nb_ptc = learn_ptc(paths, tk_nextmove, tk_output, cm, buckets_dir, args)
  nb_ptc = np.array(nb_ptc).reshape(len(feats), len(nb_pc))

  # Normalize to 1 on the term axis
  for i in range(nb_ptc.shape[1]):
    nb_ptc[:,i] = (1/np.exp(nb_ptc[:,i][None,:] - nb_ptc[:,i][:,None]).sum(1))

  w = dict(zip(feats, np.abs((nb_ptc[:,0] - nb_ptc[:,1]) / nb_ptc.sum(1))))
  write_weights(w, weights_path)
  print "wrote weights to {0}".format(weights_path)
