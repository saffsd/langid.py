from multiprocessing import Pool, JoinableQueue, Process, Array
from Queue import Empty

from langid import _tokenize as tokenize

feature_space = [
  ' a', ' d', ' e', ' f', ' i', ' l', ' n', ' p', ' q', ' r', 
  ' s', ' t', ' v', ', ', 'Il', 'a ', 'af', 'al', 'ar', 'at', 
  #'co', 'da', 'de', 'e ', 'e,', 'eg', 'el', 'en', 'er', 'es',
  #'et', 'ff', 'fi', 'fu', 'gi', 'ic', 'il', 'in', 'io', 'iv',
  #'l ', 'la', 'le', 'll', 'ne', 'no', 'o ', 'o.', 'on', 'pa',
  #'qu', 'ra', 're', 'ri', 'rl', 'si', 'so', 'ss', 'st', 'te',
  #'ti', 'to', 'tr', 'ua', 'us', 'va', 've'
  ]
test_tokens = dict((tuple(k),v) for v,k in enumerate(feature_space))


import array
import itertools
import ctypes

# Tokenize
def f(instance):
  return tokenize(instance, test_tokens)

def single_batch_tokenize(instances):
  return map(f,instances)

def multi_batch_tokenize(instances):
  p = Pool(4)
  return p.map(f, instances)

def multi_ibatch_tokenize(instances):
  p = Pool(4)
  return p.imap_unordered(f, iter(instances), chunksize=625)

def multi_shared_tokenize(instances):
  num_feats = len(feature_space)

  def feed_queue(q):
    for i,inst in enumerate(instances):
      q.put_nowait((i, inst))

  def consume_queue(q,a):
    while True:
      i, inst = q.get()
      a[i][:] = f(inst)
      q.task_done()

  q = JoinableQueue()
  a = Array( ctypes.c_long * num_feats, len(instances))
  queue_feeder = Process(target=feed_queue, args=(q,))
  tokenizers = [ Process(target=consume_queue, args=(q,a)) for i in range(4) ]
  for t in tokenizers:
    t.daemon = True

  queue_feeder.start()
  [ t.start() for t in tokenizers ]

  queue_feeder.join()
  q.join()

  return a
  
  
