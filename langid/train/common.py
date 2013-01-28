"""
Common functions

Marco Lui, January 2013
"""

from itertools import islice
import marshal

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

import os, errno
def makedir(path):
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise
