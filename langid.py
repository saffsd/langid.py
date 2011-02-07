# langid.py
# Marco Lui, Feb 2011
import itertools
import array

def tokenize(instance, tokens):
  def ngram(n, seq):
    tee = itertools.tee(seq, n)
    for i in xrange(n):
      for j in xrange(i):
        # advance iterators, ignoring result
        tee[i].next()
    while True:
      token = tuple(t.next() for t in tee)
      if len(token) < n: break
      yield token
  
  retval = array.array('L', itertools.repeat(0, len(tokens)))
  for token in ngram(2,instance):
    if token in tokens:
      retval[tokens[token]] += 1
  return retval
