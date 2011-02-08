import timeit

setup_instances =\
"""
from tokenizer import multi_batch_tokenize, single_batch_tokenize, multi_shared_tokenize, multi_ibatch_tokenize
teststr = "Il veneto deriva dalla fusione tra venetico parlato nella regione, il quale era del resto affine al latino stesso."
#teststr *= 100
instances = 10000 * [teststr]
instances = [ '%d %s' % p for p in enumerate(instances) ]
"""

RUNS = 5
if False:
  exec(setup_instances)
  x = multi_shared_tokenize(instances)
  #x = multi_batch_tokenize(instances)
  #import pdb;pdb.set_trace()
  print "DONE"


if __name__ == "__main__":
  print 'single-thread:',
  print timeit.timeit(
    'single_batch_tokenize(instances)',
    setup = setup_instances,
    number = RUNS,
  )

  print 'multprocessing map:',
  print timeit.timeit(
    'multi_batch_tokenize(instances)',
    setup = setup_instances,
    number = RUNS,
  )

  print 'multprocessing workers:',
  print timeit.timeit(
    'multi_shared_tokenize(instances)',
    setup = setup_instances,
    number = RUNS,
  )
