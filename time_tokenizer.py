import timeit

setup_instances =\
"""
from tokenizer import multi_batch_tokenize, single_batch_tokenize
teststr = "Il veneto deriva dalla fusione tra venetico parlato nella regione, il quale era del resto affine al latino stesso."
#teststr *= 100
instances = 10000 * [teststr]
"""

RUNS = 5
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
