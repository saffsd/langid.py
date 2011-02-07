from multiprocessing import Pool

feature_space = [
  ' a', ' d', ' e', ' f', ' i', ' l', ' n', ' p', ' q', ' r', 
  ' s', ' t', ' v', ', ', 'Il', 'a ', 'af', 'al', 'ar', 'at', 
  'co', 'da', 'de', 'e ', 'e,', 'eg', 'el', 'en', 'er', 'es',
  'et', 'ff', 'fi', 'fu', 'gi', 'ic', 'il', 'in', 'io', 'iv',
  'l ', 'la', 'le', 'll', 'ne', 'no', 'o ', 'o.', 'on', 'pa',
  'qu', 'ra', 're', 'ri', 'rl', 'si', 'so', 'ss', 'st', 'te',
  'ti', 'to', 'tr', 'ua', 'us', 'va', 've'
  ]
test_tokens = dict((tuple(k),v) for v,k in enumerate(feature_space))


# Tokenize
def f(instance):
  return tokenize(instance, test_tokens)

p = Pool(4)
def multi_batch_tokenize(instances):
  return p.map(f, instances)

def single_batch_tokenize(instances):
  return map(f,instances)
  
