import urllib
import urllib2
import json
import os
import time

data_file = '/lt/work/mlui/data/twitter/msgs21AUG2010'
base_url = 'http://localhost/langid/detect'
COUNT=50

def test_wiki10k():
  for p in os.listdir('wiki10k')[:COUNT]:
    path = os.path.join('wiki10k', p)
    with open(path) as f:
      opener = urllib2.build_opener(urllib2.HTTPHandler)
      request = urllib2.Request(base_url, data=f.read())
      request.add_header('Content-Type', 'text/plain')
      request.get_method = lambda: 'PUT'
      response = opener.open(request)
      response = json.loads(response.read())
      print response['responseData']['language'], response['responseData']['confidence'], p

def test_twitter():
  with open(data_file) as f:
    for msg in list(f)[:COUNT]:
      if isinstance(msg, unicode): msg = msg.encode('utf-8')
      query = {'q':msg}
      payload = urllib.urlencode(query)
      req = urllib2.Request(base_url+'?'+payload)
      response = urllib2.urlopen(req)
      response = json.loads(response.read())
      print response['responseData']['language'], response['responseData']['confidence'], msg,
      
if __name__ == "__main__":
  start = time.time()
  #test_wiki10k()
  test_twitter()
  duration = time.time() - start
  print "Rate: %f inst/s" % (COUNT / duration)

