================
langid.py readme
================

Introduction
------------

langid.py is a standalone Language Identification (LangID) tool.

The design principles are as follows:

1. Fast
2. Pre-trained over a large number of languages (currently 97)
3. Not sensitive to domain-specific features (e.g. HTML/XML markup)
4. Single .py file with minimal dependencies
5. Deployable as a web service

All that is required to run langid.py is >= Python 2.5 and numpy. 

langid.py comes pre-trained on 97 languages (ISO 639-1 codes given):

|  af, am, an, ar, as, az, be, bg, bn, br, 
|  bs, ca, cs, cy, da, de, dz, el, en, eo, 
|  es, et, eu, fa, fi, fo, fr, ga, gl, gu, 
|  he, hi, hr, ht, hu, hy, id, is, it, ja, 
|  jv, ka, kk, km, kn, ko, ku, ky, la, lb, 
|  lo, lt, lv, mg, mk, ml, mn, mr, ms, mt, 
|  nb, ne, nl, nn, no, oc, or, pa, pl, ps, 
|  pt, qu, ro, ru, rw, se, si, sk, sl, sq, 
|  sr, sv, sw, ta, te, th, tl, tr, ug, uk, 
|  ur, vi, vo, wa, xh, zh, zu

The training data was drawn from 5 different sources:
- JRC-Acquis 
- ClueWeb 09
- Wikipedia
- Reuters RCV2
- Debian i18n

langid.py is WSGI-compliant. 

langid.py will use fapws3 as a web server if available, and default to
wsgiref.simple_server otherwise.

Usage
-----

::
  Usage: langid.py [options]

  Options:
    -h, --help            show this help message and exit
    -s, --serve           
    --host=HOST           host/ip to bind to
    --port=PORT           port to listen on
    -v                    increase verbosity (repeat for greater effect)
    -m MODEL              load model from file
    -l LANGS, --langs=LANGS
                          comma-separated set of target ISO639 language codes
                          (e.g en,de)


The simplest way to use langid.py is as a command-line tool. Invoke using `python langid.py`.
This will cause a prompt to display. Enter text to identify, and hit enter::

  >>> This is a test 
  ('en', -55.106250761034801)
  >>> Questa e una prova
  ('it', -35.417712211608887)

The value returned is a score for the language. It is not a probability esimate, as it is not
normalized by the document probability since this is unnecessary for classification.

You can also use langid.py as a python library::

  # python
  Python 2.7.2+ (default, Oct  4 2011, 20:06:09) 
  [GCC 4.6.1] on linux2
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import langid
  >>> langid.classify("This is a test")
  ('en', -55.106250761034801)
  
Finally, langid.py can use Python's built-in wsgiref.simple_server (or fapws3 if available) to
provide language identification as a web service. To do this, launch `python langid.py -s`, and
access localhost:9008/detect . The web service supports GET, POST and PUT. If GET is performed
with no data, a simple HTML forms interface is displayed.

The response is generated in JSON, here is an example::

  {"responseData": {"confidence": -55.106250761034801, "language": "en"}, "responseDetails": null, "responseStatus": 200}

A utility such as curl can be used to access the web service::

  # curl -d "q=This is a test" localhost:9008/detect
  {"responseData": {"confidence": -55.106250761034801, "language": "en"}, "responseDetails": null, "responseStatus": 200}

You can also use HTTP PUT::

  # curl -T readme.rst localhost:9008/detect
    % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  100  2871  100   119  100  2752    117   2723  0:00:01  0:00:01 --:--:--  2727
  {"responseData": {"confidence": -3728.4490563860536, "language": "en"}, "responseDetails": null, "responseStatus": 200}

langid.py supports constraining of the output language set using the "-l" flag and a comma-separated list of ISO639-1 
language codes::

  # python langid.py -l it,fr
  >>> Io non parlo italiano
  ('it', -38.538481712341309)
  >>> Je ne parle pas français
  ('fr', -116.95343780517578)
  >>> I don't speak english
  ('it', -8.8632845878601074)

When using langid.py as a library, the set_languages method can be used to constrain the language set::

  python                      
  Python 2.7.2+ (default, Oct  4 2011, 20:06:09) 
  [GCC 4.6.1] on linux2
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import langid
  >>> langid.classify("I do not speak english")
  ('en', -48.104645729064941)
  >>> langid.set_languages(['de','fr','it'])
  >>> langid.classify("I do not speak english")
  ('it', -52.895359516143799)
  >>> langid.set_languages(['en','it'])
  >>> langid.classify("I do not speak english")
  ('en', -48.104645729064941)

