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

:
Usage: langid.py [options]

Options:
  -h, --help            show this help message and exit
  -s, --serve           launch web service
  --host=HOST           host/ip to bind to
  --port=PORT           port to listen on
  -v                    increase verbosity (repeat for greater effect)
  -m MODEL              load model from file
  -l LANGS, --langs=LANGS
                        comma-separated set of target ISO639 language codes
                        (e.g en,de)
  -r, --remote          auto-detect IP address for remote access
  --demo                launch an in-browser demo application


The simplest way to use langid.py is as a command-line tool. Invoke using `python langid.py`.
This will cause a prompt to display. Enter text to identify, and hit enter::

  >>> This is a test
  ('en', 0.99999999099035441)
  >>> Questa e una prova
  ('it', 0.98569847366134222)

langid.py can also detect when the input is redirected (only tested under Linux), and in this
case will process until EOF rather than until newline like in interactive mode::

  python langid.py < readme.rst 
  ('en', 1.0)

The value returned is the probability estimate for the language. Full estimation is
not actually necessary for classification, and can be disabled in the source code
of langid.py for a slight performance boost. 

You can also use langid.py as a python library::

  # python
  Python 2.7.2+ (default, Oct  4 2011, 20:06:09) 
  [GCC 4.6.1] on linux2
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import langid
  >>> langid.classify("This is a test")
  ('en', 0.99999999099035441)
  
Finally, langid.py can use Python's built-in wsgiref.simple_server (or fapws3 if available) to
provide language identification as a web service. To do this, launch `python langid.py -s`, and
access localhost:9008/detect . The web service supports GET, POST and PUT. If GET is performed
with no data, a simple HTML forms interface is displayed.

The response is generated in JSON, here is an example::

  {"responseData": {"confidence": 0.99999999099035441, "language": "en"}, "responseDetails": null, "responseStatus": 200}

A utility such as curl can be used to access the web service::

  # curl -d "q=This is a test" localhost:9008/detect
  {"responseData": {"confidence": 0.99999999099035441, "language": "en"}, "responseDetails": null, "responseStatus": 200}

You can also use HTTP PUT::

  # curl -T readme.rst localhost:9008/detect
    % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  100  2871  100   119  100  2752    117   2723  0:00:01  0:00:01 --:--:--  2727
  {"responseData": {"confidence": 1.0, "language": "en"}, "responseDetails": null, "responseStatus": 200}

If no "q=XXX" key-value pair is present in the HTTP POST payload, langid.py will interpret the entire
file as a single query. This allows for redirection via curl::

  # echo "This is a test" | curl -d @- localhost:9008/detect
  {"responseData": {"confidence": 0.99999999099035441, "language": "en"}, "responseDetails": null, "responseStatus": 200}

langid.py will attempt to discover the host IP address automatically. Often, this is set to localhost(127.0.1.1), even 
though the machine has a different external IP address. langid.py can attempt to automatically discover the external
IP address. To enable this functionality, start langid.py with the "-r" flag.

langid.py supports constraining of the output language set using the "-l" flag and a comma-separated list of ISO639-1 
language codes::

  # python langid.py -l it,fr
  >>> Io non parlo italiano
  ('it', 0.99999999988965627)
  >>> Je ne parle pas français
  ('fr', 1.0)
  >>> I don't speak english
  ('it', 0.92210605672341062)

When using langid.py as a library, the set_languages method can be used to constrain the language set::

  python                      
  Python 2.7.2+ (default, Oct  4 2011, 20:06:09) 
  [GCC 4.6.1] on linux2
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import langid
  >>> langid.classify("I do not speak english")
  ('en', 0.57133487679900674)
  >>> langid.set_languages(['de','fr','it'])
  >>> langid.classify("I do not speak english")
  ('it', 0.99999835791478453)
  >>> langid.set_languages(['en','it'])
  >>> langid.classify("I do not speak english")
  ('en', 0.99176190378750373)

Training a model
----------------
Training a model for langid.py requires a large amount of computation for the feature selection stage.
We provide a parallelized model generator that can run on a modern desktop machine. It uses a sharding
technique similar to map-reduce to allow paralellization while running in constant memory.

The model training is broken into two steps:

1. LD Feature Selection (LDfeatureselect.py)
2. Naive Bayes learning (train.py)

The two steps are fully independent, and can potentially be run on different data sets. It is also possible 
to replace the feature selection with an alternative set of features. 

To train a model, we require multiple corpora of monolingual documents. Each document should be a single file,
and each file should be in a 2-deep folder hierarchy, with language nested within domain. For example, we
may have a number of English files:

  ./corpus/domain1/en/File1.txt
  ./corpus/domainX/en/001-file.xml

This is the hierarchy that both LDfeatureselect.py and train.py expect. The -c argment for both is the name
of the directory containing the domain-specific subdirectories, in this example './corpus'. The output file
is specified with the '-o' option.

To learn features, we would invoke::

    python LDfeatureselect.py -c corpus -o features

This would create a file called 'features' containing features in a one-per-line format that can be parsed 
by python's eval().

To then generate a model using the same corpus and the selected features, we would invoke::
    
    python train.py -c corpus -o model -i features

This will generate a compressed model in a file called 'model'. The path to this file can then be passed 
as a command-line argument to langid.py::

    python langid.py -m model

Read more
---------
langid.py is based on our published research. [1] describes the LD feature selection technique in detail,
and [2] provides more detail about the module langid.py itself.

[1] Lui, Marco and Timothy Baldwin (2011) Cross-domain Feature Selection for Language Identification, 
In Proceedings of the Fifth International Joint Conference on Natural Language Processing (IJCNLP 2011), 
Chiang Mai, Thailand, pp. 553—561. Available from http://www.aclweb.org/anthology/I11-1062

[2] Lui, Marco and Timothy Baldwin (to appear) langid.py: An Off-the-shelf Language Identification Tool, 
In Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (ACL 2012), 
Demo Session, Jeju, Republic of Korea.

Acknowledgements
----------------
Thanks to aitzol for help with packaging langid.py for PyPI.
