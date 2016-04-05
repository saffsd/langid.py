================
``langid.py`` readme
================

Introduction
------------

``langid.py`` is a standalone Language Identification (LangID) tool.

The design principles are as follows:

1. Fast
2. Pre-trained over a large number of languages (currently 97)
3. Not sensitive to domain-specific features (e.g. HTML/XML markup)
4. Single .py file with minimal dependencies
5. Deployable as a web service

All that is required to run ``langid.py`` is >= Python 2.7 and numpy.  
The main script ``langid/langid.py`` is cross-compatible with both Python2 and
Python3, but the accompanying training tools are still Python2-only.

``langid.py`` is WSGI-compliant.  ``langid.py`` will use ``fapws3`` as a web server if 
available, and default to ``wsgiref.simple_server`` otherwise.

``langid.py`` comes pre-trained on 97 languages (ISO 639-1 codes given):

    af, am, an, ar, as, az, be, bg, bn, br, 
    bs, ca, cs, cy, da, de, dz, el, en, eo, 
    es, et, eu, fa, fi, fo, fr, ga, gl, gu, 
    he, hi, hr, ht, hu, hy, id, is, it, ja, 
    jv, ka, kk, km, kn, ko, ku, ky, la, lb, 
    lo, lt, lv, mg, mk, ml, mn, mr, ms, mt, 
    nb, ne, nl, nn, no, oc, or, pa, pl, ps, 
    pt, qu, ro, ru, rw, se, si, sk, sl, sq, 
    sr, sv, sw, ta, te, th, tl, tr, ug, uk, 
    ur, vi, vo, wa, xh, zh, zu

The training data was drawn from 5 different sources:

* JRC-Acquis 
* ClueWeb 09
* Wikipedia
* Reuters RCV2
* Debian i18n


Usage
-----

    langid.py [options]

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
  -b, --batch           specify a list of files on the command line
  --demo                launch an in-browser demo application
  -d, --dist            show full distribution over languages
  -u URL, --url=URL     langid of URL
  --line                process pipes line-by-line rather than as a document
  -n, --normalize       normalize confidence scores to probability values


The simplest way to use ``langid.py`` is as a command-line tool, and you can 
invoke using ``python langid.py``. If you installed ``langid.py`` as a Python 
module (e.g. via ``pip install langid``), you can invoke ``langid`` instead of 
``python langid.py -n`` (the two are equivalent).  This will cause a prompt to 
display. Enter text to identify, and hit enter::

  >>> This is a test
  ('en', -54.41310358047485)
  >>> Questa e una prova
  ('it', -35.41771221160889)


``langid.py`` can also detect when the input is redirected (only tested under Linux), and in this
case will process until EOF rather than until newline like in interactive mode::

  python langid.py < README.rst 
  ('en', -22552.496054649353)


The value returned is the unnormalized probability estimate for the language. Calculating 
the exact probability estimate is disabled by default, but can be enabled through a flag::

  python langid.py -n < README.rst 
  ('en', 1.0)

More details are provided in this README in the section on `Probability Normalization`.

You can also use ``langid.py`` as a Python library::

  # python
  Python 2.7.2+ (default, Oct  4 2011, 20:06:09) 
  [GCC 4.6.1] on linux2
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import langid
  >>> langid.classify("This is a test")
  ('en', -54.41310358047485)
  
Finally, ``langid.py`` can use Python's built-in ``wsgiref.simple_server`` (or ``fapws3`` if available) to
provide language identification as a web service. To do this, launch ``python langid.py -s``, and
access http://localhost:9008/detect . The web service supports GET, POST and PUT. If GET is performed
with no data, a simple HTML forms interface is displayed.

The response is generated in JSON, here is an example::

  {"responseData": {"confidence": -54.41310358047485, "language": "en"}, "responseDetails": null, "responseStatus": 200}

A utility such as curl can be used to access the web service::

  # curl -d "q=This is a test" localhost:9008/detect
  {"responseData": {"confidence": -54.41310358047485, "language": "en"}, "responseDetails": null, "responseStatus": 200}

You can also use HTTP PUT::

  # curl -T readme.rst localhost:9008/detect
    % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  100  2871  100   119  100  2752    117   2723  0:00:01  0:00:01 --:--:--  2727
  {"responseData": {"confidence": -22552.496054649353, "language": "en"}, "responseDetails": null, "responseStatus": 200}

If no "q=XXX" key-value pair is present in the HTTP POST payload, ``langid.py`` will interpret the entire
file as a single query. This allows for redirection via curl::

  # echo "This is a test" | curl -d @- localhost:9008/detect
  {"responseData": {"confidence": -54.41310358047485, "language": "en"}, "responseDetails": null, "responseStatus": 200}

``langid.py`` will attempt to discover the host IP address automatically. Often, this is set to localhost(127.0.1.1), even 
though the machine has a different external IP address. ``langid.py`` can attempt to automatically discover the external
IP address. To enable this functionality, start ``langid.py`` with the ``-r`` flag.

``langid.py`` supports constraining of the output language set using the ``-l`` flag and a comma-separated list of ISO639-1 
language codes (the ``-n`` flag enables probability normalization)::

  # python langid.py -n -l it,fr
  >>> Io non parlo italiano
  ('it', 0.99999999988965627)
  >>> Je ne parle pas français
  ('fr', 1.0)
  >>> I don't speak english
  ('it', 0.92210605672341062)

When using ``langid.py`` as a library, the set_languages method can be used to constrain the language set::

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

Batch Mode
----------

``langid.py`` supports batch mode processing, which can be invoked with the ``-b`` flag.
In this mode, ``langid.py`` reads a list of paths to files to classify as arguments.
If no arguments are supplied, ``langid.py`` reads the list of paths from ``stdin``,
this is useful for using ``langid.py`` with UNIX utilities such as ``find``.

In batch mode, ``langid.py`` uses ``multiprocessing`` to invoke multiple instances of
the classifier, utilizing all available CPUs to classify documents in parallel. 

.. Probability Normalization

Probability Normalization
-------------------------

The probabilistic model implemented by ``langid.py`` involves the multiplication of a
large number of probabilities. For computational reasons, the actual calculations are
implemented in the log-probability space (a common numerical technique for dealing with
vanishingly small probabilities). One side-effect of this is that it is not necessary to
compute a full probability in order to determine the most probable language in a set
of candidate languages. However, users sometimes find it helpful to have a "confidence"
score for the probability prediction. Thus, ``langid.py`` implements a re-normalization
that produces an output in the 0-1 range.

``langid.py`` disables probability normalization by default. For
command-line usages of ``langid.py``, it can be enabled by passing the ``-n`` flag. For
probability normalization in library use, the user must instantiate their own 
``LanguageIdentifier``. An example of such usage is as follows::
  
  >> from langid.langid import LanguageIdentifier, model
  >> identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
  >> identifier.classify("This is a test")
  ('en', 0.9999999909903544)


Training a model
----------------
We provide a full set of training tools to train a model for ``langid.py`` 
on user-supplied data.  The system is parallelized to fully utilize modern 
multiprocessor machines, using a sharding technique similar to MapReduce to 
allow parallelization while running in constant memory.

The full training can be performed using the tool ``train.py``. For 
research purposes, the process has been broken down into indiviual steps, 
and command-line drivers for each step are provided. This allows the user 
to inspect the intermediates produced, and also allows for some parameter 
tuning without repeating some of the more expensive steps in the 
computation. By far the most expensive step is the computation of 
information gain, which will make up more than 90% of the total computation 
time.

The tools are:

1. index.py  - index a corpus. Produce a list of file, corpus, language pairs.
2. tokenize.py - take an index and tokenize the corresponding files
3. DFfeatureselect.py - choose features by document frequency
4. IGweight.py - compute the IG weights for language and for domain
5. LDfeatureselect.py - take the IG weights and use them to select a feature set
6. scanner.py - build a scanner on the basis of a feature set
7. NBtrain.py - learn NB parameters using an indexed corpus and a scanner

The tools can be found in ``langid/train`` subfolder. 

Each tool can be called with ``--help`` as the only parameter to provide an overview of the 
functionality. 

To train a model, we require multiple corpora of monolingual documents. Each document should 
be a single file, and each file should be in a 2-deep folder hierarchy, with language nested 
within domain. For example, we may have a number of English files:

    ./corpus/domain1/en/File1.txt
    ./corpus/domainX/en/001-file.xml

To use default settings, very few parameters need to be provided. Given a corpus in the format
described above at ``./corpus``, the following is an example set of invocations that would
result in a model being trained, with a brief description of what each step 
does.

To build a list of training documents::

    python index.py ./corpus

This will create a directory ``corpus.model``, and produces a list of paths to documents in the
corpus, with their associated language and domain.

We then tokenize the files using the default byte n-gram tokenizer::

    python tokenize.py corpus.model

This runs each file through the tokenizer, tabulating the frequency of each token according
to language and domain. This information is distributed into buckets according to a hash
of the token, such that all the counts for any given token will be in the same bucket.

The next step is to identify the most frequent tokens by document 
frequency::

    python DFfeatureselect.py corpus.model

This sums up the frequency counts per token in each bucket, and produces a list of the highest-df
tokens for use in the IG calculation stage. Note that this implementation of DFfeatureselect
assumes byte n-gram tokenization, and will thus select a fixed number of features per ngram order.
If tokenization is replaced with a word-based tokenizer, this should be replaced accordingly.

We then compute the IG weights of each of the top features by DF. This is computed separately
for domain and for language::

    python IGweight.py -d corpus.model
    python IGweight.py -lb corpus.model

Based on the IG weights, we compute the LD score for each token::

    python LDfeatureselect.py corpus.model

This produces the final list of LD features to use for building the NB model.

We then assemble the scanner::

    python scanner.py corpus.model

The scanner is a compiled DFA over the set of features that can be used to 
count the number of times each of the features occurs in a document in a 
single pass over the document. This DFA is built using Aho-Corasick string 
matching.

Finally, we learn the actual Naive Bayes parameters::

    python NBtrain.py corpus.model

This performs a second pass over the entire corpus, tokenizing it with the scanner from the previous
step, and computing the Naive Bayes parameters P(C) and p(t|C). It then compiles the parameters
and the scanner into a model compatible with ``langid.py``. 

In this example, the final model will be at the following path::

  ./corpus.model/model

This model can then be used in ``langid.py`` by invoking it with the ``-m`` command-line option as 
follows:

    python langid.py -m ./corpus.model/model

It is also possible to edit ``langid.py`` directly to embed the new model string.


Read more
---------
``langid.py`` is based on our published research. [1] describes the LD feature selection technique in detail,
and [2] provides more detail about the module ``langid.py`` itself. [3] compares the speed of ``langid.py``
to Google's Chrome CLD2, as well as my own pure-C implementation and the authors' implementation on specialized
hardware.

[1] Lui, Marco and Timothy Baldwin (2011) Cross-domain Feature Selection for Language Identification, 
In Proceedings of the Fifth International Joint Conference on Natural Language Processing (IJCNLP 2011), 
Chiang Mai, Thailand, pp. 553—561. Available from http://www.aclweb.org/anthology/I11-1062

[2] Lui, Marco and Timothy Baldwin (2012) langid.py: An Off-the-shelf Language Identification Tool, 
In Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (ACL 2012), 
Demo Session, Jeju, Republic of Korea. Available from www.aclweb.org/anthology/P12-3005

[3] Kenneth Heafield and Rohan Kshirsagar and Santiago Barona (2015) Language Identification and Modeling in Specialized Hardware,
In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint 
Conference on Natural Language Processing (Volume 2: Short Papers).
Available from http://aclweb.org/anthology/P15-2063

Contact
-------
Marco Lui <saffsd@gmail.com>

I appreciate any feedback, and I'm particularly interested in hearing about 
places where ``langid.py`` is being used. I would love to know more about 
situations where you have found that ``langid.py`` works well, and about
any shortcomings you may have found.

Acknowledgements
----------------
Thanks to aitzol for help with packaging ``langid.py`` for PyPI.
Thanks to pquentin for suggestions and improvements to packaging.

Related Implementations
-----------------------
Dawid Weiss has ported ``langid.py`` to Java, with a particular focus on
speed and memory use. Available from https://github.com/carrotsearch/langid-java

I have written a Pure-C version of ``langid.py``, which an external evaluation (see `Read more`)
has found to be up to 20x as fast as the pure Python implementation here. 
Available from https://github.com/saffsd/langid.c

I have also written a JavaScript version of ``langid.py`` which runs entirely in the browser.
Available from https://github.com/saffsd/langid.js

Changelog
---------
v1.0: 
  * Initial release

v1.1:
  * Reorganized internals to implement a LanguageIdentifier class

v1.1.2:
  * Added a 'langid' entry point

v1.1.3:
  * Made `classify` and `rank` return Python data types rather than numpy ones

v1.1.4:
  * Added set_languages to __init__.py, fixing #10 (and properly fixing #8)

v1.1.5:
  * remove dev tag
  * add PyPi classifiers, fixing #34 (thanks to pquentin)

v1.1.6:
  * make nb_numfeats an int, fixes #46, thanks to @remibolcom
