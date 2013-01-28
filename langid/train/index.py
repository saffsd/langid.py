#!/usr/bin/env python
"""
index.py - 
Index a corpus that is stored in a directory hierarchy as follows:

- corpus
  - domain1
    - language1
      - file1
      - file2
      - ...
    - language2
    - ...
  - domain2
    - language1
      - file1
      - file2
      - ...
    - language2
    - ...
  - ...

Marco Lui, January 2013

Copyright 2013 Marco Lui <saffsd@gmail.com>. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of the copyright holder.
"""

######
# Default values
# Can be overriden with command-line options
######
TRAIN_PROP = 1.0 # probability than any given document is selected
MIN_DOMAIN = 1 # minimum number of domains a language must be present in to be included

import os, sys, argparse
import csv
import random
import numpy
from itertools import tee, imap, islice
from collections import defaultdict

from common import Enumerator

class CorpusIndexer(object):
  """
  Class to index the contents of a corpus
  """
  def __init__(self, root, min_domain=MIN_DOMAIN, proportion=TRAIN_PROP):
    self.root = root
    self.min_domain = min_domain
    self.proportion = proportion 

    self.lang_index = defaultdict(Enumerator())
    self.domain_index = defaultdict(Enumerator())
    self.coverage_index = defaultdict(set)
    self.items = list()

    self.index(root)
    self.prune_min_domain(self.min_domain)

  def index(self, root):
    # build a list of paths
    paths = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
      for docname in filenames:
        if random.random() < self.proportion:
          # Each file has 'proportion' chance of being selected.
          path = os.path.join(dirpath, docname)

          # split the dirpath into identifying components
          d, lang = os.path.split(dirpath)
          d, domain = os.path.split(d)

          # index the language and the domain
          lang_id = self.lang_index[lang]
          domain_id = self.domain_index[domain]

          # add the domain-lang relation to the coverage index
          self.coverage_index[domain].add(lang)

          # add the item to our list
          self.items.append((domain_id,lang_id,docname,path))

  def prune_min_domain(self, min_domain):
    # prune files for all languages that do not occur in at least min_domain 
     
    # Work out which languages to reject as they are not present in at least 
    # the required number of domains
    lang_domain_count = defaultdict(int)
    for langs in self.coverage_index.values():
      for lang in langs:
        lang_domain_count[lang] += 1
    reject_langs = set( l for l in lang_domain_count if lang_domain_count[l] < min_domain)

    # Remove the languages from the indexer
    if reject_langs:
      #print "reject (<{0} domains): {1}".format(min_domain, sorted(reject_langs))
      reject_ids = set(self.lang_index[l] for l in reject_langs)
    
      new_lang_index = defaultdict(Enumerator())
      lm = dict()
      for k,v in self.lang_index.items():
        if v not in reject_ids:
          new_id = new_lang_index[k]
          lm[v] = new_id

      # Eliminate all entries for the languages
      self.items = [ (d, lm[l], n, p) for (d, l, n, p) in self.items if l in lm]

      self.lang_index = new_lang_index


  @property
  def dist_lang(self):
    """
    @returns A vector over frequency counts for each language
    """
    retval = numpy.zeros((len(self.lang_index),), dtype='int')
    for d, l, n, p in self.items:
      retval[l] += 1
    return retval

  @property
  def dist_domain(self):
    """
    @returns A vector over frequency counts for each domain 
    """
    retval = numpy.zeros((len(self.domain_index),), dtype='int')
    for d, l, n, p in self.items:
      retval[d] += 1
    return retval

  # TODO: Remove this as it should no longer be needed
  @property 
  def classmaps(self):
    num_instances = len(self.items)
    if num_instances == 0:
      raise ValueError("no items indexed!")
    cm_domain = numpy.zeros((num_instances, len(self.domain_index)), dtype='bool')
    cm_lang = numpy.zeros((num_instances, len(self.lang_index)), dtype='bool')

    # Populate the class maps
    for docid, (domain_id, lang_id, docname, path) in enumerate(self.items):
      cm_domain[docid, domain_id] = True
      cm_lang[docid, lang_id] = True
    return cm_domain, cm_lang

  @property
  def paths(self):
    return [ p for (d,l,n,p) in self.items ]


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-c","--corpus", help="read corpus from DIR", metavar="DIR")
  parser.add_argument("-p","--proportion", type=float, help="proportion of training data to use", default=TRAIN_PROP)
  parser.add_argument("-o","--output", help="produce output in DIR", metavar="DIR", default='.')
  parser.add_argument("--min_domain", type=int, help="minimum number of domains a language must be present in", default=MIN_DOMAIN)

  args = parser.parse_args()

  # check options
  if not args.corpus:
    parser.error("corpus(-c) must be specified")

  corpus_name = os.path.basename(args.corpus)

  langs_path = os.path.join(args.output, corpus_name + '.langs')
  domains_path = os.path.join(args.output, corpus_name + '.domains')
  index_path = os.path.join(args.output, corpus_name + '.index')

  # display paths
  print "corpus path:", args.corpus
  print "writing langs to:", langs_path
  print "writing domains to:", domains_path
  print "writing index to:", index_path

  indexer = CorpusIndexer(args.corpus, args.min_domain, args.proportion)

  # Compute mappings between files, languages and domains
  lang_dist = indexer.dist_lang
  lang_index = indexer.lang_index
  lang_info = ' '.join(("{0}({1})".format(k, lang_dist[v]) for k,v in lang_index.items()))
  print "langs({0}): {1}".format(len(lang_dist), lang_info)

  domain_dist = indexer.dist_domain
  domain_index = indexer.domain_index
  domain_info = ' '.join(("{0}({1})".format(k, domain_dist[v]) for k,v in domain_index.items()))
  print "domains({0}): {1}".format(len(domain_dist), domain_info)

  print "identified {0} files".format(len(indexer.items))

  # output the language index
  with open(langs_path,'w') as f:
    for lang in sorted(lang_index.keys(), key=lang_index.get):
      f.write(lang+'\n')

  # output the domain index
  with open(domains_path,'w') as f:
    for domain in sorted(domain_index.keys(), key=domain_index.get):
      f.write(domain+'\n')

  # output items found
  with open(index_path,'w') as f:
    writer = csv.writer(f)
    writer.writerows( (d,l,p) for (d,l,n,p) in indexer.items )


