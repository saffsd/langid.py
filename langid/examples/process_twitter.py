"""
Example for using langid.py to identify the language of messages
on a twitter livestream. Optionally, it can also filter messages
and display only those in a target language(s).

Expects a Twitterstream on STDIN, such as the one provided by:

# curl https://stream.twitter.com/1/statuses/sample.json -u<username> -s

Outputs lang:message one-per-line to STDOUT

Marco Lui, June 2012
"""

import sys
import langid
import json
import optparse
import re

import _twokenize


to_clean = re.compile(_twokenize.regex_or(
  _twokenize.Hearts,
  _twokenize.url,
  _twokenize.Email,
  _twokenize.emoticon,
  _twokenize.Arrows,
  _twokenize.entity,
  _twokenize.decorations,
  _twokenize.Hashtag,
  _twokenize.AtMention,
).decode('utf8'), re.UNICODE)


def clean_tweet(text):
  return to_clean.sub('', text)


def squeeze_whitespace(text):
  return re.sub('\s+', ' ', text)


if __name__ == "__main__":
  parser = optparse.OptionParser()
  parser.add_option('-l', '--langs', dest='langs', help='comma-separated set of target ISO639 language codes (e.g en,de)')
  opts, args = parser.parse_args()

  lang_set = set(opts.langs.split(",")) if opts.langs else None

  try:
    for line in sys.stdin:
      j = json.loads(line)
      if j.get('retweet_count') == 0:
        text = j.get('text')
        if text:
          lang, conf = langid.classify(clean_tweet(text))
          if lang_set is None or lang in lang_set:
            print "{0}: {1}".format(lang, squeeze_whitespace(text).encode('utf8'))
  except (IOError, KeyboardInterrupt):
    # Terminate on broken pipe or ^C
    pass

