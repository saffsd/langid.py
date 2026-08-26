#!/usr/bin/env python3
"""
This file bundles language identification functions.

Modifications (fork): Copyright (c) 2021, Adrien Barbaresi.

Original code: Copyright (c) 2011 Marco Lui <saffsd@gmail.com>.
Based on research by Marco Lui and Tim Baldwin.

See LICENSE file for more info.
"""

import bz2
import json
import logging
import lzma
import pickle
from base64 import b64decode
from collections import Counter
from http import HTTPStatus
from operator import itemgetter
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np

LOGGER = logging.getLogger(__name__)

IDENTIFIER = None
MODEL_FILE = 'data/model.plzma'
MODEL_DIR = Path(__file__).parent


def _load_identifier(model_path=None, norm_probs=False, langs=None):
    """Load an identifier: external model if given, else the bundled one."""
    identifier = None
    if model_path:
        try:
            identifier = LanguageIdentifier.from_modelpath(model_path, norm_probs=norm_probs)
            LOGGER.info("Using external model: %s", model_path)
        except OSError as e:
            LOGGER.warning("Failed to load %s: %s", model_path, e)
    if identifier is None:
        identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=norm_probs)
    if langs:
        identifier.set_languages(langs)
    return identifier


def _get_identifier():
    """Return the global identifier, loading the default model if needed."""
    global IDENTIFIER
    if IDENTIFIER is None:
        LOGGER.debug('initializing identifier')
        IDENTIFIER = _load_identifier()
    return IDENTIFIER


def set_languages(langs=None):
    """Set the language subset used by the global identifier."""
    return _get_identifier().set_languages(langs)


def classify(instance):
    """Classify a text string, returning (language, confidence)."""
    return _get_identifier().classify(instance)


def rank(instance):
    """Rank all languages by likelihood, returning [(language, confidence), ...]."""
    return _get_identifier().rank(instance)


def _init_worker(model_path, norm_probs, langs):
    # spawned Pool workers get a fresh module: rebuild the parent's identifier
    global IDENTIFIER
    IDENTIFIER = _load_identifier(model_path, norm_probs, langs)


def _process_file(path, dist=False):
    with open(path, 'rb') as f:
        text = f.read()
    return path, (rank(text) if dist else classify(text))


class LanguageIdentifier:
    __slots__ = [
        '_full_model',
        '_norm_probs',
        'nb_classes',
        'nb_ptc',
        'tk_nextmove',
        'tk_output',
    ]

    @classmethod
    def _from_model_data(cls, nb_ptc, _nb_pc, nb_classes, tk_nextmove, tk_output, *args, **kwargs):
        n_classes = len(nb_classes)
        nb_ptc = np.array(nb_ptc).reshape(len(nb_ptc) // n_classes, n_classes)
        # {state: features} dict -> dense list; 256 transitions per DFA state
        output_list = [None] * (len(tk_nextmove) // 256)
        for s, v in tk_output.items():
            output_list[s] = v
        return cls(nb_ptc, nb_classes, tk_nextmove, output_list, *args, **kwargs)

    @classmethod
    def from_pickled_model(cls, pickled_file, *args, **kwargs):
        with lzma.open(MODEL_DIR / pickled_file) as f:
            data = pickle.load(f)
        return cls._from_model_data(*data, *args, **kwargs)

    @classmethod
    def from_modelstring(cls, string, *args, **kwargs):
        data = pickle.loads(bz2.decompress(b64decode(string)))
        return cls._from_model_data(*data, *args, **kwargs)

    @classmethod
    def from_modelpath(cls, path, *args, **kwargs):
        with open(path, 'rb') as f:
            return cls.from_modelstring(f.read(), *args, **kwargs)

    def __init__(self, nb_ptc, nb_classes, tk_nextmove, tk_output, norm_probs=False):
        self.nb_ptc = nb_ptc
        self.nb_classes = nb_classes
        self.tk_nextmove = tk_nextmove
        self.tk_output = tk_output
        self._norm_probs = norm_probs
        self._full_model = nb_ptc, nb_classes

    def set_languages(self, langs=None):
        LOGGER.debug("restricting languages to: %s", langs)
        nb_ptc, nb_classes = self._full_model

        if langs is None:
            self.nb_classes, self.nb_ptc = nb_classes, nb_ptc
        else:
            lang_set = set(langs)
            unknown = lang_set - set(nb_classes)
            if unknown:
                raise ValueError(f"Unknown language code(s): {unknown}")

            indices = [i for i, c in enumerate(nb_classes) if c in lang_set]
            self.nb_classes = [nb_classes[i] for i in indices]
            self.nb_ptc = nb_ptc[:, indices]

    def _score(self, text):
        if isinstance(text, bytes):
            # decode so case normalization applies uniformly to str and bytes
            try:
                text = text.decode('utf8')
            except UnicodeDecodeError:
                pass
        if isinstance(text, str):
            if text.isupper():
                text = text.lower()
            text = text.encode('utf8', errors='surrogatepass')

        # DFA walk
        state, indexes = 0, []
        extend = indexes.extend
        nm, out = self.tk_nextmove, self.tk_output

        for letter in text:
            state = nm[(state << 8) + letter]
            v = out[state]
            if v:
                extend(v)

        if indexes:
            feat_counts = Counter(indexes)
            idx = np.fromiter(feat_counts.keys(), dtype=np.intp, count=len(feat_counts))
            counts = np.fromiter(feat_counts.values(), dtype=np.float32, count=len(feat_counts))
            probs = counts @ self.nb_ptc[idx]
        else:
            # no features: minimal confidence (softmax turns this into uniform)
            fill = 0.0 if self._norm_probs else -np.inf
            probs = np.full(len(self.nb_classes), fill, dtype=np.float32)

        if self._norm_probs:
            e = np.exp(probs - probs.max())
            probs = e / e.sum()

        return probs

    def classify(self, text):
        probs = self._score(text)
        cl = probs.argmax()
        return self.nb_classes[cl], float(probs[cl])

    def rank(self, text):
        probs = self._score(text)
        return sorted(
            ((lang, float(p)) for lang, p in zip(self.nb_classes, probs)),
            key=itemgetter(1), reverse=True,
        )


def _detect(data):
    lang, conf = classify(data)
    return {'language': lang, 'confidence': conf}


_ROUTES = {'detect': _detect, 'rank': rank}


def application(environ, start_response):
    """WSGI-compatible langid web service."""
    path = environ.get('PATH_INFO', '').strip('/').partition('/')[0]
    handler = _ROUTES.get(path)
    if handler is None:
        return _return_response(start_response, 404, None, 'Not found')

    method = environ['REQUEST_METHOD']
    if method not in ('GET', 'POST', 'PUT'):
        return _return_response(start_response, 405, None, f'{method} not allowed')

    data = _get_data(environ)
    if data is None:
        return _return_response(start_response, 400, None, 'No data provided')

    return _return_response(start_response, 200, handler(data), None)


def _get_data(environ):
    method = environ['REQUEST_METHOD']
    if method in ('PUT', 'POST'):
        try:
            length = int(environ.get('CONTENT_LENGTH', 0))
        except ValueError:
            return None
        if length <= 0:
            return None
        data = environ['wsgi.input'].read(length)
        if method == 'POST':
            try:
                data = parse_qs(data)[b'q'][0]
            except KeyError:
                pass
        return data
    if method == 'GET':
        try:
            return parse_qs(environ.get('QUERY_STRING', ''))['q'][0]
        except KeyError:
            return None
    return None


def _return_response(start_response, status_code, response_data, response_details):
    status = HTTPStatus(status_code)
    response = {
        'responseData': response_data,
        'responseStatus': status_code,
        'responseDetails': response_details,
    }
    headers = [('Content-type', 'application/json; charset=utf-8')]
    start_response(f"{status.value} {status.phrase}", headers)
    return [json.dumps(response).encode('utf-8')]


def main():

    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--serve', action='store_true', help='launch web service')
    parser.add_argument('--host', help='host/ip to bind to')
    parser.add_argument('--port', default=9008, type=int, help='port to listen on')
    parser.add_argument('-v', action='count', dest='verbosity', help='increase verbosity (repeat for greater effect)')
    parser.add_argument('-m', dest='model', help='load model from file')
    parser.add_argument('-l', '--langs', help='comma-separated set of target ISO639 language codes (e.g en,de)')
    parser.add_argument('-r', '--remote', action='store_true', help='auto-detect IP address for remote access')
    parser.add_argument('-b', '--batch', action='store_true', help='specify a list of files on the command line')
    parser.add_argument('-d', '--dist', action='store_true', help='show full distribution over languages')
    parser.add_argument('-u', '--url', help='langid of URL')
    parser.add_argument('--line', action='store_true', help='process pipes line-by-line rather than as a document')
    parser.add_argument('-n', '--normalize', action='store_true', help='normalize confidence scores to probability values')
    options = parser.parse_args()

    if options.verbosity:
        logging.basicConfig(level=max((5-options.verbosity)*10, 0))
    else:
        logging.basicConfig()

    if options.batch and options.serve:
        parser.error("cannot specify both batch and serve at the same time")

    global IDENTIFIER

    langs = options.langs.split(",") if options.langs else None
    IDENTIFIER = _load_identifier(options.model, options.normalize, langs)

    _process = IDENTIFIER.rank if options.dist else IDENTIFIER.classify

    if options.url:
        from urllib.request import urlopen
        with urlopen(options.url) as url:
            text = url.read()
            output = _process(text)
            print(options.url, len(text), output)

    elif options.serve:
        import socket
        from wsgiref.simple_server import make_server

        if options.remote and options.host is None:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("google.com", 80))
                hostname = s.getsockname()[0]
        elif options.host is None:
            hostname = socket.gethostbyname(socket.gethostname())
        else:
            hostname = options.host

        print(f"Listening on {hostname}:{options.port}")
        print("Press Ctrl+C to exit")
        httpd = make_server(hostname, options.port, application)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass

    elif options.batch:
        import csv
        from functools import partial
        from multiprocessing import Pool

        def generate_paths():
            for line in sys.stdin:
                p = line.strip()
                if p and Path(p).is_file():
                    yield p

        writer = csv.writer(sys.stdout)
        with Pool(initializer=_init_worker,
                  initargs=(options.model, options.normalize, langs)) as pool:
            if options.dist:
                writer.writerow(['path'] + IDENTIFIER.nb_classes)
                for path, ranking in pool.imap_unordered(partial(_process_file, dist=True), generate_paths()):
                    ranking = dict(ranking)
                    row = [path] + [ranking[c] for c in IDENTIFIER.nb_classes]
                    writer.writerow(row)
            else:
                for path, (lang, conf) in pool.imap_unordered(_process_file, generate_paths()):
                    writer.writerow((path, lang, conf))
    else:
        if sys.stdin.isatty():
            # Interactive mode
            while True:
                try:
                    print(">>>", end=' ')
                    text = input()
                except (KeyboardInterrupt, EOFError):
                    break
                print(_process(text))
        else:
            # Redirected
            if options.line:
                for line in sys.stdin:
                    print(_process(line))
            else:
                print(_process(sys.stdin.read()))


if __name__ == "__main__":
    main()
