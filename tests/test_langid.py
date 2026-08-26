
import subprocess
import sys
from pathlib import Path

import pytest

import py3langid as langid
from py3langid.langid import MODEL_FILE, LanguageIdentifier


@pytest.fixture
def identifier():
    return LanguageIdentifier.from_pickled_model(MODEL_FILE)


@pytest.fixture
def norm_identifier():
    return LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)


@pytest.mark.parametrize('text,expected', [
    (b'This text is in English.', 'en'),
    ('This text is in English.', 'en'),
    ('Test Unicode sur du texte en français', 'fr'),
])
def test_classify_and_rank(text, expected):
    assert langid.classify(text)[0] == expected
    assert langid.rank(text)[0][0] == expected


def test_norm_probs(norm_identifier):
    _, prob = norm_identifier.classify('Test Unicode sur du texte en français')
    assert 0 <= prob <= 1


def test_unnormalized(identifier):
    _, prob = identifier.classify('Test Unicode sur du texte en français')
    assert prob < 0


def test_language_subset(identifier):
    identifier.set_languages(['de', 'en', 'fr'])
    assert identifier.classify('这样不好')[0] != 'zh'


def test_allcaps_lowering():
    '''All-caps text should be lowered before classification'''
    assert langid.classify('CECI EST UN TEST EN FRANÇAIS')[0] == 'fr'
    assert langid.classify('DIES IST EIN DEUTSCHER TEXT')[0] == 'de'
    assert langid.classify('ЭТО РУССКИЙ ТЕКСТ ДЛЯ ТЕСТА')[0] == 'ru'
    # title case and mixed case are not lowered
    assert langid.classify('This is normal English text')[0] == 'en'
    assert langid.classify('NASA launched a SpaceX rocket')[0] == 'en'


def test_bytes_str_parity():
    '''bytes and str input give identical results, all-caps included'''
    text = 'CECI EST UN TEST EN FRANÇAIS'
    assert langid.classify(text) == langid.classify(text.encode('utf8'))
    assert langid.rank(text) == langid.rank(text.encode('utf8'))


def test_empty_and_short():
    '''Feature-less input scores -inf, short input does not crash'''
    for empty in ('', b'', '12345'):
        lang, score = langid.classify(empty)
        assert isinstance(lang, str)
        assert score == float('-inf')
    lang, score = langid.classify('a')
    assert isinstance(lang, str)


def test_norm_probs_empty(norm_identifier):
    '''norm_probs=True on empty input returns uniform distribution'''
    _, prob = norm_identifier.classify('')
    assert abs(prob - 1.0 / len(norm_identifier.nb_classes)) < 1e-6


def test_rank_sorted(identifier):
    '''rank() returns all languages sorted by descending score'''
    ranking = identifier.rank('Test Unicode sur du texte en français')
    assert ranking[0][0] == 'fr'
    scores = [s for _, s in ranking]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)
    assert len(ranking) == len(identifier.nb_classes)


def test_set_languages_error(identifier):
    '''set_languages raises on unknown codes'''
    with pytest.raises(ValueError, match="Unknown language code"):
        identifier.set_languages(['xx_invalid'])


def test_set_languages_reset(identifier):
    '''set_languages(None) restores the full model'''
    full = len(identifier.nb_classes)
    identifier.set_languages(['en', 'fr'])
    assert len(identifier.nb_classes) == 2
    identifier.set_languages(None)
    assert len(identifier.nb_classes) == full


def test_redirection():
    '''Test if STDIN redirection works'''
    thisdir = Path(__file__).parent
    langid_path = str(thisdir.parent / 'py3langid' / 'langid.py')
    readme_path = str(thisdir.parent / 'README.rst')
    with open(readme_path, 'rb') as f:
        readme = f.read()
    result = subprocess.check_output([sys.executable, langid_path, '-n'], input=readme)
    assert b'en' in result and b'1.0' in result


def test_cli_batch(tmp_path):
    '''Batch mode classifies files via the multiprocessing pool'''
    en = tmp_path / 'en.txt'
    en.write_bytes(b'This is an English text for testing purposes.')
    fr = tmp_path / 'fr.txt'
    fr.write_text('Ceci est un texte en français pour les tests.', encoding='utf8')
    paths = f'{en}\n{fr}\n'.encode()
    out = subprocess.check_output(['langid', '-b'], input=paths).decode()
    results = dict(line.split(',')[:2] for line in out.strip().splitlines())
    assert results[str(en)] == 'en' and results[str(fr)] == 'fr'


def test_cli_external_model(tmp_path):
    '''-m loads a model in the modelstring format (b64 + bz2 pickle)'''
    import bz2
    import lzma
    from base64 import b64encode

    from py3langid.langid import MODEL_DIR
    with lzma.open(MODEL_DIR / MODEL_FILE) as f:
        raw = f.read()
    model_path = tmp_path / 'external.model'
    model_path.write_bytes(b64encode(bz2.compress(raw, compresslevel=1)))
    result = subprocess.check_output(['langid', '-n', '-m', str(model_path)],
                                     input=b'This should be enough text.')
    assert b'en' in result and b'1.0' in result


def test_cli():
    '''Test console scripts entry point'''
    result = subprocess.check_output(['langid', '-n'], input=b'This should be enough text.')
    assert b'en' in result and b'1.0' in result
    result = subprocess.check_output(['langid', '-n', '-l', 'bg,en,uk'], input=b'This should be enough text.')
    assert b'en' in result and b'1.0' in result
