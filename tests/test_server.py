import json
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from py3langid.langid import application


def _request(path, method='GET', body=None, query=None, content_length='auto'):
    """Run a WSGI request and return (status, payload, headers)."""
    start_response = MagicMock()
    environ = {'REQUEST_METHOD': method, 'PATH_INFO': path}
    if query is not None:
        environ['QUERY_STRING'] = query
    if body is not None:
        environ['wsgi.input'] = BytesIO(body)
        if content_length == 'auto':
            environ['CONTENT_LENGTH'] = str(len(body))
        elif content_length is not None:
            environ['CONTENT_LENGTH'] = content_length
    response = application(environ, start_response)
    status = start_response.call_args[0][0]
    headers = dict(start_response.call_args[0][1])
    payload = json.loads(response[0].decode('utf-8'))
    return status, payload, headers


@pytest.mark.parametrize('path', ['/detect', '/rank'])
@pytest.mark.parametrize('method,kwargs', [
    ('GET', {'query': 'q=This+is+a+test'}),
    ('POST', {'body': b'q=This+is+a+test'}),
    ('PUT', {'body': b'This is a test'}),
])
def test_ok(path, method, kwargs):
    status, payload, _ = _request(path, method, **kwargs)
    assert status == '200 OK'
    data = payload['responseData']
    if path == '/detect':
        assert data['language'] == 'en'
    else:
        assert data[0][0] == 'en'


def test_invalid_method():
    status, _, _ = _request('/detect', 'DELETE')
    assert status == '405 Method Not Allowed'


@pytest.mark.parametrize('path', ['/invalid', ''])
def test_invalid_path(path):
    status, _, _ = _request(path)
    assert status == '404 Not Found'


def test_no_query_string():
    status, payload, _ = _request('/detect')
    assert status == '400 Bad Request'
    assert payload['responseData'] is None


def test_get_missing_q_param():
    status, _, _ = _request('/detect', query='x=hello')
    assert status == '400 Bad Request'


def test_content_type():
    _, _, headers = _request('/detect', query='q=test')
    assert headers['Content-type'] == 'application/json; charset=utf-8'


def test_missing_content_length():
    status, _, _ = _request('/detect', 'POST', body=b'q=test', content_length=None)
    assert status == '400 Bad Request'


def test_invalid_content_length():
    status, _, _ = _request('/detect', 'POST', body=b'q=test', content_length='abc')
    assert status == '400 Bad Request'
