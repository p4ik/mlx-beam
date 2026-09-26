"""Access: the key policy that follows the bind, the Host names the server
answers to, and how the routes refuse what does not match."""

import http.client
import json
import threading

import pytest

from mlx_beam.api.access import Access, AccessError
from mlx_beam.engine import Engine
from mlx_beam.server import BeamServer, Served
from tests.stub_tokenizer import StubTokenizer
from tests.test_vendor_optiq_kv import tiny_llama


def test_the_policy_follows_the_bind():
    assert Access.resolve("127.0.0.1").mode == "loopback"
    assert Access.resolve("localhost").mode == "loopback"
    assert Access.resolve("::1").mode == "loopback"
    with pytest.raises(AccessError, match="--api-key.*--skip-api-key"):
        Access.resolve("0.0.0.0")
    with pytest.raises(AccessError):
        Access.resolve("192.168.1.10")
    assert Access.resolve("0.0.0.0", api_key="k").mode == "key"
    assert Access.resolve("0.0.0.0", skip_api_key=True).mode == "skipped"
    with pytest.raises(AccessError, match="exclude"):
        Access.resolve("127.0.0.1", api_key="k", skip_api_key=True)
    # A key on loopback is honoured too: the operator asked for it.
    assert Access.resolve("127.0.0.1", api_key="k").mode == "key"


def test_hosts_are_localhost_the_bind_and_the_named_ones():
    a = Access.resolve("0.0.0.0", skip_api_key=True, allowed_hosts=["Mac.LAN"])
    assert a.wildcard
    for ok in ("localhost:8000", "127.0.0.1", "[::1]:8000", "::1", "mac.lan:8000"):
        assert a.host_allowed(ok), ok
    assert not a.host_allowed("evil.test:8000")
    assert not a.host_allowed("192.168.1.10:8000")
    assert a.host_allowed(None)  # HTTP/1.0 sends none; no browser does that
    b = Access.resolve("192.168.1.10", api_key="k")
    assert b.host_allowed("192.168.1.10:8000") and not b.wildcard
    assert b.describe() == {
        "mode": "key",
        "allowed_hosts": ["127.0.0.1", "192.168.1.10", "::1", "localhost"],
    }


def test_the_key_is_read_from_either_header_and_compared_whole():
    a = Access.resolve("0.0.0.0", api_key="s3cret")
    assert a.authorized({"Authorization": "Bearer s3cret"})
    assert a.authorized({"authorization": "bearer s3cret"}) is False  # exact key
    assert a.authorized({"x-api-key": "s3cret"})
    assert not a.authorized({"Authorization": "Bearer s3cre"})
    assert not a.authorized({"Authorization": "Basic s3cret"})
    assert not a.authorized({})
    # A header value outside ASCII (http.server decodes Latin-1) is a wrong
    # key, not an exception that drops the connection.
    assert not a.authorized({"x-api-key": "s3cr\u00e9t"})
    assert not a.authorized({"Authorization": "Bearer s3cr\u00e9t"})
    assert Access.local().authorized({})
    assert Access.resolve("0.0.0.0", skip_api_key=True).authorized({})


@pytest.fixture(scope="module")
def keyed():
    engine = Engine(tiny_llama()).start()
    served = Served(
        engine,
        StubTokenizer(),
        "tiny",
        allowed_origins=["http://ok.test"],
        access=Access.resolve("127.0.0.1", api_key="s3cret", allowed_hosts=["mac.lan"]),
    )
    httpd = BeamServer(served, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    engine.stop()


def request(server, method, path, headers=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=30)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=data, headers=headers or {})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp, raw


def test_routes_want_the_key_and_say_so(keyed):
    resp, raw = request(keyed, "GET", "/health")
    err = json.loads(raw)["error"]
    assert resp.status == 401 and resp.getheader("WWW-Authenticate") == "Bearer"
    assert err["code"] == "invalid_api_key" and "x-api-key" in err["message"]
    resp, raw = request(keyed, "GET", "/health", {"Authorization": "Bearer s3cret"})
    assert resp.status == 200
    assert json.loads(raw)["api"]["auth"] == {
        "mode": "key",
        "allowed_hosts": ["127.0.0.1", "::1", "localhost", "mac.lan"],
    }
    resp, raw = request(keyed, "GET", "/metrics", {"x-api-key": "s3cret"})
    assert resp.status == 200
    resp, raw = request(keyed, "GET", "/health", {"x-api-key": "w\u00e4rong"})
    assert resp.status == 401
    resp, raw = request(
        keyed,
        "POST",
        "/v1/messages",
        {"x-api-key": "wrong", "Content-Type": "application/json"},
        {"model": "tiny", "max_tokens": 1, "messages": []},
    )
    # Anthropic's envelope on its endpoints.
    assert resp.status == 401
    assert json.loads(raw) == {
        "type": "error",
        "error": {
            "type": "authentication_error",
            "message": "a valid API key is required: Authorization: Bearer <key> "
            "or x-api-key",
        },
    }


def test_a_foreign_host_is_refused_before_the_key_is_looked_at(keyed):
    resp, raw = request(
        keyed, "GET", "/health", {"Host": "evil.test:8000", "x-api-key": "s3cret"}
    )
    assert resp.status == 403 and json.loads(raw)["error"]["code"] == "host_not_allowed"
    resp, _ = request(
        keyed, "GET", "/health", {"Host": "mac.lan", "x-api-key": "s3cret"}
    )
    assert resp.status == 200


def test_a_preflight_needs_no_key(keyed):
    resp, _ = request(keyed, "OPTIONS", "/v1/models", {"Origin": "http://ok.test"})
    assert resp.status == 204
    assert resp.getheader("Access-Control-Allow-Origin") == "http://ok.test"
    resp, _ = request(keyed, "OPTIONS", "/v1/models", {"Origin": "http://other.test"})
    assert resp.status == 204 and resp.getheader("Access-Control-Allow-Origin") is None


def test_a_public_bind_without_a_policy_fails_before_the_load(monkeypatch, caplog):
    import logging

    import mlx_beam._vendor.mlx_lm.utils as utils
    from mlx_beam.cli import main

    loaded = []
    monkeypatch.setattr(utils, "load", lambda *a, **k: loaded.append(1) or (None, None))
    with caplog.at_level(logging.ERROR, logger="beam"):
        assert main(["serve", "--model", "x", "--host", "0.0.0.0"]) == 3
    assert "--skip-api-key" in caplog.text and not loaded
