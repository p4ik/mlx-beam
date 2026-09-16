"""The HTTP server end to end on a tiny model with the stub tokenizer."""

import http.client
import json
import threading

import pytest

from mlx_beam.engine import Engine
from mlx_beam.server import BeamServer, Served
from tests.stub_tokenizer import StubTokenizer
from tests.test_vendor_optiq_kv import tiny_llama


@pytest.fixture(scope="module")
def server():
    engine = Engine(tiny_llama()).start()
    served = Served(engine, StubTokenizer(), "tiny")
    httpd = BeamServer(served, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    engine.stop()


def call(server, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=60)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=data, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, resp.getheader("Content-Type", ""), raw


def sse_payloads(raw: bytes):
    out = []
    for block in raw.decode().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                out.append(line[6:])
    return out


def test_health_and_models(server):
    status, _, raw = call(server, "GET", "/health")
    h = json.loads(raw)
    assert status == 200 and h["alive"] and h["model"] == "tiny"
    assert h["kv"]["applied"][0]["type"] == "KVCache"
    status, _, raw = call(server, "GET", "/v1/models")
    models = json.loads(raw)
    assert status == 200 and models["data"][0]["id"] == "tiny"
    assert models["data"][0]["capabilities"]["tools"] is True


def test_chat_completion(server):
    body = {
        "messages": [{"role": "user", "content": "w1 w2"}],
        "max_tokens": 4,
        "temperature": 0,
    }
    status, ctype, raw = call(server, "POST", "/v1/chat/completions", body)
    out = json.loads(raw)
    assert status == 200 and ctype.startswith("application/json")
    assert out["object"] == "chat.completion" and out["model"] == "tiny"
    assert out["usage"]["completion_tokens"] <= 4
    assert out["choices"][0]["finish_reason"] in ("stop", "length")


def test_chat_completion_streams(server):
    body = {
        "messages": [{"role": "user", "content": "w1 w2"}],
        "max_tokens": 3,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    status, ctype, raw = call(server, "POST", "/v1/chat/completions", body)
    assert status == 200 and ctype.startswith("text/event-stream")
    payloads = sse_payloads(raw)
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "usage" in chunks[-1] and chunks[-1]["choices"] == []
    assert any(c["choices"] and c["choices"][0]["finish_reason"] for c in chunks)


def test_completions_with_token_ids(server):
    status, _, raw = call(
        server, "POST", "/v1/completions", {"prompt": [1, 2, 3], "max_tokens": 2}
    )
    out = json.loads(raw)
    assert status == 200 and out["object"] == "text_completion"
    assert out["usage"]["prompt_tokens"] == 3


def test_responses(server):
    status, _, raw = call(
        server, "POST", "/v1/responses", {"input": "w1", "max_output_tokens": 2}
    )
    out = json.loads(raw)
    assert status == 200 and out["object"] == "response"
    assert out["status"] in ("completed", "incomplete")
    status, ctype, raw = call(
        server,
        "POST",
        "/v1/responses",
        {"input": "w1", "max_output_tokens": 2, "stream": True},
    )
    assert status == 200 and ctype.startswith("text/event-stream")
    assert "event: response.created" in raw.decode()


def test_errors_are_openai_shaped(server):
    status, _, raw = call(
        server, "POST", "/v1/chat/completions", {"messages": [], "seed": 3}
    )
    err = json.loads(raw)["error"]
    assert status == 400 and err["type"] == "invalid_request_error" and err["param"]
    status, _, raw = call(server, "POST", "/v1/nothing", {})
    assert status == 404
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    assert (
        resp.status == 400
        and "invalid JSON" in json.loads(resp.read())["error"]["message"]
    )
    conn.close()


def test_concurrent_clients(server):
    results = {}

    def go(i):
        body = {"messages": [{"role": "user", "content": f"w{i + 1}"}], "max_tokens": 3}
        results[i] = call(server, "POST", "/v1/chat/completions", body)[0]

    threads = [threading.Thread(target=go, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert results == {i: 200 for i in range(5)}
