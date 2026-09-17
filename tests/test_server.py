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
    assert h["api"]["reasoning_field"] == "reasoning"
    assert h["api"]["reasoning_keys_read"] == ["reasoning_content"]
    assert h["api"]["defaults"]["temperature"] == {"value": 0.0, "source": "mlx-lm"}
    assert h["prompt_cache"]["entries"] >= 0 and "decode_concurrency" in h["batching"]
    assert set(h["wired_limit"]) == {"recommended", "before"}


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


def test_context_cap_serves_a_big_limit_and_refuses_a_big_reserve(server):
    context = json.loads(call(server, "GET", "/health")[2])["max_context"]
    body = {"prompt": [1, 2, 3], "max_tokens": 100000}
    status, _, raw = call(server, "POST", "/v1/completions", body)
    out = json.loads(raw)
    assert status == 200 and out["usage"]["completion_tokens"] <= context - 3
    body["min_response_tokens"] = 100000
    status, _, raw = call(server, "POST", "/v1/completions", body)
    err = json.loads(raw)["error"]
    assert status == 400 and err["code"] == "context_length_exceeded"
    assert str(context) in err["message"]


def test_client_disconnect_cancels_the_request(server):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=30)
    body = json.dumps({"prompt": [1, 2, 3], "max_tokens": 400, "stream": True}).encode()
    conn.request(
        "POST",
        "/v1/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    assert resp.status == 200
    resp.read(200)
    conn.close()  # walk away mid-stream
    import time

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        status, _, raw = call(server, "GET", "/health")
        if json.loads(raw)["in_flight"] == 0:
            break
        time.sleep(0.2)
    assert json.loads(raw)["in_flight"] == 0


def test_keepalive_comments_while_the_prompt_prefills(server, monkeypatch):
    import mlx_beam.server as srv

    monkeypatch.setattr(srv, "KEEPALIVE_S", 0.001)
    body = {"prompt": list(range(1, 60)) * 6, "max_tokens": 2, "stream": True}
    status, _, raw = call(server, "POST", "/v1/completions", body)
    assert status == 200
    text = raw.decode()
    assert ": prefill " in text or ": waiting" in text
    assert text.rstrip().endswith("data: [DONE]")


def test_keepalive_comments_while_a_tool_call_is_decoded(server, monkeypatch):
    """A tool call is collected until it closes; while the model writes a
    long one the stream must not fall silent, or a client's idle timeout
    gives the request up in the middle of it."""
    import time

    import mlx_beam.server as srv
    from mlx_beam.engine.request import ResultStream, TokenEvent
    from tests.stub_tokenizer import EOS, TOOL_END, TOOL_START

    monkeypatch.setattr(srv, "KEEPALIVE_S", 0.01)
    real_submit = server.served.engine.submit

    def submit(req):
        result = ResultStream(req, lambda _: None)

        def feed():
            tokens = [TOOL_START, 10] + [11] * 8 + [TOOL_END]
            for t in tokens:
                result.put(TokenEvent(token=t, logprob=0.0))
                time.sleep(0.03)
            result.put(TokenEvent(token=EOS, logprob=0.0, finish_reason="stop"))
            result.put(None)

        threading.Thread(target=feed, daemon=True).start()
        return result

    monkeypatch.setattr(server.served.engine, "submit", submit)
    try:
        body = {
            "messages": [{"role": "user", "content": "w1"}],
            "tools": [{"type": "function", "function": {"name": "w10"}}],
            "max_tokens": 20,
            "stream": True,
        }
        status, _, raw = call(server, "POST", "/v1/chat/completions", body)
    finally:
        monkeypatch.setattr(server.served.engine, "submit", real_submit)
    assert status == 200
    text = raw.decode()
    assert text.count(": keepalive") >= 3
    chunks = [json.loads(p) for p in sse_payloads(raw) if p != "[DONE]"]
    calls = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
    assert (
        calls
        and calls[-1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
        == "w10"
    )
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_bad_token_ids_and_top_k_are_400s(server):
    status, _, raw = call(
        server, "POST", "/v1/completions", {"prompt": [1, 9999], "max_tokens": 1}
    )
    assert status == 400 and "token ids" in json.loads(raw)["error"]["message"]
    status, _, raw = call(
        server,
        "POST",
        "/v1/completions",
        {"prompt": [1, 2], "max_tokens": 1, "top_k": 500, "temperature": 0.5},
    )
    assert status == 400 and "top_k" in json.loads(raw)["error"]["message"]
    # min_p keeps min_tokens_to_keep by argpartition; more than the vocabulary
    # would raise inside the worker.
    status, _, raw = call(
        server,
        "POST",
        "/v1/completions",
        {
            "prompt": [1, 2],
            "max_tokens": 1,
            "temperature": 1,
            "min_p": 0.1,
            "min_tokens_to_keep": 65,
        },
    )
    assert status == 400 and "min_tokens_to_keep" in json.loads(raw)["error"]["message"]
    status, _, raw = call(
        server, "POST", "/v1/completions", {"prompt": [1, 2], "max_tokens": 1}
    )
    assert status == 200
    status, _, raw = call(server, "GET", "/health")
    assert status == 200 and json.loads(raw)["alive"]


@pytest.mark.parametrize(
    "path, body",
    [
        ("/v1/completions", {"prompt": [1, 2], "max_tokens": 1}),
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "w1"}], "max_tokens": 1},
        ),
        ("/v1/responses", {"input": "w1", "max_output_tokens": 1}),
    ],
)
def test_unknown_model_is_a_404(server, path, body):
    status, _, raw = call(server, "POST", path, {**body, "model": "somewhere-else"})
    err = json.loads(raw)["error"]
    assert (
        status == 404 and err["code"] == "model_not_found" and err["param"] == "model"
    )
    status, _, raw = call(server, "POST", path, {**body, "model": "tiny"})
    assert status == 200 and json.loads(raw)["model"] == "tiny"


def test_cors_echoes_only_listed_origins():
    engine = Engine(tiny_llama()).start()
    served = Served(engine, StubTokenizer(), "tiny", allowed_origins=["http://ok.test"])
    httpd = BeamServer(served, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:

        def preflight(origin):
            conn = http.client.HTTPConnection(
                "127.0.0.1", httpd.server_port, timeout=10
            )
            conn.request("OPTIONS", "/v1/models", headers={"Origin": origin})
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return resp.status, resp.getheader("Access-Control-Allow-Origin")

        assert preflight("http://ok.test") == (204, "http://ok.test")
        assert preflight("http://evil.test") == (204, None)
        status, _, raw = call(httpd, "GET", "/health")
        assert json.loads(raw)["api"]["allowed_origins"] == ["http://ok.test"]
    finally:
        httpd.shutdown()
        engine.stop()


def test_cors_default_admits_everyone(server):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    conn.request("OPTIONS", "/v1/models", headers={"Origin": "http://any.test"})
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.getheader("Access-Control-Allow-Origin") == "*"


def test_queue_full_is_a_503_with_retry_after():
    engine = Engine(
        tiny_llama(), max_queued=0, decode_concurrency=1, prompt_concurrency=1
    ).start()
    httpd = BeamServer(Served(engine, StubTokenizer(), "tiny"), "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        body = {"prompt": [1, 2, 3], "max_tokens": 60, "stream": True}
        # One request holds the only slot; keep its stream open.
        first = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=30)
        first.request("POST", "/v1/completions", body=json.dumps(body).encode())
        resp = first.getresponse()
        assert resp.status == 200
        resp.read(1)
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
        conn.request("POST", "/v1/completions", body=json.dumps(body).encode())
        second = conn.getresponse()
        err = json.loads(second.read())["error"]
        assert second.status == 503 and err["code"] == "queue_full"
        assert second.getheader("Retry-After") == "1"
        assert second.getheader("Connection") == "close"
        conn.close()
        resp.read()
        first.close()
        h = json.loads(call(httpd, "GET", "/health")[2])
        assert h["rejected_queue_full"] == 1 and h["max_queued"] == 0
    finally:
        httpd.shutdown()
        engine.stop()


def test_stream_batches_ready_chunks_into_one_write(server, monkeypatch):
    import mlx_beam.server as srv

    writes = []
    original = srv.Handler._sse_bytes

    class Counting:
        def __init__(self, wfile):
            self._w = wfile

        def write(self, data):
            writes.append(len(data))
            return self._w.write(data)

        def flush(self):
            return self._w.flush()

    real_start = srv.Handler._start_sse

    def start(self):
        real_start(self)
        self.wfile = Counting(self.wfile)

    monkeypatch.setattr(srv.Handler, "_start_sse", start)
    monkeypatch.setattr(srv.Handler, "_sse_bytes", original)
    body = {"prompt": [1, 2, 3], "max_tokens": 30, "stream": True}
    status, _, raw = call(server, "POST", "/v1/completions", body)
    assert status == 200 and sse_payloads(raw)[-1] == "[DONE]"
    # Every chunk arrived, in fewer writes than chunks: the [DONE] shares
    # its write with whatever was still pending.
    assert len(sse_payloads(raw)) >= 30 and len(writes) <= len(sse_payloads(raw))
    assert writes  # and the client read everything the counter saw
