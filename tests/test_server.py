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
    entry = models["data"][0]
    assert entry["capabilities"]["tools"] is True
    assert entry["owned_by"] == "mlx-beam" and entry["input_modalities"] == ["text"]
    assert entry["context_length"] == entry["max_model_len"] == h["max_context"]
    assert (
        entry["max_completion_tokens"]
        == h["api"]["defaults"]["max_completion_tokens"]["value"]
    )
    assert entry["capabilities"]["effort"] == {"source": "none"}
    assert h["reasoning"]["start"] == "<think>" and h["tools"]["start"] == "<tool_call>"
    assert set(h["tools"]["repairs"]) == {"strict", "repaired", "coerced", "failed"}
    assert h["api"]["reasoning_field"] == "reasoning"
    assert h["api"]["reasoning_keys_read"] == ["reasoning_content"]
    assert h["api"]["defaults"]["temperature"] == {"value": 0.0, "source": "mlx-lm"}
    assert h["prompt_cache"]["entries"] >= 0 and "decode_concurrency" in h["batching"]
    assert set(h["wired_limit"]) == {"recommended", "before"}
    # Allocator counters, not RSS: the only place a Metal transient shows.
    assert set(h["memory"]) == {"active", "peak", "cache"}
    assert all(isinstance(v, int) and v >= 0 for v in h["memory"].values())
    assert h["memory"]["peak"] >= h["memory"]["active"]


def test_the_peak_memory_window_resets_on_request(server):
    """A phase's peak, not the process's: the reset starts the window at
    zero (mlx does not seed it with the live allocation), and the next
    generation's allocations set it."""
    status, _, raw = call(server, "GET", "/health")
    peak_before = json.loads(raw)["memory"]["peak"]
    status, _, raw = call(server, "POST", "/health/reset-peak")
    assert status == 200
    assert json.loads(raw)["memory"]["peak"] == peak_before
    status, _, raw = call(server, "GET", "/health")
    after = json.loads(raw)["memory"]
    assert after["peak"] <= peak_before
    call(
        server,
        "POST",
        "/v1/completions",
        {"prompt": [1, 2, 3, 4, 5, 6, 7, 8], "max_tokens": 4, "temperature": 0},
    )
    status, _, raw = call(server, "GET", "/health")
    window = json.loads(raw)["memory"]["peak"]
    assert 0 < window and window >= after["peak"]


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


def test_metrics_expose_the_health_counters(server):
    call(
        server,
        "POST",
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "w1"}], "max_tokens": 2},
    )
    status, ctype, raw = call(server, "GET", "/metrics")
    assert status == 200 and ctype.startswith("text/plain")
    text = raw.decode()
    _, _, h = call(server, "GET", "/health")
    h = json.loads(h)
    assert (
        f'mlx_beam_generation_tokens_total{{model="tiny"}} {h["counters"]["generation_tokens"]}'
        in text
    )
    assert 'mlx_beam_alive{model="tiny"} 1' in text
    assert "# TYPE mlx_beam_prompt_cache_hits_total counter" in text
    assert 'mlx_beam_tool_calls_total{model="tiny",rung="strict"}' in text
    assert 'vllm:prompt_tokens_total{model="tiny"}' in text
    assert "mlx_beam_speculative" not in text  # no proposer configured
    # One HELP/TYPE pair per family, however many label values it has:
    # the format's parsers refuse a second pair for the same name.
    assert text.count("# TYPE mlx_beam_tool_calls_total counter") == 1
    assert text.count('mlx_beam_tool_calls_total{model="tiny",rung=') >= 2
    for line in text.splitlines():
        assert not line.startswith("# TYPE") or text.count(line) == 1


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


def test_no_keepalive_while_the_worker_makes_no_progress(monkeypatch):
    """A stalled worker must leave the stream silent: the silence is what a
    watchdog in front of the server sees. A stepping worker earns a comment."""
    import io
    import queue as q

    import mlx_beam.server as srv
    from mlx_beam.engine.request import TokenEvent

    class Stalled:
        def __init__(self):
            self.calls = 0

        def next_event(self, timeout):
            self.calls += 1
            if self.calls <= 3:
                raise q.Empty
            return None

    class FakeEngine:
        last_step = 1.0

    monkeypatch.setattr(srv, "KEEPALIVE_S", 0.001)
    handler = object.__new__(srv.Handler)
    handler.wfile = io.BytesIO()
    handler._pending = bytearray()
    handler._last_write = 0.0
    handler._client_gone = lambda: False
    handler.served = type("S", (), {"engine": FakeEngine()})()
    first = TokenEvent(1, -0.1)
    assert list(handler._events(first, Stalled())) == [first]
    assert handler.wfile.getvalue() == b""
    # The worker stepped between waits: one comment per observed step.
    stepping = Stalled()
    real = stepping.next_event

    def next_event(timeout):
        FakeEngine.last_step += 1
        return real(timeout)

    stepping.next_event = next_event
    handler.wfile = io.BytesIO()
    list(handler._events(first, stepping))
    assert handler.wfile.getvalue().count(b": keepalive") == 3


def test_a_failed_stream_write_is_not_retried(server, monkeypatch):
    import mlx_beam.server as srv

    writes = []
    real_start = srv.Handler._start_sse

    class Failing:
        def __init__(self, wfile):
            self._w = wfile

        def write(self, data):
            writes.append(bytes(data))
            if len(writes) == 2:
                raise TimeoutError("timed out")
            return self._w.write(data)

        def flush(self):
            return self._w.flush()

    def start(self):
        real_start(self)
        self.wfile = Failing(self.wfile)

    monkeypatch.setattr(srv.Handler, "_start_sse", start)
    body = {"prompt": [1, 2, 3], "max_tokens": 30, "stream": True}
    call(server, "POST", "/v1/completions", body)
    # The bytes that failed were not written a second time.
    assert len(writes) == 2 or writes[1] != writes[2]


def test_preflight_allows_the_headers_the_browser_asks_for(server):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    conn.request(
        "OPTIONS",
        "/v1/chat/completions",
        headers={
            "Origin": "http://app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-stainless-os, content-type",
        },
    )
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.status == 204
    assert (
        resp.getheader("Access-Control-Allow-Headers") == "x-stainless-os, content-type"
    )
    assert resp.getheader("Access-Control-Max-Age") == "600"


def test_health_says_how_the_template_takes_developer(server):
    status, _, raw = call(server, "GET", "/health")
    assert status == 200
    assert json.loads(raw)["template"]["roles"]["developer"] == "as system"


def test_a_background_response_is_refused_as_a_service(server):
    status, _, raw = call(
        server, "POST", "/v1/responses", {"input": "w1", "background": True}
    )
    assert status == 400 and json.loads(raw)["error"]["param"] == "background"


def test_a_socket_error_while_reading_the_body_closes_the_connection():
    """A connection that fails under the body read cannot carry an answer:
    the handler closes it and returns, no 500 written into the fault and
    no exception out of the handler thread."""
    import io
    from http.client import HTTPMessage

    from mlx_beam.server import Handler

    class Broken:
        def read(self, n):
            raise ConnectionAbortedError("aborted under the read")

    h = Handler.__new__(Handler)
    h.served = None
    h.command, h.path, h.request_version = "POST", "/v1/completions", "HTTP/1.1"
    h.requestline = "POST /v1/completions HTTP/1.1"
    h.client_address = ("127.0.0.1", 1)
    h.headers = HTTPMessage()
    h.headers["Content-Length"] = "5"
    h.rfile, h.wfile = Broken(), io.BytesIO()
    h.close_connection = False
    h.do_POST()
    assert h.close_connection and h.wfile.getvalue() == b""


def test_bad_bodies_are_400s_and_methods_405(server):
    def raw(method, headers, body=b""):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        conn.putrequest(method, "/v1/completions")
        for k, v in headers.items():
            conn.putheader(k, v)
        conn.endheaders()
        if body:
            conn.send(body)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data

    status, data = raw("POST", {"Content-Length": "-1"})
    assert status == 400 and b"negative" in data
    body = b'{"prompt": "\xff\xfe", "max_tokens": 1}'
    status, data = raw("POST", {"Content-Length": str(len(body))}, body)
    assert status == 400 and b"UTF-8" in data
    status, data = raw("POST", {"Transfer-Encoding": "chunked"})
    assert status == 411
    assert raw("HEAD", {})[0] == 405  # no body on HEAD, by the protocol
    status, data = raw("PUT", {"Content-Length": "0"})
    err = json.loads(data)["error"]
    assert status == 405 and err["type"] == "invalid_request_error"
    assert err["code"] == "method_not_allowed"


def test_no_keepalive_before_the_first_token_without_worker_progress(monkeypatch):
    """The wait for the first token follows the same rule as the stream
    after it: a comment per wait only while the worker steps."""
    import io
    import queue as q

    import mlx_beam.server as srv

    class Stalled:
        progress = None

        def __init__(self):
            self.calls = 0

        def next_event(self, timeout):
            self.calls += 1
            if self.calls <= 3:
                raise q.Empty
            return None

    class FakeEngine:
        last_step = 1.0

    monkeypatch.setattr(srv, "KEEPALIVE_S", 0.001)
    handler = object.__new__(srv.Handler)
    handler.wfile = io.BytesIO()
    handler._last_write = 0.0
    handler._client_gone = lambda: False
    handler.served = type("S", (), {"engine": FakeEngine()})()
    assert handler._await_first(Stalled(), keepalive=True) is None
    assert handler.wfile.getvalue() == b""
    stepping = Stalled()
    real = stepping.next_event

    def next_event(timeout):
        FakeEngine.last_step += 1
        return real(timeout)

    stepping.next_event = next_event
    handler.wfile = io.BytesIO()
    handler._await_first(stepping, keepalive=True)
    assert handler.wfile.getvalue().count(b": waiting") == 3


def test_a_non_streaming_client_that_leaves_is_cancelled(server, monkeypatch):
    """Nothing is written until the answer is complete, so the handler has
    to look at the socket itself: a client that closed the connection ends
    the row instead of being decoded to max_tokens for nobody."""
    import mlx_beam.server as srv

    monkeypatch.setattr(srv, "DISCONNECT_CHECK_S", 0.01)
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=30)
    body = json.dumps({"prompt": [1, 2, 3], "max_tokens": 4000}).encode()
    conn.request(
        "POST",
        "/v1/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    import time

    time.sleep(0.3)  # a few tokens in
    conn.close()  # walk away before the answer
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        status, _, raw = call(server, "GET", "/health")
        if json.loads(raw)["in_flight"] == 0:
            break
        time.sleep(0.2)
    assert json.loads(raw)["in_flight"] == 0
