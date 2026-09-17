"""The HTTP server: routes, JSON in and out, server-sent events for streams.

Standard library only, one thread per connection; every request ends up as
a token request on the engine's single worker.
"""

from __future__ import annotations

import json
import logging
import queue
import select
import signal
import socket
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from mlx_beam import __version__
from mlx_beam.api import chat, completions, responses
from mlx_beam.api.defaults import RequestDefaults
from mlx_beam.api.errors import ApiError
from mlx_beam.api.reasoning import renderer_reasoning_keys
from mlx_beam.engine import (
    ContextTooLong,
    Engine,
    EngineDead,
    InvalidRequest,
    QueueFull,
)

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 64 << 20
# Seconds of silence on a stream before an SSE comment goes out: while the
# prompt prefills, and while a tool call is decoded without a chunk.
KEEPALIVE_S = 5.0
# A socket that neither delivers the body nor takes the stream for this long
# is given up: the request is cancelled instead of holding a batch slot.
SOCKET_TIMEOUT_S = 30.0
# Tokens ready at once go out in one write; a slow socket blocks once per
# batch of chunks instead of once per token.
SSE_WRITE_BATCH_BYTES = 64 << 10


class Served:
    """What the handler threads share: the engine, the tokenizer, the name."""

    def __init__(
        self,
        engine: Engine,
        tokenizer,
        model_name: str,
        reasoning_field: str = "reasoning",
        defaults: RequestDefaults | None = None,
        allowed_origins: Sequence[str] = ("*",),
        chat_template_source: str = "model",
    ):
        if reasoning_field not in chat.REASONING_FIELDS:
            raise ValueError(f"unknown reasoning field {reasoning_field!r}")
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.reasoning_field = reasoning_field
        self.defaults = defaults or RequestDefaults()
        self.allowed_origins = tuple(allowed_origins)
        # "model", "flag" or "default": where the chat template came from.
        self.chat_template_source = chat_template_source
        self.started_at = time.time()

    def capabilities(self) -> dict:
        t = self.tokenizer
        return {
            "chat": bool(getattr(t, "has_chat_template", True)),
            "tools": bool(getattr(t, "has_tool_calling", False)),
            "thinking": bool(getattr(t, "has_thinking", False)),
            "vision": False,
            "audio": False,
        }

    def models(self) -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_name,
                    "object": "model",
                    "created": int(self.started_at),
                    "owned_by": "local",
                    "capabilities": self.capabilities(),
                }
            ],
        }

    def health(self) -> dict:
        h = self.engine.health()
        h["model"] = self.model_name
        h["capabilities"] = self.capabilities()
        h["api"] = {
            "reasoning_field": self.reasoning_field,
            "defaults": self.defaults.describe(),
            "allowed_origins": list(self.allowed_origins),
            "chat_template": self.chat_template_source,
            # Which message keys the template reads earlier reasoning from.
            "reasoning_keys_read": list(renderer_reasoning_keys(self.tokenizer)),
        }
        h["version"] = __version__
        return h


class Handler(BaseHTTPRequestHandler):
    served: Served  # set by make_handler
    server_version = f"mlx-beam/{__version__}"
    timeout = SOCKET_TIMEOUT_S
    _streaming = False
    _sse_events = False
    _pending = b""
    _last_write = 0.0

    def log_message(self, fmt, *args):  # one line per request, via logging
        logger.info("%s %s", self.address_string(), fmt % args)

    # -- plumbing ---------------------------------------------------------

    def _send_json(
        self, status: int, payload: dict, extra_headers: dict | None = None
    ) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        if status >= 400:
            # Nothing more is coming on a failed request.
            self.send_header("Connection", "close")
            self.close_connection = True
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, err: ApiError) -> None:
        # A full queue clears; a dead engine does not come back on its own.
        headers = {"Retry-After": "1"} if err.code == "queue_full" else None
        self._send_json(err.status, err.body(), headers)

    def _cors(self) -> None:
        allowed = self.served.allowed_origins
        if "*" in allowed:
            origin = "*"
        else:
            # Echo the caller's origin only when it is on the list.
            self.send_header("Vary", "Origin")
            origin = self.headers.get("Origin")
            if origin not in allowed:
                return
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        # A browser client sends its own header names (the OpenAI SDK adds
        # x-stainless-*); the preflight must allow what it asked for.
        asked = self.headers.get("Access-Control-Request-Headers")
        self.send_header(
            "Access-Control-Allow-Headers", asked or "Content-Type, Authorization"
        )
        self.send_header("Access-Control-Max-Age", "600")

    def _start_sse(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()

    def _sse_bytes(self, payload: Any, event: str | None = None) -> bytes:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        head = f"event: {event}\n" if event else ""
        return f"{head}data: {data}\n\n".encode()

    def _sse(self, payload: Any, event: str | None = None) -> None:
        self.wfile.write(self._sse_bytes(payload, event))
        self.wfile.flush()

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ApiError("Content-Length must be a number") from None
        if length < 0:
            raise ApiError("Content-Length must not be negative")
        if length > MAX_BODY_BYTES:
            raise ApiError("request body too large", status=413)
        if not length and "chunked" in (self.headers.get("Transfer-Encoding") or ""):
            raise ApiError("chunked request bodies are not supported", status=411)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw or b"{}")
        except UnicodeDecodeError as e:
            raise ApiError(f"request body is not UTF-8: {e.reason}") from None
        except json.JSONDecodeError as e:
            raise ApiError(f"invalid JSON: {e.msg}") from None

    # -- routes -----------------------------------------------------------

    def do_HEAD(self):
        self._send_error(
            ApiError("method not allowed", status=405, type="not_found_error")
        )

    do_PUT = do_DELETE = do_PATCH = do_HEAD

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/health":
            h = self.served.health()
            self._send_json(200 if h["alive"] else 503, h)
        elif path == "/v1/models":
            self._send_json(200, self.served.models())
        else:
            self._send_error(ApiError("not found", status=404, type="not_found_error"))

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        routes: dict[str, Callable[[dict], None]] = {
            "/v1/chat/completions": self._chat,
            "/v1/completions": self._completions,
            "/v1/responses": self._responses,
        }
        handler = routes.get(path)
        self._streaming = False
        try:
            if handler is None:
                raise ApiError("not found", status=404, type="not_found_error")
            handler(self._read_json())
        except ApiError as e:
            self._fail(e)
        except (ContextTooLong, InvalidRequest) as e:
            code = "context_length_exceeded" if isinstance(e, ContextTooLong) else None
            self._fail(ApiError(str(e), code=code))
        except EngineDead as e:
            self._fail(
                ApiError(str(e), status=503, type="server_error", code="engine_dead")
            )
        except QueueFull as e:
            self._fail(
                ApiError(str(e), status=503, type="server_error", code="queue_full")
            )
        except (BrokenPipeError, ConnectionResetError):
            logger.info("client went away")
        except TimeoutError:
            logger.info("client stalled for %.0fs", SOCKET_TIMEOUT_S)
            self.close_connection = True
        except Exception as e:  # noqa: BLE001 - reported, never swallowed
            logger.exception("request failed")
            self._fail(ApiError(repr(e), status=500, type="server_error"))

    def _fail(self, err: ApiError) -> None:
        """An error before the headers is a status; inside a stream it is the
        last event, since the 200 is already on the wire."""
        if not self._streaming:
            self._send_error(err)
            return
        self._sse(err.body(), "error" if self._sse_events else None)
        if not self._sse_events:
            self._sse("[DONE]")

    # -- the three generation endpoints -----------------------------------

    def _run(self, gen_request, responder, stream: bool, sse_events: bool = False):
        """Submit, then either stream chunks as they come or answer once.

        A plain request waits for the first event before any header goes
        out, so one the worker rejects gets a proper error status; a stream
        opens with the 200 at once and reports a rejection as an error event.
        While a long prefill runs, a streaming client gets SSE comments with
        the progress so proxies and clients do not give up on an idle
        connection."""
        result = self.served.engine.submit(gen_request)
        self._sse_events = sse_events
        try:
            if stream:
                self._start_sse()
                self._streaming = True
                self._last_write = time.monotonic()
                first = self._await_first(result, keepalive=True)
                if first is None:
                    raise ApiError("the engine produced nothing", status=500)
                events = self._events(first, result)
                self._pending = bytearray()
                try:
                    for payload in responder.stream(events, result.prompt_cached):
                        self._pending += self._sse_bytes(
                            payload, payload.get("type") if sse_events else None
                        )
                        # Flush when the engine has nothing more ready, or
                        # the batch is large enough as it is.
                        if (
                            not result.ready
                            or len(self._pending) >= SSE_WRITE_BATCH_BYTES
                        ):
                            self._flush_pending()
                    if not sse_events:
                        self._pending += self._sse_bytes("[DONE]")
                except OSError:
                    # The socket is the problem: nothing more goes out.
                    self._pending = bytearray()
                    raise
                finally:
                    # What was generated goes out before any error does.
                    self._flush_pending()
            else:
                first = self._await_first(result, keepalive=False)
                if first is None:
                    raise ApiError("the engine produced nothing", status=500)
                events = _chain(first, result)
                self._send_json(200, responder.complete(events, result.prompt_cached))
        finally:
            # A text-level stop word or an exception may end the response
            # while the engine is still generating.
            result.cancel()

    def _client_gone(self) -> bool:
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def _await_first(self, result, keepalive: bool):
        while True:
            try:
                return result.next_event(timeout=KEEPALIVE_S)
            except queue.Empty:
                if self._client_gone():
                    raise ConnectionResetError("client went away") from None
                if keepalive:
                    p = result.progress
                    self._comment(
                        f"prefill {p.processed}/{p.total}" if p else "waiting"
                    )

    def _events(self, first, result) -> Iterator:
        """The token events, with an SSE comment whenever nothing went out
        for KEEPALIVE_S while the worker made progress. A tool call is
        collected until it closes, so a long one is decoded in silence and a
        client's idle timeout would give the request up in the middle of
        it. A worker that stopped stepping gets no comment: the silence is
        what lets a watchdog see the hang."""
        yield first
        if first.finish_reason is not None:
            return
        engine = self.served.engine
        seen_step = engine.last_step
        while True:
            try:
                event = result.next_event(timeout=KEEPALIVE_S)
            except queue.Empty:
                if self._client_gone():
                    raise ConnectionResetError("client went away") from None
                if engine.last_step != seen_step:
                    seen_step = engine.last_step
                    self._comment("keepalive")
                continue
            if event is None:
                return
            if time.monotonic() - self._last_write >= KEEPALIVE_S:
                if self._pending:
                    self._flush_pending()
                else:
                    self._comment("keepalive")
            yield event

    def _comment(self, note: str) -> None:
        self.wfile.write(f": {note}\n\n".encode())
        self.wfile.flush()
        self._last_write = time.monotonic()

    def _flush_pending(self) -> None:
        if self._pending:
            self.wfile.write(self._pending)
            self.wfile.flush()
            self._pending = bytearray()
        self._last_write = time.monotonic()

    def _check_model(self, body: dict) -> None:
        """A model name that is not the served one is a 404, as at OpenAI;
        a missing name means the served model."""
        name = body.get("model") if isinstance(body, dict) else None
        if name and name != self.served.model_name:
            raise ApiError(
                f"model {name!r} is not served here; loaded: {self.served.model_name!r}",
                status=404,
                type="invalid_request_error",
                param="model",
                code="model_not_found",
            )

    def _chat(self, body: dict) -> None:
        self._check_model(body)
        tok = self.served.tokenizer
        req = chat.parse_chat_request(
            body, self.served.model_name, self.served.defaults
        )
        gen_request = chat.to_generation_request(
            tok, req, self.served.defaults, self.served.engine.max_context
        )
        responder = chat.ChatResponder(
            tok, req, gen_request.tokens, reasoning_field=self.served.reasoning_field
        )
        self._run(gen_request, responder, req.stream)

    def _completions(self, body: dict) -> None:
        self._check_model(body)
        tok = self.served.tokenizer
        req = completions.parse_completion_request(
            body, self.served.model_name, self.served.defaults
        )
        gen_request = completions.to_generation_request(tok, req, self.served.defaults)
        responder = completions.CompletionResponder(tok, req, gen_request.tokens)
        self._run(gen_request, responder, req.stream)

    def _responses(self, body: dict) -> None:
        self._check_model(body)
        tok = self.served.tokenizer
        req = responses.parse_responses_request(
            body, self.served.model_name, self.served.defaults
        )
        gen_request = responses.to_generation_request(
            tok, req, self.served.defaults, self.served.engine.max_context
        )
        responder = responses.ResponsesResponder(
            tok,
            req,
            gen_request.tokens,
            route_thinking=self.served.reasoning_field != "none",
        )
        self._run(gen_request, responder, req.stream, sse_events=True)


def _chain(first, rest) -> Iterator:
    yield first
    if first.finish_reason is None:
        yield from rest


def make_handler(served: Served):
    return type("BeamHandler", (Handler,), {"served": served})


class BeamServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, served: Served, host: str = "127.0.0.1", port: int = 8000):
        super().__init__((host, port), make_handler(served))
        self.served = served


def serve(served: Served, host: str, port: int) -> None:
    with BeamServer(served, host, port) as httpd:
        logger.info("listening on http://%s:%d", host, httpd.server_port)

        def stop(signum, frame):
            # shutdown() blocks until serve_forever returns; not from its thread.
            threading.Thread(target=httpd.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, stop)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        logger.info("stopped")
