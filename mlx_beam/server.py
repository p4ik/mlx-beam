"""The HTTP server: routes, JSON in and out, server-sent events for streams.

Standard library only, one thread per connection; every request ends up as
a token request on the engine's single worker.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from mlx_beam import __version__
from mlx_beam.api import chat, completions, responses
from mlx_beam.api.errors import ApiError
from mlx_beam.engine import Engine, EngineDead

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 64 << 20


class Served:
    """What the handler threads share: the engine, the tokenizer, the name."""

    def __init__(self, engine: Engine, tokenizer, model_name: str):
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
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
        h["version"] = __version__
        return h


class Handler(BaseHTTPRequestHandler):
    served: Served  # set by make_handler
    server_version = f"mlx-beam/{__version__}"

    def log_message(self, fmt, *args):  # one line per request, via logging
        logger.info("%s %s", self.address_string(), fmt % args)

    # -- plumbing ---------------------------------------------------------

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, err: ApiError) -> None:
        self._send_json(err.status, err.body())

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _start_sse(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()

    def _sse(self, payload: Any, event: str | None = None) -> None:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        head = f"event: {event}\n" if event else ""
        self.wfile.write(f"{head}data: {data}\n\n".encode())
        self.wfile.flush()

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise ApiError("request body too large", status=413)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError as e:
            raise ApiError(f"invalid JSON: {e.msg}") from None

    # -- routes -----------------------------------------------------------

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
        try:
            if handler is None:
                raise ApiError("not found", status=404, type="not_found_error")
            handler(self._read_json())
        except ApiError as e:
            self._send_error(e)
        except EngineDead as e:
            self._send_error(
                ApiError(str(e), status=503, type="server_error", code="engine_dead")
            )
        except (BrokenPipeError, ConnectionResetError):
            logger.info("client went away")
        except Exception as e:  # noqa: BLE001 - reported, never swallowed
            logger.exception("request failed")
            self._send_error(ApiError(repr(e), status=500, type="server_error"))

    # -- the three generation endpoints -----------------------------------

    def _run(self, gen_request, responder, stream: bool, sse_events: bool = False):
        """Submit, then either stream chunks as they come or answer once.

        The first event is awaited before any header goes out, so a request
        the worker rejects still gets a proper error status."""
        result = self.served.engine.submit(gen_request)
        events = iter(result)
        try:
            first = next(events, None)
            if first is None:
                raise ApiError("the engine produced nothing", status=500)
            chain = _chain(first, events)
            if stream:
                self._start_sse()
                for payload in responder.stream(chain, result.prompt_cached):
                    self._sse(payload, payload.get("type") if sse_events else None)
                if not sse_events:
                    self._sse("[DONE]")
            else:
                self._send_json(200, responder.complete(chain, result.prompt_cached))
        except BaseException:
            result.cancel()
            raise

    def _chat(self, body: dict) -> None:
        tok = self.served.tokenizer
        req = chat.parse_chat_request(body, self.served.model_name)
        gen_request = chat.to_generation_request(tok, req)
        responder = chat.ChatResponder(tok, req, gen_request.tokens)
        self._run(gen_request, responder, req.stream)

    def _completions(self, body: dict) -> None:
        tok = self.served.tokenizer
        req = completions.parse_completion_request(body, self.served.model_name)
        gen_request = completions.to_generation_request(tok, req)
        responder = completions.CompletionResponder(tok, req, gen_request.tokens)
        self._run(gen_request, responder, req.stream)

    def _responses(self, body: dict) -> None:
        tok = self.served.tokenizer
        req = responses.parse_responses_request(body, self.served.model_name)
        gen_request = responses.to_generation_request(tok, req)
        responder = responses.ResponsesResponder(tok, req, gen_request.tokens)
        self._run(gen_request, responder, req.stream, sse_events=True)


def _chain(first, rest) -> Iterator:
    yield first
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
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
