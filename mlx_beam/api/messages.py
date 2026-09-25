"""Anthropic's Messages API, natively on the same blocks as the chat
dialect: the request's system field and content blocks become the
messages the chat template renders, tools and tool results map onto the
template's tool calling, the reply is content blocks in the order
thinking, text, tool_use, streamed as Anthropic's events.

Tool arguments go out in one `input_json_delta` when the call closes: what
was streamed is out, and an unparseable call comes back as text - which
only works while the call is held back until its end (the OpenAI dialect
does the same). A client shows the tool's name from `content_block_start`,
which comes with the parsed call.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from mlx_beam.api.chat import (
    DEFAULTS,
    ChatRequest,
    _normalise_messages,
    _number,
    build_prompt,
    parse_sampling,
)
from mlx_beam.api.chat import to_generation_request as chat_to_generation_request
from mlx_beam.api.defaults import RequestDefaults
from mlx_beam.api.errors import ApiError, bool_field, missing_extra, unsupported
from mlx_beam.api.text import TextAssembler
from mlx_beam.engine.request import GenerationRequest


@dataclass
class MessagesRequest:
    chat: ChatRequest
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    # Cache markers the client set, each mapped onto a message boundary.
    cache_marks: list[str] = field(default_factory=list)


# -- request -----------------------------------------------------------------


def _text_blocks(content: Any, where: str) -> str:
    """The text of a block list (or a string); other block kinds are the
    caller's to route."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ApiError(f"{where} must be a string or a list of blocks", param=where)
    out = []
    for j, block in enumerate(content):
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text":
            out.append(str(block.get("text", "")))
        else:
            raise ApiError(f"{where}[{j}] has an unsupported block type {kind!r}")
    return "".join(out)


def _cache_control(block: dict, where: str, last: bool, marks: list[str]) -> None:
    """A cache marker maps onto a checkpoint only at a message boundary:
    the last block of a message is one, a block before it is not."""
    cc = block.get("cache_control")
    if cc is None:
        return
    if not last:
        raise ApiError(
            f"{where}.cache_control sits on a block inside a message; the prompt "
            "cache checkpoints at message boundaries only - put it on the "
            "message's last block",
            param="messages",
        )
    marks.append(where)


def _convert_message(m: Any, i: int, marks: list[str]) -> list[dict]:
    """One Anthropic message into the OpenAI-shaped messages the chat path
    renders: text and thinking stay with the turn, tool_use becomes the
    turn's tool_calls, each tool_result its own tool message."""
    where = f"messages[{i}]"
    if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
        raise ApiError(f"{where}.role must be user or assistant", param="messages")
    role = m["role"]
    content = m.get("content")
    if isinstance(content, str) or content is None:
        return [{"role": role, "content": content or ""}]
    if not isinstance(content, list):
        raise ApiError(f"{where}.content must be a string or blocks", param="messages")
    texts: list[str] = []
    thinking: list[str] = []
    tool_calls: list[dict] = []
    out: list[dict] = []

    def flush() -> None:
        # The turn's own text and calls, in the order the blocks came.
        if texts or thinking or tool_calls:
            turn: dict[str, Any] = {"role": role, "content": "".join(texts)}
            if thinking:
                turn["reasoning_content"] = "".join(thinking)
            if tool_calls:
                turn["tool_calls"] = list(tool_calls)
            out.append(turn)
            texts.clear(), thinking.clear(), tool_calls.clear()

    for j, block in enumerate(content):
        kind = block.get("type") if isinstance(block, dict) else None
        here = f"{where}.content[{j}]"
        _cache_control(block, here, j == len(content) - 1, marks)
        if kind == "text":
            texts.append(str(block.get("text", "")))
        elif kind == "thinking":
            thinking.append(str(block.get("thinking", "")))
        elif kind == "redacted_thinking":
            continue  # nothing to render: the trace is not ours to read
        elif kind == "tool_use":
            if role != "assistant":
                raise ApiError(
                    f"{here}: tool_use belongs to the assistant", param="messages"
                )
            args = block.get("input")
            if not isinstance(args, dict):
                raise ApiError(f"{here}.input must be an object", param="messages")
            tool_calls.append(
                {
                    "id": str(block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {"name": str(block.get("name", "")), "arguments": args},
                }
            )
        elif kind == "tool_result":
            if role != "user":
                raise ApiError(
                    f"{here}: tool_result belongs to the user", param="messages"
                )
            flush()
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id", "")),
                    "content": _text_blocks(block.get("content"), f"{here}.content"),
                }
            )
        elif kind == "image":
            raise missing_extra("image input", "vision")
        elif kind == "document":
            raise unsupported("document blocks", "messages")
        else:
            raise ApiError(
                f"{here} has an unknown block type {kind!r}", param="messages"
            )
    flush()
    if not out:
        out.append({"role": role, "content": ""})
    return out


def _tools_to_chat(tools: Any) -> list[dict] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise ApiError("tools must be a list", param="tools")
    out = []
    for i, t in enumerate(tools):
        if not isinstance(t, dict) or not t.get("name"):
            raise ApiError(f"tools[{i}] needs a name", param="tools")
        if t.get("type") not in (None, "custom"):
            raise unsupported(f"tools[{i}].type {t['type']!r} (server tools)", "tools")
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object"},
                },
            }
        )
    return out


def parse_messages_request(
    body: dict, default_model: str, defaults: RequestDefaults = DEFAULTS
) -> MessagesRequest:
    if not isinstance(body, dict):
        raise ApiError("the request body must be a JSON object")
    for key in ("mcp_servers", "container", "context_management"):
        if body.get(key) is not None:
            raise unsupported(key, key)
    marks: list[str] = []
    messages: list[dict] = []
    system = body.get("system")
    if system is not None:
        if isinstance(system, list):
            for j, block in enumerate(system):
                if isinstance(block, dict):
                    _cache_control(block, f"system[{j}]", j == len(system) - 1, marks)
        messages.append({"role": "system", "content": _text_blocks(system, "system")})
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise ApiError("messages must be a non-empty list", param="messages")
    for i, m in enumerate(raw):
        messages.extend(_convert_message(m, i, marks))
    tools = _tools_to_chat(body.get("tools"))
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        kind = choice.get("type")
        if kind == "none":
            tools = None
        elif kind != "auto":
            raise unsupported(f"tool_choice type {kind!r}", "tool_choice")
    elif choice is not None:
        raise ApiError("tool_choice must be an object", param="tool_choice")
    max_tokens = _number(
        body, "max_tokens", defaults.max_completion_tokens, 1, None, int
    )
    stop = body.get("stop_sequences") or []
    if not isinstance(stop, list) or any(not isinstance(s, str) for s in stop):
        raise ApiError(
            "stop_sequences must be a list of strings", param="stop_sequences"
        )
    template_kwargs: dict[str, Any] = dict(defaults.chat_template_args)
    max_reasoning = None
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        if thinking.get("type") == "disabled":
            template_kwargs["enable_thinking"] = False
        elif thinking.get("type") in ("enabled", "adaptive"):
            template_kwargs["enable_thinking"] = True
            budget = thinking.get("budget_tokens")
            if budget is not None:
                max_reasoning = _number(thinking, "budget_tokens", None, 0, None, int)
        else:
            raise ApiError(
                "thinking.type must be enabled or disabled", param="thinking"
            )
    elif thinking is not None:
        raise ApiError("thinking must be an object", param="thinking")
    chat = ChatRequest(
        messages=_normalise_messages(messages),
        model=body.get("model") or default_model,
        max_tokens=max_tokens,
        sampling=parse_sampling(body, defaults),
        stream=bool_field(body, "stream"),
        stop=list(stop),
        tools=tools,
        max_reasoning_tokens=max_reasoning,
        template_kwargs=template_kwargs,
    )
    metadata = body.get("metadata")
    return MessagesRequest(
        chat=chat,
        stream=chat.stream,
        metadata=metadata if isinstance(metadata, dict) else {},
        cache_marks=marks,
    )


def to_generation_request(
    tokenizer,
    req: MessagesRequest,
    defaults: RequestDefaults = DEFAULTS,
    max_context: int | None = None,
) -> GenerationRequest:
    return chat_to_generation_request(tokenizer, req.chat, defaults, max_context)


def count_tokens(tokenizer, req: MessagesRequest) -> dict:
    """The prompt as the template renders it, counted."""
    return {"input_tokens": len(build_prompt(tokenizer, req.chat))}


# -- response ----------------------------------------------------------------

STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


class MessagesResponder:
    def __init__(
        self,
        tokenizer,
        req: MessagesRequest,
        prompt_tokens: list[int],
        route_thinking: bool = True,
    ):
        self.req = req
        self.id = f"msg_{uuid.uuid4().hex[:24]}"
        self.prompt_len = len(prompt_tokens)
        self.assembler = TextAssembler(
            tokenizer,
            prompt_tokens=prompt_tokens,
            tools=req.chat.tools,
            streaming=False,
            route_thinking=route_thinking,
            tools_enabled=req.chat.tools is not None,
        )

    def _usage(self, cached: int) -> dict:
        return {
            "input_tokens": self.prompt_len,
            "output_tokens": self.assembler.tokens,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
        }

    def _stop(self, finish: str | None) -> tuple[str | None, str | None]:
        if finish is None:
            return None, None
        if finish == "stop" and self.assembler.stopped:
            # A client stop sequence matched in the text; the engine reports
            # the stop, not which sequence, so that field stays null.
            return "stop_sequence", None
        return STOP_REASONS.get(finish, "end_turn"), None

    def _message(self, content: list[dict], stop, cached: int) -> dict:
        stop_reason, stop_sequence = stop
        return {
            "id": self.id,
            "type": "message",
            "role": "assistant",
            "model": self.req.chat.model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence,
            "usage": self._usage(cached),
        }

    def complete(self, events, cached: int) -> dict:
        """The same blocks the stream builds, as one message."""
        for _ in self.stream(events, cached):
            pass
        return self.final

    def stream(self, events, cached: int) -> Iterator[dict]:
        """Anthropic's stream: message_start, a ping, per block start /
        delta / stop, message_delta with the stop reason and the output
        usage, message_stop. `final` holds the whole message afterwards."""
        self.final: dict | None = None
        yield {
            "type": "message_start",
            "message": self._message([], (None, None), cached),
        }
        yield {"type": "ping"}
        content: list[dict] = []
        index = -1
        current: str | None = None  # "thinking" or "text" while open
        text = ""
        finish = None

        def close_current():
            nonlocal current, text
            if current == "thinking":
                content.append({"type": "thinking", "thinking": text, "signature": ""})
            elif current == "text":
                content.append({"type": "text", "text": text})
            if current is not None:
                yield {"type": "content_block_stop", "index": index}
            current = None
            text = ""

        for event in events:
            d = self.assembler.feed(event)
            if d.reasoning:
                if current != "thinking":
                    yield from close_current()
                    index += 1
                    current = "thinking"
                    yield {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "thinking",
                            "thinking": "",
                            "signature": "",
                        },
                    }
                text += d.reasoning
                yield {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": d.reasoning},
                }
            for tc in d.tool_calls:
                yield from close_current()
                index += 1
                block = {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": {},
                }
                yield {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": block,
                }
                yield {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": tc["function"]["arguments"],
                    },
                }
                yield {"type": "content_block_stop", "index": index}
                content.append(
                    {**block, "input": json.loads(tc["function"]["arguments"])}
                )
            if d.content:
                if current != "text":
                    yield from close_current()
                    index += 1
                    current = "text"
                    yield {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    }
                text += d.content
                yield {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": d.content},
                }
            if d.finish_reason:
                finish = d.finish_reason
                break
        yield from close_current()
        stop = self._stop(finish)
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": stop[0], "stop_sequence": stop[1]},
            "usage": {"output_tokens": self.assembler.tokens},
        }
        self.final = self._message(content, stop, cached)
        yield {"type": "message_stop"}


def anthropic_error(err: ApiError) -> dict:
    """Anthropic's error envelope for the messages endpoints."""
    kinds = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "request_too_large",
        429: "rate_limit_error",
        500: "api_error",
        503: "overloaded_error",
    }
    return {
        "type": "error",
        "error": {"type": kinds.get(err.status, "api_error"), "message": err.message},
    }
