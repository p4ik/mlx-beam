"""POST /v1/responses: the Responses format as a stateless translator.

Items in, items out. What Responses adds on top of chat completions stays
out of the core on purpose: no stored state (``previous_response_id`` and
``conversation`` are refused with the instruction to send the full input)
and no hosted tools (``function`` tools only; every other tool type is
refused). Both are services, not formats; a refusal, never a silent no-op.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from mlx_beam.api.chat import (
    DEFAULTS,
    ChatRequest,
    _normalise_messages,
    _number,
    completion_details,
    parse_sampling,
    parse_thinking,
)
from mlx_beam.api.chat import to_generation_request as chat_to_generation_request
from mlx_beam.api.defaults import RequestDefaults
from mlx_beam.api.errors import (
    ApiError,
    bool_field,
    missing_extra,
    object_field,
    text_field,
    unsupported,
)
from mlx_beam.api.reasoning import read_aliases
from mlx_beam.api.text import TextAssembler, TextDelta
from mlx_beam.engine.request import GenerationRequest


@dataclass
class ResponsesRequest:
    chat: ChatRequest
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    # Function tools in Responses shape, echoed back in the response object.
    tools: list[dict] = field(default_factory=list)
    instructions: str | None = None
    tool_choice: str = "auto"


def _content_text(content: Any, item_index: int) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ApiError(
            f"input[{item_index}].content must be text or parts", param="input"
        )
    texts = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in ("input_text", "output_text", "text"):
            texts.append(text_field(part, "text", f"input[{item_index}]"))
        elif kind in ("input_image",):
            raise missing_extra("image input", "vision")
        elif kind in ("input_audio", "input_file"):
            raise unsupported(f"input part {kind}", "input")
        else:
            raise ApiError(
                f"input[{item_index}] has an unknown part type {kind!r}", param="input"
            )
    return "".join(texts)


def items_to_messages(inp: Any, instructions: str | None) -> list[dict]:
    messages: list[dict] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
        return messages
    if not isinstance(inp, list):
        raise ApiError("input must be a string or a list of items", param="input")
    pending_calls: list[dict] = []
    pending_reasoning = ""
    pending_calls_reasoning = ""

    def flush_calls() -> None:
        nonlocal pending_calls, pending_calls_reasoning
        if not pending_calls:
            return
        turn = {"role": "assistant", "content": None, "tool_calls": pending_calls}
        if pending_calls_reasoning:
            turn["reasoning_content"] = pending_calls_reasoning
        messages.append(turn)
        pending_calls, pending_calls_reasoning = [], ""

    for i, item in enumerate(inp):
        if not isinstance(item, dict):
            raise ApiError(f"input[{i}] must be an object", param="input")
        kind = item.get("type", "message")
        if kind == "message":
            flush_calls()
            role = item.get("role")
            if role not in ("user", "assistant", "system", "developer"):
                raise ApiError(
                    f"input[{i}] has an unknown role {role!r}", param="input"
                )
            message = {
                "role": "system" if role == "developer" else role,
                "content": _content_text(item.get("content"), i),
            }
            if role == "assistant" and pending_reasoning:
                message["reasoning_content"] = pending_reasoning
                pending_reasoning = ""
            messages.append(message)
        elif kind == "function_call":
            if pending_reasoning:
                # The block preceded the call: it belongs to that turn.
                pending_calls_reasoning = pending_reasoning
                pending_reasoning = ""
            pending_calls.append(
                {
                    "id": item.get("call_id") or item.get("id") or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }
            )
        elif kind == "function_call_output":
            flush_calls()
            output = item.get("output")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": (
                        output if isinstance(output, str) else json.dumps(output)
                    ),
                }
            )
        elif kind == "reasoning":
            # The client hands the earlier think block back; a template that
            # reads reasoning_content gets it on the assistant turn that
            # follows (mirror_reasoning fills the other keys).
            parts = item.get("content") or []
            if not isinstance(parts, list):
                raise ApiError(f"input[{i}].content must be a list", param="input")
            text = "".join(
                text_field(part, "text", f"input[{i}]")
                for part in parts
                if isinstance(part, dict)
            )
            if text:
                pending_reasoning = text
            continue
        else:
            raise unsupported(f"input item type {kind}", "input")
    flush_calls()
    if not messages:
        raise ApiError("input is empty", param="input")
    return messages


def _tools_to_chat(tools: Any) -> list[dict] | None:
    if not tools:
        return None
    if not isinstance(tools, list):
        raise ApiError("tools must be a list", param="tools")
    out = []
    for t in tools:
        kind = t.get("type") if isinstance(t, dict) else None
        if kind != "function":
            raise ApiError(
                f"unsupported tool type {kind!r}", param="tools", code="unsupported"
            )
        fn = {
            k: t[k] for k in ("name", "description", "parameters", "strict") if k in t
        }
        if "function" in t and isinstance(t["function"], dict):
            fn = {**t["function"], **fn}
        out.append({"type": "function", "function": fn})
    return out


def parse_responses_request(
    body: dict, default_model: str, defaults: RequestDefaults = DEFAULTS
) -> ResponsesRequest:
    if not isinstance(body, dict):
        raise ApiError("the request body must be a JSON object")
    for key in ("previous_response_id", "conversation"):
        if body.get(key):
            raise ApiError(
                f"{key} is not supported: this server keeps no state, send the full input",
                param=key,
                code="unsupported",
            )
    if body.get("n", 1) not in (None, 1):
        raise unsupported("n > 1", "n")
    text_format = object_field(object_field(body, "text"), "format", "text").get("type")
    if text_format not in (None, "text"):
        raise missing_extra("text.format", "structured")
    tool_choice = body.get("tool_choice") or "auto"  # null is the default too
    if tool_choice not in ("auto", "none"):
        raise unsupported("tool_choice other than auto or none", "tool_choice")
    tools = _tools_to_chat(body.get("tools")) if tool_choice != "none" else None
    max_tokens = _number(
        body, "max_output_tokens", defaults.max_completion_tokens, 1, None, int
    )
    template_kwargs: dict[str, Any] = dict(defaults.chat_template_args)
    aliases = read_aliases(body)
    level, thinking = parse_thinking(aliases)
    if thinking is not None:
        template_kwargs["enable_thinking"] = thinking
    if level is not None:
        template_kwargs["reasoning_effort"] = level
    chat = ChatRequest(
        messages=_normalise_messages(
            items_to_messages(body.get("input"), body.get("instructions"))
        ),
        model=body.get("model") or default_model,
        max_tokens=max_tokens,
        sampling=parse_sampling(body, defaults),
        stream=bool_field(body, "stream"),
        stop=[],
        tools=tools,
        reasoning_effort=level,
        max_reasoning_tokens=_number(
            aliases, "max_reasoning_tokens", None, 0, None, int
        ),
        min_response_tokens=_number(body, "min_response_tokens", None, 0, None, int),
        max_prompt_tokens=_number(body, "max_prompt_tokens", None, 1, None, int),
        template_kwargs=template_kwargs,
    )
    return ResponsesRequest(
        chat=chat,
        stream=chat.stream,
        instructions=body.get("instructions") or None,
        tool_choice=tool_choice,
        metadata=object_field(body, "metadata"),
        tools=list(body.get("tools") or []),
    )


def to_generation_request(
    tokenizer,
    req: ResponsesRequest,
    defaults: RequestDefaults = DEFAULTS,
    max_context: int | None = None,
) -> GenerationRequest:
    # Same prompt, budget and markers as chat; only the response shape differs.
    return chat_to_generation_request(tokenizer, req.chat, defaults, max_context)


class ResponsesResponder:
    def __init__(
        self,
        tokenizer,
        req: ResponsesRequest,
        prompt_tokens: list[int],
        route_thinking: bool = True,
    ):
        self.req = req
        self.id = f"resp_{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.prompt_len = len(prompt_tokens)
        self.assembler = TextAssembler(
            tokenizer,
            prompt_tokens=prompt_tokens[req.chat.assistant_start :],
            stop_words=req.chat.stop,
            tools=req.chat.tools,
            streaming=False,
            route_thinking=route_thinking,
            tools_enabled=bool(req.chat.tools),
        )

    def _response(
        self, status: str, output: list[dict], cached: int, incomplete=None
    ) -> dict:
        out = {
            "id": self.id,
            "object": "response",
            "created_at": self.created,
            "status": status,
            "model": self.req.chat.model,
            "output": output,
            "error": None,
            "incomplete_details": incomplete,
            "instructions": self.req.instructions,
            "metadata": self.req.metadata,
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "store": False,
            "temperature": self.req.chat.sampling.temperature,
            "top_p": self.req.chat.sampling.top_p,
            "tool_choice": self.req.tool_choice,
            "tools": self.req.tools,
            "usage": {
                "input_tokens": self.prompt_len,
                "output_tokens": self.assembler.tokens,
                "total_tokens": self.prompt_len + self.assembler.tokens,
                "input_tokens_details": {"cached_tokens": cached},
                "output_tokens_details": completion_details(self.assembler),
            },
        }
        return out

    def complete(self, events, cached: int) -> dict:
        """The same items the stream builds, without the events: the last
        stream event carries the whole response."""
        final = None
        for ev in self.stream(events, cached):
            final = ev
        return final["response"]

    def stream(self, events, cached: int) -> Iterator[dict]:
        """Server-sent events of the Responses stream: created, in_progress,
        one item at a time (reasoning, message, function calls) with their
        part and text events, then completed or incomplete."""
        seq = 0

        def ev(kind: str, **data):
            nonlocal seq
            seq += 1
            return {"type": kind, "sequence_number": seq, **data}

        yield ev("response.created", response=self._response("in_progress", [], cached))
        yield ev(
            "response.in_progress", response=self._response("in_progress", [], cached)
        )
        total = TextDelta()
        output: list[dict] = []
        index = -1
        current: str | None = None  # "reasoning" or "message" while open
        # Every opened item has its own id and text; the model may return
        # to reasoning or text after a tool call, and that is a new item.
        item_id = ""
        item_text = ""

        def close_current(cut: bool = False):
            nonlocal current
            if current == "reasoning":
                item = {
                    "id": item_id,
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": item_text}],
                }
                yield ev(
                    "response.reasoning_text.done",
                    item_id=item_id,
                    output_index=index,
                    content_index=0,
                    text=item_text,
                )
                yield ev("response.output_item.done", output_index=index, item=item)
                output.append(item)
            elif current == "message":
                part = {"type": "output_text", "text": item_text, "annotations": []}
                item = {
                    "id": item_id,
                    "type": "message",
                    "status": "incomplete" if cut else "completed",
                    "role": "assistant",
                    "content": [part],
                }
                yield ev(
                    "response.output_text.done",
                    item_id=item_id,
                    output_index=index,
                    content_index=0,
                    text=item_text,
                    logprobs=[],
                )
                yield ev(
                    "response.content_part.done",
                    item_id=item_id,
                    output_index=index,
                    content_index=0,
                    part=part,
                )
                yield ev("response.output_item.done", output_index=index, item=item)
                output.append(item)
            current = None

        for event in events:
            d = self.assembler.feed(event)
            # Calls go out before the text of the same delta: what parsed came
            # first, and a cut-off tail is what the length limit hit - it is
            # the item still open at the end, closed as incomplete below.
            if d.reasoning:
                if current != "reasoning":
                    yield from close_current()
                    index += 1
                    current = "reasoning"
                    item_id = f"rs_{uuid.uuid4().hex[:24]}"
                    item_text = ""
                    yield ev(
                        "response.output_item.added",
                        output_index=index,
                        item={
                            "id": item_id,
                            "type": "reasoning",
                            "summary": [],
                            "content": [],
                        },
                    )
                total.reasoning += d.reasoning
                item_text += d.reasoning
                yield ev(
                    "response.reasoning_text.delta",
                    item_id=item_id,
                    output_index=index,
                    content_index=0,
                    delta=d.reasoning,
                )
            for tc in d.tool_calls:
                yield from close_current()
                index += 1
                item = {
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
                yield ev(
                    "response.output_item.added",
                    output_index=index,
                    item={**item, "status": "in_progress", "arguments": ""},
                )
                yield ev(
                    "response.function_call_arguments.done",
                    item_id=item["id"],
                    output_index=index,
                    name=item["name"],
                    arguments=item["arguments"],
                )
                yield ev("response.output_item.done", output_index=index, item=item)
                output.append(item)
                total.tool_calls.append(tc)
            if d.content:
                if current != "message":
                    yield from close_current()
                    index += 1
                    current = "message"
                    item_id = f"msg_{uuid.uuid4().hex[:24]}"
                    item_text = ""
                    yield ev(
                        "response.output_item.added",
                        output_index=index,
                        item={
                            "id": item_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    )
                    yield ev(
                        "response.content_part.added",
                        item_id=item_id,
                        output_index=index,
                        content_index=0,
                        part={"type": "output_text", "text": "", "annotations": []},
                    )
                total.content += d.content
                item_text += d.content
                yield ev(
                    "response.output_text.delta",
                    item_id=item_id,
                    output_index=index,
                    content_index=0,
                    delta=d.content,
                    logprobs=[],
                )
            if d.finish_reason:
                total.finish_reason = d.finish_reason
                break
        # The item's status goes out with its done event; nothing already
        # on the wire is changed afterwards.
        yield from close_current(cut=total.finish_reason == "length")
        incomplete = (
            {"reason": "max_output_tokens"} if total.finish_reason == "length" else None
        )
        status = "incomplete" if incomplete else "completed"
        final = self._response(status, output, cached, incomplete)
        yield ev(
            "response.completed" if not incomplete else "response.incomplete",
            response=final,
        )
