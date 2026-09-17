"""POST /v1/responses: the Responses format as a stateless translator.

Items in, items out. What Responses adds on top of chat completions stays
out on purpose: no stored state (``previous_response_id`` and
``conversation`` are refused with the instruction to send the full input)
and no hosted tools (``function`` tools only; every other tool type is
refused). Both are extras, never a silent no-op.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from mlx_beam.api.chat import (
    DEFAULT_MAX_TOKENS,
    ChatRequest,
    _normalise_messages,
    _number,
    build_prompt,
    parse_sampling,
)
from mlx_beam.api.errors import ApiError, unsupported
from mlx_beam.api.text import TextAssembler, TextDelta, stop_sequence_ids
from mlx_beam.engine.request import GenerationRequest


@dataclass
class ResponsesRequest:
    chat: ChatRequest
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    # Function tools in Responses shape, echoed back in the response object.
    tools: list[dict] = field(default_factory=list)


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
            texts.append(part.get("text", ""))
        elif kind in ("input_image",):
            raise ApiError(
                "image input needs the 'vision' extra", code="extra_not_installed"
            )
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
    for i, item in enumerate(inp):
        if not isinstance(item, dict):
            raise ApiError(f"input[{i}] must be an object", param="input")
        kind = item.get("type", "message")
        if kind == "message":
            if pending_calls:
                messages.append(
                    {"role": "assistant", "content": None, "tool_calls": pending_calls}
                )
                pending_calls = []
            role = item.get("role")
            if role not in ("user", "assistant", "system", "developer"):
                raise ApiError(
                    f"input[{i}] has an unknown role {role!r}", param="input"
                )
            messages.append(
                {
                    "role": "system" if role == "developer" else role,
                    "content": _content_text(item.get("content"), i),
                }
            )
        elif kind == "function_call":
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
            if pending_calls:
                messages.append(
                    {"role": "assistant", "content": None, "tool_calls": pending_calls}
                )
                pending_calls = []
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
            # Carried by the client for its own bookkeeping; the template
            # does not need it.
            continue
        else:
            raise unsupported(f"input item type {kind}", "input")
    if pending_calls:
        messages.append(
            {"role": "assistant", "content": None, "tool_calls": pending_calls}
        )
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


def parse_responses_request(body: dict, default_model: str) -> ResponsesRequest:
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
    if body.get("seed") is not None:
        raise unsupported("seed", "seed")
    text_format = ((body.get("text") or {}).get("format") or {}).get("type")
    if text_format not in (None, "text"):
        raise ApiError(
            "text.format needs the 'structured' extra", code="extra_not_installed"
        )
    tool_choice = body.get("tool_choice", "auto")
    if tool_choice not in ("auto", "none"):
        raise unsupported("tool_choice other than auto or none", "tool_choice")
    tools = _tools_to_chat(body.get("tools")) if tool_choice != "none" else None
    max_tokens = _number(body, "max_output_tokens", DEFAULT_MAX_TOKENS, 1, None, int)
    template_kwargs: dict[str, Any] = {}
    effort = (body.get("reasoning") or {}).get("effort")
    if effort == "none":
        template_kwargs["enable_thinking"] = False
    chat = ChatRequest(
        messages=_normalise_messages(
            items_to_messages(body.get("input"), body.get("instructions"))
        ),
        model=body.get("model") or default_model,
        max_tokens=max_tokens,
        sampling=parse_sampling(body),
        stream=bool(body.get("stream", False)),
        stop=[],
        tools=tools,
        reasoning_effort=effort,
        template_kwargs=template_kwargs,
    )
    return ResponsesRequest(
        chat=chat,
        stream=chat.stream,
        metadata=dict(body.get("metadata") or {}),
        tools=list(body.get("tools") or []),
    )


def to_generation_request(tokenizer, req: ResponsesRequest) -> GenerationRequest:
    prompt = build_prompt(tokenizer, req.chat)
    return GenerationRequest(
        tokens=prompt,
        max_tokens=req.chat.max_tokens,
        sampling=req.chat.sampling,
        stop_sequences=stop_sequence_ids(tokenizer, None),
    )


class ResponsesResponder:
    def __init__(self, tokenizer, req: ResponsesRequest, prompt_tokens: list[int]):
        self.req = req
        self.id = f"resp_{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.prompt_len = len(prompt_tokens)
        self.assembler = TextAssembler(
            tokenizer,
            prompt_tokens=prompt_tokens,
            tools=req.chat.tools,
            streaming=False,
        )
        self.msg_id = f"msg_{uuid.uuid4().hex[:24]}"

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
            "instructions": None,
            "metadata": self.req.metadata,
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "store": False,
            "temperature": self.req.chat.sampling.temperature,
            "top_p": self.req.chat.sampling.top_p,
            "tool_choice": "auto",
            "tools": self.req.tools,
            "usage": {
                "input_tokens": self.prompt_len,
                "output_tokens": self.assembler.tokens,
                "total_tokens": self.prompt_len + self.assembler.tokens,
                "input_tokens_details": {"cached_tokens": cached},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
        return out

    def _output_items(self, total: TextDelta) -> list[dict]:
        items: list[dict] = []
        if total.reasoning:
            items.append(
                {
                    "id": f"rs_{uuid.uuid4().hex[:24]}",
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": total.reasoning}],
                }
            )
        if total.content:
            items.append(
                {
                    "id": self.msg_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": total.content,
                            "annotations": [],
                        }
                    ],
                }
            )
        for tc in total.tool_calls:
            items.append(
                {
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
            )
        return items

    def complete(self, events, cached: int) -> dict:
        total = TextDelta()
        for event in events:
            d = self.assembler.feed(event)
            total.content += d.content
            total.reasoning += d.reasoning
            total.tool_calls += d.tool_calls
            if d.finish_reason:
                total.finish_reason = d.finish_reason
        incomplete = (
            {"reason": "max_output_tokens"} if total.finish_reason == "length" else None
        )
        status = "incomplete" if incomplete else "completed"
        return self._response(status, self._output_items(total), cached, incomplete)

    def stream(self, events, cached: int) -> Iterator[dict]:
        """Server-sent events of the Responses stream: created, text deltas,
        the finished items, completed."""
        seq = 0

        def ev(kind: str, **data):
            nonlocal seq
            seq += 1
            return {"type": kind, "sequence_number": seq, **data}

        yield ev("response.created", response=self._response("in_progress", [], cached))
        total = TextDelta()
        started = False
        for event in events:
            d = self.assembler.feed(event)
            total.reasoning += d.reasoning
            total.tool_calls += d.tool_calls
            if d.content:
                if not started:
                    started = True
                    yield ev(
                        "response.output_item.added",
                        output_index=0,
                        item={
                            "id": self.msg_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    )
                total.content += d.content
                yield ev(
                    "response.output_text.delta",
                    item_id=self.msg_id,
                    output_index=0,
                    content_index=0,
                    delta=d.content,
                )
            if d.finish_reason:
                total.finish_reason = d.finish_reason
        if started:
            yield ev(
                "response.output_text.done",
                item_id=self.msg_id,
                output_index=0,
                content_index=0,
                text=total.content,
            )
        incomplete = (
            {"reason": "max_output_tokens"} if total.finish_reason == "length" else None
        )
        status = "incomplete" if incomplete else "completed"
        final = self._response(status, self._output_items(total), cached, incomplete)
        yield ev(
            "response.completed" if not incomplete else "response.incomplete",
            response=final,
        )
