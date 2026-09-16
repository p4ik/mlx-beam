"""POST /v1/chat/completions: request → prompt tokens, token events → JSON."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

from mlx_beam.api.defaults import RequestDefaults
from mlx_beam.api.errors import ApiError, missing_extra, unsupported
from mlx_beam.api.reasoning import (
    effort_candidates,
    is_effort_rejection,
    map_effort,
    mirror_reasoning,
    read_aliases,
    renderer_reasoning_keys,
)
from mlx_beam.api.text import (
    TextAssembler,
    TextDelta,
    initial_state,
    logprob_entry,
    stop_sequence_ids,
    xtc_special_ids,
)
from mlx_beam.engine.request import GenerationRequest, SamplingParams
from mlx_beam.engine.thinking import ReasoningLimits, budget

# What a parser falls back to when no server defaults are handed in (tests).
DEFAULTS = RequestDefaults()


@dataclass
class ChatRequest:
    messages: list[dict]
    model: str
    max_tokens: int
    sampling: SamplingParams
    stream: bool = False
    stop: list[str] = field(default_factory=list)
    tools: list[dict] | None = None
    logprobs: bool = False
    top_logprobs: int = 0
    stream_usage: bool = False
    # The template level (low, medium, xhigh) or None: off or not asked.
    reasoning_effort: str | None = None
    # Decoder limits the request set; None means the server's value.
    max_reasoning_tokens: int | None = None
    min_response_tokens: int | None = None
    max_prompt_tokens: int | None = None
    template_kwargs: dict[str, Any] = field(default_factory=dict)


def _number(body, key, default, lo=None, hi=None, kind=float):
    value = body.get(key, default)
    if value is None:
        return default
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ApiError(f"{key} must be a finite number", param=key)
    value = kind(value)
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ApiError(f"{key} must be between {lo} and {hi}", param=key)
    return value


def parse_sampling(body: dict, defaults: RequestDefaults = DEFAULTS) -> SamplingParams:
    logit_bias = body.get("logit_bias")
    if logit_bias:
        try:
            logit_bias = {int(k): float(v) for k, v in logit_bias.items()}
        except (TypeError, ValueError, AttributeError):
            raise ApiError(
                "logit_bias must map token ids to numbers", param="logit_bias"
            ) from None
    d = defaults
    penalty = _number(body, "repetition_penalty", d.repetition_penalty, 0.0)
    presence = _number(body, "presence_penalty", d.presence_penalty or 0.0, -2.0, 2.0)
    frequency = _number(
        body, "frequency_penalty", d.frequency_penalty or 0.0, -2.0, 2.0
    )
    return SamplingParams(
        temperature=_number(body, "temperature", d.temperature, 0.0, 2.0),
        top_p=_number(body, "top_p", d.top_p, 0.0, 1.0),
        # -1 and 0 both mean "off" (the vLLM and HF convention).
        top_k=max(0, _number(body, "top_k", d.top_k, -1, None, int)),
        min_p=_number(body, "min_p", d.min_p, 0.0, 1.0),
        min_tokens_to_keep=_number(body, "min_tokens_to_keep", 1, 1, None, int),
        xtc_probability=_number(body, "xtc_probability", 0.0, 0.0, 1.0),
        xtc_threshold=_number(body, "xtc_threshold", 0.1, 0.0, 0.5),
        repetition_penalty=penalty if penalty not in (None, 1.0) else None,
        repetition_context_size=_number(
            body, "repetition_context_size", 20, 1, None, int
        ),
        presence_penalty=presence or None,
        presence_context_size=_number(body, "presence_context_size", 20, 1, None, int),
        frequency_penalty=frequency or None,
        frequency_context_size=_number(
            body, "frequency_context_size", 20, 1, None, int
        ),
        logit_bias=logit_bias or None,
        seed=_number(body, "seed", None, 0, 2**63 - 1, int),
    )


def parse_top_logprobs(body: dict) -> int:
    """OpenAI's shape: ``logprobs: true`` plus ``top_logprobs: 0..20``."""
    n = _number(body, "top_logprobs", 0, 0, 20, int)
    if n and not body.get("logprobs"):
        raise ApiError("top_logprobs needs logprobs: true", param="top_logprobs")
    return n


def parse_stop(body: dict) -> list[str]:
    stop = body.get("stop")
    if stop is None or stop == []:
        return []
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list) and all(isinstance(s, str) for s in stop):
        return stop
    raise ApiError("stop must be a string or a list of strings", param="stop")


def _reject_unsupported(body: dict) -> None:
    n = body.get("n", 1)
    if n not in (None, 1):
        raise unsupported("n > 1", "n")
    rf = body.get("response_format")
    if rf and rf.get("type") not in (None, "text"):
        raise missing_extra("response_format", "structured")
    if body.get("tool_choice") not in (None, "auto", "none"):
        raise unsupported("tool_choice other than auto or none", "tool_choice")


def _normalise_messages(messages: Any) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ApiError("messages must be a non-empty list", param="messages")
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or "role" not in m:
            raise ApiError(f"messages[{i}] needs a role", param="messages")
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            texts = []
            for part in content:
                kind = part.get("type") if isinstance(part, dict) else None
                if kind == "text":
                    texts.append(part.get("text", ""))
                elif kind in ("image_url", "input_image"):
                    raise missing_extra("image input", "vision")
                elif kind in ("input_audio", "audio", "video_url"):
                    raise missing_extra("audio and video input", "audio")
                else:
                    raise ApiError(
                        f"messages[{i}].content has an unknown part type {kind!r}",
                        param="messages",
                    )
            m["content"] = "".join(texts)
        elif content is not None and not isinstance(content, str):
            raise ApiError(f"messages[{i}].content must be text", param="messages")
        out.append(m)
    return out


def parse_chat_request(
    body: dict, default_model: str, defaults: RequestDefaults = DEFAULTS
) -> ChatRequest:
    if not isinstance(body, dict):
        raise ApiError("the request body must be a JSON object")
    _reject_unsupported(body)
    # max_tokens is OpenAI's old name; it is accepted here and nowhere else.
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    max_tokens = (
        defaults.max_completion_tokens
        if max_tokens is None
        else _number(body, "max_completion_tokens", max_tokens, 1, None, int)
    )
    tools = body.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or any(
            not isinstance(t, dict) or t.get("type") != "function" for t in tools
        ):
            raise ApiError("tools must be a list of function tools", param="tools")
        if body.get("tool_choice") == "none":
            tools = None
    template_kwargs = {
        **defaults.chat_template_args,
        **(body.get("chat_template_kwargs") or {}),
    }
    aliases = read_aliases(body)
    level, thinking = parse_thinking(aliases)
    if thinking is not None:
        # An explicit chat_template_kwargs.enable_thinking still wins.
        template_kwargs.setdefault("enable_thinking", thinking)
    if level is not None:
        template_kwargs.setdefault("reasoning_effort", level)
    stream_opts = body.get("stream_options") or {}
    return ChatRequest(
        messages=_normalise_messages(body.get("messages")),
        model=body.get("model") or default_model,
        max_tokens=max_tokens,
        sampling=parse_sampling(body, defaults),
        stream=bool(body.get("stream", False)),
        stop=parse_stop(body),
        tools=tools,
        logprobs=bool(body.get("logprobs", False)),
        top_logprobs=parse_top_logprobs(body),
        stream_usage=bool(stream_opts.get("include_usage", False)),
        reasoning_effort=level,
        max_reasoning_tokens=_number(
            aliases, "max_reasoning_tokens", None, 0, None, int
        ),
        min_response_tokens=_number(body, "min_response_tokens", None, 0, None, int),
        max_prompt_tokens=_number(body, "max_prompt_tokens", None, 1, None, int),
        template_kwargs=template_kwargs,
    )


def parse_thinking(aliases: dict) -> tuple[str | None, bool | None]:
    """(template level, enable_thinking) from the resolved aliases: an effort
    of none switches thinking off, a level asks for it."""
    thinking = aliases.get("enable_thinking")
    if "reasoning_effort" not in aliases:
        return None, thinking
    level = map_effort(aliases["reasoning_effort"])
    if level is None:
        return None, False if thinking is None else thinking
    return level, thinking


def build_prompt(tokenizer, req: ChatRequest) -> list[int]:
    if not getattr(tokenizer, "has_chat_template", True):
        raise ApiError("this model has no chat template; use /v1/completions")
    mirror_reasoning(req.messages, renderer_reasoning_keys(tokenizer))
    kwargs = dict(req.template_kwargs)
    if req.tools:
        kwargs["tools"] = req.tools

    def render(**extra):
        return tokenizer.apply_chat_template(
            req.messages,
            add_generation_prompt=True,
            tokenize=True,
            **{**kwargs, **extra},
        )

    level = kwargs.get("reasoning_effort")
    try:
        tokens = render()
    except Exception as e:  # noqa: BLE001 - the template's verdict, reported as such
        if not (level and is_effort_rejection(e)):
            raise ApiError(
                f"the chat template rejected the messages: {e}", param="messages"
            ) from None
        # The template knows the levels but not this name: climb the ladder
        # and remember the rung that rendered, so every re-render agrees.
        tokens = None
        for candidate in effort_candidates(level)[1:]:
            try:
                tokens = render(reasoning_effort=candidate)
            except Exception as e2:  # noqa: BLE001
                if not is_effort_rejection(e2):
                    raise ApiError(
                        f"the chat template rejected the messages: {e2}",
                        param="messages",
                    ) from None
                continue
            req.template_kwargs["reasoning_effort"] = candidate
            break
        if tokens is None:
            raise ApiError(
                f"the chat template accepts none of the reasoning_effort levels "
                f"{', '.join(effort_candidates(level))}: {e}",
                param="reasoning_effort",
            ) from None
    if not tokens:
        raise ApiError("the prompt is empty", param="messages")
    return list(tokens)


# Boundaries are found by re-rendering message prefixes; the last few user
# turns are enough, older ones are covered by earlier stored entries.
BOUNDARY_TURNS = 4


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):  # noqa: B905 - lengths differ on purpose
        if x != y:
            break
        n += 1
    return n


def prompt_boundaries(tokenizer, req: ChatRequest, prompt: list[int]) -> list[int]:
    """Prompt positions where a recurrent-state checkpoint pays off: the end
    of the system block and the end of the last user turns."""
    return boundaries_and_system_end(tokenizer, req, prompt)[0]


def boundaries_and_system_end(
    tokenizer, req: ChatRequest, prompt: list[int]
) -> tuple[list[int], int | None]:
    """The boundaries plus, separately, where the system block ends (None
    without a system message)."""
    kwargs = dict(req.template_kwargs)
    if req.tools:
        kwargs["tools"] = req.tools
    messages = req.messages
    ends: set[int] = set()

    def render(msgs, generation_prompt):
        try:
            return list(
                tokenizer.apply_chat_template(
                    msgs,
                    add_generation_prompt=generation_prompt,
                    tokenize=True,
                    **kwargs,
                )
            )
        except Exception:  # noqa: BLE001 - a template that refuses a prefix
            return []

    n_system = 0
    for m in messages:
        if m["role"] != "system":
            break
        n_system += 1
    system_end = None
    if n_system:
        sys_tokens = render(
            messages[:n_system] + [{"role": "user", "content": ""}], False
        )
        system_end = _common_prefix(sys_tokens, prompt)
        ends.add(system_end)
    user_idx = [i for i, m in enumerate(messages) if m["role"] == "user"]
    for i in user_idx[-BOUNDARY_TURNS:]:
        if i == len(messages) - 1:
            continue  # the prompt end is a boundary by itself
        ends.add(_common_prefix(render(messages[: i + 1], True), prompt))
    bounds = sorted(e for e in ends if 0 < e < len(prompt))
    if system_end is not None and not 0 < system_end < len(prompt):
        system_end = None
    return bounds, system_end


def to_generation_request(
    tokenizer,
    req: ChatRequest,
    defaults: RequestDefaults = DEFAULTS,
    max_context: int | None = None,
) -> GenerationRequest:
    prompt = build_prompt(tokenizer, req)
    min_response = (
        defaults.min_response_tokens
        if req.min_response_tokens is None
        else req.min_response_tokens
    )
    max_reasoning = (
        defaults.max_reasoning_tokens
        if req.max_reasoning_tokens is None
        else req.max_reasoning_tokens
    )
    completion_cap, reasoning_cap = budget(
        max_context, len(prompt), req.max_tokens, min_response
    )
    bounded = [v for v in (max_reasoning, reasoning_cap) if v is not None]
    limit = min(bounded) if bounded else None
    if limit == 0 and getattr(tokenizer, "has_thinking", False):
        # No room to think: ask the template to leave the block out, which
        # a Qwen template does; then check whether it did.
        if req.template_kwargs.get("enable_thinking") is not False:
            req.template_kwargs["enable_thinking"] = False
            prompt = build_prompt(tokenizer, req)
            completion_cap, _ = budget(
                max_context, len(prompt), req.max_tokens, min_response
            )
        limits = reasoning_limits(tokenizer, prompt, 0)
        if limits is not None and limits.seeded and len(limits.close) >= completion_cap:
            raise ApiError(
                "the prompt opens a think block the template cannot switch off; "
                f"closing it takes {len(limits.close)} tokens, the request leaves "
                f"{completion_cap} for the answer",
                code="context_length_exceeded",
            )
    bounds, system_end = boundaries_and_system_end(tokenizer, req, prompt)
    return GenerationRequest(
        tokens=prompt,
        max_tokens=req.max_tokens,
        sampling=with_xtc_specials(tokenizer, req.sampling),
        stop_sequences=stop_sequence_ids(tokenizer, req.stop),
        boundaries=bounds,
        system_end=system_end,
        top_logprobs=req.top_logprobs if req.logprobs else 0,
        min_response_tokens=min_response,
        max_prompt_tokens=req.max_prompt_tokens,
        reasoning=reasoning_limits(tokenizer, prompt, limit),
    )


def reasoning_limits(
    tokenizer, prompt: list[int], max_tokens: int | None
) -> ReasoningLimits | None:
    """The think markers as ids and the budget; None for a model without."""
    if not getattr(tokenizer, "has_thinking", False):
        return None
    start = tuple(getattr(tokenizer, "think_start_tokens", None) or ())
    end = tuple(getattr(tokenizer, "think_end_tokens", None) or ())
    if not start or not end:
        return None
    newline = tuple(tokenizer.encode("\n", add_special_tokens=False))
    # Closed the way the models were trained to close: a line break, the
    # marker, a blank line.
    close = newline + end + tuple(tokenizer.encode("\n\n", add_special_tokens=False))
    return ReasoningLimits(
        start=start,
        end=end,
        close=close,
        seeded=initial_state(tokenizer, prompt)[0] == "reasoning",
        max_tokens=max_tokens,
    )


def with_xtc_specials(tokenizer, sampling: SamplingParams) -> SamplingParams:
    if sampling.xtc_probability <= 0.0:
        return sampling
    return replace(sampling, xtc_special_tokens=xtc_special_ids(tokenizer))


def _usage(prompt_tokens: int, assembler: TextAssembler, cached: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": assembler.tokens,
        "total_tokens": prompt_tokens + assembler.tokens,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": completion_details(assembler),
    }


def completion_details(assembler: TextAssembler) -> dict:
    """OpenAI's reasoning_tokens plus which limit, if any, cut the output."""
    return {
        "reasoning_tokens": assembler.reasoning_tokens,
        "thinking_truncated": assembler.thinking_truncated,
        "response_truncated": assembler.response_truncated,
    }


# Where the thinking goes in a chat completion: the field name(s), or
# "none" to leave the think markers in the content for the client to parse.
REASONING_FIELDS = {
    "reasoning": ("reasoning",),
    "reasoning_content": ("reasoning_content",),
    "both": ("reasoning", "reasoning_content"),
    "none": (),
}


class ChatResponder:
    """Builds chat.completion objects (one, or a stream of chunks)."""

    def __init__(
        self,
        tokenizer,
        req: ChatRequest,
        prompt_tokens: list[int],
        reasoning_field: str = "reasoning",
    ):
        if reasoning_field not in REASONING_FIELDS:
            raise ValueError(f"unknown reasoning field {reasoning_field!r}")
        self.req = req
        self._tokenizer = tokenizer
        self.id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.prompt_len = len(prompt_tokens)
        self._sent_role = False
        self._reasoning_keys = REASONING_FIELDS[reasoning_field]
        self.assembler = TextAssembler(
            tokenizer,
            prompt_tokens=prompt_tokens,
            stop_words=req.stop,
            tools=req.tools,
            streaming=req.stream,
            route_thinking=reasoning_field != "none",
        )

    def _envelope(self, kind: str) -> dict:
        return {
            "id": self.id,
            "object": kind,
            "created": self.created,
            "model": self.req.model,
        }

    def _message(self, d: TextDelta) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": d.content or None}
        if d.reasoning:
            for key in self._reasoning_keys:
                msg[key] = d.reasoning
        if d.tool_calls:
            msg["tool_calls"] = d.tool_calls
        return msg

    def chunk(self, d: TextDelta, cached: int, final: bool = False) -> dict:
        delta = {}
        if d.content:
            delta["content"] = d.content
        if d.reasoning:
            for key in self._reasoning_keys:
                delta[key] = d.reasoning
        if d.tool_calls:
            delta["tool_calls"] = d.tool_calls
        if not self._sent_role:
            delta = {"role": "assistant", **delta}
            self._sent_role = True
        out = self._envelope("chat.completion.chunk")
        choice = {"index": 0, "delta": delta, "finish_reason": d.finish_reason}
        if self.req.logprobs:
            choice["logprobs"] = self._logprobs()
        out["choices"] = [choice]
        return out

    def _logprobs(self) -> dict:
        events, self.assembler.pending_events = self.assembler.pending_events, []
        return {"content": [logprob_entry(self._tokenizer, e) for e in events]}

    def stream(self, events, cached: int) -> Iterator[dict]:
        """Chunks as the tokens arrive; tool-call text is held until complete."""
        for event in events:
            d = self.assembler.feed(event)
            if d.finish_reason is None and (d.empty() or self.assembler.in_tool_call):
                continue
            yield self.chunk(d, cached, final=d.finish_reason is not None)
            if d.finish_reason is not None:
                break  # a text-level stop ends before the engine does
        if self.req.stream_usage:
            out = self._envelope("chat.completion.chunk")
            out["choices"] = []
            out["usage"] = _usage(self.prompt_len, self.assembler, cached)
            yield out

    def complete(self, events, cached: int) -> dict:
        """One chat.completion once the generation has finished."""
        total = TextDelta()
        for event in events:
            d = self.assembler.feed(event)
            total.content += d.content
            total.reasoning += d.reasoning
            total.tool_calls += d.tool_calls
            if d.finish_reason:
                total.finish_reason = d.finish_reason
                break
        out = self._envelope("chat.completion")
        choice: dict[str, Any] = {
            "index": 0,
            "message": self._message(total),
            "finish_reason": total.finish_reason or "stop",
        }
        if self.req.logprobs:
            choice["logprobs"] = self._logprobs()
        out["choices"] = [choice]
        out["usage"] = _usage(self.prompt_len, self.assembler, cached)
        return out
