"""POST /v1/chat/completions: request → prompt tokens, token events → JSON."""

from __future__ import annotations

import base64
import binascii
import json
import math
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

from mlx_beam.api.defaults import RequestDefaults
from mlx_beam.api.errors import (
    ApiError,
    bool_field,
    missing_extra,
    no_vision,
    object_field,
    text_field,
    unsupported,
)
from mlx_beam.api.reasoning import (
    EFFORT_KWARGS,
    effort_candidates,
    effort_capability,
    is_effort_rejection,
    mirror_reasoning,
    normalise_effort,
    read_aliases,
    renderer_reasoning_keys,
    translate_effort,
)
from mlx_beam.api.roles import check_role, for_template
from mlx_beam.api.text import (
    TextAssembler,
    TextDelta,
    initial_state,
    logprob_entry,
    stop_sequence_ids,
    xtc_special_ids,
)
from mlx_beam.engine.request import GenerationRequest, SamplingParams
from mlx_beam.engine.thinking import ReasoningLimits, budget, close_tail
from mlx_beam.modalities import Image

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
    # Where the assistant's own turn begins in the prompt (set by
    # build_prompt): the marker search that seeds the reasoning state looks
    # only from here - a `<think>` or a ` to=` inside a user message is
    # text, not an open block.
    assistant_start: int = 0
    # Images in the order their parts appear in the messages, and the
    # spans the frontend built for them (set by build_prompt).
    images: list[Image] = field(default_factory=list)
    spans: list = field(default_factory=list)


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
            if any(isinstance(v, bool) for v in logit_bias.values()):
                raise ValueError
            logit_bias = {int(k): float(v) for k, v in logit_bias.items()}
        except (TypeError, ValueError, AttributeError):
            raise ApiError(
                "logit_bias must map token ids to numbers", param="logit_bias"
            ) from None
        if any(
            not math.isfinite(v) or not -100 <= v <= 100 for v in logit_bias.values()
        ):
            raise ApiError("logit_bias values must be in -100..100", param="logit_bias")
    d = defaults
    penalty = _number(body, "repetition_penalty", d.repetition_penalty, 0.0)
    if penalty is not None and penalty <= 0.0:
        raise ApiError(
            "repetition_penalty must be positive", param="repetition_penalty"
        )
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
    if n not in (None, 1) or isinstance(n, bool):
        raise unsupported("n > 1", "n")
    if object_field(body, "response_format").get("type") not in (None, "text"):
        raise missing_extra("response_format", "structured")
    if body.get("tool_choice") not in (None, "auto", "none"):
        raise unsupported("tool_choice other than auto or none", "tool_choice")


def decode_image(part: dict, where: str) -> Image:
    """The bytes of an image part: an OpenAI `image_url` (or a Responses
    `input_image`) whose url is a `data:` URL. Nothing is fetched - a
    remote URL is refused, the client sends the bytes. A `detail` field
    is accepted and ignored: the processor decides the resolution."""
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str):
        raise ApiError(f"{where} needs an image_url", param="messages")
    if not url.startswith("data:"):
        raise ApiError(
            f"{where}: only data: URLs are accepted; the server fetches nothing",
            param="messages",
        )
    head, sep, payload = url[5:].partition(",")
    if not sep or ";base64" not in head:
        raise ApiError(f"{where}: the data URL must be base64", param="messages")
    try:
        # Line breaks between the base64 lines are how some encoders wrap.
        data = base64.b64decode("".join(payload.split()), validate=True)
    except (binascii.Error, ValueError):
        raise ApiError(
            f"{where}: the data URL is not valid base64", param="messages"
        ) from None
    if not data:
        raise ApiError(f"{where}: the image is empty", param="messages")
    return Image(data, head.split(";")[0] or "image/png")


def _normalise_messages(messages: Any, images: list[Image] | None = None) -> list[dict]:
    """Messages as the template renders them. Text parts fold into a
    string; with `images` given, image parts are decoded into it and the
    content stays a list of parts (`{"type": "image"}` in place), which is
    what a vision template renders; without it, an image is refused with
    the extra that serves it."""
    if not isinstance(messages, list) or not messages:
        raise ApiError("messages must be a non-empty list", param="messages")
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or "role" not in m:
            raise ApiError(f"messages[{i}] needs a role", param="messages")
        m = dict(m)
        # A role outside the format is refused here; how the template takes
        # `developer` is decided where the prompt is rendered (roles.py).
        check_role(m["role"], f"messages[{i}]")
        content = m.get("content")
        if isinstance(content, list):
            parts: list[dict] = []
            with_image = False
            for j, part in enumerate(content):
                kind = part.get("type") if isinstance(part, dict) else None
                where = f"messages[{i}].content[{j}]"
                if kind == "text":
                    parts.append(
                        {"type": "text", "text": text_field(part, "text", where)}
                    )
                elif kind in ("image_url", "input_image"):
                    if images is None:
                        raise no_vision()
                    images.append(decode_image(part, where))
                    parts.append({"type": "image"})
                    with_image = True
                elif kind in ("input_audio", "audio", "video_url"):
                    raise missing_extra("audio and video input", "audio")
                else:
                    raise ApiError(
                        f"messages[{i}].content has an unknown part type {kind!r}",
                        param="messages",
                    )
            m["content"] = parts if with_image else "".join(p["text"] for p in parts)
        elif content is None:
            # A turn made of tool calls has no content; a template that trims
            # or concatenates it must see "", not None.
            m["content"] = ""
        elif not isinstance(content, str):
            raise ApiError(f"messages[{i}].content must be text", param="messages")
        if m.get("tool_calls"):
            if not isinstance(m["tool_calls"], list):
                raise ApiError(
                    f"messages[{i}].tool_calls must be a list", param="messages"
                )
            m["tool_calls"] = [
                _normalise_tool_call(tc, f"messages[{i}].tool_calls[{j}]")
                for j, tc in enumerate(m["tool_calls"])
            ]
        out.append(m)
    return out


def _normalise_tool_call(tc: Any, where: str) -> dict:
    """Arguments reach the template as a mapping. The wire carries them as a
    JSON string; templates iterate them (Qwen3.8 raises on a string, Gemma 4
    on anything that is not a mapping), and a call without arguments comes
    as "" - which is {}, not an error."""
    if not isinstance(tc, dict) or not isinstance(tc.get("function"), dict):
        raise ApiError(f"{where} needs a function", param="messages")
    tc = dict(tc)
    fn = dict(tc["function"])
    args = fn.get("arguments")
    if args is None or args == "":
        fn["arguments"] = {}
    elif isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError as e:
            raise ApiError(
                f"{where}.function.arguments is not JSON: {e.msg}", param="messages"
            ) from None
        if not isinstance(parsed, dict):
            raise ApiError(
                f"{where}.function.arguments must be a JSON object", param="messages"
            )
        fn["arguments"] = parsed
    elif not isinstance(args, dict):
        raise ApiError(
            f"{where}.function.arguments must be a JSON object or string",
            param="messages",
        )
    tc["function"] = fn
    return tc


def parse_chat_request(
    body: dict,
    default_model: str,
    defaults: RequestDefaults = DEFAULTS,
    vision: bool = False,
) -> ChatRequest:
    """`vision` says a frontend serves images: their parts are decoded and
    kept; without one an image part is a 400 naming the extra."""
    if not isinstance(body, dict):
        raise ApiError("the request body must be a JSON object")
    _reject_unsupported(body)
    images: list[Image] | None = [] if vision else None
    # max_tokens is OpenAI's old name; it is accepted here and nowhere else.
    # An explicit null under the new name does not hide a value under the old.
    key = "max_completion_tokens"
    if body.get(key) is None and body.get("max_tokens") is not None:
        key = "max_tokens"
    max_tokens = _number(body, key, defaults.max_completion_tokens, 1, None, int)
    tools = body.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or any(
            not isinstance(t, dict) or t.get("type") != "function" for t in tools
        ):
            raise ApiError("tools must be a list of function tools", param="tools")
        if body.get("tool_choice") == "none":
            tools = None
    # Server defaults, then what the request says through its aliases, then
    # an explicit chat_template_kwargs on top of everything.
    template_kwargs = dict(defaults.chat_template_args)
    aliases = read_aliases(body)
    level, thinking = parse_thinking(aliases)
    if thinking is not None:
        template_kwargs["enable_thinking"] = thinking
    # The effort word reaches the template in build_prompt, translated to
    # what the loaded template takes (effort_capability), above a server
    # default; an explicit chat_template_kwargs entry goes through as written.
    if level is not None:
        for name in EFFORT_KWARGS:
            template_kwargs.pop(name, None)
        # A word asks for thinking: it beats a server default that switched
        # it off (the request is the more specific), not the request's own
        # enable_thinking.
        if thinking is None:
            template_kwargs.pop("enable_thinking", None)
    elif "reasoning_effort" in aliases:
        # Effort none: off, and no server-default effort word either.
        for name in EFFORT_KWARGS:
            template_kwargs.pop(name, None)
    template_kwargs.update(object_field(body, "chat_template_kwargs"))
    stream_opts = object_field(body, "stream_options")
    return ChatRequest(
        messages=_normalise_messages(body.get("messages"), images),
        model=body.get("model") or default_model,
        max_tokens=max_tokens,
        sampling=parse_sampling(body, defaults),
        stream=bool_field(body, "stream"),
        stop=parse_stop(body),
        tools=tools,
        logprobs=bool_field(body, "logprobs"),
        top_logprobs=parse_top_logprobs(body),
        stream_usage=bool(stream_opts.get("include_usage", False)),
        reasoning_effort=level,
        max_reasoning_tokens=_number(
            aliases, "max_reasoning_tokens", None, 0, None, int
        ),
        min_response_tokens=_number(body, "min_response_tokens", None, 0, None, int),
        max_prompt_tokens=_number(body, "max_prompt_tokens", None, 1, None, int),
        template_kwargs=template_kwargs,
        images=images or [],
    )


def parse_thinking(aliases: dict) -> tuple[str | None, bool | None]:
    """(effort word, enable_thinking) from the resolved aliases: an effort
    of none switches thinking off, a word asks for it (and, in
    parse_chat_request, lifts a server default that switched it off)."""
    thinking = aliases.get("enable_thinking")
    if "reasoning_effort" not in aliases:
        return None, thinking
    level = normalise_effort(aliases["reasoning_effort"])
    if level is None:
        return None, False if thinking is None else thinking
    return level, thinking


def build_prompt(tokenizer, req: ChatRequest, frontend=None) -> list[int]:
    """The prompt's token ids; with images, rendered and expanded by the
    modality frontend (which also fills `req.spans`) - through the same
    effort ladder and checks as text."""
    if not getattr(tokenizer, "has_chat_template", True):
        raise ApiError("this model has no chat template; use /v1/completions")
    if req.images and frontend is None:
        raise no_vision()
    mirror_reasoning(req.messages, renderer_reasoning_keys(tokenizer))
    kwargs = dict(req.template_kwargs)
    if req.tools:
        kwargs["tools"] = req.tools
    cap = effort_capability(tokenizer)
    effort_kwarg = cap.kwarg or "reasoning_effort"
    if req.reasoning_effort and cap.kwarg and effort_kwarg not in kwargs:
        # The client's word, as this template takes it: unchanged when it
        # does not check, the nearest rung it accepts when it does; a
        # template without the kwarg gets nothing.
        value = translate_effort(req.reasoning_effort, cap)
        if value is not None:
            kwargs[effort_kwarg] = value
            req.template_kwargs[effort_kwarg] = value

    messages = for_template(tokenizer, req.messages)

    def render(add_generation_prompt=True, **extra):
        if req.images:
            built = frontend.build(messages, req.images, {**kwargs, **extra})
            req.spans = list(built.spans)
            req.assistant_start = built.assistant_start
            return built.tokens
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **{**kwargs, **extra},
        )

    rejected = (
        "the image input was refused"
        if req.images
        else "the chat template rejected the messages"
    )
    level = kwargs.get(effort_kwarg)
    try:
        tokens = render()
    except Exception as e:  # noqa: BLE001 - the template's verdict, reported as such
        if not (level and is_effort_rejection(e)):
            raise ApiError(f"{rejected}: {e}", param="messages") from None
        # The template rejects this word after all (it answers differently
        # with tools in the context, say): climb the ladder and remember
        # the rung that rendered, so every re-render agrees.
        tokens = None
        rungs = list(effort_candidates(level)[1:])
        rungs += [w for w in cap.levels if w not in rungs and w != level]
        for candidate in rungs:
            try:
                tokens = render(**{effort_kwarg: candidate})
            except Exception as e2:  # noqa: BLE001
                if not is_effort_rejection(e2):
                    raise ApiError(f"{rejected}: {e2}", param="messages") from None
                continue
            req.template_kwargs[effort_kwarg] = candidate
            break
        if tokens is None:
            raise ApiError(
                f"the chat template accepts none of the {effort_kwarg} levels "
                f"{', '.join([level, *rungs])}: {e}",
                param="reasoning_effort",
            ) from None
    if not tokens:
        raise ApiError("the prompt is empty", param="messages")
    tokens = list(tokens)
    # The rung the ladder settled on, so this render agrees with the last.
    # With images the frontend said where the frame begins (a second build
    # would move the spans); the probe is for text prompts.
    settled = {k: v for k, v in req.template_kwargs.items() if k == effort_kwarg}
    if not req.images:
        req.assistant_start = _assistant_start(render, tokens, settled)
    opener = getattr(tokenizer, "answer_opener_tokens", None)
    if opener and kwargs.get("enable_thinking") is False and not req.tools:
        # A family whose template has no switch (Harmony, Muse): the answer
        # is opened in the prompt, so the model writes it without a block.
        # Not with tools on offer: a call needs the channel or recipient
        # the opener would skip past (Harmony's `commentary`, Muse's
        # ` to=<tool>`), so there the model keeps the choice.
        tokens += list(opener)
    return tokens


def _assistant_start(render, tokens: list[int], extra: dict) -> int:
    """Where the generation prompt begins: the prompt rendered without it is
    a prefix of the one with it, and what follows that prefix is the
    assistant's frame. A template that will not render without it puts the
    start at 0 - the whole prompt is searched, as before."""
    try:
        without = list(render(add_generation_prompt=False, **extra))
    except Exception:  # noqa: BLE001 - the template's business
        return 0
    return _common_prefix(without, tokens)


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


def boundaries_and_system_end(
    tokenizer, req: ChatRequest, prompt: list[int]
) -> tuple[list[int], int | None]:
    """The boundaries plus, separately, where the system block ends (None
    without a system message)."""
    kwargs = dict(req.template_kwargs)
    if req.tools:
        kwargs["tools"] = req.tools
    messages = for_template(tokenizer, req.messages)
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
    frontend=None,
) -> GenerationRequest:
    prompt = build_prompt(tokenizer, req, frontend)
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
    markers = reasoning_limits(tokenizer, prompt[req.assistant_start :], None)
    if reasoning_cap is not None and markers is not None:
        reasoning_cap = max(0, reasoning_cap - close_tail(markers))
    bounded = [v for v in (max_reasoning, reasoning_cap) if v is not None]
    limit = min(bounded) if bounded else None
    if (
        req.tools
        and req.template_kwargs.get("enable_thinking") is False
        and getattr(tokenizer, "answer_opener_tokens", None)
    ):
        # A family without a template switch (Harmony, Muse), thinking off
        # with tools on offer: the answer's opener stays out of the prompt
        # (build_prompt - a call needs the channel it would skip past), so
        # the block is kept out by the budget instead. Zero masks the
        # reasoning label behind the shared opener and leaves the answer's
        # and the tools' channels open: the model may call, not think.
        limit = 0
    if limit == 0 and getattr(tokenizer, "has_thinking", False):
        # No room to think: ask the template to leave the block out, which
        # a Qwen template does; then check whether it did.
        if req.template_kwargs.get("enable_thinking") is not False:
            req.template_kwargs["enable_thinking"] = False
            prompt = build_prompt(tokenizer, req, frontend)
            completion_cap, _ = budget(
                max_context, len(prompt), req.max_tokens, min_response
            )
        markers = reasoning_limits(tokenizer, prompt[req.assistant_start :], None)
        limits = markers
        if limits is not None and limits.seeded and len(limits.close) >= completion_cap:
            raise ApiError(
                "the prompt opens a think block the template cannot switch off; "
                f"closing it takes {len(limits.close)} tokens, the request leaves "
                f"{completion_cap} for the answer",
                code="context_length_exceeded",
            )
    if req.images:
        # Boundaries come from re-rendering message prefixes as text; with
        # images the placeholders expand in the processor, so no boundary
        # is claimed - a whole-entry hit (the next turn) still works, the
        # image's digest is part of the key.
        bounds, system_end = [], None
    else:
        bounds, system_end = boundaries_and_system_end(tokenizer, req, prompt)
    return GenerationRequest(
        tokens=prompt,
        max_tokens=req.max_tokens,
        sampling=with_xtc_specials(tokenizer, req.sampling),
        stop_sequences=stop_sequence_ids(tokenizer, req.stop),
        boundaries=bounds,
        system_end=system_end,
        spans=tuple(req.spans),
        top_logprobs=req.top_logprobs if req.logprobs else 0,
        min_response_tokens=min_response,
        max_prompt_tokens=req.max_prompt_tokens,
        reasoning=None if markers is None else replace(markers, max_tokens=limit),
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
    close = getattr(tokenizer, "think_close_tokens", None)
    if not close:
        newline = tuple(tokenizer.encode("\n", add_special_tokens=False))
        # Closed the way the models were trained to close: a line break, the
        # marker, a blank line.
        close = (
            newline + end + tuple(tokenizer.encode("\n\n", add_special_tokens=False))
        )
    label = tuple(getattr(tokenizer, "reasoning_label_tokens", None) or ())
    label_end = tuple(getattr(tokenizer, "think_label_end_tokens", None) or ())
    return ReasoningLimits(
        start=start,
        end=end,
        close=close,
        seeded=initial_state(tokenizer, prompt)[0] in ("reasoning", "label"),
        max_tokens=max_tokens,
        labels=(label,) if label and label_end else (),
        label_end=label_end if label else (),
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
        fingerprint: str | None = None,
    ):
        if reasoning_field not in REASONING_FIELDS:
            raise ValueError(f"unknown reasoning field {reasoning_field!r}")
        self.req = req
        self._tokenizer = tokenizer
        self.id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        # The served configuration's id (model, version, KV layout): the
        # same value for every answer the same set-up gives.
        self.fingerprint = fingerprint
        self.created = int(time.time())
        self.prompt_len = len(prompt_tokens)
        self._sent_role = False
        self._reasoning_keys = REASONING_FIELDS[reasoning_field]
        self.assembler = TextAssembler(
            tokenizer,
            prompt_tokens=prompt_tokens[req.assistant_start :],
            stop_words=req.stop,
            tools=req.tools,
            streaming=req.stream,
            route_thinking=reasoning_field != "none",
            # Without tools on offer a tool-call block is the model's text.
            tools_enabled=bool(req.tools),
        )

    def _envelope(self, kind: str) -> dict:
        return {
            "id": self.id,
            "object": kind,
            "created": self.created,
            "model": self.req.model,
            "system_fingerprint": self.fingerprint,
        }

    def _message(self, d: TextDelta) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": d.content or None}
        if d.reasoning:
            for key in self._reasoning_keys:
                msg[key] = d.reasoning
        if d.tool_calls:
            msg["tool_calls"] = d.tool_calls
        return msg

    def chunk(self, d: TextDelta) -> dict:
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
        # OpenAI's logprobs.content covers the message content: not the
        # think block, not the markers, not the stop token.
        return {
            "content": [
                logprob_entry(self._tokenizer, e) for e, content in events if content
            ]
        }

    def stream(self, events, cached: int) -> Iterator[dict]:
        """Chunks as the tokens arrive; tool-call text is held until complete."""
        for event in events:
            d = self.assembler.feed(event)
            if d.finish_reason is None and (d.empty() or self.assembler.in_tool_call):
                continue
            yield self.chunk(d)
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
