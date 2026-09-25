"""POST /v1/completions: a text prompt or token ids, no chat template."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from mlx_beam.api.chat import (
    DEFAULTS,
    _number,
    completion_details,
    parse_sampling,
    parse_stop,
    with_xtc_specials,
)
from mlx_beam.api.defaults import RequestDefaults
from mlx_beam.api.errors import ApiError, bool_field, object_field, unsupported
from mlx_beam.api.text import TextAssembler, logprob_entry, stop_sequence_ids
from mlx_beam.engine.request import GenerationRequest, SamplingParams


@dataclass
class CompletionRequest:
    prompt: Any
    model: str
    max_tokens: int
    sampling: SamplingParams
    stream: bool = False
    stop: list[str] = field(default_factory=list)
    echo: bool = False
    # OpenAI's legacy shape: an integer, the number of alternatives per token.
    logprobs: int | None = None
    stream_usage: bool = False
    max_prompt_tokens: int | None = None
    min_response_tokens: int | None = None


def parse_completion_request(
    body: dict, default_model: str, defaults: RequestDefaults = DEFAULTS
) -> CompletionRequest:
    if not isinstance(body, dict):
        raise ApiError("the request body must be a JSON object")
    if body.get("n", 1) not in (None, 1):
        raise unsupported("n > 1", "n")
    if body.get("best_of", 1) not in (None, 1):
        raise unsupported("best_of", "best_of")
    if body.get("suffix"):
        raise unsupported("suffix (insertion)", "suffix")
    prompt = body.get("prompt")
    ok = isinstance(prompt, str) or (
        isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt)
    )
    if not ok:
        raise ApiError("prompt must be a string or a list of token ids", param="prompt")
    stream_opts = object_field(body, "stream_options")
    return CompletionRequest(
        prompt=prompt,
        model=body.get("model") or default_model,
        max_tokens=_number(
            body, "max_tokens", defaults.max_completion_tokens, 1, None, int
        ),
        sampling=parse_sampling(body, defaults),
        stream=bool_field(body, "stream"),
        stop=parse_stop(body),
        echo=bool_field(body, "echo"),
        logprobs=_number(body, "logprobs", None, 0, 20, int),
        stream_usage=bool(stream_opts.get("include_usage", False)),
        max_prompt_tokens=_number(body, "max_prompt_tokens", None, 1, None, int),
        min_response_tokens=_number(body, "min_response_tokens", None, 0, None, int),
    )


def build_prompt(tokenizer, req: CompletionRequest) -> list[int]:
    if isinstance(req.prompt, str):
        tokens = list(tokenizer.encode(req.prompt))
    else:
        tokens = list(req.prompt)
    if not tokens:
        raise ApiError("the prompt is empty", param="prompt")
    return tokens


def to_generation_request(
    tokenizer, req: CompletionRequest, defaults: RequestDefaults = DEFAULTS
) -> GenerationRequest:
    # Raw text: no chat template, no think block to budget; the reserve
    # still decides whether a prompt near the context end is served.
    return GenerationRequest(
        tokens=build_prompt(tokenizer, req),
        max_tokens=req.max_tokens,
        sampling=with_xtc_specials(tokenizer, req.sampling),
        stop_sequences=stop_sequence_ids(tokenizer, req.stop),
        top_logprobs=req.logprobs or 0,
        max_prompt_tokens=req.max_prompt_tokens,
        min_response_tokens=(
            defaults.min_response_tokens
            if req.min_response_tokens is None
            else req.min_response_tokens
        ),
    )


class CompletionResponder:
    def __init__(self, tokenizer, req: CompletionRequest, prompt_tokens: list[int]):
        self.req = req
        self.id = f"cmpl-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.prompt_tokens = prompt_tokens
        self._tokenizer = tokenizer
        self.echo_text = tokenizer.decode(prompt_tokens) if req.echo else ""
        self._offset = len(self.echo_text)
        # The raw endpoint returns what the model wrote: markers stay in
        # the text, a tool-call block is text, thinking is only counted.
        self.assembler = TextAssembler(
            tokenizer,
            prompt_tokens=prompt_tokens,
            stop_words=req.stop,
            route_thinking=False,
            tools_enabled=False,
            lead=False,
        )

    def _envelope(self) -> dict:
        return {
            "id": self.id,
            "object": "text_completion",
            "created": self.created,
            "model": self.req.model,
        }

    def _usage(self, cached: int) -> dict:
        p, c = len(self.prompt_tokens), self.assembler.tokens
        return {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": p + c,
            "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens_details": completion_details(self.assembler),
        }

    def _choice(self, text: str, finish_reason) -> dict[str, Any]:
        return {
            "index": 0,
            "text": text,
            "logprobs": self._logprobs() if self.req.logprobs is not None else None,
            "finish_reason": finish_reason,
        }

    def _logprobs(self) -> dict:
        """The legacy columns; offsets count characters of the text so far."""
        events, self.assembler.pending_events = self.assembler.pending_events, []
        out: dict[str, list] = {
            "tokens": [],
            "token_logprobs": [],
            "top_logprobs": [],
            "text_offset": [],
        }
        for e, _ in events:
            entry = logprob_entry(self._tokenizer, e)
            out["tokens"].append(entry["token"])
            out["token_logprobs"].append(entry["logprob"])
            out["top_logprobs"].append(
                {t["token"]: t["logprob"] for t in entry.get("top_logprobs", [])}
            )
            out["text_offset"].append(self._offset)
            self._offset += len(entry["token"])
        return out

    def stream(self, events, cached: int) -> Iterator[dict]:
        first = True
        for event in events:
            d = self.assembler.feed(event)
            text = d.content + d.reasoning
            if first and self.echo_text:
                text = self.echo_text + text
            first = False
            if not text and d.finish_reason is None:
                continue
            out = self._envelope()
            out["choices"] = [self._choice(text, d.finish_reason)]
            yield out
            if d.finish_reason:
                break
        if self.req.stream_usage:
            # As at OpenAI: the usage rides on a last chunk with no choices.
            out = self._envelope()
            out["choices"] = []
            out["usage"] = self._usage(cached)
            yield out

    def complete(self, events, cached: int) -> dict:
        text = self.echo_text
        finish = "stop"
        for event in events:
            d = self.assembler.feed(event)
            text += d.content + d.reasoning
            if d.finish_reason:
                finish = d.finish_reason
                break
        out = self._envelope()
        out["choices"] = [self._choice(text, finish)]
        out["usage"] = self._usage(cached)
        return out
