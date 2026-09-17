"""POST /v1/completions: a text prompt or token ids, no chat template."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from mlx_beam.api.chat import DEFAULT_MAX_TOKENS, _number, parse_sampling, parse_stop
from mlx_beam.api.errors import ApiError, unsupported
from mlx_beam.api.text import TextAssembler, stop_sequence_ids
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
    stream_usage: bool = False


def parse_completion_request(body: dict, default_model: str) -> CompletionRequest:
    if not isinstance(body, dict):
        raise ApiError("the request body must be a JSON object")
    if body.get("n", 1) not in (None, 1):
        raise unsupported("n > 1", "n")
    if body.get("seed") is not None:
        raise unsupported("seed", "seed")
    if body.get("best_of", 1) not in (None, 1):
        raise unsupported("best_of", "best_of")
    prompt = body.get("prompt")
    ok = isinstance(prompt, str) or (
        isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt)
    )
    if not ok:
        raise ApiError("prompt must be a string or a list of token ids", param="prompt")
    stream_opts = body.get("stream_options") or {}
    return CompletionRequest(
        prompt=prompt,
        model=body.get("model") or default_model,
        max_tokens=_number(body, "max_tokens", DEFAULT_MAX_TOKENS, 1, None, int),
        sampling=parse_sampling(body),
        stream=bool(body.get("stream", False)),
        stop=parse_stop(body),
        echo=bool(body.get("echo", False)),
        stream_usage=bool(stream_opts.get("include_usage", False)),
    )


def build_prompt(tokenizer, req: CompletionRequest) -> list[int]:
    if isinstance(req.prompt, str):
        tokens = list(tokenizer.encode(req.prompt))
    else:
        tokens = list(req.prompt)
    if not tokens:
        raise ApiError("the prompt is empty", param="prompt")
    return tokens


def to_generation_request(tokenizer, req: CompletionRequest) -> GenerationRequest:
    return GenerationRequest(
        tokens=build_prompt(tokenizer, req),
        max_tokens=req.max_tokens,
        sampling=req.sampling,
        stop_sequences=stop_sequence_ids(tokenizer, req.stop),
    )


class CompletionResponder:
    def __init__(self, tokenizer, req: CompletionRequest, prompt_tokens: list[int]):
        self.req = req
        self.id = f"cmpl-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.prompt_tokens = prompt_tokens
        self.echo_text = tokenizer.decode(prompt_tokens) if req.echo else ""
        self.assembler = TextAssembler(
            tokenizer, prompt_tokens=prompt_tokens, stop_words=req.stop
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
        }

    def _choice(self, text: str, finish_reason) -> dict[str, Any]:
        return {
            "index": 0,
            "text": text,
            "logprobs": None,
            "finish_reason": finish_reason,
        }

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
            if d.finish_reason and self.req.stream_usage:
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
        out = self._envelope()
        out["choices"] = [self._choice(text, finish)]
        out["usage"] = self._usage(cached)
        return out
