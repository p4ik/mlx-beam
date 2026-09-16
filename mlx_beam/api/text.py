"""From token events to text, reasoning and tool calls.

One ``TextAssembler`` per response: it detokenizes, runs the tokenizer's
text state machine (think and tool-call markers are stripped and routed) and
hands back deltas the chat, completions and responses formats all consume.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field

from mlx_beam._vendor.mlx_lm.generate import (
    TextStateMachine,
    make_text_state_machine,
)
from mlx_beam.engine.request import TokenEvent

logger = logging.getLogger(__name__)


@dataclass
class TextDelta:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    # "stop", "length" or "tool_calls" on the last delta, else None.
    finish_reason: str | None = None
    tokens: int = 0

    def empty(self) -> bool:
        return not (self.content or self.reasoning or self.tool_calls)


def initial_state(tokenizer, prompt_tokens: list[int]) -> str:
    """A prompt that ends inside an open think block continues as reasoning."""
    if getattr(tokenizer, "has_thinking", False):
        if tokenizer.rfind_think_start(prompt_tokens) > tokenizer.rfind_think_end(
            prompt_tokens
        ):
            return "reasoning"
    return "normal"


def stop_sequence_ids(tokenizer, stop_words: list[str] | None) -> list[tuple]:
    """EOS ids plus the encoded stop words, for the engine's stop matcher."""
    seqs = [(t,) for t in tokenizer.eos_token_ids]
    for w in stop_words or []:
        ids = tuple(tokenizer.encode(w, add_special_tokens=False))
        if ids:
            seqs.append(ids)
    return seqs


class ToolCallParser:
    """Turns the model's tool-call text into OpenAI tool_call objects."""

    def __init__(self, tool_parser, tools, streaming: bool):
        self._parser = tool_parser
        self._tools = tools
        self._streaming = streaming
        self._idx = 0

    def __call__(self, texts: Iterable[str]) -> list[dict]:
        out = []
        for text in texts:
            try:
                parsed = self._parser(text, self._tools)
            except (ValueError, json.JSONDecodeError) as e:
                # Truncated mid-generation, most often; the text is not a call.
                logger.warning("tool call not parseable (%s): %r", e, text[:200])
                continue
            for tc in parsed if isinstance(parsed, list) else [parsed]:
                tc_id = tc.pop("id", None) or f"call_{uuid.uuid4().hex[:24]}"
                tc["arguments"] = json.dumps(tc["arguments"], ensure_ascii=False)
                item = {"id": tc_id, "type": "function", "function": tc}
                if self._streaming:
                    item["index"] = self._idx
                    self._idx += 1
                out.append(item)
        return out


class TextAssembler:
    def __init__(
        self,
        tokenizer,
        *,
        prompt_tokens: list[int],
        stop_words: list[str] | None = None,
        tools: list[dict] | None = None,
        streaming: bool = False,
    ):
        self._detok = tokenizer.detokenizer
        self._sm = make_text_state_machine(tokenizer, stop_words)
        self._state = self._sm.make_state(initial_state(tokenizer, prompt_tokens))
        self._prev = self._state[0]
        self._tool_text = ""
        self._pending_tools: list[str] = []
        self._made_tool_call = False
        self._parse_tools = ToolCallParser(
            getattr(tokenizer, "tool_parser", None), tools, streaming
        )
        self.tokens = 0
        self.token_ids: list[int] = []
        self.logprobs: list[float] = []

    def feed(self, event: TokenEvent) -> TextDelta:
        """One token in, the text it releases out."""
        self.tokens += 1
        self.token_ids.append(event.token)
        self.logprobs.append(event.logprob)
        delta = TextDelta(tokens=1)

        if event.finish_reason == "stop":
            # The stop token itself is never shown.
            self._state, current = TextStateMachine.discard(self._state)
            clean = ""
        else:
            self._detok.add_token(event.token)
            if event.finish_reason == "length":
                self._detok.finalize()
            self._state, clean, current = TextStateMachine.step(
                self._state, self._detok.last_segment
            )
            if event.finish_reason == "length":
                self._state, flushed, current = TextStateMachine.flush(self._state)
                clean += flushed

        if current == "reasoning":
            delta.reasoning = clean
        elif current == "tool":
            self._tool_text += clean
        else:
            if self._prev == "tool":
                self._pending_tools.append(self._tool_text)
                self._tool_text = ""
                self._made_tool_call = True
            delta.content = clean
        self._prev = current

        if event.finish_reason is not None:
            if current == "tool" and self._tool_text:
                self._pending_tools.append(self._tool_text)
                self._tool_text = ""
                self._made_tool_call = True
            delta.finish_reason = event.finish_reason
            if event.finish_reason == "stop" and self._made_tool_call:
                delta.finish_reason = "tool_calls"

        if self._pending_tools and (current != "tool" or delta.finish_reason):
            delta.tool_calls = self._parse_tools(self._pending_tools)
            self._pending_tools = []
        return delta

    @property
    def in_tool_call(self) -> bool:
        return self._prev == "tool"
