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

from mlx_beam._vendor.mlx_lm.generate import TextStateMachine
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


def initial_state(tokenizer, prompt_tokens: list[int]) -> tuple[str, str]:
    """A prompt that ends inside an open think block continues as reasoning;
    the second value is that open marker as the prompt carries it."""
    if getattr(tokenizer, "has_thinking", False):
        start = tokenizer.rfind_think_start(prompt_tokens)
        if start > tokenizer.rfind_think_end(prompt_tokens):
            return "reasoning", tokenizer.decode(prompt_tokens[start:])
    return "normal", ""


def stop_sequence_ids(tokenizer, stop_words: list[str] | None) -> list[tuple]:
    """EOS ids plus the encoded stop words, for the engine's stop matcher."""
    seqs = [(t,) for t in tokenizer.eos_token_ids]
    for w in stop_words or []:
        ids = tuple(tokenizer.encode(w, add_special_tokens=False))
        if ids:
            seqs.append(ids)
    return seqs


def xtc_special_ids(tokenizer) -> tuple[int, ...]:
    """Ids XTC must not cut: eos and the newline (mlx-lm's server does the same)."""
    ids = list(tokenizer.eos_token_ids)
    ids += list(tokenizer.encode("\n", add_special_tokens=False))
    return tuple(dict.fromkeys(ids))


def logprob_entry(tokenizer, event: TokenEvent) -> dict:
    """One OpenAI logprobs item: the token's text, bytes and its alternatives."""

    def item(token_id: int, logprob: float) -> dict:
        text = tokenizer.decode([token_id])
        return {"token": text, "logprob": logprob, "bytes": list(text.encode())}

    out = item(event.token, event.logprob)
    if event.top_logprobs is not None:
        out["top_logprobs"] = [item(t, lp) for t, lp in event.top_logprobs]
    return out


def make_state_machine(tokenizer, stop_words) -> TextStateMachine:
    """Think and tool markers switch state; a stop word ends the stream.

    The engine matches stop words on token ids, which misses a word whose
    tokenization differs in context; the text match catches those and the
    stream ends (state None) instead of merely hiding the word."""
    transitions: dict = {}
    thinking = getattr(tokenizer, "has_thinking", False)
    if thinking:
        transitions.setdefault("normal", []).append(
            (tokenizer.think_start, "reasoning")
        )
        transitions["reasoning"] = [(tokenizer.think_end, "normal")]
    if getattr(tokenizer, "has_tool_calling", False):
        transitions.setdefault("normal", []).append((tokenizer.tool_call_start, "tool"))
        if thinking:
            transitions["reasoning"].append((tokenizer.tool_call_start, "tool"))
        transitions["tool"] = (
            [(tokenizer.tool_call_end, "normal")] if tokenizer.tool_call_end else []
        )
    for w in getattr(tokenizer, "structural_markers", None) or ():
        transitions.setdefault("normal", []).append((w, "normal"))
    for state in set(transitions) | {"normal"}:
        for w in stop_words or []:
            transitions.setdefault(state, []).append((w, None))
    return TextStateMachine(transitions or None)


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
        route_thinking: bool = True,
    ):
        self._detok = tokenizer.detokenizer
        # The automaton always runs, so reasoning is counted in every mode;
        # routing off puts the markers back into the text where they were cut.
        self._sm = make_state_machine(tokenizer, stop_words)
        start, seeded = initial_state(tokenizer, prompt_tokens)
        self._state = self._sm.make_state(start)
        self._prev = self._state[0]
        self._route = route_thinking
        self._think_start = getattr(tokenizer, "think_start", None) or ""
        self._think_end = getattr(tokenizer, "think_end", None) or ""
        # A think block the prompt opened: the client sees it once, up front.
        self._lead = seeded if not route_thinking else ""
        self._tool_text = ""
        self._pending_tools: list[str] = []
        self._made_tool_call = False
        self._parse_tools = ToolCallParser(
            getattr(tokenizer, "tool_parser", None), tools, streaming
        )
        self.tokens = 0
        self.token_ids: list[int] = []
        self.logprobs: list[float] = []
        # Events since the last chunk went out; the responder drains them.
        self.pending_events: list[TokenEvent] = []
        # True once a text-level stop word ended the stream early.
        self.stopped = False
        self.reasoning_tokens = 0

    def feed(self, event: TokenEvent) -> TextDelta:
        """One token in, the text it releases out."""
        if self.stopped:
            return TextDelta()
        self.tokens += 1
        self.token_ids.append(event.token)
        self.logprobs.append(event.logprob)
        self.pending_events.append(event)
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
            if event.finish_reason == "length" and current is not None:
                # A stop word already matched leaves its tail in the buffer;
                # that tail is not output.
                self._state, flushed, current = TextStateMachine.flush(self._state)
                clean += flushed

        if current is None:
            # A stop word matched in the text: what came before it is the
            # last of the output, whatever state we were in.
            if self._prev == "reasoning" and self._route:
                delta.reasoning = clean
            elif self._prev != "tool":
                delta.content = self._take_lead() + clean
            self.stopped = True
            self._prev = "normal"
            delta.finish_reason = "stop"
            if self._made_tool_call or self._tool_text or self._pending_tools:
                if self._tool_text:
                    self._pending_tools.append(self._tool_text)
                    self._tool_text = ""
                delta.tool_calls = self._parse_tools(self._pending_tools)
                self._pending_tools = []
                delta.finish_reason = "tool_calls"
            return delta
        if current == "reasoning":
            self.reasoning_tokens += 1
            if self._route:
                delta.reasoning = clean
            else:
                opened = self._think_start if self._prev != "reasoning" else ""
                delta.content = (self._take_lead() or opened) + clean
        elif current == "tool":
            self._tool_text += clean
        else:
            if self._prev == "tool":
                self._pending_tools.append(self._tool_text)
                self._tool_text = ""
                self._made_tool_call = True
            closed = self._think_end if self._prev == "reasoning" else ""
            delta.content = (closed if not self._route else "") + clean
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

    def _take_lead(self) -> str:
        lead, self._lead = self._lead, ""
        return lead

    @property
    def in_tool_call(self) -> bool:
        return self._prev == "tool"
