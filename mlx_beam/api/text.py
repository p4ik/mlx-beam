"""From token events to text, reasoning and tool calls.

One ``TextAssembler`` per response: it detokenizes, runs the tokenizer's
text state machine (think and tool-call markers are stripped and routed) and
hands back deltas the chat, completions and responses formats all consume.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field

from mlx_beam._vendor.mlx_lm.generate import TextStateMachine
from mlx_beam.api import repair
from mlx_beam.engine.request import TokenEvent

logger = logging.getLogger(__name__)


@dataclass
class TextDelta:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    # "stop", "length" or "tool_calls" on the last delta, else None.
    finish_reason: str | None = None

    def empty(self) -> bool:
        return not (self.content or self.reasoning or self.tool_calls)


def route_label(tokenizer, label: str) -> str:
    """Where the text after a label goes: reasoning, normal (the answer) or
    tool. Gemma's channels are all reasoning; Harmony's `analysis` is,
    `final` is the answer, a `commentary` with a recipient (or a recipient
    before the channel) a tool call; Muse's `self` is the reasoning, `user`
    the answer, any other recipient a tool."""
    family = getattr(tokenizer, "think_family", None)
    label = label.strip()
    if family == "harmony":
        if "to=" in label or label.startswith("functions."):
            return "tool"
        return "reasoning" if label.startswith("analysis") else "normal"
    if family == "muse":
        if label == "self":
            return "reasoning"
        return "normal" if label in ("", "user", "assistant") else "tool"
    return "reasoning"


def initial_state(tokenizer, prompt_tokens: list[int]) -> tuple[str, str]:
    """A prompt that ends inside an open think block continues as reasoning -
    or inside the opener's label, when the label's end is still to come;
    the second value is that open marker as the prompt carries it. A label
    already complete is routed: a prompt ending in the answer's opener
    (thinking switched off) starts in the answer."""
    if getattr(tokenizer, "has_thinking", False):
        start = tokenizer.rfind_think_start(prompt_tokens)
        if start > tokenizer.rfind_think_end(prompt_tokens):
            opened = tokenizer.decode(prompt_tokens[start:])
            label_end = getattr(tokenizer, "think_label_end", None)
            if label_end and label_end not in opened:
                return "label", opened
            if label_end:
                label = opened[len(tokenizer.think_start) :].split(label_end, 1)[0]
                if route_label(tokenizer, label) != "reasoning":
                    return "normal", ""
            return "reasoning", opened
    frame = getattr(tokenizer, "frame_start", None)
    if frame:
        # The generation prompt of these families ends with the message's
        # frame: what comes first is the recipient or the channel, not
        # text - unless the prompt already opened the answer (thinking
        # switched off appends the answer's opener).
        opener = tuple(getattr(tokenizer, "answer_opener_tokens", None) or ())
        if opener and tuple(prompt_tokens[-len(opener) :]) == opener:
            return "normal", ""
        return "frame", ""
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


def holds_incomplete_bytes(detokenizer) -> bool:
    """Whether the text the detokenizer holds back ends in a cut UTF-8
    sequence. mlx-lm's BPE and SPM detokenizers keep bytes until they form
    a character; only those bytes tell a fragment from a real U+FFFD. With
    no bytes to look at (the naive detokenizer) nothing is dropped: a
    spare U+FFFD is cosmetic, a deleted character is not."""
    held = getattr(detokenizer, "_unflushed", None)
    if isinstance(held, str):  # BPE: byte-level characters
        try:
            from mlx_beam._vendor.mlx_lm.tokenizer_utils import _byte_decoder

            held = bytes(_byte_decoder()[c] for c in held)
        except KeyError:
            return False
    if not isinstance(held, bytes):
        return False
    try:
        held.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


# A SentencePiece byte token names its byte in hex (`<0xE2>`); the
# vendored SPM detokenizer reads it the same way.
_SPM_BYTE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def token_bytes(tokenizer, token_id: int, text: str) -> list[int]:
    """The token's bytes: the raw ones for a byte-level token (a piece of
    a character decodes to U+FFFD, whose bytes are not the token's) - the
    hex of a SentencePiece byte token, the byte alphabet of a BPE one -
    the text's otherwise."""
    if "\ufffd" in text and hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            from mlx_beam._vendor.mlx_lm.tokenizer_utils import _byte_decoder

            (piece,) = tokenizer.convert_ids_to_tokens([token_id])
            spm = _SPM_BYTE.match(piece)
            if spm:
                return [int(spm.group(1), 16)]
            decoder = _byte_decoder()
            return [decoder[c] for c in piece]
        except (KeyError, ValueError, TypeError):
            pass
    return list(text.encode())


def logprob_entry(tokenizer, event: TokenEvent) -> dict:
    """One OpenAI logprobs item: the token's text, bytes and its alternatives."""

    def item(token_id: int, logprob: float) -> dict:
        text = tokenizer.decode([token_id])
        return {
            "token": text,
            "logprob": logprob,
            "bytes": token_bytes(tokenizer, token_id, text),
        }

    out = item(event.token, event.logprob)
    if event.top_logprobs is not None:
        out["top_logprobs"] = [item(t, lp) for t, lp in event.top_logprobs]
    return out


def make_state_machine(tokenizer, stop_words, tools: bool = True) -> TextStateMachine:
    """Think and tool markers switch state; a stop word ends the stream.

    The engine matches stop words on token ids, which misses a word whose
    tokenization differs in context; the text match catches those and the
    stream ends (state None) instead of merely hiding the word. With
    ``tools`` off the tool markers are ordinary text."""
    transitions: dict = {}
    thinking = getattr(tokenizer, "has_thinking", False)
    label_end = getattr(tokenizer, "think_label_end", None)
    via_label = getattr(tokenizer, "tool_call_via_label", False)
    # A family whose opener is plain text (` to=`) gets a `frame` state:
    # entered by the marker that begins a message, left by the first
    # structural marker after it. The textual opener counts there only -
    # in the answer, `send(msg, to=addr)` is text.
    frame = getattr(tokenizer, "frame_start", None)
    if thinking and label_end:
        # A labelled opener (Gemma 4's channel, Harmony's channel, Muse's
        # recipient): the label runs to its end and is part of the marker,
        # whatever tokens it comes as. The assembler finds that end itself
        # - the automaton reports one text per segment and only the state
        # after it, so an end it matched could not be told apart from the
        # reasoning text behind it - and routes by what the label says.
        openers = getattr(tokenizer, "think_openers", None) or (tokenizer.think_start,)
        for opener in openers:
            textual = not opener.startswith("<")
            if frame:
                transitions.setdefault("frame", []).append((opener, "label"))
                if textual and opener != opener.lstrip():
                    # The detokenizer drops the space a sequence starts
                    # with: at the frame, ` to=` arrives as `to=`.
                    transitions["frame"].append((opener.lstrip(), "label"))
            if not (frame and textual):
                transitions.setdefault("normal", []).append((opener, "label"))
        transitions["label"] = [(tokenizer.think_end, "normal")]
        transitions["reasoning"] = [(tokenizer.think_end, "normal")]
    elif thinking:
        transitions.setdefault("normal", []).append(
            (tokenizer.think_start, "reasoning")
        )
        transitions["reasoning"] = [(tokenizer.think_end, "normal")]
    if tools and getattr(tokenizer, "has_tool_calling", False):
        if not via_label:
            transitions.setdefault("normal", []).append(
                (tokenizer.tool_call_start, "tool")
            )
            if thinking:
                transitions["reasoning"].append((tokenizer.tool_call_start, "tool"))
        transitions["tool"] = (
            [(tokenizer.tool_call_end, "normal")] if tokenizer.tool_call_end else []
        )
    for w in getattr(tokenizer, "structural_markers", None) or ():
        if frame and w == frame:
            transitions.setdefault("normal", []).append((w, "frame"))
            transitions.setdefault("frame", []).append((w, "frame"))
        else:
            transitions.setdefault("normal", []).append((w, "normal"))
            if frame:
                transitions.setdefault("frame", []).append((w, "normal"))
    for state in set(transitions) | {"normal"}:
        for w in stop_words or []:
            transitions.setdefault(state, []).append((w, None))
    return TextStateMachine(transitions or None)


# Tool-call dialects whose call text is JSON throughout (what the text
# repair may mend); the XML-like and Python-literal dialects keep their
# values outside JSON quotes.
JSON_DIALECTS = frozenset(
    {"json_tools", "mistral", "harmony", "gemma4", "function_gemma", "kimi_k3"}
)


class ToolCallParser:
    """Turns the model's tool-call text into OpenAI tool_call objects.

    What does not parse is handed back as text, markers and all: the client
    sees what the model said instead of a call that never existed. A text
    that holds several calls without an end marker (Mistral) is cut at the
    longest prefix that parses, so the calls before a cut are kept."""

    def __init__(
        self, tool_parser, tools, streaming: bool, start="", end="", dialect=None
    ):
        self._parser = tool_parser
        self._tools = tools
        self._streaming = streaming
        self._start = start or ""
        self._end = end or ""
        self._idx = 0
        # The text repair mends JSON; a dialect whose call text is XML or
        # Python (values not in JSON quotes) would have its values changed
        # by it, so the rung runs for the JSON dialects only.
        self._json = dialect in JSON_DIALECTS

    def __call__(
        self, blocks: Iterable[tuple[str, bool]]
    ) -> tuple[list[dict], list[str]]:
        """``blocks`` are (text, closed) pairs; an unclosed block was cut off
        and gets no end marker back."""
        calls: list[dict] = []
        unparsed: list[str] = []
        for block in blocks:
            text, closed = block[0], block[1]
            opener = block[2] if len(block) > 2 else None
            parsed = self._parse(text, closed)
            if parsed is None and self._start and not self._end:
                # Try the text up to each later start marker, longest first.
                # A marker inside a JSON string is never a cut: the prefix
                # ending inside the string does not parse.
                for pos in self._marker_positions(text):
                    head = self._parse(text[:pos], closed)
                    if head is not None:
                        calls.extend(head)
                        unparsed.append(self._raw(text[pos:], closed, opener))
                        break
                else:
                    unparsed.append(self._raw(text, closed, opener))
                    repair.STATS["failed"] += 1
                continue
            if parsed is None:
                unparsed.append(self._raw(text, closed, opener))
                repair.STATS["failed"] += 1
            else:
                calls.extend(parsed)
        return calls, unparsed

    def _marker_positions(self, text: str) -> list[int]:
        positions = []
        pos = text.find(self._start)
        while pos > 0:
            positions.append(pos)
            pos = text.find(self._start, pos + 1)
        return positions[::-1]

    def _raw(self, text: str, closed: bool, opener: str | None = None) -> str:
        # The block's own marker is what the automaton removed; markers
        # inside the text are the model's and stay where they are. A label
        # family hands back the opener the model used (`<|channel|>` or the
        # recipient's ` to=`), not the parser's default.
        start = opener if opener is not None else self._start
        if text.startswith(start):
            return text + (self._end if closed else "")
        return start + text + (self._end if closed else "")

    def _read(self, text: str) -> list[dict]:
        parsed = self._parser(text, self._tools)
        return [dict(tc) for tc in (parsed if isinstance(parsed, list) else [parsed])]

    def _parse(self, text: str, closed: bool = True) -> list[dict] | None:
        """The repair ladder: the parser's own reading, validated against
        the tool's schema; a reading of the mended text when the parser
        refused; the values coerced to the declared types when the schema
        objected. Every rung is validated again, what a rung did is the
        call's `repair_actions`, and a call no rung makes valid is None. A
        block the stream cut off (`closed` false) gets no mending: closing
        its open string and brackets would make a call with truncated
        values look valid."""
        actions: list[str] = []
        try:
            calls = self._read(text)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            # TypeError: a literal the parser evaluated (a set, bytes) that
            # JSON cannot carry; KeyError: a call without arguments.
            mended = repair.repair_json_text(text) if closed and self._json else text
            if mended == text:
                logger.warning("tool call not parseable (%s): %r", e, text[:200])
                return None
            try:
                calls = self._read(mended)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e2:
                logger.warning("tool call not parseable (%s): %r", e2, text[:200])
                return None
            actions.append("json repaired")
        out = []
        for tc in calls:
            if "arguments" not in tc or "name" not in tc:
                return None
            args = tc["arguments"]
            own = list(actions)
            if isinstance(args, str):
                # The wire format's own shape (arguments as a JSON string),
                # which Granite writes inside its block: decoded, not refused.
                try:
                    decoded = json.loads(args)
                except ValueError:
                    decoded = None
                if isinstance(decoded, dict):
                    args = decoded
                    own.append("arguments decoded")
            schema = repair.schema_for(self._tools, tc["name"])
            if schema is None and repair.declared_names(self._tools):
                # A tool the request never declared: the model's text, not a
                # call the client could run.
                logger.warning("tool call to an undeclared tool %r", tc["name"])
                return None
            if schema is not None:
                problems = repair.validate(args, schema)
                if problems and isinstance(args, dict):
                    args, coerced = repair.coerce(args, schema)
                    own += coerced
                    problems = repair.validate(args, schema)
                if problems:
                    logger.warning(
                        "tool call %s fails its schema (%s): %r",
                        tc["name"],
                        "; ".join(problems),
                        text[:200],
                    )
                    return None
            tc_id = tc.pop("id", None) or f"call_{uuid.uuid4().hex[:24]}"
            try:
                tc["arguments"] = json.dumps(args, ensure_ascii=False)
            except TypeError as e:
                # A literal the parser evaluated (a set, bytes) JSON cannot carry.
                logger.warning("tool call not parseable (%s): %r", e, text[:200])
                return None
            item = {"id": tc_id, "type": "function", "function": tc}
            if own:
                item["repair_actions"] = own
            out.append(item)
        rung = (
            "coerced"
            if any(len(i.get("repair_actions", ())) > len(actions) for i in out)
            else ("repaired" if actions else "strict")
        )
        repair.STATS[rung] += 1
        if self._streaming:
            for item in out:
                item["index"] = self._idx
                self._idx += 1
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
        tools_enabled: bool = True,
        lead: bool = True,
        raw: bool = False,
    ):
        self._tokenizer = tokenizer
        # The raw endpoint: every marker the automaton cuts goes back into
        # the text where it was, structural ones included.
        self._raw = raw
        self._detok = tokenizer.detokenizer
        # The automaton always runs, so reasoning is counted in every mode;
        # routing off puts the markers back into the text where they were cut.
        self._sm = make_state_machine(tokenizer, stop_words, tools=tools_enabled)
        start, seeded = initial_state(tokenizer, prompt_tokens)
        self._state = self._sm.make_state(start)
        self._prev = self._state[0]
        self._route = route_thinking
        self._think_start = getattr(tokenizer, "think_start", None) or ""
        self._think_end = getattr(tokenizer, "think_end", None) or ""
        self._label_end = getattr(tokenizer, "think_label_end", None) or ""
        # The opener's label so far, until its line end shows up, and how
        # many tokens it took (they count as reasoning only if it opens one).
        self._label = seeded[len(self._think_start) :] if start == "label" else ""
        self._label_tokens = 0
        # A think block the template opened: with routing off the client
        # sees its opener once, up front - not for a raw prompt the client
        # wrote itself (completions), where the opener is already theirs.
        self._lead = seeded if lead and not route_thinking else ""
        self._tool_text = ""
        # (text, closed): a block still open when the stream ended was cut.
        self._pending_tools: list[tuple[str, bool, str | None]] = []
        # The opener of a label that turned out a tool call (Harmony's
        # channel or recipient): what an unparsable call is handed back with.
        self._label_opener: str | None = None
        self._tool_opener: str | None = None
        self._made_tool_call = False
        # A fragment the detokenizer held when the forced close began.
        self._fragment = False
        self._fragment_checked = False
        # The engine's stop matcher holds the eos ids and the stop words
        # alike; only an eos token is dropped unseen (feed).
        self._eos = set(getattr(tokenizer, "eos_token_ids", None) or ())
        # Byte tokens of a character still forming: content or not is
        # decided by the token that completes it (logprobs).
        self._held: list[TokenEvent] = []
        self._parse_tools = ToolCallParser(
            getattr(tokenizer, "tool_parser", None),
            tools,
            streaming,
            start=getattr(tokenizer, "tool_call_start", None),
            end=getattr(tokenizer, "tool_call_end", None),
            dialect=getattr(tokenizer, "tool_parser_type", None),
        )
        self.tokens = 0
        # Events since the last chunk went out, each with whether its text
        # went to the content; the responder drains them for logprobs.
        self.pending_events: list[tuple[TokenEvent, bool]] = []
        # True once a text-level stop word ended the stream early, and
        # which word it was (the longest that fits the text's tail).
        self.stopped = False
        self.stopped_word: str | None = None
        self._stop_words = sorted(stop_words or [], key=len, reverse=True)
        self._tail = ""
        self._pieces: list = []
        self.reasoning_tokens = 0
        # From the engine's last event: which limit cut what.
        self.thinking_truncated = False
        self.response_truncated = False

    def feed(self, event: TokenEvent) -> TextDelta:
        """One token in, the text it releases out."""
        if self.stopped:
            return TextDelta()
        self.tokens += 1
        self.thinking_truncated |= event.thinking_truncated
        self.response_truncated |= event.response_truncated
        delta = TextDelta()

        if event.finish_reason == "stop":
            # An eos token is never shown. The last token of a stop
            # sequence is: it goes through the detokenizer so the automaton
            # sees the whole word and cuts there - held back as a prefix,
            # its start would otherwise come out as text. What the
            # detokenizer and the automaton still hold is text - a marker
            # that never completed is what the model said.
            if event.token not in self._eos:
                self._detok.add_token(event.token)
            self._detok.finalize()
            segment = self._detok.last_segment
            self._tail = self._tail[-self._tail_len() :] + segment
            clean, before, after, markers, current = self._step(segment)
            if current is not None:
                clean, before, after = self._flush(clean, before, after, markers)
        else:
            if event.forced and not self._fragment_checked:
                # The budget may cut a byte-level tokenizer mid-character:
                # the detokenizer then flushes the fragment as U+FFFD with
                # the first forced token. Only a real fragment is dropped.
                self._fragment = holds_incomplete_bytes(self._detok)
            self._fragment_checked = event.forced
            self._detok.add_token(event.token)
            if event.finish_reason == "length":
                self._detok.finalize()
            segment = self._detok.last_segment
            self._tail = self._tail[-self._tail_len() :] + segment
            clean, before, after, markers, current = self._step(segment)
            if event.forced and self._fragment:
                clean = clean.replace("\ufffd", "", 1)
                before = before.replace("\ufffd", "", 1)
                self._fragment = False
            if event.finish_reason == "length" and current is not None:
                # A stop word already matched leaves its tail in the buffer;
                # that tail is not output.
                clean, before, after = self._flush(clean, before, after, markers)

        # What the automaton said, and the parts of this event's text with
        # the state each was written in - for the logprobs' verdict; the
        # label branch splits its text by hand (the label's end is no
        # marker of the automaton's, it is read from the text).
        automaton = current
        parts: list[tuple[str | None, str]] | None = None
        if current is None:
            # A stop word matched in the text: what came before it is the
            # last of the output, whatever state we were in - except a
            # label, which is the opener's and never text.
            # The match ended inside the last segment; the tail keeps the
            # longest stop word's worth of text before it, so the word that
            # matched is in there whole (the longest that is, when nested).
            self.stopped_word = next(
                (w for w in self._stop_words if w in self._tail), None
            )
            # A transition before the stop word in the same segment: the
            # text before it is the previous state's, the rest the state
            # the stop word was matched in (a block that closed, then text).
            head, tail = (before, after) if markers[:-1] else (clean, "")
            if self._prev == "reasoning" and self._route:
                delta.reasoning = head
            elif self._prev == "label" and self._route:
                # The segment may hold the label's end and reasoning text
                # before the stop word; only the label is dropped.
                delta.reasoning = self._after_label(head) or ""
            elif self._prev != "tool":
                delta.content = self._take_lead() + head
            else:
                # The call's last text before the stop word: it belongs to
                # the cut-off block, which goes back as text unrepaired.
                self._tool_text += head
            if tail:
                if self._prev == "tool":
                    self._pending_tools.append(
                        (self._tool_text, True, self._tool_opener)
                    )
                    self._tool_text = ""
                elif self._prev in ("reasoning", "label") and not self._route:
                    tail = self._take_lead() + self._think_end + tail
                delta.content += tail
            self.stopped = True
            self._book(event, "", bool(delta.content), own=False)
            self._prev = "normal"
            delta.finish_reason = "stop"
            if self._tool_text:
                self._pending_tools.append((self._tool_text, False, self._tool_opener))
                self._tool_text = ""
            self._flush_tools(delta)
            if self._made_tool_call:
                delta.finish_reason = "tool_calls"
            return delta
        if current == "label":
            # The label is the opener's, not text; with routing off the
            # client gets the block as the model wrote it, label included.
            self.reasoning_tokens += 1
            shown = ""
            if not self._route:
                # With routing off the block goes out as the model wrote
                # it, label and opener included - unless the label turns
                # out a tool call, which is parsed, not shown.
                opened = self._think_start if self._prev != "label" else ""
                shown = (self._take_lead() or opened) + clean
                delta.content = shown
            if self._prev != "label":
                self._label = ""
                self._label_tokens = 0
                # Which opener it was (the segment carried it): a call that
                # does not parse is handed back behind the same one.
                openers = getattr(self._tokenizer, "think_openers", None) or ()
                self._label_opener = next((o for o in openers if o in segment), None)
            self._label_tokens += 1
            label = self._label
            rest = self._after_label(clean)
            parts = [("label", clean)]
            if rest is not None:
                # The label's end may arrive in one segment with the label,
                # or with the first text behind it: that text goes where
                # the label says - reasoning, the answer, or a tool call
                # (the header stays with the call's text for the parser).
                label = (label + clean).split(self._label_end, 1)[0]
                target = route_label(self._tokenizer, label)
                if target == "tool" and "tool" not in self._state[2]:
                    target = "normal"  # tools off: the model's text as is
                self._state = self._enter(target)
                current = target
                parts = [("label", clean[: len(clean) - len(rest)]), (target, rest)]
                if target == "reasoning":
                    if self._route:
                        delta.reasoning = rest
                elif target == "tool":
                    self.reasoning_tokens -= self._label_tokens
                    self._tool_text = label + self._label_end + rest
                    self._tool_opener = self._label_opener
                    if shown:
                        delta.content = delta.content[: -len(shown)]
                else:
                    # The label opened the answer: its tokens were no reasoning.
                    self.reasoning_tokens -= self._label_tokens
                    if not shown:  # with routing off `clean` already holds it
                        delta.content += rest
        elif current == "reasoning":
            self.reasoning_tokens += 1
            entered = self._prev not in ("reasoning", "label")
            head = before if entered and markers else ""
            body = after if entered and markers else clean
            if self._route:
                delta.content = head
                delta.reasoning = body
            else:
                opened = self._think_start if self._prev != "reasoning" else ""
                delta.content = head + (self._take_lead() or opened) + body
        elif current == "tool":
            if self._prev != "tool" and markers:
                # What the opener released was written before it: the
                # block's when a think block ended in the call.
                if self._prev == "reasoning" and self._route:
                    delta.reasoning = before
                else:
                    delta.content = before
                self._tool_text += after
            else:
                self._tool_text += clean
        else:
            head, body = (before, after) if markers else ("", clean)
            if self._prev == "tool":
                # The end marker may sit inside a text segment (ATEM's is
                # plain text): what the automaton released with it is the
                # call's last text, not the answer's - and what follows the
                # marker is the answer's, not the call's.
                self._pending_tools.append(
                    (self._tool_text + head, True, self._tool_opener)
                )
                self._tool_text = ""
                head = ""
            elif self._prev in ("reasoning", "label"):
                # Text held back before the end marker (a `<` that could
                # have begun it) is the block's, released with the marker.
                if self._route:
                    delta.reasoning = (
                        self._after_label(head) or "" if self._prev == "label" else head
                    )
                    head = ""
                else:
                    # A block the prompt opened may close on the first token.
                    head = head + self._take_lead() + self._think_end
            elif self._raw and markers:
                # The raw endpoint: a structural marker (the frame's, a
                # message's) is what the model wrote and goes back where it
                # was cut - the automaton only tracked it.
                head, body = "", self._verbatim()
            self._label = ""
            delta.content = head + body
        # A marker token releases no text; the stop token none either -
        # what its arrival flushed belongs to the tokens held before it.
        # The verdicts follow the text's: what came out before a marker
        # was the old state's, written by earlier tokens (held by the
        # automaton or the detokenizer, or booked already), what came
        # after it the new one's. The marker's own token is content when
        # the marker went back into the content - then the held tokens
        # that spelled it (a textual marker) are too.
        if markers:
            shown = self._markers_shown(markers, automaton)
            held = self._shown(self._prev, before) or (shown and not before)
            own = parts if parts is not None else [(current, after)]
            own_text = shown or any(self._shown(st, tx) for st, tx in own)
        else:
            own = parts if parts is not None else [(current, clean)]
            held = own_text = any(self._shown(st, tx) for st, tx in own)
        self._book(
            event,
            segment,
            own_text,
            held=held,
            released=bool(clean),
            own=False if event.finish_reason == "stop" else None,
        )
        self._prev = current

        if event.finish_reason is not None:
            if current == "tool" and self._tool_text:
                self._pending_tools.append((self._tool_text, False, self._tool_opener))
                self._tool_text = ""
            delta.finish_reason = event.finish_reason

        if self._pending_tools and (current != "tool" or delta.finish_reason):
            self._flush_tools(delta)
        if event.finish_reason == "stop" and self._made_tool_call:
            # Only calls that really parsed make this a tool-call turn; a
            # cut-off one keeps the engine's reason and its text.
            delta.finish_reason = "tool_calls"
        return delta

    def _flush_tools(self, delta: TextDelta) -> None:
        calls, unparsed = self._parse_tools(self._pending_tools)
        self._pending_tools = []
        delta.tool_calls += calls
        self._made_tool_call |= bool(calls)
        if unparsed:
            delta.content += "".join(unparsed)

    def _shown(self, state: str | None, text: str) -> bool:
        """Text released in `state` went to the content: the answer's
        always, a block's and its label when the reasoning is not routed
        apart (the client gets the block as written); a tool call's never."""
        return bool(text) and (
            state == "normal" or (not self._route and state in ("reasoning", "label"))
        )

    def _markers_shown(self, markers: list[str], current: str | None) -> bool:
        """The segment's markers went back into the content: every one on
        the raw endpoint; with the reasoning not routed apart, the block's
        opener on the way in and its end on the way out - judged by the
        transition, not the text: Harmony's `<|end|>` also ends a message,
        and there it is cut. A stop word is cut everywhere, a tool call's
        marker stays with the call."""
        kept = [m for m in markers if m not in self._stop_words]
        if self._raw:
            return bool(kept)
        if self._route or not kept:
            return False
        block = ("reasoning", "label")
        entering = current in block and self._prev not in block
        leaving = current == "normal" and self._prev in block
        return entering or leaving

    def _book(
        self,
        event: TokenEvent,
        segment: str,
        content: bool,
        held: bool | None = None,
        released: bool = True,
        own: bool | None = None,
    ) -> None:
        """The event into `pending_events` with whether its text went to the
        content. A token whose text is still held back - by the detokenizer
        (a cut UTF-8 sequence, a lone space it waits with) or by the
        automaton (the start of what may be a marker or a stop word) - is
        not decided yet: it is booked with the token that releases the
        text, under `held` (the verdict for what was released before a
        marker, the state it was written in). A marker releases its segment
        and the automaton swallows it: decided at once, not content. `own`
        overrides the verdict for the event itself (the stop token: never
        content, whatever it released)."""
        buffered = bool(self._state[3]) if self._state is not None else False
        if event.finish_reason is None and (not segment or (not released and buffered)):
            self._held.append(event)
            return
        for event_held in self._held:
            self.pending_events.append((event_held, content if held is None else held))
        self._held = []
        self.pending_events.append((event, content if own is None else own))

    def _step(self, segment: str):
        """The automaton's step, with the released text kept apart: `clean`
        is all of it, `before` what came out while still in the state the
        segment began in (held back before a marker), `after` the rest;
        `markers` the control sequences cut on the way. The pieces are kept
        for the raw endpoint, which gives structural markers back."""
        self._state, pieces, current = TextStateMachine.step_pieces(
            self._state, segment
        )
        self._pieces = pieces
        before, after, markers = [], [], []
        for text, kind, _ in pieces:
            if kind == "marker":
                markers.append(text)
            elif markers:
                after.append(text)
            else:
                before.append(text)
        clean = "".join(before) + "".join(after)
        return clean, "".join(before), "".join(after), markers, current

    def _flush(self, clean: str, before: str, after: str, markers: list):
        """The automaton's buffer when the stream ends is text the model
        wrote (a marker that never completed): released in the state the
        segment ended in - after its marker when it carried one - and kept
        with the pieces, which the raw endpoint gives back whole."""
        self._state, flushed, current = TextStateMachine.flush(self._state)
        if flushed:
            self._pieces.append((flushed, "text", current))
            clean += flushed
            if after or markers:
                after += flushed
            else:
                before += flushed
        return clean, before, after

    def _verbatim(self) -> str:
        """The segment as the model wrote it: text and markers in order."""
        return "".join(text for text, _, _ in self._pieces)

    def _tail_len(self) -> int:
        return max((len(w) for w in self._stop_words), default=0)

    def _take_lead(self) -> str:
        lead, self._lead = self._lead, ""
        return lead

    def _after_label(self, clean: str) -> str | None:
        """Adds ``clean`` to the label; the text behind the label's line end
        once it is there (the label is dropped), None while it is not."""
        self._label += clean
        if self._label_end not in self._label:
            return None
        rest = self._label.split(self._label_end, 1)[1]
        self._label = ""
        return rest

    def _enter(self, state: str):
        """The automaton moved to ``state`` by the assembler's own decision:
        same buffer, trie at its root - a partial match held in the buffer
        is scanned again from there on the next step."""
        _, _, states, buf = self._state
        return (state, states[state][0], states, buf)
