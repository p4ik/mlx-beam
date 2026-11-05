"""The reasoning budget: how many tokens the think block may take, and the
close that is forced when it runs out.

The generator samples one step ahead of what Python has seen, so a decision
taken on the token just observed reaches the logits two tokens later. The
tracker arms early enough for that, and the processor guards on the device
against a block the model closed by itself in between.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mlx.core as mx


class ContextTooLong(ValueError):
    """Prompt plus the room the answer needs do not fit the model's context."""


@dataclass(frozen=True)
class ReasoningLimits:
    """The think markers as token ids and the budget for this request."""

    start: tuple[int, ...]
    end: tuple[int, ...]
    # The sequence forced at the budget, as the model was trained to close.
    close: tuple[int, ...]
    # The prompt ends inside an open block (the template put the marker there).
    seeded: bool = False
    # Reasoning tokens allowed, markers and close included; None: no budget.
    max_tokens: int | None = None
    # Families whose opener is followed by a label (Harmony's channel,
    # Muse's recipient): the reasoning label in the forms the vocabulary
    # writes it (these the opener mask cuts; the first is the canonical
    # one), and the tokens that end a label. An opener whose label is
    # another (the answer's channel, a tool's name) does not open a block.
    labels: tuple[tuple[int, ...], ...] = ()
    label_end: tuple[int, ...] = ()
    # Whether a label the model wrote means reasoning - the API's routing,
    # so the budget counts exactly what the API shows as reasoning (a label
    # with stray whitespace too); without it, only `labels` do.
    label_means_reasoning: Callable[[tuple[int, ...]], bool] | None = None
    # Whether the opener is masked once no block fits; False for a family
    # whose block is closed at once instead and counts nothing (Gemma 4's
    # channel, whose model writes the head without a masked opener).
    mask_opener: bool = True
    # A family whose textual opener is the frame's (Muse's ` to=` right
    # after `<|start|>assistant`): the frame's tokens, and whether the
    # prompt ends with them, so the opener may be the first token.
    frame: tuple[int, ...] = ()
    at_frame: bool = False
    # Whether the opener itself is a reasoning token: a marker (`<|channel|>`)
    # is, a textual opener the assembler holds as a prefix (` to`) is not -
    # the budget counts what usage.reasoning_tokens counts.
    opener_counted: bool = True
    # Frame family: whether the head written so far can still be an opener
    # (` to` followed by `=`); None never aborts a head.
    head_may_open: Callable[[tuple[int, ...]], bool] | None = None


def budget(
    max_context: int | None,
    prompt_tokens: int,
    max_tokens: int,
    min_response_tokens: int,
) -> tuple[int, int | None]:
    """(completion cap, reasoning cap) for one request.

    A small client limit is served as it is; only a context that cannot hold
    the reserve the answer needs is refused. The reasoning cap is None when
    nothing bounds it, 0 when the answer needs all the room there is."""
    completion_cap = max_tokens
    if max_context:
        available = max_context - prompt_tokens
        requested_reserve = min(min_response_tokens, max_tokens)
        if available < max(1, requested_reserve):
            raise ContextTooLong(
                f"prompt ({prompt_tokens} tokens) leaves {max(0, available)} of "
                f"the context ({max_context}); the answer needs "
                f"{max(1, requested_reserve)}"
            )
        completion_cap = min(max_tokens, available)
    if not min_response_tokens:
        return completion_cap, None
    return completion_cap, completion_cap - min(min_response_tokens, completion_cap)


def _end_index(limits: ReasoningLimits) -> int:
    end = limits.end
    return next(
        (
            i
            for i in range(len(limits.close) - len(end) + 1)
            if limits.close[i : i + len(end)] == end
        ),
        len(limits.close) - len(end),
    )


def close_counted(limits: ReasoningLimits) -> int:
    """Tokens of the close that still count as reasoning: everything before
    the end marker's last token, since only that one leaves the block."""
    return _end_index(limits) + len(limits.end) - 1


def close_tail(limits: ReasoningLimits) -> int:
    """Tokens the forced close spends after the block: the marker's last
    token and what follows it. The answer's reserve begins after them."""
    return len(limits.close) - close_counted(limits)


class _Matcher:
    """Advances through a token sequence; True when the last token completed it."""

    def __init__(self, sequence: Sequence[int]):
        self._seq = tuple(sequence)
        self._pos = 0

    def feed(self, token: int) -> bool:
        if not self._seq:
            return False
        if token == self._seq[self._pos]:
            self._pos += 1
            if self._pos == len(self._seq):
                self._pos = 0
                return True
            return False
        # Fall back to the longest prefix the recent tokens still match, so
        # a marker whose prefix repeats (7 7 9 fed 7 7 7 9) is not missed.
        window = self._seq[: self._pos] + (token,)
        self._pos = next(
            (
                k
                for k in range(min(self._pos, len(self._seq) - 1), 0, -1)
                if window[-k:] == self._seq[:k]
            ),
            0,
        )
        return False

    @property
    def matched(self) -> int:
        """Tokens of the sequence seen so far, in order."""
        return self._pos


class ThinkingBudget:
    """Per-request: the logits processor the generator calls each step, and
    the tracker the engine feeds with every token as it becomes known.

    The budget is a hard cap for single- and multi-token markers: the opener
    is masked when the remaining allowance could not hold a block, a close
    the model starts on its own is completed rather than second-guessed, and
    a forced close disarms so a reopened block is judged afresh. A family
    whose opener is plain text (Muse's ` to=`) reads it right after the
    message's frame only and judges the head at its end. It counts what the
    API's usage counts: the head after the opener, the block, the counted
    part of the close."""

    def __init__(self, limits: ReasoningLimits):
        self.limits = limits
        self._state = "reasoning" if limits.seeded else "normal"
        self._start = _Matcher(limits.start)
        self._end = _Matcher(limits.end)
        self._label_end = _Matcher(limits.label_end)
        self._label: list[int] = []
        # A frame family: the window in which the opener is the frame's -
        # open at the frame, shut by the first token that is no opener.
        self._frame = _Matcher(limits.frame)
        self._window = limits.at_frame
        # Set while the label written so far means reasoning and its header
        # leaves no room for the free token the arming counts on (a form
        # longer than the canonical one): the step that ends the label
        # forces the close at once, decided in the graph (observe is a
        # step behind the logits, the label's end is not known to it yet).
        self._close_on_end = False
        self._label_end_ids = mx.array(limits.label_end or (0,), dtype=mx.int32)
        self.reasoning_tokens = 0
        self.thinking_truncated = False
        # Set by observe(): the token just seen came from the force queue,
        # or belongs to a block the budget closed at once (suppressed).
        self.last_forced = False
        self.last_suppressed = False
        # Progress through a block, every token seen (opener, head, content,
        # the counted part of the close) whether the block counts or not:
        # the force queue runs on it, the budget on reasoning_tokens.
        self._seen = 0
        # Set at the opener of a block the budget has no room for
        # (mask_opener False): closed at once, counted nothing.
        self._suppressed = False
        # The free token began the end marker itself: the queue stood down.
        self._natural = False
        # Forced tokens, consumed by the processor one per step, and the
        # alternative when the free token already began the end marker: the
        # rest of that marker, then the model goes on by itself.
        self._queue: tuple[int, ...] = ()
        # Set at arming: the marker token the free token would have to be
        # for the model's own close to count, and the rest of the marker.
        self._guard = limits.end[0] if limits.end else -1
        self._cont = tuple(limits.end[1:])
        self._pos = 0
        self._armed_at: int | None = None
        # Tokens sampled before the force reaches the logits: one when armed
        # from an observation, none when armed before the first step.
        self._free = 0
        # Lazy scalar: the model began the close itself on the free token.
        self._closed = None
        self._close_counted = close_counted(limits)
        # What a block costs before its first free token: the opener when
        # it counts, and for a labelled family the shortest head with its
        # end - Harmony's header is three counted tokens, Muse's two, the
        # channel's one (observe books them together).
        self._opener_cost = 1 if limits.opener_counted else 0
        self._entry = self._opener_cost
        if limits.labels:
            self._entry += len(limits.labels[0]) + len(limits.label_end)
        elif limits.label_end:
            self._entry += 1 + len(limits.label_end)
        self._one_hot: dict[int, mx.array] = {}
        # What the mask cuts to keep a block from forming: the opener's last
        # token - or, for a family whose opener is shared by every channel
        # (Harmony's <|channel|>), the reasoning label's first token after
        # it, so the answer's and a tool's channel stay open.
        cut = tuple(limits.start)
        if limits.labels:
            cut += tuple(limits.labels[0][:1])
        self._cut = cut
        self._cut_prefix = mx.array(cut[:-1], dtype=mx.int32)
        # The token cut, in every form the label takes (` analysis` is one
        # token next to `analysis`; masking one would hand the model the other).
        self._cut_last = tuple(
            sorted({label[0] for label in limits.labels if label} or {cut[-1]})
        )
        # While set, the opener cannot complete: no block fits any more.
        self._block = False
        # Tokens of the close's tail still to come after the end marker.
        self._closing = 0
        limit = limits.max_tokens
        if limit is None:
            return
        if limits.seeded and limit <= self._close_counted + 1:
            # Below this the observe-and-arm path cannot land exactly; force
            # from the first token instead and stay under the budget.
            self._arm(free=0)
        elif not limits.seeded:
            self._gate_opener()

    # -- the generator's side -----------------------------------------------

    def __call__(self, context: mx.array, logits: mx.array) -> mx.array:
        shape, vocab = logits.shape, logits.shape[-1]
        if self._block:
            logits = self._mask_opener(context, logits, vocab)
        if self._close_on_end and self._pos >= len(self._queue):
            ends = self._ends_label(context)
            if ends is not None:
                forced = self._forced(self.limits.close[0], vocab, logits.dtype)
                logits = mx.broadcast_to(mx.where(ends, forced, logits), shape)
        if self._pos < len(self._queue):
            k = self._pos
            self._pos += 1
            forced = self._forced(self._queue[k], vocab, logits.dtype)
            if self._closed is None:
                self._closed = context[-1] == self._guard
            if k < len(self._cont):
                # The model started the marker: finish it, do not double it.
                other = self._forced(self._cont[k], vocab, logits.dtype)
                logits = mx.where(self._closed, other, forced)
            else:
                logits = mx.where(self._closed, logits, forced)
            logits = mx.broadcast_to(logits, shape)
        return logits

    def _forced(self, token: int, vocab: int, dtype) -> mx.array:
        return mx.where(self._hot(token, vocab), 0.0, -mx.inf).astype(dtype)

    def _hot(self, token: int, vocab: int) -> mx.array:
        hot = self._one_hot.get(token)
        if hot is None:
            hot = self._one_hot[token] = mx.arange(vocab) == token
        return hot

    def _hot_any(self, tokens: tuple[int, ...], vocab: int) -> mx.array:
        hot = self._one_hot.get(tokens)
        if hot is None:
            hot = self._hot(tokens[0], vocab)
            for token in tokens[1:]:
                hot = hot | self._hot(token, vocab)
            self._one_hot[tokens] = hot
        return hot

    def _ends_label(self, context: mx.array):
        """Lazy: the context ends with the label's end sequence."""
        n = len(self.limits.label_end)
        if not n or context.shape[-1] < n:
            return None
        return mx.all(context[-n:] == self._label_end_ids)

    def _mask_opener(self, context: mx.array, logits: mx.array, vocab: int):
        cut = self._cut
        last = self._hot_any(self._cut_last, vocab)
        if len(cut) == 1:
            return mx.where(last, -mx.inf, logits)
        # A longer opener is cut at its last token, once the ones before it
        # are in place; the fragment before stays ordinary text.
        n = len(cut) - 1
        if context.shape[-1] < n:
            return logits
        prefix = mx.all(context[-n:] == self._cut_prefix)
        return mx.where(prefix & last, -mx.inf, logits)

    # -- the engine's side --------------------------------------------------

    def observe(self, token: int, finish_reason: str | None) -> None:
        """One generated token, in order; called after the generator returned it."""
        self.last_forced = self._is_forced(token)
        # This token's flag: the block state on entry, or a block suppressed
        # in this very call. _disarm clears the state later in the call, so
        # the close's last token still carries the flag.
        self.last_suppressed = self._suppressed
        if self._closing:
            # The close's tail after the end marker (a line break, or the
            # answer's opener for a family that closes by opening it): the
            # queue keeps forcing it, the arming ends with its last token.
            self._closing -= 1
            if self._closing == 0:
                self._disarm()
        elif self._state == "reasoning":
            if self._end.feed(token):
                self._state = "normal"
                if self._armed_at is not None:
                    # Before this count the close came from the model itself
                    # (finished by the force, at most); from it on it was forced.
                    forced = self._seen >= (
                        self._armed_at + self._free + self._close_counted
                    )
                    if not self._suppressed:
                        self.thinking_truncated |= forced
                    # The marker's last token was this one; the rest of the
                    # close follows from the queue.
                    tail = close_tail(self.limits) - 1
                    if tail > 0 and not self._natural:
                        self._closing = tail
                    else:
                        self._disarm()
            else:
                self._count(1)
                self._gate_opener()
                self._maybe_arm()
        elif self._state == "label":
            # The opener's label: reasoning only when it says so (Harmony's
            # `analysis`, Muse's `self`); the answer's channel or a tool's
            # name opens no block, and its tokens count for nothing.
            self._label.append(token)
            if self._label_end.feed(token):
                self._end_head()
            elif self._head_aborts():
                # ` to` followed by text, not by a recipient: no head.
                self._leave_head()
            else:
                self._close_on_end = self._header_overruns()
        elif self.limits.frame:
            self._at_frame(token)
        elif self._start.feed(token):
            self._open()
        if finish_reason == "length" and self.in_reasoning:
            self.thinking_truncated = True

    def _at_frame(self, token: int) -> None:
        """A frame family outside a block: the frame opens the window, in
        which only the opener keeps it open; any other token is the
        answer's text and shuts it until the next frame."""
        if self._frame.feed(token):
            self._window = True
            self._start = _Matcher(self.limits.start)
        elif self._window:
            if self._start.feed(token):
                self._open()
            elif not self._start.matched:
                self._window = False

    def _open(self) -> None:
        """The opener is complete: a labelled family reads the head next,
        the others are inside the block from here."""
        if self.limits.label_end:
            self._state = "label"
            self._label = []
            return
        self._state = "reasoning"
        if not self.limits.mask_opener and self._no_room():
            # The opener came through unmasked into a block the budget
            # has no room for: closed at once, counted nothing.
            self._suppressed = self.last_suppressed = True
        self._count(self._opener_cost)
        self._gate_opener()
        self._maybe_arm()

    def _end_head(self) -> None:
        """The label's end: a head that means reasoning opens the block
        and is booked whole; the answer's or a tool's head opens nothing."""
        label = tuple(self._label[: -len(self.limits.label_end)])
        fired = self._close_on_end
        self._close_on_end = False
        if self._means_reasoning(label):
            self._state = "reasoning"
            cost = self._opener_cost + len(self._label)
            if not self.limits.mask_opener and self._overflows(cost):
                # Only the head says whether it is reasoning, and it does
                # not fit what is left: closed at once, counted nothing.
                self._suppressed = self.last_suppressed = True
            self._count(cost)
            if fired:
                # This step's logits forced the close's first token:
                # armed from here without a free token, the queue goes
                # on from its second.
                self._arm(free=0)
                self._pos = 1
                self._closed = mx.array(False)
            self._gate_opener()
            self._maybe_arm()
        else:
            self._state = "normal"
        self._label = []
        self._window = False

    def _head_aborts(self) -> bool:
        verdict = self.limits.head_may_open
        return verdict is not None and not verdict(tuple(self._label))

    def _leave_head(self) -> None:
        self._state = "normal"
        self._label = []
        self._label_end = _Matcher(self.limits.label_end)
        self._close_on_end = False
        self._window = False

    def _count(self, n: int) -> None:
        """`n` tokens of a block seen: the progress always, the published
        count only for a block the budget admitted."""
        self._seen += n
        if not self._suppressed:
            self.reasoning_tokens += n

    def _no_room(self) -> bool:
        """Another block - its entry (opener, label and label end), a free
        token and the counted part of the close - does not fit what is left."""
        limit = self.limits.max_tokens
        return (
            limit is not None
            and limit - self.reasoning_tokens < self._entry + 1 + self._close_counted
        )

    def _overflows(self, cost: int) -> bool:
        """A head of `cost` counted tokens and the close's counted part
        exceed what is left: a head that fits exactly is counted and
        closed at once (_close_on_end), a longer one is suppressed."""
        limit = self.limits.max_tokens
        return (
            limit is not None
            and self.reasoning_tokens + cost + self._close_counted > limit
        )

    def _header_overruns(self) -> bool:
        """The label so far, were it to end now, means reasoning and costs
        more than the gate planned for: no free token fits after it in
        what the budget has left (earlier blocks spent their share)."""
        limit = self.limits.max_tokens
        if limit is None:
            return False
        so_far = tuple(self._label[: len(self._label) - self._label_end.matched])
        if not so_far or not self._means_reasoning(so_far):
            return False
        cost = self._opener_cost + len(so_far) + len(self.limits.label_end)
        return self.reasoning_tokens + cost > limit - 1 - self._close_counted

    def _means_reasoning(self, label: tuple[int, ...]) -> bool:
        verdict = self.limits.label_means_reasoning
        if verdict is not None:
            return verdict(label)
        return label in self.limits.labels

    @property
    def in_reasoning(self) -> bool:
        """Inside a block that counts; a suppressed block is no reasoning,
        so a length cut inside it is the answer's."""
        return self._state == "reasoning" and not self._suppressed

    def stable_for(self, tokens: int) -> bool:
        """The processor's state cannot change over the next `tokens`
        logits: no force in progress or closing, and `tokens` more counted
        tokens cannot arm one. A speculative cycle may then verify that
        many positions, calling the processor per position like a plain
        step would - the opener mask it may hold is a function of the
        context alone. Whether the tokens turn out counted or not does not
        matter: the bound assumes every one of them counts."""
        if self._pos < len(self._queue) or self._closing:
            return False
        if self._state == "label":
            return False  # the label decides whether counting starts
        limit = self.limits.max_tokens
        if limit is None:
            return True
        if self._block and self._state != "reasoning":
            # Outside a block with the opener masked nothing counts any
            # more: the rest of the answer is stable.
            return True
        # The force arms at `limit - 1 - close` (_maybe_arm) and the opener
        # mask at `limit - 2 - close` from the other side (_gate_opener):
        # the same bound, written once.
        return self.reasoning_tokens + tokens < limit - 1 - self._close_counted

    def _is_forced(self, token: int) -> bool:
        """Once armed, the tokens after the free one are the queue's - unless
        the free token itself began the end marker, then the model's own
        close is merely completed and nothing counts as forced."""
        if self._armed_at is None:
            return False
        if self._state != "reasoning":
            return bool(self._closing) and not self._natural
        since = self._seen - self._armed_at
        if since < self._free:
            self._natural = token == self._guard
            return False
        return not self._natural

    def _maybe_arm(self) -> None:
        limit = self.limits.max_tokens
        if limit is None or self._armed_at is not None:
            return
        # After arming, the free token follows, then the counted part of the
        # close: arm so that those still fit the budget.
        if self.reasoning_tokens >= limit - 1 - self._close_counted:
            self._arm(free=1)

    def _arm(self, free: int) -> None:
        self._queue = tuple(self.limits.close)
        self._pos = 0
        self._closed = None
        self._natural = False
        self._armed_at = self._seen
        self._free = free
        # The model may be inside the marker already (its first tokens were
        # observed): then the free token continues it, not begins it.
        end = self.limits.end
        done = self._end.matched if free else 0
        self._guard = end[done] if done < len(end) else -1
        self._cont = tuple(end[done + 1 :])

    def _disarm(self) -> None:
        self._queue = ()
        self._pos = 0
        self._armed_at = None
        self._closed = None
        self._closing = 0
        self._suppressed = False
        self._window = False

    def _gate_opener(self) -> None:
        """Mask the opener as soon as the allowance left cannot hold another
        block (_no_room). Checked on every counted token, because the mask
        reaches the logits two tokens later - by the time a block closes it
        must already be in place."""
        if not self.limits.mask_opener:
            # A masked opener keeps the model from the opener, not from the
            # channel: the block is let in and closed at once instead.
            return
        if self._no_room():
            self._block = True
