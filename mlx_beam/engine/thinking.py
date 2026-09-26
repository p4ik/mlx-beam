"""The reasoning budget: how many tokens the think block may take, and the
close that is forced when it runs out.

The generator samples one step ahead of what Python has seen, so a decision
taken on the token just observed reaches the logits two tokens later. The
tracker arms early enough for that, and the processor guards on the device
against a block the model closed by itself in between.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    # Muse's recipient): the label sequences that mean reasoning, and the
    # tokens that end a label. An opener whose label is another (the
    # answer's channel, a tool's name) does not open a block.
    labels: tuple[tuple[int, ...], ...] = ()
    label_end: tuple[int, ...] = ()


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
    a forced close disarms so a reopened block is judged afresh."""

    def __init__(self, limits: ReasoningLimits):
        self.limits = limits
        self._state = "reasoning" if limits.seeded else "normal"
        self._start = _Matcher(limits.start)
        self._end = _Matcher(limits.end)
        self._label_end = _Matcher(limits.label_end)
        self._label: list[int] = []
        self.reasoning_tokens = 0
        self.thinking_truncated = False
        # Set by observe(): the token just seen came from the force queue.
        self.last_forced = False
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
        # What a block costs before its first free token: the opener, and
        # for a labelled family the label with its end - Harmony's header
        # is three counted tokens, not one (observe books them together).
        self._entry = 1
        if limits.labels:
            self._entry += len(limits.labels[0]) + len(limits.label_end)
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

    def _mask_opener(self, context: mx.array, logits: mx.array, vocab: int):
        cut = self._cut
        if len(cut) == 1:
            return mx.where(self._hot(cut[0], vocab), -mx.inf, logits)
        # A longer opener is cut at its last token, once the ones before it
        # are in place; the fragment before stays ordinary text.
        n = len(cut) - 1
        if context.shape[-1] < n:
            return logits
        prefix = mx.all(context[-n:] == self._cut_prefix)
        return mx.where(prefix & self._hot(cut[-1], vocab), -mx.inf, logits)

    # -- the engine's side --------------------------------------------------

    def observe(self, token: int, finish_reason: str | None) -> None:
        """One generated token, in order; called after the generator returned it."""
        self.last_forced = self._is_forced(token)
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
                    forced = self.reasoning_tokens >= (
                        self._armed_at + self._free + self._close_counted
                    )
                    self.thinking_truncated |= forced
                    # The marker's last token was this one; the rest of the
                    # close follows from the queue.
                    tail = close_tail(self.limits) - 1
                    if tail > 0 and not self._natural:
                        self._closing = tail
                    else:
                        self._disarm()
            else:
                self.reasoning_tokens += 1
                self._gate_opener()
                self._maybe_arm()
        elif self._state == "label":
            # The opener's label: reasoning only when it says so (Harmony's
            # `analysis`, Muse's `self`); the answer's channel or a tool's
            # name opens no block, and its tokens count for nothing.
            self._label.append(token)
            if self._label_end.feed(token):
                label = tuple(self._label[: -len(self.limits.label_end)])
                if label in self.limits.labels:
                    self._state = "reasoning"
                    self.reasoning_tokens += 1 + len(self._label)
                    self._gate_opener()
                    self._maybe_arm()
                else:
                    self._state = "normal"
                self._label = []
        elif self._start.feed(token):
            if self.limits.labels:
                self._state = "label"
                self._label = []
            else:
                self._state = "reasoning"
                self.reasoning_tokens += 1
                self._gate_opener()
                self._maybe_arm()
        if finish_reason == "length" and self._state == "reasoning":
            self.thinking_truncated = True

    @property
    def in_reasoning(self) -> bool:
        return self._state == "reasoning"

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
        since = self.reasoning_tokens - self._armed_at
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
        self._armed_at = self.reasoning_tokens
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

    def _gate_opener(self) -> None:
        """Another block costs its entry (opener, label and label end), a
        free token and the counted part of the close before the force can
        land: mask the opener as soon as the allowance left cannot hold
        that. Checked on every counted token, because the mask reaches the
        logits two tokens later - by the time a block closes it must
        already be in place."""
        limit = self.limits.max_tokens
        if (
            limit is not None
            and limit - self.reasoning_tokens < self._entry + 1 + self._close_counted
        ):
            self._block = True
