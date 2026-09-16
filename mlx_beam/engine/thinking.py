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


def close_counted(limits: ReasoningLimits) -> int:
    """Tokens of the close before the end marker: the part that still counts
    as reasoning. The rest of the close is generated but belongs to neither."""
    end = limits.end
    return next(
        (
            i
            for i in range(len(limits.close) - len(end) + 1)
            if limits.close[i : i + len(end)] == end
        ),
        len(limits.close),
    )


def close_tail(limits: ReasoningLimits) -> int:
    """Tokens the forced close spends after the block: the marker and what
    follows it. The answer's reserve begins after them."""
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
        self._pos = 1 if token == self._seq[0] else 0
        return False


class ThinkingBudget:
    """Per-request: the logits processor the generator calls each step, and
    the tracker the engine feeds with every token as it becomes known."""

    def __init__(self, limits: ReasoningLimits):
        self.limits = limits
        self._state = "reasoning" if limits.seeded else "normal"
        self._start = _Matcher(limits.start)
        self._end = _Matcher(limits.end)
        self.reasoning_tokens = 0
        self.thinking_truncated = False
        # Forced tokens, consumed by the processor one per step.
        self._queue: tuple[int, ...] = ()
        self._pos = 0
        self._armed_at: int | None = None
        # Tokens sampled before the force reaches the logits: one when armed
        # from an observation, none when armed before the first step.
        self._free = 0
        # Lazy scalar: the model closed the block itself as the forcing began.
        self._closed = None
        # Observations left to confirm a close the guard took for natural.
        self._suspect = 0
        self._close_counted = close_counted(limits)
        self._one_hot: dict[int, mx.array] = {}
        # A single-token opener can be masked out: no block at budget 0, no
        # reopening after a forced close.
        self._block = False
        limit = limits.max_tokens
        if limit is None:
            return
        if limits.seeded:
            # Below this the observe-and-arm path cannot land exactly; force
            # from the first token instead and stay under the budget.
            if limit <= self._close_counted + 1:
                self._arm(free=0)
        elif limit < 2 + self._close_counted:
            # The opener, a free token and the close would already exceed it.
            self._block_opener()

    # -- the generator's side -----------------------------------------------

    def __call__(self, context: mx.array, logits: mx.array) -> mx.array:
        vocab = logits.shape[-1]
        if self._block:
            logits = mx.where(self._hot(self.limits.start[0], vocab), -mx.inf, logits)
        if self._pos < len(self._queue):
            token = self._queue[self._pos]
            self._pos += 1
            forced = mx.where(self._hot(token, vocab), 0.0, -mx.inf).astype(
                logits.dtype
            )
            if self._closed is None:
                # The free token in between: did it start the close by itself?
                self._closed = context[-1] == self.limits.end[0]
            logits = mx.where(self._closed, logits, forced)
        return logits

    def _hot(self, token: int, vocab: int) -> mx.array:
        hot = self._one_hot.get(token)
        if hot is None:
            hot = self._one_hot[token] = mx.arange(vocab) == token
        return hot

    # -- the engine's side --------------------------------------------------

    def observe(self, token: int, finish_reason: str | None) -> None:
        """One generated token, in order; called after the generator returned it."""
        if self._state == "reasoning":
            if self._end.feed(token):
                self._state = "normal"
                self._suspect = 0
                if self._armed_at is not None:
                    # A close this early came from the model itself, before
                    # the forced one could reach the logits.
                    forced = self.reasoning_tokens >= (
                        self._armed_at
                        + self._free
                        + self._close_counted
                        + len(self.limits.end)
                        - 1
                    )
                    self._after_close() if forced else self._disarm()
            else:
                self.reasoning_tokens += 1
                self._check_guard(token)
                self._maybe_arm()
        elif self._start.feed(token):
            self._state = "reasoning"
            self.reasoning_tokens += 1
            self._maybe_arm()
        if finish_reason == "length" and self._state == "reasoning":
            self.thinking_truncated = True

    @property
    def in_reasoning(self) -> bool:
        return self._state == "reasoning"

    def _check_guard(self, token: int) -> None:
        """The device guard reads only the first token of the end marker. A
        free token that starts the marker without finishing it fooled it:
        once that shows, arm again with a fresh guard."""
        end = self.limits.end
        if self._armed_at is None or len(end) < 2:
            return
        if self.reasoning_tokens == self._armed_at + self._free and token == end[0]:
            self._suspect = len(end) - 1
        elif self._suspect:
            self._suspect -= 1
            if not self._suspect:
                self._disarm()

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
        self._armed_at = self.reasoning_tokens
        self._free = free

    def _disarm(self) -> None:
        self._queue = ()
        self._pos = 0
        self._armed_at = None
        self._closed = None
        self._suspect = 0

    def _after_close(self) -> None:
        # The forced close is through. A single-token opener is masked from
        # here on; a longer one that reopens is closed again at once.
        self.thinking_truncated = True
        self._disarm()
        self._block_opener()

    def _block_opener(self) -> None:
        self._block = len(self.limits.start) == 1
