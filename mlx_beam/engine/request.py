"""What a caller hands the engine and what it gets back."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_beam.engine.thinking import ReasoningLimits


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    # Tokens min_p may never filter away.
    min_tokens_to_keep: int = 1
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.1
    # Ids XTC leaves alone (eos and newline, so it cannot cut the answer short).
    xtc_special_tokens: tuple[int, ...] = ()
    repetition_penalty: float | None = None
    repetition_context_size: int = 20
    presence_penalty: float | None = None
    presence_context_size: int = 20
    frequency_penalty: float | None = None
    frequency_context_size: int = 20
    logit_bias: dict[int, float] | None = None
    # A seeded request samples with its own random key.
    seed: int | None = None


@dataclass
class GenerationRequest:
    """Token ids in, tokens out. Text belongs to the API layer."""

    tokens: list[int]
    max_tokens: int = 256
    sampling: SamplingParams = field(default_factory=SamplingParams)
    # Token-id sequences that end the generation; the model's eos ids belong here.
    stop_sequences: Sequence[Sequence[int]] = ()
    # Prompt positions worth a recurrent-state checkpoint (system end, user
    # ends); the prompt end always gets one.
    boundaries: Sequence[int] = ()
    # Where the system block ends; the store keeps that prefix as its own
    # entry, evicted after every conversation entry.
    system_end: int | None = None
    # How many of the most likely tokens each event reports (0: none).
    top_logprobs: int = 0
    # Room the answer keeps after the think block; the reasoning budget is
    # cut to leave it (0: none reserved).
    min_response_tokens: int = 0
    # A prompt cap this request asks for; only lower than the engine's.
    max_prompt_tokens: int | None = None
    # Think markers and the reasoning budget; None: no budget, no forcing.
    reasoning: ReasoningLimits | None = None
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:16]}")

    def __post_init__(self):
        if not self.tokens:
            raise ValueError("prompt is empty")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.top_logprobs < 0:
            raise ValueError("top_logprobs must not be negative")
        if self.min_response_tokens < 0:
            raise ValueError("min_response_tokens must not be negative")


@dataclass(frozen=True)
class TokenEvent:
    token: int
    logprob: float
    # "stop", "length", or None while the sequence keeps going.
    finish_reason: str | None = None
    # (token id, logprob) pairs, best first; only when the request asked.
    top_logprobs: tuple[tuple[int, float], ...] | None = None
    # On the last event: the think block was closed by force or cut by the
    # length limit; the answer after it was cut by the length limit.
    thinking_truncated: bool = False
    response_truncated: bool = False


@dataclass(frozen=True)
class PromptProgress:
    processed: int
    total: int
    # Tokens the prompt cache already covered; not counted in ``processed``.
    cached: int


class ResultStream:
    """Per-request queue the worker fills; iterate to consume token events.

    Ends with ``finish_reason`` set on the last event. An exception put by the
    worker is raised to the consumer. ``cancel()`` asks the engine to drop the
    request; whatever was queued already stays readable.
    """

    def __init__(self, request: GenerationRequest, on_cancel):
        self.request = request
        self._queue: queue.Queue = queue.Queue()
        self._on_cancel = on_cancel
        self._cancelled = threading.Event()
        self.progress: PromptProgress | None = None
        self.prompt_cached = 0
        # What the engine admitted: generated tokens, reasoning tokens within.
        self.completion_cap = request.max_tokens
        self.reasoning_cap: int | None = None
        self._done = False

    def put(self, item) -> None:
        self._queue.put(item)

    def cancel(self) -> None:
        if not self._cancelled.is_set():
            self._cancelled.set()
            self._on_cancel(self.request.request_id)

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def next_event(self, timeout: float | None = None) -> TokenEvent | None:
        """The next token event; None once the stream has ended. Raises
        ``queue.Empty`` when ``timeout`` passes without one (progress updates
        are folded into ``self.progress`` while waiting)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            item = self._queue.get(timeout=remaining)
            if item is None:
                self._done = True
                return None
            if isinstance(item, PromptProgress):
                self.progress = item
                continue
            if isinstance(item, BaseException):
                self._done = True
                raise item
            if item.finish_reason is not None:
                self._done = True
            return item

    def __iter__(self) -> Iterator[TokenEvent]:
        while not self._done:
            item = self.next_event()
            if item is None:
                return
            yield item
