"""What a caller hands the engine and what it gets back."""

from __future__ import annotations

import queue
import threading
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float | None = None
    repetition_context_size: int = 20
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[int, float] | None = None


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
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:16]}")

    def __post_init__(self):
        if not self.tokens:
            raise ValueError("prompt is empty")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")


@dataclass(frozen=True)
class TokenEvent:
    token: int
    logprob: float
    # "stop", "length", or None while the sequence keeps going.
    finish_reason: str | None = None


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

    def put(self, item) -> None:
        self._queue.put(item)

    def cancel(self) -> None:
        if not self._cancelled.is_set():
            self._cancelled.set()
            self._on_cancel(self.request.request_id)

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def __iter__(self) -> Iterator[TokenEvent]:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if isinstance(item, PromptProgress):
                self.progress = item
                continue
            if isinstance(item, BaseException):
                raise item
            yield item
            if item.finish_reason is not None:
                return
