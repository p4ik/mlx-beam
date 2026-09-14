"""The engine: one worker thread owns the model and runs the batch generator.

Callers submit token requests from any thread and read token events from a
per-request stream. Nothing here knows about text, HTTP or chat formats.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import asdict
from typing import Any

import mlx.core as mx

from mlx_beam._vendor.mlx_lm.generate import BatchGenerator, StopSequences
from mlx_beam._vendor.mlx_lm.models.cache import (
    LRUPromptCache,
    can_trim_prompt_cache,
    trim_prompt_cache,
)
from mlx_beam._vendor.mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_beam.engine.kv import KVPolicy, describe_caches, make_request_cache
from mlx_beam.engine.request import (
    GenerationRequest,
    PromptProgress,
    ResultStream,
    SamplingParams,
    TokenEvent,
)


class EngineDead(RuntimeError):
    """The worker thread has ended; the engine answers nothing any more."""


def _sampler(p: SamplingParams):
    return make_sampler(temp=p.temperature, top_p=p.top_p, min_p=p.min_p, top_k=p.top_k)


def _logits_processors(p: SamplingParams):
    return make_logits_processors(
        logit_bias=p.logit_bias,
        repetition_penalty=p.repetition_penalty,
        repetition_context_size=p.repetition_context_size,
        presence_penalty=p.presence_penalty,
        frequency_penalty=p.frequency_penalty,
    )


class Engine:
    def __init__(
        self,
        model: Any,
        *,
        model_key: str = "model",
        kv_policy: KVPolicy | None = None,
        completion_batch_size: int = 8,
        prefill_batch_size: int = 2,
        prefill_step_size: int = 2048,
        prefill_slice: int = 512,
        decode_share: float = 0.5,
        prompt_cache_size: int = 16,
        prompt_cache_bytes: int | None = None,
    ):
        self.model = model
        self.model_key = model_key
        self.kv_policy = kv_policy or KVPolicy()
        self._gen_args = dict(
            completion_batch_size=completion_batch_size,
            prefill_batch_size=prefill_batch_size,
            prefill_step_size=prefill_step_size,
            prefill_slice=prefill_slice,
            decode_share=decode_share,
        )
        self.prompt_cache = LRUPromptCache(
            max_size=prompt_cache_size,
            max_bytes=prompt_cache_bytes if prompt_cache_bytes else 1 << 63,
        )
        self._inbox: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._cancelled: set = set()
        self._live: dict[int, ResultStream] = {}
        self._admitting: ResultStream | None = None
        self._gen: BatchGenerator | None = None
        self._last_step = 0.0
        self._applied_caches: list[dict] | None = None
        self._started_at = 0.0
        self._ready = threading.Event()
        self.warmup_tokens: list[int] = [0]

    # -- lifecycle --------------------------------------------------------

    def start(self, timeout: float | None = 600.0) -> Engine:
        """Start the worker and wait for its warm-up: one token through the
        model with the configured KV layout. A policy the model cannot carry
        fails here, not on the first client request."""
        if self._thread is not None:
            return self
        # Lazy parameters carry the loading thread's stream; the worker cannot
        # evaluate them ("There is no Stream(cpu, 0) in current thread").
        mx.eval(self.model.parameters())
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name="beam-engine", daemon=True
        )
        self._thread.start()
        while not self._ready.wait(0.05):
            if not self._thread.is_alive():
                raise EngineDead(self._death_message())
            if timeout is not None and time.monotonic() - self._started_at > timeout:
                raise EngineDead("engine warm-up did not finish in time")
        return self

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def __enter__(self) -> Engine:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def alive(self) -> bool:
        return (
            self._thread is not None and self._thread.is_alive() and self._error is None
        )

    # -- submission -------------------------------------------------------

    def submit(self, request: GenerationRequest) -> ResultStream:
        if not self.alive:
            raise EngineDead(self._death_message())
        stream = ResultStream(request, self._request_cancel)
        self._inbox.put(stream)
        return stream

    def _request_cancel(self, request_id: str) -> None:
        with self._lock:
            self._cancelled.add(request_id)

    def _death_message(self) -> str:
        if self._error is not None:
            return f"engine worker died: {self._error!r}"
        if self._thread is None:
            return "engine not started"
        return "engine worker stopped"

    # -- health -----------------------------------------------------------

    def health(self) -> dict:
        gen = self._gen
        counters = asdict(gen._counters) if gen is not None else None
        return {
            "alive": self.alive,
            "error": repr(self._error) if self._error else None,
            "uptime_s": (
                round(time.monotonic() - self._started_at, 1)
                if self._started_at
                else 0.0
            ),
            "in_flight": len(self._live),
            "queued": self._inbox.qsize(),
            "last_step_age_s": (
                round(time.monotonic() - self._last_step, 3)
                if self._last_step
                else None
            ),
            "counters": counters,
            "batching": dict(self._gen_args),
            "kv": {
                "policy": self.kv_policy.describe(),
                # What the last batch was built from, layer by layer.
                "applied": self._applied_caches,
            },
            "prompt_cache": {
                "entries": len(self.prompt_cache),
                "bytes": self.prompt_cache.nbytes,
            },
        }

    # -- the worker -------------------------------------------------------

    def _run(self) -> None:
        try:
            self._loop()
        except BaseException as e:  # noqa: BLE001 - the death itself is the finding
            self._error = e
            traceback.print_exc()
            self._fail_everything(e)

    def _fail_everything(self, error: BaseException) -> None:
        dead = EngineDead(f"engine worker died: {error!r}")
        if self._admitting is not None:
            self._admitting.put(dead)
            self._admitting = None
        for stream in self._live.values():
            stream.put(dead)
        self._live.clear()
        while True:
            try:
                stream = self._inbox.get_nowait()
            except queue.Empty:
                break
            stream.put(EngineDead(f"engine worker died: {error!r}"))

    def _admit(self, gen: BatchGenerator, stream: ResultStream) -> None:
        req = stream.request
        try:
            cache, rest = self.prompt_cache.fetch_nearest_cache(
                self.model_key, req.tokens
            )
            if cache is None:
                cache = make_request_cache(self.model, self.kv_policy)
                rest = req.tokens
            elif not rest:
                # An exact hit: the generator still needs one token to start.
                if can_trim_prompt_cache(cache):
                    trim_prompt_cache(cache, 1)
                    rest = req.tokens[-1:]
                else:
                    cache = make_request_cache(self.model, self.kv_policy)
                    rest = req.tokens
            cached = len(req.tokens) - len(rest)
            stop = StopSequences(req.stop_sequences or None)
            (uid,) = gen.insert_segments(
                segments=[[list(rest)]],
                max_tokens=[req.max_tokens],
                caches=[cache],
                all_tokens=[list(req.tokens[:cached])],
                samplers=[_sampler(req.sampling)],
                logits_processors=[_logits_processors(req.sampling)],
                stop_sequences=[stop],
            )
        except Exception as e:  # noqa: BLE001 - one bad request, not the worker
            stream.put(e)
            return
        stream.prompt_cached = cached
        stream.put(PromptProgress(0, len(rest), cached))
        self._live[uid] = stream

    def _drop_cancelled(self, gen: BatchGenerator) -> None:
        with self._lock:
            if not self._cancelled:
                return
            ids = set(self._cancelled)
            self._cancelled.clear()
        uids = [u for u, s in self._live.items() if s.request.request_id in ids]
        if uids:
            gen.remove(uids)
        for u in uids:
            self._live.pop(u).put(None)

    def _warmup(self, gen: BatchGenerator) -> None:
        cache = make_request_cache(self.model, self.kv_policy)
        (uid,) = gen.insert([list(self.warmup_tokens)], max_tokens=[1], caches=[cache])
        while True:
            _, generated = gen.next()
            if any(r.uid == uid and r.finish_reason for r in generated):
                break
        self._applied_caches = describe_caches(cache)
        self._last_step = time.monotonic()

    def _loop(self) -> None:
        gen = BatchGenerator(self.model, stop_tokens=[], **self._gen_args)
        self._gen = gen
        try:
            self._warmup(gen)
            self._ready.set()
            while not self._stop.is_set():
                # Admit what is waiting; block only when there is nothing to do.
                try:
                    stream = self._inbox.get(timeout=0.1 if not self._live else 0)
                except queue.Empty:
                    stream = None
                    if not self._live:
                        continue
                while stream is not None:
                    self._admitting = stream
                    if stream.cancelled:
                        stream.put(None)
                    else:
                        self._admit(gen, stream)
                    self._admitting = None
                    try:
                        stream = self._inbox.get_nowait()
                    except queue.Empty:
                        stream = None

                self._drop_cancelled(gen)
                if not self._live:
                    continue

                prompt_responses, gen_responses = gen.next()
                self._last_step = time.monotonic()

                for r in prompt_responses:
                    s = self._live.get(r.uid)
                    if s is not None:
                        s.put(
                            PromptProgress(
                                r.progress[0], r.progress[1], s.prompt_cached
                            )
                        )

                for r in gen_responses:
                    s = self._live.get(r.uid)
                    if s is None:
                        continue
                    s.put(
                        TokenEvent(
                            token=r.token,
                            logprob=float(r.logprobs[r.token].item()),
                            finish_reason=r.finish_reason,
                        )
                    )
                    if r.finish_reason is not None:
                        del self._live[r.uid]
                        self.prompt_cache.insert_cache(
                            self.model_key,
                            list(r.all_tokens),
                            r.prompt_cache,
                            cache_type="assistant",
                        )
        finally:
            gen.close()
            self._gen = None
