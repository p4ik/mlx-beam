"""The engine on a tiny random-weight hybrid: concurrent requests, cancel,
prompt cache reuse, health with evidence, a dying worker that says so, and
the prompt cache that outlives a restart through --prompt-cache-dir."""

import fcntl
import json
import logging
import math
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import pytest

from mlx_beam._vendor.mlx_lm.models import qwen3_next
from mlx_beam._vendor.mlx_lm.models.cache import CacheList, KVCache
from mlx_beam.engine import (
    Engine,
    EngineDead,
    GenerationRequest,
    KVPolicy,
    persist,
)
from mlx_beam.engine.kv import describe_caches, for_store, make_request_cache
from mlx_beam.engine.prefix import HEAD
from mlx_beam.engine.request import SamplingParams
from tests.test_vendor_mlx_lm import TINY_HYBRID
from tests.test_vendor_optiq_kv import tiny_llama


def tiny_hybrid(seed=0):
    mx.random.seed(seed)
    model = qwen3_next.Model(qwen3_next.ModelArgs(**TINY_HYBRID))
    mx.eval(model.parameters())
    return model


def tiny_hybrid_q(seed=0):
    """`tiny_hybrid` with a head dim a quantization group of 64 divides."""
    mx.random.seed(seed)
    model = qwen3_next.Model(qwen3_next.ModelArgs(**{**TINY_HYBRID, "head_dim": 64}))
    mx.eval(model.parameters())
    return model


def collect(stream):
    return [e.token for e in stream]


def test_two_requests_match_solo_runs():
    model = tiny_hybrid()
    a, b = [3, 7, 11, 13], [5, 9, 13, 17, 21, 25, 29]
    with Engine(model) as solo:
        ra = collect(solo.submit(GenerationRequest(a, max_tokens=6)))
    with Engine(model) as solo:
        rb = collect(solo.submit(GenerationRequest(b, max_tokens=6)))
    with Engine(model) as both:
        sa = both.submit(GenerationRequest(a, max_tokens=6))
        sb = both.submit(GenerationRequest(b, max_tokens=6))
        assert collect(sa) == ra
        assert collect(sb) == rb
        assert sa.progress is not None and sa.progress.total == len(a)


def test_finish_reasons_and_stop_sequence():
    model = tiny_hybrid()
    with Engine(model) as engine:
        events = list(engine.submit(GenerationRequest([3, 7, 11], max_tokens=3)))
        assert [e.finish_reason for e in events] == [None, None, "length"]
        first = events[0].token
        events = list(
            engine.submit(
                GenerationRequest([3, 7, 11], max_tokens=8, stop_sequences=[[first]])
            )
        )
        assert events[-1].finish_reason == "stop" and events[-1].token == first


def test_prompt_cache_is_reused():
    # A plain-attention model: its cache is trimmable, so a stored sequence
    # serves every prefix. Recurrent state is not, until checkpoints land.
    model = tiny_llama()
    prompt = list(range(1, 20))
    with Engine(model) as engine:
        first = collect(engine.submit(GenerationRequest(prompt, max_tokens=4)))
        assert engine.health()["prompt_cache"]["entries"] == 1
        again = engine.submit(GenerationRequest(prompt, max_tokens=4))
        assert collect(again) == first
        assert again.prompt_cached >= len(prompt) - 1


@pytest.mark.parametrize("ending", ["length", "stop", "cancel"])
def test_next_event_is_terminal_after_the_end(ending):
    """None once the stream ended, on every later call too: a finishing
    token leaves no sentinel in the queue, so the flag has to answer."""
    import queue

    from mlx_beam.engine.request import ResultStream, TokenEvent

    stream = ResultStream(GenerationRequest([1]), lambda _: None)
    stream.put(None if ending == "cancel" else TokenEvent(2, -0.1, ending))
    first = stream.next_event(timeout=0.5)
    assert (first is None) == (ending == "cancel")
    try:
        assert stream.next_event(timeout=0.05) is None
    except queue.Empty:
        raise AssertionError("next_event waited on an ended stream") from None


def test_cancel_drops_the_request():
    model = tiny_hybrid()
    with Engine(model) as engine:
        stream = engine.submit(GenerationRequest([3, 7, 11], max_tokens=200))
        it = iter(stream)
        next(it)
        stream.cancel()
        rest = list(it)
        assert len(rest) < 200
        deadline = time.monotonic() + 5
        while engine.health()["in_flight"] and time.monotonic() < deadline:
            time.sleep(0.05)
        assert engine.health()["in_flight"] == 0


def test_health_reports_the_kv_layout_actually_built():
    # head_dim 64 so a quantization group fits; float32 because the CPU build
    # has no half-precision gather-matmul for the experts.
    mx.random.seed(0)
    model = qwen3_next.Model(qwen3_next.ModelArgs(**{**TINY_HYBRID, "head_dim": 64}))
    mx.eval(model.parameters())
    for prefill, kind in (
        ("exact", "DeferredQuantizedKVCache"),
        ("quantized", "MergeableQuantizedKVCache"),
    ):
        with Engine(model, kv_policy=KVPolicy(bits=4, prefill=prefill)) as engine:
            collect(engine.submit(GenerationRequest([3, 7, 11], max_tokens=2)))
            h = engine.health()
            assert h["alive"] and h["kv"]["policy"]["bits"] == 4
            assert h["kv"]["policy"]["prefill"] == prefill
            applied = h["kv"]["applied"]
            # Attention layers carry 4 bits; the recurrent layers stay as they are.
            assert {e["type"] for e in applied} >= {kind, "ArraysCache"}
            assert all(e.get("bits", 4) == 4 for e in applied)
            assert {e.get("prefill") for e in applied if "bits" in e} == {prefill}


def test_a_policy_the_model_cannot_carry_fails_at_start():
    # head_dim 16 cannot be quantized in groups of 64: the warm-up must say so
    # before any client sees the server as healthy.
    model = tiny_hybrid()
    engine = Engine(model, kv_policy=KVPolicy(bits=4))
    with pytest.raises(EngineDead, match="group size"):
        engine.start()
    assert not engine.alive and "group size" in engine.health()["error"]


def test_a_dying_worker_fails_loudly():
    model = tiny_hybrid()
    engine = Engine(model).start()
    boom = RuntimeError("metal said no")

    def explode(*_):
        raise boom

    engine._admit = explode  # the first request kills the worker
    stream = engine.submit(GenerationRequest([3, 7, 11], max_tokens=2))
    with pytest.raises(EngineDead):
        list(stream)
    deadline = time.monotonic() + 5
    while engine.alive and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not engine.alive and "metal said no" in engine.health()["error"]
    with pytest.raises(EngineDead):
        engine.submit(GenerationRequest([1], max_tokens=1))


def test_sampling_with_temperature_runs():
    model = tiny_hybrid()
    with Engine(model) as engine:
        params = SamplingParams(temperature=0.8, top_p=0.9, top_k=10)
        out = collect(
            engine.submit(GenerationRequest([3, 7, 11], max_tokens=5, sampling=params))
        )
        assert len(out) == 5


def test_submissions_from_many_threads():
    model = tiny_hybrid()
    with Engine(model, decode_concurrency=8, prompt_concurrency=4) as engine:
        results = {}

        def run(i):
            results[i] = collect(engine.submit(GenerationRequest([i + 1, 5, 9], 4)))

        threads = [threading.Thread(target=run, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert sorted(results) == list(range(6))
        assert all(len(v) == 4 for v in results.values())


def test_store_budget_is_its_own_limit_and_context_caps_max_tokens():
    from mlx_beam.engine import ContextTooLong

    model = tiny_llama()
    with Engine(model, prompt_cache_bytes=1) as engine:
        # One byte of budget: nothing fits in the store, whatever runs.
        collect(engine.submit(GenerationRequest([1, 2, 3], max_tokens=2)))
        assert engine.health()["prompt_cache"]["entries"] == 0
    with Engine(model) as engine:
        collect(engine.submit(GenerationRequest([1, 2, 3], max_tokens=2)))
        one = engine.health()["prompt_cache"]["bytes"]
        assert one > 0
    # A budget for two entries keeps two, although each request that ran
    # held caches of its own meanwhile: those do not count against the store.
    with Engine(model, prompt_cache_bytes=2 * one) as engine:
        collect(engine.submit(GenerationRequest([1, 2, 3], max_tokens=2)))
        collect(engine.submit(GenerationRequest([4, 5, 6], max_tokens=2)))
        cache = engine.health()["prompt_cache"]
        assert cache["entries"] == 2 and cache["bytes"] <= 2 * one
        collect(engine.submit(GenerationRequest([1, 2, 3, 7], max_tokens=2)))
        assert engine.health()["prompt_cache"]["tokens_found"] >= 3
    with Engine(model, max_context=8) as engine:
        # A large client limit is served and capped at what the context holds.
        out = collect(engine.submit(GenerationRequest([1, 2, 3], max_tokens=6)))
        assert len(out) == 5
        # Only a reserve the context cannot hold is refused.
        with pytest.raises(ContextTooLong):
            engine.submit(
                GenerationRequest([1, 2, 3], max_tokens=6, min_response_tokens=6)
            )
        with pytest.raises(ContextTooLong):
            engine.submit(GenerationRequest([1, 2, 3, 4, 5, 6, 7, 8], max_tokens=1))
        assert engine.health()["max_context"] == 8


def test_store_budget_ignores_the_caches_of_running_requests():
    """The budget bounds what the store holds, nothing else: a request whose
    live caches exceed it does not push stored prefixes out while it runs."""
    import time

    model = tiny_llama()
    with Engine(model) as engine:
        collect(engine.submit(GenerationRequest([1, 2, 3], max_tokens=2)))
        one = engine.health()["prompt_cache"]["bytes"]
    with Engine(model, prompt_cache_bytes=2 * one) as engine:
        collect(engine.submit(GenerationRequest([1, 2, 3], max_tokens=2)))
        assert engine.health()["prompt_cache"]["entries"] == 1
        # A long request holds caches far beyond the budget while it runs;
        # the stored entry must be there whenever we look meanwhile.
        long = engine.submit(GenerationRequest(list(range(4, 64)), max_tokens=600))
        time.sleep(0.05)
        other = engine.submit(GenerationRequest([7, 8, 9], max_tokens=2))
        seen = [engine.health()["prompt_cache"]["entries"] for _ in long]
        # The last token commits the long request's own entry, which does
        # not fit and takes the store with it (an oversized entry evicts
        # everything before itself) - a store rule, not a budget one.
        assert len(seen) > 100 and min(seen[:-1]) >= 1
        collect(other)


def test_short_request_is_not_stuck_behind_a_long_prefill():
    # The claim on the site: a short request beside a long prefill answers in
    # a fraction of that prefill. Rule 2 of the scheduler (slice width).
    import random

    from mlx_beam._vendor.mlx_lm.models import llama

    mx.random.seed(0)
    cfg = dict(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=4,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
        head_dim=64,
        max_position_embeddings=4096,
    )
    model = llama.Model(llama.ModelArgs(**cfg))
    mx.eval(model.parameters())
    rng = random.Random(1)

    def long_prompt():
        return [rng.randint(1, 60) for _ in range(1500)]

    def ttft(engine, toks):
        t0 = time.monotonic()
        stream = engine.submit(GenerationRequest(toks, max_tokens=1))
        next(iter(stream))
        return time.monotonic() - t0

    with Engine(model, prefill_slice=64, prompt_cache_size=0) as engine:
        solo_long = ttft(engine, long_prompt())
        result = {}
        prompt = long_prompt()
        t = threading.Thread(target=lambda: result.update(long=ttft(engine, prompt)))
        t.start()
        time.sleep(min(0.15, solo_long / 10))
        short = ttft(engine, [7, 11, 13, 17])
        t.join()
        # One short request is no starvation: the guard never stepped in.
        assert engine.health()["prefill_starved_calls"] == 0
    assert short < 0.5 * solo_long, (short, solo_long)
    assert result["long"] < 2.0 * solo_long, (result["long"], solo_long)


def test_max_queued_refuses_the_overflow_with_a_count():
    from mlx_beam.engine import QueueFull

    model = tiny_hybrid()
    # One sequence decodes at a time, so the rest wait in the backlog.
    with Engine(
        model, max_queued=2, decode_concurrency=1, prompt_concurrency=1
    ) as engine:
        streams = [
            engine.submit(GenerationRequest([3, 7, 11], max_tokens=40))
            for _ in range(3)
        ]
        with pytest.raises(QueueFull) as exc:
            engine.submit(GenerationRequest([5, 9], max_tokens=40))
        assert "limit is 2" in str(exc.value)
        assert engine.health()["rejected_queue_full"] == 1
        for s in streams:
            assert len(collect(s)) == 40
        # The backlog drained: the next one is admitted again.
        assert len(collect(engine.submit(GenerationRequest([5, 9], max_tokens=2)))) == 2


def test_cancelling_a_row_that_never_prefilled_does_not_kill_the_worker(monkeypatch):
    """A one-token prompt is handed to generation before the rows admitted
    with it are prefilled; those rows hold empty caches. Cancelling one of
    them then must drop it, not the engine."""
    import importlib

    G = importlib.import_module("mlx_beam._vendor.mlx_lm.generate")
    fired = threading.Event()
    target = {}
    real_next = G.BatchGenerator.next

    def cancel_when_unprefilled(self):
        out = real_next(self)
        pb = self._prompt_batch
        if (
            not fired.is_set()
            and target.get("stream") is not None
            and pb.uids
            and getattr(pb.prompt_cache[0], "keys", 1) is None
        ):
            fired.set()
            target["stream"].cancel()
        return out

    monkeypatch.setattr(G.BatchGenerator, "next", cancel_when_unprefilled)
    with Engine(
        tiny_llama(), decode_concurrency=4, prompt_concurrency=2, prompt_cache_size=0
    ) as engine:
        busy = engine.submit(GenerationRequest([20, 21, 22], max_tokens=300))
        next(iter(busy))
        outcome = {}
        for _ in range(20):
            one = engine.submit(GenerationRequest([42], max_tokens=3))
            many = engine.submit(GenerationRequest(list(range(7, 15)), max_tokens=3))
            target["stream"] = many
            threads = [
                threading.Thread(
                    target=lambda n, s: outcome.setdefault(n, collect(s)), args=a
                )
                for a in (("one", one), ("many", many))
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
            if fired.is_set():
                break
            outcome.clear()
        assert fired.is_set(), "the window was never hit"
        assert engine.alive and len(outcome["one"]) == 3
        assert collect(busy)  # the bystander finishes normally


def _tiny(module, seed=0, **overrides):
    mx.random.seed(seed)
    model = module.Model(module.ModelArgs(**overrides))
    mx.eval(model.parameters())
    return model


GEMMA4_SHARED = dict(
    model_type="gemma4_text",
    hidden_size=64,
    num_hidden_layers=6,
    intermediate_size=128,
    num_attention_heads=2,
    head_dim=64,
    global_head_dim=64,
    num_key_value_heads=1,
    num_kv_shared_layers=2,
    vocab_size=64,
    vocab_size_per_layer_input=64,
    hidden_size_per_layer_input=16,
    sliding_window=8,
    sliding_window_pattern=3,
    use_double_wide_mlp=False,
)

# MeroMero-26B-A4B's shape in small: no shared layers, keys are the values,
# the global layers have twice the head dim of the sliding ones.
GEMMA4_MEROMERO = dict(
    GEMMA4_SHARED,
    num_kv_shared_layers=0,
    attention_k_eq_v=True,
    global_head_dim=128,
)

PLAMO2 = dict(
    model_type="plamo2",
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    hidden_size_per_head=64,
    intermediate_size=128,
    vocab_size=64,
    mamba_num_heads=2,
    mamba_d_state=8,
)


@pytest.mark.parametrize(
    "module_name, cfg",
    [
        ("gemma4_text", GEMMA4_SHARED),
        ("gemma4_text", GEMMA4_MEROMERO),
        ("plamo2", PLAMO2),
    ],
)
def test_quantized_kv_on_kv_sharing_and_direct_sdpa_models(module_name, cfg):
    """Gemma 4's KV-shared layers get another layer's packed cache with no
    cache of their own; PLaMo-2 called MLX's attention directly. Both must
    run under a KV policy, stay close to the unquantized logits, and serve
    through the engine (whose worker thread must see every lazy array)."""
    import importlib

    from mlx_beam._vendor.mlx_lm.models.cache import make_prompt_cache
    from mlx_beam.engine.kv import make_request_cache

    module = importlib.import_module(f"mlx_beam._vendor.mlx_lm.models.{module_name}")
    model = _tiny(module, **cfg)
    prompt = [3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]

    def logprobs(caches):
        model(mx.array([prompt]), cache=caches)
        out = model(mx.array([[5]]), cache=caches)[0, -1]
        return out - mx.logsumexp(out)

    ref = logprobs(make_prompt_cache(model))
    # Random weights make flat distributions, so 4 bit moves them visibly;
    # 8 bit is within a few hundredths and keeps the argmax.
    for bits, tolerance in ((8, 0.1), (4, 0.6)):
        quant = logprobs(make_request_cache(model, KVPolicy(bits=bits)))
        assert mx.abs(ref - quant).max().item() < tolerance, bits
        if bits == 8:
            assert mx.argmax(ref).item() == mx.argmax(quant).item()
    with Engine(model, kv_policy=KVPolicy(bits=8)) as engine:
        out = collect(engine.submit(GenerationRequest(prompt, max_tokens=4)))
        applied = engine.health()["kv"]["applied"]
    assert len(out) == 4 and any(a.get("bits") == 8 for a in applied), applied


def test_a_group_size_the_head_dim_cannot_carry_names_the_policy():
    from mlx_beam._vendor.mlx_lm.models import gemma4_text

    model = _tiny(gemma4_text, **dict(GEMMA4_SHARED, head_dim=96, global_head_dim=96))
    engine = Engine(model, kv_policy=KVPolicy(bits=4, group_size=64))
    with pytest.raises(EngineDead, match="group size 64 does not divide") as exc:
        engine.start()
    assert "head_dim 96" in str(exc.value) and "[32]" in str(exc.value)


# Qwen3.5/3.8 ship as a multimodal wrapper: `Model.args` holds only the
# model type and a `text_config` dict, the sizes live one level down.
TINY_QWEN35_WRAPPED = {
    "model_type": "qwen3_5",
    "text_config": dict(
        model_type="qwen3_5_text",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        vocab_size=64,
        max_position_embeddings=512,
        full_attention_interval=2,
    ),
}


def test_sizes_come_from_the_wrapper_text_config():
    from mlx_beam._vendor.mlx_lm.models import qwen3_5
    from mlx_beam.engine import ContextTooLong
    from mlx_beam.engine.core import model_context_length, model_vocab_size

    mx.random.seed(0)
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(TINY_QWEN35_WRAPPED))
    mx.eval(model.parameters())
    assert model_context_length(model) == 512
    assert model_vocab_size(model) == 64
    with Engine(model) as engine:
        out = collect(engine.submit(GenerationRequest([3, 7, 11], max_tokens=3)))
        assert len(out) == 3
        assert engine.health()["max_context"] == 512
        with pytest.raises(ContextTooLong):
            engine.submit(GenerationRequest(list(range(1, 60)) * 9, max_tokens=8))


def test_a_failing_store_costs_the_entry_not_the_engine(monkeypatch):
    """The entry of a finished row could not be written: the answer still
    reaches the caller, the engine lives, health counts the loss."""
    model = tiny_hybrid()
    with Engine(model) as engine:

        def broken(*a, **k):
            raise RuntimeError("no room in the store")

        monkeypatch.setattr(engine.prefix_store, "insert", broken)
        out = collect(engine.submit(GenerationRequest([3, 7, 11], max_tokens=3)))
        assert len(out) == 3
        h = engine.health()
        assert h["alive"] and h["prompt_cache"]["store_failures"] == 1
        # And the next request runs as if nothing happened.
        assert len(collect(engine.submit(GenerationRequest([3, 7], max_tokens=2)))) == 2


# -- the store on disk --------------------------------------------------------

# A fixed identity for the tiny models, which have no files on disk.
IDENTITY = {
    "model": {
        "snapshot": None,
        "config": "c0ffee",
        "index": None,
        "kv_config": None,
        "weights": [["model.safetensors", 1, 1]],
    },
    "mlx": mx.__version__,
    "engine": "test",
    "vision": None,
    "head_digest": None,
}

SYS = list(range(1, 13))
U1, U2, U3 = [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]


class Clock:
    """A counting clock for `persist`: `sleep(dt)` advances it by `dt`, and
    the `release_at`-th sleep runs `release` (a lock given up meanwhile)."""

    def __init__(self, release_at=None, release=None):
        self.now = 1000.0
        self.sleeps = 0
        self.release_at = release_at
        self.release = release

    def monotonic(self):
        return self.now

    def sleep(self, dt):
        self.sleeps += 1
        self.now += dt
        if self.release is not None and self.sleeps == self.release_at:
            self.release()


def fake_clock(monkeypatch, **kw):
    clock = Clock(**kw)
    monkeypatch.setattr(persist, "monotonic", clock.monotonic)
    monkeypatch.setattr(persist, "sleep", clock.sleep)
    return clock


def hold(path):
    """An exclusive flock on `path` from this thread; the fd keeps it."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def run(engine, tokens, max_tokens=4, boundaries=(), system_end=None):
    stream = engine.submit(
        GenerationRequest(
            list(tokens),
            max_tokens=max_tokens,
            boundaries=boundaries,
            system_end=system_end,
        )
    )
    return [e.token for e in stream], stream.prompt_cached


def persistent_engine(model, tmp_path, identity=IDENTITY, **kw):
    kw.setdefault("kv_policy", KVPolicy(bits=8))
    return Engine(model, prompt_cache_dir=str(tmp_path), cache_identity=identity, **kw)


def persistent(engine):
    return engine.health()["prompt_cache"]["persistent"]


def wait_loaded(engine, timeout=60.0):
    """The worker reads the store after the readiness; health says when."""
    deadline = time.monotonic() + timeout
    while persistent(engine)["loading"]:
        assert engine.alive, engine.health()["error"]
        assert time.monotonic() < deadline, "the store did not finish loading"
        time.sleep(0.01)


def turns(engine, upto):
    """A conversation of up to three turns over a system block: the prompts,
    the answers and the tokens the cache covered per turn."""
    s = len(SYS)
    p1 = SYS + U1
    a1, c1 = run(engine, p1, boundaries=[s], system_end=s)
    if upto == 1:
        return [p1], [a1], [c1]
    p2 = p1 + a1 + U2
    a2, c2 = run(engine, p2, boundaries=[s, len(p1 + a1)], system_end=s)
    if upto == 2:
        return [p1, p2], [a1, a2], [c1, c2]
    p3 = p2 + a2 + U3
    a3, c3 = run(engine, p3, boundaries=[s, len(p1 + a1), len(p2 + a2)], system_end=s)
    return [p1, p2, p3], [a1, a2, a3], [c1, c2, c3]


def namespaces(tmp_path):
    return sorted(d for d in tmp_path.iterdir() if d.is_dir())


def the_namespace(tmp_path):
    """The one namespace folder a stopped engine left, with its manifest."""
    found = namespaces(tmp_path)
    assert len(found) == 1, found
    assert (found[0] / "manifest.json").is_file()
    return found[0]


def manifest(ns_dir):
    return json.loads((ns_dir / "manifest.json").read_text())


def disk_total(tmp_path):
    return sum(f.stat().st_size for f in tmp_path.rglob("*") if f.is_file())


def assert_same_state(x, y):
    if isinstance(x, mx.array):
        assert isinstance(y, mx.array)
        assert (x.dtype, x.shape) == (y.dtype, y.shape)
        assert mx.array_equal(x, y)
    elif isinstance(x, (list, tuple)):
        assert type(x) is type(y) and len(x) == len(y)
        for u, v in zip(x, y, strict=True):
            assert_same_state(u, v)
    else:
        assert x == y


def assert_same_entry(got, kept):
    """Bit-equal: every array, every scalar, the classes, the checkpoints."""
    assert got.length == kept.length and got.cache_type == kept.cache_type
    assert [type(c) for c in got.cache] == [type(c) for c in kept.cache]
    for a, b in zip(got.cache, kept.cache, strict=True):
        assert_same_state(a.state, b.state)
    assert sorted(got.checkpoints) == sorted(kept.checkpoints)
    for pos in kept.checkpoints:
        assert set(got.checkpoints[pos]) == set(kept.checkpoints[pos])
        for layer in kept.checkpoints[pos]:
            assert_same_state(got.checkpoints[pos][layer], kept.checkpoints[pos][layer])
    assert got.nbytes == kept.nbytes


def test_the_store_survives_a_restart_bit_exact(tmp_path):
    """Engine A stops after turn 2; B starts on A's directory and serves
    turn 3 from the file as a warm engine would: the conversation entry
    comes back bit-equal, the hit covers the same tokens, the answer is
    the same."""
    model = tiny_hybrid_q()
    with Engine(model, kv_policy=KVPolicy(bits=8)) as ref:
        _, answers_r, cached_r = turns(ref, 3)
    with persistent_engine(model, tmp_path) as a:
        prompts, answers, _ = turns(a, 2)
        key = prompts[1] + answers[1]
        kept = a.prefix_store._trie.get(a.model_key, key)
    ns = the_namespace(tmp_path)
    m = manifest(ns)
    assert len(m["entries"]) >= 2
    with persistent_engine(model, tmp_path) as b:
        wait_loaded(b)
        assert persistent(b)["loaded_entries"] >= 2
        rec = next(
            e
            for e in m["entries"]
            if e["cache_type"] == "assistant" and e["length"] == len(key)
        )
        got_key, got = persist.read_entry(str(ns / rec["file"]), rec)
        assert got_key == key
        assert_same_entry(got, kept)
        out, cached = run(
            b,
            key + U3,
            boundaries=[len(SYS), len(prompts[0] + answers[0]), len(key)],
            system_end=len(SYS),
        )
    assert cached == cached_r[2] and out == answers_r[2]


def test_a_changed_configuration_is_a_miss(tmp_path):
    """Another KV layout, other model files or another mlx-beam version is
    another namespace: nothing loads, the old one is counted as foreign. A
    manifest of another format, and an image entry written under another
    vision package, are misses inside the namespace."""
    model = tiny_hybrid_q()
    policy = KVPolicy(bits=8)
    with persistent_engine(model, tmp_path) as a:
        prompts, answers, _ = turns(a, 2)
        # An image entry: negative ids in its key, written with this run's
        # vision identity (none).
        a.prefix_store.insert(
            a.model_key, [-5, -6, 3, 7, 9], for_store(make_request_cache(model, policy))
        )
    p3 = prompts[1] + answers[1] + U3
    bounds = [len(SYS), len(prompts[0] + answers[0]), len(prompts[1] + answers[1])]
    saved = manifest(the_namespace(tmp_path))
    text_entries = sum(1 for e in saved["entries"] if not e["image"])
    assert text_entries == len(saved["entries"]) - 1

    def copy(name):
        target = tmp_path.parent / f"{tmp_path.name}-{name}"
        shutil.copytree(tmp_path, target)
        return target

    with persistent_engine(model, copy("bits"), kv_policy=KVPolicy(bits=4)) as b:
        wait_loaded(b)
        h = persistent(b)
        assert h["loaded_entries"] == 0 and h["other_namespaces"] == 1
        out, cached = run(b, p3, boundaries=bounds, system_end=len(SYS))
        assert cached == 0 and len(out) == 4
    for name, change in (
        ("model", {"model": {**IDENTITY["model"], "config": "deadbeef"}}),
        ("engine", {"engine": "other"}),
    ):
        with persistent_engine(model, copy(name), identity={**IDENTITY, **change}) as c:
            wait_loaded(c)
            h = persistent(c)
            assert h["loaded_entries"] == 0 and h["other_namespaces"] == 1
    bumped = copy("format")
    ns = the_namespace(bumped)
    m = manifest(ns)
    m["format"] = persist.FORMAT + 1
    (ns / "manifest.json").write_text(json.dumps(m))
    with persistent_engine(model, bumped) as d:
        wait_loaded(d)
        h = persistent(d)
        assert h["loaded_entries"] == 0
        assert h["skipped"]["incompatible"] == len(m["entries"])
    with persistent_engine(
        model, copy("vision"), identity={**IDENTITY, "vision": {"package": "2.0"}}
    ) as e:
        wait_loaded(e)
        h = persistent(e)
        assert h["skipped"]["incompatible"] == 1
        assert h["loaded_entries"] == text_entries


def test_a_corrupt_entry_is_a_miss_and_the_rest_loads(tmp_path):
    model = tiny_hybrid_q()
    with persistent_engine(model, tmp_path) as a:
        prompts, answers, _ = turns(a, 2)
    ns = the_namespace(tmp_path)
    m = manifest(ns)
    key = prompts[1] + answers[1]
    rec = next(e for e in m["entries"] if e["length"] == len(key))
    damaged = ns / rec["file"]
    damaged.write_bytes(damaged.read_bytes()[: rec["bytes"] // 2])
    leftover = ns / "abc.1.4242.new.safetensors"
    leftover.write_bytes(b"not a tensor file")
    with persistent_engine(model, tmp_path) as b:
        wait_loaded(b)
        h = persistent(b)
        assert h["skipped"]["corrupt"] == 1 and h["skipped"]["partial"] == 1
        assert h["loaded_entries"] == len(m["entries"]) - 1
        assert not damaged.exists() and not leftover.exists()
        # The conversation is gone; the system block still serves.
        _, cached = run(
            b,
            key + U3,
            boundaries=[len(SYS), len(prompts[0] + answers[0]), len(key)],
            system_end=len(SYS),
        )
        assert cached == len(SYS)


def test_the_head_sidecar_round_trips_and_a_foreign_head_is_dropped(tmp_path):
    """The draft head's history in a checkpoint comes back for the same
    head (its digest) and is dropped, counted, for another - the entry
    itself stays."""
    from tests.test_speculative import (
        OracleProposer,
        plain_transcript,
        run_speculative,
        tiny_qwen35,
    )

    class HeadStub(OracleProposer):
        def __init__(self, digest):
            super().__init__(lambda call: 0)
            self.digest = digest
            self.resumed = []

        def snapshot(self, uid):
            return [
                mx.full((1, 1, 4, 8), 1.5),
                mx.full((1, 1, 4, 8), -2.5, dtype=mx.float16),
                mx.arange(32, dtype=mx.float32)[None, None],
            ]

        def begin(self, uid, prompt_len, snapshot):
            self.resumed.append(snapshot)

        def describe(self):
            return {"kind": "stub", "digest": self.digest}

    def engine_with(digest):
        stub = HeadStub(digest)
        engine = run_speculative(
            model,
            stub,
            prompt_cache_dir=str(tmp_path),
            cache_identity={**IDENTITY, "head_digest": digest},
        )
        return engine, stub

    model = tiny_qwen35()
    prompt = [3, 7, 11, 13, 5]
    truth = plain_transcript(model, prompt, 6)
    a, stub = engine_with("head-1")
    with a:
        stub.start(truth)
        out = collect(a.submit(GenerationRequest(prompt, max_tokens=6)))
    key = prompt + out
    b, stub = engine_with("head-1")
    with b:
        wait_loaded(b)
        hit = b.prefix_store.fetch(b.model_key, key + [9, 9])
        assert hit is not None and hit.covered == len(key)
        head = hit.checkpoints[len(key)].get(HEAD)
        assert head is not None
        assert_same_state(head, stub.snapshot(0))
        assert persistent(b)["sidecars_dropped"] == 0
    c, _ = engine_with("head-2")
    with c:
        wait_loaded(c)
        h = persistent(c)
        assert h["sidecars_dropped"] == 1 and h["loaded_entries"] == 1
        hit = c.prefix_store.fetch(c.model_key, key + [9, 9])
        assert hit is not None and HEAD not in hit.checkpoints[len(key)]


def test_nothing_touches_the_disk_without_the_flag(tmp_path):
    model = tiny_hybrid()
    with Engine(model) as engine:
        run(engine, [3, 7, 11])
        assert engine.health()["prompt_cache"]["persistent"] is None
    assert list(tmp_path.iterdir()) == []


def test_the_save_respects_free_space(tmp_path, monkeypatch):
    model = tiny_hybrid_q()
    engine = persistent_engine(model, tmp_path)
    with engine:
        wait_loaded(engine)
        run(engine, SYS + U1)

        class Full:
            f_bavail = 0
            f_frsize = 4096

        monkeypatch.setattr(os, "statvfs", lambda path: Full())
    h = persistent(engine)
    assert h["save_skipped_no_space"] == 1 and h["failures"] == 0
    assert h["saved_entries"] == 0
    ns = the_namespace(tmp_path)
    assert not (ns / "manifest.json").exists()
    assert not engine.alive and engine.health()["error"] is None


def test_health_reports_the_persistent_store(tmp_path):
    model = tiny_hybrid_q()
    with persistent_engine(model, tmp_path) as engine:
        h = persistent(engine)
    assert h["dir"] == str(tmp_path)
    assert set(h) == {
        "dir",
        "namespace",
        "loading",
        "loaded_entries",
        "loaded_bytes",
        "load_ms",
        "skipped",
        "sidecars_dropped",
        "saved_entries",
        "saved_bytes",
        "save_ms",
        "save_skipped_entries",
        "save_skipped_no_space",
        "save_skipped_over_cap",
        "save_skipped_unsupported",
        "last_save",
        "disk_cap_bytes",
        "budget_wait_ms",
        "namespaces_evicted",
        "namespace_bytes_evicted",
        "sweep_skipped_deadline",
        "disk_bytes",
        "disk_bytes_total",
        "other_namespaces",
        "failures",
    }
    assert set(h["skipped"]) == {"incompatible", "partial", "corrupt", "over_budget"}
    assert isinstance(h["loading"], bool)
    assert h["last_save"] is None or isinstance(h["last_save"], str)
    assert h["namespace"] is None or isinstance(h["namespace"], str)
    counters = {
        k: v
        for k, v in h.items()
        if k not in ("dir", "namespace", "loading", "last_save", "skipped")
    }
    assert all(
        isinstance(v, int) and not isinstance(v, bool) for v in counters.values()
    )
    assert h["disk_cap_bytes"] == 100 * 1024**3


def test_an_unsupported_cache_class_is_skipped_at_save(tmp_path, caplog):
    model = tiny_hybrid_q()
    a = persistent_engine(model, tmp_path)
    with a:
        wait_loaded(a)
        run(a, SYS + U1, boundaries=[len(SYS)], system_end=len(SYS))
        kv = KVCache()
        k = mx.zeros((1, 1, 4, 8))
        kv.update_and_fetch(k, k)
        a.prefix_store.insert(a.model_key, [50, 51, 52], [CacheList(kv)])
        assert a.health()["prompt_cache"]["entries"] == 3
    h = persistent(a)
    assert h["save_skipped_unsupported"] == 1 and h["saved_entries"] == 2
    ns = the_namespace(tmp_path)
    m = manifest(ns)
    assert sorted(e["length"] for e in m["entries"]) == [len(SYS), len(SYS + U1) + 4]
    assert "CacheList" in caplog.text


def test_the_loader_reads_only_what_the_caps_keep(tmp_path, monkeypatch):
    """A smaller store at the restart: the manifest's order and sizes say
    what the caps would evict; those files are not even read."""
    model = tiny_hybrid_q()
    prompts = [list(range(1, 9)), list(range(9, 17)), list(range(17, 25))]
    keys = []
    with persistent_engine(model, tmp_path, prompt_cache_size=8) as a:
        wait_loaded(a)
        for prompt in prompts:
            out, _ = run(a, prompt)
            keys.append(prompt + out)
    reads = []
    real = persist.read_entry

    def counting(path, expected):
        reads.append(path)
        return real(path, expected)

    monkeypatch.setattr(persist, "read_entry", counting)
    with persistent_engine(model, tmp_path, prompt_cache_size=2) as b:
        wait_loaded(b)
        h = persistent(b)
        assert h["skipped"]["over_budget"] == 1 and h["loaded_entries"] == 2
        assert len(reads) == 2
        assert not b.prefix_store.has(b.model_key, keys[0])
        assert all(b.prefix_store.has(b.model_key, k) for k in keys[1:])


def own_namespace(model, policy=None, identity=IDENTITY):
    policy = policy or KVPolicy(bits=8)
    applied = describe_caches(make_request_cache(model, policy))
    return persist.namespace(persist.namespace_parts(identity, policy, applied))


def test_a_second_process_on_the_namespace_is_refused(tmp_path, monkeypatch):
    model = tiny_hybrid_q()
    ns = own_namespace(model)
    (tmp_path / ns).mkdir(mode=0o700)
    fd = hold(str(tmp_path / ns / "lock"))
    clock = fake_clock(monkeypatch)
    try:
        engine = persistent_engine(model, tmp_path)
        with pytest.raises(EngineDead, match=re.escape(str(tmp_path / ns))):
            engine.start()
    finally:
        os.close(fd)
    assert clock.sleeps == math.ceil(persist.LOCK_WAIT_S / 0.1)
    assert clock.now - 1000.0 >= persist.LOCK_WAIT_S
    assert not engine.health()["alive"]


def test_a_lock_released_in_time_lets_the_engine_start(tmp_path, monkeypatch):
    model = tiny_hybrid_q()
    with persistent_engine(model, tmp_path) as a:
        run(a, SYS + U1)
    saved = persistent(a)["saved_entries"]
    assert saved == 1
    ns = the_namespace(tmp_path)
    fd = hold(str(ns / "lock"))
    monkeypatch.setattr(persist, "LOCK_WAIT_S", 2.0)
    clock = fake_clock(
        monkeypatch, release_at=5, release=lambda: fcntl.flock(fd, fcntl.LOCK_UN)
    )
    try:
        with persistent_engine(model, tmp_path) as b:
            assert b.alive
            wait_loaded(b)
            assert persistent(b)["loaded_entries"] == saved
    finally:
        os.close(fd)
    assert clock.sleeps == 5


def foreign_namespace(tmp_path, name, saved_at, files):
    ns = tmp_path / name
    ns.mkdir(mode=0o700)
    (ns / "lock").touch()
    (ns / "manifest.json").write_text(
        json.dumps({"format": persist.FORMAT, "saved_at": saved_at, "entries": []})
    )
    for file, size in files.items():
        (ns / file).write_bytes(bytes(size))
    return ns


def listing(path):
    return sorted((f.relative_to(path), f.stat().st_size) for f in path.rglob("*"))


def test_the_oldest_foreign_namespace_falls_under_the_cap(tmp_path, monkeypatch):
    """Three foreign namespaces, the oldest with an old generation beside the
    one its manifest names: the sweep counts every file, and takes the
    oldest namespace whole - its data and manifest, never its lock."""
    model = tiny_hybrid_q()
    with persistent_engine(model, tmp_path) as a:
        run(a, SYS + U1)
    own = persistent(a)["disk_bytes"]
    assert own > 0
    old = foreign_namespace(
        tmp_path, "f-old", 1, {"aa.0.safetensors": 4000, "aa.1.safetensors": 4000}
    )
    mid = foreign_namespace(tmp_path, "f-mid", 2, {"bb.1.safetensors": 4000})
    new = foreign_namespace(tmp_path, "f-new", 3, {"cc.1.safetensors": 4000})
    before = {p: listing(p) for p in (mid, new)}
    reports, totals = [], []
    real = persist.sweep

    def observed(dir, own_ns, cap_bytes, deadline, log):
        totals.append(disk_total(tmp_path))
        report = real(dir, own_ns, cap_bytes, deadline, log)
        reports.append(report)
        return report

    monkeypatch.setattr(persist, "sweep", observed)
    cap = own + 12000
    with persistent_engine(model, tmp_path, prompt_cache_disk_bytes=cap) as b:
        wait_loaded(b)
    assert reports and reports[0].total_before == totals[0]
    assert reports[0].total_before >= own + 16000
    assert sorted(f.name for f in old.iterdir()) == ["lock"]
    assert {p: listing(p) for p in (mid, new)} == before
    assert disk_total(tmp_path) <= cap
    h = persistent(b)
    assert h["namespaces_evicted"] == 1 and h["namespace_bytes_evicted"] == 8000
    assert h["other_namespaces"] == 2 and h["disk_bytes_total"] <= cap
    assert h["disk_bytes_total"] == disk_total(tmp_path)


def test_the_own_and_a_locked_namespace_never_fall(tmp_path, caplog):
    model = tiny_hybrid_q()
    with persistent_engine(model, tmp_path) as a:
        run(a, SYS + U1)
    own = persistent(a)["disk_bytes"]
    own_ns = the_namespace(tmp_path)
    generation = manifest(own_ns)["generation"]
    locked = foreign_namespace(tmp_path, "f-locked", 1, {"dd.1.safetensors": 4000})
    fd = hold(str(locked / "lock"))
    was = listing(locked)
    try:
        with persistent_engine(
            model, tmp_path, prompt_cache_disk_bytes=own + 2000
        ) as b:
            wait_loaded(b)
    finally:
        os.close(fd)
    h = persistent(b)
    assert h["failures"] == 0 and h["namespaces_evicted"] == 0
    assert manifest(own_ns)["generation"] == generation + 1
    assert listing(locked) == was
    assert h["disk_bytes_total"] > h["disk_cap_bytes"]
    assert h["disk_bytes_total"] == disk_total(tmp_path)
    assert "over the cap" in caplog.text


def test_a_new_generation_leaves_the_old_file_until_the_manifest_turns(
    tmp_path, monkeypatch
):
    """A key that comes back with other checkpoints gets a new file name;
    the old file stays until the new manifest is published. A save that
    dies in between leaves the old state whole and a stray new file."""
    model = tiny_hybrid_q()
    with persistent_engine(model, tmp_path) as a:
        out, _ = run(a, SYS + U1)
        key = SYS + U1 + out
        kept = a.prefix_store._trie.get(a.model_key, key)
    ns = the_namespace(tmp_path)
    (rec,) = manifest(ns)["entries"]
    first = ns / rec["file"]
    assert first.name.endswith(".1.safetensors")
    first_bytes = first.read_bytes()

    def boom(ns_dir, m):
        raise OSError("disk went away")

    with persistent_engine(model, tmp_path) as b:
        wait_loaded(b)
        entry = b.prefix_store._trie.get(b.model_key, key)
        checkpoints = dict(entry.checkpoints)
        checkpoints[5] = checkpoints[max(checkpoints)]
        b.prefix_store.insert(b.model_key, key, entry.cache, checkpoints)
        monkeypatch.setattr(persist, "write_manifest", boom)
    monkeypatch.undo()
    assert persistent(b)["failures"] == 1
    second = ns / first.name.replace(".1.safetensors", ".2.safetensors")
    assert first.read_bytes() == first_bytes and second.exists()
    assert manifest(ns)["generation"] == 1
    with persistent_engine(model, tmp_path) as c:
        wait_loaded(c)
        h = persistent(c)
        assert h["loaded_entries"] == 1 and h["skipped"]["partial"] == 1
        got_key, got = persist.read_entry(str(first), rec)
        assert got_key == key
        assert_same_entry(got, kept)
        assert not second.exists()


def test_two_concurrent_saves_end_under_the_cap(tmp_path, monkeypatch):
    """Two engines of different namespaces stop at once into one directory
    whose cap holds one of them: neither fails, and a sweep with no lock
    held afterwards brings the directory under the cap."""
    model = tiny_hybrid_q()
    scratch = tmp_path / "scratch"
    with persistent_engine(model, scratch) as probe:
        run(probe, SYS + U1)
    one = persistent(probe)["disk_bytes"]
    cap = one + one // 2
    shared = tmp_path / "shared"
    shared.mkdir()
    engines = [
        persistent_engine(
            model,
            shared,
            identity={**IDENTITY, "engine": name},
            prompt_cache_disk_bytes=cap,
        )
        for name in ("one", "two")
    ]
    for engine in engines:
        engine.start()
        wait_loaded(engine)
        run(engine, SYS + U1)
    barrier = threading.Barrier(2)
    real = persist.save_store

    def together(*args, **kwargs):
        barrier.wait(timeout=30)
        return real(*args, **kwargs)

    monkeypatch.setattr(persist, "save_store", together)
    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(e.stop) for e in engines]:
            future.result(timeout=60)
    for engine in engines:
        assert not engine.alive and engine.health()["error"] is None
        assert persistent(engine)["failures"] == 0
        assert persistent(engine)["saved_entries"] == 1
    assert disk_total(shared) <= cap + 2 * one
    report = persist.sweep(
        str(shared), "none", cap, persist.monotonic() + 5, logging.getLogger("test")
    )
    assert report.total_after <= cap and report.evicted >= 1
    assert disk_total(shared) <= cap


def test_the_sweep_keeps_the_save_deadline(tmp_path, monkeypatch, caplog):
    """One monotonic deadline for the whole save: the sweep waits for the
    budget lock only until it, then gives up and says so; a lock that
    frees in time is taken with the save's own deadline, not a fresh one."""
    model = tiny_hybrid_q()
    clock = fake_clock(monkeypatch)
    starts, deadlines = [], []
    real_save, real_sweep, real_write = (
        persist.save_store,
        persist.sweep,
        persist.write_entry,
    )

    def save(*args, **kwargs):
        starts.append(clock.now)
        return real_save(*args, **kwargs)

    def sweep(dir, own_ns, cap_bytes, deadline, log):
        deadlines.append(deadline)
        return real_sweep(dir, own_ns, cap_bytes, deadline, log)

    def write(*args, **kwargs):
        n = real_write(*args, **kwargs)
        clock.now += 0.3
        return n

    monkeypatch.setattr(persist, "save_store", save)
    monkeypatch.setattr(persist, "sweep", sweep)
    monkeypatch.setattr(persist, "write_entry", write)
    a = persistent_engine(model, tmp_path, prompt_cache_save_seconds=0.5)
    with a:
        wait_loaded(a)
        run(a, SYS + U1)
        fd = hold(str(tmp_path / "budget.lock"))
    h = persistent(a)
    assert h["sweep_skipped_deadline"] == 1 and h["budget_wait_ms"] == 200
    assert h["saved_entries"] == 1 and h["save_skipped_entries"] == 0
    ns = the_namespace(tmp_path)
    assert (ns / "manifest.json").exists()
    assert "budget lock" in caplog.text
    assert deadlines[-1] == starts[-1] + 0.5
    # Freed at the first sleep: the run goes on under the save's deadline.
    clock.release_at, clock.release = clock.sleeps + 1, lambda: fcntl.flock(
        fd, fcntl.LOCK_UN
    )
    b = persistent_engine(model, tmp_path, prompt_cache_save_seconds=0.5)
    with b:
        wait_loaded(b)
    os.close(fd)
    h = persistent(b)
    assert h["sweep_skipped_deadline"] == 0 and h["budget_wait_ms"] == 100
    assert deadlines[-1] == starts[-1] + 0.5


def test_the_sweep_at_start_has_its_own_deadline(tmp_path, monkeypatch):
    model = tiny_hybrid_q()
    with persistent_engine(model, tmp_path) as a:
        run(a, SYS + U1)
    saved = persistent(a)["saved_entries"]
    fd = hold(str(tmp_path / "budget.lock"))
    clock = fake_clock(monkeypatch)
    try:
        with persistent_engine(model, tmp_path) as b:
            wait_loaded(b)
            h = persistent(b)
            assert h["sweep_skipped_deadline"] == 1
            assert clock.sleeps == math.ceil(persist.LOCK_WAIT_S / 0.1)
            assert h["loaded_entries"] == saved and b.alive
        clock = fake_clock(
            monkeypatch, release_at=1, release=lambda: fcntl.flock(fd, fcntl.LOCK_UN)
        )
        with persistent_engine(model, tmp_path) as c:
            wait_loaded(c)
            assert persistent(c)["sweep_skipped_deadline"] == 0
            assert persistent(c)["budget_wait_ms"] == 100
    finally:
        os.close(fd)


def test_a_stopped_engine_frees_its_namespace_for_the_next_one(tmp_path, monkeypatch):
    model = tiny_hybrid_q()
    monkeypatch.setattr(persist, "LOCK_WAIT_S", 0.3)
    with persistent_engine(model, tmp_path) as a:
        run(a, SYS + U1)
    saved = persistent(a)["saved_entries"]
    with persistent_engine(model, tmp_path) as b:
        assert b.alive
        wait_loaded(b)
        assert persistent(b)["loaded_entries"] == saved
    real = persist.read_manifest

    def unreadable(ns_dir):
        raise OSError("manifest unreadable")

    monkeypatch.setattr(persist, "read_manifest", unreadable)
    with persistent_engine(model, tmp_path) as c:
        wait_loaded(c)
        assert persistent(c)["failures"] == 1 and persistent(c)["loaded_entries"] == 0
        run(c, SYS + U2)
    monkeypatch.setattr(persist, "read_manifest", real)
    with persistent_engine(model, tmp_path) as d:
        assert d.alive
        wait_loaded(d)
        assert persistent(d)["loaded_entries"] == 1


def test_an_unchanged_store_is_not_rewritten_on_stop(tmp_path, monkeypatch):
    model = tiny_hybrid_q()
    with persistent_engine(model, tmp_path) as a:
        run(a, SYS + U1, boundaries=[len(SYS)], system_end=len(SYS))
    ns = the_namespace(tmp_path)
    before = manifest(ns)
    assert len(before["entries"]) == 2
    writes = []
    real = persist.write_entry

    def counting(*args, **kwargs):
        writes.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(persist, "write_entry", counting)
    with persistent_engine(model, tmp_path) as b:
        wait_loaded(b)
    assert writes == []
    h = persistent(b)
    assert h["saved_entries"] == 0 and h["saved_bytes"] == 0 and h["failures"] == 0
    after = manifest(ns)
    assert sorted(e["file"] for e in after["entries"]) == sorted(
        e["file"] for e in before["entries"]
    )
    assert after["generation"] == before["generation"] + 1
