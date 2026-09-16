"""The engine on a tiny random-weight hybrid: concurrent requests, cancel,
prompt cache reuse, health with evidence, and a dying worker that says so."""

import threading
import time

import mlx.core as mx
import pytest

from mlx_beam._vendor.mlx_lm.models import qwen3_next
from mlx_beam.engine import (
    Engine,
    EngineDead,
    GenerationRequest,
    KVPolicy,
)
from mlx_beam.engine.request import SamplingParams
from tests.test_vendor_mlx_lm import TINY_HYBRID
from tests.test_vendor_optiq_kv import tiny_llama


def tiny_hybrid(seed=0):
    mx.random.seed(seed)
    model = qwen3_next.Model(qwen3_next.ModelArgs(**TINY_HYBRID))
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
    with Engine(model, kv_policy=KVPolicy(bits=4)) as engine:
        collect(engine.submit(GenerationRequest([3, 7, 11], max_tokens=2)))
        h = engine.health()
        assert h["alive"] and h["kv"]["policy"]["bits"] == 4
        applied = h["kv"]["applied"]
        # Attention layers carry 4 bits; the recurrent layers stay as they are.
        assert {e["type"] for e in applied} >= {
            "MergeableQuantizedKVCache",
            "ArraysCache",
        }
        assert all(e.get("bits", 4) == 4 for e in applied)


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


def test_stored_prefixes_yield_to_running_requests():
    from mlx_beam.engine import ContextTooLong

    model = tiny_llama()
    with Engine(model, prompt_cache_bytes=1) as engine:
        collect(engine.submit(GenerationRequest([1, 2, 3], max_tokens=2)))
        # One byte of budget: the store cannot keep anything once a request runs.
        collect(engine.submit(GenerationRequest([4, 5, 6], max_tokens=2)))
        assert engine.health()["prompt_cache"]["entries"] <= 1
    with Engine(model, max_context=8) as engine:
        with pytest.raises(ContextTooLong):
            engine.submit(GenerationRequest([1, 2, 3], max_tokens=6))
        assert engine.health()["max_context"] == 8


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
    assert short < 0.5 * solo_long, (short, solo_long)
    assert result["long"] < 2.0 * solo_long, (result["long"], solo_long)


def test_a_failing_store_reaches_the_finishing_stream():
    # The store runs before the last token is delivered; if it raises, the
    # worker dies and that stream must hear about it, not wait forever.
    model = tiny_llama()
    with Engine(model) as engine:

        def boom(*args, **kwargs):
            raise MemoryError("no room for the entry")

        engine.prefix_store.insert = boom
        stream = engine.submit(GenerationRequest([1, 2, 3], max_tokens=1))
        with pytest.raises(EngineDead, match="no room"):
            for _ in stream:
                pass
        assert not engine.alive
