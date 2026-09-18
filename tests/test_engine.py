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
