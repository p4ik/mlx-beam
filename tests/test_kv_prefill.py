"""When a quantized KV cache becomes quantized: after the prefill (exact,
the default) or as it is written (quantized). The acceptance: the exact
mode computes what mlx-lm's `generate_step` computes with
`quantized_kv_start` at the prompt's end - to the bit, whatever the
prefill chunking - and the store never holds a layer at model precision.
(mlx-lm's default start of 5000 leaves shorter prompts unquantized; a
start of 0 quantizes after every chunk; neither is a reference here.)"""

import time

import mlx.core as mx
import pytest

from mlx_beam._vendor.mlx_lm.generate import generate_step
from mlx_beam._vendor.mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_beam._vendor.optiq.kv_batch import MergeableQuantizedKVCache
from mlx_beam.engine import Engine, GenerationRequest, KVPolicy
from mlx_beam.engine.kv import (
    DeferredQuantizedKVCache,
    for_prefill,
    for_store,
    make_request_cache,
)
from mlx_beam.engine.request import PromptProgress
from tests.test_vendor_optiq_kv import tiny_llama

PROMPT = [3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43] * 35  # 420 tokens


def prefill(model, caches, tokens):
    for i in range(0, len(tokens), 512):
        model(mx.array([tokens[i : i + 512]]), cache=caches)
        mx.eval([c.state for c in caches])


def logprobs(model, caches, last):
    logits = model(mx.array([[last]]), cache=caches)[0, -1].astype(mx.float32)
    return logits - mx.logsumexp(logits)


def mlx_lm_way(model, tokens, bits, chunk=512):
    """mlx-lm's own generate_step, quantizing at the prompt's end: the
    reference the exact mode is held to."""
    gen = generate_step(
        mx.array(tokens),
        model,
        max_tokens=1,
        kv_bits=bits,
        kv_group_size=64,
        prefill_step_size=chunk,
        quantized_kv_start=len(tokens) - 1,
    )
    _, ref = next(gen)
    mx.eval(ref)
    gen.close()
    return ref


def first_event(engine, tokens):
    return next(
        iter(engine.submit(GenerationRequest(tokens, max_tokens=1, top_logprobs=5)))
    )


@pytest.mark.parametrize("chunk", [512, 64])
def test_exact_prefill_matches_generate_step_bit_for_bit_and_quantized_does_not(chunk):
    model = tiny_llama()
    for bits in (8, 4):
        ref = mlx_lm_way(model, PROMPT, bits, chunk)
        gaps = {}
        for mode in ("exact", "quantized"):
            with Engine(
                model,
                kv_policy=KVPolicy(bits=bits, prefill=mode),
                prefill_step_size=chunk,
                prefill_slice=chunk,
            ) as engine:
                ev = first_event(engine, PROMPT)
                gaps[mode] = max(abs(ref[t].item() - lp) for t, lp in ev.top_logprobs)
                # Whatever the prefill did, the store holds quantized layers.
                entry = engine.prefix_store._trie.get(
                    engine.model_key, PROMPT + [ev.token]
                )
                assert all(
                    isinstance(c, MergeableQuantizedKVCache) for c in entry.cache
                )
        assert gaps["exact"] == 0.0, gaps
        assert gaps["quantized"] > 0.001, gaps


def test_generate_step_with_its_default_start_is_not_the_reference():
    # The default quantized_kv_start of 5000 never quantizes a 420-token
    # prompt; a start of 0 quantizes after every chunk. Both differ from
    # the exact mode - the reference is the start at the prompt's end.
    model = tiny_llama()
    exact = mlx_lm_way(model, PROMPT, 8, 64)
    for start in (5000, 0):
        gen = generate_step(
            mx.array(PROMPT),
            model,
            max_tokens=1,
            kv_bits=8,
            kv_group_size=64,
            prefill_step_size=64,
            quantized_kv_start=start,
        )
        _, other = next(gen)
        mx.eval(other)
        gen.close()
        assert mx.abs(exact - other).max().item() > 0.0005, start


def test_default_policy_is_exact_and_the_request_cache_says_so():
    model = tiny_llama()
    policy = KVPolicy(bits=8)
    assert policy.prefill == "exact"
    caches = make_request_cache(model, policy)
    assert all(isinstance(c, DeferredQuantizedKVCache) for c in caches)
    # Not `bits`: that name is what routes attention to the quantized kernels.
    assert not any(hasattr(c, "bits") for c in caches)
    with Engine(model, kv_policy=policy) as engine:
        applied = engine.health()["kv"]["applied"]
        assert {(e["bits"], e["prefill"]) for e in applied} == {(8, "exact")}


def test_a_stored_prefix_comes_back_exact_for_the_next_turn():
    # The hit is dequantized for the prefill and quantized again when
    # stored; the second turn matches a reference that does the same with
    # plain mlx-lm calls.
    model = tiny_llama()
    turn1 = PROMPT[:200]
    bits = 4
    with Engine(model, kv_policy=KVPolicy(bits=bits)) as engine:
        ev = first_event(engine, turn1)
        turn2 = turn1 + [ev.token] + [5, 9, 13]
        ev2 = first_event(engine, turn2)
        assert engine.prefix_store.stats.tokens_restored == len(turn1) + 1
    # Reference with plain calls: turn 1 the mlx-lm way, the generated token
    # written into the quantized cache by the decode step, the prefix back at
    # model precision for the new tokens, quantized again for the last one.
    caches = make_prompt_cache(model)
    prefill(model, caches, turn1[:-1])
    for i, c in enumerate(caches):
        if type(c) is KVCache:
            caches[i] = c.to_quantized(group_size=64, bits=bits)
    model(mx.array([turn1[-1:]]), cache=caches)
    model(mx.array([[ev.token]]), cache=caches)
    for c in caches:
        if hasattr(c, "bits"):
            c.__class__ = MergeableQuantizedKVCache
    caches = for_prefill(caches, KVPolicy(bits=bits))
    prefill(model, caches, turn2[len(turn1) + 1 : -1])
    caches = for_store(caches)
    ref = logprobs(model, caches, turn2[-1])
    assert max(abs(ref[t].item() - lp) for t, lp in ev2.top_logprobs) == 0.0


def test_dequantize_and_requantize_round_trip_keeps_the_data():
    kv = KVCache()
    k = mx.random.normal((1, 2, 300, 64))
    v = mx.random.normal((1, 2, 300, 64))
    kv.update_and_fetch(k, v)
    stored = kv.to_quantized(group_size=64, bits=8)
    stored.__class__ = MergeableQuantizedKVCache
    back = for_prefill([stored], KVPolicy(bits=8))[0]
    assert isinstance(back, DeferredQuantizedKVCache) and back.offset == 300
    again = for_store([back])[0]
    assert isinstance(again, MergeableQuantizedKVCache)
    # The prefix keeps its codes; only tokens added later are quantized fresh.
    for a, b in zip(stored.keys, again.keys, strict=True):
        assert mx.array_equal(a[..., :300, :], b[..., :300, :])
    for a, b in zip(stored.values, again.values, strict=True):
        assert mx.array_equal(a[..., :300, :], b[..., :300, :])
    # The quantized policy keeps the stored form as it is.
    assert for_prefill([stored], KVPolicy(bits=8, prefill="quantized"))[0] is stored


def test_a_cancelled_exact_prefill_is_stored_quantized():
    model = tiny_llama()
    prompt = [2, 6, 1, 2, 7, 3, 5, 4, 3, 3, 6, 7, 1, 2, 6, 5]
    with Engine(model, kv_policy=KVPolicy(bits=8), prefill_slice=4) as engine:
        stream = engine.submit(GenerationRequest(prompt, max_tokens=4))
        while True:
            item = stream._queue.get(timeout=30)
            if isinstance(item, PromptProgress) and item.processed >= 4:
                break
        stream.cancel()
        for _ in stream:
            pass
        for _ in range(50):
            if engine.prefix_store.describe()["entries"]:
                break
            time.sleep(0.05)
        key = next(k for _, k in engine.prefix_store._lru["assistant"])
        entry = engine.prefix_store._trie.get(engine.model_key, list(key))
        assert all(isinstance(c, MergeableQuantizedKVCache) for c in entry.cache)
        again = engine.submit(GenerationRequest(prompt, max_tokens=2))
        list(again)
        assert again.prompt_cached >= 4
