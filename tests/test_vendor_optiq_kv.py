"""Quantized batch KV cache and tiled quantized SDPA (vendored from mlx-optiq),
and the rotating-merge guard in the vendored mlx-lm."""

import mlx.core as mx
import pytest

from mlx_beam._vendor.mlx_lm.generate import BatchGenerator
from mlx_beam._vendor.mlx_lm.models import llama
from mlx_beam._vendor.mlx_lm.models.base import (
    quantized_scaled_dot_product_attention,
)
from mlx_beam._vendor.mlx_lm.models.cache import (
    BatchRotatingKVCache,
    RotatingKVCache,
)
from mlx_beam._vendor.optiq.fused_quant_sdpa import (
    fused_quantized_scaled_dot_product_attention,
)
from mlx_beam._vendor.optiq.kv_batch import (
    BatchQuantizedKVCache,
    MergeableQuantizedKVCache,
)

TINY_LLAMA = dict(
    model_type="llama",
    hidden_size=64,
    num_hidden_layers=2,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    rms_norm_eps=1e-5,
    vocab_size=64,
    head_dim=64,
    max_position_embeddings=512,
)


def tiny_llama(seed=0):
    mx.random.seed(seed)
    model = llama.Model(llama.ModelArgs(**TINY_LLAMA))
    mx.eval(model.parameters())
    return model


def dequant(triple, group_size, bits):
    return mx.dequantize(*triple, group_size=group_size, bits=bits)


def test_batch_cache_stores_what_was_written():
    mx.random.seed(1)
    cache = BatchQuantizedKVCache([0, 2], group_size=64, bits=4)
    k = mx.random.normal((2, 2, 5, 64)).astype(mx.float16)
    v = mx.random.normal((2, 2, 5, 64)).astype(mx.float16)
    qk, qv = cache.update_and_fetch(k, v)
    assert qk[0].dtype == mx.uint32 and qk[0].shape == (2, 2, 5, 64 // 8)
    ref = mx.dequantize(*mx.quantize(k, group_size=64, bits=4), group_size=64, bits=4)
    assert mx.array_equal(dequant(qk, 64, 4), ref)
    assert cache.size() == 5 and cache.offset.tolist() == [5, 3]


def test_merge_keeps_bits_of_fresh_caches():
    # Fresh per-request caches hold no keys; the batch must still take their
    # bits instead of the 8-bit default.
    fresh = [MergeableQuantizedKVCache(group_size=32, bits=4) for _ in range(2)]
    batch = BatchQuantizedKVCache.merge(fresh)
    assert (batch.bits, batch.group_size) == (4, 32)
    assert batch.empty() and batch.nbytes == 0
    assert all(c.nbytes == 0 and c.size() == 0 for c in fresh)


def test_extract_then_merge_round_trip():
    mx.random.seed(2)
    cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=8)
    k = mx.random.normal((2, 2, 6, 64)).astype(mx.float16)
    v = mx.random.normal((2, 2, 6, 64)).astype(mx.float16)
    cache.update_and_fetch(k, v)
    one = cache.extract(1)
    assert isinstance(one, MergeableQuantizedKVCache) and one.size() == 6
    again = BatchQuantizedKVCache.merge([one, MergeableQuantizedKVCache(bits=8)])
    assert again.size() == 6 and again.left_padding.tolist() == [0, 6]
    a = dequant(again.keys_and_values()[0], 64, 8)[0]
    b = dequant(cache.keys_and_values()[0], 64, 8)[1]
    assert mx.array_equal(a, b)


def test_tiled_sdpa_matches_stock():
    mx.random.seed(3)
    B, Hq, Hkv, L, N, D = 1, 4, 2, 3, 40, 64
    q = mx.random.normal((B, Hq, L, D)).astype(mx.float16)
    k = mx.random.normal((B, Hkv, N, D)).astype(mx.float16)
    v = mx.random.normal((B, Hkv, N, D)).astype(mx.float16)
    qk = mx.quantize(k, group_size=64, bits=8)
    qv = mx.quantize(v, group_size=64, bits=8)
    tiled = fused_quantized_scaled_dot_product_attention(
        q, qk, qv, scale=D**-0.5, mask="causal", group_size=64, bits=8, n_chunk=16
    )
    # Last: the stock function scales its queries in place (mx `*=` mutates).
    stock = quantized_scaled_dot_product_attention(
        q, qk, qv, scale=D**-0.5, mask="causal", group_size=64, bits=8
    )
    assert stock.shape == tiled.shape
    assert mx.abs(stock.astype(mx.float32) - tiled.astype(mx.float32)).max() < 2e-2


def test_batch_generation_with_quantized_caches():
    # The engine hands quantized per-request caches to the generator; the
    # batch it builds must carry the configured bits and decode.
    model = tiny_llama()
    model.set_dtype(mx.float16)
    gen = BatchGenerator(model, max_tokens=4, stop_tokens=[], prefill_batch_size=2)
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8]]
    caches = [
        [MergeableQuantizedKVCache(group_size=64, bits=4) for _ in model.layers]
        for _ in prompts
    ]
    uids = gen.insert(prompts, caches=caches)
    out = {u: [] for u in uids}
    seen = set()
    while any(len(t) < 4 for t in out.values()):
        _, generated = gen.next()
        for r in generated:
            out[r.uid].append(r.token)
        if generated and len(gen._generation_batch) > 0:
            layer0 = gen._generation_batch.prompt_cache[0]
            seen.add((type(layer0), layer0.bits, layer0.group_size))
    gen.close()
    assert seen == {(BatchQuantizedKVCache, 4, 64)}
    assert all(len(t) == 4 for t in out.values())


def test_rotating_merge_survives_a_zero_length_cache():
    trimmed = RotatingKVCache(max_size=8)
    k = mx.zeros((1, 2, 4, 16), mx.float16)
    trimmed.update_and_fetch(k, k)
    trimmed.trim(4)
    assert trimmed.size() == 0 and trimmed.keys is not None
    other = RotatingKVCache(max_size=8)
    other.update_and_fetch(k[..., :3, :], k[..., :3, :])
    merged = BatchRotatingKVCache.merge([trimmed, other])
    assert merged.size() == 3


def test_tiled_sdpa_matches_stock_with_a_batch_mask():
    # The batch caches hand attention a bool mask per row (left padding);
    # that is the mask the tiled path sees in the engine.
    from mlx_beam._vendor.mlx_lm.models.base import create_causal_mask
    from mlx_beam._vendor.optiq import fused_quant_sdpa

    mx.random.seed(4)
    B, Hq, Hkv, L, N, D = 2, 4, 2, 70, 134, 64
    q = mx.random.normal((B, Hq, L, D)).astype(mx.float16)
    k = mx.random.normal((B, Hkv, N, D)).astype(mx.float16)
    v = mx.random.normal((B, Hkv, N, D)).astype(mx.float16)
    qk = mx.quantize(k, group_size=64, bits=4)
    qv = mx.quantize(v, group_size=64, bits=4)
    mask = create_causal_mask(L, offset=N - L, left_padding=mx.array([0, 5]))
    assert mask.dtype == mx.bool_ and mask.shape == (B, 1, L, N)
    assert fused_quant_sdpa.supported(q, 4, 64, mask)
    # Below the threshold the stock path is cheaper: one token, and the
    # handful a speculative verify runs (fused_quant_sdpa.MIN_TILED_QUERIES).
    assert not fused_quant_sdpa.supported(q[:, :, :1], 4, 64, mask[:, :, :1])
    assert not fused_quant_sdpa.supported(q[:, :, :4], 4, 64, mask[:, :, :4])
    tiled = fused_quantized_scaled_dot_product_attention(
        q, qk, qv, scale=D**-0.5, mask=mask, group_size=64, bits=4, n_chunk=16
    )
    stock = quantized_scaled_dot_product_attention(
        q, qk, qv, scale=D**-0.5, mask=mask, group_size=64, bits=4
    )
    assert mx.abs(stock.astype(mx.float32) - tiled.astype(mx.float32)).max() < 2e-2


def test_prefill_takes_the_tiled_path(monkeypatch):
    from mlx_beam._vendor.optiq import fused_quant_sdpa

    calls = {"tiled": 0}
    original = fused_quant_sdpa.fused_quantized_scaled_dot_product_attention

    def counting(*args, **kwargs):
        calls["tiled"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        fused_quant_sdpa, "fused_quantized_scaled_dot_product_attention", counting
    )
    model = tiny_llama()
    model.set_dtype(mx.float16)
    gen = BatchGenerator(model, max_tokens=2, stop_tokens=[])
    caches = [[MergeableQuantizedKVCache(group_size=64, bits=4) for _ in model.layers]]
    gen.insert(
        [list(range(1, 80))], caches=caches
    )  # a prefill above the tile threshold
    while calls["tiled"] == 0:
        _, generated = gen.next()
        if any(r.finish_reason for r in generated):
            break
    gen.close()
    assert calls["tiled"] > 0


def test_merge_refuses_mixed_bits():
    a = MergeableQuantizedKVCache(group_size=64, bits=4)
    b = MergeableQuantizedKVCache(group_size=64, bits=8)
    with pytest.raises(ValueError, match="quantized differently"):
        BatchQuantizedKVCache.merge([a, b])
