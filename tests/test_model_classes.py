"""One tiny random-weight model per architecture class the engine serves,
each pushed through the same four paths: batched decode, the KV policy, the
prefix store across a boundary, the speculative verify. A class the engine
cannot carry on one of these paths must say so at start, never die on the
first request.

The classes mirror the test-model list: attention sinks (gpt-oss), sliding
window with NoPE full layers and logit softcapping (Muse), MLA (GLM-4.7),
Mamba-2 + attention (Granite 4), plain attention behind a vision wrapper
(Mistral 3). Qwen3.5 (gated delta net), Gemma 4 (KV sharing) and Llama have
their own tests elsewhere.
"""

import mlx.core as mx
import pytest

from mlx_beam._vendor.mlx_lm.models import (
    glm4_moe_lite,
    gpt_oss,
    granitemoehybrid,
    mistral3,
    muse_glimmer,
)
from mlx_beam.engine import Engine, EngineDead, GenerationRequest, KVPolicy
from tests.test_speculative import OracleProposer, bit_exact, run_speculative

WINDOW = 8

CLASSES = {
    "gpt_oss": dict(
        module=gpt_oss,
        cfg=dict(
            model_type="gpt_oss",
            num_hidden_layers=4,
            num_local_experts=4,
            num_experts_per_tok=2,
            vocab_size=64,
            hidden_size=64,
            intermediate_size=64,
            head_dim=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            sliding_window=WINDOW,
            rope_theta=10000,
        ),
        window=True,
        recurrent=False,
        unquantizable="attention sinks",
    ),
    "muse_glimmer": dict(
        module=muse_glimmer,
        cfg=dict(
            model_type="muse_glimmer",
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            vocab_size=64,
            sliding_window=WINDOW,
        ),
        window=True,
        recurrent=False,
        unquantizable=None,
    ),
    "glm4_moe_lite": dict(
        module=glm4_moe_lite,
        cfg=dict(
            model_type="glm4_moe_lite",
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=2,
            n_shared_experts=1,
            n_routed_experts=4,
            kv_lora_rank=32,
            q_lora_rank=32,
            qk_rope_head_dim=16,
            qk_nope_head_dim=16,
            v_head_dim=32,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            num_nextn_predict_layers=1,
        ),
        window=False,
        recurrent=False,
        unquantizable="MLA",
    ),
    "granitemoehybrid": dict(
        module=granitemoehybrid,
        cfg=dict(
            model_type="granitemoehybrid",
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            max_position_embeddings=512,
            num_attention_heads=2,
            num_key_value_heads=1,
            attention_bias=False,
            embedding_multiplier=1.0,
            attention_multiplier=0.1,
            logits_scaling=1.0,
            residual_multiplier=1.0,
            layer_types=["mamba", "attention", "mamba", "mamba"],
            rms_norm_eps=1e-5,
            num_local_experts=4,
            num_experts_per_tok=2,
            shared_intermediate_size=64,
            mamba_n_heads=4,
            mamba_d_head=16,
            mamba_proj_bias=False,
            mamba_d_state=32,  # the Metal step kernel splits the state 32 ways
            mamba_d_conv=4,
            mamba_n_groups=1,
            mamba_conv_bias=True,
        ),
        window=False,
        recurrent=True,
        unquantizable=None,
    ),
    "mistral3": dict(
        module=mistral3,
        cfg=dict(
            model_type="mistral3",
            text_config=dict(
                model_type="mistral",
                hidden_size=64,
                num_hidden_layers=3,
                intermediate_size=128,
                num_attention_heads=2,
                num_key_value_heads=1,
                rms_norm_eps=1e-5,
                vocab_size=64,
                head_dim=32,
            ),
        ),
        window=False,
        recurrent=False,
        unquantizable=None,
    ),
}


def tiny(name, seed=0):
    spec = CLASSES[name]
    mx.random.seed(seed)
    module = spec["module"]
    model = module.Model(module.ModelArgs.from_dict(spec["cfg"]))
    mx.eval(model.parameters())
    return model


def collect(stream):
    return [e.token for e in stream]


def plain(model, prompt, n):
    with Engine(model) as engine:
        return collect(engine.submit(GenerationRequest(prompt, max_tokens=n)))


# Prompts longer than the window, so a sliding layer has rotated before the
# path under test runs.
SYSTEM = [
    3,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    5,
    9,
    15,
    21,
    25,
    27,
    33,
    35,
]
A = [2, 4, 6, 8]
B = [10, 12, 14, 16]


@pytest.mark.parametrize("name", list(CLASSES))
def test_two_rows_decode_like_solo_runs(name):
    model = tiny(name)
    a, b = SYSTEM + A, SYSTEM[:10] + B
    ra, rb = plain(model, a, 6), plain(model, b, 6)
    with Engine(model) as both:
        sa = both.submit(GenerationRequest(a, max_tokens=6))
        sb = both.submit(GenerationRequest(b, max_tokens=6))
        assert collect(sa) == ra and collect(sb) == rb


@pytest.mark.parametrize("name", list(CLASSES))
def test_kv_policy_applies_or_refuses_at_start(name):
    """A layer the quantized kernels cannot serve (MLA keeps the latent
    projection in the cache and reads it back as an array; attention sinks
    have no quantized SDPA) is refused when the engine starts, with the
    reason - not at the first decode step, which used to kill the worker."""
    spec = CLASSES[name]
    model = tiny(name)
    if spec["unquantizable"]:
        engine = Engine(model, kv_policy=KVPolicy(bits=8, group_size=32))
        with pytest.raises(EngineDead, match=spec["unquantizable"]):
            engine.start()
        with Engine(model) as engine:
            kv = engine.health()["kv"]
        assert kv["quantizable_layers"] == 0
        assert any(spec["unquantizable"] in e["reason"] for e in kv["exceptions"])
        return
    with Engine(model, kv_policy=KVPolicy(bits=8, group_size=32)) as engine:
        out = collect(engine.submit(GenerationRequest(SYSTEM, max_tokens=4)))
        kv = engine.health()["kv"]
    assert len(out) == 4
    assert any(a.get("bits") == 8 for a in kv["applied"]), kv["applied"]
    assert kv["quantizable_layers"] > 0
    # Sliding-window and recurrent layers are excepted by design, with a reason.
    excepted = {e["layer"] for e in kv["exceptions"]}
    caches = model.make_cache() if hasattr(model, "make_cache") else []
    expected = {i for i, c in enumerate(caches) if type(c).__name__ != "KVCache"}
    assert excepted == expected


@pytest.mark.parametrize("name", list(CLASSES))
def test_system_prefix_is_reused_across_the_boundary(name):
    """The system block of the first conversation is stored at its boundary;
    the second conversation, sharing it, prefills only its own turn. A
    sliding layer that has rotated past the boundary restores the boundary
    snapshot, like a recurrent layer does - and the answer stays the plain
    one."""
    model = tiny(name)
    n = len(SYSTEM)
    solo = plain(model, SYSTEM + B, 4)
    with Engine(model) as engine:
        collect(
            engine.submit(
                GenerationRequest(
                    SYSTEM + A, max_tokens=4, boundaries=[n], system_end=n
                )
            )
        )
        before = engine.health()["prompt_cache"]
        out = collect(
            engine.submit(
                GenerationRequest(
                    SYSTEM + B, max_tokens=4, boundaries=[n], system_end=n
                )
            )
        )
        after = engine.health()["prompt_cache"]
    assert after["hits"] == before["hits"] + 1
    assert after["tokens_restored"] - before["tokens_restored"] >= n
    assert out == solo


@pytest.mark.parametrize("mode", ["block", "positions", "kernels"])
@pytest.mark.parametrize("name", list(CLASSES))
def test_speculative_verify_matches_plain_greedy(name, mode):
    """Drafts right and wrong across a prompt longer than the window: every
    committed token is the plain greedy token, the rejected drafts are taken
    back from sliding, recurrent and plain layers alike. The `positions`
    mode never feeds a rejected draft and rolls nothing back; it is the
    reference the block mode is measured against."""
    model = tiny(name)
    prompt = SYSTEM + A
    n = 12
    truth = plain(model, prompt, n)
    oracle = OracleProposer(lambda call: [0, 3, 1, 2, 0, 1][call % 6])
    engine = run_speculative(
        model, oracle, exact_verify="off" if mode == "block" else mode
    )
    with engine:
        oracle.start(truth)
        out = collect(engine.submit(GenerationRequest(prompt, max_tokens=n)))
        spec = engine.health()["speculative"]
    assert len(out) == n
    if bit_exact(spec):
        assert out == truth and not oracle.diverged
    assert spec["cycles"] > 0
    assert set(spec["exact"]) >= {
        "width",
        "block_equals_positions",
        "max_abs_logit_diff",
        "argmax_equal",
    }
    if mode == "kernels":
        # The tiny models are not quantized, so nothing gets swapped and the
        # exact path is the per-query attention alone; either the check
        # holds and the mode stays, or it falls back and says so.
        assert spec["mode"] in ("kernels", "block")
        if spec["mode"] == "block":
            assert spec["exact"]["fallback"] == "block" and spec["exact"]["reason"]
    else:
        assert spec["mode"] == mode
