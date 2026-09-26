"""Multimodal positions (Qwen's MRoPE): the text model rotates every token
with three positions - time, height, width - one per frequency by the
interleaved layout. Text tokens carry the same value on all three axes and
get the fast kernel's numbers; an image's tokens spread over the height
and width axes, the text after it continues from the block's largest
position, fewer than its token count. The engine hands the prompt's
positions in and shifts the decode by the delta they leave."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_beam._vendor.mlx_lm.models import qwen3_5, qwen3_vl
from mlx_beam._vendor.mlx_lm.models.mrope import (
    apply_mrope,
    interleaved_selector,
    mrope_section_of,
)
from mlx_beam.engine import Engine, GenerationRequest
from tests.test_engine import TINY_QWEN35_WRAPPED, collect

# Two text tokens, one image of a 2 x 2 grid (a [1, 4, 4] grid merged by 2),
# two text tokens: the reference's positions and the decode delta.
POSITIONS = np.array(
    [
        [0, 1, 2, 2, 2, 2, 4, 5],
        [0, 1, 2, 2, 3, 3, 4, 5],
        [0, 1, 2, 3, 2, 3, 4, 5],
    ],
    dtype=np.int32,
)
DELTA = 5 + 1 - 8


def test_the_selector_reads_the_sections():
    # freq 0, 3, 6 read time; 1, 4, 7 height (three); 2, 5 width (two)
    assert interleaved_selector([3, 3, 2], 8).tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
    assert mrope_section_of({"mrope_section": [3, 3, 2], "rope_type": "default"}) == [
        3,
        3,
        2,
    ]
    assert (
        mrope_section_of({"mrope_section": [3, 3, 2], "mrope_interleaved": False})
        is None
    )
    assert (
        mrope_section_of({"rope_type": "yarn"}) is None
        and mrope_section_of(None) is None
    )


def test_text_positions_give_the_fast_kernels_numbers():
    rope = nn.RoPE(16, traditional=False, base=500000.0)
    selector = interleaved_selector([3, 3, 2], 8)
    x = mx.random.normal((1, 2, 8, 16))
    positions = mx.tile(mx.arange(5, 13, dtype=mx.int32)[None, None], (3, 1, 1))
    ours = apply_mrope(x, positions, rope, selector)
    fast = rope(x, offset=5)
    assert mx.abs(ours - fast).max().item() < 1e-5
    # A partial rotary factor: only the first dims turn, the rest pass.
    partial = nn.RoPE(8, traditional=False, base=500000.0)
    ours = apply_mrope(x, positions, partial, interleaved_selector([2, 1, 1], 4))
    assert mx.abs(ours - partial(x, offset=5)).max().item() < 1e-5
    assert mx.array_equal(ours[..., 8:], x[..., 8:])


def test_image_positions_match_the_transformers_reference():
    torch = pytest.importorskip("torch")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    official = Qwen3VLTextRotaryEmbedding(
        Qwen3VLTextConfig(
            hidden_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 500000.0,
                "mrope_section": [3, 3, 2],
            },
        )
    )
    q = torch.arange(1, 1 + 2 * 8 * 16, dtype=torch.float32).reshape(1, 2, 8, 16) / 256
    positions = torch.tensor(POSITIONS)[:, None, :]
    cos, sin = official(q, positions)
    expected, _ = apply_rotary_pos_emb(q, q, cos, sin)
    rope = nn.RoPE(16, traditional=False, base=500000.0)
    ours = apply_mrope(
        mx.array(q.numpy()),
        mx.array(POSITIONS)[:, None, :],
        rope,
        interleaved_selector([3, 3, 2], 8),
    )
    np.testing.assert_allclose(np.array(ours), expected.numpy(), atol=2e-6, rtol=2e-6)


def test_the_vision_text_models_carry_the_layout():
    cfg = dict(
        model_type="qwen3",
        hidden_size=32,
        num_hidden_layers=1,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        max_position_embeddings=128,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
        },
        tie_word_embeddings=True,
    )
    model = qwen3_vl.Model(qwen3_vl.ModelArgs("qwen3_vl", cfg))
    attn = model.layers[0].self_attn
    assert attn.mrope_selector.tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
    hybrid = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(TINY_QWEN35_WRAPPED))
    attention_layers = [layer for layer in hybrid.layers if not layer.is_linear]
    assert attention_layers[0].self_attn.mrope_selector is not None
    # Without the layout, explicit positions are refused, not silently 1D.
    plain = qwen3_vl.Model(
        qwen3_vl.ModelArgs("qwen3_vl", {**cfg, "rope_scaling": None})
    )
    with pytest.raises(ValueError, match="no interleaved MRoPE"):
        plain(mx.array([[1, 2, 3]]), position_ids=mx.zeros((3, 1, 3), dtype=mx.int32))


VL_CONFIG = dict(
    model_type="qwen3",
    hidden_size=32,
    num_hidden_layers=2,
    intermediate_size=64,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    rms_norm_eps=1e-6,
    vocab_size=64,
    max_position_embeddings=128,
    rope_theta=500000.0,
    rope_scaling={
        "rope_type": "default",
        "mrope_section": [3, 3, 2],
        "mrope_interleaved": True,
    },
    tie_word_embeddings=True,
)
TOKENS = [3, 4, 62, 62, 62, 62, 5, 6]


def _reference_greedy(model, tokens, positions, n):
    """The model called with explicit positions for the prompt and for each
    generated token, continuing from the largest prompt position - or, with
    `positions` None, the plain path at every token's index; the tokens and
    their logprobs."""
    from mlx_beam._vendor.mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)

    def step(ids, pos):
        kw = {} if pos is None else {"position_ids": pos}
        return model(mx.array([ids]), cache=cache, **kw)

    logits = step(
        tokens, None if positions is None else mx.array(positions)[:, None, :]
    )
    out, logprobs = [], []
    next_pos = None if positions is None else int(positions.max()) + 1
    for _ in range(n):
        lp = logits[0, -1] - mx.logsumexp(logits[0, -1])
        token = int(mx.argmax(lp).item())
        out.append(token)
        logprobs.append(lp[token].item())
        pos = None if next_pos is None else mx.full((3, 1, 1), next_pos, dtype=mx.int32)
        logits = step([token], pos)
        if next_pos is not None:
            next_pos += 1
    return out, logprobs


def test_the_engine_rotates_by_the_positions_and_decodes_from_the_delta():
    # Seed 5: a tiny model whose greedy answer depends on the positions.
    mx.random.seed(5)
    model = qwen3_vl.Model(qwen3_vl.ModelArgs("qwen3_vl", VL_CONFIG))
    mx.eval(model.parameters())
    expected, expected_lp = _reference_greedy(model, TOKENS, POSITIONS, 6)
    plain_expected, _ = _reference_greedy(model, TOKENS, None, 6)
    assert (
        expected != plain_expected
    ), "the positions must matter for this check to mean anything"
    # Another prompt for the text row: the store keys on tokens and image
    # digests, and a real text prompt never carries the image's placeholders.
    text = [3, 4, 7, 8, 9, 10, 5, 6]
    text_expected, _ = _reference_greedy(model, text, None, 6)
    with Engine(model) as engine:
        assert engine.image_capabilities["position_ids"]
        events = list(
            engine.submit(
                GenerationRequest(
                    TOKENS,
                    max_tokens=6,
                    positions=mx.array(POSITIONS),
                    rope_delta=DELTA,
                )
            )
        )
        assert [e.token for e in events] == expected
        np.testing.assert_allclose([e.logprob for e in events], expected_lp, atol=1e-4)
        # The same prompt again: the store restores it, the delta still applies.
        again = collect(
            engine.submit(
                GenerationRequest(
                    TOKENS,
                    max_tokens=6,
                    positions=mx.array(POSITIONS),
                    rope_delta=DELTA,
                )
            )
        )
        assert again == expected and engine.health()["prompt_cache"]["hits"] >= 1
        # Without the delta the decode continues from the token count: wrong.
        without = collect(
            engine.submit(
                GenerationRequest(TOKENS, max_tokens=6, positions=mx.array(POSITIONS))
            )
        )
        assert without != expected
        # A text prompt is the plain path, and an image row beside a text
        # row in one batch each get their own positions and delta.
        streams = [
            engine.submit(
                GenerationRequest(
                    TOKENS,
                    max_tokens=6,
                    positions=mx.array(POSITIONS),
                    rope_delta=DELTA,
                )
            ),
            engine.submit(GenerationRequest(text, max_tokens=6)),
        ]
        assert [collect(s) for s in streams] == [expected, text_expected]
    with pytest.raises(ValueError, match="shape"):
        GenerationRequest(TOKENS, positions=mx.zeros((3, 4), dtype=mx.int32))


def test_the_hybrid_trunk_takes_the_positions_too():
    """Qwen3.5/3.8: attention every other layer, a partial rotary factor;
    the positions reach the attention layers, the delta the decode."""
    mx.random.seed(3)
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(TINY_QWEN35_WRAPPED))
    mx.eval(model.parameters())
    expected, expected_lp = _reference_greedy(model, TOKENS, POSITIONS, 4)
    with Engine(model) as engine:
        events = list(
            engine.submit(
                GenerationRequest(
                    TOKENS,
                    max_tokens=4,
                    positions=mx.array(POSITIONS),
                    rope_delta=DELTA,
                )
            )
        )
    assert [e.token for e in events] == expected
    np.testing.assert_allclose([e.logprob for e in events], expected_lp, atol=1e-4)
