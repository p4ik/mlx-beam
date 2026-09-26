"""Samplers: shared by setting, keyed when seeded, alternatives on request."""

import mlx.core as mx
import pytest

from mlx_beam._vendor.mlx_lm.sample_utils import apply_top_p
from mlx_beam.engine import Engine, GenerationRequest
from mlx_beam.engine.request import SamplingParams
from mlx_beam.engine.sampling import (
    SamplerPool,
    SeededSampler,
    greedy_sampler,
    position_key,
    top_logprobs,
)
from tests.test_engine import collect, tiny_hybrid


def test_samplers_are_shared_by_their_settings():
    pool = SamplerPool(capacity=2)
    a = SamplingParams(temperature=0.7, top_p=0.9, top_k=40)
    b = SamplingParams(temperature=0.7, top_p=0.9, top_k=40)
    c = SamplingParams(temperature=0.7, top_p=0.9, top_k=40, xtc_probability=0.5)
    assert pool.get(a) is pool.get(b)
    # xtc is part of the key; the old sampler without it must not be reused.
    assert pool.get(c) is not pool.get(a)
    assert pool.get(SamplingParams()) is greedy_sampler
    assert pool.describe()["samplers"] <= 2
    seeded = SamplingParams(temperature=0.7, seed=1)
    s1, s2 = pool.get(seeded), pool.get(seeded)
    assert isinstance(s1, SeededSampler) and s1 is not s2
    # A seed on a greedy request changes nothing.
    assert pool.get(SamplingParams(seed=1)) is greedy_sampler


def draws(sampler, n=32):
    logprobs = mx.log(mx.array([[0.3, 0.25, 0.2, 0.15, 0.1]]))
    return [sampler(logprobs).item() for _ in range(n)]


def test_seeded_sampler_ignores_the_global_state():
    p = SamplingParams(temperature=1.0, top_k=4, seed=7)
    mx.random.seed(0)
    first = draws(SeededSampler(p))
    mx.random.seed(123)
    assert draws(SeededSampler(p)) == first
    assert (
        draws(SeededSampler(SamplingParams(temperature=1.0, top_k=4, seed=8))) != first
    )
    # The filters still apply: top_k 2 never draws the tail.
    narrow = SamplingParams(temperature=1.0, top_k=2, seed=7)
    assert set(draws(SeededSampler(narrow))) <= {0, 1}
    xtc = SamplingParams(
        temperature=1.0, xtc_probability=1.0, xtc_threshold=0.12, seed=3
    )
    # XTC with probability 1 removes every candidate above the threshold but
    # the least likely of them: index 3 (0.15) and the tail below stay.
    assert set(draws(SeededSampler(xtc))) == {3, 4}


def chi_square(counts, expected):
    return sum((c - e) ** 2 / e for c, e in zip(counts, expected, strict=True))


def test_seeded_draws_follow_the_filtered_distribution():
    """4096 draws under 4096 position keys against the top-k-filtered,
    tempered distribution: chi-square over the four survivors, and the
    tail never appears. Critical value at 3 degrees of freedom and
    p = 0.001 is 16.27."""
    p = SamplingParams(temperature=0.7, top_k=4, seed=5)
    sampler = SeededSampler(p)
    probs = mx.array([[0.3, 0.25, 0.2, 0.15, 0.1]])
    logprobs = mx.log(probs)
    n = 4096
    tokens = [int(sampler(logprobs).item()) for _ in range(n)]
    assert sampler.position == n
    counts = [tokens.count(t) for t in range(5)]
    assert counts[4] == 0
    kept = probs[0, :4] ** (1 / 0.7)
    expected = (kept / kept.sum() * n).tolist()
    assert chi_square(counts[:4], expected) < 16.27


def test_draws_are_keyed_by_position_not_by_call():
    """The n-th call and `draw(.., n)` are the same token; a sampler that
    skips ahead with `advance` continues where the other one is - so a
    verify cycle that draws several positions at once and a plain step
    that draws one produce the same transcript."""
    p = SamplingParams(temperature=1.0, top_k=4, seed=21)
    logprobs = mx.log(mx.array([[0.3, 0.25, 0.2, 0.15, 0.1]]))
    a, b = SeededSampler(p), SeededSampler(p)
    called = [int(a(logprobs).item()) for _ in range(12)]
    drawn = [int(b.draw(logprobs, n).item()) for n in range(12)]
    assert called == drawn and b.position == 0
    b.advance(7)
    assert int(b(logprobs).item()) == called[7] and b.position == 8
    # A seed given at construction stands in for the request's.
    c = SeededSampler(SamplingParams(temperature=1.0, top_k=4), seed=21)
    assert [int(c(logprobs).item()) for _ in range(12)] == called
    # Distinct keys per (seed, position); neighbouring seeds do not overlap.
    keys = {tuple(position_key(s, n).tolist()) for s in (0, 1, 2) for n in range(4)}
    assert len(keys) == 12


def test_top_logprobs_are_sorted_best_first():
    lp = mx.array([-3.0, -0.5, -2.0, -1.0])
    assert top_logprobs(lp, 3) == ((1, -0.5), (3, -1.0), (2, -2.0))
    assert len(top_logprobs(lp, 10)) == 4


def test_engine_reports_alternatives_only_when_asked():
    model = tiny_hybrid()
    with Engine(model) as engine:
        plain = list(engine.submit(GenerationRequest([3, 7, 11], max_tokens=3)))
        assert all(e.top_logprobs is None for e in plain)
        asked = list(
            engine.submit(GenerationRequest([3, 7, 11], max_tokens=3, top_logprobs=2))
        )
        assert [e.token for e in asked] == [e.token for e in plain]
        for e in asked:
            assert len(e.top_logprobs) == 2
            # Greedy: the sampled token is the best alternative.
            assert e.top_logprobs[0] == (e.token, e.logprob)


def test_seeded_requests_repeat_and_batch_with_shared_samplers():
    model = tiny_hybrid()
    p = SamplingParams(temperature=1.0, top_k=8, seed=11)
    with Engine(model) as engine:
        a = collect(
            engine.submit(GenerationRequest([3, 7, 11], max_tokens=6, sampling=p))
        )
        b = collect(
            engine.submit(GenerationRequest([3, 7, 11], max_tokens=6, sampling=p))
        )
        assert a == b
        shared = SamplingParams(temperature=1.0, top_k=8)
        streams = [
            engine.submit(GenerationRequest([3, 7, 11], max_tokens=6, sampling=p)),
            engine.submit(GenerationRequest([5, 9], max_tokens=6, sampling=shared)),
            engine.submit(GenerationRequest([5, 9, 13], max_tokens=6, sampling=shared)),
        ]
        outs = [collect(s) for s in streams]
        assert all(len(o) == 6 for o in outs)
        assert engine.health()["sampling"]["samplers"] == 1


def test_ruled_out_tokens_are_no_alternatives():
    import json

    lp = mx.array([0.0, -mx.inf, -mx.inf, -1.5])
    out = top_logprobs(lp, 3)
    assert out == ((0, 0.0), (3, -1.5))
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize(
    "dtype,top_p",
    [
        (mx.float32, 0.1),
        (mx.float32, 1e-8),
        (mx.float16, 1e-4),
        (mx.bfloat16, 1e-3),
    ],
)
def test_top_p_always_keeps_the_most_likely_token(dtype, top_p):
    """A tiny top_p, or bfloat16 where `1 - top_p` rounds to 1.0, must not
    empty the candidate set: the most likely token stays, and a seeded
    draw picks it rather than falling back on token 0."""
    logits = mx.array([[0.0, 1.0, 4.0, 2.0]], dtype=dtype)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    filtered = apply_top_p(logprobs, top_p)
    assert mx.isfinite(filtered[0, 2]).item()
    assert not mx.isfinite(filtered[0, 0]).item()
    draw = SeededSampler(SamplingParams(temperature=1.0, top_p=top_p, seed=7))
    assert draw(logprobs).item() == 2


def test_a_vanishing_temperature_is_greedy_not_random():
    """1/temp overflows float32 below ~1e-38; such a request means greedy."""
    from mlx_beam.engine.sampling import SamplerPool, sampler_key

    pool = SamplerPool()
    tiny = SamplingParams(temperature=1e-40, seed=7)
    assert sampler_key(tiny) == ("greedy",)
    sampler = pool.get(tiny)
    logits = mx.array([[0.0, 0.0, 3.0, 0.0]])
    assert all(int(sampler(logits).item()) == 2 for _ in range(20))
