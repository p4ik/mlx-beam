"""Samplers: shared by setting, keyed when seeded, alternatives on request."""

import mlx.core as mx

from mlx_beam.engine import Engine, GenerationRequest
from mlx_beam.engine.request import SamplingParams
from mlx_beam.engine.sampling import (
    SamplerPool,
    SeededSampler,
    greedy_sampler,
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
    assert draws(SeededSampler(SamplingParams(temperature=1.0, top_k=4, seed=8))) != first
    # The filters still apply: top_k 2 never draws the tail.
    narrow = SamplingParams(temperature=1.0, top_k=2, seed=7)
    assert set(draws(SeededSampler(narrow))) <= {0, 1}
    xtc = SamplingParams(temperature=1.0, xtc_probability=1.0, xtc_threshold=0.12, seed=3)
    # XTC with probability 1 removes every candidate above the threshold but
    # the least likely of them: index 3 (0.15) and the tail below stay.
    assert set(draws(SeededSampler(xtc))) == {3, 4}


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
        a = collect(engine.submit(GenerationRequest([3, 7, 11], max_tokens=6, sampling=p)))
        b = collect(engine.submit(GenerationRequest([3, 7, 11], max_tokens=6, sampling=p)))
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
