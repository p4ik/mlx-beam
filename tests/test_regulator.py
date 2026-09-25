"""The depth regulator: rates from measured acceptance and cost, parking
on losses with a doubling cooldown, and the engine following it."""

import pytest

from mlx_beam.engine import Engine, GenerationRequest
from mlx_beam.engine.regulator import (
    COOLDOWN_MAX,
    COOLDOWN_MIN,
    HEAD_SHARE,
    PARK_AFTER,
    PROBE_EVERY,
    Regulator,
)
from tests.test_speculative import (
    OracleProposer,
    collect,
    plain_transcript,
    run_speculative,
    tiny_qwen35,
)


def loaded(cap=3, plain=1.0, acceptance=(0.9, 0.8, 0.7), costs=None):
    """A regulator with a frozen cost model."""
    reg = Regulator(cap, measure=False)
    reg.plain_cost = plain
    reg.acceptance = list(acceptance)
    reg.cycle_cost = dict(costs or {})
    return reg


def test_expected_tokens_and_rates_follow_the_chain():
    reg = loaded(costs={1: 1.06, 2: 1.12, 3: 1.18})
    assert reg.expected_committed(0) == 1.0
    assert reg.expected_committed(3) == pytest.approx(1 + 0.9 + 0.72 + 0.504)
    assert reg.rate(0) == 1.0
    assert reg.rate(3) == pytest.approx(3.124 / 1.18)
    # Unmeasured depth: the plain step plus its head calls.
    reg.cycle_cost.pop(2)
    assert reg.cost(2) == pytest.approx(1 + 2 * HEAD_SHARE)
    assert reg.choose() == 3
    # A chain that dies at the first position still adds a little at
    # depth 3 for cheap head calls; a dear depth 3 makes the shallow
    # cycle best.
    reg.acceptance = [0.2, 0.9, 0.9]
    assert reg.choose() == 3
    reg.cycle_cost.update({2: 1.5, 3: 1.6})
    assert reg.choose() == 1


def test_nothing_measured_runs_the_cap_and_measures():
    reg = Regulator(3)
    assert reg.cost(1) is None and reg.rate(1) is None
    assert reg.choose() == 3
    reg.observe_plain(0.010)
    reg.observe_cycle(3, 2, 0.012)
    assert reg.plain_cost == 0.010 and reg.cycle_cost == {3: 0.012}
    # First observation sets a position, later ones smooth it; the chain
    # broke at position 3, so it stays at its prior.
    assert reg.acceptance[:2] == [1.0, 1.0] and reg.acceptance[2] == 0.0
    reg.observe_cycle(3, 0, 0.012)
    assert reg.acceptance[0] < 1.0 and reg.acceptance[1] == 1.0
    assert reg.tokens_saved == 2


def test_losses_park_and_the_cooldown_doubles_until_a_win():
    reg = loaded(costs={3: 1.2})
    for _ in range(PARK_AFTER - 1):
        reg.observe_cycle(3, 0, 1.2)  # one token in 1.2 steps' time: a loss
    assert not reg.parked and reg.losses == PARK_AFTER - 1
    reg.observe_cycle(3, 0, 1.2)
    assert reg.parked and reg.choose() == 0 and reg.parks == 1
    assert reg.describe()["cooldown_left"] == COOLDOWN_MIN
    for _ in range(COOLDOWN_MIN):
        reg.observe_plain(1.0)
    # Back in business - at the depth the lowered acceptance now favours.
    assert not reg.parked and reg.choose() >= 1
    for _ in range(PARK_AFTER):
        reg.observe_cycle(3, 0, 1.2)
    assert reg.parked and reg.cooldown == 2 * COOLDOWN_MIN
    for _ in range(2 * COOLDOWN_MIN):
        reg.observe_plain(1.0)
    # A winning cycle resets the ladder.
    reg.observe_cycle(3, 2, 1.2)
    assert reg.losses == 0
    for _ in range(PARK_AFTER):
        reg.observe_cycle(3, 0, 1.2)
    assert reg.cooldown == COOLDOWN_MIN
    for _ in range(20):
        for _ in range(reg.cooldown):
            reg.observe_plain(1.0)
        for _ in range(PARK_AFTER):
            reg.observe_cycle(3, 0, 1.2)
    assert reg.cooldown == COOLDOWN_MAX


def test_a_neighbour_is_probed_now_and_then():
    reg = loaded(costs={1: 1.06, 2: 1.12, 3: 1.18})
    depths = [reg.choose() for _ in range(2 * PROBE_EVERY)]
    assert depths.count(3) == 2 * PROBE_EVERY - 2 and 2 in depths


def test_fixed_depth_neither_regulates_nor_parks():
    reg = loaded(costs={3: 1.2})
    reg.fixed = 2
    for _ in range(2 * PARK_AFTER):
        assert reg.choose() == 2
        reg.observe_cycle(2, 0, 1.2)
    assert not reg.parked


def test_engine_parks_on_a_losing_proposer_and_returns_after_the_cooldown():
    model = tiny_qwen35()
    prompt = [3, 7, 11, 13]
    n = PARK_AFTER + COOLDOWN_MIN + 8
    truth = plain_transcript(model, prompt, n)
    oracle = OracleProposer(lambda call: 0)  # never a draft that holds
    engine = run_speculative(model, oracle)
    with engine:
        reg = engine.speculator.regulator
        reg.measure = False
        reg.plain_cost, reg.cycle_cost = 1.0, {3: 1.2}
        oracle.start(truth)
        out = collect(engine.submit(GenerationRequest(prompt, max_tokens=n)))
        h = engine.health()["speculative"]
    assert out == truth
    # PARK_AFTER losing cycles, then plain steps through the cooldown, then
    # cycles again - and the row still ends with the plain transcript.
    assert h["regulator"]["parks"] == 1
    assert h["plain_steps"] >= COOLDOWN_MIN
    assert h["cycles"] > PARK_AFTER
    assert h["regulator"]["acceptance_by_position"][0] < 0.5
    assert 1 <= h["depth"] <= 3 and h["max_depth"] == 3
    assert h["reason"] is None and not h["parked"]
