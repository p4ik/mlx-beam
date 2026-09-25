"""The prefill valve with injected memory readers: it learns bytes per
token from the calls that move the peak, cuts a call to the width that
fits under the ceiling, stalls below the narrowest, and lets everything
through while it knows nothing."""

from mlx_beam._vendor.mlx_lm.generate import BatchGenerator
from mlx_beam.engine.valve import GRID, MIN_WIDTH, PrefillValve
from tests.test_engine import tiny_hybrid


class Memory:
    def __init__(self, active=1_000, peak=1_000, free=10_000, recommended=None):
        self.active, self.peak, self.free = active, peak, free
        self.recommended = recommended

    def valve(self):
        ticks = iter(range(0, 10_000, 2))  # every reading is past the cache's TTL
        return PrefillValve(
            active=lambda: self.active,
            peak=lambda: self.peak,
            free=lambda: self.free,
            recommended=self.recommended,
            clock=lambda: float(next(ticks)),
        )


def test_learns_from_calls_that_move_the_peak_and_cuts_to_what_fits():
    mem = Memory(active=1_000, peak=1_000, free=100_000, recommended=None)
    v = mem.valve()
    # Nothing known: the call goes through as asked, and is measured.
    assert v.width(512) == 512
    mem.peak = 1_000 + 512 * 10  # the call's transient: 10 bytes a token
    v.observe(512)
    assert v.bytes_per_token == 10 and v.samples == 1
    # A call that does not move the peak teaches nothing.
    assert v.width(128) == 128
    v.observe(128)
    assert v.samples == 1
    # Ceiling = held + free: room for 100_000 / 10 tokens, more than asked.
    assert v.width(2048) == 2048 and v.last_ceiling == 101_000
    # Less free memory: 3000 bytes of room fit 300 tokens, on the grid 256.
    mem.free = 3_000
    assert v.width(2048) == 256 and v.shrunk_calls == 1
    # Below the narrowest width: stall.
    mem.free = 500
    assert v.width(2048) is None and v.stalled_calls == 1
    assert v.describe()["last_width"] is None
    # The recommended working set caps the ceiling too.
    mem.free = 1_000_000
    v.recommended = 1_000 + 640 * 10
    assert v.width(2048) == 640 // GRID * GRID


def test_smoothing_and_unknown_readers():
    mem = Memory(active=0, peak=0, free=None)
    v = mem.valve()
    v.width(100)
    mem.peak = 1_000
    v.observe(100)
    v.width(100)
    mem.peak = 3_000  # a bigger transient: 30 a token against 10 before
    v.observe(100)
    assert 10 < v.bytes_per_token < 30 and v.samples == 2
    # No free reading and no recommended size: nothing to hold against.
    assert v.ceiling() is None and v.width(4096) == 4096
    assert MIN_WIDTH == GRID


def test_generator_stalls_the_prefill_round_and_resumes():
    """The generator asks the valve before every prefill call; a stalled
    round prefills nobody, and the request still completes once the valve
    lets it through - in the slices the valve allows."""
    model = tiny_hybrid()
    mem = Memory(active=1_000, peak=1_000, free=1_000_000)
    v = mem.valve()
    v.bytes_per_token = 1_000  # as if learned: a token costs a thousand
    gen = BatchGenerator(model, prefill_valve=v, prefill_slice=4, prefill_step_size=4)
    (uid,) = gen.insert([[3, 7, 11, 13, 5, 9, 2, 8]], max_tokens=[3])
    mem.free = 10  # nothing fits: the round stalls
    for _ in range(3):
        prompt_responses, generated = gen.next()
        assert not prompt_responses and not generated
    assert gen.stalled_calls == 3 and v.stalled_calls == 3
    mem.free = 1_000_000  # room again: the prefill goes on
    tokens = []
    for _ in range(40):
        _, generated = gen.next()
        tokens += [r.token for r in generated if r.uid == uid]
        if any(r.finish_reason for r in generated):
            break
    assert len(tokens) == 3 and v.shrunk_calls == 0
    gen.close()
