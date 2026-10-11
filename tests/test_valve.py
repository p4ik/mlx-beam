"""The prefill valve with injected memory readers: it learns bytes per
token from the calls that move the peak, cuts a call to the width that
fits under the ceiling, stalls below the narrowest, and lets everything
through while it knows nothing."""

import contextlib
import threading

import pytest

from mlx_beam._vendor.mlx_lm.generate import BatchGenerator
from mlx_beam.engine.valve import EMA, GRID, MIN_WIDTH, PrefillValve
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
    # recommended=0: no device size known - on a Mac the default would read
    # the device's own working set and there would be a ceiling after all.
    mem = Memory(active=0, peak=0, free=None, recommended=0)
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


class Stalled:
    """A valve that never finds room: every width is None."""

    stalled_calls = 0
    shrunk_calls = 0
    last_width = None
    last_ceiling = 0
    recommended = 0
    bytes_per_token = 1.0
    samples = 0

    def width(self, wanted):
        self.stalled_calls += 1
        return None

    def peek(self, wanted):
        return None

    def observe(self, tokens):
        pass

    def describe(self):
        return {}

    def round(self):
        return contextlib.nullcontext()  # no memory to read


def test_the_query_without_side_effects_answers_as_width_would():
    mem = Memory(active=1_000, peak=1_000, free=100_000, recommended=None)
    v = mem.valve()
    assert hasattr(v, "peek")  # the query is the fix
    assert v.peek(512) == 512  # nothing known: everything goes through
    v.bytes_per_token = 10.0
    mem.free = 3_000
    assert v.peek(2048) == 256 and v.peek(200) == 200
    mem.free = 500
    assert v.peek(2048) is None
    # Nothing counted, nothing recorded for observe: the call may never run.
    assert v.shrunk_calls == 0 and v.stalled_calls == 0
    assert v.last_width is None and v._before is None
    assert v.width(2048) is None and v.stalled_calls == 1


def test_a_round_answers_every_query_and_width_from_one_reading():
    """Inside `round()` what is held is read once: a rise between a query
    and the width leaves the width's answer as the query's, `observe` still
    measures the call over what was held when it ran, and the reading
    expires with the round, also one that ends early."""
    mem = Memory(active=1_000, peak=1_000, free=1_000_000, recommended=3_560)
    v = mem.valve()
    v.bytes_per_token = 10.0
    assert hasattr(v, "round")  # one reading a round is the fix
    with v.round():
        assert v.peek(256) == 256  # room for 2_560 bytes: 256 tokens
        mem.active = 3_000  # another thread allocated in between
        assert v.width(256) == 256 and v.peek(256) == 256
        assert v.stalled_calls == 0 and v.shrunk_calls == 0
        mem.peak = 3_000 + 256 * 20
        v.observe(256)
    # The sample is the peak's rise over the 3_000 held when the call ran.
    assert v.bytes_per_token == 10 + EMA * (20 - 10) and v.samples == 1
    # Outside the round every reading is fresh: 560 bytes of room fit no
    # call of the narrowest width.
    assert v.width(256) is None and v.stalled_calls == 1
    mem.active = 1_000  # room for 2_560 bytes again: 204 tokens, 192 on the grid
    with pytest.raises(RuntimeError), v.round():
        assert v.peek(192) == 192
        raise RuntimeError("the round ends early")
    mem.active = 3_000
    assert v.peek(192) is None  # no reading left behind


def test_a_stall_that_cannot_end_evicts_the_store_then_refuses_the_prompt():
    """Nobody decodes, the valve finds no room: first the store gives up
    its entries (memory the prefill may have), then the widest prompt is
    refused with a 400 instead of holding the worker for ever - and the
    engine goes on serving."""
    from mlx_beam.engine import ContextTooLong, Engine, GenerationRequest

    model = tiny_hybrid()
    with Engine(model, prefill_slice=2) as engine:
        # One entry in the store, to be given up first.
        list(engine.submit(GenerationRequest([3, 7, 11], max_tokens=2)))
        assert engine.prefix_store.describe()["entries"] == 1
        engine.prefill_valve = Stalled()
        engine._gen.prefill_valve = engine.prefill_valve
        stream = engine.submit(GenerationRequest(list(range(1, 40)), max_tokens=2))
        with pytest.raises(ContextTooLong, match="does not fit in memory"):
            for _ in stream:
                pass
        h = engine.health()
        assert h["alive"] and h["prefill_valve"]["refused"] == 1
        assert engine.prefix_store.describe()["evicted_for_memory"] == 1
        assert engine.prefix_store.describe()["entries"] == 0
        # Room again: the next request runs.
        engine.prefill_valve = engine._gen.prefill_valve = PrefillValve(
            active=lambda: 0, peak=lambda: 0, free=lambda: None, recommended=0
        )
        assert len(list(engine.submit(GenerationRequest([3, 7], max_tokens=2)))) == 2


class Armed(Stalled):
    """Lets every width through until a request decodes; from then on it
    finds no room and notes, per `width()` call, whether something was
    decoding and how many entries the store held. Every call waits for
    `gate`: the test opens it once both requests are submitted, so the
    second is in the engine before the first has prefilled."""

    def __init__(self, engine):
        self.engine = engine
        self.armed = False
        self.calls = []
        self.gate = threading.Event()

    def width(self, wanted):
        assert self.gate.wait(10), "the gate was never opened"
        gen = self.engine._gen
        if not self.armed and len(gen._generation_batch) == 0:
            return wanted
        self.armed = True
        self.calls.append(
            (len(gen._generation_batch) > 0, len(self.engine.prefix_store))
        )
        return super().width(wanted)


def test_a_valve_stall_is_refused_only_after_the_last_decode_and_a_failed_prefill():
    """Rounds stalled beside a decode, or rounds that free a store entry,
    do not count towards the refusal: the refusal comes only after the
    last decode has ended, its entry has been given up, and STALL_ROUNDS
    prefill attempts failed with nothing decoding and the store empty -
    never before a prefill call has seen the memory the finished request
    frees."""
    from mlx_beam.engine import ContextTooLong, Engine, GenerationRequest
    from mlx_beam.engine.core import STALL_ROUNDS

    model = tiny_hybrid()
    with Engine(model, prefill_slice=2, prompt_concurrency=1) as engine:
        list(engine.submit(GenerationRequest([3, 7, 11], max_tokens=2)))
        assert engine.prefix_store.describe()["entries"] == 1
        valve = Armed(engine)
        engine.prefill_valve = engine._gen.prefill_valve = valve
        # One prompt at a time, and the valve waits for both submits: the
        # second is admitted once the first has left the prompt batch, so
        # its attempts run beside the first's decode, then after it.
        first = engine.submit(GenerationRequest([5, 9, 13, 17, 19], max_tokens=20))
        second = engine.submit(GenerationRequest(list(range(1, 40)), max_tokens=2))
        valve.gate.set()
        assert len(list(first)) == 20
        with pytest.raises(ContextTooLong, match="does not fit in memory"):
            for _ in second:
                pass
        assert engine.health()["prefill_valve"]["refused"] == 1
    decoding = [i for i, (busy, _) in enumerate(valve.calls) if busy]
    # The second prompt stalled beside the first's decode for more than
    # STALL_ROUNDS attempts and was not refused there.
    assert len(decoding) > STALL_ROUNDS
    tail = valve.calls[decoding[-1] + 1 :]
    # After the last decode: the stalled rounds first give up what it left
    # in the store, then exactly STALL_ROUNDS attempts with nothing
    # decoding and the store empty precede the refusal.
    assert all(entries > 0 for _, entries in tail[:-STALL_ROUNDS])
    assert tail[-STALL_ROUNDS:] == [(False, 0)] * STALL_ROUNDS
