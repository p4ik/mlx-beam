"""The reasoning budget: admission arithmetic, the forced close, the flags."""

import mlx.core as mx
import pytest

from mlx_beam.engine import ContextTooLong, Engine, GenerationRequest, ReasoningLimits
from mlx_beam.engine.thinking import ThinkingBudget, budget
from tests.test_engine import collect, tiny_hybrid

START, END, NL, NL2 = 50, 51, 52, 53
LIMITS = ReasoningLimits(start=(START,), end=(END,), close=(NL, END, NL2))


def test_budget_serves_small_limits_and_refuses_a_reserve_that_cannot_fit():
    # No context known: the request's own numbers.
    assert budget(None, 100, 80, 0) == (80, None)
    assert budget(None, 100, 80, 30) == (80, 50)
    # The examples the rule was written for: free 100, reserve 512.
    assert budget(200, 100, 80, 512) == (80, 0)
    with pytest.raises(ContextTooLong) as exc:
        budget(200, 100, 1000, 512)
    assert "100" in str(exc.value) and "200" in str(exc.value)
    # A large limit is capped, not refused, when the reserve fits.
    assert budget(200, 100, 1000, 40) == (100, 60)
    assert budget(200, 100, 1000, 0) == (100, None)
    # A prompt that fills the context has no room at all.
    with pytest.raises(ContextTooLong):
        budget(200, 200, 10, 0)


class Steps:
    """Drives a ThinkingBudget the way the generator does: the processor runs
    one token ahead of what observe() has seen."""

    def __init__(self, tracker, model_tokens, vocab=64):
        self.tracker = tracker
        self.model = list(model_tokens)
        self.vocab = vocab
        self.out = []
        # What the tracker said about each token after observing it.
        self.forced = []

    def run(self, n):
        context = [0]
        for _ in range(n):
            # What the model would say next, as logits with one clear winner.
            wanted = self.model[len(self.out)] if len(self.out) < len(self.model) else 1
            logits = mx.where(mx.arange(self.vocab) == wanted, 5.0, 0.0)
            logits = self.tracker(mx.array(context), logits)
            token = int(mx.argmax(logits).item())
            self.out.append(token)
            context.append(token)
            # Python sees the previous token only now, one step behind.
            if len(self.out) >= 2:
                self.tracker.observe(self.out[-2], None)
                self.forced.append(self.tracker.last_forced)
        if self.out:
            self.tracker.observe(self.out[-1], None)
            self.forced.append(self.tracker.last_forced)
        return self.out


def test_forced_close_lands_exactly_at_the_budget():
    limits = ReasoningLimits(**{**LIMITS.__dict__, "max_tokens": 5})
    tracker = ThinkingBudget(limits)
    # The model opens a block and never closes it.
    out = Steps(tracker, [START, 10, 11, 12, 13, 14, 15, 16, 17]).run(9)
    # <think> (1), three free tokens (4), the forced newline (5), then the
    # marker and the blank line; the answer follows.
    assert out[:7] == [START, 10, 11, 12, NL, END, NL2] and out[7] not in (START, END)
    assert tracker.reasoning_tokens == 5 and tracker.thinking_truncated
    # The opener is masked afterwards: a model that wants to reopen cannot.
    tracker2 = ThinkingBudget(limits)
    out = Steps(tracker2, [START, 10, 11, 12, 13, 14, 15, START, 20]).run(9)
    assert out[7] != START and out[8] == 20


def test_a_block_the_model_closes_itself_is_not_forced():
    limits = ReasoningLimits(**{**LIMITS.__dict__, "max_tokens": 5})
    tracker = ThinkingBudget(limits)
    # The close arrives on the free token, right where the force would start.
    out = Steps(tracker, [START, 10, 11, END, 20, 21, 22, 23]).run(8)
    assert out == [START, 10, 11, END, 20, 21, 22, 23]
    assert not tracker.thinking_truncated and tracker.reasoning_tokens == 3


def test_seeded_prompt_and_zero_budget():
    seeded = ReasoningLimits(**{**LIMITS.__dict__, "seeded": True, "max_tokens": 0})
    tracker = ThinkingBudget(seeded)
    # Budget 0 on an open block: the close is forced from the first token.
    out = Steps(tracker, [10, 11, 12, 13, 14]).run(5)
    assert out == [NL, END, NL2, 13, 14] and tracker.thinking_truncated
    # Budget 0 on a closed prompt: the opener is masked, no block ever opens.
    tracker = ThinkingBudget(ReasoningLimits(**{**LIMITS.__dict__, "max_tokens": 0}))
    out = Steps(tracker, [START, 10, 11]).run(3)
    assert START not in out and tracker.reasoning_tokens == 0
    # No budget: nothing is forced, the block is only counted.
    tracker = ThinkingBudget(LIMITS)
    out = Steps(tracker, [START, 10, 11, 12, END, 13]).run(6)
    assert out == [START, 10, 11, 12, END, 13] and tracker.reasoning_tokens == 4


def test_length_inside_the_block_flags_the_thinking():
    tracker = ThinkingBudget(LIMITS)
    tracker.observe(START, None)
    tracker.observe(10, "length")
    assert tracker.thinking_truncated and tracker.in_reasoning


def test_engine_forces_the_close_and_reports_the_flags():
    model = tiny_hybrid()
    with Engine(model) as engine:
        free = collect(engine.submit(GenerationRequest([3, 7, 11], max_tokens=8)))
        # Marker ids the greedy run never emits, so nothing closes by itself.
        unused = [t for t in range(63, 0, -1) if t not in free]
        start, end, nl, nl2 = unused[:4]
        limits = ReasoningLimits((start,), (end,), (nl, end, nl2), seeded=True)
        req = GenerationRequest(
            [3, 7, 11], max_tokens=8, reasoning=limits, min_response_tokens=3
        )
        events = list(engine.submit(req))
        tokens = [e.token for e in events]
        # 8 to generate, 3 kept for the answer, 2 for the close's tail: the
        # block gets 3, the last of them the forced newline, then the marker
        # and the blank line, then the 3 answer tokens.
        assert tokens[2:5] == [nl, end, nl2] and len(tokens) == 8
        assert tokens[:2] == free[:2]
        # The close's tail after the marker is the queue's as well.
        assert [e.forced for e in events] == [False, False, True, True, True] + [
            False
        ] * 3
        assert events[-1].thinking_truncated and events[-1].finish_reason == "length"
        assert events[-1].response_truncated
        # A client stop on the marker ends the request there, flagged.
        req = GenerationRequest(
            [3, 7, 11],
            max_tokens=8,
            reasoning=limits,
            min_response_tokens=3,
            stop_sequences=[(end,)],
        )
        events = list(engine.submit(req))
        assert [e.token for e in events][-1] == end
        assert events[-1].finish_reason == "stop" and events[-1].thinking_truncated
        assert not events[-1].response_truncated
        # Without a budget the same request runs free.
        plain = collect(engine.submit(GenerationRequest([3, 7, 11], max_tokens=8)))
        assert plain == free


def test_prompt_cap_is_a_limit_the_request_may_only_lower():
    model = tiny_hybrid()
    with Engine(model, max_prompt_tokens=4) as engine:
        collect(engine.submit(GenerationRequest([3, 7, 11], max_tokens=2)))
        with pytest.raises(ContextTooLong):
            engine.submit(GenerationRequest([3, 7, 11, 13, 17], max_tokens=2))
        with pytest.raises(ContextTooLong):
            engine.submit(
                GenerationRequest([3, 7, 11], max_tokens=2, max_prompt_tokens=2)
            )
        with pytest.raises(ValueError, match="only lower"):
            engine.submit(
                GenerationRequest([3, 7, 11], max_tokens=2, max_prompt_tokens=8)
            )


def test_chat_budget_switches_thinking_off_or_refuses(monkeypatch):
    from mlx_beam.api import chat
    from mlx_beam.api.defaults import RequestDefaults
    from mlx_beam.api.errors import ApiError
    from tests.stub_tokenizer import THINK_START, StubTokenizer

    class Seeding(StubTokenizer):
        """Opens the block at the prompt end unless told not to, Qwen-style."""

        def apply_chat_template(self, messages, enable_thinking=True, **kw):
            ids = super().apply_chat_template(messages, **kw)
            return ids + [THINK_START] if enable_thinking else ids

    class Stubborn(Seeding):
        def apply_chat_template(self, messages, **kw):
            kw.pop("enable_thinking", None)
            return super().apply_chat_template(messages, **kw)

    msgs = [{"role": "user", "content": "w1"}]
    d = RequestDefaults.resolve(flags={"min_response_tokens": 512})
    # Free 100, asked 80, reserve 512: served, with the block left out.
    req = chat.parse_chat_request({"messages": msgs, "max_tokens": 80}, "m", d)
    gen = chat.to_generation_request(Seeding(), req, d, max_context=100 + 4)
    assert req.template_kwargs["enable_thinking"] is False
    assert gen.tokens[-1] != THINK_START and gen.reasoning.max_tokens == 0
    assert not gen.reasoning.seeded and gen.min_response_tokens == 512
    # A template that cannot leave it out: the close must fit the answer room.
    req = chat.parse_chat_request({"messages": msgs, "max_tokens": 80}, "m", d)
    gen = chat.to_generation_request(Stubborn(), req, d, max_context=104)
    assert gen.reasoning.seeded and gen.reasoning.max_tokens == 0
    # The stub encodes no newline, so its close is the marker alone: 1 token.
    req = chat.parse_chat_request({"messages": msgs, "max_tokens": 1}, "m", d)
    with pytest.raises(ApiError) as exc:
        chat.to_generation_request(Stubborn(), req, d, max_context=104)
    assert exc.value.code == "context_length_exceeded"
    # Room to think: the request's own budget wins when it is the smaller.
    req = chat.parse_chat_request(
        {"messages": msgs, "max_tokens": 1000, "max_reasoning_tokens": 30}, "m", d
    )
    gen = chat.to_generation_request(Seeding(), req, d, max_context=2000)
    assert gen.reasoning.seeded and gen.reasoning.max_tokens == 30
    assert gen.reasoning.close == gen.reasoning.end  # no newline in the stub
    # The server's budget when the request names none, cut to the reserve.
    d = RequestDefaults.resolve(
        flags={"min_response_tokens": 100, "max_reasoning_tokens": 5000}
    )
    req = chat.parse_chat_request({"messages": msgs, "max_tokens": 300}, "m", d)
    gen = chat.to_generation_request(Seeding(), req, d, max_context=2000)
    # 300 minus the reserve of 100 minus the close's tail (the stub's close
    # is the marker alone, so 1).
    assert gen.reasoning.max_tokens == 199


@pytest.mark.parametrize("limit", [1, 2, 3, 4, 6])
def test_small_budgets_never_overshoot(limit):
    limits = ReasoningLimits(**{**LIMITS.__dict__, "max_tokens": limit})
    tracker = ThinkingBudget(limits)
    out = Steps(tracker, [START] + list(range(10, 30))).run(12)
    assert tracker.reasoning_tokens <= limit
    if limit < 3:
        # Too small for the opener, a free token and the close: no block.
        assert START not in out and tracker.reasoning_tokens == 0
    else:
        assert tracker.reasoning_tokens == limit and END in out
    seeded = ReasoningLimits(**{**LIMITS.__dict__, "seeded": True, "max_tokens": limit})
    tracker = ThinkingBudget(seeded)
    out = Steps(tracker, list(range(10, 30))).run(12)
    assert tracker.reasoning_tokens <= limit and END in out


def test_two_token_end_marker_is_exact_and_a_started_close_is_finished():
    E1, E2 = 54, 55
    limits = ReasoningLimits((START,), (E1, E2), (NL, E1, E2, NL2), max_tokens=5)
    tracker = ThinkingBudget(limits)
    # Opener, two free tokens, then the forced newline and the marker's
    # first token still count: five, not six.
    steps = Steps(tracker, [START, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    out = steps.run(10)
    assert out[:6] == [START, 10, 11, NL, E1, E2]
    assert tracker.reasoning_tokens == 5 and tracker.thinking_truncated
    # The forced tokens are marked as such, the close's tail included.
    assert steps.forced[:7] == [False, False, False, True, True, True, True]
    assert not any(steps.forced[7:])
    # The model starts the marker on the free token and would wander off:
    # the marker is completed instead, so no second run-up eats the answer.
    tracker = ThinkingBudget(limits)
    steps = Steps(tracker, [START, 10, E1, 12, 13, 14, 15, 16])
    out = steps.run(8)
    assert out[:4] == [START, 10, E1, E2] and out[4:] == [13, 14, 15, 16]
    assert tracker.reasoning_tokens == 3 and not tracker.thinking_truncated
    assert not tracker.in_reasoning and not any(steps.forced)


def test_no_reopening_once_the_allowance_is_spent():
    limits = ReasoningLimits(**{**LIMITS.__dict__, "max_tokens": 5})
    tracker = ThinkingBudget(limits)
    # Closed by itself at three, reopened: the rest could not hold another
    # block, so the opener is masked and the count stays at three.
    out = Steps(tracker, [START, 10, 11, END, START, 20, 21, 22, 23, 24]).run(10)
    assert out[:4] == [START, 10, 11, END] and START not in out[4:]
    assert tracker.reasoning_tokens == 3 and not tracker.thinking_truncated
    # With room left, a second block is allowed and closed at the cap.
    limits = ReasoningLimits(**{**LIMITS.__dict__, "max_tokens": 8})
    tracker = ThinkingBudget(limits)
    out = Steps(tracker, [START, 10, END, START, 20, 21, 22, 23, 24, 25]).run(10)
    assert out.count(END) == 2 and tracker.reasoning_tokens == 8
    assert tracker.thinking_truncated


def test_multi_token_opener_is_cut_at_its_last_token():
    S1, S2 = 56, 57
    limits = ReasoningLimits((S1, S2), (END,), (NL, END, NL2), max_tokens=0)
    tracker = ThinkingBudget(limits)
    # Budget 0: the block cannot form; the fragment before stays text.
    out = Steps(tracker, [S1, S2, 10, 11, 12, 13]).run(6)
    assert out[0] == S1 and out[1] != S2 and tracker.reasoning_tokens == 0
    # After a forced close the same gate holds for the reopening.
    limits = ReasoningLimits((S1, S2), (END,), (NL, END, NL2), max_tokens=4)
    tracker = ThinkingBudget(limits)
    out = Steps(tracker, [S1, S2, 10, 11, 12, 13, 14, S1, S2, 20, 21, 22]).run(12)
    assert out[:6] == [S1, S2, 10, 11, NL, END] and tracker.thinking_truncated
    assert tracker.reasoning_tokens == 4
    later = out[6:]
    assert all(later[i + 1] != S2 for i in range(len(later) - 1) if later[i] == S1)


def test_flags_are_right_without_a_budget():
    model = tiny_hybrid()
    with Engine(model) as engine:
        free = collect(engine.submit(GenerationRequest([3, 7, 11], max_tokens=6)))
        start = next(t for t in range(63, 0, -1) if t not in free)
        # Seeded, no budget: the length limit hits inside the block.
        limits = ReasoningLimits((start,), (start - 1,), (start - 1,), seeded=True)
        last = list(
            engine.submit(GenerationRequest([3, 7, 11], max_tokens=6, reasoning=limits))
        )[-1]
        assert last.thinking_truncated and not last.response_truncated


def test_forced_close_logprobs_serialize_strictly():
    import json

    from mlx_beam.api import chat, completions
    from tests.stub_tokenizer import THINK_END, THINK_START, StubTokenizer

    model = tiny_hybrid()
    tok = StubTokenizer()
    with Engine(model) as engine:
        req = chat.parse_chat_request(
            {
                "messages": [{"role": "user", "content": "w1"}],
                "max_tokens": 8,
                "logprobs": True,
                "top_logprobs": 3,
                "max_reasoning_tokens": 2,
            },
            "m",
        )
        gen = chat.to_generation_request(tok, req)
        gen.tokens.append(THINK_START)
        gen.reasoning = ReasoningLimits(
            (THINK_START,), (THINK_END,), (THINK_END,), seeded=True, max_tokens=2
        )
        out = chat.ChatResponder(tok, req, gen.tokens).complete(engine.submit(gen), 0)
        json.dumps(out, allow_nan=False)
        content = out["choices"][0]["logprobs"]["content"]
        # logprobs.content is the message content: the forced marker and the
        # think block stay out of it, the answer tokens are in.
        assert content and all(c["token"] != "</think>" for c in content)
        assert (
            "".join(c["token"] for c in content)
            == out["choices"][0]["message"]["content"]
        )
        req.stream = True
        gen = chat.to_generation_request(tok, req)
        gen.tokens.append(THINK_START)
        gen.reasoning = ReasoningLimits(
            (THINK_START,), (THINK_END,), (THINK_END,), seeded=True, max_tokens=2
        )
        for chunk in chat.ChatResponder(tok, req, gen.tokens).stream(
            engine.submit(gen), 0
        ):
            json.dumps(chunk, allow_nan=False)
        creq = completions.parse_completion_request(
            {"prompt": "w1", "logprobs": 2, "max_tokens": 6}, "m"
        )
        cgen = completions.to_generation_request(tok, creq)
        cgen.tokens.append(THINK_START)
        cgen.reasoning = ReasoningLimits(
            (THINK_START,), (THINK_END,), (THINK_END,), seeded=True, max_tokens=0
        )
        out = completions.CompletionResponder(tok, creq, cgen.tokens).complete(
            engine.submit(cgen), 0
        )
        json.dumps(out, allow_nan=False)
        # The raw endpoint lists every token; the forced marker has no
        # alternatives at all.
        lp = out["choices"][0]["logprobs"]
        i = lp["tokens"].index("</think>")
        assert lp["top_logprobs"][i] == {"</think>": 0.0}


def test_a_marker_the_model_is_inside_of_when_the_force_lands_is_completed():
    """The marker's first token was observed at arming, the next comes on the
    free token: the model is closing by itself, the queue must not answer
    with a newline in place of the first answer token."""
    E1, E2, E3 = 54, 55, 56
    # Two-token marker: E1 on the arming token, E2 on the free token.
    limits = ReasoningLimits((START,), (E1, E2), (NL, E1, E2, NL2), max_tokens=6)
    tracker = ThinkingBudget(limits)
    steps = Steps(tracker, [START, 10, E1, E2, 20, 21, 22, 23])
    out = steps.run(8)
    assert out == [START, 10, E1, E2, 20, 21, 22, 23]
    assert not tracker.thinking_truncated and not any(steps.forced)
    assert tracker.reasoning_tokens == 3 and not tracker.in_reasoning
    # Three-token marker: E1 and E2 observed, E3 on the free token.
    limits = ReasoningLimits(
        (START,), (E1, E2, E3), (NL, E1, E2, E3, NL2), max_tokens=7
    )
    tracker = ThinkingBudget(limits)
    steps = Steps(tracker, [START, 10, E1, E2, E3, 20, 21, 22])
    out = steps.run(8)
    assert out == [START, 10, E1, E2, E3, 20, 21, 22]
    assert not tracker.thinking_truncated and not any(steps.forced)
    # A marker the model began before arming and continued on the free
    # token but would then abandon is completed, as a started close is.
    tracker = ThinkingBudget(limits)
    steps = Steps(tracker, [START, 10, E1, E2, 30, 31, 32, 33])
    out = steps.run(8)
    assert out == [START, 10, E1, E2, E3, 31, 32, 33]
    assert not tracker.thinking_truncated and not tracker.in_reasoning


def test_matcher_survives_a_repeated_prefix():
    from mlx_beam.engine.thinking import _Matcher

    m = _Matcher((7, 7, 9))
    assert [m.feed(t) for t in (7, 7, 7, 9)] == [False, False, False, True]
    m = _Matcher((7, 8, 7, 9))
    assert [m.feed(t) for t in (7, 8, 7, 8, 7, 9)] == [False] * 5 + [True]
    m = _Matcher((1, 2))
    assert [m.feed(t) for t in (1, 1, 2, 1, 2)] == [False, False, True, False, True]


def test_seeded_block_below_its_close_keeps_the_answer_reserve_or_refuses():
    """With the block open and no room to think, the close still costs its
    whole length; a reserve that then cannot be kept is a refusal, not a
    shorter answer."""
    model = tiny_hybrid()
    with Engine(model) as engine:
        free = collect(engine.submit(GenerationRequest([3, 7, 11], max_tokens=8)))
        unused = [t for t in range(63, 0, -1) if t not in free]
        start, end, nl, nl2 = unused[:4]
        limits = ReasoningLimits((start,), (end,), (nl, end, nl2), seeded=True)
        ok = GenerationRequest(
            [3, 7, 11], max_tokens=8, reasoning=limits, min_response_tokens=5
        )
        tokens = collect(engine.submit(ok))
        assert tokens[:3] == [nl, end, nl2] and len(tokens) == 8
        with pytest.raises(ContextTooLong, match="closing it takes 3"):
            engine.submit(
                GenerationRequest(
                    [3, 7, 11], max_tokens=8, reasoning=limits, min_response_tokens=6
                )
            )
