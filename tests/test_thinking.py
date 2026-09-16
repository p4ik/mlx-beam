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
        if self.out:
            self.tracker.observe(self.out[-1], None)
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
            [3, 7, 11], max_tokens=8, reasoning=limits, min_response_tokens=4
        )
        events = list(engine.submit(req))
        tokens = [e.token for e in events]
        # 8 to generate, 4 kept for the answer: the block gets 4, the last of
        # them the forced newline, then the marker and the blank line.
        assert tokens[3:6] == [nl, end, nl2]
        assert tokens[:3] == free[:3]
        assert events[-1].thinking_truncated and events[-1].finish_reason == "length"
        assert events[-1].response_truncated
        # A client stop on the marker ends the request there, flagged.
        req = GenerationRequest(
            [3, 7, 11],
            max_tokens=8,
            reasoning=limits,
            min_response_tokens=4,
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
    assert gen.reasoning.max_tokens == 200
