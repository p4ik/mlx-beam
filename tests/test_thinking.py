"""The reasoning budget: admission arithmetic, the forced close, the flags."""

from dataclasses import replace

import mlx.core as mx
import pytest

from mlx_beam.api.text import TextAssembler, route_label
from mlx_beam.engine import ContextTooLong, Engine, GenerationRequest, ReasoningLimits
from mlx_beam.engine.thinking import ThinkingBudget, budget
from tests.stub_tokenizer import (
    ASST,
    CHANNEL_CLOSE,
    CHANNEL_OPEN,
    CHECK,
    ELF,
    EOM,
    EOS,
    EQ,
    FS,
    LABEL,
    MSG,
    MUSE_CLOSE,
    NEWLINE,
    SELF,
    SELF2,
    SELFSP,
    SP,
    TO,
    USER,
    ChannelStubTokenizer,
    MuseStubTokenizer,
    S,
)
from tests.test_engine import collect, tiny_hybrid
from tests.tracked import limits_for, tracked_events

START, END, NL, NL2 = 50, 51, 52, 53
LIMITS = ReasoningLimits(start=(START,), end=(END,), close=(NL, END, NL2))
# Gemma 4's channel: no label the budget reads, the close the model was
# trained to write, and the opener never masked - the block is closed at
# once when no room is left.
CHANNEL = ReasoningLimits(
    start=(CHANNEL_OPEN,),
    end=(CHANNEL_CLOSE,),
    close=(NEWLINE, CHANNEL_CLOSE, NEWLINE, NEWLINE),
    mask_opener=False,
)
# A channel the budget closed at once, as the stream carries it: opener,
# the free token (the label), the forced close.
GHOST = [CHANNEL_OPEN, LABEL, NEWLINE, CHANNEL_CLOSE, NEWLINE, NEWLINE]
# Muse's recipient: a textual opener read at the message's frame only, its
# head judged at `<|message|>` as the assembler judges it, counted as usage
# counts it (the opener ` to` not). The verdicts read the stub's words.
MUSE_TOK = MuseStubTokenizer()
FRAME = (FS, ASST)
MUSE_PROMPT = [6, 1, 7, FS, ASST]


def _head_text(ids):
    return MUSE_TOK.decode(ids).lstrip()


def muse_verdict(ids):
    text = _head_text(ids)
    return text.startswith("=") and route_label(MUSE_TOK, text[1:]) == "reasoning"


def muse_head_may_open(ids):
    text = _head_text(ids)
    return text.startswith("=") or "=".startswith(text)


MUSE = ReasoningLimits(
    start=(TO,),
    end=(EOM,),
    close=tuple(MUSE_CLOSE),
    label_end=(MSG,),
    label_means_reasoning=muse_verdict,
    mask_opener=False,
    frame=FRAME,
    at_frame=True,
    opener_counted=False,
    head_may_open=muse_head_may_open,
)


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

    def __init__(self, tracker, model_tokens, vocab=64, prompt=(0,)):
        self.tracker = tracker
        self.model = list(model_tokens)
        self.vocab = vocab
        # The context the processor reads: the prompt's end, as the
        # generator carries it, then every token sampled.
        self.prompt = list(prompt)
        self.out = []
        # What the tracker said about each token after observing it.
        self.forced = []
        self.suppressed = []

    def run(self, n):
        context = list(self.prompt)
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
                self._observe(self.out[-2])
        if self.out:
            self._observe(self.out[-1])
        return self.out

    def _observe(self, token):
        self.tracker.observe(token, None)
        self.forced.append(self.tracker.last_forced)
        self.suppressed.append(self.tracker.last_suppressed)


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


def test_budget_holds_while_another_prompt_prefills():
    """While a prefill shares the worker, the generator decodes several
    steps per call; the budget must see each token before the next step,
    or the forced close arms late and the block overshoots its limit."""
    model = tiny_hybrid()
    # A large decode share: many decode steps per prefill slice of the
    # other request, which is sliced small to keep it in the prefill.
    with Engine(model, prefill_slice=2, decode_share=200.0) as engine:
        free = collect(engine.submit(GenerationRequest([3, 7, 11], max_tokens=40)))
        unused = [t for t in range(63, 0, -1) if t not in free]
        start, end, nl, nl2 = unused[:4]
        limits = ReasoningLimits(
            (start,), (end,), (nl, end, nl2), seeded=True, max_tokens=6
        )
        budgeted = engine.submit(
            GenerationRequest([3, 7, 11], max_tokens=40, reasoning=limits)
        )
        other = engine.submit(GenerationRequest(list(range(1, 60)), max_tokens=2))
        events = list(budgeted)
        collect(other)
        tokens = [e.token for e in events]
        # Seeded: the block is open from the prompt; five free tokens, the
        # forced newline is the sixth, the marker follows - whatever the
        # prefill beside it did. Without the per-step observation the whole
        # run decodes in one call and the close never comes.
        assert end in tokens
        assert tokens.index(end) == 6, tokens[:8]
        assert (
            tokens[tokens.index(end) - 1] == nl and tokens[tokens.index(end) + 1] == nl2
        )


@pytest.mark.parametrize("limit", range(9))
def test_a_labelled_block_never_exceeds_a_small_budget(limit):
    """A labelled opener costs three counted tokens before the first free
    one (opener, label, label end); the gate that keeps a block from
    opening must count them, or budgets of two and three overshoot to four."""
    A, MSG = 40, 42
    limits = ReasoningLimits(
        start=(START,),
        end=(END,),
        close=(END, START, 41, MSG),
        max_tokens=limit,
        labels=((A,),),
        label_end=(MSG,),
    )
    tracker = ThinkingBudget(limits)
    out = Steps(tracker, [START, A, MSG] + list(range(10, 25))).run(18)
    assert tracker.reasoning_tokens <= limit
    if limit < 4:
        # Header (3) plus one free token do not fit: no block at all.
        assert A not in out and tracker.reasoning_tokens == 0
    else:
        assert tracker.reasoning_tokens == limit and END in out


@pytest.mark.parametrize("limit", [0, 4])
def test_a_label_variant_the_api_routes_as_reasoning_is_counted_and_cut(limit):
    """The API routes a label by its text (` analysis` and `analysis ` are
    the reasoning channel like `analysis`); the budget must count what the
    API shows as reasoning, or a variant runs unbounded. The API's verdict
    is asked for the label the model wrote; the mask that keeps a block
    from forming cuts every label form it was given - a form it does not
    know (the label and a space) begins with one it does. A header longer
    than the canonical one leaves no room for the free token the arming
    counts on: the close is forced in the same step the label ends, decided
    in the graph, so the budget holds exactly - unless the header alone
    exceeds it. A label that goes on into a tool's name forces nothing."""
    A, A_SPACED, SP, T, F, MSG = 40, 39, 38, 37, 41, 42
    reasoning = ((A,), (A_SPACED,), (A, SP), (A, SP, SP))
    limits = ReasoningLimits(
        start=(START,),
        end=(END,),
        close=(END, START, F, MSG),
        max_tokens=limit,
        labels=((A,), (A_SPACED,)),
        label_end=(MSG,),
        label_means_reasoning=lambda ids: ids in reasoning,
    )
    body = list(range(10, 26))
    for label in ((A,), (A_SPACED,), (A, SP), (A, SP, SP)):
        tracker = ThinkingBudget(limits)
        steps = Steps(tracker, [START, *label, MSG, *body])
        out = steps.run(20)
        header = 2 + len(label)
        if limit == 0:
            assert label[0] not in out and tracker.reasoning_tokens == 0, label
        else:
            assert END in out and tracker.thinking_truncated, label
            assert tracker.reasoning_tokens == max(limit, header), label
            # The close sits right where the budget is up: after the free
            # token, or right after the label's end when the header used
            # that token up.
            assert out.index(END) == header + (limit > header), label
            assert steps.forced[out.index(END)]
    # A label the API routes elsewhere opens nothing - the answer's channel,
    # and a tool's name that began like the reasoning label.
    for label in ((F,), (A, SP, T)):
        tracker = ThinkingBudget(limits)
        steps = Steps(tracker, [START, *label, MSG, *body])
        out = steps.run(20)
        assert END not in out and tracker.reasoning_tokens == 0, label
        assert not any(steps.forced)


def test_a_longer_label_in_a_later_block_is_judged_against_the_rest():
    """The forced close at a label's end is decided against what the budget
    has left, not against the whole of it: a second block whose header
    takes the rest exactly gets its close at once."""
    A, SP, F, MSG = 40, 38, 41, 42
    reasoning = ((A,), (A, SP))
    limits = ReasoningLimits(
        start=(START,),
        end=(END,),
        close=(END, START, F, MSG),
        max_tokens=8,
        labels=((A,),),
        label_end=(MSG,),
        label_means_reasoning=lambda ids: ids in reasoning,
    )
    body = list(range(10, 30))
    # First block: header 3 and one token, closed by the model - 4 spent.
    first = [START, A, MSG, 10, END]
    for label, expected in (((A,), 8), ((A, SP), 8)):
        tracker = ThinkingBudget(limits)
        steps = Steps(tracker, first + [START, *label, MSG, *body])
        out = steps.run(24)
        second = out.index(START, 1)
        assert tracker.reasoning_tokens == expected, label
        # Header 3 leaves one free token; header 4 none.
        assert out.index(END, second) == second + 2 + len(label) + (len(label) == 1)
        assert tracker.thinking_truncated


def test_a_labelled_opener_counts_only_when_its_label_means_reasoning():
    """Harmony and Muse open the answer and a tool call with the same
    marker as the reasoning block, followed by a label. The budget waits
    for the label: `analysis` opens a block that counts, `final` does not
    - so a natural close followed by the answer's channel is not a second
    block, and no forced close lands in the answer."""
    A, F, MSG = 40, 41, 42  # "analysis", "final", "<|message|>"
    limits = ReasoningLimits(
        start=(START,),
        end=(END,),
        close=(END, START, F, MSG),
        max_tokens=6,
        labels=((A,),),
        label_end=(MSG,),
    )
    tracker = ThinkingBudget(limits)
    # Analysis: opener, label, its end, three free tokens - counted.
    for t in (START, A, MSG, 10, 11):
        tracker.observe(t, None)
    assert tracker.in_reasoning and tracker.reasoning_tokens == 5
    tracker.observe(END, None)  # the model closes by itself
    assert not tracker.in_reasoning
    # The answer's channel: the same opener, another label - no block.
    for t in (START, F, MSG, 12, 13, 14, 15, 16):
        tracker.observe(t, None)
    assert not tracker.in_reasoning and tracker.reasoning_tokens == 5
    assert not tracker.thinking_truncated
    tracker.observe(17, "length")
    assert not tracker.thinking_truncated
    # And a budget that is up forces the close inside the block, never in
    # the answer that follows it.
    tracker = ThinkingBudget(limits)
    out = Steps(tracker, [START, A, MSG, 10, 11, 12, 13, 14, 15, 16]).run(10)
    assert END in out and tracker.thinking_truncated
    assert out.index(END) <= 6
    # Budget zero: the shared opener stays open for the answer's channel
    # and a tool's; only the reasoning label after it is cut.
    tracker = ThinkingBudget(replace(limits, max_tokens=0))
    out = Steps(tracker, [START, A, MSG, 10, START, F, MSG, 11]).run(8)
    assert out[0] == START and out[1] != A and START in out[1:]
    assert tracker.reasoning_tokens == 0
    tracker = ThinkingBudget(replace(limits, max_tokens=0))
    out = Steps(tracker, [START, F, MSG, 10, 11]).run(5)
    assert out == [START, F, MSG, 10, 11] and not tracker.in_reasoning


def test_a_channel_family_at_budget_zero_lets_the_opener_through_and_closes_at_once():
    tracker = ThinkingBudget(replace(CHANNEL, max_tokens=0))
    steps = Steps(tracker, [CHANNEL_OPEN, LABEL, NEWLINE, 10, 11, 12, 13])
    out = steps.run(7)
    # The opener is not masked; the label is the free token, then the
    # forced close; the answer follows.
    assert out == GHOST + [13]
    assert tracker.reasoning_tokens == 0 and not tracker.thinking_truncated
    assert steps.suppressed == [True] * 6 + [False]


def test_two_ghost_channels_are_both_closed_at_once():
    tracker = ThinkingBudget(replace(CHANNEL, max_tokens=0))
    wishes = [CHANNEL_OPEN, LABEL, NEWLINE, 10, 11, 12]
    wishes += [CHANNEL_OPEN, LABEL, NEWLINE, 20, 21, 22, 11]
    out = Steps(tracker, wishes).run(13)
    assert out == GHOST + GHOST + [11]
    assert tracker.reasoning_tokens == 0 and not tracker._block


def test_an_exhausted_positive_budget_closes_the_next_channel_at_once():
    tracker = ThinkingBudget(replace(CHANNEL, max_tokens=3))
    wishes = [CHANNEL_OPEN, LABEL, NEWLINE, 20, 21, 22]
    wishes += [CHANNEL_OPEN, LABEL, NEWLINE, 30, 31, 32, 30]
    steps = Steps(tracker, wishes)
    out = steps.run(13)
    # The first block spends the budget and is forced shut; the second
    # opens (not masked), is closed at once and counts nothing.
    assert out == GHOST + GHOST + [30]
    assert tracker.reasoning_tokens == 3 and tracker.thinking_truncated
    assert steps.suppressed == [False] * 6 + [True] * 6 + [False]


def test_think_and_xtml_openers_stay_masked_at_budget_zero():
    tracker = ThinkingBudget(ReasoningLimits(**{**LIMITS.__dict__, "max_tokens": 0}))
    out = Steps(tracker, [START, 10, 11]).run(3)
    assert START not in out and tracker.reasoning_tokens == 0
    # XTML's opener is three tokens, cut at the last.
    S1, S2, S3 = 44, 45, 46
    tracker = ThinkingBudget(
        ReasoningLimits((S1, S2, S3), (END,), (NL, END, NL2), max_tokens=0)
    )
    out = Steps(tracker, [S1, S2, S3, 10, 11]).run(5)
    assert out[:2] == [S1, S2] and out[2] != S3 and tracker.reasoning_tokens == 0


def test_a_labelled_opener_stays_masked_at_budget_zero():
    A, F, MSG = 40, 41, 42
    limits = ReasoningLimits(
        start=(START,),
        end=(END,),
        close=(END, START, F, MSG),
        max_tokens=0,
        labels=((A,),),
        label_end=(MSG,),
    )
    tracker = ThinkingBudget(limits)
    out = Steps(tracker, [START, A, MSG, 10, 11]).run(5)
    assert out[0] == START and out[1] != A and tracker.reasoning_tokens == 0


def test_a_seeded_block_at_budget_zero_is_still_forced_shut():
    tracker = ThinkingBudget(replace(CHANNEL, seeded=True, max_tokens=0))
    out = Steps(tracker, [10, 11, 12, 13, 14, 15]).run(6)
    assert out == [NEWLINE, CHANNEL_CLOSE, NEWLINE, NEWLINE, 14, 15]
    assert tracker.thinking_truncated


def test_a_suppressed_block_counts_nothing_in_the_assembler():
    from mlx_beam.api.text import TextAssembler

    limits = replace(CHANNEL, max_tokens=0)
    tracker = ThinkingBudget(limits)
    evs = tracked_events(limits, GHOST + [10], tracker=tracker)
    asm = TextAssembler(ChannelStubTokenizer(), prompt_tokens=[6, 1, 7])
    deltas = [asm.feed(e) for e in evs]
    assert tracker.reasoning_tokens == 0
    assert asm.reasoning_tokens == 0
    assert [e.reasoning_counted for e in evs] == [0] * len(evs)
    assert not asm.thinking_truncated
    assert "".join(d.content for d in deltas) == "w10 "
    assert not any(d.reasoning for d in deltas)


def test_the_forced_close_of_a_suppressed_block_is_still_forced():
    tracker = ThinkingBudget(replace(CHANNEL, max_tokens=0))
    steps = Steps(tracker, [CHANNEL_OPEN, LABEL, 10, 11, 12, 13, 20])
    out = steps.run(7)
    assert out == GHOST + [20]
    # The label is the model's free token; the counted line break, the
    # marker and the tail come from the queue, down to its last token.
    assert steps.forced == [False, False, True, True, True, True, False]
    assert not tracker._natural
    # The block's progress runs on while the published count stays at 0.
    assert tracker._seen == 3 and tracker.reasoning_tokens == 0


def test_a_suppressed_block_cut_by_length_is_a_cut_answer():
    limits = replace(CHANNEL, max_tokens=0)
    # Cut at the opener, inside the label, at the end marker, in the tail.
    for ids in (GHOST[:1], GHOST[:2], GHOST[:4], GHOST[:5]):
        tracker = ThinkingBudget(limits)
        for t in ids[:-1]:
            tracker.observe(t, None)
        tracker.observe(ids[-1], "length")
        assert not tracker.thinking_truncated, ids
        # What the engine reads: not inside the reasoning, so the cut is
        # the answer's (response_truncated = cut and not in_reasoning).
        assert not tracker.in_reasoning and tracker.reasoning_tokens == 0, ids
    # A flag a real first block set stays.
    tracker = ThinkingBudget(replace(CHANNEL, max_tokens=3))
    for t in GHOST:
        tracker.observe(t, None)
    assert tracker.thinking_truncated and tracker.reasoning_tokens == 3
    tracker.observe(CHANNEL_OPEN, "length")
    assert tracker.thinking_truncated and tracker.reasoning_tokens == 3


def test_a_newline_as_the_free_token_shows_nothing():
    from mlx_beam.api import messages, responses
    from mlx_beam.api.text import TextAssembler

    tok = ChannelStubTokenizer()
    limits = replace(CHANNEL, max_tokens=0)
    # The free token ends the label: the forced line break would be
    # reasoning text, the tail's two would be content.
    ids = [CHANNEL_OPEN, NEWLINE, NEWLINE, CHANNEL_CLOSE, NEWLINE, NEWLINE, 10]
    asm = TextAssembler(tok, prompt_tokens=[6, 1, 7])
    deltas = [asm.feed(e) for e in tracked_events(limits, ids)]
    assert "".join(d.content for d in deltas) == "w10 "
    assert not any(d.reasoning for d in deltas)
    assert asm.reasoning_tokens == 0
    req = responses.parse_responses_request({"input": "w1"}, "m")
    gen = responses.to_generation_request(tok, req)
    final = responses.ResponsesResponder(tok, req, gen.tokens).complete(
        tracked_events(limits, ids), 0
    )
    assert [i["type"] for i in final["output"]] == ["message"]
    req = messages.parse_messages_request(
        {"max_tokens": 16, "messages": [{"role": "user", "content": "w1"}]}, "m"
    )
    gen = messages.to_generation_request(tok, req)
    final = messages.MessagesResponder(tok, req, gen.tokens).complete(
        tracked_events(limits, ids), 0
    )
    assert [b["type"] for b in final["content"]] == ["text"]


def test_the_suppressed_flag_ends_with_the_last_tail_token():
    from types import SimpleNamespace

    from mlx_beam.engine.request import SamplingParams
    from tests.test_vendor_optiq_kv import tiny_llama

    limits = replace(CHANNEL, max_tokens=0)
    # _on_step reads only the budgets and the observed tuples.
    uid = 7
    stand_in = SimpleNamespace(_budgets={uid: ThinkingBudget(limits)}, _observed={})
    for t in GHOST + [10]:
        Engine._on_step(
            stand_in, [SimpleNamespace(uid=uid, token=t, finish_reason=None)]
        )
    tuples = list(stand_in._observed[uid])
    flags = [t[3] for t in tuples]
    # The close's last token still carries the flag, the first answer
    # token no longer: observe reads it before the block state is cleared.
    assert flags[5]
    assert flags == [True] * 6 + [False]
    assert [t[0] for t in tuples] == [False, False, True, True, True, True, False]
    assert [t[4] for t in tuples] == [0] * 7
    # The event build in the engine: the bias draws the opener and the
    # free token, the forced close overrides it, the stop sequence ends
    # the completion on the close's last token.
    with Engine(tiny_llama()) as solo:
        events = list(
            solo.submit(
                GenerationRequest(
                    [3, 7, 11],
                    max_tokens=8,
                    sampling=SamplingParams(logit_bias={CHANNEL_OPEN: 100.0}),
                    reasoning=limits,
                    stop_sequences=[(NEWLINE, CHANNEL_CLOSE, NEWLINE, NEWLINE)],
                )
            )
        )
    tokens = [e.token for e in events]
    assert tokens == [
        CHANNEL_OPEN,
        CHANNEL_OPEN,
        NEWLINE,
        CHANNEL_CLOSE,
        NEWLINE,
        NEWLINE,
    ]
    assert [e.suppressed for e in events] == [True] * 6
    assert [e.forced for e in events] == [False, False, True, True, True, True]
    assert [e.reasoning_counted for e in events] == [0] * 6
    assert events[-1].finish_reason == "stop" and not events[-1].response_truncated
    assert not events[-1].thinking_truncated
    # The helper mirrors the engine, the logprob apart.
    mirrored = tracked_events(limits, tokens, finish="stop", eos=None)
    assert mirrored == [replace(e, logprob=-0.1) for e in events]


def test_a_suppressed_channel_stays_visible_without_routing():
    from mlx_beam.api import chat

    tok = ChannelStubTokenizer()
    limits = replace(CHANNEL, max_tokens=0)
    req = chat.parse_chat_request(
        {"messages": [{"role": "user", "content": "w1"}]}, "m"
    )
    gen = chat.to_generation_request(tok, req)
    out = chat.ChatResponder(tok, req, gen.tokens, reasoning_field="none").complete(
        tracked_events(limits, GHOST + [10]), cached=0
    )
    # Counted 0 in every mode; with routing off the block is the client's
    # to parse, as the model wrote it.
    assert out["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0
    content = out["choices"][0]["message"]["content"]
    assert content == "<|channel>thought\n<channel|>\n\nw10 "
    out = chat.ChatResponder(tok, req, gen.tokens).complete(
        tracked_events(limits, GHOST + [10]), cached=0
    )
    msg = out["choices"][0]["message"]
    assert msg["content"] == "w10 " and "reasoning" not in msg


def muse_steps(limits, wishes, n=None):
    """Steps at the frame: the prompt ends with `<|start|>assistant`."""
    tracker = ThinkingBudget(limits)
    steps = Steps(tracker, wishes, prompt=FRAME)
    steps.run(len(wishes) if n is None else n)
    return tracker, steps


def muse_assembled(limits, ids, **kw):
    """(tracker, assembler, deltas) for `ids` through a fresh tracker."""
    tracker = ThinkingBudget(limits)
    asm = TextAssembler(MuseStubTokenizer(), prompt_tokens=MUSE_PROMPT, **kw)
    evs = tracked_events(limits, ids, tracker=tracker)
    return tracker, asm, [asm.feed(e) for e in evs]


def test_the_muse_stub_derives_the_frame_limits():
    tok = MuseStubTokenizer()
    limits = limits_for(tok, MUSE_PROMPT, 0)
    # The opener is the token before the head, the head runs to the
    # message marker, every recipient form is judged by its text.
    assert limits.start == (TO,) and limits.label_end == (MSG,)
    assert limits.labels == () and not limits.opener_counted
    assert limits.frame == FRAME and limits.at_frame
    assert limits.mask_opener is False
    assert limits.label_means_reasoning((SELF,))
    assert limits.label_means_reasoning((EQ, SP, SELF2))
    assert not limits.label_means_reasoning((USER,))
    assert not limits.label_means_reasoning((SELF, CHECK))
    assert limits.head_may_open((EQ,)) and not limits.head_may_open((20,))
    # A prompt that does not end with the frame: the window is shut.
    assert not limits_for(tok, MUSE_PROMPT + [MSG, 10], 0).at_frame
    # The channel keeps its one-token opener and no frame.
    channel = limits_for(ChannelStubTokenizer(), [6, 1, 7], 0)
    assert channel.frame == () and channel.opener_counted
    assert channel.head_may_open is None


def test_the_window_shuts_at_the_first_token_that_is_no_opener():
    limits = replace(MUSE, max_tokens=4)
    # The answer, after `<|message|>` or from its first word, writes
    # ` to=self` in its text: no head, no block, no forced close.
    for wishes in ([MSG, 20, 21, TO, SELF, MSG, 22, 23], [20, TO, SELF, MSG, 21, 22]):
        tracker, steps = muse_steps(limits, wishes)
        assert steps.out == wishes, wishes
        assert tracker.reasoning_tokens == 0 and not tracker.thinking_truncated
        assert tracker._state == "normal" and not any(steps.suppressed)


def test_a_frame_the_model_writes_reopens_the_window():
    limits = replace(MUSE, max_tokens=8)
    wishes = [TO, SELF, MSG, 20, EOM, FS, ASST, TO, SELF, MSG, 30, 31, 32]
    wishes += MUSE_CLOSE + [40]
    tracker, steps = muse_steps(limits, wishes)
    # Head (2) and one token, closed by the model; the model's own frame
    # opens the window again: head (2) and three tokens, closed by force.
    assert steps.out == wishes
    assert tracker.reasoning_tokens == 8 and tracker.thinking_truncated
    assert steps.out.count(EOM) == 2


def test_a_forced_close_leaves_the_window_shut():
    limits = replace(MUSE, max_tokens=3)
    wishes = [TO, SELF, MSG, 10] + MUSE_CLOSE + [TO, SELF, MSG, 30]
    tracker, steps = muse_steps(limits, wishes)
    assert steps.out == wishes
    # The recipient right after the forced close is the answer's text.
    assert tracker.reasoning_tokens == 3 and steps.out.count(EOM) == 1
    assert not any(steps.suppressed)


def test_families_without_a_frame_keep_the_matcher():
    # Harmony: two natural blocks, both counted (header 3 and one token each).
    A, HMSG = 40, 42
    harmony = ReasoningLimits(
        start=(START,),
        end=(END,),
        close=(END, START, 41, HMSG),
        labels=((A,),),
        label_end=(HMSG,),
    )
    tracker = ThinkingBudget(harmony)
    for t in (START, A, HMSG, 10, END, 11, START, A, HMSG, 12, END):
        tracker.observe(t, None)
    assert tracker.reasoning_tokens == 8 and not tracker.in_reasoning
    # The channel: two blocks, opener and one token each.
    tracker = ThinkingBudget(CHANNEL)
    for t in (CHANNEL_OPEN, 10, CHANNEL_CLOSE, 11, CHANNEL_OPEN, 12, CHANNEL_CLOSE):
        tracker.observe(t, None)
    assert tracker.reasoning_tokens == 4 and not tracker.in_reasoning


def test_every_recipient_form_the_automaton_routes_opens_a_block():
    limits = replace(MUSE, max_tokens=4)
    for head in ([SELF], [EQ, SELF2], [EQ, SELFSP], [EQ, SP, SELF2], [EQ, S, ELF]):
        wishes = [TO, *head, MSG, 10, 11, 12, 13, 14, 15, 16, 17]
        tracker, steps = muse_steps(limits, wishes, 12)
        assert tracker.reasoning_tokens == 4 and tracker.thinking_truncated, head
        assert steps.out[: len(head) + 2] == [TO, *head, MSG], head
        # The head costs its tokens and the marker; a free token follows
        # only while the budget holds one.
        free = 4 - (len(head) + 1)
        assert steps.out.index(EOM) == len(head) + 2 + free, head
        assert not any(steps.suppressed), head


def test_a_recipient_that_is_no_reasoning_opens_no_block():
    limits = replace(MUSE, max_tokens=4)
    # The answer's recipient, and a tool whose name begins like `self`;
    # after the head the window is shut, a later ` to=self` is text.
    for wishes in (
        [TO, USER, MSG, 10, TO, SELF, MSG, 11],
        [TO, SELF, CHECK, MSG, 10, TO, SELF, MSG, 11],
    ):
        tracker, steps = muse_steps(limits, wishes)
        assert steps.out == wishes and tracker.reasoning_tokens == 0, wishes
        assert tracker._state == "normal" and not tracker.thinking_truncated
        assert not any(steps.forced) and not any(steps.suppressed)


def test_a_head_that_is_no_opener_aborts():
    limits = replace(MUSE, max_tokens=0)
    # ` to` followed by text, not by `=`: the head is given up at once,
    # and the `<|message|>` later on ends no head.
    wishes = [TO, 20, 21, MSG, 22, 23]
    tracker, steps = muse_steps(limits, wishes)
    assert steps.out == wishes and tracker.reasoning_tokens == 0
    assert tracker._state == "normal" and not any(steps.forced)
    assert not any(steps.suppressed)


def test_no_token_of_the_frame_family_is_masked():
    tok = MuseStubTokenizer()
    logits = mx.zeros((64,))
    tracker = ThinkingBudget(limits_for(tok, MUSE_PROMPT, 0))
    out = tracker(mx.array([FS, ASST, TO]), logits)
    assert all(float(out[t]) == 0.0 for t in (SELF, EQ, USER)) and not tracker._block
    # After a forced close just the same.
    tracker = ThinkingBudget(limits_for(tok, MUSE_PROMPT, 3))
    steps = Steps(tracker, [TO, SELF, MSG, 10] + MUSE_CLOSE + [TO], prompt=FRAME)
    steps.run(9)
    assert steps.out[-1] == TO and tracker.reasoning_tokens == 3
    out = tracker(mx.array([*FRAME, *steps.out]), logits)
    assert all(float(out[t]) == 0.0 for t in (SELF, EQ, USER)) and not tracker._block


def test_a_reasoning_head_at_budget_zero_is_closed_at_once_and_counts_nothing():
    limits = replace(MUSE, max_tokens=0)
    for head in ([SELF], [EQ, SELF2], [EQ, SP, SELF2]):
        wishes = [TO, *head, MSG] + MUSE_CLOSE + [10, 11]
        tracker, steps = muse_steps(limits, wishes)
        cost = len(head) + 1  # the head and its end marker
        assert steps.out == wishes, head
        assert tracker.reasoning_tokens == 0 and tracker._seen == cost, head
        assert not tracker.thinking_truncated
        # Suppressed from the head's end to the close's last token; forced
        # on the four tokens of the close.
        assert steps.suppressed == [False] * cost + [True] * 5 + [False] * 2, head
        assert steps.forced == [False] * (cost + 1) + [True] * 4 + [False] * 2, head
        # The assembler over the sequence as it was emitted: the head's
        # tokens without a flag, every count 0, nothing shown.
        tracker, asm, deltas = muse_assembled(limits, steps.out)
        assert tracker.reasoning_tokens == 0 == asm.reasoning_tokens, head
        assert "".join(d.content for d in deltas) == "w10 w11 ", head
        assert not any(d.reasoning for d in deltas) and not asm.thinking_truncated


def test_a_tool_head_at_budget_zero_is_untouched():
    limits = replace(MUSE, max_tokens=0)
    wishes = [TO, SELF, CHECK, MSG, 20, 21, EOM]
    tracker, steps = muse_steps(limits, wishes)
    assert steps.out == wishes and tracker.reasoning_tokens == 0
    assert not any(steps.forced) and not any(steps.suppressed)
    # The call reaches the parser as written: head, marker, arguments.
    tok = MuseStubTokenizer()
    seen = []
    tok.tool_parser = lambda text, tools: seen.append(text) or {
        "name": "self_check",
        "arguments": {},
    }
    asm = TextAssembler(tok, prompt_tokens=MUSE_PROMPT, tools=[{"type": "function"}])
    evs = tracked_events(limits, steps.out)
    assert not any(e.suppressed or e.forced or e.reasoning_counted for e in evs)
    deltas = [asm.feed(e) for e in evs]
    assert len(seen) == 1 and seen[0].startswith("self_check<|message|>w20 w21 ")
    calls = [c["function"]["name"] for d in deltas for c in d.tool_calls]
    assert calls == ["self_check"] and asm.reasoning_tokens == 0
    assert not any(d.content or d.reasoning for d in deltas)


def test_an_answer_head_at_budget_zero_is_shown():
    limits = replace(MUSE, max_tokens=0)
    wishes = [TO, USER, MSG, 10, 11]
    tracker, steps = muse_steps(limits, wishes)
    assert steps.out == wishes and tracker.reasoning_tokens == 0
    assert not any(steps.forced) and not any(steps.suppressed)
    tracker, asm, deltas = muse_assembled(limits, steps.out)
    assert "".join(d.content for d in deltas) == "w10 w11 "
    assert asm.reasoning_tokens == 0 and not any(d.reasoning for d in deltas)


def test_a_reasoning_head_after_an_exhausted_budget_is_closed_at_once():
    limits = replace(MUSE, max_tokens=3)
    wishes = [TO, SELF, MSG, 10] + MUSE_CLOSE + [FS, ASST, TO, SELF, MSG]
    wishes += MUSE_CLOSE + [30]
    tracker, steps = muse_steps(limits, wishes)
    assert steps.out == wishes
    # The first block spent the budget; the second head is let through,
    # closed at once and counts nothing. The flag of the first stays.
    assert tracker.reasoning_tokens == 3 and tracker.thinking_truncated
    assert steps.suppressed == [False] * 12 + [True] * 5 + [False]
    assert steps.forced == [False] * 4 + [True] * 4 + [False] * 5 + [True] * 4 + [False]


def test_a_head_that_does_not_fit_is_suppressed_whatever_its_length():
    cases = [
        # limit, head, counted, suppressed, free tokens before the close
        (3, [EQ, SP, SELF2], 0, True, 0),
        (3, [EQ, SP, S, ELF], 0, True, 0),
        (3, [EQ, SELF2], 3, False, 0),
        (3, [SELF], 3, False, 1),
        (4, [EQ, SP, SELF2], 4, False, 0),
        (5, [EQ, SP, SELF2], 5, False, 1),
    ]
    for limit, head, counted, suppressed, free in cases:
        limits = replace(MUSE, max_tokens=limit)
        wishes = [TO, *head, MSG] + [10] * free + MUSE_CLOSE + [11, 12]
        tracker, steps = muse_steps(limits, wishes)
        assert steps.out == wishes, (limit, head)
        assert tracker.reasoning_tokens == counted <= limit, (limit, head)
        assert any(steps.suppressed) is suppressed, (limit, head)
        assert tracker.thinking_truncated is (not suppressed), (limit, head)
        tracker, asm, _ = muse_assembled(limits, steps.out)
        assert asm.reasoning_tokens == tracker.reasoning_tokens == counted


def test_a_head_cut_before_its_end_counts_nothing():
    for limit in (0, 3):
        limits = replace(MUSE, max_tokens=limit)
        for ids, finish, eos, stop in (
            ([TO, SELF], "length", EOS, None),
            ([TO, EQ], "length", EOS, None),
            ([TO, SELF, EOM], "stop", EOS, None),
            ([TO, SELF, 10], "stop", None, ["w10"]),
        ):
            tracker = ThinkingBudget(limits)
            evs = tracked_events(limits, ids, finish=finish, eos=eos, tracker=tracker)
            asm = TextAssembler(
                MuseStubTokenizer(), prompt_tokens=MUSE_PROMPT, stop_words=stop
            )
            for e in evs:
                asm.feed(e)
            assert tracker.reasoning_tokens == 0 == asm.reasoning_tokens, (limit, ids)
            assert not tracker.thinking_truncated and not asm.thinking_truncated
            assert asm.response_truncated is (finish == "length"), (limit, ids)
            assert not any(e.suppressed for e in evs), (limit, ids)
    # After a real first block the count and its flag stay.
    limits = replace(MUSE, max_tokens=3)
    ids = [TO, SELF, MSG, 10] + MUSE_CLOSE + [FS, ASST, TO, SELF]
    tracker = ThinkingBudget(limits)
    evs = tracked_events(limits, ids, finish="length", tracker=tracker)
    asm = TextAssembler(MuseStubTokenizer(), prompt_tokens=MUSE_PROMPT)
    for e in evs:
        asm.feed(e)
    assert tracker.reasoning_tokens == 3 == asm.reasoning_tokens
    assert tracker.thinking_truncated and asm.thinking_truncated
    assert asm.response_truncated


def test_a_suppressed_head_stays_visible_without_routing():
    from mlx_beam.api import chat

    tok = MuseStubTokenizer()
    limits = limits_for(tok, MUSE_PROMPT, 0)
    req = chat.parse_chat_request(
        {"messages": [{"role": "user", "content": "w1"}]}, "m"
    )
    gen = chat.to_generation_request(tok, req)
    assert gen.tokens[-2:] == [FS, ASST]
    ids = [TO, SELF, MSG] + MUSE_CLOSE + [10]
    out = chat.ChatResponder(tok, req, gen.tokens, reasoning_field="none").complete(
        tracked_events(limits, ids), cached=0
    )
    # Counted 0 in every mode; with routing off the head, its marker and
    # the end marker stay, the frame of the close is structural and cut.
    assert out["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0
    content = out["choices"][0]["message"]["content"]
    assert content == " to=self<|message|><|eom|>w10 "
    out = chat.ChatResponder(tok, req, gen.tokens).complete(
        tracked_events(limits, ids), cached=0
    )
    msg = out["choices"][0]["message"]
    assert msg["content"] == "w10 " and "reasoning" not in msg


@pytest.mark.parametrize(
    "head, limit, counted, free",
    [
        ([SELF], 0, 0, 0),
        ([SELF], 1, 0, 0),
        ([SELF], 2, 2, 0),
        ([SELF], 3, 3, 1),
        ([SELF], 4, 4, 2),
        ([EQ, SELF2], 3, 3, 0),
        ([EQ, SELF2], 4, 4, 1),
    ],
)
def test_small_budgets_at_the_frame(head, limit, counted, free):
    """Budgets 0 and 1 cannot hold the canonical head: suppressed, 0.
    Budget 2 holds the head and no free token: counted, closed at once.
    From 3 on the free tokens follow. The split head costs one more."""
    limits = replace(MUSE, max_tokens=limit)
    wishes = [TO, *head, MSG] + [10, 11][:free] + MUSE_CLOSE + [20]
    tracker, steps = muse_steps(limits, wishes)
    assert steps.out == wishes
    assert tracker.reasoning_tokens == counted
    assert any(steps.suppressed) is (counted == 0)
    assert tracker.thinking_truncated is (counted > 0)


def test_entry_per_family():
    """What a block costs before its first free token: the opener alone
    for <think>, XTML and the channel; opener, label and label end for
    Harmony; the shortest head and its end for Muse, the opener not."""
    think = ReasoningLimits(start=(START,), end=(END,), close=(NL, END, NL2))
    xtml = ReasoningLimits(start=(44, 45, 46), end=(END,), close=(NL, END, NL2))
    harmony = ReasoningLimits(
        start=(START,),
        end=(END,),
        close=(END, START, 41, 42),
        labels=((40,),),
        label_end=(42,),
    )
    assert ThinkingBudget(think)._entry == 1
    assert ThinkingBudget(xtml)._entry == 1
    assert ThinkingBudget(CHANNEL)._entry == 1
    assert ThinkingBudget(harmony)._entry == 3
    assert ThinkingBudget(MUSE)._entry == 2


def test_the_head_counts_like_the_assembler():
    tok = MuseStubTokenizer()
    limits = limits_for(tok, MUSE_PROMPT)
    cases = [
        ([TO, SELF, MSG, 10, 11, EOM, FS, ASST, MSG, 12], 4, "w10 w11 ", "w12 "),
        ([TO, EQ, SELF2, MSG, 10, EOM, FS, ASST, MSG, 12], 4, "w10 ", "w12 "),
        ([TO, EQ, SP, SELF2, MSG, 10, EOM, FS, ASST, MSG, 12], 5, "w10 ", "w12 "),
        ([TO, USER, MSG, 10], 0, "", "w10 "),
    ]
    for ids, counted, reasoning, content in cases:
        tracker, asm, deltas = muse_assembled(limits, ids)
        # The head after ` to` with its marker, the block, not the opener.
        assert tracker.reasoning_tokens == counted, ids
        assert asm.reasoning_tokens == tracker.reasoning_tokens, ids
        assert "".join(d.reasoning for d in deltas) == reasoning, ids
        assert "".join(d.content for d in deltas) == content, ids
    # At the bounds the processor shapes the stream: the emitted sequence
    # goes through a fresh tracker, the assembler's sum is its count.
    for limit in (2, 3, 4):
        bounded = replace(limits, max_tokens=limit)
        wishes = [TO, SELF, MSG] + list(range(10, 18))
        tracker, steps = muse_steps(bounded, wishes, 11)
        again, asm, deltas = muse_assembled(bounded, steps.out)
        assert tracker.reasoning_tokens == again.reasoning_tokens == limit
        assert asm.reasoning_tokens == limit
        free = limit - 2
        assert "".join(d.reasoning for d in deltas) == "".join(
            f"w{t} " for t in range(10, 10 + free)
        )
        assert "".join(d.content for d in deltas) == "".join(
            f"w{t} " for t in range(14 + free, 18)
        )
