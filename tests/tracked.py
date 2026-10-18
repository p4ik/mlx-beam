"""Token events the way the engine builds them, without a model: the ids
go through a ThinkingBudget, and each event carries what the budget said."""

from __future__ import annotations

from mlx_beam.api.chat import reasoning_limits
from mlx_beam.engine.request import TokenEvent
from mlx_beam.engine.thinking import ThinkingBudget
from tests.stub_tokenizer import EOS


def limits_for(tok, prompt_tokens, max_tokens=None):
    """The limits the server derives for this tokenizer and prompt."""
    return reasoning_limits(tok, prompt_tokens, max_tokens)


def tracked_events(limits, ids, finish="stop", eos=EOS, tracker=None):
    """The events the engine would build for `ids`, the sequence as it was
    emitted (forced closes included): observe, then read, as _on_step and
    _loop do. `finish` goes on the last id, or on the `eos` id appended for
    a natural stop (None: nothing appended). `tracker` is a fresh budget
    for `limits` the caller wants to read afterwards."""
    ids = list(ids) + ([eos] if finish == "stop" and eos is not None else [])
    if tracker is None and limits is not None:
        tracker = ThinkingBudget(limits)
    out = []
    for i, t in enumerate(ids):
        fr = finish if i == len(ids) - 1 else None
        if tracker is None:  # a model without thinking: no budget, zeros
            out.append(TokenEvent(t, -0.1, fr, response_truncated=fr == "length"))
            continue
        before = tracker.reasoning_tokens
        tracker.observe(t, fr)
        out.append(
            TokenEvent(
                t,
                -0.1,
                fr,
                thinking_truncated=tracker.thinking_truncated,
                response_truncated=fr == "length" and not tracker.in_reasoning,
                forced=tracker.last_forced,
                suppressed=tracker.last_suppressed,
                reasoning_counted=tracker.reasoning_tokens - before,
            )
        )
    return out
