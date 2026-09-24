"""The verify path on a tiny hybrid: whatever the proposer drafts - nothing
right, everything right, or a ragged mix - the committed tokens are the
plain greedy transcript, the caches hold exactly the committed tokens, and
finishes inside a block cut the block short."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_beam._vendor.mlx_lm.models import qwen3_5
from mlx_beam.engine import Engine, EngineDead, GenerationRequest
from mlx_beam.engine.proposer import (
    BundledHeadProposer,
    MTPHead,
    load_bundled_head,
    shift_norms,
)
from mlx_beam.engine.request import SamplingParams
from mlx_beam.engine.thinking import ReasoningLimits
from tests.test_engine import TINY_QWEN35_WRAPPED


def tiny_qwen35(seed=0):
    mx.random.seed(seed)
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(TINY_QWEN35_WRAPPED))
    mx.eval(model.parameters())
    return model


def collect(stream):
    return [e.token for e in stream]


def plain_transcript(model, prompt, n):
    with Engine(model) as engine:
        return collect(engine.submit(GenerationRequest(prompt, max_tokens=n)))


class OracleProposer:
    """Drafts from a script: `start(transcript)` hands it the plain greedy
    continuation, `plan(call)` says how many of a call's drafts are right
    before a wrong one follows."""

    kind = "oracle"

    def __init__(self, plan, depth=3):
        self.plan = plan
        self.depth = depth
        self.calls = 0
        self.commits = []
        self.dropped = []

    def propose(self, uid, hidden, token):
        seq, pos = self._current, self._pos
        if not seq:
            return mx.array([], dtype=mx.uint32)  # unscripted: decode plainly
        self.calls += 1
        t = int(token.item())
        assert seq[pos] == t, (seq[pos], t, pos)
        right = self.plan(self.calls)
        drafts = []
        for i in range(self.depth):
            true = seq[pos + 1 + i] if pos + 1 + i < len(seq) else 0
            drafts.append(true if i < right else (true + 1) % 64)
        return mx.array(drafts, dtype=mx.uint32)

    def commit(self, uid, hidden, tokens):
        self.commits.append(list(tokens))
        self._pos += 1 + len(tokens)

    def drop(self, uid):
        self.dropped.append(uid)

    def describe(self):
        return {"kind": self.kind}

    def start(self, seq):
        self._current, self._pos = seq, 0


def warm_script(model):
    """What the warm-up row generates: the oracle must script it too."""
    return plain_transcript(model, [1], 8)


def run_speculative(model, proposer, **kw):
    """An engine whose warm-up the oracle can follow; the caller scripts the
    request's transcript after entering the context."""
    engine = Engine(model, proposer=proposer, max_draft_tokens=proposer.depth, **kw)
    engine.warmup_tokens = [1]
    proposer.start(warm_script(model))
    return engine


@pytest.mark.parametrize(
    "name,plan",
    [
        ("nothing", lambda call: 0),
        ("everything", lambda call: 3),
        ("ragged", lambda call: [0, 3, 1, 2, 0, 1][call % 6]),
    ],
)
def test_committed_tokens_equal_plain_greedy(name, plan):
    model = tiny_qwen35()
    prompt = [3, 7, 11, 13, 5]
    n = 24
    truth = plain_transcript(model, prompt, n)
    oracle = OracleProposer(plan)
    engine = run_speculative(model, oracle)
    with engine:
        oracle.start(truth)
        out = collect(engine.submit(GenerationRequest(prompt, max_tokens=n)))
    assert out == truth
    spec = engine.speculator.describe()
    assert spec["cycles"] > 0
    if name == "everything":
        assert spec["accepted"] == spec["drafted"]
    if name == "nothing":
        assert spec["accepted"] == 0


def test_health_reports_the_cycle_counters():
    model = tiny_qwen35()
    oracle = OracleProposer(lambda call: 3)
    engine = run_speculative(model, oracle)
    with engine:
        h = engine.health()["speculative"]
        assert h["proposer"]["kind"] == "oracle" and h["depth"] == 3
        assert h["cycles"] >= 1 and h["parked"] is False


def test_finish_inside_a_block_trims_the_block():
    """A stop token accepted mid-block: the tokens after it are not emitted,
    and the finishing row's cache holds exactly prompt + emitted tokens."""
    model = tiny_qwen35()
    prompt = [3, 7, 11]
    truth = plain_transcript(model, prompt, 12)
    stop = truth[5]
    oracle = OracleProposer(lambda call: 3)
    engine = run_speculative(model, oracle)
    with engine:
        oracle.start(truth)
        events = list(
            engine.submit(
                GenerationRequest(prompt, max_tokens=12, stop_sequences=[[stop]])
            )
        )
    tokens = [e.token for e in events]
    assert events[-1].finish_reason == "stop"
    assert tokens == truth[: truth.index(stop) + 1]
    # the stored entry covers prompt + emitted tokens, nothing past the stop
    entry = engine.prefix_store._trie.get("model", prompt + tokens)
    assert entry is not None and entry.length == len(prompt) + len(tokens)
    attention = [c for c in entry.cache if hasattr(c, "offset")]
    assert attention and all(c.offset == entry.length for c in attention)


def test_length_limit_inside_a_block():
    model = tiny_qwen35()
    prompt = [3, 7, 11]
    truth = plain_transcript(model, prompt, 7)
    oracle = OracleProposer(lambda call: 3)
    engine = run_speculative(model, oracle)
    with engine:
        oracle.start(truth)
        events = list(engine.submit(GenerationRequest(prompt, max_tokens=7)))
    assert [e.token for e in events] == truth and events[-1].finish_reason == "length"


def test_two_rows_decode_plainly_and_alone_speculates_again():
    model = tiny_qwen35()
    a, b = [3, 7, 11, 13], [5, 9, 13, 17, 21]
    ta, tb = plain_transcript(model, a, 10), plain_transcript(model, b, 10)
    oracle = OracleProposer(lambda call: 3)
    engine = run_speculative(model, oracle)
    with engine:
        # Nothing scripted: two rows never ask the proposer, and the one that
        # ends alone gets no drafts either - both decode plainly, together.
        oracle.start([])
        asked = oracle.calls
        sa = engine.submit(GenerationRequest(a, max_tokens=10))
        sb = engine.submit(GenerationRequest(b, max_tokens=10))
        assert collect(sa) == ta and collect(sb) == tb
        assert oracle.calls == asked
    assert engine.speculator.describe()["plain_steps"] > 0


def test_non_greedy_and_processed_rows_decode_plainly():
    model = tiny_qwen35()
    oracle = OracleProposer(lambda call: 3)
    engine = run_speculative(model, oracle)
    with engine:
        before = engine.speculator.cycles
        oracle.start([])
        collect(
            engine.submit(
                GenerationRequest(
                    [3, 7, 11], max_tokens=6, sampling=SamplingParams(temperature=0.8)
                )
            )
        )
        collect(
            engine.submit(
                GenerationRequest(
                    [3, 7, 11],
                    max_tokens=6,
                    sampling=SamplingParams(logit_bias={2: -5.0}),
                )
            )
        )
        assert engine.speculator.cycles == before


def test_thinking_budget_row_speculates_while_inert():
    model = tiny_qwen35()
    prompt = [3, 7, 11, 13]
    limits = ReasoningLimits(start=(40,), end=(41,), close=(41,), max_tokens=64)
    with Engine(model) as plain:
        truth = collect(
            plain.submit(GenerationRequest(prompt, max_tokens=16, reasoning=limits))
        )
    oracle = OracleProposer(lambda call: 3)
    engine = run_speculative(model, oracle)
    with engine:
        oracle.start(truth)
        before = engine.speculator.cycles
        out = collect(
            engine.submit(GenerationRequest(prompt, max_tokens=16, reasoning=limits))
        )
        assert out == truth
        assert engine.speculator.cycles > before


def test_cancel_inside_a_cycle_keeps_the_worker():
    model = tiny_qwen35()
    prompt = [3, 7, 11, 13]
    truth = plain_transcript(model, prompt, 40)
    oracle = OracleProposer(lambda call: 3)
    engine = run_speculative(model, oracle)
    with engine:
        oracle.start(truth)
        stream = engine.submit(GenerationRequest(prompt, max_tokens=40))
        got = [stream.next_event().token for _ in range(3)]
        stream.cancel()
        rest = collect(stream)
        assert got == truth[:3] and got + rest == truth[: 3 + len(rest)]
        assert engine.alive
        oracle.start(truth)
        assert (
            collect(engine.submit(GenerationRequest(prompt, max_tokens=6))) == truth[:6]
        )


def test_proposer_that_drafts_nothing_fails_the_warm_up():
    class Silent:
        kind, depth = "silent", 3

        def propose(self, uid, hidden, token):
            return mx.array([], dtype=mx.uint32)

        def commit(self, *a):
            pass

        def drop(self, uid):
            pass

        def describe(self):
            return {}

    engine = Engine(tiny_qwen35(), proposer=Silent())
    with pytest.raises(EngineDead, match="drafted nothing"):
        engine.start()


# -- the bundled head ------------------------------------------------------


def synthetic_head_checkpoint(tmp_path: Path, model, layout: str) -> Path:
    """A checkpoint directory with a random MTP head: `layout` 'beam' names
    the file in config.json, 'shards' puts the tensors into a shard listed in
    the index, 'none' ships no head."""
    args = model.language_model.args
    head = MTPHead(args, type(model.language_model.model.layers[1]))
    mx.eval(head.parameters())
    flat = dict(nn.utils.tree_flatten(head.parameters()))
    cfg = dict(TINY_QWEN35_WRAPPED)
    if layout == "beam":
        (tmp_path / "mtp").mkdir()
        mx.save_safetensors(
            str(tmp_path / "mtp" / "weights.safetensors"),
            {f"mtp.{k}": v for k, v in flat.items()},
        )
        cfg["mtp_file"] = "mtp/weights.safetensors"
        cfg["beam"] = {"mtp": {"norm_convention": "hf"}}
    elif layout == "shards":
        mx.save_safetensors(
            str(tmp_path / "model-00002-of-00002.safetensors"),
            {f"mtp.{k}": v for k, v in flat.items()},
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        f"mtp.{k}": "model-00002-of-00002.safetensors" for k in flat
                    }
                }
            )
        )
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path


@pytest.mark.parametrize("layout", ["beam", "shards"])
def test_bundled_head_loads_quantized_with_shifted_norms(tmp_path, layout):
    model = tiny_qwen35()
    path = synthetic_head_checkpoint(tmp_path, model, layout)
    head, info = load_bundled_head(model, path)
    assert info["kind"] == "mtp" and info["bits"] == 4 and info["norms_shifted"]
    assert isinstance(head.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(head.fc, nn.Linear) and not isinstance(
        head.fc, nn.QuantizedLinear
    )
    raw = mx.load(
        str(
            path
            / (
                "mtp/weights.safetensors"
                if layout == "beam"
                else "model-00002-of-00002.safetensors"
            )
        )
    )
    assert mx.allclose(head.norm.weight, raw["mtp.norm.weight"] + 1.0)


def test_shift_norms_touches_all_seven_and_nothing_else():
    w = {
        "norm.weight": mx.zeros(4),
        "layers.0.self_attn.q_norm.weight": mx.ones(4),
        "layers.0.self_attn.q_proj.weight": mx.ones((4, 4)),
    }
    out = shift_norms(w)
    assert mx.array_equal(out["norm.weight"], mx.ones(4))
    assert mx.array_equal(out["layers.0.self_attn.q_norm.weight"], mx.ones(4) * 2)
    assert (
        out["layers.0.self_attn.q_proj.weight"] is w["layers.0.self_attn.q_proj.weight"]
    )


def test_missing_head_is_named(tmp_path):
    model = tiny_qwen35()
    path = synthetic_head_checkpoint(tmp_path, model, "none")
    with pytest.raises(FileNotFoundError, match="bundles no draft head"):
        load_bundled_head(model, path)


def test_random_head_keeps_plain_greedy_output(tmp_path):
    """A head that predicts nothing useful: the verify still yields the plain
    transcript, at the cost of cycles that accept little."""
    model = tiny_qwen35()
    path = synthetic_head_checkpoint(tmp_path, model, "beam")
    head, info = load_bundled_head(model, path)
    prompt = [3, 7, 11, 13, 5, 9]
    truth = plain_transcript(model, prompt, 32)
    engine = Engine(model, proposer=BundledHeadProposer(model, head, 3, info))
    with engine:
        out = collect(engine.submit(GenerationRequest(prompt, max_tokens=32)))
        h = engine.health()["speculative"]
    assert out == truth
    assert h["proposer"]["kind"] == "mtp" and h["cycles"] >= 8
    assert h["proposer"]["rows_with_history"] == 0  # dropped at finish


def test_head_history_holds_exactly_the_confirmed_pairs(tmp_path):
    """The head's KV cache after each call: the confirmed pairs, never the
    chained draft entries the target rejected."""
    model = tiny_qwen35()
    path = synthetic_head_checkpoint(tmp_path, model, "beam")
    head, info = load_bundled_head(model, path)
    proposer = BundledHeadProposer(model, head, 3, info)
    hidden = mx.random.normal((1, 1, 32))
    drafts = proposer.propose(7, hidden, mx.array([5], dtype=mx.uint32))
    mx.eval(drafts)
    assert drafts.shape == (3,)
    cache = proposer._cache[7][0]
    assert cache.offset == 3  # the pair (hidden, 5) and two chained entries
    # two drafts held: the chained entries go, the two pairs wait for the next call
    proposer.commit(
        7, mx.random.normal((1, 2, 32)), [int(drafts[0].item()), int(drafts[1].item())]
    )
    assert cache.offset == 1
    drafts = proposer.propose(7, hidden, mx.array([9], dtype=mx.uint32))
    mx.eval(drafts)
    assert cache.offset == 1 + 2 + 1 + 2  # pairs so far, the new pair, two chained
    proposer.commit(7, mx.random.normal((1, 0, 32)), [])
    assert cache.offset == 4
    proposer.drop(7)
    assert 7 not in proposer._cache and proposer.describe()["rows_with_history"] == 0


def test_cli_resolves_the_proposer_from_the_flag(tmp_path, caplog):
    import logging
    from types import SimpleNamespace

    from mlx_beam.cli import proposer_from_args

    model = tiny_qwen35()
    path = synthetic_head_checkpoint(tmp_path, model, "beam")
    log = logging.getLogger("beam-test")
    args = SimpleNamespace(model=str(path), draft_model=None, max_draft_tokens=3)
    with caplog.at_level(logging.INFO, logger="beam-test"):
        assert proposer_from_args(args, model, path, log) is None
    assert "bundles a draft head" in caplog.text  # a hint, not a default
    args.draft_model = "some/repo"
    with pytest.raises(ValueError, match="only 'bundled'"):
        proposer_from_args(args, model, path, log)
    args.draft_model = "bundled"
    proposer = proposer_from_args(args, model, path, log)
    assert proposer.kind == "mtp" and proposer.depth == 3
