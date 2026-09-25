"""The verify path on a tiny hybrid: whatever the proposer drafts - nothing
right, everything right, or a ragged mix - the committed tokens are the
plain greedy transcript, the caches hold exactly the committed tokens, and
finishes inside a block cut the block short."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_beam._vendor.mlx_lm.models import qwen3_5, qwen3_next
from mlx_beam._vendor.mlx_lm.models.cache import ArraysCache, KVCache
from mlx_beam.engine import Engine, EngineDead, GenerationRequest
from mlx_beam.engine.prefix import HEAD
from mlx_beam.engine.proposer import (
    BundledHeadProposer,
    MTPHead,
    load_bundled_head,
    shift_norms,
    trunk,
)
from mlx_beam.engine.request import SamplingParams
from mlx_beam.engine.speculative import rollback_recurrent
from mlx_beam.engine.thinking import ReasoningLimits
from tests.test_engine import TINY_QWEN35_WRAPPED
from tests.test_vendor_mlx_lm import TINY_HYBRID


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
    """Drafts from a script: `start(transcript)` hands it the plain
    continuation, `plan(call)` says how many of a call's drafts are right
    before a wrong one follows. It follows each row's position through the
    script: a plain step feeds one token of the script (`follow`), a cycle
    `1 + accepted` (`commit`); the prompt's last token, fed by the generation
    batch's first step, is followed with no hidden state before it."""

    kind = "oracle"

    def __init__(self, plan, depth=3):
        self.plan = plan
        self.depth = depth
        self.calls = 0
        self.commits = []
        self.dropped = []
        self.followed = []
        self._current = []
        self._pos = {}

    def propose(self, uid, hidden, token, choose=None):
        seq, pos = self._current, self._pos.get(uid, 0)
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
        self._pos[uid] = self._pos.get(uid, 0) + 1 + len(tokens)

    def follow(self, uid, hidden, token):
        self.followed.append(uid)
        if hidden is not None:  # None: the prompt's last token, not the script's
            self._pos[uid] = self._pos.get(uid, 0) + 1

    def begin(self, uid, prompt_len, snapshot):
        pass

    def prime(self, uid, hidden, tokens, start):
        pass

    def snapshot(self, uid):
        return None

    def drop(self, uid):
        self.dropped.append(uid)

    def describe(self):
        return {"kind": self.kind}

    def start(self, seq):
        self._current, self._pos = seq, {}


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


class FixedDraftProposer:
    """Drafts the same tokens on every call: what the target makes of a
    draft must not depend on what was drafted."""

    kind = "fixed"

    def __init__(self, drafts):
        self.drafts = list(drafts)
        self.depth = len(self.drafts)

    def propose(self, uid, hidden, token, choose=None):
        return mx.array(self.drafts, dtype=mx.uint32)

    def commit(self, uid, hidden, tokens):
        pass

    def follow(self, uid, hidden, token):
        pass

    def begin(self, uid, prompt_len, snapshot):
        pass

    def prime(self, uid, hidden, tokens, start):
        pass

    def snapshot(self, uid):
        return None

    def drop(self, uid):
        pass

    def describe(self):
        return {"kind": self.kind}

    def start(self, seq):
        pass


SAMPLED = SamplingParams(temperature=0.9, top_k=16, seed=17)


@pytest.mark.parametrize("mode", ["off", "positions"])
def test_sampled_row_speculates_and_matches_its_plain_seeded_transcript(mode):
    """Keyed by position, the draws inside a verify cycle are the draws a
    plain step would have made: the transcript of a seeded request is the
    same with and without the proposer, however the drafts fall."""
    model = tiny_qwen35()
    prompt = [3, 7, 11, 13]
    n = 24
    with Engine(model) as plain:
        truth = collect(
            plain.submit(GenerationRequest(prompt, max_tokens=n, sampling=SAMPLED))
        )
        assert truth != collect(plain.submit(GenerationRequest(prompt, max_tokens=n)))
    oracle = OracleProposer(lambda call: [0, 3, 1, 2][call % 4])
    engine = run_speculative(model, oracle, exact_verify=mode)
    with engine:
        oracle.start(truth)
        out = collect(
            engine.submit(GenerationRequest(prompt, max_tokens=n, sampling=SAMPLED))
        )
    assert out == truth
    spec = engine.speculator.describe()
    assert spec["cycles"] > 0 and spec["accepted"] > 0


def test_unseeded_sampled_row_speculates():
    model = tiny_qwen35()
    oracle = OracleProposer(lambda call: 3)
    engine = run_speculative(model, oracle)
    with engine:
        oracle.start([])
        before = engine.speculator.cycles
        # Unscripted, the oracle drafts nothing: the fixed proposer below
        # is what runs the cycles.
        engine.speculator.proposer = FixedDraftProposer([1, 2, 3])
        out = collect(
            engine.submit(
                GenerationRequest(
                    [3, 7, 11], max_tokens=12, sampling=SamplingParams(temperature=0.8)
                )
            )
        )
        assert len(out) == 12
        assert engine.speculator.cycles > before


def test_every_draft_leaves_the_sampled_transcript_alone():
    """The enumeration oracle: with the first draft run through the whole
    vocabulary, the committed tokens are the target's own draws every time
    - accepted when the draft hit them, replaced when it did not."""
    model = tiny_qwen35()
    prompt = [3, 7, 11, 13]
    vocab = TINY_QWEN35_WRAPPED["text_config"]["vocab_size"]
    with Engine(model) as plain:
        truth = collect(
            plain.submit(GenerationRequest(prompt, max_tokens=6, sampling=SAMPLED))
        )
    proposer = FixedDraftProposer([0, truth[2], truth[3]])
    engine = Engine(model, proposer=proposer, max_draft_tokens=3)
    engine.warmup_tokens = [1]
    hits = 0
    with engine:
        for draft in range(vocab):
            proposer.drafts[0] = draft
            accepted = engine.speculator.accepted
            out = collect(
                engine.submit(GenerationRequest(prompt, max_tokens=4, sampling=SAMPLED))
            )
            assert out == truth[:4], draft
            if draft == truth[1]:
                hits += 1
                assert engine.speculator.accepted - accepted >= 3
    assert hits == 1


def chi_square(counts, expected):
    return sum((c - e) ** 2 / e for c, e in zip(counts, expected, strict=True))


# Chi-square critical values at p = 0.001 by degrees of freedom.
CRITICAL_001 = {
    1: 10.83, 2: 13.82, 3: 16.27, 4: 18.47, 5: 20.52, 6: 22.46, 7: 24.32,
    8: 26.12, 9: 27.88, 10: 29.59, 11: 31.26, 12: 32.91, 13: 34.53,
    14: 36.12, 15: 37.70, 16: 39.25,
}  # fmt: skip


def test_committed_draws_follow_the_target_under_an_adversarial_draft():
    """Chi-square over the first two tokens of 2048 seeds, drafted wrong on
    purpose (always token 0, never in the top): the first token is a plain
    step's draw, the second the verify's; their joint distribution is the
    target's own, top-k-filtered at each position. Every pair falls in a
    bucket, buckets with an expected count under 5 are pooled."""
    model = tiny_qwen35()
    prompt = [3, 7, 11, 13]
    proposer = FixedDraftProposer([0, 0, 0])
    engine = Engine(model, proposer=proposer, max_draft_tokens=3)
    engine.warmup_tokens = [1]
    n = 2048
    pairs = []
    with engine:
        for seed in range(n):
            p = SamplingParams(temperature=1.0, top_k=4, seed=seed)
            out = collect(
                engine.submit(GenerationRequest(prompt, max_tokens=2, sampling=p))
            )
            pairs.append(tuple(out))
        assert engine.speculator.cycles >= n and engine.speculator.accepted == 0

    def top4(tokens):
        logits = model(mx.array([tokens]))[0, -1]
        logprobs = logits - mx.logsumexp(logits)
        top = mx.argsort(-logprobs)[:4].tolist()
        return dict(zip(top, mx.softmax(logprobs[mx.array(top)]).tolist(), strict=True))

    firsts = top4(prompt)
    expected = {
        (a, b): n * pa * pb
        for a, pa in firsts.items()
        for b, pb in top4([*prompt, a]).items()
    }
    assert set(pairs) <= set(expected)
    counts = {key: pairs.count(key) for key in expected}
    big = [k for k, e in expected.items() if e >= 5]
    small = [k for k in expected if k not in big]
    observed = [counts[k] for k in big]
    wanted = [expected[k] for k in big]
    if small:
        observed.append(sum(counts[k] for k in small))
        wanted.append(sum(expected[k] for k in small))
    assert chi_square(observed, wanted) < CRITICAL_001[len(observed) - 1], (
        observed,
        wanted,
    )


@pytest.mark.parametrize(
    "sampling",
    [
        SamplingParams(repetition_penalty=1.3),
        SamplingParams(presence_penalty=2.0, frequency_penalty=0.5),
    ],
)
def test_penalised_row_speculates_and_matches_plain(sampling):
    """A penalty reads the context: the verify hands every position the
    context a plain step would have, so the transcript is the same."""
    model = tiny_qwen35()
    prompt = [3, 7, 11, 13]
    with Engine(model) as plain:
        free = collect(plain.submit(GenerationRequest(prompt, max_tokens=16)))
        truth = collect(
            plain.submit(GenerationRequest(prompt, max_tokens=16, sampling=sampling))
        )
    assert truth != free
    oracle = OracleProposer(lambda call: [3, 1, 0, 2][call % 4])
    engine = run_speculative(model, oracle)
    with engine:
        oracle.start(truth)
        before = engine.speculator.cycles
        out = collect(
            engine.submit(GenerationRequest(prompt, max_tokens=16, sampling=sampling))
        )
        assert out == truth
        assert engine.speculator.cycles > before


def test_logit_bias_row_speculates_and_matches_plain():
    """A bias is stateless: the verify adds it to every position, so a row
    with one still speculates and still equals its plain transcript."""
    model = tiny_qwen35()
    prompt = [3, 7, 11, 13]
    with Engine(model) as plain:
        first = collect(plain.submit(GenerationRequest(prompt, max_tokens=4)))
        bias = SamplingParams(logit_bias={first[1]: -100.0, first[2]: 3.0})
        truth = collect(
            plain.submit(GenerationRequest(prompt, max_tokens=16, sampling=bias))
        )
    assert truth != first[:4] or first[1] != truth[1]
    oracle = OracleProposer(lambda call: 3)
    engine = run_speculative(model, oracle)
    with engine:
        oracle.start(truth)
        before = engine.speculator.cycles
        out = collect(
            engine.submit(GenerationRequest(prompt, max_tokens=16, sampling=bias))
        )
        assert out == truth
        assert engine.speculator.cycles > before


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


def _flags(events):
    return [(e.token, e.forced, e.thinking_truncated, e.finish_reason) for e in events]


def test_budget_row_speculates_after_the_forced_close():
    """A seeded block cut at its budget: the close itself is forced in
    plain steps (the processor's state changes there), the answer after
    it speculates although the opener stays masked for its rest - the mask
    is applied per verify position. Tokens and flags equal the plain run."""
    model = tiny_qwen35()
    prompt = [3, 7, 11, 13]
    with Engine(model) as plain:
        free = collect(plain.submit(GenerationRequest(prompt, max_tokens=32)))
        unused = [t for t in range(63, 0, -1) if t not in free]
        start, end, nl = unused[:3]
        limits = ReasoningLimits((start,), (end,), (nl, end), seeded=True, max_tokens=6)
        req = GenerationRequest(prompt, max_tokens=32, reasoning=limits)
        truth = list(plain.submit(req))
    tokens = [e.token for e in truth]
    assert tokens[5:7] == [nl, end] and len(tokens) == 32
    oracle = OracleProposer(lambda call: [3, 2, 3, 1][call % 4])
    engine = run_speculative(model, oracle)
    with engine:
        oracle.start(tokens)
        before = engine.speculator.describe()
        out = list(engine.submit(req))
        after = engine.speculator.describe()
    assert _flags(out) == _flags(truth)
    # Cycles ran in the answer: more than one per plain step would allow.
    assert after["cycles"] - before["cycles"] >= 4
    assert after["accepted"] > before["accepted"]


def test_budget_that_arms_inside_a_block_is_left_to_plain_steps():
    """A budget one cycle away from arming holds the row in plain steps for
    that stretch and speculates again once the close is through: the same
    transcript as without a proposer, budget counter included."""
    model = tiny_qwen35()
    prompt = [3, 7, 11, 13]
    with Engine(model) as plain:
        free = collect(plain.submit(GenerationRequest(prompt, max_tokens=24)))
        unused = [t for t in range(63, 0, -1) if t not in free]
        start, end = unused[:2]
        limits = ReasoningLimits((start,), (end,), (end,), seeded=True, max_tokens=9)
        req = GenerationRequest(prompt, max_tokens=24, reasoning=limits)
        truth = list(plain.submit(req))
    tokens = [e.token for e in truth]
    assert tokens[9] == end and truth[-1].thinking_truncated
    oracle = OracleProposer(lambda call: 3)
    engine = run_speculative(model, oracle)
    with engine:
        oracle.start(tokens)
        out = list(engine.submit(req))
    assert _flags(out) == _flags(truth)


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

        def propose(self, uid, hidden, token, choose=None):
            return mx.array([], dtype=mx.uint32)

        def commit(self, *a):
            pass

        follow = begin = prime = commit

        def snapshot(self, uid):
            return None

        def drop(self, uid):
            pass

        def describe(self):
            return {}

    engine = Engine(tiny_qwen35(), proposer=Silent())
    with pytest.raises(EngineDead, match="drafted nothing"):
        engine.start()


# -- the bundled head ------------------------------------------------------


def synthetic_head_checkpoint(
    tmp_path: Path, model, layout: str, manifest_mtp: dict | None = None
) -> Path:
    """A checkpoint directory with a random MTP head: `layout` 'beam' is the
    package layout (config.json `extras.manifest`, the head under `mtp/` as
    `parts.mtp` with its SHA-256; `manifest_mtp` overrides that entry),
    'mtp_file' names the file in config.json only, 'shards' puts the tensors
    into a shard listed in the index, 'none' ships no head."""
    from mlx_beam.package import sha256_of

    args = model.language_model.args
    head = MTPHead(args, type(model.language_model.model.layers[1]))
    mx.eval(head.parameters())
    flat = dict(nn.utils.tree_flatten(head.parameters()))
    cfg = dict(TINY_QWEN35_WRAPPED)
    if layout in ("beam", "mtp_file"):
        (tmp_path / "mtp").mkdir()
        head_file = tmp_path / "mtp" / "weights.safetensors"
        mx.save_safetensors(str(head_file), {f"mtp.{k}": v for k, v in flat.items()})
        cfg["mtp_file"] = "mtp/weights.safetensors"
    if layout == "beam":
        (tmp_path / "extras").mkdir()
        cfg["extras"] = {"manifest": "extras/manifest.json"}
        entry = {
            "file": "mtp/weights.safetensors",
            "sha256": sha256_of(head_file),
            "norm_convention": "hf",
        }
        if manifest_mtp is not None:
            entry = manifest_mtp
        (tmp_path / "extras" / "manifest.json").write_text(
            json.dumps({"format": "beam", "version": 1, "parts": {"mtp": entry}})
        )
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


@pytest.mark.parametrize("layout", ["beam", "mtp_file", "shards"])
def test_bundled_head_loads_quantized_with_shifted_norms(tmp_path, layout):
    model = tiny_qwen35()
    path = synthetic_head_checkpoint(tmp_path, model, layout)
    head, info = load_bundled_head(model, path)
    assert info["kind"] == "mtp" and info["bits"] == 4 and info["norms_shifted"]
    assert info["manifest"] == info["verified"] == (layout == "beam")
    assert isinstance(head.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(head.fc, nn.Linear) and not isinstance(
        head.fc, nn.QuantizedLinear
    )
    raw = mx.load(
        str(
            path
            / (
                "model-00002-of-00002.safetensors"
                if layout == "shards"
                else "mtp/weights.safetensors"
            )
        )
    )
    assert mx.allclose(head.norm.weight, raw["mtp.norm.weight"] + 1.0)


def test_manifest_sets_bits_and_norm_convention_and_checks_the_hash(tmp_path):
    """The package manifest is the source for the head's quantization and
    norm convention; a head whose bytes do not match the manifest is refused."""
    model = tiny_qwen35()
    path = synthetic_head_checkpoint(
        tmp_path,
        model,
        "beam",
        manifest_mtp={
            "file": "mtp/weights.safetensors",
            "bits": 8,
            "group_size": 32,
            "norm_convention": "mlx",
        },
    )
    head, info = load_bundled_head(model, path)
    assert (info["bits"], info["group_size"]) == (8, 32)
    assert not info["norms_shifted"] and info["manifest"] and not info["verified"]
    raw = mx.load(str(path / "mtp/weights.safetensors"))
    assert mx.allclose(head.norm.weight, raw["mtp.norm.weight"])

    manifest = path / "extras" / "manifest.json"
    entry = json.loads(manifest.read_text())
    entry["parts"]["mtp"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="sha256"):
        load_bundled_head(model, path)
    # A part without a file cannot be checked; config.json's mtp_file is no
    # substitute, the manifest never vouched for those bytes.
    del entry["parts"]["mtp"]["file"]
    manifest.write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="names no file"):
        load_bundled_head(model, path)


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
    """The head's KV cache after each cycle: the confirmed pairs and nothing
    else - offset and contents equal a cache fed those pairs alone, whatever
    the chained draft entries the target rejected held in between."""
    model = tiny_qwen35()
    path = synthetic_head_checkpoint(tmp_path, model, "beam")
    head, info = load_bundled_head(model, path)
    proposer = BundledHeadProposer(model, head, 3, info)
    reference = [KVCache()]  # fed the confirmed pairs only, one cycle at a time
    hidden = mx.random.normal((1, 1, 32))
    drafts = proposer.propose(7, hidden, mx.array([5], dtype=mx.uint32))
    mx.eval(drafts)
    assert drafts.shape == (3,)
    cache = proposer._cache[7][0]
    assert cache.offset == 3  # the pair (hidden, 5) and two chained entries
    mx.eval(head(hidden, mx.array([[5]]), proposer._embed, reference))
    # two drafts held: the chained entries go, the two pairs wait for the next call
    confirmed = mx.random.normal((1, 2, 32))
    held = [int(drafts[0].item()), int(drafts[1].item())]
    proposer.commit(7, confirmed, held)
    assert cache.offset == 1 == reference[0].offset
    drafts = proposer.propose(7, hidden, mx.array([9], dtype=mx.uint32))
    mx.eval(drafts)
    assert cache.offset == 1 + 2 + 1 + 2  # pairs so far, the new pair, two chained
    mx.eval(
        head(
            mx.concatenate([confirmed, hidden], axis=1),
            mx.array([held + [9]]),
            proposer._embed,
            reference,
        )
    )
    proposer.commit(7, mx.random.normal((1, 0, 32)), [])
    assert cache.offset == 4 == reference[0].offset
    for got, want in (
        (cache.keys, reference[0].keys),
        (cache.values, reference[0].values),
    ):
        assert mx.array_equal(got[..., :4, :], want[..., :4, :])
    proposer.drop(7)
    assert 7 not in proposer._cache and proposer.describe()["rows_with_history"] == 0


def head_engine(tmp_path, model, **kw):
    path = synthetic_head_checkpoint(tmp_path, model, "beam")
    head, info = load_bundled_head(model, path)
    proposer = BundledHeadProposer(model, head, 3, info)
    engine = Engine(model, proposer=proposer, **kw)
    engine.warmup_tokens = [1]
    return engine, proposer


def head_pairs(model, tokens):
    """(hidden at every position but the last, the tokens after them): the
    pairs a head primed over `tokens` holds, from one forward."""
    inner, _, _ = trunk(model)
    hidden = inner(mx.array([tokens]))
    return hidden[:, :-1, :], tokens[1:]


def test_prefill_primes_the_head_and_the_store_keeps_its_history(tmp_path):
    """The prompt's pairs go through the head in the prefill; at the end of
    the row the head's history - every position of the transcript but the
    last, whose hidden state waits as the tail - sits in the store entry."""
    model = tiny_qwen35()
    prompt = [3, 7, 11, 13, 5, 9, 2, 8]
    truth = plain_transcript(model, prompt, 6)
    engine, proposer = head_engine(tmp_path, model)
    with engine:
        out = collect(engine.submit(GenerationRequest(prompt, max_tokens=6)))
        assert out == truth
        d = proposer.describe()
        # The chunk holds all but the last prompt token: one pair less.
        assert d["primed_pairs"] == len(prompt) - 2
        assert d["rows_with_history"] == 0
        entry = engine.prefix_store._trie.get(engine.model_key, prompt + out)
        keys, values, tail = entry.checkpoints[len(prompt) + len(out)][HEAD]
        assert keys.shape[2] == len(prompt) + len(out) - 1
        assert tail is not None and tail.shape == (1, 1, 32)
        assert entry.nbytes >= keys.nbytes + values.nbytes + tail.nbytes
        # The same pairs, fed to a fresh head in one call, give the same cache.
        hidden, nexts = head_pairs(model, prompt + out)
        reference = [KVCache()]
        mx.eval(proposer.head(hidden, mx.array([nexts]), proposer._embed, reference))
        ref_keys, ref_values = reference[0].keys_and_values()
        assert mx.allclose(keys, ref_keys, atol=1e-5, rtol=1e-4)
        assert mx.allclose(values, ref_values, atol=1e-5, rtol=1e-4)


def test_next_turn_resumes_the_head_history_from_the_store(tmp_path):
    """A conversation continued: the store restores the entry whole and the
    head's history with it; the prefill primes only the new tokens. The
    transcript stays the plain one."""
    model = tiny_qwen35()
    prompt = [3, 7, 11, 13, 5, 9]
    engine, proposer = head_engine(tmp_path, model)
    with engine:
        first = collect(engine.submit(GenerationRequest(prompt, max_tokens=5)))
        primed = proposer.describe()["primed_pairs"]
        turn2 = prompt + first + [21, 22, 23, 24]
        truth = plain_transcript(model, turn2, 5)
        stream = engine.submit(GenerationRequest(turn2, max_tokens=5))
        out = collect(stream)
        assert out == truth
        assert stream.prompt_cached == len(prompt) + len(first)
        d = proposer.describe()
        assert d["histories_restored"] == 1
        # Three new tokens in the chunk (the fourth is fed by the first
        # step): the tail's pair plus two pairs inside the chunk.
        assert d["primed_pairs"] - primed == 3
        entry = engine.prefix_store._trie.get(engine.model_key, turn2 + out)
        keys, _, tail = entry.checkpoints[len(turn2) + len(out)][HEAD]
        assert keys.shape[2] == len(turn2) + len(out) - 1 and tail is not None


def test_boundary_checkpoint_carries_the_head_and_a_prefix_hit_resumes_it(tmp_path):
    """A request that shares only the system block resumes the head from
    the boundary's sidecar; one that shares nothing starts the head empty."""
    model = tiny_qwen35()
    system = [3, 7, 11, 13, 5, 9, 2, 8]
    engine, proposer = head_engine(tmp_path, model)
    with engine:
        first = collect(
            engine.submit(
                GenerationRequest(
                    system + [30, 31, 32], max_tokens=4, boundaries=[len(system)]
                )
            )
        )
        entry = engine.prefix_store._trie.get(
            engine.model_key, system + [30, 31, 32] + first
        )
        keys, _, tail = entry.checkpoints[len(system)][HEAD]
        assert keys.shape[2] == len(system) - 1 and tail is not None
        other = system + [40, 41, 42, 43]
        truth = plain_transcript(model, other, 4)
        stream = engine.submit(
            GenerationRequest(other, max_tokens=4, boundaries=[len(system)])
        )
        assert collect(stream) == truth
        assert stream.prompt_cached == len(system)
        assert proposer.describe()["histories_restored"] == 1
        stream = engine.submit(GenerationRequest([50, 51, 52, 53], max_tokens=2))
        collect(stream)
        assert stream.prompt_cached == 0
        assert proposer.describe()["histories_restored"] == 1


def test_head_history_survives_plain_steps_beside_another_row(tmp_path):
    """Two rows decode plainly (no verify beside another row); the pairs
    they go through are queued and the histories are whole at the end."""
    model = tiny_qwen35()
    a, b = [3, 7, 11, 13], [5, 9, 13, 17, 21]
    engine, proposer = head_engine(tmp_path, model)
    with engine:
        sa = engine.submit(GenerationRequest(a, max_tokens=8))
        sb = engine.submit(GenerationRequest(b, max_tokens=8))
        oa, ob = collect(sa), collect(sb)
        for prompt, out in ((a, oa), (b, ob)):
            entry = engine.prefix_store._trie.get(engine.model_key, prompt + out)
            keys, _, tail = entry.checkpoints[len(prompt) + len(out)][HEAD]
            assert keys.shape[2] == len(prompt) + len(out) - 1 and tail is not None


def test_history_window_and_queue_cap(tmp_path, monkeypatch):
    """A prompt longer than the window is primed from its last positions
    on, as a fresh head that saw only those; a queue past its cap restarts
    the history from the queue alone."""
    from mlx_beam.engine import proposer as proposer_mod

    monkeypatch.setattr(proposer_mod, "HEAD_HISTORY", 4)
    model = tiny_qwen35()
    prompt = [3, 7, 11, 13, 5, 9, 2, 8, 6, 4]
    engine, proposer = head_engine(tmp_path, model)
    with engine:
        out = collect(engine.submit(GenerationRequest(prompt, max_tokens=3)))
        # Window 4 of 10: pairs at positions 6 and 7 from the chunk (8 sits
        # on the prompt's last token, fed by the first step).
        assert proposer.describe()["primed_pairs"] == 2
        entry = engine.prefix_store._trie.get(engine.model_key, prompt + out)
        keys, _, _ = entry.checkpoints[len(prompt) + len(out)][HEAD]
        assert keys.shape[2] == 4 + len(out) - 1
    monkeypatch.setattr(proposer_mod, "PENDING_CAP", 3)
    (tmp_path / "second").mkdir()
    engine, proposer = head_engine(tmp_path / "second", model)
    with engine:
        sa = engine.submit(GenerationRequest([3, 7, 11], max_tokens=8))
        sb = engine.submit(GenerationRequest([5, 9, 13], max_tokens=8))
        oa, _ = collect(sa), collect(sb)
        entry = engine.prefix_store._trie.get(engine.model_key, [3, 7, 11] + oa)
        keys, _, _ = entry.checkpoints[3 + len(oa)][HEAD]
        assert keys.shape[2] <= 3 + 1


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


# -- the recurrent rollback on the real kernels ------------------------------


def gdn_layer(family: str, head_dim: int, dtype):
    """One gated-delta layer of either family with a key width the Metal
    kernel takes (the suite's tiny models use 16, which runs the ops path)."""
    mx.random.seed(91)
    if family == "qwen3_5":
        cfg = dict(TINY_QWEN35_WRAPPED["text_config"])
        cfg.update(linear_key_head_dim=head_dim, linear_value_head_dim=32)
        layer = qwen3_5.GatedDeltaNet(qwen3_5.TextModelArgs(**cfg))
    else:
        cfg = dict(TINY_HYBRID)
        cfg.update(linear_key_head_dim=head_dim, linear_value_head_dim=32)
        layer = qwen3_next.Qwen3NextGatedDeltaNet(qwen3_next.ModelArgs(**cfg))
    layer.set_dtype(dtype)
    layer.eval()
    mx.eval(layer.parameters())
    return layer


@pytest.mark.exactness
@pytest.mark.skipif(not mx.metal.is_available(), reason="the Metal kernels")
@pytest.mark.parametrize("family", ["qwen3_5", "qwen3_next"])
@pytest.mark.parametrize("head_dim", [32, 128])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("keep", [1, 2, 3])
def test_rollback_on_metal_matches_a_shorter_forward(family, head_dim, dtype, keep):
    """A four-token verify rolled back to `keep` tokens against a forward of
    just those tokens from the same state. Bit for bit where both run the
    same kernels (bf16 everywhere; float32 from two tokens on, where the
    projections take the GEMM path on both sides); at one token in float32
    the projections come from GEMV and differ by ~1e-8, which the rollback
    cannot undo - measured 2026-09-24, the contract in speculative.py."""
    layer = gdn_layer(family, head_dim, dtype)
    x = mx.random.normal((1, 9, 32)).astype(dtype)
    spec, plain = ArraysCache(2), ArraysCache(2)
    mx.eval(layer(x[:, :5], cache=spec), layer(x[:, :5], cache=plain))
    spec.stash = {}
    mx.eval(layer(x[:, 5:], cache=spec))
    assert spec.stash["use_kernel"] and spec.stash["k"].shape[-1] == head_dim
    rollback_recurrent(spec, keep, 4)
    spec.stash = None
    mx.eval(layer(x[:, 5 : 5 + keep], cache=plain))
    mx.eval(*spec.cache, *plain.cache)
    for got, want in zip(spec.cache, plain.cache, strict=True):
        g, w = got.astype(mx.float32), want.astype(mx.float32)
        assert mx.allclose(g, w, atol=1e-5, rtol=1e-4)
        if keep > 1 or dtype == mx.bfloat16:
            assert mx.array_equal(got, want)


# -- the Mamba-2 rollback (Granite 4) ----------------------------------------


def mamba2_layer(dtype=mx.float32):
    from mlx_beam._vendor.mlx_lm.models import granitemoehybrid
    from tests.test_model_classes import CLASSES

    mx.random.seed(93)
    args = granitemoehybrid.ModelArgs.from_dict(CLASSES["granitemoehybrid"]["cfg"])
    layer = granitemoehybrid.GraniteMoeHybridMamba2Mixer(args)
    layer.set_dtype(dtype)
    layer.eval()
    mx.eval(layer.parameters())
    return layer


@pytest.mark.parametrize("keep", [1, 2, 3])
def test_mamba2_rollback_matches_a_shorter_forward(keep):
    """The Mamba-2 replay from the stash: a four-token verify rolled back to
    `keep` tokens equals a forward of just those tokens from the same state
    - conv window and SSM state alike. The selective-scan update runs the
    same ops on both sides, so the states match to the last bit here."""
    layer = mamba2_layer()
    x = mx.random.normal((1, 9, 64))
    spec, plain = ArraysCache(2), ArraysCache(2)
    mx.eval(
        layer(x[:, :5], mask=None, cache=spec), layer(x[:, :5], mask=None, cache=plain)
    )
    spec.stash = {}
    mx.eval(layer(x[:, 5:], mask=None, cache=spec))
    assert spec.stash["kind"] == "mamba2"
    rollback_recurrent(spec, keep, 4)
    spec.stash = None
    mx.eval(layer(x[:, 5 : 5 + keep], mask=None, cache=plain))
    mx.eval(*spec.cache, *plain.cache)
    for got, want in zip(spec.cache, plain.cache, strict=True):
        assert mx.allclose(got, want, atol=1e-5, rtol=1e-4)
        assert mx.array_equal(got, want)


# -- the sliding-window rollback -----------------------------------------------


@pytest.mark.parametrize("keep", [1, 2, 3, 4])
@pytest.mark.parametrize("fill", [3, 8, 13])
def test_window_rollback_matches_a_shorter_feed(keep, fill):
    """A rotated ring cannot trim (its stale tail stays), so the verify
    restores the state before the cycle and writes the kept tokens again.
    The window then holds exactly what feeding those tokens alone would
    have left - before the ring is full, at the fill line, and rotated."""
    from mlx_beam._vendor.mlx_lm.models.cache import BatchRotatingKVCache
    from mlx_beam.engine.speculative import rollback_window

    mx.random.seed(7)
    keys = mx.random.normal((1, 1, fill + 4, 4))
    values = mx.random.normal((1, 1, fill + 4, 4))
    spec, plain = (BatchRotatingKVCache(8, [0]) for _ in range(2))
    for i in range(fill):  # one token at a time, the decode path
        for c in (spec, plain):
            c.update_and_fetch(keys[..., i : i + 1, :], values[..., i : i + 1, :])
    before = tuple(v[:] if isinstance(v, mx.array) else v for v in spec.state)
    spec.update_and_fetch(keys[..., fill:, :], values[..., fill:, :])  # the verify
    rollback_window(spec, before, keep, 4)
    plain.update_and_fetch(
        keys[..., fill : fill + keep, :], values[..., fill : fill + keep, :]
    )
    for c in (spec, plain):
        c._temporal_order()
    got_k, got_v = spec.keys_and_values()
    want_k, want_v = plain.keys_and_values()
    assert spec.offset.tolist() == plain.offset.tolist()
    assert mx.array_equal(
        got_k[..., -min(8, fill + keep) :, :], want_k[..., -min(8, fill + keep) :, :]
    )
    assert mx.array_equal(
        got_v[..., -min(8, fill + keep) :, :], want_v[..., -min(8, fill + keep) :, :]
    )


# -- the exact verify: class swap, context, width check -----------------------


def test_exact_install_swaps_the_quantized_projections_and_back():
    import mlx.nn as nn

    from mlx_beam.engine import exact
    from tests.test_model_classes import tiny

    model = tiny("gpt_oss")
    nn.quantize(model, group_size=32, bits=8)
    mx.eval(model.parameters())
    counts = exact.install(model)
    assert counts["linear"] > 0 and counts["switch"] > 0
    kinds = {type(m).__name__ for _, m in model.named_modules()}
    assert "ExactQuantizedLinear" in kinds and "QuantizedLinear" not in kinds
    x = mx.array([[3, 7, 11, 13]])
    # Outside the context the swapped classes run the stock path: same bytes.
    before = model(x)
    exact.uninstall(model)
    kinds = {type(m).__name__ for _, m in model.named_modules()}
    assert "ExactQuantizedLinear" not in kinds and "QuantizedLinear" in kinds
    assert mx.array_equal(before, model(x))


def test_width_check_reports_the_probe_and_the_kernel_path():
    from mlx_beam.engine import KVPolicy, exact
    from mlx_beam.engine.speculative import width_check
    from tests.test_model_classes import tiny

    model = tiny("mistral3")
    plain = width_check(model, KVPolicy(), [3, 7, 11, 13, 17, 19, 23, 29], 4)
    assert plain["width"] == 4 and plain["path"] == "block"
    assert set(plain) >= {
        "block_equals_positions",
        "max_abs_logit_diff",
        "argmax_equal",
    }
    exact.install(model)
    kernels = width_check(
        model, KVPolicy(), [3, 7, 11, 13, 17, 19, 23, 29], 4, exact=True
    )
    assert kernels["path"] == "kernels"
    # Per-position projections and per-query attention: exact by construction
    # where no Metal kernel is involved (the CPU here).
    assert kernels["block_equals_positions"] is True
