"""The vendored mlx-lm: self-contained import, and the local changes listed in
VENDORED.md hold on tiny random-weight models and on the caches alone."""

import subprocess
import sys

import mlx.core as mx
import pytest

from mlx_beam._vendor import mlx_lm as vendored
from mlx_beam._vendor.mlx_lm.generate import BatchGenerator
from mlx_beam._vendor.mlx_lm.models import qwen3_next
from mlx_beam._vendor.mlx_lm.models.cache import ArraysCache, BatchRotatingKVCache

TINY_HYBRID = dict(
    model_type="qwen3_next",
    hidden_size=32,
    num_hidden_layers=4,
    intermediate_size=64,
    num_attention_heads=2,
    linear_num_value_heads=2,
    linear_num_key_heads=2,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    num_experts=4,
    num_experts_per_tok=2,
    decoder_sparse_step=1,
    shared_expert_intermediate_size=32,
    mlp_only_layers=[],
    moe_intermediate_size=32,
    rms_norm_eps=1e-6,
    vocab_size=64,
    num_key_value_heads=1,
    rope_theta=10000.0,
    partial_rotary_factor=0.25,
    max_position_embeddings=512,
    head_dim=16,
    full_attention_interval=2,
)


def tiny_hybrid(seed=0):
    mx.random.seed(seed)
    model = qwen3_next.Model(qwen3_next.ModelArgs(**TINY_HYBRID))
    mx.eval(model.parameters())
    return model


def greedy_tokens(model, prompts, max_tokens, **kw):
    gen = BatchGenerator(model, max_tokens=max_tokens, stop_tokens=[], **kw)
    uids = gen.insert(prompts)
    out = {u: [] for u in uids}
    while any(len(t) < max_tokens for t in out.values()):
        _, generated = gen.next()
        for r in generated:
            out[r.uid].append(r.token)
    gen.close()
    return [out[u] for u in uids]


def test_import_is_self_contained():
    code = (
        "import sys; import mlx_beam._vendor.mlx_lm as v; "
        "from mlx_beam._vendor.mlx_lm.models import cache, qwen3_next, llama; "
        "assert 'mlx_lm' not in sys.modules, 'top-level mlx_lm imported'; "
        "print(v.UPSTREAM_COMMIT)"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == vendored.UPSTREAM_COMMIT


def test_arrays_cache_mask_prefers_lengths():
    # merge() of fresh caches leaves left_padding = [0, 0]; a right-padded
    # prefill then sets lengths, and lengths must decide the mask.
    merged = ArraysCache.merge([ArraysCache(1), ArraysCache(1)])
    assert merged.left_padding.tolist() == [0, 0]
    merged.prepare(lengths=[3, 5])
    mask = merged.make_mask(5)
    assert mask.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    merged.finalize()
    assert merged.make_mask(5) is None


def test_batch_matches_solo_on_hybrid():
    # Two prompts of different length share a prefill batch; each must decode
    # exactly as it does alone. The scheduler never pads (width = shortest
    # segment) and the hand-over of the short one must not disturb the other.
    model = tiny_hybrid()
    short = [3, 7, 11]
    long = [5, 9, 13, 17, 21, 25, 29, 33, 37, 41]
    solo_short = greedy_tokens(model, [short], 8)[0]
    solo_long = greedy_tokens(model, [long], 8)[0]
    batched = greedy_tokens(model, [short, long], 8, prefill_batch_size=2)
    assert batched[0] == solo_short
    assert batched[1] == solo_long


def test_prefill_slice_keeps_output():
    # Chunk boundaries change only the reduction order, not the tokens.
    model = tiny_hybrid()
    prompt = list(range(1, 40))
    wide = greedy_tokens(model, [prompt], 6, prefill_step_size=64, prefill_slice=64)
    narrow = greedy_tokens(model, [prompt], 6, prefill_step_size=64, prefill_slice=5)
    assert wide == narrow


def test_scheduler_alternates_under_sharing():
    # A decoding sequence keeps receiving tokens while a long prompt prefills.
    model = tiny_hybrid()
    gen = BatchGenerator(
        model, max_tokens=6, stop_tokens=[], prefill_slice=4, prefill_batch_size=1
    )
    (first,) = gen.insert([[3, 7, 11]])
    seen_first = 0
    while seen_first < 2:
        seen_first += sum(r.uid == first for r in gen.next()[1])
    (second,) = gen.insert([list(range(1, 30))])
    calls = 0
    first_tokens_during_prefill = 0
    while True:
        _, generated = gen.next()
        calls += 1
        first_tokens_during_prefill += sum(r.uid == first for r in generated)
        if any(r.uid == second for r in generated):
            break
        assert calls < 50
    gen.close()
    assert first_tokens_during_prefill >= 1


def test_decode_burst_ends_when_a_row_finishes():
    """While a prefill shares the worker, decode runs for a time budget; a
    row that finishes inside that burst ends it, so its extracted cache is
    evaluated before the next step instead of forcing batch-wide copies."""
    model = tiny_hybrid()
    gen = BatchGenerator(
        model, max_tokens=40, stop_tokens=[], prefill_slice=4, prefill_batch_size=2
    )
    short, long_ = gen.insert([[3, 7, 11], [5, 9, 13]], max_tokens=[3, 40])
    seen = {short: 0, long_: 0}
    while min(seen.values()) < 1:
        for r in gen.next()[1]:
            seen[r.uid] += 1
    # A prefill now shares the worker; give every burst a budget of seconds.
    gen.insert([list(range(1, 30))])
    for _ in range(20):
        gen._last_prefill_s = 10.0
        _, generated = gen.next()
        if any(r.uid == short and r.finish_reason is not None for r in generated):
            break
    else:
        raise AssertionError("the short row never finished")
    gen.close()
    # Both rows step together; the burst stopped at the step that finished
    # the short row, so the long one got no more tokens in this call.
    assert sum(r.uid == long_ for r in generated) == sum(
        r.uid == short for r in generated
    )


def test_a_trickle_of_short_prompts_cannot_starve_a_long_prefill():
    """Width is the shortest row's segment, so one short newcomer per call
    would hold a long prompt at a few tokens per call for as long as the
    trickle lasts. After two such calls the next one admits nobody, and the
    long row gets a whole slice; the newcomers still wait at most one call
    more than usual."""
    model = tiny_hybrid()
    gen = BatchGenerator(
        model, max_tokens=2, stop_tokens=[], prefill_slice=64, prefill_batch_size=2
    )
    (long_,) = gen.insert([list(range(1, 601))])
    waited = {}
    calls = 0
    while True:
        # A fresh eight-token newcomer whenever the previous one is admitted.
        if not gen._unprocessed_sequences:
            (uid,) = gen.insert([[3, 5, 7, 9, 11, 13, 15, 17]])
            waited[uid] = calls
        _, generated = gen.next()
        calls += 1
        for r in generated:
            if r.uid in waited and isinstance(waited[r.uid], int):
                waited[r.uid] = ("done", calls - waited[r.uid])
        if any(r.uid == long_ for r in generated):
            break
        assert calls < 200, "the long prompt is starving"
    gen.close()
    # Without the guard this takes ~295 calls (measured); decode turns for
    # the newcomers alternate with the prefill calls, hence the margin.
    assert calls < 150 and gen.starved_calls >= 3
    # A newcomer waits at most one call more than without the guard (7).
    delays = [v[1] for v in waited.values() if isinstance(v, tuple)]
    assert delays and max(delays) <= 8, delays


def test_an_emptied_batch_keeps_no_current_tokens():
    """filter([]) leaves the batch empty; extend() must then start from the
    incoming batch's current tokens instead of concatenating onto the
    stale ones (a verify cycle takes the next tokens and never replaces
    the current ones, so an emptied batch would hold them for good)."""
    import mlx.core as mx

    from mlx_beam._vendor.mlx_lm.generate import GenerationBatch, StopSequences

    def batch(uids, tokens):
        # A model whose step samples a constant: the constructor runs one.
        def model(inputs, cache=None):
            return mx.zeros((inputs.shape[0], 1, 4))

        return GenerationBatch(
            model=model,
            uids=list(uids),
            inputs=mx.array(tokens),
            prompt_cache=[],
            tokens=[[] for _ in uids],
            samplers=[None] * len(uids),
            fallback_sampler=lambda lp: mx.argmax(lp, axis=-1),
            logits_processors=[[] for _ in uids],
            stop_sequences=[StopSequences([]) for _ in uids],
            max_tokens=[4] * len(uids),
        )

    a = batch([1, 2], [11, 12])
    assert a._current_tokens.tolist() == [11, 12]
    a.filter([])
    assert a._current_tokens is None and a._current_logprobs == []
    a.extend(batch([3], [13]))
    assert a.uids == [3] and a._current_tokens.tolist() == [13]


WINDOW = 8


def _window_keys(rows, positions):
    # A key names its row and position, so the mask can be read off the
    # keys it lets through; a right-padding token is -1, an unwritten column 0.
    vals = [
        [1000.0 * r + p + 1 if p is not None else -1.0 for p in ps]
        for r, ps in zip(rows, positions, strict=True)
    ]
    keys = mx.array(vals, dtype=mx.float32)[:, None, :, None]
    return keys, keys


def _feed(cache, rows, n, positions=None):
    at = cache.offset.tolist()
    if positions is None:
        positions = [[at[i] + j for j in range(n)] for i in range(len(rows))]
    return cache.update_and_fetch(*_window_keys(rows, positions))


@pytest.mark.parametrize(
    "source, ones, alone, before",
    [
        ("extend", 0, False, ([-3, 14], False, 17)),
        ("extend", 1, False, ([-13, 4], True, 1)),
        ("extend", 3, True, ([2], True, 3)),
        ("right_padding", 0, False, ([0, 19], False, 21)),
        ("right_padding", 1, False, ([-14, 5], True, 1)),
    ],
)
def test_rotating_mask_trims_what_the_update_trims(source, ones, alone, before):
    """A call of several tokens puts a rotated ring in temporal order and
    trims it; the mask is made before that and must trim the same. After
    `ones` one-token calls on a full window a row with left padding - it
    joined through `extend`, or it was right-padded - must still see its
    oldest key in the window. `alone` is the state of a verify of one row
    after decode steps; `before` is (left_padding, rotated, _idx) going in."""
    if source == "extend":
        cache = BatchRotatingKVCache(WINDOW, [0])
        _feed(cache, [0], 10)
        _feed(cache, [0], 10)
        other = BatchRotatingKVCache(WINDOW, [0])
        _feed(other, [1], 3)
        cache.extend(other)
    else:
        cache = BatchRotatingKVCache(WINDOW, [0, 0])
        _feed(cache, [0, 1], 1)
        cache.prepare(lengths=[20, 1], right_padding=[0, 19])
        _feed(cache, [0, 1], 20, [[1 + j for j in range(20)], [1] + [None] * 19])
        cache.finalize()
    rows = [0, 1]
    for _ in range(ones):
        _feed(cache, rows, 1)
    if alone:
        cache.filter([1])
        rows = [1]
    assert (cache.left_padding.tolist(), cache.rotated, cache._idx) == before

    at = cache.offset.tolist()
    mask = cache.make_mask(3, window_size=WINDOW)
    keys, _ = _feed(cache, rows, 3)
    L = keys.shape[2]
    assert mask.shape[-1] == L
    mask = mx.broadcast_to(mask, (len(rows), 1, 3, L))
    for r, row in enumerate(rows):
        line = keys[r, 0, :, 0].tolist()
        for q in range(3):
            p = at[r] + q
            seen = [k for k, m in zip(line, mask[r, 0, q].tolist(), strict=True) if m]
            want = [1000.0 * row + x + 1 for x in range(max(0, p - WINDOW + 1), p + 1)]
            assert seen == want, (row, p, seen, want)


@pytest.mark.parametrize("tail", [2, 3])
@pytest.mark.parametrize("name", ["gemma4_text", "gpt_oss", "muse_glimmer"])
def test_a_joined_row_matches_its_solo_run_after_a_one_token_call(name, tail):
    """A row that joins a longer batch sits behind left padding in the
    sliding layers. With `tail` 2 the other row's last segments make the
    joined row's first call one token wide on a full window, and its next
    call must still see the row's first token: its caches and the logprobs
    of its first decode step are those of a run alone. `tail` 3 has no
    one-token call and is the control."""
    from mlx_beam._vendor.mlx_lm.models import gemma4_text
    from tests.test_engine import GEMMA4_SHARED, _tiny
    from tests.test_model_classes import tiny

    if name == "gemma4_text":
        model = _tiny(gemma4_text, **GEMMA4_SHARED)
    else:
        model = tiny(name)
    a = [(3 * i + 5) % 64 for i in range(10 + tail)]
    b = [(7 * i + 11) % 64 for i in range(6)]

    def generator():
        return BatchGenerator(
            model,
            max_tokens=8,
            stop_tokens=[],
            prefill_batch_size=2,
            prefill_step_size=64,
            prefill_slice=64,
        )

    def caches_of(gen, uid):
        i = gen._prompt_batch.uids.index(uid)
        caches = [c.extract(i) for c in gen._prompt_batch.prompt_cache]
        mx.eval([(c.keys, c.values) for c in caches])
        return caches

    def first_logprobs(gen, uid):
        calls = 0
        while True:
            _, generated = gen.next()
            calls += 1
            assert calls < 50
            for r in generated:
                if r.uid == uid:
                    return r.logprobs

    gen = generator()
    (ua,) = gen.insert_segments([[a[:10], a[10:]]], max_tokens=[8])
    prompts, _ = gen.next()
    assert [(r.uid, r.progress) for r in prompts] == [(ua, (10, 10 + tail))]
    window = gen._prompt_batch.prompt_cache[0]
    assert isinstance(window, BatchRotatingKVCache)
    (ub,) = gen.insert([b], max_tokens=[2])
    progress = []
    calls = 0
    while not progress or progress[-1] != (5, 6):
        prompts, _ = gen.next()
        calls += 1
        assert calls < 50
        mine = [r.progress for r in prompts if r.uid == ub]
        if mine and not progress:
            # After the joined row's first call: the ring has wrapped with
            # `tail` 2, and the row sits behind the other's ten tokens.
            i = gen._prompt_batch.uids.index(ub)
            assert window.rotated is (tail == 2)
            assert tail == 3 or window._idx == 1
            assert window.left_padding.tolist()[i] == 7
        progress += mine
    assert progress == ([(1, 6), (5, 6)] if tail == 2 else [(2, 6), (5, 6)])
    batched = caches_of(gen, ub)
    batched_logprobs = first_logprobs(gen, ub)
    gen.close()

    solo = generator()
    (us,) = solo.insert([b], max_tokens=[2])
    prompts, _ = solo.next()
    assert [(r.uid, r.progress) for r in prompts] == [(us, (5, 6))]
    alone = caches_of(solo, us)
    alone_logprobs = first_logprobs(solo, us)
    solo.close()

    assert len(batched) == len(alone)
    for got, want in zip(batched, alone, strict=True):
        for what, x, y in (
            ("keys", got.keys, want.keys),
            ("values", got.values, want.values),
        ):
            assert (x.shape, x.dtype) == (y.shape, y.dtype), what
            assert mx.allclose(x, y, atol=1e-4), (what, mx.abs(x - y).max().item())
    x, y = batched_logprobs, alone_logprobs
    assert (x.shape, x.dtype) == (y.shape, y.dtype)
    assert mx.allclose(x, y, atol=1e-4), mx.abs(x - y).max().item()
