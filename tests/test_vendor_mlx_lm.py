"""The vendored mlx-lm: self-contained import, and the local changes listed in
VENDORED.md hold on a tiny hybrid model with random weights."""

import subprocess
import sys

import mlx.core as mx

from mlx_beam._vendor import mlx_lm as vendored
from mlx_beam._vendor.mlx_lm.generate import BatchGenerator
from mlx_beam._vendor.mlx_lm.models import qwen3_next
from mlx_beam._vendor.mlx_lm.models.cache import ArraysCache

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
