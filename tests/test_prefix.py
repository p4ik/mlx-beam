"""Prefix store with recurrent-state checkpoints, on tiny models.

The acceptance: a hybrid that resumes from a checkpoint decodes the same
tokens as a cold run, and the store never hands out a state it cannot vouch
for."""

import mlx.core as mx

from mlx_beam._vendor.mlx_lm.models.cache import KVCache
from mlx_beam.api import chat
from mlx_beam.engine import Engine, GenerationRequest
from mlx_beam.engine.prefix import PrefixStore, cut_back
from tests.stub_tokenizer import StubTokenizer
from tests.test_engine import tiny_hybrid
from tests.test_vendor_optiq_kv import tiny_llama


def run(engine, tokens, max_tokens=4, boundaries=(), system_end=None):
    stream = engine.submit(
        GenerationRequest(
            list(tokens),
            max_tokens=max_tokens,
            boundaries=boundaries,
            system_end=system_end,
        )
    )
    out = [e.token for e in stream]
    return out, stream.prompt_cached


def test_plain_model_cuts_anywhere():
    store = PrefixStore()
    caches = [KVCache()]
    k = mx.zeros((1, 1, 10, 8))
    caches[0].update_and_fetch(k, k)
    store.insert("m", list(range(10)), caches)
    hit = store.fetch("m", list(range(6)) + [99])
    assert hit is not None and hit.kind == "longer" and hit.covered == 6
    assert hit.cache[0].offset == 6
    hit = store.fetch("m", list(range(10)))  # exact: one token stays
    assert hit.covered == 9
    hit = store.fetch("m", list(range(12)))  # shorter: whole entry
    assert hit.kind == "shorter" and hit.covered == 10
    assert store.fetch("m", [42, 43]) is None
    assert store.describe()["hits"] == 3 and store.describe()["lookups"] == 4


def test_hybrid_needs_a_checkpoint():
    model = tiny_hybrid()
    with Engine(model) as engine:
        prompt = list(range(1, 21))
        cold, cached = run(engine, prompt)
        assert cached == 0
        # Stored with one checkpoint, before the prompt's last token: a
        # strict prefix cannot be served, a longer prompt continues from
        # there, and the same prompt again has one token left to prefill.
        assert engine.prefix_store.describe()["entries"] == 1
        _, cached = run(engine, prompt[:10] + [50, 51])
        assert cached == 0
        out, cached = run(engine, prompt + [60, 61])
        assert cached == 19
        again, cached = run(engine, prompt)
        assert cached == 19 and again == cold


def test_hybrid_resumes_from_boundary_exactly():
    model = tiny_hybrid()
    prompt = list(range(1, 25))
    # Cold references for the two prompts that share the first 12 tokens.
    with Engine(model) as engine:
        ref_a, _ = run(engine, prompt, boundaries=[12])
    with Engine(model) as engine:
        ref_b, _ = run(engine, prompt[:12] + [40, 41, 42], boundaries=[12])
    with Engine(model) as engine:
        out_a, _ = run(engine, prompt, boundaries=[12])
        assert out_a == ref_a
        store = engine.prefix_store
        entry = store._trie.get(engine.model_key, prompt + out_a)
        assert sorted(entry.checkpoints) == [12, 23]
        out_b, cached = run(engine, prompt[:12] + [40, 41, 42], boundaries=[12])
        assert cached == 12 and out_b == ref_b
        d = store.describe()
        assert d["tokens_restored"] == 12 and d["tokens_found"] >= 12


def test_cut_back_refuses_without_checkpoint():
    from mlx_beam._vendor.mlx_lm.models.cache import ArraysCache

    rec = ArraysCache(2)
    rec.cache = [mx.zeros((1, 4)), mx.ones((1, 4))]
    kv = KVCache()
    k = mx.zeros((1, 1, 10, 8))
    kv.update_and_fetch(k, k)
    assert cut_back([kv, rec], 10, 5, {}) == -1
    snap = {1: [mx.full((1, 4), 7.0), None]}
    assert cut_back([kv, rec], 10, 5, {3: snap}) == 3
    assert kv.offset == 3 and rec.cache[0].tolist() == [[7.0] * 4]


def test_llama_conversation_collapses_to_one_entry():
    model = tiny_llama()
    with Engine(model, prompt_cache_size=4) as engine:
        turn1 = [2, 6, 1, 2, 7]
        out1, _ = run(engine, turn1, 3)
        turn2 = turn1 + out1 + [6, 3, 7]
        out2, cached = run(engine, turn2, 3)
        assert cached == len(turn1) + len(out1)
        assert engine.prefix_store.describe()["entries"] == 1


def test_chat_boundaries_from_the_template():
    tok = StubTokenizer()
    req = chat.parse_chat_request(
        {
            "messages": [
                {"role": "system", "content": "w1 w2"},
                {"role": "user", "content": "w3"},
                {"role": "assistant", "content": "w4"},
                {"role": "user", "content": "w5"},
            ]
        },
        "m",
    )
    prompt = chat.build_prompt(tok, req)
    assert prompt == [2, 5, 1, 2, 6, 3, 7, 4, 6, 5, 7]
    bounds = chat.prompt_boundaries(tok, req, prompt)
    # The system block ends after the first user header (the header is the
    # same for any next user message); the first user turn ends after the
    # assistant header that follows it.
    assert bounds == [5, 7]


def test_system_entry_outlives_the_conversations():
    model = tiny_llama()
    system = [2, 6, 1, 2]
    with Engine(model, prompt_cache_size=2) as engine:
        for turn in ([7, 3], [5, 4], [3, 3, 6]):
            run(engine, system + turn, 2)
        d = engine.prefix_store.describe()
        # Two conversation entries fit; the third pushed the oldest out, the
        # system entry stayed untouched.
        assert d["by_type"] == {"assistant": 2, "user": 0, "system": 0}
    with Engine(model, prompt_cache_size=2) as engine:
        for turn in ([7, 3], [5, 4], [3, 3, 6]):
            run(engine, system + turn, 2, system_end=len(system))
        d = engine.prefix_store.describe()
        assert d["by_type"]["system"] == 1 and d["by_type"]["assistant"] == 1
        # A fourth conversation still starts from the system block.
        _, cached = run(engine, system + [9, 9], 2, system_end=len(system))
        assert cached == len(system)


def test_hybrid_system_entry_needs_its_checkpoint():
    model = tiny_hybrid()
    system = list(range(1, 9))
    with Engine(model, prompt_cache_size=1) as engine:
        run(engine, system + [20, 21], 2, boundaries=[8], system_end=8)
        assert engine.prefix_store.describe()["by_type"]["system"] == 1
        run(engine, system + [30, 31, 32], 2, boundaries=[8], system_end=8)
        _, cached = run(engine, system + [40], 2, boundaries=[8], system_end=8)
        assert cached == 8


def test_stride_checkpoints_inside_a_long_prefill():
    model = tiny_hybrid()
    prompt = list(range(1, 41))
    with Engine(model, prefill_slice=4) as cold:
        ref, _ = run(cold, prompt[:20] + [50, 51], 2)
    with Engine(
        model, prefill_slice=4, checkpoint_stride=8, checkpoint_stride_max=4
    ) as engine:
        run(engine, prompt, 2)
        entry = engine.prefix_store._trie.get(
            engine.model_key, prompt + entry_tail(engine, prompt)
        )
        assert {8, 16, 24, 32} <= set(entry.checkpoints)
        out, cached = run(engine, prompt[:20] + [50, 51], 2)
        assert cached == 16 and out == ref


def test_stride_checkpoints_are_thinned_to_the_cap():
    model = tiny_hybrid()
    prompt = list(range(1, 41))
    with Engine(
        model, prefill_slice=4, checkpoint_stride=8, checkpoint_stride_max=2
    ) as engine:
        run(engine, prompt, 2)
        entry = engine.prefix_store._trie.get(
            engine.model_key, prompt + entry_tail(engine, prompt)
        )
        # the prompt's own checkpoint (39) is a boundary, not a stride point
        stride_points = [p for p in entry.checkpoints if p < 39]
        assert len(stride_points) == 2 and max(stride_points) == 32


def test_cancel_during_prefill_keeps_the_prefilled_part():
    from mlx_beam.engine.request import PromptProgress

    model = tiny_llama()
    prompt = [2, 6, 1, 2, 7, 3, 5, 4, 3, 3, 6, 7, 1, 2, 6, 5]
    with Engine(model, prefill_slice=4) as engine:
        stream = engine.submit(GenerationRequest(prompt, max_tokens=4))
        while True:
            item = stream._queue.get(timeout=30)
            if isinstance(item, PromptProgress) and item.processed >= 4:
                break
        stream.cancel()
        for _ in stream:
            pass
        # Whatever the worker had computed is in the store now.
        for _ in range(50):
            if engine.prefix_store.describe()["entries"]:
                break
            import time

            time.sleep(0.05)
        _, cached = run(engine, prompt, 2)
        assert cached >= 4


def entry_tail(engine, prompt):
    """The generated tokens of the last run, from the store's own keys."""
    store = engine.prefix_store
    for _model, key in list(store._lru["assistant"]):
        if list(key[: len(prompt)]) == list(prompt):
            return list(key[len(prompt) :])
    raise AssertionError("no entry for the prompt")


def test_system_entries_are_capped_so_conversations_keep_their_place():
    """Many distinct system prompts must not turn the whole store into
    system entries: past a quarter of it the oldest system entry goes, and
    a conversation's second turn still finds its first."""
    model = tiny_llama()
    with Engine(model, prompt_cache_size=4) as engine:
        assert engine.prefix_store.system_cap == 1
        for n in range(5):
            system = [2, 6, 1, n + 10]
            run(engine, system + [7, 3], 2, system_end=len(system))
        d = engine.prefix_store.describe()["by_type"]
        assert d["system"] == 1 and d["assistant"] >= 2
        system = [2, 6, 1, 14]
        _, cached = run(engine, system + [7, 3] + [5, 4], 2, system_end=len(system))
        # The last conversation is still there: more than its system block hit.
        assert cached > len(system)


def test_a_prefix_entry_owns_only_its_own_tokens():
    """A system entry cut from a long conversation must not keep the whole
    conversation's KV buffers alive."""
    store = PrefixStore()
    caches = [KVCache()]
    k = mx.zeros((1, 1, 600, 8))
    caches[0].update_and_fetch(k, k)
    mx.eval(caches[0].keys)
    tokens = list(range(600))
    assert store.insert_prefix("m", tokens, caches, {}, 4)
    entry = store._trie.get("m", tokens[:4])
    assert entry.cache[0].offset == 4 and entry.cache[0].keys.shape[2] == 4
    assert entry.nbytes == 2 * 4 * 8 * 4  # keys + values, 4 tokens, 8 dims, fp32
    hit = store.fetch("m", tokens[:4] + [99])
    assert hit is not None and hit.covered == 4


def test_a_second_turn_replaces_the_first_entry():
    # A hybrid cannot trim, so the first turn's entry could only be dropped
    # if the second turn can be cut back to it: the snapshot taken where the
    # restored entry ends makes that possible. One entry per conversation.
    model = tiny_hybrid()
    prompt = list(range(1, 21))
    with Engine(model) as engine:
        out1, _ = run(engine, prompt)
        turn2 = prompt + out1 + [50, 51, 52]
        bounds = [len(prompt), len(prompt) + len(out1)]
        with Engine(model) as cold:
            ref2, _ = run(cold, turn2, boundaries=bounds)
        out2, cached = run(engine, turn2, boundaries=bounds)
        assert cached == len(prompt) + len(out1) and out2 == ref2
        d = engine.prefix_store.describe()
        assert d["entries"] == 1 and d["by_type"]["assistant"] == 1
        entry = engine.prefix_store._trie.get(engine.model_key, turn2 + out2)
        assert len(prompt) + len(out1) in entry.checkpoints
        # and the conversation's first turn is still served from it
        _, cached = run(engine, prompt + out1 + [60])
        assert cached == len(prompt) + len(out1)


def test_two_branches_from_one_entry_replace_it_and_keep_their_own_ends():
    model = tiny_hybrid()
    prompt = list(range(1, 21))
    with Engine(model) as engine:
        out, _ = run(engine, prompt)
        prefix = prompt + out
        a, b = prefix + [50, 51, 52], prefix + [55, 56, 57]
        with Engine(model) as cold:
            ref_a, _ = run(cold, a)
        with Engine(model) as cold:
            ref_b, _ = run(cold, b)
        out_a, cached_a = run(engine, a)
        out_b, cached_b = run(engine, b)
        assert (out_a, out_b) == (ref_a, ref_b)
        assert cached_a == cached_b == len(prefix)
        # the first turn's entry is gone, each branch is its own entry
        assert engine.prefix_store.describe()["entries"] == 2
        assert not engine.prefix_store.has(engine.model_key, prefix)
        again, cached = run(engine, a)
        assert again == ref_a and cached == len(a) - 1


def test_a_divergent_continuation_keeps_the_entry_it_branched_from():
    # Sharing a checkpoint with an entry is not extending it: nothing is
    # inherited, the entry stays, the new one has only its own checkpoints.
    model = tiny_hybrid()
    prompt = list(range(1, 25))
    with Engine(model) as engine:
        run(engine, prompt, boundaries=[12])
        div = prompt[:12] + [40] * 8
        out, cached = run(engine, div, boundaries=[12])
        assert cached == 12
        assert engine.prefix_store.describe()["entries"] == 2
        entry = engine.prefix_store._trie.get(engine.model_key, div + out)
        assert sorted(entry.checkpoints) == [12, len(div) - 1]
