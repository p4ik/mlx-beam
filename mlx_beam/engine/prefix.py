"""The prefix store: one entry per conversation, checkpoints at its boundaries.

A stored entry holds the caches of a finished request (KV for every token
plus the recurrent state at the end) and, for the recurrent and the
sliding-window layers of a model, snapshots taken at the segment boundaries
of its prompt (system end, each user end). A new request that shares a
prefix restores the entry, cuts it back to the last boundary at or before
the shared length - full-attention layers trim to any position, recurrent
layers only to a checkpoint, and a sliding-window layer that has rotated
past the boundary (its ring buffer overwrote the tokens a trim would need)
only to a checkpoint as well - and prefills the rest. Plain-attention
models need no checkpoints; every position is a boundary for them.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from mlx_beam._vendor.mlx_lm.generate import _cache_arrays
from mlx_beam._vendor.mlx_lm.models.cache import (
    ArraysCache,
    BatchRotatingKVCache,
    PromptTrie,
    RotatingKVCache,
)

# Bytes of a recurrent snapshot are counted like cache bytes.
# Eviction order: assistant entries (one per finished turn) go before the
# system entries (one per conversation, kept under their own cap).
_ORDER = ("assistant", "system")


def recurrent_layers(cache: list[Any]) -> list[int]:
    return [i for i, c in enumerate(cache) if isinstance(c, ArraysCache)]


def window_layers(cache: list[Any]) -> list[int]:
    return [
        i
        for i, c in enumerate(cache)
        if isinstance(c, (RotatingKVCache, BatchRotatingKVCache))
    ]


def snapshot_window(c: RotatingKVCache) -> list:
    """A sliding-window layer's state as a checkpoint holds it: the ring
    buffers (views; a later in-place write copies first) and the scalars
    that place them - the same tuple ``state`` reads and writes."""
    keys = None if c.keys is None else c.keys[:]
    values = None if c.values is None else c.values[:]
    return [keys, values, c.offset, c.keep, c.max_size, c._idx]


def restore_window(c: RotatingKVCache, snap: list) -> None:
    keys, values, offset, keep, max_size, idx = snap
    c.state = (
        None if keys is None else keys[:],
        None if values is None else values[:],
        offset,
        keep,
        max_size,
        idx,
    )


def checkpoint_layers(cache: list[Any]) -> list[int]:
    """Layers a boundary checkpoint must capture: recurrent state and
    sliding windows. Full-attention layers trim to any position."""
    return sorted(set(recurrent_layers(cache)) | set(window_layers(cache)))


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):  # noqa: B905 - lengths differ on purpose
        if x != y:
            break
        n += 1
    return n


def compact_kv(cache: list[Any]) -> None:
    """Cut every attention cache's buffers to its offset and own the bytes.
    trim() only moves the offset, and deepcopy shares buffers: an entry cut
    from a longer one would otherwise pin the whole longer buffer."""
    from mlx.utils import tree_map

    for c in cache:
        keys = getattr(c, "keys", None)
        if keys is None or isinstance(c, ArraysCache):
            continue
        n = c.offset
        c.keys = tree_map(lambda a, n=n: mx.contiguous(a[..., :n, :]), keys)
        c.values = tree_map(lambda a, n=n: mx.contiguous(a[..., :n, :]), c.values)


# The key of the draft head's history in a checkpoint, beside the layer
# indices: (keys, values, tail hidden) of the head's cache at that position,
# taken back by the proposer when a request resumes exactly there.
HEAD = "head"


def _snapshot_bytes(snap: dict) -> int:
    # A window snapshot carries scalars next to its arrays; only arrays weigh.
    return sum(getattr(a, "nbytes", 0) for arrays in snap.values() for a in arrays)


@dataclass
class Entry:
    cache: list[Any]
    length: int
    # position -> layer index -> state at that position, and the draft
    # head's history under HEAD (empty for plain models without a head)
    checkpoints: dict[int, dict]
    cache_type: str
    nbytes: int = 0

    def __post_init__(self):
        self.nbytes = sum(c.nbytes for c in self.cache) + sum(
            _snapshot_bytes(s) for s in self.checkpoints.values()
        )

    def add_checkpoint(self, position: int, snap: dict[int, list]) -> None:
        self.checkpoints[position] = snap
        self.nbytes += _snapshot_bytes(snap)


@dataclass
class Hit:
    cache: list[Any]
    # tokens the restored cache already covers; the rest needs a prefill
    covered: int
    kind: str  # "exact", "longer", "shorter"
    # the entry's checkpoints at or before ``covered``, still valid - and
    # for a recurrent model always one at ``covered`` itself
    checkpoints: dict[int, dict] = field(default_factory=dict)


@dataclass
class StoreStats:
    lookups: int = 0
    hits: int = 0
    tokens_found: int = 0
    tokens_restored: int = 0


def cut_back(cache: list[Any], entry_len: int, target: int, checkpoints) -> int:
    """Cut a restored cache to the last usable boundary at or before target.

    Returns the position the cache now stands at, or -1 when no boundary
    works (a recurrent layer without a checkpoint at or before target).
    """
    if target >= entry_len:
        return entry_len
    rec = recurrent_layers(cache)
    # A rotated window cannot trim (the ring overwrote what a trim would
    # need); until it rotates, it trims like a full-attention layer.
    stuck = [i for i in window_layers(cache) if not cache[i].is_trimmable()]
    if rec or stuck:
        usable = [
            p
            for p, snap in checkpoints.items()
            if p <= target and all(i in snap for i in (*rec, *stuck))
        ]
        if not usable:
            return -1
        pos = max(usable)
    else:
        pos = target
    for i, c in enumerate(cache):
        if i in rec:
            c.cache = [
                None if a is None else copy.deepcopy(a) for a in checkpoints[pos][i]
            ]
        elif i in stuck:
            restore_window(c, checkpoints[pos][i])
        else:
            if not c.is_trimmable():
                return -1
            c.trim(entry_len - pos)
    return pos


class PrefixStore:
    def __init__(self, max_entries: int = 16, max_bytes: int = 1 << 63):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._trie = PromptTrie()
        self._lru: dict[str, deque] = {k: deque() for k in _ORDER}
        self._nbytes = 0
        self.stats = StoreStats()

    def __len__(self) -> int:
        return sum(len(q) for q in self._lru.values())

    @property
    def nbytes(self) -> int:
        return self._nbytes

    # -- lookup -------------------------------------------------------------

    def fetch(self, model: Any, tokens: list[int]) -> Hit | None:
        """A deep copy of the best entry, cut back so at least one token of
        ``tokens`` remains to prefill; None on a miss."""
        self.stats.lookups += 1
        want = len(tokens) - 1  # the generator needs one token to start
        res = self._trie.search(model, tokens)
        candidates: list[list[int]] = []
        for key in (res.exact, res.longer, res.shorter):
            if key is not None and key not in candidates:
                candidates.append(key)
        # Longest shared prefix first; a candidate that cannot be cut back to
        # a usable boundary yields to the next.
        scored = sorted(
            ((min(len(k), _common_prefix(k, tokens)), k) for k in candidates),
            key=lambda t: -t[0],
        )
        for shared, key in scored:
            entry = self._trie.get(model, key)
            target = min(shared, want)
            cache = copy.deepcopy(entry.cache)
            pos = cut_back(cache, entry.length, target, entry.checkpoints)
            if pos <= 0:
                continue
            self._touch(model, key, entry.cache_type)
            self.stats.hits += 1
            self.stats.tokens_found += shared
            self.stats.tokens_restored += pos
            carried = {p: s for p, s in entry.checkpoints.items() if p <= pos}
            stateful = checkpoint_layers(cache)
            if stateful and pos == entry.length:
                # Restored whole: the entry's end state is the restore point
                # this request stands on. Carried along (shared, not copied),
                # it survives the entry being replaced or evicted before the
                # request finishes. A head sidecar already there stays.
                end = dict(carried.get(pos, {}))
                for i in stateful:
                    if i not in end:
                        end[i] = (
                            entry.cache[i].cache
                            if isinstance(entry.cache[i], ArraysCache)
                            else snapshot_window(entry.cache[i])
                        )
                carried[pos] = end
            if len(key) == len(tokens):
                kind = "exact"
            elif len(key) < len(tokens):
                kind = "shorter"
            else:
                kind = "longer"
            return Hit(cache, pos, kind, carried)
        return None

    # -- insert and eviction ------------------------------------------------

    def insert(
        self,
        model: Any,
        tokens: list[int],
        cache: list[Any],
        checkpoints: dict[int, dict] | None = None,
        cache_type: str = "assistant",
    ) -> None:
        # extract() hands out lazy slices of the batch buffers; evaluated
        # here they own their bytes and the batch can be released.
        mx.eval(*_cache_arrays(cache))
        entry = Entry(cache, len(tokens), dict(checkpoints or {}), cache_type)
        prev = self._trie.add(model, tokens, entry)
        if prev is not None:
            self._forget(model, tokens, prev)
        self._nbytes += entry.nbytes
        self._lru[cache_type].append((model, tokens))
        # Prefixes this entry can be cut back to are redundant copies - except
        # a system entry, which exists to outlive the conversations above it.
        stateful = checkpoint_layers(cache)
        cuttable = all(
            c.is_trimmable() for i, c in enumerate(cache) if i not in stateful
        )
        for prefix_len, old in self._trie.pop_prefixes(model, tokens):
            if old.cache_type == "system" or not cuttable:
                self._trie.add(model, tokens[:prefix_len], old)  # keep it
                continue
            if stateful:
                # Whatever the old entry could restore, the new one can too:
                # its checkpoints and its end state (the recurrent state at
                # prefix_len). Moved, not copied - the old KV cache is what
                # goes away. One entry per conversation, not one per turn.
                inherited = dict(old.checkpoints)
                end = dict(inherited.get(prefix_len, {}))
                for i in stateful:
                    if i not in end:
                        end[i] = (
                            old.cache[i].cache
                            if isinstance(old.cache[i], ArraysCache)
                            else snapshot_window(old.cache[i])
                        )
                inherited[prefix_len] = end
                for position, snap in inherited.items():
                    if position <= prefix_len and position not in entry.checkpoints:
                        entry.add_checkpoint(position, snap)
                        self._nbytes += _snapshot_bytes(snap)
            self._nbytes -= old.nbytes
            self._remove_lru(model, tokens[:prefix_len])
        self._evict()

    def has(self, model: Any, tokens: list[int]) -> bool:
        try:
            return self._trie.get(model, tokens) is not None
        except KeyError:
            return False

    def insert_prefix(
        self,
        model: Any,
        tokens: list[int],
        cache: list[Any],
        checkpoints: dict[int, dict],
        upto: int,
        cache_type: str = "system",
    ) -> bool:
        """Store the first ``upto`` tokens of a finished cache as an entry of
        its own; a recurrent layer needs a checkpoint at exactly ``upto``.
        Returns False when the prefix is already stored or cannot be cut."""
        if upto <= 0 or upto > len(tokens) or self.has(model, tokens[:upto]):
            return False
        copy_ = copy.deepcopy(cache)
        if cut_back(copy_, len(tokens), upto, checkpoints) != upto:
            return False
        compact_kv(copy_)
        kept = {p: s for p, s in checkpoints.items() if p <= upto}
        self.insert(model, tokens[:upto], copy_, kept, cache_type)
        return True

    def _touch(self, model, key, cache_type) -> None:
        q = self._lru[cache_type]
        try:
            q.remove((model, key))
        except ValueError:
            return
        q.append((model, key))

    def _forget(self, model, key, entry) -> None:
        self._nbytes -= entry.nbytes
        self._remove_lru(model, key)

    def _remove_lru(self, model, key) -> None:
        for q in self._lru.values():
            try:
                q.remove((model, key))
                return
            except ValueError:
                pass

    @property
    def system_cap(self) -> int:
        """System entries outlive conversations, but only so many: past a
        quarter of the store they would crowd every conversation out."""
        return max(1, self.max_entries // 4)

    def _pop_victim(self):
        # Assistant entries go first, then system; within a type the least
        # recently used (surplus system entries went in _evict already).
        for kind in _ORDER:
            if self._lru[kind]:
                return self._lru[kind].popleft()
        return None

    def _evict(self) -> None:
        while len(self._lru["system"]) > self.system_cap:
            model, key = self._lru["system"].popleft()
            self._nbytes -= self._trie.pop(model, key).nbytes
        while len(self) > self.max_entries or self._nbytes > self.max_bytes:
            victim = self._pop_victim()
            if victim is None:
                return
            model, key = victim
            self._nbytes -= self._trie.pop(model, key).nbytes

    def describe(self) -> dict:
        s = self.stats
        return {
            "entries": len(self),
            "bytes": self._nbytes,
            "lookups": s.lookups,
            "hits": s.hits,
            "tokens_found": s.tokens_found,
            "tokens_restored": s.tokens_restored,
            "by_type": {k: len(q) for k, q in self._lru.items()},
        }
