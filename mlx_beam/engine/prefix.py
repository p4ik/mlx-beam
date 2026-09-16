"""The prefix store: one entry per conversation, checkpoints at its boundaries.

A stored entry holds the caches of a finished request (KV for every token
plus the recurrent state at the end) and, for the recurrent layers of a
hybrid, snapshots taken at the segment boundaries of its prompt (system
end, each user end). A new request that shares a prefix restores the entry,
cuts it back to the last boundary at or before the shared length - KV
layers trim to any position, recurrent layers only to a checkpoint - and
prefills the rest. Plain-attention models need no checkpoints; every
position is a boundary for them.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_beam._vendor.mlx_lm.models.cache import ArraysCache, PromptTrie

# Bytes of a recurrent snapshot are counted like cache bytes.
_ORDER = ("assistant", "user", "system")


def recurrent_layers(cache: list[Any]) -> list[int]:
    return [i for i, c in enumerate(cache) if isinstance(c, ArraysCache)]


def snapshot_recurrent(cache: list[Any]) -> dict[int, list]:
    """The recurrent layers' state, copied, keyed by layer index."""
    return {
        i: [None if a is None else copy.deepcopy(a) for a in cache[i].cache]
        for i in recurrent_layers(cache)
    }


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):  # noqa: B905 - lengths differ on purpose
        if x != y:
            break
        n += 1
    return n


def cache_arrays(cache: list[Any]) -> list:
    """Every array a cache list holds, read as storage."""
    out = []
    for c in cache:
        for leaf in getattr(c, "caches", None) or (c,):
            if leaf is None:
                continue
            for v in vars(leaf).values():
                if v is None or isinstance(v, (int, float, str, bool)):
                    continue
                for _, a in tree_flatten(v):
                    if isinstance(a, mx.array):
                        out.append(a)
    return out


def _snapshot_bytes(snap: dict[int, list]) -> int:
    return sum(a.nbytes for arrays in snap.values() for a in arrays if a is not None)


@dataclass
class Entry:
    cache: list[Any]
    length: int
    # position -> recurrent state at that position (empty for plain models)
    checkpoints: dict[int, dict[int, list]]
    cache_type: str
    nbytes: int = 0

    def __post_init__(self):
        self.nbytes = sum(c.nbytes for c in self.cache) + sum(
            _snapshot_bytes(s) for s in self.checkpoints.values()
        )


@dataclass
class Hit:
    cache: list[Any]
    # tokens the restored cache already covers; the rest needs a prefill
    covered: int
    # what the store found before cutting back (for the miss/hit accounting)
    found: int
    kind: str  # "exact", "longer", "shorter"
    # the entry's checkpoints at or before ``covered``, still valid
    checkpoints: dict[int, dict[int, list]] = field(default_factory=dict)


@dataclass
class StoreStats:
    entries: int = 0
    nbytes: int = 0
    lookups: int = 0
    hits: int = 0
    tokens_found: int = 0
    tokens_restored: int = 0
    by_type: dict = field(default_factory=dict)


def cut_back(cache: list[Any], entry_len: int, target: int, checkpoints) -> int:
    """Cut a restored cache to the last usable boundary at or before target.

    Returns the position the cache now stands at, or -1 when no boundary
    works (a recurrent layer without a checkpoint at or before target).
    """
    if target >= entry_len:
        return entry_len
    rec = recurrent_layers(cache)
    if rec:
        usable = [p for p in checkpoints if p <= target]
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
            if len(key) == len(tokens):
                kind = "exact"
            elif len(key) < len(tokens):
                kind = "shorter"
            else:
                kind = "longer"
            return Hit(cache, pos, shared, kind, carried)
        return None

    # -- insert and eviction ------------------------------------------------

    def insert(
        self,
        model: Any,
        tokens: list[int],
        cache: list[Any],
        checkpoints: dict[int, dict[int, list]] | None = None,
        cache_type: str = "assistant",
    ) -> None:
        # extract() hands out lazy slices of the batch buffers; evaluated
        # here they own their bytes and the batch can be released.
        mx.eval(*cache_arrays(cache))
        entry = Entry(cache, len(tokens), dict(checkpoints or {}), cache_type)
        prev = self._trie.add(model, tokens, entry)
        if prev is not None:
            self._forget(model, tokens, prev)
        self._nbytes += entry.nbytes
        self._lru[cache_type].append((model, tokens))
        # Prefixes this entry can be cut back to are redundant copies - except
        # a system entry, which exists to outlive the conversations above it.
        rec = recurrent_layers(cache)
        cuttable = all(c.is_trimmable() for i, c in enumerate(cache) if i not in rec)
        for prefix_len, old in self._trie.pop_prefixes(model, tokens):
            if (
                old.cache_type == "system"
                or not cuttable
                or (rec and prefix_len not in entry.checkpoints)
            ):
                self._trie.add(model, tokens[:prefix_len], old)  # keep it
                continue
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
        checkpoints: dict[int, dict[int, list]],
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

    def _pop_victim(self):
        # Assistant entries go first, then user, then system; within a type
        # the least recently used.
        for kind in _ORDER:
            if self._lru[kind]:
                return self._lru[kind].popleft()
        return None

    def _evict(self) -> None:
        while len(self) > self.max_entries or self._nbytes > self.max_bytes:
            victim = self._pop_victim()
            if victim is None:
                return
            model, key = victim
            self._nbytes -= self._trie.pop(model, key).nbytes

    def trim_to_bytes(self, n_bytes: int) -> None:
        while self._nbytes > max(0, n_bytes) and len(self):
            model, key = self._pop_victim()
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
