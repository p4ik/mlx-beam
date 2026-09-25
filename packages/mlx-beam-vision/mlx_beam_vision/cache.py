"""Encoder outputs by image digest: the same image in the next turn costs
no tower pass. Bounded in bytes, least recently used out first; the
family's name is part of the key. Request threads share it, so every
access holds its lock."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import mlx.core as mx


class FeatureCache:
    def __init__(self, max_bytes: int = 512 << 20):
        self.max_bytes = max_bytes
        self._entries: OrderedDict[tuple, tuple[Any, int]] = OrderedDict()
        self._lock = threading.Lock()
        self.nbytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple):
        with self._lock:
            hit = self._entries.get(key)
            if hit is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return hit[0]

    def put(self, key: tuple, value: Any, arrays: list[mx.array]) -> None:
        size = sum(int(a.nbytes) for a in arrays)
        if size > self.max_bytes:
            return  # one image bigger than the whole cache: not worth holding
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self.nbytes -= old[1]
            self._entries[key] = (value, size)
            self.nbytes += size
            while self.nbytes > self.max_bytes and self._entries:
                _, (_, gone) = self._entries.popitem(last=False)
                self.nbytes -= gone

    def describe(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self.nbytes,
                "max_bytes": self.max_bytes,
                "hits": self.hits,
                "misses": self.misses,
            }
