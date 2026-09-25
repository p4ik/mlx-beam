"""Samplers for the batch: one shared object per distinct setting, an own
keyed one for a seeded request.

The generator samples the whole batch in one call only when every row holds
the *same* sampler object; a fresh closure per request would send every step
down the per-row path. mlx-lm's samplers draw from the global random state,
so a seed can only be honoured by a sampler that carries its own key.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import mlx.core as mx

from mlx_beam._vendor.mlx_lm.sample_utils import (
    apply_min_p,
    apply_top_k,
    apply_top_p,
    greedy_sampler,
    make_sampler,
)
from mlx_beam.engine.request import SamplingParams

# Below this a temperature is greedy: 1/temp overflows float32 near 1e-38
# and the draw turns random, the opposite of what the client asked for.
GREEDY_BELOW = 1e-4


def is_greedy(p: SamplingParams) -> bool:
    return p.temperature < GREEDY_BELOW


def sampler_key(p: SamplingParams) -> tuple:
    """Everything make_sampler reads; two requests with equal keys share."""
    if is_greedy(p):
        return ("greedy",)
    return (
        p.temperature,
        p.top_p,
        p.top_k,
        p.min_p,
        p.min_tokens_to_keep,
        p.xtc_probability,
        p.xtc_threshold,
        tuple(p.xtc_special_tokens or ()),
    )


class SamplerPool:
    def __init__(self, capacity: int = 64):
        self._pool: OrderedDict[tuple, object] = OrderedDict()
        self._capacity = capacity

    def get(self, p: SamplingParams):
        if p.seed is not None and not is_greedy(p):
            return SeededSampler(p)
        key = sampler_key(p)
        sampler = self._pool.get(key)
        if sampler is None:
            sampler = make_sampler(
                temp=0.0 if is_greedy(p) else p.temperature,
                top_p=p.top_p,
                min_p=p.min_p,
                min_tokens_to_keep=p.min_tokens_to_keep,
                top_k=p.top_k,
                xtc_probability=p.xtc_probability,
                xtc_threshold=p.xtc_threshold,
                xtc_special_tokens=list(p.xtc_special_tokens or ()),
            )
            self._pool[key] = sampler
            if len(self._pool) > self._capacity:
                self._pool.popitem(last=False)
        else:
            self._pool.move_to_end(key)
        return sampler

    def describe(self) -> dict:
        return {"samplers": len(self._pool), "capacity": self._capacity}


def _xtc(logits, probability, threshold, special, key):
    probs = mx.softmax(logits, -1)
    mask = probs > mx.where(probs > threshold, probs, mx.inf).min(
        axis=-1, keepdims=True
    )
    if special:
        mask[..., special] = False
    return mx.where(
        mx.random.uniform(0, 1, key=key) > probability,
        logits,
        mx.where(mask, -mx.inf, logits),
    )


# Folds a seed and a position into one 64-bit key seed; the odd constant
# makes seed -> seed * PHI a bijection, so two seeds never share a stream.
_PHI = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1


def position_key(seed: int, position: int) -> mx.array:
    """The key for a row's draw at `position` (its n-th generated token).
    Keyed by position, not by call: a verify cycle that samples several
    positions at once and a plain step that samples one use the same key
    for the same position, so a seeded transcript is the same either way."""
    return mx.random.key((seed * _PHI + position) & _MASK)


class SeededSampler:
    """mlx-lm's chain with an explicit key per position, so the row's draws
    depend on neither the global state nor who else sits in the batch. Same
    seed, prompt and settings give the same tokens; the logits themselves
    may still differ by batch. The draw is Gumbel-max over the filtered
    logprobs - the noise is independent of the logits, which is what lets a
    speculative verify couple the target's draw to the proposer's."""

    def __init__(self, p: SamplingParams, seed: int | None = None):
        self.seed = p.seed if seed is None else seed
        # Draws made so far: the position the next draw belongs to.
        self.position = 0
        self._temp = p.temperature
        # Same order as make_sampler: top_p, min_p, xtc, top_k.
        self._steps = []
        if 0 < p.top_p < 1.0:
            self._steps.append(lambda x, key: apply_top_p(x, p.top_p))
        if p.min_p != 0.0:
            self._steps.append(
                lambda x, key: apply_min_p(x, p.min_p, p.min_tokens_to_keep)
            )
        if p.xtc_probability > 0.0:
            special = list(p.xtc_special_tokens or ())
            self._steps.append(
                lambda x, key: _xtc(x, p.xtc_probability, p.xtc_threshold, special, key)
            )
        if p.top_k > 0:
            self._steps.append(lambda x, key: apply_top_k(x, p.top_k))

    def draw(self, logprobs: mx.array, position: int) -> mx.array:
        """The token at `position` for these (1, V) logprobs: the filters,
        the temperature, then argmax over logprobs plus Gumbel noise under
        the position's key. Does not advance the position."""
        k_xtc, k_noise = mx.random.split(position_key(self.seed, position))
        for step in self._steps:
            logprobs = step(logprobs, k_xtc)
        scaled = logprobs * (1.0 / self._temp)
        return mx.argmax(scaled + mx.random.gumbel(scaled.shape, key=k_noise), axis=-1)

    def advance(self, n: int) -> None:
        """`n` positions were drawn elsewhere (a verify cycle)."""
        self.position += n

    def __call__(self, logprobs: mx.array) -> mx.array:
        token = self.draw(logprobs, self.position)
        self.position += 1
        return token


def top_logprobs(logprobs: mx.array, n: int) -> tuple[tuple[int, float], ...]:
    """The n most likely token ids with their log probabilities, best first.
    Tokens a processor ruled out (-inf) are no alternatives and are left
    out, so the list is JSON-clean and may be shorter than n."""
    n = min(n, logprobs.shape[-1])
    idx = mx.argpartition(-logprobs, kth=n - 1, axis=-1)[..., :n]
    vals = mx.take_along_axis(logprobs, idx, axis=-1)
    pairs = sorted(zip(idx.tolist(), vals.tolist(), strict=True), key=lambda x: -x[1])
    return tuple((int(i), float(v)) for i, v in pairs if math.isfinite(v))


__all__ = ["SamplerPool", "SeededSampler", "greedy_sampler", "top_logprobs"]
