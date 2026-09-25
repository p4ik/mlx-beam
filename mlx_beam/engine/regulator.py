"""The depth regulator: how many drafts a cycle verifies, chosen per cycle
from what was measured on this machine - the acceptance at each chain
position and the wall time of a cycle at each depth, against the wall time
of a plain step (depth 0, measured, not assumed).

rate(k) = expected committed tokens at depth k / cost(k), with
E[committed] = 1 + p1 + p1 p2 + ... (the chain holds while every earlier
position held). The depth with the best rate runs. A cycle that commits
fewer tokens than its depth's smoothed cost in plain steps is a loss; after
`PARK_AFTER` losses in a row the regulator parks for a cooldown of plain
steps that doubles each time it parks again without a winning cycle in
between (MTPLX's rule: 128 to 4096), then cycles again. A depth never
measured is priced from the plain step and the head's share of it, until
its first cycle replaces the guess.
"""

from __future__ import annotations

# Acceptance and cost smoothing: 1/16 keeps ~16 cycles of memory.
EMA = 1 / 16
# Losses in a row before the regulator parks (MTPLX, batch_policy.py).
PARK_AFTER = 16
# Cooldown in plain steps: first park, and the cap of the doubling.
COOLDOWN_MIN = 128
COOLDOWN_MAX = 4096
# Cost of one head call as a share of a plain step, for depths not yet
# measured: 0.056 on the 27B head (4.6 ms against an 82 ms step, 2026-09-19).
HEAD_SHARE = 0.056
# Acceptance assumed for a position never reached: what the 27B head held
# at positions 1..4 (0.87 / 0.79 / 0.74 / 0.66, 2026-09-19), then flat.
PRIOR_ACCEPTANCE = (0.87, 0.79, 0.74, 0.66)
# Cycles between probes of a neighbouring depth, so its cost stays current.
PROBE_EVERY = 32
# Cycles between one measured plain step, so the cost the cycles are held
# against is the plain step at the current context, not the warm-up's.
PLAIN_EVERY = 64


class Regulator:
    def __init__(self, cap: int, measure: bool = True):
        if cap < 1:
            raise ValueError("the draft cap must be at least 1")
        self.cap = cap
        # Whether observed wall times update the cost model; off, the costs
        # stay what a test set them to.
        self.measure = measure
        # A depth that runs every cycle, no regulation and no parking - for
        # tests that want a known shape of every cycle.
        self.fixed: int | None = None
        self.acceptance = [
            PRIOR_ACCEPTANCE[min(i, len(PRIOR_ACCEPTANCE) - 1)] for i in range(cap)
        ]
        # Positions reached so far, per position: an unreached position keeps
        # its prior, a reached one its EMA.
        self._seen = [0] * cap
        self.plain_cost: float | None = None
        self.cycle_cost: dict[int, float] = {}
        self.depth = cap
        self.losses = 0
        self.parked = False
        self.cooldown = 0
        self._next_cooldown = COOLDOWN_MIN
        self.parks = 0
        self.tokens_saved = 0
        self._cycles = 0

    # -- the model --------------------------------------------------------------

    def expected_committed(self, k: int) -> float:
        total, chain = 1.0, 1.0
        for i in range(k):
            chain *= self.acceptance[i]
            total += chain
        return total

    def cost(self, k: int) -> float | None:
        """Wall time of a cycle at depth k; None until a plain step was
        measured. Unmeasured depths: the plain step plus k head calls."""
        if k == 0:
            return self.plain_cost
        if k in self.cycle_cost:
            return self.cycle_cost[k]
        if self.plain_cost is None:
            return None
        return self.plain_cost * (1 + k * HEAD_SHARE)

    def rate(self, k: int) -> float | None:
        c = self.cost(k)
        return None if not c else self.expected_committed(k) / c

    def choose(self) -> int:
        """The depth for the next cycle: 0 while parked, else the depth with
        the best expected rate - with a probe of a neighbour now and then so
        its cost does not go stale, and a measured plain step now and then
        so the plain cost follows the context the cycles run in."""
        if self.fixed is not None:
            self.depth = self.fixed
            return self.fixed
        if self.parked:
            return 0
        if self.plain_cost is None:
            # Nothing measured yet: one plain step first, the price every
            # cycle is held against; the cap follows and measures itself.
            self.depth = 0
            return 0
        self._cycles += 1
        if self._cycles % PLAIN_EVERY == 0:
            self.depth = 0
            return 0
        rates = {k: self.rate(k) for k in range(1, self.cap + 1)}
        best = max(rates, key=lambda k: rates[k])
        if self._cycles % PROBE_EVERY == 0:
            neighbours = [k for k in (best - 1, best + 1) if 1 <= k <= self.cap]
            if neighbours:
                best = neighbours[(self._cycles // PROBE_EVERY) % len(neighbours)]
        self.depth = best
        return best

    def skipped(self) -> None:
        """The chosen cycle did not run (the row was not eligible): the
        choice does not count as a cycle for the probe and plain cadence."""
        if self._cycles > 0:
            self._cycles -= 1

    def reset(self) -> None:
        """Forget what the warm-up measured: its cycles ran cold (first
        shapes, kernels compiled on the way) over a prompt of a few tokens,
        and its acceptance is one greedy row's - none of it is this
        machine's steady state. The priors come back; the first real cycles
        measure."""
        self.acceptance = [
            PRIOR_ACCEPTANCE[min(i, len(PRIOR_ACCEPTANCE) - 1)] for i in range(self.cap)
        ]
        self._seen = [0] * self.cap
        self.plain_cost = None
        self.cycle_cost = {}
        self.depth = self.cap
        self.losses = 0
        self.parked = False
        self.cooldown = 0
        self._next_cooldown = COOLDOWN_MIN
        self.parks = 0
        self.tokens_saved = 0
        self._cycles = 0

    # -- observations -----------------------------------------------------------

    def observe_cycle(
        self, k: int, accepted: int, seconds: float, rejected: bool = True
    ) -> None:
        """A cycle over k drafts committed `accepted` of them in `seconds`;
        `rejected` says the position after the accepted ones was compared
        and refused (false when the row's limit ended the cycle first)."""
        for i in range(k):
            if i >= accepted and not rejected:
                break  # never compared: no verdict for this position
            held = 1.0 if i < accepted else 0.0
            if self._seen[i] == 0:
                self.acceptance[i] = held
            else:
                self.acceptance[i] += EMA * (held - self.acceptance[i])
            self._seen[i] += 1
            if i >= accepted:
                break  # the chain ended here: deeper positions were not tried
        self.tokens_saved += accepted
        if self.measure and k > 0:
            old = self.cycle_cost.get(k)
            self.cycle_cost[k] = seconds if old is None else old + EMA * (seconds - old)
        plain, cost = self.plain_cost, self.cost(k)
        if plain is None or not cost or self.fixed is not None:
            return
        # Against the depth's smoothed cost, not this one sample: a loss is
        # a cycle that committed fewer tokens than its price in plain steps.
        if (1 + accepted) / cost < 1 / plain:
            self.losses += 1
            if self.losses >= PARK_AFTER:
                self._park()
        else:
            self.losses = 0
            self._next_cooldown = COOLDOWN_MIN

    def observe_plain(self, seconds: float) -> None:
        """A plain step of one row took `seconds`; while parked it also
        counts the cooldown down."""
        if self.measure:
            old = self.plain_cost
            self.plain_cost = seconds if old is None else old + EMA * (seconds - old)
        if self.parked:
            self.cooldown -= 1
            if self.cooldown <= 0:
                self.parked = False
                self.losses = 0

    def _park(self) -> None:
        self.parked = True
        self.parks += 1
        self.cooldown = self._next_cooldown
        self._next_cooldown = min(self._next_cooldown * 2, COOLDOWN_MAX)
        self.losses = 0
        self.depth = 0

    def describe(self) -> dict:
        return {
            "cap": self.cap,
            "depth": self.depth,
            "acceptance_by_position": [round(p, 3) for p in self.acceptance],
            "positions_seen": list(self._seen),
            "cost_ms": {
                "plain": (
                    None if self.plain_cost is None else round(self.plain_cost * 1e3, 3)
                ),
                "cycle": {
                    k: round(c * 1e3, 3) for k, c in sorted(self.cycle_cost.items())
                },
            },
            "rate": {
                k: (None if r is None else round(r, 3))
                for k in range(self.cap + 1)
                for r in (self.rate(k),)
            },
            "tokens_saved": self.tokens_saved,
            "parked": self.parked,
            "cooldown_left": self.cooldown if self.parked else 0,
            "parks": self.parks,
            "losses": self.losses,
            "reason": (
                f"{PARK_AFTER} cycles in a row committed fewer tokens than their "
                "cost in plain steps"
                if self.parked
                else None
            ),
        }
