"""The prefill valve: the next prefill call's peak is estimated before it
is sent, because Metal's "Insufficient Memory" cannot be caught after.

Bytes per prompt token of a prefill call are learned from the allocator's
peak as calls run (an EWMA over the calls that moved the peak); the
ceiling is the smaller of the device's recommended working set and what
the process holds plus what the system has free - dynamic, since other
engines may share the memory. A call whose estimate does not fit is cut
to the width that does, on a grid of 64 tokens; below the smallest width
the admission stalls for this round instead of the engine dying.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable

import mlx.core as mx

# Smoothing of the bytes-per-token estimate: a few calls of memory.
EMA = 1 / 4
# Width grid and floor: a call narrower than this is not worth its overhead.
GRID = 64
MIN_WIDTH = 64
# How long a reading of the system's free memory is trusted.
FREE_MEMORY_TTL_S = 1.0


def _free_memory_darwin() -> int | None:
    try:
        text = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=2
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    page = 4096
    free = 0
    for line in text.splitlines():
        if line.startswith("Mach Virtual Memory Statistics"):
            if "page size of" in line:
                page = int(line.split("page size of")[1].split()[0])
        elif line.startswith(("Pages free", "Pages speculative", "Pages inactive")):
            free += int(line.split(":")[1].strip().rstrip("."))
    return free * page if free else None


def _free_memory_linux() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def free_memory() -> int | None:
    """Bytes the system has free for us now; None where it cannot be read."""
    if sys.platform == "darwin":
        return _free_memory_darwin()
    if sys.platform.startswith("linux"):
        return _free_memory_linux()
    return None


class PrefillValve:
    def __init__(
        self,
        active: Callable[[], int] = lambda: int(mx.get_active_memory()),
        peak: Callable[[], int] = lambda: int(mx.get_peak_memory()),
        free: Callable[[], int | None] = free_memory,
        recommended: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._active = active
        self._peak = peak
        self._free = free
        self._clock = clock
        # None: ask the device; 0: known to be unknown (nothing to hold
        # against on that side, the free reading alone decides).
        if recommended is None:
            recommended = mx.device_info().get("max_recommended_working_set_size")
        self.recommended = recommended
        self.bytes_per_token: float | None = None
        self.samples = 0
        self.shrunk_calls = 0
        self.stalled_calls = 0
        self.last_width: int | None = None
        self.last_ceiling: int | None = None
        self._free_cached: tuple[float, int | None] = (float("-inf"), None)
        self._before: tuple[int, int] | None = None

    # -- the ceiling ------------------------------------------------------------

    def _free_now(self) -> int | None:
        stamp, value = self._free_cached
        now = self._clock()
        if now - stamp > FREE_MEMORY_TTL_S:
            value = self._free()
            self._free_cached = (now, value)
        return value

    def ceiling(self) -> int | None:
        """min(recommended working set, held + free); None when neither is
        known - then the valve lets every call through."""
        held = self._active()
        free = self._free_now()
        candidates = [
            c for c in (self.recommended, None if free is None else held + free) if c
        ]
        self.last_ceiling = min(candidates) if candidates else None
        return self.last_ceiling

    # -- the decision -----------------------------------------------------------

    def width(self, wanted: int) -> int | None:
        """The width the next call may have: `wanted` when its estimated
        peak fits under the ceiling, a narrower width on the grid when it
        does not, None when not even the narrowest fits (stall)."""
        self._before = (self._active(), self._peak())
        if self.bytes_per_token is None or wanted <= 0:
            self.last_width = wanted
            return wanted
        ceiling = self.ceiling()
        if ceiling is None:
            self.last_width = wanted
            return wanted
        room = ceiling - self._before[0]
        fits = int(room // self.bytes_per_token)
        if fits >= wanted:
            self.last_width = wanted
            return wanted
        width = max(0, (fits // GRID) * GRID)
        if width < MIN_WIDTH:
            self.stalled_calls += 1
            self.last_width = None
            return None
        self.shrunk_calls += 1
        self.last_width = width
        return width

    def observe(self, tokens: int) -> None:
        """A call of `tokens` ran: when it moved the allocator's peak, the
        peak's rise over what was held before is this call's transient."""
        if self._before is None or tokens <= 0:
            return
        active_before, peak_before = self._before
        self._before = None
        peak_after = self._peak()
        if peak_after <= peak_before:
            return  # an earlier, bigger call set the peak; nothing new
        sample = (peak_after - active_before) / tokens
        if self.bytes_per_token is None:
            self.bytes_per_token = sample
        else:
            self.bytes_per_token += EMA * (sample - self.bytes_per_token)
        self.samples += 1

    def describe(self) -> dict:
        return {
            "bytes_per_token": (
                None if self.bytes_per_token is None else round(self.bytes_per_token)
            ),
            "samples": self.samples,
            "ceiling": self.last_ceiling,
            "recommended": self.recommended,
            "shrunk_calls": self.shrunk_calls,
            "stalled_calls": self.stalled_calls,
            "last_width": self.last_width,
        }
