"""`/metrics` in Prometheus' text exposition format, from the same counters
`/health` reports: nothing is measured twice. Counters never fall (the
generator's, the store's, the repair ladder's, the speculator's); gauges
are the moment's values (memory, rows in flight, the queue). Our names
carry the `mlx_beam_` prefix; the four names dashboards built for vLLM
read most are given as well, under `vllm:`.
"""

from __future__ import annotations

from typing import Any


def _line(name: str, value: Any, labels: dict | None = None) -> str:
    if value is None or isinstance(value, bool):
        value = int(bool(value)) if isinstance(value, bool) else 0
    tags = ""
    if labels:
        tags = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
    return f"{name}{tags} {value}"


def render(health: dict, model: str) -> str:
    """The exposition text for one health snapshot."""
    labels = {"model": model}
    out: list[str] = []

    def metric(name, kind, help_text, value, extra=None):
        out.append(f"# HELP {name} {help_text}")
        out.append(f"# TYPE {name} {kind}")
        out.append(_line(name, value, {**labels, **(extra or {})}))

    metric(
        "mlx_beam_alive",
        "gauge",
        "1 while the engine worker runs.",
        health.get("alive"),
    )
    metric(
        "mlx_beam_uptime_seconds",
        "gauge",
        "Seconds since the engine started.",
        health.get("uptime_s"),
    )
    metric(
        "mlx_beam_requests_in_flight",
        "gauge",
        "Rows being prefilled or decoded.",
        health.get("in_flight"),
    )
    metric(
        "mlx_beam_requests_queued",
        "gauge",
        "Requests waiting for admission.",
        health.get("queued"),
    )
    metric(
        "mlx_beam_requests_rejected_queue_full_total",
        "counter",
        "Requests refused with 503 because the queue was full.",
        health.get("rejected_queue_full"),
    )
    counters = health.get("counters") or {}
    metric(
        "mlx_beam_prompt_tokens_total",
        "counter",
        "Prompt tokens prefilled.",
        counters.get("prompt_tokens"),
    )
    metric(
        "mlx_beam_generation_tokens_total",
        "counter",
        "Tokens generated.",
        counters.get("generation_tokens"),
    )
    metric(
        "mlx_beam_generation_steps_total",
        "counter",
        "Decode steps taken (one step serves every row in the batch).",
        counters.get("generation_steps"),
    )
    metric(
        "mlx_beam_prompt_seconds_total",
        "counter",
        "Wall seconds spent prefilling.",
        counters.get("prompt_time"),
    )
    metric(
        "mlx_beam_decode_seconds_total",
        "counter",
        "Wall seconds spent decoding.",
        counters.get("decode_time"),
    )
    metric(
        "mlx_beam_prefill_starved_calls_total",
        "counter",
        "Prefill calls that admitted nobody so a starved prompt got its full slice.",
        health.get("prefill_starved_calls"),
    )
    memory = health.get("memory") or {}
    for key in ("active", "peak", "cache"):
        metric(
            f"mlx_beam_memory_{key}_bytes",
            "gauge",
            f"Metal allocator {key} memory in bytes.",
            memory.get(key),
        )
    store = health.get("prompt_cache") or {}
    metric(
        "mlx_beam_prompt_cache_entries",
        "gauge",
        "Entries in the prefix store.",
        store.get("entries"),
    )
    metric(
        "mlx_beam_prompt_cache_bytes",
        "gauge",
        "Bytes the prefix store holds.",
        store.get("bytes"),
    )
    metric(
        "mlx_beam_prompt_cache_lookups_total",
        "counter",
        "Prefix store lookups.",
        store.get("lookups"),
    )
    metric(
        "mlx_beam_prompt_cache_hits_total",
        "counter",
        "Prefix store hits.",
        store.get("hits"),
    )
    metric(
        "mlx_beam_prompt_cache_tokens_restored_total",
        "counter",
        "Prompt tokens served from the prefix store instead of a prefill.",
        store.get("tokens_restored"),
    )
    repairs = (health.get("tools") or {}).get("repairs") or {}
    for rung, value in repairs.items():
        metric(
            "mlx_beam_tool_calls_total",
            "counter",
            "Tool calls by the rung of the repair ladder that settled them.",
            value,
            {"rung": rung},
        )
    spec = health.get("speculative")
    if spec:
        metric(
            "mlx_beam_speculative_cycles_total",
            "counter",
            "Verify cycles run.",
            spec.get("cycles"),
        )
        metric(
            "mlx_beam_speculative_plain_steps_total",
            "counter",
            "Plain decode steps taken while a proposer was configured.",
            spec.get("plain_steps"),
        )
        metric(
            "mlx_beam_speculative_drafted_total",
            "counter",
            "Draft tokens verified.",
            spec.get("drafted"),
        )
        metric(
            "mlx_beam_speculative_accepted_total",
            "counter",
            "Draft tokens accepted.",
            spec.get("accepted"),
        )
        metric(
            "mlx_beam_speculative_depth",
            "gauge",
            "Draft depth the regulator chose for the next cycle (0 while parked).",
            spec.get("depth"),
        )
        metric(
            "mlx_beam_speculative_parked",
            "gauge",
            "1 while the proposer is parked.",
            spec.get("parked"),
        )
        for i, p in enumerate(
            (spec.get("regulator") or {}).get("acceptance_by_position") or ()
        ):
            metric(
                "mlx_beam_speculative_acceptance",
                "gauge",
                "Smoothed acceptance at each chain position.",
                p,
                {"position": i + 1},
            )
    # What vLLM's dashboards read.
    out.append(_line("vllm:num_requests_running", health.get("in_flight"), labels))
    out.append(_line("vllm:num_requests_waiting", health.get("queued"), labels))
    out.append(_line("vllm:prompt_tokens_total", counters.get("prompt_tokens"), labels))
    out.append(
        _line("vllm:generation_tokens_total", counters.get("generation_tokens"), labels)
    )
    return "\n".join(out) + "\n"
