# mlx-beam

**B.E.A.M. — Batched Engine for Apple Metal.** Light and modular inference engine, built on [MLX](https://github.com/ml-explore/mlx).

> **Work in progress.** See the status table below and the [changelog](https://github.com/p4ik/mlx-beam/blob/main/CHANGELOG.md).

## Install

```bash
uv tool install mlx-beam
beam doctor
```

Inside a uv project: `uv add mlx-beam`, then `uv run beam doctor`.

`beam doctor` reports the Python, MLX, device and memory it sees (`--json` for scripts) and exits non-zero when MLX is missing or fails to load. It is the only command so far.

## What sets it apart

- **Robust prefix cache** — RAM and SSD tiers, checkpoints for hybrid models. Survives model swaps and restarts.
- **Expert streaming** — Mixture-of-experts models larger than memory. Residency configurable, from minimal RAM to fully resident.
- **No bloat** — The core is the token path. Vision, audio, conversion, structured output and tool-call repair are optional extras.
- **Batched MTP** — Multi-token prediction stays on with many requests at once.
- **Batched vision** — Images go through the same scheduler; no request waits behind a picture.
- **No stalls** — A short request beside a long prefill answers in seconds.
- **Mixed-precision KV cache** — Bits per layer, set at conversion.
- **Thinking budget** — A hard cap on the reasoning trace, per request.
- **Responses API** — Next to chat completions, stateless.

The engine reads standard MLX checkpoints and the B.E.A.M. package layout (`extras/` next to the shards; see the model cards under [huggingface.co/p4ik](https://huggingface.co/p4ik)).

## Why it exists

Existing MLX servers either stop at the basics or grow things that have no place in an inference engine: a built-in game, a cloud path that arrives with an update. The ones we ran daily also had bugs where it matters most: prefix cache, batching under load, vision. B.E.A.M. keeps the core to the token path and fixes those paths at the source. Everything else is an extra you choose to install; nothing ever ships in the core that you did not ask for.

## Status

| Piece | State |
|---|---|
| CLI, packaging, CI | skeleton |
| Vendored mlx-lm base | planned |
| Prefix cache with recurrent-state checkpoints | planned |
| Multi-token prediction in the batch | planned |
| Expert streaming from SSD | planned |

Measured numbers are published as they are measured, with machine, model and date.

## Contributing

See [CONTRIBUTING.md](https://github.com/p4ik/mlx-beam/blob/main/CONTRIBUTING.md). Rules for coding agents are in [AGENTS.md](https://github.com/p4ik/mlx-beam/blob/main/AGENTS.md).

## License

Apache-2.0. Vendored components keep their own licenses; see [NOTICE](https://github.com/p4ik/mlx-beam/blob/main/NOTICE).
