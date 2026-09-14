# mlx-beam

**B.E.A.M. — Batched Engine for Apple Metal.** An [MLX](https://github.com/ml-explore/mlx) inference engine for hybrid-attention and mixture-of-experts language models on Apple silicon.

> **Work in progress.** This repository holds the project skeleton. Nothing here serves a model yet. Follow the [changelog](CHANGELOG.md) for what lands.

## What it is for

Models like Qwen3.8 (Gated DeltaNet + attention) and Qwen3.8-Flash-Next (125B-A6B MoE with an on-disk n-gram table) do not fit the assumptions of a plain transformer server: their recurrent state cannot be trimmed like a KV cache, their experts do not fit in memory, and speculative decoding under batching needs a rollback that most engines lack. B.E.A.M. is built around those cases:

- a prefix cache with recurrent-state checkpoints, so a long conversation resumes from disk instead of re-prefilling;
- multi-token prediction inside continuous batching;
- expert streaming from SSD with a resident pool, for models larger than RAM;
- a small, explicit OpenAI-compatible API, with everything else as optional extras.

The engine reads standard MLX checkpoints and the B.E.A.M. package layout (`extras/` next to the shards; see the model cards under [huggingface.co/p4ik](https://huggingface.co/p4ik)).

## Install

Not on PyPI yet. Until the first pre-release, install the command line tool straight from the repository:

```bash
uv tool install "mlx-beam @ git+https://github.com/p4ik/mlx-beam"
beam doctor
```

Inside an existing uv project, add it as a dependency instead and run it through uv:

```bash
uv add "mlx-beam @ git+https://github.com/p4ik/mlx-beam"
uv run beam doctor
```

`beam doctor` prints the Python, MLX, device and memory it sees (`--json` for scripts) and exits non-zero when MLX is missing or fails to load. It is the only command so far.

## Status

| Piece | State |
|---|---|
| CLI, packaging, CI | skeleton |
| vendored mlx-lm base | planned |
| prefix cache with recurrent-state checkpoints | planned |
| MTP in the batch | planned |
| expert streaming | planned |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Working rules for people and agents are in [AGENTS.md](AGENTS.md).

## License

Apache-2.0. Vendored components keep their own licenses; see [NOTICE](NOTICE) and `VENDORED.md`.
