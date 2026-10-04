# mlx-beam

**B.E.A.M. — Batched Engine for Apple Metal.** Light and modular inference engine, built on [MLX](https://github.com/ml-explore/mlx).

> **Work in progress.** See the feature table below and the [changelog](https://github.com/p4ik/mlx-beam/blob/main/CHANGELOG.md).

## Install

```bash
uv tool install mlx-beam
beam doctor
```

Inside a uv project: `uv add mlx-beam`, then `uv run beam doctor`.

`beam doctor` reports the Python, MLX, device and memory it sees (`--json` for scripts) and exits non-zero when MLX is missing or fails to load.

## Serve

```bash
beam serve --model p4ik/Qwen3.8-27B-MLX-OptiQ-5bit --port 8000 \
  --max-completion-tokens 4096 --min-response-tokens 512 --max-reasoning-tokens 8192 \
  --kv-bits 8
```

An OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`) with Anthropic's Messages API beside it (`/v1/messages`; point `ANTHROPIC_BASE_URL` at it), `/health` reporting what was actually built and `/metrics` as plain-text counters. Flags follow mlx-lm's names where mlx-lm has one; `beam serve --help` lists them, and the [configuration page](https://p4ik.github.io/mlx-beam/configuration/) has the same tables next to the request fields and what a checkpoint brings along. Off loopback the server wants `--api-key`. `--draft-model bundled` turns on speculative decoding with the draft head a checkpoint ships.

## Features

Everything the engine does, with where each piece stands. Extras are packages or optional installs beside the core.

| Feature | What it does | State |
|---|---|---|
| **OpenAI API** | Chat completions, completions, Responses and models, with streaming. | built |
| **Anthropic API** | Messages with thinking blocks and tool use, on the same token path. | built |
| **Health and metrics** | `/health` shows each capability with evidence; `/metrics` serves counters. | built |
| **Access control** | An API key is required off loopback; Host check and CORS allowlist. | built |
| **CLI** | `beam serve` takes mlx-lm's flag names; `beam doctor` checks the machine. | built |
| **Model classes** | Dense, MoE, hybrid-recurrent, sliding window, sinks, MLA, MRoPE. | built |
| **Package layout** | Manifest with KV profile and draft-head quantization; plain MLX loads too. | built |
| **Vision [Extra]** | Qwen3-VL/3.5/3.6/3.8, Mistral 3, Gemma 4, Muse Glimmer, Granite. | built |
| **Audio [Extra]** | Audio input for models that take it, as its own package. | planned |
| **Images [Extra]** | Image generation and editing through the Images API. | planned |
| **GGUF [Extra]** | GGUF checkpoints load behind a guard. | planned |
| **Continuous batching** | Short requests are answered beside a long prefill, capped by a valve. | built |
| **Prefix cache** | The KV of a shared prefix is reused, for recurrent and window layers too. | built |
| **Prefix cache SSD tier** | A second tier on SSD lets entries outlive eviction from RAM and restarts. | planned |
| **Mixed-precision KV cache** | Bits per layer come from the conversion; quantized after or during prefill. | built |
| **Multi-token prediction** | An own draft head predicts ahead, verified exactly, one request at a time. | built |
| **Batched MTP** | Several requests speculate at once. | planned |
| **Expert streaming** | Models larger than memory run with experts streamed from SSD. | planned |
| **Reasoning control** | Thinking on or off, a budget per request, effort levels, marker families. | built |
| **Tool calling** | Parsers per model family and a repair ladder that reports every step. | built |
| **Sampling controls** | Temperature, top-p/k, min-p, penalties, logit bias, seeds and defaults. | built |
| **Structured [Extra]** | JSON schema and grammar-constrained decoding. | planned |

The engine reads standard MLX checkpoints. A checkpoint in the B.E.A.M. package layout (`extras/manifest.json` next to the shards; see the model cards under [huggingface.co/p4ik](https://huggingface.co/p4ik)) also tells it how its draft head was quantized and which KV prefill mode its profile was measured with. Which model families the engine handles with what - architecture, thinking, tool calls, vision, draft head - is on the [models page](https://p4ik.github.io/mlx-beam/models/).

## Why it exists

Existing MLX servers either stop at the basics or grow things that have no place in an inference engine: a built-in game, a cloud path that arrives with an update. The ones we ran daily also had bugs where it matters most: prefix cache, batching under load, vision. B.E.A.M. keeps the core to the token path and fixes those paths at the source. Everything else is a package or an extra you choose to install; nothing ever ships in the core that you did not ask for.

## Measured

Measured numbers are published as they are measured, with machine, model and date. Mac mini M4 Pro 64 GB, Qwen3.8-27B 5-bit with 8-bit KV, one request, 2026-09-26 (`0.1.0a6`): decode 11.4 tok/s at 12 000 generated tokens; prefill 115 tok/s over a 16 588-token prompt; a short request answered in 2.4 s while that prefill ran; 45 minutes under a steady stream, 5 057 requests, none failed. Speculative acceptance and batch numbers follow when they are measured the same way.

## Contributing

See [CONTRIBUTING.md](https://github.com/p4ik/mlx-beam/blob/main/CONTRIBUTING.md). Rules for coding agents are in [AGENTS.md](https://github.com/p4ik/mlx-beam/blob/main/AGENTS.md).

## License

Apache-2.0. Vendored components keep their own licenses; see [NOTICE](https://github.com/p4ik/mlx-beam/blob/main/NOTICE).
