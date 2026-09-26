# mlx-beam

**B.E.A.M. — Batched Engine for Apple Metal.** Light and modular inference engine, built on [MLX](https://github.com/ml-explore/mlx).

> **Work in progress.** See the status table below and the [changelog](https://github.com/p4ik/mlx-beam/blob/main/CHANGELOG.md).

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

That is an OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`), Anthropic's Messages API next to it (`/v1/messages`, `/v1/messages/count_tokens`; point `ANTHROPIC_BASE_URL` at it), plus `/health` and `/metrics` (Prometheus), the first of which reports what was actually built: the KV layout per layer, the batching and cache settings, the Metal allocator's `memory` counters (`active`, `peak`, `cache` — RSS does not see these), and every request default with where it came from (flag, the model's `generation_config.json`, or mlx-lm's own). `POST /health/reset-peak` starts a fresh peak window for a measurement.

Flags follow mlx-lm's names where mlx-lm has one (`--temp`, `--top-p`, `--kv-bits`, `--prompt-cache-size`, `--chat-template`, …). Token limits say what they count: `--max-context` (prompt plus generated, a hard cap), `--max-prompt-tokens` (prompt, a hard cap), `--max-completion-tokens` (generated, the default a request may override), `--max-reasoning-tokens` (the think block; closed by force at the budget) and `--min-response-tokens` (what the answer keeps after the block). `beam serve --help` lists them all with their units, and the [configuration page](https://p4ik.github.io/mlx-beam/configuration/) has the same tables next to the request fields and what a checkpoint may bring along.

`--draft-model bundled` turns on speculative decoding with the draft head the checkpoint ships (Qwen3.5/3.8 packs carry one): a request decoding alone gets up to `--max-draft-tokens` + 1 tokens per model call (the drafts and the token the model samples after them; a regulator picks the depth per cycle from measured acceptance and cost, and parks the head when it loses), each one the model's own - its argmax over the verify forward for a greedy request, its own draw for a sampled one (the draw is keyed by position, so a seeded request gives the same tokens with and without the draft head). Logit bias, penalties and the thinking budget apply per verify position as in plain decoding. The head is primed over the prompt in the prefill and keeps its history across turns through the prefix cache. The output equals plain decoding up to kernel rounding at another width (bit-identical in bf16 in every measured case); `--exact-verify` closes that gap. Several requests at once decode plainly; `/health.speculative` shows cycles, drafted and accepted tokens.

Access follows the bind. On a loopback address (`127.0.0.1`, `::1`, `localhost` - the default) the server asks for nothing. Any other `--host` refuses to start without `--api-key <key>` (sent as `Authorization: Bearer <key>` or `x-api-key`; `/health` and `/metrics` sit behind it too) or `--skip-api-key`, an open server on purpose, said so at start. The `Host` header must name the machine as the server knows it - `localhost`, the bind address, or what `--allowed-hosts` adds - or the request gets 403, which is what keeps a page in a browser from reaching the server through a rebound name. CORS admits no origin until `--allowed-origins` names it. `/health.api.auth` says which mode is on.

Requests may use the names other servers taught clients: `max_tokens`, `thinking_token_budget`, `reasoning: {effort, max_tokens}`, `enable_thinking`, `reasoning_effort`. The model's thinking is returned in `reasoning` (`--reasoning-field` switches to `reasoning_content`, both, or none), counted in `usage.completion_tokens_details.reasoning_tokens`, and flagged there when a limit cut it (`thinking_truncated`, `response_truncated`). A tool call the model wrote badly goes through a repair ladder - mended JSON, values coerced to the declared schema - with every step reported in the call's `repair_actions`, and comes back as text when no step makes it valid.

## What sets it apart

- **Robust prefix cache** — Trie-backed store with its own byte budget, checkpoints for hybrid (recurrent) models, partial hits cut back to the last usable boundary. An SSD tier that survives restarts is planned.
- **No stalls** — A short request beside a long prefill answers in seconds.
- **Mixed-precision KV cache** — Bits per layer, set at conversion. Quantized after the prefill by default (the prefill never reads quantized data); `--kv-prefill quantized` writes it quantized from the first token for profiles measured that way.
- **Multi-token prediction** — The checkpoint's own draft head, verified exactly, greedy or sampled. On today for one request at a time; in the batch planned.
- **Thinking budget** — A hard cap on the reasoning trace, per request.
- **Responses and Messages APIs** — Next to chat completions, stateless: OpenAI's Responses shape and Anthropic's Messages shape on the same token path.
- **No bloat** — The core is the token path and the API formats. Vision is its own package, `mlx-beam-vision`, selected through the `vision` extra of this one (Qwen3-VL / Qwen3.5 / Qwen3.8, Mistral 3, Gemma 4, Muse Glimmer and Granite Vision towers); audio will join it; structured output and GGUF come as extras with a guard - a request that needs what is not installed gets a clear refusal. Expert streaming for models larger than memory is planned.

The engine reads standard MLX checkpoints. A checkpoint in the B.E.A.M. package layout (`extras/manifest.json` next to the shards; see the model cards under [huggingface.co/p4ik](https://huggingface.co/p4ik)) also tells it how its draft head was quantized and which KV prefill mode its profile was measured with.

## Why it exists

Existing MLX servers either stop at the basics or grow things that have no place in an inference engine: a built-in game, a cloud path that arrives with an update. The ones we ran daily also had bugs where it matters most: prefix cache, batching under load, vision. B.E.A.M. keeps the core to the token path and fixes those paths at the source. Everything else is a package or an extra you choose to install; nothing ever ships in the core that you did not ask for.

## Status

| Piece | State |
|---|---|
| CLI, packaging, CI | done |
| Vendored mlx-lm base (pinned; the local changes are listed in `VENDORED.md`) | done |
| OpenAI-compatible server, continuous batching, quantized KV cache | done, text only |
| Prefix cache with recurrent-state checkpoints | done, RAM tier |
| Reasoning budget, request defaults, sampling controls | done |
| Multi-token prediction | done for one request at a time (greedy); in the batch and under sampling planned |
| Vision, structured output | planned, as extras |
| Expert streaming from SSD | planned |

Measured numbers are published as they are measured, with machine, model and date.

## Contributing

See [CONTRIBUTING.md](https://github.com/p4ik/mlx-beam/blob/main/CONTRIBUTING.md). Rules for coding agents are in [AGENTS.md](https://github.com/p4ik/mlx-beam/blob/main/AGENTS.md).

## License

Apache-2.0. Vendored components keep their own licenses; see [NOTICE](https://github.com/p4ik/mlx-beam/blob/main/NOTICE).
