# Changelog

All notable changes to mlx-beam. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
PEP 440 with SemVer meaning (`0.y` may break, `0.y.z` fixes).

## [Unreleased]

### Added
- Request defaults with provenance: `--max-completion-tokens`, `--temp`,
  `--top-p`, `--top-k`, `--min-p` beat the model's `generation_config.json`,
  which beats mlx-lm's defaults; `/health` and the start banner say which.
- Reasoning budget: `--max-reasoning-tokens` and `--min-response-tokens`
  (a request's `max_reasoning_tokens` / `min_response_tokens` override);
  at the budget the think block is closed by force, the answer keeps its
  reserve, and `usage.completion_tokens_details` reports
  `thinking_truncated` / `response_truncated`. A small client limit is
  served as sent; only a context that cannot hold the reserve is a 400.
- Request aliases: `max_tokens`, `thinking_token_budget`,
  `reasoning: {effort, max_tokens}`, top-level `enable_thinking`.
  `reasoning_effort` is mapped onto the levels the template accepts, with
  a ladder of aliases when it rejects one by name. Earlier turns'
  reasoning is copied into the message key the template reads.
- Sampling: `seed` per request (own random key per row), `xtc_probability`,
  `xtc_threshold`, `min_tokens_to_keep`, `presence_context_size`,
  `frequency_context_size`; `logprobs` / `top_logprobs` on chat and the
  legacy integer `logprobs` on completions. Rows with equal settings share
  one sampler, so the batch samples in a single call.
- `--chat-template` (text or `.jinja` path), `--use-default-chat-template`,
  `--chat-template-args`, `--allowed-origins`, `--max-prompt-tokens`
  (a request's `max_prompt_tokens` may only lower it), `--max-queued`
  (a 503 with `Retry-After` beyond it), `--model-alias`,
  `--prompt-cache-bytes`.
- Server: 30 s socket timeout with the request cancelled on a stalled
  client, `Connection: close` on errors, chunks that are ready together go
  out in one write.
- Prefix store: the system block as its own entry (evicted last), a
  checkpoint every 2048 prefill tokens, a cancelled prefill keeps its part.

### Changed
- `--reasoning-field none` keeps counting reasoning tokens and shows a think
  block the prompt opened; the text automaton runs in every mode.
- Batch sizes are `--decode-concurrency` / `--prompt-concurrency` (mlx-lm's
  server names); `/health` reports the store under `prompt_cache`.
- Engine: one worker thread, continuous batching, KV policy per layer,
  `/health` with the cache layout that was actually built.
- OpenAI-compatible server: `/v1/chat/completions`, `/v1/completions`,
  `/v1/responses` (stateless, function tools), `/v1/models`; `beam serve`.
  The model's thinking goes to the `reasoning` field; `--reasoning-field`
  switches to `reasoning_content`, both, or none (markers stay in the text).
- Prefix store with recurrent-state checkpoints: hybrid models resume from
  the last system or user boundary instead of re-prefilling everything.
- Quantized KV cache for the batch path and a tiled quantized attention,
  ported from mlx-optiq; the rotating-cache merge guard (see `VENDORED.md`).
- Vendored mlx-lm (inference subset, pinned commit) with four local changes:
  package-relative imports, prefill mask for hybrids, per-step cache eval,
  and a scheduler that neither pads nor stalls (see `VENDORED.md`).
- Project skeleton: `beam --version`, `beam doctor`, packaging with hatch-vcs,
  pre-commit (black, isort, ruff), CI on Apple silicon, contribution rules.
