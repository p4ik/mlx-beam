---
layout: ../layouts/Doc.astro
title: Configuration
description: Where a running server takes its values from - the command line, the request, the checkpoint - and which one wins.
---

A value reaches the engine at one of three points: when the server starts (flags), with each request (body fields), or from the checkpoint itself (what the model ships). The rule between them is short: **a flag beats the checkpoint, and a request beats both** where a request may say anything at all. Hard caps only go one way - a request may lower `--max-prompt-tokens`, never raise it, and `--max-context` never exceeds the model's own window. `/health` reports every default together with its source (`flag`, `generation_config.json`, `mlx-lm`, or `mlx-beam` for the two reasoning defaults the engine sets itself), so what is in effect is never a guess.

## Start: `beam serve` flags

Flags take mlx-lm's names where mlx-lm has one (`--temp`, `--top-p`, `--kv-bits`, `--prompt-cache-size`, `--chat-template`). Token limits say what they count; each one counts a different set. `beam serve --help` prints the same list - the tables below are generated from the parser and checked against it in CI.

<!-- generated: beam serve flags -->

### Server

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--model` | text | — | local path or Hugging Face repo id |
| `--model-alias` | text | — | model id shown to clients (default: --model) |
| `--reasoning-field` | `reasoning` / `reasoning_content` / `both` / `none` | `reasoning` | where a chat completion carries the model's thinking: the field name(s), or none to leave the think markers in the content |
| `--host` | text | `127.0.0.1` | address to listen on |
| `--port` | integer | `8000` | TCP port to listen on |
| `--allowed-origins` | `ORIGIN` (one or more) | `*` | origins CORS admits (default: any) |
| `--max-queued` | integer | — | requests allowed to wait for a batch slot; one more is a 503 with Retry-After (default: unlimited) |
| `--trust-remote-code` | switch | — | run a model_file shipped inside the checkpoint |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO` | how much the server log says |

### Chat template

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--chat-template` | text | — | Jinja text, or the path of a .jinja file, used instead of the model's own template |
| `--use-default-chat-template` | switch | — | give a model that ships no chat template a plain ChatML one |
| `--chat-template-args` | `JSON` | — | handed to every template render, e.g. '{"enable_thinking": false}'; a request's chat_template_kwargs override it |

### Token limits

Every value counts tokens; each one counts a different set.

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--max-context` | integer | — | prompt plus generated tokens, a hard cap: a prompt whose reserve does not fit is a 400, a larger max_tokens is served capped at what the context holds (default: the model's own context length) |
| `--max-prompt-tokens` | integer | — | prompt tokens, a hard cap below the context: a longer prompt is a 400; a request's max_prompt_tokens may only lower it |
| `--max-completion-tokens` | integer | — | generated tokens when the client sends no max_tokens / max_completion_tokens / max_output_tokens; the request overrides (mlx-lm: --max-tokens, default 512) |
| `--max-reasoning-tokens` | integer | — | reasoning tokens (what usage.reasoning_tokens counts) when the client sends no max_reasoning_tokens; the think block is closed by force at the budget (default: unbounded) |
| `--min-response-tokens` | integer | `0` | tokens kept for the answer after the think block when the client sends no min_response_tokens; the reasoning budget is cut to leave them, and a request whose context cannot hold them is a 400 |

### Sampling defaults

Used when the client sends nothing; a flag beats the model's generation_config.json, which beats mlx-lm's defaults.

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--temp` | number | — | temperature |
| `--top-p` | number | — | nucleus sampling |
| `--top-k` | integer | — | top-k sampling (0 = off) |
| `--min-p` | number | — | min-p sampling (0 = off) |

### KV cache

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--kv-bits` | `4` / `8` | — | quantize the full-attention KV cache to this many bits |
| `--kv-group-size` | `32` / `64` / `128` | `64` | group size of the KV quantization |
| `--kv-config` | text | — | JSON file, bits per layer: {"bits": 4, "group_size": 64, "layers": {"3": 8}}, or the list a quantized package ships ([{"layer_idx": 3, "bits": 4, "group_size": 64}, ...]): listed layers take their bits, unlisted ones follow --kv-bits (optiq leaves them at full precision and ignores --kv-bits) |
| `--kv-prefill` | `exact` / `quantized` | — | when a quantized layer becomes quantized: 'exact' (default) keeps the prompt at model precision while it is prefilled and quantizes at the handover to decoding (mlx-lm's generate_step with quantized_kv_start at the prompt's end); 'quantized' writes it quantized from the first token, which saves the prompt's full-precision transient (~2 GB for a 64k prompt on a 27B) and on some models costs accuracy - use it for a profile that was measured with it (a kv_config object may carry "prefill": "quantized") |

### Batching

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--decode-concurrency` | integer | `8` | sequences decoded in one batch |
| `--prompt-concurrency` | integer | `2` | prompts prefilled in one batch |
| `--prefill-step-size` | integer | `2048` | prompt tokens per model call while prefilling |
| `--prefill-slice` | integer | `512` | prompt tokens a prefill runs before decode gets a turn |
| `--decode-share` | `SHARE` | `0.5` | share of the worker's time decode keeps while a prefill runs (0-1) |

### Speculative decoding

Off unless asked; a checkpoint that bundles a draft head says so at start.

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--draft-model` | text | — | the proposer that drafts tokens for the verify pass: 'bundled' takes the draft head the checkpoint ships (the package manifest's parts.mtp, config.json mtp_file, or mtp.* tensors in the shards); a repo or path for an external drafter is not supported yet |
| `--max-draft-tokens` | integer | `3` | cap on the drafts verified per cycle (default: 3, the fixed depth at this stage; a lower value lowers it) |

### Prompt cache

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--prompt-cache-size` | integer | `16` | stored prefixes: a number of entries, not a size in bytes |
| `--prompt-cache-bytes` | integer | — | RAM budget in bytes for the stored prefixes (default: unlimited); the store's own limit, separate from the caches of running requests |

<!-- /generated -->

## Request: fields a call may send

Chat completions (`/v1/chat/completions`) take the fields below; `/v1/completions` takes the sampling, penalty and stop fields plus `prompt`, `echo` and an integer `logprobs` (0-20); `/v1/responses` takes them with its own shape (`input`, `instructions`, `max_output_tokens`, `reasoning`). A field left out falls back to the server's default for it.

| Field | Values | What it does |
|---|---|---|
| `max_completion_tokens`, `max_tokens` | integer ≥ 1 | Generated tokens for this call. `max_tokens` is the older OpenAI name and is accepted as the same thing. A small limit is served as sent; only a context that cannot hold the answer's reserve is refused. |
| `max_prompt_tokens` | integer ≥ 1 | A prompt cap for this call; it may only lower the server's. |
| `max_reasoning_tokens`, `thinking_token_budget`, `reasoning.max_tokens` | integer ≥ 0 | Reasoning tokens the think block may take; at the budget the block is closed by force and the answer keeps `min_response_tokens`. The first name wins when several are sent. |
| `min_response_tokens` | integer ≥ 0 | Tokens kept for the answer after the think block. |
| `reasoning_effort`, `reasoning.effort` | `none` / `off` / `false`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` / `ultra` | `none` turns thinking off. Otherwise the word reaches the template the way the template takes it, measured at load (`/health.reasoning.effort`): a template that checks the word gets it when it is in its set, else the nearest rung it accepts (up first, then down); a template that takes the word unchecked gets it as sent; a template without the kwarg gets nothing. A word the template still rejects at request time is retried through its neighbours. |
| `enable_thinking` | boolean, or the words `true` / `false` | Thinking on or off for this call; handed to the template. |
| `chat_template_kwargs` | object | Extra variables for the template render, on top of `--chat-template-args` and the aliases above. |
| `temperature`, `top_p`, `top_k`, `min_p` | 0-2, 0-1, integer (0 or -1 = off), 0-1 | Sampling; `temperature` 0 is greedy. |
| `min_tokens_to_keep` | integer ≥ 1 | Tokens `min_p` may never filter away. |
| `xtc_probability`, `xtc_threshold` | 0-1, 0-0.5 | Exclude-top-choices sampling; eos and newline are never cut. |
| `seed` | integer | A seeded call samples with its own random key, so it repeats. |
| `repetition_penalty`, `repetition_context_size` | ≥ 0 (1 = off), integer ≥ 1 (20) | Sign-aware multiplicative penalty on tokens seen in the last *n*. |
| `presence_penalty`, `presence_context_size` | -2 to 2, integer ≥ 1 (20) | Additive penalty on tokens present in the last *n*. |
| `frequency_penalty`, `frequency_context_size` | -2 to 2, integer ≥ 1 (20) | Additive penalty in proportion to how often a token appeared in the last *n*. |
| `logit_bias` | object, token id → -100 to 100 | Added to the logits of those ids at every step. |
| `logprobs`, `top_logprobs` | boolean, integer 0-20 | Log-probabilities per token; `top_logprobs` needs `logprobs: true`. |
| `stop` | string or list of strings | Sequences that end the answer; the model's eos ids always do. |
| `tools`, `tool_choice` | list of function tools; `auto` or `none` | Tools the template renders and the answer is parsed for; `none` sends none. Other `tool_choice` values are refused. |
| `stream`, `stream_options.include_usage` | boolean | Server-sent events; with `include_usage` the last event carries `usage`. |

Refused, and said so in the error: `n` above 1, `best_of`, `suffix` (insertion), `response_format` other than `text` and stored state on `/v1/responses` (`previous_response_id`, `conversation`). Structured output is an extra, not part of the core.

## Checkpoint: what the model ships

The engine reads standard MLX checkpoints, and a checkpoint can carry settings of its own:

- **`generation_config.json`** - `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `presence_penalty` and `frequency_penalty` become the server's sampling defaults, below the flags and above mlx-lm's own; `do_sample: false` means greedy.
- **The chat template** in `tokenizer_config.json` or `chat_template.jinja` - the tool-call markers the engine watches for are inferred from the template that actually renders, the think markers from the model's vocabulary; `--chat-template` replaces the template, `--use-default-chat-template` gives a model without one a plain ChatML template.
- **A KV profile** - a quantized package may ship a bits-per-layer list (`[{"layer_idx": 3, "bits": 4, "group_size": 64}, …]`) or an object (`{"bits": 4, "group_size": 64, "layers": {"3": 8}, "prefill": "quantized"}`); the engine does not read it on its own, `--kv-config` points at the file. Listed layers take their bits, the rest follow `--kv-bits`. The prefill mode comes from the object's own `prefill`, else - when the file is the one the package manifest names (`parts.kv_config`: the same bytes as its SHA-256, or the same path for a manifest without one; a profile edited in place is not the measured one) - from the mode the package was measured with; `--kv-prefill` beats both.
- **A draft head** - the file the package manifest names under `parts.mtp` (checked against its SHA-256), else the file `mtp_file` in `config.json` names, else the `mtp.*` tensors in the shards. It is used only with `--draft-model bundled`; started without the flag, a checkpoint that bundles one says so. Its bits and group size come from the manifest, else from `mtplx_mtp_quantization` in `config.json`, else the head is quantized to 4 bits at load; its norm weights are shifted by +1 unless the manifest's `norm_convention` is `mlx`.
- **A package manifest** - a checkpoint in the B.E.A.M. package layout names it in `config.json` (`extras.manifest`, normally `extras/manifest.json`); its `parts` describe the draft head and the KV profile above, each with the file and its SHA-256. A plain checkpoint has none and every reader uses its own defaults.

What the engine built from all of this is in `/health`: the KV layout layer by layer under `kv.applied`, the batching and cache settings, the draft head and its counters under `speculative`, and every request default with its source.
