---
layout: ../layouts/Doc.astro
title: Configuration
description: Where a running server takes its values from, and which one wins.
---

A value comes from one of three places: a flag at start, a field in the request, or the checkpoint itself. **A flag beats the checkpoint, a request beats both.** Hard caps only go down: a request may lower `--max-prompt-tokens`, never raise it, and `--max-context` never exceeds the model's own window. `/health` names every default with its source.

## Start: `beam serve` flags

Flags take mlx-lm's names where mlx-lm has one. `beam serve --help` prints the same list; the tables are generated from the parser and checked in CI. Hover a shortened description for the whole text.

<!-- generated: beam serve flags -->

### Server

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--model` | text | — | local path or Hugging Face repo id |
| `--model-alias` | text | — | <span class="hint" tabindex="0" data-tip="model id shown to clients (default: --model)">model id shown to clients</span> |
| `--reasoning-field` | `reasoning` / `reasoning_content` / `both` / `none` | `reasoning` | <span class="hint" tabindex="0" data-tip="where a chat completion carries the model&#x27;s thinking: the field name(s), or none to leave the think markers in the content">where a chat completion carries the model's thinking</span> |
| `--host` | text | `127.0.0.1` | address to listen on |
| `--port` | integer ≥ 1 | `8000` | TCP port to listen on |
| `--allowed-origins` | `ORIGIN` (one or more) | — | <span class="hint" tabindex="0" data-tip="origins CORS admits, so a page in a browser may call the server; by default none">origins CORS admits, so a page in a browser may call the server</span> |
| `--max-queued` | integer ≥ 1 | — | <span class="hint" tabindex="0" data-tip="requests allowed to wait for a batch slot; one more is a 503 with Retry-After (default: unlimited)">requests allowed to wait for a batch slot</span> |
| `--trust-remote-code` | switch | — | run a model_file shipped inside the checkpoint |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO` | how much the server log says |

### Access

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--api-key` | `KEY` | — | <span class="hint" tabindex="0" data-tip="the key every request must carry (Authorization: Bearer or x-api-key); required off loopback unless --skip-api-key">the key every request must carry</span> |
| `--skip-api-key` | switch | — | serve without a key on a non-loopback --host, on purpose |
| `--allowed-hosts` | `HOST` (one or more) | — | <span class="hint" tabindex="0" data-tip="Host header values accepted next to localhost and --host, e.g. the machine&#x27;s name or LAN address behind a wildcard bind; other hosts get 403">Host header values accepted next to localhost and --host, e.g. the machine's name or LAN address behind a wildcard bind</span> |

### Chat template

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--chat-template` | text | — | Jinja text, or the path of a .jinja file, used instead of the model's own template |
| `--use-default-chat-template` | switch | — | give a model that ships no chat template a plain ChatML one |
| `--chat-template-args` | `JSON` | — | <span class="hint" tabindex="0" data-tip="handed to every template render, e.g. &#x27;{&quot;enable_thinking&quot;: false}&#x27;; a request&#x27;s chat_template_kwargs override it">handed to every template render, e.g. '{"enable_thinking": false}'</span> |

### Token limits

Every value counts tokens; each one counts a different set.

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--max-context` | integer ≥ 1 | — | <span class="hint" tabindex="0" data-tip="prompt plus generated tokens, a hard cap: a prompt whose reserve does not fit is a 400, a larger max_tokens is served capped at what the context holds (default: the model&#x27;s own context length)">prompt plus generated tokens, a hard cap</span> |
| `--max-prompt-tokens` | integer ≥ 1 | — | <span class="hint" tabindex="0" data-tip="prompt tokens, a hard cap below the context: a longer prompt is a 400; a request&#x27;s max_prompt_tokens may only lower it">prompt tokens, a hard cap below the context</span> |
| `--max-completion-tokens` | integer ≥ 1 | — | <span class="hint" tabindex="0" data-tip="generated tokens when the client sends no max_tokens / max_completion_tokens / max_output_tokens; the request overrides (mlx-lm: --max-tokens, default 512)">generated tokens when the client sends no max_tokens / max_completion_tokens / max_output_tokens</span> |
| `--max-reasoning-tokens` | integer ≥ 0 | — | <span class="hint" tabindex="0" data-tip="reasoning tokens (what usage.reasoning_tokens counts) when the client sends no max_reasoning_tokens; the think block is closed by force at the budget (default: unbounded)">reasoning tokens</span> |
| `--min-response-tokens` | integer ≥ 0 | `0` | <span class="hint" tabindex="0" data-tip="tokens kept for the answer after the think block when the client sends no min_response_tokens; the reasoning budget is cut to leave them, and a request whose context cannot hold them is a 400">tokens kept for the answer after the think block when the client sends no min_response_tokens</span> |

### Sampling defaults

Used when the client sends nothing; a flag beats the model's generation_config.json, which beats mlx-lm's defaults.

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--temp` | number | — | temperature |
| `--top-p` | number | — | nucleus sampling |
| `--top-k` | integer ≥ 0 | — | <span class="hint" tabindex="0" data-tip="top-k sampling (0 = off)">top-k sampling</span> |
| `--min-p` | number | — | <span class="hint" tabindex="0" data-tip="min-p sampling (0 = off)">min-p sampling</span> |

### KV cache

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--kv-bits` | `4` / `8` | — | quantize the full-attention KV cache to this many bits |
| `--kv-group-size` | `32` / `64` / `128` | — | <span class="hint" tabindex="0" data-tip="group size of the KV quantization (default: 64, or what the checkpoint&#x27;s kv_config carries; the flag beats the file)">group size of the KV quantization</span> |
| `--kv-config` | text | — | <span class="hint" tabindex="0" data-tip="JSON file, bits per layer: {&quot;bits&quot;: 4, &quot;group_size&quot;: 64, &quot;layers&quot;: {&quot;3&quot;: 8}}, or the list a quantized package ships ([{&quot;layer_idx&quot;: 3, &quot;bits&quot;: 4, &quot;group_size&quot;: 64}, ...]): listed layers take their bits, unlisted ones follow --kv-bits (optiq leaves them at full precision and ignores --kv-bits)">JSON file, bits per layer</span> |
| `--kv-prefill` | `exact` / `quantized` | — | <span class="hint" tabindex="0" data-tip="when a quantized layer becomes quantized: &#x27;exact&#x27; (default) keeps the prompt at model precision while it is prefilled and quantizes at the handover to decoding (mlx-lm&#x27;s generate_step with quantized_kv_start at the prompt&#x27;s end); &#x27;quantized&#x27; writes it quantized from the first token, which saves the prompt&#x27;s full-precision transient (~2 GB for a 64k prompt on a 27B) and on some models costs accuracy - use it for a profile that was measured with it (a kv_config object may carry &quot;prefill&quot;: &quot;quantized&quot;)">when a quantized layer becomes quantized</span> |

### Batching

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--decode-concurrency` | integer ≥ 1 | `8` | sequences decoded in one batch |
| `--prompt-concurrency` | integer ≥ 1 | `2` | prompts prefilled in one batch |
| `--prefill-step-size` | integer ≥ 1 | `2048` | prompt tokens per model call while prefilling |
| `--prefill-slice` | integer ≥ 1 | `512` | prompt tokens a prefill runs before decode gets a turn |
| `--decode-share` | `SHARE` | `0.5` | <span class="hint" tabindex="0" data-tip="share of the worker&#x27;s time decode keeps while a prefill runs (0-1)">share of the worker's time decode keeps while a prefill runs</span> |

### Speculative decoding

Off unless asked; a checkpoint that bundles a draft head says so at start.

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--draft-model` | text | — | <span class="hint" tabindex="0" data-tip="the proposer that drafts tokens for the verify pass: &#x27;bundled&#x27; takes the draft head the checkpoint ships (the package manifest&#x27;s parts.mtp, config.json mtp_file, or mtp.* tensors in the shards); a repo or path for an external drafter is not supported yet">the proposer that drafts tokens for the verify pass</span> |
| `--exact-verify` | `off` / `kernels` / `positions` | `off` | <span class="hint" tabindex="0" data-tip="how the verify runs: &#x27;off&#x27; checks the k+1 drafts in one forward (the fast path; /health.speculative.exact says whether that forward gives the same logits as one-token forwards on this machine); &#x27;kernels&#x27; runs that forward through projections and attention that keep single-row arithmetic for a block (vendored from mlx-vlm) and keeps them only if the warm-up finds them bit-equal, else falls back to &#x27;off&#x27; and says so; &#x27;positions&#x27; feeds one token per forward and stops at the first rejected draft - exact by construction at plain decoding&#x27;s cost, the reference for the other two">how the verify runs</span> |
| `--max-draft-tokens` | integer ≥ 1 | `3` | <span class="hint" tabindex="0" data-tip="cap on the drafts verified per cycle; the regulator picks each cycle&#x27;s depth below it from the acceptance and the cycle cost it measures (default: 3, the depth with the best gain measured on a 27B)">cap on the drafts verified per cycle</span> |

### Prompt cache

| Flag | Value | Default | What it does |
|---|---|---|---|
| `--prompt-cache-size` | integer ≥ 1 | `16` | <span class="hint" tabindex="0" data-tip="stored prefixes: a number of entries, not a size in bytes">stored prefixes</span> |
| `--prompt-cache-bytes` | integer ≥ 1 | — | <span class="hint" tabindex="0" data-tip="RAM budget in bytes for the stored prefixes (default: unlimited); the store&#x27;s own limit, separate from the caches of running requests">RAM budget in bytes for the stored prefixes</span> |

<!-- /generated -->

## Request: fields a call may send

`/v1/chat/completions` takes the fields below. `/v1/completions` takes the sampling, penalty and stop fields plus `prompt`, `echo` and an integer `logprobs`; `/v1/responses` takes them in its own shape; `/v1/messages` takes Anthropic's fields and answers in Anthropic's blocks, events and errors. `cache_control` markers are accepted; the prompt cache checkpoints every message boundary on its own and reports hits in `usage.cache_read_input_tokens`. Roles are `system`, `developer`, `user`, `assistant` and `tool`; `developer` becomes `system` for a template that does not know it. A field left out falls back to the server's default.

| Field | Values | What it does |
|---|---|---|
| `max_completion_tokens`<br>`max_tokens` | integer ≥ 1 | <span class="hint" tabindex="0" data-tip="Generated tokens for this call. max_tokens is the older OpenAI name and is accepted as the same thing. A small limit is served as sent; only a context that cannot hold the answer&#x27;s reserve is refused.">Generated tokens for this call.</span> |
| `max_prompt_tokens` | integer ≥ 1 | <span class="hint" tabindex="0" data-tip="A prompt cap for this call; it may only lower the server&#x27;s.">A prompt cap for this call</span> |
| `max_reasoning_tokens`<br>`thinking_token_budget`<br>`reasoning.max_tokens` | integer ≥ 0 | <span class="hint" tabindex="0" data-tip="Reasoning tokens the think block may take; at the budget the block is closed by force and the answer keeps min_response_tokens. The first name wins when several are sent.">Reasoning tokens the think block may take</span> |
| `min_response_tokens` | integer ≥ 0 | Tokens kept for the answer after the think block. |
| `reasoning_effort`<br>`reasoning.effort` | `none` / `off` / `false`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` / `ultra` | <span class="hint" tabindex="0" data-tip="none turns thinking off. Otherwise the word reaches the template the way the template takes it, measured at load (/health.reasoning.effort): a template that checks the word gets it when it is in its set, else the nearest rung it accepts (up first, then down); a template that takes the word unchecked gets it as sent; a template without the kwarg gets nothing. A word the template still rejects at request time is retried through its neighbours.">`none` turns thinking off.</span> |
| `enable_thinking` | boolean, or the words `true` / `false` | <span class="hint" tabindex="0" data-tip="Thinking on or off for this call; handed to the template. For a family whose template has no switch (Harmony, Muse) the answer is opened in the prompt instead; with tools declared the opener would skip the channel or recipient a call needs, so there the block is kept out by a reasoning budget of zero instead - the model may call a tool, not think.">Thinking on or off for this call</span> |
| `chat_template_kwargs` | object | Extra variables for the template render, on top of `--chat-template-args` and the aliases above. |
| `temperature` | 0-2 | Sampling temperature; 0 is greedy. |
| `top_p` | 0-1 | Nucleus sampling. |
| `top_k` | integer (0 or -1 = off) | Top-k sampling. |
| `min_p` | 0-1 | Min-p sampling. |
| `min_tokens_to_keep` | integer ≥ 1 | Tokens `min_p` may never filter away. |
| `xtc_probability` | 0-1 | <span class="hint" tabindex="0" data-tip="Exclude-top-choices sampling; eos and newline are never cut.">Exclude-top-choices sampling</span> |
| `xtc_threshold` | 0-0.5 | The probability above which a token counts as a top choice. |
| `seed` | integer | A seeded call samples with its own random key, so it repeats. |
| `repetition_penalty` | ≥ 0 (1 = off) | Sign-aware multiplicative penalty on tokens seen in the last *n*. |
| `repetition_context_size` | integer ≥ 1 (20) | The *n* the repetition penalty looks back. |
| `presence_penalty` | -2 to 2 | Additive penalty on tokens present in the last *n*. |
| `presence_context_size` | integer ≥ 1 (20) | The *n* the presence penalty looks back. |
| `frequency_penalty` | -2 to 2 | Additive penalty in proportion to how often a token appeared in the last *n*. |
| `frequency_context_size` | integer ≥ 1 (20) | The *n* the frequency penalty looks back. |
| `logit_bias` | object, token id → -100 to 100 | Added to the logits of those ids at every step. |
| `logprobs` | boolean | Log-probabilities per token. |
| `top_logprobs` | integer 0-20 | The most likely alternatives per token; needs `logprobs: true`. |
| `stop` | string or list of strings | <span class="hint" tabindex="0" data-tip="Sequences that end the answer; the model&#x27;s eos ids always do.">Sequences that end the answer</span> |
| `tools` | list of function tools | Tools the template renders and the answer is parsed for. |
| `tool_choice` | `auto` or `none` | <span class="hint" tabindex="0" data-tip="none sends no tools; other values are refused.">`none` sends no tools</span> |
| `stream` | boolean | Server-sent events. |
| `stream_options.include_usage` | boolean | With `stream`, the last event carries `usage`. |

Refused with an error: `n` above 1, `best_of`, `suffix`, `response_format` other than `text`, and stored state on `/v1/responses` (`previous_response_id`, `conversation`). Structured output is an extra.

## Checkpoint: what the model ships

A standard MLX checkpoint can carry settings of its own:

- **`generation_config.json`** - its sampling values become the server's defaults, below the flags and above mlx-lm's own; `do_sample: false` means greedy.
- **The chat template** - tool-call markers are inferred from the template that renders, think markers from the vocabulary. `--chat-template` replaces it; `--use-default-chat-template` gives a model without one plain ChatML.
- **A KV profile** - bits per layer, as a list or an object, read only through `--kv-config`. Listed layers take their bits, the rest follow `--kv-bits`. The prefill mode comes from the profile, else from the mode the package was measured with (the manifest's file, SHA-256 checked); `--kv-prefill` beats both.
- **A draft head** - named by the manifest (`parts.mtp`, SHA-256 checked), else by `mtp_file` in `config.json`, else found in the shards (Qwen's `mtp.*` tensors, GLM-4.7 / DeepSeek's NextN layer). Used only with `--draft-model bundled`; a checkpoint that bundles one says so when started without it. Its bits come from the manifest, else `config.json`, else it is quantized to 4 bits at load.
- **A package manifest** - `extras/manifest.json` in the B.E.A.M. package layout names the draft head and the KV profile, each with file and SHA-256. A plain checkpoint has none.

`/health` shows what was built from all of this: the KV layout per layer, batching and cache settings, the draft head and its counters, and every default with its source.
