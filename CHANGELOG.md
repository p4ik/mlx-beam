# Changelog

All notable changes to mlx-beam. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
PEP 440 with SemVer meaning (`0.y` may break, `0.y.z` fixes).

## [Unreleased]

### Fixed
- The same prompt sent again resumes from the stored entry instead of
  prefilling from scratch on hybrid models: the prompt's checkpoint is taken
  before its last token, which is the token a lookup keeps to prefill.
- A conversation is one store entry, not one per turn: when a request
  continues a stored entry, that point is checkpointed, so the finished
  turn can replace the entry it grew from.
- `max_context` (and the vocabulary size) are read through multimodal
  wrapper configs (`text_config`, the nested language model), so a
  Qwen3.5-class model reports its window instead of `null`.

### Added
- `--kv-config` also takes the list a quantized package ships
  (`[{"layer_idx": 3, "bits": 4, "group_size": 64}, ...]`): the listed
  layers override `--kv-bits`, the rest follow it.

## [0.1.0a1] - 2026-09-18

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

### Fixed
- A tool call the parser cannot read is returned as text, markers and all,
  and `finish_reason` says what really happened (`stop` / `length`) instead
  of `tool_calls` with no call. Calls before a cut-off one are kept by
  parsing the longest prefix that parses, so a `[TOOL_CALLS]` quoted inside
  a JSON argument never becomes a call of its own, and the Mistral header
  parser no longer reads a `name[ARGS]` out of a string argument of a
  cut-off JSON list. A parser result JSON cannot carry (a set from a
  literal) is an unreadable call, not a 500.
- Qwen tool-call parameters end at their last `</parameter>`, so a single
  literal end tag in a value survives; a parameter name seen twice, one a
  schema with `additionalProperties: false` rules out, or text left between
  two parameters by a literal tag is an error the caller sees instead of an
  argument silently overwritten or shortened. The
  vendored Qwen and Mistral parsers carry upstream's fixes for a parameter
  name without `>` and for the JSON list form (see `VENDORED.md`).
- A stream sends an SSE comment whenever nothing went out for five seconds
  while the worker made progress, not only during the prefill: a tool call
  is collected until it closes, so a long one used to be decoded in silence
  and clients with an idle timeout gave the request up in the middle of it.
  A worker that stopped stepping gets no comment, before the first token as
  well as after it, so a watchdog in front of the server still sees the
  hang.
- `/v1/completions` returns what the model wrote: think markers and
  tool-call blocks stay in the text; they were parsed away. Chat parses a
  tool-call block only when the request offered tools.
- Cancelling a request that was admitted with a one-token prompt in the
  same round, before its own prefill began, killed the worker and every
  other request with it (`extract()` on an empty cache).
- Text the detokenizer or the marker automaton still held when the stop
  token came is no longer lost; a real U+FFFD the model wrote before a
  forced close is kept, only a byte fragment the cut left is dropped - and
  only when the detokenizer shows its bytes (BPE, SPM); without that
  evidence nothing is dropped.
- A think-block end marker the model began on the arming token and
  continued on the next is completed instead of answered with a forced
  newline in place of the first answer token (multi-token markers). A
  marker whose prefix repeats is matched.
- With the block open and no room to think, the close costs its whole
  length; a `min_response_tokens` that then cannot be kept is a 400, not a
  shorter answer.
- Prefix store: system entries are capped at a quarter of the store, so
  many distinct system prompts cannot crowd every conversation out; an
  entry cut from a longer one owns only its own tokens instead of pinning
  the whole conversation's buffers; a cancelled request's recurrent state is
  stored as its own bytes, not as a view into the batch.
- Scheduler: a long prefill can no longer be starved by a trickle of short
  prompts. The prefill width is the shortest row's segment (nobody is
  padded); after two calls in which newcomers held a row with a whole slice
  to go under a quarter slice, the next call admits nobody and that row gets
  its full width. A single short request beside a long prefill is served
  exactly as before; `/health` counts the guard's calls as
  `prefill_starved_calls`.
- Quantized KV on models whose layers share a cache (Gemma 4) and on
  PLaMo-2, which called MLX's attention directly: both died at warm-up with
  a `TypeError`. A KV group size the model's head dim cannot carry is
  refused with a message that names the policy and the sizes that fit. The
  engine's worker thread evaluates every lazy array of the model, not only
  its parameters: Gemma 4's and Llama 3's rope tables are lazy and killed
  the worker with "There is no Stream(cpu, 0) in current thread".
- A decode burst under a shared prefill ends when a row finishes, so its
  extracted cache is evaluated before the next step instead of forcing a
  copy of the whole batch KV every step meanwhile.
- Responses API: every output item has its own id and text when text and
  tool calls interleave; `output_text.*` carry `logprobs`,
  `function_call_arguments.done` carries `name`, a cut-off message item is
  `incomplete` from its done event on (also the cut-off tail next to a
  call that did parse, which goes out after that call), `instructions` and
  `tool_choice` are
  echoed, and a `reasoning` input item reaches the template as the
  reasoning of the turn it preceded (a badly shaped one is a 400).
- Wrongly shaped request fields (`response_format`, `stream_options`,
  `chat_template_kwargs`, `metadata`, `text`, a content part's `text`,
  `stream`/`echo`/`logprobs` that are not booleans, `n: true`) are 400s,
  not 500s; `logit_bias` values must be finite and within -100..100;
  `suffix` is refused as unsupported; a server default a request could not
  ask for (`generation_config.json` with `temperature: 3.0`, a flag out of
  range) fails at start instead of on every request.
- HTTP: a failed stream write is not retried, so a dead client releases its
  batch slot after one timeout; the CORS preflight allows the headers the
  browser asked for (the OpenAI SDK sends `x-stainless-*`); a negative
  `Content-Length`, a body that is not UTF-8 and a chunked body are 400/411;
  HEAD, PUT, DELETE and PATCH are 405 with a JSON body; `Retry-After` only
  on `queue_full`, since a dead engine does not come back.
- `--chat-template` with the template's text (longer than a file name may
  be) no longer crashes at start; `--kv-config` and `--log-level` are
  checked before the model loads; `--decode-share` must be within 0..1;
  `--max-context` above the model's own window is capped to it with a
  warning; a temperature below 1e-4 is greedy, since `1/temp` overflowed
  float32 and the draw turned random.
- Chat `logprobs.content` covers the message content only, not the think
  block, the markers or the stop token; completions streaming puts `usage`
  on a last chunk without choices, as at OpenAI.
- Tool-call arguments in the conversation reach the chat template as a
  mapping (the wire carries a JSON string; `""` means no arguments), and a
  `null` content is `""` there. The Qwen3.8 template raises on a string, so
  the second step of every tool loop used to fail; an argument string that
  is not a JSON object is a 400.

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
