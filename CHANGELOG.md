# Changelog

All notable changes to mlx-beam. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
PEP 440 with SemVer meaning (`0.y` may break, `0.y.z` fixes).

## [Unreleased]

### Added
- Two marker families: Harmony (gpt-oss) and Muse (ATEM). The reasoning,
  the answer and a tool call sit behind a label - the channel, or the
  recipient after `<|start|>assistant` - and the text assembler routes by
  what the label says: `analysis` / `self` to the reasoning field, `final`
  / `user` to the content, a recipient naming a tool to a tool call. Tool
  parsers for both (`harmony`: the JSON body of a commentary message;
  `atem`: the `<atem:invoke>` block, values typed by the tool's schema).
  A thinking budget closes the block the way these families do - by
  opening the answer - and forces the whole sequence; `enable_thinking:
  false` opens the answer in the prompt, since their templates have no
  switch. A recipient marker (` to=`) is read only where the family puts
  one - right after `<|start|>assistant` - so the same characters inside
  an answer stay text. `/health.reasoning` names the family and its
  markers, `/health.tools` the parser.
- The reasoning effort follows what the loaded template does with it,
  measured at load: the kwarg it reads (`reasoning_effort`,
  `reasoning_strength`, ...), whether it checks the word, and the set it
  accepts, reported under `/health.reasoning.effort` and in each model's
  `capabilities.effort` in `/v1/models`. A template that checks gets the
  client's word when it is in the set, else the nearest rung it accepts;
  one that takes the word unchecked gets it as sent (the vocabulary such
  models were trained on); one without the kwarg gets nothing, and one
  whose kwarg takes a number (a token budget) gets no word either
  (`capabilities.effort.takes: tokens`). Before, every word was mapped
  onto one template's three names.
- Anthropic's Messages API: `/v1/messages` (system field, content blocks
  with text, thinking, tool_use and tool_result, tools with
  `input_schema`, `tool_choice` auto or none, `stop_sequences`,
  `thinking.budget_tokens`, `top_k`) answering in content blocks in the
  order thinking, text, tool_use with Anthropic's stop reasons, usage
  (`cache_read_input_tokens` from the prefix cache) and stream events
  (`message_start`, per block start / delta / stop, `message_delta`,
  `message_stop`, `ping`), errors in Anthropic's envelope; and
  `/v1/messages/count_tokens`. Tool arguments go out in one
  `input_json_delta` when the call closes. A `cache_control` marker on
  any block is accepted and does nothing: the prefix store caches every
  turn by itself, and a client that marks blocks is not refused for it.
  `input_tokens` in the usage counts what was prefilled, the cached part
  excluded, so the two add up to the prompt. A `stop_sequence` stop
  reason names the sequence that ended the text.
- A repair ladder for tool calls: the parser's own reading, validated
  against the tool's declared schema; when the parser refuses, a reading
  of the text with the usual defects of model JSON mended (a code fence,
  Python's literals, single-quoted strings read with their escapes so an
  apostrophe inside stays one, a trailing comma, braces left open - in
  the syntax only, a string's content is never touched); when the
  schema objects, values coerced where they plainly are the
  declared type written another way ("42" for an integer, "true" for a
  boolean, a JSON text for an object). Every rung is validated again,
  nothing is invented, what a rung did is the call's `repair_actions`,
  and a call no rung makes valid still comes back as text. The mending
  applies to parsers that read JSON and to blocks the model closed: a
  block cut off by `max_tokens` is not completed into a call the model
  never made. A call to a tool the request did not declare is text, not
  a call. `/health.tools.repairs` counts the rungs.
- `/metrics` in Prometheus' text format from the same counters `/health`
  reports (`mlx_beam_*`: requests, tokens, seconds, memory, the prefix
  store, tool-call rungs, the speculator), plus the four `vllm:` names
  dashboards read most.
- `/v1/models` entries carry `context_length` and `max_model_len` (the
  context the engine serves), `max_completion_tokens` (the default a
  request may override), `input_modalities`, `owned_by: mlx-beam` and the
  `capabilities` object with the effort measurement; chat and text
  completions carry a `system_fingerprint` (model, version and KV layout
  hashed).
- A prefill valve. The next prefill call's peak is estimated from the
  bytes per token the calls before cost on this machine (learned from the
  allocator's peak) and held under the smaller of the device's
  recommended working set and what the process holds plus what the
  system has free - dynamic, for a memory other engines share. A call
  that would not fit is cut to the width that does (on a grid of 64
  tokens); below the narrowest width the round prefills nobody and
  decoding goes on, instead of Metal failing with an error nothing can
  catch. A stall that cannot end by itself - nothing decodes, so nothing
  will free memory - first evicts the prefix store one entry at a time,
  then refuses the widest waiting prompt with a 400 (it does not fit
  beside what runs) instead of holding every prompt behind it for ever.
  `/health.prefill_valve` reports the estimate, the ceiling, the calls
  cut, the rounds stalled and the prompts refused. Thresholds and the
  estimate's accuracy are to be measured on the Mac.
- A checkpoint in the package layout is read through its manifest:
  `config.json` names it (`extras.manifest`), `parts.mtp` names the draft
  head's file, bits, group size and norm convention, and the file is
  checked against the manifest's SHA-256 before it is loaded - a head whose
  bytes differ from what the package was measured with is refused. A
  plain checkpoint keeps the old paths (`mtp_file`, `mtp.*` in the shards)
  and the loader's defaults.
- The KV prefill mode a package measured for its own profile
  (`parts.kv_config.prefill.mode` in the manifest) applies when
  `--kv-config` names that file: the same bytes as the manifest's SHA-256
  when it has one, else the same path; the profile's own `prefill` and
  `--kv-prefill` still win. `/health.kv.prefill_source` says `manifest`
  when it came from there.
- The site's pages share one navigation; the current page is marked and
  links that leave the site say so.

### Changed
- The `forced` flag of a token now covers the whole forced close, the part
  after the end marker included (a line break, or the answer's opener).
- The thinking budget of a label family (Harmony, Muse) counts from the
  label, not the opener: `<|start|>assistant<|channel|>final` is the
  answer, and only a label the family names as reasoning starts the
  budget.
- A request's `reasoning_effort` word lifts a server-side
  `enable_thinking: false` when the request says nothing about thinking
  itself (the client asked for effort, so it wants the reasoning);
  `reasoning_effort: null` in a request clears the server's default word.
  Where a request and a server default disagree, the request wins.
- `/metrics` writes a metric's `# HELP` and `# TYPE` once, before its
  first sample, as the exposition format requires; a metric with a
  label per value used to repeat the pair, which its readers reject.
- `--prompt-cache-bytes` is the store's own limit: the caches of running
  requests no longer count against it, so a parallel request cannot push a
  stored conversation out. The budget bounds what the store holds, nothing
  else.
- The README and the site describe what is built in the present tense and
  name what is planned as planned; vision and audio will be a package of
  their own, structured output and GGUF extras with a guard - none of them
  in this release, and the `extra_not_installed` message says so instead
  of suggesting an install line that would do nothing.
- Versions between tags count towards the release they are heading for
  (`0.1.0a5.devN` after `v0.1.0a4`) instead of the next minor; the rolling
  `dev` GitHub release follows every merge to `main`, titled with version
  and date; a tag is refused without its changelog section.
- `--max-context` help text: only a prompt whose reserve does not fit is a
  400, a larger `max_tokens` is served capped.

## [0.1.0a4] - 2026-09-25

### Added
- Speculative decoding, first stage: `--draft-model bundled` loads the
  draft head a checkpoint ships (the `mtp_file` its config names, or the
  `mtp.*` tensors in its shards) and verifies its drafts in one forward
  per cycle. Greedy requests decoding alone get up to three drafts per
  cycle, four tokens with the one the model samples after them
  (`--max-draft-tokens` caps the drafts); everything else - several rows
  at once, sampling, a request with repetition penalties, a thinking
  budget about to act - decodes plainly through the same path; a logit
  bias is applied to every verified position, so a request with one still
  speculates. Every committed token is the model's own argmax over the
  verify forward; a rejected draft is taken back from the attention caches
  by a trim and from the recurrent layers by redoing their recurrence over
  the accepted prefix. Against plain decoding that is the same output up
  to the rounding of kernels running at another width (bit-identical in
  bf16 in every measured case, 2026-09-24; in float32 a logit tie can fall
  the other way). Without the flag nothing changes; a checkpoint that
  bundles a head says so at start.
  `/health.speculative` reports the proposer, the depth, cycles, drafted
  and accepted tokens and the plain steps taken instead. Measured on a
  27B hybrid with 8-bit KV, one request: 2.1x tokens per second at depth 3
  (2026-09-19, M4 Pro).
- The head's RMSNorm weights are all shifted by +1 at load, the (1 + w)
  convention of the checkpoint against mlx's w; a per-tensor guess by the
  mean leaves three of the seven unshifted and costs ~14 points of draft
  acceptance.
- A configuration page on the site: the `beam serve` flags as tables
  generated from the parser (CI fails when the page is stale), the request
  fields with their ranges, what a checkpoint may bring along, and the
  order in which flag, checkpoint and request win. `--host`, `--port` and
  `--log-level` gained the help text they lacked.

### Changed
- The tiled quantized attention runs from 64 query tokens on rather than
  from two: a speculative verify of four tokens paid +40 ms per step at
  32k context on the tiles (2026-09-19, M4 Pro, Qwen3.8-27B, 8-bit KV).

## [0.1.0a3] - 2026-09-19

### Added
- `/health` reports `memory: {active, peak, cache}` in bytes from the Metal
  allocator (`mx.get_active_memory`, `get_peak_memory`, `get_cache_memory`).
  Process RSS does not include these buffers, so the full-precision transient
  of an exact prefill was invisible to anything watching `ps`. `POST
  /health/reset-peak` starts a fresh peak window and returns the counters as
  they were, so a test can read one phase's peak rather than the process's;
  mlx zeroes the peak on reset, so it is 0 until the next allocation.
- `--kv-prefill exact|quantized` (default `exact`): a quantized layer is
  kept at model precision while the prompt is prefilled and quantized once
  at the move to decoding, so the prefill never reads quantized data. This
  is what mlx-lm's `generate_step` computes with `quantized_kv_start` at
  the prompt's end (its default of 5000 leaves shorter prompts unquantized
  altogether; a start of 0 quantizes after each prefill chunk). `quantized`
  writes the cache quantized from the first token, the batch path's former
  only behaviour: it saves the prompt's full-precision transient and on
  some models costs accuracy (Gemma 4 at 8 bit: KL 0.022 against 0.0005).
  A `kv_config` object may carry `"prefill": "quantized"` for a profile
  measured with it; the flag beats the profile; `/health` shows the value
  and where it came from. A prefix restored from the prompt store keeps its
  quantized codes across the round trip.

### Fixed
- Gemma 4's think channel is recognised whatever token its label comes
  as. After a tool response the model opens the channel itself and writes
  the label as ` thought` rather than the template's `thought`; the fixed
  two-token opener never matched and `<|channel> thought\n<channel|>`
  leaked into the content. The opener is now the channel marker alone and
  the label up to the line end is part of it, as the template's own
  `strip_thinking` treats it. The reasoning budget counts that marker
  like any single-token opener, so `reasoning_tokens` on Gemma 4 is one
  higher per block than before.

## [0.1.0a2] - 2026-09-18

### Fixed
- The same prompt sent again resumes from the stored entry instead of
  prefilling from scratch on hybrid models: the prompt's checkpoint is taken
  before its last token, which is the token a lookup keeps to prefill.
- A conversation is one store entry, not one per turn: when a finished
  turn extends a stored entry, it inherits that entry's checkpoints and
  its recurrent end state, then replaces it. What is freed is the old
  entry's KV cache and its place in the entry count; every position the
  old entry could restore, the new one can. A request restored on an
  entry's end carries that state along, so it keeps its restore point
  even if the entry is replaced or evicted before the request finishes.
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
