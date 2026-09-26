# Changelog

All notable changes to mlx-beam. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
PEP 440 with SemVer meaning (`0.y` may break, `0.y.z` fixes).

## [Unreleased]

### Added
- Access control that follows the bind: on loopback nothing is asked; on
  any other `--host` the server starts only with `--api-key <key>`
  (`Authorization: Bearer` or `x-api-key`, `/health` and `/metrics`
  included, 401 otherwise) or `--skip-api-key`, and warns at start that it
  is reachable. The `Host` header must be `localhost`, the bind address or
  one of `--allowed-hosts`, else 403 - a browser's DNS rebinding sends
  another. `/health.api.auth` reports the mode and the hosts.

### Changed
- CORS admits no origin by default; `--allowed-origins` names the pages
  that may call the server (was `*`, any page in a browser could read the
  answers of a server on loopback).

### Fixed
- Muse: the reasoning went into the answer, with `to=self` leaking in
  front of it. The detokenizer drops the space a sequence starts with, so
  the frame's opener ` to=` arrived as `to=`; and the vocabulary merges
  `=self` into one token, so the budget's token form of the opener never
  matched. The frame now takes the opener without its space, and the
  opener's token form is read from the encoding of opener plus label.
- A tool call whose `arguments` are a JSON string (the wire format's own
  shape, which Granite writes inside its block) was refused as "not an
  object"; it is decoded first, reported as `arguments decoded`.
- Harmony and Muse with thinking off and tools declared: the answer's
  opener was forced into the prompt, which skips the channel or recipient
  a tool call needs, so the model could only answer in text. With tools on
  offer the opener stays out and the reasoning budget is set to zero
  instead, which masks the reasoning label alone: the model may call a
  tool, not think.
- A message role outside the format (`system`, `developer`, `user`,
  `assistant`, `tool`) was handed to the template, which rendered it as
  a foreign frame or failed; it is refused with 400 naming the roles.
- `developer` was always rewritten to `system`. A template that knows the
  role (gpt-oss) now gets it as sent; one that would only write the word
  into its frame gets `system` - measured once at load,
  `/health.template.roles.developer`.
- A connection that failed while the body was read (an aborted socket)
  raised out of the handler thread after a 500 was written into the
  fault; it is closed quietly.
- `HEAD`, `PUT`, `DELETE` and `PATCH` answered 405 with the error type
  `not_found_error`; now `invalid_request_error` with the code
  `method_not_allowed`. Responses: `background: true` (a stored response
  polled later) is refused as unsupported instead of ignored.
- A reasoning budget of zero (or one used up) masked Harmony's shared
  `<|channel|>` marker, which the answer's and a tool's channel need as
  much as the reasoning; the mask now cuts the reasoning label after the
  marker, so `final` and `commentary` stay open.
- `mlx-beam-vision` claimed vision for a checkpoint without
  `preprocessor_config.json` (AutoProcessor hands a bare tokenizer back)
  and every image request failed with a KeyError; such a checkpoint is
  refused at load with the reason.
- A dense Granite checkpoint quantized under the HF names (mlx-vlm's
  conversions, the Granite Vision 8-bit packs) failed to load: only the
  weights of `shared_mlp` and `lm_head` were renamed or dropped, their
  scales and biases stayed behind as "parameters not in model". The two
  projections are renamed independently, so a checkpoint that quantizes
  one of them loads too.
- An emptied generation batch kept the current tokens of its last step,
  and every batch extended onto it concatenated its own: cleared when the
  batch empties.
- Images inside tool results reach the vision frontend: parts in a
  Responses `function_call_output.output` were serialized to JSON text,
  image blocks in a Messages `tool_result.content` were refused with 400.
- An image span ending at the last prompt token is refused before
  admission: that token is fed by the first generation step, which knows
  no image features, so the image would have been read as its
  placeholder's embedding.

## [0.1.0a6] - 2026-09-25

### Fixed
- `mlx-beam-vision` installs torch and torchvision. The processors
  transformers ships for its families need them, so a clean install
  refused every vision checkpoint with "requires the Torchvision library"
  and served no image.

## [0.1.0a5] - 2026-09-25

### Added
- One tiny model per architecture class in the test suite - attention
  sinks (gpt-oss), sliding window with NoPE full layers (Muse), MLA
  (GLM-4.7), Mamba-2 + attention (Granite 4), plain attention behind a
  vision wrapper (Mistral 3) - each pushed through batched decode, the KV
  policy, the prefix store across a boundary and the speculative verify
  (`tests/test_model_classes.py`).
- The KV policy checks what a layer can carry before the engine starts:
  attention sinks have no quantized SDPA, an MLA attention reads its
  latent projection back from the cache as an array. Bits on such a layer
  are refused at start with the reason, instead of a worker that dies at
  the first decode step; `/health.kv` lists `quantizable_layers` and the
  `exceptions` with their reason (sliding-window and recurrent layers among
  them).
- Sliding-window layers under the prefix store and the speculative verify.
  A rotated ring cannot trim (its stale tail would stay), so a boundary
  checkpoint now holds the window's state next to the recurrent state and
  a request that shares a system prompt longer than the window restores
  it from there; the verify restores the state before a cycle and writes
  the accepted tokens again. Before, a store entry longer than the window
  was usable only as an exact match, and a full window ended the
  speculative path with an error that took the engine down.
- `--exact-verify`: `positions` runs the verify one token per forward and
  stops at the first rejected draft - exact by construction at plain
  decoding's cost, the reference; `kernels` runs the block forward through
  projections and attention that keep single-row arithmetic (kernels
  vendored from mlx-vlm, MIT) and keeps them only when the warm-up finds
  them bit-equal to single forwards on this machine, else falls back to
  the block verify and says so. In every mode `/health.speculative.exact`
  reports the warm-up width check: does the block forward give the same
  logits as single forwards here (`block_equals_positions`,
  `max_abs_logit_diff`, `argmax_equal`), for every width a cycle can run.
  `positions` checks the row's stop and length limits on each token
  before it feeds the next, so nothing past the cut enters the caches,
  and hands the penalties the tokens fed earlier in the cycle, as a plain
  step would. On Metal (mlx 0.32.2) the kernel path does not pass its own
  check yet and the fallback is what runs; `positions` is the mode that
  is exact there.
- The speculative verify on Mamba-2 hybrids (Granite 4): the layer stashes
  what a partial rollback needs, the rollback replays the selective scan
  over the accepted prefix, bit for bit against a forward of those tokens.
- Two marker families: Harmony (gpt-oss) and Muse (ATEM). The reasoning,
  the answer and a tool call sit behind a label - the channel, or the
  recipient after `<|start|>assistant` - and the text assembler routes by
  what the label says: `analysis` / `self` to the reasoning field, `final`
  / `user` to the content, a recipient naming a tool to a tool call. Tool
  parsers for both (`harmony`: the JSON body of a commentary message;
  `atem`: the `<atem:invoke>` block, values typed by the tool's schema).
  A thinking budget counts from the label the family names as reasoning
  (the answer's channel opens no block) and closes the block the way
  these families do - by opening the answer - forcing the whole sequence; `enable_thinking:
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
  with text, image - a base64 source, served like a chat image part -,
  thinking, tool_use and tool_result, tools with `input_schema`,
  `tool_choice` auto or none, `stop_sequences`, `thinking.budget_tokens`,
  `top_k`) answering in content blocks in the
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
  dashboards read most; each metric's `# HELP` and `# TYPE` once, as the
  exposition format requires.
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
- Speculative decoding for sampled requests. The target draws every
  verify position by Gumbel-max under a key per position, the draft head
  drafts under the same keys, and a draft is accepted when it equals the
  target's draw: no residual distribution, no draft probabilities, and
  the committed token has the target's own distribution whatever was
  drafted (chi-square over 2048 seeds against a draft that is always
  wrong; every possible first draft enumerated against the plain
  transcript). A seeded request gives the same tokens with and without
  the draft head. Logit bias, repetition, presence and frequency
  penalties apply per verify position over the context that position
  would have seen; a thinking budget speculates whenever its state cannot
  change within the cycle, the opener mask included - only the forced
  close itself runs in plain steps.
- The draft head keeps its history over the whole conversation. The
  prefill runs the trunk without the `lm_head` over each prompt chunk
  (its logits were discarded before) and feeds the hidden states to the
  head as pairs with the next token; every plain step queues its pair, so
  a row that decoded beside another for a while still drafts with a whole
  history when it is alone again; the history at each message boundary
  and at the end of a row is stored beside the prefix-cache entry, and a
  conversation that continues, or a request that shares the system block,
  resumes it from there. Prompts longer than 8192 tokens are primed from
  their last 8192 positions on, and a stored history longer than that is
  cut to its last 8192 pairs on restore with their rotary positions
  rewound, as a head that started at the window would hold them - the
  rotated rows of the head's own layout (MLA keeps them in the values),
  moved by the module's rotation alone (a YaRN amplitude is not applied
  twice); a head whose rotation cannot be rewound keeps its history whole. `/health.speculative.proposer` counts
  `primed_pairs` and `histories_restored`. Measured before on the 27B
  head: 2.39 against 2.23 tokens per cycle with and without the prompt
  primed.
- A depth regulator. Each cycle's draft depth is chosen below
  `--max-draft-tokens` from what was measured on the machine: the
  acceptance at each chain position and the wall time of a cycle at each
  depth against the wall time of a plain step (measured, not assumed);
  the depth with the best expected tokens per time runs, a neighbour is
  probed now and then. A cycle that commits fewer tokens than its cost in
  plain steps is a loss; sixteen in a row park the proposer for a
  cooldown of plain steps that doubles each time (128 to 4096) until a
  cycle wins again. The plain step is measured again every 64 cycles, in
  the context the row is in by then, and the regulator starts over after
  the warm-up (whose cycles ran cold over a few tokens); a cycle the row
  was not eligible for does not count, the acceptance is learned from the
  drafts actually made. `/health.speculative` reports the current `depth`
  (the cap as `max_depth`) and under `regulator` the acceptance by
  position, the costs, the rates, the tokens saved, `parked` with the
  reason and the cooldown left.
- A second draft-head form: the NextN layer of GLM-4.7 / DeepSeek-V3
  (`enorm`, `hnorm`, `eh_proj`, one full decoder layer with its MoE,
  `shared_head`), read from `mtp/weights.safetensors` or the shards under
  the checkpoint's own names through the trunk's sanitize (experts
  stacked, latent projection split), with its own embedding and output
  head when the checkpoint ships them. The tensors say the form; the
  manifest's `parts.mtp.form` may confirm it, not contradict it.
  `/health.speculative.proposer.form` names it. Exercised against a tiny
  GLM with a random head; acceptance on the real package is still to be
  measured.
- The core's side of images. A request carries image spans - placeholder
  positions with the features a frontend computed, and per-layer extras
  for towers that add them after the first text layers (DeepStack) - and
  the prefill embeds them in place of the placeholders' vocabulary
  embeddings, chunk by chunk, beside text-only rows. The prefix cache
  keys on the image's digest, so the same placeholders with another
  image never meet; a stride checkpoint never lands inside an image.
  Chat and Responses requests take OpenAI's `image_url` / `input_image`
  parts as `data:` URLs (nothing is fetched; a base64 payload may be
  wrapped in lines); an image request goes through the same effort
  ladder and template checks as text, and a processor's refusal is the
  client's 400. The frontend says where the assistant's own turn begins
  in the prompt, so a think marker inside a user message beside an image
  is text, not an open block. What serves the images is a separate package found
  through the entry-point group `mlx_beam.modalities`; without one the
  core answers image input with a 400 that says what is missing and
  points at `/health.vision`. A frontend states what it needs of the
  text model (input embeddings; the layer hook for per-layer extras) -
  before its tower is read, when the provider can tell from the config -
  and a frontend the model cannot serve is refused with the reason in
  `/health.vision`; an image request the model cannot serve is a 400,
  never a dead worker. `--chat-template` and `--trust-remote-code` reach
  the frontend's processor. `/health.vision` and `capabilities.vision`
  say what was found.
- `mlx-beam-vision`, the first package beside the core
  (`packages/mlx-beam-vision`, its own project on PyPI, installed by
  `mlx-beam[vision]`): five tower families vendored from mlx-vlm (MIT) -
  Qwen3-VL (which Qwen3.5 and Qwen3.8 carry unchanged, DeepStack
  included), Pixtral for Mistral 3 (an image's rows as separate spans;
  the HF layout and mlx-vlm's conversions both load), Gemma 4 E-series
  (image; the processor's patchified inputs with their patch positions),
  Muse Glimmer, Granite Vision 4.1 (AnyRes tiles, window Q-Former
  projectors adding features ahead of their text layers, the tiles a
  processor pads onto an image left out, the unpadding rounded as the
  reference does; its checkpoints load as text through a
  `granite4_vision` model class, the dense or the Mamba-2 hybrid text
  model by the config) - each with its projector after the reference
  implementation; the checkpoint's own processor through transformers
  (5.15 or later) renders the template and expands the placeholders, its
  BOS not added twice; the tower's tensors come from the checkpoint's
  index (a package's own vision file included) and `/health.vision`
  names the shards; encoder outputs are cached by image digest, bounded
  in bytes, only the images the cache lacks go through the tower; an
  image above 32 megapixels is refused before it is decoded, EXIF
  orientation is applied. Gemma 4 12B (`gemma4_unified`, encoder-free,
  its image tokens attending bidirectionally in the text model) is
  refused with that reason. The workspace builds and tests both
  packages, the release ships both wheels at the same version, and
  import-linter keeps the core from importing the package. Exercised on
  tiny towers against a direct forward; the processors and the real
  towers are a Mac round. Not yet: audio, video, quantized towers.

### Changed
- A seeded request draws by Gumbel-max under a key derived from the seed
  and the position of the token, no longer from a key chain advanced per
  step. Same seed, prompt and settings still give the same tokens; the
  tokens themselves differ from earlier versions for the same seed.
- Dead weight out: an unused recurrent snapshot helper, a duplicate of the
  cache-array walk, store statistics fields nothing set, a speculator flag
  nothing read, a test-only wrapper around the boundary finder; the think
  markers of a request are read once instead of up to three times. A fixed
  request series against the tiny model gives the same bytes before and
  after (golden transcript), the suite is unchanged.
- The `forced` flag of a token now covers the whole forced close, the part
  after the end marker included (a line break, or the answer's opener).
- A request's `reasoning_effort` word lifts a server-side
  `enable_thinking: false` when the request says nothing about thinking
  itself (the client asked for effort, so it wants the reasoning);
  `reasoning_effort: null` in a request clears the server's default word.
  Where a request and a server default disagree, the request wins.
- `--prompt-cache-bytes` is the store's own limit: the caches of running
  requests no longer count against it, so a parallel request cannot push a
  stored conversation out. The budget bounds what the store holds, nothing
  else.
- The README and the site describe what is built in the present tense and
  name what is planned as planned: structured output and GGUF come as
  extras with a guard, audio joins the vision package - and the
  `extra_not_installed` message says so instead of suggesting an install
  line that would do nothing.
- Versions between tags count towards the release they are heading for
  (`0.1.0a5.devN` after `v0.1.0a4`) instead of the next minor; the rolling
  `dev` GitHub release follows every merge to `main`, titled with version
  and date; a tag is refused without its changelog section.
- `--max-context` help text: only a prompt whose reserve does not fit is a
  400, a larger `max_tokens` is served capped.
- The test suite asserts bit-equality with plain decoding only where the
  machine's own width probe finds the block forward bit-equal to single
  steps (the CPU); on Metal the block verify's tokens are the kernels' and
  the tests check the run, not the transcript. The per-position mode stays
  exact everywhere.

### Fixed
- A proposer or sampler that fails at admission fails that request, not
  the worker.
- The reasoning budget sees every token before the next decode step. While
  a prefill shared the worker the generator ran several steps per call and
  the budget observed them afterwards, so a forced close armed late and the
  block overran its cap (up to a prefill slice's worth of tokens); the
  engine now observes each step from inside the generator (`on_step`).
- A response that is not streamed notices a client that went away: the
  socket is peeked once a second and the row cancelled, instead of
  decoding to `max_tokens` for nobody.
- `logprobs.content` books a token whose text the automaton held back (the
  start of a possible stop word or marker) with the token that releases
  it; the eos token is never an entry, whatever its arrival flushed. Byte
  tokens carry their own bytes, not those of U+FFFD - a SentencePiece
  byte token (`<0xE2>`) its hex, a BPE one its byte alphabet.
- `/v1/completions` no longer repeats a think opener the raw prompt ends
  with; the reasoning state is seeded from the assistant's own frame only,
  so a `<think>` or a recipient label inside a user message is text.
- The warm-up runs a short prompt through the prefill (three tokens plus
  a decode step), so a KV policy or a prefill path the model cannot carry
  fails at start, as promised, not on the first request.
- A prefix-store entry that cannot be written costs the entry, not the
  engine: the answer is delivered, `/health.prompt_cache.store_failures`
  counts it.
- The speculative batch decodes every row through the trunk's halves; a
  family that scales or softcaps its logits after the projection (Granite,
  Gemma, Muse) now gets that post-processing there too, and the warm-up
  checks the composition against the model's own forward bit for bit.
- A Hugging Face repo id downloads a package's draft head and tower
  sidecars (`mtp/`, `optiq/`): a second pass fetches every file the
  checkpoint's config, manifest and weight index name beyond the default
  patterns.
- `beam serve` checks its flags before the load: the KV policy, the
  template file (a path that names no file is refused instead of rendering
  the path as the template), `--draft-model`, the request defaults and the
  port; every refusal to start is exit code 3, argparse's own stay 2.
  Count flags refuse 0 and negative values. `--trust-remote-code` reaches
  the tokenizer. `--kv-bits` and `--kv-group-size` beat the checkpoint's
  kv_config, as the help text said; the group size's default moved into
  the resolution so a file's value applies only when the flag is unset.
- Chat: `developer` is `system`; `tool_calls` must be a list (400, not
  500); `max_completion_tokens: null` does not hide `max_tokens`;
  `repetition_penalty` must be positive; `tools: []` is no tools.
- A stop sequence of more than one token no longer leaves its start in
  the answer: the sequence's last token goes through the detokenizer so
  the text-level match completes and cuts there (only an eos token is
  dropped unseen). A prefix before an eos, or cut by the length limit, is
  still text.
- `logprobs.content` keeps every token of a multi-byte character: byte
  tokens the detokenizer holds until the character completes are booked
  with the token that completes it, instead of being dropped as
  non-content.

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
