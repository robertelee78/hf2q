# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Increase DeepSeek-V4's default pure-decode slot quantum from 8 to 64 after
  the cold-prefill barrier lifts, amortizing session swaps and scheduler
  publication across saturated cohorts. Genuinely mixed prefill/decode work
  remains clamped to the measured eight-token/two-window interactive budget.
  The intermediate 16-token exact-main candidate still failed closed at
  55.024 s. Although two fresh 32-token discriminator waves passed at
  48.859–52.452 s, its exact-main packed gate still exposed a 55.411 s tail.
  Two fresh same-binary 64-token waves then passed at 45.725–51.415 s with
  unchanged 6,677-token prefix reuse and all tool-result checks green. Its
  exact-main packed gate subsequently exposed one 55.585 s tail under a fully
  Nominal thermal envelope. Two pinned same-input llama.cpp waves measured
  68.438 s and 69.944 s (69.191 s median), so the protected hf2q ceiling is
  rebaselined to 60 s—9.2 s below the current peer median. Exact packed
  two-wave and cross-family release authority remains pending.

## [0.1.6] — 2026-08-10

### Fixed

- Keep long agentic continuations attached to their strongest retained prefix
  across all three SlotAware families. Admission now distinguishes an active
  prefix from an idle reusable prefix: a continuation waits for the active
  owner when that match is strictly better, while an equal idle match remains
  runnable. This prevents Qwen, Gemma, and DeepSeek from admitting a 100K-token
  continuation cold into another free slot.
- Preserve the newest committed checkpoint when Qwen, Gemma, or DeepSeek work
  is cancelled. Gemma and DeepSeek keep in-request candidates private until
  success. Qwen promotes a stable pre-generation checkpoint immediately after
  its bounded prefill transaction commits, before observing a disconnect, so
  cancellation may safely retain newly committed prompt work without retaining
  a partial generation tail. Recovery restores only a validated checkpoint and
  otherwise resets cold. Inline embedding invalidates the selected slot's
  retained metadata before touching physical KV, so a later generation cannot
  advertise a stale cache hit.
- Let dead requests bypass physical-slot affinity waits in every SlotAware
  family, and let Qwen deterministic response-cache hits bypass them without
  claiming KV. Qwen prompt-cache identity now includes `reasoning_forced_open`,
  preventing a response split under one reasoning mode from being replayed
  under another. Requests for token logprobs bypass Qwen terminal response
  caching until that cache can retain and replay the full logprob payload.
  Gemma and DeepSeek continue to reuse their family-specific KV
  checkpoints rather than advertising a terminal-response cache they do not
  own. Bounded wait queues keep Shutdown non-evictable and allow runnable work
  to displace a blocked active-prefix waiter without overtaking earlier
  runnable requests.
- Resume DeepSeek-V4 retained-prefix continuations whose remaining suffix is
  below the 33-token matrix-append minimum. All nonempty 1–32-token segments
  now use incremental verifier replay, including the segment immediately
  before the eight-token recovery boundary, instead of failing with an empty
  resumable-prefill chunk.
- Restore bulk prefill for a saturated DeepSeek four-cold cohort. Once a
  filling cohort still has another cold request queued, cold-wave unary decode
  is deferred through its draining phase while any cold prefill remains because
  unary output cannot be delivered before that barrier. Full 2,048-token
  prefill transactions resume; streaming and warm decoders remain responsive,
  as does the lopsided eight-token decode/two-window prefill case.
- Pair large DeepSeek routed-expert gate/up projections through the
  family-neutral `mlx-native 0.10.8` schedule primitive. Eligible prefill work
  builds one expert-routing schedule for both existing quantized projections;
  decode-sized work and forced diagnostic routes remain independent. This is
  not an arithmetic-fusion claim, and exact packed hf2q hardware acceptance
  remains required before publishing an end-to-end speedup.
- Refine an over-ceiling macOS process RSS with `footprint` before rejecting a
  100 GiB DeepSeek load. This avoids treating reclaimable WebKit and
  IOAccelerator mappings from a remote-inference coding client as private
  resident memory; a missing, failed, or malformed footprint probe keeps the
  conservative RSS value and still fails closed.
- Resolve the published, checksum-pinned `mlx-native 0.10.8` backend, retaining
  the 0.10.7 command-buffer
  lifetime correction. Internal unlabeled GraphSession command buffers retain
  direct owner release, while labeled or publicly escaped buffers preserve the
  autorelease scope required by external Objective-C ownership. No local Cargo
  patch participates in the release candidate.
- Keep Gemma SerialFifo long-prefix continuations on the bounded live-append
  graph even for 1–31-token suffixes. Sliding-cache staging now derives each
  query position from the actual staged cache capacity rather than capping it
  independently at the 1,024-token semantic window. This preserves the
  current suffix K/V and chronological history beyond the window, and lets a
  successful append publish a reusable latest anchor for the following turn.

### Validation

- Add a cross-family long-context cache-lifecycle gate that establishes an
  agentic tool turn, starts a streamed turn on the retained prefix, queues its
  exact retry while that prefix is active, cancels the owner, requires the
  queued retry to reuse the restored checkpoint, and verifies an unrelated
  conversation cannot inherit private history.
- Pin the DeepSeek paired-prefill selector to large automatic routes with
  scratch and no diagnostic threshold override. The published native package
  and broad DeepSeek model-free suite are locked gates; real-model quality,
  cache, overlap, and calibrated cold-wave performance remain hardware gates.
- Add a guarded self-hosted `Cache lifecycle` workflow that packages one exact
  main-branch commit, builds only from that extracted crate, runs DeepSeek,
  Gemma, and Qwen one process at a time under continuously checked AC power,
  verifies protected GGUF digests, and uploads a
  source/crate/binary/model-bound receipt. The publication workflow now
  requires that successful exact-SHA receipt and reproduces its crate digest
  before publishing.
- Run the real-model Gemma N=4/N=8, transaction-boundary, and long-resume
  parity checks with the optimized release profile and bind that profile in
  the hardware receipt. Metal command scheduling is profile-sensitive; a
  debug-only timing divergence must not be mislabeled as production evidence.
- Compare Gemma long-resume output with a cold request through the same
  production graph, require two consecutive LCP engagements, and verify both
  resumed turns byte-for-byte. The former forced-linear control compared two
  numerically distinct prefill graphs and could report cache corruption when
  only the graph choice differed.
- Make the release-authority Gemma overlap timeout explicit and overridable.
  The protected gate now supplies 1,800 seconds for its fixed 175K-token
  full-attention prompt instead of inheriting a non-overridable 900-second
  curl cutoff. It also allows 180 seconds to observe cancellation at the next
  transaction boundary when a 4,096-token Metal call is already in flight at
  a 183K-token cursor; invalid or non-positive values fail before dependency
  discovery or model traffic. Its hosted contract uses platform `grep` rather
  than assuming a runner has ripgrep installed.
- Stabilize Qwen's automatic-tool acceptance prompt around the real operator
  path `/opt/hf2q/Cargo.toml`: the system turn now tells the coding agent to
  invoke tools directly instead of imitating them in Markdown, and the mock
  tool result explicitly says the completed `read_file` call must not be
  repeated. The packed candidate still supplies the returned Cargo.toml bytes.
  Receipts bind SHA-256 identities for both prompt surfaces, and publication
  rejects a different path, prompt, or result envelope.
- Keep Gemma's four-slot release-default agentic latency limits unchanged,
  while giving the experimental eight-slot correctness/aggregate-cap probe an
  explicit 40-second cold-TTFT and 30-second tool-result completion envelope.
  On the exact M5 Max discriminator the eight-slot wave measured 25.279 seconds
  cold and 23.932 seconds at the slowest tool-result turn; every receipt keeps
  the actual timing, cache, and semantic fields rather than converting this
  experiment into an eight-slot latency claim.
- Keep that dispatch compatible with macOS Bash 3.2 under `set -u` by using
  explicit four-slot and eight-slot command branches instead of expanding an
  empty array. The release wrapper now also preserves the originating failure
  status through its cleanup trap, so a pre-manifest harness error cannot be
  reported as a successful workflow step after servers and guards are reaped.
- Make that receipt enforce the family-specific shipping gates as well as the
  shared lifecycle: Qwen overlap/continuation, cold four-agent heap waves, and
  one-slot disconnect; Gemma long-prefill overlap/cancellation, fresh-versus-
  reused 4,096/8,193 bounded-output parity, four/eight-slot aggregate caps,
  and heap waves; and
  DeepSeek cached-suffix cancellation, terminal parking, and two fresh
  four-agent waves. Release rehashes every downloaded receipt and validates
  the detailed evidence before registry credentials are exposed.
- Freeze the four-agent agentic repository context to the exact 21,204-byte
  calibration fixture instead of embedding the growing live README. The
  producer and protected release path now require its fixture ID, SHA-256,
  byte/character counts, exact 6,685-token DeepSeek render, zero cold reuse,
  semantic/tool proof, and the literal 60-second cold limits; a model-free
  negative matrix rejects stale or incomplete receipts.
- Run the two calibrated DeepSeek four-agent waves before the long functional
  lifecycle heat-soaks the M5 runner. Each wave now requires at least 60
  seconds of Nominal samples at five-second cadence with no hf2q/llama model
  runtime loaded, then fails if two-second telemetry becomes non-Nominal,
  malformed, or unavailable before all four atomic cold receipts arrive. The
  workload is not paused or reordered; cached work may overlap the cold tail as
  before. Thermal receipts bind all four cold-receipt names and hashes, while
  the same live cache must still finish cached unary/SSE, automatic tool
  choice, and tool-result continuation under unchanged limits. The cohort
  parent immediately reports a child that exits before publishing its cold
  receipt.
- Add a developer-only matched llama.cpp cold-wave discriminator for the
  frozen DeepSeek fixture. It binds peer binary/model/request identity, exact
  zero-cache `read_file` semantics, runtime-specific prompt-token counts,
  monotonic timing, AC power, and fail-closed thermal telemetry without making
  the reference runtime part of hf2q serving or release authority.
- Update vulnerable transitive dependencies (`crossbeam-epoch`,
  `quinn-proto`, `rkyv`, `anyhow`, `memmap2`, and the legacy Rustls chain via
  `ruvector-core 2.3`) and make a zero-vulnerability `cargo audit 0.22.2` run
  blocking in CI and release packaging.

## [0.1.5] — 2026-08-09

### Added — foreground serving dashboard

- Add a TTY-only live operator dashboard for foreground `hf2q serve` runs.
  It keeps one stable row per active request with slot, phase, cached versus
  new prompt tokens, prefill percentage/rate/ETA, and decode rate instead of
  printing one log line per transaction. `--operator-ui auto` is the default;
  `plain` preserves the traditional stream and `dashboard` fails early when
  stderr is not interactive or JSON logging is selected. Inference publishes
  through a bounded nonblocking channel, so terminal rendering cannot delay
  GPU or scheduler work.

### Fixed

- Refine DeepSeek lopsided cold-cohort scheduling with an interactive
  mixed-work budget of up to eight decode tokens between prefill slices capped
  at two native windows. When no runnable decoder remains, restore the proven
  full 2,048-token prefill transaction.
- Capture Qwen's stable native transcript boundary before the rewriteable
  generation cue and reuse it for normal continuations even when the generated
  live tail does not prefix-match the client's restored reasoning/tool history.
  Continuations restore that verified KV and prefill only the changed tail.

## [0.1.4] — 2026-08-09

### Fixed — bounded multi-agent prefill and Metal command-buffer lifetime

- Advance Qwen 3.5/3.6 slot-aware prefill in bounded, scheduler-yielding
  transactions. Active decoders run before each cold-prefill slice, cold
  prefills rotate fairly, and cache/ledger state is published only after every
  full-attention and MTP cursor agrees at a successful transaction boundary.
- Bound inline Qwen embeddings to one admission quantum, and reject
  soft-token/deepstack SlotAware requests explicitly until those multimodal
  paths have scheduler-yielding prefill and decode. No multimodal field is
  silently discarded into the plain-text graph.
- Treat Metal command-buffer timeouts and ignored submissions as
  SlotAware-worker-fatal for Qwen, Gemma, and DeepSeek. Active, detached, and
  queued requests terminate once, readiness fails closed, and the poisoned
  device generation receives no further GPU submissions or cache resets.
- Supervise individual Metal transactions from outside the model worker.
  A transaction that never returns poisons readiness, unblocks unary/SSE
  waiters, and requires process restart; slow SSE consumers are cancelled
  locally rather than backpressuring every slot. Fatal reply fanout is
  nonblocking even when a stream channel is already full.
- Preserve Qwen worker failures at the OpenAI boundary: invalid request shapes
  are rejected before Qwen LM scheduler/SSE admission and before Qwen LM
  generation, unsupported SlotAware capabilities
  return 501, queue/physical-budget pressure returns 429, and an unhealthy
  engine returns 503 instead of collapsing these cases into a generic 500.
- End Qwen's final encoder-session stage without opening an unused replacement
  command buffer. The companion `mlx-native` correction scopes Objective-C
  autoreleases at command-buffer, compute-encoder, and label-string seams.
  hf2q resolves the published, checksum-pinned `mlx-native 0.10.6`; no local
  Cargo patch participates in the 0.1.4 release.
- Graduate the validated Qwen3.6 autoregressive route to the default product
  surface. The canonical launcher no longer sets the investigation-only
  `HF2Q_QWEN36_AUTOREG` gate; unsafe chunk-scan remains a separate experiment.
- Add deterministic scheduling, cancellation, fatal-fanout, readiness,
  cross-layer cache-cursor, and 87,972-token/347-tool watchdog regressions.
- Advance long Gemma text prefills in bounded 4,096-token transactions, split
  exactly at the stable-prefix boundary, validate and publish every configured
  physical cache cursor only after a successful transaction, and run decode
  before prefill in `Mixed`. Cross-slot batches validate every lane before
  admission and share one 4,096-row aggregate transaction cap instead of
  multiplying that cap by the slot count. Already-installed long-text prefill
  states may share that same aggregate transaction while preserving per-lane
  cache, scheduler, and reply ownership. Long soft-token requests remain
  fail-closed until their own resumable path is proven.
- Route meaningful DeepSeek retained-prefix suffixes through the existing
  atomic resumable-prefill machinery. During a lopsided cold cohort, one
  decode token runs before each remaining prefill transaction; a terminal
  response stays parked until the cold-prefill barrier lifts, preserving the
  retained cache for its continuation. Cancellation and ordinary failure now
  reconcile the cohort back to an admissible phase instead of stranding an
  idle worker. Staggered warm continuations can join an existing decoder while
  another physical slot is free, and cancellation restores a valid,
  position-consistent pre-request turn anchor instead of deleting reusable KV.
- Add a focused DeepSeek cached-suffix gate that requires three resumable
  transactions with peer decode progress between them, exact middle-transaction
  cancellation accounting, no terminal Done after disconnect, retained-prefix
  reuse after rollback, readiness, and a clean fatal-log delta.

## [0.1.3] — 2026-08-08

### Fixed — registry verification and Eagle3 ordering

- Include the small DeepSeek encoding fixtures, quantizer byte-comparison
  fixtures, and continuous-batching source audit used by crate-local unit
  tests. The 0.1.2 runtime archive built and installed correctly, but
  `cargo test` on the downloaded crate could not compile or complete because
  those inputs were omitted.
- Order the Eagle3 FC-to-normalization, normalization-to-concat, and
  concat-to-projection dependencies explicitly when mlx-native uses its
  concurrent Metal encoder. This removes an intermittent bit-identity drift
  exposed by the packed-crate release gate without serializing the independent
  Q/K/V projections.
- 0.1.3 replaces the yanked 0.1.2 package.

## [0.1.2] — 2026-08-08

### Added — full-context multi-agent serving

- Slot-aware serving now gives every Gemma 4, Qwen 3.6, and DeepSeek V4
  agent the model's full logical context. Context is never divided by the
  number of concurrent slots; weights remain shared while each slot owns
  independent, demand-grown KV and recurrent state.
- Shared physical KV admission accounts for retained slot high-water and
  active worst-case growth. Requests queue or fail explicitly under physical
  pressure instead of silently shrinking context or overcommitting unified
  memory.
- Prefix-affine scheduling keeps each conversation on the slot with its best
  reusable native cache state. Normal OpenCode turns prefill only the appended
  suffix, including DeepSeek recovery anchors across cache-capacity growth.
- Canonical OpenCode launchers now default to four full-context slots and
  expose the shared KV budget, scheduler, cache reuse, time-to-first-token,
  prefill rate, and decode rate in verbose operator telemetry.

### Fixed — native agentic semantics across all three families

- Gemma 4 and Qwen 3.6 use their native GGUF chat templates and tool-call
  encodings; cross-family or incomplete template state fails closed rather
  than falling through an approximately compatible formatter.
- Tool definitions, required and automatic calls, source-shaped arguments,
  tool-result continuations, unary responses, and SSE streams are covered by
  realistic four-agent gates for Gemma 4, Qwen 3.6, and DeepSeek V4.
- DeepSeek cold prefill resumes only after atomic cache-and-ledger commits.
  At most two cold agents alternate matrix transactions through one scratch
  arena; the completed cohort then decodes in fair eight-token quanta, and
  retained-prefix continuations run before unrelated cold work can evict them.

### Performance and proof

- On the target M5 Max with AC power, two exact four-agent DeepSeek gates
  completed their cold cohorts in 53.86 and 52.32 seconds (53.09-second
  median), versus the then-observed approximately 54.1 seconds for matched
  llama.cpp with `--kv-unified`,
  four parallel slots, and 131,072 logical tokens per slot. llama.cpp's
  524,288-token unified allocation did not fit beside the 100 GiB model on
  this 128 GiB host; hf2q retained 524,288 logical tokens per slot through
  demand-grown physical admission.
- A DeepSeek continuation after 131,072-to-262,144 cache growth reused
  119,692 of 119,778 prompt tokens (99.92%) and reached the first semantic
  stream event in 1.132 seconds.
- Gemma's matched 24,200-token four-slot continuation sustained about
  1,831 aggregate prefill tokens/s, slightly ahead of the same GGUF and
  settings under llama.cpp at about 1,733 tokens/s. Gemma and Qwen four-agent
  gates retained at least 7,089 and 6,980 cached prompt tokens respectively,
  with native tool-result continuation and the full 262,144-token logical
  context per agent.
- `mlx-native` 0.10.4 supplies the family-neutral lazy overwrite allocation,
  ring linearization, F16 mask construction, and direct TQ-HB-to-F16 staging
  primitives used by the bounded full-context paths.

## [0.1.1] — 2026-08-06

### Added — DeepSeek V4 Flash agentic conversion and serving

- Rust-native conversion of the official
  `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint to the owned
  `deepseek4-agentic-q2` Q2_K/Q3_K/Q8_0 profile; no external converter,
  quantizer, or inference runtime is invoked.
- OpenAI-compatible DeepSeek serving for unary and SSE completions,
  reasoning, official DSML tool calls, required/automatic tool choice,
  multiple calls, and real OpenCode multi-turn coding sessions.
- Native live-prefix caching and reasoning-tail recovery. A 119,916-token
  post-tool request reused 119,813 tokens (99.91%) and reached its first
  semantic token in 1.275 seconds instead of recomputing the conversation.
- Adaptive gathered sparse prefill for long ratio-four compressed history,
  backed by `mlx-native` 0.10.1. On the source-bound M5 Max gate, hf2q
  completed the 119,821-token cold tool prompt in 494.449 seconds versus
  roughly 556 seconds for the same artifact under llama.cpp build 10293.
- Canonical OpenCode launcher and fail-closed agentic/long-context gates:
  `scripts/serve_deepseek4_opencode.sh`,
  `scripts/test_deepseek4_agentic.sh`,
  `scripts/test_deepseek4_opencode.sh`, and
  `scripts/test_deepseek4_long_context_cache.sh`.

### Added — Gemma 4 LCP long-resume past sliding_window + production hardening ("gemma-hybrid-lcp" follow-up)

- **LCP now works past `sliding_window` (1024) under the hybrid regime.**
  Sliding layers allocate linear buffers on both prefill routes;
  hybrid encode writes slot=logical at prefill and decode (capacity-
  derived predicate); the hybrid SDPA kernel's `mask_type=2` windowing
  is covered by `gemma_hybrid_long_resume_byte_identity`. The current
  release-candidate correction compares resumed and cold requests through
  the same production graph; the earlier forced-linear control was not a
  same-graph cache-coherence oracle.
  `HF2Q_KV_LCP_RESUME_CAPACITY=8g` is the documented envelope knob
  (snapshots carry +4096/turn multi-turn headroom).
- **Fixed: SerialFifo consume-gate 500 on growing conversations** — a
  stale `dense_kvs` mount from turn N hard-errored turn N+1 on the
  non-batched route when N+1's prompt was longer (`capacity < required`).
  Leftover mounts now drop + fresh-alloc; slot-aware scaffold bails
  preserved.
- **Fixed: pre-copy snapshot budget gate** — dual-leg snapshot bytes are
  estimated from shapes BEFORE the alloc+memcpy and skipped when over
  the registry budget; a ~97K-token dual-leg snapshot (~64 GB)
  previously swap-stormed the box before rejection could fire.
- **Launcher + envelope:** `scripts/serve_gemma4_opencode.sh` (one-model
  OOM guard, mmproj, `BATCHED=0` forces the linear-memory non-batched
  route for >32K contexts — the batched route's O(n²) bf16 masks OOM
  past ~32K at ~100K-token prompts). opencode-scale ~100K sessions are
  documented as qwen-arch territory (fixed-size DeltaNet + TQ full-attn).
- **Documented pre-existing divergences (reproduced at clean main, NOT
  introduced by the gemma-hybrid-lcp arc):** batched-prefill sliding-
  layer output diverges from the non-batched parity reference at seq>sw;
  `iter5_r_c4_lcp_5_fraction_sweep` fraction 0 (tiny K) diverges —
  dense-side, follow-up work.

### Added — Gemma 4 LCP partial-prefill resume under the production hybrid regime ("gemma-hybrid-lcp")

- **Gemma 4 has prefix caching in production for the first time.** The
  LCP resume path previously restored only dense F32 K/V and its
  snapshot was gated `HF2Q_USE_DENSE=1` — under the production hybrid
  regime (F16-K + TQ-HB V, `HF2Q_HYBRID_KV` default-on) no snapshot was
  ever taken, so every multi-turn request re-prefilled the full
  conversation.
- Root-cause fix for a silent production gap: the iter-344 batched
  prefill route (`forward_prefill_batched`, the SerialFifo default for
  gemma chat) populated **neither** LCP snapshot — so the registry
  stayed empty and every probe missed, for dense AND hybrid regimes
  alike. Both snapshots are now populated there.
- New regime-aware per-layer registry payload `GemmaLcpLayerKv`
  (`Dense` / `DenseAndHybrid`): prefill attention reads the dense leg,
  decode under hybrid reads the hybrid leg — an LCP resume restores
  BOTH, closing the same silent-corruption class ADR-027 sub-iter 23d-γ
  closed for qwen35. Regime-consistency check at install rejects
  cross-substrate entries (dense-only entry under hybrid = clean miss,
  never a zero-restore).
- `effective_kv_lcp_resume` widened: resumable substrates are now
  gemma-dense, gemma-hybrid, and qwen35-TQ (23d-γ); the HB-encoded
  opt-out regime stays auto-disabled. LCP is now default-on for
  production gemma and qwen35 serves.
- Gates: durable integration test
  `gemma_hybrid_lcp_partial_prefix_byte_identity` (two-server,
  engagement-asserted) + live ENGAGED trace `K=516 of N=537` with
  byte-identical output vs cold control (non-streaming and streaming).
  Known boundary: gemma LCP still skips prompts > `sliding_window`
  (1024) — LONG_RESUME is dense-only today; hybrid long-resume is
  follow-up work.

### Fixed — TQ-only LCP resume + disk persist made production-correct (ADR-027 sub-iter 23d-γ)

- **Silent coherence corruption fixed in TQ-only LCP resume.**
  `HybridKvCache::restore_partial` now partial-copies the first-`n_tokens`
  positions of all four TQ buffers (U8 packed + F32 norms) per slot for
  full-attn AND MTP slots. Previously, under production `tq_kv_active=true`,
  TQ buffers stayed zero-initialized after an LCP resume while
  `current_len` advanced — the resumed request attended over zeroed K/V
  for the entire cached prefix.
- **`cfg_from_cache` no longer panics in TQ-only mode** (was an engine
  worker panic → HTTP 500 on any request with `HF2Q_KV_PERSIST` set).
  Shape is derived from `slot.tq.k_packed` when F32 backing is absent;
  new `KvSubstrate` (F32Only/TqOnly/Both) classification with a
  uniform-substrate invariant check.
- **QH35 disk codec bumped to v4** — per-MTP `kv_present` byte (mirrors
  the full-attn v2 byte; v1..v3 envelopes still deserialize), and
  TQ blobs now deserialize into logical 4-rank shapes (was flat rank-1,
  which hard-failed `restore_partial` on hydrated snapshots).
- **On-disk fingerprint is substrate-namespaced** — snapshots written
  under one KV substrate never hydrate into a cache allocated under
  another (cross-mode hydration would have silently zero-restored; now a
  clean cache miss).
- Live gates (qwen36 APEX Q5_K_M, M5 Max): needle-recall byte-identical
  cold vs LCP-resumed (TTFT 2920ms → 326ms = 9.0×), 182 MB checkpoints
  written through to disk, cold-process restart hydrate byte-identical.
  Unit: 33 persistor + 189 kv_cache + 228 kv_persist tests green.
- **`Qwen35DiskPersistor` now enforces `HF2Q_KV_PERSIST_BUDGET_BYTES`**
  with LRU-by-mtime eviction per cfg subdir — pre-fix, one ~100K-token
  opencode session wrote 105 GB of chunk snapshots unbudgeted.
- New canonical launcher `scripts/serve_qwen36_opencode.sh` (full
  production env incl. `HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE=4096` for
  long-context checkpoint economics).

### Fixed — tool-call grammar for structured parameters (ADR-005 iter-231)

- **Agentic clients (opencode, MCP tool servers) can now call tools whose
  schemas declare `object` / `array` parameters.** Previously every
  `/v1/chat/completions` request carrying such a schema failed with
  HTTP 400 (`parameter 'X' uses unsupported schema type 'object'` — the
  wave-2.5 scalar-only emitter gate).
- Tool-call GBNF emitters (Qwen 3.5/3.6 + Gemma 4) now compile nested
  schemas recursively with full declared-structure fidelity: per-key
  value grammars, any-order `required` (≤8) permutation, typed array
  `items`, enums, type unions, `anyOf`/`oneOf`, and
  `additionalProperties:false` key-set closure. Free-form objects
  (no declared `properties`/`items`) compile to permissive recursive
  JSON value rules matching exactly what the chat templates emit
  (`tojson` / `format_argument`).
- The JSON Schema `pattern` regex keyword is now COMPILED into the
  tool-call value grammars via a real regex→GBNF compiler
  (`grammar/regex_gbnf.rs`, iter-231c) — literal/class/group/alternation/
  quantifier support with honest errors for non-regular features
  (backreferences, look-around, property classes). This unblocks MCP
  tool schemas like ruvnet-brain's `argv` items (`^[a-z][a-z0-9-]*$`).
- Schemas using features the grammar cannot enforce (`allOf`, `$ref`,
  tuple `items`, …) now return an honest 400 naming the feature and
  dot-path instead of failing the whole request class.
- Gemma 4 tool-call parser now round-trips nested structured arguments
  (objects/arrays containing commas) into `arguments_json` as real JSON
  instead of mangling them into string fields.
- Top-level untyped parameters accept structured values (both families).
- Also fixed two pre-existing sweep failures: H1 structural audit
  (`GOLDEN_OUTPUT` hoisted to module scope) and upstream citation drift
  (qwen35/qwen35moe catalog line numbers re-transcribed against current
  llama.cpp); cleared 8 pre-existing build warnings. Full sweep:
  4120 passed / 0 failed / 0 warnings.

## [0.1.0] — 2026-05-16

First public release.

### Added — convert pipeline

- HuggingFace → GGUF / mlx-lm safetensors converter (`hf2q convert`).
- Quantization families:
  - Float passthrough — `f16`, `bf16`, `auto`.
  - Legacy block — `Q2`, `Q4` (alias `Q4_0`), `Q8` (alias `Q8_0`).
  - K-quants — `Q2_K{,_S}`, `Q3_K_{S,M,L}`, `Q4_K_{S,M}`, `Q5_K_{S,M}`, `Q6_K`.
  - Imatrix-weighted K-quants — `imatrix-*` variants plus `imatrix-adaptive`.
  - Mixed-bit — `dynamic-quant-{4-6, 4-8, 6-8, 2-8}`.
- DWQ training (`hf2q dwq-train`) producing an mlx-format safetensors
  overlay layered on top of a GGUF (ADR-020).
- Two-pass intermediate GGUF for activation capture during
  Qwen 3.5 / 3.6 conversion (ADR-012).
- Streaming convert pipeline with disk-floor preflight (ADR-014).

### Added — inference + serving

- Apple-Silicon-only inference path on top of `mlx-native` (ADR-008).
- GGUF reader, KV cache, prefill, and decode for:
  - Gemma 4 (dense + MoE).
  - Qwen 3.5 / 3.6 (dense + MoE + multi-token-prediction).
  - Qwen 3-VL (vision + text).
  - BERT / Nomic-BERT (embedding-only).
- TurboQuant 8-bit KV cache (ADR-007) with Hadamard packing — drops
  F32 K/V allocations on Qwen 3.6 35B-A3B at 32K context for a
  **3.94× memory savings** vs the F32 baseline (ADR-027 iter-34).
- Flash-Attention prefill kernel (ADR-011) — `1.07–1.09× peer-FA`
  AHEAD at `pp1800–pp3700` (ADR-029 iter-160).
- Batched prefill V3 with byte-identical decode parity (ADR-029 Step
  1j.2).
- Speculative decode (ADR-030) — n-gram drafter Plan B available
  opt-in via `HF2Q_SPEC_NGRAM=1` (default OFF; DFlash investigation
  closed without shipping).

### Added — HTTP API (`hf2q serve`)

- OpenAI-compatible endpoints: `/v1/chat/completions`,
  `/v1/embeddings`, `/v1/models`.
- Streaming SSE.
- Tools / function-calling.
- Vision (`qwen3vl`) via `image_url` data-URI and HTTPS fetch.
- Grammar-constrained sampling.
- Persistent block-prefix KV cache (ADR-017).
- Multi-model serving from a single process.
- Jinja2 chat-template rendering matching `llama.cpp`'s vendored
  Jinja parser and mlx-lm's Python behavior, including the
  `pycompat` adapter for `.split()` / `.strip()` on strings
  (ADR-005 Phase 2a iter-133).

### Added — operator tools

- `hf2q doctor` — hardware / cache / RuVector / disk diagnostics.
- `hf2q info` — inspect an HF or GGUF model without converting.
- `hf2q validate` — cosine similarity, KL divergence, perplexity vs a
  reference.
- `hf2q parity` — ADR-009 parity validation against locked reference
  outputs.
- `hf2q smoke` — ADR-012 end-gate smoke test for a registered
  architecture.
- `hf2q gguf-patch` — rewrite GGUF metadata in place.
- `hf2q cache` — manage `~/.cache/hf2q/`.
- `hf2q completions` — shell completions.

### Performance highlights (M5 Max, thermal-fair alt-pair, σ < 1% per arm)

- **Gemma-4 decode** — `1.05× peer-FA` AHEAD across
  `tg200 / tg2000 / tg5000` after ADR-029 Step 1i (parallel
  SG-tournament top-K in `fused_moe_routing`) + Step 1j.2 (V3
  batched softmax tree-reduce). Byte-identical greedy decode vs V2
  baseline.
- **Qwen 3.6 35B-A3B-APEX-Q5_K_M decode** — `~1.34× peer-FA`
  sustained to 1000-tok (ADR-028 iter-308 → iter-324).
- **Prefill** — `1.07–1.09× peer-FA` AHEAD at `pp1800–pp3700`
  (ADR-029 iter-160).
- **KV-cache footprint** — 3.94× vs F32 baseline at 32K context on
  Qwen 3.6 35B-A3B (ADR-027 iter-34, Hadamard-packed TQ path).

### Notes

- macOS / Apple Silicon only (M1 or newer). The inference path is
  Metal-only by design (ADR-008 "candle divorce").
- DWQ at the production-default `perturb=1.0` is mathematically
  equivalent to the underlying K-quant baseline (ADR-020 finding
  2026-05-08). Wins materialize only at lower perturb values that
  move scales / biases off the K-quant projection.
- Per-arch disk floor for convert: 100 GB (Qwen 3.5 dense),
  150 GB (Qwen 3.5 MoE). Smoke preflight refuses to start below
  `disk_floor_gb + 10`.

[Unreleased]: https://github.com/robertelee78/hf2q/compare/v0.1.6...HEAD
[0.1.6]: https://github.com/robertelee78/hf2q/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/robertelee78/hf2q/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/robertelee78/hf2q/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/robertelee78/hf2q/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/robertelee78/hf2q/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/robertelee78/hf2q/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/robertelee78/hf2q/releases/tag/v0.1.0
