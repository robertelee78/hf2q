# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[0.1.0]: https://github.com/robertelee78/hf2q/releases/tag/v0.1.0
