# hf2q — Architecture

This document is a source-grounded map of the `hf2q` codebase at
`main` HEAD. It exists to bring a new contributor up to "I can find
the code that owns X" in under thirty minutes.

It is **descriptive** (what is) not **prescriptive** (what should be).
For target-state designs see the per-ADR files under `docs/`; for the
prior-art inference path see `docs/arch-current-inference-path.md`.

---

## 1. What `hf2q` is

`hf2q` is a single Rust binary with two cooperating halves:

1. **Convert** — read a HuggingFace model directory (`config.json` +
   `*.safetensors`), normalize tensor names per architecture, run a
   quantization algorithm, and currently emit a GGUF file. A production
   MLX-affine output is target-state work governed by ADR-046, not a current
   converter backend.

2. **Serve / Generate** — load a GGUF, run prefill + decode on the
   Apple-Silicon GPU through the `mlx-native` crate, and expose
   OpenAI-compatible HTTP endpoints (chat completions, embeddings,
   models) with SSE streaming, tool calls, vision, grammar-constrained
   sampling, and a persistent block-prefix KV cache.

Both halves share the same internal IR (`src/ir/`) and the same arch
registry (`src/arch/`). Conversion-family implementations live under
`src/convert/arch/`; runtime-family graphs live under
`src/inference/models/`.

### Sovereignty rule (`docs/arch-onboarding.md`)

- **Pure Rust.** No `llama.cpp` / `candle` code, crate, binary, or
  build artifact in hf2q deliverables at build / test / CI time.
  Enforced by ADR-008 ("candle divorce").
- **Spec sources are read-only.** `llama-arch.cpp`,
  `convert_hf_to_gguf.py`, `clip.cpp`, `clip-model.h` are *read* to
  derive specs; every transcribed value carries a `// citation:` line
  back to the source file + line.
- **No external oracles in tests.** Correctness is proven by
  hand-authored expected values, spec-driven synthetic inputs, or
  round-trip gates (emit → load through our own loader).

---

## 2. Crate layout

```
hf2q (one binary `hf2q`, one narrow [lib] facade for tests)
├── src/main.rs          process entry, exit-code classification
├── src/lib.rs           narrow library facade (kv-persist only,
│                        for tests under `tests/`)
├── src/cli.rs           clap derive — every subcommand + arg
├── src/doctor.rs        `hf2q doctor` runtime diagnostic
├── src/preflight.rs     ADR-012 preflight checks (disk, token, …)
├── src/progress.rs      indicatif-based progress reporting
├── src/gguf_patch.rs    metadata-only GGUF rewriter (no tensor I/O)
├── src/distribution/   ADR-045 release/install bounded context
│   ├── schema/         strict bounded manifest, receipt, marker schemas;
│                       marker v2 records exact preparation-role versions and
│                       deterministically reconstructs first-install receipts
│   ├── install_state/  shared descriptor-relative installation lock,
│       ├── metadata/   canonical crash-durable update-metadata journal;
│       │                stored bytes are not cryptographic authority
│       ├── extraction/ private exact-replay extraction tree, bounded retained
│       │                stages, and descriptor-relative crash recovery
│       └── …           sequence-one activation and bounded restart cleanup
│   ├── update_auth/    transport-free strict TUF profile, one-use request
│                       tokens, historical replay floors, root-authorized
│                       timestamp/snapshot recovery, sealed advancing
│                       commit guard, and durable metadata-baseline proof;
│                       all internal until the real root and public update
│                       authority exist
│   ├── update_transport/ closed Pages-pointer/GitHub-asset routes and manual
│                       one-hop release redirect, exact bounded bodies, and anonymous
│                       same-FD streamed archive staging; no extraction,
│                       prepared-version, installer, or CLI authority
│   └── prepared_release/ bounded classic-ZIP structural/profile validation,
│                       canonical embedded-manifest and exact payload binding,
│                       shared-lock inert extraction, and a filesystem-free
│                       strict thin-arm64 Mach-O profile parser; no native
│                       codesign, prepared-version, publication, or CLI authority
│
├── src/arch/            ADR-012 arch registry (single source of truth)
│   ├── catalog.rs       TensorCatalog — expected tensor names + dtypes
│   ├── conformance.rs   quality thresholds, smoke prompts, MTP/vision flags
│   ├── entries/         one file per registered arch (qwen35, qwen35moe, …)
│   ├── registry.rs      ArchRegistry, ArchEntry, ArchError
│   └── smoke.rs         end-gate smoke driver (`hf2q smoke`)
│
├── src/input/           external model I/O — nothing else touches raw model files
│   ├── config_parser.rs HF config.json → ModelMetadata
│   ├── safetensors.rs   streaming mmap shard reader → TensorMap
│   ├── hf_download.rs   HF Hub download (with x-linked-etag integrity)
│   └── integrity.rs     per-shard SHA-256 verification
│
├── src/ir/              internal representation crossing modules
│   ├── mod.rs           ModelMetadata, TensorMap, DType, QuantizedTensor, …
│   └── lazy.rs          lazy-tensor handle (ADR-014 streaming convert)
│
├── src/models/          legacy/shared model conversion support
│   └── vit/             ADR-012 P10 pure-Rust mmproj (ViT) emitter
│
├── src/convert/arch/    active per-family conversion and metadata mapping
│   ├── qwen35_dense.rs  Qwen 3.5/3.8 dense conversion
│   ├── qwen35moe*.rs    Qwen 3.5/3.6 MoE conversion
│   └── …                other explicit supported families
│
├── src/quantize/        active pure-Rust quantization stack (ADR-033)
│   ├── ggml_quants/           block/K/IQ codecs + StandardPolicy
│   │   ├── apex/              APEX mixed-precision policy
│   │   ├── q{2,3,4,5,6}_k.rs K-quant codecs
│   │   ├── q{4,5}_{0,1}.rs    legacy block codecs
│   │   └── quantizer.rs       codec dispatch
│   └── imatrix/               corpus, capture, accumulator, GGUF I/O
│
├── src/backends/        active output writers
│   └── gguf/                  streaming GGUF metadata + tensor writer
│
├── src/quality/         disconnected experimental quality code (not production-wired)
│   ├── cosine_sim.rs          weight-level cosine similarity
│   ├── kl_divergence.rs       output-logit KL
│   ├── perplexity.rs          PPL on a corpus
│   ├── ppl_driver.rs          forward-pass driver for PPL
│   └── regression.rs          regression-gate accountant
│
├── src/intelligence/    capacity planning + measured selection + RuVector
│   ├── fingerprint.rs         stable model fingerprint (for cache keys)
│   ├── auto_quant.rs          legacy estimate-only planner
│   ├── measured_auto_quant.rs ADR-046 exact-evidence selector
│   ├── heuristics.rs          rule-based fallback when RuVector is silent
│   └── ruvector.rs            optional self-learning store (cargo feature)
│
├── src/inference/       runtime model + spec-decode + vision
│   ├── models/                per-arch forward graphs
│   │   ├── gemma4/            dense + MoE 30-layer Gemma 4
│   │   ├── qwen35/            dense + MoE Qwen 3.5 / 3.6
│   │   ├── qwen3vl_text/      Qwen 3-VL text tower (vision lives elsewhere)
│   │   ├── bert/              BERT embedding model
│   │   └── nomic_bert/        Nomic embedding model
│   ├── spec_decode/           ADR-029 speculative-decode primitives
│   │   ├── ngram_proposer.rs  pure-CPU n-gram drafter
│   │   ├── dflash/            ADR-030 dFlash block-diffusion drafter
│   │   └── verifier.rs        multi-token verify forward
│   └── vision/                mmproj load + image embed
│
├── src/serve/           HTTP API, KV-cache, multi-model
│   ├── api/                   axum router + handlers + state
│   │   ├── schema.rs                  OpenAI wire types
│   │   ├── handlers.rs                /v1/* request handlers
│   │   ├── router.rs                  axum router + middleware
│   │   ├── sse.rs                     SSE encoder
│   │   ├── engine.rs                  Gemma 4 engine wrapper
│   │   ├── engine_qwen35.rs           Qwen 3.5 engine wrapper
│   │   ├── engine_qwen3vl.rs          Qwen 3-VL engine wrapper
│   │   ├── grammar/                   grammar-constrained sampling
│   │   ├── kv_spill_descriptor.rs     KV-spill metadata
│   │   ├── tq_packed_descriptor.rs    TurboQuant packed metadata
│   │   ├── registry.rs                model registry (multi-model serve)
│   │   ├── embedding_pool.rs          /v1/embeddings request pool
│   │   ├── middleware.rs              CORS, request-id, auth
│   │   └── state.rs                   AppState, ServerConfig
│   ├── forward_mlx.rs                 Gemma-4 forward via mlx-native
│   ├── forward_prefill.rs             per-token prefill
│   ├── forward_prefill_batched.rs     ADR-015 batched prefill (35× wins here)
│   ├── kv_persist/                    ADR-017 persistent block-prefix cache
│   │   ├── block_store.rs             disk-backed block store
│   │   ├── writer.rs                  async writer + fsync barriers
│   │   ├── recovery.rs                crash-recovery on startup
│   │   ├── format.rs                  envelope + sidecar codecs
│   │   ├── index.rs                   in-memory index
│   │   ├── lcp_registry.rs            longest-common-prefix registry
│   │   ├── spiller.rs                 KvSpiller<E> trait impl
│   │   └── metrics.rs                 cache-side telemetry seam
│   ├── multi_model.rs                 multi-model registry + eviction
│   ├── encoder_worker_singleton.rs    Metal encoder worker
│   ├── auto_pipeline.rs               serve-time pipeline selection
│   ├── cache.rs                       global model cache (~/.cache/hf2q)
│   ├── parity_quality.rs              ADR-009 parity assertions
│   ├── provenance.rs                  GGUF → producer fingerprint
│   ├── quant_select.rs                model → quant-variant selector
│   ├── sampler_pure.rs                temp / top-k / top-p sampling
│   ├── spec_decode_cli.rs             generate-time spec-decode driver
│   ├── header.rs                      GGUF header read + validate
│   ├── gpu.rs                         shared GPU resource init
│   ├── layer_ctx.rs                   per-layer mutable context
│   ├── load_info.rs                   structured `loaded` event
│   ├── config.rs                      ServeArgs validation
│   └── mod.rs                         cmd_serve + cmd_generate entry
│
└── src/bin/             one-off audit binaries (iter23/24/25, dump_gguf_*)
```

The library facade (`src/lib.rs`) deliberately re-exports only
`serve::kv_persist::{block_store, format, index, metrics, recovery,
writer, lcp_registry}` — everything else stays binary-private. Tests
under `tests/` are integration-style; they bind to either the public
CLI surface (via `assert_cmd`) or to that narrow lib facade.

---

## 3. The convert pipeline

```
                          HF input directory or HF Hub repo
                                       │
                  ┌────────────────────┴────────────────────┐
                  │       src/input/ (mmap, config parse,    │
                  │       integrity, optional HF download)   │
                  └────────────────────┬────────────────────┘
                                       │ ModelMetadata + TensorMap
                                       v
                  ┌─────────────────────────────────────────┐
                  │  src/arch/   look up arch entry         │
                  │  src/convert/arch/ map names + metadata │
                  └────────────────────┬────────────────────┘
                                       │ canonical-named TensorMap
                                       v
                  ┌─────────────────────────────────────────┐
                  │  src/quantize/ggml_quants/ policy+codec │
                  │      [optional] imatrix importance      │
                  └────────────────────┬────────────────────┘
                                       │ QuantizedModel
                                       v
                  ┌─────────────────────────────────────────┐
                  │  src/backends/gguf/  →  *.gguf          │
                  │  conversion receipt + tensor manifest   │
                  └─────────────────────────────────────────┘
```

Streaming is real: `safetensors` shards are mmap'd, tensors are quantized in
bounded chunks, and the writer sinks blocks to disk as soon as they are ready.
ADR-033 owns the exact memory contract. The files under `src/quality/` are not
declared by the production crate, and real quantized quality acceptance must
not be inferred from their presence; ADR-046 tracks the receipt-producing
replacement.

### Quantization families

| Family | Where it lives | Notes |
|---|---|---|
| **Legacy block** (`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`) | `src/quantize/ggml_quants/q*.rs` | Pure-Rust 32-element block codecs. Converter support does not imply runtime support; ADR-046 Gate 0 tracks the current Q4_1/Q5_0 seam. |
| **K-quant** (`q2_k`…`q6_k`, `q4_k_m`, `q5_k_m`, …) | `src/quantize/ggml_quants/` | 256-element super-block codecs and llama.cpp-parity `StandardPolicy`. |
| **APEX mixed precision** | `src/quantize/ggml_quants/apex/` | Per-tensor GGUF policy; exact tensor encodings still use the codecs above. |
| **Imatrix input** | `src/quantize/imatrix/` | Corpus/capture and `.imatrix.gguf` producer/consumer for supported families. It is importance evidence, not DWQ. |
| **MLX affine overlay consumer (legacy)** | `src/core/mlx_safetensors_loader.rs`, `src/serve/forward_mlx_shared.rs` | Narrow Q4/group-32 consume-only path. There is no current DWQ producer or production full-model affine artifact. |

`dwq` remains a typed reserved selector. ADR-046 defines the source-agnostic
quality, artifact, kernel, and benchmark gates required before that name can be
activated.

---

## 4. The inference pipeline

### 4.1 Load

`serve::cmd_serve` / `serve::cmd_generate` →

1. **Header read** (`serve/header.rs`) validates the GGUF magic +
   producer fingerprint (`serve/provenance.rs`) against the arch
   registry.
2. **Arch dispatch** picks an explicit engine wrapper under
   `serve/api/engine*.rs` (Gemma 4, Qwen 3.5/3.6, Qwen 3-VL,
   DeepSeek-V4). Unsupported or cross-family shapes fail rather than falling
   through an approximately compatible graph.
3. **Weight load** dequantizes-on-demand into `mlx-native` MTL buffers
   (`inference/models/<arch>/...`). Fused-kernel pipelines compile at
   load time so the first request doesn't pay shader-compile latency.
4. **Warmup** runs a 1-token decode + a 10-token prefill, clears the
   KV cache, and emits the structured `loaded` event
   (`serve/load_info.rs`).

### 4.2 Prefill

Two paths share the same forward graph but differ in dispatch shape:

- `serve/forward_prefill.rs` — per-token; safe default; the ADR-009
  parity reference. Used when `HF2Q_BATCHED_PREFILL=0` or for arches
  not yet on the batched path.
- `serve/forward_prefill_batched.rs` — ADR-015's batched prefill;
  used by applicable Gemma and Qwen graph paths. Was the single largest serve
  speedup in the project (35× over per-token at `pp1024` on Gemma 4)
  when the HTTP path was wired in ADR-028 Phase 15. Flash-Attention
  (ADR-011) lives in this path.

Slot-aware scheduling adds a transaction boundary above the family forward
graph (`serve/api/engine.rs`, ADR-040). For Qwen 3.5/3.6 text serving, one
prefill transaction contains at most 2,048 new prompt tokens on one slot-local
hybrid KV cache. A successful transaction publishes the scheduler ledger only
after all full-attention and MTP cursors agree. `Mixed` steps decode active
streams before advancing the next cold prefill, and cold prefills rotate
round-robin. This is distinct from the unsafe chunk-scan DeltaNet experiment:
the outer transaction bounds the complete attention/MoE graph and gives the
worker cancellation and fairness boundaries.

Gemma 4 uses the same outer state-machine shape for long plain-text prompts,
with a family-specific candidate cap of 4,096 tokens and mandatory splits at
the stable-prefix boundary. Each successful transaction validates and commits
the engaged HB, hybrid, dense, and MLX cursor rows before scheduler
publication. DeepSeek uses its native verifier transaction width instead of a
generic token cap. Both cold work and meaningful retained-prefix suffixes are
resumable; cached suffixes remain outside the cold-cohort policy.

DeepSeek's `Mixed` step has a separate interactive budget. While a visible
decode lane is runnable, one prefill transaction is capped at two 128-token
verifier windows and decode receives up to eight tokens before the next
prefill slice. When a filling cohort still has another cold request queued,
cold-wave unary decoders are deferred through `Draining` while any cold
prefill remains because unary output cannot be delivered before that barrier;
streaming and warm decoders remain visible. If every decode owner is deferred,
parked, or absent, the cap is removed and the proven 2,048-token bulk-prefill
plan resumes. This avoids the latency cost of small transactions when no peer
can expose semantic progress.

When no prefill transaction remains, DeepSeek pure decode uses a separate
64-token slot quantum to amortize session swaps and scheduler publication.
That wider pure-decode quantum never enters the `Mixed` budget above, which
continues to clamp visible decode to eight tokens.

Large automatic DeepSeek MoE prefills use the family-neutral paired expert
projection primitive in `mlx-native 0.10.8`: gate and up share one routing
schedule while retaining their existing quantized arithmetic and distinct
outputs. Small/decode work, forced routing diagnostics, and threshold-override
measurements stay on the independent projection path. This is an hf2q
candidate optimization until the exact packed real-model gates in ADR-042 and
the shipping contract pass.

### 4.3 Decode

`serve/forward_mlx.rs` is the per-token decode hot loop. It:

1. Runs the per-arch graph (`inference/models/<arch>/`) through
   `mlx-native` MTL dispatches.
2. Reads / writes the KV cache through TurboQuant
   (`docs/operating-kv-cache.md`, ADR-007): K and V are Hadamard-
   transformed and quantized to 8-bit with a per-block scale, giving
   ~2× memory savings vs an F16 KV cache at negligible quality loss
   (Gate A cosine mean 0.9998, Gate B argmax divergence 0.8%). For
   Qwen 3.5 / 3.6 (ADR-027) the TQ-HB path drops F32 K/V allocations
   entirely, delivering 3.94× savings against the F32-only baseline
   (340 MiB vs 1.34 GiB at 32K context). The TQ-HB encode is fused
   into the dense KV-store path; the on-load path lazily promotes
   from the persisted block store.
3. Samples through `serve/sampler_pure.rs` (temp / top-k / top-p) and
   optionally a grammar-constrained `serve/api/grammar/` sampler for
   tool calls and JSON-mode.

### 4.4 Speculative decode (ADR-029 / ADR-030)

`inference/spec_decode/`:

- **N-gram proposer** (`ngram_proposer.rs`) — pure-CPU drafter,
  cost-free when the suffix repeats.
- **dFlash drafter** (`dflash/`) — ADR-030 block-diffusion neural
  drafter. Currently default-OFF; runs through the same verify
  forward as the n-gram path.
- **Verifier** (`verifier.rs`) — multi-token verify forward that
  returns per-position logits + a rollback handle for the KV cache so
  rejected drafts don't bias the production state.

A sourdough byte-identity gate ensures spec-decode never diverges from
the vanilla path at `K=0` — this is the production safety contract
that lets the drafter ship behind a default flag.

### 4.5 Vision

`inference/vision/` loads an mmproj GGUF (emitted by `models/vit/`)
and runs the vision tower as a Metal kernel chain identical to the
text tower's primitive set. `inference/models/qwen3vl_text/` consumes
the projected embeddings via the chat-template's `<|vision_start|>`
markers.

---

## 5. The HTTP server

`serve/api/` is a thin axum 0.7 service. The router
(`serve/api/router.rs`) is fixed at:

| Route | Handler |
|---|---|
| `GET /health` | `handlers::health` — process liveness; remains 200 when generation readiness has failed |
| `GET /readyz` | `handlers::readyz` — 200 only while generation is ready and every pooled engine worker is healthy |
| `GET /metrics` | `handlers::metrics` — Prometheus exposition |
| `GET /v1/models` | `handlers::list_models` |
| `GET /v1/models/:model_id` | `handlers::get_model` |
| `POST /v1/chat/completions` | `handlers::chat_completions` |
| `POST /v1/embeddings` | `handlers::embeddings` |
| `POST /shutdown` | `handlers::shutdown` (auth-gated) |

`AppState` (`serve/api/state.rs`) carries the engine handle, the
multi-model registry, the embedding pool, and a warmed
`KernelRegistry` for `/v1/embeddings` so handlers never pay
shader-compile latency.

Middleware (`serve/api/middleware.rs`) layers CORS, optional Bearer
auth, and request-id propagation. SSE encoding lives in
`serve/api/sse.rs`; the grammar sampler emits tool-call deltas that
the SSE encoder threads into the OpenAI-shaped stream.

Metal watchdog, ignored-submission, and device-loss errors are worker-fatal,
not slot-local. An `EngineSupervisor` outside the model worker also observes
individual Metal transaction leases, so a call that never returns still
poisons readiness and unblocks unary/SSE waiters. Qwen, Gemma, and DeepSeek
workers close admission, terminate active, detached, buffered, and
pre-close-permitted requests once, stop submitting GPU work, and make
`/readyz` return 503. The guarded SSE bridge treats a full downstream queue as
request-local cancellation instead of blocking the sole worker. `/health`
remains a liveness endpoint. OS supervision recreates the process/device
generation; hf2q does not attempt an in-process reset of a poisoned Metal
queue.

The persistent block-prefix cache (`serve/kv_persist/`) is the most
operationally interesting piece: it makes the first prefill of a
recurring system prompt nearly free across process restarts.
`block_store.rs` is the atomic-rename-under-SIGKILL surface that's
proved by a child-process kill-9 integration test
(`tests/kv_persist_writer_kill_minus_9.rs`).

---

## 6. The arch registry (`src/arch/`)

The arch registry is the **single source of truth** for everything an
architecture needs to be a first-class hf2q citizen. The struct
(`src/arch/registry.rs:56-84`):

```rust
pub struct ArchEntry {
    pub arch:                &'static str,             // GGUF arch string ("qwen35", "qwen35moe")
    pub hf_architectures:    &'static [&'static str],  // HF config.json::architectures[0]
    pub tensor_catalog:      &'static TensorCatalog,   // P4 tensor-name templates
    pub has_mtp:             bool,                     // emits blk.{L}.nextn.* tensors?
    pub has_vision:          bool,                     // --emit-vision-tower path?
    pub smoke_prompts:       &'static [&'static str],  // deterministic inputs for `hf2q smoke`
    pub ppl_corpus:          EvalCorpus,               // Decision-17 PPL eval corpus
    pub quality_thresholds:  QualityThresholds,        // per-arch quality bounds
    pub disk_floor_gb:       u32,                      // smoke preflight EXIT_INSUFFICIENT_DISK
    pub hf_repos:            &'static [&'static str],  // expected HF repos for smoke
    pub auto_override:       Option<&'static str>,     // P8 Decision-18 AutoResolver override
}
```

Per-family GGUF metadata emission and HF→GGUF tensor-name mapping live in
`src/convert/arch/`; the writer lives under `src/backends/gguf/`.

Adding a new arch is mechanical: add `src/arch/entries/<arch>.rs`
register it in `src/arch/entries/mod.rs`, transcribe the tensor
catalog with `// citation:` lines, add a smoke prompt, and the
following registry-driven tooling becomes available, while conversion and
runtime mappings still require explicit family implementations:

- `hf2q smoke --arch <arch>` (ADR-012 Decision 16 end-gate)
- `hf2q parity --arch <arch>` (ADR-009 parity validation)
- the convert pipeline (rename + metadata emission)
- the `hf2q info` inspector

The contract is "one file per arch + ~50 LOC registration + 200–400
LOC arch-specific transforms" replacing the ~1500-LOC harness rewrite
every new arch paid pre-`src/arch/`. The canonical reference is
`docs/arch-onboarding.md`.

---

## 7. Observability + operator surface

- **Logging.** `--log-format text|json` with `--log-level
  debug|info|warn|error`. JSON logs are one object per line; safe for
  Loki / Datadog ingest.
- **Serve dashboard.** `--operator-ui auto|dashboard|plain`. `auto` uses an
  alternate-screen live view only for interactive text stderr. Engine events
  enter a bounded `try_send` channel, never the inference critical path. Each
  request exposes family-local identity, slot, phase, cache/new-token split,
  prefill completion/rate/ETA, and decode rate without exposing prompt or tool
  contents. Pipes, CI, services, and JSON logging retain plain output.
- **Progress.** `indicatif` bars at convert time; suppressed when
  stderr is not a TTY.
- **Metrics.** Prometheus exposition on `GET /metrics` covering
  request latency, token throughput, KV-cache hit rate, MTL dispatch
  count and the regression-gate counters.
- **Verified remote-conversion receipt** (`src/convert/receipt.rs`): a
  successful remote-source conversion atomically binds exact source files,
  converter revision, selected quant, output identity, and peak chunk bounds.
  It is not yet the quality/performance candidate receipt defined by ADR-046.
- **Environment flags.** Investigation-only env vars are listed in
  `docs/operator-env-vars.md`. Defaults are the safe-production
  choice; opt-in flags carry a one-shot ack at startup.
- **Exit codes.**
  - `0` success.
  - `1` conversion error.
  - `2` quality threshold exceeded (ADR-009 parity).
  - `3` input / validation error.
  - `4–8` `hf2q smoke` preflight failures (per ADR-012 Decision 16:
    each failure mode gets a distinct code so CI can tell them apart).

---

## 8. Testing

`tests/` hosts integration tests; `src/**/*.rs` carries unit tests inline. The
harness leans on three patterns:

1. **Spec-citation tests.** Every K-quant codec has a hand-authored
   spec-driven test that matches `llama.cpp`'s block layout byte-for-
   byte without linking against `llama.cpp`.
2. **Round-trip gates.** Convert → reload via our own GGUF reader →
   assert tensor name + shape + dtype + (for float passes) byte
   identity.
3. **End-gate smoke prompts.** `hf2q smoke` runs the arch's canonical
   prompts and asserts the model emits the expected first / stop
   tokens. Failure modes get distinct exit codes (see §7).
4. **Isolated security spikes.** `tests/adr045_tuf_spike/` is an unpublished,
   separately locked Rust 1.88 workspace. It compares signed-update verifiers
   and crash-durable journal hypotheses without adding either candidate to the
   production dependency graph or published crate. Its wire types grant no
   production authority.
5. **Installation identity boundary.** `distribution/install_state/identity.rs`
   owns the 16 KiB canonical root-identity wire's descriptor-relative
   publication and reopen protocol. A UUID-bearing exact-prefix intent is
   full-synced before one no-replace final rename; real process-abort tests
   exercise every durability barrier. The resulting non-cloneable capability
   retains and repeats the exact root, update, lock, and identity inode/bytes.
   Metadata, artifact staging, inert extraction, and first activation must
   acquire/reopen through that capability rather than treating a path or
   copied UUID as authority. The root inventory remains open for preserved
   hf2q state, while the reserved `update/` identity inventory is bounded and
   malformed residue is retained fail-closed.
6. **Production signed-metadata boundary.** `distribution/update_auth/` uses
   `sigstore-tuf::TrustedMetadataSet` as its only library verification state
   machine behind hf2q's strict bounded profile. The stock `Updater`,
   `FileStore`, and `Repository` APIs are not imported or used, and the
   dependency's fetch/HTTP/TLS features are disabled. A retained Python-TUF
   corpus, generated with canonical key IDs and a fully hashed dependency lock,
   proves cross-implementation root rotation and lower-role authentication.
   The closed root profile independently recomputes every canonical key ID and
   accepts only exact Ed25519/Ed25519 keys with lowercase raw public bytes; key
   IDs in maps, role bindings, and signatures use exact lowercase SHA-256 form.
   Aliases, type/scheme mismatches, normalized key bytes, and extensions fail
   before the library verification state machine.
   The same bounded context now owns the structural `ChannelPointerV1`, typed
   logical/consistent-snapshot target names, and a sealed current-time replay
   that accepts one stable pointer plus complete retained release pairs. Pointer
   binding exposes only the selected pointer/manifest/archive descriptors; it
   is not generic target lookup or downgrade authority. Lock-held successor
   authentication makes versioned pairs append-only by comparing the exact
   selected predecessor and candidate before selector commit; the pointer may
   move and new complete pairs may be appended, but old descriptors cannot be
   rewritten or removed. The production module
   still owns no URL, HTTP, download, archive extraction, activation, installer,
   or CLI authority. The private sibling `distribution/update_transport/`
   maps only those sealed typed artifact descriptors to fixed Pages/GitHub origins,
   accepts at most one exact GitHub asset-CDN redirect, and verifies bounded
   pointer/manifest bytes plus an unlinked same-FD streamed archive. A one-use
   fetch capability replays the ordinary selected journal under the shared
   installation lock before and after archive I/O and rejects generation or
   clock drift. The result remains inert transport data: it grants no ZIP extraction,
   codesign, prepared-version, activation, installer, or CLI authority. The
   private `distribution/prepared_release/` boundary consumes that inert bundle,
   revalidates the same anonymous archive descriptor, and uses a bounded custom
   classic-ZIP parser before pinned `flate2` 1.1.9/`zlib-rs` proves exact raw
   Deflate consumption and an actual `StreamEnd`; pinned `zip` 7.2.0 then
   supplies an independent Stored/Deflate decode, CRC, and payload-digest view.
   It requires canonical central/local layout, exact raw inventory/order/modes,
   a byte-identical deterministic embedded manifest, and every streamed payload
   digest. It then extracts only through directory descriptors into the
   deterministic private `update/extractions/.extract-vVERSION-SHA256` stage.
   At most eight retained stages are accepted; files remain `0600` and all
   directories remain `0700`. Exact authenticated replay can reconstruct torn
   scratch in the same inode, while unexpected names/types/modes/links fail
   closed without deletion. The same shared installation lock brackets a
   current-time selected-metadata replay before and after local I/O, and the
   anonymous archive descriptor is revalidated on both sides. The sealed result
   remains inert and grants no path/FD, mode normalization,
   prepared-version publication, activation, installer, or CLI authority.
   The same private boundary contains a dormant filesystem-free bounded read-at
   validator for the exact thin arm64-ALL `MH_EXECUTE`/modern-dyld profile. It
   proves segment, section, link-edit, entry-point, deployment, and terminal
   code-signature structure. A sibling dormant macOS module pins the native
   Security.framework wrappers, builds the closed Developer ID requirement,
   validates all architectures with strict/trusted-anchor/no-network flags,
   and type-checks the bounded signing-information dictionary, certificate
   chain, timestamp, Hardened Runtime flags, entitlement absence, identifier,
   Team ID, and leaf common name. Its only policy constructor is test-only;
   the real Team ID and protected positive fixture are not yet present, so it
   grants no native-signature or prepared-version authority. Descriptor-relative
   acquisition and retained-lock integration remain in the publication
   transaction.
   The schema boundary now defines installed-version marker v2 and a narrow
   first-standalone record builder. Marker v2 carries the exact metadata-role
   versions needed to regenerate the same install-receipt-v1 transition after
   a crash, and first activation requires full equality with that derived
   receipt rather than trusting only its marker digest. Dormant marker-v1
   fixture bytes are rejected fail-closed. This
   builder remains structural output and cannot authenticate or publish a
   prepared version.
   Verifier-request metadata HTTP remains a pending production-root slice.
   Fresh-process recovery repairs the selected
   rollback floor, crash-durably discards only the derived exact unselected
   write prefix, and requires a wholly fresh transcript. An authenticated root
   transition that changes the effective timestamp or snapshot authorization
   may reset only those two floors; the sealed receipt binds the exact prior
   and final roots, while root and targets floors remain monotonic.

Benchmarks live in `benches/` and `scripts/`; the latter directory
also carries every ADR's repro runbook.

---

## 9. ADR index (where the rationale lives)

The "why" of every load-bearing design decision lives in numbered
ADRs under `docs/`. The most architecturally consequential ones:

| ADR | Subject |
|---|---|
| **ADR-004** | GGUF compatibility — what we promise to `llama.cpp` consumers. |
| **ADR-005** | Inference server — Phase 1/2/3 of the HTTP API. |
| **ADR-006** | `mlx-native` GPU backend — why Metal, why not MPS-graph. |
| **ADR-007** | TurboQuant KV cache — Hadamard-quantized K/V at 4 bits. |
| **ADR-008** | Candle divorce — sovereignty rule, single-backend invariant. |
| **ADR-009** | Reference parity + coherence recovery — the parity contract. |
| **ADR-010** | Exact batched-kernel parity — verified-kernel ledger. |
| **ADR-011** | Flash-Attention prefill — the prefill speedup. |
| **ADR-012** | Qwen35MoE conversion — and the arch-registry contract. |
| **ADR-013** | Qwen3.5 inference — per-arch inference module pattern. |
| **ADR-014** | Streaming convert pipeline + peer-parity gates (cross-arch). |
| **ADR-015** | mlx-native — general decode-path speed improvements (qwen35 + gemma). |
| **ADR-016** | coreml-native opportunistic encoder offload — P2 ViT + P3 BERT. |
| **ADR-017** | Persistent Block Prefix Cache for serve mode — `serve/kv_persist/`. |
| **ADR-018** | Uniform Model-Load UX Across Families — `hf2q serve --model PATH` invariants. |
| **ADR-019** | mlx-native Encoder Architecture — Per-Stage Fence Design. |
| **ADR-020** | Historical DWQ + mixed-precision work; superseded by ADR-046. |
| **ADR-021** | Qwen3VL ViT prelude GPU port — vision tower. |
| **ADR-022** | Kernel-coverage parity with `llama.cpp`. |
| **ADR-027** | Qwen3.5 TQ KV cache + persist family. |
| **ADR-028** | Peer parity, coherence + speed (the perf canonical). |
| **ADR-029** | Gemma4 MoE pipeline is the gap — perf investigation. |
| **ADR-030** | dFlash block-diffusion spec-decode. |
| **ADR-040** | Full-context agent slots, scheduler admission, fairness, and per-slot state. |
| **ADR-046** | Evidence-driven Apple-Silicon auto quantization and the hf2q/mlx-native ownership seam. |

Each ADR carries phase status, acceptance tests, and a "what comes
next" section. ADRs are append-only; superseded ones are linked
forward rather than deleted.

---

## 10. Where to look first

| If you want to … | Start at |
|---|---|
| Read the public CLI surface | `src/cli.rs` |
| Trace a `convert` request | `src/serve/mod.rs` → `cmd_generate` is the wrong one; `src/main.rs` dispatches `Command::Convert` into `quantize::cmd_convert`. |
| Trace a serve chat request | `src/serve/api/handlers.rs::chat_completions` → `engine*.rs` → `inference/models/<arch>/forward.rs` |
| Add a new model family | `docs/arch-onboarding.md` |
| Add a new quant variant | `src/quantize/` + register in `src/cli.rs::QuantArg` |
| Tune the KV cache | `docs/operating-kv-cache.md` + `src/serve/kv_persist/` |
| Add a new HTTP route | `src/serve/api/router.rs` + `handlers.rs` |
| Find a perf number | the ADR-028 / ADR-029 iter-logs under `docs/` |

For anything time-sensitive, prefer reading the relevant ADR over
this document. ADRs are the system of record; this file is the map.
