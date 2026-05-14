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
   quantization algorithm, and emit either a GGUF file (single binary,
   `llama.cpp` consumers) or an mlx-lm-shaped safetensors directory
   (mlx-lm, Candle, vLLM, hf2q's own serve loader).

2. **Serve / Generate** — load a GGUF, run prefill + decode on the
   Apple-Silicon GPU through the `mlx-native` crate, and expose
   OpenAI-compatible HTTP endpoints (chat completions, embeddings,
   models) with SSE streaming, tool calls, vision, grammar-constrained
   sampling, and a persistent block-prefix KV cache.

Both halves share the same internal IR (`src/ir/`) and the same arch
registry (`src/arch/`), so adding a new model family is one ADR + one
module under `src/models/` and `src/inference/models/`.

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
├── src/report.rs        machine-readable convert-result report
├── src/gguf_patch.rs    metadata-only GGUF rewriter (no tensor I/O)
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
├── src/models/          per-arch conversion (tensor rename, MoE merge, metadata)
│   ├── qwen35/          ADR-012 Qwen 3.5 / 3.6 convert (dense + MoE)
│   └── vit/             ADR-012 P10 pure-Rust mmproj (ViT) emitter
│
├── src/quantize/        every Q-format codec
│   ├── q_legacy.rs      Q4_0 / Q5_0 / Q8_0 (`llama.cpp` block formats)
│   ├── k_quant.rs       K-quant common super-block math
│   ├── k_quant_codec.rs Q2_K…Q6_K block encoders / decoders
│   ├── k_quant_codec_quantizer.rs  the bind-it-all-together quantizer
│   ├── variant_quantizer.rs        K-quant variant dispatch
│   ├── mixed.rs                    dynamic-quant-{4-6,4-8,6-8,2-8}
│   ├── layer_mix.rs                per-layer sensitivity → bit choice
│   ├── dwq_k_quantizer.rs          DWQ-overlay K-quant adapter
│   └── static_quant.rs             dynamic→static plan freezer
│
├── src/calibrate/       activation capture + DWQ training (ADR-020)
│   ├── imatrix.rs              imatrix capture
│   ├── imatrix_calibrator.rs   imatrix → sensitivity prior
│   ├── imatrix_xvalidate.rs    leave-one-out validation
│   ├── dwq.rs                  per-Linear DWQ training entry
│   ├── dwq_loop.rs             multi-Linear orchestration
│   ├── dwq_e2e.rs              end-to-end DWQ run + bookkeeping
│   ├── dwq_targets.rs          FP32-teacher target derivation
│   ├── dwq_activation.rs       activation cache for hybrid arches
│   ├── dwq_calibrator.rs       teacher provider + Adam loop
│   ├── dwq_benchmark.rs        benchmark harness
│   ├── adam.rs                 Adam optimizer (CPU + GPU variants)
│   ├── autograd*.rs            forward-mode-tape autograd primitives
│   ├── gguf_teacher.rs         GGUF as FP32 teacher provider
│   ├── hf_safetensors_teacher.rs HF safetensors teacher
│   ├── qdq_gpu.rs              quantize-dequantize round-trip on GPU
│   ├── fd_sensitivity.rs       finite-difference sensitivity
│   ├── sensitivity.rs          Hessian-trace sensitivity prior
│   ├── sensitivity_comparison.rs cross-method comparison
│   ├── dynamic_quant{_gpu}.rs  dynamic-quant kernels
│   ├── calibrator.rs           top-level calibrate orchestration
│   ├── calibration_batcher.rs  windowed-corpus tokenizer
│   ├── cache.rs                calibrate-side cache
│   ├── apex.rs                 APEX adaptive-quant planner
│   ├── mlx_safetensors_loader.rs mlx-lm overlay loader
│   ├── qwen35_*.rs             qwen35-specific per-layer DWQ ops
│   └── …                       (~35 files; this is the dense quarter)
│
├── src/backends/        output writers — GGUF + mlx-lm safetensors
│   ├── gguf.rs                GGUF emitter (header, tensor stream, sidecars)
│   ├── safetensors_out.rs     mlx-lm safetensors directory writer
│   ├── chat_templates.rs      per-arch chat template lookup
│   └── chat_templates/        canonical Jinja2 templates
│
├── src/quality/         post-convert quality measurement
│   ├── cosine_sim.rs          weight-level cosine similarity
│   ├── kl_divergence.rs       output-logit KL
│   ├── perplexity.rs          PPL on a corpus
│   ├── ppl_driver.rs          forward-pass driver for PPL
│   ├── kernel_parity.rs       kernel-equivalence proofs
│   └── regression.rs          regression-gate accountant
│
├── src/intelligence/    hardware probe + auto-quant + RuVector
│   ├── hardware.rs            chip detection, MTL device, memory probe
│   ├── fingerprint.rs         stable model fingerprint (for cache keys)
│   ├── auto_quant.rs          auto-mode planner
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
                  │  src/models/<arch>/   rename + MoE merge│
                  └────────────────────┬────────────────────┘
                                       │ canonical-named TensorMap
                                       v
                  ┌─────────────────────────────────────────┐
                  │  src/quantize/   pick codec by --quant   │
                  │      [optional] src/calibrate/   DWQ    │
                  │      [optional] imatrix sensitivity     │
                  └────────────────────┬────────────────────┘
                                       │ QuantizedModel
                                       v
                  ┌─────────────────────────────────────────┐
                  │  src/backends/                          │
                  │      gguf.rs  →  *.gguf                 │
                  │      safetensors_out.rs  →  mlx-lm dir  │
                  └────────────────────┬────────────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  │  src/quality/   cosine + (optional) PPL │
                  │  src/report.rs   structured convert log  │
                  └─────────────────────────────────────────┘
```

Streaming is real: `safetensors` shards are mmap'd, tensors are
quantized in rayon-parallel chunks (`src/quantize/`), and the writer
sinks blocks to disk as soon as they're ready. Memory ceiling is set
by the largest individual tensor + its quant scratch — typically a few
GB even for 35 B-parameter models.

### Quantization families

| Family | Where it lives | Notes |
|---|---|---|
| **Legacy block** (`q4_0`, `q5_0`, `q8_0`) | `src/quantize/q_legacy.rs` | Per-block 32-element row layout. Single scale; no min. |
| **K-quant** (`q2_k`…`q6_k`, `q4_k_m`, `q5_k_m`, …) | `src/quantize/k_quant*.rs`, `variant_quantizer.rs` | 256-element super-blocks with per-row scale + min. Codec maths verified bit-identical to `llama.cpp` via round-trip + spec-citation tests. |
| **Imatrix-K-quant** (`imatrix-q4_k_m`, `imatrix-adaptive`, …) | `src/calibrate/imatrix*.rs` + same codecs | Activation-imatrix-weighted residual minimization. Per-layer sensitivity threshold drives bit assignment. |
| **Dynamic / DWQ-mixed** (`dynamic-quant-{4-6,4-8,6-8,2-8}`) | `src/quantize/mixed.rs`, `layer_mix.rs` | Per-layer mixed-bit; `static_quant.rs` freezes a dynamic plan. |
| **DWQ-overlay** (`dwq-4`, `hf2q dwq-train`) | `src/calibrate/dwq*.rs`, `src/quantize/dwq_k_quantizer.rs` | Trains `<stem>.scales` + `<stem>.biases` per-Linear over an FP32 teacher (GGUF or HF safetensors) with Adam to minimize KL. Emits an mlx-format safetensors overlay. |

The full menu, gated by per-arch availability, lives in
`docs/converting-a-model.md`. The decommissioned variants (e.g. `apex`,
`dwq-mixed-*`) emit structured "did you mean" errors via
`src/cli.rs:map_deleted_quant`.

---

## 4. The inference pipeline

### 4.1 Load

`serve::cmd_serve` / `serve::cmd_generate` →

1. **Header read** (`serve/header.rs`) validates the GGUF magic +
   producer fingerprint (`serve/provenance.rs`) against the arch
   registry.
2. **Arch dispatch** picks an engine wrapper under
   `serve/api/engine*.rs` (Gemma 4, Qwen 3.5, Qwen 3-VL).
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
  HEAD-default for Gemma 4 + Qwen 3.5. Was the single largest serve
  speedup in the project (35× over per-token at `pp1024` on Gemma 4)
  when the HTTP path was wired in ADR-028 Phase 15. Flash-Attention
  (ADR-011) lives in this path.

### 4.3 Decode

`serve/forward_mlx.rs` is the per-token decode hot loop. It:

1. Runs the per-arch graph (`inference/models/<arch>/`) through
   `mlx-native` MTL dispatches.
2. Reads / writes the KV cache through TurboQuant
   (`docs/operating-kv-cache.md`, ADR-007): K and V are Hadamard-
   quantized down to ≈4 bits with a per-block scale, giving
   ~3.94× memory advantage vs an F16 KV cache at negligible quality
   loss. The TQ-HB encode is fused into the dense KV-store path; the
   on-load path lazily promotes from the persisted block store.
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
| `GET /health` | `handlers::health` — always 200 once warm |
| `GET /readyz` | `handlers::readyz` — 200 once the engine is loaded |
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

The persistent block-prefix cache (`serve/kv_persist/`) is the most
operationally interesting piece: it makes the first prefill of a
recurring system prompt nearly free across process restarts.
`block_store.rs` is the atomic-rename-under-SIGKILL surface that's
proved by a child-process kill-9 integration test
(`tests/kv_persist_writer_kill_minus_9.rs`).

---

## 6. The arch registry (`src/arch/`)

The arch registry is the **single source of truth** for everything an
architecture needs to be a first-class hf2q citizen:

```rust
pub struct ArchEntry {
    pub arch:           ArchId,                  // "qwen35", "qwen35moe", …
    pub catalog:        TensorCatalog,            // expected tensor names + dtypes
    pub quality:        QualityThresholds,        // KL / PPL ceilings
    pub eval_corpus:    EvalCorpus,              // smoke + PPL corpus
    pub smoke_prompts:  &'static [SmokePrompt],   // end-gate prompts + expected stops
    pub has_mtp:        bool,                    // multi-token prediction?
    pub has_vision:     bool,                    // mmproj sidecar?
    pub metadata:       MetadataEmitter,         // arch-specific GGUF metadata
    pub layer_map:      LayerMapper,             // HF → GGUF tensor name fn
}
```

Adding a new arch is mechanical: add `src/arch/entries/<arch>.rs`
register it in `src/arch/entries/mod.rs`, transcribe the tensor
catalog with `// citation:` lines, add a smoke prompt, and the
following tooling all "just works":

- `hf2q smoke --arch <arch>` (ADR-012 Decision 16 end-gate)
- `hf2q parity --arch <arch>` (ADR-009 parity validation)
- `hf2q dwq-train` (ADR-020) cohort sensitivity priors
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
- **Progress.** `indicatif` bars at convert time; suppressed when
  stderr is not a TTY.
- **Metrics.** Prometheus exposition on `GET /metrics` covering
  request latency, token throughput, KV-cache hit rate, MTL dispatch
  count and the regression-gate counters.
- **Structured convert report** (`src/report.rs`): every convert run
  emits a machine-readable JSON manifest pointing at the input
  hashes, the arch entry version, the chosen quant, per-tensor
  drift summaries, and the wall-clock breakdown.
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

`tests/` (77 files) hosts integration tests; `src/**/*.rs` carries
unit tests inline. The harness leans on three patterns:

1. **Spec-citation tests.** Every K-quant codec has a hand-authored
   spec-driven test that matches `llama.cpp`'s block layout byte-for-
   byte without linking against `llama.cpp`.
2. **Round-trip gates.** Convert → reload via our own GGUF reader →
   assert tensor name + shape + dtype + (for float passes) byte
   identity.
3. **End-gate smoke prompts.** `hf2q smoke` runs the arch's canonical
   prompts and asserts the model emits the expected first / stop
   tokens. Failure modes get distinct exit codes (see §7).

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
| **ADR-014** | Streaming convert pipeline — memory-bounded conversion. |
| **ADR-015** | Single-CB decode — the encoder-worker design. |
| **ADR-016** | mlx-vs-CoreML strategic comparison — backend decision dossier. |
| **ADR-017** | Persistent block-prefix cache — `serve/kv_persist/`. |
| **ADR-018** | Uniform model-load UX — `hf2q serve --model PATH` invariants. |
| **ADR-019** | mlx-native encoder architecture — the worker model. |
| **ADR-020** | DWQ streaming calibration — `hf2q dwq-train`. |
| **ADR-021** | Qwen3VL ViT prelude GPU port — vision tower. |
| **ADR-022** | Kernel-coverage parity with `llama.cpp`. |
| **ADR-027** | Qwen3.5 TQ KV cache + persist family. |
| **ADR-028** | Peer parity, coherence + speed (the perf canonical). |
| **ADR-029** | Gemma4 MoE pipeline is the gap — perf investigation. |
| **ADR-030** | dFlash block-diffusion spec-decode. |

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
