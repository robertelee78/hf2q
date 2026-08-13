# hf2q Shipping Contract

This document defines the public hf2q 0.1.5 product surface and the
**Unreleased next-release candidate** where explicitly marked. It also defines
the policy each environment variable is classified under. Per-variable
effects live in `docs/operator-env-vars.md`; this document sits one level above
and defines *what is supported*.

**Anything not listed in categories 1–3 below may be removed or
relocated without notice** — it is not part of the supported surface.

---

## Category 1 — Production contract and next-release candidate

What the default release binary does with **no environment variables set**.
The exact model-family surface is explicit below; no family inherits another
family's graph, cache, or scheduler contract by approximation.

- Batched `forward_prefill_batched` (default-on since ADR-028
  iter-344; per-token `forward_prefill` was 14-45× slower than peer).
  Opt out to per-token via `HF2Q_BATCHED_PREFILL=0` for parity
  diagnostics — see Category 2.
- Dense **F32** KV cache.
- Default decode (single-buffer or dual-buffer internal tuning; not
  user-configurable).
- **Auto Q8 lm_head** with exact F32 rerank, selected when
  `hidden_size % 32 == 0` **and** F16 lm_head weight > 256 MB;
  otherwise F16.
- **Public by 0.1.5; strengthened in the Unreleased 0.1.6 candidate:**
  Qwen3.5/Qwen3.6 generation and OpenAI-compatible
  serving use the shared autoregressive `qwen35`/`qwen35moe` graph by default.
  Slot-aware Qwen prefill
  is bounded and scheduler-yielding; no `HF2Q_QWEN36_AUTOREG` activation is
  required. This default contract is the plain-text unary/SSE chat surface,
  including native tools, reasoning, grammar, and retained-prefix
  continuations. SlotAware soft-token/deepstack/3D-position requests fail
  before Qwen LM scheduler/SSE admission until their own prefill and decode are scheduler-yielding;
  the historical multimodal primitive remains available only under
  SerialFifo. The separate chunk-scan prefill experiment remains Category 3.
- **Public by 0.1.5; strengthened in the Unreleased 0.1.6 candidate:** long
  plain-text Gemma SlotAware prefill advances in
  at most 4,096-token transactions, split at the stable-prefix boundary. The
  transaction publishes all configured per-layer cache cursors together.
  Compatible installed prefill states may share those 4,096 aggregate rows;
  the bound never multiplies by the number of slots. Long soft-token work
  remains fail-closed until a resumable graph is proven.
- **Public by 0.1.5; strengthened in the Unreleased 0.1.6 candidate:** DeepSeek
  meaningful cached suffixes use the same
  atomic resumable verifier transactions as cold prefill. Lopsided cold waves
  use an interactive budget of up to eight decode tokens between prefill slices
  capped at two native windows. When a filling cold cohort still has another
  cold request queued, cold-wave unary decode is deferred through its draining
  phase while any cold prefill remains and full 2,048-token transactions
  resume; unary output could not be delivered before that barrier. Streaming
  and warm decode remain responsive. Without a runnable visible decoder, the
  full prefill transaction is also restored.
  Outside a cold-cohort barrier, staggered warm work may occupy any free
  physical slot. Cancellation restores only a valid, position-consistent
  pre-request turn anchor; poisoned or inconsistent state resets fully.
  The Unreleased 0.1.6 candidate also pairs large automatic MoE gate/up
  projections through the published `mlx-native 0.10.8` routing-schedule
  primitive. Decode-sized and forced diagnostic routes remain independent;
  native microbenchmarks do not replace the exact packed hf2q hardware gates.
- A typed fatal Metal command-buffer/watchdog/ignored-submission error, or an
  independently observed transaction deadline that never returns, fails the
  affected Qwen, Gemma, or DeepSeek worker closed. Every owned reply
  terminates once; saturated SSE consumers cannot block fatal fanout; no cache
  reset or later GPU submission is permitted. `/health` remains process
  liveness while `/readyz` and new generation fail closed. OS process
  supervision, not an in-process slot reset, owns device recovery.

### Required gates before merging

Every Gemma change that could affect the forward pass or lm_head must pass
`scripts/release-check.sh`:

| Gate | Floor |
|---|---|
| `short_hello` exact-byte match vs locked llama.cpp reference | ≥ 29 bytes |
| `sourdough` common-byte-prefix with llama.cpp | ≥ 3094 bytes |
| `sliding_wrap` common-byte-prefix with locked hf2q reference | ≥ 700 bytes |
| Decode perf sanity on the sourdough prompt | ≥ 95 tok/s |

Before the Unreleased 0.1.6 Qwen cache-lifecycle corrections may ship, every
Qwen SlotAware serving change must additionally pass all of these gates
from a clean packed artifact that resolves the published, checksum-pinned
`mlx-native` dependency:

| Gate | Contract |
|---|---|
| Hosted model-free gates | Bounded 2,048-token plan, decode-first `Mixed`, cold round-robin, fatal fanout, readiness, request-boundary tests, stable fixture bytes, and the receipt-parser negative matrix pass in CI. |
| Apple-Silicon artifact gates | Cross-layer/MTP cursor-ledger coherence and transaction-boundary cancellation pass against the packed candidate; these require the native cache/model path and are not inferred from hosted scheduler tests. |
| Exact overlap | The deterministic 552-token SSE lane is enqueued immediately before the 87,972-token/347-tool lane; the short lane makes semantic progress while the long lane completes exactly 42×2,048 + 1,956 prompt tokens. |
| Disconnect | Dropping the long SSE is observed at a transaction boundary, releases the same physical slot once, and a following request succeeds. |
| Agentic four-slot gate | Required/automatic tools, unary/SSE, tool-result continuation, exact arguments, and retained-prefix reuse pass for four independent slots. Qwen uses the canonical prompt-visible `/opt/hf2q/Cargo.toml` path, a direct-tool system instruction, and an unambiguous completed-tool-result envelope; their SHA-256 identities are receipt-bound so an ephemeral package path or prompt rewrite cannot silently change the workload. |
| Native lifetime/fatal recovery | Exact-artifact hardware waves keep command-buffer and CFString populations bounded and reject every timeout or ignored-submission signature. Packed model-free fail-stop and supervisor tests inject the fatal return/dead-worker state, prove no post-fatal submission, preserve `/health` as process liveness, and require `/readyz` plus new generation to fail closed. The hardware gate does not intentionally poison Metal. |

The shared cross-family changes additionally require:

| Family | Candidate artifact gate |
|---|---|
| Gemma 4 | Fresh-versus-reused bounded output parity at the 4,096 boundary and the non-aligned 8,193-token tail; aggregate cross-slot and installed-state transaction rows remain <=4,096 at both four and eight configured slots; short-SSE/long-prefill overlap; transaction cancellation; existing agentic/cache gate; bounded native object populations. The two four-slot calibrated waves retain the default latency limits, run before the destructive 175K/120K soak, and each require a trailing 60 seconds of Nominal state plus fail-closed two-second sampling through the complete cold/cached/tool-result sequence. The transaction cap is not accepted until this passes. |
| DeepSeek-V4 | Cached suffix spanning at least three native transactions with a live decode peer; middle-transaction cancellation and recovery; lopsided cold SSE progress with terminal parking; the four-agent cold/cached/tool gate twice using the immutable `full-context-agentic-v1` fixture (SHA-256 `2c894c9e…b4ef`), exactly 6,685 rendered prompt tokens per agent, zero cold reuse, and the literal 60-second cold bounds. The ceiling remains 9.2 seconds below the current thermally valid matched llama.cpp median. Each calibrated wave starts only after at least 60 seconds of Nominal samples at five-second cadence with no hf2q/llama model runtime loaded, then remains under fail-closed two-second thermal sampling until all four atomic cold receipts exist. The request schedule is unchanged: cached work may overlap the cold tail, and the same live cache must finish cached unary/SSE, automatic tool choice, and tool-result continuation under the existing bounds. The thermal receipt binds the four cold-receipt names and hashes. The server log must prove the paired large-prefill route engaged, while semantic/tool parity and retained-prefix counts remain unchanged. |
| All three | The generic fail-stop ownership test covers origin, installed, buffered, and pre-close-permitted replies; synthetic dead workers keep `/health` live while `/readyz` and new generation fail with 503. |

---

## Category 2 — Supported operator knobs

User-facing escape hatches. Stable in the contractual sense: we will
not remove or silently change them without an ADR.

| Var | Values | Purpose |
|---|---|---|
| `HF2Q_LMHEAD_Q8` | `1`, `0`, unset | Force Q8 on, force F16, or auto-select. Escape hatch for models the auto heuristic classifies incorrectly. |
| `HF2Q_BATCHED_PREFILL` | `0`/`false`/`off`, unset | Opt out of the default batched prefill path (Category 1) back to per-token `forward_prefill`. For parity diagnostics only — per-token is 14-45× slower than peer. Default-on since ADR-028 iter-344; decoupled from the `HF2Q_UNSAFE_EXPERIMENTS` ack at that iter. The remaining `sliding_wrap` long-sequence byte-parity gap is the operator-signed deferral (2026-04-16; see ADR-010), a coherence deferral — not a runtime error. |
| `HF2Q_STREAMING_PHASE3` | `1`, unset | ADR-014 P7 iter-3 production wire-up. Routes all 4 Phase 3 quantize dispatch arms (K-quant codec direct / ImatrixAdaptive / StaticQuantizer / DwqK) and Phase 4.5 quality measurement through the streaming `LazyTensorMap` pipeline (`quantize_via_streaming_borrowed` + `measure_quality_streaming_lazy`). Output is byte-identical to the eager path — every wired arm has a per-arm byte-identity gate. Currently a TEST INTEGRATION channel, not a memory win (wedge clones bytes ~2× peak briefly); actual memory savings land when iter-3 wholesale surgery removes the upstream `materialize_all()` bridge. Default OFF; default behavior unchanged. |

---

## Category 3 — Benchmarking-only (user-triggerable but unsafe)

Documented knobs for controlled measurement. Activating them requires
an explicit acknowledgment: `HF2Q_UNSAFE_EXPERIMENTS=1`.

| Var | Unsafe-ack | Purpose |
|---|---|---|
| `HF2Q_LMHEAD_RERANK=0` | **required** | Measure raw Q8 argmax cost. Reintroduces the rare near-tiebreak flip (observed as mid-decode `<pad>` emission). |
| `HF2Q_CHUNK_SCAN_PREFILL=1` | **required** | Wave 5b iter 5 opt-in: route Qwen3.6 prefills at `seq_len > 64` through the mlx-native chunk-parallel delta-rule pipeline (`mlx_native::ops::chunk_gated_delta_rule::dispatch_chunk_gated_delta_rule_fwd`). This is a performance experiment distinct from the production autoregressive path. Decode parity ±5% (AC 5468) and walk-bar parity at pp4096+ (W-5b.3) are required before this experimental kernel can become Category 1. |

---

## Category 4 — Investigation-only (not part of product surface)

Internal scaffolding. Not listed in `docs/operator-env-vars.md` as
operator-facing; loaded through `src/debug/investigation_env.rs`
(centralized), not read ad-hoc in hot paths.

**Ack-required (known to risk correctness or runtime reliability):**

| Var | Notes |
|---|---|
| `HF2Q_F16_KV` | Known-worse KV cache representation; separate bug vs F32 path. |
| `HF2Q_SKIP_TQ_ENCODE` | Bisection scaffolding; produces garbage output. |
| `HF2Q_SKIP_TQ_SDPA` | Bisection scaffolding; produces garbage output. |

**Warn-on-activation, no ack (ineffective but safe):**

| Var | Notes |
|---|---|
| `HF2Q_GRAPH_OPT` | No measured win; reorder aborts on unannotated dispatches. |
| `HF2Q_LMHEAD_COMPARE` | Keeps both F16 and Q8 resident; inert (not wired into live decode). |
| `HF2Q_DUAL_BUFFER` | Internal perf tuning; default (3) is part of category 1. |

**Silent / read-only diagnostics (no warning, no ack):**

| Var | Notes |
|---|---|
| `HF2Q_PREFILL_DUMP`, `HF2Q_BATCHED_DUMP`, `HF2Q_BATCHED_LAYER_SCAN`, `HF2Q_DUMP_LAYERS`, `HF2Q_DUMP_BOUNDARY`, `HF2Q_DUMP_ALL_CACHE`, `HF2Q_DUMP_LAYER_DETAIL`, `HF2Q_DUMP_NORM_WEIGHT`, `HF2Q_DUMP_DIR` | Hidden-state / cache dumps; output-only, cannot affect decode. |
| `HF2Q_DUMP_RENDERED_PROMPT`, `HF2Q_DUMP_PROMPT_TOKENS` | Prompt-path diagnostics. |
| `HF2Q_MLX_TIMING`, `HF2Q_SPLIT_TIMING`, `HF2Q_MLX_KERNEL_PROFILE`, `HF2Q_MLX_PROFILE` | Timing / kernel-attribution diagnostics. |

---

## Classification rule

A toggle requires the `HF2Q_UNSAFE_EXPERIMENTS=1` acknowledgment when
it is **known to risk correctness or runtime reliability** — not
merely because it is experimental or inert. Toggles that are
ineffective-but-safe get a startup warning, not a gate. Toggles that
are read-only diagnostics get neither.

When a new toggle is introduced, classify it by this rule and register
it in `src/debug/investigation_env.rs` (for category 4) or update this
document (for categories 2–3).

---

---

## Historical ADR-012 conversion acceptance (superseded for inference)

ADR-012 originally accepted `qwen35` (dense 27B) and `qwen35moe` (MoE
35B) as convert-only classes. That historical conversion contract remains
authoritative for emitted artifacts. ADR-013, ADR-027, and ADR-040 now own the
shipped inference, cache, and SlotAware serving contracts; Qwen is no longer
convert-only.

### Acceptance gates for a converted GGUF

| Gate | Criterion |
|---|---|
| Structural validity | File begins with magic `GGUF`, version 3, tensor_count > 0, kv_count > 0 |
| Metadata completeness | Every key in the ADR-012 Decision 7 catalog is present |
| Tensor naming | Every tensor name matches the ADR-012 Decision 8 naming spec |
| llama.cpp load | `llama-cli --model out.gguf -p "Hello" -n 8` exits 0 |
| Sidecar set | `tokenizer.json`, `tokenizer_config.json`, `config.json`, `generation_config.json`, `special_tokens_map.json` (and `chat_template.jinja` when present) are byte-identical copies alongside the GGUF |
| MTP tensors (when `mtp_num_hidden_layers > 0`) | Round-trip integrity gate at `tests/convert_qwen35_mtp_roundtrip.rs` (Decision 19); 4 tensors land at `blk.{num_hidden_layers}.nextn.{enorm,hnorm,embed_tokens,eh_proj}.weight` |
| mmproj (when `--emit-vision-tower` and `vision_config` present) | Pure-Rust emitter at `src/models/vit/`; produces `mmproj-<slug>-F16.gguf` per Decision 18 with three layers of structural / round-trip / spec-driven test coverage |
| Smoke harness | `hf2q smoke --arch <qwen35\|qwen35moe> --quant q4_0` exits 0 with byte-identical transcripts across two fresh runs (Decision 16) |

### DWQ activation-based quantization for qwen35/qwen35moe

**Shipped 2026-04-25** under ADR-012 P9 + P9b (formerly listed as
"out-of-scope" pending ADR-013 P12). The convert pipeline now runs
the full two-pass activation calibration end-to-end:

  1. Emit intermediate F16 GGUF from the in-memory tensor_map
     (`backends::gguf::emit_gguf_from_tensor_map`, P9b.1).
  2. Construct `RealActivationCapture::new(intermediate_gguf, tokenizer)`
     which loads via the ADR-013 `Qwen35Model::load_from_gguf` path
     (P9b.3b).
  3. Run `quantize::dwq_activation::run_dwq_activation_calibration`
     which generates calibration tokens, runs the CPU forward pass
     through the loaded model, computes per-layer sensitivity, and
     produces a derived `MixedBitQuantizer` configured with
     activation-driven sensitive layers (P9b.3a).
  4. Final GGUF is emitted at the user-specified output path. The
     intermediate is dropped via `tempfile::TempDir` RAII (P9b.5).

No weight-space fallback for these architectures (Decision 13).

Real-model artifact production for the four end-deliverable GGUFs
(qwen35/qwen35moe × dwq46/dwq48) is gated only on environment
(HF_TOKEN + ~150 GB disk + Metal-validated llama.cpp build).

### Out-of-scope for ADR-012

- Inference coherence (sourdough gate, sliding-window parity) — ADR-013.
- MTP head **inference** (speculative decoding) — ADR-013 P14. ADR-012 P11
  ships the conversion-side tensor round-trip integrity gate; runtime
  draft/accept loops are owned by ADR-013.
- ViT compute path for the converted mmproj — ADR-005 phase 2c. ADR-012 P10
  ships the GGUF emitter; forward-pass execution is ADR-005's deliverable.

### CI integration tests

`tests/convert_qwen35_integration.rs` and
`tests/convert_qwen35moe_integration.rs` run the full convert pipeline on
synthetic tiny models (4 layers, hidden=64, 4 experts) to validate structural
correctness and sidecar behavior without downloading real model weights.

---

## Peer-parity gates (ADR-014 P10)

ADR-014 P10 lands the **8-cell peer-parity benchmark harness**
(`tests/peer_parity_gates.rs`) that compares hf2q's streaming convert
pipeline against `llama.cpp` and `mlx-lm` across the matrix locked in
ADR-014 Decision 15. Closure of ADR-014 (and final P12 doc-refresh)
gates on every cell measuring green.

### The 8-cell matrix

`GateCell { model_id, backend, calibrator_variant, peer_id,
speed_tolerance, rss_tolerance, ppl_tolerance }`, populated **verbatim**
from Decision 15 lines 575–582. The `gate_cells_match_decision_15_verbatim`
smoke test wedges this against a duplicate literal table to catch silent
edits.

| # | Model | Backend | Calibrator | Peer | Speed gate | RSS gate | PPL gate |
|---|---|---|---|---|---|---|---|
| 1 | 27B dense | GGUF | None (`q4_k_m`) | llama.cpp uncalibrated Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 2 | 27B dense | GGUF | Imatrix (`imatrix-q4_k_m`) | llama.cpp imatrix Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 3 | 27B dense | safetensors | DWQ (`dwq-4-6`) | mlx-lm DWQ | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 4 | 27B dense | GGUF | DWQ (`dwq-4-6`) | (no peer; vs hf2q current pipeline) | ≤ 1.0× | ≤ 0.50× | ≤ 1.0× |
| 5 | apex MoE | GGUF | None (`q4_k_m`) | llama.cpp uncalibrated Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 6 | apex MoE | GGUF | Imatrix (`imatrix-q4_k_m`) | llama.cpp imatrix Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 7 | apex MoE | safetensors | DWQ (`dwq-4-6`) | mlx-lm DWQ | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 8 | apex MoE | GGUF | DWQ (`dwq-4-6`) | (no peer; vs hf2q current pipeline) | ≤ 1.0× | ≤ 0.50× | ≤ 1.0× |

**Tolerance triple semantics:**

- **Speed**: `hf2q_wall ≤ tolerance × peer_wall`. The 1.10× headroom on
  rows 1–3 and 5–7 is the documented "no-regression-vs-peer" budget for
  ADR-014's streaming pipeline.
- **RSS**: `hf2q_peak_rss ≤ tolerance × peer_peak_rss`. Rows 4 and 8
  encode the central correctness/sanity claim of the ADR — streaming
  halves peak resident vs the pre-streaming pipeline (≤ 0.50×).
- **PPL**: `hf2q_ppl ≤ tolerance × peer_ppl`. Wikitext-2 perplexity at
  1.02× headroom on the cross-peer rows; rows 4 and 8 require strict
  PPL parity (≤ 1.0×) against the hf2q current pipeline.

### Verdict surface

```rust
pub enum Verdict {
    Pass,
    Fail { reason: String },
    NotMeasured { reason: String },
}
```

`NotMeasured` is the canonical outcome for cells that cannot run yet
(no real model on disk, peer binary missing, hf2q-side driver pending).
**Distinct from `Pass` and `Fail` so the markdown table surfaces the
deferred state honestly** — no fake-green, no fake-red. The
`reason` field carries the disqualifier so the table reader sees *why*
the cell was deferred.

### Markdown table emitter

`emit_markdown_table(results, hardware_fingerprint, sha) -> String` is a
**pure function** (no I/O) that produces the full markdown document for
a slice of `CellResult`s. Header columns (14 total):

```
Model | Backend | Calibrator | Peer | hf2q wall (s) | peer wall (s) |
speed ratio | hf2q RSS (B) | peer RSS (B) | RSS ratio | Verdict |
hf2q PPL | peer PPL | PPL ratio
```

PPL cells render `f32` to 4 decimal places; un-measured PPLs render as
the em-dash `—` so the deferred state is visually distinct from a real
`0.0000`.

`write_results_to_dated_doc(results, hardware, sha, today, docs_dir)`
writes the table to `docs/peer-parity-results-<YYYY-MM-DD>.md`. The
write is only callable from `#[ignore]`-gated cells so the always-on
test suite does not pollute `docs/`.

### Smoke-vs-full corpus auto-pick (P10 iter-3)

The PPL corpus loader (`load_corpus_tokens` in
`tests/peer_parity_gates.rs`) **auto-picks** between two corpora:

| Corpus | File | Size | When used |
|---|---|---|---|
| **Smoke** | `tests/fixtures/ppl-corpus/wikitext2-smoke.tokens` | 2 KB (512 little-endian u32 tokens, deterministic ramp `(i*17+3) % 32000`) | Committed to the repo; used when the full corpus is missing or fails validation. Default for every `cargo test` run. |
| **Full** | `tests/fixtures/ppl-corpus/wikitext2-full.tokens` | ~700 KB+ on disk (≥ 280 000 tokens) | Generated by `scripts/fetch_wikitext2.sh` (Stephen Merity / Salesforce wikitext-2 raw v1; SHA-256 locked). Gitignored. Used by P11 for parity-grade PPL. |

The fetcher refuses corrupt downloads (SHA mismatch) and undersized
output (`< 280 000` tokens or `< 1 MiB`). The loader logs the selected
corpus to stderr so a CI run that silently falls back to smoke is
diagnosable from the build log alone.

### P11 hardware gate

All 8 cells are `#[ignore]`-gated as of P10 with reasons like:

```
#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk + Qwen35Model::load_from_gguf for Variant::Moe"]
```

P11 swaps the sentinel `/var/empty/...gguf` paths in
`hf2q_model_path(&cell)` for real model artefacts staged on disk and
runs the harness via `scripts/peer_parity_run.sh` (1 warmup discarded
→ 60 s thermal cooldown → 3 timed runs each wrapped in
`/usr/bin/time -l`; the harness reads the median of 3).

### Calibrator cross-validation gate (P6 close iter-1)

Independent of the speed/RSS/PPL parity matrix, ADR-014 P6 ships a
**byte-equivalent (with documented float tolerance) cross-validation
gate** for the imatrix calibrator:
`src/calibrate/imatrix_xvalidate.rs::cross_validate_imatrix_gguf`.

Given two GGUF v3 imatrix files (the schema landed by llama.cpp
PR #9400 / commit `90083283` / 2025-07-19), the comparator:

1. Loads both via `ImatrixCollector::load_imatrix_gguf` — the same
   reader the runtime quantize loop uses (load-bearing: a regression
   in the loader breaks both production and the gate, not just one).
2. Diffs the tensor-name set; tensors present in only one side surface
   as `tensors_in_a_only` / `tensors_in_b_only`.
3. For every shared tensor, computes element-wise
   `max(abs(a - b))` and `max(abs(a - b) / max(abs(a), abs(b), 1e-12))`
   over `in_sum2` (the importance vector).
4. Asserts `counts` arrays are byte-equal (counts are exact integer
   token counts; no float-precision leeway).
5. Returns an `XValidationReport` with `is_pass()` predicate.

**Tolerance defaults:** `abs_tolerance = 1e-3`, `rel_tolerance = 1e-2`.
Justification (P7 iter-3x/3y dequant round-trip RMSE bounds): Q4_K
≤ 0.05, Q5_K ≤ 0.025, Q6_K ≤ 0.012 — the gate would still catch a
Q6_K-level regression even if the imatrix port introduced noise at the
Q4_K-precision level. Callers wanting tighter or looser tolerances
pass them explicitly.

The `#[ignore]`-gated cell `xvalidation_vs_llama_imatrix_qwen35_smoke`
(at `tests/imatrix_xvalidation.rs`) wires this comparator against the
external `llama-imatrix` binary on a Qwen3.5-0.6B fixture; that cell
is the pre-P11 close gate proving hf2q's pure-Rust port produces
per-tensor `in_sum2` + `counts` numerically equivalent to llama.cpp's
C++ implementation.

---

## Known out-of-scope

These are deliberately not part of any category:

- Byte-identical batched-prefill parity with llama.cpp at the ~752-byte
  `sliding_wrap` level (see `docs/ADR-010-exact-batched-kernel-parity.md`;
  deferred).
- Qwen SlotAware soft-token, deepstack, and 3D-position generation. Those
  request shapes fail before Qwen LM scheduler/SSE admission; Qwen3-VL and
  the historical SerialFifo multimodal primitive have separate contracts.
- In-process recovery after a fatal Metal command-buffer/watchdog/ignored-
  submission failure or an expired non-returning transaction. The worker and
  HTTP surfaces fail closed, but an OS supervisor must recreate the
  process/device generation.

---

## References

- `docs/operator-env-vars.md` — per-variable effects and defaults.
- `docs/ADR-009-reference-parity-and-coherence-recovery.md` — why
  F32-KV is the default, and the original per-token prefill baseline
  (since superseded as the default by ADR-028 iter-344).
- `docs/ADR-010-exact-batched-kernel-parity.md` — why batched-prefill
  is now the default and why its `sliding_wrap` byte-parity is deferred.
- `docs/ADR-028-peer-parity-coherence-and-speed.md` — iter-344
  default-flip of batched prefill and ack-decoupling.
- `docs/ADR-012-qwen35moe-conversion.md` — qwen35/qwen35moe convert spec.
- `docs/ADR-014-streaming-convert-pipeline.md` — streaming pipeline +
  Decision-15 peer-parity gate matrix (the source of truth for the
  8-cell table above).
- `docs/converting-qwen35.md` — canonical convert commands for Qwen3.5/3.6.
- `docs/converting-a-model.md` — generic convert reference including Gemma.
- `docs/calibrator-onboarding.md` — developer guide for adding new
  Calibrator implementations (Imatrix, DWQ, future ones).
- `scripts/release-check.sh` — the reproducible gate runner.
- `scripts/peer_parity_run.sh` — cold-cache protocol for the peer-parity
  harness (1 warmup discarded → 60 s thermal cooldown → 3 timed runs;
  median read).
- `scripts/fetch_wikitext2.sh` — full wikitext-2 corpus fetcher for the
  PPL gate (smoke fixture is committed; full corpus is gitignored).
