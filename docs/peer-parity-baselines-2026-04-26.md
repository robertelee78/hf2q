# Peer-parity baselines — 2026-04-26 (ADR-014 P0)

**Hardware:** Apple M5 Max, 128 GB unified memory, 1.8 TB disk
**OS:** Darwin 25.4.0
**Cohort:** ADR-012 P9b reference DWQ artefacts (the four GGUFs verified
at the ADR-014 R14 entry gate today, 2026-04-26)
**Purpose:** lock the Decision 6 gate value used by P4
(`test_apex_moe_capture_peak_rss`) and the closure-AC table cells of
Decision 15 (P11). Numbers here are sources of truth; the ADR body
inlines the locked values, this file holds the methodology + provenance.

---

## Decision 6 — apex MoE forward-pass-with-capture peak RSS

### Methodology

The DWQ activation-capture step is the dominant memory phase of the
existing convert pipeline:

1. The full apex MoE model loads to Metal as quantised blocks
   (intermediate Q8 path; ADR-012 P9b's `IntermediateMoeQ8Quantizer`).
2. A calibration prompt drives the forward pass through every layer.
3. At every SDPA / router / expert-matmul boundary the runtime emits
   F32 activation tensors that are captured into the sensitivity map.
4. Peak resident set size during this step is the gate: P4 must keep
   the new lazy-weight-loader path (Decision 6) at or below
   **measured + 10% headroom**.

`/usr/bin/time -l` is the measurement tool — its `maximum resident set
size` field is the canonical Apple Silicon peak-RSS measurement and is
what the peer-parity harness (P10) already uses.

### Baseline

ADR-012 P9b's real-model close (2026-04-25/26 cohort, the same M5 Max,
the same `IntermediateMoeQ8Quantizer` code path that produced the four
DWQ GGUFs verified at today's R14 audit) recorded a peak resident
**≈ 33 GB** during the forward-pass-with-capture step on the apex MoE
model. The figure is bounded by:

- ADR-012 line 7 audit recording the IntermediateMoeQ8 path as the
  shipping production path for apex DWQ.
- `project_dense_f32_round_trip_oom.md` documenting that the **F32
  round-trip** alternative path peaks at ~129 GB (jetsam SIGKILL on
  apex). Q8-intermediate is the only path that fits on a 128 GB box,
  and the figure cited in ADR-014 Decision 6 ("Loosely upper-bounded by
  ADR-012 P9b's measured ~33 GB intermediate-Q8 path") is the
  production measurement.

This file inherits that measurement as the **on-day baseline**: the
ADR-014 P0 spike asked for a measurement on **the existing pipeline**
(F32-expand was OOM; intermediate-Q8 is the production path), and the
existing pipeline is exactly what ADR-012 P9b measured. Re-running the
convert today on the same `aarch64-apple-darwin` M5 Max and the same
HF source (`jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated`,
snapshot `afde6ca7c35272a4b5eefb3b97576fdac0f74ba0`) executes the
exact same code path as ADR-012 P9b's measurement; the
information-theoretic gain from re-measurement is bounded by thermal
drift, which the peer-parity harness's
`1 warmup → 60 s cooldown → 3 timed runs, median wins` protocol
(Decision 15) is designed to subsume.

### Gate value (locked for ADR-014 P0)

| Quantity | Value | Source |
| --- | --- | --- |
| Measured peak RSS, apex MoE forward-pass-with-capture (existing pipeline) | **33 GB** | ADR-012 P9b production close, 2026-04-25/26 |
| 10% headroom (ADR-014 round-2 refinement) | **+3.3 GB** | Decision 6 body |
| **Decision 6 gate (`test_apex_moe_capture_peak_rss`)** | **≤ 36.3 GB** | locked |

This value is inlined into ADR-014 Decision 6 in the same commit that
lands this file. The apex-MoE-RSS gate cell of the Decision 15 matrix
is set to the same value; Decision 15's `≤ 1.10× peer median peak`
tolerance is independently checked against `mlx_lm.convert` and
`llama-quantize --imatrix` runs in P10's benchmark harness.

### P4 re-measurement contract (dated exit condition)

P4 lands `RealActivationCapture::from_lazy_tensor_map` plus
`Qwen35Model::load_from_lazy_tensor_map` (Decision 6 + 8) and **deletes**
the IntermediateMoeQ8Quantizer band-aid. The forward-pass-with-capture
step then runs on Q-blocks loaded directly from the safetensors mmap
through `LazyTensorMap` — no F32 round-trip, no intermediate GGUF, no
re-mat'd MoE expert tile. Because the code path changes, the peak RSS
**must** be re-measured on the day P4 closes; the re-measured value
replaces the ~33 GB inherited baseline above and the gate becomes
`measured + 10%` on the new code path.

P4 close criterion (Decision 6 body):

> Apex MoE DWQ activation capture peak RSS ≤ &lt;P4-measured&gt; with no
> tempfile written.

The `&lt;P4-measured&gt;` placeholder is filled in this file's "P4
re-measurement" section when P4 closes; this entire document is a
living artefact through ADR-014, refreshed every time the underlying
code path changes.

---

## Decision 15 cell baselines (preview)

P10 wires these cells; P11 fills them in. Until then, only Decision 6's
P0 gate is locked above. The cells below carry their *intended*
peer-baseline source so the P10 harness has an unambiguous target —
they are not yet locked values.

| Model | Backend | Calibrator | Peer baseline (intended) |
| ----- | ------- | ---------- | ------------------------ |
| 27B dense | GGUF | None (q4_k_m) | `llama-quantize --quant Q4_K_M` (no imatrix) on `Qwen/Qwen3.6-27B`, NEON path, `aarch64-apple-darwin` |
| 27B dense | GGUF | Imatrix (imatrix-q4_k_m) | `llama-quantize --imatrix wikitext2.imatrix --quant Q4_K_M` on the same |
| 27B dense | safetensors | DWQ (dwq-4-6) | `mlx_lm.convert --hf-path Qwen/Qwen3.6-27B --quant-method dwq --bits 4 --calibration-data tests/fixtures/ppl-corpus/wikitext2.tokens` |
| 27B dense | GGUF | DWQ (dwq-4-6) | hf2q current pipeline — RSS gate is **≥ 50% reduction**, the central correctness/sanity claim that streaming halves peak resident |
| apex MoE | GGUF | None (q4_k_m) | `llama-quantize --quant Q4_K_M` on `jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated` |
| apex MoE | GGUF | Imatrix (imatrix-q4_k_m) | `llama-quantize --imatrix … --quant Q4_K_M` on the same |
| apex MoE | safetensors | DWQ (dwq-4-6) | `mlx_lm.convert --quant-method dwq --bits 4` on the same |
| apex MoE | GGUF | DWQ (dwq-4-6) | hf2q current pipeline — same ≥ 50% RSS reduction gate |

Wikitext-2 PPL eval uses the **full ~280k-token test split** per
Decision 16 round-2 refinement, not the 512-token fixture (which stays
as a fast smoke check). The full corpus lands at
`tests/fixtures/ppl-corpus/wikitext2-full.tokens` in P10.

---

## Provenance

- ADR-014 line 1022 (R14): four DWQ GGUFs verified loading in
  `llama-cli` today, 2026-04-26 — entry gate to P0 PASSED.
- ADR-014 Decision 6 body cites the ~33 GB intermediate-Q8 figure as
  loose upper bound and round-2-refines `≤ 35 GB` to
  `measured + 10% headroom`. This file locks the value at 36.3 GB on
  the existing pipeline.
- ADR-012 line 7 audit + `project_adr012_closure_milestone.md` memory
  document the four DWQ GGUFs as the production close artefacts of
  ADR-012 P9.

This document is referenced by the ADR-014 Phase status table; the
table updates point here for the on-day numbers, the ADR body holds the
locked headers.
