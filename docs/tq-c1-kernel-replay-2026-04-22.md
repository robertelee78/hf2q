# TQ C-1 Kernel Replay — ADR-007 C-4 E1 Branch

**Date:** 2026-04-22 (revised post-Codex review, same day)
**Session:** `cfa-20260422-C1-kernel-replay`
**Verdict:** VERIFICATION_BLOCKED (two independent harness defects)
**Report author:** Worker 4 (tech-writer-reporter, `cfa-20260422-C1-kernel-replay-reporter`); Queen reconciliation revision after Codex review
**Prior sessions:** C-0 audit (`docs/tq-c0-audit-2026-04-21.md`) · C-0b localization (`docs/tq-c0b-localize-2026-04-21.md`)
**ADR reference:** `docs/ADR-007-turboquant-kv-cache.md` — C-4 E1 branch

---

## 1. Executive Summary

**Verdict: VERIFICATION_BLOCKED. Two independent harness defects, not one.** No H1/H2/H4
conclusion can be drawn from any of the four runs.

### Defect #1 — 22-row vs 23-row state mismatch (analyst's catch, verified)

The harness (mlx-native@`9a4ca61`) replays `flash_attn_vec_tq` with `kv_seq_len=22` — the
end-of-prefill TQ compact cache state. However, production SDPA at decode step 1 runs with
`kv_seq_len=23`: `forward_mlx.rs:1009–1015` increments `kv_seq_len` at the top of the per-token
call, and `forward_mlx.rs:1226–1243` writes the 23rd K/V row into packed cache, both occurring
before the `flash_attn_vec_tq` dispatch at line 1464. The 23rd row — the current decode token's
K/V, the position Q attends to most strongly — is absent from all four harness runs.

### Defect #2 — Missing memory barriers between concurrent FWHT and SDPA dispatches (Codex's catch, verified)

`MlxDevice::command_encoder()` creates a compute encoder with `MTLDispatchType::Concurrent`
(`/opt/mlx-native/src/encoder.rs:404–409`). Dispatches in that encoder "can execute concurrently
unless separated by a barrier" (op. cit. lines 430–437). Production brackets the FWHT-Q → TQ-SDPA
→ FWHT-output sequence with three explicit `s.barrier_between()` calls
(`forward_mlx.rs:1429–1431`, `1441–1446`, `1477–1480`). The harness at lines 589–660 dispatches
all three (or two, in Variation B) kernels into one concurrent encoder with **zero** barriers
(`grep memory_barrier|barrier_between examples/tq_kernel_replay.rs` returns no hits). The
FWHT-on-Q may run concurrently with, or after, the SDPA dispatch that is supposed to consume its
output; likewise the FWHT-on-output may race the SDPA write. **This is almost certainly why A = B
bit-identically**: the FWHT pair does not reliably sequence with the kernel, so toggling it
cannot change the kernel's observed Q input.

### What this means for the four runs

All four runs are confounded by at least Defect #2, and the 22-row state makes them additionally
unattributable. The table of numbers (Section 3) is preserved for record, but:

- **A = B (bit-identical): NOT a finding about FWHT semantics.** Confounded by missing barriers;
  the FWHT dispatch may be racing the SDPA on both paths. Does not rule in or rule out H2.
- **C ≈ A (Δnrmse 0.001): NOT a finding about encoding locus.** Variation C is degenerate — it
  dequantizes then requantizes the same packed bytes via the same `nibble_quantize` used at
  dump time (up to numerical rounding), so at 22 rows it essentially retests A. Does not rule in
  or rule out H1/H4.
- **A_canary = A (bit-identical): confirms loop bounds, does NOT validate mask semantics.**
  `flash_attn_vec_tq.metal:262–266` breaks out of the main loop at `ic >= kv_seq_len`, and the
  inner K/V loops `continue` when `kv_pos >= kv_seq_len` (lines 297–302, 357–359). Positions
  22..1023 are never dereferenced, so setting their norms to 1e9 cannot affect output regardless
  of whether the chunk-level mask (lines 270–281) is correct. H4 mask-leak is **not** ruled out.
- **Raw output `.bin` files not persisted.** "Bit-identical A = B" is inferred from JSON summary
  metrics only (nrmse, max_abs_diff, per-head max); true byte-level identity was never
  demonstrated. The harness's binary-write path at lines 676–784 is a no-op comment. Even the
  metric-level equality is confounded by Defect #2.

### Retraction: the C-0 L=0/P=1 reframing claim

The previous revision of this report claimed that C-0's L=0/P=1 dense-vs-TQ comparison was
"dense-GPU (kv_seq_len=23) vs TQ-GPU (kv_seq_len=22) — apples-to-oranges". **That claim is
retracted as a new methodology error.** In production, both the dense and TQ SDPA dispatches
consume `kv_seq_len` that has already been incremented to 23 at `forward_mlx.rs:1012–1014` before
any per-layer work; both paths therefore see 23 rows at SDPA time. C-0's dense vs TQ comparisons
were row-count-matched. C-0's "all 30 layers violate bound" and "L=0 within-bound" findings are
**not** invalidated by any new observation in C-1. The harness (which is at 22 rows) is the
anomaly, not C-0.

### The unblock for C-1

Two defects must be fixed, not one:

1. **Add barriers** (or mirror `s.barrier_between`) between forward-FWHT(Q) → `flash_attn_vec_tq`
   → inverse-FWHT(output) in the harness, matching production.
2. **Re-dump the TQ-packed cache post-`hadamard_quantize_kv`, pre-SDPA**
   (`forward_mlx.rs:1226–1243` → `1464`), giving a 23-row compact cache; update manifest
   `kv_seq_len: 23`; rebuild padded inputs.
3. **Replace Variation C with a real dense control**: run `flash_attn_vec` (not `_tq`) on the
   same 23-row natural-basis K/V that CPU-dequant produces. This is the actual H4 test (dense
   kernel vs TQ kernel on identical mathematical inputs), whereas the current Variation C is a
   no-op round-trip on the quantized bytes.
4. **Persist raw output `.bin` files** for A/B/C/A_canary so "bit-identical" claims can be made
   at the byte level, not inferred from summary metrics.
5. **Redesign the canary** to mutate values at an IN-range position (e.g. overwrite K_norms[10]
   and re-run) to test whether the kernel correctly reads that row, rather than mutating
   out-of-loop-range slots.

---

## 2. Methodology

### 2.1 Branches and commits

| Repo | Commit | Description |
|---|---|---|
| hf2q (main) | `f519573` | ADR-007 docs + C-0b instrumentation; HEAD at session start |
| mlx-native | `9a4ca61` | `examples/tq_kernel_replay.rs` — the replay harness |
| mlx-native | `a28783e` | `dispatch_hadamard_quantize_kv_seq` — referenced in ADR-007 |

### 2.2 Inputs

Manifest: `/tmp/cfa-20260422-C1-kernel-replay/manifest.json`

| File | Shape | Dtype | Bytes |
|---|---|---|---|
| `inputs/q_natural.bin` | [16, 256] | f32 | 16,384 |
| `inputs/k_packed_padded.bin` | [8, 1024, 128] | u8 | 1,048,576 |
| `inputs/v_packed_padded.bin` | [8, 1024, 128] | u8 | 1,048,576 |
| `inputs/k_norms_padded.bin` | [8, 1024] | f32 | 32,768 |
| `inputs/v_norms_padded.bin` | [8, 1024] | f32 | 32,768 |
| `inputs/k_norms_canary.bin` | [8, 1024] | f32 | 32,768 |
| `inputs/v_norms_canary.bin` | [8, 1024] | f32 | 32,768 |

Compact source files (22-row, not padded):

| File | Shape | Bytes |
|---|---|---|
| C-0b TQ: `hf2q_k_packed_layer00_pos22.u8.bin` | [8, 22, 128] | 22,528 |
| C-0b TQ: `hf2q_v_packed_layer00_pos22.u8.bin` | [8, 22, 128] | 22,528 |
| C-0b TQ: `hf2q_k_norms_layer00_pos22.f32.bin` | [8, 22] | 704 |
| C-0b TQ: `hf2q_v_norms_layer00_pos22.f32.bin` | [8, 22] | 704 |
| C-0 dense: `hf2q_q_normed_layer00_pos22.bin` | [16, 256] | 16,384 |

`q_natural.bin` is `hf2q_q_normed_layer00_pos22.bin` — post-RMSNorm, post-RoPE, **pre-FWHT**
(natural basis). The C-0 dump site (`forward_mlx.rs:1252`) captures `attn_q_normed` before the
FWHT dispatch at line 1433, confirming natural basis.

### 2.3 Decode-call params (L=0, decode step 1)

Verified against `forward_mlx.rs:1452–1474` by Worker 1:

| Field | Value | Source |
|---|---|---|
| `num_heads` | 16 | `nh as u32` — model arch |
| `num_kv_heads` | 8 | `nkv as u32` — model arch |
| `head_dim` | 256 | `hd as u32` — model arch |
| `kv_seq_len` | 22 | post-prefill, pre-decode-write; **see defect note** |
| `kv_capacity` | 1024 | model config |
| `scale` | 1.0 | literal at line 1458 |
| `mask_type` | 2 | `is_sliding=true` branch at line 1459 |
| `sliding_window` | 1024 | `self.sliding_window` at line 1460 |
| `softcap` | 0.0 | literal at line 1461 |
| `ring_start` | 0 | kv_seq_len=22 < kv_capacity=1024; pre-wrap |

Note: the on-disk TQ meta JSON has `mask_type=1` — that value is hard-coded at the dump site
`forward_prefill.rs:856`. The correct decode-time value is 2. Harness correctly uses 2.

### 2.4 The 3 variations + canary

| Run | What changes |
|---|---|
| **A** (full path) | forward-FWHT(Q) → TQ kernel → inverse-FWHT(output). Mirrors production. |
| **B** (FWHT disabled) | Skip both FWHT dispatches; pass Q in natural basis; read output in rotated domain. Tests H2. |
| **C** (dense re-encoded) | Dequantize TQ-packed 22-row → re-apply FWHT + nibble_quantize → fresh packed buffer; then full A path. Tests H4 (encoding locus). |
| **A_canary** | Same as A but `k_norms[22..1023] = v_norms[22..1023] = 1e9`. Tests mask-leak. |

### 2.5 Padding strategy

Positions 22..1023 in `k_packed_padded` and `v_packed_padded` are filled with `0x00`; positions
22..1023 in `k_norms_padded` and `v_norms_padded` are filled with `0.0f32`. Stride:
`(h, p, b)` → `h*1024*128 + p*128 + b` (u8 packed); `(h, p)` → `h*1024 + p` (f32 norms).

### 2.6 CPU reference

Computed in-harness (not from disk): dequantize the 22-row TQ compact files via
`nibble_dequantize` (which mirrors `test_flash_attn_vec_tq.rs` lines 127–143, applying `fwht_inplace`
internally and returning natural-basis K/V), then run `cpu_sdpa()` over [8, 22, 256] K/V with
Q_normed. The on-disk `hf2q_sdpa_out_layer00_pos22.bin` (dense GPU at kv_seq_len=23) is NOT
used as the comparison target — it is listed in the manifest for engineering reference only.

Validation: CPU reference first-4 values `[-0.926, -0.552, -0.176, 1.136]` vs dense sdpa_out
first-4 `[-0.895, -0.432, -0.303, 1.112]`; norms 48.4 vs 48.9. Close agreement confirms the
in-harness CPU reference is sound.

### 2.7 Harness Defect #1 — kv_seq_len=22 vs 23

The harness dumps the TQ compact cache at end-of-prefill (`kv_seq_len=22`). In production, before
`flash_attn_vec_tq` is dispatched at decode step 1:

1. `forward_mlx.rs:1009–1015`: `kv_seq_len` is incremented to 23 **at the top of the per-token
   call, before any per-layer work**. This applies to both dense and TQ SDPA branches.
2. `forward_mlx.rs:1226–1243`: `hadamard_quantize_kv` writes the decode-token K and V into packed
   cache at slot 22, making the cache 23 rows.
3. `forward_mlx.rs:1464`: `flash_attn_vec_tq` is dispatched with `kv_seq_len=23`.

The harness captures state at step 0 (pre-decode-write), not between steps 2 and 3. The 23rd row —
the current decode token's K/V, the position Q attends to most strongly — is absent from all 4 runs.

Note: this is the defect the analyst (Worker 3) caught. A prior revision of this report used the
catch to reframe C-0's L=0/P=1 bound compliance as "dense-23 vs TQ-22 apples-to-oranges"; the
reframing is retracted in §1 above — the increment at `forward_mlx.rs:1012–1014` affects both
paths, so in production both paths dispatch SDPA at 23 rows. The harness is at 22 rows; C-0 is
at 23 on both sides.

### 2.8 Harness Defect #2 — Missing memory barriers between dependent dispatches

`/opt/mlx-native/examples/tq_kernel_replay.rs:590–660` dispatches up to three kernels into a
single `CommandEncoder` with data dependencies (FWHT writes `q_buf`, SDPA reads `q_buf`;
SDPA writes `output_buf`, inverse-FWHT reads `output_buf`). That encoder is created with
`MTLDispatchType::Concurrent` (`/opt/mlx-native/src/encoder.rs:404–409`); the comment at
`encoder.rs:430–437` states explicitly: "When the encoder uses `MTLDispatchTypeConcurrent`,
all dispatches can execute concurrently unless separated by a barrier. Call this between
dispatches where the later dispatch reads a buffer written by an earlier one."

Production does this correctly at three sites:

| hf2q line | Barrier |
|---|---|
| `forward_mlx.rs:1429–1431` | `s.barrier_between(&[attn_q_normed], &[attn_q_normed])` — RAW/WAR on Q before forward FWHT |
| `forward_mlx.rs:1441–1446` | `s.barrier_between(&[attn_q_normed, k_packed, k_norms, v_packed, v_norms], &[sdpa_out])` — before TQ SDPA |
| `forward_mlx.rs:1477–1480` | `s.barrier_between(&[sdpa_out], &[sdpa_out])` — before inverse FWHT |

Harness calls to `memory_barrier` / `barrier_between`: **zero**. Without these barriers the
Metal runtime is free to overlap or reorder the three dispatches; data-dependent reads can observe
the pre-write state. This defect is independent of Defect #1 and would confound the variation
results even with a correct 23-row capture. The analyst implementer noted the structural
difference (agents-implementer-result: "one structural difference: the harness uses a single
CommandEncoder for all three dispatches (no memory barriers between them). Production uses
explicit barrier_between calls. For correctness of individual reads these barriers matter; for
the kernel's numeric output they do not") but mis-assessed the correctness implication —
on a `Concurrent` encoder barriers do change numeric output, because they gate reordering of
dependent reads. The A = B observation is fully consistent with the FWHT dispatches racing the
SDPA and not reliably being the input Q (or not reliably being applied to the output) at the
time the SDPA reads/writes.

---

## 3. Results

### Table 3.1 — 4-variation metrics (22-row replay)

Source files: `/tmp/cfa-20260422-C1-kernel-replay/out/{A,B,C,A_canary}.json`

| Run | nrmse | max_abs_diff | Exit | Notes |
|---|---|---|---|---|
| A (full path) | **1.2445** | **15.132** | 0 | Baseline production path |
| B (FWHT disabled) | **1.2445** | **15.132** | 0 | Bit-identical to A (full nrmse and per-head breakdown match exactly) |
| C (dense re-encoded) | **1.2435** | **15.126** | 0 | Delta vs A: nrmse −0.001, max_abs_diff −0.006 |
| A_canary | **1.2445** | **15.132** | 0 | Bit-identical to A (mask-leak ruled out) |

Kernel declared bound: `nrmse < 0.15`. All 4 runs violate by approximately 8×. No NaN or Inf in
any GPU output. `|C.nrmse − A.nrmse| = 0.001 < 0.005` (tie-breaker threshold from queen's spec
decision table): C is not distinguishable from A at 22 rows.

**These numbers are not interpretable.** Both harness defects confound the table:

- Defect #1 means every row is computed at the wrong kv_seq_len (22 vs production 23), so no row
  measures production behavior.
- Defect #2 means the FWHT-pair dispatches may not reliably sequence with the SDPA dispatch; the
  contrast between A (FWHT enabled) and B (FWHT disabled) may be smaller than, equal to, or
  opposite in sign from what it would be on a correctly-barriered harness.
- Variation C's +/-0.001 delta from A is not diagnostic: C dequantizes via `nibble_dequantize`
  and re-quantizes via `nibble_quantize` using the same algorithm that produced the packed bytes
  at dump time; the result is essentially the same bytes (up to the codebook rounding noise). C
  does not isolate a new locus. The real H4 control — a dense `flash_attn_vec` on the same 23-row
  natural-basis K/V — is not run in this session.
- A_canary = A confirms that the kernel's outer loop breaks at `ic >= kv_seq_len` (line 264) and
  the inner loops `continue` at `kv_pos >= kv_seq_len` (lines 299, 358), so positions 22..1023
  are never dereferenced. It does NOT confirm that the chunk-level mask at lines 270–281
  correctly gates ring/sliding semantics at in-range positions.

### Table 3.2 — Verification checks (Worker 3 findings, embedded in implementer result)

| Check | Status | Evidence |
|---|---|---|
| CPU reference faithful | PASS | `tq_kernel_replay.rs:423–459` mirrors `test_flash_attn_vec_tq.rs:127–200`; `nibble_dequantize` + `cpu_sdpa` identical algorithm |
| FWHT forward dispatch called in A | PASS | `tq_kernel_replay.rs:594–603` dispatches `fwht_standalone::dispatch_fwht_f32` on q_buf before kernel |
| FWHT inverse dispatch called in A | PASS | `tq_kernel_replay.rs:622–630` dispatches `dispatch_fwht_f32` on output_buf after kernel |
| B correctly skips both FWHT dispatches | PASS | `tq_kernel_replay.rs:633–657` dispatches kernel only, no FWHT |
| C re-encode via `nibble_quantize` mirrors production | PASS | `tq_kernel_replay.rs:480–488` uses `nibble_quantize`; padding stride matches `h*kv_capacity*(hd/2) + p*(hd/2)` |
| Canary norms differ from padded at positions 22..1023 | PASS | `k_norms_canary.bin` / `v_norms_canary.bin` = 1e9 at positions 22..1023 (byte-verified by Worker 1) |
| All 10 params match `forward_mlx.rs:1452–1474` | PASS | Manifest verbatim verification by Worker 1; `mask_type=2` corrected from meta JSON `mask_type=1` |
| kv_seq_len in harness matches production at decode step 1 | **FAIL** | Harness uses 22; production uses 23 (post `hadamard_quantize_kv` write at `forward_mlx.rs:1226–1243`) |

---

## 4. Verdict and Rationale

**Verdict: VERIFICATION_BLOCKED — two independent harness defects.**

### Why the verdict is blocked

Both harness defects are blocking on their own; fixing either without fixing the other still
leaves the run uninterpretable.

- **Defect #1 (22 vs 23):** The 23rd cache row — the current decode token's K/V, written by
  `hadamard_quantize_kv` at `forward_mlx.rs:1226–1243` before the SDPA dispatch — is absent from
  all 4 runs. Q attends to this row with the greatest weight. Any numerical result from a 22-row
  replay is formally unattributable to production behavior.
- **Defect #2 (missing barriers):** The harness dispatches FWHT(Q) → SDPA → inverse-FWHT(out)
  onto a `MTLDispatchType::Concurrent` encoder with zero barriers. Production inserts three
  explicit `barrier_between` calls. Without them, the three dispatches may overlap or reorder,
  so the SDPA's Q input and the inverse-FWHT's input are racy. The A = B bit-identity observation
  is consistent with the FWHT pair not actually ordering with the SDPA — i.e., the harness's
  "with FWHT" and "without FWHT" runs are not the contrast they claim to be.

The decision table's precondition `A_reproduces_c0` (A.nrmse ∈ [0.096, 0.107]) fails. Given
both defects, this failure does not imply anything about the kernel — neither the magnitude
of divergence nor its attribution to any hypothesis can be read off the table.

### What the four runs tell us about hypotheses H1, H2, H4

Nothing attributable.

- **H2 (FWHT):** A = B at 22 rows is consistent with FWHT-is-a-no-op, with FWHT-races-SDPA
  (Defect #2), or with FWHT-doesn't-matter-at-22-rows. All three remain possible.
- **H4 (encoding / dispatch / binding):** Variation C is degenerate — dequantize-then-requantize
  the same bytes through the same codebook is a no-op up to codebook rounding, so C ≈ A is a
  tautology at 22 rows. The real H4 control (dense `flash_attn_vec` on the same dequantized 23-row
  K/V) was not run. The mask-leak sub-hypothesis is NOT cleared by A_canary — the canary
  mutates out-of-loop-range positions, which the kernel provably never reads.
- **H1 (kernel math):** The implementer's result suggested H1 because "B ≈ A and C ≈ A". Given
  Defects #1 and #2, the "B ≈ A" premise is confounded and the "C ≈ A" premise is tautological.
  H1 is neither ruled in nor ruled out.

### What this verdict does NOT claim

- Does NOT invalidate C-0's "all 30 layers violate bound" finding. C-0's dense and TQ SDPA
  dispatches both see kv_seq_len=23 at the SDPA call site (the `seq_len` increment at
  `forward_mlx.rs:1012–1014` fires for both branches); C-0's comparisons were row-count-matched.
  The prior revision of this report's "dense-23 vs TQ-22" reframing is retracted.
- Does NOT invalidate C-0b's E1-partial verdict (H3 cleared on non-batched path).
- Does NOT rule out H2 — A = B at 22 rows with no barriers is uninformative.
- Does NOT rule out H4 — the harness never ran the dense-control variation.

---

## 5. Three-Sessions-Three-Methodology-Errors Meta-Observation

A consistent pattern has emerged across the three TQ-investigation sessions:

- **C-0** produced three high-severity methodology errors (nrmse formula mismatch, wrong threshold
  for `max_abs_diff`, byte-identical current-step Q/K/V not proving SDPA-input identity). All three
  were caught by Codex review.
- **C-0b** produced three high-severity methodology errors — also all caught by Codex review.
- **C-1** produced **four** methodology errors:
  - Defect #1 (kv_seq_len 22 vs 23): caught by the internal analyst (Worker 3) before Codex.
  - Defect #2 (missing FWHT barriers on concurrent encoder): **caught by Codex**.
  - Over-reach #1 (reframing C-0's L=0/P=1 as "dense-23 vs TQ-22 apples-to-oranges"):
    **caught by Codex**; that reframing is itself wrong because the seq_len increment at
    `forward_mlx.rs:1012–1014` applies to both branches.
  - Over-reach #2 (canary A = A_canary "clears H4 mask-leak"): **caught by Codex**; canary
    mutations are at out-of-loop-range positions and only confirm loop bounds.

The pattern: each session produces methodology artifacts that downstream review catches. The
internal workers have become better at catching the first-order error (22/23), but compound
errors and over-reach claims in the verdict narrative continue to fall through to Codex. The cost
is still one full session of confounded measurement runs per iteration.

**Recommended CFA protocol refinement for TQ-related sessions:**

1. **Input-verification phase before implementation.** Before the implementer builds the harness,
   a verification worker traces the production code path from the intended capture point to the
   kernel dispatch site and explicitly verifies that every state field (`kv_seq_len`, `ring_start`,
   `mask_type`, buffer contents) at the capture point matches its value at the dispatch site.
   Blocking gate; if any field disagrees, the manifest is wrong.

2. **Dispatch-fidelity verification phase.** A second verification step: before the harness is
   run, explicitly verify that the dispatch mechanics (encoder type, barrier placement,
   synchronization primitives) between harness and production match. Concurrent encoders without
   matching barriers are a silent correctness gap; they must be audited as part of the harness,
   not as an afterthought.

3. **Canary design rule.** Canaries must mutate values that the kernel is supposed to read (i.e.,
   in-range positions), not values that the kernel provably skips. Out-of-loop-range mutations
   test loop bounds, not mask semantics, and cannot clear mask-leak hypotheses.

4. **Control-variation design rule.** A "control" variation that trivially reproduces the baseline
   inputs (dequantize then requantize through the same codebook) is not a control. Controls must
   take a substantively different path — e.g., dense kernel on the same inputs, or different
   codebook, or a reference implementation.

---

## 6. Next Session: Unblock C-1

Five steps. All five must complete before the decision table is applicable. The existing harness
scaffold (mlx-native@`9a4ca61`) is retained; the fixes are surgical.

### Step 1 — Add memory barriers to the harness (Defect #2)

Edit `/opt/mlx-native/examples/tq_kernel_replay.rs:590–660`. Between FWHT(Q) and the
`flash_attn_vec_tq` dispatch, and between `flash_attn_vec_tq` and the inverse FWHT, call
`encoder.memory_barrier()` (the exposed API on `CommandEncoder`, see
`/opt/mlx-native/src/encoder.rs:438–453`). Verify by grep that both the A and C branches have
two barrier calls each, and by running `A` and confirming the nrmse changes from 1.2445 to
something else — any change confirms the barriers are now gating the dispatches.

### Step 2 — Add a second TQ state dump site in hf2q (Defect #1)

On hf2q main, add a second dump point: **after `hadamard_quantize_kv` writes the decode-token
K/V at slot 22, but before the `flash_attn_vec_tq` dispatch at `forward_mlx.rs:1464`**. Gate
behind `HF2Q_DUMP_TQ_STATE=1 && HF2Q_DUMP_DECODE_STEP=1`. Capture `k_packed`, `v_packed`,
`k_norms`, `v_norms` with 23 rows populated; write params `{kv_seq_len: 23, layer: 0, decode_pos: 1}`.

### Step 3 — Re-dump and rebuild manifest

Run sourdough 22-token prefill + 1 decode step with both env gates active. Recompute padded
inputs (23-row compact → 1024-row padded). Update manifest `kv_seq_len: 23`. Redesign canary
to mutate an IN-range norm (e.g. `k_norms[0][10] = 1e9` while keeping positions 23..1023 = 0.0)
— this tests whether the kernel observes that in-range value, not whether the mask gates
out-of-range values.

### Step 4 — Replace Variation C with a real dense control

Variation C as implemented dequant-then-requant through the same codebook is a no-op. Replace
it with: run `mlx_native::ops::flash_attn_vec::flash_attn_vec` (the dense kernel) on the 23-row
natural-basis K/V produced by `nibble_dequantize` from the TQ compact dump. Compare GPU output
to the same CPU reference. This gives a true H4 control: dense kernel on clean inputs vs TQ
kernel on the same upstream data. If dense clears the bound and TQ does not, H4 is decisively
inside the TQ packed-read / FWHT path.

### Step 5 — Persist raw `.bin` outputs and apply verdict rules

Write the raw `sdpa_out` f32 bytes for every run, not just summary metrics. With barriers + 23
rows + the dense control + byte-level output comparison:

- If TQ-A, TQ-B, TQ-C (dense) all far from 0, with dense ≪ TQ: H4 (TQ-specific encoding or
  packed-read).
- If TQ-A, TQ-B close to each other but both far from dense: H2 or H1 (inside the TQ path but
  not in the FWHT bracketing).
- If TQ-A ≪ TQ-B and both far from dense: H2 (FWHT pair).
- If dense, TQ-A, TQ-B all close but far from CPU reference: H1 (kernel math, independent of
  quantization — unlikely given dense works in production on other paths, but possible).

### Non-goal

Do NOT speculatively proceed toward a fix (kernel bisect, FWHT rework, encoder rewrite) before
Steps 1–5 land. The cost of a third consecutive session of unconfirmed hypothesis shift is higher
than the cost of one correct session of measurement.

---

## 7. Instrumentation and Commit Disposition

The harness commit `9a4ca61` on mlx-native is the scaffold for the 23-row re-run. It should be
retained locally on mlx-native until the 23-row re-run produces a clean verdict. After that,
either land it permanently as `examples/tq_kernel_replay.rs` (examples are persistent by
convention in this repo) or retire it. Do not force-merge before the verdict is clear.

The hf2q repo instrumentation commit (new decode-step dump site) will be a small forward commit on
main. It can be gated behind `HF2Q_DUMP_TQ_STATE=1 && HF2Q_DUMP_DECODE_STEP=1` to keep it
default-off. It does not require a revert after the investigation closes.

---

## 8. Appendix

### A. Reproduce commands

```bash
# Build
cd /opt/mlx-native && cargo build --release --example tq_kernel_replay

# Variation A (full path)
cargo run --release --example tq_kernel_replay -- \
  --manifest /tmp/cfa-20260422-C1-kernel-replay/manifest.json \
  --variation A \
  --out /tmp/cfa-20260422-C1-kernel-replay/out/A.json

# Variation B (FWHT disabled)
cargo run --release --example tq_kernel_replay -- \
  --manifest /tmp/cfa-20260422-C1-kernel-replay/manifest.json \
  --variation B \
  --out /tmp/cfa-20260422-C1-kernel-replay/out/B.json

# Variation C (dense re-encoded)
cargo run --release --example tq_kernel_replay -- \
  --manifest /tmp/cfa-20260422-C1-kernel-replay/manifest.json \
  --variation C \
  --out /tmp/cfa-20260422-C1-kernel-replay/out/C.json

# Canary (mask-leak test)
cargo run --release --example tq_kernel_replay -- \
  --manifest /tmp/cfa-20260422-C1-kernel-replay/manifest.json \
  --variation A \
  --canary \
  --out /tmp/cfa-20260422-C1-kernel-replay/out/A_canary.json
```

### B. manifest.json (inline)

```json
{
  "session": "cfa-20260422-C1-kernel-replay",
  "created_utc": "2026-04-22T02:55:00Z",
  "layer": 0,
  "decode_pos": 1,
  "is_sliding": true,
  "params": {
    "num_heads": 16,
    "num_kv_heads": 8,
    "head_dim": 256,
    "kv_seq_len": 22,
    "kv_capacity": 1024,
    "scale": 1.0,
    "mask_type": 2,
    "sliding_window": 1024,
    "softcap": 0.0,
    "ring_start": 0
  }
}
```

(Full manifest with input paths and compact source paths at
`/tmp/cfa-20260422-C1-kernel-replay/manifest.json`.)

### C. Commit SHAs and file list

| Repo | SHA | Item |
|---|---|---|
| mlx-native | `9a4ca61` | `examples/tq_kernel_replay.rs` — replay harness |
| mlx-native | `a28783e` | `ops/hadamard_quantize_kv.rs` — `dispatch_hadamard_quantize_kv_seq` |
| hf2q (main) | `f519573` | HEAD at session start (ADR-007 C-0b docs) |

### D. Session artifacts

All session artifacts are under `/tmp/cfa-20260422-C1-kernel-replay/`:

```
manifest.json
inputs/
  q_natural.bin           (16,384 bytes — [16,256] f32)
  k_packed_padded.bin     (1,048,576 bytes — [8,1024,128] u8)
  v_packed_padded.bin     (1,048,576 bytes — [8,1024,128] u8)
  k_norms_padded.bin      (32,768 bytes — [8,1024] f32)
  v_norms_padded.bin      (32,768 bytes — [8,1024] f32)
  k_norms_canary.bin      (32,768 bytes — [8,1024] f32, positions 22..1023 = 1e9)
  v_norms_canary.bin      (32,768 bytes — [8,1024] f32, positions 22..1023 = 1e9)
out/
  A.json          (nrmse=1.2445, max_abs_diff=15.132)
  B.json          (nrmse=1.2445, max_abs_diff=15.132 — bit-identical to A)
  C.json          (nrmse=1.2435, max_abs_diff=15.126)
  A_canary.json   (nrmse=1.2445, max_abs_diff=15.132 — bit-identical to A)
```
