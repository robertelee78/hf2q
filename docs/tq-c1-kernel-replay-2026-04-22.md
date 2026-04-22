# TQ C-1 Kernel Replay — ADR-007 C-4 E1 Branch

**Date:** 2026-04-22
**Session:** `cfa-20260422-C1-kernel-replay`
**Verdict:** VERIFICATION_BLOCKED
**Report author:** Worker 4 (tech-writer-reporter, `cfa-20260422-C1-kernel-replay-reporter`)
**Prior sessions:** C-0 audit (`docs/tq-c0-audit-2026-04-21.md`) · C-0b localization (`docs/tq-c0b-localize-2026-04-21.md`)
**ADR reference:** `docs/ADR-007-turboquant-kv-cache.md` — C-4 E1 branch

---

## 1. Executive Summary

**Verdict: VERIFICATION_BLOCKED.**

The harness (mlx-native@`9a4ca61`) replays the `flash_attn_vec_tq` kernel with `kv_seq_len=22` — the
end-of-prefill TQ compact cache state. However, production SDPA at decode step 1 runs with
`kv_seq_len=23`: the decode-token K/V is written into the packed cache at slot 22 **before** the
SDPA dispatch fires. Evidence: `forward_mlx.rs:1009–1015` increments `kv_seq_len`, and
`forward_mlx.rs:1226–1243` writes the 23rd K/V row into packed cache, both occurring before the
`flash_attn_vec_tq` dispatch at line 1464. The 22-row replay cannot formally attribute the observed
divergence to any hypothesis because the 23rd row — containing the current decode token — is the
position Q attends to most heavily and is absent from the harness inputs.

What was learned despite the blocked verdict:

- **Canary A = A_canary (bit-identical):** norms set to 1e9 at positions 22..1023 produce
  identical output to the zero-padded run. The mask correctly ignores positions ≥ kv_seq_len=22.
  H4 mask-leak sub-hypothesis is ruled out.
- **A = B (bit-identical):** enabling vs. disabling the forward-FWHT-on-Q + inverse-FWHT-on-output
  pair produces identical GPU output. Either the FWHT dispatch silently no-ops, or the kernel is
  not reading Q at all in a way that depends on rotation. FWHT is not the discriminating variable
  at 22 rows — but this observation is ambiguous without 23-row inputs.
- **C ≈ A at 22 rows (nrmse 1.2435 vs 1.2445, delta 0.001):** re-encoding K/V from a
  known-good F32 source via `nibble_quantize` does not improve kernel output. Consistent with
  H1 (kernel math) but not formally attributable.
- **C-0's L=0/P=1 "within-bound" claim:** C-0 measured dense-GPU (kv_seq_len=23) vs
  TQ-GPU (kv_seq_len=22) — an apples-to-oranges comparison. The claim holds as stated for that
  particular measurement but deserves a footnote. The harness's in-memory CPU reference (computed
  from 22-row TQ dequant) closely matches the dense sdpa_out (norm 48.4 vs 48.9), confirming the
  CPU reference is sound. C-0's "all 30 layers violate bound" finding at pos 5+ used matched
  kv_seq_len on both sides and is not invalidated.

**The one experiment that unblocks C-1:** re-dump the TQ-packed cache **after**
`hadamard_quantize_kv` writes the decode-token K/V at slot 22 (i.e., post-write pre-SDPA at
decode step 1, `forward_mlx.rs:1226–1243` → `1464`), giving a 23-row compact cache. Update the
manifest to `kv_seq_len: 23` and re-run all 4 variations.

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

### 2.7 Harness defect (kv_seq_len=22 vs 23)

The harness dumps the TQ compact cache at end-of-prefill (`kv_seq_len=22`). In production, before
`flash_attn_vec_tq` is dispatched at decode step 1:

1. `forward_mlx.rs:1009–1015`: `kv_seq_len` is incremented to 23.
2. `forward_mlx.rs:1226–1243`: `hadamard_quantize_kv` writes the decode-token K and V into packed
   cache at slot 22, making the cache 23 rows.
3. `forward_mlx.rs:1464`: `flash_attn_vec_tq` is dispatched with `kv_seq_len=23`.

The harness captures state at step 0 (pre-decode-write), not between steps 2 and 3. The 23rd row —
the current decode token's K/V, the position Q attends to most strongly — is absent from all 4 runs.

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
decision table): C is not distinguishable from A.

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

**Verdict: VERIFICATION_BLOCKED.**

### Why the verdict is blocked

The harness methodology is invalid for measuring the production kernel's behavior at decode step 1.
The 23rd cache row — the current decode token's K/V, written by `hadamard_quantize_kv` at
`forward_mlx.rs:1226–1243` before the SDPA dispatch — is absent from all 4 runs. This is the row
Q attends to with greatest weight (position 22, attended by the current query). Any numerical
result from the 22-row replay is formally uninterpretable with respect to production behavior.

The decision table from the queen's spec (precondition `A_reproduces_c0`: A.nrmse ∈ [0.096, 0.107])
is not satisfied. A.nrmse = 1.2445 far exceeds that range — but this is now understood to be
because C-0's 0.1017 figure was computed dense-GPU (23-row) vs TQ-GPU (22-row), not TQ vs TQ at
matched row count. The harness is measuring a correct and consistent quantity (TQ-GPU vs
CPU-dequant-of-22-row-TQ); that quantity happens not to match C-0's reference, because C-0 used a
different reference. This does NOT mean the harness is broken — it means the sourdough check
precondition in the spec was written against a C-0 measurement that had its own 22/23-row mismatch.

### Partial findings — what they suggest but do not prove

- **B = A (bit-identical):** Consistent with the FWHT pair having no effect on this particular
  22-row computation, OR with the FWHT dispatch silently no-oping. Does not clear H2 — the missing
  23rd row could be where FWHT matters most.
- **C ≈ A (delta 0.001):** Consistent with H1 (kernel math bug independent of input encoding).
  Does not formally rule out H4 on the 23-row path.
- **A_canary = A:** H4 mask-leak sub-hypothesis is cleared. The mask correctly gates out positions
  22..1023 regardless of norm values at those positions.

These partial findings **suggest** H1 but do not confirm it.

### What this verdict does NOT claim

- Does NOT invalidate C-0's "all 30 layers violate bound" finding at pos 5+ (those positions used
  matched kv_seq_len on both sides of the comparison).
- Does NOT invalidate C-0b's E1-partial verdict (H3 cleared on non-batched path).
- Does NOT rule out H2 (FWHT) — A=B at 22 rows is ambiguous.
- Does NOT rule out H4 (dispatch / buffer binding / stride) for the 23-row case — only the
  mask-leak sub-hypothesis is cleared.

---

## 5. Three-Sessions-Three-Methodology-Errors Meta-Observation

A consistent pattern has emerged across the three TQ-investigation sessions:

- **C-0** produced three high-severity methodology errors (nrmse formula mismatch, wrong threshold
  for `max_abs_diff`, byte-identical current-step Q/K/V not proving SDPA-input identity). All three
  were caught by the Codex reviewer in the same session.
- **C-0b** produced three high-severity methodology errors (caught by Codex review). The report
  at `docs/tq-c0b-localize-2026-04-21.md` documents these.
- **C-1** has at least one methodology error (kv_seq_len=22 vs 23) — caught by Worker 3 before
  Codex review.

The pattern is: every session produces a methodology artifact that the next level of review catches.
The cost is one full session of wasted measurement runs.

**Recommended CFA protocol refinement for TQ-related sessions:** add a dedicated
input-verification phase before any measurement runs. Specifically, before the implementer builds
the harness, a verification worker should trace the production code path from the intended capture
point to the kernel dispatch site and explicitly verify that all state fields (kv_seq_len, ring_start,
mask_type, buffer contents) at the capture point match their values at the dispatch site. This phase
should be a blocking gate — if any field does not match, the manifest is wrong and the implementer
should not proceed.

---

## 6. Next Session: Unblock C-1

The following session is expected to be short (3 workers, one session).

### Step 1 — Instrumentation extension (~30 lines)

On the hf2q main branch (or re-cherry-picked from the codex branch), add a second TQ state dump
site. Move the dump from end-of-prefill to **after `hadamard_quantize_kv` writes the decode-token
K/V at slot 22, but before the `flash_attn_vec_tq` dispatch at `forward_mlx.rs:1464`**. The
existing `HF2Q_DUMP_TQ_STATE=1` env gate can be extended with a decode-step flag.

The new dump site captures:
- `kv_caches[0].k_packed` and `v_packed` with 23 rows populated
- `kv_caches[0].k_norms` and `v_norms` with 23 rows populated
- Capture params: `kv_seq_len=23`, layer=0, decode_pos=1

### Step 2 — Re-dump

Re-run the sourdough 22-token prefill + 1 decode step with `HF2Q_DUMP_TQ_STATE=1` active at the
new decode-step site. Output: compact 23-row K/V packed files for L=0.

### Step 3 — Update manifest

Set `kv_seq_len: 23`. Point to new 23-row compact files. Recompute padded files (23-row compact
padded to 1024 rows). Update canary norms to cover positions 23..1023.

### Step 4 — Re-run all 4 variations

Use the existing harness at mlx-native@`9a4ca61` (update manifest path only). Run A, B, C, A_canary.

### Step 5 — Apply verdict rules

With 23-row data, the decision table preconditions can be evaluated properly. Expected indicators:
- If A ≈ B ≈ C and all far from 0: H1 confirmed (kernel math).
- If B < A: H2 (FWHT) is the locus or a contributor.
- If C < A and B ≈ A: H4 (encoding or dispatch/stride bug).

### Step 6 — If the user wants to skip measurement and go directly to fix

The C-4 E1 branch could proceed speculatively toward H1 (kernel math) given the partial findings.
The kernel bisect entry point would be the QK dot-product accumulation inner loop in
`flash_attn_vec_tq.metal` (approximately lines 291–317 per the C-0 ADR reference). Risk: if the
actual locus is H2 or H4, the bisect wastes the session. Recommendation: do the 23-row re-dump
first; it is a short session with high information value.

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
