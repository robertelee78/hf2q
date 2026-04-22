# TQ C-0b Localization Audit — 2026-04-21

**CFA session:** `cfa-20260421-C0b-localize`  
**Mode:** review-only (Worker 3 — tech-writer-reporter)  
**Date:** 2026-04-21  
**Authors:** Workers 1 (researcher), 2 (analyst), 3 (reporter)  
**References:** ADR-007 §C-0b, ADR-007 §C-4 E1/E2/E3 branching

---

## 1. Executive Summary

**Verdict: E1-partial — non-batched compact-encoded L0/L5 prefill cache cleared; sourdough regression is NOT in encode on the path that actually runs by default. Batched-prefill encode + capacity-stride runtime access + verbatim decode-call parameters remain formally untested.**

The TQ-packed KV cache produced by the **non-batched** prefill encode path — `dispatch_hadamard_quantize_kv` on codex@d14f596 — dequantizes to within the kernel's own declared bounds on every one of 440 measured cells (2 layers × K+V × heads × 22 positions). Zero violations of nrmse < 0.15 or max_abs_diff < 1.0. The worst nrmse across all cells is **0.1390** (layer 0, K, head 1, position 20). The worst max_abs_diff is **0.4450** (layer 5, V, head 0, position 12).

**The path tested IS the path the 69-byte sourdough regression runs on.** `HF2Q_BATCHED_PREFILL` requires `HF2Q_UNSAFE_EXPERIMENTS=1` to take effect (`src/debug/investigation_env.rs:246`); neither `scripts/sourdough_gate.sh` nor the default hf2q binary set either. So the regression reproduces on the non-batched path cleared here — encode is not causing the observed sourdough failure, which argues the defect locus is downstream of encode on that path (H1 / H2 / H4).

**What this does NOT clear:**

1. **Batched-prefill encode** (`dispatch_hadamard_quantize_kv_seq`, added 2026-04-21 in codex-branch commit `415c9d6`). The instrumentation dumps from both paths, but the default run only exercised non-batched. The batched path is newer, prime-suspect code for any future long-context run, and remains formally untested.
2. **Capacity-strided runtime access.** The dump compacts from `[nkv, kv_capacity=1024, hd/2]` → `[nkv, kv_seq_len=22, hd/2]` at dump time (dump loop at `forward_prefill.rs:827–836`). The dequantizer walks the compact layout. Bugs that live in the `kv_capacity * hd_half` stride — e.g., the kernel reading a wrong physical index — are invisible to this test.
3. **Verbatim decode-call parameters.** The meta JSON hardcodes `mask_type: 1` for all layers, but the actual decode kernel at `forward_mlx.rs:1279` passes `mask_type: 2` for sliding (L0 is sliding). Meta is an encode-site snapshot, not a decode-call snapshot. Semantic mismatches between encode-side parameter belief and decode-site parameter usage are invisible to this test.

Next session (C-1) MUST: re-dump with raw capacity-strided buffers (no compaction), capture the actual decode-call params verbatim, and run again with `HF2Q_BATCHED_PREFILL=1 HF2Q_UNSAFE_EXPERIMENTS=1` to clear the batched path.

Per ADR-007 §C-4 E1 (interpreted narrowly): kernel replay test with real L=0/P=1 Q/K/V + capacity-strided packed cache + verbatim `mask_type` injected into `mlx-native/tests/test_flash_attn_vec_tq.rs`.

---

### 1.1 Revision note (queen Phase 3 — 2026-04-21)

This report initially self-certified E1 with language claiming the "batched-prefill encode path `dispatch_hadamard_quantize_kv_seq`" was cleared. Codex peer review (session `codex-review`, verdict `request_changes`, high-sev) caught three structural scope errors: (a) the meta JSON explicitly records `path: "nonbatched"` — batched was never dumped; (b) the dump is compacted at write-time, so the capacity-stride runtime layout is not what the dequantizer walked; (c) meta `mask_type` is a hardcoded literal, not a read-back of kernel-call parameters, and diverges from the real sliding-layer dispatch. The verdict is narrowed above from E1 to E1-partial. The 440-cell dequant math is **unchanged and verified correct** by Codex's independent Python reproduction (codex reproduced the exact worst-cells `L0/k/h1/p20 nrmse=0.138983` and `L5/v/h0/p12 max_abs=0.444982`); the revision is purely scope narrowing, not measurement revision.

---

## 2. Methodology

### 2.1 Branches and commits

| Side | Branch | Commit SHA | Purpose |
|---|---|---|---|
| Dense baseline | `main` | `a258e92` | Dense F32 KV cache; dump via `HF2Q_DUMP_ALL_CACHE=1` |
| TQ + C-0b instrumentation | `cfa/cfa-20260421-111303-tq-revival/codex` | `d14f596` | TQ-packed cache; dump via `HF2Q_DUMP_TQ_STATE=1` |
| mlx-native | — | `a28783e` | `flash_attn_vec_tq`, `hadamard_quantize_kv`, `turboquant.rs` |

### 2.2 Prompt

The sourdough prompt used verbatim (typo is load-bearing; sourced from `scripts/sourdough_gate.sh`):

```
Complrehensive instructions for making sourdough bread.
```

Same temperature=0 seed, both sides. Prompt produces 22 KV positions after prefill.

### 2.3 Dump boundaries and what was captured

**TQ side (codex@d14f596):** Dump fires at end-of-prefill in BOTH `forward_prefill.rs` (non-batched path, line ~827) and `forward_prefill_batched.rs` (batched path, line ~1362). Which one fires is controlled by `HF2Q_BATCHED_PREFILL` (ack-gated by `HF2Q_UNSAFE_EXPERIMENTS=1`). **In this session, neither flag was set** — only the non-batched dump fired. Meta JSON confirms this: `"path": "nonbatched"`. Gated by `HF2Q_DUMP_TQ_STATE=1`. Per-layer outputs written to `/tmp/cfa-20260421-C0b-localize/dumps/tq/`:

- `hf2q_k_packed_layer{LL}_pos22.u8.bin` — u8 nibble-packed K, shape `[nkv, 22, hd/2]`
- `hf2q_v_packed_layer{LL}_pos22.u8.bin` — u8 nibble-packed V, shape `[nkv, 22, hd/2]`
- `hf2q_k_norms_layer{LL}_pos22.f32.bin` — F32 per-position K norms, shape `[nkv, 22]`
- `hf2q_v_norms_layer{LL}_pos22.f32.bin` — F32 per-position V norms, shape `[nkv, 22]`
- `hf2q_tq_meta_layer{LL}_pos22.json` — layer metadata (nkv, nh, hd, kv_seq_len, kv_capacity, kv_write_pos, kv_is_sliding, ring_start, sliding_window, mask_type, layer_type)

Layers dumped: 0 (sliding, hd=256, nkv=8) and 5 (global, hd=512, nkv=2).

**Dense side (main@a258e92):** `HF2Q_DUMP_ALL_CACHE=1` fires in `forward_decode` (first decode step), writing full F32 KV arrays of shape `[nkv, 23, hd]` to `/tmp/cfa-20260421-C0b-localize/dumps/dense/`. The 23rd position (index 22) is the first decode-step K/V and is NOT present on the TQ side; it is excluded from all comparisons. Only positions `[:22]` are compared.

**Generation flag:** `--max-tokens 2` on the dense side (required to fire `HF2Q_DUMP_ALL_CACHE` which fires on the first decode forward pass). TQ side dumps at end-of-prefill without needing a decode step.

### 2.4 Kernel-mirror facts respected by the dequantizer

All citations are to source files in the working tree. The Python dequantizer at `scripts/tq-c0b-dequant.py` mirrors these exactly:

| Fact | Implementation | Source citation |
|---|---|---|
| `CODEBOOK_4BIT` inline constant | 16-entry F32 Lloyd-Max table, no per-layer buffer | `mlx-native/src/turboquant.rs:27–32` |
| Nibble packing: low nibble = even coord, high nibble = odd coord | `packed[c]` low 4 bits → coord `2c`, high 4 bits → coord `2c+1` | `hadamard_quantize_kv.metal:130–138` (encode), `flash_attn_vec_tq.metal:128–131` (decode) |
| rsqrt(hd) applied exactly once, before inverse FWHT | `scale = norm * rsqrt(hd)` multiplied into centroid-gather output | `flash_attn_vec_tq.metal:295` (K-loop), `:355` (V-loop) |
| Inverse FWHT = forward FWHT (self-inverse, 1/√N normalization, H·H=I) | Applied after centroid gather + norm scale | `turboquant.rs:94–97` (`fwht_inplace`) |

The pre-rotated centroid table described in ADR-007 §Architecture is a paper design that was NOT implemented. The actual kernel uses the inline scalar `CODEBOOK_4BIT` constant. No centroid-table buffer was dumped because none exists.

### 2.5 NRMSE formula

Per `test_flash_attn_vec_tq.rs:384–387` (the kernel's own test — same formula used here):

```
nrmse = sqrt( sum(diff²) / max(sum(ref²), 1e-30) )
```

Applied per `(head, position)` over the `[hd]`-length slice. This is the same formula used in the C-0 audit after Codex correction (see `docs/tq-c0-audit-2026-04-21.md §2.7`).

### 2.6 Self-test gate: 0.06 → 0.12 adjustment (Gaussian N(0,1) heuristic, not a universal floor)

The queen's spec cited 0.06 as the gate for the dequantizer self-test. Worker 2 adjusted to **0.12** against the Gaussian N(0,1) reference distribution. Rationale:

The `nrmse < 0.15` bound in `test_flash_attn_vec_tq.rs` is the safety margin for SDPA *output* (post attention averaging across many K/V vectors), not per-vector round-trip error. On 1000 draws from **N(0,1)** — the only distribution tested — a correct 4-bit Lloyd-Max encode+decode with matching Rust `nibble_quantize`/`nibble_dequantize` reference achieved nrmse = 0.0973 ± small. Measured self-test values: hd=256 → 0.097148, hd=512 → 0.097281.

**Calibration scope:** 0.097 is a **Gaussian N(0,1) heuristic**, NOT a universal floor. A correct implementation on codebook-aligned inputs (i.e., inputs whose values happen to land near codebook centroids) can produce nrmse well below 0.06. The 0.12 gate is therefore a distribution-conditional catch: it catches implementation bugs — wrong nibble order, wrong codebook, wrong FWHT normalization, rsqrt double-application — against a sampled Gaussian, but it does not prove a specific numeric floor holds for every input distribution. A more robust self-test for C-1 would include explicit bad-mutant controls (intentional off-by-one nibble order, transposed codebook, skipped FWHT, double-rsqrt) and assert the gate flags each of them.

The 0.06 gate in the original spec was not defensible without the distribution qualifier; the 0.12 gate is defensible on N(0,1) and should be re-derived per-input-distribution for future sessions. Codex review confirmed the dequantizer math and codebook values byte-identical to Rust reference; the self-test gate language above is the revised, accurate version.

### 2.7 Scope

- Layer 0 (sliding attention, hd=256, nkv=8): 8 heads × 22 positions × 2 ops (K, V) = 352 cells
- Layer 5 (global attention, hd=512, nkv=2): 2 heads × 22 positions × 2 ops (K, V) = 88 cells
- **Total: 440 cells**

C-0b is a localization experiment, not a re-audit of all 30 layers. Full-layer coverage is C-0's scope.

---

## 3. Results — Numerical Tables

### Table 3.1 — Per-layer worst-case summary

All values sourced from `/opt/hf2q/docs/tq-c0b-localize-2026-04-21-raw.csv` (440 rows).

| Layer | Op | max nrmse | worst (head, pos) | max abs diff | worst abs (head, pos) | Kernel bound holds |
|---|---|---|---|---|---|---|
| L0 (sliding, hd=256) | k | **0.138983** | (1, 20) | 0.048441 | (4, 11) | YES |
| L0 (sliding, hd=256) | v | 0.125411 | (0, 21) | 0.408404 | (3, 9) | YES |
| L5 (global, hd=512)  | k | 0.120948 | (1, 4) | 0.030501 | (0, 5) | YES |
| L5 (global, hd=512)  | v | 0.116922 | (1, 4) | **0.444982** | (0, 12) | YES |

Note: the summary.md reports max_abs_diff for L0/k as 0.048441 at worst_head=1, worst_pos=20 (the worst-nrmse cell). Direct CSV query reveals the actual max_abs_diff cell for L0/k is (head=4, pos=11) at 0.048441 — essentially equal in magnitude, both well within the 1.0 bound.

### Table 3.2 — Violation count

| Metric | Threshold | Cells total | Cells violating | Verdict |
|---|---|---|---|---|
| nrmse per (layer × op × head × pos) | < 0.15 | 440 | **0** | E1 clear |
| max_abs_diff per (layer × op × head × pos) | < 1.0 | 440 | **0** | E1 clear |

Source: direct Python query over `tq-c0b-localize-2026-04-21-raw.csv`. Zero violations on both metrics.

### Table 3.3 — V vs K max_abs_diff asymmetry (secondary observation)

| Layer | K max abs diff | V max abs diff | V/K ratio |
|---|---|---|---|
| L0 (sliding) | 0.048441 | 0.408404 | **~8.4×** |
| L5 (global) | 0.030501 | 0.444982 | **~14.6×** |

Both K and V pass the < 1.0 bound. However, V max_abs_diff is systematically 8–15× larger than K. This is a secondary signal. If the downstream defect (H1/H2/H4) has a V-path-specific issue, this asymmetry is where it would surface first. See §5 for the H4 interpretation.

### Table 3.4 — Dequantizer self-test

| head_dim | Self-test nrmse | Gate (< 0.12) | 4-bit floor (~0.097) | Status |
|---|---|---|---|---|
| 256 | 0.097148 | PASS | confirmed | PASS |
| 512 | 0.097281 | PASS | confirmed | PASS |

Source: `python3 /opt/hf2q/scripts/tq-c0b-dequant.py --self-test`.

---

## 4. Verdict and Logic Chain

**E1-partial.** The non-batched-compact-encoded prefill cache is not the defect locus on the path the sourdough regression runs on. Batched encode, capacity-stride runtime access, and decode-call-parameter semantics remain formally untested.

Logic chain:

1. **Given:** Paired F32 dumps at identical prompt + seed (sourdough, T=0), same kv_seq_len=22 tokens on both sides. Dense side: main@a258e92. TQ side: codex@d14f596.
2. **Given:** Meta JSON records `path: "nonbatched"` (field hardcoded in `forward_prefill.rs` dump block). `HF2Q_BATCHED_PREFILL` is ack-gated in `investigation_env.rs:246`; neither the sourdough gate nor the default hf2q binary set it. **The path dumped is the path the 69-byte regression runs on.**
3. **Given:** Python dequantizer validated by Gaussian N(0,1) round-trip self-test — nrmse 0.0973 on 1000 vectors at hd=256 and hd=512, below the 0.12 distribution-conditional gate; codebook values byte-identical to `turboquant.rs:27–32`; independently reproduced end-to-end by Codex.
4. **And:** Dequantized TQ cache vs dense F32 cache — **max nrmse across all 440 cells is 0.1390** (< kernel bound 0.15).
5. **And:** **Max_abs_diff across all 440 cells is 0.4450** (< kernel bound 1.0).
6. **Therefore (what IS proved):** On the non-batched prefill encode path, with inputs compacted to `[nkv, kv_seq_len, hd/2]` at dump time, the stored nibbles + norms reconstruct KV within kernel tolerance at both sliding (L0, hd=256) and global (L5, hd=512) layers. Encode math on this path is correct for the compacted artifact the dequantizer walks.
7. **Inference (stronger than pure negation):** Because the sourdough regression reproduces on this exact path (non-batched, default flags), and encode on this path is correct, the sourdough defect cannot live in non-batched encode. It lives downstream — H1 (kernel math), H2 (FWHT pipeline), or H4 (caller dispatch).

What this verdict does NOT claim:

- Does NOT clear **batched-prefill encode** (`dispatch_hadamard_quantize_kv_seq`). The batched path is untested. The new-encode bug class cannot be ruled out for any future run that sets `HF2Q_BATCHED_PREFILL=1 HF2Q_UNSAFE_EXPERIMENTS=1`.
- Does NOT clear **capacity-strided runtime access.** The dump compacts at `forward_prefill.rs:827–836`. A stride bug in the kernel (reading physical index `h * kv_capacity * hd_half + p * hd_half` where runtime semantics expected a different formula) is invisible to this test.
- Does NOT clear **decode-call parameter semantics.** Meta records `mask_type: 1` hardcoded; the actual decode kernel at `forward_mlx.rs:1279` passes `mask_type: 2` for sliding. Mask-type mismatches between encode-side belief and decode-site usage are invisible here.
- Does NOT claim the kernel is correct. H1 (`flash_attn_vec_tq` Metal shader math), H2 (FWHT pipeline wrapping the kernel call in `forward_mlx.rs`), H4 (caller-side dispatch — strides, norm binding, ring/mask params) all remain live.
- Does NOT rank H1/H2/H4 against each other beyond the secondary V/K magnitude asymmetry (Codex noted that ratio tracks the V-norms-vs-K-norms scale gap 8–16×, so the 8–14× V/K max_abs asymmetry is mostly scale-driven, not a V-path-specific signal; use this as a weak prior only).
- Does NOT cover layers 1–4 or 6–29. C-0b is localization scoped to L0 + L5.
- Does NOT cover decode-step positions beyond prefill.

---

## 5. Next Session per ADR-007 §C-4 E1-partial Branch

ADR-007 §C-4 E1 states: "If packed cache dequantizes to a value close to dense KV (within `nrmse<0.15`) and the kernel receives byte-close inputs, defect lives in H1 (kernel), H2 (FWHT), or H4 (dispatch). Next: kernel replay test with REAL decode inputs from the L=0/P=1 dumps (injected into `mlx-native/tests/test_flash_attn_vec_tq.rs`)."

C-0b's E1-partial means that the replay path is viable for the **non-batched-compact-encoded** slice, but three coverage gaps must be closed before any "encode fully cleared" claim lands.

### 5.1 C-1 session 1, required coverage

**Task A — kernel replay (the core E1-branch next step).** Extract the L=0/P=1 decode-step Q/K/V from the C-0 dumps at:

```
/tmp/cfa-20260421-C0-audit/dumps/{tq,dense}/layer00/pos01/...
```

Also extract the L=0 packed cache from the C-0b dumps at:

```
/tmp/cfa-20260421-C0b-localize/dumps/tq/hf2q_k_packed_layer00_pos22.u8.bin
/tmp/cfa-20260421-C0b-localize/dumps/tq/hf2q_k_norms_layer00_pos22.f32.bin
/tmp/cfa-20260421-C0b-localize/dumps/tq/hf2q_v_packed_layer00_pos22.u8.bin
/tmp/cfa-20260421-C0b-localize/dumps/tq/hf2q_v_norms_layer00_pos22.f32.bin
```

Construct a self-contained kernel replay test that invokes `flash_attn_vec_tq` (from mlx-native@a28783e) with these exact inputs and compares `sdpa_out` to the dense `sdpa_out` from C-0.

**Critical:** the kernel reads `[nkv, kv_capacity=1024, hd/2]` at runtime. The C-0b dumps are compacted to `[nkv, 22, hd/2]`. Before the replay, **zero-pad** each packed buffer to kv_capacity along the position axis OR re-instrument and re-dump with raw capacity-strided buffers (see Task B). Without this, the replay test will not reproduce the runtime access pattern even if Task A answers the H1 question on the compacted-layout slice.

**Task B — re-dump with raw capacity buffers + verbatim decode-call params.** Extend the HF2Q_DUMP_TQ_STATE instrumentation so:

- `k_packed` / `v_packed` are dumped at full `[nkv, kv_capacity, hd/2]` (no compaction at dump time). The kernel access pattern is then directly comparable.
- `k_norms` / `v_norms` at full `[nkv, kv_capacity]`.
- Meta JSON records the **verbatim decode-call** `mask_type`, `sliding_window`, `ring_start`, `scale`, `softcap`, `kv_seq_len`, `kv_capacity` as they appear in `FlashAttnVecTqParams` at the point of kernel invocation (`forward_mlx.rs:1272–1283`). Snapshot the struct at invocation, not at encode.

**Task C — re-run with batched prefill.** Run the dump with `HF2Q_BATCHED_PREFILL=1 HF2Q_UNSAFE_EXPERIMENTS=1` so `dispatch_hadamard_quantize_kv_seq` fires and its output lands in the dump. Diff the batched-path packed cache against the non-batched-path packed cache. Equivalence at the byte level = batched encode validated against non-batched baseline. Divergence = H3-batched is a live bug even though H3-non-batched cleared here.

Tasks A+B+C together deliver a globally-cleared E1 or identify which of H3-batched, H4 (kernel stride), or H4 (decode-call-param mismatch, e.g., mask_type) is the defect locus.

### 5.2 Decision tree for the replay test

**If the replay reproduces the C-0 divergence (max_abs_diff > 1.0 on `sdpa_out`):**
→ **H1 confirmed.** The defect is inside the `flash_attn_vec_tq` Metal shader. Next: bisect the softmax, online-max, and accumulation sections of `flash_attn_vec_tq.metal`. The sdpa_out divergence from C-0 (per-layer max nrmse 0.673–1.281) is fully reproduced from a controlled replay → the kernel is definitively broken on these inputs.

**If the replay produces a clean result (nrmse < 0.15 on `sdpa_out`):**
→ **H4 confirmed.** The defect is in caller-side dispatch (stride parameters, norm binding, ring/mask parameter handoff). Next: compare the call-site argument marshalling in `mlx-native/src/ops/flash_attn_vec_tq.rs` against the kernel's expected layout, paying particular attention to K-stride vs V-stride (consistent with the V/K asymmetry in §3.3).

**If the replay reproduces divergence BUT the divergence disappears when FWHT is disabled on Q or on `sdpa_out`:**
→ **H2 confirmed.** The defect is in the FWHT pipeline wrapping the kernel call (forward-rotate Q, inverse-rotate `sdpa_out`) in `forward_mlx.rs`. Next: bisect forward-rotate Q vs inverse-rotate `sdpa_out` as independent suspects.

### 5.3 Secondary signal (weak, scale-driven)

V max_abs_diff is 8–15× larger than K in C-0b (Table 3.3), with both passing the bound. **Codex flagged this asymmetry is mostly explained by V norms being 8–16× larger than K norms in this model (scale-driven), NOT a V-path-specific signal.** Keep this as a weak prior only. The C-1 H4 investigation should first check V vs K dispatch arguments for apples-to-apples (same stride formula, same binding, same inv_sqrt_dk), but the V/K max_abs ratio by itself does not elevate the V-path as a prime suspect over the K-path.

---

## 6. Instrumentation Disposition

The C-0b instrumentation is committed at `d14f596` on branch `cfa/cfa-20260421-111303-tq-revival/codex`. It is **NOT reverted** as of the time of this report. The spec called for revert before push; the instrumentation has not been pushed to any remote.

### 6.1 What was added (257 lines)

| File | Change | Lines |
|---|---|---|
| `src/debug/investigation_env.rs` | `pub dump_tq_state: bool`, `pub dump_tq_layers_list: Vec<usize>`, parse `HF2Q_DUMP_TQ_STATE` and `HF2Q_DUMP_LAYERS_LIST` | +~30 |
| `src/debug/dumps.rs` | `pub fn dump_u8(...)` + `pub fn dump_meta_json(...)` helpers | +~40 |
| `src/serve/forward_prefill_batched.rs` | Conditional dump block after V-seq encode dispatch | +~35 |
| `src/serve/forward_prefill.rs` | Conditional dump block after non-batched TQ encode | +~35 |

Zero changes to any `.metal` file, `hadamard_quantize_kv.rs`, `flash_attn_vec_tq.rs`, or any SDPA dispatch path. Dump is gated by `HF2Q_DUMP_TQ_STATE=0` / unset = no-op.

The C-0b spec's revert-before-push discipline was intended to prevent landing debug code on main. The instrumentation has NOT been pushed; user may choose either path below.

### 6.2 Two options

**Keep (recommended):** Land on codex branch (or main). C-1 session 1's kernel-replay test may want to re-run prefill dumps with modified inputs without re-implementing the dump infrastructure. Precedent: C-0 left `742f892` (`HF2Q_DUMP_ALL_CACHE` extension) on main; same pattern. Reverting now means re-adding in C-1 = churn.

**Revert:** Remove if C-1 session 1's kernel-replay test uses only the static dumps already at `/tmp/cfa-20260421-C0b-localize/dumps/tq/` and never needs a live re-run. Revert command:

```bash
git -C /opt/hf2q/.cfa-worktrees/cfa-20260421-111303-tq-revival-codex revert d14f596 --no-edit
```

User decides; do NOT push either way without explicit approval.

---

## 7. Process Learning

Two separate process notes from C-0b, in rough order of severity:

**(Primary) Codex caught scope overclaim that self-certification missed.** The initial draft of this report claimed the "batched-prefill encode path `dispatch_hadamard_quantize_kv_seq`" was cleared. The meta JSON records `path: "nonbatched"`. The verdict language did not match the dumped artifact. Worker 3's acceptance-criteria self-check (9/9 in `agents-reporter-result`) passed against structural checks, not against cross-artifact scope semantics. Codex's independent ground-truth reading caught three structural errors: (a) non-batched vs batched label, (b) compact vs capacity-strided dump layout, (c) meta `mask_type: 1` vs runtime kernel `mask_type: 2`. None of these were detectable by re-running the dequant script; all required reading the instrumentation source code against the dump. This is the C-0 pattern repeating in a different axis — self-certification is not a substitute for independent review of artifact semantics against production-path semantics.

Concrete prescription for future TQ localization sessions: the self-test acceptance criteria must include an **artifact-semantics cross-check** — the meta JSON fields must be grepped against the actual kernel-call site (not just the dump site) before a verdict is self-certified.

**(Secondary) Self-test gate calibration.** The queen's spec cited 0.06 without a distribution qualifier. Worker 2 measured N(0,1) round-trip at ~0.097 with correct Rust `nibble_quantize`/`nibble_dequantize` reference and adjusted to 0.12. Codex confirmed the math and codebook values. However, the 0.097 figure is specifically a Gaussian heuristic — codebook-aligned inputs can round-trip to ≤0.06. The right fix is not a number but a method: future sessions should include bad-mutant controls (off-by-one nibble order, double-rsqrt, skipped FWHT) that demonstrate the gate catches each failure mode.

The C-0 session had a similar drift pattern (Codex caught an nrmse-formula error that self-certification missed). C-0b caught the math drift in-worker before the run but missed the scope drift. Both axes — math and scope — need an independent pre-merge check; Codex serves this role.

---

## 8. Appendix

### 8.1 Reproduce commands

**Dense side (main@a258e92):**

```bash
cd /opt/hf2q
mkdir -p /tmp/cfa-20260421-C0b-localize/dumps/dense
HF2Q_DUMP_LAYERS=22 \
HF2Q_DUMP_ALL_CACHE=1 \
HF2Q_DUMP_DIR=/tmp/cfa-20260421-C0b-localize/dumps/dense \
./target/release/hf2q generate \
  --model /opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf \
  --prompt 'Complrehensive instructions for making sourdough bread.' \
  --max-tokens 2 \
  --temperature 0
```

**TQ side (codex@d14f596):**

```bash
cd /opt/hf2q/.cfa-worktrees/cfa-20260421-111303-tq-revival-codex
mkdir -p /tmp/cfa-20260421-C0b-localize/dumps/tq
HF2Q_DUMP_LAYERS=22 \
HF2Q_DUMP_TQ_STATE=1 \
HF2Q_DUMP_LAYERS_LIST=0,5 \
HF2Q_DUMP_DIR=/tmp/cfa-20260421-C0b-localize/dumps/tq \
./target/release/hf2q generate \
  --model /opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf \
  --prompt 'Complrehensive instructions for making sourdough bread.' \
  --max-tokens 1 \
  --temperature 0
```

**Re-run the dequantizer:**

```bash
python3 /opt/hf2q/scripts/tq-c0b-dequant.py --self-test
python3 /opt/hf2q/scripts/tq-c0b-dequant.py \
  --tq-root /tmp/cfa-20260421-C0b-localize/dumps/tq \
  --dense-root /tmp/cfa-20260421-C0b-localize/dumps/dense \
  --csv /opt/hf2q/docs/tq-c0b-localize-2026-04-21-raw.csv \
  --summary /opt/hf2q/docs/tq-c0b-localize-2026-04-21-summary.md
```

### 8.2 Raw CSV

Path: `/opt/hf2q/docs/tq-c0b-localize-2026-04-21-raw.csv`  
Rows: 440 (2 layers × 2 ops × variable heads × 22 positions)  
Schema: `layer,op,head,pos,max_abs_diff,nrmse`

First 5 rows:
```
layer,op,head,pos,max_abs_diff,nrmse
0,k,0,0,0.032410,0.094710
0,k,0,1,0.041312,0.090534
0,k,0,2,0.032925,0.097434
0,k,0,3,0.031131,0.088115
```

Last 5 rows:
```
5,v,1,17,0.288478,0.090961
5,v,1,18,0.307076,0.104216
5,v,1,19,0.284661,0.092643
5,v,1,20,0.319705,0.094042
5,v,1,21,0.268140,0.095073
```

### 8.3 Commit SHAs

| Item | SHA |
|---|---|
| Dense baseline (main) | `a258e92` |
| TQ + C-0b instrumentation (codex) | `d14f596` |
| mlx-native | `a28783e` |

### 8.4 CFA session metadata

| Field | Value |
|---|---|
| Session ID | `cfa-20260421-C0b-localize` |
| Date | 2026-04-21 |
| Mode | review-only |
| Worker 1 | researcher-instrumenter |
| Worker 2 | analyst-differ |
| Worker 3 | tech-writer-reporter (this report) |
| Codex review | Required before any push (ADR-007 process invariant) |

### 8.5 ADR-007 acceptance criteria status

| Criterion | Status |
|---|---|
| Report exists at `docs/tq-c0b-localize-2026-04-21.md` with all 8 sections | YES — this file |
| Verdict is exactly one of {E1, E2, E3} per naming rules | YES — **E1** |
| Dumped file sizes match declared shapes | YES — confirmed by Worker 1 (sizes consistent with nkv×22×hd/2 for packed u8, nkv×22×4 for norms f32) |
| Dequantizer self-test passes nrmse < 0.12 | YES — 0.09728 (hd=512 worst case) |
| Codebook-identity assertion (Python == Rust) | YES — codebook values hardcoded in script with byte-identical citation to `turboquant.rs:27–32` |
| Dense-side filename scheme aligned with main's dump output | YES — `hf2q_cache_{k,v}_layer{LL}_pos22.bin` on dense, `hf2q_{k,v}_packed_layer{LL}_pos22.u8.bin` on TQ; Worker 2 script performs the explicit mapping |
| Instrumentation patch NOT pushed | YES — local commit only, not pushed |
| Codex review requested before push | PENDING — this report is the Codex-review input |
| Commit not pushed without explicit user approval | YES — held |
