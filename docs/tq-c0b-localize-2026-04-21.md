# TQ C-0b Localization Audit — 2026-04-21

**CFA session:** `cfa-20260421-C0b-localize`  
**Mode:** review-only (Worker 3 — tech-writer-reporter)  
**Date:** 2026-04-21  
**Authors:** Workers 1 (researcher), 2 (analyst), 3 (reporter)  
**References:** ADR-007 §C-0b, ADR-007 §C-4 E1/E2/E3 branching

---

## 1. Executive Summary

**Verdict: E1 — H3 (prefill encode/cache) cleared. Defect lives in {H1 kernel / H2 FWHT / H4 dispatch}.**

The TQ-packed KV cache produced by the batched-prefill encode path — `dispatch_hadamard_quantize_kv_seq` on codex@d14f596 — dequantizes to within the kernel's own declared bounds on every one of 440 measured cells (2 layers × K+V × heads × 22 positions). Zero violations of nrmse < 0.15 or max_abs_diff < 1.0. The worst nrmse across all cells is **0.1390** (layer 0, K, head 1, position 20). The worst max_abs_diff is **0.4450** (layer 5, V, head 0, position 12).

This proves: the packed cache + per-position norms at end-of-prefill correctly reconstruct KV within the kernel's declared tolerance on both the sliding-attention layer (L0, hd=256) and the global-attention sanity layer (L5, hd=512).

This does NOT prove which of H1/H2/H4 is the defect locus. Next per ADR-007 §C-4 E1: kernel replay test with real L=0/P=1 Q/K/V inputs injected into `mlx-native/tests/test_flash_attn_vec_tq.rs`.

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

**TQ side (codex@d14f596):** Dump fires at end-of-prefill in `forward_prefill_batched.rs` immediately after the TQ V-seq encode dispatch (`dispatch_hadamard_quantize_kv_seq`), gated by `HF2Q_DUMP_TQ_STATE=1`. Per-layer outputs written to `/tmp/cfa-20260421-C0b-localize/dumps/tq/`:

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

### 2.6 Self-test gate: 0.06 → 0.12 adjustment

The queen's spec cited 0.06 as the gate for the dequantizer self-test. Worker 2 flagged this and adjusted to **0.12**. Rationale:

The `nrmse < 0.15` bound in `test_flash_attn_vec_tq.rs` is the safety margin for SDPA *output* (post attention averaging across many K/V vectors), not per-vector round-trip error. The mathematical floor for 4-bit Lloyd-Max encode+decode on random N(0,1) vectors under this specific nrmse formula is approximately **0.097** — confirmed by running the self-test with the Rust `nibble_quantize`/`nibble_dequantize` reference in `mlx-native/tests/test_flash_attn_vec_tq.rs`. Measured self-test values: hd=256 → 0.097148, hd=512 → 0.097281.

A gate of 0.06 would unconditionally reject a correct 4-bit implementation. A gate of 0.12 catches all implementation bugs (wrong nibble order, wrong codebook, wrong FWHT normalization, rsqrt double-application) while passing the genuine quantization noise floor.

**Flag for Codex review:** This gate adjustment is Worker 2's independent judgment. Codex should verify: (a) the 0.097 floor is confirmed by the Rust reference, and (b) 0.12 is a sufficient margin above 0.097 to catch bugs while not being so tight as to fail correct implementations on edge-case input distributions.

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

**E1.** The packed cache is not the defect locus.

Logic chain:

1. **Given:** Paired F32 dumps at identical prompt + seed (sourdough, T=0), same kv_seq_len=22 tokens on both sides. Dense side: main@a258e92. TQ side: codex@d14f596.
2. **Given:** Python dequantizer validated by round-trip self-test — nrmse 0.0973 on 1000 N(0,1) vectors at hd=256 and hd=512, both within 4-bit Lloyd-Max mathematical floor and below the 0.12 gate.
3. **And:** Dequantized TQ cache vs dense F32 cache — **max nrmse across all 440 cells is 0.1390** (< kernel bound 0.15).
4. **And:** **Max_abs_diff across all 440 cells is 0.4450** (< kernel bound 1.0).
5. **Therefore:** The packed cache produced by `dispatch_hadamard_quantize_kv_seq` (batched prefill encode on codex@d14f596) plus per-position norms reconstruct KV within the kernel's own declared tolerance at both sliding (L0, hd=256) and global (L5, hd=512) layers.
6. **Therefore: H3 (prefill encode/cache) is NOT the defect locus.**

What this verdict does NOT claim:

- Does NOT claim the kernel is correct. H1 (`flash_attn_vec_tq` Metal shader math), H2 (FWHT pipeline wrapping the kernel call in `forward_mlx.rs`), and H4 (caller-side dispatch — strides, norm binding, ring/mask params) are all still live hypotheses.
- Does NOT rank H1/H2/H4 against each other beyond the secondary V/K asymmetry signal.
- Does NOT cover layers 1–4 or 6–29. C-0b is a localization experiment scoped to L0 (primary) + L5 (sanity).
- Does NOT cover decode-step positions beyond prefill. The dump is taken at end-of-prefill; decode-written slots are not in scope.

---

## 5. Next Session per ADR-007 §C-4 E1 Branch

ADR-007 §C-4 E1 states: "If packed cache dequantizes to a value close to dense KV (within `nrmse<0.15`) and the kernel receives byte-close inputs, defect lives in H1 (kernel), H2 (FWHT), or H4 (dispatch). Next: kernel replay test with REAL decode inputs from the L=0/P=1 dumps (injected into `mlx-native/tests/test_flash_attn_vec_tq.rs`)."

### 5.1 C-1 session 1, first task

Extract the L=0/P=1 decode-step Q/K/V from the C-0 dumps at:

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

### 5.2 Decision tree for the replay test

**If the replay reproduces the C-0 divergence (max_abs_diff > 1.0 on `sdpa_out`):**
→ **H1 confirmed.** The defect is inside the `flash_attn_vec_tq` Metal shader. Next: bisect the softmax, online-max, and accumulation sections of `flash_attn_vec_tq.metal`. The sdpa_out divergence from C-0 (per-layer max nrmse 0.673–1.281) is fully reproduced from a controlled replay → the kernel is definitively broken on these inputs.

**If the replay produces a clean result (nrmse < 0.15 on `sdpa_out`):**
→ **H4 confirmed.** The defect is in caller-side dispatch (stride parameters, norm binding, ring/mask parameter handoff). Next: compare the call-site argument marshalling in `mlx-native/src/ops/flash_attn_vec_tq.rs` against the kernel's expected layout, paying particular attention to K-stride vs V-stride (consistent with the V/K asymmetry in §3.3).

**If the replay reproduces divergence BUT the divergence disappears when FWHT is disabled on Q or on `sdpa_out`:**
→ **H2 confirmed.** The defect is in the FWHT pipeline wrapping the kernel call (forward-rotate Q, inverse-rotate `sdpa_out`) in `forward_mlx.rs`. Next: bisect forward-rotate Q vs inverse-rotate `sdpa_out` as independent suspects.

### 5.3 Secondary signal to watch

V max_abs_diff is 8–15× larger than K in C-0b (Table 3.3), with both passing the bound. If H4 is confirmed, this asymmetry is consistent with a V-path-specific parameter bug — e.g., wrong V stride, wrong V-norms buffer binding, or V-loop `inv_sqrt_dk` applied differently. Worth checking the V-path dispatch arguments first in the H4 investigation.

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

The self-test gate came into this session from the queen's spec at 0.06, copied from the kernel unit test's `nrmse < 0.15` assertion scaled down by an unstated factor. Worker 2 flagged independently that 0.06 is below the mathematical floor for correct 4-bit Lloyd-Max under this nrmse formula on random N(0,1) vectors (~0.097, confirmed by Rust `nibble_quantize`/`nibble_dequantize` reference). Worker 2 adjusted to 0.12.

This mirrors the C-0 session's nrmse-formula drift: in C-0, Codex caught a formula error that Claude's own 9/9 self-certification missed. In C-0b, the formula drift was caught in-worker before the run, not after. This is the intended behavior for the review-only CFA mode.

**Flag for Codex:** Verify that the gate adjustment from 0.06 to 0.12 is legitimate (the 0.097 floor is derived from the correct implementation, not from a buggy one), and that the spec's original 0.06 was indeed a copy-error rather than a deliberate tighter-than-floor requirement.

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
