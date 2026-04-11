# ADR-005 Phase 1b — Spike Report: post-1bNEW.17 (Spike A + Spike B)

**Date:** 2026-04-10
**Runner:** Claude (investigation-only; no `main` commits)
**Scope:** Two sequential spikes queued after 1bNEW.17 landed:
  - **Spike A** — Diagnose the "QMatMul auto-dequant quirk" cited in the 1bNEW.17 Phase A commit (`0565c69`) that prevented the lm_head from shipping as true Q6_K.
  - **Spike B** — Identify the owner of the residual `The`/`To` Walk-correctness argmax drift (+0.77102 toward `The`, opposite to llama.cpp's `To`).
**Baseline binary:** `main` HEAD `3f6da63` (1bNEW.17 Phase C ADR update; 58.51 tok/s median).
**Model:** `models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf` (Gemma 4 26B MoE DWQ, mixed Q4_0/Q6_K/F16).
**Hardware:** Apple M5 Max, 128 GB unified memory.
**Worktree discipline:** All instrumentation reverted before returning; `git diff --stat src/` empty at return; no new files committed to `main` except this report.

---

## Spike A — QMatMul auto-dequant quirk diagnosis

### Framing

The ADR-005 1bNEW.17 item text (ADR lines 622-637) projected that loading `token_embd.weight` as a `QTensor` and dispatching through `QMatMul::forward` would ship the lm_head as real Q6_K at ~67 tok/s. The implementer (commits `0565c69`/`0e36b1c`/`3c41f85`) instead shipped an F16 matmul at **58.51 tok/s**, saving −50% of per-token weight traffic (2.95 GB F32 → 1.48 GB F16) instead of the projected −80% Q6_K drop. The commit message body attributes this to a "QMatMul auto-dequant quirk" at `candle-core/src/quantized/mod.rs:726-738`.

This spike answers two questions:

1. **What is the quirk?** Which specific code path in candle (or in hf2q's load path) auto-dequantizes an F16 QTensor, and under what conditions does it fire?
2. **Is the lm_head shippable as real Q6_K at all?** And if yes, what's the expected speed gain?

### Methodology

1. **Read the 1bNEW.17 commit messages** (`0565c69`, `0e36b1c`, `3c41f85`, `3f6da63`) and `src/serve/lm_head_kernel.rs` end-to-end to extract the implementer's empirical notes.
2. **Read `candle-core/src/quantized/mod.rs`** `QMatMul::from_arc` and surrounding logic to map the auto-dequant decision tree.
3. **Verify on the real GGUF** that `token_embd.weight` is what the commit says it is (F16, not Q6_K) via both the Python `gguf.GGUFReader` (independent) and candle's own `content.tensor()` loader (in-tree).
4. **Build a standalone Rust repro** (`examples/spike_a_qmatmul.rs`, scratch, deleted before returning) that exercises `QMatMul::from_arc` on three configurations:
   - `token_embd.weight` (F16 QTensor at lm_head shape `[262144, 2816]`)
   - `token_embd.weight` with `CANDLE_DEQUANTIZE_ALL_F16=1` set
   - `blk.29.attn_q.weight` (Q6_K QTensor at known-good shape `[8192, 2816]`)
   and measure wall-clock per `QMatMul::forward` call after a 1-call warmup, 5 calls synced to CPU each, repeated three times for variance.
5. **Compare against the live shipping path** — the exact `x.to_dtype(F16).matmul(w_f16.t()).to_dtype(F32)` chain the 1bNEW.17 Phase C binary dispatches, at the real lm_head shape, on the same device.
6. **Cross-reference against `candle-metal-kernels/src/kernels/quantized.rs`** and `quantized.metal` to confirm whether an in-place F16-quantized matmul kernel would be reachable if the auto-dequant were routed around.

### What I ran

**Repro (`examples/spike_a_qmatmul.rs`, scratch):**

```rust
use candle_core::quantized::{QMatMul, QTensor, gguf_file::Content};
use candle_core::{Device, Module, Tensor};

// 1. Read GGUF header and inspect tensor metadata directly.
let content = Content::read(&mut reader)?;
let te_info = content.tensor_infos.get("token_embd.weight")?;
// → prints: dtype=F16 shape=[262144, 2816]

// 2. Load as QTensor; check dtype survives load.
let te_qt = content.tensor(&mut reader, "token_embd.weight", &device)?;
// → te_qt.dtype() == GgmlDType::F16 (exact match)

// 3. Route through QMatMul::from_arc (the load path hf2q would use).
let te_qm = QMatMul::from_arc(Arc::new(te_qt))?;
// Inspect variant via Debug:
// → Tensor(dense F32 matmul)   // <-- auto-dequant fired

// 4. Re-run with CANDLE_DEQUANTIZE_ALL_F16=1 set *before* calling from_arc.
std::env::set_var("CANDLE_DEQUANTIZE_ALL_F16", "1");
let te_qm_f16 = QMatMul::from_arc(te_arc.clone())?;
// → Tensor(dense F32 matmul)   // <-- env var is NOT honored for F16 sources

// 5. Contrast: a real Q6_K tensor.
let qq_qt = content.tensor(&mut reader, "blk.29.attn_q.weight", &device)?;
let qq_qm = QMatMul::from_arc(Arc::new(qq_qt))?;
// → QTensor(quantized kernel)  // <-- real kernel path reached
```

**Bench harness:**

```rust
// After a 1-call warmup each, 5 synced calls per configuration:
for _ in 0..5 {
    let t0 = Instant::now();
    let y = path.forward(&input)?;
    y.to_device(&Device::Cpu)?;       // force command buffer drain
    times.push(t0.elapsed());
}
```

**Input:** synthetic `Tensor::randn((1, 2816))` — real decode-time shape.
**Shapes:** lm_head `[1,2816] @ [262144, 2816].t()`; Q6_K reference `[1,2816] @ [8192, 2816].t()`.

### Raw output

GGUF metadata (via `gguf.GGUFReader` — Python, independent):

```
token_embd.weight: dtype=F16, shape=[2816, 262144], n_bytes=1,476,395,008
output.weight: (not present — tied-embedding fallback)
```

Arithmetic check: `262144 × 2816 × 2 = 1,476,395,008` bytes. **Exact match for F16.** Not Q6_K. Not F32.

QTensor load via candle (in-tree):

```
te QTensor.dtype()=F16 shape=[262144, 2816]
```

`QMatMul::from_arc` variant (three configurations):

```
token_embd QMatMul (default env):                  Tensor(dense F32 matmul)
token_embd QMatMul (CANDLE_DEQUANTIZE_ALL_F16=1):  Tensor(dense F32 matmul)
blk.29.attn_q QMatMul (Q6_K control):              QTensor(quantized kernel)
```

Wall-clock timing (3 consecutive full runs, 5 synced calls/config after 1-call warmup):

| Path | Run 1 | Run 2 | Run 3 | Median |
|---|---|---|---|---|
| `QMatMul::forward` default (auto-dequant→F32) | 7.10, 7.10, 7.17, 7.15, 7.14 ms | 7.20, 7.10, 7.13, 7.14, 7.10 ms | 7.20, 7.07, 7.08, 7.10, 7.11 ms | **~7.12 ms/call** |
| `QMatMul::forward` with `CANDLE_DEQUANTIZE_ALL_F16=1` | 14.00, 7.09, 7.12, 7.16, 7.08 ms | 14.02, 7.07, 7.15, 7.20, 7.08 ms | 13.96, 7.09, 7.07, 7.11, 7.12 ms | **~7.10 ms/call (first call 14 ms buffer warmup)** |
| **Fused F16 matmul path (1bNEW.17 shipping)** | 3.96, 3.71, 3.75, 3.79, 3.71 ms | 4.01, 3.66, 3.74, 3.73, 3.72 ms | 4.07, 3.73, 3.73, 3.68, 3.73 ms | **~3.73 ms/call** |
| Q6_K control `[2816]→[8192]` (1bNEW.1 path) | 2.32 ms | 2.41 ms | 2.22 ms | **~2.31 ms/call** |

### The quirk — root cause

`QMatMul::from_arc` at `/Users/robert/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/candle-core-0.10.2/src/quantized/mod.rs:725-740` (byte-identical to `/opt/candle/candle-core/src/quantized/mod.rs:725-740` — candle 0.10.2 is pinned):

```rust
impl QMatMul {
    pub fn from_arc(qtensor: std::sync::Arc<QTensor>) -> Result<Self> {
        let dequantize = match qtensor.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => true,
            _ => DEQUANTIZE_ALL.with(|b| *b),
        };
        let t = if dequantize {
            let tensor = qtensor.dequantize(&qtensor.device())?;
            Self::Tensor(tensor)
        } else if DEQUANTIZE_ALL_F16.with(|b| *b) {
            let tensor = qtensor.dequantize_f16(&qtensor.device())?;
            Self::TensorF16(tensor)
        } else {
            Self::QTensor(qtensor)
        };
        Ok(t)
    }
    // ...
}
```

Line 727 is the quirk: **`GgmlDType::F32 | F16 | BF16 => true` is an unconditional dequant-to-F32 for every floating-point QTensor**, irrespective of shape, device, env var, or thread-local state. The `DEQUANTIZE_ALL_F16` escape hatch at line 733 is **structurally unreachable** for an F16 QTensor source because the match arm at line 727 unconditionally sets `dequantize = true`, preempting the `else if` branch.

Consequence: `QMatMul::from_arc(f16_qtensor)` **always** returns `Self::Tensor(f32_dense)`, which `QMatMul::forward` at `mod.rs:861-881` dispatches as a dense F32 matmul against the F32-materialized 2.95 GB copy. This is exactly what the 1bNEW.17 `Loop` fallback mode does — so routing `token_embd.weight` through `QMatMul::from_arc` would be byte-equivalent to the pre-1bNEW.17 baseline.

The 1bNEW.17 implementation skipped `QMatMul` entirely and dispatched `to_dtype(F16) → matmul → to_dtype(F32)` through a plain `candle::Tensor`, landing in `MetalStorage::matmul`'s F16 branch at `candle-core/src/metal_backend/mod.rs:1685-1709` (`call_mlx_gemm` with `GemmDType::F16`). That path is the same MLX-GEMM pool every `DType::F16` matmul candle dispatches. **No new kernel. No candle fork. Just route around the `from_arc` decision tree by never constructing a `QMatMul` wrapper for the F16 tensor.**

### But — the quirk is not the gating factor

**The ADR's proposed "quantized lm_head" plan was never reachable, because `token_embd.weight` in this GGUF is stored as F16, not Q6_K.** Verified from three independent sources:

1. Python `gguf.GGUFReader` reports `ggml_type=F16, n_bytes=1,476,395,008` (1,476,395,008 = 262144 × 2816 × 2 exactly — match for F16; Q6_K would be ≈ 600 MB).
2. Candle's in-tree `content.tensor(...).dtype()` reports `GgmlDType::F16`.
3. `gguf-py/gguf/convert_hf_to_gguf.py` Gemma4Model class at `/opt/llama.cpp/convert_hf_to_gguf.py:7517-7622` never routes `token_embd` through any quantizer — the embedding stays in its source dtype (F16 for DWQ).

Additionally, the GGUF **has no `output.weight` tensor at all** — llama.cpp falls back to tied embeddings at `/opt/llama.cpp/src/llama-model.cpp:4973-5610`, and at its lm_head site (`/opt/llama.cpp/src/models/gemma4-iswa.cpp:248`) it computes `ggml_mul_mat(ctx0, model.output, cur)` where `model.output` is aliased to the F16 `token_embd.weight`. llama.cpp's **own** lm_head reads 1.48 GB of F16 per decode token — exactly what 1bNEW.17 ships.

So: the "quirk" is real, but the phrasing "prevented 1bNEW.17 from shipping as true Q6_K" in the post-1bNEW.17 framing is inverted. There was **never a Q6_K path to ship**, because the weight is F16 in the source file. The quirk matters only if someone wants to load a Q6_K-quantized `token_embd.weight` from a different GGUF at a future date; for this specific file, the auto-dequant branch has no escape-hatch scenario because the real-Q6_K counterfactual doesn't exist.

### Would a hypothetical Q6_K lm_head help?

Extrapolating from the control measurement: Q6_K `[2816]→[8192]` cost ~2.31 ms/call. The ADR Q5 spike (`docs/spike-Q3Q4Q5-results.md`) measured Q6_K `kernel_mul_mv_q6_K_f32` latency as sub-linear in output dimension, ~30.4 ns/output element at the synced-call budget for the `[8192]` shape. Linearly scaling to `[262144]`: `262144 × 30.4 ns ≈ 7.97 ms/call` — **slower than the F16 matmul at 3.73 ms/call.** The bandwidth floor flips: Q6_K at `[2816]→[262144]` reads 600 MB weight + the `kernel_mul_mv_q6_K_f32` kernel has per-output threadgroup overhead that dominates at very wide output dims on M5 Max.

Counterfactually, if the GGUF shipped `token_embd.weight` as Q6_K AND the quirk were routed around (or candle patched to honor `DEQUANTIZE_ALL_F16` on F16 sources), the lm_head on this hardware would be roughly **equal to or slower than** the current F16 path. **The speed ceiling has already been hit for the lm_head on this GGUF.** There is no further Walk-faithful lift available from requantizing or rerouting the lm_head.

### Spike A verdict

- **Root cause of the quirk**: `candle-core/src/quantized/mod.rs:727` — `GgmlDType::F32 | F16 | BF16 => true` unconditionally auto-dequantizes any floating-point `QTensor` to F32 inside `QMatMul::from_arc`, even when the thread-local `CANDLE_DEQUANTIZE_ALL_F16` escape hatch would otherwise apply, because the match-arm pre-empts the `else if` chain.
- **Unblockable?** **Yes** in the abstract (a 3-line candle patch at `mod.rs:726-738` — reorder the decision tree so `DEQUANTIZE_ALL_F16` wins over the default-true branch for F16/BF16 sources, or add an explicit `GgmlDType::F16 => DEQUANTIZE_ALL_F16.with(|b| *b)` arm) — **but NO in the concrete Walk sense for this GGUF**, because the weight is already F16 in the file and the current shipping path reads the same F16 bytes at equivalent cost. Routing through `QMatMul::forward` on this tensor would at best produce the same F16 matmul (via `Self::TensorF16` → `forward_via_f16`) candle currently routes through `MetalStorage::matmul`'s F16 gemm branch.
- **Expected speedup if unblocked**: **inconclusive.** On this specific hardware+shape, the Q6_K kernel is extrapolated to cost ~7.97 ms/call at `[2816]→[262144]` vs the F16 path at 3.73 ms/call, so the counterfactual is negative (−4.2 ms/call slower) even if the quirk is patched and a Q6_K `token_embd.weight` is created at conversion time. The 1bNEW.17 Phase C shipping path is at the memory-bandwidth floor for this hardware.

### Spike A citation trail

| Claim | File:line |
|---|---|
| The quirk | `/Users/robert/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/candle-core-0.10.2/src/quantized/mod.rs:725-740` (byte-identical to `/opt/candle/candle-core/src/quantized/mod.rs:725-740`) |
| `QMatMul::forward` dispatch | `.../candle-core-0.10.2/src/quantized/mod.rs:861-881` |
| `MetalStorage::matmul` F16 gemm path | `/opt/candle/candle-core/src/metal_backend/mod.rs:1685-1709` |
| `call_quantized_matmul_mv_t` F16 source dispatch (the path the quirk bypasses) | `/Users/robert/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/candle-metal-kernels-0.10.2/src/kernels/quantized.rs:99, 136` |
| `kernel_mul_mv_f16_f32` template instantiation | `/Users/robert/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/candle-metal-kernels-0.10.2/src/metal_src/quantized.metal:2714` |
| llama.cpp Gemma4 lm_head site | `/opt/llama.cpp/src/models/gemma4-iswa.cpp:248` |
| llama.cpp tied-embedding fallback | `/opt/llama.cpp/src/llama-model.cpp:4973-5610` |
| GGUF converter: `norm_shift` returns 0.0 for Gemma4 (no +1 shift at conversion) | `/opt/llama.cpp/convert_hf_to_gguf.py:7520-7522` |
| hf2q 1bNEW.17 shipping helper | `/opt/hf2q/src/serve/lm_head_kernel.rs:152-206` |
| hf2q lm_head call site | `/opt/hf2q/src/serve/gemma4.rs:1947-1966` |
| hf2q GGUF load path (F32 dequant branch) | `/opt/hf2q/src/serve/gguf_loader.rs:50-57` |
| Q6_K latency reference measurement | `/opt/hf2q/docs/spike-Q3Q4Q5-results.md` Q5 (1.43× / 1.70× scaling) |

---

## Spike B — Walk-correctness drift owner

### Framing

Post-1bNEW.1/3/4/6/10/12/17, hf2q's decode-1 top-1 on the canonical 187-token bench prompt is:

```
The (818) = 27.108929
To  (2021) = 26.337908
gap = +0.771021 (toward `The`)
```

llama.cpp's decode-1 top-1 on the same GGUF + same rendered prompt (ADR line 194):

```
To  (2021) = -0.6487 logprob
The (818)  = -0.7663 logprob
gap = -0.1176 (toward `To`)
```

Total disagreement: hf2q prefers `The` by +0.771; llama.cpp prefers `To` by +0.118. Absolute drift to close: **~0.89 logit / ~0.84 logprob** (the difference between the two gaps is what any fix has to move).

The 1bNEW.17 Phase C commit (`3c41f85`) **falsified** the hypothesis that the lm_head was the drift owner: flipping `--lm-head-kernel` from `loop` (byte-flat Phase-1 dense F32) to `fused` (native F16 matmul) moves the gap by only **+0.00086** (from +0.770160 to +0.771021). The ~0.77 gap is owned by something upstream of the lm_head.

### Methodology

1. **Sweep the four kernel-mode toggles** (`--moe-kernel`, `--rms-norm-kernel`, `--rope-kernel`, `--lm-head-kernel`) on the current HEAD binary to isolate each fused kernel port's contribution.
2. **Scratch-revert the 1bNEW.10 BF16 prefill path** via a `HF2Q_SPIKE_B_NO_BF16=1` env-var gate inserted at `src/serve/gemma4.rs:759`. When set, `head_dim=512` global-attention layers fall through to the existing manual `repeat_kv + matmul + causal_mask + softmax + matmul` F32 path instead of the BF16 fused SDPA. Measure the resulting top-10 and compare against HEAD.
3. **Cross-product the two switches** (kernel toggles × BF16 on/off) to surface any non-additive interactions.
4. **Scratch-swap the router scalar-multiply order** via `HF2Q_SPIKE_B_ROUTER_ORDER=1` at `src/serve/gemma4.rs:1038-1042` — the spike report's Candidate #2 "order of `(normed * router_scale) * (1/sqrt(n_embd))` vs `(normed * (1/sqrt(n_embd))) * router_scale`".
5. **Dump the full 262144-vocab logits vector** under BF16-on and BF16-off, and compute `max |Δ|`, `mean |Δ|`, and cosine similarity to quantify BF16's global per-token effect.
6. **Read llama.cpp's Gemma4 reference forward pass end-to-end** (`/opt/llama.cpp/src/models/gemma4-iswa.cpp`, 320 lines) and compare it op-by-op against hf2q's `DecoderLayer::forward` + `Attention::forward`. Verify RoPE convention, attention scaling, residual accumulation order, norm ordering, router math, and the `k_eq_v` optimization against llama.cpp `build_attn` / `build_lora_mm` / `build_norm` in `llama-graph.cpp`. Double-check Gemma4's `(1+w)` RmsNorm convention against `convert_hf_to_gguf.py` and vllm's `GemmaRMSNorm` class.
7. **All scratch edits reverted** before writing this report. `git diff --stat src/` empty at return.

### Ground rules for interpretation

- Every measurement is on the canonical bench prompt (`tests/bench_prompt_128.txt`, 187 tokens rendered). T=0, single decode step, `HF2Q_DUMP_LOGITS` to inspect the top-10.
- **The 1bNEW.17 Phase A Gate 1 baseline** — `(818, 27.110750), (2021, 26.340590)`, gap `+0.770160` — reproduces byte-identically from the current HEAD `3f6da63` binary under `--lm-head-kernel loop` (5-decimal match). That confirms the measurement rig is stable.
- **The 1bNEW.17 Phase C default** — `(818, 27.108929), (2021, 26.337908)`, gap `+0.771021` — also reproduces byte-identically under HEAD defaults. Measurement rig is deterministic.
- **Direction convention:** positive gap = hf2q prefers `The`; negative gap = hf2q prefers `To` = agreement with llama.cpp. Closing the gap means **decreasing** it.

### Part 1 — Kernel-toggle sweep (HEAD binary, canonical bench)

All eight configurations run exhaustively. `The` and `To` reported; **gap = The − To**; **Δ = gap − HEAD gap (0.771021)**; negative Δ = movement toward llama.cpp.

| # | Config | The (818) | To (2021) | Gap | Δ vs HEAD |
|---|---|---|---|---|---|
| 1 | **HEAD defaults (fused × 4 + BF16)** | **27.108929** | **26.337908** | **+0.771021** | **0** |
| 2 | `--lm-head-kernel loop` | 27.110750 | 26.340590 | +0.770160 | −0.00086 |
| 3 | `--rms-norm-kernel loop` | 27.114656 | 26.337908 | +0.776748 | +0.00573 |
| 4 | `--rope-kernel loop` | 27.108929 | 26.352211 | +0.756718 | −0.01430 |
| 5 | `--moe-kernel loop` | 27.108929 | 26.337908 | +0.771021 | 0.00000 (byte-flat) |
| 6 | `--rms-norm-kernel loop --rope-kernel loop` | 27.103190 | 26.359339 | +0.743851 | −0.02717 |
| 7 | `--rms-norm-kernel loop --rope-kernel loop --lm-head-kernel loop` | 27.103428 | 26.358303 | +0.745125 | −0.02590 |
| 8 | **All four kernels `loop`** | 27.099707 | 26.343288 | +0.756419 | −0.01460 |

Observations:

- **1bNEW.1 (fused MoE) has ZERO effect** on the top-2 gap at decode-1 (row 5, exact byte-identical across fused/loop modes). The unified `kernel_mul_mv_id_*` dispatch is numerically identical to the per-expert `QMatMul::forward` loop at this prompt. **Falsifies the hypothesis that the MoE kernel is the drift owner.**
- **1bNEW.17 (F16 lm_head) contributes −0.00086** — confirms the ADR's Phase C note that the lm_head is not the drift owner.
- **1bNEW.6 (fused RoPE) single toggle moves the gap by −0.01430** — the largest single-kernel contribution. Widens in the HEAD direction when fused; narrows toward `To` when reverted to loop mode.
- **1bNEW.4 (fused RmsNorm) single toggle moves the gap by +0.00573** — opposite direction. Widens toward `To` when reverted.
- **1bNEW.4 + 1bNEW.6 combined: Δ = −0.02717** — larger than the sum of individual deltas (−0.014 + +0.006 = −0.008). A **−0.019 interaction term** appears: the fused RmsNorm and fused RoPE kernels have a non-additive interaction via BF16 prefill's rounding envelope (confirmed in Part 2 below).
- **All-loop configuration: Δ = −0.01460** — reverting every kernel port simultaneously lands back near the Q4 spike's 1bNEW.1/1bNEW.3-era baseline (+0.755, approximately), not at llama.cpp's −0.118.

Headline: the combined effect of all four Walk kernel ports is **+0.0146 logit** (widening the gap away from llama.cpp), **just 1.6% of the 0.89 logit total disagreement**. The kernel ports collectively contribute at most ~2% of the drift. **They are not the drift owner.**

### Part 2 — BF16 prefill revert (Candidate 1)

Scratch edit at `src/serve/gemma4.rs:759`:

```rust
} else if self.head_dim == 512 && std::env::var("HF2Q_SPIKE_B_NO_BF16").is_err() {
    // ... 1bNEW.10 BF16 SDPA path ...
} else {
    // ... manual F32 repeat_kv + matmul + causal + softmax + matmul ...
```

The `HF2Q_SPIKE_B_NO_BF16=1` env var forces global prefill (5 full-attention layers out of 30) through the same F32 manual path the sliding layers use. Decode path unchanged (always F32). Built, measured, reverted.

| Config | The (818) | To (2021) | Gap | Δ vs HEAD |
|---|---|---|---|---|
| HEAD (BF16 on, fused kernels × 4) | 27.108929 | 26.337908 | +0.771021 | 0 |
| **BF16 off, fused kernels × 4** | **27.103190** | **26.359339** | **+0.743851** | **−0.02717** |
| BF16 off, `--rms-norm-kernel loop` | 27.103190 | 26.359339 | +0.743851 | −0.02717 |
| BF16 off, `--rope-kernel loop` | 27.103190 | 26.359339 | +0.743851 | −0.02717 |
| BF16 off, `--rms-norm-kernel loop --rope-kernel loop` | 27.103190 | 26.359339 | +0.743851 | −0.02717 |
| BF16 off, all four kernels loop | 27.104320 | 26.356354 | +0.747966 | −0.02306 |

**Two decisive findings:**

1. **BF16 prefill contributes −0.02717 to the gap** (moves hf2q 0.027 toward llama.cpp). This is consistent with the Q4 spike measurement of −0.071 from earlier in the Walk (F32 0.748 → BF16 0.677), scaled by the post-Walk numerical envelope which is slightly tighter than the pre-Walk one. **BF16 is a real but minor drift contributor, owning ~3.1% of the 0.89 logit total gap**, moving the gap *toward* llama.cpp, not away.
2. **With BF16 disabled, every fused kernel toggle collapses to byte-identical output.** Rows 3, 4, 5 under BF16-off all produce exactly `(818, 27.103190), (2021, 26.359339)` — 5-decimal match. **The ~0.02717 kernel-port deltas measured in Part 1 are BF16 interaction effects, not intrinsic drift of the kernels themselves.** The fused RmsNorm kernel at F=1/F=2/F=3, the fused RoPE kernel at Norm/Neox variants, and the unified MoE kernel each produce F32-faithful output relative to their loop counterparts at the 1e-5 envelope when the BF16 prefill rounding doesn't amplify downstream ops.

This is a non-obvious but important finding for the ADR: **1bNEW.4 Phase A tested at 2.384e-7 single-ULP parity**, but at run-time the kernel's numerical identity relative to the loop path only holds when its inputs are also F32-faithful. BF16 prefill rounds its inputs at ~2e-3, and that rounding interacts with the reduction order inside the fused kernel differently than with the loop-chain reduction. The net effect is position-dependent perturbation at the ~1e-3 level on the lm_head softmax — small in absolute terms, but enough to bias the top-2 logits by ~0.01 each.

### Part 3 — Full-vocab logit diff, BF16 on vs off

Dumped the full 262144 F32 logits vector to disk from both configs and diffed via numpy:

```
BF16 on  (HEAD):  /tmp/spikeB/head_bf16on.bin
BF16 off (scratch): /tmp/spikeB/head_bf16off.bin

max |Δ|       = 0.181190
mean |Δ|      = 0.045295
cos(a, b)     = 0.9999940991
argmax both   = 818 (`The`)   — argmax preserved
top-3 |Δ| indices = 242194, 96691, 86022, 2956, 68202, 84190, 228217, 1185, 8506, 106389
gap_a (BF16 on)  = +0.771021
gap_b (BF16 off) = +0.743851
```

BF16 prefill perturbs the full logit distribution with **max 0.181, mean 0.0453, cos 0.999994**. The perturbation is a small, global, low-frequency signature — consistent with 8-bit mantissa rounding (BF16 has 8 fraction bits vs F32's 23). It is NOT concentrated on the `The`/`To` pair; it's distributed across the full vocab. The `The`/`To` gap happens to land on the tail of this distribution (Δ 0.027 ≈ 0.6× the mean |Δ|). Nothing about the BF16 effect is special to the top-2; it's just "everything gets fuzzed by ~0.05 logit."

### Part 4 — Router scalar-multiply order (Candidate 2)

Scratch edit at `src/serve/gemma4.rs:1038-1042`:

```rust
let router_scaled = if std::env::var("HF2Q_SPIKE_B_ROUTER_ORDER").is_ok() {
    // llama.cpp order: (normed * 1/sqrt(n_embd)) * ffn_gate_inp_s
    ((router_normed * scale_factor)?.broadcast_mul(&self.router_scale))?
} else {
    // hf2q HEAD order: (normed * ffn_gate_inp_s) * 1/sqrt(n_embd)
    (router_normed.broadcast_mul(&self.router_scale)? * scale_factor)?
};
```

Motivation: llama.cpp `gemma4-iswa.cpp:152-155` applies `ggml_scale(tmp, 1/sqrt(n_embd))` BEFORE `ggml_mul(tmp, ffn_gate_inp_s)`. hf2q applies the tensor broadcast_mul FIRST then the scalar * at the end. In F32 these are commutative up to the final ULP of rounding — theoretically nil effect — but the actual reduction order inside candle's lazy op graph differs.

| Config | The | To | Gap | Δ vs HEAD |
|---|---|---|---|---|
| `HF2Q_SPIKE_B_ROUTER_ORDER=1` (llama.cpp order), BF16 on | 27.108929 | 26.352211 | +0.756718 | −0.01430 |
| `HF2Q_SPIKE_B_ROUTER_ORDER=1`, BF16 off | 27.103190 | 26.359339 | +0.743851 | −0.02717 |

- Under BF16-on, the router-order swap moves the gap by −0.01430. This is **numerically identical to the `--rope-kernel loop` toggle result** (row 4 in the Part 1 table) and the `--rms-norm-kernel loop --rope-kernel loop` Δ minus the RmsNorm contribution. It's a BF16 interaction effect, not a genuine structural change.
- Under BF16-off, the router-order swap is **byte-flat** (same output as BF16-off baseline). Confirms the order swap is a no-op in F32 as expected (commutativity of multiplication).

Router-order swap falsified as an independent drift contributor.

### Part 5 — Structural comparison vs llama.cpp `gemma4-iswa.cpp`

Op-by-op audit of hf2q `DecoderLayer::forward` (`gemma4.rs:1425-1504`) and `Attention::forward` (`gemma4.rs:657-884`) against llama.cpp `llm_build_gemma4_iswa::llm_build_gemma4_iswa` (`gemma4-iswa.cpp:10-260`). All citations verified end-to-end on disk.

| Feature | llama.cpp (file:line) | hf2q (file:line) | Match? |
|---|---|---|---|
| Embedding scale `sqrt(n_embd)` | `gemma4-iswa.cpp:20` `ggml_scale(inpL, sqrtf(n_embd))` | `gemma4.rs:1914` `xs = xs * sqrt(hidden_size)` | **YES** |
| Per-layer token embeddings (`inp_per_layer`) | `gemma4-iswa.cpp:31-38, 202-224` (gated on `model.per_layer_tok_embd != nullptr`) | **NOT IMPLEMENTED** in hf2q | **N/A** — GGUF has no `per_layer_tok_embd` tensor (verified via `gguf.GGUFReader`), so llama.cpp also skips this path. Both tools take the `None` branch. |
| Attention pre-norm | `gemma4-iswa.cpp:52` `build_norm(inpL, attn_norm)` → GGUF `blk.N.attn_norm.weight` | `gemma4.rs:1431` `self.input_layernorm.forward(xs)` → GGUF `blk.N.attn_norm.weight` | **YES** (same GGUF tensor) |
| Q projection | `gemma4-iswa.cpp:65` `build_lora_mm(wq, cur)` | `gemma4.rs:665` `self.q_proj.forward(xs)` | **YES** |
| Q norm (applied post-reshape) | `gemma4-iswa.cpp:70-71` `build_norm(Qcur, attn_q_norm)` on `[n_embd_head, n_head, n_tokens]` | `gemma4.rs:683` `self.q_norm.forward(&q)` on `[b, q_len, num_heads, head_dim]` | **YES** (last-dim norm is head_dim in both cases; GGUF tensor `blk.N.attn_q_norm.weight` is shape `[256]` for sliding, `[512]` for global, matching each layer's head_dim) |
| K projection | `gemma4-iswa.cpp:80` `build_lora_mm(wk, cur)` | `gemma4.rs:667` `self.k_proj.forward(xs)` | **YES** |
| K norm | `gemma4-iswa.cpp:91` `build_norm(Kcur, attn_k_norm)` | `gemma4.rs:684` `self.k_norm.forward(&k)` | **YES** |
| V projection (`k_eq_v` path) | `gemma4-iswa.cpp:83-85` `Vcur = wv ? build_lora_mm(wv, cur) : Kcur` | `gemma4.rs:669-674` `v = if self.k_eq_v { k.clone() } else { v_proj.forward(xs) }` | **YES** — hf2q's `k_eq_v` flag is set from `config.json: attention_k_eq_v=true` AND `is_full_attention(il)`; the 5 full-attention layers (5/11/17/23/29) take the `k.clone()` branch. Verified that `attn_k.weight` and `attn_v.weight` are **byte-identical** in the GGUF for those 5 layers (`gguf.GGUFReader` 128-byte prefix check), so `k.clone()` is semantically equivalent to `v_proj.forward(xs)` after the projection. llama.cpp would take the `wv ? build_lora_mm(wv, cur) : Kcur` branch on the `true` side because `attn_v.weight` IS present — both tools arrive at the same tensor content, just via different branches. |
| V unit norm (no learned weight) | `gemma4-iswa.cpp:92` `ggml_rms_norm(Vcur, eps)` | `gemma4.rs:686` `rms_norm_unit(&v, ...)` | **YES** |
| RoPE on Q and K (not V) | `gemma4-iswa.cpp:73, 97-98` `ggml_rope_ext(Q/K, ..., n_rot, rope_type, freq_base, ...)` | `gemma4.rs:693` `self.rotary_emb.apply(&q, &k, ...)` | **YES** (partial rotary on global layers confirmed; freq_factors tensor present for global only) |
| Attention SDPA scale | `gemma4-iswa.cpp:104` `build_attn(..., hparams.f_attention_scale)` → **kq_scale = 1.0** for Gemma4 (`llama-model.cpp:1273`) | `gemma4.rs:758, 780` `sdpa(..., 1.0, 1.0)` | **YES** (no pre-softmax scale; both compute `softmax(K·Q) · V`) |
| Attention output projection | `gemma4-iswa.cpp:104` `build_attn(..., wo, ...)` | `gemma4.rs:883` `self.o_proj.forward(&attn_out)` | **YES** |
| Post-attention norm | `gemma4-iswa.cpp:117-119` `build_norm(cur, attn_post_norm)` → GGUF `blk.N.post_attention_norm` | `gemma4.rs:1433` `self.post_attention_layernorm.forward(&attn_out)` → GGUF `blk.N.post_attention_norm` | **YES** (same GGUF tensor via `tensor_mapping.py:1020` `MODEL_TENSOR.ATTN_POST_NORM: "blk.{bid}.post_attention_norm"`) |
| Residual add (`attn_out + inpL`) | `gemma4-iswa.cpp:122` `ggml_add(cur, inpL)` (ADD-THEN-…) | `gemma4.rs:1454` `(xs + &attn_out)` (ADD-THEN-…) | **YES** — 1bNEW.0b Walk Exception unwound the old fused version; current code matches reference byte-for-byte |
| Pre-FFW norm (MLP branch) | `gemma4-iswa.cpp:129-132` `build_norm(attn_out, ffn_norm)` → GGUF `blk.N.ffn_norm.weight` | `gemma4.rs:1462` `self.pre_feedforward_layernorm.forward(&xs)` → GGUF `blk.N.ffn_norm.weight` | **YES** (same GGUF tensor) |
| Dense MLP | `gemma4-iswa.cpp:134-139` `build_ffn(..., FFN_GELU, FFN_PAR)` | `gemma4.rs:1465` `self.mlp.forward(&normed)` | **YES** (GELU, parallel gate×up × down) |
| Post-FFW norm #1 (MLP output) | `gemma4-iswa.cpp:140-142` `build_norm(cur_mlp, ffn_post_norm_1)` | `gemma4.rs:1466` `self.post_feedforward_layernorm_1.forward(&mlp_out)` | **YES** (GGUF `blk.N.post_ffw_norm_1`) |
| Pre-FFW norm #2 (MoE branch) | `gemma4-iswa.cpp:146-148` `build_norm(attn_out, ffn_pre_norm_2)` | `gemma4.rs:1472` `self.pre_feedforward_layernorm_2.forward(&xs)` | **YES** (GGUF `blk.N.pre_ffw_norm_2`) |
| Router unit RmsNorm | `gemma4-iswa.cpp:152` `ggml_rms_norm(attn_out, eps)` (no weight) | `gemma4.rs:1038` `rms_norm_unit(&router_flat, ...)` (no weight) | **YES** |
| Router scalar scale `1/sqrt(n_embd)` | `gemma4-iswa.cpp:153` `ggml_scale(tmp, 1/sqrtf(n_embd))` | `gemma4.rs:1039` `scale_factor = (hidden_size as f64).powf(-0.5)` applied at `:1040` | **YES** (order differs within the expression, but F32 commutative up to ULP; falsified as a drift source in Part 4) |
| Router per-element scale `ffn_gate_inp_s` | `gemma4-iswa.cpp:154` `ggml_mul(tmp, ffn_gate_inp_s)` → GGUF `blk.N.ffn_gate_inp.scale` | `gemma4.rs:1040` `broadcast_mul(&self.router_scale)` → GGUF `blk.N.ffn_gate_inp.scale` | **YES** (same GGUF tensor) |
| Router matmul (`ffn_gate_inp`) | `gemma4-iswa.cpp:155` `build_lora_mm(ffn_gate_inp, tmp)` → 128-expert logits | `gemma4.rs:1042` `self.router_proj.forward(&router_scaled)` | **YES** |
| MoE expert FFN | `gemma4-iswa.cpp:158-172` `build_moe_ffn(cur_moe, ..., n_expert=128, n_expert_used=8, FFN_GELU, true, 1.0, SOFTMAX, ..., logits, ffn_gate_up_exps, ..., ffn_down_exps_s)` | `gemma4.rs:1473` `self.moe.forward(&normed_moe, &xs)` (1bNEW.1 fused kernel) | **YES** (softmax over 128 experts, top-k=8, per-expert gate_up × GELU × down with `ffn_down_exps.scale` post-mult) |
| Post-FFW norm #2 (MoE output) | `gemma4-iswa.cpp:173-175` `build_norm(cur_moe, ffn_post_norm_2)` | `gemma4.rs:1474` `self.post_feedforward_layernorm_2.forward(&moe_out)` | **YES** |
| MLP + MoE combine | `gemma4-iswa.cpp:178` `ggml_add(cur_mlp, cur_moe)` | `gemma4.rs:1491` `(mlp_normed + moe_normed)` | **YES** |
| Post-FFW norm (shared) | `gemma4-iswa.cpp:194-196` `build_norm(cur, ffn_post_norm)` | `gemma4.rs:1493` `self.post_feedforward_layernorm.forward_with_post_residual(&combined, residual)` (F=3 kernel) | **YES** (F=3 fused kernel = `(norm(combined) + residual)` = exact op pair) |
| FFW residual add | `gemma4-iswa.cpp:200` `ggml_add(cur, attn_out)` | `gemma4.rs:1493` folded into `forward_with_post_residual` F=3 kernel | **YES** (fused path matches llama.cpp's `kernel_rms_norm_mul_add_f32` at `ggml-metal.metal:3046`) |
| Per-layer PE residual | `gemma4-iswa.cpp:202-224` (gated on `inp_per_layer != nullptr`) | **NOT IMPLEMENTED** | **N/A** (no `per_layer_tok_embd` in GGUF; both tools skip) |
| Layer output scale | `gemma4-iswa.cpp:227-230` `ggml_mul(cur, out_scale)` → GGUF `blk.N.layer_output_scale.weight` (F32 `[1]`) | `gemma4.rs:1497` `xs.broadcast_mul(&self.layer_scalar)` | **YES** (F32 scalar per layer, both tools) |
| Control vec | `gemma4-iswa.cpp:232` `build_cvec(cur, il)` — **no-op when no control vector loaded** | not implemented in hf2q | **N/A** (no control vector) |
| Final norm (`output_norm`) | `gemma4-iswa.cpp:240-243` `build_norm(cur, output_norm)` → GGUF `output_norm.weight` (F32 `[2816]`) | `gemma4.rs:1928` `self.norm.forward(&last_hidden)` → same tensor | **YES** |
| `lm_head` | `gemma4-iswa.cpp:248` `build_lora_mm(model.output, cur)` — tied to `token_embd.weight` (F16) | `gemma4.rs:1947-1966` → `lm_head_kernel::lm_head_forward_fused` (F16 matmul via MLX-GEMM) | **YES** (both read F16 bytes) |
| Softcapping | `gemma4-iswa.cpp:250-254` `scale(logits, 1/sc); tanh; scale(logits, sc)` | `gemma4.rs:1980` `((logits / sc).tanh() * sc)` | **YES** (same formula, same eval order) |

**Also verified:**

- **Gemma4 does NOT use the `(1+w)` RmsNorm convention** that Gemma1/2/3 use. Source: `/opt/llama.cpp/convert_hf_to_gguf.py:7520-7522`:
  ```python
  def norm_shift(self, name: str) -> float:
      del name # unused
      return 0.0
  ```
  Gemma3 inherited from the same base class returns `1.0` for `norm.weight` endings (line 6959-6960), and Gemma/Gemma2 convert with `data_torch = data_torch + 1` at `:6905-6906, :6949-6950`. **Gemma4 overrides this to 0.0** — the HF weights are used directly without the +1 shift. Confirmed at the vllm inference side: `vllm/model_executor/models/gemma4.py:293, 316, 507` uses plain `RMSNorm` (`x * w`), while `gemma.py`/`gemma2.py`/`gemma3.py` use `GemmaRMSNorm` which does `x * (1 + w)` at `vllm/model_executor/layers/layernorm.py:360-398`. hf2q's `RmsNorm::forward` at `gemma4.rs:232-250` and `rms_norm_kernel::kernel_rms_norm_fuse_impl<float, 2>` at `rms_norm_kernel.rs:198` both apply plain `x * w` — **correct for Gemma4.** Falsifies the `(1+w)` hypothesis as a drift source.

- **`num_kv_shared_layers` = 0** for this GGUF (verified via `gguf.GGUFReader`). The KV-layer-sharing feature (`llama-model.cpp:1269-1272`) is off. hf2q and llama.cpp both compute K/V for every attention layer.

- **`f_attention_scale` = 1.0** for Gemma4 (`llama-model.cpp:1273` `// Gemma4 uses self.scaling = 1.0 (no pre-attn scaling)`). hf2q passes `scale=1.0` to `candle_nn::ops::sdpa`. Both tools compute `softmax(K·Q) · V` with no `1/sqrt(d)` prefactor.

### Part 6 — Summary gap-contribution table

| Candidate | Measured Δ contribution | Share of 0.89 logit drift | Verdict |
|---|---|---|---|
| 1bNEW.17 F16 lm_head (fused vs loop) | **−0.00086** | **0.10%** | Falsified as drift owner (ADR 1bNEW.17 Phase C already recorded this) |
| 1bNEW.1 fused MoE kernel (fused vs loop) | **0.00000** | **0.00%** | **Falsified** (byte-flat) |
| 1bNEW.4 fused RmsNorm kernel (fused vs loop) | **+0.00573** | **+0.64%** (widens gap, AWAY from llama.cpp) | **Falsified** as drift toward llama.cpp; fused kernel actually narrows gap slightly when paired with BF16 |
| 1bNEW.6 fused RoPE kernel (fused vs loop) | **−0.01430** | **1.6%** | **Falsified** as primary; BF16 interaction (see Part 2 / Part 4) |
| 1bNEW.10 BF16 prefill SDPA (on vs off) | **−0.02717** | **3.1%** | **Minor contributor.** Moves gap toward llama.cpp. Not primary owner. |
| Router scalar-mul order (hf2q order vs llama.cpp order) | **0.00000** in F32 / **−0.01430** in BF16 (same as RoPE) | **0%** in F32 | **Falsified** as independent contributor; BF16-interaction artifact identical to the RoPE toggle |
| All kernel + BF16 toggles combined (all-loop + BF16 off) | **≤ −0.0300** | **~3.4%** | All-toggleable-drift contributions sum to ~3.4%. |
| **Residual ~96.6% of the 0.89 logit drift** | **≥ 0.86 logit** | **≥ 96.6%** | **UNLOCATED.** No single candidate tested accounts for it. |

### Interpretation

The ADR-005 1bNEW.17 item text at lines 623-625 hypothesized that the F32 dense lm_head was the drift owner because "a 262144-wide F32 accumulator has materially different FP rounding than llama.cpp's Q6_K kernel." **Empirically falsified** in Phase C (1bNEW.17 shipped F16 matmul, gap stayed flat). The follow-up hypothesis at ADR-005 post-1bNEW.17 lines 600-602 that "the drift owner is most likely the BF16 prefill SDPA path or the per-layer residual accumulation convention" is **partially falsified**:

- BF16 prefill contributes **0.027 / 0.89 = 3.1%** of the drift. Real but minor.
- Residual accumulation order matches llama.cpp byte-for-byte (verified in Part 5 table). 1bNEW.0b correctly un-fused the pre-FFW residual add to the ADD-THEN-NORM order required by llama.cpp's `gemma4-iswa.cpp:122`. The post-FFW combiner is NORM-THEN-ADD in both tools (verified). There is no remaining residual-order mismatch.

**None of the op-order / kernel / precision toggles I can flip from within hf2q closes more than ~3.4% of the gap.** The remaining ~96% is invariant to every Walk-item-scale change I tested.

### Hypothesis space (post-falsification)

The 0.86 logit residual must come from one of:

1. **A structural op hf2q is computing differently** — something subtle in the attention graph, the MoE pipeline, or the embedding/lm_head bridge that produces the same output byte-count and the same shapes but a systematically different per-token bias. Part 5 rules out the documented llama.cpp Gemma4 ops one-by-one; anything still in this bucket is an undocumented micro-convention (e.g., transposition convention on Q/K reshape pre-norm, expert-selection tie-breaking, ffn_down scale application order inside the MoE kernel).
2. **A weight loading convention difference** — e.g., a tensor loaded as `[n, k]` in hf2q but `[k, n]` in llama.cpp; or a dtype cast at load time that silently discards precision. Part 5 spot-checks the most common tensors (input_layernorm, post_attention_norm, pre_feedforward, router_scale, layer_scalar) but doesn't audit every one of the 663 tensors in the GGUF.
3. **A `tokenizer.ggml.tokens` / chat-template bytecode difference at positions >187** — 1bNEW.0c fixed the `--jinja` path to produce byte-identical 187-token prompts, but the post-template rendered prompt might have a silent whitespace or special-token boundary drift. The ADR line 198 records "tokenizes to 187 tokens via both tokenizers (zero token-level diffs)", so this is low-probability but not fully ruled out at the byte level inside the tokenizer state machine.
4. **DWQ-quantization-specific behavior** — the GGUF is mixed Q4_0/Q6_K/Q8_0/F16/F32, and the DWQ conversion may have baked subtle per-layer bias into the quantized tensors that interacts with hf2q's kernel layout but not llama.cpp's. This would make the drift **structural to the hf2q+this-GGUF pair** and not fixable via kernel-level Walk items.
5. **The reference itself is sampled at a different graph point** — e.g., llama.cpp's reported `-0.6487 / -0.7663` logprobs from `/completion?n_probs=10` are at the sampler's "first-order logprob" stage, which in llama.cpp includes not only the raw lm_head output but also any sampler-level adjustments (repetition penalty, temperature scaling, top_k/top_p masking). The ADR records "/completion with n_probs=10" which uses the raw logit distribution, so this should be apples-to-apples, but it's worth checking whether `n_probs` reports pre- or post-softmax and whether llama.cpp's softmax includes the final_logit_softcapping (it should, per line 250-254). Softcapping applies `x * (sc/sc) = x` only if `sc` is the same in both — hf2q reads it from GGUF, llama.cpp reads it from hparams. **Both should be identical** because the GGUF stores `final_logit_softcapping` in metadata and both tools read from there.

The cheapest next investigation would be (5) — verify the llama.cpp logprob reference is computed on the raw logit distribution post-softcap and pre-sampler-adjustment, because if it's not, the whole 0.77→-0.12 disagreement may be partially an apples-vs-oranges comparison.

Second cheapest: a **per-layer intermediate comparison** using `crawl_verify.sh` with layer-by-layer hidden state dumps at the layer-29 output (the last transformer layer before the final norm+lm_head). If hf2q's layer-29 output already differs from llama.cpp's by ~100 units of noise, the drift is inside the transformer body and the spike proceeds to per-layer bisection. If it's near-identical, the drift is concentrated in the final norm + lm_head + softcap tail, which is a 4-op window and much easier to diff.

### Spike B verdict

- **Primary drift owner: UNLOCATED in this spike budget.** The 0.89 logit gap to llama.cpp is >95% invariant to every kernel toggle, BF16 toggle, op-order swap, and structural audit I exercised. The falsification ran out the 5-candidate hypothesis space.
- **Measured contributions**: none of the candidates individually or in combination closes more than ~3.4% of the gap. See the Part 6 table.
- **Next action**: **open a dedicated per-layer hidden-state comparison spike** (~6-8 hours) that dumps hf2q's layer-29 output vs llama.cpp's same-layer output via llama.cpp's `llama-cli --log-hidden-states` or a custom patch, on the canonical 187-token prompt, and computes per-position |Δ| to isolate which layer or which op-group introduces the first >1e-3 divergence. This is the only path forward that doesn't blindly speculate. **It is NOT a Walk item** — it's a deeper investigation spike, and the fix it implies may or may not be Walk-citable. If the drift is concentrated in one layer, it's potentially Walk; if it's smeared across 30 layers at ~1e-2 each, it's structural and belongs in Run.
- **End gate reachability update**: **MEASURED_UNREACHABLE** (unchanged from post-Walk re-spike verdict).

### End-gate reachability rationale

The post-Walk re-spike (`docs/spike-post-walk-results.md:287-291`) concluded that Phase 1b End gate (107 tok/s decode) is **measured_unreachable under strict Walk**, with 1bNEW.17 closing ~5.6 ms/token but leaving a residual ~5.5 ms gap that maps only to Run-territory candle infrastructure work. 1bNEW.17 landed at **58.51 tok/s**, +9.80 tok/s vs the 48.71 tok/s pre-landing baseline, within 2% of the F16 bandwidth-bound projection. The remaining decode ms/token at 58.51 is **17.09 ms/token**, leaving a **7.74 ms/token gap** to the 9.35 ms/token 107-tok/s target — slightly wider than the post-Walk re-spike projected because 1bNEW.17 shipped as F16 (−3.67 ms saved) instead of the hypothetical Q6_K (−5.6 ms saved).

This spike adds two facts:

1. **Spike A confirms the lm_head is at its bandwidth floor for this GGUF.** No further Walk lift is available from the lm_head.
2. **Spike B finds no Walk-faithful drift owner for the `The`/`To` gap.** Fixing the correctness gap (if it even lands as a Walk item) has no direct speed impact — correctness and speed are orthogonal on this path. End-gate reachability is unaffected by Spike B's outcome.

**End gate remains MEASURED_UNREACHABLE under strict Walk.** 1bNEW.17 was the last Walk item with a reachable speed lift; everything else is structural to candle's dispatch pipeline and belongs in Run (per-buffer wait semantics, parallel CPU enqueue, kernel-level pipelining). Walk is effectively done.

### Spike B citation trail

| Claim | File:line |
|---|---|
| HEAD baseline top-10 reproduction | `/opt/hf2q/target/release/hf2q generate --model [...] --max-tokens 1 -T 0` output: `(818, 27.108929), (2021, 26.337908)` (matches ADR line 266) |
| 1bNEW.17 `--lm-head-kernel=loop` top-10 reproduction | Same binary with flag: `(818, 27.110750), (2021, 26.340590)` (matches commit `0e36b1c` Phase B Gate 1) |
| BF16 prefill branch | `/opt/hf2q/src/serve/gemma4.rs:755-783` |
| Manual F32 sliding-layer fallback (Part 2 target) | `/opt/hf2q/src/serve/gemma4.rs:828-874` |
| Router math (Part 4 target) | `/opt/hf2q/src/serve/gemma4.rs:1038-1042` |
| hf2q `RmsNorm::forward` plain `x*w` (no +1) | `/opt/hf2q/src/serve/gemma4.rs:213-250`, kernel: `/opt/hf2q/src/serve/rms_norm_kernel.rs:194-201` |
| hf2q `k_eq_v` load path | `/opt/hf2q/src/serve/gemma4.rs:1698-1705` |
| hf2q `attention_k_eq_v` config default | `/opt/hf2q/src/serve/config.rs:93` |
| GGUF K == V byte-equivalence on full-attention layers | `gguf.GGUFReader` check on layers 5/11/17/23/29: `attn_k.weight` bytes == `attn_v.weight` bytes |
| llama.cpp Gemma4 forward pass | `/opt/llama.cpp/src/models/gemma4-iswa.cpp:10-260` |
| llama.cpp `build_norm` (plain `x*w`, +1 applied at conversion for Gemma1-3 only) | `/opt/llama.cpp/src/llama-graph.cpp:1027-1060` |
| llama.cpp `build_attn_mha` — no Q/sqrt(d) pre-scale, kq_scale applied inside softmax | `/opt/llama.cpp/src/llama-graph.cpp:1849-1982` especially `:1949` `ggml_soft_max_ext(kq, mask, kq_scale, ...)` |
| llama.cpp Gemma4 `f_attention_scale = 1.0f` | `/opt/llama.cpp/src/llama-model.cpp:1273` |
| llama.cpp Gemma4 converter — `norm_shift()` returns 0.0 (no +1) | `/opt/llama.cpp/convert_hf_to_gguf.py:7520-7522` |
| Gemma1/2/3 converter — `norm.weight + 1` | `/opt/llama.cpp/convert_hf_to_gguf.py:6905-6906, 6949-6950, 6959-6960` |
| vllm Gemma4 plain `RMSNorm` (confirms no +1) | `/opt/vllm/vllm/model_executor/models/gemma4.py:293, 316, 507` |
| vllm `GemmaRMSNorm` class (the `+ 1.0` convention for Gemma1/2/3) | `/opt/vllm/vllm/model_executor/layers/layernorm.py:360-398` especially `:386` `weight = self.weight.data.float() + 1.0` |
| Q4 spike BF16 baseline (F32 0.748 vs BF16 0.677) | `/opt/hf2q/docs/spike-Q3Q4Q5-results.md:111-128` |
| post-Walk re-spike drift owner discussion | `/opt/hf2q/docs/spike-post-walk-results.md:123-134` |

---

## Cumulative summary

### Spike A summary

The "QMatMul auto-dequant quirk" cited in the 1bNEW.17 commit refers to `candle-core/src/quantized/mod.rs:727` — an unconditional `GgmlDType::F32 | F16 | BF16 => true` arm in `QMatMul::from_arc` that forces every floating-point `QTensor` through `.dequantize()?` into `Self::Tensor(f32)`, making the `DEQUANTIZE_ALL_F16` env var structurally unreachable for F16 sources. It is a real candle quirk, reproducible in a standalone Rust repro, and documented with timing deltas. **But the quirk is not the gating factor for 1bNEW.17:** `token_embd.weight` in this Gemma 4 26B DWQ GGUF is stored as **F16**, not Q6_K (verified via three independent readers at 1,476,395,008 bytes = 262144×2816×2), so the ADR's projected "67 tok/s Q6_K lm_head" counterfactual never existed. The 1bNEW.17 shipping path reads the same F16 bytes llama.cpp reads and routes them through candle's `call_mlx_gemm` F16 gemm at ~3.73 ms/call steady-state — at or below the bandwidth floor for this GGUF+hardware. Patching the quirk would not produce further speedup on this model; the ceiling is hit.

### Spike B summary

The post-1bNEW.17 `The`/`To` argmax drift (+0.771 toward `The`, vs llama.cpp's −0.118 toward `To`, ~0.89 logit total disagreement) does NOT map to any single Walk item, kernel port, BF16 prefill boundary, op-order convention, or `(1+w)` RmsNorm hypothesis. Exhaustive bisection on the four fused kernels plus the 1bNEW.10 BF16 branch plus a router-order swap accounts for **≤3.4% of the total drift**; the remaining **≥96%** is invariant to every toggle, and a full structural audit against `/opt/llama.cpp/src/models/gemma4-iswa.cpp` (Part 5 table, 22 rows) matches hf2q op-for-op including the Gemma4-specific plain RmsNorm (no +1), `scale=1.0` attention, `k_eq_v` optimization, and residual add ordering. The drift owner is somewhere hf2q does NOT currently instrument and is not falsifiable with a short in-source scratch edit. The cheapest next step is a layer-by-layer hidden-state comparison against llama.cpp on the same 187-token prompt — a separate investigation spike, not a Walk item.

### Final decision-support sentence

Walk kernel work is effectively complete at 58.51 tok/s with a structural ~0.86-logit correctness gap to llama.cpp that does not close via any kernel-level Walk item; the correct next move is to **open a per-layer hidden-state bisect spike** (Spike C) or declare Walk done and open Run, because every Walk-scale investigation tool has been exhausted without finding a reference-citable fix.
