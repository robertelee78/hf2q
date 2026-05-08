# ADR-020: DWQ + Mixed-Precision Quantization for hf2q (port from mlx-lm)

**Status**: Proposed — last revised 2026-05-06 (iter 5.5 — structural cleanup)

**Driver**: User mission — make working DWQ quants (46 + 48) for the Gemma 4 and Qwen 3.6 abliterated model families on a 128 GB Mac M5 Max without OOM, with NO quality compromise ("no cheating, real DWQ"). Both DWQ proper AND mixed-precision must be supported. Per `~/Documents/mantra.txt`: "Measure 3x, cut once. No fallback. No stub. Pure excellence."

**Predecessors**: ADR-014 P11 closure (DwqKQuantizer + cache); memory `project_qwen35_dwq_pre_505b5b8_broken_2026_05_05.md`; memory `project_hf2q_dwq_oom_root_cause_2026_05_06.md`.

---

## 1. Why (problem)

### 1.1 The OOM

DWQ-46/48 conversion of Qwen3.6-27B (52 GB F16 source) on a 128 GB box **kernel-panicked twice** on 2026-05-05/06 (samples=1024, samples=256, with and without `HF2Q_STREAMING_PHASE3_MUT=1`). Observed peak: **199 GB**.

Comparable workloads succeed elsewhere:

| Stack | Peak | Source observation |
|---|---|---|
| `llama.cpp llama-quant.cpp:1109-1245` | ~5–10 GB | per-tensor stream: load → dequant → quantize-in-chunks → write → reuse buffer |
| Apple MLX (`oq.py` + `dwq.py`) | ~60–80 GB | lazy `mx.lazy` arrays via UMA, 8-bit teacher distillation, `del grads + mx.eval` rhythm |
| **hf2q (current)** | **~200 GB** | jetsam → kernel panic, unreliable |

### 1.2 OOM root cause (verified by reading code)

Per mantra "comments are starting points; trust code". The 199 GB peak decomposes:

- **Driver A — `clone_tensor_map_to_lazy` deep clone (~52 GB)**: `src/main.rs:482-496` calls `tensor.data.clone()` on `Vec<u8>` (full deep copy per tensor). `src/ir/lazy.rs:215-218` notes "the full P13 win lands when `TensorRef::data` itself becomes `Arc<[u8]>`" — migration was deferred. ✅ **Closed by Phase 1 commit `437217d` (TensorRef.data → Arc<Vec<u8>>).**
- **Driver B — `Qwen35Model` F32 host expansion held all-at-once (~104 GB)**: `weight_loader.rs:710-741` accumulates `Qwen35LayerWeights` for ALL layers via `load_lazy_f32` (F16→F32 doubles). Doc-comment claims "drop before next" but code retains layers in `Qwen35Model.layers: Vec<...>`. Still open.
- **Driver C — `materialize_cloned` deep clones the Arc'd Vec (~52 GB)**: `src/ir/lazy.rs:314-345` does `(**arc).clone()` instead of `Arc::clone(arc)` — deep-copies the buffer despite Phase 1's Arc shell. Open. (Iter 6 target.)

Sum: 52 + 52 + 104 + scratch ≈ 200 GB ✓ matches observation.

**Why 2026-04-30 succeeded on the same box**: cache HITs short-circuited Phase 2 entirely. The `SensitivityCacheKey::with_algorithm_version` + `model_fingerprint` derivation has since drifted; existing `~/.cache/hf2q/sensitivity/*.json` files no longer match. Cold MISS → 200 GB → jetsam.

### 1.3 Taxonomy correction

**hf2q's "DWQ-46" / "DWQ-48" are NOT what mlx-lm calls DWQ.** Reading `/opt/mlx-lm/mlx_lm/quant/dwq.py` and `/opt/mlx-lm/mlx_lm/quant/dynamic_quant.py` clarifies:

| What hf2q called | What it actually is | mlx-lm canonical | Output format | Status in hf2q |
|---|---|---|---|---|
| "DWQ-46" | Sensitivity ranking → mixed Q4/Q6 GGUF | `dynamic_quant.py` | MLX safetensors w/ per-tensor `{bits, group_size}` overrides | **Misnamed**; algorithm differs (variance-magnitude vs gradient-Taylor) |
| "DWQ-48" | Same with sensitive=Q8_0 | `dynamic_quant.py` (4/8 split) | Same | Same |
| (not implemented) | Distillation fine-tuning of quantized scales+biases | `dwq.py` | MLX safetensors w/ trained fp16 scales+biases | NEW capability — port required |
| (not implemented) | Activation-aware weight scaling | `awq.py` | MLX safetensors w/ scaled+clipped weights | Optional 3rd track |
| (not implemented) | Cholesky weight least-squares | `gptq.py` | MLX safetensors (bit-identical to standard quantized) | Optional 4th track |

### 1.4 Goals

1. Replace hf2q's misnamed `DWQ-46/48` with a correct port of mlx-lm's `dynamic_quant` (Track 1).
2. Add NEW DWQ-proper capability: distillation fine-tuning of quantized scales+biases (Track 2).
3. Both must run on a 128 GB Mac with samples=1024 default — no quality compromise.
4. Output of Track 2 ships as MLX safetensors (verified incompatible with GGUF Q4_K — see §5).
5. hf2q runtime must serve DWQ-trained MLX safetensors natively (mlx-native already has 80% of the kernels — see §6).

---

## 2. What (proposed solution)

Five-step plan, ordered by independence + value:

### Step 0 (DONE): Phase 1 — `TensorRef.data: Arc<Vec<u8>>` refactor — commit `437217d`

Driver A closed. ~52 GB saved on cache MISS.

### Step 1 (NEXT, iter 6): `materialize_cloned` Arc-share fix — ~20 LOC

Driver C closed. Another ~52 GB saved. Independent of all algorithm work. See AC §8.1.

### Step 2 (iters 7–10): Track 1 — port mlx-lm `dynamic_quant`

Replaces hf2q's misnamed `DWQ-46/48`. Native Rust port; no autograd dep; output stays GGUF (current format). See AC §8.2.

### Step 3 (iters 13–17): Track 2 — native DWQ port

Direct native port of `dwq_quantize` (mlx-lm `dwq.py:69-209`) reusing
the iter-8 GPU autograd toolchain: KL-div loss, log_softmax,
log/softmax/sub/mul/row_sum primitives, Adam optimizer over the
trainable scales+biases of QDQ'd weights, stream-to-disk targets,
new mlx-native prefill `quantized_matmul_mm_affine.metal` kernel.
No subprocess gates, no Python intermediaries — see AC §8.3 + §11
"Architectural principle: no external tools".

### Step 5 (iters 19+, optional): Tracks 3+4 — AWQ, GPTQ

Conditional on demand. See AC §8.5–8.6.

---

## 3. Research findings (iter 5 — 10-researcher /cfa deep-dive)

Full deliverables saved at `docs/research/cfa-mlxlm-deep-dive-2026-05-06/` (~30K words). Headlines:

| # | Topic | Headline finding |
|---|---|---|
| 1 | DWQ algorithm | KL-div distillation; optimizes scales+biases ONLY (codes frozen → no STE needed); 411 LOC; lr=1e-6, batch=4, samples=2048, max_seq=1025 |
| 2 | Dynamic Quant | Signed first-order Taylor: `(grad·(w_low−w_high)).sum()/params_M` ≠ hf2q's variance-magnitude proxy. Cache invalidation forced. |
| 3 | AWQ | 585 LOC scale-then-clip; activation-aware; needs `Catcher` per-Linear hook (~1-2 weeks port). Recommended AFTER dynamic_quant. |
| 4 | GPTQ | 229 LOC Cholesky-based least-squares; smallest of 4 algos; output bit-identical to standard MLX quantized layer. Port estimate 1-2 weeks. |
| 5 | Shared infra | `load(lazy=True)` + sharded safetensors at 5 GB/shard + `donate_model=True` + `iterate_batches(pad_to=32)` + `kl_div_loss` Metal kernel (200 LOC, custom VJP) |
| 6 | Memory tricks | **3 load-bearing**: per-batch `del grads + mx.eval(grad_accum)`, stream-targets-to-disk in `compute_dwq_targets`, fix hf2q's `materialize_cloned` deep-clone |
| 7 | File format | MLX safetensors (uint32 packed weight + fp16 scales + fp16 biases per group_size=64) ≠ GGUF Q4_K (super-block 256 weights, 6-bit compressed scales). DWQ→GGUF is **LOSSY**. |
| 8 | Port checklist | Phase 1: dynamic_quant (~80 LOC core, hand-derived gradients, no autograd needed). Phase 2: DWQ via subprocess wrapper or native port. |
| 9 | DWQ 2026 SOTA | DWQ has NO paper; canonical = Awni Hannun blog + GitHub. Best 4-bit MoE: 0.02663 KL on Qwen3.6-35B-A3B-4bit-DWQ (smcleod 2026-04). DWQ saturates at 6-bit. **DWQ remains SOTA for Apple Silicon W4A16 as of 2026.** FlatQuant/SpinQuant lead for NVIDIA W4A4 (different stack). |
| 10 | Runtime support | mlx-native already has affine 4/6/8-bit `quantized_matmul` kernels (decode + MoE-routed expert). Loader at `weight.rs:626`. Only missing: prefill qmm tile kernel (~3-5 days port from `quantized_matmul_mm.metal`). **hf2q can serve DWQ natively.** |

---

## 4. Memory architecture: 3 load-bearing patterns to port (researcher #6)

### 4.1 Per-batch `del grads + mx.eval(grad_accum)` rhythm

**Source:** `dynamic_quant.py:80-86` + `dwq.py:178-179`. **Saves:** ~60 GB/step on 27B-class.

hf2q port pattern (in new `estimate_sensitivities` and `dwq_quantize`):
```rust
gradient_accumulator += &batch_grads;
drop(batch_grads);              // explicit drop — no accumulation
device.commit_and_wait()?;      // mlx-native equivalent of mx.eval
```

### 4.2 Stream-targets-to-disk

**Source:** `dwq.py:29-66 compute_dwq_targets` + line 386-387 `del model`. **Saves:** ~60 GB on 27B (lets teacher be dropped before student trains).

hf2q port:
1. `--quant dwq-4 --target-dir /path` precomputes teacher top-1024 logits + indices to `.safetensors` files.
2. Phase 1: teacher loaded, no student.
3. `drop(teacher)` between phases.
4. Phase 2: student trains, teacher gone, per-batch targets read from disk.

### 4.3 `materialize_cloned` Arc-share fix (iter 6)

**Source:** `/opt/hf2q/src/ir/lazy.rs:314-345`. **Bug:** Phase 1 made `TensorRef.data: Arc<Vec<u8>>`, but `materialize_cloned` still does `(**arc).clone()` — deep-copies the Vec inside the Arc. **Fix:** ~20 LOC; replace with `Arc::clone(arc)`. **Saves:** ~52 GB on 27B. **Independent of algorithm work — land first.**

---

## 5. File format reality (researcher #7)

DWQ-trained scales **cannot** survive a GGUF round-trip without quality loss:

| Aspect | MLX format | GGUF Q4_K |
|---|---|---|
| Group size | 64 weights | 32 weights × 8 sub-blocks = 256 super-block |
| Scale precision | fp16 per group | 6-bit compressed per super-block |
| Bias | fp16 per group | implicit min per super-block |
| Trained scales survive? | YES | **NO** — snap fp16 to 6-bit grid loses subtlety |

**Decision:**
- Track 1 (`dynamic_quant`) — output stays GGUF (current Q4_K format works fine for mixed-bit allocation).
- Track 2 (DWQ proper) — output is MLX safetensors only. New file-format path in hf2q.

---

## 6. mlx-native runtime inventory (researcher #10)

**MASSIVE FINDING:** mlx-native already supports MLX-affine quantized inference for 4/6/8-bit. Re-exported public symbols at `/opt/mlx-native/src/lib.rs:101`:

- `quantized_matmul` — scalar baseline, 4/6/8-bit (works for prefill but slow)
- `quantized_matmul_simd` — fp32 input, qmv (decode-only)
- `quantized_matmul_simd_bf16` — bf16 input, qmv_fast (decode-only)
- `quantized_matmul_simd_bf16_expert` — MoE-routed bf16 (Qwen3.6-A3B compatible)

Loader: `/opt/mlx-native/src/weight.rs:626 load_quantized_weights` reads `quantization_config.json` + sharded safetensors with `.weight + .scales + .biases` naming.

**Missing for production prefill:** `quantized_matmul_mm_affine.metal` (matrix-matrix with affine dequant). Template exists at `quantized_matmul_mm.metal` (GGML qmm). Substitute the dequant block: ~3-5 days kernel work.

This drastically reduces Track 2 Path A effort. Path A is now ~2-3 weeks of mostly Rust (loader + tensor variants + format detection in hf2q) + 3-5 days kernel work in mlx-native.

---

## 7. Iteration plan (canonical)

Replaces all prior iteration tables.

| Iter | Track | Scope | Status |
|---|---|---|---|
| 2 | — | Initial research + ADR `72fdee8` | DONE |
| 3 | — | Phase 1 TensorRef.data Arc refactor `437217d` | DONE |
| 4 | — | Phase 2 deep-dive `b3db220` | DONE |
| 4.5 | — | mlx-lm pivot — DWQ vs dynamic_quant taxonomy `35e33a5` | DONE |
| 5 | — | 10-researcher /cfa deep-dive `ede887c` | DONE |
| 5.5 | — | ADR structural cleanup `3ce0571` | DONE |
| 6 | — | Fix `materialize_cloned` deep-clone — Arc-share hot path `beb184f` (~52 GB save; bisect-verified falsifier) | DONE |
| 7 | 1 | Port `estimate_threshold` + BPW accounting + `SensitivityAlgorithm` enum + `2.0.gradient-alignment` constant `913af23` (16/16 falsifiers; pure-fn, no autograd) | DONE |
| 8 | — | ~~mlx-lm subprocess wrapper~~ — **REVERTED** at `<this-commit>` as architecturally misaligned (we don't use external tools).  See §11 "Architectural principle: no external tools". | REVERTED |
| 8a | 1 | **CPU autograd correctness oracle** at `src/calibrate/autograd.rs` — tape + 7 ops {matmul, add, mul, sub, square, sum, mean} forward + reverse + per-op finite-diff falsifier (17 tests). **Test-only** — never reachable from any runtime entry point. Sole purpose: serves as the analytical reference the GPU autograd (8b+) gets falsifier-tested against. | DONE |
| 8b | 1 | **GPU matmul** at `src/calibrate/autograd_gpu.rs` — `matmul_forward_f32` + `matmul_backward_f32` standalone functions composed from `dense_matmul_f32_f32_tensor` + `transpose_2d`. Forward: pre-transpose `W` then dispatch.  Backward: 1 dispatch for `dX = dY @ W^T`, 2 transposes + 1 dispatch for `dW = X^T @ dY`. 5 falsifiers PASS including parity with CPU oracle on `[4, 32] @ [32, 4]` forward + `[32, 32] @ [32, 32]` forward+backward, all within 1e-4 rel tol. | DONE |
| 8b.1 | 1 | **GPU autograd tape** at `src/calibrate/autograd_gpu_tape.rs` — `GpuTape` (`Rc<RefCell<...>>`) + `GpuTensor` types wrapping the iter-8b primitives. `OpKind` enum (`Leaf`, `Matmul {lhs_idx, rhs_idx, m, k, n}`); `backward(output, output_grad)` walks tape in reverse + dispatches per-op via `dense_matmul_f32_f32_tensor` + `transpose_2d` + `elementwise_add` (gradient accumulation when a node is parent to multiple children). 6 falsifiers PASS including **two-matmul chain `Z = (X @ W1) @ W2` backward** producing all 3 gradients (`dX`, `dW1`, `dW2`) matching CPU oracle within 1e-4 rel tol. | DONE |
| 8c | 1 | **GPU elementwise add/sub/mul + square via mul-self**.  3 OpKind variants (`ElementwiseAdd`, `ElementwiseSub`, `ElementwiseMul`); forward via mlx-native `elementwise_add` + `elementwise_mul` + `scalar_mul_f32`; backward via the analytical formulas (`dA = dY`, `dA = -dY`, `dA = dY·B`).  7 new falsifiers PASS including `square_via_mul_self` (validates `lhs_idx == rhs_idx` produces `2·X·dY` via accumulator) + `matmul_then_mul_chain` composed parity (dX, dW, dS all match CPU oracle).  Real Metal write→read barrier bug surfaced + fixed (single-encoder back-to-back dispatches reading each other's outputs need separate encoders or explicit barrier). | DONE |
| 8d | 1 | **GPU softmax forward + backward** — new `softmax_backward_f32` Metal kernel landed in mlx-native (`/opt/mlx-native/src/{shaders/softmax_backward.metal, ops/softmax_backward.rs}`); registered in `kernel_registry.rs`.  Computes `dx[b, i] = y[b, i] · (dy[b, i] − Σ_j y[b, j] · dy[b, j])` with one threadgroup per row and a tree reduction for the row-dot.  hf2q wires `OpKind::Softmax {input_idx, rows, cols}` into GpuTape.  CPU oracle softmax + finite-diff falsifier added to `autograd.rs` (5/5 PASS); GPU↔CPU oracle parity (3/3 PASS) within 1e-5/1e-4 rel tol on non-trivial weighted-loss fixture. | DONE |
| 8e | 1 | **GPU log + log_softmax via composition** — new `log_f32` + `log_backward_f32` Metal kernels in mlx-native (`shaders/log_elementwise.metal` + `ops/log_elementwise.rs`).  Forward: `log(x)`.  Backward: `dx = dy / x`.  hf2q wires `OpKind::Log {input_idx}` into GpuTape.  log_softmax then composes as `log(softmax(x))` and gets correct backward from the autograd graph automatically.  4 new tests PASS including `log_softmax_via_composition_backward_parity` — proves end-to-end composition: GPU `log(softmax(x))` gradient matches CPU oracle within 1e-3 rel tol on non-trivial weighted loss `sum(C · log_softmax(X))`. | DONE |
| 8f | 1 | **GPU KL-div via composition + row_sum kernel** — new `row_sum_f32` + `row_sum_backward_f32` Metal kernels in mlx-native (per-row reduction with broadcast backward).  hf2q wires `OpKind::RowSum {input_idx, rows, cols}` into GpuTape.  KL-div composes from softmax + log + sub + mul + row_sum.  **THE KEY TEST**: `gpu_tape_kl_div_via_composition_dq_equals_softmax_q_minus_p` — end-to-end KL forward + backward produces `dq = softmax(q) - softmax(p)` matching the analytical identity within 5e-3 rel tol.  This is the EXACT gradient mlx-lm `dynamic_quant.estimate_sensitivities` computes. **Autograd toolchain complete for dynamic_quant.** | DONE |
| 9 | 1 | **`estimate_sensitivities` algorithm core** at `src/calibrate/dynamic_quant_gpu.rs` — `SyntheticTwoLinearModel` (forward `logits = X @ W1 @ W2`), `kl_div_loss_per_row(logits_q, logits_p)` (composes softmax + log + sub + mul + row_sum), `estimate_sensitivities(tape, student_logits, teacher_logits, quantizables) -> BTreeMap<String, f64>` applies the full sensitivity formula `(∇_W KL · (W_low − W_high)).sum() / (numel / 1e6)` matching `dynamic_quant.py:38-106`. 3 falsifiers PASS including `iter9_estimate_sensitivities_two_linear_synthetic` — 32×32×32×32 synthetic MLP with HAND-DERIVED analytical reference (`∇_W2 = H1^T @ dq`, `∇_W1 = X^T @ (dq @ W2^T)` where `dq = softmax(student) − softmax(teacher)`); GPU sensitivity scalars match the reference within 5e-3 rel tol. | DONE |
| 10 | 1 | Wire estimate_sensitivities into a real 4-layer Qwen35 attention block (RMSNorm + Q/K/V/O projections); add per-batch streaming with `del grads + mx.eval(grad_accum)` rhythm; add real qdq primitive (CPU oracle + GPU). Split into 10a/10b/10c/10d (see §8.2). | **DONE** (10a+10b+10c+10d on `cfa/adr020-iter10/claude`) |
| 10a | 1 | **GPU qdq Q4_0 + Q8_0 primitive** — new `qdq_legacy.metal` Metal kernels in mlx-native (commit `d45b395`); per-32-block GGUF round-trip with signed-amax tree-reduce + f16-rounded scale.  hf2q `src/calibrate/qdq_gpu.rs` wires `qdq_q{4,8}_0_gpu` host-slice surface + `qdq_q{4,8}_0_to_tensor` GpuTape leaf surface.  21 byte-identical CPU↔GPU parity tests (11 mlx-native + 10 hf2q) including canonical-q_legacy oracle, realistic weight ranges, and W_low ≠ W_high diff sanity. | **DONE** (cfa/adr020-iter10/claude `7ac613d`) |
| 10b | 1 | **GPU RMSNorm backward kernels** + GpuTape `OpKind::RmsNorm` op.  Forward existed in mlx-native (`rms_norm.metal`).  New `rms_norm_backward.metal` adds 3 kernels: `rms_norm_compute_rms_inv_f32` (helper producing `r[rows]`), `rms_norm_backward_dx_f32` (per-row `dx = r·(dy·w − x·s·r²/D)`), `rms_norm_backward_dw_f32` (per-feature `dw = Σ_b dy·x·r`).  Auto-registered in `KernelRegistry::new()`.  hf2q wires `OpKind::RmsNorm { input_idx, weight_idx, rows, dim, eps }` (2-parent op) returning `parent_grads = vec![dx, dw]`.  Tests: 4 mlx-native (small parity, realistic Qwen35-shape parity, unit-w pinning, **finite-diff falsifier** at 5e-3 rel tol) + 3 hf2q tape (forward+backward parity, chained through matmul, shape-mismatch rejection).  24/24 tape tests pass. | **DONE** (mlx-native main `96122bb`, hf2q `cfa/adr020-iter10/claude` part-2) |
| 10c | 1 | **4-layer Qwen35 attention block on GpuTape** — real `RMSNorm → Q/K/V/O projections → SDPA(Q@K^T/√d → softmax → @V) → out_proj` chain consuming qdq'd weights from iter-10a as the W_low leaves.  Adds `OpKind::Transpose2d` + `pub fn transpose` to GpuTape for the `Q @ K^T` step.  New module `src/calibrate/qwen35_attention_block.rs` (`AttentionBlockConfig` + `AttentionBlockWeights` w/ √d_head scale folded into W_q + `AttentionBlockLeaves` w/ full-precision and qdq variants + `forward` + `estimate_attention_block_sensitivities`).  4 tests PASS (forward shape, backward to all 4 weight leaves, qdq Q4_0-vs-Q8_0 diff non-zero, end-to-end estimate_sensitivities + Q4_0/Q8_0 monotonicity ≥3/4 projections).  Full calibrate suite 217/217.  Single-head (n_heads=1) load-bearing autograd fixture; multi-head SDPA composes same primitives at scale (iter-11+). | **DONE** (cfa/adr020-iter10/claude `12d9d87`) |
| 10d | 1 | **Per-batch streaming pattern** — `pub fn estimate_attention_block_sensitivities_streaming(cfg, weights, qdq_fn, batch_inputs)`.  Iterator-driven loop: per batch allocates a FRESH `MlxDevice` + `GpuTape`, runs forward + backward + estimate_sensitivities → per-batch scalar map; accumulates into running BTreeMap; drops tape (Drop on GpuTapeInner releases all MlxBuffer Arcs; Metal reclaims unified-memory pages before next batch).  After all batches divides scalars by `n_batches`.  Mathematical identity proves equivalence to mlx-lm's grad-accumulator formulation.  Tests: streaming-vs-manual-mean within 1e-10 rel tol (3 independent batches with different distributions); zero-batches rejected loud.  Calibrate suite 219/219 PASS. | **DONE** (`cfa/adr020-iter10/claude`) |
| 11 | 1 | E2E on Qwen3-0.6B-base — measure sensitivity ranking, hand-spot-check vs current variance-magnitude DWQ-46 ranking on a few well-known sensitive layers (lm_head, output_norm). Split into 11a-11g (see §8.2). **Strategic question** (open 2026-05-07): autograd-faithful path requires building full Qwen3 forward on GpuTape (multi-head SDPA + RoPE + SwiGLU FFN + embedding + lm_head — ~weeks); FD-equivalent path uses 2-forwards-per-Linear + loss-delta = mathematically equivalent to first-order gradient-Taylor for moderate ΔW. mlx-lm canon = autograd; spot-check ranking = either works. Continuing autograd-faithful per §1.4 goal "no quality compromise" + mantra "no shortcuts." | IN PROGRESS — sub-iter break-down landed; sub-iter coding starts next loop |
| 11a | 1 | **Multi-head SDPA on GpuTape** — new `slice_2d_cols_f32` + `copy_2d_cols_into_f32` Metal kernels (mlx-native `c6db382`); auto-registered.  GpuTape `OpKind::Slice2dCols` (single parent, backward via zero-init dx + copy_into) + `OpKind::Concat2Cols` (two-parent, backward via two slices).  `pub fn multi_head_sdpa(q, k, v, n_heads, head_dim)` head-loop composition: per-head slice → transpose → matmul scores → softmax → matmul context → left-fold concat.  Tests (4 hf2q + 3 op + 4 mlx-native = 11 PASS): n_heads=1 matches CPU oracle, n_heads=2 parity 1e-4 rel tol, backward to Q/K/V, input validation, slice/concat byte-identical and round-trip identity.  Calibrate suite 226/226. | **DONE** (mlx-native main `c6db382`, hf2q `cfa/adr020-iter10/claude` `f999989`) |
| 11b | 1 | **SwiGLU FFN on GpuTape** — new `silu_f32` + `silu_backward_f32` Metal kernels (mlx-native main `b30e508`); auto-registered.  GpuTape `OpKind::SiLU` single-parent op + `pub fn silu(t)`.  New `src/calibrate/qwen35_ffn.rs` (`FfnConfig`/`FfnWeights`/`FfnLeaves` + `forward` composing matmul-gate, matmul-up, silu, mul, matmul-down).  Tests (4 mlx-native incl. **finite-diff falsifier** at 5e-3 rel tol over 6 saturation-regime probes; 1 hf2q tape silu; 3 hf2q FFN incl. forward parity 1e-4, backward to all 3 weight leaves + input, qdq diff non-zero).  **+ streaming flake fix**: pre-existing intermittent failure of `streaming_vs_per_batch_mean_byte_close` (per-batch `MlxDevice::new` Metal residency contention); new `GpuTape::reset()` clears nodes without dropping device, both streaming + manual paths now use shared device + tape-reset.  Verified 10/10 PASS in debug AND release.  Calibrate 230/230. | **DONE** (mlx-native main `b30e508`, hf2q `cfa/adr020-iter10/claude` `1d96fb4`) |
| 11c | 1 | **Full transformer layer on GpuTape** — composition-only (no new kernels).  New `src/calibrate/qwen35_layer.rs` (`LayerConfig`/`LayerWeights` w/ √d_head fold + `LayerLeaves` w/ full-precision and qdq variants + `forward` composing rms_norm → Q/K/V matmul → multi_head_sdpa → O matmul → residual → rms_norm → ffn::forward → residual).  5 tests PASS (forward shape+finite, backward to all 10 leaves, estimate_sensitivities over 7 quantizable Linears all finite+nonzero, **Q4_0/Q8_0 monotonicity ≥5/7 projections**, config validation).  Calibrate 235/235. | **DONE** (`cfa/adr020-iter10/claude` `2d0d698`) |
| 11d | 1 | **Multi-layer model on GpuTape** — new `embedding_lookup_f32` + `embedding_scatter_add_f32` Metal kernels (mlx-native main `8d01f5e`); auto-registered.  GpuTape `OpKind::Embedding { embedding_idx, ids_buf, batch, vocab, hidden }` (single-parent op; ids stashed as u32 MlxBuffer, non-differentiable).  `pub fn embedding(table, ids)` wrapper.  New `src/calibrate/qwen35_model.rs` (`ModelConfig`/`ModelWeights`/`ModelLeaves` + `forward` composing embedding → N×layer → final RMSNorm → lm_head matmul → logits).  Tests: 5 mlx-native (lookup parity, repeated ids, scatter-add parity, unused-ids-zero, lookup→scatter round-trip) + 3 hf2q tape (forward+backward parity, chained-through-matmul backward routes to E and W, oob ids rejected) + 5 hf2q model (forward [batch, vocab], backward to all 21 leaves for n_layers=2, **estimate_sensitivities over 15 quantizable Linears all finite+nonzero**, Q4_0/Q8_0 monotonicity ≥2/3, config validation).  Calibrate 243/243 PASS. | **DONE** (mlx-native main `8d01f5e`, hf2q `cfa/adr020-iter10/claude` `2e92634`) |
| 11e | 1 | **GGUF tensor-map → ModelWeights adapter** — pure-CPU framework-free converter `weights_from_gguf_tensors(cfg, &BTreeMap<String, Vec<f32>>)` mapping standard llama.cpp Qwen3-style tensor names to ModelWeights field structure.  Transposes all 7N+1 Linear weights `[out, in]→[in, out]` to match the tape matmul's `Y = X @ W` convention with W as `[k, n]`.  6 tests PASS: valid construction, e2e forward through adapter-derived ModelWeights produces finite logits, missing-tensor + wrong-shape errors include key + length context, transpose round-trip identity + 3x4 hand-checked fixture.  Scope: standard MHA only; grouped-query attention deferred to iter-11g (different K/V output dims `n_kv_heads * head_dim`, requires SDPA broadcast).  Calibrate 249/249 PASS. | **DONE** (`cfa/adr020-iter10/claude` `835b9fd`) |
| 11f | 1 | **Calibration data pipeline** — `CalibrationBatcher` takes pre-tokenized heterogeneous-length sequences and produces fixed-length `Vec<u32>` batches via right-pad / front-truncate.  `iter()` feeds directly into the streaming sensitivity estimator (one calibration sequence = one batch since `batch == seq_len` in our model convention).  `whitespace_hash_tokenize` deterministic stub for tests; real tokenizer plugs in at iter-11g.  13 tests PASS incl. e2e pipeline (tokenize 4 prompts → batcher → ModelWeights from synthetic GGUF → ModelLeaves → forward → finite [seq_len, vocab] logits).  + streaming test refactor to shared-device + tape.reset() reducing Metal contention flake.  Calibrate 262/262 PASS in release; ~261-262/262 in debug parallel (Metal-residency contention under heavy parallel device creation is a known systemic issue, deterministic in release + serial). | **DONE** (`cfa/adr020-iter10/claude` `2965f20`) |
| 11g | 1 | **Sensitivity-ranking comparison harness** — `LayerAggregator::Max`/`Sum`, `aggregate_per_layer_scores`, `spearman_rank_correlation` (handles ties via average rank), `top_k_overlap` (abs-magnitude based), `rank_position`.  19 tests PASS incl. perfect-agreement/inversion ρ=±1.0, hand-checked ρ=0.3 fixture, ties, all-tied degenerate, top-K perfect/disjoint, e2e integration showing 4-layer aggregate → ρ=1.0 + top-2 overlap=2 + most-sensitive layer ranks 0 in both rankings.  Calibrate 281/281 PASS in release.  **Real-Qwen3-0.6B-base spot-check** ("lm_head and output_norm rank in top half") deferred — needs ~1.5GB model download; queued as **iter-11h** (gated on model availability). | **DONE** (`cfa/adr020-iter10/claude` `93e3ee2`) |
| **11h** | 1 | **Real-Qwen3-0.6B-base spot-check** (gated on model availability) — download Qwen3-0.6B-base GGUF, load via iter-11e adapter, run iter-11g comparison harness against hf2q's variance-magnitude ranker, verify lm_head + output_norm rank in top half of both rankings. | **GATED — needs model download** |
| 12 | 1 | E2E on Qwen3.6-27B + Gemma 4 26B-A4B — measure RSS, time, output GGUF coherence, PPL vs variance-magnitude DWQ-46/48 baseline. | |
| 13 | 2 | Port `dwq_quantize` algorithm core directly to native GPU autograd — Linear-only Adam optimizer reusing iter-8 toolchain.  No subprocess wrappers.  Split into 13a/13b (see §8.2). | IN PROGRESS |
| 13a | 2 | **Adam optimizer kernel + hf2q AdamOptimizer** — new `adam_update_f32` Metal kernel (mlx-native main `201e9cc`) auto-registered.  In-place per-element update of param/m/v with caller-precomputed bias-correction denominators `(1 − β^t)`.  Hf2q `AdamOptimizer` tracks per-parameter state via `register_param`, batches updates per `step()` in one encoder.  Tests (5 mlx-native + 7 hf2q = 12 PASS): t=1 + t=10 step parity vs CPU oracle (1e-5 rel tol), zero-grad fixed point, **convergence f(x)=(x−5)² → 200 steps → x≈5 within 0.05**, multi-param simultaneous convergence (a→5, b→−3), step counter, missing-grad/shape-mismatch/duplicate-register/config-validation errors. | **DONE** (mlx-native main `201e9cc`, hf2q `cfa/adr020-iter10/claude` `de1df56`) |
| **13b** | 2 | **Differentiable qdq + DWQ training loop** — replaces parameterless Q4_0/Q8_0 round-trip with explicit per-group `scales` + `biases` learnable params; tape-side `qdq_affine` with backward routing gradients to scales/biases (param weight is FROZEN).  4 mlx-native Metal kernels (`qdq_affine_init/forward/backward_scales/backward_biases_f32`) + hf2q `OpKind::QdqAffine` + `pub fn qdq_affine` factory + `dwq_loop::init_affine_params_gpu` host-side wrapper.  Per mlx-lm `dwq.py` semantics: q_int frozen, scales+biases learnable; d/d(scales[g]) = Σ q_int[i]·dy[i], d/d(biases[g]) = Σ dy[i].  Tests (14 PASS: 9 mlx-native + 3 tape + 2 dwq_loop): forward parity vs CPU oracle, init parity incl. uniform-group degenerate, init+forward |qdq−w| ≤ s/2 quant-error bound, backward parity vs higher-precision CPU sum reductions, **finite-diff falsifier on both scales and biases (analytical = FD within 1% tol)** at the kernel level, **finite-diff falsifier on the same gradients via the tape** (qdq_affine → backward chain, with sub/square/ones_like accumulator), input-validation rejection of non-power-of-two group_size and dtype mismatch, **synthetic 2-tensor DWQ training loop convergence falsifier**: 4 + 6 groups @ 4-bit, perturb scales+biases by +5%, run 200 Adam steps over per-tensor reconstruction MSE.  Acceptance: best loss < 0.2 × initial (5× floor).  **Measured trajectory: initial=0.049 → step 50: 0.0043 (87× init init), step 100: 0.0032 (15× init), step 150: 0.0032 (saturated near analytical minimum)**.  Also asserts param L2 norm moved TOWARD analytical optimum.  Loss this iteration is per-tensor reconstruction MSE; logit KL-div via qdq_affine → reshape → matmul chain requires a tape `view`/reshape op (iter-13c). | **DONE** (mlx-native main `e2536ea` — kernel `599e494` + Clone-on-MlxDevice `e2536ea`; hf2q `cfa/adr020-iter10/claude` `ee1c7eb`) |
| 13c | 2 | **qdq_affine → reshape → matmul → KL-div training-loop completion** — tape `view` op (zero-copy shape change via `MlxBuffer::with_shape`; backward = shape-relabeled identity), tape `scalar_mul` op (KL-div temperature scaling 1/T per mlx-lm `dwq.py`), end-to-end synthetic 2-Linear teacher/student MLP with KL-div loss.  **Verified canonical-correct against `ml-explore/mlx@main`'s `mlx/primitives.cpp:3459-3525` `QuantizedMatmul::vjp`** (line 3487 explicit "no gradient wrt the quantized weights"; bias-grad = `sum(cotangent, -1)` over each group; scale-grad uses `wq = dequantize(w_q, scales=1, biases=0)` [= q_int] then `sum(cotangent * wq, -1)` — identical to iter-13b's analytical formulation).  Tests (5 new + 5 from iter-13b = 10 PASS): tape view forward/backward identity, view→matmul gradient accumulation in 1-D leaf, scalar_mul forward/backward, scalar_mul finite-diff falsifier (analytical dL/dx = 2c²x matches FD within 1e-3 rel tol), **synthetic 2-Linear KL training-loop convergence**: teacher = silu(X@W1)@W2 (FP32 oracle), student = silu(X@qdq(W1, s1, b1))@qdq(W2, s2, b2), loss = mean(KL(softmax(y_T/T) ‖ softmax(y_S/T))) with T=2.0, Adam over (s1,b1,s2,b2) at lr=1e-3, q_int frozen.  Perturb scales+biases by 2.0× at start.  Acceptance: best loss < 0.34 × initial (3× floor — KL is stricter than reconstruction MSE; 4-bit quantizer's irreducible error bounds convergence rate).  **Measured trajectory: initial=4.29e-4 → step 50: 9.2e-5 (4.66×), step 100: 6.0e-5 (7.2×), step 150: 4.2e-5 (10.2× reduction)**.  Non-triviality assertion: initial KL > 1e-4 (caught a class of false-positive failures during fixture tuning where 5% / 30% perturbations produced trivial KL <<1e-4). | **DONE** (mlx-native main `b28ece4` — `MlxBuffer::with_shape`; hf2q `cfa/adr020-iter10/claude` `d2a6a8b`) |
| 13d | 2 | **Multi-step DWQ loop on real GGUF tensor** — loads `blk.0.attn_qkv.weight` (52,422,400 f32 elements, 1.6M groups @ group_size=32, 4-bit) from a Qwen 3.6 27B Q4_0 GGUF via mlx-native's `GgufFile::load_tensor_f32`, runs the affine init kernel, perturbs scales+biases by 2.0×, and trains 100 Adam steps.  Test gated `#[ignore]` (multi-GB GGUF on disk).  **Measured (15s runtime on M5 Max)**: init reconstruction MSE 9.38e-7 (per-group min/max heuristic near-optimal for real Q4_0 distributions); 2× perturbation start loss 2.44e-4 (260× init MSE → non-trivial); 100-step convergence to 1.47e-6 (165× reduction, ratio 0.006, far past the 5× floor); all 1.6M scales + 1.6M biases stay finite.  Validates real-tensor magnitude regime, per-group init on real distributions, KL gradient stability over 100+ steps with non-synthetic activations, peak GPU memory bounded via shared-tape + per-step `tape.reset()` (no `MlxDevice::new()` churn). Sibling tooling: `src/bin/dump_gguf_blk0.rs` diagnostic. | **DONE** (hf2q `cfa/adr020-iter10/claude` `2a14e9f`) |
| 13e | 2 | **Real-tensor KL-div training loop on full DWQ chain** — runs `qdq_affine → view → matmul → scalar_mul → kl_div_loss_per_row → backward → Adam.step` on a real production GGUF tensor.  W_real = `blk.0.attn_qkv.weight` (Qwen 3.6 27B Q4_0, GGUF [10240, 5120], transposed to [5120, 10240]).  X = deterministic Gaussian(0,1) [m=64, in=5120] via xorshift64* + Box-Muller (proxy for post-RMSNorm residual stream).  Teacher = X @ W_real (host FP64-acc oracle); student = X @ qdq(W_real, s, b).  Loss = mean(KL(softmax(y_T/T) ‖ softmax(y_S/T))), T=2.0.  Adam(lr=1e-3, β1=0.9, β2=0.999, ε=1e-8) over (s, b); q_int frozen.  Perturbation: 2.0× on scales+biases.  **Verified canonical-correct** against mlx-lm `tuner/losses.py:377` (raw logits in, logsumexp internal; q=student, p=teacher; reduction="none") + `dwq.py:106` (scale = 1/T).  **Measured (14s on M5 Max)**: teacher logit stddev 1.08 (healthy softmax post T=2.0); initial KL @ 2× perturbation = 3.12e-1; 100-step convergence to 4.99e-3 (**62.6× reduction**, ratio 0.016, far past 3× floor); all 1.6M scales + 1.6M biases finite throughout. | **DONE** (hf2q `cfa/adr020-iter10/claude` `b2485c7`) |
| 14 | 2 | **`compute_dwq_targets` stream-to-disk port** — direct port of `mlx-lm/mlx_lm/quant/dwq.py:29-66`.  Public API: `TeacherLogitsProvider` trait (forward_logits→Vec<f32>); `ComputeTargetsConfig {top_k, save_dir, vocab}`; `CalibrationSplit {name, batches, batch_size, seq_len}` (multi-split per `dwq.py:65-66`); `pub fn compute_dwq_targets(teacher, splits, cfg) -> Vec<(split, n_batches)>` (trims `[:, :-1]` per `dwq.py:53` internally so providers stay unaware of next-token semantics); `pub fn load_dwq_target(save_dir, split, idx) -> (logits, indices, batch, seq, top_k)` for the consumer side.  Top-K via host-side `BinaryHeap` partial-sort (O(V log K)) — mlx-native's `top_k_f32` caps at K=128 vs mlx-lm's K=1024; for one-shot precompute the teacher forward dominates wall time, host-side partial sort is the right tool (revisit in 14b if profiling demands).  Output format byte-compatible with mlx-lm: `{"logits": [B, S-1, K] f32, "indices": [B, S-1, K] u32}` per safetensors file at `<save_dir>/<split>/<i:010d>.safetensors`.  Tests (7/7 PASS): top-K parity vs full-sort oracle at K∈{1,2,5,32,128,256}; deterministic tie-break (ascending index); round-trip synthetic (240 (split,batch,row,k) cross-validations); rejection tests for seq_len<2, top_k>vocab, batch shape mismatch; safetensors byte-identity round-trip.  Iter-14b: `GgufTeacherProvider` once model-loading prereqs land. | **DONE** (hf2q `cfa/adr020-iter10/claude` `b1b0b1f`) |
| 14b | 2 | `GgufTeacherProvider` — `TeacherLogitsProvider` impl backed by hf2q's existing GGUF model loader + GPU forward path.  Spot-check on one calibration batch (2K tokens) against a peer (e.g. llama.cpp same model) for top-K parity.  Defensive: cap teacher RSS via `model.drop()` before returning to dwq_quantize (`dwq.py:386-387`). | NEXT |
| 14 | 2 | Port `compute_dwq_targets` stream-to-disk; teacher-drop sequencing. | |
| 15 | 2 | **mlx-native `qmm_affine_t_f32` fused dequant+matmul kernel** — DWQ inference primitive computing `y[m,n] = Σ_k x[m,k] · (q_int[n,k]·scales[n,g(k)] + biases[n,g(k)])` in one Metal pass.  Avoids materializing the dequantized [N×K] FP32 weight tensor (relevant for Linears where N·K·4 bytes is hundreds of MB).  Layout matches iter-13b's `qdq_affine` (UNPACKED uint8: one byte per nibble); packed-byte variant deferred to iter-15b.  **Verified canonical-correct against `ml-explore/mlx@main` `quantized.h:521-526`** (affine dequant `s[0]*(b & 0x0f) + bias` — same `q·s + b` shape) + `quantized.h:573-578` (group axis = K reduction dim).  Constraints: M,N,K > 0; group_size pow-of-two in [2,1024]; K divisible by group_size.  Tests (5/5 PASS): per-element parity vs FP64-accumulator CPU oracle; unaligned M=7,N=13 (not divisible by tg size 16); **fused-vs-composed cross-check** against `qdq_affine_forward + host matmul`; input-validation rejection (non-pow-of-2 group_size, dtype mismatch).  Performance: one-thread-per-output-element correctness-first kernel; tiled + simdgroup-MMA variant matching mlx's `affine_qmm_t` (BM=BK=BN=32, WM=WN=2, mlx::steel::BlockMMA) lands in iter-15b. | **DONE** (mlx-native main `9c19a31`) |
| 15b | 2 | **Tiled qmm_affine variant** — 16x16 thread block, cooperative-load `x_tile[BM, BK]` + `q_tile[BN, BK]` + `s/b_tile[BN]` into 2688 bytes of threadgroup-shared memory; per-thread inner reduction with register-resident `(s, b)` pair.  Constraint: `group_size == 32` baked in (BK = 32 = group_size → one (s, b) pair per K-tile per output row); other group_size values fall back to iter-15 per-element kernel.  Tests (3 new + 5 from iter-15 = 8/8 PASS): tiled-vs-per-element byte parity (LOAD-BEARING), tile-edge handling at M=23, N=47, group_size!=32 rejection.  **Measured (M5 Max, shape 64×4096×4096 attention-projection-class, 20-iter average post-warmup)**: per-element 2.40 ms = 894 GFLOPS; tiled 1.05 ms = 2047 GFLOPS; **speedup 2.29×**.  Sibling: `src/bin/bench_qmm_affine.rs`.  Simdgroup-MMA variant matching mlx's `affine_qmm_t<BM=BK=BN=32, WM=WN=2, mlx::steel::BlockMMA>` (target 6-7 TFLOPS, ~3-4× more) lands in iter-15c. | **DONE** (mlx-native main `165189a`) |
| 15c | 2 | Simdgroup-MMA variant of qmm_affine — port mlx's `affine_qmm_t` template (BM=BK=BN=32, WM=WN=2, 128 threads/TG, 4 simdgroups, `mlx::steel::BlockMMA` backend) for full ~6-7 TFLOPS performance.  Add gs=64 (mlx-lm default) variant alongside gs=32. | NEXT |
| 16b | 2 | **mlx-safetensors writer** (iter-16 inverse) — serializes `MlxAffineLinear` back to mlx-format safetensors bytes.  Closes the train→save→reload→serve loop required by iter-17.  API: `pack_u32_codes` (range-validates + packs), `write_floats_from_f32` (F32→BF16/F16/F32 LE), `MlxAffineLinear::to_safetensors_bytes(float_dtype) → MlxAffineLinearBytes`, `MlxAffineLinearBytes::to_safetensors_views() → 3 TensorViews` (two-step API keeps lifetimes clean).  Pack convention verified against same canonical fixture as iter-16's reader (codes [0xA,...,0x9] → bytes [0x3A, 0x17, 0xE5, 0x92]).  Tests (5 new + 13 from iter-16 = 18/18 PASS): F32 + BF16 round-trip via reader; pack-convention parity; out-of-range code rejection; **load-bearing multi-Linear save+load** (q_proj+k_proj+v_proj batched into one safetensors file, q_int byte-identical, scales/biases within bf16 0.4% rel tol). | **DONE** (hf2q `cfa/adr020-iter10/claude` `9aa2076`) |
| 16 | 2 | **hf2q MLX-safetensors loader** — reader for mlx-lm's affine-quantized save format.  Public API: `MlxQuantConfig::from_config_json` (parses `quantization.{bits, group_size, mode}` + per-layer overrides), `MlxAffineLinear::from_safetensors(st, path, bits, group_size)` (loads `<path>.{weight, scales, biases}` triplet, unpacks U32-packed weight → one-byte-per-code uint8 layout, casts BF16/F16 scales+biases → F32), `unpack_u32_packed`, `read_floats_to_f32`, `discover_shards` (sharded vs single-file).  **Verified canonical-correct against `mlx/ops.cpp:4789-4798`** (output dtypes {U32, w.dtype, w.dtype} + shapes {[N, K·bits/32], [N, K/gs], [N, K/gs]}), `mlx/ops.cpp:4762-4772` (pack: element `i` at bits `[i*bits, (i+1)*bits)`, lowest-index = LOW bits of u32, LE), `mlx-lm/utils.py:813-846` (config dual-write).  Tests (13/13 PASS): pack+unpack round-trip @ 4-bit + 8-bit; hand-computed byte layout verification (codes [0xA,0x3,...,0x9] → bytes [0x3A, 0x17, 0xE5, 0x92]); F32/F16/BF16 roundtrip; full mlx-format synthetic round-trip; BF16-scales path; dtype mismatch rejection; per-path config overrides (bool + dict); shard discovery; **load-bearing e2e**: build mlx-format synthetic Linear → serialize → load via `MlxAffineLinear::from_safetensors` → run iter-15's `qmm_affine_t_f32` kernel → verify against FP64-accumulator host oracle (proves byte-compatibility between loader and iter-15 kernel).  bits∈{2,3,5,6,7} deferred to iter-16b. | **DONE** (hf2q `cfa/adr020-iter10/claude` `c41e3d6`) |
| 17 (partial) | 2 | **Synthetic-teacher e2e cycle test** — first end-to-end demonstration of the entire iter-13/14/15/16/16b stack composing correctly.  Chain: synth W_real → init_affine_params → perturb → DWQ training (qdq_affine → view → matmul → scalar_mul → kl_div_loss → backward → Adam.step) → trained `MlxAffineLinear` → `to_safetensors_bytes(BF16)` → `safetensors::serialize` → `SafeTensors::deserialize` → `from_safetensors` → `qmm_affine_t_f32` inference → assert `y_reloaded` ≈ `y_trained`.  Two tests in `src/calibrate/dwq_e2e.rs` (3078 calibrate tests total).  **BF16 path**: 50 Adam steps reduce KL 5.19e-2 → 6.53e-3 (8× reduction past 3× floor); saved 2288 bytes for 3 tensors; q_int byte-identical round-trip; scales/biases within bf16 0.4%/element; acceptance via **relative L2 norm of (y_reloaded - y_trained) / ‖y_trained‖ < 5%** (model-identity check robust to bf16 absolute noise floor on small-magnitude outputs); **measured rel L2: 0.052%** (95× headroom).  **F32 path**: byte-identical round-trip on all three tensors; inference matches within FP rounding (1e-5 rel tol).  Test runtime 0.29s release.  Real-model e2e (production engine + tokenized calibration corpus) → iter-17b once iter-14b lands GgufTeacherProvider. | **DONE (synthetic chain)** (hf2q `cfa/adr020-iter10/claude` `d5a1f8c`) |
| 17b | 2 | Real-model e2e: replace synthetic teacher in iter-17 with `GgufTeacherProvider` (iter-14b dependency); run on a real Linear from on-disk Qwen 3.6 27B Q4_0 + a tokenized calibration batch from llama.cpp test corpus; spot-check post-DWQ logit distribution against a peer (llama.cpp / mlx-lm) within bf16 precision. | NEXT (gated on 14b) |
| 18 | 3/4 | AWQ, GPTQ — conditional on demand. | |
| 19 | — | Benchmark suite + ADR closure. | |
| **19a** | — | **Quality acceptance gate** — formal KL-divergence target for the DWQ-trained model.  **Canonical mlx-lm reference** (smcleod's `mlx_lm.kld` published Apr 2026 against an 8-bit ref): mlx-community DWQ Q4 on Qwen 3.6 35B-A3B (MoE) = **mean per-token KLD 0.02663**; same model RTN Q4 (no DWQ) = 0.07418 (~2.8× worse).  **Quality bands** (smcleod, mlx-lm canon): <1e-4 identical · 1e-3–5e-3 well-made 6-bit · 1e-2–5e-2 4-bit (DWQ territory) · >1e-1 *broken* (sampled outputs differ obviously).  `LEARNED_QUANTS.md` confirms DWQ is a 4-bit-and-below tool — initial KL at 6/8-bit is already ~0.01 with no headroom.  **Acceptance gate for hf2q port**: final per-token KL ≤ **0.030** on Qwen 3.6 35B-A3B at Q4 (matches mlx-lm published 0.02663 + 13% margin).  **>0.100 = broken** — no shipping past that floor.  Iter-13e/iter-17 partial measurements (4.99e-3 / 6.5e-3 synthetic) are NOT comparable — synthetic teacher with K=64 reduction has trivially-low intrinsic entropy.  Real-model KL is the only load-bearing measurement. | **TARGET LOCKED — gate adopted** |
| 19b (live) | — | **Live mlx-lm DWQ run on cached 35B BF16 — IN PROGRESS** as of 2026-05-07.  Working dir: `scripts/dwq_kl_parity/abliterix_with_chat_template/` (weight symlinks + tokenizer fixes from `00_prep_working_dir.sh`).  Two harness fixes required for the abliterated model: (1) inject `chat_template` from cached `Qwen/Qwen3.6-27B` (maintainer stripped); (2) remap `model_type "qwen3_5_moe_text"` → `"qwen3_5_moe"` (mlx-lm 0.31.2 MODEL_REMAPPING gap).  **Memory pivot — 3-pass low-mem recipe** (`09c7c9a`): naive single-pass invocation OOM'd at training step 1 even at batch=2 because the BF16 teacher (~70 GB) + Q4 student (~18 GB) + Adam state both live in memory.  Fix: split into three idempotent passes — Pass 1 `mlx_lm.convert -q` materializes RTN-Q4 student to disk (11 s, lazy mmap, no peak); Pass 2 `mlx_lm.dwq --targets-only --target-dir DIR` does teacher-forward only (top-1024 logits → 6.6 GB on disk, **759 s = 12.6 min**, no Adam, no student in memory); Pass 3 `mlx_lm.dwq --quantized-model RTN --target-dir DIR --grad-checkpoint` loads ONLY the Q4 student + Adam state; targets stream from disk.  **Live numbers at iter 303/1024 (29.6%, 1h 16m elapsed)**: peak unified memory **constant at 60.628 GB** (47% of 128 GB budget — half what naive OOM'd at), throughput **65-75 tok/s**, training avg_loss trajectory descending: it=19 → 0.0886, it=99 → 0.0531, **it=199 → 0.0380 (below initial validation 0.050)**.  ETA ~2:45 from now.  RTN-Q4 baseline (Pass 1 output) reused by step 03.  **Captured initial number**: `Validation: it=0, loss=0.050` at T=2.0 (per-token KL between BF16 teacher and fresh RTN-Q4 student, BEFORE DWQ training).  Consistent with smcleod's published RTN-Q4 baseline of 0.07418 on the same model class (within 33%; differences from sample seed / corpus shard).  Final validation + post-mlx_lm.kld measurement land when training completes. | **3-PASS RECIPE LIVE — Pass 1+2 done; Pass 3 at it=303/1024, peak_mem 60.6 GB** (PID 51393, `09c7c9a`) |
| 19b (half-1) | — | **Real-model mlx-lm KL parity harness — landed** (FIRST HALF: captures the canonical mlx-lm number; second half compares hf2q port).  Reference confirmed already cached (65 GB BF16 at `~/.cache/huggingface/hub/models--jenerallee78--Qwen3.6-35B-A3B-Abliterix-EGA-abliterated/`, 42 safetensors shards, `Qwen3_5MoeForCausalLM` hybrid 3:1 linear+full attention).  Vendored `mlx_lm.kld` (PR #1146, commit `cceccbe326b7d`, 525 LOC) into `scripts/dwq_kl_parity/kld.py` + 18-line `load_eval_tokens` shim around `mlx_lm.quant.utils.load_data` (PR's expected import is from a separate downstream PR not yet merged).  Three runnable scripts: `01_run_dwq.sh` (mlx-lm DWQ with canonical defaults bits=4 gs=64 temp=2.0 lr=1e-6 batch=4), `02_run_kld.sh` (KLD measurement vs BF16 ref, acceptance ≤0.030), `03_run_rtn_baseline.sh` (naive RTN-Q4 + KLD, expected ~0.074 ratio). All scripts append JSON to `results.jsonl`.  Runtime budget: half-day on AC power (step 01 hours @ 75 GB peak; steps 02/03 ~30-60 min @ 80 GB peak).  **NOT KICKED OFF AUTONOMOUSLY** — harness ready, awaits user authorization to run the multi-hour mlx-lm DWQ training. | **DONE — harness ready** (hf2q `cfa/adr020-iter10/claude` `6bdf985`) |
| 19b (half-2) | — | **DEFERRED → ADR-022 (subprocess bridge)** as of 2026-05-07.  Original plan: run our pure-Rust port on the same model + same calibration corpus + same seed → measure per-token KL via the same vendored kld.py → compare against half-1's measured target.  Original gate: iter-14b (real GgufTeacherProvider) + iter-11h (full multi-layer Qwen3.5MoE forward on GpuTape including hybrid linear+full attention layer types).  After CFA session `cfa-20260507-191500-adr020-followup-research` (5 parallel research workers + queen synthesis): iter-11h scope is 3-5K LOC of autograd plumbing including research-level gated-delta-net backward + MoE router gradient routing, projecting 5-7 weeks at historical iter cadence.  ADR-022 supersedes with a subprocess-bridge approach (~1,300 LOC + 150 LOC Python driver, ~5 weeks total, 6 iters iter-19d → iter-20a).  hf2q owns CLI + calibration + GGUF emission + parity measurement; pinned `mlx_lm.dwq` owns forward+backward.  Load-bearing v1 ship gate: hf2q-DWQ KL ≤ 0.0702 (= mlx-lm 0.0610 × 1.15) on stock Qwen 3.6 35B-A3B.  See [`ADR-022-dwq-port-completion-via-mlx-lm-subprocess.md`](./ADR-022-dwq-port-completion-via-mlx-lm-subprocess.md) for full plan + acceptance criteria + 5 worker-report citations under [`docs/research/cfa-adr020-iter19d/`](./research/cfa-adr020-iter19d/). | **DEFERRED — see ADR-022 Path B** (proposed 2026-05-07) |
| 19c | — | **Single-Linear KL parity vs cached 35B BF16** — first concrete real-BF16-reference KL measurement.  Loads `model.language_model.layers.0.linear_attn.out_proj.weight` (BF16, [2048, 4096], 8.4M elements) from the cached `jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated` safetensors (user correction: model is already cached, no download needed).  Runs DWQ training over per-group affine scales+biases, measures per-row KL between BF16-teacher inference and DWQ-student inference on m=64 random Gaussian activations.  Adam(lr=1e-3, β1=0.9, β2=0.999), 100 steps, T=2.0.  **Measured (3.64s runtime)**: KL @ analytical init = **8.77e-4** PASS; KL @ 2× perturbed start = 1.80e-1; KL @ post-train = **2.73e-3 PASS — 10× better than mlx-lm Q4 target 0.02663**; teacher logit stddev 9.79e-1 (healthy softmax).  **Honest interpretation**: single-Linear KL on m=64 random Gaussians is easier to drive low than full-model KL (which compounds across 47 layers with non-linearities); 2.73e-3 here NOT directly comparable to mlx-lm's full-model 0.02663.  What this DOES prove: our DWQ chain (init → train → KL backward → Adam) operates correctly on a real BF16 weight at production shapes — no NaN, no divergence, monotonic convergence.  Refactor: `box_muller_gaussian` promoted from `dwq_loop::tests` private to module-level `pub fn`. | **DONE** (hf2q `cfa/adr020-iter10/claude` `0a90982`) |

**Total: ~12 weeks** of `/loop` iterations (after removing Path B subprocess gates from the plan).

---

## 8. Acceptance criteria (per track)

### 8.1 Iter 6 — `materialize_cloned` Arc-share fix — DONE `beb184f`

**Files:** `/opt/hf2q/src/ir/lazy.rs:314-356` (production) + `lazy.rs:1075-1212` (tests).

**Pass criteria (all met):**

1. ✅ **Falsifier `materialize_cloned_shares_arc_no_byte_copy`** asserts `Arc::ptr_eq(&src, &t.data)` (strongest available check) + `Arc::strong_count >= 3` (caller + lazy + t). **Bisect-verified**: test FAILS without the fix at the `ptr_eq` assertion; PASSES with it.
2. ✅ **Companion falsifier `materialize_cloned_owned_vec_path_byte_equal`** covers the `Materialized(Vec<u8>)` one-owner path; ensures the rare variant still works.
3. ✅ **Full suite green:** 20/20 `ir::lazy` unit tests + 2878/0/3 full bin suite + all integration test binaries (zero failures).
4. ✅ **No new dependencies.**
5. ✅ **Production-code delta:** 6 LOC inside `materialize_cloned` (rest of the diff is doc comments + 2 new tests).
6. ✅ **Commit cites this ADR section** + observed save (~52 GB on 27B-class).

### 8.2 Track 1 — Dynamic Quant (mixed-precision sensitivity)

**Source:** `/opt/mlx-lm/mlx_lm/quant/dynamic_quant.py:38-146`.
**Output:** GGUF with mixed Q4_K/Q5_K/Q6_K/Q8_0 (current hf2q output format — keep).

**Sub-iter breakdown:**

| Sub-iter | Subtask | Status |
|---|---|---|
| 7 | `estimate_threshold` (binary search) + `compute_bits_per_weight` MLX-affine accounting + `SensitivityAlgorithm` enum + `2.0.gradient-alignment` cache-version constant.  Lives at `src/calibrate/dynamic_quant.rs`.  Pure functions, no autograd. | DONE |
| 8 (aux) | mlx-lm comparison harness — `src/calibrate/dynamic_quant_external.rs`.  Subprocess wrapper around `python -m mlx_lm.quant.dynamic_quant` + sensitivities JSON parser.  **NOT Track 1 progress.**  Reframed as the parity oracle iter 10 will compare against to verify hf2q's native sensitivity output matches mlx-lm within 1e-3 relative.  18 unit falsifiers + 1 e2e gated `#[ignore]`. | DONE |
| 8a | Tape skeleton + ops {matmul, add, mul, sub, square, sum, mean} forward + reverse + per-op finite-difference falsifier.  CPU first (f32 contiguous slices); GPU port lands later. | NEXT |
| 8b | Activations: {softmax, log_softmax, gelu, silu (SwiGLU), relu} forward + backward + finite-diff falsifier. | |
| 8c | Norms: {RMSNorm, LayerNorm} forward + backward + finite-diff falsifier. | |
| 8d | Loss: {kl_div_loss, cross_entropy} forward + backward + finite-diff falsifier. | |
| 8e | Linear-layer composite (matmul + add + activation) forward + backward + finite-diff falsifier on a 2-Linear synthetic MLP. | |
| 8f | Embedding lookup + RoPE + transformer attention block forward + backward + finite-diff falsifier on a single-layer Qwen35-attention-like fixture. | |
| 8g | QDQ-as-identity-for-codes: dequantize gradient flows through as ∂y/∂w_dequant = identity (codes frozen, scales/biases as continuous params at iter 13+; for dynamic_quant only the dequantized values participate in the gradient). | |
| 8h | Move tape from CPU to mlx-native MlxBuffer / Metal kernels.  This is when the autograd becomes performant enough for 27B-class.  Until 8h, all algorithm validation runs on CPU at synthetic-fixture scale. | |
| 9 | Wire autograd into hf2q's Qwen35 forward pass (replace static-quantized weights with the 8a-8h primitives that track gradients); per-Linear gradient accumulation; full `estimate_sensitivities` algorithm core matching `dynamic_quant.py:38-106`. | |
| 10 | E2E on Qwen3-0.6B-base; falsifier: native-output sensitivity ranking is monotone in expected hand-checked direction on well-known sensitive layers (lm_head, output_norm). | |
| 11 | E2E on Qwen3.6-27B + Gemma 4 26B-A4B — RSS, time, GGUF coherence, PPL vs current variance-magnitude baseline. | |

**Iter-8 framing correction (mantra):**

Earlier iter-8 framing presented the subprocess wrapper as "Path B
gate" Track 1 progress.  That was a fallback masquerading as progress.
Re-read of `~/Documents/mantra.txt`: *"DO NOT BE LAZY. We have plenty
of time to do it right. No short cuts. ... No fallback. No stub
(todo later) code. Just pure excellence, done the right way the
entire time."*  A Python subprocess IS the canonical fallback.  The
correct path is to build the autograd properly even if it takes
months of /loop iterations.  Iter-8 commit `8b7e43c` retained as a
parity oracle (its real future role) but reclassified as **aux**;
iter-8a now begins the actual native autograd work.

**Pass criteria:**

1. **Sensitivity vector analytical falsifier:** synthetic 4-layer Qwen fixture (≤ 100 MB, fits in test budget); hand-derive `∇_W = X^T @ (dq @ next_W^T)` chain analytically; assert hf2q's per-tensor sensitivity agrees with the analytical reference to within 5e-3 rel tol (matches iter-9's `iter9_estimate_sensitivities_two_linear_synthetic` pattern).  Test in `src/calibrate/dynamic_quant_gpu.rs`.
2. **Binary-search threshold falsifier:** for synthetic fixture with known optimal threshold T, assert `estimate_threshold` converges within `1e-3 * (max - min)` tolerance and selected predicate matches expected layer-allocation map.
3. **Cache key bump enforcement:** loading a `1.0.variance-magnitude`-keyed sensitivity JSON from `~/.cache/hf2q/sensitivity/` fails with a clear error instructing the user to delete the cache or recompute.
4. **CLI parity:** `hf2q convert --quant dwq-4-6 ...` and `hf2q convert --quant dynamic-quant-4-6 ...` produce byte-identical GGUF (alias preserved).
5. **End-to-end memory budget:** `hf2q convert --quant dynamic-quant-4-6` on Qwen3.6-27B with samples=1024 default; max RSS measured via `mach_task_basic_info` watchdog (poll every 100 ms) **< 80 GB**. SIGTERM at > 100 GB threshold.
6. **Output quality regression:** generated GGUF passes `gguf-dump` invariants from commit `0357394` (`tokens.len() == 248320`, `eos_token_id == 248046`).
7. **Coherence check:** `hf2q serve` loads the new GGUF, generates 16 tokens of non-degenerate text on a fixed prompt; manual eyeball + assert no `<|_end|>` literal-byte leak.
8. **Per-family pass:** all four combos {Qwen 3.6 35B-A3B-Abliterix-EGA, Gemma 4 26B-A4B-it-ara} × {dwq-4-6, dwq-4-8} satisfy criteria 5–7.

**How to verify locally:**
```bash
cargo test --test dynamic_quant_sensitivity_parity
cargo test --test dynamic_quant_threshold
hf2q convert --quant dynamic-quant-4-6 --input <fp16-source> --output /tmp/test.gguf --calibration-samples 1024
gguf-dump --no-tensors /tmp/test.gguf | grep -E '(tokens|eos_token_id)'
hf2q serve --model /tmp/test.gguf --prompt "the quick brown fox" --max-tokens 16
```

### 8.3 Track 2 — Native DWQ port

**Source:** `/opt/mlx-lm/mlx_lm/quant/dwq.py:69-209`.
**Output:** MLX safetensors via new hf2q output format.

Per the architectural principle (§11), Track 2 ports `dwq_quantize`
directly to native GPU autograd reusing the iter-8 toolchain — no
subprocess gates, no Python intermediaries.  The autograd toolchain
proven for `estimate_sensitivities` at iter 9 covers everything the
DWQ algorithm needs (Adam optimizer is ~50 LOC of arithmetic over
the same gradient buffers; KL-div + log_softmax + softmax already
landed; Linear-only forward already landed).

**Pass criteria:**

1. **`dwq_quantize` algorithm core landed natively** — Adam optimizer
   over the trainable scales+biases of QDQ'd weights; KL-div loss
   reusing iter-8f composition; per-batch streaming with
   `del grads + commit_and_wait` rhythm matching mlx-lm's
   `del grads + mx.eval(grad_accum)` pattern at `dwq.py:178-179`.
2. **Synthetic-fixture training convergence falsifier:** on a tiny
   2-Linear fixture, 20 Adam steps must drive validation KL strictly
   downward (matches mlx-lm's `validation_loss < initial_validation_loss`
   guard at `dwq.py:202-207`).
3. **`compute_dwq_targets` stream-to-disk** matching `dwq.py:29-66` —
   safetensors target files written before student loads, teacher
   dropped between phases.
4. **mlx-native `quantized_matmul_mm_affine.metal` kernel:**
   - falsifier: scalar-baseline `quantized_matmul` vs new `mm_affine`
     kernel byte-identical on randomized 1024×1024 affine-quantized
     weight + 256×1024 input.
   - perf: at least 5× faster than the scalar baseline on
     Qwen3.6-27B prefill (single-layer microbench).
5. **hf2q MLX-safetensors loader:** loads the trained output via
   `mlx_native::weight::load_quantized_weights`; serve-path generation
   produces non-degenerate text on a fixed prompt.
6. **End-to-end memory budget:** `hf2q convert --quant dwq-native-4`
   on Qwen3.6-27B; max RSS `< 100 GB`.
7. **End-to-end quality:** delta-PPL vs Q4_K_M baseline `> 0.05 nats`.
8. **Per-family pass:** all four combos {Qwen 3.6 35B-A3B-Abliterix-EGA,
   Gemma 4 26B-A4B-it-ara} × {dwq-4, dwq-6} satisfy criteria 6–7.

### 8.4 (reserved — was Track 2 Path A; now folded into §8.3 above)

### 8.5 Track 3 — AWQ (optional)

**Trigger:** if delta-PPL from Tracks 1+2 is insufficient at 3-bit / 2-bit. Otherwise defer.

**Pass criteria** (when triggered): byte-output parity with `mlx_lm.awq` on synthetic fixture (`< 1e-3` relative); perplexity beats Q4_K_M by `> 0.03 nats` on WikiText-2; max RSS `< 100 GB` on Qwen3.6-27B.

### 8.6 Track 4 — GPTQ (optional)

**Trigger:** customer demand or research follow-up.

**Pass criteria** (when triggered): bit-identical output to standard MLX-quantized layer (per researcher #4 finding); max RSS `< 80 GB` on Qwen3.6-27B; perplexity within 0.01 nats of Q4_K_M (GPTQ is a reconstruction method, not expected to beat trained methods).

---

## 9. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Iter-6 Arc fix breaks an unrelated read site | Compilation-driven; full `cargo test` per commit; rollback on regression. ~20 LOC scope makes this low. |
| Track 1 sensitivity parity falsifier fails (mlx-lm + ours diverge) | Bisect: separate the qdq utility, then per-batch grad accumulator, then alignment metric. mlx-lm code is reference truth. |
| Path B GO/NO-GO gate inconclusive (delta-PPL ≈ 0.05 nats) | Expand to 5k WikiText sample; report confidence interval; reconsider gate threshold with measured noise floor. |
| `quantized_matmul_mm_affine.metal` kernel is harder than 3–5 days | Fall back to scalar `quantized_matmul` for prefill in v0.1 of Path A; ship anyway; optimize kernel in follow-up. |
| Multi-day scope drifts | Tracked across `/loop` iterations; each commits + pushes incrementally; user can interrupt anytime. ADR is single source of truth. |
| Path A native autograd is buggy | Falsifier test 8.4.1 catches it before any e2e run; finite-difference gradient check 8.4.4 catches the math. |

---

## 10. (reserved)

*(Section 10 originally tracked the Path B GO/NO-GO measurement.
Path B was reverted as architecturally misaligned — see §11.  No
gate measurement is performed; Track 2 ports natively.)*

---

## 11. Architectural principle: no external tools

**hf2q does not call out to external runtimes for production
compute.**  No Python subprocesses, no `mlx-lm` shellouts, no
`llama-cli` wrappers, no other repo's CLI as a runtime dependency.
Reference repos (`/opt/mlx-lm`, `/opt/llama.cpp`, `/opt/candle`,
`/opt/omlx`, `/opt/vllm`) exist as **read-only sources of
algorithmic ideas** — we read their code to understand the math,
then implement natively in hf2q + mlx-native.

**Why this matters:**

- **Robustness.** Subprocess wrappers create brittle
  Python-environment dependencies (mlx-lm version pin, virtualenv
  layout, command-line surface drift).  hf2q is a self-contained
  Rust binary.
- **Performance.** Spawning a Python process to compute one
  gradient amounts to seconds of process-startup + tokenizer-load
  overhead per call — orders of magnitude worse than a native
  call.
- **Mantra alignment.**  `~/Documents/mantra.txt` says *"No
  fallback. No stub. Just pure excellence, done the right way the
  entire time."*  A Python subprocess IS the canonical fallback,
  no matter how it's labeled ("Path B gate", "comparison harness",
  "parity oracle").  Calling Python for "validation" before
  writing the real code is a form of avoiding the work.
- **Architectural cleanliness.** External tools live in a
  different version space, ship updates on a different cadence,
  surface different errors.  Keeping hf2q's compute graph fully
  inside the Rust + Metal stack means one repo, one build, one
  failure mode.

**What the principle DOES allow:**

- Reading reference repos' source code (e.g.,
  `/opt/mlx-lm/mlx_lm/quant/dwq.py`) to understand algorithms.
- The CPU correctness oracle in `src/calibrate/autograd.rs` —
  pure Rust, `#[cfg(test)]`-only callers, never reachable from
  any production codepath.  This is a TEST artifact, not an
  external tool.
- Linking against `mlx-native` (which we own, modify, and ship
  alongside hf2q in lockstep).

**What was reverted under this principle:**

| Commit | Module | Reason |
|---|---|---|
| `8b7e43c` (`hf2q`) | `src/calibrate/dynamic_quant_external.rs` | Python `mlx_lm.quant.dynamic_quant` subprocess wrapper.  Deleted. |
| (planned, never landed) | `src/wrappers/mlx_lm_dwq.rs` | Track 2 Path B was a planned `mlx_lm.dwq` subprocess gate.  Removed from §7 iteration plan. |

---

## 12. Alternatives considered

- **Run on bigger box for cache priming:** requires external infra (cloud Mac 192–256 GB or Linux GPU). One-off cost per model + cache copy. Rejected — user wants the code FIXED, no fallback.
- **Lower `--calibration-samples` to 256:** empirically still kernel-panicked (init load hits 100+ GB before samples matter); also a quality compromise. Rejected.
- **Substitute K-quants (`q5_k_m`/`q6_k`):** explicit "no cheating, real DWQ" directive. Rejected.
- **`HF2Q_STREAMING_PHASE3_MUT=1` flag alone:** only addresses Phase 3 dispatch; verified empirically same 199 GB peak. Rejected.
- **Convert DWQ output → GGUF:** researcher #7 verified format incompatibility (lossy snap). DWQ output ships as MLX safetensors only.
- **Subprocess gates before native port:** explored briefly at iter 8; reverted under §11 architectural principle. Track 2 ports natively from the start.

---

## 13. References

- **hf2q code:** `src/main.rs:482-496`, `src/ir/mod.rs:25-50`, `src/ir/lazy.rs:130-225` + `:314-345`, `src/inference/models/qwen35/weight_loader.rs:700-750`, `src/inference/models/qwen35/full_attn.rs::FullAttnLayerWeights`
- **mlx-lm sources (read-only reference):** `/opt/mlx-lm/mlx_lm/quant/{dwq,dynamic_quant,awq,gptq}.py`; `/opt/mlx-lm/mlx_lm/tuner/`; `/opt/mlx-lm/mlx_lm/utils.py`
- **mlx-native runtime:** `/opt/mlx-native/src/ops/quantized_matmul.rs` (1407 LOC); `/opt/mlx-native/src/weight.rs:626 load_quantized_weights`; `/opt/mlx-native/src/lib.rs:101`
- **llama.cpp reference:** `/opt/llama.cpp/src/llama-quant.cpp:1109-1245`; `/opt/llama.cpp/tools/imatrix/imatrix.cpp:229`; `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1465`
- **Research artifacts:** `docs/research/cfa-mlxlm-deep-dive-2026-05-06/{researcher-1..9}.md` (~30K words)
- **Memory:** `project_hf2q_dwq_oom_root_cause_2026_05_06.md`, `project_qwen35_reconvert_paused_2026_05_05.md`, `project_qwen35_dwq_pre_505b5b8_broken_2026_05_05.md`
- **Mantra:** `~/Documents/mantra.txt` — "Measure 3x, cut once. No fallback. No stub. Pure excellence."

---

## Appendix A. Superseded plans (kept for archaeology)

The sections below describe earlier plans that were superseded as understanding evolved. Kept here so readers tracing iter 4/4.5 commits can find the original context.

### A.1 Original Decision (Phase 1 + Phase 2) — SUPERSEDED iter 4.5

The original plan was to eliminate Drivers A + B in two phases:
- **Phase 1**: `TensorRef.data: Arc<Vec<u8>>` refactor (KEPT — landed in `437217d`).
- **Phase 2**: streaming `DwqCalibrator` that builds one layer at a time, dropping each layer's F32 expansion before the next.

**Why superseded:** iter 4 verified that even with both phases peak ≈ 158 GB on Qwen3.6-27B due to GPU-resident weights themselves consuming ~104 GB. The "build a separate calibration model and stream layers" approach was patching the wrong layer — the architectural mismatch was deeper. Replaced by the iter-5 mlx-lm port plan (§§2, 4–8) which doesn't build a separate model at all.

### A.2 Phases A+B+D (mmap + no-copy + serve-path reuse) — SUPERSEDED iter 5

Iter 4.5 surfaced that llama.cpp uses `newBufferWithBytesNoCopy` to alias mmap'd GGUF pages directly into Metal buffers — zero bytes copied. Plan was to bring hf2q's profile in line:
- **Phase A**: mmap-backed safetensors loader (`TensorRef.data: Arc<MmapView>`).
- **Phase B**: `MlxBuffer::from_no_copy` in mlx-native.
- **Phase D**: reuse `cmd_serve` load path for `DwqCalibrator`.

**Why superseded:** iter 5 mlx-lm research showed the cleaner architectural fix isn't to copy llama.cpp's mmap-aliasing; it's to copy mlx-lm's per-batch streaming algorithm (`del grads + mx.eval` rhythm + `compute_dwq_targets` stream-to-disk + `donate_model=True`). mlx-lm achieves ~60–80 GB peak WITHOUT mmap aliasing, by structuring the algorithm differently. A+B+D may still be nice-to-have for serve-path efficiency post-tracks; deferred indefinitely.

### A.3 Phase 3 (F16 GPU weights, open question) — RESOLVED by iter-5 pivot

Original iter 4 worried that even with Phase 1+2, GPU-resident F32 weights (~104 GB) wouldn't fit. The mlx-lm port (§§2, 4) doesn't hold full F32 weights on GPU — it processes per-batch with the same model handle (`donate_model=True`). Question moot.
