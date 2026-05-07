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

### Step 3 (iters 11–12): Track 2 Path B — subprocess validation of mlx-lm `dwq.py`

GO/NO-GO gate before committing to native port. Validates the algorithm + output format end-to-end via `mlx_lm.dwq` + `mlx_lm.generate`. See AC §8.3.

### Step 4 (iters 13–18, conditional): Track 2 Path A — native DWQ port

Conditional on Path B passing the perplexity gate. Native Rust port + Adam + Linear-only autograd + stream-to-disk targets + mlx-native prefill kernel. See AC §8.4.

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
| 8 | aux | mlx-lm comparison harness `8b7e43c` — subprocess wrapper + JSON parser. **NOT Track 1 native progress** (mantra: no fallback). Repurposed as a parity/oracle harness for iter 13+ falsifiers (compare native sensitivity output against mlx-lm ground truth on synthetic fixtures). 18/18 falsifiers PASS. | DONE |
| 8a | 1 | **CPU autograd correctness oracle** at `src/calibrate/autograd.rs` — tape + 7 ops {matmul, add, mul, sub, square, sum, mean} forward + reverse + per-op finite-diff falsifier (17 tests). **Test-only** — never reachable from any runtime entry point. Sole purpose: serves as the analytical reference the GPU autograd (8b+) gets falsifier-tested against. | DONE |
| 8b | 1 | **GPU matmul** at `src/calibrate/autograd_gpu.rs` — `matmul_forward_f32` + `matmul_backward_f32` standalone functions composed from `dense_matmul_f32_f32_tensor` + `transpose_2d`. Forward: pre-transpose `W` then dispatch.  Backward: 1 dispatch for `dX = dY @ W^T`, 2 transposes + 1 dispatch for `dW = X^T @ dY`. 5 falsifiers PASS including parity with CPU oracle on `[4, 32] @ [32, 4]` forward + `[32, 32] @ [32, 32]` forward+backward, all within 1e-4 rel tol. | DONE |
| 8b.1 | 1 | **GPU autograd tape** at `src/calibrate/autograd_gpu_tape.rs` — `GpuTape` (`Rc<RefCell<...>>`) + `GpuTensor` types wrapping the iter-8b primitives. `OpKind` enum (`Leaf`, `Matmul {lhs_idx, rhs_idx, m, k, n}`); `backward(output, output_grad)` walks tape in reverse + dispatches per-op via `dense_matmul_f32_f32_tensor` + `transpose_2d` + `elementwise_add` (gradient accumulation when a node is parent to multiple children). 6 falsifiers PASS including **two-matmul chain `Z = (X @ W1) @ W2` backward** producing all 3 gradients (`dX`, `dW1`, `dW2`) matching CPU oracle within 1e-4 rel tol. | DONE |
| **8c** | 1 | GPU activations: {softmax, log_softmax, gelu, silu, relu} + finite-diff parity vs CPU oracle.  mlx-native already has `softmax`, `gelu`, `silu_mul` kernels — port the wrappers + add backward dispatches. | **NEXT** |
| 9 | 1 | Wire autograd into Qwen35 forward pass; per-Linear gradient capture; `estimate_sensitivities` algorithm core matching `dynamic_quant.py:38-106`. | |
| 10 | 1 | E2E on Qwen3-0.6B-base; falsifier: native-output sensitivity ranking matches mlx-lm subprocess output (using iter-8 harness) within 1e-3 relative on a synthetic fixture. | |
| 11 | 1 | E2E on Qwen3.6-27B + Gemma 4 26B-A4B — measure RSS, time, output GGUF coherence, PPL vs current variance-magnitude DWQ-46/48 baseline. | |
| 11 | 2-B | Subprocess wrapper around `mlx_lm.dwq`; produce MLX safetensors | |
| 12 | 2-B | GO/NO-GO gate: perplexity vs Q4_K_M baseline (>0.05 nats threshold) | |
| 13–14 | 2-A | Port `dwq_quantize` algorithm core; Linear-only autograd; Adam optimizer | |
| 15 | 2-A | Port `compute_dwq_targets` stream-to-disk; teacher-drop sequencing | |
| 16 | 2-A | mlx-native `quantized_matmul_mm_affine.metal` kernel | |
| 17 | 2-A | hf2q MLX-safetensors loader + tensor variants for runtime support | |
| 18 | 2-A | Track 2 e2e: hf2q produces + serves DWQ end-to-end | |
| 19 | 3/4 | AWQ, GPTQ — conditional on demand | |
| 20 | — | Benchmark suite + ADR closure | |

**Total: 14–16 weeks** of `/loop` iterations.

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
| 10 | E2E on Qwen3-0.6B-base; falsifier: native-output sensitivity ranking matches mlx-lm subprocess output (using iter-8 aux harness) within 1e-3 relative on a synthetic fixture. | |
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

1. **Sensitivity vector parity falsifier:** synthetic 4-layer Qwen fixture (≤ 100 MB, fits in test budget); compute sensitivities via hf2q's new path AND via subprocess to `python -m mlx_lm dynamic-quant` on same fixture; assert per-tensor relative diff `< 1e-3`. Test in `tests/dynamic_quant_sensitivity_parity.rs`.
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

### 8.3 Track 2 Path B — Subprocess validation gate

**Source:** `/opt/mlx-lm/mlx_lm/quant/dwq.py` invoked as `python -m mlx_lm dwq ...`.
**Output:** MLX safetensors directory.

**Pass criteria (all must hold to gate Path A):**

1. **Subprocess wrapper lands** in `src/wrappers/mlx_lm_dwq.rs` (or `scripts/`); takes `--input <fp16-source>` + `--bits 4 --group-size 64` + `--num-samples 2048 --max-seq-length 1025 --learning-rate 1e-6 --batch-size 4`; returns MLX safetensors at `--mlx-path <out>`.
2. **DWQ output validates with `mlx_lm.generate`:** loads + generates 32 tokens of coherent English on a fixed prompt; no NaN, no degeneracy.
3. **Perplexity GO/NO-GO gate** on 1k WikiText-2 test split:
   - Baseline: `mlx_lm.generate` on the same source quantized to Q4_K_M via existing hf2q.
   - Test: DWQ output via Path B.
   - **GO if:** `ppl(DWQ) < ppl(Q4_K_M) - 0.05 nats` (i.e., DWQ is at least 0.05 nats better in cross-entropy).
   - **NO-GO if:** delta < 0.05 nats (Path A not worth the effort).
4. **Memory profile during subprocess:** max RSS observed `< 100 GB` on Qwen3.6-27B with default samples=2048. (mlx-lm is the gold-standard here; if it OOMs, our box is too small for default-samples DWQ regardless.)
5. **Decision logged in ADR §10 "Path A GO/NO-GO outcome"** with measured perplexity numbers + recommendation.

**How to verify locally:**
```bash
hf2q dwq-via-mlx-lm --input <fp16-source> --output /tmp/dwq-test/ --bits 4
python -m mlx_lm.evaluate --model /tmp/dwq-test/ --task wikitext-2-ppl
python -m mlx_lm.evaluate --model <Q4_K_M-baseline> --task wikitext-2-ppl
# delta-PPL > 0.05 nats → GO
```

### 8.4 Track 2 Path A — Native DWQ port

**Conditional:** only execute if Path B's GO gate passes.

**Source:** `/opt/mlx-lm/mlx_lm/quant/dwq.py:69-209`.
**Output:** MLX safetensors via new hf2q output format.

**Pass criteria:**

1. **`dwq_quantize` algorithm parity falsifier:** on a synthetic 4-layer Qwen fixture, run hf2q's native dwq + mlx-lm's `mlx_lm.dwq` with identical seeds + hyperparameters; assert per-parameter relative diff `< 1e-3` after 20 training steps.
2. **`compute_dwq_targets` stream-to-disk parity:** byte-identical safetensors target files vs mlx-lm reference on synthetic fixture.
3. **Adam optimizer parity:** Adam state (m, v, t) byte-identical to mlx-lm's `optimizers.Adam(learning_rate=1e-6, bias_correction=True)` after 20 steps on synthetic gradient stream.
4. **Linear-only autograd correctness:** gradient-check via finite differences on `kl_div_loss(logits, targets)` for a 2-Linear synthetic forward pass; max relative error `< 1e-3`.
5. **mlx-native `quantized_matmul_mm_affine.metal` kernel:**
   - falsifier: scalar-baseline `quantized_matmul` vs new `mm_affine` kernel produce byte-identical output on randomized 1024×1024 affine-quantized weight + 256×1024 input.
   - perf: at least 5× faster than the scalar baseline on Qwen3.6-27B prefill (single-layer microbench).
6. **hf2q MLX-safetensors loader:** loads the output of (1) via `mlx_native::weight::load_quantized_weights`; serve-path generation matches the perplexity from Path B's `mlx_lm.generate` round-trip within 0.01 nats.
7. **End-to-end memory budget:** `hf2q convert --quant dwq-native-4 --target-dir /tmp/targets` on Qwen3.6-27B; max RSS `< 100 GB`.
8. **End-to-end quality:** delta-PPL vs Q4_K_M baseline `> 0.05 nats` (matches Path B gate, confirms native parity).
9. **Per-family pass:** all four combos {Qwen 3.6 35B-A3B-Abliterix-EGA, Gemma 4 26B-A4B-it-ara} × {dwq-4, dwq-6} satisfy criteria 7–8.

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

## 10. Path A GO/NO-GO outcome

*Filled in after iter 12 measurement.*

- Baseline Q4_K_M perplexity: TBD
- Path B DWQ perplexity: TBD
- Delta: TBD
- **Decision:** TBD (GO / NO-GO)

---

## 11. Alternatives considered

- **Run on bigger box for cache priming:** requires external infra (cloud Mac 192–256 GB or Linux GPU). One-off cost per model + cache copy. Rejected — user wants the code FIXED, no fallback.
- **Lower `--calibration-samples` to 256:** empirically still kernel-panicked (init load hits 100+ GB before samples matter); also a quality compromise. Rejected.
- **Substitute K-quants (`q5_k_m`/`q6_k`):** explicit "no cheating, real DWQ" directive. Rejected.
- **`HF2Q_STREAMING_PHASE3_MUT=1` flag alone:** only addresses Phase 3 dispatch; verified empirically same 199 GB peak. Rejected.
- **Convert DWQ output → GGUF:** researcher #7 verified format incompatibility (lossy snap). DWQ output ships as MLX safetensors only.
- **Skip Path B, go straight to Path A:** rejected. Path A is 3–4 weeks of native autograd + kernel work; Path B is 3–5 days and answers "is DWQ worth porting natively for these models" before committing.

---

## 12. References

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
