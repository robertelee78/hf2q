# ADR-020: DWQ + Mixed-Precision Quantization for hf2q (port from mlx-lm)

**Status**: Proposed — 2026-05-06; PIVOTED iter 4.5 (taxonomy correction); EXPANDED iter 5 (10-researcher /cfa deep-dive)

## Iter 5 research output (10 parallel researchers)

Full deliverables at `/tmp/cfa-mlx-lm-deep-research/`. Headlines:

| # | Topic | Headline finding |
|---|---|---|
| 1 | DWQ algorithm | KL-div distillation; optimizes scales+biases ONLY (codes frozen → no STE needed); 411 LOC; lr=1e-6, batch=4, samples=2048, max_seq=1025 |
| 2 | Dynamic Quant | Signed first-order Taylor: `(grad·(w_low−w_high)).sum()/params_M` ≠ hf2q's variance-magnitude proxy. Cache invalidation forced. |
| 3 | AWQ | 585 LOC scale-then-clip; activation-aware; needs `Catcher` per-Linear hook (~1-2 weeks port). Recommended AFTER dynamic_quant. |
| 4 | GPTQ | 229 LOC Cholesky-based least-squares; smallest of 4 algos; output bit-identical to standard MLX quantized layer. Port estimate 1-2 weeks. |
| 5 | Shared infra | `load(lazy=True)` + sharded safetensors at 5GB/shard + `donate_model=True` + `iterate_batches(pad_to=32)` + `kl_div_loss` Metal kernel (200 LOC, custom VJP) |
| 6 | Memory tricks | **3 load-bearing**: per-batch `del grads + mx.eval(grad_accum)` (saves ~60GB/step), stream-targets-to-disk in compute_dwq_targets (drops teacher ~60GB), fix hf2q's `materialize_cloned` deep-clone defect at lazy.rs:314-345 (~20 LOC) |
| 7 | File format | MLX safetensors (uint32 packed weight + fp16 scales + fp16 biases per group_size=64) ≠ GGUF Q4_K (super-block 256 weights, 6-bit compressed scales). DWQ→GGUF is LOSSY. DWQ output goes to MLX safetensors. |
| 8 | Port checklist | Phase 1: dynamic_quant (~80 LOC core, hand-derived gradients, no autograd needed). Phase 2: DWQ via subprocess wrapper (mlx_lm.dwq) or native port (autograd needed). |
| 9 | DWQ 2026 SOTA | DWQ has NO paper; canonical = Awni Hannun blog + GitHub. Best 4-bit MoE: 0.02663 KL on Qwen3.6-35B-A3B-4bit-DWQ (smcleod 2026-04). DWQ saturates at 6-bit. **DWQ remains SOTA for Apple Silicon W4A16 as of 2026.** FlatQuant/SpinQuant lead for NVIDIA W4A4 (different stack). |
| 10 | Runtime support | **MASSIVE**: mlx-native already has affine 4/6/8-bit quantized_matmul kernels at `/opt/mlx-native/src/ops/quantized_matmul.rs` (1407 LOC) + `.metal`. Decode (qmv_fast) + MoE-routed expert variant present. Loader at `/opt/mlx-native/src/weight.rs:626 load_quantized_weights`. Only missing: prefill qmm tile kernel (~3-5 days port from existing GGML qmm template). hf2q can serve DWQ natively. |

## Taxonomy lock-in (mlx-lm naming as of 2026)

| What hf2q called | What it actually is | mlx-lm canonical | Output format | Status in hf2q |
|---|---|---|---|---|
| "DWQ-46" | Sensitivity ranking → mixed Q4/Q6 GGUF | `dynamic_quant.py` | MLX safetensors per-tensor `{bits, group_size}` overrides | Misnamed; algorithm differs (variance-magnitude vs gradient-Taylor) |
| "DWQ-48" | Same with sensitive=Q8_0 | `dynamic_quant.py` (4/8 split) | Same | Same |
| (not implemented) | Distillation fine-tuning of quantized scales+biases | `dwq.py` | MLX safetensors w/ trained fp16 scales+biases | NEW capability — port required |
| (not implemented) | Activation-aware weight scaling | `awq.py` | MLX safetensors w/ scaled+clipped weights | Optional 3rd track |
| (not implemented) | Cholesky-based weight least-squares | `gptq.py` | MLX safetensors (bit-identical to standard quantized) | Optional 4th track |

## Port plan (4 algorithms, 3-week core + optional extensions)

Per researcher #8 + #10 synthesis:

### Track 1: Dynamic Quant (mixed-precision sensitivity-based) — REPLACE existing hf2q DWQ
**Source:** `/opt/mlx-lm/mlx_lm/quant/dynamic_quant.py:38-146`
**Output:** GGUF with per-tensor mixed Q4_K/Q5_K/Q6_K (this is what hf2q already does — keep this)
**Port subtasks:**
1. `estimate_sensitivities` algorithm (~80 LOC core): per-batch loop, qdq utility, gradient-alignment metric. Hand-derive `∂L/∂y · x^T` per Linear (no autograd graph).
2. `estimate_threshold` binary search for target BPW (~40 LOC).
3. Mixed-bit predicate dispatcher (already exists at `src/quantize/{mixed,layer_mix}.rs`).
4. Cache key bumped from "1.0.variance-magnitude" → "2.0.gradient-alignment" (hard-fail on legacy reads).
5. CLI: `--quant dynamic-quant-4-5` etc. (preserve `--quant dwq-4-6` as alias for backward compat).
6. Falsifier: synthetic 4-layer Qwen → sensitivity vector matches mlx-lm output within 1e-3 relative.

**Effort: 1-2 weeks. No autograd dep. No new file format. No mlx-native changes.**

### Track 2: DWQ proper (distillation fine-tuning of quantized scales+biases) — NEW capability
**Source:** `/opt/mlx-lm/mlx_lm/quant/dwq.py:69-209`
**Output:** MLX safetensors (NOT GGUF — researcher #7 verified incompatible byte layouts)
**Two viable paths per researcher #10:**

- **Path B (validation, days 1-3):** hf2q produces DWQ-quantized MLX safetensors via subprocess wrapper around `mlx_lm.dwq`. Validates the algorithm + format. Runtime is `mlx_lm.generate` (Python) for measurement only.
- **Path A (production, weeks 2-4, conditional on Path B success):** native port. Components:
  - `dwq_quantize` algorithm core (~140 LOC port). Adam optimizer + KL-div + Linear-only autograd.
  - `compute_dwq_targets` stream-to-disk (~60 LOC). Saves teacher logits per-batch to .safetensors files; teacher dropped before student loaded.
  - hf2q runtime: load MLX safetensors via `mlx_native::weight::load_quantized_weights` (already exists). Add prefill `quantized_matmul_mm_affine.metal` kernel to mlx-native (3-5 days; template exists at `quantized_matmul_mm.metal` for GGML).
  - Validation: `mlx_lm.generate` round-trip + perplexity vs Q4_K_M baseline.

**GO/NO-GO gate after Path B (day 5):** if perplexity beats Q4_K_M baseline by >0.05 nats on 1k WikiText, commit to Path A.

**Effort: 3-5 days for Path B (gate), then 3-4 weeks for Path A (full native).**

### Track 3: AWQ (optional, post-Track-1+2)
**Source:** `/opt/mlx-lm/mlx_lm/quant/awq.py:399-510`
**Trigger:** if DWQ + dynamic_quant don't deliver sufficient quality at 3-bit / 2-bit
**Effort: 1-2 weeks. Activation-aware needs `Catcher`-style per-Linear hooks; doubles as scaffolding for future GPTQ.**

### Track 4: GPTQ (optional)
**Source:** `/opt/mlx-lm/mlx_lm/quant/gptq.py:52-159`
**Effort: 1-2 weeks. Smallest algo (229 LOC). Cholesky-based; mlx-native already has Cholesky binding.**

## Memory architecture: 3 load-bearing patterns to port (researcher #6)

Per the 199 GB observed peak vs llama.cpp's ~10 GB:

### A. Per-batch `del grads + mx.eval(grad_accum)` rhythm
**Source:** `dynamic_quant.py:80-86` + `dwq.py:178-179`
**Saves:** ~60 GB/step on 27B-class
**hf2q port:** in the new `estimate_sensitivities` and `dwq_quantize` Rust impls, after each batch:
```rust
gradient_accumulator += &batch_grads;
drop(batch_grads);  // explicit drop — no accumulation
device.commit_and_wait()?;  // mlx-native equivalent of mx.eval
```

### B. Stream-targets-to-disk
**Source:** `dwq.py:29-66 compute_dwq_targets` + line 386-387 `del model`
**Saves:** ~60 GB on 27B (lets teacher be dropped before student trains)
**hf2q port:**
1. New mode: `--quant dwq-4 --target-dir /path` precomputes teacher top-1024 logits + indices to .safetensors files
2. Phase 1 (target precompute): teacher loaded, no student
3. `drop(teacher)` between phases
4. Phase 2 (student training): teacher gone, only student + per-batch targets read from disk

### C. Fix `materialize_cloned` deep-clone defect — INDEPENDENT improvement
**Source:** Researcher #6 verified at `/opt/hf2q/src/ir/lazy.rs:314-345`
**Bug:** even though Phase 1 (commit `437217d`) made TensorRef.data: Arc<Vec<u8>>, the `materialize_cloned` impl at lines 317-321 still does `(**arc).clone()` — deep-copies the Vec inside the Arc. Should be `Arc::clone(arc)` returning the Arc directly.
**Fix:** ~20 LOC in lazy.rs.
**Saves:** ~52 GB on 27B (the deep clone of source weights for every materialize call).
**Land first** — it's independent of the algorithm work and unblocks all subsequent paths.

## File format reality check (researcher #7)

DWQ-trained scales **cannot** survive a GGUF round-trip without quality loss:
- MLX format: per-group fp16 scales + fp16 biases (group_size=64). 32 GB metadata overhead for a 4096×4096 layer.
- GGUF Q4_K: per-super-block 6-bit compressed scales (256 weights/super-block, 8×32 sub-blocks, fixed format). Quantization op: snap fp16 trained scales onto 6-bit grid → loses subtlety.

**Decision:** DWQ output is MLX safetensors only. Track 1 (dynamic_quant) keeps GGUF output. Track 2 (DWQ) gets new MLX-safetensors output path.

## mlx-native runtime kernels — existing inventory (researcher #10)

**MASSIVE FINDING:** mlx-native already supports MLX-affine quantized inference for 4/6/8-bit. Re-exported public symbols at `/opt/mlx-native/src/lib.rs:101`:
- `quantized_matmul` — scalar baseline, 4/6/8-bit (works for prefill but slow)
- `quantized_matmul_simd` — fp32 input, qmv (decode-only)
- `quantized_matmul_simd_bf16` — bf16 input, qmv_fast (decode-only)
- `quantized_matmul_simd_bf16_expert` — MoE-routed bf16 (Qwen3.6-A3B compatible!)

Loader: `/opt/mlx-native/src/weight.rs:626 load_quantized_weights` reads `quantization_config.json` + sharded safetensors with `.weight + .scales + .biases` naming.

**Missing for hf2q:** `quantized_matmul_mm_affine.metal` (prefill matrix-matrix with affine dequant). Template exists at `quantized_matmul_mm.metal` (GGML qmm). Substitute the dequant block: ~3-5 days kernel work.

This drastically reduces Path A effort. Path A is now ~2-3 weeks of mostly Rust (loader + tensor variants + format detection in hf2q) + 3-5 days kernel work in mlx-native.

## Updated iteration plan

| Iter | Scope | Status |
|---|---|---|
| 2 | Initial research + ADR `72fdee8` | DONE |
| 3 | Phase 1 TensorRef.data Arc refactor `437217d` | DONE |
| 4 | Phase 2 deep-dive `b3db220` | DONE |
| 4.5 | mlx-lm pivot: DWQ vs dynamic_quant taxonomy `35e33a5` | DONE |
| 5 (this) | 10-researcher /cfa deep-dive | DONE — committing now |
| 6 | Fix `materialize_cloned` deep-clone defect (researcher #6 finding); independent ~20 LOC win | NEXT |
| 7-8 | Track 1: port `dynamic_quant.estimate_sensitivities` + `estimate_threshold` (algorithm core, hand-derived gradients) | |
| 9 | Track 1 falsifier tests + GGUF emit verification | |
| 10 | Track 1 e2e on Qwen3.6-27B + Gemma 4 26B-A4B; benchmark vs hf2q's existing variance-magnitude DWQ-46/48 | |
| 11 | Track 2 Path B: subprocess wrapper around `mlx_lm.dwq`; produce MLX safetensors; validate via `mlx_lm.generate` | |
| 12 | GO/NO-GO gate: if DWQ beats Q4_K_M by >0.05 nats, commit to Path A | |
| 13-14 | Track 2 Path A part 1: port `dwq_quantize` algorithm core; Linear-only autograd; Adam optimizer | |
| 15 | Track 2 Path A part 2: port `compute_dwq_targets` stream-to-disk; teacher-drop sequencing | |
| 16 | Track 2 Path A part 3: mlx-native `quantized_matmul_mm_affine.metal` kernel | |
| 17 | Track 2 Path A part 4: hf2q MLX-safetensors loader + tensor variants for runtime support | |
| 18 | Track 2 e2e: hf2q produces + serves DWQ end-to-end | |
| 19 | Tracks 3+4 (AWQ, GPTQ) — conditional on demand | |
| 20 | Benchmark suite + ADR closure | |

**Total: 14-16 weeks** of /loop iterations to land both algorithms with full runtime support. Phase 1 (commit 437217d) already shipped saves ~52 GB; iter 6 saves another ~52 GB; together they likely fit dynamic_quant-style runs on 128 GB without the rest of the stack.

## Critical taxonomy correction (2026-05-06 iter 4.5)

**hf2q's "DWQ-46" and "DWQ-48" are NOT what mlx-lm calls DWQ.** Reading
`/opt/mlx-lm/mlx_lm/quant/dwq.py` and `/opt/mlx-lm/mlx_lm/quant/dynamic_quant.py`
clarifies the gap:

| Algorithm | mlx-lm name | What hf2q has | What hf2q SHOULD have |
|---|---|---|---|
| Distillation fine-tuning of quantized scales+biases via KL-div from full-precision teacher (gradient-based) | `DWQ` (`dwq.py`) | not implemented | (not in scope for this ADR) |
| Sensitivity ranking → mixed-bit allocation (4-bit base, higher bits for sensitive tensors) | `dynamic_quant` (`dynamic_quant.py`) | misnamed as `DWQ-46`/`DWQ-48` | rename to `dynamic-quant` aligned with upstream + port mlx-lm's clean algorithm |

`dynamic_quant.py:38-106` is the canonical clean implementation:
- 60 lines of code
- Per-batch sequential loop: `for batch ... in tqdm(...): targets = model(batch); _, grads = nn.value_and_grad(...); grad_accum += grads; del grads; mx.eval(grad_accum)`
- Single persistent state: `grad_accum` (small — just gradient norms per layer)
- Per-batch activations freed after each iteration
- Total memory: model + 1 batch activations + grad_accum

hf2q's current `DwqCalibrator` instead:
- Builds a separate `RealActivationCapture(Qwen35Model)` with all-layer F32 host expansion (104 GB)
- Holds full activations during all batches
- Uses a custom sensitivity-from-activations metric (variance-magnitude) rather than gradient-based
- Has accreted complexity over 100+ iters with cache-priming hoists, explicit drops, etc.

## What this changes (revised iteration 4.6 — dual-track)

User directive (2026-05-06 22:55): **"we should support dwq and mixed precision; do both in ways that do not OOM, but cause best possible model outcome."**

Two independent, complementary algorithms — port BOTH:

### Track 1: Dynamic Quant (sensitivity-based mixed precision)
Source: `/opt/mlx-lm/mlx_lm/quant/dynamic_quant.py:38-106` (`estimate_sensitivities` + `estimate_threshold`).
- 60-line algorithm; gradient-based per-tensor sensitivity; binary-search bit threshold
- Output: mixed-bit GGUF (4-bit base, 5/6/8-bit on sensitive tensors per target BPW)
- Replaces hf2q's misnamed "DWQ-46/48" — same intent, mlx-lm's clean implementation
- Memory: model + 1 batch activations + grad_accum

### Track 2: DWQ proper (distillation fine-tuning)
Source: `/opt/mlx-lm/mlx_lm/quant/dwq.py:69-209` (`dwq_quantize`).
- 140-line algorithm; KL-div distillation; Adam optimizer fine-tunes scales+biases
- Output: improved-quality quantized weights (TRAINED scales packed into GGUF block format)
- New capability — hf2q didn't have this before
- Memory: same as Track 1 + Adam state per trainable param (~2× the trainable param size)

### Both tracks share infrastructure:
- Calibration data pipeline (tokenize → batch → seed)
- Per-batch sequential loop with `del grads` between iterations
- qdq utility (quantize + dequantize for sensitivity OR for student model)
- KL-div loss
- GGUF emission with per-tensor bit/scale config

### Cascade option (per mlx-lm LEARNED_QUANTS.md):
Run dynamic_quant first → produces a mixed-bit model → then run DWQ on top → fine-tunes the trained scales for that mixed-bit allocation. **Best possible quality.** Costs more wall time but no extra peak RAM.

### Phase 1 status (commit 437217d)
TensorRef.data Arc<Vec<u8>> refactor stays — memory hygiene improvement that benefits both tracks. Don't revert.

### A+B+D status
- **A (mmap-backed loader)**: still nice-to-have for serve-path efficiency; not blocking either track since mlx-lm-style per-batch processing doesn't need full-model-in-RAM. Defer to post-tracks.
- **B (zero-copy MlxBuffer)**: same — defer.
- **D (serve-path reuse)**: **OBSOLETE.** mlx-lm pattern doesn't build a separate model at all; activations stream through during forward+backward, free between batches. The whole concept of "reuse serve path" was patching the wrong layer.

**Driver**: User mission "/loop fully complete the research, and once research is complete, make working dwq quants (46 and 48) for the gemma4 and qwen3.6 families..."
**Predecessors**: ADR-014 P11 closure (DwqKQuantizer + cache), `project_qwen35_dwq_pre_505b5b8_broken_2026_05_05.md`, `project_hf2q_dwq_oom_root_cause_2026_05_06.md`

## Context

DWQ-46/48 conversion of Qwen3.6-27B (52 GB F16 source) on a 128 GB Mac M5 Max **kernel-panicked the box twice** on 2026-05-05/06 (samples=1024, samples=256, with and without `HF2Q_STREAMING_PHASE3_MUT=1`). Observed peak: 199 GB. Comparable workloads succeed elsewhere:

- **llama.cpp `llama-quant.cpp:1109-1245`**: per-tensor stream — load → dequant → quantize-in-chunks → write → reuse buffer. Peak ~5-10 GB regardless of model size.
- **Apple MLX `oq.py` + `dwq.py`**: lazy `mx.lazy` arrays via UMA, 8-bit teacher distillation. Peak ~1.2-1.5× source ≈ 60-80 GB.
- **hf2q (current)**: 3.8× source ≈ 200 GB (jetsam → kernel panic, unreliable).

## Hypothesis (verified by reading code, not comments)

Per mantra "Comments in code or ADR can be starting points, but never trust them over code" — the 199 GB peak was decomposed by direct source inspection.

**Driver A — `clone_tensor_map_to_lazy` deep clone (52 GB)**

`src/main.rs:482-496`:
```rust
fn clone_tensor_map_to_lazy(tensor_map: &TensorMap) -> LazyTensorMap {
    for (_, tensor) in tensor_map.iter() {
        out.insert(LazyTensor::from_bytes(meta, tensor.data.clone()));  // ← FULL Vec<u8>::clone deep copy
    }
}
```

`src/ir/mod.rs:31`: `pub data: Vec<u8>` — owned, not `Arc`. Every `.clone()` is a deep copy.

This adds 52 GB on cache MISS (the ONLY path the chain has ever taken since cache key derivation drifted from the 2026-04-30 working caches). The dev team was aware — `lazy.rs:215-218`: *"The full P13 win lands when `crate::ir::TensorRef::data` itself becomes `Arc<[u8]>` so the upstream tensor_map can hand `Arc::clone(&t.data)` to the wedge. This constructor is the API foothold for that future migration."* The migration was deferred.

**Driver B — Qwen35Model F32 expansion held all-at-once (~104 GB)**

`src/inference/models/qwen35/full_attn.rs::FullAttnLayerWeights` (and `delta_net.rs::DeltaNetLayerWeights`) hold `Vec<f32>` per weight (`attn_norm`, `wq`, `wk`, `wv`, `w_gate`, ...). All layers are accumulated into `Qwen35Model.layers: Vec<Qwen35LayerWeights>` at `src/inference/models/qwen35/weight_loader.rs:710-741`:

```rust
let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
for i in 0..cfg.num_hidden_layers {
    let layer = Qwen35LayerWeights::FullAttn { attn: load_lazy_full_attn_layer(...)?, ffn: ... };
    layers.push(layer);   // ← accumulates ALL layers' F32 expansions
}
```

`load_lazy_f32` at `weight_loader.rs:` calls `materialize_cloned()` (deep-copies the Vec<u8>) and then `tensor_ref_to_f32` (F16→F32 doubles size). For a 52 GB F16 source, the held F32 expansion is ~104 GB.

The accompanying doc comment claims *"each requested tensor is materialized, converted/uploaded, and then dropped before the next tensor is loaded"* — **that is not what the code does**. The intermediate Vec<u8> is dropped, but the resulting `Vec<f32>` is held in the layer struct. Layers accumulate. Mantra applies: code wins.

**Sum: 52 + 52 + 104 + scratch ≈ 200 GB peak** ✓ matches observation.

**Why 2026-04-30 succeeded on the same box**: cache HITs short-circuited Phase 2 entirely (per `p11_re_emit_dwq.sh:48-53` iter-95/97 comments — "cache HIT short-circuit cut peak from 158 GB → 52 GB"). The cache key derivation has since changed (`SensitivityCacheKey::with_algorithm_version` + `model_fingerprint` evolved through iter-96/103); existing `~/.cache/hf2q/sensitivity/*.json` files are debug stubs or stale-key payloads that no longer match. Subsequent runs are cold MISS → 200 GB → jetsam.

## Decision

Eliminate **both** drivers. Either alone is insufficient: removing only Driver A still leaves ~148 GB peak (over 128 GB box). Both must land for default-samples DWQ to fit.

**Phase 1 — Driver A elimination (Arc refactor)**
- Change `TensorRef.data: Vec<u8>` → `TensorRef.data: Arc<Vec<u8>>` in `src/ir/mod.rs`
- 9 write sites: wrap `tensor.data = bytes` → `tensor.data = Arc::new(bytes)`
- 267 read sites: most work via `Arc<Vec<u8>>: Deref<Target=Vec<u8>>` (`&tensor.data[..]`, `.len()`, `.as_slice()`); a few that take ownership need `Arc::try_unwrap()` or explicit `(**arc).clone()`
- Replace `clone_tensor_map_to_lazy`'s `tensor.data.clone()` (deep) with `Arc::clone(&tensor.data)` (pointer bump) — emits via existing `LazyTensor::from_arc_bytes`
- Falsifier test: assert `Arc::strong_count(&tensor.data) >= 2` after `clone_tensor_map_to_lazy`, proving no byte clone

Estimated 200-400 LOC (mostly mechanical) + 2-3 unit tests. Compilation-driven: change the type, fix breakage as the compiler reports.

**Phase 2 — Driver B elimination (streaming calibrator)**
- Refactor `DwqCalibrator` to compute sensitivity per-layer: build ONE layer's `Qwen35LayerWeights`, run forward pass on calibration corpus for that layer's sensitivity, drop the layer's F32 expansion before building the next.
- Mirror llama.cpp's per-tensor callback pattern at `imatrix.cpp:229 IMatrixCollector::collect_imatrix` (per-call activation accumulation) but adapted to MLX's tensor-graph semantics in mlx-native.
- Internal API change: `Qwen35Model::load_from_lazy_tensor_map` becomes a layer iterator instead of returning a fully-loaded model. Calibrator pulls layers on demand.

Estimated 1-2 days of focused refactor. Touches `src/calibrate/dwq_calibrator.rs` + `src/inference/models/qwen35/weight_loader.rs` + `src/inference/models/qwen35/activation_capture_real.rs`.

**Expected post-fix peak**: tensor_map (52 GB, mmap-backed) + 1 layer F32 (~2 GB) + scratch (~5 GB) = **~60 GB**. Fits comfortably in 128 GB box, samples=1024 default, real DWQ — no quality compromise.

## Acceptance criteria

1. Falsifier unit test for Phase 1 PASSES (Arc strong_count proves no byte clone).
2. `tests/convert_qwen35_*` regression suite stays green (101/101 from `0357394` baseline).
3. Bare DWQ-46 27B convert with `ulimit -v 100GB` safety net completes — process never exceeds 100 GB virtual; `/usr/bin/time -l` reports max RSS < 80 GB.
4. Output GGUF passes `gguf-dump` invariants from `0357394`: `tokens.len()==248320`, `eos_token_id=248046`.
5. Coherence check via hf2q runtime: model loads, 16-token generation produces non-degenerate text.
6. Same five criteria for Qwen3.6 35B-A3B-Abliterix-EGA-abliterated and Gemma 4 26B-A4B-it-ara-abliterated families, dwq-4-6 + dwq-4-8.

## Alternatives considered

- **Run on bigger box for cache priming**: requires external infrastructure (cloud Mac 192-256 GB, or Linux GPU). One-off cost per model + cache copy. Rejected as default — mantra "no fallback, no stub" + user wants the code FIXED long-term.
- **Lower `--calibration-samples` to 256**: empirically still kernel-panics (init load hits 100+ GB before samples matter). And quality compromise. Rejected.
- **Substitute K-quants (`q5_k_m`/`q6_k`)**: user explicit "no cheating, real DWQ".
- **`HF2Q_STREAMING_PHASE3_MUT=1` flag alone**: only addresses Phase 3 dispatch; the OOM is in Phase 2. Verified empirically: same 199 GB peak.

## Risk + mitigation

- **Refactor breakage in Phase 1**: 267 read sites. Mitigation: compilation-driven, run full test suite per commit, roll back if regression.
- **Streaming calibrator alters numerical output**: bit-by-bit reproducibility may shift if F32 expansion ordering changes. Mitigation: golden-output test — record sensitivity vector for a 2-layer synthetic model BEFORE refactor, assert identical AFTER.
- **`Arc::try_unwrap` failures at write sites if external Arc clones leak**: handled by `Arc::unwrap_or_clone` — explicit one-time clone fallback if refcount > 1.
- **Multi-day scope**: tracked across `/loop` iterations. Each iteration commits + pushes incremental progress. User can interrupt anytime.

## Implementation iteration plan

**REVISION 2026-05-06 iteration 4 → 5:** dropped naive Phase 2 (drop-CPU-F32-after-upload).
Even with that fix, peak ~158 GB (vs 128 GB box) — insufficient. User question
"what is llama.cpp doing differently" surfaced the real architectural gap, captured
in `Phase A+B+D` plan below. Commits `72fdee8` + `437217d` + `b3db220` retained;
they are still load-bearing (Phase 1 + the deep-dive that informed this pivot).

| Iter | Scope | Status |
|---|---|---|
| 2 | Research synthesis, ADR, memory updates, commit `72fdee8` | DONE |
| 3 | Phase 1: TensorRef.data Arc refactor, falsifier test, regression check (104/104 PASS), commit `437217d` pushed | DONE |
| 4 | Phase 2 deep-dive: trace GPU upload + cache lifecycle, verify MlxBuffer COPIES, commit `b3db220` | DONE |
| 4.5 (this) | "What does llama.cpp do?" deep-dive surfaces architectural gap; pivot to A+B+D | DONE |
| 5 | Phase A: mmap-backed safetensors load (replace `mmap[..].to_vec()` at safetensors.rs:437) | NEXT |
| 6 | Phase B: `MlxBuffer::from_no_copy` in /opt/mlx-native using metal-rs `new_buffer_with_bytes_no_copy` (cross-repo) | |
| 7 | Phase D: reuse serve path for DwqCalibrator (eliminate separate Qwen35Model build) | |
| 8 | E2E DWQ-46 27B with watchdog; expected peak ≤ 10 GB matching llama.cpp profile | |
| 9 | Roll forward to all 4 family×bit combinations + Gemma 4 26B-A4B | |
| 10 | Benchmark vs llama.cpp imatrix peer reference; ADR closure | |

## Why naive Phase 2 alone is insufficient — architectural finding

Iteration 4.5 verified llama.cpp's load architecture (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1465`):
```objc
res->buffers[0].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:res->all_data
                                                                length:size_aligned
                                                               options:MTLResourceStorageModeShared
                                                            deallocator:nil];
```
Metal's `newBufferWithBytesNoCopy` **aliases an existing pointer** — for llama.cpp,
that pointer is the mmap'd GGUF file. CPU and GPU see the SAME memory. **Zero bytes copied.**

vs hf2q's mlx-native at `/opt/mlx-native/src/device.rs:143`: only has `device.new_buffer(...)`
which always allocates fresh; uploads via `slice.copy_from_slice(data)` — full byte copy.

Combined with hf2q's **deep-copy-from-mmap** at `src/input/safetensors.rs:437`
(`mmap[..].to_vec()`) and host-F32 expansion in `weight_loader.rs::load_lazy_f32`, the
3 multipliers produce the 199 GB peak:

| Step | hf2q (current) | llama.cpp |
|---|---|---|
| Load weights from disk | deep-copy mmap → owned Vec<u8> (~52 GB) | mmap pointer kept (~0 GB resident) |
| Build calibration model | F16/BF16 → F32 host expansion (~104 GB) | reuse same model handle (~0 GB extra) |
| Upload to GPU | `alloc_buffer + copy_from_slice` (+~104 GB) | `newBufferWithBytesNoCopy` (~0 GB extra, alias) |
| **Total peak** | **~260 GB** (with scratch ≈ 199 observed) | **~10 GB**  |

The **mantra-correct fix** is architectural alignment, not patching the symptom. Phases
A+B+D (below) bring hf2q's profile in line with llama.cpp.

## Phase A: mmap-backed safetensors loader

Replace `src/input/safetensors.rs:437` `mmap_clone[abs_start..abs_end].to_vec()` with
a pattern that keeps the source Mmap alive and exposes tensor bytes as a slice into it.

Two viable shapes:
1. **`TensorRef.data: Arc<MmapView>`** where `MmapView { mmap: Arc<Mmap>, offset: usize, len: usize }`. Deref to `&[u8]`. Eliminates the owned Vec entirely.
2. **Keep `Arc<Vec<u8>>` but defer materialization**: the Arc holds a closure that reads from mmap on first access. Worse — still materializes when accessed.

Going with shape (1). Touches:
- `src/input/safetensors.rs::read_tensors_eager` and `read_tensors_lazy` (already mmap-backed; the to_vec was the eager bridge)
- `src/ir/mod.rs::TensorRef.data` field type
- `src/ir/mod.rs::TensorRef::take_data_as_arc` semantics (was `Arc::new(mem::take(...))`; needs MmapView equivalent)
- All existing read sites (267) — most work via Deref

Falsifier test: load Qwen3.6-27B safetensors via the new path, assert `mach_task_basic_info` RSS stays under 5 GB (mmap-paged, demand-fault working set only). Compare to baseline before-A which would be ~52 GB.

## Phase B: MlxBuffer::from_no_copy in /opt/mlx-native (cross-repo)

mlx-native already depends on `metal = "0.33"` which exposes `Device::new_buffer_with_bytes_no_copy`.
Add to `/opt/mlx-native/src/device.rs`:
```rust
/// Wrap an existing pointer as a StorageModeShared Metal buffer
/// without copying bytes. Keeps `holder` alive for the buffer's lifetime
/// via a deallocator closure — so dropping the MlxBuffer drops the holder
/// at the right moment. Mirrors ggml-metal's `newBufferWithBytesNoCopy`
/// pattern at ggml-metal-device.m:1465 (zero-copy mmap → GPU alias).
pub fn alloc_buffer_no_copy(
    &self,
    holder: Arc<dyn AsRef<[u8]> + Send + Sync>,
    dtype: DType,
    shape: Vec<usize>,
) -> Result<MlxBuffer> { ... }
```

Falsifier: build a 100 MB Arc<Vec<u8>>, call `alloc_buffer_no_copy`, assert
`metal_buf.contents() as usize == arc.as_ref().as_ptr() as usize` — same memory.

Cross-repo: commit + push to /opt/mlx-native, bump version, update hf2q Cargo.toml dep.

## Phase D: reuse serve path for DwqCalibrator

`src/calibrate/dwq_calibrator.rs:333-345` builds `RealActivationCapture::from_lazy_tensor_map`
which calls `Qwen35Model::load_from_lazy_tensor_map` — the all-layer-F32 expansion path.

After A+B land, the same `Qwen35Model` build path will be cheap (mmap-aliased GPU buffers,
no F32 host expansion needed if we ALSO migrate `load_lazy_f32` to skip the F32 cast and
use type-specialized GPU kernels for F16/BF16 weights).

Phase D investigates whether the existing `cmd_serve` load path is already memory-efficient
enough to be tapped for activation capture, OR if Phase D needs to add a new
`Qwen35Model::load_for_capture(Arc<Mmap>)` constructor that avoids the host F32 expansion.

Falsifier: bare `hf2q convert --quant dwq-4-6 --input <Qwen3.6-27B>` with watchdog (poll RSS
via mach_task_basic_info, SIGTERM if > 20 GB threshold) → process completes successfully,
max RSS observed ≤ 20 GB. Compares to pre-fix 199 GB.

### Why no naive Phase 2 (drop CPU F32 after upload)

Even if we hoisted the GPU upload into `Qwen35Model::load_from_lazy_tensor_map` and dropped
each layer's host F32 immediately after upload (the original Phase 2 plan), peak briefly
hits both source bytes + uploaded GPU buffer + transient F32 expansion = ~158 GB. Still
over 128 GB. The real fix has to eliminate the F32 expansion AND the upload copy — that's
what A+B+D do together.

## Phase 2 implementation plan (verified iteration 4)

### What's actually held in memory
After `Qwen35Model::load_from_lazy_tensor_map` completes:
- `Qwen35Model.layers: Vec<Qwen35LayerWeights>` — each with multiple `Vec<f32>` (attn_norm, wq, wk, wv, etc.). For 27B dense: ~104 GB total CPU F32.
- `Qwen35Model.token_embd: Vec<f32>`, `output_weight: Vec<f32>`, `output_norm: Vec<f32>` — additional CPU F32.

After `forward_gpu_impl` first call (cache prime):
- Thread-local `cell` cache holds `Vec<LayerWeightsGpu>` + `OutputHeadGpu` — ALL uploaded via `upload_f32(data, device)` at `gpu_full_attn.rs:258-270`.
- `upload_f32` calls `device.alloc_buffer(byte_len, ...)` → fresh StorageModeShared `MlxBuffer` → `slice.copy_from_slice(data)` — **copies bytes**, not aliases.
- After cache prime: BOTH the host `Vec<f32>` AND the GPU `MlxBuffer` hold ~104 GB each. Total: ~208 GB just for layers.

### Precise fix
Drop the host `Vec<f32>` fields **after** the GPU cache primes (one-time, per `forward_gpu_impl:1869-1900`). This is safe because:
- `MlxBuffer` owns its own StorageModeShared allocation (not aliased to the Vec<f32>); dropping the source Vec doesn't affect the GPU buffer.
- Cache is keyed on `model_ptr` so subsequent calls with the same `&Qwen35Model` reuse the cache without needing the host F32.

### Implementation shape
The clean fix is to hoist the upload into `Qwen35Model::load_from_lazy_tensor_map` itself (consume CPU F32 → produce GPU bundle → drop CPU). Concrete API:
```rust
pub struct Qwen35Model {
    pub cfg: Qwen35Config,
    pub gpu_layers: Vec<LayerWeightsGpu>,    // pre-uploaded
    pub gpu_output_head: OutputHeadGpu,       // pre-uploaded
    pub gpu_token_embd: MlxBuffer,            // pre-uploaded
    // CPU `layers: Vec<Qwen35LayerWeights>` REMOVED — replaced by `gpu_layers`
    // CPU `token_embd: Vec<f32>` REMOVED
    // CPU `output_weight: Vec<f32>` REMOVED
    // CPU `output_norm: Vec<f32>` REMOVED
    pub mtp: Option<...>,
}
```
Then `forward_gpu_impl` reads from `self.gpu_layers` directly — no upload, no cache rebuild. The constructor (`load_from_lazy_tensor_map`) becomes:
1. Build CPU `Qwen35LayerWeights` structures with `Vec<f32>` (peak: 104 GB CPU + 52 GB tensor_map = 156 GB transient).
2. Upload each layer to GPU via existing `LayerWeightsGpu::from_cpu(&w, &device)` paths (peak briefly +104 GB during upload = 260 GB).
3. **Immediately drop the CPU layer data** before next layer's upload — `mem::replace(&mut layer.attn.wq, Vec::new())` etc. — so peak only holds 1 layer's F32 host + 1 layer's GPU + cumulative-uploaded GPU.
4. After loop: only `gpu_layers` resident (~104 GB GPU), CPU F32 fully dropped.
5. Final peak: ~52 GB tensor_map (Arc-shared post Phase 1) + ~104 GB GPU layers + scratch ≈ 160 GB.

That's still over 128 GB. To fit, **also stream the CPU→GPU per-tensor inside layer construction**: load one F32 tensor → upload → drop, instead of building the whole layer's Vec<f32>s before uploading any of them. That keeps peak per layer at ~1-2 GB CPU F32 + cumulative GPU.

Final expected peak: 52 GB tensor_map + 104 GB GPU bundle + 2 GB transient scratch = **~158 GB**. Hmm — still over 128 GB by ~30 GB.

### Phase 2 may not be enough — ADR-020 amendment

Even with both Phase 1 (52 GB clone removed) AND Phase 2 (host F32 freed), peak ≈ 158 GB on Qwen3.6-27B due to the GPU-resident weights themselves consuming 104 GB. To fit in 128 GB:
- **Phase 3 (open question)**: do GPU weights need to be F32, or can they be F16? F16 GPU = 52 GB → total peak ~106 GB → fits.
- Look at how `forward_gpu_impl` uses the F32 weights — most ML inference does FP16/BF16 attention math; F32 host might just be the "expand for safety" that gets cast back at GPU op time.
- The `upload_q4_0_from_f32` for output weight already shows we can push weights through narrower formats. Same for layer weights?

**Phase 3 may be required** — keep weights in F16 on GPU. ~24 hours additional refactor. Or accept Phase 2-only result + run with watchdog at 130 GB threshold (slightly above box capacity, swap absorbs slack).

### MoE-specific note (35B-A3B)
For Qwen3.6-35B-MoE the `MoeQ` path is already quantized (`load_lazy_moe_ffn_quantized`) — experts go straight to GPU q4 buffers, NO F32 expansion. So the 35B case is much smaller: ~10-15 GB CPU F32 (just attention + shared experts + small tensors). 35B-MoE may already fit post-Phase 1; Phase 2 may suffice for 35B but be insufficient for 27B-dense.

### Test strategy
- Synthetic Qwen 4-layer dense fixture with measurable weights (~100 MB scale).
- Falsifier: assert RSS delta from before-load to after-load is ~ source size + ~scratch, NOT 2× source. Use `mach_task_basic_info` for RSS measurement on macOS (no `RLIMIT_AS` enforcement available — must measure post-hoc rather than gate via ulimit).
- Production verification: `/usr/bin/time -l` on bare DWQ-46 27B run (after watchdog scaffold).

## References

- Code: `src/main.rs:482-496`, `src/ir/mod.rs:25-50`, `src/ir/lazy.rs:130-225`, `src/inference/models/qwen35/weight_loader.rs:700-750`, `src/inference/models/qwen35/full_attn.rs::FullAttnLayerWeights`
- Reference impls: `/opt/llama.cpp/src/llama-quant.cpp:1109-1245`, `/opt/llama.cpp/tools/imatrix/imatrix.cpp:229`, `/opt/omlx/omlx/oq.py`
- Memory: `project_hf2q_dwq_oom_root_cause_2026_05_06.md`, `project_qwen35_reconvert_paused_2026_05_05.md`, `project_qwen35_dwq_pre_505b5b8_broken_2026_05_05.md`
- Mantra: ~/Documents/mantra.txt — "Measure 3x, cut once. No fallback. No stub. Pure excellence."
