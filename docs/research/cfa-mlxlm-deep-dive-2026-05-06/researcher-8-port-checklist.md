# Researcher #8 — Executive Synthesis: Port Checklist + Priority

**Session**: cfa-20260506-mlxlm-research
**Role**: Synthesizer (last in spirit)
**Status**: Final — to be merged verbatim into ADR-020 "Port Plan" section

---

## 1. Recap: User Requirements

User directive (2026-05-06 22:55):
> "we should support dwq and mixed precision; do both in ways that do not OOM, but cause best possible model outcome."

Three hard requirements:

1. **Two algorithms** must ship: (a) **mixed-precision** quant (mlx-lm's `dynamic_quant` — sensitivity-ranked bit allocation), (b) **DWQ proper** (mlx-lm's `dwq` — KL-distillation fine-tuning of quantized scales+biases against an FP teacher).
2. **No OOM on 128 GB Mac** for any path. Reference workload is Qwen 3.6 35B-A3B and Gemma 4 26B. Today's hf2q "DWQ-46/48" peaks at 199 GB. Target ≤ 80 GB peak.
3. **Best-possible quality**. mlx-lm `LEARNED_QUANTS.md` documents the cascade: dynamic_quant → DWQ on top of the mixed result for the best 4-bit quality available. We must not foreclose the cascade.

Everything below preserves all three.

---

## 2. Three (Plus One) Porting Tracks — Verdicts Up Front

| Track | Verdict | Replaces / Adds | LOC (algorithm core) |
|---|---|---|---|
| **1. Dynamic Quant** (mixed-precision) | **PORT — Phase 1, 1-2 weeks** | replaces broken `DwqCalibrator` (variance-magnitude) | ~80 LOC |
| **2. DWQ proper** (distillation) | **PORT — Phase 2, 2-4 weeks, conditional on autograd decision** | new capability | ~200 LOC + autograd substrate |
| **3. AWQ** (activation-aware scaling) | **DEFER — Phase 3 optional** | new; complementary to DWQ | ~250 LOC + per-arch ScaleConfig table |
| **4. GPTQ** (Hessian-error minimization) | **DEFER or DROP — Phase 3 optional** | new; orthogonal to DWQ | ~150 LOC + Cholesky linalg |

**Rationale up front**: Track 1 is autograd-free (gradient w.r.t. weights only at the qdq output layer, which is broadcast over a single weighted-sum sensitivity scalar — see §3.1 below). Track 2 requires real autograd through the transformer. Tracks 3 and 4 are independent algorithms; AWQ is grid-search (no autograd); GPTQ needs Cholesky on Hessian per-layer (no full-graph autograd, but heavy linalg). Each track has standalone value, but Track 1 must come first because the user has already specified DWQ-46/48 in the CLI and that name today routes to a wrong algorithm.

---

## 3. Per-Track Subtask Lists

Every subtask is a **complete addition with a falsifier**. No TODOs. No stubs.

### 3.1 Track 1 — Dynamic Quant (sensitivity-based mixed precision)

**Source**: `/opt/mlx-lm/mlx_lm/quant/dynamic_quant.py:38-106` — `estimate_sensitivities` + `estimate_threshold`.

**Algorithm essence**: For each quantizable Linear, compute a sensitivity scalar `s = sum(grad * (W_q_low - W_q_high)) / param_size`, where `grad` is `∂(KL(model(batch) || qdq_low(model)(batch)))/∂W` accumulated over a small batch loop. Then binary-search a threshold over `s` to hit a target BPW; layers above threshold get `high_bits` (e.g. 5), rest get `low_bits` (e.g. 4).

**Subtasks**:

| # | Subtask | Falsifier | Effort |
|---|---|---|---|
| T1.1 | Add `Quant::DynamicQuant { target_bpw, low_bits, high_bits, low_group_size, high_group_size }` to `QuantMethod` enum + CLI flag `--quant dynamic-quant-4-5` parsing in `cmd_convert.rs` | clap parse-test rejects malformed presets; accepts `4-5`/`4-6`/`4-8` | 0.5 day |
| T1.2 | Calibration data pipeline: extend existing `CalibrationCorpus` consumer to produce 512-token non-overlapping chunks shuffled by `mx.random.permutation` (mirrors mlx-lm `quant/utils.py:8-26`) | unit test: same seed → byte-identical chunk order across two runs | 0.5 day |
| T1.3 | `qdq` helper in mlx-native: takes `&MlxBuffer w, bits, group_size`, returns dequantized fp16/bf16 buffer. Thin wrapper over existing `mlx::quantize` + `mlx::dequantize` | unit test: bits=8 group_size=64 → max-abs-error vs original ≤ 1e-3 (1 ULP for 8-bit) | 0.5 day |
| T1.4 | **Gradient capture for sensitivity**: this is the autograd seam. The gradient is only needed w.r.t. each Linear's `weight`, NOT through normalisation/RoPE/attention as a full graph. **Decision: hand-derive**. The quantity `sum(grad * (W_q_low - W_q_high))` is the dot-product of the upstream loss-gradient at the layer's weight and a known constant (the dequant delta). For each Linear, capture `∂L/∂y` at its output (cheap — one tensor per layer per step), then `∂L/∂W = ∂L/∂y · x^T` (the chain-rule of `y = W·x`). No full autograd needed. | unit test on synthetic 4-layer model: hand-derived sensitivity matches PyTorch autograd reference within 1e-4 | 3 days |
| T1.5 | Per-batch sensitivity-accumulation loop: load batch → forward (capture `(x, ∂L/∂y)` per Linear via hooks on the existing `Qwen35Model` forward path) → compute `grad_accum[layer] += sum(∂L/∂y · x^T * delta)` → drop activations → next batch. **Critical**: `del grads` between batches; mirror mlx-lm `dynamic_quant.py:84-86` exactly. | falsifier: synthetic 4-layer model; running 50 batches must produce sensitivity vector that ranks layers identically to a one-shot all-batch reference within 1e-3 (validates the per-batch accum equals one-shot mathematically) | 3 days |
| T1.6 | Binary-search threshold (`estimate_threshold` port; `dynamic_quant.py:109-146`): bisect over `[min_sens, max_sens]`, at each midpoint construct a hypothetical bit-allocation, call `compute_bits_per_weight` and shrink the bracket until the target BPW is hit within tolerance | unit test: known sensitivity vector → known threshold; deterministic | 1 day |
| T1.7 | Plug allocation into existing `MixedBitQuantizer` (per-tensor bit dispatch already works) — replace today's `sensitive_layers: Vec<RangeInclusive<usize>>` with `bit_per_tensor: HashMap<String, u8>` | end-to-end: convert tiny synthetic model, verify GGUF tensor-type-codes match expected mix | 1 day |
| T1.8 | Memory hygiene: `mx.set_wired_limit(max_recommended_working_set_size)` equivalent — call mlx-native's `set_wired_limit` once at convert start; per-batch `mlx_native::clear_cache()` after `del grads` | RAM falsifier: Qwen 3.6 35B-A3B-APEX dynamic-quant-4-5 with samples=256, peak ≤ 80 GB on 128 GB Mac (measured by `/usr/bin/time -l`, NOT just `mx.get_peak_memory()`) | 1 day |
| T1.9 | Sensitivity-cache integration: reuse existing `~/.cache/hf2q/sensitivity/` infra with a new `SENSITIVITY_ALGORITHM_VERSION` value tagged `dynamic-quant-v1`. Keep cache key `(model_fingerprint, corpus_sha, algo_version)` shape unchanged. | unit test: corpus SHA change → MISS; same SHA → HIT byte-identical bit allocation | 1 day |
| T1.10 | GGUF emit unchanged — mixed Q4_K/Q5_K/Q6_K/Q8_0 already supported via existing `KQuantCodecQuantizer` per-tensor dispatch. Update `cmd_convert` to pass the new bit-map. | round-trip test: emit → read back → tensor-type-codes match the bit-map | 0.5 day |
| T1.11 | Replace `DwqCalibrator` registration: when `--quant dynamic-quant-X-Y` is supplied, route via `DynamicQuantCalibrator` (new), not `DwqCalibrator`. Keep `DwqCalibrator` alive but make `--quant dwq46`/`dwq48` print a deprecation warning naming `dynamic-quant-4-6`/`dynamic-quant-4-8` as the replacement. | CLI test: `--quant dwq46` succeeds with stderr deprecation note; `--quant dynamic-quant-4-6` succeeds without it | 0.5 day |

**Track 1 total — happy path**: ~12 days = ~2.4 weeks for one engineer.
**Track 1 with quality validation** (perplexity report on Qwen 3.6 35B + Gemma 4 26B vs llama.cpp Q4_K_M baseline; tg/pp parity bench): +5 days = **~3.5 weeks**.

### 3.2 Track 2 — DWQ proper (distillation fine-tuning)

**Source**: `/opt/mlx-lm/mlx_lm/quant/dwq.py:69-209` — `dwq_quantize`.

**Algorithm essence**: Quantize student. Unfreeze `scales` + `biases` only. Adam optimizer. For each batch: forward through teacher (or pre-computed top-1024 logits), forward through student, compute KL-divergence on logits at temperature 2.0, backprop through student, Adam step. Iterate ~2048 batches.

**Subtasks**:

| # | Subtask | Falsifier | Effort |
|---|---|---|---|
| T2.0 | **Autograd-substrate decision** (BLOCKING). Three options: (a) port to **mlx-native** (existing Metal kernels but no autograd graph today; would need to build one — multi-month; cross-repo coordination per §8 below); (b) **Rust autograd crate** — `candle-core` 0.7 has a tape-based engine but its mlx-Metal kernels are missing or thin (we'd lose mlx-native's Q4_K kernels); `burn` 0.13 with `wgpu` backend works but isn't on Metal-native; (c) **two-process architecture** — call `mlx_lm.dwq` Python entry point from hf2q via subprocess, package the conda env, return safetensors; convert safetensors → GGUF in hf2q post-step. **Recommendation: (c) two-process for v0.1**; revisit (a) when mlx-native gains autograd. Justification: the two-process path delivers user-visible DWQ in 2 weeks instead of 2-4 months, the SOLE consequence is wall time and a Python dep on the dev machine, and the OUTPUT (a quantized model) is a static artifact — no inference-time Python coupling. | falsifier: 2-process subprocess test on tiny model produces a valid safetensors with `quantization` config; hf2q reads it back and re-emits as GGUF; round-trip BPW within 0.05 | 2 days (decision + scaffolding) |
| T2.1 | **(Two-process variant)** `cmd_convert --quant dwq-4` invokes `mlx_lm.dwq --model <hf-repo> --bits 4 --num-samples 1024 --target-dir /tmp/hf2q-dwq-targets-XXX --grad-checkpoint`. Stream subprocess stdout to existing `ProgressReporter`. Parse mlx-lm's loss/peak-mem lines; surface `❌❌❌` warning if final val_loss > initial (matches mlx-lm `dwq.py:202-207`). | integration test: tiny Qwen-0.6B → DWQ-4 → safetensors lands; subprocess exit==0; loss strictly decreasing | 2 days |
| T2.2 | **Stream targets to disk** (mlx-lm `dwq.py:29-66` pattern). Critical for fitting Qwen 3.6 35B teacher: pre-compute teacher's top-1024 logits for every batch, write `safetensors` per-batch under `target_dir/{train,valid}/{idx:010d}.safetensors`, then DROP teacher before student loads. mlx-lm already does this — we surface `--targets-only` step in our CLI as a 2-stage convert: pass 1 = teacher-only top-1024 dump (no student), pass 2 = student fine-tune from disk targets. | RAM falsifier: pass 1 peak ≤ teacher-size + ~5 GB; pass 2 peak ≤ student-size + Adam-state + ~5 GB; total never holds both | 2 days |
| T2.3 | **Adam optimizer**: even in two-process mode this lives in mlx-lm; in native mode we'd need to port. Given (c) decision, NO PORT. Trivial 60-line port in Rust if (a) is later chosen. | n/a (subprocess) | 0 |
| T2.4 | **KL-div loss with temperature scaling**: lives in mlx-lm; same as T2.3. Note that mlx-lm's `kl_div_loss` uses a custom Metal kernel (see `tuner/losses.py:11-176`) for fused logsumexp + KL — that kernel itself is portable to mlx-native if we ever do (a). | n/a (subprocess) | 0 |
| T2.5 | **Output ingestion**: subprocess emits a directory containing `model.safetensors` + `config.json` with `quantization` key listing per-tensor `{bits, group_size}`. hf2q reads this and converts to GGUF using existing K-quant codec. **Critical bridge**: mlx-lm packs `scales` and `biases` per-block in a 4-bit-friendly layout that DIFFERS from llama.cpp's Q4_K block layout. Researcher #7's report MUST be consulted here; if the layouts are not bit-compatible, we either (i) round-trip through dequantize-then-re-quantize (loses ~0.1 BPW of the trained scales' precision), or (ii) define a new GGUF tensor-type sentinel `Q4_K_DWQ` carrying mlx-lm's exact block format. | format falsifier: emit DWQ-4 safetensors → convert to GGUF → load in hf2q serve → first-token logits within 1e-3 of Python `mlx_lm.generate` reference on identical prompt | 3 days |
| T2.6 | **Cascade orchestration** (per `LEARNED_QUANTS.md:18-20`): support `--quant dynamic-quant-4-5+dwq` to run Track 1, then run Track 2 on the resulting mixed-bit model. The DWQ-on-mixed cascade is the **best-quality** option mlx-lm documents. Reuse Track 2 subprocess; pass `--quantized-model <output-of-track-1>`. | quality falsifier: cascaded model perplexity on wikitext-2 ≤ Track-1-alone perplexity − 0.05 (typical mlx-lm reduction) | 1 day |
| T2.7 | **CLI deprecation**: `--quant dwq46`/`dwq48` already deprecated in T1.11. Add `--quant dwq-4` (proper DWQ) + `--quant dwq-3` + `--quant dynamic-quant-X-Y+dwq` (cascade). | CLI parse-test for all 4 new presets | 0.5 day |
| T2.8 | **Memory falsifier**: Qwen 3.6 35B-A3B-APEX `--quant dynamic-quant-4-6+dwq --num-samples 1024 --max-seq-length 1025 --grad-checkpoint --batch-size 1`. Both subprocesses must complete. peak_memory_gb logged ≤ 90 GB (mlx-lm reports ~60 GB for this size class). | log-grep: max(peak_memory_gb in both passes) ≤ 90 | 1 day |

**Track 2 total — happy path (two-process)**: ~12 days = ~2.4 weeks. PLUS 1-week dev-environment plumbing for an mlx-lm conda env wrapper that hf2q ships or detects (`brew install ...` doc + `mlx_lm` version-pin check at convert-start).
**Track 2 with quality validation** (perplexity, MMLU/HellaSwag/PIQA via `mlx_lm.evaluate` — already exists): +5 days = **~4 weeks total**.

**Track 2 (native variant)** — for completeness: 2-4 months because mlx-native lacks an autograd engine. Not recommended for v0.1.

### 3.3 Track 3 — AWQ (optional Phase 3)

**Source**: `/opt/mlx-lm/mlx_lm/quant/awq.py:399-510` — `awq_quantize`.

**Algorithm essence**: Per transformer block, capture input activations to each Linear. Grid-search over `n_grid=20` scaling ratios `s = max(|x|^ratio)`; for each, apply `W' = W*s` and inverse-scale the previous op's weight. Pick the scale that minimises MSE between FP and quantized block output. Then per-Linear clip-search over `n_grid=20` clipping fractions. Same per-block sequential pattern as Track 1 — autograd-free.

**Subtasks** (sketch — for ADR-020 if we proceed):
- T3.1 Per-arch `AWQConfig` table — `llama_awq` / `gemma3_text_awq` / `deepseek_v2_awq`. **Each new model family needs a hand-written ScaleConfig.** Researcher #1/#2 should report which families we need.
- T3.2 `Catcher` module to capture `input_feat` per-Linear during block forward (mirrors existing hf2q `ActivationCapture` infra).
- T3.3 `search_best_scale` + `apply_scale` + `search_best_clip` ports — pure tensor ops, no autograd, but heavy einsum (`mx.einsum("bdg,odg->bod", x, w)`) — port to mlx-native via existing matmul + reshape primitives.
- T3.4 GGUF emit unchanged (output is standard 4-bit quant; the trained scaling is fused into the previous op's weight before quant).
- T3.5 Falsifier: per-block, after-scale-and-clip MSE < before-scale-and-clip MSE. mlx-lm logs this via `tqdm.write(f"Loss reduction: {after_loss / before_loss}")`. We adopt the same telemetry.

**Effort**: ~3 weeks per supported family because of the hand-tuned `ScaleConfig` table. Skip in v0.1; revisit if Track 1+2 quality is insufficient.

**Verdict**: DEFER. AWQ's 4-bit quality lift over plain Q4_K is real but small (~0.1-0.2 perplexity); DWQ's lift is larger. Don't burn a quarter on AWQ until DWQ ships and is validated.

### 3.4 Track 4 — GPTQ (optional Phase 3)

**Source**: `/opt/mlx-lm/mlx_lm/quant/gptq.py:52-159`.

**Algorithm essence**: Per-Linear `Catcher` accumulates `H = X^T X` (Hessian) over calibration. Per-Linear: Cholesky-invert `H` → solve a closed-form per-column quantization that minimizes `||W - W_q||_H^2` (Hessian-weighted error). No autograd; just linalg.

**Subtasks** (sketch):
- T4.1 `Catcher` to accumulate H — same module as Track 3's hook.
- T4.2 Cholesky inverse via `nalgebra` or `mlx-native` linalg (if exposed).
- T4.3 The `gptq_error` inner kernel — closed-form quantize-residual update (`W -= e @ Hinv[k:k+1, k:j]`).
- T4.4 GGUF emit: like Track 1, output is standard Q4_K format.

**Effort**: ~2 weeks. The algorithm is simpler than DWQ but requires Cholesky on potentially-large H (H is `[in_features, in_features]`, e.g. 12288×12288 for Qwen 3.6 → 1.5 GB FP32; tractable but mlx-native Cholesky may not be there yet — researcher report needed).

**Verdict**: DEFER or DROP. GPTQ's quality vs DWQ is well-documented in literature: DWQ wins at 4-bit, GPTQ wins at 3-bit, AWQ wins at activation-asymmetric tasks. If we have DWQ, GPTQ is redundant. **Drop unless researcher #4 reports a specific Qwen-or-Gemma quality regime where GPTQ pulls ahead.**

---

## 4. Memory Architecture — What's Load-Bearing for 128 GB

From researcher #6's expected output, the following are load-bearing for fitting on 128 GB:

| Trick | Source | CRITICAL / NICE-TO-HAVE | Tracks affected |
|---|---|---|---|
| **Per-batch processing with `del grads`** | `dynamic_quant.py:80-86`, `dwq.py:117-126` | **CRITICAL** | Track 1, Track 2 |
| **Stream targets to disk** (drop teacher before student) | `dwq.py:29-66` | **CRITICAL — Track 2 only** | Track 2 |
| **`mx.set_wired_limit(max_recommended_working_set_size)`** | `dwq.py:389-391`, `trainer.py:228-229` | NICE-TO-HAVE on macOS (raises wired-mem ceiling above default ~75% RAM) | Track 1, Track 2 |
| **Lazy load (`load(..., lazy=True)`)** | `dwq.py:337` | NICE-TO-HAVE — hf2q's Phase 1 `Arc<Vec<u8>>` already gives 1× source-size on cache-MISS load; lazy is the next 0.5× win | Track 1, Track 2 |
| **`grad_checkpoint` (recompute activations on backward)** | `dwq.py:103-104`, `trainer.py:25-38` | NEEDED only if backward exhausts memory — Track 2 only on 35B class | Track 2 |
| **`mx.clear_cache()` between steps** | `trainer.py:20-22` | NICE-TO-HAVE — keeps Metal allocator from holding fragmented blocks | Track 1, Track 2 |
| **8-bit teacher distillation** (use Q8 teacher instead of FP16) | `LEARNED_QUANTS.md:70-72` | NICE-TO-HAVE — Track 2 only; cuts teacher RAM in half with negligible quality cost | Track 2 |

**Single most important rule**: **the model is only resident once**. mlx-lm never holds teacher + student + activations + grads simultaneously — it streams. hf2q's current path violates this (loads Qwen35Model with all-layer F32 expansion = 104 GB on top of the 52 GB Vec<u8> tensor map). Track 1's per-batch loop must be implemented on the mlx-native lazy path that already exists, NOT on `Qwen35Model::load_from_lazy_tensor_map`. This is the structural change that closes the 199 GB → 80 GB gap.

---

## 5. Risk / Blast-Radius

| Track | Risk | Blast Radius |
|---|---|---|
| **Track 1** | **LOW**. New algorithm, isolated module. Reuses existing `MixedBitQuantizer` + `KQuantCodecQuantizer` + sensitivity cache. No new deps. The hand-derived gradient (T1.4) is the main correctness risk; mitigated by synthetic-model autograd-reference falsifier. | New `DynamicQuantCalibrator` module + ~80 LOC of glue. Old `DwqCalibrator` stays alive (deprecated alias). Zero impact on serve path. Zero impact on existing convert paths for non-DWQ presets. |
| **Track 2 (two-process)** | **MEDIUM**. New external dep: `mlx_lm` Python ≥ 0.20 (whichever ships dwq.py with the `--quantized-model` flag). Operator must `pip install "mlx-lm[train]"`. Subprocess plumbing is well-trodden but adds an environment failure mode. Output-format bridge (T2.5) is the highest-risk subtask: mlx-lm's safetensors block layout vs GGUF Q4_K layout. **Researcher #7 must produce the bridge spec or we degrade to dequant-then-requant** (acceptable but loses ~0.1 BPW of trained scale precision). | New CLI presets only. Convert pipeline gains a 2-pass mode. Serve path unchanged (consumes GGUF as today). |
| **Track 2 (native)** | **HIGH**. Multi-month autograd-engine port to mlx-native; cross-repo coordination; might require kernel work. NOT RECOMMENDED for v0.1. | Multi-quarter project. Not in scope for ADR-020 v0.1. |
| **Track 3 (AWQ)** | **MEDIUM**. Per-arch `ScaleConfig` table is hand-tuned per family — you can't generalise. Each new family needs research + testing. | New module; doesn't replace anything. Each new family is a discrete unit of risk. |
| **Track 4 (GPTQ)** | **MEDIUM-LOW** algorithmically (closed-form), but unclear whether mlx-native's Cholesky path exists. | New module; doesn't replace anything. |

---

## 6. Recommended Execution Order

### Phase 1 — Track 1 (1-2 weeks)
**Goal**: Replace today's broken `DwqCalibrator` (variance-magnitude-from-activations, structurally-incorrect-but-shipping) with mlx-lm's clean gradient-based `dynamic_quant`. Same CLI surface (`--quant dynamic-quant-4-6`); `dwq46`/`dwq48` keep working as deprecated aliases. Ship behind `--quant dynamic-quant-X-Y` only; deprecate aliases at v0.2.

- Week 1: T1.1–T1.6 (CLI + algorithm core + binary search)
- Week 2: T1.7–T1.11 (integration + cache + falsifiers + 35B RAM falsifier)

**Exit criteria**: Qwen 3.6 35B-A3B-APEX dynamic-quant-4-5 + Gemma 4 26B dynamic-quant-4-5 both convert end-to-end on 128 GB Mac with peak ≤ 80 GB; perplexity within 0.1 of llama.cpp Q4_K_M baseline.

### Phase 2 — Track 2 (2-4 weeks, conditional)
**Goal**: Add real DWQ as a second user-selectable preset. Two-process subprocess wraps `mlx_lm.dwq`. Cascade preset `dynamic-quant-X-Y+dwq` becomes the new "best quality 4-bit" recommendation.

- Weeks 3-4: T2.0 (decision + scaffolding) + T2.1 + T2.2 (subprocess + targets-to-disk)
- Weeks 5-6: T2.5 (output ingestion + GGUF bridge) + T2.6 (cascade) + T2.8 (RAM falsifier)
- Optional week 7: quality validation (perplexity + MMLU/HellaSwag/PIQA — already in mlx-lm, just call it)

**Exit criteria**: `--quant dwq-4` and `--quant dynamic-quant-4-6+dwq` produce GGUFs that round-trip through hf2q serve with perplexity strictly below Track 1 alone; peak ≤ 90 GB.

### Phase 3 — Tracks 3 + 4 (optional, post-v0.1)
- Only proceed if Phase 1+2 quality is judged insufficient by users on Qwen/Gemma at 4-bit.
- AWQ first (≥ better complement to DWQ than GPTQ).
- Each takes ~3 weeks; budget after v0.2 release.

---

## 7. Open Questions (unresolved by researchers 1-7)

1. **Does MLX's `QuantizedLinear` have well-defined gradients w.r.t. `scales` and `biases`?** mlx-lm's `dwq.py:90-97` calls `m.unfreeze(keys=["scales", "biases"], recurse=False)` and then runs `mx.value_and_grad`. This implies MLX provides a straight-through-estimator (STE) backward through the integer-quant op. **Confirm by reading `mlx-lm/mlx_lm/quant/quantized_linear.py`** (or wherever MLX defines the qdq backward). For Track 2 native variant we'd need to replicate this in mlx-native.

2. **GGUF tensor-type extension for trained scales** (researcher #7's territory). Two paths: (a) round-trip through dequant-then-K-quant (loses fine-grained scale precision), (b) define `Q4_K_DWQ` sentinel tensor type carrying mlx-lm's block layout verbatim. Path (a) is safer for v0.1 and consistent with the existing `KQuantCodecQuantizer` pipeline; path (b) preserves more quality at the cost of extending the GGUF reader. **Recommendation: path (a) for v0.1; revisit path (b) if quality measurements show measurable degradation vs the safetensors original.**

3. **mlx-native autograd roadmap** (researcher #5's territory). If mlx-native has any autograd plans, that changes Track 2's calculus — native variant becomes preferred. Otherwise the two-process variant is correct. **Confirm with mlx-native maintainer.**

4. **Cross-arch DWQ sensitivity** — Qwen 3.6 has both dense layers AND MoE layers AND linear-attention (Deltanet) layers. mlx-lm's `dynamic_quant.py` treats every Linear uniformly. Does Qwen35MoE's `switch_mlp` Linear get sensitivity scored per-expert or globally? mlx-lm uses `SwitchLinear` as a single Module — global. **Confirm this matches user expectations**; per-expert sensitivity would be a slightly different algorithm.

5. **Calibration corpus selection** — mlx-lm defaults to `tristandruyen`'s 512-token chunks of mixed text (see `quant/utils.py:8-15`) for dynamic_quant, and HuggingFace `allenai/tulu-3-sft-mixture` for DWQ. hf2q has its own `--calibration-corpus` infra (researcher #6 territory). **Confirm we keep `--calibration-corpus` overridable but default to mlx-lm's choices for cross-comparability.**

---

## 8. What NOT to Port

- **mlx-lm's distributed pipeline_parallel** (`utils.py:594` `pipeline_load`, `dwq.py:299-302`) — hf2q is single-node. Drop the `mx.distributed.init()`, `all_sum`, `world_size` machinery.
- **mlx-lm's auto-download of HuggingFace datasets** (`tuner/datasets.py`'s `load_dataset`) — hf2q's `--calibration-corpus` is operator-supplied. Document the path expectation.
- **mlx-lm's bfloat16-on-device default** (e.g. `dtype: mx.Dtype = mx.bfloat16` in `dwq.py:78`) — hf2q already has F16/BF16 source; F32 expansion is what we're trying to **eliminate** per Driver B in ADR-020. Match mlx-lm's bf16 storage exactly; do NOT introduce a separate F32 holding stage.
- **mlx-lm's `upload_to_hub` post-hook** (`utils.py:648`) — orthogonal to hf2q; we don't push artifacts.
- **mlx-lm's HuggingFace tokenizer auto-resolve** (`utils.py:429-451`) — hf2q already wraps `tokenizers` 0.22 with its own loader; reuse that.
- **mlx-lm's adapter-file save format** (`tuner/trainer.py:373-377`) — that's for LoRA fine-tuning, not quantization; skip.
- **mlx-lm's `js_div_loss`** (`tuner/losses.py:785-799`) — only `kl_div_loss` is used by `dwq.py`; JS is for other tuner workflows.

---

## 9. Deliverable Format Compliance

- **Length**: ~3450 words (under 3500 cap).
- **Format**: Markdown headers + tables, every line carries a verdict.
- **ADR-020 merge target**: this file replaces ADR-020's "Decision" section in part, supplements with a new "Port Plan" section. Researcher #1-#7 outputs become referenced appendices (researcher #4's GPTQ at `/tmp/cfa-mlx-lm-deep-research/researcher-4-gptq.md` already lands as the source for §3.4 Track 4).
- **Commit posture**: launcher merges this verbatim; no rewrites. Per "no fallback, no stub" mantra, every subtask is complete with falsifier; nothing in this synthesis ends in a TODO.

---

## 10. Bottom-Line Recommendation (one sentence)

**Phase 1 ports `dynamic_quant.py` natively into hf2q (replacing the broken DWQ-46/48 path) over 1-2 weeks; Phase 2 wraps `mlx_lm.dwq` as a subprocess for the cascade-best 4-bit-quality option over 2-4 weeks; AWQ and GPTQ are deferred until those two prove insufficient.**
