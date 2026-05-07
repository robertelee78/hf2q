# Researcher #4 — mlx-lm GPTQ deep-read

CFA session: cfa-20260506-mlxlm-research
Sources read end-to-end:
- `/opt/mlx-lm/mlx_lm/quant/gptq.py` (229 LOC)
- `/opt/mlx-lm/mlx_lm/quant/utils.py` (26 LOC, calibration loader)
- `/opt/mlx-lm/mlx_lm/LEARNED_QUANTS.md` (GPTQ section + footnote `[^2]`)

LOC comparison across the four methods (`wc -l /opt/mlx-lm/mlx_lm/quant/*.py`):
awq.py 585 · dwq.py 411 · dynamic_quant.py 259 · gptq.py 229. GPTQ is the smallest by ~30 LOC.

---

## 1. Algorithm summary

GPTQ (Frantar et al. 2022, arXiv:2210.17323, footnote `[^2]` in `LEARNED_QUANTS.md:169`) is a one-shot post-training weight quantizer. For each linear layer it solves a layer-local least-squares problem: given calibration inputs `X`, find a quantized `Wq` minimizing `||W X − Wq X||²` per output row. The closed-form result depends on the input Hessian `H = XᵀX`, and GPTQ's contribution is a column-by-column (or here, group-by-group) greedy quantizer that uses the inverse-Hessian's Cholesky factor to propagate each column's quantization error into the still-unquantized columns, so that subsequent columns absorb prior error.

`LEARNED_QUANTS.md:15-16` summarizes it cleanly: "GPTQ finds quantized weights which minimize the squared error of each layer's output given the provided input." Compared with siblings: DWQ fine-tunes float scales/biases via teacher distillation, AWQ scales+clips weights pre-quantization, dynamic quant does per-layer sensitivity-driven bit-allocation (`LEARNED_QUANTS.md:11-16`).

## 2. Code flow

Three phases, all inside `gptq_quantize` (`gptq.py:52-159`).

**Phase A — instrumentation (`gptq.py:61-66`):** Walk `model.leaf_modules()`, wrap every `nn.Linear` and `SwitchLinear` (the gated `gptq_types` set on line 62) in a `Catcher` (`gptq.py:40-49`). `Catcher.__call__` flattens the input to 2-D and accumulates `H += xfᵀ @ xf` on every forward pass. Note: the wrapped module's output is forwarded unchanged, so calibration is a normal forward of the float model.

**Phase B — Hessian collection (`gptq.py:68-76`):** Iterate calibration batches of size `batch_size` (default 8); call `model(batch)` and `mx.eval(layers)` after each step to flush. The same `H` accumulator is shared across all batches per layer. After this loop, every Catcher holds the layer's full input gram matrix.

**Phase C — per-layer quantization (`gptq.py:95-142`):** For each captured layer:

1. `compute_inverse_hessian(H)` (`gptq.py:78-86`) on the **CPU stream** (`with mx.stream(mx.cpu)`, line 79):
   - Add Tikhonov-style damping: `damp = 1e-2 * mean(diag(H))`, add to diagonal in-place (line 80-82).
   - `cholesky(H)` → `cholesky_inv(...)` → `cholesky(..., upper=True)`. The final `Hinv` is the upper-triangular Cholesky factor of `H⁻¹`. This factor is what GPTQ needs: column `k`'s residual gets distributed into columns `>k` using row `k` of this triangular factor.
2. `del l.H` immediately (line 101) — frees the dense `[in_features, in_features]` Hessian before quantizing, which is the single biggest memory win.
3. Walk the input dimension in **groups** of `group_size` (default 64, line 109). For each group `[i, j)`:
   - Call MLX's built-in `mx.quantize(Wl, bits, group_size)` (line 115) just to extract canonical `(scales, biases)` for that group. Indices are discarded — they get recomputed after error propagation.
   - For each column `k` in the group (line 119-128): compute the per-element quantization error via the compiled `gptq_error(w, d, scales, biases)` kernel (`gptq.py:88-93`), where `d = Hinv[k, k]` is the pivot. Then propagate that error into all unquantized columns within the same group: `W[..., k:k+j] -= e @ Hinv[k:k+1, k:k+j]` (line 126). `mx.eval(err, W)` per inner step (line 128) bounds graph depth.
   - After the group is done, propagate the group's accumulated error into all subsequent groups: `W[..., j:] -= err @ Hinv[i:j, j:]` (line 130). This is the "lazy batching" of the GPTQ paper.
4. Concatenate all group scales/biases and call `quantize(W, bits, scales, biases)` (the file-local helper, `gptq.py:26-37`) to pack the corrected weight into uint32-packed form. Replace the layer with `l.module.to_quantized(...)` and assign packed weight + scales + biases (lines 132-141).

**Phase D — fallback quantization (`gptq.py:146-158`):** Any remaining quantizable leaf (e.g. embeddings, norms, anything not `Linear`/`SwitchLinear`) is quantized at `fallback_bits`/`fallback_group_size`. The per-key dict in `config[k] = fallback_config` (line 155) means the safetensors metadata records mixed precision per layer.

## 3. Hyperparameters and defaults

From `main()` argparse (`gptq.py:162-200`):

| Flag | Default | Notes |
|---|---|---|
| `--model` | `Qwen/Qwen3-0.6B-base` | HF id |
| `--mlx-path` | `mlx_model` | output dir |
| `--bits` | 4 | GPTQ-quantized layer bits; asserted ∈ {2,4,8} at `gptq.py:27` |
| `--group-size` | 64 | input-dim group |
| `--fallback-bits` | 6 | for layers GPTQ doesn't touch |
| `--fallback-group-size` | 64 | |
| `--num-samples` | -1 (= all) | calibration chunks |
| `--sequence-length` | 512 | calibration chunk length |
| `--seed` | 123 | for `mx.random.permutation` chunk pick |
| `batch_size` | 8 | **not exposed in argparse**; hard-coded default in `gptq_quantize` signature line 59 |

Internal magic numbers worth flagging:
- Damping factor `1e-2 * mean(diag(H))` at `gptq.py:80` — standard GPTQ default; not configurable.
- `parallel_tool_calls`-style packing assumption in `quantize()` (line 26-37): packs `32//bits` elements per uint32. Uses `mx.power(2, mx.arange(0, 32, bits, mx.uint32))` for shift table — same little-endian packing MLX's built-in `mx.quantize` uses, so layers stay binary-compatible with stock MLX kernels.

LEARNED_QUANTS.md (lines 138-142) only documents `--bits`. `--num-samples` is described as `-1` ("use all") in argparse help.

## 4. Memory profile

**Peak per layer = the Hessian.** For input dim `d_in`, `H` is dense `float32 [d_in, d_in]`. For a Qwen3-0.6B `q_proj` (d_in ≈ 1024) that's 4 MB. For a 35B-class model with d_in = 5120 it's ~100 MB — still tractable. For an MoE `SwitchLinear` with the same d_in, the Catcher accumulates one shared `H` per gating group, not per-expert (the input to all experts is the same router-routed activation). This is why the wrapper is stateless about expert routing — it just sees `xf.T @ xf`.

**Key memory hygiene moves:**

1. **Sequential per-layer freeing** (`gptq.py:100-102`): `Hinv` is computed, then `del l.H` immediately. The model never holds more than one Hessian + one inverse-Hessian factor at a time. This is the structural reason mlx-lm's GPTQ scales to 30B+ models on a Mac — peak is `O(max_layer_d_in²)`, not sum.
2. **CPU-stream linalg** (`gptq.py:79`): Cholesky/inv runs on CPU, freeing GPU for nothing in particular here, but more importantly avoiding GPU-side dense linalg pressure that would conflict with `mx.eval(layers)` flushes.
3. **`mx.eval(err, W)` inner loop** (`gptq.py:128`): caps the lazy-graph depth at one column. Without it, MLX would build a graph 64-deep per group and OOM on intermediate buffers.
4. **uint32 packing in-place** (`gptq.py:30-37`): the corrected `W` (still float32) is consumed once by `quantize()`, then dropped when `layer.weight = Wq` overwrites. No double-buffer.

The classic GPTQ memory scaling problem (storing many `d_in²` Hessians + per-block inverse columns) is sidestepped here by (a) wrapping at the leaf, not maintaining a global activation cache, and (b) doing one layer at a time with `del`. The trade-off: you must run the full forward pass once *per quantization run* (Phase B), which costs time but not memory.

## 5. Output format

GPTQ doesn't define its own format — it reuses MLX's standard quantized layer encoding via `to_quantized(bits, group_size)` (`gptq.py:136`). The safetensors that `save()` (`gptq.py:219-225`) writes contain the standard MLX trio per quantized layer: `weight` (uint32-packed, little-endian shifts per `gptq.py:30-37`), `scales`, `biases`.

The metadata is the GPTQ-specific bit. `config["quantization"]` returned at line 159 is a flat dict:
- top-level keys: `"bits"`, `"group_size"` (the GPTQ defaults, line 150),
- per-layer keys: `config[k] = fallback_config` for fallback layers (line 155).

So the saved JSON config looks like `{"bits": 4, "group_size": 64, "model.embed_tokens": {"bits": 6, "group_size": 64}, ...}`. Standard MLX loaders read this and instantiate the right precision per leaf.

The packed weight format is **bit-identical** to a stock `mx.quantize` output. There is no GPTQ marker, no Hessian residual stored, no calibration-data fingerprint. Once corrected weights are packed, the file is indistinguishable from a normally-quantized MLX model. That's intentional — GPTQ is a quality-improvement front-end, not a new format.

## 6. Compute cost (relative)

From the doc and code:

| Method | Per-layer work | Doc characterization |
|---|---|---|
| `dynamic_quant` | one fwd for sensitivity + final quant | "fastest to run" (`LEARNED_QUANTS.md:18`) |
| `awq` | grid search `n_grid=10` × samples × layers | "few minutes to several hours" (`LEARNED_QUANTS.md:109-110`) |
| `gptq` | 1 fwd (Hessian) + per-layer Cholesky + per-column LS | "few minutes to several hours" (`LEARNED_QUANTS.md:135-136`) |
| `dwq` | gradient training, multi-epoch | "takes longer but typically yields better results" (`LEARNED_QUANTS.md:18-19`) |

GPTQ sits in the middle. Per-layer Phase C is `O(d_in² · d_out / bits_packed)` for the error propagation (the `e @ Hinv[k:k+1, k:k+j]` step dominates for large `d_in`). For Qwen3-0.6B-class targets it's minutes; for 30B class it's hours but no GPU training — which means it tolerates a Mac mini better than DWQ.

## 7. Port viability for hf2q

What hf2q would need to add:

1. **Calibration-data flow.** mlx-lm uses a 26-LOC fetcher (`utils.py`) that downloads a fixed text corpus (`calibration_v5.txt`, gist URL on `utils.py:14`), tokenizes the entire file once, reshapes into `[N, seq_len]` chunks, and samples `num_samples` rows by a random permutation. hf2q already has tokenizer plumbing and chat-template handling — this is a 1-day port, mostly file I/O.

2. **Hessian accumulator.** Apple's `Catcher` (`gptq.py:40-49`) leans on MLX's lazy graph + `xf.T @ xf` autocomputing on whatever stream the forward is on. In hf2q (Rust + mlx-native) this means: hooking forward at every linear projection in the model, accumulating an `[d_in, d_in]` f32 buffer per layer, and ensuring we drain it via `commit_and_wait` before the second pass. MlxBuffer Drop/residency rules apply (`solution_mlx_native_residency_lifetime_race`). Estimated 3-5 days because we don't have a generic forward-hook abstraction — we'd add a per-family GptqHook trait paralleling FamilyHookFactory (ADR-017 B-dense.2, `project_adr017_bdense2_landed.md`). MoE/SwitchLinear: one `H` per gating group, not per-expert (mlx-lm gets this for free via wrapping the `SwitchLinear` module itself).

3. **Solver.** mlx-lm's `compute_inverse_hessian` is three MLX linalg calls (`gptq.py:83-85`). mlx-native has Cholesky support (it's a wrapped Accelerate/Metal Performance Shaders call), so the math itself is a thin Rust binding. The damping rule (`1e-2 * mean(diag(H))`, line 80) is the only non-trivial scalar prep. Cholesky factorization of `H⁻¹` to get the upper-triangular factor used in error propagation is the GPTQ-specific bit — straightforward, but verify upper- vs lower-triangular convention against the paper to avoid a silent off-by-transposition.

4. **Per-column quantize/propagate kernel.** `gptq_error` (`gptq.py:88-93`) is `@mx.compile`-decorated; in Rust we'd want either a tight Metal compute shader or a fused mlx-native op. Implementing this naively as separate Tensor ops would inherit the kernel-launch-count cost we just spent ADR-013 P21 fighting (`project_adr013_p21_kernel_bound_finding`, `project_adr013_p21_stage4_canonical_2026_05_02`). The inner loop is `group_size` (64) iterations per group — fusing them is meaningful, not optional.

5. **Mixed-precision config plumbing.** The per-layer fallback dict (`gptq.py:155`) requires that hf2q's GGUF/safetensors writer can emit per-tensor bit-width metadata. GGUF's quant-type-per-tensor field already supports this; the writer change is a few dozen lines.

**Estimate:** ~1-2 weeks of focused work to get a 4-bit GPTQ pass running on a small model and produce byte-identical-format output. Most of the risk is in (4) — the hot inner loop has to fuse or it's unusably slow on 30B+. mlx-lm avoids this entirely by living above MLX.

## 8. Recommendation — port priority

**My ranking for hf2q port priority, considering only this method's profile:**

1. **`dynamic_quant` first.** Cheapest to run, smallest file, decision-only output (a sensitivities JSON + per-layer bit-width). It composes with anything we already produce — including any GPTQ output we make later — so it's a force-multiplier. (Researcher #3's territory; flagging here for cross-reference.)

2. **`gptq` second.** Smallest learned-quant file (229 LOC), no training loop, no gradient backward, CPU-resident linalg. Output is bit-identical to an ordinary MLX/GGUF quant — *no new format*, no inference-side changes in hf2q. The only real engineering risk is the fused per-column kernel (item 4 above). The reward is large: GPTQ is the de-facto baseline learned-quant in the broader ecosystem (AutoGPTQ), so producing GPTQ-improved Q4_0 GGUFs lets us compare apples-to-apples with llama.cpp's `llama-quantize --imatrix` flow.

3. **`awq` third.** Larger surface, grid-search compute cost, and AWQ's win is mostly in the absence of ALU-quantization-aware tools we already have downstream.

4. **`dwq` last.** It's the highest-quality method per the doc (`LEARNED_QUANTS.md:18-20`) but it's a full training loop with teacher forcing — biggest port surface, biggest GPU memory footprint, and the reward over a GPTQ+dynamic_quant cascade is incremental. Worth doing eventually; not the next thing.

**Cascade note** (`LEARNED_QUANTS.md:20-21`): mlx-lm explicitly recommends layering — "a dynamically quantized model can be further refined with DWQ." The same composition (`dynamic_quant` → `gptq` → optional `dwq`) is the natural hf2q roadmap, and ports 1+2 alone close most of the ecosystem-parity gap with llama.cpp's imatrix flow.

---

## Non-obvious choices flagged

- **CPU stream for linalg** (`gptq.py:79`) — not just for memory; it's also stability. Metal's `cholesky` was historically less robust than Accelerate's; this avoids the question.
- **`del l.H` immediately after computing `Hinv`** (`gptq.py:101`) — load-bearing for memory; not just hygiene.
- **`mx.eval(err, W)` inside the per-column inner loop** (`gptq.py:128`) — this is the difference between "runs on a Mac" and "OOMs at 64 columns deep".
- **`SwitchLinear` is wrapped at the gating level, not per-expert** (`gptq.py:62`, `Catcher` accumulates `xf.T @ xf` on the pre-routing input). hf2q's MoE port must replicate this; per-expert Hessians would be wrong (different traffic per expert) and ~N×-larger memory.
- **`batch_size=8` is hard-coded in `gptq_quantize`'s signature** (`gptq.py:59`) and not surfaced via argparse — easy to miss when tuning memory.
- **The GPTQ output is format-indistinguishable from a stock MLX quantized model** — no version bump, no marker, no Hessian residual stored. Discoverability for downstream tools relies entirely on filename/hub-card discipline.
- **Damping `1e-2 * mean(diag(H))`** (`gptq.py:80`) — chosen, not derived. The original GPTQ paper uses `1%` of mean-diag too; mlx-lm doesn't deviate. Worth keeping identical in hf2q for cross-validation.
