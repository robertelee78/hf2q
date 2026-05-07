# mlx-lm AWQ — Deep Research (researcher-3)

**Files read:** `/opt/mlx-lm/mlx_lm/quant/awq.py` (585 LOC), `/opt/mlx-lm/mlx_lm/LEARNED_QUANTS.md`, plus `quant/utils.py:8-26` (`load_data`) and `utils.py:925-950` (`save`) for cross-reference.

---

## 1. Algorithm summary

AWQ ("Activation-aware Weight Quantization", Lin et al. 2023, arxiv 2306.00978; `LEARNED_QUANTS.md:166-168`) is a **calibration-driven, training-free** weight quantizer. The paper insight: **a small fraction of weight channels carry the bulk of activation magnitude**; protecting them dramatically reduces post-quant error.

Two stages run per transformer block:

1. **Per-channel scaling.** For each linear group sharing an input (e.g. q/k/v sharing input-layernorm output), find a scale vector `s` such that `quant(W * s) / s` minimises MSE against the FP block output. This moves quant error from sensitive (high-activation) channels onto less-sensitive ones, then **fuses `1/s` into the previous op** (`apply_scale`, `awq.py:247-273`) so the network stays mathematically equivalent.
2. **Per-group weight clipping.** For each linear (excluding `q_proj`/`k_proj`, see no_clip lists at `:52,:72,:94`), search symmetric clip thresholds `p * |W|.max()` for `p ∈ [0.5, 1.0]` and pick the one that minimises MSE on a sub-sampled activation batch (`search_best_clip`, `:307-374`).

The procedure is a **block-by-block forward pass** on calibration data; weights mutated in place; no gradients, no optimiser. Faithful re-implementation of LLM-AWQ in MLX. `LEARNED_QUANTS.md:13` confirms: *"AWQ scales and clips the weights prior to quantization."*

---

## 2. Code flow

**Entry: `main()`** at `awq.py:533-585`.
1. Parse args (see §3).
2. `mx.distributed.init()` — AWQ runs multi-rank if launched under distributed; calibration data is sharded by `dist_split` (`:185-193`).
3. `load(args.model, lazy=True, return_config=True)` — fetch FP model.
4. Look up family-specific `AWQConfig` from `AWQ_MODEL_CONFIGS` (`:143-151`). Currently supported: llama / mistral / qwen2 / qwen3 (one schema), gemma3_text / gemma3 (gemma-RMSNorm), deepseek_v2 (MoE-aware, with `switch_mlp` branches).
5. `load_data` → `awq_quantize`.
6. `update_config` writes per-tensor bits/group_size into `config["quantization"]` (`:513-530`).
7. `save` writes safetensors + tokenizer + config (`utils.py:925`).

**Core: `awq_quantize()`** at `:399-510`.
- Build `quantize_func` = `dequantize(quantize(w))` round-trip at the target bits/group_size (`:414-416`). This is the "fake quant" used during the search; weights stay FP until the final `nn.quantize` call.
- **Quantize the embedding immediately** at `embed_bits=4` / `embed_group_size=32` and run inputs through it (`:420-424`). The embedding output is the per-block activation tensor that flows through the loop.
- Build a `Catcher` wrapper (`:426-450`) — a stateful nn.Module that intercepts every `nn.Linear`/`SwitchLinear` call, **concatenates the input** onto `module.input_feat`, and (for MoE) concatenates routing `indices`. This is how activation statistics get collected in one forward pass.
- **Per-block loop** (`:452-505`):
  - Wrap leaves with Catchers, run a forward pass (`run_layer`, `:165-182`, batched at 32) to populate `input_feat` on every linear.
  - Restore original modules; capture FP block output `outputs`.
  - Quantize the block naively (no AWQ) and measure `before_loss` against `outputs`. This is the reference.
  - Reload original FP weights from `orig_params`.
  - **`scale_block`** (`:276-304`): for each `ScaleConfig`, call `search_best_scale` (`:196-244`) over `n_grid` scaling ratios. The grid sweeps `ratio ∈ [0, 1)` — `scales = max(x_max ** ratio, 1e-4) / sqrt(scales.max() * scales.min())`. For each ratio, fake-quant the layer weights, re-run the local block, compute MSE, keep the best. Then `apply_scale` fuses `1/s` into the previous op (Linear, LayerNorm, RMSNorm, or gemma-style RMSNorm with the `(1 + w)` re-parameterisation at `:261-265` — non-obvious, gemma-specific).
  - **`clip_block`** (`:377-396`): for each linear not in `no_clip`, sub-sample `n_frames=512` activation rows, batch the W matrix `batch_size=64` rows at a time, sweep `n_grid * max_shrink = 10` clip thresholds, write back the clipped W.
  - Re-quantize the block, measure `after_loss`. **Per-block fallback at `:495-500`:** if `after_loss > before_loss` reload the original weights and quantize without AWQ. Per-block defensible — neat.
  - Pipe the (now quantized) block's output forward as `inputs` for the next block.
  - `mx.eval(block); mx.clear_cache()` after every block.
- After the loop, quantize `lm_head` at the embedding settings (`:507-510`).

`Catcher` and `input_feat` storage are the trick: a single forward pass per block populates per-linear activation statistics in place, then the search loops re-execute the *block* (not the whole model) per ratio/clip candidate. That keeps the inner search cheap.

---

## 3. Hyperparameters + defaults

From `argparse` (`:534-547`):

| Flag | Default | Notes |
|---|---|---|
| `--model / -m` | `mlx-community/Qwen2.5-7B-Instruct-bf16` | source FP model |
| `--mlx-path` | `mlx_model` | output dir |
| `--bits` | **4** | weight bits for transformer blocks |
| `--group-size` | **64** | quant group along input-channel axis |
| `--embed-bits` | **4** | embedding + lm_head precision |
| `--embed-group-size` | **32** | embedding + lm_head group |
| `--num-samples` | **128** | calibration sequences (rounded up to multiple of distributed group size, `:551-553`) |
| `--sequence-length` | **512** | tokens per sample |
| `--n-grid` | **20** | scale-search granularity (and base for clip's `0.5*n_grid=10`) |
| `--seed` | 123 | calibration sample selection |

Note the **defaults disagree with the README** at `LEARNED_QUANTS.md:114-119`, which says `--num-samples 32` and `--n-grid 10`. The code is the ground truth; the README is stale or describes a recommended fast path. Flag for hf2q porting decisions: pick code-side numbers.

The `awq.py:404` function-default `bits=3` does not surface to the CLI (CLI passes `bits=4` via `args.bits`); it is a quirk of the internal default and is harmless.

Calibration data (`utils.py:8-26`): downloaded once on first run from a public gist (`calibration_data_v5_rc.txt`), tokenised, randomly chunked into non-overlapping `sequence_length`-sized windows. Same source as DWQ. **Network dependency on first run.**

---

## 4. Memory profile

Phases:

1. **Load FP model (lazy).** Standard.
2. **Quantize embedding + run.** Embedding → 4-bit; `inputs` is `[num_samples, sequence_length, hidden]` — for Qwen3-7B (hidden 3584, fp16) defaults give ~470 MB. **Held in RAM the entire run**; every block consumes/produces it.
3. **Per-block Catcher pass.** Each `nn.Linear`/`SwitchLinear` accumulates `input_feat` by **concatenation** at `:434`. Per linear: `[num_samples, seq_len, in_features]` (MoE: only routed tokens). For attention, q/k/v share input → 3× the same tensor; gate_proj/up_proj → 2×. **Dominant transient.** ~470 MB *per linear* on 7B at defaults, ~6-7 linears per block.
4. **Scale search.** `search_best_scale` (`:196-244`) does `n_grid=20` block re-runs on the same `x`. Each iteration fake-quants weights then runs the block; weights reloaded from `tree_flatten` snapshot at `:217` after each ratio. Peak +~2× block-output memory.
5. **Clip search.** `search_best_clip` (`:307-374`) sub-samples `x` to `n_frames=512` rows (`:319-321`), processes W rows `batch_size=64` at a time. Sub-sampling is deliberate memory control — the full `bdg,odg->bod` einsum would blow up otherwise.
6. **Final quantize + `after_loss`.** Cheap.
7. **Cross-block.** `inputs = outputs` feeds the next block. `mx.eval(block); mx.clear_cache()` at `:504-505` releases prior transients.

**Stream-saved like dwq.py? No.** DWQ at `dwq.py:57-65` writes targets to safetensors on disk via `mx.save_safetensors`. AWQ keeps `inputs` and `input_feat` **entirely in memory**. Memory profile closer to dynamic_quant than DWQ.

**Scaling:** memory ≈ `num_samples * seq_len * max_in_features * concurrent_linears_per_block`. MoE `deepseek_v2_awq` (`:91-141`) runs `switch_mlp` only on routed tokens (some mitigation). Operators on 35B MoE should start `--num-samples 32 --sequence-length 512`.

---

## 5. Output format

`save(dst_path, src_path_or_repo, model, tokenizer, config)` at `utils.py:925-950`:
- **Safetensors** with the same packed-quantized layout as any `nn.quantize`-ed mlx model — `weight`, `scales`, `biases` per quantized linear, with the embedding and lm_head at their own (typically 4-bit / group_size 32) settings.
- **`config.json`** with a `quantization` dict that records `{group_size, bits}` *per tensor path* (`update_config`, `:513-530`). The dummy top-level `{"group_size": 64, "bits": 4}` at `:518` is a placeholder for tooling that expects a single shape; the per-path entries are the truth. Layers without a `bits` attr (i.e. unquantized) record `False` (`:527`).
- `tokenizer.save_pretrained(dst_path)` and copy of `*.py` / `generation_config.json`.

This is **identical in shape to `dynamic_quant.py`'s output** — both emit per-tensor `quantization[path]` entries plus a top-level dummy. The only structural difference vs a uniform `nn.quantize` save is the per-path config; same on-disk byte layout for the tensors themselves. Crucially: **no AWQ-specific metadata** survives (the scales are fused into prior ops, the clipping is folded into the quantized W). The output is indistinguishable at the format level from a plain quantized model — which is exactly the AWQ paper's design (zero inference-time cost).

---

## 6. Compute cost

LEARNED_QUANTS.md:109-110: *"can take anywhere from a few minutes to several hours to run depending on the model size and the number of samples"*.

Vs `dynamic_quant`: AWQ is **slower per block** because of the inner grid searches. Per block: 1 Catcher pass + 1 reference quant pass + `n_grid=20` scale-search re-runs per `ScaleConfig` (3-7 configs per family) + `n_grid=10` clip-search re-runs per linear. Dynamic quant just does 1 sensitivity pass per layer.

Vs DWQ: AWQ is **faster** because it's training-free (no backward pass, no optimiser, no epochs). DWQ at `dwq.py` runs an actual gradient-descent loop over scales+biases for many steps; AWQ is one forward pass plus a closed-form grid search.

**Rough ranking on a 7B model (LEARNED_QUANTS.md:18-20):** dynamic_quant (minutes) < AWQ (10s of minutes to ~1 hr) < DWQ (~hours). GPTQ similar to AWQ.

`run_layer`'s `batch_size=32` (`:165-182`) is the inner-loop batching knob — not exposed via CLI, and fine for default sample counts.

---

## 7. Quality

LEARNED_QUANTS.md:18-20 ranks DWQ above the others: *"Dynamic quantization is the fastest to run. DWQ takes longer but typically yields better results."* AWQ and GPTQ are positioned between — not explicitly ranked against each other.

The README also notes (`:21-22`) that **methods cascade**: *"a dynamically quantized model can be further refined with DWQ."* No claim is made that AWQ + DWQ cascades, but the fall-through pattern is structurally compatible (AWQ output is just a quantized safetensors).

The per-block fallback at `awq.py:495-500` ("Loss is not reduced, falling back to original weights") is honest engineering — it concedes that AWQ doesn't always help and self-disables per block.

**Empirical baseline:** the AWQ paper reports ~0.5-1 ppl improvement vs round-to-nearest at 4-bit on llama-7B/13B. mlx-lm's port is a faithful re-implementation, so similar results expected.

---

## 8. Port viability for hf2q

**(a) Algorithm core.** ~600 LOC of Rust translating `awq.py:196-510`. Core ops are `mse`, `quantize`/`dequantize` round-trip, broadcasts, grid-search loops — all mlx-native primitives hf2q already binds for inference. No gradient infra (unlike DWQ).

**(b) Activation collection.** **Non-trivial new requirement.** Per-linear input capture during forward. mlx-lm wraps every `nn.Linear` in a `Catcher` (`:426-450`); hf2q's forward kernels are direct GPU dispatches with no wrapping abstraction. Two options: a calibration-only forward path parallel to inference, or a per-projection hook system in `forward_prefill_*.rs`. Capture is read-once per block then released — a transient `Vec<MlxBuffer>` keyed by `(block_idx, linear_name)` populated on calibration-mode forward, dropped after the block's AWQ pass. **Estimate ~1-2 weeks** to wire across Gemma + Qwen3.5/3.6 + llama.

**(c) Scale + clip.** MSE grid searches over fake-quant'd weights — pure mlx-native, no novelty. Per-family `AWQConfig` (`:49-141`) encodes which projections share an input. **Brittle** — every new family needs an entry, and the gemma RMSNorm `(1+w)` re-param at `:261-265` shows non-obvious wrinkles.

**(d) MoE handling.** `deepseek_v2_awq`'s `switch_mlp` branches (`:111-139`) and `indices` capture (`:439-446`) are essential for Qwen3.5/3.6 — hf2q's live target. Not avoidable.

**(e) Output format.** Byte-compatible with hf2q's GGUF/safetensors loader; per-tensor `config["quantization"]` already a known pattern from dynamic_quant (see researcher-2).

---

## 9. Comparison: AWQ vs dynamic_quant vs DWQ

The four methods are **orthogonal in their unknowns** and thus compose:

| Method | Tunes | Mechanism |
|---|---|---|
| **dynamic_quant** | Per-tensor bit width | Sensitivity ranking → high-bits for sensitive layers, low-bits for the rest |
| **AWQ** | Per-channel scale `s` (fused) + per-group clip `c` | Closed-form MSE grid search over fake-quant; one block at a time |
| **DWQ** | Quant scales + biases (already-quantized model) | Gradient descent vs FP teacher logits |
| **GPTQ** | Quantized weights (column-by-column) | Hessian-aware closed-form per-layer error minimisation |

**Cascading order (sketch, none of this is enforced by code, all of it is structurally compatible):**

1. **dynamic_quant FIRST** — picks bit widths. Output: a per-tensor `{group_size, bits}` map plus a quantized model.
2. **AWQ SECOND** — operates per-block at *whatever bit width that block was assigned*. The `quantize_func` closure at `awq.py:414-416` is closed over `bits`/`group_size` at quantize-time; making it a *per-block* lookup is a one-line change. AWQ then optimises scale+clip *given* the bit assignment.
3. **DWQ THIRD** — fine-tunes the already-quantized model's scales/biases. README explicitly endorses dynamic_quant→DWQ at `LEARNED_QUANTS.md:21-22`. AWQ→DWQ is structurally identical — DWQ doesn't care how the quant scales were initialised.
4. **GPTQ** sits at the same position as AWQ — they are alternative weight-quant initialisers; you would pick one, not both.

The dependency graph: **dynamic_quant → {AWQ | GPTQ} → DWQ**. AWQ and GPTQ are mutually exclusive at any given tensor; the others compose freely.

---

## 10. Recommendation for hf2q

**Yes, AWQ is a viable third track**, but **not the highest-priority one** given hf2q's current state.

**Pros:**
- Training-free → tractable inside hf2q's existing inference-only build (no autograd, no optimiser).
- ~600 LOC algorithm core — small, contained.
- Output is a plain quantized model — no runtime changes; fits the existing GGUF loader path.
- Robust per-block fallback (`:495-500`) makes failure modes graceful.
- MoE-aware (deepseek_v2 schema) — directly applicable to Qwen3.5/3.6 quant pipelines.
- Cascades cleanly with dynamic_quant (orthogonal axis), so adding AWQ extends rather than replaces.

**Cons / costs:**
- The activation-capture mechanism is the load-bearing new infrastructure (~1-2 weeks).
- Per-family `AWQConfig` schemas are hand-coded and brittle — each new family is bespoke (5 currently supported in mlx-lm).
- Quality win is **modest** (~0.5-1 ppl at 4-bit) — DWQ's gain is larger per the README ranking.
- Compute cost is non-trivial (10s of minutes to hours per model) — not something operators run casually.

**Suggested ordering:**

1. **Track 1 (priority): dynamic_quant.** Per researcher-2; it is the cheapest both to port and to run. Bit-width assignment is the highest-leverage axis for hf2q's mixed-precision story (especially for the Q5_K_M-vs-Q4_0 default-pick problem).
2. **Track 2: DWQ.** Highest quality, but the **most expensive port** (gradient infrastructure). Defer until hf2q has a training/calibration runtime worth the build-out — or wrap mlx-lm's DWQ as an external producer and just consume its output.
3. **Track 3 (this proposal): AWQ.** Port **after** dynamic_quant lands. The activation-capture infra it requires is **a strict subset of what DWQ would need** (DWQ also needs to score per-tensor outputs against a teacher), so AWQ doubles as scaffolding for an eventual DWQ port.
4. **Track 4: GPTQ.** Lowest priority — overlaps with AWQ in the design space, smaller implementation but offers redundant rather than complementary capability.

**Recommendation:** **Yes, AWQ as a third track — but sequenced after dynamic_quant, and explicitly framed as scaffolding-on-the-way-to-DWQ.** Build the activation-capture hook system once, exploit it for AWQ first (faster validation, lower risk), then layer DWQ on top later. Avoid the trap of porting AWQ in isolation and re-paying the activation-capture cost a second time when DWQ comes around.

---

## Citation index

- Algorithm core: `awq.py:196-244` (search_best_scale), `awq.py:307-374` (search_best_clip), `awq.py:399-510` (awq_quantize)
- Activation capture: `awq.py:426-450` (Catcher)
- Family schemas: `awq.py:49-151` (AWQConfig instances + AWQ_MODEL_CONFIGS)
- Per-block fallback: `awq.py:495-500`
- Output format: `awq.py:513-530` (update_config), `utils.py:925-950` (save)
- Calibration data: `quant/utils.py:8-26` (load_data, gist URL at line 14)
- Defaults inventory: `awq.py:534-547`
- README ranking: `LEARNED_QUANTS.md:18-22`
- Paper citation: `LEARNED_QUANTS.md:166-168` (arxiv 2306.00978, Lin et al.)
- Stale README defaults: `LEARNED_QUANTS.md:116,118` (--num-samples 32, --n-grid 10) vs `awq.py:543,545` (128, 20)
- Gemma RMSNorm wrinkle: `awq.py:261-265` (the `(1+w)` re-parameterisation)
