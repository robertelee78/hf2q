# mlx-lm Dynamic Quantization — Deep Research

**Source files** (read in full):
- `/opt/mlx-lm/mlx_lm/quant/dynamic_quant.py` (259 LOC)
- `/opt/mlx-lm/mlx_lm/quant/utils.py` (26 LOC)
- `/opt/mlx-lm/mlx_lm/LEARNED_QUANTS.md` §"Dynamic Quantization" (lines 76-99)
- Cross-refs: `/opt/mlx-lm/mlx_lm/utils.py:210-215` (`compute_bits_per_weight`), `:774-850` (`quantize_model`)

Comparison anchor: `/opt/hf2q/src/calibrate/sensitivity.rs:14-80`, `/opt/hf2q/src/calibrate/dwq_calibrator.rs:265-279`.

---

## 1. Algorithm Flow (mapped to dynamic_quant.py)

```
main() [149-256]
 ├─ load(model, return_config=True)                                   # 188 — full FP teacher
 ├─ load_data(tokenizer, num_samples=-1, sequence_length=512)         # 192 — calibration corpus
 ├─ estimate_sensitivities(model, data, low/high bits+group_size,...) # 194-203
 │   ├─ build q_model (deep-copy of teacher, low-bit qdq on weights)  # 53-63
 │   ├─ for each batch: targets = teacher(batch); grads = ∇ KL(q,t)   # 75-86
 │   ├─ accumulate grads; sensitivity = ⟨g, w_low − w_high⟩ / size    # 88-94
 │   └─ flatten to list[(path, score)]                                # 104
 ├─ json.dump(sensitivities, ...)                                     # 205-206
 ├─ estimate_threshold(model, sens, target_bpw, ...)                  # 219-227
 │   └─ binary-search threshold τ:
 │       repeat: q_model = deepcopy(model);
 │               nn.quantize(q_model, predicate(τ));
 │               bpw = compute_bits_per_weight(q_model);
 │               narrow [lo, hi] toward target_bpw                    # 130-144
 ├─ quant_predicate via threshold τ                                   # 229-234
 ├─ quantize_model(model, config, low_bits, low_group_size,           # 236-242
 │       quant_predicate)                                                — in-place
 └─ save(mlx_path, ...); print(peak memory)                           # 248-255
```

Two-pass: first pass produces a per-layer scalar score; second pass binary-searches a single scalar threshold over those scores to land on `target_bpw`.

---

## 2. `estimate_sensitivities` (lines 38-106) — line-by-line

### `qdq` helper (49-51)
Round-trip `quantize → dequantize` at requested `bits`/`group_size`. Produces a same-shape, same-dtype tensor whose values are the lossy reconstruction. Used twice: (a) once on every layer at `low_bits` to build `q_model` (line 58); (b) once per layer at `high_bits` inside `compute_sensitivity` (line 91) to compute the *high-bit reconstruction* on demand without ever materializing a fully high-bit copy of the model.

### Layer enumeration (53-54)
```python
layers = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
layers = {k: l for k, l in layers if hasattr(l, "to_quantized")}
```
`leaf_modules()` returns the dict of leaf `nn.Module`s; `tree_flatten` with `is_leaf=nn.Module.is_module` halts recursion at the module boundary so it yields `(dotted_path, module)` pairs. Filter by `hasattr(l, "to_quantized")` keeps only modules MLX considers quantizable (i.e. `nn.Linear`, `nn.Embedding`). Non-obvious: this *includes embeddings*, not just MLP/attention projections.

### `q_model` construction + freeze (55-63)
```python
q_model = copy.deepcopy(model)                          # 55  — second full FP copy
q_layers = copy.deepcopy(layers)                        # 56  — third copy of the leaves
for l in q_layers.values():
    l.weight = qdq(l.weight, low_bits, low_group_size)  # 58  — replace weight w/ low-bit QDQ
    l.freeze(); l.unfreeze(keys=["weight"])             # 60-61 — only weight gets gradients
q_model.freeze()                                        # 62
q_model.update_modules(tree_unflatten(...))             # 63  — splice qdq'd layers back in
```
What's frozen vs unfrozen: **only the QDQ'd `weight` of quantizable leaves is trainable**. Biases, layernorms, embeddings-of-non-quantizable-modules, RoPE caches — frozen. The autograd run downstream computes `∂loss/∂weight` exclusively for these QDQ'd weights.

Critically, `l.weight = qdq(...)` replaces with a *dequantized FP tensor whose values are quantized to a low-bit grid*. Gradients flow through the dequantize (which is just an affine map with stored `scales`/`biases`); the quantize itself was an `mx.eval`'d constant by the time `value_and_grad` looks at the graph. So this is straight-through-estimator-free — gradients are exact w.r.t. the dequantized values.

### `loss_fn` closure (65-66)
```python
def loss_fn(batch, targets):
    return kl_div_loss(q_model(batch), targets).mean()
```
Optimizes KL-divergence between the *low-bit student's* logits and the *FP teacher's* logits. **Note**: `targets` is *teacher logits* (line 81 `targets = model(batch)`), not ground-truth tokens. This is logit-distillation. `mean()` reduces over batch × sequence × vocab.

### Per-batch loop (75-86)
```python
for e, s in tqdm(enumerate(range(0, len(data), batch_size)), ...):
    batch = data[s : s + batch_size]                # 80 — batch_size=4 default
    targets = model(batch)                          # 81 — teacher forward
    mx.eval(targets)                                # 82 — force materialize (frees teacher graph)
    _, grads = nn.value_and_grad(q_model, loss_fn)(batch, targets)  # 83
    grad_accum = tree_map(lambda x, y: x + y, grad_accum, grads)    # 84
    del grads                                       # 85 — release reference
    mx.eval(grad_accum)                             # 86 — force materialize accumulator
```
Each iteration: (a) `mx.eval(targets)` forces the teacher's compute graph to evaluate and be discarded — without this, MLX's lazy eval would chain the teacher's activations into the student's grad graph, blowing memory; (b) `del grads` + `mx.eval(grad_accum)` — `grads` is a python-side pytree of mx.arrays; deleting + eval'ing the accumulator means the next iteration's `value_and_grad` allocates fresh grad buffers rather than retaining the prior trace. This is the **single biggest memory-control lever** in the whole script.

### Sensitivity formula (88-94)
```python
def compute_sensitivity(gradient, low_q_weight, original_weight):
    n_batches = (len(data) + batch_size - 1) // batch_size
    gradient = gradient / n_batches
    high_q_weight = qdq(original_weight, high_bits, high_group_size)
    param_size = original_weight.size / 1e6
    alignment = (gradient * (low_q_weight - high_q_weight)).sum()
    return alignment / param_size
```

**Theoretical reading.** For a small perturbation `δw = w_low − w_high`, the first-order change in the (KL) loss is `δL ≈ ⟨∇L, δw⟩`. So `(gradient * (low_q_weight − high_q_weight)).sum()` is the **first-order Taylor estimate of how much extra KL-loss you incur by quantizing this layer at low_bits instead of at high_bits**. Higher value → higher cost of going low → "more sensitive" → spend high_bits on it.

Two non-obvious choices:
- **Sign matters**: this is a *signed* dot product, not an absolute. Layers where `δw` aligns with `∇L` (you'd be moving uphill) score positive; layers where it's anti-aligned (going low actually *helps* a bit) score negative. The threshold predicate `sensitivities[p] > τ` (line 121) preserves sign.
- **Per-million normalization** (`/ 1e6`): divides by parameter count in millions. Without this, larger matrices would dominate the ranking purely by element count. This makes scores comparable across e.g. tiny attention out-projections vs huge MoE expert matrices.

### Output shape (104)
```python
sensitivities = [(k[:-7], s.item()) for k, s in tree_flatten(sensitivities)]
```
`k[:-7]` strips the trailing `".weight"` suffix (7 chars) so paths line up with `nn.quantize`'s class-predicate path key. List of `(layer_path, scalar)` tuples.

---

## 3. `estimate_threshold` (109-146) — binary search

### Predicate (118-123)
```python
def predicate(p, m, high_threshold):
    if not hasattr(m, "to_quantized"):
        return False
    if sensitivities[p] > high_threshold:
        return {"bits": high_bits, "group_size": high_group_size}
    return True
```
- `False` → don't quantize at all (non-quantizable modules).
- `True` → quantize at the *defaults* passed to `nn.quantize` (which are `low_bits, low_group_size`).
- `dict` → quantize at *override* params (`high_bits, high_group_size`).

### Bounds + tolerance (127-130)
```python
sens_vals = list(sensitivities.values())
min_threshold = min(sens_vals)
max_threshold = max(sens_vals)
tolerance = 1e-3 * (max_threshold - min_threshold)
```
Tolerance is `0.1%` of the range — no fixed iteration cap, just convergence. Iteration count is `ceil(log2(1/tolerance)) ≈ 10` iters regardless of model size.

### Per-iteration cost (131-144)
```python
mid = (max_threshold + min_threshold) / 2
class_predicate = lambda p, m: predicate(p, m, mid)
q_model = copy.deepcopy(model)                  # ← FULL FP MODEL DEEPCOPY EVERY ITER
nn.quantize(q_model, group_size=low_group_size, bits=low_bits, class_predicate=class_predicate)
bpw = compute_bits_per_weight(q_model)          # /opt/mlx-lm/mlx_lm/utils.py:210-215
if bpw > target_bpw: min_threshold = mid
else:                max_threshold = mid
```
**Each iteration deepcopies the entire FP model**, then quantizes it, then computes `compute_bits_per_weight` (a tree-reduce over `nbytes`). The deepcopy is the dominant cost — for a 35B FP16 model that's a fresh ~70 GB allocation per iter × ~10 iters. *However:* `q_model` is a stack-local that goes out of scope at the bottom of each loop iteration, so Python should free it between iters. That depends on no MLX lazy-eval graph holding it; the `compute_bits_per_weight` call is eager (calls `.nbytes`), so it forces materialization, then GC reclaims. Peak still hits at ~2× FP-model size during the deepcopy *plus* the quantize-in-place.

This is the single most memory-hungry part of the pipeline for large models. On 128 GB it would not fit a 70B FP16 model. The script has no `--target-bpw` interval-arithmetic shortcut to skip iterations.

---

## 4. Hyperparameters + defaults (150-185)

| Flag | Default | Meaning |
|---|---|---|
| `--low-bits` | 4 | Bits for *non*-sensitive (majority) layers (line 165) |
| `--low-group-size` | 64 | Group size at low_bits (line 166) |
| `--high-bits` | 5 | Bits for sensitive (minority) layers (line 167) |
| `--high-group-size` | 64 | Group size at high_bits (line 168) |
| `--target-bpw` | 5.0 | Average bits-per-weight including `scales`+`biases` storage (line 163) |
| `--accumulation-dtype` | `float32` | Grad-accumulator precision (line 180-184) |
| `--grad-checkpoint` | False | Recompute layer-0 forward in backward (line 175-178) |
| `--sensitivities` | None | Skip estimate, load JSON (line 156-161) |
| `--seed` | 123 | Calibration sample selection (line 155) |

`target_bpw=5.0` means the *effective* bits per weight averaged across the whole model — including the per-group `scales` (FP16) and `biases` (FP16), which `compute_bits_per_weight` (utils.py:210-215) does count via `nbytes`. So `target_bpw=5.0` with `low=4, high=5` is *barely below pure-5-bit* — most layers stay at 4, and a few (the most sensitive) get bumped to 5; the FP16 scales push the average up to ~5.

LEARNED_QUANTS.md:96 confirms: *"with the default parameters a BPW in the range `[4.5, 5.5]` is achievable."* The achievable range is bounded by all-low (≈4.5 inclusive of scales) and all-high (≈5.5).

---

## 5. Calibration data (utils.py)

```python
save_dir = Path.home() / ".cache/mlx-lm/calibration_v5.txt"          # utils.py:9
url = "https://gist.githubusercontent.com/.../calibration_data_v5_rc.txt"  # :14
tokens = tokenizer.encode(texts, return_tensors="mlx")[0]            # :18
tokens = tokens[: (tokens.size // sequence_length) * sequence_length]
tokens = tokens.reshape(-1, sequence_length)
segments = mx.random.permutation(tokens.shape[0])                    # :23
if num_samples > 0: segments = segments[:num_samples]
return tokens[segments]
```

- Source: `calibration_data_v5_rc.txt` — a community calibration corpus by Tristan Druyen, hosted as a GitHub gist. Lazily downloaded on first use into `~/.cache/mlx-lm/`.
- `num_samples=-1` (dynamic_quant.py:192) means *return all non-overlapping 512-token chunks* (no truncation). Used for the calibration loop.
- `sequence_length=512` (line 192) is hard-coded in the dynamic-quant call site (dynamic_quant.py:192, 213). `dwq.py` and `awq.py` expose this as a flag, but `dynamic_quant.py` does not.
- Random non-overlapping chunks — `mx.random.permutation` over chunk indices, deterministic per `--seed`.

---

## 6. Memory profile (per phase)

| Phase | Resident objects | Peak |
|---|---|---|
| **1. Load** (line 188) | `model` (FP16/BF16), tokenizer | ~2× param-bytes (HF safetensors → MLX) briefly |
| **2. estimate_sensitivities** | `model` (teacher) + `q_model` (deepcopy + qdq'd weights) + `q_layers` (3rd copy of leaf weights) + `grad_accum` (FP32 by default — *2× param-bytes for grads*) + per-batch teacher activations + per-batch student activations + grads | **Worst case** ≈ `model + q_model + grad_accum_fp32 + 2× activations`. For 35B BF16 model: 70 + 70 + 140 + activations = **>280 GB** before optimizations. With `--accumulation-dtype bfloat16` halve grad_accum (~70 GB savings); with `--grad-checkpoint` halve activations. |
| **3. estimate_threshold** | `model` (teacher, retained) + per-iter `q_model = copy.deepcopy(model)` then `nn.quantize` in-place. | ~`2× model FP-bytes` *during* deepcopy; drops to `model + quantized_model` between iters. The quantized snapshot is freed at loop bottom only if no mx-array reference is held (it isn't — `bpw` is a plain float). |
| **4. quantize_model** (line 236) | In-place — `nn.quantize` mutates `model` to swap `nn.Linear` → `nn.QuantizedLinear`. After this `model` shrinks to ~`bpw/16` of original. | Brief 2× during the swap; then permanent reduction. |
| **5. save** (line 248) | Quantized weights + config | Negligible above phase 4. |
| **Reporting** | `mx.get_peak_memory()` line 256 | Reports peak across the full process via MLX's allocator (works because MLX uses a unified allocator on Metal). |

**Implications for 128 GB / 35B-class targets**:
- Phase 2 is the killer. For a 35B BF16 model, naive-default settings need >256 GB.
- `--accumulation-dtype bfloat16` is the cheapest win (~param-bytes saved).
- `--grad-checkpoint` (line 69 wires it onto `q_model.layers[0]` only — not the whole stack; this is suspicious, see Open Questions).
- `--batch-size` is *not exposed via CLI* in dynamic_quant.py — defaults to 4 (line 45) and stays there. To reduce phase-2 peak further you'd need to edit the source.

For hf2q's "memory-bounded execution that won't OOM on 128 GB": the port should expose `--batch-size`, default `--accumulation-dtype=bfloat16`, and *unconditionally* enable grad-checkpointing across all layers (not just `layers[0]`).

---

## 7. Output format

```python
with open(f"{model_name}_sensitivities.json", "w") as fid:    # 205
    json.dump(sensitivities, fid)
```
JSON: `[[layer_path, score], ...]` — list of `[str, float]` pairs. `layer_path` already had `.weight` stripped (line 104). Roundtrip via `json.load` → `dict(sensitivities)` (line 211) keys it for the predicate.

The mixed-bit allocation is **persisted into the model config**, not as side-channel metadata. `quantize_model` (utils.py:823-834) routes per-layer dict overrides into `quantized_config["quantization"][path] = bool_or_params`, then `quantized_config["quantization_config"] = quantized_config["quantization"]` (line 845) for HF-tree compatibility. So at load time, `nn.QuantizedLinear` for each path knows its own `bits`/`group_size`. No external "policy" file is needed once the model is saved. The sensitivity JSON exists purely as a compute-cache for re-runs at different `target_bpw`.

---

## 8. What hf2q needs to port — checklist

1. **`qdq` helper** — quantize→dequantize round-trip in `low_bits` and `high_bits`. hf2q already has the quantize primitives; needs a thin wrapper that returns a same-shape FP tensor.
2. **Layer enumeration matching `to_quantized`** — hf2q must enumerate the same set of leaves mlx-lm does (Linear + Embedding) so paths line up.
3. **KL-divergence loss + autograd path** — this is the heaviest port. Requires: (a) student forward producing logits from QDQ'd weights, (b) reverse-mode autograd through the dequantize affine, (c) batched accumulation with explicit eval/del between batches.
4. **Sensitivity formula** — `(grad * (w_low − w_high)).sum() / param_size_M`. Trivial once gradients exist.
5. **Binary-search threshold** — straight port of lines 127-144. The deepcopy-quantize-bpw loop translates 1:1; budget ~10 iters at ~1× model-size memory per iter.
6. **Sensitivity JSON cache** — invaluable for resumable runs and for hf2q's existing `~/.cache/hf2q/sensitivity/` cache priming. mlx-lm's plain JSON `[[path, score]]` is forward-compatible.
7. **Per-layer config persistence in safetensors metadata + config.json** — hf2q's existing GGUF-emitter path doesn't have a direct HF-style `quantization_config` dict to drop into; it has the synthetic `blk.<i>.sensitivity` tensor (`/opt/hf2q/src/calibrate/dwq_calibrator.rs:68`). The port should keep emitting that synthetic tensor for round-trip with the existing loader.

**Defer**: `--report-ppl` (line 215-217, 244-246) — convenience-only.

---

## 9. Comparison vs hf2q's current DWQ

`/opt/hf2q/src/calibrate/sensitivity.rs:14-80`:
```rust
pub variance: f64,
pub max_magnitude: f64,
// score = sqrt(variance) * log2(1 + max_magnitude)         // line 27
let variance = (mean_sq - mean * mean).max(0.0);            // line 64
let score = variance.sqrt() * (1.0 + max_magnitude).log2(); // line 68
```

`/opt/hf2q/src/calibrate/dwq_calibrator.rs:265-279` shows the cache key tagged `algorithm_version` (`SENSITIVITY_ALGORITHM_VERSION`), keyed e.g. `"1.0.variance-magnitude"`.

| Property | hf2q current | mlx-lm dynamic_quant |
|---|---|---|
| Inputs | Activation tensors (forward-only capture) | Calibration tokens + teacher logits + autograd grads |
| Score | `sqrt(var(act)) * log2(1 + max\|act\|)` | `⟨∇KL/N, w_low − w_high⟩ / params_M` |
| Computation | Forward pass + activation stats | Forward (teacher) + forward+backward (student) per batch |
| Cost | ~1× model forward over calib | ~`(1 + 2)·N_batches` model forwards (teacher fwd, student fwd+bwd) |
| Sign | Always non-negative (sqrt × log) | Signed (positive ≈ "low_bits hurts") |
| Granularity | Per-layer (single scalar) | Per-quantizable-leaf (matches HF-style path keys) |
| Scale invariance | None | Per-million-param normalization |
| What it captures | Activation magnitude/spread proxy for quant-error | First-order Taylor estimate of *actual* KL-loss change |

**Cache invalidation is one-way and permanent**: switching the algorithm changes the sensitivity *ranking* itself — high-variance layers and high-gradient-aligned layers are not the same set in general. Existing cache files keyed `"1.0.variance-magnitude"` cannot be remapped; the port must bump to e.g. `"2.0.gradient-alignment"` and recompute. Re-emitting models from cached sensitivities will produce *different* mixed-bit allocations than fresh recomputes if mismatched. Plan to hard-fail on cache-key version mismatch (don't auto-fall-through).

A practical migration path: keep both algorithms behind a flag (`--sensitivity-algo=variance-magnitude|gradient-alignment`), keep the cache-key versioning rigorous, and treat existing DWQ-46/DWQ-48 emissions as legacy artifacts that don't need byte-for-byte reproduction at the new algorithm.

---

## 10. Open questions (cannot resolve from code alone)

1. **Why default `low_bits=4, high_bits=5`** rather than 4/6 or 4/8 (which would match hf2q's "DWQ-46"/"DWQ-48" naming)? A 4/5 split wastes one bit-plane (5 bits doesn't pack as cleanly as 4 or 8); presumably it's empirically the sweet spot at `target_bpw=5.0` — but there's no comment or commit message in the file justifying it. LEARNED_QUANTS.md:99 just states it as default. Worth a perf comparison at 4/6 + `target_bpw=5.0` vs the default 4/5.
2. **Why `grad_checkpoint(q_model.layers[0])` only?** (line 69). Most autograd implementations need checkpointing applied to every transformer block to get the memory win. Possibly `mlx_lm.tuner.trainer.grad_checkpoint` is implemented to recursively wrap, but reading just dynamic_quant.py I cannot tell. If it only wraps `layers[0]`, this flag does almost nothing on a 60-layer model.
3. **Why `sequence_length=512` hard-coded** (lines 192, 213) when `dwq.py`/`awq.py` expose it as a CLI flag? Memory budget pressure suggests it should be a knob.
4. **Predicate `True` semantic ambiguity**: in `predicate` (line 123) returning `True` means "quantize at `nn.quantize`'s outer defaults" — which is set by the `bits=low_bits, group_size=low_group_size` call at line 137. So `True` → low-bits. If a future caller invoked `nn.quantize` with different outer defaults, the predicate semantics would silently shift. A safer port would always return an explicit dict.
5. **`compute_bits_per_weight` correctness for `nn.QuantizedEmbedding`**: utils.py:202-204 special-cases `hasattr(m, "bits")` for parameter counting (uses `weight.size * 32 / bits` to recover original count) — does this match how hf2q counts shared input/output embeddings? Worth confirming with a tiny model where you can hand-derive the expected BPW.
6. **No exposed `--batch-size`** — the script hardcodes 4 (line 45). For a 35B model on 128 GB this might already exceed memory. Port must expose it.
7. **Distributed init at line 187** (`mx.distributed.init()`) is called but `group` is never used. Either dead code or relies on side-effects — unclear without the distributed runtime context.
