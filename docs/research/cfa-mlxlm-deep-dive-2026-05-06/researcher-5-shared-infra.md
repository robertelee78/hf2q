# Researcher #5 — mlx-lm shared infrastructure for DWQ / dynamic_quant / AWQ / GPTQ

Session: cfa-20260506-mlxlm-research
Scope: read-only deep dive into `mlx_lm/utils.py`, `mlx_lm/tuner/{trainer,datasets,losses,utils}.py`, `mlx_lm/quant/utils.py`.
Cross-cite: how `quant/{dwq,awq,gptq,dynamic_quant}.py` consume each helper, and what hf2q already has versus what's missing.

---

## 1. `load(path_or_hf_repo, *, tokenizer_config=None, model_config=None, adapter_path=None, lazy=False, return_config=False, revision=None)`

**Source:** `/opt/mlx-lm/mlx_lm/utils.py:453-502`. Public surface used by every quant entrypoint (`dwq.py:20`, `awq.py`, `dynamic_quant.py`, `gptq.py`).

### What it actually does

`load` is a thin orchestrator over four primitives:

1. `_download(path_or_hf_repo, revision=revision)` — `utils.py:489` → `utils.py:218-256`. If the path doesn't exist as a local directory, calls `huggingface_hub.snapshot_download` (or `modelscope.snapshot_download` when `MLXLM_USE_MODELSCOPE=true`, gated at `utils.py:27-33`). Default `allow_patterns` are JSON / `model*.safetensors` / `*.py` / tokenizer files / `*.jinja` (`utils.py:237-247`). Returns a `Path` to the snapshot dir.
2. `load_model(model_path, lazy, model_config=model_config)` — `utils.py:491` → `utils.py:282-420`. The real worker; see below.
3. Optionally `load_adapters(model, adapter_path)` (`utils.py:492-494` → `tuner/utils.py:113-138`) which calls `linear_to_lora_layers` from `adapter_config.json` then `model.load_weights(.../adapters.safetensors, strict=False)`.
4. `load_tokenizer(model_path, tokenizer_config, eos_token_ids=config.get("eos_token_id"))` — `utils.py:495-497` → `utils.py:429-450`.

### `lazy` semantics — actually defers materialization, not label-only

`load_model` always calls `model.load_weights(list(weights.items()), strict=strict)` at `utils.py:415` regardless of `lazy`. The MLX semantics are: `load_weights` **assigns** parameters into the module tree but does **not** force computation/materialization in the array system. Materialization happens only when:

- `lazy=False` (default) → `mx.eval(model.parameters())` at `utils.py:417-418`. This forces every parameter to be realized in memory; you pay the full I/O + dequant-prep cost up front.
- `lazy=True` → no `mx.eval`. Arrays exist as lazy graph nodes referencing the safetensors-mapped buffers. First forward pass triggers materialization. Used by `sharded_load` (`utils.py:530`, `utils.py:579`) so a model can be partially constructed, queried for shard layout, then fully loaded.

Note: MLX `safetensors` are memory-mapped (`mx.load(wf)` at `utils.py:323` produces lazy arrays backed by mmap). So `lazy=True` largely defers **read I/O** (page faults) and **dequant prep**, while `lazy=False` walks the whole tree.

### File format expected

- `config.json` (mandatory, read by `load_config` at `utils.py:263-279`)
- Optional `generation_config.json` (eos_token_id pulled out at `utils.py:268-277`)
- One or more `model*.safetensors` files (`utils.py:316`, glob)
- Optional `model.safetensors.index.json` for sharded models (used by `sharded_load`, `utils.py:557-558`)
- Optional `model_file` field in config → loads custom Python module (`utils.py:325-331`, `importlib.util.spec_from_file_location`)

### Quantized-model detection on load

`load_model` walks four config shapes (`utils.py:336-390`):

1. **`config["quantization"]`** present (mlx-lm-native): calls `_quantize` at `utils.py:348-363` which invokes `nn.quantize` with `class_predicate` that allows per-path overrides via `config["quantization"][path]` and gates on the presence of `"{path}.scales"` in weights. Mode defaults to `"affine"`.
2. **`config["quantization_config"]["quant_method"]`** legacy HF schemes (`utils.py:368-390`):
   - `"bitnet"` → `bitnet_quantize(model, quantization_config)` from `models.bitlinear_layers`
   - `"mxfp4"` → fixed `{group_size:32, bits:4, mode:"mxfp4"}`
   - `"compressed-tensors"` → `{group_size:32, bits:4, mode:"affine"}`
   - `"awq"` / `"gptq"` → `_transform_awq_weights(weights, quantization_config)` at `utils.py:387` which **unpacks → transposes → repacks** every `.qweight` tensor (see `_unpack_awq_weights` at `utils.py:72-80` and full transform at `utils.py:83-172`). Critically: **AutoAWQ stores `[in_features, out_features // pack_factor]` while MLX expects `[out_features, in_features // pack_factor]`** (`utils.py:111-112`), and the bit-shifts use the AWQ interleave `[0,4,1,5,2,6,3,7] * bits` (`utils.py:78`). Asymmetric AWQ converts zero-points to MLX biases via `bias = -zero * scale` (`utils.py:144-147`); symmetric uses `-2^(bits-1) * scale` (`utils.py:150-151`). g_idx reshuffling is **explicitly rejected** (`utils.py:95-100`) — non-contiguous group permutations not supported.
3. `config["text_config"]["quantization_config"]` lifted up (`utils.py:336-339`) so VLMs (Qwen2-VL etc.) work with the same pathway.
4. `config["quantize_activations"]` triggers a post-hoc rewrite of every `nn.QuantizedLinear` into `nn.QQLinear` (activation-quantized, only `nvfp4`/`mxfp8` modes) — `utils.py:392-412`.

### Memory profile

- **Resident:** the model parameters after `mx.eval` (lazy=False) or the empty graph (lazy=True). Weights dict at `utils.py:321-323` holds **all** safetensors entries simultaneously before `model.load_weights` consumes them — this is a one-time spike. For an AWQ model the spike is 2× (original packed + repacked at `utils.py:153-155`) until `new_weights` overwrites and the GC drops the originals.
- **Streamed:** safetensors are mmap'd; OS handles paging. The `RLIMIT_NOFILE` is bumped to `(2048, 4096)` at module import (`utils.py:36`) because large MoE models can have 50+ shards.

### Hf2q equivalent

hf2q does **not** ingest HF-format safetensors at runtime. Its load path is GGUF-only:

- `src/backends/gguf.rs` (the load path; supersedes `gguf_patch.rs` for new code) reads GGUF tensors with magic + version dispatch — see standing memory `project_qwen35_dwq_pre_505b5b8_broken_2026_05_05.md` for the vocab-truncation fix landed at `gguf.rs:2647+2706-2799`.
- No HF Hub `snapshot_download` integration; users either supply a local GGUF or use the conversion path (`src/quantize/`).
- AWQ/GPTQ unpack-transpose-repack: hf2q's `convert_qwen35_*` family roughly fills the same role on the convert side, but at runtime nothing in hf2q re-formats AWQ packed layouts. **Gap if the V3 plan wants to ingest HF-AWQ/HF-GPTQ models directly: would need a Rust port of `_transform_awq_weights` (`utils.py:83-172`, ~90 LOC including the AWQ↔MLX shift-table re-interleave).**
- `lazy` materialization: hf2q's GGUF loader is essentially eager — it owns `Vec<u8>` blocks for k-quant codebooks. Closest analog to `lazy=True` is the streaming-in-progress phase-3 gating discussed in `project_qwen35_reconvert_paused_2026_05_05.md` (HF2Q_STREAMING_PHASE3=1).

---

## 2. `save(dst_path, src_path_or_repo, model, tokenizer, config, donate_model=True)` and `save_model` / `save_config`

**Source:** `utils.py:925-950` (`save`), `utils.py:714-771` (`save_model`), `utils.py:899-922` (`save_config`).

### Output file layout

`save` orchestrates:

1. `save_model(dst_path, model, donate_model=True)` — writes shards.
2. `save_config(config, dst_path / "config.json")` — JSON config, sorted keys.
3. `tokenizer.save_pretrained(dst_path)` — HF transformers tokenizer dump.
4. Copies any `*.py` and `generation_config.json` from the source HF repo (`utils.py:946-948`) so custom model code travels with the artifact.
5. `create_model_card(dst_path, hf_repo)` — `utils.py:622-645` writes README.md with `library_name=mlx`, `tags=["mlx"]`.

`save_model` actual layout (`utils.py:714-771`):

- `weights = dict(tree_flatten(model.parameters()))` — `utils.py:725`. Flattens param tree to dotted-key dict.
- `make_shards(weights, max_file_size_gb=5)` — `utils.py:598-619`. **Greedy bin-packing** by `nbytes`; constant `MAX_FILE_SIZE_GB=5` at `utils.py:57`. Returns list of `{key: array}` dicts.
- Filename format: `model.safetensors` if 1 shard, else `model-{:05d}-of-{:05d}.safetensors` (`utils.py:728-732`).
- Writes via `mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})` (`utils.py:756`). The `"format": "mlx"` metadata is the canary; it tells subsequent loaders that biases/scales follow MLX layout (vs AWQ/GPTQ).
- Writes `model.safetensors.index.json` with `{"metadata":{"total_size","total_parameters"},"weight_map":{...}}` (`utils.py:735-741`, `utils.py:766-771`).

### Quantized-weight on-disk encoding

`mx.save_safetensors` doesn't know quantization — it just serializes whatever `mx.array`s are in the dict. For a `nn.QuantizedLinear`, the model's `parameters()` returns:

- `weight`: packed `uint32` array, shape `[out_features, in_features // pack_factor]` where `pack_factor = 32 // bits`
- `scales`: per-group fp16/bf16 scales, shape `[out_features, in_features // group_size]`
- `biases` (affine modes): per-group fp16/bf16 biases, same shape as scales

So no special "quantized writer" — just whatever the module exposes. Round-trip works because `nn.quantize` (called from `load_model::_quantize` at `utils.py:357-363`) installs `QuantizedLinear` modules that consume these exact key suffixes via the `class_predicate` `f"{p}.scales" in weights` gate (`utils.py:355`).

### `config.json["quantization"]` schema

Two cooperating keys (`utils.py:914-915` mirrors them):

```json
{
  "quantization": {
    "group_size": 64,
    "bits": 4,
    "mode": "affine",
    "model.layers.0.self_attn.q_proj": {"group_size": 32, "bits": 8},   // optional per-path
    "model.layers.10.mlp.gate": false                                    // skip
  },
  "quantization_config": { /* mirror of quantization, for HF tooling */ }
}
```

The per-path overrides are written by `quantize_model::wrapped_predicate` at `utils.py:823-835` — when `quant_predicate` returns a dict, it's stored under the layer path; when it returns `True` and the model already had a `quantization` key (`fine_grained_config=True`, `utils.py:818`), the global params get pinned per-layer too.

### `donate_model=True` semantics

`utils.py:742-748`: replaces every parameter with `mx.array([])` so the array refs drop and Python GC can free the safetensors-mapped backing memory **between shard writes**. Critical for low-RAM CI; without this you'd hold the entire model + all written shards simultaneously. Each shard is also nulled in-place (`shards[i] = None`, `utils.py:752`) for the same reason.

### Hf2q equivalent

- Round-tripping convert: `src/quantize/k_quant_codec.rs` is the GGUF k-quant codec (frozen on-disk format). hf2q **writes GGUF, not safetensors**, so there's no direct port required.
- Sharding: GGUF is single-file (no `model-00001-of-N.safetensors` analog). If the V3 plan keeps GGUF, no shard code needed.
- `donate_model` analog: GGUF writer is already streaming, so memory pressure is naturally lower (see `project_qwen35_reconvert_paused_2026_05_05.md` — HF2Q_STREAMING_PHASE3 trades streaming-ness for clone-into-Arc which **increases** peak; the `_MUT` variant is the lower-peak path).
- `config.json["quantization"]` schema equivalent: GGUF metadata keys — `general.quantization_version`, `general.file_type`, plus per-tensor types in the tensor table. No per-layer override JSON; instead each tensor in the GGUF carries its own type.
- Model card / README.md: not generated by hf2q; would need to be added if HF Hub publishing is in scope.

---

## 3. `quantize_model(model, config, group_size, bits, mode="affine", quant_predicate=None)`

**Source:** `utils.py:774-850`.

### Signature + defaults

- `group_size: Optional[int]` — None falls back to mode-default at `utils.py:800-808`: `affine→64`, `mxfp4→32`, `nvfp4→16`, `mxfp8→32`.
- `bits: Optional[int]` — None falls back to mode-default: `affine→4`, `mxfp4→4`, `nvfp4→4`, `mxfp8→8`.
- `mode: str = "affine"` — string; passed through to `nn.quantize`.
- `quant_predicate: Callable[(path:str, module) -> Union[bool, dict]] | None` — caller can override per-layer; if `None`, falls through to `getattr(model, "quant_predicate", None)` at `utils.py:812` so families like Gemma can ship a default predicate as a class attr.

### Algorithm

`utils.py:823-836` defines `wrapped_predicate(path, module)`:

1. Gates on `hasattr(module, "to_quantized")` — only quantizable layers (Linear, Embedding, SwitchLinear).
2. Gates on `module.weight.shape[-1] % group_size == 0` — refuses non-divisible shapes (returns False, layer stays fp).
3. Calls user `quant_predicate(path, module)` if provided.
4. If user returned a dict → record per-layer params in `quantized_config["quantization"][path]` (`utils.py:831-832`).
5. If user returned `True` and the config is "fine-grained" (model already had per-layer entries), record the global params for this path too (`utils.py:833-834`).
6. Returns the user value (bool or dict) to `nn.quantize`.

`nn.quantize(model, group_size, bits, mode=mode, class_predicate=wrapped_predicate)` (`utils.py:837-843`) **mutates the model in-place** — replaces every `nn.Linear` for which the predicate fired with `nn.QuantizedLinear`, computing scales+biases from the current weights. Returns `(model, quantized_config)` where `quantized_config["quantization_config"]` is also written (`utils.py:845`) for HF tree compatibility.

`compute_bits_per_weight(model)` is run + printed (`utils.py:847-848`) so the operator sees the achieved BPW post-quantization.

### Mixed-bit handling (the dict return)

The `dict` return path is **the critical mechanism** for `dynamic_quant.py` / `awq.py` / `dwq.py`:

```python
def my_predicate(path, module):
    if "self_attn" in path:
        return {"group_size": 32, "bits": 8}  # 8-bit attn
    if "mlp" in path:
        return {"group_size": 64, "bits": 3}  # 3-bit MLP
    return False  # skip embeddings, norms
```

When `wrapped_predicate` sees the dict, it stores it on `quantized_config["quantization"][path]` (`utils.py:831-832`), and **`nn.quantize` itself reads the dict to override `group_size`/`bits` for that specific module**. The global args become defaults.

### In-place vs new-model

In-place mutation (`nn.quantize` rewrites `model.update_modules(...)`). Returns the same `model` object plus a freshly-deep-copied config (`copy.deepcopy(config)` at `utils.py:810`).

### Hf2q equivalent

- `src/quantize/mod.rs` is the orchestration layer; `src/quantize/mixed.rs` and `src/quantize/layer_mix.rs` are the per-layer mixing logic — direct analog of the `dict` return from `quant_predicate`.
- `src/quantize/static_quant.rs` is the eager-quantize path (analog of `nn.quantize` with a fixed predicate).
- `src/quantize/dwq_k_quantizer.rs` is the DWQ-specific quantizer (consumes per-layer sensitivity from `~/.cache/hf2q/sensitivity/*` — see `project_qwen35_reconvert_paused_2026_05_05.md`).
- `src/quantize/k_quant.rs` and `src/quantize/k_quant_codec.rs` are the codec layer (closest analog to `nn.QuantizedLinear`'s compute kernel + on-disk encoding).
- **Gap:** mlx-lm's `quant_predicate` is a runtime callable; hf2q uses a static config + sensitivity-table lookup. The flexibility difference matters if V3 wants user-supplied predicates without a recompile — likely needs a Rust trait-object or a small DSL.

---

## 4. `pipeline_load(repo, return_config=False)` and `sharded_load`

**Source:** `utils.py:594-595` (`pipeline_load`), `utils.py:505-591` (`sharded_load`).

### Architecture

`pipeline_load` is a one-line wrapper: `sharded_load(repo, mx.distributed.init(), None, return_config)`. It calls `mx.distributed.init()` to bring up the MPI/Thunderbolt-ring world (`utils.py:548-550` shows the lazy init when neither group is given) and uses pipeline parallelism only.

Pipeline-parallel layer dispatch:

- Model must implement `model.model.pipeline(group)` (gated at `utils.py:532` via `hasattr`). Each rank calls `model.model.pipeline(group)` (`utils.py:554`) which **rewires the layer list** so only this rank's slice of layers is kept; the others are replaced with identity / shape-preserving stubs.
- `model.safetensors.index.json` is then read (`utils.py:557-558`); for each parameter key in the local-shard `model.parameters()` tree the index gives the file containing it, so only those shards are downloaded (`utils.py:560-569`).
- Weights are loaded a second time after sharding (`utils.py:579`) so the local rank only allocates the layers it owns.
- Tensor-parallel path is parallel: `model.shard(group)` at `utils.py:580-581` — split per-tensor (each rank owns a slice of a single weight along some dim).
- Mixed: model can support both, but `pipeline_load` only invokes pipeline.

### Per-rank what gets loaded

- Tokenizer: full copy on every rank (`utils.py:574-578`).
- Config: full copy.
- Weights: only the safetensors files containing the rank-local parameter keys (`local_files` set at `utils.py:560-566`).
- Final `mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))` at `utils.py:587` is a synchronization barrier so processes don't time out racing through `mx.eval(model.parameters())` (`utils.py:584`).

### `mx.distributed.init()` usage

Returns a `Group` representing the world. Called explicitly at `utils.py:548-550` (default-init a group when caller didn't supply one) and at `utils.py:594` (`pipeline_load` shortcut). The all-sum barrier on `mx.cpu` is deliberate — the GPU stream may be mid-eval when the barrier runs.

### Hf2q equivalent

**No equivalent exists.** hf2q is single-process, single-GPU. Pipeline parallel + tensor parallel multi-node serving would be a from-scratch addition. Closest design surface: the engine could be made shardable along the same lines (per-rank layer slice, weight-key→shard-file map). But `mx.distributed.init()` has no Rust analog in hf2q today.

---

## 5. `iterate_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None)`

**Source:** `tuner/trainer.py:102-173`.

### Algorithm

1. Sort all dataset indices by length (`tuner/trainer.py:111-115`). For `CacheDataset` uses `itemlen(idx)` (`datasets.py:163-164`); else `len(dataset[idx][0])` — i.e. token count.
2. Distributed offsetting: each rank takes `idx[i+offset : i+offset+batch_size : step]` where `step=comm_group.size()` and `offset=comm_group.rank()` (`trainer.py:124-137`). Batch size **must** be divisible by world size (`trainer.py:130-131`).
3. Pre-build all batch index lists (`trainer.py:134-137`).
4. Permute with `np.random.permutation(len(batch_idx))` after `np.random.seed(seed)` if seed provided (`trainer.py:138-141`). Critical: numpy seed is used, **not** mlx — because `iterate_batches` is data-only and doesn't touch the GPU randomness state.
5. For each batch: process samples (returns `(tokens, offset)` 2-tuples or just tokens — see `trainer.py:144-147`).

### Padding + masking

- Padding constant `pad_to=32` at `trainer.py:157` — round max-length-in-batch up to nearest multiple of 32, plus 1 (so the autoregressive `inputs[:, :-1]` and `targets[:, 1:]` slicing still has aligned dims). Hard-coded; no arg.
- `batch_arr = np.zeros((B, L), np.int32)` (`trainer.py:161`) — zero-padded.
- Truncation to `max_seq_length` (`trainer.py:164-165`) with a printed warning (`trainer.py:149-154`).
- Lengths after truncation written back (`trainer.py:166-168`) so the loss mask is correct.

Yields `(batch_array, lengths_array)` where `lengths_array` is `mx.array(zip(offsets, lengths))` shape `[B, 2]` (`trainer.py:170`).

### Mask interaction with the loss

`default_loss` at `trainer.py:86-99` builds the mask:

```python
steps = mx.arange(1, targets.shape[1] + 1)
mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
```

This is the prompt-mask + pad-mask combined: token positions before `offset` (the prompt portion when `mask_prompt=True`, see `datasets.py:65-77`) are zeroed, and positions beyond the true length are zeroed. CE is summed and divided by `mask.sum()` (`trainer.py:97-98`).

### Generator vs list

It's a Python `yield`-generator (`trainer.py:170`). Memory profile: only the current batch's `mx.array` is resident; sample bytes for upcoming batches stay in the dataset Python list. For `loop=True` (training) the outer `while True` (`trainer.py:140`) keeps yielding forever; `loop=False` (eval) breaks after one pass (`trainer.py:172-173`).

### Reproducibility

`seed` truthy → numpy seed reset every entry into the function. **Not deterministic across distributed ranks unless all ranks pass the same seed** (each rank picks its own slice via offset/step, so they get disjoint subsets in a deterministic split).

### Hf2q equivalent

hf2q has no training/fine-tuning code path — only inference + offline quantization. Closest patterns:

- Calibration sample iteration in `quant/utils.py:8-26` (mlx side) is much simpler: chunked random non-overlapping segments, `mx.random.permutation` (`utils.py:23`).
- hf2q's calibration path (e.g. for k-quant scale fitting in `src/quantize/k_quant.rs`) reads tokenizer-encoded calibration data once and slices it. No yielding generator, no pad-to-32, no distributed split.
- **Gap if hf2q wants to add fine-tuning:** would need length-sorted bucket-batching with `pad_to=32`, prompt-offset masks. ~150 LOC of Rust.

---

## 6. `grad_checkpoint(layer)`

**Source:** `tuner/trainer.py:25-38`.

### What it actually does

```python
def grad_checkpoint(layer):
    fn = type(layer).__call__
    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)
        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)
    type(layer).__call__ = checkpointed_fn
```

Monkey-patches **the class** of `layer` (not the instance). Every future call to any instance of `type(layer)` will:

1. Call `mx.checkpoint(inner_fn)` — MLX's recompute-on-backward primitive.
2. `inner_fn(params, ...)` — the wrapped function takes `params` as its first arg so MLX's autograd can re-feed the trainable params at backward time.
3. `model.update(params)` re-installs the params before re-running `fn(model, ...)`.

Net effect: forward activations for this layer's computation graph are **not** held; backward recomputes them.

### Memory savings vs compute cost

Standard checkpointing trade: ~2× compute on the patched layer (forward runs once during forward + once during backward), but activation memory drops by the layer's intermediate-activation size. For a transformer block, that's the attention scratch + MLP intermediate — typically the dominant activation cost.

### Why `grad_checkpoint(model.layers[0])` only?

`dwq.py:104` patches only `model.layers[0]`. The trick: because `grad_checkpoint` rewrites `type(layer).__call__`, **every layer of the same Python class is now checkpointed** — not just instance 0. Since transformer blocks all share one class, one call covers the whole stack. Same idiom in `trainer.py:237-238`.

### Hf2q equivalent

hf2q has no autograd / training path. No equivalent. If V3 adds fine-tuning, gradient checkpointing would need a Rust reverse-mode autograd primitive — non-trivial. Most likely route: leverage mlx-native's autograd (which is C++-side) and expose checkpoint via FFI rather than reimplement.

---

## 7. `kl_div_loss(student_logits, teacher_logits)`

**Source:** `tuner/losses.py:377-386` (public), `tuner/losses.py:11-176` (Metal forward kernel), `tuner/losses.py:179-337` (Metal backward kernel), `tuner/losses.py:344-374` (custom-VJP wrapper).

### Formula

`kl_div_loss(logits_q, logits_p)` computes **KL(p || q)** per-row (last axis = vocabulary):

```
kl = sum_i softmax(p)_i * (log_softmax(p)_i - log_softmax(q)_i)
   = sum_i exp(p_i - lse_p) * (p_i - q_i + lse_q - lse_p)
```

The Metal kernel literally computes `metal::fast::exp(vals_p[j] - lse_p) * (vals_p[j] - vals_q[j] + lse_q_minus_p)` (`losses.py:135-136`). Standard form, **p is the teacher** (mass weighting comes from softmax(p)). In `dwq.py:114` the call is `kl_div_loss(scale * logits, scale * targets)` where `targets = teacher_logits` — so the second arg is `p` = teacher.

Important sign convention check: at `losses.py:117` `lse_q_minus_p = max_q + log(sum_exp_q) - lse_p`. Then `(vals_p[j] - vals_q[j] + lse_q_minus_p) = (p_i - lse_p) - (q_i - lse_q) = log(softmax(p)_i / softmax(q)_i)`. Multiplied by `exp(p_i - lse_p) = softmax(p)_i` and summed — yes, **forward KL(p || q)**, the standard distillation loss with teacher=p.

### Temperature handling

Not in `kl_div_loss` itself — done at the call site. `dwq.py:106` sets `scale = 1 / temperature` (with `temperature: float = 2.0` default at `dwq.py:80`), and `dwq.py:114` calls `kl_div_loss(scale * logits, scale * targets)` so both student and teacher are scaled identically before softmax. This is the standard Hinton-distillation trick: higher T → smoother distributions → more weight on the dark-knowledge tail.

### Token-level vs sequence-level reduction

**Token-level returns**, no reduction. The Metal kernel's output shape is `logits_q.shape[:-1]` (`losses.py:351`) — i.e. one KL scalar per row of the input logits tensor. Caller decides whether to mean/sum — `dwq.py` typically does `loss.mean()` over the masked tokens.

### Numerical stability

Two-pass online algorithm (`losses.py:33-118`):
- Pass 1: streaming max + sum-exp simultaneously for both q and p, with `prev_max → max` rescaling so old partial sum-exps stay valid.
- Threadgroup-shared simd-reduction at `losses.py:86-114` to combine across simdgroups within a tile.
- Pass 2 (`losses.py:122-167`): re-reads logits, computes kl using the now-final `lse_p` and `lse_q_minus_p`, simd_sum across threads, writes one scalar.

Vocabulary chunked into `block = 1024 * M = 4096` elements per simd-rake (`losses.py:16-18`), with `extra` tail handled separately.

### Custom VJP

`@_kl_div_loss.vjp` at `losses.py:359-374`. Backward gives `dq = c * (softmax(q) - softmax(p))` (`losses.py:307`), `dp = zeros_like(logits_p)` (teacher gradient discarded — `dp = mx.zeros_like(logits_p)` at `losses.py:364`). This matches the math: `d KL(p||q) / d q_i = softmax(q)_i - softmax(p)_i` and we don't propagate through the frozen teacher.

### CPU/Metal fallback

`losses.py:378-386`: when `can_run_metal()` is false, falls back to `nn.losses.kl_div_loss` with explicit log-softmax pre-norm, axis=-1, reduction="none". The Metal kernel saves the log-softmax materialization (which would be a separate vocab-sized tensor pass).

### Hf2q equivalent

No KL loss in hf2q. **The Metal kernel itself is the hard part to port** — ~200 LOC of templated Metal that mlx-native could borrow as-is or wrap via FFI. The CPU fallback at `losses.py:381-386` is trivial to write in Rust against `mlx-rs` arrays. If V3 plans to do DWQ in-process, this kernel should be lifted to mlx-native (where it already could compile alongside flash_attn_prefill.metal and the rest of the kernel set in /opt/mlx-native/mlx_native/kernels/).

---

## 8. `print_trainable_parameters(model)`

**Source:** `tuner/utils.py:160-168`.

### What it counts

```python
total_p = get_total_parameters(model) / 1e6
trainable_p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 1e6
print(f"Trainable parameters: {(trainable_p * 100 / total_p):.3f}% ({trainable_p:.3f}M/{total_p:.3f}M)")
```

- `total_p` uses `get_total_parameters` (`utils.py:196-207`) which **counts quantized weights at their de-quantized parameter count** (`m.weight.size * 32 // m.bits`) — see below in compute_bits_per_weight.
- `trainable_p` uses `model.trainable_parameters()` — the un-frozen leaves. For LoRA fine-tuning that's just the LoRA A/B matrices; for DWQ it's the scales+biases of QuantizedLinear (see `dwq.py:97`: `m.unfreeze(keys=["scales", "biases"], recurse=False)`).
- Both shown in millions; ratio printed as %.

### Hf2q equivalent

None. hf2q has no concept of "trainable" vs "frozen" because there's no training path. If/when added: this is a 5-line helper.

---

## 9. `load_dataset(args, tokenizer)`

**Source:** `tuner/datasets.py:309-332`.

### Decision tree

1. `args.hf_dataset` truthy → `load_custom_hf_dataset(args, tokenizer)` (`datasets.py:249-306`). Handles a list-of-dicts spec, each with `path`, `train_split`, `valid_split`, `test_split`, `mask_prompt`, plus arbitrary `config` kwargs forwarded to `datasets.load_dataset(...)`. Multiple datasets get wrapped in `ConcatenatedDataset` (`datasets.py:136-155`) which rerouters `__getitem__` to the right backing dataset by index arithmetic.
2. Else: `Path(args.data)` exists locally → `load_local_dataset` (`datasets.py:205-219`) reads `train.jsonl` / `valid.jsonl` / `test.jsonl` line-delimited JSON.
3. Else: `load_hf_dataset(args.data, ...)` (`datasets.py:222-246`) — `datasets.load_dataset(data_id)` with default splits.

### Output format

Three of `(TextDataset | ChatDataset | CompletionsDataset | ConcatenatedDataset)`. The format is auto-detected by `create_dataset` (`datasets.py:175-202`):

- `prompt_feature` + `completion_feature` keys present → `CompletionsDataset` (`datasets.py:86-133`)
- `chat_feature` (default `"messages"`) → `ChatDataset` (`datasets.py:39-83`)
- `text_feature` (default `"text"`) → `TextDataset` (`datasets.py:11-37`)

Each dataset's `process(d)` method tokenizes lazily and returns `(tokens: List[int], offset: int)` where `offset` is either `0` (no prompt mask) or `len(prompt_tokens_after_chat_template)` (prompt-masked, `datasets.py:65-75`).

`CacheDataset` (`datasets.py:158-172`) is the lazy-tokenize wrapper used by `iterate_batches`'s sort-by-length path: it tokenizes on first `__getitem__` and caches; `itemlen(idx)` returns raw byte length (avoiding tokenization just for sorting).

### HF Hub support

Yes via `args.hf_dataset` or via `args.data` falling through to `load_hf_dataset`. Both ultimately call `datasets.load_dataset(...)` from the HF `datasets` library — which itself does HF Hub download + caching.

### Hf2q equivalent

No fine-tuning data path. Closest analog: calibration corpus loading at `quant/utils.py:8-26` reads a single text file from `~/.cache/mlx-lm/calibration_v5.txt` (downloaded via urllib if missing). The hf2q `~/.cache/hf2q/sensitivity/` is a totally different artifact (per-layer DWQ sensitivity, mostly debug stubs per the standing memory).

---

## 10. `compute_bits_per_weight(model)` and `get_total_parameters(model)`

**Source:** `utils.py:196-215`.

```python
def get_total_parameters(model):
    leaf_modules = tree_flatten(model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module))
    def nparams(m):
        if hasattr(m, "bits"):
            n = 0 if not hasattr(m, "bias") else m.bias.size
            return n + m.weight.size * 32 // m.bits
        return sum(v.size for _, v in tree_flatten(m.parameters()))
    return sum(nparams(m) for _, m in leaf_modules)

def compute_bits_per_weight(model):
    model_bytes = tree_reduce(lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0)
    model_params = get_total_parameters(model)
    return model_bytes * 8 / model_params
```

### What it does

- `get_total_parameters`: walks leaf modules, returns the **logical (de-quantized) parameter count**. For a quantized layer (`hasattr(m, "bits")`), `m.weight.size` is the *packed uint32 element count*; multiplying by `32 // bits` recovers the original element count. Bias size added if present.
- `compute_bits_per_weight`: total bytes resident (counts scales, biases, packed weights, residual fp16 layers) × 8 bits/byte ÷ logical parameter count.

So a 4-bit quant with fp16 scales every group=64 has: `(weight_bytes_4bit + scale_bytes_fp16/64) * 8 / nparams ≈ 4.5 BPW`. This is the metric printed at `utils.py:848` after `nn.quantize`.

### Used in `estimate_threshold`'s binary search

Used by `dynamic_quant.py` to pick a per-layer mixed-bit cocktail that meets a target BPW: binary-search the cutoff threshold on the layer-sensitivity histogram, recompute BPW after each candidate, converge.

### Hf2q equivalent

- `src/report.rs` (likely emits the equivalent BPW summary post-conversion — would need to confirm).
- `src/preflight.rs` reasons about RAM budget pre-load, which uses an analogous bytes-per-param estimate.
- The mixed-bit binary search equivalent lives in `src/quantize/mixed.rs` / `src/quantize/layer_mix.rs` (see file list above). Direct port semantically; the only nuance is hf2q's "bytes" come from k-quant block sizes (Q4_K = 144 bytes per 256-element block etc.) rather than mlx's `(packed_weight_bytes + scale_bytes + bias_bytes)`.

---

## 11. Cross-cutting: dependency edges between these utilities and the four quant scripts

| Utility | Used by | Citation |
|---|---|---|
| `load` (utils.py:453) | dwq, awq, gptq, dynamic_quant | dwq.py:20 imports |
| `save` (utils.py:925) | all four | dwq.py:20 imports |
| `quantize_model` (utils.py:774) | dynamic_quant primarily; awq/gptq for re-quant | dwq.py:20 imports |
| `compute_bits_per_weight` (utils.py:210) | quantize_model internal + dynamic_quant binary search | utils.py:847 |
| `iterate_batches` (trainer.py:102) | dwq calibration loop | dwq.py:18 |
| `grad_checkpoint` (trainer.py:25) | dwq | dwq.py:18, dwq.py:104 |
| `kl_div_loss` (losses.py:377) | dwq, dynamic_quant (KD-style refinement) | dwq.py:17, dwq.py:114 |
| `load_data` (quant/utils.py:8) | calibration sample fetcher for dwq/awq/gptq | local helper |
| `load_dataset` (datasets.py:309) | tuner-side LoRA, also dynamic_quant calibration data when `--data` passed | tuner/lora.py |
| `print_trainable_parameters` (tuner/utils.py:160) | dwq's "scales+biases unfrozen" report | dwq.py |

The whole DWQ flow (`quant/dwq.py`) is essentially:

1. `load(...)` → model + tokenizer
2. Freeze all, then `m.unfreeze(keys=["scales","biases"], recurse=False)` per quantized module (dwq.py:97)
3. `grad_checkpoint(model.layers[0])` (dwq.py:104) to halve activation memory
4. `quant/utils.load_data` (calibration)
5. Loop over `iterate_batches(...)` yielding teacher-vs-student logits, compute `kl_div_loss(scale * student, scale * teacher)` (dwq.py:114), gradient step on scales+biases only
6. `save(...)` to write the refined-quantized model

---

## 12. Summary: where hf2q stands

| mlx-lm surface | hf2q analog | Status |
|---|---|---|
| `load` (HF safetensors + HF Hub + AWQ/GPTQ unpack) | `src/backends/gguf.rs` (GGUF only) | **Different format**; AWQ/GPTQ direct ingest is a gap if needed |
| `save` (sharded safetensors + index.json + model card) | GGUF writer in `src/quantize/k_quant_codec*.rs` | Format differs, sharding+index N/A for GGUF |
| `quantize_model` + per-layer dict predicate | `src/quantize/{mixed,layer_mix,static_quant}.rs` | Equivalent in capability; predicate API is config-driven not callable |
| `pipeline_load` / `sharded_load` | none | **Missing** entirely; multi-rank serving not in scope today |
| `iterate_batches` | none (no fine-tune path) | Missing; trivial port if needed |
| `grad_checkpoint` | none | Missing; needs autograd, lift to mlx-native |
| `kl_div_loss` Metal kernel | none | **The expensive port** — 200 LOC Metal; lift to mlx-native kernel set |
| `print_trainable_parameters` | none | Trivial helper |
| `load_dataset` (jsonl + HF datasets) | `quant/utils.py` calibration corpus only | Missing for FT use case |
| `compute_bits_per_weight` | implicit in `src/report.rs` + `src/quantize/mixed.rs` | Equivalent semantics on GGUF block sizes |

**Standing recommendations for V3 planning** (citing standing memories where relevant):

1. AWQ/GPTQ ingest path is non-trivial but isolated (`utils.py:72-172`, ~100 LOC). Cleanest port target if HF compatibility needed.
2. `kl_div_loss` Metal kernel should be ported to mlx-native, not hf2q-side — kernel-set ownership boundary already exists per ADR-013/ADR-015 territory (`solution_mlx_native_residency_lifetime_race.md`).
3. Mixed-bit predicate API: hf2q's static config + sensitivity table is fine for V3; only escalate to runtime callable if user-driven mix-tuning is a roadmap item.
4. Sharded/pipeline serving is a multi-week effort; punt unless V3 specifically targets multi-node Apple Silicon.
5. Per the multi-bin hf2q lesson (`project_hf2q_no_lib_target_unit_test_friction.md`), if any of these utilities get ported, lift them into a `quant/` sub-crate with a `[lib]` target so they can be unit-tested without `assert_cmd` ceremony.

---

**File paths referenced:**
- `/opt/mlx-lm/mlx_lm/utils.py`
- `/opt/mlx-lm/mlx_lm/tuner/trainer.py`
- `/opt/mlx-lm/mlx_lm/tuner/datasets.py`
- `/opt/mlx-lm/mlx_lm/tuner/losses.py`
- `/opt/mlx-lm/mlx_lm/tuner/utils.py`
- `/opt/mlx-lm/mlx_lm/quant/utils.py`
- `/opt/mlx-lm/mlx_lm/quant/dwq.py` (consumer cross-reference)
- `/opt/hf2q/src/quantize/{mod,mixed,layer_mix,static_quant,k_quant,k_quant_codec,dwq_k_quantizer}.rs`
- `/opt/hf2q/src/backends/gguf.rs`
- `/opt/hf2q/src/{report,preflight}.rs`
