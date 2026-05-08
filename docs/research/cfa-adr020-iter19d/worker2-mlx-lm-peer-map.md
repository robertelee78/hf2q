# mlx-lm Peer Code Map for hf2q DWQ Port (CFA worker 2)

CFA session: `adr020-iter19d` · worker 2 of 5
Peer source root: `/opt/mlx-lm/mlx_lm/`
Peer install (mlx wheel): `/Users/robert/.pyenv/versions/3.13.12/lib/python3.13/site-packages/mlx/`

This document maps every load-bearing surface of mlx-lm's full-model
DWQ + KLD code paths so hf2q can make informed build/port/wrap
decisions. Worker 3 will mirror this on the hf2q side.

---

## 1. Top-level dispatch — `mlx_lm.dwq.main()`

**File**: `/opt/mlx-lm/mlx_lm/quant/dwq.py:242-411`
**Console script**: `mlx_lm.dwq = mlx_lm.quant.dwq:main` (`/opt/mlx-lm/setup.py:56`).

Step-by-step from CLI args to saved model:

1. **Argparse** (`dwq.py:243-303`) — flags worth porting verbatim:
   - `--model` / `-m` (required) — base model HF repo or path used for distillation **and** as student source if `--quantized-model` is omitted.
   - `--quantized-model` (optional) — pre-quantized student to "polish".
   - `--mlx-path` (default `mlx_model`) — output dir.
   - `--bits` (default 4), `--group-size` (default 64).
   - `--num-samples` (default 2048), `--max-seq-length` (default 1025), `--seed` (default 123).
   - `--learning-rate` (default 1e-6), `--batch-size` (default 4).
   - `--data-path` (default `allenai/tulu-3-sft-mixture`).
   - `--grad-checkpoint` (flag) — wraps `model.layers[0]`.
   - `--target-dir` — disk cache for top-1024 logits/indices.
   - `--targets-only` — pre-bake targets and exit.
   - `--pipeline` — pipeline-parallel teacher load (vs data-parallel).

2. **Distributed init** (`dwq.py:306`): `mx.distributed.init()`. Single-host runs return rank-0 size-1 group.

3. **Sample-count rounding** (`dwq.py:308-310`): when not pipelining, rounds `num_samples` up to the next multiple of `group.size()` so each worker gets equal shards.

4. **Seeding** (`dwq.py:312-313`): both `np.random.seed(seed)` and `mx.random.seed(seed)`.

5. **Target-cache probe** (`dwq.py:315-324`): if `--target-dir` exists with both `train/*.safetensors` and `valid/*.safetensors`, `has_targets=True`; otherwise the teacher will be invoked.

6. **Tokenizer load** (`dwq.py:326`): `tokenizer = load_tokenizer(args.model)`. Tokenizer comes from the **base** model — not the quantized one — so vocab matches the teacher. (Critical: `load_tokenizer` only downloads `*.json/*.py/tokenizer.model/*.txt/*.jsonl/*.jinja` — no weights — see `utils.py:429-450`.)

7. **Calibration data load** (`dwq.py:328-330`): `train_data, valid_data = load_data(tokenizer, data_path, num_samples, max_seq_length)` — see §6.

8. **Teacher load** (`dwq.py:333-339`): only if we still need it (no cached targets, **or** student needs to be derived from teacher).
   - Pipeline mode: `pipeline_load(args.model, return_config=True)` → `(model, tokenizer, config)`.
   - Data-parallel: `load(args.model, return_config=True, lazy=True)` → same.
   - **`lazy=True` is mandatory** here — loads the safetensors into mx but doesn't `mx.eval(model.parameters())`. Saves wired-memory pressure on a 35B teacher.

9. **Pre-bake targets** (`dwq.py:342-352`): if no cache exists and `--target-dir` was passed, calls `compute_dwq_targets(...)`.

10. **`--targets-only` exit** (`dwq.py:354-355`): bails out before student build. This is the "cluster-time-saver" path.

11. **Define `target_fn`** (`dwq.py:357-366`): two flavors.
    - Cached: `target_fn(_, idx, split)` reads `target_dir/{split}/{idx:010d}.safetensors` and returns `(logits, indices)` tuple of two arrays.
    - On-the-fly: `target_fn(batch, idx, split)` returns `model(batch)` — full vocab logits.

12. **Student model build** (`dwq.py:368-383`):
    - With `--quantized-model`: `load(...)` already-quantized student via `lazy=True`. Reject if config has no `quantization` block.
    - Without: `q_model = copy.deepcopy(model)` then `_, config = quantize_model(q_model, config, group_size, bits)` — fresh affine quantization off the teacher.

13. **Teacher destruction** (`dwq.py:386-387`): if targets were cached, `del model` so only the student stays resident.

14. **Wired-memory bump** (`dwq.py:389-391`): `mx.set_wired_limit(max_recommended_working_set_size)` on Metal devices — same trick `tuner/trainer.py:228-229` does.

15. **Optimizer** (`dwq.py:393`): `optimizers.Adam(learning_rate=args.learning_rate, bias_correction=True)`. **Note `bias_correction=True`** — this is non-default for mlx Adam in older versions and matters for tiny LRs (1e-6).

16. **Train** (`dwq.py:394-404`): `dwq_quantize(q_model, target_fn, opt, train, valid, batch_size, max_seq_length, seed, gradient_checkpoint=args.grad_checkpoint)`. Note `dtype=mx.bfloat16` and `temperature=2.0` are *not* exposed on the CLI — defaults inside `dwq_quantize` signature.

17. **Save** (`dwq.py:405-411`): `save(args.mlx_path, args.model, q_model, tokenizer, config)` — see `utils.py:925-950`. This both writes safetensors shards and copies `*.py` and `generation_config.json` from the source.

---

## 2. Core training loop — `dwq_quantize()`

**File**: `/opt/mlx-lm/mlx_lm/quant/dwq.py:69-209`

### Signature (`dwq.py:69-81`)
```python
dwq_quantize(model, target_fn, opt, train_data, valid_data,
             batch_size, max_seq_length, seed,
             dtype: mx.Dtype = mx.bfloat16,
             gradient_checkpoint: bool = False,
             temperature: float = 2.0)
```

### Inner functions

**`unfreeze(_, m)` (`dwq.py:90-97`)** — selects which modules to train.
```python
if hasattr(m, "bits") and hasattr(m, "group_size") \
   and m.mode == "affine" and m.bits < 8:
    m.unfreeze(keys=["scales", "biases"], recurse=False)
```
- Touches **only `nn.QuantizedLinear`-like modules**: things with `bits`, `group_size`, `mode`.
- Skips 8-bit groups (`m.bits < 8`) — hf2q must mirror this gate or the gate-projection layers (e.g. `mlp.gate` in MoE, which `qwen3_5.py:340-342` keeps at 8-bit) won't get DWQ-perturbed, but they also don't need it.
- `keys=["scales", "biases"]` and `recurse=False` together mean only the per-group scales/biases of *this* module become trainable — **never the int-quantized `weight`**. The integer codebook is frozen post-affine-quantize.
- Applied via `model.apply_to_modules(unfreeze)` (`dwq.py:100`).

**`loss_fn(params, x, targets, lengths)` (`dwq.py:108-118`)**
```python
model.update(tree_map(lambda x: x.astype(dtype), params))   # cast fp32 master to bf16 view
logits = model(x)
if isinstance(targets, tuple):                              # cached top-1024 path
    targets, ids = targets
    logits = mx.take_along_axis(logits, ids, axis=-1)       # gather candidate logits at teacher's top-1024 ids
losses = kl_div_loss(scale * logits, scale * targets)        # scale = 1/T = 0.5
mask = mx.arange(1, 1 + targets.shape[1]) < lengths[:, 1:]  # right-pad mask
ntoks = mask.sum()
loss = (mask * losses).sum() / ntoks
return loss, ntoks
```
- **Master params live in fp32** (`dwq.py:153-156`); each forward casts down to bf16. This is standard mixed-precision: gradients come back in bf16 but Adam state stays fp32.
- The per-token `kl_div_loss` is the fused Metal kernel from `tuner/losses.py:377-386` — see §10.
- `temperature=2.0` → `scale=0.5`. KL is computed on softer distributions.
- The mask `mx.arange(1, 1 + targets.shape[1]) < lengths[:, 1:]` excludes pad tokens (everything past `lengths[i, 1]`). Note targets correspond to positions `[0..L-1]` predicting `[1..L]`, so we mask `[1..L]`.
- **Critical invariant**: when targets are cached top-1024, both the teacher and student logits are reduced to those 1024 candidates **before** softmax+KL. The `kl_div_loss` Metal kernel then sees `V=1024` and runs over a tractable vocabulary regardless of the real model's V.

**`step(inputs, targets, lengths, params)` (`dwq.py:120-126`)**
```python
(loss, ntoks), grads = mx.value_and_grad(loss_fn)(params, inputs, targets, lengths)
grads = nn.average_gradients(grads)        # all-reduce across distributed group
params = opt.apply_gradients(grads, params)
return loss, ntoks, params
```
**Autograd shape: what's traced, what isn't.**
- `mx.value_and_grad(loss_fn)` differentiates **only with respect to `params`** (positional arg 0). Inputs, targets, lengths are leaves — no gradient flows backward into the data pipeline.
- Inside `loss_fn`, `model.update(tree_map(astype, params))` is what *connects* the trainable subset back to the model graph. Without that line, `loss_fn` would compute gradients for an isolated parameter tree disconnected from `model`'s actual forward.
- The **only** parameters in `params` are the unfrozen scales+biases (from `model.trainable_parameters()` after `unfreeze`), so the trace covers: `[scales] / [biases]` → dequantize → matmul → ... → logits → KL.
- The integer `weight` array is NOT in `params`; mlx treats it as a frozen op input to dequantize. Gradients flow through dequantize (which is a closed-form linear function of scales+biases) to the trainables.

**`validate(params, it)` (`dwq.py:128-150`)** — runs `loss_fn` over `valid_data`, all-sums via `mx.distributed.all_sum(..., stream=mx.cpu)`. Always runs once before training (it=0) and every 200 iters. The pre-vs-post comparison is the model-quality safety check (`dwq.py:202-207`).

### Training driver (`dwq.py:158-201`)
- `params = tree_map(lambda x: x.astype(mx.float32), model.trainable_parameters())` — fp32 master.
- Initial validate.
- For each `(batch, lengths)` from `iterate_batches`:
  - `batch = batch[:, :-1]` — drop last token (we predict next).
  - `targets = target_fn(batch, it, split="train")`; `mx.eval(targets)`.
  - `loss, ntoks, params = step(batch, targets, lengths, params)`; `mx.eval(loss, params)`.
  - Distributed all-sum on `loss`+`ntoks` (CPU stream — important to avoid GPU stalls on `.item()`).
  - Every 20 iters, print `it`, `avg_loss`, `total_tokens`, `toks_per_sec`, `peak_memory_gb` via `tqdm.write` (rank 0 only).
  - Every 200 iters, run `validate(params, it=it)`.
- Final validate; warn if final ≥ initial.
- `model.update(tree_map(astype(dtype), params))` — bake the bf16 view back into the model so `save()` writes the trained scales+biases.

---

## 3. Target precomputation — `compute_dwq_targets()`

**File**: `/opt/mlx-lm/mlx_lm/quant/dwq.py:29-66`

```python
def compute_dwq_targets(model, save_dir, train_data, valid_data,
                        batch_size, max_seq_length, seed):
    rank = mx.distributed.init().rank()
    def _compute_targets(data, path, split):
        if rank == 0:
            (path / split).mkdir(parents=True, exist_ok=True)
        for i, (batch, _) in enumerate(iterate_batches(data, batch_size, max_seq_length, seed=seed)):
            batch = batch[:, :-1]
            logits = model(batch)
            logits = mx.stop_gradient(logits, stream=mx.cpu)  # CPU-side eval to dodge timeout
            mx.eval(logits)
            if rank == 0:
                idx = mx.argpartition(logits, kth=-1024, axis=-1)[..., -1024:]
                logits = mx.take_along_axis(logits, idx, axis=-1)
                file = path / f"{i:010d}.safetensors"
                mx.save_safetensors(file, {"logits": logits, "indices": idx})
    _compute_targets(valid_data, save_dir, "valid")
    _compute_targets(train_data, save_dir, "train")
```

**Top-1024 logit storage format**:
- One `.safetensors` per batch.
- Two arrays: `logits` (shape `[B, L-1, 1024]`, dtype matches teacher fwd output — usually bf16) and `indices` (shape `[B, L-1, 1024]`, int32-ish from `argpartition`).
- Note: **logits are RAW logits** (not log-softmax, not log-probs). Soft targets for KL are reconstructed at training time via `kl_div_loss(scale * student_logits, scale * teacher_logits)`.
- `argpartition(kth=-1024)` is unsorted — top-1024 in arbitrary order. The student gather must use the same `indices` so positions align; absolute order doesn't matter because KL is symmetric in vocab indices once you've committed to a 1024-d sub-distribution.

**Disk layout**:
```
target_dir/
├── train/0000000000.safetensors
├── train/0000000001.safetensors
├── ...
└── valid/0000000000.safetensors
```
Filename format `f"{i:010d}.safetensors"` — 10 digits, zero-padded. The training loop reloads via `mx.load(target_dir / split / f"{idx:010d}.safetensors")` (`dwq.py:359-361`).

**Reload path** (`dwq.py:359-361`): the cached `target_fn` returns `(targets["logits"], targets["indices"])` as a tuple, which `loss_fn` detects via `isinstance(targets, tuple)` (`dwq.py:111`).

**Distributed semantics**: only rank 0 writes; all other ranks compute and discard via `mx.eval`. The disk cache is shared (one rank's worth of data per file).

**Subtle**: `mx.stop_gradient(logits, stream=mx.cpu)` (`dwq.py:56`) is the "make the last op pre-eval on CPU to avoid eval timeout" hack noted in the comment. It moves the final no-op to the CPU stream so the GPU stream doesn't stall waiting for the result.

---

## 4. Model load + quantize — `load`, `quantize_model`

### `load_model(model_path, lazy=False, strict=True, ...)` — `utils.py:282-420`

Returns `(model, config)`. Logic:
1. `load_config(model_path)` → reads `config.json`, splices `eos_token_id` from `generation_config.json` if present.
2. Glob `model*.safetensors`, `mx.load` each, merge dicts.
3. Model class lookup: `_get_classes(config)` (`utils.py:175-193`) — uses `MODEL_REMAPPING` then `importlib.import_module(f"mlx_lm.models.{model_type}")` and pulls `arch.Model`, `arch.ModelArgs`. For Qwen 3.5 MoE, this lands at `/opt/mlx-lm/mlx_lm/models/qwen3_5_moe.py`. For non-MoE Qwen 3.5 the module is `qwen3_5.py`.
4. `model = model_class(model_args)`.
5. `weights = model.sanitize(weights)` — model-specific renames (`qwen3_5_moe.py:23-52`, `qwen3_5.py:307-331` and `:384-398`). MoE specifically: splits packed `experts.gate_up_proj` into `switch_mlp.{gate,up}_proj.weight`.
6. **Quantization replay** (`utils.py:348-390`): if `config["quantization"]` exists, calls `nn.quantize(model, group_size, bits, mode, class_predicate=...)`. The `class_predicate` is keyed off `f"{p}.scales" in weights` — so the model only quantizes layers that have prebaked scales in the safetensors.
7. Handles legacy `quantization_config` from HF (mxfp4/awq/gptq/compressed-tensors). For DWQ flow, irrelevant.
8. `model.eval()`, `model.load_weights(list(weights.items()), strict=strict)`.
9. If `not lazy`: `mx.eval(model.parameters())` materializes everything.

### `load_tokenizer(model_path, ...)` — `utils.py:429-450`

Light: downloads tokenizer files only (no safetensors), wraps in `TokenizerWrapper` from `tokenizer_utils`. **Returns the wrapper object**, not the raw HF tokenizer. The `.encode(...)`, `.apply_chat_template(...)`, `.eos_token_id` API works identically.

### `load(path_or_hf_repo, ...)` — `utils.py:453-502`

Public entry. Calls `_download(...)` → `load_model(...)` → optional `load_adapters(...)` → `load_tokenizer(...)`. Returns `(model, tokenizer)` or `(model, tokenizer, config)` if `return_config=True`.

### `save(dst_path, src_path_or_repo, model, tokenizer, config, donate_model=True)` — `utils.py:925-950`

1. `save_model(dst_path, model, donate_model=True)` (`utils.py:714-771`) — flattens params, shards into ≤5 GB safetensors files, writes `model.safetensors.index.json`. **Crucially**: `mx.save_safetensors(..., metadata={"format": "mlx"})` — see line 756. **kld.py rejects baselines without this metadata** (`kld.py:328-333`).
2. `save_config(config, dst/config.json)` — pops `_name_or_path`/`vision_config`, sets both `quantization` and `quantization_config` (HF-tree compat — `utils.py:914-915`).
3. `tokenizer.save_pretrained(dst)`.
4. Copies `*.py` and `generation_config.json` from src.
5. `create_model_card(dst, hf_path)` — writes minimal HF model card.

### `quantize_model(model, config, group_size, bits, mode="affine", quant_predicate=None)` — `utils.py:774-850`

What it expects:
- A *fresh* (unquantized) model (or one with partial `config["quantization"]` for fine-grained per-layer overrides — flagged via `fine_grained_config`).
- `config` dict with `model_type` etc.

Where the affine quant scales/biases come from at init time: **inside `nn.quantize(model, group_size, bits, mode="affine", class_predicate=wrapped_predicate)`**, which walks every linear-like module and calls `module.to_quantized(group_size, bits, mode)`. For `nn.Linear`, this dispatches to `QuantizedLinear.from_linear(...)` (`mlx/nn/layers/quantized.py:280-298`):
```python
ql.weight, ql.scales, *biases = mx.quantize(linear_layer.weight, group_size, bits, mode=mode)
ql.biases = biases[0] if biases else None
```
So `mx.quantize(W, G, B, mode='affine')` returns three arrays:
- `weight`: int-packed (uint32, packed `32//bits` integers per word).
- `scales`: per-group fp scale, shape `[..., n_groups]`.
- `biases`: per-group fp offset, shape `[..., n_groups]`.

The `wrapped_predicate` (`utils.py:823-835`) skips layers whose `weight.shape[-1] % group_size != 0` and respects an optional model-level `quant_predicate` property. For Qwen 3.5 MoE, that property forces `mlp.gate` and `shared_expert_gate` to `{group_size: 64, bits: 8}` (`qwen3_5.py:338-343`).

Returns `(model, quantized_config)`. After this point, `model.parameters()` includes `*.weight (uint32) / *.scales (fp16/bf16) / *.biases (fp16/bf16)` for every quantized layer — **and** unquantized norms, embeddings, gate.weight (8-bit special).

The bit-precision report at `utils.py:847` (`compute_bits_per_weight`) is what shows up as `[INFO] Quantized model with X.XXX bits per weight.`

---

## 5. Qwen3.5MoE model architecture

### Forward graph

**File**: `/opt/mlx-lm/mlx_lm/models/qwen3_5.py` (the actual logic) and `qwen3_5_moe.py` (a 52-LOC subclass that adds vision-prefix sanitization + expert weight unpacking — see `qwen3_5_moe.py:36-50`).

Top: `Model.__call__ → language_model.__call__ → TextModel.__call__ → Qwen3_5TextModel.__call__ → DecoderLayer.__call__ × num_hidden_layers → norm → lm_head` (or `embed_tokens.as_linear` for tied embeddings).

```
inputs (B, L) [int32]
└── embed_tokens(inputs)               nn.Embedding(vocab, hidden)
    │
    └── for layer in 47 layers:        # qwen3.5-A3B-A35B has 47 layers
    │   ├── if (layer_idx + 1) % full_attention_interval != 0:    # default 4 → 3 of every 4 are linear
    │   │       linear_attn = GatedDeltaNet(args)                  # qwen3_5.py:86-206
    │   │   else:
    │   │       self_attn = Qwen3NextAttention(args)               # imported from qwen3_next.py:81-158
    │   ├── input_layernorm: nn.RMSNorm
    │   ├── post_attention_layernorm: nn.RMSNorm
    │   └── if num_experts > 0:
    │           mlp = Qwen3NextSparseMoeBlock                      # imported from qwen3_next.py:308-354
    │       else:
    │           mlp = Qwen3NextMLP(hidden, intermediate)           # gate/up/down projs + swiglu
    └── norm: nn.RMSNorm
        └── lm_head: nn.Linear(hidden, vocab) OR embed_tokens.as_linear if tie_word_embeddings
```

Pattern: `out = h + self.mlp(self.post_attention_layernorm(h))` where `h = x + r` and `r` is from attention. Pre-norm residual.

### Layer types: linear vs full attention (3:1)

- `(layer_idx + 1) % full_attention_interval != 0` → linear (gated DeltaNet).
- `(layer_idx + 1) % full_attention_interval == 0` → full softmax attention (`Qwen3NextAttention`).

Default `full_attention_interval = 4` (see `qwen3_5.py:43`), giving exactly the 3:1 ratio claimed by Qwen 3.5 docs. For 47 layers, that's ~12 full-attention layers + ~35 linear layers.

`DecoderLayer.__init__` (`qwen3_5.py:209-226`) selects via `self.is_linear` boolean; cache layout follows in `make_cache()` (`qwen3_5.py:304-305`): `ArraysCache(size=2)` for linear (conv state + recurrent state), `KVCache()` for full.

Mask routing (`qwen3_5.py:268-273`): `fa_mask = create_attention_mask(...)` and `ssm_mask = create_ssm_mask(...)` are pre-computed once per forward, then `mask = ssm_mask if layer.is_linear else fa_mask`.

### MoE routing — `Qwen3NextSparseMoeBlock` (`qwen3_next.py:308-354`)

```python
gates = self.gate(x)                                    # (B, L, num_experts)  — 8-bit nn.Linear
gates = mx.softmax(gates, axis=-1, precise=True)         # softmax over experts
k = self.top_k                                           # = num_experts_per_tok
inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:] # top-k experts unsorted
scores = mx.take_along_axis(gates, inds, axis=-1)        # gather their probs
if self.norm_topk_prob:
    scores = scores / scores.sum(axis=-1, keepdims=True) # renormalize to sum=1 over top-k

y = self.switch_mlp(x, inds)                             # SwitchGLU dispatch
y = (y * scores[..., None]).sum(axis=-2)                 # weighted combine

shared_y = self.shared_expert(x)
shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
y = y + shared_y
```

Specifics for Qwen 3.5 35B-A3B-APEX:
- `num_experts = 128`
- `num_experts_per_tok = 8`
- `norm_topk_prob = True`
- One shared expert with `shared_expert_intermediate_size`.
- The `mlp.gate` Linear is **8-bit** by mandate (`qwen3_5.py:340-342`); all other matmuls are 4-bit when DWQ-quantized.

`SwitchGLU.__call__` (`switch_layers.py:176-199`):
1. Expand x dims `(-2, -3)`.
2. If indices >= 64 tokens dispatched: `_gather_sort` reorders by expert id for cache locality.
3. `up_proj(x, idx)` and `gate_proj(x, idx)` — both are `QuantizedSwitchLinear` after `nn.quantize`.
4. `swiglu(x_up, x_gate)` → `down_proj(...)`.
5. `_scatter_unsort` if pre-sorted.

`QuantizedSwitchLinear` (`switch_layers.py:27-90`) has the same `weight/scales/biases/bits/group_size/mode` shape as `QuantizedLinear`, but with shape `(num_experts, output_dims, input_dims)`. Calls `mx.gather_qmm` for the routed matmul.

### Differentiable vs not

- **All matmuls** (q/k/v/o, gate/up/down in dense, gate/expert/shared in MoE) flow autograd through `mx.quantized_matmul` and `mx.gather_qmm`. The integer weights are constants; gradients flow only into scales/biases.
- **`mx.argpartition` and `mx.take_along_axis`** in routing are differentiated as `stop_gradient` on the index path (mlx convention; see `switch_layers.py:187`: `idx = mx.stop_gradient(idx)` if `self.training`).
- **`gated_delta_update`** (`qwen3_5.py:183`) is the recurrent SSM-style update; it has its own kernel (`use_kernel=not self.training`) — during training the Python fallback is used, which is differentiable.
- **`scaled_dot_product_attention`** (full-attn path, `qwen3_next.py:153`) is mx's fused softmax-attention with backward.

### `mx.compile` / `mx.vmap` usage

- `dwq.py` itself does **not** use `mx.compile` (the regular `tuner/trainer.py:248` does, but DWQ skips it because `mx.value_and_grad` over a `params` arg is already traceable, and recompilation per-iter would defeat the cached top-K target gather pattern).
- No `mx.vmap`. Batch dimension is explicit throughout.
- `grad_checkpoint(model.layers[0])` (see §below) is the only graph-modification trick used.

---

## 6. Calibration data path

DWQ uses **`load_data` from `quant/dwq.py:212-239`**, which delegates to `tuner/datasets.py:load_dataset`. Important: this is **NOT** the simpler `quant/utils.py:load_data` (calibration_v5.txt) — that one is for AWQ/GPTQ only.

### `quant/dwq.py:load_data` (`dwq.py:212-239`)

```python
args = types.SimpleNamespace(
    hf_dataset={"path": data_path,
                "train_split": "train",
                "valid_split": "train[:1]"},
    train=True, test=False,
)
dataset = load_dataset(args, tokenizer)[0]               # train slice from tuner/datasets.py
perm = np.random.permutation(len(dataset))
train_perm = perm[:num_samples].tolist()
valid_perm = perm[num_samples : num_samples + num_valid_samples].tolist()  # default 32
def process(idx):
    tokens, offset = dataset.process(dataset[idx])
    return (tokens[:max_seq_length], offset)
return ([process(i) for i in train_perm],
        [process(i) for i in valid_perm])
```

Returns `[(tokens, offset), ...]` lists. `offset` is always 0 here (no `mask_prompt`). `tokens` are int lists (Python list, not mx array yet — `iterate_batches` converts via `np.zeros` then `mx.array`).

### `tuner/datasets.py:load_dataset(args, tokenizer)` (`datasets.py:309-332`)

For `tulu-3-sft-mixture`:
1. `load_custom_hf_dataset` (`:249-306`) — uses HF `datasets.load_dataset(path, split=...)`.
2. `train_split = "train"` (full), `valid_split = "train[:1]"` (1-sample dummy that DWQ ignores).
3. `create_dataset(ds, tokenizer, config)` (`:175-202`) inspects first sample.
4. `tulu-3-sft-mixture` has `messages` field → `ChatDataset` (`:39-83`).

### `ChatDataset.process` (`datasets.py:57-77`)
```python
messages = d["messages"]
tools = d.get("tools", None)
tokens = self.tokenizer.apply_chat_template(messages, tools=tools, return_dict=False)
return (tokens, 0)  # mask_prompt=False
```

So calibration data is **chat-templated multi-turn conversations** through the model's tokenizer. NOT raw text. NOT calibration_v5.txt. The `tokenizer.apply_chat_template` produces the same token stream the model would see at inference.

### Tokenization details

- `tokens` is a Python list of int token IDs.
- `iterate_batches` (`tuner/trainer.py:102-173`) sorts by length, makes batches of `batch_size`, pads to `1 + pad_to * ((max_len + pad_to - 1) // pad_to)` with pad_to=32, capped at `max_seq_length`. Pad value is `0` (zero-init `np.zeros` then overwrite).
- `lengths` is `mx.array([(offset, length), ...])` — used by `loss_fn`'s mask.

### `quant/utils.py:load_data` (`/opt/mlx-lm/mlx_lm/quant/utils.py:8-26`) — DIFFERENT

This is the **AWQ/GPTQ** calibration loader, downloading `~/.cache/mlx-lm/calibration_v5.txt` from a tristandruyen gist. It tokenizes raw text and reshapes into non-overlapping `sequence_length`-chunks.

DWQ does not use this; it imports `load_data` only as the local function defined at `dwq.py:212`. The kld.py shim (`/opt/hf2q-adr020-iter10/scripts/dwq_kl_parity/kld.py:27-38`) **does** import this `quant/utils.py:load_data` and treats `calibration_v5` as the canonical eval corpus — which **disagrees with DWQ's actual training corpus** (tulu-3-sft-mixture). Worth flagging.

---

## 7. KLD measurement — kld.py

**File**: `/opt/hf2q-adr020-iter10/scripts/dwq_kl_parity/kld.py` (vendored from PR #1146 against mlx-lm; see line 22-26 comment on missing upstream `load_eval_tokens` shim).

### Format the baseline must be in

`kld.py:321-333`:
```python
weight_file = next(model_path.glob("model*.safetensors"), None)
metadata = (... safe_open(...).metadata() or {})
if metadata.get("format") != "mlx":
    raise ValueError("kld requires MLX-converted weights saved by mlx-lm "
                     "(expected safetensors metadata format='mlx').")
```
The `format='mlx'` metadata is set by `utils.py:756` (`mx.save_safetensors(..., metadata={"format": "mlx"})`). HF safetensors don't carry this; only mlx-lm-saved models pass this check. **This is the reason hf2q must produce mlx-format safetensors before kld can score them.**

### How it builds the cache (`kld.py:258-309`)

1. `load_model_or_raise(args.baseline_model, "baseline", lazy=True)` — loads the *unquantized* teacher.
2. `load_eval_tokens(tokenizer, data_path, num_samples, sequence_length, seed=seed)` — **shim'd** to call `quant/utils.py:load_data` with calibration_v5 corpus (data_path arg ignored). NOTE: this is **not** the same corpus DWQ trains on.
3. `cache.save_tokens(tokens)` writes `tokens.safetensors`.
4. Per batch: `baseline_logprobs = nn.log_softmax(model(batch[:, :-1]).astype(mx.float32))`.
5. `cache_topk_batch(...)` — see below.
6. `cache.save_batch(batch_idx, indices, top_logprobs, tail_mass)` writes `baseline_{idx:06d}.safetensors`.
7. `write_manifest(cache.path, manifest)` — JSON with `baseline_model`, `top_k`, `data_path`, `sequence_length`, `num_samples`, `seed`, `batch_size`, `vocab_size`, `vocab_hash`. Cache dir is derived as `kld_cache/{slug}-{8-char-sha256}` from these fields.

### Top-K compression (`kld.py:461-469`)

```python
def cache_topk_batch(logprobs, top_k):
    kth = logprobs.shape[-1] - top_k
    indices = mx.argpartition(logprobs, kth=kth, axis=-1)[..., -top_k:].astype(mx.int32)
    top_logprobs = mx.take_along_axis(logprobs, indices, axis=-1)
    order = mx.argsort(-top_logprobs, axis=-1)              # sort top-K descending by logprob
    indices = mx.take_along_axis(indices, order, axis=-1)
    top_logprobs = mx.take_along_axis(top_logprobs, order, axis=-1)
    tail_mass = mx.clip(1.0 - mx.sum(mx.exp(top_logprobs), axis=-1), 0.0, 1.0)
    return indices, top_logprobs, tail_mass.astype(mx.float32)
```

Differences from DWQ's target cache:
- Stores **logprobs** (not raw logits).
- Sorts top-K descending by probability (DWQ leaves them unsorted).
- Adds `tail_mass = 1 - sum(top_K_probs)` for the residual KL contribution.

### Per-token KL from cached top-K + candidate full-vocab logits (`kld.py:472-495`)

```python
def kl_from_cached_batch(model_logprobs, cached_batch):
    base_top_logprobs = cached_batch["logprobs"]
    model_top_logprobs = mx.take_along_axis(model_logprobs,
                                             cached_batch["indices"], axis=-1)
    base_top_probs = mx.exp(base_top_logprobs)
    kl_top = mx.sum(base_top_probs * (base_top_logprobs - model_top_logprobs), axis=-1)

    base_tail_mass = mx.clip(cached_batch["tail_mass"], 0.0, 1.0)
    model_top_mass = mx.sum(mx.exp(model_top_logprobs), axis=-1)
    model_tail_mass = mx.clip(1.0 - model_top_mass, 1e-30, 1.0)
    base_tail_log = mx.log(mx.clip(base_tail_mass, 1e-30, 1.0))
    model_tail_log = mx.log(model_tail_mass)
    kl_tail = mx.where(base_tail_mass > 0,
                        base_tail_mass * (base_tail_log - model_tail_log),
                        0.0)
    return kl_top + kl_tail
```

Decomposition: `KL(P_base || P_model) ≈ Σ_top p_i (logp_i - logq_i) + tail_mass * (log(tail_p) - log(tail_q))`. The tail term aggregates the off-top-K vocabulary as a single bucket — a closed-form **upper bound** on the true KL.

This is mathematically distinct from DWQ's loss:
- DWQ loss: `KL(scale*teacher || scale*student)` over the **gathered top-1024 sub-distribution only** (no tail correction; the gather happens **before** softmax — see `loss_fn` at `dwq.py:113-114` calling `mx.take_along_axis(logits, ids, axis=-1)`).
- KLD eval: `KL(softmax(teacher) || softmax(student))` over **full vocab**, approximated via top-K + tail.

So DWQ is a *consistent biased* estimator of the true KL (it commits to teacher's top-K and ignores tail mass entirely), while kld.py is a *near-unbiased* estimator. This is why the two numbers won't match exactly even on the same data.

### Output stats (`kld.py:506-519`, `:359-374`)

Mean, median, p95, max KL per token, plus stderr (sample std of per-sequence means, divided by sqrt(N)) and tokens/sec.

---

## 8. Affine quantization primitives

### `mx.quantize / mx.dequantize / mx.quantized_matmul` (C++ side)

These are mx core primitives, not Python. The Python `nn.QuantizedLinear` is a thin wrapper.

`mx.quantize(weight, group_size, bits, mode='affine')` returns:
- `q_int`: uint32 array, packed `32//bits` integers per word; shape `(out, in // (32//bits))`.
- `scales`: fp array, shape `(out, in // group_size)`.
- `biases`: fp array, shape `(out, in // group_size)` — only emitted when `mode='affine'`. Other modes (`mxfp4`, `nvfp4`, `mxfp8`) return only `q_int, scales`.

Dequant rule: `qdq[g*group_size + i] = q_int[g*group_size + i] * scales[g] + biases[g]`. **Group size 64 means one (scale, bias) pair per 64 input columns of the weight matrix.**

### `nn.QuantizedLinear` (`/Users/robert/.pyenv/versions/3.13.12/lib/python3.13/site-packages/mlx/nn/layers/quantized.py:200-302`)

```python
class QuantizedLinear(Module):
    def __init__(self, input_dims, output_dims, bias=True,
                 group_size=None, bits=None, mode='affine'):
        self.group_size, self.bits = _defaults_for_mode(mode, group_size, bits)
        self.mode = mode
        weight = mx.random.uniform(...)
        self.weight, self.scales, *biases = mx.quantize(weight, group_size, bits, mode=mode)
        self.biases = biases[0] if biases else None
        if bias: self.bias = mx.zeros((output_dims,))
        self.freeze()
    def __call__(self, x):
        x = mx.quantized_matmul(x, self["weight"],
                                 scales=self["scales"], biases=self.get("biases"),
                                 transpose=True, group_size=self.group_size,
                                 bits=self.bits, mode=self.mode)
        if "bias" in self: x = x + self["bias"]
        return x
    @classmethod
    def from_linear(cls, linear_layer, group_size, bits, mode='affine'):
        ...  # same path, but quantize linear_layer.weight instead of random
```

**Key point**: at construction time `self.freeze()` is called — all params (`weight`, `scales`, `biases`, `bias`) are added to `self._no_grad`. DWQ then selectively re-enables `["scales", "biases"]` via `m.unfreeze(keys=["scales", "biases"], recurse=False)`.

### How `unfreeze` exposes the trainable subset to autograd

`unfreeze` (`mlx/nn/layers/base.py:519-...`) removes keys from `self._no_grad`. The `model.trainable_parameters()` method (Module API) walks the tree and emits parameters whose key is **NOT** in any ancestor's `_no_grad`. So after the DWQ unfreeze pass, `model.trainable_parameters()` is a nested dict containing only the `scales` and `biases` of every sub-8-bit affine-quantized module.

DWQ then takes `params = tree_map(astype(fp32), model.trainable_parameters())` — a **separate tree** from the model. The `loss_fn` reconnects via `model.update(params_cast_to_dtype)` at the start of each forward. `mx.value_and_grad(loss_fn)` differentiates only with respect to the explicit `params` arg, so gradients are computed exactly for these scales+biases — nothing else.

### Per-group scales/biases storage layout

For a Linear with `(out=4096, in=4096)`, `group_size=64`, `bits=4`:
- `weight`: uint32, shape `(4096, 4096 // (32//4))` = `(4096, 512)`.
- `scales`: fp16/bf16, shape `(4096, 4096 // 64)` = `(4096, 64)`.
- `biases`: fp16/bf16, shape `(4096, 64)`.

Total trainable params per Linear: `2 × 4096 × 64 = 524k` (vs 16.78M dense). DWQ has roughly `64*2/4096 = 3.1%` of the dense parameter count to optimize over — far cheaper than full FT, far higher rank than LoRA.

---

## 9. What hf2q would need to port (preliminary)

Worker 3 will mirror in detail. Obvious gaps we already see:

**hf2q has** (from project memory):
- HF→GGUF converter pipeline.
- K-quant (mixed precision) primitives — but for **GGUF**, not mlx safetensors.
- Inference engine (Rust) on top of mlx-native — distinct from mlx-lm's MLX-Python forward.

**hf2q needs to build**:
- An mlx-Python (or equivalent) **affine quantizer** (`mx.quantize` ↔ hf2q K-quant Q4_0). Either reuse `mx.quantize` directly via Python orchestration, or implement Q4_0-with-per-group-scales+biases matching mlx's `weight/scales/biases/group_size/bits/mode` schema in pure Rust.
- A **trainable surrogate** for the int-packed weight: mlx-lm's design relies on the integer codebook being a frozen op input that `mx.quantized_matmul` knows how to differentiate through (gradients flow analytically into scales/biases). hf2q would need either (a) the same Python+mlx call chain, or (b) a custom autograd-capable Q4_0 dequantize-then-matmul kernel.
- **Loss kernel parity**: the fused `kl_div_loss` Metal kernel in `tuner/losses.py:11-176`. Either port to Rust+Metal (substantial), call out to mlx-Python via PyO3, or fall back to the explicit `nn.losses.kl_div_loss` after `logsumexp`.
- **Calibration data loader**: `tuner/datasets.py:load_dataset` with `tulu-3-sft-mixture` HF dataset → `apply_chat_template` → `(tokens, offset)` tuples → `iterate_batches` length-bucketed padded batches. Currently nothing in hf2q does HF datasets or chat-template tokenization for training.
- **Optimizer**: mlx Adam with `bias_correction=True`. Either bind to mlx Python or reimplement a fp32 master + bf16 view Adam.
- **Top-1024 target cache**: `compute_dwq_targets` is straightforward (argpartition + take_along + safetensors write). hf2q would need a safetensors writer (or .npz, since it's an internal disk format).
- **Save path**: hf2q's currency is GGUF, but kld.py and mlx-lm's load expect `model*.safetensors` + `model.safetensors.index.json` + `config.json` + `metadata={"format": "mlx"}`. hf2q either (a) writes mlx-format safetensors as an output mode, (b) keeps shelling out to `mlx_lm.dwq` until parity reached, or (c) builds a parallel kld implementation that reads GGUF directly.
- **Qwen 3.5 MoE forward**: hf2q's qwen35moe inference path (memory: `forward_gpu.rs`, `cmd_generate_qwen35`) is **inference-only**, no autograd. To run DWQ in-process, hf2q needs a differentiable forward — most realistically by depending on mlx-lm Python for training and just reading back the trained scales/biases.

**Minimum-viable port** likely looks like:
1. Drive mlx-lm DWQ from a hf2q subprocess wrapper — but with hf2q owning the (a) student model materialization (build mlx-format safetensors from hf2q's K-quant), (b) calibration data path, (c) target cache management, (d) postprocess back to GGUF.
2. Stretch goal: replace the subprocess with an in-process Python embedded interpreter via PyO3, sharing the model object across hf2q and mlx-lm.
3. Full port (all-Rust): rebuild differentiable Q4_0 kernels + Adam + KL kernel — this is several months of work and duplicates mlx-lm.

---

## 10. Subtle details that matter

1. **`tqdm.write` doesn't flush stdout** — known. `dwq.py:88` uses `tqdm.write` for periodic `rprint`. Our wrappers must capture both stdout and stderr; tqdm writes to stdout by default but `_log` in kld.py (`kld.py:530`) uses stderr.

2. **`tokenizer = load_tokenizer(args.model)` BEFORE deciding whether to load the teacher** (`dwq.py:326-330`) — the tokenizer is *always* loaded from the base model, never the quantized one. This matters when the quantized model directory has its own tokenizer files (post-`save()` they will), and you naively re-run DWQ on it as the teacher.

3. **`lazy=True` for both teacher and quantized-student loads** (`dwq.py:337`, `dwq.py:371`) — without this, peak RSS spikes on a 35B teacher.

4. **`mx.stop_gradient(logits, stream=mx.cpu)` in `compute_dwq_targets`** (`dwq.py:56`) — workaround for an mx eval-timeout on multi-GB intermediate logits. Don't drop it when porting.

5. **Adam `bias_correction=True`** (`dwq.py:393`) — non-default flag, matters at 1e-6 LR.

6. **The training loop uses `iterate_batches` from `tuner/trainer.py`**, NOT a separate DWQ batcher. Length-sorted bucketing across the train set means **early iters see short sequences, late iters see long ones**. This biases the loss curve; users see early loss go down then bounce. Not a bug.

7. **`model.update(tree_map(astype(dtype), params))` happens INSIDE `loss_fn`** (`dwq.py:109`) — every forward, the bf16 view is rewritten from the fp32 master. Skipping this once means stale weights on the next forward.

8. **The mask `mask = mx.arange(1, 1 + targets.shape[1]) < lengths[:, 1:]`** (`dwq.py:115`) uses `lengths[:, 1]` (the second column, "length") — the first column is `offset` (always 0 for chat-template-no-mask data). When porting, don't drop the `[:, 1:]`.

9. **`model.train()` is called on a quantized model** (`dwq.py:99`). This enables training-mode features in some submodules (notably `SwitchGLU.__call__` at `switch_layers.py:186-187` does `idx = mx.stop_gradient(idx)` only when `self.training=True`). Without `model.train()`, expert-routing indices would receive nonzero gradients via the soft routing — breaking everything.

10. **`grad_checkpoint(model.layers[0])`** (`dwq.py:104`, defined `tuner/trainer.py:25-38`) does a clever monkey-patch: it sets `type(layer).__call__ = checkpointed_fn` so **all** instances of `DecoderLayer` use mx checkpointed forward. It rebuilds the call as `mx.checkpoint(inner_fn)(model.trainable_parameters(), *args)`. Subtle: the `inner_fn` inside captures `model` by closure and re-`update`s on every recompute, which means each layer's params are re-bound during the backward recompute. Only triggered with `--grad-checkpoint`.

11. **`Qwen3_5MoeModel` weight unpacking** (`qwen3_5_moe.py:36-50`) — HF stores a single `experts.gate_up_proj` with mid-axis splittable into gate and up. mlx wants them split into two SwitchLinears. Sanitization happens in `Model.sanitize` and runs **before** quantization. If hf2q's GGUF→mlx writer doesn't replicate this split, MoE layers will be broken.

12. **Two distinct calibration corpuses across DWQ and KLD**:
    - DWQ training: `allenai/tulu-3-sft-mixture` chat-templated.
    - KLD eval: `~/.cache/mlx-lm/calibration_v5.txt` raw text (gist tristandruyen/9e207a95).
    - These were chosen independently. If you swap eval to tulu, KL numbers change. If you swap DWQ to v5, model quality changes. Document the choice; don't assume they should match.

13. **`save()` calls `save_model(... donate_model=True)`** (`utils.py:942`). After save, the model object's parameters are replaced with empty arrays (`utils.py:743`). Calling forward on the model post-save will fail. Order operations carefully if porting.

14. **Cache dir derivation** in kld.py is content-addressed: `kld_cache/{slug}-{8-char-sha256}` (`kld.py:218-224`) over the manifest-key fields. Same baseline + same args = cache reuse. Different args = different cache. This means if hf2q changes any of `(baseline_model, top_k, data_path, sequence_length, num_samples, seed, batch_size)`, the cache rebuild is automatic.

15. **`mx.distributed.init()` is mandatory even on single-host** — both `dwq.py:38` and `:82` call it. Returns a group with size=1 rank=0. Without the init, `nn.average_gradients` and `mx.distributed.all_sum` will error.

---

## Summary

The mlx-lm DWQ implementation is structurally simple — ~407 lines plus four small dependencies — but it leans hard on mlx primitives that are non-trivial to replace: `mx.quantize/quantized_matmul/gather_qmm`, `mx.value_and_grad` over `model.trainable_parameters()`, the fused KL-divergence Metal kernel, mlx Adam with fp32 master + bf16 view, and `nn.QuantizedLinear`'s frozen-int / unfrozen-(scales,biases) split. The **load/save** flow demands mlx-format safetensors (`metadata={"format": "mlx"}`) which kld.py rigorously enforces. The **calibration path** is `tulu-3-sft-mixture` chat-templated through the model's tokenizer (NOT the calibration_v5.txt corpus — that's KLD's eval corpus, a separate choice). The **Qwen 3.5 MoE forward graph** is 47 layers (3:1 linear:full attention via `(layer+1) % full_attention_interval`), with top-8-of-128 expert routing through `Qwen3NextSparseMoeBlock`, plus a shared expert + sigmoid gate, and `mlp.gate` mandated to 8-bit by `quant_predicate`. For an in-process hf2q port, the cheapest path is subprocess-driving `mlx_lm.dwq` while owning model materialization (mlx-format safetensors out of K-quant), calibration management, and GGUF roundtrip; a full Rust+Metal port is 3-6 months of duplicated infrastructure.
