# mlx-lm DWQ Deep Research (port-ready)

Source-of-truth files (all paths absolute):

- `/opt/mlx-lm/mlx_lm/quant/dwq.py` (411 LOC) — the algorithm + CLI.
- `/opt/mlx-lm/mlx_lm/LEARNED_QUANTS.md` — narrative.
- `/opt/mlx-lm/mlx_lm/quant/utils.py` (27 LOC) — calibration_v5 fallback dataset loader (NOT used by DWQ; AWQ/GPTQ use it). DWQ uses HF tulu-3-sft-mixture via `mlx_lm.tuner.datasets.load_dataset` (`/opt/mlx-lm/mlx_lm/quant/dwq.py:16,228`).
- `/opt/mlx-lm/mlx_lm/tuner/losses.py:344-386` — KL kernel.
- `/opt/mlx-lm/mlx_lm/tuner/trainer.py:25-38, 102-173` — `grad_checkpoint`, `iterate_batches`.
- `/opt/mlx-lm/mlx_lm/tuner/utils.py:160-168` — `print_trainable_parameters`.
- `/opt/mlx-lm/mlx_lm/utils.py:594-595, 714-771, 774-850, 925-950` — `pipeline_load`, `save_model`, `quantize_model`, `save`.
- `/Users/robert/.pyenv/versions/3.13.12/lib/python3.13/site-packages/mlx/nn/layers/quantized.py:200-302` — `QuantizedLinear` (the layer DWQ tunes).

---

## 1. High-level algorithm flow

Pseudocode (one-line per stage, mapped to `dwq.py` line numbers):

```
1. parse args, init mx.distributed group              # 243-310
2. load tokenizer + tulu-3-sft train/valid splits     # 326-330  (load_data: 212-239)
3. (lazy) load FP teacher (or pipeline-shard it)      # 333-337
4. if --target-dir: precompute & dump top-1024 logits # 342-352  (compute_dwq_targets: 29-66)
5. (optional) exit if --targets-only                  # 354-355
6. load already-quantized student OR copy+quantize    # 368-383
7. delete teacher if targets are on disk              # 386-387
8. set Metal wired-limit to max recommended           # 389-391
9. build Adam(lr=1e-6, bias_correction=True)          # 393
10. dwq_quantize(student, target_fn, opt, ...)        # 394-404  (core loop: 69-209)
11. save HF-shape MLX dir                             # 405-411
```

The core loop inside `dwq_quantize` (69-209) is:

```
unfreeze {scales, biases} on every affine QuantizedLinear with bits<8     # 90-100
upcast trainable params to fp32                                           # 153-156
initial_valid_loss = validate(...)                                        # 165
for each (batch, lengths) in iterate_batches(train, ...):                 # 167-197
    targets = target_fn(batch, it, "train")  # disk or live teacher       # 176
    loss, ntoks, params = step(...)          # autograd + Adam             # 178
    all_sum loss/ntoks across ranks                                        # 180-181
    every 200 iters: validate                                              # 198-199
final_validate; warn if worse than initial; write fp32→bf16 back to model # 201-209
```

---

## 2. Hyperparameters (with defaults + rationale)

All from `argparse` block `dwq.py:243-302`:

| Flag | Default | Line | Note |
|---|---|---|---|
| `--model` (`-m`) | required | 244-252 | FP teacher repo/path. If `--quantized-model` not given, this is also the source the student is quantized FROM. |
| `--quantized-model` | `None` | 253-258 | Pre-quantized student (cascading: e.g. dynamic-quant → DWQ refine). |
| `--mlx-path` | `mlx_model` | 260-261 | Output dir. |
| `--bits` | `4` | 262-267 | Affine-quant bit-width. LEARNED_QUANTS.md:51-55 — DWQ shines at 2-4 bit; 8-bit "starts so low it's hard to reduce." |
| `--group-size` | `64` | 268-270 | Affine group size. LEARNED_QUANTS.md:56-57 — `--group-size 32` doubles tunable params, often much better. |
| `--num-samples` | `2048` | 271-276 | Train sample count. README at LEARNED_QUANTS.md:40 says 1024 — code default is 2048; **README is stale**. |
| `--max-seq-length` | `1025` | 277 | One past the off-by-one (label shift `batch[:, :-1]` at line 53/175 leaves 1024 input tokens). README says 2048 default — also stale. |
| `--seed` | `123` | 278 | Drives `np.random.seed` + `mx.random.seed` at 312-313. |
| `--learning-rate` | `1e-6` | 279 | Adam lr. LEARNED_QUANTS.md:58-63: lower precision can take higher lr; if oscillating, drop. |
| `--batch-size` | `4` | 280 | Per-step batch. LEARNED_QUANTS.md:42 says default 8 — **code says 4**, README stale. |
| `--data-path` | `allenai/tulu-3-sft-mixture` | 281-286 | HF dataset id. |
| `--grad-checkpoint` | `False` | 287-291 | Wraps `model.layers[0]`'s `__call__` only (`dwq.py:103-104`); since `grad_checkpoint` mutates the class via `type(layer).__call__ = …` (`trainer.py:38`) ALL layers of that type get checkpointing. **Surprising side-effect; flag for porters.** |
| `--target-dir` | `None` | 292-294 | If set, precompute teacher logits to disk. Idempotent: if both `train/` and `valid/` already contain `*.safetensors`, recompute is skipped (315-323). |
| `--targets-only` | `False` | 295-297 | Dump targets and exit (line 355: `exit(0)`). |
| `--pipeline` | `False` | 298-302 | Pipeline-parallel teacher load via `pipeline_load` (`utils.py:594`). |

**Two args bound INSIDE `dwq_quantize` (not exposed on CLI)**:

- `dtype: mx.Dtype = mx.bfloat16` (`dwq.py:78`) — FP storage dtype after each step's write-back. Live arithmetic happens in fp32 (see §10).
- `temperature: float = 2.0` (`dwq.py:80`) — KL-div temperature. `scale = 1 / temperature = 0.5` at line 106 multiplies BOTH student and teacher logits before KL. Standard distillation softening; flatter softmax → richer gradient on non-top tokens. **Not exposed via CLI; porters should expose it.**

---

## 3. Step-by-step `dwq_quantize` (lines 69-209)

### 3a. Distributed handle (82-84)

`group = mx.distributed.init()` returns the world. `world_size = group.size()`, `rank = group.rank()`. `rprint` (86-88) is rank-0 stderr-via-tqdm.

### 3b. `unfreeze` traversal (90-100)

```python
def unfreeze(_, m):
    if (hasattr(m, "bits") and hasattr(m, "group_size")
            and m.mode == "affine" and m.bits < 8):
        m.unfreeze(keys=["scales", "biases"], recurse=False)

model.train()
model.apply_to_modules(unfreeze)
```

`apply_to_modules` walks every submodule. Predicate hits `QuantizedLinear` (and `QuantizedEmbedding`) — both expose `bits`, `group_size`, `mode`. `QuantizedLinear.__init__` ends with `self.freeze()` (`/Users/robert/.pyenv/.../quantized.py:255`); DWQ's call to `m.unfreeze(keys=["scales","biases"], recurse=False)` re-flags ONLY those two leaves as trainable. The packed int weight (`self.weight`) and the optional `self.bias` (linear bias, distinct from quant `biases`) stay frozen.

**Why bits<8?** LEARNED_QUANTS.md:53-55 — at 8-bit the initial KL loss is already so low further reduction is hard; cycles wasted. **Why mode=="affine"?** Because mxfp4/nvfp4/mxfp8 are scalar-format quantization without per-group scales+biases that map onto the same code path; their parameter trees differ (see `mx.nn.layers.quantized` `QuantizedEmbedding/Linear` constructed with `mode` — the `*biases` unpack at `quantized.py:245-248` is empty for non-affine modes, so `self.biases is None`).

### 3c. Gradient checkpointing (103-104)

```python
if gradient_checkpoint:
    grad_checkpoint(model.layers[0])
```

Calls `mlx_lm.tuner.trainer.grad_checkpoint(layer)` (`trainer.py:25-38`):

```python
def grad_checkpoint(layer):
    fn = type(layer).__call__
    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params); return fn(model, *args, **kwargs)
        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)
    type(layer).__call__ = checkpointed_fn
```

Two non-obvious things:

- It's a class-level monkey-patch (`type(layer).__call__ = …`). All instances of that decoder-block class get checkpointing — not just `layers[0]`.
- It re-injects `model.trainable_parameters()` into `mx.checkpoint`'s captured pytree at every forward — required because activations are dropped + recomputed on the backward pass.

### 3d. The loss closure (108-118)

```python
def loss_fn(params, x, targets, lengths):
    model.update(tree_map(lambda x: x.astype(dtype), params))   # 109
    logits = model(x)                                           # 110
    if isinstance(targets, tuple):                              # 111
        targets, ids = targets                                  # 112
        logits = mx.take_along_axis(logits, ids, axis=-1)       # 113
    losses = kl_div_loss(scale * logits, scale * targets)       # 114
    mask = mx.arange(1, 1 + targets.shape[1]) < lengths[:, 1:]  # 115
    ntoks = mask.sum()                                          # 116
    loss = (mask * losses).sum() / ntoks                        # 117
    return loss, ntoks
```

Critical mechanics:

- **Param injection at line 109**: Every forward, `params` (the fp32 trainable tree) is cast to `dtype` (default bf16) and `model.update(...)` writes into the live `nn.Module`. This is how `mx.value_and_grad(loss_fn)` (line 121) gets gradients w.r.t. `params` even though the model otherwise holds frozen tensors. The downcast bf16 copy is what flows through `mx.quantized_matmul`.
- **Two `targets` shapes**:
  1. Tuple `(top1024_logits, top1024_indices)` when `target_fn` reads from `--target-dir`.
  2. Full vocab `targets` when computing teacher live (`target_fn = lambda b,i,split: model(b)` at line 366).
- **`mx.take_along_axis(logits, ids, axis=-1)` (line 113)**: gather student logits at the SAME 1024 vocab positions the teacher saved. KL is then computed over the 1024-entry slice. **Both sides MUST be sliced to the same indices**, otherwise the KL is meaningless. Porters: ensure your gather implementation matches `axis=-1` semantics.
- **Mask (115)**: `lengths` is per-sample `(offset, length)` tuples (`trainer.py:170` `mx.array(list(zip(offsets, lengths)))`). `lengths[:, 1:]` selects the length column. Mask zeros padding tokens past the true sequence length. Note `mx.arange(1, 1+L) < lengths[:, 1:]` — index starts at 1 because `batch[:, :-1]` has already dropped the last token (`dwq.py:53,175`).
- **Loss reduction (117)**: token-mean (sum / ntoks). NOT batch-mean.

### 3e. `step` and autograd (120-126)

```python
def step(inputs, targets, lengths, params):
    (loss, ntoks), grads = mx.value_and_grad(loss_fn)(params, inputs, targets, lengths)
    grads = nn.average_gradients(grads)
    params = opt.apply_gradients(grads, params)
    return loss, ntoks, params
```

- `mx.value_and_grad(loss_fn)` differentiates w.r.t. the FIRST argument (`params`). MLX implements an autograd VJP for `mx.quantized_matmul` w.r.t. its `scales` + `biases` inputs (the pytree leaves). KL has a custom VJP at `tuner/losses.py:359-374` (Metal kernel `_kl_backward_kernel`).
- `nn.average_gradients(grads)`: cross-rank gradient mean. No-op when world_size==1.
- `opt.apply_gradients(grads, params) -> new_params`: returns NEW pytree (functional), Adam state lives inside `opt`.

### 3f. Distributed `all_sum` reductions (124, 144-145, 180-181)

Loss + token counts are summed across ranks via `mx.distributed.all_sum(..., stream=mx.cpu)`. The `stream=mx.cpu` is intentional — small scalar collectives don't justify GPU ↔ CPU bounce; CPU stream avoids serializing the GPU command queue and improves overlap.

### 3g. Validation cadence (198-199 + final 201)

```python
if (it + 1) % 200 == 0:
    valid_loss = validate(params, it=it)
...
valid_loss = validate(params, it=it)  # always at end
```

The `validate` function (128-150) evaluates on `valid_data` with the same `loss_fn` (no gradient), `all_sum`s loss + tokens, returns mean.

### 3h. fp32 → live-dtype write-back (209)

```python
model.update(tree_map(lambda x: x.astype(dtype), params))
```

After the loop, the running fp32 params are cast back to bf16 (default) and copied into the model. Then `save(...)` at `dwq.py:405` serializes weights via `tree_flatten(model.parameters())` (`utils.py:725`).

---

## 4. Pre-target computation (`compute_dwq_targets`, 29-66)

```python
def compute_dwq_targets(model, save_dir, train_data, valid_data, batch_size, max_seq_length, seed):
    rank = mx.distributed.init().rank()
    def _compute_targets(data, path, split):
        if rank == 0:
            (path / split).mkdir(parents=True, exist_ok=True)
        for i, (batch, _) in enumerate(iterate_batches(...)):
            batch = batch[:, :-1]
            logits = model(batch)
            logits = mx.stop_gradient(logits, stream=mx.cpu)   # 56
            mx.eval(logits)
            if rank == 0:
                idx = mx.argpartition(logits, kth=-1024, axis=-1)[..., -1024:]   # 59
                logits = mx.take_along_axis(logits, idx, axis=-1)                # 60
                file = path / f"{i:010d}.safetensors"                            # 62
                mx.save_safetensors(file, {"logits": logits, "indices": idx})    # 63
```

**Why disk-cache the teacher?**
- The FP teacher is HUGE (Qwen3-32B-fp16 is 64 GB resident) and runs forward every step otherwise. Caching trades RAM for disk + lets us `del model` (`dwq.py:387`) after target compute, freeing peak GPU/RAM for the trainable student.
- `--num-samples 2048` × `--batch-size 4` → 512 batches per train; per file shape is `(4, 1024, 1024)` of bf16 logits + same shape int32 indices ≈ `4*1024*1024*(2+4) = 24 MiB/batch * 512 = ~12 GiB` train, `~32/4 = 8` files × 24 MiB = ~192 MiB valid. Trivial vs holding teacher resident.

**Why top-1024 not full vocab?** Vocabularies are ~150K-256K (Qwen3 is 151,936). Top-1024 is ~0.7% of vocab but captures ~all probability mass under softmax (long tail is sub-1e-6). KL on a truncated top-K is a standard distillation approximation and is exact-enough when `temperature=2.0` flattens the softmax: even with T=2, the 1024th-largest logit pre-softmax has post-softmax mass <<1e-3. Flag for porters: `argpartition` returns UNSORTED top-1024 — student must re-gather via the SAME `idx` (line 113), not its own top-K.

**Filename convention**: `{i:010d}.safetensors` (zero-padded 10-digit batch index). Two splits (`train/`, `valid/`). Reload at line 360-361:
```python
targets = mx.load(target_dir / split / f"{idx:010d}.safetensors")
return targets["logits"], targets["indices"]
```

**Idempotency check (`dwq.py:317-321`)**: existence of `*.safetensors` in BOTH `train/` and `valid/` skips recompute. No checksum. Porters: this is fragile — if a previous run was interrupted mid-write, half-populated dirs will silently use stale targets. Consider a sentinel file (`done.json`) in the port.

**Open question / blocker**: `iterate_batches` is called with `seed=seed` BOTH at target-dump time (line 47) and at training time (line 170). The shuffling `np.random.permutation` at `trainer.py:141` is deterministic given `seed`, so batch `i` at dump time matches batch `i` at train time. **If the porter forgets the matching seed, the cached targets will be applied to the wrong batches and DWQ will quietly degrade quality.** Make this an invariant, not a convention.

---

## 5. Distributed semantics

`mx.distributed.init()` returns a `Group` with `.rank()`, `.size()`, supporting `mx.distributed.all_sum`, `nn.average_gradients`. With one host (laptop) `world_size=1`, all collectives are identity.

Two distributed modes:

- **Data parallel (default)**: `iterate_batches` with `comm_group` (`trainer.py:124-129`) slices each batch into `step=world_size` interleaved sub-batches; each rank processes its slice; `nn.average_gradients` (`dwq.py:124`) means weights stay in sync. `dwq.py:309-310` rounds `num_samples` UP to a multiple of `world_size` to keep the data evenly partitioned.
- **Pipeline parallel (`--pipeline`)**: `pipeline_load` (`utils.py:594-595`) shards the teacher across ranks vertically (decoder-block stripes). Used at line 334-336 only for the teacher model. Surprising: pipeline mode does NOT round `num_samples` (`dwq.py:309` `if not args.pipeline`) — pipeline shards the model not the data, so each rank sees the full batch.

For hf2q port: a single-host laptop port can ignore distributed entirely (treat `all_sum = identity`, `average_gradients = identity`, `world_size = 1`). Pipeline parallelism is a separate optimization tier.

---

## 6. `mx.set_wired_limit(max_rec_size)` at line 391

```python
if mx.metal.is_available():
    max_rec_size = mx.device_info()["max_recommended_working_set_size"]
    mx.set_wired_limit(max_rec_size)
```

`mx.set_wired_limit(N)` tells Metal/MLX it MAY pin up to `N` bytes of unified memory as wired (non-pageable). This avoids macOS swapping out hot tensors mid-training; on a 64 GB M-series, swap during DWQ would tank throughput and can OOM-crash via `dynamic_pager` SIGKILL (cf. project memory entry on Qwen35/36 reconvert SIGKILL @ 199 GB peak).

`mx.device_info()["max_recommended_working_set_size"]` is what Apple's IOKit reports as a safe wireable headroom — typically ~75% of physical RAM on Apple Silicon. Documented in MLX at `https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.metal.set_wired_limit.html` (the function lives under `mx.metal` historically; `mx.set_wired_limit` is the re-exported alias).

When would it fail? If max_rec_size > available unified memory (e.g. another process holds wired pages already), set_wired_limit will silently clip per Apple docs. Not an exception path in normal use.

**hf2q equivalent**: there is no Rust `mlx_native::set_wired_limit` binding visible in /opt/mlx-native today. Either (a) call through a small C-shim into `IOServiceGetMatchingService` + the same Metal API, or (b) for a CPU port, no equivalent — Metal-specific. **Open question for hf2q port: do we need it on day-1, or is the tested Metal path good enough?** Probably fine without on day-1 because hf2q manages its own residency via ResidencySet (cf. ADR-013 work) — set_wired_limit is mlx-only convenience.

---

## 7. Save format (line 405)

`save(args.mlx_path, args.model, q_model, tokenizer, config)` (`utils.py:925-950`) does five things:

1. Resolve `src_path_or_repo` to a local path (HF download if needed).
2. `save_model(dst_path, model, donate_model=True)` (`utils.py:714-771`):
   - `weights = dict(tree_flatten(model.parameters()))` — every parameter leaf as `path/leaf_name → mx.array`.
   - Shard via `make_shards` (`utils.py:598-619`) at 5 GB/shard default.
   - Filename: `model.safetensors` if 1 shard, else `model-{i:05d}-of-{N:05d}.safetensors`.
   - Metadata: `{"format": "mlx"}` per shard. Index file: `model.safetensors.index.json` with `{metadata: {total_size, total_parameters}, weight_map: {leaf_name: shard_filename, ...}}`.
   - `donate_model=True` (line 942) → blanks the model's parameter tree to free RAM during save (line 743).
3. `save_config(config, ...)` writes `config.json`. Config carries `quantization` (`utils.py:821, 832`) AND duplicate `quantization_config` (`utils.py:845` "support hf model tree #957"). Both keys describe `{"group_size", "bits", "mode"}` plus per-layer-path overrides if `quant_predicate` was used.
4. `tokenizer.save_pretrained(dst_path)`.
5. Copy `*.py` and `generation_config.json` from src; write `README.md` (model card) via `create_model_card`.

Quantized weight serialization is the affine-quantized layout that `mx.dequantize` understands: weight (int packed), scales (per-group fp), biases (per-group fp), all as named leaves under each `QuantizedLinear` path.

---

## 8. What hf2q needs to port — checklist

| Item | mlx-lm artifact | hf2q gap |
|---|---|---|
| Train loop scaffolding | `dwq.py:108-209` | Straightforward Rust port. |
| Adam optimizer w/ bias correction | `mx.optimizers.Adam(lr, bias_correction=True)` (393) | hf2q has no autograd/optimizer. Implement a fixed Adam: `m = β1 m + (1-β1) g; v = β2 v + (1-β2) g²; m̂ = m / (1-β1ᵗ); v̂ = v / (1-β2ᵗ); θ ← θ - lr · m̂ / (√v̂ + ε)` with `β1=0.9, β2=0.999, ε=1e-8` (mlx defaults). Bias correction is the `m̂/v̂` denominators. |
| `mx.value_and_grad` | line 121 | **Biggest port gap.** hf2q has no autograd. We must write a hand-derived backward for: (a) `mx.quantized_matmul` w.r.t. `scales` and `biases`, (b) the model's transformer stack from `scales/biases` through to logits. Option A: reverse-mode hand-rolled VJP for each kernel (large engineering bill). Option B: use a tiny numerical JVP just for `scales/biases` per group — but with millions of groups this is slow. Option C: build a small autograd shim around the trainable subset only; treat the frozen quantized matmul output as the differentiation boundary by using STE w.r.t. `scales/biases`. **Open question §9.** |
| Per-batch gradient | `step` at 120-126 | Drives whatever autograd choice. |
| `unfreeze keys=["scales","biases"]` semantics | `dwq.py:97`, `quantized.py:255` | In MLX, `QuantizedLinear.__init__` ends with `self.freeze()`; `unfreeze(keys=…)` flips the tracked-gradient flag on those leaves only. **Net effect: scales (per-group fp16/bf16) and biases (per-group fp16/bf16, optional, present in affine mode) are real fp tensors that participate in autograd, even though `self.weight` (the int-packed payload) is frozen.** For hf2q, the equivalent is: keep the int-packed weight + per-group fp `scale[group]` and `zero_point[group]` (mlx calls it `bias`). Make `scale` and `zero_point` the "trainable parameter set"; treat `weight_int` as fixed. |
| KL-div loss with VJP | `losses.py:344-386`, `_make_kl_forward_kernel` / `_make_kl_backward_kernel` | Rust port needs forward (sum over V of pᵢ·(log pᵢ − log qᵢ) with logsumexp normalization) AND backward (`dq = (q − p) · cotangent / V` after softmaxing). Reference `nn.losses.kl_div_loss` fallback at `losses.py:381-386` for the math. |
| Gradient checkpointing | `tuner/trainer.py:25-38` | Per-block re-forward on backward. Not strictly required for v0; LEARNED_QUANTS.md:67-74 says "use a smaller batch size" as alternative memory reducer. Defer. |
| `mx.distributed` | wrapped at every reduction site | Single-host port can stub all collectives as identity. |
| `iterate_batches` | `trainer.py:102-173` | Port: sort-by-length, pad-to-32, np.random.permutation seed semantics. |
| `iterate_batches` MUST be reproducible across target-compute and train-time | seed=seed at lines 47 and 170 | Make this an enforced invariant in the port. |
| Top-1024 logit cache | `dwq.py:59-63` | Argpartition + take_along_axis + safetensors I/O. Standard. |
| Save in HF-shape MLX dir | `utils.py:925-950`, `714-771` | hf2q already has GGUF; for cross-tool compat, we'd need to either (a) write `model.safetensors[.index.json]` + `config.json` with `quantization` block, OR (b) define an hf2q-native save format and run an `mlx-lm.fuse`-like converter externally. |

---

## 9. Open questions / blockers for hf2q port

1. **The big one — autograd through `mx.quantized_matmul` w.r.t. `scales/biases`.** The forward op is `y = (W_int * scale + bias) @ x` (per-group). Gradient w.r.t. `scale` is `dL/dy · (W_int @ x)` summed within each group; gradient w.r.t. `bias` is `dL/dy · 1` summed within each group. STE not needed because the integer `W_int` is frozen — there is no `round/quantize` op in the gradient path. **This is good news for hf2q: we do NOT need autograd through `mx.round` or a straight-through estimator. We need a single hand-derived backward kernel for affine-quantized matmul w.r.t. the per-group scale and bias scalars.** That kernel is small. The backward through the rest of the transformer (attention → MLP → embedding) is the hard part — but if all OTHER weights are frozen FP teacher-side already-resident, the gradient only needs to flow from the loss back to the QuantizedLinear's `scales/biases`, which means the hf2q port has to carry intermediate activations of every QuantizedLinear forward pass for backward. This is non-trivial but bounded.

2. **Where is the fp32 master copy stored?** `dwq.py:153-156` upcasts `model.trainable_parameters()` to fp32 ONCE. From then on, `params` (Python local) is the fp32 master; `model` itself holds bf16 copies that get refreshed every loss_fn (line 109). Porters: replicate exactly — keeping master in fp32 prevents accumulation drift; the bf16 copy is only what the forward kernel consumes (and what falls into `mx.quantized_matmul`).

3. **Does `mx.quantized_matmul` accept fp32 scales/biases as inputs in production?** Need to check whether the Metal kernel accepts mixed-precision input or whether `model.update(tree_map(lambda x: x.astype(dtype), params))` (line 109) is forced for kernel compatibility, not just performance. Inspect `quantized.py:265-278` — it uses whatever dtype `self["scales"]` carries, so the bf16 cast at line 109 is a kernel-input dtype constraint. Porters' kernel must match.

4. **`unfreeze(..., recurse=False)` (line 97)** — `recurse=False` means only the immediate module's parameters are unfrozen. Why? Because `apply_to_modules` already recurses. If we said `recurse=True` we'd double-unfreeze nested submodules. Subtle but correct.

5. **Top-1024 → quality trade-off**: porters considering smaller K (e.g. top-256 to halve disk) need to validate against hf2q's eval suite (winogrande/arc/hellaswag) before committing. mlx-lm ships 1024 with no further explanation.

6. **Sequence length 1025 oddity** — code says 1025 but downstream is `batch[:, :-1]` so effective is 1024. This is a "context-length-plus-one for next-token shift" idiom; harmless once understood. Don't accidentally use 1025 as the kernel context.

---

## 10. Interesting tricks / non-obvious choices

1. **`mx.stop_gradient(logits, stream=mx.cpu)` (line 56) in `compute_dwq_targets`** — Comment at line 55: *"Hack to make the last op pre-eval on the CPU to avoid even timeout."* Translation: after the GPU forward of the teacher, the result tensor is enqueued but not yet realized. If the immediate next op is `mx.eval(logits)` (line 57) on the GPU stream, the GPU may still be saturated and the eval can hit Metal's command-buffer watchdog ("even timeout" is a typo for "Metal command-buffer eval timeout"). Wrapping in `mx.stop_gradient` with `stream=mx.cpu` schedules a tiny CPU-stream op that depends on the GPU output, which forces a GPU→CPU dependency edge so the GPU finishes its work, releases the CB, and only then does the CPU stream's no-op (stop_gradient) complete. This is a Metal CB hygiene trick — **flag for porters: hf2q's mlx-native already has CB-budget and ResidencySet machinery (cf. ADR-013, ADR-015) that addresses the same class of issue differently. We probably do NOT need this exact stop_gradient hack but we DO need the equivalent: a CB break between long teacher forward and the eval that triggers actual GPU sync.**

2. **`mx.argpartition(logits, kth=-1024)` (line 59)** — Top-K via partial sort, not full sort. `argpartition` is O(V) vs `argsort`'s O(V log V); for V=151,936 that's ~17× faster. Returned indices are UNSORTED within the top-K, but that doesn't matter because student and teacher both gather with the SAME `idx` and KL is permutation-invariant in the vocab axis after gather.

3. **fp32 master, bf16 storage (lines 78, 109, 153-156, 209)** — Standard mixed-precision training pattern: model uses bf16 for memory/throughput; trainable params live in fp32 in the optimizer state to prevent loss-of-precision on small Adam updates (lr=1e-6 means each update may be in the bf16 last-bit noise). The "every step we re-cast and re-inject" pattern at line 109 is the price of MLX's design where the model holds the parameters but the optimizer holds the master copy.

4. **Final-vs-initial validation warning (lines 202-207)** —
   ```python
   if initial_valid_loss < valid_loss:
       rprint("❌❌❌\n[WARNING] Final validation loss {valid_loss:.3f} is "
              "worse than initial validation loss {initial_valid_loss:.3f}."
              " Model quality will likely be degraded.\n❌❌❌")
   ```
   Surprising failure mode: DWQ can DEGRADE the quantized student vs the unfine-tuned quantization (post-`quantize_model` baseline before any DWQ steps). Causes: lr too high (loss oscillating per LEARNED_QUANTS.md:58-63), too few samples, or starting from 8-bit where there's no headroom. The warning is informational only — no rollback. **Porters: implement rollback. Save `params_init = copy(params)` after line 156, and at line 209 use whichever (init or final) had lower validation loss.** Cheap insurance.

5. **`opt = optimizers.Adam(learning_rate=args.learning_rate, bias_correction=True)` at line 393** — `bias_correction=True` is NOT mlx's Adam default in older versions; explicitly passed here. Standard Adam-with-bias-correction. Matters because at iter=1, without correction `m̂` is biased toward 0 and the first step is too small.

6. **`donate_model=True` in `save` (utils.py:942 → 743)** — Blanks the parameter pytree DURING save to free RAM. Porters: harmless if save is the last action (it is), surprising if anyone reuses the model after `save()` returns (they shouldn't).

7. **`args.targets_only` semantics (354-355)** — `exit(0)` not `return`. Useful for two-stage runs: `mlx_lm.dwq --targets-only --target-dir /tmp/t` on a large host, then `mlx_lm.dwq --target-dir /tmp/t` on a smaller host. Porters: replicate the exit-code-0 contract so external orchestration scripts can chain.

8. **Sort-by-length in iterate_batches (`trainer.py:115`)** then `np.random.permutation(len(batch_idx))` (141) — sequences are GROUPED by similar length per batch (less padding), but BATCHES are then shuffled. Reduces wasted compute on padding without losing shuffle guarantees. Porters: replicate or padding overhead bites at long-seq-length DWQ.

---

## Closing notes for the next porter

- The DWQ algorithm is small (~200 LOC). The cost is the autograd dependency. **Confirm whether hf2q can target a "trainable subset autograd" rather than a full autograd port — that's the make-or-break decision.** §9.1 makes the case that affine-quant's `scales/biases` backward is hand-derivable in <500 LOC of Metal/Rust per kernel.
- README defaults (LEARNED_QUANTS.md:36-43) lag the code by ~2× on `--num-samples`, `--batch-size`, `--max-seq-length`. Trust `dwq.py:243-302`, not the README.
- The seed-reproducibility coupling between target-dump and train-time iterate_batches (§4 open question) is the silent-correctness landmine. Make it an invariant in the port — fail loudly if the seeds differ.
- Distillation temperature 2.0 is hard-coded (`dwq.py:80`) and not on the CLI. Expose it in the hf2q port; values of 1.5-3.0 are commonly tuned for distillation quality.
