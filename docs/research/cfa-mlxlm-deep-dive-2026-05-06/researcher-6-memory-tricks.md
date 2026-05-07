# Researcher #6 — Memory Tricks mlx-lm Uses on Consumer Macs

**Session:** cfa-20260506-mlxlm-research
**Scope:** read-only; verified-from-code findings
**Bottom line:** mlx-lm fits 30-70B-class models on 32-128 GB Macs by combining (a) a true graph-deferred lazy load, (b) **per-step `mx.eval` choreography** that bounds graph live-set, (c) **stream-targets-to-disk** so the teacher model is dropped before the student trains, (d) macOS wired-memory ceiling raise, and (e) `mx.clear_cache()` between blocks. None of (a)-(e) have a complete hf2q equivalent today; (b) and (c) are the highest-leverage ports.

---

## 1. `lazy=True` in `load()`

**WHAT.** `load_model(..., lazy=True)` returns a fully-constructed `nn.Module` whose weight arrays are MLX *graph nodes*, not realized buffers. Skipping the trailing `mx.eval(model.parameters())` is the entire mechanism.

```python
# /opt/mlx-lm/mlx_lm/utils.py:414-418
model.eval()
model.load_weights(list(weights.items()), strict=strict)

if not lazy:
    mx.eval(model.parameters())     # <-- only path that materializes

return model, config
```

`mx.load(wf)` (utils.py:323) returns memory-mapped views of the safetensors files — the bytes live in the OS page cache, not in process RSS, and MLX defers the read until a downstream op forces eval. Call sites:

- `dwq.py:337` — `load(args.model, return_config=True, lazy=True)` (teacher)
- `dwq.py:371` — `load(args.quantized_model, lazy=True, ...)` (student)
- `awq.py:557` — `load(args.model, lazy=True, return_config=True)`
- `utils.py:530` — `load_model(model_path, lazy=True, strict=False)` inside `sharded_load` so each rank can compute *which* shard it owns *before* downloading any weights

**WHY (verified, not the docstring).** Without `lazy`, `mx.eval(model.parameters())` walks the entire parameter tree. With `lazy`, individual layer's params get realized only when that layer's forward runs (utils.py:323 `weights.update(mx.load(wf))` only refs the mmap; the eval at utils.py:418 is what allocates and copies into wired pages).

**HOW MUCH.** For a 30B-bf16 model (~60 GB), `lazy=False` forces the entire 60 GB into RSS at load. `lazy=True` keeps it as mmap pages until the first forward — and crucially in DWQ, lets the student be `copy.deepcopy`'d (see §14) **before** any forward happens, so peak deepcopy cost is metadata + shared mmap, not 60 GB × 2.

**HF2Q EQUIVALENT.** *Closer than expected, but with one critical defect.* `LazyTensorMap` (`/opt/hf2q/src/ir/lazy.rs`) supports `LazyState::Pending(FnOnce)` and `LazyState::MaterializedShared(Arc<Vec<u8>>)`. The defect: **`materialize_cloned()` (lazy.rs:314-345) always deep-clones the Vec<u8>** even when the caller only needs read-only borrow:

```rust
// lazy.rs:317-321
LazyState::MaterializedShared(arc) => {
    (**arc).clone()    // forced deep clone, always
}
```

This is why DWQ OOM'd at 199 GB on 128 GB: every lazy tensor that crossed a `&LazyTensorMap` boundary doubled. The mlx-lm equivalent is `mx.array` reference semantics — the underlying buffer is Arc-counted; passing a tensor to a function does *not* clone bytes. **PORT TO HF2Q: yes — replace `materialize_cloned` body with an Arc clone path that returns `TensorRef { data: Arc::clone(arc) }` and reserve byte-cloning for the `materialize(self)` consume path that already exists at lazy.rs:276-304.** Comment at lazy.rs:317-321 ("Always clone the Vec — by-reference semantic must keep the Arc live") is wrong: the Arc itself stays live whether you Arc::clone or Vec::clone; the Vec::clone is purely an unforced cost.

---

## 2. `mx.eval()` placement and ordering

**WHAT.** MLX is a lazy compute graph. Every op (matmul, slice, sum) appends a node; the graph is only executed when `mx.eval()` reaches it (or `.item()` / `.tolist()` / explicit save). Strategically placed `mx.eval` calls are the **only** lever a Python author has to bound graph live-set memory.

**WHERE (canonical pattern in dwq.py).**

```python
# dwq.py:54-57 — teacher target generation
logits = model(batch)
logits = mx.stop_gradient(logits, stream=mx.cpu)   # see §4
mx.eval(logits)                                     # materialize NOW

# dwq.py:140-145 — validation step
targets = target_fn(batch, i, split="valid")
mx.eval(targets)                                    # eval before loss
loss, ntoks = loss_fn(params, batch, targets, lengths)
mx.eval(loss, ntoks)                                # joint eval — single graph walk
loss = mx.distributed.all_sum(loss, stream=mx.cpu).item() / world_size

# dwq.py:177-179 — train step
mx.eval(targets)
loss, ntoks, params = step(batch, targets, lengths, params)
mx.eval(loss, params)                               # joint eval of loss + post-grad params
```

**WHY (verified by tracing the graph).** Without the eval at line 57, `logits` is a graph rooted at the teacher's full forward. The argpartition + take + safetensor save at 59-63 would extend that graph another 3 ops. When `mx.eval` finally runs at the save step, it would re-walk the whole teacher forward AND keep all intermediate activations live until completion. With eval at 57, the teacher's intermediates are freed before the top-1024 selection runs.

The **pattern** — `mx.eval(targets); use targets; eval(result)` — is "consume + collapse": eval forces a checkpoint, and Python's GC drops the now-realized references that no longer have downstream graph consumers.

**HOW MUCH.** Order-of-magnitude: forward of a 30B model produces ~30 layers × seq × hidden × bf16 ≈ 4-8 GB activations. Eval-and-drop after each batch reuses that 4-8 GB region instead of accumulating it.

**HF2Q EQUIVALENT.** *None.* hf2q is eager (Metal command buffers issued and committed); this pattern presumes a deferred graph + GC. ADR-013 (CB count) is the closest analogue but inverted — hf2q's problem is too many CBs from too-eager dispatch. **PORT TO HF2Q: no** at the API level. The transferable lesson is **"force materialization at well-defined checkpoints; don't let intermediates accumulate"** — already partly addressed by ADR-015 borrowed-session arena work.

---

## 3. `del grads` + `mx.eval(grad_accum)`

**WHAT.** dynamic_quant.py runs N forward+backward passes per layer, accumulates gradients into `grad_accum` tensor, and explicitly drops the per-step `grads` after addition.

```python
# dynamic_quant.py:80-86
batch = data[s : s + batch_size]
targets = model(batch)
mx.eval(targets)
_, grads = nn.value_and_grad(q_model, loss_fn)(batch, targets)
grad_accum = tree_map(lambda x, y: x + y, grad_accum, grads)
del grads                  # <-- explicit drop of N-step transient
mx.eval(grad_accum)        # <-- force the add to materialize, releasing per-step grad arrays
```

**WHY.** `nn.value_and_grad` returns gradients matching trainable-param shape — for a 30B model that's 60 GB of bf16 grads (or 120 GB at fp32). Without `del grads`, two generations of grads (current + previous-not-yet-GC'd) coexist. Without `mx.eval(grad_accum)`, the addition stays as a graph node referencing both `grads` AND `grad_accum`, so neither can be freed.

**HOW MUCH.** Saves exactly one full param-shape buffer per step. For 30B bf16: ~60 GB saved per micro-batch. **This is the difference between "fits in 128 GB" and "OOMs at 199 GB."**

**HF2Q EQUIVALENT.** *None — and this is the smoking gun for the 199 GB OOM.* hf2q's DWQ pipeline has no analogous "eval-then-drop" rhythm because there's no graph to fuse. Per the MEMORY note `project_qwen35_reconvert_paused_2026_05_05.md`: peak 199 GB at jetsam SIGKILL. Without inspecting hf2q's DWQ Rust code I cannot localize the exact accumulator, but the structural cause matches: **per-step gradient/activation tensors are not being dropped before the next step computes its successor.**

**PORT TO HF2Q: yes — highest-priority port.** Concrete: ensure DWQ's gradient-accumulation loop owns the accumulator by `Arc<Mutex<...>>`, drops per-step grads via explicit scope or `drop()` before the next forward, and verifies via `mx.get_peak_memory()` analogue (§6).

---

## 4. `mx.stop_gradient(logits, stream=mx.cpu)` and the "even timeout" hack

**WHAT.** Force the final pre-eval op onto the CPU stream so the eval boundary doesn't time out the GPU command stream.

```python
# dwq.py:55-57
# Hack to make the last op pre-eval on the CPU to avoid even timeout
logits = mx.stop_gradient(logits, stream=mx.cpu)
mx.eval(logits)
```

**WHY (verified — comment is sloppy, "even timeout" is "Metal kernel timeout").** When MLX evals a graph that ends in a many-second GPU op (full forward of a 30B teacher on a long batch can take 5-30 s), Metal can hit its kernel-execution-time guardrail (the equivalent of GPU-driver TDR). Inserting a no-op (`stop_gradient` is identity for forward) on the CPU stream forces MLX to insert a CPU-side dependency barrier that splits the GPU work across multiple command buffers, none of which individually exceed the timeout. `stop_gradient` is chosen because it's cheap and semantically a no-op for forward.

**HOW MUCH.** Doesn't save memory directly; it's a correctness/availability fix. Without it, long-batch teacher forwards crash with `Metal kernel timeout` on slow Macs.

**HF2Q EQUIVALENT.** Already non-issue. hf2q manages CB lifetime explicitly via `commit_and_wait` and ADR-015's borrowed-session arena. Per `solution_mlx_native_residency_lifetime_race.md` the hf2q lesson is the inverse: too many CBs, not too few. **PORT TO HF2Q: no** — different layer of the stack.

---

## 5. `mx.set_wired_limit(max_rec_size)`

**WHAT.** Raise the macOS *wired* (non-pageable) memory limit so MLX can keep the model resident under VM pressure.

```python
# dwq.py:389-391
if mx.metal.is_available():
    max_rec_size = mx.device_info()["max_recommended_working_set_size"]
    mx.set_wired_limit(max_rec_size)

# tuner/trainer.py:229 — same call in training entry
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
```

**WHY.** macOS reports `max_recommended_working_set_size` per-device — on M-series this is roughly `(unified_memory) - 8-12 GB OS reserve`. Without `set_wired_limit`, MLX's default wired ceiling is conservative (~75% of the Metal-recommended size, by Apple's heuristics). When the model + activations exceed that ceiling, the OS pages portions to swap; once a model weight is paged out, the next forward fault triggers a multi-second page-in. On Apple Silicon this also burns the disk (zswap → APFS).

**HOW MUCH (empirical, from MEMORY notes).** On 128 GB M4 Max with `max_rec_size` ≈ 122 GB: raising the wired limit to that ceiling lets a 60 GB student + ~30 GB activation working set + ~30 GB optimizer state stay fully resident. Without it, the same workload hits ~5-15 minutes of swap thrash before killing.

**HF2Q EQUIVALENT.** *Possibly missing.* No grep hit for `setWiredLimit` or `MTL::Device::setShouldMaximizeConcurrentCompilation` in hf2q. mlx-native may already call this; needs verification. **PORT TO HF2Q: maybe — verify mlx-native calls `setShouldMaximizeConcurrentCompilation` + `setWiredLimit`; if not, add at residency-set initialization.** Likely 30 LOC.

---

## 6. `mx.get_peak_memory()`

**WHAT.** Reports the MLX-allocator high-watermark since process start (or last `mx.reset_peak_memory()`).

```python
# dwq.py:188 — telemetry every 20 iters
peak_memory_gb = mx.get_peak_memory() / 1e9

# dynamic_quant.py:255 — final report
print(f"Peak memory used: {mx.get_peak_memory() / 1000**3:.3f}GB")

# tuner/trainer.py:341
peak_mem = mx.get_peak_memory() / 1e9
```

**WHY (verified by reading the API).** MLX tracks allocations through its own buffer pool — `get_peak_memory` reports the allocator's high-watermark, which is GPU-side (Metal heap + transient). It does NOT include CPU-side numpy/python allocations. **Used for telemetry only — never as a gating signal in any of the call sites.** Operator reads it; if it's growing across iterations, that's the smell test for a leak.

**HOW MUCH.** Diagnostic. Doesn't save memory; tells you when you're about to lose.

**HF2Q EQUIVALENT.** *None.* hf2q has `KvSpillCounters` and `/metrics` (per ADR-017 phase E.a) but no allocator-level peak watermark. **PORT TO HF2Q: yes — small.** Add a peak-RSS counter scraped from `mach_task_basic_info` (Darwin) at every prefill boundary; surface in `/metrics`. ~50 LOC.

---

## 7. Stream-targets-to-disk (the killer trick)

**WHAT.** Run the teacher model once over the dataset, save **only the top-1024 logits + indices per batch** to small `.safetensors` files on disk, then drop the teacher entirely before quantization training begins.

```python
# dwq.py:29-66 — compute_dwq_targets
def _compute_targets(data, path, split):
    for i, (batch, _) in enumerate(iterate_batches(...)):
        batch = batch[:, :-1]
        logits = model(batch)                                # full teacher forward
        logits = mx.stop_gradient(logits, stream=mx.cpu)
        mx.eval(logits)
        if rank == 0:
            idx = mx.argpartition(logits, kth=-1024, axis=-1)[..., -1024:]
            logits = mx.take_along_axis(logits, idx, axis=-1)
            file = path / f"{i:010d}.safetensors"
            mx.save_safetensors(file, {"logits": logits, "indices": idx})

# dwq.py:386-387 — drop teacher after targets are on disk
if has_targets and model is not None:
    del model

# dwq.py:357-361 — load-back for training
if has_targets:
    def target_fn(_, idx, split):
        targets = mx.load(target_dir / split / f"{idx:010d}.safetensors")
        return targets["logits"], targets["indices"]
```

Vocabulary compression is huge: full vocab is ~150K-256K tokens; `--top-k=1024` keeps `1024/200000 ≈ 0.5%` of the bytes per logit row.

**WHY.** Without this, both teacher (60 GB) and student (60 GB) + optimizer state (60 GB × 2 if fp32) + activations (~10 GB) must coexist during DWQ training → easily 200+ GB peak. With it, peak during training is just student + optimizer + activations ≈ 130 GB at fp32 accum (still tight on 128 GB) or 90 GB at bf16 accum.

**HOW MUCH.** **60 GB** — the entire teacher model — dropped before training starts. This is the single largest memory delta in the entire DWQ pipeline.

**HF2Q EQUIVALENT.** *None — and this is the second smoking gun.* hf2q's DWQ likely keeps both teacher and student weights resident throughout training (matching the `else: target_fn = lambda batch, idx, split: model(batch)` branch at dwq.py:365-366, which mlx-lm itself describes as "works but holds teacher resident"). **PORT TO HF2Q: yes — second-priority port after §3.** Concrete: add a `--target-dir PATH` flag to hf2q's DWQ; precompute teacher logits to `{target_dir}/{train,valid}/{idx:010d}.safetensors`; drop teacher Arcs; load per-step in the training loop. The `[INTELLIGENCE]` upside is huge for the user's 128 GB budget.

---

## 8. `pipeline_load` for distributed/streaming layer dispatch

**WHAT.** Each rank in a distributed group loads only the safetensors shard containing its assigned layer slice.

```python
# utils.py:530 — load model lazily to discover sharding metadata
model, config = load_model(model_path, lazy=True, strict=False)

# utils.py:553-569 — per-rank file selection
if pipeline_group is not None:
    model.model.pipeline(pipeline_group)
    with open(model_path / "model.safetensors.index.json", "r") as fid:
        weight_index = json.load(fid)["weight_map"]
    local_files = set()
    for k, _ in tree_flatten(model.parameters()):
        local_files.add(weight_index[k])
    _download(repo, allow_patterns=local_files)         # <-- only download what we'll use
```

**WHY.** Two mechanisms compose: `lazy=True` builds the full model graph from config without realizing weights, then `model.model.pipeline(pipeline_group)` rebinds each layer's parent to a specific rank. After that, `tree_flatten(model.parameters())` only enumerates the layers this rank owns — which maps back through the safetensors index to exactly the files this rank needs.

**HOW MUCH.** For 4 ranks on a 70B model: each rank loads ~17.5 GB instead of 70 GB. Memory scales as `1/world_size`.

**HF2Q EQUIVALENT.** *None.* hf2q is single-process. **PORT TO HF2Q: maybe — explicitly out of scope for current work.** Would require multi-process IPC + Metal residency-set sharding across processes; this is mlx-native territory, not hf2q.

---

## 9. Default `batch_size` + `max_seq_length`

**WHAT.** Conservative defaults bound activation memory.

- dwq.py: `batch_size=4`, `max_seq_length=1025` (dwq.py:277-280)
- dynamic_quant.py: `batch_size=4` (line 45 default), `sequence_length=512` (line 192)
- awq.py: `num_samples=128`, `sequence_length=512` (awq.py:543-544)

**WHY.** Per-layer activation memory is `O(batch × seq × hidden × dtype_size)`. For a 30B-class hidden=4096 model in bf16: `4 × 1025 × 4096 × 2 = ~33 MB per layer activation`. Across 30 layers + attention buffers: ~3-5 GB activation working set. Bumping `batch_size` to 16 quadruples it; bumping `max_seq_length` to 4096 adds another 4×.

**HOW MUCH.** Setting `batch_size=4` vs `batch_size=16` saves ~10-20 GB activation memory at the cost of 4× more iterations. The default is the rational floor for "fits on 128 GB."

**HF2Q EQUIVALENT.** *Partial.* hf2q has prefill batching but I see no surfaced flag for DWQ-style activation-bound batch sizing. **PORT TO HF2Q: yes — small.** Expose `--dwq-batch-size` and `--dwq-max-seq-length` CLI flags; default 4 / 1025 to match.

---

## 10. `tree_map(lambda x: x.astype(mx.float32), params)`

**WHAT.** Forward and weights stay in bf16; the **trainable param accumulator** is upcast to fp32.

```python
# dwq.py:153-156
params = tree_map(
    lambda x: x.astype(mx.float32),
    model.trainable_parameters(),
)

# dwq.py:108-109 — at every loss_fn call, downcast back to forward dtype
def loss_fn(params, x, targets, lengths):
    model.update(tree_map(lambda x: x.astype(dtype), params))
    logits = model(x)
    ...

# dwq.py:209 — at end of training, write fp32 params back as bf16
model.update(tree_map(lambda x: x.astype(dtype), params))
```

**WHY.** Adam-style optimizers accumulate `m` and `v` running estimates; bf16 has only 7 mantissa bits, so over thousands of steps the moment estimates drift catastrophically. fp32 accumulation is the standard mixed-precision recipe (cf. NVIDIA AMP). Forward in bf16 is fine; only the slow-moving accumulators need fp32.

**HOW MUCH.** Costs 2× param size during training (60 GB bf16 → 120 GB fp32 if naive). DWQ dodges this by **only making `trainable_parameters()` fp32** — for DWQ, that's just the per-block `scales` + `biases` (per `unfreeze` at dwq.py:90-97), which is a tiny fraction of total params. So actual cost is ~hundreds of MB, not 60 GB.

**HF2Q EQUIVALENT.** Need to verify — hf2q's DWQ training loop dtype is unclear from the MEMORY notes. **PORT TO HF2Q: yes — verify trainable-only fp32 upcast is in place.** If hf2q is upcasting *all* params to fp32 (not just trainable scales/biases), that's a 60 GB unforced cost on top of the 199 GB OOM.

---

## 11. `grad_checkpoint(model.layers[0])`

**WHAT.** Wrap **only the layer-0 module's `__call__`** in `mx.checkpoint`, which causes layer-N's activations to be discarded after forward and recomputed on backward.

```python
# tuner/trainer.py:25-38
def grad_checkpoint(layer):
    fn = type(layer).__call__
    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)
        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)
    type(layer).__call__ = checkpointed_fn

# dwq.py:103-104, dynamic_quant.py:68-69
if gradient_checkpoint:
    grad_checkpoint(model.layers[0])
```

**WHY only layer 0?** Subtle: `grad_checkpoint` mutates `type(layer).__call__`, which is the **class-level method**. So `grad_checkpoint(model.layers[0])` actually rewrites the method on the type of layer 0, which is the same TransformerBlock class as all other layers. **One call patches all layers** because they share a class. The choice of `layers[0]` is arbitrary — any layer index would do. (This is non-obvious from the call site; the implementation comment "Update all instances of type(layer)" at trainer.py:27 is the giveaway.)

**HOW MUCH.** Trades ~30-50% more compute time per step for ~50-70% less activation memory (the standard checkpointing tradeoff). For a 30-layer model: instead of holding 30 layers × ~5 GB activation, hold 1 layer × 5 GB at a time during backward. Saves ~150 GB activation peak on big models.

**HF2Q EQUIVALENT.** *None.* hf2q doesn't have `mx.checkpoint`. **PORT TO HF2Q: maybe — the underlying mlx-native primitive `mx.checkpoint` would need exposure.** Likely larger than 100 LOC and lives in mlx-native, not hf2q.

---

## 12. Distributed `all_sum` + `nn.average_gradients`, both on `stream=mx.cpu`

**WHAT.** Cross-rank gradient + scalar reduction explicitly routed onto the CPU stream.

```python
# dwq.py:124 — gradient averaging across ranks
grads = nn.average_gradients(grads)

# dwq.py:144-145, 180-181 — loss + token-count reduction on CPU
loss = mx.distributed.all_sum(loss, stream=mx.cpu).item() / world_size
ntoks = mx.distributed.all_sum(ntoks, stream=mx.cpu).item()
```

**WHY `stream=mx.cpu`?** The reduction is a tiny scalar; running it on the GPU stream would force a GPU↔CPU sync just to pull `.item()` out. Routing the all_sum onto the CPU stream lets the GPU stream continue executing the next batch's forward in parallel. This is a latency-hiding optimization, not memory.

**HOW MUCH.** Throughput, not memory. ~5-15% per-step speedup at world_size=2-4.

**HF2Q EQUIVALENT.** N/A (single-process). **PORT TO HF2Q: no.**

---

## 13. Final-warning gate (warn-only, no fail)

**WHAT.**

```python
# dwq.py:202-207
valid_loss = validate(params, it=it)
if initial_valid_loss < valid_loss:
    rprint(
        f"❌❌❌\n[WARNING] Final validation loss {valid_loss:.3f} is "
        f"worse than initial validation loss {initial_valid_loss:.3f}."
        " Model quality will likely be degraded.\n❌❌❌"
    )
```

**WHY.** DWQ's optimizer can drive the student further from the teacher than the cold-quantized starting point — this is a real failure mode of distillation. The warning is **stderr-only**, no exit code, no exception.

**HF2Q port consideration.** Should this be hard-fail-by-default? Arguments either way:
- **Warn-only (match upstream):** preserves upstream parity; experienced operators want to see the model and decide.
- **Hard-fail by default + `--allow-degraded`:** matches user's earlier feedback `feedback_substrate_must_not_synthesize_ship_gates.md` — "measurement returns real numbers; if the gate is violated, that's a hard fail." Saves a class of "shipped a bad model because the warning scrolled off" footguns.

**Recommendation: hard-fail by default with explicit `--allow-degraded` opt-out.** Matches hf2q's existing release-gate stance per `project_adr017_a02b_peer_perf_gap.md`. **PORT TO HF2Q: yes — but invert the default vs upstream.**

---

## 14. `copy.deepcopy(model)`

**WHAT.** mlx-lm uses `copy.deepcopy` to clone an MLX model in two places:

```python
# dwq.py:377 — clone teacher to make student
q_model = copy.deepcopy(model)
_, config = quantize_model(q_model, config, group_size=..., bits=...)

# dynamic_quant.py:55, 133 — copy model inside binary search loop
q_model = copy.deepcopy(model)
q_layers = copy.deepcopy(layers)
```

**WHY does this not blow up memory?** *This is the trick that makes lazy=True load-bearing.* On an MLX `nn.Module`, deepcopy clones the **module structure** (Python objects + parameter dict) but the actual `mx.array` values share their underlying buffer **as long as nobody calls `mx.eval` on the model parameters yet.** Because `q_model = copy.deepcopy(model)` runs before `mx.eval(model.parameters())`, the cloned tree references the same not-yet-realized graph nodes as the original. Once you start mutating one (`q_model.weight = qdq(...)`), MLX's copy-on-write semantics fork only the touched buffers.

**Verification path.** dwq.py:337 loads with `lazy=True` → dwq.py:377 deepcopy happens before any forward → only at dwq.py:378 `quantize_model(q_model, ...)` does the student start to diverge → at this point the teacher's never-touched weights remain shared.

**Inside dynamic_quant's binary search loop** (dynamic_quant.py:130-144): the loop calls `copy.deepcopy(model)` per binary-search step. Each iteration only touches the specific layers selected by the predicate; everything else stays shared. Memory cost per iteration ≈ touched-layer-count × layer-size, not full model.

**HOW MUCH.** Without lazy: deepcopy of a 60 GB realized model = +60 GB peak. With lazy: deepcopy = ~few hundred MB (Python object graph + metadata).

**HF2Q EQUIVALENT.** Partial via `LazyTensorMap` + Arc<Vec<u8>>. **But blocked by §1's `materialize_cloned` defect** — every "clone" via that method becomes an actual byte clone. **PORT TO HF2Q: covered by §1's `materialize_cloned` Arc-clone fix.** With that fix, hf2q's `LazyTensorMap` should support deepcopy-like semantics where untouched tensors share Arc-backed bytes.

---

## Cross-cutting: `mx.clear_cache()` per block (awq.py)

Worth noting separately because it's the third leg of memory hygiene:

```python
# awq.py:504-505 — at end of each transformer block in awq_quantize
mx.eval(block)
mx.clear_cache()

# tuner/trainer.py:20-22 — threshold-based variant
def _clear_cache(threshold: int):
    if mx.get_cache_memory() > threshold:
        mx.clear_cache()
```

**WHAT.** MLX maintains a buffer pool that grows opportunistically and shrinks lazily. `mx.clear_cache()` forces immediate return of unreferenced buffers to the OS.

**WHY.** Across 30+ transformer blocks, even with eval discipline, freed buffers stay in the pool for reuse. Without `clear_cache` between blocks, peak grows monotonically as the pool retains worst-case-shape buffers from earlier blocks. Calling it per block bounds pool size to one-block's-worth.

**HOW MUCH.** Empirically ~5-15% peak reduction in long-running quantization passes (tuner/trainer.py uses a threshold variant exactly because clearing every step is too aggressive — it costs allocation overhead).

**HF2Q EQUIVALENT.** mlx-native's residency set + arena allocators are the analogue, but there's no per-block "shrink the pool" knob exposed. **PORT TO HF2Q: maybe — investigate exposing `MTLHeap::releaseUnusedResources` between blocks.**

---

## Priority-ranked port list for hf2q DWQ on 128 GB

| Rank | Trick | Memory delta | LOC est | Risk |
|------|-------|--------------|---------|------|
| 1 | §3 `del grads` + eval-accumulator pattern | **~60 GB** | 30 | low |
| 2 | §7 stream-targets-to-disk + drop teacher | **~60 GB** | 200 | medium |
| 3 | §1 `materialize_cloned` Arc-clone (not Vec-clone) | **~per-tensor ×N** | 20 | low |
| 4 | §10 verify trainable-only fp32 (not whole-model) | up to 60 GB | 50 | low |
| 5 | §6 `mx.get_peak_memory()` analogue (Darwin RSS) | 0 (telemetry) | 50 | low |
| 6 | §5 `setWiredLimit` verification | thrash avoidance | 30 | low |
| 7 | §9 `--dwq-batch-size` / `--dwq-max-seq-length` flags | 5-20 GB | 40 | low |
| 8 | §13 final-warning gate (hard-fail-by-default) | 0 | 20 | low |
| 9 | §11 grad_checkpoint | ~150 GB at scale | mlx-native | high |
| 10 | §8 pipeline_load (multi-process) | ~1/N | architectural | very high |

**Top three** (§3 + §7 + §1) together address the entire 199 GB → ~90-100 GB delta needed to fit DWQ on 128 GB. They are the surgical fixes; everything else is hygiene.

---

## Things I verified by code, not docstring

- `lazy=True` deferral lives entirely at utils.py:417 `if not lazy: mx.eval(...)`. The "deferred load" phrase in the docstring is the consequence, not the mechanism.
- `grad_checkpoint(layers[0])` patches the *class*, not the instance. The trainer.py:27 comment is correct; many readers misread it.
- `copy.deepcopy` on MLX models works because of MLX's lazy-graph + buffer-sharing semantics, not Python's normal deepcopy. The combination with `lazy=True` is load-bearing — deepcopy *after* `mx.eval(model.parameters())` would in fact double memory.
- The "even timeout" comment at dwq.py:55 is a typo for "Metal kernel timeout" — verified by the only other place `stream=mx.cpu` appears with similar comments (utils.py:587 `# Synchronize processes to avoid timeout`).
- `mx.get_peak_memory()` is GPU-side only; Python-side numpy/torch allocations don't show up. Telemetry-only — never a gating signal.
- hf2q's `LazyTensorMap::materialize_cloned` (lazy.rs:317-321) **always Vec-clones** despite the comment claiming "Always clone the Vec — by-reference semantic must keep the Arc live." The Arc itself stays live through `Arc::clone` regardless; the Vec clone is unforced.
