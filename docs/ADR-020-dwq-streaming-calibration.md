# ADR-020: DWQ streaming calibration — eliminate the 199 GB peak on 27B/35B

**Status**: Proposed — 2026-05-06; PIVOTED iteration 4.5 (taxonomy + algorithm correction)

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

## What this changes
1. The "make working DWQ quants" target stays the same — sensitivity-based mixed-bit allocation in GGUF format. Just rename internally to `dynamic-quant` so future readers understand the taxonomy.
2. **Replace `DwqCalibrator` with `DynamicQuantCalibrator`** that ports `mlx_lm/quant/dynamic_quant.py:estimate_sensitivities` exactly. The algorithm is gradient-based (KL-div between full-precision and qdq'd model) — different metric than hf2q's current variance-magnitude, but matches the published reference and is provably memory-bounded.
3. **Drop A+B+D** as scoped earlier — A (mmap) and B (zero-copy MlxBuffer) are still useful but **not load-bearing for fitting on 128 GB**, because mlx-lm's algorithm doesn't need the all-layer F32 expansion in the first place. Per-batch processing + `del grads` keeps peak bounded by model size + 1-batch activations.

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
