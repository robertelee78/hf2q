# ADR-020: DWQ streaming calibration — eliminate the 199 GB peak on 27B/35B

**Status**: Proposed — 2026-05-06
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

| Iter | Scope | Status |
|---|---|---|
| 2 | Research synthesis, ADR, memory updates, commit `72fdee8` | DONE |
| 3 | Phase 1: TensorRef.data Arc refactor, falsifier test, regression check (104/104 PASS), commit `437217d` pushed | DONE |
| 4 (this) | Phase 2 deep-dive: trace GPU upload + cache lifecycle, verify MlxBuffer COPIES (not aliases) | DONE |
| 5 | Phase 2 implementation: drop host F32 vecs after one-time GPU upload completes | NEXT |
| 6 | Phase 2 testing: synthetic model peak measurement + regression suite | |
| 7 | E2E DWQ-46 27B with watchdog (no `ulimit -v` on macOS — RLIMIT_AS unimplemented) | |
| 8 | Roll forward to all 4 family×bit combinations | |
| 9 | Benchmark + ADR closure | |

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
