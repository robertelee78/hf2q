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
| 2 (this) | Research synthesis, ADR, memory updates, commit | DONE |
| 3 | Phase 1: TensorRef.data Arc refactor, falsifier test, regression check, commit + push | NEXT |
| 4 | Phase 1 polish: any remaining 267-site fallout, full test suite | |
| 5 | Phase 2 design doc: streaming calibrator API contract before code | |
| 6-8 | Phase 2 implementation: DwqCalibrator + Qwen35Model::load_layer streaming | |
| 9 | E2E DWQ-46 27B with ulimit safety net, validate criterion 1-5 | |
| 10 | Roll forward to all 4 family×bit combinations | |
| 11 | Benchmark + ADR closure | |

## References

- Code: `src/main.rs:482-496`, `src/ir/mod.rs:25-50`, `src/ir/lazy.rs:130-225`, `src/inference/models/qwen35/weight_loader.rs:700-750`, `src/inference/models/qwen35/full_attn.rs::FullAttnLayerWeights`
- Reference impls: `/opt/llama.cpp/src/llama-quant.cpp:1109-1245`, `/opt/llama.cpp/tools/imatrix/imatrix.cpp:229`, `/opt/omlx/omlx/oq.py`
- Memory: `project_hf2q_dwq_oom_root_cause_2026_05_06.md`, `project_qwen35_reconvert_paused_2026_05_05.md`, `project_qwen35_dwq_pre_505b5b8_broken_2026_05_05.md`
- Mantra: ~/Documents/mantra.txt — "Measure 3x, cut once. No fallback. No stub. Pure excellence."
