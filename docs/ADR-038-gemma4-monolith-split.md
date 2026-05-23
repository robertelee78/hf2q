# ADR-038: Split `src/serve/forward_mlx.rs` monolith per-arch + Gemma 4 EAGLE-3 enablement

- **Status**: 🚧 IN PROGRESS — Steps 1 + 2 SHIPPED at hf2q `c2406402` (2026-05-22)
- **Date**: 2026-05-22

## Phase status

| Step | Status | Commit | Notes |
|---|---|---|---|
| Step 1 — shared primitives | ✅ SHIPPED | hf2q `05a1d73a` | `src/serve/forward_mlx_shared.rs` 1764 LOC; `forward_mlx.rs` shrunk 10142 → 8409 (~1733 LOC extracted); `pub use` shim preserves `crate::serve::forward_mlx::X` paths — zero external consumer edits; all 17 inventoried items moved; 3 test modules relocated (`cosine_tests`, `dispatch_qmatmul_f32_router_test`, `ac5_iter_b_affine_qweight_roundtrip`); `dense_placeholder_tests` stayed; `#[inline(always)]` on `rms_norm_f32_hs_cached` preserved; cargo check clean, 3110 tests pass (same 3 pre-existing GPU hardware failures, not regressions). All AC-1.1 through AC-1.10 verified green. |
| Step 2 — Gemma KV-cache extraction | ✅ SHIPPED | hf2q `c2406402` | `src/inference/models/gemma4/kv_cache.rs` (466 LOC) with all 6 items: `MlxKvCache` + `trim`/`visible_len` impl, `HbKvBuffers`, `DenseKvBuffers` + `ByteSized` impl, `HybridKvBuffers` + `ByteSized` impl, `alloc_hybrid_kv_for_layer` (promoted `pub(super)` → `pub(crate)`), `DecodeRegime`. Added `pub mod kv_cache` to gemma4/mod.rs. Forward_mlx.rs shrunk to 8118 LOC (~304 LOC cut). `pub use` re-exports preserve `crate::serve::forward_mlx::X` paths — same strangler-fig pattern as Step 1. Fixed pre-existing missing `norm_before_residual` field in `eagle3_orchestrator.rs` default initializer (added to `Eagle3DrafterConfig` ahead of this commit, was blocking cargo check). 12 inline tests in kv_cache.rs cover all paths (MlxDevice-gated, skip on no GPU). cargo check clean; full suite passes (98 kv_cache tests). All AC-2.1 through AC-2.10 verified green. |
| Step 3 — rename to gemma4/ tree | ⏳ TODO | — | The big move — atomic commit |
| Step 4 — Gemma 4 EAGLE-3 enablement | ⏳ TODO | — | 6 G4-CFAs targeting ≥1.72× bench on M5 Max |
- **Author**: claude-flow
- **Supersedes**: nothing (ADR-013's per-arch commitment is honored — gemma4 was the lone holdout)
- **Related**: ADR-008 (mlx-native sole backend), ADR-013 (qwen35 per-arch split + Chesterton's fence), ADR-017 (TQ-packed KV persist), ADR-022 (per-arch tokenizers), ADR-028 (Phase 10 hybrid KV), ADR-031 (parallel encode/decode forward), ADR-037 (EAGLE-3 tree decoding)

> ## Mantra (verbatim from `~/Documents/mantra.txt`)
>
> *DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.*

---

## 1. Why (the problem we're solving)

### 1.1 Surface symptom

`/opt/hf2q/src/serve/forward_mlx.rs` is **10,142 lines** in a single Rust file. Its growth trajectory:

| Commit | Date | Lines |
|---|---|---|
| `1228b501` (creation) | 2026-04-12 | 204 |
| `2afbfe90` | 2026-04-16 | 3,278 |
| `cf38ddda` | 2026-04-29 | 5,013 |
| `ef8f643d` | 2026-05-09 | 7,319 |
| `beb49e5a` | 2026-05-10 | 7,886 |
| `2163b9bc` | 2026-05-14 | 9,203 |
| `7293e983` (HEAD) | 2026-05-22 | **10,142** |

That's **50× growth in 40 days.** Every ADR-028, ADR-029, ADR-030, ADR-031 iteration touched this file. Code review diff blast radius blows out; cargo recompile cascades through `forward_prefill.rs` → `engine.rs` → tests.

### 1.2 Real architectural defect

Surface diagnosis: "multi-arch monolith." **Deep audit (4 parallel research agents) overturns this framing**:

- Module declaration: `use super::config::{Gemma4Config, LayerType}` at line 44. The file is **statically typed against Gemma 4**.
- Branching: **zero `match arch` statements**. All branches are intra-Gemma shape choices (sliding vs full layer-type, TQ vs hybrid vs dense SDPA, decode-regime gates).
- Pub-item census across ~50 top-level items:
  - **SHARED (8-13 items)**: `MlxQWeight`, `MlxAffineExtra`, `MlxAffineMoeStack`, `MoeBaseRole`, `DwqOverlayRole`, DWQ parse helpers, `dispatch_qmatmul`, `RmsNormPerHeadArgs`, `dispatch_rms_norm_unit_perhead{,_dual_perm}`, `rms_norm_f32_hs_cached`, `cosine_pairwise_f32`, `load_gguf_qweight`. Consumed by `qwen35/gpu_ffn.rs:77`, `qwen35/model.rs:410`, `forward_prefill.rs:316`, `parity_quality.rs:57`, `quantize/imatrix/forward.rs`.
  - **GEMMA-SPECIFIC (~30 items)**: `MlxAttentionWeights`, `MlxMlpWeights`, `MlxMoeWeights`, `MlxLayerNorms`, `MlxDecoderLayerWeights`, `MlxKvCache`, `MlxActivationBuffers`, `MlxModelWeights` + all impls, `HbKvBuffers/DenseKvBuffers/HybridKvBuffers`, `DecodeRegime`, `KernelTypeProfile`, `TokenProfile`, `ProfileAccumulator`, `encode_parallel_layers_chunked`.
  - **QWEN-LEAKED**: **0 items.**

**Conclusion:** `forward_mlx.rs` is *a Gemma-4 file with shared primitives leaking inside it* — not a tangled multi-arch monolith. Qwen 3.5 inference path doesn't enter this file at all; qwen35 already has its own clean module at `src/inference/models/qwen35/` (52,325 LOC across 19 files).

### 1.3 Load-bearing prior decision

**ADR-013** §"Chesterton's fence" already committed to per-arch layout (quoting verbatim from research):

> *"The current Gemma-4 inference surface (`src/serve/forward_mlx.rs`...) encodes heterogeneous-attention dispatch, mixed head-dim per layer, V=K tying, frequency-factor-masked RoPE, 7-norm per-layer stack, and MoE expert stacking — all explicitly for Gemma-4's mixed sliding-vs-global + MoE shape... Qwen3.5 gets its own per-variant layer-kind enum in its own module."*

`src/inference/models/qwen35/` **honored** this rule. `src/inference/models/gemma4/mod.rs:5` documents the open TODO:

> *"Future: dedicated Gemma4 forward graph if/when the `src/serve/forward_mlx.rs` monolith is split per-arch."*

**Gemma 4 is the lone holdout.** The architectural direction is already decided; this ADR executes it.

### 1.4 Peer codebases (unanimously per-arch)

| Peer | Pattern | Examples |
|---|---|---|
| **llama.cpp** (`/opt/llama.cpp/src/models/`) | One file per arch | `gemma4.cpp` 457 LOC, `qwen35.cpp` 635, `qwen35moe.cpp` 731, `qwen3moe.cpp` 177 |
| **vLLM** (`/opt/vllm/vllm/model_executor/models/`) | One file per arch + shared `model_executor/layers/` | `gemma4.py` 1721 LOC, `gemma3.py` 514, `qwen3.py`, `qwen2_moe.py` 603, `qwen3_5_mtp.py` 457 |

hf2q's `forward_mlx.rs` is an **outlier of one**.

### 1.5 Downstream consequence — Gemma 4 EAGLE-3 is blocked

ADR-037 shipped EAGLE-3 verifier stack for Qwen 3.6 27B (F1-F5: per-layer, full-layer, Q4_0 dense, multi-layer chain + serve flag, MoE dispatch, orchestrator dense/MoE). The path is **functionally complete on hf2q's side** but blocked on community drafter weights:

| Model | hf2q inference path | EAGLE-3 weights on HF |
|---|---|---|
| Qwen 3.5/3.6 27B (dense) | ✅ in `models/qwen35/` | ❌ none (SpecForge #486 open RFC, no PR) |
| Qwen 3.6 27B-A3B (MoE) | ✅ in `models/qwen35/` | ❌ none |
| Gemma 4 31B (dense) | ✅ in `forward_mlx.rs` monolith | ✅ `RedHatAI/gemma-4-31B-it-speculator.eagle3` (1.72× published bar) |

**The model with community-trained EAGLE-3 weights is precisely the model trapped in the monolith.** Without splitting Gemma 4 into its own per-arch module, porting the EAGLE-3 F1-F5 stack from qwen35 to gemma4 is a tangled mess. Splitting unblocks real SOTA benchmarks on M5 Max against the published 1.72× bar.

### 1.6 Costs of the status quo

- **Review burden**: 10,142-line file blows out PR diffs. Every ADR-028/029/030/031 iteration paid this tax.
- **Recompile blast radius**: any Gemma decode change recompiles `forward_mlx.rs` → `forward_prefill.rs` → `engine.rs` → tests.
- **Test isolation**: 4× `#[cfg(test)]` modules duplicated at lines 8480, 8555, 8658, 8781 — symptom of pressure already felt.
- **Onboarding friction**: `gemma4/mod.rs` documents the TODO as a permanent module-level comment.
- **Implicit Chesterton's fence violation**: every Qwen 3.5 contributor reads "shared" types defined inside the Gemma file and risks assuming Gemma-specific semantics are universal.

---

## 2. What (the proposed solution)

A **4-phase strangler-fig migration**: extract shared primitives → extract Gemma KV types → rename monolith into the per-arch module tree → enable Gemma 4 EAGLE-3 against the published `RedHatAI/gemma-4-31B-it-speculator.eagle3` drafter weights.

**Big-bang split is rejected.** Two risks make atomic monolith surgery dangerous:

1. **`dispatch_qmatmul` reaches 4 unrelated trees** (forward_prefill + qwen35/gpu_ffn + quantize/imatrix + parity_quality). Big-bang would touch all four in one PR.
2. **`encode_parallel_layers_chunked` uses `unsafe` lifetime transmutes** tied to ADR-031's soundness argument. Big-bang puts the unsafe contract at module-relocation risk.

Strangler-fig **sequencing defuses both**: Step 1 extracts the cross-tree shared types with a `pub use` shim (zero behavior change, zero external import-site edits); Step 2 carves out the Gemma KV types into the gemma4 module via the established qwen35 pattern; Step 3 then does the rename with minimal residual risk; Step 4 ports EAGLE-3 onto the clean foundation.

### 2.1 Target architecture

After all 4 steps land:

```
src/serve/
├── forward_mlx_shared.rs       (NEW — ~1500 LOC; cross-arch primitives)
├── mod.rs                       (forward_mlx removed; forward_mlx_shared declared)
├── forward_prefill.rs           (imports updated)
├── forward_prefill_batched.rs   (imports updated)
├── parity_quality.rs            (imports updated)
├── spec_decode_cli.rs           (Gemma 4 EAGLE-3 dispatch added)
└── api/engine.rs                (imports updated)

src/inference/models/gemma4/
├── mod.rs                       (expanded — declares 8 submodules)
├── model.rs                     (~1950 LOC — MlxModelWeights + loaders)
├── kv_cache.rs                  (~700 LOC — Step 2 destination)
├── gpu_full_attn.rs             (~2477 LOC — encode_one_layer + future gemma4_tree_verify_full_layer*)
├── gpu_ffn.rs                   (stub under Path A; populated under Path B follow-up)
├── forward_gpu.rs               (~1850 LOC — forward_decode + ADR-031 parallel worker)
├── io_heads.rs                  (~500 LOC — argmax/logits/NLL)
├── kv_persist.rs                (~410 LOC — ADR-017 TQ snapshot/restore)
├── profile.rs                   (~470 LOC — KernelTypeProfile, TokenProfile, ProfileAccumulator)
└── tokenizer.rs                 (unchanged)

src/inference/spec_decode/eagle3_orchestrator.rs
                                 (generalized: ModelFamily enum dispatches Qwen35/Gemma4)
```

---

## 3. How (executable detail)

### 3.1 Step 1 — Extract shared primitives to `src/serve/forward_mlx_shared.rs`

**Goal**: lowest-risk, zero-behavior-change extraction. Removes the "shared types inside Gemma file" smell. Defuses `dispatch_qmatmul`'s cross-tree blast radius before Step 3 deletes `forward_mlx.rs`.

**Scope**: ~1500 LOC.

#### 3.1.1 Inventory (exact lines in current HEAD `7293e983`)

| Item | Lines | LOC | External callers |
|---|---|---|---|
| `MlxAffineExtra` | 351–360 | 10 | None directly (embedded via `MlxQWeight.affine`) |
| `MlxQWeight` struct | 364–402 | 39 | None direct; consumed via `parity_quality.rs`, `qwen35/model.rs` transitively |
| `impl MlxQWeight { matmul_params, from_mlx_affine_linear }` | 404–520 | 117 | Same as struct |
| `MlxAffineMoeStack` | 570–587 | 18 | `qwen35/gpu_ffn.rs:77, 400, 401, 402, 497, 498, 499`; `qwen35/weight_loader.rs:83, 84, 85`; `qwen35/model.rs:411` |
| `DwqOverlayRole` | 1428–1441 | 14 | None direct; logically shared |
| `parse_dwq_overlay_metadata` | 1447–1472 | 26 | `qwen35/model.rs:411, 420` |
| `parse_dwq_overlay_role` | 1475–1493 | 19 | None direct; logically shared |
| `MoeBaseRole` | 1501–1506 | 6 | `qwen35/model.rs:411, 426`; transitive via `qwen35/gpu_ffn.rs` |
| `parse_dwq_moe_expert_role` | 1511–1527 | 17 | `qwen35/model.rs:411` |
| `cosine_pairwise_f32` | 8458–8478 | 21 | `parity_quality.rs:58` |
| `load_gguf_qweight` | 9359–9389 | 31 | None direct; logically shared (qwen35 re-implements; consolidate) |
| `populate_f16_shadow_if_enabled` | 9406–9455 | 50 | None direct; logically shared |
| `RmsNormPerHeadArgs<'a>` | 9467–9480 | 14 | `forward_prefill.rs:317` |
| `dispatch_rms_norm_unit_perhead` | 9493–9534 | 42 | `forward_prefill.rs:317`; sites at `forward_mlx.rs:3466, 3482` |
| `dispatch_rms_norm_unit_perhead_dual_perm` | 9554–9591 | 38 | `forward_prefill_batched.rs:33` |
| `rms_norm_f32_hs_cached` | 9624–9648 | 25 | None direct (in-file 5078, 5089, 5100); mark `pub(crate)` |
| `dispatch_qmatmul` | 9658–9974 | 317 | `forward_prefill.rs:316, 1831, 1844`; `forward_prefill_batched.rs:32, 3242, 3255`; docstring refs in `quantize/imatrix/forward.rs:21, 54, 182` |

**Explicitly EXCLUDED** (stay in Gemma file):
- `f32_slice_to_le_bytes` (1416–1422) — only used by `tq_v2_snapshot_block` (Gemma TQ v2 spill); not shared.
- `alloc_one_f32_placeholder` (546–557) — only used inside `MlxMoeWeights::dense_placeholder`; tightly coupled, not shared.

#### 3.1.2 Dependency order (topological clusters)

**Cluster 1** — Quantized weight types (atomic):
1. `MlxAffineExtra` → 2. `MlxQWeight` + impl → 3. `MlxAffineMoeStack` → 4. `load_gguf_qweight` → 5. `populate_f16_shadow_if_enabled` → 6. `dispatch_qmatmul`

**Cluster 2** — DWQ overlay parsing (atomic):
7. `DwqOverlayRole` → 8. `parse_dwq_overlay_metadata` → 9. `parse_dwq_overlay_role` → 10. `MoeBaseRole` → 11. `parse_dwq_moe_expert_role`

**Cluster 3** — Norm dispatch helpers (atomic):
12. `RmsNormPerHeadArgs` → 13. `dispatch_rms_norm_unit_perhead` → 14. `dispatch_rms_norm_unit_perhead_dual_perm` → 15. `rms_norm_f32_hs_cached`

**Cluster 4** — `cosine_pairwise_f32` (standalone, atomic)

#### 3.1.3 Re-export shim (preserves `crate::serve::forward_mlx::X` imports for one release cycle)

Drop into `src/serve/forward_mlx.rs` immediately after the existing `use` block (~line 46):

```rust
// Step 1 shared-primitives split — re-export to preserve external import paths.
// Remove after Step 3 retires forward_mlx.rs entirely.
mod forward_mlx_shared;
pub use forward_mlx_shared::{
    cosine_pairwise_f32, dispatch_qmatmul, dispatch_rms_norm_unit_perhead,
    dispatch_rms_norm_unit_perhead_dual_perm, parse_dwq_moe_expert_role,
    parse_dwq_overlay_metadata, parse_dwq_overlay_role, DwqOverlayRole,
    MlxAffineExtra, MlxAffineMoeStack, MlxQWeight, MoeBaseRole, RmsNormPerHeadArgs,
};
pub(crate) use forward_mlx_shared::{
    load_gguf_qweight, populate_f16_shadow_if_enabled, rms_norm_f32_hs_cached,
};
```

#### 3.1.4 Test movement

3 test modules move with the code (~770 LOC of tests):
- `cosine_tests` (8480–8539) → `forward_mlx_shared`
- `dispatch_qmatmul_f32_router_test` (8658–8767) → `forward_mlx_shared`
- `ac5_iter_b_affine_qweight_roundtrip` (8781–9352) → `forward_mlx_shared`

`dense_placeholder_tests` (8555–8645) **stays in `forward_mlx.rs`** (tests `MlxMoeWeights::dense_placeholder` which is Gemma-specific).

#### 3.1.5 Imports needed by `forward_mlx_shared.rs`

```rust
use anyhow::Result;
use mlx_native::{
    DispatchRecord, GgmlQuantizedMatmulParams, GgmlType, GraphSession, KernelRegistry,
    MlxBuffer, MlxDevice,
};
use mlx_native::ops::dense_gemm::DenseGemmF16Params;
use crate::core::mlx_safetensors_loader::MlxAffineLinear;
use crate::quantize::imatrix::{intercept_qmatmul_with_hint, ImatrixHint};
use crate::serve::gpu::QuantWeightInfo;
```

(All other `mlx_native::ops::*` calls inside `dispatch_qmatmul` use fully-qualified paths.)

#### 3.1.6 Step 1 risks (defused)

1. **`#[inline(always)]` on `rms_norm_f32_hs_cached` is load-bearing** per its doc comment ("call-site inlining is the optimization; without it Step 1f bench was neutral"). Move the attribute verbatim; Rust honors `#[inline(always)]` across modules.
2. **`MlxQWeight` field visibility** is fully `pub`; cross-module access from 153 in-file sites continues working.
3. **`dispatch_qmatmul`'s `intercept_qmatmul_with_hint` call** is the only non-`mlx_native` dependency; add `use crate::quantize::imatrix::*` to the new file.
4. **`pub use` shim is delete-at-end-of-Step-3** — no `#[deprecated]` needed (internal monorepo).

---

### 3.2 Step 2 — Extract Gemma KV-buffer + decode-regime to `src/inference/models/gemma4/kv_cache.rs`

**Goal**: bounded extraction mirroring `qwen35/kv_cache.rs` layout. Removes ~700 LOC from `forward_mlx.rs`.

#### 3.2.1 Inventory (exact lines in HEAD `7293e983`)

| Item | Lines | LOC | External callers |
|---|---|---|---|
| `MlxKvCache` + impl (`trim`, `visible_len`) | 744–811 | 67 | `serve/kv_persist/families/tq_packed.rs:320, 445, 578, 583, 1348`; `serve/api/tq_packed_descriptor.rs:106` (comment) |
| `HbKvBuffers` | 1170–1187 | 18 | `forward_prefill.rs:316, 856, 876`; `forward_prefill_batched.rs:32, 453, 473` |
| `DenseKvBuffers` + `impl ByteSized` | 1189–1226 | 37 | `forward_prefill.rs` (14 sites); `forward_prefill_batched.rs` (5 sites); `api/engine.rs` (11 sites incl. `:1150, 1155, 3899, 3914, 3934, 3941, 4210, 4973`); `serve/kv_persist/lcp_registry.rs:36–43` |
| `HybridKvBuffers` + `impl ByteSized` | 1228–1294 | 66 | `forward_prefill.rs:843`; `forward_prefill_batched.rs:439` |
| `alloc_hybrid_kv_for_layer` (+ docs) | 1296–1359 | 64 | `forward_prefill.rs:849`, `forward_prefill_batched.rs:445` (full-path qualified) |
| `DecodeRegime` enum + docs | 1361–1402 | 42 | `serve/parity_quality.rs:58, 505, 521` |

**Net extraction: ~313 LOC** of types + impls (well under 700 LOC budget; remainder is doc-comment volume).

#### 3.2.2 Dependencies — clean

None of the 6 types reference `Gemma4Config` or `LayerType`. New module imports:

```rust
use anyhow::{anyhow, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};
```

Does NOT need `use super::super::super::serve::config::*`. The types are architecturally agnostic but logically Gemma 4-owned.

#### 3.2.3 Visibility upgrade required

`alloc_hybrid_kv_for_layer` is currently `pub(super)` (line 1305) — reachable from `forward_prefill*.rs` because they share parent `crate::serve::`. After cross-tree move to `crate::inference::models::gemma4::kv_cache`, `pub(super)` would scope to `gemma4` only, breaking the two callers. **Promote to `pub(crate)`** in the move.

All other items already have `pub` struct fields → 153 in-file `self.kv_caches[i].k_packed` field-access sites continue working unchanged after import.

#### 3.2.4 `ByteSized` orphan rule

Both `impl ByteSized` blocks travel with their structs (Rust orphan rule: impls live with either trait or type; the trait stays at `serve/kv_persist/lcp_registry.rs:82`). Move-with-types satisfies the local-type arm. Path already fully-qualified at source (`impl crate::serve::kv_persist::lcp_registry::ByteSized for ...`).

#### 3.2.5 `forward_mlx.rs` residual usage

Add near line 43:

```rust
use crate::inference::models::gemma4::kv_cache::{
    MlxKvCache, HbKvBuffers, DenseKvBuffers, HybridKvBuffers,
    DecodeRegime, alloc_hybrid_kv_for_layer,
};
```

#### 3.2.6 External caller updates (3 files)

```rust
// serve/forward_prefill.rs:315 — split the import block
use super::forward_mlx::MlxModelWeights;
use crate::inference::models::gemma4::kv_cache::{DenseKvBuffers, HbKvBuffers};
use crate::serve::forward_mlx_shared::{dispatch_qmatmul, dispatch_rms_norm_unit_perhead, RmsNormPerHeadArgs};
// Lines 843, 849: replace `crate::serve::forward_mlx::HybridKvBuffers` and `::alloc_hybrid_kv_for_layer`
// with `crate::inference::models::gemma4::kv_cache::...`

// serve/parity_quality.rs:57 — pull DecodeRegime out
use crate::serve::forward_mlx_shared::cosine_pairwise_f32;
use crate::serve::forward_mlx::MlxModelWeights;
use crate::inference::models::gemma4::kv_cache::DecodeRegime;

// serve/api/engine.rs:1155, 3899: 3 sites updating `crate::serve::forward_mlx::DenseKvBuffers`
//   → `crate::inference::models::gemma4::kv_cache::DenseKvBuffers`
```

#### 3.2.7 New tests (mirror qwen35/kv_cache.rs inline pattern; ~12 tests)

`mlx_kv_cache_trim_linear_decrements_seq_len`, `mlx_kv_cache_trim_sliding_errors`, `mlx_kv_cache_trim_overflow_errors`, `mlx_kv_cache_visible_len_eq_seq_len`, `decode_regime_default_via_default_trait`, `hybrid_kv_buffers_byte_len_sums_fields`, `dense_kv_buffers_byte_len_sums_k_plus_v`, `alloc_hybrid_kv_for_layer_*` (3 env-gated variants), `alloc_hybrid_kv_for_layer_norms_per_pos_d256_d512`.

Each guarded with `let dev = match MlxDevice::new() { Ok(d) => d, Err(_) => { eprintln!("skip: no MlxDevice"); return; } };` per existing convention.

---

### 3.3 Step 3 — Rename `forward_mlx.rs` → `src/inference/models/gemma4/` tree

**Goal**: complete the per-arch split. Largest step (~5000 LOC relocation post-Steps 1+2) but architecturally lowest-risk because Steps 1+2 defused the worst coupling.

#### 3.3.1 Critical structural insight

`forward_decode` (the 929-line method at lines 6050–6978) **is NOT a monolithic body**. The layer loop has already been refactored to call `self.encode_one_layer(...)` — defined at line 3324 as `pub(crate) fn encode_one_layer<'sess>(&self, layer_idx: usize, ...) -> Result<()>`. **Note: `&self`, not `&mut self`** — required for ADR-031's parallel-encode `Sync`-safety.

`encode_one_layer` itself is 2477 lines spanning attention + FFN + MoE interleaved under one method. **It is NOT internally decomposed into attn-vs-FFN helpers.**

#### 3.3.2 Decision: Path A vs Path B

- **Path A (RECOMMENDED for Step 3 proper)**: keep `encode_one_layer` whole; place it in `gpu_full_attn.rs`. Leave `gpu_ffn.rs` as a doc-only stub. Zero semantic refactor — pure rename. Risk: minor cosmetic divergence from qwen35 layout (qwen35 has `encode_attention_block` + `encode_ffn_block` separately).
- **Path B (Step 3-followup, separate CFA)**: extract `encode_attention_block` + `encode_ffn_block` from `encode_one_layer`. ~1500 LOC of methodical surgery. Defer until after Step 4's EAGLE-3 enablement validates the architecture in practice.

**This ADR commits to Path A.** Path B is logged as a follow-up.

#### 3.3.3 File-by-file content distribution (Path A)

| Target file | Source line range | Content | Post-S1+S2 LOC |
|---|---|---|---|
| `gemma4/model.rs` | 351–520, 522–710, 712–737, 813–894, 896–1156, 1158–1168, 1424–1527, 1529–1571, 2089–3201 | `MlxAttentionWeights`, `MlxMlpWeights`, `MlxMoeWeights`+impl, `MlxLayerNorms`, `MlxDecoderLayerWeights`, `MlxActivationBuffers`, `MlxModelWeights` struct + Send+Sync assert, `load_from_gguf`, `apply_dwq_overlay`, `embed_tokens`, DFlash install/take | ~1950 |
| `gemma4/gpu_full_attn.rs` | 3303–5774 (entire `encode_one_layer`) | Per-layer attention + FFN dispatch (Path A keeps interleaved) | ~2477 |
| `gemma4/gpu_ffn.rs` | (stub) | Doc-only placeholder for Path B follow-up | ~30 |
| `gemma4/forward_gpu.rs` | 5775–6047, 6049–6978, 6999–7012, 7027–7039, 7050–7700 | `encode_parallel_layers_chunked`, `forward_decode`, `forward_decode_verify_serial`, `rollback_kv`, `forward_decode_kernel_profile` | ~1850 |
| `gemma4/io_heads.rs` | 1619–1850, 1852–2078, 7914–7924, 7938–7955 | `per_position_argmax_from_hidden{,_opt}`, `per_position_argmax_from_hidden_batched_impl`, `logits_view`, `token_nll_from_logits` | ~500 |
| `gemma4/kv_persist.rs` | 8005–8114, 8124–8414 | `tq_v2_snapshot_block`, `tq_v2_restore_block` (ADR-017) | ~410 |
| `gemma4/profile.rs` | 52–54, 77–113, 156–335, 7701–7887 | `profiling_enabled`, `KernelTypeProfile`, `TokenProfile`, `ProfileAccumulator`+impl, `print_kernel_profile_report` | ~470 |
| `gemma4/forward_cpu.rs` | — | **SKIP** (no CPU forward exists; ADR-008 mlx-native-only) | 0 |

#### 3.3.4 ADR-031 unsafe lifetime contract preservation

The transmute at lines 5863–5869 in `encode_parallel_layers_chunked`:

```rust
// SAFETY: We forge &'a-bound refs into &'static so the 'static-bounded
// closure required by submit_to_global_worker can capture them.  This is
// sound IFF done_rx.recv() below COMPLETES unconditionally before this
// function returns.
let (self_static, ctx_static, exec_static) = unsafe {
    (
        std::mem::transmute::<&Self, &'static Self>(self),
        std::mem::transmute::<&super::layer_ctx::LayerCtx<'_>, &'static super::layer_ctx::LayerCtx<'static>>(ctx),
        std::mem::transmute::<&mlx_native::GraphExecutor, &'static mlx_native::GraphExecutor>(exec),
    )
};
```

**The unsafe contract is fully local to the function body**: soundness depends on (1) `submit_to_global_worker` invariant, (2) `done_rx.recv()` blocking semantics, (3) the closure consuming forged refs and never letting them escape, (4) `MlxModelWeights: Send + Sync` (compile-time assert lines 1165–1168).

Item (4) — the `assert_send_sync::<MlxModelWeights>()` — **travels with the type into `gemma4/model.rs`**. **No reasoning depends on file or module location.**

One spelling change: `super::layer_ctx::LayerCtx<'_>` becomes `crate::serve::layer_ctx::LayerCtx<'_>` after the move. `serve/mod.rs:13` already declares `pub mod layer_ctx;` so visibility is satisfied. **Update both the transmute type annotation and the function signature in one commit.**

#### 3.3.5 Cross-tree import updates (atomic — must land in one commit)

~25 consumer sites; representative subset:

```rust
// src/serve/api/engine.rs:44
use crate::inference::models::gemma4::{model::MlxModelWeights, profile::ProfileAccumulator};

// src/serve/api/tq_packed_descriptor.rs:98, kv_spill_descriptor.rs:215
&crate::inference::models::gemma4::model::MlxModelWeights

// src/serve/forward_prefill.rs:315
use crate::inference::models::gemma4::model::MlxModelWeights;
use crate::inference::models::gemma4::kv_cache::{DenseKvBuffers, HbKvBuffers};
use crate::serve::forward_mlx_shared::{dispatch_qmatmul, dispatch_rms_norm_unit_perhead, RmsNormPerHeadArgs};

// src/serve/parity_quality.rs:57
use crate::serve::forward_mlx_shared::cosine_pairwise_f32;
use crate::inference::models::gemma4::{kv_cache::DecodeRegime, model::MlxModelWeights};

// src/serve/spec_decode_cli.rs:86, 695
&mut crate::inference::models::gemma4::model::MlxModelWeights

// src/inference/spec_decode/ngram_orchestrator.rs:88
use crate::inference::models::gemma4::model::MlxModelWeights;

// src/inference/spec_decode/dflash/{orchestrator.rs,target.rs} (8 sites)
crate::inference::models::gemma4::{model::MlxModelWeights, profile::TokenProfile}

// src/inference/models/qwen35/{model.rs:410, gpu_ffn.rs:77, weight_loader.rs:83}
use crate::serve::forward_mlx_shared::{MlxAffineMoeStack, MoeBaseRole, parse_dwq_*};
// (NOT crate::inference::models::gemma4::*; qwen35 cannot depend on gemma4)
```

#### 3.3.6 `gemma4/mod.rs` (new)

Replace the 7-line stub:

```rust
//! Per-architecture Gemma 4 inference support.
//!
//! Owned by ADR-008 (mlx-native forward path) and ADR-017 (TQ-packed KV
//! persistence). Variant detection in `serve::api::engine::LoadedModel::load`.
//!
//! # Module layout
//!
//! * `model.rs` — `MlxModelWeights` struct, GGUF loader, DWQ overlay, embed_tokens.
//! * `kv_cache.rs` — KV-buffer structs + `DecodeRegime` (Step 2).
//! * `gpu_full_attn.rs` — per-layer attention dispatch (`encode_one_layer`).
//! * `gpu_ffn.rs` — per-layer FFN dispatch (Path A stub; populated in follow-up).
//! * `forward_gpu.rs` — outer `forward_decode` + ADR-031 parallel worker.
//! * `io_heads.rs` — argmax/logits/NLL surface.
//! * `kv_persist.rs` — ADR-017 TQ snapshot/restore.
//! * `profile.rs` — Kernel/Token profile + accumulator.
//! * `tokenizer.rs` — GGUF-embedded tokenizer (unchanged).

pub mod model;
pub mod kv_cache;
pub mod gpu_full_attn;
pub mod gpu_ffn;
pub mod forward_gpu;
pub mod io_heads;
pub mod kv_persist;
pub mod profile;
pub mod tokenizer;

// Re-exports collapse import-site churn for the most-touched types.
pub use model::MlxModelWeights;
pub use profile::{ProfileAccumulator, TokenProfile, KernelTypeProfile};
pub use kv_cache::{DenseKvBuffers, HbKvBuffers, HybridKvBuffers, MlxKvCache, DecodeRegime};
```

#### 3.3.7 `src/serve/mod.rs` change

Line 12 today: `pub mod forward_mlx;`. After Step 3:

```rust
pub mod forward_mlx_shared;  // shared primitives (Step 1, ADR-038 §3.1)
// pub mod forward_mlx;  REMOVED — gemma4 moved to inference::models::gemma4 (ADR-038 §3.3)
```

---

### 3.4 Step 4 — Gemma 4 EAGLE-3 enablement

**Goal**: port EAGLE-3 tree-verify stack (F1-F5 pattern shipped via ADR-037 for qwen35) to Gemma 4 architecture. Load `RedHatAI/gemma-4-31B-it-speculator.eagle3` from HF. Empirically validate **≥1.72×** on M5 Max against the published bar.

**Total new code (Gemma 4 dense path only — MoE deferred since 31B target is dense): ~1,550 LOC** across 4 files.

#### 3.4.1 Architectural deltas Gemma 4 vs Qwen 3.5 (matter for tree-verify)

| Concern | Qwen 3.5 | Gemma 4 |
|---|---|---|
| `head_dim` | 128 (uniform) | 256 sliding / 512 global per-layer (`LayerType`) |
| Tree-attention kernel | `tree_attention_dk128` (Phase E1) | `tree_attention_dk256` + `tree_attention_dk512` — **both shipped** in `mlx-native/src/shaders/tree_attention.metal:295-313` (also `f16kv_*` variants) |
| `num_kv_heads` | 8 (uniform) | 16 sliding / 2 global (for 31B) |
| RoPE | IMROPE, partial rotary 50%, uniform `freq_base` | Standard RoPE, per-layer `rope_theta` (10000 sliding, 1M global), full-head rotation with `freq_factors` mask (ADR-005 1bNEW.18) on global layers |
| RMSNorm sites/layer | 4 (`attn_norm`, `q_norm`, `k_norm`, `post_attn_norm`) | 7+ (`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`, parallel-MoE norms, `q_norm`, `k_norm`) + `layer_scalar` per-layer scalar mul |
| Output gate | `attn_output_gate=true` sigmoid gate | **No output gate** |
| MoE topology | Every layer, shared expert + sigmoid gate | Per-layer `enable_moe_block` flag, PARALLEL block summed with dense MLP, NO shared expert, softmax routing |
| `attn_logit_softcapping` | None | Configurable (verify per-model; 31B has `final_logit_softcapping=30.0` only — confirm if attn softcap present) |
| KV layout | F32 dense (tree-verify path) | TQ-packed / hybrid_kv / dense_kv per layer — tree-verify path needs PARALLEL F32 cache (mirrors qwen35) |

#### 3.4.2 RedHatAI checkpoint schema (confirmed via Speculators source + config.json fetch)

```
hidden_size: 5376
num_attention_heads: 32 (q_proj_out = 8192)
num_key_value_heads: 16 (kv_proj_out = 4096)
head_dim: 256
intermediate_size: 21504
num_hidden_layers: 1
vocab_size: 262144
draft_vocab_size: 32000
rope_theta: 10000.0
tie_word_embeddings: false
attention_bias: false
norm_before_residual: true        # NEW knob — see §3.4.3
norm_before_fc: false
target_hidden_size: null → 5376    # equals drafter hidden_size
eagle_aux_hidden_state_layer_ids: [2, 30, 57]
model_type: llama                  # → no q_norm/k_norm in drafter
```

Saved tensor names (derived from `vllm-project/speculators/.../eagle3/{core,model_definitions}.py`):
- `embed_tokens.weight`, `fc.weight` `[5376, 16128]` (3×hidden_size cat)
- `layers.0.{input_layernorm, hidden_norm, post_attention_layernorm}.weight`
- `layers.0.self_attn.{q,k,v,o}_proj.weight`
- `layers.0.mlp.{gate,up,down}_proj.weight`
- `norm.weight`, `lm_head.weight` `[32000, 5376]` (draft vocab)
- `d2t`, `t2d` (vocab remap; our loader already handles)
- `verifier_lm_head.weight`, `verifier_norm.weight` (in `_keys_to_ignore_on_save`; skip if present)

#### 3.4.3 `Eagle3Weights` loader changes (already family-agnostic; minimal patches)

1. **NEW knob** `norm_before_residual: bool` (default `false`) in `Eagle3DrafterConfig`. The loader doesn't interpret it; the drafter forward (`GpuDrafter`) reads it. **Per `vllm-project/speculators/.../eagle3/model_definitions.py:75-79`**: when `norm_before_residual=true` the residual stream is the NORMED hidden (post `hidden_norm`), not the raw cat-input.
2. **Skip verifier tensors** in `resolve_name` closure (`weights.rs:333-341`) — return `None` for `verifier_lm_head.weight` / `verifier_norm.weight`.
3. No `midlayer.` rename needed (Speculators saves as `layers.0.` already).

#### 3.4.4 New Gemma 4 EAGLE-3 functions (mirror Qwen35 F1-F5 pattern)

| New function | Mirrors | LOC | Key delta |
|---|---|---|---|
| `dispatch_gemma4_tree_verify_attention` | `dispatch_qwen35_tree_verify_attention` (gpu_full_attn.rs:1661) | ~140 | Drop `head_dim==128` gate; accept dk256+dk512; pass per-layer `rope_theta` + optional `freq_factors_buf` |
| `gemma4_tree_verify_attention_block` | `qwen35_tree_verify_attention_block` (gpu_full_attn.rs:1976) | ~450 | input_layernorm → Q/K/V → q_norm/k_norm → RoPE-gemma4 → permute → CPU KV append → tree-attention (dk256/512) → O proj → post_attention_layernorm → residual. **No sigmoid gate.** Accept `layer_type: LayerType`, branch head_dim/num_kv_heads/rope_theta |
| `gemma4_tree_verify_full_layer_q` | `qwen35_tree_verify_full_layer_q` (gpu_full_attn.rs:2883) | ~340 | Compose attn-block + pre_feedforward_layernorm + dense Q4_0 SwiGLU + post_feedforward_layernorm + residual + `*= layer_scalar`. **THE primary CFA** (RedHatAI 31B drafter target is dense). |
| `gemma4_tree_verify_full_layer_q_moe` | `qwen35_tree_verify_full_layer_q_moe` (gpu_full_attn.rs:3204) | ~520 | **DEFERRED** — RedHatAI 31B is dense; 26B-A4B MoE drafter not yet published |
| `Gemma4Model::forward_tree_verify_gpu` | `Qwen35Model::forward_tree_verify_gpu` (forward_gpu.rs:5814) | ~280 | Embed → loop with `LayerType` branch (Sliding/Full) and `attn_shape_base` built INSIDE the loop (head_dim varies per layer) |
| `FullAttnWeightsGpuGemma4` struct | `FullAttnWeightsGpu` (gpu_full_attn.rs:308) | ~50 | 7+ norm slots + no `w_gate` + `layer_scalar` |
| `Gemma4TreeVerifyFullLayerShapeQ` | `Qwen35TreeVerifyFullLayerShapeQ` | ~80 | Per-layer `head_dim` + `num_kv_heads` + `layer_type` + `freq_factors_present` + optional `attn_logit_softcap` |
| `ModelFamily` enum dispatch | `FfnTopology` (eagle3_orchestrator.rs:28) | ~80 | `{Qwen35Dense, Qwen35Moe, Gemma4Dense, Gemma4Moe}`. Sibling `Gemma4Eagle3Orchestrator` initially; trait-extract later |
| `try_dispatch_gemma4_eagle3_spec_decode` | `try_dispatch_qwen35_eagle3_spec_decode` (spec_decode_cli.rs:553) | ~85 | Reuse `HF2Q_SPEC_EAGLE3=1` + `HF2Q_EAGLE3_DRAFTER_PATH` |
| `default_gemma4_eagle3_drafter_config` | `default_qwen35_eagle3_drafter_config` (eagle3_orchestrator.rs:333) | ~40 | `use_qk_norm: false`, `fc_norm: false`, `norm_before_residual: true`, `head_dim: 256`, `rope_theta: 10000`, capture `[2, 30, 57]` |

#### 3.4.5 Per-CFA breakdown (Step 4)

| CFA | Scope | LOC | Risk |
|---|---|---|---|
| **G4-CFA-1** | `gemma4_tree_verify_attention_block` + `dispatch_gemma4_tree_verify_attention` + shape struct + Q4_0 attn weight container | ~700 | Medium — RoPE freq_factors mask correctness on global layers |
| **G4-CFA-2** | `gemma4_tree_verify_full_layer_q` (dense Q4_0) + cross-variant parity vs F32 baseline | ~450 | Medium — 7-norm chain ordering |
| **G4-CFA-3** | `Gemma4Model::forward_tree_verify_gpu` + parallel F32 KV cache per-layer + prefill `LayerActivations` capture extension | ~350 | Low-medium — boilerplate composition |
| **G4-CFA-4** | Extend `Eagle3DrafterConfig` with `norm_before_residual` knob + thread through `GpuDrafter` + `ModelFamily` dispatch | ~300 | Medium — `norm_before_residual=true` changes residual stream semantics |
| **G4-CFA-5** | RedHatAI checkpoint load + smoke test | ~150 | Medium — first read of 4.47 GB safetensors |
| **G4-CFA-6** | Empirical benchmark vs 1.72× bar on M5 Max + sweep `HF2Q_EAGLE3_TREE_BUDGET` + `HF2Q_EAGLE3_TOP_K` | ~250 | Low instrumentation; medium hitting the bar |

#### 3.4.6 Step 4 risks

1. **Gemma 4 RoPE `freq_factors` mask on global layers (dk512).** The mask drives indices [64..256) to ~0 rotation (per ADR-005 1bNEW.18). Tree-attention kernel does NOT apply RoPE — expects pre-roped Q/K. *Mitigation*: byte-level fixture test comparing pre-roped Q/K against `forward_mlx.rs` decode-path RoPE on identical hidden state.
2. **`norm_before_residual=true` semantic** in EAGLE-3 first-layer drafter. *Mitigation*: explicit `GpuDrafter` test running both branches on a stub; pipe `cfg.norm_before_residual` into `GpuDrafter::new`.
3. **Per-layer head_dim/num_kv_heads variance breaks `attn_shape_base` precomputation** (Qwen35 builds ONCE outside the loop). *Mitigation*: build `attn_shape_base` INSIDE the loop using `cfg.head_dim_for_layer(i)` (already exists at `src/serve/config.rs:333-340`).
4. **F32 parallel KV cache memory at 32K context**: 50 sliding × 1.07 GB + 10 global × 268 MB ≈ **56 GB** for 31B. Fits 128 GB M5 Max but coexists with TQ cache. *Mitigation*: gate F16 K cache via `tree_attention_f16kv_dk256/512` kernel variants (already shipped — ADR-028 Phase 10b/10c validated).
5. **`Eagle3Orchestrator::new` calls Qwen35-specific methods** (`ensure_gpu_cache_primed`, `with_gpu_cache_mut`). *Mitigation*: build parallel `Gemma4Eagle3Orchestrator` first (~200 LOC duplication); refactor to a `TreeVerifyTarget` trait in a post-SOTA-benchmark cleanup pass.

---

## 4. Acceptance Criteria (how do we know we're done)

Each step has its own definition of done. The ADR is closed when all 4 steps' criteria pass.

### 4.1 Step 1 acceptance

- [ ] **AC-1.1** `src/serve/forward_mlx_shared.rs` exists; contains all 17 items inventoried in §3.1.1 (Cluster 1-4).
- [ ] **AC-1.2** `src/serve/forward_mlx.rs` shrinks by ~1500 LOC (the items moved) + gains the ~14-line `pub use` shim from §3.1.3.
- [ ] **AC-1.3** `cargo check --workspace` passes with zero errors.
- [ ] **AC-1.4** `cargo clippy --release -p hf2q --no-deps -- -D warnings` passes with zero new warnings on the new file.
- [ ] **AC-1.5** `cargo test --release -p hf2q --lib forward_mlx_shared` runs the 3 relocated test modules (`cosine_tests`, `dispatch_qmatmul_f32_router_test`, `ac5_iter_b_affine_qweight_roundtrip`); all PASS.
- [ ] **AC-1.6** `cargo test --release -p hf2q --lib forward_mlx` runs the residual `dense_placeholder_tests`; all PASS.
- [ ] **AC-1.7** `cargo test --release -p hf2q` (full suite) — all pre-existing tests PASS (zero regressions).
- [ ] **AC-1.8** Live coherence smoke: byte-identical logits on a pinned Gemma 4 fixture before/after Step 1 (greedy temp=0, fixed prompt). Captures behavior preservation across the move.
- [ ] **AC-1.9** Zero external consumers (qwen35/, forward_prefill, parity_quality, imatrix, engine.rs) require import edits — all resolve via the `pub use` shim.
- [ ] **AC-1.10** `#[inline(always)]` on `rms_norm_f32_hs_cached` preserved verbatim. Decode tok/s smoke shows no regression (±2% noise band).

### 4.2 Step 2 acceptance

- [ ] **AC-2.1** `src/inference/models/gemma4/kv_cache.rs` exists; contains all 6 items from §3.2.1.
- [ ] **AC-2.2** `src/inference/models/gemma4/mod.rs:6` adds `pub mod kv_cache;` declaration.
- [ ] **AC-2.3** `forward_mlx.rs` shrinks by ~313 LOC (KV types) + ~700 LOC (impls + docs).
- [ ] **AC-2.4** `alloc_hybrid_kv_for_layer` promoted from `pub(super)` to `pub(crate)`.
- [ ] **AC-2.5** External callers updated: `forward_prefill.rs:315,843,849`, `forward_prefill_batched.rs:32,439,445`, `parity_quality.rs:57-58`, `api/engine.rs:1155,3899` (3 sites). All other transitive references resolve via `gemma4/mod.rs` re-exports.
- [ ] **AC-2.6** `cargo check --workspace` passes.
- [ ] **AC-2.7** New inline tests added (~12) per §3.2.7; all PASS (env-gated tests skip silently without MlxDevice).
- [ ] **AC-2.8** `ByteSized` orphan rule satisfied: `impl ByteSized for DenseKvBuffers` + `impl ByteSized for HybridKvBuffers` move with their types; `lcp_registry.rs` unchanged.
- [ ] **AC-2.9** `cargo test --release -p hf2q` full suite PASS.
- [ ] **AC-2.10** Live coherence smoke: byte-identical logits before/after Step 2.

### 4.3 Step 3 acceptance

- [ ] **AC-3.1** `src/serve/forward_mlx.rs` **deleted**.
- [ ] **AC-3.2** `src/serve/mod.rs:12` no longer declares `pub mod forward_mlx;` (replaced by `pub mod forward_mlx_shared;` already from Step 1).
- [ ] **AC-3.3** `src/inference/models/gemma4/{model.rs, gpu_full_attn.rs, gpu_ffn.rs, forward_gpu.rs, io_heads.rs, kv_persist.rs, profile.rs}` all exist with content per §3.3.3.
- [ ] **AC-3.4** `gemma4/mod.rs` expanded per §3.3.6 (9 submodules + 3 re-export groups).
- [ ] **AC-3.5** All ~25 cross-tree consumer sites updated atomically in one commit per §3.3.5; `cargo check --workspace` passes.
- [ ] **AC-3.6** ADR-031 unsafe transmute type annotation updated from `super::layer_ctx::LayerCtx<'_>` to `crate::serve::layer_ctx::LayerCtx<'_>`; lines preserved verbatim otherwise.
- [ ] **AC-3.7** `assert_send_sync::<MlxModelWeights>()` (lines 1165–1168) travels with `MlxModelWeights` into `gemma4/model.rs`.
- [ ] **AC-3.8** `cargo build --release` succeeds.
- [ ] **AC-3.9** `cargo test --release -p hf2q --lib gemma4` — all relocated tests PASS.
- [ ] **AC-3.10** `cargo test --release -p hf2q --lib serve::parity_quality` — TokenProfile/DecodeRegime import paths resolve.
- [ ] **AC-3.11** `cargo test --release -p hf2q` (full suite) — zero regressions.
- [ ] **AC-3.12** Live release smoke: byte-identical greedy output before/after Step 3 (the entire step is a pure rename).
- [ ] **AC-3.13** `HF2Q_PARALLEL_ENCODE=1 cargo run --release -- generate ...` succeeds (ADR-031 parallel-worker path still exercises the relocated transmute).
- [ ] **AC-3.14** `cargo doc --workspace` — intra-doc links resolve after Section 3.3.5's path rewrites.

### 4.4 Step 4 acceptance

- [ ] **AC-4.1** All 6 G4-CFAs from §3.4.5 ship with passing tests (mirror F1-F5 acceptance pattern from ADR-037 CFAs #3-#7).
- [ ] **AC-4.2** `Eagle3DrafterConfig` gains `norm_before_residual: bool` field; `GpuDrafter::forward` honors it (separate explicit test).
- [ ] **AC-4.3** `Eagle3Weights::load` skip-pattern for `verifier_lm_head.weight` / `verifier_norm.weight` validated against a synthetic safetensors blob containing both.
- [ ] **AC-4.4** Load `RedHatAI/gemma-4-31B-it-speculator.eagle3` from local mirror via `HF2Q_EAGLE3_DRAFTER_PATH`; **zero missing tensors**; loader completes < 30 s on M5 Max.
- [ ] **AC-4.5** `HF2Q_SPEC_EAGLE3=1 HF2Q_EAGLE3_DRAFTER_PATH=... hf2q generate ...` runs end-to-end on `gemma-4-31B-it.gguf`; emits ≥50 tokens; **accept rate > 0.30** (sanity floor; 1.72× bar requires higher).
- [ ] **AC-4.6** F32-cast variant parity test: synthetic 1-layer sliding (dk256) and 1-layer global (dk512) against CPU reference; |GPU − CPU|∞ < 0.20.
- [ ] **AC-4.7** Cross-variant parity test: Q4_0 GPU output vs F32-cast GPU output on identical F32-source weights → Q4_0-quantized; |Δ|∞ < 0.20 (matches F2's AC-7 pattern).
- [ ] **AC-4.8** 3-rep byte-identity determinism test on tree-verify output via `to_bits()` (matches F1-F5 AC-6).
- [ ] **AC-4.9** Multi-iteration cache continuity (5+ iterations) — accept-walk argmax stable.
- [ ] **AC-4.10** **Empirical SOTA bench on M5 Max 128GB**: ≥**1.72× tokens/s** speedup vs `gemma-4-31B-it` greedy baseline (greedy temp=0, conversational prompt set per RedHatAI card). Sweep `HF2Q_EAGLE3_TREE_BUDGET ∈ {6, 10, 16}` and `HF2Q_EAGLE3_TOP_K ∈ {2, 3, 5}`. Report p50/p99 per-iteration latency + accept-rate distribution.
- [ ] **AC-4.11** Zero regression: `qwen35_tree_verify_full_layer` (F1+F2+F4) 21/21 + `eagle3_orchestrator` 10/10 all PASS.
- [ ] **AC-4.12** ADR-037 Phase E6 row updated; ADR-038 Step 4 row added to ADR-037 phase table; commit message references `ADR-037 §F (Gemma 4 dense EAGLE-3 — out of the Qwen scope, enabled by ADR-038)`.

---

## 5. Open follow-ups (out of this ADR's scope; logged for tracking)

- **Path B**: extract `encode_attention_block` + `encode_ffn_block` from `encode_one_layer` into `gemma4/gpu_full_attn.rs` + `gemma4/gpu_ffn.rs` separately (matches qwen35 exactly). ~1500 LOC. Defer until after Step 4 ships.
- **Trait extraction**: replace the parallel `Gemma4Eagle3Orchestrator` with `pub trait TreeVerifyTarget { ... }` implemented by both Gemma4Model and Qwen35Model. Eliminates ~200 LOC duplication.
- **G4-CFA-7**: `gemma4_tree_verify_full_layer_q_moe` MoE variant (~520 LOC). Required when a 26B-A4B EAGLE-3 drafter is published.
- **HASS / Hydra Phase E9** (ADR-037 §10): drafter family extension. Once Gemma 4 EAGLE-3 lands and SOTA bench validates, HASS becomes the next ceiling-raising target.
- **codec ownership refactor (ADR-035)**: independent of this work; remains queued.

---

## 6. Consequences

### 6.1 Positive

- **Architectural debt cleared**: gemma4 honors ADR-013's per-arch commitment; codebase aligns with llama.cpp + vLLM peer conventions.
- **Review burden reduced**: per-arch files keep PR diffs scoped to one module.
- **Recompile blast radius cut**: Gemma decode changes no longer cascade through `forward_prefill.rs` and the entire `serve/` tree on every iteration.
- **Test isolation**: each gemma4 file gets inline `#[cfg(test)]` mods (qwen35 pattern); no more 4× duplicated `#[cfg(test)]` blocks in one 10kloc file.
- **EAGLE-3 unblocked for the only model family with community-trained drafter weights** (Gemma 4 31B via RedHatAI).
- **SOTA bench on owned hardware**: M5 Max can finally hit the published 1.72× EAGLE-3 bar in-house against `gemma-4-31B-it`.
- **Future ADR-037 phases (HASS, Hydra) inherit a cleaner foundation**.

### 6.2 Negative

- **Step 3 is atomic** — ~25 import sites + file delete + module move land in one commit. Bisect remains useful because the commit is a pure rename, but CI must pass atomically.
- **Path A keeps `encode_one_layer` interleaved attn+FFN** (Step 3 doesn't match qwen35's `gpu_full_attn.rs` / `gpu_ffn.rs` semantic split exactly). Path B follow-up addresses this; until then, the gemma4 layout is documented as a cosmetic divergence.
- **Multi-day effort**: Steps 1+2 are 2-3 days each; Step 3 is 3-5 days (the rename); Step 4 is 6 CFAs each in the F1-F5 ship-pattern. Multi-week in total.
- **Parallel F32 KV cache memory cost** at long context (~56 GB for 31B at 32K) until F16 K kernel-variant adoption.
- **`norm_before_residual` knob** introduces a non-trivial code path in `GpuDrafter` — must be tested explicitly to avoid silent accept-rate degradation.

### 6.3 Neutral

- **Step 4 is gated on community drafter publication** for HASS/Qwen variants; out of our control.
- **Mantra adherence**: every step explicitly follows "measure 3× cut once" — the deep-research-before-edit discipline already burned 4 parallel research agents to validate the sequencing.

---

## 7. Decision

**Approved scope**: 4-phase strangler-fig migration per §3.

**First action (Step 1)**: extract shared primitives to `src/serve/forward_mlx_shared.rs` per §3.1. Lowest-risk; zero behavior change; defuses cross-tree blast radius.

**Final completion**: AC-4.10 (empirical SOTA ≥1.72× on M5 Max via `RedHatAI/gemma-4-31B-it-speculator.eagle3` against `gemma-4-31B-it` greedy baseline).

**Mantra check** (operator's "no laziness" reminder): surface-level grep would have said "incremental — don't bother splitting." Deep audit shows: ADR-013 committed; `gemma4/mod.rs` documents TODO; llama.cpp + vLLM both unanimous per-arch-file; monolith is the LAST blocker for parallel Gemma 4 EAGLE-3 work. **The lazy path was the wrong call.** This ADR commits to the right one.

## Links

- ADR-008 — full Candle divorce → mlx-native sole backend
- ADR-013 — qwen35 inference + Chesterton's fence rule for per-arch layout
- ADR-017 — TQ-packed KV persist (the `kv_persist.rs` target)
- ADR-028 — Phase 10 hybrid KV (F16 K variant referenced in §3.4.6)
- ADR-031 — parallel encode/decode forward (the unsafe lifetime contract preserved in §3.3.4)
- ADR-037 — EAGLE-3 tree decoding (the F1-F5 pattern mirrored in §3.4)
- vLLM `gemma4.py` — peer reference for Gemma 4 architecture
- llama.cpp `gemma4.cpp` / `qwen35.cpp` — peer reference for per-arch layout
- Speculators `eagle3/{core,model_definitions}.py` — checkpoint schema source-of-truth
- HF: `RedHatAI/gemma-4-31B-it-speculator.eagle3` — target drafter weights
- HF: `thoughtworks/Gemma-4-31B-Eagle3` — alternative base-model variant
