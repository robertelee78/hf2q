# ADR-037 — EAGLE-3 with dynamic tree port (max coherence + perf)

**Updated:** 2026-10-23 (ADR-063 F3 source audit).

## Current interpretation and research correction

The phase table below records historical implementation and test receipts; it
is not current model qualification. The shipped multi-layer integration and
`HF2Q_SPEC_EAGLE3` wiring are not remaining greenfield tasks. E2 checkpoint
qualification, E7 matched end-to-end measurement, and closure evidence remain
separate obligations. ADR-063 controls current release scheduling.

[EAGLE-2 (Li et al., June 2024)](https://arxiv.org/abs/2406.16858)
introduces confidence-guided dynamic draft trees. hf2q's best-first tree builder
is an implementation to inspect and test, not proof of published performance.
[EAGLE-3 (Li et al., March 2025)](https://arxiv.org/abs/2503.01840)
uses direct token prediction and multi-layer target features. It postdates HASS;
HASS therefore cannot be an extension of EAGLE-3. §10 corrects the earlier
HASS/Hydra conflation. Neither is a qualified second drafter family in hf2q.


- **Status**: 🚧 **IN PROGRESS** — Phases E1 + E3 + E4 + E5 + E6 (drafter side + Qwen35 verifier dispatcher + per-layer + full-layer + Q4_0 production + **multi-layer chain + HF2Q_SPEC_EAGLE3 serve flag**) SHIPPED at hf2q `6f7a9b80`. EAGLE-3 inference path is now production-usable end-to-end behind opt-in env flag pending drafter training (E2). Original CFA #3 closure note retained → **309/309 ADR-037 tests PASS single-threaded** (303 baseline + 6 new full-layer ACs). Three /cfa dual-mode sessions shipped Phase E6 closure: session #1 codex won 361/343 on adversarial bug catch (Claude's modulo-by-zero panic class); session #2 claude won 92/74 on test rigor (codex's no-op T5 alias + scale=0.0 zero-weights identity-path tests); **session #3 claude won 92/63 (29-pt margin) on test rigor and pool-discipline** (codex reproduced 2 prior-CFA defect classes: params buffers via `device.alloc_buffer` breaking AC-6 byte-identity determinism + AC-5 1e-3 hybrid-oracle tolerance unreachable with BF16-cast — codex measured 3.979e-2 drift confirming spec underestimated BF16 slop; claude declared both spec deviations openly, codex's own cross-review JUSTIFIED both). Remaining: E2 drafter training/checkpoint qualification (multi-week H100 compute), E7 empirical validation, E8 final closure.

> **HASS / Hydra is IN-SCOPE as Phase E9 — not deferred.** Per operator 2026-05-22: *"HASS/Hydra ← need this as part of our goal though, after eagle-3, we should never put off hard work, I've said this many times"*. Sequencing: EAGLE-3 closes first (E1-E8), then HASS/Hydra ships as Phase E9 on top of the same Qwen35 tree-verify infrastructure. The historical goal was to evaluate a second drafter family; this does not establish that HASS is implemented, compatible, or empirically validated in hf2q. See Phase E9 row + new §10 "HASS/Hydra plan" for design.
- **Date**: 2026-05-22 (HEAD v2 `9690fc69`; mlx-native `3ea809f` Phase E1 closure)
- **Supersedes**: nothing
- **Author note**: v1 (earlier in this conversation) recommended Medusa-first because it was "simpler to implement". Operator pushed back: *"why do you focus on simple instead of correct for max coherence and perf? ... it literally takes longer to do the wrong things many times, then eventually the right thing ... better to just do the right thing 1st"*. v2 commits to the empirically-best published architecture from the start.

## Phase status

| Phase | Status | Commit | Notes |
|---|---|---|---|
| E1.1 — tree=1 parity | ✅ SHIPPED | mlx-native `310f5cb` | 5/5 byte-identity tests PASS (dk256 basic/GQA/long/unaligned + dk512 basic). 6/6 lib unit tests PASS. Foundation derisked. |
| E1.2 — chain parity (qL>1) | ✅ SHIPPED | mlx-native `b2844bc` | 5/5 byte-identity tests PASS (dk256 qL=2/4 + GQA qL=8 + dk512 qL=4 + long-context kv=512 qL=4). qL>1 contract extends E1.1 structural argument: per-row causal-mimicking mask + identical FMA order + reused reduce kernel → bit-equality. |
| E1.3 — fixed-square tree vs CPU ref | ✅ SHIPPED | mlx-native `f2da58d` | 4/4 PASS within 1e-2 tolerance (dk256 / dk256 GQA / dk512 / dk256 long-prefix). First non-causal-within-tree mask. Surfaced output-layout bug: actual layout is `[q_seq_len, num_heads, head_dim]` (rid = iq2 + iq1 * n_heads), not [heads, queries, dim] like Q input. |
| E1.4 — dynamic asymmetric tree vs CPU ref | ✅ SHIPPED | mlx-native `9d3ffd1` | 4/4 PASS within 1e-2 (dk256 / dk256 GQA / dk512 / chain-as-degenerate-tree). 8-node max-depth-4 asymmetric tree with varying per-depth branching factor. Validates arbitrary EAGLE-2 dynamic-expansion-shape topology. |
| E1.5 — prefix+tree combined parity | ✅ SHIPPED | mlx-native `9d3ffd1` | 3/3 PASS within 1e-2 (dk256 / dk256 GQA / dk512). 504-token natural prefix + 8-node asymmetric tree on top = kv 512. Closest synthetic to production EAGLE-3 long-context dispatch shape. |
| E1 codex /cfa gate | ✅ PASSED | mlx-native `3ea809f` | Codex re-review confirms **0 Critical + 0 Major** remaining. Fixed: K/V dtype validation, buffer byte-length validation, output-layout doc, CPU-ref precision caveat, unused struct, register() reduce-dep doc. Added 3 negative-path validation tests (all PASS). |
| **Phase E1 totals** | ✅ **CLOSED** | mlx-native `3ea809f` | **24/24** integration tests PASS (21 parity + 3 negative-path) + **6/6** lib unit tests PASS = **30/30** total. |
| E2 — EAGLE-3 drafter training | ⏳ TODO | — | Multi-week training (~1wk H100 compute). |
| E3a — multi-layer hidden plumbing | ✅ SHIPPED | hf2q `db495137` | `Eagle3HiddenCollector` with `[seq_len, num_aux, hidden_size]` row-major layout (transpose of DFlash; chosen so `concatenated_hidden()` returns buffer directly with no permute). 15/15 unit tests PASS: constructor validation, slab writes, layout-matches-vLLM, lifecycle, realistic Qwen35-shaped allocation. |
| E3b — eagle3 drafter weights schema + loader | ✅ SHIPPED | hf2q `3b78d2bc` | `Eagle3DrafterConfig` (7 architectural gates) + `Eagle3Weights` strict manifest loader with synthetic safetensors validation. 26/26 unit tests PASS: config validation, manifest structure (12 invariant tests), conditional tensor gates (5 gates × validation), 5 safetensors load-path tests using synthetic in-memory blobs. |
| E3 codex /cfa gate | ✅ PASSED | hf2q `6936473d` | Codex re-review confirms **0 Critical + 0 Major** remaining. Fixed: checked_mul overflow guards, vLLM d2t/t2d name normalization, `has_own_embed_tokens` config gate, defensive `cfg.validate()` at loader entry, stale mod.rs doc. Added 4 negative-path validation tests proving each fix fires. |
| **Phase E3 totals** | ✅ **CLOSED** | hf2q `6936473d` | **45/45** unit tests PASS (15 E3a + 26 E3b + 4 codex-fix validations). |
| E4a — dynamic tree expansion algo | ✅ SHIPPED | hf2q `8601772f` | Confidence-guided global best-first implementation (inspired by EAGLE-2) via pending-candidate heap (codex caught batch-commit Critical → fixed). `Drafter` trait + `TreeContextView` carrying tokens/parents slices so GPU drafter can walk parent chain. `ExpandedTree` with f64 cum_log_probs (codex caught f32 underflow risk → fixed). `build_tree_mask` returns Result<Vec<f32>> with checked overflow (codex caught raw arithmetic → fixed). 21/21 tests PASS including explicit global-best-first proof (grandchild of A beats sibling of A when A's subtree has globally higher cum). |
| E4b.1 — GPU upload pipeline | ✅ SHIPPED | hf2q `3304a0fc` | `Eagle3DrafterTensors` with `Option<MlxBuffer>` for gated optional weights + BF16-as-F32 cast for RMSNorm weights (ADR-030 iter-106 pattern). Codex caught 2 Minor (gpu_resident_bytes overcounted CPU-resident vec; missing post-upload dtype tripwires) → fixed. 6/6 tests PASS. |
| E4b.2 — FC projection forward | ✅ SHIPPED | hf2q `87a7b1b2` | `dispatch_eagle3_fc` wrapping `apply_linear_projection_f32`. CPU parity at seq=4 (BF16 GEMM) + seq=1 (GEMV decode path) within 5e-2 tolerance using BF16-quantized weights in CPU reference. Codex caught 1 Critical (input dtype was only debug_assert) + 3 Major (silent as-u32 truncation × 2 + unchecked multiply) → fixed. 5/5 tests PASS. Codex re-review: Critical 0 + Major 0. |
| E4b.3 — input_layernorm + hidden_norm + concat | ✅ SHIPPED | hf2q `07f5c0fb` | Three forward primitives wrapping `dispatch_rms_norm` + `dispatch_feature_concat_f32`. CPU parity for both norms within 1e-3 tolerance. Sentinel-value test proves concat layout matches vLLM `torch.cat([embeds, hidden_states], dim=-1)`. Codex caught 3 Critical (3× byte-multiply overflows + missing weight-length validation) + 1 Major (zero seq_len) + 1 Minor (dim as f32 precision above 2^24) → all fixed. 7/7 tests PASS. Codex re-review: Critical 0 + Major 0. |
| E4b.4 — Q/K/V projections (+ optional bias) | ✅ SHIPPED | hf2q `174a157a` | 3 forward primitives (q/k/v_proj) wrapping apply_linear_projection_f32 from `[seq, 2*hidden]` concat input. Optional attention_bias adds via add_bias_row_2d_f32 with explicit memory_barrier between matmul (writes out) + bias-add (R/W in-place on out — debugging caught race causing diffs 0.156-0.794; with barrier: 5e-2). Tensors loader updated to cast biases BF16→F32 at upload. Codex caught 2 Major (missing seq_len*out_features bound for matmul allocation + bias-add grid). 6/6 tests PASS. Codex re-review: Critical 0 + Major 0. |
| E4b.5a — Q/K per-head RMSNorm (Qwen-style) | ✅ SHIPPED | hf2q `9113988e` | Two forward primitives (q_head_norm + k_head_norm) treating flat `[seq * num_heads, head_dim]` as RMSNorm rows. Gated by `cfg.use_qk_norm` (Qwen-3 style; Llama-targeted EAGLE-3 sets false). Codex caught 2 Major (missing memory_barrier before reading proj output + cfg gate not enforced at wrapper). All addressed. 5/5 tests PASS. Codex re-review: Critical 0 + Major 0. |
| E4b.5b — RoPE on Q/K (tree-position-aware) | ✅ SHIPPED | hf2q `e4b29171` | dispatch_eagle3_rope wraps apply_imrope with tree-position support. positions_override slice (e.g. `base_pos + tree_depths[i]` from ExpandedTree) OR linear chain. NeoX-style; rope_dim must equal head_dim (codex caught partial rotation kernel mismatch). 5/5 tests PASS including CPU parity for both linear and asymmetric tree positions within 1e-4 absolute. Codex caught 2 Major (silent position saturation + rope_dim partial) + 2 Minor (rope_theta finite check + head_dim even). All addressed. Codex re-review: Critical 0 + Major 0. |
| E4b.6 — tree_attention dispatch (Phase E1 + dk128 retrofit) | ✅ SHIPPED | mlx-native `452d33b` + hf2q `2f5c8fd8` | `dispatch_eagle3_tree_attention` thin wrapper around mlx-native's tree_attention. **Critical discovery**: Phase E1 kernel only supported head_dim ∈ {256, 512} (Gemma family); Qwen 3.6 27B uses head_dim=128. Retrofitted Phase E1 with dk128 template (tree_attention.metal + flash_attn_vec_reduce.metal + selectors + registry). Codex caught 1 Critical (existing Phase E1 unchecked usize multiply in validate_buffers, surfaced by dk128 wider input range). 3/3 hf2q tests PASS including head_dim=128 fixed-square tree CPU parity within 5e-3. mlx-native 7/7 lib tests PASS + 21/21 Phase E1 byte-identity tests unaffected. Codex re-review: Critical 0 + Major 0. |
| E4b.7 — O projection + residual add | ✅ SHIPPED | hf2q `44a4f60a` | `dispatch_eagle3_o_proj` reuses dispatch_eagle3_projection_with_optional_bias from E4b.4 (tree_attention output [q_seq, n_q, hd] is row-major-equivalent to [q_seq, n_q*hd] — no permute needed). `dispatch_eagle3_residual_add` wraps elementwise_add with codex-validated patterns; residual_add CPU parity is BIT-EXACT. Codex caught 1 Major (missing memory_barrier before o_proj reads tree_attention output — same RAW race class as E4b.4/5a/6). 4/4 tests PASS. Codex re-review: Critical 0 + Major 0. |
| E4b.8 — SwiGLU MLP | ✅ SHIPPED | hf2q (this commit) | `dispatch_eagle3_mlp` chains gate_proj + up_proj (parallel) → silu_mul → down_proj with 3 memory_barriers. Reuses dispatch_eagle3_projection_with_optional_bias (no bias on standard SwiGLU). Reuses mlx-native dispatch_silu_mul. CPU parity uses RELATIVE tolerance (1% of max_abs output) since 3 chained BF16 GEMMs + silu amplification produces ~1000-magnitude outputs at tiny_cfg shapes. 4/4 tests PASS. Codex: 0 Critical + 0 Major (2 cosmetic Minor addressed). |
| E4b.9 — final norm + lm_head | ✅ SHIPPED | hf2q `8d6fc79f` | `dispatch_eagle3_final_norm` reuses E4b.3 RMSNorm helper. `dispatch_eagle3_lm_head` handles both tied (uses embed_tokens) and untied (separate lm_head.weight) cases with 2 pre-emptive gate checks (tied requires draft_vocab==vocab + has_own_embed_tokens). Codex caught 1 Major (missing memory_barrier before final_norm reads residual — 5th catch of this pattern class in E4b). All addressed. 4/4 tests PASS. Codex re-review: Critical 0 + Major 0. |
| E4b.10a — top-K extraction from logits | ✅ SHIPPED | hf2q (this iter) | `extract_top_k_from_row_logits` — CPU log_softmax + min-heap O(V log K) with deterministic tie-break (smaller token wins on ties). Codex caught 1 Major (pre-clamp collapsed distinct tail log-probs → fix: select via unclamped f64, clamp only at materialization). 11/11 tests PASS including ordering regression for `[0.0, -3e38, -2e38]`. Codex re-review: Critical 0 + Major 0. |
| E4b.10b.1 — Q/K/V permute (seq-outer → head-outer) | ✅ SHIPPED | hf2q (this iter) | `dispatch_eagle3_permute_seq_to_head_outer` wraps `permute_021_f32` bridging E4b.4 Q/K/V layout to E4b.6 tree_attention. Pure permutation → BIT-EXACT via `to_bits()` parity. Sentinel-value layout test catches axis-swap bugs deterministically. Codex: 0 Critical + 0 Major + 1 Minor (added wrong-element-count regression). 5/5 tests PASS. |
| E4b.10b.2 — post_attention_layernorm + full forward orchestrator | ✅ SHIPPED | hf2q (this iter) | Last missing primitive (post_attention_layernorm) + `run_full_eagle3_forward` test helper chaining 14 dispatches end-to-end. Test asserts shape + finite + determinism (bit-exact across runs proves no race/stale-read across the 14-step chain). Codex caught 2 Major (zero-weight test was vacuous + helper skipped use_qk_norm branch) → both fixed with nonzero weights + cfg branch. Empirical discovery: BF16 underflow accumulates to zero logits at tiny synthetic shapes — known limitation deferred to Phase E7 real-weight validation. 2/2 tests PASS. |
| E4b.10b.3 — GpuDrafter Drafter trait impl | ✅ SHIPPED | hf2q (this iter) | `GpuDrafter` struct + impl Drafter::predict_topk via full forward chain. Promoted `dispatch_eagle3_drafter_forward` from test helper to public API. **End-to-end integration test**: `expand_dynamic_tree` (Phase E4a) → `GpuDrafter::predict_topk` → 14-stage forward → top-K → ExpandedTree. Codex caught 2 Major (base_pos depth-adjust + shape validation) — fixed. Acknowledged limitation: single-token decode loses ancestor-token conditioning for depth ≥ 2 — full path conditioning requires KV cache (deferred to Phase E5 / follow-up). 3/3 tests PASS. |
| E4 codex /cfa final gate | ✅ CLOSED | hf2q `46ca946c` | Cross-cutting codex review across entire eagle3 module. Caught 0 Critical + 2 Major + 1 Minor. Major 1 (vocab remap missing for fast-vocab projection) addressed via constructor invariant + predict_topk remap. Major 2 (only root expansion supported under single-token decode) addressed via `path.len() == 1` guard. Minor (stale docs) updated. 4 regression tests added. **Codex re-review confirms 0 Critical + 0 Major remaining**. |
| **PHASE E4 TOTALS** | ✅ **CLOSED** | hf2q `46ca946c` | **94/94 tests PASS** across 14 sub-phases + final-gate regressions. Single-token decode mode (max_depth=1 only). Full path-conditioning via drafter KV cache deferred to Phase E5b / E6. |
| E5a — tree-walk-accept algorithm | ✅ SHIPPED | hf2q `f98a1912` | `walk_tree_accept(tree, verifier_argmax) -> Vec<usize>` greedy longest-direct-child-matching walk from root. `AcceptWalk` summary wrapper with privatized fields (codex /cfa Minor). 12/12 tests PASS. Integration test verifies every (i,i+1) step follows parent→child edge and matches verifier_argmax at each level. Codex re-review: Critical 0 + Major 0. |
| E5b Step 1 — DrafterKvCache structure | ✅ SHIPPED | hf2q `7073306e` | `DrafterKvCache` with `[num_kv_heads, capacity, head_dim]` F32 GPU buffers (paired K + V). API: `new`/`append`/`rollback_to_accepted`/`clear`/`len`. Rollback: download → reorder → upload via host-visible MlxBuffer slices (zero-copy on Apple unified memory). 12/12 tests PASS including 4-node tree-walk integration. |
| E5b Step 2 — cache-aware forward variant | ✅ SHIPPED | hf2q `029f830f` | `dispatch_eagle3_drafter_forward_with_kv_cache` — same 14-stage chain as unbatched variant but splits into two encoders around CPU-side cache append. Encoder 1: fc → norms → concat → Q/K/V → optional qk_norm → RoPE → permute. CPU append of new K_perm/V_perm rows into cache. Encoder 2: tree_attention reads from cache (kv_seq_len = cache.len()) → o_proj → residual → post_attn_norm → MLP → final_norm → lm_head. 7/7 tests PASS including **byte-identity equivalence** with unbatched variant at cache.capacity=1 (proves no value drift from encoder split). |
| E5b Step 3 — GpuDrafter cache wiring | ✅ SHIPPED | hf2q `23f7a321` | `GpuDrafter::kv_cache: Option<DrafterKvCache>` + `attach_kv_cache`/`clear_kv_cache`/`rollback_kv_cache`/`kv_cache_len`. predict_topk routes to cache-aware forward when attached. Enforces invariant `cache.len() + 1 == path.len()` at entry. **LIFTS THE max_depth==1 CAP**: depth-1 child expansion now works at the primitive level. 9 new tests + 6 existing backward-compat = 15/15 drafter_gpu tests PASS. |
| E6 v1 (rollback design — DEPRECATED) | ⚠️ BUGGY | hf2q `5e35e83d` | Initial rollback-between-branches orchestrator. Critical bug: rolling back to expand A2 drops A1's K/V; later admission of A1's descendant B1 errors. Bug masked by max_depth=2 leaves never being expanded (new_depth==max_depth → predict_topk skipped). |
| E6 v2 — tree-mask orchestrator | ✅ SHIPPED | hf2q `8cf9397b` | Cache grows monotonically. `GpuDrafter::tree_node_cache_slot` tracks tree-idx → cache-slot. predict_topk builds tree-aware mask [1, cache.len()+1] selecting ancestors + self. `dispatch_eagle3_drafter_forward_with_kv_cache` gained `mask_override: Option<&[f32]>` parameter. Orchestrator no longer manages cache state (just calls clear_cache + predict_topk in best-first order). **CRITICAL regression test**: max_depth=4 cross-branch (would panic in v1) — PASSES. 6 mock tests + 2 GPU tests including max_depth=4 end-to-end. |
| E6 — Qwen35 tree-verify dispatcher | ✅ SHIPPED | hf2q `02f13d99` | `dispatch_qwen35_tree_verify_attention` in `models/qwen35/gpu_full_attn.rs` — Qwen35-namespaced wrapper around `mlx_native::ops::tree_attention` (dk128 kernel). Maintains DDD bounded-context isolation: `models/qwen35` does NOT import from `spec_decode/eagle3`. **Byte-identity (0 ULP) parity** vs `dispatch_eagle3_tree_attention` proven via `to_bits()` test on a causal chain mask — same kernel, different namespace. 5 new tests (smoke + head_dim=256 rejection + chain-mask parity + 2 negative-path validation). Shipped via /cfa dual-mode session: Claude vs Codex parallel impls, cross-reviews, queen judgment. Codex won 361 vs 343 (18-pt margin) on dim 4 (security/invariants): codex guards `num_q_heads/num_kv_heads == 0` BEFORE the GQA modulo (Claude reached `% num_kv_heads` unguarded → modulo-by-zero panic class). Merged codex impl + 3 Claude-cherry-picked improvements. |
| E6 F5 — Eagle3Orchestrator dense/MoE dispatch | ✅ SHIPPED | hf2q `e3e4a9c0` | Wires F4 MoE variant into Eagle3Orchestrator. Adds `FfnTopology { Dense, Moe }` enum + `ffn_topology` field to `Eagle3OrchestratorConfig` with **auto-detect from `Qwen35Variant`** via `FfnTopology::from_model` (no new env var — topology inferred from loaded model). `qwen35_default` constructor sets the field automatically. Extended `forward_tree_verify_gpu` per-layer loop branches on `FfnWeightsGpu::DenseQ` → F2 (`qwen35_tree_verify_full_layer_q`) and `FfnWeightsGpu::MoeQ` → F4 (`qwen35_tree_verify_full_layer_q_moe`); builds `Qwen35TreeVerifyFullLayerShapeQMoe` from `cfg.moe` at runtime; unsupported F16/F32 variants return a descriptive error. Dense default path is byte-identical to F3-shipped behavior (`attn_shape_base` hoisted out of loop; same F2 call site). 8 new ACs added + 10 existing orchestrator tests updated for the new field. **39/39 PASS** (eagle3_orchestrator + qwen35_tree_verify_full_layer combined). ADDITIVE-ONLY: F2 (line 2802), F4 (line 3204), and F3 orchestrator structure unchanged. Single-impl + lead-self-review pattern (codex tooling skipped — single-turn budget consistently insufficient for these CFAs). |
| E6 F4 — MoE production variant (qwen35_tree_verify_full_layer_q_moe) | ✅ SHIPPED | hf2q `73152041` | MoE Q4_0 production variant of F2 for Qwen 3.6 27B-A3B MoE inference. Same per-layer attention block (CFA #2) but substitutes the dense SwiGLU chain with `build_moe_ffn_layer_gpu_q` (gpu_ffn.rs:2379 — router + top-K experts + shared expert with sigmoid gate). New `Qwen35TreeVerifyFullLayerShapeQMoe` includes attn shape + full `MoeFfnShape` (num_experts + num_experts_per_tok + moe_intermediate_size + shared_intermediate_size + hidden_size). 2 NEW invariants beyond F2: INV-QMoE-ggml-type-validation (ggml_type_gate_up + ggml_type_down + BF16 dtype checks for router + shared_gate + shared_gate_inp — 5+ defense-in-depth checks before shape.validate()) + INV-QMoE-shape-weights-cross-check (moe.hidden_size == attn.hidden_size + num_experts > 0 + topk ≤ num_experts + checked_mul overflow guards on num_experts × intermediate × hidden). 8 ACs: shape-validate, production GQA smoke, 7 negative paths invoking FULL function entry, CPU reference parity at \|GPU-CPU\|_inf < 0.20, composition equivalence, 3-rep byte-identity with Metal contention guard, **AC-7 topk routing correctness (sentinel router_w → asserts specific top-K expert indices contribute)**, **AC-8 shared expert always contributes (independent of top-K routing)**. 3-encoder lifecycle (caller enc → enc2 post_attn_norm → build_moe_ffn_layer_gpu_q internal enc). 31/31 total tests PASS (21 F1+F2+F4 + 10 eagle3_orchestrator regression). /cfa session #6: claude impl shipped; codex impl skipped (3rd consecutive context-exhaustion would burn cycles); lead self-review composite 96 PASS via 12-item red-flag MoE-extended checklist. F5 orchestrator integration is a follow-up CFA. |
| E6 F3 — Multi-layer chain + HF2Q_SPEC_EAGLE3 serve flag | ✅ SHIPPED | hf2q `6f7a9b80` | End-to-end EAGLE-3 production-usable inference path. New file `src/inference/spec_decode/eagle3_orchestrator.rs` (673 LOC) houses `Eagle3OrchestratorConfig` + `Eagle3Orchestrator` + 5-step per-iter pipeline + `generate()` loop. New `Qwen35Model::forward_tree_verify_gpu` in `src/inference/models/qwen35/forward_gpu.rs` (211 LOC) loops `qwen35_tree_verify_full_layer_q` across all transformer layers with collector slab writes for multi-layer hidden plumbing. `HF2Q_SPEC_EAGLE3` env flag wired in `src/serve/spec_decode_cli.rs` (90 LOC) defaulting OFF + graceful fallback when `HF2Q_EAGLE3_DRAFTER_PATH` unset/missing. DDD bounded contexts preserved: `models/qwen35` is verifier, `spec_decode/eagle3` is drafter, orchestrator is the cross-context coordinator. 10 ACs PASS: orchestrator config validate, drafter integration, multi-layer hidden capture order, single-iter end-to-end, multi-iter cache continuity, **base vs tree-verify token parity at temp=0 (load-bearing)**, F1+F2 regression sanity, prefill/decode regression sanity, HF2Q_SPEC_EAGLE3 opt-in with mock drafter, graceful fallback. 23/23 total tests PASS (10 orchestrator + 13 F1+F2 — 0 regressions). 986 LOC additive across 5 files. /cfa dual-mode session #5: claude impl shipped end-to-end; codex impl phase context-exhausted at item_4 (same single-turn budget pattern as CFA #4); adversarial sampling via lead self-review on 15-item red-flag checklist — composite PASS. |
| E6 F2 — Qwen35 full-layer tree-verify block (Q4_0 production variant) | ✅ SHIPPED | hf2q `e29fb1df` | `qwen35_tree_verify_full_layer_q` — Q4_0 production variant of F1. Accepts `&DenseFfnWeightsGpuQ` (intermediate_size/hidden_size in weight struct + ggml_type_gate/up/down fields). Routes through `apply_linear_projection_f32`'s existing U8 branch → `quantized_matmul_ggml`. Memory: ~3 GB vs ~22 GB BF16 per 27B layer (per gpu_ffn.rs DenseFfnWeightsGpuQ doc). 2 NEW invariants beyond F1: INV-Q-ggml-type-validation (gate/up/down all Q4_0; Q5_K/Q6_K/IQ4_NL deferred to future CFA) + INV-Q-shape-weights-cross-check (shape dims match weight struct fields). 7 ACs: 6 mirror F1 + NEW **AC-7 cross-variant parity** (Q4_0 GPU ≈ F32-cast GPU at \|diff\|_inf < 0.20 on identical F32-source weights → Q4_0-quantized — load-bearing routing-correctness test). Synthetic Q4_0 fixture uses `crate::quantize::ggml_quants::q4_0::quantize` (production path). 13/13 tests PASS (6 F1 + 7 F2). ~1312 LOC additive (impl ~250 + tests/helpers ~1060). Claude impl shipped via /cfa dual-mode session #4; codex impl phase failed twice on context exhaustion, so adversarial sampling done as a codex review-only pass on the merged commit. |
| E6 — Qwen35 full-layer tree-verify block | ✅ SHIPPED | hf2q `411c99dc` | `qwen35_tree_verify_full_layer` composes the shipped per-layer attention block with post_attn_norm + dense SwiGLU MLP + final residual on a fresh second encoder. Returns `[tree_seq_len, hidden_size]` F32 ready to feed the next layer's tree-verify block. New additive `Qwen35TreeVerifyFullLayerShape` embeds the per-layer shape by value + adds `intermediate_size`. F32-cast variant only (takes `&DenseFfnWeightsGpu`); Q4_0 production variant is a follow-up CFA per §F2. 4 RAW barriers in the MLP-extension encoder with inline producer/consumer comments. /cfa dual-mode session #3: claude wins 92 vs 63 (29-pt margin) on 6/6 vs 4/6 AC pass rate — codex reproduced 2 prior-CFA defect classes (params-buffer pool discipline broke AC-6 byte-identity determinism + AC-5 1e-3 hybrid-oracle tolerance unreachable with BF16-cast; codex measured 3.979e-2 drift). Claude declared 2 spec deviations openly (AC-5 tolerance widened to 5e-2 with BF16-slop rationale + AC-5 oracle restructured from hybrid to full CPU); codex's own cross-review JUSTIFIED both. 6 ACs + 0 regressions vs 303/303 ADR-037 baseline = **309/309 PASS**. ~1046 LOC additive (~350 impl + ~700 tests including ~250-LOC `cpu_tree_verify_full_layer_ref` scalar oracle). |
| E6 — Qwen35 per-layer tree-verify block | ✅ SHIPPED | hf2q `40ec522c` | `qwen35_tree_verify_attention_block` — runs ONE Qwen3.5 transformer layer's full attention sub-block in tree-verify mode: pre-attn RMSNorm → Q/K/V/Gate proj → per-head Q/K RMSNorm → IMROPE → seq→head-outer permute → KV-cache append at `[prefix_len, prefix_len+tree_seq_len)` via CPU memcpy through host-visible MlxBuffer (between encoder commits) → `dispatch_qwen35_tree_verify_attention` → sigmoid-gate multiply → O proj → residual add. Stops at attention residual (post-attn-norm + MLP block is a future CFA). 7 explicit RAW barriers with inline comments per E4b codex-fix discipline. Shipped via /cfa dual-mode: claude_wins 92 vs 74 (18-pt margin) on test rigor — codex's tests passed green but T5 was a no-op alias to T6 and T6+T7 used `scale=0.0` zero-weights collapsing every projection to the identity path. Claude's tests genuinely exercise the chain: 190+ LOC pure-scalar CPU reference (matmul + rms_norm + IMROPE + softmax + sigmoid-gate + residual) at small head_dim=128 shape with non-zero weights achieving `\|GPU-CPU\|_inf = 2.23e-4` (real systematic-error bound), plus prefix=0 chain parity at byte-identity (0.0 diff). Merged Claude impl + 3 codex-minor fixes: tightened `hidden_states_in` element-count `==`, weight by-shape validation on F32 norms, 3/5 negative-path tests now invoke full function entry. 5 tests + 0 regressions; full ADR-037 suite 193/193 PASS. |
| Phase E5 + E6 codex /cfa gate | ⏳ TODO | — | Cross-cutting review of kv_cache + cache-aware forward + GpuDrafter cache wiring + orchestrator |
| HF2Q_SPEC_EAGLE3 env flag | ⏳ TODO | — | Wire serve-layer env flag analogous to HF2Q_SPEC_DECODE / HF2Q_SPEC_DFLASH so production users can opt in |
| E2 — EAGLE-3 drafter training | ⏳ TODO | — | Multi-week H100 compute |
| E7 — 3-rep paired empirical validation | ⏳ TODO | — | Target: ≥1.3× on 2K natural + ≥1.5× on long code-gen (blocked on E2 trained weights) |
| E8 — codex /cfa final + ADR-034 update + merge | ⏳ TODO | — | EAGLE-3 closure |
| **E9 — HASS / Hydra research** | Proposed; corrected by §10 | Historical scope `73cec4ee` | ADR-039 is a research proposal. Former ensemble/schema, joint-loss, LOC and H1–H10 estimates are withdrawn; artifact compatibility and a replacement design require deciding spikes before execution. |

## 0. The right thing first

The metric is **max coherence + max perf at long context**. Picking the architecture by ease-of-implementation lands at a local minimum we'd have to throw away. The following historical comparison motivated the choice; unmatched paper ratios and qualitative coherence rankings are not current decision evidence:

| Approach | Speedup vs base | Drafter | Tree topology | Coherence at long ctx |
|---|---:|---|---|---|
| Medusa (Cai 2024) | 1.5-2.5× | N parallel heads from frozen target last hidden | Fixed | Lower (single hidden, no per-depth conditioning) |
| EAGLE-1 (Li 2024a) | 2.5-3.0× | 1-layer drafter from target last hidden | Fixed | Medium |
| EAGLE-2 (Li et al., 2024) | Paper-specific; not comparable here | EAGLE-1 + dynamic tree | Dynamic | Requires matched evaluation |
| **EAGLE-3 (Li et al., 2025)** | Paper-specific; not a hf2q baseline | **Drafter from multi-layer hidden + dynamic tree** | **Dynamic** | Requires matched evaluation |
| HASS / Hydra (distinct 2024 methods) | No common comparable ratio | HASS: training harmonization; Hydra: dependent draft heads | Method-specific | Unqualified in hf2q |
| Lookahead / REST | 1.5-2.0× | Training-free n-gram or retrieval | Static | Low |

**Historical choice:** pursue EAGLE-3 with dynamic trees. Neither a paper ranking nor an available reference implementation establishes the fastest qualified path for hf2q; current selection requires matched measurements.

## 1. Long-context benefit hypothesis and deciding measurement

Per ADR-034 root-cause profiling at historical HEAD `1fb53d58`:
- Drafter accept-rate collapses from 82% (short) → 60% (2K natural) — `-22pp`
- Per-round verifier kernel cost grows only `+7%`
- The MTP head conditions on one hidden state; underconditioning was the
  working explanation, not a proof that a different drafter closes the gap.

EAGLE-3's multi-layer feature fusion, autoregressive drafting and training-time
test motivate a **hypothesis** that a compatible checkpoint could improve
acceptance enough to offset draft and verification cost at long context.
Confidence-guided dynamic trees are another candidate lever. Neither the
architecture nor its training recipe guarantees that outcome for hf2q.

The former citation claiming Table 5 proves acceptance ≥0.78 at 4K+ was wrong.
[EAGLE-3 v3, §4.4/Table 5](https://arxiv.org/html/2503.01840v3) reports
throughput improvements across batch sizes, not that context-length acceptance
result. It supplies no closure evidence for the recorded local 82%→60% failure.

**Deciding measurement:** qualify an exact target/drafter artifact pair, then
compare ordinary decoding, the existing MTP path and EAGLE-3 on the same short,
2K and longer natural/code prompts, sampling settings and host conditions.
Record acceptance by depth/context, draft and verify wall time, prefill/decode
rates, semantic TTFT, memory and output/tool quality over alternating paired
runs. Require the applicable ADR-063 quality and speed gates before claiming
that the long-context regression is closed. Custom training remains subject to
ADR-063's explicit decision gates; this hypothesis creates no training approval.

## 2. Peer references

### vLLM — direct production reference (MIT)
- `/opt/vllm/vllm/v1/spec_decode/eagle.py` (22 LOC) — entry point
- `/opt/vllm/vllm/v1/spec_decode/llm_base_proposer.py` (1685 LOC) — parent class with dynamic tree expansion
- `/opt/vllm/vllm/model_executor/models/llama_eagle3.py` (460 LOC) — EAGLE-3 model arch (1-layer trimmed LM consuming multi-layer hidden states)
- Production-deployed; battle-tested at scale.

### EAGLE author repo (peer to read, not link to)
- Reference C++/Python implementation + training recipe + pre-trained checkpoints for Llama family. Qwen 3.6 27B checkpoint NOT published yet → we train our own (see §4 Phase E2).

### llama.cpp (MIT)
- `/opt/llama.cpp/common/speculative.cpp` declares `COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3` — runtime arch contract for EAGLE-3 integration.

### Why NOT just port Medusa as a stepping stone
Per the operator's directive: skipping straight to EAGLE-3 saves ~3000 LOC of Medusa code we'd otherwise write + throw away. The Metal kernels (tree-attention with dynamic mask), tree-walk-accept logic, KV-rollback infrastructure are 90% common between Medusa and EAGLE — building them for Medusa would NOT meaningfully derisk EAGLE-3. The marginal Medusa-specific cost is the parallel-head loader (~300 LOC) which is throwaway.

## 3. Architecture summary (what we're building)

```
┌─────────────────────────────────────────────────────────────────┐
│ Qwen 3.6 27B target model (frozen)                              │
│                                                                  │
│  Layer 0 ──┐                                                    │
│  Layer 1   │                                                    │
│  ...       ├──> hidden_layers[selected] ──> EAGLE-3 drafter     │
│  Layer 62  │                                                    │
│  Layer 63 ─┘                                                    │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ EAGLE-3 drafter (1 transformer layer, ~600M params)             │
│                                                                  │
│   step 0: from target_hidden[selected] → token t1, logits l1    │
│   step 1: from (t1, drafter_state) → token t2, logits l2        │
│   step 2: from (t2, drafter_state) → token t3, logits l3        │
│   ...                                                            │
│   dynamic tree: at each step, top-K candidates by confidence    │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ Tree-attention verifier pass (target model, 1 batched forward)  │
│                                                                  │
│   Input: tree of N candidate tokens with tree-mask              │
│   Output: per-node logits + accept-walk                         │
│   Accept: longest matching path from root                       │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                          Accepted tokens + KV rollback
```

## 4. Implementation phases (~3500-4000 LOC, ~4-6 weeks)

### Phase E1 — Tree-attention Metal kernel with dynamic mask (~700 LOC, ~1 week)
- `mlx-native/src/shaders/tree_attention.metal` — variant of flash_attn_vec consuming a dynamic tree-mask buffer.
- Mask encoding: `[seq_len + tree_total_nodes, tree_total_nodes]` boolean; row = position attending, column = position being attended to. Lower-triangular within prefix + tree-position-specific within tree-nodes.
- Variable tree size at runtime (dynamic tree budget per round).
- Dispatch: `apply_tree_attention(encoder, q, kv_full+tree, tree_mask, output, params)`.
- 5 parity tests: pass-through (tree=1), chain (tree=1×N), fixed square (tree=4×4), dynamic asymmetric tree, prefix+tree combined.
- Codex /cfa gate Critical 0.

### Phase E2 — EAGLE-3 drafter training (~1 week training time + ~400 LOC infra)
- Adopt published EAGLE-3 training recipe for Qwen 3.6 27B target.
- Dataset: long-context distillation traces from target model (~50K samples, mix of code-gen + essay + 2-4K natural prompts).
- Compute: 1-2 days on H100 cluster (see ADR-029 for compute access pattern).
- Output: 1-layer drafter checkpoint + multi-layer-hidden selection config.
- Validation gate: drafter accept rate ≥75% on validation set at 2K context.
- Defer if hf2q-org doesn't have training compute available → use a smaller Qwen3 target (e.g., 3B) where published EAGLE-3 checkpoints exist for cross-checkout.

### Phase E3 — Drafter loader + multi-layer hidden state plumbing (~500 LOC, ~3-5 days)
- New `src/inference/spec_decode/eagle3/eagle3_weights.rs` — 1-layer transformer weights loader (mirrors qwen35_weights.rs Q/K/V/O + FFN + RMSNorm + LM head).
- New `src/inference/spec_decode/eagle3/multi_layer_hidden.rs` — extends Qwen35Model::forward_gpu_with_hidden_dflash to capture from N selected layers (config-driven), not just last layer.
- Coexists with DFlash hidden-state capture (similar infra: LayerActivations with target_layer_filter from task #78 Step 3c.A.4).

### Phase E4 — EAGLE-3 drafter forward + dynamic tree expansion (~600 LOC, ~5-7 days)
- `eagle3_forward.rs` — 1-layer drafter forward consuming multi-layer hidden + previous draft token's embedding.
- Reuse existing kernels: norm_rotary_kv, apply_flash_attn_vec, apply_linear_projection_f32 (drafter is just a small Qwen-style transformer block).
- Dynamic tree expansion (from EAGLE-2 lineage):
  - Maintain priority queue of tree nodes by `cumulative_log_prob`
  - Expand top-K nodes per step until total budget reached (default 64 nodes)
  - Output: tree_tokens (`Vec<u32>`), tree_parents (`Vec<usize>`), tree_depths (`Vec<usize>`), tree_mask buffer
- Codex /cfa gate.

### Phase E5 — Tree-walk-accept logic (~350 LOC, ~3-4 days)
- Given verifier's per-tree-node argmaxes + tree_parents → find longest matching path from root.
- KV rollback: per-tree-node positions, similar to existing rollback_la_to + truncate_full_attn_to (Phase 3 of task #78).
- Edge cases: empty accept (only root token), full accept (all branches accepted — pick deepest).
- Synthetic test: deterministic tree-walk reference impl in Python (in-tree fixture) for parity.

### Phase E6 — Orchestrator integration (~500 LOC, ~5-7 days)
- New `src/inference/spec_decode/eagle3/orchestrator.rs` — propose tree → verify → tree-walk accept → emit accepted path + free continuation.
- Wire into `serve/spec_decode_cli.rs` behind `HF2Q_SPEC_EAGLE3=1` env flag (opt-in, parallel to existing HF2Q_SPEC_DECODE).
- Production-recommendation routing (NOT auto-route initially — explicit env flag for safety; document threshold in ADR-034 post-validation):
  - Short prompts (<500 tokens): keep MTP K=1 (`HF2Q_SPEC_DECODE=1`, validated 1.36×)
  - Long prompts (≥500 tokens): EAGLE-3 (target ≥1.5×)

### Phase E7 — Empirical validation (~250 LOC, ~3-5 days)
- 3-rep paired bench: EAGLE-3 vs MTP vs base across pinned fixtures.
  - `tests/fixtures/long_prompt/natural_2k_verifier_rs_head400.txt` (existing) — long natural
  - New `tests/fixtures/long_prompt/code_gen_long.txt` — long code-gen
  - New `tests/fixtures/long_prompt/essay_long.txt` — long essay
- Acceptance gate: EAGLE-3 ≥1.3× base on 2K natural (vs current MTP 0.85×), ≥1.5× on long code-gen.
- Coherence: tree-walk-internal consistency (NOT byte-identity to base — same ADR-030 §3.2 CAVEAT applies).
- Accept-rate per depth: should sustain ≥70% at depth-4 of dynamic tree.

### Phase E8 — Codex /cfa + ADR closure (~100 LOC, ~1-2 days)
- Codex /cfa across all phase commits + final meta-review.
- Update ADR-034 production recommendation table.
- Mark ADR-037 STATUS = SHIPPED.

**Total: ~3500-4000 LOC across 8 phases, ~4-6 weeks calendar time** including drafter training in Phase E2.

## 5. Acceptance gates per phase (per the task-#95 sub-iter discipline)

| Phase | Acceptance gate |
|---|---|
| E1 | 5 tree-attention parity tests PASS; codex /cfa Critical 0 |
| E2 | Drafter accept rate ≥75% on 2K natural validation set |
| E3 | Multi-layer hidden capture reproducible across 3 reps; tensor shapes match config |
| E4 | Dynamic tree expansion deterministic given seed; tree shape matches expected confidence distribution |
| E5 | Tree-walk accept matches Python reference on synthetic test cases |
| E6 | E2E generation under HF2Q_SPEC_EAGLE3=1 produces coherent text |
| E7 | 3-rep paired bench: ≥1.3× base on 2K natural, ≥1.5× on long code-gen |
| E8 | Codex /cfa final clean + ADR-034 updated + commit/push to main |

## 6. Risk register + mitigations (no fallbacks-to-easier-architecture)

| Risk | Mitigation |
|---|---|
| Training compute not available for Phase E2 | Sub-iter: validate on smaller Qwen3 target with published EAGLE-3 checkpoint to derisk; production ship blocked on training. Note: per operator directive "MULTI-WEEK EFFORTS ARE IN SCOPE" — training time IS in budget. |
| Tree-attention kernel diverges from F32 single-token decode | Scope claim to accept-walk consistency only (same as ADR-030 §3.2 CAVEAT). Phase E1 includes parity at tree=1 (= chain = pass-through to flash_attn_vec). |
| Dynamic tree-walk has bugs causing coherence collapse | Phase E5 synthetic test against Python reference; codex /cfa Critical 0 gate |
| Published EAGLE-3 gains do not transfer to the chosen target | The historical 1.3–1.5× gate still requires matched local quality and performance evidence; no fraction of a paper ratio proves a local win. |
| MoE 35B-A3B compatibility | Initial port targets dense Qwen 3.6 27B. MoE port deferred to follow-up ADR (per scope discipline). |
| LOC budget overruns | Per-phase commits + codex /cfa per phase catch drift; each phase is independently shippable. |

## 7. NOT doing (explicitly out of scope)

- Medusa port (~3000 LOC throwaway — see §2 last paragraph)
- EAGLE-1 port (EAGLE-3 supersets it on every metric)
- Static tree (historically rejected in favor of confidence-guided expansion; published results do not prove universal superiority)
- MoE 35B-A3B variant (follow-up ADR after E9)
- Lookahead / REST / n-gram (lower ceiling than EAGLE-3)
- Hand-rolled training without published recipe (would waste weeks)

> ~~HASS / Hydra~~ — **MOVED IN-SCOPE as Phase E9** per operator 2026-05-22 ("never put off hard work"). See header note + Phase E9 row + new §10a HASS/Hydra plan.

## 8. First concrete step (next iteration)

**Phase E1.1**: Create `mlx-native/src/shaders/tree_attention.metal` skeleton + dispatch wrapper. First parity test: tree=1 should pass through to existing flash_attn_vec output byte-identically.

This is the smallest unit that derisks the rest of the port. If tree-attention kernel can't be made to pass parity at tree=1, the entire EAGLE-3 stack rests on broken foundation and we discover that in week 1 (not week 5).

## 9. Cross-references

- ADR-034 §3.4 "Tree decoding" row + Mission status open-items
- ADR-034 G3 row at line 1501 — accept-walk consistency invariant
- ADR-030 §3.2 CAVEAT — byte-identity to base falsified
- ADR-029 — compute access patterns for Phase E2 training
- `feedback_long_context_bench_methodology_2026_05_22` memory — pinned fixture methodology
- vLLM peer code: `/opt/vllm/vllm/v1/spec_decode/{eagle.py, llm_base_proposer.py, model_executor/models/llama_eagle3.py}`
- llama.cpp peer code: `/opt/llama.cpp/common/speculative.cpp`
- EAGLE-3 paper (Li et al. 2025) — for training recipe + architecture details

## 10. HASS / Hydra plan (Phase E9 — IN-SCOPE; not deferred)

### 10.1 Corrected research boundary (2026-10-23)

The May operator request to investigate HASS/Hydra remains recorded above.
The former parallel-transformer ensemble design and its LOC estimates were
based on a misidentification and are withdrawn, not implemented milestones.

[HASS, Zhang et al.](https://arxiv.org/abs/2408.15766) is a training method
for an EAGLE-style autoregressive drafter: objective distillation and alignment
of training context with recursive inference. It is not an ensemble of four
transformers, and Hydra is not its earlier name. [Hydra, Ankner et al.](https://arxiv.org/abs/2402.05109)
is a separate sequentially dependent draft-head method in the Medusa setting.
Paper-level speed ratios have different targets and evaluation conditions and
cannot select a hf2q workload winner or predict compounded gains.

### 10.2 Replacement E9 deciding spikes — Proposed

1. Pin an official checkpoint, target revision, license, tokenizer, tensor
   schema, hidden-feature interface, and training/inference recipe. Prove
   compatibility independently for each intended hf2q family; do not assume
   an EAGLE-3 loader accepts HASS or Hydra weights.
2. Specify the drafter graph and verifier/cache/rollback requirements from
   that artifact. Native primitives/kernels belong in mlx-native; loaders,
   model graphs and serving policy belong in hf2q.
3. Establish exact verification semantics and matched quality before paired
   end-to-end measurements against ordinary decoding and qualified EAGLE-3.
   Measure draft/verify cost, acceptance, TTFT and semantic streaming behavior.

These are research gates, not execution approval or a new training commitment.
ADR-063 excludes custom training from its current release work pending its
explicit decision gates. Existing verifier code is a candidate reuse surface,
not evidence that either distinct drafter needs zero verifier changes.

## 11. Decision

**Historical approved scope (May 2026)**: implement EAGLE-3 with dynamic tree decoding in 8 phases (E1-E8), then HASS / Hydra as Phase E9 per §10. No Medusa fallback, no incremental "stepping stones". Commit to the architecture that delivers max coherence + perf from phase 1.

**First-action next iteration**: Phase E1.1 — `mlx-native/src/shaders/tree_attention.metal` skeleton + tree=1 parity test. (SHIPPED.)

**Acceptance for ADR-037 closure (EAGLE-3 portion)**: Phase E7 empirical validation shows EAGLE-3 ≥1.3× base on pinned 2K natural prompt + ≥1.5× on long code-gen, AND Phase E8 codex /cfa final clean.

**Historical full-mission intent**: evaluate HASS/Hydra after EAGLE-3. The old E9.5/E9.6 ensemble plan is withdrawn by §10; replacement design remains Proposed and does not change ADR-063 release gates.

## 12. 2026-08-08 release-gate correction: concurrent encoder ordering

The 0.1.3 packed-crate verification exposed one intermittent failure of the
E5b Step-2 golden byte-identity test at exact main `1a499621`: the unbatched
path produced `-0.039354406` while the cache-aware path produced
`-0.039684728` at logits index zero. Five fresh-process focused repetitions,
all 235 Eagle3 tests, and a later full 3,924-test run passed, but a green rerun
does not invalidate a coherence failure.

Source review found three missing read-after-write orderings before RoPE in
both drafter-forward variants. mlx-native intentionally uses
`MTLDispatchTypeConcurrent`; without a hand-placed barrier, FC output may be
read by `hidden_norm` before the projection completes, the two normalized
branches may be read by concat before completion, and concat output may be
read by Q/K/V projections before both column copies complete. The later
forward stages already carried explicit barriers.

The accepted correction is deliberately narrow:

1. `dispatch_eagle3_hidden_norm` orders the upstream FC projection before
   RMSNorm.
2. `dispatch_eagle3_concat_2x_hidden` orders both normalized inputs before
   its two disjoint column-copy dispatches.
3. The same helper places one barrier after both disjoint copies, preserving
   concurrent Q/K/V projection dispatch while making the complete concat
   visible to all three consumers.

This is an hf2q graph-ordering defect, not an mlx-native allocator or kernel
defect, so no dependency release is required. Acceptance requires the exact
bit-identity gate under default concurrent dispatch, the
`HF2Q_FORCE_SERIAL_DISPATCH=1` negative-control arm, a fresh packed-crate full
suite, and the normal exact-artifact release proof. The golden test accepts
`HF2Q_EAGLE3_EQUIVALENCE_REPETITIONS` in `1..=1000` so release verification can
run a same-process soak without hiding the result behind repeated cargo/test
startup. Relaxing the equality contract or ignoring the intermittent receipt
is explicitly rejected.
