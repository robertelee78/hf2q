# ADR-037 — EAGLE-3 with dynamic tree port (max coherence + perf)

- **Status**: 🚧 **IN PROGRESS** — Phases E1 + E3 + E4 + E5 + E6 SHIPPED at hf2q `5e35e83d`. **185/185 eagle3 module tests PASS single-threaded**. Remaining: codex /cfa Critical 0 final gate on E5+E6, HF2Q_SPEC_EAGLE3 env flag wiring, E2 drafter training (multi-week compute), E7 empirical validation, E8 final closure.
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
| E4a — dynamic tree expansion algo | ✅ SHIPPED | hf2q `8601772f` | EAGLE-2 GLOBAL best-first expansion via pending-candidate heap (codex caught batch-commit Critical → fixed). `Drafter` trait + `TreeContextView` carrying tokens/parents slices so GPU drafter can walk parent chain. `ExpandedTree` with f64 cum_log_probs (codex caught f32 underflow risk → fixed). `build_tree_mask` returns Result<Vec<f32>> with checked overflow (codex caught raw arithmetic → fixed). 21/21 tests PASS including explicit global-best-first proof (grandchild of A beats sibling of A when A's subtree has globally higher cum). |
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
| E6 — cache-aware orchestrator | ✅ SHIPPED | hf2q `5e35e83d` | `CacheControlDrafter` trait + `expand_dynamic_tree_with_cache` orchestrator. Maintains `cache_nodes: Vec<usize>` slot↔tree-node mapping; rolls back cache between best-first branches by computing keep_slots from the ancestor walk of the next node. End-to-end GPU test at budget=8, max_depth=2, top_k=3 expands valid tree. 6 new tests (5 mock-tracked + 1 GPU). |
| Phase E5 + E6 codex /cfa gate | ⏳ TODO | — | Cross-cutting review of kv_cache + cache-aware forward + GpuDrafter cache wiring + orchestrator |
| HF2Q_SPEC_EAGLE3 env flag | ⏳ TODO | — | Wire serve-layer env flag analogous to HF2Q_SPEC_DECODE / HF2Q_SPEC_DFLASH so production users can opt in |
| E2 — EAGLE-3 drafter training | ⏳ TODO | — | Multi-week H100 compute |
| E7 — 3-rep paired empirical validation | ⏳ TODO | — | Target: ≥1.3× on 2K natural + ≥1.5× on long code-gen (blocked on E2 trained weights) |
| E8 — codex /cfa final + ADR-034 update + merge | ⏳ TODO | — | Mission closure |

## 0. The right thing first

The metric is **max coherence + max perf at long context**. Picking the architecture by ease-of-implementation lands at a local minimum we'd have to throw away. Compare published numbers and architectural properties:

| Approach | Speedup vs base | Drafter | Tree topology | Coherence at long ctx |
|---|---:|---|---|---|
| Medusa (Cai 2024) | 1.5-2.5× | N parallel heads from frozen target last hidden | Fixed | Lower (single hidden, no per-depth conditioning) |
| EAGLE-1 (Li 2024a) | 2.5-3.0× | 1-layer drafter from target last hidden | Fixed | Medium |
| EAGLE-2 (Li 2024b) | 3.0-3.5× | EAGLE-1 + dynamic tree | Dynamic | Higher |
| **EAGLE-3 (Li 2024c)** | **4.06×** | **1-layer drafter from multi-layer hidden + dynamic tree** | **Dynamic** | **Highest** |
| HASS / Hydra (2024) | 3.5× | Multiple parallel transformer drafters | Static or dynamic | Medium-high |
| Lookahead / REST | 1.5-2.0× | Training-free n-gram or retrieval | Static | Low |

**EAGLE-3 + dynamic tree** is the published SOTA for autoregressive draft-based speculative decoding. The training recipe is published; the vLLM implementation is open-source MIT-licensed reference. Committing directly.

## 1. Why EAGLE-3 closes the long-context regression specifically

Per ADR-034 root-cause profiling at HEAD `1fb53d58`:
- Drafter accept-rate collapses from 82% (short) → 60% (2K natural) — `-22pp`
- Per-round verifier kernel cost grows only `+7%`
- MTP head conditions on ONE hidden state — underconditioned at long context

EAGLE-3 addresses this on every axis:

1. **Multi-layer hidden state aggregation**: drafter consumes hidden states from MULTIPLE target layers (not just the last). At long context, where information about long-range structure is distributed across layers, the drafter has more signal to condition on. Empirically: EAGLE-3 paper Table 5 shows accept rate stays ≥0.78 even at 4K+ tokens.

2. **Dedicated drafter LM**: 1-layer transformer that maintains its OWN context across draft steps. Unlike Medusa's frozen-hidden-state parallel heads (which can't update predictions based on earlier draft tokens), EAGLE-3's drafter generates progressively-conditioned drafts — token at depth-3 conditions on draft tokens at depths 1, 2.

3. **Dynamic tree (from EAGLE-2 lineage)**: instead of fixed N×D, expand candidates by confidence. Budget that would be wasted on low-confidence branches is reallocated to high-confidence ones. Critical for long-context where confidence distribution is skewed.

4. **Trained alignment**: EAGLE-3 training recipe explicitly aligns drafter+verifier on long-context corpora. Generic Medusa heads (Hugging Face pre-trained) won't have this property.

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
| EAGLE-3 paper's 4.06× doesn't transfer to Qwen 3.6 27B | Acceptance gate is 1.3-1.5× (NOT 4.06×) — we set a conservative floor that still beats current MTP 0.85× long-context regression. Even if EAGLE-3 only delivers ½ its paper claim on Qwen, it still recovers the regression. |
| MoE 35B-A3B compatibility | Initial port targets dense Qwen 3.6 27B. MoE port deferred to follow-up ADR (per scope discipline). |
| LOC budget overruns | Per-phase commits + codex /cfa per phase catch drift; each phase is independently shippable. |

## 7. NOT doing (explicitly out of scope)

- Medusa port (~3000 LOC throwaway — see §2 last paragraph)
- EAGLE-1 port (EAGLE-3 supersets it on every metric)
- Static tree (EAGLE-2 dynamic tree is strictly better per published numbers)
- MoE 35B-A3B variant (follow-up ADR after E8)
- Lookahead / REST / n-gram (lower ceiling than EAGLE-3)
- Hand-rolled training without published recipe (would waste weeks)

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
- EAGLE-3 paper (Li et al. 2024c) — for training recipe + architecture details

## 10. Decision

**Approved scope**: implement EAGLE-3 with dynamic tree decoding in 8 phases (E1-E8) per §4. No Medusa fallback, no incremental "stepping stones". Commit to the architecture that delivers max coherence + perf from phase 1.

**First-action next iteration**: Phase E1.1 — `mlx-native/src/shaders/tree_attention.metal` skeleton + tree=1 parity test.

**Acceptance for full ADR-037 closure**: Phase E7 empirical validation shows EAGLE-3 ≥1.3× base on pinned 2K natural prompt + ≥1.5× on long code-gen, AND Phase E8 codex /cfa final clean.
