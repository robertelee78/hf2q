# ADR-017 Phase E.a iter-3.6 + Phase B-hybrid — Architecture Research

- **Date:** 2026-05-05
- **Author:** main session, deep-research mode (per mantra: dive deep before changing code)
- **Hardware target:** M5 Max 128 GB unified memory
- **Family scope:** Gemma 4 (iter-3.6) + Qwen 3.6 (Phase B-hybrid)
- **Status:** READ-ONLY research; no code changes; no /cfa swarm spawned

## Why this dossier

Earlier in this loop session I spawned two parallel /cfa swarms (iter-3.6 + Phase B-hybrid) without first verifying the architectural assumptions in their briefs. The user's mantra-redirect ("re-read mantra; we need the best possible outcome") flagged that I'd violated "Always dive deep and ensure you know the problem you're solving" and "Chesterton's fence; always understand current fully before changing it." This dossier corrects that: I read the actual code paths before any further plan.

## TL;DR

- **iter-3.6 (Gemma long-prompt LCP) — feasible at ~300-500 LOC.** The flash_attn_vec Metal kernel already supports `mask_type=2 + sliding_window=sw` (verified at `/opt/mlx-native/src/shaders/flash_attn_vec.metal:167-169`). The Chesterton's fence at `forward_mlx.rs:2632-2635` documents WHY today uses `mask_type=1 + ring`: with a ring buffer, slot index ≠ logical position, so kernel-side sliding-window masking (which keys off slot index) would mask the wrong slots. With a LINEAR buffer (slot == position), `mask_type=2 + sliding_window=sw` is correct. Scope: prefill alloc + per-token write + flash_attn_vec params (prefill+decode) + snapshot guard lift. TQ-packed and HB-encoded caches stay ring-wrapped (they're unaffected because LCP path uses dense_kvs only). New env flag `HF2Q_KV_LCP_LONG_RESUME=1` (default OFF, opt-in).

- **Phase B-hybrid (Qwen 3.6 LCP) — feasible BUT scope-bifurcated.** Qwen 3.6 is a hybrid SSM (gated DeltaNet) + full-attn architecture. 48 of 64 layers in Qwen3.6 27B are linear-attn DeltaNet (recurrent SSM state, NOT slot-addressable KV); 16 of 64 are full-attn (linear K/V buffers, slot-addressable like Gemma's global layers). `HybridKvCacheSnapshot` and `HybridPromptCache` already exist (Wedge-3 / ADR-005 iter-216 Phase B) and serve full-equality replay (= Phase E option (b)). For PARTIAL-prefill resume the architectural challenge is the recurrent SSM state: it's POSITION-DEPENDENT (state at end-of-prefill of Q ≠ state at end-of-prefix of P even when LCP(Q, P) > 0). Three workable options analyzed below; recommended path is **Option 1 (full-attn-only resume)** for a v1 ship — moderate speedup, simple architecture, low risk.

## Section 1 — iter-3.6 (Gemma long-prompt LCP) findings

### 1.1 Why current iter-3.5c guard exists (Chesterton's fence)

`engine.rs:4516` (non-streaming) and `engine.rs:7027` (streaming): `let prefill_safe = !has_sliding_layer || prompt_len <= sliding_window;`. When `prefill_safe == false`, the snapshot-store is SKIPPED.

Reason: when a sliding layer's ring (capacity=sw) wraps during prefill (prompt_len > sw), the live ring contains slots representing positions [N-sw..N), not [0..N). The end-of-prefill snapshot captures whatever's in the ring — which after wrap is post-wrap state. Future resume at K < N-sw would expect slots to represent [0..K); the snapshot doesn't contain that. iter-3.5c skips store rather than ship a wrap-corrupted snapshot.

### 1.2 The decode-path Chesterton's fence

`forward_mlx.rs:2632-2635` carries this critical comment:

> "In ring mode we use mask_type=1 (causal) since the ring itself applies the sliding-window constraint — the kernel's sliding-window mask would incorrectly mask slots whose logical positions don't equal their slot index."

This is the load-bearing architectural reason today's code uses `mask_type=1 + ring + cap=sw` instead of `mask_type=2 + linear + sliding_window=sw`. The kernel's sliding-window masking keys off SLOT INDEX. With a ring, slot index ≠ logical position once wrap happens. So kernel-side masking would mask the wrong slots.

For iter-3.6's design, the buffer is LINEAR (no wrap) when env-flag is ON. Slot index == logical position. Kernel masking is correct.

**Verified at `/opt/mlx-native/src/shaders/flash_attn_vec.metal:166-170`:**
```metal
uint window_start = 0;
if (params.mask_type == 2 && params.sliding_window > 0) {
    window_start = (abs_pos >= params.sliding_window)
        ? (abs_pos - params.sliding_window + 1) : 0;
}
```

The kernel uses `abs_pos` (= `kv_seq_len - 1`, derived from `params.kv_seq_len`) as the query position. For prefill at token tok_i, `kv_seq_len = tok_i + 1`, so `abs_pos = tok_i`. The kernel masks K positions outside `[abs_pos - sw + 1, abs_pos]` — exactly sliding-window semantic. CORRECT for linear buffer.

### 1.3 TQ-packed and HB-encoded caches are UNAFFECTED

`forward_prefill.rs:1136-1167` writes to TQ-packed `kv_caches[layer].k_packed/v_packed`. `forward_prefill.rs:1184+` writes to HB-encoded `leg_hb_encoded[layer]`. Both use ring semantics for sliding layers.

When `INVESTIGATION_ENV.use_dense == true` AND `dense_kvs.is_some()`, decode reads `dense_kvs` (line 2495: `if use_dense_sdpa`). The TQ-packed and HB-encoded caches are populated but UNUSED in this regime. So iter-3.6 does NOT need to change their ring/wrap behavior.

(If a future operator runs without `use_dense=1`, decode would read TQ-packed and the long-prompt LCP wouldn't engage at all — the existing iter-3 env-gate `HF2Q_USE_DENSE=1` already guards this.)

### 1.4 iter-3.6 implementation plan (validated against code)

#### 1.4a Files to touch

| File | Change | LOC est |
|---|---|---|
| `src/debug/investigation_env.rs` | Add `kv_lcp_long_resume: bool` field + env-parse line | ~5 |
| `src/serve/forward_prefill.rs` | Sliding-layer alloc (line 639): `cap = sw.max(seq_len + max_decode_tokens)` when env-on; per-token write_slot (line 1095-1099): `slot = tok_i` (no `% cap`) when env-on AND layer is sliding; flash_attn_vec params (line 1249-1259): `mask_type=2`, `sliding_window=sw` when env-on AND layer is sliding; snapshot guard (line 1778-1817): drop the `seq_len <= sw` predicate when env-on | ~80 |
| `src/serve/forward_mlx.rs` | Decode flash_attn_vec params (line 2641-2651): same change as prefill — `mask_type=2`, `sliding_window=sw` when env-on AND layer is sliding | ~30 |
| `src/serve/api/engine.rs` | Engine guards (line 4516 + 7027): drop the `prompt_len <= sliding_window` skip when env-on | ~20 |
| `tests/lcp_partial_prefill_byte_identity.rs` | Falsifier test `iter3_6_long_prompt_resume_byte_identity` | ~120 |
| Total | | **~255 LOC** |

#### 1.4b Walk discipline

1. **Test first**: write `iter3_6_long_prompt_resume_byte_identity` that EXPECTS `lcp_detected_total ≥ 1` after sending Q (long, > sw) then P (long, shares prefix with Q). Today this test FAILS (engagement assertion fires because iter-3.5c skips store). Document the failure.
2. **Implement**: 4-file change per 1.4a.
3. **Re-run test**: must PASS post-implementation.
4. **Regression**: 2804/0/13 bin tests + iter-3 + iter-5 + iter-6 + iter-7 + iter-8 all PASS.

#### 1.4c Capacity check policy at the probe site

The engine's per-layer capacity check at `engine.rs:6220-6225` (streaming probe) and equivalent non-streaming site already does:
```rust
let required_cap = if layer_is_ring { model_sw } else { new_linear };
arc.capacity >= required_cap
```

When iter-3.6 is on AND layer is sliding, the cached buffer was sized to `max(sw, prev_prompt_len + prev_max_decode)`. The new request's required cap is `max(sw, new_prompt_len + new_max_decode)`. The check `arc.capacity >= required_cap` still works as-is — but the `required_cap` formula must change for sliding layers (currently `model_sw`, should become `max(model_sw, new_prompt_len + max_decode)` when env-on).

#### 1.4d Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| `mask_type=2 + sliding_window=sw` produces different output than `mask_type=1 + ring` | HIGH (kernel-level invariant; silent miscompute if wrong) | Falsifier test asserts byte-identity to control (no resume) — kernel divergence surfaces immediately. Plus a unit test in mlx-native that compares mask_type=1+ring vs mask_type=2+linear on a synthetic shape. |
| Linear sliding buffer growth blows memory at long prompts | LOW on M5 Max 128 GB | Per-layer K+V at N=20K: 8 × 20480 × 256 × 2 × 2 = 168 MB. ~30 layers ⇒ ~5 GB. Headroom is ~95 GB after model load. |
| Capacity-check formula bug in probe site causes spurious cache misses or installs | MED | Test sweep at multiple K/N pairs (mirror iter-5 pattern). |
| Multi-turn growth: turn N+1's prompt > turn N's snapshot capacity | MED | Snapshot capacity sized to `max_decode_tokens` headroom. If N grows past prev cap + headroom, the next turn misses cache (acceptable degradation). |

### 1.5 Iter-3.6 verdict

**Feasible at ~255 LOC across 5 files. Walk discipline mandatory. Risk is bounded by falsifier test + existing iter-3 audit precedent.**

---

## Section 2 — Phase B-hybrid (Qwen 3.6 LCP) findings

### 2.1 Architecture: Qwen 3.6 is HYBRID

From `forward_gpu.rs:2831`: "48 of 64 layers in Qwen3.6 27B are linear-attn DeltaNet."

| Layer type | Count (Qwen 3.6 27B) | KV mechanism | Slot-addressable? |
|---|---|---|---|
| **Full-attn** | 16 / 64 | Linear K/V buffer `[D_k, n_kv_heads, max_seq_len, n_seqs]` | YES — slot=position |
| **Linear-attn (DeltaNet)** | 48 / 64 | conv_state `[K-1, channels]` (last K-1=3 rows of QKV) + recurrent state `[D_k, D_v, num_v_heads]` | NO — recurrent state is position-dependent |

`HybridKvCache::snapshot` (`kv_cache.rs:529-564`) deep-copies all of these. `HybridKvCache::restore_from` (`kv_cache.rs:582-633`) restores by memcpy.

### 2.2 The recurrent-state problem

For full-attn layers, K/V at position p lives at slot p in a linear buffer. Restore at K < N just truncates `current_len` and the tail bytes are unread. SAME-SHAPE as Gemma's global layers — directly slot-addressable.

For DeltaNet layers, the recurrent state at end-of-prefill of Q represents the SSM state AT POSITION |Q|. The SSM dynamics:
```
h_{t+1} = f(h_t, x_t, gate_t)  (recurrence; gated_delta_rule kernel)
```
The state at position K in prompt P (where LCP(Q, P) ≥ K) is computed via the same recurrence on tokens P[0..K). Tokens P[0..K) == Q[0..K) (LCP). So h_K computed from P[0..K) == h_K computed from Q[0..K). The state IS the same — IF we have it.

Today's snapshot captures only h_|Q| (end-of-prefill). It does NOT contain h_K for K < |Q|.

### 2.3 Three workable options

#### Option 1 — Full-attn-only resume (RECOMMENDED for v1)

- LCP probe runs as today.
- On hit at K < |Q|: restore full-attn layers' K/V at slot K (truncate current_len to K). Re-prefill DeltaNet layers from scratch (K-1 tokens of conv warmup + K tokens of SSM recurrence).
- Snapshot at end-of-prefill captures full-attn + DeltaNet end-states; on partial resume, only full-attn restore is used.
- Speedup: full-attn layers (16/64) get LCP-style speedup; DeltaNet layers (48/64) re-prefill. Net speedup is modest — depends on per-layer FLOPs ratio. DeltaNet is ~50% of total prefill FLOPs typically; saving 25% (16/64 layers) → ~12-15% speedup at K=80% LCP.

| Pro | Con |
|---|---|
| Simple architecture; mirrors iter-3 closely | Modest speedup vs Option 2 |
| HybridKvCache::snapshot already provides the substrate | Doesn't unlock the full /cfa-shape gain |
| Falsifier test mirrors iter-3 exactly | DeltaNet still re-prefills entirely |

LOC estimate: ~400-600 (engine.rs + qwen35 prefill hook + tests + dossier).

#### Option 2 — Sparse DeltaNet checkpoints

- Snapshot the SSM recurrent state at every C tokens (chunk boundary; e.g. C=1024).
- On LCP hit at K, find nearest snapshot at K' = floor(K / C) × C. Restore SSM state there. Re-prefill DeltaNet for [K', K).
- Memory: 48 layers × (N/C) checkpoints × ~2 MB per checkpoint. For N=8192, C=1024 ⇒ 48 × 8 × 2 MB = 768 MB. Acceptable on 128 GB.
- Compute: re-prefill at most C-1 tokens of DeltaNet per hit. Saves ~93% of DeltaNet prefill at K=4K, C=1024.

| Pro | Con |
|---|---|
| Near-full speedup (full-attn LCP + DeltaNet partial-LCP) | More invasive: requires hook in `chunk_gated_delta_rule_fwd` to emit checkpoints |
| Memory cost manageable on 128 GB | Re-prefill of last partial chunk has small FLOP overhead |
| Composes cleanly with HybridKvCache existing snapshot path | Two snapshot pathways (full-attn end-state + DeltaNet checkpoints) |

LOC estimate: ~700-1000 (mlx-native checkpoint hook + engine.rs + qwen35 prefill + tests + dossier).

#### Option 3 — Full-equality replay only (= Phase E option (b), already shipped)

- Already in production via HybridPromptCache.
- Hits only when prompt P == prompt Q exactly (no partial-prefix).
- Doesn't extend to partial — by design.

### 2.4 Recommended Phase B-hybrid v1 plan: Option 1

Reasons:
1. **Walk discipline**: ship the simplest version first; measure; decide on Option 2 lift with real speedup data.
2. **Modest speedup is still a speedup**: 12-15% on multi-turn chat is meaningful.
3. **Architecture matches iter-3 pattern**: full-attn-only resume on Qwen35 mirrors Gemma's dense-kv resume directly. Codex audit precedent applies.
4. **Snapshot substrate exists**: HybridKvCache::snapshot already does what we need. No mlx-native kernel changes.
5. **Risk-bounded**: DeltaNet layers always re-prefill (no SSM-state reuse), so silent miscompute via state corruption is impossible. The only correctness path is the full-attn restore — same shape as Gemma iter-3.

### 2.5 Phase B-hybrid Option 1 implementation plan (high-level)

#### 2.5a Files to touch

| File | Change | LOC est |
|---|---|---|
| `src/serve/api/engine.rs` | Add `lcp_registry: LcpRegistry<HybridKvSnapshot>` field to `Qwen35LoadedModel` (mirror Gemma's at line 1102); add probe site in qwen35 generate path (mirror engine.rs:3838-3940); add resume gate; add restore-installs in worker arms (replace the Err returns at 3152, 3205) | ~250 |
| `src/inference/models/qwen35/forward_gpu.rs` | Add `forward_prefill_with_resume(restored_lcp_full_attn_only: Option<usize>)` entry point. On `Some(K)`: skip full-attn KV writes for tokens [0..K); DeltaNet layers run normally for all tokens. Emit end-of-prefill snapshot to `dense_kvs_snapshot_for_lcp` equivalent. | ~150 |
| `src/inference/models/qwen35/kv_cache.rs` | Possibly extend HybridKvCacheSnapshot with `current_len_at_K` for partial-restore; or compute at restore time | ~30 |
| `tests/lcp_partial_prefill_byte_identity.rs` | New `phase_b_hybrid_qwen35_partial_prefix_byte_identity` test (mirror iter-3) | ~150 |
| Replace `iter7_qwen35_hybrid_lcp_deferred_marker` | Rename to `iter7_qwen35_lift_landed` per the FUTURE LIFT INSTRUCTIONS in test header | ~5 |
| Total | | **~585 LOC** |

#### 2.5b Walk discipline

1. **Test first**: `phase_b_hybrid_qwen35_partial_prefix_byte_identity` — two-server byte-identity using Qwen 3.6 27B-DWQ46 (`/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf`). Server A `HF2Q_KV_LCP_RESUME=1` Qwen35; server B control. Falsifier: bytes diverge OR `lcp_detected_total < 1`.
2. **Implement** Option 1 plan above.
3. **Re-run test**: PASS.
4. **Regression**: 2804/0/13 bin tests + Gemma iter-3/5/7/8 all PASS.
5. **Codex Phase-2b audit** mandatory (mirror iter-7 precedent).

#### 2.5c Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Re-prefilling DeltaNet from scratch produces non-byte-identical output to fresh prefill | HIGH | The DeltaNet recurrence is deterministic at T=0; same tokens produce same state. Falsifier test asserts byte-identity. |
| Full-attn `current_len` truncation at slot K leaves stale bytes at slots [K..|Q|) which kernel might read | HIGH | Verify the full-attn kernel reads `[0..current_len)` only. If it reads beyond, zero-fill at restore time. (Per `kv_cache.rs:582-633` restore_from already memcpys all bytes; current_len is the read bound.) |
| Qwen3.5/3.6 has no sliding-window full-attn (verify) | LOW (just needs verification) | Read the layer config. If full-attn is global-only (linear, no sw), iter-3.5c-style guard is N/A. |
| HybridPromptCache lookup races with LCP probe (both hit between Q and P) | LOW | LCP probe runs AFTER HybridPromptCache::lookup miss (mirror Gemma's `engine.rs:3830-3836`). Same ordering. |

---

## Section 3 — Recommended phasing

### 3.1 Sequencing options

**Option A**: iter-3.6 first (Gemma long-prompt). Simple, well-understood, clean falsifier. ~255 LOC. ~1-2 days to ship + audit.

**Option B**: Phase B-hybrid Option 1 first (Qwen 3.6 full-attn-only). Doubles family coverage. ~585 LOC. ~3-5 days to ship + audit.

**Option C**: Both in parallel via /cfa swarms. ~840 LOC combined; ~4-6 days wall-time.

### 3.2 Recommendation

**Start with Option A (iter-3.6 first), then Option B (Phase B-hybrid).**

Reasoning:
- iter-3.6 is smaller, lower-risk, and reuses ALL the iter-3 audit precedent + infrastructure (LcpRegistry, snapshot path, probe sites).
- It surfaces the "linear sliding buffer" pattern that Phase B-hybrid full-attn layers can borrow.
- Phase B-hybrid has independent architectural risk (DeltaNet re-prefill correctness) that needs its own falsifier — landing it AFTER iter-3.6 means we have a confidence baseline.
- Codex Phase-2b audit cycles work well one-at-a-time; parallel audits dilute attention.
- Total wall-time A→B is similar to parallel C, but with lower coordination cost.

### 3.3 What stays deferred

- **iter-3.7 (DeltaNet sparse checkpoints)** — Phase B-hybrid Option 2 lift. Defer until Option 1 ships and we have measured speedup for "is the extra LOC worth the extra speedup" sanity check.
- **iter-4 (env-flip default-ON)** — gated on operator soak (≥ 2 weeks under HF2Q_KV_LCP_RESUME=1). User-discretion.

---

## Appendix A — Verified code references

| Reference | Verified |
|---|---|
| `forward_mlx.rs:2632-2635` ring/mask Chesterton's fence | YES — read verbatim |
| `flash_attn_vec.metal:166-170` mask_type=2 implementation | YES — read verbatim |
| `forward_prefill.rs:1095-1099` per-token write_slot logic | YES — read verbatim |
| `forward_prefill.rs:1136-1167` TQ-packed write site | YES — read verbatim |
| `forward_prefill.rs:1249-1259` flash_attn_vec params | YES — read verbatim |
| `forward_mlx.rs:2495-2499` decode use_dense_sdpa branch | YES — read verbatim |
| `engine.rs:4516, 7027` iter-3.5c guards | YES — read verbatim |
| `kv_cache.rs:107-122` HybridKvCache struct | YES — read verbatim |
| `kv_cache.rs:152-166` HybridKvCacheSnapshot fields | YES — read verbatim |
| `kv_cache.rs:529-633` snapshot/restore_from impls | YES — read verbatim |
| `forward_gpu.rs:2831` "48 of 64 layers DeltaNet" | YES — read verbatim |
| `gpu_delta_net.rs:62-138` chunk_gated_delta_rule import + chunk_size | YES — read verbatim |

## Appendix B — User decisions (CONFIRMED 2026-05-05)

User responses to the open questions in the original dossier draft:

1. **Phasing**: A (iter-3.6) then B (Phase B-hybrid) **sequentially**.
2. **Phase B-hybrid scope**: **all options shipped in v1** — Option 1 (full-attn-only resume) AND Option 2 (sparse DeltaNet chunk-boundary checkpoints). User: "implement all of the ideas we already know we want. That's all v1 in my mind."

## Appendix C — Peer-implementation cross-validation (added 2026-05-05 after goalie research)

User directed deeper research via goalie + peer repos. Findings reinforce the Section 2 design:

### C.1 Marconi (arXiv 2506.xxxxx) — canonical reference for SSM partial-resume

The fundamental asymmetry: **Transformer KV caches** can be partitioned and trimmed to represent arbitrary subsequences (slot-addressable). **SSM states cannot be rolled back** to represent a prefix — they must be computed sequentially. Marconi's solution: checkpoint SSM state at chunk boundaries during prefill; on LCP hit, restore from the nearest aligned checkpoint and re-prefill the partial chunk after the boundary. This is exactly Option 2.

### C.2 vLLM — `mamba_cache_mode = "align"` (verified at `/opt/vllm/vllm/v1/core/single_type_kv_cache_manager.py:794-906`)

vLLM ships chunk-aligned prefix caching for hybrid (Mamba+Attention) models. The `MambaManager.find_longest_cache_hit` searches blocks right-to-left and only accepts a hit when `(i + 1) * block_size % alignment_tokens == 0` — i.e. the hit length is a multiple of the SSM chunk size. The block-manager aligns prefix-cache `block_size` to the SSM chunk size so hits land on safe boundaries.

Key lines:
```python
if (block_size != alignment_tokens
    and (i + 1) * block_size % alignment_tokens != 0):
    continue  # skip non-aligned candidate
```

This is **block-manager-level alignment** (not kernel-level). The SSM state is captured at chunk-aligned positions and cached as a "block." On hit, the SSM state for that block is restored.

### C.3 llama.cpp — `delta-net-base.cpp` chunked structure (verified at `/opt/llama.cpp/src/models/delta-net-base.cpp:60-63`)

```cpp
const int CS = kda ? 16 : 64;  // chunk size: 16 for KDA, 64 otherwise
const int pad = (CS - n_tokens % CS) % CS;
const int n_chunks = (n_tokens + pad) / CS;
```

The recurrent state `last_recurrent_state` propagates BETWEEN chunks: each chunk's final state becomes the next chunk's initial state. Checkpointing at chunk boundaries = saving the state passed between chunks. Natural alignment.

### C.4 SSM state size is O(1) per layer (CRITICAL for memory math)

Per the goalie synthesis: "SSM states are fixed-size hidden states (constant memory, independent of N)". This means **checkpoints are cheap per checkpoint** — only memory grows linearly with checkpoint count, not chunk content.

For Qwen 3.6 27B (48 DeltaNet layers):
- Per-layer recurrent state: `[D_k × D_v × num_v_heads]` = `[128 × 128 × 32]` × 4 bytes = 2 MB
- Per-layer conv state: `[K-1 × channels]` = `[3 × 8192]` × 4 bytes = 96 KB (negligible)
- Per checkpoint (all 48 DeltaNet layers): 48 × ~2 MB = **96 MB**

At chunk-aligned checkpoints every 1024 tokens for an N=8192 prompt: 8 checkpoints × 96 MB = **768 MB per cached entry**. With registry capacity=1, this is the upper bound on additional resident memory.

For sparse checkpointing every 256 tokens: 32 × 96 MB = 3 GB per cached entry. Fits in M5 Max 128 GB easily.

### C.5 Reported speedups

vLLM and SGLang report 2-4× speedup on long contexts with chunk-aligned partial-prefill resume. For Qwen 3.6 specifically (48/64 layers DeltaNet), saving the DeltaNet computation on a 4K-token shared prefix at chunk_size=1024 saves ~75% of DeltaNet prefill (3 chunks resumed; last partial chunk re-prefilled). Combined with full-attn LCP, total speedup approaches Gemma-class numbers (5-10×) on multi-turn chat.

### C.6 Chunk size for Qwen 3.6 in hf2q

Need to verify the actual chunk size used by hf2q's `chunk_gated_delta_rule_fwd`. From `/opt/hf2q/src/inference/models/qwen35/gpu_delta_net.rs:110`: `mlx_native::ops::chunk_gated_delta_rule::FIXED_BT = 64`. Confirmed: chunk_size CS = 64.

For LCP alignment, our checkpoint granularity should be a MULTIPLE of 64 (e.g. 1024 = 16 chunks, 512 = 8 chunks, 256 = 4 chunks). Operator-tunable via env var `HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE`, default 1024.

## Appendix D — Refined v1 scope (post-user-confirmation 2026-05-05)

### D.1 Phase A: iter-3.6 (Gemma long-prompt LCP) — sequential first

| File | Change | LOC |
|---|---|---|
| `src/debug/investigation_env.rs` | `kv_lcp_long_resume: bool` field | ~5 |
| `src/serve/forward_prefill.rs` | Sliding-layer alloc + write_slot + flash_attn_vec params + snapshot guard | ~80 |
| `src/serve/forward_mlx.rs` | Decode flash_attn_vec params for sliding | ~30 |
| `src/serve/api/engine.rs` | Engine guards lift; capacity check formula update | ~30 |
| `tests/lcp_partial_prefill_byte_identity.rs` | Falsifier test | ~120 |
| Total | | **~265 LOC** |

### D.2 Phase B: Phase B-hybrid full v1 = Option 1 + Option 2

**B.1 — Option 1 (full-attn-only resume) — ship first**

| File | Change | LOC |
|---|---|---|
| `src/serve/api/engine.rs` | Qwen35LoadedModel `lcp_registry` field + probe sites + restore replacing Err returns | ~250 |
| `src/inference/models/qwen35/forward_gpu.rs` | `forward_prefill_with_resume(restored_full_attn_lcp_only: Option<usize>)` entry point | ~150 |
| `src/inference/models/qwen35/kv_cache.rs` | Partial-restore current_len truncation; HybridKvCacheSnapshot variant for LCP | ~30 |
| `tests/lcp_qwen35_byte_identity.rs` | New test: `phase_b_hybrid_qwen35_partial_prefix_byte_identity` | ~150 |
| Rename `iter7_qwen35_hybrid_lcp_deferred_marker` → `phase_b_hybrid_landed_marker` | Per FUTURE LIFT INSTRUCTIONS | ~5 |
| Total | | **~585 LOC** |

**B.2 — Option 2 (DeltaNet chunk-boundary checkpoints) — extends B.1**

| File | Change | LOC |
|---|---|---|
| `src/inference/models/qwen35/kv_cache.rs` | `HybridKvCacheCheckpoint` per-chunk-boundary structure: snapshot `(conv_state, recurrent_state)` per layer at every Cth chunk | ~80 |
| `src/inference/models/qwen35/forward_gpu.rs` | Hook in chunk_gated_delta_rule loop: emit checkpoint after every C chunks (default C=16 ⇒ 1024-token stride) | ~100 |
| `src/serve/api/engine.rs` | LCP resume gate: align K to floor(K / stride) * stride; restore DeltaNet checkpoints at K_aligned; partial re-prefill of [K_aligned..K) | ~100 |
| `src/debug/investigation_env.rs` | `kv_lcp_deltanet_checkpoint_stride: usize` (default 1024); `kv_lcp_long_resume: bool` already exists from Phase A | ~5 |
| `tests/lcp_qwen35_byte_identity.rs` | Sweep test: 5 K fractions × {aligned, mid-chunk} = 10 byte-identity assertions | ~150 |
| Total | | **~435 LOC** |

**Phase B total**: 585 + 435 = **~1020 LOC**

### D.3 Cumulative v1 scope

iter-3.6 + Phase B.1 + Phase B.2 = ~265 + 585 + 435 = **~1285 LOC** across 6-8 source files + 3 test files.

Walk discipline mandatory at each phase. Codex Phase-2b audit mandatory at end of each phase before merge to main.

### D.4 Sequencing per user direction

1. **Phase A (iter-3.6)** — this loop iter and the next.
2. **Phase B.1 (Qwen35 full-attn LCP)** — after Phase A merges + Codex audit clears.
3. **Phase B.2 (Qwen35 DeltaNet checkpoints)** — after Phase B.1 ships and we measure speedup.

Total estimated wall-time: 5-8 days across ~10-15 loop iters. The user accepts this cost ("we have plenty of time to do it right").

### D.5 — Phase B.2a chunked-prefill blocker (2026-05-05)

**Finding**: the foundational invariant for B.2 — that running prefill in chunks (with `kv_cache` state propagation between calls) produces output byte-identical to a single monolithic call — DOES NOT HOLD on Qwen 3.6 27B-DWQ46 today.

Test: `tests/lcp_qwen35_chunked_prefill.rs::phase_b2a_chunked_vs_monolithic_byte_identity` with stride=64, prompt tokenizing to 108 tokens. Result: chunked decode = 41 bytes; monolithic decode = 47 bytes; diverge at byte offset 4.

**Hypothesis (prime suspect, FALSIFIED 2026-05-05)**: the DeltaNet kernel dispatcher uses different code paths based on `seq_len > 64 && seq_len % 64 == 0` (chunk-pipeline at `chunk_gated_delta_rule_fwd`) vs the fallback (autoregressive `gated_delta_net_decode`). Chunked prefill at stride=64 produces partial-tail chunks (seq_len ∈ [1, 63]) which take the autoregressive path; the cumulative state across multiple kernel-path-switching calls differs from a single monolithic call's state.

**Diagnosis 2026-05-05 (stride=54 isolation)**: ran chunked-prefill with stride=54 against the same 108-token prompt. With stride=54, BOTH chunks (54 + 54 tokens) have seq_len < 64, forcing the autoregressive path. Monolithic seq_len=108 also takes the autoregressive path (108 % 64 != 0). So both paths now use the **same kernel route** (autoregressive throughout).

Result: server A (chunked, both autoregressive) decoded gibberish — the model echoed the chat template (`<|im_start|>assistant\n\n<|im_start|>user\n\n`), implying the post-prefill state was broken. Server B (monolithic, autoregressive) decoded coherent text.

**FALSIFIED**: the prime-suspect kernel-path-mismatch hypothesis is wrong. The bug exists even when chunked + monolithic take identical kernel routes.

**ACTUAL ROOT CAUSE**: state propagation BETWEEN consecutive autoregressive `forward_gpu_last_logits` calls is broken. Two calls of [54-token autoregressive prefill] ≠ one call of [108-token autoregressive prefill] when run on the same `kv_cache`, even though both use the same kernel.

Possible deeper causes (not yet investigated):
- DeltaNet recurrent_state ping-pong (`HybridKvCache::swap_recurrent`) may not handle multi-call state hand-off correctly. The autoregressive path uses scratch + active swap internally; the swap state at end-of-call-1 may not match what call-2 expects.
- DeltaNet `conv_state` (last K-1=3 token rows of QKV) may not be correctly persisted between calls. The chunk-pipeline path writes a final conv_state; the autoregressive per-token loop maintains conv_state via roll-and-append. If end-of-call-1 leaves conv_state in a non-canonical layout, call-2 misreads it.
- An internal arena or scratch buffer within forward_gpu_impl is per-call but should persist across calls.
- Some kv_cache invariant (e.g. expected current_len[0] alignment with conv_state contents) is violated by partial-prefill end-state.

**B.2 status**: BLOCKED pending deeper root-cause. Mitigations:
1. ~~Constrain chunked to divisible strides~~ — falsified by stride=54 experiment; bug isn't path-mismatch.
2. **Pinpoint the per-state divergence**: ✅ DONE via diagnostic test
   `forward_gpu::tests::phase_b2a_chunked_kv_cache_divergence_diagnostic`
   (commit `e8910a2`). On Qwen 3.6 27B-DWQ46 with N=24, K=12:
   * full_attn[0]: NO divergence (first layer; reads embeddings, not upstream KV).
   * full_attn[1..16]: diverge at byte 12288 (= token 12 = split point K). Tokens [0..K) match; tokens [K..N) differ.
   * ALL 48 linear_attn layers: conv_state + recurrent diverge from byte 0.
3. Fix the per-state bug. **Diagnosis: not small.** The cascade pattern (all DeltaNet layers diverge end-to-end; downstream full-attn layers diverge only on tokens [K..N)) implies the broken hand-off is in forward_gpu_impl's interaction with a warm kv_cache. The kernel itself is correct (verified via `gated_delta_net_decode.metal:145 for(t=0;t<n_tokens)` autoregressive loop reads state_in / writes state_out per token; final state_out captures the seq_len-th state). Likely culprits (further investigation needed):
   - Some forward_gpu_impl per-call setup (decode_pool reset, arena alloc, conv_state initialization) implicitly assumes fresh kv_cache and silently mis-handles a warm one.
   - The `dn_prefill_arena` per-call lifecycle leaks state across calls when used in the chunked-prefill pattern.
   - A subtle conv_state-vs-conv_state_scratch swap-parity invariant that holds for the "single monolithic call" use case but breaks across calls.

**Realistic scope for fixing**: each candidate culprit requires hours of code-reading + targeted tests. Multi-iter effort. Without dedicated effort, Phase B.2 (chunked-prefill substrate for sparse DeltaNet checkpoints) is infeasible.

**Phase B fallback architecture**: if the chunked-prefill path can't be made byte-identical to monolithic, the alternatives are:
  - **B.2-end-only**: store ONLY the end-of-prefill snapshot (= full-equality replay = Phase E option (b), already shipped via HybridPromptCache). Provides full-prompt reuse, NOT partial-prefix.
  - **B.2-kernel-hooks**: modify mlx-native kernels to emit checkpoints from within a single monolithic call. Multi-week scope; out of immediate reach.
  - **B.2-chunked-fixed**: fix the chunked-prefill state hand-off (this iter's blocker). Investigation required.

Given the cost trade-off (multi-iter blocker for an N×LCP_speedup that even Option 2 only delivers at ~75 % of DeltaNet bandwidth), the user should decide whether to invest more loop iters here or accept Qwen 3.5/3.6 LCP at full-equality-only and ship.

**Default impact**: zero. The chunked path is gated on `HF2Q_KV_LCP_CHUNKED_PREFILL=1` (default OFF). Default `cargo test` passes 2814/0/3. The falsifier test runs only under `HF2Q_KV_PERSIST_PHASE_D=1 + HF2Q_KV_PERSIST_QWEN35_E2E_MODEL_PATH=<gguf>` (explicit operator opt-in).

The falsifier IS KEPT failing intentionally (per mantra "code + test == truth") so the bug stays visible until rooted out.
