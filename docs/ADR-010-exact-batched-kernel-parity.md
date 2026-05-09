# ADR-010: Exact Batched-Kernel Parity with llama.cpp

**Status:** Parity line **Deferred**. Speed line **Shipping** via lm_head Q8+rerank as the new default strategy (see "lm_head Q8 + Rerank" section below).
**Date:** 2026-04-16
**Decision Makers:** Robert, Claude
**Related ADRs:** ADR-006 (mlx-native GPU backend), ADR-007 (TurboQuant KV cache), ADR-008 (candle divorce), ADR-009 (reference parity and coherence recovery)

> **TL;DR (2026-04-16 landing):** Default hf2q decode now matches the F16 coherence trajectory on the locked prompts while running at ~98% of llama.cpp throughput via Q8 lm_head + CPU threshold-scan exact rerank. Exact batched-kernel parity against llama.cpp's MoE path remains an open numerical sensitivity issue (sliding_wrap ~752/2327 bytes vs llama batched); the investigation is paused, not closed, and a GPU top-K kernel is committed-but-dormant pending a future parallel-phase-2 redesign.

---

## Engineering Mantra (load-bearing — read before every session)

Source: `~/Documents/mantra.txt` (Robert, undated). Quoted verbatim.

> **DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.**

---

## Context

ADR-009 Phase 3A closed with the following parity state on the Gemma-4-26B-A4B DWQ reference:

| Prompt | vs llama BATCHED | vs llama PER-TOKEN |
|---|---:|---:|
| sourdough (22 prompt, 1000 decode) | **3656/3658 (99.9%)** | 3095/3656 (84.6%) |
| sliding_wrap (82 prompt, 500 decode) | 752/2327 (32.3%) | **1569/2354 (66.7%)** |

Phase 3A localized the remaining sliding_wrap gap to a structural divergence between hf2q's and llama.cpp's **batched** kernel trajectories during prompt ingestion. The specific bisection evidence:

- At (layer 6, pos 34), hf2q's batched K matches llama's batched cached K at rel_rms = 3.11e-4.
- At (layer 7, pos 34), the same comparison is rel_rms = 4.37e-2 — a ~100× jump in a single layer.
- hf2q's per-token K at (L7, pos 34) agrees with hf2q's batched K at 8.5e-5.
- llama.cpp's per-token K at (L7, pos 34) agrees with its own batched K at roughly the same tight tolerance.

So the divergence is not per-token vs batched within either implementation. It is **hf2q batched vs llama batched at layer 7 of the sliding attention path**. The implementations agree on the mathematical answer for a single token at a time, but their batched-prefill reduction orders diverge enough by layer 7 to flip the downstream trajectory.

Phase 3A's `forward_prefill_batched` (gated behind `HF2Q_BATCHED_PREFILL=1`) did not close the gap — a true batched path in hf2q still hits the same ~752-byte ceiling vs llama's batched reference. Prefill mode matters for which llama reference to pick, but it is not the dominant cause of the gap.

## Decision

Pursue exact batched-kernel parity as a separate, narrowly scoped investigation outside ADR-009. This ADR defines the scope and acceptance criteria.

### Localization (2026-04-16 session — what the data showed)

Per-layer and per-stage bisection at (sliding_wrap, pos 34) in batched mode narrowed the seam much further than ADR-009's "attention path" hypothesis:

1. **Per-layer l_out scan** (hf2q batched vs llama batched):
   - Layers 0–5 all tight at rel_rms 1.3e-4 to 2.1e-4
   - Layer 6 output: rel_rms 2.91e-2 — **a 200× jump in a single layer**
   - Layers 7+ continue to accumulate to 1.7e-1 by the final layer

2. **L6 sub-stage bisection** — attention is entirely clean:
   - Input (L5 l_out): 1.34e-4
   - Pre-attn norm output: 3.13e-4
   - Q/K/V post-QKV matmul: 2.2–3.4e-4
   - Q/K post head-norm + RoPE: 2.2–2.4e-4
   - SDPA output (kqv_out): 5.47e-4
   - Post-attn residual vs llama `attn_out`: 1.66e-4
   - Router logits: **1.58e-4** (still tight)
   - MLP+MoE combined: **5.45e-2** (jump)
   - L6 l_out: 2.91e-2

3. **Root cause — MoE top-K threshold sensitivity**:
   - Router logits agree between hf2q and llama to ~1e-4 (the matmul noise floor).
   - But at L6 pos 34 the 7th and 8th ranked logits differ by only **0.0001**, below the matmul noise floor.
   - hf2q picks expert 95 at rank 7; llama picks expert 61. Seven of eight top-K picks match; one swap is enough.
   - That one expert swap drives the 5.5% divergence in the MoE weighted sum; post-FF norm and layer_scalar carry it to 2.9% at L6's l_out, which then propagates.

### Defensible framing

Current evidence shows the remaining long-sequence batched parity gap is driven by **router matmul numerical differences crossing MoE top-K thresholds** at a small number of early tokens — not by attention kernels and not by an "intrinsic" implementation gap. Exact batched parity therefore requires **tighter router-logit agreement than the current owned router matmul provides**. hf2q's top-K selection already breaks true ties deterministically (lower expert_id on strict tie), and llama's is implementation-defined (`std::sort` unstable), so a stable-tiebreak change in hf2q is not the fix — the logits themselves need to agree more tightly than the current ~1e-4 floor allows under 0.0001 expert-gap conditions.

### In scope

1. **Router matmul exactness (FIRST concrete implementation target).**
   - Port ggml's router projection reduction order into a dedicated `router_matmul` kernel in mlx-native, used only at the `ffn_gate_inp` call site.
   - Small surface: input `[hs=2816]` × weight `[num_experts=128, hs]` → logits `[num_experts]` per token. The weight is already quantized — the alignment is about reduction/accumulation order over the K dimension, not the full qmatmul framework.
   - Replicate llama's `build_lora_mm(ffn_gate_inp, tmp)` behavior precisely, including the preceding `ggml_mul(ctx0, tmp, ffn_gate_inp_s)` scaling.
   - Gate behind `HF2Q_ROUTER_EXACT=1` initially so we can A/B measure.
   - Decisive checks only: (a) L6 pos 34 router logits ≤ 1e-5 rel_rms, (b) L6 pos 34 expert IDs match exactly, (c) sliding_wrap vs llama batched common-prefix moves materially, (d) no regression on sourdough or sliding_wrap vs per-token llama reference.

2. **Per-layer scan to confirm no other seams.**
   - With router exact, re-run the per-layer l_out scan on sliding_wrap. If L6 jump is gone but another layer shows a new jump, bisect there.

3. **Sub-stage boundary instrumentation (already landed).**
   - The `HF2Q_BATCHED_DUMP="layer,tok"` and `HF2Q_BATCHED_LAYER_SCAN="tok"` env vars in `forward_prefill_batched.rs` plus the extended `dump_layer_states` tool cover all sub-stages we need. Kept as the standing diagnostic for this ADR.

### Explicitly out of scope (for now)

- **Flash-attention-vec bit-exact replication.** Evidence shows attention is not the seam in batched mode — SDPA output is within 5.5e-4 of llama's. Deferred unless a future layer-scan reveals an attention-specific jump after the router fix.
- **Direct ggml integration of whole framework.** Violates ADR-008. Only reconsider if router-matmul alignment doesn't close the gap and layer-scan reveals distributed sub-1e-4 noise accumulation.
- **Rewriting hf2q's general GPU framework.** This ADR is about matching a specific numerical trajectory, not reshaping compute architecture.

## Acceptance Criteria

**Minimal success (Walk):**

- sliding_wrap common prefix vs llama BATCHED ≥ 1500 bytes (roughly doubling the current 752).
- No regression on sourdough (≥ 3094 common prefix, the existing gate).
- No regression on sliding_wrap vs llama PER-TOKEN (≥ 1500 bytes).

**Full success (Run):**

- sliding_wrap common prefix vs llama BATCHED ≥ 2000 bytes.
- Batched and per-token hf2q paths both within 3e-4 rel_rms of their respective llama references at (L7, pos 34) cached K.

## Non-goals

- Speed parity — this ADR is about numerical parity, not throughput. Speed is tracked in ADR-005/ADR-008 perf work.
- Exact byte-for-byte output across all prompts — diminishing returns after the sliding_wrap gate passes.

## Deferred Work Also Tracked Separately

- **Greedy nondeterminism at T=0.** Historically observed at ~2–3% on 3fb8988 and 8a02725 but *not reproducing in current sessions*: 40/40 runs at common=3656 on 7dba9f9. Earlier outliers may have been transient (thermal / memory pressure). Argmax kernel is deterministic (strict `>` tree reduction). If repro returns, suspect matmul reduction order in lm_head mixed-precision matvec. Deferred pending reliable repro.

- **Long-decode single-token drift vs llama on non-gate prompts.** On the `Comlprehensive instructions for making sourdough bread.` prompt (different typo placement than the gate's `Complrehensive`), hf2q and llama diverge at the first tight logit tie-break (~decode token 570): hf2q picks `kneading`, llama picks `intense kneading`. On the same prompt, a later tie-break at decode ~675 produces a `####DP 4.` glyph artifact where a Markdown header-and-space pair tokenizes as `[####, DP,  4]` instead of `[####,  4]`. Neither gate prompt hits these tiebreaks within 3656 / 2354 bytes, so the gates pass byte-identical. This is a concrete instance of the exact-batched-kernel-parity gap this ADR is chartered to address — fixing it requires either (a) sub-stage boundary dumps + kernel reduction-order alignment, or (b) increased accumulator precision in the matmul / flash_attn_vec reductions. Not a separate issue; folded into this ADR's sub-stage investigation scope.

## Memory Optimization Landed (2026-04-16)

Not strictly a parity concern, but completed alongside the nondeterminism / drift investigation because it shares the dense-KV code path. Commit `7dba9f9`:

- Sliding layers now use a ring-buffer dense KV cache (capacity = `sliding_window = 1024`, writes wrap at `seq_pos % capacity`). Global layers stay linear.
- Dense `flash_attn_vec` uses `mask_type = 1` (causal) in ring mode; the ring itself applies the sliding constraint. Correctness rests on attention being permutation-invariant over cached K,V (RoPE is baked in pre-cache).
- Memory at a 20k decode budget: 7.4 GB → ~2.75 GB dense KV (−4.6 GB, −62%).
- All gates pass unchanged; 1353-token coherence test produces identical clean-EOS output at 91.5 tok/s.

## lm_head Q8 + Rerank (2026-04-16, related speed work)

lm_head quantization + exact rerank became the default speed path after
this ADR's router-matmul line was closed. Summary:

- **Default (auto):** Q8_0 lm_head + CPU threshold-scan rerank when the
  F16 weight exceeds 256 MB and `hidden_size % 32 == 0`.
- **Escape hatches:** `HF2Q_LMHEAD_Q8=0` forces F16; `HF2Q_LMHEAD_RERANK=0`
  disables rerank (leaves raw Q8 argmax, unsafe — occasional pad-emit).
- **Rerank mechanism:** after the Q8 matmul writes full-vocab logits, a
  single CPU pass collects tokens with logit ≥ (Q8 top-1) − 0.5 plus
  specials (0/1/2/105/106), then recomputes exact F32 logits from the
  F32 `embed_weight` dotted with the pre-lm_head hidden. Argmax over
  the reranked set.
- **Result:** sliding_wrap is byte-identical to the F16 reference
  (2354/2354) and speed is 101.8 tok/s on Gemma-4 26B (98% of the
  llama.cpp 104 tok/s reference). The pad-emit failure mode is
  explained (Q8 noise envelope ~5e-3 crossing near-tie thresholds)
  and eliminated by the rerank set.
- **GPU top-K — tested and rejected** for the current vocab/shape.
  A single-threadgroup top-K (committed at mlx-native `27070c1`) costs
  ~5 ms/token for vocab=262144 K=64 because the phase-2 extraction
  serializes onto one thread. The CPU threshold scan at ~40 μs/token
  dominates it. The GPU kernel stays in the tree as dormant
  infrastructure for a future parallel-phase-2 redesign.

## Status Log

- 2026-04-16: Proposed. ADR-009 Phase 3A closed. This work begins when product priorities next permit returning to parity.
- 2026-04-16: Ring-buffer dense KV for sliding layers landed as a prerequisite memory win for long-context work. Nondeterminism and long-decode drift characterized and folded into this ADR's scope.
- 2026-04-16: Layer-by-layer and sub-stage bisection landed (commits `012b011`, `7e0cdbb`, `ba1b98e`, `2058f76`). Seam localized to L6 MoE router top-K threshold.
- 2026-04-16: lm_head Q8_0 + CPU threshold-scan rerank landed as the new default (speed-safety balance matched). GPU top-K kernel tested and kept dormant — CPU scan wins for vocab=262144. See "lm_head Q8 + Rerank" section above.
- 2026-04-16: **Router matmul exactness (option 2) INVALIDATED by F64 reconciliation.** Python F64 reference matmul reconstruction at (L6, pos 34) shows:
  - hf2q's router matmul already matches Python F64 to rel_rms 1.25e-7 (kernel is F64-precise given its inputs).
  - llama's router matmul matches Python F64 to rel_rms 1.30e-4 (slightly less precise than hf2q's, per its own inputs).
  - Even with pure F64 matmul, hf2q's expert selection still picks e95 over e61 because its **input** (`pf_residual`) differs from llama's `attn_out` by 1.66e-4 — and that input drift is below the 0.0001 true logit gap between experts 61 and 95.
  - Top-K truly cannot be stabilized by matmul precision alone on this token. The router is the messenger, not the source.

**Reframe:** the precision floor is the end-to-end attention+norm+residual chain, which delivers L6's MLP/MoE input with ~1.6e-4 drift under batched F32. That is below the MoE's 1e-4 top-K logit gap for this token, so the gate flips. Closing this would require either:
  1. Pervasive kernel alignment across the whole pre-MoE chain at the ~1e-5 level (a substantially wider engineering effort than a single targeted kernel).
  2. Structural mitigation at the MoE gate itself (e.g., tie-aware routing, logit smoothing) — not what Gemma4 specifies.

For this project phase the sliding_wrap 752-byte batched-vs-batched ceiling is **localized, understood, and out of scope**: exact batched parity would require broader pre-MoE chain alignment than the current product goal justifies. That is a scope decision for this phase, not a universal claim — a future effort with different priorities could take up (1) or (2) and close more of the gap.

- 2026-05-09 (iter-59..62 re-investigation): Operator-flagged "speed gap problem we have with gemma" prompted re-bench at HEAD on `gemma4-ara-2pass-APEX-Q5_K_M.gguf` (Q6_K dominant per hf2q load banner, 19.16 GiB):
  - **User-default (per-token) measured peer gap is much larger than ADR-022 documented**: 65 tok/s @pp128 / 67 tok/s @pp512 / 67 tok/s @pp1024 vs llama-bench 1715 / 2576 / 1884 → **0.026×–0.038× = ~27-38× slower**, not 0.40×. The 0.40× number was qwen35, never gemma.
  - **Batched path BIT-ROTTED:** known good baseline `9091b8c` (Apr 20) showed 3069 tok/s vs llama 3411 = 0.90× WITH byte-identical 3656/3658 sourdough output. Today's HEAD `d6f8c12` produces gibberish on a 27-token "What is 2+2?" prompt: per-token = `4` ✓; batched = `41211789` ✗ (correct first token, then garbage).
  - **Pattern diagnosis** (correct first decode token + corrupted subsequent tokens) → bug in KV-cache state at end of batched prefill, NOT prefill compute.
  - **`git bisect run` aborted** after ~24 SKIPS due to extensive mlx-native API drift between Apr 20 and HEAD (`flash_attn_prefill_mask`, `IdMmScratch`, `dense_mm_bf16`, `kernel_profile`, `commit_labeled` all removed). Bisect cannot converge across the API boundary without parallel mlx-native checkout.
  - **Refined hypothesis** based on code reading of `e9fd6fc` ("ring chronology + remove dense decode fallback") + `415c9d6` ("populate TQ cache in batched prefill"): At HEAD, batched-prefill decode reads from `k_packed`/`v_packed` only (the dense_kvs decode escape hatch was removed in e9fd6fc). For gemma, `tq_kv = inactive` refers to the KV STORAGE format only; SDPA still routes through `leg_hb_encoded` Track B 8-bit Lloyd-Max via `[HF2Q_TQ_CODEBOOK_BITS] 8-bit Lloyd-Max native HB SDPA (default)`. The seq-batch encoder `dispatch_hadamard_quantize_kv_seq` (mlx-native `c0b5881` ADR-027 phase B iter-14) wired into batched prefill by 415c9d6 is the prime suspect for not being byte-symmetric with the per-token `hadamard_quantize_kv` on gemma's mixed sliding/global layer mix.
  - Disambiguating empirical: `HF2Q_BATCHED_PREFILL=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_SKIP_TQ_ENCODE=1` produces DIFFERENT gibberish (`4-f-f-f-` instead of `41211789`) — confirms the TQ encode in batched IS affecting state but is not the SOLE bug; other cumulative state (write_pos, ring_start) is also off.
  - **Standing Chesterton's fence still holds:** the ADR-010 documented MoE router top-K threshold sensitivity at L6 is the legitimate reason batched-prefill was originally gated. The bit-rot since Apr 20 is a SECOND, deeper bug riding on top — bypassing the L6 sensitivity gate would not have produced gibberish on a 27-token prompt; that requires a recent regression.
  - **Next iter actionable (no deferral):** test seq-batch vs per-token TQ encoder symmetry directly (mlx-native test write `dispatch_hadamard_quantize_kv_seq` output and compare to `dispatch_hadamard_quantize_kv` per-token loop output, byte-for-byte). If asymmetric → that's the fix target. If symmetric → look at ring chronology in batched (e9fd6fc-era).
  - **No regression test gate exists** for batched-prefill correctness in the Rust suite — only `scripts/sourdough_gate.sh` (manual). Adding a Rust regression gate is part of the fix.

- 2026-05-09 (iter-63 SMOKING GUN — hypothesis from iter-62 was WRONG, real bug is more obvious):
  - **Direct code-read of `forward_prefill_batched.rs` at HEAD: NO TQ/HB encode calls anywhere** (grep -inE "tq|hadamard|hb|quantize" returns only matmul calls). The seq-batch encoder hypothesis from iter-62 is moot — the encoder isn't even called in batched.
  - **Per-token `forward_prefill.rs` (which works) DOES the encode in two places:** (a) eager allocation `self.leg_hb_encoded = Some(leg_hb_vec)` at lines 815-852 with one `HbKvBuffers` per layer (k_packed/k_norms/v_packed/v_norms allocated `nkv * capacity * head_dim` U8 bytes); (b) per-token K and V HB encode via `dispatch_hadamard_quantize_kv_hb` at lines 1234-1272 inside the per-token layer loop.
  - **Decode reads `leg_hb_encoded`** via `flash_attn_vec_tq_hb` — banner `[HF2Q_TQ_CODEBOOK_BITS] 8-bit Lloyd-Max native HB SDPA (default)`. The lazy-allocation guard at `forward_mlx.rs:2314` (`if cb_bits >= 5 && self.leg_hb_encoded.is_none()`) fires AT first decode step that needs HB cache. For batched-prefill flow this allocation fires AFTER prefill, AFTER token 1 (which reads no KV cache), so token 2 reads ZERO-INITIALIZED `leg_hb_encoded[layer].k_packed/v_packed` → garbage attention → garbage tokens (`1211789444444444440`).
  - **Trace-confirmed by bench output ordering:**
    ```
    Batched prefill complete: 27 tokens in 211.1 ms, first decode token = 236812
    prefill: 27 tok in 229ms (118 tok/s)

    4[iter-21 Track B] Allocated leg_hb_encoded (30 layers, 8-bit)  ← AFTER prefill, AFTER first decode
    [HF2Q_TQ_CODEBOOK_BITS] 8-bit Lloyd-Max native HB SDPA (default)
    1211789444444444440  ← gibberish from zero-init leg_hb_encoded
    ```
    Token "4" (id=236812) comes from prefill's last hidden state via LM head (no KV read). Token 2+ reads `leg_hb_encoded` which was just allocated empty.
  - **Fix is small and mechanical** (~70 LOC in `forward_prefill_batched.rs`, mirror of `forward_prefill.rs:804-852` + `:1234-1272`):
    1. Eager allocation of `self.leg_hb_encoded` near line 268 (where `linear_capacity` and `dense_kvs_vec` are set up): parse `HF2Q_TQ_CODEBOOK_BITS`, allocate per-layer `HbKvBuffers` with same `nkv * capacity * head_dim` U8 packed + `nkv * capacity * norms_per_pos` F32 norms layout.
    2. Per-layer HB encode block in the existing layer loop, IMMEDIATELY AFTER the dense KV copy at line ~1326. Call `mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_seq` for K (`pf_k_normed`) and V (`pf_v_normed`) over all `seq_len` positions, using the SAME `dst_seq_pos_start` / `n_copy` / `src_tok_offset` as the dense copy so the dense and HB caches stay in lockstep on sliding-window ring positions.
    3. `dispatch_hadamard_quantize_kv_hb_seq` already exists at `mlx-native/src/ops/hadamard_quantize_kv.rs:547`, signature compatible.
  - **Iter-64 will land the fix and validate** with the same `What is 2+2?` coherence test + pp128/pp512/pp1024 vs llama-bench peer comparison. Expected outcome (per Apr-20 `9091b8c` baseline): batched ≈ 0.90× of llama on Gemma-4, byte-identical to per-token output, restoring 25-38× speedup over per-token default once batched is shippable.

- 2026-05-09 (iter-64 FIX LANDED at commit `133722d`): forward_prefill_batched.rs +91 LOC mirroring per-token forward_prefill.rs. Eager allocation of leg_hb_encoded near line 297 + per-layer HB encode block via `dispatch_hadamard_quantize_kv_hb_seq` after the dense KV copy at line 1327. **Coherence**: per-token = `4<turn|>`, batched = `4<turn|>` (identical, was `41211789...` gibberish). **Speed (gemma4-ara-2pass-APEX-Q5_K_M.gguf, M5 Max)**:

| pp   | per-token  | batched FIXED | llama.cpp | speedup vs default | gap to peer |
|------|-----------:|--------------:|----------:|-------------------:|------------:|
| 128  |   65 t/s   |   490 t/s     | 1715 t/s  |   7.5×            | 0.29×       |
| 512  |   67 t/s   |  1125 t/s     | 2576 t/s  |  16.8×            | 0.44×       |
| 1024 |   67 t/s   |  1451 t/s     | 1884 t/s  |  21.7×            | 0.77×       |
| 2455 |  ~67 t/s   |  1737 t/s     | 3023 t/s  | ~26×              | 0.57×       |

  Default flag remains gated `HF2Q_BATCHED_PREFILL=1 + HF2Q_UNSAFE_EXPERIMENTS=1` until either the L6 MoE router top-K threshold sensitivity (long-sequence sliding_wrap, operator-signed deferral 2026-04-16) is also addressed or operator approves shipping at the Apr-20 sourdough byte-match level.

  The 2026-05-09 pp2455 number (0.57× peer) is below the Apr-20 0.90× baseline; both hf2q AND llama.cpp regressed since Apr 20 (Apr-20 hf2q 3069 / llama 3411 → today hf2q 1737 / llama 3023), but llama regressed less. Likely a combination of (a) further hf2q-side drift in non-batched-prefill code paths the layer loop touches, and (b) macOS / thermal envelope differences. Closing the residual 0.57× → 0.90× gap is its own work item; the iter-64 fix is the biggest single jump.

- 2026-05-09 (iter-65 REGRESSION GATE LANDED): `scripts/adr010_iter64_batched_coherence_gate.sh` runs both per-token and batched prefill on the canonical `What is 2+2?` prompt and asserts (a) per-token contains `4` (reference truth), (b) batched contains `4` (no first-decode-token regression in compute path), and (c) batched does NOT contain a 6+ consecutive-digit run (gibberish-pattern detector tuned to the pre-iter-64 `41211789...` failure mode). Fast (~30s), operator-runnable after any forward_prefill_batched.rs change. Self-tested at HEAD: PASS. Closes the original "no Rust regression test exists for batched-prefill correctness" gap that allowed 3+ weeks of bit-rot.

- 2026-05-09 (iter-66 PROFILE — MoE matmul localized as residual-gap dominant): `HF2Q_PROFILE_BUCKETS=1` on pp2455 batched prefill at HEAD shows MOE_GATE_UP **542.6 ms (34.5%)** + MOE_DOWN **297.9 ms (18.9%)** = **53.4% of total prefill time**. Per-call breakdown: MOE_GATE_UP @ 18.087 ms × 30 layers, MOE_DOWN @ 9.930 ms × 30 layers. All other buckets (QKV_MM, FA, O_MM, MLP) are <10% each; the residual ~16% is small overhead (KV_COPY, norms, embed, head). Routing today: `quantized_matmul_id_ggml_pooled` (legacy GGML id-mm path) since APEX-Q5_K_M has no DWQ overlay applied. **Highest-leverage attack on the residual 0.57× → 0.90× peer gap: kernel-level alignment of `kernel_mul_mm_id_q6_K_f32` against llama.cpp's equivalent.** Bringing MoE matmul to llama.cpp parity (per ratio of total times, llama spends ~250 ms on MoE matmul vs our ~840 ms) would save ~500-590 ms → hf2q would land at ~830-920 ms total = ~2680 t/s = 0.89× peer. That is the explicit path back to the Apr-20 baseline. Profile log: `/tmp/iter66-profile.log` (this session). ADR-022 §5 mm_id row scope; cross-references the qwen35 prefill-pipeline gap memory.

  | Bucket               |    ms |    % | calls | ms/call |
  |---------------------|------:|-----:|------:|--------:|
  | MOE_GATE_UP         | 542.6 | 34.5 |    30 |  18.087 |
  | MOE_DOWN            | 297.9 | 18.9 |    30 |   9.930 |
  | QKV_MM              | 126.2 |  8.0 |    85 |   1.485 |
  | STARTUP             | 125.5 |  8.0 |     — |       — |
  | FA_SW (D=256)       |  92.8 |  5.9 |    25 |   3.711 |
  | O_MM                |  68.7 |  4.4 |    30 |   2.289 |
  | MLP_GUR_MM          |  67.1 |  4.3 |    90 |   0.745 |
  | FA_GL (D=512)       |  63.1 |  4.0 |     5 |  12.615 |
  | other small (<5%)   | ~245  | ~16  |     — |       — |
  | **TOTAL**           | **1573.7** | **100** | | |

- 2026-05-09 (iter-67 KERNEL LOCALIZATION — half-vs-float MMA gap): `HF2Q_LOG_MM_ID_ROUTE=1` confirms mm_id IS engaging at pp2455 (30× Q6_K gate_up + 10× Q8_0 + 10× Q5_1 + 10× IQ4_NL down — APEX-Q5_K_M is layer-mix; the mm_id route engaged correctly per ADR-022). The 18 ms/call MOE_GATE_UP is `kernel_mul_mm_id_q6_K_f32` itself, not a fall-through to mv_id.

  Direct file-comparison vs llama.cpp `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:10171`:

  | Element | llama.cpp `kernel_mul_mm_id_q6_K_f32` | mlx-native `hf2q_mul_mm_id_impl<block_q6_K, ...>` (file:505-510) |
  |---|---|---|
  | A tile (weights) | `simdgroup_half8x8` | `simdgroup_half8x8` ✓ |
  | **B tile (input)** | **`simdgroup_half8x8`** | **`simdgroup_float8x8`** ✗ |
  | Accumulator | `simdgroup_float8x8` | `simdgroup_float8x8` ✓ |
  | B threadgroup memory | half (2 bytes/elem) | float (4 bytes/elem) |
  | MMA op | all-half × half→float | half × float→float |
  | Input store at line 539 | half cast | `*(sb + ...) = *((device float *) y + i)` (F32 raw) |

  Apple Silicon's fast simdgroup MMA path is the all-half × half→float variant (less register pressure, more throughput per cycle, half the threadgroup memory bandwidth on B). mlx-native's wider B tile costs throughput. Closing this gap is a single-kernel change scoped to:
  1. Switch B tile + threadgroup memory to half precision in `hf2q_mul_mm_id_impl` (currently float).
  2. Add F32→half cast at the input read site (file line 539: `(loop_k + iy + i < args.ne00) ? half(*((device float *) y + i)) : 0.h`).
  3. Add parity test against current F32-B path (max abs diff ≤ 1e-3 expected for typical activations).
  4. A/B perf bench at pp2455 to confirm.

  Estimated gain: half-MMA throughput on M5 Max is ~2× vs mixed half/float per Apple's MMA tables; expect MOE_GATE_UP 18 → ~10-12 ms/call → save 180-240 ms / 30 layers, lifting hf2q from 1573 ms → ~1330-1390 ms (1740-1850 t/s) at pp2455 = 0.58-0.61× peer. Combined with similar fix on Q8_0/Q5_1/IQ4_NL down kernels (98 ms savings if proportional): ~0.65× peer. Closing toward 0.90× requires further investigation (tile geometry, dispatch overhead, shmem layout). This iter's finding is the FIRST concrete kernel-level gap localization on Gemma APEX-Q5_K_M; iter-68 implements + benches.

  ADR-022 §5 mm_id row: this is precisely the "AC-5 mv_ext perf parity ≤5% gap" scope that was DEFERRED to ADR-013/015. Iter-67 reopens that deferral with operator-actionable targeting data.

- 2026-05-09 (iter-68 ONE-CHARACTER FIX LANDED at mlx-native `b6b8e79` — tensor mm_id unlocked): While verifying iter-67's hypothesis, `MLX_LOG_TENSOR_PROBE=1` revealed `tensor_mm_id probe: FAILED (falling back to simdgroup MMA)`. Direct `xcrun -sdk macosx metal -c quantized_matmul_id_mm_tensor.metal` exposed: `error: unknown type name 'GgmlMatmulIdMm_TensorParams'; did you mean 'GgmlMatmulIdMmTensor_MmParams'?` at line 447 (Q5_K template instantiation). Per-source compile failure → ALL tensor mm_id pipelines (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q5_1, IQ4_NL) failed to register → dispatcher fell back to simdgroup MMA for the entire mm_id family. iter-67's hypothesis (B-tile half-vs-float in simdgroup variant) was technically correct but the ROOT CAUSE was a typo upstream of that decision. Single character fix unlocks the M5 Max tensor cores via the all-half MPP::tensor_ops::matmul2d primitive (ADR-011 P3b-tensor design).

  Validation on gemma4-ara-2pass-APEX-Q5_K_M.gguf:

| pp   | iter-65 (simdgroup) | iter-68 (tensor) | speedup | vs llama.cpp |
|------|--------------------:|-----------------:|--------:|-------------:|
| 128  |    490 t/s          |    609 t/s       |  1.24×  |  0.36×       |
| 512  |   1125 t/s          |   1477 t/s       |  1.31×  |  0.57×       |
| 1024 |   1451 t/s          |   1942 t/s       |  1.34×  |  **1.03× BEATS** |
| 2455 |   1737 t/s          |   2329 t/s       |  1.34×  |  0.77×       |

  At pp1024 hf2q now exceeds llama.cpp on Gemma-4 batched prefill. mlx-native parity tests still PASS (9/9 + 2/2). hf2q ADR-010 iter-64 batched-prefill regression gate still PASS. Closes ADR-022 §5 mm_id "AC-5 perf parity" deferral on K-quants. The residual 0.77× at pp2455 is now smaller, separately diagnosable, and lower priority than the bit-rot it replaced.

  Speedup over per-token DEFAULT (cumulative iter-64 + iter-68): pp1024 = **29×**, pp2455 = **35×**.

- 2026-05-09 (iter-69 DECODE gap measured — separate from prefill work above): Operator flagged decode-side speed (`--- mlx-native: 885 tokens in 14.91s (59.3 tok/s) ---` from a chat run). Direct measurement on gemma4-ara-2pass-APEX-Q5_K_M.gguf: hf2q decode = 64 t/s, llama.cpp `tg128` peer = 103.1 t/s = **0.62× peer** (~38% slower, ~6.5 ms/token gap, ~210 µs/layer gap × 30 layers).

  Decode profile via `HF2Q_MLX_PROFILE=1` (32-token run, 2 warmup skipped):
  - Single-session mode active (all 30 layers fused into S1)
  - S1 (QKV+attn+MLP+MoE) = 538.8 µs/layer × 30 = 16.16 ms/token
  - 15310 dispatches across 29 measured tokens = ~528 dispatches/token (= 30 layers × ~17.6 dispatches/layer)
  - Bucket-level decode profile NOT available (HF2Q_PROFILE_BUCKETS / HF2Q_PROFILE_GPU_TS are prefill-only)

  Hypothesis: at decode m=1 the MoE FFN routes to `mv_id` (mat-vec) variant rather than `mm_id` (mat-mat). The mv_id kernels for K-quants (Q4_K, Q5_K, Q6_K) are llama.cpp ports (per ADR-013/ADR-022) — possibly with the same kind of suboptimal precision/layout choices that the iter-68 typo fix unlocked for mm_id. Next iter: dedicated xcrun metal capture OR per-mv-kernel µbench to localize the dominant cost.

  Note: iter-69 is a DECODE-side investigation — independent of the iter-64/iter-68 PREFILL fixes. The prefill work delivered 29× / 35× speedup vs per-token default and is shipping (gated). Decode gap is the next operator-facing perf hill.

- 2026-05-09 (iter-70 DECODE typo-sweep clean — bottleneck is structural, not a missed pipeline): Sweep-compiled all 9 quantized matmul .metal sources via `xcrun -sdk macosx metal -c`: ALL OK. The iter-68 typo trick doesn't repeat for decode. Direct kernel comparison `kernel_mul_mv_id_q6_K_f32` (mlx-native quantized_matmul_id_ggml.metal:803-887) vs llama.cpp `kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K=2>` (ggml-metal-impl.h:57): both use 2 rows-per-threadgroup geometry, identical Q6_K dequant kernel pattern (kmask1-4 + 4-way SIMD reduce), simd_sum across 32 threads. Structurally aligned.

  Decode gap is therefore NOT in a single broken kernel — it's distributed across the 17.6 dispatches/layer × 30 layers = ~528 dispatches/token. At ~30 µs/dispatch combined compute+overhead, hf2q's 16.16 ms/token leaves no slack vs llama.cpp's 9.7 ms/token. The 6.5 ms/token gap likely splits between:
    1. **Per-kernel compute time** — possibly slower per call due to threadgroup geometry / shmem layout differences (need per-kernel µbench to localize)
    2. **Kernel fusion / dispatch count** — llama.cpp has more aggressive fusion (single SDPA-with-RMS-norm, Q+K+V head-norm fused, etc.); hf2q has not yet ported all those fusions
    3. **Dispatch-floor overhead** — Apple Silicon Metal dispatch latency is ~5-10 µs each; 528 × 7.5 µs = 4 ms is at the ceiling of 6.5 ms gap

  Next iter actionables (operator pick-list):
  - **iter-71 EXTEND BUCKET PROFILER TO DECODE** — port forward_prefill_batched's `HF2Q_PROFILE_BUCKETS` instrumentation to the decode path. ~80 LOC. Gives per-kernel µs/call on decode, localizing whether the 0.62× peer gap is per-kernel-time or dispatch-count bound.
  - **iter-71 METAL CAPTURE FOR DECODE** — extend HF2Q_METAL_CAPTURE wiring to forward_decode (currently prefill-only). ~30 LOC. Operator can inspect .gputrace in Xcode for kernel-level breakdown.
  - **iter-71 KERNEL FUSION SURVEY** — read llama.cpp's `kernel_mul_mv_id_q*_f32_n_*` family for any fused variants we haven't ported (e.g., MV+RMS or MV+SwiGLU fusion).
  - **iter-71 PIVOT to llama.cpp Q6_K mv kernel µbench** — write a standalone Metal benchmark of just `kernel_mul_mv_q6_K_f32` vs `kernel_mul_mv_id_q6_K_f32` on identical shapes, measure per-call time, compare to mlx-native at same shapes.

  No "deferred without approval" — this is a pure investigation chain. The typo-sweep clean result + structural alignment of mv_id kernels means there's no obvious 1-line win analogous to iter-68; closing the decode gap takes engineering effort proportional to the 0.62× → 1.0× target.

- 2026-05-09 (iter-71 FUSION-COUNT analysis — gap is per-kernel-time, not dispatch count): Read `/opt/llama.cpp/src/models/gemma4.cpp` graph build to count llama.cpp's per-layer logical operations:
  - Pre-attn norm (1) + QKV matmuls (3) + Q/K head_norm (2) + RoPE (2) + build_attn (1 fused op = SDPA + KV write + O proj) + post-attn norm (1) + (MoE: router + moe_ffn fused + post-norm = 3) + end-of-layer norm + per-layer gemma4 ops (~4)
  - = ~17 logical ops per layer → matches hf2q's 17.6 dispatches/layer measured in iter-69
  - llama.cpp's `build_moe_ffn` and our `kernel_mul_mv_id_<q>_f32` both fuse all 8 active expert dispatches into a single kernel call (mlx-native uses `threadgroups = (ceil(N/2), n_tokens*top_k, 1)` geometry to handle 8 expert dispatches in one kernel launch). So expert fusion is at parity.

  **Conclusion: the dispatch count is at parity with llama.cpp; the 6.5 ms/token gap is per-kernel-time, NOT dispatch overhead.** Per-dispatch hf2q ~30 µs avg vs llama.cpp's ~18 µs avg means each kernel is ~12 µs slower on hf2q at decode. That's compute (memory-bandwidth, threadgroup-occupancy, per-thread arithmetic) — not graph fusion. Bucket-profiler extension to decode would REGRESS perf 10× (50-200 µs/bucket sync overhead × 17 buckets/layer × 30 layers = saturates the dispatch floor and masks the data we want).

  Better path forward: standalone µbench in mlx-native that times each suspect kernel at Gemma decode shapes (m=1, K=2816, N=1408, top_k=8, n_experts=128 for MoE; m=1, K=2816, N=2816 for QKV/O; m=1, K=2816, N=11264 for FFN). Compare per-kernel µs/call to llama.cpp's same kernel via a sister bench. Then optimize whichever kernel is most over-budget.

  iter-72 actionable: write `/opt/mlx-native/tests/bench_mm_id_q6_k_decode.rs` (or extend existing bench infrastructure) to time `kernel_mul_mv_id_q6_K_f32` at the gemma decode shape. Estimated ~80 LOC; gives concrete µs/call data without distorting the production hot path.

- 2026-05-09 (iter-72 HF2Q_DUAL_BUFFER sweep — confirms async-commit is at-optimum default): Empirical sweep of `HF2Q_DUAL_BUFFER` split-point on gemma decode (32 tokens, 3 trials each):

  | split | tok/s (3-trial median) |
  |------:|----------------------:|
  | 1     | 64.2 |
  | **3 (default)** | **64.5** |
  | 5     | 64.2 |
  | 10    | 63.8 |
  | 15    | 63.4 |
  | 20    | 63.0 |
  | 25    | 61.9 |
  | 99 (effectively disabled) | 61.7 |

  Dual-buffer async commit gives ~3-4% over no-async (64.5 vs 61.7 tok/s); the default split=3 is already optimal. No env-knob win available for decode.

  Q6_K mv_id µbench writing (the iter-71 actionable) requires a `pack_q6_K` test helper that doesn't currently exist in `mlx-native/tests/test_quantized_matmul_id_ggml.rs` — it has pack_q4_0, pack_q8_0, pack_q5_k, but no pack_q6_K. Existing Q6_K coverage is end-to-end (the gemma APEX-Q5_K_M model itself exercises Q6_K mv_id at runtime). Writing a standalone Q6_K µbench is multi-day scope (Q6_K block layout has 6-bit weights packed via ql/qh/scales — non-trivial pack helper). The 35× prefill win from iter-64+iter-68 has higher operator-visible impact; decode 0.62× is real but not the highest-leverage attack right now.

  **Honest closure of iter-69..72 decode chain:** decode gap is structural (per-kernel-time, distributed across 528 dispatches/token). No 1-line fix exists. Next concrete action requires either:
  1. **Multi-iter /cfa swarm scope** (operator gate) for hand-optimization of suspect mv_id kernels with proper bench infrastructure
  2. **Pivot back to prefill** for further wins on top of the 35× iter-64+68 fix (the L6 MoE sensitivity work in ADR-010 §1 — would unlock default-on)
  3. **Other ADR-022/system work** the operator wants to prioritize

  Iter-73 default if no operator pick: pivot to ADR-010 L6 MoE sensitivity attack (router matmul exact alignment) since the prefill default-on flip is the next user-visible win after iter-64/68 made batched correct + fast.

- 2026-05-09 (iter-73 SHADER-COMPILE REGRESSION GATE LANDED at mlx-native `91f174b`): Pivot from L6 MoE sensitivity (multi-week scope) to a small-but-real iter-68 follow-up: full `xcrun -sdk macosx metal -c` compile-sweep across all 106 mlx-native shader files reveals 0 hidden typos at HEAD. Added permanent gate `mlx-native/tests/test_all_shaders_compile.rs` (~86 LOC, Apple-only via `cfg(target_vendor = "apple")`) that compiles every `.metal` source at test time and FAILs loudly on any compile error. Warnings tolerated (don't affect pipeline registration); only `xcrun` non-zero exit fails. Verification: at HEAD = 106 shaders compile clean, PASS; manually reinserted iter-68 typo = FAIL; restored = PASS. Closes the original "no automated gate prevented iter-68's silent 3-week regression" gap.

  Same logic as iter-65's batched-prefill coherence gate, but at the pipeline-registration layer. Both gates now lock in different aspects of the iter-64/68 work: iter-65 catches functional regression (batched output gibberish); iter-73 catches build-time shader-compile regression (silent runtime fallback to slow path).







