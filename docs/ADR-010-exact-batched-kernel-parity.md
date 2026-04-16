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
