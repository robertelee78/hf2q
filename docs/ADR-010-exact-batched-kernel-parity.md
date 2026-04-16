# ADR-010: Exact Batched-Kernel Parity with llama.cpp

**Status:** Proposed
**Date:** 2026-04-16
**Decision Makers:** Robert, Claude
**Related ADRs:** ADR-006 (mlx-native GPU backend), ADR-007 (TurboQuant KV cache), ADR-008 (candle divorce), ADR-009 (reference parity and coherence recovery)

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

### In scope

1. **Boundary dumps at batched-kernel sub-stages**
   - After QK GEMM, before softmax (raw attention logits, all rows)
   - After softmax, before V aggregation (attention weights)
   - After V aggregation (sdpa_out per head, per row)
   - For layers 0–8 of the sliding attention stack at prompt positions covering the full prefill window.

2. **Reduction-order alignment**
   - Compare hf2q's tiled `sdpa` reduction order with ggml's flash-attention-ext tile loop structure.
   - Identify any divergence in accumulator zeroing, tile boundary FMA order, or row-max / row-sum stability.

3. **Accumulator/precision alignment**
   - Verify both stacks use identical accumulator precision (F32) at every reduction.
   - Check for unintended F16 intermediate in the hf2q path.

4. **Kernel-level reference replication**
   - If sub-stage dumps localize the gap to a single kernel (e.g., batched QK GEMM), consider porting that specific kernel to mirror llama's implementation. This is a targeted bit-exact replication, not a full framework rewrite.

### Explicitly out of scope (for now)

- **Direct ggml integration.** Violates ADR-008 ("we own the stack"). Only reconsider if the parity cost/benefit dramatically favors it after sub-stage evidence is collected.
- **Rewriting hf2q's general GPU framework.** This ADR is about matching a specific numerical trajectory, not about reshaping the compute architecture.

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

## Status Log

- 2026-04-16: Proposed. ADR-009 Phase 3A closed. This work begins when product priorities next permit returning to parity.
- 2026-04-16: Ring-buffer dense KV for sliding layers landed as a prerequisite memory win for long-context work. Nondeterminism and long-decode drift characterized and folded into this ADR's scope.
