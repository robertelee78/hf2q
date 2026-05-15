# ADR-029 iter-175 Step 1ab — prefill measurement methodology problem

**Date**: 2026-05-15
**HEAD**: hf2q `0ace22ef`, mlx-native `22dc55b`, llama.cpp `389ff61d7`
**Iteration**: 32 of /loop autonomous

## What I tried to measure

After Step 1aa's decode ratio (0.9365×), verify whether prefill is still
**AHEAD** of peer-FA per the iter-160 standing memory
(`project_adr029_iter160_prefill_AHEAD_2026_05_13.md`: 1.072× pp1800,
1.087× pp3700).

## What I measured

3-cycle alt-pair at HEAD, pp~1800:

```
hf2q    : mean 2715.17  σ 65.36 (2.41%)  samples: [2622.9, 2756.5, 2766.1]
peer-FA : mean 2834.96  σ  7.06 (0.25%)  samples: [2826.37, 2843.67, 2834.85]
ratio   : 0.9577×  (-4.23% behind peer)
```

## Why this is NOT a clean regression claim

The two arms ran **different workloads**, not the same workload at
different speeds:

1. **hf2q arm**: real prompt `"the quick brown fox jumps over the lazy dog. " × 200`
   = ~2000 real BPE tokens. Tokenization, model embedding lookup, real KV-cache
   allocation. Returns one decode token after prefill.

2. **peer arm**: `llama-bench -p 1800 -n 0` uses **synthetic random token IDs**
   (peer's `g_pp_buf[i] = std::rand() % vocab_size`). No tokenization,
   no real prompt content. Pure prefill matmul throughput measurement.

Different per-token work + different actual token count (~2000 vs exactly 1800).

iter-160's measurement used `hf2q_bench_prefill --pp1800` with synthetic
random tokens on both sides — apples-to-apples.  Today's measurement
mixed methodologies — apples-to-oranges.

## What the data DOES say (cautiously)

- hf2q's σ-pct is 2.41% — too high for confident comparison.  Cycle 0
  (cold) was 2622.9 (~5% slower than cycles 1-2 mean 2761).  Suggests
  PSO-cache warmup wasn't sufficient.
- Steady-state hf2q (cycles 1-2): mean 2761, vs peer 2839 ≈ 0.972× = -2.8%.
- Even discarding cycle 0, ratio < 1.0 suggests prefill is no longer
  AHEAD at this measurement geometry — BUT methodology isn't comparable
  to iter-160.

## What should be done next (not this iter)

1. Use `hf2q_bench_prefill --pp 1800` (synthetic random tokens) instead of
   real prompt to re-establish apples-to-apples vs peer's `-p 1800`.
2. Larger warmup (5+ runs) to stabilize PSO cache.
3. Re-bench at pp3700 too to track iter-160's other regime.
4. If prefill regression is confirmed, bisect between iter-160 commit
   (around 2026-05-13) and HEAD to find the offending lever — most likely
   suspects are Step 1d (tracked-dispatch), Step 1e (q6_K_nr2 site migration),
   or Step 1m (precompiled metallib + first-run PSO compile cost).

## Why this iteration didn't finish the work

This is a "report, don't claim" iteration.  The proper apples-to-apples
prefill bench tool (`hf2q_bench_prefill --pp1800`) requires checking the
hf2q codebase for the exact CLI invocation — I didn't have it ready, and
running with a mismatched workload would produce a misleading claim
("prefill regressed").  The mantra says no shortcuts, no guesses.

Documenting the methodology problem now so a future iter can do the
clean comparison.

## Standing-context update

iter-160's "prefill AHEAD 1.07-1.09×" claim **NOT VERIFIED at HEAD**.
Apples-to-apples re-measurement needed before confirming or refuting.

## Cross-references

* iter-160 prior measurement: `project_adr029_iter160_prefill_AHEAD_2026_05_13.md`
* Step 1m prefill +3-5% direction: `project_adr029_iter175_step1m_precompiled_metallib_DEFAULT_ON_2026_05_15.md`
* Decode ratio at HEAD: `docs/research/ADR-029-iter-175-step-1aa-peer-ratio-at-HEAD-2026-05-15.md`
* Apples-to-apples rule: `feedback_targets_must_be_apples_to_apples_2026_05_11.md`
