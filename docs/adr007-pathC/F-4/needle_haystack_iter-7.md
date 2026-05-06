# ADR-007 Path C / F-4.4 — needle-in-haystack iter-7 partial

**Date:** 2026-05-05 (iter-7)
**Status:** Partial — 4 of 6 planned trials complete, 100% PASS rate so far. 64K F-4.2/3 bench + remaining 8K trials in flight (iter-8 harvest).

## What landed

- `scripts/bench-needle-haystack.sh` — F-4.4 harness. Plants a 5-char
  alphanumeric needle at fractional positions {0.1, 0.5, 0.9} in a
  long-context prompt and asks the model to retrieve it. Per-trial JSON
  records target tokens, actual prefill tokens, position fraction,
  needle, verdict, and a snippet of the response.

- Harness verdict logic patched in iter-7: hf2q emits a `[HF2Q_TQ_CODEBOOK_BITS]
  8-bit Lloyd-Max native HB SDPA (default)` banner at the first decode step
  that gets interleaved with model output via `2>&1`. The original `grep -F
  "$needle"` failed when the needle was split across the banner injection
  ("E[HF2Q_...] IEDA" instead of "EIEDA"). Fix: strip `[HF2Q_*]` and
  `[iter-*]` lines and concatenate all remaining whitespace before the grep.

## Iter-7 results (partial)

| Target tokens | Actual prefill | Position | Needle | Verdict |
|---|---|---|---|---|
| 4096  |  5559 | 0.1 | AAEBE | PASS |
| 4096  |  5557 | 0.5 | EIEDA | PASS |
| 4096  |  5560 | 0.9 | ADEFE | PASS |
| 8192  | 11086 | 0.1 | DDABA | PASS |
| 8192  |     - | 0.5 | (in flight) | (iter-8) |
| 8192  |     - | 0.9 | (in flight) | (iter-8) |

**4/4 PASS rate at 4K + 8K early-position needles.** The model retrieves
the planted needle at every trial completed so far, even when the needle
is at the start (pos 0.1) or end (pos 0.9) of the haystack.

Notes:
- Actual prefill is ~36% larger than the target (chat-template overhead
  + Gemma's BPE tokenizer on the filler text). Worth a more accurate
  pre-tokenizer estimate in a future iteration but doesn't change the
  pass/fail finding at these scales.
- Prefill speed: ~53 tok/s at 5559 tokens; ~45 tok/s at 11086 tokens.
  Decay consistent with Gemma 4 sliding+global mix at increasing context.

## Background tasks (still running)

Two long-running tasks were launched at iter-7 start:

1. **64K F-4.2/3 bench** (`bg task bfas3ib2w`):
   `scripts/bench-long-context.sh 65536` capturing prefill+decode t/s
   and peak RSS at ~64K context. Estimated 30-40 min wall time.

2. **Remaining 8K trials** (`bg task bsgal766w`):
   8192 pos=0.5 and pos=0.9 needle retrievals.

Both should be complete by iter-8 wakeup (~5 min cache window from
this commit).

## Implication for F-4 closure

If the 4/4 pass rate holds through iter-8 and iter-9 (16K + 32K), and
no failures show up at 64K+, F-4.4 closes with:
- **PASS at all sampled (length, position) cells up to 64K.**

If failures show up at higher contexts (e.g., the model fails at 32K
pos=0.5 because the needle falls outside the sliding window AND the
global layer doesn't capture it), we have a real ADR-finding to write
up — specifically the trade-off between sliding window (1024) and
global attention (sparse layers, 5 of 30) at sub-32K range.

## Iter-8 plan

1. Harvest 64K bench JSON + remaining 8K needle trials.
2. Run 16K + 32K needle trials.
3. If GPU memory permits, run 64K needle trial (single position only;
   each trial is ~10-15 min wall time at 64K).
4. Document. Commit.

## Iter-9+ plan (only if user proceeds)

5. 128K + 262K bench (each ~30-90 min wall time per trial).
6. F-5 MMLU/LongBench harness (separate from F-4; ~1-2 days of
   benchmark integration work).
7. F-6 CLI flag (cosmetic; ~50 LOC).
