# ADR-029 iter-175 Step 1ac — prefill apples-to-apples: hf2q 1.037× AHEAD

**Date**: 2026-05-15
**HEAD**: hf2q `b1876855`, mlx-native `22dc55b`, llama.cpp `389ff61d7`
**Iteration**: 33 of /loop autonomous

## Step 1ab was wrong — parser caught the wrong line

Step 1ab claimed hf2q prefill at HEAD was 0.9577× peer (-4.23% behind),
contradicting iter-160's "1.07-1.09× AHEAD".  The error: Step 1ab
parsed the `Prompt: X t/s` summary line in hf2q's generate output —
that's the **legacy-path prefill** number which includes ~9% non-matmul
overhead, not the pure batched prefill rate.

hf2q emits TWO prefill numbers:
- `Batched prefill complete: 2013 tokens in 663.4 ms (3034.3 tok/s)` ← **pure matmul rate** (correct for vs-peer comparison)
- `prefill: 2013 tok in 728ms (2765 tok/s)` ← **legacy-path total** (includes setup/wrap overhead)
- `Prompt: 2734.7 t/s` ← same as legacy-path

Step 1ab read the legacy number (~2735) and compared it to peer's pure
matmul rate (~2835).  Apples-to-oranges.  Sorry for the iter-32 false alarm.

## Step 1ac — proper apples-to-apples

Same machine, same session, same exact pp length (2013 tokens; tokenize
the hf2q prompt first, then run peer with `-p 2013`).  3-cycle alt-pair:

```
[iter-175 Step 1ac] 3-cycle alt-pair PREFILL hf2q@2013 vs peer@2013

--- cycle 0 ---
  hf2q pp2013 (batched): 3023.3 tok/s
  peer pp2013          : 2907.01 tok/s

--- cycle 1 ---
  peer pp2013          : 2930.44 tok/s
  hf2q pp2013 (batched): 3034.4 tok/s

--- cycle 2 ---
  hf2q pp2013 (batched): 3032.8 tok/s
  peer pp2013          : 2928.95 tok/s

=== aggregate ===
hf2q    : mean 3030.17  σ  4.90 (0.16%)  samples: [3023.3, 3034.4, 3032.8]
peer-FA : mean 2922.13  σ 10.71 (0.37%)  samples: [2907.01, 2930.44, 2928.95]
ratio   : 1.0370×  (+3.70% AHEAD)
```

Both arms σ < 0.5% — tightest prefill bench yet recorded for iter-175.

## Comparison to iter-160 standing memory

| Iter | pp | hf2q t/s | peer t/s | ratio |
|---|---|---|---|---|
| iter-160 | pp1800 | — | — | 1.072× |
| iter-160 | pp3700 | — | — | 1.087× |
| **iter-175 Step 1ac** | **pp2013** | **3030** | **2922** | **1.0370×** |

iter-160 ratios are ~3-5pp higher than today's 1.037×.  Possible explanations:

1. **Peer build progression** — peer between iter-160 (around 2026-05-13 HEAD)
   and today (`389ff61d7`, May 2026) likely gained prefill commits.  E.g.,
   `da4495332` FC-promote (which hf2q also ported as H93) was around that
   timeframe and helps prefill MUL_MM too.
2. **Different pp regime** — iter-160 was pp1800 and pp3700; today is pp2013.
   The gap may be non-monotonic in pp length.
3. **Different prompt content** — iter-160 likely used synthetic random
   tokens via a peer-side bench; today uses real-tokenized prose.  Both are
   apples-to-apples per-arm but the per-token compute may differ slightly
   based on token distribution (rare-token cache misses, etc.).

The direction is confirmed: **prefill is AHEAD at HEAD by 3.7%**.  Lower
than iter-160's 7-9% lead, but solidly above 1.0×.

## Standing-context update

* prefill at pp2013 (apples-to-apples): **1.0370× peer-FA AHEAD**
* hf2q prefill canonical: **3030 ± 4.9 t/s** at pp2013
* peer-FA prefill canonical: **2922 ± 10.7 t/s** at pp2013

For iter-176+ regression gates:
- Drop below 1.00× = prefill regression to investigate.
- Drop below 0.95× would indicate something serious in the prefill path.

## What this iteration RULED OUT

* Step 1ab's "prefill regression" claim is **WRONG** — caused by parser bug
  catching the wrong output line.  The standing memory entry
  `project_adr029_iter160_prefill_AHEAD_2026_05_13.md` direction
  (prefill AHEAD) holds at HEAD.
* No further investigation of "prefill regression cause" needed.

## Cross-references

* Step 1ab parser bug (this supersedes it): `docs/research/ADR-029-iter-175-step-1ab-prefill-methodology-2026-05-15.md`
* iter-160 prior measurement: `project_adr029_iter160_prefill_AHEAD_2026_05_13.md`
* Step 1aa decode ratio at HEAD: `docs/research/ADR-029-iter-175-step-1aa-peer-ratio-at-HEAD-2026-05-15.md`
* Apples-to-apples rule: `feedback_targets_must_be_apples_to_apples_2026_05_11.md`
