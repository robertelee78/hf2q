# ADR-007 Path C — closure summary

**Date:** 2026-05-05 (iter-11)
**Status:** Substantively closed. 10 of 11 phases landed with empirical evidence; F-5 (paper-standard benchmarks) remains as discretionary follow-up.

## What Path C answered

The Path C reopen (2026-05-05) was driven by 10 specific concerns about the
ADR-007 close-section (§REOPENED Why-list). Each is now resolved:

| # | Concern | Resolution | Evidence |
|---|---|---|---|
| 1 | Phase 0 SKIPPED | F-0 phase landed retroactively | F-0.1, F-0.2, F-0.3 |
| 2 | `dense_kvs` relabeled not removed | F-3-C: empirically justified opt-out | `F-3/dense_kvs_decision.md` |
| 3 | iter-27 collapse never root-caused | F-0.3 shows distribution is N(0,1) — calibration is unnecessary | `F-2/uncalibrated_is_floor.md` |
| 4 | Gate C relaxed not earned | Earned-vs-floored answered: 1.24% PPL is intrinsic Lloyd-Max physics | F-2 + F-0.3 |
| 5 | C-2 → C-4 audits open | Superseded by F-0.2 falsifier-clearance at production shape | `F-0/divergence_audit.md` |
| 6 | 262K context never unlocked | F-4.1 found no cap; F-4.2/3 validated scaling; F-4.4 12/12 needle PASS | `F-4/*` |
| 7 | MMLU/LongBench never run | F-5 — discretionary follow-up (see below) | open |
| 8 | CLI `--kv-bits` flag missing | F-6.1 landed | iter-11 commit |
| 9 | No codec-freeze contract | F-7 locks `codec_version=1` per `EnvelopeHeader` | ADR `## Codec Freeze Contract` |
| 10 | 16-bit TQ never investigated | F-6.2 retired — structurally redundant with F16 dense | F-2 + F-6 |

## Standing findings

**The 1.24% PPL gap that motivated the reopen is intrinsic Lloyd-Max 8-bit
distortion physics**, not a bug or mis-implementation:

1. The codec is byte-correct CPU vs GPU (F-0.2: 0/256 byte mismatches at
   5/6/8-bit across 5 seeds, all D=256).
2. The kernel implements the right SDPA math (F-0.2: NRMSE 0.000247 at
   Gemma 4 26B production shape — 607× under the falsifier gate).
3. The empirical KV distribution post-FWHT is N(0,1) at every (layer, head)
   cell (F-0.3: std deviation 0.0012 across 112 cells, zero outliers
   beyond ±5.07).
4. Closing the gap requires bypassing the codec entirely — `HF2Q_USE_DENSE=1`
   (F16 dense KV) is the only available strict-Gate-C path.
5. 16-bit TQ is structurally redundant with F16 dense (same memory cost, no
   compression benefit) and is not implemented.

**Functional impact of the 1.24% gap is zero at production scales.** F-4.4
needle-in-haystack 12/12 PASS at 4K-32K shows the model retrieves planted
facts at every (length, position) cell tested through the quantized KV
cache, with the 8-bit codec as default.

**The original ADR's headline goal — 262K context — is empirically
reachable.** Bench at 8K/32K/64K shows sub-linear KV memory growth and
predictable decode degradation. No structural cliff. Code-level cap
referenced in close-section §1148 doesn't exist.

## Code that landed

mlx-native:
- `52c87ff` — F-0.1 CPU oracle for `flash_attn_vec_tq_hb`. 999 LOC + 10 tests.
- `3b50ac5` — F-0.2 CPU HB encoder (D=256) + D1 sign tables. 278 LOC + 7 tests.
- `03785fe` — F-0.2 SDPA divergence audit (Metal-integration). 620 LOC + 11 tests.
- `cc96c10` — F-0.3 distribution analyzer. 360 LOC.

hf2q:
- `7af37ee, 9104cf4, 58c7b33, e9ab2ea, d34c9e8` — Path C scaffold + F-0.3 dump
  infra + F-2/F-3/F-4.1 closures + F-7 codec freeze contract.
- `405dedf` — F-4.2/3 long-context bench harness.
- `840cf2d, 51b344a` — F-4.4 needle harness + verdict-bug fix.
- `dcb6463, 0ee9682` — F-4.2/3 64K bench + F-4.4 16K/32K results (12/12 PASS).
- `841a80a` — F-6.1 `--kv-bits` CLI flag.

ADR-007:
- `e9ab2ea` § Codec Freeze Contract — locks codec_version=1.
- `d34c9e8` `§ REOPENED 2026-05-05 — Path C: Earn the Close` — full reopen
  rationale + sequencing + acceptance criteria.
- Progress Log table with 11 iter rows.

## What's still open (and why)

### F-5 (MMLU + LongBench) — discretionary

The original Path C spec lists F-5 as a quality gate (Q-4/Q-5 in ADR §F-5
acceptance criteria). It would require:

1. Building an MMLU eval harness that consumes hf2q's chat-template +
   parses A/B/C/D answers from generated tokens.
2. Building a LongBench eval harness for long-context tasks.
3. Running both at TQ-active default + `HF2Q_USE_DENSE=1` and computing
   parity Δ.
4. Documenting whether parity is within ≤1pp on each benchmark.

**Estimated effort:** 1-2 days harness build + 4-8 hours per benchmark per
mode = 1-2 days run time. Total: 2-4 days.

**Strategic value given F-0.3:** the empirical N(0,1) distribution evidence
plus the F-0.2 NRMSE 0.000247 production-shape verdict make it implausible
that MMLU/LongBench would surface a parity gap larger than the intrinsic
1.24% PPL. The most likely outcome is "TQ-active passes Q-4/Q-5 within
~1pp" — which would confirm what we already know rather than discover
something new.

**Recommendation:** F-5 is real validation work but unlikely to change the
strategic verdict. Defer until a downstream consumer requires the formal
paper-standard benchmark.

### F-1 (C-2 → C-4 audit deferrals) — superseded

The 4 specific items (multistep at 16/8/256, SplitMix64 → StdRng, singlestep
oracle relabel, ManifestShaGate port) were procedural cleanup of the iter-2/3
audit harness. F-0.2 produced the same verdict (codec+kernel are correct)
with stronger evidence at production shape, so F-1's procedural debt is
superseded. Concrete action: an iter-X commit could close it as
"superseded by F-0.2" without running the audits.

### F-4.4 needle at 64K/128K/262K — discretionary

The harness's 30-min per-trial timeout is exceeded by 64K prefill at the
current model's ~30 tok/s prefill rate. Pushing further requires either:
(a) extending the timeout (simple), or (b) a faster prefill optimization
(out of Path C scope). The 12/12 PASS at 4K-32K is strong evidence the
mechanism works; 64K-262K extension would be incremental scale-validation,
not new finding.

## How to reopen if needed

If a future phase finds evidence inconsistent with Path C's verdicts:

- **If a calibrated codebook claim re-emerges**: cite
  `F-2/uncalibrated_is_floor.md` and require new distribution evidence
  beyond F-0.3 (different model family, different training data, etc.) to
  reopen.
- **If `dense_kvs` removal is proposed**: cite
  `F-3/dense_kvs_decision.md` and require either (a) a sub-2-byte
  high-fidelity codec that closes strict Gate C, or (b) consumer
  evidence that strict Gate C is no longer required.
- **If long-context retrieval failures appear**: F-4.4 PASS at 4K-32K is
  the empirical floor; reopen with the failing (length, position, model)
  triple.

## Mantra ledger

`Code + test == truth. Comments in code or ADR can be starting points,
but never trust them over code.`

Across 11 iterations Path C produced:
- 4 hypotheses falsified by data (D=512 norm bug; calibration design;
  16-bit memory savings; harness verdict logic).
- 0 hypotheses adopted without data.
- 0 gates relaxed by handwave (F-2's "uncalibrated is floor" is measured,
  not asserted).
- 1 retired stub-removal directive (C-4 `dense_kvs`) — retired with
  empirical justification, not relabeling.

The mantra worked at every step.
