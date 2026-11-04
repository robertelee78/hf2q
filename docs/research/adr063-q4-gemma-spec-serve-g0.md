# ADR-063 Q4-GEMMA-SPEC-SERVE: M0 round-cost curve and gate G0, result (2026-11-04)

Status: **G0 missed; the story is parked** (operator decision, 2026-11-04).
The windowed-verify code stays in draft PR #694. Q4's Gemma budget counts this
lever as 0 until the re-run below.

Design: `docs/research/adr063-q4-gemma-spec-serve-design.md` §8.1 (frozen
formula, D6 threshold 0.40 s). Issue #692.

## Verdict

| Head path | K\* | ΔD(K\*) | G0 (≥ 0.40 s) |
|---|---|---:|---|
| fused (production default) | 1 | **−0.841 s** | miss |
| unfused (`HF2Q_FUSE_LMHEAD=0`) | 1 | −0.066 s | miss |

**The verify alone misses.** At width 4, `tv(2,4) / t1(4) = 34.38 / 18.67 = 1.84`.
The limit is `c(1,4) ≤ 1.80`, so the verify is over it before any drafter,
save or restore cost.

Setting the drafter, save and restore all to zero, the projection is still
about **+0.26 s** [arithmetic], below 0.40 s. So a zero-cost drafter at L(1) = 1.881 also
misses on this verify route.

**Cause: the verify round does not amortize.** It costs about what the same
number of ordinary rows costs:
- one lane with a 2-row window costs 12.00 ms; two ordinary lanes cost 11.96 ms;
- 8 verify rows cost 34.38 ms; four ordinary lanes cost 18.67 ms.

This matches risk R1. MoE expert reads stay per row on the per-token `mv_id`
route, which is row-identical at any N.

## Class A (this artifact)

- Every verify row was bitwise equal to the teacher-forced ordinary ticks at the
  same positions. This held for window widths 2–4, with both head paths:
  - 9 rows at decoding width 1;
  - 36 rows at decoding width 4.
- The reference re-run after the ring restore was also bitwise equal.
- Two of the four lanes (lane 0 and lane 1) reach the wrapped sliding ring.

## Figures

All figures are fused head, medians of 200 ticks after 20 warm-up, in ms.

| n | t1(n) | tv(2,n) | tv(3,n) | tv(4,n) | ts(1,n) | tr(1,n) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 8.88 | 12.00 | 15.91 | 19.48 | 0.35 | 0.26 |
| 2 | 11.96 | 18.42 | 27.97 | 35.52 | 0.54 | 0.31 |
| 3 | 14.74 | 26.50 | 49.12 | 65.44 | 0.55 | 0.32 |
| 4 | 18.67 | 34.38 | 63.55 | 85.87 | 0.55 | 0.31 |

- `td_bound(1, n)` = 1.35 ms at n = 1 and 3.53 ms at n = 4.
- At K = 1, save plus restore exceeds 3% of `tv`, so the batched restore variant
  would be needed.
- K ≥ 2 figures are biased high: past 8 rows, each row runs its own m = 1 pass.

Round-cost ratio c(K, n) and projected saving ΔD:

| Head | K | c(K,1) | c(K,3) | c(K,4) | ΔD (s) |
|---|---|---:|---:|---:|---:|
| fused | 1 | 1.548 | 2.027 | 2.067 | −0.841 |
| fused | 2 | 2.145 | 3.752 | 3.824 | −3.885 |
| fused | 3 | 2.702 | 5.050 | 5.214 | −4.974 |
| unfused | 1 | 1.584 | 1.864 | 1.903 | −0.066 |
| unfused | 2 | 2.071 | 3.321 | 3.498 | −2.779 |
| unfused | 3 | 2.624 | 4.371 | 4.562 | −3.220 |

## Inputs

**Time-weighted width histogram h(n).** Taken from one speculation-off Q5 run
(`Q4G-H.json`, verdict CalibrationPass, makespan 14.43 s) by scraping
`hf2q_gemma_decode_tick_seconds_total` from `/metrics`:

| Width n | Ticks | Seconds | h(n) |
|---|---:|---:|---:|
| 1 | 23 | 0.218 | 0.0271 |
| 3 | 1 | 0.017 | 0.0021 |
| 4 | 397 | 7.788 | 0.9708 |

The ticks account for all 1,614 completion tokens.

**Lanes.** Recorded requests from the #660 L1 run `001-b01-hf2q`, rendered with
the production template and tokenizer. Each matches its recorded prompt length.

| Lane | Request | Prompt tokens | pos0 |
|---|---|---:|---:|
| 0 | agent-0 turn-4 | 1222 | 1218 |
| 1 | agent-1 turn-5 | 1372 | 1368 |
| 2 | agent-2 turn-2 | 707 | 703 |
| 3 | agent-3 turn-3 | 927 | 923 |

**Frozen model.** α = 0.881 (L(3) = 3.34 → L(1) = 1.881), D = 9.203 s, and
`E[tr] = (1 − α^(K·n)) · tr`.

## Identities

| Item | Identity |
|---|---|
| hf2q source | b72331f837406fd03e30036e7fc833fec44fd342 |
| mlx-native | local main 2140b4dc via path override, **unpublished**. Its Cargo version string reads 0.15.18; the restore primitive ships in 0.15.19 |
| Serve binary sha256 | ccfb1e1fe54c52762cb931398f22931d9b5d997c94452d9573566e9435d61071 |
| M0 test binary sha256 | 285c0c3f5e45b725da34b7523cb7db0f333d63969ea7957a30499a09a236106d |
| Target model | bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53 |
| Embedded metallib | ca3425a9290659e9fbc860519a5c0d9a8ff9fc7f6d822e7249dfc8acd1c038d6 (precompiled) |
| M0 case sha256 | dae9ff982b4a274ba148bf4f3898474a1b777cd0a84f7595ee5443e6634ed57f |
| M0 receipt sha256 | 211f3de258b25e55ee199dc0625103629b5269f7bbdc05482ebec580eff77a62 |

Evidence: `~/.local/state/hf2q/v0130/q4-gemma-spec-serve/`.
- `m0-20261104/` holds `case.json`, `run.log`, `out/m0-round-cost.json` and
  `RESULT.md`.
- `q5-spec-off-20261104T232551Z/` and its `.metrics.jsonl` are the h(n) run.

The M0 harness ran as a test binary outside the Q5 supervisor; only the h(n)
run carries a CalibrationPass.

## Re-run condition (operator decision)

1. When Q4-GEMMA-DECODE's bitwise expert-grouped `mv_id` (E1) lands, re-run
   the same M0/G0 with the same harness, case lanes and frozen formula. Refresh
   h(n) from a new speculation-off Q5 run of that binary.
2. **If G0 passes**, build the drafter (design §4–§5.2), then do M1 and S1.
3. **If G0 misses again**, delete the windowed verify, the rank-phase code and
   the pre-image code from hf2q in a clean cutover (D1). The mlx-native restore
   primitive stays.

Re-run command:
`HF2Q_GEMMA_SPEC_M0_CASE=<case.json> cargo test --release --bin hf2q gemma_spec_round_cost_curve -- --ignored --nocapture --test-threads=1`
