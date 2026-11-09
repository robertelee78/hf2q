# ADR-065 E8: G0 decision

Story: [#511](https://github.com/robertelee78/hf2q/issues/511). Governing decision: [ADR-065 E8](../adr/ADR-065-four-agent-joint-speculative-verify.md#e8-record-the-g0-gate-decision).

## Estimator pre-registration

This section and `scripts/four_agent/adr065_g0.py` were committed before the E2 full-grid receipt was produced or inspected. The estimator is fixed as follows.

- E3's MTP K3 accepted-prefix histogram supplies expected committed tokens per lane. A prefix of length $k$ commits $k+1$ tokens because the canonical bonus token follows the accepted prefix.
- The E2 near-2k `1x4`, `2x4`, `3x4` and `4x4` configurations match E4 active widths one through four. Each configuration's complete-round denominator is the median of its seven timed repetitions; its numerator is active width times E3's expected committed tokens per lane.
- E4 joint-turn duration at each active width supplies the configuration weights. Durations are summed across the retained E4 run clusters and normalized over widths one through four.
- B1's phase-1 baseline rate is the mean, over the 20 passing `n4` baseline run clusters, of completion tokens divided by makespan milliseconds. Every completion token was emitted by the joint scheduler; this is the only B1 token-rate estimand frozen before candidate measurement.
- $s$ is the E4-duration-weighted candidate token rate divided by the B1 baseline token rate.
- $f$ is the mean E4 eligible critical-path fraction. $S=1/((1-f)+f/s)$.

Uncertainty is a 100,000-replicate run-clustered percentile bootstrap with seed 650511. Each replicate resamples:

1. the seven E2 timed repetitions independently within each matching configuration and takes the resampled median;
2. the 20 B1 `n4` runs and takes the resampled mean token rate;
3. the three E4 runs, recomputing width-duration weights and mean $f$.

E3 has one retained run cluster. Run-cluster resampling therefore returns that one aggregate unchanged; the report must state that between-run E3 variation is not estimable rather than inventing proposal-level independence.

The decision is `Go` only when all three preregistered conditions hold: $s\ge1.15$, $S\ge1.05$, and the lower 95% bootstrap bound of $S$ is strictly above 1.0. Otherwise it is `NoGo`. E4 already measured $f=0$ in all three runs, so $S=1$ for every finite positive $s$; E2 and E7 remain required independent measurements and are not skipped.
