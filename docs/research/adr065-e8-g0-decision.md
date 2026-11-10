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

## Bound inputs and result

The preregistered estimator ran after the E2 report was finalized.

| Input | SHA-256 |
| --- | --- |
| E2 full grid | `82e7f4bbe2bc4558b8128cc7d2beb6f811e7624688db4ce75f708bad313e6021` |
| E3 acceptance | `d5120aae936b469b8c86c2af213366b4ca28c9fa08f870de3e42fd74fd1e16dd` |
| E4 critical path | `121fa2b4cdda00b34319792614007a094e14893de661c05f76a28c838c5f51ed` |
| B1 Qwen3.8 campaign | `1760b6394a4ff2c52b37383c0a9ed3e47f6e0be46a50908894b02f8a7280edbd` |
| G0 report | `1daa6257a4ae39328909ec901d0c985d958b45bf2fee554f0e1d348ccbcc608c` |

E3 gives 3.623974 expected committed tokens per active lane. B1's mean
baseline rate is 0.0333114 completion tokens/ms. E4's duration weights for
active widths one through four are 0.027937, 0.045410, 0.044337 and 0.882316.
The corresponding E2 candidate rates are 0.054207, 0.034445, 0.049763 and
0.066096 tokens/ms.

| Metric | Point | 95% bootstrap interval | Gate |
| --- | ---: | ---: | --- |
| Complete-round speed factor $s$ | 1.9093 | [1.7742, 2.0364] | $s\ge1.15$: pass |
| Eligible critical-path fraction $f$ | 0 | [0, 0] | input |
| Projected makespan speedup $S$ | 1.0000 | [1.0000, 1.0000] | $S\ge1.05$: fail; lower bound $>1$: fail |

**G0 decision: `NoGo`.** The prototype's isolated complete-round rate clears
the local $s$ threshold, but the canonical four-agent workload exposes no
eligible critical-path time. A faster operation that cannot run changes no
makespan: $S=1$ exactly.

Per ADR-065, J4 (#515), J5 (#516), J6 (#517), A1 (#518), A2 (#519), P3a
(#524) and P3b (#525) are not pursued. E2, E7 and every story outside that
explicit stop set stand on their own evidence.

An independent implementation re-read the four raw inputs without importing
the checked-in estimator, re-derived every point value and repeated the
100,000-replicate bootstrap. It matched all input hashes, $s$, $f$, $S$, the
$s$ interval, full-grid/repetition completeness and the `NoGo` verdict
exactly. Its retained review is `$E8/independent-review.json`, sha256
`1f4608a1298b3ae04b448eed0b9c3495a3c2467df60de4cda75f9177af017b06`.

## Reproduction

```sh
python3 -B scripts/four_agent/adr065_g0.py \
  --e2 "$E2/full/report.json" \
  --e3 "$E3/acceptance.json" \
  --e4 "$E4/critical-path-report.json" \
  --b1 "$B1/baseline-v3/qwen38/campaign.json" \
  --output "$E8/g0.json"
```

`$E8` is `/Users/robert/.local/state/hf2q/v0130/e8-511`. The report retains
all 27 E2 configuration medians, input identities, bootstrap parameters,
point estimates, intervals, threshold checks and the `NoGo` verdict.

## Supersession: production J4/J5 candidate

On 2026-11-10 the operator explicitly superseded the original E8 stop for the
production J4/J5 candidate and approved default promotion. This does not alter
the original G0 computation: its $f=0$ result correctly described a diagnostic
prototype that phase 1 could never schedule. The production candidate changes
that eligibility invariant. Every eligible lane joins a joint MTP step; batched
drafting and a single joint target forward replace the serial K3 round, and the
padded route uses the column-invariant few-row MMA projection.

The candidate was qualified default-off at source commit
`cdb180d523bae933359f7e5f5484701ad698ba9d`. The campaign uses the same binary
for A and B and changes only `HF2Q_QWEN_JOINT_MTP` from absent to `1`.

| Evidence | Result | SHA-256 |
| --- | --- | --- |
| Fixed-count campaign | Complete, 70/70 eligible runs | `a93266007a15b36f2c8563e212033903f37704443908aeeda3c1344b94be5f6d` |
| Case-bound fairness ceilings | Qwen3.8 baseline-v4 | `464fabf04e6efc87f1deebc82feea4ffc946c244e321fbee8f6f05f5cebca0a0` |
| Independent analysis | `Win`; no gate failures | `0f1a077bc5484cb190257ac5c46dd22a8d71f990dd6424dc157fd6841882f05c` |
| Candidate Q5 | `CalibrationPass` | `bfe99a65cc7d6379e3d22971d16e0c8ce3c715e5c252727313d9b3cacda1f89a` |
| Promoted unset-switch Q5 (`6adfbcde`) | `CalibrationPass`; joint MTP/MMA active; cleanup proven | `d2b61412de248476e0536ac746af3777c9dac2e77e43a2b5f2fb8b383b72cef0` |
| Cancellation Q5 | `CalibrationPass`; cleanup proven | `6514df591735d0d37e4aed5b14d604981f3ff5bafe4a142cae731678b54431d3` |
| Cancellation client | `CancelSmokePass` | `7dbd24d73b95ff633dc1d216c4ba81d46b6a9186599e80ca0a7434a221e44e4d` |

The primary n4 makespan reduction is 34.600%, with a 95% interval of
[33.738%, 35.452%] over six fixed blocks. Reported control reductions are
65.809% at n1, 36.554% at 3+1, 38.260% at 2+2 and 38.615% at 1+3. Every run is
Q5-passing, quality-eligible and thermally valid; no workload has a fairness
problem. In the cancel gate the active MTP width narrows from four lanes to
three after the close, the three survivors are byte-exact against control, the
victim is a prefix of control, the server remains ready and no error or poison
line occurs.

Decision: `Go` for J4/J5 and make the combined route the dense-MTP default.
`HF2Q_QWEN_JOINT_MTP=0` restores serial K3 and canonical projection. MoE and
artifacts without an MTP head remain off by default. The remaining stories in
the original E8 stop set are independent designs and receive no implied
approval from this supersession.
