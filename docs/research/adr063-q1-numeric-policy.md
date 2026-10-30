# ADR-063 Q1: numerical-invariance policy and per-family closure plan

Status: proposal for root approval, 2026-10-30. Story: Q1 ([#367](https://github.com/robertelee78/hf2q/issues/367)).
It resolves the `ExplicitInconclusiveNumericPolicy` gap left by the
[Q1 execution plan](adr063-q1-execution-plan.md) ("Numerical policies"). The
operator's per-family closure rule (ADR-063, "Per-family closure and release
scope", PR #462) applies: Q1 closes per family on independently reviewed
hardware evidence, and Q1 no longer gates other stories.

The thresholds below are frozen in code as `POLICY` in
[`scripts/four_agent/q1_decision_policy.py`](../../scripts/four_agent/q1_decision_policy.py)
(`hf2q.q1.numeric-policy.v1`). Root approval freezes them before any
candidate data is collected. Changing a threshold later requires a new policy
version, applies only to new data, and never rescores old runs.

`$E` is `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`.
Offline numbers in this document were computed with that script from retained
stores (outputs: `/tmp/adr063-q1-decision-envelope-v1/*.json`). Three Gemma
breakdowns came from one-off read-only reads of the same stores: per-layer
CacheKey, rows 0–28, and the routing-gap distribution from
`q1-gemma-routing1511-t0-on-actual-v1/output/routing-witness.jsonl`. Root can
reproduce them before relying on them. No model was run for this document.

## 1. Policy summary

Three comparison classes, each with its own pass rule:

| Class | What is compared | Rule |
|---|---|---|
| A. Same-route invariants | hf2q against itself with the same arithmetic: resume vs cold at the same chunk boundaries, cancel+restore, request order, slot index, repeat runs, and batched routes hf2q declares exact | **Bitwise.** Every captured byte (logits, cache, recurrent state, token IDs, expert IDs, cursors) equal. One differing byte is `DivergedExactInvariant`, a failure. No tolerance. |
| B. Declared alternate routes | hf2q against itself where arithmetic is legitimately reordered: prefill chunk size, M1 vs multi-row verifier, batched vs serial where not declared exact | **Decision envelope.** For every logit row over identical frozen inputs, decision error D ≤ 0.25 nat (defined in §3.2). Otherwise `RouteSensitive`, which must be localized and blocks any "numerically neutral" claim for that route. Element-wise metrics are recorded, never gated. |
| C. hf2q vs the peer | Different engines and representations (TQ8 vs F16 KV, other kernels) | **Never bitwise.** First divergence is classified `NearTie` / `NumericSuspect` / `Inconclusive` from logprobs (§3.4). Task quality is judged statistically over ≥ 24 distinct, paired task instances (§3.5). |

A Class A candidate (for example DeepSeek joint decode) inherits task quality
from stock: identical inputs give identical outputs. Its A/B runs only need
transcript equality (§3.6), not a new quality campaign.

## 2. Evidence to date

### 2.1 Qwen3.8 27B (dense; artifact `cf0eb010…`, native 0.15.7)

| Comparison (retained stores) | Class | Result |
|---|---|---|
| Cancel at 128 → production recovery to 64, before vs restored state (`q1-qwen38-recovered-continuation-native0157-actual-v1`, `before/` vs `recovered/`) | A | `ExactEqual`, 1,408 records |
| Chunk-matched cold 64+64 vs cancelled-then-recovered 64+64, separate processes (`q1-qwen38-chunkmatched64-matched-actual-v1` vs recovered) | A | `ExactEqual`, 3,136 records: all 320 logit rows plus full active cache at 128/144 |
| T5 first fork: fresh-arm common position 129, target and MTP state; draft+discard and MTP catch-up side effects (`/tmp/adr063-q1-t5-first-fork-offline-result-review-v1.md`) | A | Exact: 352/352 target, 16/16 MTP, 368/368 before vs after discard, outputs unchanged by catch-up |
| Cold All128 vs 64+64 (chunk size), rows 64–143 | B | `WithinDecisionEnvelope`: top-1 318/320; D p50 0.0045, p95 0.015, max 0.028 nat; KL max 1.0e-4 nat. The 2 flips sit on baseline margins of 0.0044 and 0.00084 nat |
| T5 forced K3: ordinary M1 vs R4 verifier rows, committed rows 128–143 | B | `WithinDecisionEnvelope`: top-1 64/64; D p95 0.028, max 0.093 nat; KL max 3.3e-4 nat |
| Scalar vs R4 target output, row 129 (first fork) | B | Logits max-abs 0.0914. First differing captured point: layer-0 post-attention residual (DeltaNet). Cause unknown (target-layer review) |
| Serving transcripts, 3 runs of one binary (`qwen38-matched-baseline1512-*`) and baseline vs shared-forward ON vs OFF (`qwen38-shared-production-*`) | A | `ExactTranscripts`: 0 same-request/different-output in 52 groups each |

Element-wise metrics mislead. The Rust comparator reports max-abs 5.71 for the
T5 pair and 0.204 for the chunk pair. Both maxima are in `CacheValueNorm`,
yet the decision error never exceeds 0.093 nat.

Both Q1 Qwen workloads are synthetic. Prompts are vocabulary IDs 1000..1127
per lane; T5 adds 16 model-chosen tokens. 159 of 320 continuation rows have
a baseline top-1 margin below 0.25 nat.

### 2.2 Gemma 4 26B-A4B (artifact `bf65b7e1…`)

Endpoints T0–T3 (source `190abe5b`, native 0.15.8) used workload `922bec…`:
29 template tokens, then token 496 repeated 115 times. The Gemma
greedy-preparation note (branch `test/q1-gemma-greedy144-preparation`,
`docs/research/adr063-q1-gemma-model-assisted-greedy-preparation.md`) already
records that this tail is not a greedy continuation.

| Comparison | Class | Result |
|---|---|---|
| T1 resume at 64 vs T3 cold 64+64 | A | `ExactOnCommon`: 8,136 common records equal; T1's 2,520 extra restore records were exact-checked inside T1 (43,479,040 B) |
| T0 OFF vs T0 ON routing witness, two processes, accepted greedy144 dataset `842768` (source `9df42ba0`, native 0.15.11) | A | `ExactEqual`, 5,616 records (788,765,696 B); instrumentation does not interfere |
| T0 All128 vs T1/T3 64+64 | B | `RouteSensitive`: top-1 391/576; D p50 0.83, p95 20.0, max 36.8 nat; KL p95 8.7 nat. Rows 0–28 (real template prefix): 6/116 flips, D up to ≈6.7 nat |
| T0 serial vs T2 packed four-sequence prefill and batched decode | B | `RouteSensitive`: top-1 362/576; D p95 20.2 nat; KL p95 8.2 nat. Rows 0–28: 17/116 flips |
| T0 vs T1 CacheKey by layer, agent-0 | B | Rows < 64 are bitwise through layer 5, then diverge (relative RMS 0.0046 at layer 6 → 0.048 at layer 29). Rows ≥ 64 diverge from layer 1 (0.0021) up to 0.43 at layer 29 |
| Routing witness on the greedy144 T0 (17,280 rows, top-8 of 128 experts) | — | Gap between 8th and 9th softmax probability: median 7.5e-4, p5 5.1e-5, p1 1.0e-5. 171 rows (1.0%) below 1e-5, 15 below 1e-6. Relative gap p1 6.2e-4 |
| T5 n-gram transaction probe v3 | — | Accepted and rejected transitions observed on all four lanes. No ordinary-vs-speculative numeric comparison yet |

**[INFERENCE]** The thin routing gaps explain why Gemma amplifies small route
differences far more than dense Qwen3.8. A relative perturbation of about 1e-3
in router inputs can change about 1% of expert selections, and each change
alters the hidden state. The degenerate 922bec tail magnifies this further. So
Gemma's alternate-route envelope must be re-measured on the accepted greedy144
dataset before any Gemma chunk or batch route is called neutral. The 922bec
numbers are kept as a diagnostic, not used for calibration.

### 2.3 DeepSeek-V4 Flash

The Q1 adapters exist (`deepseek_actual`, `deepseek_localizer` on
`integrate/adr063-q1-deliverable`). Only metadata preparation has run
(`q1-deepseek-metadata-prepare-root-execution-v1`,
`q1-deepseek-endpoint-preparation-actual-v1`). The bitwise proofs come from
the decode work:

- `official_artifact_joint_decode_matches_serial_decode_bitwise` (#458) passed on 2026-10-30: 140 steps over eight ragged lanes, with state, logits and cache bitwise equal to serial decode.
- In the #458 A/B (`$E/ds4-joint-ab-v1`, 4 serial + 4 joint runs), `ExactTranscripts` held: 0 same-request/different-output in 112 groups. All 32 agent instances completed the protocol (awaiting independent patch review).

### 2.4 Qwen3.6 35B-A3B (MoE)

Q1 endpoints: the active64 adapter is built (`q1-qwen36-active64-root-build-v1`);
nothing has executed. Task quality: the hf2q N4 coding task fails (agents 1–2
write E0282 code; PR #461), while the peer passes 4/4
(`peer-qwen36-e86-none-normal-n4-20261027T185853Z`).

Comparing TQ8 against F32 KV (`q5-qwen36-fix-20261030T044749Z` vs
`q5-qwen36-tqkv0-20261030T045814Z`) gives `DivergedTranscripts` for every
agent within turns 0–2. That is expected: it is a Class C representation
change. No logprob margin has been captured, so "near-tie" is not yet a
measured fact. `/tmp/adr063-qwen36-divergence-plan-v1.md` is the measurement
design.

### 2.5 What remains, by family

| Family | Class A still missing | Class B still missing | Quality |
|---|---|---|---|
| Qwen3.8 | T4 order, T7/T8 at endpoint level (serving transcripts already show cross-run determinism) | — (chunk and verifier rows measured) | Peer arm never run; hf2q 4/4 in every recorded cohort |
| Qwen3.6 | T1, T3, T4, T6, T7 endpoints | T0 vs T3 chunk, T5 | Fork classification (§3.4), then the paired campaign (§3.5) |
| DeepSeek-V4 | T1 vs T3, T4, T7 endpoints (adapter exists); T6 needs its separate variant | T0 vs T3 chunk (if chunk sizes differ in production) | No peer adapter: floor-only verdict, A/B vs stock for candidates |
| Gemma 4 | T4, T6, T7 on greedy144 | T0 vs T3 and T0 vs T2 on greedy144; localize the layer-5/layer-1 onsets | No current-binary F2 rows; no peer adapter |

## 3. Policy

### 3.1 Class A: bitwise, same route

Must be byte-identical, across processes as well as within one:

1. **Cold vs resume at identical chunk boundaries** (T1 vs T3); **cancel then restore** vs never-cancelled (T6 vs T3); snapshot then restore round-trips.
2. **Request order and slot index** (T4 vs T0, joined by agent identity), and any lane vs the same lane run alone, as long as the per-lane graph is the same.
3. **Repeat determinism** (T7/T8 vs T0): fresh process, same binary, inputs and config.
4. **Batched routes hf2q declares exact**: DeepSeek joint decode, the B=4 cohort and cooperative warm prefill; the Gemma batched decode body (ADR-040 S2/S3 byte-identity); Qwen four-slot per-slot parity. A route that is not proven exact must refuse or fall back. `joint_decode_routes.rs` does this for unproven storage types, and that pattern is required.
5. **Speculation side effects**: draft generation and discard must not change committed state, and MTP catch-up must not overwrite committed outputs.
6. **Integer decisions** within one route: token IDs, expert IDs and order, cursors, cache metadata.

Any byte difference fails. A same-route flip is never a "near-tie"
explanation. ADR-040 recorded a 13% "greedy flip" in
`slot_aware_staggered_eviction_no_peer_perturbation`. It turned out to be a
real stale-KV corruption bug (ADR-040 lines 561–630), not numerics.

Tooling: `q1_decision_policy.py exact` checks all stages; the Rust comparator
(`compare.rs`) checks with `NumericPolicy::Exact`; `q1_decision_policy.py
transcripts` checks serving runs.

### 3.2 Class B: decision envelope for declared alternate routes

**Statistic.** Over identical frozen inputs, take each logit row L (baseline)
and R (alternate). Let a be L's top-1 token. Let c be R's top-1 if it differs
from a, otherwise L's second token. Define

  D = |(L[a] − L[c]) − (R[a] − R[c])|  (nats, from raw logits)

D measures how far the route moved the decision boundary the output actually
depends on. A top-1 flip at a row with D ≤ τ is by construction a flip
between two tokens within τ of each other on both sides: a near-tie.

**Threshold.** τ_route = **0.25 nat**:
- The largest observed benign alternate-route error is 0.093 nat (Qwen3.8, M1 vs R4 verifier). The chunk-size pair maxes at 0.028. 0.25 gives 2.7× headroom over the largest of these.
- Gemma's real-prefix rows reach D ≈ 6.7 nat across routes, so the threshold separates the two regimes observed.
- It equals the fork noise ceiling (§3.4) and is half the `NumericSuspect` floor.

**Verdicts.** `ExactEqual`, `WithinDecisionEnvelope` (all rows D ≤ τ), or
`RouteSensitive` (some row D > τ), plus `Incomplete` if the logit row sets
differ. Never compare a subset.

`RouteSensitive` is a valid Q1 diagnostic outcome. It must carry a localized
first divergence from the existing stage localizers. It forbids calling the
route neutral, and any candidate that moves production onto that route needs
the paired quality campaign against stock (§3.5).

**Element-wise metrics** (max-abs, RMS, relative RMS, cosine from the Rust
comparator) stay recorded with disposition `ExplicitInconclusiveNumericPolicy`.
No model-level `SourceQualified` element tolerance is adopted, because these
metrics do not predict decisions (§2.1: 5.71 max-abs with 0.093 nat D). The
ADR-027 NRMSE ≤ 0.15 remains a narrow SDPA falsifier, not a model tolerance.

**Inputs.** Class B calibration and verdicts use the accepted greedy datasets
(Gemma `842768`) or real agent prompts. The synthetic Qwen vectors and the
Gemma `922bec` tail are diagnostic only.

### 3.3 MoE routing

- Same route: expert IDs and order are exact (Class A). An injected valid-index swap must be caught; `q1_routing_id_swap_cannot_pass_under_float_tolerance` covers this.
- Alternate route or peer: record every changed selection with its kth/next margin in the family's selection domain. That is softmax-before-top-k for Gemma (from the witness), and biased sqrt-softplus for DeepSeek scored layers once the native witness is published. Flips are never smoothed and tie thresholds never changed.
- Dense Qwen3.8 records routing as N/A.

### 3.4 Class C: hf2q vs the peer at the first divergence

Inputs: both engines' non-streamed responses with `logprobs: true,
top_logprobs: 20` on byte-identical prompt token IDs (verified first; an ID
mismatch is an input bug, not numerics). Speculation is off in the logprob
arm; the auto arm is compared separately. Let k be the first byte-differing
generated token, a = hf2q's token, b = the peer's. Then:

- D = |(lh[a] − lh[b]) − (lp[a] − lp[b])|.
- E is the 95th percentile, over shared positions i < k and token pairs inside both engines' top-5, of the same pairwise difference.
- `NearTie` if {a, b} is the top-2 in both engines and D ≤ max(2E, 0.10 nat).
- `NumericSuspect` if b is outside hf2q's top-5, a is outside the peer's top-5, D > max(4E, 0.5 nat), or E > 0.25 nat.
- `Inconclusive` otherwise. It is resolved by the F32-KV hf2q arm and, if still open, by layer localization.

These are the divergence plan's numbers. The 0.10 floor sits just above the
largest internal alternate-route D (0.093): a peer disagreement no larger than
hf2q's own reordering noise is a near-tie. The tool reports whether each
engine's chosen token was its raw top-1, because repetition penalty is applied
after the logprob capture.

A `NearTie` makes a single-run task failure a coin flip. It is a quality
sample, not a numerics defect. A `NumericSuspect` is a defect until localized.

### 3.5 Task quality under near-ties

The frozen four-agent prompts are identical on every run: the agent identity
text omits the run ID (`qwen_n4_live_smoke.identity`), and the DeepSeek A/B
turn-0 requests are byte-identical across all 8 runs. So repeated cohorts of a
deterministic engine are one sample per agent, not independent trials.

Quality sampling therefore needs distinct, semantically null prompt variants.
That requires one harness change: an explicit instance label in the
identity/system text, frozen per campaign and identical across arms.

**Design.** 24 paired instances per family: 6 instance labels × 4 agents.
Every instance runs once on the subject arm and once on the reference arm
(the peer, or stock hf2q when judging a candidate). Production sampling and
the existing task oracle apply (`ProtocolComplete…` plus independent patch
review). No reruns or extra instances after seeing data.

**Verdict** (`q1_decision_policy.py paired`). Let b = subject-only failures
and c = reference-only failures.

| Verdict | Condition |
|---|---|
| `Insufficient` | n < 24 |
| `Loss` | exact one-sided sign test P(X ≥ b \| X ~ Bin(b+c, ½)) < 0.05 |
| `BelowFloor` | subject passes < 75% of instances |
| `Inconclusive` | b − c > ⌊n/8⌋ (3 of 24) without reaching `Loss` |
| `Pass` | otherwise |

With no peer arm (DeepSeek, Gemma), the baseline verdict is the 75% floor
only. The current Qwen3.6 cohort (2 failing agents vs 0 for the peer) is
b = 2, c = 0, p = 0.25. One cohort cannot distinguish a near-tie coin flip
from a real loss, which is why the paired campaign is needed.

Power is limited, and here it is stated rather than hidden. The rule's
operating characteristics at n = 24, assuming independent instances, were
computed exactly with the tool:

| True pass rate, subject / reference | Pass | Loss | Inconclusive | BelowFloor |
|---|---:|---:|---:|---:|
| 1.00 / 1.00 | 1.00 | 0 | 0 | 0 |
| 0.95 / 1.00 | 0.97 | 0.006 | 0.024 | 0 |
| 0.90 / 1.00 | 0.79 | 0.085 | 0.13 | 0 |
| 0.80 / 1.00 | 0.26 | 0.54 | 0.20 | 0 |
| 0.70 / 1.00 | 0.04 | 0.89 | 0.07 | 0 |
| 0.90 / 0.90 | 0.95 | 0.010 | 0.031 | 0.005 |
| 0.80 / 0.90 | 0.65 | 0.12 | 0.12 | 0.11 |

A 20-point deficit fails to pass about three times in four; a 10-point
deficit usually passes. Detecting smaller gaps needs a larger, separately
frozen n.

### 3.6 Class A candidates inherit quality

If a candidate's changed route passes Class A (bitwise against stock), the
task-quality obligation against stock is met by `transcripts` returning
`ExactTranscripts` over its A/B runs. Any same-request/different-output pair
voids the exactness claim. The DeepSeek joint decode A/B already meets this
(§2.3).

### 3.7 Never allowed

Routing or logit smoothing, tie-threshold changes, approximate recurrent
replay, disabling TQ8 or raising precision to pass, comparing subsets of an
extent, retry-until-pass, and learning a tolerance from candidate data.

## 4. Existing tests that prove the bitwise parts

Families: DS = DeepSeek-V4, G = Gemma 4, Q = Qwen 3.x. "Ignored" means the
test needs a real artifact or Metal and runs only on root hardware.

| Test (file) | Family | Proves | Gate |
|---|---|---|---|
| `official_artifact_joint_decode_matches_serial_decode_bitwise` (`deepseek4/real_artifact_joint_decode_tests.rs`, #458) | DS | Joint decode = serial: state, logits, cache; 140 steps, 8 ragged lanes | Ignored, `HF2Q_DEEPSEEK4_GGUF`; passed 2026-10-30 |
| `official_artifact_b4_decode_body_is_exact_and_measured` (`deepseek4/real_artifact_decode_cohort_tests.rs`) | DS | B=4 cohort decode = serial | Ignored, `HF2Q_DEEPSEEK4_GGUF` |
| `official_artifact_cooperative_warm_prefill_is_exact_and_faster` (`deepseek4/real_artifact_cohort_tests.rs`) | DS | Cooperative packed prefill = serial: prefill state, logits, next decode | Ignored, `HF2Q_DEEPSEEK4_GGUF` |
| `q8_0_independent_rows_match_one_row_matvec_bitwise`, `dense_f32_independent_rows_match_one_row_matvec_bitwise`, `q8_0_grouped_output_rows_match_one_row_groups_bitwise` (`deepseek4/row_matmul_tests.rs`, #458) | DS | Primitive multi-row = one-row, bitwise | Hosted Metal |
| `official_artifact_types_admit_joint_decode`, `float_output_head_falls_back_to_serial`, `unproven_block_types_fall_back_to_serial` (`deepseek4/joint_decode_routes.rs`, #458) | DS | Unproven routes refuse joint decode | Hosted |
| `compact_snapshot_restores_overwritten_window_state_and_position_without_aliasing` (`deepseek4/cache_tests.rs`) | DS | Snapshot/restore restores exact values | Hosted Metal |
| `deepseek_image_exact_codec_matches_independent_old_format_golden` (`deepseek4/cache/anchor_image/serialize_tests.rs`), `anchor_c1_deepseek_*` (`c1_anchor_layout_tests.rs`) | DS | Anchor image codec and roles | Hosted |
| `slot_aware_n4_per_slot_parity_vs_serial`, `slot_aware_n8_per_slot_parity_vs_serial`, `slot_aware_n1_byte_equivalent_to_serial_slot_aware`, `slot_aware_per_slot_kv_offset_isolation`, `slot_aware_staggered_eviction_no_peer_perturbation`, `slot_aware_n4_batched_body_same_prompt_parity_vs_serial` (`serve/api/engine.rs`) | G | Per-slot output bytes = serial, including batched body and eviction churn | `HF2Q_BYTE_EQUIV_E2E=1`, `HF2Q_BYTE_EQUIV_E2E_GGUF` |
| `iter_g_a_bf16_determinism_isolation_gate`, `iter_g_a_multi_seq_prefill_first_token_isolation` (`serve/api/engine.rs`) | G | Multi-sequence prefill: run-to-run determinism and per-sequence isolation of the first token | Same gate |
| `gemma_n8_decode_then_tiny_cold_prefill_is_repeat_invariant` (`serve/api/engine_gemma4_tiny_prefill_tests.rs`) | G | Width-8 decode then tiny cold prefill = N=1 references | Real Gemma GGUF |
| `slot_aware_qwen35_n8_per_slot_parity_vs_serial`, `qwen35_slot_aware_engine_n1_parity` (`serve/api/engine.rs`) | Q | Per-slot output bytes = serial (MoE) | `HF2Q_BYTE_EQUIV_E2E=1`, `HF2Q_QWEN35_E2E_GGUF` |
| `qwen35_hybrid_kv_byte_identical_at_slot_0_n_seqs_4_vs_1`, `snapshot_prefix_compacts_sequence_buffers_and_restores_exactly`, `slot_anchor_rewinds_only_target_cursor_and_linear_state`, `qwen38_spec_reject_rewinds_only_target_slot_ping_pong` (`qwen35/kv_cache.rs`) | Q | Slot layout, snapshot restore, rewind isolation | Hosted Metal |
| `qh35_v5_tq_only_with_mtp_round_trip_byte_exact` (`serve/kv_persist/families/qwen35_hybrid_persistor.rs`), `tq_v2_capture_restore_byte_identity` (`serve/kv_persist/families/tq_packed.rs`) | Q | Persisted TQ8/MTP state round-trips byte-exact | Hosted |
| `anchor_c1_*` (`inference/anchor_contract/tests.rs`, `qwen35/kv_cache/c1_anchor_layout_tests.rs`, `gemma4/c1_anchor_budget_tests.rs`) | all | Atomic anchor publication and stale-state refusal (contract, not numerics) | Hosted |
| Q1 oracle controls on `integrate/adr063-q1-deliverable`: `q1_routing_id_swap_cannot_pass_under_float_tolerance`, `q1_corrupt_payload_or_missing_coverage_is_not_numeric_divergence`, `q1_store_refuses_padding_overflow_duplicate_and_unchecked_capture`, `q1_acquisition_missing_same_nonce_or_unreviewed_provenance_refuses`, the `raw_audit_*` refusals | all | Comparator positive and negative controls | Hosted |

Not bitwise; these are Class B evidence:

- `qwen38_real_four_position_normal_forward_parity` (`qwen35/spec_decode.rs`, ignored) checks greedy decisions only (M4 vs 4×M1).
- `delta_net_layer_chunked_equals_monolithic` (`qwen35/delta_net.rs`) is a CPU test with a 1e-5 tolerance.

## 5. Minimal per-family measurement plan (root only)

### 5.0 Common, model-free

```sh
# Policy tool controls (any checkout containing this change)
python3 -B -m unittest discover -s scripts/four_agent -p 'test_q1_decision_policy.py'

# Q1 oracle controls (Q1 integration checkout rebased on main; root builds)
cargo test --locked --bin hf2q adr063_invariance -- --test-threads=1

# Reproduce §2 from retained stores (read-only)
E=/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023
T=scripts/four_agent/q1_decision_policy.py
python3 -B $T exact    $E/q1-gemma-t1-endpoints-actual-v1/output/trace.json $E/q1-gemma-t3-endpoints-actual-v1/output/trace.json
python3 -B $T envelope $E/q1-qwen38-cold-continuation-native0157-actual-v1/output/continuation/trace.json \
                       $E/q1-qwen38-recovered-continuation-native0157-actual-v1/output/continuation/trace.json
python3 -B $T transcripts $E/ds4-joint-ab-v1/{a1-20261030T042036Z,a3-20261030T042752Z,a4-20261030T043042Z,a5-20261030T043754Z} \
                          $E/ds4-joint-ab-v1/{b1-20261030T042326Z,b2-20261030T042540Z,b3-20261030T043330Z,b4-20261030T043541Z}
```

Hardware runs use the existing ignored filters on a test binary built from
`integrate/adr063-q1-deliverable` rebased on main, plus the Gemma
greedy144/routing-witness branches. Root first lists the filters
(`TEST_BINARY --list --ignored | grep q1_`), then binds hash-reviewed packets
and runs each case under the existing Q5 supervisor, one model at a time:

```sh
<ENV>=<packet.json> TEST_BINARY <filter> --exact --ignored --nocapture --test-threads=1
```

### 5.1 Qwen3.8

1. **Class A, T4 + T7:** `HF2Q_Q1_QWEN_PACKET`, `adr063_invariance::qwen_actual::q1_qwen_actual_endpoint_trace`, trajectories `T4ReverseOrder` and `T7RepeatCold` against a fresh `T0ColdSerial`. Check with `exact`; expect `ExactEqual`. That is 3 model processes.
2. **Class C + quality:** run the peer N4 arm and the hf2q arm on the 24-instance paired design, then `paired`. T2 stays the source-proved refusal (independent request handles).

### 5.2 Qwen3.6

1. **Fork first** (no source change): follow `/tmp/adr063-qwen36-divergence-plan-v1.md` steps 0a–2, then run `q1_decision_policy.py fork $O/hf-lp-hf-t0.resp.json $O/peer-t0.resp.json` (and the same for t2). `NearTie` goes to step 3; `NumericSuspect` goes to plan step 4 (localize).
2. **Class A, T1/T3/T4/T7:** same filter as 5.1 with the Qwen3.6 packet (the active64 build exists). Check T1 vs T3, T4 vs T0 and T7 vs T0 with `exact`.
3. **Class B, T0 vs T3:** `envelope`.
4. **Quality:** the 24-instance paired campaign against the peer, then `paired`. This, not a single cohort, decides the E0282 question.

### 5.3 DeepSeek-V4

1. **Class A:** `HF2Q_Q1_DS_PACKET`, `adr063_invariance::deepseek_actual::q1_deepseek_actual_endpoint_trace` for `T0ColdSerial`, `T1Resume64`, `T3Chunk64`, `T4ReverseOrder` and `T7RepeatCold` (5 processes; the preparation packet already exists). `exact` must give T1 vs T3, T4 vs T0 and T7 vs T0 equal. T0 vs T3 goes through `envelope`.
2. **Joint decode (S3):** already proven by the bitwise test and `ExactTranscripts`. Rerun `transcripts` on any new A/B.
3. **Quality:** no peer adapter, so the verdict is floor-only over the 24 instances, from the hf2q arm of any A/B (Class A arms are identical by construction).

### 5.4 Gemma 4

1. **Class B on accepted inputs:** `HF2Q_Q1_GEMMA_ENDPOINT_PACKET` with the greedy144 workload (`842768`), filters `adr063_invariance::gemma_actual::q1_gemma_actual_ordinary_endpoint_trace` (T0/T1/T3/T4/T7), `…::q1_gemma_actual_cohort_endpoint_trace` (T2) and `…::q1_gemma_actual_cancel_endpoint_trace` (T6). That is 7 processes at about 45 s each, per the routing-witness receipts.
2. **Checks:** `exact` on T1 vs T3, T6 vs T3, T4 vs T0 and T7 vs T0; `envelope` on T0 vs T3 and T0 vs T2.
3. **Localize any `RouteSensitive` pair** with `adr063_invariance::gemma_localizer::q1_gemma_actual_stage_localizer` (`HF2Q_Q1_GEMMA_LOCALIZER_PACKET`). Start at the layer-5 onset for rows < 64 and the layer-1 onset for rows ≥ 64. Record routing flips with their witness margins.
4. **Quality:** floor-only (no peer) on 24 instances; any candidate that changes chunk or batch routing needs the paired A/B against stock.

### 5.5 Per-family closure checklist

A family's Q1 closes when an independent reviewer accepts all of:

- (a) oracle controls pass;
- (b) every applicable Class A entry is `ExactEqual` or `ExactOnCommon` with explained extra records;
- (c) every Class B entry is classified, and each `RouteSensitive` one carries a localized first divergence;
- (d) routing margins are recorded where a witness exists, or explicitly N/A or Missing;
- (e) a task-quality verdict exists: `paired` `Pass`, `transcripts` `ExactTranscripts` for Class A candidates, or a recorded `Loss`/`BelowFloor`/`Inconclusive`. A failure verdict is a valid closing record for Q1; it is not a pass for Q3/Q4.

Unsupported axes are listed with their source refusal, never filled with
substitutes.
