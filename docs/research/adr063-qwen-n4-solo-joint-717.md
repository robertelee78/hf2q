# #717: repeatable Qwen3.8 four-agent serving (every ordinary decode step on the padded joint forward)

Status: accepted (Review717: REVISE with 7 findings, all applied; then ACCEPT). Owner: Qwen38DetInv. Issue: #717. Based on main after #719 (Qwen3.6 joint decode on by default) merged.

## Problem

Under the Qwen3.8 production routing policy (`HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1`) the SlotAware worker decodes each turn as follows (`src/serve/api/engine/qwen35_joint_decode.rs`, `partition_qwen35_joint_lanes`):
- the first `max_rows` eligible lanes run `forward_gpu_decode_joint`, packed to width 4 with `mul_mv_ext`;
- eligible lanes beyond `max_rows`, and every lane of a turn with fewer than two eligible lanes, run the serial one-row forward (`decode_batch_qwen35` → `decode_tick`).

The two forwards are not bitwise equal; the S3 contract accepts this as Q1 Class B. Whether a step of a request runs serially depends on when the other agents' requests arrive and finish. Identical requests therefore produce different transcripts when a step lands on a near-tie.

T-J1-8 shows this, and the replay (`~/.local/state/hf2q/v0130/q38-det-inv/campaign-20261105T214953Z/`) reproduces it deterministically, arm against arm:
- Agent-1 turn-4 gives the " and" transcript when 13 or fewer of its steps are joint, and the " to" transcript when 17 or more are.
- Token 16 is a near-tie: " and" −0.797 vs " to" −0.804 solo.

The padded joint row is already cohort-invariant. `qwen38_joint_decode_padded_mv_ext_rows_are_cohort_invariant_bitwise` proves that a row's bytes are the same alone and in any cohort of 2–4 rows, in any packed position. The serving serial path sits outside that proof: the gate's "alone" variant is the joint forward with one row, which serving never calls.

## Change

In padded mode, every eligible ordinary lane runs the padded joint forward, including a lone one.
- The eligible lanes of a turn are split, in scheduler order, into consecutive groups of at most `max_rows`.
- Each group runs one joint step (lanes sorted by slot, packed to 4 as every padded step already is).
- A group of one lane runs joint when that lane is ordinary (rule 3 below). Otherwise it stays on the serial loop, so a lone speculating lane keeps its K3 or history round.

Every ordinary decode step of a request then uses the same arithmetic whatever the cohort, the number of tenants and `max_rows`. That includes `max_slots >= 5` and `HF2Q_QWEN_JOINT_DECODE_MAX_ROWS=2|3`. The existing invariance gate covers exactly this path.

Scope: the change applies only when all of these hold.
1. **Padded mode.** The worker's joint mode is `JointRouteMode::PaddedMvExt`.
   - In exact mode (the Qwen3.6 default after #719, and the Qwen3.8 mvN arm) the joint row is bitwise equal to the serial one-token forward. Exact mode keeps today's partition: lanes beyond `max_rows` and lone lanes stay serial, and determinism holds by construction.
2. **Multi-slot worker.** The worker has `n_slots >= 2`. N1 runs with one slot (`ctx262144_slots1_verified`) and is unchanged.
3. **Ordinary lane (for a group of one).** The lane has no MTP state and no history-lookup state, so its serial tick would be the ordinary target forward. Groups of two or more admit any joint-ready lane, as today.
4. **Joint-eligible.** The lane passes the existing eligibility check: installed handle, open client, `joint_decode_ready`. No DeltaNet capture is active.
   - With speculation off, the only reasons a lane fails `joint_decode_ready` are per-request: step bound reached, empty ledger, cursor mismatch.
   - DeltaNet capture is only set inside K3 transactions.

Kill switch: `HF2Q_QWEN_JOINT_SOLO` (`documented-escape`).
- Unset or `1`/`on`/`true` enables the change; `0`/`off`/`false` restores today's partition: one group of at most `max_rows`, at least two lanes, leftovers serial.
- It is read once at worker start and logged with the joint-decode configuration as `solo`. Invalid values warn and keep the default.
- It is the A/B arm, and the escape hatch if a solo regression appears.

Code:
- `qwen35_joint_decode.rs`:
  - `Qwen35JointDecodeConfig` gains `solo: bool`, resolved from mode, `n_slots` and the switch by `qwen35_joint_decode_config(model, n_slots)`.
  - `partition_qwen35_joint_lanes` returns the joint groups plus the serial leftovers.
  - `run_qwen35_joint_decode` runs the groups one after another through a per-group function. The group body is today's body unchanged: take lanes, prepare, one forward, settle.
- `engine_qwen35/joint_decode.rs`: `Qwen35DecodeState::ordinary_target_only()`.
- `engine.rs`: pass `n_slots` at the worker call site.
- No model or kernel change. `forward_gpu_decode_joint` already accepts 1..=4 rows in padded mode.

## What changes for users

- **Default N4 Qwen3.8 serving (speculation off at `max_slots >= 4`):** every ordinary decode step runs the padded joint row. Transcripts become independent of request timing; this is the target property.
  - Tokens that previously ran serially change from serial M1 to the padded row. That is the same Class B relation as joint vs serial today, and the same kernel family as stock's width-4 K3 verify rows. Transcripts therefore differ from earlier receipts.
  - A single active user on an N4 server now runs every step as a padded one-lane joint step. Its latency was to be gated separately (gate 5); that gate was not run, per operator directive 2026-11-06.
- **N1:** unchanged (one slot).
- **`max_slots` 2–3:** speculation defaults to `auto` there. Lone lanes with speculation state keep speculating; only ordinary lanes change. Under `auto`, requests that cannot speculate (sampled, logprobs, stop strings) become route-invariant.
- **Exact mode (Qwen3.6 default after #719, Qwen3.8 mvN arm):** unchanged.

## Repeatability that remains out of scope

These are documented only.
- **Speculation `auto`** is not repeat-deterministic, under any joint setting.
  - Cause 1: its proposers are switched on and off by wall-time cost controllers (`SpeculationCostController` observes elapsed target time).
  - Cause 2: an MTP or history lane alternates between joint M1 rows, when it shares a turn, and serial K3/history rounds, when it is alone or not joint-ready. That depends on cohort timing. A lane with pending verified output or MTP state can itself be pushed into a serial step. An ordinary lane no longer can, because it runs joint alone.
- **Defaults that remain non-repeatable.** Qwen3.8 resolves the unset speculation policy to `off` only for dense joint-decode workers with `max_slots >= 4` (`qwen35_speculation.rs:44`); `auto` is the default at 2–3 slots and is opt-in at 4.
  - Qwen3.6 (MoE) keeps speculation `auto` at every width, so default Qwen3.6 N4 serving is not repeat-deterministic, for both causes above.
  - With `HF2Q_QWEN_SPECULATION=off`, Qwen3.6 N4 runs exact joint decode, which is bitwise equal to serial and cohort-invariant (`qwen36_joint_decode_matches_serial_decode_bitwise`, `qwen36_joint_decode_rows_are_cohort_invariant_bitwise`). It is therefore repeatable by construction. #719's fork check agrees: 32/32 outputs identical across the serial, repeated-serial and joint arms.
- **Host-state-dependent reuse.** The LCP resume budget is 5% of available memory at startup. A budget difference could in principle change cross-slot reuse and therefore prefill chunk boundaries. No difference was observed in T-J1-8 (identical hit and chunk logs across runs); recorded as a residual risk. Gate 4 covers live resume boundaries.

## Gates

1. **Host tests.**
   - Partition: one ordinary lane runs joint with `solo` on and serial with it off; one MTP or history lane stays serial; 2–4 lanes form one group; with 5–8 lanes and `max_rows` 2, 3 or 4 every eligible lane lands in a group, and a speculating leftover of one stays serial; `solo` off reproduces today's partition exactly.
   - Configuration: `solo` is on only for padded mode with `n_slots >= 2`; the switch parses; it is off for exact mode and for one slot.
2. **Existing official-artifact gates, rerun on the branch:** `qwen38_joint_decode_padded_mv_ext_rows_are_cohort_invariant_bitwise` (now the serving path) and `qwen38_joint_decode_padded_mv_ext_keeps_decision_parity`.
3. **Serving k-sweep (GPU).** The replay from the investigation, with the companion leaving after k ∈ {0 (solo), 13, 17, all} joint steps.
   - Pass: with the switch on, agent-1 turn-4 is byte-identical across all k. With the switch off, the control reproduces the two classes.
   - Validity is arm against arm: every arm's per-request `cached_tokens` must equal the first arm's, including the companion's. The companion stream now requests `stream_options.include_usage`, so its usage is captured.
   - The warm-up uses `max_completion_tokens=1` and so resumes at the full previous prompt rather than the live prompt−5 boundary. The live resume boundary is covered only by gate 4.
4. **Four-agent repeat (GPU).** Q5-supervised T-J1-8-style runs on the branch binary, Qwen3.8 with speculation off (the default; T-J1-8 family a).
   - **Pass rule.** Four runs with the switch on; `q1_decision_policy.py transcripts` over all four must give `ExactTranscripts`.
   - **Negative control.** Two runs with `HF2Q_QWEN_JOINT_SOLO=0`, reporting their `DivergedTranscripts` rate against each other and against the on runs. It is expected nonzero, as in T-J1-8. A zero control rate weakens the evidence and is reported as such.
   - Families b (Qwen3.8 `auto`) and c (Qwen3.6 repair1): not run. Both use speculation `auto`, which this change does not make repeatable (out-of-scope section); T-J1-8 already records their behaviour.
5. **Cost: not run per operator directive 2026-11-06 (no speed tests).** The lone-lane decode-rate rule (on/off median ≥ 0.95) and the N4 makespan ABBA (≥ 8 blocks, CI upper bound ≤ 1.02) were pre-registered and are dropped. Option 2 (exact mode at N4: `HF2Q_DECODE_MVN=1 HF2Q_DECODE_MV_EXT=0` for Qwen3.8 workers with `max_slots >= 4`, where joint equals serial bitwise) stays the documented alternative if a cost problem is found later.

## Results (branch 616dd750 on main dc72f452, mlx-native 0.16.0)

Evidence: `~/.local/state/hf2q/v0130/q38-det-inv/fix717/` (`release-616dd750…/build-receipt.json`; server sha256 in the receipt; `p1-20261106T165219Z/report.json`; `p2-repeat/report.json`).

1. **Host tests:** 13 passed (`serve::api::engine::qwen35_joint_decode::`); replay tool 14 passed.
2. **Official-artifact gates:** `qwen38_joint_decode_padded_mv_ext_rows_are_cohort_invariant_bitwise` and `qwen38_joint_decode_padded_mv_ext_keeps_decision_parity` both passed.
3. **Serving k-sweep: pass.** All nine arms have equal per-request `cached_tokens`.
   - Switch on: solo ×2, k = 13, k = 17 and all-joint (k = 67) give one byte-identical agent-1 turn-4 output, the a-base " to" transcript. The debug lines show the one-lane joint steps (`lanes=1`).
   - Switch off (control): solo and k = 13 give " and", while k = 17 and all-joint give " to". This is the defect reproduced.
4. **Four-agent repeat: pass.** All six Q5 runs were `CalibrationPass` with cleanup proven and the protocol complete.
   - The four switch-on runs give `ExactTranscripts`: 28/28 turn positions, 0 same-request/different-output.
   - The two switch-off runs also agree with each other (`ExactTranscripts`), so the control showed no run-to-run divergence in this sample (two runs). That weakens the gate-4 evidence as pre-registered; the deterministic reproduction is gate 3.
   - Across all six runs, the on and off arms differ at agent-1 turn-4, the same near-tie: the route change itself, not run-to-run noise.
5. **Cost:** not run (operator directive 2026-11-06).

## Docs and consumers

- S3 doc (`adr063-s3-qwen-joint-decode.md`): the phase-1 width rule (padded mode groups every eligible lane; lone ordinary lanes run joint) and the numeric contract (gap closed), with the gates' results.
- `docs/operator-env-vars.md` and `docs/env-var-inventory.md`: `HF2Q_QWEN_JOINT_SOLO`, added to `scripts/audit_hf2q_env_surface.sh`. CHANGELOG entry.
- Q1 policy §2.5 (Qwen3.8): after gate 4, padded-mode N4 with speculation off is repeatable; `auto` is not.
- **Trace consumers.**
  - Lone-lane and leftover steps move from `ordinary_or_temporal_mtp_decode` transactions, which record a witnessed `forward_input`, to `joint_ordinary_decode` transactions.
  - Until Q4-TRACE lands the joint input manifest, each such transaction adds 48 unmapped witness records. Q2 `witness_progress` (`scripts/four_agent/f2_trace.py`) loses the witnesses those steps previously contributed.
  - `adr063-q4-trace-design.md` expects `logical1_pad3: 0` from the n4-fix shape. After this change, lone padded steps are normal and `logical1_pad3 > 0`.
  - Both docs get a note; the Q4-TRACE owner is told.
- **Replay tool.** It is committed to `scripts/four_agent/qwen_cohort_replay.py`:
  - the recording directory, the target and companion requests and the reference-output labels come from a JSON manifest;
  - validity is arm against arm as in gate 3;
  - `test_qwen_cohort_replay.py` covers stall detection, k and the cached-token check with synthetic fixtures.
