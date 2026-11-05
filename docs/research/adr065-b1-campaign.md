# ADR-065 B1: phase-1 baseline, fairness ceilings and campaign parameters

Story: [#503](https://github.com/robertelee78/hf2q/issues/503), [ADR-065](../adr/ADR-065-four-agent-joint-speculative-verify.md) "Measurement and acceptance rules" and B1.

This document and the campaign file [`scripts/four_agent/adr065/campaign-b1.json`](../../scripts/four_agent/adr065/campaign-b1.json) are predeclared, before any baseline or candidate data. Change them together, before data. Every run binds the campaign file by sha256. `scripts/four_agent/adr065_stats.py` (pure) and `scripts/four_agent/adr065_campaign.py` (driver) implement them. The tests are `test_adr065_stats.py` (synthetic receipts) and `test_adr065_campaign.py` (fake runner and the campaign file).

## Phase-1 default and identities

**Operator decision (2026-11-04).** B1 baselines the fixed engine: hf2q main, once mlx-native 0.15.18 is published and pinned on main. 0.15.18 carries the RoPE partial-rotary fix for a bug in `rope_multi.metal`, which affects Qwen3.6 and Qwen3.8. B1 is never run on the v0.1.29 release binary.

**Operator decision (2026-11-05).** Rust 1.99.1 from here forward. The host's stable toolchain was updated at 07:14Z, and `38eea839` no longer builds on it. The baseline is therefore rebuilt with stable rustc 1.99.1 (commit `58cf8f9c35b355199727627d07ae34a4e2846d67`) from the newest main, once the Rust 1.99 fix merges. It remains "fixed main".

The campaign file holds that build's identity in `baseline`. While a field is `null` (a placeholder), the driver refuses to launch a baseline.

**Superseded build (2026-11-04), kept as evidence only and never the baseline:**

- Source commit `38eea839`.
- Binary sha256 `627d152a…`, 55,518,824 bytes, built with rustc `4a4ef493…`.
- Build receipt `a7846eb8…`.
- Files are in `$B1/main-build-38eea839/`; build log `$B1/main-build.log`.

| Item | Value |
| --- | --- |
| Binary | `$B1/main-build/hf2q`. It is built from a clean checkout of `source_commit` with `RUSTC_WRAPPER= CARGO_INCREMENTAL=0 GIT_COMMIT_SHA=<source_commit> CARGO_TARGET_DIR=/Users/robert/.local/state/hf2q/ext/cargo-targets/B1Tool cargo build --release --locked`, using the stable 1.99.1 toolchain, then copied out of the target directory. The campaign file records its sha256 and bytes. |
| Source commit | The newest hf2q main commit after the Rust 1.99 fix (the `source_commit` field). |
| Build receipt | `$B1/main-build/build-receipt.json` (`hf2q.adr065.build-receipt.v1`, written by `adr065_campaign.py build-receipt`), bound by sha256. It records:<br>• source commit, and binary sha256 and bytes;<br>• mlx-native version and checksum, read from `Cargo.lock` at that commit;<br>• `rustc -vV` output (release and commit hash), run in the checkout so a `rust-toolchain` file applies;<br>• the mlx-native versions and rustc commits embedded in the binary;<br>• the build command. |
| Native | mlx-native `=0.15.18`, with its crates.io checksum from `Cargo.lock` at `source_commit` (`native.checksum`). If main's pin has moved by then, `native.version` moves with it and the change is reviewed. |
| Toolchain | rustc 1.99.1, commit `58cf8f9c35b355199727627d07ae34a4e2846d67` (`toolchain` in the campaign file). |

Before launch, the driver checks five things against the campaign file:
- the binary's sha256 and size;
- the build receipt's sha256 and each of its identity fields, including `rustc_release` and `rustc_commit_hash`;
- the mlx-native version and checksum pinned in `Cargo.lock` at `source_commit`, read with `git show` from the harness repository;
- that the binary embeds exactly one `mlx-native-<version>` path, equal to `native.version`;
- that the binary embeds exactly one `/rustc/<commit>/` path, equal to `toolchain.rustc_commit_hash`.

Everything else in this document is unchanged by the decision: artifacts, cases, design, fairness rule, counts, cooldown and budget.

| Family | Official artifact | sha256 | Bytes | Q5 case (sha256) | Client approval template |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8 27B | `$MODELS/qwen38-stock/Qwen3.8-27B-vanilla.gguf` | `cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7` | 16,810,714,976 | `q5_cases/qwen38.json` (`4358e4c3…`) | `$E/approval-templates-proof-8b286684/qwen38-client-approval.json` (`d6c89cee…`) |
| DeepSeek-V4 Flash | `$MODELS/deepseek4/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf` | `50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6` | 107,431,343,264 | `q5_cases/deepseek4.json` (`d362f00e…`) | `…/deepseek4-client-approval.json` (`dfa0bf88…`) |
| Gemma 4 26B-A4B | `$MODELS/gemma4-google/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf` | `bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53` | 16,796,018,144 | `q5_cases/gemma4.json` (`71c218dd…`) | `…/gemma4-client-approval.json` (`d2e30e96…`) |

**Harness update, 2026-11-05.** The harness now includes main `1fb5e854`. That brings #704, which normalizes run-specific paths, durations and panic thread ids in the model-visible `run_tests` result, so four-agent prompts no longer differ between runs. The baseline binary is unchanged: it is still the build of `38eea839`. Main also changed `q5_cases/gemma4.json` (`c83d4de4…` → `71c218dd…`) and `serve_gemma4_opencode.sh`. They add three forwarded variables: `GEMMA_JOINT_COLD_PREFILL`, `GEMMA_PREFILL_MOE_DOWN` and `MM_ID_SCHEDULE`. All three default to empty, and the launcher forwards nothing when they are empty, so the Gemma server environment for the baseline is unchanged. The Qwen3.8 and DeepSeek-V4 cases are unchanged.

`$MODELS` is `/Users/robert/.local/share/hf2q/models`; `$E` is `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`; `$B1` is `/Users/robert/.local/state/hf2q/v0130/b1-503`.

On 2026-11-03 all three GGUFs were hashed in full, and each sha256 equals its conversion receipt's `output.sha256` (`shasum -a 256`, evidence `/Users/robert/.local/state/hf2q/v0130/b1-503/artifact-sha256.txt`). Gemma 4 is included because Q4-GEMMA and E1 need its baseline. Runs do not re-hash the GGUF. Q5 binds it as Q3 does, through the conversion-receipt digest, checked against the campaign file's `MODEL_SHA256`. The driver also refuses an artifact whose digest or size differs from the campaign file.

Every run also uses the sandbox proof `owned-cleanup-sandbox-capability-20261031T222313Z/proof.json` (`8b286684…`) and the Q5 negative-control receipt `q5-negative-control-20261030T041437Z/receipt.json` (`b3c1db0b…`). At campaign start the driver freezes the binary, model, cases, argv input files, harness sources and commit, Python and host. It re-checks them before every run and against every receipt, as in [Q3](adr063-q3-campaign.md). Any drift is a terminal `IdentityDrift`.

## Design (every later two-arm A/B)

- **Arms.** A is the previous default: the family case. B is the candidate: the same binary, with a case that differs from A's only in `serve.env`, the kill switch. The driver refuses any other difference and records the changed variables.
- **Blocks.** One block is one paired A and B run. Odd blocks run A then B, even blocks B then A, so each pair of consecutive blocks forms ABBA. This is the two-arm Williams design (`q3_stats.block_order`). Each stage begins with one discarded warmup per arm per driver session, in the next block's order.
- **Block count.** Fixed before qualification: 6 by default for every family (`ab_blocks`). The alternatives are 8, 12, 16 or 20, chosen from a separately labelled `pilot` campaign that is named with `--pilot-of` and bound by sha256. Pilot runs are never pooled, and a pilot has no verdict. There is no sequential peeking: a stage stops only at its fixed count.
- **Stages.** `n4` is primary. The controls follow in this order: `n1`, then `3+1`, `2+2` and `1+3`, using the same client phase plans as Q3.
- **Effect.** On every complete block whose two runs both passed (quality-eligible), take d = ln(B makespan / A makespan), where makespan is the client's `own_output_four_task_makespan_seconds`. The interval is mean(d) ± t(0.975, n−1)·sd(d)/√n, reported as a percentage-point reduction 100·(1 − e^x). This is `q3_report.effect` / `q3_stats.paired_effect`, unchanged. The half-width is ADR-063's 100·(e^h − 1). When B is faster, it bounds the reduction interval's half-width from above.
- **Precision test is slightly stricter than the ADR text (review finding 4, accepted as written).** The ADR asks for a Win half-width of at most 2.5 points. The tool tests Q3's 100·(e^h − 1), which is always at least the half-width of the reduction interval itself. In the reviewer's 20,000 synthetic campaigns, 115 (0.6%) moved from Win to Inconclusive because of this, and none moved the other way.
- **Verdict**, assigned in order, with the cases mutually exclusive:
  1. **Loss** if the interval excludes zero unfavourably (upper bound < 0).
  2. **Win** if it excludes zero favourably (lower bound > 0) with a half-width ≤ 2.5 points.
  3. **Equivalent** if it contains zero and lies within [−2.5, +2.5].
  4. Otherwise **Inconclusive**, which is never reported as neutral or as a win.

  If fewer than the fixed count of blocks are quality-eligible, a Loss stands, but a Win or Equivalent becomes Inconclusive. A stage that did not complete is Inconclusive. An `Invalid` or `IdentityDrift` campaign has no verdict.
- **Failed runs.** A run with a task, tool, protocol, isolation or truncation failure is quality-ineligible. It stays in the receipt (`analysis.json` `runs`, with its outcome, failure kind, reason and attempts), and the campaign continues to its fixed count. Any quality-ineligible B run fails acceptance. Q3's terminal `Invalid` rule is unchanged: a run whose Pass the Q2 audit contradicts, or whose receipt is missing, still halts the campaign.
- **Controls.** By default the change should not engage with one agent. In that case the `n1` stage must be `Complete`, all of its fixed blocks must be quality-eligible, and its interval must lie within ±2.5 points; otherwise the A/B fails acceptance. `--n1-engages` marks a change that does engage with one agent, and its `n1` effect is then only reported. J6, whose primary comparison is itself an N=1 A/B, has no n1 control: it runs with `--n1-engages`. Mixed `3+1`, `2+2` and `1+3` effects are reported, with their stage status.
- **Fairness (operator decision 2026-11-03).** Two checks, both on arm B only:
  1. **Paired regression test, the primary check.** This is Q3's existing test, reused unchanged (`q3_report.FAIRNESS`, `q3_report.effect`, `q3_stats.regression`). For each workload and each of two run metrics, take the block-paired ln(B/A) over quality-eligible blocks and its Student-t 95% interval. The metrics are the run's worst-slot p95 semantic TTFT and its worst-slot maximum semantic SSE gap. The candidate fails if the ratio interval's lower bound exceeds 1.025, meaning it is reliably worse beyond the 2.5-point neutral band. A metric that is measured but has no interval (fewer than two paired blocks) is `Unverifiable` and fails. A metric that no passing run measured does not gate; DeepSeek-V4 emits one semantic event per streamed turn, so it has no SSE gap.
  2. **Absolute safety cap.** In every measured B run, each slot's p95 semantic TTFT must be at or below 1.5 × the frozen TTFT ceiling for the workload. The run's maximum semantic SSE gap must be at or below 1.5 × the SSE ceiling. A run missing a needed value is `Unverifiable` and fails. A baseline with no SSE ceiling passes the SSE cap only for a candidate that still measures none.

  No median or pooled value replaces either check.

  **Why the change.** The ADR's original rule required every candidate run and slot to stay at or below the largest baseline value. The B1 review (finding 3) simulated that rule with the real `stage_ceilings` and `fairness_check` on exchangeable null data. A candidate identical to the baseline was flagged at about 0.39–0.41 per stage on `n4` and about 0.75 per 6-run control. It passed all five stages only 0.15–0.35% of the time. Holding the false-flag rate to 5% per metric would need about 114 baseline `n4` runs. The paired test compares the arms within each block, so its false-flag rate does not depend on how many baseline runs there are. The +50% cap still catches a single pathological run that the paired test cannot detect on its own.
- **Other gates.** Every run must reach a passing Q5 verdict, and none may show a serious or critical thermal state. The campaign must be `Complete`: a `Halted` campaign, even one whose `n4` stage completed, fails acceptance, so analysing part-way never accepts. `analysis.json` lists every failed gate in `gate_failures`, beside the verdict. `gates_pass` is true only when the list is empty. A candidate is accepted as a Win only with verdict `Win` and `gates_pass`.

## Baseline campaign (B1)

Arm A only, on the phase-1 default binary. For each family in the run order Qwen3.8, DeepSeek-V4, Gemma 4, the stages run in workload order. Each stage is one discarded warmup followed by a fixed number of measured runs: 20 for `n4`, and 6 each for `n1`, `3+1`, `2+2` and `1+3`.

For each workload, a ceiling is the largest value over that stage's eligible measured runs and their slots. A run is eligible when its outcome is Pass, its Q5 verdict is `CalibrationPass`, and it saw no serious or critical thermal state.
- The TTFT ceiling is taken over per-slot p95 semantic TTFT.
- The SSE ceiling is taken over per-slot maximum semantic SSE gap, which equals the largest per-run maximum.

Warmups, failed runs and excluded runs never set a ceiling. Every run is kept in `ceilings.json` (`runs`). Failed runs are also listed in `retained_failures`, and each stage lists its `excluded_runs` with the reason. `ceilings` refuses a stage with no eligible run. Ceilings are written only for a `Complete` baseline campaign, and they bind the campaign by sha256.

The ceilings serve only as the base of the +50% safety cap. Fairness acceptance rests on the paired test, so the baseline counts (20 `n4`, 6 per control) are sized for the baseline means and the `n4` throughput that E1, E8 and Q4-GEMMA consume, not for the cap.

## Execution and budget

- At least 60 s separate engine loads. The driver waits the full 60 s before a session's first load, and waits out the remainder of the 60 s after each run.
- One engine at a time; host reserved under `HOST-LOCK`; no peer or other hf2q process loaded; every run is one `q5_calibration.py` runtime that must reach `CalibrationPass`.
- **Attempt cap.** A campaign may make at most its planned runs plus 6 attempts. The 6 spare attempts cover resumed sessions' fresh warmups and the retries of runs that did not start. Past the cap, the campaign halts for good.

| Campaign | Planned runs | Max attempts | Qwen3.8 | DeepSeek-V4 | Gemma 4 |
| --- | --- | --- | --- | --- | --- |
| B1 baseline, per family | 49 (5 warmups + 44 measured) | 55 | 2.1–2.7 h | 2.3–3.0 h | 1.2–1.3 h |
| Two-arm A/B at 6 blocks, five stages, per family | 70 | 76 | 3.0–3.8 h | 3.3–4.3 h | 1.7–1.8 h |

The hours are runs × (observed Q5 wall time per run + 60 s cooldown). The per-run wall times come from Q3: Qwen3.8 95–135 s, DeepSeek-V4 111–160 s, Gemma 4 29–35 s. `n1` runs are shorter, so these hours are upper-leaning. All three baselines together take 5.6–6.9 h.

## Root commands

Run from a clean checkout of the harness commit; the driver refuses uncommitted `scripts/` changes. Use `--dry-run` first: it prints the plan, the frozen identities, the native versions found in the binary, the launch problems and the block schedule, and launches nothing. Re-run any command with `--resume` and the same `--output` to continue a halted campaign.

```sh
B1=/Users/robert/.local/state/hf2q/v0130/b1-503
MAIN=<40-hex main commit that pins mlx-native 0.15.18>
# 1. Build (wait while HOST-LOCK exists; kill rust-analyzer first).
git -C /opt/hf2q fetch origin && git -C /opt/hf2q worktree add --detach /opt/hf2q-worktrees/b1-main-build $MAIN
(cd /opt/hf2q-worktrees/b1-main-build && RUSTC_WRAPPER= CARGO_INCREMENTAL=0 GIT_COMMIT_SHA=$MAIN \
   CARGO_TARGET_DIR=/Users/robert/.local/state/hf2q/ext/cargo-targets/B1Tool cargo build --release --locked)
mkdir -p $B1/main-build && cp /Users/robert/.local/state/hf2q/ext/cargo-targets/B1Tool/release/hf2q $B1/main-build/hf2q
python3 -B scripts/four_agent/adr065_campaign.py build-receipt --binary $B1/main-build/hf2q \
  --source-checkout /opt/hf2q-worktrees/b1-main-build --output $B1/main-build/build-receipt.json \
  --command "RUSTC_WRAPPER= CARGO_INCREMENTAL=0 GIT_COMMIT_SHA=$MAIN CARGO_TARGET_DIR=/Users/robert/.local/state/hf2q/ext/cargo-targets/B1Tool cargo build --release --locked"
# 2. Finalize campaign-b1.json: baseline.binary.{source_commit, sha256, bytes}, build_receipt.sha256 and
#    native.checksum from that output (rustc_release/rustc_commit_hash must already equal toolchain); commit;
#    independent re-check of the identity change before the run.
# 3. Dry-run, then the baseline (GPU lane, HOST-LOCK held, nothing else loaded; one family at a time):
mkdir -p "$B1/baseline"
for F in qwen38 deepseek4 gemma4; do
  python3 -B scripts/four_agent/adr065_campaign.py baseline --family $F \
    --qe-identity "<named session/host>" --dry-run > "$B1/dry-run-$F.json"
done
python3 -B scripts/four_agent/adr065_campaign.py baseline --family qwen38 \
  --qe-identity "<named session/host>" --output "$B1/baseline/qwen38"
python3 -B scripts/four_agent/adr065_campaign.py ceilings "$B1/baseline/qwen38"
# ... the same for deepseek4, then gemma4.

# A later A/B (story S): B's case is the family case plus the kill switch in serve.env.
python3 -B scripts/four_agent/adr065_campaign.py ab --family qwen38 --binary <bin> --commit <40-hex> \
  --b-case <B case> --qe-identity "<named session/host>" --output <dir>
python3 -B scripts/four_agent/adr065_campaign.py analyze <dir> --ceilings "$B1/baseline/qwen38/ceilings.json"
```

After an independent review of this file, the campaign file and the baseline receipts, the reviewed `ceilings.json` of each family is published with the B1 receipts on main.

## Draft ADR-065 amendment (for the root to land)

The fairness rule above departs from the current text of ADR-065 ("Measurement and acceptance rules", third paragraph, and the B1 acceptance bullet). The ADR text needs this amendment before any candidate is judged by it.

> **Amendment 2026-11-03 (operator decision; B1 review finding 3).** Replace the fairness-ceiling paragraph with:
>
> Fairness MUST be checked per workload with ADR-063 Q3's paired fairness test. For the run's worst-slot p95 semantic TTFT and its worst-slot maximum semantic SSE gap (ADR-063 F2 estimands), compute the block-paired candidate/previous-default log ratio and its Student-t 95% interval. A candidate fails if the interval's lower bound exceeds 1.025, the 2.5-point neutral band. A metric that is measurable but has no interval fails. A metric that no passing run measures (one semantic event per streamed turn) does not gate.
>
> In addition, the phase-1 baseline freezes per-run, per-slot p95 semantic TTFT and per-run maximum semantic SSE gap, each ceiling being the largest value over eligible baseline runs. A run is eligible when it passed, its Q5 verdict is `CalibrationPass`, and it saw no serious or critical thermal state. Every candidate run and slot MUST stay at or below 1.5 × both ceilings, an absolute safety cap. No median or pooled aggregate may replace either check.
>
> Rationale: under the previous rule (every run and slot at or below the largest baseline value), a candidate identical to the baseline failed in about 99.7% of five-stage campaigns.
>
> In B1's acceptance, read "publish the ceilings" as "publish the ceilings used by the safety cap".

The ADR's acceptance text also implies, without stating, the following rules, which this tool enforces:
- A campaign that is not `Complete` fails acceptance.
- A required `n1` control must complete with all of its fixed blocks quality-eligible.
- The Win precision test uses ADR-063's 100·(e^h − 1) half-width.
