# ADR-065 B1: phase-1 baseline, fairness ceilings and campaign parameters

Story: [#503](https://github.com/robertelee78/hf2q/issues/503), [ADR-065](../adr/ADR-065-four-agent-joint-speculative-verify.md) "Measurement and acceptance rules" and B1.

This document and the campaign file [`scripts/four_agent/adr065/campaign-b1.json`](../../scripts/four_agent/adr065/campaign-b1.json) are predeclared, before any baseline or candidate data. Change them together, before data. Every run binds the campaign file by sha256. `scripts/four_agent/adr065_stats.py` (pure) and `scripts/four_agent/adr065_campaign.py` (driver) implement them. The tests are `test_adr065_stats.py` (synthetic receipts) and `test_adr065_campaign.py` (fake runner and the campaign file).

## Phase-1 default and identities

| Item | Value |
| --- | --- |
| Binary | v0.1.29 release asset `hf2q-aarch64-apple-darwin`, sha256 `86b64b029c6f717f3b6c58654c2aa5b29f420ffa93fe09f31348755fd2df48f2`, 55,972,368 bytes |
| Source commit | `c40fbf37bc2bc24cebd2c57faa60891debd1179a`. Main `d817dffa` differs from it only in release tooling: no change under `src/`, `Cargo.toml` or `Cargo.lock`. |
| Native | mlx-native `=0.15.16`. Before launch, the driver checks that the binary embeds exactly one `mlx-native-<version>` path and that it is 0.15.16. |

| Family | Official artifact | sha256 | Bytes | Q5 case (sha256) | Client approval template |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8 27B | `$MODELS/qwen38-stock/Qwen3.8-27B-vanilla.gguf` | `cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7` | 16,810,714,976 | `q5_cases/qwen38.json` (`4358e4c3…`) | `$E/approval-templates-proof-8b286684/qwen38-client-approval.json` (`d6c89cee…`) |
| DeepSeek-V4 Flash | `$MODELS/deepseek4/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf` | `50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6` | 107,431,343,264 | `q5_cases/deepseek4.json` (`d362f00e…`) | `…/deepseek4-client-approval.json` (`dfa0bf88…`) |
| Gemma 4 26B-A4B | `$MODELS/gemma4-google/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf` | `bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53` | 16,796,018,144 | `q5_cases/gemma4.json` (`c83d4de4…`) | `…/gemma4-client-approval.json` (`d2e30e96…`) |

`$MODELS` is `/Users/robert/.local/share/hf2q/models`; `$E` is `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`.

On 2026-11-03 all three GGUFs were hashed in full, and each sha256 equals its conversion receipt's `output.sha256` (`shasum -a 256`, evidence `/Users/robert/.local/state/hf2q/v0130/b1-503/artifact-sha256.txt`). Gemma 4 is included because Q4-GEMMA and E1 need its baseline. Runs do not re-hash the GGUF. Q5 binds it as Q3 does, through the conversion-receipt digest, checked against the campaign file's `MODEL_SHA256`. The driver also refuses an artifact whose digest or size differs from the campaign file.

Every run also uses the sandbox proof `owned-cleanup-sandbox-capability-20261031T222313Z/proof.json` (`8b286684…`) and the Q5 negative-control receipt `q5-negative-control-20261030T041437Z/receipt.json` (`b3c1db0b…`). At campaign start the driver freezes the binary, model, cases, argv input files, harness sources and commit, Python and host. It re-checks them before every run and against every receipt, as in [Q3](adr063-q3-campaign.md). Any drift is a terminal `IdentityDrift`.

## Design (every later two-arm A/B)

- **Arms.** A is the previous default: the family case. B is the candidate: the same binary, with a case that differs from A's only in `serve.env`, the kill switch. The driver refuses any other difference and records the changed variables.
- **Blocks.** One block is one paired A and B run. Odd blocks run A then B, even blocks B then A, so each pair of consecutive blocks forms ABBA. This is the two-arm Williams design (`q3_stats.block_order`). Each stage begins with one discarded warmup per arm per driver session, in the next block's order.
- **Block count.** Fixed before qualification: 6 by default for every family (`ab_blocks`). The alternatives are 8, 12, 16 or 20, chosen from a separately labelled `pilot` campaign that is named with `--pilot-of` and bound by sha256. Pilot runs are never pooled, and a pilot has no verdict. There is no sequential peeking: a stage stops only at its fixed count.
- **Stages.** `n4` is primary. The controls follow in this order: `n1`, then `3+1`, `2+2` and `1+3`, using the same client phase plans as Q3.
- **Effect.** On every complete block whose two runs both passed (quality-eligible), take d = ln(B makespan / A makespan), where makespan is the client's `own_output_four_task_makespan_seconds`. The interval is mean(d) ± t(0.975, n−1)·sd(d)/√n, reported as a percentage-point reduction 100·(1 − e^x). This is `q3_report.effect` / `q3_stats.paired_effect`, unchanged. The half-width is ADR-063's 100·(e^h − 1). When B is faster, it bounds the reduction interval's half-width from above.
- **Verdict**, assigned in order, with the cases mutually exclusive:
  1. **Loss** if the interval excludes zero unfavourably (upper bound < 0).
  2. **Win** if it excludes zero favourably (lower bound > 0) with a half-width ≤ 2.5 points.
  3. **Equivalent** if it contains zero and lies within [−2.5, +2.5].
  4. Otherwise **Inconclusive**, which is never reported as neutral or as a win.

  If fewer than the fixed count of blocks are quality-eligible, a Loss stands, but a Win or Equivalent becomes Inconclusive. A stage that did not complete is Inconclusive. An `Invalid` or `IdentityDrift` campaign has no verdict.
- **Failed runs.** A run with a task, tool, protocol, isolation or truncation failure is quality-ineligible. It stays in the receipt (`analysis.json` `runs`, with its outcome, failure kind, reason and attempts), and the campaign continues to its fixed count. Any quality-ineligible B run fails acceptance. Q3's terminal `Invalid` rule is unchanged: a run whose Pass the Q2 audit contradicts, or whose receipt is missing, still halts the campaign.
- **Controls.** By default the change should not engage with one agent. In that case the `n1` interval must lie within ±2.5 points, and the A/B fails acceptance otherwise. `--n1-engages` marks a change that does engage with one agent, and its `n1` effect is then only reported. J6, whose primary comparison is itself an N=1 A/B, has no n1 control: it runs with `--n1-engages`. Mixed `3+1`, `2+2` and `1+3` effects are reported.
- **Fairness.** In every measured B run, each slot's p95 semantic TTFT must be at or below the frozen TTFT ceiling for the workload, and the run's maximum semantic SSE gap at or below the SSE ceiling. A run that misses a needed value is `Unverifiable`, which fails acceptance. A baseline with no SSE ceiling (DeepSeek-V4 emits one semantic event per streamed turn) passes the SSE check only for a candidate that still measures none. Arm A is never checked. No median or pooled value is used.
- **Other gates.** Every run must reach a passing Q5 verdict, and none may show a serious or critical thermal state. `analysis.json` lists every failed gate in `gate_failures`, beside the verdict. `gates_pass` is true only when the list is empty. A candidate is accepted as a Win only with verdict `Win` and `gates_pass`.

## Baseline campaign (B1)

Arm A only, on the phase-1 default binary. For each family in the run order Qwen3.8, DeepSeek-V4, Gemma 4, the stages run in workload order. Each stage is one discarded warmup followed by a fixed number of measured runs: 20 for `n4`, and 6 each for `n1`, `3+1`, `2+2` and `1+3`.

For each workload, a ceiling is the largest value over that stage's passing measured runs and their slots:
- the TTFT ceiling over per-slot p95 semantic TTFT;
- the SSE ceiling over per-slot maximum semantic SSE gap, which is the same as the largest per-run maximum.

Warmups and failed runs never set a ceiling. They are kept in `ceilings.json` (`runs`, `retained_failures`). A stage whose measured runs all failed has no ceiling, and `ceilings` refuses it. Ceilings are written only for a `Complete` baseline campaign, and they bind the campaign by sha256.

Property of the rule: a candidate identical to the baseline exceeds the largest of the baseline's 4·20 n4 slot values with probability about 24/(80+24) ≈ 23% per metric in a 6-block A/B (24 candidate slot values). The 6-run controls carry a higher rate. The ADR fixes the max rule. The 20-run `n4` baseline keeps the false-flag rate as low as the budget allows.

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
BIN=$B1/hf0129-release/hf2q-aarch64-apple-darwin   # gh release download v0.1.29 -p 'hf2q-aarch64-apple-darwin*'
mkdir -p "$B1/baseline"
for F in qwen38 deepseek4 gemma4; do
  python3 -B scripts/four_agent/adr065_campaign.py baseline --family $F --baseline-bin "$BIN" \
    --qe-identity "<named session/host>" --dry-run > "$B1/dry-run-$F.json"
done
# GPU lane, HOST-LOCK held, nothing else loaded; one family at a time:
python3 -B scripts/four_agent/adr065_campaign.py baseline --family qwen38 --baseline-bin "$BIN" \
  --qe-identity "<named session/host>" --output "$B1/baseline/qwen38"
python3 -B scripts/four_agent/adr065_campaign.py ceilings "$B1/baseline/qwen38"
# ... the same for deepseek4, then gemma4.

# A later A/B (story S): B's case is the family case plus the kill switch in serve.env.
python3 -B scripts/four_agent/adr065_campaign.py ab --family qwen38 --binary <bin> --commit <40-hex> \
  --b-case <B case> --qe-identity "<named session/host>" --output <dir>
python3 -B scripts/four_agent/adr065_campaign.py analyze <dir> --ceilings "$B1/baseline/qwen38/ceilings.json"
```

After an independent review of this file, the campaign file and the baseline receipts, the reviewed `ceilings.json` of each family is published with the B1 receipts on main.
