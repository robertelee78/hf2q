# ADR-063 Q3: three-arm crossover campaign per family

Status: harness and predeclared rules (2026-10-30). This document records no new measurement. It defines how `scripts/four_agent/q3_campaign.py` runs the Q3 comparison for one model family and how it turns raw receipts into a per-family verdict. Everything below the "Design" heading is predeclared: change it only before campaign data exists, together with `q3_stats.py`, `q3_metrics.py` and `q3_report.py`.

Sources: ADR-063 Q3 and Q4 (`docs/adr/ADR-063-apple-silicon-four-agent-speed-coherence.md`), the operator update "Per-family closure and release scope (2026-10-30)" (per-family win/loss/unavailable/failed verdicts; only wins count; F2 and Q1 do not gate), issue #369.

## Arms

Every arm runs the same GGUF, the canonical four-agent coding client (`qwen_n4_live_smoke.py`), 4 slots and 262144 tokens per slot, through the Q5 supervisor (`q5_calibration.py`), one runtime at a time.

| Arm | Binary | Case |
| --- | --- | --- |
| `stock` | Released v0.1.28 standalone binary: `$E/hf0128-publication-verify-v1/hf2q-aarch64-apple-darwin` (sha256 `c150329c…65d7`, source `9a0b79c751ede7caea6ee52ee17821282b95fea1`, signed and notarized per its `.proof.json`) | `q5_cases/<family>.json` with `HF2Q_BIN` from `--stock-bin` |
| `experiment` | `main` HEAD built by the root (`cargo build --release` at a recorded commit that descends from `9a0b79c7`) | the same case with `HF2Q_BIN` from `--experiment-bin` |
| `peer` | The peer's best valid configuration: `q5_cases/peer-<family>.json` (e86f5fd1 build, `build_info` `b128-e86f5fd`) unless `--peer-case` names another | peer case |
| `peer_matched` (optional) | `--peer-matched-case`: a separately labeled matched-settings diagnostic. It is interleaved like the other arms and reported as `diagnostic_vs_peer_matched`, and never enters the verdict | peer case |

The checked-in peer cases match hf2q's model-facing settings (template, thinking, sampling, context, slots; see their notes and `family_peer.py`). No better valid peer configuration has been identified for these workloads, so they are the declared best valid configuration. The DeepSeek-V4 `--load-mode` mmap, `--ubatch-size 256` variant (`$E/peer-arms-v1/peer-deepseek4-mmap.json`) was the best-valid-configuration attempt for that family. Declaring a different best configuration is an operator decision made before a campaign starts: pass it as `--peer-case` and keep the matched one as `--peer-matched-case`.

`HF2Q_Q5_HF2Q_BIN` and `--stock-var/--experiment-var HF2Q_BIN=…` are refused. The hf2q binaries come only from `--stock-bin` and `--experiment-bin`, so a case default can never stand in for one of them.

## Frozen identities

At campaign start the driver records the following in `campaign.json` `identities`:
- Per arm: binary path, sha256 and size. For the peer, every sibling `*.dylib` is hashed too, because its server is a launcher over them.
- Per arm: the model identity from Q5, which is the conversion-receipt sha256 plus size and mtime. Models are never hashed inline.
- Per arm: the declared peer `build_info` and the sha256 of every input file named in the arm's serve and client argv: launcher, approval template, chat template, sandbox proof, cargo and Python.
- The harness: the Git commit, the uncommitted paths under `scripts/` (a real run requires none), and the sha256 of every `scripts/four_agent/*.py`, case, fixture and `scripts/serve_*.sh`.
- The Python executable, the host (`hw.memsize`, chip, OS version) and the negative-control receipt.
- The declared source commits of the stock and experiment binaries (`--stock-commit`, `--experiment-commit`) and the QE identity (`--qe-identity`).

The driver recomputes every identity before each run. Each run's own Q5 receipt must also agree with the frozen binary sha256 and model size/sha256, and a peer run's `/props` `build_info` with the frozen one. Any difference sets `IdentityDrift`, which is terminal: `--resume` refuses, and a new campaign is required. All arms must serve the same model, and the stock and experiment binaries must differ.

## Workloads and controls

The server is always the frozen four-slot C262144 configuration; only the client's release plan changes. `--workloads` (default `n4,n1,3+1,2+2,1+3`) lists the stages in order, and the primary `n4` comes first.

| Key | Client | Definition |
| --- | --- | --- |
| `n4` (primary, pure phase) | canonical, no new option | All four agents are released by one barrier, so all four start prefill together and decode in the same phase. Byte-identical to every earlier N4 run. |
| `n1` (one-slot control) | `--phase-plan 1` | Only `agent-0` runs, with prompt and workload bytes identical to `agent-0` of the N4 run, on the same four-slot server. It isolates single-stream latency from cross-slot sharing. It is not the separate N1 client (`--diagnostic-slots 1`, one-slot server), which has its own approval schema and no peer adapter. |
| `3+1`, `2+2`, `1+3` (mixed phase) | `--phase-plan K+M` | Agents `agent-0..K-1` are released first. The remaining M agents are released (`release-2.json`) as soon as each of the first K has received its first decoded token, so their turn-0 prefill overlaps the first cohort's decode. |

**First decoded token.** Turn 0 is unary, so the earliest decode a client can observe is in its first streamed turn (turn 1): the first SSE frame that adds reasoning or semantic content. A role-only frame does not count. The agent publishes `barrier/<agent>.first-decode.json` (`hf2q.n4-first-decode.v1`, with its monotonic clock) at that point. The coordinator polls every 50 ms and records each trigger clock and the second release clock in `summary.phase.releases`. For every control, makespan runs from the first release to the last task completion.

`--phase-plan` cannot be combined with `--instance` or `--benchmark-profile`. Each agent result records its `{plan, cohort}`, and aggregation refuses a mismatch. A canonical result that carries a phase is also refused.

## Design

- **Blocks.** Williams sequences over the measured arms. With three arms there are six sequences: all permutations, each arm twice in every position, and each ordered adjacent pair twice. Two arms alternate AB/BA. Four arms use one 4×4 square in which every ordered pair is adjacent once. Block `b` uses sequence `(b-1) mod len`. An arm that becomes unavailable is left out of later blocks; the remaining arms keep their relative order.
- **Warmup.** In each driver session, before a stage's next block, one run per active arm in the order of that next block. Warmup runs are discarded for timing and kept as receipts. Their task outcomes still count, so an experiment warmup failure is an experiment failure. A warmup left unfinished by an interrupted session is marked `Abandoned`, and the resumed session runs fresh warmups.
- **Blocks per stage.** At least 5 and at most 20. After each complete block from block 5 onward, the stage stops when every required contrast has a 95% half-width ≤ 2.5 percentage points. Required contrasts are experiment-vs-stock, plus experiment-vs-peer while the peer is available and has not failed. At 20 blocks without that precision the stage stops as `max_blocks`, which is inconclusive. The rule stops on precision, never on the size of the effect.
- **Run outcomes** (from `receipt.json`, `client/summary.json` and `server.stderr`):
  - `Pass`: the client reports `FourAgentProtocolCompleteAwaitingIndependentReview` with exit 0 and no supervisor stop, and the Q2 run audit (`q2_run_audit.py`) re-derives that claim from the raw client evidence: archived request/response hashes, a parser replay of every turn, lineage, v3 timing, makespan arithmetic, cohort membership, verified cleanup, and a client-observed co-progress width of at least 4 for `n4`, the largest cohort for `K+M`, and exactly agent-0 for `n1`. See [adr063-q2-adversarial-controls.md](adr063-q2-adversarial-controls.md).
  - `Fail`: after the four-agent release, a failed client, `server_exited` or `client_timeout` (task failure), or a Q5 memory stop (`memory_stop`). A Metal out-of-memory marker in `server.stderr` (`kIOGPUCommandBufferCallbackErrorOutOfMemory`, `Insufficient Memory`) makes the kind `out_of_memory`.
  - `NotStarted`: Q5 preflight refusal, or a server exit, readiness timeout or memory stop before the client, or a client that ended before the release. The driver halts. `--resume` retries the same entry as a new attempt, and every attempt is kept.
  - `Invalid`: a missing receipt, a supervisor error, any other supervisor stop after the release, or a claimed Pass whose evidence the run audit rejects (for example a serialized service, a corrupted receipt or a makespan that disagrees with its clocks). Terminal, with no reruns after a task has started.
- **Peer unavailability.** A peer run that fails to load, or that fails with `out_of_memory` or a Q5 memory stop, marks the peer `Unavailable` with that run as evidence. Its later entries are `Skipped` and the remaining blocks run stock and experiment only. `--peer-unavailable-evidence FILE…` declares unavailability up front from earlier receipts (hashed into the plan), and no peer arm runs.
- **Experiment failure** (`Fail` in any stage, warmup included) stops the campaign: the stage is `Stopped`, later stages are `NotRun` and the family is `Failed`. Stock and peer task failures are recorded, and the campaign continues.
- **Resume.** `--resume` with the same `--output` and an identical plan continues a `Halted` campaign. An identical plan means the same cases, overrides, workloads, declarations and evidence. Attempts the driver was running when it died are scored from disk.

## Estimator (predeclared)

For a contrast of the experiment against reference arm R, over the complete blocks in which both runs passed:

$$d_b = \ln M_{E,b} - \ln M_{R,b}, \qquad \bar d = \tfrac1n\sum_b d_b, \qquad h = t_{0.975,\,n-1}\, \frac{s_d}{\sqrt n}$$

$M$ is the client's `own_output_four_task_makespan_seconds`, measured from the barrier release to the last validated task completion; model load is excluded. Each block is one cluster: one independent run per arm, each a whole cohort of conversations. Tokens and turns are never units. The reported ratio is $e^{\bar d}$ with 95% CI $[e^{\bar d-h}, e^{\bar d+h}]$, the improvement is $100(1-e^{\bar d})$ percent with its CI mapped the same way, and the half-width is $100(e^{h}-1)$ percentage points. The t quantiles for df 1–19 are tabulated in `q3_stats.T975`.

Secondary block-paired effects use the same estimator on: successful-task goodput (higher is better), client elapsed, worst-slot p95 and max semantic TTFT, and worst-slot p95 and max SSE gap.

## Per-run metrics

| Metric | Source and definition |
| --- | --- |
| Task completion | Protocol completion of every agent, plus optional independent patch review (`--patch-review`, `hf2q.adr063.q3-patch-review.v1`: final `src/ports.rs` sha256 → `Pass`/`Fail`). `verdict.json` lists the hashes still pending review. |
| Makespan | `client/summary.json` `own_output_four_task_makespan_seconds` |
| Successful-task goodput | completed agent tasks × 3600 / makespan (tasks per hour) |
| Per-slot semantic TTFT p95/max | streamed turns' `semantic_timing_v3.semantic_ttft_seconds` per agent (nearest rank); run value = worst slot |
| Per-slot SSE gap p95/max | adjacent `semantic_event_monotonic_ns` differences over the agent's streamed turns (nearest rank); run value = worst slot |
| Memory | Q5 receipt: verdict, min Free+cached, swap growth (`samples.deltas.swap_used_bytes`), pressure sample counts, peak footprint, max compressor, effective floor |
| Thermal | Q5 `preflight.thermal_state` before the run, and `NSProcessInfo.thermalState` read by the driver after each run. The Q5 sampler has no hook, so in-run thermal state is reported as `unrecorded`. |

## Verdict (primary `n4` workload)

Rules apply in this order:
1. `Failed`: any experiment run failed its task (Q4: "a family whose hf2q arm fails its task is recorded as failed"), or task quality was lost.
2. `Unavailable`: no runnable peer arm, with evidence. The experiment-vs-stock effect is still reported.
3. `Failed`: the peer failed its task. A failed peer cannot yield a win.
4. `Pending`: the primary stage has not reached its stopping rule.
5. `Loss`: the experiment-vs-peer makespan ratio CI lies above 1, so the experiment is slower than the peer. This holds at any precision, because a longer makespan with an interval excluding a tie is a measured loss.
6. `Win`: every one of the following holds.
   - Both required contrasts reached the precision target.
   - Against stock, the CI upper bound is below 1.
   - Against the peer, the CI upper bound is ≤ 0.95. This is Q4's "at least 5% lower makespan … with paired uncertainty excluding a tie": the whole interval must clear the margin.
   - Neither fairness effect against stock (worst-slot p95 TTFT, worst-slot max SSE gap) has a CI lower bound above the 2.5-point neutral band (ratio 1.025).
   - Quality is not unresolved.
   - No run observed a serious or critical thermal state.
   - Every passing experiment run has a passing Q5 memory verdict.
7. `Inconclusive`: everything else, with every unmet condition listed as a reason. An unresolved interval is inconclusive, not faster.

**Quality.** `--quality-verdict MANIFEST` takes a `quality_campaign.py` manifest that carries a Q1 paired verdict. It counts only if its subject runs' receipts show the experiment binary sha256. `Loss` or `BelowFloor` → `Failed`. `Inconclusive`, `Insufficient` or an unbound binary → unresolved, so no win. Without a Q1 verdict, quality is task-completion parity: no experiment run failed.

The `n1` and mixed-phase stages are reported in `verdict.json` `controls` with the same effects and arm summaries. They do not change the family verdict, except that an experiment failure in any stage fails the family.

## Outputs

- `campaign.json` (`hf2q.adr063.q3-campaign.v1`) contains:
  - the plan and its sha256;
  - the frozen identities and their sha256;
  - sessions, and every stage with its warmups, blocks, Williams order and stop;
  - every attempt with its argv, Q5 receipt sha256, Q5 verdict, stop, client status, outcome, failure kind and metrics;
  - unavailable arms with their evidence, and the halt state.
- Q5 run directories are under `runs/`, and the derived cases are under `cases/<workload>-<arm>.json`. The `n4` case files are exact copies of the source cases.
- `verdict.json` (`hf2q.adr063.q3-verdict.v1`) contains:
  - the family verdict and its reasons;
  - primary effects with CIs, secondary effects and per-arm summaries (median makespan and goodput, memory, maximum observed thermal state);
  - controls, failures, peer status, quality inputs and patch-review status;
  - rejected trials (Invalid, Abandoned, Skipped or not-started attempts), the number of discarded warmups, and the predeclared design.
- `q3_campaign.py --rows CAMPAIGN --evidence-root "$E"` prints a `{groups, rows}` fragment in the `hf2q.f2.scoreboard-rows.v1` vocabulary. It holds the experiment-vs-stock and experiment-vs-peer makespan effects as `% improvement`, with `$E/...` receipt references and JSON extraction keys into `verdict.json`, so `f2_scoreboard_render.py --verify` can re-read them. Merge it into `docs/research/adr063-f2-scoreboard-rows.json` only after independent review. Statuses: `Win`/`Inconclusive`/`Unavailable` → `Descriptive`, `Loss` → `NegativeResult`, `Failed` → `Failure`.

## Existing evidence (not rerun)

These earlier receipts motivate the expected outcomes. None of them is a Q3 verdict: none has a stock arm, Williams blocks or the precision rule.
- **Qwen3.8 27B.** `$E/peer-arms-v1/analysis-q38-gemma-vs-peer.json`: the hf2q joint binary `qwen-joint-bin-a2aa1fc5` had a median client elapsed of 89.8 s, against 112.4 s for the peer. That is 3 runs each, alternating, a peer-over-hf2q time ratio of 1.241 (95% CI 1.138–1.353), or roughly 19% lower makespan. A Q3 win is plausible if the experiment keeps that binary's gain.
- **Gemma 4 26B-A4B.** Same file: hf2q 23.4 s against peer 13.5 s median (ratio 0.555, CI 0.408–0.755). hf2q was slower, so `Loss` is expected unless the experiment changes Gemma decode. Gemma's run-to-run spread (22.7–28.9 s) suggests it will need close to 20 blocks to reach the 2.5-point target.
- **DeepSeek-V4 Flash.** No runnable peer arm at C262144 × 4 with f16 KV. `$E/peer-arms-v1/peer-deepseek4-20261030T165847Z` (`--load-mode none`) and `peer-deepseek4-mmap-20261030T165948Z` (mmap, ubatch 256) both loaded, then hit Metal `kIOGPUCommandBufferCallbackErrorOutOfMemory` on the first requests: every agent got an API reject and the result was `ClientFailed`. The first attempt, `peer-deepseek4-20261030T165437Z`, failed the template check, and its server log shows the same OOM. Expected verdict: `Unavailable`. Run with `--peer-unavailable-evidence` so the stock-vs-experiment effect (joint decode, #458: 141.7 → 105.0 s median) is still measured.
- **Qwen3.6 35B-A3B.** The hf2q arm fails its coding task (#461; for example `$E/quality-qwen36-peer-exp0b95-20261030T174020Z`, where most subject instances are `ClientFailed`). Expected verdict: `Failed`, decided by the first experiment failure, unless the experiment fixes the task.

## Root commands

Run from the repository root of a clean checkout at the committed harness. Run one family at a time with nothing else loaded. Add `--dry-run` first: it prints the plan, the frozen identities, the launch problems and the full 20-block Williams schedule for every stage, and launches nothing. Every command below is also resumable: re-run it with `--resume` and the same `--output`.

```sh
E=/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023
Q3=$E/q3-campaign-v1
export HF2Q_Q5_CARGO=/Users/robert/.rustup/toolchains/stable-aarch64-apple-darwin/bin/cargo
export HF2Q_Q5_TOOLCHAIN=stable-aarch64-apple-darwin
export HF2Q_Q5_SANDBOX_PROOF=$E/owned-cleanup-sandbox-capability-20261026T145912Z/proof.json
export HF2Q_Q5_SANDBOX_PROOF_SHA256=3337f26cc1cd6b9a6948c03f8e06dec0c7944a3cb553dc31eb05deb2e5fe05a9
NC=$E/q5-negative-control-20261030T041437Z/receipt.json
STOCK=(--stock-bin "$E/hf0128-publication-verify-v1/hf2q-aarch64-apple-darwin"
       --stock-commit 9a0b79c751ede7caea6ee52ee17821282b95fea1 --stock-label "v0.1.28 release")
EXP_COMMIT=<40-hex main HEAD the root built>
EXP=(--experiment-bin "$E/q3-experiment-bin-$EXP_COMMIT/hf2q" --experiment-commit "$EXP_COMMIT"
     --experiment-label "main ${EXP_COMMIT:0:8}")
COMMON=(--negative-control-receipt "$NC" --qe-identity "<named QE session/host>")
MODELS=/Users/robert/.local/share/hf2q/models
# Peer approval templates (durable copies of the ones the peer-arms-v1 receipts used). Their sha256 must equal
# those receipts' `inputs`: qwen38 and qwen36 4598269649a0aba1735ac4aeae075372fc6c4d4ec6c69acb66165fe968c48fbc (identical bytes),
# gemma4 8e060367d7aaae564acca5e672ea90ed1e1f66c08fb7b3dc7de8021cea2e0975,
# deepseek4 ff86b7e70df67ba8215af4587e756d87f64ed59b4d22aad22e1f8316fac3c129.
T=$E/peer-templates

# Qwen3.8 27B
HF2Q_Q5_MODEL=$MODELS/qwen38-stock/Qwen3.8-27B-vanilla.gguf \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=$E/qwen38-matched-baseline1512-normal-n4-20261029T024609Z/client-approval.json \
  python3 -B scripts/four_agent/q3_campaign.py --family qwen38 "${STOCK[@]}" "${EXP[@]}" "${COMMON[@]}" \
  --peer-var CLIENT_APPROVAL_TEMPLATE=$T/peer-qwen38-client-approval.json \
  --output "$Q3/qwen38"

# DeepSeek-V4 Flash (no runnable peer: stock vs experiment only)
HF2Q_Q5_MODEL=$MODELS/deepseek4/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf \
HF2Q_Q5_MODEL_SHA256=50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6 \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=$E/current-port-balanced-c1-normal-n4-v1/client-approval.json \
  python3 -B scripts/four_agent/q3_campaign.py --family deepseek4 "${STOCK[@]}" "${EXP[@]}" "${COMMON[@]}" \
  --peer-unavailable-evidence $E/peer-arms-v1/peer-deepseek4-20261030T165847Z/receipt.json \
  --peer-unavailable-evidence $E/peer-arms-v1/peer-deepseek4-20261030T165847Z/server.stderr \
  --peer-unavailable-evidence $E/peer-arms-v1/peer-deepseek4-mmap-20261030T165948Z/receipt.json \
  --peer-unavailable-evidence $E/peer-arms-v1/peer-deepseek4-mmap-20261030T165948Z/server.stderr \
  --output "$Q3/deepseek4"

# Qwen3.6 35B-A3B
HF2Q_Q5_MODEL=$MODELS/qwen36-stock-adr063/Qwen3.6-35B-A3B-vanilla-Q4_K_M.gguf \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=$E/qwen36-template-order-coding-operating90-n4-20261027T231028Z/client-approval.json \
  python3 -B scripts/four_agent/q3_campaign.py --family qwen36 "${STOCK[@]}" "${EXP[@]}" "${COMMON[@]}" \
  --peer-var CLIENT_APPROVAL_TEMPLATE=$T/peer-qwen36-client-approval.json \
  --output "$Q3/qwen36"

# Gemma 4 26B-A4B
HF2Q_Q5_MODEL=$MODELS/gemma4-google/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=$E/gemma4-full-input-n4-20261026T210724Z/client-approval.json \
  python3 -B scripts/four_agent/q3_campaign.py --family gemma4 "${STOCK[@]}" "${EXP[@]}" "${COMMON[@]}" \
  --peer-var CLIENT_APPROVAL_TEMPLATE=$T/peer-gemma4-client-approval.json \
  --output "$Q3/gemma4"

# After a Q1 paired campaign on the experiment binary, or an independent patch review:
python3 -B scripts/four_agent/q3_campaign.py --verdict "$Q3/qwen38" \
  --quality-verdict "$E/<quality campaign>/manifest.json" --patch-review "$Q3/qwen38-patch-review.json"
# F2 scoreboard rows (merge after review):
python3 -B scripts/four_agent/q3_campaign.py --rows "$Q3/qwen38" --evidence-root "$E"
```

`HF2Q_Q5_*` exports apply to every arm whose case declares the variable. `--peer-var` overrides the peer arm only. The hf2q approval templates above are the ones the 2026-10-29/30 receipts used. Their `sources` are recomputed at run time, so they stay valid after the client change. Before starting, the root checks that the stock binary accepts the current launchers' flags: a server that refuses an option makes the first stock warmup `NotStarted` and halts the campaign before any measured block.

### Estimated wall time

Observed per-run Q5 wall time on 2026-10-30, including load, workload, cleanup and the post-exit memory sample:
- Qwen3.8: hf2q joint binary 95–101 s, pre-joint 124–132 s, peer 108–129 s.
- Gemma 4: hf2q 29–35 s, peer 20–24 s.
- Qwen3.6: 27–64 s.
- DeepSeek-V4: joint binary 111–118 s, serial 144–160 s.

A stage costs one warmup per arm plus 5–20 blocks. Estimates per family for all five stages, assuming controls cost about as much as `n4`; `n1` is shorter:

| Family | Arms | Per stage | Five stages |
| --- | --- | --- | --- |
| Qwen3.8 | 3 | 0.6–2.0 h | 3–10 h |
| Gemma 4 | 3 | 8–30 min; expect near 20 blocks | 1–2.5 h |
| DeepSeek-V4 | 2 | 0.45–1.6 h | 2.5–8 h |
| Qwen3.6 | 3 | 15–45 min if the task passes | ends at the first experiment failure if it still fails (minutes) |

The driver adds about 1 s per run to re-hash identities.

## Independent QE and reproduction

ADR-063 Q3 asks a separate, named QE session or host to execute Q3. Under this repository's rules only the root may run models, so the root executes the commands above and records the declared QE identity (`--qe-identity`) in the plan. The QE-separation requirement is met by an independent reviewer who did not implement the candidate. That reviewer reproduces every verdict from the raw receipts:
- re-derives each run's metrics from `runs/*/receipt.json` and `client/summary.json`;
- recomputes the Williams order, the effects and the stop from `campaign.json` (`q3_campaign.py --verdict` is deterministic over the campaign directory);
- checks the frozen identities against the binaries, the model receipt and the harness commit;
- reviews the listed final-source hashes;
- confirms that `rejected_trials`, abandoned warmups and halts are all reported.

A verdict is evidence for Q3 and Q4 only after that review accepts it. Faster incorrect tasks, serial scheduling, mismatched settings, changed artifacts or cherry-picked trials fail review.
