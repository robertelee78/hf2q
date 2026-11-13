# ADR-063 Q3: three-arm crossover campaign per family

Status: harness and predeclared rules (2026-10-30). This document records no new measurement. It defines how `scripts/four_agent/q3_campaign.py` runs the Q3 comparison for one model family and how it turns raw receipts into a per-family verdict. Everything below the "Design" heading is predeclared: change it only before campaign data exists, together with `q3_stats.py`, `q3_metrics.py` and `q3_report.py`.

Sources: ADR-063 Q3 and Q4 (`docs/adr/ADR-063-apple-silicon-four-agent-speed-coherence.md`), the operator update "Per-family closure and release scope (2026-10-30)" (per-family win/loss/unavailable/failed verdicts; only wins count; F2 and Q1 do not gate), issue #369.

## v0.1.30 release-gate amendment (operator decision 2026-11-10)

New v0.1.30 campaigns use the published v0.1.29 release, source
`c40fbf37bc2bc24cebd2c57faa60891debd1179a`, as `stock`. A release-pass
`Win` requires the candidate/stock 95% CI upper ratio bound to be at most
0.95 and every candidate/valid-peer upper bound to be at most 1.00. Thus the
5% minimum is against the immediately prior hf2q release; peers are
non-inferiority gates. `campaign.json` records both thresholds as
`design.stock_margin_ratio` and `design.peer_margin_ratio`.

Historical campaigns remain reproducible: `q3_report.py` reads margins from
the campaign plan, and plans made before this amendment retain the former
stock-gain/peer-0.95 rules. Qwen3.6 still requires a valid Splash contrast.
Qwen3.8's corrected Splash arm is not a valid performance denominator, so its
separate ADR-067 mechanism gate applies as specified by ADR-063.

## v0.1.30 precision-cap amendment (execution decision 2026-11-10)

The default sequential cap remains 20 blocks. A fresh campaign may predeclare
`--max-blocks 40` when a completed 20-block campaign exhausted its cap solely
because a required contrast missed the unchanged 2.5-point precision target.
The extended campaign starts from new immutable identities and receipts; it
does not pool the exhausted campaign. The Williams order, minimum five blocks,
precision stop, estimator, release thresholds and every quality/fairness gate
remain unchanged. `campaign.json` freezes the selected cap, and resume requires
the identical plan.

This extension responds to
`q4-gemma-release-gate-27a9ddc4-v2`: all runs were valid and both release
thresholds were demonstrated, but the candidate/peer half-width was 3.254
points at block 20 (ratio 0.9153, 95% CI [0.8864, 0.9450]). The campaign is
still `Inconclusive`; its observed effect is not accepted or pooled. The
extension changes only the maximum opportunity to reach the already declared
precision target in a new campaign.



## Arms

Every arm runs the same GGUF, the canonical four-agent coding client (`qwen_n4_live_smoke.py`), 4 slots and 262144 tokens per slot, through the Q5 supervisor (`q5_calibration.py`), one runtime at a time.

| Arm | Binary | Case |
| --- | --- | --- |
| `stock` | Released v0.1.28 standalone binary: `$E/hf0128-publication-verify-v1/hf2q-aarch64-apple-darwin` (sha256 `c150329c…65d7`, source `9a0b79c751ede7caea6ee52ee17821282b95fea1`, signed and notarized per its `.proof.json`) | `q5_cases/<family>.json` with `HF2Q_BIN` from `--stock-bin` |
| `experiment` | `main` HEAD built by the root (`cargo build --release` at a recorded commit that descends from `9a0b79c7`) | the same case with `HF2Q_BIN` from `--experiment-bin` |
| `peer` | The peer's best valid configuration: `q5_cases/peer-<family>.json` (e86f5fd1 build, `build_info` `b128-e86f5fd`) unless `--peer-case` names another | peer case |
| `peer_matched` (optional) | `--peer-matched-case`: a separately labeled matched-settings diagnostic. It is interleaved like the other arms and reported as `diagnostic_vs_peer_matched`, and never enters the verdict | peer case |

The checked-in peer cases match hf2q's model-facing settings (template, thinking, sampling, context, slots; see their notes and `family_peer.py`). No better valid peer configuration has been identified for these workloads, so they are the declared best valid configuration. The DeepSeek-V4 `--load-mode` mmap, `--ubatch-size 256` variant (`$E/peer-arms-v1/peer-deepseek4-mmap.json`) was the best-valid-configuration attempt for that family. Declaring a different best configuration is an operator decision made before a campaign starts: pass it as `--peer-case` and keep the matched one as `--peer-matched-case`.

**Native ds4 arm (DeepSeek-V4 only, operator decision 2026-10-31).** `--peer-case q5_cases/native-ds4-deepseek4.json` with the same-GGUF `--peer-unavailable-evidence` runs antirez ds4 c97964b in the `peer` slot, on its own ds4f-q2 Flash 0731 artifact. The plan labels the arm `ds4 c97964b (native Flash-0731 q2 artifact)`. Its artifact is exempt from the same-model rule but must carry a sha256. Its build identity is the ds4 commit plus the `ds4-server` sha256, and its 26 runtime Metal sources are frozen too. The family verdict stays `Unavailable`, because there is no same-GGUF peer. `verdict.json` `native_peer` reports the task-makespan comparison separately, and F2 rows use baseline `ds4_native`. Like every campaign with a native peer arm, it emits only those native rows (plus control failure rows), never an experiment-vs-stock row: that effect is of record only in the same-GGUF campaign. The same-file case `native-ds4-deepseek4-samefile.json` (ds4 on the ds4 file the hf2q arm loads, `artifact_relation: same_as_hf2q`) runs only under `--native-comparison` instead: see "No-stock-arm campaigns". Adapter and controls: `scripts/four_agent/README.md` "Native ds4 arm".

**Native TensorFold arm (Qwen3.8 only, operator decision 2026-10-31).** Qwen3.8's same-GGUF peer is runnable, so TensorFold never replaces it in the campaign of record. A separate campaign passes `--peer-case q5_cases/native-tensorfold-qwen38.json` (no drafter) or `native-tensorfold-qwen38-dflash2.json` (DFlash2) together with `--native-comparison`. TensorFold 1157ea5b then runs in the `peer` slot on its own MLX 4-bit checkpoint, and the plan records `native_comparison`.
- `--peer-unavailable-evidence` and `--peer-matched-case` both refuse with this flag. Without the flag, a TensorFold `--peer-case` refuses.
- The artifact is the pinned target snapshot's `config.json` (TensorFold's `config_sha256`), with `TARGET_CONFIG_SHA256` required. Build identity is `tensorfold <version> <commit> binary-sha256 <TF_BIN_SHA256>`, and the target and drafter snapshot file listings are frozen with the campaign identity.
- `verdict.json` sets `verdict_of_record: false` and `campaign_label: "native-comparison campaign; not the verdict of record"`, and the family verdict is `Unavailable` with that reason. `native_peer` carries the task-makespan statement, with the same withholding rules as ds4.
- F2 rows go to group `qwen38-q3-tensorfold-native` or `qwen38-q3-tensorfold-dflash2-native` (`-controls-` groups under `--controls-of`), and every row carries the not-of-record label. No experiment-vs-stock row is emitted: that effect belongs to the campaign of record. The TensorFold row renders only under "Native-engine comparisons".
- A TensorFold or oMLX comparison keeps the stock arm by default. `--no-stock-arm --native-comparison` drops it (no vs-stock row exists in a native comparison anyway): see "No-stock-arm campaigns" below.
- Adapter and controls: `scripts/four_agent/README.md`, section "Native TensorFold arm".

**Native oMLX arm (Qwen3.8, Qwen3.6, Gemma 4; operator decision 2026-10-31).** Every one of these families has a runnable same-GGUF peer, so oMLX runs only in a separate `--native-comparison` campaign, with TensorFold's rules: `--peer-case q5_cases/native-omlx-<family>.json --native-comparison`. oMLX 6b413c90 then runs in the `peer` slot on the pinned mlx-community 4-bit snapshot.
- The artifact is the snapshot's `config.json`; `MODEL_CONFIG_SHA256` and `TEMPLATE_SHA256` are required. Build identity is `omlx <version> <commit> venv-freeze-sha256 <sha>`, from the venv's dist-info inventory. The snapshot file listing and the venv inventory are frozen with the campaign identity.
- `verdict.json` and F2 rows carry the not-of-record label. `native_peer` is withheld under the TensorFold rules, and also when any oMLX run logged tool-parser WARNING/INFO lines; tool-call repairs oMLX does not log are disclosed with every statement as `recovery_attribution: Unavailable:…`. F2 groups are `<family>-q3-omlx-<family>-native`, and the oMLX row renders only under "Native-engine comparisons".
- Adapter and controls: `scripts/four_agent/README.md`, section "Native oMLX arm".

**Splash arm (Qwen3.8 and Qwen3.6; ADR-063 "Splash peer of record for the Qwen families", operator decisions 2026-11-03/11-04).**
Splash is a same-artifact peer of record. It is not a native comparison, so it gets its own measured arm, `splash`,
beside `peer`.
- **Default.** A default campaign for `qwen38` or `qwen36` runs `--splash-case`, which defaults to
  `q5_cases/splash-<family>.json`.
- **Unavailable up front.** `--splash-unavailable-evidence FILE…` declares Splash unavailable before the campaign
  (hashed into the plan), and no Splash arm runs.
- **Same artifact.** The Splash artifact must equal the experiment's GGUF sha256 (same-model rule plus the adapter's
  launch check).
- **Identity.** Its build identity covers the commit, `build_id`, the binary, metallib and host-step sha256s, the
  toolchain and the venv freeze. The draft, configuration and argv template are frozen with the campaign identity.
- **Stages (decision D7).** The arm runs in the `n4` and `n1` stages only. The mixed `K+M` controls leave it out of
  their measured arms and Williams orders.
- **Memory stop (decision D6).** A Splash Q5 memory stop halts the campaign. The fix is to record a `--max-memory`
  cap in `SPLASH_CONFIG`, re-confirm that configuration and start a new campaign.
- **Adapter and controls.** `scripts/four_agent/README.md`, section "Splash peer of record"; design
  `docs/research/adr063-splash-peer-design.md`.

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

The server is always the frozen four-slot C262144 configuration; only the client's release plan changes. `--workloads` (default `n4,n1,3+1,2+2,1+3`) lists the stages in order, and the primary `n4` comes first, except in a controls-only campaign (below).

| Key | Client | Definition |
| --- | --- | --- |
| `n4` (primary, pure phase) | canonical, no new option | All four agents are released by one barrier, so all four start prefill together and decode in the same phase. Byte-identical to every earlier N4 run. |
| `n1` (one-slot control) | `--phase-plan 1` | Only `agent-0` runs, with prompt and workload bytes identical to `agent-0` of the N4 run, on the same four-slot server. It isolates single-stream latency from cross-slot sharing. It is not the separate N1 client (`--diagnostic-slots 1`, one-slot server), which has its own approval schema and no peer adapter. |
| `3+1`, `2+2`, `1+3` (mixed phase) | `--phase-plan K+M` | Agents `agent-0..K-1` complete turn 0 and become ready for their first streamed request. The remaining M agents are released (`release-2.json`) to begin turn-0 prefill. After a head start equal to half the shortest first-cohort turn-0 request duration (with a 1-second floor for fast fixtures), the first cohort is released to stream. Q2 requires all four agents to make decode progress concurrently. |

**Ready, prefill head start, then stream.** Each first-cohort agent completes unary turn 0 and its local tool, then publishes `barrier/<agent>.first-stream-ready.json` (`hf2q.n4-first-stream-ready.v1`) with its ready, turn-0 send and turn-0 terminal clocks. The coordinator waits for every marker, releases the later cohort, and holds the first cohort for half the shortest recorded unary-turn duration, with a 1-second minimum. That scaled, source-bound head start places the later cohort inside turn-0 prefill when `stream-release-1.json` starts the first cohort's streamed turns. The summary records every marker, both release clocks and the head start. The Q2 audit binds markers to the archived turn-0 receipts, re-derives the head start, proves later turn-0 requests started before the stream release, proves every first-cohort task remained active, and still requires observed four-agent co-progress. For every control, makespan runs from the initial cohort release to the last task completion.

Three retained 2026-11-08 `3+1` attempts established the phase trigger boundary. `$E/v0130/b1-503/baseline/qwen38` waited for every first decoded token; `$E/v0130/b1-503/phase-trigger-smoke-8f674a1c` released on the first decoded token; `$E/v0130/b1-503/phase-trigger-smoke-e19c226d` released when every first streamed request was sent. All completed the tasks and reached four overlapping HTTP requests, but Q2 correctly rejected decode co-progress width 3: the later cohort's unary prefill consumed nearly the first cohort's entire streamed turn. None is pooled. The ready/head-start protocol removes that measured scheduling ambiguity without weakening the width-4 gate.

The replacement smoke at `$E/v0130/b1-503/phase-trigger-smoke-82af777e` passed Q5 and cleanup, reached HTTP overlap 4, and passed Q2 at decode co-progress width 4. The witness spans turn 1 of all agents for 3.847 seconds, with 61/62/62/40 progress frames and overlap fraction 0.5005.

`--phase-plan` cannot be combined with `--instance` or `--benchmark-profile`. Each agent result records its `{plan, cohort}`, and aggregation refuses a mismatch. A canonical result that carries a phase is also refused.

## Design

- **Blocks.** Williams sequences over the measured arms. With three arms there are six sequences: all permutations, each arm twice in every position, and each ordered adjacent pair twice. Two arms alternate AB/BA. Four arms use one 4×4 square in which every ordered pair is adjacent once. Block `b` uses sequence `(b-1) mod len`. An arm that becomes unavailable is left out of later blocks; the remaining arms keep their relative order.
- **Warmup.** In each driver session, before a stage's next block, one run per active arm in the order of that next block. Warmup runs are discarded for timing and kept as receipts. Their task outcomes still count, so an experiment warmup failure is an experiment failure. A warmup left unfinished by an interrupted session is marked `Abandoned`, and the resumed session runs fresh warmups.
- **Blocks per stage.** At least 5 and at most 20 by default. A fresh campaign may predeclare the approved 40-block precision extension above. After each complete block from block 5 onward, the stage stops when every required contrast has a 95% half-width ≤ 2.5 percentage points. Required contrasts are experiment-vs-stock (while the stock arm has not exhausted its reruns, below), plus experiment-vs-peer while the peer is available and has not failed. At the plan's 20- or 40-block cap without that precision the stage stops as `max_blocks`, which is inconclusive. The rule stops on precision, never on the size of the effect.
- **Run outcomes** (from `receipt.json`, `client/summary.json` and `server.stderr`):
  - `Pass`: the client reports `FourAgentProtocolCompleteAwaitingIndependentReview` with exit 0 and no supervisor stop, and the Q2 run audit (`q2_run_audit.py`) re-derives that claim from the raw client evidence: archived request/response hashes, a parser replay of every turn, lineage, v3 timing, makespan arithmetic, cohort membership, verified cleanup, and a client-observed co-progress width (streamed requests active together over a sustained common window) of 4 for `n4` and for every `K+M` plan, and exactly agent-0 for `n1`. A serialized or slot-capped service fails it. See [adr063-q2-adversarial-controls.md](adr063-q2-adversarial-controls.md).
  - `Fail`: after the four-agent release, a failed client, `server_exited` or `client_timeout` (task failure), or a Q5 memory stop (`memory_stop`). A Metal out-of-memory marker in `server.stderr` (`kIOGPUCommandBufferCallbackErrorOutOfMemory`, `Insufficient Memory`) makes the kind `out_of_memory`.
  - `NotStarted`: Q5 preflight refusal, or a server exit, readiness timeout or memory stop before the client, or a client that ended before the release. The driver halts. `--resume` retries the same entry as a new attempt, and every attempt is kept.
  - `Invalid`: a missing receipt, a supervisor error, any other supervisor stop after the release, or a claimed Pass whose evidence the run audit rejects (for example a serialized service, a corrupted receipt or a makespan that disagrees with its clocks). Terminal, with no reruns after a task has started.
- **Peer unavailability.** A peer run that fails to load, or that fails with `out_of_memory` or a Q5 memory stop, marks the peer `Unavailable` with that run as evidence. Its later entries are `Skipped` and the remaining blocks run stock and experiment only. `--peer-unavailable-evidence FILE…` declares unavailability up front from earlier receipts (hashed into the plan), and no peer arm runs.
- **Experiment failure** (`Fail` in any stage, warmup included) stops the campaign: the stage is `Stopped`, later stages are `NotRun` and the family is `Failed`. The experiment is never rerun.
- **Comparison-arm failure policy** (ADR-063 Q4 amendment, operator decision 2026-11-12; plan field `comparison_failure_policy`, version 1: `{"version": 1, "reruns_per_position": 2, "rerun_failure_kinds": ["task"], "arms": [...]}` with the plan's measured arms among `stock`, `peer`, `splash`; never `experiment`, never the `peer_matched` diagnostic).
  - A `Fail` of kind `task` in a policy arm is discarded for timing and kept as a receipt: the attempt record gets `disposition: "rerun"`, the entry stays unsettled, and the driver launches the next attempt of the same entry at once, with the same derived case, before the next arm of the block. Warmup entries are handled the same way.
  - At most 2 reruns per entry. A third task failure settles the entry as `Fail` (`disposition: "exhausted"`) and records the arm in `unavailable` with `kind: "task_failures_exhausted"` and the three run directories as evidence. Its later entries are `Skipped`; the family verdict is `Unavailable`.
  - Only kind `task` qualifies. `q3_metrics.classify` gives that kind to every failure the client itself reports after the four-agent release with no supervisor stop and no out-of-memory marker. The client aborts the run at its first failing agent and stops the others, so the kind covers a broken workflow, a failed test with no repair cycle left, an API or context reject and a client request, agent or tool deadline (900 s, 7200 s, 45 s) alike. A comparison run that stalls past a client deadline is therefore rerun; this removes the reference arm's slowest run and works against the experiment, never for it. `server_exited`, the supervisor's `client_timeout`, `memory_stop` and `out_of_memory` failures, `NotStarted` and `Invalid` runs keep the handling above. An attempt that does not start between two failures does not use up a rerun.
  - The decision is `q3_stats.rerun_decision(policy, arm, outcome, failure_kind, prior_failures)`. It has no metric or effect input, so a passing run is never rerun and nothing is examined between a failure and its rerun. Timing reaches it only through the failure kind, as described above.
  - The policy is written into the plan before the first run and is part of the plan sha256, so `--resume` refuses a different rule. `--comparison-reruns 0` plans a new campaign without the field, under the earlier rule: stock and peer task failures are recorded, the campaign continues, and a peer or required-Splash failure is `Failed`. It does not make an older campaign resumable: this change edits frozen harness sources, so a campaign planned before it resumes only from the harness commit that started it (a resume from a later commit is refused or ends as `IdentityDrift`). A controls-only campaign follows its primary of record.
  - Design effects: block order and the Williams sequence are unchanged, and a block still contributes one passing run per arm. The stopping rule is evaluated only after a complete block and only on passing runs. In a block with a rerun, the arm that follows sees that arm twice or three times before it (the discarded attempts are truncated at the first failing agent); this is disclosed, not corrected, and each stage reports `rerun_sensitivity`: the makespan effects over the blocks without a discarded run, descriptive only. A comparison arm's speed is conditional on passing, which is neutral or against the experiment, and its failure rate is reported beside it.
  - A stock arm exhausted this way no longer has a vs-stock contrast to wait for; a stage with no contrast left stops as `no_contrast`. An exhausted arm is read from the attempt records (`disposition: "exhausted"`) as well as from `unavailable`, both by the verdict and by a resuming driver.
- **Resume.** `--resume` with the same `--output` and an identical plan continues a `Halted` campaign. An identical plan means the same cases, overrides, workloads, declarations and evidence. Attempts the driver was running when it died are scored from disk.

## Estimator (predeclared)

For a contrast of the experiment against reference arm R, over the complete blocks in which both runs passed:

$$d_b = \ln M_{E,b} - \ln M_{R,b}, \qquad \bar d = \tfrac1n\sum_b d_b, \qquad h = t_{0.975,\,n-1}\, \frac{s_d}{\sqrt n}$$

$M$ is the client's `own_output_four_task_makespan_seconds`, measured from the barrier release to the last validated task completion; model load is excluded. Each block is one cluster: one independent run per arm, each a whole cohort of conversations. Tokens and turns are never units. The reported ratio is $e^{\bar d}$ with 95% CI $[e^{\bar d-h}, e^{\bar d+h}]$, the improvement is $100(1-e^{\bar d})$ percent with its CI mapped the same way, and the half-width is $100(e^{h}-1)$ percentage points. The t quantiles for df 1–39 are tabulated in `q3_stats.T975`.

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
3. `Unavailable` (Qwen3.8 and Qwen3.6 only): no runnable Splash arm. This covers declared evidence, a load or memory
   failure, or a campaign without a Splash arm. A Win is never measured against the peer alone.
   Under the comparison-arm failure policy, rules 2 and 3 include a peer or Splash arm that failed its task in all
   three attempts of one block position, and a stock arm that did so is `Unavailable` here too ("no usable stock arm").
4. `Failed`: the peer failed its task. A failed peer cannot yield a win. Under the comparison-arm failure policy this
   is reached only by a failure the policy does not rerun (server exit, client timeout); a task failure rerun to a
   pass is counted in `arm_reliability`, not here.
5. `Failed` (Qwen families): the Splash arm failed its task. A failed Splash arm cannot yield a win. The same
   policy note applies.
6. `Pending`: the primary stage has not reached its stopping rule.
7. `Loss`: the experiment-vs-peer makespan ratio CI lies above 1, so the experiment is slower than the peer. This holds at any precision, because a longer makespan with an interval excluding a tie is a measured loss.
8. `Loss` (Qwen families): the experiment-vs-Splash makespan ratio CI lies above 1.
9. `Win`: every one of the following holds under the thresholds frozen in the campaign plan.
   - Every required contrast (stock, peer, and Splash where required) reached the precision target. The stopping
     rule waits for vs-Splash while a required Splash arm is available and has not failed.
   - Against v0.1.29 stock, the CI upper bound is ≤ 0.95.
   - Against the peer, the CI upper bound is ≤ 1.00. For Qwen3.6, the vs-Splash CI upper bound must also be ≤ 1.00.
   - Neither fairness effect against stock (worst-slot p95 TTFT, worst-slot max SSE gap) has a CI lower bound above the 2.5-point neutral band (ratio 1.025).
   - Quality is not unresolved.
   - No run observed a serious or critical thermal state.
   - Every passing experiment run has a passing Q5 memory verdict.
10. `Inconclusive`: everything else, with every unmet condition listed as a reason. An unresolved interval is inconclusive, not faster.

**Quality.** `--quality-verdict MANIFEST` takes a `quality_campaign.py` manifest that carries a Q1 paired verdict. It counts only if its subject runs' receipts show the experiment binary sha256. `Loss` or `BelowFloor` → `Failed`. `Inconclusive`, `Insufficient` or an unbound binary → unresolved, so no win. Without a Q1 verdict, quality is task-completion parity: no experiment run failed.

The `n1` and mixed-phase stages are reported in `verdict.json` `controls` with the same effects and arm summaries. They do not change the family verdict, except that an experiment failure in any stage fails the family.

## Outputs

- `campaign.json` (`hf2q.adr063.q3-campaign.v1`) contains:
  - the plan and its sha256;
  - the frozen identities and their sha256;
  - sessions, and every stage with its warmups, blocks, Williams order and stop;
  - every attempt with its argv, Q5 receipt sha256, Q5 verdict, stop, client status, outcome, failure kind, makespan and metrics, and its `disposition` (`rerun` or `exhausted`) where the comparison-arm failure policy acted on it;
  - unavailable arms with their evidence, and the halt state.
- Q5 run directories are under `runs/`, and the derived cases are under `cases/<workload>-<arm>.json`. The `n4` case files are exact copies of the source cases.
- `verdict.json` (`hf2q.adr063.q3-verdict.v1`) contains:
  - the family verdict and its reasons, the campaign status (`campaign_status`) and the halt record (`halted`: entry, reason, resumable);
  - in a default campaign, a control stage that halts the campaign (for example `Invalid` after the primary completed) does not replace the primary verdict: `verdict` stays the primary's, while `campaign_status` and `halted` record the halt, and every primary row note states `Q3 verdict <verdict>, campaign <status>`;
  - primary effects with CIs, secondary effects and per-arm summaries (median makespan and goodput, memory, maximum observed thermal state);
  - controls, failures, peer status, quality inputs and patch-review status;
  - the failure rule the verdict was computed under (`comparison_failure_policy`, null for a plan without one, and `failure_rule`), `discarded_runs` (stage, block, position, arm, attempt, run directory, receipt sha256, failure kind, the client's reason, the failing agent's own error as `detail`, and `client_elapsed_s`, the time to the abort; `makespan_s` is null because the client records a makespan only for a successful run) and `arm_reliability` (per arm: attempts that ran the task, every post-release failure with `failures_by_kind`, failure rate and its Clopper-Pearson 95% interval, split into measured and warmup runs, with reruns spent and exhausted positions; the interval is nominal, since attempts are rerun until one passes); each stage's arm summary carries `reliability` for its measured runs, and each stage a `rerun_sensitivity`;
  - rejected trials (Invalid, Abandoned, Skipped or not-started attempts), the number of discarded warmups, and the predeclared design.
- `q3_campaign.py --rows CAMPAIGN --evidence-root "$E"` prints a `{groups, rows}` fragment in the `hf2q.f2.scoreboard-rows.v1` vocabulary. It holds the experiment-vs-stock (baseline `v0128_stock`, the v0.1.28 release) and experiment-vs-peer makespan effects as `% improvement`, with `$E/...` receipt references and JSON extraction keys into `verdict.json`, so `f2_scoreboard_render.py --verify` can re-read them. Merge it into `docs/research/adr063-f2-scoreboard-rows.json` only after independent review. Statuses: `Win`/`Inconclusive`/`Unavailable` → `Descriptive`, `Loss` → `NegativeResult`, `Failed` → `Failure`.
- `--rows CAMPAIGN --of-record [--quality-gate REFERENCE=MANIFEST ...] [--review FILE]` emits a reviewed verdict of record for the F2 "Q3 verdicts of record" tier (F2 closure option B), in place of that campaign's default fragment. The group `<family>-q3-record` carries `q3_record`:
  - the verdict, `campaign_status`, the halt (entry, stage, reason from `campaign.json`), every stage's status, the primary stop and the reasons;
  - the experiment as `hf2q <commit8>` (row variants and every `native_peer` statement use the same label, never "main HEAD" or another `--experiment-label`);
  - peer unavailability, read as "peer out of memory (prior evidence)" when declared evidence includes a server log with a Metal out-of-memory marker;
  - fairness effects the verdict could not estimate although the primary has paired blocks (`unmeasurable`; a primary stopped before its first block has no fairness effect at all), thermal and memory validity, and the final-source hashes still pending patch review;
  - one quality gate per reference kind (`stock-hf2q`, `peer`, `<kind>-native`). A gate is `Pending` while the `quality_campaign.py` manifest is absent or running, `Halted` when it halted, otherwise `Pass` or `Fail` from its Q1 paired verdict. A settled manifest is refused unless its subject receipts are the experiment binary and its reference receipts are the campaign's arm of that kind: the stock binary for `stock-hf2q`, the peer binary for `peer`, and for `<kind>-native` an arm of that engine kind with one native readiness identity and one binary;
  - the `$E/` review.

  The tier requires the `stock-hf2q` gate and the `peer` gate, or, when the same-GGUF peer is unavailable, the family's native stand-in (`ds4-native` for DeepSeek-V4). Primary rows carry `q3_of_record: true`: the makespan and worst-slot TTFT p95 / SSE-gap max effects vs stock and peer, each with a verified full-precision `half_width_pp`, plus per arm the median prompt and completion tokens and completion tokens per makespan-second over the primary stage's passing runs, re-read from each run's `client/summary.json`. Completed control stages of the same campaign (`n1`, and mixed stages that completed) are `Descriptive` rows with `control_of` this verdict, shown in the Q3 controls section. Controls-only, native-comparison and native-peer campaigns are refused (their rows keep their own sections). The rows manifest holds one record per family, and a row outside the record that cites the record's `verdict.json` or `campaign.json` (for example the same campaign's default fragment) is refused unless it is a control of that record. `--verify` re-derives the record from `verdict.json`, `campaign.json` and the quality manifests, so a gate that settles after the merge is reported as drift until the fragment is regenerated.

  A `Failed` verdict of record whose primary stage was stopped by an experiment task failure (Q3 rule 1) gives one `Failure` `task_pass` row of record per failed experiment run, verified against `campaign.json`. Its note names the agents that failed on their own (not those the coordinator stopped afterwards), each with the target and compiler error code of its last failed `run_tests` and its error, read from the run's `client/summary.json`; the count is the agents that finished before the coordinator stopped the run, never a partial success. The tier shows it in the stock cell in place of a speed effect.

  The record's completed control rows against the same-GGUF peer carry the operator's peer wording, as controls-only rows do (below). The rows manifest may carry `q3_family_notes` (family → sentences): the reviewed scoreboard sentences and required disclosures, rendered as "Family notes" under the tier. They cite the reviews and add no number of their own. It may also carry `q3_verdict_notices` (family → sentence), appended to that family's verdict in the tier, for example a record that a pre-registered campaign will replace.

  A same-GGUF peer row names its baseline from the case files the peer runs launched: a peer that ran `--spec-type none` is the "peer, speculation off (matched settings)", in row notes and in the tier's peer sentence, never "best valid configuration". Outside the tier, `f2_scoreboard_render.py` never labels a comparison against the peer a Win: a separated faster or slower result against the peer reads `Faster` or `Slower` and is listed with the non-wins. A family whose verdict of record is `Loss` is never shown as a Win against stock: the tier's stock cell reads "not a Q3 Win: the family verdict is Loss vs the peer".

- `--rows CAMPAIGN --superseded-by RECORD_VERDICT_JSON` keeps a former verdict of record published beside the record that replaced it (for example Qwen3.8 v7, replaced by v11). The group `<family>-q3-superseded` holds its primary and completed-control makespan effects as `Descriptive` rows carrying `superseded_by` (which must be the family's current record's `verdict_ref`), with the campaign status and, for same-GGUF peer rows, the operator's peer wording in each note. They render only in "Superseded Q3 records", never among wins or non-wins.

## Controls-only campaign (#369 amendment)

The pre-registered amendment on [#369](https://github.com/robertelee78/hf2q/issues/369#issuecomment-6551952975) keeps the v7 `n4` and `n1` results of record and does not re-run them. A new campaign runs only the mixed stages `3+1`, `2+2` and `1+3`, with the same arms, binaries and identities. A width-only `Invalid` stands as reported.

- `--controls-of PRIMARY_VERDICT_JSON` enables the mode and needs exactly one `--family`. `--workloads` must then name distinct control keys (`n1`, `3+1`, `2+2`, `1+3`), never `n4` and never a stage that is `Complete` in the bound `campaign.json`: completed stages are never re-run. In every v7 record `n1` is `Complete`, so only `3+1`, `2+2` and `1+3` can run. Without the flag, a list that does not start with `n4` is refused, and the default campaign is unchanged.
- The record must be a file named `verdict.json` (a superseded copy such as `verdict.pre-patch-review.json` is refused). Its primary stage must be `Complete`, and it must bind the `campaign.json` beside it (its `campaign_sha256` and `identities_sha256`). The plan records `controls_of`: both files' paths and sha256, the family, the recorded verdict, primary effects and stage statuses, and the recorded frozen identities and harness source map.
- The launch is refused when anything outside the harness differs from the recorded campaign:
  - the family, the measured arms and the peer-unavailable evidence sha256;
  - the host identity and the Python sha256;
  - per arm: case sha256 (hf2q and peer), binary sha256, model sha256 and size, and override contents. A file named in the arm's argv counts by its sha256, and other non-path values by their literal value; model, binary and directory paths are bound by their own identities instead.
  - for the hf2q arms, the declared commits; for the peer, `build_info`, sibling dylibs and native runtime.
- Harness sources may differ, and so may `CLIENT_APPROVAL_TEMPLATE`. The approval template pins the client's source hashes, so newer client modules require a new template.
- `verdict.json`: `verdict` is `ControlsOnly`, `primary` and `primary_workload` are null, and `controls` is as in a full campaign. `primary_of_record` holds the path, sha256, recorded verdict and effects, plus both harness source maps (`harness.of_record`, `harness.this_campaign`, `harness.differs`) and `q2_run_audit_sha256` for both. `--verdict` refuses when either bound record file (`verdict.json` or `campaign.json`) has changed. Reasons list the campaign status, experiment and peer failures, peer unavailability and a quality loss (for example a rejected patch review).
- Halting is unchanged. An experiment failure stops the campaign, `Invalid` (including a width-only audit failure) is terminal and stands, and `IdentityDrift` is terminal.
- `--rows` writes to group `<family>-q3-controls`. Every row carries `control_of` (the `$E/` reference of the primary of record) and is never derived from a Win or Loss:
  - A control stage that reached its stopping rule gives `Descriptive` makespan effect rows. Each note ends with the campaign status and, when a stage halted the campaign, the halted stage, entry and reason. A same-GGUF peer row's note carries the operator's peer wording for that stage: the task-makespan sentence on the slower arm's base (X = 1 − faster/slower, medians), the peer/hf2q completion and prompt token ratios and the per-output-token rate ratio (paired over blocks, with per-arm medians and the rate CI), and hf2q's worst-slot p95 TTFT relative to the peer's.
  - A control stage `Stopped` by an experiment task failure gives one `Failure` `task_pass` row per failed experiment run, verified against `campaign.json`.
  - `f2_scoreboard_render.py` renders these rows only in their own "Q3 controls of a primary verdict of record" section, never among wins or non-wins.
- **Operating rule: freeze the record before binding.** The plan binds the record's `verdict.json` and `campaign.json` bytes. Finish every recomputation of the record first (`--verdict` with `--quality-verdict` or `--patch-review`), then make both files read-only (`chmod a-w`) before the controls campaign starts. A later change breaks `--resume` (plan mismatch) and the verdict recompute (`ValueError`), and nothing can re-bind the campaign.

```sh
python3 -B scripts/four_agent/q3_campaign.py --family qwen38 "${STOCK[@]}" "${EXP[@]}" "${COMMON[@]}" \
  --peer-var CLIENT_APPROVAL_TEMPLATE=$T/peer-qwen38-client-approval.json \
  --controls-of "$Q3/<v7 campaign>/qwen38/verdict.json" --workloads 3+1,2+2,1+3 \
  --output "$Q3/<controls campaign>/qwen38"
```

## No-stock-arm campaigns (ADR-063 Q4-DS4-RACE, decisions D1 and D2)

Design: `docs/research/adr066-iq2xxs-subset-design.md` §7.4 and §8. v0.1.28 and v0.1.29 cannot open ds4's DeepSeek-V4 Flash 0731 file (IQ2_XXS), so a campaign on that file has no stock arm. Its vs-stock terms come from a same-artifact record on hf2q's own artifact, measured with the same candidate binary.

- **Arguments.** `--no-stock-arm` needs exactly one of `--stock-record VERDICT_JSON` (the verdict-of-record mode) or `--native-comparison` (no vs-stock row exists). `--stock-record` without `--no-stock-arm` refuses, and so does `--stock-record` with `--peer-unavailable-evidence` (the campaign is judged against the peer, so it would have no contrast). With `--no-stock-arm`, `--stock-bin`, `--stock-commit`, `--stock-label` and `--stock-var` refuse, and so does `--controls-of` (the controls run in the same campaign). `--stock-record` needs exactly one `--family` and applies to `deepseek4` only.
- **Plan.** The measured arms omit `stock`, `declared.stock` is `null`, and only the experiment needs a binary and a 40-hex commit. `plan.stock_record` binds the record, following the `--controls-of` pattern: the paths and sha256 of its `verdict.json` and `campaign.json`, the family, the primary status, the recorded verdict, the stock and experiment declarations, the stock and experiment binary sha256 and the hf2q artifact sha256. The binding refuses a record that:
  - is not a `verdict.json` beside the `campaign.json` it binds (`campaign_sha256`, `identities_sha256`);
  - is not a complete `deepseek4` primary (a `Complete` primary stage of a default campaign, not controls-only);
  - is itself a no-stock campaign, or has a native peer arm or a native comparison;
  - has no vs-stock effect or no hf2q artifact sha256.
- **Launch and every run.** `launch_problems` refuses when the record's experiment binary sha256 differs from this campaign's `--experiment-bin`, or when either bound file changed since binding. The frozen identities carry the experiment binary sha256 and the bound files' current sha256 (`identities.stock_record`), which are re-checked before every run, so a rebuilt binary or a changed record halts the campaign with `IdentityDrift`. `plan.stock_record` is part of the plan a `--resume` must reproduce. **Operating rule:** freeze the record read-only (`chmod a-w`) before binding, as for `--controls-of`.
- **Stopping rule.** It waits on `vs_stock` only when the plan has a stock arm. A stock-record or no-stock native-comparison campaign waits on `vs_peer`. When the peer becomes unavailable or fails, such a stage has no contrast left: it stops at once with reason `no_contrast` (`q3_stats.NO_CONTRAST`), and so does every later stage, instead of running experiment-only blocks to the maximum.
- **Verdict.** `verdict.json` `stock_record` re-binds the record by sha256 and carries its primary `vs_stock` makespan effect and the `vs_stock` fairness secondaries, its stock and experiment failures, and its thermal and memory validity. The family verdict uses them as its `vs_stock` and `fairness`. `stock_failures` and `experiment_failures` are the union of both campaigns, each record entry tagged `(stock record <path>)`. `thermal_valid` and `memory_valid` are the conjunction of both. `q3_stats.family_verdict` takes `stock_source` (the record's path, sha256 and status): every vs-stock reason names it, and a record that is missing, changed since binding or not `Complete` gives `Pending` with that reason, never a silent skip. The release decision rules are unchanged, so a v0.1.30 Win still needs the bound candidate/v0.1.29 ratio's 95% CI upper bound ≤ 0.95, precision and the fairness band, all from the record.
- **No-stock native comparison.** Without a record (`--no-stock-arm --native-comparison`), `vs_stock` is the empty effect and `fairness` is `{}`. The native branch makes the verdict `Unavailable` with the comparison reason, as for any native comparison.
- **Rows.** The workload text reads "stock `<label>` on hf2q's artifact, from record `<$E ref>`" (or "no stock arm" without a record). The vs-stock rows read `verdict.json` `stock_record.effects.vs_stock` (and `stock_record.secondary.<metric>.vs_stock` under `--of-record`), carry the record's hf2q artifact sha256, and note "baseline stock `<label>`, from the same-artifact record `<$E ref>` on hf2q's artifact". `--of-record` accepts a stock-record campaign. Its `q3_record` carries `stock_record` (the record's `$E/` verdict ref, hf2q artifact sha256 and stock label), and the tier labels the stock cell and the fairness cell "vs stock (hf2q artifact record)". The quality-gate references stay `stock-hf2q` and `peer`: the `stock-hf2q` gate must run the record's stock binary with the record's (equal) experiment binary as subject, every run on the record's hf2q artifact, and the `peer` gate must run this campaign's peer binary on the peer arm's artifact (ds4's file).
- **Same-file ds4 arm (D2).** `q5_cases/native-ds4-deepseek4-samefile.json` declares `artifact_relation: same_as_hf2q` in its approval `native_ds4` block, a closed field (`different` by default; not a case variable, bound by `case_sha256` and the ds4 `campaign_identity`). The plan labels the arm `ds4 c97964b (same ds4 file)` and its F2 baseline is `ds4_samefile`. It runs only with `--native-comparison` (`--peer-unavailable-evidence` refuses: the same-file peer runs), and the different-artifact case only with `--peer-unavailable-evidence`. Launch needs the native and the hf2q artifact sha256 both present and equal; the `MODEL_FILE` name check stays. TensorFold and oMLX refuse a `same_as_hf2q` case. `native_peer`, row notes and the scoreboard say "same file", task makespan only, and the scoreboard renders same-file rows in their own "Native-engine comparisons (same file as the hf2q arm, task makespan only)" section. The weights are the same bytes, but prompts are not byte-identical (ds4 renders tool schemas first, and its Think Max prefix differs).
- **Artifact verification.** No sidecar receipt accompanies the model file. Each case's `MODEL_SHA256` is verified by the §6.1 T2 full-file digest, `scripts/four_agent/ds4_file_digest.py` (per-tensor and full-file sha256 in one streaming pass), whose receipt is `/Users/robert/.local/state/hf2q/v0130/q4-ds4-race/t2-digest.json`.

Root commands, in order (§7.4; variables as in "Root commands" below; `$CAND`/`$CAND_COMMIT` are the candidate binary and its 40-hex commit, one candidate for all three campaigns):

```sh
R=/Users/robert/.local/state/hf2q/v0130/q4-ds4-race
DS4FILE=/Users/robert/.local/share/hf2q/peers/ds4-c97964b/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
# §6.1 T2: the file digest that verifies MODEL_SHA256 (heavy disk; reads the whole 87 GB file once)
python3 -B scripts/four_agent/ds4_file_digest.py "$DS4FILE" --output "$R/t2-digest.json" \
  --expect-sha256 ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0

# 1. Stock record on hf2q's own artifact (existing harness). Its own family_verdict is Unavailable (no peer) and is
#    not the verdict of record. Freeze verdict.json and campaign.json read-only before step 2 binds them.
HF2Q_Q5_MODEL=$MODELS/deepseek4/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf \
HF2Q_Q5_MODEL_SHA256=50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6 \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=$E/current-port-balanced-c1-normal-n4-v1/client-approval.json \
  python3 -B scripts/four_agent/q3_campaign.py --family deepseek4 "${STOCK[@]}" \
  --experiment-bin "$CAND" --experiment-commit "$CAND_COMMIT" "${COMMON[@]}" \
  --peer-unavailable-evidence $E/peer-arms-v1/peer-deepseek4-20261030T165847Z/receipt.json \
  --peer-unavailable-evidence $E/peer-arms-v1/peer-deepseek4-20261030T165847Z/server.stderr \
  --peer-unavailable-evidence $E/peer-arms-v1/peer-deepseek4-mmap-20261030T165948Z/receipt.json \
  --peer-unavailable-evidence $E/peer-arms-v1/peer-deepseek4-mmap-20261030T165948Z/server.stderr \
  --output "$R/q3-stock-record/deepseek4"
chmod a-w "$R/q3-stock-record/deepseek4/verdict.json" "$R/q3-stock-record/deepseek4/campaign.json"

# 2. ds4-file campaign of record: experiment and peer on ds4's file, n4 primary then the controls. The hf2q approval
#    template is step 1's: the case's approval `set` writes `model` from MODEL_ID ("DeepSeek V4 Flash", the file's
#    general.name that hf2q serves) at run time.
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=$E/current-port-balanced-c1-normal-n4-v1/client-approval.json \
  python3 -B scripts/four_agent/q3_campaign.py --family deepseek4 \
  --hf2q-case scripts/four_agent/q5_cases/deepseek4-ds4file.json \
  --peer-case scripts/four_agent/q5_cases/peer-deepseek4-ds4file.json \
  --peer-var CLIENT_APPROVAL_TEMPLATE=$T/peer-deepseek4-client-approval.json \
  --no-stock-arm --stock-record "$R/q3-stock-record/deepseek4/verdict.json" \
  --experiment-bin "$CAND" --experiment-commit "$CAND_COMMIT" "${COMMON[@]}" \
  --output "$R/q3-ds4file/deepseek4"

# 3. Same-file native comparison: the experiment and ds4 (--ctx 393216) on the same file; native_peer, task makespan only.
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=$E/current-port-balanced-c1-normal-n4-v1/client-approval.json \
  python3 -B scripts/four_agent/q3_campaign.py --family deepseek4 \
  --hf2q-case scripts/four_agent/q5_cases/deepseek4-ds4file.json \
  --peer-case scripts/four_agent/q5_cases/native-ds4-deepseek4-samefile.json \
  --peer-var CLIENT_APPROVAL_TEMPLATE=$T/native-ds4-deepseek4-client-approval.json \
  --native-comparison --no-stock-arm \
  --experiment-bin "$CAND" --experiment-commit "$CAND_COMMIT" "${COMMON[@]}" \
  --output "$R/q3-ds4file-native-ds4/deepseek4"

# The verdict of record's rows (q3_report through --rows), with both quality gates:
python3 -B scripts/four_agent/q3_campaign.py --rows "$R/q3-ds4file/deepseek4" --evidence-root "$R" --of-record \
  --quality-gate stock-hf2q="$R/qg-stock/<QG-stock campaign>/manifest.json" \
  --quality-gate peer="$R/qg-peer/<QG-peer campaign>/manifest.json" \
  --review "$R/reviews/<independent verdict review>.md"
```

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

# DeepSeek-V4 Flash, native ds4 arm (separately labeled; same evidence keeps the same-GGUF verdict Unavailable)
HF2Q_Q5_MODEL=$MODELS/deepseek4/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf \
HF2Q_Q5_MODEL_SHA256=50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6 \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=$E/current-port-balanced-c1-normal-n4-v1/client-approval.json \
  python3 -B scripts/four_agent/q3_campaign.py --family deepseek4 "${STOCK[@]}" "${EXP[@]}" "${COMMON[@]}" \
  --peer-case scripts/four_agent/q5_cases/native-ds4-deepseek4.json \
  --peer-var MODEL=/opt/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --peer-var MODEL_SHA256=<sha256 of that file> --peer-var DS4_BIN_SHA256=<sha256 of /opt/ds4/ds4-server> \
  --peer-var CLIENT_APPROVAL_TEMPLATE=$T/native-ds4-deepseek4-client-approval.json \
  --peer-unavailable-evidence $E/peer-arms-v1/peer-deepseek4-20261030T165847Z/receipt.json \
  --peer-unavailable-evidence $E/peer-arms-v1/peer-deepseek4-20261030T165847Z/server.stderr \
  --peer-unavailable-evidence $E/peer-arms-v1/peer-deepseek4-mmap-20261030T165948Z/receipt.json \
  --peer-unavailable-evidence $E/peer-arms-v1/peer-deepseek4-mmap-20261030T165948Z/server.stderr \
  --output "$Q3/deepseek4-native-ds4"

# Qwen3.8 27B, native TensorFold comparison (separately labeled; never the verdict of record). For DFlash2 use
# q5_cases/native-tensorfold-qwen38-dflash2.json, a dflash2 approval template and output qwen38-native-tensorfold-dflash2.
TFHUB=$HOME/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c
HF2Q_Q5_MODEL=$MODELS/qwen38-stock/Qwen3.8-27B-vanilla.gguf \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=$E/qwen38-matched-baseline1512-normal-n4-20261029T024609Z/client-approval.json \
  python3 -B scripts/four_agent/q3_campaign.py --family qwen38 "${STOCK[@]}" "${EXP[@]}" "${COMMON[@]}" \
  --peer-case scripts/four_agent/q5_cases/native-tensorfold-qwen38.json --native-comparison \
  --peer-var TF_BIN_SHA256=$(shasum -a 256 /opt/TensorFold/zig-out/native/bin/tensorfold-native | cut -d' ' -f1) \
  --peer-var TARGET_CONFIG_SHA256=$(shasum -a 256 "$TFHUB/config.json" | cut -d' ' -f1) \
  --peer-var CLIENT_APPROVAL_TEMPLATE=$T/native-tensorfold-qwen38-client-approval.json \
  --output "$Q3/qwen38-native-tensorfold"

# Native oMLX comparisons (separately labeled; never the verdict of record). Install per scripts/four_agent/README.md
# "Native oMLX arm" first (isolated clone $HOME/.local/share/hf2q/peers/omlx-6b413c9 and the pinned HF snapshots).
OHUB=$HOME/.cache/huggingface/hub
omlx_vars() {  # $1 = snapshot dir
  echo --peer-var MODEL_CONFIG_SHA256=$(shasum -a 256 "$1/config.json" | cut -d' ' -f1) \
       --peer-var TEMPLATE_SHA256=$(shasum -a 256 "$1/chat_template.jinja" | cut -d' ' -f1)
}
Q38=$OHUB/models--mlx-community--Qwen3.8-27B-4bit/snapshots/10c35caafbb80f7dc6a7a432cdd11af10a6d4818
HF2Q_Q5_MODEL=$MODELS/qwen38-stock/Qwen3.8-27B-vanilla.gguf \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=$E/qwen38-matched-baseline1512-normal-n4-20261029T024609Z/client-approval.json \
  python3 -B scripts/four_agent/q3_campaign.py --family qwen38 "${STOCK[@]}" "${EXP[@]}" "${COMMON[@]}" \
  --peer-case scripts/four_agent/q5_cases/native-omlx-qwen38.json --native-comparison $(omlx_vars "$Q38") \
  --peer-var CLIENT_APPROVAL_TEMPLATE=$T/native-omlx-qwen38-client-approval.json \
  --output "$Q3/qwen38-native-omlx"
Q36=$OHUB/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46
HF2Q_Q5_MODEL=$MODELS/qwen36-stock-adr063/Qwen3.6-35B-A3B-vanilla-Q4_K_M.gguf \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=$E/qwen36-template-order-coding-operating90-n4-20261027T231028Z/client-approval.json \
  python3 -B scripts/four_agent/q3_campaign.py --family qwen36 "${STOCK[@]}" "${EXP[@]}" "${COMMON[@]}" \
  --peer-case scripts/four_agent/q5_cases/native-omlx-qwen36.json --native-comparison $(omlx_vars "$Q36") \
  --peer-var CLIENT_APPROVAL_TEMPLATE=$T/native-omlx-qwen36-client-approval.json \
  --output "$Q3/qwen36-native-omlx"
G4=$OHUB/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/0d77464eeb233a2da68ebf9d7dc4edaac7db956d
HF2Q_Q5_MODEL=$MODELS/gemma4-google/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=$E/gemma4-full-input-n4-20261026T210724Z/client-approval.json \
  python3 -B scripts/four_agent/q3_campaign.py --family gemma4 "${STOCK[@]}" "${EXP[@]}" "${COMMON[@]}" \
  --peer-case scripts/four_agent/q5_cases/native-omlx-gemma4.json --native-comparison $(omlx_vars "$G4") \
  --peer-var CLIENT_APPROVAL_TEMPLATE=$T/native-omlx-gemma4-client-approval.json \
  --output "$Q3/gemma4-native-omlx"
# Qwen3.6 re-test on oMLX's tokenizer/sampler stack (quality campaign; oMLX is the labeled reference arm, never the
# subject). CLIENT_APPROVAL_TEMPLATE is declared by both cases, so each arm gets its own with --subject-var/--reference-var.
HF2Q_Q5_MODEL=$MODELS/qwen36-stock-adr063/Qwen3.6-35B-A3B-vanilla-Q4_K_M.gguf \
  python3 -B scripts/four_agent/quality_campaign.py --subject-case scripts/four_agent/q5_cases/qwen36.json \
  --subject-var HF2Q_BIN=$E/q3-experiment-bin-$EXP_COMMIT/hf2q \
  --subject-var CLIENT_APPROVAL_TEMPLATE=$E/qwen36-template-order-coding-operating90-n4-20261027T231028Z/client-approval.json \
  --reference-kind omlx-native --reference-case scripts/four_agent/q5_cases/native-omlx-qwen36.json \
  --reference-var CLIENT_APPROVAL_TEMPLATE=$T/native-omlx-qwen36-client-approval.json \
  $(omlx_vars "$Q36" | sed s/--peer-var/--reference-var/g) \
  --negative-control-receipt "$NC" --output "$Q3/quality-qwen36-omlx-reference"

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
# F2 scoreboard rows (merge after review); a reviewed verdict of record for the F2 tier:
python3 -B scripts/four_agent/q3_campaign.py --rows "$Q3/qwen38" --evidence-root "$E"
python3 -B scripts/four_agent/q3_campaign.py --rows "$Q3/qwen38" --evidence-root "$E" --of-record \
  --quality-gate stock-hf2q="$E/<quality gate vs v0.1.28>/manifest.json" --quality-gate peer="$E/<quality gate vs peer>/manifest.json" \
  --review "$E/reviews/<independent verdict review>.md"
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
