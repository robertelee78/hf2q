# Repeatable speculation-on Qwen serving, and timing-dependent routes across families

Status: revised after review (ReviewSpecRep: REVISE, 10 findings, all applied). The root ruled C at N4 and a corrected B everywhere else. Owner: Qwen38DetInv. Follows #717 (merged in #721), whose design note lists this as "Cause 2".

## Problem

With speculation `auto`, identical requests can produce different transcripts. Before this change `auto` was the default for Qwen3.6 at every width, for Qwen3.8 on one slot (N1) and on 2–3 slots, and was opt-in for Qwen3.8 at 4 or more slots. There are two causes.

1. **Wall-time cost controllers** (`src/inference/spec_decode/cost_controller.rs`).
   - Each proposer (MTP and history lookup) learns the ordinary target-token time from `Instant` measurements.
   - It disables itself for the rest of the generation after two consecutive 4-round windows whose measured round time is no better than the equivalent ordinary time.
   - Host load and cohort timing therefore decide whether later tokens come from 4-row K3/history verify rows or from M1 rows. These differ numerically (Class B; Q1 policy §2.1).
   - This applies at every width, N1 included.
2. **Cohort-dependent route.**
   - A lane with MTP or history state that is joint-ready joins any joint group of two or more as an M1 row, with drafting suppressed (`qwen35_joint_decode.rs` partition, both modes).
   - Alone, or when not joint-ready, the same lane runs its serial K3 or history round.
   - Whether a lane is alone depends on when other agents' requests arrive and finish. Pending verified output, drained without a forward over the following turns, changes other lanes' cohorts too.

T-J1-8 (b) shows this. With Qwen3.8 `auto`, the diverging turn-6 agent was always the first to arrive, and the #717 replay reproduced it (agent-0's output depended on whether it arrived first, 2/2 each way). With speculation off, arrival order did not matter.

## Decision (root, 2026-11-06)

### C at N4: speculation off by default on every joint-decode worker with `max_slots >= 4`, MoE included

`default_server_policy` (`qwen35_speculation.rs`) drops its `!moe` condition, so an unset `HF2Q_QWEN_SPECULATION` resolves to `off` for Qwen3.6 as well as Qwen3.8 on four-slot joint workers. Explicit `auto` still enables speculation, with B's semantics.

The ruling is for repeatability, and the evidence for the default is #719, not D1:
- #719 measured joint vs serial decode at N4 with speculation at its default in both arms. The joint/serial makespan ratio was 0.800 (95% CI 0.776–0.824) (S3 doc). There, the joint arm ran mostly joint M1 rows and the serial arm ran per-slot K3. Per-slot serial K3 at N4 therefore cost about 25% against mostly-joint M1 rows.
- B at N4 sends warmed MTP lanes to serial K3 on almost every tick, so it would move toward #719's slower arm [INFERENCE].
- D1 does not show that speculation gains nothing for MoE at N4. D1 §7 for Qwen3.6 is Inconclusive: decode throughput auto/off 1.014 (CI 0.978–1.051), measured with joint decode off, on the canonical workload where Qwen3.6 mostly fails the task. Makespan was excluded (1.076, CI 1.004–1.154) because per-agent completion differed. D2 covers Qwen3.8 only.

Follow-on edits: the off-reason string (`DENSE_WIDE_DEFAULT_OFF_REASON` becomes `JOINT_WIDE_DEFAULT_OFF_REASON`), the `default_server_policy` doc and tests, the `serve_qwen38_opencode.sh` launcher comments (`serve_qwen36_opencode.sh` carries no speculation comment), `docs/operator-env-vars.md` and the CHANGELOG.

### B: request-local route per tick, used wherever speculation is on

B covers N1 (only the cost change matters there), 2–3 slots, workers with joint decode off, and explicit `auto` at N4. It has three parts.

1. **One decision point per tick.**
   - `Qwen35DecodeState::next_tick_kind(&mut self) -> Result<Qwen35TickKind>` is the only place that chooses a tick's route. `decode_tick` matches on its result (or on the turn's plan); no second decision exists to disagree with it (no debug assertion). Its body is today's chain of decisions, moved into one function:
     - pending output → `Commit`;
     - history lookup with drafts and an admitting controller → `History(drafts)`. The history miss path, the MTP drop on a cost-disabled miss and the ledger sync move here unchanged.
     - MTP state, at least 4 output tokens remaining and an admitting controller → `K3`;
     - otherwise → `Ordinary`: a plain target step, or the coherent target plus MTP catch-up step.
   - `plan_tick` computes the kind in the joint partition's pre-pass and stores it on the lane for the tick, so the serial loop executes the same decision. A lane at its step bound gets no plan; `decode_tick` finishes it without consulting a proposer.
2. **Joint eligibility by kind.**
   - In repeatable mode the partition's classify returns `None` for `Commit`, `K3` and `History` lanes. They never join a joint group, so a speculating lane runs its round whatever the cohort.
   - `Ordinary` lanes join as today. In padded mode they always run the joint row, alone included (#717). In exact mode they run serial alone and joint in company, which are bitwise equal (`qwen36_joint_decode_matches_serial_decode_bitwise` covers both the plain and the `LogitsAndHidden` catch-up row).
   - Eligibility is decided per lane by `Qwen35DecodeState::joint_lane_class` and tested there for every kind; the partition itself is unchanged and admits exactly the lanes classified `Some` (its existing ragged-eligibility tests).
3. **Deterministic cost controllers.**
   - The controllers keep their algorithm: 4-round windows, disable after two unprofitable windows, at least one ordinary observation before any speculation.
   - In repeatable mode they are fed modelled costs in ordinary-decision units: an ordinary decision is 1 unit, and a round is a fixed per-proposer constant. A window is then unprofitable exactly when its mean target-equivalent decisions per round are ≤ C, which is a pure function of the transcript.
   - **Constants**, from the controllers' own counters in existing receipts (no new speed runs). Rows are counter totals, with rounds = `proposals` and decisions ≈ `accepted + proposals`:

     | Constant | Value | Rounds | Round time | Ordinary decision | Source |
     |---|---|---|---|---|---|
     | `C_mtp` | 1.26 | 102 | 79.2 ms | 63.0 ms (equivalent-ordinary s ÷ 339 decisions) | D2 H1 runs, Qwen3.8 at 38485a23: `$E/d2-v1/extra-runs/{A/01,A/10,N/01,N/14}-*-H1-*.metrics/last.prom` |
     | `C_hist` | 1.49 | 17 | 96.5 ms | 64.7 ms | same four receipts |

   - `C_hist` is one constant for every row count, because aggregate counters cannot split it by rows.
   - There are no Qwen3.6 receipts with these counters, so Qwen3.6 uses the Qwen3.8 constants [INFERENCE: its MoE M1/K3 cost ratio may differ]. The constants are recorded in code with this source.
   - The source runs were four-slot `adaptive` with joint decode, so their ordinary-decision time is a joint row's (prepare to the lane's own finish). Speculation now runs by default only at one to three slots, where a serial ordinary decision is cheaper. ADR-044's one-slot profile suggests about 2.1 for MTP (68–75 ms per round against about 34 ms per token) [INFERENCE], so 1.26 probably undercharges a round there.
   - With no speed runs allowed, the accuracy of these constants is the only protection for speed; if one is wrong, the cost is speed, not correctness.
   - **Telemetry stays measured.** `/metrics` `*_round_seconds_total` and `*_equivalent_ordinary_seconds_total` keep wall-clock values from a separate measured accumulator. The modelled units only drive the gate.
   - The `cost_controller.rs` module comment changes: the repeatable gate is an acceptance threshold against a fixed cost model; the measured gate remains under `adaptive`.

Switch: `HF2Q_QWEN_SPECULATION_ROUTE` (`documented-escape`).
- `repeatable` (default): parts 1–3.
- `adaptive`: today's cohort-dependent eligibility and measured cost gate, for A/B and as an escape hatch.
- Read once at worker start and logged with the speculation policy line.

No model or kernel change.

### Why not A (speculation inside the joint step, ADR-065 J2–J6)

A would keep joint batching and speculation together. It needs J2 (per-slot DeltaNet capture), J3 (KV write and multi-row attention in the joint encoder), J4/J5 (joint verify windows and qualification) and the P-track primitives, which is far beyond this story.

A is repeatable only under three conditions:
- J6 routes solo windows through the joint reference route REF(w);
- window decisions are request-local;
- the cost gate is deterministic (B part 3).

As specified, ADR-065 conflicts with the second and third:
- A2 allocates draft depth jointly across slots: it scores all 256 four-slot allocations with a least-recently-served tie-break, so it is cohort-dependent by design.
- A1's cost table is timed on the host at load and re-timed while idle on drift, so it can differ between runs.

A2 can therefore ship only under `adaptive`, or redesigned with per-lane depth from a static table. That is recorded in ADR-065 (A1/A2 note).

## What changes for users
- **Unset speculation on a four-slot joint worker** (Qwen3.8 and now Qwen3.6): off. Every decode step is an ordinary joint or serial row, repeatable since #717. Qwen3.6 N4 default serving no longer runs lone-lane K3.
- **N1, 2–3 slots, joint-off workers, explicit `auto`:** speculation on, with request-local routes and the modelled cost gate. A speculating lane in a busy turn runs its round instead of a joint M1 row. A cost-disabled proposer is decided from acceptance, not timing.
- **Speculation off:** unchanged.

## Gates (correctness only)

1. **Model-free.**
   - `next_tick_kind` tests: pending output; drafts and no drafts; history miss with the MTP drop; fewer than 4 remaining; cost-disabled; no MTP.
   - Lane-class tests (`joint_lane_class`): planned `Ordinary` joins (a coherent MTP catch-up lane included); planned `K3`, `History`, unplanned and step-bound lanes stay serial; `adaptive` keeps the phase-1 class.
   - Modelled-controller tests: window arithmetic; a disable at mean ≤ C; identical decisions for an identical transcript; telemetry still measured.
   - Policy tests: Qwen3.6/MoE `off` at four slots and `auto` below.
   - The existing speculation-equals-greedy model-free tests.
2. **Official-artifact.**
   - `qwen38_slot_aware_speculation_auto_matches_off` (T-J1-7) and the padded and exact joint gates still pass.
   - **New:** `qwen38_k3_and_history_rounds_are_slot_independent_bitwise` and its Qwen3.6 twin. Two targets run to their bound: a copy prompt that exercises history rounds, and a free-generation prompt with about 50 K3 rounds. Each target is compared against the same request in slot 2 with empty peers, in these variants:
     - a repeat of that reference;
     - peers prefilled, mid-decode, or decoding in the same turns;
     - the target in slots 0, 1 and 3.

     Logits, hidden rows, committed tokens, routes, cursors, KV codes and scales, conv/recurrent state and the MTP cache must be bitwise equal. The reference route is also printed for comparison across processes.
   - **Finding (2026-11-06).** In serving replay, the K3 accept pattern changed from run to run even for a solo, fully sequential request; committed tokens did not change. The repeat variant reproduced it in one process on both families: on identical inputs, a round's MTP drafts differed (for example `[198, 727, 10562]` vs `[1944, 261, 310]`).
     - **Cause:** the MTP attention-output encoder dispatched the sigmoid gate multiply and the `wo` projection concurrently, with no RAW barrier (`mtp.rs` `forward_full_attention`). The target path had the same race and fixed it in ADR-015 iter21.
     - **Fix:** a `memory_barrier` between the two dispatches.
     - **Result:** with the barrier, all 14 variants per family are bitwise equal, and the reference routes are identical across four processes.
3. **Serving replay** (model load, grant). The cohort replay tool gains an arrival-order arm (T-J1-8 (b): agent-1 turn-4 running, then turn-6 of agents 0/3/2 in orders 0,3,2 and 3,0,2) and a route witness.
   - **Route witness:** per-tick debug lines with kind and cohort size. Each arm must show K3 or history rounds in busy turns, and catch-up rows both alone and in company. An arm without the routes it exercises is invalid.
   - **Arms:** Qwen3.8 with explicit `auto` on 4, 3 and 2 slots, `repeatable` vs `adaptive`.
   - **Pass:** under `repeatable`, agent-0 and agent-3 turn-6 are byte-identical across arrival orders at each width. The `adaptive` control is expected to reproduce T-J1-8 (b)'s split; a non-split control is reported as weak evidence.
   - **Qwen3.6:** first look for a recorded `auto` split in existing Qwen3.6 receipts. If none exists, its replay is declared vacuous and gate 4 stands alone.
   - **Controller arm:** one arm in which the modelled controller disables MTP. A development-only override raises `C_mtp` (`HF2Q_QWEN_SPEC_COST_MTP`), so the disable path runs and must be repeatable across orders.
4. **Four-agent repeat** (model load, grant). Q5 runs, checked with `q1_decision_policy.py transcripts`.
   - Qwen3.6 repair1 at its new default (`off`, joint): 4 runs, `ExactTranscripts`.
   - Qwen3.6 with explicit `auto` (B): 4 runs `repeatable` (`ExactTranscripts`), plus 2 `adaptive` as control.
   - Qwen3.8 with `QWEN38_SPECULATION=auto`: 4 + 2.
   - Cost gates: not run (operator directive 2026-11-06).

## Results (2026-11-06/07, branch head 3179902d, mlx-native 0.16.0)

Evidence lives under `~/.local/state/hf2q/v0130/q38-det-inv/specrep/`; the runner is `run_specrep_gpu.sh`.

| Gate | Result |
|---|---|
| 1 Model-free | `serve::api` 1587 passed, 0 failed; `cost_controller` 7 passed; replay tool 21 passed. |
| 2 Official-artifact | All pass: T-J1-7 (9/9 identical to off; MTP 151 proposals, history 138), the padded and exact joint gates, and both slot-independence gates (14/14 variants bitwise per family, routes identical across four processes after the MTP barrier fix; `p1-*`, `k3fix-*`). |
| 3 Qwen3.8 replay | w4 `repeatable` and the controller arm pass: outputs byte-identical across orders 032 and 302, and every agent's route sequence identical; the controller's MTP cost-disable fired in both arms. Four same-order processes: one route sequence per agent (four before the fix). `adaptive` w4 kept outputs but changed routes across orders. w3 and w2 are invalid: anchored slots are lost below four slots, so cached-token resume differs by order (#727). `p2-20261106T232222Z`, `p2c-20261107T000712Z`. |
| 3 Qwen3.6 replay | Turn-0 order arms, joint on and off. `repeatable` passes: outputs and route sequences identical across orders, and joint on and off give the same route mix. `adaptive` with joint on reproduces the recorded agent-3 turn-0 split. `p2b-*`. |
| 4 Qwen3.6 Q5 | Default `off` ×4: `ExactTranscripts` (0/30 groups differ). Explicit `auto`, `repeatable` ×4: `ExactTranscripts` (0/30). `adaptive` ×2: `ExactTranscripts`. All `CalibrationPass`, cleanup proven. `p3-repeat`. |
| 4 Qwen3.8 Q5 | `QWEN38_SPECULATION=auto`, `repeatable` ×4: `ExactTranscripts` (0/28). `adaptive` ×2: `ExactTranscripts`. All `CalibrationPass`, cleanup proven. `p4-repeat`. |
| Cost gates | Not run (operator directive 2026-11-06). |

## Audit: other timing- or width-dependent route choices

Read-only audit of main after #721 (RouteAudit; review corrections applied).

**SAFE** (alternatives proven bitwise, or decided by the request alone):
- Gemma: warm and cold joint prefill vs per-slot, batched decode body, decode attention buckets, dense and lm-head width loops, MoE `mv_id`, tiny cold prefill;
- DeepSeek-V4: joint decode at every width and the B=4 body;
- Qwen: prefill chunk ends, `la_capture_active` (transaction-scoped, cleared on every exit), slot-anchor vs live-prefix hits.

Neither Gemma nor DeepSeek serving has a wall-time speculation gate.

**Numerics change by timing.** One follow-up issue each (#725–#730); not in this story.

| # | Family | Route choice | Decided by | Why numerics change |
|---|---|---|---|---|
| 1 (#725) | Gemma | Long prompts (> 4,096 tokens) installed together: later chunks use `Tiled` F16 attention instead of vec-hybrid alone. The chunk is `budget / lanes_left`, and a 32-row lane takes `mm_id` jointly but `mv_id` alone (`engine.rs` ~14555–14735, ~6613–6637) | how many long prompts overlap | Tiled is documented as not bitwise per-slot; chunk size is Class B |
| 2 (#726) | DeepSeek-V4 | Prefill chunk size: cooperative chunk = 2048 ÷ width; Mixed-step serial cap when any peer decodes; alignment catch-up sized by the leader's cursor (`engine.rs` ~8185–8197, ~8747–8798, ~9325–9460) | cohort width, arrival | Bitwise equality is proven only at equal chunks; chunk vs chunk is unmeasured [INFERENCE] |
| 3 (#727) | Shared scheduler (Qwen and Gemma) | A preferred (anchored) slot is lost when it is not free or fails the shared KV budget counting others' reservations; the request is cold-prefilled (`src/serve/scheduler.rs:1189-1225`; Gemma `engine.rs:13413`) | other requests' timing and reservations | Cold vs resume is Class B |
| 4 (#728) | Gemma | Stable warm batch on `Tiled` vs per-slot admission, decided by the 25 ms wall-clock admission window (`engine.rs:6307`; `HF2Q_ADMIT_COALESCE_US` up to 100 ms, `:12043-12047`, loop `:12137-12162`; also gates `admit_gemma4_slots_batched`, `:12576`) | arrival within the window | Tiled is not bitwise per-slot; low reach (canonical cue under 32 rows) [INFERENCE] |
| 5 (#729) | Multi-model process | Kernel routing policy is process-global and cached on first use (`engine_qwen35.rs:277-292`). If Qwen3.8 loads first, every family inherits `MVN=0`/`MV_EXT=1`; if it loads second, Qwen3.8 runs padded mode on kernels fixed earlier. The per-turn joint re-check itself changes no numbers (exact joint and serial are bitwise equal) | model-load order | Different kernel routes |
| 6 (#730) | Qwen (prefix resume) | The prefix-resume budget is 5% of memory available at startup, with eviction by bytes (`lcp_registry.rs:130-227`), so resume vs cold prefill depends on host state and other requests (#717 residual risk) | host memory, other requests | Cold vs resume is Class B |

Proposed directions:
1. A per-lane chunk plan and attention kernel that do not depend on co-installation.
2. Fixed chunk sizes independent of width and peer state.
3. Reserve or wait for the anchored slot rather than cold-prefill, or document it as budget-bound behaviour.
4. Make the stable-batch kernel match the per-slot route.
5. Make routing per worker rather than process-global.
6. Document it, or pin resume to deterministic capacity.

## F4 (command buffers per step)
Secondary to this story, coordinated with Qwen36Inv. Qwen36Inv is not changing command-buffer grouping; its F2 (batched MoE FFN behind `HF2Q_QWEN_MOE_BATCHED`) keeps batched rows bitwise equal to the w=1 route.
