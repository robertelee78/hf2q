# ADR-063 Q2: adversarial controls for the four-agent harness

Status: model-free harness controls, implemented and passing. This is not Q2 certification. Independent QE, server-trace checks against actual hardware runs, and QE attestation of the raw cohort (`q2_certify.py`) are still open.

## What is under test

The Q2 acceptance asks that the comparison harness rejects false quality and speed wins and measures actual slot progress. These controls cover the parts of the harness that decide a win:

- **The client.** `qwen_n4_live_smoke.py` (coordinator and `aggregate`) and `qwen38_n1_smoke.py`. Both use `exchange` with the `qwen38_n1_verdict` parser.
- **The supervisor receipt.** The `q5_calibration.py` receipt: verdict and cleanup proof.
- **The campaigns.** The Q3 speed campaign (`q3_campaign.py`, `q3_metrics.py`, `q3_stats.py`) and the task-quality campaign (`quality_campaign.py`).

Each control injects one specific lie and asserts that the harness rejects it with a reason naming that lie. The tests are black-box: they feed adversarial bytes or receipts to the real entry points and check the outcome and the reason. They never recompute the harness's own formulas.

| File | Role |
| --- | --- |
| `scripts/four_agent/q2_fake_service.py` | Test support. A model-free OpenAI-compatible loopback service that answers the canonical seven-turn coding workflow. Its fault table turns single responses adversarial, and `serialized=True` serves one request at a time end to end. Clients are the real `exchange`/parser and the coordinator verdict is the real `aggregate`; tools are stubs. `run_live` drives four real client processes. `honest_run` writes complete client directories for campaign fixtures. |
| `scripts/four_agent/q2_run_audit.py` | Harness detection (new). Re-derives a settled run from its raw client evidence before a Pass counts. |
| `scripts/four_agent/test_q2_adversarial_controls.py` | The control matrix below (36 tests). |

## Detection added to the harness

These gaps were found while building the controls. Each fix is the smallest change that makes the control fail closed.

1. **Q3 and the quality campaign trusted the client summary.** `q3_metrics.classify` and `quality_campaign.classify` scored a run as Pass from `status` and exit code alone. Q3 also took the makespan unchecked. Now both `settle` functions call `q2_run_audit.check_run` on every Pass. If the evidence contradicts the claim, the run becomes `Invalid` with the audit reason. Invalid is terminal: no reruns after the task has started. The audit:
   - re-hashes every archived `request.json` and `response.bin` against its receipt, checks the canonical request wire, and replays each response through the actual parser;
   - compares the replayed usage and SSE event counts with the receipt, and recomputes v3 timing with the actual `semantic_timing_v3`;
   - checks request lineage turn by turn: the first request must match the agent's frozen initial transcript, and each later request must extend the validated transcript with the assistant message and tool result;
   - requires response ids to be unique across every turn of every agent;
   - rejects assistant output that carries another agent's identity sentence;
   - re-runs `aggregate` and requires verified owned cleanup;
   - recomputes the makespan from the release and task-completion clocks;
   - checks phase-cohort membership and triggers;
   - measures co-progress width.
2. **No concurrency proof on the Q3 path.** A deliberately serialized four-client service completes the protocol, so the coordinator reports four-agent success. The audit now measures client-observed co-progress (next section). Q3 requires the expected width; the quality campaign does not, because task quality is not a concurrency claim.
3. **Progress events were not recorded.** Semantic events alone are too sparse. Retained DeepSeek runs have one semantic event per streamed turn, because the tool call arrives in one frame. `exchange` now also records `progress_event_monotonic_ns`: every SSE frame that grew reasoning or semantic output. It uses the same timestamp as the semantic event when both apply, and it is charged to the same client trace quota. The semantic predicate and v3 timing are unchanged.
4. **Silent truncation with a non-length finish.** Spending the whole 8192-token allowance under `finish_reason` `stop` or `tool_calls` was accepted. `qwen38_n1_verdict.output_budget` now rejects it for both unary and SSE, and for both hf2q and peer usage.
5. **Truncated tool-argument JSON.** This used to surface as a raw `JSONDecodeError`. It is now `Invalid("incomplete/truncated tool arguments JSON")`.
6. **Reason precision.** A stream with no semantic progress now fails as `no semantic SSE progress` before the assistant policy runs. Before, a role-only stream was reported as a missing tool call.
7. **Q5 receipt contradiction.** A passing Q5 verdict whose `cleanup_proven` or `cleanup.proven` is not true is rejected.

`q2_certify.replay_response` gained an optional `progress_frames` collector. The canonical request-wire check moved into `q2_certify.request_wire`, which `audit_agent` and the run audit share.

## Co-progress, slot and phase counts

Overlapping HTTP lifetimes prove nothing: a serialized service keeps four requests open while it serves one.

- **Co-progress.** Agent B co-progresses with agent A when, inside the span of one of A's streamed requests (first to last progress frame), the merged A/B frame order switches agents at least `MIN_INTERLEAVE = 4` times. The observed width is 1 plus the largest number of agents co-progressing with any one streamed request. If no streamed request progressed twice, the width is 0.
- **Expected width.** `n4` must show 4. A `K+M` plan must show at least its largest cohort: 3 for `3+1` and `1+3`, 2 for `2+2`. The one-slot control `1` must show exactly agent-0, with width 1.
- **Cohort membership.** For phase plans, the audit checks four things:
  - The recorded plan and cohorts match the expected plan.
  - Every first-decode trigger lies inside that agent's first streamed request.
  - Each later cohort was released at or after its triggers.
  - Every agent of the earlier cohort was still running at that release, so the population really mixed.

  Each agent must also observe its own cohort's release, and its first request must follow that release.
- **Boundary.** Client co-progress is a lower bound on concurrent slot progress. A server that time-slices four requests token by token also shows width 4. Physical batch membership (which owners share one forward), active and runnable prefill/decode counts, and pure four-prefill and four-decode phases come from the server phase trace. `f2_trace.validate_trace` and `qualify_window` decide those, with `forward_demand_v2`/queued-demand rules. They are controlled by `test_q2_f2_controls.py` (all populations, serialized four HTTP labels, Unknown/duplicate/stale membership, missing drains) and `test_f2_demand.py`.

Live results: four real client processes against the fake service, with known delays of 10 ms before the first delta, 10 ms between frames and 10 ms for unary responses.

| Plan | Service | Observed width | Expected | Max open HTTP | Makespan (s) | Harness |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| n4 | concurrent | 4 | 4 | 4 | 0.316 | accepted |
| n4 | serialized | 1 | 4 | 4 | 1.208 | `serialized service: observed co-progress width 1 < 4 with up to 4 overlapping HTTP requests` |
| 1 | concurrent | 1 | 1 | 1 | 0.336 | accepted |
| 3+1 | concurrent | 4 | 3 | 4 | 0.358 | accepted |
| 3+1 | serialized | 1 | 3 | 4 | 1.222 | rejected: width 1 < 3 |
| 2+2 | concurrent | 4 | 2 | 4 | 0.356 | accepted |
| 2+2 | serialized | 1 | 2 | 4 | 1.240 | rejected: width 1 < 2 |
| 1+3 | concurrent | 4 | 3 | 4 | 0.350 | accepted |
| 1+3 | serialized | 1 | 3 | 4 | 1.270 | rejected: width 1 < 3 |

In every row the coordinator alone reported `FourAgentProtocolCompleteAwaitingIndependentReview`. A serialized arm is slower, so it hands a false win to whichever arm it is compared with. `CampaignControls` shows the Q3 driver halting `Invalid` on a serialized stock arm, and `compute_verdict` refusing to return `Win`.

## Control matrix

Layer **wire** means the live client rejects the response during `exchange`; the raw bytes are kept. Layer **audit** means `q2_run_audit` rejects a settled run. Layer **campaign** means the actual Q3 or quality driver turns the run `Invalid` and halts. All reasons are asserted verbatim or by regex.

| Acceptance item | Injected lie | Layer | Rejection reason |
| --- | --- | --- | --- |
| Deliberately serialized four-client service | one request at a time, four open HTTPs (live, n4 and 3+1) | audit, campaign | `serialized service: observed co-progress width 1 < 4` (or `< 3`) |
| Role-only SSE | role frames only, then finish/usage/DONE | wire | `no semantic SSE progress` |
| Missing semantic output | reasoning-only stream; empty final summary | wire | `no semantic SSE progress`; `final summary missing or extra tool continuation` |
| Wrong tool arguments | `tests/ports.rs` at turn 0; target `everything` | wire | `tool arguments/workflow drift` |
| Stale lineage | tool-call id reused within an agent | wire | `missing/duplicate/oversized tool id` |
| Stale lineage | response id reused (same or another turn) | audit | `stale lineage: response id of agent-0 turn 4 repeats agent-0 turn 2` |
| Stale lineage | another agent's request substituted; initial transcript changed | audit | `stale lineage: request history does not extend the validated transcript`; `stale lineage: first request …` |
| Cross-slot leakage | another agent's response replayed; another agent's identity in output | audit | `… agent-1 turn 6 repeats agent-0 turn 6`; `cross-slot leakage: agent-2 output carries another agent's identity` |
| Silent truncation | `completion_tokens=8192` with `tool_calls`/`stop` (hf2q unary, SSE, final; peer SSE); half tool-argument JSON; archived raw rewritten | wire, audit | `output allowance exhausted without finish_reason length (silent truncation)`; `incomplete/truncated tool arguments JSON` |
| Length finish | `finish_reason: length` on a tool turn | wire | `exactly one live call required` |
| Altered output budgets | client request allowance 4096 or 8193 (the server receives nothing); archived request allowance 4096; 9000 completion tokens | wire, audit | `output allowance drift`; `excess completion tokens` |
| Cache accounting | `cached_tokens > prompt_tokens` (hf2q and peer); cold turn claims reuse; warm turn reports none; receipt usage edited | wire, audit | `excess cached tokens` / `excess peer cached tokens`; `prefix reuse not proven`; `receipt usage differs from the raw response (usage/cache accounting)` |
| Usage arithmetic | `total_tokens != prompt + completion` | wire | `usage total contradiction` |
| Model identity | response model swapped (unary, SSE); archived request/response model changed mid-run | wire, audit | `response identity`; `SSE identity`; `model identity changed between turns` |
| Binary/engine identity | peer client summary scored as an hf2q arm | audit | `client summary engine differs from the arm` |
| Binary/model identity (existing) | receipt binary or artifact contradicts the frozen identity | campaign | `IdentityDrift` (`binary.sha256`, `artifact.*`), covered by `test_q3_campaign` |
| Corrupted receipts | one byte of a raw response changed | audit | `archived response hash differs from its receipt` |
| Metric units | makespan in milliseconds labelled seconds (summary and campaign); TTFT ×1000 | audit, campaign | `makespan seconds disagree with its clocks (unit/arithmetic corruption)`; `receipt timing differs from its clocks (timer direction/unit corruption)` |
| Timer direction | send and terminal swapped | audit | `monotonic send/validated terminal required` |
| Injected speedups in receipts | task completion before its last response; makespan end trimmed; progress event before send | audit | `task completion before its last validated response`; `makespan end is not the last task completion`; `progress events outside their request` |
| Result scoring | final test `passed=false` while the status claims success (summary, Q3 campaign, quality campaign); one agent `Failed` | audit, campaign | `final tests/source`; `agent incomplete` |
| Cleanup | live owned PID after client cleanup; tool cleanup false; Q5 Pass without proven cleanup | audit | `actual owned cleanup unknown/live`; `cleanup unknown`; `Q5 receipt contradiction: passing verdict without proven cleanup` |
| One-slot vs four-wide | four-agent run presented as `1`, and the reverse | audit | `agent set differs from the phase plan` |
| Mixed population | `3+1` run presented as `n4`; release after the first cohort finished; trigger outside the first stream; release before the trigger | audit | `aggregate phase cohort`; `population never mixed`; `first-decode trigger outside the agent's first streamed request`; `cohort released before its trigger` |

Positive controls: the honest service completes all seven turns on the hf2q and peer engines. Honest runs of every population pass the audit with the expected width (n4 4, `1` 1, `3+1` 3, `2+2` 2, `1+3` 3, peer n4 4). The rewritten `test_q3_campaign` and `test_quality_campaign` drivers run on audited fake-service runs.

### Timer direction and units with known delays

A single agent drives the real exchange loop against known server delays.

| Delay (s): first / gap / unary | Turn timing (s), turns 0–6 (streamed turns are TTFT, unary turns are completion) |
| --- | --- |
| 0.15 / 0.01 / 0.06 | 0.071, 0.221, 0.071, 0.217, 0.071, 0.221, 0.071 |
| 0.01 / 0.01 / 0.00 | 0.0006, 0.0755, 0.0007, 0.0755, 0.0007, 0.0726, 0.0006 |

The test asserts four properties:

- **Lower bound.** Each streamed TTFT is at least the known delay before the first semantic frame: 0.15 + 4 × 0.01 = 0.19 s. A server cannot deliver earlier than it sends.
- **Direction.** The slower service is slower on every turn.
- **Units.** `semantic_ttft_seconds` equals `(semantic_event[0] - send) / 1e9` exactly, and send ≤ first progress ≤ first semantic ≤ terminal.
- **Through to Q3.** `q3_metrics.slot_metrics` reports the same seconds.

A four-agent live run with every delay scaled by 0.25 has a shorter makespan than the unscaled run. The unscaled makespan is at least its known critical-path delay: 4 × 0.04 + 3 × (0.04 + 5 × 0.02) s. `q3_stats.paired_effect` gives the faster arm a negative log ratio, which means a ratio below 1, the "experiment faster" direction.

## Finding: hf2q tool-call ids collide across concurrent agents

A scan of 547 retained four-agent client directories under the ADR-063 evidence root found 18 runs in which two agents received the same tool-call id, all hf2q, mostly DeepSeek-V4. Response ids (UUIDs) never collided. The cause is in `src/serve/api/engine.rs`: tool-call ids are wall-clock nanoseconds XOR the call index. Two slots that open a call within one clock tick get the same id. Within one conversation the id stays unique, so the client protocol does not fail, but OpenAI-compatible clients may assume ids are globally unique.

For this reason the audit does not treat a tool-call id shared across agents as leakage. Leakage is detected by replayed response ids and by foreign identity text. A process-unique id source would fix the server. That is a production change and is not part of this slice.

## Not covered here

- **A fully re-timed forgery.** If every clock in every receipt is rewritten consistently, client evidence alone cannot detect it. Q3 binds the Q5 receipt and frozen identities, and F2 certification requires the server phase trace and hash-bound raw cohorts (`q2_certify.py`).
- **The real-hardware width threshold.** The width rule has not yet run on real hardware. Retained runs predate `progress_event_monotonic_ns`, so they cannot be re-audited for width. The first real Q3 warmups will exercise it. A genuine arm that falls below its expected width is measured evidence to review, not a reason to retune `MIN_INTERLEAVE` after the fact.
- **Independent QE.** Critical-path and failure-path review by Agentic-QE or an equivalent, and QE attestation of actual server traces, remain open. These tests are the deterministic fake-server and receipt negative controls named in the Q2 validation line.
- **Native trace budget profiles.** These were already implemented before this slice and are only listed here:
  - legacy 256 MiB / 1,000,000 events;
  - `native_command_capture_v2`: 2 GiB / 1,000,000 events;
  - `native_command_capture_v3`: 8 GiB / 16,000,000 events, explicit diagnostic native observer only;
  - client minimum 128 MiB, live native metadata 16 MiB, one-past refusals.

  They live in `src/serve/phase_trace/metadata_budget.rs` and `scripts/four_agent/f2_trace.py`, with controls in `test_f2_metadata_budget.py` (commits `231505ab`, `b3905407`, `22ed88b3`).

## Results and commands

Pure-Python suite (Python 3.13, macOS): 541 tests, OK, about 45 s. That is 505 existing tests plus 36 in `test_q2_adversarial_controls.py` (about 9 s). The live concurrency and injected-delay classes passed on six consecutive repeats.

Dry audit of retained real runs (no false rejection). Ten retained four-agent client directories were audited without the width gate, because they predate `progress_event_monotonic_ns`. Their progress timestamps were synthesized at the replayed frame count, so width was not evaluated. The runs covered hf2q and peer arms of Qwen3.8, Gemma 4, DeepSeek-V4 and a quality-campaign instance: `c5-runs-v1` ×3, `peer-arms-v1` ×2, `ds4-joint-ab-v1/b2`, `q3-campaign-v1` ×2, `gemma-joint-ab-v1/g-3-P` and `quality-qwen38-peer-exp0b95/01`. Every other check passed on all ten: hashes, canonical wire, parser replay, lineage, response-id uniqueness, identity leakage, aggregate, cleanup, makespan arithmetic.

```sh
python3 -m unittest discover -s scripts/four_agent -p 'test_*.py'
(cd scripts/four_agent && python3 -m unittest test_q2_adversarial_controls -v)
```
