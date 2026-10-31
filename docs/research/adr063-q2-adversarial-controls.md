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
| `scripts/four_agent/q2_fake_service.py` | Test support. A model-free OpenAI-compatible loopback service that answers the canonical seven-turn coding workflow. Its fault table turns single responses adversarial. `slots=N` serves at most N requests at a time end to end (`slots=1` is a serialized service), `rendezvous` holds the given streams mid-stream together until their clients acknowledge their first progress frame, and `bursty=True` streams the four-frame hf2q Gemma shape (see "Deterministic under load" below). Clients are the real `exchange`/parser and the coordinator verdict is the real `aggregate`; tools are stubs. `run_live` drives four real client processes. `honest_run` writes complete client directories for campaign fixtures, lockstep or slot-capped. |
| `scripts/four_agent/q2_run_audit.py` | Harness detection (new). Re-derives a settled run from its raw client evidence before a Pass counts. |
| `scripts/four_agent/test_q2_adversarial_controls.py` | The control matrix below (38 tests). |

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
2. **No concurrency proof on the Q3 path.** A deliberately serialized four-client service completes the protocol, so the coordinator reports four-agent success. The audit now measures client-observed co-progress (next section). Q3 requires every active agent to be co-active at some point; the quality campaign does not, because task quality is not a concurrency claim.
3. **Progress events were not recorded.** Semantic events alone are too sparse. Retained DeepSeek runs have one semantic event per streamed turn, because the tool call arrives in one frame. `exchange` now also records `progress_event_monotonic_ns`: every SSE frame that grew reasoning or semantic output. It uses the same timestamp as the semantic event when both apply, and it is charged to the same client trace quota. The semantic predicate and v3 timing are unchanged.
4. **Silent truncation with a non-length finish.** Spending the whole 8192-token allowance under `finish_reason` `stop` or `tool_calls` was accepted. `qwen38_n1_verdict.output_budget` now rejects it for both unary and SSE, and for both hf2q and peer usage.
5. **Truncated tool-argument JSON.** This used to surface as a raw `JSONDecodeError`. It is now `Invalid("incomplete/truncated tool arguments JSON")`.
6. **Reason precision.** A stream with no semantic progress now fails as `no semantic SSE progress` before the assistant policy runs. Before, a role-only stream was reported as a missing tool call.
7. **Q5 receipt contradiction.** A passing Q5 verdict whose `cleanup_proven` or `cleanup.proven` is not true is rejected.

`q2_certify.replay_response` gained an optional `progress_frames` collector. The canonical request-wire check moved into `q2_certify.request_wire`, which `audit_agent` and the run audit share.

## Co-progress, slot and phase counts

Overlapping HTTP lifetimes prove nothing: a serialized service keeps four requests open while it serves one.

- **Co-progress (v2, after independent QE review).**
  - A streamed request is *active* from its first to its last progress frame (`progress_event_monotonic_ns`).
  - A set of agents is *co-active* when one active request per agent shares a common window lasting at least `MIN_OVERLAP_FRACTION = 0.5` of the shortest member's active interval.
  - The observed width is the largest co-active set. It is 1 when there is progress but no qualifying overlap, and 0 with no progress at all.
  - The rule needs no frame alternation, so low-frame bursty streams qualify. hf2q Gemma streams one reasoning frame, then a burst of three semantic frames, in a fixed per-agent order.

  The first version required four frame alternations inside one request (`MIN_INTERLEAVE`). On retained hf2q Gemma runs that left zero margin: at most 2 switches from real semantic frames, and exactly 4 only with the reasoning frame placed from `first_reasoning_seconds`. A genuine Gemma Q3 run could therefore have been marked `Invalid`.
- **Why a capped service cannot reach the width.** A service that runs at most N requests at a time cannot keep N+1 requests mid-stream together. Its width stays at N however many HTTP requests it holds open. Only client receive jitter at request boundaries can overlap, and that is a vanishing fraction of an active interval: on real hardware the next request's first frame also waits for its prefill.
- **Expected width.** Every active agent must be co-active at some point: 4 for `n4`, 4 for `3+1`, `2+2` and `1+3`, and 1 for the one-slot control `1`, which must contain only agent-0.
  - For `K+M` plans this is an after-release condition. The later cohort exists only after its release, and the audit separately proves that the earlier cohort was still running at that release.
  - A service capped at the largest cohort (2 for `2+2`, 3 for `3+1`/`1+3`) used to pass the mixed plans. It now fails them.
- **Cohort membership.** For phase plans, the audit checks four things:
  - The recorded plan and cohorts match the expected plan.
  - Every first-decode trigger lies inside that agent's first streamed request.
  - Each later cohort was released at or after its triggers.
  - Every agent of the earlier cohort was still running at that release, so the population really mixed.

  Each agent must also observe its own cohort's release, and its first request must follow that release.
- **Boundary.** Client co-progress is a lower bound on concurrent slot progress.
  - A server that time-slices four requests token by token also shows width 4.
  - So does a server that parks a request mid-generation while it finishes another. To the client that looks the same as a model whose tokens are buffered between frames.
  - Physical batch membership (which owners share one forward), active and runnable prefill/decode counts, and pure four-prefill and four-decode phases come from the server phase trace. `f2_trace.validate_trace` and `qualify_window` decide those, with `forward_demand_v2`/queued-demand rules. They are controlled by `test_q2_f2_controls.py` (all populations, serialized four HTTP labels, Unknown/duplicate/stale membership, missing drains) and `test_f2_demand.py`.

Live results: four real client processes against the fake service. This table was measured on an unloaded host, before the rendezvous and close-wait construction described under "Deterministic under load". The live controls now use that construction, so their width outcomes no longer depend on timing; overlap and makespan values differ under it.

- **Default timing:** 2 ms before the first delta, 15 ms between frames, 2 ms unary latency.
- **Bursty rows** use the Gemma-shaped stream: one reasoning frame after 20 ms, a 60 ms pause, then three semantic frames back to back, for four progress frames per turn.
- **Overlap** is the witness set's common window over the shortest member interval.

| Plan | Service | Observed width | Expected | Overlap | Max open HTTP | Makespan (s) | Harness |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| n4 | concurrent | 4 | 4 | 0.996 | 4 | 0.347 | accepted |
| n4 | serialized | 1 | 4 | – | 4 | 1.296 | `serialized or slot-capped service: observed co-progress width 1 < 4 with up to 4 overlapping HTTP requests` |
| n4 | 2 slots | 2 | 4 | 1.0 | 4 | 0.658 | rejected: width 2 < 4 |
| n4 | 3 slots | 3 | 4 | 0.981 | 4 | 0.441 | rejected: width 3 < 4 |
| n4 | bursty, concurrent | 4 | 4 | 0.998 | 4 | 0.306 | accepted |
| n4 | bursty, serialized | 1 | 4 | – | 4 | 1.127 | rejected: width 1 < 4 |
| 1 | concurrent | 1 | 1 | – | 1 | 0.347 | accepted |
| 3+1 | concurrent | 4 | 4 | 0.845 | 4 | 0.336 | accepted |
| 3+1 | serialized | 1 | 4 | – | 4 | 1.284 | rejected: width 1 < 4 |
| 3+1 | 3 slots | 3 | 4 | 1.0 | 4 | 0.430 | rejected: width 3 < 4 |
| 2+2 | concurrent | 4 | 4 | 0.876 | 4 | 0.339 | accepted |
| 2+2 | serialized | 1 | 4 | – | 4 | 1.312 | rejected: width 1 < 4 |
| 2+2 | 2 slots | 2 | 4 | 1.0 | 4 | 0.642 | rejected: width 2 < 4 |
| 1+3 | concurrent | 4 | 4 | 1.0 | 4 | 0.355 | accepted |
| 1+3 | serialized | 1 | 4 | – | 4 | 1.317 | rejected: width 1 < 4 |
| 1+3 | 3 slots | 3 | 4 | 0.798 | 4 | 0.473 | rejected: width 3 < 4 |

In every row the coordinator alone reported `FourAgentProtocolCompleteAwaitingIndependentReview`. A serialized or capped arm is slower, so it hands a false win to whichever arm it is compared with. `CampaignControls` shows the Q3 driver halting `Invalid` on a serialized stock arm, and `compute_verdict` refusing to return `Win`.

### Retained hf2q Gemma and peer Gemma runs (margin)

All 16 completed retained runs in `gemma-stage2-ab-v1/g-*-T-*` (4) and `gemma-alloc-ab-v1/g-*` (12) were replayed through the full audit with the width gate on: 12 hf2q runs and 4 peer runs. Every one passed with width 4.

- **Method.** These runs predate `progress_event_monotonic_ns`, so the real semantic timestamps were kept. Each non-semantic progress frame (hf2q: the single reasoning frame; peer: none) was placed at the recorded `first_reasoning_seconds`.
- **hf2q Gemma.** Streams carry 3 semantic and 4 progress frames. The witness overlap was 0.9987–0.9989.
- **Semantic timestamps alone.** Without any placed frame, which is a strict lower bound, width is still 4 with overlap 0.971–0.975. Recorded frames can only add to this.
- **Peer Gemma.** Streams carry 7 or 11 semantic frames. Overlap was 0.990–0.993.

Against the 0.5 threshold that is a margin of about 2×. The alternation rule, by contrast, left zero margin on the same evidence.

## Control matrix

Layer **wire** means the live client rejects the response during `exchange`; the raw bytes are kept. Layer **audit** means `q2_run_audit` rejects a settled run. Layer **campaign** means the actual Q3 or quality driver turns the run `Invalid` and halts. All reasons are asserted verbatim or by regex.

| Acceptance item | Injected lie | Layer | Rejection reason |
| --- | --- | --- | --- |
| Deliberately serialized four-client service | one request at a time, four open HTTPs (live: n4, `3+1`, and Gemma-shaped bursty streams; retimed: n4) | audit, campaign | `serialized or slot-capped service: observed co-progress width 1 < 4` |
| Slot-capped service in a mixed population | at most 2 requests served for `2+2`, at most 3 for `3+1`/`1+3` (live); at most 2 or 3 for n4 (retimed) | audit | `… observed co-progress width 2 < 4` / `3 < 4` |
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

Positive controls: the honest service completes all seven turns on the hf2q and peer engines. Honest runs of every population pass the audit with the expected width: n4 4, `1` 1, `3+1`/`2+2`/`1+3` 4 (retimed and live), peer n4 4, and the bursty four-frame stream 4. The rewritten `test_q3_campaign` and `test_quality_campaign` drivers run on audited fake-service runs.

### Timer direction and units with known delays

A single agent drives the real exchange loop against known server delays: a slow service (0.5 s before the first delta, 0.01 s between frames, 0.2 s unary latency) and a fast one (0.01 / 0.01 / 0).

The test asserts five properties:

- **Lower bound.** Each streamed TTFT is at least the known delay before the first semantic frame: 0.5 + 4 × 0.01 = 0.54 s. A server cannot deliver earlier than it sends. Each unary completion is at least 0.2 s.
- **Upper bound.** The TTFT stays below 0.54 + 10 s. A millisecond value read as seconds would be 1000 times too large.
- **Direction.** The slower service is slower on every turn. The 0.5 s pre-semantic delay dwarfs host scheduling noise.
- **Units.** The nanosecond clocks hold at least the known delay: `semantic_event[0] - send` ≥ 0.54 × 10⁹. With the seconds bounds, this rules out a seconds/milliseconds mix-up without restating the TTFT formula. Ordering also holds: send ≤ first progress ≤ first semantic ≤ terminal.
- **Through to Q3.** `q3_metrics.slot_metrics` reports the same seconds.

A four-agent live run with every delay scaled by 0.1 has a shorter makespan than the unscaled run (0.1 s first, 0.05 s gaps, 0.1 s unary). Each makespan is at least its known critical-path delay: 4 × 0.1 + 3 × (0.1 + 5 × 0.05) = 1.45 s unscaled, 0.145 s scaled. `q3_stats.paired_effect` gives the faster arm a negative log ratio, which means a ratio below 1, the "experiment faster" direction.

### Deterministic under load

A loaded CI runner (suite 149 s against 48 s locally) once returned width 3 for a concurrent bursty run. One concurrent stream had started late, so its overlap with the others fell below 0.5 of the shortest interval. The live controls now get their concurrency, and their lack of it, by construction. The audit is unchanged.

- **Concurrent services: client-acknowledged rendezvous.** `rendezvous_groups(plan)` names the streams that a genuinely concurrent service holds open together:
  - n4: every agent's turn 1;
  - `K+M`: the earlier cohort's turn 3 with the later cohort's turn 1;
  - `1`: none.

  Each member stream stamps the clock and sends its first progress frame. The live client acknowledges that frame through its `on_first_token` hook, after `exchange` has stamped it, by writing a marker into the service's `acks` directory. The member stream then waits until every member has sent its first frame and every member's client has acknowledged it. Observing that at clock t, it holds until t + max(0.1 s, t − earliest first send), then sends the rest of its stream.

  The overlap fraction is then at least 0.5 however late any thread or process runs. Let F be the group's latest first-progress stamp, and j the member whose last progress stamp L is earliest:
  - every acknowledgement precedes t, so F ≤ t_j;
  - j's last frame is sent after its hold, so the common window L − F is at least that hold;
  - no client stamps a frame before it is sent, so F − f_j ≤ t_j − earliest first send, which is at most the same hold.

  j's active interval, (L − F) + (F − f_j), is therefore at most twice the common window, and so is the shortest member's. Server and clients stamp the same host's `CLOCK_MONOTONIC`. In `K+M` plans the rendezvous also mixes the population by construction: the earlier cohort cannot finish before the later cohort has started streaming.
- **Slot-capped services: close-wait.** A slot-capped or serialized service half-closes each response, then keeps the slot until the client closes its connection. The client closes only after it has stamped every frame. Every client-observed active interval therefore lies inside its slot-holding interval, so the width cannot exceed the cap however loaded the host is.
- **Admission gate.** For the "despite overlapping HTTP" claim, `admit_together=4` holds the first four requests until all four are open. The live serialized controls therefore always have 4 overlapping HTTP requests.
- **Loud failures.** A rendezvous, admission or close wait that fails is recorded, and the service's `__exit__` fails the test with the reason, chained to any error already raised. The waits time out after 20 s, below the client's 30 s request deadline, so a stall is reported as the named wait. A failed wait never turns into a timing-dependent width.

Three assertions still compare against wall-clock time. Each has a known server delay as its margin:

- **Speedup direction.** The slow makespan is at least 1.45 s of known delay. The fast run has 0.145 s of known delay plus client overhead.
- **Timer direction.** Each slow streamed TTFT is at least 0.54 s, and the fast one is 0.05 s plus overhead. The upper bound sits 10 s above the known delay.
- **Slot-capped mixing.** The audit checks that a mixed population really mixed before it checks width, and a slot-capped service has no rendezvous. At default timing, an earlier-cohort agent still has 0.235 s of known server delay ahead of it at its own first decode. The later cohort is released once the whole earlier cohort has decoded, and `run_live` polls for that every 2 ms.

## Finding: hf2q tool-call ids collide across concurrent agents

A scan of 547 retained four-agent client directories under the ADR-063 evidence root found 18 runs in which two agents received the same tool-call id, all hf2q, mostly DeepSeek-V4. Response ids (UUIDs) never collided. The cause is in `src/serve/api/engine.rs`: tool-call ids are wall-clock nanoseconds XOR the call index. Two slots that open a call within one clock tick get the same id. Within one conversation the id stays unique, so the client protocol does not fail, but OpenAI-compatible clients may assume ids are globally unique.

For this reason the audit does not treat a tool-call id shared across agents as leakage. Leakage is detected by replayed response ids and by foreign identity text. A process-unique id source would fix the server. That is a production change and is not part of this slice.

## Not covered here

- **A fully re-timed forgery.** If every clock in every receipt is rewritten consistently, client evidence alone cannot detect it. Q3 binds the Q5 receipt and frozen identities, and F2 certification requires the server phase trace and hash-bound raw cohorts (`q2_certify.py`).
- **The real-hardware width threshold.** The width rule has not yet run on real hardware with recorded progress events.
  - Retained Gemma runs pass with about 2× margin (see above) when their reasoning frame is placed from `first_reasoning_seconds`. Retained Qwen, DeepSeek and peer runs have more frames per stream, but their reasoning timestamps were not recorded.
  - Mixed plans now need a later-cohort streamed request that is co-active with every earlier-cohort agent at once. On hardware that depends on how the cohorts' streamed turns line up after the release.
  - The first real Q3 warmups of each family and workload must be reviewed. A genuine arm that falls below its expected width is measured evidence to review, not a reason to retune `MIN_OVERLAP_FRACTION` after the fact.
- **Independent QE.** Critical-path and failure-path review by Agentic-QE or an equivalent, and QE attestation of actual server traces, remain open. These tests are the deterministic fake-server and receipt negative controls named in the Q2 validation line.
- **Native trace budget profiles.** These were already implemented before this slice and are only listed here:
  - legacy 256 MiB / 1,000,000 events;
  - `native_command_capture_v2`: 2 GiB / 1,000,000 events;
  - `native_command_capture_v3`: 8 GiB / 16,000,000 events, explicit diagnostic native observer only;
  - client minimum 128 MiB, live native metadata 16 MiB, one-past refusals.

  They live in `src/serve/phase_trace/metadata_budget.rs` and `scripts/four_agent/f2_trace.py`, with controls in `test_f2_metadata_budget.py` (commits `231505ab`, `b3905407`, `22ed88b3`).

## Results and commands

Pure-Python suite (Python 3.13, macOS), measured before the deterministic construction: 543 tests, OK, about 48 s. That is 505 existing tests plus 38 in `test_q2_adversarial_controls.py` (about 15 s). The construction does not change the test count.

Dry audit of retained real runs (no false rejection). Ten retained four-agent client directories were audited without the width gate, because they predate `progress_event_monotonic_ns`. Their progress timestamps were synthesized at the replayed frame count, so width was not evaluated. The runs covered hf2q and peer arms of Qwen3.8, Gemma 4, DeepSeek-V4 and a quality-campaign instance: `c5-runs-v1` ×3, `peer-arms-v1` ×2, `ds4-joint-ab-v1/b2`, `q3-campaign-v1` ×2, `gemma-joint-ab-v1/g-3-P` and `quality-qwen38-peer-exp0b95/01`. Every other check passed on all ten: hashes, canonical wire, parser replay, lineage, response-id uniqueness, identity leakage, aggregate, cleanup, makespan arithmetic.

```sh
python3 -m unittest discover -s scripts/four_agent -p 'test_*.py'
(cd scripts/four_agent && python3 -m unittest test_q2_adversarial_controls -v)
```
