# ADR-063 S1: mixed cooperative prefill and decoding nonmembers

Story: ADR-063 S1 (#362), "Prove mixed cooperative prefill cannot corrupt
decoders". Branch `test/adr063-s1-mixed-step-isolation`, which already
contains the DeepSeek-V4 joint-decode branch (`perf/ds4-batched-decode`,
#458). This is a DeepSeek-specific deciding experiment. It does not change
serving policy: the production bypass stays in place until the verdict below
is recorded and a successor design is approved.

Line references are to this branch.

## Question and decision rule

Can one DeepSeek-V4 cooperative prefill cohort (2-4 aligned lanes, at most
`MAX_COOPERATIVE_PREFILL_ROWS = 2048` aggregate rows,
`src/inference/models/deepseek4/verifier_forward.rs:27`, consumed at
`src/serve/api/engine.rs:9883`) run in the same scheduler step as decoding
nonmembers, so that:

- every nonmember's window, compressed and indexer rows, compressor and
  indexer score state, cursor, logits and next tokens equal independent
  serial execution, under cohort success, late failure, poison and
  cancellation; and
- every cohort lane either commits exactly what serial execution commits or
  stays at its start cursor, poisoned?

Isolation holds only if the real-artifact oracle passes bitwise in all four
scenarios and both model-free suites pass. In that case S1 records evidence
for a successor mixed-step policy design (see below). If any nonmember
comparison fails, the bypass stays and the bounded successor design below
applies. Removing the bypass by itself does not count as a deliverable.

## Current bypass

`run_slot_aware_deepseek4` (`src/serve/api/engine.rs:8748`) matches on the
scheduler step:

- `SchedulerStep::Prefill` (pure-prefill step, `engine.rs:9219-9327`). This is
  the only arm that can reach the cooperative path. If no alignment catch-up
  applies, it calls `advance_deepseek4_prefill_quantum` (`engine.rs:9960`) with
  `max_matrix_prefill_windows = None`. That function plans a FIFO cohort
  through `plan_deepseek4_prefill_cohort` (`engine.rs:9857-9943`). The cohort
  must be 2-4 handles, each in `Deepseek4SlotWork::Prefill`, with the same
  cold or warm class, the same reply class, an open client and the same
  `Deepseek4CooperativePrefillPlan`. It then runs the supervised cohort
  transaction (`engine.rs:10015-10214`).
- `SchedulerStep::Mixed` (`engine.rs:9328-9412`). The arm runs
  `decode_batch_deepseek4` for the decode handles, then a serial
  `advance_deepseek4_prefill` for one prefill handle, capped at
  `mixed_budget.max_prefill_windows`. It never calls the cooperative path. In
  addition, `advance_deepseek4_prefill_quantum` returns to the serial path
  whenever a window cap is supplied (`engine.rs:9972-9987`). Together these are
  the bypass. In the ADR's wording, cooperative prefill runs only on
  pure-prefill steps.

Pure-prefill steps can still have parked decoders installed. On a cohort
failure the whole worker fail-stops (see "Failure blast radius").

## Mutation inventory

### A. Per-lane DeepSeek-V4 cache (`Deepseek4Cache`, `src/inference/models/deepseek4/cache.rs:232`)

Each slot owns its own allocation (`Deepseek4Session.cache`,
`src/serve/api/engine_deepseek4.rs:292`). The cohort transaction iterates
sequences and touches only `caches[sequence]` with `spans[sequence]`
(`verifier_forward.rs:657-781`).

| State | Shape / role | Cooperative prefill writes | Decode writes |
| --- | --- | --- | --- |
| `window_kv`: circular 128-row window, a view of the `attention_kv` root | BF16 `[128, head_dim]` | Uncompressed layers: `attention_forward.rs:574` reads the raw prefix and `:592` writes the window. Compressed layers: `compressed_attention.rs:491-497` reads the raw prefix and `:510` writes through `cache_window_write.rs:16`. | Joint decode: `joint_decode_core.rs:125` (window-row core) and `:306` (compressed-row core). Serial `forward_verifier_one` (`verifier_forward.rs:1227/1239`) uses the same encoders through `compressed_attention.rs:91/132`. |
| `compressed_kv`: append-only rows after the window in the same root | BF16 `[ctx/ratio, head_dim]` | `compressed_attention.rs:523` to `compressed_attention_main.rs:61` to `encode_main_tail` (`compressor_tail.rs:28`). Writes rows `[compressed_write_start, +compressed_count)`. | `joint_decode_core.rs:320`; serial decode through the same `encode_main_compressor`. |
| `main_kv_state`, `main_score_state`: recurrent compressor sidecars | F32, overwritten every token | `encode_main_tail` (`compressor_tail.rs:28`, called from `compressed_attention_main.rs:119`) | Same encoder from the decode cores |
| `indexer_kv`, `indexer_kv_state`, `indexer_score_state` (ratio-4 layers only) | BF16 rows plus F32 recurrent state | `compressed_attention.rs:554` to `compressed_attention_indexer.rs:65` to `encode_indexer_tail` (`compressor_tail.rs:136`, called at `compressed_attention_indexer.rs:187`) | `joint_decode_core.rs:342` |
| Logical cursor `next_position` | Visibility of every row above | `plan_prefill` (`cache/prefill.rs:117`), `validate_prefill_commit` (`:150`), `publish_prefill_end` (`:182`), all through `publish_prefill_cohort_after_gate` (`verifier_forward.rs:104`) | `plan_next_step` (`cache.rs:827`), `validate_step_commit` (`:907`), `publish_step_end` (`:921`), through `publish_verifier_cohort_after_gate` (`decode_cohort.rs:15`) |
| `poisoned` | Blocks planning and publication until reset or restore | Settled by `settle_prefill_cohort` (`verifier_forward.rs:155`): a failure after any submission poisons every member, a pre-submission failure poisons none, and a gate rejection or prevalidation failure poisons every member (`:104-144`) | Joint decode poisons every joint lane on a submitted failure (`joint_decode.rs:168-173`) or a gate/prevalidation failure (`decode_cohort.rs:37-52`) |
| Rollback | Compact anchor | `snapshot` (`cache.rs:693`, window plus four state buffers) and `restore` (`cache.rs:766`); `reset` (`cache.rs:939`) | Same |

Compressed and indexer rows written beyond a rolled-back cursor stay
physically present. They become invisible because `next_position` moves back,
and replay overwrites them. The real-artifact Cancel scenario replays over
those stale rows and compares bitwise.

### B. Model state shared by every lane

| State | Where | Cohort | Decode | Isolation argument / hazard |
| --- | --- | --- | --- | --- |
| Thread-local transient pools `PREFILL_POOL`, `PREFILL_SUBMISSION_INPUT_POOL`, `DECODE_POOL`, `TRANSIENT_POOL_PHASE` | `forward_support.rs:33-47` | `begin_prefill_pool_layer`/`end_prefill_pool_layer` around each sequence's attention and each FFN (`verifier_forward.rs:672, 783, 820`) | `begin_decode_pool_token`/`end_decode_pool_token` (`joint_decode.rs:159/162`) | Prefill and decode use separate pools. The hazard is a decoder output buffer (state or logits) that one pool recycles while the next transaction runs. The head allocates logits with `alloc` (`output_forward.rs:104`), which comes from a pool only inside an active cycle. [INFERENCE] The serving head runs outside a cycle. The real-artifact oracle reads decoder logits only after the same step's cohort transaction. |
| Cooperative persistent scratch: state ping/pong, combined attention, mm_id scratch | `verifier_forward.rs:622-655` | Allocated per transaction | Not used | The returned per-lane states are views of the ping/pong root (`verifier_forward.rs:826-847`). Their lifetime follows the `MlxBuffer` reference count. |
| Kernel registry and pipelines (`ctx.registry`) | `model.rs:18` | Read (pipeline cache) | Read | No request data |
| Weights, GLP, KV graft | `model.rs:17-30` | Read; graft rows planted at allocation (`cache.rs:481`) | Read | Read-only at runtime |
| Command ordering | Single worker thread | Each attention or FFN session calls `finish()` (`verifier_forward.rs:771`, `:814`) | Joint decode drains before publication | [INFERENCE] Transactions in one step are serialized on the GPU, so no concurrent cross-lane writes are possible. Only out-of-bounds kernel writes or aliasing could break isolation, and only the hardware oracle can rule those out. |

### C. Serving ledgers per slot (`Deepseek4Session`, `engine_deepseek4.rs:292`)

| Field | Cohort path | Decode path |
| --- | --- | --- |
| `committed_tokens` | `prepare_cooperative_prefill` (`:384`) reserves capacity and requires `cache == ledger` and no live logits. `validate_cooperative_prefill_result` (`:404`) and `publish_cooperative_tokens` (`:424`) are driven from `engine.rs:10137-10164`. | `commit_generated_token` (`:1821`, via `swap_with_loaded` `:578`), `commit_generated_tokens_joint` (`engine_deepseek4/joint_decode.rs:82`), `commit_generated_tokens_cohort` (`:609`) |
| `live_logits` | Must be `None` for every member | Installed by each decode commit (`joint_decode.rs:113`, `:1838`) and read by the sampler on the next tick |
| `pending_turn_anchor`/`_tokens` | `capture_cooperative_turn_anchor` (`:432`) when a chunk ends exactly at the recovery checkpoint (`engine.rs:10149-10161`) | Not touched |
| `turn_anchor`/`_tokens`, `keep_turn_anchor_on_success`, `request_anchor_transaction_active` | Read by cancellation recovery (`recover_after_cancellation`, `:510`) | Promoted by `commit_request_anchor` (`:545`) at a successful terminal |
| `disk_persistor` | Not touched by a cohort chunk | `commit_request_anchor` submits the promoted anchor (`:569`) |

Request state `Deepseek4PrefillState` (`engine_deepseek4/slots.rs`):
`plan_cooperative_prefill` (`:231`), `cooperative_tokens` (`:271`),
`validate_cooperative_prefill_commit` (`:282`), `cooperative_anchor_tokens`
(`:295`), `publish_cooperative_prefill` (`:309`, request cursor). The serial
path goes through `advance` (`:320`).

Disjoint membership is guaranteed at three points:

1. `plan_deepseek4_prefill_cohort` admits only `Deepseek4SlotWork::Prefill`
   slots.
2. `prepare_cooperative_prefill` rejects a lane that holds live logits.
3. Joint decode admits only `joint_decode_ready` lanes
   (`engine_deepseek4/joint_decode.rs:17`), which requires live logits.

### D. Scheduler ledger and publishers

- Scheduler: `prefill_handles_fifo` (`src/serve/scheduler.rs:1506`),
  `advance_after_prefill` (`:1517`, called at `engine.rs:10169` for each
  member), `advance_after_decode` (`:1565`), `release` and
  `record_retained_slot_kv` on a post-commit cancellation
  (`engine.rs:10183-10205`).
- Slot table: `slots[i].take()` moves members out for the duration of the
  transaction (`engine.rs:10027-10070`). On any pre-submission inconsistency,
  `reinstall_deepseek4_cooperative_lanes` (`engine.rs:9945`) puts them back.
  A client that closes before submission sends the whole cohort back to the
  serial path (`engine.rs:10075-10091`).
- Stats and trace publishers: `publish(&scheduler, &scheduler_stats_snapshot)`
  (`engine.rs:9217, 9326, 9411`), `f2_work_state::deepseek` (`engine.rs:9165`),
  `scheduler.trace_transaction` (`engine.rs:10094-10100`,
  `scheduler.rs:1434`), and native observation inputs (`native_observation.rs:271`
  and `:296`, called from the cohort loop).
- Reply publishers: decode tokens go out through
  `settle_deepseek4_decode_lane` (`engine.rs:10820`). A cancelled member calls
  `record_client_cancellation` (`engine.rs:10205`).

### Failure blast radius (current policy)

Any cooperative error becomes a `SlotAwareGpuFatal` over the member slots
(`engine.rs:10114-10132`, `10141-10160`). The arm then calls
`fail_stop_deepseek4_worker` (`engine.rs:11836`), which collects every
installed reply through `take_deepseek4_installed_replies` (`engine.rs:11827`).
As a result, a cohort failure today terminates decoding nonmembers with an
engine-unhealthy error even when their state is clean. A joint-decode failure
behaves the same way (`engine/deepseek4_joint_decode.rs:227, 379`). S1 measures
whether nonmember state is clean. Narrowing the fail-stop is successor scope.

## Evidence added

| Test | Kind | What it proves |
| --- | --- | --- |
| `deepseek4::mixed_step_tests::mixed_step_cohort_outcomes_never_reach_decoding_nonmembers` | Model-free; real `Deepseek4Cache` allocations and production settler/publishers, no weights or kernels | Five lanes: members `[0, 2, 3]` interleaved with decoders `[1, 4]`, both step orders. Five faults: success, pre-submission failure, late gate rejection after submit, poison (execution failure after layer 0 wrote member rows), and cancel (commit, then snapshot restore). Every physical byte, cursor and poison flag of each decoder equals an independent serial control at every step. Members either equal the serial commit or stay at their start cursor, poisoned and unplannable until reset. A cancelled member restores its visible state exactly. |
| `deepseek4::mixed_step_tests::overlapping_membership_poisons_an_innocent_decoder` | Model-free | Isolation depends on disjoint membership. A lane placed in both transactions of one step fails decode prevalidation, and the atomic decode publisher then poisons an innocent decoder. A bystander stays untouched. |
| `engine_deepseek4::mixed_step_tests::decoding_nonmember_cannot_be_drafted_into_a_cooperative_cohort` | Model-free serving ledger | The guards that keep the sets disjoint: a decoder (live logits) is rejected by `prepare_cooperative_prefill` with its ledger unchanged, and a member is never `joint_decode_ready` from planning until logits are installed. |
| `engine_deepseek4::mixed_step_tests::cohort_ledger_commits_exactly_publishes_nothing_or_rolls_back_alone` | Model-free serving ledger | Member ledgers publish exactly, publish nothing (validation fails), or roll back. Rollback goes to the anchor or a reset, and only the cancelled member moves. Decoder ledgers (tokens, cursor, live-logits identity, anchors) do not change. |
| `deepseek4::real_artifact_mixed_step_tests::official_artifact_mixed_step_isolates_cohort_from_decoders` (`#[ignore]`) | Official GGUF, Metal | Cohort `[0, 2, 3]` with a 148-token prefix runs two 640-row chunks, the production three-lane width. Decoders `[1, 4]` with prefixes 125 and 254 cross ratio-4 and ratio-128 boundaries during the cohort. Chunk 0 runs decode then cohort; chunk 1 runs cohort then decode. Decoder logits are read after the cohort. Scenarios: Success; LateFailure (gate rejects after every submission); Poison (injected failure after layer 3, once a ratio-4 and a ratio-128 layer wrote member state); Cancel (commit, roll back one member, replay it serially over the stale rows). Bitwise oracles: decoder logits, greedy tokens and full cache state against serial `forward_verifier_one` at every step; committed members' last-row logits and cache against serial `forward_verifier_prefill`; failed members poisoned at their start cursor. After reset, a poisoned member replays exactly. Cohort survivors then join joint decode for four more steps and stay exact. |

Production support, both behavior-preserving:

- `settle_prefill_cohort` (`verifier_forward.rs:155`) is extracted from
  `forward_verifier_prefill_cohort_with_commit_gate`, so the model-free suite
  drives the exact production poison policy. The error contexts are unchanged.
- `cooperative_fault.rs` is a `cfg(test)` one-shot hook, checked after each
  cohort layer (`verifier_forward.rs:822-823`). Non-test builds compile no hook.

## Root commands

Model-free (Metal device, no model):

```
cargo test --release --lib mixed_step_tests
cargo test --release --lib deepseek4::cache_tests::cooperative
```

The first command runs the four model-free tests in
`deepseek4::mixed_step_tests` and `engine_deepseek4::mixed_step_tests`. The
real-artifact test also matches the filter, but it is skipped because it is
`#[ignore]`. The second command re-runs the existing cooperative publication
tests against the extracted settler.

Hardware (official artifact; about 107 GB resident, so run it alone):

```
HF2Q_DEEPSEEK4_GGUF=/Users/robert/.local/share/hf2q/models/deepseek4/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf \
  cargo test --release --lib \
  official_artifact_mixed_step_isolates_cohort_from_decoders -- --ignored --nocapture --test-threads=1
```

Run it without `MLX_UNRETAINED_REFS`, because the test asserts that joint
decode is available. A pass prints one line per scenario:
`DeepSeek-V4 S1 mixed step: fault=<F> ... exact_nonmember_state_logits_tokens=true`.

## Verdict

**Isolation holds (2026-10-30, M5 Max 128 GB, source `7158e91e`).** S1
closes on isolation evidence; the production bypass is retained. The
successor below is a design only, not implemented by S1.

- Model-free: `mixed_step_tests` 4 passed (1 ignored, the hardware test);
  `deepseek4::cache_tests::cooperative` 3 passed; `deepseek4` 200 passed.
- Hardware, official DS4-0731 q2 artifact (sha256 `50df7e3d…`),
  `official_artifact_mixed_step_isolates_cohort_from_decoders` passed in
  88.17 s:
  - `fault=Success cohort=[0, 2, 3] decoders=[1, 4] chunk_rows=640 steps=8 exact_nonmember_state_logits_tokens=true`
  - `fault=LateFailure … exact_nonmember_state_logits_tokens=true`
  - `fault=Poison … exact_nonmember_state_logits_tokens=true`
  - `fault=Cancel … exact_nonmember_state_logits_tokens=true`
- Memory: the run completed under the live guard with no stop
  (`$E/s1-isolation-v1/memguard.json`).

Isolation is conditional on disjoint membership: the model-free overlap test
shows that a lane in both the cohort and the decode set poisons an innocent
decoder, so the successor must assert disjointness before submission.

### Successor if isolation holds (design only)

In the `Mixed` arm, after `decode_batch_deepseek4`, call
`advance_deepseek4_prefill_quantum` with the cooperative planner instead of
the serial `advance_deepseek4_prefill`. The design must be bounded as follows:

1. Assert disjoint membership before submission: no cohort handle in the
   step's decode handles. The overlap test shows that the publishers cannot
   repair a violation.
2. Keep the row cap at `MAX_COOPERATIVE_PREFILL_ROWS`, with no expansion,
   because of the prior 4096-row OOM evidence. Bound per-step cohort rows by
   the existing mixed fairness budget (`deepseek4_mixed_work_budget`,
   `engine.rs:8089`) so decoder inter-token latency stays at or below today's
   serial mixed slice.
3. Keep fail-stop on cohort failure at first. A later change may narrow it to
   member slots, citing this evidence that nonmember state is clean.
4. Measure the A/B against stock hf2q and against the bypass on the four-agent
   workload: makespan, decoder p95 inter-token latency, TTFT, and bitwise
   output identity.

### Successor if isolation fails (bounded)

Keep the bypass. Classify the failure by its label:

- Decoder logits only: live-logits buffer reuse. Copy live logits out of pool
  scope before a cohort, or allocate them persistently.
- Decoder window, compressed or state bytes: kernel out-of-bounds or aliasing.
  This belongs in mlx-native with a synchronization or bounds test, which
  mlx-native owns.
- Cohort member mismatch on success: cohort exactness itself regressed. This
  is outside S1; re-run the cohort parity test.

Any successor admits a mixed cohort only after the corresponding fix has its
own deciding test.
