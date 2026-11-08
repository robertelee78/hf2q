# ADR-063 Q4-TRACE-GPU-TIMING: per-command GPU timestamps in the phase trace

- **Status:** Design revision 3. Review677 accepted it conditional on its rev-2 finding 1, which §2.6 applies; Main approves on the reviewer's ACCEPT. Revisions 2 and 3 resolve Review677's reviews of revisions 1 (`647104db`) and 2 (`5ae19c2a`); see §6. No code yet.
- **Story:** ADR-063 §Q4-TRACE-GPU-TIMING, a follow-up to Q4-TRACE (#663, merged in `dafb63ab`).
- **Motivation:** the Q4-TRACE T1 attribution (2026-11-06, `~/.local/state/hf2q/v0130/q4-trace/attribution-handoff.md`) splits a Qwen3.8 joint step into host encode (median about 7 ms) and a completion tail (about 79 ms). It cannot say which layer chains spend that time.
  - Joint decode commits its layer chains back to back and checks them together at the end of the forward. The trace's `CheckedDrain` instants are therefore the final waits, not per-command completions.
  - The gap program needs GPU time per chain.
- **Code basis:** main `dafb63ab`, which pins `mlx-native = "=0.16.0"` (`Cargo.toml:177`). All mlx-native references below are to 0.16.0.

## 1. What mlx-native 0.16.0 already provides

- `install_submission_observer_with_gpu_timestamps(sink, event_limit, metadata_limit_bytes)` (`submission_observer.rs:180`) turns on a separate v1 stream.
  - Its events count against the same `event_limit` through `core.events`.
  - It adds no command owners and no waits.
- Each observed command emits exactly one terminal `CompletedGpuTimingEvent {schema_version: 1, command_id, invocation_epoch, queue_identity, device_registry_id, timestamps}` through `SubmissionObserver::record_gpu_timing` (`submission_observer.rs:726-737`). It is emitted in the same call as the drain, immediately after `drained(...)`:
  - for waited command buffers: `encoder.rs:2893-2894`;
  - for commands retired by pending inspection: `submission_observer/pending.rs:116-117`.
- The resulting per-command causal order is fixed. It is `CheckedDrain`, then timing, then `Dropped`. A command dropped without a checked drain gets `Unknown {DroppedWithoutCheckedTiming}` before its `Dropped` (`submission_observer.rs:856-859`).
- `timestamps` takes one of two forms:
  - `Available {clock: MetalCommandBufferHostSecondsSystemMachTime, start_seconds, end_seconds}`: Metal's `GPUStartTime`/`GPUEndTime`, read after a healthy (status 4) completion.
  - `Unknown {reason, checked_status, raw_bits}`, with `reason` one of `NotHealthyCompleted`, `EmptyCommand`, `NonFinite`, `NonPositive`, `NotStrictlyOrdered`, `DroppedWithoutCheckedTiming` or `TaintedObservation` (`gpu_timestamps.rs`).
- These are command-buffer execution envelopes. They are not shader busy time. mlx-native maps them to no CPU clock.

No mlx-native change is needed. hf2q currently installs without the stream (`src/serve/phase_trace/native.rs:147`), and its `Sink` implements only `record`. The default `record_gpu_timing` returns `Err`, which would invalidate the capture.

## 2. Server change (hf2q)

### 2.1 Opt-in

- `native_observer.gpu_timestamps: bool` uses `#[serde(default, skip_serializing_if = "is_false")]`.
- An explicit `false` is refused at config validation. The key is either absent (off) or `true`.
- `true` is accepted only with `diagnostic_case: true`.
- The trace header's `native_observer` echo is the serialized config. It therefore carries the key only when it is on, and off-path configs and headers stay byte-identical, `config_sha256` included.
- Default serving never installs an observer.

### 2.2 One ordered queue

```rust
enum Item { Event(SubmissionEvent), Timing(CompletedGpuTimingEvent) }
```

- The existing `SyncSender<SubmissionEvent>` becomes `SyncSender<Item>`, with the same bound of `QUEUED_EVENTS = 2048`.
  - `record` sends `Item::Event`.
  - `record_gpu_timing` sends `Item::Timing`, only when the flag is on. Otherwise it keeps returning `Err`.
  - Both use `try_send`. A full queue or STOP returns `Err`, and mlx-native then fails the capture closed.
- The writer drains one queue in arrival order, so per-command causal order is preserved in the JSONL.
  - It batches up to `WRITE_BATCH` items. A batch is published as consecutive runs of one kind each: `native_event` runs through the existing `publish_all`, `native_gpu_timing` runs likewise.
  - Run boundaries keep the order.
- **Writer exit condition (unchanged, now stated):** the writer stops only when the queue is empty and STOP is set, or on disconnect. `final_status` sets STOP and joins the writer before `native_final`, so every queued item is written before the final status. A callback that arrives after STOP is refused, which invalidates the capture.
- **Bookkeeping bound.** `size_of::<Item>() ≤ max(size_of::<SubmissionEvent>(), size_of::<CompletedGpuTimingEvent>()) + 8`.
  - `CompletedGpuTimingEvent` is one `u8`, four `u64` and a `CompletedGpuTimestamps` (at most 48 bytes), so it is smaller than `SubmissionEvent`.
  - The install check becomes `QUEUED_EVENTS * size_of::<Item>() <= BOOKKEEPING_BYTES / 2` (`native.rs:110`), and the existing test asserts it together with `(QUEUED_EVENTS + WRITE_BATCH) * size_of::<Item>() <= BOOKKEEPING_BYTES`.
  - If `Item` ever exceeds the half-budget, install refuses before any model load. Lowering `QUEUED_EVENTS` would then be its own review.

### 2.3 Joint chain plan record

- The reducer derives each joint command's layer range from the code's plan, not from an assumed layout.
- When the flag is on, `encode_joint_decode` publishes one closed `joint_chain_plan` record per process, the first time it encodes while armed:

```json
{"kind":"joint_chain_plan","payload":{"model":"qwen35","n_layers":64,"chain_n":4,
  "layer_kinds":["linear","linear","linear","full", ...],
  "commands":["embed", "layers[0,4)", ..., "layers[60,64)", "output_head"]}}
```

- `chain_n` is `joint_chain_n(cfg, layer_weights)` (`joint_decode.rs:141`). The command list follows the actual commit points:
  - the observed embedding encoder (`forward_gpu.rs:890`);
  - one `qwen35.joint_decode.layers` commit per `chain_n` layers, with the last group closing at `n_layers` (`joint_decode.rs:512-518`);
  - `qwen35.joint_decode.layers.flush` (`joint_decode.rs:521-523`), listed only when the plan leaves an open chain;
  - `qwen35.joint_decode.output_head`.
- A changed config, such as `HF2Q_PARTIAL_CHAIN_N`, changes the plan record. The plan can never be assumed.

### 2.4 Paired clock samples

- With the flag on, `native_arm_ack` and `native_final` each carry `clock_pair`: `{monotonic_ns, uptime_raw_ns, mach_absolute_ticks, mach_timebase: [numer, denom], read_span_ns}`.
- The three clocks are read back to back. `read_span_ns` is the CLOCK_MONOTONIC time from the first read to the last.
- This supports cross-clock figures with a measured bound (§3.3). It never changes timing.

### 2.5 Cost

- **Events:** T1 (Q4-TRACE, n4) had 41,070 observed commands and 4,380,197 native events against an 8,000,000 `event_limit`. Timing adds one event per command: +41,070, about 0.94%, for 4.42 M of 8 M.
- **Trace bytes:** a `native_gpu_timing` JSONL record is about 260 bytes, so about 10.7 MB per n4 run, against T1's 3.07 GB and a 6 GiB server budget.
- **Live observer metadata:** unchanged. Timing events are not pending-command metadata.
- **GPU work:** none. The timestamps are read after the existing checked wait or retirement.

### 2.6 Traced-run harness (`q2_traced_run.py`)

- `--gpu-timestamps` (default off) writes `native_observer.gpu_timestamps: true` into `native-config.json` only when it is set. `config_sha256` therefore changes only for flagged runs. Default runs and old configs reproduce byte-identically.
- `arm_readback` requires `clock_pair` in the worker-checked arm ACK exactly when the flag is on, and refuses it otherwise.
- `q2_traced_freeze.py` and `q2_certify.py` pass the config and header through unchanged. Scenario `forward_witness_manifest` and `trace_scenario` are not affected.
- `test_q2_traced_run` asserts that the key is absent by default and present when set, and that `clock_pair` readback is gated by the flag.

## 3. Reducer and tools

### 3.1 Closed schema and streaming order (`f2_native_trace.py`, `f2_trace.py`)

- `NATIVE_KINDS` gains `native_gpu_timing` and `joint_chain_plan`.
  - Both are refused unless the header's `native_observer` has `gpu_timestamps: true`.
  - The header's closed `native_observer` key set gains `gpu_timestamps`, only when it is `true`.
  - `native_arm_ack` and `native_final` must carry `clock_pair` exactly when the flag is on.
- `native_gpu_timing` payload: exactly `{schema_version, command_id, invocation_epoch, queue_identity, device_registry_id, timestamps}`.
  - `schema_version` must be 1. The four ids must match the command's `Created` event.
  - `Available` has exactly `{clock, start_seconds, end_seconds}`, with `clock == "MetalCommandBufferHostSecondsSystemMachTime"` and `0 < start < end`, both finite.
  - `Unknown` has exactly `{reason, checked_status, raw_bits}`. The reason must match its cause:
    - `EmptyCommand` only for a command with zero dispatches and zero non-compute operations;
    - `NotHealthyCompleted` only when the drain status is not 4;
    - `DroppedWithoutCheckedTiming` only for a command with no `CheckedDrain`;
    - `TaintedObservation` only for a tainted command;
    - `NonFinite`, `NonPositive` and `NotStrictlyOrdered` only after a status-4 drain, with `raw_bits` present.
- **Streaming order.** Each check runs when its record arrives, with no end-of-trace join:
  - a timing record must name a live (not yet `Dropped`) command;
  - `Available` must follow that command's status-4 `CheckedDrain`;
  - at most one timing record per command;
  - at `Dropped`, a command must have exactly one timing record.
- **Counting.** Timing events count toward the native-event delta check (`p["events"] - baseline == self.events`), because mlx-native counts them in `core.events`.
- **Retention.** With the flag on, each retired command keeps `(gpu_start, gpu_end)` or the Unknown reason.

### 3.2 Clock-free bounds (reducer)

All comparisons below are durations; no offset between clocks is assumed. Metal's host seconds are `f64` and the host instants are integer nanoseconds. Each bound therefore carries an explicit tolerance `ε = max(1 µs, read_span_ns of the larger clock_pair sample)`, so float rounding cannot fail a correct trace.

- **Per command, violation raises:** `gpu_end − gpu_start ≤ CheckedDrain.end_monotonic_ns − Submitted.begin_monotonic_ns + ε`, converted to ns.
- **Per epoch, violation raises:** the union of the epoch's `Available` GPU intervals must be ≤ the host epoch span (first `Created` to last `CheckedDrain`) `+ ε`.
- **Per queue, reported, not raised:** GPU start times should be non-decreasing in commit (`Submitted`) order. The count of inversions is reported.

### 3.3 Cross-clock figures (attribution only, optional)

- The launch bubble is commit end to GPU start. It is reported only when both `clock_pair` samples exist. Its bound is the larger `read_span_ns`, plus the drift of `(monotonic − mach)` between the two samples.
- Otherwise it is not reported. It is never an unbounded [INFERENCE].

### 3.4 Attribution (`q4_trace_attribution.py`)

- **Per epoch and phase:** GPU busy (the union of intervals), GPU bubbles (gaps inside the epoch's GPU envelope), and per-command GPU duration.
- **Per chain, joint decode:** the epoch's commands are ordered by `Created` instant within the epoch, and command ids are recorded.
  - The epoch maps to the `joint_chain_plan` command list only if its command count equals the plan's.
  - It also needs a matching dispatch signature: each command's dispatch count must equal the count first observed for that plan index and logical width.
  - Any mismatching epoch is refused for per-chain attribution and counted as `chain_map_refused`. It is never mapped silently.
  - Per chain, the output is median and p95 GPU time by logical width, with each chain's layer range and layer kinds taken from the plan.
- **Prefill:** prefill commands are committed synchronously, so their host span includes execution. Per-command GPU duration now separates the two.
- **Still not observable:** per-kernel time, and shader busy time inside a command buffer. Those need Metal counters or a capture, which is a separate story.

## 4. Tests and validation

### 4.1 Unit and synthetic tests (no GPU)

- **Rust:**
  - `Item` ordering: one queue, with a timing item enqueued after its event.
  - `record_gpu_timing` returns `Err` on a full queue, after STOP, and with the flag off.
  - Config: explicit `false` refused; `true` without `diagnostic_case` refused; headers without the flag byte-identical.
  - Bookkeeping bound for `Item`.
  - `joint_chain_plan` from the qwen35 plan, for `chain_n` 4 with 64 layers, and for a remainder case.
  - `clock_pair` fields.
- **Python, positive:** timing after drain and before drop; busy-union and bubble arithmetic; plan-derived chain map; launch bubble with clock pairs.
- **Python, negative (each raises):**
  - a timing record without the flag;
  - a timing record before its `CheckedDrain`, or after `Dropped`;
  - a duplicate timing record, or none at `Dropped`;
  - `Available` without a status-4 drain;
  - a wrong clock string;
  - open `Unknown` keys;
  - a reason/cause mismatch (`EmptyCommand` with dispatches, `NotHealthyCompleted` with status 4);
  - a GPU duration longer than its host submit-to-drain bracket;
  - an epoch GPU union longer than its host span;
  - an event-count delta mismatch.
- **Python, refusal:** command-count or dispatch-signature mismatch → `chain_map_refused`.
- **Compatibility:** every existing reducer, attribution, Q2 and verdict test passes unchanged. Old-policy reports are byte-identical.

### 4.2 Hardware (after a grant; correctness only, per the operator's 2026-11-06 no-speed-test directive)

- **G2 (model-level, bitwise; does not depend on run-to-run determinism):** a sibling of Q4-TRACE's G1.
  - The process installs the observer with the timestamp stream and a counting sink.
  - The same joint forwards (logical4..1 padded, identical caches) run unarmed and armed. Logits, hidden, cursors, KV and recurrent bytes must be bitwise equal.
  - Each armed forward must produce exactly one timing event per observed command. Every armed joint command is submitted (embed, the layer chains and the output head; 18 for Qwen3.8), so all must be `Available`. `EmptyCommand` and `DroppedWithoutCheckedTiming` are allowed only under the §3.1 reason-to-cause rule, and G2 asserts their count is 0 for joint forwards.
  - Run alone with the full test path and `--exact`.
- **T1n / T1g (same binary, back to back):** the new binary's traced n4 run with the flag off (T1n), then with it on (T1g).
  - `q1_decision_policy.py transcripts T1n T1g` must be `ExactTranscripts`.
  - T1g must reduce: one timing record per command; `Available` for every healthy command with work; `Unknown` only for `EmptyCommand` or `DroppedWithoutCheckedTiming`; §3.2 bounds pass.
  - Report the timing-event delta against `event_limit` and the trace-byte delta against the server budget.
- **Then:** run `q4_trace_attribution.py` on T1g and hand the per-chain table to the Qwen op-parity lead.

## 5. Risks

1. **Overlap semantics.** Metal may overlap back-to-back command buffers on one queue. Per-command envelopes and their union are reported as measured. Overlap is never converted into exclusive time. The commit-order inversion count shows when this happens.
2. **Queue pressure.** One queue now carries both streams. The timing burst at a forward's final wait is 18 events per joint step, small beside the dispatch burst the 2,048-slot queue already absorbs. A full queue invalidates the capture; it never drops silently.
3. **Plan drift.** A change to the joint commit structure must update the plan record. Otherwise the per-chain map is refused (counted), not wrong.

## 6. Review resolution (revision 2)

| Review677 finding (rev 1) | Resolution |
|---|---|
| 1 [P1] A second queue can reorder timing versus `CheckedDrain`/`Dropped` | One bounded queue of `Item {Event, Timing}`, drained in order and published as ordered runs (§2.2). Streaming order checks (§3.1). Writer exit condition stated. |
| 2 [P2] Chain-to-layer map assumed | `joint_chain_plan` record derived from `joint_chain_n` and the actual commit points (§2.3). Per-epoch command-count and dispatch-signature match, otherwise `chain_map_refused` (§3.4). |
| 3 [P2] Clock-free checks; optional clock pairs | Per-command and per-epoch duration bounds, per-queue inversion count (§3.2). `clock_pair` at arm and final, giving a launch bubble with a measured bound (§2.4, §3.3). |
| 4 [P2] Same-binary baseline; model-level bitwise test | T1n/T1g back to back on one binary, plus G2 bitwise armed vs unarmed with the stream installed (§4.2). Budget deltas reported. |
| 5 [P3] Closed reducer schema; explicit `false` | Clock string, closed `Unknown` keys, reason-to-cause rules; `false` refused; `skip_serializing_if` keeps headers byte-identical (§2.1, §3.1). |
| 6 [P3] Code basis 0.16.0; budget arithmetic | Basis `dafb63ab` / mlx-native 0.16.0, line references corrected (§1). `size_of::<Item>()` bound and event/byte arithmetic (§2.2, §2.5). |

Re-review of revision 2 (Review677, `5ae19c2a`): one P2 and three P3s. Revision 3 applies all four.

| Finding | Resolution |
|---|---|
| 1 [P2] The `q2_traced_run.py` flag was dropped | §2.6 adds `--gpu-timestamps` (default off, `config_sha256` changes only when set), `clock_pair`-gated arm readback, pass-through in freeze/certify, and tests. |
| 2 [P3] Bounds tolerance | §3.2 tolerance `ε = max(1 µs, read_span_ns)`. |
| 3 [P3] G2 `Unknown` rule | §4.2: every armed joint command is submitted, so all are `Available`. `EmptyCommand`/`DroppedWithoutCheckedTiming` fall under the §3.1 rule, with an asserted count of 0. |
| 4 [P3] Who records approval | The ADR story names who records it (Main on Review677's ACCEPT, per Main's instruction), and the operator if the story issue requires operator approval. |
