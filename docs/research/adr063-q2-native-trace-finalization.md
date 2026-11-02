# ADR-063 Q2 native trace finalization

Issue #368, items 4 and 6. These are the traced Qwen3.8 27B cohorts in
`q2-certify-v1/runs/q2-traced-*` (server source 38485a23, mlx-native
0.15.15, `native_command_capture_v3` with 8,000,000 events and 16 MiB of
metadata). Only n1 produced a clean `end` record.

| run | native events | `native_final` (seq) | pending | invalid / dropped | `end` |
|---|---:|---|---:|---|---|
| n1 | 879,026 | 940165 | 0 | false / 0 | seq 940166 |
| n4 | 2,559,923 | 2658033 | 1 | false / 0 | none |
| 3+1 | 2,168,590 | 2272159 | 1 | **true / 502,449** | none |
| 2+2 | 2,715,753 | 2819749 | 1 | false / 0 | none |
| 1+3 | 2,674,005 | 2776358 | 1 | false / 0 | none |

The runs show two separate defects. Each run with an unretired command has
`metadata_reserved_bytes: 688`, which is one retained pending entry.

## 1. MTP catch-up copy checked after its own drain

**Mechanism.** In mlx-native, a command that is committed but not waited
enters the observer's pending registry only when its `CommandEncoder` drops
(`encoder.rs` `Drop` → `retire_observation`). A later same-queue
`wait_observed` → `inspect_after_wait` checks only entries already in that
registry. `finish()` requires `pending_commands == 0`.

`MtpWeights::process_target_batch` (`src/inference/models/qwen35/mtp.rs`)
works like this:

1. It commits the shifted-hidden copy encoder `enc`.
2. It encodes the MTP projection and KV write.
3. It drains with `commit_and_wait_labeled("mtp.process_target_batch")`.

`enc` stayed alive until the function returned, which was after that drain
had inspected the registry. So the copy was never checked by its own
transaction. The next same-queue wait checked it instead. If the catch-up
was the last GPU work before shutdown, nothing checked it.

**Evidence (n4, final joint step, two lanes):**

- seq 2657990–2657992: cmd 51153 `Created`. It has one `DispatchEncoded`
  with grid `[5120,1,1]` (the pending-hidden row copy; hidden size 5120;
  pipeline 532878741888), then `Submitted`.
- seq 2657993–2658010: cmds 51154–51156 run the MTP projection and KV write.
- seq 2658011–2658012: cmd 51157 is the empty drain command.
- seq 2658013–2658024: the drain's inspection checks 51147, 51156, 51155,
  51154, 51152 and 51157. It does **not** check 51153.
- seq 2658025–2658032: both slots release.
- seq 2658033: `native_final` reports `pending_commands: 1`. No `end` record
  follows.
- The first lane of the same step shows the same pattern. Its copy, cmd 51147
  (seq 2657954–2657956), is missing from its own drain batch (seq
  2657977–2657986, after the empty cmd 51151). It is checked only at seq
  2658013, by the second lane's drain. The previous step's copy, cmd 51123
  (submitted at seq 2655883), is checked at seq 2657902.
- Across the whole run, all 3,380 copy-shaped commands in n4 missed their own
  drain: 3,379 were checked one inspection later and 51153 never. All 244
  copies in n1 were checked one inspection later. These copies have grid 5120
  (1 or 2 dispatches) or 3,681,280 (prompt catch-up). The only submitted
  command in n4 that never drained is 51153.
- 2+2 and 1+3 ended on a single-slot K3 catch-up (`seq_len` 4). The
  unmatched commands are 53531 (seq 2819710–2819713) and 54326 (seq
  2776319–2776322). Each has two copy dispatches: grid `[5120]` for the
  pending row and `[15360]` for three target rows.
- n1 is clean only because of its ordering. Its last catch-up copy, cmd 55253
  (seq 709494–709497), was checked by the next transaction's drain at seq
  711236. Its final command, cmd 73725, was itself waited.

Joint decode is therefore not the cause. The trace is left unclean when the
cohort's last GPU transaction is an MTP catch-up. A joint step with
MTP-holding rows always ends each lane with one.

A second late-check class exists but never reaches the end of a run. Some
4-dispatch prefill commands (grid `[4, N, 1]`: 96 in n1, 384 in n4) are
checked one inspection late, but always by a later wait inside the same
prefill.

**Fix.** `process_target_batch` now drops the copy encoder right after
`commit()`. The command is then in the registry before the transaction's
drain, and that drain checks it. Observer semantics are unchanged: `finish()`
still invalidates any command that is genuinely unretired.

## 2. Native sink queue overflow (3+1 dropped events)

**Mechanism.** When a sink `record` returns an error, mlx-native's `emit`
calls `fail()`. That sets a sticky invalid flag and drops every later event.
`dropped_events` is not an event-limit count here: 2,168,590 recorded plus
502,449 dropped is 2,671,039, close to the other runs' totals and well
under 8M. Metadata was 688 bytes out of 16 MiB.

The HF sink (`src/serve/phase_trace/native.rs`) calls `try_send` into
`sync_channel(1024)`. Its writer thread used to drain one event per trace
lock and append it as a `Value` with two `write(2)` calls.

**Evidence (3+1):**

- The last native record is seq 2250801. It is cmd 44060, `DispatchEncoded`
  ordinal 60, in the middle of encoding a 159-dispatch decode command.
  After it come only HF records (seq 2250802 `forward_witness_unmapped`,
  and so on).
- Queue depth can be reconstructed from each event's production time
  (`end_monotonic_ns`) and write time (`origin_monotonic_ns + at_ns`). When
  seq 2250801 was produced, exactly **1,024** produced events were still
  unwritten. The queue was full, so the next `try_send` failed. Writer lag
  had grown to 8.3 ms.
- The writer was saturated: 99% of native events in every run were written
  while the writer was more than 200 µs behind. It averaged 7.5 µs per event
  (n1 6.9 µs).
- Producer peaks in n4: 592 events per 1 ms, 1,005 per 2 ms, 1,880 per
  10 ms.
- Maximum reconstructed depth in the other runs: n4 975, 2+2 939, 1+3 938,
  n1 649. Every multi-agent run came within 9% of overflow.

**Fix.**

- The writer drains up to 64 queued events at a time. It appends them with
  one trace lock and one `write`.
- Each record is still sequenced and charged on its own. If one record fails,
  the earlier records in the batch stay written and the failure is sticky,
  exactly as with single appends.
- Records serialize through a typed `Line { schema, seq, at_ns, kind,
  payload }`. A test proves this is byte-identical to the previous `json!`
  object.
- The queue bound is raised from 1,024 to 2,048 events (475,136 bytes). This
  stays within the existing half-of-bookkeeping check, and a new test pins it
  for the current native event size.
- Measured on 100,000 real 3+1 events (release build, temporary benchmark,
  not committed): 6.05 µs/event with the old path, 0.52 µs/event with
  64-event batches. The trace lock is held about 33 µs per batch.
- Replaying the recorded producer timeline against an approximate writer
  queue model gives these maximum depths:
  - old writer at 7.56 µs per event: 851 in n4 and 860 in 3+1 (the
    reconstructed depths were 975 and 1,024);
  - new writer at 0.61 µs per event (the measured 0.52 µs scaled by the same
    1.18 contention ratio): 563 in n4 and 559 in 3+1. The writer's 1 ms idle
    park now sets the depth.

## Verification

- Host tests: `cargo test --bin hf2q serve::phase_trace` passes 19 tests.
  New ones are `batched_append_matches_single_appends_and_stops_at_first_failure`,
  `typed_native_line_matches_json_object_record_bytes` and
  `queue_and_write_batch_fit_reserved_bookkeeping`.
- No host test covers the encoder-lifetime fix. The observer installs once
  per process, before any native device exists, so it can't run in the shared
  test binary.
- Hardware confirmation still needed: a traced n4 cohort, plus 3+1, run with
  a binary built from this change. A clean run shows:
  - `native_final` with `pending_commands: 0`, `invalid: false`,
    `dropped_events: 0` and `metadata_reserved_bytes: 0`;
  - a trailing `end` with `error` null;
  - every grid-5120 copy command checked in the same inspection batch as its
    own `mtp.process_target_batch` drain.
