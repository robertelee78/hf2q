# ADR-063 Q4-TRACE follow-up: per-command GPU timestamps in the phase trace (draft)

- **Status:** Draft design (revision 1), awaiting independent review and operator approval. No code yet.
- **Motivation:** the Q4-TRACE T1 attribution (2026-11-06, `~/.local/state/hf2q/v0130/q4-trace/attribution-handoff.md`) splits a Qwen3.8 joint step into host encode (median about 7 ms) and a completion tail (about 79 ms). It cannot say which layer chains spend that time. The joint decode commits its 4-layer chains back to back and checks them together at the end of the forward, so the trace's `CheckedDrain` instants are the final waits, not per-command completions. The gap program needs per-chain GPU time.
- **Code basis:** main `dafb63ab` (Q4-TRACE merged), mlx-native 0.15.18.

## 1. What mlx-native already provides

- `install_submission_observer_with_gpu_timestamps(sink, event_limit, metadata_limit_bytes)` (`submission_observer.rs:180`) turns on a separate v1 stream. Its events count against the same event quota. It adds no command owners and no waits.
- For every observed command, `CommandObservation::completed_gpu_timing` (`submission_observer.rs:726-737`) emits exactly one `CompletedGpuTimingEvent {schema_version: 1, command_id, invocation_epoch, queue_identity, device_registry_id, timestamps}` through `SubmissionObserver::record_gpu_timing`.
  - The event is emitted after the command's checked wait (`encoder.rs:2879`, `wait_observed`).
  - A command dropped unchecked gets `Unknown {DroppedWithoutCheckedTiming}` (`submission_observer.rs:856-859`).
- `timestamps` is one of:
  - `Available {clock: MetalCommandBufferHostSecondsSystemMachTime, start_seconds, end_seconds}`: Metal's `GPUStartTime`/`GPUEndTime` of that command buffer, read after a healthy completion.
  - `Unknown {reason, checked_status, raw_bits}`: not healthy, empty, non-finite, non-positive, not ordered, dropped unchecked, or tainted (`gpu_timestamps.rs`).
- These are command-buffer execution envelopes. They are not shader busy time, and they are not mapped to CLOCK_MONOTONIC (the module states this explicitly).

So no mlx-native change is needed. hf2q installs the observer without the stream (`src/serve/phase_trace/native.rs:147`). Its `Sink` implements only `record`, and the default `record_gpu_timing` returns `Err`. Enabling the stream today would therefore invalidate the capture.

## 2. Change (hf2q only)

1. **Opt-in config.** Add `gpu_timestamps: bool` to the phase-trace `native_observer` config (`#[serde(default)]`, false). The flag is accepted only with `diagnostic_case: true`. Default serving and existing configs are unchanged. The trace header echoes the key only when it is true, so old headers stay byte-identical.
2. **Install.** When `gpu_timestamps` is true, `native::install` calls `install_submission_observer_with_gpu_timestamps`.
3. **Sink.** `Sink::record_gpu_timing` mirrors `record`: a fixed-size `Copy` payload into a bounded non-blocking queue (a second `SyncSender`), no filesystem, no blocking.
   - A full queue returns `Err`, which invalidates the capture (fail closed, same as `record`).
   - The existing writer thread drains both queues. It publishes timing events as `native_gpu_timing` records through `publish_all`.
4. **Budget.** Each observed command adds one event to the shared `event_limit`. T1 had 41,070 commands against 4.38 M native events, so this adds about 0.9% (the run's limit is 8 M). It also adds about 250 bytes of JSONL per command, about 10 MB per n4 run. The second queue's bound is folded into the existing `BOOKKEEPING_BYTES` check (`native.rs:110`) and its test.
5. **`q2_traced_run.py`:** new `--gpu-timestamps` flag, default off. It changes `config_sha256` only when set.

No inference, scheduling or kernel change. The timestamps are read only after the checked wait that already happens.

## 3. Reducer and tools

- **`f2_trace.NATIVE_KINDS`** gains `native_gpu_timing`. `NativeTrace` accepts it only when the header's `native_observer.gpu_timestamps` is true, and raises otherwise. The `native_observer` closed key set gains `gpu_timestamps`, and only when it is true.
- **Validation, fail closed:**
  - `schema_version == 1`;
  - `command_id` is a created command, its `invocation_epoch`, `queue_identity` and `device_registry_id` match its `Created` event, and it has exactly one timing event;
  - `Available` requires `0 < start < end`, both finite, and a `CheckedDrain` with status 4;
  - `Unknown` must carry a closed `reason`.
- **Counting.** Timing events count toward the native event delta check (`p["events"] - baseline == self.events`). The observer counts them there too.
- **Retention.** Only under the flag, keep `(gpu_start, gpu_end)` per retired command (two floats).
- **`q4_trace_attribution.py`, per epoch and phase:**
  - **GPU busy:** the union of command `[start, end]` intervals.
  - **GPU bubbles:** the gaps inside the epoch's GPU envelope.
  - **Per command:** GPU duration. For joint decode, commands are indexed in encode order: command 0 is the embedding gather, then the 4-layer chains, then the head. The chain index therefore maps to a layer range (16 chains of 4 layers for Qwen3.8's 64 layers).
  - **Per chain:** median and p95 of each chain's GPU time, by logical width.
  - **Prefill:** the same split, with no host-versus-GPU confusion. Prefill commands are committed synchronously, so their host spans include execution; GPU duration separates the two.
- **Clock.** No mapping between GPU seconds and the trace's CLOCK_MONOTONIC is claimed. Durations and in-epoch ordering use the GPU clock only. Host-side spans stay on CLOCK_MONOTONIC. Any cross-clock statement is marked as inference.
- **Still not observable:** per-kernel time, and the shader busy share inside a command buffer. Those need a Metal counter or capture run, which is a separate story.

## 4. Tests and validation

- **Rust:** `Sink::record_gpu_timing` queues and returns `Err` on a full queue. The config flag is refused without `diagnostic_case`. The extended bookkeeping bound check passes.
- **Python synthetic tests:**
  - **Positive:** one timing event per command; busy-union and bubble arithmetic; chain-index mapping.
  - **Negative:** a timing record without the flag; a duplicate or orphan timing event; `Available` with `end <= start`; `Available` without a healthy drain; an event-count delta mismatch.
  - **Compatibility:** every existing reducer and attribution test passes unchanged, and old-policy reports are byte-identical.
- **Hardware, after a grant, one-shot correctness runs only (no speed tests per the operator's 2026-11-06 directive):**
  - T1g: a traced n4 run with `--gpu-timestamps`. It must reduce, with every command `Available` or a justified `Unknown`, and its transcripts must be `ExactTranscripts` against the merged T1 shape (the same binary traced without timestamps). Serving is deterministic since #717.
  - Then run `q4_trace_attribution.py` on T1g and hand the per-chain table to the Qwen op-parity lead.

## 5. Risks and open questions

1. **Sink back-pressure.** Timing events arrive in bursts at a forward's final wait, 18 per joint step. The current 2,048-event queue already absorbs the dispatch bursts. A second queue of equal size covers this.
2. **`GPUStartTime`/`GPUEndTime` semantics.** Under back-to-back commits, Metal may overlap or serialize command buffers on the queue. The design reports per-command envelopes and their union as measured. It does not convert overlaps into exclusive time.
3. **Scope.** A Gemma or DeepSeek attribution can reuse the same stream with no further hf2q change beyond its own trace runs.
