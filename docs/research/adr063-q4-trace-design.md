# ADR-063 Q4-TRACE (#663): Qwen joint-decode forward witness — HLD/LLD

- **Status:** Design revision 3, awaiting operator approval (ADR-063 "Execution entry": each implementation story needs a recorded design approval). Revision 2 resolved the independent review `ReviewTraceDesign` of revision 1 (`26e2e462`, verdict ACCEPT-WITH-CONDITIONS). Revision 3 applies that review's closure-check fixes A–C to revision 2 (`b103c303`). See §8.
- **Story:** [#663](https://github.com/robertelee78/hf2q/issues/663). Parent #355, milestone v0.1.30. Blocked by #371 (Q5); blocking #658.
- **Source of truth:** `docs/adr/ADR-063-apple-silicon-four-agent-speed-coherence.md` §Q4-TRACE (lines 780–802), §Q4-QWEN38 (586–610), and the v0.1.31 successor list (1441–1446). Also `docs/adr/ADR-065-four-agent-joint-speculative-verify.md` §E1 (147–167) and §E4 (214–234).
- **Code basis:** main `d817dffa7371a2db85a43ed1d6502350906c33f5` (v0.1.29 plus later merges), mlx-native 0.15.16 (pinned, `Cargo.toml:174`). Line numbers below refer to this commit.

## 0. Acceptance (bound text, #663)

1. `scripts/four_agent/f2_native_trace.py` reports `unmapped_operator_attempts == 0` for every `joint_ordinary_decode` forward of a full traced four-agent Qwen3.8 run. The Q2 closure QE §6 found 43,008 per run, as `forward_witness_unmapped` records with reason `NoActualInputManifest`.
2. The hook is off by default and fails closed. With it on, transcripts are `ExactTranscripts` against it off, and its timing overhead is measured and stated.
3. Its output is consumable by Q4-QWEN38 attribution. The Q2 v3 counting rule that consumes it stays successor scope (v0.1.31).
4. Ownership: mlx-native owns low-level trace support; hf2q owns the witness wiring, records and validation.
5. Validation: an independent review of a traced run with `unmapped_operator_attempts == 0`, plus the trace-on/off exactness control.

## 1. Where the 43,008 come from (measured on retained evidence)

The evidence is the Q2 traced n4 run `$E/q2-certify-v1/runs/q2-traced-n4-fix-20261102T121427Z/native-phase-trace.jsonl` (server source `1c184cf2`, native 0.15.15). I read it with a streaming pass; no GPU or model work was involved.

| Fact | Value |
|---|---|
| `transaction` records with `operation == "joint_ordinary_decode"` | 896: 770 with 4 lanes, 72 with 3, 54 with 2 |
| `forward_witness_unmapped` records | 43,008, all inside joint transactions, exactly 48 per joint transaction |
| `forward_input` / `forward_witness` records | 138 / 138, all from `bounded_prefill` and `ordinary_or_temporal_mtp_decode` transactions, none from joint ones |
| Model shape (server log) | `n_layers=64`, `full_attention_interval=Some(4)`, so 48 DeltaNet layers and 16 full-attention layers |
| Joint mode (server log) | `mode="padded_mv_ext" dense_ffn="shared_weights"` (Qwen3.8 production policy) |
| Commands in one joint epoch (epoch 561) | 18. Each has `Created` → `DispatchEncoded`… → `Submitted` → `CheckedDrain` → `Dropped`. 16 of them are 4-layer chains with 117 dispatches each. |
| Existing reducer output | `windows*/windows-q2-traced-n4-fix-…json`: `"unmapped_operator_attempts": 43008` |

So 43,008 = 896 joint forwards × 48 DeltaNet pre-norm calls.

The mechanism:

- `run_qwen35_joint_decode` opens the owner invocation with `scheduler.trace_transaction("qwen35", "joint_ordinary_decode", handles)` (`src/serve/api/engine/qwen35_joint_decode.rs:355-362`) and then calls `guard.model.forward_joint_decode(...)` (363-369).
- `trace_transaction` (`src/serve/scheduler.rs:1434-1484`) publishes `transaction` and returns `native_observation::invocation(owners)`. That sets the thread-local `SELECTED` owners and a fresh `EPOCH` (`src/inference/native_observation.rs:338-363`).
- Nothing on the joint path constructs an input manifest. Compare the serial paths, which call `native_observation::qwen_input(...)` at the top of the forward (`src/inference/models/qwen35/forward_gpu.rs:3995-3996` for `qwen35.forward.actual_tokens`, `6536-6537` for `qwen35.greedy.actual_tokens`).
- The joint DeltaNet layer runs `apply_pre_norm(enc, …, x, …, width_u32, …)` (`src/inference/models/qwen35/joint_decode_core.rs:331`). That calls `witness_command_rms_norm` (`src/inference/models/qwen35/gpu_delta_net.rs:676`) → `prepare` (`src/inference/native_observation/witness.rs:171-234`). With no `INPUT` descriptor in place, `prepare` publishes `forward_witness_unmapped {operator: target_input_rms_norm, reason: NoActualInputManifest, progress_eligible: false}` (`witness.rs:180-189`) on every DeltaNet layer.
- Full-attention layers use `apply_pre_attn_rms_norm` (`joint_decode_core.rs:676`), which is not witnessed. That is why the count is 48 per forward rather than 64.
- The reducer counts each such record (`scripts/four_agent/f2_native_trace.py:181-185`) and reports the total as `unmapped_operator_attempts` (`:373`).

With a descriptor in place, `prepare` claims it exactly once (`witness.rs:190-192`, `witness_claimed.swap`). The first DeltaNet pre-norm becomes the one `forward_witness`, and the other 47 return `None` without publishing. This is the same one-witness-per-forward shape the serial path already produces (`layer: null`, one Known scope, one dispatch).

### 1.1 The padding trap (why "just call `ordered_input`" fails closed)

In `PaddedMvExt` mode every joint step is padded to `PADDED_MV_EXT_WIDTH = 4` rows (`src/inference/models/qwen35/joint_decode_routes.rs:45`, `122-127`). The padding rows replay row 0's token and position, and their outputs are discarded (`src/inference/models/qwen35/joint_decode.rs:314, 321-332`). The pre-norm matrix is the `[width, h]` embedding buffer.

- `footprint` (`witness.rs:141-166`) requires `rows == descriptor.total` and `element_count == rows × cols`.
- A manifest that declares only the logical rows (2 or 3) against a 4-row matrix therefore hits `invalidate("actual RMS matrix shape/view disagrees with forward input")` (`witness.rs:203-207`), and the whole trace is invalid.
- In the evidence run that would happen on 126 of 896 joint forwards (14.1%).
- A manifest that assigns padding rows to an owner would be false.

The design must therefore declare padding as physical rows that have no owner (§3.2).

## 2. HLD

### 2.1 Goal and non-goals

The goal is that every Qwen joint ordinary decode forward emits exactly one source-bound `forward_input` and its single `forward_witness`. Each forward must be:

- bound to the owners of its `joint_ordinary_decode` transaction, in physical (matrix) row order;
- explicit about owner-less padding rows;
- reduced by `f2_native_trace.py` with `unmapped_operator_attempts == 0` and a positive coverage count: witnessed joint forwards == joint transactions;
- bounded: a closing `forward_input_end` marker separates the forward's own native commands from the per-lane settle work (sampling, MTP catch-up) that shares its invocation epoch (§3.2, §3.4);
- invisible to the Q2 v2 counting rule: joint witnesses never enter `NativeTrace.finish()["submissions"]`, so every `forward_demand_v2` window verdict is unchanged (§3.4, test P5).

Non-goals:

- The Q2 v3 `queued_forward_demand_v3` counting rule, the `f2_demand.py` and `phase_trace.rs` queued-demand source allowlists, and any consumption of joint witnesses by `f2_trace.qualify_window` (service brackets, `witness_progress`, `slots_without_required_forward_witness`) or `operator_shared_batch4_observed`. These are successor scope (ADR-063:1443); see §3.4 "Isolation from Q2 v2" and §3.6.
- Witnesses for full-attention layers or full-body batching. The policy stays `batching_scope: operator_only_full_body_unknown`.
- Joint verify rows (ADR-065 J-series). Here a joint source with role `target_verify` is rejected (fail closed), so a future joint verify cannot pass as ordinary.
- Gemma and DeepSeek joint witnesses.
- E1 catsplit itself.
- Any mlx-native change. The existing 0.15.16 submission-observer API (`CommandEncoder::enable_submission_observation`, `set_observation_scope`, `set_observation_rotation_scope`, `encoder.rs:1132-1168`) is sufficient. Ownership item 4 is met with "no low-level change needed", and this is stated in the PR.

### 2.2 Shape of the change

```
run_qwen35_joint_decode (engine)            trace_transaction → Invocation{SELECTED=owners, EPOCH=e}
  └ forward_joint_decode (engine_qwen35)    supervisor lease, same worker thread
     └ forward_gpu_decode_joint (model)
        └ encode_joint_decode               tokens/positions built for `width` rows
           ├ NEW: _native_input = native_observation::joint_decode_input(rows, width)
           │        → publishes forward_input{source: qwen35.joint_decode.actual_slot_rows,
           │                                  packed_rows: one per row, padding_rows: width−rows}
           ├ layer 0 (DeltaNet) apply_pre_norm → witness_command_rms_norm → prepare()
           │        → scope_definition{Known, rows=logical} + forward_witness{rows=width}
           ├ layers 1..63: DeltaNet pre-norms see a claimed descriptor → silent (no unmapped)
           └ output head commit_and_wait; guard dropped on return (before finish_joint_tick)
                    → publishes forward_input_end{forward_id, epoch, scope_watermark, end_monotonic_ns}
  └ finish_joint_tick per lane              same EPOCH, after the marker: "settle" commands
```

The reducer gains five things:

- the new source, with its layout and padding rules;
- a transaction→input binding for `joint_ordinary_decode`, with an epoch freshness and consistency check;
- the `forward_input_end` boundary, splitting each joint epoch into forward and settle commands;
- isolation: joint witnesses are validated and reported, but kept out of `submissions`;
- report fields Q4-QWEN38, E1 and E4 can consume, only under the joint policy.

The harness witness manifest adds the source.

### 2.3 Off by default, fail closed

- **Off:** `joint_decode_input` returns an inactive `Input` after one `ARMED.load(Acquire)` (the same gate as `qwen_input`, `native_observation.rs:150-152`), before it consumes its iterator or allocates anything. `ARMED` is set only by `arm()` (`native_observation.rs:78-94`). That needs an installed observer, a one-shot fresh process, and a `HF2Q_PHASE_TRACE_CONFIG` with `native_observer.diagnostic_case == true`. No serving default changes, there is no new environment variable, and GPU work is unchanged in every mode. The witness only changes encoder scope metadata (`set_observation_scope`), never kernels, buffers or ordering.
- **Fail closed (server):** a malformed joint manifest calls `input(source, &[])`. Examples: a duplicate slot, a negative position, `rows > width`, `width > MAX_JOINT_DECODE_ROWS (8)`, padding on a non-joint source, or an owner that is stale or missing against `SELECTED`/`OWNERS`. That path invalidates the trace (`witness.rs:105-109`, "actual forward input lacks current owner/epoch binding"), and the trace `end` record then carries the error the reducer rejects. Inference results are never affected (`witness.rs:168-170`).
- **Fail closed (reducer):** the policy must name the source (`f2_native_trace.py:58-67, 135`). Any joint-source record that breaks the new rules raises `ValueError` (§3.4).

## 3. LLD

### 3.1 Source identity

New source string: `qwen35.joint_decode.actual_slot_rows`. It is an independent-lane source: one span per lane, `token_count == 1` per span, unique slots, 1–8 spans, operator site `pre_attention_input`, role `target` only. It is the only source that may declare `padding_rows > 0`.

The string is added to the following lists in one change:

| Site | Change |
|---|---|
| `src/inference/native_observation/witness.rs:22-48` `source_layout` | New arm: `1..=8` spans, all `count == 1`, slots pairwise distinct (same predicate shape as `gemma4.decode.actual_token_slot_position_arrays`, lines 35-45) |
| `src/serve/phase_trace/native.rs:32-44` `WitnessManifest::validate` sources | Append the string |
| `scripts/four_agent/f2_native_trace.py:15-19` `SOURCES` | Append the string |
| `scripts/four_agent/f2_trace.py:19-20` `NATIVE_KINDS` | Add `forward_input_end` (consumed only by `NativeTrace`, rejected unless the policy names the joint source) |
| `scripts/four_agent/q2_traced_run.py:57-60` `WITNESS["input_sources"]` | Append the string. This changes the config digest (`config_sha256`) for new runs only. |
| `scripts/four_agent/test_q2_traced_run.py:113-114` | Update the expected list |

### 3.2 Record format (closed schemas)

`forward_input` for the joint source is the existing closed set plus one key, `padding_rows`. Old sources keep their exact schema, so retained v2 traces validate byte-for-byte as before.

```json
{"kind":"forward_input","payload":{
  "forward_id":9001,"epoch":561,"source":"qwen35.joint_decode.actual_slot_rows","role":"target",
  "matrix_rows":4,"padding_rows":1,
  "packed_rows":[
    {"matrix_row_start":0,"matrix_row_end":1,"logical":{"request_id":0,"slot_id":0,"generation":0,"token_position":1834,"token_count":1,"phase":"ordinary"}},
    {"matrix_row_start":1,"matrix_row_end":2,"logical":{"request_id":1,"slot_id":1,"generation":0,"token_position":2210,"token_count":1,"phase":"ordinary"}},
    {"matrix_row_start":2,"matrix_row_end":3,"logical":{"request_id":3,"slot_id":3,"generation":0,"token_position":977,"token_count":1,"phase":"ordinary"}}],
  "operator_site":"pre_attention_input","execution_proven":false}}
```

The values above are illustrative.

Invariants:

- Packed spans are contiguous from row 0 and cover the logical prefix `[0, matrix_rows − padding_rows)`.
- Rows `[matrix_rows − padding_rows, matrix_rows)` are padding: they replay row 0's input, have no owner, and their outputs never reach a cache (`joint_decode.rs:321-323`).
- `token_position` is the row's `Qwen35JointRow.position`, the cache cursor being fed (`engine_qwen35/joint_decode.rs:85-110`).
- Packed order is the `rows` order. That is the slot-sorted `prepared` order (`qwen35_joint_decode.rs:189-192, 351-354`) and the matrix row index used by `embed_tokens_gpu_into` and `positions` (`joint_decode.rs:324-332`).

The following records are unchanged in schema:

- `scope_definition {membership: Known, operation: target.input_rms_norm, rows: <logical rows only>}`.
- `forward_witness {rows: matrix_rows (physical, includes padding), columns: hidden, dtype: F32, input_read_bytes: matrix_rows × hidden × 4, layer: null, dispatches_expected: 1, …}`.

`prepare` already emits both from the descriptor (`witness.rs:209-233`). Its scope rows come from `descriptor.rows` (logical), and its footprint uses `descriptor.total`, which becomes the physical `matrix_rows`.

**New record, joint source only: `forward_input_end`.** The joint input guard publishes it once when it drops, on success, on an early `?` return, and during unwinding alike, so it marks where this forward's encoding actually stopped:

```json
{"kind":"forward_input_end","payload":{"forward_id":9001,"epoch":561,
  "scope_watermark":9188,"end_monotonic_ns":1189189901234567}}
```

- `scope_watermark` is the highest id handed out by `next_id()` (`native_observation.rs:365-370`) at drop time. Forward ids, epochs and scope ids share that one monotonic counter, and every observed command receives a fresh auxiliary scope id when its encoder is created (`native_observation::encoder`/`attach`, `native_observation.rs:517-580`). So a command whose first `DispatchEncoded` scope id lies in `(forward_id, scope_watermark]` was created and encoded inside this forward. A command of the same epoch with a first-dispatch scope id above the watermark belongs to the per-lane settle work after the forward (`finish_joint_tick`, `qwen35_joint_decode.rs:372-380`, including the coherent lanes' MTP catch-up). A command of the epoch with a first-dispatch scope id below `forward_id` was created before the input (expected none, because the guard precedes the embedding gather); it is counted as `pre_forward_commands` and reported, never folded into forward time. This is an integer join and has no clock ties.
- `end_monotonic_ns` is `CLOCK_MONOTONIC` read at drop, the same clock the phase trace and native events use (`phase_trace.rs:291-296`). It is used only for commands that carry no dispatch (no scope id): `Created.begin_monotonic_ns < end` is forward, `> end` is settle, and equality is counted as `boundary_ambiguous` and excluded from per-forward timing.
- Records with kinds other than these keep their schemas. The marker exists only for the joint source, so old traces and other sources are untouched.

### 3.3 Server wiring (hf2q)

**(a) `native_observation/witness.rs`**

1. `input(source, spans)` becomes a thin wrapper over a private `input_rows(source, spans, padding_rows: u32)`.
2. In `input_rows`, after `resolve_selected`, set `total = Σ logical token_count + padding_rows` using checked arithmetic.
3. Refuse `padding_rows != 0` unless `source == JOINT_SOURCE` (that is, emit nothing and invalidate via `descriptor = None`).
4. Emit `"padding_rows"` only for `JOINT_SOURCE`.
5. `Descriptor` is unchanged. `total` is now the physical row count, which is exactly what `footprint` compares against the RMS matrix.
6. `Drop for Input` (`witness.rs:49-55`): when the guard is active and its own descriptor's source is `JOINT_SOURCE`, publish `forward_input_end` before restoring `previous`. The watermark is `NEXT.load(Acquire) - 1`, and the clock is read with `clock_gettime(CLOCK_MONOTONIC)`. Nothing is published when unarmed or for other sources.

All other callers pass `0` and emit byte-identical records.

**(b) `native_observation.rs`: new adapter next to `qwen_input` (144-171)**

```rust
/// Joint ordinary decode: one actual one-token span per packed row, in matrix
/// order; `matrix_rows` is the physical packed width (padding rows have no owner).
pub(crate) fn joint_decode_input(
    rows: impl Iterator<Item = (u32, i32)>,   // (slot, fed position) per packed row
    matrix_rows: usize,
) -> Input {
    if !ARMED.load(Ordering::Acquire) { return input(JOINT_SOURCE, &[]); }
    let spans: Option<Vec<InputSpan>> = rows.take(9)
        .map(|(slot, position)| Some(InputSpan { slot, start: u64::try_from(position).ok()?, count: 1 }))
        .collect();
    match spans {
        Some(spans) if !spans.is_empty() && spans.len() <= matrix_rows && matrix_rows <= 8 =>
            input_rows(JOINT_SOURCE, &spans, (matrix_rows - spans.len()) as u32),
        _ => input(JOINT_SOURCE, &[]),
    }
}
```

When unarmed, `input(JOINT_SOURCE, &[])` returns the inactive guard without publishing (`witness.rs:59-62, 111-120`), which is the existing off path. The 8-row cap is `MAX_JOINT_DECODE_ROWS` (`joint_decode_routes.rs:42`) and is referenced by name, not duplicated. It is re-exported in the `pub(crate) use` line (`native_observation.rs:9`).

**(c) Call site: `src/inference/models/qwen35/joint_decode.rs` `encode_joint_decode`**

Insert immediately after `positions` is built (after line 333) and before the layer loop (361):

```rust
let _native_input = crate::inference::native_observation::joint_decode_input(
    rows.iter().map(|row| (row.slot_id.0, row.position)), width);
```

The guard is placed here, not in `run_qwen35_joint_decode`, for four reasons:

- The witness contract says the manifest is made "only where the actual ordered token tensor is constructed" (`witness.rs:57-58`). The tensor is built here, from `rows` and `width`.
- All refusals happen first (`forward_gpu_decode_joint`, lines 213-257), so a refused step never publishes an input.
- The guard drops when `encode_joint_decode` returns, and its `forward_input_end` marker closes the forward there. That is before `finish_joint_tick` (`qwen35_joint_decode.rs:372-380`), whose MTP catch-up and sampling run under the same invocation epoch: `_native_invocation` lives until the end of the `if !prepared.is_empty()` block, and the encoder adapter tags their commands with the joint `EPOCH`. Those commands are reported as the epoch's settle part, never as forward time. MTP uses `rms_norm_with_weight` (`mtp.rs:603-612`), not the witnessed wrapper, so it emits no unmapped records.
- The joint path runs synchronously on the owning worker thread (`forward_joint_decode` → `forward_gpu_decode_joint` inside `GPU_CACHE.with`, `engine_qwen35/joint_decode.rs:170-198`, `joint_decode.rs:260-291`), so the thread-local `SELECTED`/`EPOCH` from `trace_transaction` are visible. A unit test pins this (§5, R4).

The same call covers Qwen3.6 MoE joint decode, which is on by default since 2026-11-05 (#719). It uses the same `encode_joint_decode` and the same DeltaNet pre-norm.

**(d) Why layer 0 is the witness.** With `full_attention_interval = 4`, layer 0 is DeltaNet, and its input `x` is exactly the `[width, h]` F32 embedding buffer the manifest describes. If a future artifact had no DeltaNet layer, the input would have no witness, and the reducer rejects that (`f2_native_trace.py:334-336`, "required forward input replaced by auxiliary/missing/duplicate witness"), which is fail closed.

### 3.4 Reducer (`scripts/four_agent/f2_native_trace.py`, `f2_trace.py`)

**forward_input branch (131-178).** For `p["source"] == JOINT_SOURCE`:

- The key set is the existing set plus `padding_rows`. For all other sources the key set is unchanged.
- `role == "target"`, and every logical row has `phase == "ordinary"` and `token_count == 1`.
- `1 <= len(rows) <= 8`, and `(slot_id, generation)` values are unique. The existing independent branch (171-177) gets a `token_count == 1` clause for the joint source.
- `number(padding_rows)`, and `matrix_rows == offset + padding_rows`. This replaces `matrix_rows == offset` (161-162) for the joint source only.
- If `padding_rows > 0`, then `matrix_rows == 4` (`JOINT_PADDED_WIDTH`, which mirrors `PADDED_MV_EXT_WIDTH`), and in every case `matrix_rows <= 8`.
- Any other source with a `padding_rows` key fails the closed-set check.

**witness check (198-227).** No change. `p["rows"] == source["matrix_rows"]` already binds the physical width, and the footprint check binds `rows × columns × 4` bytes.

**Transaction binding (new).** `f2_trace.validate_trace` already validates `transaction` (189-198). It additionally calls `native.transaction(payload)` when `native` is configured. `qualify_window`'s parser path (240-250) does the same. `NativeTrace.transaction` then works as follows:

- On any transaction: if a pending joint transaction is still unbound, add 1 to `joint_unwitnessed`. If a bound joint input is still awaiting its epoch confirmation or its `forward_input_end`, raise. Set `pending = lanes` if `operation == "joint_ordinary_decode"`, otherwise `None`, and remember `self.current_op = operation`.
- On a joint-source `forward_input`, require all of:
  - `pending` is set and `[(request_id, slot_id, generation) of packed logical rows] == pending lanes`, in order;
  - freshness, checked at input time: `p["epoch"]` is strictly greater than the epoch of every earlier bound joint input and differs from every earlier `forward_input` epoch.

  Then bind it, add 1 to `joint_witnessed`, clear `pending`, and set `self.awaiting_epoch = (forward_id, epoch)`. A second joint input inside one transaction, one outside any joint transaction, or a stale epoch raises.
- Epoch consistency, checked at the first `scope_definition` after a bound joint input: while `self.awaiting_epoch` is set, the next `scope_definition` (in the real order this is the auxiliary scope of the first encoder created after the input, ahead of the Known witness scope) must have `scope.epoch == ` the bound input's epoch; then clear `self.awaiting_epoch`. A mismatch raises. Reaching `forward_input_end` or the next `transaction` with `self.awaiting_epoch` still set also raises. The rule is: transaction epoch == epoch of the first scope created after the input. It is not tied to scopes published between the `transaction` and the input, because whether any exist depends on work done before `encode_joint_decode`.
- On `forward_input_end`: require that it is the first marker for the most recently bound joint input (same `forward_id` and `epoch`), that the epoch-consistency check has completed, that it arrives before the next `transaction`, and that `scope_watermark >= ` that forward's witness scope id. A joint input with no marker by the next transaction, or by `finish`, raises. A marker under a policy without the joint source raises.
- On `forward_witness_unmapped`: add 1 to `unmapped_by_operation[self.current_op]` as well as the existing `self.unmapped`.
- At `finish`: a pending unbound joint transaction counts as unwitnessed.

Ordering is a valid join here: `transaction`, `forward_input` and `forward_input_end` are all published by the owning worker thread inside one transaction (`scheduler.rs:1449-1483`, then the forward), and `publish` appends under one mutex. Records from other threads do not include these kinds. The epoch check makes a misbinding detectable rather than silent.

**Per-command retention (joint policy only).** At `Dropped` (317-322) the reducer keeps, in addition to today's epoch and submit/drain instants, the command's first-dispatch scope id and its `Created.begin_monotonic_ns`, so `finish` can split each joint epoch by the marker. That is two integers per command. Without the joint policy nothing extra is kept.

**Isolation from Q2 v2 (required by the review).** Joint-source witnesses go through every existing validity check (`witness`, the one-encode rule, the joined-epoch obligation at 347-359), but `finish` does not append them to `submissions`. They appear only under `joint_ordinary_decode`. As a result:

- `f2_trace.qualify_window` service brackets (`f2_trace.py:266-270`), its boundary set (311-313) and `witness_progress` (348-355) see exactly what they see today, so every `forward_demand_v2` window verdict, including `slots_without_required_forward_witness`, is unchanged;
- `operator_shared_batch4_observed` (`f2_native_trace.py:376-377`) is computed over `submissions` and therefore does not flip on four-lane joint forwards;
- `need(submissions, …)` (371) still requires at least one non-joint witness, which every Qwen run has (prefill and serial decode).

Letting Q2 consume joint witnesses is exactly the v0.1.31 Q2 v3 decision (ADR-063:1443).

**Report (370-377).** When `JOINT_SOURCE` is in the policy, the report adds the following. Old-policy reports stay byte-identical, so retained `windows*` re-freezes stay reproducible, as QE §7 did.

```json
"joint_ordinary_decode": {
  "transactions": 896, "witnessed": 896, "unwitnessed": 0,
  "unmapped_attempts": 0, "boundary_ambiguous_commands": 0, "pre_forward_commands": 0,
  "width_histogram": {"logical4_pad0": 770, "logical3_pad1": 72, "logical2_pad2": 54, "logical1_pad3": 0},
  "forwards": [{"epoch": 561, "forward_id": 9001, "lanes": [[0,0,0,1834],[1,1,0,2210]],
                "logical_rows": 2, "padding_rows": 2, "witness_command_id": 1201,
                "forward": {"commands": 17, "dispatches": 1873,
                            "first_submit_ns": 9464211000, "last_checked_return_ns": 9494480000},
                "settle":  {"commands": 1, "dispatches": 3,
                            "first_submit_ns": 9494530000, "last_checked_return_ns": 9494610000},
                "epoch_span_ns": [9464211000, 9494610000]}]},
"unmapped_by_operation": {"joint_ordinary_decode": 0}
```

The values above are illustrative except the histogram buckets, which use the n4-fix counts as the expected shape. Histogram keys are `logical<N>_pad<M>`, so they cannot be confused with the Q3 population names (`3+1` = 3 prefill + 1 decode in `q3_campaign.WORKLOADS`). `logical1_pad3` exists because a joint forward can carry one logical row: the guard is `!prepared.is_empty()` (`run_qwen35_joint_group` in `qwen35_joint_decode.rs`), and one lane's `prepare_joint_tick` can fail. In `Exact` mode the keys are `logical<N>_pad0`.

Since #717 (padded-mode solo route, `HF2Q_QWEN_JOINT_SOLO`, default on), a lone ordinary lane runs a one-lane padded joint step instead of a serial forward. Lanes beyond the joint width run further joint steps. `logical1_pad3` is therefore a normal bucket in Qwen3.8 N4 runs, and the n4-fix value of 0 no longer describes the expected shape.

Those steps are now `joint_ordinary_decode` transactions rather than `ordinary_or_temporal_mtp_decode`. With the joint input manifest, each one publishes a witnessed `forward_input` and one joint witness, and no unmapped records. Joint witnesses stay out of `submissions` (§3.4), so the Q2 `witness_progress` check in `scripts/four_agent/f2_trace.py` gets no witness from these steps either. A window in which a slot advanced only through lone steps can then report it under `slots_without_required_forward_witness`. Letting Q2 count joint witnesses is the Q2 v3 rule (successor scope). `HF2Q_QWEN_JOINT_SOLO=0` restores the old transaction mix for trace runs that need it.

Field meaning:

- `forward` covers only commands created and encoded between the joint `forward_input` and its `forward_input_end`. This is the per-forward time Q4-QWEN38 needs.
- `settle` covers the same epoch's later commands: per-lane sampling, and MTP catch-up on coherent lanes. It is reported separately and is never mixed into forward time.
- `epoch_span_ns` is the whole transaction epoch, for E4's turn intervals.

`forwards[*]` is computed in `finish` from the per-epoch commands it already groups (338-343), plus the per-command retention above. The cost is one small dict per joint forward, about 900 per run.

The acceptance predicate is `unmapped_operator_attempts == 0 and joint.unwitnessed == 0 and joint.witnessed == joint.transactions > 0`. It is evaluated by the validation script (§5.2), not hard-coded into the reducer, so diagnostic traces of other shapes still reduce.

### 3.5 Owner mapping

- The owner identity is `(request_id, slot_id, generation)`, taken from the scheduler's `in_flight` entry for each transaction handle (`scheduler.rs:1453-1462`).
- The input re-checks this identity at manifest time against live `OWNERS` (`resolve_selected`, `native_observation.rs:496-513`, which refuses recycled generations or inactive owners).
- The reducer re-checks it against the latest `slots` record (`f2_native_trace.py:80-81`).
- Physical row `i` < `logical_rows` is owned by `packed_rows[i].logical`. Padding rows are owned by the transaction as shared overhead.
- For attribution, a joint forward's `forward` time belongs jointly to its owner set. Splitting it per lane (equal share, or by context length) is a Q4-QWEN38/E1 analysis choice, not a trace fact. The trace records `width_histogram` and `padding_rows` so that the padding cost can be stated separately.
- `settle` commands run per lane, but the trace does not record which lane each belongs to. They are attributed to the owner set as settle time, and per-lane settle attribution is left to E1's category split.

### 3.6 Deliberately unchanged

- `scripts/four_agent/f2_demand.py:22-25` and `src/serve/phase_trace.rs:210-213` (queued-demand diagnostics accept only Qwen greedy/forward witnesses). Admitting the joint source there is the Q2 v3 counting rule's decision (ADR-063:1443), so it stays in v0.1.31. Without it, a queued-demand config that lists the joint source is refused (fail closed), which is the intended interim state.
- `f2_trace.qualify_window` and `q2_certify` inputs: unchanged by construction, because joint witnesses stay out of `submissions` (§3.4).
- `f2_native_trace.finish` obligation logic (347-359). A joint epoch now has one owner set, so every command of the epoch that carries work must be submitted and checked-drained. The n4-fix sample epoch already satisfies this: all 18 commands, including the `commit_labeled` layer chains (`joint_decode.rs:505-516`), have `CheckedDrain`. The validation run confirms it across all 896.

### 3.7 Overhead model (stated before measurement)

- **Trace off (default):** two Acquire loads per joint forward (the adapter's and `input()`'s). No allocation, no publish, no metadata. The `Drop` is inactive.
- **Trace on, per joint forward:** the hook publishes 4 records (`forward_input`, the Known `scope_definition`, `forward_witness`, `forward_input_end`) plus one extra auxiliary `scope_definition` after the witness dispatch (`witness.rs:274`). It also removes 48 `forward_witness_unmapped` publishes. Each publish takes the trace mutex and serializes JSON. The net trace record count goes down by about 43 per forward, roughly 38.5k fewer records per n4 run. Native event counts are unchanged (the witness re-labels one dispatch's scope), so the `event_limit` and `metadata_limit_bytes` headroom (2.57M of 8M events in n4-fix) is unaffected.

## 4. Feeding Q4-QWEN38, E1 and E4

- **Q4-QWEN38 (ADR-063:596, "with the Q4-TRACE joint-decode witness"):** `joint_ordinary_decode.forwards[*]` gives, per joint forward, the owner set, fed positions, logical and padded width, and separate `forward` and `settle` native spans with command and dispatch counts. The stage attribution can then assign joint forward GPU time to slots and turns, keep MTP catch-up and sampling apart from it, and separate padding cost and width mix from per-token rate. That is the 2.0× per-output-token gap in the Q3 scoreboard.
- **ADR-065 E1 (stage split):** E1's category-separated command buffers run inside the same invocation epoch. The native encoder adapter tags every command with `EPOCH` (`native_observation.rs:517-539`), so E1 can join its per-class times to the witnessed owner set by epoch, and to the forward or settle part by the same scope-watermark rule. E1 can also check distortion by comparing the sum of its forward classes against the witnessed `forward` span from the same traced run. Per the ADR, any class the join cannot attribute stays Unknown.
- **ADR-065 E4 (makespan composition):** `epoch_span_ns` gives native-clock intervals for every joint turn, already joined to lanes, which E4's critical-path extractor can use alongside the `LOG_FILTER` per-turn lines. `width_histogram` is E4's "joint-width distribution per run" for traced runs. The trace-on distortion E4 must state is this story's measured overhead (§5.3).
- **Q2 v3 (v0.1.31):** it may consume the same records. Nothing in v0.1.30 changes any counting rule or any Q2 v2 input: joint witnesses stay out of `submissions` (§3.4, test P5).

## 5. Tests and validation

### 5.1 Unit and synthetic tests (no GPU)

Rust tests live in `native_observation` and `witness.rs` `#[cfg(test)]`, next to the existing tests at `witness.rs:282-329` and `native_observation.rs:582-643`:

- **R1:** `source_layout(JOINT_SOURCE, …)` accepts 1–8 unique one-token spans. It rejects a duplicate slot, `count == 2`, 9 spans and the empty list. Existing sources' verdicts are unchanged.
- **R2:** the padding rule. Refuse `padding_rows > 0` for every non-joint source. Accept joint `3 + 1`. `footprint(total = 4, rows = 4, cols, …)` passes where the logical-only `total = 3` fails. That last case documents the §1.1 trap.
- **R3:** unarmed `joint_decode_input` returns an inactive guard, leaves `INPUT` as `None`, and never polls its iterator. The test uses an iterator that panics if polled.
- **R4:** the manifest resolves owners from `SELECTED` set by an `Invocation` on the same thread. A recycled generation in `OWNERS` makes it refuse (extends `native_observation.rs:586-624`).
- **R5:** a claimed descriptor makes later `prepare` calls silent. This is tested at the pure-function level by factoring the claim and unmapped decision into a small helper.
- **R6:** the end-marker decision: an active joint guard yields exactly one marker decision on drop, including on an early return and during unwinding; an inactive guard or a non-joint source yields none. Tested through a small pure helper that `Drop` calls.

Python tests extend `scripts/four_agent/test_f2_native_trace.py`, which already has `fixture`, `four_lanes` and `with_busy_owners`:

- **P1, positive:** `logical4_pad0`, `logical3_pad1`, `logical2_pad2` and `logical1_pad3` joint forwards bound to `joint_ordinary_decode` transactions, each with its `forward_input_end`, plus settle commands after the watermark. The fixture uses the real record order: `transaction`, `slots` (`owning_worker_enter_transaction`), joint `forward_input`, the first encoder's auxiliary `scope_definition` (same epoch), the Known `scope_definition` and `forward_witness`, native events, `forward_input_end`, then settle scopes and commands. A second fixture variant also puts auxiliary scopes of the same epoch between the `slots` record and the input, and it must reduce identically. Expect `unmapped_operator_attempts == 0`, `witnessed == transactions`, the histogram buckets, and correct `forward`/`settle` command splits, including a dispatch-free command on each side of `end_monotonic_ns` and one exactly at it (counted ambiguous).
- **P2, negatives (each raises):** `padding_rows` on `qwen35.forward.actual_tokens`; `padding_rows > 0` with `matrix_rows != 4`; `matrix_rows != offset + padding_rows`; joint `token_count == 2`; joint `role == target_verify`; joint row `phase == prefill`; duplicate slot; lanes differing from the transaction (order or generation); two joint inputs in one transaction; a joint input with no open joint transaction; witness `rows` equal to logical rather than physical rows; a reused (stale) joint epoch at input time; a first post-input `scope_definition` whose epoch differs from the input's; a `forward_input_end` or next `transaction` arriving before any post-input `scope_definition`; a missing, duplicate, misplaced or mismatched `forward_input_end`; a watermark below the witness scope id; a `forward_input_end` under a policy without the joint source.
- **P3, coverage:** a joint transaction with no input reduces, but reports `unwitnessed == 1`, and the acceptance script fails it. An unmapped record inside a joint transaction is attributed to `unmapped_by_operation["joint_ordinary_decode"]`.
- **P4, compatibility:** every existing test passes unchanged. Re-freezing the retained n4-fix run under the old policy yields `windows` output byte-identical to `windows-epoch-v1` (the QE §7 method; read-only, CPU).
- **P5, Q2 v2 isolation (required by the review):** take a synthetic traced cohort with serial and joint forwards (extending `four_lanes`/`with_busy_owners` and the `test_f2_demand` queued fixtures). Compute every `f2_trace.qualify_window` result for every composition and population policy, calling `qualify_window` directly on the record lists rather than through `q2_traced_freeze`, for two traces:
  - (a) the trace with joint records, under a header policy that names the joint source;
  - (b) the same trace transformed into the pre-hook shape. Every joint `forward_input`, `forward_witness` and `forward_input_end` is dropped. Each joint witness's Known `scope_definition` is kept, because native `DispatchEncoded` events reference its scope id, but it is relabelled as `OpaqueAuxiliary` with `operation: "opaque_auxiliary"` and `rows: []`. The joint source is removed from the header's `native_observer.witness_manifest.input_sources`.

  The window outputs must be identical, and `operator_shared_batch4_observed` must not change. The same comparison, with the same transform, is run read-only on the hardware run T1 (§5.2 check 2).
- `test_f2_trace.py`, `test_f2_demand.py`, `test_q2_certify.py`, `test_q2_traced_run.py` (updated WITNESS list) and `test_f2_metadata_budget.py` pass.

GPU test (needs the GPU lane; added next to `real_artifact_joint_decode_invariance_tests.rs`, which already compares joint rows bitwise across cohorts):

- **G1, hook bitwise neutrality:** `qwen38_joint_decode_witness_armed_rows_are_bitwise_unarmed` (`#[ignore]`, `HF2Q_QWEN38_GGUF`, `HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1`, run with `--exact`). For cohorts `logical4_pad0`, `logical3_pad1`, `logical2_pad2` and `logical1_pad3` on identical caches, it runs the joint forward once unarmed and once with the observer armed through a `#[cfg(test)]` arm-and-owners guard. Logits, post-norm hidden, cursors, KV codes and scales and recurrent bytes must be bitwise equal. The guard sets and restores the process-global `ARMED`, so the test is run alone with `--exact`. This proves the hook changes no output on a fixed schedule, independently of four-agent scheduling.

Commands to run once all edits land (root verifies):

```
cargo test --lib native_observation
cargo test --lib qwen35::joint_decode
# GPU lane only:
HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1 HF2Q_QWEN38_GGUF=<official artifact> cargo test --release --bin hf2q \
  qwen38_joint_decode_witness_armed_rows_are_bitwise_unarmed -- --ignored --exact
cd scripts/four_agent && python3 -m unittest test_f2_native_trace test_f2_trace test_f2_demand \
  test_q2_traced_run test_q2_certify test_f2_metadata_budget
```

Run these with `CARGO_TARGET_DIR` per agent, `RUSTC_WRAPPER=`, `CARGO_INCREMENTAL=0`, and the HOST-LOCK wait.

### 5.2 Hardware validation run (only after "GPU LANE GRANTED"; HOST-LOCK held)

- Binary H: the Q4-TRACE commit, root-built.
- Binary P: its parent on main, same mlx-native 0.15.16.
- Case: `scripts/four_agent/q5_cases/qwen38.json`.
- Artifact: the official Qwen3.8-27B GGUF.
- Every run records source commit, binary sha256, artifact digest, native version and case file, and passes Q5 CalibrationPass (`q2_traced_run.py` and the campaign runner already enforce this).
- Evidence goes to `/Users/robert/.local/state/hf2q/v0130/q4-trace/`.

| Run | Binary | Trace | Purpose |
|---|---|---|---|
| T1 | H | on (`q2_traced_run.py --population n4 --binary H …`) | Acceptance: reducer report |
| O1 | H | off (one `q5_calibration.py` run of the same case, `HF2Q_Q5_HF2Q_BIN=H`) | Exactness control vs T1; off-default timing |
| T0 | P | on (`q2_traced_run.py --population n4 --binary P …`) | Hook overhead under trace (paired with T1) |
| O0 | P | off (as O1, `HF2Q_Q5_HF2Q_BIN=P`) | Off-default control vs O1; observer-only exactness control vs T0 |
| O1′ | H | off (as O1, run right after O0) | Determinism control: untraced hook binary run to run (amendment 1) |

The untraced runs are single supervised Q5 runs, not a `q3_campaign.py` arm: `q3_campaign.py` always plans the stock and experiment arms in blocks with warmups. They are bound to their T runs afterwards by binary, artifact and case sha256 (operator-accepted runbook on #677). P is H's exact merge-base on main. Each arm runs from its own harness, and both harnesses must contain the #704 tool-result normalization.

Acceptance checks:

1. `q2_traced_freeze.py windows --run T1 --out … --population-policy forward_demand_v2` (the QE §7 command) produces a report with `unmapped_operator_attempts == 0`, `unmapped_by_operation.joint_ordinary_decode == 0`, and `joint_ordinary_decode.unwitnessed == 0` with `witnessed == transactions > 0`. Every joint epoch's commands are checked-drained (the reducer's obligation join passes). `boundary_ambiguous_commands` is reported.
2. Q2 v2 isolation on hardware: `f2_trace.qualify_window`, called directly, gives identical outputs on T1 and on T1 transformed by P5(b)'s method (joint `forward_input`/`forward_witness`/`forward_input_end` dropped, joint Known scopes relabelled `OpaqueAuxiliary`, joint source removed from the header policy). Read-only CPU.
3. Exactness, the transcript comparisons through `python3 scripts/four_agent/q1_decision_policy.py transcripts`, all reported:
   - `T1 O1`: the bound criterion, hook binary trace on vs off;
   - `T0 O0`: observer-only control, parent binary trace on vs off;
   - `O1 O1′`: determinism control, the hook binary untraced, run to run (amendment 1);
   - `T1 T0`: hook vs no hook, both traced;
   - `O1 O0`: hook binary vs parent, both untraced.
4. G1 (§5.1) passes on the same binary H.

**Pre-registered exactness rule (amendment 1).** The amendment was fixed before any run and accepted by Main on 2026-11-05. It applies with G1 as the hook test. One attempt per run, with the single repeat allowed below. Rows are evaluated top to bottom, and the first match decides.

| T1 vs O1 | G1 | O1 vs O1′ | T0 vs O0 | Classification | Outcome |
|---|---|---|---|---|---|
| any | fail | any | any | `HookAttributable` | Story fails; fix the hook. G1 proves on a fixed schedule whether the hook changes output bytes, so a G1 failure fails the story even if the four-agent transcripts happen to agree. |
| Exact | pass | any | any | `Exact` | Exactness criterion met. |
| Diverged | pass | Diverged | any | `ScheduleNondeterministic` | The hook is cleared by G1. Untraced serving of the same binary is not run-to-run transcript-deterministic, so a four-agent transcript divergence cannot be attributed to the hook. This goes to the operator with all comparisons, the `divergent[]` turns, T1's joint/serial transaction mix, and the recommended ADR wording ("bitwise neutral on a fixed schedule (G1); four-agent transcripts not run-to-run deterministic (O1 vs O1′); observer-induced schedule divergence disclosed"). The operator's decision is recorded in ADR-063 before the story closes. |
| Diverged | pass | Exact | Diverged | `ObserverAttributable` | The hook is cleared: it is bitwise-neutral on a fixed schedule, and the observer alone also diverges. This goes to the operator as in the previous row. |
| Diverged | pass | Exact | Exact | `Unresolved` | Run the repeat with identical inputs: T1′ and O1″ (hook binary) and T0′ and O0′ (parent). If `T1′ O1″` is Exact, classify `ScheduleNondeterministic`. If `T1′ O1″` diverges and `T0′ O0′` also diverges, classify `ObserverAttributable`. Both of these go to the operator. Only if `T1′ O1″` diverges while `T0′ O0′` is Exact, classify `HookAttributable`, and the story fails. That case is two hook-pair divergences, each next to an Exact observer-only pair, with O1 vs O1′ Exact on one pair. |

The rule exists so the story cannot stall without a decision. Padded joint versus serial is a Q1 Class B route (`joint_decode_routes.rs:14-21`, `docs/research/adr063-s3-qwen-joint-decode.md:87-91`), and whether a turn runs joint or serial depends on arrival timing (n4-fix has 82 serial one-lane decode transactions beside the 896 joint ones). An observer-induced schedule shift can therefore change transcripts without any fault in the hook. Dropped turns or failed runs are not retried silently; they are handled by Q5's run validity rules.

**Why amendment 1.** The Qwen38DetInv investigation (2026-11-05) found that Qwen3.8 N4 serving is not run-to-run transcript-deterministic, because its decode route switches serial↔joint with arrival timing. Under revision 3 the `Unresolved` row would have failed the story on two timing-driven divergences in a row (a chance-Exact `T0 O0` leads straight into that row), even with G1 showing the hook bitwise neutral. The O1′ control measures run-to-run divergence without hook or observer, and a cross-run divergence counts as hook evidence only once that control is Exact. Tooling: `q4_trace_verdict.py classify --o1-o1d` (schema `hf2q.q4-trace.exactness.v2`).

Review677 (2026-11-05) required the repeat to include the observer-only pair T0′ O0′. Without it, a chance-Exact O1 O1′ and T0 O0 followed by a timing-driven T1′ O1″ divergence would have failed the story with G1 passing. With it, `HookAttributable` from transcripts needs two hook-pair divergences, each next to an Exact observer-only pair. The tooling takes the pair as `classify --repeat T1′O1″ --repeat-observer T0′O0′`, both or neither.

### 5.3 Overhead measurement (stated in the PR and the evidence README)

- **Trace-on hook overhead (T1 vs T0):** per-joint-turn native epoch span (first `Created` begin → last `CheckedDrain` end of the epoch), stratified by logical width. The epoch span is used because T0 has no marker; T1's `forward`/`settle` split is reported in addition. Report the median and p95 delta with a bootstrap 95% CI over forwards, plus the makespan delta (one run per arm, descriptive).
  - T0 has no joint `forward_input`, so its joint epochs are identified by order: the first `scope_definition` after a `joint_ordinary_decode` transaction carries the epoch, as seen for epoch 561 in n4-fix.
  - A small streaming extractor, `scripts/four_agent/q4_trace_overhead.py` with tests, computes both arms with one rule.
- **Trace-on record delta:** record counts by kind and trace bytes, T1 vs T0. Expected: about −43 records per joint forward (§3.7).
- **Off-default (O1 vs O0):** makespan and per-token rate from the client receipts, descriptive. The cost is bounded by construction at two atomic loads per joint forward. The claim is "within run-to-run noise", never "zero" unless measured as such.
- **Total observer cost (T1 vs O1):** reported for E4's distortion statement. It is not part of this story's acceptance.

## 6. Files touched (hf2q only)

- `src/inference/native_observation.rs`: the `joint_decode_input` adapter and re-export.
- `src/inference/native_observation/witness.rs`: `input_rows` with padding, the source layout arm, the `forward_input_end` marker in `Drop for Input`, and tests.
- `src/inference/models/qwen35/real_artifact_joint_decode_invariance_tests.rs` (or a sibling file): the G1 GPU test and its `#[cfg(test)]` arm guard.
- `src/inference/models/qwen35/joint_decode.rs`: one call in `encode_joint_decode`.
- `src/serve/phase_trace/native.rs`: the source in the manifest allowlist, and a test.
- `scripts/four_agent/f2_native_trace.py`, `f2_trace.py`, `q2_traced_run.py`, the new `q4_trace_overhead.py`, and their tests.
- `scripts/four_agent/README.md`: the traced-run witness note.
- `docs/adr/ADR-063-…md`: the Q4-TRACE status line after validation.
- `CHANGELOG.md`: a diagnostics entry.

No mlx-native change and no serving-default change.

## 7. Design questions and risks

Answers to the three operator questions, recorded as recommended by the independent review (`ReviewTraceDesign`, 2026-11-03). They become binding with the operator's design approval.

1. **New report keys only under the new policy: approve.** Byte-identical re-freezes of retained Q2 outputs are worth the policy-conditional output. The review's condition, that joint witnesses must also stay out of the v2 consumers, is met by §3.4 "Isolation from Q2 v2" and test P5.
2. **Padding declared as owner-less physical rows: approve.** It is the only option consistent with `footprint`, which binds the physical read of the RMS dispatch. A prefix view would either fail `footprint` or misstate `input_read_bytes`.
3. **Transaction binding by record order: approve for v0.1.30, with the epoch freshness and consistency check** (§3.4). An epoch field on `transaction` is deferred to Q2 v3. It would not have to break retained traces, since it could be an optional key accepted only under the joint policy like `padding_rows`; it is deferred because the order binding plus the epoch check already makes a misbinding fail loudly, and the Q2 v3 counting rule is the consumer that would need the stronger form.

Remaining risks:

4. **Exactness:** observer-on timing can change joint composition and joint-versus-serial routing. Under `padded_mv_ext`, row bytes are width-invariant (`joint_decode_routes.rs:14-20`), but joint versus serial is Class B. The four-comparison control, G1 and the pre-registered rule (§5.2) decide every outcome.
5. **Settle attribution is per owner set, not per lane** (§3.5). Per-lane settle time belongs to E1's category split.
6. **Joint verify (ADR-065 J-series):** it will need its own role and source extension. This design rejects `target_verify` on the joint source until then.

## 8. Review resolution (revision 2)

| Review finding | Resolution |
|---|---|
| 1 [P1] Joint witnesses would enter the Q2 `forward_demand_v2` counting through `submissions` and flip `operator_shared_batch4_observed` | Joint witnesses are validated but excluded from `submissions`; reported only under `joint_ordinary_decode` (§3.4). Test P5 plus hardware check 2 prove window outputs are unchanged. |
| 2 [P2] Spans covered the whole transaction epoch, including per-lane MTP catch-up | Boundary closed with `forward_input_end` (scope watermark plus monotonic end) at guard drop, joint policy only. Commands are split into `forward` and `settle`; `epoch_span_ns` is named as such (§3.2, §3.4). |
| 3 [P2] No observer-only exactness baseline | `T0 O0` added, with all four comparisons, the G1 bitwise hook test and a pre-registered classification rule (§5.2). |
| 4 [P3] Histogram keys collided with population names; no 1-row case | Keys are `logical<N>_pad<M>`; the `logical1_pad3` case is in the report and in P1 and G1. |
| 5 [P3] Order binding could be tightened | Freshness is checked when the joint input arrives. Epoch equality is checked against the first `scope_definition` after the bound input, and it must arrive before `forward_input_end` and before the next transaction (§3.4), with P1 real-order fixtures and P2 negatives. §7.3 reworded: the epoch-on-transaction field is deferred, not impossible. |

Closure-check fixes (revision 3):

| Fix | Resolution |
|---|---|
| A [P1] The epoch-equality anchor was the first scope after the *transaction*, which can precede the input | Equality moved to the first `scope_definition` after the bound joint input, required before `forward_input_end` and the next transaction. Freshness stays at input time. P1 uses the real order (`transaction`, `slots`, `forward_input`, auxiliary scope), plus a variant with pre-input auxiliary scopes (§3.4, §5.1). |
| B [P2] The P5(b) stripped trace would fail to reduce once Known scopes were removed | P5(b) and hardware check 2 keep each joint Known `scope_definition`, relabelled `OpaqueAuxiliary` (`opaque_auxiliary`, `rows: []`). They drop only the joint `forward_input`, `forward_witness` and `forward_input_end`, remove the joint source from the header policy, and call `qualify_window` directly (§5.1, §5.2). |
| C [P3] A G1 failure with exact transcripts was unclassified | First rule row is `any \| fail \| any → HookAttributable`. Rows are evaluated top to bottom (§5.2). |
