# ADR-063 Q4-TRACE (#663): Qwen joint-decode forward witness — HLD/LLD

- **Status:** Design, awaiting operator approval (ADR-063 "Execution entry": each implementation story needs a recorded design approval).
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
- reduced by `f2_native_trace.py` with `unmapped_operator_attempts == 0` and a positive coverage count: witnessed joint forwards == joint transactions.

Non-goals:

- The Q2 v3 `queued_forward_demand_v3` counting rule, and the `f2_demand.py` and `phase_trace.rs` queued-demand source allowlists. These are successor scope (ADR-063:1443); see §3.6.
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
```

The reducer gains three things:

- the new source, with its layout and padding rules;
- a transaction→input binding for `joint_ordinary_decode`;
- report fields Q4-QWEN38, E1 and E4 can consume.

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

### 3.3 Server wiring (hf2q)

**(a) `native_observation/witness.rs`**

1. `input(source, spans)` becomes a thin wrapper over a private `input_rows(source, spans, padding_rows: u32)`.
2. In `input_rows`, after `resolve_selected`, set `total = Σ logical token_count + padding_rows` using checked arithmetic.
3. Refuse `padding_rows != 0` unless `source == JOINT_SOURCE` (that is, emit nothing and invalidate via `descriptor = None`).
4. Emit `"padding_rows"` only for `JOINT_SOURCE`.
5. `Descriptor` is unchanged. `total` is now the physical row count, which is exactly what `footprint` compares against the RMS matrix.

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
- The guard drops when `encode_joint_decode` returns. That is before `finish_joint_tick` (`qwen35_joint_decode.rs:372-380`), whose MTP catch-up and sampling run under the same invocation epoch but outside this forward's input. MTP uses `rms_norm_with_weight` (`mtp.rs:603-612`), not the witnessed wrapper, so nothing there emits unmapped records today either.
- The joint path runs synchronously on the owning worker thread (`forward_joint_decode` → `forward_gpu_decode_joint` inside `GPU_CACHE.with`, `engine_qwen35/joint_decode.rs:170-198`, `joint_decode.rs:260-291`), so the thread-local `SELECTED`/`EPOCH` from `trace_transaction` are visible. A unit test pins this (§5, R4).

The same call covers Qwen3.6 MoE joint decode, which is opt-in. It uses the same `encode_joint_decode` and the same DeltaNet pre-norm.

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

- On any transaction: if a pending joint transaction is still unbound, add 1 to `joint_unwitnessed`. Set `pending = lanes` if `operation == "joint_ordinary_decode"`, otherwise `None`, and remember `self.current_op = operation`.
- On a joint-source `forward_input`: require that `pending` is set and that `[(request_id, slot_id, generation) of packed logical rows] == pending lanes` in order. Bind it, add 1 to `joint_witnessed`, and clear `pending`. A second joint input inside one transaction, or one outside any joint transaction, raises.
- On `forward_witness_unmapped`: add 1 to `unmapped_by_operation[self.current_op]` as well as the existing `self.unmapped`.
- At `finish`: a pending unbound joint transaction counts as unwitnessed.

Ordering is a valid join here: `transaction` and `forward_input` are both published by the owning worker thread inside one transaction (`scheduler.rs:1449-1483`, then the forward). Records from other threads do not include these kinds.

**Report (370-377).** When `JOINT_SOURCE` is in the policy, the report adds the following. Old-policy reports stay byte-identical, so retained `windows*` re-freezes stay reproducible, as QE §7 did.

```json
"joint_ordinary_decode": {
  "transactions": 896, "witnessed": 896, "unwitnessed": 0,
  "unmapped_attempts": 0,
  "width_histogram": {"4+0": 770, "3+1": 72, "2+2": 54},
  "forwards": [{"epoch": 561, "forward_id": 9001, "lanes": [[0,0,0,1834],[1,1,0,2210]],
                "logical_rows": 2, "padding_rows": 2, "commands": 18, "dispatches": 1876,
                "first_submit_ns": 9464211000, "last_checked_return_ns": 9494480000,
                "witness_command_id": 1201}]},
"unmapped_by_operation": {"joint_ordinary_decode": 0}
```

The values above are illustrative except the histogram buckets, which use the n4-fix counts as the expected shape.

`forwards[*]` is computed in `finish` from the per-epoch commands it already groups (338-343). It uses the retained `submit` and `drain` instants and the dispatch counts (317-322), so it adds no new per-event retention. The cost is one small dict per joint forward, about 900 per run.

The acceptance predicate is `unmapped_operator_attempts == 0 and joint.unwitnessed == 0 and joint.witnessed == joint.transactions > 0`. It is evaluated by the validation script (§5.2), not hard-coded into the reducer, so diagnostic traces of other shapes still reduce.

### 3.5 Owner mapping

- The owner identity is `(request_id, slot_id, generation)`, taken from the scheduler's `in_flight` entry for each transaction handle (`scheduler.rs:1453-1462`).
- The input re-checks this identity at manifest time against live `OWNERS` (`resolve_selected`, `native_observation.rs:496-513`, which refuses recycled generations or inactive owners).
- The reducer re-checks it against the latest `slots` record (`f2_native_trace.py:80-81`).
- Physical row `i` < `logical_rows` is owned by `packed_rows[i].logical`. Padding rows are owned by the transaction as shared overhead.
- For attribution, a joint forward's time belongs jointly to its owner set. Splitting it per lane (equal share, or by context length) is a Q4-QWEN38/E1 analysis choice, not a trace fact. The trace records `width_histogram` and `padding_rows` so that the padding cost can be stated separately.

### 3.6 Deliberately unchanged

- `scripts/four_agent/f2_demand.py:22-25` and `src/serve/phase_trace.rs:210-213` (queued-demand diagnostics accept only Qwen greedy/forward witnesses). Admitting the joint source there is the Q2 v3 counting rule's decision (ADR-063:1443), so it stays in v0.1.31. Without it, a queued-demand config that lists the joint source is refused (fail closed), which is the intended interim state.
- `f2_native_trace.finish` obligation logic (347-359). A joint epoch now has one owner set, so every command of the epoch that carries work must be submitted and checked-drained. The n4-fix sample epoch already satisfies this: all 18 commands, including the `commit_labeled` layer chains (`joint_decode.rs:505-516`), have `CheckedDrain`. The validation run confirms it across all 896.

### 3.7 Overhead model (stated before measurement)

- **Trace off (default):** one Acquire load per joint forward. No allocation, no publish, no metadata.
- **Trace on, per joint forward:** the hook publishes 3 records (`forward_input`, the Known `scope_definition`, `forward_witness`) plus one extra auxiliary `scope_definition` after the witness dispatch (`witness.rs:274`). It also removes 48 `forward_witness_unmapped` publishes. Each publish takes the trace mutex and serializes JSON. The net trace record count goes down by about 44 per forward, roughly 39k fewer records per n4 run. Native event counts are unchanged (the witness re-labels one dispatch's scope), so the `event_limit` and `metadata_limit_bytes` headroom (2.57M of 8M events in n4-fix) is unaffected.

## 4. Feeding Q4-QWEN38, E1 and E4

- **Q4-QWEN38 (ADR-063:596, "with the Q4-TRACE joint-decode witness"):** `joint_ordinary_decode.forwards[*]` gives, per joint forward, the owner set, fed positions, logical and padded width, native span `[first_submit_ns, last_checked_return_ns]`, and command and dispatch counts. The stage attribution can then assign joint GPU time to slots and turns, and separate padding cost and width mix from per-token rate. That is the 2.0× per-output-token gap in the Q3 scoreboard.
- **ADR-065 E1 (stage split):** E1's category-separated command buffers run inside the same invocation epoch. The native encoder adapter tags every command with `EPOCH` (`native_observation.rs:517-539`), so E1 can join its per-class times to the witnessed owner set by epoch. E1 can also check distortion by comparing the sum of its classes against the witnessed epoch span from the same traced run. Per the ADR, any class the join cannot attribute stays Unknown.
- **ADR-065 E4 (makespan composition):** `forwards[*]` gives native-clock intervals for every joint turn, already joined to lanes, which E4's critical-path extractor can use alongside the `LOG_FILTER` per-turn lines. `width_histogram` is E4's "joint-width distribution per run" for traced runs. The trace-on distortion E4 must state is this story's measured overhead (§5.3).
- **Q2 v3 (v0.1.31):** it consumes the same records. Nothing in v0.1.30 changes any counting rule.

## 5. Tests and validation

### 5.1 Unit and synthetic tests (no GPU)

Rust tests live in `native_observation` and `witness.rs` `#[cfg(test)]`, next to the existing tests at `witness.rs:282-329` and `native_observation.rs:582-643`:

- **R1:** `source_layout(JOINT_SOURCE, …)` accepts 1–8 unique one-token spans. It rejects a duplicate slot, `count == 2`, 9 spans and the empty list. Existing sources' verdicts are unchanged.
- **R2:** the padding rule. Refuse `padding_rows > 0` for every non-joint source. Accept joint `3 + 1`. `footprint(total = 4, rows = 4, cols, …)` passes where the logical-only `total = 3` fails. That last case documents the §1.1 trap.
- **R3:** unarmed `joint_decode_input` returns an inactive guard, leaves `INPUT` as `None`, and never polls its iterator. The test uses an iterator that panics if polled.
- **R4:** the manifest resolves owners from `SELECTED` set by an `Invocation` on the same thread. A recycled generation in `OWNERS` makes it refuse (extends `native_observation.rs:586-624`).
- **R5:** a claimed descriptor makes later `prepare` calls silent. This is tested at the pure-function level by factoring the claim and unmapped decision into a small helper.

Python tests extend `scripts/four_agent/test_f2_native_trace.py`, which already has `fixture`, `four_lanes` and `with_busy_owners`:

- **P1, positive:** 4+0, 3+1 and 2+2 joint forwards bound to `joint_ordinary_decode` transactions. Expect `unmapped_operator_attempts == 0`, `witnessed == transactions`, and histogram buckets.
- **P2, negatives (each raises):** `padding_rows` on `qwen35.forward.actual_tokens`; `padding_rows > 0` with `matrix_rows != 4`; `matrix_rows != offset + padding_rows`; joint `token_count == 2`; joint `role == target_verify`; joint row `phase == prefill`; duplicate slot; lanes differing from the transaction (order or generation); two joint inputs in one transaction; a joint input with no open joint transaction; witness `rows` equal to logical rather than physical rows.
- **P3, coverage:** a joint transaction with no input reduces, but reports `unwitnessed == 1`, and the acceptance script fails it. An unmapped record inside a joint transaction is attributed to `unmapped_by_operation["joint_ordinary_decode"]`.
- **P4, compatibility:** every existing test passes unchanged. Re-freezing the retained n4-fix run under the old policy yields `windows` output byte-identical to `windows-epoch-v1` (the QE §7 method; read-only, CPU).
- `test_f2_trace.py`, `test_f2_demand.py`, `test_q2_certify.py`, `test_q2_traced_run.py` (updated WITNESS list) and `test_f2_metadata_budget.py` pass.

Commands to run once all edits land (root verifies):

```
cargo test --lib native_observation
cargo test --lib qwen35::joint_decode
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
| O1 | H | off (same case and population through `q3_campaign.py --family qwen38 --experiment-bin H --workloads n4`, one hf2q arm) | Exactness control vs T1; off-default timing |
| T0 | P | on (`q2_traced_run.py --population n4 --binary P …`) | Hook overhead under trace (paired with T1) |
| O0 | P | off (as O1, `--experiment-bin P`) | Off-default control vs O1 |

The exact `q3_campaign.py` flags for the single off arm are pinned in the implementation PR's runbook from that script's argparse (`q3_campaign.py:462-505`). They are not invented here.

Acceptance checks:

1. `q2_traced_freeze.py windows --run T1 --out … --population-policy forward_demand_v2` (the QE §7 command) produces a report with `unmapped_operator_attempts == 0`, `unmapped_by_operation.joint_ordinary_decode == 0`, and `joint_ordinary_decode.unwitnessed == 0` with `witnessed == transactions > 0`. Every joint epoch's commands are checked-drained (the reducer's obligation join passes).
2. `python3 scripts/four_agent/q1_decision_policy.py transcripts T1 O1` returns `ExactTranscripts`. `transcripts T1 T0` is also reported, to separate the hook from the observer.
3. If either check diverges, the story does not close. The divergent turns (`divergent[]`) are attached, and the cause is investigated: hook versus observer timing affecting scheduling.

### 5.3 Overhead measurement (stated in the PR and the evidence README)

- **Trace-on hook overhead (T1 vs T0):** per-joint-forward native span (first `Created` begin → last `CheckedDrain` end of the epoch), stratified by logical width. Report the median and p95 delta with a bootstrap 95% CI over forwards, plus the makespan delta (one run per arm, descriptive).
  - T0 has no joint `forward_input`, so its joint epochs are identified by order: the first `scope_definition` after a `joint_ordinary_decode` transaction carries the epoch, as seen for epoch 561 in n4-fix.
  - A small streaming extractor, `scripts/four_agent/q4_trace_overhead.py` with tests, computes both arms with one rule.
- **Trace-on record delta:** record counts by kind and trace bytes, T1 vs T0. Expected: about −44 records per joint forward (§3.7).
- **Off-default (O1 vs O0):** makespan and per-token rate from the client receipts, descriptive. The cost is bounded by construction at one atomic load per joint forward. The claim is "within run-to-run noise", never "zero" unless measured as such.
- **Total observer cost (T1 vs O1):** reported for E4's distortion statement. It is not part of this story's acceptance.

## 6. Files touched (hf2q only)

- `src/inference/native_observation.rs`: the `joint_decode_input` adapter and re-export.
- `src/inference/native_observation/witness.rs`: `input_rows` with padding, the source layout arm, and tests.
- `src/inference/models/qwen35/joint_decode.rs`: one call in `encode_joint_decode`.
- `src/serve/phase_trace/native.rs`: the source in the manifest allowlist, and a test.
- `scripts/four_agent/f2_native_trace.py`, `f2_trace.py`, `q2_traced_run.py`, the new `q4_trace_overhead.py`, and their tests.
- `scripts/four_agent/README.md`: the traced-run witness note.
- `docs/adr/ADR-063-…md`: the Q4-TRACE status line after validation.
- `CHANGELOG.md`: a diagnostics entry.

No mlx-native change and no serving-default change.

## 7. Risks and open points for approval

1. **New report keys only under the new policy.** This keeps retained Q2 outputs byte-identical. The cost is that `f2_native_trace` has policy-conditional output. Alternative: always emit them, which breaks byte-identity of old re-freezes. Recommended: conditional.
2. **Padding declared as owner-less physical rows.** The alternative, a prefix-view witness of logical rows only, would hide physical work and mis-state `input_read_bytes`. Recommended: explicit `padding_rows`.
3. **Transaction binding by record order.** This relies on the owning-worker publish order. It is pinned by P2 negatives and by the T1 run. An epoch field on `transaction` would be stronger, but it changes a closed schema that retained traces use. Deferred to Q2 v3 if wanted.
4. **Exactness:** observer-on timing can change joint composition. Under `padded_mv_ext`, row bytes are width-invariant (`joint_decode_routes.rs:14-20`), so greedy transcripts should not depend on composition [INFERENCE]. T1-vs-O1 measures it, and T1-vs-T0 separates hook from observer.
5. **Joint verify (ADR-065 J-series):** it will need its own role and source extension. This design rejects `target_verify` on the joint source until then.
