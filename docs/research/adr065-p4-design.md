# ADR-065 P4a/P4b: GPU restore of DeltaNet state from capture — HLD/LLD

Stories:
- [P4a #526](https://github.com/robertelee78/hf2q/issues/526): native primitive. mlx-native owns it. Blocked by E6 #509.
- [P4b #527](https://github.com/robertelee78/hf2q/issues/527): hf2q integration. Blocked by B1 #503, J1 #512 and P4a.
- ADR text: [ADR-065 §P4a/§P4b](../adr/ADR-065-four-agent-joint-speculative-verify.md#p4a-provide-gpu-restore-of-deltanet-state-from-capture).
- Downstream consumers: ADR-063 C2 #360, whose anchor restore reuses this primitive (see [adr063-c2-design.md](adr063-c2-design.md)), and J4 #515 for joint verify rollback.

Authored against hf2q `main` `d817dffa7371a2db85a43ed1d6502350906c33f5` and the pinned `mlx-native =0.15.16` (crates.io source). Source identities are listed at the end. Line numbers help reading only.

This is a design. Nothing was built, run or measured for it. Every cost figure below is arithmetic from shapes or is marked [INFERENCE].

## 1. Scope

**P4a requirement.** The program MUST provide a GPU restore of captured DeltaNet state, including the parity-aware conv re-index, using existing copy primitives where E6 shows they suffice. Acceptance:
- restored state equals the host path bitwise for full, partial and zero acceptance;
- if E6 shows existing copy primitives suffice, the story closes with that record and no release;
- otherwise a kernel is released with parity tests on both metallib paths and a microbenchmark.

**P4b requirement.** The program MUST use the P4a restore for solo K3 and, where J4 has landed, joint verify rollback. Acceptance:
- exact pin of any P4a release from a clean registry build;
- full, partial and zero acceptance transaction cases pass;
- `ExactTranscripts` against the previous default;
- a four-agent A/B with fairness.

The outcome: partial-reject rollback no longer copies DeltaNet state through host memory.

### 1.1 The host path being replaced

`HybridKvCache::rollback_la_to(slot, accepted_idx)` (`src/inference/models/qwen35/kv_cache.rs:2745-3025`) loops over every linear-attention layer (`self.linear_attn`). For each layer it does two copies.

**Recurrent copy (2821-2873).**
- Maps `capture_states` with `as_slice::<f32>()`.
- Copies the slot/token row `[slot·T·E + t·E, +E)` into a fresh `Vec` (`to_vec`).
- Copies that into `recurrent_current_mut(slot)`, the buffer that is current for this slot's ping-pong parity (`kv_cache.rs:867`), at `slot·E`.
- Here `E = D_k·D_v·n_v_heads`, and `T` is the capture's token axis derived from the buffer's element count.

**Conv copy (2875-3022), with a re-index.**
- Maps `conv_capture_states`. Its layout is `[S, T, K−1, C]` with channels innermost, written by `ssm_conv_capture_forward_f32` (mlx-native `src/shaders/ssm_conv_capture.metal:21-30`).
- Copies the `(slot, t)` row into a `Vec`.
- Transposes into `conv_current_mut(slot)` (`kv_cache.rs:856`), whose slot region is `[C, K−1]` with `K−1` innermost: `dst[c·(K−1)+k] = src[k·C + c]` (3016-3022).

"Parity-aware" therefore means two things:
1. the destination is the buffer that the slot's `pp_flipped` bit (`kv_cache.rs:163`) marks current, which is the buffer the verify forward just wrote and flipped to;
2. the conv row is transposed from capture layout to state layout.

Callers:

| Caller | Site |
|---|---|
| Slot-aware MTP K3 commit | `src/serve/api/engine_qwen35.rs:6043`; after J1, the restore seam of `decode_tick_mtp_k3` |
| Slot-aware history-lookup commit | `engine_qwen35.rs:5672` |
| Serial K=N spec decode | `src/inference/models/qwen35/spec_decode.rs:939` |
| DFlash Qwen target | `src/inference/spec_decode/dflash/qwen35_target.rs:223` |
| Unit tests | `kv_cache.rs:9125-9240`, `10700-10950`, `11660-11680` |

Capture geometry is fixed by `CaptureAllocationPlan` (`src/inference/models/qwen35/kv_cache/allocation_plan.rs:330-415`):
- recurrent capture is `[D_k, D_v, n_v, T, S]` and conv capture is `[S, T, K−1, C]`, both f32;
- multi-slot caches (`S > 1`) retain a capture buffer only when its element count equals the requested one (376-379), so `T` equals the active depth;
- single-slot caches may retain a deeper buffer, where only `s = 0` exists and the token stride is still `E`.

### 1.2 Sizes at served shapes (arithmetic, not measurement)

| Artifact (Q5 case) | DeltaNet layers | E (f32) | Recurrent bytes / slot-layer | C | Conv bytes / slot-layer | Bytes restored per slot |
|---|---|---|---|---|---|---|
| Qwen3.8-27B dense (`q5_cases/qwen38.json`; ADR-044 geometry: 48 linear layers, 16 key / 48 value heads, dim 128, conv width 4) | 48 | 786,432 | 3,145,728 | 2·16·128 + 48·128 = 10,240 | 122,880 | 156,893,184 (149.6 MiB) |
| Qwen3.6-35B-A3B MoE (`q5_cases/qwen36.json`; ADR-013 MoE geometry: 40 layers, interval 4, 32 value heads) | 30 | 524,288 | 2,097,152 | 8,192 | 98,304 | 65,863,680 (62.8 MiB) |

The host path touches each restored byte four times: capture read, `Vec` write, `Vec` read and destination write. It also makes `2·L` heap allocations of up to 3 MiB per rollback (96 on Qwen3.8). [INFERENCE] On Apple Silicon, large fresh allocations can incur first-touch page faults. The restore cost is therefore unknown until measured (P4b-M0, §5.2). No claim is made here.

## 2. P4a — native primitive contract (mlx-native)

### 2.1 Normative contract: `DeltaNetCaptureRestore`

The contract is stated in mlx-native terms, with no external engine names (mlx-native `AGENTS.md` policy).

**Inputs, per DeltaNet layer `ℓ`:**

| Input | Content |
|---|---|
| `rec_cap` | f32 view, logical shape `[E, T, S]` (E fastest) |
| `conv_cap` | f32 view, logical shape `[S, T, Km1, C]` (C fastest) |
| `rec_dst[j]`, `conv_dst[j]` | f32 destination views of exactly `E` and `C·Km1` elements, one per item `j` |
| item `j` | `(s_j < S, t_j < T, rec_dst[j], conv_dst[j])` |

**Effect.** For every item `j`, every `e < E`, `c < C` and `k < Km1`:

```
bits(rec_dst[j][e])            := bits(rec_cap[(s_j·T + t_j)·E + e])
bits(conv_dst[j][c·Km1 + k])   := bits(conv_cap[((s_j·T + t_j)·Km1 + k)·C + c])
```

`bits(·)` is the raw 32-bit pattern. Equality is bitwise for every u32 value, including signaling and quiet NaN payloads, ±0, subnormals and ±∞. No arithmetic, conversion or flush-to-zero is permitted.

**Preconditions.** The caller validates these; the primitive refuses on violation:
- destination views are pairwise disjoint;
- no destination overlaps any source;
- all views lie within their buffers' logical extents;
- `E`, `C` and `Km1` are greater than 0;
- index arithmetic does not overflow u32 inside a view.

**Cohort invariance at primitive level (ADR-065 Class A).** The bytes written for item `j` depend only on `(rec_cap, conv_cap, s_j, t_j)`. They do not depend on the other items, the item order, the number of items (1–4) or which slots the items name. Bytes outside the destination views are unchanged, including peer slots' regions in the same physical buffers.

**Ordering and completion.** All work is encoded into a caller-provided `CommandEncoder`. Nothing creates encoders, commits or waits. Work encoded before the restore (the verify forward, or nothing) completes before the restore reads its sources, through command-buffer order or an explicit barrier the helper documents. Completion is the caller's commit and wait. A GPU fault surfaces as `MlxError::CommandBufferError` from that wait.

**Anchor images.** A tensor with capture layout and `T = S = 1` is a valid `rec_cap`/`conv_cap` pair. Restoring from such an image with `(s, t) = (0, 0)` is inside this contract; C2 relies on it (C2 §3.3). The contract covers capture-layout → state-layout only. The reverse direction (state → capture image) is not part of P4a.

### 2.2 Delivery: existing primitives (path A), decided by E6

ADR-065 E6 item 4 asked whether the existing GPU copy primitives serve capture-to-state restore with the parity-aware conv re-index. E6 has answered yes. The record is on mlx-native branch `test/adr065-e6-probes` (commits `608425d`, `7cbafef`, `e8c63f1`), in `docs/adr065-e6-verify-window-probes.md` (rows 4 and "Capture restore") with test `tests/adr065_e6_capture_restore_copy.rs`:
- `e6_capture_restore_blit_and_strided_copy_match_host_rollback` passed bitwise on both `MLX_PRECOMPILED_METALLIB=1` and `=0`;
- it covered four slots, mixed ping-pong parity, accepted `t` ∈ {0, 1, 2, 3} and one command buffer, against a port of `rollback_la_to`;
- with `E6_STRICT_NAN=1` it included every NaN payload, so every u32 bit pattern passed for both the blit and the strided copy.

P4a therefore commits to **path A**. No kernel is built, no release is made, and no microbenchmark is required (ADR-065 requires one only when a kernel is built). The composition, per layer and item, is:

| Component | Primitive (mlx-native 0.15.16) | Arguments |
|---|---|---|
| Recurrent | `CommandEncoder::blit_copy_bytes` (`src/encoder.rs:1563`); byte-exact by Metal definition, and E6-verified | source = `rec_cap` at byte offset `(s·T+t)·E·4`, destination = `rec_dst`, length `E·4` |
| Conv re-index | `ops::copy::dispatch_strided_copy_f32` (`src/ops/copy.rs:68`, kernel `strided_copy_f32`, `src/shaders/copy.metal:24`) | `src = conv_cap.slice_view(((s·T+t)·Km1·C)·4, Km1·C)`, `dst = conv_dst`, `rows = C`, `cols = Km1`, `stride_row = 1`, `stride_col = C`. The kernel computes `dst[row·cols+col] = src[row·stride_row + col·stride_col]` = `dst[c·Km1+k] = src[k·C+c]` |

`offset_copy_f32` and `transpose_2d` are not used. E6 never probed `offset_copy_f32`, and one route per component keeps the qualification surface minimal.

**Encoding order.** Each `blit_copy_bytes` call ends any active compute pass and then opens and ends its own blit encoder (`src/encoder.rs:1636-1651`). The hf2q helper encodes all conv strided copies (one compute pass) for all layers and items first, then all recurrent blits (`L·items` blit encoders; 48 per item on Qwen3.8). The two families have disjoint destinations and read-only sources, so no barrier is needed between them. E6 interleaved blits and strided copies; this compute-first ordering is a different encoding, so the merged parity test covers it explicitly (T-P4a-3).

**Bounds checks.** `dispatch_strided_copy_f32` validates against `MlxBuffer::byte_len()`, the whole Metal allocation (`src/buffer.rs:338`), not the view's `data_byte_len()` (`:347`). Its native check is therefore not view-exact, and hf2q preflights every range (§3.3). `blit_copy_bytes` validates against view extents (`data_byte_len()`, 1579-1593) and rejects overlapping same-allocation ranges.

**Closure.** P4a closes on the E6 record **merged to mlx-native main**, extended with the tests of §4.1. The branch is unmerged today, so P4a does not close on it yet. The merged record includes:
- `tests/adr065_e6_capture_restore_copy.rs`, plus the compute-first ordering, anchor-image and preflight cases of §4.1;
- the C2 snapshot-direction cases (§2.4);
- `docs/adr065-e6-verify-window-probes.md`, plus the contract of §2.1, the bounds-check caveat, and a statement that P4a is satisfied without a release.

hf2q composes the primitives itself, because ADR-065 assigns encode-only helpers to hf2q.

### 2.3 Runtime-registry fix (not a P4a dependency)

E6 also found that `KernelRegistry::new()` does not register the `gated_delta_net_decode_capture_f32_*` runtime sources (record row "`gated_delta_net_decode` capture", `KernelNotFound` on the runtime path). hf2q's production registry (`forward_gpu.rs:3604-3609`) never calls `ops::gated_delta_net_decode::register`. So at 0.15.16, every K3 or history verify with `MLX_PRECOMPILED_METALLIB=0` fails with a non-fatal `KernelNotFound`, and the slot resets.

- **P4a is unaffected.** The restore primitives pass on the runtime path at 0.15.16, and the P4a tests fill capture buffers from the host.
- **P4b and J1 hardware evidence is precompiled-only** until the mlx-native fix `4e32003` ("register gated_delta_net_decode_capture runtime sources in KernelRegistry::new") is merged, released, verified on crates.io and pinned exactly in hf2q. Every receipt states the metallib path.
- **That release is not a P4a release.** P4b acceptance item 1 ("exact pin of any P4a release") stays vacuous. The pin bump is tracked separately, and the root files the release issue. It lands before, or entirely outside, the P4b A/B binary and the J1 base/candidate pair.

### 2.4 C2 snapshot-direction parity in the same record

ADR-063 C2 requires any state-copy primitive it uses to be parity-tested in mlx-native before a pinned integration. C2's snapshot (state → anchor image) uses the same two primitives in the reverse direction (C2 design §3.3). The merged qualification record therefore also covers:
- a blit **out of** the parity-current recurrent region of slot `s` (`s > 0`, both parities) into a `T = S = 1` image, bitwise;
- the inverse conv re-index with `dispatch_strided_copy_f32`: `rows = Km1`, `cols = C`, `stride_row = 1`, `stride_col = Km1`, giving `img[k·C + c] = state[c·Km1 + k]`, bitwise for every u32 class, on both metallib paths;
- a round trip from state to image and back to state that is byte-identical.

These are tests of existing primitives, not a P4a contract extension. The §2.1 contract stays capture → state.

### 2.5 Native out-of-scope

- A new native kernel or release.
- Deeper or variable `T` semantics.
- Gemma and DeepSeek state.
- Any change to `ssm_conv_capture_forward_f32` or the capture layout.
- Fusing the restore into the next forward.

## 3. P4b — hf2q integration

### 3.1 Interface

A new module, `src/inference/models/qwen35/kv_cache/la_restore.rs`, is a child of `kv_cache`, beside `allocation_plan.rs`.

```rust
/// Where a restore reads from. `CaptureRow` is the K3/history/J4 rollback
/// source; `AnchorImage` is ADR-063 C2's anchor payload, a capture-layout
/// image with T = S = 1 (P4a contract §2.1 "Anchor images").
pub enum LaRestoreSource<'a> { CaptureRow(u32), AnchorImage(&'a LaAnchorImage) }

/// Where a restore writes. `CurrentByParity` reproduces
/// `recurrent_current_mut`/`conv_current_mut` and leaves `pp_flipped`
/// unchanged (today's rollback). `NamedResetParity` writes the named
/// `recurrent`/`conv_state` regions and then sets `pp_flipped[slot] = false`
/// (today's `restore_slot_anchor`, `kv_cache.rs:1698-1716`).
pub enum LaRestoreDestination { CurrentByParity, NamedResetParity }

pub struct LaRestoreItem<'a> {
    pub slot: SlotId,
    pub source: LaRestoreSource<'a>,
    pub destination: LaRestoreDestination,
}

/// Validated plan: every per-layer source/destination byte range checked
/// against `data_byte_len()` and against each other before any encode.
pub(crate) struct LaRestorePlan { /* per layer: (src ranges, dst buffer+range) per item */ }

/// Which implementation a cache uses for slot-aware DeltaNet restores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaRestoreRoute { Host, Gpu }

impl HybridKvCache {
    /// Preflight only; no mutation, no encode.
    pub(crate) fn plan_la_restore(&self, items: &[LaRestoreItem<'_>]) -> Result<LaRestorePlan>;
    /// Encode the plan into `enc` (blits plus strided copies, §2.2).
    /// Does not commit or wait. Mutates nothing host-visible.
    pub(crate) fn encode_la_restore(&self, enc: &mut CommandEncoder,
                                    registry: &mut KernelRegistry,
                                    plan: &LaRestorePlan) -> Result<()>;
    /// Apply the host-side half of a completed plan (parity bits for
    /// `NamedResetParity` items). Called only after a successful wait.
    pub(crate) fn finish_la_restore(&mut self, plan: LaRestorePlan);
    /// Slot-aware rollback: dispatches on `self.la_restore_route`.
    /// `Gpu`: `(CaptureRow(accepted_idx), CurrentByParity)`, plan, encode
    /// into a fresh command buffer, commit and wait, then finish.
    /// `Host`: the unchanged `rollback_la_to`.
    pub(crate) fn rollback_la(&mut self, slot: SlotId, accepted_idx: u32) -> Result<()>;
    /// Today's host copy, unchanged (`kv_cache.rs:2745-3025`).
    pub fn rollback_la_to(&mut self, slot: SlotId, accepted_idx: u32) -> Result<()>;
    /// Multi-slot rollback for J4: one command buffer for all items.
    /// Shipped by P4b only if J4 has landed; otherwise J4 adds it (§3.2).
    pub(crate) fn rollback_la_many(&mut self, items: &[(SlotId, u32)]) -> Result<()>;
}
```

P4b ships only the `CaptureRow` + `CurrentByParity` combination on a production path. C2 adds the `AnchorImage` + `NamedResetParity` caller ([adr063-c2-design.md](adr063-c2-design.md) §3.3). The enums are introduced by whichever of P4b or C2 lands first that needs the variant, so no variant ships without a caller.

**Callers and route.**

| Caller | After P4b | Why |
|---|---|---|
| Slot-aware MTP K3 commit (`engine_qwen35.rs:6043`; J1 restore seam) | `rollback_la` (route-dispatched) | P4b requirement ("solo K3") |
| Slot-aware history-lookup commit (`engine_qwen35.rs:5672`) | `rollback_la` | Same transaction shape and supervision as K3 |
| J4 joint verify (when landed) | `rollback_la_many` or the plan/encode/finish API | P4b requirement ("joint verify rollback") |
| Serial K=N spec decode (`spec_decode.rs:939`) | `rollback_la_to`, host, unchanged | Not a P4b requirement. It has no supervisor, and its fail-stop policy is not the slot-aware worker's |
| DFlash Qwen target (`dflash/qwen35_target.rs:223`) | `rollback_la_to`, host, unchanged | The `rollback_kv` trait method is infallible and logs and swallows errors (213-229). A GPU `CommandBufferError` there would be hidden, which ADR-065's fail-stop rule forbids |
| Tests (`kv_cache.rs:9125-9240`, `10700-10950`, `11660-11680`) | `rollback_la_to` unchanged; it is also the parity oracle for T-P4b-2 | |

So the GPU path can only run where failures fail-stop through the slot-aware worker (§3.4). `rollback_la_to` is not renamed and has no new callers.

- **Route resolution.** `HybridKvCache` gains `la_restore_route: LaRestoreRoute`, initialized to `Host` by every constructor. Only the slot-aware Qwen engine sets it, once at worker construction, from `resolve_la_restore_route(env, artifact)`:
  - `HF2Q_QWEN_LA_GPU_RESTORE=0` → `Host`; `=1` → `Gpu`, for any artifact (operator override);
  - invalid values fail closed at load;
  - **unset, dense artifact (Qwen3.8)** → the shipped default set by the A/B verdict (§5.3);
  - **unset, MoE artifact (Qwen3.6)** → `Host`, always. ADR-065 excludes Qwen3.6 default changes, and P4b gathers no Qwen3.6 evidence. There is no owner exception.

  The resolved route is logged in the worker's startup line, beside the speculation policy (`qwen35_speculation.rs:221-224`). Caches built anywhere else (serial runner, DFlash, tests) keep `Host` unless a test sets the route explicitly.
- **Kernel registry.** `HybridKvCache` gains a lazily created `restore_registry: Option<KernelRegistry>`, used only for `strided_copy_f32` (blits need no pipeline). It is created once per cache on the first GPU restore, so it does not compile pipelines per call, unlike the graft splice (`kv_cache.rs:2311`). The metallib selection follows the same `MLX_PRECOMPILED_METALLIB` policy as the model registry. Whether `KernelRegistry` is `Send` with the cache is checked at implementation. If it is not, the registry is passed in from `with_gpu_cache_mut` at the two slot-aware callers.

### 3.2 Data flow (solo K3 after J1)

```mermaid
sequenceDiagram
  participant W as decode_tick_mtp_k3 (wrapper)
  participant K as HybridKvCache
  participant G as GPU queue
  W->>G: k3_verify_solo: 4-row verify writes K/V, capture rows 0..3, flips slot parity (committed + waited)
  W->>W: k3_verify_to_plan: catch-up, plan (valid_input_tokens, carry_hidden_row)
  W->>K: k3_commit_cursors: truncate target + MTP cursors
  alt partial (valid_input_tokens < 4)
    W->>K: rollback_la(slot, carry_hidden_row)  [route Gpu]
    K->>K: plan_la_restore (preflight all layers)
    K->>G: one command buffer: conv strided copies ×L (one compute pass), then recurrent blits ×L
    G-->>K: commit_and_wait (error → MlxError::CommandBufferError)
  end
  W->>K: k3_commit_settle: clear capture, validate cursors
  W->>W: k3_commit_finish: carry hidden, ledger
```

- **Sources are complete before the restore runs.** The verify forward's terminal command buffer was waited on: logits were read on the CPU by the semantic walk, so the capture rows are complete. The restore command buffer is the next GPU work for this slot, and the next reader of the restored state is a later forward submitted after `commit_and_wait` returns. No fence beyond command-buffer order is needed.
- **Supervision.** The slot-aware K3 and history commits wrap the restore in `supervisor.arm("Qwen35 SlotAware MTP K3 restore", QWEN35_WORKER_TRANSACTION_TIMEOUT)` (history: "Qwen35 history restore"), so a hung wait is supervised like every other GPU stage of the round. The serial K=N and DFlash callers stay on the host path (§3.1), so they need no supervision change.
- **Joint path.** If J4 has landed, its commit calls `rollback_la_many` once for every partially rejected lane, between `k3_commit_cursors` (all lanes) and `k3_commit_settle` (J1 §3.4). If J4 has not landed when P4b merges, P4b does not ship `rollback_la_many`, which would have no production caller. `plan_la_restore`/`encode_la_restore`/`finish_la_restore` already take item slices, and J4 adds `rollback_la_many` as a thin wrapper over them rather than a second restore path. The root should record that as a J4 obligation.
- **Profile line.** The `HF2Q_MTP_PHASE_PROFILE` field `rollback=` keeps measuring the wall time of the restore call, now including encode, commit and wait.

### 3.3 Preflight (`plan_la_restore`), all before any encode

For every layer:
- capture and conv capture are present;
- `recurrent.element_count() % n_seqs == 0`, and so on, with the same checks and error texts as today;
- `T` is derived as today, and `accepted_idx < T`;
- the destination buffer is chosen by the slot's `pp_flipped`, which mirrors `recurrent_current_mut` / `conv_current_mut`;
- every source and destination byte range lies within its buffer's `data_byte_len()`;
- destination ranges are pairwise disjoint across items;
- the item slots are unique.

Any failure returns `Err` with nothing encoded and nothing mutated. This matches today's host path, which also validates each layer before writing it. The difference is that today a late-layer validation failure leaves earlier layers already written. P4b strictly improves that to all-or-nothing preflight.

### 3.4 Failure semantics

| Failure | Effect | Handling |
|---|---|---|
| Preflight refusal | Nothing mutated | Error to the caller. The K3 and history commits run their existing fail-closed path (clear capture, `reset_for_slot`), as for today's host errors |
| Encode error (pipeline compile, argument validation) | Nothing committed; partially encoded commands are dropped with the uncommitted encoder | Same as preflight refusal |
| `CommandBufferError` on wait | Destination bytes of the participating slots are unknown | The error is wrapped with `.context(...)`, never stringified, so `is_fatal_command_buffer_error` (`src/serve/api/engine.rs:17903-17911`) classifies it as fatal. Per the ADR-065 failure classes, a worker-fatal error fail-stops the worker: the existing fatal path terminates the affected requests and publishes no partial state. Any `reset_for_slot` the caller runs first only zeroes host-visible bytes and is harmless |
| Supervisor timeout | As today for any supervised stage | Unchanged policy |

There is no host fallback after a GPU attempt. Once a restore was submitted, the destination state is unknown. Retrying on the host would copy correct bytes, but it would also mask a GPU fault that ADR-065 requires to fail-stop.

Non-participating slots are never written: destinations are slot regions, and preflight proves disjointness. A failed multi-slot restore (J4) poisons only the participating lanes, unless the error is fatal, in which case the whole worker stops (ADR-065 shared-submitted-forward failure semantics).

These guarantees hold because only slot-aware worker callers can reach the GPU route (§3.1 caller table). The serial K=N runner and the DFlash target keep today's host copy and today's error handling. In particular, DFlash's log-and-continue on rollback errors is unchanged and never sees a GPU error.

## 4. Tests

### 4.1 P4a (mlx-native)

All P4a tests extend the E6 record (`tests/adr065_e6_capture_restore_copy.rs`) and run on `MLX_PRECOMPILED_METALLIB=1` and `0`. They use exactly the path A composition: `blit_copy_bytes` for recurrent and `dispatch_strided_copy_f32` for conv.

| ID | Test |
|---|---|
| T-P4a-1 | (E6, existing) Parity against a port of `rollback_la_to` for four slots, mixed parity, `t ∈ {0..3}`, one command buffer, every u32 bit class with `E6_STRICT_NAN=1` |
| T-P4a-2 | Acceptance classes as rows: full (no item, a no-op), partial (`t ∈ {1, 2}`) and zero (`t = 0`). The restored state equals the host path's bytes (the reference reproduces `kv_cache.rs:2821-3022`) |
| T-P4a-3 | **Compute-first ordering**, the encoding hf2q ships: every conv strided copy for all layers and items in one compute pass, then every recurrent blit, in one command buffer. Covers 1–4 items, every subset of 4 slots in every order, served shapes (`E = 786,432`, `C = 10,240`; `E = 524,288`, `C = 8,192`; `Km1 = 3`; `T = 4`) and a 2-layer small shape. Each item's destination bytes equal the single-item run, and peer regions and the non-current parity buffer equal a sentinel fill |
| T-P4a-4 | Anchor-image form: `T = S = 1`, `(0, 0)` (C2 dependency) |
| T-P4a-5 | C2 snapshot direction (§2.4): blit out of the parity-current recurrent region and the inverse strided copy into a `T = S = 1` image, bitwise for every u32 class; round trip from state to image and back to state is byte-identical |
| T-P4a-6 | Native refusals: `blit_copy_bytes` rejects overlapping same-allocation ranges and views past `data_byte_len()`, and `dispatch_strided_copy_f32` rejects zero dims. The record also documents that the strided copy's bounds check uses `byte_len()`, so view-exact checking is hf2q's preflight (§3.3) |

### 4.2 P4b (hf2q)

| ID | Test | Kind |
|---|---|---|
| T-P4b-1 | The existing `rollback_la_to_*` tests (`kv_cache.rs:9125-9240`, `10700-10950`, `11660-11680`) pass unchanged (host). The same cases also run through `rollback_la` with the route set to `Gpu` | unit, Metal |
| T-P4b-2 | GPU vs host parity on a tiny Qwen config with real `HybridKvCache` geometry: random capture payloads with the §2.1 bit classes, both parities per slot, every `(slot, idx)`. Current-buffer bytes and parity bits equal `rollback_la_to`, and scratch and peer bytes are unchanged | unit, Metal |
| T-P4b-3 | A multi-item plan (`plan_la_restore` + `encode_la_restore` + `finish_la_restore`, one command buffer) with 2–4 slots in permuted orders equals sequential single restores byte for byte (Class A building block for J4; it also covers `rollback_la_many` when shipped) | unit, Metal |
| T-P4b-4 | Preflight atomicity: corrupt the last layer's conv capture shape. The call errors and the bytes of every layer are unchanged. Today's host path partially writes, so this is a deliberate behaviour improvement | unit, Metal |
| T-P4b-5 | Transaction cases: the speculation qualification suite (`src/serve/api/engine_qwen35/speculation_qualification_tests.rs`: full, partial and zero acceptance followed by 8 ordinary decisions; chained mixed rounds; EOS in or after the block; length boundary; cancellation mid-drain) green with the route at `Gpu` and at `Host` | unit |
| T-P4b-5b | Route resolution: unset on a MoE artifact → `Host` regardless of the dense default; `0`/`1` override on both families; an invalid value fails at load; caches built by the serial runner and DFlash stay `Host` | unit |
| T-P4b-5c | Fatal classification: an injected `MlxError::CommandBufferError` from the restore wait reaches `is_fatal_command_buffer_error` through the K3 and history commit paths (error chain not stringified) | unit |
| T-P4b-6 | Q1 actual transaction store: `q1_qwen_actual_t5_speculation_phase` on the official Qwen3.8 artifact, switch 0 vs 1. Stores are `ExactEqual` (`scripts/four_agent/q1_decision_policy.py`), including the `state_committed` snapshots after partial rounds | real artifact, GPU lane |
| T-P4b-7 | `qwen38_slot_aware_speculation_auto_matches_off` passes with the switch at 1 | real artifact, GPU lane |
| T-P4b-8 | Cancel smoke (`scripts/four_agent/cancel_smoke.py`, `q5_cases/cancel-qwen38.json`) with the switch at 1 | hardware |
| T-P4b-9 | Serving transcripts `ExactTranscripts`, switch 1 vs 0 on the same binary, from the A/B runs (§5.3) at `HF2Q_QWEN_SPECULATION=auto`, both the four-agent and the N=1 runs | hardware |
| T-P4b-10 | If J4 has landed: the J4 transaction matrix (mixed full, partial and zero across lanes) re-run with P4b on. Class A cohort invariance of post-commit DeltaNet state (current conv/recurrent and parity) between REF(w) alone and cohorts of 1–4 windows | real artifact, GPU lane |

Clean-registry pin (P4b acceptance item 1): P4a makes no release, so no P4a pin exists, and the receipt says so. The `mlx-native` pin a P4b binary uses is whatever main pins. If the runtime-registry fix release (§2.3) is pinned by then, that bump is a separate commit landed before the P4b binary is built. It is never introduced between A/B arms, which share one binary anyway.

## 5. Measurement plan

All hardware runs need the GPU lane granted by the root. Each records source commit, binary SHA-256, artifact digest, native version and case file, and passes Q5 `CalibrationPass`. Evidence goes under `/Users/robert/.local/state/hf2q/v0130/p4b-527/` (P4a qualification receipts: `…/p4a-526/`).

Until the runtime-registry fix is released and pinned (§2.3), every P4b hardware run uses the precompiled metallib (`MLX_PRECOMPILED_METALLIB` unset or `1`). The runtime metallib cannot run K3 at 0.15.16, and receipts state the path. The T-P4a tests cover both metallib paths for the restore primitives themselves.

### 5.1 Why the default four-agent configuration barely exercises P4b

With `HF2Q_QWEN_SPECULATION` unset, the dense Qwen3.8 four-slot worker resolves speculation to `off` (`default_server_policy`, `qwen35_speculation.rs:38-48`; ADR-063 D1 measured a 1.020 makespan loss). Single-slot workers default to `auto`. The changed code runs only on K3 and history partial rejections. Therefore:
- the four-agent A/B arms both run at `HF2Q_QWEN_SPECULATION=auto`. That is a declared campaign parameter, not an ADR-065 kill switch, so "previous default" is still the same binary with only `HF2Q_QWEN_LA_GPU_RESTORE` toggled;
- the N=1 stage is an engaged stage, not a ±2.5-point non-engagement control, because single-agent K3 is the path this changes, and single-slot dense workers are where the shipped default runs P4b most often. The campaign therefore runs with `--n1-engages` (B1 campaign doc, "Controls"; `scripts/four_agent/adr065_stats.py`). Under that flag the analyzer reports the `n1` effect instead of gating on the ±2.5-point window, and P4b's "N=1 not Loss" condition (§5.3) is judged from that reported effect.

Qwen3.6 is not measured. Its default route stays `Host` (§3.1).

### 5.2 P4b-M0: restore cost attribution (diagnostic, before the A/B)

- ADR-065 E1 already attributes a "rollback (CPU)" stage class. If E1's receipts exist, P4b cites their solo-K3 rollback share.
- Otherwise, one single-agent `auto` run per arm with `HF2Q_MTP_PHASE_PROFILE=1` gives the distribution of `rollback=` ms on rounds with `partial=true` and the partial-round rate.
- No decision rule hangs on M0. It sizes expectations and supplies the E1-style stage split for the record.

### 5.3 P4b A/B (acceptance)

The ADR-065 two-arm rule applies:
- one binary; arms `HF2Q_QWEN_LA_GPU_RESTORE=0` (A, previous default) vs `=1` (B);
- ABBA paired blocks with warm-up; the block count is the one fixed in B1's campaign file for Qwen3.8 A/Bs (ADR-065 B1), not chosen here;
- paired log-makespan Student-t 95% interval;
- verdicts in the order Loss, Win, Equivalent, Inconclusive;
- **fairness: B1's checks, unchanged** (`docs/research/adr065-b1-campaign.md` "Fairness" and its ADR amendment, as of `1dad1672`; implemented by `scripts/four_agent/adr065_stats.py`). Both checks apply to arm B only:
  1. **Paired regression test, the primary check.** This is B1's block-paired B-vs-A test, reused unchanged: Q3's `q3_report.FAIRNESS`, `q3_report.effect` and `q3_stats.regression`, applied to the run's worst-slot p95 semantic TTFT and worst-slot maximum semantic SSE gap. It fails if the ratio interval's lower bound exceeds 1.025, and a measured metric without an interval is `Unverifiable` and fails. Arm A is the same binary at `auto` with the switch at 0, so this test isolates P4b by construction and needs no `auto` baseline.
  2. **Absolute safety cap.** `CAP_FACTOR = 1.5` × **B1's frozen phase-1 (`off`) ceilings** for the workload. This holds in every measured B run and slot, and a missing value is `Unverifiable` and fails. The cap is an absolute bound on user-facing latency, not a comparison with the previous default, so the phase-1 ceilings are a sound base even though both arms run at `auto`: the cap asks whether any served run is pathological against the frozen phase-1 service level, and that question does not depend on the arm's speculation policy. P4b does not freeze a separate `auto` baseline.

  If an arm-A run would also breach the cap, the cap still fails B's breaching runs, and the default stays off (§5.3 mapping). The receipt then records the arm-A values beside B's, so the root can see that the breach comes from `auto` rather than from P4b and decide whether a frozen `auto` baseline is warranted for a later campaign;
- quality: the arms are `ExactTranscripts` (T-P4b-9), so B inherits task quality. A non-exact pair blocks the story pending diagnosis;
- the Q5 case is the B1 campaign file's Qwen3.8 case derived at `auto`, plus its N=1 case. Q5 admission is re-established for the P4b binary. The capture footprint is unchanged, because the GPU path allocates nothing beyond the one strided-copy pipeline.

Default mapping, for dense artifacts only (MoE stays `Host`). ADR-065 says "default set by the A/B verdict" but gives no mapping for P4b. Accepted by review with conditions (Win/Equivalent → on, following the J6 precedent):

| Four-agent verdict | Default (dense, unset) | Conditions |
|---|---|---|
| Win | on | The N=1 effect reported under `--n1-engages` is not Loss (its interval does not exclude zero unfavourably, by the same verdict order); `gates_pass` (fairness, safety cap, Q5, thermal, completeness); transcripts exact |
| Equivalent | on | Same conditions. Exact transcripts plus a bounded no-difference result: the story's outcome (no host copy) and J4's need for a batched restore favour the GPU path at no measured cost |
| Loss or Inconclusive, or any condition failing | off | The verdict is recorded. The switch stays, and the host path remains the default |

## 6. Risks

| Risk | Mitigation |
|---|---|
| Native bounds checks for the strided copy use `byte_len()`, not view extents | hf2q preflight checks every range against `data_byte_len()` (§3.3). T-P4a-6 records the caveat |
| Encoder-switch overhead from `L·items` blit encoders | Compute-first ordering (§2.2). P4b-M0 and the A/B measure the whole restore; a Loss leaves the default off |
| Command-buffer submit and wait (one per partial round) costs more than the host copy | P4b-M0 plus the A/B decide the dense default; the N=1 verdict is a separate condition |
| A stringified error hides a GPU fault from fatal classification | §3.4 rule and T-P4b-5c |
| A GPU restore reached through a caller without fail-stop semantics | Only slot-aware callers use `rollback_la`; serial K=N and DFlash keep `rollback_la_to` (§3.1). T-P4b-5b |
| Qwen3.6 default changes without evidence | Unset MoE resolves to `Host` (§3.1), tested by T-P4b-5b |
| Runtime metallib cannot run K3 at 0.15.16 | Precompiled-only hardware evidence until the registry fix is pinned (§2.3, §5) |
| The cache-owned `KernelRegistry` is not `Send` | The alternative in §3.1 (registry passed in at the slot-aware callers) |
| J4 lands later and adds its own host rollback | The slice-taking plan/encode/finish API exists; J4 adds only the thin `rollback_la_many` wrapper, plus the J4 obligation recorded by the root (§3.2) |

## 7. Non-goals

- Fusing the restore into the next target forward. This would be deferred restore with a "pending restore" slot state that every reader must honour; it can be revisited only with a deciding receipt.
- Changing capture layout or depth, J2 per-slot capture ownership, the ping-pong parity scheme, or zero-copy full-rejection rewinds (`rewind_la_ping_pong_for_slot`, `kv_cache.rs:3035`).
- Converting `snapshot_prefix_from_capture` (`kv_cache.rs:3383-3402`, host transpose at 3667-3676; SerialFifo latest-turn checkpoint) or `snapshot_slot_anchor`/`restore_slot_anchor`. C2 covers anchor restore.
- Gemma and DeepSeek state; approximate or BF16 state.
- Any served-output change.

## 8. Source identities read

| Path | SHA-256 |
|---|---|
| `src/inference/models/qwen35/kv_cache.rs` | `6bc46b78124b9e41b494a5136324e03b76edb5a584279271291d4d8b5cf4518e` |
| `src/serve/api/engine_qwen35.rs` | `47f755feeeed176aa5dd77179e430ceb2d09a6f0e5b14d883a8f2e1f5ce6bd34` |
| `src/serve/api/engine.rs` | `9383b87263ea7dc252a8ac4bb64eb16780169a82d792dfafc7987772f95f574f` |
| mlx-native 0.15.16 `src/encoder.rs` | `275ea47915de3724e630ac578c8aa3745cf872e026720e71ebf3085d0b65944b` |
| mlx-native 0.15.16 `src/ops/copy.rs` | `e9c7a5b493c3258dbff22dba51a53e51d7eb52330a2b11298ccb19de5bdf62bb` |
