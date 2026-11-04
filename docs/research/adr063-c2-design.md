# ADR-063 C2: Qwen multi-depth prompt reuse on the P4a/P4b DeltaNet restore — HLD/LLD

Story: [C2 #360](https://github.com/robertelee78/hf2q/issues/360), [ADR-063 §C2](../adr/ADR-063-apple-silicon-four-agent-speed-coherence.md#c2-implement-and-qualify-qwen-multi-depth-prompt-reuse).

Predecessors:
- C1 #359 (closed). Its contract is [adr063-anchor-state-contract.md](adr063-anchor-state-contract.md), its oracle is `src/inference/anchor_contract.rs`, and the C1 final verdict carries O1 over to C2.
- ADR-065 P4a #526 and P4b #527, designed in [adr065-p4-design.md](adr065-p4-design.md). P4b itself requires B1 #503 and J1 #512 ([adr065-j1-design.md](adr065-j1-design.md)).

Story type: qualified implementation with hardware gates. It is built regardless of measured divergent-turn demand (operator decision, 2026-11-03).

Authored against `main` `d817dffa7371a2db85a43ed1d6502350906c33f5`. Source identities are listed at the end. Line numbers help reading only.

This is a design. Nothing was built, run or measured for it. Byte figures are arithmetic from the artifacts' published geometry. Anything else that is not read from source is marked [INFERENCE].

## 1. Acceptance map

| C2 acceptance item | Where this design meets it |
|---|---|
| Bounded slot-local anchors with pending publication, atomic restore, lineage pruning, saved-logit equality hits, right-sized payloads and owned-byte telemetry. The recurrent restore builds on P4a/P4b | §3.1 store, §3.6 publication, §3.7 atomic restore, §3.5 lineage, §3.4 equality hits, §3.2 payload, §3.9 telemetry, §3.3 restore via P4a/P4b |
| An approved HLD/LLD states how anchor-snapshot→state restore reuses P4a, whose contract covers capture→state only. If an extension is needed, link the mlx-native issue | §3.3. **No extension is needed.** The anchor payload *is* a capture image (`T = S = 1`), which is inside P4a's contract (P4 design §2.1, "Anchor images"). The reverse direction, the snapshot, uses the same existing primitives (blit and strided copy), whose snapshot-direction parity is added to the merged P4a/E6 qualification record (P4 design §2.4). Neither needs a native contract change |
| Cold-versus-resumed exactness on both Qwen artifacts with graft absent, present and identity-changed; speculation off and auto; cancellation; failed restore; four concurrent clients | §7.2 exactness matrix |
| O1 success-path twin: a cold (non-restore) divergent prefill completes and then refuses the older anchor whose rows it overwrote | §7.3 |
| Zero `NotEligible` caused by anchor capture or restore (reason-tagged counter); joint eligibility rate with anchors on within the Q3 CI of anchors off | §3.8, §8.5 (paired interval) |
| Frozen limits: divergent-turn worst-slot semantic TTFT CI excludes 0 (improving); append-only makespan, worst-slot TTFT p95 and SSE-gap max ratios ≤ 1.025; footprint ≤ 1.25× a fresh Q5 calibration | §8.2–§8.4, decisions D2–D3 (§10) |
| Fresh Q5 calibration for both Qwen artifacts with the declared anchors per slot precedes any C2 hardware run | §8.1 |
| Qwen3.6 judged by bitwise exactness and `ExactTranscripts`; Qwen3.8 also passes the paired task-quality gate | §7.2, §7.5, §8.6, open decision D1 (§10) |
| Primary Qwen3.8 evidence on the vanilla official 27B; grafted variants are separate arms; Gemma and DeepSeek are successors | §8 (case files), §9 |

## 2. Current state (what C2 changes)

| Concern | Today | Source |
|---|---|---|
| Anchors per slot | One `Option<Qwen35PromptAnchor>` | `src/serve/api/engine.rs:18132-18139` |
| Anchor payload | `HybridKvSlotAnchor`: per-layer full-attention cursors, MTP cursor, and **host `Vec<u8>`** copies of the slot's current conv and recurrent regions. No full-attention K/V copy; the live append-only rows are a dependency | `src/inference/models/qwen35/kv_cache.rs:1017-1038` |
| Snapshot | `snapshot_slot_anchor`: CPU `copy_slot_region_out` of the parity-current buffers | `kv_cache.rs:1555-1626`, `1404-1426` |
| Restore | `restore_slot_anchor`. It validates *and* rewinds the full-attention cursors inside one loop, then checks MTP, then CPU-copies the linear state into the named buffers and sets `pp_flipped[slot] = false`. **Not preflight-all-then-mutate** (C1 plan: "Current Qwen restore interleaves validation/mutation") | `kv_cache.rs:1631-1718` |
| Validity | `live_cursor >= saved_cursor`. This does not prove lineage after a divergent overwrite (C1 contract, "Identity and coordinates") | `kv_cache.rs:1663-1678` |
| Capture points | (a) The stable prompt boundary at a bounded-prefill chunk end (`engine_qwen35.rs:4316-4339`), **installed immediately** via `install_qwen35_stable_prompt_checkpoint` (`engine.rs:18181-18199`, called at 20276 and 20321). (b) The prompt end, when no stable boundary exists (`engine.rs:20493-20524`) | |
| Selection | `qwen35_slot_affinity` picks the most cached tokens over live retained prefix, the single anchor and active prompt (`engine.rs:18249-18321`). `use_prefill_logits` is set on exact equality (18279) | |
| Restore sites | Admission (`engine.rs:19915-19946`; on error it resets and fails the request). Cancellation recovery (`engine.rs:17517-17621`; on restore error it resets) | |
| Invalidation | Cold admission clears ledger and anchor (`engine.rs:19910-19913`). Embedding clears them (20708-20709). Reset paths clear them | |
| Joint eligibility | `run_qwen35_joint_decode` returns an untagged `NotEligible` when `kv.la_capture_active()` (cache-global, `qwen35_joint_decode.rs:278-280`) or when fewer than 2 lanes are `joint_decode_ready` (`:281-293`; `engine_qwen35/joint_decode.rs:55-73`) | |
| Success publication | `settle_qwen35_decode_lane`, `tick.finished && !client_dropped` (`engine.rs:20981-21010`) | |

## 3. HLD

```mermaid
flowchart TD
  A[admission: prompt P] --> S{"select deepest valid:<br/>live prefix / committed anchor / active"}
  S -- "anchor a (depth d)" --> PF["preflight restore ticket<br/>(no mutation)"]
  PF -- refuse --> COLD[explicit cold path]
  PF -- ok --> TB["tombstone anchors deeper than d<br/>+ pending (before any write)"]
  TB --> GR["P4b plan/encode_la_restore(source = anchor image)<br/>route Gpu: one command buffer, supervised wait<br/>route Host: host copy"]
  GR -- ok --> CUR["cursors := saved, parity := false,<br/>ledger := anchor tokens, spec := anchor spec"]
  GR -- "non-fatal error (encode)" --> POI["poison store, reset slot → cold"]
  GR -- "CommandBufferError" --> FAT[worker fail-stop]
  S -- "live L" --> PRE
  S -- none --> RST["reset slot; tombstone all"] --> PRE
  CUR --> EQ{"P == anchor tokens?"}
  EQ -- yes --> LOG["saved-logit equality hit<br/>(no prefill)"]
  EQ -- no --> PRE["suffix prefill (2K chunks)"]
  PRE --> CAP["stable boundary or prompt end:<br/>snapshot (route) → pending"]
  LOG --> DEC
  CAP --> DEC[decode / joint decode]
  DEC -- "finished && !dropped" --> COM["pending → committed;<br/>keep newest K"]
  DEC -- "cancel / error" --> DIS["discard pending;<br/>recover to deepest valid committed"]
```

### 3.1 Per-slot anchor store (mirrors the C1 oracle)

`Qwen35AnchorStore` replaces `Option<Qwen35PromptAnchor>` per slot. Its fields:
- `committed: Vec<Qwen35Anchor>` (ascending depth, `len ≤ K`);
- `pending: Option<Qwen35Anchor>`;
- `lineage_generation`, `epoch`, `poisoned`;
- an owned-byte ledger.

Its transitions are those of `anchor_contract::Store` (`src/inference/anchor_contract.rs:149-532`): `begin_capture`/`finish_capture`, `stage` (pending), `complete` (commit or discard), `cancel`, `preflight` → ticket, `apply`, `before_writes`, `partial_failure`, `replace_identity` and `reset`.

C2 does not reuse the oracle type in production, because it is a test oracle over CPU byte vectors. A differential test drives both with the same transition script and compares observable outcomes (§7.4).

**Depth and kill switch.** `K` (committed anchors per slot) comes from `HF2Q_QWEN_ANCHOR_DEPTH`. It is read once at worker construction, accepts 1–8, and invalid values fail closed. Pending is excluded from `K` (C1 contract).
- **`K = 1` is the kill-switch value** and the anchors-off arm of every C2 campaign. It reproduces today's single-anchor selection through the same store. It is not byte-for-byte the pre-C2 code path: pending publication (§3.6), the cache-side generation binding (§3.5) and atomic restore (§3.7) still apply at `K = 1`. Those differences are covered by the refactor gate (§7.5) and the cancel smoke (§8.6).
- **Shipped default when unset**, per artifact family:
  - `K = 1` on merge, until C2's qualification verdicts exist;
  - `K = 4` for an artifact only when every C2 gate passes **for that artifact**: the §7.2 exactness matrix, the frozen limits of §8.3–§8.4, the eligibility rule of §8.5, footprint ≤ 1.25×, and quality (Qwen3.8: the paired task-quality gate; Qwen3.6: the exactness and transcript gates, §8.6);
  - otherwise that artifact stays at `K = 1`, and its verdict is recorded.

  The default is a per-variant constant (dense / MoE) changed in a separate commit that cites the verdict receipts. The worker logs the resolved `K` in its startup line.

### 3.2 Anchor payload, right-sized

`Qwen35Anchor` contains:

| Field | Content |
|---|---|
| `prompt_tokens: Vec<u32>` | Logical tokens; graft excluded, as today |
| `kv: HybridKvSlotAnchor` | Now holds per-layer full-attention cursors, the MTP cursor, and **one `LaAnchorImage`**: a single shared-storage `MlxBuffer` of exactly `L·(E + C·Km1)·4` bytes (`Km1` = conv width − 1 = 3). Layer `ℓ`'s recurrent image sits at `ℓ·(E + C·Km1)·4`, and its conv image follows immediately in **capture layout** `[Km1, C]` |
| `prefill_logits: Vec<f32>` | Exactly `vocab` entries, as today |
| `vision_fingerprint` | As today |
| `spec: Option<Qwen35SpecPrefixBoundary>` | `pending_target_hidden` is copied into a dedicated `hidden_size·4`-byte buffer (20,480 B on Qwen3.8). Retaining a view of a chunk-sized parent is refused (C1 "Byte ownership") |
| `identity: Qwen35AnchorIdentity` | §3.5 |

The ledger asserts `byte_len() == data_byte_len()` for every anchor-owned `MlxBuffer`. A view pinning a larger parent fails capture, and the anchor is skipped with reason `oversized_owner`.

Sizes per anchor (arithmetic, from P4 design §1.2):

| Artifact | DeltaNet image | Logits | Hidden | Tokens |
|---|---|---|---|---|
| Qwen3.8-27B | 156,893,184 B (149.6 MiB) | 248,320 × 4 = 993,280 B | 20,480 B | 4·len |
| Qwen3.6-35B-A3B | 65,863,680 B (62.8 MiB) | `vocab·4` | `hidden_size·4` | 4·len |

Today's single CPU anchor holds the same 149.6 MiB per slot in `Vec<u8>`s. C2 moves it to one shared-storage `MlxBuffer`, on both routes (§3.3). At `K = 4` with four slots, Qwen3.8 retains up to 16 committed images plus up to 4 pending: 20 × ≈150.6 MiB ≈ 2.94 GiB, against ≈0.59 GiB today. Q5 must admit this (§8.1).

### 3.3 Anchor snapshot → state restore via P4a/P4b

This answers the C2 HLD/LLD acceptance item.

**Restore.** A `LaAnchorImage` layer is, byte for byte, a capture row with `T = S = 1`:
- recurrent `[E]`;
- conv `[Km1, C]`, channels innermost, the layout `ssm_conv_capture_forward_f32` writes (mlx-native `src/shaders/ssm_conv_capture.metal:21-30`).

Restoring slot `s` from an anchor is therefore the P4a operation `DeltaNetCaptureRestore` with `rec_cap`/`conv_cap` = the image's layer views, `(s_j, t_j) = (0, 0)`, and destinations = slot `s`'s regions of the **named** `recurrent`/`conv_state` buffers. P4a §2.1 explicitly admits this geometry, and T-P4a-4 tests it. After completion, `pp_flipped[s] := false`, exactly as `restore_slot_anchor` does today (`kv_cache.rs:1715`). The restored bytes and parity therefore equal today's host restore.

P4b's hf2q interface (P4 design §3.1) is used, not duplicated. Its plan item carries a source and a destination policy:

```rust
pub enum LaRestoreSource<'a> { CaptureRow(u32), AnchorImage(&'a LaAnchorImage) }
pub enum LaRestoreDestination { CurrentByParity, NamedResetParity }
pub struct LaRestoreItem<'a> { pub slot: SlotId, pub source: LaRestoreSource<'a>,
                               pub destination: LaRestoreDestination }
```

- Rollback uses `(CaptureRow(row), CurrentByParity)`.
- Anchor restore uses `(AnchorImage(img), NamedResetParity)`.
- Both lower to the same `plan_la_restore` / `encode_la_restore` and therefore the same P4a composition (blits plus strided copies; P4 design §2.2).

That is the single restore path ADR-063 requires.

**Route.** Anchor snapshot and restore follow the cache's P4b route, `la_restore_route` (P4 design §3.1). That route is resolved from `HF2Q_QWEN_LA_GPU_RESTORE` and the artifact family, and it is `Host` for Qwen3.6 unless overridden. C2 adds no switch of its own:
- **`Gpu`:** the GPU snapshot and restore described here.
- **`Host`:** `plan_la_restore` lowers the same items to host copies into and out of the same shared-storage image, with the identical layout and byte results. The recurrent region is a straight copy, and the conv region uses the same re-index as `rollback_la_to` (`kv_cache.rs:3016-3022`) and its inverse. These host functions replace `copy_slot_region_out`/`copy_slot_region_in` (`kv_cache.rs:1404-1457`), which lose their callers.

So with P4b's switch off (its default if its A/B verdict is not Win or Equivalent), anchors keep a host path, and no GPU anchor copy runs. Either route uses one image format, so an anchor snapshotted on one route restores correctly on the other. That property is tested (U1).

**Snapshot (state → image), route `Gpu`.** This direction is outside P4a's §2.1 contract but uses the same existing primitives, whose snapshot-direction parity is part of the merged qualification record (P4 design §2.4, T-P4a-5):
- **Recurrent:** `CommandEncoder::blit_copy_bytes` from the slot's parity-current recurrent region to the image. A blit is a byte copy by Metal definition (mlx-native `src/encoder.rs:1563`); the layouts are identical, so no re-index is needed.
- **Conv:** one `dispatch_strided_copy_f32` per layer with `rows = Km1`, `cols = C`, `stride_row = 1`, `stride_col = Km1`, computing `img[k·C + c] = state[c·Km1 + k]`, the inverse of the restore re-index. E6 showed the strided copy is bitwise for every u32 class on both metallib paths. There is no host-transpose fallback on the `Gpu` route.
- **Spec hidden:** one blit into the dedicated hidden buffer.

Encoding order matches P4b: all strided copies in one compute pass, then the blits. All of this goes into one command buffer, committed and waited under a supervisor lease ("Qwen35 anchor snapshot"). The snapshot runs at the same two capture points as today, at a point where the slot's prefill chunk has completed and been validated (`engine_qwen35.rs:4302-4304`). Every source is therefore final.

If a future owner prefers a native state→image op, its contract is §2.1's with source and destination roles exchanged. It would need its own mlx-native issue. C2 does not require it.

### 3.4 Selection: deepest valid, equality hits

For each slot, `store.deepest_prefix_of(prompt)` returns the committed anchor with the largest `prompt_tokens.len()` such that:
- `prompt.starts_with(anchor.prompt_tokens)`;
- the identity matches the current request (vision fingerprint, graft identity, effective config);
- the anchor is not tombstoned.

`qwen35_slot_affinity` keeps its tuple shape. The anchor element becomes that deepest match, and `Qwen35SlotPreference` gains `anchor_id: Option<u64>` so that admission restores exactly the anchor the planner chose. A mismatch at admission is an invariant failure, as today's `expect` at 19919.

Ties keep today's order: live prefix (no restore) beats an anchor of equal depth, and an idle slot beats an active one.

**Equality hit.** If `prompt == anchor.prompt_tokens`, restore and then seed decode from the saved `prefill_logits` (and the spec hidden row), skipping prefill. This is today's `use_prefill_logits` behaviour, now available at any committed depth.

### 3.5 Identity and lineage pruning

`Qwen35AnchorIdentity` binds:
- the artifact digest;
- the tokenizer and template fingerprints (the readiness `measurement_snapshot` domains);
- the effective cache configuration: TQ KV mode and codebook bits, `max_seq_len`, `n_seqs`, MTP presence and layer types;
- the **cache-side slot state generation** `HybridKvCache::slot_state_generation[slot]` and the store's `lineage_generation`;
- graft generation, digest and length;
- the vision fingerprint.

**Slot state generation.** C2 adds `slot_state_generation: Vec<u64>` to `HybridKvCache`. It advances only when the slot's physical state is discarded or replaced:
- `reset_for_slot` (`kv_cache.rs:3108`), which covers cold admission, failure resets and embedding;
- `splice_graft_for_slot` (graft replant);
- `drop_seq` (`kv_cache.rs:4534`), and `fork_seq` for its destination slot. Both are the `MultiSeqKvCache` operations that replace a slot's contents wholesale.

It does **not** use `SlotHandle.generation`, which the scheduler advances on every release (`src/serve/scheduler.rs:239-242`, `957`, `1588`, `1834`). Binding that would make every anchor from an earlier request fail its identity check, so multi-depth reuse would never hit. Advancing `slot_state_generation` clears the slot's whole store before the next write, matching C1 rule 7 and the oracle's `replace_identity` (`anchor_contract.rs:474-505`).

**The `conversation` field is deliberately omitted.** C1's normative identity includes `conversation` (`anchor_contract.rs:15`). Qwen has no wire-level conversation id. Isolation is exact token-prefix identity within one slot lineage, so two clients with byte-identical prefixes legitimately share exact bytes, as they do today through the live retained prefix. This departs from C1's identity list and **needs owner sign-off (open decision D4, §10)**. A mismatch on any bound field refuses before any mutation.

**Lineage invariant.** Every committed anchor of a slot is a prefix of the slot's live token ledger, and no write since its capture touched a row below its depth.

Qwen writes are append-only at the live cursor, apart from reset, embedding and the graft splice. So the invariant is maintained by one rule: **whenever a slot's live cursor is set to `c`, tombstone every anchor (committed and pending) with depth > `c` before any GPU work for that slot is encoded.** The cursor is set to `c` by:
- an anchor restore to depth `c`;
- a reset, including the cold-admission reset and failure resets (`c` = 0, graft rows aside). This also advances `slot_state_generation`;
- an embedding (a reset);
- a graft replant, which advances `slot_state_generation` and bumps `lineage_generation`.

K3 commit truncation never goes below the prompt end, so it never crosses an anchor. That is asserted.

This is C1 rule 5 ("tombstone every descendant … before the FIRST possible divergent write"), applied conservatively at the cursor change rather than at the write. Pruning before a write that never executes is allowed by C1.

### 3.6 Publication

- **Capture → pending.** At a capture point (§3.3), the snapshot is staged as `pending`, replacing any earlier pending anchor of the same request. A pending anchor is invisible to selection, cancellation recovery and telemetry hit counts. Today's immediate install at `engine.rs:20276/20321` and `20517` becomes `store.stage(...)`.
- **Pending → committed** exactly once, in `settle_qwen35_decode_lane` after the retained ledger publication for `tick.finished && !client_dropped` (`engine.rs:20981-21010`) and before the success reply is fired. Then keep the newest `K`. Newest equals deepest in a lineage, because any shallower re-capture after a restore has already tombstoned the deeper ones. Eviction drops the oldest payload; its `MlxBuffer` is released at its last owner. Restores are synchronous, so no reader outlives the restore call.
- **Cancellation, client drop, error, poison:** discard pending. Recovery in `recover_qwen35_slot_after_cancellation` (`engine.rs:17517`) keeps today's order:
  1. **The preserve-unmodified-live-prefix branch (`engine.rs:17532-17555`) is kept unchanged.** If the live cursor still equals the retained ledger, nothing is restored and no anchor is tombstoned.
  2. Otherwise, restore the deepest valid *committed* anchor at or below the live cursor through the atomic protocol (§3.7).
  3. Otherwise, reset.

This is a deliberate behaviour change, and it needs owner acknowledgment (decision D5, §10), not just a risk entry. Today a cancelled request's own stable checkpoint is already installed and can be restored by a retry. Under C1's publication rule ("Cancellation never publishes pending anchors", rule 3), it is discarded, so a cancel-and-retry by an agent client recovers to the previous committed depth. The cancel smoke measures retry TTFT and recovered depth (§8.6).

### 3.7 Atomic restore (preflight-all-then-mutate)

**Phase A, preflight; nothing mutates.** It produces an immutable `SlotAnchorRestorePlan` bound to `(store epoch, anchor id)`. It checks:
- the store is not poisoned, and the identity equals the current one;
- the anchor is committed and live, and its depth is at most the live cursor of every full-attention layer and of the MTP cursor (presence matches);
- the anchor is a prefix of the live ledger (the lineage invariant);
- no open speculative or capture transaction: `!kv.la_capture_active()`, and the slot is idle in admission or recovery with no `pending_speculation_output`;
- full-attention and linear layer counts, and the image extent equals `L·(E + C·Km1)·4`;
- `plan_la_restore` for one `(AnchorImage, NamedResetParity)` item, which preflights every range against `data_byte_len()` (P4 design §3.3).

**Phase B, mutate, in this order:**
1. `store.before_writes(depth)`: tombstone anchors deeper than `d`, and the pending anchor.
2. Encode the restore. On route `Gpu`, commit and wait under a supervisor lease ("Qwen35 anchor restore"); on route `Host`, run the host copy.
3. Set every full-attention cursor and the MTP cursor to the saved values, and set `pp_flipped[slot] := false`.
4. Set `retained_tokens := anchor.prompt_tokens` and `spec := anchor.spec` (validated by `validate_speculative_cursors_for_slot`, as today at `engine.rs:17573-17597`).
5. `epoch += 1`.

Phase B steps 3–5 cannot fail after the preflight: they are plain field writes with pre-validated indices.

**Failures.** On route `Gpu`, a non-fatal failure can only occur during encoding (pipeline lookup or argument validation), before the command buffer is committed. Any failure after commit surfaces as `MlxError::CommandBufferError`, which is fatal.

| Failure | Handling |
|---|---|
| Preflight refusal | No mutation. Counted as `restore_refused{reason}`. The request takes the explicit cold path: reset, which tombstones everything, then full prefill |
| Non-fatal error in step 2 (encode, before commit; or a host-route error) | Step 1 already tombstoned, so conservatively poison and clear the whole store (no shallower fallback; C1 rule 6). Then `reset_for_slot`, clear the ledger and serve the request cold. Cold output must equal the cold twin (§7.2, "failed restore") |
| `MlxError::CommandBufferError` in the error chain (after commit) | The worker fails stop through the existing fatal path. The error is wrapped with `.context`, never stringified |

`restore_slot_anchor` is replaced by `plan_slot_anchor_restore` + `apply_slot_anchor_restore`. All callers move: `engine.rs:17560`, `19924`, `speculation_qualification_tests.rs:716`, and the `slot_anchor_*` tests in `kv_cache.rs:12323`, `12940`.

### 3.8 Joint-decode eligibility protection

**Structural guarantee.** No C2 operation arms DeltaNet capture: snapshot reads state, and restore reads an image. Every C2 GPU operation runs to completion (commit-and-wait) inside admission, prefill-completion or settle code, never between `run_qwen35_joint_decode`'s eligibility check and its forward. A lane is therefore never observed by `joint_decode_ready` in the middle of an anchor operation.

**Reason-tagged refusal.**
- `Qwen35JointRun::NotEligible` (`qwen35_joint_decode.rs:233-241`) becomes `NotEligible(Qwen35JointIneligible)`:
  - `CaptureActive(LaCaptureOwner)`;
  - `TooFewReady { ready, refusals: [Qwen35LaneRefusal; ≤4] }`;
  - `DuplicateSlots`;
  - `MaxRowsBelowTwo`.
- `joint_decode_ready` (`engine_qwen35/joint_decode.rs:55-73`) becomes `joint_decode_readiness(&self, kv) -> Result<(), Qwen35LaneRefusal>`:
  - `StepBound`, `PendingSpeculationOutput`, `NoSeed`, `TargetCursor`, `MtpCursor`;
  - plus `ClientClosed` and `HandleMismatch` from the partition closure;
  - plus `AnchorTransaction` when the slot's store has an open ticket or construction.
- The cache-global `la_capture_active_tokens: Option<u32>` (`kv_cache.rs:928`) gains an owner tag, `la_capture_owner: Option<LaCaptureOwner>`, with values `K3Verify`, `HistoryVerify`, `RecoveryCapture` and `AnchorSnapshot`. `ensure_la_capture` callers pass their owner. C2 itself never uses `AnchorSnapshot`; it exists so that any future capture-based snapshot is attributed. This is compatible with J2, which replaces the global flag with per-slot ownership and keeps the owner tag.

**Counter.** `hf2q_qwen_joint_not_eligible_turns_total{reason}` is a static atomic set exported in `/metrics` beside the speculation counters (`src/serve/api/handlers.rs:8657-8700`). A NotEligible turn increments exactly one reason:
- `capture_<owner>`;
- `too_few_ready`;
- `duplicate_slots`;
- `max_rows`;
- `anchor_capture` or `anchor_restore`, assigned when any lane refusal is `AnchorTransaction` or the capture owner is `AnchorSnapshot`.

This attribution covers only **open** anchor transactions. It is not conservative for after-effects. A lane refused after a completed restore or equality hit, with reasons such as `NoSeed`, `TargetCursor` or `MtpCursor`, is tagged with that reason, not `anchor_*`. A restore that left a lane unready would therefore not show in the `anchor_*` counter. Two further checks close that gap:
- the readiness test (U9, §7.1): after a restore or equality hit, a lane's `joint_decode_readiness` equals that of its cold twin at the same ledger;
- the paired eligibility-rate rule (§8.5), which catches any rate loss regardless of tag.

A second counter set, `hf2q_qwen_joint_lane_refusals_total{reason}`, counts lane refusals. The acceptance reads `anchor_capture + anchor_restore == 0` over every C2 run. The debug line `Qwen joint decode turn` gains `reason=` so that `cancel_smoke.py widths` and E4's extractor can join it.

### 3.9 Owned-byte telemetry and quota

The per-slot ledger has classes `payload_image`, `logits`, `hidden`, `tokens` and `metadata`, for committed and for pending, with the C1 alias rule: unique owner extents are charged once.

Exported:
- `hf2q_qwen_anchor_owned_bytes{slot, class, state}`;
- `hf2q_qwen_anchor_committed{slot}`;
- `hf2q_qwen_anchor_restores_total{rank}` (rank 0 = deepest);
- `hf2q_qwen_anchor_equality_hits_total`;
- `hf2q_qwen_anchor_restore_refused_total{reason}`;
- `hf2q_qwen_anchor_tombstoned_total{cause}`;
- `hf2q_qwen_anchor_evicted_total`;
- `hf2q_qwen_anchor_capture_skipped_total{reason}`;
- sums and counts of snapshot and restore milliseconds.

The idle conservation check (C1) holds: ledger total == Σ unique anchor owners. Pending and construction are 0 at idle.

**Quota.** The Q5 case declares `anchors_per_slot` and `anchor_bytes_per_slot_max`. If a capture would exceed the slot's declared reservation (K committed + 1 pending), it is skipped with reason `quota` and nothing is allocated. Eviction happens only at commit. There is no implicit growth beyond the declaration.

The same Q5 case also declares the K3 DeltaNet **capture footprint** whenever `HF2Q_QWEN_SPECULATION=auto` is in effect: ADR-065 estimates about 2.25 GiB for four slots of four rows (`[Dk, Dv, nv, n_tokens, n_seqs]` f32 × 48 layers, ADR-065 "Negative"). C2 adds no capture. That footprint coexists with the anchor reservation, and Q5 must admit both.

## 4. LLD

### 4.1 Files

| File | Change |
|---|---|
| `src/serve/api/engine_qwen35/anchor_store.rs` (new) | `Qwen35AnchorStore`, `Qwen35Anchor`, `Qwen35AnchorIdentity`, ledger and telemetry statics |
| `src/inference/models/qwen35/kv_cache.rs` | `HybridKvSlotAnchor` holds `LaAnchorImage`. `snapshot_slot_anchor` snapshots through the P4b route (§3.3). `restore_slot_anchor` is replaced by `plan_slot_anchor_restore` / `apply_slot_anchor_restore`. `slot_state_generation`. `la_capture_owner` tag. `copy_slot_region_out` and `copy_slot_region_in` are replaced by the host-route image functions; a test-only port of them is kept in the test module as the U1 oracle |
| `src/inference/models/qwen35/kv_cache/la_restore.rs` (P4b) | `LaRestoreSource` and `LaRestoreDestination` |
| `src/serve/api/engine.rs` | `prompt_anchors: &mut [Option<Qwen35PromptAnchor>]` becomes `anchor_stores: &mut [Qwen35AnchorStore]` at every signature: 17523, 17629, 18183, 19696, 20174, 20442, 20664, 20758, 20895, plus `run_qwen35_joint_decode`. Selection (18249, 18675, 19764), admission restore (19915), cancellation recovery (17557), capture install (18181, 20493-20524), success commit (20981-21010), invalidation sites (19910, 20708 and resets) |
| `src/serve/api/engine/qwen35_joint_decode.rs`, `engine_qwen35/joint_decode.rs` | Tagged `NotEligible` and lane readiness reasons (§3.8) |
| `src/serve/api/handlers.rs` | `/metrics` families (§3.8, §3.9) |
| `scripts/four_agent/coding_divergent_edit.py` (new), `q5_case.py`, `q5_admission.py`, `quality_campaign.py` | The divergent-edit workload variant (§8.2), the anchor declaration in Q5 cases, and variant + instance composition for the paired quality gate |
| `docs/operating-kv-cache.md` | Operator notes for `HF2Q_QWEN_ANCHOR_DEPTH`, metrics and quota |

### 4.2 Store API

```rust
impl Qwen35AnchorStore {
    pub fn new(identity: Qwen35AnchorIdentity, keep: usize) -> Self;
    pub fn deepest_prefix_of(&self, prompt: &[u32], id: &Qwen35AnchorIdentity) -> Option<&Qwen35Anchor>;
    pub fn on_cursor_set(&mut self, cursor_tokens: usize, cause: TombstoneCause); // lineage rule §3.5
    pub fn stage(&mut self, request: u64, anchor: Qwen35Anchor) -> Result<(), CaptureSkip>;
    pub fn commit(&mut self, request: u64);      // exactly once; no-op if no pending of this request
    pub fn discard_pending(&mut self, request: u64);
    pub fn ticket(&self, anchor_id: u64) -> Result<AnchorTicket, RestoreRefusal>; // binds epoch
    pub fn poison_and_clear(&mut self);
    pub fn replace_identity(&mut self, identity: Qwen35AnchorIdentity); // graft/model/config change
    pub fn owned(&self) -> AnchorOwnedBytes;
}
```

`HybridKvCache` functions:

```rust
pub(crate) fn snapshot_slot_anchor(&self, slot, prompt_len, registry, supervisor_lease) -> Result<HybridKvSlotAnchor>;
pub(crate) fn plan_slot_anchor_restore(&self, slot, &HybridKvSlotAnchor) -> Result<SlotAnchorRestorePlan>;
pub(crate) fn apply_slot_anchor_restore(&mut self, plan: SlotAnchorRestorePlan, registry) -> Result<()>;
```

`apply_slot_anchor_restore` consumes the plan, so a plan cannot be applied twice. The plan holds the anchor's epoch, and the engine refuses a stale ticket.

## 5. Data flow per request (slot `s`)

1. **Admission.**
   - Select the preference (§3.4).
   - **Anchor hit:** ticket, then preflight (Phase A), then `on_cursor_set(d)`, then Phase B. On equality, seed from the saved logits.
   - **Live hit (L):** no mutation. Anchors are untouched, because L equals the live cursor.
   - **Miss:** `reset_for_slot`, then `on_cursor_set(graft_len)`.
2. **Prefill.** Suffix chunks at today's 2K quantum (`engine.rs:17920`). At the stable boundary or prompt end, snapshot on the cache's route → `stage`.
3. **Decode.** Serial or joint. K3 rollback uses P4b (`CaptureRow`, route-dispatched). Anchors are not touched.
4. **Settle.**
   - Success: `commit`.
   - Cancellation or drop: `discard_pending`, then recovery (atomic restore of the deepest valid committed anchor at or below the live cursor, else reset).
   - Error: `discard_pending`, then the existing reset (which tombstones).

## 6. Failure semantics summary

| Event | Store | Slot | Request |
|---|---|---|---|
| Capture refused (quota, oversized owner, identity) | Unchanged; skip counted | Unchanged | Continues |
| Snapshot encode error (route `Gpu`, before commit) or host-route error | Pending dropped (nothing published) | Unchanged (snapshot only reads) | Continues |
| Snapshot `CommandBufferError` | — | — | Worker fail-stop |
| Restore preflight refusal | Unchanged | Reset (cold) | Continues cold |
| Restore encode error (route `Gpu`, before commit) or host-route error | Poisoned and cleared | Reset | Continues cold |
| Restore `CommandBufferError` | — | — | Worker fail-stop |
| Cancellation before or after capture | Pending discarded; deeper anchors tombstoned at any cursor rewind | Recovered or reset | Cancelled |
| Graft, model, config or `slot_state_generation` change | `replace_identity` clears it before any new write | — | — |

## 7. Tests

### 7.1 Unit (CI, Metal tiny configs under `hf2q_gpu_test_lock`)

| ID | Test |
|---|---|
| U1 | Image round trip on both routes. Snapshot then restore equals a **test-only port** of today's host `copy_slot_region_out`/`copy_slot_region_in` (kept in the test module after production loses them), for both parities and every slot, with every u32 bit class in the state. Peer slots are byte-identical. Cross-route cases: snapshot `Gpu` → restore `Host`, and snapshot `Host` → restore `Gpu` |
| U2 | Restore atomicity: inject a failure at the last layer's preflight; cursors, bytes, parity and store are unchanged. Inject a failure **during encoding**, after step 1's tombstones and before commit (the only reachable non-fatal Phase B failure on route `Gpu`); the store is poisoned, the slot reset, and no shallower anchor is attempted. An injected post-commit `CommandBufferError` reaches `is_fatal_command_buffer_error` unstringified |
| U3 | Lineage rule: A@d1 < B@d2 < C@d3. Restore B, and C is tombstoned before encode. A cold reset tombstones all. The pending anchor is tombstoned by a cursor rewind below its depth |
| U4 | Publication: pending is invisible to selection and recovery; commit happens exactly once; a duplicate completion is a no-op; cancellation discards; keep-newest-K evicts the shallowest |
| U5 | Right-sizing: a spec hidden row given as a view of a chunk parent is copied, never retained. A payload with `byte_len != data_byte_len` is refused with `oversized_owner` |
| U6 | Ledger conservation at idle with K = 4, pending, eviction, poison and reset (C1 "Byte ownership") |
| U7 | `qwen35_slot_affinity` with multiple anchors: deepest wins; live beats equal depth; equality sets `use_prefill_logits`. The existing tests at `engine.rs:45403-45600` are kept and extended |
| U8 | Joint reason tagging: synthetic slot states for each `Qwen35LaneRefusal`; the counter increments exactly one reason per NotEligible turn; `anchor_*` stays 0 when anchor transactions are closed and becomes 1 under an injected open ticket (the counter is not vacuous) |
| U9 | Readiness after restore: for a slot restored from an anchor (and for an equality hit), `joint_decode_readiness` after admission equals the result for the same slot built cold to the same ledger, including the MTP cursor with speculation `auto` |
| U10 | Slot state generation: `reset_for_slot`, `splice_graft_for_slot`, `drop_seq` and `fork_seq` (destination) advance it and clear the slot's store. Releasing and re-admitting a `SlotHandle` (a new `SlotHandle.generation`) does not, and a committed anchor from the earlier request is still selectable |

### 7.2 Cold-versus-resumed exactness matrix (real artifacts; GPU lane)

**Definition.** "Cold" is the ADR-063 Q1 Class A same-route twin (`docs/research/adr063-q1-numeric-policy.md` §2, row A: "resume vs cold at the same chunk boundaries; cancel+restore vs never-cancelled"). A fresh slot replays exactly the transaction sequence that produced the anchor's state up to depth `d`: the same prefill chunk boundaries, the same decode or K3 steps. It then runs the resumed request's suffix transactions with identical boundaries, but never executes the diverging history in between.

"Resumed" executes that history, then the divergent turn that restores the anchor, then the same suffix.

Bitwise equality is required for:
- every logits row, hidden row and token id;
- target and MTP cursors, MTP K/V region and TQ/K/V rows below the cursor;
- DeltaNet current state and parity;
- the retained ledger.

The comparison uses Q1 stores (`hf2q.adr063.invariance`) and `scripts/four_agent/q1_decision_policy.py` → `ExactEqual`. A run from 0 with different chunk boundaries is a Class B alternate route, not this test.

**Harness.** An ignored real-artifact test module, `src/serve/api/engine_qwen35/anchor_exactness_tests.rs`, scripts token-level turns through the slot-aware engine functions (pattern: `speculation_real_artifact_tests.rs`). It records Q1 stores at each transaction.

| Axis | Values |
|---|---|
| Artifact | Qwen3.8-27B vanilla (`HF2Q_QWEN38_GGUF`), Qwen3.6-35B-A3B vanilla (`HF2Q_QWEN36_GGUF`) |
| Graft | absent; present (same graft for the twin); identity change between capture and restore. Expected: restore refused, cold path, output equal to the twin without the anchor |
| Speculation | `HF2Q_QWEN_SPECULATION=off`; `auto` (anchors carry the spec boundary; K3 rounds after resume) |
| Restore depth | rank 0 (deepest) and rank 2 of K = 4, plus an equality hit |
| Cancellation | cancel before capture, after capture before commit, and mid-decode after a resumed admission. Recovery state equals the never-cancelled twin at the recovered depth |
| Failed restore | injected preflight refusal and injected encode-time failure; output equals the cold twin, and the store is poisoned in the second case |
| Concurrency | four slots: slot 0 resumes while slots 1–3 decode (joint on, `HF2Q_QWEN_JOINT_DECODE` at its default). Slot 0 equals its single-slot twin by the existing per-slot joint parity (Q1 policy §2 row A, "Qwen four-slot per-slot parity"). Slots 1–3 state digests are unchanged across slot 0's restore (Class A: no peer mutation) |

### 7.3 O1 success-path twin (C1 carry-over)

1. Slot 0 serves request A to success, so anchor A@d is committed.
2. Request B, a different prompt with `|B| > d` and a shared prefix shorter than `d`, is admitted to slot 0 **cold** (miss: `reset_for_slot`). It completes successfully and writes rows `[0, |B|)`. The live cursor is now ≥ d, so today's `live >= saved` check would pass.
3. Request C, with `C.starts_with(A.tokens)`, must not restore A:
   - `restore_refused` and `tombstoned{cause=cold_reset}` are observed;
   - C's output and state equal the cold twin of C.

Variant 3': B restores a shallower anchor A0@d0 < d and prefill writes `[d0, …)`. C must miss A (tombstoned) and may hit A0.

Negative control (test-only fault injection disables `on_cursor_set`): C restores A, and the comparison against the twin reports `DivergedExactInvariant`. This shows the test can fail.

### 7.4 Differential against the C1 oracle

A transition script generator emits the same sequences of capture, stage, commit, cancel, before_writes, preflight, apply, partial_failure, identity replacement and reset. It drives `Qwen35AnchorStore` (with CPU-mocked payload owners) and `anchor_contract::Store`. Visibility, refusal reasons, tombstones, poison and owned bytes must match. This sits under the existing `anchor_c1` filter so CI runs it.

### 7.5 Serving-level transcript gates (hardware)

- **Refactor gate:** `HF2Q_QWEN_ANCHOR_DEPTH=1` on the C2 binary vs C2's parent commit (same native pin and case file), canonical workload. Expected `ExactTranscripts`: both routes are bitwise equal to the pre-C2 host copies, and the selection is identical. The one exception is the cancellation-pending change (§3.6), so cancel runs are excluded from this gate and judged by the cancel smoke (§8.6).
- **Anchors on vs off** (K = 4 vs K = 1, same binary) on the canonical append-only workload: `ExactTranscripts`, because the deepest valid reuse is the live prefix on every turn, so the routes are identical.
- **Divergent-edit variant:** decision D1 (§10). Every row of each divergent turn must be inside the Class B envelope, and K = 4 vs K = 1 transcripts are compared and classified.

Until the mlx-native runtime-registry fix is pinned (P4 design §2.3), every C2 hardware run with speculation `auto` uses the precompiled metallib.

## 8. Measurement plan

Every hardware run:
- needs the GPU lane granted by the root and the host lock;
- runs one model process, with peers unloaded and at least 60 s between loads;
- records source commit, binary SHA-256, artifact digest, native version and case file;
- passes Q5 `CalibrationPass`.

Evidence goes under `/Users/robert/.local/state/hf2q/v0130/c2-360/`.

### 8.1 Fresh Q5 calibration (precedes every C2 hardware run)

For both `q5_cases/qwen38.json` (vanilla 27B) and `q5_cases/qwen36.json` (vanilla 35B-A3B), derive C2 cases declaring `anchors_per_slot = 4` and the per-anchor bytes from §3.2, on both the canonical and the divergent-edit workloads. Cases that run speculation `auto` also declare the K3 capture footprint (about 2.25 GiB at four slots of four rows, §3.9). Run `scripts/four_agent/q5_calibration.py`. Admission is re-established on any binary, artifact, workload or footprint change (ADR-065 execution prerequisites).

### 8.2 Divergent-edit workload variant (C2 authors it)

The new `scripts/four_agent/coding_divergent_edit.py` follows the `coding_long_context.py` `Variant` mechanism: `select`, `derive_case`, the approval key `workload_variant = divergent-edit-v1`, and a fixture digest.

At turns 3 and 5 of every agent, the client applies a deterministic context edit before sending. It truncates the tool-result message of turn `t−2` to its first 16 lines plus a fixed elision marker. That divergence point lies inside turn `t−1`'s prompt: below the latest anchor (turn `t−1`) and above the turn-`t−2` anchor. Under K = 4 the deepest valid anchor is turn `t−2`; under K = 1 the turn is cold.

`estimate` reports per-turn wire bytes. The variant must compose with `--instance` so the paired quality gate can run on it (`quality_campaign.py` currently refuses variants, at line 126).

### 8.3 Divergent-turn worst-slot semantic TTFT (frozen limit 1)

- Arms on the same binary: K = 4 vs K = 1.
- Q3 Williams-block design (`q3_stats.williams`/`block_order`) with Q3's pre-registered **sequential precision stop** (`q3_stats.stop_decision`): minimum 5 blocks (`MIN_BLOCKS`), maximum 20 (`MAX_BLOCKS`). Stop when every required contrast of this campaign (the divergent-turn TTFT effect and, in §8.4's campaign, the three append-only ratios) has a 95% half-width ≤ 2.5 points (`HALF_WIDTH_PP`). There is no fixed block count and no other peeking. If the owner instead wants a fixed count, it is chosen from a separately labelled pilot whose runs are never pooled.
- Estimand per run: the worst slot's semantic TTFT over turns 3 and 5, from each turn's `semantic_timing_v3.semantic_ttft_seconds` (the field `scripts/four_agent/q3_metrics.py:82-91` reads). `q3_metrics.slot_metrics` pools all turns, so a turn-filtered variant is added beside it.
- Paired effect: `q3_stats.paired_effect` over the blocks (K = 4 subject, K = 1 reference, lower is better). **Pass:** the 95% interval excludes 0 in the improving direction at the stop. A `max_blocks` stop without that is a fail.

### 8.4 Append-only limits and footprint (frozen limits 2–3)

- Canonical workload, K = 4 vs K = 1, the same Williams-block design and sequential stop.
- Four-agent makespan, worst-slot TTFT p95 (`q3_metrics` worst-slot p95 semantic TTFT) and `worst_slot_sse_gap_max_s` (`q3_metrics.py:134`), each a `q3_stats.paired_effect` ratio (K = 4 over K = 1). **Pass** if the 95% CI upper bound (`ratio_ci95[1]`) of each is ≤ 1.025 at the stop (decision D2). This is stricter than the existing Q3 fairness band, `q3_stats.regression`, which fails only when the *lower* bound exceeds 1.025.
- Footprint: per-run `peak_footprint_bytes` (the Q5 calibration summary field, `scripts/four_agent/q5_calibration.py:338`) with K = 4, divided by the fresh calibration's peak, ≤ 1.25 in every run.

### 8.5 Joint eligibility (acceptance item 5)

- Over all C2 runs (§8.3, §8.4 and §7.2 hardware runs), `hf2q_qwen_joint_not_eligible_turns_total{reason=~"anchor_.*"} == 0`, read from `/metrics` at run end and retained in the receipt.
- Per-run eligibility rate `r` = joint lane-steps ÷ (joint + serial lane-steps among decode-ready lanes on turns with at least 2 decode lanes), from the tagged turn lines.
- **Pass:** the paired ratio of `r` (K = 4 over K = 1) over the same Williams blocks (`q3_stats.paired_effect`, higher is better) has a 95% interval that contains 1, or lies entirely above 1, on each workload. Because `r` can be 0 in a run, a block with `r = 0` in either arm is recorded and makes the rule fail closed, pending owner review. `paired_effect` requires positive values.

### 8.6 Quality

- **Qwen3.8:** the paired 24-instance task-quality gate (`scripts/four_agent/quality_campaign.py`) between K = 4 and K = 1, on the divergent-edit variant composed with instances.
- **Qwen3.6:** bitwise exactness (§7.2) and `ExactTranscripts` against anchors off on the append-only workload (§7.5). On divergent turns, the D1 envelope applies on every row. No task pass rate (Q4-QWEN36).
- **Cancel smoke** (`scripts/four_agent/cancel_smoke.py`, `q5_cases/cancel-qwen38.json` and a Qwen3.6 derivative) with K = 4 and with K = 1. Its receipt adds, for every cancelled request whose client retries:
  - the retry's semantic TTFT;
  - the recovered depth (`restores_total{rank}` and the restored token count, or 0 for a reset);
  - whether the preserve-unmodified-live-prefix branch fired.

  These are reported beside today's behaviour (C2's parent commit) for decision D5.

### 8.7 Independent review

A reviewer distinct from the implementer reproduces the §7.2 matrix verdicts and the §8.3 paired TTFT effect from the raw receipts with the checked-in commands.

## 9. Risks and non-goals

| ID | Risk | Mitigation |
|---|---|---|
| R1 | Footprint: up to about 2.94 GiB of anchors on Qwen3.8 at K = 4, plus about 2.25 GiB of K3 capture at speculation `auto` | Q5 declaration and quota (§3.9). The 1.25× limit. K is configurable for the arms |
| R2 | Snapshot latency at the prefill chunk end delays peers [INFERENCE: one 150 MiB blit command buffer] | Measured as `snapshot_ms`. The append-only ratio limits catch regressions |
| R3 | The lineage rule misses a write path | The rule keys on cursor changes and `slot_state_generation`. Every `reset_for_slot`, `embed`, graft-splice, `drop_seq` and `fork_seq` caller is enumerated in review. O1 negative control (§7.3). U10 |
| R4 | The pending rule makes cancel-and-retry lose its same-prompt hit (behaviour change vs today) | It follows C1 rule 3 and the ADR text. Owner acknowledgment D5. The cancel smoke measures retry TTFT and recovered depth (§8.6) |
| R5 | Divergent-edit TTFT gain is small if the edited region is near the anchor | The edit position is chosen so the restored depth is turn `t−2` (§8.2); `estimate` reports the prefix shares |
| R6 | The append-only ≤ 1.025 upper-bound rule may not pass even with a zero true effect if precision is poor | The sequential precision stop (§8.3) runs until half-width ≤ 2.5 points or 20 blocks. A `max_blocks` stop without a pass keeps K = 1 for that artifact |

**Non-goals:**
- Gemma and DeepSeek multi-anchor (C1 carry-overs O2–O4 stay successors).
- Disk promotion of anchors.
- Stride or intra-turn checkpoints.
- Non-prefix or approximate recurrent reuse.
- Changing the 2K prefill quantum.
- Cross-slot anchor migration.
- Making the SerialFifo `snapshot_prefix_from_capture` path multi-depth.
- Making demand measurement a gate.
- Grafted or abliterated artifacts as primary evidence (they are compatibility arms only).

## 10. Open decisions for root or owner (before implementation)

- **D1 (tightened per review).** Anchors-off cold-prefills the divergent turn from 0, while anchors-on resumes from a state built by a different transaction history (decode-route rows, other chunk boundaries). Under the ADR-063 Q1 numeric policy that is a Class B alternate route, so bitwise-equal transcripts cannot be guaranteed on divergent turns. Proposed rule:
  - bitwise exactness is proven by the same-route twin (§7.2) on both artifacts;
  - `ExactTranscripts` is required on the append-only workload (§7.5), where the routes are identical;
  - on the divergent-edit variant, **every logit row of each divergent turn** (not just the first divergent row) is checked against the teacher-forced anchors-off rows on the same frozen prefix. Each must be `WithinDecisionEnvelope` (D ≤ 0.25 nat, Q1 numeric policy §3.2). Any `RouteSensitive` row blocks C2 until localized;
  - Qwen3.8 additionally passes the paired task-quality gate.

  This changes the ADR's literal gate, so it needs an ADR amendment with owner sign-off. Draft text for ADR-063 §C2, replacing "Qwen3.6 is judged by cold-versus-resumed bitwise exactness and `ExactTranscripts` against anchors off":

  > Qwen3.6 is judged by cold-versus-resumed bitwise exactness against the same-route twin, `ExactTranscripts` against anchors off on the append-only workload, and, on divergent turns, the Class B decision envelope (ADR-063 Q1 numeric policy §3.2) on every logit row of each divergent turn against teacher-forced anchors-off rows; any `RouteSensitive` row blocks. Task pass rate is not used while no workload on which Qwen3.6 passes is declared (Q4-QWEN36). Qwen3.8 is judged by the same rules and additionally passes the paired task-quality gate.
- **D2 (accepted by review).** "≤ 1.025" is read as the paired-ratio 95% CI upper bound at the Q3 sequential stop. This is stricter than the existing Q3 regression band (lower bound > 1.025 fails), so it needs an explicit ADR-063 amendment: "append-only … ratios have 95% CI upper bounds ≤ 1.025 under the Q3 sequential precision stop".
- **D3 (revised per review).** No fixed block count. C2's campaigns use Q3's pre-registered sequential 2.5-point precision stop (`q3_stats.stop_decision`, 5–20 blocks). A fixed count is used only if the owner chooses one from a labelled pilot that is never pooled.
- **D4.** The anchor identity omits C1's `conversation` field (§3.5). Exact-prefix sharing across clients matches today's behaviour but departs from C1's normative identity list, so it needs owner sign-off.
- **D5.** Owner acknowledgment that cancellation discards the request's own pending anchor (§3.6), with the cancel-smoke retry TTFT and recovered-depth figures (§8.6) attached.

## 11. Source identities read

| Path | SHA-256 |
|---|---|
| `src/serve/api/engine.rs` | `9383b87263ea7dc252a8ac4bb64eb16780169a82d792dfafc7987772f95f574f` |
| `src/serve/api/engine_qwen35.rs` | `47f755feeeed176aa5dd77179e430ceb2d09a6f0e5b14d883a8f2e1f5ce6bd34` |
| `src/inference/models/qwen35/kv_cache.rs` | `6bc46b78124b9e41b494a5136324e03b76edb5a584279271291d4d8b5cf4518e` |
| `src/serve/api/engine/qwen35_joint_decode.rs` | `535a6ef0847fafe6ec11388d4c665080e629199c883fefe31dfd8361fb20805b` |
| `src/serve/api/engine_qwen35/joint_decode.rs` | `cb11e0c3690da3d248f68598ad73ef08b4c0f53b57674d3f614852e01431b5f5` |
| `src/inference/anchor_contract.rs` | `3d2097d9af6da0f5cc48227462093552b6a3d088461606ac6a5c739a535ead79` |
| mlx-native 0.15.16 `src/encoder.rs` | `275ea47915de3724e630ac578c8aa3745cf872e026720e71ebf3085d0b65944b` |
