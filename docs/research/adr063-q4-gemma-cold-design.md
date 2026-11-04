# ADR-063 Q4-GEMMA-COLD: one exact joint prefill for the four simultaneous cold Gemma 4 first turns

Status: design for approval (2026-11-04). No code, build or hardware run backs
this document. Every number is quoted from a retained receipt, with its path,
or marked `[INFERENCE]`.

Revision 2 (2026-11-04) applies the ACCEPT-WITH-CONDITIONS review
(ReviewGemmaCold):
- findings 1–8: Phase C logits source; the cursor commit; the iter-G(a)
  33-row floor; the installed-prefill claim narrowed to the cursor-0 chunk;
  anchor-quota wording; global-layer staging; makespan reported, not gated;
  bitwise gates for the 32-row and mixed cues;
- its smaller notes: reservation release before retry, self-mount clear
  after Phase B, and the ADR requirement scoped to eligible cohorts;
- the D4 amendment.

Story: **Q4-GEMMA-COLD** (new, v0.1.30), one of the four stories the operator
added on 2026-11-04 to close the Gemma 4 26B-A4B gap that #668 (Q4-GEMMA-ATTN)
cannot close alone. It owns the cold-turn admission stage that the #660 profile
names as stage 4 (`docs/research/adr063-q4-gemma-profile-result.md` §6–§7, on
branch `feat/q4-gemma-profile`).

Sources:
- ADR-063 sections Q1, Q4, Q4-GEMMA, Q4-GEMMA-ATTN and "Speed program, Q1
  successors and release mapping".
- The Q1 numeric policy `hf2q.q1.numeric-policy.v1`
  (`docs/research/adr063-q1-numeric-policy.md`).
- `docs/research/adr063-q1-gemma-t2-localization.md` (#652) and
  `docs/research/adr063-q1-gemma-packed-cohort.md`.
- The joint warm precedent: `docs/research/adr063-gemma-four-agent-reuse.md`
  Stage 1 and `src/serve/api/engine_gemma_warm_batch.rs`.
- hf2q `main` at `ff4af4bd` and mlx-native at `39f60bb` (read only).

`$G` = `/Users/robert/.local/state/hf2q/v0130/q4-gemma` (the #660 evidence).
`$C` = `/Users/robert/.local/state/hf2q/v0130/q4-gemma-cold` (this story's
evidence root, created by the GPU-lane holder).

## 1. Decision summary

1. **The cause is a routing gap, not missing batching.** The four turn-0
   requests already reach the worker inside one 25 ms coalesce window, and the
   worker already has two joint-prefill routes. But every turn-0 request
   carries an exact pre-generation boundary (S = 458 of P = 465 rows). That
   makes it a "stable" candidate, and the stable batch only accepts lanes with a
   cached prefix. So the whole cohort falls back to four serial per-slot
   admissions (§2.2). None of the four lanes decodes until the last one is
   prefilled. That produces both the 422 ms queue and the +459 ms "decode".
2. **The change is a new cohort route: cold lanes with a boundary.** It runs
   in three phases:
   - **Phase B.** One joint prefill of every lane's boundary rows.
   - **Anchors.** Each lane's boundary anchor is captured in FIFO order,
     exactly where per-slot admission captures it.
   - **Phase C.** A joint replay of the short generation cue. It uses the
     batched decode body, which is already proven byte-identical to N serial
     slot-aware decodes.

   Each lane's decode state, anchor and first token are then published exactly
   as `admit_gemma4_slot` publishes them (§3).
3. **Exactness: Q1 Class A, by construction and by gate.** Phase B shares
   only the token-wise operations with the stream. These are norms, QKV,
   RoPE, dense FFN, router and MoE, and the joint warm gate already proves them
   row-invariant at more than 32 rows per lane. Attention and O-projection run
   **per lane** through one per-sequence cold attention function that is
   shared with the single-slot cold route (§4.3). The existing cold
   multi-sequence route is not reused. Its BF16 block-diagonal flash attention
   is the Class B, `RouteSensitive` "packed T2" route that #652 localized to
   layer-0 SDPA.
4. **Clean cutover (operator decision D2).** The new per-sequence cold
   attention replaces the block-diagonal route in every cold multi-sequence
   caller: the iter-G(a) cold batch and the installed cold prefill. The
   iter-G(a) lane floor rises from 32 to 33 rows (§3.3). hf2q then keeps one
   cold multi-sequence route, and Q1 T2 becomes a Class A pair. For installed
   long prompts, only the cursor-0 chunk forward becomes exact. The chunk
   plan and the later live chunks still depend on how many lanes are
   co-installed (§3.3).
5. **Expected effect [INFERENCE]: 0.20–0.35 s of makespan, not the profile's
   0.4–0.5 s.** The profile's upper bound assumed that hf2q's 275 ms per-slot
   prefill carries into a joint forward. It does not: by the joint warm fit,
   about 80% of the 187 ms per-slot boundary forward is per-row and per-lane
   attention work, which a joint forward does not remove (§10).
   This story removes the queue and the decode inflation and amortizes the
   fixed per-forward costs and the cue replay. The per-row cost belongs to the
   prefill-kernel stories.
6. **Default on, with kill switch `HF2Q_GEMMA_JOINT_COLD_PREFILL=0`**, as for
   joint warm prefill (`HF2Q_GEMMA_JOINT_WARM_PREFILL`). The official-artifact
   bitwise gates are the contract, and the root runs them before merge
   (operator decision D3).

## 2. Evidence: what turn 0 costs today

### 2.1 Measured turn 0 (#660 L1, 7 hf2q runs, binary `$G/bin-7f295b38`)

Per-lane values come from `x_hf2q_timing` in each `response.bin`. They are
read with `scripts/four_agent/gemma_stage_profile.py` `turn_rows`
(`feat/q4-gemma-profile`) over `$G/profile/l1/*-hf2q-*`. Lanes are sorted by
server queue.

| Run | send spread (ms) | queue per lane (ms) | prefill per lane (ms) | decode per lane (ms) | span (ms) |
|---|---:|---|---|---|---|
| 001-b01 | 0–25 | 1 / 284 / 559 / 831 | 283 / 275 / 272 / 273 | 1164 / 889 / 618 / 345 | 1522–1546 |
| 004-b02 | 0–10 | 1 / 281 / 551 / 821 | 280 / 270 / 269 / 270 | 1170 / 900 / 631 / 362 | 1542–1553 |
| 005-b03 | 0–6 | 1 / 286 / 563 / 838 | 285 / 277 / 275 / 274 | 1189 / 912 / 637 / 363 | 1571–1577 |
| 008-b04 | 0–21 | 1 / 285 / 560 / 834 | 284 / 275 / 274 / 276 | 1186 / 911 / 637 / 362 | 1550–1570 |
| 009-b05 | 0–2 | 1 / 284 / 560 / 832 | 283 / 275 / 271 / 272 | 1187 / 912 / 641 / 369 | 1570–1572 |
| 012-b06 | 0–4 | 1 / 285 / 562 / 835 | 284 / 277 / 273 / 275 | 1185 / 908 / 636 / 361 | 1561–1565 |
| 013-b07 | 0–4 | 1 / 285 / 563 / 838 | 284 / 277 / 275 / 277 | 1187 / 909 / 635 / 358 | 1566–1571 |

All 28 lanes complete with 18 completion tokens. The lane medians of the
queue are 1 / 285 / 560 / 834 ms.

The pattern is a strict staircase. Each lane waits about 275 ms per earlier
lane, then all four decode together. The first-served lane's "decode" is 3 ×
275 ms of waiting plus the shared 18-token decode, which is about 360 ms, as
the last-served lane shows. Spans are equal across lanes within a run. Every
agent pays about 4 × 275 + 360 + 90 ms.

Server log of run 001 (`$G/profile/l1/001-b01-hf2q-20261104T200542Z/server.stderr:53-68`):
- all four requests are dispatched (`prompt_tokens=465`) before the first
  admission;
- then four `Gemma4 slot admission planned … cached_tokens=0 cache="reset"`
  lines follow, one per slot. Each is followed by
  `Batched prefill: … seq_len=458, path=flash-attn, globals=tensor-mm` and a
  completion time of 195.8, 186.8, 186.8 and 187.7 ms;
- the profiled run 021 shows 184.9–194.0 ms (`$G/profile/l3/021-b01-hc-*/server.stderr:59-68`).

The remaining 75–100 ms of each lane's 269–285 ms prefill has two parts: the
CPU anchor snapshot, and seven single-slot decode forwards that replay the
7-token generation cue (§2.2, step 4). ADR-040 §0.17 measured one single-slot
step at about 10.6 ms on an older build, so the cue alone is about 74 ms
`[INFERENCE]`.

The peer (MT, same runs, `$G/profile/l1/*-peer-*`) reports `prompt_ms`
488–709 ms per lane, with 3 of 4 lanes sharing one value in most runs. That is
consistent with one or two joint batches. It reports `predicted_ms` of
283–371 ms and spans of 965–1089 ms.

Against the peer (profile §2): turn-0 span 1563 / 1067 ms (+496 ms), which is
**18% of the +2.734 s makespan gap**.

### 2.2 Why the cohort goes serial (source, `main` `ff4af4bd`)

1. **Classification.** `gemma4_cross_batch_candidate`
   (`src/serve/api/engine.rs:6352`) marks a request with
   `stable_prompt_prefix_tokens = Some(S)` as `Gemma4CrossBatchClass::Stable`.
   The four 465-row requests form one Stable prefix of 1,860 rows under the
   4,096-row cap (`gemma4_bounded_batch_prefix_len`, :6458). All slots are
   idle, so the collector holds them in one quantum (:12123–12155, 25 ms
   window `GEMMA4_CROSS_SLOT_ADMIT_COALESCE`, :6305).
2. **The stable batch refuses cold lanes.** `admit_gemma4_slots_stable_batched`
   (:15526) plans each lane with `preference.filter(|p| p.cached_tokens > 0 …)`
   (:15638). A turn-0 lane has no reusable prefix, so the loop breaks at the
   first lane. `plan.len() != requests.len()`, and the whole group goes to
   `fallback_single` in FIFO order (:15670). A tiny cue (7 < 32 rows) would
   refuse it as well (`has_tiny_phase`, :15654).
3. **The cold batch refuses boundaries.** `admit_gemma4_slots_batched`
   (:15004) takes only requests without a boundary (the `Cold` class). It
   never manufactures an anchor ("boundary-bearing requests are routed
   through `admit_gemma4_slot`", :15405).
4. **The per-slot path is inline and serial.** `admit_gemma4_slot` → 
   `Gemma4DecodeState::prefill_seed` (:7234) handles each lane in four steps:
   - reset the slot;
   - prefill `prompt[..S]` through `forward_prefill_with_soft_tokens_slot_aware`
     → `forward_prefill_batched_slot_mounted`
     (`src/serve/forward_prefill.rs:5139`);
   - snapshot the sliding rings at S
     (`snapshot_gemma_hybrid_slot_anchor`, a CPU copy of about 158 MB per slot,
     `src/inference/models/gemma4/kv_cache.rs:1673`);
   - process the cue through `forward_prefill_with_soft_tokens_slot_aware_resume`,
     which replays a suffix shorter than 32 rows as one `forward_decode_slot_aware`
     call per token (`forward_prefill.rs:3915-3933`).

   The lane is installed as `Decode` only afterwards. The admission quantum
   runs all four lanes before the scheduler's first decode tick.

### 2.3 Why the existing cold multi-sequence route cannot simply be extended

`forward_prefill_batched_multi_seq` (`src/serve/forward_prefill_batched.rs:6722`)
concatenates the lanes and isolates them with a GPU-built block-diagonal mask:
- it forces BF16 flash attention for sliding **and** global layers
  (`use_fa_f16 = multi_seq_prefill.is_none()`, :877;
  `force_global_nofa … && multi_seq_prefill.is_none()`, :925);
- the single-slot cold route uses F16 flash attention for sliding layers and
  tensor-mm for 64 < S ≤ 8192 global layers.

This is exactly the Q1 "packed T2" route:
- #652 localized its first divergence from serial to layer-0 prefill
  `Attention`, on all 128 rows of all four lanes. Q, K, V and RoPE were
  byte-identical before it;
- the policy table records T0 vs T2 as Class B `RouteSensitive`: top-1
  391/576, routing changed on 8,914 of 17,280 rows
  (`adr063-q1-numeric-policy.md` §2.2).

Putting turn 0 on that route would change every Gemma conversation's
boundary anchor, and with it every later turn. That needs the paired
24-instance quality campaign (Q1 §3.5), and #544 already showed one
near-tie flip derailing the four-agent task. This design therefore requires
Class A.

## 3. Proposed change (HLD)

### 3.1 The cold stable cohort rule (model-free planner)

A contiguous FIFO Stable prefix of `reqs`, collected while all slots are idle
(the existing quantum), runs as one **cold stable cohort** when all of these
hold:
- **At least two lanes.** Every lane is `Generate`/`GenerateStream` text with
  `max_tokens > 0`, no soft tokens, no `vision_fingerprint` and no effective
  full-input tag (`gemma_full_input::effective`).
- **Every lane is cold.** Its preference `preferred_gemma4_slot_excluding`,
  planned against already-selected slots exactly as the stable planner does,
  has `cached_tokens == 0` or is absent. A group with any cached lane keeps
  today's stable-batch rule unchanged, including its all-or-nothing FIFO
  fallback.
- **Boundary rows.** Every lane has `33 ≤ S_i < P_i`. `33` is
  `gemma_warm_batch::MIN_SUFFIX_TOKENS`, the threshold above which a lane's
  MoE takes `mm_id` both alone and in the joint stream (§5.1). A shorter
  boundary is cold-prefilled on the linear path per slot and cannot join.
- **Transaction cap.** `Σ P_i ≤ 4,096` (`gemma4_batch_rows_within_transaction_limit`),
  the existing Gemma transaction cap.
- **Anchor quota.** Under the serial admission order, every lane's
  `StableBoundary` reservation would be accepted (§3.2 step 4).
- **Worker substrate.** Hybrid KV; cross-slot admission on; the kill switch
  on; `kv_graft_len() == 0`; `HF2Q_BATCHED_BODY` not `0`;
  `HF2Q_PREFILL_SLOT_BATCHED` not `0`; no `HF2Q_MM_ID_ROUTING_THRESHOLD`
  override; `HF2Q_DFLASH_XLEN_SDPA` not `1`; and none of the diagnostic
  attention overrides `HF2Q_FA_F16=0`, `HF2Q_NO_FA` or `HF2Q_GLOBAL_FA`. Each
  of these either changes the per-slot reference route or bypasses the
  route the cohort reproduces. The worker resolves the decision once at
  start and logs it, as `gemma_warm_batch::worker_enabled` does.

Anything else goes through today's path. Nothing is reordered: a cohort that
fails the rule runs in FIFO order through `fallback_single`, as now.

### 3.2 The transaction

```
reserve (FIFO)  →  reset (all lanes)  →  Phase B: joint boundary prefill
     →  anchors (FIFO)  →  Phase C: joint cue  →  commit cursors  →  publish (FIFO)
```

1. **Reserve.** For each lane in FIFO order, issue exactly the `AdmitRequest`
   that `admit_gemma4_slot_routed` would issue (same `preferred_slot`, same
   byte estimates), so slot selection and the shared-KV budget decisions equal
   serial admission. Drop closed clients before any reset, as
   `admit_gemma4_slots_batched` does (:15152–15173). If fewer than two lanes
   survive, route them per slot.
2. **Reset.** Clear `retained_tokens`/`prompt_anchors` and call
   `reset_gemma4_slot` for each lane, then `clear_gemma4_self_mounts`.
3. **Phase B.** One `forward_prefill_batched_multi_seq` over
   `[(prompt_i[..S_i], slot_i)]`, with every lane at start 0, using the
   per-sequence cold attention of §4.3. The per-lane head rows are computed
   and discarded, as `prefill_seed` discards the boundary token.
   `clear_gemma4_self_mounts` runs again after Phase B, as `prefill_seed`
   does after its boundary forward (:7410).
4. **Anchors.** In FIFO order, `gemma_full_input::capture_or_skip(slot,
   AnchorRole::StableBoundary, snapshot_gemma_hybrid_slot_anchor(scaffold,
   slot, S_i))`. The capture role, order and sizes equal serial admission.
   Quota accounting is not identical, though. Step 2 resets every lane first,
   which drops all the lanes' old anchors and their quota reservations
   (`anchor_budget.rs:46`). Serial admission frees only lane k's old anchor,
   just before lane k captures. Under a binding quota the cohort could
   therefore refuse only a subset of the lanes serial admission would refuse,
   and would keep an anchor that serial admission drops, which breaks
   byte-for-byte anchor equality. To keep the claim exact, the planner
   refuses the cohort (per-slot fallback) unless the serial order's
   reservations all fit. That is checked model-free against the quota ledger
   before step 2. In practice the refusal is not reached: the checkpoint
   worker activates only when the two-generation quota plan fits
   (`decide_for_scaffold`). A hosted control covers the refusal.
5. **Phase C.** Process every lane's cue `prompt_i[S_i..P_i]`:
   - **All cues < 32 rows** (the Gemma generation cue is 7 rows): replay the
     cue through the batched decode body, one step per cue position, with each
     lane's forced token and position (§4.4). The serial reference replays
     the same tokens through `forward_decode_slot_aware`.
   - **All cues ≥ 33 rows:** one `forward_prefill_batched_multi_seq_live`
     with `GemmaLiveAttention::PerSlotExact`, the joint warm route (its
     bitwise gate covers it).
   - **A lane with a 32-row cue,** or a mix of short and long cues: those
     lanes run their cue per slot through
     `forward_prefill_with_soft_tokens_slot_aware_resume` inside the same
     transaction. This is exact by identity, and only the short-cue lanes
     share replay steps.
6. **Commit.** `commit_gemma4_slot_cursors` for `(slot, 0, P_i)`,
   all-or-nothing. Prefill and decode forwards do not move the ledger
   cursor (`seq_lens`). The serial path commits `(cached, P)` only after
   `prefill_seed` (:13919–13922). So every lane's cursor is still 0 here: the
   commit validates that itself, and no separate validation at `P_i` is made.
   The replay invariant checked before the commit is that each lane's last
   replayed position is `P_i − 1`.
7. **Publish (FIFO).** For each lane:
   - `Gemma4DecodeState::from_first_token(…, first_token, Some(row),
     elapsed, 0)`;
   - `prompt_anchors[slot] = anchor`;
   - `scheduler.advance_after_prefill(handle, P_i)`;
   - seed tick, then `finished_at_seed` handling, then install `Decode`.

   This is the tail of `admit_gemma4_slots_stable_batched` (:16031–16160).

The cohort runs only when every slot is idle. No decoding slot waits on it,
and the next scheduler tick decodes all lanes together.

### 3.3 Interaction with the other admission routes

| Group shape (all slots idle) | Today | After |
|---|---|---|
| All cold, with boundary (turn 0) | serial per slot | **cold stable cohort** |
| All cold, no boundary | iter-G(a) cold batch (block-diagonal, Class B; lanes ≥ 32 rows) | iter-G(a) cold batch on the per-sequence cold attention (Class A; D2). The lane floor rises to **33** rows (`MIN_SUFFIX_TOKENS`): a 32-row lane runs its MoE gate/up on `mv_id` alone but on `mm_id` in the joint stream, so it goes per slot |
| All cached, with boundary, cues ≥ 32 | stable batch (tiled F16 live attention) | unchanged |
| Warm tool turns (full-input tags) | joint warm cohort | unchanged |
| Mixed cold/cached | serial per slot, FIFO | unchanged |
| Long prompts (> 4,096 rows), installed lanes at cursor 0 | `forward_prefill_batched_multi_seq` block-diagonal | per-sequence cold attention (D2). **Only the cursor-0 chunk forward is exact** for its planned chunk. `gemma4_prefill_batch_plans` gives each of N co-installed lanes 4,096/N rows, while a lone lane takes 4,096, and later chunks use the live `Tiled` resume. The request-level output therefore still depends on the co-batch, and this row makes no request-level Class A claim |

### 3.4 Kill switch and observability

- `HF2Q_GEMMA_JOINT_COLD_PREFILL` accepts `1`/`on`/`true` (the default) and
  `0`/`off`/`false`. Invalid values warn and keep the default. It is read once
  per SlotAware worker and logged as `Gemma joint cold prefill worker decision
  requested=… active=… unavailable=…`. It also turns off whenever
  `cross_slot_admit` is off.
- Each cohort logs `Gemma4 cold stable cohort prefilled in one joint
  transaction requests=N boundary_rows=… cue_rows=… boundary_ms=… anchor_ms=…
  cue_ms=… elapsed_ms=…`.
- Each cohort emits `scheduler.trace_transaction("gemma4",
  "cold_stable_cohort", &handles)`, so an F2/Q2 trace attributes the shared
  forward to its owners. The multi-sequence forward already emits the
  `gemma4.prefill.actual_concat_descriptor` native input.
- Operator UI kinds: `slot-unary-cold-cohort` / `slot-stream-cold-cohort`.
- Documentation: `docs/env-var-inventory.md`, `docs/operator-env-vars.md`,
  `scripts/audit_hf2q_env_surface.sh`, a "Stage 0: exact cold cohort"
  section in `docs/research/adr063-gemma-four-agent-reuse.md`, and
  `CHANGELOG.md`.

## 4. Low-level design

### 4.1 Files

| File | Change |
|---|---|
| `src/serve/api/engine_gemma_cold_batch.rs` | Becomes the cold-admission policy module. It gains `stable_plan(…) -> StablePlan` (§3.1, model-free, `#[cfg(test)]` controls) and `worker_enabled()`. The existing `plan` is unchanged |
| `src/serve/api/engine_gemma_cold_stable_batch.rs` (new, about 400 lines) | `admit_cohort(…)`: the §3.2 transaction, its failure handling (§4.5) and the cue replay driver (§4.4). Keeps `engine.rs` from growing (AGENTS.md: legacy files should not grow) |
| `src/serve/api/engine.rs` | `admit_gemma4_slots_stable_batched` calls `gemma_cold_batch::stable_plan` first and, when it holds, `gemma_cold_stable_batch::admit_cohort`. The worker resolves `joint_cold_prefill` beside `joint_warm_prefill` (:12046) |
| `src/serve/gemma_cold_attention.rs` (new) | `encode_cold_layer_attention(s, reg, dev, &ColdLayer, seqs: impl Iterator<Item = ColdSeq>)`, the one per-sequence cold attention and O-projection (§4.3). It mirrors `gemma_live_attention.rs` |
| `src/serve/forward_prefill_batched.rs` | The cold attention block of `forward_prefill_batched_inner` (the FA-F16 sliding branch around :3283–3463, the tensor-mm global branch :2580–2830, the global F16 FA branch at seq ≤ 64 or > 8192, and the O-projection :3958ff) moves into `gemma_cold_attention`. Single-slot cold prefill calls it with one `ColdSeq { offset: 0, rows: seq_len }`. Multi-sequence cold prefill calls it once per lane. The block-diagonal mask, `pf_seq_id` and the multi-sequence BF16 forcing (`use_fa_f16`, `force_global_nofa` exclusions) are removed (D2) |
| `src/serve/api/engine_gemma_cold_stable_batch_tests.rs` (new) | Hosted planner and transaction controls, plus ignored official-artifact gates (§6) |

There is no mlx-native change. Every kernel used already exists at the
current pin: `permute_021_f32_to_f16`, `dispatch_flash_attn_prefill_f16_d256_with_blk`,
`dispatch_flash_attn_prefill_f16_d512_with_blk`, the tensor-mm chain,
`build_sdpa_mask` and `blk`, and `forward_decode_body_batched` with
`lm_head_batched`.

### 4.2 Planner

```rust
pub(super) struct StablePlan { lanes: Vec<StableLane> }      // FIFO order
pub(super) struct StableLane { boundary: usize, prompt_len: usize, cue: CueRoute }
pub(super) enum CueRoute { Replay, JointLive, PerSlot }       // <32, >=33, ==32 or mixed

pub(super) fn stable_plan(
    worker: &ColdWorker,                       // resolved once (§3.1 substrate)
    requests: &[(Vec<u32>, SamplingParams, SlotReply)],
    preference: impl FnMut(usize) -> Option<Gemma4SlotPreference>, // excluding selected
) -> Option<StablePlan>;
```

`None` means today's path. The function is pure: it reads token lengths,
params and the preference closure (the same closure `admit_gemma4_slots_stable_batched`
builds from `preferred_gemma4_slot_excluding` and the `selected` bitmap). It
performs no reservation, reset or GPU work.

### 4.3 Per-sequence cold attention (Phase B)

The rule is that everything that depends on position, sequence or layout
runs per lane through the code path the single-slot cold prefill runs. The
token-wise operations run once over the stream.

| Operation | Phase B | Why exact |
|---|---|---|
| Embedding, input norm, Q/K/V projections, fused head norm and RoPE (per-row positions, already reset per sequence, :1537), V norm | joint over T = Σ S_i rows | Row-wise kernels; equal at layer 0 on packed T2 (#652 stage table: Query, Key, Value, RopeQuery, RopeKey 512/512 equal) |
| KV write into each slot region (F16 K, TQ-HB V) | existing per-sequence scatter (iter-G(a) "delta 3", :4253) | Row-wise encode into the lane's own region; the cohort gate compares cache bytes |
| **Sliding attention** | **per lane:** `permute_021_f32_to_f16` of the lane's rows (`slice_view` at row `off_i`) into lane-local head-major F16 scratch at offset 0; the lane's `[S_i, S_i]` sliding mask and `blk`, built and re-cast in the consuming layer session (the §0.19 remask rule, :3396–3428); `flash_attn_prefill_f16_d256_with_blk` with `seq_len_q = seq_len_k = S_i`; F16→BF16 cast | Same function, same inputs, same shapes as the single-slot cold call |
| **Global attention** | **per lane, staged by recomputation, not by copy or strided view.** In the joint stream, the fused head-norm and RoPE kernel writes the tensor-mm operands (`pf_q_perm_f32`, `pf_k_perm`, and the V operand) head-major across all T rows, so a lane is a strided sub-block there. For a global layer the cohort does not use that joint head-major output. Instead, per lane, the same fused head-norm and RoPE dispatch (and the V norm and permute) that the single-slot route issues runs with `seq_len = S_i`. Its inputs are `slice_view`s of the joint seq-major Q/K/V projection rows `[off_i, off_i + S_i)` and of `pf_positions`. It writes lane-local head-major operands at offset 0, plus the lane's rows of the joint seq-major normed buffers that the KV scatter reads. Then, for 64 < S_i ≤ 8,192, the tensor-mm chain runs (Q·Kᵀ into `pf_kq` sized for the largest lane, scale-mask-softmax, ·V, permute); for other S_i, F16 d512 FA runs. The predicate is the existing one, evaluated with `S_i`. This costs 5 layers × 4 lanes of row-wise dispatches | Same kernels, same inputs, same shapes and offset-0 operands as the single-slot cold call. The projection rows are row-invariant (first table row). No strided view and no reinterpreting copy is involved |
| **O-projection** | **per lane,** the branch the single-slot route takes for the artifact's codec (perm021 tensor-MM, or permute-to-seq-major plus the ordinary route), `m = S_i`, writing rows `[off_i, off_i + S_i)` of the joint hidden buffer | Same kernel at the same m. It costs about 3 extra reads of about 0.25 GB of O-projection weights per forward, under 3 ms `[INFERENCE]` |
| Residuals, post-attention norm, dense FFN, router, MoE (`mm_id` at n_tokens = T), layer tail | joint | Row-invariant for every lane above 32 rows: the joint warm gate (`gemma_joint_warm_prefill_matches_per_slot_bitwise*`) proves the same code bitwise. The concurrent Gemma prefill-MoE design states (2026-11-04) that its `mm_id` schedule change keeps per-row output independent of total rows and expert grouping, and keeps the threshold at 32 |
| Final norm and head | per lane, last row (existing `head_rows`, :5568) | m = 1 per row, as single-slot |

The per-lane scratch is sized for the largest lane and reused across lanes
inside a layer session. `barrier_between` orders reuse. Lane-local
head-major buffers start at offset 0, so no flash-attention dispatch reads at
a non-chunk-aligned offset. That is the condition under which ADR-040 §0.19
saw F16 FA nondeterminism with block-diagonal masks (:867–876).

`MultiSeqPrefillState` loses nothing it still needs. `live_attention` keeps
its meaning for live resumes. The cold path ignores it, as it does today.

### 4.4 Cue replay (Phase C, cues < 32 rows)

Let `c_i = P_i − S_i`. For each step `j` in `0..max_i c_i`:
- the active lanes are `{i : j < c_i}` in FIFO order;
- each active lane's token is `prompt_i[S_i + j]` and its position is
  `S_i + j`. The graft length is 0 by rule, and this equals the serial
  replay's `kv_graft_len() + cached_tokens + offset`, `forward_prefill.rs:3923`;
- one `forward_decode_body_batched(tokens, slot_ids, positions,
  BatchedKvRegime::Hybrid, &mut head_out, gpu)` call runs per step, with a
  fresh `head_out: Option<BatchedHeadOut> = None`. That body is documented and
  gated as byte-identical to N serial `forward_decode_slot_aware` calls
  (`batched_body.rs:491-505`; `slot_aware_n1`/`slot_aware_n4`/`slot_aware_n8`);
- logits come from the decode tick's own source, exactly as
  `decode_batch_gemma4` reads them (engine.rs:16814):
  - under the default fused head (chunked body, `HF2Q_FUSE_LMHEAD` not 0),
    the body encodes the head as its final chunk, fills `head_out` and
    returns an empty hidden vector (`batched_body.rs:2521, 2599`);
  - otherwise it returns hidden rows and one `lm_head_batched` over them
    produces the logits.

  For lanes whose step is `j == c_i − 1`, the first token is
  `argmax_f32_first_max` of the lane's row, the decode tick's selection, and
  `from_first_token` receives `Some(row)`.

Under the fused default the head runs on **every** replay step. Intermediate
rows are read and discarded, as the serial replay discards its intermediate
tokens. Heads do not feed KV. On the unfused route, intermediate steps skip
`lm_head_batched`. Both routes leave the KV identical. The per-step cost in
§10 already includes a head, because the measured turn-0 decode step does.

Each step is its own `supervised_gemma4_gpu_call` (`gemma4_cold_cohort_cue`),
so the watchdog bound per call is one decode step. No step moves the ledger
cursor. The invariant checked after the last step is that each lane's last
replayed position is `P_i − 1`. The cursor is then committed `(0, P_i)` in
§3.2 step 6.

### 4.5 Failure and cancellation semantics

These mirror the joint warm contract (`adr063-gemma-four-agent-reuse.md`
Stage 1).

- **Pre-submission failure** (plan, reservation, reset, cursor): lanes not yet
  reset are routed per slot in FIFO order. A failed reset is an invariant
  fatal, as in `admit_gemma4_slots_batched`.
- **Phase B non-fatal forward error:** reset every lane (a cold lane's
  request-start state is the reset slot). Release each lane's reservation
  with the existing pattern (`scheduler.record_slot_high_water(handle, 0)`,
  then `scheduler.release(handle)`), so the per-slot path's own admission
  does not meet a phantom reservation. Then run each lane through
  `admit_gemma4_slot` in FIFO order, exactly as if it had been admitted
  alone. A lane's own fault recurs per slot and fails only that request.
- **Anchor capture error other than a quota refusal:** reset that lane and
  fail it request-locally. The other lanes continue into Phase C. A quota
  refusal skips the anchor, as serial admission does.
- **Phase C non-fatal error:** restore each lane from its just-captured
  boundary anchor (`restore_gemma_hybrid_slot_anchor` at `S_i`) and finish
  its cue per slot in FIFO order through the serial cue path, keeping its
  reservation. A lane whose anchor was refused is reset instead, its
  reservation is released (`record_slot_high_water(handle, 0)`, then
  `release`), and it is admitted afresh per slot.
- **Worker-fatal GPU error** (watchdog, command-buffer fault): the worker
  fails, as for every Gemma transaction.
- **Publication failure of one lane:** restore and fail that lane only.
- **Cancellation:** a client closed before reservation or reset is dropped
  and cancelled. One that closes during the transaction is recovered after
  publication from its published anchor
  (`recover_gemma4_slot_after_cancellation_for_reply`) and released, while
  the others decode.
- **Fault injection (tests only):** `INJECT_COLD_FORWARD_FAULT` (Phase B) and
  `INJECT_COLD_CUE_FAULT` (Phase C), consumed by one transaction, as
  `gemma_warm_batch::INJECT_FORWARD_FAULT`.

### 4.6 Capacity and memory

- Each lane is validated for its own `P_i + max_tokens_i ≤ ctx`
  (`validate_gemma4_generation_request`). Phase B keeps the cold rule of the
  largest `max_tokens`, since its forward bounds each lane by its own slot
  capacity (:15292).
- Phase B scratch:
  - token-wise activations scale with T ≤ 4,096, the same cap as one
    single-slot 4,096-row prefill;
  - per-lane attention scratch is sized by the largest lane. `pf_kq` is
    `16 × S_max² × 4 B`, 13.4 MB at S = 458;
  - no `[T, T]` block-diagonal masks are built. That is 2 × 1,832² × 2 B ≈
    13 MB less at turn 0.

  The peak is therefore at most that of one single-slot prefill of the same
  total rows `[INFERENCE]`. The Q5 calibration confirms it (§9).
- The anchors are the same four `StableBoundary` captures serial admission
  makes: about 158 MB each, CPU, inside the full-input quota of 1,610,612,736
  bytes (`server.stderr:19`).

### 4.7 Timing attribution

`prefill_time_secs` of every lane is the whole transaction: Phase B, anchors
and Phase C. This matches the joint warm convention (the profiler charges a
joint forward to every lane, `gemma_stage_profile.py:10-11`). Server queue is
then `total − prefill − decode ≈ 0` for every lane.

## 5. Exactness and quality contract

### 5.1 The Class A claim

ADR-063 Q1 Class A, same route: each cohort lane is **bitwise equal to its
own per-slot admission**, the `admit_gemma4_slot` path of §2.2, on the same
binary. That covers:
- first-token logits row bytes and the token;
- every committed cursor;
- all 30 layers' slot KV regions through `P_i` (global rows and sliding
  rings);
- the boundary anchor bytes;
- the decode that follows.

The claim rests on five facts. Each one is gated, not assumed:
1. Dense quantized matmuls at m > 8 take `mm` kernels whose per-row result does
   not depend on m. The MoE takes `mm_id` at n_tokens > 32, and its per-row
   result does not depend on n_tokens or on expert grouping. Both are proven
   by the joint warm bitwise gates on the same code, and lanes are restricted
   to S_i ≥ 33.
2. Attention and O-projection run per lane through the same function the
   single-slot cold route calls, with identical inputs and shapes (§4.3).
3. The cue replay uses the batched decode body, which is byte-identical to N
   serial `forward_decode_slot_aware` calls (`slot_aware_n*`).
4. The head runs per row at m = 1. The batched head's softcap covers every
   row (ADR-040 §0.16 fix).
5. Anchors, cursors and publication follow the serial order (§3.2).

### 5.2 Policy mapping

- One differing byte is `DivergedExactInvariant`, a failure (Q1 §3.1). There
  is no tolerance and no envelope.
- Because the route is Class A, it inherits task quality from the per-slot
  route (Q1 §3.6). The four-agent obligation is `ExactTranscripts`, kill
  switch on versus off. A same-request/different-output pair voids the claim.
- Quality against v0.1.28 and the peer is unchanged by this story. It is
  measured in the Q4-GEMMA-ATTN combined candidate's paired 24-instance
  campaign (Q4 acceptance), where this story adds no new route numerics.
- **No Class B fallback (decision D1).** If a Class A gate fails, the story
  stops and localizes with the existing stage localizer (§9 step 6). The
  block-diagonal route or any other reordered route does not ship in its
  place without a separate operator decision and the Q1 §3.5 campaign.

### 5.3 Q1 consequences of the cutover (D2)

After the cutover, Q1 T2, which runs `forward_prefill_batched_multi_seq`
through `q1_gemma_actual_cohort_endpoint_trace`, becomes a Class A pair
against All128. The prefill is per-sequence exact, and the batched decode is
already exact (#652: decode layer 0 exact at all 16 positions). A re-run of
`gemma-t0` and `gemma-t2` on the candidate build must give `exact`
`ExactEqual` on all 5,616 records, with routing `ExactEqual`.

The b3 and `b0dac052` T2 records stay as history for the old route. The
ADR-063 Q1 closure table gains one row for the new build. This is a Q1
record, not a Q4 gate (ADR-063 Q4: Q1 successors do not gate Q4). It is
nonetheless the cheapest independent four-lane exactness witness, so §9 runs
it.

## 6. Tests

### 6.1 Hosted, model-free (`cargo test --locked --bin hf2q gemma_cold_`)

- **`stable_plan` matrix.** Turn-0 shape 4 × (S 458, P 465) gives `Replay`.
  The refusals are:
  - any cached lane, preserving today's stable-batch outcome and FIFO order;
  - S = 32, while S = 33 is accepted;
  - P = S;
  - Σ P = 4,097;
  - soft tokens, vision or an effective full-input tag;
  - graft bound;
  - each diagnostic environment variable of §3.1;
  - a single lane.

  Cue routing covers 7 → `Replay`, 40 → `JointLive`, 32 → `PerSlot`, and the
  mixed {7, 40} → `Replay` for the 7-row lane plus `PerSlot` for the 40-row
  lane.
- **Anchor quota refusal.** A ledger fixture in which serial order would
  refuse one anchor makes `stable_plan` return `None` (§3.2 step 4).
- **iter-G(a) lane floor (D2).** `gemma_cold_batch::plan` with a 32-row lane
  routes the group per slot. A 33-row lane joins.
- **Reservation order.** Over the scheduler fixture, the cohort's
  `AdmitRequest` sequence equals the per-slot path's, including slot choice
  when a retained slot exists.
- **Replay schedule.** Unequal cue lengths {1, 7, 31} give the expected
  active-lane sets, positions and the per-lane step whose head is read.
- **Worker decision.** Switch parsing and refusal under each unavailable
  condition, as for `gemma_warm_batch::worker_enabled`.
- **Cutover controls.** The block-diagonal mask builder and `pf_seq_id` are
  gone. Every cold multi-sequence caller takes `encode_cold_layer_attention`.
  The existing `iter_g_a` hosted controls are updated to the per-sequence
  descriptor and the 33-row floor. The test
  `gemma_cold_batch_preserves_existing_tiny_aggregate_and_overflow_refusals`
  gains the 32-row case.

### 6.2 Official artifact (ignored; root runs on the GPU lane)

Helpers are `lane_image`, `slot_kv_hashes`, `slot_cursors` and
`assert_lane_equal` (`engine_gemma_warm_batch_tests.rs:76-170`).
1. **`gemma_joint_cold_prefill_matches_per_slot_bitwise`.** Transaction level,
   two arms in one process (per slot, then cohort, with slot resets between).
   Lane sets:
   - the four actual turn-0 prompts from `$G/cases/hf2q` (S 458, P 465);
   - synthetic lanes {33, 64, 65, 458} with cue 7;
   - {1,100 (> sliding window), 900} with cue 1;
   - {200, 300} with cue 31;
   - {500, 600} with cue 40 (the `JointLive` cue);
   - {300, 400} with cue 32 (the `PerSlot` cue);
   - a mixed pair: a 400-row lane with cue 7 and a 500-row lane with cue 40
     (`Replay` plus `PerSlot` in one transaction).

   The four real turn-0 prompts differ only in the agent id, so the
   synthetic lanes of unequal length and content carry the cross-lane
   contamination check.

   It compares first token, logits row bytes, cursors, KV hashes of all 30
   layers through P, anchor bytes, and 16 greedy decode rows' logits bytes
   after the transaction.
2. **`gemma_joint_cold_prefill_deterministic`.** 20 repeats of the turn-0 set,
   every repeat byte-identical to the first (the §0.19 determinism history).
3. **`gemma_cold_multi_seq_matches_per_slot_bitwise`** (cutover). The
   iter-G(a) cold batch (no boundary) with lanes {33, 64, 458, 900} matches
   per-slot cold prefill bitwise. A group containing a 32-row lane is
   asserted to take the per-slot route (cohort counter unchanged). The
   installed cold prefill's **cursor-0 chunk** matches a per-slot cold
   prefill of the same planned chunk `prompt[..end]` bitwise. This is a
   chunk-level claim, not a request-level one (§3.3). The existing
   `gemma_cold_multi_seq_long_lane_ring_deterministic` stays green.
4. **`gemma_joint_cold_engine_cohort_matches_serial_turn0`.** Engine level.
   Four concurrent unary turn-0 requests run with the switch on versus off.
   It checks identical responses, `x-hf2q-cached-tokens`, anchors and
   retained tokens, then runs turn 1 (the joint warm cohort) from both and
   checks identical outputs. It asserts `cold_stable_cohort` fired once, with
   4 lanes.
5. **`gemma_joint_cold_forward_fault_retries_each_lane_per_slot`** and
   **`…_cue_fault_restores_from_boundary_anchors`**. Fault injection. Outputs
   equal the switch-off arm, and the retry records list the lanes in FIFO
   order.
6. **`gemma_joint_cold_cancel_during_transaction`.** One lane closes during
   Phase B. The others equal their per-slot outputs, and the slot is
   recovered and released.
7. **Regression sweep.** The existing joint warm bitwise gates, `slot_aware_n4`,
   the tiny-prefill tests and the stable-batch tests stay green.

## 7. Acceptance gates

### 7.1 Speed

The numbers below are frozen before data. Stage shares are from #660 §2:
turn-0 span +496 ms is 18% of the +2.734 s gap, and the turn-0 queue
(422 ms) plus decode inflation (+459 ms) are what this story removes.
- **G-S1, transaction (diagnostic, quiet host).** The four turn-0 prompts, 10
  repeats per arm, cohort versus per slot, same binary. The cohort wall time
  must be ≤ 0.80 × the per-slot sum (median). The record also reports the
  Phase B, anchor and Phase C split. The per-slot sum measured today is about
  4 × 275 = 1,100 ms (§2.1), so the bound is about 880 ms, against an
  estimate of about 800 ms (§10).
- **G-S2, four-agent paired replay** (the #660 L1 method, 7 paired blocks in
  forward/reverse order, kill switch off versus on, same binary, Q5
  `CalibrationPass` on every run):
  - turn-0 span median reduction ≥ 0.20 s per agent, with a 95% bootstrap CI
    that excludes 0;
  - turn-0 server queue median ≤ 30 ms on every lane;
  - turn-0 decode median ≤ 420 ms, against 345–369 ms today for the
    last-served lane;
  - cohort size 4 on turn 0 in at least 6 of 7 runs, from the cohort log line;
  - makespan difference: **reported, not gated.** At 7 blocks a −0.20 to
    −0.35 s effect is within the run-to-run spread (the peer's own MT-vs-P0
    −0.35 s had a CI of −1.21 to +0.08 s). The turn-0 span, whose spans
    range only 1522–1577 ms, is the primary gate. The makespan verdict
    belongs to the combined candidate's Q3-design campaign.
- **G-S3, iter-G(a) no-boundary cold batch (D2, diagnostic, quiet host).**
  Four 465-row prompts without a boundary, plus an 8 × 64-row set, 10
  repeats each. Compare the per-sequence route against the block-diagonal
  route on the base binary. The per-sequence route's transaction median must
  be ≤ 1.05× the block-diagonal route's on each set. That shows the cutover
  does not regress the no-boundary batch on per-lane attention.
- **No Win claim here.** The Q4 Win (`family_verdict` == Win, ratio CI upper
  bound ≤ 0.95 against the peer's MT) is measured in Q4-GEMMA-ATTN's combined
  candidate.

### 7.2 Quality

- Every §6.2 gate is bitwise (Class A).
- `ExactTranscripts` (`q1_decision_policy.py transcripts`) across the G-S2
  runs, switch on versus off. Zero same-request/different-output groups.
- Against v0.1.28 and the peer: inherited (§5.2). The combined candidate's
  paired 24-instance campaign runs the gates.

### 7.3 Fairness and memory

- The stock fairness band is 1.025 (`NEUTRAL_RATIO`): worst-slot TTFT p95 and
  SSE-gap max against v0.1.28, as 95% CI upper bounds in the combined
  candidate. The expected effect on both is neutral or better `[INFERENCE]`:
  - turn 0 is unary in the canonical workload;
  - the cohort runs only with every slot idle, so no stream decodes during it;
  - for a streamed cold cohort, the worst-slot TTFT falls from about 4 × 275
    to about 800 ms, while the best slot's rises from about 275 to about
    800 ms.

  Against the peer, TTFT is disclosed, not gated.
- Q5: the release candidate's fresh calibration keeps Gemma's memory tier.
  The cohort's peak is recorded in G-S1 against the per-slot arm.

## 8. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| F16 FA nondeterminism at lane offsets (ADR-040 §0.19 history) | Class A determinism gate fails | Lane-local offset-0 staging; masks built and re-cast in the consuming session as the single-slot route does; gate §6.2-2 |
| The extracted cold attention changes the single-slot route | Every cold prefill changes, not only cohorts | The extraction is a move, not a rewrite. The existing `gemma_cold_prefill_handoff_skip_matches_legacy_bitwise` and per-slot reference stores (Q1 T0 `ExactEqual` against `gemma-t0-b3`, §9 step 5) must stay byte-identical |
| Coupling to the MoE prefill kernel (`mm_id`) | Row invariance lost | The Gemma prefill-MoE design states that its change keeps per-row output and the 32 threshold. The worker refuses under `HF2Q_MM_ID_ROUTING_THRESHOLD`. The gates rerun on the combined candidate |
| Smaller gain than the profile estimated | Win budget short by about 0.15–0.2 s | Stated in §10. The operator weighs it in the Win arithmetic |
| Arrival spread above the 25 ms window | Partial cohorts (3 + 1) | Measured in G-S2 (the cohort-size criterion). The window is the ADR-062 D1 qualified value and is not retuned here |
| Cutover changes iter-G(a) and installed-cold numerics | Different outputs for no-boundary concurrent cold requests compared with the current binary | They become equal to per-slot outputs (Class A), so quality inherits the serial route. The change is noted in `CHANGELOG.md` |
| Graft-bound workers | Cohorts unavailable | They keep per-slot cold admission. Under D2, graft-bound cold multi-sequence callers also go per slot until a graft bitwise gate exists |
| Long cohort transaction (about 0.8 s) | Later arrivals wait | Same or shorter than today's serial quantum (about 1.1 s). Phase B is under the 4,096-row cap, and each cue step is its own supervised call |

## 9. GPU validation plan (root; GPU lane granted in writing; HOST-LOCK held)

One model process at a time. Check `df -h /` ≥ 20 GB and free memory first.
The internal-disk artifact is `gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf`, sha256
`bf65b7e1…8ef53`. Evidence goes under `$C`. Each run records source commit,
binary sha256, artifact digest, native version and case file.

1. **Build** the candidate in its worktree with the README build
   environment, for example
   `CARGO_TARGET_DIR=/Users/robert/.local/state/hf2q/ext/cargo-targets/<agent>`,
   `RUSTC_WRAPPER=`, `CARGO_INCREMENTAL=0`, `GIT_COMMIT_SHA`. Then run:
   - `cargo test --locked --bin hf2q gemma_cold_` (hosted);
   - `cargo test --locked --bin hf2q gemma_joint_warm` and `slot_aware`
     hosted controls.
2. **Bitwise gates** (§6.2 items 1, 3, 4, 5, 6) with the official artifact:
   `cargo test --release --locked --bin hf2q <name> -- --ignored --test-threads=1`,
   logs in `$C/gates/`.
3. **Determinism** (§6.2 item 2): 20 repeats, `$C/gates/determinism.log`.
4. **Regression sweep** (§6.2 item 7) on the candidate build.
5. **Q1 witness** (§5.3): fresh T0 and T2 endpoint traces through the existing
   supervisor and composition flow (new composition keys for the new build).
   `exact` T0-new vs `gemma-t0-b3` must be `ExactEqual`, which proves the
   single-slot route unchanged. `exact` T0-new vs T2-new must be
   `ExactEqual`. Routing must be `ExactEqual`.
6. **On any gate failure:** stop. Localize with
   `q1_gemma_actual_cohort_stage_localizer` / `…all128_stage_localizer` on
   the candidate. Record the first differing stage and lane. Do not promote.
7. **G-S1** transaction timing: 10 + 10 repeats, ABBA, `$C/g-s1/`. **G-S3**
   (D2) no-boundary cold batch timing, per-sequence against block-diagonal
   route, `$C/g-s3/`.
8. **G-S2** four-agent paired replay: 7 blocks, kill switch off/on, the #660
   L1 case file and runner, Q5 `CalibrationPass` per run, `$C/g-s2/`. Then run
   `gemma_stage_profile.py compare` for the turn-0 table and
   `q1_decision_policy.py transcripts` for `ExactTranscripts`.
9. **Cleanup:** stop the server, remove HOST-LOCK, and record the receipts'
   sha256 in `$C/receipts.json`.

The combined Q4-GEMMA-ATTN candidate's Q3-design campaign (Win, quality,
fairness, fresh Q5) is the root's, not this story's.

## 10. Expected effect and the Win arithmetic `[INFERENCE]`

The joint warm fit (`adr063-gemma-four-agent-reuse.md` Stage 2: 37.7 ms per
forward + 0.241 ms per row + attention) bounds what a joint cold forward can
save:
- **Phase B.** The per-row work and the per-lane attention are unchanged. The
  saving is three of the four fixed per-forward costs (about 113 ms) plus any
  better MoE tile use. Estimate: 4 × 187 ≈ 748 ms serial versus about
  600–640 ms joint.
- **Anchors.** Unchanged, about 10 ms × 4, as part of today's 75–100 ms
  remainder.
- **Phase C.** The serial cue costs 4 × 7 single-slot steps × about 10.6 ms ≈
  300 ms. The joint cue costs 7 width-4 steps × about 19–21 ms (the measured
  turn-0 joint decode is about 360 ms for 18 tokens) ≈ 130–150 ms.
- **Total.** The transaction takes about 0.78–0.83 s, against 1.10 s for the
  serial quantum. The turn-0 span falls by about 0.27–0.32 s for every agent.
  Allowing for the fit's uncertainty, the makespan effect is **−0.20 to
  −0.35 s**.

This supersedes the profile table's 0.4–0.5 s row. The row is not edited; a
dated note is added (D4). The profile's upper bound
assumed that the 275 ms per-slot prefill carries into a joint 4 × 465-row
forward. That is not so: the per-row term dominates, and hf2q's cold per-row
cost is the prefill-kernel stories' stage (about 1.8× the peer's per row in
the #660 fit).

After this story the predicted turn-0 span is about 1.25–1.30 s, against the
peer's 1.07 s. The remaining about 0.2 s is per-row prefill cost and the
18-token decode (360 against 313 ms).

The levers sum in the profile's §7 becomes about 2.1–3.2 s, against the 3.0 s
Win needs. The per-row prefill and decode stories therefore carry more of the
Win budget than the profile assumed.

## 11. ADR-063 story section (extraction-ready)

### Q4-GEMMA-COLD. Prefill the simultaneous cold Gemma 4 first turns in one exact joint transaction

Outcome: The four simultaneous cold first prompts of the Gemma 4 26B-A4B four-agent task are prefilled in one joint transaction that is bitwise equal to per-slot prefill, so no agent waits for the other agents' serial prefills on turn 0.

Requirement: The program MUST admit every eligible cohort (design §3.1: all slots idle, two or more text lanes, each cold with an exact pre-generation boundary of at least 33 rows, at most 4,096 prompt rows in total, no graft, no diagnostic route override) of cold Gemma 4 requests into one joint prefill transaction that is Q1 Class A against the per-slot route, publishing each lane's boundary anchor, cursor and first token exactly as per-slot admission does. Every other group keeps its current route. It MUST measure the turn-0 effect on the four-agent task.

Acceptance:

- A written design, approved before implementation: `docs/research/adr063-q4-gemma-cold-design.md`. It covers the cohort rule (§3.1), per-sequence cold attention (§4.3), cue replay through the batched decode body (§4.4), failure semantics (§4.5) and kill switch `HF2Q_GEMMA_JOINT_COLD_PREFILL=0` (default on once the bitwise gates pass).

- Starting point (Q4-GEMMA #660, L1, 7 blocks, hf2q `7f295b38`): the four turn-0 requests coalesce but are admitted per slot in series. Because each carries a stable boundary, the stable batch refuses them as uncached and the cold batch refuses boundaries. The result is a server-queue staircase of 1/285/560/834 ms (lane medians), 275 ms of prefill per lane and 1.19 s of "decode" on the first-served lane. The turn-0 span is +496 ms against the peer's MT, 18% of the 2.73 s makespan gap.

- Class A: on the official artifact, every cohort lane's first-token logits, cursors, all 30 layers' slot KV through the prompt, boundary anchor bytes and 16 following decode rows equal the per-slot route byte for byte. This covers the turn-0 prompts and the §6.2 synthetic shapes, including the 32-row per-slot cue and mixed {7, 40} cues, and is deterministic over 20 repeats. The iter-G(a) cold batch moved to the same per-sequence attention (lane floor 33 rows) also matches per slot bitwise. For the installed long-prompt cold prefill, only the cursor-0 chunk forward is claimed exact. Any differing byte blocks; no Class B fallback ships without a separate operator decision and the Q1 §3.5 campaign.

- `ExactTranscripts` on the four-agent workload, kill switch on against off.

- Speed, paired four-agent replay (same binary, kill switch off against on, 7 paired blocks, Q5 `CalibrationPass` per run): turn-0 span median reduction ≥ 0.20 s per agent with a 95% CI excluding 0; turn-0 server queue median ≤ 30 ms; turn-0 decode median ≤ 420 ms; cohort size 4 in at least 6 of 7 runs. Makespan is reported, not gated; the makespan verdict belongs to the combined candidate. The transaction itself takes ≤ 0.80× the per-slot sum on the turn-0 prompts. The no-boundary cold batch on the new attention route takes ≤ 1.05× its previous transaction time.

- The stock fairness band 1.025 (worst-slot TTFT p95, SSE-gap max) and the Q5 memory tier hold in the Q4-GEMMA-ATTN combined candidate, which carries this change; the Q4 Win is measured there, not here.

Repository ownership: hf2q owns the cohort admission, the per-sequence cold attention route, the cue replay, the tests and the campaign. It uses existing mlx-native kernels at the current pin; no mlx-native change.

Validation: Independent review of the design, the bitwise and determinism gate receipts, the Q1 T0/T2 `ExactEqual` witness on the candidate build, the `ExactTranscripts` record and the paired four-agent replay from raw receipts.

Predecessors: Q4-GEMMA (#660, profiling result), Q5 (#371).

Priority: High. Core scope (second-tier family).

Delivery boundary: Qualified implementation merged to main for v0.1.30 and included in the Q4-GEMMA-ATTN combined candidate.

## 12. Decisions for the operator

- **D1. Class A only.** If a bitwise gate fails, the story stops for
  localization, and no Class B joint cold route ships in its place without a
  separate decision and the Q1 §3.5 campaign. Recommended: yes.
- **D2. Clean cutover of the cold multi-sequence attention.** The new
  per-sequence cold attention replaces the block-diagonal BF16 route in every
  cold multi-sequence caller: the iter-G(a) cold batch and the installed cold
  prefill at cursor 0. It comes with three conditions (ReviewGemmaCold):
  - iter-G(a)'s lane floor rises from 32 to 33 rows (§3.3);
  - the installed long-prompt prefill claims exactness only for its cursor-0
    chunk;
  - G-S3 shows the no-boundary batch does not regress.

  Graft-bound callers go per slot until a graft bitwise gate exists. This
  removes a second cold convention, and Q1 T2 becomes a Class A pair. The
  alternative keeps the old route for no-boundary cohorts and adds the new
  one only for the stable cold cohort. Recommended: cutover.
- **D3. Default on with kill switch** `HF2Q_GEMMA_JOINT_COLD_PREFILL=0`, as
  for joint warm prefill. The bitwise gates are the merge contract.
  Recommended: yes.
- **D4. ADR wiring.** Add Q4-GEMMA-COLD as a predecessor of Q4-GEMMA-ATTN,
  whose closing campaign is the combined candidate.
  - **#660 §7 row.** Do not edit the reviewed row in place. Add a dated note
    (2026-11-04, this design §10): the cold-turn lever is re-estimated at
    −0.20 to −0.35 s (from 0.4–0.5 s), which moves the sum of all levers to
    about 2.1–3.2 s against the 3.0 s Win requirement.
  - **If D1 stops this story** (a Class A gate fails and localization finds
    no exact fix inside the story), the combined Q4-GEMMA-ATTN candidate
    proceeds without Q4-GEMMA-COLD and with its kill switch default off or
    its code reverted. The root restates the Win arithmetic without this
    lever (about 1.9–2.9 s of levers against 3.0 s) in the ADR before the
    closing campaign. The story then keeps its localization record as its
    verdict and goes to the operator as `needs_decision`.
