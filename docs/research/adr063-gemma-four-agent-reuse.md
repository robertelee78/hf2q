# ADR-063 Gemma 4 four-agent reuse: checkpoint promotion, joint warm prefill, first turn

Status: source change staged on `perf/gemma-four-agent-reuse` (on origin/main
`0339c552`), revised after the independent review
(`/tmp/adr063-gemma-reuse-independent-review-v1.md`, ACCEPT-WITH-CONDITIONS).
No test, build or hardware run has been executed by the author. The default
goes ahead only if the root runs every gate below, including the quality
campaign, and they pass. If any gate fails, set the default back to off
(`Mode::Default.decide` → `Ok(false)`) before merging.

## Q1 classification: Class B, route-sensitive

The restore oracle below is Class A evidence for the cache mechanism only. The
promotion itself moves production Gemma tool turns onto a different route
relative to today's default, so it is Q1 Class B (`RouteSensitive`):
- rows `[458, P)` of a tool turn now come from the previous turn's prefill
  instead of being recomputed in this turn's suffix prefill;
- any later request extending a published full checkpoint inherits them;
- tagged cold tool turns leave the BF16 multi-seq cold cohort for the F16
  single-slot route.

ON and `=0` transcripts therefore diverge (`q1_decision_policy.py
transcripts` is expected to report `DivergedTranscripts`; the 15.87 s ON run
already saw 1249 vs 1254 prompt tokens). This is not an exactness defect.
Quality is decided by the 24-instance paired campaign, ON vs `=0`, against
stock (root runs it; `paired` must not return `Loss`, `BelowFloor` or
`Inconclusive`), or by an explicit operator waiver.

## Starting point

On the four-agent Gemma 4 26B-A4B coding workload, hf2q's makespan was
22.3–28.5 s and the peer's was 11.8–15.5 s (`E/peer-arms-v1/gemma-blk-*`). The
gap breaks down as follows:

- **About 5.5 s: tool-turn cache misses.** Every tool turn reused only the
  458-token anchor (`cache="prompt-anchor" anchor_tokens=458`).
- **About 1.9 s:** warm prefills ran one slot at a time.
- **About 1.8 s:** the first turn was slow.

The full-input checkpoint fixes the first item. It was merged in #415 behind
`HF2Q_GEMMA_FULL_INPUT_CHECKPOINT=1` plus `HF2Q_UNSAFE_EXPERIMENTS=1` and
measured a 39.87% semantic-TTFT improvement (95% CI [39.52, 40.34]). In the
four-agent run it reached 15.87 s and cached exactly the previous prompt on
each turn (514/714/943/1249/1408).

## Why it was gated, and what this change adds

The gaps below are the ones recorded in ADR-040 (2026-10-26 sections) and
`adr063-gemma-full-input-four-agent-proof.md`.

| Gap | Source-level guarantee now | Proof (test) |
|---|---|---|
| Exactness of resume vs cold | A turn resumed from a restored checkpoint must equal, bit for bit, the same-plan resume from a live image that was never overwritten. This is the repo's cache contract (see the comment on `gemma_fresh_and_reused_4096_8193_bounded_outputs_match`). A monolithic cold prefill is a different Metal graph: the V2 cross-route attempt differed by 0.81 in max-abs logit. It is printed as a diagnostic only. | `gemma_full_input_engine_resume_cancel_evict_match_uncheckpointed_reference` (ignored). It covers both the batched 64-token suffix and the tiny 5-token decode-replay suffix after decode has wrapped the 1024-row ring. It compares text, completion tokens, finish reason, and logprob bits (top-4). |
| Scheduler/HTTP admission and publication integration | The test drives the real `Engine` request path: admission, capture at Ready, publication, restore. | Same test. |
| Cancellation | Decode-time cancellation restores the published checkpoint through `recover_gemma4_slot_after_cancellation`. | Same test, cancellation arm: the stream is dropped after three token events, the test-only cancellation-restore witness must advance by one, and the next turn must have `cached_tokens == P` and match the reference bit for bit. The reference arm asserts it took the live path (admission-restore witness unchanged) and the checkpoint arm asserts one restore per warm turn. |
| Eviction / isolation | Affinity requires the entire checkpoint to be a token prefix (pre-existing `gemma4_slot_affinity`, already covered by `gemma_cold_batch_*`). A cold or diverged admission clears the anchor and resets the slot. | The engine test checks that a foreign conversation gets 0 credit and that an evicted conversation returns cold and matches the reference cold. |
| Memory/anchor quota under four slots | Each activated worker thread owns its own 1536 MiB quota, so the per-worker fit check is complete even with several Gemma engines in one pool. The worker activates only if two generations × slots at the longest prompt ledger fit (`anchor_budget::QuotaPlan`). Every capture role (full input, exact image, stable boundary, stable batch, request-start rollback) goes through `capture_or_skip`: a quota refusal keeps the previous anchor and never fails the request. | `gemma_anchor_quota_is_per_worker_so_two_full_workers_never_refuse` (two concurrent full four-slot plans at the official ≈151.8 MiB per slot), `gemma_anchor_capture_quota_refusal_skips_every_role_and_keeps_other_errors`, and the ignored `gemma_full_input_four_slot_geometry_holds_two_generations_within_quota` (real scaffold: exact reservation, then refusal past the limit). |
| Unsafe ack / routing | The ack is removed. Activation is per worker thread. An inactive worker ignores tags everywhere, including cold/stable batch routing, so it behaves exactly like `=0`. `=0` is decided before any planning and also skips the slot-route warmup; default mode treats a plan error as "unsupported". | `gemma_full_input_default_on_kill_switch_and_forced_matrix`, `gemma_full_input_tag_leaves_cold_batch_only_on_an_active_worker`, `gemma_full_input_tag_is_inert_on_an_inactive_worker_thread`. |

"At most two generations per slot" comes from reading every capture site
(`take_gemma4_request_rollback_anchor`, Ready capture, the stable boundary,
the stable batch, cancellation recovery); it is `[INFERENCE]`. The quota stays
the hard backstop.

### Activation

`HF2Q_GEMMA_FULL_INPUT_CHECKPOINT` takes three valid values:
- **Unset:** on when the worker is Gemma, SlotAware with 1–4 slots, Hybrid,
  TQ8, and the quota plan fits. Otherwise off; startup never refuses.
- **`0`:** off (kill switch).
- **`1`:** on; refuses at spawn if any condition fails. In a multi-model pool
  it also refuses every non-Gemma load.

Any other value refuses at serve start.

Isolation note: `usage.prompt_tokens_details.cached_tokens` can now reveal
that some conversation sent the identical full prompt P, not just the shared
458-token prefix. This is the same class as any prefix cache (the peer has it
too); hf2q has no tenant key. Recorded, not fixed.

## Joint warm prefill (item 2)

Branch `perf/gemma-joint-warm-prefill` (cut from `perf/gemma-four-agent-reuse`
`b4148824`). Source change only; no build, test or hardware run has been
executed by the author. Default on because the bitwise gate below is the
contract; the root runs it and turns the default off
(`JOINT_WARM_PREFILL_DEFAULT`) if it fails.

### Why the earlier "not tractable" note was wrong

It rested on the cold multi-sequence path (`use_fa_f16 =
multi_seq_prefill.is_none()`, BF16 block-diagonal FA). A single-slot *warm*
prefill never uses FA: `forward_prefill_batched_live_resume_slot_mounted`
always runs `flash_attn_vec_hybrid_batched` over F16 K and TQ-HB V in
256-query chunks (split at 512/1024), with a ring stage of
`cap - 1 + min(rows, 256)` rows. The resumed multi-sequence path
(`forward_prefill_batched_multi_seq_live`) never read `use_fa_f16`; it
defaulted to tiled F16 FA over dequantized V (`HF2Q_GEMMA_TILED_LIVE`), sized
its ring stage from the cohort's total rows and read globals through the
full scaffold with real slot ids. That route choice, not FA determinism, is
what made joint ≠ per-slot.

Baseline (A arm `E/gemma-reuse-ab-v1/g-1-A-*`, design scout): tool-turn
prefill totals 5.40 s over turns 1–6, serial at 74–406 ms per slot. Turns 1,
4 and 6 arrive within the 25 ms window; inline admission yielded after every
prefill, so slots 1–3 were admitted between decode steps and queued behind
them (+24 to +92 ms creep per later slot).

### Stage 1 (this change): exact joint cohort

- **Attention route parameter.** `forward_prefill_batched_multi_seq_live`
  takes `GemmaLiveAttention`. `PerSlotExact` runs, per lane, exactly the
  single-slot live attention: the vec-hybrid kernel, the same
  `gemma_live_query_chunk_len` chunks from each lane's absolute start, a
  per-lane ring stage `cap - 1 + min(T_i, 256)`, the lane's slot view
  addressed as slot 0 (global layers too), identical `kv_seq_len` /
  `kv_capacity` / `nsg`, per-chunk ring commit, and the FWHT sign undo. `Tiled`
  keeps the installed long-prompt lanes and the stable batch on their
  existing route. Live resume no longer builds unused block-diagonal masks.
- **Row-parallel work once.** Embedding, QKV/O projections, norms, RoPE (per
  lane absolute positions), dense FFN, MoE and the per-row m=1 head run once
  over `Σ T_i` rows.
- **Kernel-route rule (`MIN_SUFFIX_TOKENS = 33`).** The official GGUF stores
  Q4_K ×192, Q5_0 ×32, Q8_0 ×28, Q6_K ×14 and F32 ×392 tensors (no Q4_0, so
  the frozen-shape Q4 route never applies). mlx-native 0.15.15
  `plan_dense_auto_route` takes the mm route for `m > 8` and the same
  device-selected large-tile kernel for every `m > 8`; `plan_expert_auto_route`
  takes `mm_id` only for `n_tokens > MM_ID_ROUTING_THRESHOLD` (32). A per-slot
  warm prefill with `T ≤ 32` would use `mv_id` (and below 32 it is a decode
  replay) while the joint stream uses `mm_id`, so every lane needs `T ≥ 33`.
  The constant is derived from `mlx_native::MM_ID_ROUTING_THRESHOLD`; the
  worker keeps the cohort off when `HF2Q_MM_ID_ROUTING_THRESHOLD` or
  `HF2Q_PREFILL_SLOT_BATCHED=0` changes the per-slot route. That each output
  row's reduction is independent of the other rows in an mm/mm_id tile is
  `[INFERENCE]`; the gate proves it on hardware.
- **Admission (`engine_gemma_warm_batch.rs`).** With every slot idle, the
  quantum's longest contiguous FIFO prefix of plain-text requests (no soft
  tokens, vision identity or stable cue split; full-input-tagged tool turns
  included) whose preferred slots are distinct, each with `cached > 0` and
  `T ≥ 33`, within the 4,096-row cap and at least two lanes, forms the
  cohort. Those requests are admitted in FIFO order through the normal
  admission (anchor restore, rollback anchor, cursor) and installed as
  GPU-free `Gemma4PrefillState` lanes tagged with a FIFO admission ordinal
  (`joint_warm: Some(ordinal)`), with no yield in between. A lane whose
  actually selected slot breaks the rule is admitted inline exactly as
  before. Tagged requests still stay out of the cold and stable batches.
  `run_installed` then drains every joint lane before the scheduler steps,
  earliest admission first: cohorts of two or more lanes (FIFO admission
  order, 4,096-row cap) in one transaction, any remainder per slot with
  today's one-shot `forward_prefill_with_soft_tokens_slot_aware_resume`.
- **Capacity (decode reserve).** Every request is validated only for its own
  `prompt + graft + max_tokens ≤ ctx`, while the forward checks
  `roundup32(max_i end_i + graft + reserve)` against the slot capacity.
  `gemma4_batch_decode_reserve` passes
  `reserve = max_i(end_i + max_tokens_i) − max_i(end_i)` (saturating, at
  least 1), so the bound equals the binding lane's own requirement and a
  cohort fits whenever every lane fits alone. The same rule now feeds the
  other resumed multi-lane calls (installed long-prompt resumes and both
  stable-batch passes), which previously paired the largest `max_tokens`
  with the longest lane. Cold multi-lane prefill keeps the largest
  `max_tokens`; it no longer needs it to cover the whole concatenated
  stream (see "Multi-sequence cold capacity check" under Remaining gaps).
  The reserve only sizes capacity checks on the live route; it changes no
  numerics (the gate checks a per-slot prefill at a near-limit `max_tokens`
  against the 16-token reference).
- **Publication.** After the shared forward, cursors commit all-or-nothing,
  then each lane, in FIFO admission order, runs
  `finish_committed_transaction`: full-input capture (quota refusal keeps the
  previous anchor), seed from its own logits row, decode-state install. Under
  a binding checkpoint quota, captures are attempted in the same order as
  serial admission, so the lane that is refused is the one serial admission
  would refuse.
- **Failure and cancel semantics** (`settle_gemma4_prefill_batch`, shared
  with installed long-prompt lanes):
  - a pre-submission check failure (substrate, whole-suffix plan, cursor)
    leaves the lanes installed and prefills the primary per slot; the rest
    retry as a smaller cohort or per slot;
  - a submitted non-fatal forward error restores every cohort lane from its
    own request-start rollback anchor (`restore_gemma4_lane_for_retry`, which
    keeps the anchor) and re-prefills it per slot in FIFO order, exactly as
    if admitted alone. The joint error is not attributed to a lane: a fault
    that belongs to one lane (for example a non-finite argmax, or invalid
    routing that NaNs the shared `mm_id` output) recurs in that lane's
    per-slot prefill and fails only that request, as serial prefill would.
    A lane whose anchor cannot be restored is recovered, released and
    failed request-locally. Installed long-prompt lanes keep the previous
    behaviour (restore and fail, no retry);
  - a worker-fatal GPU error (watchdog, command-buffer fault) fails the
    worker, as for every Gemma transaction;
  - a non-fatal per-lane publication failure restores and fails that lane
    only (previously an installed-lane publication failure stopped the
    worker);
  - a client that disconnects before submission is left out and cancelled
    by its per-slot advance; one that disconnects during the forward is
    restored from its just-published checkpoint afterwards while the other
    lanes continue.
- **Kill switch.** `HF2Q_GEMMA_JOINT_WARM_PREFILL=0` (read once per worker,
  logged as `Gemma joint warm prefill worker decision`). Each cohort logs
  `Gemma4 installed prefills advanced in one aggregate-bounded transaction`
  with `transaction="joint warm"` and `requests=N`.

Expected [INFERENCE]: −1.2 to −1.5 s of the 5.4 s tool-turn prefill (three
fewer fixed per-forward costs per turn, no decode-queue creep, better MoE tile
use), plus collapsed client send spread on later turns.

Q1 class: Class A against the per-slot route (bitwise by gate). It changes
no numerics, so it adds nothing to the full-input checkpoint's Class B
quality campaign. Stage 2 below moves both the per-slot and the joint route
to one new kernel; the Stage 1 gate now runs on both routes.

### Stage 2 (this change): tiled F16 warm attention

Branch `perf/gemma-warm-tiled-attention` (cut from
`perf/gemma-cold-prefill-alloc` `fdd25edf`). Source change only; no build,
test or hardware run has been executed by the author.

**Why.** In the default arm of `E/gemma-alloc-ab-v1/g-1-N-*`
(`server.stderr:92-233`; the transaction times of `g-5-N` and `g-7-N` agree
within 5 ms) the six joint warm cohorts take 132.9, 424.7, 516.9, 813.5,
451.7 and 1,085.7 ms of forward (224/800/916/1,244/636/1,540 rows; 3,425 ms in
total). A three-term least-squares fit, `fixed + a·rows + b·vec_traffic`,
matches all six within 21 ms. The `vec_traffic` term is the number of keys
each query reads: 25 sliding layers at `min(pos, 1024)` keys, plus 5 global
layers at `pos` keys weighted 2× for head dim 512. The fit gives
37.7 ms + 0.241 ms/row + 10.5 ms per million key-reads. So
`flash_attn_vec_hybrid_batched`, which re-reads the whole history once per
(query, head), accounts for about 1.91 s of the 3.43 s
(40/181/279/475/260/674 ms per turn) `[INFERENCE: fit, not a kernel
profile]`.

**What.** `src/serve/gemma_live_attention.rs` holds the one per-sequence
live attention. Single-slot resumes (`forward_prefill_batched_live_resume*`,
SerialFifo included) and every multi-sequence lane both call
`encode_live_layer_attention`, passing each sequence's own KV view
addressed as slot 0. Per sequence it:
- appends a global layer's whole suffix first;
- for a sliding layer, stages up to `cap − 1` ring history rows plus the
  chunk (`cap − 1 + min(T, chunk)` rows, sized per sequence), attends, and
  only then commits the chunk into the ring;
- quantizes TQ-HB V exactly as before, and runs the FWHT sign undo once per
  layer over the output rows.

The kernel is chosen by `LiveRoute`:
- **`TiledF16`** (the default for warm resumes): compact K to dense F16 and
  dequantize the active V range to F16 (F16 V is compacted).
  - Sliding layers: `build_sdpa_mask_f16` (SWA window, causal,
    `q_abs_offset` = staged history) plus `blk`, then
    `flash_attn_prefill_f16_d256_with_blk`, in 1,024-query chunks.
  - Global layers: `flash_attn_prefill_f16_d512_resume` with
    `q_offset_in_k = graft + start`, in 4,096-query chunks (the transaction
    cap; SerialFifo rejects longer Gemma prompts, so in serving every
    sequence takes one chunk).
  - Every dispatch is `batch = 1` over one sequence.
- **`VecHybrid`**: the previous kernel, chunking (256 queries, split at
  512/1024) and stage positions, unchanged.

Route rules:
- Warm resumes (single slot and `PerSlotExact`) read
  `MlxModelWeights::warm_live_kernel`, resolved once per model load from
  `HF2Q_GEMMA_WARM_TILED_ATTENTION`. Suffixes under 32 rows keep
  `VecHybrid`: below the ADR-040 2026-08-14 tiny-prefill containment
  boundary the tiled route has no serving evidence, and the joint cohort
  never admits such a lane.
- Installed long-prompt lanes and the stable batch (`Tiled`) keep
  `HF2Q_GEMMA_TILED_LIVE` and tile every length, as before.

The old multi-sequence branch (about 600 lines) and the single-slot branch
(about 380 lines) are deleted. Live resume allocates the vec-hybrid slot
ids, stage positions and scratch only when some sequence uses that kernel.
The prefill banner prints `live=…`.

**Installed-lane behaviour.** With graft 0 and TQ-HB V (production), the
installed lanes issue the same kernel inputs as before. Their stage is now
per sequence instead of cohort-sized, which only changes the stride of a
buffer that is compacted before attention, and each lane is at most 4,096
rows, so it takes one global chunk. Two latent defects of the old tiled
branch are fixed:
- With a bound graft, `q_offset_in_k` omitted `graft_len`, hiding each
  query's last `graft_len` keys.
- F16 V fell back to vec-hybrid with 1,024-query chunks but 256-chunk stage
  positions.

**Bitwise single == joint.** A lane and the same sequence alone resolve the
same `LiveRoute` (`single_slot_and_joint_cohort_always_resolve_the_same_route`).
They then issue the same dispatches with the same arguments, over Q/K/V rows
that Stage 1 proves equal, into per-sequence scratch holding only that
sequence's rows. The masks and `blk` depend only on `(len, kv_seq_len,
q_abs_offset)`.

**Determinism.** ADR-040's F16-FA concerns do not apply to this route:
- §0.20's concern was F16 FA over a block-diagonal multi-sequence mask, and
  its isolation failures were root-caused to the `blk` classifier
  (mlx-native `cb9806c`). This route never builds a block-diagonal mask.
- The §0.19 long-prompt non-determinism was re-validated as resolved for
  single-sequence F16-D512 FA in §7.32K (2026-07-01): N=1 at 8k ×30,
  N=8 concurrent ×20, 16k ×20, 32,000 × 8 slots × 3, after fixes
  `cb9806c`, `609dfdc6`, `f070d50` and `8b16039`. Production has routed
  globals above 8,192 tokens through it since.
- mlx-native 0.15.15 `flash_attn_prefill*.metal` contains no atomics. Each
  threadgroup reduces its Q tile over the K blocks in a fixed order.
  Trailing partial K/V tiles are loaded guarded (`load_safe`,
  `flash_attn_prefill.metal:1409-1447,1543`; `row < kL` in
  `flash_attn_prefill_d512.metal:708-731,951-971`), so uninitialized scratch
  past `kv_seq_len` is never read.

Determinism is still a hardware gate below
(`gemma_warm_tiled_attention_is_deterministic`, repeated under
`MLX_PRECOMPILED_METALLIB=0`).

**First hardware run (`a2bc2007`, `E/gemma-stage2-gates-v1`) and fix.**
Passed: the vec-hybrid bitwise gate, the engine cohort gate, determinism
(5 runs), and the full-input, cold-prefill and Category 1 gates. Two gates
failed:
- The tiled bitwise gate: `case 6 lanes=[1, 2, 3] slot 3` differed from the
  per-slot reference at logit 0, while the 4-lane and `{0,3}` cohorts
  matched.
- The envelope: the last-row max |Δlogit| was 1.15 / 4.07 / 3.64 / 2.81,
  with top-1 flips on slots 1 and 2 (vec margins 0.17 and 0.39).

Root cause, from reading the code: the inherited tiled branch encoded the
sliding mask with no declared write. `build_sdpa_mask_f16` allocates its
output internally, so the conflict tracker (`GraphSession::barrier_between`,
which only sees declared ranges) never saw that write. The following
`barrier_between(&[mask], &[blk])` therefore emitted no barrier, and on the
concurrent encoder the `blk` classifier ran alongside the mask fill.
`alloc_buffer` zero-fills the mask, and `blk` treats an all-zero tile as
fully attended (`flash_attn_prefill_blk.metal:254-255`, `res = 2`). So
wherever `blk` read before the fill, attention skipped the mask: queries
saw future keys in their diagonal tiles and keys outside the window. How
often that happens depends on what else is running on the GPU. That
explains the tiled bitwise failure: results were reproducible within one
cohort shape but differed between shapes. The fix (`b0f96d86`) cleared it;
see the results table. It did not explain the envelope, which measured the
same after the fix (see the envelope disposition below).

The installed long-prompt `Tiled` lanes had the same race before this
change. The cold prefill path builds its masks with the same undeclared
write (`forward_prefill_batched.rs`, `build_sdpa_mask_bf16` followed by the
cast and the `blk` classifier). It is left unchanged and recorded under
Remaining gaps.

Two fixes in `gemma_live_attention.rs`:
- An explicit `s.barrier()` after the mask fill, before `blk`.
- The ring stage barrier now also declares its reads of the normed K/V
  stream. Before, a sequence's suffix copy into the stage could share a
  concurrent group with the head norm/RoPE that writes those rows. This
  applies to the vec-hybrid route too, and only changes ordering.

The bitwise gate now also runs the cohorts `{1,2,3}` twice, `{3,2,1,0}` and
`{2,3}`.

**Q1 class: Class B, route-sensitive.** Warm-prefill numerics change once
against the vec-hybrid route. Evidence:
- the decision envelope (`gemma_warm_tiled_attention_decision_envelope`:
  per-lane max |Δlogit|, top-1 agreement, top-1/top-2 margins);
- the paired 24-instance quality campaign, default (subject) against the
  kill switch (reference `stock-hf2q`, `GEMMA_WARM_TILED_ATTENTION=0`).

**Default and revert rule.** On by default
(`WARM_TILED_ATTENTION_DEFAULT`). If any of the following fails, set
`WARM_TILED_ATTENTION_DEFAULT = false` in `gemma_live_attention.rs` before
merging (`=0` stays the operator escape):
- the bitwise gate (tiled or vec-hybrid);
- the determinism gate;
- the envelope's top-1 rule;
- the campaign verdict (`Loss`, `BelowFloor` or `Inconclusive` without an
  operator waiver);
- the A/B, if it is not faster.

**Expected saving `[INFERENCE]`.** Tiled attention does
`4 · hd · heads` flops per (query, key) pair, without causal tile skipping:
62/282/435/741/406/1,051 GFLOP per turn, 2,978 GFLOP in total. At 10–20
TFLOP/s effective (the design's 290 GFLOP ≈ 20 ms per turn-6 slot is
14.5 TFLOP/s) that is 0.15–0.30 s. Add about 5 ms per cohort of extra
dispatches (mask, `blk`, compaction, dequant, casts; 6 cohorts). Arithmetic:
1.91 − (0.15…0.30) − 0.03 ≈ **1.6–1.7 s** off the critical-path warm
prefill. Joint warm prefill drops from 3.43 s to about 1.75 s, and the
four-agent makespan from 15.2 s to about **13.5–13.7 s**, against the peer's
12.4 s. If the fit's attention share is 25% high, the saving is about
1.2 s (makespan about 14.0 s).

The earlier design estimate (−2.0 to −2.2 s) was taken against serial
per-slot prefill, including decode-queue creep that Stage 1 already
removed.

## First turn (item 3)

The gap scout reported about 98 ms per decode step. That was misread:
`decode_time_secs` includes waiting behind the other slots' serial prefills.
In blk-1/4/5 and the full-input run, the last-admitted slot decodes 17 tokens
in 0.34 s, about 20 ms per step. The turn-0 cost sits in each slot's
`prefill_time` of about 0.55 s:

- **458-token boundary prefill:** 188 ms, and 264–392 ms for the first slot
  only.
- **The remaining ~0.3 s per slot:** first misattributed to the 7-token
  cue. A stack sample (`/tmp/adr063-gemma-turn0-sample.txt`) and JSON-log
  timelines (`gemma-turn0-diag-v1`) show where it actually goes: the
  slot-mounted batched prefill builds SerialFifo-only outputs that nothing
  reads (see "Cold admission allocation" below). A joint cue replay measured
  158 ms for all 28 cue tokens, against about 0.3 s for the per-slot cue.

Fixes:
- **First-use prefill penalty (cheap exact warmup, implemented):**
  `engine_gemma_slot_warmup.rs`. After the legacy one-token warmup, it runs
  one 256-token slot-aware cold prefill in a clean idle slot, then reinstalls
  cursor 0. No request can observe that slot. Proof:
  `gemma_slot_route_warmup_does_not_change_outputs_or_retained_slots`
  (ignored). It reports no timing: in one process the cold engine warms
  pipelines and pages for the warm one, so benefit is measured only by the
  A/B TTFT. `=0` skips this warmup, so the kill-switch arm is stock serving.
  A non-fatal warmup failure logs and continues; only a worker-fatal GPU
  fault fails the load.
- **Cue replay:** prototyped on `perf/gemma-cold-turn` (joint batched-body
  cue replay, bitwise gates pass). Measured gain: turn 0 went from 2.78 s to
  2.65 s, and makespan was unchanged (16.55 s vs 16.5 s).
- **Cold admission allocation (implemented, below).**

### Cold admission allocation

**Sample evidence.** Each cold four-agent admission takes about 500 ms
(`Gemma4 slot admission planned` to the next one). Its 458-token prefill
takes 202–236 ms of that. In the worker stack under
`prefill_seed → forward_prefill_with_soft_tokens_slot_aware_impl →
forward_prefill_batched_inner` (303 samples):

| Where | Samples |
|---|---|
| Waiting on the GPU (`commit_and_wait`) | 108 |
| `MlxDevice::alloc_buffer → __bzero` at the four LCP-snapshot allocation sites | 68 / 67 / 34 / 11 |
| Snapshot copies (`memmove`) | 15 |

`alloc_buffer` zero-fills every new buffer. The 4 : 4 : 2 : 1 ratio matches
the snapshot legs: dense K, dense V (F32), hybrid K (F16), hybrid V codes
(U8).

**Cause.** `forward_prefill_batched_inner` gated its end-of-prefill LCP
snapshots on `!live_prefix_resume` only. The non-batched sibling also
requires `!slot_aware`. So every cold slot-mounted prefill (the
SlotAware cold route) did two things nothing reads:

- **LCP snapshots.** Sized `max(sw, tile(seq_len + max_tokens))` rows for
  every layer, sliding layers included. Turn 0 has `seq_len` 458 and
  `max_tokens` 8192, so that is 8,672 rows per layer:

  | Leg | Bytes |
  |---|---|
  | Dense K + V (F32) | 3,907,256,320 |
  | Hybrid K (F16) + V codes (U8) + V norms (F32) | 1,472,505,600 |
  | **Total, zero-filled per admission** | **5,379,761,920** |

  The 458 valid rows were then copied in (284 MB). The result was parked on
  `dense_kvs_snapshot_for_lcp` / `hybrid_kv_snapshot_for_lcp`, which only
  the SerialFifo engine paths consume. So the worker also held about 5.4 GB
  per worker, about 10.8 GB at the moment the next admission allocated.
  These snapshots require `HF2Q_KV_LCP_RESUME_CAPACITY` to admit them; the
  four-agent launcher sets it to 8g.
- **Dense K/V decode handoff.** 774,635,520 bytes, plus the dense flash
  tmp and 30 per-layer copy dispatches. The slot-mounted caller discards it
  immediately, because it restores its own `dense_kvs`. No prefill kernel
  reads it: its comment says so, and the hybrid writes carry their own
  `barrier_between` on the same sources. The cold multi-sequence prefill
  (`forward_prefill_batched_multi_seq`) had the same discarded handoff and
  snapshot gate.

**Fix.** `gemma_batched_prefill_builds_dense_handoff` limits the handoff,
its flash tmp, its copy and the snapshots to the SerialFifo cold route. A
live resume, a slot mount or a multi-sequence descriptor builds none of
them. The copy window (`n_copy`) and the global-capacity check now use the
planned capacity, which equals what the handoff allocated, so every hybrid
and HB write is unchanged.

This is a removal, not a reuse. The buffers had no consumer, so there is no
persistent scratch to keep and nothing that could carry stale data between
slots or conversations. Kernels that need zero-filled scratch are untouched:
the per-prefill activations, masks and the live flash tmp still come from
`alloc_buffer`.

**Expected saving `[INFERENCE]`.** The snapshot sites hold 65% of the
sampled admission time (180 bzero + 15 copy + 2 allocation samples of
303). That is consistent with the ~0.28 s per admission spent outside the
prefill. Four serial cold admissions therefore save about 1.1 s: turn 0
goes from 2.65 s to about 1.55 s against the peer's 1.0 s, and the
four-agent makespan from 16.5 s to about 15.4 s against 13.9 s. The
handoff removal adds an unmeasured few tens of ms. The 775 MB of pages are
recycled between admissions and do not appear in the sample. About 5.4 GB
of resident memory per Gemma SlotAware worker goes away.

**Other paths checked.**
- **Warm single-slot and joint warm prefills** (`live_prefix_resume`) never
  built the handoff or snapshots. Their per-call allocations are the
  suffix-sized activations and the live flash tmp.
- **Decode** still allocates its batched scratch per step
  (`BatchedDecodeBuffers::new`, about 25 small `[N, …]` buffers) and the
  head's logits buffer (`encode_lm_head_into`, `N × 262,144 × 4` bytes).
  In the sample that is 8 of 205 decode samples (about 4%, ~0.8 ms per
  20 ms step), plus residency-set churn on release. A persistent per-width
  decode scratch is a possible successor for the per-step gap, but it needs
  its own audit of which of those buffers kernels read before writing.

## Root commands

Focused model-free tests:

```sh
cargo test --locked --bin hf2q gemma_full_input_
cargo test --locked --bin hf2q gemma_anchor_
cargo test --locked --bin hf2q gemma_slot_warmup_
cargo test --locked --bin hf2q gemma_cold_batch_
cargo test --locked --bin hf2q anchor_c1_gemma
python3 -B -m unittest discover -s scripts/four_agent -p 'test_*.py'
```

Then replay the CI activation step (`.github/workflows/ci.yml`, "Gemma
full-input checkpoint activation matrix") on the release binary: `=1` prints
the "forced on" line and no REFUSED, `=0` the kill-switch line, `=2`
`REFUSED HF2Q_GEMMA_FULL_INPUT_CHECKPOINT`, unset nothing.

Category 1 Gemma regression under the new default (each run separately):

```sh
for f in gemma_fresh_and_reused_4096_8193_bounded_outputs_match \
  slot_aware_staggered_eviction_no_peer_perturbation \
  slot_aware_n4_per_slot_parity_vs_serial \
  slot_aware_n4_batched_body_same_prompt_parity_vs_serial; do
  HF2Q_BYTE_EQUIV_E2E=1 HF2Q_BYTE_EQUIV_E2E_GGUF="$GGUF" cargo test --locked --release --bin hf2q "$f" -- --test-threads=1
done
```

Real artifact. Use one process each, with no other runtime loaded.
`$GGUF` is the official Google Gemma 4 26B-A4B GGUF
(`bf65b7e1…ef53`):

```sh
for t in \
  serve::api::engine::gemma_full_input_serving_tests::gemma_full_input_engine_resume_cancel_evict_match_uncheckpointed_reference \
  serve::api::engine::gemma_full_input_serving_tests::gemma_full_input_four_slot_geometry_holds_two_generations_within_quota \
  serve::api::engine::gemma_full_input_serving_tests::gemma_slot_route_warmup_does_not_change_outputs_or_retained_slots; do
  HF2Q_GEMMA4_GGUF="$GGUF" cargo test --locked --release --bin hf2q "$t" -- --exact --ignored --nocapture --test-threads=1
done
HF2Q_GEMMA_FULL_INPUT_TEST_GGUF="$GGUF" cargo test --locked --release --bin hf2q \
  serve::api::engine::gemma_full_input_tests::gemma_full_input_real_prefill_preserves_sample_and_exact_retry \
  -- --exact --ignored --nocapture --test-threads=1
cargo test --locked --release --bin hf2q \
  inference::models::gemma4::kv_cache::anchor_budget::actual_cpu::gemma_anchor_cpu_copy_reservation_and_exact_retry_layout \
  -- --exact --ignored --nocapture --test-threads=1
```

Expected completion markers:
- `GEMMA_FULL_INPUT_ENGINE_COMPLETE`, `GEMMA_FULL_INPUT_QUOTA_COMPLETE` and
  `GEMMA_SLOT_WARMUP_COMPLETE`.
- The existing `GEMMA_FULL_INPUT_REAL_COMPLETE` and
  `GEMMA_ANCHOR_CPU_COMPLETE`.
- `GEMMA_FULL_INPUT_MONOLITHIC_DIAGNOSTIC`, recorded but not a gate.

A/B (Q5 supervisor, `gemma4` case, ABAB with a thermal gap, at least 3 per
arm). Arm A is the default, arm B is the kill switch:

```sh
export HF2Q_Q5_HF2Q_BIN=/abs/release/hf2q HF2Q_Q5_MODEL=/abs/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf \
  HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/gemma4-client-approval.json # + CARGO/TOOLCHAIN/SANDBOX_* as in README
python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --negative-control-receipt "$NC_RECEIPT" --output "$Q5_ROOT/q5-gemma4-ckpt-on-$(date -u +%Y%m%dT%H%M%SZ)"
HF2Q_Q5_GEMMA_FULL_INPUT_CHECKPOINT=0 python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --negative-control-receipt "$NC_RECEIPT" --output "$Q5_ROOT/q5-gemma4-ckpt-off-$(date -u +%Y%m%dT%H%M%SZ)"
```

Peer comparison: interleave the peer-arm case used for
`E/peer-arms-v1/gemma-blk-{2,3,6}-p-*` (workload
`adr063-q5-peer-gemma4-n4-coding-ctx262144`; its `serve_argv` is in those
receipts) between the hf2q arms, in the same ABAB order. Compare these
fields:
- `client/summary.json` `elapsed_seconds` and per-turn
  `usage.prompt_tokens_details.cached_tokens`;
- `server.stderr` lines reading `Gemma full-input checkpoint worker decision
  active=true` and `checkpoint captured`.

Gate the A/B on `server.stderr` and `client/summary.json`:
- exactly one `Gemma full-input checkpoint worker decision` with
  `active=true` (arm A) or `active=false` (arm B);
- zero `Gemma CPU anchor reservation refused`, `Gemma anchor capture skipped`
  and `cancellation checkpoint restore failed`;
- every arm-A tool turn's `cached_tokens` equals the previous turn's
  `prompt_tokens`;
- all agents complete, and Q5 receipts are valid for both arms;
- record `q1_decision_policy.py transcripts` A vs B (expected
  `DivergedTranscripts`; classify, do not gate).

Add one stock arm (origin/main or v0.1.28 build) if the closure claim is "vs
stock hf2q". Match the generation configuration before any peer verdict: the
peer runs `--reasoning off`.

Quality (Class B): the 24-instance paired campaign, ON vs `=0`, with the
instance-label tooling now on main (`scripts/four_agent/test_quality_campaign.py`
and `q1_decision_policy.py`). `paired` must not return `Loss`, `BelowFloor`
or `Inconclusive`; otherwise record an explicit operator waiver or revert the
default.

### Joint warm prefill (Stage 1) gates

Focused model-free tests (cohort planning: who joins, the 33-row threshold,
the 4,096-row cap, FIFO fallback, distinct slots, FIFO installed-lane
selection, the per-lane decode reserve with mixed `max_tokens` near the
context limit, switch parsing) plus the neighbours whose paths changed:

```sh
cargo test --locked --bin hf2q gemma_joint_warm_
cargo test --locked --bin hf2q gemma_installed_prefill_batch_
cargo test --locked --bin hf2q gemma_cross_slot_batches_
cargo test --locked --bin hf2q gemma4_bounded_prefill_tests
cargo test --locked --bin hf2q gemma_cold_batch_
cargo test --locked --bin hf2q gemma_full_input_
python3 -B -m unittest discover -s scripts/four_agent -p 'test_q5_calibration.py'
bash scripts/audit_hf2q_env_surface.sh | grep HF2Q_GEMMA_JOINT_WARM_PREFILL
```

Real-artifact gates, one process each, no other runtime loaded, run once with
defaults and once with `MLX_PRECOMPILED_METALLIB=0`:

```sh
for t in \
  serve::api::engine::gemma_warm_batch_tests::gemma_joint_warm_prefill_matches_per_slot_bitwise \
  serve::api::engine::gemma_warm_batch_tests::gemma_joint_warm_engine_cohort_matches_serial_tool_turns; do
  HF2Q_GEMMA4_GGUF="$GGUF" cargo test --locked --release --bin hf2q "$t" -- --exact --ignored --nocapture --test-threads=1
done
```

- `gemma_joint_warm_prefill_matches_per_slot_bitwise`: four slots with
  histories 465/714/943/1254 and suffixes 56/200/311/385 (crossing the
  512/1024 chunk splits and the 1,024-row ring wrap) restored from CPU
  checkpoints; one joint `PerSlotExact` forward vs each slot's per-slot warm
  prefill, for the 4-lane cohort (five repeats, determinism), a 2-lane
  `{0,3}` and a 3-lane `{1,2,3}` cohort. Equal: first token, final logits
  `to_bits`, every layer's K, TQ-HB V codes and V scales bytes over the valid
  rows (ring rows for sliding layers), and every layer's slot cursor. In the
  subset cases every slot outside the cohort must keep its bytes (global rows
  up to the longest prompt, every ring row) and cursors. A mixed-`max_tokens`
  case gives lane 0 a `max_tokens` reaching the last full tile of its slot:
  the old reserve (largest `max_tokens` with the longest lane) must be
  refused by the capacity check, the production reserve must be accepted and
  stay bitwise equal, and a per-slot prefill at that `max_tokens` must equal
  the 16-token reference. It then shows the planner refuses a 32-row lane
  and that lanes beside it stay exact. Markers:
  `GEMMA_JOINT_WARM_BITWISE_COMPLETE`, and
  `GEMMA_JOINT_WARM_THRESHOLD_DIAGNOSTIC` (recorded, not a gate; expected
  `joint_equals_per_slot=false`).
- `gemma_joint_warm_engine_cohort_matches_serial_tool_turns`: four tagged
  conversations through the real engine. Turn 1 (4 concurrent tool turns)
  must run as one joint transaction over 4 lanes; turn 2 (the last lane has
  a 32-token suffix) over 3 lanes, the fourth falling back per slot. Text,
  reasoning, completion tokens, finish reason, top-4 logprob bits and
  `cached_tokens` must equal the same turns sent one at a time. Marker:
  `GEMMA_JOINT_WARM_ENGINE_COMPLETE`.
- Re-run the Category 1 loop above and the full-input real-artifact loop:
  the installed long-prompt and stable batches now share
  `settle_gemma4_prefill_batch` and the slot-view global binding.

Hardware A/B (Q5 supervisor, `gemma4` case, ABAB with a thermal gap, at least
3 per arm; arm A default, arm B kill switch), with the peer arm interleaved as
above:

```sh
python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --negative-control-receipt "$NC_RECEIPT" --output "$Q5_ROOT/q5-gemma4-joint-warm-on-$(date -u +%Y%m%dT%H%M%SZ)"
HF2Q_Q5_GEMMA_JOINT_WARM_PREFILL=0 python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --negative-control-receipt "$NC_RECEIPT" --output "$Q5_ROOT/q5-gemma4-joint-warm-off-$(date -u +%Y%m%dT%H%M%SZ)"
```

Gate on `server.stderr`: one `Gemma joint warm prefill worker decision` with
`active=true` (A) or `requested=false active=false` (B); in A, turns whose
four requests arrive inside the window log `transaction="joint warm"
requests=4`, and B logs none; zero `joint warm prefill batch failed`,
`failed before publication; retrying the restored lanes per slot` and
`pre-submission check failed`. Compare `client/summary.json`
`elapsed_seconds`, per-turn `send_monotonic_ns` spread and `cached_tokens`.
Record `q1_decision_policy.py transcripts` A vs B. Stage 1 is bitwise per
request, so identical transcripts are expected; investigate any divergence
(it can also come from nondeterministic tool results on the client side).

### Tiled warm attention (Stage 2) gates

Model-free tests and the neighbours whose prefill paths changed:

```sh
cargo test --locked --bin hf2q gemma_live_attention
cargo test --locked --bin hf2q route_viability_tests
cargo test --locked --bin hf2q gemma_joint_warm_
cargo test --locked --bin hf2q gemma_installed_prefill_batch_
cargo test --locked --bin hf2q gemma4_bounded_prefill_tests
cargo test --locked --bin hf2q gemma_cold_prefill_handoff_
python3 -B -m unittest discover -s scripts/four_agent -p 'test_q5_calibration.py'
bash scripts/audit_hf2q_env_surface.sh | grep -E 'HF2Q_GEMMA_(WARM_TILED_ATTENTION|TILED_LIVE)'
```

`gemma_live_attention` covers:
- the switch: default on, `0`/`off`/`false` select vec-hybrid, and an
  invalid value warns and keeps the default;
- single-slot and `PerSlotExact` always resolve the same route under every
  combination of both switches;
- the 32-row warm threshold, and installed lanes that ignore the warm
  switch and tile every length;
- vec-hybrid chunking at 512/1024, and tiled chunk coverage (1,024 sliding,
  4,096 global);
- every chunk fits its ring stage;
- the graft-shifted global causal offset;
- vec-hybrid stage positions that are identical for a lane alone and inside
  a cohort.

Real-artifact gates: one process each, no other runtime loaded. Run the
loop once with defaults, then four more times with
`MLX_PRECOMPILED_METALLIB=0` (≥20 determinism repeats in total):

```sh
for t in \
  serve::api::engine::gemma_warm_batch_tests::gemma_joint_warm_prefill_matches_per_slot_bitwise \
  serve::api::engine::gemma_warm_batch_tests::gemma_joint_warm_prefill_matches_per_slot_bitwise_vec_hybrid \
  serve::api::engine::gemma_warm_batch_tests::gemma_joint_warm_engine_cohort_matches_serial_tool_turns \
  serve::api::engine::gemma_warm_tiled_tests::gemma_warm_tiled_attention_is_deterministic \
  serve::api::engine::gemma_warm_tiled_tests::gemma_warm_tiled_attention_decision_envelope \
  serve::api::engine::gemma_warm_tiled_tests::gemma_warm_tiled_attention_layer_localization; do
  HF2Q_GEMMA4_GGUF="$GGUF" cargo test --locked --release --bin hf2q "$t" -- --exact --ignored --nocapture --test-threads=1
done
```

- **(a) Joint == per-slot, per route.** This is the Stage 1 harness:
  histories 465/714/943/1254 and suffixes 56/200/311/385, cohorts
  `[0,1,2,3]` ×5, `[0,3]`, `[1,2,3]` ×2, `[3,2,1,0]` and `[2,3]`,
  isolation of non-cohort slots, and mixed `max_tokens`. It runs on the
  tiled route (`..._bitwise`) and on the kill-switch route
  (`..._vec_hybrid`). Markers: `GEMMA_JOINT_WARM_BITWISE_COMPLETE
  kernel=TiledF16` and `kernel=VecHybrid`.
- **(b) Determinism.** On the tiled route, each lane's per-slot warm
  prefill five times and the four-lane cohort five times, all bitwise equal
  (logits bits, KV bytes, cursors), and the cohort equal to per-slot.
  Marker: `GEMMA_WARM_TILED_DETERMINISM_COMPLETE`.
- **(c) Decision envelope (Q1 evidence).** Per lane, from identical
  histories, it records max |Δlogit| at the last row, top-1 agreement and
  the top-1/top-2 margins on both routes. It fails only when top-1 differs
  and the vec-hybrid margin is at least 0.5 logits. A 31-row suffix must be
  bitwise equal on both switches (it stays vec-hybrid). Markers:
  `GEMMA_WARM_TILED_ENVELOPE_LANE` (per lane) and `GEMMA_WARM_TILED_ENVELOPE`
  (summary). The pre-run expectation (max |Δlogit| near the 0.81
  cross-graph level, with "above about 1.5 means another semantic
  difference remains") was wrong; see the envelope disposition after the
  results table.
- **Localization (diagnostic, not a gate).** Slot 1 (history 714,
  suffix 200) runs on both routes with the test-only per-layer probe
  (`forward_prefill_batched::live_layer_probe`). The probe forces a
  per-layer sync and copies each layer's attention output (after the FWHT
  undo) and residual. Per layer it prints `GEMMA_WARM_TILED_LAYER` with the
  attention and residual max |Δ|, relative Δ, last-row Δ and worst row.
  The summary `GEMMA_WARM_TILED_LOCALIZATION` names the first layer whose
  attention relative Δ exceeds 5%. Layer 0 has identical inputs on both
  routes, so its attention Δ is the pure kernel difference. Use the worst
  row to separate cases: a worst row inside the chunk diagonal points at
  causal leakage, near the window edge at the SWA mask, and at a global
  layer at the d512 resume offset.
- Re-run the Category 1 loop and the full-input and cold-prefill
  real-artifact gates above: every Gemma live resume goes through the new
  function, and `gemma_fresh_and_reused_4096_8193_bounded_outputs_match`
  exercises installed lanes (including the 1-row tail).

Hardware A/B (Q5 supervisor, `gemma4` case, ABAB with a thermal gap, at
least 3 per arm; arm A default, arm B kill switch), with the peer arm
interleaved as above:

```sh
export HF2Q_Q5_HF2Q_BIN=/abs/release/hf2q HF2Q_Q5_MODEL=/abs/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf \
  HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/gemma4-client-approval.json # + CARGO/TOOLCHAIN/SANDBOX_* as in README
python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --negative-control-receipt "$NC_RECEIPT" --output "$Q5_ROOT/q5-gemma4-warm-tiled-on-$(date -u +%Y%m%dT%H%M%SZ)"
HF2Q_Q5_GEMMA_WARM_TILED_ATTENTION=0 python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --negative-control-receipt "$NC_RECEIPT" --output "$Q5_ROOT/q5-gemma4-warm-tiled-off-$(date -u +%Y%m%dT%H%M%SZ)"
```

Gate on `server.stderr`:
- one `Gemma warm live attention kernel` line, `kernel=TiledF16` (A) or
  `kernel=VecHybrid` (B);
- every warm `Batched prefill:` banner shows `live=tiled-f16 (vec-hybrid
  below 32 rows)` (A) or `live=vec-hybrid` (B);
- the `transaction="joint warm" requests=4` cohorts appear in both arms.

Compare the per-cohort `elapsed_ms`, `client/summary.json`
`elapsed_seconds` and `cached_tokens`. Record
`q1_decision_policy.py transcripts` A vs B; divergence is expected (Class
B), so classify it rather than gate on it.

Quality campaign (Class B): 24 paired instances. The subject is the
default; the reference is the same build with the kill switch, as
`stock-hf2q`. Do not export `HF2Q_Q5_GEMMA_WARM_TILED_ATTENTION`, because
it would reach both arms.

```sh
python3 -B scripts/four_agent/quality_campaign.py \
  --subject-case scripts/four_agent/q5_cases/gemma4.json \
  --subject-var CLIENT_APPROVAL_TEMPLATE=/abs/gemma4-client-approval.json \
  --reference-case scripts/four_agent/q5_cases/gemma4.json --reference-kind stock-hf2q \
  --reference-var CLIENT_APPROVAL_TEMPLATE=/abs/gemma4-client-approval.json \
  --reference-var GEMMA_WARM_TILED_ATTENTION=0 \
  --instances 24 --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/quality-gemma4-warm-tiled-$(date -u +%Y%m%dT%H%M%SZ)"   # --dry-run first
```

`paired` must not return `Loss`, `BelowFloor` or `Inconclusive`. Otherwise
record an operator waiver or apply the revert rule above.

Measured results. Gates ran on the merge commit `1aa0b9c1`
(`E/gemma-stage2-gates-v3`); the vec-hybrid bitwise gate, envelope and
localization ran on `b0f96d86` (`E/gemma-stage2-gates-v2`). The v1 run
(`a2bc2007`) failed the tiled bitwise gate at `[1,2,3]` slot 3, from the
mask race fixed in `b0f96d86`.

| Gate | Result | Evidence |
|---|---|---|
| (a) tiled joint == per-slot (`[0123]`×5, `[03]`, `[123]`×2, `[3210]`, `[23]`, mixed `max_tokens`, isolation) | PASS, defaults and `MLX_PRECOMPILED_METALLIB=0` | v3 `a-default.log`, `a-nometallib.log`: `GEMMA_JOINT_WARM_BITWISE_COMPLETE kernel=TiledF16` |
| (a) vec-hybrid joint == per-slot (same cohorts) | PASS | v2 `a-vec.log`: `GEMMA_JOINT_WARM_BITWISE_COMPLETE kernel=VecHybrid` |
| Engine cohort == serial tool turns | PASS | v3 `a-engine.log`: `GEMMA_JOINT_WARM_ENGINE_COMPLETE` |
| (b) determinism, defaults + 4× `MLX_PRECOMPILED_METALLIB=0` | PASS, 5 runs × 5 repeats = 25 per-slot and joint repeats, all bitwise equal, joint == per-slot | v3 `b-default.log`, `b-nometallib-{1..4}.log`: `GEMMA_WARM_TILED_DETERMINISM_COMPLETE` |
| Category 1 ×4 (skip=0) | PASS | v3 `cat1-*.log` |
| (c) envelope: max \|Δlogit\| per lane, top-1 agree, min vec margin | 1.15 / 4.07 / 3.64 / 2.81; 2/4 agree; both flips are near ties (vec margins 0.167 and 0.392 < 0.5); 31-row suffix bitwise equal | v2 `c-envelope.log`: `GEMMA_WARM_TILED_ENVELOPE` |
| Localization (slot 1, history 714, suffix 200) | Layer-0 attention rel Δ 3.6e-4 (max abs 0.0044) on identical inputs. Residual Δ jumps 40× at layer 1 (0.0116 → 0.466) while attention Δ is only 0.0138. Attention rel Δ first exceeds 5% at layer 3 (0.051). Final logit Δ 4.07. | v2 `d-localize.log`: `GEMMA_WARM_TILED_LOCALIZATION`, `GEMMA_WARM_TILED_LAYER` |
| Joint cohort ms per turn t1…t6, A (tiled) vs B (vec-hybrid) | A ≈ 123 / 350 / 411 / 542 / 311 / 705 (≈2.44 s); B ≈ 147 / 452 / 545 / 849 / 476 / 1,130 (≈3.60 s). The saving is about 1.16 s of cohort time, against the projected 1.6–1.7 s. | `E/gemma-stage2-ab-v1/g-*-{T,V}-*/server.stderr` `transaction="joint warm"` `elapsed_ms` (A `g-1-T` t1 = 240 ms first-use outlier) |
| Makespan A / B / peer (median of 4) | 13.66 / 14.81 / 12.39 s (A −7.8% vs B) | `E/gemma-stage2-ab-v1` `client/summary.json` / receipts |
| Quality campaign (24 paired, subject default vs `stock-hf2q` with `GEMMA_WARM_TILED_ATTENTION=0`) | `Pass`: subject 21/24, reference 20/24, discordant 2 subject-only vs 3 reference-only, one-sided p = 0.8125 | `E/quality-gemma4-stage2-vs-vec-20261031T042314Z/manifest.json` |

**Envelope disposition (the 4.07 max |Δlogit| against the 1.5 rule).** The
section's "above about 1.5 means another semantic difference remains" rule
fires: 4.07, 3.64 and 2.81 exceed it. The localization run does not show a
semantic attention defect:
- Layer 0 sees identical inputs on both routes, and its attention differs
  by 3.6e-4 relative, about one F16 half-ulp.
- The independent review confirmed that both tiled kernels accumulate in
  F32. The only differences from vec-hybrid are F16 storage of the
  dequantized V and of O, and reduction order.
- The 40× residual jump at layer 1, while attention differs by only 0.0138,
  puts the amplification outside attention. That is consistent with MoE
  top-k routing flips on near-tied router logits `[INFERENCE]`.

So the 1.5 rule was a wrong pre-run expectation (it extrapolated the 0.81
monolithic-vs-resume figure), not a gate. The disposition is Q1 Class B,
route-sensitive, accepted on task quality: the paired campaign returned
`Pass` (21/24 vs 20/24), the A/B is 7.8% faster, top-1 flips occur only at
near ties, and the 31-row control is bitwise equal.

Open follow-ups:
- localize lane 3 (history 1254 ≥ window, so it exercises the window cut
  and d512 over a longer history);
- a tiled-vs-cold envelope;
- hardware coverage for graft > 0 and F16 V storage, which no gate
  exercises.

### Cold admission allocation gates

Model-free tests and the neighbours whose prefill paths changed:

```sh
cargo test --locked --bin hf2q gemma_cold_prefill_handoff_
cargo test --locked --bin hf2q route_viability_tests
cargo test --locked --bin hf2q gemma_cold_batch_
cargo test --locked --bin hf2q gemma_joint_warm_
cargo test --locked --bin hf2q gemma_full_input_
cargo test --locked --bin hf2q gemma4_bounded_prefill_tests
```

- `gemma_cold_prefill_handoff_only_for_serial_fifo_cold_prefill`: which
  routes build the handoff.
- `gemma_cold_prefill_handoff_four_agent_turn_zero_sizes`: the eliminated
  bytes for the turn-0 geometry.

Real-artifact gate, one process, no other runtime loaded. Rerun the
existing cold-batch, joint-warm and full-input real-artifact gates too: the
multi-sequence cold prefill and the stable/full-input admissions share the
changed function.

```sh
HF2Q_GEMMA4_GGUF="$GGUF" HF2Q_KV_LCP_RESUME_CAPACITY=8g \
  cargo test --locked --release --bin hf2q \
  serve::api::engine::gemma_cold_prefill_tests::gemma_cold_prefill_handoff_skip_matches_legacy_bitwise \
  -- --exact --ignored --nocapture --test-threads=1
```

The gate runs two arms, each a fresh model load.
- **Schedule:** seven cold admissions at `max_tokens` 8192 over
  conversations of 458, 1,300 (ring wrap) and 700 tokens. Slots are reused
  by other conversations.
- **Arms:** production against the legacy route (`FORCE_LEGACY_DENSE_HANDOFF`,
  test-only).
- **Equal per admission:** first token, logits `to_bits`, every layer's K,
  TQ-HB V codes and V scales over the valid rows, every sliding ring row
  (what the stable-boundary checkpoint copies), checkpoint size, and
  cursors.
- **Lean arm only:** every admission of a conversation must equal its
  first, whatever slot and predecessor it got, and no LCP snapshot may
  remain on the model.
- **Legacy arm only:** it must have built its snapshots.
- **Markers:** `GEMMA_COLD_PREFILL_HANDOFF_BITWISE_COMPLETE`, and
  `GEMMA_COLD_PREFILL_HANDOFF_TIMING` with per-admission prefill ms per arm
  (diagnostic, not a gate).

Hardware A/B: there is no kill switch, because the change removes dead
work. Compare against the parent build (`d3bf681b`) on the Q5 `gemma4`
case, ABAB, at least 3 runs per arm, with the peer interleaved. Gate on
`server.stderr`: the gap between consecutive turn-0 `Gemma4 slot admission
planned` lines should fall from about 500 ms to about the prefill time
plus about 20 ms. Compare turn-0 completion times, `elapsed_seconds`, and
`q1_decision_policy.py transcripts`; identical transcripts are expected.

## Template fidelity fix (2026-10-30) and the peer comparison

Up to this fix, hf2q rendered Gemma 4 tool-loop history differently from the
official template and from the peer, in three ways:

- `reasoning_content` echoed `thought\n` back for every empty thought block;
- `assistant` was renamed to `model` before the template ran, which defeated
  the template's same-turn continuation;
- tool results were rendered as `response:unknown`.

As a result, in `E/gemma-alloc-ab-v1`, hf2q's `prompt_tokens` drifted from
the peer's from turn 1 on, by +9 tokens for each extra assistant turn (−2 for
the tool name on turn 1, then +43 by turn 6). Because the prompts differed,
the generations differed from turn 4 on: hf2q kept writing the empty thought
block and the peer did not.

The fix strips the label in the reasoning splitter, keeps original roles for
every template except hf2q's fallback, and passes `tool_call_id` and `name`
through to the template.

A model-free replay checked this against the captured requests. Each request
was rendered with the reviewed template, which is byte-identical to the
vendored `src/serve/api/test_fixtures/gemma4-official-chat-template.jinja`,
and counted with the reviewed tokenizer:

- with the old message shaping, the counts match hf2q's reported
  `prompt_tokens` on all 112 `g-*-N` turns;
- with the fixed shaping, they match the peer's on all 112 `g-*-P` turns.

Gemma Q3/Q4 peer comparisons and `rendered-inputs-reviewed` captures made
before this fix measure different prompts after turn 0. Rerun or recapture
them after the fix.

## Remaining gaps

- Gates without a recorded result above remain unexecuted. Default
  promotion depends on all of them passing on hardware.
- Joint warm prefill Stage 1: the vec-hybrid bitwise and engine cohort gates
  pass (results table above). Stage 2 (tiled F16 warm attention, Q1 Class
  B): bitwise, determinism, Category 1, A/B and the quality campaign pass.
  The envelope exceeds the pre-run 1.5 expectation and is dispositioned as
  Class B above. The follow-ups listed there remain open.
- The cold prefill setup builds its sliding and global masks
  (`build_sdpa_mask_bf16` / `build_block_diagonal_sdpa_mask_bf16`) without
  declaring their writes. The F16 cast and the `blk` classifiers that read
  them therefore get no tracked barrier, the same hazard fixed for the live
  route. It is unchanged here (out of scope). In practice the classifiers
  run several dispatches later, and the cold gates pass, but nothing orders
  them `[INFERENCE]`.
- Turn 0: the cold admission allocation fix is unexecuted on hardware.
  After it, the remaining turn-0 cost is four serial cold boundary
  prefills plus the per-slot cue. The per-step decode gap (about 20 vs
  18 ms) is not addressed.
- Requests are still tagged when a default-mode worker stays inactive (the
  geometry does not fit). The worker ignores those tags in every path, so
  behaviour equals `=0`; the handler cannot see the worker's geometry decision.
- The per-slot anchor generation bound (two) comes from reading the code. The
  degrade branches are driven only through `capture_or_skip`, not by a
  request-level fault injection; the per-worker quota is the enforcing
  backstop.
- `handlers_gemma_matched_tokens_tests.rs` still classifies an empty env as
  the historical study's OFF arm; that study is source-hash bound and
  unaffected by the new default.
- Mask fill ordering (branch `fix/gemma-cold-prefill-mask-fence`, found by
  the Stage 2 review): the mlx-native mask builders allocate and fill their
  mask inside the call, so no batched-prefill session declared the fill to
  the barrier tracker. The F16 mask cast and the `blk` classifier of the
  cold setup session (single- and multi-sequence), and the `blk` classifier
  of the main-branch tiled multi-sequence warm sliding path, could run
  concurrently with the fill. In a multi-sequence cold batch a classifier
  that reads a partly filled block-diagonal mask can mark an off-diagonal
  tile as attended, leaking attention across sequences `[INFERENCE]`.
  `src/serve/gemma_cold_masks.rs` now declares every fill
  (`encode_tracked_mask`). Audit of the other alloc-and-fill helpers of
  mlx-native 0.15.15 (`quantized_matmul`, `quantized_matmul_simd`,
  `dispatch_quantized_matmul_simd_bf16{,_expert}`, `quantized_matmul_id`,
  the `GraphSession` wrappers): no caller in hf2q, so no Gemma decode,
  Qwen 3.5 or DeepSeek-V4 site exists. Its other alloc-and-return helpers
  (`compute_g_beta_gpu`, `silu_mul_gpu`, `materialize_f16_shadow`) commit
  their own command buffer. Stage 2's live attention module fences its own
  mask fill with an explicit barrier. On `7a9459db`
  (`E/gemma-cold-mask-gates-v1`) the capture tests
  (`cargo test --locked --bin hf2q gemma_cold_masks`), the cold-prefill
  handoff bitwise gate and the cold-batch, full-input and bounded suites
  passed. The multi-sequence cold determinism/isolation gate could not run:
  see the next entry.
- Multi-sequence cold capacity check: since `b9514c99` the batched prefill
  plans full-layer capacity from the longest sequence (`max_i end_i +
  max_tokens`), but the global-layer bound still compared the concatenated
  stream (`start_pos + seq_len`) with it. A cold cohort whose total length
  exceeded `roundup32(longest + max_tokens)` was refused, although each
  sequence is scattered into its own slot region and, since the dense
  handoff was dropped for this route, nothing writes the concatenated
  stream into one cache. `iter_g_a_bf16_determinism_isolation_gate` (eight
  prompts, 304 rows, longest 70, `max_tokens` 24, cap 96) failed at L5 on
  main before and after the handoff change. The bound now applies only to
  streams written into one cache (single sequence, or a forced legacy
  handoff), and the multi-sequence scatter checks `graft + len_i` against
  each full-layer slot capacity (the copy kernels index `pos % capacity`,
  so an overflow would wrap rather than fault). Production four-agent turn
  0 (4 × 458 rows, `max_tokens` 8192) never hit it; a cold batch of long
  prompts with small `max_tokens` would have `[INFERENCE]`. Unexecuted:
  `cargo test --locked --bin hf2q cold_prefill_handoff_tests` and
  `HF2Q_BYTE_EQUIV_E2E=1 HF2Q_BYTE_EQUIV_E2E_GGUF="$GGUF" cargo test
  --locked --release --bin hf2q iter_g_a_bf16_determinism_isolation_gate --
  --nocapture --test-threads=1` (20 bitwise repeats plus per-sequence
  no-leakage), which also exercises the mask-fill fence on the
  block-diagonal path.
- Multi-sequence cold ring writes: a cold cohort lane may be up to 4,096
  rows (`gemma4_cross_batch_eligible`, `engine.rs` ~6309; the cohort total
  is capped at 4,096 by `gemma_cold_batch::plan`, which batches
  `[2048, 2048]`), while the per-slot sliding ring holds `sliding_window`
  (1,024) rows (`layer_type_to_alloc_params_per_slot`). The scatter wrote
  every row of each sequence with `pos % capacity`, so rows 1,024 positions
  apart raced for the same ring row and decode could continue from a ring
  holding older positions. `gemma_multi_seq_kv_write` now applies the
  single-sequence rule per lane (rings take the last `capacity` rows; full
  layers take every row after the graft and must fit). Unexecuted:
  `cargo test --locked --bin hf2q gemma_cold_multi_seq_` (plan tests and a
  model-free copy-kernel test) and the ignored real-artifact gate
  `HF2Q_GEMMA4_GGUF="$GGUF" cargo test --locked --release --bin hf2q
  serve::api::engine::gemma_cold_prefill_tests::gemma_cold_multi_seq_long_lane_ring_deterministic
  -- --exact --ignored --nocapture --test-threads=1` (lanes 2,100 / 1,300 /
  458 rows, ten bitwise repeats, untouched neighbour slot, layer-0 ring
  K / V codes equal to per-slot cold prefill).
