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
  `max_tokens`: it sizes a dense handoff that must hold the whole
  concatenated stream, and a smaller reserve could newly refuse it.
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
quality campaign.

### Stage 2 (successor, not implemented): tiled F16 warm attention

The vec-hybrid kernel reads the whole history once per query; at 385 queries
over ~1,400 cached rows that is roughly 150 GB of sliding-layer cache traffic
per slot, matching the ~190 ms per-slot residual over cold-rate compute
[INFERENCE]. Stage 2 moves the one shared per-sequence attention function
(single-slot and joint alike) to the tiled F16 kernels: compact K to dense
F16, dequantize the active V range, then FA d256 with mask and blk for
sliding layers and `flash_attn_prefill_f16_d512_resume` for globals.
Joint stays bitwise equal to per-slot by construction, but warm single-slot
numerics change once against today, so it is **Q1 Class B**: it needs the
ADR-040 §0.19-class determinism gate (≥20 repeats, plus
`MLX_PRECOMPILED_METALLIB=0` stress) and the paired quality campaign.
Expected [INFERENCE]: a further −2.0 to −2.2 s; both stages together project
a 17.5–18.5 s makespan against the peer's 18.1 s. At that point the
`Tiled`/`PerSlotExact` split and `HF2Q_GEMMA_TILED_LIVE` collapse into the
one function.

## First turn (item 3)

The gap scout reported about 98 ms per decode step. That was misread:
`decode_time_secs` includes waiting behind the other slots' serial prefills.
In blk-1/4/5 and the full-input run, the last-admitted slot decodes 17 tokens
in 0.34 s, about 20 ms per step. The turn-0 cost sits in each slot's
`prefill_time` of about 0.55 s:

- **458-token boundary prefill:** 188 ms, and 264–392 ms for the first slot
  only.
- **The remaining ~0.36 s per slot:** the stable anchor snapshot (about
  19 ms, from the full-input run's per-capture delta) plus the 7-token
  generation cue. The cue is shorter than the 32-row batched minimum, so it
  is replayed as 7 single-slot decode forwards at about 48 ms each. [INFERENCE]
  This last split comes from the timings, not a profile.

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
- **Cue replay (implemented, joint cue replay below):** not a warmup issue.
  The cue tokens of all same-step stable lanes now go through the joint
  batched decode body, instead of 28 serial single-slot forwards.

### Turn-0 breakdown after Stage 1

Stage 1 A/B (`gemma-joint-ab-v1`, g-1-A): all four agents' turn 0 (465-token
requests, `stable_prompt_prefix_tokens=458`, so a 7-token cue) completes at
+2.663 s. The peer's completes at +0.98–1.11 s. The logs show this sequence:

1. The four requests are Stable class and reach
   `admit_gemma4_slots_stable_batched`. The stable batch plan refuses because
   no lane has a cached prefix, so `fallback_single` admits each request
   inline, one after another, in one quantum.
2. Each inline admission runs `prefill_seed`'s stable branch: a cold
   458-token boundary prefill (201/189/190/189 ms), the stable-boundary
   anchor capture (about 19 ms), then the cue. The cue is shorter than
   `GEMMA_SLOT_BATCHED_PREFILL_MIN_TOKENS` (32), so
   `forward_prefill_with_soft_tokens_slot_aware_resume` replays it as 7
   `forward_decode_slot_aware` calls.
3. The four slots then decode jointly, about 17 steps at about 20 ms.

2663 − 769 (prefills) − 76 (anchors) − 340 (joint decode) ≈ 1.48 s for the
28 cue forwards, about 53 ms each. `[INFERENCE]`: the logs carry no
per-forward timing.

### Joint cue replay (this change)

`engine_gemma_cue_batch.rs`, switch `HF2Q_GEMMA_JOINT_CUE` (default on).

- **Admission:** when a stable batch falls back (`fallback_single`) with two
  or more requests, `admit_cohort` admits each request with
  `Gemma4AdmitRoute::JointCueCandidate`. Slot choice, anchor restore,
  boundary prefill and the stable-boundary checkpoint capture stay exactly
  the inline code in arrival order. `prefill_seed` returns
  `Gemma4SeedOutcome::CueDeferred` instead of replaying the cue, but only
  when that cue would be replayed per token (0 < cue < 32), the request is
  text-only, and there are no soft tokens. Every other lane is seeded inline
  as before.
- **Replay:** `replay_cues_jointly` makes one `forward_decode_body_batched`
  call per cue position. This is the production joint decode body with the
  hybrid regime, one row per lane still inside its cue, at that lane's
  physical position. The batched head (`lm_head_batched`, or the fused head
  when `HF2Q_FUSE_LMHEAD=1`) covers each lane's last cue row. The first token
  is the first-max argmax, as production decode selects it. Lanes with
  different cue lengths drop out of the batch after their last token.
- **Publication:** each lane then follows the inline tail
  (`finish_gemma4_slot_seed`, extracted from `admit_gemma4_slot_routed`):
  cursor commit to the full prompt, stable-boundary checkpoint publication,
  cancellation recovery, scheduler advance, seed token, decode install.
  Lanes publish in arrival order.
- **Exactness contract:** per lane, first token, final logits bits, KV bytes
  and cursors equal the per-slot cue replay. The boundary checkpoint is
  captured before the cue on both routes, from the same boundary prefill, so
  it is unchanged by construction. The joint route depends on two things.
  First, decode-body row independence: per-slot attention, and row-wise
  projection and MoE kernels, are claimed byte-identical between m=1 and m=N
  (ADR-040 H-S1/H-S2 at kernel level). Second, `lm_head_batched` row
  parity. Production joint decode already relies on both, but they are
  proven only for tokens, not logits, so the new gate is the proof. A cohort
  replays per slot under any non-default GGML routing policy
  (`mlx_native::ggml_routing_policy_from_environment()`; for example
  `HF2Q_DECODE_MV_EXT=1`, which a Qwen3.8 load sets process-wide), because
  the opt-in multi-column routes are not row-identical.
- **Failure semantics:** if an admission fails, the earlier lanes and the
  remaining requests fail the way the inline loop fails them.
  - A worker-fatal GPU fault in the joint replay fails every lane.
  - A non-fatal joint error restores each lane from its boundary checkpoint
    (anchor restore plus cursor install) and replays its cue per slot, in
    arrival order.
  - A lane whose boundary capture was refused by the quota cannot be
    restored, so a non-fatal joint error fails it like an inline cue error
    would.

Expected saving `[INFERENCE]`: 28 single-slot forwards (≈1.48 s) become 7
four-row body steps at the observed joint decode cost (≈20 ms each) plus one
head, about 0.15 s. Saving ≈ 1.3 s, so turn 0 goes from 2.66 s to about
1.35 s against the peer's 1.0 s, and the four-agent makespan from 16.0 s to
about 14.7 s against the peer's 13.9 s. The remaining turn-0 gap is four
serial cold boundary prefills (≈0.77 s).

Successor (not implemented): joint cold stable-prefix prefill. It would run
the four cold boundary prefills as one `PerSlotExact`-style transaction
(shared projections/FFN/head, per-slot cold attention with each slot's own
chunking and ring staging). That needs a per-lane cold attention route
proven equal to the single-slot cold prefill (FA/tensor-mm at 458 rows), and
the anchor capture must stay per slot after the shared forward. Estimated
gain −0.2 to −0.3 s `[INFERENCE]`.

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

### Joint cue replay (turn 0) gates

Focused model-free tests (which cues are deferred, step schedule with mixed
cue lengths and head rows, routing-policy guard, switch parsing), plus the
neighbours whose paths changed (`prefill_seed`, admission tail, stable
fallback):

```sh
cargo test --locked --bin hf2q gemma_joint_cue_
cargo test --locked --bin hf2q gemma_joint_warm_
cargo test --locked --bin hf2q gemma_cross_slot_batches_
cargo test --locked --bin hf2q gemma4_bounded_prefill_tests
cargo test --locked --bin hf2q gemma_cold_batch_
cargo test --locked --bin hf2q gemma_full_input_
python3 -B -m unittest discover -s scripts/four_agent -p 'test_q5_calibration.py'
bash scripts/audit_hf2q_env_surface.sh | grep HF2Q_GEMMA_JOINT_CUE
```

Real-artifact gates, one process each, no other runtime loaded:

```sh
for t in \
  serve::api::engine::gemma_cue_batch_tests::gemma_joint_cue_replay_matches_per_slot_bitwise \
  serve::api::engine::gemma_cue_batch_tests::gemma_joint_cue_engine_turn_zero_matches_serial \
  serve::api::engine::gemma_warm_batch_tests::gemma_joint_warm_prefill_matches_per_slot_bitwise \
  serve::api::engine::gemma_warm_batch_tests::gemma_joint_warm_engine_cohort_matches_serial_tool_turns; do
  HF2Q_GEMMA4_GGUF="$GGUF" cargo test --locked --release --bin hf2q "$t" -- --exact --ignored --nocapture --test-threads=1
done
```

- `gemma_joint_cue_replay_matches_per_slot_bitwise`: four slots,
  cold-prefilled to boundaries 458/300/700/1100 (the last has a wrapped
  ring) and checkpointed. Each arm restores the boundary checkpoint with
  ledger 0 (the cold admission state). One arm is the joint replay, the
  other each slot's inline per-slot cue call. Cases: four 7-token cues
  (three repeats), mixed cues 7/3/12/31, a 2-lane `{0,3}` and a 3-lane
  `{1,2,3}` cohort. Equal: first token, final logits `to_bits`, every
  layer's K, TQ-HB V codes and V scales bytes over the valid rows, and
  cursors. Slots outside the cohort keep their bytes and cursors. Marker:
  `GEMMA_JOINT_CUE_BITWISE_COMPLETE`.
- `gemma_joint_cue_engine_turn_zero_matches_serial`: four concurrent cold
  stable-split requests (458-token boundary, 7-token cue) through the real
  engine must form one cohort over 4 lanes (`JOINT_CUE_*` counters).
  Turn 1 extends each boundary with a 64-token tool result and moves the
  boundary after it. It resumes from the turn-0 checkpoint
  (`cached_tokens=458`) and again forms one warm-lane cohort. Text,
  reasoning, completion tokens, finish reason, top-4 logprob bits and
  `cached_tokens` must equal the same turns sent one at a time, which also
  proves that the published checkpoints are equal. Marker:
  `GEMMA_JOINT_CUE_ENGINE_COMPLETE`.
- Re-run the Category 1 loop above: `prefill_seed` and the admission tail
  changed shape.

Hardware A/B (Q5 supervisor, `gemma4` case, ABAB with a thermal gap, at least
3 per arm; arm A default, arm B kill switch, the peer interleaved):

```sh
python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --negative-control-receipt "$NC_RECEIPT" --output "$Q5_ROOT/q5-gemma4-joint-cue-on-$(date -u +%Y%m%dT%H%M%SZ)"
HF2Q_Q5_GEMMA_JOINT_CUE=0 python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --negative-control-receipt "$NC_RECEIPT" --output "$Q5_ROOT/q5-gemma4-joint-cue-off-$(date -u +%Y%m%dT%H%M%SZ)"
```

Gate on `server.stderr`:
- One `Gemma joint cue replay worker decision` line, with `active=true` in A
  and `requested=false` in B.
- In A, turn 0 logs one `Gemma4 joint cue replay` with `requests=4`; B logs
  none.
- Zero `joint cue replay failed` lines in either arm.

Compare the agents' turn-0 completion times, which should fall from about
+2.66 s by about 1.3 s. Also compare `elapsed_seconds` and
`q1_decision_policy.py transcripts`, A vs B; the transcripts are expected to
be identical.

## Remaining gaps

- Every gate above is unexecuted, including the quality campaign. Default
  promotion depends on all of them passing on hardware.
- Joint warm prefill Stage 1 is unexecuted: the bitwise gates above decide
  the default. Stage 2 (tiled F16 warm attention, Q1 Class B) is design only.
- Joint cue replay is unexecuted: its bitwise gates above decide the
  default. Joint cold stable-prefix prefill (the four serial ≈190 ms
  boundary prefills) and the per-step decode gap (about 20 vs 18 ms) are not
  addressed.
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
