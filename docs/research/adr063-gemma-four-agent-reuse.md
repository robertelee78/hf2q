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

## Joint warm prefill (item 2): not tractable bitwise; design only

`forward_prefill_batched_multi_seq_live` cannot be bitwise equal to per-slot
prefill with the current kernels:
- Multi-seq prefill runs BF16 flash attention (`use_fa_f16 =
  multi_seq_prefill.is_none()`, `forward_prefill_batched.rs:672`).
- Single-seq runs F16 flash attention, with tensor-mm globals for 64–8192
  rows.

ADR-040 §0.20 records that the batched-vs-serial gap flips near-tie argmaxes
(2/2 exact at N=2, about 4/8 at N=8). Its contract is determinism plus no
cross-sequence contamination, not cross-mode identity. So
`HF2Q_GEMMA_JOINT_WARM_PREFILL` was not added.

Design for when F16 multi-seq FA is deterministic (the ADR-040 §0.19 crack):
1. Add a `FullInput` class to `gemma4_cross_batch_candidate`, so contiguous
   tagged requests form one admission prefix. Rows are the conservative full
   prompt, under the 4096 cap.
2. Add `admit_gemma4_slots_full_input_batched`, modelled on
   `admit_gemma4_slots_stable_batched`. It plans each lane's preferred slot
   with 0 < cached < P and a suffix of at least 32, and falls back
   single-slot in FIFO order otherwise. It restores every anchor, then runs
   one live multi-seq forward over the suffixes and commits cursors. It then
   runs `gemma_full_input::capture` per slot (quota degrade as in the single
   path) and installs decode states. `fail_gemma4_stable_batch` is reused.
3. Gate: an engine test with four concurrent tool turns against the same
   turns admitted one at a time must be bitwise equal. Expected [INFERENCE]:
   −1.6 to −2.0 s on the four-agent makespan.

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
- **Cue replay (≈1.35 s per four-slot turn 0):** not a warmup issue. The lever
  is to replay the cue tokens of all admitted slots through the joint B-row
  decode step, which the comments say is byte-identical to single-row decode
  `[INFERENCE]`, instead of 28 serial single-slot forwards. Not implemented.

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

## Remaining gaps

- Every gate above is unexecuted, including the quality campaign. Default
  promotion depends on all of them passing on hardware.
- Joint warm prefill needs deterministic F16 multi-seq FA (ADR-040 §0.19)
  before a bitwise joint path can exist.
- Turn-0 cue replay (about 1.35 s) and the per-step decode gap (about 20 vs
  18 ms) are not addressed.
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
