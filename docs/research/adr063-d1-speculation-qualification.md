# ADR-063 D1: qualifying Qwen warm-turn speculation at four slots

Status: source qualification and test gates written, 2026-10-30. The root has
not built or run anything yet. No speed or quality result is claimed; the A/B
plan in §5 decides the four-slot policy.

Scope (ADR-063 D1, issue #364): prove the existing MTP/history speculation at
four slots, disable measured losers, and keep the graft fallback. No new
speculation feature is added. The only production change is a
behaviour-preserving extraction of two existing gate predicates into
`qwen35_speculation.rs`, so the graft fallback can be tested where it is
decided.

## 1. What runs today (source facts)

Line numbers refer to this branch.

**Switch.** `HF2Q_QWEN_SPECULATION=off|auto` (`0`/`1` aliases). Unset means
`auto`; an invalid value warns and fails closed to `off`
(`qwen35_speculation.rs:42`). `scripts/serve_qwen38_opencode.sh` forwards
`QWEN38_SPECULATION=off|auto` to it. `scripts/serve_qwen36_opencode.sh` has no
alias, so set `HF2Q_QWEN_SPECULATION` directly. The SlotAware decode path uses
no other switch. `HF2Q_SPEC_DECODE`, `HF2Q_SPEC_DECODE_K` and the DFlash and
EAGLE-3 variables belong to the `hf2q generate` CLI.

**Admission** (`engine.rs:20009`, `classify_slot_request`):
- `off` → `Disabled`.
- Request semantics outside `is_qwen_server_speculation_exact_eligible` (`engine_qwen35.rs:1320`) → `UnsupportedSemantics`.
- A full prompt hit *without* a reusable speculative boundary → `PromptCacheHit`. A warm append whose retained prefix carries valid speculative metadata stays `Eligible`, as D1 requires.
- A bound KV graft → `RuntimeUnavailable`.

Any decision other than `Eligible` drops the cached speculative boundary
(`qwen35_cached_spec_for_decision`, `engine.rs:18118`).

**Prefill.** A bounded prefill chunk runs the MTP prompt catch-up only when
`prefill_mtp_catchup_allowed` holds: policy `auto`, exact semantics, a
text-only chunk, MTP weights plus an MTP arena, and no graft
(`engine_qwen35.rs:4170`). The history proposer is created only for `auto`,
exact semantics and a prompt that is not a full cache hit (`:5080`).

**Decode tick order** (`decode_tick`, `:5228`):
1. Queued verified output streams first, one token per tick, with no forward.
2. History lookup (6–12-token match, up to 3 drafts, `:5427`).
3. On a lookup miss, or with no lookup: fixed-depth K3 MTP (`:5745`), when MTP state exists, at least 4 output tokens remain and the MTP cost gate admits it. Otherwise the coherent ordinary decision (`:6125`): target forward plus MTP catch-up.
4. With neither proposer: the ordinary target forward.

**Acceptance.** `plan_qwen35_verified_block` (`:4731`) compares each draft
with the target's canonical decision for that row
(`Qwen35SpecSemanticState::select_and_observe`, `:4802`), which applies the
request's grammar, repetition window, forced thinking close and tool-marker
trigger. The block stops at the first mismatch:
- The target's own token replaces a rejected draft.
- A terminal decision (EOS or grammar end) ends the block. It is never streamed or cached.
- The cache keeps exactly `valid_input_tokens` rows.

**Cost gate** (`cost_controller.rs`). A proposer starts only after the
generation has an ordinary target-time baseline. It is disabled for the rest of
the generation after two consecutive four-round windows that lose to the
equivalent ordinary time. Acceptance rate is not used.

**Grafts** (ADR-059). The admission and prefill gates above keep MTP off for
the whole request. Only history lookup can still run. It is target-only, so it
verifies against the grafted target at graft-shifted positions.
[INFERENCE from `joint_decode_ready`, `joint_decode.rs:55`] A grafted slot's
cursor includes the graft rows, so the cursor-equals-ledger check never holds
and grafted slots never join a joint step.

## 2. Tests added

All tests run at the root. They are not run here.

| Test | Kind | Proves |
|---|---|---|
| `full_partial_and_zero_acceptance_then_eight_ordinary_decisions_match_greedy` | model-free | After full (3 drafts and a bonus), partial and zero acceptance, with 1 or 3 ordinary warm-up decisions, under greedy and the 1.05 repetition penalty: the stream, finish reason, repetition window, step counter and cache contents equal ordinary decode, including the next 8 ordinary target decisions. A vacuity guard asserts that the penalty changes toy decisions. |
| `chained_rounds_of_mixed_acceptance_match_greedy` | model-free | Five back-to-back rounds (full, zero, partial, full, partial), then 8 ordinary decisions, equal ordinary decode. |
| `eos_inside_or_after_a_verified_block_stops_exactly_like_greedy` | model-free | EOS as an accepted draft mid-run, as the first draft, as the bonus after full acceptance, as the target's replacement for a rejected draft, and on the first row. Each stops with `stop`, never streams EOS or later drafts, keeps no row past the EOS-predicting row, and counts EOS as one target decision. A draft EOS that the target rejects does not terminate the request, and 8 later ordinary decisions still match. |
| `verified_block_at_the_length_boundary_finishes_like_greedy` | model-free | With exactly 4 tokens left, full, partial and zero rounds all finish with `length` and the same ledger as greedy. |
| `cancellation_mid_verified_drain_publishes_nothing_and_anchor_rewinds_target_and_mtp` | Metal device, no weights | While verified output is queued (GPU state ahead of the stream), no speculative boundary can be published at any cursor. The cancellation recovery primitive `restore_slot_anchor` rewinds both the target and the MTP cursor from mid-K3 state to the verified prompt boundary and leaves a peer slot's speculative cursors untouched. A drained state does publish its boundary. |
| `sampled_and_response_shaping_requests_never_speculate` | model-free | Temperature, top-k, top-p, min-p, seed, logprobs, top_logprobs, logit bias, frequency/presence penalties, stop strings and parallel tool calls each block admission, MTP prompt catch-up, the history proposer and serial MTP. Greedy decoding, with or without the repetition penalty, enables all three. |
| `qwen35_speculation::tests::bound_graft_keeps_eligible_requests_off_mtp_at_admission_and_prefill` | pure | Graft fallback: an eligible request, cold or resuming a reusable boundary, becomes `RuntimeUnavailable` and never runs MTP prompt catch-up. Every other fallback keeps its own reason. |
| `qwen38_slot_aware_speculation_auto_matches_off` | ignored, `HF2Q_QWEN38_GGUF` | Production engine, `SlotAware{4}`, `auto` vs `off` on identical prompts: solo turns (serial K3), four concurrent warm follow-ups (joint rows mixed with K3), and a streaming request cancelled after 12 deltas, then re-served. Each `auto` response must equal `off` or first diverge at a Q1 near-tie (`off` top-1/top-2 margin ≤ 0.25 nat). `auto` must show accepted and rejected MTP drafts, and some request must stop on EOS. |

The model-free harness replaces only the GPU forward, with a toy target whose
decision hashes the whole cached context. Seeding, canonical row selection,
block planning, the streamed and ordinary commits and the queue drain are the
production functions. The harness mirrors only the K3/history tick control
flow and the cache truncation to `valid_input_tokens`. Existing gates still
cover the rest:
- `mtp_k3_full_accept_*`, `mtp_k3_partial_reject_*` and `mtp_k3_terminal_draft_*` (planner);
- `mtp_transaction_rollback_restores_target_and_drafter_cursors`;
- `qwen38_spec_reject_rewinds_only_target_slot_ping_pong`;
- `qwen_cached_spec_boundary_cannot_bypass_request_policy`;
- `qwen38_real_four_position_normal_forward_parity` (M4 vs 4×M1 decisions).

Root commands:

```sh
cargo test --release --lib speculation_qualification_tests -- --test-threads=1
cargo test --release --lib bound_graft_keeps_eligible_requests_off_mtp_at_admission_and_prefill
cargo test --release --lib mtp_k3_
cargo test --release --lib qwen_cached_spec_boundary_cannot_bypass_request_policy
HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1 HF2Q_QWEN38_GGUF=/abs/Qwen3.8-27B-vanilla.gguf \
  cargo test --release --lib qwen38_slot_aware_speculation_auto_matches_off \
  -- --ignored --exact --nocapture --test-threads=1
```

**What "identical to greedy" means on hardware.** The model-free gates prove
the transaction is exact: given the same target logits, speculative and
ordinary decode commit the same decisions and leave the same cache. On the real
model, K3 verifies on the four-row target route, which Q1 treats as a declared
Class B alternate of the one-row route (largest observed benign shift
0.093 nat on Qwen3.8, `adr063-q1-numeric-policy.md` §3.2). `auto` also picks
rounds from wall-clock cost and, at N4, from joint-turn timing. `auto`
transcripts can therefore differ from `off`, and from another `auto` run, at
near-ties. The real-artifact gate accepts exactly those divergences and fails
any divergence outside the envelope. [INFERENCE] This may explain the
non-repeatable Qwen3.6 stock transcripts recorded in S3 (two runs differing in
two groups); the A/B below checks it with off-vs-off transcripts.

## 3. Sampling (temperature > 0)

Speculation is never active for sampled requests on the server.
`is_qwen_server_speculation_exact_eligible` rejects temperature, top-k, top-p,
min-p, seed, logprobs, top_logprobs, logit bias, frequency/presence penalties,
stop strings and parallel tool calls. Every speculative entry point checks it:
admission, MTP prompt catch-up, the history proposer, and serial MTP through
`is_serial_mtp_exact_eligible`.

The server acceptance is not a distributional proof:
- It compares the draft with the target's selected token.
- It never computes the proposal distribution q, which K3 drafts are argmax of the MTP head.
- It has no rejection-sampling (p/q) step and no residual resampling.
- No distributional test exists.

A sample-and-match variant would also have to show that discarded verifier
rows never consume or leak seeded draws, and that logprob records stay exact.
None of this is proven, so the gate stays and
`sampled_and_response_shaping_requests_never_speculate` pins it. The
`hf2q generate --speculative` CLI has a separate Metropolis–Hastings sampler
(`spec_decode.rs:147`, `SpecSampler`). It is not an API path and is outside D1.
Enabling sampled speculation on the server needs a rejection-sampling
transaction plus a distributional acceptance test (for example, chi-square
over many seeded draws per row), not equal-seed anecdotes.

## 4. Joint decode (#466) × MTP at N4: when K3 runs

**Joint step formation.** Each scheduler decode turn forms a joint step when
at least two decoding slots are joint-ready (`joint_decode_ready`) and their
handles are unique. A slot is joint-ready when:
- no verified output is queued;
- its cursor equals prompt + ledger − 1, with an equal MTP cursor if it holds MTP state;
- its client is open;
- no DeltaNet capture is active.

The step takes at most `HF2Q_QWEN_JOINT_DECODE_MAX_ROWS` lanes (default 4).

**Joint rows never speculate.** In a joint row:
- A slot with MTP state takes the coherent ordinary path: shared target forward, then its own MTP catch-up, so a later solo turn can resume K3.
- A slot without MTP state takes the plain ordinary path.
- History lookup is skipped; its ledger catches up on the next solo round.

**When K3 and history run:**
- in a turn with fewer than two joint-ready slots (one agent decoding while the others prefill, wait on tools or are idle);
- for overflow lanes beyond the joint width.

Slots with queued output drain without a forward.

**Per-family policy today:**

| Family | Joint default | Four-slot behaviour with `auto` |
|---|---|---|
| Qwen3.8 (dense) | on | Overlapping decode runs joint M1 rows. Each MTP-holding lane still pays a per-token MTP catch-up with no drafting. K3/history run only in solo turns. |
| Qwen3.6 (MoE) | off | Every turn is serial. Each slot runs its own history/K3 rounds in turn, so the four slots' verify transactions serialize. |

[INFERENCE] On Qwen3.8 at N4, `auto` can lose even if K3 wins at N1: the MTP
catch-up in joint turns is pure overhead. Its cost gate never fires there,
because it only scores K3 rounds. Peer evidence also shows MTP gains shrinking
with batch size (scout: 0.86× at 8 rows in one peer, 1.10× at 4 slots in
another).

## 5. Root A/B plan

Same binary, artifact, launcher and harness. Only the speculation switch
changes. Run one family at a time with no other runtime loaded.

**Derived cases** (outside Git):

```sh
E=/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023
D=$E/d1-speculation-ab-v1; mkdir -p "$D"
python3 - "$D" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
for family, switch in (("qwen38", "QWEN38_SPECULATION"), ("qwen36", "HF2Q_QWEN_SPECULATION")):
    base = json.loads(pathlib.Path(f"scripts/four_agent/q5_cases/{family}.json").read_text())
    for policy in ("auto", "off"):
        case = json.loads(json.dumps(base))
        case["serve"]["env"].update({switch: policy, "RUST_LOG": "info"})
        case["notes"].append(f"ADR-063 D1 derived arm: {switch}={policy}; RUST_LOG=info for per-request decode lines.")
        (out / f"{family}-spec-{policy}.json").write_text(json.dumps(case, indent=2) + "\n")
PY
```

**Schedule.** ABBA × 2, so 4 runs per arm and 8 per family, with A = `auto`
(the shipped default) and B = `off`: `auto off off auto auto off off auto`.
Runs 1–2, 3–4, 5–6 and 7–8 are the four paired blocks. The Qwen3.8 runs use
the family default, joint decode on, matching shipped behaviour:

```sh
export HF2Q_Q5_HF2Q_BIN=/abs/release/hf2q HF2Q_Q5_CARGO=... HF2Q_Q5_TOOLCHAIN=... \
  HF2Q_Q5_SANDBOX_PROOF=... HF2Q_Q5_SANDBOX_PROOF_SHA256=...
run() {  # family index arm
  python3 scripts/four_agent/q5_calibration.py "$D/$1-spec-$3.json" \
    --negative-control-receipt "$NC_RECEIPT" \
    --output "$D/$1-$2-$3-$(date -u +%Y%m%dT%H%M%SZ)" | tee "$D/$1-$2-$3.stdout"
}
i=1; for arm in auto off off auto auto off off auto; do
  HF2Q_Q5_MODEL=/abs/Qwen3.8-27B-vanilla.gguf HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/qwen38-client-approval.json \
    run qwen38 $i $arm; i=$((i+1)); done
i=1; for arm in auto off off auto auto off off auto; do
  HF2Q_Q5_MODEL=/abs/Qwen3.6-35B-A3B-vanilla-Q4_K_M.gguf HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/qwen36-client-approval.json \
    run qwen36 $i $arm; i=$((i+1)); done
```

**Check that each arm ran as intended:**
- `server.stderr` prints the policy line (Qwen3.8 launcher).
- The `Qwen joint decode configuration` line shows `selected=true` for Qwen3.8 and `false` for Qwen3.6.
- `Qwen35 decode complete` lines give per-request `tokens_per_second`.

**Metrics.**
- Primary: four-agent makespan, `client/summary.json`
  `own_output_four_task_makespan_seconds`, with each run's Q5 verdict.
- Effect: `q3_stats.paired_effect([(auto, off), …])` over the four blocks
  (log-ratio mean with a df = 3 t-interval; lower is better).
- Secondary: the per-run median of per-request decode tokens/s, compared with
  `paired_effect(..., higher_is_better=True)`.
- Transcripts: `q1_decision_policy.py transcripts` over the four `off` runs
  (expect `ExactTranscripts`: `off` is repeat-deterministic) and over all
  eight runs (`DivergedTranscripts` is allowed; §2 explains why).

**Qwen3.6.** Its coding protocol fails most of the time because of the weights
(`adr063-qwen36-task-failure-diagnosis.md`, 5/24 passes). When agents abort at
different points, makespan is not comparable. Use makespan only for blocks
where both runs have the same per-agent completion. Otherwise decide on the
decode-throughput ratio, and leave task quality to the paired campaign.

**K3 engagement.** Run one extra, uncounted `auto` run per family with
`HF2Q_MTP_PHASE_PROFILE=1` added to `serve.env`. The `[MTP_PHASE]` lines count
K3 rounds and partial rejections. The flag is development-only, so it stays
out of counted runs.

**Optional diagnostic arms** (Qwen3.8, 2+2 runs): `HF2Q_QWEN_JOINT_DECODE=0`
with `auto` vs `off`. This separates K3's serial value at N4 from the joint
catch-up overhead in §4.

**Decision rule** (declared before data; ratio r = auto/off, lower is better):
1. **Validity.** Every Qwen3.8 run must end `CalibrationPass`, and the `off`
   transcripts must be `ExactTranscripts`. Otherwise localize the failure
   first; it is not a speed verdict.
2. **Loss.** If the lower bound of the 95% interval of r is above 1.0, `auto`
   is measurably slower: disable it at N4 for that family.
3. **Win.** If the upper bound is below 1.0, keep `auto`.
4. **Inconclusive.** With r ≥ 1.025 (the F2 neutral band), extend the family
   by two more ABBA blocks (8 runs per arm) and re-apply 2 and 3. With
   r < 1.025, or if the extension is still inconclusive, keep the shipped
   default and record `Inconclusive`. D1 adds no profitable-gap extension in
   that case.
5. **Quality guard.** A run where `auto` fails the protocol and its paired
   `off` run passes blocks a keep decision until the paired quality campaign
   rules on it.

**Disabling at N4 when 2 applies.** Make the unset default family- and
width-aware in `QwenSpeculationPolicy::resolve_server_policy`. Unset resolves
to `off` when the worker's `max_slots ≥ 4` and the family is on the measured
loss list. An explicit `HF2Q_QWEN_SPECULATION=auto` (or `QWEN38_SPECULATION`)
still opts in. This mirrors `default_joint_decode(moe)`. Update
`docs/operator-env-vars.md`, the Qwen3.8 launcher comment and the CHANGELOG
with the receipt path. If only the optional Qwen3.8 diagnostic shows that K3
wins serially but the joint catch-up loses, the narrower alternative is to
drop a lane's MTP state when it joins a joint step. That changes MTP lifecycle
and needs its own exactness gates.

## 6. What closes D1

- The root runs §2's gates green; the real-artifact gate reports its near-tie notes.
- The §5 A/B for both families, with receipts under `$E/d1-speculation-ab-v1`, a verdict per §5's rule and any resulting default change.
- Independent review of the tests, the A/B receipts and the verdict.
