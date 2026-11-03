# ADR-063 Q1 speculative successors: design for Q1-QWEN36-SPEC (#653) and Q1-GEMMA-SPEC (#665)

Status: design for review. Nothing in this note has been executed. No model was
loaded to write it. Counts marked [INFERENCE] come from source manifests and
retained stores, not from a run. Each one has a pre-launch check that confirms or
refutes it before any comparison is made.

`$E` = `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`, `$C` = `$E/q1-closure-v1`.
New evidence goes to `/Users/robert/.local/state/hf2q/v0130/q1-qwen36-spec/` and `.../q1-gemma-spec/`.

Binding text:
- ADR-063 §Q1-QWEN36-SPEC and §Q1-GEMMA-SPEC.
- The numeric policy `docs/research/adr063-q1-numeric-policy.md`: Class A §3.1, Class B §3.2, τ = 0.25 nat.
- The decision tool `scripts/four_agent/q1_decision_policy.py`.

## Part 1: Q1-QWEN36-SPEC (#653)

### 1.1 Adapters (existing and shared with Qwen3.8)

`matrix.rs:34-39` routes both Qwen families to the same two adapters. For Qwen3.6 only the gap text says "actual execution pending".

| Trajectory | Adapter | Actual entry point | Qwen3.6 accepted by source |
|---|---|---|---|
| T5GenuineSpeculation | `QwenCoherentSpeculation` | `serve::api::engine_qwen35::q1_speculation::runner::q1_qwen_actual_t5_speculation_phase`, env `HF2Q_Q1_QWEN_T5_PACKET`, modes `Preparation` / `OrdinaryBaseline` / `ForcedProposal` (`q1_speculation.rs:9-14`) | yes: `runner.rs:176-184` maps `Qwen36 → qwen35moe`; `validate_complete` (`runner.rs:446-476`) is family-free |
| T5 first fork (side effects) | same module, `first_fork::q1_qwen_t5_first_fork_actual` | same env | **no**: `runner.rs:127-136` pins family Qwen38, a token digest and an input hash |
| T5 offline comparison | `adr063_invariance::t5_artifact_compare::q1_t5_artifact_only_compare` | `HF2Q_Q1_T5_ARTIFACT_COMPARE_PACKET` | **no**: `t5_artifact_compare.rs:147-165` requires Qwen38 and uses `selected_header_fixture(true)` |
| T6CancelRestore | `QwenOwningWorkerRecovery` | `serve::api::engine::q1_qwen_cancel::continuation::q1_qwen_actual_recovery_continuation_trace`, env `HF2Q_Q1_QWEN_CONTINUATION_PACKET`, modes `ColdBaseline` / `ColdChunkMatched64` / `CancelledThenRecovered` (`continuation.rs:17-22`) | yes: `continuation.rs:221-225`, `242-249` |

Requirements the hosted controls cannot prove, so the run must:
- T5 requires a loaded MTP head (`runner.rs:224-227`, `model.model.mtp.is_some()`). The Qwen3.6 artifact's header reports `mtp_num_hidden_layers: 1` (`$C/qwen36-prep-b2/prepared.json`). The D1 speculation A/B changed Qwen3.6 behaviour between `auto` and `off` (`adr063-d1-speculation-qualification.md:355-358`). [INFERENCE] The MTP tensors are therefore present. If they are absent, the runner refuses before forward work. T5 is then recorded as `SourceCapabilityRefusal` with that error, not Missing.
- T5 forced mode needs `mtp_cost.may_speculate()` after one ordinary warm-up (`trajectory.rs:128-132`; `cost_controller.rs:39-41`: enabled and `ordinary_samples > 0`). There is no override. A refusal fails the case.

### 1.2 Build and ordering against ADR-065

The rule is that the binary comes from main at or after `ea6caf46` and before P4b (#527) or J1 (#512) merges. Both stories change the solo K3 path that T5 drives:
- J1 refactors `decode_tick_mtp_k3`, which hosts the Q1 hooks.
- P4b replaces the DeltaNet rollback.

| Build | Source | Use | Relation, recorded in every receipt |
|---|---|---|---|
| **b4** | main `d817dffa7371a2db85a43ed1d6502350906c33f5`; test binary `$C/build-d817dffa…/test-binary`, sha256 `8c0b6a054530beb18bec76d7d43b457095cf53b12011f77c39e8596dae26d823` | T6 block (§1.4) and the T0/T3 endpoint gate. Shareable with Q1-QWEN38-RERUN (#664) | `git merge-base --is-ancestor ea6caf46 d817dffa` is true (20 commits after it). #527 and #512 were OPEN on 2026-11-03 |
| **b5** | main after the D1 delta (§1.7) merges, while #512 and #527 are still unmerged. If either merges first, use a pinned topic commit `d817dffa + D1`, whose parent precedes both | T5 block (§1.5), including first fork | Record `gh issue view 512/527 --json state,closedAt`, plus `git merge-base --is-ancestor <J1/P4b merge> <b5>` = false for each merge that exists |

Build-log observations for b4 (`build-receipt.json` present):
- adr063_invariance: 104 passed / 0 failed / 17 ignored;
- q1_: 61 / 0 / 26;
- full non-ignored suite: 5743 / 0 / 136.

Each pair is compared within one build. Preparation is non-comparable by construction (`WorkloadPreparationNotComparable`), so it may come from a different build than its phases, as the Qwen3.8 precedent did. Every hardware run records source commit, binary sha256, artifact digest, native version (0.15.16) and case file, and passes Q5 CalibrationPass.

### 1.3 Workloads

- **T6/T3:** `Qwen36ToolRoundSuffix144`, `$C/qwen36-workload/workload.json` (sha256 `db413896…9880`). This is the same lineage as the Q1 b2 verdict. Its real-prompt status for Class B was accepted by the operator (D1, 2026-11-02).
- **T5:** a new four-prefix document `Qwen36ToolRoundTail128`, one lane per reviewed tool round in `$C/qwen36-workload/derivation.json`:
  - Each lane holds the **last 128 tokens of the reviewed prompt**: `window_start = prompt_tokens − 128`, with token files and sha256 values as listed there.
  - Preparation then appends 16 greedy tokens, which are the start of the assistant turn.
  - Reason [INFERENCE]: the `Suffix144[..128]` prefixes end 16 tokens before the generation prompt. Greedy continuation would then re-emit the end-of-turn token, and the runner fails preparation on early termination (`adr063-q1-qwen-forced-speculation-diagnostic.md`: "Early termination before that boundary fails preparation").
  - The windows come from the same reviewed prompts, so the D1 basis carries over. The derivation still gets its own review before preparation.

### 1.4 T6: stores, comparison of record, expected counts

**Cases (b4, one model resident at a time):**

| Case | Filter / mode | Stores written (`output/…/trace.json`) |
|---|---|---|
| `qwen36-t0-b4` | `qwen_actual` endpoint, T0ColdSerial | single store; supervisor T0-first gate (`q1_endpoint_supervisor.py:231-232`) |
| `qwen36-t3-b4` | `qwen_actual` endpoint, T3Chunk64 | single store |
| `qwen36-t3cm-b4` | continuation `ColdChunkMatched64` | `continuation/` (`T0ChunkMatched64Tail64To143`), `prefix64/` (`T0ChunkMatchedCheckpoint64`) |
| `qwen36-t6-b4` | continuation `CancelledThenRecovered` | `continuation/` (`T6Tail64To143`), `before/` (`T6Checkpoint64`), `recovered/` (`T6ActualEngineRecovered64`) |

**The store T6 is compared against: `qwen36-t3cm-b4`, the same-build, route-matched chunk-64 never-cancelled continuation.**

Why this store and not the endpoint T3:
- T6 builds its 0..64 prefix through the production engine (`advance_qwen35_prefill` via `run_loaded`, `q1_qwen_cancel.rs:178-214`).
- The endpoint T3 builds it with direct `forward_gpu` All (`qwen_runner.rs:138`).
- `ColdChunkMatched64` builds the prefix through the same engine call without cancellation (`run_loaded_prefix64`, `continuation.rs:380-389`) and then runs the identical tail (`continuation.rs:106-150`, call order enforced by `require_calls`, `continuation.rs:31-61`).
- So T6 vs T3cm differs only by cancel+restore, which is the Class A §3.1(1) pair "cancel then restore vs never-cancelled".
- This matches the Qwen3.8 evidence (numeric-policy §2.1 row 2: chunk-matched vs recovered, `ExactEqual` on 3,136 records).

Record extents come from source manifests:
- `qwen_continuation::expected` (`qwen_continuation.rs:28-49`) gives rows 64..143 logits plus active caches at 128 and 144.
- `checkpoint_expected` (`:75-87`) gives the active64 snapshot.
- The `qwen_cache::manifest` per-lane count at MTP length 0 is 140: 30 linear layers × 2 plus 10 full-attention layers × 4 roles × 2 KV heads.
- The retained `qwen36-t3-b2` store confirms 140: 2256 records = 576 logits + 3 boundaries × 560.
- `adr063-q1-qwen-recovery-continuation.md` lists Q36 tail 1440 and snapshot 560.

Pre-registered decisions (`q1_decision_policy.py exact`, `:96-113`, keys `:83`):

| # | Pair (left, right) | Expected | common | left_only | right_only |
|---|---|---|---:|---:|---:|
| P6-1 (verdict) | `t6/continuation` vs `t3cm/continuation` | ExactEqual | 1440 | 0 | 0 |
| P6-2 (verdict) | `t6/recovered` vs `t3cm/prefix64` | ExactEqual | 560 | 0 | 0 |
| P6-3 (verdict) | `t6/before` vs `t6/recovered` (also checked in process: `recovery_proof`, `continuation.rs:447-470`) | ExactEqual | 560 | 0 | 0 |
| P6-4 (diagnostic) | `t6/continuation` vs `t3-b4` (endpoint) | ExactOnCommon | 1440 | 0 | 816 (256 logit rows 0..63 + 560 active64) |
| P6-5 (diagnostic) | `t6/recovered` vs `t3-b4` | ExactOnCommon | 560 | 0 | 1696 |

The issue's "ExactOnCommon with restored extras only" maps onto this layout as follows. Over all stores, T6 has 2560 records and T3cm has 2000. Common is 2000 with 0 differing. The only extras are T6's 560 `before` records, the pre-cancel twin of `recovered`. Those extras are checked exact by P6-3.

Verdict rules:
- **Pass:** P6-1, P6-2 and P6-3 all as expected.
- **Defect:** any `DivergedExactInvariant` in P6-1 to P6-3. The case fails, as for the Qwen3.8 #561 cursor bug. Any other count is `Incomplete` and the case is rerun; it is not reinterpreted.
- **P6-4/P6-5:** a divergence there with P6-1 to P6-3 exact is attributed to the prefix route (engine prefill vs direct `forward_gpu`) and recorded as a finding. It is not a T6 failure.

Supervisor checks (`qwen_t6_continuation`, §3):
- `calls` equals the `require_calls` order;
- `captured_rows_per_lane` 80, `cancelled_unpublished_rows_per_lane` 64;
- `recovery` has 4 entries, with `recovered_cursor` 64, `scheduler_in_flight_after` 0 and `client_closed_after_actual_advance` true;
- `recovery_proof.disposition` ExactEqual.

### 1.5 T5: phases, record sets, classification

**Phases (b5, three model processes, then two for the first fork):**
1. `Preparation` on `Qwen36ToolRoundTail128`. It outputs `frozen-workload.json` and `preparation-only.json` (`runner.rs:404-441`), which are independently reviewed.
2. `OrdinaryBaseline`, then `ForcedProposal`, on the reviewed frozen file.
   - The forced arm injects `[g2, bad, original3]` (`q1_speculation.rs:270-307`).
   - It must yield matched 1, rejected 1, 2 valid inputs, carry row 1 and cursor 131 (`q1_speculation.rs:448-464`).
   - `bad` is the first non-EOS ID that is not canonical (`qwen_speculation.rs:111-116`).
3. First-fork baseline and forced arms (agent 0, cursor 131) on the same frozen file. This needs the D1 gate change (§1.7).

**Record sets** (`qwen_speculation.rs:24-97`):
- **Committed store**, per comparable phase:
  - 64 logit rows (rows 128..143 × 4 lanes);
  - complete target and MTP caches at 129, 131 and 144 (MTP length = position);
  - per lane-boundary: 140 target + 8 MTP = 148 [INFERENCE: the MTP layer is one more full-attention layout of the same geometry];
  - total 64 + 12 × 148 = **1840 records**, payload **872,421,120 B** (control `qwen_speculation.rs:123-124`).
- **Raw-verifier store** (forced arm only): **20 records**, 16,023,680 B. Paired total: 1,760,865,920 B.
- Pre-launch check: the root confirms 1840 against the source manifest. The supervisor `expected_stores` is generated from `committed(cfg,262144)` with the Q36 header, not typed by hand.

**Classification, frozen before the runs:**

| Axis | Class | Record set | Rule |
|---|---|---|---|
| T5-A1 integer decisions | A | Per lane: 17 canonical decisions (16 + carry) and ordinary input positions, in both phases; the 4 forced transitions | Equal to the frozen lineage (`t5_artifact_compare.rs:68-98`). Raw IDs: original 3 drafts, forced = `[g2,bad,orig3]`, canonical = `g2,g3` (`:101-145`) |
| T5-A2 pre-fork state | A | **596 records**: Logits row 128 (4) plus every cache record at step 129 (4 × 148). Both phases ran identical calls up to this point (prefill 128, one ordinary MTP warm-up) | `exact --scope t5-prefork.json` → ExactEqual, 596/0/0/0 |
| T5-A3 side effects (§3.1(5)) | A | First fork, agent 0: before vs after draft+discard, 148 records (140 target + 8 MTP); ordinary output pre vs post MTP catch-up | ExactEqual. Same rule as Qwen3.8 (352+16 = 368 there) |
| T5-B committed rows | B | All **64** Logits rows of the two committed stores. Row 128 is expected exact; rows 129..143 follow the R4 verifier and rollback | `envelope` (`q1_decision_policy.py:165-202`), τ = `POLICY.route_decision_error_nats` = 0.25 nat (`:47`), unchanged. Verdict ExactEqual / WithinDecisionEnvelope / RouteSensitive; Incomplete → rerun |
| caches at 131/144 | B | Committed cache records after the fork | Rust `compare` element-wise with NumericPolicy::Unspecified: `ExplicitInconclusiveNumericPolicy`, recorded, not gated |

Any failure in T5-A1, T5-A2 or T5-A3 is a defect.

**Routing (§5.5(d)): Missing-with-reason.** The only routing witness is `gemma_routing_witness` (Gemma-only namespaces, `gemma_routing_witness.rs:58-70`). No Qwen3.6 MoE routing capture exists, as also recorded in `$C/qwen36-verdict.md` §Routing.

**Verifier-route localization scope.** A `RouteSensitive` T5-B is reported with:
- rows, flips, the D distribution and rows over τ;
- split into verifier rows (129, 130) and post-rollback rows (131..143).

It is **not called localized**. No Qwen3.6 verifier-route localizer exists:
- `q1_target_layer_capture` refuses any graph other than the Qwen3.8 layer count (`q1_target_layer_capture.rs:83`);
- its runner pins the Qwen3.8 two-arm byte total (`runner.rs:205-206`);
- the first-fork row-129 comparison is an endpoint fork, not a stage localizer.

The v0.1.30 decision is to add no localizer: #653 is Medium successor scope and does not gate Q4. The receipt records `verifier_route_localization: Missing("no Qwen3.6 verifier-route localizer")`.

### 1.6 Order

Each step needs its own composition, operating review and Q5 CalibrationPass.

1. b4: endpoint prep and continuation prep, then `qwen36-t0-b4` → `qwen36-t3-b4` → `qwen36-t3cm-b4` → `qwen36-t6-b4`.
2. Offline: P6-1 to P6-5. Informational: `exact qwen36-t0-b4 qwen36-t0-b2`, a cross-build check.
3. Review the `Qwen36ToolRoundTail128` derivation.
4. b5: T5 Preparation → independent review of the frozen file → OrdinaryBaseline → ForcedProposal → first-fork baseline → first-fork forced.
5. Offline: `t5_artifact_compare` (Q36), `exact --scope`, `envelope`, then independent review of the receipts.

GPU time [INFERENCE, from the Qwen3.8 precedents of about 1.5 min per process]: about 15 minutes.

### 1.7 D1 source delta (cfg(test) and tooling only; no serving change)

1. **First-fork family gate** (`runner.rs:127-136`).
   - Replace the Qwen38 literal pins with a retained-preparation binding. The packet names the reviewed `preparation-only.json` through a FileBinding. The runner then requires:
     - its `frozen_file_sha256 == input.sha256`;
     - `full144_digest_computed_after_generation == identity.token_vectors_sha256`;
     - `comparable_acquisition == false`.
   - The existing Qwen3.8 literals become a control asserting the rule accepts the retained Q38 file.
   - Add a Q36 case to `first_fork_scope_and_disk_bound_do_not_relabel_four_lanes_or_sixteen_rows` (`first_fork.rs:160-176`, header `selected_header_fixture(false)`).
   - `target_layers` stays Qwen3.8-only.
2. **`t5_artifact_compare`** (`:146-165`).
   - Take the family from `packet.identity.family` (Qwen38 | Qwen36).
   - Use `selected_header_fixture(family == Qwen38)`, the same recipe with that family, and a family-matched frozen check.
   - Add a Q36 control.
3. **`q1_decision_policy.py exact --scope FILE`.**
   - The file is a closed JSON list of coordinate keys. Every key must exist in both stores, and only those keys are compared.
   - The output adds `scope_sha256` and `scoped_records`.
   - Tests: a missing key refuses; an out-of-scope difference is ignored; an in-scope difference gives `DivergedExactInvariant`.
   - The scope file for T5-A2 is generated from `committed(cfg,262144)` by the predicate Logits ∧ row = 128, or ¬Logits ∧ logical_step = 129. It is committed before the runs.
4. The evidence PR, not D1, removes "Qwen36 actual execution pending" from `matrix.rs:36,39` when the receipts merge.

## Part 2: Q1-GEMMA-SPEC (#665)

### 2.1 Gemma's speculative path

hf2q Gemma serving has no speculative decode. Its only speculative path is n-gram (prompt-lookup) verification:
- `dispatch_ngram_generate` (`ngram_orchestrator.rs:139-148`), reachable through `hf2q generate` with `HF2Q_SPEC_NGRAM=1` (`spec_decode_cli.rs:796-852`).
- It runs on the global `MlxModelWeights` route:
  - initial `forward_prefill_batched(prompt, max_new, 0)` (`:186-188`);
  - for each round, verification through an **Option C full re-prefill from position 0** of `output ++ drafts` (`:280-289`, `:303-305`), with the final-layer slab giving per-position argmax;
  - production `accept_prefix_argmax` and `rollback_kv`;
  - an empty proposal falls back to `forward_decode`, which is the same call as the baseline (`:238-259`).
- The source itself says this path is not byte-identical to the baseline (`:167-181`).
- The `xlen` path is refused by the existing preflight (`gemma_ngram/runner.rs:24-25`).
- Gemma has no MTP head. The `-assistant` drafter is out of scope (ADR-034).

The current matrix maps Gemma T5 to `Missing` (`matrix.rs:64-66`). `gemma_actual` refuses T5 (`gemma_actual.rs:45-49`).

### 2.2 What exists and why it is not a T5 comparison

`gemma_ngram` (run v3, `$E/q1-gemma-ngram-transactions-actual-v3`, source `a1b22ffb`, dataset `842768…`) observes one real transaction per serial owner:
- The production proposer runs, then an injected fixture replaces the drafts: frozen 129, frozen 130 and a non-EOS mutation of 131 (`gemma_ngram.rs:84-90`).
- Observed: on all 4 lanes the initial GPU argmax equals canonical, accepted count = 2, and 1 draft is rejected.

It is not a numeric comparison for three reasons:
1. It writes a transaction journal, not a Q1 `Store` (`adr063-q1-gemma-ngram-transactions.md`).
2. After the rollback it continues on the model's own IDs (`runner.rs:123-133`). In v3 the lineage forked from the frozen input at position 134 (owners 1 and 3) and 139 (owner 2), so rows after the fork have different inputs.
3. It has no ordinary arm on the same route. The ordinary Gemma endpoint uses the slot-aware body (`gemma_actual.rs:71-86`), which is a different route from the global path.

### 2.3 Adapter design (D2, cfg(test) only)

Extend `adr063_invariance::gemma_ngram` rather than add a parallel module. Its hooks are already the call sites in `ngram_orchestrator.rs`.

**New ignored test and packet:**
- Test: `gemma_ngram::runner::q1_gemma_actual_t5_trace`. Env: `HF2Q_Q1_GEMMA_T5_PACKET`.
- Completion line: `Q1_GEMMA_T5_COMPLETE records=<n> whole_q1_qualified=false`.
- Packet: the existing closed profile (`runner.rs:16-20`: k3, ngram 1..3, context 262144, output 8192, 4 serial owners), plus:
  - `arm` ∈ {`OrdinaryGlobal`, `NgramTransaction`};
  - `lanes` [0,1,2,3];
  - `host_payload_limit`;
  - `expected_stores` (the multistore contract, §3).
- `routing_witness` must be null. Reason in §2.5.

**Workload:** the accepted greedy144 dataset bound by review receipt `9b097208…` (`runner.rs:43-46`), with token digest `84276820…5034`. This is the lineage of `gemma-t0-b3` and of the v3 probe.

**Both arms, per lane:** clear the global banks as `runner.rs:115-117` does, then:
- **`OrdinaryGlobal`:**
  1. `forward_prefill_batched(lane[..128], 8192, 0)`, the identical call to `ngram_orchestrator.rs:186`.
  2. Capture the initial-head logits (row 127) and the cache at cursor 128.
  3. Require the initial argmax to equal `frozen[128]`; otherwise refuse (no verdict).
  4. For positions 128..143: `forward_decode(frozen[p], p)`, capturing logit row p.
  5. Capture the cache at 144.
- **`NgramTransaction`:**
  1. `dispatch_ngram_generate(lane[..128], 8192, 3, 1, 3)` with the existing injection.
  2. Row 127 and the cache at 128 are captured at the existing `initial`/`before_verify` hooks, the same state as the ordinary arm.
  3. The verifier's per-position head rows 128..131 are captured at `argmax_row`/`argmax_matrix`.
  4. Let a be the accepted count. Rows 128..128+a have accepted inputs and go to the committed store. Rows above 128+a go to `raw-verifier`.
  5. After the real rollback (cursor `129 + a`, already enforced as `pos == 132 − rejected`, `gemma_ngram.rs:44-46`), the existing test-only stop ends the round.
  6. **Teacher-forced** continuation: for positions 129+a..143, `forward_decode(frozen[p], p)`. The fallback ID and each model argmax are recorded as decisions, never as inputs. This keeps §3.2's "identical frozen inputs" over the whole row set.
  7. Capture the cache at 144.
- In T5 mode the journal writes events only (proposal, verify_input, acceptance, commit, lineage, cache_ownership). Tensors go to the stores.

**Stores, coordinates and counts** [INFERENCE: cache count from v3, which held 4200 cache tensors per phase over 4 owners = 1050 per lane per boundary = 630 Hybrid + 420 dense companion; the adapter's manifest, validated against the live banks before capture as the journal already does, is authoritative]:

| Store | Records per arm | Payload |
|---|---|---|
| `prefork` | Logits row 127 ×4 + cache@128 ×4×1050 = **4204** | 321,839,104 B |
| `committed` | Logits rows 128..143 ×4 = 64 + cache@144 ×4×1050 = **4264** | 424,459,264 B |
| `raw-verifier` (n-gram arm only) | `SpecVerifierLogits` [4, 262144] per lane = **4** | 16,777,216 B |

How the counts are built:
- Logits coordinates are `gemma_endpoint.rs:67-68`: `layer=u32::MAX`, `logical_row_start=row`.
- Cache coordinates are `gemma_cache::at` (`gemma_cache.rs:6-10`) with stages CacheKey / CacheValuePacked / CacheValueNorm.
- The dense companion uses `segment = 2000 + head`, a new namespace that cannot collide with live (0..) or restored (1000..) heads.
- Bytes per cache row: 11 × 56,320 + 880 = 620,400 (matches the `gemma_ngram.rs:27` bound).
- Per arm: 746,298,368 B (ordinary) and 763,075,584 B (n-gram), under the existing 4 GiB journal ceiling.

**Matrix:** add `Adapter::GemmaNgramTransaction` for `(Gemma4, T5GenuineSpeculation)`:
- scope: "gemma_ngram global OptionC injected K3 transaction vs same-route ordinary arm";
- gap: "not the slot-aware serving body".
- `ordinary_preflight` returns `SeparateActualAdapterRequired` for it.

**Controls:**
- manifest counts and payload on the selected geometry;
- closed packet and arm;
- the initial-argmax refusal;
- the teacher-forcing input check;
- refusal of a rollback cursor other than 129+a;
- inactive hooks return the original values.

### 2.4 Classification rule, frozen before any run

**Class B** for n-gram committed rows versus ordinary rows:
- The verify is a 132-row re-prefill from position 0, a different prefill length and a multi-row verifier against M1 decode.
- §1 of the policy lists "prefill chunk size, M1 vs multi-row verifier" as Class B.
- The source disclaims byte identity (`ngram_orchestrator.rs:167-181`).

| Axis | Class | Record set | Rule |
|---|---|---|---|
| G5-A1 pre-fork | A | `prefork`, 4204 records: both arms made the identical call | `exact ord/prefork ngram/prefork` → ExactEqual 4204/0/0/0; anything else = defect |
| G5-A2 integer decisions | A | Per lane: initial argmax = `frozen[128]` (both arms); production proposer ran; injected = `[f129, f130, mut131]`; a ∈ {0,1,2}; rejected = 3 − a ≥ 1; rollback cursor 129 + a; inputs 129+a..143 = frozen | Checked in process and by the supervisor; a is reported, and a = 0 is valid evidence |
| **G5-B committed rows** | **B** | **64 Logits rows** = rows 128..143 × lanes 0–3 of `committed` (verifier rows 128..128+a, continuation 129+a..143) | `envelope ord/committed ngram/committed`; **τ = 0.25 nat** (`POLICY.route_decision_error_nats`, unchanged); D as in §3.2; ExactEqual / WithinDecisionEnvelope / RouteSensitive; Incomplete → rerun |
| cache@144 | B | 4200 records | Rust `compare` with Unspecified policy, element-wise only, `ExplicitInconclusiveNumericPolicy` |

Reporting rules:
- The envelope result is split into verifier rows and continuation rows, and each row's D and margin are listed.
- A `RouteSensitive` result is reported, not localized. No n-gram-route stage localizer exists.
- The result has no serving consequence, because Gemma serving does not speculate. A candidate that moves serving onto this route needs the §3.5 paired quality campaign.
- The ordinary reference is the global route, so no claim about the slot-aware serving body follows.
- Disclosure (as in #652): the greedy144 prefix has token 496 repeated in rows 29–127. Prefix inputs are identical in both arms, and only rows 128..143 are classified.

### 2.5 Routing

The witness is off for both T5 arms, and routing margins are **Missing-with-reason**:
- In the n-gram arm, committed rows come from two namespaces: `T5Verify` for rows 128..128+a, from a 132-row re-prefill that also re-routes rows 0..127, and `T5Continuation` (`gemma_routing_witness.rs:64-70`).
- `routing` refuses unpaired keys (`q1_decision_policy.py:261-262`).
- A key-mapping comparator is out of scope.
- [INFERENCE] A bitwise-equal `prefork` cache across all 30 layers implies identical prefill routing.

### 2.6 Build and order

- Gemma has no ADR-065 ordering constraint.
- Build **b5g** = main after D2 merges, on native 0.15.16. Share it with #652's hooks so that one composition and acquisition review covers both.
- Order: `gemma-t0-b5g` (T0-first gate, plus the informational cross-build `exact gemma-t0-b5g gemma-t0-b3`, expected ExactEqual 5616 if the ordinary path is unchanged) → `gemma-t5ord-b5g` → `gemma-t5ngram-b5g` → offline G5-A1, G5-B and the cache compare.
- GPU time [INFERENCE]: three processes at about 1–3 min each.

## Part 3: Shared supervisor change (with Q1-GEMMA-T2LOC #652)

One change to `scripts/four_agent/q1_endpoint_supervisor.py`, with one admission review. The #652 owner implements it on `feat/q1-gemma-t2loc`; they agreed its shape on 2026-11-03 and will add only their own check, `gemma_loc4`. The four checks below are added to the same registry by the #653/#665 implementers before the one admission review.
- New `kind: "multistore"` with case keys exactly {kind, out, packet, filter, packet_env, completion_marker, marker_stream, stores, checks}, plus two optional keys: `manifest` (bound, then copied to `out/manifest.json`) and `completion_line`.
- `stores` maps a name to a directory under `output/` ("." allowed). Each store's `trace.json` must be:
  - complete and unique;
  - identity = composition key;
  - coordinates = `expected_stores[name]`.
- Every record's agent must be in `lanes`.
- Σ bytes ≤ `host_payload_limit` and ≤ the policy's phase upper bound.
- `output/execution.json` must have `packet_sha256` = the packet's sha and `whole_q1_qualified: false`.
- The completion line must be exactly `<marker> records=<Σ> whole_q1_qualified=false`.
- `checks` are names from a closed in-file registry.
- Endpoint and localizer kinds and the T0-first gate are unchanged.
- **Extent source:** either the packet carries {lanes, host_payload_limit, expected_stores}, or the case key `manifest` = {path, sha256} binds a JSON with exactly those keys. Exactly one source is allowed. This is required because the Qwen runners' packets are `deny_unknown_fields` (`runner.rs:27-39`, `continuation.rs:151-164`) and the T6 block must run on b4 unchanged. Gemma T5 packets carry the fields themselves.

Completion lines:
- The Qwen runners print fixed lines rather than `records=<Σ>`: `Q1_QWEN_T5_PHASE_COMPLETE …` (`runner.rs:443`) and `Q1_QWEN_RECOVERY_CONTINUATION_COMPLETE …` (`continuation.rs:482`).
- Agreed with the #652 owner: an optional case key `completion_line` (a non-empty string that contains `completion_marker`). When present, that exact line must appear in `marker_stream`, and it replaces the Σ rule.
- Qwen T5 and T6 cases declare their exact line, for example `Q1_QWEN_RECOVERY_CONTINUATION_COMPLETE mode=CancelledThenRecovered lanes=4 rows_per_lane=80 first64_logits=false whole_q1=false`.
- The Gemma T5 adapter prints the Σ form.

Checks registered for #653/#665:

| Check | Case | Asserts |
|---|---|---|
| `qwen_t6_continuation` | t3cm, t6 | §1.4 receipt fields; stores {continuation, prefix64} or {continuation, before, recovered} by mode |
| `qwen_t5_phase` | T5 baseline/forced | `schema hf2q.q1.qwen-t5-phase.v1`, status `CompleteScopedAcquisition`, mode as packet, `committed_records` 1840, `raw_records` 20 (forced) or null, 0 or 4 transitions |
| `gemma_t5_ordinary` | t5ord | arm, 4 lanes, initial argmax = frozen[128], inputs 128..143 = frozen, stores {prefork, committed} |
| `gemma_t5_ngram` | t5ngram | arm, 4 transactions with injected origin, a and fallback recorded, cursor 129 + a, inputs 129+a..143 = frozen, stores {prefork, committed, raw-verifier} |

The Qwen T5 preparation and first-fork phases use the same kind, with checks `qwen_t5_phase` (preparation status `CompleteWorkloadPreparationNotComparable`) and `qwen_t5_first_fork` (status `CompleteFirstForkDiagnosticNotQ1Qualified`, agent 0 only).

## Open questions for review

1. Is T3cm the store of record for T6 (§1.4), with endpoint T3 as a diagnostic? The alternative is endpoint T3 as the verdict, which mixes the prefix route into the Class A claim.
2. Is `Qwen36ToolRoundTail128` acceptable as the real-prompt T5 input (§1.3)?
3. Is first-fork Class A (T5-A3) required for #653 closure? It is the issue body's "side effects exact", and it costs the D1 gate change and two processes.
4. Does the admission review accept the case-level `completion_line` for the fixed Qwen lines (§3)?
