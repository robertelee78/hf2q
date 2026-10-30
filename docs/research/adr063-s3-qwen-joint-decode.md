# ADR-063 S3 — Qwen joint B-row decode (phase 1)

Status (2026-10-30):
- **Qwen3.8 (dense): default on.** Official-artifact gates passed and the four-agent ABBA A/B passed; see "Measured result".
- **Qwen3.6 (MoE): default off, opt-in with `HF2Q_QWEN_JOINT_DECODE=1`.** Its forward is bitwise exact, but the four-agent A/B is incomplete and both joint runs failed the coding protocol; see "Qwen3.6 verdict".
- `HF2Q_QWEN_JOINT_DECODE=0` restores the serial path for every family.
- Open: the Q1 Class B envelope on real prompts (Q-C2b) and the cohort-invariance gates below have not run yet.

## Problem

With four Qwen agents decoding, the SlotAware worker ran one target forward per slot per turn (`decode_batch_qwen35` → `Qwen35DecodeState::decode_tick`). Every slot re-read every weight. The earlier shared-forward spikes shared only the post-attention FFN and head, formed pairs at best, synchronised every layer on the host, and ran pairs on Qwen3.8's plain `Mv` route (no weight reuse). All were slower than serial (NegativeResult docs under `spike-s3-three-la-checked-group`).

## Phase 1 design

One scheduler turn advances every eligible decoding slot (2–4 rows by default) through **one layer-major target forward**, mirroring the DeepSeek-V4 joint decode (`docs/research/adr063-s3-ds4-joint-decode.md`).

**Scheduler** (`src/serve/api/engine/qwen35_joint_decode.rs`):

- **Eligibility.** A handle is eligible when all of these hold; other handles stay on the serial loop in scheduler order:
  - its installed slot matches and it holds decode work;
  - its client is open;
  - its next tick is a target forward (no queued verified speculative output, step bound not reached);
  - its cache cursor equals the committed ledger (prompt + generated − 1), with an equal MTP cursor when it holds MTP state.

  A turn with an active DeltaNet capture is serial.
- **Width.** Lanes are sorted by slot and capped at `HF2Q_QWEN_JOINT_DECODE_MAX_ROWS` (2..8, default 4). The padded `mul_mv_ext` mode caps at 4 whatever the setting, so its packed width is always 4. Fewer than two eligible lanes keep the whole turn serial, including MTP K3 and history speculation.
- **Per-lane execution.**
  - Each lane prepares exactly the inputs its serial tick would feed the target (`Qwen35DecodeState::prepare_joint_tick`).
  - The step runs under one supervisor lease.
  - Each lane commits through its serial tick's own commit code (`finish_joint_tick`).
  - Settlement (emit, progress, retained prefix, finish, cancel recovery, error reset) is `settle_qwen35_decode_lane`, extracted from `decode_batch_qwen35` and shared by both paths.
- **Configuration.**
  - The switch and width are read once at worker start. Unset means the per-family default: dense on, MoE off.
  - The worker logs `requested`, `explicit`, `moe`, `max_rows`, `selected`, `mode` and the `unavailable` reason.
  - Every turn re-checks the process routing policy. A turn whose policy no longer satisfies the worker's mode runs serially (logged once); a later turn that satisfies it again runs jointly.

**Speculation in phase 1:** joint rows run the ordinary target.

- Slots without MTP state take `decode_tick`'s ordinary path: thinking-budget forcing, the GPU argmax kernel for greedy rows, constrained sampling otherwise.
- Slots with MTP state take `decode_tick_mtp_warmup`'s coherent path.
  - After the shared target forward they run their own MTP catch-up over the same token (`process_target_batch`).
  - Target and MTP cursors therefore stay equal, and `spec_prefix_candidate` stays valid.
  - A later solo turn resumes K3 drafting.
  - The cost controllers observe each lane's wall time from prepare to its own finish (the shared forward plus earlier lanes' catch-ups) as the ordinary baseline.
- History proposals are suppressed. The lookup ledger catches up from the committed tokens on the slot's next solo round, exactly as after a serial lookup miss.

Consequence: joint rows commit M1 (one-token target) decisions where stock decode would commit K3 verify-row decisions whenever its cost gate admits K3. "Bitwise" below is therefore per target forward; per-request outputs can differ from stock.

**Graph** (`src/inference/models/qwen35/joint_decode.rs`, `joint_decode_core.rs`):

| Stage | Execution |
|---|---|
| Embedding gather, pre-norms, Q/K head norms, IMROPE (axis-major `[4×B]` positions), fused residual norm, final norm | packed over B rows (row-independent kernels) |
| DeltaNet qkv / z / α / β projections, gated-norm + out projection | packed, or one serial `m = 1` dispatch per row (see routes) |
| DeltaNet conv1d, L2 norms, q scale, gated delta net | per row, against `narrow_la_ping_pong_to_slot(slot)` with that slot's parity; `compute_g_beta` packed |
| Full attention Q+gate / K / V / O projections | packed or per row (see routes) |
| TQ/F32 KV write + SDPA | per row through the serial `apply_sdpa_with_kv_cache_decode_into` at that slot's cursor |
| Sigmoid gate | packed |
| Dense FFN (Qwen3.8) | packed through the serial `build_dense_ffn_layer_gpu_q_into` at `seq = B` |
| MoE FFN (Qwen3.6) | per row through the serial `build_moe_ffn_layer_gpu_q_into` |
| Output head | packed lm_head (or per row), GPU argmax per greedy row, full logits and post-norm hidden otherwise |
| Command buffers | the serial graph's layer grouping (`HF2Q_PARTIAL_CHAIN_N` / table default), one terminal wait per step |

The serial builders are unchanged; `joint_decode_core.rs` repeats their dispatch order around the per-row cores. Cursors and ping-pong parities advance at encode time exactly as in the serial graph.

MoE layers run per row for two reasons:
- the router and shared expert are float projections whose packed kernel differs from `m = 1`;
- the routed experts of four rows are mostly distinct, so packing them would trade exactness for little weight reuse.

Qwen3.6's gain in phase 1 therefore comes from the DeltaNet, attention and head projections only.

## Numeric contract (ADR-063 Q1)

The Q1 numeric policy is `docs/research/adr063-q1-numeric-policy.md` (branch `docs/adr063-q1-numeric-policy`, PR #465). `joint_decode_routes.rs` admits a joint step only before any cache write.

**Exact mode** (`HF2Q_DECODE_MV_EXT` not `1`; Qwen3.6, and the Qwen3.8 mvN test arm) is **Q1 Class A per target forward**.
- **What is bitwise.** Every row's logits, post-norm hidden, TQ KV codes and scales, conv and recurrent state, cursor and `pp_flipped` are bitwise those of its serial one-token forward. The row is also bitwise independent of its cohort and packed index (Q1 §3.1 item 2).
- **Packed products.** Admitted only for Q4_0, Q8_0, Q4_K, Q5_K and Q6_K storage, whose `m = 2..=8` routes (`Mv`, the `mvN` K-quant tiles, canonical Q5_K) keep one-row arithmetic.
- **Float storage** (F32/F16/BF16, e.g. small α/β projections) switches from a matvec at `m = 1` to tiled kernels at `m > 1`, so it runs one serial dispatch per row and is gathered.
- **Refused (artifact stays serial, reason logged):** any other block type, a float FFN arm, GLP steering, layer dumps, `HF2Q_LEGACY_PER_LAYER_CB=1` or `MLX_UNRETAINED_REFS=1`.
- **Untested pairing.** The admission checks only `HF2Q_DECODE_MV_EXT`. A non-default `HF2Q_Q5K_CANONICAL_Q4X4=0` pairing is not covered by a test.
- **Serving level.** This is not Class A against stock, because joint rows never speculate (see above).

**Padded `mul_mv_ext` mode** (Qwen3.8 production policy, `HF2Q_DECODE_MV_EXT=1`, dense artifacts only) is a **Q1 Class B route**: batched vs serial where not declared exact (policy §1, §3.2).
- **Pass rule.** Decision error D ≤ 0.25 nat on every logit row over identical frozen real-prompt inputs; otherwise `RouteSensitive`.
- **Arithmetic class.** The route is the same arithmetic class as stock Qwen3.8 serving: stock commits most tokens through width-4 `mul_mv_ext` R4 verify rows (ADR-044). Q1 §2.1 measured M1 vs R4 as `WithinDecisionEnvelope` (D max 0.093 nat). [INFERENCE: same kernel family and width; the recurrent scan length differs.]
- **Padding.** Every step packs exactly 4 rows (joint width is capped at 4) so the packed width never depends on how many tenants share the step. Padding rows replay row 0's token, carry row 0's core outputs, never touch a cache, and are discarded. The cohort-invariance gate checks the bytes.
- **No decision-parity guarantee.** In the A/B, serving transcripts differ from stock in 2 of 114 same-request groups (`q1_decision_policy.py transcripts`: `DivergedTranscripts`; agent-1 turn-2 and agent-3 turn-2). The four joint runs agree with each other and the four serial runs agree with each other. The forks are not yet classified (near-tie or not).
- MoE artifacts under `HF2Q_DECODE_MV_EXT=1` stay serial.

## Failure semantics

| When | What happens |
|---|---|
| Refusal before lanes are taken: switch off, policy revoked at the turn check, fewer than two eligible lanes, active DeltaNet capture | the turn is serial; nothing fails |
| A lane's prepare error | that lane fails (slot reset, request error), as its serial tick would |
| Refusal inside the joint transaction: admission plan, routing policy changed between the turn check and the forward, duplicate or out-of-range slot | refused before any cache write, but the step returns an error, so every prepared row fails and its slot is reset |
| A submitted failure | fails exactly the step's rows: each gets the serial error path (slot reset, request error); rows outside the step are untouched |
| Unhealthy supervisor or fatal command-buffer error | the worker fail-stops, as in serial decode |

In-transaction refusals are unreachable in normal serving: the same checks run on the same thread just before the forward, and an active capture is excluded at partition time.

## Gates (root-run)

**Host tests:** `qwen35_joint_decode::tests`, `joint_decode_routes::tests`.

**Metal tests** (packed vs per-row, bitwise): `qwen35::forward_gpu::joint_decode::tests`. They cover every admitted block type at `m = 2..4` through the DeltaNet projection helper and the gather path, and the packed dense FFN against per-row FFNs. A failing type must leave the admitted set.

**Official-artifact gates** (`qwen35::real_artifact_joint_decode_tests`, ignored). Four slots with ragged prompts in two identical caches: serial one-token forwards vs joint steps over full and partial cohorts, a lone row, a row that stops while others continue, and a slot reset and re-prefilled before rejoining. Refused steps must write nothing.

- `qwen36_joint_decode_matches_serial_decode_bitwise` (`HF2Q_QWEN36_GGUF`): logits, hidden, greedy tokens, cursors, parities and all KV/state bytes. **Passed** (`E/qwen-joint-exactness-v1`).
- `qwen38_joint_decode_mvn_matches_serial_decode_bitwise` (`HF2Q_QWEN38_GGUF`, `HF2Q_DECODE_MV_EXT` unset): the same bitwise contract on the exact mvN route. **Passed.**
- `qwen38_joint_decode_padded_mv_ext_keeps_decision_parity` (`HF2Q_QWEN38_GGUF`, `HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1`): teacher-forced top-1 decisions equal independent sequential forwards at every step, cursors and parities equal, then 16 further serial decisions per slot equal. **Passed.** Under Q1 it is diagnostic only (synthetic prompts, top-1).

**Cohort invariance** (`qwen35::real_artifact_joint_decode_invariance_tests`, ignored; not yet run). For each variant, one cache with four identically prefilled slots advances slot 2 for 4 steps inside a different cohort: alone, `{0,2}`, `{2,3}`, `{1,2,3}`, `{2,3,1}`, and `{0,1,3,2}` with slot 2 packed last. Slot 2's logits, hidden, cursor, parity, KV codes/scales and conv/recurrent bytes must be bitwise equal across all variants.

- `qwen36_joint_decode_rows_are_cohort_invariant_bitwise` (`HF2Q_QWEN36_GGUF`)
- `qwen38_joint_decode_mvn_rows_are_cohort_invariant_bitwise` (`HF2Q_QWEN38_GGUF`, `HF2Q_DECODE_MV_EXT` unset)
- `qwen38_joint_decode_padded_mv_ext_rows_are_cohort_invariant_bitwise` (`HF2Q_QWEN38_GGUF`, `HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1`)

**Q1 Class B envelope on real prompts** (Qwen3.8 padded mode; not yet run).

`qwen38_joint_decode_q1_envelope_trace` takes 2–4 real agent prompts (`HF2Q_QWEN_JOINT_Q1_PROMPTS`):
- **Input format.** A JSON array where each entry is either the exact rendered chat prompt as token IDs, or the rendered prompt text tokenized with the serving `tokenizer.json` at `HF2Q_QWEN35_TOKENIZER`. Use the four-agent turn-0 and turn-2 requests, including the two forked turns (agent-1 and agent-3 turn-2).
- **Run.** It prefills each prompt into two identical caches, then decodes `HF2Q_QWEN_JOINT_Q1_STEPS` (default 32) teacher-forced steps. Both caches consume the serial top-1 at every step.
- **Left store.** `$OUT/serial/trace.json` holds full-vocabulary logits from the serial one-token `forward_gpu_last_logits` (stock M1 target route).
- **Right store.** `$OUT/joint/trace.json` holds full-vocabulary logits from `forward_gpu_decode_joint` over all agents in the worker's mode.
- **Store format.** Both are `hf2q.adr063.invariance.v2` stores keyed by agent, absolute row, and step.

Root commands, the test from this branch and `q1_decision_policy.py` from any checkout containing PR #465:

```sh
OUT=$E/q1-qwen38-joint-envelope-$(date -u +%Y%m%dT%H%M%SZ)
HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1 \
HF2Q_QWEN38_GGUF=/abs/Qwen3.8-27B-vanilla.gguf \
HF2Q_QWEN_JOINT_Q1_PROMPTS=/abs/qwen38-agent-prompts.json \
HF2Q_QWEN35_TOKENIZER=/abs/qwen38-tokenizer.json \
HF2Q_QWEN_JOINT_Q1_OUT=$OUT \
  cargo test --release --lib qwen38_joint_decode_q1_envelope_trace -- --ignored --exact --nocapture --test-threads=1
python3 -B scripts/four_agent/q1_decision_policy.py envelope "$OUT/serial/trace.json" "$OUT/joint/trace.json"
```

A `WithinDecisionEnvelope` verdict on real prompts closes Q-C2b. A `RouteSensitive` verdict requires localization and blocks any "neutral" claim (policy §3.2). The two forked A/B turns are classified separately with the policy's fork rule.

## Measured result

**Qwen3.8 27B, four-agent C262144 coding** (Q5 supervisor, ABBA, 4 runs per arm, same binary, `HF2Q_QWEN_JOINT_DECODE=0` vs default; `E/qwen-joint-ab-v1`):

| Metric | Serial | Joint |
|---|---|---|
| Makespan, median | 116.4 s | 90.1 s |
| Makespan, range | 115.8–124.5 s | 87.4–91.4 s |
| Per-request decode, median | 8.9–9.7 tok/s | 12.2–12.5 tok/s |
| Client semantic TTFT, median | 9.0–9.7 s | 7.0–7.5 s |
| Client semantic TTFT, max | 20–24 s | 10.7–11.2 s |
| Outcome | 4/4 `CalibrationPass` | 4/4 `CalibrationPass` |

- The makespan medians give a 22.6% time reduction (1.29× by medians).
- `analysis-qwen38.json` gives a geometric-mean time ratio of 1.316 (95% CI 1.25–1.38).
- The per-request decode and TTFT medians are independent recomputations from the run logs (`analysis-qwen38.json` has a parsing bug in its decode medians).
- Under Q1 §3.5, frozen prompts on a deterministic engine are one sample per agent: no task-quality loss was observed, but this is not a quality verdict.

Still missing: the joint width distribution (only the first engagement was logged) and decode scratch at B=4.

**Qwen3.6 verdict: inconclusive, unfavourable.** Default off.

- **Forward.** The forward is bitwise exact (gate above).
- **A/B coverage.** The four-agent A/B (`E/qwen-joint-ab-v1/q36-*`) is incomplete: 2 joint and 2 serial runs.
- **Joint runs.** Both joint runs were `ClientFailed` with the same failure: agent-0 did not follow the initial six-step workflow, and agent-1 also failed.
- **Serial runs.** One was `CalibrationPass`. In both, agent-0 completed the protocol.
- **Transcripts.** `q1_decision_policy.py transcripts` gives `DivergedTranscripts` (6 of 49 groups), including agent-0 turn-0. Stock itself was not repeat-deterministic here (its two runs differ in two groups); the two joint runs were identical.
- **Speed.** Per-request decode rose from about 27.6–28.2 to 35.9–37.4 tok/s; there is no makespan comparison.

Re-enabling by default needs a complete ABBA (≥ 4+4) with protocol parity. The suggested serving-level Class A check is both arms with `HF2Q_QWEN_SPECULATION=off`, where `transcripts` must give `ExactTranscripts`.

## Not in phase 1

- Joint temporal verify (MTP / history proposals inside the joint step).
- Batched MTP catch-up.
- Packed routed experts.
- A native per-sequence DeltaNet dispatch.
- An N=2 measurement. [INFERENCE] A two-row padded step may lose to two serial K3 rounds on Qwen3.8; one N=2 ABBA point would decide whether MTP-capable slots should join only at ≥ 3 eligible rows.
