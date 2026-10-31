# ADR-063 S3 — Qwen joint B-row decode (phase 1)

Status (2026-10-30):
- **Qwen3.8 (dense): default on.** Official-artifact gates passed and the four-agent ABBA A/B passed; see "Measured result".
- **Qwen3.6 (MoE): default off, opt-in with `HF2Q_QWEN_JOINT_DECODE=1`.** Its forward is bitwise exact, but the four-agent A/B is incomplete and both joint runs failed the coding protocol; see "Qwen3.6 verdict".
- `HF2Q_QWEN_JOINT_DECODE=0` restores the serial path for every family.
- **Qwen3.8 shared-weight dense FFN (padded mode): default on in this change, pending root gates.** Joint steps no longer re-read the FFN gate/up weights once per packed row; `HF2Q_QWEN_JOINT_FFN_SHARED=0` restores the per-row fused kernel. See "Dense FFN weight sharing (padded mode)".
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
| Dense FFN (Qwen3.8) | packed over the padded rows. Padded mode (default): gate and up as separate weight-shared products, `silu_mul`, shared down, residual add (`JointDenseFfnRoute::SharedWeights`). Exact mode, or `HF2Q_QWEN_JOINT_FFN_SHARED=0`: the serial `build_dense_ffn_layer_gpu_q_into` at `seq = width`, whose fused gate/up/SiLU kernel reads the gate and up weights once per row |
| MoE FFN (Qwen3.6) | per row through the serial `build_moe_ffn_layer_gpu_q_into` |
| Output head | packed lm_head (or per row), GPU argmax per greedy row, full logits and post-norm hidden otherwise |
| Command buffers | the serial graph's layer grouping (`HF2Q_PARTIAL_CHAIN_N` / table default), one terminal wait per step |

The serial graphs are unchanged; `joint_decode_core.rs` repeats their dispatch order around the per-row cores. The dense FFN builder takes an explicit gate/up dispatch (`DenseGateUpDispatch`); `build_dense_ffn_layer_gpu_q_into`, used by every serial path, keeps the fused preference. Cursors and ping-pong parities advance at encode time exactly as in the serial graph.

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
- **Dense FFN.** The default shared-weight gate/up route (next section) is part of this Class B route. It is not bitwise equal to the per-row fused kernel, so no bitwise gate applies to it; exact mode refuses it before any cache write.

## Dense FFN weight sharing (padded mode)

**Finding (verified from code, mlx-native 0.15.15).** Before this change the joint graph ran the dense FFN through the serial `build_dense_ffn_layer_gpu_q_into` at `seq = 4`. For Q4_K gate/up that builder prefers `fused_gate_up_silu_q4_K` whenever fusion is on (the default; only `HF2Q_FUSED_GATE_UP_SILU=0|false|off` disables it). That kernel's grid is `(ceil(N/2), M, 1)` and the shader indexes the input row by `tgpig.y`, so every packed row streams the whole gate and up matrices: 4 passes per step at the fixed padded width, padding rows included. Every other product of the step goes through `quantized_matmul_ggml` at `m = 4`. Under the Qwen3.8 serving policy (`HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1`) the planner sends Q4_K and Q6_K at `m = 4` to `mul_mv_ext` with `r1ptg = 4` (grid `(N/r0ptg, 1, 1)`): one weight pass for all four rows. With `mvN` on, Q4_K/Q6_K take the single-tile `mvN` route at `m = 4`, also one pass.

**Change.** In the padded mode the joint graph now calls the builder with `DenseGateUpDispatch::Separate`: gate and up are two packed `quantized_matmul_ggml` products on the same width-4 routes as the attention projections, then `silu_mul`, then the unchanged down product and residual add. Serial one-token decode, prefill, MTP and exact mode keep the fused kernel. `HF2Q_QWEN_JOINT_FFN_SHARED=0` (read once at worker start, logged as `dense_ffn`) restores it for A/B.

**Weight bytes per joint step** (official `Qwen3.8-27B-vanilla.gguf`, GGUF header read with the gguf Python reader; 64 decoder layers = 48 DeltaNet + 16 full attention; the `blk.64` MTP layer and the embedding gather are not in the step):

| Product | GGUF type × layers | Shape [K, N] | Bytes per pass | Kernel at width 4 | Passes before | Passes now |
|---|---|---|---:|---|---:|---:|
| DeltaNet `attn_qkv` | Q6_K × 24, Q4_K × 24 | [5120, 10240] | 1,739,980,800 | `mul_mv_ext` | 1 | 1 |
| DeltaNet `attn_gate` (z) | Q4_K × 48 | [5120, 6144] | 849,346,560 | `mul_mv_ext` | 1 | 1 |
| DeltaNet `ssm_alpha` + `ssm_beta` | Q4_K × 48 each | [5120, 48] | 13,271,040 | `mul_mv_ext` | 1 | 1 |
| DeltaNet `ssm_out` | Q4_K × 48 | [6144, 5120] | 849,346,560 | `mul_mv_ext` | 1 | 1 |
| Attention `attn_q` (fused Q+gate) | Q4_K × 16 | [5120, 12288] | 566,231,040 | `mul_mv_ext` | 1 | 1 |
| Attention `attn_k` | Q4_K × 16 | [5120, 1024] | 47,185,920 | `mul_mv_ext` | 1 | 1 |
| Attention `attn_v` | Q6_K × 8, Q4_K × 8 | [5120, 1024] | 57,999,360 | `mul_mv_ext` | 1 | 1 |
| Attention `attn_output` | Q4_K × 16 | [6144, 5120] | 283,115,520 | `mul_mv_ext` | 1 | 1 |
| FFN `ffn_gate` | Q4_K × 64 | [5120, 17408] | 3,208,642,560 | before: fused per row; now: `mul_mv_ext` | **4** | 1 |
| FFN `ffn_up` | Q4_K × 64 | [5120, 17408] | 3,208,642,560 | before: fused per row; now: `mul_mv_ext` | **4** | 1 |
| FFN `ffn_down` | Q6_K × 32, Q4_K × 32 | [17408, 5120] | 3,943,956,480 | `mul_mv_ext` | 1 | 1 |
| Output head `output` | Q6_K × 1 | [5120, 248320] | 1,042,944,000 | `mul_mv_ext` | 1 | 1 |
| **Total per step** | | | 15,810,662,400 | | **35,062,517,760** | **15,810,662,400** |

Gate and up were 25.67 GB of the 35.06 GB the step's kernels read (73%). The change removes 19.25 GB per step (−54.9%), independent of how many of the 4 packed rows are real tenants. A serial one-token forward reads the same 15.81 GB, so the padded joint step now costs one serial forward's weight stream instead of 2.2 of them.

**Exactness class.** Q1 Class B vs stock (unchanged class for the padded mode): `mul_mv_ext` reduces in a different order from the fused kernel, so the new route is not bitwise equal to the old one. Its rows must still be cohort invariant at the fixed width 4, which the Metal gate and the official-artifact invariance gate check.

**Expected saving** [INFERENCE; nothing here is measured]. Decode is weight-bandwidth bound. The repo's planning figure for M5 Max is about 400–410 GB/s (`docs/research/adr040-n4-decode-profile-plan-2026-05-30.md`, `docs/research/omlx-2026-04-30.md`). The S3 A/B is consistent with it: joint per-request decode of 12.2–12.5 tok/s with 4 lanes is about 81 ms per joint turn, and 35.06 GB / 81 ms ≈ 433 GB/s (the turn also includes MTP catch-ups, so this is a rough upper estimate of the weight time).

| Effective bandwidth | Weight time before | Weight time now | Saved per joint step |
|---|---:|---:|---:|
| 400 GB/s | 87.7 ms | 39.5 ms | 48.1 ms |
| 433 GB/s | 81.0 ms | 36.5 ms | 44.5 ms |
| 546 GB/s | 64.2 ms | 29.0 ms | 35.3 ms |

If joint turns are fully bandwidth bound, they become up to about 2.2× faster. The makespan effect is smaller: it scales with the share of the four-agent makespan spent in joint steps, which is not attributed yet (prefill, tool time, MTP catch-ups and serial turns are unchanged). Two things can shrink the gain. The GPU's system-level cache may already absorb part of the per-row re-reads (a 100 MB gate+up pair per layer is larger than it, so only partial hits are expected). And `mul_mv_ext` at `r1ptg = 4` does four rows of dequantization arithmetic per weight read, so the new products could become partly compute bound. The A/B below decides.

**Root commands.** `E=/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`, release binary built from this branch.

A/B, same binary, default vs kill switch, ABBA × 4 (8 runs; case copies follow the `E/qwen-joint-ab-v1` pattern):

```sh
AB=$E/qwen-joint-ffn-ab-v1; mkdir -p "$AB"
python3 - "$AB" <<'PY'
import json, sys
case = json.load(open("scripts/four_agent/q5_cases/qwen38.json"))
for arm, extra in (("ffn1", {}), ("ffn0", {"HF2Q_QWEN_JOINT_FFN_SHARED": "0"})):
    c = json.loads(json.dumps(case))
    c["serve"]["env"].update({"RUST_LOG": "info", **extra})
    json.dump(c, open(f"{sys.argv[1]}/qwen38-{arm}.json", "w"), indent=2)
PY
export HF2Q_Q5_HF2Q_BIN=/abs/release/hf2q HF2Q_Q5_MODEL=/Users/robert/.local/share/hf2q/models/qwen38-stock/Qwen3.8-27B-vanilla.gguf
export HF2Q_Q5_CARGO=/abs/cargo HF2Q_Q5_TOOLCHAIN=/abs/toolchain HF2Q_Q5_SANDBOX_PROOF=/abs/sandbox-proof.json HF2Q_Q5_SANDBOX_PROOF_SHA256=<sha256> HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/qwen38-client-approval.json
i=0; for arm in ffn1 ffn0 ffn0 ffn1 ffn1 ffn0 ffn0 ffn1; do i=$((i+1))
  python3 -B scripts/four_agent/q5_calibration.py "$AB/qwen38-$arm.json" \
    --negative-control-receipt "$NC_RECEIPT" --output "$AB/q38-$i-$arm-$(date -u +%Y%m%dT%H%M%SZ)"
done
```

Every run must log `dense_ffn=shared_weights` (ffn1) or `dense_ffn=serial_builder` (ffn0) in `Qwen joint decode engaged`. Report the geometric-mean makespan ratio with its 95% CI, as for `E/qwen-joint-ab-v1`.

Paired task quality against stock (Q1 §3.5), reference = the same binary with the kill switch:

```sh
python3 -B scripts/four_agent/quality_campaign.py \
  --subject-case "$AB/qwen38-ffn1.json" \
  --reference-case "$AB/qwen38-ffn0.json" --reference-kind stock-hf2q \
  --instances 24 --negative-control-receipt "$NC_RECEIPT" \
  --output "$E/quality-qwen38-joint-ffn-$(date -u +%Y%m%dT%H%M%SZ)"   # --dry-run first
```

The reference arm is this binary with the kill switch, i.e. main's joint decode with the per-row fused gate/up. Run both commands from the worktree root. A reference without joint decode at all is the published `v0.1.28` binary on the unmodified `scripts/four_agent/q5_cases/qwen38.json` (`--reference-var HF2Q_BIN=/abs/v0.1.28/hf2q`); that comparison mixes this change with the whole S3 joint route.

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

**Host tests:** `qwen35_joint_decode::tests`, `joint_decode_routes::tests` (kill-switch parsing, exact mode never takes the shared FFN, and the planner's one-pass route for every admitted type at the Qwen3.8 FFN shapes and `m = 4` under both padded-mode policies).

**Metal tests.**
- `qwen35::forward_gpu::joint_decode::tests` (packed vs per-row, bitwise): every admitted block type at `m = 2..4` through the DeltaNet projection helper and the gather path, and the packed serial-builder dense FFN against per-row FFNs. A failing type must leave the admitted set.
- `qwen35::joint_decode_ffn_tests` (shared FFN, padded-mode policy frozen on the registry, `mvN` off and on): each row's bytes are identical across packed orders and a full padding replay, and each row is within 1e-3 (relative to the row's max magnitude) of the serial `m = 1` builder.

**Official-artifact gates** (`qwen35::real_artifact_joint_decode_tests`, ignored). Four slots with ragged prompts in two identical caches: serial one-token forwards vs joint steps over full and partial cohorts, a lone row, a row that stops while others continue, and a slot reset and re-prefilled before rejoining. Refused steps must write nothing.

- `qwen36_joint_decode_matches_serial_decode_bitwise` (`HF2Q_QWEN36_GGUF`): logits, hidden, greedy tokens, cursors, parities and all KV/state bytes. **Passed** (`E/qwen-joint-exactness-v1`).
- `qwen38_joint_decode_mvn_matches_serial_decode_bitwise` (`HF2Q_QWEN38_GGUF`, `HF2Q_DECODE_MV_EXT` unset): the same bitwise contract on the exact mvN route. **Passed.**
- `qwen38_joint_decode_padded_mv_ext_keeps_decision_parity` (`HF2Q_QWEN38_GGUF`, `HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1`): teacher-forced top-1 decisions equal independent sequential forwards at every step, cursors and parities equal, then 16 further serial decisions per slot equal. **Passed** on the per-row fused FFN; it now runs the FFN route `HF2Q_QWEN_JOINT_FFN_SHARED` selects (default shared). Under Q1 it is diagnostic only (synthetic prompts, top-1).
- Exact mode refuses a valid cohort on the shared FFN route before any cache write (checked inside the bitwise gates).

**Cohort invariance** (`qwen35::real_artifact_joint_decode_invariance_tests`, ignored; not yet run). For each variant, one cache with four identically prefilled slots advances slot 2 for 4 steps inside a different cohort: alone, `{0,2}`, `{2,3}`, `{1,2,3}`, `{2,3,1}`, and `{0,1,3,2}` with slot 2 packed last. Slot 2's logits, hidden, cursor, parity, KV codes/scales and conv/recurrent bytes must be bitwise equal across all variants.

- `qwen36_joint_decode_rows_are_cohort_invariant_bitwise` (`HF2Q_QWEN36_GGUF`)
- `qwen38_joint_decode_mvn_rows_are_cohort_invariant_bitwise` (`HF2Q_QWEN38_GGUF`, `HF2Q_DECODE_MV_EXT` unset)
- `qwen38_joint_decode_padded_mv_ext_rows_are_cohort_invariant_bitwise` (`HF2Q_QWEN38_GGUF`, `HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1`): both dense FFN routes, shared weights first.

**Q1 Class B envelope on real prompts** (Qwen3.8 padded mode; not yet run; `qwen35::real_artifact_joint_decode_q1_tests`).

`qwen38_joint_decode_q1_envelope_trace` takes 2–4 real agent prompts (`HF2Q_QWEN_JOINT_Q1_PROMPTS`) and requires the padded mode:
- **Input format.** A JSON array where each entry is either the exact rendered chat prompt as token IDs, or the rendered prompt text tokenized with the serving `tokenizer.json` at `HF2Q_QWEN35_TOKENIZER`. Use the four-agent turn-0 and turn-2 requests, including the two forked turns (agent-1 and agent-3 turn-2).
- **Run.** It prefills each prompt into three identical caches, then decodes `HF2Q_QWEN_JOINT_Q1_STEPS` (default 32) teacher-forced steps. Every cache consumes the serial top-1 at every step.
- **Stores** (`hf2q.adr063.invariance.v2`, keyed by agent, absolute row and step):
  - `$OUT/serial/trace.json`: the serial one-token `forward_gpu_last_logits` (stock M1 target route);
  - `$OUT/joint_shared_ffn/trace.json`: `forward_gpu_decode_joint` over all agents with the shared-weight FFN (production default);
  - `$OUT/joint_serial_ffn/trace.json`: the same with the per-row fused FFN (`HF2Q_QWEN_JOINT_FFN_SHARED=0`).

Root commands, the test from this branch and `q1_decision_policy.py` from any checkout containing PR #465:

```sh
OUT=$E/q1-qwen38-joint-envelope-$(date -u +%Y%m%dT%H%M%SZ)
HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1 \
HF2Q_QWEN38_GGUF=/Users/robert/.local/share/hf2q/models/qwen38-stock/Qwen3.8-27B-vanilla.gguf \
HF2Q_QWEN_JOINT_Q1_PROMPTS=/abs/qwen38-agent-prompts.json \
HF2Q_QWEN35_TOKENIZER=/abs/qwen38-tokenizer.json \
HF2Q_QWEN_JOINT_Q1_OUT=$OUT \
  cargo test --release --lib qwen38_joint_decode_q1_envelope_trace -- --ignored --exact --nocapture --test-threads=1
python3 -B scripts/four_agent/q1_decision_policy.py envelope "$OUT/serial/trace.json" "$OUT/joint_shared_ffn/trace.json"
python3 -B scripts/four_agent/q1_decision_policy.py envelope "$OUT/serial/trace.json" "$OUT/joint_serial_ffn/trace.json"
```

The first verdict gates the shared FFN route; the second is Q-C2b for the per-row route that served the S3 A/B. A `WithinDecisionEnvelope` verdict on real prompts closes the respective gate. A `RouteSensitive` verdict requires localization and blocks any "neutral" claim (policy §3.2). The two forked A/B turns are classified separately with the policy's fork rule.

**Focused root commands** for this change:

```sh
cargo test --release --lib joint_decode_routes
cargo test --release --lib qwen35_joint_decode
cargo test --release --lib joint_decode_ffn_tests -- --test-threads=1
cargo test --release --lib forward_gpu::joint_decode::tests -- --test-threads=1
HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1 \
HF2Q_QWEN38_GGUF=/Users/robert/.local/share/hf2q/models/qwen38-stock/Qwen3.8-27B-vanilla.gguf \
  cargo test --release --lib qwen38_joint_decode_padded_mv_ext_rows_are_cohort_invariant_bitwise -- --ignored --exact --nocapture
HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1 \
HF2Q_QWEN38_GGUF=/Users/robert/.local/share/hf2q/models/qwen38-stock/Qwen3.8-27B-vanilla.gguf \
  cargo test --release --lib qwen38_joint_decode_padded_mv_ext_keeps_decision_parity -- --ignored --exact --nocapture
HF2Q_QWEN38_GGUF=/Users/robert/.local/share/hf2q/models/qwen38-stock/Qwen3.8-27B-vanilla.gguf \
  cargo test --release --lib qwen38_joint_decode_mvn_matches_serial_decode_bitwise -- --ignored --exact --nocapture
```

## Measured result

**Qwen3.8 27B, four-agent C262144 coding** (Q5 supervisor, ABBA, 4 runs per arm, same binary, `HF2Q_QWEN_JOINT_DECODE=0` vs default; `E/qwen-joint-ab-v1`; joint arm on the per-row fused FFN, before the shared-weight change):

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
