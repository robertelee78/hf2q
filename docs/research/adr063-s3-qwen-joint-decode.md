# ADR-063 S3 — Qwen joint B-row decode (phase 1)

Status: **implemented, default on for admitted artifacts; hardware gates and four-agent A/B pending.** `HF2Q_QWEN_JOINT_DECODE=0` restores the serial path. Nothing in this note is a measured result: the A/B and the official-artifact gates below have not run yet.

## Problem

With four Qwen agents decoding, the SlotAware worker ran one target forward per slot per turn (`decode_batch_qwen35` → `Qwen35DecodeState::decode_tick`). Every slot re-read every weight. The earlier shared-forward spikes shared only the post-attention FFN and head, formed pairs at best, synchronised every layer on the host, and ran pairs on Qwen3.8's plain `Mv` route (no weight reuse). All were slower than serial (NegativeResult docs under `spike-s3-three-la-checked-group`).

## Phase 1 design

One scheduler turn advances every eligible decoding slot (2–4 rows by default) through **one layer-major target forward**, mirroring the DeepSeek-V4 joint decode (`docs/research/adr063-s3-ds4-joint-decode.md`).

**Scheduler** (`src/serve/api/engine/qwen35_joint_decode.rs`):

- A handle is eligible when its installed slot matches, it holds decode work, its client is open, its next tick is a target forward (no queued verified speculative output, step bound not reached), and its cache cursor equals the committed ledger (prompt + generated − 1), with an equal MTP cursor when it holds MTP state. Other handles stay on the serial loop in scheduler order.
- Lanes are sorted by slot and capped at `HF2Q_QWEN_JOINT_DECODE_MAX_ROWS` (2..8, default 4). Fewer than two eligible lanes keep the whole turn serial, including MTP K3 and history speculation.
- Each lane prepares exactly the inputs its serial tick would feed the target (`Qwen35DecodeState::prepare_joint_tick`), the step runs under one supervisor lease, and each lane commits through its serial tick's own commit code (`finish_joint_tick`). Settlement (emit, progress, retained prefix, finish, cancel recovery, error reset) is `settle_qwen35_decode_lane`, extracted from `decode_batch_qwen35` and shared by both paths.
- The switch and width are read once at worker start, which logs `requested`, `max_rows`, `selected`, `mode` and the `unavailable` reason. A per-turn check revokes joint decode (serial for the rest of the process) if the routing policy changes after start.

**Speculation in phase 1:** joint rows run the ordinary target.

- Slots without MTP state take `decode_tick`'s ordinary path: thinking-budget forcing, the GPU argmax kernel for greedy rows, constrained sampling otherwise.
- Slots with MTP state take `decode_tick_mtp_warmup`'s coherent path: after the shared target forward they run their own MTP catch-up over the same token (`process_target_batch`), so target and MTP cursors stay equal and `spec_prefix_candidate` stays valid. A later solo turn resumes K3 drafting. The cost controllers observe the joint decision's wall time as the ordinary baseline.
- History proposals are suppressed. The lookup ledger catches up from the committed tokens on the slot's next solo round, exactly as after a serial lookup miss.

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

MoE layers run per row because the router and shared expert are float projections whose packed kernel differs from `m = 1`, and the routed experts of four rows are mostly distinct. Packing them would trade exactness for little weight reuse. Qwen3.6's gain in phase 1 therefore comes from the DeltaNet, attention and head projections only.

## Exactness contract and route admission

`joint_decode_routes.rs` admits a joint step only before any cache write.

**Exact mode** (`HF2Q_DECODE_MV_EXT` not `1`; Qwen3.6 and the Qwen3.8 bitwise arm): every row's logits, post-norm hidden, TQ KV codes and scales, conv and recurrent state, cursor and `pp_flipped` are bitwise those of its serial one-token forward. Packed products are admitted only for Q4_0, Q8_0, Q4_K, Q5_K and Q6_K storage, whose `m = 2..=8` routes (`Mv`, the `mvN` K-quant tiles, canonical Q5_K) keep one-row arithmetic. Float storage (F32/F16/BF16, e.g. small α/β projections) switches from a matvec at `m = 1` to tiled kernels at `m > 1`, so it runs one serial dispatch per row and is gathered. Any other block type, a float FFN arm, GLP steering, layer dumps, `HF2Q_LEGACY_PER_LAYER_CB=1` or `MLX_UNRETAINED_REFS=1` keep the artifact serial, with the reason logged.

**Padded `mul_mv_ext` mode** (Qwen3.8 production policy, `HF2Q_DECODE_MV_EXT=1`, dense artifacts only): the production route is not bit-exact, so the contract is ADR-044's decision parity. Packed products run at a fixed padded width (4, or 8 above four rows) so a row's arithmetic never depends on how many tenants share the step; padding rows replay row 0's token and carry row 0's core outputs, never touch a cache, and are discarded. MoE artifacts under `HF2Q_DECODE_MV_EXT=1` stay serial.

## Failure semantics

- Refusal (admission, duplicate or out-of-range slot, active DeltaNet capture) happens before any write and leaves the turn serial or fails nothing.
- A submitted failure fails exactly the step's rows: each gets the serial error path (slot reset, request error); rows outside the step are untouched. An unhealthy supervisor or a fatal command-buffer error fail-stops the worker, as in serial decode.

## Gates (root-run)

Host tests: `qwen35_joint_decode::tests`, `joint_decode_routes::tests`.

Metal tests (packed vs per-row, bitwise): `qwen35::forward_gpu::joint_decode::tests` — every admitted block type at `m = 2..4` through the DeltaNet projection helper and the gather path, and the packed dense FFN against per-row FFNs. A failing type must leave the admitted set.

Official-artifact gates (`qwen35::real_artifact_joint_decode_tests`, ignored): four slots with ragged prompts in two identical caches; serial one-token forwards vs joint steps over full and partial cohorts, a lone row, a row that stops while others continue, and a slot reset and re-prefilled before rejoining; refused steps must write nothing.

- `qwen36_joint_decode_matches_serial_decode_bitwise` (`HF2Q_QWEN36_GGUF`): logits, hidden, greedy tokens, cursors, parities and all KV/state bytes.
- `qwen38_joint_decode_mvn_matches_serial_decode_bitwise` (`HF2Q_QWEN38_GGUF`, `HF2Q_DECODE_MV_EXT` unset): the same bitwise contract on the exact mvN route.
- `qwen38_joint_decode_padded_mv_ext_keeps_decision_parity` (`HF2Q_QWEN38_GGUF`, `HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1`): teacher-forced decisions equal independent sequential forwards at every step, cursors and parities equal, then 16 further serial decisions per slot equal across the two caches.

Four-agent A/B: the same binary, `HF2Q_QWEN_JOINT_DECODE=0` vs default, ABBA under `scripts/four_agent/q5_calibration.py` with the `qwen38` and `qwen36` cases (the supervisor passes only the case's `serve.env`, so each arm uses a derived case that sets the switch). Report per-agent decode rate, aggregate, TTFT, makespan, the joint width distribution, and decode scratch at B=4.

## Expected impact [INFERENCE]

From the scout's byte arithmetic, not measured: Qwen3.8 phase 1 at B=4 on `mul_mv_ext` should cost about one width-4 verify per step (~60–65 ms), i.e. roughly 1.7–1.85× today's ~9 tok/s per agent; Qwen3.6 gains less in phase 1 because MoE stays per row.

## Not in phase 1

Joint temporal verify (MTP / history proposals inside the joint step), batched MTP catch-up, packed routed experts, and a native per-sequence DeltaNet dispatch.
