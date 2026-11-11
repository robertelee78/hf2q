# HF2Q environment surface inventory

This is the source-bound lexical inventory accepted by ADR-050 on 2026-08-23.
It includes names in production code, embedded test modules, comments, and
regression guards. A name appearing here is not automatically a supported
operator setting.

Regenerate the table with:

```bash
bash scripts/audit_hf2q_env_surface.sh
```

The snapshot contains **353 distinct names** after the ADR-050 removals and
the ADR-062 D3 slot-gate removals. The pre-change audit contained 363.
Dispositions mean:

- `appropriate-env`: environment is an intentional boundary for a secret,
  XDG-like location, or shell/package integration.
- `promote-or-internalize`: normal or qualified production behavior must move
  to typed CLI/config or an internal family policy; it must not remain hidden
  shell UX.
- `documented-escape`: a safe, technical algorithm escape hatch that is not
  part of the ordinary setup path.
- `development-only`: benchmark, profile, dump, test, fixture, fault-injection,
  unsafe experiment, or unfinished feature control. These do not warrant CLI
  expansion while they remain development surfaces.
- `removed-guard`: only a regression assertion names the removed production
  environment reader.

## Completed in ADR-050

The production readers for scheduler, max slots, shared KV budget, persistent
KV disk budget, DeepSeek context, and the three server behavior defaults were
removed. `HF2Q_KV_PERSIST=0` also no longer defeats an explicit typed serve
path. Their supported replacements are typed CLI/config fields. The late
process mutation behind `generate --kv-bits` was replaced with a typed
in-process override; its legacy environment input remains development-only.

## Completed in ADR-062 D3 (2026-10-08)

The `HF2Q_MAX_BATCHED_SLOTS`, `HF2Q_SPEC_DECODE_MAX_BATCHED_SLOTS`, and
`HF2Q_SPEC_DECODE_ALLOW_OVERSIZED` production readers and their rows are
removed: slot capacity is a validated operator setting (`--max-slots`
accepts 1-8 on every family, checked at the CLI, config load, and the setup
prompt; raising the bound requires a hands-on qualification at the new
width). No `src/` occurrence of the three names remains.

## Completed in #730 (2026-11-06)

The `HF2Q_KV_LCP_RESUME_CAPACITY` production reader is removed. The
in-memory LCP resume registry budget is the typed `serve
--lcp-resume-budget <SIZE>` / `[serve] lcp_resume_budget` setting with a
host-independent built-in default of 8GiB, resolved once at server start; the
former free-memory probe (`available_memory() × 5%`) is gone. Only a
regression guard names the variable, so its row is now `removed-guard`. The
`HF2Q_KV_LCP_RESUME` and `HF2Q_KV_LCP_RESUME_CAPACITY` rows were regenerated
with the audit script for this change (the lexical count of
`HF2Q_KV_LCP_RESUME` includes the longer name); other rows were not.

## Added in ADR-063 S3 (2026-10-30)

- `HF2Q_DEEPSEEK_JOINT_DECODE` (`documented-escape`) is the DeepSeek-V4 joint decode kill switch. It is on by default; `0`/`off`/`false` restores serial per-slot decode.
- `HF2Q_DEEPSEEK_JOINT_DECODE_MAX_ROWS` (`development-only`) sets the joint width, 2..8 with a default of 4. Widths 5–8 have no serving measurement.
- `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC` (`development-only`, ADR-063 gap program, 2026-11-06) selects the DeepSeek-V4 few-row kernel family. `0`: the existing routes. Unset: the build default (`HF2Q_DEEPSEEK_ROW_EXACT_MATVEC_DEFAULT`, the existing routes unless the build set it). `1`: F16 products at `m = 1` and independent F16 rows, Q8_0 products at `m <= 8` and Q8_0 grouped output-A run mlx-native's `row_exact_matvec_stock`, one dispatch per product whose every row is bitwise the stock one-row kernel's, so both settings produce the same bits; Q8_0 takes it only where the stock route is the NR2 matvec (never under `HF2Q_Q8_0_MV_NR2=0`, only one-row products under `HF2Q_DECODE_MV_EXT=1`) and otherwise keeps the stock route; F16 prefill rows keep the stock tiled kernel. Any other value refuses model load. Read once at model load; the serial, joint, prefill and verifier graphs of the process share the choice.
- `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC_DEFAULT` (`development-only`, build time) sets the family used when `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC` is unset at run time: `1` builds a binary whose default is the row-exact family, `0` or unset keeps Stock, and any other value fails the build. Resolution order is the runtime switch, then this build default, then Stock. It exists for harnesses that strip `HF2Q_*` from the runtime environment (the Q1 endpoint supervisor): the two Q1 arms are two binaries of one commit.
- `HF2Q_DEEPSEEK_TAIL_ROWS_STEP` (`development-only`, ADR-063 gap program fix 1, 2026-11-07) runs DeepSeek-V4 token-wise prompt work (the 8-token recovery tail and cached suffixes below the 33-token matrix floor) as rows steps of up to 8 tokens instead of one-token verifier steps. A rows step is one layer-major transaction on the joint-decode routes with the per-row attention cores in row order on the request's cache, so every row's state and every cache byte equal the one-token steps (Class A). Unset or `0`: one-token steps; `1`: rows steps when joint decode is admissible for the artifact and process (otherwise one-token steps). Any other value refuses model load. Read once at model load.
- `HF2Q_DEEPSEEK_BATCHED_ATTN_CORES` (`development-only`, ADR-063 gap program fix B stage 1, 2026-11-06) selects how DeepSeek-V4 joint decode encodes the cache-dependent attention cores. Unset or `0`: one row at a time, as before. `1`: phase-major, one mlx-native `deepseek_rows` dispatch per op over all joint rows (compressor recurrences, indexer score + top-k, sparse attention; the flash path once per distinct width), each row against its own cache through a fixed binding slot; the compressor/indexer F16 projections become one `row_matmul` over the rows (one `m = B` dispatch with `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC=1`). Every row stays bitwise equal to the per-row cores; layers with a row at position 0 keep the per-row cores. Serial decode is unaffected. Any other value refuses model load. Read once at model load. Off by default until the real-artifact joint ≡ serial gate passes with the switch on.
- `HF2Q_GEMMA_FULL_INPUT_CHECKPOINT` (`documented-escape`) is the Gemma 4 full-input checkpoint switch. Unset means on where the worker's substrate and quota plan allow it; `0` is the kill switch (also skips the slot-route warmup); `1` forces it and refuses startup otherwise. It no longer needs `HF2Q_UNSAFE_EXPERIMENTS`.
- `HF2Q_GEMMA_JOINT_WARM_PREFILL` (`documented-escape`, listed in `scripts/audit_hf2q_env_surface.sh`) is the Gemma 4 joint warm prefill kill switch (ADR-063 Stage 1). Unset or `1`/`on`/`true`: same-step warm tool-turn suffixes share one bitwise-exact transaction; `0`/`off`/`false` restores per-slot warm prefill. Invalid values warn and keep the default. Read once per SlotAware worker; also off when `HF2Q_CROSS_SLOT_ADMIT` is off, `HF2Q_PREFILL_SLOT_BATCHED=0` or `HF2Q_MM_ID_ROUTING_THRESHOLD` is set.
- `HF2Q_GEMMA_WARM_TILED_ATTENTION` (`documented-escape`, listed in `scripts/audit_hf2q_env_surface.sh`) selects the Gemma 4 warm-prefill attention kernel (ADR-063 Stage 2). Unset or `0`/`off`/`false` (default since the 2026-11-01 quality gate): vec-hybrid warm attention. `1`/`on`/`true`: single-slot warm prefills, joint warm cohorts and installed long-prompt lanes attend with tiled F16 flash attention (suffixes under 32 rows keep vec-hybrid); this route changed agent decisions on the canonical four-agent task. Invalid values warn and keep the default. Read once per model load.
- `HF2Q_GEMMA4_GGUF` (`development-only`) names the official Gemma 4 GGUF for the ignored real-artifact checkpoint/warmup/joint-warm-prefill/warm-tiled-attention tests.
- `HF2Q_QWEN_JOINT_DECODE` (`documented-escape`, listed in `scripts/audit_hf2q_env_surface.sh`) is the Qwen joint decode switch. Unset, it is on for dense (Qwen3.8) and MoE (Qwen3.6) artifacts; `0`/`off`/`false` restores serial per-slot decode, including MTP K3 and history speculation.
- `HF2Q_QWEN_JOINT_DECODE_MAX_ROWS` (`development-only`) sets the Qwen joint width, 2..8 with a default of 4; the padded `mul_mv_ext` mode caps at 4.
- `HF2Q_QWEN_JOINT_FFN_SHARED` (`documented-escape`, listed in `scripts/audit_hf2q_env_surface.sh`) is the kill switch for the weight-shared dense FFN of the padded Qwen joint mode. Unset or `1`/`on`/`true`: gate and up run as separate width-4 products (weights read once per step) followed by `silu_mul`; `0`/`off`/`false` restores the serial builder's per-row fused gate/up kernel. Exact mode ignores it. Read once at worker start; invalid values warn and keep the default.
- `HF2Q_QWEN_JOINT_MMA` (`development-only`, experimental) is the diagnostic override for the padded Qwen joint mode's Q4_K/Q6_K layer products. `1`/`on`/`true` forces mlx-native's column-invariant few-row MMA kernel instead of `mul_mv_ext`; `0`/`off`/`false` forces `mul_mv_ext`; unset or blank follows `HF2Q_QWEN_JOINT_MTP`. Rows stay independent of the cohort and padding, and the output head is unchanged. This is Q1 Class B against `mul_mv_ext`. Exact mode ignores it. Read once at worker start; invalid values warn and keep the joint-MTP-derived default.
- `HF2Q_QWEN_JOINT_MTP` (`documented-escape`, listed in `scripts/audit_hf2q_env_surface.sh`) is the production kill switch for the combined joint-MTP candidate on Qwen joint workers: joint MTP verify plus, in padded mode, the column-invariant few-row MMA projection. Dense artifacts with an MTP head, including Qwen3.8, enable it when unset; `0`/`off`/`false` restores serial K3 and canonical projection. MoE artifacts and artifacts without MTP remain off by default; explicit `1`/`on`/`true` enables the route when their other eligibility checks pass. Every eligible decoding lane joins a joint step each turn. A lane with MTP state contributes a window of its pending token plus up to three drafts, at the cap8 depths (a step never exceeds 8 rows: 4 lanes draft 1 each, 3 lanes 2/2/1, 2 lanes 3/3, 1 lane 3). The drafts come from one batched MTP forward per depth. Every verify row is bitwise the spec-off joint row at its position (Class A), and its decision is spec-off's (GPU argmax for greedy requests without logprobs, else the constrained host sampler; forced thinking-budget tokens first). A partial accept replays the accepted DeltaNet rows from the untouched pre-window state, with no capture. All accepted tokens are emitted in the same turn. When active, an unset `HF2Q_QWEN_SPECULATION` resolves to `auto` (MTP prompt catch-up); an explicit `off` keeps the route off; serial K3 rounds and history lookup never run. `HF2Q_QWEN_JOINT_MMA=0` isolates the canonical-projection diagnostic. Read once at worker start (logged as `joint_mtp` and `column_invariant_mma` on the joint decode configuration line, and as `Qwen joint MTP verify route`); invalid values warn and retain the artifact-derived default.
- `HF2Q_QWEN_JOINT_POOLED_OUTPUTS` (`development-only`) takes the Qwen joint layer chain's multi-row outputs (dense FFN output and residual sum, attention projection and Q/gate outputs) from the decode pool instead of fresh Metal allocations. The arithmetic is identical (Class A). `1`/`on`/`true` enables it; unset, blank or `0`/`off`/`false` disables it. Read once at worker start; invalid values warn and keep the default (off).
- `HF2Q_QWEN_JOINT_SOLO` (`documented-escape`, listed in `scripts/audit_hf2q_env_surface.sh`) is the kill switch for the padded-mode solo route (#717). Unset or `1`/`on`/`true`: on workers with two or more slots, every eligible lane joins a joint group, and a lone ordinary lane runs a one-lane padded joint step. `0`/`off`/`false` restores the serial forward for a lone lane and for lanes beyond the joint width. Exact mode and one-slot workers ignore it. Read once at worker start; invalid values warn and keep the default.
- `HF2Q_QWEN_RESUME_ATTN_FAST` (`documented-escape`, listed in `scripts/audit_hf2q_env_surface.sh`) is the kill switch for the Class A fast path of the resumed Qwen prefill attention route for chunks with a cached prefix and more than 32 queries on TQ-only KV (route (c), the BF16 bridge). The same dispatches run on the same bytes, so outputs are bitwise identical; the route takes its scratch from the prefill's FA arena (allocated once per prefill instead of eight zero-filled allocations per FA layer), encodes the TQ dequant and the bridge in one command buffer, and drops the bridge's host wait (the layer's ops6-7 keeps its own). Paths without the arena (MTP drafts, decode, the source teacher) are unchanged. Unset, blank or `1`/`on`/`true` (default): the fast path; `0`/`off`/`false` restores the previous per-layer allocations and wait. Read once per process; invalid values warn and keep the default (on). `HF2Q_QWEN_RESUME_FAST_PROMPTS` (`development-only`) names the token-id prompts (a JSON array of 1–4 token-id lists) of its ignored real-model gate `qwen38_resume_attn_fast_real_model_exact_equal`, which loads `HF2Q_QWEN38_GGUF`; unset, the gate uses one synthetic 9100-token prompt.
- `HF2Q_QWEN_SPECULATION_ROUTE` (`documented-escape`, listed in `scripts/audit_hf2q_env_surface.sh`) selects how speculating Qwen SlotAware lanes are routed and gated. Unset or `repeatable`: each tick's route is decided from the request's own state, a speculating lane never joins a joint step, and the proposer gates use fixed round costs instead of wall time. `adaptive`: the phase-1 cohort-dependent route and measured gates. Read once at engine load; invalid values warn and keep the default (`docs/research/adr063-qwen-spec-repeatable.md`).
- `HF2Q_QWEN_LA_GPU_RESTORE` (`documented-escape`, listed in `scripts/audit_hf2q_env_surface.sh`) selects the ADR-065 P4b DeltaNet capture restore for SlotAware speculative partial rejections. Unset defaults to the qualified GPU route for dense Qwen3.8 and to the host route for Qwen3.6 MoE. `0` forces the previous host copy; `1` forces one GPU command buffer for the parity-aware conv re-index and recurrent blits, bitwise equal to the host route. Read once when the SlotAware cache is provisioned; invalid values fail startup.
- `HF2Q_QWEN_ANCHOR_DEPTH` (`documented-escape`, listed in `scripts/audit_hf2q_env_surface.sh`) bounds committed slot-local Qwen prompt anchors to 1–8. Unset defaults to `1`, the C2 kill-switch value; pending construction is additional. `4` enables the qualified multi-depth campaign shape. Read once at SlotAware worker startup; malformed or out-of-range values fail startup.
- `HF2Q_QWEN_ANCHOR_BYTES_PER_SLOT_MAX` (`documented-escape`, listed in `scripts/audit_hf2q_env_surface.sh`) optionally caps committed plus pending anchor-owned bytes for each Qwen slot. The decimal byte limit must be positive. Captures over the limit are skipped without publication; unset has no runtime quota and relies on Q5 admission. Read once at SlotAware worker startup; malformed or zero values fail startup.
- `HF2Q_ADR065_SHADOW_DRAFT` (`development-only`) enables the E3 counterfactual proposer diagnostic for Qwen3.8. Value `1` requires `HF2Q_QWEN_SPECULATION=off`; after the scheduler forms a real joint group, each lane runs request-history and MTP K3 proposals, discards proposer cache mutations before the ordinary target forward, and compares each draft prefix with later committed target decisions. JSON records include slot, cold/warm turn type, context bucket, proposal wall cost and transform eligibility. Unset is allocation-free and leaves prefill/decode unchanged.
- `HF2Q_ADR065_CRITICAL_PATH` (`development-only`) enables the E4 Qwen3.8 critical-path observer. Exact value `1` emits monotonic closed-schema intervals for each prefill slice and each joint or solo decode turn, including slot identities and per-lane counterfactual verify eligibility. Unset adds no timestamp call, allocation or output on the prefill/decode paths.
- `HF2Q_ADR065_EXPERT_UNION` (`development-only`) enables the E5 DeepSeek-V4 joint-step routed-expert diagnostic. Exact value `1` observes width-2 through width-4 joint decode only. Each FFN layer blit-copies its sanitized routed ids into one per-step buffer; the host reads it after the transaction's existing pipeline drain, adding no synchronization. One closed-schema JSON record reports per-layer unique experts, hash versus score routing, and naive versus unique-expert routed weight bytes. Unset adds no allocation or dispatch.
- `HF2Q_ADR065_STAGE_PROFILE` (`development-only`) enables the ADR-065 E1 complete-round observer for Qwen3.8, DeepSeek-V4 and Gemma 4. Unset or any value except the exact strings below is off and retains the normal command-buffer topology. `wall` emits only complete-round wall time on the normal graph. `1` requires `MLX_PROFILE_CB=1`, deliberately splits and waits at diagnostic boundaries, and emits GPU categories (projections, DeltaNet conv/GDN, KV write, attention, FFN/MoE, head, MTP draft/catch-up, Unknown), CPU categories (logit readback, host selection, rollback, per-row arena setup, host gap), and DeepSeek scratch release/reallocation events. Unknown is retained but excluded from decisions. `scripts/four_agent/adr065_stage_profile.py` compares paired `wall`/`1` Q5 receipts and requires ExactTranscripts against the previous default.
- `HF2Q_QWEN_SPEC_COST_MTP` and `HF2Q_QWEN_SPEC_COST_HISTORY` (`development-only`) override the repeatable route's modelled round costs, in ordinary decisions (defaults 1.26 and 1.49). The repeatability gate uses them to force a cost-disabled proposer. Non-positive or invalid values warn and keep the default.
- `HF2Q_QWEN_MOE_BATCHED` (`documented-escape`, listed in `scripts/audit_hf2q_env_surface.sh`) selects the batched Qwen3.6 MoE decode route. Unset or `0`/`off`/`false` (default): router and shared expert are uploaded as BF16 and every MoE layer runs once per row. `1`/`on`/`true`: they load in GGUF storage (F32 router and gate logit, Q4_K/Q6_K shared expert, no BF16 copies) and serial and joint decode both run one batched layer over the step's rows, row-exact (a row's output is the same solo or in a joint step). Read once at model load; invalid values warn and keep the default.
- `HF2Q_QWEN_MOE_STAGED_M8` (`documented-escape`, listed in `scripts/audit_hf2q_env_surface.sh`) enables the opt-in Apple10 staged-M8 Q4_K/Q6_K expert route. It requires `HF2Q_QWEN_MOE_BATCHED=1`; otherwise startup fails. Model upload prepares quantized gate/up/down planes once. Decode groups live routes once; separate F32 gate and up projections reuse that compact schedule and gathered input, then retain the established F32 SwiGLU and down input. Unset or `0`/`off`/`false` keeps the established native batched expert matvecs. Read once at model load; invalid values or unsupported shapes fail closed.
- `HF2Q_QWEN_ATTN_ROWS` (`documented-escape`, listed in `scripts/audit_hf2q_env_surface.sh`) selects the row-batched Qwen TQ full-attention decode core. Unset or `1`/`on`/`true` (default): serial and joint decode run every row's K/V encode, Q rotation, attention and output gate in one pass, and `HF2Q_QWEN_GQA_Q2=auto` uses the measured Qwen3.6 crossovers (KV ≥ 4,096 for one row, ≥ 1,024 for several). `0`/`off`/`false`: the per-row chain and the single 8,192 crossover. Both give the same bits (Class A). Read once; invalid values warn and keep the default.
- `HF2Q_QWEN_JOINT_Q1_PROMPTS`, `HF2Q_QWEN_JOINT_Q1_OUT` and `HF2Q_QWEN_JOINT_Q1_STEPS` (`development-only`) drive the ignored Q1 Class B trace test.
- `HF2Q_QWEN_MOE_Q1_PROMPTS`, `HF2Q_QWEN_MOE_Q1_OUT`, `HF2Q_QWEN_MOE_Q1_STEPS` and `HF2Q_QWEN_MOE_Q1_FORCED` (`development-only`) drive the ignored two-process Q1 trace test of the batched MoE route (`qwen36_moe_route_q1_envelope_trace`).
- `HF2Q_QWEN36_GGUF` and `HF2Q_QWEN38_GGUF` (`development-only`) select the official artifacts for the ignored Qwen joint decode hardware gates.

These rows were added by hand. The rest of this snapshot was not regenerated.

## Added in ADR-063 Q4-GEMMA (2026-11-03)

- `HF2Q_GEMMA_PREFILL_CATSPLIT` (`development-only`) is the Gemma 4 warm-prefill category split for the Q4-GEMMA profile. Only the exact value `1` enables it; it is off by default. In each layer of a warm live-prefix-resume batched prefill it closes GPU-timed command buffers before and after live attention, before the routed MoE and at the layer end (categories `pre_attn`, `attention`, `dense`, `moe`), and the Gemma 4 worker prints one `[PREFILL_CATSPLIT_JSON]` line of lifetime totals at graceful exit. It adds CPU waits but changes no dispatch; `HF2Q_GRAPH_OPT_PREFILL=1` skips it.
- `HF2Q_GEMMA_ROUTING_OVERLAP` (`development-only`) is the ADR-063 Q4-GEMMA-DECODE V2 decode-workload diagnostic. Only the exact value `1` enables it; it is off by default and read once per process. Each `[N,hidden]` batched decode step blit-copies every layer's routed expert ids into a per-step buffer, read after the step's existing final wait, and records each slot's KV length per layer type and whether the step was mixed-bucket. The Gemma 4 worker prints one `[ROUTING_OVERLAP]` JSON line at graceful exit: distinct experts per (layer, width), KV-length histograms (power-of-two and `(nwg, nsg)` bucket) per layer type, and the mixed-bucket share. It adds one blit per layer; untimed runs only.

This row was added by hand.

## Added in ADR-063 Q4-GEMMA-COLD (2026-11-04)

- `HF2Q_GEMMA_JOINT_COLD_PREFILL` (`documented-escape`, listed in `scripts/audit_hf2q_env_surface.sh`) is the Gemma 4 joint cold prefill kill switch. Unset or `1`/`on`/`true` (default, operator decision D3): simultaneous cold first turns with an exact pre-generation boundary of at least 33 rows (all slots idle, at least two lanes, at most 4,096 rows) prefill in one transaction whose every lane is bitwise equal to its per-slot admission; `0`/`off`/`false` restores per-slot admission. Invalid values warn and keep the default. Read once per SlotAware worker and logged as `Gemma joint cold prefill worker decision`; also off when cross-slot admission or hybrid KV is off, or under `HF2Q_BATCHED_BODY=0`, `HF2Q_PREFILL_SLOT_BATCHED=0`, `HF2Q_MM_ID_ROUTING_THRESHOLD`, `HF2Q_DFLASH_XLEN_SDPA=1`, `HF2Q_FA_F16=0`, `HF2Q_NO_FA` or `HF2Q_GLOBAL_FA`.
- `HF2Q_GEMMA_COLD_TURN0_PROMPTS` (`development-only`) names the JSON file of the four actual turn-0 prompts for the ignored Q4-GEMMA-COLD bitwise gates; `gemma_joint_cold_capture_turn0_prompts` writes it from the recorded run named by `HF2Q_GEMMA_COLD_TURN0_RUN` (`development-only`).
- `HF2Q_GEMMA_COLD_GS3_REPEATS` (`development-only`) sets the timed repeats of the ignored G-S3 harness (`scripts/q4_gemma_cold_gs3.sh`).
- `HF2Q_DUMP_SLIDING_MASK` is removed with the block-diagonal cold multi-sequence mask it dumped (D2 cutover).

## Added in the ADR-063 gap program (2026-11-07)

- `HF2Q_GEMMA_DECODE_ATTN_NE` (`development-only`) selects the Gemma 4 decode attention head split of d256 (sliding) layers. Unset or `0`/`1`/`off`/`false`/`stock`: stock vec-hybrid decode attention. `4` or `2`: the NE head-split kernels on every decode path (batched cohort dispatch, mixed-bucket per-slot fallback and single-slot decode), which share their per-row math, so a slot's decode stays bitwise the same on each path; d512 layers keep stock. Class B against stock. Read once at model load; invalid values warn and keep stock; published as `active_controls.gemma_decode_attn_ne` and bound by the Gemma Q1 packet field `decode_attn_ne`. `serve_gemma4_opencode.sh` forwards `DECODE_ATTN_NE`, and the Gemma Q5 case forwards `HF2Q_Q5_GEMMA_DECODE_ATTN_NE`.

This row was added by hand.

These rows were added by hand.

## Removed after ADR-063 #725 (2026-11-07)

- `HF2Q_GEMMA_TILED_LIVE` is removed with the `Tiled` live attention route it switched, which no serving path selected after #725 and #728; every live resume follows `HF2Q_GEMMA_WARM_TILED_ATTENTION`.

## Added in ADR-063 Q4-GEMMA-DECODE op parity (2026-11-06)

- `HF2Q_GEMMA_DECODE_DENSE` (`development-only`) selects the kernel of every Gemma 4 decode dense GGUF matmul — attention q/k/v/o, dense FFN gate/up/down and the lm head, in the solo decode, the `[N,hidden]` batched body, the batched/fused lm head, the speculative verify heads and tree-verify layers, and the m=1 prefill first-token head. `stock` or unset (default): the existing routes. `row_exact`: mlx-native's row-exact few-row mat-vec for Q4_K/Q6_K with `K % 256 == 0` (Q4_K also with a 16-byte-aligned weight start) and Q5_0/Q8_0 with `K % 32 == 0`. `mma`: the column-invariant few-row MMA mat-vec for Q4_K/Q6_K/Q5_0 with `K % 64 == 0` (and the block size) and Q8_0 with `K % 32 == 0` (numerics class B). Any other value fails the model load; the switch is read once per process. Admission depends on the weight alone, never on the decode width; products wider than 8 rows run in chunks of at most 8 rows, so each row is bitwise equal at every width. Other weights (F32 router, F16/BF16, affine, other block types) and the other (m > 8) prefill matmuls keep their routes. Published as `measurement_snapshot.active_controls.gemma_decode_dense_kernel` (`Stock` / `RowExact` / `ColumnInvariantMma`) and declared as `decode_dense_kernel` in Gemma Q1 endpoint and localizer packets.

This row was added by hand.

## Added for the native Q5_0 tensor-MM opt-in (2026-11-07)

- `HF2Q_Q5_0_TENSOR_MM` (`development-only`) is read by mlx-native 0.18.0, not by hf2q: only `1` routes Q5_0 dense and expert matmuls through the tensor pipeline (default off). hf2q publishes the resolved value as `/hf2q/v1/runtime` `active_controls.q5_0_tensor_mm` and logs it once at the first Gemma 4 load (`Gemma4 native routing: q5_0_tensor_mm=<true|false> mm_id_schedule=<grid|compact>`); the Gemma launcher and Q5 case forward it (`Q5_0_TENSOR_MM`, `HF2Q_Q5_Q5_0_TENSOR_MM`).
- `HF2Q_Q5_0_TENSOR_MM_TEST_EXPECT` (`development-only`) marks the child process of the unit test of that runtime field.

These rows were added by hand.

## Remaining promotion queue

The inventory deliberately identifies follow-up work without smuggling it into
this release:

1. Finish unifying development-generation persistent-KV activation, cache
   management root selection, LCP memory budget, and writer policy into typed
   persistence config. Serve activation/path and the on-disk budget are typed,
   complete, and fail-loud in this release.
2. Promote the multi-model pool byte ceiling to typed config.
3. Replace hidden cross-slot collection controls with qualified
   scheduler/admission policy. Ordinary operators should not need
   launcher-only admission variables.
4. Make required family policies—TQ KV, hybrid KV, encoder sessions, terminal-K
   batching, qualified LM-head routing, and proven batched serving—qualified
   typed defaults. Keep only explicitly unsafe diagnostic opt-outs.
5. Move Qwen speculation and process-global decode routing into typed backend
   policy. The current load-time environment mutation is order-dependent under
   multi-model serving and cannot be the final architecture.

## Complete snapshot

| Name | Disposition | Source occurrences |
|---|---|---:|
| `HF2Q_A4_INFLECTION_BENCH` | `development-only` | 2 |
| `HF2Q_A4_MOE_AB_VALIDATION_E2E` | `development-only` | 3 |
| `HF2Q_ACTIVATION_SCHEMA` | `development-only` | 14 |
| `HF2Q_ADMIT_COALESCE_US` | `promote-or-internalize` | 2 |
| `HF2Q_AUTH_TOKEN` | `appropriate-env` | 7 |
| `HF2Q_B9_FORCE_SEQUENTIAL` | `development-only` | 2 |
| `HF2Q_BARRIER_NS` | `development-only` | 2 |
| `HF2Q_BATCHED_ATTNPRE` | `development-only` | 4 |
| `HF2Q_BATCHED_BENCH` | `development-only` | 3 |
| `HF2Q_BATCHED_BODY` | `development-only` | 11 |
| `HF2Q_BATCHED_DUMP` | `development-only` | 7 |
| `HF2Q_BATCHED_FLASH` | `development-only` | 6 |
| `HF2Q_BATCHED_KVENC` | `development-only` | 4 |
| `HF2Q_BATCHED_LAYER_SCAN` | `development-only` | 4 |
| `HF2Q_BATCHED_PREFILL` | `documented-escape` | 9 |
| `HF2Q_BATCHED_WSUM` | `development-only` | 2 |
| `HF2Q_BENCH_CONC` | `development-only` | 2 |
| `HF2Q_BENCH_N` | `development-only` | 3 |
| `HF2Q_BENCH_PROMPT_LEN` | `development-only` | 2 |
| `HF2Q_BENCH_REPEAT` | `development-only` | 2 |
| `HF2Q_BENCH_SETTLE_MS` | `development-only` | 2 |
| `HF2Q_BENCH_TOKENS` | `development-only` | 2 |
| `HF2Q_BENCH_TOKIO_THREADS` | `development-only` | 2 |
| `HF2Q_BINARY_NOT_RELEASE` | `development-only` | 6 |
| `HF2Q_BISECT_ALEN` | `development-only` | 2 |
| `HF2Q_BISECT_ASEED` | `development-only` | 2 |
| `HF2Q_BISECT_BLEN` | `development-only` | 2 |
| `HF2Q_BUILD_GIT_SHA` | `development-only` | 2 |
| `HF2Q_BYTE_EQUIV_E2E` | `development-only` | 40 |
| `HF2Q_BYTE_EQUIV_E2E_GGUF` | `development-only` | 30 |
| `HF2Q_C2C_E2E` | `development-only` | 14 |
| `HF2Q_C2C_E2E_GGUF` | `development-only` | 7 |
| `HF2Q_CACHE_DIR` | `appropriate-env` | 15 |
| `HF2Q_CACHE_DIR_ENV` | `development-only` | 2 |
| `HF2Q_CB_THROUGHPUT_MODEL` | `development-only` | 1 |
| `HF2Q_CHUNK_SCAN_PREFILL` | `development-only` | 20 |
| `HF2Q_CKSUM_PERSEQ` | `development-only` | 2 |
| `HF2Q_COMPLETE` | `development-only` | 11 |
| `HF2Q_COMPLETER_PATH_PLACEHOLDER__` | `development-only` | 3 |
| `HF2Q_COMPLETION_STARTUP_FILE` | `appropriate-env` | 1 |
| `HF2Q_COMPLETION_ZDOTDIR_PROBE` | `development-only` | 3 |
| `HF2Q_CROSS_SLOT_ADMIT` | `promote-or-internalize` | 9 |
| `HF2Q_DEBUG_TOKENIZE_ONLY` | `development-only` | 5 |
| `HF2Q_DEBUG_TQ_RMS` | `development-only` | 20 |
| `HF2Q_DECODE_CATSPLIT` | `development-only` | 6 |
| `HF2Q_DECODE_CB_CHUNKS` | `development-only` | 3 |
| `HF2Q_DECODE_EMIT` | `development-only` | 9 |
| `HF2Q_DECODE_EMIT_TOKENS` | `development-only` | 3 |
| `HF2Q_DECODE_INPUT_TOKENS` | `development-only` | 5 |
| `HF2Q_DECODE_MVN` | `promote-or-internalize` | 4 |
| `HF2Q_DECODE_MV_EXT` | `promote-or-internalize` | 9 |
| `HF2Q_DECODE_PROFILE` | `development-only` | 8 |
| `HF2Q_DECODE_SPLIT_CB_AT_LAYER` | `development-only` | 2 |
| `HF2Q_DECODE_TRACE` | `development-only` | 3 |
| `HF2Q_DEEPSEEK4_AGENTIC_CONTRACT_RECEIPT` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_AGENTIC_PROMPT_CONTRACT` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_AGENTIC_REQUEST_JSON` | `development-only` | 2 |
| `HF2Q_DEEPSEEK4_COHORT_BENCH_PAIRS` | `development-only` | 2 |
| `HF2Q_DEEPSEEK4_COHORT_RECEIPT` | `development-only` | 2 |
| `HF2Q_DEEPSEEK4_DECODE_COHORT_PHASE_DIR` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_DECODE_COHORT_RECEIPT` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_DECODE_COHORT_RUN_UUID` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_GGUF` | `development-only` | 7 |
| `HF2Q_DEEPSEEK4_PHASE` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_RENDERED_PROMPT` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_REQUEST_JSON` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_TOKENIZER` | `development-only` | 8 |
| `HF2Q_DEEPSEEK4_TOKENIZER_JSON` | `development-only` | 2 |
| `HF2Q_DEEPSEEK_BATCHED_ATTN_CORES` | `development-only` | 10 |
| `HF2Q_DEEPSEEK_COMPRESSED_STAGE_PROFILE` | `development-only` | 4 |
| `HF2Q_DEEPSEEK_DUMP_ATTENTION_DIR` | `development-only` | 6 |
| `HF2Q_DEEPSEEK_DUMP_LAYER_DIR` | `development-only` | 5 |
| `HF2Q_DEEPSEEK_ENCODER_STAGES` | `development-only` | 4 |
| `HF2Q_DEEPSEEK_GRAPH_DIAG` | `development-only` | 3 |
| `HF2Q_DEEPSEEK_GRAPH_LAYERS_PER_CB` | `development-only` | 5 |
| `HF2Q_DEEPSEEK_GRAPH_REORDER` | `development-only` | 5 |
| `HF2Q_DEEPSEEK_JOINT_DECODE` | `documented-escape` | 2 |
| `HF2Q_DEEPSEEK_JOINT_DECODE_MAX_ROWS` | `development-only` | 1 |
| `HF2Q_DEEPSEEK_LAYERS_PER_CB` | `development-only` | 5 |
| `HF2Q_DEEPSEEK_MMAP_WEIGHTS` | `development-only` | 3 |
| `HF2Q_DEEPSEEK_PREFILL_TIMING` | `development-only` | 3 |
| `HF2Q_DEEPSEEK_PREFILL_WINDOWS` | `development-only` | 6 |
| `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC` | `development-only` | 10 |
| `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC_DEFAULT` | `development-only` | 5 |
| `HF2Q_DEEPSEEK_SEQUENTIAL_PREFILL` | `development-only` | 1 |
| `HF2Q_DEEPSEEK_SKIP_WARMUP` | `development-only` | 1 |
| `HF2Q_DEEPSEEK_SLOT_DECODE_QUANTUM` | `development-only` | 1 |
| `HF2Q_DEEPSEEK_STAGE_PROFILE` | `development-only` | 5 |
| `HF2Q_DEEPSEEK_TAIL_ROWS_STEP` | `development-only` | 4 |
| `HF2Q_DENSE_Q_ARENA_RESET` | `development-only` | 6 |
| `HF2Q_DENSE_Q_LEGACY` | `development-only` | 5 |
| `HF2Q_DFLASH_ACCEPT` | `development-only` | 2 |
| `HF2Q_DFLASH_BATCH_ARGMAX` | `development-only` | 2 |
| `HF2Q_DFLASH_BLOCK_SIZE` | `development-only` | 8 |
| `HF2Q_DFLASH_DRAFTER_DUMP` | `development-only` | 1 |
| `HF2Q_DFLASH_DRAFTER_PATH` | `development-only` | 9 |
| `HF2Q_DFLASH_HIDDEN_DEBUG` | `development-only` | 1 |
| `HF2Q_DFLASH_KV_CPU` | `development-only` | 1 |
| `HF2Q_DFLASH_PROFILE` | `development-only` | 6 |
| `HF2Q_DFLASH_SDPA_CPU` | `development-only` | 2 |
| `HF2Q_DFLASH_XLEN_BF16` | `development-only` | 1 |
| `HF2Q_DFLASH_XLEN_D512_OFF` | `development-only` | 2 |
| `HF2Q_DFLASH_XLEN_DEBUG` | `development-only` | 4 |
| `HF2Q_DFLASH_XLEN_SDPA` | `development-only` | 68 |
| `HF2Q_DISP_PROFILE` | `development-only` | 3 |
| `HF2Q_DUAL_BUFFER` | `development-only` | 4 |
| `HF2Q_DUMP_ALL_CACHE` | `development-only` | 10 |
| `HF2Q_DUMP_BOUNDARY` | `development-only` | 3 |
| `HF2Q_DUMP_CB_COUNT` | `development-only` | 2 |
| `HF2Q_DUMP_COUNTERS` | `development-only` | 7 |
| `HF2Q_DUMP_DIR` | `development-only` | 6 |
| `HF2Q_DUMP_FA_BF16` | `development-only` | 2 |
| `HF2Q_DUMP_FPRINT` | `development-only` | 2 |
| `HF2Q_DUMP_LAYER` | `development-only` | 32 |
| `HF2Q_DUMP_LAYERS` | `development-only` | 7 |
| `HF2Q_DUMP_LAYERS_LIST` | `development-only` | 3 |
| `HF2Q_DUMP_LAYER_ACTIVATIONS` | `development-only` | 5 |
| `HF2Q_DUMP_LAYER_ALL` | `development-only` | 2 |
| `HF2Q_DUMP_LAYER_DETAIL` | `development-only` | 3 |
| `HF2Q_DUMP_LAYER_N` | `development-only` | 5 |
| `HF2Q_DUMP_LOGITS` | `development-only` | 6 |
| `HF2Q_DUMP_NORM_WEIGHT` | `development-only` | 4 |
| `HF2Q_DUMP_PRE_QUANT` | `development-only` | 9 |
| `HF2Q_DUMP_PRE_QUANT_LAYERS` | `development-only` | 3 |
| `HF2Q_DUMP_PRE_QUANT_POSITIONS` | `development-only` | 3 |
| `HF2Q_DUMP_PROMPT_TOKENS` | `development-only` | 6 |
| `HF2Q_DUMP_RENDERED_PROMPT` | `development-only` | 10 |
| `HF2Q_DUMP_RUN_ID` | `development-only` | 2 |
| `HF2Q_DUMP_RUN_NAME` | `development-only` | 9 |
| `HF2Q_DUMP_SDPA_MAX_POS` | `development-only` | 7 |
| `HF2Q_DUMP_SLIDING_LAYER_0` | `development-only` | 9 |
| `HF2Q_DUMP_TQ_STATE` | `development-only` | 4 |
| `HF2Q_EAGLE3_DRAFTER_PATH` | `development-only` | 5 |
| `HF2Q_EAGLE3_EQUIVALENCE_REPETITIONS` | `development-only` | 3 |
| `HF2Q_EAGLE3_TOP_K` | `development-only` | 2 |
| `HF2Q_EAGLE3_TREE_BUDGET` | `development-only` | 2 |
| `HF2Q_EAGLE3_TREE_MAX_DEPTH` | `development-only` | 2 |
| `HF2Q_EMIT_NLL` | `development-only` | 3 |
| `HF2Q_ENCODER_SESSION` | `promote-or-internalize` | 18 |
| `HF2Q_F16_KV` | `development-only` | 17 |
| `HF2Q_F16_SHADOW` | `development-only` | 8 |
| `HF2Q_F32_MATVEC` | `development-only` | 2 |
| `HF2Q_FALSIFIER_EVICT` | `development-only` | 2 |
| `HF2Q_FALSIFIER_STAGGER_MS` | `development-only` | 3 |
| `HF2Q_FA_F16` | `development-only` | 13 |
| `HF2Q_FA_LAYER_CKSUM` | `development-only` | 1 |
| `HF2Q_FA_PEER_PORT` | `development-only` | 7 |
| `HF2Q_FA_PEER_PORT_NWG32` | `development-only` | 3 |
| `HF2Q_FA_TRACE` | `development-only` | 2 |
| `HF2Q_FFN_POOLED_LEGACY` | `development-only` | 2 |
| `HF2Q_FFN_SPLIT` | `development-only` | 6 |
| `HF2Q_FFN_TERMINAL_K_BATCH` | `promote-or-internalize` | 2 |
| `HF2Q_FISH_COMPLETIONS_DIR` | `appropriate-env` | 2 |
| `HF2Q_FORCE_DENSE_SDPA_ON_TQ_KV` | `development-only` | 1 |
| `HF2Q_FULL_F16_KV` | `development-only` | 55 |
| `HF2Q_FUSED_END_OF_LAYER` | `development-only` | 7 |
| `HF2Q_FUSED_GATE_UP_SILU` | `development-only` | 4 |
| `HF2Q_FUSED_MOE_GATE_UP_MM_ID` | `development-only` | 3 |
| `HF2Q_FUSED_MOE_WSUM_END_LAYER_V2` | `development-only` | 5 |
| `HF2Q_FUSED_QKVG` | `development-only` | 5 |
| `HF2Q_FUSED_TRIPLE_NORM` | `development-only` | 3 |
| `HF2Q_FUSE_LMHEAD` | `development-only` | 5 |
| `HF2Q_G4_TREE_VERIFY_NAN_DEBUG` | `development-only` | 2 |
| `HF2Q_GEMMA4_31B_DRAFTER` | `development-only` | 5 |
| `HF2Q_GEMMA4_31B_GGUF` | `development-only` | 8 |
| `HF2Q_GEMMA4_EAGLE3_MIN_ACCEPT` | `development-only` | 2 |
| `HF2Q_GEMMA4_GGUF` | `development-only` | 11 |
| `HF2Q_GEMMA_COLD_GS3_REPEATS` | `development-only` | 2 |
| `HF2Q_GEMMA_COLD_TURN0_PROMPTS` | `development-only` | 5 |
| `HF2Q_GEMMA_COLD_TURN0_RUN` | `development-only` | 1 |
| `HF2Q_GEMMA_DECODE_DENSE` | `development-only` | 11 |
| `HF2Q_GEMMA_DECODE_ATTN_NE` | `development-only` | 5 |
| `HF2Q_GEMMA_FULL_INPUT_CHECKPOINT` | `documented-escape` | 4 |
| `HF2Q_GEMMA_JOINT_COLD_PREFILL` | `documented-escape` | 4 |
| `HF2Q_GEMMA_JOINT_WARM_PREFILL` | `documented-escape` | 3 |
| `HF2Q_GEMMA_N8_EXPECTED_KV_REGIME` | `development-only` | 3 |
| `HF2Q_GEMMA_N8_PARITY_MAX_TOKENS` | `development-only` | 1 |
| `HF2Q_GEMMA_N8_PARITY_ROUNDS` | `development-only` | 1 |
| `HF2Q_GEMMA_N8_PREFILL_REPEATS` | `development-only` | 1 |
| `HF2Q_GEMMA_N8_RESUME_REPEATS` | `development-only` | 1 |
| `HF2Q_GEMMA_PREFILL_CATSPLIT` | `development-only` | 3 |
| `HF2Q_GEMMA_ROUTING_OVERLAP` | `development-only` | 5 |
| `HF2Q_GEMMA_WARM_TILED_ATTENTION` | `documented-escape` | 7 |
| `HF2Q_GLOBAL_FA` | `development-only` | 7 |
| `HF2Q_GPU_BUSY` | `development-only` | 5 |
| `HF2Q_GPU_SAMPLE` | `development-only` | 4 |
| `HF2Q_GQA_EXPAND_LEGACY` | `development-only` | 5 |
| `HF2Q_GRAPH_OPT` | `development-only` | 6 |
| `HF2Q_GRAPH_OPT_PREFILL` | `development-only` | 3 |
| `HF2Q_GROUP_STATS` | `development-only` | 1 |
| `HF2Q_HB_DUAL_LEGACY` | `development-only` | 3 |
| `HF2Q_HOST_PHASES` | `development-only` | 6 |
| `HF2Q_HYBRID_KV` | `promote-or-internalize` | 122 |
| `HF2Q_HYBRID_NWG` | `development-only` | 2 |
| `HF2Q_IMATRIX_CPU_REF` | `development-only` | 4 |
| `HF2Q_IMATRIX_METAL_REF` | `development-only` | 4 |
| `HF2Q_IMATRIX_REAL_REF` | `development-only` | 4 |
| `HF2Q_ITERGA_KRUNS` | `development-only` | 2 |
| `HF2Q_ITERGA_N8` | `development-only` | 2 |
| `HF2Q_KV_DUAL_LEGACY` | `development-only` | 3 |
| `HF2Q_KV_LCP_CAPACITY` | `promote-or-internalize` | 1 |
| `HF2Q_KV_LCP_CHUNKED_PREFILL` | `promote-or-internalize` | 11 |
| `HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE` | `development-only` | 2 |
| `HF2Q_KV_LCP_DISABLE_MID_STORE` | `development-only` | 4 |
| `HF2Q_KV_LCP_LONG_RESUME` | `promote-or-internalize` | 5 |
| `HF2Q_KV_LCP_RESUME` | `promote-or-internalize` | 52 |
| `HF2Q_KV_LCP_RESUME_CAPACITY` | `removed-guard` | 1 |
| `HF2Q_KV_PERSIST` | `promote-or-internalize` | 19 |
| `HF2Q_KV_PERSIST_PATH` | `promote-or-internalize` | 15 |
| `HF2Q_KV_PERSIST_PATH_ENV` | `development-only` | 9 |
| `HF2Q_KV_WRITER_CAPACITY` | `development-only` | 1 |
| `HF2Q_LAYER_POLICY` | `development-only` | 20 |
| `HF2Q_LEGACY_PER_LAYER_CB` | `development-only` | 5 |
| `HF2Q_LEGACY_TQ_SDPA` | `development-only` | 1 |
| `HF2Q_LMHEAD_COMPARE` | `development-only` | 3 |
| `HF2Q_LMHEAD_Q6K` | `promote-or-internalize` | 9 |
| `HF2Q_LMHEAD_Q8` | `documented-escape` | 9 |
| `HF2Q_LMHEAD_RERANK` | `development-only` | 8 |
| `HF2Q_LOAD_TIMING` | `development-only` | 5 |
| `HF2Q_MAX_SLOTS` | `removed-guard` | 1 |
| `HF2Q_METAL_CAPTURE` | `development-only` | 5 |
| `HF2Q_METAL_CAPTURE_LAYERS` | `development-only` | 3 |
| `HF2Q_MLX_KERNEL_PROFILE` | `development-only` | 4 |
| `HF2Q_MLX_PROFILE` | `development-only` | 4 |
| `HF2Q_MLX_TIMING` | `development-only` | 3 |
| `HF2Q_MM_ID_ROUTING_THRESHOLD` | `development-only` | 3 |
| `HF2Q_MTP_PHASE_PROFILE` | `development-only` | 1 |
| `HF2Q_MTP_PROFILE` | `development-only` | 5 |
| `HF2Q_MVN_BARRIER_TRACE` | `development-only` | 1 |
| `HF2Q_MVN_ENCODE_TRACE` | `development-only` | 3 |
| `HF2Q_NETWORK_TESTS` | `development-only` | 7 |
| `HF2Q_NLL` | `development-only` | 8 |
| `HF2Q_NO_COMPLETION_INSTALL` | `appropriate-env` | 5 |
| `HF2Q_NO_FA` | `development-only` | 16 |
| `HF2Q_NO_FUSED_STAGE_AB_VEC` | `development-only` | 3 |
| `HF2Q_NO_RESIDENCY` | `development-only` | 7 |
| `HF2Q_NO_VEC_SMALL_PATH` | `development-only` | 2 |
| `HF2Q_PARALLEL_ENCODE` | `development-only` | 24 |
| `HF2Q_PARALLEL_ENCODE_KV_THRESHOLD` | `development-only` | 2 |
| `HF2Q_PARALLEL_PROFILE` | `development-only` | 2 |
| `HF2Q_PARTIAL_CHAIN_LEGACY` | `development-only` | 3 |
| `HF2Q_PARTIAL_CHAIN_N` | `development-only` | 6 |
| `HF2Q_PER_LAYER_DISP` | `development-only` | 15 |
| `HF2Q_PER_LAYER_GPU_TIME` | `development-only` | 2 |
| `HF2Q_PER_LAYER_PHASE_GPU_TIME` | `development-only` | 4 |
| `HF2Q_PIPELINE_PREWARM` | `development-only` | 3 |
| `HF2Q_PIPELINE_PREWARM_LOG` | `development-only` | 1 |
| `HF2Q_POOL_BUDGET_BYTES` | `promote-or-internalize` | 3 |
| `HF2Q_PREFILL_CROSS_SLOT` | `promote-or-internalize` | 1 |
| `HF2Q_PREFILL_DUMP` | `development-only` | 4 |
| `HF2Q_PREFILL_SLOT_BATCHED` | `promote-or-internalize` | 4 |
| `HF2Q_PREFILL_TIMING` | `development-only` | 2 |
| `HF2Q_PROFILE_BUCKETS` | `development-only` | 10 |
| `HF2Q_PROFILE_DENSE_Q_SPLIT_COMMITS` | `development-only` | 5 |
| `HF2Q_PROFILE_FA` | `development-only` | 5 |
| `HF2Q_PROFILE_GPU_TS` | `development-only` | 4 |
| `HF2Q_PROFILE_LAYERS` | `development-only` | 5 |
| `HF2Q_PROFILE_MM` | `development-only` | 9 |
| `HF2Q_PROFILE_MOE` | `development-only` | 7 |
| `HF2Q_PROFILE_MOE_POST` | `development-only` | 2 |
| `HF2Q_PROFILE_RESIDENCY_ABORT` | `development-only` | 1 |
| `HF2Q_PROFILE_SYNC` | `development-only` | 2 |
| `HF2Q_PROFILE_W5B17` | `development-only` | 5 |
| `HF2Q_PROFILE_W5B22` | `development-only` | 5 |
| `HF2Q_PROFILE_W5B8` | `development-only` | 19 |
| `HF2Q_Q5_0_TENSOR_MM` | `development-only` | 7 |
| `HF2Q_Q5_0_TENSOR_MM_TEST_EXPECT` | `development-only` | 1 |
| `HF2Q_Q6K_ID_MV_NR2` | `development-only` | 2 |
| `HF2Q_Q6K_MV_NR2` | `development-only` | 3 |
| `HF2Q_Q8_0_ID_MV_NR2` | `development-only` | 1 |
| `HF2Q_QKV_SPLIT_LEGACY` | `development-only` | 2 |
| `HF2Q_QWEN35_DROP_MTP` | `development-only` | 8 |
| `HF2Q_QWEN35_E2E_GGUF` | `development-only` | 7 |
| `HF2Q_QWEN35_FA_LEGACY` | `development-only` | 2 |
| `HF2Q_QWEN35_PREFILL_SWEEP` | `development-only` | 9 |
| `HF2Q_QWEN35_PREFILL_SWEEP_COMPARE_FULL_LAST` | `development-only` | 1 |
| `HF2Q_QWEN35_PREFILL_SWEEP_FULL_LOGITS` | `development-only` | 1 |
| `HF2Q_QWEN35_PREFILL_SWEEP_TRIALS` | `development-only` | 2 |
| `HF2Q_QWEN35_PREFILL_SWEEP_WARMUPS` | `development-only` | 2 |
| `HF2Q_QWEN35_TOKENIZER` | `development-only` | 3 |
| `HF2Q_QWEN36_AUTOREG` | `development-only` | 2 |
| `HF2Q_QWEN36_GGUF` | `development-only` | 1 |
| `HF2Q_QWEN36_WATCHDOG_FIXTURE_MODEL` | `development-only` | 2 |
| `HF2Q_QWEN36_WATCHDOG_FIXTURE_OUTPUT` | `development-only` | 1 |
| `HF2Q_QWEN36_WATCHDOG_SHORT_FIXTURE_OUTPUT` | `development-only` | 1 |
| `HF2Q_QWEN38_GGUF` | `development-only` | 1 |
| `HF2Q_QWEN3VL_E2E` | `development-only` | 1 |
| `HF2Q_QWEN3VL_LM_LOAD` | `development-only` | 9 |
| `HF2Q_QWEN_ATTN_ROWS` | `documented-escape` | 5 |
| `HF2Q_QWEN_GQA_Q2` | `documented-escape` | 2 |
| `HF2Q_QWEN_ANCHOR_BYTES_PER_SLOT_MAX` | `documented-escape` | 4 |
| `HF2Q_QWEN_ANCHOR_DEPTH` | `documented-escape` | 5 |
| `HF2Q_QWEN_JOINT_DECODE` | `documented-escape` | 1 |
| `HF2Q_QWEN_JOINT_DECODE_MAX_ROWS` | `development-only` | 1 |
| `HF2Q_QWEN_JOINT_FFN_SHARED` | `documented-escape` | 8 |
| `HF2Q_QWEN_JOINT_MMA` | `development-only` | 3 |
| `HF2Q_QWEN_JOINT_MTP` | `development-only` | 2 |
| `HF2Q_QWEN_JOINT_POOLED_OUTPUTS` | `development-only` | 3 |
| `HF2Q_QWEN_JOINT_Q1_OUT` | `development-only` | 1 |
| `HF2Q_QWEN_JOINT_Q1_PROMPTS` | `development-only` | 1 |
| `HF2Q_QWEN_JOINT_Q1_STEPS` | `development-only` | 1 |
| `HF2Q_QWEN_LA_GPU_RESTORE` | `documented-escape` | 4 |
| `HF2Q_QWEN_MOE_BATCHED` | `documented-escape` | 14 |
| `HF2Q_QWEN_MOE_Q1_FORCED` | `development-only` | 2 |
| `HF2Q_QWEN_MOE_Q1_OUT` | `development-only` | 4 |
| `HF2Q_QWEN_MOE_Q1_PROMPTS` | `development-only` | 4 |
| `HF2Q_QWEN_MOE_Q1_STEPS` | `development-only` | 2 |
| `HF2Q_QWEN_MOE_STAGED_M8` | `documented-escape` | 1 |
| `HF2Q_QWEN_RESUME_ATTN_FAST` | `documented-escape` | 7 |
| `HF2Q_QWEN_RESUME_FAST_PROMPTS` | `development-only` | 3 |
| `HF2Q_QWEN_SPECULATION` | `promote-or-internalize` | 6 |
| `HF2Q_QWEN_VISION_MMPROJ` | `development-only` | 3 |
| `HF2Q_RERANK_PROFILE` | `development-only` | 5 |
| `HF2Q_RMS_NORM_V2` | `development-only` | 3 |
| `HF2Q_RUNTIME_SCHEMA` | `development-only` | 2 |
| `HF2Q_S019_CKSUM` | `development-only` | 11 |
| `HF2Q_S019_MAXTOK` | `development-only` | 2 |
| `HF2Q_S019_NO_REMASK` | `development-only` | 4 |
| `HF2Q_S019_NSEQ` | `development-only` | 2 |
| `HF2Q_S019_PROMPT_LEN` | `development-only` | 2 |
| `HF2Q_S019_REPEATS` | `development-only` | 2 |
| `HF2Q_S2C` | `development-only` | 1 |
| `HF2Q_SCALE_FORMULA` | `development-only` | 9 |
| `HF2Q_SCHEDULER` | `removed-guard` | 1 |
| `HF2Q_SERVE_BATCHED` | `promote-or-internalize` | 13 |
| `HF2Q_SERVE_BATCHED_` | `development-only` | 12 |
| `HF2Q_SERVE_BATCHED_PREFILL` | `promote-or-internalize` | 11 |
| `HF2Q_SETUP_ABORT_AT` | `development-only` | 2 |
| `HF2Q_SETUP_CRASH_CHILD` | `development-only` | 1 |
| `HF2Q_SETUP_CRASH_ROOT` | `development-only` | 2 |
| `HF2Q_SETUP_UMASK_CHILD` | `development-only` | 1 |
| `HF2Q_SETUP_UMASK_ROOT` | `development-only` | 2 |
| `HF2Q_SKIP_ATTN_QKV` | `development-only` | 2 |
| `HF2Q_SKIP_DENSE_MLP` | `development-only` | 2 |
| `HF2Q_SKIP_EMBED` | `development-only` | 1 |
| `HF2Q_SKIP_END_OF_LAYER` | `development-only` | 4 |
| `HF2Q_SKIP_END_OF_LAYER_FINAL` | `development-only` | 2 |
| `HF2Q_SKIP_HEAD_NORM_ROPE` | `development-only` | 2 |
| `HF2Q_SKIP_MMPROJ_LOAD` | `development-only` | 4 |
| `HF2Q_SKIP_MOE_EXPERTS` | `development-only` | 2 |
| `HF2Q_SKIP_MOE_SWIGLU` | `development-only` | 2 |
| `HF2Q_SKIP_O_PROJ` | `development-only` | 2 |
| `HF2Q_SKIP_POST_ATTN_NORM` | `development-only` | 2 |
| `HF2Q_SKIP_ROUTING` | `development-only` | 2 |
| `HF2Q_SKIP_TQ_ENCODE` | `development-only` | 5 |
| `HF2Q_SKIP_TQ_SDPA` | `development-only` | 4 |
| `HF2Q_SKIP_VIT_WARMUP` | `development-only` | 4 |
| `HF2Q_SKIP_V_NORM` | `development-only` | 2 |
| `HF2Q_SKIP_WEIGHTED_SUM` | `development-only` | 2 |
| `HF2Q_SPEC_DECODE` | `development-only` | 54 |
| `HF2Q_SPEC_DECODE_K` | `development-only` | 16 |
| `HF2Q_SPEC_DECODE_K1` | `development-only` | 10 |
| `HF2Q_SPEC_DECODE_K1_NO_AMORT` | `development-only` | 3 |
| `HF2Q_SPEC_DECODE_K1_TRACE` | `development-only` | 2 |
| `HF2Q_SPEC_DECODE_K1_TWO_CALLS` | `development-only` | 2 |
| `HF2Q_SPEC_DECODE_KN_HIDDEN_ROW_CAP` | `development-only` | 3 |
| `HF2Q_SPEC_DFLASH` | `development-only` | 30 |
| `HF2Q_SPEC_DFLASH_BLOCK_SIZE` | `development-only` | 1 |
| `HF2Q_SPEC_DFLASH_PHASE` | `development-only` | 1 |
| `HF2Q_SPEC_EAGLE3` | `development-only` | 25 |
| `HF2Q_SPEC_NGRAM` | `development-only` | 22 |
| `HF2Q_SPEC_NGRAM_K` | `development-only` | 4 |
| `HF2Q_SPEC_NGRAM_MAX` | `development-only` | 3 |
| `HF2Q_SPEC_NGRAM_MIN` | `development-only` | 3 |
| `HF2Q_SPEC_NGRAM_PROFILE` | `development-only` | 4 |
| `HF2Q_SPLIT_POSTATTN_NORM` | `development-only` | 1 |
| `HF2Q_SPLIT_POSTFF_NORMADD` | `development-only` | 2 |
| `HF2Q_SPLIT_POSTFF_NORMADDSCALAR` | `development-only` | 1 |
| `HF2Q_SPLIT_TIMING` | `development-only` | 4 |
| `HF2Q_STEP_PROFILE` | `development-only` | 1 |
| `HF2Q_SYNC_PER_LAYER` | `development-only` | 5 |
| `HF2Q_TEST_DESCENDANT_PID_FILE` | `development-only` | 2 |
| `HF2Q_TEST_HELPER_PID_FILE` | `development-only` | 4 |
| `HF2Q_TEST_PROMPT` | `development-only` | 1 |
| `HF2Q_TEST_QWEN38_EXPECT_MV_EXT` | `development-only` | 2 |
| `HF2Q_TEST_QWEN38_EXPECT_Q4K_MVN` | `development-only` | 2 |
| `HF2Q_TEST_QWEN38_EXPECT_Q5_K_M` | `development-only` | 2 |
| `HF2Q_TEST_QWEN38_GGUF` | `development-only` | 3 |
| `HF2Q_TOKENIZER_GGUF_EMBEDDED` | `development-only` | 4 |
| `HF2Q_TQ_CODEBOOK_BITS` | `development-only` | 44 |
| `HF2Q_TQ_FAST_FUSED_KV` | `development-only` | 3 |
| `HF2Q_TQ_FUSE_FWHT_PRE` | `development-only` | 2 |
| `HF2Q_TQ_HB_OUT_FUSED` | `development-only` | 1 |
| `HF2Q_TQ_KV` | `promote-or-internalize` | 26 |
| `HF2Q_UNSAFE_EXPERIMENTS` | `development-only` | 37 |
| `HF2Q_USE_DENSE` | `development-only` | 79 |
| `HF2Q_VERIFIER_NBENCH` | `development-only` | 2 |
| `HF2Q_VIT_DUMP` | `development-only` | 23 |
| `HF2Q_VIT_DUMP_DTYPE_AUDIT` | `development-only` | 6 |
| `HF2Q_VIT_F32_ATTENTION` | `development-only` | 10 |
| `HF2Q_W5B4_DIVERGENCE` | `development-only` | 5 |
| `HF2Q_ZDOTDIR_V1` | `development-only` | 2 |
| `HF2Q_ZSH_COMPLETIONS_DIR` | `appropriate-env` | 2 |
| `HF2Q_ZSH_STARTUP_DIR` | `appropriate-env` | 1 |
