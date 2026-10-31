# ADR-063 S3 — DeepSeek-V4 joint decode (phase 1)

Status: **default on** since 2026-10-30. `HF2Q_DEEPSEEK_JOINT_DECODE=0` restores the serial path. Exactness and the four-agent A/B passed on hardware; see "Measured result".

Closure evidence (2026-10-31):
- **Task quality against stock: Pass,** 20/24 vs 20/24 with `ExactTranscripts` (`E/quality-ds4-joint-vs-stock-20261030T230921Z`). The reference is the same binary with `HF2Q_DEEPSEEK_JOINT_DECODE=0`.
- **Cancel under joint decode: CancelSmokePass** (`E/s3-cancel-v1/deepseek4-20261031T060207Z`). Closing one stream mid-decode left the other three agents finishing with CalibrationPass. All 60 scheduler turns ran joint (width 2: 56, width 3: 3, width 4: 1; no serial lanes).
- **The peer is unavailable** for the four-agent DeepSeek-V4 comparison. It hits a Metal out-of-memory at 4×262144 f16 KV in both load modes tried (`E/peer-arms-v1/peer-deepseek4-*`).

## Problem

With four concurrent agents, each DeepSeek-V4 Flash request decodes at about 8.4 tok/s. The aggregate (~34 tok/s) equals single-stream speed.

- `decode_batch_deepseek4` gives each slot a serial quantum: 8 tokens for warm or streaming slots, 64 for a cold unary wave.
- The only joint path, the B=4 cohort, requires exactly four warm, unary, equal-position slots with no cold prefill. It was never selected in the coding workload.
- Even when selected, the cohort runs four independent one-row graphs, so weights are still read once per row.

## Decision

Phase 1 adds one layer-major forward per decode step over every eligible decoding slot (B = 2..4 by default; up to 8 with `HF2Q_DEEPSEEK_JOINT_DECODE_MAX_ROWS`), with:

- unequal positions allowed;
- streaming and unary slots mixed;
- per-slot sampling, grammar and stop logic unchanged.

Row-parallel work runs once with `m = B`. Work that touches a request's cache runs per row against that row's cache and position.

The joint path must be **bit-identical to serial decode**. It became the default on 2026-10-30, after the official-artifact proof and the four-agent A/B passed on hardware. The default width is 4 rows. Widths 5–8 need `HF2Q_DEEPSEEK_JOINT_DECODE_MAX_ROWS`; they pass the official-artifact bitwise test but have no serving measurement. The B=4 cohort is kept only as the switch-off path (`HF2Q_DEEPSEEK_JOINT_DECODE=0`). It is a strict subset of the joint path, so it can be removed once the serial A/B arm is no longer needed.

## Data flow per layer (`joint_decode.rs`, `joint_decode_core.rs`)

| Stage | Execution | Notes |
|---|---|---|
| Layer-0 embedding gather + HC expand | batched (B rows) | `encode_embedding_hyper_state(rows = B)` |
| HC prelude: RMS norm, `hc_attn_fn`, split-Sinkhorn, `hc_pre`, attention norm | batched | `encode_compressed_attention_prelude(..., RowMatmul::IndependentRows)`; the same dispatch sequence as the one-token uncompressed path |
| `q_a`, `kv`, `q_a_norm`, `q_b`, Q/KV head norms (Q8_0) | batched, `m = B` | one dispatch per weight |
| Tail RoPE for Q and KV, MXFP8 KV fake-quant | batched | positions buffer holds each row's own position; the kernel reads `positions[row]` |
| Window write | per row | the row's own `window_kv` slot from its `CacheStep` |
| Main compressor (ratio 4/128) | per row | `encode_main_compressor(rows = 1)` on a row view of `attn_norm` |
| Ratio-4 indexer + top-k | per row | `encode_ratio_four_indexer(rows = 1)`; the row's graft offset, positions and indices |
| Sparse attention (window, compressed, flash-gathered) | per row | the same kernel and parameters as the one-token path |
| Inverse RoPE, cast, grouped output-A, output-B, `hc_post` | batched | `grouped_output_a_batched` at `rows = B` |
| FFN: HC, router, hash/score routing, routed experts, shared expert, reduce, `hc_post`, GLP | batched | `forward_ffn_rows(rows = B, RowMatmul::IndependentRows)` |
| Output head (collapse, norm, vocabulary) | batched | `forward_logits([B, hc, hidden])`, as the B=4 cohort already does |
| Sampling | per row | `Deepseek4SlotState::prepare_tick` with each slot's own grammar, stops and events |

Layers are grouped two per command buffer, from the same `HF2Q_DEEPSEEK_LAYERS_PER_CB` helper the serial path uses. The pipeline is drained once per step.

Every cache publishes atomically through `publish_verifier_cohort_after_gate`, now generalised to slices and unequal positions. A submitted failure poisons every row's cache.

## What is batched vs per row

- **Batched**, one dispatch per weight for B rows:
  - HC mixers;
  - `q_a`, `q_b`, `kv`, grouped `output_a` and `output_b`;
  - FFN HC mixer, router and shared expert;
  - routed experts (per-token `mv_id` over `B × 6` routes);
  - output head and output HC;
  - all elementwise and norm kernels.
- **Per row:**
  - window write, compressor and indexer projections and recurrences, top-k selection, and sparse attention;
  - B=1 survivors, which use the serial `commit_generated_token`.

**Weights read once:** each batched product is one dispatch. On the default exact route, the Q8_0 matvec (`kernel_mul_mv_q8_0_f32_nr2`) processes one input row per threadgroup row (`r1 = tgpig.y`). Dispatch count, command-buffer encode and pipeline drains are therefore amortized, but DRAM weight reuse across rows relies on cache residency. Guaranteed read-once needs the kernel below.

## Exactness contract

Joint decode must match `forward_verifier_one` + `forward_logits` per row. This is the B=4 cohort contract (`real_artifact_decode_cohort_tests.rs`) applied to ragged rows:

- bitwise-equal F32 final state;
- bitwise-equal logits;
- byte-equal caches: window, compressed, indexer, and compressor state;
- equal cursor and poison state.

Greedy tokens are then identical by construction.

How the joint path holds the contract:

- **Block-quantized products at `m ≤ MM_ROUTING_THRESHOLD (8)`** take the default GGUF matvec route. That route's per-row arithmetic does not depend on `m`. `MAX_JOINT_DECODE_ROWS = 8` is the model's hard bound, and `row_matmul` fails closed above it.
- **Admission by storage type (fail closed).** At load, `joint_decode_routes::unproven_joint_route` lists every product joint decode issues with `m = B` and checks it against the proven set:
  - Q8_0, Q2_K and Q3_K for row products, the output head and routed experts;
  - F32/F16/BF16 only where `row_matmul` splits them per row (the router and every other prelude, epilogue or FFN product);
  - grouped output-A only for Q8_0.

  The output head goes through `raw_matmul` at `m = B` and is never split, so a float `output_hc_fn`/`output` keeps the artifact on serial decode. Any K-quant `mN` route, legacy quant, or grouped output-A other than Q8_0 does the same. `HF2Q_DECODE_MV_EXT=1` also disables joint decode. `Deepseek4Model::joint_decode_unavailable_reason` returns the reason, and both the worker and every joint transaction check it.
- **Dense float weights** (the F32 router, any F32/F16/BF16 matrix) switch kernels at `m > 1`, so `RowMatmul::IndependentRows` applies them one row at a time.
- **Routed experts** always use `ForceMv` (`quantized_matmul_id_ggml_mv`) in independent-rows mode. Grouped `mm_id` (`n_tokens > 32`) is not bit-identical.
- **Grouped output-A** at `rows > 1` runs the same `nr2` kernel per group with `m = rows`. This is pinned by a GPU test.
- **Row-independent kernels:** RMS norm (one threadgroup per row), HC split-Sinkhorn, `hc_pre`/`hc_post` (token = threadgroup row or simdgroup), tail RoPE (`positions[seq]`), MXFP8 and Hadamard (one threadgroup per row). These were checked by reading the mlx-native 0.15.15 shaders. The independent review read SwiGLU, MoE weighted reduce, hash/score routing and GLP, with the same per-row result.
- **Prefill and serial decode** pass `RowMatmul::Matrix`, so their dispatches are unchanged.

Non-exact measurement arm: `HF2Q_DECODE_MV_EXT=1` routes Q8_0 `m = 2..8` to `mul_mv_ext`, which reads the weights once but is not bit-identical. Joint decode therefore refuses it and stays serial; that arm can no longer be measured through joint decode.

### Tests

- `row_matmul_tests.rs` (Metal, not ignored):
  - Q8_0 `m = 4` equals four `m = 1` matvecs bitwise;
  - the F32 router split equals per-row matvecs;
  - Q8_0 grouped output-A at `rows = 4` equals the one-row path bitwise;
  - row-view bounds;
  - the dense-split predicate.
- `cache_tests.rs::joint_decode_publication_advances_unequal_positions_together` (Metal allocation only): atomic publication at unequal positions; stale cursor and position-length mismatch poison every row.
- `real_artifact_joint_decode_tests.rs::official_artifact_joint_decode_matches_serial_decode_bitwise` (`#[ignore]`, official GGUF):
  - 140 steps over eight lanes with prefixes 148/131/3/254/200/127/64/9 and ragged widths 8/4/3/5/2/6/7;
  - crosses ratio-4 and ratio-128 boundaries;
  - lanes stay inside their first window;
  - bitwise state, logits and cache every step;
  - checks that the artifact admits joint decode, width-1 and misaligned rejection, and a rejected commit gate poisoning only that step's rows.
  - Passed on 2026-10-30, first in the four-lane version (widths 4/3/2/2), then in this eight-lane version (59 s, source `fab4dc3b`).
- `joint_decode_routes.rs` (host only): the official type set is admitted; a float output head, unproven block types and non-Q8_0 grouped output-A fall back to serial.
- `engine/deepseek4_joint_decode.rs` (host only):
  - kill-switch parsing and default;
  - width parsing (default 4, explicit 2..8);
  - a requested switch on an unproven artifact selects serial;
  - joint and serial partition (ragged eligibility, cap, duplicates, fewer than two rows).

## Scheduler (`engine/deepseek4_joint_decode.rs`)

### Selection

A scheduler `Decode` or `Mixed` turn calls `decode_batch_deepseek4`. When the worker resolved a joint width at start (switch on, artifact and configuration admissible), the turn is partitioned:

- **Joint rows:** unique handles where the installed handle matches, the work is unfinished `Decode` (not parked), the client is still open, and `Deepseek4Session::joint_decode_ready()` holds: cursor equals the token ledger, live logits are present, and `plan_next_step()` succeeds (not poisoned, within capacity and graft bounds). Rows are sorted by slot and capped at the worker's joint width (default 4).
- **Fallback:** fewer than two eligible rows means the whole turn runs serially.
- **Serial leftovers:** parked completions, closed or ineligible slots, and overflow beyond the width run on the unchanged serial loop afterwards.

### Turn

- **Budget:** each row's existing effective quantum is computed (8 warm/streaming, up to 64 for a cold unary wave), and the minimum is taken. All rows advance in lockstep.
- **Each step:**
  1. Check for closed streams.
  2. Run `prepare_tick` per row. This handles the thinking budget, grammar, EOS, stop strings, `max_tokens` and single-tool terminal.
  3. Send all sampled rows through `commit_generated_tokens_joint`. A lone survivor uses the serial one-row commit.
  4. Install each row's logits view.

### Failure

- A worker-fatal error (supervisor unhealthy or command-buffer fault) fail-stops the turn with every row's reply attached.
- Any other error is recorded on every row of that transaction. Each row then fails, resets and replies exactly as a serial commit failure would.

### Settlement

Each row goes through `settle_deepseek4_decode_lane`, extracted unchanged from the serial loop and shared by both paths:

- scheduler tick accounting;
- cold-wave parking;
- cancellation recovery;
- error reply;
- completion delivery and anchor commit;
- slot reinstall.

### Prefill interleaving and TTFT

The existing `Mixed` step still runs the decode turn, then one prefill slice capped at 2 windows (256 tokens). A turn now costs at most 8 joint steps instead of `B × 8` serial tokens, so prefill slices come around sooner. No new timer is added.

## Kill switch

`HF2Q_DEEPSEEK_JOINT_DECODE`:

- `1`, `on` or `true`, or unset, uses the joint path (the default);
- `0`, `off` or `false` keeps the serial path plus the B=4 cohort;
- invalid values warn and fall back to the default (on).

`HF2Q_DEEPSEEK_JOINT_DECODE_MAX_ROWS` sets the joint width. The default is 4, and 2..8 are accepted. Widths 5–8 are covered only by the official-artifact bitwise test, not by any serving measurement. Invalid values warn and use 4.

**When it is read.** Both variables are read once, when the DeepSeek-V4 worker starts (`deepseek4_joint_decode_rows`), which is after model load and before the first request. Each decode turn re-checks admissibility. A routing variable can change later, for example when loading a Qwen3.8 model sets `HF2Q_DECODE_MV_EXT=1`. If that happens, the turn runs serially and a single warning is logged; requests do not fail.

**What keeps serial decode** even when requested:
- `MLX_UNRETAINED_REFS`;
- `HF2Q_DEEPSEEK_DUMP_LAYER_DIR` or `HF2Q_DEEPSEEK_DUMP_ATTENTION_DIR`;
- `HF2Q_DEEPSEEK_STAGE_PROFILE=1`;
- `HF2Q_DECODE_MV_EXT=1`;
- an artifact with unproven matrix storage.

**Logs.** One `DeepSeek-V4 joint decode configuration` line at worker start records `requested`, `max_rows`, `selected` and the `unavailable` reason. `DeepSeek-V4 joint decode engaged` logs once, and debug logs report each turn's width.

**Promotion:** done on 2026-10-30 after the official-artifact test and the A/B below passed.

## Known risks

- **Cache aliasing:** the rows' caches are distinct `&mut` borrows from distinct sessions. The staging `loaded.cache` is not used, except for the swap-in/swap-out of a lone survivor.
- **Positions:** each row's position comes from its own `plan_next_step()`, not from the batch. `positions[r]` drives RoPE, the window slot, the compressor block position and indexer start. The official test runs at ragged positions.
- **Grammar and tool calls:** sampling is unchanged per row, and streaming rows now receive their own events in joint mode. Interleaving token-by-token across rows does not mix any per-slot state.
- **Exactness assumptions not yet hardware-proven:**
  - SwiGLU, MoE weighted reduce, hash/score routing and GLP steering are row-independent by construction. The independent review read these shaders in mlx-native 0.15.15 and the GLP apply kernels. GLP under joint decode has not been tested on hardware.
  - serving at widths 5–8 (the eight-lane bitwise test passes; no four-agent measurement exists above width 4).
- **Decode scratch:** the high-water mark grows about B×. The A/B measured 197–401 MB at B=4 with roughly 0.8–2.3k-token prompts, against 62–95 MB serial. Ratio-4 indexer scratch grows further with context, and long-context (≥64k) peak memory is unmeasured. Most joint completions exceed the 256 MiB retained-scratch cap, so scratch is released and reallocated after the turn.
- **Code duplication:** the per-row cores intentionally duplicate the one-token decode branch, so the serial reference stays untouched. Dedupe the two after promotion.

## Reference: the peer's DSV4 multi-slot decode (read-only, HEAD 3c809f443, MIT, no code copied)

**Dense products run once over all decoding rows.** The DSV4 graph applies every projection to the `nt` rows of the ubatch with ordinary `mul_mat`:

- `wq_a`, `wq_b` and `wkv` (`src/models/deepseek4.cpp:920-941`);
- the compressor and indexer `wkv`/`wgate` (`:954-1041`);
- `wo_a` and `wo_b` (`:1220-1225`);
- the output head (`:1361`).

With one token per decoding slot, `ne11 = B`.

**Metal routing at `ne11 = B`.** `ggml/src/ggml-metal/ggml-metal-ops.cpp:2751-2794` routes F32/F16/BF16/Q4_0/Q5_0/**Q8_0**/… with `ne11 ∈ 2..8`, and Q2_K–Q6_K with `ne11 ∈ 4..8`, to the small-batch `mul_mv_ext` kernels (`kernels/mul_mv.metal:603-826`). These kernels dequantize each weight block once and apply it to all `r1ptg` rows.

- This is the same family as mlx-native's default-off `mul_mv_ext`.
- Its lane partition and reduction order differ from the `ne11 = 1` matvec.
- So the peer's multi-slot decode reads the weights once but does **not** promise per-row bit equality with single-slot decode.
- hf2q's contract is stricter. That is why phase 1 keeps the exact per-row `nr2` route, and why the exact read-once kernel below is specified rather than adopting `mul_mv_ext`.

**Per-row attention at different positions.**

- RoPE takes a per-token position tensor: `ggml_rope_ext(…, inp_pos, …)` at `deepseek4.cpp:931,941`, inverted at `:1213`.
- Each sequence owns a cache stream: `dsv4_stream_offset` (`src/llama-kv-cache-dsv4.cpp:51-60`) gives the per-seq offset, and `dsv4_comp_graph_n_stream` (`:200-208`) gives one graph stream per unique sequence.
- Attention visibility is a per-token mask built from each token's `n_visible` (`src/llama-graph.cpp:841-884`; the mask is `[n_kv, n_tokens/n_stream, 1, n_stream]`).
- The raw, ratio-128 and ratio-4 (CSA + lightning indexer) attentions are selected per layer at `deepseek4.cpp:1196-1209`; the indexer splits queries and weights per stream at `:638-657`.

hf2q reaches the same per-row isolation without padding or masks. Each row dispatches the existing one-token kernels against its own cache, which matches the per-request-pointer design that the oMLX V4.1 path also uses.

## mlx-native kernel needed for guaranteed weight read-once (not edited here)

**Shape.** `kernel_mul_mv_q8_0_f32_nr2_r1<R1>`, R1 ∈ 2..8:

- uses the same buffers as `kernel_mul_mv_q8_0_f32_nr2`: `src0` blocks, `src1 [batch][m][k]`, `dst [batch][m][n]`, `GgmlMatvecParams`, threadgroup memory;
- dispatch grid `(ceil(n/2), ceil(m/R1), batch)` with threads `(32, 4, 1)`;
- threadgroup memory `2 × R1 × 32 × 4` bytes.

**Semantics.** Each threadgroup loads `ax[row][ib]` once and applies it to R1 input rows. Per output `(r1, row)` it must perform exactly the `nr2` operation sequence:

- each lane computes `sumq = Σ_iq qs[iq] * yl[iq]` in the same `iq` order;
- `sumf += sumq * d` over `ib = ib0, ib0 + NSG·NQ, …`;
- then the same `simd_sum` and cross-simdgroup threadgroup-memory reduction order.

The output is then bitwise identical to `m = 1`.

**Route.** `plan_dense_auto_route` should select it for Q8_0 with `m ∈ 2..=8` when `dense_q8_0_mv_nr2` is set. It should be default-on with an `HF2Q_Q8_0_MV_R1=0` opt-out, and take precedence over `mv_ext` only when `mv_ext` is off.

**Tests.**

- Bitwise parity against `nr2` at `m = 1` for every m in 2..8, with n ∈ {1, 2, 63, 4096} and k ∈ {32, 256, 4096, 4128}.
- The batched variant (`batch = 8` groups, `m = B`) for grouped output-A.
- A capability and route receipt.
- Optional and lower value: the same treatment for Q2_K/Q3_K, used by the shared expert (about 0.4 GB/token).

**Expected effect [INFERENCE].** About 5.8 GB/token of Q8_0 attention weights would become about 5.8 GB per step.

## Measurement plan

The root runs all builds and tests. The arms share one binary and differ only in the environment:

| Arm | Environment |
|---|---|
| A (baseline) | `HF2Q_DEEPSEEK_JOINT_DECODE` unset |
| B (joint, exact) | `HF2Q_DEEPSEEK_JOINT_DECODE=1` |
| C (diagnostic, non-exact) | `HF2Q_DEEPSEEK_JOINT_DECODE=1 HF2Q_DECODE_MV_EXT=1`. The exactness gate now keeps this configuration serial, so arm C is no longer runnable through joint decode. |

Run each arm on two workloads:

1. **N=1 coding client, 7 turns, 262144 context / 1 slot.** Arms A and B must agree within noise (about ±1%), since joint never engages at N=1. Report per-request decode tok/s, server first semantic event, and client semantic TTFT.
2. **N=4 balanced coding series, 262144 context / 4 slots, KV budget 8 GiB.** 16 unary and 12 streaming requests per run, six cases with alternating arm order. Report:
   - per-agent decode tok/s (median and range);
   - aggregate tok/s while ≥2 slots decode;
   - client semantic TTFT and server first-semantic-event medians;
   - the count of `DeepSeek-V4 joint decode engaged` / joint turns and their width distribution (`RUST_LOG=hf2q::serve::api::engine_deepseek4::progress=debug`);
   - wall time per case.

**Gates:**

- exactness tests pass;
- arm B shows no N=1 regression;
- arm B per-agent N=4 tok/s is above arm A beyond noise;
- TTFT medians are not worse.

Arm C was meant to bound what the read-once kernel above would add. It is no longer available, because joint decode refuses `HF2Q_DECODE_MV_EXT=1`.

## Measured result (2026-10-30, M5 Max 128 GB)

**Exactness.** `official_artifact_joint_decode_matches_serial_decode_bitwise` passed on the DS4-0731 q2 artifact (sha256 `50df7e3d…`): 140 steps over four lanes at unequal positions, with state, logits and cache bitwise equal to serial decode.

**Four-agent C262144 coding workload.** One binary (source `a24f3f53`, sha256 `b5cd4c60…`), arms differing only in `HF2Q_DEEPSEEK_JOINT_DECODE`, order A B B A A B B A, each run under the Q5 calibration supervisor (`scripts/four_agent/q5_calibration.py`). Every run returned `CalibrationPass` and completed the four-agent protocol.

| arm | runs | makespan median (range) | per-request decode median | completion tok/s over makespan |
|---|---:|---|---:|---:|
| A serial | 4 | 141.7 s (136.2–148.6) | 8.12 | 26.8 |
| B joint | 4 | 105.0 s (104.2–111.5) | 10.98 | 35.7 |

Median makespan fell 25.9% (141.7 s → 105.0 s), which is 1.35× throughput from the medians. Across runs, the geometric-mean time ratio is 1.335× (33.5%; log-ratio Welch 95% CI 25.8–41.7%). The ranges do not overlap. Memory: every run kept free+cached at or above 3.29 GB, with no critical pressure.

Not yet measured:
- the N=1 no-regression arm (joint never engages at N=1 by construction);
- widths 5–8 in serving;
- long-context peak memory.

The exact multi-row Q8_0 kernel in mlx-native (`perf/q8-0-mv-multirow-exact`) measured 1.15–1.46× over per-row nr2 at M=4 in isolation [unverified]. End to end it measured +1.2% (95% CI −1.8% to +4.4%), so it was not published (`docs/research/adr063-k1-q8-multirow-disposition.md`).

## Changed files

- `src/inference/models/deepseek4/row_matmul.rs` (new): row semantics, row views.
- `src/inference/models/deepseek4/joint_decode.rs` (new): joint transaction and layer loop.
- `src/inference/models/deepseek4/joint_decode_core.rs` (new): per-row cores.
- `src/inference/models/deepseek4/joint_decode_routes.rs` (new): fail-closed admission by matrix storage type.
- `model.rs`: records the route proof at load.
- `src/inference/models/deepseek4/compressed_attention_common.rs`: prelude and epilogue take `RowMatmul`.
- `compressed_attention.rs`: passes `Matrix`; exposes the sparse-decode helpers.
- `ffn_forward.rs`: `forward_ffn_rows` takes `RowMatmul`; independent rows force `mv_id`.
- `verifier_forward.rs`: shared `decode_layers_per_command_buffer`; callers pass `Matrix`.
- `decode_cohort.rs`: publication generalised to slices.
- `mod.rs`.
- Tests: `row_matmul_tests.rs` and `real_artifact_joint_decode_tests.rs` (new), `cache_tests.rs`, and `real_artifact_decode_cohort_tests.rs` (shared assertions).
- `src/serve/api/engine_deepseek4/joint_decode.rs` (new): readiness and the joint commit.
- `engine_deepseek4.rs`: module registration.
- `src/serve/api/engine/deepseek4_joint_decode.rs` (new): switch, width cap, partition, turn.
- `engine.rs`: resolves the joint width at worker start, routing, and the `settle_deepseek4_decode_lane` extraction.
- `CHANGELOG.md`.

No oMLX code was copied. The designs borrowed are the layer-major B-row step, per-request attention, and decode-fairness interleaving.
