# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- ADR-067 Splash gap, cycle tail: two off-by-default, bitwise-neutral switches for the Qwen DFlash2 verify step that settles acceptance on the host (every request with a repetition penalty, a thinking budget or the lazy tool grammar). `HF2Q_DFLASH2_DRAFT_PREFETCH=1` submits the draft-context commit and the DeltaNet replay and commits the next step's draft behind them before it waits; the next step adopts that draft when its inputs are exactly the prefetched ones, otherwise it is dropped and redrawn. `HF2Q_DFLASH2_GPU_DECISIONS=1` decides each host-sampled verify row at temperature zero from candidates the output-head command computes (the repetition-window logits, gathered, and the argmax outside the window) with the host sampler's own penalty and argmax functions, instead of copying and scanning 32 rows of 248,320 logits; rows under an active grammar or with a non-finite logit use the host sampler over their own restored logits, and `check` runs both routes and compares every window (0 disagreements in 5,646 served verify windows on Qwen3.8 and Qwen3.6). On Qwen3.8, eight exact four-agent runs interleaved in one ticket (all passing, one transcript): the time from the output head's GPU end to the next draft's GPU start falls from 4.63 to 1.00 ms, the four-lane cycle from 60.1 to 58.2-58.4 ms (-3.1%) and makespan from 33.0 to 32.5 seconds (-1.5%); within one run, alternating in blocks of 16 cycles (`HF2Q_DFLASH2_TAIL_BLOCKS`, reduced by `scripts/dflash2_cycle_tail.py`), both switches give -2.75 ms per four-lane cycle, the prefetch alone -1.93 and the candidates alone -1.63. On Qwen3.6, ten interleaved runs (all passing, one transcript): four-lane cycle 27.34 to 26.28 ms (-3.9%), makespan 18.79 to 18.36 seconds (-2.3%). The replay still costs about 1.3 ms per Qwen3.8 cycle beside the prefetched draft. A GPU performance-state sampler beside Splash's `decode-profile` and the hf2q server showed that the two had never been compared at equal GPU clock: the GPU leaves its top state after a few seconds of sustained load (Splash's tool: 1620 MHz for 3.5 seconds, then 1200-1300 MHz; the hf2q server 1280-1510 MHz through a run), and at equal frequency between 1306 and 1620 MHz the hf2q cycle with the switches is within 1.5 ms of Splash's on either side (ADR-067, "Cycle tail").
- ADR-067 second combined candidate (mlx-native 0.21.9): with the prepared routes at every verify width and a shared head, fused MoE stages, replay-storage reuse, context-first commit, the fused DeltaNet mixer, the alignment-independent attention tile, the cue fold and lane-pass capture ingest all on, Qwen3.8 four-agent makespan falls a further 13.94% (95% CI 13.05–14.81%; 32.75 against 37.91 seconds; six blocks, 12 of 12 pass; four-lane cycle 60.18 against 71.92 ms) and Qwen3.6 makespan falls to 0.870 of the first combined candidate (95% CI 0.803–0.943; 18.85 against 22.82 seconds; four-lane cycle 27.27 against 32.66 ms; ten blocks, 10 of 10 candidate and 9 of 10 control runs pass). Qwen3.6 output no longer varies from run to run: all ten candidate runs emitted the same 5,322 completion tokens. Every switch stays off by default until the promotion change. Remaining distance to Splash on Qwen3.6 (16.42 seconds) is prompt prefill (4.99 against 2.78 seconds per run).
- ADR-067 Splash gap, cycle scheduling: a per-command-buffer timeline of the DFlash2 verify cycle (`HF2Q_DFLASH2_CYCLE_TIMELINE=<file>`, diagnostic, reduced by `scripts/dflash2_cycle_timeline.py`; needs mlx-native `command_timeline`, unpublished) showed where the cycle's wall time exceeds its GPU time: the GPU idled 9.3 of 73.6 ms per four-lane Qwen3.8 cycle and 4.7 of 33.3 ms on Qwen3.6, most of it while the worker allocated and zero-filled the verify step's DeltaNet replay storage (188 MB and 91 MB per step) between the draft command and the target's first command. Two off-by-default switches remove most of it, both bitwise neutral (Class A, tested): `HF2Q_QWEN_VERIFY_REPLAY_REUSE=1` keeps that storage between steps (wall minus GPU 10.5–11.9 → 3.0 ms on Qwen3.8 and 5.0–5.1 → 1.9 ms on Qwen3.6 on its own; the retained storage is outside memory admission), and `HF2Q_DFLASH2_CONTEXT_FIRST=1` submits the draft-context commit before the replay is encoded, so the two commit commands overlap on the GPU (0.4–0.5 ms off the cycle's tail; the step still waits for both commands). With both on, same binary, every other candidate switch on, the Qwen3.8 four-lane cycle fell from 73.6 / 75.4 to 66.4 / 67.1 ms with identical completion text; the paired runs and Qwen3.6 are in `splash-gap/CycleHost.md` of the campaign state. The worker's `Qwen joint decode configuration` line logs both switches.
- ADR-067 combined candidate: with the staged draft command, live-tile MoE route, tiled verify attention and prepared prefill switched on together (mlx-native 0.21.8), Qwen3.8 four-agent makespan falls 28.18% (95% CI 19.83–35.66%, six blocks, 12 of 12 pass; 41.45 against 55.65 seconds on a quiet host) and Qwen3.6 makespan falls to 0.695 of the previous best (95% CI 0.659–0.732; 21.75 against 31.31 seconds; four-lane cycle 33.57 against 51.04 ms) over twenty interleaved rounds in which 19 candidate and 18 control runs passed. Every switch stays off by default. Measured costs: Qwen3.6 draft acceptance is about 4% lower, and the output text differs from the previous routes.
- ADR-067 Splash gap, Qwen3.6 MoE stage, second pass: under `HF2Q_QWEN_MOE_LIVE_TILES=1` each MoE layer now runs mlx-native's fused expert stages: one schedule that applies the verify-row mask and emits tiles of at most eight routes for the routed experts and the shared expert, one gather, one gate/up pass and one down pass for both, and the router and shared-gate logits of all full blocks of eight rows in one dispatch each. A masked width-32 layer is 8 dispatches and 6 barriers instead of 19 and 7 (5,876 against 6,316 dispatches per four-lane cycle). Output, expert ids and routing weights are bitwise those of the previous stage-by-stage form (tested on whole layers, masked and unmasked, widths 1 to 32, experts of up to 32 routes). That form stays as the fallback for layers the fused stages do not admit, such as a routed down stack in Q6_K beside a shared down in Q4_K (tested through the route's entry point), and model upload logs once how many layers run each form (`Qwen MoE live-tile layers`). Measured inside the served cycle by alternating the two forms per verify cycle: the MoE stage takes 14.8 instead of 15.7 ms of GPU time at 20.6 verified rows (0.90 ± 0.13 ms less GPU time per four-lane cycle). Splash's own operator on the same routes and memory pattern takes 12.3 ms model-free, this route 12.5 ms. Needs mlx-native kernels that are not yet published. No switch was added and none changed its default.
- ADR-067 Splash gap, Qwen3.6 MoE stage: `HF2Q_QWEN_MOE_LIVE_TILES=1` (off by default; requires `HF2Q_QWEN_MOE_STAGED_M8=1`) runs each MoE layer through mlx-native's live-tile expert kernels: one route schedule, one gate/up pass with a BF16 SwiGLU kept in the schedule's rows, a down pass over those rows, both launched on the indirect grid of live tiles, the shared expert as a tile per block of eight rows, and router logits on the row-exact kernel at every width. A row's MoE output no longer depends on the other rows or lanes of its step: every row of a masked width-32 verify step is bitwise the row decoded alone (tested, with the F64 whole-layer bound). The one exception is an expert product that underflows FP32, where the register path and the matmul tile may round differently (observed on results of magnitude at most 4e-31). The three MoE switches are resolved together at model load: an invalid value or combination fails startup, and the worker's joint decode configuration logs the route as `moe_experts`. Needs the unpublished mlx-native live-tile kernels; not quality-qualified.
- `HF2Q_QWEN_PREFILL_PREPARED=1` (default off): the Qwen bounded prompt prefill runs its eligible Q4_K/Q6_K products (dense FFN gate/up/down; DeltaNet qkv, gate, output and fused alpha/beta; full-attention fused Q/gate, K, V and output) through the prepared row-tile kernel on one BF16 copy of each activation, sharing the planes of the width-32 joint target step. The stock matmul pays a whole 128-row tile for every started tile, so a 31-row chunk costs what 128 rows cost; the prepared kernel works in 32-row tiles with a partial last one. The route is scoped to prompt-prefill target forwards; MTP catch-up, verify windows and decode keep their kernels. Class B against the stock prefill (BF16 activation, different accumulation order): an F64 reference bound scaled with the inner dimension, bitwise repeatability on recycled storage and across resumed chunks, and independence of a row from later rows, the chunk size and other slots are tested. Real Qwen3.8 artifact, model-free, warm, two runs: 720 rows 1,112–1,117 ms to 915–950 ms, 157 rows 381 to 235–244 ms, 31 rows 206 to 64 ms, 5 rows 64–71 to 50–54 ms; the greedy token of every chunk's last row agreed. Exact four-agent Q5, same binary, two runs each, all passing: total prefill 18.0–19.3 s to 13.6–14.3 s, makespan 54.2–55.4 s to 50.1–51.4 s; the greedy transcript differs (5,361 against 5,237 completion tokens) and accepted draft rows per lane-cycle were 5.24 against 5.37 on those different transcripts. Needs mlx-native 0.21.8 (prepared projections at any row count), now pinned as `=0.21.8`. One bitwise-neutral change rides along: the fused attention stage's output of a prefill layer comes from the decode pool instead of a fresh allocation per layer. (An earlier version of this entry also listed the fused Q/gate temporary; that buffer is pooled on the prepared route only, and the stock route still allocates it per layer.)
- Qwen prompt prefill, review follow-up of the prepared route: a SlotAware worker now validates `HF2Q_QWEN_PREFILL_PREPARED` when it provisions its slots and logs `Qwen prompt prefill configuration`; an invalid value stops the server instead of warning at the first prefill, and so does the switch beside the chunk-scan prefill investigation route (its alpha/beta projections would stay on the stock kernel and a row's bits would depend on whether its chunk is a multiple of 64 rows). The model-level Class B gates run on a fixture on which every routed product of every layer moves a logit (asserted), at fresh chunk sizes 5 to 720, with decode-after-prefill and capture checks; the F64 gate covers K = 5120 and 17408 and the fused alpha/beta product and adds an RMS bound that a tenfold larger error fails.
- `HF2Q_QWEN_PREFILL_FOLD_CUE=1` (default off; requires the prepared prefill route): the last content chunk of a Qwen bounded prompt prefill and the generation cue behind it run as one forward that takes the stable-boundary checkpoint inside it (DeltaNet convolution and recurrence split at the boundary into the checkpoint image, full attention and the MoE FFN built per segment, a second head row). Class A: the checkpoint, both logit rows, the DFlash2 capture and the slot state are bit-identical to the content forward, snapshot and cue forward (synthetic model and both real artifacts), and the Qwen3.8 four-agent transcript is unchanged. A content chunk of at most 32 rows that the cue would carry past 32 keeps its two forwards (the fold measured about 11 ms slower there). Exact Qwen3.8 four-agent Q5, same binary, one undisturbed run pair: logged prefill 12.18 s to 11.43 s, makespan 37.71 s to 36.61 s; a second pair ran on a disturbed host (decode cycles 7% and 23% slower than undisturbed) and is not counted. Qwen3.6: per-turn medians 2–16 ms lower in four of the five folded size classes and 6 ms higher in one, not resolved in run totals.
- Qwen DFlash2 prefill capture ingest: a chunk's captured target rows are projected in passes of the draft workspace's whole row capacity for the one lane, in one command buffer, instead of one blocking eight-row commit per block. Unconditional and bitwise-neutral: on both pinned checkpoints, at one to four lanes and chunk sizes 5, 31, 157, 720 and past a ring wrap, the draft rings, the lane cursors and the next draft cycle's logits and proposals equal those of the block commits; the Qwen3.8 four-agent transcript is unchanged.
- Qwen cue fold and lane-pass ingest, second review follow-up: with `HF2Q_QWEN_PREFILL_FOLD_CUE=1` a request now keeps its two forwards instead of failing when a GLP vector is bound (`hf2q serve --glp`), when dump-bisect or the dense-Q split profile is enabled, or when the checkpoint storage cannot be allocated; a checkpoint that cannot be assembled after a folded forward is skipped, or fatal, by the rule of a failed snapshot between two forwards; folded checkpoints count in `hf2q_qwen_anchor_snapshot_milliseconds_count`. The prefill switches are validated before a Qwen model is loaded, and the refusal reads `Qwen prompt prefill switches: <variable and reason>`. Default-run tests added: the fold on a synthetic quantized-MoE model, at one to four slots on TQ and F16 caches, the scheduler transaction (ready with the checkpoint; a request that restores a folded checkpoint and diverges behind it), and the ingest gate with two target layers, the Dense BF16 draft projection and a non-zero row start.
- ADR-067 D0–D4 implementation candidate: add the family-neutral DFlash2 checkpoint/runtime/selector core, pinned Qwen3.8, Qwen3.6 and Gemma descriptors, explicit Qwen and Gemma target adapters, fixed-width 32-row Qwen target verification, and an off-by-default Qwen SlotAware draft/verify/commit transaction. `HF2Q_DFLASH2=1` additionally requires identity-matched `HF2Q_DFLASH2_CONFIG` and `HF2Q_DFLASH2_WEIGHTS`; any unknown checkpoint, target mismatch, restored-prefix mismatch or unsupported family fails closed. This is not a promoted route: D1 acceptance, D2 parity/performance, D3 semantic/state, D4 overlap, D5 campaign and D6 serving gates remain required.
- ADR-065 E1: add an off-by-default complete-round stage observer for Qwen3.8, DeepSeek-V4 and Gemma 4. Paired wall-only and split-profile modes quantify observer distortion; closed records separate GPU projection, DeltaNet, KV, attention, FFN/MoE, head and MTP work from CPU readback, selection, rollback, arena and host gaps, retain Unknown without using it for decisions, and count DeepSeek scratch release/reallocation. A checked-in analyzer validates retained Q5 receipts and ExactTranscripts against the previous default.
- ADR-065 E2/E7/E8: a published diagnostic-only Qwen3.8 joint-verify prototype measured complete 4×1 and 4×4 smoke rounds, then the full nine-shape grid at 2k, 16k and 64k context with seven timed repetitions and stage splits. The pre-registered G0 computation found a strong isolated round-rate factor (`s=1.9093`, 95% CI 1.7742–2.0364) but no eligible canonical critical-path share (`f=0`), so projected makespan speedup is exactly `S=1`: `NoGo`. Cost-priced allocation and >8-row projection stories are not pursued; the diagnostic prototype remains off `main`. The following J4/J5 production candidate supersedes this original stop only for the measured joint-MTP design.
- ADR-065 J4/J5: add joint Qwen MTP drafting and verification, one batched MTP forward per draft depth, one joint target forward, in-turn accepted-token drain, reusable slot anchors and the padded route's column-invariant few-row MMA projection. The predeclared six-block Qwen3.8 campaign is a `Win`: 34.600% lower n4 makespan (95% CI 33.738–35.452%), with all 70 runs passing Q5, quality, thermal and fairness gates; cancellation recovery also passes with byte-exact survivors. Dense artifacts with an MTP head enable the combined route by default. `HF2Q_QWEN_JOINT_MTP=0` restores serial K3 and canonical projection; MoE and no-MTP artifacts stay off by default.
- ADR-065 E3: add an off-by-default Qwen3.8 shadow drafter that measures request-history and MTP K3 accepted prefixes only on scheduler-confirmed joint-ready states without committing a draft. A checked-in reducer reports histograms by slot, turn type and context bucket, transform eligibility and proposal cost.
- ADR-065 E4: add an off-by-default Qwen3.8 observer with monotonic intervals for every prefill slice and joint or solo decode turn, joined to per-lane counterfactual verify eligibility. A checked-in extractor joins retained Q5 coordinator timestamps, derives queue and idle/tool gaps, partitions baseline makespan, reports joint widths, and estimates the eligible joint-decode share with run-clustered uncertainty.
- ADR-065 E5: add an off-by-default DeepSeek-V4 joint-step observer that captures sanitized routed expert ids for widths 2–4 without a pipeline stall. Closed-schema records and a checked-in reducer report per-layer unique-expert histograms, separate hash/score routing, and the routed weight-byte reduction of a unique-expert path.
- ADR-065 P4b: SlotAware Qwen partial-reject rollback restores DeltaNet recurrent and conv state directly from capture buffers in one GPU command buffer instead of copying through host memory. The fixed-count Qwen3.8 campaign was a Win (1.33% makespan improvement; 95% CI 0.67–1.98%) with N=1 also Win, every fairness/safety/Q5/thermal/completeness gate passing, exact transcripts, exact transaction stores and passing cancellation recovery. Dense Qwen3.8 therefore uses the GPU route by default; `HF2Q_QWEN_LA_GPU_RESTORE=0` restores the host route, while Qwen3.6 remains host-default.
- ADR-063 C2: Qwen SlotAware prompt reuse now keeps a bounded, pending-publication anchor store per slot and restores the deepest valid earlier prompt after divergent edits. Anchor payloads use one right-sized DeltaNet image through the P4b host/GPU route; restore preflights all ranges, binds cache lineage and graft identity, prunes descendants before writes, preserves saved-logit equality hits, and fails stop on committed Metal errors. The default remains one committed anchor; `HF2Q_QWEN_ANCHOR_DEPTH=4` selects the qualification shape, and optional per-slot quotas plus `/metrics` owned-byte, lifecycle, timing, and joint-ineligibility counters bound and expose the storage.
- ADR-065 B1: freeze the Rust 1.99.1 / mlx-native 0.15.18 phase-1 binary, a fixed-count two-arm campaign and independently re-derived per-family fairness safety ceilings for Qwen3.8, DeepSeek-V4 and Gemma 4. Mixed `K+M` controls now release every cohort's first streamed turn from one bound clock after later-cohort prefill, so the width-four audit cannot fail from schedule drift.
- Gemma combined candidate: hf2q surfaces and forwards the mlx-native Q5_0 tensor-MM opt-in (`HF2Q_Q5_0_TENSOR_MM=1`, mlx-native #99; needs mlx-native 0.18.0, pinned separately). `/hf2q/v1/runtime` publishes the resolved value as `active_controls.q5_0_tensor_mm` (`true`/`false`, next to `mm_id_schedule`), and the first Gemma 4 load of a process logs one line `Gemma4 native routing: q5_0_tensor_mm=<true|false> mm_id_schedule=<grid|compact>`. The Gemma Q5 case declares `Q5_0_TENSOR_MM` (override `HF2Q_Q5_Q5_0_TENSOR_MM`), and `serve_gemma4_opencode.sh` forwards `Q5_0_TENSOR_MM=1` as `HF2Q_Q5_0_TENSOR_MM=1` and refuses any other non-empty value.
- Diagnostics: `scripts/four_agent/q2_traced_run.py` now traces DeepSeek-V4 four-slot cases as well as Qwen3.8 (gap program S3). The case's family selects the witness manifest of sources its engine publishes, and only a case that declares a text-only locator gets one. Q2 window freezing still takes Qwen3.8 runs only. With `--gpu-timestamps`, a traced DeepSeek-V4 run records per-command GPU time.
- ADR-063 gap program: `HF2Q_GEMMA_DECODE_ATTN_NE=4` (off by default) runs Gemma 4 d256 (sliding) decode attention with mlx-native's NE head-split kernels (`flash_attn_vec_hybrid_{batched_,}ne`), the peer's decode split of each simdgroup into NE lane groups that score NE keys per pass. It covers all three decode paths: the batched cohort dispatch, the mixed-bucket per-slot fallback and single-slot decode. The kernels share their per-row math, so a slot's output does not depend on which path or cohort it decodes in; d512 layers keep stock. Class B against stock; published as `active_controls.gemma_decode_attn_ne`, bound by the Gemma Q1 packet field `decode_attn_ne`, and forwarded by `serve_gemma4_opencode.sh` (`DECODE_ATTN_NE`) and the Gemma Q5 case (`HF2Q_Q5_GEMMA_DECODE_ATTN_NE`).
- Gemma Q5 campaign plumbing now declares `HF2Q_Q5_GEMMA_DECODE_DENSE=stock|row_exact|mma` and forwards it through the canonical launcher as `HF2Q_GEMMA_DECODE_DENSE`, allowing a single case to compare the stock and combined-candidate dense routes without hard-coded per-arm environments.
- Diagnostics (ADR-063 Q4-TRACE-GPU-TIMING): traced runs can opt in to per-command GPU timestamps (`native_observer.gpu_timestamps: true`; `q2_traced_run.py --gpu-timestamps`).
  - The phase trace records mlx-native's existing per-command GPU timing stream (Metal command-buffer start and end) in the same ordered queue as the native events, plus the joint forward's command plan and paired host clock samples.
  - The reducer validates the records fail closed and checks clock-free duration bounds. The attribution tool reports per-epoch GPU busy time and bubbles and per-chain GPU time, so a Qwen joint step's completion tail can be split by layer chain.
  - Off by default: configs, headers and traces without the key are unchanged.
- ADR-063 DeepSeek-V4 gap program (fix B stage 1): `HF2Q_DEEPSEEK_BATCHED_ATTN_CORES=1` (off by default, read once at model load) encodes joint decode's cache-dependent attention cores phase-major: per layer, one mlx-native `deepseek_rows` dispatch per op covers every joint row (main and indexer compressor recurrences, indexer score + top-k, sparse attention with the flash path once per distinct width), each row reading its own cache through a fixed binding slot, instead of the same kernels once per row. The compressor/indexer F16 projections become one `row_matmul(IndependentRows)` over the rows: one stock-order `m = B` dispatch under `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC=1`, per-row one-row matvecs into separate buffers plus one exact gather on the stock family. RoPE, quantization and Hadamard of compressed rows and the indexer query run once over all rows, and emitting rows copy into their own caches. The rows kernels copy the one-row kernels' bodies, so every row is bitwise equal to the per-row cores and every switch combination stays Class A against stock (model-free tests at B = 1..4 on window, HCA and CSA layers, both matvec families). Layers with a row at position 0, and serial decode, keep the existing path.
- ADR-063 DeepSeek-V4 gap program (fixes A and C): `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC=1` (off by default) runs the DeepSeek-V4 few-row F16 and Q8_0 products through mlx-native's `row_exact_matvec_stock`, whose every row is bitwise the stock one-row kernel's (`dense_matvec_f16w_f32io` for F16, the Q8_0 NR2 matvec for Q8_0), so decode output is unchanged (Class A). Q8_0 products take it only where the stock route is that NR2 matvec, read from the registry's frozen routing policy or mlx-native's process policy; under `HF2Q_Q8_0_MV_NR2=0`, and at 2..8 rows under `HF2Q_DECODE_MV_EXT=1` (which a Qwen3.8 load sets in-process), they keep the stock route. Joint decode then issues the F16 router, HC and compressor/indexer products as one `m = B` dispatch instead of B one-row calls, the Q8_0 projections and head read their weights once for all rows, and grouped output-A runs as one dispatch on token-major activations (no permutes). Prefill F16 products at 2..8 rows keep the stock tiled kernel. The switch is read once at model load and covers serial, joint, prefill and verifier graphs; joint-decode admission is unchanged. The build-time `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC_DEFAULT=1` makes the row-exact family the default of a binary when the runtime switch is unset (for harnesses that strip `HF2Q_*`, such as the Q1 endpoint supervisor); other values fail the build. Model-free tests prove every row at B = 1..8 equals the stock switch's one-row output bitwise on the ds4 0731 shapes. The pinned mlx-native 0.18.0 release provides `row_exact_matvec_stock`.
- ADR-063 DeepSeek-V4 gap program (fix 1): `HF2Q_DEEPSEEK_TAIL_ROWS_STEP=1` (off by default, read once at model load) runs DeepSeek-V4 token-wise prompt work — the 8-token recovery tail after the rollback anchor and cached suffixes below the 33-token matrix floor — as rows steps of up to 8 tokens instead of one one-token verifier transaction per token. A rows step is one layer-major transaction on the joint-decode routes (one-token routes for every product, `mv_id` experts) whose cache-dependent attention cores run per row, in row order, on the request's one cache behind a full barrier, so every row's state, the logits and every cache byte, including the recovery anchor, equal the one-token steps (Class A). A 32-token cached suffix runs as three rows steps, the anchor, then the tail. Chunking depends only on the token-wise segment, never on cohort width or peer state. With `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC=1` the step's F16 and Q8_0 products read their weights once for all rows. New tests: model-free bitwise core equality across a window wrap, ratio-4 emits and a ratio-128 block; `official_artifact_rows_step_matches_one_token_steps_bitwise` (model load, including anchor replay). Bench: `ds4_tail_rows_microbench`; `ds4_op_microbench` takes `HF2Q_DS4_OP_BENCH_ROWS`. The width is decided per chunk: when rows steps become unavailable mid-prompt (an in-process Qwen3.8 load sets `HF2Q_DECODE_MV_EXT=1`), the remaining tokens run as one-token steps instead of failing the request.
- ADR-063 Q4-QWEN36 gap fix F2: `HF2Q_QWEN_MOE_BATCHED=1` (off by default) runs each Qwen3.6 MoE decode layer as one batched route over the step's rows instead of once per row. The router and shared-expert gate logit stay F32 and the shared expert stays Q4_K/Q6_K as stored in the GGUF (the BF16 copies are not made), read once per step by a row-exact F32 mat-vec (mlx-native `dense_matvec_f32_rows`) and the exact K-quant `mN` mat-vecs; routing runs the SIMD-group top-k over all rows; expert gate, up and down are one forced `mul_mv_id` each over rows × top-k pairs, followed by one weighted reduce. Serial decode (including MTP drafting) runs the same route at one row, so a row's output is bitwise the same solo or in a joint step of 2–8 rows; it is a Class B change against the default BF16 route. The worker-start `Qwen joint decode configuration` line logs the resolved `moe_ffn` route.
- ADR-063 DeepSeek-V4 gap program (fix 2): the DeepSeek-V4 prefill FFN's slotted Q2_K down projection now reuses the compact `mm_id` routing schedule of the paired IQ2_XXS gate/up call (same expert ids, rows, top-k, experts, scratch and encoder) instead of building a second one, so each prefill layer runs one `map0c` instead of two (quiet-window cells, pair + down per layer: −103 µs at 512 rows, −255 µs at 1,024, −481 µs at 2,048; neutral at 256). The schedule depends only on the ids, so output is byte-identical (Class A); mlx-native refuses a schedule recorded for other ids, shapes or encoder. `HF2Q_DEEPSEEK_ENCODER_STAGES=1`, which may split the encoder, keeps the self-scheduling down call (the flag is now read once per process). A refused reuse fails the prefill: a key mismatch is a wiring bug, not a fallback case. Tests: reused vs fresh bitwise in a plain and in a reorder-recorded session, and refusal for other ids. Bench: `ds4_prefill_microbench` `routed_gate_up_down` fresh/reused cells.
- ADR-063 Q4-GEMMA-PREFILL-MOE P1 (#690): Gemma 4 prefill forwards above 32 rows now run the routed-expert down projection on the slotted expert entry point (`[rows, top_k, K]` activations, `top_k = 8`) instead of flattening them to `8·rows` single-expert rows. Both calls read the same bytes, write the same offsets and list each expert's rows in the same order, so outputs are bitwise identical (Q1 Class A), and the expert matmul launches 8× fewer threadgroups. The route follows DeepSeek-V4's paired-prefill rule: 32 rows or fewer, an `HF2Q_MM_ID_ROUTING_THRESHOLD` override and affine stacks keep the flattened call. `HF2Q_GEMMA_PREFILL_MOE_DOWN=flat` selects the previous route for the same-binary A/B (published as `active_controls.gemma_prefill_moe_down`); it is removed at promotion. The Gemma Q5 case forwards `HF2Q_Q5_GEMMA_PREFILL_MOE_DOWN` and `HF2Q_Q5_MM_ID_SCHEDULE` through `serve_gemma4_opencode.sh`, and `scripts/four_agent/gemma_prefill_moe_gates.py` evaluates the frozen S2/S3/S4 and fairness gates from run receipts.
- Gemma 4 simultaneous cold first turns (ADR-063 Q4-GEMMA-COLD, the four-agent turn 0) now run as one joint transaction instead of one inline per-slot admission after another. When every slot is idle and two or more cold text requests arrive in one admission quantum, each with an exact pre-generation boundary of at least 33 tokens and at most 4,096 rows in total, the worker reserves the lanes in arrival order, prefills every boundary in one forward, captures each lane's boundary checkpoint in arrival order, and replays the short generation cues through the batched decode body (byte-identical to per-slot decode), so no lane waits for the others' serial prefills. Each request's tokens, logprobs, checkpoint and cache bytes are bitwise equal to its per-slot admission (Q1 Class A; official-artifact gates `gemma_joint_cold_*`). `HF2Q_GEMMA_JOINT_COLD_PREFILL=0` is the kill switch. Groups with a cached lane, vision input, a graft, or a diagnostic attention override keep the per-slot route.
- ADR-063 Q4-GEMMA-DECODE V2 diagnostic: `HF2Q_GEMMA_ROUTING_OVERLAP=1` (off by default) records the Gemma 4 batched decode workload: each step's routed expert ids per layer (blit-copied and read after the step's existing final wait, no added sync), each slot's KV length per layer type, and whether the step mixed attention buckets. The worker prints one `[ROUTING_OVERLAP]` JSON line of histograms at exit. The Gemma Q5 case declares `GEMMA_ROUTING_OVERLAP`, and `serve_gemma4_opencode.sh` forwards `ROUTING_OVERLAP` as `HF2Q_GEMMA_ROUTING_OVERLAP` only when set.
- ADR-063 Q4-GEMMA-PREFILL-MOE P3 (#690): pins mlx-native 0.16.0 with the compact expert-matmul tile schedule (`HF2Q_MM_ID_SCHEDULE=grid` keeps the uniform grid as the bitwise A/B arm and diagnostic; published as `active_controls.mm_id_schedule`). The Gemma prefill down call reuses the gate/up routing schedule when the session was not replaced in between (block-quantized stacks; profiling pre-finishes and the imatrix intercept take the self-scheduling slotted call). Qwen's pooled scratch check uses the native capacity accessors, and the fused gate/up-SiLU path takes the scratch through its key-invalidating mutable accessor.
- DeepSeek-V4: hf2q now serves ds4's DeepSeek-V4-Flash 0731 GGUF (IQ2_XXS gate/up experts, Q2_K down experts, Q8_0 shared experts and attention, F16 router, HC and compressor projections; ADR-063 Q4-DS4-RACE, #667). `general.file_type` 19 maps to a new `IQ2_XXS` pool identity (hf2q convert has no IQ2_XXS target and refuses it). The `joyai-llm` pre-tokenizer is accepted as an alias of `deepseek-v3`. An absent `embedding_length_out` defaults to `embedding_length × hyper_connection.count` and an absent `swiglu_clamp_shexp` to `swiglu_clamp_exp`. A `compress_ratios` array longer than `block_count` is accepted only when every extra entry is 0. F16/BF16/F32 token embeddings use the dense gather. A float `output_hc_fn` is applied one row at a time, and IQ2_XXS routed experts join the joint-decode proven set, so joint decode stays bitwise equal to serial decode on this file. One-row F16 products now use the F32-input matvec instead of the tiled kernel that rounds the input to half. hf2q's own DeepSeek artifact (Q2_K/Q3_K/Q8_0, no F16 matrices) takes none of the new routes. Requires mlx-native 0.15.18.
- ADR-063 Q4-DS4-RACE harness (`scripts/four_agent/`): `q3_campaign.py --no-stock-arm --stock-record VERDICT_JSON` judges a ds4-file campaign against the peer with the vs-stock terms read from a bound same-artifact record, and the ds4 native adapter gains a declared `artifact_relation` (`different` or `same_as_hf2q`) for a same-file `--native-comparison`. New case files `deepseek4-ds4file.json` and `native-ds4-deepseek4-samefile.json`, and `ds4_file_digest.py` for the per-tensor digest of the §6.1 tensor-load parity check.
- ADR-063 Q1-GEMMA-T2LOC (test-only): a four-lane Gemma stage localizer for the packed T2 route and its All128 reference. Packed prefill rows map to their agents through `gemma_endpoint::packed_owners`, and new batched-decode stage hooks sit in `batched_body.rs`. Each route also writes the unchanged endpoint store, which serves as the observer-effect control against the b3 stores. The Q1 endpoint supervisor gains a `multistore` case kind with closed registry checks (`gemma_loc4`). `q1_stage_localization.py` reports the per-lane first divergence. No production path changes.
- ADR-063 Q1 endpoint supervisor (`scripts/four_agent/q1_endpoint_supervisor.py`): a resource policy may select the Q5 runtime pressure rule (`runtime_pressure_rule: "q5-critical-stop"`: warning pressure is recorded, critical or unrecognized pressure stops; the pre-launch baseline stays normal-only) and a footprint-calibration policy (`footprint_calibration: true`, no process cap, the runtime floor is the cap) for the Q1-DS4 measured-footprint run. Every case receipt now records its measured memory footprint (`memory`: baseline and minimum Free+cached, charge, process peaks, pressure counts, swap/compressor/wired growth) with a per-sample `memory-series.json`. Policies without the new keys behave as before.

- ADR-063 Q4-GEMMA-ATTN (#668): the precise tiled F32 warm-attention route was evaluated and falsified (`docs/research/adr063-gemma-tiled-attn-design.md` §12). It met its kernel precision bound (layer-0 nmax 4.95e-7) and the bitwise joint/determinism gates, but changed 72 of 952 warm-prefill decisions outside the 0.10-nat near-tie policy. It is not offered; `HF2Q_GEMMA_WARM_TILED_ATTENTION` keeps `0`/`1` only. Kept from the work: `/hf2q/v1/runtime` now publishes the resolved Gemma warm-attention kernel as `active_controls.gemma_warm_attention_kernel` (`VecHybrid` or `TiledF16`, `null` for other families), and the Gemma Q5 case checks it when an arm sets `HF2Q_Q5_GEMMA_WARM_ATTENTION_KERNEL`. A Q5 readiness check may declare `skip_if_empty`: only such a check (today only the Gemma kernel check) is recorded and skipped when its expected value is empty, so stock arms that do not publish the field (v0.1.28) still pass, while every other check still fails on an empty value. The real-artifact warm-route envelope (`gemma_warm_attention_route_envelope`, formerly the last-row decision envelope) now scores every suffix row with the Q1 decision statistic, and the localization diagnostic records relative RMS.
- Four-agent harness: `qwen_n4_live_smoke.py --request-setup qwen36-vendor-coding-v1` runs the coding client with the Qwen3.6 model card's precise-coding sampling: temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence 0 and repetition penalty 1.0. Every request carries these values, identically on hf2q and the peer. The template, tools, tasks and six-step workflow stay canonical. `q5_cases/qwen36-vendor.json` and `peer-qwen36-vendor.json` run both engines with that sampling and no thinking budget. This lets the ADR-063 Q4-QWEN36 paired campaigns test Qwen's recommended setup (`docs/research/adr063-qwen36-recommended-setup.md`). The case binds the setup to its approval, and `quality_campaign.py` and `q3_campaign.py` refuse arms whose setups differ. Canonical runs are unchanged.
- Four-agent harness: `qwen_n4_live_smoke.py --repair-turns 1` adds a Qwen3.6-only repair-turn workload (ADR-063 Q4-QWEN36, operator decision 2026-11-03). After a failed `run_tests`, an agent gets exactly one more `write_file` and one more `run_tests` on the same target. A second failure fails the agent, and every other step stays canonical. The approval binds the grant, `aggregate` and the Q2 run audit replay it, and both campaign drivers refuse arms with different grants. `q5_case` refuses the grant for any family but `qwen36`. The cases are `q5_cases/qwen36-repair1.json` and `peer-qwen36-repair1.json`, both on the hf2q-converted GGUF. Without the option, runs are unchanged.
- Four-agent harness: the `--repair-turns 1` grant now also admits the Splash Qwen3.6 arm (ADR-063 Splash design decision D9), so Splash runs the same declared Qwen3.6 workload as hf2q and the peer. The client allows Splash only when its `family_controls.q3_family` is `qwen36`; ds4, TensorFold and oMLX stay refused. The Q2 run audit and the Splash audit module replay a repaired Splash run and refuse one relabelled as canonical. The case is `q5_cases/splash-qwen36-repair1.json`. Without the option, runs are unchanged.

### Changed

- Correct ADR-067's Splash comparator attribution: the retained 16.420344-second arm is Qwen3.6's, not Qwen3.8's. D5 now agrees with ADR-063's existing family-specific gates; no threshold or quality requirement changes.
- Qwen3.8 Q3 plans now enforce the ADR-063 absolute 5.304 accepted-plus-bonus token/lane-cycle floor from every passing measured candidate run's frozen raw telemetry. Explicit invalid-Splash evidence substitutes this bound mechanism gate only for Qwen3.8, never for Qwen3.6; missing, changed or mismatched source/binary/model/workload evidence blocks a win. Historical plans retain their recorded rules.

- Q4 release-gate campaigns handle failed runs by the operator's 2026-11-12 rule (ADR-063 Q4, "strict on us, fair on them"). Any failed run of the candidate still makes the family verdict `Failed`. A stock, peer or Splash run whose client reports a failure is kept, counted and rerun in the same block position, at most twice; three failures in one position make that arm `Unavailable`. A passing run is never rerun; a run the client abandoned at one of its own deadlines counts as failed, which works against the candidate. The rule is frozen in each campaign plan (`comparison_failure_policy`; `q3_campaign.py --comparison-reruns`, default 2, with 0 planning a new campaign under the earlier rule), and `verdict.json` names the rule, lists every discarded run and reports each arm's attempts, failures by kind and failure rate with a Clopper-Pearson 95% interval. Speed is compared on passing runs only; no threshold or margin changed. Before this, one peer or required-Splash task failure anywhere ended a campaign as `Failed`: in the five Qwen3.6 gate campaigns of November 2026 (their `campaign.json` records) stock failed its task in 4 of 24 runs, the peer in 1 of 22 and Splash in 1 of 20, and the DeepSeek-V4 ds4-file gate needed four campaigns. The experiment arm of that time failed 1 of 23; the candidate gets no rerun, so its pass rate on the campaign's case has to be established before a gate campaign is launched.

- The v0.1.30 four-agent release verdict now requires every family to be at least 5% faster than published hf2q v0.1.29 and statistically no slower than every valid same-artifact peer. Campaign plans freeze both 95% CI upper ratio thresholds (`stock_margin_ratio=0.95`, `peer_margin_ratio=1.0`); report recomputation preserves the thresholds embedded in older plans.

- DFlash2 Qwen lanes now bind resident draft rings to the exact committed token prefix. Exact cached follow-ups preserve or safely rewind an unwrapped ring and continue DFlash2 after suffix prefill; incompatible, same-length-different, or post-wrap restored prefixes clear the ring and fail closed to the established route. Inactive physical lanes use the selector's minimum valid one-row bound without participating in verification.
- DFlash2's fixed 32-row target verifier uses mlx-native's matching tensor-v1 tile instead of the 128-row tensor-v2 tile; every other dense MM shape retains its previous route. On the exact Qwen3.8 four-lane smoke, target verification fell from a 215.494 ms median to 123.537 ms and the full coding protocol from 103.141573 s to 74.388081 s with all clients successful. The DFlash progress DEBUG target now attributes draft, target verify, host planning, target commit, draft-context commit and total transaction wall time; normal INFO serving does not sample the timers.
- DFlash2's terminal EOS draft token is neither streamed nor retained, and logprobs requests keep the full-logit canonical decision path. The cancellation harness recognizes DFlash2's `active_lanes` width records: the four-lane Qwen3.8 cancellation gate exercised widths 1–4, kept survivor payloads byte exact, kept the victim payload as a prefix of its control, preserved server readiness and proved cleanup. The published-pin six-block campaign improved the prior joint-MTP route by 10.556% (95% CI 9.327–11.768%) and reached 5.307582 accepted-plus-bonus outputs per lane-cycle. DFlash2 remains opt-in because the unmatched roughly 80.47-second candidate and 79.7-second peer medians do not establish the required 95% CI upper ratio at or below 1.00.
- Qwen3.6's native-storage batched MoE layer now accepts DFlash2's fixed 32-row target pass instead of stopping above the eight-row joint-decode width. Ordinary one-to-eight-row decode retains its bitwise single-row invariant; width 32 uses the dense F32 router/shared-gate matmul path and is qualified against the whole-layer F64 reference. The exact Qwen3.6 checkpoint completed the four-agent DFlash2 calibration smoke with all clients successful, quality handoff intact and cleanup proven.
- Qwen3.6 DFlash2 now applies an online, source-bound verification budget: per-position acceptance calibration refits from the latest 4,096 labels every 64 cycles, cycle costs decay at 0.97, and every 32nd cycle explores the full window. The GPU handoff runs top-k over the fixed target tile, then masks unselected expert IDs and weights so only selected rows read or execute routed experts. Successive quality-passing four-agent smokes improved from 36.129 to 35.203 seconds with adaptive budgeting and to 30.947 seconds after excluding padding rows from expert work. Generic 32-row expert grouping and a five-route scalar live-expert tile were slower and are rejected; DFlash2 remains opt-in pending a prepared-weight BF16 register-matrix expert tile and matched D5/D6 campaigns.
- DFlash2 draft projections can use the source-bound AF4G64 layout behind `HF2Q_DFLASH2_DRAFT_AF4_G64=1`. hf2q quantizes BF16 projection rows to 36 bytes per 64 weights, uploads prepared projection-major code/metadata planes, concatenates Q/K/V and gate/up weights, and reuses bounded Table16 scratch through mlx-native 0.21.1. The registry-pinned exact Qwen3.6 Q5 smoke passed with cleanup proven; the isolated width-four draft command fell from 9.318 ms for BF16 to 7.878 ms. The route remains opt-in because target verification, host readbacks and separate commits still miss the binding end-to-end Splash gate.
- The DFlash2 draft command can run on Apple10-class kernels behind `HF2Q_DFLASH2_DRAFT_STAGED=1` (requires `HF2Q_DFLASH2_DRAFT_AF4_G64=1`, off by default, unpublished mlx-native kernels). AF4G64 projections of at least 960 outputs run a staged BF16-input tile with gate and up in one pass, draft attention reads the K/V rings in place instead of packing each lane's 2,047-row window per layer, and a four-lane vocabulary head runs the prepared target tile. A serialized-dispatch probe (`dflash2::probe`, real checkpoints, four lanes) measures the draft command at 5.12 ms against 16.30 ms on Qwen3.8 and 2.64 ms against 7.98 ms on Qwen3.6, and the context commit at 0.37 against 1.09 ms and 0.23 against 0.50 ms; Splash's own profile on this host is 5.51 + 0.46 ms and 3.04 + 0.32 ms. Projection and attention bits change within tested bounds (F64 reference, determinism, row and lane independence, every 32-input step of every served shape, NaN scratch, Metal shader validation). A six-block paired Qwen3.8 campaign on a loaded host (`qwen38-draft-staged-d5-v1`, GPU-token mode) passed all 12 runs with identical text in both arms: four-lane cycle −7.41% (95% CI 6.70–8.12%; 126.96 against 136.56 ms), four-lane target stage −6.87% (6.17–7.56%), client makespan −5.89% (4.02–7.71%; 67.94 against 71.80 seconds), peak footprint 71.10 against 70.08 GiB. With the draft waited, the four-lane `draft_us` fell from 17.20 to 6.01 ms on Qwen3.8 and from 8.2 to 3.07 ms on Qwen3.6. Qwen3.6 is not qualified: with `HF2Q_DFLASH2_GPU_TARGET_TOKENS` unset, the Qwen MoE default, verify rows and the verify tile width follow the draft's confidence, so a different draft changes the target's logit bits, and two of three on-runs in that mode failed the workflow oracle on model text against none of two off-runs. A checkpoint whose lane rows are not a multiple of eight now fails at load, and the prepared head is recorded only once the command that repacks it is committed. The route stays opt-in (ADR-067, "Staged draft command").
- Qwen DFlash2 can compose anchor-plus-proposal target tokens on the GPU and submit target embedding on the same queue without a draft completion wait or proposal readback. Dense Qwen3.8's matched Q5 pair fell from 86.397 to 80.758 seconds (6.53%) and uses the route by default. Qwen3.6 MoE additionally selects adaptive verify counts into physical lanes on the GPU, masks unselected routes before expert-weight reads, and excludes unverified target outputs from acceptance. Two exact four-agent Q5 smokes passed at 31.981 and 31.192 seconds versus 48.534 seconds for the staged control, but an intervening repetition failed task quality; MoE therefore supports `HF2Q_DFLASH2_GPU_TARGET_TOKENS=1` but remains staged by default pending repeatable qualification. `HF2Q_DFLASH2_GPU_TARGET_TOKENS=0` restores staged dense handoff.
- Qwen3.6 DFlash2 now places an explicit Metal read-after-write barrier between SIMD MoE top-k and the adaptive route-mask dispatch. The mask no longer races partially written expert IDs or weights, eliminating the scheduling-dependent greedy-token and tool-argument corruption seen in the staged GPU target path. Three consecutive exact four-agent repair-turn calibrations passed with complete tool workflows and proven cleanup (29.540, 31.512 and 38.738 seconds); the route remains opt-in pending the matched release campaign.
- Qwen DFlash2 transform-free exact-greedy requests now compute accepted and retained counts on the GPU and append the draft-context commit to the target output-head command through mlx-native 0.21.4. The unchanged host acceptance walk validates both counts after readback and fail-stops on any mismatch. Semantic transforms still use host acceptance, and partial Qwen DeltaNet acceptance still needs a separate target replay, so this is not yet Splash's single transaction. The fixed target-token workspace is now always 32 rows, including one-slot workers; unused logical lanes replay lane zero's anchor.

- Qwen3.6's staged target MoE now uses separate prepared F32 gate and up projections over one reused schedule/gather, retaining F32 SwiGLU and down input; this replaces the faster fused-BF16 route after a later exact run falsified its repeatable task quality. A fixed six-block balanced same-binary campaign passed all 12 exact task/tool/cleanup gates. Against native batched MoE, staged M8 cut four-lane target verification by 9.74% (95% CI 8.41–11.05%; 44.589 versus 49.387 ms per-run median), but its 6.90% client-makespan reduction was inconclusive (95% CI -2.98–15.84%). `HF2Q_QWEN_MOE_STAGED_M8=1` remains an explicit Q4_K/Q6_K Apple10 opt-in pending end-to-end precision and is still well above Splash's 17.93 ms target-forward average.

- Qwen3.8 DFlash2 target verification can prepare eligible Q4_K/Q6_K dense FFN weights once and dispatch width-32 gate, up and down projections directly from BF16 activation rows through mlx-native 0.21.6 (`HF2Q_REPACKED_GGUF_M32=1`). The route is confined to an active 32-row joint target transaction; ordinary prefill, decode, vision-language and other-family matmuls retain their established kernels. A fixed six-block balanced four-agent campaign on the registry-pinned binary, run without native profiling, passed all 12 task/tool/cleanup gates and cut client makespan by 13.92% (95% CI 12.53–15.30%; per-run medians 59.944 versus 69.758 seconds). The two routes do not emit identical text: the prepared route generated 5,361 completion tokens per run and the stock route 5,255. An earlier campaign (12.61%, medians 70.898 versus 80.979 seconds) ran with mlx-native profiling unintentionally on and is superseded. The 149.539 to 117.001 ms target-stage diagnostic and the rejections of prepared GDN and output-head weights, BF16 output epilogues, one-pass gate/up and a coherent last-arriver reduction were measured under that condition and have not been re-measured. Command-chain length was re-measured without profiling: settings from 1 to 64 layers per chain are indistinguishable at single-run precision. The route remains opt-in: this qualifies the stock-kernel improvement only. Splash's Qwen3.8 runs all abort on a workflow failure, so there is no gated Splash comparison; their per-turn data puts Splash at 41.4 completion tokens per lane-second against 28.9 for this route.

- `HF2Q_REPACKED_GGUF_M32_PROJ=1` (with `HF2Q_REPACKED_GGUF_M32=1`) extends the prepared width-32 route to the Qwen DeltaNet qkv, gate and output projections and the full-attention fused Q/gate, K, V and output projections of a DFlash2 target step. Each is prepared on first use and the projections that share an activation read one BF16 copy of it. The stock 32-row matmul has a floor of about 0.1 ms per serialized dispatch whatever its size; the prepared kernel does not. A fixed six-block same-binary Qwen3.8 campaign passed all 12 task/tool/cleanup gates and cut the four-lane target stage by 7.07% (95% CI 4.78–9.30%) and client makespan by 4.32% (95% CI 0.39–8.08%); output text differs from the stock projections, and the process footprint peaked at 69.95 GiB against 65.34 GiB. One exact Qwen3.6 run passed with a 49.19 ms four-lane target median against 51.92 ms. Off by default. ADR-067 now records Splash's own per-stage cycle times on this host (50.6 ms per four-lane Qwen3.8 cycle and 20.8 ms on Qwen3.6, against 110–118 and 56–61 ms for hf2q) as the target for each stage.

- With `HF2Q_REPACKED_GGUF_M32_PROJ=1`, the Qwen DeltaNet alpha and beta logit projections now run as one prepared product (alpha's rows, beta's rows and zero padding as one weight, split after the projection wave) instead of two stock products. Six-block Qwen3.8 campaign against the previous binary: all 12 gates pass, four-lane target stage −5.29% (95% CI 0.51–9.84%), client makespan −3.48% (95% CI 0.55–6.32%). New `HF2Q_REPACKED_GGUF_M32_HEAD=1` (with `HF2Q_REPACKED_GGUF_M32=1`, off by default) runs the 32-row output head through the prepared route: all 12 gates pass and the four-lane target stage is 3.27% shorter (95% CI 1.98–4.55%), but client makespan did not move (0.66% higher, 95% CI from 2.81% higher to 1.45% lower). Its logits differ from the stock head's, so the text differs; on this conversation it ran 377.5 cycles against 354 and accepted 5.21 against 5.33 tokens per lane-cycle. It adds 0.94 GiB.

- `HF2Q_QWEN_VERIFY_ATTN_TILE=1` (off by default) runs the full attention of a Qwen DFlash2 verify step as a tiled pass over the unchanged TQ-HB cache instead of the serial chain once per row: one batched K and one V encode for all rows, then mlx-native `tq_hb_verify_tile` (query pack, a split that computes a lane's window and a KV head's query heads together with Metal tensor matmuls, reduce, fused un-rotation and gate), six dispatches per layer against about 225. Cache bytes and cursors are the serial chain's; the attention output is Class B with an F64-referenced bound and bitwise cohort independence. One Qwen3.8 layer at four lanes and a 2,048-token history takes 0.116 ms model free against 1.812 ms (Splash: 0.130 ms). In exact Q5 runs the four-lane attention category fell from 25.95 to 2.02 GPU ms per cycle on Qwen3.8 and from 11.48 to 0.81 on Qwen3.6, and the four-lane cycle from 130.1 to 106.5–109.9 ms and from 58.2 to 45.8–46.0 ms. Requires an mlx-native release with `tq_hb_verify_tile`.

- `HF2Q_REPACKED_GGUF_ANY_WIDTH=1` (off by default, with `HF2Q_REPACKED_GGUF_M32=1`) admits the prepared dense FFN, DeltaNet and attention projections and output head at every tile width of a Qwen DFlash2 target verify step instead of only at 32 rows, through mlx-native 0.21.8's any-row-count kernel and one width rule. A Qwen3.6 verify tile of 4, 8 or 16 rows no longer runs the stock kernels: in exact Q5 runs of the combined candidate with the prepared head on (all passing) the one-lane cycle went from 22.6 / 22.1 ms to 17.1-18.8 ms and the two-lane cycle from 36.8 / 34.5 ms to 22.0-24.3 ms (seven runs); three- and four-lane cycles were unchanged within run noise. Plain joint decode and joint MTP verify steps keep their routes under the switch, so exact-mode joint rows stay bitwise the serial forward; a DFlash2-verified token and the same token decoded serially are different routes at every tile width, as they already were at 32 rows. A verified row's bits are the same at every tile width, in every cohort and beside any padding: model-free gates with F64-referenced bounds for a projection, the fused alpha/beta product and the dense FFN layer, whole-step gates on the dense and MoE layouts at widths 8, 16 and 32, and an official-artifact Qwen3.6 gate. `HF2Q_REPACKED_GGUF_SHARED_HEAD=1` (off by default) makes the model own one prepared output head that the staged DFlash2 draft and the prepared target head both read, instead of a copy each, prepared only when one of them reads it; logits are bitwise unchanged.

- `HF2Q_QWEN_VERIFY_MIXER_FUSED=1` (off by default, unreleased mlx-native kernels) issues the mixer of every target layer of a Qwen DFlash2 verify step in far fewer stages without changing a bit of its result. The prepared input products of a layer run as one wave (one kernel pass each, no reduce pass or barrier between them), the pre-mixer norm writes the BF16 rows they read, and the DeltaNet core of every lane window (convolution, Q/K norms, query scale, gates, recurrence, gated norm, the rows the replay commit retains) is one dispatch, mlx-native `gdn_window_fused`, in the structure of Splash's fused verify kernel: 7 dispatches in five stages per DeltaNet layer at four lanes instead of 159 in 14. Rows, captures, DeltaNet state and retained rows are bitwise the stock route's (Class A, qualified by test on the Apple M5 Max). In exact four-agent runs on the combined candidate the profiled projections + DeltaNet GPU time per four-lane cycle fell from 23.8 to 17.0 ms on Qwen3.8 and from 8.2 to 6.9 ms on Qwen3.6 (Splash spends 16.5 and 6.0 ms on the same work), the unprofiled four-lane cycle by about 6% on Qwen3.8 (75.1/75.6 to 70.6/70.9 ms; identical text in both arms) and 4-6% on Qwen3.6, and a Qwen3.8 cycle from 9,063 to 1,703 dispatches. Two runs per arm on each of two binaries, all passing; no paired campaign yet. With `HF2Q_REPACKED_GGUF_ANY_WIDTH=1` the wave follows the prepared routes to every tile width of the step (Qwen3.6 steps of 4, 8 and 16 rows); that combination is tested bitwise against the stock chain and not yet measured.

- Splash four-agent request parity now omits the semantic default `strict:false` from each function definition on the Splash wire, matching hf2q's rendered tool schema byte for byte; the audit inverse restores the canonical client shape and refuses drift.

- Gemma 4 prefill V unit norm (joint stream and cold multi-sequence lanes) now uses mlx-native 0.18.0's `dispatch_rms_norm_no_scale_f32_dual_perm`, which packs two (token, head) rows per threadgroup, instead of hf2q's own one-row-per-threadgroup encode of the same kernel, which is removed. The kernel runs one simdgroup per row at any 1-D dispatch geometry, so the f32 and permuted BF16 outputs are bitwise identical (Class A; model-free unit test at the Gemma 4 26B sliding and global shapes).
- CPU-sampled decode steps spend less host time after the GPU result (ADR-063 Q4-QWEN36 F4); every decision is unchanged.
  - The greedy argmax over CPU logits (temperature 0 with repetition penalty, the four-agent Qwen setting) uses a vectorized two-pass form of the same first-wins rule: about 115 → 24 µs per row over a 248K vocabulary.
  - While a tool-call grammar is active, the temperature-0 grammar-valid choice finds its first candidate with one scan instead of ordering the 64 best of the whole vocabulary first; only a rejected first candidate orders the rest, with the same probe budget and fallback.
  - A Qwen joint decode step takes its logits and argmax buffers from the decode pool instead of a fresh, CPU-zeroed allocation (about 170 µs per four-row step).
- Qwen3.5/3.6 MoE prefill builds the routed-expert schedule once per layer instead of three times (ADR-063 Qwen3.6 prefill P1). Above 32 rows, gate and up run as one paired expert matmul that shares a single schedule, and the down projection runs on the slotted entry point (`[rows, top_k, K]`), reusing that schedule. The schedule depends only on the expert ids, so outputs are bitwise identical (Class A). Model-free GPU tests compare the shared route bit for bit with the previous per-call route (Q4_K gate/up, Q6_K and Q4_K down, 33–512 rows, uniform and collapsed routing) and check it is deterministic. Prefill top-k routing keeps the legacy kernel: the SIMD top-k kernel picks the same experts on real-like logits but its routing weights differ in the last bits on about a third of rows, and its ids differ on NaN and infinite logits.
- Qwen resumed prefill is faster (ADR-063 gap program, route (c): chunks with a cached prefix and more than 32 queries on TQ-only KV). The BF16 bridge's dispatches now run on scratch held by the prefill's FA arena instead of eight fresh CPU-zero-filled allocations per FA layer (60 MiB at a 2048-token prefix, 132 MiB at 8192, for 512 queries), in one command buffer with no host wait inside the layer. Same dispatches on the same bytes, so outputs are bitwise identical (Class A).
  - Per FA layer on the model-free bench (M5 Max, n 512, 16-layer chunk, drained every layer as in production): 6945 → 3525 µs at a 2048-token prefix and 18262 → 11301 µs at 8192. The zero-filled allocations were 2.6 / 5.7 ms of it.
  - Real-model gate `qwen38_resume_attn_fast_real_model_exact_equal` on Qwen3.8 (both routes in one process; chunks of 796 + 2048): every chunk's last-row logits, every decode step's logits and the emitted tokens are bitwise equal. Synthetic 9100-token prompt: 6 chunks, 16 decode steps, logits sha256 `3aecbc9c…`, 80 fast-path layer calls. Four 40,065-token prompts: 84 chunks, 64 decode steps, `6fe118d6…`, 1,280 calls.
  - On by default; `HF2Q_QWEN_RESUME_ATTN_FAST=0` restores the previous per-layer allocations and wait. Model-free GPU tests check bitwise equality against the previous route at Qwen3.8 shapes and over a 16-layer chunk; the ignored `resume_attn_profile` and `resume_attn_fast_bench` tests give the stage breakdown and the comparison.
- Gemma 4 long prompts (over 4,096 tokens) that prefill at the same time are now bitwise equal to the same prompt prefilled alone (#725). Before, a co-installed cohort split one 4,096-row transaction among its lanes (four lanes at 1,024 rows each where a prompt alone takes 4,096), ran tiled attention on every lane (`HF2Q_GEMMA_TILED_LIVE`) where a prompt alone takes the warm route, and could put a lane with 32 rows into the joint MoE stream (`mm_id`) where alone it takes `mv_id`. Which of these happened depended on which other prompts were mid-prefill, so identical requests could answer differently. Now each lane keeps its own single-slot chunk plan; lanes join in order while the transaction stays within 4,096 rows and each lane's chunk has at least 33 rows, the others prefill per slot on their own turn, and installed lanes run the single-slot warm attention route (`PerSlotExact`, so `HF2Q_GEMMA_WARM_TILED_ATTENTION` now covers them and `HF2Q_GEMMA_TILED_LIVE`, with stable-boundary groups per slot since #728, no longer affects any serving path). Like the joint warm and joint cold paths, the installed cohort is off (every lane per slot) under `HF2Q_PREFILL_SLOT_BATCHED=0` or a set `HF2Q_MM_ID_ROUTING_THRESHOLD`, resolved once per worker. This is Class B relative to the previous release's co-installed outputs only; prompts of 4,096 tokens or fewer never take this path and are unchanged. Middle chunks of overlapping long prompts now run one after another and overlapping lanes lose tiled attention (they use vec-hybrid, as alone); only chunks that fit together (mostly tails and stable-boundary chunks) share a transaction. A lone long prompt is unchanged.
- ADR-063 DeepSeek-V4 gap program (S3, cause 1): DeepSeek-V4 prefill cohorts no longer require equal plans. Any prefilling lane whose next work is a matrix chunk of at least 33 rows joins at its own cursor and chunk length (each lane planned at the cohort's per-lane window cap; ineligible lanes are skipped, the oldest request always leads), so staggered agents and partial cold waves share the row-local FFN/MoE work; the cold wave's four-lane minimum and the staggered-lane alignment catch-up are removed. While decoders run, the Mixed slice may pack two lanes at one window each within the unchanged two-window budget (recovery tails still first). Before a pure-prefill transaction the worker waits at most 25 ms, once, while another request for the engine is still being rendered and tokenized (a new in-flight admission gauge); a lone request never waits; the gauge counts a request until the engine has queued it, and a summarize-on-overflow request stops counting while its nested summary generation runs. In a cold wave, a lane whose next work is its recovery tail waits while a cold peer still has matrix prefill work, so stragglers that miss the first cohort prefill (as one cohort) before the early lanes start decoding; the whole wave then decodes together, as before, instead of stragglers waiting behind the early lanes' entire answers in Mixed steps. Each lane's bits equal its solo route (Class A); the completion log reports every lane's own span (`spans=[(slot, start, rows), ...]`), and the agentic release gate's DeepSeek waves now require one cohort across unequal spans (`unequal_span_cohorts`) instead of an alignment catch-up.
- Qwen3.6 full-attention decode (ADR-063 Q4-QWEN36, per-row small ops) runs every row of a joint step in one pass on the TQ KV cache: one batched K and one V encode at the rows' cursors, one Q rotation over all rows' heads, every row's attention behind a single barrier (mlx-native `flash_attn_vec_tq_hb_rows`, each row still on its own scalar or GQA-Q2 kernel) written straight into the packed rows, and one fused output un-rotation + sigmoid gate. This replaces six barrier-separated dispatches plus a row copy per row; greedy serial decode (the single-command-buffer layer) takes the same pass at one row, while the sampled serial path and MTP keep the per-row chain (same bits). Each kernel computes a row from that row's inputs and slot with the previous per-row arithmetic, so outputs, KV bytes and cursors are bitwise unchanged at every width and row order (Class A; `qwen35::joint_attn_rows_tests`, mlx-native `test_qwen36_attn_rows`). The F32 control cache (`HF2Q_TQ_KV=0`) and a full slot keep the per-row chain.
  - For Qwen3.6's 16/2-head geometry, `HF2Q_QWEN_GQA_Q2=auto` now selects the cooperative GQA-Q2 attention kernel from KV 4,096 for one row and from KV 1,024 when rows share the pass. It was 8,192 before; other geometries keep 8,192. Both kernels give the same bits.
  - `HF2Q_QWEN_ATTN_ROWS=0` (documented escape; default on) restores the per-row chain and the 8,192 crossover.
- Qwen speculation (`auto`) is now repeatable run to run. Previously a speculation-capable slot ran joint one-token rows when it shared a decode turn and serial K3 or history verify rounds when alone, and the per-generation cost gates switched proposers off from measured wall time. Both depended on request timing, and the two routes differ numerically, so identical requests could answer differently.
  - Each tick's route (drain verified output, history round, K3 round, or ordinary decision) is now decided once from the request's own state.
  - A slot that speculates runs its round whatever the cohort; only ordinary ticks join joint steps.
  - The cost gates compare acceptance with fixed round costs (MTP 1.26, history 1.49 ordinary decisions, from existing receipts' controller counters) instead of wall time; `/metrics` timing stays wall-clock.
  - `HF2Q_QWEN_SPECULATION_ROUTE=adaptive` restores the previous behaviour.
  - The route is request-local wherever speculation is on: one slot, 2–3 slots, workers with joint decode off, and explicit `auto` at four or more slots. A speculating slot in a busy turn now runs its own round rather than sharing a joint step.
  - End-to-end repeatability is shown at four slots and with one request per slot. When requests outnumber slots, a reclaimed slot is now spilled and restored (#727, below), but a conversation evicted from the spill store or the LCP resume budget can still change prefix resume by arrival order; the resume budget no longer depends on the host (#730).
  - See `docs/research/adr063-qwen-spec-repeatable.md`.
- Unset `HF2Q_QWEN_SPECULATION` now resolves to `off` for Qwen3.6 (MoE) at four or more slots with joint decode, as it already did for Qwen3.8. Four-agent Qwen3.6 serving is then repeatable by construction (exact joint decode is bitwise equal to serial); `HF2Q_QWEN_SPECULATION=auto` opts back in. This is an operator ruling for repeatable serving, supported by #719's evidence that per-slot serial K3 at four agents costs about 25% against mostly-joint one-token rows.
- Qwen3.8 four-agent serving with speculation off (the default at four or more slots) is now repeatable run to run (#717). Under the padded `mul_mv_ext` joint mode, a slot decoding alone used to run the serial one-row forward, which is not bitwise equal to the joint row. Which steps ran alone depended on when the other agents' requests arrived and finished, so identical requests could answer differently on near-tie tokens. Now every eligible slot runs in padded joint steps of at most the joint width, including a lone slot whose next tick is ordinary decode, so a request's arithmetic no longer depends on which tenants share its turns. A lone slot with MTP or history speculation state keeps its serial speculative round (speculation `auto` is made repeatable by the entry above). One-slot workers and exact mode are unchanged. Tokens that previously ran serially now come from the padded joint row (Q1 Class B against the serial forward), so transcripts can differ from earlier releases. `HF2Q_QWEN_JOINT_SOLO=0` restores the serial forward (`docs/research/adr063-qwen-n4-solo-joint-717.md`).
- Gemma 4 batched prefill host overhead (ADR-063 Q4-GEMMA host P1/P2; host-only: every kernel reads the same bytes as before). The prefill activations (residual stream, Q/K/V, attention, MLP and MoE intermediates, positions, token ids) now come from a per-worker arena that keeps them across forwards, in the DeepSeek-V4 `PREFILL_POOL` pattern, instead of a fresh zero-filled, residency-registered allocation per buffer per forward. Buffers every layer fully writes before reading skip the zero-fill; buffers that can be read unwritten (`pf_moe_accum`, the cold-route `pf_sdpa_out`, the tensor-mm and DFlash xlen scratch) are still zeroed on every hand-out, and the `HF2Q_BATCHED_DUMP`, `HF2Q_S019_CKSUM` and imatrix-calibration paths zero all of them. The arena holds at most 2 GiB of power-of-two buckets, in use and free together, during a forward and between forwards: it releases buckets the forward is not using before growing past the bound, and requests that still do not fit use the previous allocation. A forward that fails mid-flight drains the command queue before its buffers are reused. The grammar mask's 262,144-entry `token_bytes` table is built when the engine is constructed, concurrently with the worker warmup, instead of on the first grammar request, which all turn-0 requests used to wait on.
- The in-memory LCP resume registry budget is now a typed operator setting with a host-independent default (#730). `hf2q serve --lcp-resume-budget <SIZE>` (or `[serve] lcp_resume_budget`; CLI wins) accepts bytes or units such as `8GiB`; omitted, it is a constant 8GiB. Serve resolves it once at startup, logs `[hf2q lcp] byte_budget=<bytes> (source=default|config|cli)`, and passes it to every Qwen and Gemma engine load, including model swaps; Gemma's per-request snapshot check uses the same value. Before, the default was 5% of free memory clamped to 1–16 GiB, read when each engine loaded and again on every Gemma request, so which prompts were snapshotted and evicted depended on the host and on load order. `/hf2q/v1/runtime` reports the value as `engine_config.lcp_resume_budget_bytes`. **Breaking:** `HF2Q_KV_LCP_RESUME_CAPACITY` is removed and no longer read; use `--lcp-resume-budget 8GiB` for `=8g`, and `N × 300MiB` for a legacy entry count `N` (`=8` → `2400MiB`). `serve_gemma4_opencode.sh` now passes `--lcp-resume-budget` (override with `LCP_RESUME_BUDGET`, formerly `LCP_CAPACITY`), and the LCP bench and soak scripts pass the byte form. Zero is rejected.
- Gemma 4 prompt-anchor capture host cost (ADR-063 Q4-GEMMA host P3; host-only, every anchor holds the same bytes as before). A capture (the full-input checkpoint after each warm forward, and every other Hybrid slot anchor) now copies the slot's sliding-ring buffers on the rayon pool instead of one after another on the worker thread. On a worker with the full-input checkpoint active, a dropped anchor's ring buffers (about 158 MB per slot on Gemma 4 26B-A4B) go to a per-worker pool, and the next capture overwrites them in full instead of allocating fresh memory every turn. The pool only keeps what fits the room the anchor quota leaves (live anchors plus pooled buffers stay within the 1.5 GiB quota), each new reservation frees any excess first, and refusals still depend on live anchors alone. Pooled buffers stay allocated between turns, so a worker keeps up to its planned two anchor generations per slot resident after a turn ends instead of one.
- Qwen3.6 (MoE) four-agent decode now runs joint decode by default, as Qwen3.8 does: every eligible decoding slot advances through one shared target forward per scheduler turn. On the declared Qwen3.6 repair-turn workload (ADR-063 Q4-QWEN36), the paired 24-instance task campaign against serial decode is `Pass`. The canonical n4 ABBA cuts makespan by a ratio of 0.800 (95% CI 0.776–0.824). A same-prefix fork check on the logprobs route finds joint and serial decode bitwise identical on all 32 recorded requests (`docs/research/adr063-s3-qwen-joint-decode.md`). `HF2Q_QWEN_JOINT_DECODE=0` restores serial decode.
- Build: hf2q builds cleanly on rustc 1.99, which deprecates `AtomicU64::fetch_update`. The five counters now use the equivalent `compare_exchange_weak` loop with unchanged orderings (its replacement, `try_update`, is newer than the MSRV).
- Gemma 4 cold multi-sequence prefill (the no-boundary cold batch and the installed long-prompt cursor-0 chunk) now runs attention and the O-projection per sequence through the single-slot cold attention (`gemma_cold_attention`; ADR-063 Q4-GEMMA-COLD decision D2) instead of BF16 flash attention over a block-diagonal mask. A lane above 32 rows is now bitwise equal to its single-slot cold prefill (Class A; before, the route was Class B `RouteSensitive`), so concurrent cold requests can answer differently than on the previous release, now equal to serial. The no-boundary cold batch's lane floor rises from 32 to 33 rows; graft-bound workers admit cold lanes per slot. For installed long prompts only the cursor-0 chunk forward is exact; the chunk plan still depends on how many lanes are co-installed. `HF2Q_DUMP_SLIDING_MASK` and the block-diagonal mask builder are removed.
- Release reliability: the protected sign job now checks the crates.io token and the Apple credentials before touching the candidate, so a dead credential fails before signing. Before, a bad crates.io token surfaced only at `cargo publish`. Release publishes via crates.io Trusted Publishing, falling back to `CARGO_REGISTRY_TOKEN` until that is configured. Publish no longer runs after a failed candidate build. A fresh dispatch replaces stale assets on a still-draft release, while published releases stay immutable. A candidate commit cut before this change is refused before the build. Release jobs and the CI `release`/`contracts` jobs are pinned to `macos-26`, and the candidate `build.json` (schema 2) records and verifies the macOS, Xcode, SDK and runner image. `scripts/release.sh` dispatches a release and re-runs the same run's jobs when a hosted runner is never acquired.
- PR CI now checks the packed crate (`cargo check --all-targets` in the unpacked `cargo package` root) and the Mach-O LINKEDIT string-pool alignment of the release build. Before, these ran only during a release.
- ADR-063 Q1 test harness (no serving change): the Qwen3.8 T1 endpoint now captures the source and restored 64-row active caches and checks them byte for byte, as Qwen3.6 already did. Qwen3.8 T1 vs T3 is now `ExactOnCommon` with exactly the 1,408 restored-64 records as extras (#664; `docs/research/adr063-q1-qwen38-rerun-b5.md`). The Qwen endpoint payload cap is 4 GiB, because the Qwen3.8 T1 store is 3.1 GB.
- Four-agent workload normalization (ADR-063, operator decision 2026-11-05, effective from the ADR-065 B1 baseline; harness only, no serving change): the model now sees the same `run_tests` result in every run. Before, the result carried cargo's absolute fixture path (run directory and agent id) and the cargo/libtest wall-clock durations, so a four-agent workload sent different prompts in every run. In Q4-GEMMA-COLD G-S2 run 011 those digits alone moved all four Gemma lanes from `src/ports.rs` to `src/ports.py`/`src/ports.rypts`, on every prefill route including a cold full prefill. `scripts/four_agent/qwen38_n1_tools.py` replaces the owned tool root with `/owned` and the cargo working directory with `/cargo-cwd`, and sets every duration to `0.00s`. It also drops the OS thread id that Rust ≥ 1.91 prints in failing-test panic lines (`thread 'name' (37675355) panicked at …`), keeping the test name, location and message. The receipt keeps the verbatim output as `raw_output`. Error codes and the libtest result marker are unchanged, and `read_file`/`write_file` results already were run-invariant. Tool-call ids stay as each engine assigns them: the chat templates only match results to calls by id and do not render it. Earlier results stay as recorded; cross-run transcript comparisons must not mix runs from before and after the change. `test_qwen38_n1_tools.py` covers it model-free.
- Qwen MTP speculation: the solo K3 tick (`decode_tick_mtp_k3`) now runs as entry, draft, solo verify, verify-to-plan and a split commit (cursors, DeltaNet restore, settle, finish) in `engine_qwen35/k3_transaction.rs`, so a joint verify window (ADR-065 J4) can reuse the same transaction (ADR-065 J1, #512). Behaviour-preserving refactor: the canonical accept walk, every failure arm and its error context, the `HF2Q_MTP_PHASE_PROFILE` timers, the cost controller and the Q1 test-hook positions are unchanged.

### Fixed

- Prepared Q4_K/Q6_K planes of the Qwen prepared routes (`HF2Q_REPACKED_GGUF_M32`, `_M32_PROJ`, `_M32_HEAD`, `HF2Q_QWEN_PREFILL_PREPARED`) are now published only after the command buffer that writes them has completed. They were recorded when the repack was encoded into the caller's command buffer, so a forward that failed between that encode and its commit, or whose command buffer failed, left all-zero planes cached for the life of the model, and later prefill chunks and verify steps silently read them. Each weight's repack now runs in a command buffer of its own (a dense FFN layer's gate, up and down share one) before the cell is set; a failed preparation leaves the cell empty and the next use prepares again. The cost is one GPU wait per weight on its first use: on the exact four-agent runs the first prefill chunk after load took 560-613 ms against 520-576 ms on Qwen3.6 and 2.43-2.48 s against 2.05-2.07 s on Qwen3.8, and steady-state cycles are unchanged.

- Tiled Qwen DFlash2 verify attention (`HF2Q_QWEN_VERIFY_ATTN_TILE=1`): a token's attention output no longer depends on the verify window that carries it. With mlx-native 0.21.8 the kernel cut a lane's history into splits by the pages of the whole window, so a window that crossed one page boundary in every 768 tokens of history from 1,536 on gave its rows before the boundary other bits than the same tokens get in a window that ends there (a review measured a logit delta of 0.85 at cursor 1,525). Window starts follow the wall-clock verify budget, so the text could change from run to run. From mlx-native 0.21.9 a token's split length is a function of its own page count and such a window runs as two tiles; elsewhere bits and cost are unchanged (−2 to +1% model free at 512 to 32,768 tokens, one and four lanes). hf2q's route code is unchanged. New `verify_attn_tile_window_tests` verifies 24 positions per lane around every such boundary below 9,000 tokens under eleven window plans (every row index 0-7, short windows, partial acceptance), one to four lanes, widths 8, 16 and 32, both served geometries: every observation of a token and the cache bytes are bitwise equal. It fails on mlx-native 0.21.8 at position 1,529, so it needs the 0.21.9 pin. Exact Q5 runs with every combined-candidate switch on: with the any-width switches (`HF2Q_REPACKED_GGUF_ANY_WIDTH=1`, `_M32_HEAD=1`, `_SHARED_HEAD=1`) seven of seven Qwen3.6 runs and two of two Qwen3.8 runs have identical transcripts (the 0.21.8 tile gave two transcripts in eight Qwen3.6 runs); without them five valid Qwen3.6 runs differ from each other from the first assistant message on, so this fix alone does not make the served text independent of timing.

- Qwen activation captures of short forwards read the wrong layer. A multi-row forward of at most 32 rows batches its per-layer GPU waits (`HF2Q_FFN_TERMINAL_K_BATCH`, default 8) and keeps the layer output in a two-slot ring, but the host copies of the residual stream inside the layer loop (the DFlash and DFlash2 target-layer capture, the activation-calibration capture and the calibration residual capture) did not wait for the GPU. At a layer that was not a wait boundary they returned an earlier layer's residual stream, and a one-row forward never waited at all. A captured layer is now a wait boundary, the layer's input is copied after that wait instead of before the layer, and a one-row forward drains the queue before the copy. For the Qwen3.8 DFlash2 draft this corrects target layers 5, 19, 33 and 61 (47 was a boundary) for every generation-cue chunk and every prompt chunk of at most 32 rows, 32 chunks of a four-agent run; chunks of more than 32 rows and the joint verify capture were already correct. Qwen3.6 (target layers 1, 6, 11, 16, 22, 27, 32, 37, none a boundary) had every captured layer wrong for such chunks. On the exact Qwen3.8 four-agent Q5 (two runs before, two after, identical 5,237 completion tokens) accepted rows per lane-cycle went from 5.330 to 5.370 (1,192 to 1,183 lane-cycles) and the cue chunk stayed at 72–75 ms.

- Q5 supervisor: a case control left at its empty default is no longer exported to the server or client process. The Qwen3.8, Gemma 4 and DeepSeek-V4 cases exported `MLX_PROFILE_CB=""`, and mlx-native up to 0.21.6 enables per-command-buffer profiling for any present value, so every Qwen3.8 run launched from those cases since 2026-11-09 ran with synchronous labeled commits and printed `[CB_PROFILE …]` dumps (188 retained Qwen3.8 runs). On the Qwen3.8 DFlash2 four-agent workload, the same binary with identical output (16 requests, 5,361 completion tokens) measured 72.072 and 69.694 seconds with the empty variable and 61.145, 60.267 and 60.089 seconds without it. Gemma 4 and DeepSeek-V4 production paths issue no labeled commits, so their runs carried the variable without that effect (source reading; not re-measured). Launcher variables are read as `${VAR:-}`, so their behavior is unchanged. Earlier Qwen3.8 receipts measured under this condition are not corrected by this change.

- Qwen3.6 DFlash2: `HF2Q_QWEN_MOE_STAGED_M8=1` together with `HF2Q_DFLASH2_GPU_TARGET_TOKENS=1` failed every request (`embedding_q4_k: token_ids[9]=4294967295 exceeds vocabulary`). The GPU verify-row mask marks unselected routes with the id `num_experts`; the staged-M8 compact schedule treated that as invalid metadata, poisoned itself, and the grouped M8 kernels then skipped the expert projections of every row in the step. With mlx-native 0.21.7 the schedule skips masked routes and the weighted reduce does not read their rows. On a development build the exact four-agent Q5 with both switches passed 10 of 11 runs (one task failure: an agent did not follow the workflow; the same build without staged M8 passed 11 of 11), and the build pinned to the published 0.21.7 passed 2 of 2. A six-block same-binary campaign against the masked native matvecs did not resolve a makespan effect (5.77% lower, 95% CI from 8.88% higher to 18.45% lower). The combination no longer fails; no speedup or quality qualification is claimed. The DFlash2 DEBUG stage-timing record now also reports the dispatches, command buffers, host waits, barriers and worker-thread CPU time of each cycle.

- The Qwen3.6 C2 calibration and divergent-edit cases now use the family’s declared one-repair-turn workload instead of the failing no-repair workflow. Case validation refuses future Qwen3.6 C2 cases that omit the repair grant.

- `scripts/serve_gemma4_opencode.sh` now checks whether the selected `HF2Q_BIN` supports `--lcp-resume-budget`: current binaries still receive the explicit 8 GiB default, while older frozen baselines preserve their built-in behavior instead of failing before model load.
- DeepSeek-V4 phase trace: a joint decode turn in which every remaining lane finished during sampling (EOS, stop string, max_tokens or a terminal tool call) no longer publishes a `joint_decode_input` transaction with no lanes. Such a turn commits no token, so it has no joint forward to attribute; any sampling work runs as before, outside the joint transaction. The empty transaction made `f2_trace.validate_trace` refuse the whole trace ("transaction cannot invent rows"); 3 of 990 transactions in the traced DS4 n4 run. The transaction now opens only when at least one lane commits a token. Serving output is unchanged.
- Gemma 4 (#728): a group of cached requests with an exact pre-generation boundary that arrived in one admission window could be prefilled as one shared multi-slot suffix body with tiled attention, which is not bitwise equal to the per-slot resume, so a request's output depended on the 25 ms admission coalesce window (`HF2Q_ADMIT_COALESCE_US`). Such groups are now admitted per slot in arrival order (a qualifying cold group still runs as the exact joint cold transaction), so for prompts up to 4,096 tokens the window only affects throughput (longer prompts become installed long-prompt lanes, tracked in #725). Gate: `gemma_stable_warm_group_is_independent_of_the_admission_window` (window 0 serial vs 100 ms concurrent, outputs and resumed anchors bitwise).
- DeepSeek-V4 prompt prefill no longer depends on the scheduler route (#726). Whether a prompt row ran as a matrix row or token-wise used to depend on the per-turn window cap that solo (2,048 rows), Mixed (256 rows while peers decode), cohort width (1,024/640/512 rows) and alignment catch-up pick. A chunk that left 1–8 rows before the recovery anchor replayed those rows token-wise, and token-wise rows differ numerically from matrix rows, so the same request could answer differently under load. A matrix chunk now never leaves a remainder below 33 rows: it shrinks so the remainder is exactly 33. The row class is now fixed by the request alone. Every row before the recovery anchor is a matrix row, unless the whole pre-anchor segment is below 33 rows at a nonzero cursor (then it is token-wise), and the anchor's last 8 rows are always token-wise. The serial `prefill_suffix` path uses the same rule. The route still moves matrix chunk boundaries. Matrix chunks of at least 33 rows are bitwise split-invariant, solo and in cohorts: model-free at real layer shapes on synthetic weights (`chunk_split_invariance_tests`), and on the release artifact across solo, Mixed (alone and with decoding peers) and cohort widths 2/3/4 (`official_artifact_prefill_route_boundary_is_route_independent`).
- Qwen SlotAware serving with more agent conversations than slots (#727): when a request is granted a slot that holds another conversation's resumable state, that state is no longer discarded. Admission copies the slot to host memory first: every full-attention and MTP row below its cursor (TQ codes and norms, or F32 K/V), the slot's region of both DeltaNet conv and recurrent ping-pong buffers with its parity, every cursor and the graft length, plus the retained-prefix and prompt-anchor ledger. When that conversation returns, the same longest-prefix rule slot affinity uses finds the spill; if it resumes more of the prompt than the granted slot does in place, it is restored into the granted slot and the request takes the same live-tail or anchor resume it would have taken in its original slot. Before, which conversation lost its prefix depended on arrival order, and cold prefill and live-tail resume differ numerically, so the order could change answers. The spill store holds at most `--slot-spill-budget` / `[serve] slot_spill_budget` bytes (default 16 GiB, `0` disables) and four entries per slot, never sized from available memory. The oldest spill is evicted first, an entry larger than the budget is not stored, and a restored spill leaves the store. A slot taken by an embedding request, or rewound to its anchor by a request that starts new content exactly there (a fork, or another conversation sharing that anchor), spills the same way. A conversation's own next turn or retry in its own slot resumes in place and copies nothing, and the store is searched only while it holds entries.
- Gemma SlotAware serving with more agent conversations than slots (#727, Gemma side; same store and `--slot-spill-budget` as Qwen). A request granted a slot that holds another conversation's published prompt anchor spills that slot first: the live KV scaffold's global rows below the cursor (extended to the anchor boundary) and its sliding rings, every scaffold's cursors (graft rows included), the retained prefix, and the anchor with its saved rings (parked without an anchor-quota lease, re-leased on restore). A returning conversation whose spill resumes more of the prompt than its granted slot does in place is restored there byte for byte and takes the same live-tail or anchor resume, also as a joint-warm cohort lane; the cold and joint-warm batch planners treat a spilled prefix as warm. A conversation whose own idle slot the shared KV budget refuses is moved into the granted slot in place (no host copy) instead of going cold; this also covers stable-boundary groups, which #728 admits per slot. Unlike Qwen, an in-place anchor rewind never spills a copy: a full-input checkpoint ends where the next turn diverges, so the copy would run on every turn. The official-artifact gates that would show the resumed output bitwise equal to the in-place run (`gemma_spilled_slot_resumes_bitwise_in_any_slot`, `gemma_resume_is_slot_independent_bitwise`) are pending.
- Qwen MTP drafts are deterministic. In the MTP block's attention output, the sigmoid gate multiply and the `wo` projection were dispatched in one concurrent encoder with no barrier between them. The projection could read partially written gated rows, so identical K3 rounds drafted different tokens. Committed output stayed target-verified, but the accept pattern, and with it which verify row committed each token, changed from run to run. A `memory_barrier` now orders the two dispatches, as on the target path (ADR-015 iter21). On both Qwen3.6 and Qwen3.8, the slot-independence gate's repeated K3-heavy request is now bitwise equal within a process and route-identical across processes.
- Gemma 4 serving no longer chooses between batched and linear cold prefill from free host memory (#738). The auto-route rule used to read `sysinfo` available memory on every request and engage the batched route only when its O(n²) scratch (masks, F16 mask casts, the tensor-mm `pf_kq` global scratch) fit in a sixth of it. That choice also decides LCP resume vs. cold batched prefill, and the routes are different arithmetic, so the same prompt could produce different bytes depending on model residency, KV growth, other processes and other requests. The budget is now fixed once at model load as `(total physical memory − model file bytes) / 6` and logged at load (`batched_route_scratch_budget_bytes`), so the route depends only on the prompt length, the model and the process configuration. On a 128 GiB host with the 26B Q4_K_M model the default batched envelope is about 16K tokens (about 50K with `HF2Q_GLOBAL_FA=1`). Other models that a multi-model pool keeps resident are not reserved; such servers can set `HF2Q_SERVE_BATCHED_PREFILL=0` to force the linear route (`=1` still forces the batched route).
- Serving route choice no longer depends on model-load order (#729). Loading a Qwen3.8-identified model used to call `set_var` for `HF2Q_DECODE_MVN=0` and `HF2Q_DECODE_MV_EXT=1`, which changed the routing policy for the whole process: a model loaded afterwards (Gemma, DeepSeek-V4, Qwen3.6) could inherit the non-exact `mul_mv_ext` route, and DeepSeek-V4 and Qwen workers dropped joint decode mid-run, depending on load order and on when mlx-native first cached the variables. Now the Qwen3.8 default is resolved once per model at load, from the model's identity and only explicitly exported operator values, and frozen into that model's kernel registries. Qwen's thread-local GPU cache only matches a model whose policy equals its registry's frozen one, so a policy cannot leak between models that share a worker thread. Every other model (each `GpuContext`) freezes the operator-environment policy. Qwen joint decode, the DeepSeek-V4 joint check and Qwen trace metadata read the model's policy instead of the environment, and the per-turn Qwen joint re-check no longer looks at routing variables, which cannot change mid-run. The Qwen3.8 padded joint gates bind the policy `hf2q serve` resolves, so they run with no routing variables exported (the artifact path must name Qwen3.8).
- `scripts/serve_qwen38_opencode.sh` no longer exports `HF2Q_DECODE_MVN=0` / `HF2Q_DECODE_MV_EXT=1` (#729). Exported values are now process-wide operator overrides, so the launcher's defaults would have imposed the Qwen3.8 route on every model in the process; the Qwen3.8 model now selects it for itself at load. The launcher still validates operator-exported values (`0` or `1`) and prints `per-model default` when they are unset.
- Qwen3.5 / Qwen3.6 / Qwen3.8 full-attention RoPE (with the mlx-native 0.15.18 `rope_multi` fix): partial rotary (head_dim 256, rotary_dim 64) now pairs `(p, p + 32)` and passes dims 64..255 through, as the reference does. Before, the kernel paired `(p, p + 128)`, which moved layer-3 Q/K after RoPE to rel-L2 0.60/0.54 against the reference and every later full-attention layer with it. The CPU oracles in `full_attn.rs` and `gpu_full_attn.rs`, which encoded the old pairing, are corrected, and `imrope_matches_cpu_ref` now covers the 256/64 production shape up to position 262143. Persisted Qwen3.5/3.6 KV snapshots written before the fix hold K rotated the old way; the persistor fingerprint now binds the RoPE convention, so they are no longer hydrated. The lazy tensor-map loader (activation capture from a lazy map) set `rotary_dim = head_dim/2`; it now uses the family's 0.25 factor and sections `[11, 11, 10, 0]`.
- Qwen K3/history speculative verify no longer fails with `KernelNotFound` when the precompiled metallib is disabled (`MLX_PRECOMPILED_METALLIB=0`). hf2q builds its registries with `KernelRegistry::new()` and never called `ops::gated_delta_net_decode::register` or `ops::ssm_conv::register`. That left the gated delta net and conv-state capture kernels (`gated_delta_net_decode_capture_f32_{1,2,4}`, `ssm_conv_capture_forward_f32`) with no source on the runtime-source path. `mlx-native 0.15.18` gives `KernelRegistry::new()` a runtime source for every kernel that only an op's `register` used to supply (54 kernels, including the vision `bilinear_resize_2d`, `block_merge_2x2` and `im2col_2d_3ch` kernels hf2q dispatches without registering them), plus the IQ4_XS matmul kernels. The default precompiled path is unchanged, and an explicit `register` call still selects runtime source as before. Found by the ADR-065 E6 probes.
- Qwen3.6 and Qwen3.8 `tool_choice=auto` tool calls: the lazy tool grammar now admits the chat template's own encoding, `<tool_call>\n<function=NAME>\n…</function>\n</tool_call>`. Before, the grammar woke on `<tool_call>` and required `<function=` next, so it masked the template's newline. The #662 fork replay showed hf2q emitting `<tool_call><function=` in 192/192 outputs, while the peer emitted the native form in 64/64. The newline is now required rather than optional. A zero-argument call must be `<function=NAME>\n</function>`, as the template renders it. Auto multi-function parallel calls join with `\n<tool_call>` (Qwen3-VL too), so no call needs a doubled newline. Unary and SSE parsing accept both the native and the old compact body. The new `scripts/test_qwen_tool_call_native_wire.sh` runs the shared Qwen agentic gate, then a three-turn `read_file` session that replays `reasoning_content`. It also runs an Auto multi-tool parallel request and a zero-argument tool under `auto` and `required`. It checks the generated tool-call tokens through logprobs, plus SSE validity (parallel indexes 0 and 1), tool-result continuation and prefix reuse, and fails on `finish_reason=length`.
- Gemma release gate `gemma_n8_decode_then_tiny_cold_prefill_is_repeat_invariant` (test-only, #702): the retained-suffix resume check now compares against a fresh N=1 model that runs the same base prefill and tiny resume, instead of a cold full-prompt prefill. The two are different numeric paths. The cold prefill runs batched flash attention over in-flight F32 K/V, while the tiny resume reads the slot cache: F16 K and TQ8 V in Hybrid, TQ K and V in full-TQ. On the Gemma 4 26B-A4B q4_k_m GGUF, the test's control/newline prompt is a near-tie, so the two paths pick `"\n\n"` and `"\n\n\n"` respectively. That failure was deterministic and reproduces on v0.1.29, so it was never a regression. The live resume after width-two decode rows must still match the same-path reference exactly.

### Added

- Diagnostics: traced four-agent Qwen runs now witness joint decode forwards (ADR-063 Q4-TRACE). Each joint ordinary decode forward publishes one source-bound input manifest (`qwen35.joint_decode.actual_slot_rows`, padding rows declared owner-less), one operator witness at its first DeltaNet pre-norm, and a `forward_input_end` marker that separates the forward's own native commands from the per-lane sampling and MTP catch-up of the same turn. Before, every DeltaNet pre-norm of a joint forward was recorded as an unmapped operator attempt (43,008 in one traced n4 run). The phase-trace reducer binds each joint input to its transaction, reports joint forwards (owners, logical and padded width, forward and settle spans) under `joint_ordinary_decode`, and keeps them out of the Q2 v2 window counting. The hook only runs with a native diagnostic trace armed; untraced serving does two atomic loads per joint forward and is otherwise unchanged. New tools `q4_trace_verdict.py` (acceptance, Q2 isolation, pre-registered exactness rule) `q4_trace_overhead.py` (traced overhead extractor) and `q4_trace_attribution.py` (per-phase and per-pipeline attribution of a traced run).

### Removed

- `HF2Q_GEMMA_TILED_LIVE` and the Gemma `Tiled` live attention route it switched. After #728 (stable-boundary groups are admitted per slot) and #725 (installed long-prompt lanes take the single-slot warm route), no serving path selected it. Every live resume, single-slot or multi-sequence, now resolves one route from `HF2Q_GEMMA_WARM_TILED_ATTENTION`, and `forward_prefill_batched_multi_seq_live` no longer takes an attention-route argument. Setting the variable has no effect.

## [0.1.29] — 2026-11-03

### Changed

- Gemma 4 tiled F16 warm attention is now off by default (`HF2Q_GEMMA_WARM_TILED_ATTENTION=1` opts in). Under the ADR-063 quality gate, a replay of the canonical four-agent task showed the tiled route moving a follow-up turn's decision log-probabilities from −0.149/−1.978 to −0.368/−1.179 against vec-hybrid and changing the code the agent wrote, solo and in a four-agent cohort alike. Warm prefills now use vec-hybrid attention; the four-agent speed effect is re-measured in the Q3 campaign.
- Gemma 4 full-input checkpoint is now on by default for SlotAware Hybrid/TQ8 workers with 1–4 slots (ADR-063). A tool-result turn resumes from the slot's checkpoint of the previous full input instead of re-prefilling everything after the 458-token system/tools anchor. The worker activates it only when the provisioned geometry holds two CPU anchor generations per slot within its own 1536 MiB quota (each Gemma worker has a separate quota). Any anchor capture refused by the quota keeps the previous anchor and serves the request unchanged. The route is Q1 Class B: transcripts differ from `=0`, and quality is gated by the paired campaign against stock. `HF2Q_GEMMA_FULL_INPUT_CHECKPOINT=0` is the kill switch; `=1` forces it and refuses startup when unsupported; `HF2Q_UNSAFE_EXPERIMENTS` is no longer required. The SlotAware warmup also primes the slot-mounted Gemma prefill route in a clean idle slot (skipped with `=0`; a non-fatal failure only logs). The Q5 `gemma4` case exposes `HF2Q_Q5_GEMMA_FULL_INPUT_CHECKPOINT` for A/B runs. Evidence and gates: `docs/research/adr063-gemma-four-agent-reuse.md`.

- Gemma 4 warm tool-turn prefills that arrive in the same admission quantum (all slots idle, including the 25 ms coalescing window) now run as one joint transaction instead of one inline prefill and decode-step yield per slot (ADR-063 Stage 1). Every lane must resume a reusable prefix (full-input checkpoint or retained slot) with a suffix longer than 32 tokens, and the cohort is capped at 4,096 rows; other requests keep per-slot prefill. Projections, norms, FFN/MoE and the output head run once over all suffix rows, and attention runs per slot through the same per-sequence function, kernel, chunking and ring staging as a single-slot warm prefill (see the tiled warm attention entry below), so each lane's tokens, logprobs and KV bytes are bitwise identical to its per-slot prefill (gated by an ignored official-artifact test). Checkpoint capture and publication stay per lane, in FIFO admission order. A submitted non-fatal failure restores every cohort lane from its rollback anchor and re-prefills it per slot, so only a lane whose own prefill fails is failed; a non-fatal per-lane publication failure in any multi-lane Gemma prefill is now request-local instead of stopping the worker. Resumed multi-lane Gemma prefills (joint warm, installed long-prompt resumes and both stable-batch passes) now reserve decode capacity for the binding lane's own `end + max_tokens` instead of pairing the largest `max_tokens` with the longest lane, which could fail a valid cohort near the context limit; cold multi-lane prefill keeps its reserve because it sizes the dense handoff for the whole concatenated stream. The multi-sequence live path no longer builds unused block-diagonal masks. `HF2Q_GEMMA_JOINT_WARM_PREFILL=0` restores per-slot warm prefill, and the Q5 `gemma4` case exposes `HF2Q_Q5_GEMMA_JOINT_WARM_PREFILL` for A/B runs. Design and gates: `docs/research/adr063-gemma-four-agent-reuse.md`.

- Gemma 4 warm prefills (every single-slot prefill that resumes a cached prefix, and every lane of a joint warm cohort) now attend with tiled F16 flash attention instead of the decode-style vec-hybrid kernel (ADR-063 Stage 2). One shared per-sequence function serves both: it compacts the slot's K and dequantizes its TQ-HB V range to F16, then runs d256 flash attention with an SWA mask and tile skip for sliding layers (1,024-query chunks) and the d512 resume kernel for global layers (4,096-query chunks), one sequence per dispatch with no block-diagonal mask, so joint cohorts stay bitwise equal to per-slot prefill. Suffixes under 32 tokens keep vec-hybrid. This changes warm-prefill numerics once against the previous route: last-row logits differ by up to about 4 logits, and the top-1 token changes only at near ties. That makes it Q1 Class B; on the paired quality campaign against the kill-switch arm it passed, 21/24 instances against 20/24. On the four-agent workload (M5 Max, 4 runs per arm), joint warm prefill fell from about 3.6 s to about 2.4 s per run and median makespan fell from 14.81 s to 13.66 s (−7.8%; the peer measured 12.39 s). `HF2Q_GEMMA_WARM_TILED_ATTENTION=0` restores vec-hybrid warm attention; the four-agent launcher forwards `WARM_TILED_ATTENTION=0|1` and the Q5 `gemma4` case exposes `HF2Q_Q5_GEMMA_WARM_TILED_ATTENTION`. Installed long-prompt lanes keep `HF2Q_GEMMA_TILED_LIVE`, and now get three fixes. Their sliding-window tile classifier no longer runs concurrently with the mask fill it reads; before, it could classify an unwritten (zero) mask tile as fully attended and let queries see future or out-of-window keys. With a bound KV graft, their global layers use the graft-shifted causal offset. With F16 V storage, their sliding layers no longer fall back to vec-hybrid with mismatched stage positions. Every live resume (both routes) now also orders its sliding-ring stage copy after the head norm/RoPE that writes the suffix K/V.

- Gemma 4 SlotAware cold prefills no longer build the SerialFifo-only outputs of the batched prefill, which nothing on the slot route reads: the LCP registry snapshots (dense F32 K/V plus hybrid K/V, sized `seq_len + max_tokens` rows for every layer) and the request-local dense K/V decode handoff. At the four-agent turn-0 shape (458-token boundary, `max_tokens` 8192) each cold slot admission was allocating and zero-filling about 5.4 GB of snapshots plus a 775 MB handoff, and the worker kept the last snapshot resident (about 5.4 GB). That accounted for about 0.28 s of each roughly 0.5 s admission. The cold multi-sequence prefill drops the same discarded handoff. The SerialFifo route, and every logit, KV byte and checkpoint on the slot route, are unchanged (gated by an ignored official-artifact test against the previous route). Expected turn-0 saving is about 1.1 s, not yet measured. Details: `docs/research/adr063-gemma-four-agent-reuse.md`.

- DeepSeek-V4 four-agent decode now advances every eligible decoding slot (2–4 rows by default, unequal positions, streaming and unary) through one layer-major forward per token step instead of serial per-slot quanta. Row-parallel projections, FFN and output head dispatch once per step; cache-dependent attention stays per row. Each row is bitwise identical to serial decode on the official artifact. Joint decode engages only when every batched product of the loaded artifact uses a proven-exact storage type (Q8_0, Q2_K or Q3_K; float weights only where split per row; grouped output-A only as Q8_0); other artifacts keep serial decode and log the reason at worker start. On the four-agent C262144 coding workload (M5 Max, 4 runs per arm), median makespan fell 25.9% from 141.7 s to 105.0 s (1.35× throughput; geometric-mean time ratio 1.335×, 95% CI 1.258–1.417×), and per-request decode rose from 8.1 to 11.0 tok/s. `HF2Q_DEEPSEEK_JOINT_DECODE=0` restores the serial path. `HF2Q_DEEPSEEK_JOINT_DECODE_MAX_ROWS` (2–8) widens joint steps beyond 4; those widths have no serving measurement.
- Qwen four-agent decode can advance every eligible decoding slot (2–4 rows by default, unequal positions) through one shared target forward per scheduler turn instead of one forward per slot. Projections, the dense FFN and the output head dispatch once per turn; DeltaNet recurrence, KV writes and attention stay per slot, and MoE FFNs run per slot. Joint rows decode with the ordinary target: MTP drafting and history proposals are suppressed for them, and slots with MTP state run their MTP catch-up so a later solo turn resumes K3. With fewer than two eligible slots the turn stays serial. Under the default routing policy each joint row is bitwise identical to its serial target forward and independent of its cohort; per-request output can still differ from stock because joint rows do not speculate. Under the Qwen3.8 `mul_mv_ext` policy every joint step is padded to exactly 4 rows (joint width capped at 4); this is an ADR-063 Q1 Class B route against stock (decision envelope, not bitwise), and in the A/B 2 of 114 same-request transcript groups differed from stock. On by default for Qwen3.8: on the four-agent C262144 coding workload (M5 Max, ABBA, 4 runs per arm, 8/8 `CalibrationPass`) median makespan fell from 116.4 s to 90.1 s (22.6% less time; geometric-mean time ratio 1.316×, 95% CI 1.25–1.38×), per-request decode medians rose from 8.9–9.7 to 12.2–12.5 tok/s, and client semantic TTFT medians fell from 9.0–9.7 s to 7.0–7.5 s. Off by default for Qwen3.6 (MoE): its forward is bitwise exact, but the incomplete A/B (2 runs per arm) failed the coding protocol on both joint runs; `HF2Q_QWEN_JOINT_DECODE=1` opts in. Artifacts with unproven storage types stay serial and log the reason at worker start. `HF2Q_QWEN_JOINT_DECODE=0` restores serial decode; `HF2Q_QWEN_JOINT_DECODE_MAX_ROWS` (2–8, padded mode capped at 4) sets the width ([ADR-063 S3](docs/research/adr063-s3-qwen-joint-decode.md)).
- Qwen3.8 joint decode (padded `mul_mv_ext` mode, `HF2Q_DECODE_MV_EXT=1`) now reads the dense FFN gate and up weights once per step instead of once per packed row. The serial builder's fused gate/up/SiLU kernel launches one threadgroup row per input row, so at the fixed padded width of 4 it streamed each layer's Q4_K gate and up weights (2 × 50.1 MB) four times, padding rows included. Joint steps now issue gate and up as separate packed products on the same width-4 routes the attention projections use, followed by `silu_mul`; the down projection and output head were already shared. Per joint step this cuts the weight bytes the kernels read on the official Qwen3.8 artifact from 35.06 GB to 15.81 GB. On the four-agent C262144 coding workload (M5 Max, ABBA, 4 runs per arm, same binary against `HF2Q_QWEN_JOINT_FFN_SHARED=0`), median makespan fell from 91.4 s to 87.8 s: 4.9% less time, with every block favouring the change (block-paired 95% CI 2.2–7.6%; unpaired 0.3–9.6%). Median per-request decode rose from 12.19 to 13.21 tok/s (+8.4%, 112 requests per arm), and token-weighted decode from 12.15 to 13.04 tok/s (+7.3%). That is far below the up-to-2.2× a pure bandwidth model projected, so joint steps are not purely weight-bandwidth bound. The route is part of the padded mode's Q1 Class B contract: it is not bitwise equal to the fused kernel, and 2 of 114 same-request transcript groups differed between the two routes. Cohort invariance (both routes), the exact-mode bitwise gate and the padded decision-parity diagnostic passed; the real-prompt Q1 envelope and the paired quality campaign against stock are still open. Exact mode and serial one-token decode are unchanged. `HF2Q_QWEN_JOINT_FFN_SHARED=0` restores the per-row fused kernel ([ADR-063 S3](docs/research/adr063-s3-qwen-joint-decode.md)).
- Qwen speculation is now off by default for dense artifacts (Qwen3.8) on SlotAware workers that selected joint decode and have four or more slots (ADR-063 D1, #364). An unset `HF2Q_QWEN_SPECULATION` now resolves once at worker start from the slot count, the family and the joint-decode decision, and the worker logs `Qwen speculation worker policy` with the requested value, resolved policy, `max_slots`, `moe`, `joint_decode` and the reason. SerialFifo, workers with fewer than four slots, workers whose joint decode is off or unavailable, and MoE artifacts (Qwen3.6) keep `auto`. On the four-agent Qwen3.8 coding workload, the `auto` vs `off` own-output makespan ratio was 1.020 (95% CI 1.0095–1.0308; client elapsed 1.020, CI 1.0093–1.0306; 8 adjacent ABBA pairs, `auto` slower in every pair; off-arm transcripts `ExactTranscripts`). Other dense Qwen artifacts with joint decode at four or more slots get the same default; only Qwen3.8 was measured. Receipts: `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023/d1-speculation-ext-v2`; analysis in `docs/research/adr063-d1-speculation-qualification.md` §7. Explicit `HF2Q_QWEN_SPECULATION=auto` (launcher: `QWEN38_SPECULATION=auto`) opts back in; explicit values always win.

### Added

- ADR-063 F2 scoreboard tier "Q3 verdicts of record" (`scripts/four_agent/f2_q3_tier.py`, `f2_q3_view.py`; F2 closure option B, operator decision 2026-10-31/11-01). Above the Descriptive results it shows, per family:
  - the reviewed Q3 verdict (Win, Loss, Unavailable or Failed);
  - the makespan effect of `hf2q <commit8>` vs v0.1.28 stock, and the task-makespan sentence vs the peer (X = 1 − faster/slower, computed on the slower arm's time);
  - worst-slot TTFT p95 and SSE-gap max ratios. An SSE gap the verdict could not estimate shows as not measurable and does not gate; against the peer, SSE gaps show as not comparable;
  - token volumes and output rate per arm;
  - the quality gate per reference (Pass, Pending, Halted or Fail) and any final sources pending patch review;
  - the campaign status with any halt, precision shortfall and the measured/unmeasured control stages, and the verdict and review references.
- ADR-063 native oMLX arm for Qwen3.8 27B, Qwen3.6 35B-A3B and Gemma 4 26B-A4B (`scripts/four_agent/native_peer_omlx.py`, `native_peer_omlx_ready.py`, `native_peer_omlx_launch.py`, `native_peer_omlx_audit.py`; cases `q5_cases/native-omlx-{qwen38,qwen36,gemma4}.json`). It runs oMLX 6b413c90 (0.7.3rc1) on pinned mlx-community 4-bit snapshots as a separately labeled different-artifact arm. oMLX silently drops unknown request fields, so the client enforces a closed request and reply envelope. It sends hf2q's controls explicitly: greedy sampling, repetition penalty 1.05 over 64 tokens, Qwen thinking with the 2048/512 budgets, and Gemma thinking off. A launcher writes the run's own `model_settings.json`/`settings.json` (hot-only prefix cache, no SSD tier), links the snapshot as the only model and records the launch. Identity is bound through:
  - the clean detached checkout and the venv's mlx/mlx-lm/mlx-vlm pins;
  - the snapshot revision and its config/template digests;
  - the launch record, `/api/status` (custom kernels), `/v1/models/status` and the saved settings;
  - a per-turn `/v1/tokenize` rendered-prompt proof, with session counters equal to the client receipts at closing;
  - a post-run audit matching the server's completion lines to the client turns.

  Q3 runs it only in a `--native-comparison` campaign; `quality_campaign.py` accepts it as `--reference-kind omlx-native`. The native-arm registry now declares family scope as tuples (`FAMILIES`, `Q3_FAMILIES`). This tooling establishes no speed result.
- ADR-063 Q2 traced cohort tooling (`scripts/four_agent/q2_traced_run.py`, `q2_traced_freeze.py`). `q2_traced_run.py` runs one supervised attempt of the Qwen3.8 four-slot case with `HF2Q_PHASE_TRACE_CONFIG` set (`native_command_capture_v3`, 6 GiB server and 128 MiB client budget, 8,000,000 native events). The population is `n4` or a Q3 control (`n1`, `3+1`, `2+2`, `1+3`). Client and trace share one run id. Before launch, the identity must match the binary, build receipt, approval template, model receipt and mlx-native pin. The run keeps every agent's test executables, and it writes the certify-shaped `receipt.json` (with its cleanup proof) and `readiness.json`. `q2_traced_freeze.py` freezes the expected plan and manifest input for an n4 run. It also validates and qualifies each population's pre-registered windows offline. This tooling establishes no Q2 result; a run is pending root execution, certification and independent QE.

  "Win" appears only vs stock, and only when the stock contrast, stock fairness, stop, thermal, memory and patch-review conditions hold and the quality gate passed against the stock and peer references (ds4 native for DeepSeek-V4). `q3_campaign.py --rows --of-record` (with `--quality-gate REFERENCE=MANIFEST` and `--review`) emits these rows plus the record's completed controls (`n1`, mixed) as controls of that verdict. A settled gate must run its reference on the campaign's arm identity (the v0.1.28 binary, the peer binary, or one native readiness identity). `f2_scoreboard_render.py --verify` re-derives each record from `verdict.json`, `campaign.json` and the quality manifests, and validation refuses a default fragment merged beside its record. The scoreboard's stock arm is now v0.1.28 (release `9a0b79c7`; it was mislabeled v0.1.27), and Q3 experiment-vs-stock rows use baseline `v0128_stock` instead of `hf2q_current`. Q3 row variants name the experiment commit instead of "main HEAD". A Q3 campaign with a native peer arm (ds4, TensorFold) now emits only its native rows (plus control failure rows), never an experiment-vs-stock row. This tooling establishes no speed result.
- ADR-063 native TensorFold arm for Qwen3.8 27B (`scripts/four_agent/native_peer_tensorfold.py`, `native_peer_tensorfold_ready.py`, `native_peer_tensorfold_audit.py`; cases `q5_cases/native-tensorfold-qwen38.json` and `native-tensorfold-qwen38-dflash2.json`). It runs TensorFold 1157ea5b on its own MLX 4-bit checkpoint, without a drafter or with DFlash2, as a separately labeled different-artifact arm and as the D2 T0/T2 arms. The client sends hf2q's Qwen3.8 controls explicitly (greedy, thinking at effort xhigh, the 2048/512 thinking ceilings) and enforces a closed request/reply envelope, because TensorFold ignores unknown fields. Identity is bound through:
  - the binary sha256, the pinned commit and version, and the exact server argv;
  - the HF snapshot revisions, with TensorFold's `config_sha256`/`tokenizer_sha256` recomputed locally;
  - `/health`, `/v1/models` and the startup log;
  - a per-turn `/tokenize` rendered-prompt proof;
  - a post-run audit binding each reply to the server's own `done` line.

  Q3 runs it only in a `--native-comparison` campaign, whose verdict and rows are labeled "not the verdict of record". A new registry, `scripts/four_agent/native_peers.py`, replaces the per-consumer ds4 branches; ds4 behaviour is unchanged. This tooling establishes no speed result.
- ADR-063 Q1 numeric policy (`docs/research/adr063-q1-numeric-policy.md`), approved by the operator on 2026-10-30, and the offline `scripts/four_agent/q1_decision_policy.py` checker. The checker reads retained Q1 trace stores, logprob responses, task outcomes and four-agent client runs. It reports bitwise equality, a logit decision envelope, near-tie fork classification, paired task-quality verdicts and transcript equality, and never loads a model.
- ADR-063 Q3 three-arm campaign driver (`scripts/four_agent/q3_campaign.py`, rules in `docs/research/adr063-q3-campaign.md`). It runs stock hf2q, experiment hf2q and the peer's best valid configuration in Williams-ordered blocks through the Q5 supervisor, one runtime at a time. Binary, model, peer and harness identities are frozen at campaign start. Blocks run until the 95% interval of the makespan ratio is within 2.5 points (5 to 20 blocks). The driver reports per-family `Win`, `Loss`, `Inconclusive`, `Unavailable` or `Failed` verdicts and F2 scoreboard rows. The four-agent client gains `--phase-plan` for the Q3 one-agent and mixed-phase (3+1, 2+2, 1+3) controls; without it the canonical workload is byte-identical. This tooling establishes no speed result.
- ADR-063 Q2 adversarial controls (`scripts/four_agent/test_q2_adversarial_controls.py`, matrix in `docs/research/adr063-q2-adversarial-controls.md`) and a settled-run audit (`scripts/four_agent/q2_run_audit.py`). The Q3 and quality campaigns no longer take a client's Pass at face value. Each claimed Pass is re-derived from the archived requests and responses (hash, canonical wire, parser replay, lineage, unique response ids, v3 timing, makespan arithmetic, cohort membership, cleanup). Q3 additionally requires all active agents to be co-active (streamed requests active together over a sustained common window), so a deliberately serialized or slot-capped service is rejected for n4 and for the 3+1, 2+2 and 1+3 controls. Contradicted runs become terminal `Invalid`. The client now records `progress_event_monotonic_ns` (reasoning or semantic SSE frames). It also rejects an 8192-token response labelled `stop` or `tool_calls` as silent truncation, and truncated tool-argument JSON. The controls run against a model-free fake service and establish no model, speed or hardware result.
- ADR-063 D1 qualification gates for exact Qwen warm-turn speculation (`docs/research/adr063-d1-speculation-qualification.md`). Model-free tests drive the production seeding, verifier-row selection, block planning and commit paths over a toy target. They cover full, partial and zero acceptance followed by eight ordinary target decisions, chained rounds, EOS inside or after a verified block, the length boundary, the sampling gate, and cancellation while verified output is queued (target and MTP cursors rewind to the prompt anchor; no speculative boundary is published). A policy test pins the bound-graft MTP fallback at admission and at prompt catch-up. An ignored Qwen3.8 engine test (`HF2Q_QWEN38_GGUF`) compares `auto` against `off` at four slots, including a cancelled stream; it accepts only Q1 near-tie divergences. The two existing graft/MTP gate predicates moved into `qwen35_speculation.rs` unchanged. Serving behaviour is unchanged; the four-slot default is decided by the root A/B in the research note.

### Fixed

- Gemma 4 no longer writes U+FFFD into tool-call arguments when it samples a lone byte-fallback token (#553). On the i13 quality-gate turn the model's top choice after `(1-3` was `<0x82>` (id 368), a bare UTF-8 continuation byte. The grammar's per-token byte table came from `decode(&[id])`, which renders every lone byte token as the valid character U+FFFD, so the tool grammar accepted it and the file got `(1-3�2)` in 12 of 12 runs. The table now holds each token's raw bytes (`<0xNN>` byte-fallback and byte-level BPE pieces; all other tokens are unchanged), and the grammar accepts only well-formed UTF-8 (Unicode Table 3-7: no surrogates, overlong forms or values above U+10FFFF). A token ending in an incomplete UTF-8 sequence survives the grammar only where some completion of that character can match; an excluded set (`[^…]`) rejects it only when it excludes every such completion. Lazy-grammar trigger patterns still fire when the output before the trigger holds an invalid byte: they match a same-length copy of the buffer in which each invalid byte reads as one ASCII placeholder. The i13 replay writes `(1-32)`.
- Gemma 4 and Qwen text output no longer splits characters that the vocabulary spells over several byte tokens. Every Gemma and Qwen unary, streaming and slot decode path holds back an incomplete UTF-8 sequence until its last byte arrives, flushes it at end of generation and drops it after a stop string. On main, `𓀀 ꙮ 𐌰` (and, on Qwen3.8, `龘靐齉` and `🦀🧬`) came back as one U+FFFD per byte in content, SSE deltas and tool arguments. They now arrive exactly, and a JSON-schema or tool `enum` value containing an emoji (`🦀 crab`) can be generated (Qwen3.8 previously could only pick an ASCII value). Logprobs `bytes` now report raw token bytes.
- Gemma 4 decode is run-to-run deterministic again. `mlx-native 0.15.16` adds the missing threadgroup barriers in the decode MoE routing kernels and the fused two-reduction end-of-layer RMS norms. Their shared-memory write-after-read race occasionally rescaled half the experts before top-k (choosing a different expert set) or mis-normalized a slice. ADR-063 Q1 caught it: a repeated Gemma endpoint run diverged in 1 of 12 runs (4 of 6 under stress), and 15 of 15 stressed repeats are bit-exact after the fix.
- Keep the Qwen tool-continuation thinking ceiling (default 512) at every tool-chain depth. Halving it to a 256-token floor from the second tool result forced the reasoning close inside the first edit plan of an ordinary read/read/write coding workflow.
- Preserve Qwen tool-call parameter whitespace in unary and streaming responses. Only the single newline the native template wraps around each `<parameter>` value is removed; a file's trailing newline and leading indentation now reach the client intact.
- Order Gemma 4 cold batched-prefill attention masks before their readers. The mask builders allocate and fill the mask on the GPU inside the call, and that write was never declared to the session's barrier tracker, so the F16 mask cast and the tile-skip classifier could run concurrently with the fill on cold prefills (single-sequence and multi-sequence). A classifier reading a partly filled mask can mark masked tiles as attended: future or out-of-window keys, or another sequence's keys in a multi-sequence cold batch. Every mask fill is now declared to the tracker, which places a barrier before its first reader.
- Stop refusing Gemma 4 multi-sequence cold prefills whose concatenated length exceeds the longest sequence plus `max_tokens`. Full-layer capacity is planned per sequence and each sequence is written into its own slot, but the bound compared the whole concatenated stream against it. Each sequence is now checked against its own slot capacity.
- Gemma 4 multi-sequence cold prefill now writes only each sequence's last 1,024 rows into its sliding-window ring, as single-slot prefill does. A cold batch may admit prompts up to 4,096 tokens, and writing every row of a longer prompt let rows 1,024 positions apart race for the same ring row, so the conversation could continue decoding from a ring holding stale positions.
- Tool-call ids are now unique per process. They were derived from wall-clock nanoseconds, or from a counter that restarted at 1 for DeepSeek-V4, and collided between concurrent agents (18 of 547 retained four-agent runs). Ids keep the `call_hf2q_<16 hex>` format and come from one shared counter offset by a random per-process salt, so ids from a restarted server do not repeat ids already in a client's history.
- Gemma 4 `reasoning_content` no longer includes the `thought` channel label. An empty thought block (which the model writes after each tool result) now yields no `reasoning_content`, in both unary and streaming responses. This matches the tokenizer's reasoning `x-regex`. History that an older client echoes as `"thought\n…"` renders as if it had no label.
- Gemma 4 tool-loop history now renders the way the official chat template intends. Assistant messages keep their `assistant` role; only hf2q's fallback template still renames them to `model`. Each tool-loop step therefore continues the open model turn instead of closing and reopening it. Tool results pass their `tool_call_id` and `name` to the template, so they render as `response:<function>` and no longer as `response:unknown`. Gemma prompt bytes change for multi-turn tool conversations, so cached prefixes from earlier renders stop matching once. Persisted Gemma response-replay snapshots move to format version 3.
- Qwen prefix snapshots and `restore_partial` now preserve the true MTP cache cursor (the live cursor, truncated to the restored prefix). Previously capture snapshots and `restore_partial` forced it to the prefix length, and plain prefix snapshots left it unclamped. An ordinary (non-speculative) prefix previously restored claiming MTP K/V positions that were never written; it now restores with an empty MTP cursor, and speculative prefixes restore exactly. Served outputs are unchanged: ordinary decode never reads the MTP slot, and speculative reuse already required validated target/MTP cursor equality. Capture snapshots persisted by earlier builds still restore their old cursor until evicted; no restore consumer reads it.
- Native phase traces of multi-agent Qwen cohorts now finish with their `end` record. The MTP catch-up kept its shifted-hidden copy encoder alive past the catch-up's own drain, so the native observer checked that copy only at the next same-queue wait. When a catch-up was the last GPU work, `native_final` reported `pending_commands: 1` and the trace was invalidated. That happened at the end of every traced n4, 3+1, 2+2 and 1+3 cohort. The copy encoder now drops at its commit, so the catch-up's drain checks it. The TQ bridge dequant, the MTP residual norm and the legacy output-head encoders now drop at their commits for the same reason (`docs/research/adr063-q2-native-trace-finalization.md`).
- The native trace event writer no longer overflows its queue during multi-slot decode. It wrote one event per lock with two file writes, about 7.5 µs per event, against bursts of about 1,000 events per 2 ms. A full queue fails the observer sink, and every later event is dropped: 502,449 in a traced 3+1 cohort. The writer now appends up to 64 events per lock in one write, with record bytes unchanged (0.52 µs per event, against 6.05 µs before, on 100,000 recorded events), and the queue holds 2,048 events.

## [0.1.28] — 2026-10-29

### Fixed

- Accept omitted, `false`, or `null` non-strict function-tool hints from OpenAI-compatible clients without changing the rendered prompt or weakening locked tool policy. Strict tools remain unsupported.
- Preserve insertion order when rendering native tool JSON, including the `name`/`arguments` order used by model chat templates.
- Preserve reusable Gemma 4 prefixes before cold batch admission. Correct DeepSeek DSML parameter boundaries and compile bounded tool grammars without collisions with public schema names.

### Changed

- Reduce DeepSeek ratio-four indexer scratch through the native tiled workspace plan, and reuse a pristine bootstrap cache for the first serving slot. These reduce requested allocation sizes; they do not establish whole-model physical fit or a serving speedup.
- Derive native evidence-version metadata from the exact dependency pin.

### Added

- Explicit, bounded native command/phase tracing, queued-forward-demand diagnostics, and source-bound agentic trace auditing. These remain investigation tools, separate from performance qualification.
- Default-off Gemma full-input checkpoint experiments with unsafe acknowledgement and a bounded anchor quota. The production prefix fix does not enable this experiment.
- Native backing, requested-root and cache-owner observation controls for Qwen and DeepSeek. Observations do not authorize physical-memory credit or largest-profile admission.
- Pin published `mlx-native 0.15.15` and add bounded same-file DeepSeek metadata/map diagnostics; these do not load a model or grant memory admission.


## [0.1.27] — 2026-10-22

### Added

- GLP apply support for Gemma 4 — the `residual_stream_post_layer`
  apply arm (the family's second hook site), so GLP steering applies on
  Gemma 4 at the post-layer residual site (ADR-053, #299).
- `hf2q convert --source-ref` records hub-verified receipts for
  local-source conversions, pinning the exact upstream revision a
  local source directory was verified against (#337).
- The Qwen3.8-27B vanilla graft lane is registered (the triple-combo
  QE prerequisite), and the ADR-059 flagship record is landed: the
  trained KV-graft scoreboard on stock Qwen3.6-35B-A3B — lane
  48/60 → 0/60 refusals (harmless 0/20 intact), two-judge StrongREJECT
  73.7%/71.8% → 33.9%/24.4% at unchanged capability (9/15 → 9/15),
  teacher-forced KL 0.038 vs the published weight-surgery row's 0.115
  (Abliterix-EGA) — runtime unalignment at ~1/3 the divergence,
  boot-reversible, zero weight edits (#205, #336).

### Fixed

- Dependency advisories cleared: rustls 0.23.46,
  rustls-webpki 0.103.16, webpki-root-certs and webpki-roots 1.0.10
  (RUSTSEC-2026-0346..0349, 0381..0383) (#345).
- The OpenCode web-search stack: the `web-search-fetch` plugin is
  ported to the OpenCode 2.x plugin API (the built-in fetch-only
  `webfetch` is removed outright rather than denied), general searches
  route via duckduckgo web to dodge the JS-endpoint CAPTCHA, and the
  getting-started contract asserts the new markers (#342, #343, #345).


## [0.1.26] — 2026-10-12

### Added

- `hf2q calibrate` works on every served model family — Qwen 3.5/3.6/3.8
  joins DeepSeek-V4, each with its own capture arms, register report
  (measured refusal openers replace hardcoded defaults), and family
  defaults; a new `--hook-point`/`--capture-site` surface selects the
  capture and apply sites (the family's default is same-site
  capture/apply; cross-site pairs stay expressible). The measured outcome
  is recorded honestly: the dose/site measurement ladder ran to its
  pre-registered bar and the calibrated vectors remain behaviorally inert
  on every family (the all-null record — a calibrated vector stays a
  candidate, not a validated derivation) (ADR-054/ADR-053, #300).

### Fixed

- Gemma 4 under an explicit `--scheduler fifo-serial` no longer returns
  HTTP 500 on the second sequential request — the resumable-prefill
  capacity is now tile-rounded like its sibling path, and any residual
  capacity mismatch degrades to a cold prefill with a warning instead of
  a server error (#325).

## [0.1.25] — 2026-10-12

### Added

- The KV-cache graft surface is live on every served family — Gemma 4
  (the trained-bank product scoreboard: harmful refusals 36/60 → 1/60 with
  the harmless suite fully preserved, on the unmodified stock artifact) and
  DeepSeek-V4's compressed_kv site (the reader/bind/splice plus the hardware
  canary matrix ALL PASS on both schedulers; the behavioral scoreboard is
  pending the reference training, which needs ≥320 GB of memory — see Known
  issues) (ADR-059, #303-#310).
- Per-family GLP apply arms: Qwen's `ffn_out_pre_residual` writer site and
  DeepSeek-V4's `residual_stream_post_layer` post-layer site (project-mode
  mHC, both widths), with bind accepting each family's supported site set
  (ADR-053, #295-#297).
- `hf2q calibrate`-measured GLP re-measurement at the capture site
  (ADR-054 follow-up: a second null, honestly recorded — the calibrated
  vector stays a candidate, not a validated derivation) (#298).
- The requalification scoreboard: StrongREJECT two-judge plus a 256-prompt
  harmful suite and capability checks on Gemma 4 (60.9% → 26.9% ensemble
  refusals, capability preserved), Qwen 4B (63.6% → 25.3%, capability
  preserved — the reference's GSM8K cost did not reproduce at this
  site/dose), and the Qwen 35B flagship row (67.5% → 18.5%, capability
  preserved) (#310).

### Fixed

- DeepSeek-V4 `--kv-persist` now works: the exact recovery anchor persists
  as one prefix image per conversation and hydrates on a cold miss on both
  schedulers (a restart resumes in seconds; corrupt or missing files only
  cause a cold prefill; no cross-conversation leaks) (ADR-062 D4, #293).
- Reasoning controls behave the same on every family: one effort table
  (none/minimal off, low 512, medium 2048, high 8192, xhigh/max unbounded;
  DeepSeek medium→high), the `reasoning.{effort,enabled,max_tokens}`
  aliases, one shared budget enforcer including Gemma 4, server defaults
  that never cause a 4xx, and `none` as an accepted no-op everywhere
  (ADR-062 D5, #294).

### Known issues

- The DeepSeek-V4 compressed_kv graft's behavioral scoreboard is pending:
  the phantom-kv reference training needs ≥320 GB of unified memory (the
  FP8 checkpoint dequantizes to ~310 GB bf16) or a CUDA/FP8 host; the
  mechanism, canaries, and toolchain are landed and verified (#309, moves
  to v0.1.26).
- Gemma 4 under an explicit `--scheduler fifo-serial` returns HTTP 500 on
  the second sequential request (a resumable-prefill capacity check) — the
  default scheduler is unaffected; filed as #325.

## [0.1.24] — 2026-10-09

### Added

- Per-family built-in serving profiles: the values each canonical launcher
  uses now apply built-in, with or without a `config.toml` — Qwen
  (1.05/2048/512), Gemma 4 (1.05, cross-slot admission on), and DeepSeek-V4
  (1.0 with a 512-token tool-thinking budget) (ADR-062 D1, #288, #314).
- `hf2q info` and `/hf2q/v1/runtime` report each effective serve value and
  where it came from (ADR-062 D1, #289, #316).

### Changed

- A fresh `hf2q serve` with no flags and no `config.toml` now uses the
  inflight-batched scheduler with 4 slots; fifo-serial is an explicit
  compatibility mode (ADR-062 D0, #287, #280).
- The default serve port is 8081 on every path (ADR-062 D1, #289, #316).

### Fixed

- A grammar or tool call cut by `max_tokens` finishes with
  `finish_reason: "length"` and the partial text instead of HTTP 500
  (ADR-062 D2, #256, #311).
- One typed 4xx error contract: DeepSeek-V4 `/v1/embeddings` returns 400
  until supported (no longer 500), every length/capacity rejection is a 400
  `context_length_exceeded` (the code OpenCode maps to compaction), and 501
  remains only for genuinely unimplemented features (ADR-062 D2, #290, #312).
- `--max-slots` accepts 1–8 for every family with one clear validation; the
  `HF2Q_MAX_BATCHED_SLOTS` / `HF2Q_SPEC_DECODE_ALLOW_OVERSIZED` environment
  gates are gone (ADR-062 D3, #291, #313).
- `/hf2q/v1/runtime`'s `kv_persist_enabled` reports what each loaded engine
  actually does (ADR-062 D4, #292, #315).
- The canonical launchers need no `HF2Q_*` variables for qualified behavior
  (ADR-062 D1, #289, #316).

## [0.1.23] — 2026-10-08

### Fixed

- `hf2q serve` of a Gemma 4 repository (for example
  `jenerallee78/gemma-4-26B-A4B-it-ara-abliterated`) no longer fails with
  "vision_config.image_size missing" before download: projector sizing now
  follows the model family.
- Vision never blocks a serve: when a model's projector cannot be planned or
  converted, hf2q warns and serves the model text-only instead of failing.
- Tool calls work for fine-tunes whose display name does not contain the
  family name: the tool-call format is chosen from the GGUF
  `general.architecture`, so a renamed Gemma 4 fine-tune can call tools and
  run OpenCode.
- Gemma 4 vision normalizes pixels with the projector's own mean and std,
  matching llama.cpp on the same GGUF and projector; images were previously
  shifted and described wrongly.
- Integer range grammars grow linearly with the number of digits instead of
  doubling per digit, so stock OpenCode tool schemas compile again.
- `--glp` steers every model the server loads, including a model chosen with
  `hf2q chat --model`, which 0.1.22 served unsteered.
- DeepSeek-V4 logs the GLP load step at debug level instead of printing to
  stderr.

## [0.1.22] — 2026-10-08

### Fixed

- Fix the decode hard-lock on Qwen 3.5/3.6 (qwen35 and qwen35moe): a first
  prompt under 16 tokens (for example "hi") panicked the engine thread on the
  TQ cache layout, after which the server reported "warming up" forever.
- Report a dead inference engine as a terminal failure: `/readyz` and HTTP 503
  `engine_failed` say the engine stopped and the server must be restarted,
  the operator screen shows it, and `hf2q chat` explains it, instead of
  "warming up" forever.
- Compile tool and response JSON schemas with length or count bounds above
  2000 exactly instead of rejecting the whole request; OpenCode with MCP tools
  that declare `maxLength: 4096` works again. Combining tools past the grammar
  size limit returns HTTP 400 instead of crashing.
- The `install.sh` one-liner and `hf2q update` no longer need Xcode developer
  tools: the architecture check reads the Mach-O header instead of calling the
  `lipo` shim, which failed on Macs without accepted developer tools.
- Shell completions install only under the user's home; hf2q no longer writes,
  registers, or removes files in Homebrew directories, so uninstalling one
  installation cannot delete another installation's completion.
- `--glp` resolves Hugging Face repository and file-URL references (0.1.21
  treated them as local paths), and binds checkpoint identity explicitly.
- Bound the hf-hub client so exact-revision queries cannot hang startup.
- The crate package fits under the crates.io 10 MiB upload cap.
- Upgrade rustls for RUSTSEC-2026-0285.

### Added

- `hf2q chat`: `/max-tokens N` sets the reply length for the session, `/status`
  shows the current limit and its source, and a reply cut by the limit says so.
- `--gcd-schema-locked` makes the server schema mandatory (ADR-057 lockdown).
- Model download progress shows "estimating", "buffering", or "stalled" with a
  smoothed, bounded ETA instead of looking hung and printing absurd ETAs.
- `docs/RELEASING.md` (release checklist) and `docs/qe-playbook.md` (ad hoc,
  hands-on QE).

### Changed

- The release workflow's model qualification job is opt-in; QE is ad hoc and
  never gates a release (ADR-061).

## [0.1.21] — 2026-09-03

### Added

- Add one fail-closed structured-output subsystem for every supported text
  serving family, covering OpenAI response formats, peer-compatible GBNF and
  JSON Schema inputs, and vLLM choice, regex, JSON, grammar, and structural-tag
  modes. The generic JSON Schema profile includes the r2c ReviewLens Stage 6
  and CWE Stage 9 contracts without adding an r2c runtime dependency.
- Add exact signed-binary release receipts for every enabled text-generation
  GGUF architecture (`gemma4`, `qwen35`, `qwen35moe`, and `deepseek4`). The
  gate covers empty and nested Stage 6 branches, Stage 9
  primary/related/null unary and SSE branches, and adversarial
  `const`/`additionalProperties` enforcement, while binding each reference
  artifact's actual architecture so same-branch substitutions fail closed.

### Changed

- Make the protected Release workflow run and independently verify the
  cross-family model/grammar qualification manifest before publication, then
  publish deterministic qualification evidence with the release assets.

### Fixed

- Qualify exact-artifact model and grammar gates on battery when macOS High
  Power Mode is active. The gate now binds and continuously monitors the live
  performance policy instead of treating an AC cable as a compute proxy.
- Make the OpenCode web-search stack install reliably across supported hosts,
  including environments that require an explicit SearXNG build backend.
- Keep release-build proc macros loadable under macOS 27's stricter Mach-O
  validation, while preserving the normal optimized profile for the hf2q
  executable and verifying LINKEDIT alignment before signing.

## [0.1.15] — 2026-08-24

### Added

- Accept `owner/repository[:QUANT]` directly in `convert`, `serve`, and `chat`.
  Model-targeted serve/chat reuse verified local artifacts first, adopt exact
  matching manual downloads without another payload transfer, download the
  requested compatible hosted GGUF when needed, and use native source
  conversion only when no supported hosted GGUF exists.
- Add `hf2q serve list` and `hf2q chat list` as read-only local model
  inventories, and automatically prepare a revision-matched projector for
  supported multimodal repository models or a valid present projector beside
  an explicit local GGUF, while retaining explicit `--mmproj` fail-closed
  behavior.

### Changed

- Make `convert` output optional for remote repositories. It now selects the
  setup/live hardware quant when unspecified, writes revision-bound artifacts
  below the canonical hf2q data directory, treats `repo:QUANT` and
  `--quant QUANT` equivalently, and reuses only a receipt-verified native hf2q
  conversion.
- Make targeted chat own and authenticate the exact server child it starts,
  propagate the selected setup state root, retain its prebound loopback TCP
  listener, and require a matching private inherited READY message plus
  periodic first-use preparation heartbeats while keeping credentials away
  from untrusted discovery candidates.
- Admit hosted and exact manually downloaded Qwen GGUFs from bounded GGUF
  metadata, tokenizer, operational-scalar, normal-layer and supported
  one-layer MTP/NextN topology, packed-byte geometry, and storage-role checks
  before adoption or payload transfer. Serving does not require or copy a
  sibling `config.json`.
- Preflight exact uncached source, quantized text, and F16 projector extents as
  one native-conversion plan, and recover no-clobber paired publication from
  both pre-journal and journaled crashes.
- Use collision-resistant managed repository directories with bounded legacy
  receipt migration, retain verified source descriptors through independent
  CoW/copy publication, and serialize text/projector conversion destinations
  without rejecting extensionless text-only outputs.

## [0.1.14] — 2026-08-23

### Added

- Add `hf2q info --model <GGUF> [--mmproj <GGUF>]` for a static preview that
  does not decode/upload tensor payloads or initialize Metal, covering model
  family, declared and effective context, tokenizer/template capabilities,
  vision pairing, tensor catalog, memory planning, and the exact reason
  serving would be rejected.
- Add typed `serve --ctx <TOKENS>` and `serve --kv-cache-budget <SIZE>`
  controls, with matching optional `[serve]` configuration keys. Omitted
  context uses the GGUF declaration; an explicit over-limit context fails
  before tensor loading.
- Add `serve --kv-persist-budget <SIZE>`, `[serve] kv_persist_budget`, and
  `setup --serve-kv-persist-budget` for a human-readable, fail-loud disk
  ceiling shared by the generic and Qwen persistent-KV stores.

### Changed

- Keep every admitted agent slot logically capable of the complete effective
  context while applying one shared physical KV-residency budget. Context is
  never divided by `--max-slots`.
- Replace hidden environment-variable serving defaults with typed Clap/config
  controls for scheduler, slots, active-KV budget, persistent-KV disk budget,
  repetition penalty, and thinking budgets. CLI values override configuration;
  shell state no longer disables an explicit `--kv-persist` path.
- Extend generated Bash, Zsh, Fish, Elvish, and PowerShell completion plus
  dynamic GGUF completion to the `info` command and all new planning flags.

### Fixed

- Make explicit `generate --kv-bits` reach runtime policy without mutating the
  process environment, and reject the removed `--max-seq-len` and raw
  `--kv-cache-budget-bytes` surfaces instead of accepting inert controls.

## [0.1.13] — 2026-08-22

### Fixed

- Apply canonical managed-model completion consistently to every user-facing
  local GGUF selector: chat, generation, serving, embedding, projector, and
  parity arguments. Hybrid `chat --model` still accepts endpoint model IDs and
  Hugging Face repositories, explicit customer paths remain explicit, and the
  repository-only `cache clear --model` selector remains unchanged.

## [0.1.12] — 2026-08-22

### Changed

- Prefer `${XDG_DATA_HOME:-$HOME/.local/share}/hf2q/models` when dynamically
  completing an empty or bare `hf2q serve --model` or `--mmproj` value. Decoder
  and projector GGUFs are filtered separately, while explicit relative,
  home-relative, and absolute customer paths retain normal filesystem
  completion.

## [0.1.11] — 2026-08-22

### Added

- Execute packed Qwen3.8 Q5_K/Q6_K embedding rows directly on Metal while
  preserving native GGUF storage for embeddings, projections, output heads,
  and the shared MTP head. Invalid native routing or storage geometry now
  fails model admission before readiness.

### Changed

- Pin the published, checksum-bound `mlx-native 0.11.2` dependency and bind
  Qwen hosted, cache, and pool identity to exact GGUF file type 17.
- Replace DeepSeek-V4's falsified host-global zero-page acceptance rule with
  phase-bound process residency evidence. The release gate retains fail-closed
  swap, process-page-in, pressure, thermal, phase, and contention checks while
  recording host-global VM counters as diagnostics rather than inventing
  thresholds for them.

### Fixed

- Preserve the required-tool answer capacity for Qwen agentic requests and
  make equal-logit greedy argmax deterministic at the lowest token index.
- Replace the getting-started guide's truncated PNG with a strictly valid
  vision fixture, surface vision-server error bodies, and make the web-stack
  installer quiet and actionable under `curl | bash` on macOS Bash 3.2.

## [0.1.10] — 2026-08-22

### Changed

- Persist the qualified agentic serving profile in `hf2q setup`: answering
  yes to "Optimize serving for long agent and tool-use prompts?" (the
  default) writes repetition penalty `1.05`, thinking budget `2048`, and
  tool-continuation budget `512` to `config.toml`, and `hf2q serve` applies
  them to `HF2Q_DEFAULT_REPETITION_PENALTY`,
  `HF2Q_DEFAULT_THINKING_TOKEN_BUDGET`, and
  `HF2Q_DEFAULT_TOOL_THINKING_TOKEN_BUDGET` only when the operator has not
  exported those variables. Existing v2 configs without the new keys keep
  loading unchanged.
- Default `HF2Q_QWEN_SPECULATION` to `auto` in the qwen35 server engine
  when the variable is unset (ADR-044, updated 2026-08-21); every admission
  path still fails closed to ordinary target decode, and explicit `off`
  remains the escape hatch.
- Apply the Qwen3.8-qualified decode route (`HF2Q_DECODE_MVN=0` +
  `HF2Q_DECODE_MV_EXT=1`) by default when loading a Qwen3.8-identified
  model, at engine load, without overriding operator-exported values. The
  route stays Qwen3.8-scoped because `mul_mv_ext` is not bit-exact and no
  other family carries the qualifying receipt.
- Rewrite `docs/getting-started.md` as the single first-run journey:
  install, `hf2q setup`, verified pair download, one-line foreground serve,
  `hf2q chat` generation proof, single-request vision proof, OpenCode
  provider setup, full Agentic Kit, and the local research stack. The
  environment-variable ritual, background-server apparatus, and shell SSE
  parsing are gone; qualified behavior now ships as product defaults.
- Make the OpenCode web-stack installer's `--status` a real health check
  (plugin state, LaunchAgents, fetch health content, live fetch, live
  search) and fail installation on empty SearXNG results instead of
  accepting an unproven search path.

## [0.1.9] — 2026-08-21

### Added

- Detect standalone, Cargo registry/Git/path, source-development, ambiguous,
  and unmanaged installations so `hf2q update` and `hf2q uninstall` honor the
  channel that owns the active executable instead of guessing or crossing
  channels.
- Add exact, non-mutating config/cache purge previews and confirmed execution
  that removes only named hf2q-owned data while preserving unknown siblings,
  external models, Hugging Face data, persistent-KV roots, locks, and logs.
- Add the source-bound Qwen3.8 reference acceptance verifier and retain its
  exact comparison, thresholds, and quality evidence.
- Add the verified Qwen3.8 operator guides for the core hf2q journey and the
  separately scoped OpenCode, Agentic Kit, and local search/fetch integration.

### Changed

- Preserve Cargo root, source selector, features, profile, target, and version
  requirements during delegated updates; source checkouts remain explicitly
  non-mutating and receive exact rebuild instructions.
- Make native Qwen3.8 conversion, Q4_K_M quantization, foreground serving,
  `hf2q chat`, and direct API proof the documented core CLI sequence.

### Fixed

- Publish automatically converted text/projector GGUF pairs as one
  crash-recoverable, generation-bound transaction, and enforce the text
  artifact's projector digest for local as well as receipt-backed models.
- Bind Qwen3.8 qualification inputs to the exact paired artifact identity and
  keep peer-reference wording within the checked-in policy boundary.

## [0.1.8] — 2026-08-21

### Added

- Let `hf2q chat` discover receipt-backed and canonical-cache GGUF artifacts
  local to the server, verify their exact bytes before activation, prefer them
  over hosted downloads, and keep server paths and digests out of the terminal
  and HTTP catalog.

### Changed

- Widen Qwen single-slot prefill transactions when no decoder is waiting,
  retaining the bounded interactive transaction size once mixed work appears.
- Make one `Release` dispatch build, Developer-ID sign, notarize, publish, and
  post-verify the exact standalone candidate. The independently dispatchable
  candidate workflow remains available for diagnostics and optional model
  qualification; publishing no longer requires copying a run ID between two
  manual workflows.
- Parameterize standalone-candidate and optional cache-lifecycle qualification
  by the requested stable version instead of embedding the previous release
  number.

### Fixed

- Harden the readable standalone installer against truncated `curl | sh`
  input, candidate stdin capture, Rosetta architecture misdetection,
  unbounded transfers, production `file://` overrides, and group/world-
  writable install directories. Require exact HTTP 200 responses in the
  standalone updater and preserve the same directory invariant in Rust.
- Drop the publication job's GitHub API credentials immediately after exact
  candidate artifact download and before either candidate is executed.

### Removed

- Remove the unreachable custom TUF verifier and spike, sealed update
  transport, metadata journal, archive/Mach-O preparation stack, first-
  activation graph, and their structural CI gates. Also remove setup's
  read-only installation-identity coupling: it was reachable, but depended on
  state no shipped installation channel creates or uses. The standalone
  installer, updater, rollback, uninstaller, Apple trust checks, and atomic
  publisher remain unchanged.
- Remove the unreachable ADR-045 no-options model recipe, preparation plan,
  paired-artifact publication, prepared-profile registry, retention, and
  calibration-pending stack. Normal Hugging Face `convert` behavior remains
  unchanged; ADR-046 now owns a minimal read-only source-teacher manifest.

## [0.1.7] — 2026-08-21

### Fixed

- Guarantee serve-side error semantics for the guarantees tune-up:
  `tools[]` on a family without a registered tool-call emitter now refuses
  up front with `501 capability_unsupported`; never-fits KV-budget
  admissions return a non-retryable `400 kv_budget_unsatisfiable` instead
  of `429`; dense Qwen3-VL is refused at spawn until the ADR-041 engine
  seam lands; SSE responses report cached tokens and carry the
  `X-HF2Q-Cached-Tokens` header.

### Added

- Add a root `NOTICE` file carrying the single MIT attribution for the
  pinned upstream GGUF engine ("the peer"), and a fail-closed CI ratchet
  (`scripts/peer_ref_ratchet.sh` + `data/peer_ref_baseline.txt`) that
  blocks new literal references outside the whitelisted zones.

- Add a canonical, text-only getting-started guide for the exact
  `jenerallee78/Qwen3.8-27B-Abliterated-SFT` source revision. The guide uses
  the existing native convert, Q4_K_M quantization, serve, OpenAI-compatible
  API, and optional OpenCode surfaces without adding onboarding orchestration.
- Add `hf2q setup`, a no-download Apple-Silicon inventory and deterministic
  schema-2 configuration producer with explicit noninteractive flags,
  descriptor-relative crash recovery, and a custom absolute state-root option.
  Its conversion quantization and serving host, port, scheduler, and active-slot
  defaults are consumed by the existing `convert` and `serve` commands, with
  explicit command options retaining precedence.
- Add the first reachable standalone lifecycle: a generated exact-release
  installer, canonical channel marker, atomic executable replacement, one
  retained rollback version, `hf2q update`, and marker-gated
  `hf2q uninstall --yes`.
- Add the protected standalone release rail: locked packed-source arm64/macOS
  14 build, checkout-owned Developer ID signing without candidate execution,
  ZIP-carried notarization for the raw CLI, exact signed-byte verification,
  immutable draft assets, and local plus public installed setup/uninstall
  verification. v0.1.7 passed the credential-backed release and live
  stable-record gates; `https://hf2q.us/install.sh` now serves the exact
  published installer.
- Accept canonical Hugging Face model IDs and official model/tree/blob/resolve
  URLs at `hf2q convert`, resolving mutable names to an exact commit before
  transfer and recording the normalized identity in conversion receipt v3.
- Add native Qwen3.8 conversion and Apple-Silicon inference, including the
  model's hybrid attention graph, multimodal projector path, OpenAI-compatible
  reasoning/tools, retained-prefix serving, and exact-artifact agentic gates.
- Add strict signed distribution, activation-state, and update-journal schemas.

### Changed

- Separate standalone publication from model qualification. Routine releases
  consume the exact packed, signed, notarized CLI candidate and no longer wait
  for DeepSeek, Gemma, Qwen, or Qwen3.8 cache/performance runs; those remain
  independently dispatchable gates under their model and serving decisions.
  The packed release smoke now checks only compilation, explicit completions,
  setup, standalone distribution, and installed CLI behavior.
- Adopt the single-attribution reference policy repo-wide: prose in docs,
  comments, scripts, and command output now says "the peer" instead of
  naming the upstream engine; `LlamaFtype` is renamed `GgufFtype`
  (`llama_ftype.rs` → `ftype.rs`), vision resize helpers and the
  special-token resolvers take peer naming, the operator runbook moved to
  `docs/operator-peer-refresh-runbook.md`, and the reddit posts moved to
  `archive/`. Parity-harness code, `LLAMA_*` env vars, real binary names,
  pinned source paths, receipt labels, and frozen historical ADRs are
  unchanged.
- Read the macOS-reported performance-level count during setup instead of
  assuming every Apple Silicon host exposes exactly two levels.

### Fixed

- Validate the Developer ID hardened-runtime flag at its actual location in
  Apple's `codesign --display --verbose=4` `CodeDirectory` line.
- Bind raw-CLI notarization to Apple's accepted ticket CDHash at release time;
  replace the app-bundle-only `spctl` assessment with `codesign`'s forced
  online ticket lookup plus explicit `=notarized` requirement, while retaining
  strict signature-chain checks in install/update.
- Keep shell-completion installation explicit through
  `hf2q completions --shell <shell>`; ordinary CLI startup no longer writes
  completion registrations or shell startup files.
- Remove the unreachable second managed-session cache and its runtime
  authorization boundary from setup. Existing persistence remains owned by
  ADR-017 and ADR-027.
- Replace the external Hugging Face CLI conversion download path with the
  in-process `hf-hub` client. The downloader authenticates bounded metadata
  before transfer, consumes only index-selected safetensors, verifies LFS
  SHA-256 and Git blob SHA-1 identities, and records every selected local
  SHA-256 without granting authority to unrelated cache files.
- Cooperatively execute compatible DeepSeek-V4 warm matrix suffixes across two
  to four live Apple-Silicon sessions. Attention and cache state remain
  sequence-local while the row-local FFN/MoE shares one bounded layer-major
  transaction; cold, mixed, recovery-tail, and decode work keep their proven
  serial paths. Exact-artifact acceptance prebuilds before thermal settle,
  clears benchmark overrides, derives medians from sealed timing arrays, and
  independently verifies every benchmark, thermal, and production-trace log.
- Pin published `mlx-native 0.10.16` and use the register-resident,
  bit-exact GQA-cooperative Q2 TQ-HB attention path behind a measured
  context-length selector. Publication
  requires an exact-source, thermally guarded OFF/AUTO ABBA receipt with at
  least 15% end-to-end improvement near 105K tokens, identical greedy output,
  and no short-context route regression. The final dependency also bounds
  D512 prefill reads at a partial logical KV tail without changing aligned
  kernel performance.
- Bound Qwen's fresh-turn and tool-continuation reasoning budgets, expose the
  effective thinking phase and answer progress in the operator dashboard, and
  warn on repeated tool-result signatures without blocking legitimate retries.
- Preserve text-prefix KV across a first image suffix when exact tokens,
  multimodal spans, DeepStack positions, and all mRoPE axes prove the earlier
  text state is causally reusable.

### Fixed

- Bound DeepSeek-V4's forced-open reasoning before a narrowly constrained
  single required tool, and align staggered compatible warm-prefill cursors
  before resuming four-lane cooperative execution. The protected gate proves
  both transitions without widening the 60/15/35-second product limits.
- Keep the OpenAI `parallel_tool_calls` omission consistent with hf2q's
  established single-call default, and stop DeepSeek generation as soon as a
  constrained non-parallel tool body is accepted. This removes an unused final
  full-model forward while preserving explicit parallel calls and exact tool
  semantics.
- Bind the DeepSeek four-agent release workload to the exact insertion-ordered
  request, rendered prompt, token IDs, historical tool-result bytes, recovery
  anchor, and continuation suffix. Artifact download revalidates every
  per-agent digest and explicitly rejects the former key-sorted rendering.
- Emit authoritative, exactly-once Qwen SlotAware decode-complete telemetry so
  the long-context receipt cannot accept response timing detached from the
  engine path that generated it; requests with a detected client cancellation
  never publish success.
- Update `h2` to 0.4.16 to resolve RUSTSEC-2026-0258 and keep the exact-source
  security rebuild inside the release gate.
- Upgrade the direct `indicatif` dependency to 0.18.4, removing the
  unmaintained `number_prefix` transitive dependency from the release graph.
- Isolate Hugging Face cache environment mutation so parallel hosted tests no
  longer race over process-global cache paths.
- Preserve client JSON-object insertion order while rendering native chat
  templates. DeepSeek-V4's published encoder treats the serialized tool schema
  as prompt bytes; sorting the function and parameter keys changed the greedy
  trajectory and could push an otherwise correct required tool call beyond a
  128-token completion budget. The corrected renderer matches the source
  prompt, while conversion, quantization, and model weights remain unchanged.
- Rebind the frozen DeepSeek four-agent release fixture to its deterministic
  6,684-token insertion-ordered render. The prior 6,685-token assertion was the
  legacy key-sorted count and stopped an otherwise coherent hardware wave.
  Release validation now renders and tokenizes the exact first request before
  loading the 100 GiB verifier, so future prompt drift fails early.
- Hash each immutable release model once per stable file identity instead of
  rereading the same large GGUF in every release run and child harness. Child
  and terminal checks bind the verified digest to the unchanged path,
  device/inode, size, modification time, and change time; standalone harnesses
  retain full hashing.
  The self-hosted runner persists that receipt across release attempts, so an
  unchanged model is not content-hashed again on every rerun.
- Supervise the hardware gate as its own process group so workflow cancellation
  also stops an active compiler, model server, and power helpers instead of
  leaving the canceled gate running as an orphan.
- Recalibrate the public Qwen3.6 347-tool/87,972-token release fixture for that
  insertion-order contract. The production renderer adds 28 tokens per tool
  relative to the historical key-sorted fixture, so its deterministic padding
  drops by 9,716 tokens and its canonical request digest is rebound without
  changing the long-overlap workload or stable prompt boundary.
- Make GitHub release-tag creation distinguish an explicit missing-ref 404
  from every other API failure. Release automation now validates existing and
  newly created tag SHAs, re-reads a created ref before publishing assets, and
  carries model-free regressions for missing, mismatched, and failed tag paths.

## [0.1.6] — 2026-08-13

### Changed

- Run Gemma's two calibrated four-slot agent waves before its destructive
  175K-token overlap/cancellation and 120K-token lifecycle soak. Each wave now
  requires a trailing 60 seconds of Nominal thermal state and remains under
  fail-closed two-second sampling through the complete cold, cached, automatic-
  tool, and tool-result sequence. The exact packed `fbcf4637` gate had already
  proved Gemma overlap, rollback, isolation, and retained-prefix correctness,
  but all four post-soak tool turns took 16.562–16.686 seconds against the
  unchanged 16-second default limit; historical fresh waves are 12.156–14.399
  seconds. This fixes the calibration order and receipt authority rather than
  weakening the user-facing limit.
- Keep Gemma's full-wave thermal supervisor in the release-gate process. The
  first exact-main rerun (`31691093129`) completed wave 1 with all four
  tool-result turns at 13.886–14.008 seconds under entirely Nominal telemetry,
  but failed after the harness and before sealing the terminal thermal summary.
  The old unsheltered process-state assignment could trigger `set -e` exactly
  as the harness disappeared; cleanup then wrote the observed stop sentinel.
  Foreground supervision preserves the same two-second coverage while removing
  that receipt-lifecycle race; inference limits and acceptance predicates are
  unchanged.
- Increase DeepSeek-V4's default pure-decode slot quantum from 8 to 64 after
  the cold-prefill barrier lifts, amortizing session swaps and scheduler
  publication across saturated cohorts. Genuinely mixed prefill/decode work
  remains clamped to the measured eight-token/two-window interactive budget.
  The intermediate 16-token exact-main candidate still failed closed at
  55.024 s. Although two fresh 32-token discriminator waves passed at
  48.859–52.452 s, its exact-main packed gate still exposed a 55.411 s tail.
  Two fresh same-binary 64-token waves then passed at 45.725–51.415 s with
  unchanged 6,677-token prefix reuse and all tool-result checks green. Its
  exact-main packed gate subsequently exposed one 55.585 s tail under a fully
  Nominal thermal envelope. Two pinned same-input llama.cpp waves measured
  68.438 s and 69.944 s (69.191 s median), so the protected hf2q ceiling is
  rebaselined to 60 s—9.2 s below the current peer median. Exact packed
  two-wave and cross-family model-qualification authority remains pending.

### Fixed

- Contain an intermittent Gemma N=8 tiny slot-prefill coherence fault without
  changing the established long-prefill graph. Exact-main packed run
  `31823149737` reached Gemma's release-profile parity test after the preceding
  DeepSeek and Gemma gates passed, then exposed an all-non-finite logit row
  after batched decode followed by a 2–5-token cold mounted prefill. Cold text
  work below 32 tokens now uses the linear slot-aware path, 1–31-token retained
  suffixes append through the existing one-token slot primitive, and initial,
  installed, and stable cross-slot admission cannot bypass that boundary. A
  production-worker discriminator then found a separate defect in the linear
  fallback: pinned `mlx-native` 0.10.8 `flash_attn_vec` loads a complete 32-row
  K/V tile before masking, while tiny requests previously allocated only their
  logical 3–29-row extent. Linear physical KV capacity is now rounded through
  the final 32-row tile; the pinned dependency's `MlxDevice::alloc_buffer`
  zero-fills that allocation, so masked padding is finite. Linear and batched
  prefill both reject the observed non-finite argmax poison, and a one-token
  generation budget now terminates at the prefill seed instead of decoding an
  extra token. The failed command-buffer/fence experiments are not shipped,
  the original tiny batched-prefill cause remains open behind the containment
  boundary. Focused
  dirty-source evidence passed 25/25 fresh N=8 worker rounds at each of
  `max_tokens=1`, `2`, and `24`, plus 64 cold/staggered and 16 retained-suffix
  rounds in both Hybrid and full-TQ/HB regimes. A new exact-packed N=8 worker
  and cross-family model-qualification run is still required; its worker
  receipt now
  requires canonical cross-slot admission, 25 repeated normal 24-token and
  one-token seed-budget parity rounds, and both Hybrid and full-TQ/HB direct
  tiny-prefill discriminators.
- Make the shared large-model sleep/thermal guard robust to macOS power-log
  rollover. Exact-main run `31755150906` attempt 4 completed the full
  DeepSeek and Gemma matrices, then Qwen accepted both its 552-token lane and
  87,972-token overlap lane. macOS entered clamshell DarkWake while the lanes
  were running; the old matcher ignored DarkWake and then false-failed
  `402 -> 401` as older `pmset` rows expired. Cleanup ended the sleep
  assertions before the later full-sleep row. The guard now captures DarkWake,
  full sleep, and thermal events and compares baseline/final snapshots as
  multisets: expired rows are harmless, every newly added event remains fatal,
  and the established receipt delta stays monotonic and fail-closed. Each leaf
  gate rechecks immediately before committing its receipt, and release replay
  rehashes the exact snapshot inventory and recomputes every empty delta.
- Keep long agentic continuations attached to their strongest retained prefix
  across all three SlotAware families. Admission now distinguishes an active
  prefix from an idle reusable prefix: a continuation waits for the active
  owner when that match is strictly better, while an equal idle match remains
  runnable. This prevents Qwen, Gemma, and DeepSeek from admitting a 100K-token
  continuation cold into another free slot.
- Preserve the newest committed checkpoint when Qwen, Gemma, or DeepSeek work
  is cancelled. Gemma and DeepSeek keep in-request candidates private until
  success. Qwen promotes a stable pre-generation checkpoint immediately after
  its bounded prefill transaction commits, before observing a disconnect, so
  cancellation may safely retain newly committed prompt work without retaining
  a partial generation tail. Recovery restores only a validated checkpoint and
  otherwise resets cold. Inline embedding invalidates the selected slot's
  retained metadata before touching physical KV, so a later generation cannot
  advertise a stale cache hit.
- Let dead requests bypass physical-slot affinity waits in every SlotAware
  family, and let Qwen deterministic response-cache hits bypass them without
  claiming KV. Qwen prompt-cache identity now includes `reasoning_forced_open`,
  preventing a response split under one reasoning mode from being replayed
  under another. Requests for token logprobs bypass Qwen terminal response
  caching until that cache can retain and replay the full logprob payload.
  Gemma and DeepSeek continue to reuse their family-specific KV
  checkpoints rather than advertising a terminal-response cache they do not
  own. Bounded wait queues keep Shutdown non-evictable and allow runnable work
  to displace a blocked active-prefix waiter without overtaking earlier
  runnable requests.
- Resume DeepSeek-V4 retained-prefix continuations whose remaining suffix is
  below the 33-token matrix-append minimum. All nonempty 1–32-token segments
  now use incremental verifier replay, including the segment immediately
  before the eight-token recovery boundary, instead of failing with an empty
  resumable-prefill chunk.
- Restore bulk prefill for a saturated DeepSeek four-cold cohort. Once a
  filling cohort still has another cold request queued, cold-wave unary decode
  is deferred through its draining phase while any cold prefill remains because
  unary output cannot be delivered before that barrier. Full 2,048-token
  prefill transactions resume; streaming and warm decoders remain responsive,
  as does the lopsided eight-token decode/two-window prefill case.
- Pair large DeepSeek routed-expert gate/up projections through the
  family-neutral `mlx-native 0.10.8` schedule primitive. Eligible prefill work
  builds one expert-routing schedule for both existing quantized projections;
  decode-sized work and forced diagnostic routes remain independent. This is
  not an arithmetic-fusion claim, and exact packed hf2q hardware acceptance
  remains required before publishing an end-to-end speedup.
- Refine an over-ceiling macOS process RSS with `footprint` before rejecting a
  100 GiB DeepSeek load. This avoids treating reclaimable WebKit and
  IOAccelerator mappings from a remote-inference coding client as private
  resident memory; a missing, failed, or malformed footprint probe keeps the
  conservative RSS value and still fails closed.
- Resolve the published, checksum-pinned `mlx-native 0.10.8` backend, retaining
  the 0.10.7 command-buffer
  lifetime correction. Internal unlabeled GraphSession command buffers retain
  direct owner release, while labeled or publicly escaped buffers preserve the
  autorelease scope required by external Objective-C ownership. No local Cargo
  patch participates in the release candidate.
- Keep Gemma SerialFifo long-prefix continuations on the bounded live-append
  graph even for 1–31-token suffixes. Sliding-cache staging now derives each
  query position from the actual staged cache capacity rather than capping it
  independently at the 1,024-token semantic window. This preserves the
  current suffix K/V and chronological history beyond the window, and lets a
  successful append publish a reusable latest anchor for the following turn.

### Validation

- Keep the Qwen one-slot cancellation trigger on its first observed third
  2,048-token transaction without weakening the atomic cancellation bound.
  Exact-main run `31804314143` passed the full DeepSeek and Gemma matrices,
  Qwen's 87,972-token overlap, three four-agent cache waves, and Qwen's shared
  lifecycle, then the final cancellation harness observed four transactions
  and stopped before testing rollback. The runtime had received no disconnect;
  the harness was scanning the complete macOS power log inside its 50 ms poll,
  a measured 2.06-second operation versus 1.36–1.58 seconds per transaction.
  Power validation now brackets the low-latency exact-boundary observer, and
  the unchanged post-disconnect contract still permits at most one in-flight
  transaction, requires exactly one cancellation, rejects a successful SSE
  terminal, and proves fresh one-slot reuse. A new exact packed run remains
  required. A focused same-model discriminator with the repaired script passed
  at `3 -> 4 -> 4` transactions, cancellation delta 1, no terminal, ready 200,
  exact one-slot `OK` reuse in 0.415 seconds, and zero power events; it is
  focused evidence, not exact packed release authority.
- Add a cross-family long-context cache-lifecycle gate that establishes an
  agentic tool turn, starts a streamed turn on the retained prefix, queues its
  exact retry while that prefix is active, cancels the owner, requires the
  queued retry to reuse the restored checkpoint, and verifies an unrelated
  conversation cannot inherit private history.
- Pin the DeepSeek paired-prefill selector to large automatic routes with
  scratch and no diagnostic threshold override. The published native package
  and broad DeepSeek model-free suite are locked gates; real-model quality,
  cache, overlap, and calibrated cold-wave performance remain hardware gates.
- Add a guarded self-hosted `Cache lifecycle` workflow that runs DeepSeek,
  Gemma, and Qwen one process at a time under continuously checked AC power,
  verifies protected GGUF digests, and uploads a
  source/crate/binary/model-bound qualification receipt. It now consumes the
  independently packed and signed standalone candidate; publication verifies
  that candidate without depending on the large-model receipt.
- Seal the packed `hf2q` executable into the runner's temporary evidence
  directory before any hardware work, export its build-time digest as a
  distinct immutable input, and recheck the copy against that digest before
  every model load. Exact-main run `31730699128` proved the need for this boundary:
  all DeepSeek and Gemma gates passed, including the thermally guarded
  eight-slot wave and release-profile parity, but Gemma's final Cargo
  integration test relinked the package-local `target/release/hf2q` path.
  Qwen loaded the replacement bytes but its cumulative harness correctly
  rejected the changed digest before sending a request. The original
  receipt-bound bytes remained available under Cargo's hashed dependency
  output, so this is an artifact-identity/harness failure—not Qwen inference
  or cache evidence.
  A new exact packed run remains required.
- Run the real-model Gemma N=4/N=8, transaction-boundary, and long-resume
  parity checks with the optimized release profile and bind that profile in
  the hardware receipt. Metal command scheduling is profile-sensitive; a
  debug-only timing divergence must not be mislabeled as production evidence.
- Compare Gemma long-resume output with a cold request through the same
  production graph, require two consecutive LCP engagements, and verify both
  resumed turns byte-for-byte. The former forced-linear control compared two
  numerically distinct prefill graphs and could report cache corruption when
  only the graph choice differed.
- Make the release-authority Gemma overlap timeout explicit and overridable.
  The protected gate now supplies 1,800 seconds for its fixed 175K-token
  full-attention prompt instead of inheriting a non-overridable 900-second
  curl cutoff. It also allows 180 seconds to observe cancellation at the next
  transaction boundary when a 4,096-token Metal call is already in flight at
  a 183K-token cursor; invalid or non-positive values fail before dependency
  discovery or model traffic. Its hosted contract uses platform `grep` rather
  than assuming a runner has ripgrep installed.
- Stabilize Qwen's automatic-tool acceptance prompt around the real operator
  path `/opt/hf2q/Cargo.toml`: the system turn now tells the coding agent to
  invoke tools directly instead of imitating them in Markdown, and the mock
  tool result explicitly says the completed `read_file` call must not be
  repeated. The packed candidate still supplies the returned Cargo.toml bytes.
  Receipts bind SHA-256 identities for both prompt surfaces, and publication
  rejects a different path, prompt, or result envelope.
- Keep Gemma's four-slot release-default agentic latency limits unchanged,
  while giving the experimental eight-slot correctness/aggregate-cap probe an
  explicit 40-second cold-TTFT, 60-second whole-response, and 30-second
  tool-result functional envelope.
  On the exact M5 Max discriminator the eight-slot wave measured 25.279 seconds
  cold and 23.932 seconds at the slowest tool-result turn; every receipt keeps
  the actual timing, cache, and semantic fields rather than converting this
  experiment into an eight-slot latency claim. Protected run `31701418005`
  exposed that the wrapper had omitted the whole-response override and therefore
  inherited the shared 40-second default after the destructive long-context
  soak. A fresh exact-binary/model discriminator after 60 continuously Nominal
  seconds completed all eight cold responses in 27.943–28.023 seconds, reused
  7,112 tokens, and completed tool-result turns in 11.340–25.316 seconds. The
  explicit 60-second functional ceiling fixes the wiring mismatch without
  weakening the separate TTFT or tool-result bounds. Exact-main protected run
  `31714959928` then passed every preceding DeepSeek and four-slot Gemma gate,
  but process-b's eight-slot wave started after the destructive Gemma soak
  without its own thermal settle or telemetry and measured 40.882 seconds to
  first semantic output, 882 ms above the unchanged functional ceiling. This
  does not prove thermal throttling, and the correct cold responses/zero-cold-
  reuse state do not erase the timing failure; cached and tool continuations
  never ran. The eight-slot lane now gets an independent
  trailing 60 seconds of Nominal state after its eight-slot model is loaded,
  plus fail-closed two-second monitoring through the entire cold, cached,
  automatic-tool, SSE, and tool-result wave. Its thermal summary, raw logs,
  and exact eight cold-receipt hashes are independently bound and replayed at
  publication; a new exact packed run remains required authority.
- Keep that dispatch compatible with macOS Bash 3.2 under `set -u` by using
  explicit four-slot and eight-slot command branches instead of expanding an
  empty array. The release wrapper now also preserves the originating failure
  status through its cleanup trap, so a pre-manifest harness error cannot be
  reported as a successful workflow step after servers and guards are reaped.
- Make that receipt enforce the family-specific shipping gates as well as the
  shared lifecycle: Qwen overlap/continuation, cold four-agent heap waves, and
  one-slot disconnect; Gemma long-prefill overlap/cancellation, fresh-versus-
  reused 4,096/8,193 bounded-output parity, four/eight-slot aggregate caps,
  and heap waves; and
  DeepSeek cached-suffix cancellation, terminal parking, and two fresh
  four-agent waves. Release rehashes every downloaded receipt and validates
  the detailed evidence before registry credentials are exposed.
- Freeze the four-agent agentic repository context to the exact 21,204-byte
  calibration fixture instead of embedding the growing live README. The
  producer and protected release path now require its fixture ID, SHA-256,
  byte/character counts, exact 6,684-token DeepSeek render, zero cold reuse,
  semantic/tool proof, and the literal 60-second cold limits; a model-free
  negative matrix rejects stale or incomplete receipts.
- Run the two calibrated DeepSeek four-agent waves before the long functional
  lifecycle heat-soaks the M5 runner. Each wave now requires at least 60
  seconds of Nominal samples at five-second cadence with no hf2q/llama model
  runtime loaded, then fails if two-second telemetry becomes non-Nominal,
  malformed, or unavailable before all four atomic cold receipts arrive. The
  workload is not paused or reordered; cached work may overlap the cold tail as
  before. Thermal receipts bind all four cold-receipt names and hashes, while
  the same live cache must still finish cached unary/SSE, automatic tool
  choice, and tool-result continuation under unchanged limits. The cohort
  parent immediately reports a child that exits before publishing its cold
  receipt.
- Add a developer-only matched llama.cpp cold-wave discriminator for the
  frozen DeepSeek fixture. It binds peer binary/model/request identity, exact
  zero-cache `read_file` semantics, runtime-specific prompt-token counts,
  monotonic timing, AC power, and fail-closed thermal telemetry without making
  the reference runtime part of hf2q serving or release authority.
- Update vulnerable transitive dependencies (`crossbeam-epoch`,
  `quinn-proto`, `rkyv`, `anyhow`, `memmap2`, and the legacy Rustls chain via
  `ruvector-core 2.3`) and make a zero-vulnerability `cargo audit 0.22.2` run
  blocking in CI and release packaging.

## [0.1.5] — 2026-08-09

### Added — foreground serving dashboard

- Add a TTY-only live operator dashboard for foreground `hf2q serve` runs.
  It keeps one stable row per active request with slot, phase, cached versus
  new prompt tokens, prefill percentage/rate/ETA, and decode rate instead of
  printing one log line per transaction. `--operator-ui auto` is the default;
  `plain` preserves the traditional stream and `dashboard` fails early when
  stderr is not interactive or JSON logging is selected. Inference publishes
  through a bounded nonblocking channel, so terminal rendering cannot delay
  GPU or scheduler work.

### Fixed

- Refine DeepSeek lopsided cold-cohort scheduling with an interactive
  mixed-work budget of up to eight decode tokens between prefill slices capped
  at two native windows. When no runnable decoder remains, restore the proven
  full 2,048-token prefill transaction.
- Capture Qwen's stable native transcript boundary before the rewriteable
  generation cue and reuse it for normal continuations even when the generated
  live tail does not prefix-match the client's restored reasoning/tool history.
  Continuations restore that verified KV and prefill only the changed tail.

## [0.1.4] — 2026-08-09

### Fixed — bounded multi-agent prefill and Metal command-buffer lifetime

- Advance Qwen 3.5/3.6 slot-aware prefill in bounded, scheduler-yielding
  transactions. Active decoders run before each cold-prefill slice, cold
  prefills rotate fairly, and cache/ledger state is published only after every
  full-attention and MTP cursor agrees at a successful transaction boundary.
- Bound inline Qwen embeddings to one admission quantum, and reject
  soft-token/deepstack SlotAware requests explicitly until those multimodal
  paths have scheduler-yielding prefill and decode. No multimodal field is
  silently discarded into the plain-text graph.
- Treat Metal command-buffer timeouts and ignored submissions as
  SlotAware-worker-fatal for Qwen, Gemma, and DeepSeek. Active, detached, and
  queued requests terminate once, readiness fails closed, and the poisoned
  device generation receives no further GPU submissions or cache resets.
- Supervise individual Metal transactions from outside the model worker.
  A transaction that never returns poisons readiness, unblocks unary/SSE
  waiters, and requires process restart; slow SSE consumers are cancelled
  locally rather than backpressuring every slot. Fatal reply fanout is
  nonblocking even when a stream channel is already full.
- Preserve Qwen worker failures at the OpenAI boundary: invalid request shapes
  are rejected before Qwen LM scheduler/SSE admission and before Qwen LM
  generation, unsupported SlotAware capabilities
  return 501, queue/physical-budget pressure returns 429, and an unhealthy
  engine returns 503 instead of collapsing these cases into a generic 500.
- End Qwen's final encoder-session stage without opening an unused replacement
  command buffer. The companion `mlx-native` correction scopes Objective-C
  autoreleases at command-buffer, compute-encoder, and label-string seams.
  hf2q resolves the published, checksum-pinned `mlx-native 0.10.6`; no local
  Cargo patch participates in the 0.1.4 release.
- Graduate the validated Qwen3.6 autoregressive route to the default product
  surface. The canonical launcher no longer sets the investigation-only
  `HF2Q_QWEN36_AUTOREG` gate; unsafe chunk-scan remains a separate experiment.
- Add deterministic scheduling, cancellation, fatal-fanout, readiness,
  cross-layer cache-cursor, and 87,972-token/347-tool watchdog regressions.
- Advance long Gemma text prefills in bounded 4,096-token transactions, split
  exactly at the stable-prefix boundary, validate and publish every configured
  physical cache cursor only after a successful transaction, and run decode
  before prefill in `Mixed`. Cross-slot batches validate every lane before
  admission and share one 4,096-row aggregate transaction cap instead of
  multiplying that cap by the slot count. Already-installed long-text prefill
  states may share that same aggregate transaction while preserving per-lane
  cache, scheduler, and reply ownership. Long soft-token requests remain
  fail-closed until their own resumable path is proven.
- Route meaningful DeepSeek retained-prefix suffixes through the existing
  atomic resumable-prefill machinery. During a lopsided cold cohort, one
  decode token runs before each remaining prefill transaction; a terminal
  response stays parked until the cold-prefill barrier lifts, preserving the
  retained cache for its continuation. Cancellation and ordinary failure now
  reconcile the cohort back to an admissible phase instead of stranding an
  idle worker. Staggered warm continuations can join an existing decoder while
  another physical slot is free, and cancellation restores a valid,
  position-consistent pre-request turn anchor instead of deleting reusable KV.
- Add a focused DeepSeek cached-suffix gate that requires three resumable
  transactions with peer decode progress between them, exact middle-transaction
  cancellation accounting, no terminal Done after disconnect, retained-prefix
  reuse after rollback, readiness, and a clean fatal-log delta.

## [0.1.3] — 2026-08-08

### Fixed — registry verification and Eagle3 ordering

- Include the small DeepSeek encoding fixtures, quantizer byte-comparison
  fixtures, and continuous-batching source audit used by crate-local unit
  tests. The 0.1.2 runtime archive built and installed correctly, but
  `cargo test` on the downloaded crate could not compile or complete because
  those inputs were omitted.
- Order the Eagle3 FC-to-normalization, normalization-to-concat, and
  concat-to-projection dependencies explicitly when mlx-native uses its
  concurrent Metal encoder. This removes an intermittent bit-identity drift
  exposed by the packed-crate release gate without serializing the independent
  Q/K/V projections.
- 0.1.3 replaces the yanked 0.1.2 package.

## [0.1.2] — 2026-08-08

### Added — full-context multi-agent serving

- Slot-aware serving now gives every Gemma 4, Qwen 3.6, and DeepSeek V4
  agent the model's full logical context. Context is never divided by the
  number of concurrent slots; weights remain shared while each slot owns
  independent, demand-grown KV and recurrent state.
- Shared physical KV admission accounts for retained slot high-water and
  active worst-case growth. Requests queue or fail explicitly under physical
  pressure instead of silently shrinking context or overcommitting unified
  memory.
- Prefix-affine scheduling keeps each conversation on the slot with its best
  reusable native cache state. Normal OpenCode turns prefill only the appended
  suffix, including DeepSeek recovery anchors across cache-capacity growth.
- Canonical OpenCode launchers now default to four full-context slots and
  expose the shared KV budget, scheduler, cache reuse, time-to-first-token,
  prefill rate, and decode rate in verbose operator telemetry.

### Fixed — native agentic semantics across all three families

- Gemma 4 and Qwen 3.6 use their native GGUF chat templates and tool-call
  encodings; cross-family or incomplete template state fails closed rather
  than falling through an approximately compatible formatter.
- Tool definitions, required and automatic calls, source-shaped arguments,
  tool-result continuations, unary responses, and SSE streams are covered by
  realistic four-agent gates for Gemma 4, Qwen 3.6, and DeepSeek V4.
- DeepSeek cold prefill resumes only after atomic cache-and-ledger commits.
  At most two cold agents alternate matrix transactions through one scratch
  arena; the completed cohort then decodes in fair eight-token quanta, and
  retained-prefix continuations run before unrelated cold work can evict them.

### Performance and proof

- On the target M5 Max with AC power, two exact four-agent DeepSeek gates
  completed their cold cohorts in 53.86 and 52.32 seconds (53.09-second
  median), versus the then-observed approximately 54.1 seconds for matched
  llama.cpp with `--kv-unified`,
  four parallel slots, and 131,072 logical tokens per slot. llama.cpp's
  524,288-token unified allocation did not fit beside the 100 GiB model on
  this 128 GiB host; hf2q retained 524,288 logical tokens per slot through
  demand-grown physical admission.
- A DeepSeek continuation after 131,072-to-262,144 cache growth reused
  119,692 of 119,778 prompt tokens (99.92%) and reached the first semantic
  stream event in 1.132 seconds.
- Gemma's matched 24,200-token four-slot continuation sustained about
  1,831 aggregate prefill tokens/s, slightly ahead of the same GGUF and
  settings under llama.cpp at about 1,733 tokens/s. Gemma and Qwen four-agent
  gates retained at least 7,089 and 6,980 cached prompt tokens respectively,
  with native tool-result continuation and the full 262,144-token logical
  context per agent.
- `mlx-native` 0.10.4 supplies the family-neutral lazy overwrite allocation,
  ring linearization, F16 mask construction, and direct TQ-HB-to-F16 staging
  primitives used by the bounded full-context paths.

## [0.1.1] — 2026-08-06

### Added — DeepSeek V4 Flash agentic conversion and serving

- Rust-native conversion of the official
  `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint to the owned
  `deepseek4-agentic-q2` Q2_K/Q3_K/Q8_0 profile; no external converter,
  quantizer, or inference runtime is invoked.
- OpenAI-compatible DeepSeek serving for unary and SSE completions,
  reasoning, official DSML tool calls, required/automatic tool choice,
  multiple calls, and real OpenCode multi-turn coding sessions.
- Native live-prefix caching and reasoning-tail recovery. A 119,916-token
  post-tool request reused 119,813 tokens (99.91%) and reached its first
  semantic token in 1.275 seconds instead of recomputing the conversation.
- Adaptive gathered sparse prefill for long ratio-four compressed history,
  backed by `mlx-native` 0.10.1. On the source-bound M5 Max gate, hf2q
  completed the 119,821-token cold tool prompt in 494.449 seconds versus
  roughly 556 seconds for the same artifact under llama.cpp build 10293.
- Canonical OpenCode launcher and fail-closed agentic/long-context gates:
  `scripts/serve_deepseek4_opencode.sh`,
  `scripts/test_deepseek4_agentic.sh`,
  `scripts/test_deepseek4_opencode.sh`, and
  `scripts/test_deepseek4_long_context_cache.sh`.

### Added — Gemma 4 LCP long-resume past sliding_window + production hardening ("gemma-hybrid-lcp" follow-up)

- **LCP now works past `sliding_window` (1024) under the hybrid regime.**
  Sliding layers allocate linear buffers on both prefill routes;
  hybrid encode writes slot=logical at prefill and decode (capacity-
  derived predicate); the hybrid SDPA kernel's `mask_type=2` windowing
  is covered by `gemma_hybrid_long_resume_byte_identity`. The current
  release-candidate correction compares resumed and cold requests through
  the same production graph; the earlier forced-linear control was not a
  same-graph cache-coherence oracle.
  `HF2Q_KV_LCP_RESUME_CAPACITY=8g` is the documented envelope knob
  (snapshots carry +4096/turn multi-turn headroom).
- **Fixed: SerialFifo consume-gate 500 on growing conversations** — a
  stale `dense_kvs` mount from turn N hard-errored turn N+1 on the
  non-batched route when N+1's prompt was longer (`capacity < required`).
  Leftover mounts now drop + fresh-alloc; slot-aware scaffold bails
  preserved.
- **Fixed: pre-copy snapshot budget gate** — dual-leg snapshot bytes are
  estimated from shapes BEFORE the alloc+memcpy and skipped when over
  the registry budget; a ~97K-token dual-leg snapshot (~64 GB)
  previously swap-stormed the box before rejection could fire.
- **Launcher + envelope:** `scripts/serve_gemma4_opencode.sh` (one-model
  OOM guard, mmproj, `BATCHED=0` forces the linear-memory non-batched
  route for >32K contexts — the batched route's O(n²) bf16 masks OOM
  past ~32K at ~100K-token prompts). opencode-scale ~100K sessions are
  documented as qwen-arch territory (fixed-size DeltaNet + TQ full-attn).
- **Documented pre-existing divergences (reproduced at clean main, NOT
  introduced by the gemma-hybrid-lcp arc):** batched-prefill sliding-
  layer output diverges from the non-batched parity reference at seq>sw;
  `iter5_r_c4_lcp_5_fraction_sweep` fraction 0 (tiny K) diverges —
  dense-side, follow-up work.

### Added — Gemma 4 LCP partial-prefill resume under the production hybrid regime ("gemma-hybrid-lcp")

- **Gemma 4 has prefix caching in production for the first time.** The
  LCP resume path previously restored only dense F32 K/V and its
  snapshot was gated `HF2Q_USE_DENSE=1` — under the production hybrid
  regime (F16-K + TQ-HB V, `HF2Q_HYBRID_KV` default-on) no snapshot was
  ever taken, so every multi-turn request re-prefilled the full
  conversation.
- Root-cause fix for a silent production gap: the iter-344 batched
  prefill route (`forward_prefill_batched`, the SerialFifo default for
  gemma chat) populated **neither** LCP snapshot — so the registry
  stayed empty and every probe missed, for dense AND hybrid regimes
  alike. Both snapshots are now populated there.
- New regime-aware per-layer registry payload `GemmaLcpLayerKv`
  (`Dense` / `DenseAndHybrid`): prefill attention reads the dense leg,
  decode under hybrid reads the hybrid leg — an LCP resume restores
  BOTH, closing the same silent-corruption class ADR-027 sub-iter 23d-γ
  closed for qwen35. Regime-consistency check at install rejects
  cross-substrate entries (dense-only entry under hybrid = clean miss,
  never a zero-restore).
- `effective_kv_lcp_resume` widened: resumable substrates are now
  gemma-dense, gemma-hybrid, and qwen35-TQ (23d-γ); the HB-encoded
  opt-out regime stays auto-disabled. LCP is now default-on for
  production gemma and qwen35 serves.
- Gates: durable integration test
  `gemma_hybrid_lcp_partial_prefix_byte_identity` (two-server,
  engagement-asserted) + live ENGAGED trace `K=516 of N=537` with
  byte-identical output vs cold control (non-streaming and streaming).
  Known boundary: gemma LCP still skips prompts > `sliding_window`
  (1024) — LONG_RESUME is dense-only today; hybrid long-resume is
  follow-up work.

### Fixed — TQ-only LCP resume + disk persist made production-correct (ADR-027 sub-iter 23d-γ)

- **Silent coherence corruption fixed in TQ-only LCP resume.**
  `HybridKvCache::restore_partial` now partial-copies the first-`n_tokens`
  positions of all four TQ buffers (U8 packed + F32 norms) per slot for
  full-attn AND MTP slots. Previously, under production `tq_kv_active=true`,
  TQ buffers stayed zero-initialized after an LCP resume while
  `current_len` advanced — the resumed request attended over zeroed K/V
  for the entire cached prefix.
- **`cfg_from_cache` no longer panics in TQ-only mode** (was an engine
  worker panic → HTTP 500 on any request with `HF2Q_KV_PERSIST` set).
  Shape is derived from `slot.tq.k_packed` when F32 backing is absent;
  new `KvSubstrate` (F32Only/TqOnly/Both) classification with a
  uniform-substrate invariant check.
- **QH35 disk codec bumped to v4** — per-MTP `kv_present` byte (mirrors
  the full-attn v2 byte; v1..v3 envelopes still deserialize), and
  TQ blobs now deserialize into logical 4-rank shapes (was flat rank-1,
  which hard-failed `restore_partial` on hydrated snapshots).
- **On-disk fingerprint is substrate-namespaced** — snapshots written
  under one KV substrate never hydrate into a cache allocated under
  another (cross-mode hydration would have silently zero-restored; now a
  clean cache miss).
- Live gates (qwen36 APEX Q5_K_M, M5 Max): needle-recall byte-identical
  cold vs LCP-resumed (TTFT 2920ms → 326ms = 9.0×), 182 MB checkpoints
  written through to disk, cold-process restart hydrate byte-identical.
  Unit: 33 persistor + 189 kv_cache + 228 kv_persist tests green.
- **`Qwen35DiskPersistor` now enforces `HF2Q_KV_PERSIST_BUDGET_BYTES`**
  with LRU-by-mtime eviction per cfg subdir — pre-fix, one ~100K-token
  opencode session wrote 105 GB of chunk snapshots unbudgeted.
- New canonical launcher `scripts/serve_qwen36_opencode.sh` (full
  production env incl. `HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE=4096` for
  long-context checkpoint economics).

### Fixed — tool-call grammar for structured parameters (ADR-005 iter-231)

- **Agentic clients (opencode, MCP tool servers) can now call tools whose
  schemas declare `object` / `array` parameters.** Previously every
  `/v1/chat/completions` request carrying such a schema failed with
  HTTP 400 (`parameter 'X' uses unsupported schema type 'object'` — the
  wave-2.5 scalar-only emitter gate).
- Tool-call GBNF emitters (Qwen 3.5/3.6 + Gemma 4) now compile nested
  schemas recursively with full declared-structure fidelity: per-key
  value grammars, any-order `required` (≤8) permutation, typed array
  `items`, enums, type unions, `anyOf`/`oneOf`, and
  `additionalProperties:false` key-set closure. Free-form objects
  (no declared `properties`/`items`) compile to permissive recursive
  JSON value rules matching exactly what the chat templates emit
  (`tojson` / `format_argument`).
- The JSON Schema `pattern` regex keyword is now COMPILED into the
  tool-call value grammars via a real regex→GBNF compiler
  (`grammar/regex_gbnf.rs`, iter-231c) — literal/class/group/alternation/
  quantifier support with honest errors for non-regular features
  (backreferences, look-around, property classes). This unblocks MCP
  tool schemas like ruvnet-brain's `argv` items (`^[a-z][a-z0-9-]*$`).
- Schemas using features the grammar cannot enforce (`allOf`, `$ref`,
  tuple `items`, …) now return an honest 400 naming the feature and
  dot-path instead of failing the whole request class.
- Gemma 4 tool-call parser now round-trips nested structured arguments
  (objects/arrays containing commas) into `arguments_json` as real JSON
  instead of mangling them into string fields.
- Top-level untyped parameters accept structured values (both families).
- Also fixed two pre-existing sweep failures: H1 structural audit
  (`GOLDEN_OUTPUT` hoisted to module scope) and upstream citation drift
  (qwen35/qwen35moe catalog line numbers re-transcribed against current
  llama.cpp); cleared 8 pre-existing build warnings. Full sweep:
  4120 passed / 0 failed / 0 warnings.

## [0.1.0] — 2026-05-16

First public release.

### Added — convert pipeline

- HuggingFace → GGUF / mlx-lm safetensors converter (`hf2q convert`).
- Quantization families:
  - Float passthrough — `f16`, `bf16`, `auto`.
  - Legacy block — `Q2`, `Q4` (alias `Q4_0`), `Q8` (alias `Q8_0`).
  - K-quants — `Q2_K{,_S}`, `Q3_K_{S,M,L}`, `Q4_K_{S,M}`, `Q5_K_{S,M}`, `Q6_K`.
  - Imatrix-weighted K-quants — `imatrix-*` variants plus `imatrix-adaptive`.
  - Mixed-bit — `dynamic-quant-{4-6, 4-8, 6-8, 2-8}`.
- DWQ training (`hf2q dwq-train`) producing an mlx-format safetensors
  overlay layered on top of a GGUF (ADR-020).
- Two-pass intermediate GGUF for activation capture during
  Qwen 3.5 / 3.6 conversion (ADR-012).
- Streaming convert pipeline with disk-floor preflight (ADR-014).

### Added — inference + serving

- Apple-Silicon-only inference path on top of `mlx-native` (ADR-008).
- GGUF reader, KV cache, prefill, and decode for:
  - Gemma 4 (dense + MoE).
  - Qwen 3.5 / 3.6 (dense + MoE + multi-token-prediction).
  - Qwen 3-VL (vision + text).
  - BERT / Nomic-BERT (embedding-only).
- TurboQuant 8-bit KV cache (ADR-007) with Hadamard packing — drops
  F32 K/V allocations on Qwen 3.6 35B-A3B at 32K context for a
  **3.94× memory savings** vs the F32 baseline (ADR-027 iter-34).
- Flash-Attention prefill kernel (ADR-011) — `1.07–1.09× peer-FA`
  AHEAD at `pp1800–pp3700` (ADR-029 iter-160).
- Batched prefill V3 with byte-identical decode parity (ADR-029 Step
  1j.2).
- Speculative decode (ADR-030) — n-gram drafter Plan B available
  opt-in via `HF2Q_SPEC_NGRAM=1` (default OFF; DFlash investigation
  closed without shipping).

### Added — HTTP API (`hf2q serve`)

- OpenAI-compatible endpoints: `/v1/chat/completions`,
  `/v1/embeddings`, `/v1/models`.
- Streaming SSE.
- Tools / function-calling.
- Vision (`qwen3vl`) via `image_url` data-URI and HTTPS fetch.
- Grammar-constrained sampling.
- Persistent block-prefix KV cache (ADR-017).
- Multi-model serving from a single process.
- Jinja2 chat-template rendering matching `llama.cpp`'s vendored
  Jinja parser and mlx-lm's Python behavior, including the
  `pycompat` adapter for `.split()` / `.strip()` on strings
  (ADR-005 Phase 2a iter-133).

### Added — operator tools

- `hf2q doctor` — hardware / cache / RuVector / disk diagnostics.
- `hf2q info` — inspect an HF or GGUF model without converting.
- `hf2q validate` — cosine similarity, KL divergence, perplexity vs a
  reference.
- `hf2q parity` — ADR-009 parity validation against locked reference
  outputs.
- `hf2q smoke` — ADR-012 end-gate smoke test for a registered
  architecture.
- `hf2q gguf-patch` — rewrite GGUF metadata in place.
- `hf2q cache` — manage `~/.cache/hf2q/`.
- `hf2q completions` — shell completions.

### Performance highlights (M5 Max, thermal-fair alt-pair, σ < 1% per arm)

- **Gemma-4 decode** — `1.05× peer-FA` AHEAD across
  `tg200 / tg2000 / tg5000` after ADR-029 Step 1i (parallel
  SG-tournament top-K in `fused_moe_routing`) + Step 1j.2 (V3
  batched softmax tree-reduce). Byte-identical greedy decode vs V2
  baseline.
- **Qwen 3.6 35B-A3B-APEX-Q5_K_M decode** — `~1.34× peer-FA`
  sustained to 1000-tok (ADR-028 iter-308 → iter-324).
- **Prefill** — `1.07–1.09× peer-FA` AHEAD at `pp1800–pp3700`
  (ADR-029 iter-160).
- **KV-cache footprint** — 3.94× vs F32 baseline at 32K context on
  Qwen 3.6 35B-A3B (ADR-027 iter-34, Hadamard-packed TQ path).

### Notes

- macOS / Apple Silicon only (M1 or newer). The inference path is
  Metal-only by design (ADR-008 "candle divorce").
- DWQ at the production-default `perturb=1.0` is mathematically
  equivalent to the underlying K-quant baseline (ADR-020 finding
  2026-05-08). Wins materialize only at lower perturb values that
  move scales / biases off the K-quant projection.
- Per-arch disk floor for convert: 100 GB (Qwen 3.5 dense),
  150 GB (Qwen 3.5 MoE). Smoke preflight refuses to start below
  `disk_floor_gb + 10`.

[Unreleased]: https://github.com/robertelee78/hf2q/compare/v0.1.15...HEAD
[0.1.15]: https://github.com/robertelee78/hf2q/compare/v0.1.14...v0.1.15
[0.1.14]: https://github.com/robertelee78/hf2q/compare/v0.1.13...v0.1.14
[0.1.13]: https://github.com/robertelee78/hf2q/compare/v0.1.12...v0.1.13
[0.1.12]: https://github.com/robertelee78/hf2q/compare/v0.1.11...v0.1.12
[0.1.11]: https://github.com/robertelee78/hf2q/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/robertelee78/hf2q/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/robertelee78/hf2q/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/robertelee78/hf2q/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/robertelee78/hf2q/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/robertelee78/hf2q/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/robertelee78/hf2q/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/robertelee78/hf2q/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/robertelee78/hf2q/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/robertelee78/hf2q/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/robertelee78/hf2q/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/robertelee78/hf2q/releases/tag/v0.1.0
