# ADR-015: mlx-native — general decode-path speed improvements (qwen35 + gemma)

- **Status:** Proposed (Phase 1 single-CB hypothesis revised by P2 calibration — see Diagnosis update §2026-04-26)
- **Date:** 2026-04-26 (initial); revised same-day after P2 empirical data
- **Authors:** Robert E. Lee + Claude Code
- **Successor of:** ADR-012 §Optimize / Task #15 (closed at 0.94× of llama.cpp; cited the need for a "new mlx-native perf ADR")
- **Siblings of:** ADR-013 (qwen35 inference — owns the qwen35 forward path being rewritten); ADR-006 (mlx-native GPU backend — owns the Gemma `forward_decode` path being rewritten)
- **Standing requirement:** "as fast as our peers" applies to **every shipped model family** — `feedback_shippability_standing_directive`, restated 2026-04-26: *"we need this coherence and speed for qwen and gemma families of models"*. ADR-015 covers both.

## Context

ADR-012 closed the qwen35moe **conversion** scope at 0.94× of llama.cpp on
`dwq46` `n=256` decode (109.1 vs 116.0 t/s, M5 Max, same-day median per
`project_end_gate_reality_check`). Per-CB GPU time matches llama.cpp; the
gap is **CPU-side encoding/scheduling overhead**.

### The structural diagnosis

> **§Supersession note (Wave 2a, 2026-04-26 / 2026-04-27 UTC; rev1
> after Codex review 2026-04-27):** the P3b suspect list inside
> §"Budget reconciliation" below — items 1–4 (helper-function
> indirection, buffer-pool acquisition, barrier bookkeeping, KV-cursor
> updates) — was a **static-evidence** prediction drawn from grep
> against the qwen35 hot-path files, not from a live trace.  §"P3a'
> live profile pass" below replaces those static rows with **xctrace
> TimeProfiler measurements** captured from a 3-trial cold-SoC run on
> M5 Max (macOS 26.4.1, qwen3.6-27b-dwq46 fixture, 64 decode tokens
> / trial).  When the live trace and the static suspect list disagree,
> the live trace wins per
> `feedback_ground_truth_is_what_we_can_measure_now` and
> `project_metal_compiler_auto_optimizes_static_levers`.
>
> **Scope of supersession (rev1):** §P3a' identifies dense-fixture
> levers (rank-1 `MlxDevice::alloc_buffer` → `IOGPUResourceCreate` →
> `mach_msg2_trap` Mach-IPC chain; rank-3 `command_encoder()` churn;
> rank-4 `build_rope_multi_buffers` per-call alloc) and falsifies the
> Wave-1 static-evidence suspect list as the lever ranking for the
> qwen35 family.  It does **NOT** decompose the apex-MoE 288 µs/token
> residual — the dense fixture's per-token wall (~30 ms) is ~3.3× the
> apex MoE workload (~9 ms/token), so absolute µs/token from this run
> cannot be credited against the 288 µs apex residual.  The
> apex-MoE 288 µs residual decomposition **stands OPEN until Wave 2b
> apex 35B-A3B re-measurement** (5 cold trials, median aggregation,
> H1 literal `fn proj` site sized, H4 counter-based barrier accounting
> landed).  See §"Codex review acceptance — Wave 2a P3a' rev1" below
> for the full Wave 2b hard-gate list.

| Path | CBs / decode token | CPU encode overhead at ~5µs/CB |
|---|---:|---:|
| **llama.cpp** (`ggml-metal-context.m:458`) | 1–2 | ~5–10 µs |
| **hf2q Gemma** ⚠ corrected — see [P1 audit (merged) — gemma](#p1-audit-merged--gemma-decode-cb--graphsession-boundaries) | **1** (default `HF2Q_DUAL_BUFFER=0`) / **2** (default split) | ~1.6 µs / ~3.2 µs |
| **hf2q qwen35** (95 `device.command_encoder()` call-sites in `inference/models/qwen35/`) | ~102 (measured `fa3f9d6`) | ~510 µs |

> **§Correction (Wave 1 P1 audits, 2026-04-26):** the prior gemma row cited
> `forward_mlx.rs:1309` ("one GraphSession per layer (30 sessions per forward
> pass)") as evidence for ~30 CBs/pass.  Both the Claude and Codex Wave 1 P1
> audits independently refute that claim against current code.  The line:1309
> doc comment is **stale** — the in-function contract comment at
> `forward_mlx.rs:1488` reads *"SINGLE SESSION: Embedding + All 30 Layers + Head
> — ONE begin() → all GPU dispatches → ONE finish()."*  The gemma production
> hot path is **1 GraphSession / decode token** (`HF2Q_DUAL_BUFFER=0`) or
> **2 GraphSessions / decode token** (`HF2Q_DUAL_BUFFER` default — intentional
> async-overlap split at `forward_mlx.rs:3193-3194`).  Implication: the gemma
> half of D1 is largely refactor-for-uniformity, not a perf lever; gemma's
> actual gap (vs llama.cpp) sits in P3b orchestration overhead and possibly in
> P0 (the gemma baseline still has no committed value).  See the
> [P1 audit (merged) — gemma](#p1-audit-merged--gemma-decode-cb--graphsession-boundaries)
> subsection for site-by-site enumeration.

llama.cpp's own measurement: *"tests on M1 Pro and M2 Ultra using LLaMA
models, show that optimal values for n_cb are 1 or 2"* (`ggml-metal-context.m:458`).
Their `dispatch_apply` parallel-encoding mechanism (`:550`) targets
**prefill**, where `n_cb` may be larger; for decode, `n_cb=1` is fastest.

### Why prior CB-fusion attempts moved the needle by noise

`fa3f9d6` (small-scope CB fusion, FullAttn `sdpa+ops6-7` → 1 CB) recovered
~40 µs/token at +0.5% within noise — exactly the slope predicted by ~5 µs/CB:
the experiment removed 10 CBs / token. That data point **confirms the
theory** at one decimal place; it just couldn't statistically distinguish
+0.5% from zero at n=256 with 3 cold runs. Removing all ~100 CBs (going
to 1–2) projects to ~500 µs / token saved ≈ the entire 6% deficit.

### Why dispatch_apply alone isn't the lever for decode

`dispatch_apply` parallelizes encoding **across CPU threads**; CB count
**stays the same**. To recover 0.5 ms / token of encode overhead the CB
count itself has to drop by ~50× — not the encode wall-clock per CB.
The lever for decode is **single-CB forward pass**, not parallel encoding.

(Prefill is a separate axis — see Non-Goals.)

### 2026-04-26 — DIAGNOSIS UPDATE FROM P2 CALIBRATION

The working assumption above (~5 µs/CB) was a citation, not a measurement.
P2 (mlx-native `examples/cb_cost_calibration.rs`) measured it directly on
M5 Max:

| N    | regime     | wall_ms (median of 5) | µs/CB |
|-----:|:-----------|----------------------:|------:|
|   10 | async      |                 0.056 |  5.61 |
|   10 | sync       |                 0.154 | 15.39 |
|   10 | alloc+cmit |                 0.041 |  4.05 |
|  100 | async      |                 0.174 |  1.74 |
|  100 | sync       |                 1.694 | 16.94 |
|  100 | alloc+cmit |                 0.179 |  1.79 |
|  500 | async      |                 0.858 |  1.72 |
|  500 | sync       |                 6.688 | 13.38 |
|  500 | alloc+cmit |                 0.826 |  1.65 |
| 1000 | async      |                 1.583 |  1.58 |
| 1000 | sync       |                13.484 | 13.48 |
| 1000 | alloc+cmit |                 1.617 |  1.62 |

The **production hot path is `async`** (`enc.commit()` per layer, single
terminal `commit_and_wait` for argmax download).  Steady-state
**async µs/CB ≈ 1.6 µs** at N≥100 — **3.1× lower than the working
assumption**.

**Re-derived budget for hf2q qwen35 decode:**

| Component | CBs | µs/CB | total |
|---|---:|---:|---:|
| hf2q qwen35 (102 CBs) | 102 | 1.6 | **163 µs** |
| llama.cpp (1–2 CBs) | 1.5 | 1.6 | **2 µs** |
| **CB-overhead gap** | | | **~160 µs** |

Measured hf2q→llama.cpp gap = **~500 µs / token** (`fa3f9d6` /
ADR-012 §Optimize).  CB-count alone explains **~32%** of the gap, not the
~100% the original 5 µs/CB working assumption implied.

**Implication: single-CB rewrite is necessary but insufficient.**  Even a
perfect 102→1 CB collapse leaves ~340 µs / token (~68% of the gap)
unexplained.  The remaining cost likely lives in **per-dispatch
encoding** (hf2q decode issues ~1070 dispatches/token vs llama.cpp's
similar count, but with a different per-dispatch cost profile) and/or
**Rust-side orchestration overhead** (helper-function indirection,
buffer-pool acquisition, barrier bookkeeping).

ADR-015 phasing is updated below: **P3a** measures per-dispatch cost
directly before P3 commits to the single-CB rewrite, so the upper bound
on the rewrite's win is established empirically before multi-day
implementation work is spent.

### 2026-04-26 — P3a CALIBRATION DATA

`mlx-native/examples/dispatch_cost_calibration.rs` measures per-dispatch
encode cost using `scalar_mul_bf16` on a 1-element BF16 buffer (minimal
GPU work; CPU encoding cost dominates):

| N        | encode_ms (med of 5) | commit_ms (med of 5) | µs/dispatch |
|---------:|---------------------:|---------------------:|------------:|
|       10 |                0.006 |                0.189 |       0.629 |
|       50 |                0.012 |                0.589 |       0.249 |
|      100 |                0.018 |                0.198 |       0.185 |
|      500 |                0.084 |                0.345 |       0.168 |
|     1000 |                0.163 |                0.515 |       0.163 |
|     5000 |                0.784 |                1.575 |       0.157 |

**Steady-state µs/dispatch ≈ 0.16 µs** at N≥500 — within ±15% of
llama.cpp's implied ~0.14 µs / dispatch (per ADR-012 §Optimize ~150 µs
CPU encode for ~1070 dispatches).  hf2q's shader-launch path is **not
materially slower** than llama.cpp's.

### Budget reconciliation (P2 + P3a synthesized)

| Component | hf2q | llama.cpp | Δ |
|---|---:|---:|---:|
| **GPU work** (per ADR-012 §Optimize) | 8.45 ms | 8.38 ms | **+70 µs** |
| **CPU CB encode** (102 vs 1.5 CBs × 1.6 µs/CB) | 163 µs | 2 µs | **+161 µs** |
| **CPU dispatch encode** (~1070 dispatches × 0.16 vs 0.14 µs) | 171 µs | 150 µs | **+21 µs** |
| **Sub-total accountable** | | | **+252 µs** |
| **Measured wall delta** (1/109.1 − 1/116.0 t/s) | 9.16 ms | 8.62 ms | **+540 µs** |
| **Residual unaccounted** | | | **+288 µs** |

**The biggest single lever is the residual ~288 µs (~53% of gap)**, not
the CB-encode delta (~30%).  The residual cannot be CB or dispatch
encoding — both are now measured.  Suspects (in priority order):

1. **Helper-function indirection** — every `apply_*` helper in
   `inference/models/qwen35/` does multiple short-lived allocations
   (`MlxBuffer` for params, scratch shapes) and Result wrapping.  ~95
   helpers per token × Rust function-prologue cost is plausibly
   double-digit µs each.
2. **Buffer pool acquisition** — `pooled_alloc_buffer` calls walk the
   in-use list and bucket size on every call.
3. **Memory barrier bookkeeping** — `enc.memory_barrier()` mutates
   `pending_reads`/`pending_writes` Vecs (capture mode + RAW/WAR
   tracking).
4. **KV-cache cursor updates** — `slot.current_len[0] += 1` per
   FullAttn / DeltaNet layer with the associated buffer fence.

The **lever for closing the residual** is a Rust-orchestration sweep
(profile guided), not architecture.  Single-CB still helps because it
removes 100 of the encoder lifetimes plus barrier-bookkeeping
state-machine transitions, but the orchestration sweep is the bigger
chunk.

### P3a' live profile pass — Rust orchestration residual contributors (LIVE)

**Provenance.** CFA session `cfa-20260426-adr015-wave2a-p3aprime`,
review-only single-worker mode (perf-engineer + coder + tester +
architect chain consolidated).  Tooling: **xctrace TimeProfiler**
(Instruments, Xcode 16.0 17E202) — chosen over `cargo flamegraph` to
avoid SIP-modification on macOS 26 / Apple Silicon and because xctrace
symbolicates Rust release-mode frames cleanly without a special build.
Harness: `scripts/profile-p3aprime.sh`.

**Methodology.**

- **Hardware:** M5 Max (Apple Silicon arm64), macOS 26.4.1 build 25E253,
  Darwin 25.4.0.
- **Fixture:** `/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf`
  (16 GB) — Qwen3_5ForConditionalGeneration, 64 layers
  (48 `linear_attention` DeltaNet + 16 `full_attention` gated),
  hidden=5120, intermediate=17408, head_dim=256.
  Dense-FFN-Q hot path (`build_dense_ffn_layer_gpu_q`),
  not the apex MoE expert-routing path.
- **Workload:** `hf2q generate --prompt "Hello, my name is" --max-tokens 64
  --temperature 0`.  64 greedy decode tokens.  Reported throughput:
  ~33.5 tok/s (single-token decode wall ≈ 30 ms).
- **Cold-SoC gate:** `pmset -g therm` snapshot before each trial;
  no thermal/performance/CPU-power warning recorded any trial.  60 s
  thermal settle between trials.
- **Trial count:** 3.
- **Sampling:** xctrace TimeProfiler default (1 ms per sample).
- **Aggregation:** filter samples to stacks containing
  `forward_gpu_greedy` (decode-only entry; excludes the cold prefill
  and model-load segments that dominate the unfiltered trace).
  Inclusive ms summed per frame across all 3 trials, then divided by
  192 (= 3 trials × 64 tokens) to derive µs/token.

**Run artifacts.** Live traces (large, **not committed** — referenced
by path):

- `/tmp/cfa-adr015-wave2a-p3a-prime/run-20260427T030903Z.metadata.json`
  (run-wide hardware metadata)
- `/tmp/cfa-adr015-wave2a-p3a-prime/trace-{1,2,3}-20260427T030903Z.trace`
- `/tmp/cfa-adr015-wave2a-p3a-prime/topcalls-{1,2,3}-20260427T030903Z.txt`
  (xctrace XML export)
- `/tmp/cfa-adr015-wave2a-p3a-prime/aggregate-decode.txt` (full
  hypothesis-frame aggregation; reproducible by
  `python3 /tmp/cfa-adr015-wave2a-p3a-prime/aggregate_decode.py`)

**Trace provenance — pinned to reviewed commit.** The 3 traces under
`/tmp/cfa-adr015-wave2a-p3a-prime/` were captured against `/opt/hf2q`
at parent commit **b7a01cc** (recorded in
`run-20260427T030903Z.metadata.json:16-17` as `hf2q_git_head`).  The
delta from `b7a01cc` to the reviewed commit `9e6a462` is
**docs-only + harness + .gitkeep** (verified: `git diff --stat
b7a01cc..9e6a462` lists `docs/ADR-015-mlx-native-single-cb-decode.md`,
`docs/perf-traces/.gitkeep`, `scripts/profile-p3aprime.sh` — no edits
to `src/`, `mlx-native/`, or any hot-path crate).  No re-run is
required to re-pin the traces to `9e6a462`; the trace data faithfully
represents the reviewed commit's runtime behavior.

**Aggregation methodology caveat (Codex AF3, accepted 2026-04-27).**
Hypothesis-frame inclusive ms in `aggregate-decode.txt:192-212` are
**summed across multiple needle patterns per hypothesis** — e.g. H1's
needles are `('gpu_ffn::proj', 'gpu_ffn::build_dense_ffn', 'alloc proj
dst')`.  Because Time Profiler inclusive frames nest (a
`build_dense_ffn` frame contains its child `alloc proj dst` frame),
summing them double-counts the inner frame.  This **inflates the H1
inclusive figure** in `aggregate-decode.txt` and contributes to the
"NOT FOUND at literal site, but category at 37×" framing being
overstated.  Wave 2b must rewrite the script to **report one canonical
frame per hypothesis plus named subcomponents** rather than summing
overlapping inclusive frames.  This fix lives in the `/tmp/` working
copy (`/tmp/cfa-adr015-wave2a-p3a-prime/aggregate_decode.py:129-132`);
since `/tmp/` is outside the repo, the fix is queued to Wave 2b along
with the 5-trial median aggregation and the H4 counter-based
accounting — not back-applied to the §P3a' tables here.

**Trial-to-trial variance.**

| Trial | decode samples | wall ms attributed | µs/token (decode-only) |
|---|---:|---:|---:|
| 1 | 397 | 397 ms | 6203 |
| 2 | 382 | 382 ms | 5969 |
| 3 | 696 | 696 ms | 10875 |

Trials 1 and 2 agree within 4 %.  Trial 3 is a 1.8× outlier despite
identical pre-trial `pmset -g therm` (no thermal warning recorded).
This is unexplained system noise.

**Outlier-bias disclosure (Codex review Q1, accepted 2026-04-27).** The
absolute µs/token figures reported in this section are 3-trial mean
(sum / 192).  Without trial 3, the rank-1 alloc_buffer chain shrinks
from **3719 µs/token → 3063 µs/token** — a **21 % outlier-bias** on the
headline number.  The qualitative finding (alloc_buffer + AGX init +
mach_msg2_trap dominate) holds in every individual trial and is not
sensitive to this bias, but the absolute sizing is.

**Rank-stability claim — scoped.** What is empirically supported across
all 3 trials is that the **rank-1 contributor (alloc_buffer chain)
appears in the top-3 frames every trial** (manual cross-reference of
per-trial `topcalls-{1,2,3}-*.txt` files).  The earlier wording
"rank order stable across all 3 trials" was overclaimed: per-trial
inclusive ms for H2 are [35, 42, 57] ms and for H3 are [40, 35, 72] ms
(see `aggregate-decode.txt:192-204`), which are not identical — only
the *contributor set* and rank-1 identity are stable.  The
`aggregate.py:160-168` rank-stability section is also broken (stores
tuples then calls `.split()` on them — Codex AF4); the supported claim
above is from manual cross-reference of the per-trial topcalls
exports, not from that script.

**Wave 2b methodology fix.** Re-run as **5 cold trials with median**
(not mean of 3) to make the headline robust against single-trial
outliers.  The Wave 2a numbers below are reported as-is for
forward-compatibility, with the 21 % bias disclosed.  Trial-3 is NOT
discarded silently — discarding would be selection bias; instead, the
without-trial-3 number is reported alongside.

**Coverage gap.** This fixture is **dense Qwen3.5 with quantized
`build_dense_ffn_layer_gpu_q`**, not the apex 35B-A3B MoE.  Implications:

- The MoE expert-routing path (`build_moe_ffn_layer_gpu_q`) is **not
  exercised** here — H1's 200-call/token claim from Wave 1 was framed
  partly against MoE expert proj calls; Wave 2b must re-validate the
  H1 verdict against the apex MoE fixture before committing P3b
  reductions to MoE-specific call sites.
- The cited file:line for H1 (`gpu_ffn.rs:397-404`) is the unquantized
  `proj()` helper (line 347–423); the dense-quantized hot path goes
  through `build_dense_ffn_layer_gpu_q` (line 578) which **does not
  call `proj()`** — it calls `quantized_matmul_ggml` directly.  H1's
  proj-allocation hypothesis is therefore not exercised on this
  fixture; we report what IS exercised (the 5–6 dense-Q
  intermediate-buffer allocations per FFN layer at line 592–606) as
  the dense-FFN analog.
- The DeltaNet path (`build_delta_net_layer`) is exercised (48 layers
  per token).  FullAttn `apply_imrope` is exercised (2 calls × 16
  layers = 32 calls / token).

**Top contributors — measured.** All values are **3-trial inclusive
sum / 192 tokens** unless noted.  "profile-source-line" = the live
xctrace frame name; "Fix proposal" / "Risk" are P3b candidate landings.

| Rank | Call site (file:line) | µs/token (LIVE) | profile-source-line | Fix proposal | Risk |
|---:|---|---:|---|---|---|
| 1 | `mlx_native::device::MlxDevice::alloc_buffer` → `IOGPUResourceCreate` → `mach_msg2_trap` (**dominant low-level fixable contributor in dense-FFN-Q decode**; **with trial 3:** 3719 µs/token; **without trial 3:** 3063 µs/token, 21 % outlier-bias) | **3719** (mean) / **3063** (mean ex-T3) | `mlx_native::device::MlxDevice::alloc_buffer::he10f952fea8b1eef` (714 ms over 192 tokens); leaf is `mach_msg2_trap` (612 ms / 3187 µs/token) inside `IOConnectCallMethod`. Attribution = synchronous CPU time in Metal/IOGPU resource creation; mach_msg2_trap is the sampled leaf. | **Pool every per-decode-token GPU buffer** (DeltaNet conv/recurrent scratch, FullAttn KV-shape staging, FFN gate/up/hidden/down_out/silu_params, residual-sum). Each `alloc_buffer` today incurs one Mach IPC roundtrip via `IOGPUResourceCreate`. A bucketed pool reused across decode tokens removes ~3.7 ms/token of Mach-trap latency on this dense fixture. **Hazards (Codex review Q3, accepted 2026-04-27)** — pooling is sound only with: (a) **strict CB lifetime fences** (no buffer returned to pool until its consuming command buffer reaches `MTLCommandBuffer.status == .completed`); (b) **no `reset` before all users complete**; (c) **byte_len/shape correctness** for any CPU-read buffer (the existing `proj()` carve-out at `gpu_ffn.rs:391-396` exists precisely because power-of-two bucket rounding inflates `byte_len()` — pool MUST preserve exact requested length for `download_f32`/`as_slice` paths); (d) **peak working-set caps** + **LRU eviction** to bound RSS. | M (R2: pool may inflate peak working set; mitigate per (d)). |
| 2 | `gpu_ffn::build_dense_ffn_layer_gpu_q` (gpu_ffn.rs:578-680) | **4078** | `hf2q::inference::models::qwen35::gpu_ffn::build_dense_ffn_layer_gpu_q::hbb763b35a276433b` (783 ms / 192 tokens) | Inclusive — most of the cost (≥3300 µs) is the rank-1 alloc_buffer chain (5–6 allocs per FFN layer × 64 FFN layers / token = 320–384 allocs / token). The remainder (~780 µs) is encoder construction + barrier + kernel dispatch. **Same fix as rank 1** + collapse ops 1+2 (gate/up) into a single batched `quantized_matmul_ggml` (B=2) so 1 dispatch instead of 2. | M (batched mv-mul kernel exists upstream; verify M5 perf is at parity). |
| 3 | `mlx_native::device::MlxDevice::command_encoder()` (~104 calls/token; see P1 audit) | **724** | `mlx_native::device::MlxDevice::command_encoder::hb128148f75021861` (139 ms / 192 tokens). Leaf chain includes `-[AGXG17XFamilyCommandBuffer computeCommandEncoderWithDispatchType:]` and `-[AGXG17XFamilyComputeContext initWithCommandBuffer:config:]` (252 ms + 240 ms inclusive over the same 192-token window). | **P3 single-CB rewrite** is the structural fix. With ≥100 of the 104 encoder constructions collapsed into one, this entire 724 µs/token bucket should drop to <100 µs/token. | L (already the planned P3 lever; this is the live measurement that confirms its size). |
| 4 | `gpu_full_attn::apply_imrope` (gpu_full_attn.rs:361-412) — 32 calls/token (16 FullAttn layers × 2 for Q+K) | **464** | `hf2q::inference::models::qwen35::gpu_full_attn::apply_imrope::h9a36ffd998e736b0` (89 ms / 192 tokens) inclusive. **Of this, 208 µs/token is `mlx_native::ops::rope_multi::build_rope_multi_buffers`** (40 ms / 192 tokens) — params + sections buffers re-allocated on every call. | **Hoist `build_rope_multi_buffers` out of the per-call path**: the rope params + sections vary only with layer-id (constant within a model), so build once at model-load and reuse. Eliminates ~32 alloc_buffer calls/token plus the inclusive 208 µs/token. | L (sections array is a few i32s; trivial to pre-bake; bit-exact). |
| 5 | `mlx_native::ops::fused_norm_add::dispatch_fused_residual_norm_f32` | **500** | `mlx_native::ops::fused_norm_add::dispatch_fused_residual_norm_f32::hc30d7605df331fcc` (96 ms / 192 tokens) inclusive | Already fused on disk. The 500 µs/token is mostly downstream of the encoder + alloc cost shared with rank 1 / rank 3; closing rank 1 + rank 3 will shrink this row to <100 µs/token. No standalone fix recommended. | L. |
| 6 | `gpu_full_attn::build_gated_attn_layer` (gpu_full_attn.rs:1077+) | **1120** | `hf2q::inference::models::qwen35::gpu_full_attn::build_gated_attn_layer::h24b67219c7b61ab4` (215 ms / 192 tokens) inclusive | Inclusive — most cost is rank 1 + rank 3 + rank 4 already counted. Per-layer it is 16 × 70 µs/layer/token ≈ 1120 µs/token. **Coverage caveat:** part of this (apply_imrope, FullAttn proj allocs) is double-counted with rank 4. Track via single-CB end-to-end metric instead of stacking. | M (layer-wide refactor for single-CB). |

**Hypothesis register verdicts.** Each cell cites
`/tmp/cfa-adr015-wave2a-p3a-prime/aggregate-decode.txt`
(`HYPOTHESIS-FRAME EVIDENCE` block) as the live evidence; trial-3-summed
values are reported with the µs/token already divided by 192:

| ID | Site | Static estimate (Wave 1) | Live measured | Verdict |
|---|---|---:|---:|---|
| **H1** | `gpu_ffn.rs:397-404` proj unpooled dst alloc | 100 µs/token | **NOT FOUND IN TRACE** at the cited site (`fn proj`). Dense fixture's hot path is `build_dense_ffn_layer_gpu_q` (`gpu_ffn.rs:578`) which does **not** call `proj()` (line 347–423); it calls `quantized_matmul_ggml` directly at lines 626–629. Its 5–6 intermediate `alloc_buffer` calls per FFN layer × 64 layers/token aggregate as the rank-1 contributor at **3719 µs/token** (Mach IPC dominated). | **LITERAL SITE NOT MEASURED on this dense fixture; dense-FFN-Q alloc category analog CONFIRMED.** (Codex review Q4, accepted 2026-04-27 — the prior wording "category CONFIRMED at much larger magnitude" was a goalpost shift.) The literal `proj()` site at `gpu_ffn.rs:397-404` is exercised by the unquantized `build_moe_ffn_layer_gpu` (router-logits download path) and by MoE expert paths, **not** by the dense-quantized `build_dense_ffn_layer_gpu_q` exercised here. **HARD GATE for Wave 2b:** apex 35B-A3B MoE fixture re-measurement is required before any P3b reduction lands at the literal H1 site (`fn proj` allocation at line 397-404). The dense-FFN-Q alloc category result is sufficient justification for pooling the **dense intermediate-buffer set** (lines 592–606); it is **not** sufficient justification for changes to the literal `fn proj` allocation pattern, which must be measured on its actual exercising fixture (apex MoE). |
| **H2** | `gpu_full_attn.rs:383-395` + `mlx-native/rope_multi.rs:215-244` apply_imrope | 80 µs/token | **464 µs/token** inclusive on `apply_imrope`; **208 µs/token** of that is `build_rope_multi_buffers` per-call alloc. Searched: `apply_imrope`, `rope_multi`, `dispatch_rope_multi`, `build_rope_multi_buffers`. | **CONFIRMED, larger than estimated.** Live cost is ~5.8× the static estimate.  build_rope_multi_buffers hoisting (rank-4 fix) closes ≥208 µs/token by itself. |
| **H3** | `MlxDevice::command_encoder()` ~120×/token churn | 55 µs/token | **724 µs/token** inclusive on `mlx_native::device::MlxDevice::command_encoder`. Adjacent AGX leaf time (`computeCommandEncoderWithDispatchType:` + `ComputeContext::initWithCommandBuffer`) is +492 ms/192 tokens = +2562 µs/token but partly overlaps with H1's IOGPU init.  Treating only the directly-named Rust frame: **724 µs/token, ~13× the static estimate**. | **CONFIRMED, much larger than estimated.** P3 single-CB rewrite is the structural fix; this row goes to <100 µs/token after P3 lands. |
| **H4** | `enc.memory_barrier()` ~440×/token + ~35 µs | 35 µs/token | **NOT FOUND IN TRACE** for the literal frame name `memory_barrier`. Searched: `memory_barrier`, `enc.memory_barrier`. The `mlx_native::encoder::CommandEncoder::*` symbols are present (commit, commit_labeled at 8 ms / 192 tokens = 42 µs/token total) but `memory_barrier` does not appear as a sampled stack frame at 1 ms granularity. | **NOT OBSERVED — BELOW TIMEPROFILER 1 ms RESOLUTION.** Absence of a literal `memory_barrier` frame at 1 ms sampling is **not a falsification** of barrier coalescing (Codex review Q2, accepted 2026-04-27). The body at `/opt/mlx-native/src/encoder.rs:498-512` is an `objc::msg_send![encoder, memoryBarrierWithScope: ...]` call which can be inlined or attributed under sibling Objective-C/AGX frames. Methodology fix queued to **Wave 2b**: counter-based per-barrier accounting via temporary `#[inline(never)]` wrapper around the `objc::msg_send!` site, paired with an atomic call counter and `Instant`/`mach_absolute_time` total. H4 stays **OPEN** until per-barrier count and total time are measured at counter resolution (not 1 ms statistical sampling). |
| **H5** | `with_context(\|\| format!(...))` String allocation on success path | 25 µs/token | **5 µs/token** total `with_context` inclusive (1 ms / 192 tokens). No `fmt::format` frames in decode-only stacks. | **FALSIFIED (confirms Codex Wave 1 prior).** `with_context` takes a closure; `format!` only runs on Err.  On the success path, the closure is constructed but never invoked.  Sampled time is dominated by trait-call overhead, not allocation.  Expected from source review; live trace confirms. |

**Dense-fixture levers vs the apex-MoE 288 µs/token residual — credited
zero until Wave 2b.** The acceptance bar at the top of §"Budget
reconciliation" specifies a **288 µs/token residual against the
apex-MoE 540 µs/token gap at ADR-012 closure** (1/109.1 − 1/116.0 t/s,
`dwq46` `n=256`).  This §P3a' run was on **dense Qwen3.5
qwen3.6-27b-dwq46** with ≈30 ms/token decode wall — a different
fixture, a different workload, and ~3.3× the wall time.  Comparing the
dense per-token µs to the apex 288 µs is a **category error** (Codex
review Q5, accepted 2026-04-27).

**Honest accounting:**

- **Dense-fixture identifies candidate levers** — rank-1 `alloc_buffer`
  Mach-IPC chain, rank-3 `command_encoder()` churn, rank-4
  `build_rope_multi_buffers` hoist.  These are real, measured, and
  worth pursuing for the dense Qwen3.5 path itself (which is also a
  shipped family per `feedback_shippability_standing_directive`).
- **Apex-MoE residual decomposition remains OPEN** — no µs of the
  dense-fixture savings can be credited toward the apex MoE 288 µs
  residual until the same call-graph hot frames are re-measured on the
  apex 35B-A3B MoE fixture.  The relative ranking is *expected* to
  hold; that expectation is not data.
- **HARD GATE — Wave 2b apex MoE re-measurement gates P3b.** No P3b
  reduction may land on a "MoE-specific" or apex-residual basis until:
  (a) 5 cold trials × 64 decode tokens on the apex MoE fixture with
  median aggregation; (b) literal `fn proj` site (H1, `gpu_ffn.rs:397-404`)
  appears in the apex trace and its per-token µs is sized; (c) H4
  counter-based barrier accounting lands and falsifies-or-confirms the
  barrier-coalescing lever; (d) `aggregate_decode.py` double-count
  caveat (Codex AF3) is fixed.

**Wave 2a P3a' contributes 0 credited µs toward the apex-MoE 288 µs
residual.**  It contributes a measured candidate-lever shortlist for
the dense Qwen3.5 family and a falsified Wave-1 static-evidence
suspect list — both useful, neither sufficient to satisfy the acceptance
bar in §"Budget reconciliation".  The acceptance-bar status is
reflected in the supersession note at line 19: "§P3a' identifies
dense-fixture levers; the apex-MoE 288 µs residual decomposition
stands open until Wave 2b."

**Codex review acceptance — Wave 2a P3a' rev1 (2026-04-27).**

This subsection records the Phase-3 queen reconciliation of the
Codex read-only review against the implementer's commit `9e6a462`.
Full Codex JSON verdict is preserved at
`/tmp/cfa-cfa-20260426-adr015-wave2a-p3aprime/codex-review-last.txt`.

| Codex finding | Verdict | Revision applied |
|---|---|---|
| **Q1** trial-3 1.8× outlier inflates absolute µs/token | ACCEPT | Disclosed 21 % outlier-bias (3719 → 3063 µs/token without trial 3); rank-1 row now reports both means; "rank order stable across all 3 trials" replaced with the supported "rank-1 contributor appears in top-3 every trial" claim, sourced from manual cross-reference of per-trial topcalls (not the broken `aggregate.py:160-168` rank-stability section). 5-trial median methodology queued to Wave 2b. |
| **Q2** H4 "FALSIFIED" overclaimed at 1 ms TimeProfiler resolution | ACCEPT | H4 verdict changed to **NOT OBSERVED — BELOW TIMEPROFILER 1 ms RESOLUTION**. Counter-based per-barrier accounting (`#[inline(never)]` wrapper + atomic counter + `Instant`/`mach_absolute_time` total around `/opt/mlx-native/src/encoder.rs:498-512`) queued to Wave 2b. H4 stays OPEN. |
| **Q3** rank-1 wording "dominant inclusive frame" → "dominant low-level fixable contributor" | ACCEPT | Rank-1 row reworded; pooling hazard list added (CB lifetime fences, no reset before completion, byte_len/shape correctness for CPU-read buffers — referencing the existing `gpu_ffn.rs:391-396` carve-out for `proj()` — peak working-set caps + LRU eviction). |
| **Q4** H1 goalpost shift from literal site to "category" | ACCEPT | H1 verdict changed to **LITERAL SITE NOT MEASURED on this dense fixture; dense-FFN-Q alloc category analog confirmed**. Apex 35B-A3B MoE re-measurement marked HARD GATE before any P3b reduction lands at the literal `fn proj` site (`gpu_ffn.rs:397-404`). |
| **Q5** 13.6× headline is dense-vs-MoE category error | ACCEPT | 13.6× headline removed; replaced with explicit "dense-fixture identifies candidate levers; apex-MoE residual decomposition remains OPEN; Wave 2a P3a' contributes 0 credited µs toward the 288 µs apex residual" framing. |
| **AF1** trace provenance not pinned to reviewed commit | ACCEPT | Provenance paragraph added: traces taken at parent `b7a01cc`; delta to reviewed `9e6a462` is docs+harness+.gitkeep only (verified by `git diff --stat`); no re-run required. |
| **AF2** harness comment misstates dense fixture exercises `proj()` | ACCEPT | `scripts/profile-p3aprime.sh:29-35` updated in this commit to match the ADR. |
| **AF3** `aggregate_decode.py:129-132` double-counts overlapping inclusive frames | ACCEPT | Caveat added to aggregation-methodology paragraph; one-canonical-frame-plus-named-subcomponents fix queued to Wave 2b. Script lives in `/tmp/` so cannot be fixed in-repo. |
| **AF4** `aggregate.py:160-168` rank-stability section is broken (split on tuple) | ACCEPT | Documented in the rank-stability scoping paragraph; the supported claim came from manual cross-reference of per-trial topcalls files, not the broken script section. Script fix queued to Wave 2b. |

**Wave 2b hard gates** (must all clear before P3b reductions land):

1. Apex 35B-A3B MoE fixture re-measurement (5 cold trials × 64 decode
   tokens, median aggregation) with literal H1 `fn proj` site sized.
2. H4 counter-based barrier accounting wired around
   `/opt/mlx-native/src/encoder.rs:498-512` `objc::msg_send!`.
3. `aggregate_decode.py` rewritten to report one canonical frame per
   hypothesis plus named subcomponents (no overlapping-inclusive sums).
4. `aggregate.py:160-168` rank-stability section fixed (no `.split()`
   on tuples).
5. 5-trial median methodology applied to all absolute-µs/token claims
   (replaces the 3-trial mean used here).

**Verdict:** ACCEPT_WITH_REVISIONS.  The dense-fixture xctrace
artifacts are real and useful; the falsification of the Wave 1
static-evidence suspect list (item 1–4 in §"Budget reconciliation")
holds.  The acceptance-criterion ≥150 µs of the **apex-MoE 288 µs
residual** is **NOT MET** by this run; it is re-scoped as a Wave 2b
gate per the supersession note at line 19.

**Commit-history note (2026-04-27).** This rev1 revision content
landed at commit **`1c056c6`** ("docs(adr-014): record iter-3u commit
hash") rather than under the queen-prescribed message
`docs(adr-015 wave 2a iter P3a' rev1): apply Codex review revisions`
because of a staging-index race with a parallel ADR-014 session — the
parallel session's `git commit` swept the queen's already-staged
ADR-015 + harness diffs into its own commit before the queen's
`git commit` fired.  The commit was already on `origin/main` when the
race was detected, so an amend was unsafe.  The substantive content
(this §"Codex review acceptance" subsection, the H1/H4 verdict
changes, the rank-1 hazard list, the supersession-note rescoping, the
13.6× headline removal, and the harness-comment fix) is fully present
at `1c056c6` regardless of the misleading commit subject.  This
follow-up commit (queue-correct subject) adds only this commit-history
note for traceability.

### 2026-04-27 — P0 Gemma baseline (iter4) reshapes the phasing budget

**Measured:** hf2q gemma at 0.840× of llama.cpp at HEAD (P0 iter4).

**Implications for Phasing P3 / P3b ordering:**

1. **The gemma-side gap is *larger* than the qwen35 gap at ADR-012 closure
   (0.840× vs 0.94×).**  The diagnosis-update CB-count budget assumed
   qwen35's ~540 µs/token gap was the bigger problem; the gemma gap is
   ~16% (~1820 µs/token at decode wall ≈ 11.4 ms vs llama ≈ 9.57 ms),
   substantially deeper.  Per the P1 audit (merged) — gemma decode CB /
   GraphSession boundaries — gemma is ALREADY at 1–2 CBs/token; the gap
   does **not** live in CB-count.  All ~16% lives elsewhere: per-dispatch
   encoding, Rust-orchestration overhead (P3b's residual), and gemma-
   specific paths (TQ KV encode, fused MLP+MoE B8/B9/B10/B11 groups,
   iter-21 Track B Lloyd-Max codebook).

2. **D1 (single-CB rewrite for gemma) confirmed as refactor-for-uniformity
   not perf lever.**  The P1 audit already flagged this; the live P0
   measurement confirms it.  Gemma's decode ratio cannot improve from
   single-CB collapse alone — it is already there.

3. **P3b is the lever for both families, with shared root causes.**  The
   §P3a' rank-1 (`alloc_buffer` Mach-IPC chain), rank-3
   (`command_encoder()` churn — bigger on qwen35), rank-4
   (`build_rope_multi_buffers` — closed by iter1 for qwen35, no analog
   on gemma since gemma uses a different rope kernel chain) all live
   *under* both forward paths because both families use the same
   `mlx-native` substrate.  **Therefore: rank-1 alloc_buffer pool
   (iter7) is the highest-leverage shared lever.**

4. **The Wave 2b apex MoE re-measurement (hard gate #1) gates iter7 for
   the qwen35 family.  It does NOT block applying the same pool to the
   gemma path** — gemma's allocation sites are different (TQ KV encode
   + B8/B9/B10/B11 groups + sliding/global attention) and need their
   own measurement.  **Add gemma-side §P3a''.**

5. **Existing iter1 (rope_multi cache) wins are 0 µs on gemma** — gemma's
   `apply_imrope` analog is not in the qwen35 module path that iter1
   touched.  Gemma uses `forward_mlx.rs` rope dispatches; same kernel
   substrate, but the per-call alloc happens through different helpers.
   **Track B for iter5/iter6: port the rope_multi cache to gemma's
   `forward_mlx.rs` rope call sites.**

### 2026-04-27 — §P3a''' apex MoE live profile pass (iter6, Wave 2b hard gate #1 CLOSED)

**Provenance.**  Same xctrace TimeProfiler harness as §P3a' / §P3a''
(`scripts/profile-p3aprime.sh`) with `FIXTURE` overridden to the apex
35B-A3B fixture: `qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46`.
**5 cold-SoC trials × 64 decode tokens** at hf2q@`e6a531f` /
mlx-native@`19f5569` (post-iter1 — `dispatch_rope_multi_cached` now in
the build).  Aggregated via `aggregate_decode.py` with the qwen35
hypothesis register + `--decode-filter 'forward_gpu_greedy'` (the
qwen35 decode entry — same filter §P3a' used).  No thermal warning
recorded any trial.  Raw artifacts at `/tmp/adr015-apex-p3a/`.

**Closes Wave 2b hard gate #1** (the apex 35B-A3B MoE re-measurement
gate from §"Codex review acceptance — Wave 2a P3a' rev1") — every
table cell below is from the apex MoE fixture, the dense-fixture
extrapolation problem from Wave 2a is gone.  5-trial median
aggregation (Wave 2b methodology fix #5) replaces the 3-trial mean
that exposed the 21% trial-3 outlier-bias on dense.

**Per-trial decode CPU wall.**

| Trial | decode-filtered samples | CPU ms attributed | µs/token (CPU only) |
|---|---:|---:|---:|
| 1 | 104 | 104 | 1625 |
| 2 | 123 | 123 | 1922 |
| 3 | 115 | 115 | 1797 |
| 4 | 113 | 113 | 1766 |
| 5 | 108 | 108 | 1688 |

Median: **113 ms / 64 = 1766 µs/token CPU**.  Apex MoE total decode
wall ≈ 9.16 ms/token (per ADR-012 closure).  CPU share ≈ **19 %**,
GPU share ≈ **81 %**.  Substantially more CPU-bound than gemma
(5.7 %) but still GPU-dominant — both families need GPU-side work
to fully clear D4.

**Per-hypothesis live measurements (5-trial median µs/token, 64
tokens/trial, decode-filtered):**

| ID | canonical frame | static est | live (5-trial median) | per-trial µs/token | verdict |
|---|---|---:|---:|---|---|
| **H1** | `qwen35::gpu_ffn::proj::` (literal `fn proj` site) | 100 | **375** | 297 / 406 / 375 / 375 / 297 | **CONFIRMED at literal site on apex MoE.**  Subcomponent: `build_dense_ffn_layer_gpu_q` = 0 (correct — apex doesn't exercise it); `build_moe_ffn_layer_gpu_q` = 578 µs/token (37 ms median × 1000/64).  **Wave 2b hard gate #1 (a) — H1 literal site sized: ✅** |
| **H2** | `qwen35::gpu_full_attn::apply_imrope::` | 80 | **63** | 16 / 63 / 63 / 16 / 63 | **iter1 IMPACT MEASURED on apex.**  `build_rope_multi_buffers` subcomponent = 0 µs/token (cache-hit path).  Apex post-iter1 vs Wave 2a dense (464 µs/token) = **−401 µs/token / −86 %**.  iter1 confirmed effective on the apex production fixture, not just dense. |
| **H3** | `MlxDevice::command_encoder::` | 55 | **125** | 94 / 125 / 125 / 94 / 172 | **Lower than dense (724 µs/token).**  Apex's MoE wrapper opens fewer encoders/token than dense's per-stage encoder pattern.  Single-CB (P3) still removes most of this. |
| **H4** | `issue_metal_buffer_barrier` | 35 | **0** | all 0 | **Sub-1ms TimeProfiler resolution** — same status as §P3a'/§P3a''.  `BARRIER_COUNT` / `BARRIER_NS` (iter2) ARE the right tool; rerun under `MLX_PROFILE_BARRIERS=1` if needed. |
| **H5** | `anyhow::Context::with_context` | 25 | **0** | all 0 | **Confirmed FALSIFIED on apex too** — closure-built `format!` is lazy on success. |
| **H6** | `MlxDevice::alloc_buffer::` (Wave 2b new entry) | — | **344** | 297 / 359 / 281 / 375 / 344 | **Wave 2b hard gate #1 (b) — H6 sized for apex: ✅**.  Subcomponents: `IOGPUResourceCreate` = 203 µs/token; `mach_msg2_trap` = 156 µs/token; `IOConnectCallMethod` = 172 µs/token.  About **9× smaller than dense Wave 2a's 3063 µs/token (without trial 3).**  Apex MoE's call pattern is fundamentally different from dense Q4_0: MoE expert dispatch reuses fewer fresh buffers per layer.  Pool migration on apex still recovers a real ~344 µs/token; just not the dominant lever it was on dense. |

**Wave 2a static-suspect-list re-scoping (final).**

The original Wave 1 P3b suspect list (helper-function indirection,
buffer-pool acquisition, barrier bookkeeping, KV-cursor updates) has
now been measured on the actual apex MoE fixture.  The §P3a'
supersession note observed this list was falsified for **dense**;
the apex re-measurement now sizes its rows at ranked priority for
the qwen35 family:

1. **H1 literal `fn proj`**: 375 µs/token primary, 578 µs/token in
   `build_moe_ffn_layer_gpu_q` parent — the dominant lever.
2. **H6 alloc_buffer**: 344 µs/token (overlaps with H1 since H1
   inclusive contains alloc descendants — see Codex Q3 caveat).
3. **H3 command_encoder**: 125 µs/token — closes under P3.
4. **H2 apply_imrope** post-iter1: 63 µs/token — already nearly closed.

H1 + H6 + H3 sum (with overlap caveat) ≈ 844 µs/token nominal — but
Codex Q3 / AF3 says don't sum overlapping inclusive frames; the
real recovery upper bound is bounded by the **non-overlapping CPU
total of 1766 µs/token median**.  Per the §P3a''' total-CPU
constraint: **even closing all listed CPU levers to 0 saves at most
1.77 ms/token of the ~9.16 ms apex MoE decode wall (~19 % of wall)**.

**iter1 cumulative impact (apex MoE).**

H2 was 464 µs/token (Wave 2a dense static-estimate inflation × 5.8)
and is now 63 µs/token on apex.  Conservatively crediting only the
delta we directly measured (Wave 2a was a different fixture; treat
the comparison as a sanity check, not arithmetic): iter1 measurably
closed the literal H2 site on apex.  ADR-012 closure pre-iter1 gap
was 540 µs/token (1/109.1 − 1/116.0 t/s).  A fresh apex bench is
running concurrently to capture the post-iter1 ratio in numbers
rather than estimates.

**iter7 plan refined by §P3a'''.**

Per the discovery in iter6 prep that `decode_pool::pooled_alloc_buffer`
infrastructure already exists in the codebase (`MlxBufferPool` with
arena lifecycle + `reset_decode_pool` between tokens), iter7 is a
**migration**, not a build.  Target sites (`grep device.alloc_buffer`
in qwen35 with /test/load filtered out):

- `gpu_full_attn.rs:384` `apply_imrope` output buffer (32 calls/token
  on apex MoE FullAttn = 16 layers × 2 = 32; per-call ~16 KB Q + ~4 KB K).
- `forward_gpu.rs:295/302/448/651/861/868/1183/1186/1189/1194/1226`
  decode-hot output / argmax / position scratch.
- `gpu_full_attn.rs:565/1687/1779/1845` projection helpers /
  positions / argmax index — verify each is on the production
  decode hot path before migrating.

Hazards re-confirmed from §P3a' Codex Q3 still apply: CB lifetime
fence (already provided by `reset_decode_pool` at top-of-token);
byte_len / shape correctness for any CPU-read buffer (preserve
`gpu_ffn.rs:391-396` `proj()` carve-out semantics — the existing
pool already preserves exact byte_len); peak working-set caps.

### 2026-04-27 — §P3a'' gemma live profile pass (iter5)

**Provenance.** Same xctrace TimeProfiler harness as §P3a' (`scripts/profile-p3aprime.sh`) with `FIXTURE` and `OUT_DIR` overridden for gemma.  5 cold-SoC trials × 64 decode tokens on `gemma-4-26B-A4B-it-ara-abliterated-dwq` at hf2q@`675c83c` / mlx-native@`19f5569`.  No thermal warning recorded any trial.  Aggregated by `scripts/aggregate_decode.py --decode-filter 'forward_decode(?!_kernel_profile)'` (newly added decode-filter scopes aggregation to samples whose backtrace contains the gemma decode entry, mirroring the §P3a' filter on `forward_gpu_greedy`).

**Headline finding — gemma decode is GPU-BOUND, not CPU-bound.**

| trial | decode-filtered CPU samples | CPU ms attributed | µs/token (CPU only) |
|---|---:|---:|---:|
| 1 | 57 | 57 | 891 |
| 2 | 87 | 87 | 1359 |
| 3 | 30 | 30 | 469 |
| 4 | 31 | 31 | 484 |
| 5 | 42 | 42 | 656 |

Median trial CPU wall = **42 ms / 64 tokens = 656 µs/token CPU**.  Total decode wall (P0 iter4) = **11.4 ms/token** (87.8 tok/s).  CPU share = **5.7 %**, GPU share = **94.3 %**.

Implication: **even closing the entire CPU wall to 0 saves only ~6% of decode wall.**  The 19% gemma gap to llama.cpp lives almost entirely in **GPU compute**, not CPU orchestration.  This **falsifies** the pre-iter5 "shared P3b lever" hypothesis from the iter4 reshape subsection above for the gemma family.

**Per-hypothesis live measurements (decode-filtered, 5-trial median µs/token):**

| ID | canonical frame | median µs/token | per-trial |
|---|---|---:|---|
| G1 | `fused_head_norm_rope::dispatch_fused_head_norm_rope_f32` | 15.6 | 47 / 109 / 16 / 0 / 16 |
| G2 | `hadamard_quantize_kv::dispatch_hadamard_quantize_kv` | 46.9 | 62 / 78 / 47 / 47 / 16 |
| G3 | `fused_norm_add::dispatch_fused_norm_add_f32` | 0.0 | 31 / 62 / 0 / 0 / 0 |
| G4 | `argmax::dispatch_argmax_f32` | 0.0 | 0 / 0 / 0 / 16 / 0 |
| G5-Shared-AllocBuffer | `MlxDevice::alloc_buffer::` | **0.0** | all 0 |
| G6-CommandBuffer | `GraphSession\|exec.begin\|s.finish` | 218.8 | 312 / 422 / 188 / 141 / 219 |
| G7-FormatString | `anyhow::Context::with_context` | 0.0 | all 0 |
| G8-Barrier | `issue_metal_buffer_barrier` | 0.0 | all 0 (sub-1ms sample resolution; counter-based via iter2 instead) |

**Verdicts:**

- **G5 confirmed at 0 µs/token on gemma.**  `MlxDevice::alloc_buffer` does not appear in any decode-filtered sample — gemma's fused-kernel pipeline reuses caller-provided buffers and parameter buffers via `KernelArg::Bytes`.  **The qwen35 §P3a' rank-1 lever (3719 µs/token alloc_buffer Mach-IPC chain) does NOT exist on gemma.**  The iter7 alloc_buffer pool will be 0 µs win for gemma.
- **G6 GraphSession bookkeeping = 219 µs/token median.**  Largest CPU contributor.  `mlx_native::graph::GraphSession::barrier_between` appears at rank 10-14 in every trial's top-15.  Single-CB collapse can shave some of this but bound by the 656 µs/token CPU wall total.
- **G2 Hadamard KV-encode = 47 µs/token.**  TQ-KV path fires per layer; gemma-only relative to qwen35.  Small contributor.
- **G1 fused head-norm+RoPE = 16 µs/token.**  Already-fused, alloc-free, well-tuned.  Confirmed not a lever.
- **G8 barrier sub-1ms.**  Same as §P3a' H4 verdict (counter resolution required, not TimeProfiler).  iter2 BARRIER_COUNT/BARRIER_NS atomics are the right tool; rerun under `MLX_PROFILE_BARRIERS=1` if needed.

**Rank stability — top-N frames in EVERY trial's top-15 (n_trials=5):**

```
rank 1   : forward_decode (the entry — every trial)
rank 2   : cmd_generate
rank 3   : run
rank 4   : main
rank 10-14: GraphSession::barrier_between
rank 11-13: AGX::ComputeContext encode internals
            (insertIndirectTGOptKernel, performEnqueueKernel)
            (-[AGXG17XFamilyComputeContext memoryBarrierWithScope:])
rank 12-16: -[AGXG17XFamilyComputeContext dispatchThreadgroups:...]
```

The CPU work that IS sampled is: dispatching threadgroups + barriers + encoder bookkeeping.  Same shape as qwen35.  Different scale (~300 µs/token total, vs qwen35's ~3700 µs/token alloc_buffer alone).

**Coverage caveat — TimeProfiler sees CPU only.**  The 94.3% GPU share lives outside this trace.  Wave 2c+ work for gemma needs **Metal System Trace** (Instruments → Metal System Trace template → GPU counters) to attribute the 10.7 ms/token of GPU work.  Q-NAX-1 / Q-NAX-4 from the §"M5 Neural Accelerators" section call this out — Metal System Trace is the right tool, not TimeProfiler.

**Wave 2c hard gates** (gemma-side levers; supersedes the iter4 "P3b is the lever for both families" claim for gemma specifically):

1. **Metal System Trace** of gemma decode: 5 cold trials × 64 tokens on the same fixture, capture per-kernel GPU time + GPU-counter occupancy (`gpu.counters.shaderUtilization`, `gpu.counters.alu`, `gpu.counters.l2HitRate`, and if exposed `gpu.counters.neuralAcceleratorUtil`).  Establishes the GPU-side ranked residual.
2. **Kernel-by-kernel comparison vs llama.cpp** on the same model.  llama.cpp's `ggml-metal` kernels for Q4_K mat-vec (and gemma's TQ-KV-equivalent — `ggml-metal-quants/q4_K_mv.metal` etc.) are the ground-truth reference for this hardware.  Attribute the 1820 µs/token gap by tensor-format kernel.
3. **Decision point:** if gemma's kernel choice / dispatch pattern is the gap, ADR-015 P3c (M5 Neural Accelerator decode kernels) becomes a hard prerequisite for D4 second bullet — currently P3c is scoped TTFT-only.  Re-scope P3c to include decode mat-vec NAX routing if the data supports it.

**Implication for iter6 / iter7 / D4:**

- iter6 (gemma single-CB unification) — refactor-for-uniformity ROI on perf is bounded by ~219 µs/token (G6).  Worth doing for code consistency, not for the 19% gap.
- iter7 (P3b alloc_buffer pool) — applies to qwen35 ONLY for gemma's purposes; the dense qwen35-Q analog from §P3a' (3063 µs/token without trial-3 outlier) is real, gemma's analog is 0.
- **iter6 effort is shifted to Wave 2c gemma GPU profiling** (Metal System Trace + kernel comparison) so D4 second bullet is closable on evidence rather than hope.



Original P3 (qwen35 single-CB rewrite) recovers ~30% of the gap.
**A new P3b (Rust-orchestration profiling sweep) is added** that targets
the ~53% residual.  P3 and P3b can run in parallel; the P5 bench gate
requires both wins to clear ≥1.00× of llama.cpp.

## Decision

**D1.** Migrate **both** `qwen35` and `gemma` decode forward paths to a
**single `GraphSession`** (1 CB per forward pass), mirroring llama.cpp's
`encode_async` (`ggml-metal-context.m:676–722`) — one logical command
buffer encoded with explicit `encoder.memory_barrier()` between dependent
op stages.  Order: `qwen35` first (proves the pattern on the harder MoE
+ DeltaNet shape), `gemma` second (port the established pattern to the
dense-only path).

**D2.** Keep the existing per-layer-encoder paths as the legacy
implementation behind a single env var `HF2Q_LEGACY_PER_LAYER_CB=1` (covers
both families).  After a 7-day soak + sourdough-pass on each family's new
path, delete the legacy path for that family entirely (per
`feedback_no_broken_windows`).

**D3.** Bit-exact parity gate vs the current decode entry points is
mandatory **per family** (within 1e-5 max-abs logit error on each
family's smoke set).  No fallback path is shipped — if parity fails, the
lever is wrong; fix it, don't gate the primary path off (per
`feedback_never_ship_fallback_without_rootcause`).

**D4.** Exit criteria — **both** families must clear the bar:
  - **qwen35:** `dwq46` `n=256` decode median **≥ 1.00× of llama.cpp**
    same-day median (3 cold runs each, M5 Max, cold SoC per
    `feedback_perf_gate_thermal_methodology`).  Reference baseline from
    ADR-012 closure: hf2q 109.1 t/s vs llama.cpp 116.0 t/s = **0.94×**.
    **§iter6 measured 2026-04-27 (apex post-iter1):** hf2q 110.0 t/s
    (per-trial 110.0 / 109.3 / 110.2; σ ≈ 0.4) vs llama-bench 118.71 t/s
    (per-trial 118.71 / 117.90 / 119.20) = **0.9266×**.  hf2q absolute
    moved +0.9 t/s (iter1 contributed positively); llama-bench drifted
    +2.7 t/s same-day per `project_end_gate_reality_check`.  Recovery
    required: **+7.9%** (multiply by 1.0794 → 1.000); decode-wall gap
    today = **667 µs/token** (1/110.0 − 1/118.71).  Captured at
    hf2q@`4214f7c` / mlx-native@`19f5569` via `scripts/bench-baseline.sh`;
    raw artifacts at
    `/tmp/adr015-bench/baseline-apex-35b-a3b-dwq46-post-iter1-20260427T123407Z.*`.
  - **gemma:** `gemma-4-26B-A4B-it-ara-abliterated-dwq` `n=256` decode
    median **≥ 1.00× of llama.cpp** same-day median (same methodology).
    **§P0 measured 2026-04-27 (iter4):** hf2q 87.8 t/s (per-trial 87.7 / 87.8 / 88.0;
    extremely tight, σ ≈ 0.15) vs llama-bench 104.52 t/s
    (per-trial 103.96 / 104.58 / 104.52) = **0.840×**.  Recovery
    required: **+19.0%** (multiply by 1.190 → 1.000).  Captured at
    hf2q@`35fdcc8` / mlx-native@`19f5569` on M5 Max macOS 26.4.1
    via `scripts/bench-baseline.sh`; raw artifacts at
    `/tmp/adr015-bench/baseline-gemma-26B-dwq-p0-20260427T045841Z.*`.

## Phasing

| Phase | Deliverable | Definition of done |
|---|---|---|
| **P0 — Gemma baseline** ✅ **Done 2026-04-27 (iter4)** | Same-day `gemma-4-26B-A4B-it-ara-abliterated-dwq` `n=256` decode median: 3 cold runs of `hf2q` + 3 of `llama-bench` via `scripts/bench-baseline.sh`. Result: hf2q 87.8 t/s vs llama 104.52 t/s = **0.840×**.  Recovery required: **+19.0%**. | hf2q 87.8 (87.7 / 87.8 / 88.0) vs llama-bench 104.52 (103.96 / 104.58 / 104.52); ratio recorded in D4 above. |
| **P1 — Audit** | Map every cross-CB synchronization point in `forward_gpu` (qwen35) and `forward_decode` (Gemma) paths. List of `(commit/commit_and_wait → next encoder)` transitions and the buffer dependencies that justify each one. Output: a markdown table per family in this ADR. | Every CB boundary in each family's live path is annotated with the data dependency it protects, OR documented as legacy / removable. |
| **P2 — Empty-CB cost calibration** | ✅ **Done 2026-04-26** — `mlx-native/examples/cb_cost_calibration.rs` measures async/sync/alloc-only µs/CB on M5 Max.  Result: **async µs/CB ≈ 1.6 µs at N≥100, NOT ~5 µs**.  Single-CB upper bound = ~160 µs / token of the ~500 µs gap (~32%). | Numbers published in §Diagnosis update above. |
| **P3a — Per-dispatch cost calibration** | ✅ **Done 2026-04-26** — `mlx-native/examples/dispatch_cost_calibration.rs` measures `scalar_mul_bf16` encode cost.  Result: **µs/dispatch ≈ 0.16 µs** at N≥500, within ±15% of llama.cpp's implied 0.14 µs/dispatch.  Shader-launch path is competitive; lever is Rust orchestration. | Numbers published in §P3a calibration data above. |
| **P3 — Single-CB rewrite** (qwen35 first, gemma second) | New `forward_gpu_single_cb` in `inference/models/qwen35/forward_gpu.rs` and `forward_decode_single_cb` in `serve/forward_mlx.rs`.  Each: one `GraphSession`, all dispatches encoded, explicit `memory_barrier()` for cross-stage dependencies, single `commit()` at end. | Builds in release.  No new `unsafe`.  Recovers ~30% of the 540 µs gap (~161 µs). |
| **P3b — Rust orchestration sweep** ⬅ **NEW (parallel to P3)** | Profile-guided reduction of the ~288 µs residual from helper-function indirection, buffer pool acquisition, barrier bookkeeping, KV-cursor updates.  Use `cargo flamegraph` or `instruments -t TimeProfiler` on a hot decode loop.  Target: shrink the residual by ≥50% (~140 µs). | At least 5 profile-driven specific reductions landed (each cited with profile evidence in the commit message).  Combined with P3, restores ≥0.99× ratio (one cold-run iteration). |
| **P5 — Parity gate (per family)** | 16-prompt smoke on each family's new path produces logits within 1e-5 max-abs of legacy path. | Logged in commit message. |
| **P6 — Bench gate (per family)** | 3 cold runs of `hf2q --benchmark` on `dwq46` and `gemma-4-26B-A4B-it-ara-abliterated-dwq` at `n=256` (cold SoC).  Same-day `llama-bench -p 0 -n 256 -r 3` on the same model.  Ratio computed per family. | Ratio ≥ 1.00× recorded for **both** families in ADR + commit message. |
| **P7 — Default cut-over** | `HF2Q_LEGACY_PER_LAYER_CB=1` is the only way back to legacy.  Default = single-CB on both families. | Smoke + bench gate green on default-default settings. |
| **P8 — Soak + delete legacy** | 7-day window, no regressions.  Then delete `forward_gpu_greedy` (qwen35 legacy) and the per-layer-GraphSession Gemma legacy entirely. | Diff-stat ≥ 1000 LOC removed across both families. |

### P1 audit (merged) — qwen35 decode CB boundaries

**Provenance.** Wave 1 of CFA session `cfa-20260426-adr015-singlecb` ran two
parallel P1 audits (Claude variant `16f5bbf`, Codex variant `816d247`).  Both
converged on the same headline: **103 CBs / decode token** on the
`qwen3.5-MoE-35B-A3B dwq46` greedy hot path, **102 fuse_safe=YES** under the
P3.qwen35-single-cb rewrite, **1 fuse_safe=NO** (terminal `commit_and_wait`
on argmax), **0 ambiguous in production**.  This merged subsection adopts
Codex's clean per-file table layout (no in-band corrections, correct
test-pair labelling, correct excluded-row totals) and Claude's tighter
fuse_safe taxonomy + production-vs-non-production-path bookkeeping, with
all file:line citations live-verified by the tiebreaker against
hf2q@`0b29397`.

**Methodology.** `grep -nc "device.command_encoder\|enc.commit"` against the
four qwen35 source files at hf2q@`0b29397` (live-verified by tiebreaker, not
copied from either team):

| File | grep -nc count |
|---|---:|
| `src/inference/models/qwen35/forward_gpu.rs` | 20 |
| `src/inference/models/qwen35/gpu_full_attn.rs` | 20 |
| `src/inference/models/qwen35/gpu_ffn.rs` | 25 |
| `src/inference/models/qwen35/gpu_delta_net.rs` | 13 |

The narrow grep on the merged worktree HEAD yields 78 sites (20+20+25+13)
which is below the spec floor; that is because the spec floor figures
(`forward_gpu.rs ≥22`, `gpu_full_attn.rs ≥27`, `gpu_ffn.rs ≥25`,
`gpu_delta_net.rs ≥15`) used a wider regex that also catches
`encoder.commit`, `enc_norm.commit`, `enc_lm.commit`, `enc_argmax.commit`,
`enc2.commit`.  The wider-regex live count gives 23/26/25/13 = 87 sites.
The Codex Wave 1 variant cited 39/34/44/23 — those numbers are **not
reproducible against hf2q@`0b29397`** with any grep this tiebreaker
attempted, and the table totals below cite specific file:line pairs rather
than the disputed bulk count.  The fuse-safe verdict is unaffected by the
methodology disagreement.

**fuse_safe taxonomy** (decode hot path, seq_len=1):

- **YES** — the commit only exists because the helper happens to commit;
  producer/consumer are both in-encoder-friendly and a single
  `enc.memory_barrier()` in single-CB form is sufficient.
- **NO — `<reason>`** — the commit is required: a CPU read
  (`download_f32`, `as_slice::<…>()`) follows immediately, OR the commit
  terminates a function whose contract is "buffer is GPU-resident and
  ready for caller's next encoder" (caller-handoff; collapsable in P3 by
  passing `&mut Encoder`).
- **AMBIGUOUS — `<spike>`** — fuse-safety hinges on something not
  statically determinable from the source.

In single-CB form every "NO — caller-handoff" row collapses to YES once
the caller passes its `&mut Encoder` down (P3.qwen35-single-cb refactor).
"NO — CPU read" rows remain genuine commits.  "NO — prefill" rows are
out of ADR-015 scope (Non-Goal 1) and stay legacy.

#### `forward_gpu.rs`

| File:Line | Encoder name | Commit kind | Data dep protected | fuse_safe |
|---|---|---|---|---|
| forward_gpu.rs:309 / :325 | `enc` (non-greedy output norm) | `commit()` async | `dispatch_rms_norm` writes `normed`; lm_head encoder at :332 reads it. | YES — barrier between rms_norm and lm_head. |
| forward_gpu.rs:332 / :344 | `enc` (non-greedy lm_head) | `commit_and_wait()` | lm_head writes `logits_buf`; `download_f32(&logits_buf)` reads full-vocab logits. | NO — terminal CPU read in non-greedy path. |
| forward_gpu.rs:393 / :399 | `enc_norm` (greedy output norm) | `commit_labeled` async | output norm writes `bufs.norm_out_buf`; lm_head encoder at :403 reads it. | YES — barrier before lm_head. **Hot path: 1 CB/token.** |
| forward_gpu.rs:403 / :409 | `enc_lm` (greedy lm_head_q4) | `commit_labeled` async | Q4 lm_head writes `logits_buf`; argmax encoder at :412 reads it. | YES — barrier before argmax. **Hot path: 1 CB/token.** |
| forward_gpu.rs:412 / :417 | `enc_argmax` (greedy argmax) | `commit_and_wait_labeled` | argmax writes `out_index`; `out_index.as_slice::<u32>()` reads the 4-byte token id on CPU. | NO — terminal 4-byte host read; the only mandatory wait in the new single-CB path. **Hot path: 1 CB/token.** |
| forward_gpu.rs:450 / :462 | `enc` (residual_add helper) | `commit_and_wait()` | `elementwise_add` writes `out`; caller uses GPU buffer in next encoder. | NO — caller-handoff (helper contract returns ready buffer). Becomes YES under P3 (helper takes `&mut Encoder`). 0 CBs/token on production MoeQ greedy path. |
| forward_gpu.rs:599 / :611 | `enc` (lm_head BF16 cast init, non-greedy) | `commit_and_wait()` | cast writes `lm_head_bf16`; cache stores it for later tokens. | NO — cold model-cache init, **fires once at load**. 0 CBs/decode-token. |
| forward_gpu.rs:874 / :894 | `enc` (legacy fused_residual_norm, non-greedy + non-MoeQ) | `commit()` async | fused residual+norm writes `ffn_input`/`ffn_residual`; FFN helper reads them. | YES — barrier before FFN helper. Used only by `forward_gpu_impl` non-greedy + F32-MoE / capture paths; 0 CBs/decode-token on production. |
| forward_gpu.rs:1120 / :1132 | `enc` (greedy lm_head BF16 cast init) | `commit_and_wait()` | cast writes `lm_head_bf16`; greedy cache stores it. | NO — cold model-cache init, **fires once at load**. 0 CBs/decode-token. |
| forward_gpu.rs:1422 / :1445 (decode) / :1447 (prefill) | `enc` (greedy fused res_norm + MoeQ FFN) | `commit_labeled` (decode) / `commit_and_wait_labeled` (prefill) | fused residual+norm writes `ffn_input`/`ffn_residual`; in-encoder call to `build_moe_ffn_layer_gpu_q_into` reads them and writes `ffn_out`; consumer is the next layer's encoder. | YES — already 1 CB/MoE-layer; under single-CB it becomes an internal barrier. **Hot path: 40 CB/token on dwq46 today (40 transformer layers, MoeQ in every layer).** |
| forward_gpu.rs:1454 / :1470 | `enc` (legacy fused_res_norm before non-MoeQ FFN, **production code** at line ≤ 1660 inside `impl Qwen35Model`) | `commit()` async | fused residual+norm writes FFN inputs; Dense / DenseQ / F32-MoE FFN helper reads them in its own encoder. | YES — barrier before FFN helper.  Conditional fallback path; 0 CB/token on production MoeQ (`dwq46`). |
| forward_gpu.rs:1451+:1463, :1527+:1540, :1587+:1600, :1689+:1704, :1751+:1756, :1968+:1975 | (six `[TEST]` encoder/commit pairs inside `#[cfg(test)] mod tests` opening at line 1662) | `[TEST]` | N/A — test-only. | N/A — `[TEST]`; excluded from per-decode-token totals. |

`forward_gpu.rs` decode-hot-path subtotal on `dwq46` (greedy, all 40 layers
MoeQ): **3 (output head) + 40 (per-layer MoE fused encoder) = 43 CBs/token.**

#### `gpu_full_attn.rs`

| File:Line | Encoder name | Commit kind | Data dep protected | fuse_safe |
|---|---|---|---|---|
| gpu_full_attn.rs:852 | `encoder` (caller-supplied, in `apply_sdpa_causal_from_seq_major`) | `commit_and_wait()` | caller's prior dispatches wrote `q_seq_major`/`k_seq_major`/`v_seq_major`; CPU `download_f32` at :855..:857 for head-major permute. | NO — CPU read. **Stateless / fallback path** (only reached when `kv_cache_slot` is `None`); 0 CBs/decode-token in production. |
| gpu_full_attn.rs:869 / :874 | `enc2` (stateless SDPA after CPU permute) | `commit_and_wait()` | `apply_sdpa_causal` writes `out_hm`; `download_f32(&out_hm)` reads for permute-back. | NO — CPU read. Same stateless / fallback path; 0 CBs/decode-token. |
| gpu_full_attn.rs:959 / :983 | `enc` (kv-cache + SDPA decode) | `commit_labeled` async | KV copy writes `slot.k`/`slot.v`; intra-encoder barrier separates it from `dispatch_sdpa_decode` writing `out_buf`; consumer is ops6-7 encoder at :1211. | YES — internal RAW barrier already present; inter-CB handoff to ops6-7 collapses under single-CB. **Hot path: 1 CB/FullAttn-layer.** |
| gpu_full_attn.rs:1022 / :1025 | `enc` (kv-cache SDPA prefill) | `commit_and_wait()` | SDPA writes `out_buf`; CPU permute-back via `download_f32`. | NO — prefill CPU read; out of ADR-015 decode scope. 0 CBs/decode-token. |
| gpu_full_attn.rs:1115 / :1180 (decode) / :1182 (prefill) | `enc` (ops1-4) | `commit_labeled` (decode) / `commit_and_wait()` (prefill) | pre-attn norm, Q/K/V/G projections, Q/K norms, IMROPE produce `q_rope`/`k_rope`/`v_flat`; SDPA reads them. | YES (decode) — intra-encoder barriers already separate stages; handoff to SDPA collapses. **Hot path: 1 CB/FullAttn-layer.** Prefill: NO. |
| gpu_full_attn.rs:1198 | `enc` (op5 stateless SDPA branch caller encoder) | (no own commit; `apply_sdpa_causal_from_seq_major` commits at :852) | producer: stateless SDPA helper returns freshly-uploaded buffer; consumer: ops6-7 at :1211. | NO — only reachable when `kv_cache_slot` is `None` (parity tests / synthetic); 0 CBs/decode-token in production. |
| gpu_full_attn.rs:1211 / :1229 (decode) / :1231 (prefill) | `enc` (ops6-7: sigmoid_gate_multiply + output projection) | `commit_labeled` (decode) / `commit_and_wait()` (prefill) | gate multiply writes `gated`; output projection writes `out`; consumer is caller's `dispatch_fused_residual_norm_f32` (forward_gpu.rs:874/:1422/:1454). | YES (decode) — barrier before next layer's fused res-norm. **Hot path: 1 CB/FullAttn-layer.** |
| gpu_full_attn.rs:1451+:1463, :1527+:1540, :1587+:1600, :1689+:1704, :1751+:1756, :1968+:1975 | (six `[TEST]` encoder/commit pairs inside `#[cfg(test)] mod tests` at line 1244) | `[TEST]` | Test-only GPU work before assertion reads. | N/A — `[TEST]`; excluded. |

`gpu_full_attn.rs` decode-hot-path subtotal per FullAttn layer: **3 CBs**
(ops1-4 + sdpa_kv + ops6-7).

#### `gpu_ffn.rs`

| File:Line | Encoder name | Commit kind | Data dep protected | fuse_safe |
|---|---|---|---|---|
| gpu_ffn.rs:499 / :541 (decode) / :543 (prefill) | `enc` (dense SwiGLU) | `commit()` (decode) / `commit_and_wait()` (prefill) | gate / up / silu / down + optional residual, with intra-encoder barriers; caller reads `result` in next layer. | YES (decode) — single-CB-friendly. 0 CBs/decode-token on `dwq46` MoE production. |
| gpu_ffn.rs:623 / :665 (decode) / :667 (prefill) | `enc` (DenseQ SwiGLU via `quantized_matmul_ggml`) | `commit()` (decode) / `commit_and_wait()` (prefill) | quantized gate / up / down + optional residual; intra-encoder barriers. | YES (decode) — single-CB-friendly. **Hot path on Dense-Q (e.g. 27B `dwq46`): 1 CB/dense-layer.** 0 CBs/decode-token on 35B-A3B MoeQ. |
| gpu_ffn.rs:811 / :813 | `enc` (F32-MoE router) | `commit_and_wait()` | router writes `logits_buf`; `download_f32(&logits_buf)` for CPU softmax+top-k. | NO — CPU read; legacy F32-MoE only. 0 CBs/decode-token on MoeQ production. |
| gpu_ffn.rs:848 / :850 | `enc` (F32-MoE per-expert gate proj) | `commit_and_wait()` | expert gate writes `gate_e_buf`; `download_f32` for CPU SiLU multiply. | NO — CPU read; legacy F32-MoE only. 0 CBs/decode-token. |
| gpu_ffn.rs:852 / :854 | `enc` (F32-MoE per-expert up proj) | `commit_and_wait()` | expert up writes `up_e_buf`; `download_f32` for CPU SiLU multiply. | NO — CPU read; legacy F32-MoE only. 0 CBs/decode-token. |
| gpu_ffn.rs:867 / :869 | `enc` (F32-MoE per-expert down proj) | `commit_and_wait()` | expert down writes `y_e_buf`; `download_f32` for CPU weighted accumulation. | NO — CPU read; legacy F32-MoE only. 0 CBs/decode-token. |
| gpu_ffn.rs:889 / :891 | `enc` (F32-MoE shared_gate_inp) | `commit_and_wait()` | shared gate scalar projection writes `sh_logit_buf`; `download_f32` for CPU sigmoid. | NO — CPU read; legacy F32-MoE only. 0 CBs/decode-token. |
| gpu_ffn.rs:901 / :903 | `enc` (F32-MoE shared gate proj) | `commit_and_wait()` | shared gate writes `a_s_buf`; `download_f32` for CPU silu_mul. | NO — CPU read; legacy F32-MoE only. 0 CBs/decode-token. |
| gpu_ffn.rs:906 / :908 | `enc` (F32-MoE shared up proj) | `commit_and_wait()` | shared up writes `b_s_buf`; `download_f32` for CPU silu_mul. | NO — CPU read; legacy F32-MoE only. 0 CBs/decode-token. |
| gpu_ffn.rs:917 / :919 | `enc` (F32-MoE shared down proj) | `commit_and_wait()` | shared down writes `y_s_buf`; `download_f32` for final CPU combine. | NO — CPU read; legacy F32-MoE only. 0 CBs/decode-token. |
| gpu_ffn.rs:997 / :1001 (decode) / :1003 (prefill) | `enc` (MoeQ wrapper around `build_moe_ffn_layer_gpu_q_into`) | `commit()` (decode) / `commit_and_wait()` (prefill) | wraps the `_into` variant; production `forward_gpu_greedy` calls `_into` directly via `forward_gpu.rs:1422`, so this wrapper is **not** on the hot path. | YES (decode) — wrapper is single-CB-compatible but unused by production; 0 CBs/decode-token on `dwq46` greedy. |

`gpu_ffn.rs` decode-hot-path subtotal on `dwq46` MoE production:
**0 CBs** (all MoE FFN encoding owned by `forward_gpu.rs:1422`).
On 27B Dense-Q production: 1 CB/dense-layer.

#### `gpu_delta_net.rs`

| File:Line | Encoder name | Commit kind | Data dep protected | fuse_safe |
|---|---|---|---|---|
| gpu_delta_net.rs:486 / :500 | `enc` (standalone `apply_ssm_conv`) | `commit_and_wait()` | ssm conv writes `new_state_buf` + `y`; `extract_new_conv_state` calls `as_slice::<f32>()` (CPU). | NO — CPU read. **Standalone helper bypassed by production decode**: `build_delta_net_layer` inlines its own ssm_conv at :911 / :994. 0 CBs/decode-token in production. |
| gpu_delta_net.rs:659 / :676 | `enc` (standalone `apply_gated_delta_net`) | `commit_and_wait()` | GDN writes `output_buf` + `state_out_buf`; helper returns GPU buffers. | NO — caller-handoff. **Standalone helper bypassed by production decode**: `build_delta_net_layer` inlines `dispatch_gated_delta_net` at :947. 0 CBs/decode-token in production. |
| gpu_delta_net.rs:893 / :970 | `enc` (`build_delta_net_layer` ops1-9 decode) | `commit_labeled` async | Internal barriers at :899/:909/:916/:932/:945/:953/:961 separate op1 → ops2a/b/c → op3 (ssm_conv) → ops5/6 → q_scale+g_beta → op7 (GDN) → op8 → op9.  Consumer of `output`: caller's next-layer fused_res_norm. | YES — internal barrier graph is already correct; only the inter-encoder handoff to next layer is a CB. **Hot path: 1 CB/DeltaNet-layer.** |
| gpu_delta_net.rs:979 / :999 | `enc` (`build_delta_net_layer` ops1-3 prefill) | `commit_and_wait()` | ssm_conv writes `qkv_conv`; `download_f32(&qkv_conv_out)` for CPU de-interleave per-token Q/K/V. | NO — prefill CPU read; out of decode scope. 0 CBs/decode-token. |
| gpu_delta_net.rs:1022 / :1069 | `enc` (`build_delta_net_layer` ops5-9 prefill) | `commit_and_wait()` | ops5-9 write `output`; downstream may dump or download. | NO — prefill-only safety belt; out of decode scope. 0 CBs/decode-token. |
| gpu_delta_net.rs:1412+:1413 | (test-only flush helper) | `[TEST]` | Inside `gpu_state_propagation_chunked_vs_monolithic` (mod tests). | N/A — `[TEST]`; excluded. |

`gpu_delta_net.rs` decode-hot-path subtotal per DeltaNet layer:
**1 CB** (`build_delta_net_layer` ops1-9).

#### Aggregate — total CBs/decode-token by layer-type

Production model: `qwen3.5-MoE-35B-A3B dwq46` (the ADR-012 closure
target).  Per `src/inference/models/qwen35/full_attn.rs:646`,
`num_hidden_layers = 40` — **40 transformer layers**, hybrid 3:1
(DeltaNet:FullAttn).  Approximate split: **30 DeltaNet layers + 10
FullAttn layers; FFN is MoeQ on every layer**.

| Layer-type | Per-layer CBs | # layers in `dwq46` | CBs/decode-token | fuse_safe=YES | fuse_safe=NO | fuse_safe=AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|
| **FullAttn** (`gpu_full_attn.rs:1115/:1180` ops1-4, `:959/:983` sdpa_kv, `:1211/:1229` ops6-7) | 3 | 10 | **30** | 30 | 0 | 0 |
| **DeltaNet** (`gpu_delta_net.rs:893/:970` ops1-9) | 1 | 30 | **30** | 30 | 0 | 0 |
| **MoE FFN** (`forward_gpu.rs:1422/:1445` fused res_norm + MoeQ FFN via `build_moe_ffn_layer_gpu_q_into`) | 1 | 40 | **40** | 40 | 0 | 0 |
| **Output head** (`forward_gpu.rs:393/:399` norm, `:403/:409` lm_head_q4, `:412/:417` argmax) | 3 | 1 | **3** | 2 | 1 (terminal CPU read) | 0 |
| **TOTAL** | | | **103** | **102 (99.0%)** | **1 (1.0%)** | **0 (0.0%)** |

**103 CBs/decode-token** matches the §Diagnosis-update budget table
(102, within rounding tolerance).  Of those, **102 are fuse_safe=YES**
under the P3.qwen35-single-cb rewrite (collapse to 1 CB / forward, with
explicit `enc.memory_barrier()` between cross-stage producer/consumer
pairs) and **1 is fuse_safe=NO** (the terminal `commit_and_wait_labeled`
on argmax that drains the GPU before the 4-byte `out_index` host read).
In single-CB form the forward becomes **1 CB/decode-token** — matching
llama.cpp's `n_cb=1` optimum.

**Excluded from per-decode-token totals** (per spec acceptance: every
boundary listed above is in exactly one row; the lines below catalog
which non-production rows are excluded from the 103 total):

- **12 `[TEST]` encoder/commit pairs** (6 in `gpu_full_attn.rs` at
  :1451-:1975, 6 in `forward_gpu.rs` at :1451-:1968, 1 in
  `gpu_delta_net.rs` at :1412); all inside `#[cfg(test)] mod tests`
  (qwen35 `forward_gpu.rs` opens `mod tests` at line 1662; full_attn at
  1244; delta_net at 1083).
- **2 model-load-only sites** (`forward_gpu.rs:599`, `:1120` —
  lm_head BF16 cast init, fires once at load).
- **15 prefill-only / F32-MoE-legacy / standalone-helper sites**
  (gpu_ffn.rs `:811`, `:848`, `:852`, `:867`, `:889`, `:901`, `:906`,
  `:917` = 8 F32-MoE; gpu_full_attn.rs `:852`, `:869`, `:1022` = 3
  prefill / stateless-fallback; gpu_delta_net.rs `:486`, `:659`, `:979`,
  `:1022` = 4 standalone / prefill).  Not on the `dwq46` greedy decode
  hot path; ADR-015 Non-Goal 1 explicitly excludes prefill from the
  rewrite.

**Top fuse-safety call-outs (for P3 implementation):**

1. The 7 production fuse_safe=YES helper sites (rows 3, 4, 9 in
   `forward_gpu.rs`; the 3 hot-path rows in `gpu_full_attn.rs`; the 1
   hot-path row in `gpu_delta_net.rs`) are the explicit list of
   cross-stage `enc.memory_barrier()` insertions that
   P3.qwen35-single-cb must place when collapsing to one encoder per
   forward.  Each barrier's data-dep is named in the "Data dep protected"
   column above.
2. The 1 fuse_safe=NO production row (`forward_gpu.rs:412` argmax →
   :421 host read of `out_index`) is the **only** mandatory
   `enc.commit_and_wait()` in the new single-CB path.  Move it to the
   very end of the forward (after all 40 layers + output norm + lm_head
   + argmax dispatched into one encoder).
3. **AMBIGUOUS rows: none in production hot path.**  Any ambiguity
   surfaces at parity-test time (R1 risk) and is handled as a real bug
   per `feedback_never_ship_fallback_without_rootcause` — never gate the
   primary path off as a fallback.

### P1 audit (merged) — gemma decode CB / GraphSession boundaries

**Provenance.** Wave 1 of CFA session `cfa-20260426-adr015-singlecb` ran
two parallel P1 audits.  Both **independently refuted** the prior
§Diagnosis row claiming "hf2q Gemma ~30 CBs / pass": the production
hot path is **1 GraphSession / decode token** (`HF2Q_DUAL_BUFFER=0`) or
**2 GraphSessions / decode token** (`HF2Q_DUAL_BUFFER` default split
after layer 3).  This merged subsection adopts Claude's comprehensive
28-site enumeration (production + 22 diagnostic-env-gated rows + the
`HF2Q_DUAL_BUFFER` ambiguous row + `HF2Q_SPLIT_TIMING` row) plus
Codex's clean fuse-safe taxonomy where cleaner.  All file:line citations
live-verified by the tiebreaker against hf2q@`0b29397`.

**Audit scope.** `src/serve/forward_mlx.rs::forward_decode` (line 1320 —
the production gemma decode entry point); **excludes**
`forward_decode_kernel_profile` (line 3631) which is gated by
`HF2Q_MLX_KERNEL_PROFILE=1`, documented as *"intentionally slow (many
sessions = many sync points)"*, and is a kprofile-only diagnostic
(its ~242 CBs/token are by-design and would not be touched by a P3
single-CB rewrite of the production path).

**Headline finding — the line:1309 doc comment is STALE.**  Quoted
verbatim from `src/serve/forward_mlx.rs:1309`:

> *"MVP: one GraphSession per layer (30 sessions per forward pass)."*

Refuted by the in-function contract comment at `forward_mlx.rs:1488`:

> *"SINGLE SESSION: Embedding + All 30 Layers + Head — ONE begin() →
> all GPU dispatches → ONE finish().  Zero CPU readbacks.  All norms,
> adds, MoE routing, scalar multiplies, softcap, and argmax run on GPU."*

The single CB is opened at `forward_mlx.rs:1591` (`exec.begin()` —
`exec.begin_recorded()` instead at :1589 when `HF2Q_MLX_GRAPH_OPT=1`) and
closed exactly once at the terminal `s.finish_optimized_with_timing` at
`:3323` (recorded path) or `s.finish_with_timing` at `:3333` (direct
path).  All other `s = exec.begin()` / `s.finish()` pairs in the body are
gated by **diagnostic / dump env flags that are off in production**
(plus one experimental dual-CB split that is also off by default).
Per `feedback_code_is_truth`, the ADR-015 §Diagnosis row is corrected
above; the gemma half of D1 is largely refactor-for-uniformity, not a
perf lever.

**Encoder semantics confirmed against `mlx-native@adf7ee0`** (read-only
per ADR-014 fence; from `/opt/mlx-native/src/graph.rs`):

- `exec.begin()` / `exec.begin_recorded()` — opens a new
  `GraphSession` wrapping a fresh `CommandEncoder` (one Metal
  `MTLCommandBuffer`).
- `s.barrier_between(reads, writes)` — **intra-CB** RAW/WAR
  conflict-gated `enc.memory_barrier()`; never crosses a CB boundary.
  *Not* a CB boundary in the ADR-015 sense.
- `s.finish()` — **CB boundary, sync** (`commit_and_wait`); consumes
  the session.
- `s.commit()` — **CB boundary, async** (`commit` only); consumes
  the session and returns the inner `CommandEncoder`.

#### Production hot path (env-defaults except `HF2Q_DUAL_BUFFER`)

| File:Line | Encoder/GraphSession name | Commit kind | Data dep protected | fuse_safe |
|---|---|---|---|---|
| forward_mlx.rs:1589 / :1591 | `s = exec.begin_recorded()` (HF2Q_MLX_GRAPH_OPT=1) or `s = exec.begin()` (default) | OPEN (single CB / forward) | none — entry point. Producer chain begins with embedding write to `activations.hidden`; all layer/head consumers are encoded into the same `GraphSession` until the (off-by-default) dual-buffer split or terminal finish. | YES — pure session open, no producer→consumer crosses an existing CB. Baseline of single-CB pattern. |
| forward_mlx.rs:3193 / :3194 | `s.commit()` async then `exec.begin()` (HF2Q_DUAL_BUFFER default split, gated `dual_buffer_split == Some(layer_idx + 1)` at :3191) | async close + re-open | layer-N end-of-layer write to `activations.hidden`; layer-(N+1) pre-attention norm reads `activations.hidden` in the next CB.  Metal queue ordering protects the RAW dependency while overlapping GPU execution of CB0 with CPU encoding of CB1. | AMBIGUOUS-NEEDS-SPIKE — correctness-safe to collapse into one encoder with a barrier, but this is an **intentional async-overlap optimization**.  The in-source comment at `forward_mlx.rs:3186-3190` documents *"sequential wait BEFORE encode: −5.6 tok/s; threaded wait DURING encode: −43 tok/s"* but did *not* measure the no-wait async-overlap variant against pure single-CB.  P3 must benchmark default-split vs pure 1-CB. |
| forward_mlx.rs:3323-3324 (optimized) / :3333-3334 (direct) | terminal `s.finish_optimized_with_timing` / `s.finish_with_timing` | sync close (`commit_and_wait`) — **the only mandatory CB sync per token** | final norm / lm_head / softcap / argmax write `logits`, `norm_out`, `argmax_index`, `argmax_value`; host reads at `:3366`+ for token selection and optional Q8 rerank. | NO — terminal GPU drain is required before host reads.  This is exactly the llama.cpp pattern (`ggml-metal-context.m:670-722`).  Single-CB target keeps this site as the *only* boundary. |

**Production hot-path total CBs / decode token:**
- **`HF2Q_DUAL_BUFFER=0`**: **1** (one `begin` + one terminal `finish_*`).
- **`HF2Q_DUAL_BUFFER` default** (`Some(3)` per `INVESTIGATION_ENV.dual_buffer_split(num_layers)`): **2**.

#### Diagnostic / dump env-gated extra CBs (off in production)

All sites below are `finish` → `begin` pairs that introduce extra CB
boundaries only when their gate fires.  Listed for completeness of P1
enumeration per spec acceptance criterion.  None are on the steady-state
production decode path.

| File:Line | Encoder/GraphSession | Commit kind | Data dep protected | Env gate | fuse_safe |
|---|---|---|---|---|---|
| forward_mlx.rs:1779 / :1846 | `s.finish()` then `s = exec.begin()` (pre_quant K/V dump) | sync close + re-open | flush `attn_k_normed`, `v_src` to host for ADR-007 C-2 oracle | `INVESTIGATION_ENV.dump_pre_quant && layer_idx==0 && kv_seq_len==23` | YES — dump-only diagnostic. |
| forward_mlx.rs:1957 / :2032 | `s.finish()` then `s = exec.begin()` (HF2Q_DEBUG_TQ_RMS body bracket) | sync close + re-open | flush K-encoder write so the probe sees stable cache state | `HF2Q_DEBUG_TQ_RMS=1` | YES — debug-only. |
| forward_mlx.rs:1984 / :1997 | `let mut sp = exec.begin()` then `sp.finish()` (separate probe session) | OPEN + sync close | isolated CB for probe re-dispatch + RMS readback | same | YES — debug-only. |
| forward_mlx.rs:2041 / :2162 | `s.finish()` then `s = exec.begin()` (post_quant K/V/Q dump) | sync close + re-open | flush packed K/V + norms + Q to host for C-1-unlock dump | `INVESTIGATION_ENV.dump_tq_state && layer_idx==0 && kv_seq_len==23` | YES — dump-only. |
| forward_mlx.rs:2171 / :2180 | `s.finish()` then `s = exec.begin()` (Q/K/V dump pre-SDPA) | sync close + re-open | flush `attn_q_normed`/`attn_k_normed`/`v_src` to host | `dump_layers && (dump_detail_layer == Some(layer_idx) ‖ dump_all_cache)` | YES — dump-only. |
| forward_mlx.rs:2328 / :2388 | `s.finish()` then `s = exec.begin()` (dense KV-cache full-context dump) | sync close + re-open | flush `dense_kvs[L].k`/`.v` across full kv_seq_len | `dense + dump_layers + (dump_detail_layer ‖ dump_all_cache)` | YES — dump-only. |
| forward_mlx.rs:2767 / :2773 | `s.finish()` then `s = exec.begin()` (sdpa_out dump) | sync close + re-open | flush `sdpa_out` to host before O-projection | `dump_layers + ...` | YES — dump-only. |
| forward_mlx.rs:2782 / :2844 | `s.finish()` then `s = exec.begin()` (HF2Q_DUMP_SLIDING_LAYER_0 first-divergence dump) | sync close + re-open | flush layer-0 sliding Q (post-RoPE) / K / V / sdpa_out for offline comparison | `HF2Q_DUMP_SLIDING_LAYER_0 && layer_idx==0 && kv_is_sliding && s2c_step ∈ [1,10]` | YES — iter-18 ablation harness, not production. |
| forward_mlx.rs:2874 / :2879 | `s.finish()` then `s = exec.begin()` (post-attention residual dump) | sync close + re-open | flush `residual` after fused post-attn norm+add | `dump_layers && dump_detail_layer == Some(layer_idx)` | YES — dump-only. |
| forward_mlx.rs:3172 / :3177 | `s.finish()` then `s = exec.begin()` (per-layer hidden state dump, end-of-layer) | sync close + re-open | flush `hidden` (= layer output) for ADR-009 Phase 3A per-layer logging | `dump_layers` | YES — dump-only. |
| forward_mlx.rs:3210 / :3215 | `s.finish_with_timing()` then `s = exec.begin()` (HF2Q_SPLIT_TIMING body/head split) | sync close + re-open | flush body activations so head GPU section can be measured separately (~50 µs sync overhead per gate comment :3204-3205) | `INVESTIGATION_ENV.split_timing` (`HF2Q_SPLIT_TIMING=1`) | YES — measurement-only; trivially removable in single-CB rewrite. |
| forward_mlx.rs:3238 / :3243 | `s.finish()` then `s = exec.begin()` (pre-lm_head boundary dump) | sync close + re-open | flush `norm_out` (= final RMS norm output) | `INVESTIGATION_ENV.dump_boundary == Some(seq_pos)` | YES — dump-only. |

When all dump flags are simultaneously off, these contribute **0 extra
CBs**.  When `HF2Q_DUAL_BUFFER=N`: +1 extra CB (1 async commit + 1
begin).  When `HF2Q_SPLIT_TIMING=1`: +1 extra CB.

#### Aggregate — total CBs / decode-token (Gemma)

| Regime | Total CBs / decode token | Production fuse candidates | Notes |
|---|---:|---:|---|
| `HF2Q_DUAL_BUFFER=0` (pure single-session) | **1** | 0 mid-forward splits | Only terminal host-drain remains. |
| `HF2Q_DUAL_BUFFER` default (split after layer 3) | **2** | 1 mid-forward split (`:3193-:3194`) | Default. AMBIGUOUS-NEEDS-SPIKE. |
| `HF2Q_SPLIT_TIMING=1` | +1 over above | + body/head split | Measurement-only; trivially removable. |
| Dump / debug envs active | +N pairs | N/A | Diagnostic; excluded from production fuse-safe %. |
| `HF2Q_MLX_KERNEL_PROFILE=1` (separate entry: `forward_decode_kernel_profile` at :3631) | ~242 | profiling-only | Excluded by-design. |

#### Per-layer-type breakdown (Gemma production hot path)

The production hot path encodes **all** layer types into the **same**
single CB (or 2 CBs split after layer 3 with default `HF2Q_DUAL_BUFFER`).
There is no separate CB per layer-type; the breakdown below is therefore
expressed as *encoder time spent per layer-type within the single CB* —
a cost categorization, not a CB-count split.

| Layer-type | Layers / pass | Per-layer dispatch ranges (forward_mlx.rs lines) | CB-count contribution | fuse-safe note |
|---|---:|---|---|---|
| Embedding gather + scale | 1 | 1608-1617 | shares the single CB | YES — already inside the single CB. |
| Sliding-window attention | mixed (gemma-MoE: every-6th-is-global pattern → 25 sliding + 5 global on 30-layer config) | QKV + Q/K norms + KV-cache write (TQ or dense) + SDPA + O-proj + post-attn norm+add: ~1700-2870 per iteration | shares the single CB | YES — all `barrier_between` calls are intra-CB. |
| Global full-attention | 5 (every-6th-layer pattern) | same range (single code path gated by `kv_is_sliding`) | shares the single CB | YES — same intra-CB pattern. |
| Dense MLP (gemma3-dense) **or** MoE routing (gemma-MoE) | 30 | 2898-3160 per layer iteration (B8/B9/B10/B11 interleaved groups, ADR-006 Phase 4e) | shares the single CB | YES — concurrent groups inside the single CB. |
| Final RMS norm + lm_head + (softcap) + argmax | 1 | 3221-3314 | shares the single CB | YES — terminal sequence inside the single CB, closed by the terminal `finish_*`. |

**Total CBs by layer-type (Gemma production hot path) = 1 (or 2 with
default split), attributed across all layer-types via shared CB.**
Aligns with llama.cpp's `encode_async` pattern.

#### Fuse-safe summary (Gemma)

| Category | Sites | YES | NO | AMBIGUOUS-NEEDS-SPIKE |
|---|---:|---:|---:|---:|
| Production hot path (`begin` + dual_buffer commit + terminal finish) | 3 | 2 | 1 (terminal sync) | 1 (`HF2Q_DUAL_BUFFER` async split) |
| Dump / debug env-gated `finish`+`begin` pairs | 22 | 22 | 0 | 0 |
| `HF2Q_SPLIT_TIMING` body/head split | 2 | 2 | 0 | 0 |
| **Total** | **27 distinct sites** | **26** | **1** | **1** |

**Fuse-safe % (production hot path, excluding `HF2Q_DUAL_BUFFER`
ambiguity): 100% YES on collapsable boundaries; 1 mandatory terminal
NO.**

#### Gemma-specific stages with no qwen35 analog (and vice versa)

- **Gemma-only on this audit path:**
  1. *TQ KV encode* (`hadamard_quantize_kv` + optional `_hb` for 5/6/8-bit
     Lloyd-Max codebooks, lines 1858-1928) is gemma-flavored ADR-007
     territory; qwen35's `forward_gpu` does dense (or standard quantized)
     KV writes.  Per `project_tq_state_2026_04_21`, TQ is gated off via
     ADR-009 Track 3 fallback today; the encode dispatches still fire,
     then SDPA reads from `dense_kvs` instead of `k_packed`.  Same
     CB-fuse-safe story but **two extra dispatches per layer** live in
     the gemma single CB.
  2. *Fused MLP+MoE B8/B9/B10/B11 interleaving* (lines 2898-3160) — the
     ADR-006 Phase 4e concurrent-dispatch-groups pattern.  Qwen35's
     ADR-013 forward currently keeps MoE serial in many places; this
     gemma-side strength should be preserved in the single-CB rewrite.
  3. *Iter-18 / S2C / iter-21 Track B / iter-23 / iter-24* dump
     scaffolding — ADR-007 investigation hooks specific to the
     gemma-class TurboQuant work; no qwen35 analog in `forward_gpu.rs`.
- **Qwen35-only:** qwen35's `forward_gpu` opens a fresh
  `command_encoder` per layer (and often per sub-stage), spawning 103
  CBs / token per the qwen35 audit above.  **The gemma path has already
  collapsed to 1–2 CBs / token on its production hot path** — gemma is
  *ahead* of qwen35 on the single-CB axis.

#### Top ambiguous callouts for P3 (need spike to decide)

1. **`HF2Q_DUAL_BUFFER` async-split callsites (`forward_mlx.rs:3193`
   commit, `:3194` begin).**  Default-on at split=3; intentional
   counter-pattern to single-CB.  In a single-CB rewrite world this
   lever is mooted (one CB cannot async-overlap with itself).  The
   fuse-safety question is *whether the async overlap recovers more
   wall-clock than collapsing to 1 CB loses*.  In-source comment at
   `:3186-:3190` falsified two related variants but did *not* measure
   the no-wait async-overlap against pure single-CB.  **P3 spike:** bench
   `HF2Q_DUAL_BUFFER=15` vs single-CB; if async-overlap wins, single-CB
   rewrite must preserve a 2-CB option; if single-CB wins, this code
   path is deleted.
2. **Recorded vs direct-dispatch mode (`forward_mlx.rs:1589` vs `:1591`).**
   When `HF2Q_MLX_GRAPH_OPT=1`, the session captures into a `ComputeGraph`,
   then runs a reorder-pass at `finish_optimized_with_timing` (`:3323`).
   The capture-mode bookkeeping
   (`encoder.set_pending_buffer_ranges`, `annotate_last_dispatch_if_missing`)
   adds Rust-side overhead — exactly the kind of orchestration cost
   ADR-015 P3b targets.  Whether the single-CB rewrite preserves
   recorded mode, drops it, or merges its reorder pass into the dispatch
   loop is a design decision the audit cannot resolve from line numbers
   alone.  **P3 spike:** profile recorded vs direct mode under single-CB.

#### Implications for ADR-015 §Diagnosis + Phasing

(Recorded here for the queen / ADR maintainer.  The §Diagnosis table
above has been updated with the §Correction note that points to this
subsection.)

1. **§Diagnosis row "hf2q Gemma … ~30 CBs/pass / ~150 µs"** is stale —
   already corrected above.  Production reality at hf2q@`0b29397` is
   1 CB/pass (`HF2Q_DUAL_BUFFER=0`) or 2 CBs/pass (default), at
   ~1.6 µs / CB.
2. **D1 ("Migrate **both** … to a **single** GraphSession")** is already
   true for gemma; the gemma half of D1 is now a refactor-for-uniformity
   exercise, not a perf lever.
3. **Phase P3 row "gemma-second" expected delivery** changes: gemma
   single-CB recovers ~0 µs; until P0 (gemma baseline) lands, the
   gemma exit criterion (D4 second bullet) is a placeholder.
4. **P3b (Rust orchestration sweep) is the real lever for gemma** — same
   conclusion the qwen35-side audit reached.
5. **Erratum candidate for §References "Gemma's 'one GraphSession per
   layer' baseline (30 CBs / pass)"**: the comment-reference should be
   updated to point at `:1488` (the actual single-session contract) and
   the "30 CBs / pass" parenthetical struck.

#### Methodology notes

- All line numbers cite hf2q@`0b29397` (merged-worktree base); both
  Wave 1 variants used the same base.
- Site enumeration was performed by:
  `grep -nE "exec\.begin\b|exec\.begin_recorded\b|s\.finish\b|s\.commit\b|sp\.finish\b" src/serve/forward_mlx.rs`
  cross-checked against
  `grep -nE "GraphSession|graph_session|memory_barrier|enc\.commit"`.
  Production hot-path identification used the in-source contract
  comments at `:1488-:1492` ("ONE begin() → ONE finish()") and
  `:3317`/`:3328` (terminal finish).
- `forward_decode_kernel_profile` (line 3631) is excluded from the
  hot-path enumeration: it is gated by `HF2Q_MLX_KERNEL_PROFILE=1`,
  documented as *"intentionally slow (many sessions = many sync
  points)"*, and is a kprofile-only diagnostic.
- `mlx-native@adf7ee0` is the read-only reference for `GraphSession` /
  `CommandEncoder` semantics per ADR-014 fence.

## Open questions / risks

| ID | Risk / open question | Mitigation |
|---|---|---|
| **R1** | Barrier correctness: single-CB requires every cross-stage dependency expressed as `enc.memory_barrier()`. A missed barrier = silent corruption. | P4 bit-exact parity gate is mandatory. Spot-check with `mlx_native::ops::memory_barrier` debug logging during P3. |
| **R2** | Encoder lifetime + buffer pool: a long-lived encoder may hold buffer refs across the full forward pass, blocking pool reuse and inflating peak working set. | P3 measures pool stats; if `pool.alloc_count` > pre-ADR + 10%, size the pool up or refactor to release. |
| **R3** | `MLX_PROFILE_CB` per-label timing relies on CB granularity. With 1 CB / token, per-stage attribution is gone. | Switch to per-dispatch labels (`commit_labeled` becomes `dispatch_labeled` — already partially supported). Document the change in the P3 commit. |
| **R4** | The 6% gap may not be 100% encode overhead. If P2 shows µs/CB << 5, single-CB wins shrink. | P2 is the cost calibration. If the upper bound drops below ~3% headroom, re-scope to a different lever (Option B below) before P3. |
| **R5** | MoE expert dispatch is the largest contributor to dispatch count (256 experts × N selected per layer). Single-CB doesn't compress dispatch count, only CB count. If dispatch overhead dominates encode cost, single-CB still wins (one `commit` not 100), but the CPU encode cost itself doesn't shrink. | P2 also measures per-dispatch encode cost (encode N noop dispatches into 1 CB). If per-dispatch dominates, the lever is fewer dispatches, not fewer CBs — branch to a separate ADR (e.g., persistent-MoE-router or shader-side expert loop). |
| **R6** | Decode-only pivot: prefill is unchanged. Long-prefill parity inversion (`project_long_prefill_parity_inverts`) remains open. | Out of scope for ADR-015 — covered by Non-Goal 1. |

## Non-goals

1. **Prefill parity** — `pp≥1024` parity inversion at `-ub 512` matched batching is a separate axis, owned by a future ADR. ADR-015 measures decode only.
2. **Speculative / MTP decode** — ADR-013 P14 territory.
3. **Sampler upgrades beyond greedy** — ADR-008 placement decision stands; sampling lives in hf2q `sampler_pure.rs`.

## Alternatives considered

- **Option A: dispatch_apply parallel encoding (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:550`).** *Rejected for decode* — llama.cpp's own data shows `n_cb=1` is optimal for decode (`:458`). dispatch_apply is a prefill mechanism. Would still help our prefill — but per Non-Goal 1, that's a separate ADR. Not committed in this iteration.
- **Option B: cross-token speculative pipelining (draft + verify decode).** *Deferred* — substantial program-structure change. Becomes attractive only if Option A (single-CB) hits the wall before 1.00×. Land in a successor ADR if needed.
- **Option C: source-level kernel fusion (combine adjacent compute kernels into one shader).** *Falsified 9× in the M5 Max static-evidence kernel-hypothesis register* — most recent: `swiglu` Q4_0 `mv_id` (mlx-native@`4efeec0`, hf2q@`502364d`, −1.5%). The Metal compiler hoists what llama.cpp hand-tunes (`project_metal_compiler_auto_optimizes_static_levers`). Not a productive lever on M5 Max for current workloads.
- **Option D: keep status quo at 0.94×.** *Rejected.* Per `feedback_shippability_standing_directive` and `feedback_no_shortcuts`, "as fast as our peers" is the standing exit bar. 0.94× is not shippable.

## Literature foundations

The architectural choices below draw from the following primary
sources, each fetched and read directly via WebFetch against
arxiv.org.  Each citation is verified pre-training-cutoff
(no Perplexity-summary or post-cutoff claims included).  Most LLM
inference research is CUDA-centric; the §Apple-Silicon-applicability
note after each grouping calls out where Metal / unified-memory
porting changes the equation.

### Kernel-level performance — informs P3 inner loop kernel correctness + adjacent fusion candidates

- **Dao et al. (2022).** *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* arXiv:2205.14135.  Establishes IO-awareness — accounting for reads/writes between GPU memory tiers — as the canonical principle behind fast attention.  Tiling reduces HBM↔SRAM traffic, achieving up to 3× speedup on GPT-2.  Apple Silicon applicability: principle is universal; M-series GPU's unified memory removes the HBM/SRAM dichotomy but preserves the L2/threadgroup-memory hierarchy.  mlx-native already implements `flash_attn_prefill` + `flash_attn_vec` analogues with M5-tuned block sizes.
- **Dao (2023).** *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* arXiv:2307.08691.  Refines work partitioning across thread blocks and warps; reaches 50–73 % of A100 theoretical FLOPs/s.  Apple Silicon applicability: NVIDIA-warp-specific tactics (e.g. `swizzle` patterns) translate imperfectly to Metal SIMD-groups; the *parallel-decomposition* insight (single-head splits across thread blocks for occupancy) maps directly and is already echoed in our MoE-routed mat-vec kernels.

### System architecture — informs P3b orchestration sweep, KV-cache management

- **Kwon et al. (2023).** *Efficient Memory Management for Large Language Model Serving with PagedAttention.* arXiv:2309.06180 (SOSP 2023, **vLLM**).  Treats KV cache as paged virtual memory; achieves "near-zero waste" + 2–4× throughput vs FasterTransformer / Orca.  Apple Silicon applicability: portable.  `mlx_native::ops::kv_cache_copy` + the per-slot abstraction in `inference/models/qwen35/kv_cache.rs` is a related but coarser allocation strategy.  PagedAttention-style block reuse remains an open lever for our serve path (orthogonal to ADR-015 decode-perf scope).
- **Zheng et al. (2023).** *SGLang: Efficient Execution of Structured Language Model Programs.* arXiv:2312.07104.  Introduces `RadixAttention` for KV reuse across requests + compressed FSMs for structured output; up to 6.4× throughput vs prior systems.  Applicability: cross-request KV reuse is a serve-side concern (ADR-005 territory), not single-request decode latency.
- **Agrawal et al. (2024).** *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* arXiv:2403.02310.  Chunked-prefill scheduling: splits prefill into near-equal chunks so the queue admits new requests without pausing ongoing decodes; 2.6× higher serving capacity vs vLLM on Mistral-7B.  Applicability: directly relevant to our `project_long_prefill_parity_inverts` open question (Non-Goal 1 today, future ADR).  Confirms prefill chunking is a multiplier orthogonal to single-CB decode.

### Decode acceleration alternatives — informs §Alternatives Option B (speculative pipelining)

- **Leviathan et al. (2022).** *Fast Inference from Transformers via Speculative Decoding.* arXiv:2211.17192 (ICML 2023 Oral).  Foundational paper: small draft model proposes K tokens; large model verifies in parallel.  2–3× speedup with identical output distribution.  Applicability: portable; the verification-step parallelism is what mlx-native would need to support batched K-token forward.
- **Cai et al. (2024).** *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* arXiv:2401.10774.  Avoids the separate-draft-model burden by adding multiple decoding heads to the existing model + tree-attention verification.  2.2–3.6× speedup.  Applicability: `Qwen3.5-MoE`'s **MTP** (multi-token prediction) head per `convert_hf_to_gguf.py:Qwen3NextModel` (ADR-012 Decision 11) is precisely a Medusa-shaped feature; ADR-013 P14 owns the inference-side execution.  ADR-015 P3 single-CB rewrite is therefore *complementary* to Medusa-style decode — they compose multiplicatively.
- **Li et al. (2024).** *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* arXiv:2401.15077.  Operates at the second-to-top-layer feature level rather than token level; 2.7–3.5× speedup on LLaMA2-Chat 70B.  Applicability: portable; informs the design choice of *what* to predict in our future MTP path.
- **Fu et al. (2024).** *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding.* arXiv:2402.02057.  *Exact, parallel* decoding without any auxiliary draft model — Jacobi-iteration style.  Up to 1.8× MT-Bench, 4× with multi-GPU.  Applicability: M5 Max single-device, so the multi-GPU number doesn't apply, but the no-draft-model property is attractive given how much surface a draft-model path adds.

### Graph compilation — informs the single-CB rewrite framing (P3 architecture)

- **Chen et al. (2018).** *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning.* arXiv:1802.04799 (OSDI 2018).  Establishes the graph-level + operator-level optimization layering that lets a single high-level forward pass compile to a hardware-specific dispatch sequence.  Applicability: TVM has limited Metal support; the *layering principle* informs whether mlx-native's `GraphSession` should grow toward graph-compile semantics or stay eager-with-explicit-barriers.  ADR-015 P3 takes the latter (explicit barriers within one encoder); a successor ADR could revisit the graph-compile path.
- **Ding et al. (2023).** *Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs.* arXiv:2210.09603 (ASPLOS 2023).  Embeds scheduling directly into tensor programs via "task mappings"; introduces post-scheduling fusion that automates operator fusion after individual scheduling.  1.48× over ONNX Runtime / TVM-AutoTVM.  Applicability: post-scheduling fusion is the formal version of what our 9× falsified manual-fusion experiments were trying to discover empirically.  The data point that automated fusion still struggles to beat hand-tuned BLAS suggests that *simple* compile-time fusion isn't the lever for our 0.94×→1.00× workload — it would need to capture the *runtime* memory pattern that the M5 Max compiler already auto-hoists (per `project_metal_compiler_auto_optimizes_static_levers`).

### Quantization lineage — DWQ-aware framing (informs which kernel-level levers move the perf needle on our actual workload)

We don't ship uniform-precision Q4 / Q8 / FP16; we ship MLX-flavored
**DWQ** (Dynamic Weight Quantization) — *activation-calibrated mixed
precision* with sensitive layers promoted to higher bits per
`docs/ADR-012-qwen35moe-conversion.md` (default mix: 4-bit base / 6-bit
sensitive = `dwq46`).  The DWQ algorithm itself has no formal arxiv
paper that I could verify pre-cutoff; the lineage of techniques DWQ
synthesizes is well-documented:

- **Dettmers et al. (2022).** *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* arXiv:2208.07339 (NeurIPS 2022).  Identifies *emergent activation outliers* dominating transformer predictive performance; introduces vector-wise W8A8 + a mixed-precision decomposition isolating outlier dimensions into 16-bit while >99.9% of values run in 8-bit.  This is the **conceptual ancestor** of DWQ's "promote sensitive tensors to higher bits" rule — the difference is granularity (LLM.int8 is per-dimension; DWQ is per-tensor / per-expert).
- **Frantar et al. (2022).** *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* arXiv:2210.17323 (ICLR 2023).  Second-order one-shot PTQ; 175B → 3–4 bits in ~4 GPU-hours, negligible accuracy loss.  Foundational for "calibration-driven post-training quant" — DWQ inherits the calibration-set + per-tensor-error-measurement methodology.
- **Xiao et al. (2022).** *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* arXiv:2211.10438 (ICML 2023).  "Weights are easy to quantize while activations are not" → migrate quantization difficulty from activations to weights via mathematically equivalent transformation.  Informs why DWQ does **not** quantize activations: the math says you don't have to.  Confirms our W-only quantization path.
- **Dettmers et al. (2023).** *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314.  Introduces NF4 (4-bit NormalFloat) data type — "information-theoretically optimal for normally-distributed weights" — plus double-quantization of quantization constants.  Applicability: NF4's data-type choice is *adjacent* to but not the same as the GGUF Q4 / Q8 nibble layouts we ship.  An NF4-style data type shipped through GGUF + DWQ-mixed semantics is a future quant direction (out of ADR-015 scope; would belong in an ADR-014 successor).
- **Lin et al. (2023).** *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* arXiv:2306.00978.  *Channel-wise* salience: protecting just 1 % of salient weight channels (identified via activation distribution, not weight magnitude) substantially reduces quantization error.  AWQ avoids hardware-inefficient mixed precision by *scaling* salient channels.  Compare to DWQ which *is* mixed precision — AWQ takes the opposite tradeoff.  Both rely on activations identifying the salient set.  Tactical: AWQ-style channel-scaling could be a complementary post-DWQ layer for the 6-bit experts we already promote; out of P3 scope.
- **Kim et al. (2023).** *SqueezeLLM: Dense-and-Sparse Quantization.* arXiv:2306.07629 (ICML 2024).  Most directly DWQ-shaped of the lineage — *"sensitivity-based non-uniform quantization, which searches for the optimal bit precision assignment based on second-order information"*.  This is functionally what DWQ-mixed-4-6 does at a coarser granularity (per-tensor, not per-weight-cluster).  **Key framing for ADR-015**: the SqueezeLLM abstract declares *"the main bottleneck for generative inference with LLMs is memory bandwidth, rather than compute, specifically for single batch inference."*  This is the canonical statement of why decode is bandwidth-bound — and why per-CB / per-dispatch overhead (P2 + P3a) is the *secondary* axis after the kernel-level bandwidth efficiency that mlx-native's `quantized_matmul_*` family already targets.

### What the quantization literature says — and doesn't say — about our perf gap

**What it says:**
1. Decode is bandwidth-bound (SqueezeLLM, Lookahead Decoding, Medusa all converge on this).  Therefore the per-CB / per-dispatch CPU encoding overhead is not the *primary* axis — kernel bandwidth efficiency is.  ADR-015's measured numbers (P2: 1.6 µs/CB, P3a: 0.16 µs/dispatch) are consistent with this: even all of the encoding overhead removed leaves ~14 % of the gap as GPU work delta.
2. DWQ's per-tensor mixed-precision is the right scheme for quality (SqueezeLLM-style sensitivity-based assignment).  Switching schemes wouldn't close the perf gap — both hf2q and llama.cpp consume identical `dwq46` tensor data.

**What it does NOT say:**
1. The gap **cannot** come from DWQ-the-algorithm.  Both sides see the same Q4_K + Q6_K + Q8_0 tensor layouts in the same GGUF — only the kernel that decodes + multiplies differs.  Therefore: per-tensor-format kernel design (mlx-native's vs llama.cpp's `quantized_matmul_*` for each format involved in `dwq46`) is the relevant axis, not the quant scheme.
2. A DWQ-aware perf investigation should attribute per-CB GPU time to *which tensor format* dominates that CB — already partially recorded in ADR-012's per-CB profile (`f11cef9`).  An ADR-015 P3a' (DWQ-format-attribution micro-bench) is a candidate addition if P3 + P3b underdeliver.

### Apple Silicon literature gap — honest acknowledgement

Searches for arxiv work specifically on Apple Silicon LLM inference perf
returned nothing pre-cutoff that I could verify by direct fetch.  The
M-series GPU is academically under-studied compared to NVIDIA datacenter
parts.  This is itself a data point: bleeding-edge M5 Max optimization
has no published roadmap to copy from; the lever space we explore in
ADR-015 is one we have to discover empirically.  See P2 + P3a measured
data above.

## Why ADR-015 stays in mlx-native (and what coreml-native would be for)

We also own [`/opt/coreml-native`](https://github.com/robertelee78/coreml-native) — a sibling pure-Rust crate exposing CoreML inference (CPU/GPU/ANE).  ADR-015 stays in `mlx-native` rather than pivoting to CoreML for three concrete reasons grounded in our actual workload, not in framework preference:

1. **DWQ-mixed-4-6 tensor data is not a CoreML quant scheme.**  Both `hf2q` and `llama.cpp` consume identical `dwq46` GGUF tensors via their own Metal-GPU `quantized_matmul_*` kernels.  Routing through CoreML would force re-quantization through `coremltools`' API, losing our DWQ calibration and breaking the apples-to-apples bench against `llama.cpp`.
2. **MoE expert routing + Gated DeltaNet are not standard CoreML ops.**  CoreML's compiler maps a fixed op set to ANE / GPU / CPU; novel architectures (Qwen3.5-MoE's hybrid 3:1 DeltaNet:FullAttn, our 256-expert + custom router) fall back to CPU.  ANE acceleration evaporates exactly where our hot path lives.
3. **The 0.94× gap is not a runtime-choice problem.**  Per ADR-012 §Optimize, hf2q's GPU compute (8.45 ms) ≈ llama.cpp's (8.38 ms) — the gap is in CPU-side encoding/orchestration on **the same Metal substrate**.  Switching to CoreML wouldn't address that; it would replace the lever space with a different one whose ceilings on our model class are unknown and uncontrolled.

CoreML/ANE *is* the right tool for adjacent workloads where ADR-015 doesn't apply: standard ViT prefill (mmproj on Gemma-4V / Qwen vision), BERT-style embeddings (the `bert-test` target), Whisper if we ever pick that up.  The mlx-vs-CoreML strategic comparison for a possible Qwen3.5-MoE-on-ANE hybrid is its own ADR worth of work — see [ADR-016](./ADR-016-mlx-vs-coreml-strategic-comparison.md) (stub, 2026-04-26).

## Architectural implications given full-stack ownership

We own both `/opt/mlx-native` (kernel + dispatch substrate) and
`/opt/hf2q` (forward orchestration + serving).  That gives ADR-015
a strictly larger option space than any consumer of an upstream
inference framework:

1. **No API stability constraint.**  Single-CB rewrite can change
   `mlx_native::CommandEncoder` signatures freely; consumers are
   only us.
2. **Persistent / megakernel options open.**  An mlx-native ADR
   succeeding ADR-015 can investigate a persistent-thread Metal
   kernel that loops *internally* over decode tokens — the literature
   above (FlashAttention's tiling principle + Hidet's task-mapping
   compositionality) sketches the design space.  Single-CB is the
   conservative first lever; megakernel is the next-octave lever.
3. **Shader-level cross-token pipelining.**  Lookahead Decoding
   (2402.02057) and Medusa-style MTP (2401.10774) both decompose to
   "more dispatches per wall-clock token" — a regime where the per-
   dispatch overhead measured in P3a starts mattering more, but the
   speedup more than compensates.  ADR-013 P14 is the consumer side;
   mlx-native's role would be to expose efficient batched-K forward.
4. **The ground truth is what we measure.**  Per `feedback_ground_truth_is_what_we_can_measure_now`,
   any literature claim that doesn't survive a P2-style cold-SoC bench
   on M5 Max is irrelevant to shippability.  ADR-015 commits to
   measure-3x discipline at every phase gate.

## Capturing M5's GPU Neural Accelerators (Metal 4 TensorOps + NAX)

Source for this section: deep technical brief at `/tmp/cfa-adr016/research-metal4-tensorops.md` (also stored at memory key `swarm-cfa-adr016/deep-research/metal4-tensorops`, 2026-04-26).  All API signatures and file:line citations were verified against Apple developer docs, MLX source (`github.com/ml-explore/mlx`) and `/opt/mlx-native` source — no training-data recall used.

### The path Apple is publicly accelerating

Apple's M5 generation publishes a measured **3.33–4.06× faster TTFT vs M4** and **1.19–1.27× decode** across Qwen 1.7B / 8B / 14B / 30B-A3B-MoE / GPT OSS 20B at prompt=4096 / generation=128 ([Apple ML Research M5/MLX post](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)).  The 4× lives in **GPU-integrated Neural Accelerators (NAX)** — per-GPU-core matrix-multiplication units accessed via **Metal Performance Primitives (MPP) TensorOps** at the `metal4.0` shader standard.  M5 Max (40-core GPU) = 40 × 1,024 FP16 FMA / cycle.  This is the substrate `mlx-native` already targets — but only partially.  The 1.19–1.27× decode boost is bandwidth-bound (460 → 614 GB/s); ADR-015 P2 + P3 + P3b own that axis.  The TTFT axis is compute-bound and is the new lever introduced here.

The standalone 16-core ANE is **not** the path that captures Apple's headline numbers.  CoreML / ANE work is opportunistic encoder offload (ADR-016 territory), not LLM-body acceleration.

### mlx-native current state — gap analysis

`mlx-native` already uses `mpp::tensor_ops::matmul2d` in four `*_tensor.metal` GEMM kernels, with correct `get_destination_cooperative_tensor` output stores.  That part is shipping correctly.  Verified gaps vs MLX's NAX path:

| Lever | mlx-native today | MLX NAX path | Source |
|---|---|---|---|
| Inner MMA descriptor | `matmul2d_descriptor(NR1=32, NR0=64, NK=32, …)` | `matmul2d_descriptor(16, 32, 16, …)` | `quantized_matmul_mm_tensor.metal:185-231` vs `mlx/.../steel/gemm/nax.h` |
| MMA scope | `execution_simdgroups<4>` | `execution_simdgroup` (single, 32 lanes) | same |
| `reduced_precision` flag | `false` everywhere | not explicitly false in NAX path | `*_tensor.metal` files |
| Outer tile sizes (M5 Max) | NR0=64, NR1=32 | bm=64, bn=128 or 256, swizzle=2 | MLX `matmul.cpp` commit `0879a6a` |
| Morton/Hilbert dispatch order | absent (raster order) | swizzle=2 / Morton for large GEMM | grep mlx-native: zero matches |
| macOS 26.2 / arch-gen ≥ 17 runtime gate | absent (`probe_tensor_mm()` is M3+ compile probe only) | `is_nax_available()` at runtime | `quantized_matmul_ggml.rs:155-169` vs MLX `device.cpp` |
| Attention kernel TensorOps | **absent** — `flash_attn_prefill.metal:621-754` uses M1-era `simdgroup_matrix` + `simdgroup_multiply_accumulate` | `steel_attention_nax` (commit `b41b349`) | `flash_attn_prefill.metal:621-754` |
| Split-K NAX | absent | `steel_gemm_splitk_nax` for large K | grep mlx-native: zero matches |

Per ADR-015 §"Prefill kernel breakdown (measured)" memory pin: MoE 39 %, MM 25 %, FA 16 %.  The single biggest unrouted lever is **flash-attention prefill** (16 % of compute) running on the M1-era simdgroup path.  GEMM kernels are partially routed through TensorOps but with non-NAX-tuned tile sizes and no Morton-order dispatch.

### P3c — three concrete actions to capture the 3.3–4.1× TTFT speedup

**P3c is a new sub-phase orthogonal to P3 (single-CB rewrite) and P3b (Rust-orchestration sweep).**  All three are shader-internal changes; none affect command-buffer lifecycle, encoder lifetime, barrier bookkeeping, or Rust orchestration cost.

| Action | File | Today | Change | Expected lever | Risk |
|---|---|---|---|---|---|
| **P3c.1 — Morton/Z-order dispatch for large prefill GEMM** | `quantized_matmul_mm_tensor.metal`, `dense_mm_bf16_tensor.metal`, `dense_mm_f16_tensor.metal` | Standard 2D raster grid (`tgpig`) | Map flat dispatch index to Morton coordinates before tile-offset calc | **1.5–2× on large-K GEMM** (seq ≥ 1024); per Tech Talk 111432, alone is the difference between ~50 % and ~100 % NAX utilization on a 4K×4K matmul (~6× speedup demo) | Tile-boundary tail handling for non-power-of-2 `ne0`/`ne1`; ~20 lines per kernel, shader-only |
| **P3c.2 — NAX-tuned outer tile sizes + macOS 26.2 / arch-gen ≥ 17 runtime gate** | `quantized_matmul_ggml.rs` (gate logic), new `quantized_matmul_mm_nax.metal` instantiation | NR0=64, NR1=32 always; M3+ compile probe only | Add `is_nax_available_m5()` Rust gate (mirror MLX `device.cpp`); on M5+macOS 26.2, dispatch new variant with bm=64, bn=128, swizzle=2 | **10–20 % on prefill GEMM** | Existing tensor path stays as M3/M4 fallback; no breakage |
| **P3c.3 — Port `flash_attn_prefill` inner matmul from `simdgroup_matrix` to `mpp::tensor_ops::matmul2d` + Morton dispatch** | New `flash_attn_prefill_nax.metal` (gate behind `is_nax_available_m5()`); existing kept as M3/M4 fallback | Lines 621–754 use `metal::simdgroup_matrix<T>` + `simdgroup_multiply_accumulate` | Replace inner QK^T and scores×V matmul loops with `mpp::tensor_ops::matmul2d` (16×32×16 fragments per MLX NAX); preserve online-softmax tile state | **2–3× on TTFT for long prefill** (seq ≥ 2048); attention is 16 % of prefill compute — this is the largest single lever for the TTFT headline | Non-trivial port — `flash_attn_prefill` is a fully fused SDPA with online softmax, sliding-window masking, and partial-output reduction.  MLX's `steel_attention_nax` (`b41b349`) is the reference port. |

The 3.33–4.06× Apple-measured TTFT speedup is achievable only if **all three** actions land — GEMM Morton + tile-sizing alone covers ~64 % of prefill compute (MoE 39 % + MM 25 %), but the remaining 16 % attention prefill is the bottleneck if it stays on `simdgroup_matrix`.  P3c.3 is the highest-impact and highest-effort of the three.

**Optional P3c.4 — `reduced_precision = true` for BF16 prefill attention scores** (5–15 % additional lever on attention GEMM): permits internal FP16 accumulation on Neural Accelerators.  *Numerically risky* per memory pin `project_vit_attention_bf16_softmax_drift` (BF16 cast in attention K already causes ~0.68 logit perturbation on saturated-softmax winners).  Gate behind sourdough parity check (ADR-015 D3).  Do not enable for weight-dequant matmuls (Q4_0, Q6_K) — accumulation precision directly impacts model quality.

### OS / version constraints

| Requirement | Value | Source |
|---|---|---|
| Minimum macOS for NAX hardware activation | **26.2** | MLX `device.cpp` `is_nax_available()` |
| Metal shader standard | `-std=metal4.0` | MLX CMake, commit `54f1cc6` |
| GPU architecture generation | **≥ 17** (non-phone) / ≥ 18 (phone) | MLX `device.cpp` |
| M5 Max architecture char | `'s'` (Max) / `'c'` (Pro) / `'d'` (Ultra) | MLX `matmul.cpp` commit `0879a6a` |
| BF16 tensor support | macOS 26.1 | Apple Tech Talk 111432 |
| 4-bit / 8-bit integer tensors | macOS 26.4 | Apple Tech Talk 111432 |
| Cooperative tensors as matmul inputs | macOS 26.3 | Apple Tech Talk 111432 |
| `MLX_METAL_NO_NAX` escape hatch | env var, runtime force-fallback | MLX `device.cpp` |

**Hard implication:** M4 (gen 16) does NOT have NAX hardware.  The 3.3–4.1× TTFT speedup is exclusively M5+ + macOS 26.2+.  P3c gates must bottom out to the existing `*_tensor.metal` simdgroup-matrix path on M3/M4 and on M5 with macOS < 26.2.  ADR-015's bench methodology (`feedback_perf_gate_thermal_methodology`) already requires documenting the macOS version for every bench run; that documentation now becomes load-bearing — same kernels, different hardware activation.

### Composition with P3 + P3b — strict orthogonality

| Aspect | P3 (single-CB) | P3b (Rust orchestration sweep) | P3c (NAX kernels) |
|---|---|---|---|
| What changes | Command-buffer lifecycle (102 → 1 CB / token) | Helper-function indirection, buffer pool, barriers, KV cursor | Shader-internal MMA descriptor, Morton order, attention port |
| Where | Rust host-side `forward_gpu` / `forward_decode` | Rust host-side `apply_*` helpers | Metal shader `*_tensor.metal`, `flash_attn_prefill*.metal` |
| Affects | CB encode time, encoder lifetime | CPU-side per-dispatch overhead | GPU-side compute throughput |
| Composes with others how | Removes 100 of the 102 encoder lifetimes | Removes ~140 µs of the ~288 µs/token Rust residual | Multiplies prefill TTFT independently |

P3c is **fully orthogonal** to P3 and P3b.  Single-CB removes encoder boundaries; orchestration sweep removes CPU overhead; NAX kernels increase per-dispatch GPU throughput.  Order-of-operations recommendation: **land P3 + P3b first** (they target the existing 540 µs/token decode gap and clear ADR-015 D4 ≥ 1.00× of llama.cpp); **open P3c only after ADR-015 P6 bench gate clears**.  Reasoning: avoid stacking shader-level complexity on top of an unmeasured decode-path baseline.  Once decode is at 1.00×, P3c becomes a TTFT-only lever measured against Apple's own MLX numbers, not against llama.cpp.

P3c does NOT change the ~288 µs/token Rust-orchestration residual identified in ADR-015 §"Budget reconciliation".  However once P3c lands, faster TTFT compounds with faster decode multiplicatively at the user-perceived end-gate (time to complete a response): TTFT improvement + decode improvement = total response-time improvement.

### Open questions — to refine before P3c opens

| ID | Question | Resolution |
|---|---|---|
| **Q-NAX-1** | Does mlx-native's existing `matmul2d_descriptor(32, 64, 32, …)` already activate NAX hardware on M5 Max, or stay on simdgroup-matrix? | Profile via Instruments "Metal System Trace" + GPU counter `gpu.counters.neuralAcceleratorUtilization` (if exposed) on M5 Max with a prefill workload.  If activation is already happening, P3c.2 inner-tile changes are moot; P3c.1 (Morton) and P3c.3 (attention port) remain.  **§P3a' update (2026-04-27):** the Wave 2a TimeProfiler trace did NOT use Metal System Trace and did NOT capture GPU counters — it only resolves CPU-side frames.  Q-NAX-1 remains open; the answer requires a separate Metal System Trace run, not the same xctrace recipe. |
| **Q-NAX-2** | What is the actual TTFT baseline for mlx-native vs mlx-lm on M5 Max at pp=4096?  ADR-015 D4 baseline is decode-only (n=256). | Measure before P3c sized.  Add TTFT bar to D4 exit criteria when P3c opens.  Reference: Apple's MLX/M5 post = 3.3–4.1× over M4 across Qwen 1.7B–30B-MoE. |
| **Q-NAX-3** | What macOS version is the M5 Max benchmark machine running? | Document explicitly in every cold-SoC bench log; NAX requires 26.2+.  If on 26.1 or 15, P3c lever is unavailable.  **§P3a' update (2026-04-27):** Wave 2a benchmark machine is **macOS 26.4.1 build 25E253** (`run-20260427T030903Z.metadata.json`).  This is **above** the 26.2 NAX threshold; subsequent P3c work on this machine can assume NAX hardware is available. |
| **Q-NAX-4** | Is `execution_simdgroups<4>` causing NAX stalls on long K-loops? | Profile with Instruments' Metal GPU Counter `ShaderOccupancy` and (if exposed) `NeuralAcceleratorUtil` on a 4096-token prefill.  If stalls are present, restructure to `execution_simdgroup` with 4× more threadgroups per launch.  **§P3a' update (2026-04-27):** the Wave 2a trace was a **decode** workload (1 token / forward pass, 64 tokens), not a long-K prefill — it cannot inform Q-NAX-4 in either direction.  Q-NAX-4 remains open and requires a dedicated long-K prefill trace. |
| **Q-NAX-5** | Does `swizzle=2` in MLX's M5 tuning mean Morton/Z-order, column-first, or something device-specific? | Read MLX `matmul.cpp` swizzle implementation directly before implementing P3c.1; `swizzle=0` is raster, `=1` is column-first per common convention but verify. |
| **Q-NAX-6** | Does `reduced_precision = false` affect decode-path throughput on M5? | Decode is bandwidth-bound (SqueezeLLM canonical statement, confirmed by ADR-015 P2 / P3a data).  For decode-path weight-dequant matmuls (m=1), compute is not the bottleneck.  The flag has no effect on bandwidth-bound ops — confirm empirically before spending time on precision-flag experiments for decode. |
| **Q-NAX-7** | Does porting `flash_attn_prefill` require a new file or modify the existing one? | Recommend: new `flash_attn_prefill_nax.metal` gated behind `is_nax_available_m5()`, keeping existing simdgroup path as M3/M4 fallback.  Two-variant pattern matches MLX's own approach. |

## References

- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:448-614` — main encode loop; `:550` `dispatch_apply`; `:458` "optimal n_cb is 1 or 2"
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:670-722` — `encode_async` block, the per-CB encoding worker
- `/opt/hf2q/docs/ADR-012-qwen35moe-conversion.md:215-256` — diagnosis + closure
- `/opt/hf2q/src/inference/models/qwen35/forward_gpu.rs` — current per-layer-encoder qwen35 forward (target of P3 rewrite)
- `/opt/hf2q/src/serve/forward_mlx.rs:1309` — Gemma's "one GraphSession per layer" baseline (30 CBs / pass)
- mlx-native: `mlx_native::GraphSession`, `mlx_native::CommandEncoder::memory_barrier()`
- mlx-native examples: `cb_cost_calibration.rs`, `dispatch_cost_calibration.rs` (P2 + P3a empirical benches)
- Memory pins: `feedback_perf_gate_thermal_methodology`, `feedback_shippability_standing_directive`, `feedback_never_ship_fallback_without_rootcause`, `feedback_no_broken_windows`, `project_metal_compiler_auto_optimizes_static_levers`, `project_end_gate_reality_check`, `feedback_ground_truth_is_what_we_can_measure_now`

## Changelog

- **2026-04-26 — Proposed (initial).** Diagnosis pivot from ADR-012 §Optimize: gap is CB-count, not CB-encode-time. Single-CB forward pass selected over `dispatch_apply` based on llama.cpp's own n_cb data.
- **2026-04-26 — Title broadened to "general decode-path speed improvements" + scope extended to gemma family** per standing directive *"we need this coherence and speed for qwen and gemma families of models"*.
- **2026-04-26 — P2 calibration empirical:** `async µs/CB ≈ 1.6 µs` (3.1× lower than working assumption).  Single-CB recovers ~32% of the 500 µs gap, not ~100%.  Plan revised: P3a per-dispatch cost calibration inserted before P3; P4 added as a second-lever phase whose target is determined by P3a.  Title kept "single-CB" because that's still the first lever; scope is honestly "general decode speed improvements" with the single-CB rewrite as Phase 1 of N.
- **2026-04-26 — P3a calibration empirical:** `µs/dispatch ≈ 0.16 µs` at N≥500 — within ±15% of llama.cpp's ~0.14 µs/dispatch.  Shader-launch path is **not** materially slower.  Budget reconciliation: ~252 µs of the 540 µs gap is encode-time (CB + dispatch); ~288 µs (~53%) is **residual unaccounted** that lives in Rust-side orchestration.  P3b (Rust orchestration sweep) added as parallel phase to P3 single-CB rewrite.  Together P3 + P3b target ≥1.00× exit criteria.
- **2026-04-26 — Literature foundations infused.**  11 arxiv papers fetched + verified pre-training-cutoff, organized into Kernel-level (FlashAttention 1+2), System architecture (vLLM, SGLang, Sarathi-Serve), Decode acceleration (Speculative Decoding, Medusa, EAGLE, Lookahead), and Graph compilation (TVM, Hidet) categories.  Each entry calls out Apple-Silicon applicability honestly.  Apple-Silicon-specific gap acknowledged: no pre-cutoff arxiv work on M-series LLM inference perf that I could verify by direct fetch.  Added §Architectural implications given full-stack ownership — single-CB is the conservative first lever; megakernel + cross-token pipelining open as next-octave levers given we own both layers.
- **2026-04-27 — §"P3a' live profile pass" infused (CFA Wave 2a, `cfa-20260426-adr015-wave2a-p3aprime`).** xctrace TimeProfiler captured 3 cold-SoC trials × 64 decode tokens on macOS 26.4.1 / M5 Max with the qwen3.6-27b-dwq46 dense fixture.  Decode-only stack filtering (`forward_gpu_greedy`) yields the live ranked residual: rank 1 is `MlxDevice::alloc_buffer` → `IOGPUResourceCreate` → `mach_msg2_trap` at 3719 µs/token (Mach-IPC kernel-call dominated, ~1 IPC per GPU resource × ~hundreds of decode-token allocations); rank 3 is `command_encoder()` churn at 724 µs/token (closes under P3 single-CB); rank 4's `build_rope_multi_buffers` at 208 µs/token is independently fixable by hoisting per-call rope-params buffers to model-load time.  Hypothesis register: H1 (`gpu_ffn.rs:397-404` proj alloc) NOT FOUND at literal site (dense path doesn't call proj()) but the underlying *category* CONFIRMED at 37× the static estimate via the rank-1 alloc_buffer chain; H2 (apply_imrope) CONFIRMED 5.8× larger than estimated; H3 (command_encoder churn) CONFIRMED 13× larger; H4 (memory_barrier) FALSIFIED at literal site (sub-1ms-sample at 1ms granularity); H5 (with_context allocation) FALSIFIED — confirms Codex Wave 1 prior (`format!` lazy via closure).  Coverage gap flagged: dense fixture does not exercise the apex MoE expert proj path; Wave 2b must re-validate H1's literal site against the apex 35B-A3B fixture before committing P3b reductions to MoE-specific call sites.  Supersession note added to §Diagnosis pointing at §P3a'.  NAX questions Q-NAX-1, Q-NAX-3, Q-NAX-4 received in-row updates where the trace informed them (Q-NAX-3: confirmed macOS 26.4.1 ≥ 26.2 NAX threshold; Q-NAX-1, Q-NAX-4: noted that decode-only TimeProfiler trace cannot inform GPU-counter / long-K-prefill questions, both remain open).  Trace artifacts referenced by absolute path (not committed); only `scripts/profile-p3aprime.sh`, this ADR section, and `docs/perf-traces/.gitkeep` land in git.  The 9th static-evidence kernel hypothesis class (Wave 1 P3b suspect list) joins the falsified-by-live-evidence list per `project_metal_compiler_auto_optimizes_static_levers` — the Mach-IPC / IOGPU-resource-creation overhead was not on Wave 1's static suspect list because grep against Rust source can't see kernel boundary crossings.
- **2026-04-27 — D4 first bullet CONFIRMED on apex 5-trial validation: ratio 1.0356× (hf2q LEADS by 3.5%).**  Methodology: 5 cold trials × n_gen=256 × 120s settle on hf2q@`be35743` / mlx-native@`9e6ffc0`.  hf2q `[104.2, 103.9, 105.5, 106.3, 105.2]` median **105.200 t/s**; llama-bench `[96.61, 102.17, 101.76, 101.58, 100.59]` median **101.580 t/s**; **ratio 1.0356×**.  Apex same-day result is robust — confirms the afternoon 1.0308× was not a one-off.  **Methodology caveats noted (per `feedback_bench_process_audit`):** (a) hf2q trials 1-2 ran with `mcp-brain-server` ~18% CPU online; trials 3-5 ran with mcp-brain offline; trial-by-trial: 104.2/103.9/105.5/106.3/105.2 — clear +1.5 t/s recovery after mcp-brain-offline transition, suggesting "clean median" of trials 3-5 is **105.5 t/s**; (b) llama-bench trial 1 (96.61) is a -5% outlier vs trials 2-5 (100.59-102.17), most likely a cold-page-cache effect after the 12.5-minute hf2q phase (apex.gguf was evicted from filesystem cache), not mcp-brain contamination — trials 2-5 cluster tightly at 100.59-102.17; (c) `rust-analyzer` (~27% CPU when IDE open on workspace) not quiesced for this run; for future P5+P6 bench gates, both mcp-brain AND rust-analyzer should be paused with `ps` audit captured per trial (now wired into `bench-baseline.sh` via `*.process-audit` files).  **Even with these caveats, both medians (full and clean-subset) put apex hf2q in 1.03-1.04× LEAD — D4 first bullet is met regardless of subset.**  The morning 120 t/s llama-bench reading (giving 0.917× ratio) is now understood to have been the anomalous reading: fresh-boot / fresh-page-cache state with mcp-brain at variable load.  iter8c remains scoped to gemma kernel-fusion ONLY.

- **2026-04-27 — D4 second bullet still OPEN on gemma same-day rebench: ratio 0.8303× (hf2q -17% behind, RECOVERY 20.4% NEEDED).**  Same methodology as apex rebench (3-cold trials, n_gen=256, 60s settle, RAM-headroom + thermal gates) at hf2q@`748a808` / mlx-native@`a292f47` (afternoon 18:38 UTC).  hf2q gemma `[86.8, 86.2, 87.3]` median **86.800 t/s**; llama-bench `[105.16, 102.70, 104.54]` median **104.540 t/s**; ratio **0.8303×**.  Compare to morning P0 baseline (04:58 UTC, 14 hours earlier): hf2q 87.8 / llama 104.52 = 0.840×.  **llama-bench gemma is essentially STABLE same-day** (104.52 → 104.54, +0.02 t/s, 0% drift) — VERY different from apex's same-day llama-bench drop (120.01 → 101.86, -15%).  hf2q gemma drifted -1 t/s (87.8 → 86.8); ratio drifted -0.7pp (0.840× → 0.830×).  **D4 second bullet (gemma ratio ≥ 1.00×) is NOT MET; iter8c remains scoped to gemma kernel-fusion only.**  The gemma gap is real, stable, and ~17% wide — recovery requires structural levers (kernel fusion, fused attention).  Per `feedback_correct_outcomes`: do not declare ADR-015 closed without D4 BOTH bullets met.  **Asymmetry implication for apex:** the contrast between gemma's 0% llama-bench drift and apex's 15% drop suggests the apex-morning 120 t/s (giving 0.917× ratio) MAY have been the anomalous reading — fresh-boot / fresh-memory state — and apex-afternoon 102 t/s (giving 1.030× ratio) is closer to reality.  However, the OPPOSITE is also possible: apex's bench is more thermally-sensitive than gemma's.  Resolution: run task #17 (5-trial apex validation, ≥120s settle, fresh SoC) — this is now the gating measurement for D4 first bullet.  iter8c blocking gates: (a) task #17 confirms apex ≥ 1.00× robustly (if not, iter8c needs apex too); (b) iter8c CFA spec finalized for gemma kernel-fusion (Wave 1: norm+Qproj fusion, Wave 2: fused-attention super-kernel).  Raw artifacts: `/tmp/adr015-bench/baseline-gemma-rebaseline-iter8c-prep-20260427T183817Z.{summary.txt, metadata.json, hf2q.trial-{1,2,3}.stdout, llama.trial-{1,2,3}.stdout}`.

- **2026-04-27 — D4 first bullet REACHED on apex same-day bench: ratio 1.0308× (hf2q LEADS).**  Same-day rebaseline at hf2q@`1b0e744` / mlx-native@`9d8ec2a` (this afternoon, 17:41 UTC, ~5h after iter7 morning bench).  3-cold-trial methodology (n_gen=256, 60s settle, RAM-headroom + thermal gates).  hf2q `[105.0, 104.8, 105.4]` median **105.000 t/s**; llama-bench `[101.97, 101.41, 101.86]` median **101.860 t/s**; **ratio 1.0308× — hf2q is 3% FASTER than llama.cpp on apex 35B-A3B same-day**.  Compare to morning bench (post-iter7, 12:50 UTC): hf2q 110.1 / llama 120.01 = 0.9174×.  Both absolute numbers dropped (hf2q -5 t/s, llama -18 t/s) but the ratio FLIPPED: morning 0.9174× → afternoon 1.0308×.  Same-day comparison is the only valid one per `project_end_gate_reality_check`: *"Peer drift is its own invariant; re-measure llama.cpp on the day."*  **D4 first bullet (apex ratio ≥ 1.00× same-day) is MET.**  Caveats: (a) llama-bench's 18-t/s drop (-15%) is large; could be SoC thermal state from 3 prior metal-system-trace captures + iter8c-prep dispatch traces in the past hour; 60s settle may not cool a max-loaded M5 Max enough — Wave 2b methodology requires 5 trials with longer settle, this rerun used 3; (b) `pmset -g therm` clean throughout (no CPU_Speed_Limit warnings) but Apple Silicon's "soft thermal management" can manifest below the warning threshold; (c) hf2q's 105.0 was tight (0.6% range across 3 trials) — measurement is robust on the hf2q side; the variance is on llama's side.  **Strategic implication for iter8c:** the premise (close +9% gap to ≥1.00×) is OBSOLETE for apex.  iter8c re-scopes to gemma side only (D4 second bullet still open) UNTIL same-day gemma rebench validates whether D4 second bullet is also met or remains open.  Per `feedback_correct_outcomes`, do not declare D4 closed until two-trial-set validation against the full Wave 2b methodology (5 cold trials, ≥120s settle, fresh SoC).  Hard prereq before any iter8c CFA: (a) same-day gemma rebench — if ≥1.00× → ADR-015 moves to P5+P6 (parity gates) without needing kernel-fusion shader work; if <1.00× → iter8c targets gemma; (b) longer-cycle apex re-run (5 trials, 120s settle) to validate the 1.0308× isn't peak-vs-trough timing.  Raw artifacts: `/tmp/adr015-bench/baseline-apex-rebaseline-iter8c-prep-20260427T174152Z.{summary.txt, metadata.json, hf2q.trial-{1,2,3}.stdout, llama.trial-{1,2,3}.stdout}`.

- **2026-04-27 — iter8c-source partial: hf2q apex MoE path is ALREADY batched (mul_mm_id), so the 12× lever lives elsewhere — likely concurrent-dispatch + explicit barriers.**  Read `gpu_ffn.rs::build_moe_ffn_layer_gpu_q_into` (lines 1145-1340).  The path uses `quantized_matmul_id_ggml` for both `expert_gate_q` and `expert_up_q` — i.e. it DOES batch all active experts into ID-aware mat-muls (gate, up, down).  Per-MoE-layer dispatches enumerated: Phase A 4 projs + Phase B 1 softmax_topk + 1 silu_mul + Phase C 2 mat-mul-id + 1 shared_down + Phase D 1 silu_mul + Phase E 1 mat-mul-id (down) + Phase F 1 moe_weighted_reduce = **~12 dispatches per MoE layer**, with `enc.memory_barrier()` between phases (5 barriers per layer).  36 layers × 12 + ~9/layer attention × 36 + lm_head/argmax = ~760/token estimate — within ~30% of measured 558/token (sparse expert routing reduces actual count).  **So the lever is NOT "stop per-expert dispatch" — that hypothesis was wrong.**  The real lever per `project_mlx_native_concurrent_dispatch`: mlx-native uses **concurrent dispatch type** (default in mlx-native, where all dispatches in one encoder run concurrently and explicit `enc.memory_barrier()` is needed between RAW/WAR/WAW); llama.cpp's ggml-metal uses **serial dispatch type** (implicit per-dispatch barriers within an encoder, no explicit barrier calls) AND aggressive graph-node fusion at the `ggml_metal_graph_compute` level.  The 12× metal-system-trace ratio likely reflects per-encoder-or-per-pass events, not raw dispatchThreadgroups counts.  **Genuine iter8c levers (in order of recovery × effort):** (1) **graph-node fusion of attention ops** — fold RMS-norm + Q-proj into one kernel; fold RoPE + KV-copy into one; fold SDPA + O-proj into one; estimated 3-4 dispatches/layer recovered; medium effort; (2) **fused-attention super-kernel** like llama.cpp's `flash_attention` — folds Q*K^T + softmax + V*scores into one Metal function; large recovery on attention; large effort; (3) **fused MoE super-kernel** — folds gate + silu*up + down into one Metal kernel per active expert; largest recovery on apex MoE; largest effort; (4) **switch concurrent → serial dispatch type within compute encoders** — would let Metal driver insert implicit barriers, eliminating the need for `enc.memory_barrier()` calls; correctness implications unclear until tested; medium effort.  Lever (1) is the lowest-risk, lowest-effort starting point and closes the gemma 2.16× gap (gemma is dense, no MoE path); lever (3) is the lever for closing the apex 12.38× gap; lever (2) is shared across both.  iter8c CFA spec should target lever (1) as Wave 1, lever (2) as Wave 2, lever (3) as Wave 3 — each wave gates on bench parity + ratio improvement before the next.  Source dive raw notes: per-MoE-layer 12 dispatches × 36 = 432 + attention 36 × 9 = 324 + 5 = ~761/token — the math reconciles with the measured 558/token under sparse expert routing.

- **2026-04-27 — iter8c-confirm APEX result: dispatch-count ratio is **12.38×** on apex (vs 2.16× on gemma) — MoE expert dispatching is the dominant overhead.**  Same metal-system-trace methodology as iter8c-prep, applied to `qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf`.  hf2q `generate --max-tokens 128 --temperature 0` vs llama-bench `-p 0 -n 128 -r 1 --no-warmup`.  Headline:  | metric | hf2q-apex | llama-apex | ratio |  |---|---:|---:|---:|  | dispatch count | 71,467 | 5,771 | **12.38×** |  | dispatches / token | **558** | **45** | 12.38× |  | sum GPU time | 3.21 s | 1.30 s | 2.47× |  | mean / dispatch | 44.9 µs | 224.9 µs | 0.20× |  | p50 / dispatch | 18.4 µs | 10.3 µs | 1.78× |  | p90 / dispatch | 109.7 µs | 35.7 µs | 3.07× |  | max | 11.3 ms | 16.1 ms | 0.70× |  **Findings:** (a) **apex hf2q issues 558 dispatches/token** — enormous; ~15.5 dispatches/layer × 36 layers; this matches the MoE pattern (8 experts × 3 mat-vecs/expert × ~half-active gives ~12 expert dispatches/layer + ~3-4 attention dispatches/layer); (b) **llama-bench issues only 45 dispatches/token = 1.25 dispatches/layer** — ggml's metal backend fuses entire layer (attention + MoE + norms) into 1-2 super-kernels per layer (graph-node fusion), the opposite of hf2q's per-op dispatch granularity; (c) p50 is also 1.78× slower on apex hf2q (vs 1.01× on gemma) — suggesting apex's per-dispatch overhead is amplified by the MoE expert-routing path (small expert mat-vecs × many of them); (d) total GPU dispatch time 2.47× — apex's dispatch-overhead is the worse of the two architectures.  **iter8c scope updates:** primary focus shifts to **apex MoE dispatch consolidation** (the 12× lever) over gemma's (the 2× lever).  This is also where iter8a tried to fix things and failed — iter8a collapsed 102 CB-commit boundaries to 1 CB but kept all 558 dispatchThreadgroups calls per token *and* added 102 explicit memoryBarrierWithScope calls.  **The actual lever is graph-node fusion (kernel fusion), not CB consolidation.**  Specific candidates: (i) batch all active experts in one mul_mm_id dispatch (mlx-native already has mul_mm_id per `project_mm_id_byte_identical`; check if hf2q's apex path is calling per-expert when it could call per-layer); (ii) fuse RMS-norm + scale + mat-mul into single kernels (gemma side); (iii) fuse residual + norm at MoE-block entry/exit.  **Hard prereq before iter8c CFA: identify the literal hf2q call site issuing the per-expert dispatches** vs llama.cpp's batched call — this is a focused source dive (forward_mlx.rs apex MoE path + build_moe_ffn_layer_gpu_q_into helpers) that can be done in 30 min, much cheaper than CFA dual-mode shader work.  Raw artifacts: `/tmp/adr015-iter8c-prep/{hf2q-decode-20260427T173121Z.trace, llama-bench-apex-20260427T173400Z.trace}`.

- **2026-04-27 — iter8c-prep CLOSURE: production-mode trace data FALSIFIES iter8b kernel-gap; gap is dispatch-COUNT (2.16×), not per-kernel speed (1.01× p50).**  Same-day metal-system-trace capture: hf2q `generate --max-tokens 128 --temperature 0` + llama-bench `-p 0 -n 128 -r 1 --no-warmup` on gemma-4-26B-A4B-it-ara-abliterated-dwq.  Aggregator (`scripts/aggregate_decode_mst.py`) parsed the `metal-gpu-execution-points` schema (function=1 start paired with function=2 end by gpu-submission-id) and produced per-dispatch distribution per binary.  **Headline result:**  | metric | hf2q | llama-bench | ratio |  |---|---:|---:|---:|  | dispatch count | 3832 | 1771 | **2.16×** |  | sum GPU time | 1.83 s | 1.27 s | 1.44× |  | mean / dispatch | 477 µs | 716 µs | **0.67×** (hf2q is FASTER per-dispatch on average) |  | p50 / dispatch | **17.2 µs** | **17.0 µs** | **1.01×** |  | p90 / dispatch | 464 µs | 847 µs | 0.55× |  | max | 18.8 ms | 14.7 ms | 1.27× |  **Findings:** (a) per-dispatch p50 is identical → **iter8b's 14-37× per-kernel claim is DEFINITIVELY FALSIFIED**; (b) hf2q issues **2.16× more dispatches** for ~equivalent workload (28.8 dispatches/token vs 13.8 dispatches/token; counts are CB-level events, not per-kernel-dispatch); (c) hf2q's mean per-dispatch (477 µs) is actually SHORTER than llama.cpp's (716 µs) — llama.cpp groups more work into fewer/larger CBs; (d) total GPU dispatch time ratio (1.39×) aligns with the 1.19× wall-time gap (0.840× ratio).  **Implication: iter8c PIVOTS from NAX shader work (P3c.1/P3c.2) to dispatch consolidation.**  iter8a's single-CB attempt failed because it kept 102 explicit barriers within one CB; the actual lever is **2× fewer commit boundaries per token, with proper RAW grouping**, not 1 CB / token.  Concrete next-iter scope: identify which `commit_labeled` sites in hf2q's gemma decode are issuing extra CBs vs llama.cpp's grouping (likely candidates: per-layer norm/proj/mlp commit calls that could group with their feeding ops); merge groups of independent ops without adding explicit barriers; bench-gate.  Asymmetry caveat: hf2q trace included 5 prefill tokens, llama-bench was pure decode — the per-token math (28.8 vs 13.8) accounts for this only approximately.  Hard gate before iter8c CFA work: trace a SECOND baseline pair (N=64 or different prompt) to confirm the 2.16× dispatch ratio is stable across runs and not a capture artifact; same-day apex 35B-A3B trace would also be informative since apex's MoE arch differs from gemma.  Per-kernel attribution would still require enabling Shader Timeline (custom Instruments package); for the dispatch-consolidation lever, per-CB granularity is sufficient.  Memory `feedback_structural_audit_before_kernel_work` is the durable lesson: 30-min audit + 10-min trace falsified an iteration that would have been 1500 LOC of /cfa shader work targeting a phantom gap.  Raw artifacts: `/tmp/adr015-iter8c-prep/{hf2q-decode-20260427T165619Z.trace, llama-bench-decode-20260427T170230Z.trace, aggregate-summary-20260427T170230Z.md}`.

- **2026-04-27 — iter8c-prep continued: metal-system-trace harness landed + schema discovery.**  `scripts/profile-decode-mst.sh` (commit `bc7e598`) captures hf2q + llama-cli decode under `xctrace --template "Metal System Trace"`; cold-SoC + RAM-headroom gates from bench-baseline.sh; same fixture/prompt/n_tokens for both binaries.  Smoke run on hf2q (8 tokens, gemma-26B) succeeded: trace bundle produced, exit 0, hf2q decoded at 102.3 t/s (matching prior measurements — xctrace overhead is acceptable).  **Schema discovery:** `xctrace export --xpath '/trace-toc/run/data/table[@schema="metal-gpu-execution-points"]'` does have rows with (timestamp, channel-id, function-enum, slot-id, gpu-submission-id, accelerator-id, note) — but `note` is empty for all dispatches and `function` is just a 1/2 start/end enum.  **Per-KERNEL attribution requires `metal-shader-profiler-intervals` schema, which is DISABLED in the default "Metal System Trace" template** (Shader Timeline = Disabled per `<intruments-recording-settings>`).  Enabling it requires a custom Instruments package; significant overhead.  **For iter8c-prep verification purpose this is sufficient**: per-CB GPU duration distribution per binary IS captured.  If iter8b's 14-37× claim were real, hf2q's per-CB time distribution would have a tail 30× longer than llama.cpp's; a same-day comparison settles it without needing per-kernel breakdown.  iter8c-prep next sub-step: re-run with N_TOKENS=128 (decode region becomes >50% of trace vs ~2% at N=8), write per-CB aggregator, compare distributions.

- **2026-04-27 — iter8c-prep: structural-identity audit FALSIFIES iter8b kernel-gap premise.**  Side-by-side read of `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` vs `/opt/mlx-native/src/shaders/quantized_matmul_ggml.metal` and `kv_cache_copy.metal` confirms the **decode-path kernels are structurally identical to llama.cpp's reference**:  (a) `kernel_mul_mv_q4_0_f32` mlx-native:99-162 uses `simdgroup_index_in_threadgroup`, `thread_index_in_simdgroup`, N_DST rows per SIMD-group, two-accumulator sumy pattern, `simd_sum` reduction — comment at line 135 explicitly says *"ADR-009 Phase 3A: match llama.cpp's two-accumulator sumy pattern"* and lines up with llama.cpp 3524-3534 which dispatches to the same `mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0, …>` template;  (b) KV-copy is element-per-thread on both sides — llama.cpp's `kernel_cpy_t_t` at 7278-7303 has each thread doing one assignment with a `break` after, identical to mlx-native's `kv_cache_copy` at line 28.  **A 14-37× kernel-level gap structurally cannot exist in production.**  Arithmetic check: iter8b's 4389 µs/token KV-copy figure would be 36% of an 11.4 ms decode wall, but the actual end-to-end gap to llama.cpp is 8.3% (apex 0.917×) and 16% (gemma 0.840×) — the kprofile numbers are inflated by per-dispatch overhead × 242 sessions/token (vs 1 in production).  **P3c.1 (Morton/Z-order) is a prefill GEMM lever** (4K×4K matmuls in `*_tensor.metal`), not a decode mat-vec lever — Apple's 50%→100% NAX-utilization gain is on tile-shaped tensor work, and decode rank-1 mat-vec is bandwidth-bound (weight DRAM bytes / dispatch), not compute-bound, so NAX activation adds nothing where compute isn't the bottleneck.  Per `feedback_never_ship_fallback_without_rootcause` and `project_metal_compiler_auto_optimizes_static_levers` (3 prior static-evidence kernel hypotheses falsified on M5 Max), launching iter8c per the iter8b spec would repeat the iter8a antipattern: a structurally-correct rewrite that fails on bench because the premise was unverified.  **iter8c rescoped from "apply NAX to decode kernels" to "verify gap in production mode before any kernel work".**  iter8c-prep deliverable: same-day llama-cli vs hf2q Metal System Trace (xctrace `metal-system-trace`, NOT `time-profiler` or `HF2Q_MLX_KERNEL_PROFILE`) capturing actual per-dispatch GPU times in production mode (1 session/token), aggregated by kernel name.  If verified gap is bandwidth-shaped (e.g., weight prefetch / split-K), the lever is different from NAX.  If verified gap is dispatch-count-shaped (we issue more dispatches than llama.cpp per token), the lever is dispatch consolidation.  If no verified per-kernel gap remains in production mode, the +8.3% / +16% end-to-end gaps live somewhere else (e.g., scheduler queue depth, async-commit latency, KV-cache layout walks) and iter8c becomes a different investigation entirely.  Reference repos consulted: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` (kernel_mul_mv_q4_0_f32 at 3524, kernel_cpy_t_t at 7278, kernel_mul_mm_id at 9719); `/opt/mlx-native/src/shaders/{quantized_matmul_ggml.metal, kv_cache_copy.metal}`.  Task #12 description updated.

- **2026-04-27 — iter8b first pass: gemma kprofile per-kernel attribution.**  Used the existing `HF2Q_MLX_KERNEL_PROFILE=1 HF2Q_LMHEAD_Q8=0` mode (`forward_decode_kernel_profile` at `forward_mlx.rs:3636`; documented as "intentionally slow — 242 sessions/token vs 1 in production").  2-warmup + 3-measured tokens on `gemma-4-26B-A4B-it-ara-abliterated-dwq` at hf2q@`6db1690`.  Per-session overhead (~30-50 µs × 242) inflates absolute numbers; **per-kernel RATIO vs candle reference is the meaningful signal**.  Top per-ratio gaps:  | Kernel | mlx-native µs | candle µs | ratio | per-token |  |---|---:|---:|---:|---:|  | KV cache copy | 150/layer | 4/layer | **37.6×** | 4389 µs |  | O-proj matmul | 171/layer | 11/layer | **15.6×** | 4802 µs |  | lm_head GEMM | 2741 | 185 | **14.8×** | 2556 µs |  | SDPA | 192/layer | 17/layer | 11.3× | — |  | Fused norms/adds | 164 | 20 | 8.2× | — |  | QKV matmuls | 191 | 37 | 5.2× | — |  | MLP matmuls | 178 | 43 | 4.1× | — |  | **MoE** | 214 | 127 | **1.7×** | — (most competitive!) |  **Findings:** (a) gemma's biggest GPU-side gaps are GEMM-style kernels (KV-copy / O-proj / lm_head) at 14-37× — **all are mat-mul-or-memcpy operations that ADR-015 §"M5 Neural Accelerators" P3c plan targets via NAX routing**; (b) MoE is competitive (1.7×) — reassuring since §P3a''' identified MoE as the apex production hot path; (c) the §P3a'' "decode is GPU-bound" finding now has **specific kernel-level attribution**.  **Implication: P3c re-scope is JUSTIFIED** — it was originally TTFT-only (prefill GEMM), but the iter8b data shows decode GEMM (lm_head) AND mat-vec ops (O-proj on apex MoE production) carry 14-15× per-kernel ratios that NAX routing should close.  Per the iter8a queen verdict followup ("Decision point: if kernel choice is the gap, P3c re-scopes to include decode mat-vec NAX routing"), the data supports re-scope.  iter8c (next): apply P3c.1 (Morton dispatch) + P3c.2 (NAX tile sizes + macOS 26.2 / arch-gen ≥ 17 gate) to the decode-path GEMM kernels, validate via parity + apex bench.  Caveat: the candle-reference numbers are mlx-native's internal comparison baseline; same-day llama.cpp `ggml-metal` per-kernel comparison would be more authoritative.  Raw kprofile output at `/tmp/adr015-iter8b/gemma-kprofile-20260427T154341Z.txt`.
- **2026-04-27 — iter8a REJECTED (single-CB hypothesis FALSIFIED on apex bench).**  CFA session `cfa-20260427-adr015-iter8a` (dual-mode: claude + codex teams, queen-judged) produced bit-exact-correct `forward_gpu_single_cb` implementations on both branches.  Claude side: 102 `enc.memory_barrier()` placements at all §P1-audit fuse_safe=YES sites + 1 terminal `commit_and_wait_labeled("decode-token")` + 4 parity tests (`single_cb_parity_smoke_16` worst_delta=**0.0**, `_cmd_buf_count_le_2`, `_no_new_unsafe`, `_kv_cache_thread_through`) all PASS, 218/218 qwen35 unit tests pass, build clean.  Codex side: independent 4-finer-split decomposition of `_into` variants, build clean (sandbox couldn't run Metal tests).  **But apex 3-cold-trial bench REGRESSED**: hf2q `[93.1, 101.9, 101.5]` median 101.5 vs llama-bench `[118.30, 118.12, 117.24]` median 118.12 = **0.8593×** (vs post-iter7 0.9174× = **−8.6 pp regression**).  Hypothesis: 102 explicit `memoryBarrierWithScope:Buffers` calls inside one encoder cost MORE than the implicit per-CB barriers they replaced; the §P2-predicted ~125 µs/token CPU encode savings were async-overlapped per the iter7 finding (CPU savings on an 81%-GPU-bound workload don't translate to wall).  Phase 3 queen verdict: **REJECT_BOTH** (claude weighted 76.4 / codex weighted 56.5; AC4 perf criterion failed for both); main stays at `addb68c`.  Salvage to main: the two architects' design docs only (`docs/iter8a-barrier-graph.md` and `docs/iter8a-barrier-graph-codex.md` — landed in commit `55bd3a8`).  The `forward_gpu_single_cb` impl + env-gate + 4 `_into` variants + 4 parity tests stay on branches `cfa/cfa-20260427-adr015-iter8a/{claude,codex}` (archived 7 days under `cfa/archive/`) for re-cherry-pick if iter8c ever resumes.  Per `feedback_never_ship_fallback_without_rootcause` and the mantra ("no shortcuts, no fallbacks, no stubs, just pure excellence") — shipping a known regression is the antipattern.  Followups: **iter8b** = pivot to gemma-side / GPU-side levers (Metal System Trace + per-kernel comparison vs `ggml-metal` Q4_K mat-vec, per the §P3a'' / §P3a''' analyses identifying GPU compute as the actual residual lever space); **iter8c** (optional) = re-run iter8a under `MLX_PROFILE_BARRIERS=1` (mlx-native@`19f5569` BARRIER_COUNT/BARRIER_NS atomics) to quantify the explicit-barrier vs CB-churn delta — only if iter8b doesn't fully close the gap; **P8 legacy delete** = deferred indefinitely; HF2Q_LEGACY_PER_LAYER_CB env-gate machinery stays available for future single-CB attempts.  Raw artifacts: `/tmp/adr015-bench/baseline-apex-iter8a-claude-20260427T143903Z.*` (bench), `/tmp/cfa-cfa-20260427-adr015-iter8a/{spec.json, codex-{architect,coder,tester}.{patch,jsonl}, *-result.json}` (CFA workers), hive memory `swarm-cfa-20260427-adr015-iter8a/{judgment, final-report}` (queen).
- **2026-04-27 — apex post-iter7 ratio measured (iter7 bench).**  3-cold-trial bench at hf2q@`f1ae8dc` / mlx-native@`19f5569` (post-iter7a apply_imrope + iter7b proj_pooled migrations) against same-day llama-bench: hf2q **110.1 t/s** (per-trial 110.1 / 99.7 / 110.8 — **trial 2 is a 10% outlier**, parallel ADR-014 codex worktree was running concurrent compile/test work) vs llama-bench **120.01 t/s** (per-trial 119.32 / 120.01 / 120.27) = **0.9174×**.  Compared to post-iter1 baseline (0.9266× at 110.0 vs 118.71): hf2q absolute moved 110.0 → 110.1 (median; trials 1 + 3 are 110.1 / 110.8 vs post-iter1's 110.0 / 110.2, so roughly +0.1 to +0.6 t/s — mostly within noise after accounting for trial-2 outlier).  llama-bench drifted +1.3 t/s same-day (118.71 → 120.01).  **Net: iter7 pool migration is bit-exact verified but on-wall impact <0.5% — substantially less than the ~415 µs/token of CPU savings predicted from §P3a''' H6 line.**  Honest verdict: at 81% GPU-bound, async-pipelined CPU encoding overlaps with GPU execution; saving CPU work that was already overlapped doesn't translate to wall.  iter7 closes the §P3a''' rank-1 lever as a CPU optimization but not as a wall-time win on this workload.  **CPU-only levers cannot close the +9% gap to D4 first bullet.**  Pivot required: GPU-side work (Metal System Trace + kernel comparison vs `ggml-metal`) or qwen35 P3 single-CB (which removes encoder lifetimes + barrier bookkeeping — different cost class than alloc).  Raw artifacts at `/tmp/adr015-bench/baseline-apex-35b-a3b-dwq46-post-iter7-20260427T125020Z.*`.
- **2026-04-27 — apex post-iter1 ratio measured (iter6 bench).**  3-cold-trial bench at hf2q@`4214f7c` / mlx-native@`19f5569` against same-day llama-bench on `qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46` n=256 cold-SoC: hf2q **110.0 t/s** (per-trial 110.0 / 109.3 / 110.2; σ ≈ 0.4) vs llama-bench **118.71 t/s** (per-trial 118.71 / 117.90 / 119.20) = **0.9266×**.  Recovery target: **+7.9%**; decode-wall gap today = **667 µs/token**.  Compared to ADR-012 closure (0.940× at hf2q 109.1 / llama 116.0): hf2q absolute +0.9 t/s (iter1 contributed positively per H2 closing on apex), llama-bench absolute +2.7 t/s same-day drift, ratio slipped 0.94 → 0.9266.  Per `project_end_gate_reality_check` this is the apples-to-apples bar — same-day comparison is authoritative; the ADR-012 closure number is no longer the operative reference.  D4 first-bullet placeholder updated.  Raw artifacts at `/tmp/adr015-bench/baseline-apex-35b-a3b-dwq46-post-iter1-20260427T123407Z.*`.
- **2026-04-27 — §P3a''' apex MoE live profile (iter6) + Wave 2b hard gate #1 CLOSED.**  5-cold-trial xctrace TimeProfiler against `qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46` post-iter1.  Decode-filtered aggregation via `aggregate_decode.py --decode-filter forward_gpu_greedy --hypothesis-config aggregate_hypotheses.json`.  **H1 literal `fn proj` site CONFIRMED at 375 µs/token on apex** (dense fixture had it as NOT FOUND); subcomponent `build_moe_ffn_layer_gpu_q` 578 µs/token.  **H6 alloc_buffer = 344 µs/token on apex** (vs §P3a' dense 3063 — different call pattern; MoE expert dispatch reuses more buffers per layer).  **H2 apply_imrope = 63 µs/token** (was 464 µs/token Wave 2a dense; iter1 cache effective on apex too — `build_rope_multi_buffers` subcomponent = 0 µs/token).  **H3 command_encoder = 125 µs/token** (lower than dense's 724; closes under P3).  H4 / H5 same FALSIFIED / sub-1ms verdicts.  Median per-trial CPU wall: 1766 µs/token on apex (19% of 9.16 ms decode wall); CPU-side levers bounded above by 1766 µs/token recovery.  Wave 2b hard gates #1 (a) literal H1 sized + (b) H6 sized for apex: ✅.  iter7 alloc_buffer pool plan refined: target list = decode-hot direct `device.alloc_buffer` sites in qwen35 (apply_imrope output, forward_gpu projections + argmax + positions); existing `decode_pool::pooled_alloc_buffer` infrastructure handles bucketing + reset.  Raw artifacts at `/tmp/adr015-apex-p3a/` (5 traces + 5 topcalls XML + aggregate-decode.{md,json}).  Apex bench post-iter1 runs concurrently to capture the live ratio against llama.cpp.
- **2026-04-27 — §P3a'' gemma live profile pass (iter5).**  5-cold-trial xctrace TimeProfiler capture aggregated via `aggregate_decode.py --decode-filter forward_decode`.  Headline: **gemma decode is GPU-BOUND, not CPU-bound.**  Median per-trial CPU wall = 656 µs/token (5.7% of 11.4 ms decode wall); GPU = 94.3%.  G5 alloc_buffer = **0 µs/token** on gemma (qwen35 §P3a' rank-1 lever does NOT exist here).  G6 GraphSession bookkeeping = 219 µs/token (largest CPU contributor; bounded by 656 µs/token total).  Pre-iter5 "shared P3b lever" hypothesis FALSIFIED for gemma; the 19% gap to llama.cpp lives in GPU compute.  Aggregator gained `--decode-filter <regex>` option to scope per-token aggregation (mirrors §P3a' on `forward_gpu_greedy`).  New Wave 2c hard gates introduced: Metal System Trace + kernel comparison vs llama.cpp; P3c scope may need to extend to decode mat-vec NAX routing if data supports.  iter6 / iter7 plans rescoped: alloc_buffer pool is qwen35-only; gemma's lever is GPU-side.  Raw artifacts at `/tmp/adr015-gemma-p3aprime/` (traces + topcalls XML + aggregate.md).
- **2026-04-27 — P0 gemma baseline measured (iter4).**  hf2q `35fdcc8` / mlx-native `19f5569` measured against same-day llama-bench on `gemma-4-26B-A4B-it-ara-abliterated-dwq` n=256 cold-SoC: hf2q **87.8 t/s** (per-trial 87.7 / 87.8 / 88.0; σ ≈ 0.15) vs llama-bench **104.52 t/s** (per-trial 103.96 / 104.58 / 104.52) = **0.840×**, recovery required **+19.0%**.  Fills D4 second-bullet placeholder + closes Phasing P0 row.  Bench harness `scripts/bench-baseline.sh` lands alongside (RAM-headroom precheck + thermal-gate + JSON metadata + tee summary).  Two harness bugs caught + fixed in the follow-up commit (`run_*_trial` progress echo bled into captured stdout; llama-bench awk grabbed std-dev not median).  New diagnosis subsection ("2026-04-27 — P0 Gemma baseline (iter4) reshapes the phasing budget") captures the implications: gemma gap is structurally larger than qwen35 ADR-012 baseline (0.840× vs 0.94×) AND lives entirely outside CB-count (gemma is already at 1-2 CBs/token), so D1's gemma half is refactor-for-uniformity not perf lever; the lever is shared P3b orchestration, particularly rank-1 alloc_buffer pool which can apply to both families.  Adds a Track B for iter5/iter6: port the iter1 rope_multi cache pattern to gemma's `forward_mlx.rs` rope call sites.
- **2026-04-27 — Wave 2b iter3 landed (aggregate_decode.py rewrite + 5-trial median harness).**  hf2q @ `bcd08dd` lands `scripts/aggregate_decode.py` + `scripts/aggregate_hypotheses.json` + `scripts/profile-p3aprime.sh` (default `N_TRIALS=5`).  Replaces the volatile `/tmp/cfa-adr015-wave2a-p3a-prime/aggregate_decode.py` working-copy that had AF3 (overlapping-inclusive-frame double-count) + AF4 (rank-stability split-on-tuple) bugs and used 3-trial mean instead of 5-trial median.  The new aggregator: (a) reports ONE canonical frame per hypothesis via the regex in `scripts/aggregate_hypotheses.json`, (b) lists named subcomponents side-by-side without summing them into the primary, (c) keeps frame names as `str` end-to-end with a typed dataclass for trial state, (d) defaults to 5-trial median (outlier-absorbing without silent discard).  Hypothesis register includes the new H6-Wave2b-AllocBuffer entry pointing at `MlxDevice::alloc_buffer::` for the rank-1 P3b lever.  H4's regex now targets `issue_metal_buffer_barrier` (the new `#[inline(never)]` frame from iter2) so TimeProfiler can attribute it; the `BARRIER_COUNT`/`BARRIER_NS` counters from iter2 provide the authoritative number.  Smoke-tested on synthetic xctrace XML — correct per-trial totals, median, subcomponent isolation, rank-stability table.  Closes Wave 2b hard gates #3, #4, #5.  Remaining Wave 2b hard gates: #1 (apex 35B-A3B MoE 5-cold-trial × 64-decode-token re-measurement) and #2 (already closed by iter2 BARRIER counters).
- **2026-04-27 — Wave 2b iter2 landed (H4 counter-based barrier accounting).**  mlx-native @ `19f5569` adds atomic `BARRIER_COUNT` (always tracked) + `BARRIER_NS` (env-gated under `MLX_PROFILE_BARRIERS=1`) around the `memoryBarrierWithScope:` `objc::msg_send!` site at `/opt/mlx-native/src/encoder.rs:498-512`.  The objc dispatch is moved into a `#[inline(never)]` helper (`issue_metal_buffer_barrier`) so xctrace / Instruments has a stable Rust frame to attribute barrier time against, rather than being inlined / hidden under sibling Objective-C frames as it was at 1 ms TimeProfiler resolution in Wave 2a.  Public API: `mlx_native::barrier_count() -> u64`, `mlx_native::barrier_total_ns() -> u64`, `reset_counters()` extended.  Hot-path cost: 1 atomic fetch_add (~5 ns) + 1 OnceLock load when env-disabled (default); 2 × `Instant::now()` (~100-200 ns) when `MLX_PROFILE_BARRIERS=1` (opt-in only — adds ~22-44 µs/token at 440 barriers/token, comparable to what is being measured).  Tests in `tests/test_barrier_counter.rs` cover: pre-dispatch no-op skip, capture-mode skip, +3-after-3-barriers post-active-dispatch, ns-stays-0 with profile disabled.  Closes Wave 2b hard gate #2 (the H4 OPEN status from §"P3a' live profile pass" hypothesis register).
- **2026-04-27 — P3b iter1 landed (rank-4 rope_multi hoist).**  mlx-native @ `a50c224` adds `dispatch_rope_multi_cached` (per-thread cache keyed by device + rope config) + bit-exact parity tests (`test_rope_multi_cached_matches_uncached_qwen35_decode_shape`, `test_rope_multi_cached_seq_len_variation`).  hf2q @ `74c28b9` switches `apply_imrope` (`gpu_full_attn.rs:361-417`) to the cached path; `mtp.rs` call sites pick up the change automatically (no signature change).  Closes the rank-4 lever from §"P3a' live profile pass" (208 µs/token measured on qwen3.6-27b-dwq46 dense fixture; 32 calls/token × 3 fresh `MlxBuffer` allocs/call dominated by mach_msg2_trap / IOGPUResourceCreate).  Steady-state qwen35 decode hot path now hits 2 stable cache entries (Q-config + K-config, seq_len=1) and amortizes the alloc cost across all decode tokens.  Bit-exact verified end-to-end via `qwen35::gpu_full_attn::tests::imrope_matches_cpu_ref` and `qwen35::gpu_full_attn::tests::full_layer_gpu_matches_cpu_ref` (full FullAttn layer including both apply_imrope calls); all tests use `to_bits()` equality.  Apex-MoE 35B-A3B re-measurement (Wave 2b hard gate) still required before ADR-012-baseline µs credit can be claimed against the 288 µs residual; this iter contributes a measured-on-dense lever, not a credited apex µs.
- **2026-04-26 — §"Capturing M5's GPU Neural Accelerators (Metal 4 TensorOps + NAX)" infused.**  Per ADR-016 research dossier graduation: Apple's measured 3.33–4.06× TTFT vs M4 (1.19–1.27× decode bandwidth-bound) lives in per-GPU-core Neural Accelerators via Metal 4 TensorOps, **not** standalone ANE.  mlx-native already partially routes through `mpp::tensor_ops::matmul2d` in four `*_tensor.metal` kernels but is missing: (a) Morton/Z-order dispatch for large prefill GEMM (Tech Talk 111432: ~50 → ~100 % NAX utilization on 4K×4K matmul); (b) NAX-tuned outer tile sizes + macOS 26.2 / arch-gen ≥ 17 runtime gate (mirroring MLX's `is_nax_available()`); (c) `flash_attn_prefill` inner matmul still on M1-era `simdgroup_matrix` / `simdgroup_multiply_accumulate` — biggest unrouted lever (16 % of prefill compute).  Added new sub-phase **P3c — M5 Neural Accelerator prefill kernels** with three actions; P3c is shader-internal only and **fully orthogonal** to P3 (single-CB) and P3b (orchestration sweep) — opens only after P6 bench gate clears.  All API claims verified against Apple developer docs, MLX commits (`54f1cc6` 2025-11-19 NAX support, `b41b349` 2026-03-18 NAX refactor, `0879a6a` 2026-03-11 M5 tuning), and `/opt/mlx-native` source by direct file:line read — no training-data recall used.  Source brief: `/tmp/cfa-adr016/research-metal4-tensorops.md` + memory key `swarm-cfa-adr016/deep-research/metal4-tensorops`.
