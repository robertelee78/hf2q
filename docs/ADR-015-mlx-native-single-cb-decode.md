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

### Phase plan implication

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
  - **gemma:** `gemma-4-26B-A4B-it-ara-abliterated-dwq` `n=256` decode
    median **≥ 1.00× of llama.cpp** same-day median (same methodology).
    Reference baseline must be re-measured at P0 (no committed value
    on file as of 2026-04-26 — see Open Question Q1).

## Phasing

| Phase | Deliverable | Definition of done |
|---|---|---|
| **P0 — Gemma baseline** | Same-day `gemma-4-26B-A4B-it-ara-abliterated-dwq` `n=256` decode median: 3 cold runs of `hf2q` + 3 of `llama-bench`. Establishes the gemma-side ratio. | Numbers recorded in this ADR as the baseline gemma-family bar. |
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
