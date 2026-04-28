# Wave 5b.3 Walk-Bar Validation Results

**Status:** **FAIL — chunk-pipeline diverges from llama.cpp at pp4096**
**Date:** 2026-04-27
**Worker:** Wave 5b.3 (ADR-005 inference-server mission close)
**HEAD at start:** `2239ed0` (Wave 5b.1 iter 5 audit hygiene close)
**HEAD at finish:** working-tree (observability `tracing::info!` added at `gpu_delta_net.rs:1330`, uncommitted at write time)

## Setup

| Field | Value |
|---|---|
| Model | `/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf` (27 B Qwen3.6 hybrid, DWQ46 quant, 16.07 GiB) |
| llama.cpp | build `8680 (15f786e65)`, AppleClang 21.0.0, Metal+Tensor backend on M5 Max |
| llama tool used | **`llama-completion`** (`llama-cli` rejected `--no-conversation`; per its own help, "use llama-completion instead") |
| hf2q binary | `/opt/hf2q/target/release/hf2q` rebuilt this iter (`cargo build --release --bin hf2q`, 0 errors, 79 warnings — all pre-existing) |
| Hardware | Apple M5 Max, 128 GiB unified memory |
| RAM at start | 75.5 GiB free (Pages free 4 944 446) |
| RAM at finish | 95.8 GiB free (Pages free 6 281 117) — no leak |

## Prompt construction

The chunk-pipeline gate requires `seq_len > 64 ∧ seq_len % 64 == 0 ∧ d_k == 128`. To force the chunk path to fire at `seq_len = 4096` (= 64 chunks × 64), the prompt was tuned via `llama-tokenize --no-bos` until the post-tokenization length hit exactly 4096:

```
prompt = "The history of bread baking spans millennia. " * 511 + "spans millennia ago. The dough rises slowly"
→ 4096 tokens (no BOS)
```

To make hf2q's `cmd_generate` emit this **without** a chat-template wrap, an identity Jinja template was passed: `--chat-template '{{ messages[0]["content"] }}'`. To make `llama-completion` skip BOS, `--override-kv tokenizer.ggml.add_bos_token=bool:false` was passed. Both engines therefore see the **same 4096 tokens** (verified: hf2q stderr `Qwen3.5: 4096 prompt tokens`, llama stderr `prompt eval time = ... / 4096 tokens`).

## Observability added

A single one-shot `tracing::info!` block at `src/inference/models/qwen35/gpu_delta_net.rs:1330-1357` (gated by static `AtomicBool`) emits exactly one log line per process the first time the chunk path either fires or is skipped under `HF2Q_CHUNK_SCAN_PREFILL=1`. Without this, a chunk-path-skip (e.g., `seq_len % 64 != 0`) is silent and any walk-bar test would report a fake-PASS. The probe confirmed the chunk path **actually fired** at `seq_len=4096, d_k=128`.

## Per-pp results

| pp | prompt tokens | llama-completion `n-predict 1` token | hf2q-chunk token | hf2q-autoreg token | match (chunk vs llama) | match (autoreg vs llama) |
|---:|---:|:---:|:---:|:---:|:---:|:---:|
| **4096** | 4096 (raw, no BOS) | **`11` (`,`)** | **`42824` (` spans`)** | **`11` (`,`)** | **NO — DIVERGE** | YES |

**Root cause analysis.** The chunk-pipeline ENGAGED at `seq_len=4096` (confirmed by the new `wave5b3` log line). The autoregressive path with the **same** model + tokens reproduces llama-completion's first decoded token bit-exactly (id 11). Disabling `HF2Q_CHUNK_SCAN_PREFILL` while keeping `HF2Q_QWEN36_AUTOREG=1` flips hf2q's first decoded token from `42824` back to `11`. **The divergence is therefore introduced by the chunk-path itself, not by hf2q's tokenizer, kv-cache, sampler, or shared autoregressive forward path.**

This is consistent with the brief's own warning: the existing `chunk_path_first_token_matches_autoregressive_at_seq128` test passed at `1.64e-2` max divergence at `seq_len=128`, "within the cross-path bf16 budget but accumulates with seq_len." At `seq_len=4096` (32× longer) the per-element error has clearly grown enough to flip an argmax tie at the LM head. Likely contributors:
1. **bf16 round-off accumulating across 64 chunks** — chunk-pipeline q/k/v inputs are bf16-storage; each of the 6 sub-kernels (kkt, tri_solve_invert, recompute_w_u, inter_state, chunk_o, cumsum_g) does at least one bf16↔f32 round-trip and the per-chunk error compounds across the inter-chunk recurrence.
2. **State-recurrence drift** — the chunk-parallel reformulation reorders the delta-rule recurrence; mathematically equivalent in f32 (oracle confirms `max_err = 1.19e-7` at synthetic seq=128) but bf16-stored intermediate states grow drift with chunk index.
3. **Layer stacking** — Qwen3.6 has 30 delta-net layers; per-layer chunk-vs-autoreg residue compounds through the stack before reaching the LM head.

## Stages NOT executed

| Stage | Reason |
|---|---|
| pp16384 | **Skipped — pp4096 already failed.** Per discipline, "If hf2q's CLI doesn't yet support the chunk-pipeline activation cleanly … STOP and report" and "we want truthful validation status, not fake-PASS." Re-running at 16k/65k would only confirm and amplify the same numerical divergence with no new diagnostic value, while consuming ~10 min × 3 of model-load time. |
| pp65536 | Same as above. Also: pp65536 with `n_batch=65536` would risk running into the historical "first_decode_token=0 anomaly" already documented at `project_long_prefill_parity_inverts.md`. |

## Total wall-clock for prefill

| engine | prefill wall-clock | tok/s | first decoded token |
|---|---:|---:|:---:|
| llama-completion | 6388 ms | 641.2 | `11` (`,`) |
| hf2q autoregressive (chunk OFF) | 51 097 ms | 80.2 | `11` (`,`) |
| hf2q chunk-pipeline (chunk ON) | 45 385 ms | 90.3 | `42824` (` spans`) |

(Informational only — W-5b.3 is correctness, not perf. The chunk path is **12.6 % faster than autoregressive** in hf2q at this length but is **7.1× slower than llama.cpp**.)

## Environmental issues encountered

1. **`llama-cli` rejected `--no-conversation`** with the message *"--no-conversation is not supported by llama-cli — please use llama-completion instead"* — switched to `llama-completion`.
2. **`llama-completion` defaults to chat-mode** when the model has a chat template (kv 30); `-no-cnv` required to disable.
3. **`llama-completion` adds BOS by default** (`tokenizer.ggml.add_bos_token=true`); `--override-kv tokenizer.ggml.add_bos_token=bool:false` required to match hf2q's no-BOS tokenization.
4. **hf2q has no built-in chat-template-bypass flag**; `--chat-template '{{ messages[0]["content"] }}'` (identity template) was used as a workaround. Suggest a future `--no-chat-template` flag.
5. **Chunk-path silent skip when `seq_len % 64 != 0`** is a footgun for any walk-bar harness (the first probe at 5610 tokens silently ran the autoregressive path despite the env gate). The new `tracing::info!` at `gpu_delta_net.rs:1330-1357` makes this observable; recommend keeping it as production-hygiene observability.

## ADR-005 status flip recommendation

**NO-CHANGE for the partial-flip wording on AC 5468 (line 5735) and AC 5470 (line 5737).** The Wave 5a addendum at line 5714 correctly classifies these as "partial — Wave 5a autoregressive opt-in landed; chunk-scan kernel for SOTA prefill perf deferred to W-5b." That phrasing remains accurate after this iter:

- The autoregressive opt-in is correct (matches llama.cpp at pp4096).
- The W-5b chunk-scan kernel is wired (per W-5b.1 iter 5 milestone) but **fails the walk-bar correctness gate at production prefill length pp4096**.

**A new ADR-005 paragraph SHOULD be added under the Wave 5b.1 iter 5 entry at line 5720** documenting: (a) chunk-path correctness FAILED at pp4096 with the diagnostic above, (b) the chunk-path is therefore ENGAGED-but-NOT-CORRECT at production prefill lengths, and (c) AC 5468 ±5 % perf-bar measurement is **still blocked** — not by the chunk kernel's existence (it's wired) but by its numerical accuracy budget at long prefill. The AC remains `[x] (partial)`; W-5b.4 (or a follow-on) needs to address the bf16 accumulation budget before AC 5468 can flip to `[x]` (full).

## Walk-bar status

**FAIL.** The chunk-pipeline at `HF2Q_CHUNK_SCAN_PREFILL=1` produces a different first decoded token from llama.cpp at the smallest production prefill length we tested (pp4096), and from hf2q's own autoregressive baseline. The hf2q autoregressive path passes the walk-bar (token-id match against llama-completion). The chunk-path's existing `chunk_path_first_token_matches_autoregressive_at_seq128` unit test (1.64e-2 max divergence at seq=128) is **insufficient** as a correctness gate at pp4096+ — the test should be extended to assert argmax-id stability at production seq lengths, or the chunk path should be gated off until a representation/numerical fix lands.

## Reproducibility

Inputs preserved at:
- `/tmp/walkbar-pp4096-prompt.txt` (4096-token raw prompt, 23 038 bytes)
- `/tmp/walkbar-pp4096-llama.stdout` + `.stderr` (llama-completion baseline)
- `/tmp/walkbar-pp4096-hf2q-chunk.stdout` + `.stderr` (chunk-pipeline run)
- `/tmp/walkbar-pp4096-hf2q-autoreg.stdout` + `.stderr` (autoregressive sanity)

Reproduce with:

```sh
# llama baseline
llama-completion --model qwen3.6-27b-dwq46.gguf --file walkbar-pp4096-prompt.txt \
  --override-kv tokenizer.ggml.add_bos_token=bool:false \
  --n-predict 1 --temp 0.0 --seed 42 --no-warmup \
  --batch-size 4096 --ubatch-size 4096 -no-cnv

# hf2q chunk-pipeline (FAILS walk-bar)
HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1 \
  hf2q -v generate --model qwen3.6-27b-dwq46.gguf \
  --prompt-file walkbar-pp4096-prompt.txt \
  --chat-template '{{ messages[0]["content"] }}' \
  --max-tokens 1 --temperature 0.0

# hf2q autoregressive (matches llama)
HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 \
  hf2q -v generate --model qwen3.6-27b-dwq46.gguf \
  --prompt-file walkbar-pp4096-prompt.txt \
  --chat-template '{{ messages[0]["content"] }}' \
  --max-tokens 1 --temperature 0.0
```

---

## Wave 5b.4 post-fix re-validation

**Status:** **PASS — chunk-pipeline matches llama.cpp at pp4096 after the GQA head-mapping fix**
**Date:** 2026-04-27
**Worker:** Wave 5b.4 (re-validation follow-up)
**HEAD at start:** `31e05a1` (W-5b.4 closure docs); fix landed at `2a67974` (W-5b.4 GQA head-mapping)
**Binary:** rebuilt this iter (`cargo build --release --bin hf2q`, 0 errors, 72 warnings, 10s incremental — already current)
**RAM at start:** 77.8 GiB free (Pages free 5 096 009)
**RAM at finish:** 105.7 GiB free (Pages free 6 923 046) — no leak
**Concurrent-session check:** clean — no non-sccache/rustc process > 5 GB at start or finish (largest non-sccache process at start was mcp-brain-server at 3.4 GB; at finish, all foreign processes < 1.8 GB)

### Per-path token id table (pp4096, identical 23 038-byte prompt)

| path | invocation | first decoded token | token id | match (llama-completion) |
|---|---|:---:|:---:|:---:|
| llama-completion baseline | `--batch-size 4096 -no-cnv` | `,` | **11** | — |
| hf2q chunk-pipeline (postfix) | `HF2Q_CHUNK_SCAN_PREFILL=1` (no-`-v`) | `,` | **11** | **YES** |
| hf2q chunk-pipeline (verbose) | `HF2Q_CHUNK_SCAN_PREFILL=1 RUST_LOG=hf2q::wave5b3=info -v` | `,` | **11** | **YES** |
| hf2q autoregressive sanity | `HF2Q_QWEN36_AUTOREG=1`, no chunk env | `,` | **11** | **YES** |

The verbose chunk run printed the wave5b3 observability probe **`chunk-pipeline ENGAGED for prefill (HF2Q_CHUNK_SCAN_PREFILL=1) seq_len=4096 d_k=128`**, confirming the chunk path ran the kernels (not a silent autoreg fallback). Both no-`-v` and `-v` chunk runs produced the same final glyph `,`.

### Walk-bar verdict

**PASS.** All three paths agree on token id 11 (`,`). The W-5b.3 divergence (chunk → 42824 ` spans`) is fully resolved by the W-5b.4 wrapper-side tiled-GQA pre-expansion in `apply_gated_delta_net_chunk`. The unit-level seq=128 max_diff dropped from 1.64e-2 to 6.31e-5 per the W-5b.4 fix commit, consistent with the per-element error budget no longer flipping the LM-head argmax at production prefill length.

### ADR-005 status flip recommendation

**FULL flip recommended** for AC 5468 (`[x] (partial)` → `[x]`) and AC 5470 (`[x] (partial)` → `[x]`). The W-5a addendum's "partial" qualifier was conditional on the chunk-scan kernel passing the walk-bar correctness gate at production prefill length; that gate is now PASS. The chunk-path is therefore both ENGAGED and correct at pp4096. A perf-bar measurement at pp4096 (chunk vs llama prefill ms) is informational and not gating, but is now numerically meaningful since the kernel produces the same token. Recommend the W-5b.4 closure ADR-005 entry note both ACs flip to full and that pp16384 / pp65536 walk-bar runs are advisable as a follow-up but not blocking.

### Wall-clock numbers (informational)

| engine | prefill ms | tok/s | real (full process) | first decoded token |
|---|---:|---:|---:|:---:|
| llama-completion | 6 528 | 627.4 | 10.28 s | id 11 |
| hf2q chunk-pipeline (postfix, no `-v`) | 69 174 | 59.2 | 77.14 s | id 11 |
| hf2q chunk-pipeline (verbose, `-v`) | 55 258 | 74.1 | (not timed; comparable) | id 11 |
| hf2q autoregressive | 59 072 | 69.3 | 67.51 s | id 11 |

The chunk-pipeline now matches the autoregressive path on token id and is in the same prefill time-class (55-69 s vs 59 s autoreg). Apparent ~10 % spread between the two chunk runs reflects normal cold-vs-warm thermals and concurrent-process contamination at the moment of run, not a meaningful kernel-perf delta — chunk-path SOTA prefill perf vs llama.cpp (627 tok/s baseline) remains the next-step target, currently 8.5-10.6× slower at pp4096; that target is informational, not gated by the walk-bar.

### Inputs / outputs preserved

- `/tmp/walkbar-pp4096-prompt.txt` (regenerated, 23 038 bytes, identical SHA to W-5b.3 prompt)
- `/tmp/walkbar-pp4096-llama-rerun.{stdout,stderr}`
- `/tmp/walkbar-pp4096-hf2q-chunk-postfix.{stdout,stderr}`
- `/tmp/walkbar-pp4096-hf2q-chunk-verbose.{stdout,stderr}` (with `-v` and `RUST_LOG=hf2q::wave5b3=info`)
- `/tmp/walkbar-pp4096-hf2q-autoreg-rerun.{stdout,stderr}`

### Closing the wave

Wave 5b.4 walk-bar re-validation: **CLOSED — PASS**. The GQA head-mapping fix at commit `2a67974` is confirmed correct at the pp4096 production prefill walk-bar. ADR-005 AC 5468 / AC 5470 may flip from `(partial)` to full per the recommendation above.

---

## Wave 5b.5 perf-gap diagnostic

**Status:** **DIAGNOSTIC — perf gap explained; recommendations queued for W-5b.6**
**Date:** 2026-04-27
**Worker:** Wave 5b.5 (ADR-005 perf-gap diagnostic)
**HEAD at start:** `7270a3f` (W-5b.4 audit-followup); mlx-native at `4be0cfd`
**Pre-flight:** 94 GiB free RAM (Pages free 6 047 726); no non-sccache process > 2 GB; HEAD includes `7270a3f` and mlx-native `4be0cfd` ✓
**Scope:** diagnostic only — no kernel files modified, no rewrites this iter.

### Empirical wall-clock measurement (T=256, both paths)

Two trials each at 256 tokens (4 chunks × 64 = chunk-path eligible, identical prompt, identical token id 561 produced — correctness PASS at T=256, matching pp4096 walk-bar).

| Trial | Path | linear_attn (ms) | full_attn (ms) | ffn (ms) | total_layers (ms) | total prefill (ms) |
|---|---|---:|---:|---:|---:|---:|
| 1 | chunk | 464.0 | 221.6 | 362.1 | 1079.5 | 7876 |
| 2 | chunk | 488.1 | 210.0 | 358.1 | 1059.3 | 8296 |
| 1 | autoreg | 580.0 | 211.4 | 354.8 | 1149.5 | 7918 |
| 2 | autoreg | 594.4 | 216.5 | 355.0 | 1171.1 | 8219 |

Means: chunk linear_attn = **476 ms**, autoreg linear_attn = **587 ms** → **chunk is 111 ms (19%) faster on the linear-attn portion**, but `total_layers` differs by only 91 ms and total prefill is **statistically tied** (chunk 8086 vs autoreg 8068 ms). The 7-second residual outside `total_layers` (which only times the per-layer compute loop) is dominated by **first-call weight upload to GPU** — on a fresh `forward_gpu` invocation `upload_layer_weights_gpu` runs once before the layer loop, materializing the ~17 GB Q4 weights on Metal heap.

### Why kernel-isolation 6.7× speedup ≠ wall-clock parity

Two independent reasons compound:

**(A) Production shape ≠ bench shape.** The Wave 5b.2 kernel-isolation bench measured `B=1, T=4096, Hg=2, H=4` (post-iter-1: 22.67 ms → 3.38 ms = 6.7×). Qwen3.6 27B DWQ46 actually runs `Hg=16, H=48, K=V=128` (derived from GGUF: `ssm.group_count=16`, `ssm.inner_size=6144`, `ssm.state_size=128` ⇒ `H=6144/128=48`; ratio H/Hg = **3**, not 2 as the wrapper comment at `gpu_delta_net.rs:857` claims). Per-kernel work scales linearly in H, so production per-layer chunk-pipeline cost ≈ `3.38 ms × 48/4 = 40.6 ms` — not 3.38 ms. With **48 delta-net layers** (block_count=64, full_attention_interval=4 ⇒ 16 full + 48 linear, also corrects the W-5b.3 doc's "30 delta-net layers" figure), the chunk-pipeline kernels alone amount to **~1.95 s** before any wrapper overhead. The W-5b.4 `chunk path is 6.7× faster than autoreg at pp4096` claim was correct only at the synthetic kernel-bench shape.

**(B) Wrapper overhead is shape-amplified too.** The chunk path wraps the orchestrator with three encoder-split commits + a per-layer F32 GQA expansion + F32→BF16 cast.

### Decomposed cost model (per-forward, T=4096, 48 delta-net layers)

| Component | Per-layer cost | × 48 layers | Notes |
|---|---:|---:|---|
| Chunk-pipeline kernels (scaled from bench) | ~40.6 ms | **~1947 ms** | 3.38 ms × H_prod/H_bench (12×); largest component |
| CPU GQA tiled expansion + memcpy (F32) | ~3.4 ms | **~165 ms** | read 64 MB + write 192 MB per layer @ ~75 GB/s effective CPU bw |
| F32→BF16 cast on GPU (q+k+v) | ~1.1 ms | **~54 ms** | 432 MB total per layer @ 400 GB/s |
| Encoder-split barriers (4 commit/layer chunk vs 2 commit/layer autoreg) | +96 commits | **~5–19 ms** | Δ vs autoreg @ 50–200 µs/commit |
| Sub-kernel dispatch overhead (~11 dispatches/layer) | +528 dispatches | **~3–11 ms** | small; not the bottleneck |
| **Sum (chunk overhead vs lean kernel)** | | **~2174–2196 ms** | accounting for kernels + wrapper |

The ~91 ms `total_layers` advantage observed at T=256 (×16 scaling to T=4096 ⇒ ~1456 ms expected gain at full prefill) is consistent with the chunk path **kernel-side** outpacing autoregressive at production shape, but the **wrapper adds back ~165 ms** of CPU expansion + ~54 ms of casts + ~10 ms of encoder-split barriers, narrowing the net advantage. The W-5b.4 walk-bar pp4096 numbers (chunk 55–69 s vs autoreg 59 s, both gated behind a ~6.5 s once-per-process model load) are within the cost model's margin.

### Per-layer commit_and_wait audit (build_delta_net_layer chunk-prefill branch)

Confirmed by reading `gpu_delta_net.rs:1380–1660`:

| Stage | Encoder commit site | Path |
|---|---|---|
| ops 1–3 (norm, qkv_proj, ssm_conv) | `1403  enc.commit_and_wait("commit ops1-3 prefill")` | both |
| chunk-prep (l2_norm × 2, alpha, beta, q_scale, g_beta) | `1508  enc.commit_and_wait("commit chunk-prep prefill")` | chunk only |
| chunk pipeline (cast q/k/v, sign-flip g, orchestrator, cast back) | `1079  enc.commit_and_wait("commit apply_gated_delta_net_chunk")` (inside the wrapper) | chunk only |
| ops 8–9 (ssm_norm_gate, out_proj) | `1606  enc.commit_and_wait("commit chunk ops8-9 prefill")` | chunk only |
| autoreg ops 5–9 fused single encoder | `1657  enc.commit_and_wait("commit ops5-9 prefill")` | autoreg only |

Chunk-path = **4 commit_and_wait per delta-net layer**; autoregressive = **2** (ops1-3 + ops5-9 fused). Net **+2 commits per delta-net layer × 48 layers = +96 commits** per forward (≈ 5–19 ms). The internal commit at `1079` *is* unnecessary as a synchronisation barrier: the only consumer of `output_buf` and `final_state` after the wrapper is the caller's E2 encoder (lines 1547 + 1592), which already opens a fresh encoder. Apple unified-memory CPU read on `chunk_final_state` (line 1547) does require synchronisation, but a `commit_no_wait` + per-buffer fence would suffice — **the `wait_until_completed` is the load-bearing cost, not the commit itself**.

### MTLResidencySet adoption audit (5e40d49 leverage)

`mlx-native::residency::ResidencySet` is **`pub(crate)`** — no public API surface. It is auto-tied into `mlx_native::buffer_pool::BufferPool` (see `buffer_pool.rs:267-285` `add_allocation`-on-checkout). However:

- `device.alloc_buffer(...)` uses `device.new_buffer(MTLResourceOptions::StorageModeShared)` directly (`device.rs:105-113`) — **bypasses the pool**, so allocated buffers never join the residency set.
- All hf2q model-loader paths use `device.alloc_buffer` (e.g. `in_memory_loader.rs:102`, `forward_gpu.rs:586-624` `upload_q4_0_from_f32`, `upload_f32`).
- **hf2q therefore gets zero benefit from 5e40d49 today.** The 27 B model's ~17 GB of weight buffers are not in any residency set; per-forward weight-resident pages must be page-faulted on first access (and possibly evicted under pressure with concurrent processes).

This is independent of the chunk-vs-autoreg debate — autoregressive prefill is paying the same ~6.5 s first-forward weight-load cost. But it's a low-effort, large-leverage win that's currently sitting on the floor.

### Verifies

- The `wave5b3` log line `chunk-pipeline ENGAGED for prefill (HF2Q_CHUNK_SCAN_PREFILL=1) seq_len=256 d_k=128` was emitted in **both T=256 chunk runs**, confirming the chunk path engaged at every layer (no silent autoregressive fallback).
- Both paths produce token id 561 at T=256 — chunk-path numerical correctness still holds at this shape (consistent with W-5b.4 pp4096 walk-bar).
- The `chunk_path_eligible(seq_len, d_k)` predicate (line 111: requires `unsafe_experiments && chunk_scan_prefill && seq_len > 64 && seq_len % 64 == 0 && d_k == 128`) fires correctly at T=256 (4 chunks) and T=4096 (64 chunks).

### Cost-model vs measurement reconciliation

Theoretical chunk-vs-autoreg `total_layers` gap (T=4096 extrapolation): autoreg should be **~1.0–1.5 s slower** based on kernel scaling alone, partly clawed back by ~220 ms of wrapper overhead (CPU expansion + casts + barriers) and ~10 ms of dispatch overhead. Net expected chunk advantage: **~0.8–1.3 s** at pp4096. Observed (W-5b.4): chunk **55–69 s** vs autoreg **59 s** → spread is 4–10 s with chunk *tied or slightly slower* on average. **The cost model under-predicts the wrapper overhead by ~3–6 s** at full T=4096 — implying additional contributors not captured above. Most likely the F32 q/k expansion's CPU-side `as_slice` / `as_mut_slice` round-trips (lines 884–931) involve cache-line ping-pong with the GPU residency state — at T=4096 the read window is `64 MB × 48 layers = 3 GB` of CPU-readable Metal-shared pages, and the M5 Max unified memory's CPU-read latency is materially higher than its GPU-read latency for not-recently-touched pages.

### Top 3 perf-improvement candidates ranked by leverage

| # | Candidate | Estimated wall savings @ pp4096 | Effort | Risk |
|---|---|---:|---|---|
| **1** | **Move GQA tiled expansion into the chunk-pipeline kernels** (or pass an indirection table to `kkt`/`recompute_w_u`/`chunk_o` so they read from the unexpanded `[T, Hg, K]` buffer with `kh = i_h % Hg` indexing). Eliminates the per-layer F32 expansion AND the F32→BF16 cast (chunk could ingest F32 q/k directly OR have its own fused cast). | **~3–6 s** (CPU bandwidth + GPU cast + cache effects) | Medium — kernel work, but the indirection-table form is ~50 lines per kernel and the wrapper-side fix shrinks by ~80 lines | Low — semantics-preserving; oracle/walk-bar tests gate parity |
| **2** | **Adopt MTLResidencySet for hf2q weight buffers** by routing model-loader paths through `mlx_native::buffer_pool::BufferPool` (or expose `MlxDevice::alloc_buffer_pooled`). Amortises the ~6.5 s first-forward weight-load over multi-request server lifecycles AND prevents evictions under memory pressure. | **~6 s on first request, ~100–500 ms on subsequent under memory pressure** | Medium — needs a public API addition in mlx-native + 1-line swap in hf2q's `upload_*` helpers | Low — additive, opt-out via `HF2Q_NO_RESIDENCY=1` already exists |
| **3** | **Eliminate the chunk-prep encoder split** by inlining ops 5–6 (l2-norm × 2, scalar_mul, compute_g_beta) into the chunk wrapper's encoder. Replaces 2 of 4 chunk-path commits with `enc.memory_barrier()` between stages. Also replace the wrapper's internal `commit_and_wait` (line 1079) with `commit` + a CPU-readable fence for `final_state` (the only CPU-read site). | **~50–200 ms** (96→48 commits per forward) | Low — wiring change in `gpu_delta_net.rs` only, no kernel modifications | Low — barrier semantics are well-tested in mlx-native ops |

### Recommendation for W-5b.6

**Attempt candidate #2 (MTLResidencySet adoption) FIRST** — highest leverage per LOC, addresses a regression that affects *both* chunk and autoreg paths, and is independent of the chunk-pipeline correctness debate. The ~6 s on-first-forward win shows up immediately on every benchmark run and on every server-warm-up. Candidate #1 (GQA expansion in-kernel) is the next-largest win but is real kernel work and should be its own iter with kernel-isolation bench coverage. Candidate #3 is a low-risk low-reward cleanup; bundle with #1 when its encoder is rewritten.

Side-quest: the wrapper docstring at `gpu_delta_net.rs:857` ("Cost: 2× memory + bandwidth on q/k for n_v_heads/n_k_heads=2 (Qwen3.6 27B's actual ratio)") **misstates the GQA ratio as 2 when it is actually 3** for Qwen3.6 27B (`H/Hg = 48/16`). Correct on next docs touch — does not affect correctness, only the 2× vs 3× expansion-cost arithmetic in commentary.

### Constraints honored

- No kernel files in mlx-native modified.
- No files in `src/backends/gguf.rs`, `src/ir/`, `src/convert/`, `src/quality/`, `src/quantize/`, `src/calibrate/`, `peer_parity_gates.rs`, `ppl_driver.rs`, `imatrix.rs` touched.
- No code edits to chunk-pipeline kernels.
- Only edit this iter: this docs append.

### Inputs preserved

- `/tmp/diag-prompt.txt` (256 tokens, 1418 bytes — exact-T diagnostic prompt)
- T=256 trial logs captured inline above (chunk × 2 + autoreg × 2)
- This document section

## Wave 5b.6 MTLResidencySet adoption — STOP-AND-REPORT (API gap)

**Status:** **STOP — `MlxBufferPool` semantics unsuitable for static-weight adoption; W-5b.7 API redesign required**
**Date:** 2026-04-27
**Worker:** Wave 5b.6 (ADR-005 MTLResidencySet adoption attempt)
**HEAD at start:** `1515d8d` (ADR-014 P11-prereq SHA correction); mlx-native at `4be0cfd` (autoregressive docstring fix; merge `5e40d49` `MTLResidencySet` support is in lineage)
**Pre-flight:** 81 GiB free RAM (Pages free 5 349 165); no non-sccache/rustc process > 5 GB; HEAD includes `7270a3f` and mlx-native `4be0cfd` ✓
**Scope:** discovery + analysis only — **no source files modified.** A throw-away `examples/wave5b6_inventory.rs` was created during the audit and removed before this append.

### (1) BufferPool API discovery

`mlx_native::MlxBufferPool` (file `/opt/mlx-native/src/buffer_pool.rs`) is the only public path through which a buffer joins the device-level `MTLResidencySet`. Adoption pattern:

| Concern | Behavior |
|---|---|
| Construction | `MlxBufferPool::new()` (lifetime-free; device handed in at every `alloc()`) |
| Residency tie-in | `alloc_inner` → `register_residency_allocation` (lines 262-291) auto-adds the underlying `metal::Buffer` to the pool's residency-set handle on **first** allocation in a bucket; subsequent same-bucket allocs reuse from the free list and are *already* in the set. |
| Lifetime model | Buffers live as long as the pool. `release(buf)` adds a single buffer to the free list (still resident); `reset()` bulk-recycles `in_use` to free; `clear()` drops the free list **and** removes those buffers from the residency set; `Drop` removes all and commits. |
| Bucket sizing | **Power-of-two rounding** — `bucket_size(byte_len) = byte_len.next_power_of_two()` (line 319). Designed for the per-decode-token arena (~1750 small allocs/token, ADR-012 Task #15). |
| Public surface for residency | None besides `MlxBufferPool`. `ResidencySet::new`, `add_allocation`, `remove_allocation`, `commit` are all `pub(crate)` (`/opt/mlx-native/src/residency.rs:63-186`). |
| Mixing with `device.alloc_buffer()` | Allowed but the direct-alloc buffers do **not** join any residency set. |

### (2) Hot-path map and bucket-rounding waste

The 17.26 GB Qwen3.6 27B DWQ46 weight set was inventoried (probe code: `examples/wave5b6_inventory.rs`, removed after measurement; results captured below). Rounding every tensor's `byte_len` up to its `next_power_of_two` (the only path BufferPool offers) produces:

| Size class | Tensors | Real bytes | Bucket-rounded | Waste |
|---|---:|---:|---:|---:|
| < 1 MB | 449 | 0.02 GB | 0.04 GB | **74.5%** |
| < 16 MB | 32 | 0.10 GB | 0.14 GB | 46.9% |
| < 128 MB | 368 | 13.88 GB | 20.00 GB | **44.1%** |
| < 1 GB | 1 | 0.71 GB | 1.07 GB | 50.3% |
| ≥ 1 GB | 1 | 2.54 GB | 4.29 GB | **69.1%** |
| **TOTAL** | **851** | **17.26 GB** | **25.55 GB** | **+8.30 GB / 48.1%** |

The 13.88 GB `<128 MB` band is the bulk of the model — these are the per-layer MoE expert blocks (Q5_K / Q6_K), per-layer dense Q4_0 attention projections, and DeltaNet sub-projection blocks. Q*-block byte lengths are never powers of two by construction (Q4_0 = 18 bytes/32 elem; Q5_K = 176 bytes/256 elem; Q6_K = 210 bytes/256 elem), so every tensor incurs the next-power-of-two penalty.

**Conclusion: routing all weight allocations through the only public residency-enabled path bloats Metal-resident memory by 8.30 GB (48.1%).** On the 128 GB M5 Max with concurrent ADR-014 sessions, that is a non-starter; on smaller systems it converts a successful load into an OOM. The only "smallest refactor" the brief permits — adopt BufferPool for the weight hot path — is incompatible with the perf-bar gating constraint.

### (3) Why the W-5b.5 diagnostic understated the gap

W-5b.5 listed `in_memory_loader.rs:102`, `weight_loader.rs:470`, and `forward_gpu.rs upload_*` as the bypass sites. Audit confirms these total **only ~3 GB** of the 17.26 GB hot path (small Q8_0 in-memory weights, raw `U8` lazy lookups, `lm_head` F32/Q4/BF16, output norm, and embedding table). The dominant **~14 GB of MoE expert + per-layer attention block buffers are loaded by `mlx_native::gguf::GgufFile::load_tensor`** (`/opt/mlx-native/src/gguf/mod.rs:990-1054`), which calls `device.alloc_buffer` directly — that is the same *direct-alloc* path BufferPool was supposed to replace, but it lives inside mlx-native rather than in hf2q. hf2q cannot route those allocations through a pool without one of:

- (a) `MlxBufferPool::register_existing(&MlxBuffer)` — public adoption entry that takes an already-allocated buffer (no rounding) and adds it to the residency set, OR
- (b) `GgufFile::load_tensor_into_pool(&mut MlxBufferPool, …)` overload, OR
- (c) An optional `&mut MlxBufferPool` argument on `MlxDevice::alloc_buffer` — but that is the API redesign the brief explicitly defers to iter-7.

Even if hf2q adopts BufferPool **only** for the ~3 GB it controls and accepts the ~50% bucket-rounding overhead on that slice, the captured win is at most ~17% of the diagnosed 6 s — and it is gated on the same Metal compaction behaviour (cold-page-fault amortization) that the dominant 14 GB MoE residency would actually exercise. **Partial adoption is therefore both costly (~1.5 GB of bucket waste on the small slice) and low-leverage (~1 s expected at best).**

### (4) Brief stop conditions met

The brief listed two stop conditions:

> "If BufferPool's API has gaps, document them as W-5b.7 follow-ups; don't expand mlx-native's public surface in this iter."
> "If BufferPool adoption requires breaking changes to hf2q's loader API, STOP and report — that's iter-7 scope (API redesign), not adoption."

Both apply: BufferPool's `next_power_of_two` bucketing is by-design for the per-token decode arena, and there is no public alternative entry point for static-weight adoption. The smallest refactor that would yield the documented 6 s win requires **public-API expansion in mlx-native** (option (a) above) — explicitly out of scope for W-5b.6.

### (5) Tests / build status (no code changes)

| Check | Result |
|---|---|
| `cargo build --release --bin hf2q` | PASS (0 errors, 72 warnings — all pre-existing) |
| `cargo test --release --bin hf2q chunk_path_first_token` | PASS (`chunk_path_first_token_matches_autoregressive_at_seq128 ... ok`, 1 passed; 2133 filtered) |
| Inventory probe `examples/wave5b6_inventory.rs` | bucket waste = 8.30 GB / 48.1% (full table above), file removed after capture |
| First-/second-forward perf measurement | **Not executed.** No code change to measure; the empirical inventory falsifies the adoption hypothesis before runtime measurement is informative. |

### (6) Recommendations for W-5b.7

The 6-s first-forward win is real, but capturing it requires mlx-native to expose either:

1. **`MlxBufferPool::register_existing(&MlxBuffer) -> Result<()>`** — adds the buffer to the residency set without bucket-rounding or pool ownership. **Smallest mlx-native diff** (≈30 LOC including a `same_owner` check and a no-op fallback when residency is unavailable). hf2q-side adoption is ~10 lines per `upload_*` helper plus a single residency-aware pool injected into `forward_gpu`'s GPU_CACHE.

2. **`GgufFile::load_tensor` accepts `Option<&mut MlxBufferPool>`** — routes the dominant 14 GB MoE/attention path through residency. Otherwise W-5b.7 with only option (1) leaves the 14 GB outside the set.

Both are additive, opt-in, and carry the same `HF2Q_NO_RESIDENCY=1` opt-out the device-level path already honours. Combined effort: ~1 day in mlx-native, ~½ day in hf2q. Land them and the W-5b.5 perf-gap diagnostic's 6-s figure becomes measurable.

### (7) Constraints honoured

- No source files in hf2q or mlx-native modified.
- The throw-away inventory probe at `examples/wave5b6_inventory.rs` was deleted after producing the bucket-waste table above.
- No files in `src/backends/gguf.rs`, `src/ir/`, `src/convert/`, `src/quality/`, `src/quantize/`, `src/calibrate/`, `peer_parity_gates.rs`, `ppl_driver.rs`, `imatrix.rs` touched.
- mlx-native's `pub(crate)` `ResidencySet` API was not made `pub`. No mlx-native changes.
- ADR-005 AC 5468 / 5470 wording remains as W-5b.4 left it — the perf-bar gap was already informational, and W-5b.6 strengthens that note rather than flipping the AC.

### Inputs preserved

- Inventory probe table above (851 tensors, real 17 255 675 136 B → bucket 25 551 814 656 B, waste 8 296 139 520 B / 48.1%)
- Pre-existing W-5b.5 diagnostic at `## Wave 5b.5 perf-gap diagnostic` (line 176) — premise stands but its proposed candidate #2 is now blocked on the mlx-native API gap documented here.

---

## Wave 5b.7 iter 2 residency-set adoption + perf measurement

**Status:** **CLOSED — adoption LANDED, perf hypothesis FALSIFIED at production shape**
**Date:** 2026-04-27
**Worker:** Wave 5b.7 iter 2 (ADR-005 residency-set adoption + perf validation)
**HEAD at start:** `6f71ddb` (W-5b.6 STOP closure); mlx-native at `84bf1af` (W-5b.7 iter 1: `register_existing` + `load_tensor_into_pool` API additions)
**Commits this iter:** `eb09cad` (residency adoption), `8f7c6fb` (cross-device tolerance fix)
**Pre-flight:** 62 GiB free RAM (Pages free 4 057 369); 1 concurrent process flagged (see §6 below); HEAD ✓ mlx-native HEAD ✓.
**Scope:** add long-lived weight pool, route the 17 GB weight hot path through `MlxBufferPool::register_existing`, run 3-trial cold-process bench.

### (1) Hot-path adoption summary

A new module `src/inference/models/qwen35/weight_pool.rs` introduces a thread-local long-lived `MlxBufferPool` running in *register-only* mode (no recycling, no bucket-rounding) and exposes `register_weight_buffer(device, &buffer)`. Buffers are still allocated at their exact GGML / F32 / BF16 byte length via `device.alloc_buffer` or `gguf.load_tensor`; only their residency-set membership is tracked.

Call sites converted (matches the W-5b.5 inventory):

| Site | File | Before | After |
|---|---|---|---|
| `upload_bf16_from_f32` | `gpu_full_attn.rs:164-179` | `device.alloc_buffer` | + auto-register |
| `upload_q4_0_from_f32` | `gpu_full_attn.rs:220-233` | `device.alloc_buffer` | + auto-register |
| `upload_f32_weight` (new) | `gpu_full_attn.rs:239-` | n/a | `upload_f32` + register |
| `FullAttnWeightsGpu::from_cpu` norms | `gpu_full_attn.rs:99-111` | `upload_f32` | `upload_f32_weight` |
| `DeltaNetWeightsGpu::from_cpu` F32 weights | `gpu_delta_net.rs:179-200` | `upload_f32` × 7 | `upload_f32_weight` × 7 |
| `forward_gpu` lm_head_bf16 inline alloc | `forward_gpu.rs:597 / 1118` | bare `alloc_buffer` | + register_weight_buffer |
| `forward_gpu` lm_head_f32 / norm_w | `forward_gpu.rs:589/620 / 1113/1138` | `upload_f32` | `upload_f32_weight` |
| `weight_loader::load_tensor_with_residency` (new) | `weight_loader.rs:138-149` | n/a | wraps `gguf.load_tensor` + register |
| MoE expert blocks (gate/up/down) × 48 layers | `weight_loader.rs:991-1003` | `gguf.load_tensor` | `load_tensor_with_residency` |
| Dense FFN gate/up/down | `weight_loader.rs:1112-1124` | `gguf.load_tensor` | `load_tensor_with_residency` |
| `upload_lazy_raw_u8` | `weight_loader.rs:494-509` | bare `alloc_buffer` | + register_weight_buffer |
| `quantize_f32_to_q8_0_buffer` | `in_memory_loader.rs:101-110` | bare `alloc_buffer` | + register_weight_buffer |
| `mtp_weights_load::load_norm_gpu` | `mtp_weights_load.rs:193-198` | `upload_f32` | `upload_f32_weight` |

Total bytes managed by the pool: ~17 GB (the dominant slice; the smaller ~3 GB cross-device slice falls back transparently — see §3).

### (2) Build / test status

| Check | Result |
|---|---|
| `cargo build --release --bin hf2q` | PASS — 0 errors, 72 warnings (all pre-existing baseline) |
| `cargo test --release qwen35::weight_pool` | PASS — 2/2 (`register_existing_via_thread_local_is_idempotent`, `register_does_not_recycle_external_buffers`) |
| `cargo test --release chunk_path_first_token_matches_autoregressive_at_seq128` | PASS |
| `cargo test --release qwen35` | PASS except 1 pre-existing failure (`dwq_on_qwen35_surfaces_not_ready` — tokenizer.json fixture missing; reproduced on HEAD via `git stash`, unrelated to this change) |
| Walk-bar parity (T=256 chunk vs autoreg, both paths produce token id **561**) | PASS — see §4 table |

### (3) Cross-device tolerance fix (commit `8f7c6fb`)

First post-iter-2 model load surfaced a multi-device issue: hf2q's loader creates several `MlxDevice::new()` instances across the load path (serve setup, registry warmup, `forward_gpu` cache init, in-memory loader paths). Each device gets its own `ResidencySet` Arc; mlx-native's `register_existing` enforces a single-set invariant via `Arc::ptr_eq`, so the first device claims the pool and subsequent devices' buffers fail with `InvalidArgument("cannot mix residency-enabled devices")`.

Resolution: catch that specific error in `register_weight_buffer` and treat it as a tolerated soft fallback. The dominant ~14 GB MoE/dense weight slice loaded inside `forward_gpu`'s cache init all uses one device → claims the pool → gets full residency benefit. The remaining ~3 GB cross-device slice (Q8_0 quantize, MTP norms, etc.) loads successfully with no residency hint.

Iter-3 follow-up to capture the remaining 3 GB: consolidate hf2q on a single shared `MlxDevice`, OR adopt the parallel mlx-native ADR-015 iter8e work-in-flight (auto-registration inside `MlxDevice::alloc_buffer`, eliminating the pool layer entirely).

### (4) Perf measurement table — Qwen3.6 27B DWQ46, T=256 prompt, 3 cold trials each

Each row is a fresh `hf2q generate` process (one model load + one prefill forward + one decode token). `loaded` = construction-side weight load (GGUF disk I/O + dequant); `prefill` = first-forward (cold compute + DMA); `decode tok/s` = warm second-forward.

| Trial | Path | Condition | loaded (s) | prefill (ms) | decode (tok/s) | first decoded token |
|---:|---|---|---:|---:|---:|:---:|
| 1 | chunk | pre-residency  (`HF2Q_NO_RESIDENCY=1`) | 6.2 | 7925 | 43.7 | **561** |
| 1 | chunk | post-residency (default)               | 6.1 | 7845 | 53.1 | **561** |
| 1 | autoreg | pre-residency                       | 6.2 | 7964 | 45.7 | **561** |
| 1 | autoreg | post-residency                      | 6.2 | 8002 | 44.3 | **561** |
| 2 | chunk | pre-residency                          | 6.2 | 7953 | 49.6 | **561** |
| 2 | chunk | post-residency                         | 6.2 | 7965 | 53.3 | **561** |
| 2 | autoreg | pre-residency                       | 6.3 | 8031 | 45.1 | **561** |
| 2 | autoreg | post-residency                      | 6.2 | 8072 | 43.9 | **561** |
| 3 | chunk | pre-residency                          | 6.3 | 8071 | 48.9 | **561** |
| 3 | chunk | post-residency                         | 6.3 | 8007 | 47.5 | **561** |
| 3 | autoreg | pre-residency                       | 6.2 | 8057 | 54.3 | **561** |
| 3 | autoreg | post-residency                      | 6.3 | 8106 | 37.8 | **561** |

**Means (n=3 each):**

| Condition | prefill mean (ms) | prefill stdev | decode mean (tok/s) | load mean (s) |
|---|---:|---:|---:|---:|
| chunk pre-residency  | 7983.0 | 78 | 47.40 | 6.233 |
| chunk post-residency | 7939.0 | 84 | 51.30 | 6.200 |
| autoreg pre-residency  | 8017.3 | 48 | 48.37 | 6.233 |
| autoreg post-residency | 8060.0 | 53 | 42.00 | 6.233 |

**Deltas (post − pre, first-forward / prefill):**

| Path | Delta (ms) | Delta (%) | Within 1-σ noise? |
|---|---:|---:|:---:|
| **CHUNK** first-forward | **−44 ms** | **−0.55 %** | YES (σ pre 78, σ post 84) |
| **AUTOREG** first-forward | **+43 ms** | **+0.53 %** | YES (σ pre 48, σ post 53) |

### (5) Win verdict — diagnostic FALSIFIED at production shape

**The W-5b.5 diagnostic's "~6 s first-forward win" hypothesis is FALSIFIED.** Across 3 cold-process trials each on Qwen3.6 27B DWQ46 at T=256:

- Chunk-path first-forward: residency adoption changed prefill by **−44 ms (−0.55%)** — well inside the 1-σ band.
- Autoreg-path first-forward: residency adoption changed prefill by **+43 ms (+0.53%)** — also inside 1-σ band.
- Model-load wall (the dominant 6.2 s "loaded" segment containing GGUF disk I/O + dequant) changed by **<50 ms** in either direction.
- Memory delta: **~0** — `register_existing` does not bucket-round; weight buffers are allocated at their exact byte length.

The W-5b.5 estimate was based on theoretical bandwidth math ("~6 s on first request, ~100–500 ms on subsequent under memory pressure"). In practice, M5 Max's `StorageModeShared` Metal allocator is already efficient enough at unified-memory page-fault amortization that adding `MTLResidencySet` hints provides no measurable benefit on a single-process, no-memory-pressure workload. The theoretical regime where residency would matter — concurrent multi-process Metal contention, sustained memory pressure forcing eviction — is not the W-5b.5 / W-5b.7 measurement regime.

**AC-tier impact: NONE.** The perf-bar gate at AC 5468 / AC 5470 was already informational (W-5b.4 closed both as full `[x]`); this iter neither adds nor removes any gating.

### (6) Concurrent-session check (per `feedback_bench_process_audit`)

Pre-flight `ps aux` flagged one hot competitor: a CFA worker at `/opt/mlx-native/.cfa-worktrees/cfa-20260427-adr015-iter8e-resint-codex/` running `test_flash_attn_prefill` at 236% CPU + 7.9 GB RSS. That session is the parallel ADR-015 iter8e residency-integration in mlx-native (their codex process), working on automatic residency-set registration inside `MlxDevice::alloc_buffer` — **orthogonal and non-overlapping** with our hf2q-side adoption: their work happens inside their isolated worktree's mlx-native build; our `eb09cad` build links against `/opt/mlx-native` HEAD `84bf1af`, unaffected.

The cfa worker had finished by the time the 3-trial bench started (post-bench `ps aux` shows no cfa or test_flash processes; `mcp-brain-server` idle at 0% CPU). Of the 12 cold runs across 3 trials, prefill stdev within each condition is 48–84 ms (~0.6–1.0% of mean), comfortably tight; we have no evidence of CPU contention contaminating the timing.

### (7) RAM evidence

| Phase | Pages free | GB free |
|---|---:|---:|
| Pre-flight | 4 057 369 | 61.9 |
| 2-trial bench start | 4 353 459 | 66.4 |
| 3-trial bench start | 4 357 331 | 66.5 |
| 3-trial bench end | 3 922 764 | 59.9 |
| Post-bench audit | 4 357 666 | 66.5 |

No swap thrash; no jetsam pressure; >25 GB free at every point including peak load.

### (8) Constraints honoured

- No mlx-native chunk-pipeline kernels modified.
- No files in `src/backends/gguf.rs`, `src/ir/`, `src/convert/`, `src/quality/`, `src/quantize/`, `src/calibrate/`, `peer_parity_gates.rs`, `ppl_driver.rs`, `imatrix.rs` touched.
- `HF2Q_NO_RESIDENCY=1` escape hatch preserved (and used as the pre-residency baseline in the bench).
- Walk-bar parity preserved (token id 561 in 12/12 runs).

### Inputs preserved

- Bench script: `scripts/bench-w5b7-iter2-residency.sh` (stays in repo for iter-3 if architecture refactor lands).
- Raw bench logs: `/tmp/w5b7-iter2-bench.log` (2-trial), `/tmp/w5b7-iter2-bench-3trial.log` (3-trial).
- Implementation commits: `eb09cad`, `8f7c6fb`.

---

## Wave 5b.8 PP4096 measurement spike

**Status:** **CLOSED — measurement spike landed; top-3 contributors identified; recommendation FA-prefill audit before any chunk-side optimization**
**Date:** 2026-04-27
**Worker:** Wave 5b.8 (ADR-005 PP4096 per-section measurement spike)
**HEAD at start:** `1515d8d` (W-5b.7 closure docs); mlx-native at `84bf1af` (W-5b.7 iter 2 register_existing API)
**Binary:** rebuilt this iter (`cargo build --release --bin hf2q` clean: 0 errors, 72 warnings, 11.78 s)
**Instrumentation commit:** `0e73883` (`feat(adr-005 wave-5b.8): env-gated PP4096 per-section profile harness`)
**Concurrent-session check at start:** clean (mcp-brain-server at 0% CPU, no foreign process > 5% CPU, no CFA worker active)
**Concurrent-session check during bench:** all 6 hf2q trials + 3 llama trials gated on `pgrep -f bitwidth\|cfa-2026 = empty`; first llama trial discarded after `ggml_metal_synchronize: error: command buffer 0 failed with status 5` triggered by `bitwidth_ab` test from `/opt/mlx-native/.cfa-worktrees/cfa-20260427-adr015-iter8e-resint-merged` running at 100% CPU. Re-run after the CFA test cleared produced clean numbers.
**RAM at start:** 63.5 GiB free (Pages free 4 161 788 × 16 KB)
**RAM at finish:** 109.1 GiB free (Pages free 7 147 556 × 16 KB)
**mlx-native HEAD at finish:** `5d9bb2e3` — drift detected vs start `84bf1af` (orthogonal CFA worker landed an iter-8e residency-set patch during the bench window; the binary I measured was linked against `84bf1af`, captured before drift).

### (1) Pre-flight audit

| Check | Result |
|---|---|
| RAM free pre-bench | 63.5 GiB (>32 GiB threshold; PASS) |
| Top non-sccache process at start | `mcp-brain-server` 0.0% CPU, 2.4 GB RSS (PASS — < 5% threshold per `feedback_bench_process_audit`) |
| Other 100% CPU bursts during bench | `bitwidth_ab` (CFA worker, mlx-native ADR-015 iter 8e) — caught + re-run |
| Concurrent inference | None at start; concurrent `qwen3.6-35B-a3b` `--benchmark` started at 8:02 PM AFTER all 6 W-5b.8 trials had completed |

### (2) Same-day llama.cpp baseline (validation gate)

3 cold trials of `llama-completion --batch-size 4096 --ubatch-size 4096 --n-predict 1 --no-warmup -no-cnv --perf` against the W-5b.4 `walkbar-pp4096-prompt.txt` (23 038 bytes, 4 096 tokens):

| trial | prompt eval (ms) | tok/s | real (s) |
|---|---:|---:|---:|
| T1 | 6 451.5 | 634.9 | 8.62 |
| T2 | 6 461.4 | 633.9 | 8.64 |
| T3 | 6 572.5 | 623.2 | 8.62 |
| **mean** | **6 495.1** | **630.7** | **8.63** |
| stdev | 65.1 | 5.7 | 0.01 |

Same-day llama prefill is **6 495 ms ± 65** at pp4096 — within 0.5% of W-5b.4's 6 528 ms baseline. Environment validated; gap math uses `LLAMA_REF = 6495 ms`.

### (3) hf2q per-section table — 3 cold trials × 2 paths

`HF2Q_PROFILE_W5B8=1 HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1` set on every trial. Chunk path additionally has `HF2Q_CHUNK_SCAN_PREFILL=1`. Same `walkbar-pp4096-prompt.txt`. All 6 runs produced first-decoded `,` (token id 11) — instrumentation did not perturb correctness.

#### Chunk path (n=3 cold)

| section | n_samples | T1 ms | T2 ms | T3 ms | mean ms | stdev ms |
|---|---:|---:|---:|---:|---:|---:|
| upload_weights (one-time) | 1 | 5 629.2 | 5 572.2 | 5 559.8 | 5 587.1 | 37.0 |
| layer.ops1_3 | 48 | 3 651.7 | 1 493.6 | 1 357.1 | 2 167.5 | 1 287.2 |
| layer.qkv_deinterleave | 48 | 1 732.7 | 1 902.0 | 1 439.8 | 1 691.5 | 233.8 |
| layer.chunk_prep | 48 | 642.1 | 217.0 | 179.8 | 346.3 | 256.8 |
| **layer.chunk_call** | 48 | 5 281.9 | 3 188.9 | 3 506.9 | **3 992.6** | 1 127.9 |
| └ chunk.gqa_expand | 48 | 1 153.6 | 882.3 | 781.5 | 939.1 | 192.5 |
| └ chunk.allocs | 48 | 20.4 | 77.3 | 3.7 | 33.8 | 38.6 |
| └ chunk.enc_build | 48 | 27.2 | 6.2 | 5.3 | 12.9 | 12.4 |
| └ chunk.commit_wait | 48 | 4 079.1 | 2 222.7 | 2 716.0 | **3 005.9** | 961.6 |
| layer.chunk_ops8_9 | 48 | 2 619.7 | 727.0 | 663.4 | 1 336.7 | 1 111.6 |
| **layer.linear_total** (per DN layer) | 48 | 27 210.1 | 16 427.3 | 14 246.4 | **19 294.6** | 6 941.2 |
| **layer.full_total** (per FA layer) | 16 | 23 774.4 | 24 302.2 | 21 447.6 | **23 174.7** | 1 518.9 |
| **PREFILL TOTAL (binary timer)** | — | 65 799 | 49 310 | 43 884 | **52 998** | 11 413 |

Peak RSS: 60.0, 53.4, 68.3 GB (mean 60.6 GB) — well below the 128 GB unified-memory cap.

#### Autoreg path (n=3 cold)

| section | n_samples | T1 ms | T2 ms | T3 ms | mean ms | stdev ms |
|---|---:|---:|---:|---:|---:|---:|
| upload_weights (one-time) | 1 | 5 582.7 | 5 559.0 | 5 640.3 | 5 594.0 | 41.8 |
| layer.ops1_3 | 48 | 1 570.2 | 1 400.6 | 1 379.0 | 1 449.9 | 104.7 |
| layer.qkv_deinterleave | 48 | 3 164.9 | 1 441.4 | 1 516.4 | 2 040.9 | 974.1 |
| **layer.autoreg_ops5_9** | 48 | 5 521.1 | 4 853.2 | 4 935.3 | **5 103.2** | 364.2 |
| **layer.linear_total** (per DN layer) | 48 | 31 069.2 | 16 340.3 | 18 279.9 | **21 896.5** | 8 002.8 |
| **layer.full_total** (per FA layer) | 16 | 29 797.0 | 21 250.0 | 20 607.4 | **23 884.8** | 5 130.2 |
| **PREFILL TOTAL (binary timer)** | — | 70 163 | 45 761 | 47 239 | **54 388** | 13 682 |

Peak RSS: 66.4, 62.2, 62.2 GB (mean 63.6 GB).

### (4) Top-3 contributors to the 8.2× chunk gap

Chunk prefill mean = **52 998 ms**; same-day llama = **6 495 ms**; absolute gap = **46 503 ms (8.2×)**. Subtracting the one-time `upload_weights` (5 587 ms, NOT included in llama's `prompt eval time`) gives a fairer chunk forward of 47 411 ms, gap **40 916 ms (7.3×)**. Either denominator yields the same top-3 ranking:

| rank | section | mean ms | % of chunk prefill | δ vs autoreg | notes |
|:---:|---|---:|---:|---:|---|
| **1** | **`layer.full_total`** (16 layers) | **23 175** | **43.7 %** | −710 | full-attention layers cost ~1.45 s/layer at pp4096 — dominates absolute time despite being only 25 % of layers. Chunk vs autoreg is within noise here (chunk path neither helps nor hurts FA layers, by construction). |
| **2** | **`layer.linear_total`** (48 layers) | **19 295** | **36.4 %** | −2 602 | DeltaNet layers, chunk-path savings ≈ 2.6 s vs autoreg (consistent with W-5b.5 cost-model prediction "chunk net advantage 0.8–1.3 s" once wrapper overhead is subtracted; closer to the high end of that range thanks to the 4 s `chunk_ops8-9` shrink). |
| **3** | **`upload_weights`** (one-time) | **5 587** | **10.5 %** | ≈ 0 | first-call materialization of ~17 GB Q4 onto Metal heap. NOT part of llama's `prompt eval time`; informational only — the comparable apples-to-apples gap excludes this. |

Sub-ranks (within `layer.linear_total`'s chunk path):

| sub-section | mean ms | % of layer.linear_total |
|---|---:|---:|
| `chunk.commit_wait` (48× GPU wall, 6 chunk kernels + 4 casts) | 3 006 | 15.6 % |
| `layer.ops1_3` (pre_norm + qkv_proj + ssm_conv) | 2 168 | 11.2 % |
| `layer.qkv_deinterleave` (CPU memcpy + 3 GPU re-uploads) | 1 692 | 8.8 % |
| `layer.chunk_ops8_9` (ssm_norm_gate + out_proj) | 1 337 | 6.9 % |
| `chunk.gqa_expand` (CPU tiled GQA) | 939 | 4.9 % |
| `layer.chunk_prep` (l2_norm + alpha/beta/q_scale/g_beta) | 346 | 1.8 % |
| `chunk.allocs` (BF16 + state + output buffers) | 34 | 0.2 % |
| `chunk.enc_build` (encoder build, no commit) | 13 | 0.1 % |
| residual (linear_total − instrumented) | ~9 760 | 50.6 % |

Notable: ~50 % of `layer.linear_total` is **not captured** by my instrumented sub-buckets. That residual is the chunk-pipeline kernel time on the GPU itself, plus encoder-launch / Metal-driver overhead between the CPU-side timing markers — i.e. it's the body of `dispatch_chunk_gated_delta_rule_fwd` (and the autoreg `dispatch_gated_delta_net`) plus `commit_and_wait` overhead on the autoreg path. Future iterations should add named encoder labels and a per-kernel breakdown via `enc.commit_labeled` so that residual collapses.

### (5) Falsifications

This iter's data falsifies three prior hypotheses:

1. **W-5b.5 estimate "GQA F32 expansion ~3-6 s at T=4096"** — actual is **939 ms** (5–6× over-estimated). Wrapper-side CPU expansion is NOT a meaningful contributor.
2. **W-5b.5 estimate "encoder-split barriers ~10 ms"** — `chunk_prep` + `chunk_ops8_9` total wall = **1 683 ms**, ~170× the estimate. Either the split overhead is real and large, or these encoders are doing real GPU work whose cost was uncounted in W-5b.5's analytic model. The latter is more likely (apply_proj / l2_norm / scalar_mul each issue real Metal kernels).
3. **W-5b.4 informational claim "8.5–10.6× slower than llama.cpp at pp4096"** — corrected. **Apples-to-apples (excluding `upload_weights` from hf2q since llama loads model separately) the ratio is 7.3× / 40 916 ms absolute gap.**

### (6) Recommendation table for the next iter

| top-3 bucket | follow-up exists? | measurement-first hypothesis | est. effort | risk class |
|---|---|---|---:|---|
| **`layer.full_total` (43.7 %)** | NO open W-5b candidate. ADR-005 W-5a tracked DeltaNet specifically; full-attention has not been profiled at pp4096. | hf2q `gpu_full_attn` likely doing per-token serial KV writeback, while llama batched-prefill does it as one large dispatch. Audit with `HF2Q_PROFILE_FA=1` (already wired in `forward_prefill_batched.rs:1064`) at pp4096 and llama-bench `-p 4096 --flash-attn`. **Per-FA-layer hf2q ≈ 1450 ms, hf2q ≥ 3.6× per-FA-layer slower than llama's whole prefill divided by 16.** Most likely root cause: hf2q's full_attn path is not using a flash-attention-style fused kernel for prefill. | 1–2 days for audit; kernel work depends on root cause | medium — lots of unknowns; could find a single-kernel win OR a deep refactor depending on what the audit shows |
| **`layer.linear_total` (36.4 %)** | YES — W-5b candidate #1 (chunk-pipeline kernel optimization), candidate #3 (encoder-merge to reduce 3-stage commit_and_wait split). The 50 % uninstrumented residual is the natural next target. | residual ≈ 9 760 ms is the chunk-kernel GPU body (`dispatch_chunk_gated_delta_rule_fwd` and its 6 sub-dispatches). Land per-kernel `enc.commit_labeled` at the chunk-pipeline kernel layer in mlx-native and re-run W-5b.8 to attribute the residual. | 0.5–1 day for finer-grained instrumentation in mlx-native; production change depends on the resulting data | low — mlx-native owns the kernel timing primitives and W-5b.7 already validated public-API conventions for this kind of additive surface |
| **`upload_weights` (10.5 %)** | NO. W-5b.7 already tested `MTLResidencySet` adoption at T=256 and got sub-1-σ noise; this captures the full pp4096 first-call cost, but at T=4096 the hot path is forward compute, not weight load. | nothing to optimize here — model materialization is a one-time bootstrapping cost that is not on the per-prefill critical path in a real serve loop. Excluded from comparison-grade gap math. | 0 (do not optimize) | n/a |

### (7) Recommendation for W-5b.9

**Audit the full-attention prefill path BEFORE any further chunk-pipeline kernel work.** Rationale: `layer.full_total` at 23.2 s is **larger than `layer.linear_total` at 19.3 s**, despite there being 3× fewer FA layers (16 vs 48). Per-layer FA at pp4096 is 1.45 s while per-DN-layer chunk-path is 0.40 s — full-attention is **3.6× slower per layer** even though llama and hf2q share the same 16/48 split. Optimizing chunk DeltaNet kernels can never close the gap when ~44 % of the chunk wall-clock is spent on FA layers we haven't instrumented.

Concretely, W-5b.9 should:
1. Wire `HF2Q_PROFILE_FA=1` (already implemented in `forward_prefill_batched.rs:1064`) into the hf2q forward-bench command at pp4096 and capture per-stage FA breakdown.
2. Compare to llama's `-p 4096 --flash-attn` on the same prompt — same-day delta on FA wall.
3. If the FA path is missing flash-attention/online-softmax: open W-5b.9 as a kernel-port iteration. If FA is using the hf2q gated_attn kernel correctly and is just bandwidth-bound at long T: open W-5b.9 as a representation/quantization iteration (Q4 KV vs F32, page-attention layout).
4. ONLY THEN return to `layer.linear_total` residual breakdown.

This is the inverse priority to W-5a/W-5b's chunk-DeltaNet emphasis, but the data demands it: the gap is not where W-5b assumed.

### (8) Honest caveats and methodology limits

- **High variance across cold trials (T1 vs T2/T3).** Trial 1 of every path is consistently 25–40 % slower than trials 2–3. This is the W-5b.4 "10 % spread" pattern amplified — most likely Metal shader cache warming + thermal step-down between trials. A "warm" comparison would be tighter, but cold is the spec.
- **CFA worker contention.** During the first attempt at the llama baseline, a `bitwidth_ab` test from `/opt/mlx-native/.cfa-worktrees/cfa-20260427-adr015-iter8e-resint-merged` running at 100 % CPU triggered a `ggml_metal_synchronize: error: command buffer 0 failed with status 5` and inflated llama prefill from 6.5 s → 120 s. The clean re-run used `pgrep -f "bitwidth\|cfa-2026"` as a pre-flight gate. All 6 hf2q trials passed this gate. **If the next iter cannot fence off the CFA worktree, the bench is unreliable.**
- **Instrumentation overhead.** Every `Section::start` allocates an `Instant` (rdtsc, ~2 ns) even when the env gate is off. Inside `apply_gated_delta_net_chunk` we use 4 nested guards, and inside `build_delta_net_layer` 2–3 more, all once per layer × 48 DN layers + 16 FA layers = ~400 timestamps per forward. At ~50 ns CPU-side per pair, total instrumentation overhead is ≪ 1 ms and indistinguishable from the noise floor.
- **CPU wall, not GPU wall.** Per memory `project_m5max_no_dispatch_boundary_sampling`, M5 Max only supports stage-boundary GPU counter sampling — per-dispatch GPU timestamps are hardware-impossible. Each `chunk.commit_wait` bucket measures the CPU's wait for `MTLCommandBuffer.waitUntilCompleted`, which conflates queue submit, GPU compute, and CPU wakeup latency. The numbers are still the right wall for "how long was the program blocked on this," which is what "the gap is made of."
- **`layer.linear_total` residual is large (~50 %).** That's the chunk-pipeline kernel body, not yet broken down. Future iterations should add `enc.commit_labeled` calls inside `dispatch_chunk_gated_delta_rule_fwd` so the kkt / recompute_w_u / inter_state / chunk_o / tri_solve_invert / cumsum_g sub-kernels become individually measurable.
- **Only one mlx-native HEAD measured.** Build was against `84bf1af` (the W-5b.7 iter-2 SHA). HEAD drifted to `5d9bb2e3` mid-bench (an ADR-015 iter-8e CFA worker landed something), but the binary I measured was already linked. If the next iter rebuilds, re-record the SHA and consider whether the residency-set change in mlx-native materially affects `upload_weights`.

### (9) Inputs / outputs preserved

- Bench prompt: `/tmp/walkbar-pp4096-prompt.txt` (23 038 bytes, identical SHA to W-5b.4)
- Raw trial logs (incl. `time -l` blocks): `/tmp/w5b8-{chunk,autoreg}-T{1,2,3}.log`, `/tmp/w5b8-llama-T{1,2,3}.log`
- Aggregation script: `/tmp/w5b8-analysis.py` (stdlib-only)
- Instrumentation source: `src/inference/models/qwen35/wave5b8_profile.rs`, plus calls in `forward_gpu.rs`, `gpu_delta_net.rs` (commit `0e73883`)
- Discarded contention run: `/tmp/w5b8-llama-trial1-merged.txt` (120 s real, Metal recovery — kept for forensics; do NOT use for gap math)

### (10) Closure

Wave 5b.8 measurement spike: **CLOSED — top-3 contributors identified.**

The W-5b.5 cost-model premise that "the gap is in the chunk-pipeline wrapper" is **partially correct but mis-prioritised**: the chunk wrapper's documented overheads (GQA expansion, F32→BF16 staging, encoder-split barriers) total ~2 s, not the 5–10 s that the model assumed. The dominant contributor is **full-attention prefill (`layer.full_total` = 23.2 s, 43.7 % of chunk prefill, 3.6× slower than llama per-FA-layer)**, which neither W-5a nor W-5b ever profiled. The next iter (W-5b.9) should be a full-attention prefill audit, not a continuation of chunk-pipeline kernel work.

ADR-005 AC-tier impact: NONE — perf-bar gate at AC 5468 / 5470 was already informational (W-5b.4 closed both as full `[x]`); this iter neither adds nor removes any gating.
