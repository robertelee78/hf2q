# ADR-005 Phase 1b — Post-1bNEW.20 Spike (fresh gap decomposition)

**Date:** 2026-04-11
**Runner:** Claude (investigation-only; no `src/` commits in this session beyond the `HF2Q_DUMP_TOKEN_SEQUENCE` env-gated token dump and the sourdough correctness gate)
**Scope:** Refresh the post-Walk gap decomposition against the post-1bNEW.20.FIX HEAD baseline. The original re-spike (`docs/spike-post-walk-results.md`) was measured on 48.82 tok/s pre-1bNEW.17 with F32 lm_head and slice-scatter KV, and its bandwidth-bound component table was explicitly marked as empirically falsified by 1bNEW.20's +26.83 tok/s actual vs 0.3-0.5 tok/s projected (ADR-005 line 857 / 938). This spike rebuilds the table for HEAD so the next Walk item has a current evidence base.
**Baseline binary:** `main` HEAD `9deb18a` (post-1bNEW.20.FIX + sourdough gate) — gate confirms byte-identical output to llama.cpp through 3094 bytes.
**Model:** `models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf` (Gemma 4 26B MoE DWQ, mixed Q4_0/Q6_K/F16).
**Hardware:** Apple M5 Max, 128 GB unified memory.

---

## Executive summary

At the post-1bNEW.20.FIX baseline **hf2q decodes at 85.3 tok/s** on the canonical 187-token bench prompt (5-run median, p95 85.5, variance 0.3). The target is llama.cpp's **107 tok/s** on the same hardware / same GGUF. Gap: **21.7 tok/s** = **2.37 ms/token**.

**Finding #1 (first-principles decomposition).** Pure memory-bandwidth floor for hf2q's forward pass on this model:

| Component | GB/token | ms at 400 GB/s |
|---|---|---|
| lm_head F16 `[1,2816] @ [2816,262144]` | 1.48 | 3.69 |
| MoE Q-mixed (30 layers × 8 experts × (gate_up 2×704 + down 704) × 2816 × 0.5 b/w) | 0.71 | 1.78 |
| MLP Q4_0 (30 × 3 × 2816 × 2112 × 0.5 b/w) | 0.27 | 0.67 |
| Attention Q4_0 (q_proj, k_proj, v_proj=k, o_proj × 30) | 0.52 | 1.30 |
| **Pure weight-bandwidth floor** | **2.98** | **7.44** |
| KV-cache reads (~0.22 GB/token, sliding-dominant) | 0.22 | 0.55 |
| Activations / small intermediates | ~0.05 | 0.13 |
| **Total bandwidth floor** | **3.25** | **8.12** |
| Measured hf2q | | **11.72** |
| Measured llama.cpp | | **9.35** |

**hf2q residual above bandwidth floor: 11.72 − 8.12 = 3.60 ms/token.**
**llama.cpp residual above bandwidth floor: 9.35 − 8.12 = 1.23 ms/token.**
**Gap between hf2q and llama.cpp: 2.37 ms/token**, entirely in the non-bandwidth bucket (CPU dispatch overhead / CPU-GPU overlap slack / per-encoder Metal setup / op-graph traversal cost).

At `dispatches_per_token = 2104.52` (from metrics.txt on HEAD), an implied per-dispatch cost of 3.60 ms / 2104 ≈ **1.71 μs/dispatch** for hf2q and 1.23 ms / 2104 ≈ **0.58 μs/dispatch** for llama.cpp (if llama.cpp issued the same count, which it does not — llama.cpp's ggml graph is flatter). The cost-per-dispatch framing is a rough reference point, not a tight model.

**Finding #2 (empirical buffer-tuning sweep, the surprise).** Candle exposes `CANDLE_METAL_COMPUTE_PER_BUFFER` (default 50, at `/opt/candle/candle-metal-kernels/src/metal/commands.rs:14`) and `CANDLE_METAL_COMMAND_POOL_SIZE` (default 5, at `commands.rs:15`) as env-var knobs on its command-pool commit/swap mechanics. The pre-spike hypothesis — derived from the post-Walk re-spike's "RUN-3 single-command-buffer-per-forward" framing — was that bigger batches = fewer swaps = less overhead = more throughput. **The sweep falsified that hypothesis completely**:

| `CANDLE_METAL_COMPUTE_PER_BUFFER` | Median tok/s (5-run) | Δ vs default (50) |
|---|---|---|
| 10 | 83.5 | **−1.8** |
| 20 | 84.5 | **−0.8** |
| **50 (default)** | **85.3** | — |
| 100 | 85.9 | **+0.6** |
| 200 | 85.9 | **+0.6** (plateau) |
| 5000 | **79.4** | **−5.9** |

Interpretation: candle's per-buffer commit threshold is a **CPU/GPU-overlap-enabling** mechanism, not an overhead burden. Every time the threshold fires, candle commits the current command buffer to the GPU and swaps in a fresh one. That commit is what lets the GPU **start executing** the first batch of ops while the CPU is still encoding the next batch into the new buffer. With a 5000-per-buffer threshold, the GPU sits idle until the *entire forward pass* is encoded (all 2104 dispatches) and then runs serialized with the CPU — regressing 85.3 → 79.4.

The 50-per-buffer default is already near-optimal. A bump to 100 gives **+0.6 tok/s free** (no code change, just the env var), and the gain plateaus there. Pool size is entirely flat across 2/3/5/10 entries.

**What this means for 1bNEW.21.** The naive "collapse all dispatches into one command buffer" direction — which the post-Walk re-spike (`docs/spike-post-walk-results.md` line 194, RUN-3) classified as a top candidate for the remaining gap — is **actively wrong** on candle's architecture. Pool mechanics are already tuned within 0.6 tok/s of their optimum. **The remaining 21-tok/s gap is NOT a pool-mechanics problem.**

**Finding #3 (what the 2.37 ms actually is).** Since it is not pool-mechanics, the 2.37 ms delta between hf2q and llama.cpp must live in one of these buckets (ranked by my current best guess):

1. **CPU-side op-graph overhead inside candle per dispatch.** Candle builds a lazy op graph: every `Tensor` op records the op, shape, dtype, layout, and device in Arc'd metadata structures. On the decode path, every `narrow`, `reshape`, `broadcast_mul`, `to_dtype` incurs Arc refcount updates, small allocations, and graph-traversal work. This happens on the CPU, *outside* the Metal kernel dispatch itself, and is not exposed by `CANDLE_METAL_COMPUTE_PER_BUFFER` tuning. llama.cpp's ggml graph is compiled ahead of time and executed with much flatter per-op setup — it doesn't rebuild the graph every token. Attacking this bucket requires either candle-side work (which is deep upstream infra, arguably Run territory under the sharpened definition) or hf2q-side work to reduce the number of candle Tensor ops by fusing more kernel boundaries.

2. **Per-encoder Metal setup overhead.** Every time candle creates a new `ComputeCommandEncoder` on a command buffer, Metal has to bind argument tables, set the pipeline state object, and set buffers. In candle today, `command_encoder()` creates a fresh encoder on every `finalize_entry` call (`commands.rs:101-104`) — *even when the previous encoder is on the same buffer*. A "sticky encoder" optimization where consecutive ops on the same buffer reuse the same encoder would cut encoder-setup cost per op. Not exposed by any existing env var; would require a small candle patch at `Commands::command_encoder` and the call sites.

3. **GPU idle time between dispatches.** Even with CPU/GPU overlap enabled by the 50-per-buffer swap, there may be short GPU gaps between successive kernel dispatches where the GPU waits for the next encoder's arg-table bindings. This is a GPU-side effect and only visible via Metal profiling traces (Xcode Instruments → Metal System Trace), not via CPU wall-clock sweeps like the buffer-tuning experiment above.

4. **Inefficient op fusion at the op-graph boundary.** MoE fan-in currently pays 17 candle Tensor ops (narrow, gelu, broadcast_mul, reshape, sum, to_dtype) around the two fused expert kernels. llama.cpp's ggml graph folds some of these into fused subgraphs. Each saved op is ~1.7 μs × 30 layers = 0.05 ms — small individually, but summed across multiple sites could total ~0.3-0.5 ms.

5. **PSO compilation overhead on warm-up paths.** 1bNEW.12 extended warmup addressed the prefill PSO cold-start, but secondary kernels (e.g., `arg_sort_last_dim` for top-k, `index_select`, `broadcast_div`) may still hit cold paths on the first forward of each new batch boundary.

Buckets 1, 2, and 4 are attackable from hf2q's side without touching candle; bucket 3 requires Metal-level instrumentation; bucket 5 is a continuation of 1bNEW.12. Bucket 1 is the single largest expected contributor (the whole 4.28 ms residual sits on top of 2104 candle ops), so the highest-leverage concrete next item is likely a **dispatch count reduction** effort — take the op graph apart and fold adjacent candle ops into fewer kernel boundaries.

**Finding #4 (the hf2q-vs-llama.cpp correctness floor is already in place).** Post-1bNEW.20.FIX, hf2q and llama.cpp agree byte-for-byte for 3094 bytes on the sourdough prompt (`scripts/sourdough_gate.sh` enforces ≥3094 as a mandatory pre-merge gate for every future speed item, with 1-byte safety margin under the measured 3095). The only real hf2q drift in 1000 decoded tokens is a single-letter case flip at decode token 840 (`' On'` vs `' ON'` in `"**Phase 1 (Lid On/ON):**"`), which is the ADR-line-268 residual f32-reduction-order drift already scoped as follow-up. **Correctness is Walk-ready**; the speed End gate is the remaining Walk blocker.

---

## Rejected Walk-CAPABILITY candidates (post-sweep)

The post-Walk re-spike (`docs/spike-post-walk-results.md:190-200`) proposed three remaining candidates. This spike partially re-classifies each against the measured behavior of HEAD:

### RUN-1 → Walk-CAPABILITY: Candle per-buffer wait semantics

**Original framing:** "Replace the pool-wide `flush_and_wait` with per-tensor wait using `MTLCommandBuffer::addCompletedHandler`, unblocking the 1bNEW.3 projected +3-6 tok/s that was measured-as-zero."

**Post-1bNEW.20 status:** **Dissolved.** 1bNEW.20 already unblocked this via a different mechanism — replacing `slice_scatter` in the KV-append path removed the implicit drain points that were serializing the windowed greedy decode. The full projected gain landed as part of the +26.83 tok/s 1bNEW.20 lift. There is no remaining "per-buffer wait" residual to recover; the decode path's pool-wide drain was never the bottleneck once the implicit drains behind copies were eliminated.

**Verdict: CLOSED. No follow-up item.**

### RUN-2 → Walk-CAPABILITY: Fuse RmsNorm into adjacent QMatMul

**Original framing:** "Fold the pre-o_proj norm / pre-router norm / pre-gate_proj norm into their adjacent QMatMul kernels, following mlx-lm's lazy-graph fusion pattern."

**Post-1bNEW.20 status:** **Contingent on the sticky-encoder work landing first.** With `norm_dispatches_per_token == 331` and each fused RmsNorm call at ~1 dispatch × 1.71 μs ≈ 0.57 ms/token, the theoretical ceiling of RmsNorm fusion is ~0.5 ms saved (if every norm folds into its adjacent matmul). That's ~4.5 tok/s, which IS meaningful, but it only fires at its full potential after the per-op CPU overhead is already reduced. Doing this first on the current candle substrate would deliver significantly less (dominated by the new kernel's own CPU-side binding cost, which is ~equal to the norm kernel's cost).

**Verdict: DEFER. Revisit after a dispatch-count-reduction item lands.**

### RUN-3 → Walk-CAPABILITY: Single-command-buffer-per-forward

**Original framing:** "Wrap a whole `Gemma4Model::forward` in one MTLCommandBuffer, dispatch all ~2192 ops on it, commit+waitUntilCompleted at the end. Projected savings: 1-2 ms/token."

**Post-1bNEW.20 status:** **Empirically falsified.** The `CANDLE_METAL_COMPUTE_PER_BUFFER=5000` sweep is precisely this pattern (one buffer per forward, ~2104 dispatches all in the same buffer, committed at the pool boundary). It regressed 85.3 → 79.4 tok/s (**−5.9**) because it eliminates the CPU/GPU overlap the 50-per-buffer default provides. Single-command-buffer-per-forward is **strictly worse** than candle's existing windowed commit pattern.

**Verdict: REJECTED. Do not pursue.**

---

## Proposed 1bNEW.21 candidates (fresh, post-sweep)

With the RUN-1/2/3 candidates disposed of, the remaining high-leverage Walk-CAPABILITY items are:

### Candidate A — Bump `CANDLE_METAL_COMPUTE_PER_BUFFER` default 50→100 (free, trivial)

Single-line change (or env var in the CLI wrapper). Measured **+0.6 tok/s** (85.3 → 85.9 median). Net gain ~0.7% of the 21.7 tok/s gap. Negligible individually, but zero-risk. Walk-correctness unchanged by construction (this changes only *when* commits fire, not *what* gets dispatched).

**Scope:** either (a) vendor a candle-metal-kernels patch at `commands.rs:14` `DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 100` (same vendor pattern as the 1bNEW.20.FIX candle-nn vendor), or (b) set the env var in hf2q's `main.rs` / `serve/mod.rs` init path via `std::env::set_var` gated to the `metal` feature.

**Walk classification:** neither llama.cpp nor mlx-lm exposes this exact knob. Not strictly a kernel port, but the resulting behavior (~100 dispatches per command buffer with overlap) more closely matches llama.cpp's ggml graph-scheduler submission pattern than candle's default. Walk-CAPABILITY by the outcome-based definition.

### Candidate B — Dispatch count reduction via candle op-graph flattening in hot paths (real work, largest expected lever)

**Target sites** (in decreasing order of dispatch cost / fusibility):

1. **MoE fan-in after fused down kernel** (5 ops → ~2): `w_total = top_k_weights * gathered_scale` → `unsqueeze` → `broadcast_mul` → `sum(1)` → `reshape` → `to_dtype`. Five candle ops per layer × 30 layers = 150 dispatches. Could be collapsed to a single weighted-sum kernel (one Metal kernel that reads the down_out 3D tensor, broadcasts the weights, sums over top_k, and writes the 2D output). Reference: llama.cpp's `ggml_mul_mat_id` output combine path at `ggml-cpu/ops.cpp:ggml_compute_forward_mul_mat_id` does this as a single loop. Estimated saving: 100 dispatches × 1.71 μs = **0.17 ms/token ≈ 1.5 tok/s**.

2. **MoE router top-k selection** (10 ops → ~3): `contiguous` → `arg_sort_last_dim` → `narrow` → `contiguous` → `gather` → `sum_keepdim` → `broadcast_div`. Seven+ ops × 30 layers = 210+ dispatches. Could be a single top-k + softmax kernel. Reference: mlx-lm has `mx.topk` as a single op; llama.cpp does top-k with `ggml_top_k`. Estimated saving: 150+ dispatches × 1.71 μs = **0.26 ms/token ≈ 2.3 tok/s**.

3. **Attention Q/K/V projection chain** (~8 ops → ~4): currently does q_proj → narrow/split/reshape → q_norm → rope → ... with several intermediate shape-rewrites. Some of these (the narrow → reshape → contiguous triads) are likely foldable. Estimated saving: 50-100 dispatches × 1.71 μs = **0.1-0.17 ms/token ≈ 0.9-1.5 tok/s**.

4. **Decoder layer residual / layer_scalar fusion** (4 ops → ~1): the `layer_scalar.broadcast_mul` followed by the residual add followed by the next norm's input could be one fused dispatch. Estimated saving: ~60 dispatches × 1.71 μs = **0.1 ms/token ≈ 0.9 tok/s**.

**Combined envelope:** ~400-500 dispatches eliminated out of 2104, ~0.6-0.7 ms/token saved, **~6 tok/s gained** if all four land cleanly. Takes `dispatches_per_token` from 2104 → ~1600-1700.

**Walk classification:** each of the four target sites has a reference in either llama.cpp or mlx-lm where the same logical operation is expressed as fewer-but-bigger kernels. Walk-KERNEL-PORT at the aggregate level; each sub-item would cite its specific `ggml-*.cpp` or `mlx-lm/models/*.py` reference in its task spec.

**Risk:** correctness — every new fused kernel needs the full Phase A unit-test battery plus the sourdough gate. Single-ULP drift at any of these sites could push the common prefix below 3094 and fail the gate.

### Candidate C — Sticky-encoder optimization in candle (speculative, larger leverage)

**Target:** `Commands::command_encoder` at `commands.rs:101-104` currently creates a fresh `ComputeCommandEncoder` on every call. On Metal, consecutive dispatches on the same command buffer can share an encoder until an encoder boundary is required (blit op, another encoder type, etc.). A "sticky encoder" version would return the existing encoder when the next request is for the same type and same buffer, saving the `computeCommandEncoder()` call itself plus the per-encoder arg-table binding.

**Estimated saving:** speculative, needs measurement. If per-encoder setup is ~0.5-1 μs (Metal's ~500 ns bind-overhead × a few bindings), savings at 2104 dispatches/token are **1.0-2.1 ms/token ≈ 9-19 tok/s**. This is the biggest potential lever but also the least certain — it requires Metal-level instrumentation to verify, and the candle patch is non-trivial (need to handle encoder lifecycle across buffer-swap boundaries, blit boundaries, and cross-thread safety).

**Walk classification:** both llama.cpp's ggml-metal and mlx-lm use shared encoders across multiple dispatches (llama.cpp via `ggml_metal_graph_compute_encoder_dispatch`, mlx-lm via its lazy graph executor). Walk-CAPABILITY port target.

**Risk:** medium-high. Candle's existing architecture assumes encoder-per-op; the refactor needs to preserve the mutex locking that the command pool currently relies on for thread-safety, and needs to handle the encoder.endEncoding() call correctly at buffer-swap boundaries.

---

## Recommendation

Ship **Candidate A** as 1bNEW.21 (it's free and measurable, closes 3% of the gap), then **Candidate B** as 1bNEW.22 (it's the largest Walk-faithful lift with clear kernel-port citations for each sub-site, closes ~30% of the gap if fully realized). Revisit **Candidate C** only after B lands — the dispatch-count reduction in B reduces the surface area where the sticky encoder win would apply, so C's measurement and risk are cleaner to evaluate after the easier dispatches are gone.

Combined: A+B ship ~6.5 tok/s (85.3 → ~91.8) with Walk-discipline correctness gates (new Phase A unit tests + sourdough prefix gate). If the post-B measurement shows a ~15 tok/s residual, C becomes the logical 1bNEW.23. If post-B we're close to 107, no C needed.

**Order of operations:**
1. **1bNEW.21** (Candidate A) — vendor candle-metal-kernels patch to set `DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER = 100`. Same vendor pattern as 1bNEW.20.FIX. Update `Cargo.toml` `[patch.crates-io]`. Re-run canonical bench + sourdough gate. Expected: 85.3 → 85.9 tok/s, sourdough gate 3095 bytes (unchanged by construction).
2. **1bNEW.22** (Candidate B) — implement the MoE fan-in fused combine kernel first (biggest sub-lever, cleanest reference). Phase A: unit test at strict max|Δ|=0 vs the unfused path. Phase B: bench + sourdough gate. Expected: ~+1.5 tok/s.
3. **1bNEW.22.B** (if needed) — MoE router top-k+softmax fused kernel. Same cadence.
4. **1bNEW.22.C** — attention projection chain flattening.
5. **1bNEW.22.D** — decoder residual + layer_scalar fusion.
6. **1bNEW.23** (if post-B residual ≥ 10 tok/s) — Candidate C sticky-encoder in vendored candle-metal-kernels.

Each item is individually Walk-verifiable against `scripts/sourdough_gate.sh` (mandatory ≥3094 byte common prefix vs llama.cpp on the DWQ GGUF) plus 5-run canonical bench (no regression below current median 85.3 tok/s unless explicitly documented).

---

## Measurement methodology notes (for future spikes)

This spike was **first-principles + empirical-sweep-only**; no Rust-level forced-sync instrumentation was added. The prior spike's `HF2Q_SPIKE_SYNC=<region>` pattern is still the gold standard when specific per-op timing numbers are needed, but it requires ~30 minutes of instrumentation + rebuild per region. The buffer-tuning sweep here gave enough signal to reject three candidates and narrow the remaining ones without that investment.

**When the next spike needs per-op wall-clock numbers:** re-apply the `HF2Q_SPIKE_SYNC` pattern from `docs/spike-post-walk-results.md:31-39` for the decode path, targeting specifically the MoE fan-in chain (5 ops), MoE router top-k (10 ops), and attention q/k/v preamble. Expected per-region numbers at the current 85.3 tok/s baseline: MoE fused ≈ 1.8 ms (bandwidth-bound floor), MoE non-kernel ops ≈ 0.5 ms, attention ≈ 1.3 ms, RmsNorm ≈ 0.5 ms, RoPE ≈ 0.3 ms, lm_head F16 ≈ 3.7 ms, all other (KV append, sampler drain, graph overhead) ≈ 3.6 ms.

**Sourdough correctness gate:** `scripts/sourdough_gate.sh` is the enforceable floor. Every candidate above must pass with ≥3094 byte common prefix before merging. The gate runs in ~30 seconds and is mandatory for every future speed PR.

---

## Citation trail

- Post-1bNEW.20.FIX baseline: HEAD `9deb18a` — `cargo build --release --features metal --bin hf2q` + 5-run canonical bench → 85.3 tok/s median, 85.5 p95.
- `metrics.txt` on HEAD: `dispatches_per_token: 2104.52`, `moe_to_vec2_count: 0.00`, `moe_dispatches_per_layer: 34.00`, `sampler_sync_count: 0.26`, `norm_dispatches_per_token: 331.00`.
- Candle command pool default: `/opt/candle/candle-metal-kernels/src/metal/commands.rs:14` (`DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 50`) and `:15` (`DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE: usize = 5`).
- Buffer tuning sweep: 5-run medians at thresholds 10, 20, 50 (default), 100, 200, 5000 — all identical prompt / binary / cold-start procedure.
- Sourdough correctness gate: `scripts/sourdough_gate.sh` committed in `9deb18a`; passes on HEAD at 3095-byte common prefix.
- Post-Walk re-spike (superseded by this doc for the Walk-CAPABILITY ranking): `docs/spike-post-walk-results.md`.
- Post-1bNEW.17 spike (orthogonal; lm_head analysis): `docs/spike-post-1bNEW17-results.md`.
- ADR-005 Walk-CAPABILITY re-classification (the "retroactive sharpened Walk/Run definition" that pre-classified RUN-1/2/3 as Walk-CAPABILITY): `docs/ADR-005-inference-server.md:933-938`.
- Sourdough investigation (weights artifact resolution): `docs/ADR-005-inference-server.md` Walk Exception Register entry for the 2026-04-11 "Sourdough hiragana glitch" investigation, plus this spike section.

---

## Return message (concise)

- HEAD baseline is 85.3 tok/s median (21.7 tok/s short of 107 target = 2.37 ms/token gap).
- Bandwidth floor is ~8.12 ms/token; hf2q is at 11.72 ms, llama.cpp is at 9.35 ms. The **2.37 ms gap is entirely non-bandwidth** (CPU dispatch overhead, op-graph cost, CPU/GPU overlap slack).
- Empirical sweep of `CANDLE_METAL_COMPUTE_PER_BUFFER` falsifies the "single-command-buffer-per-forward" hypothesis (RUN-3): going from 50 → 5000 regresses to 79.4 tok/s. Optimum is 100 at **+0.6 tok/s** free. Pool mechanics are near-optimal.
- RUN-1 (per-buffer wait) was dissolved by 1bNEW.20's removal of implicit drain points. RUN-2 (RmsNorm+QMatMul fusion) is small and should follow B. RUN-3 is REJECTED empirically.
- Proposed order: **1bNEW.21** = bump the compute-per-buffer default 50→100 (+0.6 tok/s, free). **1bNEW.22** = dispatch-count reduction via op-graph flattening at 4 hot sites (MoE fan-in, MoE top-k, attention preamble, decoder residual) for ~6 tok/s. **1bNEW.23** = sticky-encoder in vendored candle-metal-kernels, speculative ~10-15 tok/s, pursue only if post-22 residual justifies.
- Combined 21+22 envelope: 85.3 → ~92 tok/s (closes ~30% of remaining gap). Remaining residual after 22 decides whether 23 is needed.
- Correctness floor: `scripts/sourdough_gate.sh` is the mandatory pre-merge gate for all of the above. Passes on HEAD at 3095 common-byte prefix vs llama.cpp.
