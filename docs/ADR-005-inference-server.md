# ADR-005: Inference Server — OpenAI-Compatible API for hf2q

**Status:** In Progress (Phase 1 complete; Phase 1b in Walk replan as of 2026-04-10)  
**Date:** 2026-04-09 (original), 2026-04-10 (comprehensive Walk-discipline rewrite)  
**Decision Makers:** Robert, Claude

## Problem Statement

There is no pure-Rust pipeline that takes a HuggingFace model from download through quantization to inference serving. Existing tools (ollama, llama.cpp server, vLLM) are separate programs in C++/Python that each handle one piece. Users stitch them together manually, and developers can't extend or embed the pipeline in their own Rust applications.

hf2q already handles download → quantize. Adding inference completes the pure-Rust, single-binary pipeline: `hf2q serve --model google/gemma-4-27b-it` does everything. The work is structured as reusable crates so other Rust developers can use any layer (quantization, inference, serving) independently.

## Context

hf2q can now quantize HuggingFace models to llama.cpp-compatible GGUF format. The next step is serving these models via an OpenAI-compatible HTTP API, enabling use with Open WebUI and other clients. The goal is a complete pipeline: download → quantize → serve, matching or beating ollama/vLLM/llama.cpp server performance.

## Product Requirements

### Core Functionality
1. **OpenAI-compatible HTTP API** on user-specified port
   - `POST /v1/chat/completions` — streaming SSE + non-streaming
   - `GET /v1/models` — model listing
   - `POST /v1/embeddings` — embedding generation (pooled output from loaded model by default; optional `--embedding-model` flag for a dedicated embedding model)
   - `GET /health` — health check
   - `GET /metrics` — performance metrics
   - Tool calling / function calling support — template-driven, model-agnostic (tool schemas injected into prompt via the model's chat template)

2. **Multimodal: text + vision** (no audio for 26B MoE)
   - Base64 image data URIs (Open WebUI format)
   - Multiple images per message
   - OpenAI content parts format: `[{type: "text"}, {type: "image_url"}]`

3. **CLI one-shot mode** — `hf2q generate --model X --prompt "..." --image photo.jpg`

### Model Management
4. **Model input flexibility:**
   - `--model google/gemma-4-27b-it` → auto-download from HuggingFace + auto-quantize for hardware + serve
   - `--model ./models/gemma4/` → local safetensors directory
   - `--model ./model.gguf` → explicit GGUF file
   - Cache quantized models in `~/.cache/hf2q/`
   - Resumable HF downloads; quantization restarts from scratch if interrupted (design quantizer so resumability can be added later without rearchitecting)

5. **Chat templates** — priority order: CLI `--chat-template` override > GGUF metadata (`tokenizer.chat_template`) > `tokenizer_config.json` fallback

### Performance
6. **Speed target:** match or beat llama.cpp on the same hardware, verified with real benchmarks
   - QMatMul on quantized tensors (no dequantize-to-float)
   - If candle lacks Metal QMatMul kernels for specific K-quant types, we write our own Metal kernels
   - Metal GPU acceleration (primary)
   - Current baseline: 107 tok/s decode on M5 Max via llama.cpp with our GGUF (Q4_K_M, Gemma 4 26B MoE). Baseline is hardware-specific — re-measure on each target chip

7. **Concurrency:** continuous batching (vLLM-style) — batch multiple requests through the model simultaneously for maximum throughput

### Platform
8. **GPU backends:** Metal (primary), CUDA (second), Intel/AMD Vulkan/oneAPI (third)
   - Hard requirement: a supported GPU must be present. No GPU → refuse to start (not just slow — won't run)

9. **Logging:** `-v` verbose flag, request logs with tok/s, memory usage, queue depth

### Future Phases
10. Multi-model hot-swap (ollama-style)
11. Additional architectures: Qwen3, Mistral — use candle-transformers implementations when they meet our performance bar; write our own only for unsupported architectures or when targeting state-of-the-art for a specific model
12. CoreML/ANE backend — later phase requiring dedicated research to determine where ANE outperforms Metal GPU (coreml-native crate available). No size cutoff assumed; benchmark to find the real crossover point

## Architecture Decisions

### Inference Engine: candle + QMatMul
Use candle's quantized tensor primitives (`QTensor`, `QMatMul`) for direct inference on GGUF-loaded quantized weights. No dequantization step. candle provides Metal and CUDA kernels for quantized matmul. For any K-quant types where candle lacks Metal kernels, we write custom Metal compute kernels rather than falling back to dequantize.

The existing `serve/gemma4.rs` MoE implementation is the reference for the forward pass — it already handles dual head dims, SigMoE routing, tied K=V, and partial RoPE. Upgrade it to use QMatMul instead of dequantized Tensors.

### Vision Pipeline
Candle's `gemma4/vision.rs` provides the ViT encoder. The mmproj GGUF contains the vision tower weights. hf2q produces mmproj files from safetensors (quantizing the vision tower) AND accepts user-provided mmproj files. Pipeline:
1. Load image → preprocess (resize, normalize, patch)
2. ViT forward pass (from mmproj weights)
3. Multimodal embedder projects vision features → text dim
4. Replace `<image>` token positions in text embeddings
5. MoE text decoder forward pass

### API Layer: Restore Spec Layer, Rebuild Inference Integration
The prior OpenAI API implementation (deleted in commit fe54bc2) had 5,868 lines of production code. It was deleted during the pivot from MLX to candle/Metal. The spec-compliance layers (schema, SSE protocol, tool parsing) are engine-agnostic and should be restored. Anything that touched the inference path must be rewritten for candle + GGUF.

**Restore from git history** (engine-agnostic spec compliance):
- `schema.rs` (1,215 lines) — OpenAI request/response types
- `sse.rs` (681 lines) — Server-Sent Events streaming
- `tool_parser.rs` (867 lines) — tool call extraction
- `router.rs` — axum routes

**Note on middleware (2026-04-10):** the prior ADR text referenced a `middleware.rs` file; `git show fe54bc2 --stat` does not list `middleware.rs`. CORS and other middleware configuration previously lived inline inside `router.rs` / `mod.rs`. Treat middleware as **rebuild-from-scratch**, not restore.

**Rebuild from scratch** (was wired to MLX, needs candle):
- `handlers.rs` — new inference integration with candle QMatMul engine
- `mod.rs` — AppState with continuous batching scheduler, model loading
- Middleware (CORS, request logging) — re-create at the `router.rs`/`mod.rs` boundary; no file to restore

Deep review of the deleted code is required before restoring to confirm each file is truly engine-agnostic.

### Dependencies (already in Cargo.toml)
- `tokio` (full) — async runtime
- `axum` 0.7 — HTTP framework
- `tower-http` 0.5 — CORS middleware
- `uuid` — request IDs
- `futures`, `async-stream` — SSE streaming
- `tokenizers` 0.22 — tokenization

## Implementation Phases

### Phase 1: Text Inference (candle QMatMul) — COMPLETE (correctness, 2026-04-09)
- [x] Load GGUF with candle's quantized tensor system
- [x] Forward pass using QMatMul (no dequantize) for attention, MLP, and MoE router linear layers
- [x] MoE expert weights dequantized to BF16 and pre-sliced at load time (eliminates per-token narrow/squeeze/contiguous; quantized expert matmul deferred to 1b.1)
- [x] KV cache for efficient decode
- [x] `hf2q generate` CLI working, produces coherent correct output
- [x] 6 forward pass bugs fixed: GELU variant, MLP/MoE parallel, softmax routing, router input processing, per-expert scale, norm assignments
- [x] 2 GGUF norm name mismatches fixed (post_attention_norm, ffn_norm)
- [x] Chat template: uses GGUF Jinja-style control tokens (<|turn>, <|think|>, <turn|>)
- [x] Benchmarked: 18 tok/s decode (M5 Max) after GPU routing + batched MoE — llama.cpp does 107 tok/s
- [x] GPU-side top-k routing (arg_sort + gather on GPU, only 8 values pulled to CPU)
- [x] Batched expert matmuls (Tensor::stack + batched BF16 matmul, reduced per-token dispatch count)
- [ ] Speed gap: 23.8 tok/s current vs 107 tok/s target — deferred to Phase 1b (see below)

### Phase 1b: Speed Optimization (WALK REPLAN, 2026-04-10)

**Current benchmark result: 23.8 tok/s median** (M5 Max, Gemma 4 26B MoE Q6_K/Q8_0 DWQ, commit `0a703d7`, 5 runs with zero variance, coherent output). llama.cpp target: 107 tok/s on the same hardware and GGUF. Gap: ~4.5x remaining.

#### Crawl, Walk, Run discipline (2026-04-10)

This replan reframes Phase 1b around a Crawl/Walk/Run discipline. The earlier plan treated speed as the primary axis and innovated kernel-by-kernel toward numeric gates. That produced one silent correctness regression (commit `a0952e2` → gibberish; see bisect below) and a narrative that drifted from what the code actually does (see the 1b.6 "done vs PARTIAL" contradiction below). This rewrite hard-codes fidelity as the primary axis.

> **Crawl (done, commit `0a703d7`).** Produce coherent, correct output on quantized inference. Achieved at 13.8 → 23.8 tok/s, zero variance, with a known-good Paris baseline and a bisectable commit series.
>
> **Walk.** Port proven patterns from C/Python reference implementations — llama.cpp, mlx-lm, candle — to Rust. **No innovation.** Every Walk item must cite a specific `file:line` in one of those references that it is porting. Items without a working citation are Run territory and deferred. Phase 1b is Walk-only.
>
> **Run.** Write something neither llama.cpp nor mlx-lm does (novel kernels, novel algorithms, novel fusions). Deferred to a later phase. Not in Phase 1b.

**Rationale.** llama.cpp already hits 107 tok/s on the same hardware and the same GGUF. If we port its dispatch patterns faithfully, our math stays right and our speed approaches theirs. If we innovate before matching, we can't tell if a difference is a bug or a speedup. Crawl, Walk, Run — in order.

**Walk Exception Register.** Items where we deliberately diverge from the reference and document why:
- At Crawl (`0a703d7`): none. The baseline matches reference math modulo the fusions below.
- `forward_with_residual` (hf2q-original ADD→NORM fusion at `src/serve/gemma4.rs:47-51`, used at `:505`) un-fused in Phase 1b pre-flight to re-align with references. May be re-fused in Run if measured perf justifies it.
- Any future exception: add a row here with the reference it diverges from, the reason, and the conditions under which it would be revisited.

#### Canonical Benchmark Harness

- Model: Gemma 4 26B MoE Q4_K_M GGUF
- Hardware: M5 Max (report chip variant and memory)
- Prompt: 128 tokens (fixed test prompt, committed to repo)
- Generation: 128 tokens, greedy (temperature=0)
- Runs: 5 consecutive, report median tok/s and p95
- Metric: decode tok/s (exclude first-token latency)
- Tool: `hf2q generate --model <gguf> --prompt-file <test_prompt> --max-tokens 128 --temperature 0 --benchmark`

#### Progress discipline (2026-04-10, replaces the Tier-gate table)

There are no intermediate speed gates in Phase 1b. Each item is either a valid Walk port (with reference citation) or it's not. Speed is measured after each item and recorded in the bisect table as **trajectory data**, not as a pass/fail gate. The per-commit pass/fail gate is the token-match fixture (Design C Layer A; see Correctness Protocol below).

**Stop-and-diagnose rule.** If two consecutive items each deliver **<1 tok/s** improvement, pause work, dump `metrics.txt` (counters: `dispatches_per_token`, `moe_to_vec2_count`, `sampler_sync_count`, `norm_dispatches_per_token`), and diagnose what's in the critical path before starting the next item. Do not paper over a plateau by piling more items on top.

**End gate.** ≥107 tok/s decode on M5 Max Gemma 4 26B MoE Q4_K_M, measured via the canonical benchmark harness. Walk is complete when this is met **AND** the Design C Layer B fixture (llama.cpp reference divergence point) has not worsened.

#### Correctness Protocol (Design C layered fixtures, 2026-04-10)

The a0952e2 → gibberish regression cost a day. The fix is cheap, deterministic, and committed to the repo: golden token-id sequences that every commit must match exactly.

Three fixtures live under `tests/fixtures/`:

```
tests/fixtures/crawl_baseline.tokens       # hf2q HEAD greedy output, 128 tokens
tests/fixtures/llama_cpp_reference.tokens  # llama-cli greedy output on same GGUF
tests/fixtures/crawl_verified.meta         # commit SHAs, GGUF sha256, divergence point
```

**Layer A (per-commit, bisect-safe).** Token-for-token exact match against `crawl_baseline.tokens`. Every Phase 1b commit must reproduce this sequence byte-for-byte under `--temperature 0 --seed 42`. Fixture regeneration is allowed **only** when an item's math provably changes the tokens, and the justification must appear in that item's commit message.

**Layer B (milestone, drift tracking).** `llama_cpp_reference.tokens` records `llama-cli` greedy output on the same GGUF. We track the *divergence point* (first token index where hf2q and llama.cpp diverge) in `crawl_verified.meta`. The divergence point must not *worsen* across a milestone boundary — this catches slow numerical drift that Layer A alone would miss if we ever regenerate Layer A legitimately.

Layer B is re-checked at milestone boundaries, not on every commit. Layer A is the hot gate.

**Materialization.** Both layers are produced by a single script, `scripts/crawl_verify.sh`, which takes a `--commit` flag and writes all three fixtures atomically. Agent #3 is creating the script in parallel with this rewrite; this ADR is the authoritative spec.

**Red flags (immediate rollback, same as before):**
- Token repetition patterns: `own own own`, `the the the`, `\n\n\n\n` runs.
- Control-token floods: `<unk>`, `<pad>`, `<s>`, `</s>` mid-generation.
- Non-ASCII gibberish runs > 3 tokens.
- Output length < 5 tokens (model silently EOS'd).
- The exact a0952e2 failure signature: nonsense characters with no coherent reasoning structure.

**Any correctness regression:** stop, bisect, fix. No "I'll fix it in the next item."

#### Phase 1 Items — Completed Status (preserved from Crawl)

| Item | Status | Notes |
|------|--------|-------|
| 1b.1 | DONE | Per-expert QMatMul via candle's SIMD-optimized Metal kernels. Expert weights byte-sliced from 3D GGUF QTensor into 128 2D QMatMul objects per layer. Skips F32 dequantization entirely — experts stay quantized in GPU memory. |
| 1b.2 | N/A | Superseded by 1b.1 (no per-token Tensor::stack needed with QMatMul) |
| 1b.3 | DONE | Expert weights stay quantized as QMatMul (Q6_K gate_up + Q8_0 down, ~20 GB vs ~45 GB dequantized F32) |
| 1b.4 | DONE | F32 forward pass, ~690 GPU dtype casts eliminated |
| 1b.5 | DONE | 3 unnecessary `.contiguous()` removed |
| 1b.6 | INVESTIGATION ONLY | See rewrite below. **No code change landed.** 60 routing syncs/token + 1 sampler sync = 61 total still present in `0a703d7`. |
| 1b.7 | DONE | Pre-allocated KV cache with `slice_scatter`, auto-grow 2x. **Stride gotcha**: `slice_scatter` on dim ≠ 0 returns non-standard strides; requires `.contiguous()` on the narrow'd view before SDPA. |
| 1b.8 | DONE (decode) | SDPA vector path for decode (native GQA, no repeat_kv needed). Manual attention for prefill (candle's SDPA full kernel comment about the 32 KB threadgroup limit for head_dim=512 is now known stale; see 1bNEW.10). Requires `softcapping=1.0` (candle convention for disabled, NOT 0.0). |
| 1b.9 | DONE | Sliding window KV truncation — sliding layers only expose last 1024 tokens (correctness fix) |
| 1b.10 | DONE (to be un-fused in pre-flight) | `RmsNorm::forward_with_residual` at `gemma4.rs:47-51`, used at one site per layer (`:505`). See Walk Exception Register — to be un-fused at pre-flight item 0b. |
| 1b.11 | SKIPPED | 1 dispatch saved, not worth Metal plumbing |
| 1b.12 | N/A | Superseded by 1b.4 (F32 forward pass) |
| 1b.13 | N/A | Not needed — candle already batches dispatches; the real problem was forced `waitUntilCompleted` from `to_vec2()`, not commit frequency. |
| 1b.14 | DONE | Warmup dummy token at load (eliminates ~37 ms TTFT spike from Metal shader compilation) |
| 1b.15 | DONE | Batched MoE prefill: tokens grouped per expert, one QMatMul per active expert |
| 1b.16 | DONE (investigation) | Root causes identified: QMatMul kernel behavior at large dims, `repeat_kv` contiguous copy, sliding window not enforced. Fixes applied via 1b.8 and 1b.9. The 8192-dim QMatMul cliff remains an open empirical question (see 1bNEW.13). |
| Benchmark | DONE | `--benchmark` + `--prompt-file` flags, 5-run median/p95 reporting |

**1b.6 rewritten (2026-04-10):** *Investigation only; no code change landed in `gemma4.rs`.* The GPU-side top-k routing pipeline (`arg_sort` + `narrow` + `gather` + `broadcast_div`) was already present at the Phase 1 baseline `0a703d7` (see `gemma4.rs:420-426`). The two forced `to_vec2()` calls at `gemma4.rs:428-429` remain because the per-expert `Vec<QMatMul>` dispatch loop is CPU-driven; removing the syncs requires replacing the CPU loop with a GPU-batched MoE kernel — that work is now consolidated in the **Unified MoE kernel** item (1bNEW.1). **60 routing syncs/token** (two `to_vec2` × 30 layers) **+ 1 sampler sync = 61 total per token** remains the baseline. `git log --oneline -S"to_vec2" -- src/serve/gemma4.rs` confirms only `0a703d7` has ever touched those lines.

#### Correctness Regression Bisect (2026-04-10, preserved)

An initial Phase 1b implementation (commit `a0952e2`) reached 26.3 tok/s but produced **gibberish output** — token repetition, nonsense characters, no coherent reasoning. Speed without correctness is worthless. Systematic bisect from the baseline (commit `0a703d7`) applying one change at a time:

| Test | Change Added | Correct? | tok/s |
|------|--------------|----------|-------|
| 1 | Baseline only | ✓ | 13.8 |
| 2 | + F32 forward (1b.4) | ✓ | 8.9 |
| 3 | + QMatMul per-expert MoE (1b.1) | ✓ | 27.3 |
| 4 | + F32 + QMatMul | ✓ | 26.4 |
| 5 | + Pre-allocated KV cache (1b.7) | ✓ | 22.9 |
| 6 | + SDPA decode (naive port of 1b.8) | ✗ **gibberish** | — |
| 7 | + SDPA with `.contiguous()` fix | ✓ | 23.4 |
| 8 | + Sliding window (1b.9) | ✓ | 26.1 |
| 9 | + Fused norm+residual (1b.10) | ✓ | 26.0 |
| 10 | Final (all + benchmark harness) | ✓ | **23.8 median** |

**Root cause**: `Tensor::slice_scatter` on dim ≠ 0 internally does `transpose(0, dim).slice_scatter0().transpose(0, dim)`. After this sequence, the tensor has shape `[1, heads, seq, hd]` but its underlying memory is laid out as `[seq, heads, 1, hd]` contiguous. The strides become `[hd, hd, heads*hd, 1]` — position stride is `num_kv_heads * head_dim`, not `head_dim`. Candle's SDPA vector kernel reads keys with constant stride `BN * D` where `D = head_dim`, assuming positions are contiguous. Our `slice_scatter`'d KV cache violated the assumption; the kernel read garbage. **Fix**: `.contiguous()` on the narrow'd view before returning from `KvCache::append`.

This is a general lesson: when combining candle's Tensor ops with its raw Metal kernels, verify stride assumptions. "Fast views" may not match what hand-written kernels expect.

#### Walk Replan (2026-04-10)

> **Mantra (`~/Documents/mantra.txt`) — applies to every item below:**
> *DO NOT BE LAZY. We have plenty of time to do it right. No shortcuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.*
>
> Operating consequences for this plan under Walk:
> - **Every item cites a reference `file:line`.** No reference = not Walk = deferred.
> - **Every item has a Chesterton check** — WHY the current code exists before proposing to change it. Items without a Chesterton section are not executed.
> - **Every commit is token-matched** against `crawl_baseline.tokens`. No stubs, no "WIP", no "I'll fix it in the next item."
> - **No fallback.** If a port fails its correctness gate and cannot be made to pass, the item is abandoned, not half-landed.
> - **Crawl, Walk, Run** — in order. Innovation is Run, deferred.

##### 1. Gap Decomposition (42 ms/token → 9.3 ms/token)

Current: 23.8 tok/s median = **42.0 ms/token**. Target: llama.cpp 107 tok/s = **9.3 ms/token**. Gap: **32.7 ms/token, ~4.5x**.

Attribution (from direct reading of `src/serve/gemma4.rs` and `src/serve/sampler.rs`, rather than profiler output, because the profiler hooks themselves force GPU syncs and distort the measurement):

| Component | Estimated cost | Source / citation | Share of gap |
|---|---|---|---|
| MoE routing `to_vec2` syncs — 2 per layer × 30 layers = **60** forced `waitUntilCompleted`/token. The aggressive variant eliminates all 60; the conservative variant halves them (30). | 18–25 ms | `gemma4.rs:428-429`; each `to_vec2()` flushes the command buffer and blocks the CPU until GPU drains. | ~55–75% |
| Sequential per-expert CPU loop — 30 layers × 8 experts × ~13 ops/expert + **30** `Tensor::zeros`/token + **30** scalar `Tensor::new`/token (at decode, `num_tokens == 1`; `gemma4.rs:443, 459`) | 8–12 ms | `gemma4.rs:436-465`; 104 dispatches + 60 small GPU buffer allocations per token; no batching across experts, no pipelining across the `combined + ...` chain. | ~25–35% |
| RmsNorm fragmentation — ~11 RmsNorms per layer × **10** real F32 dispatches each = ~3,300 dispatches/token just for norms | 3–5 ms | `gemma4.rs::RmsNorm::forward` lines **32-43** manually chains `to_dtype`, `sqr`, `mean_keepdim`, `ones_like`, `affine`, `add`, `sqrt`, `recip`, `broadcast_mul`, `broadcast_mul` (and a trailing `to_dtype` that is a no-op at F32 steady-state but counts as a dispatch). | ~10–15% |
| Global attention Q/K/O projection QMatMul at 8192 dims (5 global layers × 4 ops) + partial RoPE overhead | 1–3 ms | `gemma4.rs:280-325`; 2x the GEMM size of sliding layers. 1b.16 called out the 8192-dim QMatMul cliff as unconfirmed. | ~3–10% |
| CPU overhead: ~7,000 lazy `Tensor` allocations + `Arc` refcounts per token | 3–8 ms | Counted from ops in `gemma4.rs`; at 23.8 tok/s that is ~166,000 Tensor alloc/drop/sec. | ~10–25% |
| Pure GPU kernel execution | ~10–14 ms | hf2q uses the same candle Metal QMatMul kernels as dense layers; llama.cpp's 9.3 ms is the compute floor, hf2q's compute cannot be dramatically slower. | — (near floor) |

**Headline:** the MoE CPU-driven dispatch path (routing syncs + sequential expert loop) is **60–90% of the gap**. Everything else is secondary.

Cross-reference. MLX-LM collapses the entire 8-expert dispatch to a single `affine_gather_qmm_*` Metal kernel invocation (`switch_layers.py:76-87`). llama.cpp collapses it to one `kernel_mul_mv_id_q6_K_f32` + one `kernel_mul_mv_id_q8_0_f32` per layer (`ggml-metal.metal:7624-7642`) by redirecting pointers inside the same base `kernel_mul_mv_q6_K_f32` used for dense layers. Neither of them does what hf2q currently does: 60 forced GPU→CPU round-trips and 240 sequential `QMatMul::forward` calls per token.

##### 2. Item Ordering (Walk, no tier gates)

Items are ordered by dependency and Walk safety. Each is either a valid port with a live citation or it is not executed.

**Pre-flight:**
- **1bNEW.0 — Metrics instrumentation.** Promoted from phantom to first item, satisfying Anti-Goal #11 (no items without a Chesterton section).
- **1bNEW.0b — Un-fuse `forward_with_residual`.** First code change. Re-aligns the layer with mlx-lm and llama.cpp references before anything else touches hot code.

**Walk items (reference-cited ports):**
- 1bNEW.1 — Unified MoE kernel (ports `kernel_mul_mv_id_*` from llama.cpp)
- 1bNEW.2 — Batched per-expert state hoist (candle idiomatic port)
- 1bNEW.3 — Single-sync sampler path (ports `mx.async_eval` pattern)
- 1bNEW.4 — Fused RmsNorm kernel (ports llama.cpp `kernel_rms_norm_fuse_impl` F=1/2/3)
- 1bNEW.6 — Fused RoPE kernel (ports llama.cpp `kernel_rope_norm` / `kernel_rope_neox`)
- 1bNEW.10 — BF16 prefill SDPA at head_dim=512 (uses candle's existing compiled kernel)
- 1bNEW.11 — llama.cpp flash-attn vec port (contingent escape for 1bNEW.10)
- 1bNEW.12 — Extended warmup (port of llama.cpp's pre-compiled `.metallib` approach via runtime warmup)
- 1bNEW.13 — 8192-dim QMatMul cliff (measurement-first)

**Dissolved / retired from the prior plan:**
- Old 1bNEW.5 (fused residual+RmsNorm) dissolves into **1bNEW.4** as kernel variant F=3, because llama.cpp's single kernel already handles it.
- Old 1bNEW.7 (fused mul_mv_id kernel) dissolves into **1bNEW.1** — it's the same work: you cannot remove the routing syncs without removing the CPU expert loop, and you cannot remove the CPU expert loop without the batched kernel.
- Old 1bNEW.8 (stride-aware Q/K prelude) dissolves into **1bNEW.6** — llama.cpp's `kernel_rope_norm/neox` is **stride-aware by construction** (source/dest strides in `ggml_metal_kargs_rope`), so a faithful port eliminates the `.contiguous()` calls at `gemma4.rs:113-114, 118-121` automatically.
- Old 1bNEW.9 (scalar-weight alloc removal) is entirely **subsumed by 1bNEW.1** (the fused kernel takes weights as a buffer).

Metrics instrumentation (required **before** any code item):

> Add to `mod.rs` so every benchmark run dumps `metrics.txt` with: `dispatches_per_token`, `moe_to_vec2_count`, `moe_dispatches_per_layer`, `sampler_sync_count`, `norm_dispatches_per_token`. These turn "stop and diagnose on plateau" into a mechanical check.

##### 3. Per-Item Detail

---

**1bNEW.0 — Metrics instrumentation (Pre-flight)**
- **What it does:** Add counter fields to `AppState` in `mod.rs` and increment sites at dispatch-issuance points. Emit `metrics.txt` alongside `bench.log` after every `--benchmark` run. Counters: `dispatches_per_token`, `moe_to_vec2_count`, `moe_dispatches_per_layer`, `sampler_sync_count`, `norm_dispatches_per_token`.
- **Why it helps:** Turns Progress-discipline checks and Stop-and-diagnose into mechanical assertions instead of "trust me." No counter instrumentation = no way to detect plateau = no discipline.
- **Correctness risk:** NONE. Counters are observe-only.
- **Validation plan:** After landing, run the canonical benchmark and verify `metrics.txt` exists and reports: `moe_to_vec2_count == 60`, `sampler_sync_count == 1`, `norm_dispatches_per_token ≈ 3300` for the current baseline. Commit these numbers as a fixture in `tests/fixtures/metrics_baseline.txt`.
- **Dependencies:** None.
- **Estimated LOC:** ~60 (instrumentation) + ~20 (emission).
- **Chesterton's fence:** No counters exist today because Phase 1 shipped without per-op instrumentation — the profiler is `Instant::now()` at the `generate()` boundary only. We are adding visibility, not replacing a hidden mechanism.
- **Reference citation:** N/A — pure infrastructure. Justified by Anti-Goal #11.

---

**1bNEW.0b — Un-fuse `forward_with_residual` (Pre-flight, Walk Exception unwind)**
- **What it does:** Inline `forward_with_residual` at `gemma4.rs:505` to the explicit two-op reference pattern: `xs = residual + attn_out; normed = pre_feedforward_layernorm.forward(&xs);`. Delete `RmsNorm::forward_with_residual` at `gemma4.rs:47-51`.
- **Why it helps:** Not a speed item. It restores Walk fidelity. Both references do this unfused:
  - `mlx-lm/models/gemma4_text.py:339-340`: `h = self.post_attention_layernorm(h); h = residual + h`
  - `llama.cpp/src/llama-model.cpp` (gemma4-iswa build_cb): `cur = build_norm(cur, attn_post_norm, ...); attn_out = ggml_add(ctx0, cur, inpL)`
- **Correctness risk:** LOW. The math is identical modulo FP associativity; the fused and unfused forms compute the same value at F32. If Layer A tokens diverge, that is a bug in the Crawl baseline, not in this item, and must be fixed before continuing.
- **Validation plan:** Layer A token match. Layer B drift check.
- **Dependencies:** 1bNEW.0.
- **Estimated LOC:** -5 (a deletion + 2-line inline).
- **Chesterton's fence:** `forward_with_residual` was added in the Phase 1 1b.10 item as a small dispatch reduction (ADD then NORM → one Rust call). Under Walk we re-align with the references first and only re-fuse in Run if measurement justifies it.
- **Reference citation:** mlx-lm `gemma4_text.py:339-340`; llama.cpp gemma4-iswa layer build.

---

**1bNEW.1 — Unified MoE kernel (ports llama.cpp `kernel_mul_mv_id_*`)**
- **What it does:** Replace the CPU-driven per-expert loop at `gemma4.rs:436-465` and the two forced `to_vec2()` syncs at `gemma4.rs:428-429` with a single GPU dispatch per projection that reads the GPU-resident `[num_tokens, top_k]` expert-index tensor directly. Retain an `Arc<QTensor>` of the 3D source expert weight at the layer level (alongside the existing per-expert `Vec<QMatMul>` fallback behind a feature flag). Output: `[num_tokens, top_k, out]`.
- **Why this is one item, not four:** The `to_vec2()` calls exist solely to feed the CPU expert loop; the loop cannot be removed without replacing it with a batched GPU kernel; removing the loop eliminates the `Tensor::zeros` / `Tensor::new` scalar allocations as a consequence. One mechanism — the CPU-driven dispatch — is responsible for all four symptoms. Fixing them in sequence is impossible; they are a single edit.
- **Why it helps:** Collapses 30 layers × 8 experts × 3 projections = 720 separate `QMatMul::forward` calls → **3 fused dispatches per layer = 90 total**. Eliminates 60 forced syncs/token (aggressive variant) or 30 (conservative). Eliminates 60 small GPU allocations per token at decode (`Tensor::zeros` at `gemma4.rs:443` + `Tensor::new` at `:459`, both inside `for tok_idx in 0..num_tokens` — at decode `num_tokens == 1`, so it is 1 per MoeBlock × 30 layers = 30 of each = 60 combined).
- **Expected speed effect:** Trajectory data only under Progress discipline. Estimated 18–25 ms saved from syncs + 4–8 ms saved from loop state hoist = 22–33 ms of the 32.7 ms gap. Recorded in bisect table, not as a gate.
- **Correctness risk:** **HIGH.** This is the exact area that produced the `a0952e2` gibberish. The GPU-side path must align expert weights and per-expert scales correctly; off-by-one in index ordering = silent gibberish.
- **Validation plan:** (1) **Phase A — numerical:** Rust test feeds known inputs, compares against running 8 separate `QMatMul::forward` calls; ε=1e-5 elementwise. (2) **Phase B — single layer:** replace MoE at layer 0 only via feature flag; Layer A token match; per-token logprob comparison for first 16 tokens. (3) **Phase C — all layers:** flip flag to all layers; Layer A token match; 5-run benchmark. (4) **Phase D — adversarial:** 2048-token prompt with specific-fact recall.
- **Dependencies:** 1bNEW.0, 1bNEW.0b. Spike Q1 (stride layout of the `ids` buffer) must be resolved first — **30 minutes**, not a blocker.
- **Estimated LOC:** ~80 Rust wrapper + **0** Metal (the kernel already exists in candle's compiled `quantized.metal`). No candle fork.
- **Chesterton's fence:** The per-expert `Vec<QMatMul>` exists because Phase 1 1b.1 landed it as the baseline that works today. We keep it as a feature-flagged fallback (`--moe-kernel=fused|loop`). The 3D source QTensor is already in memory (`gemma4.rs:620-621` loads it via `gguf.get_qtensor`, then `:645-652` slices per-expert) — we add an `Arc<QTensor>` field on the layer to retain the 3D view alongside the slices. The two `to_vec2()` calls are there solely because the CPU loop needs Rust-side indices; once the loop is gone, the syncs are gone.
- **Reference citations (all verified from code):**
  - llama.cpp `kernel_mul_mv_id` template at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:7545-7618`.
  - Instantiations `kernel_mul_mv_id_q4_K_f32`, `_q6_K_f32`, `_q8_0_f32` at `ggml-metal.metal:7624-7642`.
  - candle already compiles `quantized.metal` as a single library via `Source::Quantized` at `/opt/candle/candle-metal-kernels/src/kernel.rs:86-103`.
  - `Kernels::load_pipeline(device, Source::Quantized, "kernel_mul_mv_id_q6_K_f32")` will work as-is because `load_function` at `/opt/candle/candle-metal-kernels/src/kernel.rs:129-139` is a plain `library.get_function(name)` symbol lookup with **no allowlist**.
  - Rust wrapper pattern to mirror: `call_quantized_matmul_mv_t` at `/opt/candle/candle-metal-kernels/src/kernels/quantized.rs:25-176`. The new `call_quantized_matmul_mv_id_t` mirrors it with extra parameters `ids: &Buffer, nei0, nei1, nbi1` (see kernel signature at `ggml-metal.metal:7545`).
  - mlx-lm reference for the dispatch shape: `switch_layers.py:76-90` (`mx.gather_qmm`) at `/Users/robert/.pyenv/versions/3.13.12/lib/python3.13/site-packages/mlx_lm/models/switch_layers.py`.
- **Known unknown (Q1 spike):** the stride layout for the `ids` buffer. Spike: read `ggml-metal.m` in the llama.cpp source — search for `kernel_mul_mv_id` and inspect how it packs the ids buffer before dispatch. **30 minutes.**

---

**1bNEW.2 — Batched per-expert MoE forward: hoist per-token state (subsumed by 1bNEW.1 at decode)**
- **What it does:** Placeholder for the `Tensor::zeros` / `Tensor::new` removal. At decode, this is entirely absorbed by 1bNEW.1 (no CPU loop → no per-iteration allocations). At prefill, when the fused kernel is dispatched per expert in the batched path from 1b.15, the same hoist applies.
- **Why it helps:** 60 GPU buffer allocations/token eliminated at decode (already counted in 1bNEW.1's gain estimate).
- **Correctness risk:** LOW. Pure algebraic identity.
- **Validation plan:** Inherits 1bNEW.1's Phase A–D gates.
- **Dependencies:** 1bNEW.1.
- **Estimated LOC:** 0 at decode (subsumed); ~20 for the prefill path delta.
- **Chesterton's fence:** `Tensor::zeros` at `gemma4.rs:443` exists because the old (pre-1b.4) code accumulated F32 outputs from BF16 inputs; post-1b.4 the whole pipeline is F32, so the zero-buffer reset is vestigial. `Tensor::new(&[w], device)` at `gemma4.rs:459` exists because `broadcast_mul` with an `f32` scalar isn't a direct candle API — but `affine(w as f64, 0.0)` is.
- **Reference citation:** candle idiomatic API — `candle_core::Tensor::affine` docs.

---

**1bNEW.3 — Single-sync sampler path (speculative argmax enqueue)**
- **What it does:** For greedy decode (T=0), restructure `sampler.rs:49` so the `argmax + to_scalar` doesn't sync after every token. Use candle's command-pool batching to enqueue the argmax alongside the next token's embed lookup, syncing only every N tokens (e.g., every 4). When the sync fires, retrieve N u32 tokens at once. Valid at T=0 because the forward pass is deterministic.
- **Why it helps:** After 1bNEW.1 collapses routing syncs, the sampler's 1 per token becomes a meaningful fraction. Ports the MLX pattern.
- **Expected speed effect:** 1–3 tok/s, free after 1bNEW.1.
- **Correctness risk:** LOW. At T=0 with `repetition_penalty == 1.0`, deterministic. With `repetition_penalty != 1.0` we need the previous token before computing the next, so the fast path must gate on `(temperature == 0.0 && repetition_penalty == 1.0)`; otherwise per-token sync.
- **Validation plan:** Layer A token match (must be bitwise identical — same ops, different commit boundary). 5-run benchmark variance must stay zero.
- **Dependencies:** 1bNEW.1.
- **Estimated LOC:** ~30 across `sampler.rs` and the decode loop in `mod.rs`.
- **Chesterton's fence:** The current sampler pattern exists because the decode loop is sequential at the API level. But at the GPU-dispatch level, candle's lazy command pool can enqueue forward pass N+1 before sync N resolves — as long as we haven't called `to_scalar` yet. Streaming SSE still works as long as we sync before the SSE writer iterates.
- **Reference citation:** MLX `mx.async_eval` usage pattern in `mlx-lm/models/base.py` (async evaluate in `generate_step`); see also llama.cpp's per-command-buffer commit pattern in `ggml-metal.m`.

---

**1bNEW.4 — Fused RmsNorm Metal kernel (ports llama.cpp `kernel_rms_norm_fuse_impl`, F=1/2/3)**
- **What it does:** Port llama.cpp's single fused RmsNorm kernel, which has **three template modes** selected by parameter `F`:
  - **F=1:** `y = x * scale` — plain RmsNorm (no weight).
  - **F=2:** `y = (x * scale) * f0` — RmsNorm with weight (the common case; all 10 norm sites per layer other than the residual-norm pair).
  - **F=3:** `y = (x * scale) * f0 + f1` — RmsNorm with weight + post-norm residual add.
  Replaces the 10-dispatch manual chain at `gemma4.rs:32-43`. Float4 variants from llama.cpp are ported alongside for SIMD throughput.
- **Subtlety on F=3 applicability (important):** F=3 computes **NORM→ADD** (normalize first, then add residual). hf2q's former `forward_with_residual` was ADD→NORM (which is why 1bNEW.0b un-fuses it). After that un-fuse, the only site in `DecoderLayer::forward` that is *actually* NORM→ADD is `gemma4.rs:521-522` — `let combined = post_feedforward_layernorm.forward(&combined)?; let xs = (residual + combined)?;`. **That site — and only that site — matches F=3.** One site per layer × 30 layers = 30 F=3 invocations per token. Everywhere else uses F=2.
- **Why it helps:** ~11 RmsNorms per layer × 10 dispatches each = ~3,300 dispatches/token just for norms. Fused kernel = 1 dispatch each = **~3,000 dispatches saved** + ~60 command-buffer commits saved.
- **Expected speed effect:** 4–8 ms saved (trajectory; not a gate).
- **Correctness risk:** LOW-MEDIUM. RmsNorm variance computation is numerically delicate; the port must match the manual version's reduction order to stay within the token-match envelope. Use two-pass reduction or Welford as the reference kernel does.
- **Validation plan:** (1) Rust unit test comparing fused vs manual on random `[1, 128, 2816]` inputs at ε=1e-5. (2) Bisect: replace one RmsNorm site at a time, Layer A token match per site. (3) Full benchmark.
- **Dependencies:** 1bNEW.0, 1bNEW.0b.
- **Estimated LOC:** ~80 Rust wrapper + ~50 Metal (port of llama.cpp source, including the F=3 variant and the float4 variants).
- **Chesterton's fence:** The 10-op manual decomposition exists because candle 0.10.2 has no exposed fused F32 RmsNorm on Metal (`candle-nn::ops::rms_norm` is not in the public `Source` enum for Metal). We are replacing manual Tensor composition, not a hidden-but-optimized path. The new kernel is upstreamable to candle later.
- **Reference citations:**
  - llama.cpp `kernel_rms_norm_fuse_impl` template at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:2981-3050`.
  - float4 variants and F=1/2/3 instantiations at `ggml-metal.metal:3044-3050`.
- **Target counter after commit:** `norm_dispatches_per_token ≤ 330` (330 fused sites × 1 dispatch each = 330).
- **Note on dissolved items:** Old 1bNEW.5 (fused residual+RmsNorm) no longer exists — its F=3 form is a template variant of **this** kernel, not a separate item.

---

**1bNEW.6 — Fused RoPE kernel (ports llama.cpp `kernel_rope_norm` / `kernel_rope_neox`)**
- **What it does:** Replace the 9-op `rope_apply` at `gemma4.rs:133-140` (narrow×2, broadcast_mul×4, sub, add, cat — 9 total) and the partial-RoPE split/cat path at `gemma4.rs:107-130` with a single Metal kernel port. Two flavors: `kernel_rope_norm` for standard RoPE and `kernel_rope_neox` for the NeoX convention. Both take source/dest strides via `ggml_metal_kargs_rope` — **the kernels are stride-aware by construction**.
- **Why this subsumes the old 1bNEW.8:** Because the ported kernel is stride-aware, the `.contiguous()` calls at `gemma4.rs:113-114, 118-121` become unnecessary — a faithful port eliminates them automatically. There is no separate "stride-aware Q/K prelude" item; it is a *consequence* of porting 1bNEW.6 faithfully. The "~30 Metal delta" the old plan attributed to stride awareness was already baked into the reference kernel.
- **Why it helps:** ~22 RoPE-related dispatches per attention-layer × 30 layers = ~660 RoPE dispatches per token. Fusing to 2 per layer (one Q, one K) = ~600 dispatches saved, plus the eliminated `.contiguous()` copies.
- **Expected speed effect:** 3–6 ms saved (trajectory).
- **Correctness risk:** MEDIUM. Partial RoPE is exactly where the `project_coherence_bug.md` regression lived. The kernel must correctly handle the pass-through tail (positions `[rotary_dim, head_dim)` are not rotated). Both variants ported verbatim from llama.cpp with no algorithmic deviation.
- **Validation plan:** (1) Unit test vs existing `rope_apply` on `[1, 16, 1, 256]` and `[1, 16, 1, 512]` inputs with `partial_rotary_factor=0.5`, ε=1e-5. (2) Layer-by-layer Q/K output diff vs baseline for first 4 decode tokens at ε=1e-6. (3) Layer A token match.
- **Dependencies:** 1bNEW.4 (proves the custom-kernel integration pattern).
- **Estimated LOC:** ~60 Rust + ~80 Metal (ported from llama.cpp, no delta beyond the port).
- **Chesterton's fence:** The current 9-op implementation exists because candle 0.10.2 does not expose a fused RoPE on Metal. The 9 ops are memcpy + elementwise on disjoint halves — the exact pattern the llama.cpp kernel already absorbs. The `.contiguous()` calls exist because the *current* implementation uses `Tensor::narrow` views that the current `cat` path can't consume; the ported kernel reads strided memory directly.
- **Reference citations:**
  - llama.cpp `kernel_rope_norm` at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:4323` (f32 and f16 instantiations at `:4583-4584`).
  - llama.cpp `kernel_rope_neox` at `ggml-metal.metal:4376`.
  - Stride-aware arg struct `ggml_metal_kargs_rope` in `ggml-metal-impl.h`.

---

**1bNEW.10 — BF16 prefill SDPA at head_dim=512 (unblocks the existing candle kernel)**
- **What it does:** In `Attention::forward` prefill branch (`gemma4.rs:308-318`), cast Q/K/V to BF16 before calling `candle_nn::ops::sdpa`, cast output back to F32. **The current code's comment that "candle's SDPA full kernel exceeds 32 KB threadgroup mem for head_dim=512" is stale.** candle `sdpa.rs:86-94` selects reduced tiles `(bq=8, bk=8, wm=1, wn=1)` for `BD=512` in f16/bf16, totaling **24.1 KB** threadgroup memory. The only blocker is F32 input (rejected at `sdpa.rs:87-92`). The kernel variant `steel_attention_bfloat16_bq8_bk8_bd512_wm1_wn1_maskbfloat16` is already compiled at `candle/candle-metal-kernels/src/metal/scaled_dot_product_attention.metal:2334-2337`.
- **Why it helps:** Eliminates the manual `repeat_kv` + matmul + softmax + matmul fallback for global attention prefill (~10–12 dispatches/layer + 5 MB of temporary `repeat_kv` allocations per global layer). Replaces with one fused SDPA call.
- **Expected speed effect:** Decode: +0–1 tok/s (decode already uses vector SDPA at f32). Prefill: 3–5x faster on global attention layers, improving TTFT and long-prompt throughput.
- **Correctness risk:** MEDIUM. F32 prefill was the conservative choice from 1b.4. BF16 cast at the SDPA boundary only (decode stays F32) is narrower but still a numerical change.
- **Validation plan:** (1) Compare post-softmax attention weights BF16 vs F32 manual on a 128-token prefill at ε=1e-3. (2) Compare next-token top-5 distribution. (3) Layer A token match at a 128-token prompt prefix. (4) Adversarial: 2048-token document with specific-fact recall, output must match.
- **Dependencies:** None — orthogonal to 1bNEW.1/2/3/4/6.
- **Estimated LOC:** ~20.
- **Chesterton's fence:** The manual path was added when `head_dim=512` SDPA was *genuinely* unsupported. Candle has since added the reduced-tile variant; the comment was not updated. We're removing a stale workaround, not a functional safeguard. If BF16 correctness fails, escalate to 1bNEW.11 — **never** add a CPU fallback.
- **Reference citation:** candle `sdpa.rs:86-94`, `scaled_dot_product_attention.metal:2334-2337`.

---

**1bNEW.11 — Port llama.cpp `kernel_flash_attn_ext_vec_f16_dk512_dv512` (contingent escape for 1bNEW.10)**
- **What it does:** Port llama.cpp's flash-attn vec kernel for head_dim=512 at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:7165` with f32 accumulation. `FATTN_SMEM ≈ 3.5 KB` for the vec path (fits trivially in 32 KB). Kernel template parameterized on `(DK, DV)` so a 512 instantiation is direct.
- **Why it helps:** Same as 1bNEW.10 but without BF16 cast risk. Matches llama.cpp's known-fast path exactly.
- **Expected speed effect:** Same as 1bNEW.10.
- **Correctness risk:** HIGH. ~600 LOC of new Metal in unfamiliar template machinery.
- **Validation plan:** ε=1e-3 vs F32 manual; Layer A token match; 5-run benchmark.
- **Dependencies:** Only executed if 1bNEW.10 fails its correctness gate.
- **Estimated LOC:** ~600.
- **Chesterton's fence:** This is the contingency. The baseline manual path stays as the final fallback only if *both* 10 and 11 fail — and "fails" means the token fixtures say so, not feel.
- **Reference citation:** llama.cpp `ggml-metal.metal:7165` (`kernel_flash_attn_ext_vec_f16_dk512_dv512`).

---

**1bNEW.12 — Extended warmup: compile prefill PSOs at model load**
- **What it does:** Extend the existing single-token warmup at `mod.rs:231-234` with a *second* warmup pass using a short prefill sequence (8 tokens). Triggers Metal PSO compilation for prefill-specific kernel variants (SDPA full, prefill-path softmax, mask construction, prefill MoE batched matmul). Cleanup via `clear_kv_cache()` after.
- **Why it helps:** candle compiles Metal libraries from MSL source at runtime; first prefill call cold-compiles prefill PSOs (~37–100 ms TTFT spike). llama.cpp ships a pre-compiled `.metallib`; we can't, but we can pre-warm. Critical for serve mode where TTFT matters.
- **Expected speed effect:** Decode unchanged. **TTFT −37 to −100 ms** on first request after model load. Satisfies the no-VW-cheating principle (every code path must be fast, including the first request).
- **Correctness risk:** LOW. Warmup runs throwaway input.
- **Validation plan:** Measure TTFT on a fresh process for an 8-token prompt; must be < 150 ms with extended warmup.
- **Dependencies:** 1bNEW.10 (so the warmup pass exercises the new fused SDPA path, not the stale manual fallback).
- **Estimated LOC:** ~15.
- **Chesterton's fence:** The single-token warmup was added in 1b.14 to eliminate the shader-compile spike on decode. It only covers decode because the benchmark hot path is decode. Serve-mode TTFT was not the benchmark; we extend, not replace.
- **Reference citation:** llama.cpp `.metallib` precompilation (build-time, not runtime port); candle runtime compilation behavior in `candle-metal-kernels/src/kernel.rs`.

---

**1bNEW.13 — QMatMul 8192-dim cliff: profile and conditionally fix**
- **What it does:** 1b.16 noted a suspected performance cliff in candle's QMatMul kernel at 8192 output dim (q_proj/o_proj for global attention layers). This item is the empirical measurement: use Metal System Trace to time `kernel_mul_mv_q6_K_f32` at `[2816] → [8192]` and compare to the dense case `[2816] → [2816]`. If the cliff is real, options: (a) supply a different threadgroup configuration; (b) split into two `[2816] → [4096]` calls with a concat; (c) upstream a fix to candle.
- **Why it helps:** If the cliff is real, 5 layers × 4 projections = 20 calls/token, potentially 1–3 ms wasted.
- **Expected speed effect:** 0 if no cliff; up to 3 ms if real.
- **Correctness risk:** Depends on the fix. Threadgroup config change is correctness-safe; split-and-concat is not (concat overhead may eat the gain).
- **Validation plan:** Measure first. Then if a fix exists, Layer A token match.
- **Dependencies:** None.
- **Estimated LOC:** ~0–50 depending on outcome.
- **Chesterton's fence:** 1b.16 raised this as a hypothesis. Do **not** apply a speculative fix before the measurement.
- **Reference citation:** candle `kernel_mul_mv_q6_K_f32` in `candle-metal-kernels/src/metal_src/quantized.metal` — measurement only; any fix becomes its own Walk citation.

---

##### 4. Anti-Goals (Explicitly Rejected, with Reasoning)

These are direct consequences of the mantra and of Walk discipline. Each has been considered and rejected — not skipped, rejected with reasoning. Do not relitigate during execution.

1. **Custom scalar-dequant Metal kernel (1 thread per output element).** REJECTED. Already attempted as `moe_expert_q4k.metal` (Phase 1 1b.1 "Failed approach"), benchmarked slower than candle's SIMD QMatMul. 1bNEW.1 reuses candle's compiled `kernel_mul_mv_id_*` or is abandoned — no scalar-path fallback.
2. **Move routing back to CPU-computed indices.** REJECTED. The pre-`0a703d7` path did this and was slower. 1bNEW.1 moves the entire dispatch forward, not backward.
3. **Switch the whole forward pass to BF16.** REJECTED. Phase 1 1b.4 showed F32 saves ~690 dtype casts/token because candle's QMatMul requires F32 input. Only 1bNEW.10 converts select boundaries (global attention prefill SDPA).
4. **CPU fallback for head_dim=512 SDPA prefill.** REJECTED — violates `feedback_gpu_everything.md`. If 1bNEW.10 fails, the escape is 1bNEW.11 (port llama.cpp flash-attn), NOT a CPU path.
5. **"Try it and see" without a reference citation.** REJECTED — Walk discipline. Every Walk item cites `file:line` in llama.cpp, mlx-lm, or candle. Items without citations are Run and deferred.
6. **Benchmark-tune (VW cheating).** REJECTED — direct mantra violation. Every kernel must be correct for arbitrary prompts, arbitrary `num_tokens`, arbitrary seq lengths. No hardcoded fast path for the 128-token canonical prompt. Verify after every item by running a 1-token prompt AND a 512-token prompt AND the canonical prompt through the *same* code path.
7. **Stub code or "TODO later" placeholders.** REJECTED — direct mantra violation. Every commit lands a complete, correct, token-matched change.
8. **Try to replicate candle's `moe_gemm_gguf`** (`candle-transformers/src/fused_moe.rs`). REJECTED. It is CUDA-only in candle 0.10.2 (`candle-nn/src/moe.rs:339` — `#[cfg(not(feature = "cuda"))]` bails). Writing a Metal equivalent **is** 1bNEW.1.
9. **Fork candle as a default.** REJECTED. 1bNEW.1's citations confirm the kernels are reachable via candle's existing `Kernels::load_pipeline` + `Source::Quantized`, with no allowlist blocking the symbol lookup. No fork.
10. **Move to MLX-affine quantization.** REJECTED — hf2q produces K-quant GGUF and must stay ecosystem-compatible. MLX-LM is a reference, not a target.
11. **Skip the Chesterton check on any item.** REJECTED. Every item documents WHY the current code exists before proposing to change it. This is why 1bNEW.0 (metrics) is first.
12. **Escape-hatch ship at 75–85 tok/s ("worth shipping").** REJECTED (2026-04-10). Walk is **107 tok/s or not Walk**. The End gate is the End gate. If 1bNEW.1 fails, we abandon it and re-plan in Run, not half-ship.
13. **Disable the bisect table.** REJECTED. The `a0952e2` regression is recent. Every item extends the bisect table in this ADR.
14. **Parallelize implementation across many agents.** REJECTED for *implementation*. Research was valuable as a swarm (this plan is the proof). The actual code changes land one item at a time with a fixture checkpoint per item.
15. **Intermediate speed gates (old Tier 1/2/3/4 ≥45/≥70/≥95/≥107).** REJECTED (2026-04-10). They were born from "speed-targets-drive-architecture" thinking. Under Walk, every item either ports something reference-citable or it's not Walk — speed comes from fidelity, not gates. Replaced with the Progress-discipline / Stop-and-diagnose rule above.

##### 5. Open Questions (Require Empirical Spike Before Dependent Items Begin)

Each question is a measurement task. Spike duration is bounded; results go into the ADR before code changes.

1. **Q1 — Stride layout of the `ids` buffer for `kernel_mul_mv_id_*`.** The only remaining unknown for 1bNEW.1. Spike: read llama.cpp's caller in `ggml-metal.m` (search for `kernel_mul_mv_id`) and inspect how it packs the ids buffer (strides `nbi0`, `nbi1` in the arg struct). **30 minutes.** Not a blocker — it's a read-only spike that produces a one-line answer.
2. **Q2 — Can a downstream `.metal` source be compiled at runtime via `Device::new_library_with_source` and dispatched through candle's existing command encoder without ordering hazards?** Required for the llama.cpp F=3 RmsNorm port in 1bNEW.4 if we end up compiling fresh Metal rather than hitting an existing candle symbol. Spike: trivial `increment_by_one.metal` between two candle ops; check ordering and 50-ops-per-buffer batching applies. **Half a day.**
3. **Q3 — Does candle's command pool flush a partial buffer on forced sync, wasting pipelining slots?** Affects the gain estimate of 1bNEW.1 — if candle stalls more than expected, the gain is larger. Instrument `Commands::flush_and_wait` at `candle-metal-kernels/src/metal/commands.rs:176-202`. **1 hour.**
4. **Q4 — BF16 prefill correctness on Gemma 4: does it produce identical top-k tokens as F32 for long prompts?** 2048-token prompt with adversarial recall (a specific date/name at position 100, query at position 2000); compare top-5 BF16 vs F32. **30 minutes.** Blocks 1bNEW.10 commit; on regression, escalate to 1bNEW.11.
5. **Q5 — Is the 8192-dim QMatMul cliff real?** (1bNEW.13.) Metal System Trace of `kernel_mul_mv_q6_K_f32` at `[2816] → [8192]` vs `[2816] → [2816]`. **Half a day.**
6. **Q6 — Does candle's `arg_sort_last_dim` Metal kernel handle n=128 in a single threadgroup?** Read `candle-metal-kernels/src/kernels/sort.rs`. **1 hour.** Affects whether 1bNEW.1's GPU-side top-k path has hidden overhead.
7. ~~**Q7**~~ **CLOSED (2026-04-10).** `per_expert_scale_cpu` is **runtime-used** at `gemma4.rs:446`: `let w = top_k_weights_cpu[tok_idx][k] * per_expert_scale_cpu[eid]`. The CPU copy is a deliberate cache to avoid a per-token GPU→CPU sync on `self.per_expert_scale`. 1bNEW.1 must therefore gather from the **GPU** `per_expert_scale` tensor when it removes the CPU loop; the CPU cache is deleted along with the loop. See Resolved Questions.
8. **Q8 — Does the candle SDPA full kernel for `bd=512` accept the mask dtype hf2q's prefill currently builds?** The pre-instantiated mask dtypes are `{bool, f16, bf16}`. **30 minutes.** Blocks 1bNEW.10.

##### 6. Risks and Escape Hatches

**Risk #1: 1bNEW.1 causes a silent correctness regression like a0952e2.**
- *Likelihood:* MEDIUM-HIGH. The a0952e2 bug was a stride assumption in `slice_scatter`, exactly the class of issue that can hide in a GPU-only rewrite of routing.
- *Impact:* Another day-plus bisect; cascading delays.
- *Mitigation:* Correctness Protocol with `crawl_baseline.tokens` committed. Bisect-safe incremental commits: first the fused-kernel wiring behind a feature flag for layer 0 only; then for all layers. Each substep is a separate commit, bisectable in seconds.
- *No escape hatch to half-ship.* Per Anti-Goal #12, if 1bNEW.1 cannot be made correct + fast, it is abandoned and the work re-plans in Run.

**Risk #2: Q2 fails → 1bNEW.4 / 1bNEW.6 cannot load downstream `.metal` sources.**
- *Likelihood:* LOW. 1bNEW.1 already establishes that candle's `Kernels::load_pipeline` resolves any symbol in the compiled `quantized.metal` library, and `Device::new_library_with_source` is a documented candle-metal API path.
- *Impact:* MEDIUM — 1bNEW.4/6 would need to stage their kernels into candle's own source tree via upstream PRs before execution.
- *Mitigation:* Q2 spike runs before 1bNEW.4. If it fails, the items are blocked on upstream candle work and re-prioritized behind 1bNEW.10/11.

**Risk #3: Tier-gate replacement drifts into "no discipline at all".**
- *Likelihood:* LOW if the Stop-and-diagnose rule is enforced.
- *Mitigation:* The metrics counters from 1bNEW.0 are the mechanical check. Two consecutive <1 tok/s items → pause → dump `metrics.txt` → diagnose. Documented, measurable, not a vibe.

**Risk #4: Fused kernels introduce numerical drift below the token-match envelope.**
- *Likelihood:* LOW. These are well-understood reductions; candle and llama.cpp both have correct precedents.
- *Impact:* MEDIUM. Output quality regression even if Paris gate passes — caught only by adversarial prompts.
- *Mitigation:* Unit tests at ε=1e-5 vs manual BEFORE any benchmark run. Two-pass reduction or Welford if needed. Layer A is exact token match — silent numerical drift shows up as a fixture diff.
- *Escape hatch:* Per-site bisect — replace one norm site at a time, Layer A per site. If a specific site regresses, leave it on the manual path.

### Phase 2: HTTP Server (2026-04-10 clarifications)
- Restore spec-layer code (schema, SSE, tool parsing, router) from git history after deep review. **Middleware is not restorable** — `git show fe54bc2 --stat` does not list `middleware.rs`; CORS/logging middleware lived inline in `router.rs`/`mod.rs` and must be re-created rather than restored.
- Rebuild handlers and AppState from scratch for candle inference.
- **Continuous batching scheduler (vLLM port).** The vLLM scheduler lives in `vllm/core/scheduler.py` class `Scheduler` (specific `file:line` to be pinned once Phase 2 begins; treat as a port, not an original design). Walk discipline applies: we port their continuous-batching state machine, not invent one.
- **Concurrency target (2026-04-10):** serve concurrent requests at ≥60% of single-stream tok/s at N=4 concurrent, ≥40% at N=8, with throughput still scaling between N=8 and N=16 before saturating. These numbers are provisional and will be revalidated once Phase 1b End gate (107 tok/s single-stream) is met.
- `hf2q serve --model X.gguf --port 8080`.

### Phase 3: Vision (2026-04-10 clarifications)
- Image preprocessing pipeline (image crate).
- hf2q produces mmproj GGUF from safetensors (vision tower quantization).
  - **mmproj quant format:** F16 (de facto standard for vision towers). We do not Q4/Q6 the vision tower unless a later measurement shows a specific model tolerates it.
- Also accepts user-provided mmproj files.
- Load mmproj GGUF for vision tower.
- ViT forward pass + multimodal embedding injection.
- **Reference implementation to port the multimodal embedder projection from:** candle-transformers `gemma4/vision.rs` (preferred — already pure Rust and candle-native). If a specific op is missing, fall back to porting from HuggingFace `modeling_gemma4.py`. Pick one and commit in the item that lands Phase 3.
- **Accuracy gate:** hf2q vision output matches mlx-lm's Gemma 4 vision output on a canonical set of 5 standard prompts × 5 images (token-match for the first 50 generated tokens at T=0). Fixtures committed alongside `crawl_baseline.tokens`.
- `--image` flag for CLI, base64 for API.

### Phase 4: Auto Pipeline (2026-04-10 clarifications)
- `hf2q serve --model google/gemma-4-27b-it` → download + quantize + serve.
- **Hardware detection → static quant selection rule** (decision table, provisional thresholds; refined by measurement before Phase 4 ships):

  | Available GPU/unified memory | Quant type |
  |------|------|
  | ≥ 64 GB | Q8_0 |
  | 32 – 64 GB | Q6_K |
  | 16 – 32 GB | Q4_K_M |
  | 8 – 16 GB | Q3_K_M |
  | < 8 GB | refuse with a clear error naming the minimum supported config |

  The table lives in code (`src/serve/quant_select.rs`) and is unit-tested against synthetic `GpuInfo` fixtures. Thresholds can be tuned; the *rule shape* (static table, VRAM-indexed, documented) is committed.
- **Cache policy:** `~/.cache/hf2q/` keyed by `{model-id}/{quant-type}/{sha256}`. Re-download on HuggingFace safetensors sha256 mismatch OR on explicit `hf2q cache clear`. Otherwise offline mode uses the cached quantized GGUF indefinitely.
- **Integrity check:** compare sha256 of downloaded safetensors shards against the HuggingFace-published hashes from the model card's `safetensors` metadata; refuse to quantize on mismatch.

### Phase 5: Multi-Model + Architectures (2026-04-10 clarifications)
- Hot-swap model loading (ollama-style).
- **Hot-swap algorithm:** LRU pool of loaded models, memory-bounded, ollama-compatible semantics. Reference for the pool management pattern: ollama `llm/memory.go` and `llm/server.go`. The pool supports up to **N = 3** cached loaded models; eviction is LRU, bounded by a memory ceiling of **80% of system unified memory** (configurable).
- Qwen3, Mistral model support — prefer candle-transformers when they meet our performance bar; own implementations only when needed.
- Model discovery from cache directory.
- **GGUF provenance detection:** check metadata for hf2q origin; serve any valid GGUF regardless of who made it. On provenance match, apply hf2q-specific optimizations:
  1. Skip safetensors re-validation (trust our own hashes recorded in the GGUF).
  2. Assume quant type is one we wrote/verified the kernels for.
  3. Skip tokenizer re-download (hf2q embeds the tokenizer it used into GGUF metadata).
  4. Skip the mmproj integrity check if the mmproj was produced in the same `hf2q` run (recorded as a paired hash in metadata).

## Acceptance Criteria

### Phase 1: Inference Engine (Correctness) — DONE
- [x] `hf2q generate --model ./models/gemma4/ --prompt "..."` produces correct, coherent text output
- [x] All K-quant types produced by hf2q load and infer correctly via candle QMatMul (attention, MLP, router)
- [x] Primary model: Gemma 4 26B MoE

### Phase 1b: Speed Optimization (Walk)
- [ ] Decode speed matches or exceeds llama.cpp on the same hardware (≥107 tok/s on M5 Max, Gemma 4 26B MoE, Q4_K_M), measured using the canonical benchmark harness
- [ ] Prefill speed matches or exceeds llama.cpp on the same hardware
- [ ] MoE expert matmul runs on quantized weights directly via a fused `kernel_mul_mv_id_*`-style dispatch (no CPU-driven per-expert loop)
- [ ] Every Phase 1b commit token-matches `tests/fixtures/crawl_baseline.tokens` exactly
- [ ] At each milestone boundary, the divergence point vs `tests/fixtures/llama_cpp_reference.tokens` has not worsened
- [ ] Multi-shape sweep: 1-token, 128-token, 512-token, and 2048-token prompts produce coherent output through the same code path (no benchmark-only specialization)
- [ ] `metrics.txt` at End gate shows: `moe_to_vec2_count == 0`, `sampler_sync_count ≤ 1`, `norm_dispatches_per_token ≤ 330`, `moe_dispatches_per_layer ≤ 4`

### Phase 2: HTTP Server
- [ ] OpenAI Python/JS SDK clients connect and work (chat completions, streaming, tool calling, embeddings)
- [ ] Open WebUI connects and provides full multi-turn chat experience (streaming, vision, tool use)
- [ ] Continuous batching handles concurrent requests without serializing to one-at-a-time
- [ ] Concurrency target met: ≥60% single-stream tok/s at N=4, ≥40% at N=8, throughput still scaling through N=16
- [ ] **Primary embeddings path:** `POST /v1/embeddings` returns pooled output from the loaded chat model (no `--embedding-model` required) — verified against a committed set of reference embeddings for 5 canonical inputs
- [ ] `--embedding-model` flag loads a separate model for `/v1/embeddings` — verified against the secondary path

### Phase 3: Vision
- [ ] `hf2q generate --model ./models/gemma4/ --prompt "describe this" --image photo.jpg` produces correct image-aware output
- [ ] Vision accuracy gate: hf2q matches mlx-lm's Gemma 4 vision output on 5 prompts × 5 images (token-match, first 50 tokens, T=0)
- [ ] mmproj produced by hf2q is F16 and loads in both hf2q and llama.cpp

### Phase 4: Auto Pipeline
- [ ] `hf2q serve --model google/gemma-4-27b-it` on a fresh machine: downloads, auto-quantizes for detected hardware, starts serving — zero manual steps
- [ ] Subsequent runs use `~/.cache/hf2q/` (offline mode works for previously cached models)
- [ ] Hardware detection selects quant per the static table; unit tests cover every threshold boundary
- [ ] Safetensors sha256 integrity check refuses to quantize on mismatch
- [ ] `hf2q cache clear` invalidates the cache entry for a model

### Phase 5: Multi-Model + Architectures
- [ ] Hot-swap between two cached GGUFs in under 10 seconds, measured on M5 Max
- [ ] Cached pool holds up to 3 loaded models with LRU eviction bounded by 80% of system unified memory (configurable)
- [ ] Qwen3 / Mistral decode speed matches llama.cpp on the same hardware within ±5% (same Walk bar as Gemma 4)
- [ ] GGUF provenance path: loading an hf2q-produced GGUF skips the four optimizations listed in Phase 5 above; loading an external GGUF runs full validation
- [ ] Fast follow: Qwen3.5 added under the same Walk bar (moved from Phase 3, 2026-04-10)

### Cross-Cutting
- [ ] No GPU detected → hard error with clear message naming supported backends
- [ ] All inference, quantization, and serving code is pure Rust (no C++ deps)
- [ ] Work is structured as reusable crates (quantization, inference engine, API server)

## Resolved Questions
- **QMatMul kernel gaps:** We write custom Metal kernels for any K-quant types candle doesn't cover. No dequantize fallback.
- **Speed target:** Match or beat llama.cpp on same hardware, verified with real benchmarks.
- **No-GPU policy:** Hard fail — refuse to start without a supported GPU (Metal, CUDA, or Vulkan/oneAPI).
- **Code restore strategy:** Spec-layer files (schema, SSE, tool parsing, router) restored after deep review; inference-path code rebuilt from scratch (old code was MLX-based); middleware rebuilt from scratch (no `middleware.rs` in the fe54bc2 deletion).
- **Concurrency model:** Continuous batching from day one, not a simple queue. Port of vLLM's scheduler.
- **Vision mmproj:** hf2q produces mmproj (F16) AND accepts user-provided files.
- **Chat template priority:** CLI > GGUF metadata > tokenizer_config.json.
- **Embeddings:** Pooled output from loaded model by default; `--embedding-model` for dedicated embedding model. Both paths have acceptance tests.
- **Tool calling:** Template-driven, model-agnostic — schemas injected via chat template.
- **Resumable quantization:** Restart from scratch for now; design for future resumability.
- **Model implementations:** Use candle-transformers when they meet our performance bar. Write our own only for unsupported architectures or state-of-the-art optimization on specific models.
- **CoreML/ANE:** Keep as a later phase. Requires dedicated research to find where ANE beats Metal GPU — no arbitrary size cutoff. coreml-native crate is available when ready.
- **GGUF provenance:** Detect hf2q origin via GGUF metadata. Serve any valid GGUF regardless of who made it. Apply hf2q-specific optimizations (trusted metadata, skip extra validation) when provenance is confirmed.
- **MoE kernel quant format (2026-04-09):** Write ggml K-quant dequant kernels, not MLX affine. Rationale: hf2q produces K-quant GGUF; llama.cpp proves K-quants are fast on Metal; switching formats would break ecosystem compatibility.
- **SDPA scale omission (2026-04-09):** Gemma 4 intentionally uses `scaling = 1.0` (no `1/sqrt(head_dim)`). Per-head Q/K RmsNorm normalizes dot-product magnitudes, making the traditional scale unnecessary. Verified against HuggingFace `modeling_gemma4.py`.
- **Phase 1 vs 1b boundary (2026-04-09):** Phase 1 = correctness. Phase 1b = performance.
- **1b.6 scope (2026-04-10):** Investigation only, no code change landed. 60 routing syncs/token + 1 sampler sync = 61 total remain. Fixing them requires removing the CPU expert loop, which requires the unified MoE kernel (1bNEW.1). The prior ADR narrative ("done — investigation + fix") and status table ("PARTIAL, 30 syncs/layer remaining") were both wrong; code verification (`git log -S"to_vec2" -- src/serve/gemma4.rs`) shows only `0a703d7` touched those lines.
- **Q7: `per_expert_scale_cpu` lifecycle (2026-04-10):** Runtime-used, not load-time-only. `gemma4.rs:446` reads it per token: `let w = top_k_weights_cpu[tok_idx][k] * per_expert_scale_cpu[eid]`. The CPU copy is a deliberate cache to avoid a per-token GPU→CPU sync on `self.per_expert_scale`. Under 1bNEW.1 the CPU copy is deleted entirely and the weight is gathered from the GPU `per_expert_scale` tensor inside the fused dispatch. Q7 moved out of Open Questions.
- **Crawl/Walk/Run discipline (2026-04-10):** Phase 1b is Walk-only. Every item cites a reference `file:line`. Innovation is Run, deferred.
- **Tier-gate removal (2026-04-10):** Replaced with Progress discipline and Stop-and-diagnose on plateau. The only per-commit gate is Layer A token match; the only End gate is ≥107 tok/s.
- **Walk Exception Register (2026-04-10):** `forward_with_residual` is the only current exception, to be un-fused in Phase 1b pre-flight item 1bNEW.0b.
