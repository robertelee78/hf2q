# ADR-005: Inference Server — OpenAI-Compatible API for hf2q

**Status:** In Progress (Phase 1 complete)  
**Date:** 2026-04-09  
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
- `router.rs` + `middleware.rs` — axum routes + CORS

**Rebuild from scratch** (was wired to MLX, needs candle):
- `handlers.rs` — new inference integration with candle QMatMul engine
- `mod.rs` — AppState with continuous batching scheduler, model loading

Deep review of the deleted code is required before restoring to confirm each file is truly engine-agnostic.

### Dependencies (already in Cargo.toml)
- `tokio` (full) — async runtime
- `axum` 0.7 — HTTP framework
- `tower-http` 0.5 — CORS middleware
- `uuid` — request IDs
- `futures`, `async-stream` — SSE streaming
- `tokenizers` 0.22 — tokenization

## Implementation Phases

### Phase 1: Text Inference (candle QMatMul) -- COMPLETE (correctness, 2026-04-09)
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
- [ ] Speed gap: 18 vs 107 tok/s — deferred to Phase 1b (see below)

#### Phase 1b: Speed Optimization (IN PROGRESS — 23.8 tok/s correct, target 107 tok/s; see "Phase 1b Remaining Plan" below for the 4.5x closure plan)

**Profiled decode breakdown (per layer, single token, M5 Max, Gemma 4 26B MoE Q4_K_M, 30 layers):**

Note: "profiled" timings include `to_scalar` GPU sync per component (~7ms overhead total). The 18 tok/s baseline is CPU-loop throughput without forced GPU sync; actual GPU-complete throughput is unknown without Metal instrumentation.

| Component | Time | % | vs llama.cpp target (0.31ms/layer) |
|-----------|------|---|-------------------------------------|
| Attention (sliding) | 0.5-0.6ms | 19% | ~2x over budget |
| Attention (global, every 6th) | 5-8ms | — | 16-26x over budget; see 1b.16 |
| Dense MLP | 0.3-0.4ms | 13% | ~1x (at target) |
| MoE experts | 1.5-2.5ms steady / up to 8ms spike | 60-70% | 5-8x over budget; #1 bottleneck |
| Rest (norms, residual) | 0.3ms | 9% | ~1x |
| **Total per layer (steady-state)** | **~2.6ms** | | ~80ms/30 layers = ~12 tok/s (profiled), 18 tok/s (unprofiled) |

- Target: llama.cpp ~0.31ms/layer = 9.3ms/30 layers = 107 tok/s
- **Gap is ~8x per layer. MoE is the biggest single target, but attention and overhead also need reduction. No silver bullet — need 10-15 optimizations.**

**Approach: measure → change → measure. Profile after every optimization using the canonical benchmark (see below).**

**Canonical Benchmark Harness:**
- Model: Gemma 4 26B MoE Q4_K_M GGUF
- Hardware: M5 Max (report chip variant and memory)
- Prompt: 128 tokens (fixed test prompt, committed to repo)
- Generation: 128 tokens, greedy (temperature=0)
- Runs: 5 consecutive, report median tok/s and p95
- Metric: decode tok/s (exclude first-token latency)
- Tool: `hf2q generate --model <gguf> --prompt-file <test_prompt> --max-tokens 128 --temperature 0 --benchmark`

**Tier Gates:**
- After Tier 1: must reach ≥40 tok/s decode before starting Tier 2 (validates MoE kernel + core overhead reductions)
- After Tier 2: must reach ≥70 tok/s decode before starting Tier 3 (validates per-op overhead elimination)
- After Tier 3+4: target ≥100 tok/s decode (parity with llama.cpp)
- If a tier's gate is not met, investigate root cause before proceeding — do not skip tiers

#### Tier 1: MoE dispatch (biggest single bottleneck, 60-70% of decode)

**1b.1: Per-expert QMatMul via candle's optimized Metal kernels** (highest impact)
- Load each expert's weights as a `QMatMul` object (128 per layer, stays quantized)
- For decode: call `qmatmul.forward()` for each of the top_k=8 selected experts using candle's SIMD-optimized Metal kernels (same kernels used for dense layers)
- For prefill: batch tokens per expert, call `qmatmul.forward()` on the batch
- Eliminates F32 dequantization at load time; expert weights stay as Q6_K/Q8_0 in GPU memory (~20 GB vs ~45 GB dequantized)

**Key learning from llama.cpp study (2026-04-09):** llama.cpp's `kernel_mul_mv_id` is a thin wrapper that redirects pointers into the merged expert weight buffer and delegates to the SAME optimized `kernel_mul_mv_q6_K_f32` used for dense layers. The SIMD optimization is in the base matmul kernel, not the MoE dispatch. Writing custom scalar dequant kernels (attempted and reverted) was fundamentally wrong — the right approach reuses existing optimized matmul kernels.

**Failed approach (reverted):** A custom Metal shader (`moe_expert_q4k.metal`) with scalar dot-product dequant was written but benchmarked slower than candle's built-in QMatMul. Scalar dot product (1 thread per output element) cannot compete with SIMD-optimized matmul (2 simdgroups × 32 threads per row). The custom kernel also required handling multiple quant formats (Q4_K, Q6_K, Q8_0), non-256-aligned dimensions, and atomic accumulation — complexity without payoff.

**Future optimization: fused mul_mat_id kernel** — once the per-expert QMatMul approach is validated and benchmarked, a fused kernel (llama.cpp-style single dispatch for all experts, grid Z = n_experts) could eliminate the 8 separate kernel launches per layer. This is only worth doing after the QMatMul baseline proves the remaining gap is in dispatch overhead, not in the matmul itself.

**1b.2: Pre-allocate MoE expert weight stacks** (medium impact, done)
- Replaced per-token `Tensor::stack` with pre-stacked tensors + `index_select`
- **Note:** Superseded by 1b.1's QMatMul approach which doesn't need pre-stacked F32 tensors

**1b.3: Keep expert weights quantized** (memory + load speed, comes with 1b.1)
- Expert weights stored as QMatMul objects (~20 GB quantized) instead of dequantized F32 (~45 GB)
- Model load time reduced from ~3 minutes to ~30 seconds (no per-expert dequant + transpose + contiguous)

#### Tier 2: Reduce per-op overhead across all layers

**1b.4: Run entire forward pass in F32** (medium impact, done)
- Changed `MODEL_DTYPE` from BF16 to F32, eliminating ~690 GPU dtype cast dispatches per token
- candle's QMatMul Metal kernels require F32 input and output F32 — the BF16 round-trips were pure waste
- Trade-off: 2x activation memory bandwidth, but vastly fewer kernel dispatches

**1b.5: Reduce unnecessary `.contiguous()` calls** (small impact, done)
- Audited all contiguous calls in forward path: removed 3 unnecessary GPU memcpy, kept 9 with justification

**1b.6: Eliminate MoE routing GPU syncs** (high impact, done — investigation + fix)
- **Finding:** 35 GPU syncs per decode token, 34 from MoE routing `to_vec2()` calls (one per layer)
- Each sync is a full `waitUntilCompleted` pipeline stall — estimated 17-34ms wasted per token
- **Fix:** GPU-side routing via `index_select` + `gather` for combined weights, passed as GPU Tensors to Metal kernel. Reduces to 1 irreducible sync (sampler argmax)
- **1b.13 (command buffer batching) confirmed unnecessary** — candle already batches 50 dispatches per command buffer; the problem was the forced flushes from `to_vec2()`, not commit frequency

#### Tier 3: Attention and KV cache

**1b.7: Pre-allocated KV cache** (medium impact, done)
- Pre-allocate `[1, num_kv_heads, 4096, head_dim]` buffers at model load, write via `slice_scatter`
- Eliminates per-token GPU allocation; auto-grows at 2x if sequence exceeds buffer
- `reset()` is O(1) — just resets position counter, no GPU deallocation

**1b.8: SDPA fused attention** (medium impact, done for decode)
- Decode (seq_len=1): uses `candle_nn::ops::sdpa()` with native GQA — no repeat_kv expansion needed. This is the fused Metal SDPA vector kernel.
- Prefill (seq_len>1): manual attention with repeat_kv. Candle's SDPA "full" kernel exceeds 32KB threadgroup memory limit for head_dim=512 (global attention layers). **TODO:** Write tiled prefill attention kernel that handles head_dim=512 within memory budget.
- **Gemma 4 specifics:** scale=1.0 (Q/K are RmsNorm'd), softcapping=1.0 (candle convention for "disabled" — 0.0 would cause division by zero)

**1b.9: Sliding window KV cache truncation** (correctness fix + perf, done)
- Sliding layers now expose only the last `sliding_window` (1024) tokens from KV cache
- Previously both sliding and global layers attended to full history — a correctness bug
- KvCache `sliding_window: Option<usize>` parameter controls truncation per layer

#### Tier 4: Fused kernels

**1b.10: Fused RmsNorm + residual add** (small impact, done)
- `RmsNorm::forward_with_residual()` combines residual add + normalize in one pass
- Applied at 2 sites per decoder layer = ~60 fewer GPU dispatches per forward pass

**1b.11: Fused GELU + element-wise multiply in MLP/MoE** (small impact, skipped)
- Saves only 1 dispatch out of ~120 per forward pass — not worth the Metal plumbing
- The fused op already exists for MoE in the (reverted) custom kernel; can be extracted later if needed

**1b.12: F16 intermediates** (superseded by 1b.4)
- The F32 forward pass decision makes F16 intermediates moot

#### Tier 5: Infrastructure

**1b.13: Command buffer batching** (not needed)
- candle already batches up to 50 dispatches per command buffer before auto-commit
- The real bottleneck was forced `waitUntilCompleted` syncs from MoE routing (fixed in 1b.6)

**1b.14: Warmup dummy token at model load** (TTFT improvement, done)
- Runs dummy BOS token through model at load time to force Metal shader compilation
- Eliminates ~37ms first-token latency spike from global attention layers

**1b.15: Batched MoE prefill** (prefill speed, done)
- Groups tokens by expert and dispatches batched matmuls (one per active expert)
- Reduces GPU dispatches from `num_tokens * 2` to `num_active_experts * 2` per layer

**1b.16: Global attention 10x spike** (investigation complete)
- **Root causes identified:**
  1. QMatMul kernel performance cliff at 8192 dims (q_proj/o_proj for global layers)
  2. `repeat_kv` forces full contiguous copy of expanded KV cache (8x for global layers)
  3. Sliding window was not enforced (fixed in 1b.9)
- **Fixes applied:** 1b.9 (sliding window), 1b.8 (SDPA eliminates repeat_kv for decode)
- **Remaining:** QMatMul kernel profiling at 8192 dims to confirm if it's a candle kernel issue

#### Phase 1b Implementation Status (2026-04-10)

**Current benchmark result: 23.8 tok/s median** (M5 Max, Gemma 4 26B MoE Q6K/Q8_0 DWQ, 5 runs with zero variance, coherent output). Up from baseline 13.8 tok/s (~1.7x). llama.cpp target: 107 tok/s. Gap: ~4.5x remaining.

| Item | Status | Notes |
|------|--------|-------|
| 1b.1 | DONE | Per-expert QMatMul via candle's SIMD-optimized Metal kernels. Expert weights byte-sliced from 3D GGUF QTensor into 128 2D QMatMul objects per layer. Skips F32 dequantization entirely — experts stay quantized in GPU memory. |
| 1b.2 | N/A | Superseded by 1b.1 (no per-token Tensor::stack needed with QMatMul) |
| 1b.3 | DONE | Expert weights stay quantized as QMatMul (Q6K gate_up + Q8_0 down, ~20 GB vs ~45 GB dequantized F32) |
| 1b.4 | DONE | F32 forward pass, ~690 GPU dtype casts eliminated |
| 1b.5 | DONE | 3 unnecessary `.contiguous()` removed |
| 1b.6 | PARTIAL | MoE routing still does 1 `to_vec2()` sync per layer (30/token). GPU-only routing path deferred until fused mul_mv_id kernel lands. |
| 1b.7 | DONE | Pre-allocated KV cache with `slice_scatter`, auto-grow 2x. **Critical gotcha found**: slice_scatter on dim ≠ 0 uses a transpose trick that produces non-standard strides — returned view has position stride = `num_kv_heads * head_dim` instead of `head_dim`. Requires `.contiguous()` on the narrow'd view before SDPA can read it correctly. |
| 1b.8 | DONE (decode) | SDPA vector path for decode (native GQA, no repeat_kv needed). Manual attention for prefill (candle's SDPA full kernel exceeds 32KB threadgroup mem for head_dim=512). Requires `softcapping=1.0` (candle convention for disabled, NOT 0.0). |
| 1b.9 | DONE | Sliding window KV truncation — sliding layers only expose last 1024 tokens (correctness fix) |
| 1b.10 | DONE | `RmsNorm::forward_with_residual()` fuses residual add into norm, ~60 fewer dispatches per forward pass |
| 1b.11 | SKIPPED | 1 dispatch saved, not worth Metal plumbing |
| 1b.12 | N/A | Superseded by 1b.4 (F32 forward pass) |
| 1b.13 | N/A | Not needed — candle already batches dispatches; the real problem was forced `waitUntilCompleted` from `to_vec2()` (addressed by 1b.6 direction) |
| 1b.14 | DONE | Warmup dummy token at load (eliminates ~37ms TTFT spike from Metal shader compilation) |
| 1b.15 | DONE | Batched MoE prefill: tokens grouped per expert, one QMatMul per active expert (was num_tokens × 2 dispatches, now num_active_experts × 2) |
| 1b.16 | DONE | Root causes identified: QMatMul kernel behavior at large dims, repeat_kv contiguous copy, sliding window not enforced. Fixes applied via 1b.8 (SDPA eliminates repeat_kv for decode) and 1b.9 (sliding window). |
| Benchmark | DONE | `--benchmark` + `--prompt-file` flags, 5-run median/p95 reporting |

#### Correctness Regression Bisect (2026-04-10)

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

**Root cause**: `Tensor::slice_scatter` on dim ≠ 0 internally does `transpose(0, dim).slice_scatter0().transpose(0, dim)`. After this sequence, the tensor has shape `[1, heads, seq, hd]` but its underlying memory is laid out as `[seq, heads, 1, hd]` contiguous. The strides become `[hd, hd, heads*hd, 1]` — position stride is `num_kv_heads * head_dim`, not `head_dim`.

Candle's SDPA vector kernel (`scaled_dot_product_attention.metal`) reads keys with constant stride `BN * D` where `D = head_dim`, assuming positions are contiguous in memory. Our slice_scatter'd KV cache violated this assumption, so the kernel read garbage from between-position gaps in the pre-allocated buffer.

**Fix**: Call `.contiguous()` on the narrow'd view before returning from `KvCache::append`. This copies only the active portion per decode step (small: `num_kv_heads * current_len * head_dim` elements) while preserving the fast slice_scatter write path on the full buffer.

This is a general lesson: when combining candle's Tensor ops with its raw Metal kernels, verify the stride assumptions hold. Ops that return "fast views" may not match what hand-written kernels expect.

#### Phase 1b Remaining Plan (2026-04-10): 23.8 → 107 tok/s — MERGED v2

> **Mantra (`~/Documents/mantra.txt`) — applies to every item below:**
> *DO NOT BE LAZY. We have plenty of time to do it right. No shortcuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.*
>
> Operating consequences for this plan:
> - **No shortcuts.** Every item has a Chesterton check, a cited source, and a validation plan. Items without all three are not executed.
> - **Measure 3x, cut once.** Open Questions §6 must be resolved (with measured numbers) before the items that depend on them begin. No "let's try it and see."
> - **No fallback.** No CPU fallback path, no benchmark-only fast path, no stub code that says "TODO later." If a kernel can't be built correctly, the item is abandoned and the escape hatch in §7 takes over — we do not ship a half-fast intermediate.
> - **No stubs.** Every commit lands a complete, correct, benchmarked change with passing correctness gates. No "WIP" merges to main.
> - **Chesterton's fence.** Every item documents WHY the current code exists before proposing to change it. The 1b.6 PARTIAL note is the cautionary tale: it claimed "30 syncs irreducible" without actually checking — Agent #5 found 60 syncs and that they're entirely eliminable. Read first; rewrite second.
> - **Pure excellence the entire time.** No "we'll fix it in the next item" — the a0952e2 lesson is that small regressions compound into undiagnosable ones. Stop, bisect, fix.

This plan is the result of a 7-agent research swarm: Agent #2 (llama.cpp MoE hot path), Agent #3 (MLX-LM MoE reference), Agent #4 (candle 0.10.2 internals), Agent #5 (hf2q decode profiling by direct code reading), Agent #6 (prefill / warmup / head_dim=512), Agent #7 (system-architect synthesis), Agent #1 (queen / cross-synthesis + ADR write-up). All numbers traceable to cited source files; estimates are marked.

The first cut of this plan was written by the queen alone (Agent #7 was thought to have bailed); when Agent #7's synthesis arrived later it surfaced four items the queen had missed (1bNEW.3 sampler-batching, 1bNEW.6 fused RoPE, 1bNEW.8 stride-aware Q/K prelude, 1bNEW.12 extended prefill warmup) plus a metrics-instrumentation requirement. This merged version is the union — best item from each agent, deduplicated.

##### 1. Gap Decomposition (42 ms/token → 9.3 ms/token)

Current: 23.8 tok/s median = **42.0 ms/token**. Target: llama.cpp 107 tok/s = **9.3 ms/token**. Gap: **32.7 ms/token, ~4.5x**.

Attribution (from direct reading of `src/serve/gemma4.rs` and `src/serve/sampler.rs`, rather than profiler output, because the profiler hooks themselves force GPU syncs and distort the measurement — see "Open Questions" for the empirical spike that still needs to be run):

| Component | Estimated cost | Source / citation | Share of the 32.7 ms gap |
|---|---|---|---|
| MoE routing `to_vec2` syncs — 2 per layer × 30 layers = 60 forced `waitUntilCompleted` per token | 18–25 ms | `gemma4.rs:428-429`; each `to_vec2()` flushes the command buffer and blocks the CPU until GPU drains. Agent #5 quoted ADR-005 1b.6's pre-fix "17–34 ms wasted per token" for 34 syncs; we now have 60 | ~55–75% |
| Sequential per-expert CPU loop — 30 layers × 8 experts × ~13 ops/expert + 240 `Tensor::zeros` + 240 scalar `Tensor::new` allocations | 8–12 ms | `gemma4.rs:436-465`; 104 dispatches + 240 GPU buffer allocations per token. No batching across experts, no pipelining across the `combined + ...` chain | ~25–35% |
| RmsNorm fragmentation — ~10 RmsNorms per layer × 9 sub-dispatches each = ~2,700 dispatches/token just for norms | 3–5 ms | `gemma4.rs::RmsNorm::forward` manually chains `sqr`, `mean_keepdim`, `ones_like`, `affine`, `add`, `sqrt`, `recip`, `broadcast_mul`, `broadcast_mul`, `to_dtype`; candle has no fused f32 RmsNorm | ~10–15% |
| Global attention Q/K/O projection QMatMul at 8192 dims (5 layers × 4 ops) + partial RoPE overhead | 1–3 ms | `gemma4.rs:280-325`; 2x the GEMM size of sliding layers. ADR-005 item 1b.16 calls out the 8192-dim QMatMul cliff as unconfirmed | ~3–10% |
| CPU overhead: ~7,000 lazy Tensor allocations + Arc refcounts per token | 3–8 ms | Agent #5 derived 7,009 dispatches/token by counting ops in `gemma4.rs`. At 23.8 tok/s that is ~166,000 Tensor alloc/drop/sec | ~10–25% |
| Pure GPU kernel execution | ~10–14 ms | Agent #5 bounds by noting hf2q uses the same candle Metal QMatMul kernels as dense layers; llama.cpp's 9.3 ms is the compute floor, hf2q's compute cannot be dramatically slower | — (already near floor) |

**Headline:** the MoE CPU-driven dispatch path (routing syncs + sequential expert loop) is **60–90% of the gap**. Everything else is secondary.

Cross-reference: MLX-LM collapses the entire 8-expert dispatch to a **single** `affine_gather_qmm_*` Metal kernel invocation (`switch_layers.py:76-87`, verified in `mlx.metallib`). llama.cpp collapses it to **one** `kernel_mul_mv_id_q6_K_f32` + **one** `kernel_mul_mv_id_q8_0_f32` per layer (`ggml-metal.metal:7620-7630`) by redirecting pointers inside the same base `kernel_mul_mv_q6_K_f32` used for dense layers. Neither of them does what hf2q currently does: 60 forced GPU→CPU round-trips and 240 sequential `QMatMul.forward()` calls per token.

##### 2. Tiered Plan

Each tier has (a) an aggregate goal, (b) specific items, (c) a gate benchmark that must be met before the next tier starts, and (d) the exact correctness check. No tier begins until the previous tier's gate is met AND correctness is green. If a tier regresses correctness, stop and bisect before continuing.

| Tier | Goal | Items | Gate (median tok/s) | Cumulative |
|---|---|---|---|---|
| **Tier 1 — Sync stalls + per-expert loop** | Remove the 60 forced `to_vec2` syncs and the 240 per-token allocations in the expert loop | 1bNEW.1, 1bNEW.2, 1bNEW.3 | ≥ **45 tok/s** AND `metrics.txt` shows `moe_to_vec2_count == 0` and `sampler_sync_count ≤ 1` | 23.8 → ~45–55 |
| **Tier 2 — Fused norm + RoPE kernels** | Single-dispatch f32 RmsNorm (with optional residual add) and fused RoPE kernel — kills ~3,400 dispatches/token | 1bNEW.4, 1bNEW.5, 1bNEW.6 | ≥ **70 tok/s** AND `dispatches_per_token < 2500` | ~55 → ~70–80 |
| **Tier 3 — Fused MoE kernel + attention prelude** | `mul_mv_id`-style Metal kernel reusing candle's `simd_sum` row matmul; stride-aware Q/K prelude removes wasted `.contiguous()` calls; eliminate residual scalar allocs | 1bNEW.7, 1bNEW.8, 1bNEW.9 | ≥ **95 tok/s** AND `moe_dispatches_per_layer ≤ 4` | ~70–80 → ~95–105 |
| **Tier 4 — Prefill head_dim=512 + warmup + cliff** | bf16 SDPA at bd=512 for global-attention prefill, llama.cpp flash-attn port as escape, extended warmup compiles prefill PSOs at load, optional 8192-dim QMatMul cliff fix | 1bNEW.10, 1bNEW.11, 1bNEW.12, 1bNEW.13 | ≥ **107 tok/s** decode AND prefill ≥ llama.cpp parity AND TTFT < 150 ms | ~95–105 → ≥107 + prefill |

Any item that fails its individual gain target by more than 50% triggers a re-plan — do not paper over a missed estimate by piling more items on top.

**Metrics instrumentation (required to land before Tier 1).** Add to `mod.rs` so every benchmark run dumps `metrics.txt` with: `dispatches_per_token`, `moe_to_vec2_count`, `moe_dispatches_per_layer`, `sampler_sync_count`, `norm_dispatches_per_token`. These turn the tier gates from "trust me" into mechanical checks.

##### 3. Per-Item Detail

13 items, ordered by tier and dependency. The numbering reflects the merge: items 1, 2, 9, 10, 13 came from the queen's first cut; items 3, 5, 6, 8, 11, 12 came from Agent #7's later synthesis; items 4 and 7 appeared in both with consistent designs.

---

**1bNEW.1 — Eliminate MoE routing GPU→CPU syncs via GPU-only dispatch tables**
- **Tier:** 1
- **What it does:** Replace the two `top_k_indices.to_vec2()` / `top_k_weights.to_vec2()` calls at `gemma4.rs:428-429` with a GPU-side expansion. Keep `top_k_indices` and `top_k_weights` as GPU `Tensor`s. The per-expert dispatch loop is driven by a *single* small `to_vec1` of an `[8]`-element active-experts vector at the end of the routing block — pull 32 bytes once per layer instead of two `to_vec2` flushes per layer. Aggressive variant: defer the drain until layer 29 and combine all 30 layers of routing into a single sync at end of forward pass.
- **Why it helps:** Drops sync count from **61** (Agent #5: 60 from MoE routing × 30 layers + 1 sampler) to ~31 conservative or ~2 aggressive. At the conservative cost of 0.3 ms/sync this saves 9 ms; at the upper bound of 0.5 ms/sync it saves 15 ms. Cross-confirmed: llama.cpp does **0** `waitUntilCompleted` per token (Agent #2 §5); MLX-LM does 1 `async_eval` per token (Agent #3).
- **Expected gain (working):** 30 stalls/token × ~0.3 ms/stall = 9 ms saved → 42 - 9 = 33 ms/token = **30 tok/s**. Aggressive variant: +12-15 ms saved → ~36-40 tok/s.
- **Correctness risk:** **MEDIUM-HIGH.** This is the exact area that produced the a0952e2 gibberish. The GPU-side path must align expert weights and per-expert scales correctly; off-by-one in index ordering = silent gibberish.
- **Validation plan:** (1) Capture `top_k_indices` / `top_k_weights` values at layers 0/5/15/29 from the current baseline using the Paris prompt, commit as fixture in `tests/fixtures/routing_baseline.npy`. (2) Refactor; assert bitwise-identical or 1e-6 match. (3) Paris gate. (4) Bisect-safe incremental commits: first remove `to_vec2` for weights, verify; then for indices.
- **Dependencies:** None — first item.
- **Estimated LOC:** ~80.
- **Chesterton's fence:** The current sync exists because the per-expert loop is CPU-driven — it indexes `self.expert_gate_up[eid]` (a Rust `Vec<QMatMul>`). The 1b.1 design that put weights into per-expert `QMatMul` objects is correct and load-time only; we change *how the forward pass feeds expert IDs into it*, not the load layout.

---

**1bNEW.2 — Batched per-expert MoE forward: hoist per-token state out of the loop**
- **Tier:** 1
- **What it does:** Restructure `MoeBlock::forward` so the inner `for k in 0..self.top_k` loop no longer allocates `Tensor::zeros` (`gemma4.rs:443`), `Tensor::new(&[w])` (`gemma4.rs:459`), or temporary `combined` tensors per iteration. Build a single `[top_k]` weights tensor on GPU once via `gather` + `broadcast_mul` against `per_expert_scale`; accumulate via `combined.affine(1.0, 0.0) + expert_out * w_broadcast` using a persistent `combined` buffer reused across the loop. Eliminates 240 `Tensor::zeros` + 240 `Tensor::new` + 240 `to_dtype` per token.
- **Why it helps:** Agent #5 §4 #2: 240 GPU buffer allocs/token + 240 sequential accumulator-add data dependencies that prevent candle's command pool from pipelining across experts. Removing the `combined + ...` chain lets the 16 expert matmuls per layer batch in one command buffer.
- **Expected gain:** 4–8 ms. Working: 240 sequential accumulates × ~0.02 ms (command buffer overhead per dependency edge) + 480 alloc/free × ~0.005 ms = 4.8 + 2.4 = 7.2 ms. Conservative: 4 ms.
- **Correctness risk:** **LOW.** Pure algebraic identity; the accumulation math is unchanged.
- **Validation plan:** Unit test feeds synthetic routing into `MoeBlock::forward` and asserts bitwise identity to a reference scalar implementation. Then Paris gate.
- **Dependencies:** 1bNEW.1.
- **Estimated LOC:** ~40.
- **Chesterton's fence:** `Tensor::zeros` exists because the old (pre-1b.4) code accumulated F32 outputs from BF16 inputs; post-1b.4 the whole pipeline is F32, so the cast and zero-buffer reset are vestigial. `Tensor::new(&[w])` exists because `broadcast_mul` with an `f32` scalar isn't a direct candle API — but `affine(mul, 0.0)` is.

---

**1bNEW.3 — Single-sync sampler path (speculative argmax enqueue)**
- **Tier:** 1
- **What it does:** For greedy decode (T=0), restructure `sampler.rs:49` so the `argmax + to_scalar` doesn't sync after every token. Use candle's command-pool batching to enqueue the argmax alongside the *next* token's embed lookup, syncing only every N tokens (e.g., every 4). When the sync fires, retrieve N u32 tokens at once. Classic speculative-enqueue — valid because at T=0 the forward pass is deterministic and we already know what we're going to compute.
- **Why it helps:** Agent #5 §9: S = 61 = 60 routing + 1 sampler. After 1bNEW.1 collapses routing syncs, the sampler's 1 becomes a meaningful fraction. Agent #3: MLX-LM uses `mx.async_eval` for exactly this pattern; Agent #2: llama.cpp commits per command buffer, not per op.
- **Expected gain:** 1–3 tok/s, free after 1bNEW.1.
- **Correctness risk:** **LOW.** At T=0 with `repetition_penalty == 1.0`, deterministic. Risk: with `repetition_penalty != 1.0` we need the previous token before computing the next, so the fast path must gate on `(temperature == 0.0 && repetition_penalty == 1.0)`; otherwise per-token sync.
- **Validation plan:** Bitwise-identical greedy output vs current path (must be — same ops, different commit boundary). 5-run benchmark variance must stay zero.
- **Dependencies:** 1bNEW.1.
- **Estimated LOC:** ~30 across `sampler.rs` and decode loop in `mod.rs`.
- **Chesterton's fence:** The current sampler pattern exists because the decode loop is sequential at the API level. But at the GPU-dispatch level, candle's lazy command pool can enqueue forward pass N+1 before sync N resolves — as long as we haven't called `to_scalar` yet. Streaming SSE delivery still works as long as we sync before the SSE writer iterates.

---

**1bNEW.4 — Fused F32 RmsNorm Metal kernel**
- **Tier:** 2
- **What it does:** Custom Metal kernel that performs RmsNorm in a single dispatch: load vector, compute mean-of-squares via SIMD threadgroup reduction, rsqrt, multiply by weight, store. Replaces the 9-op manual chain (`gemma4.rs::RmsNorm::forward` lines 32-41). Built via `Device::new_library_with_source` (Agent #4 §5 recipe — no candle fork needed). Two-pass reduction inside the kernel to match the manual version's numerics order-of-ops exactly. Lives at `src/serve/kernels/rms_norm.metal` (the empty `kernels/` directory already exists in git status).
- **Why it helps:** Agent #5 §4 #3: ~10 RmsNorms/layer × 30 layers = 300 calls/token, each currently ~9 dispatches = ~2,700 dispatches just for norms. Fused kernel = 1 dispatch each = 2,400 dispatches saved + ~48 command-buffer commits saved.
- **Expected gain:** 4–8 ms saved → +5–10 tok/s. Conservative: 4 ms.
- **Correctness risk:** **LOW-MEDIUM.** RmsNorm variance computation is numerically delicate; use Welford or two-pass reduction to match precision.
- **Validation plan:** (1) Rust unit test in `gemma4.rs::tests` comparing fused vs manual on random `[1, 128, 2816]` inputs at ε=1e-5. (2) Bisect: replace one RmsNorm site at a time, Paris gate per site. (3) Full benchmark.
- **Dependencies:** Tier 1 complete; Open Question Q1 resolved (custom kernel integration verified).
- **Estimated LOC:** ~80 Rust + ~50 Metal.
- **Chesterton's fence:** The 9-op manual decomposition exists because candle 0.10.2 has no exposed fused F32 RmsNorm on Metal (Agent #4 confirmed — `candle-nn::ops::rms_norm` is not in the public Source enum for Metal). We are replacing manual Tensor composition, not a hidden-but-optimized path. The new kernel is upstreamable to candle later.

---

**1bNEW.5 — Fused residual-add + RmsNorm (extends 1bNEW.4)**
- **Tier:** 2
- **What it does:** Extend the kernel from 1bNEW.4 to take an optional `residual` tensor and compute `(x + residual) → normed` in one pass. Replaces `RmsNorm::forward_with_residual` at `gemma4.rs:47-51` which currently does `(x + residual)?` (1 dispatch) then `self.forward(&sum)?` (will be 1 dispatch after 1bNEW.4 — but we can do better).
- **Why it helps:** Agent #5: 2 residual-add + norm sites per layer (`gemma4.rs:505, 521-522`) = 60 fused-norm calls/token. Eliminating the separate add saves 60 dispatches/token AND halves the memory traffic on the residual path (read x + read residual + read weight → write once, vs read x + read residual → write sum → read sum + read weight → write normed).
- **Expected gain:** 1–3 ms. Working: 60 fewer dispatches + halved memory traffic on a hot 2816-element tensor.
- **Correctness risk:** **LOW.** Trivial extension of 1bNEW.4.
- **Validation plan:** Same kernel test plan as 1bNEW.4 with an additional residual-path branch test.
- **Dependencies:** 1bNEW.4.
- **Estimated LOC:** ~30 delta on top of 1bNEW.4.
- **Chesterton's fence:** `forward_with_residual` was added in 1b.10 as a small dispatch reduction. We are not removing it; we are making its inner kernel fused.

---

**1bNEW.6 — Fused RoPE kernel (standard and partial variants)**
- **Tier:** 2
- **What it does:** Replace the 8-op `rope_apply` at `gemma4.rs:133-140` (narrow×2, broadcast_mul×4, sub, add, cat) with a single Metal kernel. Two flavors: full RoPE for sliding layers (`rotary_dim == head_dim`) and partial RoPE for global layers (`rotary_dim < head_dim`, with a pass-through tail). Kernel: Q/K shaped `[B, H, T, head_dim]`, cos/sin shaped `[T, rotary_dim/2]`, rotary_dim parameter; one threadgroup per (B, H, T).
- **Why it helps:** Agent #5 §6: standard RoPE = 22 dispatches/attention-layer; partial RoPE = 26. Across 30 layers: ~720 RoPE dispatches per token. Fusing to ~2 per layer (one Q, one K) = 660 dispatches saved. Agent #2 §2: llama.cpp uses one `kernel_rope_*` call per Q and per K.
- **Expected gain:** 3–6 ms.
- **Correctness risk:** **MEDIUM.** Partial RoPE is exactly where the `project_coherence_bug.md` regression lived. The kernel must correctly handle the pass-through tail (positions `[rotary_dim, head_dim)` are not rotated).
- **Validation plan:** (1) Unit test vs existing `rope_apply` on `[1, 16, 1, 256]` and `[1, 16, 1, 512]` inputs with `partial_rotary_factor=0.5`. (2) Layer-by-layer Q/K output diff vs baseline for first 4 decode tokens at ε=1e-6. (3) Paris gate.
- **Dependencies:** 1bNEW.4 (proves the custom-kernel integration pattern).
- **Estimated LOC:** ~60 Rust + ~80 Metal.
- **Chesterton's fence:** The current 8-op implementation exists because candle 0.10.2 does not expose a fused RoPE on Metal (Agent #4 — not in the Source enum). The 8 ops are memcpy + elementwise on disjoint halves — the exact pattern a single kernel can absorb.

---

**1bNEW.7 — Fused `mul_mv_id` Metal kernel for MoE (the big one)**
- **Tier:** 3
- **What it does:** A Metal kernel that takes (a) the full 3D expert weight QTensor `[num_experts, out, in]` for one projection (gate, up, or down), kept alive alongside the per-expert `Vec<QMatMul>` from 1b.1; (b) a GPU-resident `[num_tokens, top_k]` expert-index tensor from 1bNEW.1; (c) input `[num_tokens, in]`. Output: `[num_tokens, top_k, out]`. Internally: each threadgroup reads `tgpig.z` (active expert slot), translates through the ids buffer to a byte offset into the 3D slab, then **delegates to the same `simd_sum`-based row-matmul inner loop** that candle's `kernel_mul_mv_q6_K_f32` and `kernel_mul_mv_q8_0_f32` already use. Threadgroup config: Q6K → `nr0=2, nsg=2` (Agent #2 §1, from `ggml-metal-impl.h:44-45`); Q8_0 → `nr0=2, nsg=4` (`:26-27`).
  - **Critical shortcut discovered by Agent #4:** `kernel_mul_mv_id_q4_K_f32`, `_q6_K_f32`, and `_q8_0_f32` **already exist** in candle's `quantized.metal` at lines 7625-7642 — they are unwrapped on the Rust side (Agent #4 §1 Q3). This collapses 1bNEW.7 from "write a new Metal kernel" to "wire up an existing kernel via `Device::new_library_with_source` plus the appropriate `MTLBuffer` setup."
- **Why it helps:** Collapses 30 layers × 8 experts × 3 projections = 720 separate `QMatMul::forward` calls → **3 fused dispatches per layer = 90 total**. Each fused dispatch covers 8 experts in a single Metal command. This is the largest single lever after Tier 1.
- **Expected gain:** 8–15 ms saved → +20–35 tok/s. Working: llama.cpp's `kernel_mul_mv_id_*` runs at 0.31 ms/layer (Agent #2 §6); hf2q post-Tier 2 would be at ~0.6 ms/layer; closing 0.3 ms/layer × 30 = 9 ms.
- **Correctness risk:** **HIGH.** Net-new dispatch path. The reverted `moe_expert_q4k.metal` failed because it was scalar (1 thread per output element). This attempt MUST reuse candle's SIMD inner loop — Open Question Q1 must resolve "yes, we can call the existing kernel function" before this item begins. If reuse is impossible, this item is ABANDONED — see escape hatch in §7.
- **Validation plan:** (1) **Phase A — numerical:** Rust test feeds known inputs, compares against running 8 separate `QMatMul::forward` calls; ε=1e-5 elementwise. (2) **Phase B — single layer:** replace MoE at layer 0 only via feature flag; Paris gate; per-token logprob comparison for first 16 tokens. (3) **Phase C — all layers:** flip flag to all layers; Paris gate; 5-run benchmark. (4) **Phase D — adversarial:** 2048-token prompt with specific-fact recall.
- **Dependencies:** 1bNEW.1, 1bNEW.2; Open Question Q1 RESOLVED before code starts.
- **Estimated LOC:** ~200 Rust (Metal dispatch wrapper modeled on `candle-metal-kernels/src/kernels/quantized.rs`) + 0–50 Metal (only if the existing candle kernels don't expose the variant we need at the function level).
- **Chesterton's fence:** The per-expert `Vec<QMatMul>` exists because 1b.1 landed it as the baseline that works today. We keep it as a feature-flagged fallback (`--moe-kernel=fused|loop`). The 3D source QTensor is already in memory (`gemma4.rs:620-621` loads it via `gguf.get_qtensor`, then `:645-652` slices per-expert) — we add an `Arc<QTensor>` field on the layer to retain the 3D view alongside the slices.

---

**1bNEW.8 — Fused Q/K projection prelude (stride-aware RoPE elimination of `.contiguous()`)**
- **Tier:** 3
- **What it does:** Combine `q_proj.forward` + `reshape` + `q_norm.forward` + `transpose` + `rotary_emb.apply` into one command-buffer-grouped sequence (and symmetrically for K) with no intermediate `.contiguous()` calls. Make the RoPE kernel from 1bNEW.6 stride-aware so it can read `narrow`'d views directly without forcing a copy. Removes the `.contiguous()` calls at `gemma4.rs:113-114, 118-121` that exist solely to give the current RoPE kernel contiguous memory.
- **Why it helps:** Agent #5 §6: ~22 dispatches per sliding-attn prelude, ~26 per global; many are wasted `.contiguous()` copies. Stride-aware kernel eliminates ~120 dispatches/token plus the corresponding allocations.
- **Expected gain:** 2–4 ms.
- **Correctness risk:** **MEDIUM.** Stride-aware kernels are error-prone — the `slice_scatter` stride bug at a0952e2 is exactly this class of mistake.
- **Validation plan:** Unit test that the stride-aware RoPE produces identical output for stride-1 and stride-N inputs. Layer 0 Q/K diff vs baseline. Paris gate.
- **Dependencies:** 1bNEW.6.
- **Estimated LOC:** ~40 Rust + ~30 Metal delta on top of 1bNEW.6.
- **Chesterton's fence:** The `.contiguous()` calls exist because the current `rope_apply` uses `Tensor::narrow` (strided views) but the final `Tensor::cat` requires contiguous inputs. A fused stride-aware kernel sidesteps both issues.

---

**1bNEW.9 — Eliminate the residual scalar-weight allocations**
- **Tier:** 3
- **What it does:** Replace `Tensor::new(&[w], device)?` + `broadcast_mul` at `gemma4.rs:459-460` with `expert_out.affine(w as f64, 0.0)?` if 1bNEW.7 hasn't subsumed it. Mostly subsumed by 1bNEW.7 (the fused kernel takes weights as a buffer); remains as the fallback for the per-expert `Vec<QMatMul>` path that lives behind the feature flag.
- **Why it helps:** 240 GPU allocations/token × ~0.005 ms = ~1.2 ms when the fallback path is active.
- **Expected gain:** 0.5–1.5 ms (subsumed if 1bNEW.7 lands).
- **Correctness risk:** **LOW.**
- **Validation plan:** Bitwise output comparison.
- **Dependencies:** 1bNEW.2. Subsumed by 1bNEW.7.
- **Estimated LOC:** ~5.
- **Chesterton's fence:** `Tensor::new(&[w])` was the naive way to get a scalar onto the GPU. `affine` does the same without allocating a tensor.

---

**1bNEW.10 — BF16 prefill SDPA at head_dim=512 (unblocks the existing candle kernel)**
- **Tier:** 4
- **What it does:** In `Attention::forward` prefill branch (`gemma4.rs:308-318`), cast Q/K/V to BF16 before calling `candle_nn::ops::sdpa`, cast output back to F32. **The current code's comment that "candle's SDPA full kernel exceeds 32KB threadgroup mem for head_dim=512" is STALE.** Agent #4 §4 Q12 and Agent #6 Q1 both confirm: `sdpa.rs:86-94` already selects reduced tiles `(bq=8, bk=8, wm=1, wn=1)` for BD=512 in f16/bf16, totaling **24.1 KB** threadgroup memory. The only blocker is F32 (rejected at `sdpa.rs:87-92`). The kernel variant `steel_attention_bfloat16_bq8_bk8_bd512_wm1_wn1_maskbfloat16` is already compiled and instantiated (`scaled_dot_product_attention.metal:2334-2337`).
- **Why it helps:** Eliminates the manual `repeat_kv` + matmul + softmax + matmul fallback for global attention prefill (Agent #6 Q6: ~10–12 dispatches/layer + 5 MB of temporary `repeat_kv` allocations per global layer). Replaces with one fused SDPA call.
- **Expected gain:** Decode: +0–1 tok/s (decode already uses vector SDPA at f32 — bd=512 *vector* path is supported at f32). Prefill: **3–5x faster** on global attention layers, improving TTFT and long-prompt throughput.
- **Correctness risk:** **MEDIUM.** F32 prefill was the conservative choice from 1b.4. BF16 cast at the SDPA boundary only (decode stays F32) is narrower but still a numerical change.
- **Validation plan:** (1) Compare post-softmax attention weights BF16 vs F32 manual on a 128-token prefill at ε=1e-3. (2) Compare next-token top-5 distribution. (3) Paris gate at 128-token prompt prefix. (4) Adversarial: 2048-token document with specific-fact recall, output must match. See Open Question Q4.
- **Dependencies:** None — orthogonal to Tiers 1-3.
- **Estimated LOC:** ~20.
- **Chesterton's fence:** The manual path was added when head_dim=512 SDPA was *genuinely* unsupported. Candle has since added the reduced-tile variant; the comment was not updated. We're removing a stale workaround, not a functional safeguard. If BF16 correctness fails, fall back to the manual path *only for prefill global layers* and escalate to 1bNEW.11.

---

**1bNEW.11 — Port llama.cpp `kernel_flash_attn_ext_vec_f16_dk512_dv512` (escape hatch for 1bNEW.10)**
- **Tier:** 4 (contingent — only if 1bNEW.10 fails Q4)
- **What it does:** Port llama.cpp's flash-attn vec kernel for head_dim=512 (`ggml-metal.metal:7165`) with f32 accumulation. Agent #2 §4: FATTN_SMEM ≈ 3.5 KB for the vec path (fits trivially in 32 KB). Kernel template parameterized on `(DK, DV)` so a 512 instantiation is direct.
- **Why it helps:** Same as 1bNEW.10 but without BF16 cast risk. Matches llama.cpp's known-fast path exactly.
- **Expected gain:** Same as 1bNEW.10.
- **Correctness risk:** **HIGH.** ~600 LOC of new Metal in unfamiliar template machinery.
- **Validation plan:** ε=1e-3 vs F32 manual; full Paris gate; 5-run benchmark.
- **Dependencies:** 1bNEW.10 must be tried first; only executed if Q4 reports BF16 regression.
- **Estimated LOC:** ~600.
- **Chesterton's fence:** This is the contingency. The baseline manual path stays as the final fallback if even this fails.

---

**1bNEW.12 — Extended warmup: compile prefill PSOs at model load**
- **Tier:** 4
- **What it does:** Extend the existing single-token warmup at `mod.rs:231-234` with a *second* warmup pass using a short prefill sequence (8 tokens). This triggers Metal PSO compilation for prefill-specific kernel variants (SDPA full, prefill-path softmax, mask construction, prefill MoE batched matmul) that the decode-only warmup misses. Cleanup `clear_kv_cache()` after.
- **Why it helps:** Agent #6 Q7-Q10: candle compiles Metal libraries from MSL source at runtime; first *prefill* call cold-compiles prefill PSOs (~37–100 ms TTFT spike). Llama.cpp ships a pre-compiled `.metallib`; we can't, but we can pre-warm. This is what makes the serve mode (not just benchmark) genuinely fast.
- **Expected gain:** Decode throughput unchanged. **TTFT –37 to –100 ms** on first request after model load. Critical for serve mode (and per the no-VW-cheating principle, every code path must be fast — including the first request a user sends).
- **Correctness risk:** **LOW.** Warmup runs the forward pass with throwaway input.
- **Validation plan:** Measure TTFT on a fresh process for an 8-token prompt; must be < 150 ms with extended warmup vs current ~200+ ms.
- **Dependencies:** 1bNEW.10 lands first (so the warmup pass exercises the new fused SDPA path, not the manual fallback).
- **Estimated LOC:** ~15.
- **Chesterton's fence:** The single-token warmup was added in 1b.14 to eliminate the shader-compile spike on decode. It only covers decode because the benchmark hot path is decode. For production serving where TTFT matters, prefill kernels also need to be warm. We extend, not replace.

---

**1bNEW.13 — QMatMul 8192-dim cliff: profile and conditionally fix**
- **Tier:** 4 (optional — measure first)
- **What it does:** Phase 1b item 1b.16 noted a suspected performance cliff in candle's QMatMul kernel at 8192 output dim (q_proj/o_proj for global attention layers). This item is the empirical measurement: use Metal System Trace to time `kernel_mul_mv_q6_K_f32` at `[2816] → [8192]` and compare to the dense case `[2816] → [2816]`. If the cliff is real, options: (a) supply a different threadgroup configuration; (b) split into two `[2816] → [4096]` calls with a concat; (c) upstream a fix to candle.
- **Why it helps:** If the cliff is real, 5 layers × 4 projections = 20 calls/token, potentially 1–3 ms wasted.
- **Expected gain:** 0 if no cliff; up to 3 ms if cliff is real.
- **Correctness risk:** Depends on the fix. Threadgroup config change is correctness-safe; split-and-concat is not (concat overhead may eat the gain).
- **Validation plan:** Measure first. Then if a fix exists, bisect-validate.
- **Dependencies:** None.
- **Estimated LOC:** ~0–50 depending on outcome.
- **Chesterton's fence:** 1b.16 raised this as a hypothesis. Do **not** apply a speculative fix before the measurement.

---

##### 4. Correctness Protocol (Run Between Every Item)

The a0952e2 → gibberish regression cost a full day. The mantra is explicit: *no shortcuts, never make assumptions, measure 3x cut once.* Every item above **must** pass this protocol before merging.

**Baselines to record before Tier 1 begins** (one-time setup, committed to repo):
1. **`tests/fixtures/paris_baseline.txt`** — full generated text + token-id sequence from running the Paris prompt at HEAD with `--temperature 0`. This is the canonical "correct output" snapshot.
2. **`tests/fixtures/bench_prompt_128.expected.tokens`** — full token-id sequence from the canonical benchmark prompt at HEAD. Used for greedy-determinism comparison.
3. **`tests/fixtures/logits_baseline.npy`** — for the first 4 decode tokens, the pre-softmax logits at layers 0/5/15/29. Used for layer-by-layer numerical-drift comparison.
4. **`tests/fixtures/routing_baseline.npy`** — `top_k_indices` and `top_k_weights` GPU tensor values at layers 0/5/15/29 for the first 4 decode tokens. Used to validate 1bNEW.1 doesn't reorder experts.

These fixtures are *committed to the repo*, not generated on demand — that way every developer / future bisect run has the same ground truth.

**Per-item procedure** (run all steps in order; any failure → stop, bisect, fix):

1. **Branch.** Create `phase-1b-new-N` from the commit that landed 1bNEW.(N−1) successfully. One branch per item.
2. **Implement.** Build must succeed with **zero new warnings** relative to the previous commit. Run `cargo clippy` and address every clippy lint introduced by the change.
3. **Unit tests first.** If the item adds a Rust-visible function or kernel, write a `#[test]` in the relevant module that feeds synthetic inputs and asserts the expected output vs a reference implementation (manual or scalar). The tolerance is per-item:
   - Items 1bNEW.2, 1bNEW.3, 1bNEW.8, 1bNEW.9, 1bNEW.13 (mathematically equivalent rewrites): **ε = 1e-6**.
   - Items 1bNEW.4, 1bNEW.5, 1bNEW.6 (fused-kernel reductions, may have minor reduction-order drift): **ε = 1e-5**.
   - Items 1bNEW.1, 1bNEW.7 (changes index order or kernel internals): **ε = 1e-4**.
   - Item 1bNEW.10 (BF16 cast at SDPA boundary): **ε = 1e-2 on logits**, with additional top-5-token-distribution match.
4. **Determinism check.** Run `./target/release/hf2q generate --benchmark --prompt-file tests/bench_prompt_128.txt --max-tokens 128 --temperature 0 --seed 42` and diff the token-id sequence against `tests/fixtures/bench_prompt_128.expected.tokens`. Greedy sampling at T=0 is fully deterministic — *any* token-id divergence beyond the item's tolerance is a regression. Update the fixture only when the divergence is justified by the item (and document in the commit message).
5. **Paris gate.** Run `./target/release/hf2q generate --prompt "What is the capital of France?" --max-tokens 32 --temperature 0`. Output must (a) be coherent English, (b) contain the substring `Paris`, (c) trigger no red flag from the list below. Compare against `tests/fixtures/paris_baseline.txt`.
6. **Logit drift check.** For the first 4 decode tokens, dump pre-softmax logits at layers 0/5/15/29 and compare against `tests/fixtures/logits_baseline.npy`. Max-abs-diff must be within the item's tolerance from step 3.
7. **Adversarial spot check.** Generate 128 tokens for a question with a known fact ("Explain the difference between a transformer and a state-space model.") and read the output. Gibberish that passes steps 4–6 means a fixture is wrong — regenerate from the previous green commit and start over.
8. **Multi-shape sweep (no VW cheating).** Run the same change with three additional inputs to ensure no benchmark-only specialization snuck in: a 1-token prompt, a 512-token prompt, and a 2048-token prompt. All three must produce coherent output via the *same* code path.
9. **Benchmark.** Only after correctness gates pass. Run `--benchmark --max-tokens 128`. Report median, p95, variance. Variance across 5 runs must be < 2% (the current baseline has zero variance, so any larger variance is itself a bug to investigate).
10. **Metrics check.** Verify `metrics.txt` counters match the expected values for the item:
    - After 1bNEW.1: `moe_to_vec2_count == 0`.
    - After 1bNEW.3: `sampler_sync_count ≤ 1` per N decoded tokens.
    - After 1bNEW.4: `norm_dispatches_per_token ≤ 300`.
    - After 1bNEW.7: `moe_dispatches_per_layer ≤ 4`.
11. **Append to bisect table.** Add a row to the Phase 1b bisect table (existing in this ADR) with item ID, correctness ✓/✗, and median tok/s.
12. **Commit.** Only if all above pass. Commit message must name the item, cite the source file:line for the change, and include the benchmark median tok/s. This enables fast bisect if a later item regresses.

**Red flags (immediate rollback):**
- Token repetition patterns: `own own own`, `the the the`, `\n\n\n\n` runs.
- Control-token floods: `<unk>`, `<pad>`, `<s>`, `</s>` mid-generation.
- Non-ASCII gibberish runs > 3 tokens.
- Output length < 5 tokens (model silently EOS'd).
- The exact a0952e2 failure signature: nonsense characters with no coherent reasoning structure.

**Tier gate enforcement.** Do not start the next tier until the current tier's benchmark median reaches the target AND the correctness gate passes for every item in the tier. If a tier stalls (gain < 80% of estimate), do a root-cause profiling pass using `metrics.txt` counters before proceeding. **Do not "wishful-think" past a missed gate** — the mantra forbids it.

Any correctness regression: **stop, bisect, fix.** No "I'll fix it in the next item." The a0952e2 lesson is that small regressions compound into undiagnosable ones.

##### 5. Anti-Goals (Explicitly Rejected, with Reasoning)

These are *direct consequences of the mantra*. Each one has been considered and rejected — not skipped, rejected with reasoning. Do not relitigate during execution.

1. **Custom scalar-dequant Metal kernel (1 thread per output element).** REJECTED. Already attempted as `moe_expert_q4k.metal` (ADR 1b.1 "Failed approach"), benchmarked *slower* than candle's SIMD QMatMul because scalar dot products cannot compete with `simd_sum`-based row matmul. 1bNEW.7 explicitly reuses candle's SIMD inner loop or is abandoned — no scalar-path fallback.
2. **Move routing back to CPU-computed indices.** REJECTED. The pre-0a703d7 path did this and was slower; 1b.6 PARTIAL was the first half of the fix, and 1bNEW.1 completes it. Direction is forward, not backward.
3. **Switch the whole forward pass to BF16.** REJECTED. 1b.4 showed F32 saves ~690 dtype casts/token because candle's QMatMul requires F32 input. Only 1bNEW.10 converts select boundaries (global attention prefill SDPA).
4. **CPU fallback for head_dim=512 SDPA prefill.** REJECTED — violates the GPU-only principle (`feedback_gpu_everything.md`). If 1bNEW.10 (BF16 SDPA) fails Q4, the escape is 1bNEW.11 (port llama.cpp flash-attn), NOT a CPU path.
5. **Skip tier gates.** REJECTED. If Tier 1 does not reach 45 tok/s, the problem is deeper than the plan and Tier 2 will not rescue it. Stop and re-plan.
6. **Benchmark-tune (VW cheating).** REJECTED — direct mantra violation. Every kernel must be correct for arbitrary prompts, arbitrary `num_tokens`, arbitrary seq lengths. No hardcoded fast path for the 128-token canonical prompt. Verify after every tier by running a 1-token prompt AND a 512-token prompt AND the canonical prompt through the *same* code path.
7. **Stub code or "TODO later" placeholders.** REJECTED — direct mantra violation. Every commit lands a complete, correct, benchmarked change. No `unimplemented!()`, no `// TODO: handle the other quant type`, no half-fast intermediates.
8. **Try to replicate candle's `moe_gemm_gguf`** (`candle-transformers/src/fused_moe.rs`). REJECTED. It is CUDA-only in candle 0.10.2 (`candle-nn/src/moe.rs:339` — `#[cfg(not(feature = "cuda"))]` bails). Writing a Metal equivalent **is** 1bNEW.7; the rejection here is against pretending the Metal backend exists.
9. **Fork candle as a default.** REJECTED. Agent #4 §5 shows we can compile custom Metal sources via `Device::new_library_with_source` and dispatch through `MetalDevice::command_encoder()` without a fork. Decision per item, not upfront. **Q1** guards this.
10. **Move to MLX-affine quantization.** REJECTED per ADR-005 resolved question: hf2q produces K-quant GGUF and must stay ecosystem-compatible. MLX-LM is a *reference*, not a target.
11. **Skip the Chesterton check on any item.** REJECTED — direct mantra violation. Every item documents WHY the current code exists and what we're replacing. Items without a Chesterton section are not executed.
12. **"Try it and see" without a profiling hypothesis.** REJECTED — direct mantra violation ("never make assumptions"). Every item has a cited source (file:line or measured number) explaining why it matters. Items without citations are moved to Open Questions and resolved before execution.
13. **Disable the bisect table.** REJECTED. The a0952e2 regression is recent. Every tier must extend the bisect table in this ADR.
14. **Parallelize implementation across many agents.** REJECTED for *implementation*. A swarm was valuable for *research* (this plan is the proof). The actual code changes must land one item at a time with a bisect checkpoint per item. Opening multiple PRs in parallel violates "measure 3x, cut once."

##### 6. Open Questions (Require Empirical Spike Before Dependent Items Begin)

Each question is a *measurement task*, not a debate. Spike duration is bounded; results go into the ADR before code changes.

1. **Q1 — Real `waitUntilCompleted` cost per sync on M5 Max.** Reported estimates span 0.1–0.5 ms (5x range). This determines whether 1bNEW.1 lands +5 or +15 tok/s. Spike: Xcode Instruments → Metal System Trace on a single token of the current baseline; 1 hour. Blocks 1bNEW.1 gain calibration (not execution).
2. **Q2 — Can a downstream `.metal` source be compiled at runtime via `Device::new_library_with_source` and dispatched through candle's existing `command_encoder` without ordering hazards?** Agent #4 §5 says yes in theory; verify in practice. Spike: trivial `increment_by_one.metal` between two candle ops; check ordering and 50-ops-per-buffer batching applies. Half a day. **Blocks 1bNEW.4, 1bNEW.6, 1bNEW.7. This is the single biggest gating question.**
3. **Q3 — Does candle's command pool flush a partial buffer on forced sync, wasting pipelining slots?** Agent #5 unknown #3; Agent #4 didn't trace. Spike: instrument `Commands::flush_and_wait` at `candle-metal-kernels/src/metal/commands.rs:176-202`; 1 hour. Affects the gain estimate of 1bNEW.1 — if candle stalls more than expected, the gain is *larger*.
4. **Q4 — BF16 prefill correctness on Gemma 4: does it produce identical top-k tokens as F32 for long prompts?** Specifically: a 2048-token prompt with adversarial recall (a specific date or name mentioned once at position 100, query at position 2000); compare top-5 next-token distributions BF16 vs F32. Blocks 1bNEW.10 commit; if BF16 regresses, escalate to 1bNEW.11. 30 minutes.
5. **Q5 — Is the 8192-dim QMatMul cliff real?** (1bNEW.13.) Metal System Trace timing of `kernel_mul_mv_q6_K_f32` at `[2816] → [8192]` vs `[2816] → [2816]`. Half a day. Blocks the decision to execute 1bNEW.13 at all.
6. **Q6 — Does candle's `arg_sort_last_dim` Metal kernel handle n=128 in a single threadgroup, or spill to multiple?** Affects whether 1bNEW.1's GPU-side top-k path has hidden overhead. Read `candle-metal-kernels/src/kernels/sort.rs`; 1 hour.
7. **Q7 — Is the `per_expert_scale_cpu` field load-time-only or runtime-used?** Check `MoeBlock::load` and all callers. If load-time-only, 1bNEW.1 can gather directly from the GPU `per_expert_scale` tensor and the CPU copy can be deleted entirely. 30 minutes.
8. **Q8 — Does the candle SDPA full kernel for `bd=512` accept the mask dtype hf2q's prefill currently builds?** Agent #6 didn't validate the mask path for the reduced-tile (bq=8, bk=8) variant; the pre-instantiated mask dtypes are `{bool, f16, bf16}`. 30 minutes; blocks 1bNEW.10.

##### 7. Risks and Escape Hatches

**Risk #1: Q2 fails → 1bNEW.7 (fused MoE kernel) is blocked.**
- *Likelihood:* MEDIUM. The kernel functions exist in `candle-metal-kernels/src/metal_src/quantized.metal:7625-7642` (Agent #4 §1 Q3 confirmed `kernel_mul_mv_id_q4_K_f32`, `_q6_K_f32`, `_q8_0_f32` are all present), but they are reachable only via the Metal source string — and candle's `Source` enum is closed (Agent #4 §5). We need a clean way to invoke them outside candle's enum dispatch.
- *Impact:* CATASTROPHIC for the 107 tok/s target. Tier 3 is the single biggest step.
- *Mitigation:* Q2 spike runs **before** Tier 3 begins. If Q2 says yes, 1bNEW.7 is mostly Rust glue (the kernels are written). If Q2 says no, options in order of preference: (a) include the relevant chunk of `quantized.metal` as a string literal in our crate and compile it via the `Device::new_library_with_source` path (Agent #4 explicitly shows this works); (b) upstream `mul_mat_id` Rust API to candle (~2 weeks); (c) accept Tiers 1+2+4 alone at ~75–85 tok/s and ship that as v1 with 107 tok/s as v2.
- *Escape hatch:* Tiers 1+2+4 without 3 lands ~75–85 tok/s — still 3.2–3.6x today. Worth shipping.

**Risk #2: 1bNEW.1 causes a silent correctness regression like a0952e2.**
- *Likelihood:* MEDIUM-HIGH. The a0952e2 bug was a stride assumption in `slice_scatter`, exactly the class of issue that can hide in a GPU-only rewrite of routing.
- *Impact:* Another day-plus bisect; cascading delays for everything downstream.
- *Mitigation:* Correctness Protocol §4 with committed golden tokens (`tests/fixtures/paris_baseline.txt` and `tests/fixtures/routing_baseline.npy`). Bisect-safe incremental commits: first remove `to_vec2` for weights, verify Paris gate; then for indices, verify Paris gate. Each substep is a separate commit, bisectable in seconds.
- *Escape hatch:* Land only the weights-side `to_vec2` removal (30 syncs/token down from 60). Defer the indices side to a separate PR after Q6 confirms `arg_sort` is single-threadgroup.

**Risk #3: Q1 reports `waitUntilCompleted` ≈ 0.1 ms (not 0.4 ms) → gap attribution wrong, real bottleneck is kernel execution.**
- *Likelihood:* LOW-MEDIUM. The pre-1b.6 ADR note documented 34 syncs costing "17–34 ms," which back-solves to 0.5–1.0 ms/sync — higher than the conservative 0.3 ms estimate. So the conservative estimate may be *understated*, not overstated.
- *Impact:* Tier 1 lands ~3 ms instead of ~9 ms; trajectory shifts right; 107 tok/s requires Tier 3 to over-deliver.
- *Mitigation:* Q1 spike runs **first**, before any code changes, to anchor the gap attribution in measured numbers (mantra: "measure 3x, cut once").
- *Escape hatch:* If real bottleneck is GPU kernel execution rather than dispatch, reallocate effort to 1bNEW.7 (fused kernel — actually does compute) and 1bNEW.13 (8192-dim cliff) rather than 1bNEW.1.

**Risk #4: Tier 2 fused norm/RoPE kernels introduce numerical drift below the 1e-5 threshold.**
- *Likelihood:* LOW. These are well-understood reductions; candle and llama.cpp both have correct precedents.
- *Impact:* MEDIUM. Output quality regression even if Paris gate passes — caught only by adversarial prompts.
- *Mitigation:* Validation plans in 1bNEW.4 and 1bNEW.6 require unit tests at ε=1e-5 vs the manual implementation BEFORE any benchmark run. Two-pass reduction or Welford summation if needed.
- *Escape hatch:* Per-norm-site bisect — replace one site at a time, run Paris gate per site. If a specific site regresses, leave it on the manual path.

### Phase 2: HTTP Server
- Restore spec-layer code (schema, SSE, tool parsing) from git history after deep review
- Rebuild handlers and AppState from scratch for candle inference
- Continuous batching scheduler (vLLM-style concurrent inference)
- `hf2q serve --model X.gguf --port 8080`

### Phase 3: Vision
- Image preprocessing pipeline (image crate)
- hf2q produces mmproj GGUF from safetensors (vision tower quantization)
- Also accepts user-provided mmproj files
- Load mmproj GGUF for vision tower
- ViT forward pass + multimodal embedding injection
- `--image` flag for CLI, base64 for API

### Phase 4: Auto Pipeline
- `hf2q serve --model google/gemma-4-27b-it` → download + quantize + serve
- Hardware detection for optimal quant selection
- `~/.cache/hf2q/` caching with integrity checks

### Phase 5: Multi-Model + Architectures
- Hot-swap model loading (ollama-style)
- Qwen3, Mistral model support — prefer candle-transformers when they meet performance bar; own implementations only when needed
- Model discovery from cache directory
- GGUF provenance detection: check metadata for hf2q origin, serve any valid GGUF regardless, apply hf2q-specific optimizations (trusted metadata, skip extra validation) when provenance is confirmed

## Acceptance Criteria

### Phase 1: Inference Engine (Correctness) — DONE
- [x] `hf2q generate --model ./models/gemma4/ --prompt "..."` produces correct, coherent text output
- [x] All K-quant types produced by hf2q load and infer correctly via candle QMatMul (attention, MLP, router)
- [x] Primary model: Gemma 4 26B MoE

### Phase 1b: Speed Optimization
- [ ] Decode speed matches or exceeds llama.cpp on the same hardware (≥107 tok/s on M5 Max, Gemma 4 26B MoE, Q4_K_M), measured using the canonical benchmark harness
- [ ] Prefill speed matches or exceeds llama.cpp on the same hardware
- [ ] MoE expert matmul runs on quantized weights directly (no dequantize-to-BF16 at load time)
- [ ] Tier gates met (per "Phase 1b Remaining Plan" §2): ≥45 tok/s after Tier 1, ≥70 tok/s after Tier 2, ≥95 tok/s after Tier 3, ≥107 tok/s after Tier 4
- [ ] All correctness gates from §4 pass on every commit (no regressions vs `tests/fixtures/paris_baseline.txt`)
- [ ] Multi-shape sweep passes after every tier (1, 128, 512, 2048-token prompts produce coherent output through the same code path — no benchmark-only specialization)

### Phase 3: Vision
- [ ] `hf2q generate --model ./models/gemma4/ --prompt "describe this" --image photo.jpg` produces correct image-aware output
- [ ] Fast follow: Qwen3.5 and additional architectures

### Phase 2: HTTP Server
- [ ] OpenAI Python/JS SDK clients connect and work (chat completions, streaming, tool calling, embeddings)
- [ ] Open WebUI connects and provides full multi-turn chat experience (streaming, vision, tool use)
- [ ] Continuous batching handles concurrent requests without serializing to one-at-a-time
- [ ] `--embedding-model` flag loads a separate model for `/v1/embeddings`

### Phase 4: Auto Pipeline
- [ ] `hf2q serve --model google/gemma-4-27b-it` on a fresh machine: downloads, auto-quantizes for detected hardware, starts serving — zero manual steps
- [ ] Subsequent runs use `~/.cache/hf2q/` (offline mode works for previously cached models)
- [ ] Hardware detection selects optimal quant type for available GPU memory

### Cross-Cutting
- [ ] No GPU detected → hard error with clear message naming supported backends
- [ ] All inference, quantization, and serving code is pure Rust (no C++ deps)
- [ ] Work is structured as reusable crates (quantization, inference engine, API server)

## Resolved Questions
- **QMatMul kernel gaps:** We write custom Metal kernels for any K-quant types candle doesn't cover. No dequantize fallback.
- **Speed target:** Match or beat llama.cpp on same hardware, verified with real benchmarks.
- **No-GPU policy:** Hard fail — refuse to start without a supported GPU (Metal, CUDA, or Vulkan/oneAPI).
- **Code restore strategy:** Spec-layer files (schema, SSE, tool parsing) restored after deep review; inference-path code rebuilt from scratch (old code was MLX-based).
- **Concurrency model:** Continuous batching from day one, not a simple queue.
- **Vision mmproj:** hf2q produces mmproj AND accepts user-provided files.
- **Chat template priority:** CLI > GGUF metadata > tokenizer_config.json.
- **Embeddings:** Pooled output from loaded model by default; `--embedding-model` for dedicated embedding model.
- **Tool calling:** Template-driven, model-agnostic — schemas injected via chat template.
- **Resumable quantization:** Restart from scratch for now; design for future resumability.

## Resolved Questions (continued)
- **Model implementations:** Use candle-transformers when they meet our performance bar. Write our own only for unsupported architectures or state-of-the-art optimization on specific models.
- **CoreML/ANE:** Keep as a later phase. Requires dedicated research to find where ANE beats Metal GPU — no arbitrary size cutoff. coreml-native crate is available when ready.
- **GGUF provenance:** Detect hf2q origin via GGUF metadata. Serve any valid GGUF regardless of who made it. Apply hf2q-specific optimizations (trusted metadata, skip extra validation) when provenance is confirmed.
- **MoE kernel quant format (2026-04-09):** Write ggml K-quant dequant kernels, not MLX affine. Rationale: hf2q produces K-quant GGUF; llama.cpp proves K-quants are fast on Metal; switching formats would break ecosystem compatibility. mlx-native's infrastructure (device, encoder, registry, dispatch patterns) is reusable; only the innermost dequant loop differs.
- **SDPA scale omission (2026-04-09):** Gemma 4 intentionally uses `scaling = 1.0` (no `1/sqrt(head_dim)`). Per-head Q/K RmsNorm normalizes dot-product magnitudes, making the traditional scale unnecessary. Verified against HuggingFace `modeling_gemma4.py`.
- **Phase 1 vs 1b boundary (2026-04-09):** Phase 1 = correctness (functional inference, coherent output). Phase 1b = performance (match llama.cpp speed). These are separate milestones with separate acceptance criteria.
