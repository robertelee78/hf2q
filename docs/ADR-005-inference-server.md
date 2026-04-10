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

#### Phase 1b: Speed Optimization (IN PROGRESS — 23.8 tok/s correct, target 107 tok/s)

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
- [ ] Tier gates met: ≥40 tok/s after Tier 1, ≥70 tok/s after Tier 2, ≥100 tok/s after Tier 3+4

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
