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
   - Current baseline: 100 tok/s decode on M4 via llama.cpp with our GGUF

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

### Phase 1: Text Inference (candle QMatMul) -- COMPLETE (2026-04-09)
- [x] Load GGUF with candle's quantized tensor system
- [x] Forward pass using QMatMul (no dequantize) for all linear layers (attention, MLP, MoE router)
- [x] MoE expert weights pre-sliced at load time (eliminates per-token narrow/squeeze/contiguous)
- [x] KV cache for efficient decode
- [x] `hf2q generate` CLI working, produces coherent correct output
- [x] 6 forward pass bugs fixed: GELU variant, MLP/MoE parallel, softmax routing, router input processing, per-expert scale, norm assignments
- [x] 2 GGUF norm name mismatches fixed (post_attention_norm, ffn_norm)
- [x] Chat template: uses GGUF Jinja-style control tokens (<|turn>, <|think|>, <turn|>)
- [x] Benchmarked: 18 tok/s decode (M5 Max) after GPU routing + batched MoE — llama.cpp does 107 tok/s
- [x] GPU-side top-k routing (arg_sort + gather on GPU, only 8 values pulled to CPU)
- [x] Batched expert matmuls (Tensor::stack + batched matmul, 480→60 dispatches per token)
- [ ] Speed gap: 18 vs 107 tok/s — requires custom Metal mul_mat_id kernel (see below)

#### Phase 1b: Speed Optimization (PLANNED)

**Profiled decode breakdown (per layer, single token, M5 Max):**
| Component | Time | % | Notes |
|-----------|------|---|-------|
| Attention (sliding) | 0.6ms | 19% | Fast, QMatMul efficient |
| Attention (global, every 6th) | 5-8ms | — | 10x spike, needs investigation |
| Dense MLP | 0.4ms | 13% | Fast |
| MoE experts | 1.5-8ms | 55-65% | #1 bottleneck, high variance from Tensor::stack allocs |
| Rest (norms, residual) | 0.3ms | 9% | Negligible |
| **Total per layer** | **2.6-4.0ms** | | 30 layers → ~90ms/token → ~11 tok/s (18 without profiling overhead) |

**Steady-state decode breakdown (confirmed via profiling, token 2+):**
- Attention: 0.5ms/layer (fast, stable)
- MLP: 0.3ms/layer (fast, stable)
- MoE: 1.5-2.5ms/layer (60-70% of total, high variance from Tensor::stack allocs)
- Total: ~80ms/30 layers = ~12 tok/s with profiling, 18 tok/s without
- Target: llama.cpp ~0.31ms/layer = 9.3ms/30 layers = 107 tok/s
- **Gap is ~6x per layer across ALL components, not just MoE. No silver bullet — need 10-15 optimizations.**

**Approach: measure → change → measure. Profile after every optimization.**

#### Tier 1: MoE dispatch (biggest single bottleneck, 60-70% of decode)

**1b.1: Custom Metal mul_mat_id kernel** (highest impact, hardest)
- Port llama.cpp's `kernel_mul_mv_id` (MV decode path) — indexed matmul into 3D quantized GGUF tensor
- Leverage mlx-native crate (`/opt/mlx-native`): kernel_registry.rs, device.rs, encoder.rs, buffer.rs
- Reuse mlx-native's moe_gate.metal (GPU-side router), moe_dispatch.metal (fused GELU+accumulate)
- New kernel: `moe_expert_qmatmul.metal` with inline ggml dequant (Q4_K, Q6_K, Q8_0)
- Load expert weights as merged 3D QTensors (not dequantized) — kernel indexes by expert_id * stride
- Dispatch via metal-rs alongside candle buffers (QMetalStorage::buffer() is public)
- Expected: MoE from ~2.0ms/layer to ~0.3ms/layer

**1b.2: Pre-allocate MoE expert weight stacks** (medium impact, easy)
- Current `Tensor::stack` allocates a new [top_k, hidden, intermediate*2] tensor per token per layer
- Pre-allocate a reusable buffer at model load time, copy selected expert weights into it
- Eliminates allocation jitter (MoE variance 1.3-3.9ms → stable ~1.3ms)

**1b.3: Keep expert weights as 3D QTensor** (memory, comes with 1b.1)
- Currently dequantize 3D → BF16 → pre-slice at load (~45 GB). With 1b.1, keep as quantized 3D (~28 GB)
- Saves ~17 GB GPU memory

#### Tier 2: Reduce per-op overhead across all layers

**1b.4: Eliminate F32↔BF16 round-trips in QLinear** (medium impact, easy)
- Every QLinear does: BF16→F32 (input cast) → QMatMul(F32) → F32→BF16 (output cast)
- 4 attention + 3 MLP + 1 router = 8 cast pairs per layer = 480 casts per token
- Option A: run entire forward in F32 (tested — same output, no precision issue)
- Option B: check if candle's Metal QMatMul actually needs F32 or if BF16 works on M5

**1b.5: Reduce unnecessary `.contiguous()` calls** (small-medium impact, easy)
- Each `.contiguous()` is a GPU memcpy. Audit all contiguous calls in forward path
- Some may be needed (Metal kernel requirements), some may be defensive

**1b.6: Profile and eliminate hidden GPU syncs** (unknown impact, investigation)
- candle may internally call `commit_and_wait` more than we realize
- Use Metal GPU capture or Instruments to count actual command buffer commits per token

#### Tier 3: Attention and KV cache

**1b.7: Pre-allocated KV cache (fixed max_seq_len)** (medium impact)
- Currently `Tensor::cat` on every decode step creates a new tensor
- Pre-allocate [max_seq_len, heads, head_dim] and write to a slice each step
- Eliminates allocation + copy per layer per token

**1b.8: Flash attention** (medium impact for long sequences)
- candle supports flash attention — reduces memory bandwidth for attention computation
- Biggest win during prefill (many tokens) and long-context decode

**1b.9: Sliding window attention** (medium impact)
- Sliding attention layers (24 of 30) only attend to last 1024 tokens
- Currently attend to full KV cache — wasted computation for long sequences
- Implement sliding window mask to skip old KV entries

#### Tier 4: Fused kernels

**1b.10: Fused RmsNorm + residual add** (small impact)
- 7 RmsNorm calls per layer, each a separate kernel dispatch
- Fuse norm + residual add into single kernel (mlx-native has fused_residual_norm_bf16.metal)

**1b.11: Fused GELU + element-wise multiply in MLP/MoE** (small impact)
- gate → GELU → mul(up) currently 3 ops, could be 1 (mlx-native has fused_gelu_mul in moe_dispatch.metal)

**1b.12: Use F16 intermediates instead of BF16** (small impact)
- Metal is generally faster with F16 than BF16 for element-wise ops
- Would need validation that F16 precision is sufficient

#### Tier 5: Infrastructure

**1b.13: Command buffer batching** (unknown impact)
- Submit multiple operations to Metal command encoder before committing
- Reduces driver overhead per operation

**1b.14: Warmup dummy token at server startup** (TTFT, Phase 2)
- Global attention layers spike 5-13ms on first decode token (~37ms extra TTFT)
- Run dummy BOS token through model at startup, clear KV cache
- Zero cost to steady-state throughput

**1b.15: Eliminate per-token MoE loop for prefill** (prefill speed)
- During prefill, all tokens could be processed together through MoE
- Currently each token loops independently — N×8 expert matmuls instead of batched

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

### Phase 1 + 3: Inference Engine (Text + Vision)
- [ ] `hf2q generate --model ./models/gemma4/ --prompt "..."` produces correct, coherent text output
- [ ] Vision: `hf2q generate --model ./models/gemma4/ --prompt "describe this" --image photo.jpg` produces correct image-aware output
- [ ] Benchmarked decode speed matches or exceeds llama.cpp and mlx-lm on the same hardware (M5 Max, Gemma 4 26B MoE, Q4_K_M)
- [ ] Benchmarked prefill speed matches or exceeds same baselines
- [ ] All K-quant types produced by hf2q load and infer correctly (no dequantize fallback)
- [ ] Primary model: Gemma 4 26B MoE. Fast follow: Qwen3.5 and additional architectures

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
