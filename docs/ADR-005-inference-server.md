# ADR-005: Inference Server — OpenAI-Compatible API for hf2q

**Status:** Planned  
**Date:** 2026-04-09  
**Decision Makers:** Robert, Claude

## Context

hf2q can now quantize HuggingFace models to llama.cpp-compatible GGUF format. The next step is serving these models via an OpenAI-compatible HTTP API, enabling use with Open WebUI and other clients. The goal is a complete pipeline: download → quantize → serve, competitive with ollama/vLLM/llama.cpp server.

## Product Requirements

### Core Functionality
1. **OpenAI-compatible HTTP API** on user-specified port
   - `POST /v1/chat/completions` — streaming SSE + non-streaming
   - `GET /v1/models` — model listing
   - `POST /v1/embeddings` — embedding generation
   - `GET /health` — health check
   - `GET /metrics` — performance metrics
   - Tool calling / function calling support

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
   - Resumable HF downloads, restart quantization from scratch if interrupted

5. **Chat templates** — dynamic from model files (`tokenizer_config.json`, `chat_template.jinja`), CLI override with `--chat-template`

### Performance
6. **Speed target:** competitive with ollama/mlx-lm/llama.cpp on same hardware (real benchmarks required)
   - QMatMul on quantized tensors (no dequantize-to-float)
   - Metal GPU acceleration (primary)
   - Current baseline: 100 tok/s decode on M4 via llama.cpp with our GGUF

7. **Concurrency:** request queue with configurable depth, semaphore-based backpressure

### Platform
8. **GPU backends:** Metal (primary), CUDA (second), Intel/AMD Vulkan/oneAPI (future)
   - No CPU-only mode — not our target user

9. **Logging:** `-v` verbose flag, request logs with tok/s, memory usage, queue depth

### Future Phases
10. Multi-model hot-swap (ollama-style)
11. Additional architectures: Qwen3, Mistral
12. CoreML/ANE backend for small models

## Architecture Decisions

### Inference Engine: candle + QMatMul
Use candle's quantized tensor primitives (`QTensor`, `QMatMul`) for direct inference on GGUF-loaded quantized weights. No dequantization step. candle provides Metal and CUDA kernels for quantized matmul.

The existing `serve/gemma4.rs` MoE implementation is the reference for the forward pass — it already handles dual head dims, SigMoE routing, tied K=V, and partial RoPE. Upgrade it to use QMatMul instead of dequantized Tensors.

### Vision Pipeline
Candle's `gemma4/vision.rs` provides the ViT encoder. Our GGUF mmproj file contains the vision tower weights. Pipeline:
1. Load image → preprocess (resize, normalize, patch)
2. ViT forward pass (from mmproj weights)
3. Multimodal embedder projects vision features → text dim
4. Replace `<image>` token positions in text embeddings
5. MoE text decoder forward pass

### API Layer: Restore + Adapt Prior Code
The prior OpenAI API implementation (deleted in commit fe54bc2) had 5,868 lines of production code: handlers, schema, SSE streaming, tool parsing, embeddings. The schema and SSE protocol layers are reusable (pure spec compliance). The inference integration needs rebuilding for candle + GGUF.

Restore from git history:
- `schema.rs` (1,215 lines) — OpenAI request/response types
- `sse.rs` (681 lines) — Server-Sent Events streaming
- `tool_parser.rs` (867 lines) — tool call extraction
- `router.rs` + `middleware.rs` — axum routes + CORS

Rebuild:
- `handlers.rs` — rewire to candle inference engine
- `mod.rs` — AppState, GenerationQueue, model loading

### Dependencies (already in Cargo.toml)
- `tokio` (full) — async runtime
- `axum` 0.7 — HTTP framework
- `tower-http` 0.5 — CORS middleware
- `uuid` — request IDs
- `futures`, `async-stream` — SSE streaming
- `tokenizers` 0.22 — tokenization

## Implementation Phases

### Phase 1: Text Inference (candle QMatMul)
- Load GGUF with candle's quantized tensor system
- Forward pass using QMatMul (no dequantize)
- KV cache for efficient decode
- `hf2q generate` CLI working with proper speed

### Phase 2: HTTP Server
- Restore schema + SSE + tool parsing from git history
- Build new handlers for candle-based inference
- Queue with semaphore backpressure
- `hf2q serve --model X.gguf --port 8080`

### Phase 3: Vision
- Image preprocessing pipeline (image crate)
- Load mmproj GGUF for vision tower
- ViT forward pass + multimodal embedding injection
- `--image` flag for CLI, base64 for API

### Phase 4: Auto Pipeline
- `hf2q serve --model google/gemma-4-27b-it` → download + quantize + serve
- Hardware detection for optimal quant selection
- `~/.cache/hf2q/` caching with integrity checks

### Phase 5: Multi-Model + Architectures
- Hot-swap model loading (ollama-style)
- Qwen3, Mistral model support via candle-transformers
- Model discovery from cache directory

## Open Questions
- Should we use candle-transformers' existing model implementations (Qwen3, Mistral) or maintain our own?
- CoreML/ANE backend via coreml-native for models under 15B?
- Should `hf2q serve` auto-detect if a GGUF was made by hf2q vs downloaded from HuggingFace Hub?
