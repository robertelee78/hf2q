# ADR-004: GGUF Compatibility with llama.cpp

**Status:** Implemented  
**Date:** 2026-04-09  
**Decision Makers:** Robert, Claude

## Context

hf2q converts HuggingFace models to quantized GGUF format. The GGUF files must load and produce correct output in llama.cpp, ollama, and any tool using the ggml ecosystem. Prior to this work, hf2q's GGUF output was rejected by llama.cpp due to multiple format issues.

## Decisions

### 1. Q4_0/Q8_0 Block Repacking

**Problem:** hf2q's quantizer produces packed nibbles + separate scales. GGUF Q4_0 format requires interleaved 18-byte blocks (2-byte f16 scale + 16 packed nibbles per 32 elements).

**Decision:** Repack hf2q's internal format into proper ggml block format during GGUF writing. Streaming repack (one tensor at a time) to avoid OOM on 26B+ models.

**Alternatives considered:** Write raw data and let llama-quantize re-quantize. Rejected because we want pre-quantized GGUFs.

### 2. Dimension Ordering

**Problem:** GGUF stores dimensions in ggml order (innermost first), opposite of PyTorch/safetensors (outermost first).

**Decision:** Reverse shape dimensions with `.iter().rev()` in `write_tensor_info()`. Data bytes stay in row-major order (unchanged from safetensors).

### 3. Architecture-Aware Tensor Name Mapping

**Problem:** Same HF tensor name maps to different GGUF names depending on architecture (e.g., `post_attention_layernorm` → `ffn_norm` for LLaMA but `post_attention_norm` for Gemma4).

**Decision:** Per-arch layer_map tables selected by `model_type`. Shared base entries + architecture-specific overrides.

### 4. Text/Vision Split (mmproj)

**Problem:** llama.cpp expects multimodal models as two separate GGUF files.

**Decision:** Automatically produce both `model.gguf` (text) and `model-mmproj.gguf` (vision) when vision tensors are present. One convert command, two files.

### 5. Tokenizer Embedding

**Problem:** llama.cpp requires tokenizer data (vocab, merges, special tokens) embedded in the GGUF.

**Decision:** Parse `tokenizer.json` and `tokenizer_config.json` from the model directory. Write all required keys including `tokenizer.ggml.model`, `tokens`, `merges`, `scores`, `token_type`, special token IDs, `pre`, and `chat_template`.

### 6. V=K Duplicate Tensors

**Problem:** Gemma4 full-attention layers tie V to K — no separate `v_proj.weight` in safetensors. llama.cpp expects `attn_v.weight` to exist.

**Decision:** Duplicate K tensor data as V for full-attention layers (5, 11, 17, 23, 29).

### 7. K-Quant Block Size Fallback

**Problem:** K-quant types (Q6_K, Q5_K, etc.) require `ne[0] % 256 == 0`. MoE expert tensors with `intermediate_size=704` don't satisfy this.

**Decision:** Match llama.cpp's `tensor_type_fallback()`: Q6_K→Q8_0, Q5_K→Q5_1, Q4_K→Q5_0, Q3_K/Q2_K→Q4_0. This is the SOTA behavior used by all major quantizers.

### 8. Layer-Streaming Forward Pass

**Problem:** Loading all 30 Gemma4 layers into GPU memory for activation-based DWQ calibration exceeds 128GB (48GB TensorMap + 48GB Candle tensors).

**Decision:** Load one transformer layer at a time from TensorMap. Peak memory: ~54GB instead of ~96GB.

### 9. 1D Tensors as F32

**Problem:** llama.cpp Metal kernels assert F32 for element-wise binary operations.

**Decision:** Convert all 1D preserved tensors (norms, scales, scalars) from F16 to F32 during GGUF writing.

### 10. Synthetic rope_freqs Tensor

**Problem:** Gemma4's partial RoPE on full-attention layers requires frequency scaling factors not present in safetensors.

**Decision:** Generate `rope_freqs.weight` as `[1.0]*n_rot + [1e30]*n_unrot` (matching llama.cpp's converter).

## Results

- GGUF loads in llama.cpp with correct, coherent text output
- 100+ tok/s decode on Apple M4 Metal
- 2.9x compression (48GB → 16.8GB)
- Mixed-precision: Q4_0 base + Q6_K/Q8_0 for sensitive layers
