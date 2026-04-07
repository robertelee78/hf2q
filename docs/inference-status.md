# Pure Rust Inference: Status & Remaining Work

## What We Built

A pure Rust + Metal inference engine for Gemma 4 (9B, 4-bit quantized, 128-expert MoE) at `/opt/hf2q/src/inference/models/gemma4.rs` backed by custom Metal GPU kernels at `/opt/mlx-native/src/shaders/`.

### Metal Shader Inventory (all ours, zero C/C++)

| Shader | bf16 variant | Status |
|--------|-------------|--------|
| `quantized_matmul.metal` | native bf16 dequant + SIMD cooperative | ✅ |
| `rms_norm.metal` | bf16 input/weight/output | ✅ |
| `sdpa.metal` | bf16 I/O, f32 accumulation | ✅ |
| `sdpa_sliding.metal` | bf16 I/O with sliding window | ✅ |
| `rope.metal` | bf16 I/O | ✅ |
| `elementwise.metal` | bf16 add/mul | ✅ |
| `embedding.metal` | bf16 scales/biases | ✅ |
| `softcap.metal` | f32 (final logits only) | ✅ |

### Forward Pass Pipeline

The hidden state flows as bf16 Metal buffers throughout:

```
Embedding (GPU, bf16 output)
  → Scale by bf16(sqrt(hidden_size))
  → 30 Transformer Layers:
      → RMS norm (GPU bf16 kernel)
      → Q/K/V projections (GPU SIMD quantized matmul)
      → QK norms (CPU, bf16 I/O)
      → V norm (CPU, bf16 I/O)
      → RoPE (CPU, bf16 I/O)
      → KV cache append
      → SDPA attention (GPU bf16 kernel)
      → O projection (GPU SIMD quantized matmul)
      → Residual + post-attn norm (GPU bf16)
      → Dense MLP + MoE experts (CPU matmul with bf16 operands)
      → Residual + layer scalar (bf16)
  → Final RMS norm (GPU bf16 kernel)
  → lm_head projection (GPU quantized matmul → f32 logits)
  → Softcap (CPU, f32)
  → Sampling
```

## Bugs Found & Fixed

### 1. k_eq_v restricted to Global layers only
**File:** `gemma4.rs:939`
**Bug:** `let use_k_eq_v = self.config.attention_k_eq_v && layer_type == LayerAttentionType::Global`
**Fix:** `let use_k_eq_v = self.config.attention_k_eq_v`
**Impact:** 25 of 30 layers computed V from wrong weights. Pre-softcap max logit went from ~15 to ~31.

### 2. GPU quantized matmul read bf16 scales as f16
**File:** `quantized_matmul.metal:78-79`
**Bug:** `device const half* scales` (half = IEEE f16, but data is bf16)
**Fix:** `device const uint16_t* scales` + `as_type<bfloat>(scales[i])`
**Impact:** Logits went from ~5800 (completely wrong) to ~31 (correct range).

### 3. Embedding scale used f32 instead of bf16
**File:** `gemma4.rs:404`
**Bug:** `let scale = (hidden_size as f32).sqrt()` → 53.066
**Fix:** `let scale = half::bf16::from_f32((hidden_size as f32).sqrt()).to_f32()` → 53.0
**Impact:** 0.12% error on every embedding value, propagating through all layers.

### 4. Chat template missing enable_thinking
**File:** `chat_template.rs:148`
**Bug:** Jinja context didn't set `enable_thinking => true`
**Fix:** Added `enable_thinking => true` to render context
**Impact:** Wrong prompt format: missing BOS, missing system turn with `<|think|>`, extra `<|channel>thought` tags.

### 5. BOS token resolved to empty string
**File:** `engine.rs:222-226`
**Bug:** `self.tokenizer.decode(&[bos_id])` returns "" because decode skips special tokens
**Fix:** `self.tokenizer.id_to_token(bos_id)` returns "<bos>"
**Impact:** BOS token missing from prompt, producing 20 tokens instead of 22.

## Bugs Found & Fixed (Session 2 — 2026-04-07)

### 6. SDPA kernels hardcoded attention scale
**Files:** `sdpa.metal`, `sdpa_sliding.metal`, `sdpa.rs`, `sdpa_sliding.rs`, `gemma4.rs`
**Bug:** All SDPA kernels computed `scale = 1/sqrt(head_dim)` internally. Gemma 4 requires `scale = 1.0` (QK norms handle scaling). The workaround of pre-multiplying Q by `sqrt(head_dim)=16` in bf16 introduced 16x worse precision in attention scores.
**Fix:** Added `scale` field to SDPA params structs (both Metal and Rust). Caller passes `scale = 1.0` for Gemma 4. Removed Q pre-scaling hack from `gemma4.rs`.
**Impact:** Attention scores for later tokens no longer garbled by precision loss. Logit statistics now match Python reference (mean -18.65 vs -18.63).

### 8. k_eq_v incorrectly applied to ALL layers instead of global-only
**File:** `gemma4.rs:939`
**Bug:** `let use_k_eq_v = self.config.attention_k_eq_v` applied K=V sharing to ALL 30 layers.
**Fix:** `let use_k_eq_v = self.config.attention_k_eq_v && layer_type == LayerAttentionType::Global`
**Impact:** 25 sliding layers used k_proj weights for V instead of separate v_proj weights. This was a regression from the original Bug #1 fix — the original code was correct to restrict k_eq_v to global layers; the "fix" that removed the restriction was wrong.
**Reference:** Python: `self.use_k_eq_v = config.attention_k_eq_v and not self.is_sliding`

### 7. RoPE used wrong pair convention (traditional vs Neox/split)
**File:** `gemma4.rs:1765-1788`
**Bug:** CPU RoPE paired consecutive dimensions `(d[2i], d[2i+1])` (traditional convention). Gemma 4 uses `rope_traditional=False`, meaning pairs are `(d[i], d[i + dim/2])` (Neox/split convention).
**Fix:** Changed RoPE loop to pair `(d[i], d[i + half_rope])` with frequency `1/theta^(2i/rope_dim)`.
**Impact:** Q/K values for positions > 0 were completely wrong (different signs and magnitudes). Layer 0 attention output for last token now matches Python within ULP. This was the primary cause of garbled output.
**Note:** The GPU RoPE shader (`rope.metal`) also uses the traditional convention but is not currently used by gemma4.rs (CPU path only). If the GPU path is enabled in the future, it will also need to be fixed.

## Current Status (After Fixes 6, 7, 8) — WORKING

### Inference is producing correct, coherent output!

Example: prompt "What is 2+2?" generates:
```
thought
*   Question: "What is 2+2?"
    *   Subject: Basic arithmetic.
    *   Goal: Provide the correct answer.
    *   The sum of 2 and 2.
    *   $2 + 2 = 4$.
    *   Simple: "4"
    *   Formal: "The sum of 2 and 2 is 4."
```

### Remaining Work
1. **GPU RoPE shader** — `rope.metal` still uses the traditional pair convention. If the GPU path is enabled, it needs to support non-traditional (Neox/split) convention.
2. **Performance** — prefill ~0.4 tok/s, decode ~0.3 tok/s. Significant optimization opportunity.
3. **Multi-turn / longer context** — needs testing beyond short prompts.

## Architecture

```
/opt/hf2q/                              # Main Rust crate
  src/inference/
    models/gemma4.rs                    # Gemma 4 forward pass (~3200 lines)
    engine.rs                           # Autoregressive generation loop
    kv_cache.rs                         # Sliding + global KV cache
    sampler.rs                          # Temperature/top-p/top-k sampling
    weight_loader.rs                    # Safetensors loading
  src/tokenizer/
    chat_template.rs                    # Jinja2 template rendering
    mod.rs                              # HuggingFace tokenizer wrapper

/opt/mlx-native/                        # Pure Rust Metal GPU crate (zero C/C++)
  src/shaders/                          # 13 Metal shader files
  src/ops/                              # Rust dispatch for each shader
  src/kernel_registry.rs                # Shader compilation + caching
  src/buffer.rs                         # Metal buffer management
  src/device.rs                         # Metal device abstraction
```

## Dependencies

- `mlx-rs` (optional, `mlx-backend` feature) — legacy MLX FFI path, to be removed
- `mlx-native` (our crate) — pure Rust + Metal, zero C/C++ dependencies
- `half` — bf16/f16 type support
- `metal` — Rust bindings to Apple Metal framework
- `tokenizers` — HuggingFace tokenizer
- `minijinja` — Jinja2 template engine for chat templates
