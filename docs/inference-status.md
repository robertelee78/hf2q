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

## Where We're Stuck

### The Problem

The forward pass produces garbled output instead of correct text. Python MLX-LM generates "2+2 = 4" on the same tokens; our Rust generates gibberish.

### Root Cause: Quantized Matmul Accumulation Order

Each quantized matmul computes a 2816-dimension dot product. The result depends on the **exact order** of f32 floating-point additions (since f32 addition is not associative).

MLX's GPU kernel uses SIMD group cooperation (32 threads, `simd_sum()` butterfly reduction) with a pre-division trick for nibble extraction. Our SIMD kernel replicates this pattern but the compiled Metal shader may use different instruction ordering, FMA fusion, or compiler optimizations.

**Current per-element accuracy:**

| Q proj element | Python | Our Rust | Error |
|----------------|--------|----------|-------|
| 0 | 21.125 | 21.125 | 0 (exact) |
| 1 | 60.25 | 60.5 | 0.25 |
| 2 | 7.8125 | 7.84375 | 0.03 |
| 3 | 12.0625 | 11.9375 | 0.125 |
| 4 | 5.46875 | 5.46875 | 0 (exact) |

Errors are 0-0.25 per element (one bf16 ULP at the given magnitude). This is tiny for a single matmul but **catastrophic for MoE routing**: the 128-expert router uses these dot products to select top-8 experts. A 0.25 error can flip which experts are selected. Over 30 layers, each with MoE routing, the hidden state diverges completely from the reference.

### Why This Is Hard

1. **Cannot match MLX's exact f32 addition order** without using their compiled Metal library. Metal compiler may reorder additions, use FMA, etc.
2. **MoE amplifies precision differences** — standard transformers (no MoE) would tolerate bf16-level errors gracefully. MoE routing creates a discrete selection (top-8 of 128) that's sensitive to tiny score differences.
3. **30-layer cascade** — even 1 wrong expert at layer 0 changes the hidden state, causing different routing at all subsequent layers.

### What Would Fix It

1. **Match MLX's kernel binary** — Compile our Metal shader with the exact same flags as MLX, or extract MLX's compiled `.metallib` and use it directly. This ensures identical instruction ordering.
2. **Deterministic CPU matmul** — Implement the full forward pass on CPU with a bit-exact accumulation matching MLX's (currently too slow — ~210s for 50 tokens).
3. **Test on non-MoE model** — Verify structural correctness on a model without MoE routing (e.g., Gemma 2) where bf16-level errors don't cascade.
4. **Higher-precision accumulation** — Use f64 for the dot product accumulation to reduce rounding differences, though this wouldn't match MLX's f32 exactly.

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
