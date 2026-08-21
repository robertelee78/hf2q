# ADR-008: Full Candle Divorce — Port to Pure mlx-native

**Status:** Accepted. Implemented 2026-04-14 in commit 9e23b7d ("feat(ADR-008): full candle divorce — pure mlx-native inference"); ~47k lines of candle-derived code deleted, hf2q's forward path runs entirely on mlx-native.
**Date:** 2026-04-15 (Proposed) → 2026-04-14 (Implemented in 9e23b7d, recorded post-hoc)
**Decision Makers:** Robert, Claude
**Related ADRs:** ADR-006 (mlx-native GPU backend), ADR-007 (TurboQuant KV cache), ADR-005 (inference server)

---

## Engineering Mantra (load-bearing — read before every session)

Source: `~/Documents/mantra.txt` (Robert, undated). Quoted verbatim.

> **DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.**

**Operational reading for this ADR:**

- **Chesterton's fence** — candle exists in this codebase because it was the fastest path to a working inference engine. It served that purpose. This ADR replaces it not because it's bad, but because we own mlx-native and can't match llama.cpp performance through someone else's framework.
- **No fallback** — no `#[cfg]` dual-path, no "candle as optional backend." Once mlx-native owns each responsibility, the candle code is deleted. Clean stack.
- **Measure 3x** — each phase has correctness gates (logit match, coherence test) before proceeding.

---

## Problem Statement

The hf2q inference engine currently depends on candle (Hugging Face's Rust ML framework) for three distinct responsibilities:

1. **GGUF file format parsing** — reading tensor metadata, offsets, and raw block data from `.gguf` files
2. **Quantized weight storage** — `QTensor` and `QMatMul` types that hold GGML block data in Metal buffers
3. **Forward pass execution** — the candle-based Gemma4 forward pass (3,493 lines) with candle's Metal kernels

The mlx-native backend (ADR-006) has already replaced responsibility #3 for production inference. But responsibilities #1 and #2 still force every model load to flow through candle:

```
GGUF file → candle gguf_file::Content::read() → candle QTensor → gpu.rs bridge → MlxBuffer
```

This creates four problems:

**P-1: Weight loading is a CPU-GPU-CPU-GPU round-trip.** Candle loads GGUF data into its own Metal buffers, then `gpu.rs` copies bytes from candle Metal storage → CPU → mlx-native Metal buffer. For Gemma-4-27B, this copies ~15 GB of weight data through CPU memory unnecessarily.

**P-2: Two Metal device contexts.** Candle creates its own `metal::Device` and command queue. mlx-native creates another. Both compete for GPU resources. The candle device is used only during model loading but persists for the process lifetime.

**P-3: ~8,000 lines of dead-on-arrival code.** `gemma4.rs` (3,493 lines), the candle kernel files (`rope_kernel.rs`, `rms_norm_kernel.rs`, `moe_kernel.rs`, `lm_head_kernel.rs` = 3,258 lines), and the vendored candle crates (`vendor/candle-nn`, `vendor/candle-metal-kernels`) are unused in production. They're maintained, compiled, and add 30s to clean builds.

**P-4: Sampling still uses candle Tensors.** The sampler (`sampler.rs`, 252 lines) operates on `candle_core::Tensor` for argmax, softmax, and repetition penalty. These are simple arithmetic operations that could be pure Rust or mlx-native GPU ops. The candle Tensor dependency forces a GPU→CPU sync per token that wouldn't otherwise be needed.

### Why now?

The mlx-native forward pass (`forward_mlx.rs`) uses **zero** candle operations at inference time. Candle exists in the hot path only because:
- The sampler takes `&Tensor` (candle type) as input
- The output logits are wrapped in a candle `Tensor` for the sampler

Every other candle dependency is cold-path (model loading). The architecture is ready for a clean cut.

---

## Decision

**Port GGUF parsing and weight loading to pure Rust in mlx-native, replace the candle sampler with pure Rust, then delete all candle code and dependencies from hf2q.**

### What stays (already in mlx-native)

These are production-ready and unchanged by this ADR:

- Quantized matmul kernels (`quantized_matmul.metal`, `quantized_matmul_ggml.metal`) — Q4_0, Q4_K, Q6_K, Q8_0
- SDPA kernels (`flash_attn_vec.metal`, `flash_attn_vec_tq.metal`, `sdpa_sliding.metal`)
- Fused kernels (`fused_head_norm_rope_bf16.metal`, `fused_norm_add_bf16.metal`, etc.)
- RoPE, RMS norm, softmax, argmax, GELU, MoE gate/dispatch kernels
- TurboQuant KV cache (ADR-007): `hadamard_quantize_kv.metal`
- Graph executor with barrier optimization
- `MlxBuffer`, `MlxDevice`, `CommandEncoder`, `KernelRegistry`

### What must be ported

| Component | Current owner | Lines | Target | Effort |
|-----------|--------------|-------|--------|--------|
| GGUF file format parser | `candle_core::quantized::gguf_file` | ~800 (in candle) | `mlx-native::gguf` module | Medium |
| GGUF tensor loading | `gguf_file::Content::tensor()` | ~200 (in candle) | `mlx-native::gguf::load_tensor()` | Medium |
| GGUF metadata access | `gguf_file::Content::metadata` | ~100 (in candle) | `mlx-native::gguf::Metadata` | Low |
| Sampler (greedy) | `sampler.rs` → `Tensor::argmax` | 30 LOC | Pure Rust `argmax` on `&[f32]` | Trivial |
| Sampler (non-greedy) | `sampler.rs` → `softmax_last_dim` + top-k/p | 120 LOC | Pure Rust softmax + sampling | Trivial |
| Repetition penalty | `sampler.rs` → `Tensor::gather` | 35 LOC | Pure Rust gather on `&[f32]` | Trivial |
| Weight loading bridge | `gpu.rs` → candle Metal → CPU → MlxBuffer | 120 LOC | Direct GGUF → MlxBuffer (zero-copy) | Medium |

### What gets deleted

| Component | File(s) | Lines | Reason |
|-----------|---------|-------|--------|
| Candle forward pass | `gemma4.rs` | 3,493 | Replaced by `forward_mlx.rs` |
| Candle RoPE kernel | `rope_kernel.rs` | 1,371 | mlx-native has `rope.metal` |
| Candle RMS norm kernel | `rms_norm_kernel.rs` | 888 | mlx-native has `rms_norm.metal` |
| Candle MoE kernel | `moe_kernel.rs` | 628 | mlx-native has `moe_gate.metal` + `moe_dispatch.metal` |
| Candle lm_head kernel | `lm_head_kernel.rs` | 371 | mlx-native has `dense_gemm.metal` |
| Candle sampler | `sampler.rs` | 252 | Replaced by pure Rust sampler |
| Candle gguf loader | `gguf_loader.rs` | 113 | Replaced by mlx-native GGUF module |
| Candle↔mlx bridge | `gpu.rs` (most of it) | ~200 | No longer needed |
| Vendored candle-nn | `vendor/candle-nn/` | ~5,000 | No longer needed |
| Vendored candle-metal-kernels | `vendor/candle-metal-kernels/` | ~10,000 | No longer needed |
| Cargo dependencies | `Cargo.toml` | — | `candle-core`, `candle-nn`, `candle-metal-kernels` |
| Legacy GPU module | `src/gpu/forward.rs`, `src/gpu/mod.rs` | ~200 | Unused |

**Total deletion estimate: ~22,000+ lines** (including vendored crates).

---

## Architecture

### Current Flow (candle-dependent)

```
CLI args
  ↓
gguf_loader.rs (candle gguf_file parser)
  ↓
gemma4.rs: Gemma4Model::load_with_modes()
  → candle Device, QTensor, Tensor, QMatMul (stored on struct)
  ↓
forward_mlx.rs: MlxModelWeights::new()
  → gpu.rs bridge: candle QTensor → CPU bytes → MlxBuffer  [P-1: wasteful copy]
  → gpu.rs bridge: candle Tensor → .to_vec1() → MlxBuffer  [P-1: wasteful copy]
  ↓
forward_mlx.rs: forward_decode()
  → 100% mlx-native ops (no candle at inference time)
  → returns logits as MlxBuffer
  ↓
mod.rs: wraps logits in candle Tensor for sampler  [P-4: unnecessary]
  ↓
sampler.rs: candle argmax/softmax → token_id  [P-4: unnecessary]
```

### Target Flow (candle-free)

```
CLI args
  ↓
mlx-native::gguf::GgufFile::open()
  → parse header, tensor metadata, offsets (pure Rust)
  ↓
forward_mlx.rs: MlxModelWeights::load_from_gguf()
  → mlx-native gguf::load_tensor_to_buffer() → MlxBuffer  [direct: file → GPU, no intermediate]
  → for quantized: read raw GGML blocks → MlxBuffer as U8
  → for dense (norms, scalars): read F32 → MlxBuffer as F32
  ↓
forward_mlx.rs: forward_decode()
  → 100% mlx-native ops (unchanged)
  → returns logits as MlxBuffer
  ↓
mod.rs: read logits to CPU via MlxBuffer::as_slice()
  ↓
sampler.rs: pure Rust argmax/softmax on &[f32] → token_id
```

### GGUF Parser Design

The GGUF file format (v3) is well-specified. Key structures:

```
GGUF Header:
  magic: u32 = 0x46475547 ("GGUF")
  version: u32 = 3
  tensor_count: u64
  metadata_kv_count: u64
  metadata_kv: [MetadataKV; metadata_kv_count]
  tensor_infos: [TensorInfo; tensor_count]
  alignment_padding
  tensor_data: [u8; ...]  ← raw GGML blocks, mmap-friendly

MetadataKV:
  key: string (length-prefixed)
  value_type: u32 (enum: UINT8..STRING..ARRAY)
  value: variant

TensorInfo:
  name: string (length-prefixed)
  n_dims: u32
  dims: [u64; n_dims]
  ggml_type: u32 (Q4_0=2, Q4_K=12, Q6_K=14, Q8_0=8, F32=0, F16=1, ...)
  offset: u64  ← from start of tensor_data section
```

**Implementation plan for `mlx-native::gguf`:**

```rust
pub struct GgufFile {
    metadata: HashMap<String, MetadataValue>,
    tensors: HashMap<String, TensorInfo>,
    tensor_data_offset: u64,  // byte offset where tensor data starts
    reader: Mutex<BufReader<File>>,
}

pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_type: GgmlType,
    pub offset: u64,   // relative to tensor_data_offset
    pub byte_len: usize,
}

impl GgufFile {
    /// Open and parse header + tensor index. Does NOT read tensor data.
    pub fn open(path: &Path) -> Result<Self>;

    /// Read raw tensor bytes into an MlxBuffer.
    /// For quantized types: returns U8 buffer with raw GGML blocks.
    /// For F32/F16: returns typed buffer.
    pub fn load_tensor(&self, name: &str, device: &MlxDevice) -> Result<MlxBuffer>;

    /// Read and dequantize a tensor to F32.
    /// Used for norm weights, scalars, and other small dense tensors.
    pub fn load_tensor_f32(&self, name: &str, device: &MlxDevice) -> Result<MlxBuffer>;

    /// Get metadata value.
    pub fn metadata(&self, key: &str) -> Option<&MetadataValue>;

    /// Get metadata as string.
    pub fn metadata_string(&self, key: &str) -> Option<&str>;

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str>;
}
```

**Key design choice: direct-to-GPU loading.**

For quantized tensors (Q4_K, Q6_K, Q8_0), the raw GGML block bytes are the GPU format — mlx-native's `quantized_matmul_ggml.metal` kernel reads them directly. So loading is:

```rust
// Read raw bytes from GGUF file
let bytes = read_tensor_bytes(name)?;
// Allocate MlxBuffer and copy bytes directly
let mut buf = device.alloc_buffer(bytes.len(), DType::U8, shape)?;
buf.as_mut_slice::<u8>()?.copy_from_slice(&bytes);
```

No dequantization, no intermediate candle QTensor, no CPU→GPU→CPU→GPU round-trip.

For dense F32 tensors (norm weights, scalars), the bytes are F32 and the loading is the same pattern with `DType::F32`.

**GGML block dequantization (for `load_tensor_f32`):**

Some tensors need CPU-side dequantization (e.g., embedding weights that get converted to F16 for the lm_head). The GGML block formats are:

| Type | Block size | Elements/block | Bytes/block | Structure |
|------|-----------|----------------|-------------|-----------|
| Q4_0 | 32 | 32 | 18 | f16 scale + 16 bytes (32 nibbles) |
| Q4_K | 256 | 256 | 144 | 2×f16 scale/min + 12 bytes scale_factors + 128 bytes data |
| Q6_K | 256 | 256 | 210 | f16 scale + 256 bytes quants_low + 128 bytes quants_high + 32 bytes scales |
| Q8_0 | 32 | 32 | 34 | f16 scale + 32 bytes (32 int8 values) |
| F32 | 1 | 1 | 4 | raw f32 |
| F16 | 1 | 1 | 2 | raw f16 |

For the GGUF parser MVP, we need Q4_K and Q6_K dequantization (used by Gemma-4-27B GGUF). Q4_0 and Q8_0 are bonuses. The dequant logic is well-documented in `ggml-common.h` and `candle-core/src/quantized/k_quants.rs`.

### Sampler Design

The sampler becomes pure Rust operating on `&[f32]` logit slices:

```rust
pub fn sample_token(
    logits: &[f32],
    params: &SamplingParams,
    previous_tokens: &[u32],
) -> u32 {
    // Greedy: argmax (scan logits, return index of max)
    // Non-greedy: softmax → top-k → top-p → multinomial
    // Repetition penalty: gather → scale → scatter
}
```

The logits come from `MlxBuffer::as_slice::<f32>()` — one GPU→CPU sync per token. This is the same sync the candle sampler did via `Tensor::to_vec1()`, so no performance change.

For greedy decode at T=0, the GPU argmax kernel in mlx-native (`argmax.metal`) can avoid the CPU roundtrip entirely — dispatch argmax on GPU, pass the u32 result buffer directly to the next forward pass's embedding gather. This is a Phase 2 optimization.

---

## Implementation Plan

### Phase 0 — Measure Baseline (Chesterton's fence + measure 3x)

**Goal:** Establish exact performance baseline for the mlx-native path so we can detect regressions.

**0.1 — Benchmark mlx-native production path**
- Run `--backend=mlx-native --benchmark` (5 runs, 64 tokens, T=0.7)
- Record: median tok/s, p95 tok/s, dispatch count, prefill tok/s
- **Also benchmark the candle path** for comparison at same context
- Record both numbers as the Phase 0 baseline

**0.2 — Coherence test**
- Run `--backend=mlx-native --temperature 0 --max-tokens 100 --prompt "Comprehensive instructions for making sourdough bread."`
- Save output to `docs/coherence-test-adr008-baseline.txt`
- This is the reference for all subsequent phases

**Gate:** Baselines recorded. No code changes in Phase 0.

### Phase 1 — GGUF Parser in mlx-native

**Goal:** Pure Rust GGUF parser that loads tensors directly into MlxBuffer.

**1.1 — GGUF header parser (`mlx-native/src/gguf/mod.rs`)**
- Parse GGUF v3 header: magic, version, tensor count, metadata count
- Parse metadata key-value pairs (all value types: uint8..string..array)
- Parse tensor info array (name, dims, ggml_type, offset)
- Compute tensor_data_offset (header + metadata + tensor_infos + alignment padding)
- **Test:** Parse the Gemma-4-27B GGUF file header, verify tensor count and metadata match candle's parse

**1.2 — Raw tensor loading (`GgufFile::load_tensor`)**
- Seek to tensor_data_offset + tensor.offset
- Read tensor.byte_len bytes
- Allocate MlxBuffer with correct shape and dtype
- Copy bytes into buffer
- **Test:** Load `blk.0.attn_q.weight` (Q4_K), verify byte-identical to candle's `QTensor::data()`

**1.3 — F32 dequantization (`GgufFile::load_tensor_f32`)**
- Implement Q4_K → F32 dequantization (256-element blocks)
- Implement Q6_K → F32 dequantization (256-element blocks)
- Implement Q8_0 → F32 dequantization (32-element blocks)
- Implement F16 → F32 conversion
- **Test:** Dequantize `blk.0.attn_norm.weight`, compare to candle's `QTensor::dequantize()` within ε < 1e-6

**1.4 — Metadata API**
- `metadata_string(key)` for chat template extraction
- `metadata_u32(key)`, `metadata_f32(key)` for model config
- **Test:** Extract `general.architecture`, `general.name`, `tokenizer.chat_template` from Gemma GGUF

**Gate:** All tensor loads byte-match candle. All metadata matches. Unit tests pass.

### Phase 2 — Direct Weight Loading

**Goal:** `MlxModelWeights::load_from_gguf()` loads directly from GGUF without candle.

**2.1 — New loading path in `forward_mlx.rs`**
- Add `MlxModelWeights::load_from_gguf(gguf: &GgufFile, device: &MlxDevice, cfg: &ModelConfig)` 
- Load quantized weights: `gguf.load_tensor("blk.{i}.attn_q.weight")` → MlxBuffer (raw GGML blocks)
- Load dense weights: `gguf.load_tensor_f32("blk.{i}.attn_norm.weight")` → MlxBuffer (F32)
- Load embedding + lm_head F16 conversion
- Load MoE expert weights (128 experts × 30 layers): raw GGML block copy to MlxBuffer
- Stacked expert buffer construction (if used)

**2.2 — Model config extraction from GGUF metadata**
- Extract: hidden_size, num_layers, num_heads, num_kv_heads, head_dim, vocab_size, etc.
- Extract: layer_types (sliding vs global), sliding_window, rope_theta, softcapping
- Currently these come from `Gemma4ModelConfig` in `gemma4.rs` which reads `config.json`
- GGUF metadata contains most of these; for any missing, fall back to `config.json` parsing

**2.3 — Wire new loading path into `mod.rs`**
- When `--backend=mlx-native`, load via `GgufFile::open()` + `MlxModelWeights::load_from_gguf()`
- Skip `Gemma4Model::load_with_modes()` entirely (no candle model creation)
- **Gate:** Identical inference output (byte-identical greedy decode at T=0) with both loading paths

**2.4 — Tokenizer loading**
- Currently tokenizer comes from `tokenizers` crate (not candle) — no change needed
- Verify `tokenizer.json` path resolution works without candle's model config

**Gate:** `--backend=mlx-native` works end-to-end without creating any candle objects. Coherence test matches Phase 0 baseline.

### Phase 3 — Pure Rust Sampler

**Goal:** Replace candle Tensor sampling with pure Rust on `&[f32]` slices.

**3.1 — Implement pure Rust sampler**
- `fn sample_greedy(logits: &[f32]) -> u32` — scan for max index
- `fn sample_with_params(logits: &[f32], params: &SamplingParams, prev: &[u32]) -> u32`
  - Temperature scaling: `logits[i] /= temperature`
  - Softmax: `exp(x - max) / sum(exp(x - max))`
  - Top-k: partial sort, truncate
  - Top-p: cumulative sum, truncate
  - Repetition penalty: `logits[id] *= (1/penalty if positive, penalty if negative)`
  - Multinomial sample from filtered distribution
- **Test:** Identical token sequences vs candle sampler on 100 greedy + 100 T=0.7 generations

**3.2 — Wire into mod.rs**
- After `forward_decode()`, read logits: `let logits: &[f32] = output_buf.as_slice()?;`
- Call `sample_greedy(logits)` or `sample_with_params(logits, ...)`
- Remove candle `Tensor` wrapping of logits
- **Gate:** Byte-identical greedy decode. Statistical equivalence for sampled decode (same distribution, verified by KL divergence < 0.001 over 1000 tokens).

### Phase 4 — Delete Candle

**Goal:** Remove all candle code, dependencies, and vendored crates.

**4.1 — Delete inference-path candle code**
- Delete `src/serve/gemma4.rs` (3,493 lines)
- Delete `src/serve/rope_kernel.rs` (1,371 lines)
- Delete `src/serve/rms_norm_kernel.rs` (888 lines)
- Delete `src/serve/moe_kernel.rs` (628 lines)
- Delete `src/serve/lm_head_kernel.rs` (371 lines)
- Delete `src/serve/sampler.rs` (252 lines — replaced by Phase 3)
- Delete `src/serve/gguf_loader.rs` (113 lines — replaced by Phase 1)
- Clean up `src/serve/gpu.rs` (remove candle bridge functions)
- Clean up `src/serve/mod.rs` (remove candle backend dispatch, candle model loading)

**4.2 — Delete vendored crates**
- Delete `vendor/candle-nn/` (~5,000 lines)
- Delete `vendor/candle-metal-kernels/` (~10,000 lines)

**4.3 — Remove Cargo dependencies**
- Remove `candle-core`, `candle-nn`, `candle-metal-kernels` from `[dependencies]`
- Remove `metal` and `cuda` feature flags that gate candle features
- Remove `[patch.crates-io]` overrides for vendored candle crates
- Remove `objc2-metal` if no longer needed (check mlx-native's own deps)

**4.4 — Delete legacy GPU module**
- Delete `src/gpu/forward.rs` and `src/gpu/mod.rs` if unused after cleanup

**4.5 — Clean up remaining references**
- `grep -r "candle" src/` should return zero matches
- `grep -r "candle" Cargo.toml` should return zero matches
- Build with `cargo build --release` (no feature flags — single backend)

**Gate:** `cargo build --release` succeeds with zero warnings. All tests pass. Coherence test matches Phase 0 baseline. Benchmark matches or exceeds Phase 0 baseline. `grep -r "candle" src/ Cargo.toml vendor/` returns nothing.

### Phase 5 — Optimization (Post-Divorce)

**Goal:** Exploit the clean architecture for performance wins.

**5.1 — Zero-copy GGUF loading via mmap**
- `mmap()` the GGUF file
- Tensor data is already page-aligned (GGUF spec guarantees alignment)
- Create MlxBuffer pointing directly to mmapped pages (Metal `newBufferWithBytesNoCopy`)
- Eliminates all model-load memcpy — startup goes from ~5s to <0.5s

**5.2 — GPU-side greedy argmax**
- Dispatch `argmax.metal` on logits MlxBuffer
- Pass u32 result buffer directly to embedding gather (next forward pass)
- Zero GPU→CPU sync for greedy decode
- Expected: +5-10% tok/s from eliminating the per-token sync

**5.3 — Single-feature build**
- No more `--features metal,mlx-native-backend` — just `cargo build --release`
- Simplify CI, testing, documentation

---

## Acceptance Criteria (End Gates)

### Functional

- [ ] **F-1:** `cargo build --release` succeeds with zero candle dependencies and zero warnings
- [ ] **F-2:** `grep -rn "candle" src/ Cargo.toml vendor/` returns zero matches
- [ ] **F-3:** `vendor/candle-nn/` and `vendor/candle-metal-kernels/` directories deleted
- [ ] **F-4:** All existing model functionality preserved: generate, serve, convert, validate
- [ ] **F-5:** GGUF files load correctly for Gemma-4-27B (Q4_K_M quantization)
- [ ] **F-6:** Coherence test: byte-identical greedy 100-token decode at T=0 vs Phase 0 baseline
- [ ] **F-7:** Tokenizer, chat template, and metadata extraction work from GGUF without candle

### Performance

- [ ] **P-1:** Decode speed ≥ Phase 0 baseline (no regression from architecture change)
- [ ] **P-2:** Model loading time ≤ Phase 0 baseline (weight loading should be faster without CPU roundtrip)
- [ ] **P-3:** Clean build time reduced (removing ~25,000 lines of compiled code)

### Engineering

- [ ] **E-1:** GGUF parser has unit tests covering: header parse, metadata extraction, Q4_K tensor load, Q6_K tensor load, F32 tensor load, byte-match vs candle reference
- [ ] **E-2:** Pure Rust sampler has unit tests covering: greedy argmax, temperature scaling, top-k, top-p, repetition penalty, statistical distribution match
- [ ] **E-3:** No `#[allow(dead_code)]` bandaids — all code is live or deleted
- [ ] **E-4:** Each phase commits + pushes on completion (per `feedback_commit_push_cadence.md`)
- [ ] **E-5:** Zero `#[cfg]` dual-path conditionals for candle vs mlx-native

---

## Risks & Mitigations

| # | Risk | Severity | Likelihood | Mitigation |
|---|------|----------|------------|------------|
| R-1 | GGUF parser misses an edge case in the format spec | Medium | Low | Test against 3+ different GGUF files; cross-validate every tensor load against candle's output |
| R-2 | Q4_K/Q6_K dequantization has subtle numeric differences | Medium | Medium | Bit-exact comparison against candle's dequant for every block. The GGML format is precisely specified. |
| R-3 | convert/validate subcommands break without candle | Low | Medium | These commands use safetensors and IR, not candle tensors. Audit before Phase 4. If they do use candle, port or gate behind optional dep. |
| R-4 | Model loading is slower without candle's optimized paths | Low | Low | Our path is simpler (file → buffer, no intermediate). Measure in Phase 2. |
| R-5 | Some GGUF metadata keys have different names than config.json | Medium | Medium | Cross-reference llama.cpp's GGUF metadata key names. Fall back to config.json for missing keys. |

---

## Dependency Inventory (for deletion verification)

### Cargo.toml entries to remove

```toml
# Direct dependencies
candle-core = "0.10"
candle-nn = "0.10"
candle-metal-kernels = { version = "0.10", optional = true }

# Feature flags
metal = ["candle-core/metal", "candle-nn/metal", "dep:candle-metal-kernels", "dep:objc2-metal"]
cuda = ["candle-core/cuda", "candle-nn/cuda"]
mlx-native-backend = ["dep:mlx-native", "metal"]  # ← redefine without candle

# Patch overrides
[patch.crates-io]
candle-nn = { path = "vendor/candle-nn" }
candle-metal-kernels = { path = "vendor/candle-metal-kernels" }
```

### Files to delete (22,000+ lines)

```
src/serve/gemma4.rs            3,493 lines
src/serve/rope_kernel.rs       1,371 lines
src/serve/rms_norm_kernel.rs     888 lines
src/serve/moe_kernel.rs          628 lines
src/serve/lm_head_kernel.rs      371 lines
src/serve/sampler.rs             252 lines  (replaced by pure Rust)
src/serve/gguf_loader.rs         113 lines  (replaced by mlx-native)
src/gpu/forward.rs                ~100 lines
src/gpu/mod.rs                    ~100 lines
vendor/candle-nn/              ~5,000 lines
vendor/candle-metal-kernels/  ~10,000 lines
```

### Files to modify

```
src/serve/mod.rs         — remove candle loading path, rewire to mlx-native GGUF
src/serve/forward_mlx.rs — add load_from_gguf(), remove candle type refs
src/serve/gpu.rs         — remove candle bridge, keep MlxDevice/GpuContext
Cargo.toml               — remove all candle deps and features
```

---

## What This ADR Does NOT Cover

- **Multi-model support** — only Gemma-4-27B GGUF is validated. Other architectures (Qwen-3, Llama-3) are future work.
- **GGUF writing** — hf2q's `convert` command writes GGUF via its own backend (`src/backends/gguf.rs`), which doesn't use candle. No change needed.
- **Prefill optimization** — prefill is per-token in the mlx-native path. Batched prefill is a separate optimization.
- **Performance parity with llama.cpp** — this ADR removes candle. Closing the remaining speed gap is separate engineering work.
