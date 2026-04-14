# ADR-007: TurboQuant KV Cache Compression for 262K Context

**Status:** Proposed  
**Date:** 2026-04-14  
**Decision Makers:** Robert, Claude  
**Related ADRs:** ADR-006 (mlx-native GPU backend — KV cache path lives here), ADR-005 (inference server — speed gates)  
**Reference:** Zandieh, Daliri, Hadian, Mirrokni — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (arXiv:2504.19874, April 2025)

---

## Engineering Mantra (load-bearing — read before every session)

Source: `~/Documents/mantra.txt` (Robert, undated). Quoted verbatim.

> **DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.**

**Operational reading for this ADR:**

- **Chesterton's fence** — the F16 KV cache and 8192 global cap exist for good reason (memory pressure). This ADR replaces them with a quantized cache that has been validated by information-theoretic bounds AND empirical benchmarks (LongBench, needle-in-haystack). We are not removing the F16 path on faith.
- **Measure 3x, cut once** — Phase 0 validates correctness (logit match) before Phase 1 touches the SDPA kernel. Phase 2 benchmarks bandwidth savings before Phase 3 lifts the context cap. No step proceeds without measurement.
- **No fallback** — the F16 KV cache path is deleted once TurboQuant is validated. No feature flag, no dual-path. Clean stack.
- **Never make assumptions** — the research swarm (5 agents, 2026-04-14) produced concrete FLOP counts, bandwidth estimates, and kernel architecture analysis. Every claim in this ADR has a derivation. Where a claim depends on an untested hypothesis (e.g., gather throughput in SDPA), it is flagged as Phase 0 measurement work.

---

## Problem Statement

Gemma-4-27B's `max_position_embeddings` is 262,144 tokens. The current inference engine caps global-layer KV cache at 8,192 positions (grep: `let max_global_kv = 8192` in `forward_mlx.rs`) because F16 storage at full context is prohibitive:

| Context length | F16 KV cache (global layers only) | Total F16 KV cache | Feasible on 192 GB M5 Max? |
|---|---|---|---|
| 8,192 (current cap) | 160 MB | 360 MB | Yes |
| 32,768 | 640 MB | 840 MB | Yes, but 1/6 of bandwidth budget |
| 131,072 | 2,560 MB | 2,760 MB | Tight |
| 262,144 (full) | **5,120 MB** | **5,320 MB** | Yes for memory, **no for bandwidth** |

At 262K context on M5 Max (546 GB/s bandwidth):
- F16 KV cache read per decode token: 5,320 MB → **9.7 ms** just for cache reads
- Current total forward pass: ~10.6 ms/token at 8K context
- At full context, KV cache bandwidth alone exceeds the entire current decode budget

**The 8192 cap is not a temporary convenience — it is a hard wall imposed by F16 storage economics.** Breaking through it requires sub-F16 KV cache representation.

### Why not simpler quantization (INT4 per-channel)?

The research swarm (Agent: quant-comparator) established:
- At 4 bits, INT4 per-channel and TurboQuant are equivalent — both near-lossless
- At 3 bits, TurboQuant beats KIVI (INT-based) by ~1 LongBench point (49.44 vs 48.50)
- At 2 bits, TurboQuant is decisively better — distribution-free bounds vs INT2's severe degradation
- **TurboQuant's random rotation eliminates outlier channels**, which plague INT-based schemes at low bit-widths

For the 262K goal, we need ≤3.5 bits to keep KV cache under ~1.2 GB. At that bit-width, TurboQuant's quality advantage over INT3/INT4 is measurable and its theoretical guarantees are proven.

---

## Decision

**Implement TurboQuant_mse with Hadamard rotation and mixed-precision channel splitting for KV cache compression, enabling full 262K context on M5 Max.**

Configurable bit-width via `--kv-bits <2|3|4>`, default 3-bit for quality. For aggressive compression (262K context), 2.5-bit effective rate via fixed channel splitting (see design choice #3).

### Key design choices (each justified by research findings):

1. **TurboQuant_mse, not TurboQuant_prod** — the QJL residual correction stage (for unbiased inner products) is unnecessary at ≥2.5 bits. Agent: qjl-necessity established that multiplicative bias β > 0.97 at 2.5 bits, producing negligible softmax distortion (KL ≈ 0.001). The paper's own KV cache experiments used TurboQuant_mse. Skipping QJL means all bits go to MSE precision — simpler and better.

2. **Hadamard rotation, not dense random matrix** — **This is an engineering deviation from the paper.** The paper (Algorithm 1) uses a uniformly random orthogonal matrix Π generated via QR decomposition of a Gaussian matrix. We substitute the deterministic Walsh-Hadamard Transform for O(d log d) performance vs O(d²). Agent: cost-analyst showed dense rotation is acceptable at encode time (~290 μs) but **catastrophic at SDPA decode time** (7.6 ms for Gemma, 72% of budget). Hadamard reduces rotation cost to ~20 μs total. Prior art (QuaRot, QuIP#) validates Hadamard for outlier spreading in LLM quantization. **Phase 0.1 validates this substitution empirically:** if Hadamard MSE exceeds random rotation MSE by >20%, we fall back to randomized block-diagonal rotation (QuIP#-style).

3. **2.5-bit effective via fixed channel splitting** — The paper achieves non-integer bit-widths through outlier channel splitting: a subset of channels is quantized at a higher bit-width than the rest. Specifically, for 2.5-bit effective at head_dim=d: the first d/4 channels use 3-bit quantization (8 Lloyd-Max centroids), the remaining 3d/4 channels use 2-bit (4 centroids). Effective rate: (d/4 × 3 + 3d/4 × 2) / d = 2.5 bits/coordinate. **The split is a compile-time constant, not per-position.** Rationale: after Hadamard rotation, all coordinates are approximately N(0, 1/d) with equal magnitude (confirmed by QuaRot: rotation eliminates outlier channels entirely, producing kurtosis ≈ 3). Per-position outlier detection would add ~167 MB storage at 262K with negligible quality benefit. Phase 0.1 validates this assumption by measuring per-coordinate magnitude variance after rotation.

   At 262K context, 2.5-bit TQ achieves 49.44 vs 50.06 on LongBench (−1.2%) while saving 40% more memory vs 3.5-bit. The paper's needle-in-haystack recall at 2.5-bit is 0.997 (matching F16 baseline).

4. **Pre-rotated centroid tables** — Agent: metal-feasibility showed that naive per-position dequantization during SDPA requires O(d²) rotation per cached token (infeasible at long contexts). Pre-rotating centroid vectors at init time eliminates rotation from the SDPA read path entirely: dequant = table lookup only. Each bit-width has its own centroid table; for 2.5-bit mode, both the 2-bit and 3-bit tables are pre-rotated.

5. **Mixed nibble packing for storage** — For uniform bit-widths (2, 3, or 4-bit), indices are nibble-packed (4-bit aligned) for Metal-friendly access. For 2.5-bit mode, the two channel groups use their respective widths: 3-bit channels packed in nibbles (wasting 1 bit), 2-bit channels packed in nibbles (wasting 2 bits). Effective storage is ~4 bits/coordinate despite 2.5-bit quality. If memory pressure demands true mixed-width packing, Phase 3 introduces it.

   **Revision note:** This is a simplifying assumption for Phase 0-1. If nibble packing at 262K context exceeds memory targets, Phase 2 triggers true mixed-width packing. The acceptance criteria below specify the memory gate.

---

## Architecture

### Current KV Cache Pipeline (F16)

```
Token → QKV projection (F32) → kv_cache_copy_batch_f32_to_f16 → [F16 cache]
                                                                      ↓
                                                          flash_attn_vec reads F16
```

2 dispatches for cache write (K + V). 2 dispatches for SDPA (main + reduce). Total: 4 dispatches per layer.

### TurboQuant KV Cache Pipeline

```
Token → QKV projection (F32) → hadamard_quantize_kv → [packed cache + norms]
                                                              ↓
                                                 flash_attn_vec_tq reads packed
                                                 (centroid gather from pre-rotated tables)
```

2 dispatches for cache write (K + V). 2 dispatches for SDPA (main + reduce). Total: **4 dispatches per layer** (unchanged).

### New Metal Kernels

#### 1. `hadamard_quantize_kv` (replaces `kv_cache_copy_batch_f32_to_f16`)

**Input:** F32 KV vector `[num_kv_heads, head_dim]`  
**Output:** Packed indices `[num_kv_heads, head_dim]` as nibbles + F32 norm scalar per head  
**Operation per head:**
1. Fast Walsh-Hadamard Transform on the head vector (in-place, shared memory)
2. Scale by `1/√d` (we use the normalized convention: H_normalized = H_unnormalized / √d, so H·H = I)
3. Compute and store `‖x_rotated‖₂` as F32
4. Normalize: `x̂ = x_rotated / ‖x_rotated‖₂`
5. For each coordinate: find nearest Lloyd-Max centroid. In 2.5-bit mode: coordinates 0..d/4-1 use the 3-bit codebook (8 centroids), coordinates d/4..d-1 use the 2-bit codebook (4 centroids). In uniform mode (2, 3, or 4-bit): all coordinates use the same codebook.
6. Pack centroid indices into nibble (4-bit) output buffer

**Threadgroup design:** One threadgroup per head. d threads (256 for sliding, 512 for global). Hadamard butterfly uses `log₂(d)` stages with shared memory barriers. Quantize step is embarrassingly parallel across coordinates. In 2.5-bit mode, the first d/4 threads and remaining 3d/4 threads follow uniform code paths (no per-thread branching).

**Dispatch count:** 1 per K, 1 per V = **2 per layer** (same as current).

#### 2. `flash_attn_vec_tq` (modified `flash_attn_vec`)

**Change from current:** Replace F16 cache loads with:
1. Read packed nibble indices for cached position
2. Read F32 norm scalar for that position
3. Gather centroid values from pre-rotated lookup table
4. Use gathered F32 vector in dot product / value accumulation

**Pre-rotated centroid table:** `[2^b, head_dim]` F32 per layer. For b=4 (nibble), d=256: 16 × 256 × 4 = 16 KB per layer. 30 layers total: **480 KB**. Negligible.

**Critical assumption to validate in Phase 0:** Gather-based cache reads (indexed by nibble values) vs sequential F16 reads. Apple Silicon's texture/buffer gather throughput may differ from sequential bandwidth. This is the #1 risk item.

#### 3. Lloyd-Max Codebook (compile-time constant)

Precomputed centroids for the Beta distribution `f_X(x) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1−x²)^((d−3)/2)` at practical dimensions and bit-widths. In high dimensions (d ≥ 128) this converges to `N(0, 1/d)`, so we use a single Gaussian-optimal codebook. **This is an approximation** — the exact Beta distribution depends on d. Phase 0.1 validates that Gaussian-optimal codebooks achieve MSE within 5% of dimension-specific Beta-optimal codebooks at d=128, 256, and 512.

| Bit-width | Centroids | Storage |
|-----------|-----------|---------|
| 2 | 4 scalars | 16 bytes |
| 3 | 8 scalars | 32 bytes |
| 4 | 16 scalars | 64 bytes |

Baked into the binary as `const` arrays. No runtime computation.

### Data Layout

**Packed KV cache buffer per layer per K or V:**
```
[num_kv_heads × capacity × ⌈head_dim/2⌉] u8     // nibble-packed indices
[num_kv_heads × capacity] f32                     // per-position norm scalars
```

**Pre-rotated centroid table per layer:**
```
[2^b × head_dim] f32                              // Hadamard-rotated centroids
```

Precomputed at model load: apply inverse Hadamard to each centroid vector. We use the normalized convention (H·H = I), so H⁻¹ = H — the inverse is the same transform. This is one FWHT per centroid — at most 16 transforms of length 512, trivial. For 2.5-bit mode, both 2-bit (4 centroids) and 3-bit (8 centroids) tables are pre-rotated independently.

### Memory Budget at 262K Context

**Gemma-4-27B, 2.5-bit effective (stored as 4-bit nibble):**

Sliding layers: 25 layers, 8 KV heads, head_dim=256, capacity=1024  
Global layers: 5 layers, 2 KV heads, head_dim=512, capacity=262144

| Component | Calculation | Size |
|-----------|------------|------|
| Sliding packed (25 layers) | 25 × 2(K+V) × 8 heads × 1024 pos × ⌈256/2⌉ u8 | 50.0 MB |
| Sliding norms | 25 × 2 × 8 × 1024 × 4 bytes | 1.6 MB |
| Global packed (5 layers) | 5 × 2 × 2 × 262144 × ⌈512/2⌉ u8 | 1,280.0 MB |
| Global norms | 5 × 2 × 2 × 262144 × 4 bytes | 40.0 MB |
| Centroid tables | 30 × 16 × 512 × 4 (worst case, global head_dim) | 0.94 MB |
| **Total KV** | | **1,372.5 MB** |

vs F16 at 262K: **5,320 MB**. Compression: **3.88×**.

**With true mixed-width packing** (Phase 3 optimization if needed):

For 2.5-bit mode: d/4 channels at 3-bit + 3d/4 channels at 2-bit = 2.5 bits/coord average.  
Packed storage per position: ⌈d × 2.5 / 8⌉ bytes.

| Component | Calculation | Size |
|-----------|------------|------|
| Sliding packed (25 layers) | 25 × 2 × 8 × 1024 × ⌈256×2.5/8⌉ bytes + norms | 33.2 MB |
| Global packed (5 layers) | 5 × 2 × 2 × 262144 × ⌈512×2.5/8⌉ bytes + norms | **860.0 MB** |
| **Total KV** | | **~893 MB** |

Compression: **5.96×**.

### Bandwidth at 262K Context (M5 Max, 546 GB/s)

| Representation | KV read/token | Time @ 546 GB/s | Decode estimate |
|---|---|---|---|
| F16 | 5,320 MB | 9.7 ms | ~60 tok/s |
| TQ nibble (4-bit stored) | 1,373 MB | 2.51 ms | ~105 tok/s |
| TQ 2.5-bit packed | 893 MB | 1.64 ms | ~115 tok/s |

---

## Implementation Plan

### Phase 0 — Validate & Measure (Chesterton's fence + measure 3x)

**Goal:** Prove TurboQuant preserves coherence and measure gather throughput on M5 Max.

**0.1 — CPU reference implementation**
- Implement Hadamard + Lloyd-Max quantize/dequantize in pure Rust (no GPU)
- Validate: quantize a KV vector, dequantize, measure MSE vs paper's bounds
- Validate: run a full forward pass with CPU-side quantize/dequantize intercepting the KV cache write/read, compare logits to F16 baseline
- **Hadamard vs random rotation comparison:** Quantize 1000 KV vectors using both Hadamard rotation and dense random orthogonal rotation (QR of Gaussian). Compare MSE. **Gate:** Hadamard MSE ≤ 1.2× random rotation MSE at each head_dim (128, 256, 512). If exceeded, fall back to randomized block-diagonal rotation.
- **Gaussian codebook validation:** Compare MSE using Gaussian-optimal codebooks vs dimension-specific Beta-optimal codebooks at d=128, 256, 512. **Gate:** Gaussian codebook MSE ≤ 1.05× Beta-optimal codebook MSE.
- **Fixed channel split validation:** After Hadamard rotation of 1000 KV vectors, measure per-coordinate magnitude variance. Identify top-d/4 highest-magnitude coordinates per vector; measure overlap ratio across vectors. **Gate:** If overlap < 50% (outlier channels are position-dependent), revisit fixed-split assumption. Expected: overlap is low but magnitudes are near-uniform, confirming fixed split is sufficient.
- **Gate:** Top-1 token agreement on 100-token greedy decode at 8K context. If top-1 disagrees on >2 tokens, investigate before proceeding.

**0.2 — Gather throughput microbench**
- Write a standalone Metal kernel that reads from a nibble-packed buffer using index-based gather (simulating SDPA cache reads)
- Measure throughput vs sequential F16 reads at capacity=8192 and capacity=262144
- **Gate:** Gather throughput ≥ 50% of sequential F16 throughput. If below 50%, the bandwidth savings from smaller representation are negated by gather penalty — revisit the architecture (consider dequant-to-temp-buffer instead).

**0.3 — Hadamard transform microbench**
- Write a standalone Metal kernel for in-place FWHT at d=128, 256, 512
- Measure latency per head and total per-token overhead across all layers
- **Gate:** Total Hadamard overhead ≤ 200 μs/token (< 2% of decode budget).

### Phase 1 — Metal Kernels

**Goal:** Replace F16 KV cache with TurboQuant on GPU.

**1.1 — `hadamard_quantize_kv` kernel**
- Metal compute kernel: FWHT + normalize + quantize + nibble-pack
- One threadgroup per head, d threads per threadgroup
- Shared memory for butterfly stages
- **Correctness gate:** Output indices match CPU reference from Phase 0.1 exactly (bitwise on indices, ε < 1e-6 on norms)

**1.2 — Pre-rotated centroid table computation**
- At model load, apply inverse Hadamard to each of the 2^b centroid vectors
- Store as `[2^b, head_dim]` F32 Metal buffer per layer
- **Correctness gate:** `H^(-1) · centroid` round-trips to original centroid within ε < 1e-7

**1.3 — `flash_attn_vec_tq` kernel**
- Fork `flash_attn_vec`, replace F16 cache loads with:
  - Nibble extraction from packed buffer
  - Gather from pre-rotated centroid table
  - F32 dot product and value accumulation
- Keep the existing workgroup partitioning and reduce pattern
- **Correctness gate:** SDPA output matches CPU reference from Phase 0.1 within ε < 1e-4
- **Sourdough gate:** Full 100-token greedy decode matches F16 path output through ≥95 tokens

**1.4 — Integration into `forward_mlx.rs`**
- Replace `MlxKvCache` fields: `k: MlxBuffer` (F16) → `k_packed: MlxBuffer` (u8 nibble) + `k_norms: MlxBuffer` (F32)
- Replace `dispatch_kv_cache_copy_batch_f32_to_f16` calls with `hadamard_quantize_kv` dispatch
- Replace `flash_attn_vec` calls with `flash_attn_vec_tq` dispatch
- Add centroid table buffers to `MlxForwardState`
- For 2.5-bit mode: add both 2-bit and 3-bit pre-rotated centroid tables per layer
- **Dispatch count gate:** Total dispatches per forward pass must not increase vs pre-TurboQuant baseline at the same model configuration and context length. (Do not hardcode a specific number — it varies by model config.)
- **Sourdough gate:** Byte-identical 16-token greedy gen at T=0 vs llama.cpp at 8K context. (At longer contexts where quantization error accumulates, this gate relaxes to top-1 agreement ≥ 95/100 tokens.)

### Phase 2 — 262K Context Unlock

**Goal:** Remove the 8192 cap and validate at full context length.

**2.1 — Remove the cap**
- Delete `let max_global_kv = 8192;` (grep to locate; line number may have shifted)
- Set `capacity = cfg.max_position_embeddings` for global layers
- Validate memory allocation succeeds on M5 Max 192 GB

**2.2 — Memory budget validation**
- Measure actual RSS with 262K context KV cache allocated
- **Gate:** Total KV cache allocation ≤ 1,400 MB (nibble packing) or ≤ 900 MB (true mixed-width packing if implemented)
- If nibble packing exceeds 1,400 MB, implement true mixed-width packing before proceeding

**2.3 — Long-context correctness**
- Needle-in-haystack test: insert a unique fact at various positions in a 100K+ token document, verify retrieval
- **Gate:** Retrieval accuracy ≥ 95% at all insertion positions (paper reports 99.7%)

**2.4 — Long-context performance**
- 5-run median benchmark at 8K, 32K, 131K, 262K context lengths
- Measure decode tok/s at each length
- **Gate:** Decode speed at 262K ≥ 80 tok/s on M5 Max (conservative; estimate is ~105 tok/s)

### Phase 3 — Optimization (only if Phase 2 gates are met)

**3.1 — True mixed-width packing** (if not already done in Phase 2)
- For 2.5-bit mode: pack 3-bit channels densely (8 indices → 24 bits = 3 bytes) and 2-bit channels densely (4 indices → 8 bits = 1 byte)
- Custom extraction logic in SDPA kernel keyed on the compile-time channel split boundary
- **Gate:** Memory reduction ≥ 35% vs nibble packing

**3.2 — Configurable bit-width** (already wired in Phase 1 via `--kv-bits`)
- Supported modes: `--kv-bits <2|2.5|3|4>`, where 2.5 activates fixed channel splitting
- Default: 3 (best quality/memory tradeoff)
- Lloyd-Max codebooks for all integer widths (2, 3, 4) already baked in; 2.5-bit uses both 2-bit and 3-bit codebooks

**3.3 — Per-layer adaptive bit-width**
- Sliding layers (small cache, 1024 positions): use 4-bit TurboQuant (minimal memory impact, highest quality)
- Global layers (large cache, 262K positions): use 2-3 bit (maximum savings where it matters)
- No F16 KV path remains for any layer type after Phase 1
- **Gate:** No quality regression vs uniform bit-width on needle-in-haystack

---

## Acceptance Criteria (End Gates)

### Functional

- [ ] **F-1:** Full 262,144-token context supported without OOM on M5 Max 192 GB
- [ ] **F-2:** All KV cache layers use TurboQuant_mse with Hadamard rotation — no F16 KV path remains for any layer type (sliding or global)
- [ ] **F-3:** Sourdough gate passes at 8K context: byte-identical 16-token greedy gen at T=0 vs llama.cpp. At contexts >8K, relaxed gate: top-1 token agreement ≥ 95/100 on greedy decode
- [ ] **F-4:** Needle-in-haystack retrieval accuracy ≥ 95% at 100K+ tokens
- [ ] **F-5:** Lloyd-Max codebooks for 2, 3, and 4-bit widths baked into binary as compile-time constants. 2.5-bit mode uses both 2-bit and 3-bit codebooks with fixed channel split

### Performance

- [ ] **P-1:** KV cache memory at 262K context ≤ 1,400 MB (nibble) or ≤ 900 MB (true mixed-width packing)
- [ ] **P-2:** Decode speed at 262K context ≥ 80 tok/s on M5 Max (5-run median)
- [ ] **P-3:** Decode speed at 8K context: no regression vs current F16 path (≥ 94 tok/s, 5-run median)
- [ ] **P-4:** Hadamard + quantize overhead ≤ 200 μs/token total across all layers
- [ ] **P-5:** Dispatch count per forward pass: unchanged from pre-TurboQuant baseline at same model config and context length
- [ ] **P-6:** Gather throughput in SDPA ≥ 50% of sequential F16 throughput (Phase 0 microbench)

### Quality

- [ ] **Q-1:** Top-1 token agreement with F16 baseline ≥ 98/100 on greedy decode at 8K context
- [ ] **Q-2:** LongBench average score ≥ 49.0 (paper: 49.44 at 2.5-bit, baseline: 50.06)
- [ ] **Q-3:** No catastrophic attention flattening at any layer (validate via attention entropy monitoring in Phase 1)

### Engineering

- [ ] **E-1:** Zero new Metal dispatches per forward pass vs pre-TurboQuant baseline (modify existing kernels, don't add new ones)
- [ ] **E-2:** No C/C++ dependencies added (pure Rust + Metal shading language, consistent with project constraints)
- [ ] **E-3:** Centroid tables and Hadamard computation are deterministic (no runtime randomness — the Hadamard matrix is fixed, not random)
- [ ] **E-4:** All new Metal kernels have standalone microbench tests in `mlx-native`
- [ ] **E-5:** Each phase commits + pushes on completion (per `feedback_commit_push_cadence.md`)

---

## Risks & Mitigations

| # | Risk | Severity | Likelihood | Mitigation |
|---|------|----------|------------|------------|
| R-1 | Gather throughput on M5 Max is poor (< 50% of sequential) | High | Medium | Phase 0.2 microbench kills this early. Fallback: dequant-to-temp-buffer (adds 2 dispatches per layer but preserves bandwidth savings). Sub-vector VQ is an independent optimization, not a fallback for this risk. |
| R-2 | Hadamard rotation produces higher MSE than random orthogonal rotation | High | Low | Phase 0.1 measures this directly: Hadamard MSE vs random rotation MSE at all head_dims. Gate: ≤1.2× ratio. Hadamard is orthogonal (energy-preserving); QuaRot/QuIP# validate it empirically for LLM quantization. |
| R-3 | Modified SDPA kernel is slower than F16 SDPA despite reading less data | Medium | Medium | Phase 1.3 measures wall-clock, not just throughput. If slower, profile: is it ALU (centroid gather compute) or memory (random access pattern)? |
| R-4 | Nibble packing wastes memory at 262K (1.37 GB vs 0.89 GB at true mixed-width) | Medium | Low | Phase 2.2 memory gate triggers true mixed-width packing. On 192 GB M5 Max, 1.37 GB is <1% of memory — likely acceptable |
| R-5 | Quality regression on specific model architectures (Qwen-3, etc.) | Medium | Low | Phase 2 validates on Gemma-4 (primary). Qwen-3 support is a future extension validated separately |
| R-6 | Fixed channel split is suboptimal vs per-position outlier selection | Low | Low | Phase 0.1 measures per-coordinate magnitude variance after rotation. QuaRot shows rotation eliminates outliers (kurtosis ≈ 3), making channel selection arbitrary. If variance is unexpectedly high, revisit. |
| R-7 | Gaussian-optimal codebook is suboptimal for small head_dim (d=128) | Low | Low | Phase 0.1 compares Gaussian vs Beta-optimal codebooks at d=128, 256, 512. Gate: ≤5% MSE gap. |

---

## Information-Theoretic Foundation

From TurboQuant (Theorem 1, Theorem 3):

**MSE Distortion Bounds:**

| Bit-width | Upper bound (TurboQuant) | Lower bound (any algorithm) | Gap |
|-----------|------------------------|-----------------------------|-----|
| 2 | 0.117 | 0.0625 | 1.87× |
| 3 | 0.030 | 0.0156 | 1.92× |
| 4 | 0.009 | 0.0039 | 2.31× |

TurboQuant is within 2.7× of the information-theoretic optimum at all bit-widths. No algorithm — learned, calibrated, or otherwise — can beat the lower bound. This means TurboQuant's quality at 2.5 bits is close to the best any quantization scheme could achieve at 2.5 bits.

**Why this matters for the ADR:** We are not betting on an empirical hack that might fail on other models. TurboQuant's guarantees are mathematical, distribution-free, and worst-case. If it works on Gemma-4-27B at 2.5 bits (and the paper shows it does on Llama-3.1-8B), it will work on any model with similar or larger head dimensions.

---

## Research Provenance

This ADR is grounded in a 5-agent research swarm (2026-04-14) analyzing arXiv:2504.19874:

| Agent | Finding | Impact on ADR |
|-------|---------|---------------|
| cost-analyst | Rotation is per-head (d=256), not per-hidden-state (d=3584). Dense rotation: ~290 μs. Hadamard: ~20 μs. SDPA-side dense rotation: 7.6 ms (infeasible). | → Hadamard mandatory. Encode-side feasible. |
| quant-comparator | INT4 ties at 4 bits. TurboQuant wins at ≤3 bits. Outlier spreading is genuine advantage. | → TurboQuant justified at 2.5-bit target. |
| metal-feasibility | Option C (pre-rotated centroids) eliminates rotation from SDPA path. Dispatch count unchanged. Gather throughput is the key risk. | → Architecture design. Phase 0.2 risk gate. |
| qjl-necessity | QJL unnecessary at ≥2.5 bits (bias β > 0.97, KL ≈ 0.001). Paper's own KV cache results used MSE-only. | → TurboQuant_mse, no QJL. Simpler implementation. |
| mem-calculator | 262K F16 = 5.3 GB. TQ nibble = 1.4 GB. TQ 2.5b = 0.88 GB. Bandwidth savings: 9.7 ms → 2.6 ms at 262K. | → Memory and performance gates. |

---

## Deviations from the Paper

This ADR implements the core TurboQuant_mse algorithm faithfully but makes two engineering substitutions, both validated empirically in Phase 0:

| Paper specifies | ADR uses | Reason | Validation |
|-----------------|----------|--------|------------|
| Dense random orthogonal matrix (QR of Gaussian) | Walsh-Hadamard Transform | O(d log d) vs O(d²); critical for SDPA decode latency. Prior art (QuaRot, QuIP#) validates Hadamard for outlier spreading. | Phase 0.1: MSE comparison, gate ≤1.2× |
| Per-instance outlier channel selection for non-integer bit-widths | Fixed compile-time channel split (first d/4 at higher bit-width) | After rotation, all coordinates ≈ N(0,1/d) with equal magnitude — outlier identity is arbitrary. Eliminates 167 MB per-position storage at 262K. | Phase 0.1: magnitude variance measurement |

Everything else — Lloyd-Max codebooks, per-vector norm storage, TurboQuant_mse (not _prod), coordinate-wise scalar quantization — follows the paper directly.

---

## What This ADR Does NOT Cover

- **Prompt/prefill-phase quantization** — TurboQuant applies to decode-phase KV cache. Prefill processes tokens in bulk and may benefit from different optimizations.
- **Weight quantization** — model weights remain Q4_K_M via GGUF. This ADR is exclusively about KV cache.
- **Multi-model support** — Phase 0-2 target Gemma-4-27B only. Qwen-3 and other architectures are future work after the Gemma path is validated.
- **Streaming/incremental quantization during prefill** — the paper notes TurboQuant is online/data-oblivious and can quantize during streaming generation. This is a natural extension but not in scope for the initial implementation.
- **TurboQuant_prod / QJL residual correction** — not needed at ≥2.5 bits per the paper's own KV cache experiments.
