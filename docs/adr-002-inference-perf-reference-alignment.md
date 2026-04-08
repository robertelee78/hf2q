# ADR-002: Inference Performance and Reference Alignment

**Status:** Accepted
**Date:** 2026-04-08
**Authors:** Robert
**Deciders:** Robert

## Context

hf2q is a pure-Rust Gemma 4 inference engine using custom Metal GPU kernels via the `mlx-native` crate (zero C++ dependencies). A comprehensive audit against the mlx-lm Python reference implementation revealed significant divergences in both correctness and performance.

### The Audit

Five parallel research agents compared our code against mlx-lm (`/Users/robert/.pyenv/versions/3.13.12/lib/python3.13/site-packages/mlx_lm/`) and Ollama (`/opt/ollama/`) across sampler, quantized matmul kernels, attention/KV cache, MoE routing, and generation pipeline.

**Key discovery:** Ollama does NOT support Gemma4 — its `gemma3.go` is Gemma3 only. **mlx-lm is our sole authoritative reference.**

### Performance Baseline

Before optimization: ~4.7 tok/s prefill, ~2.8 tok/s decode on Apple Silicon (M4 Max, 128GB).

Root cause analysis revealed kernel launch overhead (~5ms) exceeded actual GPU compute time (~3-4ms) per decode step, driven by ~305 `commit_and_wait()` calls and ~2,278 kernel dispatches per forward pass.

## Decision

### D1: Follow mlx-lm as the single source of truth

All inference behavior — sampler defaults, model architecture, weight handling, generation loop — must match mlx-lm. When our implementation diverges, mlx-lm wins unless we have a documented, measured reason to differ.

**Rationale:** mlx-lm is maintained by Apple's ML team, is the canonical MLX inference implementation, and is what users compare against. Ollama wraps MLX's C library for its Apple Silicon path but doesn't implement Gemma4.

### D2: Reduce GPU sync points via mega-encoders

Restructure `forward_layer()` from ~10 `commit_and_wait()` calls per layer to 4, by merging adjacent GPU dispatches into "mega-encoders" that batch multiple kernel dispatches into a single Metal command buffer.

**Mega-encoder A:** norm + QKV proj + head norms + RoPE + KV append + SDPA + O proj + post-attn norm + residual
**Mega-encoder B:** Pre-FFN norms (forced sync — MoE CPU routing reads the result)
**Mega-encoder C:** Dense FFN + MoE expert dispatch + MoE accumulation
**Mega-encoder D:** Post-FFN norms + combine + residual + scalar

**Rationale:** Metal guarantees ordered execution within a command buffer. Adjacent GPU-only dispatches with no CPU readback between them can safely share an encoder. This reduces ~305 syncs to ~120 (4 per layer × 30 layers + overhead).

### D3: GPU-native MoE routing

Replace CPU-side MoE routing (which reads 22KB GPU→CPU per layer, does 360K FLOPs on CPU) with a parallel GPU kernel that outputs only 64 bytes (8 expert IDs + 8 weights).

**Rationale:** The CPU routing forced a sync point before every MoE layer, preventing encoder merging across the FFN boundary. GPU routing eliminates this bottleneck and 660KB of GPU→CPU transfers per decode step.

### D4: Fix sampler to match mlx-lm pipeline order

Current order: `logits → rep_penalty → temperature → top_k → top_p → softmax → sample`
Correct order: `logits → rep_penalty → logsoftmax → top_p → min_p → top_k → categorical(logprobs / temp)`

Key changes:
- Temperature moves to the END (applied at sampling time, not before filtering)
- Pipeline operates in log-probability domain after penalty step
- Softmax computed exactly once
- Add min_p sampling (both references support it)
- Add repetition_context_size with default 20 (both references window the penalty)

**Rationale:** Temperature before top-k/top-p changes which tokens survive filtering, producing different outputs than mlx-lm for the same parameters. The current default `repetition_penalty: 1.0` (disabled) with no context window causes unbounded repetition in long generations.

### D5: Implement KV sharing / donor layers for 27B

The last `num_kv_shared_layers` layers (20 for 27B) should reuse K/V from earlier "donor" layers of the same type (sliding/global). Shared layers still compute their own Q projection but skip K/V projection entirely.

**Rationale:** Without this, 20 layers compute redundant K/V projections producing wrong attention outputs and wasting 20× the K/V compute + cache memory. This is a correctness issue, not just performance.

### D6: Prefill last-token-only lm_head + chunked prefill

During prefill, skip the lm_head projection for all but the last hidden state position. Add chunked prefill at 2048 tokens (matching mlx-lm and Ollama).

**Rationale:** For a 2048-token prompt, we currently compute 2048 × 262144 = 537M floats of logits and discard 99.95%. This wastes ~2GB of GPU memory. Chunked prefill prevents OOM for long prompts.

### D7: Build bf16 I/O quantized matmul kernel (matching MLX qmv_fast)

Create a bf16-input/bf16-output variant of the SIMD quantized matmul that eliminates 2 cast dispatches per projection. All kernels must use consistent MLX qmv_fast parameters: `packs_per_thread=2, values_per_thread=16` for 4-bit.

**Rationale:** The current cast chain (bf16→f32→qmatmul→f32→bf16) adds 420 unnecessary kernel dispatches per forward pass. The initial bf16 kernel had a bug (`values_per_thread=8` instead of 16 for 4-bit) plus the expert variant had inconsistent parameters. Additionally, the GPU moe_gate kernel fuses per_expert_scale into the softmax denominator, cancelling the scale's effect — must apply scale AFTER softmax.

### D8: GPU-side sampling for greedy decode

For greedy decoding (temperature=0), run argmax on GPU and return only the token ID (4 bytes vs 1MB logits readback). For stochastic sampling, run temperature + softmax + categorical on GPU.

**Rationale:** The 1MB logits readback per decode step is a significant bottleneck. Greedy decode is the most common case and benefits most from this optimization.

## Consequences

### Positive
- Correct output matching mlx-lm (the user's reference)
- ~60% reduction in GPU sync points (305 → 120)
- GPU MoE routing eliminates 660KB/step of CPU transfers
- Sampler matches mlx-lm behavior for all parameter combinations
- KV sharing saves 20 K/V projections per forward pass on 27B
- Prefill optimization saves ~2GB memory for typical prompts

### Negative
- gemma4.rs grows to ~5K lines (refactor into ~9 modules is queued)
- bf16 qmatmul kernel requires careful debugging (values_per_thread and per_expert_scale bugs found)
- KV sharing + double-wide MLP adds model config complexity
- Some optimizations gated behind env vars until validated (HF2Q_BF16_QMATMUL, HF2Q_GPU_ARGMAX, HF2Q_FUSED_NORM_ADD)

### Risks
- Merged encoders could introduce Metal command buffer ordering bugs (mitigated by bit-exact logit comparison testing)
- bf16 qmatmul numerical precision may differ from cast+f32+cast chain (mitigated by gating behind env var)
- Async pipeline could introduce race conditions (mitigated by Metal MTLEvent synchronization)

## Implementation Status

### Completed

| Phase | What | Commits |
|-------|------|---------|
| Phase 0 | Instrumentation: sync/dispatch counters, TTFT, API timing | mlx-native `19a05fd`, hf2q `f412d0c` |
| Phase 1 | Sync elimination: 305→120 syncs via mega-encoders | hf2q `f412d0c` |
| Phase 2a | GPU moe_gate kernel: parallel, bf16, fused norm+matmul+topK | mlx-native `19a05fd` |
| Phase 2b | bf16 qmatmul + expert offset kernels | mlx-native `19a05fd` |
| Phase 2 integration | GPU MoE routing wired into gemma4.rs | hf2q `368d8a1` |
| Phase 3a | Double-buffer decode loop prep | hf2q `353ea56` |
| Phase 4 kernels | argmax_f32 + softmax_sample_f32 GPU kernels | mlx-native `8f1a2da` |
| Phase 5 kernels | fused_head_norm_rope, fused_residual_norm, fused_norm_add | mlx-native `19a05fd`, `145c246` |
| Phase 6d | Sampler O(V) partial sort, deduplicated softmax | hf2q `f412d0c` |
| Phase 6a | Embedding CPU round-trip eliminated | hf2q `368d8a1` |
| Phase 6f | MoE buffer pooling (480 fewer allocs/forward) | hf2q `a7ad3ba` |
| Bugfix | bf16 qmatmul SIMD fallback cast, garbled output fix | mlx-native `4a49073`, `f550a59`, hf2q `9e5dc39` |

### Remaining (Priority Order)

| Phase | What | Blocker |
|-------|------|---------|
| **A1** | Fix GPU moe_gate per_expert_scale (apply after softmax, not fused) | None — shader fix |
| **A2** | Fix all qmatmul kernels to use consistent qmv_fast params | None — shader fix |
| **A3** | Rewrite sampler to mlx-lm pipeline order | None — Rust code |
| **B1** | Implement KV sharing / donor layers | Config parsing + forward_layer changes |
| **B2** | Implement double-wide MLP for shared layers | Depends on B1 |
| **C1** | Prefill last-token-only lm_head | forward() change |
| **C2** | Chunked prefill at 2048 tokens | engine.rs change |
| **C3** | Async forward_start/forward_wait | gemma4.rs + engine.rs |
| **D1** | Refactor gemma4.rs into ~9 modules | Post-stabilization |

## Divergence Registry

Complete list of all divergences found between hf2q and mlx-lm, with severity and status.

### Correctness (must match mlx-lm exactly)

| # | Area | Divergence | Severity | Status |
|---|------|-----------|----------|--------|
| 1 | MoE | GPU moe_gate per_expert_scale fused into softmax denominator | CRITICAL | Open |
| 2 | QMatmul | bf16 expert kernel: wrong values_per_thread/packs_per_thread | CRITICAL | Open |
| 3 | Pipeline | KV sharing / donor layers missing (27B: 20 layers affected) | CRITICAL | Open |
| 4 | Pipeline | PLE (Per-Layer Embedding) missing (2B/4B models) | CRITICAL | Open (N/A for 12B+) |
| 5 | Sampler | Temperature applied before top-k/top-p (should be last) | HIGH | Open |
| 6 | Sampler | No repetition context window (should default to 20) | HIGH | Open |
| 7 | Sampler | Missing min_p sampling | HIGH | Open |
| 8 | Pipeline | RoPE freq computation may use wrong denominator for global layers | HIGH | Needs verification |
| 9 | Pipeline | Double-wide MLP missing for KV-shared layers (27B) | MEDIUM | Open |

### Performance (diverges from reference approach)

| # | Area | Divergence | Impact | Status |
|---|------|-----------|--------|--------|
| 10 | Pipeline | Prefill computes all logits, not just last | ~2GB waste | Open |
| 11 | Pipeline | No chunked prefill (OOM risk for long prompts) | Memory | Open |
| 12 | Attention | No flash attention (O(N^2) memory) | Slow for long ctx | Open |
| 13 | KV Cache | Extra transpose before SDPA (layout mismatch) | 1 dispatch/layer | Open |
| 14 | Pipeline | Synchronous decode loop (no CPU/GPU overlap) | ~30-50% slower | Phase 3a done |
| 15 | Sampler | Double softmax when top_p active | Minor CPU waste | Partially fixed |
| 16 | QMatmul | bf16 qmatmul kernel produces wrong output | Cast chain required | Open |

### Correct (no action needed)

| Area | What | Status |
|------|------|--------|
| Attention | Scale = 1.0 | Matches |
| Attention | K=V optimization | Matches |
| Attention | V-norm (unweighted RMSNorm) | Matches |
| Attention | GQA head expansion | Matches |
| MoE | Router normalization (fused scale * root_size) | Matches |
| MoE | Dense + MoE parallel composition | Matches |
| MoE | Expert FFN activation (GELU tanh) | Matches |
| MoE | CPU routing path weights (correct per_expert_scale) | Matches |
| Pipeline | Embedding scale (sqrt(hidden_size)) | Matches |
| Pipeline | Final logit softcap (30.0) | Matches |
| Pipeline | Layer scalar placement | Matches |

## References

- mlx-lm source: `/Users/robert/.pyenv/versions/3.13.12/lib/python3.13/site-packages/mlx_lm/`
- mlx-lm Gemma4 model: `mlx_lm/models/gemma4_text.py`
- mlx-lm generation: `mlx_lm/generate.py`
- Ollama source: `/opt/ollama/` (Gemma3 only — NOT a valid Gemma4 reference)
- PRD: `/opt/hf2q/docs/prd_inference_perf.md`
- MLX quantized matmul reference: MLX C++ `quantized.h` (`qmv_fast` kernel)
