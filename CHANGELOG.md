# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-16

First public release.

### Added — convert pipeline

- HuggingFace → GGUF / mlx-lm safetensors converter (`hf2q convert`).
- Quantization families:
  - Float passthrough — `f16`, `bf16`, `auto`.
  - Legacy block — `Q2`, `Q4` (alias `Q4_0`), `Q8` (alias `Q8_0`).
  - K-quants — `Q2_K{,_S}`, `Q3_K_{S,M,L}`, `Q4_K_{S,M}`, `Q5_K_{S,M}`, `Q6_K`.
  - Imatrix-weighted K-quants — `imatrix-*` variants plus `imatrix-adaptive`.
  - Mixed-bit — `dynamic-quant-{4-6, 4-8, 6-8, 2-8}`.
- DWQ training (`hf2q dwq-train`) producing an mlx-format safetensors
  overlay layered on top of a GGUF (ADR-020).
- Two-pass intermediate GGUF for activation capture during
  Qwen 3.5 / 3.6 conversion (ADR-012).
- Streaming convert pipeline with disk-floor preflight (ADR-014).

### Added — inference + serving

- Apple-Silicon-only inference path on top of `mlx-native` (ADR-008).
- GGUF reader, KV cache, prefill, and decode for:
  - Gemma 4 (dense + MoE).
  - Qwen 3.5 / 3.6 (dense + MoE + multi-token-prediction).
  - Qwen 3-VL (vision + text).
  - BERT / Nomic-BERT (embedding-only).
- TurboQuant 8-bit KV cache (ADR-007) with Hadamard packing — drops
  F32 K/V allocations on Qwen 3.6 35B-A3B at 32K context for a
  **3.94× memory savings** vs the F32 baseline (ADR-027 iter-34).
- Flash-Attention prefill kernel (ADR-011) — `1.07–1.09× peer-FA`
  AHEAD at `pp1800–pp3700` (ADR-029 iter-160).
- Batched prefill V3 with byte-identical decode parity (ADR-029 Step
  1j.2).
- Speculative decode (ADR-030) — n-gram drafter Plan B available
  opt-in via `HF2Q_SPEC_NGRAM=1` (default OFF; DFlash investigation
  closed without shipping).

### Added — HTTP API (`hf2q serve`)

- OpenAI-compatible endpoints: `/v1/chat/completions`,
  `/v1/embeddings`, `/v1/models`.
- Streaming SSE.
- Tools / function-calling.
- Vision (`qwen3vl`) via `image_url` data-URI and HTTPS fetch.
- Grammar-constrained sampling.
- Persistent block-prefix KV cache (ADR-017).
- Multi-model serving from a single process.
- Jinja2 chat-template rendering matching `llama.cpp`'s vendored
  Jinja parser and mlx-lm's Python behavior, including the
  `pycompat` adapter for `.split()` / `.strip()` on strings
  (ADR-005 Phase 2a iter-133).

### Added — operator tools

- `hf2q doctor` — hardware / cache / RuVector / disk diagnostics.
- `hf2q info` — inspect an HF or GGUF model without converting.
- `hf2q validate` — cosine similarity, KL divergence, perplexity vs a
  reference.
- `hf2q parity` — ADR-009 parity validation against locked reference
  outputs.
- `hf2q smoke` — ADR-012 end-gate smoke test for a registered
  architecture.
- `hf2q gguf-patch` — rewrite GGUF metadata in place.
- `hf2q cache` — manage `~/.cache/hf2q/`.
- `hf2q completions` — shell completions.

### Performance highlights (M5 Max, thermal-fair alt-pair, σ < 1% per arm)

- **Gemma-4 decode** — `1.05× peer-FA` AHEAD across
  `tg200 / tg2000 / tg5000` after ADR-029 Step 1i (parallel
  SG-tournament top-K in `fused_moe_routing`) + Step 1j.2 (V3
  batched softmax tree-reduce). Byte-identical greedy decode vs V2
  baseline.
- **Qwen 3.6 35B-A3B-APEX-Q5_K_M decode** — `~1.34× peer-FA`
  sustained to 1000-tok (ADR-028 iter-308 → iter-324).
- **Prefill** — `1.07–1.09× peer-FA` AHEAD at `pp1800–pp3700`
  (ADR-029 iter-160).
- **KV-cache footprint** — 3.94× vs F32 baseline at 32K context on
  Qwen 3.6 35B-A3B (ADR-027 iter-34, Hadamard-packed TQ path).

### Notes

- macOS / Apple Silicon only (M1 or newer). The inference path is
  Metal-only by design (ADR-008 "candle divorce").
- DWQ at the production-default `perturb=1.0` is mathematically
  equivalent to the underlying K-quant baseline (ADR-020 finding
  2026-05-08). Wins materialize only at lower perturb values that
  move scales / biases off the K-quant projection.
- Per-arch disk floor for convert: 100 GB (Qwen 3.5 dense),
  150 GB (Qwen 3.5 MoE). Smoke preflight refuses to start below
  `disk_floor_gb + 10`.

[0.1.0]: https://github.com/robertelee78/hf2q/releases/tag/v0.1.0
