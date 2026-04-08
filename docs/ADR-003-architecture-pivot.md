# ADR-003: Architecture Pivot — Pure Quantization Tool

**Status:** Accepted
**Date:** 2026-04-08
**Deciders:** Robert (owner)
**Supersedes:** Previous CoreML-first architecture (docs/architecture.md)

---

## 1. Executive Summary

hf2q pivots from a multi-format conversion tool with built-in inference to a **focused, pure-Rust quantization CLI** that reads HuggingFace safetensors and writes optimally quantized weights for major inference runtimes (llama.cpp/Ollama, inferrs/Candle, vLLM). The built-in inference server and Apple-only output formats (CoreML, MLX) are removed. GPU-accelerated calibration is added via Candle in Phase 2.

---

## 2. Context and Problem Statement

### Current State (Pre-Pivot)

hf2q is a 12,598-line Rust codebase with three concerns tangled together:

1. **Quantization** — DWQ, mixed-bit, static quant, auto-quant intelligence (the core value)
2. **Format emission** — CoreML (1,013 lines, only implemented backend), GGUF/NVFP4 (stubs)
3. **Inference serving** — Full inference engine with model-specific code (Gemma4), vision encoder, stub runner (389 lines + dependencies)

Problems:
- **Only one output format works** (CoreML) — and it's Apple-only
- **Inference duplicates inferrs** — a separate Candle-based inference server that already supports Gemma4, Qwen3, multi-backend (Metal/CUDA/ROCm/Vulkan), and OpenAI/Anthropic/Ollama-compatible API
- **Heavy Apple-only dependencies** — `mlx-rs`, `coreml-native`, `mlx-native`, Metal toolchain required
- **Can't run on Linux** — the only implemented backend (CoreML) requires macOS
- **GGUF and safetensors output never got built** — the formats that 90% of users actually need

### The Opportunity

The quantization intelligence (DWQ, mixed-bit, sensitive layer protection, auto-quant, model fingerprinting) is genuinely differentiated. No other pure-Rust tool does calibration-aware quantization with per-layer sensitivity analysis. This is the moat — everything else is infrastructure.

---

## 3. Decision

**Strip hf2q to its quantization core. Target the output formats the ecosystem actually uses. Let inference runtimes handle serving.**

### What Changes

| Component | Action | Rationale |
|-----------|--------|-----------|
| `src/inference/` (389 lines) | **Delete entirely** | inferrs handles inference; hf2q doesn't need its own |
| `src/backends/coreml.rs` (1,013 lines) | **Delete** | Apple-only, not cross-platform |
| `src/backends/nvfp4.rs` (12 lines) | **Delete** | Stub, NVIDIA-specific |
| `mlx-backend` feature + deps | **Delete** | `mlx-rs`, `mlx-native`, Metal toolchain |
| `coreml-backend` feature + deps | **Delete** | `coreml-native` |
| `serve` feature + deps | **Delete** | `axum`, `tokio`, `tower-http`, `image`, `base64`, `minijinja`, `tokenizers`, `rand` |
| `src/backends/gguf.rs` | **Implement** (currently stub) | llama.cpp/Ollama compatibility |
| New: `src/backends/safetensors_out.rs` | **Implement** | inferrs/Candle/vLLM compatibility |
| `src/quantize/dwq.rs` | **Refactor** | Remove `InferenceRunner` dep; keep weight-space calibration |

### What Stays (Unchanged)

| Component | Lines | Why |
|-----------|-------|-----|
| `src/quantize/static_quant.rs` | 453 | Core: F16, Q8, Q4, Q2 round-to-nearest |
| `src/quantize/mixed.rs` | 471 | Core: Mixed-bit with sensitive layer protection |
| `src/quantize/mod.rs` | 135 | Core: Quantizer trait + quantize_model pipeline |
| `src/intelligence/auto_quant.rs` | 1,018 | Core: Auto-quant intelligence |
| `src/intelligence/fingerprint.rs` | 285 | Core: Model fingerprinting |
| `src/intelligence/heuristics.rs` | 346 | Core: Quantization heuristics |
| `src/intelligence/hardware.rs` | 440 | Keep: Hardware detection still useful |
| `src/intelligence/ruvector.rs` | 768 | Core: Self-learning vector DB |
| `src/quality/cosine_sim.rs` | 252 | Core: Weight-level quality check (no inference needed) |
| `src/quality/kl_divergence.rs` | 267 | Defer: Needs forward pass (Phase 2) |
| `src/quality/perplexity.rs` | 249 | Defer: Needs forward pass (Phase 2) |
| `src/input/` | 1,412 | Core: Safetensors reader, HF download, config parser |
| `src/ir.rs` | 524 | Core: Intermediate representation (no changes needed) |
| `src/cli.rs` | 409 | Modify: Update OutputFormat enum, remove dead formats |
| `src/main.rs` | 849 | Modify: Remove inference/serve code paths |
| `src/report.rs` | 554 | Core: Quantization reports |
| `src/preflight.rs` | 739 | Core: Pre-conversion checks |
| `src/progress.rs` | 199 | Core: Progress bars |
| `src/doctor.rs` | 333 | Core: Diagnostics |

### Net Impact

- **Lines deleted:** ~1,414 (inference: 389, coreml: 1,013, nvfp4: 12)
- **Lines added:** ~800-1,200 (GGUF backend, safetensors backend, CLI updates)
- **Dependencies removed:** 10+ optional crates (mlx-rs, coreml-native, axum, tokio, tower-http, image, base64, minijinja, tokenizers, rand)
- **Dependencies added:** 1 (`ggus` 0.5 — GGUF file writer, MIT, 29K monthly downloads)
- **Platform support:** macOS-only → macOS + Linux

---

## 4. Product Requirements

### 4.1 Mission Statement

hf2q is a **pure-Rust CLI tool** that takes any HuggingFace model and produces optimally quantized weights for any major inference runtime. One binary, cross-platform, no Python.

### 4.2 Target Users

| User | Workflow | Priority |
|------|----------|----------|
| **Ollama/llama.cpp user** | `hf2q convert --repo X --format gguf --quant q4` → drop GGUF into Ollama | High |
| **inferrs/Candle user** | `hf2q convert --repo X --format safetensors --quant mixed-4-6` → load in inferrs | High |
| **vLLM deployer** | `hf2q convert --repo X --format safetensors --quant q4` → serve on NVIDIA cluster | Medium |
| **Quality-conscious researcher** | `hf2q convert --repo X --format gguf --quant apex` → best quality/size tradeoff | High (Phase 2) |
| **CI/automation pipeline** | `hf2q convert ... --json-report --yes` → automated quantization | Medium |

### 4.3 Output Formats

| Format | Crate | Consumer | Phase |
|--------|-------|----------|-------|
| **GGUF** | `ggus` 0.5 | llama.cpp, Ollama | 1 |
| **Quantized safetensors** | `safetensors` 0.7 (already a dep) | inferrs, Candle, vLLM | 1 |
| **GPTQ** | TBD | vLLM (optimized CUDA kernels) | 4 (if benchmarks justify) |
| **AWQ** | TBD | vLLM (optimized CUDA kernels) | 4 (if benchmarks justify) |

### 4.4 Quantization Methods

| Method | Type | Calibration? | GPU? | Phase |
|--------|------|-------------|------|-------|
| **F16** | Passthrough | No | No | 1 |
| **Q8** | Static round-to-nearest | No | No | 1 |
| **Q4** | Static round-to-nearest | No | No | 1 |
| **Q2** | Static round-to-nearest | No | No | 1 |
| **Mixed-2-6** | Fixed mixed-bit per layer | No | No | 1 |
| **Mixed-3-6** | Fixed mixed-bit per layer | No | No | 1 |
| **Mixed-4-6** | Fixed mixed-bit per layer | No | No | 1 |
| **DWQ-Mixed-4-6** | Weight-space calibrated mixed-bit | Weight-space (CPU) | No | 1 |
| **DWQ-Mixed-4-6** | Activation-calibrated mixed-bit | Forward pass | Yes | 2 |
| **Apex** | imatrix-calibrated, 2-pass, per-tensor optimal | Forward pass | Yes | 2 |
| **Auto** | Intelligence-selected optimal method | Varies | Varies | 1 (static) / 2 (calibrated) |

### 4.5 CLI Surface

```
hf2q convert --repo <hf-id> --format <gguf|safetensors> --quant <method> [options]
hf2q info    --repo <hf-id>                     # inspect model metadata
hf2q validate --input <quantized-dir>            # quality checks on existing output (Phase 2+)
hf2q doctor                                      # diagnose environment (GPU, disk, deps)
hf2q completions --shell <bash|zsh|fish>         # shell completions
```

### 4.6 Architecture Post-Pivot

```
HuggingFace Hub / local safetensors
        |
        v
   [src/input/]          read safetensors + parse config.json
        |
        v
   [src/intelligence/]   auto_quant, fingerprint, heuristics, hardware detection
        |
        v
   [src/quantize/]       static, mixed-bit, DWQ, Apex --> IR (QuantizedModel)
        |
        v
   [src/quality/]        cosine_sim (Phase 1), perplexity + KL (Phase 2)
        |
        v
   [src/backends/]       write to target format
        |--- gguf.rs           --> llama.cpp / Ollama
        |--- safetensors.rs    --> inferrs / Candle / vLLM
        |--- gptq.rs           --> vLLM (Phase 4)
        '--- awq.rs            --> vLLM (Phase 4)
```

### 4.7 IR Contract (No Changes)

The existing Intermediate Representation works for all backends:

- **Input:** `TensorMap` (name -> `TensorRef` with shape, dtype, data bytes) + `ModelMetadata`
- **Quantize:** `TensorMap` -> `QuantizedModel` (name -> `QuantizedTensor` with packed data, scales, per-tensor `TensorQuantInfo`)
- **Output:** `OutputBackend::write(QuantizedModel)` -> `OutputManifest`

The `OutputBackend` trait already supports both pre-quantized IR consumption (`write()`) and native quantization (`quantize_and_write()`). No structural changes needed.

---

## 5. Phased Implementation Plan

### Phase 1: Foundation — Format Pivot & CPU Quantization

**Goal:** Working CLI that reads HF models and writes GGUF or quantized safetensors with all CPU-based quantization methods. Cross-platform (macOS + Linux).

#### Epic 1.1: Strip Inference & Dead Backends

| Story | Description | Files Affected |
|-------|-------------|----------------|
| 1.1.1 | Delete `src/inference/` entirely | `src/inference/**` |
| 1.1.2 | Delete `src/backends/coreml.rs` | `src/backends/coreml.rs` |
| 1.1.3 | Delete `src/backends/nvfp4.rs` | `src/backends/nvfp4.rs` |
| 1.1.4 | Remove feature flags: `mlx-backend`, `coreml-backend`, `mlx-native`, `serve` | `Cargo.toml` |
| 1.1.5 | Remove all dependencies behind deleted features | `Cargo.toml` |
| 1.1.6 | Update `src/backends/mod.rs` — remove coreml/nvfp4 module refs | `src/backends/mod.rs` |
| 1.1.7 | Refactor `src/quantize/dwq.rs` — remove `InferenceRunner` import, keep weight-space calibration | `src/quantize/dwq.rs` |
| 1.1.8 | Update `src/main.rs` — remove inference/serve code paths | `src/main.rs` |
| 1.1.9 | Update `src/quality/mod.rs` — gate perplexity/KL behind future feature flag | `src/quality/mod.rs` |

#### Epic 1.2: GGUF Backend

| Story | Description |
|-------|-------------|
| 1.2.1 | Add `ggus` 0.5 dependency to `Cargo.toml` |
| 1.2.2 | Implement `GgufBackend` struct implementing `OutputBackend` trait |
| 1.2.3 | Map IR `TensorQuantInfo` (bits, method) to GGML dtype enum |
| 1.2.4 | Write GGUF header + metadata (architecture, vocab_size, layer count, etc.) |
| 1.2.5 | Write quantized tensors with correct GGUF tensor naming conventions |
| 1.2.6 | Validate output loads in llama.cpp (`llama-cli`) |
| 1.2.7 | Integration test: quantize small reference model -> load in llama.cpp |

#### Epic 1.3: Quantized Safetensors Backend

| Story | Description |
|-------|-------------|
| 1.3.1 | Implement `SafetensorsBackend` struct implementing `OutputBackend` trait |
| 1.3.2 | Write quantized weights as safetensors with quantization metadata in header |
| 1.3.3 | Write `quantization_config.json` sidecar (per-layer bit assignments, method, group_size) |
| 1.3.4 | Support multi-shard output for large models |
| 1.3.5 | Validate output loads in Candle / inferrs |

#### Epic 1.4: CLI & Integration

| Story | Description |
|-------|-------------|
| 1.4.1 | Update `OutputFormat` enum: `{Gguf, Safetensors}` (remove Coreml, Nvfp4, Gptq, Awq) |
| 1.4.2 | Wire GGUF + Safetensors backends into main conversion pipeline |
| 1.4.3 | Update `resolve_convert_config` — remove "not yet implemented" bail for GGUF |
| 1.4.4 | Add `QuantMethod::Apex` to enum (stub — prints "requires Phase 2 GPU support") |
| 1.4.5 | Update `hf2q doctor` — remove MLX/CoreML checks, add GGUF/safetensors validation |
| 1.4.6 | Update report generation for new output formats |
| 1.4.7 | End-to-end test: `hf2q convert --repo <small-model> --format gguf --quant q4` |
| 1.4.8 | End-to-end test: `hf2q convert --repo <small-model> --format safetensors --quant mixed-4-6` |
| 1.4.9 | CI: Build and test on macOS + Linux (GitHub Actions) |

#### Phase 1 Acceptance Criteria

- [ ] `hf2q convert --format gguf --quant q4` produces valid GGUF loadable by llama.cpp
- [ ] `hf2q convert --format safetensors --quant mixed-4-6` produces valid quantized safetensors
- [ ] All quantization methods work: F16, Q8, Q4, Q2, Mixed-2-6, Mixed-3-6, Mixed-4-6, DWQ (weight-space)
- [ ] HF Hub download works (`--repo`)
- [ ] Builds on macOS (ARM64) and Linux (x86_64)
- [ ] Zero Apple-specific dependencies in default build
- [ ] All existing tests pass (minus deleted inference tests)
- [ ] JSON report output works for CI integration

---

### Phase 2: GPU Compute & Smart Quantization

**Goal:** Add Candle as GPU compute backend for calibration-based quantization (DWQ activation-based, Apex).

#### Epic 2.1: Candle GPU Backend

| Story | Description |
|-------|-------------|
| 2.1.1 | Add `candle-core` + `candle-nn` as optional deps (feature: `gpu`) |
| 2.1.2 | Metal support via `candle-core` features (macOS / Apple Silicon) |
| 2.1.3 | CUDA support via `candle-core` features (Linux / NVIDIA) |
| 2.1.4 | Hardware detection: pick Metal vs CUDA vs CPU fallback |
| 2.1.5 | Minimal forward pass module: load weights -> run through transformer layers -> capture activations |

#### Epic 2.2: Activation-Based DWQ Calibration

| Story | Description |
|-------|-------------|
| 2.2.1 | Implement calibration dataset loading (tokenization via `tokenizers` crate) |
| 2.2.2 | Run forward passes on original model, capture per-layer activation statistics |
| 2.2.3 | Compute per-layer sensitivity scores from activation variance/magnitude |
| 2.2.4 | Use sensitivity scores to drive mixed-bit allocation (smarter than fixed ranges) |
| 2.2.5 | Compare activation-DWQ vs weight-space-DWQ quality |

#### Epic 2.3: Apex Quantization

| Story | Description |
|-------|-------------|
| 2.3.1 | Implement importance matrix (imatrix) computation from calibration forward passes |
| 2.3.2 | Per-tensor optimal dtype selection using importance scores |
| 2.3.3 | Two-pass pipeline: Pass 1 (imatrix generation) -> Pass 2 (quantize with imatrix) |
| 2.3.4 | Support for GGUF K-quant types (Q4_K_M, Q5_K_M, Q6_K, etc.) in per-tensor assignment |
| 2.3.5 | Wire `--quant apex` through CLI to full pipeline |
| 2.3.6 | Validate APEX GGUF output matches quality of Python-produced reference |

#### Epic 2.4: Quality Validation Pipeline

| Story | Description |
|-------|-------------|
| 2.4.1 | Enable perplexity measurement using Candle forward passes |
| 2.4.2 | Enable KL divergence: original vs quantized output distributions |
| 2.4.3 | Activation-level cosine similarity (not just weight-level) |
| 2.4.4 | Quality thresholds: warn/fail if degradation exceeds configurable limits |
| 2.4.5 | Implement `hf2q validate` subcommand |

#### Phase 2 Acceptance Criteria

- [ ] `hf2q convert --format gguf --quant apex` produces imatrix-calibrated GGUF
- [ ] GPU-accelerated calibration works on Metal (macOS) and CUDA (Linux)
- [ ] CPU fallback works (slower but functional) when no GPU available
- [ ] Perplexity measurement validates quantization quality
- [ ] Apex output quality competitive with Python imatrix quantization

---

### Phase 3: Intelligence & Automation

**Goal:** Full auto-quant intelligence, model fingerprinting, and automated quality assurance.

#### Epic 3.1: Auto-Quant Intelligence

| Story | Description |
|-------|-------------|
| 3.1.1 | Given model fingerprint + target size, recommend optimal quant method + bit assignment |
| 3.1.2 | Model fingerprinting: identify architecture, layer count, parameter distribution, MoE structure |
| 3.1.3 | Per-layer sensitivity profiling: rank layers by quantization sensitivity |
| 3.1.4 | RuVector integration: learn from past conversion outcomes to improve recommendations |
| 3.1.5 | `--quant auto` selects optimal method based on model + target format + hardware |

#### Epic 3.2: Quality Assurance Automation

| Story | Description |
|-------|-------------|
| 3.2.1 | Automatic quality regression detection across quantization runs |
| 3.2.2 | Quality reports with before/after comparison metrics |
| 3.2.3 | CI-friendly JSON output with pass/fail quality gates |

#### Phase 3 Acceptance Criteria

- [ ] `--quant auto` produces near-optimal quantization without user tuning
- [ ] Quality validation catches bad quantizations automatically
- [ ] RuVector learns and improves recommendations over time

---

### Phase 4: Advanced Formats & Ecosystem

**Goal:** GPTQ/AWQ output for vLLM ecosystem, quantization manifest, ecosystem polish.

#### Epic 4.1: GPTQ Backend (Conditional)

| Story | Description |
|-------|-------------|
| 4.1.1 | Implement GPTQ column-wise weight packing |
| 4.1.2 | Write vLLM-compatible GPTQ safetensors layout |
| 4.1.3 | Validate output loads in vLLM |

#### Epic 4.2: AWQ Backend (Conditional)

| Story | Description |
|-------|-------------|
| 4.2.1 | Implement AWQ group-aware scaling |
| 4.2.2 | Write vLLM-compatible AWQ safetensors layout |
| 4.2.3 | Validate output loads in vLLM |

#### Epic 4.3: Ecosystem Integration

| Story | Description |
|-------|-------------|
| 4.3.1 | Quantization manifest sidecar (JSON): per-layer decisions, sensitivity scores, reproducibility metadata |
| 4.3.2 | HuggingFace model card generation: auto-generate README with quant details |
| 4.3.3 | Benchmark suite: quantization speed, output size, quality metrics across reference models |
| 4.3.4 | Plugin architecture for community-contributed backends |

#### Phase 4 Acceptance Criteria

- [ ] GPTQ/AWQ output (if implemented) loads in vLLM with optimized kernel paths
- [ ] Quantization manifest enables reproducible builds
- [ ] Benchmark suite validates performance across model families

---

## 6. Decisions Log

### D1: Drop CoreML/MLX output formats

**Decision:** Remove CoreML (only implemented backend) and all MLX dependencies.
**Rationale:** Apple-only, can't run on Linux, duplicates work that Apple's own tools do better. The ecosystem has moved to GGUF and safetensors.
**Trade-off:** Loses the 1,013 lines of working CoreML backend code. Acceptable — nobody was using it for production quantization.

### D2: Keep GGUF as an output format

**Decision:** Implement real GGUF backend (currently a 12-line stub).
**Rationale:** llama.cpp and Ollama are the dominant local inference ecosystem. GGUF is the universal format. hf2q's smart quantization (DWQ, mixed-bit, Apex) applied to GGUF is differentiated vs. llama.cpp's built-in quantize tool.

### D3: Kill the built-in inference server

**Decision:** Delete `src/inference/` entirely.
**Rationale:** inferrs (separate Candle-based server) already handles serving with Gemma4, Qwen3, multi-backend support. hf2q was duplicating this stack. inferrs is not our repo — we can't depend on it as a library, but we don't need to. hf2q's job is quantization, not serving.

### D4: Use `ggus` crate for GGUF writing

**Decision:** Depend on `ggus` 0.5 (MIT, 29K monthly downloads, InfiniTensor) instead of writing our own GGUF serializer.
**Rationale:** GGUF is a binary format with alignment quirks and metadata encoding conventions. `ggus` provides `GGufFileWriter` and `GGufTensorWriter` APIs. Focus engineering effort on quantization intelligence, not file format plumbing.

### D5: Candle for Phase 2 GPU compute (not inferrs)

**Decision:** Use `candle-core` + `candle-nn` as optional dependencies for GPU-accelerated calibration.
**Rationale:** inferrs is not our repo — can't depend on it as a crate or expect API changes. Candle is a published crate on crates.io with Metal + CUDA support. We use it as a compute library (tensor ops, forward passes), not as an inference engine.

### D6: DWQ weight-space calibration works in Phase 1 without GPU

**Decision:** Keep DWQ's weight-space closed-form scale calibration (optimal_scale = dot(W,Q)/dot(Q,Q)) in Phase 1. Defer activation-based calibration to Phase 2.
**Rationale:** The weight-space calibration in `dwq.rs` (lines 205-272) doesn't need forward passes or GPU. It's GPTQ-style optimal scale computation — strictly better than static quantization, works on CPU, and requires only removing the `InferenceRunner` import (which is used for a reserved-for-future step, not the actual calibration).

### D7: Apex is a quantization method, not an output format

**Decision:** Apex (imatrix-calibrated, 2-pass, per-tensor optimal precision) is a `QuantMethod` variant that outputs to any format (GGUF, safetensors).
**Rationale:** Based on the existing APEX GGUF at `jenerallee78/gemma-4-26B-A4B-it-ara-abliterated` — APEX is the calibration approach (importance matrix + per-tensor dtype selection), GGUF is the output container. The method is orthogonal to the format.

### D8: GPTQ/AWQ deferred to Phase 4 (conditional)

**Decision:** GPTQ and AWQ output formats are Phase 4, implemented only if benchmarks show meaningful speed/accuracy improvement over generic quantized safetensors in vLLM.
**Rationale:** vLLM loads quantized safetensors directly. GPTQ/AWQ provide optimized CUDA kernel paths that may or may not be faster depending on model size and hardware. Build the generic path first, measure, then decide.

---

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `ggus` crate has breaking changes | Delays GGUF backend | Pin version, vendoring as fallback |
| GGUF output incompatible with some llama.cpp versions | Users can't load output | Test against llama.cpp release + HEAD |
| Candle Phase 2 integration is harder than expected | Delays Apex/DWQ activation calibration | Phase 1 DWQ weight-space works without Candle |
| Quantized safetensors layout not standard | inferrs/vLLM can't load output | Follow HF transformers quantization_config.json convention |
| Losing CoreML backend alienates Apple-only users | Reduced user base | These users can use `coremltools` (Python) directly; hf2q targets the broader ecosystem |

---

## 8. Success Metrics

| Metric | Target | Phase |
|--------|--------|-------|
| Output formats that work end-to-end | 2 (GGUF + safetensors) | 1 |
| Platform support | macOS + Linux | 1 |
| Quantization methods working | 8 (F16, Q8, Q4, Q2, mixed x3, DWQ weight-space) | 1 |
| GGUF loads in llama.cpp/Ollama | Yes | 1 |
| Safetensors loads in Candle/inferrs | Yes | 1 |
| Apex quality matches Python imatrix pipeline | Within 0.1 perplexity | 2 |
| GPU-accelerated calibration | Metal + CUDA | 2 |
| Auto-quant recommends optimal settings | 90%+ user acceptance | 3 |
| Optional dependencies removed | 10+ crates | 1 |

---

## 9. References

- inferrs: `/opt/inferrs` — Candle-based inference server (not our repo, reference only)
- Existing architecture doc: `docs/architecture.md` (superseded by this ADR)
- APEX reference model: `huggingface.co/jenerallee78/gemma-4-26B-A4B-it-ara-abliterated`
- `ggus` crate: crates.io/crates/ggus (GGUF file writer)
- Candle: github.com/huggingface/candle (Rust ML framework)
