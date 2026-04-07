# hf2q

**Pure Rust CLI for converting HuggingFace models to hardware-optimized formats.**

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Rust: 1.81.0+](https://img.shields.io/badge/rust-1.81.0%2B-orange.svg)](https://www.rust-lang.org)

---

## What is hf2q?

hf2q is a command-line tool that converts HuggingFace safetensors models into hardware-optimized formats for local inference. It is written entirely in Rust with zero Python dependencies -- no virtual environments, no pip, no torch. Point it at a HuggingFace repo or a local model directory and get optimized weights ready for Apple Silicon in a single command.

The tool supports MLX and CoreML output formats, with static quantization (f16, q8, q4, q2), mixed-bit quantization for fine-grained control, and DWQ (Data-aware Weight Quantization) calibration for maximum quality. A self-learning auto mode powered by RuVector analyzes your hardware and model characteristics, then improves its recommendations with every conversion you run.

hf2q is designed for developers who want a fast, reproducible model conversion pipeline that fits cleanly into CI/CD workflows. It provides structured JSON reports, deterministic exit codes, and non-interactive flags for automation. For interactive use, it offers dry-run previews, quality measurement, and progress reporting.

## Features

- **Zero Python** -- pure Rust from download to output, no Python runtime needed
- **Direct HuggingFace download** -- `--repo org/model` fetches safetensors directly from the Hub
- **Static quantization** -- f16, q8, q4, q2 round-to-nearest methods
- **Mixed-bit quantization** -- per-layer bit allocation with mixed-2-6, mixed-3-6, mixed-4-6 profiles
- **DWQ calibration** -- data-aware weight quantization with configurable sample counts
- **Sensitive layer protection** -- `--sensitive-layers 13-24` keeps modified layers (LoRA, DPO, ARA) at higher precision
- **Self-learning auto mode** -- RuVector stores conversion results and improves recommendations over time
- **Quality measurement** -- KL divergence, perplexity delta, and cosine similarity scoring
- **Hardware profiling** -- automatic detection of chip, memory, and core count for optimal settings
- **JSON reports** -- structured output for CI/automation with `--json-report`
- **Dry-run mode** -- preview the conversion plan without writing files
- **Graceful interruption** -- Ctrl+C cleans up partial output directories
- **Memory-mapped I/O** -- handles large models (48GB+) efficiently
- **Shell completions** -- bash, zsh, and fish completion scripts
- **Deterministic exit codes** -- 0 (success), 1 (conversion error), 2 (quality exceeded), 3 (input error)

## Quick Start

### Install

```bash
# Clone and build
git clone https://github.com/robertelee78/hf2q.git
cd hf2q
cargo build --release

# Optional: install to PATH
cargo install --path .
```

### First Conversion

```bash
# Convert a local model directory to MLX format with auto quantization
hf2q convert --input ./my-model --format mlx

# Download from HuggingFace and convert to MLX at 4-bit
hf2q convert --repo mlx-community/Qwen3-8B --format mlx --quant q4

# Preview what would happen without converting
hf2q convert --repo mlx-community/Qwen3-8B --format mlx --dry-run

# Inspect model metadata before converting
hf2q info --repo mlx-community/Qwen3-8B
```

## CLI Reference

### Global Flags

| Flag | Description |
|------|-------------|
| `-v`, `-vv`, `-vvv` | Increase logging verbosity (info, debug, trace) |
| `--version` | Print version |
| `--help` | Print help |

### `hf2q convert`

Convert a HuggingFace model to a hardware-optimized format.

```bash
hf2q convert [OPTIONS] --format <FORMAT>
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input <DIR>` | -- | Local safetensors directory (mutually exclusive with `--repo`) |
| `--repo <ID>` | -- | HuggingFace repo ID, downloads automatically |
| `--format <FMT>` | required | Output format: `mlx`, `coreml` |
| `--quant <METHOD>` | `auto` | Quantization method (see table below) |
| `--sensitive-layers <SPEC>` | -- | Layer ranges to protect, e.g. `13-24` or `1,5,13-24` |
| `--calibration-samples <N>` | `1024` | Sample count for DWQ calibration |
| `--bits <N>` | method default | Custom bit width (2-8) |
| `--group-size <N>` | `64` | Quantization group size: `32`, `64`, `128` |
| `-o`, `--output <DIR>` | auto-generated | Output directory |
| `--json-report` | off | Emit structured JSON report |
| `--skip-quality` | off | Skip quality measurement |
| `--dry-run` | off | Show conversion plan without writing files |
| `--yes` | off | Non-interactive mode, skip confirmation prompts |
| `--unsupported-layers <POLICY>` | -- | How to handle unsupported layers: `passthrough` |

#### Examples

```bash
# Mixed-bit quantization with sensitive layer protection
hf2q convert --input ./finetuned-model --format mlx \
  --quant mixed-4-6 --sensitive-layers 13-24

# DWQ calibration with custom sample count
hf2q convert --repo org/model --format mlx \
  --quant dwq-mixed-4-6 --calibration-samples 2048

# CI-friendly: non-interactive with JSON report to stdout
hf2q convert --input ./model --format mlx --quant q4 \
  --json-report --yes --skip-quality

# Custom output directory and group size
hf2q convert --input ./model --format coreml \
  --quant q8 --group-size 128 --output ./optimized
```

### `hf2q info`

Inspect model metadata without converting.

```bash
hf2q info --input ./model-dir
hf2q info --repo mlx-community/Qwen3-8B
```

### `hf2q doctor`

Diagnose system setup: RuVector availability, hardware detection, MLX backend, and disk space.

```bash
hf2q doctor
```

### `hf2q completions`

Generate shell completion scripts.

```bash
# Bash
hf2q completions --shell bash > ~/.local/share/bash-completion/completions/hf2q

# Zsh
hf2q completions --shell zsh > ~/.zfunc/_hf2q

# Fish
hf2q completions --shell fish > ~/.config/fish/completions/hf2q.fish
```

## Quantization Methods

| Method | Flag | Bits | Description |
|--------|------|------|-------------|
| Auto | `auto` | varies | Self-learning selection based on hardware and model (default) |
| Half precision | `f16` | 16 | Lossless float16 conversion, largest output |
| 8-bit | `q8` | 8 | High quality, moderate compression |
| 4-bit | `q4` | 4 | Good balance of quality and size |
| 2-bit | `q2` | 2 | Maximum compression, noticeable quality loss |
| Mixed 2-6 | `mixed-2-6` | 2-6 | Aggressive mixed-bit, sensitive layers at 6-bit |
| Mixed 3-6 | `mixed-3-6` | 3-6 | Moderate mixed-bit, sensitive layers at 6-bit |
| Mixed 4-6 | `mixed-4-6` | 4-6 | Conservative mixed-bit, sensitive layers at 6-bit |
| DWQ Mixed 4-6 | `dwq-mixed-4-6` | 4-6 | Data-aware calibration for optimal bit allocation |

Use `--sensitive-layers` with any mixed-bit method to protect specific layers (e.g., LoRA-modified layers) at the higher bit width.

## Auto Mode

When `--quant auto` is used (the default), hf2q resolves the optimal quantization settings through a three-step process:

1. **RuVector exact match** -- checks for a stored result from a previous conversion of the same model on the same hardware. If found, reuses it immediately.
2. **RuVector similar match** -- searches for conversions of similar models on similar hardware. Adapts the stored recommendation to the current context.
3. **Heuristic fallback** -- if no stored data exists, applies rule-based heuristics considering available memory, model size, layer count, and architecture type.

After every conversion, hf2q stores the result (quantization method, quality metrics, hardware profile, model fingerprint) in the local RuVector database. Over time, auto mode becomes more accurate for your specific hardware and the models you work with.

The resolution is displayed before conversion starts, showing the chosen method, confidence level, and reasoning.

## Output Formats

| Format | Flag | Platform | Status |
|--------|------|----------|--------|
| MLX | `mlx` | macOS Apple Silicon | Available |
| CoreML | `coreml` | macOS Apple Silicon | Available |
| GGUF | `gguf` | Cross-platform | Planned |
| NVFP4 | `nvfp4` | NVIDIA GPUs | Planned |
| GPTQ | `gptq` | NVIDIA GPUs | Planned |
| AWQ | `awq` | NVIDIA GPUs | Planned |

**MLX** output produces sharded safetensors files with an MLX-compatible weight map, ready for use with the MLX framework on Apple Silicon.

**CoreML** output produces a Core ML model package optimized for Apple's Neural Engine and GPU.

## Quality Measurement

Unless `--skip-quality` is passed, hf2q measures the quality impact of quantization using three metrics:

| Metric | Description | Ideal |
|--------|-------------|-------|
| **KL Divergence** | Measures how much the quantized output distribution diverges from the original | 0.0 (identical) |
| **Perplexity Delta** | Change in perplexity between original and quantized model | 0.0 (no change) |
| **Cosine Similarity** | Average cosine similarity of tensor values before and after quantization | 1.0 (identical) |

Quality measurement requires the `mlx-backend` feature to be enabled for inference. Without it, quality metrics are skipped with a log message.

When `--json-report` is used, quality metrics are included in the structured output for programmatic analysis.

## Architecture

hf2q follows a pipeline architecture with trait-based extensibility:

```
Input (safetensors) --> IR (TensorMap + ModelMetadata) --> Quantize --> Backend --> Output
```

- **Input layer** -- reads safetensors shards via memory-mapped I/O, parses `config.json` for model metadata, downloads from HuggingFace Hub
- **IR (Intermediate Representation)** -- `TensorMap` holds all tensors in a normalized format; `ModelMetadata` captures architecture details
- **Quantization** -- the `Quantizer` trait dispatches to static, mixed-bit, or DWQ implementations
- **Backend** -- the `OutputBackend` trait produces format-specific output (MLX shards, CoreML packages)
- **Intelligence** -- hardware profiling, model fingerprinting, RuVector self-learning, and heuristic auto mode

All backends implement the `OutputBackend` trait. All quantization methods implement the `Quantizer` trait. Adding a new format or quantization method requires implementing the relevant trait without modifying existing code.

## Development

### Prerequisites

- Rust 1.81.0 or later
- macOS with Apple Silicon (for MLX and CoreML backends)
- Metal toolchain (for `mlx-backend` feature)

### Build

```bash
# Default build (no optional backends)
cargo build

# With MLX backend
cargo build --features mlx-backend

# With all features
cargo build --features mlx-backend,coreml-backend,ruvector

# Release build
cargo build --release --features mlx-backend,ruvector
```

### Test

```bash
# Run all tests
cargo test

# Run tests with all features
cargo test --all-features

# Run a specific test
cargo test test_parse_sensitive_layers
```

### Lint

```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt -- --check
```

### Benchmarks

```bash
cargo bench --bench quantize_bench
cargo bench --bench shard_read_bench
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `mlx-backend` | off | MLX inference via mlx-rs (requires Apple Silicon + Metal) |
| `coreml-backend` | off | CoreML output generation (requires Apple Silicon) |
| `ruvector` | off | Self-learning auto mode via RuVector |

No features are enabled by default. The tool compiles and runs basic conversions (f16, q8, q4, q2 to MLX/CoreML format) without any optional features. Quality measurement and DWQ calibration require `mlx-backend`. Auto mode learning requires `ruvector`.

## Roadmap

- **Inference engine** -- `hf2q serve` with an OpenAI-compatible API for local inference
- **GGUF output** -- cross-platform format for llama.cpp and compatible runtimes
- **NVIDIA formats** -- NVFP4, GPTQ, AWQ for CUDA-based inference
- **More architectures** -- Gemma 4, Llama, Mistral, and other popular model families
- **Linux support** -- full backend support on Linux with NVIDIA/ROCm
- **AMD/Intel targets** -- ROCm (AMD) and OpenVINO (Intel) backends

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
