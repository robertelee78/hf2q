# hf2q

Pure-Rust CLI that converts any HuggingFace model into optimally quantized weights for major inference runtimes. One binary, cross-platform, no Python.

## Features

- **GGUF output** for llama.cpp and Ollama
- **Quantized safetensors output** for Candle, inferrs, and vLLM
- **10 quantization methods** from simple round-to-nearest to calibration-aware mixed-bit
- **Apex imatrix quantization** -- importance-matrix-calibrated, per-tensor optimal precision
- **DWQ calibration** -- weight-space and activation-based mixed-bit allocation
- **Auto-quant intelligence** -- `--quant auto` selects the optimal method based on model architecture, hardware, and past outcomes
- **Quality validation** -- cosine similarity, KL divergence, and perplexity checks against the original model
- **GPU acceleration** -- Metal (macOS) and CUDA (Linux) via Candle, with CPU fallback
- **Self-learning** -- RuVector database stores conversion outcomes so recommendations improve over time
- **Cross-platform** -- builds and runs on macOS (ARM64) and Linux (x86_64)

## Quick Start

```bash
# Install from source
cargo install --path .

# Convert a model to GGUF with 4-bit quantization
hf2q convert --repo meta-llama/Llama-3-8B --format gguf --quant q4

# Convert to quantized safetensors with mixed-bit precision
hf2q convert --repo google/gemma-2-9b --format safetensors --quant mixed-4-6

# Let hf2q choose the optimal quantization method
hf2q convert --repo mistralai/Mistral-7B-v0.3 --format gguf --quant auto
```

## Usage

### convert

Convert a HuggingFace model to a quantized output format.

```bash
# From a HuggingFace repo (downloads automatically)
hf2q convert --repo <hf-repo-id> --format <gguf|safetensors> --quant <method>

# From a local safetensors directory
hf2q convert --input ./my-model/ --format gguf --quant q4

# Specify output directory
hf2q convert --repo <hf-repo-id> --format gguf --quant q8 --output ./output-dir/

# Protect sensitive layers at higher precision
hf2q convert --repo <hf-repo-id> --format gguf --quant mixed-4-6 --sensitive-layers "13-24"

# DWQ with custom calibration sample count
hf2q convert --repo <hf-repo-id> --format safetensors --quant dwq-mixed-4-6 --calibration-samples 2048

# Apex with target bits-per-weight
hf2q convert --repo <hf-repo-id> --format gguf --quant apex --target-bpw 4.5

# Custom bit width and group size
hf2q convert --repo <hf-repo-id> --format safetensors --quant q4 --bits 4 --group-size 128

# Dry run -- print the plan without converting
hf2q convert --repo <hf-repo-id> --format gguf --quant auto --dry-run

# CI mode -- JSON report, no prompts, quality gate
hf2q convert --repo <hf-repo-id> --format gguf --quant q4 --json-report --yes --quality-gate
```

### info

Inspect model metadata before converting.

```bash
hf2q info --repo meta-llama/Llama-3-8B
hf2q info --input ./local-model/
```

### validate

Check quantization quality by comparing a quantized model against its original.

```bash
hf2q validate --original ./original-model/ --quantized ./quantized-model/

# Custom thresholds
hf2q validate --original ./original/ --quantized ./quantized/ \
    --max-kl 0.05 --max-ppl-delta 1.0 --min-cosine 0.98

# JSON output for CI
hf2q validate --original ./original/ --quantized ./quantized/ --json
```

### doctor

Diagnose the environment: GPU availability, disk space, RuVector database health.

```bash
hf2q doctor
```

### completions

Generate shell completions.

```bash
hf2q completions --shell bash > ~/.bash_completion.d/hf2q
hf2q completions --shell zsh > ~/.zfunc/_hf2q
hf2q completions --shell fish > ~/.config/fish/completions/hf2q.fish
```

## Quantization Methods

| Method | Description | Calibration | GPU Required |
|--------|-------------|:-----------:|:------------:|
| `f16` | Float16 passthrough (no quantization) | No | No |
| `q8` | 8-bit round-to-nearest | No | No |
| `q4` | 4-bit round-to-nearest | No | No |
| `q2` | 2-bit round-to-nearest | No | No |
| `mixed-2-6` | Fixed mixed-bit, 2-6 bits per layer | No | No |
| `mixed-3-6` | Fixed mixed-bit, 3-6 bits per layer | No | No |
| `mixed-4-6` | Fixed mixed-bit, 4-6 bits per layer | No | No |
| `dwq-mixed-4-6` | DWQ weight-space calibrated mixed-bit | Weight-space (CPU) | No |
| `apex` | imatrix-calibrated, per-tensor optimal precision | Forward pass | Yes |
| `auto` | Intelligence-selected optimal method | Varies | Varies |

Use `--sensitive-layers` with any mixed-bit method to protect specific layers at higher precision:

```bash
hf2q convert --repo <id> --format gguf --quant mixed-4-6 --sensitive-layers "1,5,13-24"
```

## Output Formats

### GGUF

For llama.cpp and Ollama. Produces a single `.gguf` file ready to load.

```bash
hf2q convert --repo <id> --format gguf --quant q4
# Output: model-gguf-q4/model.gguf
```

### Quantized Safetensors

For Candle, inferrs, and vLLM. Produces quantized `.safetensors` files with a `quantization_config.json` sidecar describing per-layer bit assignments, method, and group size.

```bash
hf2q convert --repo <id> --format safetensors --quant mixed-4-6
# Output: model-safetensors-mixed-4-6/*.safetensors + quantization_config.json
```

## Auto Mode

When `--quant auto` is specified, hf2q selects the optimal quantization method through a three-tier decision process:

1. **RuVector lookup** -- queries the self-learning database for past conversion outcomes on similar models (requires the `ruvector` feature)
2. **auto_quant intelligence** -- analyzes the model fingerprint (architecture, layer count, parameter distribution, MoE structure) against target format and available hardware
3. **Heuristic fallback** -- applies rule-based defaults when no prior data exists

Auto mode considers model size, target format constraints, available memory, and GPU capabilities. Over time, RuVector learns which methods produce the best quality-to-size ratio for each model family.

## Quality Validation

hf2q measures quantization quality at two levels:

**During conversion** (unless `--skip-quality` is set):
- Weight-level cosine similarity between original and quantized tensors

**Post-conversion** via `validate`:
- KL divergence between original and quantized output distributions
- Perplexity delta on calibration text
- Activation-level cosine similarity

Use `--quality-gate` to fail the conversion (exit code 2) if quality thresholds are exceeded. Use `--json-report` to emit structured output for CI pipelines.

## GPU Acceleration

hf2q uses Candle for GPU-accelerated operations. GPU is used for calibration-based quantization methods (Apex, activation-based DWQ) and quality validation forward passes.

| Platform | Backend | Feature Flag |
|----------|---------|-------------|
| macOS (Apple Silicon) | Metal | `--features metal` |
| Linux (NVIDIA) | CUDA | `--features cuda` |
| Any | CPU fallback | default (no flag) |

All CPU-only quantization methods (F16, Q8, Q4, Q2, mixed-bit, DWQ weight-space) work without GPU acceleration.

## Building

```bash
# Default build (CPU only)
cargo build --release

# With Metal GPU support (macOS)
cargo build --release --features metal

# With CUDA GPU support (Linux)
cargo build --release --features cuda

# With RuVector self-learning
cargo build --release --features ruvector

# All features (macOS)
cargo build --release --features "metal,ruvector"
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `metal` | Metal GPU acceleration via Candle (macOS / Apple Silicon) |
| `cuda` | CUDA GPU acceleration via Candle (Linux / NVIDIA) |
| `ruvector` | RuVector self-learning database for auto-quant intelligence |

### Running Tests

```bash
cargo test
```

### Minimum Rust Version

Rust 1.81.0 or later.

## License

Apache-2.0
