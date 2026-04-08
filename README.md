# hf2q

**Pure Rust CLI for converting HuggingFace models to hardware-optimized formats.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust: 1.81.0+](https://img.shields.io/badge/rust-1.81.0%2B-orange.svg)](https://www.rust-lang.org)

---

hf2q converts HuggingFace safetensors models to hardware-optimized formats. Pure Rust, zero Python dependencies. Point it at a HuggingFace repo or local directory, get optimized weights.

## Quick Start

```bash
git clone https://github.com/robertelee78/hf2q.git
cd hf2q
cargo build --release
```

### Convert a model

```bash
# Download from HuggingFace and convert to CoreML at 4-bit
hf2q convert --repo Qwen/Qwen3-8B --format coreml --quant q4

# Auto quantization (learns from your hardware over time)
hf2q convert --input ./my-model --format coreml

# Preview without writing files
hf2q convert --repo Qwen/Qwen3-8B --format coreml --dry-run
```

## Conversion

### Quantization Methods

| Method | Flag | Bits | Description |
|--------|------|------|-------------|
| Auto | `auto` | varies | Self-learning selection based on hardware and model (default) |
| Half precision | `f16` | 16 | Lossless float16, largest output |
| 8-bit | `q8` | 8 | High quality, moderate compression |
| 4-bit | `q4` | 4 | Good balance of quality and size |
| 2-bit | `q2` | 2 | Maximum compression, noticeable quality loss |
| Mixed 2-6 | `mixed-2-6` | 2-6 | Aggressive mixed-bit |
| Mixed 3-6 | `mixed-3-6` | 3-6 | Moderate mixed-bit |
| Mixed 4-6 | `mixed-4-6` | 4-6 | Conservative mixed-bit |
| DWQ Mixed 4-6 | `dwq-mixed-4-6` | 4-6 | Data-aware calibration for optimal bit allocation |

Use `--sensitive-layers` with mixed-bit methods to protect specific layers (e.g., LoRA-modified) at higher precision.

### Output Formats

| Format | Flag | Status |
|--------|------|--------|
| CoreML | `coreml` | Available |
| GGUF | `gguf` | Planned |
| NVFP4 | `nvfp4` | Planned |
| GPTQ | `gptq` | Planned |
| AWQ | `awq` | Planned |

### Auto Mode

With `--quant auto` (the default), hf2q picks quantization settings automatically:

1. Checks RuVector for an exact match from a prior conversion on the same hardware
2. Falls back to similar-model matches, adapting the stored recommendation
3. If no stored data exists, applies heuristics based on memory, model size, and architecture

Results are stored after each conversion, so auto mode improves over time.

## Other Commands

| Command | Description |
|---------|-------------|
| `hf2q info --repo <ID>` | Inspect model metadata |
| `hf2q doctor` | Diagnose system setup |
| `hf2q completions --shell <SHELL>` | Generate shell completions (bash, zsh, fish) |

## Building

```bash
cargo build                              # base conversion only
cargo build --features coreml-backend    # with CoreML output generation
cargo build --features ruvector           # with self-learning auto mode
cargo build --release --all-features     # everything
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `coreml-backend` | CoreML output generation |
| `ruvector` | Self-learning auto mode |

### Test & Lint

```bash
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

## Roadmap

- **GGUF output** -- cross-platform format for llama.cpp
- **NVIDIA formats** -- NVFP4, GPTQ, AWQ for CUDA-based inference
- **More architectures** -- Llama, Mistral, Qwen, and other model families
- **Inference server** -- OpenAI-compatible API server
- **Linux support** -- NVIDIA/ROCm backends

## License

Apache-2.0 ([LICENSE](LICENSE))
