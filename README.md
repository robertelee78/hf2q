# hf2q

**Pure Rust CLI for converting and serving HuggingFace models on Apple Silicon.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust: 1.81.0+](https://img.shields.io/badge/rust-1.81.0%2B-orange.svg)](https://www.rust-lang.org)

---

hf2q converts HuggingFace safetensors models to hardware-optimized formats and serves them locally with an OpenAI-compatible API. Pure Rust, zero Python dependencies. Point it at a HuggingFace repo or local directory, get optimized weights or a running inference server.

## Quick Start

```bash
git clone https://github.com/robertelee78/hf2q.git
cd hf2q
cargo build --release
```

### Convert a model

```bash
# Download from HuggingFace and convert to MLX at 4-bit
hf2q convert --repo mlx-community/Qwen3-8B --format mlx --quant q4

# Auto quantization (learns from your hardware over time)
hf2q convert --input ./my-model --format mlx

# Preview without writing files
hf2q convert --repo mlx-community/Qwen3-8B --format mlx --dry-run
```

### Serve a model

```bash
# Start an OpenAI-compatible server (requires --features serve)
hf2q serve --model ./quantized-model --port 8080

# Then use any OpenAI-compatible client
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello"}]}'
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
| MLX | `mlx` | Available |
| CoreML | `coreml` | Available |
| GGUF | `gguf` | Planned |

### Auto Mode

With `--quant auto` (the default), hf2q picks quantization settings automatically:

1. Checks RuVector for an exact match from a prior conversion on the same hardware
2. Falls back to similar-model matches, adapting the stored recommendation
3. If no stored data exists, applies heuristics based on memory, model size, and architecture

Results are stored after each conversion, so auto mode improves over time.

## Inference Server

The `serve` subcommand starts an OpenAI-compatible API server backed by a Gemma 4 inference engine running on Metal. Requires the `serve` feature flag.

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List loaded models |
| `POST /v1/chat/completions` | Text generation (streaming and non-streaming) |
| `POST /v1/embeddings` | Text embeddings |
| `GET /health` | Health check |

### Capabilities

- **Streaming** -- SSE token streaming via `"stream": true`
- **Tool calling** -- function/tool call support in chat completions
- **Vision** -- multimodal input with SigLIP vision encoder for image understanding
- **Embeddings** -- text embeddings via prefill-only forward pass (separate concurrency lane)
- **Prompt caching** -- multi-turn conversations reuse cached KV state, reducing time-to-first-token
- **Concurrency** -- dual-lane semaphores: generation and embedding requests run independently

### `hf2q serve` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model <PATH>` | required | Local model directory or HuggingFace Hub ID |
| `--port <N>` | `8080` | Port to listen on |
| `--host <ADDR>` | `0.0.0.0` | Host address to bind |
| `--queue-depth <N>` | `16` | Max concurrent generation requests |
| `--embedding-concurrency <N>` | `4` | Max concurrent embedding requests |
| `--chat-template <PATH>` | -- | Custom Jinja2 chat template |
| `--no-prompt-cache` | off | Disable prompt caching |

### Model Support

Currently supports Gemma 4 (1B, 4B, 12B, 27B) with 30-layer MoE transformer, dual attention (sliding window + global), and tied embeddings.

## Other Commands

| Command | Description |
|---------|-------------|
| `hf2q info --repo <ID>` | Inspect model metadata |
| `hf2q doctor` | Diagnose system setup |
| `hf2q completions --shell <SHELL>` | Generate shell completions (bash, zsh, fish) |

## Building

```bash
cargo build                              # base conversion only
cargo build --features serve             # with inference server
cargo build --features mlx-backend       # with MLX quality measurement
cargo build --features ruvector           # with self-learning auto mode
cargo build --release --all-features     # everything
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `serve` | OpenAI-compatible inference server (includes mlx-native Metal backend) |
| `mlx-backend` | MLX inference via mlx-rs for quality measurement and DWQ calibration |
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
- **Linux support** -- NVIDIA/ROCm backends

## License

Apache-2.0 ([LICENSE](LICENSE))
