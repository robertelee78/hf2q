---
stepsCompleted: ['step-01-init', 'step-02-discovery', 'step-02b-vision', 'step-02c-executive-summary', 'step-03-success', 'step-04-journeys', 'step-05-domain', 'step-06-innovation', 'step-07-project-type', 'step-08-scoping', 'step-09-functional', 'step-10-nonfunctional', 'step-11-polish']
inputDocuments: []
workflowType: 'prd'
documentCounts:
  briefs: 0
  research: 0
  brainstorming: 0
  projectDocs: 0
classification:
  projectType: cli_tool
  domain: scientific
  complexity: medium
  projectContext: greenfield
---

# Product Requirements Document - hf2q

**Author:** Robert
**Date:** 2026-04-06

## Engineering Philosophy

> DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.

This mantra applies to all implementation work. Every epic, story, and code change must embody this standard.

## Executive Summary

**hf2q** is a pure Rust CLI tool that converts HuggingFace model weights to hardware-optimized formats — eliminating Python entirely from the model conversion pipeline. Initially targeting Apple Silicon (MLX, CoreML), with a roadmap to Linux/NVIDIA targets (NVFP4, GPTQ, AWQ, GGUF). It serves AI practitioners who work with open-weight models and want a single, Python-free tool for quantization and format conversion across platforms.

The tool operates at three tiers: a quick mode for fast static quantization, an expert gold mode with Distilled Weight Quantization (DWQ) and per-layer bit allocation, and a self-learning auto mode powered by RuVector that determines optimal conversion settings based on the user's hardware and model architecture. Auto mode improves with every conversion, building a knowledge base of `(hardware + model) → optimal config` mappings that can be shared across the community.

hf2q supports arbitrary HuggingFace model architectures (not just specific families), downloads models directly from HuggingFace Hub via the `hf-hub` crate (with `hf` CLI fallback), and produces consolidated, publication-quality output — including per-layer quantization configs and quality benchmarks.

### What Makes This Special

1. **First pure Rust HF model converter.** The entire ML ecosystem funnels through Python for model conversion. Even Rust-native inference tools like `mlx-server` require `pip install mlx-lm` to prepare models. hf2q closes that gap — zero Python, anywhere in the pipeline. macOS first, Linux next.

2. **Gold mode with `--sensitive-layers`.** DWQ calibration with per-layer bit allocation and explicit protection for ARA-steered or otherwise modified layers. Purpose-built for practitioners who uncensor, fine-tune, or steer models and need quantization that respects their modifications. No other tool offers this.

3. **Self-learning auto mode via RuVector.** The tool profiles hardware, fingerprints the model architecture, and queries a self-optimizing vector database for the best conversion settings. Unknown combinations trigger a calibration sweep, learn, and store results. The tool gets smarter with every use.

## Project Classification

- **Project Type:** CLI Tool
- **Domain:** Scientific / ML Tooling
- **Complexity:** Medium — sophisticated quantization algorithms and hardware-aware optimization, no regulatory overhead
- **Project Context:** Greenfield — first-of-its-kind tool, no existing codebase

## Success Criteria

### User Success

- **Zero-friction conversion:** A single `hf2q convert` command takes a HuggingFace model (local or remote) to a hardware-optimized format with no Python dependency, no environment setup, no version conflicts.
- **Confidence in output quality:** Every conversion produces quality metrics (KL divergence, perplexity delta) by default so the user knows the output is correct — not just "it ran."
- **Auto mode delivers:** Running `--quant auto` (the default) produces results equal to or better than manual tuning. The tool remembers what worked.
- **Gold mode respects modifications:** Practitioners who uncensor, steer, or fine-tune models can protect their modified layers during quantization with `--sensitive-layers`.

### Business Success

- **Genuinely useful to Robert:** The tool reliably handles every model conversion needed in an uncensoring/optimization workflow. If it solves this one user's real problem every time, it's successful.
- **Community validation (bonus, not goal):** If others find it useful — great. Built for utility, not vanity metrics.
- **Foundation for multi-platform:** Architecture supports adding Linux/NVIDIA output targets without rewriting the core pipeline.

### Technical Success

- **Arbitrary HF architectures:** Handles any model with a config.json and safetensors weights — not hardcoded to specific model families.
- **Output parity:** Converted models produce equivalent inference quality to Python `mlx-lm` conversions (within measurable KL divergence tolerance).
- **Performance:** Conversion speed competitive with or faster than Python tooling, leveraging Rust + rayon parallelism on 48GB+ weight files.
- **RuVector integration:** Auto mode learns from conversions and improves recommendations over time. Knowledge persists across sessions.

### Measurable Outcomes

- KL divergence for DWQ mixed-4-6 gold mode: < 0.3 vs f16 baseline
- Conversion of a 26B model (32 shards): completes without OOM on 128GB M5 Max
- Auto mode: after 5+ conversions, recommendations match or beat manual expert settings
- Zero Python processes spawned, ever

## User Journeys

### Journey 1: Robert — The Expert Practitioner (Gold Mode)

**Opening Scene:** Robert has just finished a dual-pass ARA uncensoring run on a new Gemma 4 27B model. The output is 32 safetensors shards sitting in `/opt/gemma4/`. He wants to run inference on his M5 Max — but the weights are raw bf16, unoptimized, and no MLX-compatible config exists.

**Rising Action:** He runs `hf2q info --input ./gemma4` to inspect the model — confirms 26B params, MoE 128x8, identifies the ARA-steered layers 13-24. Then:
```bash
hf2q convert --input ./gemma4 --format mlx --quant dwq-mixed-4-6 \
  --sensitive-layers 13-24 --calibration-samples 1024
```
The tool profiles his hardware (M5 Max, 128GB), loads shards in parallel via rayon, runs calibration passes, and applies 6-bit precision to the steered layers while compressing the rest to 4-bit. Progress bars show each phase.

**Climax:** Conversion completes. The output directory contains 4 consolidated shards, an MLX config, a `quantization_config.json` showing the per-layer bit map, and a KL divergence report: overall delta of 0.028, steered layers at 0.003-0.005. The modifications are intact.

**Resolution:** Robert loads the model in `mlx-server` and it runs perfectly — fast inference, uncensored behavior preserved, no Python touched at any point.

### Journey 2: Sam — The Curious Newcomer (Quick Mode)

**Opening Scene:** Sam just got an M4 MacBook Pro with 48GB. He wants to try running a local LLM. He found hf2q mentioned in a Reddit thread that said "just install it with cargo and run one command."

**Rising Action:** Sam installs hf2q (`cargo install hf2q`) and runs:
```bash
hf2q convert --repo meta-llama/Llama-3.1-8B-Instruct --format mlx --quant q4
```
The tool downloads the model from HuggingFace Hub, shows a progress bar, then converts. Sam doesn't know what group sizes or bit allocations are — and he doesn't need to.

**Climax:** Two minutes later, the output directory appears with clean MLX-format safetensors. Sam points `mlx-server` at it and gets a working chat interface. No Python, no conda, no version conflicts.

**Resolution:** Sam starts experimenting with other models. He tries `--quant auto` on a larger model and hf2q picks optimal settings for his 48GB hardware automatically.

### Journey 3: Robert — Error Recovery (Edge Case)

**Opening Scene:** Robert tries to convert a brand-new architecture with a custom attention mechanism that just dropped on HuggingFace.

**Rising Action:** He runs `hf2q convert --input ./experimental-model --format mlx --quant q4`. The tool reads config.json, parses the architecture, but hits an unsupported layer type.

**Climax:** Instead of silently producing garbage, hf2q reports:
```
Warning: Unknown layer type 'cross_mamba_attention' in layers 4-8
  → Falling back to f16 passthrough for these layers
  → Remaining layers quantized to q4 as requested
  → Output may be larger than expected (est. 12GB vs 8GB)
Proceed? [Y/n]
```

**Resolution:** Robert proceeds. The model works — slightly larger than a full q4, but functional. The tool degraded gracefully instead of failing.

### Journey 4: CI Pipeline — Batch Automation

**Opening Scene:** A project wants to automate model conversion: when a new model appears on HuggingFace, convert it and publish the optimized version.

**Rising Action:** The CI script runs:
```bash
hf2q convert --repo $MODEL_REPO --format mlx --quant auto \
  --output ./converted/$MODEL_NAME --json-report --yes
```
The `--json-report` flag outputs structured results for automated quality checks. `--yes` skips prompts. Fully scriptable.

**Climax:** The pipeline catches a model where auto mode's KL divergence exceeds the threshold. It flags the conversion for manual review.

**Resolution:** The project maintains a catalog of quality-verified optimized models, all converted without Python, all with published quant configs and benchmarks.

### Journey Requirements Summary

| Journey | Capabilities Revealed |
|---|---|
| Robert (Gold) | DWQ calibration, `--sensitive-layers`, per-layer bit map, KL divergence report, shard consolidation, hardware profiling |
| Sam (Quick) | HF Hub download, sane defaults, progress reporting, zero-config conversion, `--quant auto` |
| Robert (Edge Case) | Graceful fallback for unknown architectures, clear warnings, user confirmation prompts |
| CI Pipeline | `--json-report`, `--yes` non-interactive mode, scriptable exit codes, quality thresholds |

## Domain-Specific Requirements

### Numerical Correctness & Quality Measurement

- **KL divergence** between base (pre-quant) and quantized output distributions is the primary quality metric. Computed per-layer and overall during all conversions by default.
- **Perplexity delta** as a secondary quick-check metric.
- **Cosine similarity** of layer activations pre/post quant available for debugging which layers degraded most.
- Per-layer KL divergence report included in output, especially for `--sensitive-layers` to prove steered layers maintained fidelity.
- Auto mode optimizes for minimum KL divergence, not just perplexity.

### Weight Format Compatibility

- Each output target (MLX, CoreML, GGUF, NVFP4) has specific tensor layout requirements, endianness, dtype expectations, and metadata schemas. Output must be validated against format spec before writing.
- Getting any format detail wrong produces models that load but output garbage — validation is mandatory.

### Memory Safety During Conversion

- 48GB+ weight files must be processed via streaming/memory-mapped I/O — never load all shards simultaneously.
- Conversion must complete without OOM on machines where the model barely fits (e.g., 27B model on 64GB machine).
- Rayon parallelism must respect memory bounds, not just CPU cores.

### Model Architecture Drift

- HuggingFace model architectures evolve weekly — new attention mechanisms, MoE routing schemes, and layer types appear constantly.
- Unknown layer types must fall back to f16 passthrough with clear warnings, not hard failures.
- Architecture support should be data-driven (config.json parsing) rather than hardcoded per-model-family where possible.

## Innovation & Novel Patterns

### Detected Innovation Areas

1. **First pure Rust model conversion pipeline.** Every existing tool — `mlx-lm`, `coremltools`, `auto-gptq`, `bitsandbytes` — is Python. hf2q breaks this dependency entirely, enabled by the maturation of the Rust ML crate ecosystem (`mlx-rs`, `safetensors`, `hf-hub`).

2. **Self-learning conversion optimizer.** CLI tools don't learn. hf2q's integration of RuVector (self-optimizing vector database with SONA engine) means conversion settings improve over time. Each `(hardware + model + target) → optimal config` mapping is stored, learned from, and refined.

3. **Modification-aware quantization.** The `--sensitive-layers` flag acknowledges that model practitioners modify weights (ARA uncensoring, LoRA fine-tuning, DPO steering) and those modifications must survive quantization. Existing tools treat all layers uniformly. hf2q lets users declare which layers matter most.

4. **KL divergence as optimization objective.** Auto mode minimizes measured KL divergence between pre-quant and post-quant distributions, with per-layer reporting. Research-grade quality assurance packaged into a CLI flag.

### Competitive Landscape

- **Python tools (mlx-lm, coremltools, auto-gptq):** Dominant but require Python. No self-learning, no modification-aware quantization.
- **llama.cpp quantize:** C++ based, GGUF-only, no MLX/CoreML output, no adaptive optimization.
- **mlx-rs / mlx-server:** Rust inference but defers to Python for conversion — hf2q fills exactly this gap.
- **No direct competitor** exists in the pure Rust model conversion space.

### Validation Approach

- **Output parity testing:** Convert identical models with hf2q and Python `mlx-lm`, compare KL divergence and perplexity — must be equivalent or better.
- **Auto mode convergence:** Track whether recommendations improve over 10+ conversions on the same hardware.
- **Sensitive-layers validation:** Convert ARA-steered models with and without `--sensitive-layers`; measure KL divergence on steered layers specifically.

## CLI Tool Specific Requirements

### Command Structure

```
hf2q
├── convert          # Primary command — convert model to target format
│   ├── --input      # Local safetensors directory
│   ├── --repo       # HuggingFace repo ID (downloads automatically)
│   ├── --format     # Output target: mlx, coreml (future: nvfp4, gptq, gguf)
│   ├── --quant      # Quantization: auto (default), f16, q8, q4, q2,
│   │                #   q4-mxfp, mixed-2-6, mixed-3-6, mixed-4-6,
│   │                #   dwq-mixed-4-6, or custom --bits/--group-size
│   ├── --sensitive-layers  # Layer ranges to protect at higher precision (e.g., 13-24)
│   ├── --calibration-samples  # Sample count for DWQ calibration (default: 1024)
│   ├── --bits       # Custom bit width (2-8)
│   ├── --group-size # Custom group size (32, 64, 128)
│   ├── --output     # Output directory (default: ./model-{format}-{quant}/)
│   ├── --json-report  # Emit structured JSON report for CI/automation
│   ├── --skip-quality # Skip KL divergence / perplexity measurement
│   ├── --dry-run    # Run preflight + auto resolution, print plan, exit
│   └── --yes        # Non-interactive, skip confirmation prompts
├── info             # Inspect model before converting
│   └── --input      # Local safetensors directory or HF repo
├── doctor           # Diagnose RuVector, hardware, mlx-rs, disk space
└── completions      # Generate shell completions
    └── --shell      # bash, zsh, fish
```

### Output Formats

**Terminal output (default):**
- Progress bars for download, shard loading, quantization, and quality measurement phases
- Conversion summary with quant config, output size, and quality metrics (KL divergence, perplexity delta)
- Warnings for unknown layer types or fallback decisions
- Confirmation prompts for edge cases

**JSON report (`--json-report`):**
- Machine-readable: input/output paths, quant config, per-layer bit allocation, quality metrics, timings, hardware profile
- Suitable for CI pipelines, quality gates, and automated cataloging

**Model output directory:**
- Consolidated safetensors shards (target: 4 shards)
- Format-specific config (MLX `config.json`, CoreML `.mlpackage`, etc.)
- `quantization_config.json` — per-layer bit map, group sizes, method used
- `tokenizer.json` + `tokenizer_config.json` (copied from source)

### Config & Auth

- CLI flags only (v1). No config file.
- `--quant auto` is the default.
- Quality measurement (KL divergence) is on by default.
- HuggingFace auth token from `~/.huggingface/token` or `HF_TOKEN` env var.

### Scripting Support

- `--yes` for non-interactive mode (skip all prompts)
- `--json-report` for machine-readable output
- Exit codes: 0 (success), 1 (conversion error), 2 (quality threshold exceeded), 3 (input error)
- Stderr for warnings/progress, stdout clean for piping when `--json-report` used
- Shell completions via `hf2q completions --shell {bash,zsh,fish}`

## Project Scoping & Development Strategy

### Philosophy: Build What We Want

No artificial MVP/growth/vision phasing. hf2q is a tool built for one user who knows exactly what he needs. The PRD describes the complete tool. Implementation is iterative — some features land before others — but the spec is the full vision.

### Implementation Order (Not Scope Cuts)

Features naturally sequence due to dependencies:

1. **Foundation:** Safetensors reading, config parsing, shard consolidation, MLX output, static quant (f16/q8/q4)
2. **Downloads:** HF Hub integration via `hf-hub` + `hf` CLI fallback
3. **Quality:** KL divergence, perplexity measurement, quality reporting
4. **Gold mode:** DWQ calibration, mixed-bit, `--sensitive-layers`
5. **Intelligence:** Hardware profiling, RuVector integration, auto mode
6. **CoreML:** CoreML output target via `coreml-native`
7. **Multi-platform:** NVIDIA (NVFP4, GPTQ, AWQ), AMD (ROCm), Intel (OpenVINO), GGUF

This is implementation sequencing, not scope reduction.

### Output Target Roadmap

| Target | Platform | Format | Status |
|---|---|---|---|
| MLX | macOS (Apple Silicon) | Safetensors + MLX config | v1 |
| CoreML | macOS (Apple Silicon) | `.mlpackage` / `.mlmodelc` | v1 |
| NVFP4 | Linux (NVIDIA) | NVIDIA FP4 | Planned |
| GPTQ | Linux (NVIDIA) | GPTQ safetensors | Planned |
| AWQ | Linux (NVIDIA) | AWQ safetensors | Planned |
| GGUF | Universal | GGUF (llama.cpp) | Planned |
| ROCm | Linux (AMD) | ROCm-optimized | Future |
| OpenVINO | Linux (Intel) | OpenVINO IR | Future |

### Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| `mlx-rs` missing needed ops | Blocks MLX output | Implement quant math in pure Rust; `mlx-rs` for inference passes only |
| RuVector recommendations poor initially | Bad auto mode defaults | Heuristic-based auto first; RuVector replaces heuristics. Manual override always available. |
| KL divergence computation too slow | Slow conversions | On by default; `--skip-quality` available. Optimize with sampling. |
| New HF architectures break parser | Broken conversions | f16 fallback per-layer; known-good architecture registry |
| 48GB+ models OOM during conversion | Unusable for large models | Streaming shard processing, mmap I/O, configurable parallelism from day one |
| RuVector database corruption | Auto mode fails | Fall back to heuristic auto; database is optional, not required |

### Dependencies

```toml
[dependencies]
# Core ML pipeline
mlx-rs = { version = "0.25", features = ["safetensors", "metal", "accelerate"] }
safetensors = "0.7"
half = "2.4"                          # bf16 ↔ f16 conversion

# HuggingFace integration
hf-hub = "0.5"                        # Model downloads (primary)

# Self-learning intelligence
ruvector-core = "2.1"                 # Self-optimizing vector DB for auto mode

# CoreML output target
coreml-native = "0.2"                 # CoreML compilation + inference validation

# CLI framework
clap = { version = "4.5", features = ["derive"] }
clap_complete = "4.5"                 # Shell completions

# Performance
rayon = "1.10"                        # Parallel shard processing
memmap2 = "0.9"                       # Memory-mapped I/O for large models

# UX
indicatif = "0.17"                    # Progress bars
console = "0.15"                      # Terminal colors + formatting

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"                      # Config parsing, JSON report output

# Error handling
anyhow = "1"                          # Application-level errors
thiserror = "2"                       # Library-level typed errors

# System info
sysinfo = "0.32"                      # Hardware profiling (chip, memory)
```

```toml
[dev-dependencies]
assert_cmd = "2"                      # CLI integration tests
predicates = "3"                      # Test assertions
tempfile = "3"                        # Temp dirs for test conversions
criterion = "0.5"                     # Benchmarks
```

**MSRV:** 1.81.0 (required by `mlx-rs`)

**Platform:** macOS (Apple Silicon) primary. Linux compiles but MLX/CoreML targets return `UnsupportedPlatform`. Future NVIDIA/AMD/Intel backends behind feature flags.

## Functional Requirements

### Model Input & Discovery

- **FR1:** User can provide a local safetensors directory as conversion input
- **FR2:** User can provide a HuggingFace repo ID to download and convert in one command
- **FR3:** System downloads from HuggingFace Hub via `hf-hub` crate with automatic fallback to `hf` CLI
- **FR4:** System reads HuggingFace auth tokens from `~/.huggingface/token` or `HF_TOKEN` env var
- **FR5:** User can inspect a model's metadata before converting (`info` subcommand) — architecture, parameter count, layer types, expert count, dtype, shard count
- **FR6:** System parses arbitrary HuggingFace model architectures via config.json without hardcoded model-family support

### Quantization & Conversion

- **FR7:** User can convert models to MLX-compatible safetensors format
- **FR8:** User can convert models to CoreML format (`.mlpackage` / `.mlmodelc`)
- **FR9:** User can select a quantization method: f16, q8, q4, q2, q4-mxfp, mixed-2-6, mixed-3-6, mixed-4-6, dwq-mixed-4-6
- **FR10:** User can specify custom quantization parameters (bit width 2-8, group size 32/64/128)
- **FR11:** User can run DWQ calibration with configurable sample count
- **FR12:** User can designate sensitive layers to protect at higher precision (`--sensitive-layers`)
- **FR13:** System consolidates N input shards into 4 output shards
- **FR14:** System converts bf16 tensors to f16 during processing
- **FR15:** System falls back to f16 passthrough for unknown layer types with clear warnings

### Quality Measurement

- **FR16:** System measures KL divergence between pre-quant and post-quant output distributions (per-layer and overall) by default
- **FR17:** System measures perplexity delta (pre-quant vs post-quant) by default
- **FR18:** System computes cosine similarity of layer activations pre/post quantization for debugging
- **FR19:** System generates `quantization_config.json` with per-layer bit allocation, group sizes, and method
- **FR20:** User can skip quality measurement with `--skip-quality`
- **FR21:** System reports quality metrics in terminal output and JSON report

### Hardware Intelligence

- **FR22:** System detects and profiles hardware (chip model, unified memory, compute units)
- **FR23:** System fingerprints model architecture (type, param count, layer count, expert count, attention types)
- **FR24:** System determines optimal quantization settings based on hardware + model fingerprint
- **FR25:** User can run `--quant auto` (the default) to let the system choose optimal settings

### Self-Learning (RuVector Integration)

- **FR26:** System stores conversion results (hardware + model + quant config + quality metrics) in local RuVector database
- **FR27:** System queries RuVector for known-good configurations when auto mode encounters a previously-seen combination
- **FR28:** System improves recommendations over time as more conversions are performed
- **FR29:** System re-calibrates recommendations when hf2q version changes

### Output & Reporting

- **FR30:** System produces format-specific output directory with weights, config, tokenizer, and quantization metadata
- **FR31:** User can specify a custom output directory
- **FR32:** User can generate structured JSON report (`--json-report`)
- **FR33:** System displays progress bars for all long-running phases
- **FR34:** System displays conversion summary with quality metrics on completion

### Robustness & Error Handling

- **FR35:** System processes large models (48GB+) via streaming/mmap I/O without loading all shards simultaneously
- **FR36:** System warns and requests confirmation for unknown layer types before fallback
- **FR37:** System respects memory bounds during parallel processing

### Scripting & Automation

- **FR38:** User can run in non-interactive mode (`--yes`)
- **FR39:** System returns structured exit codes: 0 success, 1 conversion error, 2 quality exceeded, 3 input error
- **FR40:** User can generate shell completions for bash, zsh, fish

### Future Output Targets (Planned)

- **FR41:** User can convert to NVFP4 format (NVIDIA)
- **FR42:** User can convert to GPTQ format
- **FR43:** User can convert to AWQ format
- **FR44:** User can convert to GGUF format (llama.cpp)
- **FR45:** User can convert to AMD ROCm-optimized format
- **FR46:** User can convert to Intel OpenVINO format

## Non-Functional Requirements

### Performance

- **NFR1:** 26B model conversion (32 shards, ~48GB) completes without OOM on 128GB unified memory
- **NFR2:** Conversion completes within 2x the time of equivalent Python `mlx-lm` conversion
- **NFR3:** Shard loading saturates available disk bandwidth via parallel I/O (rayon)
- **NFR4:** Memory usage during conversion does not exceed 2x the size of a single shard
- **NFR5:** Progress reporting updates at least once per second

### Reliability

- **NFR6:** Conversion never produces silently corrupted output — errors result in clear messages and non-zero exit codes
- **NFR7:** Interrupted conversions (Ctrl+C) clean up partial output directories
- **NFR8:** RuVector database corruption does not prevent hf2q from functioning — falls back to heuristic auto

### Compatibility

- **NFR9:** MLX output loadable by `mlx-server` and `mlx-lm` without modification
- **NFR10:** CoreML output compilable by Apple's CoreML compiler and loadable by `coreml-native`
- **NFR11:** Compiles and runs on macOS 14+ with Apple Silicon (M1 through M5+)
- **NFR12:** MSRV: Rust 1.81.0
- **NFR13:** Single static binary — no runtime dependency on Python, Swift, or other runtimes

### Maintainability

- **NFR14:** New output format backends (GGUF, NVFP4, etc.) do not require modifying core conversion pipeline — clean `input → IR → backend` separation
- **NFR15:** New HuggingFace architectures using standard layer types require no code changes — config.json-driven parsing
