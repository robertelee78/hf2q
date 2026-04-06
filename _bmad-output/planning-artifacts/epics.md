---
stepsCompleted: [1, 2, 3, 4]
status: 'complete'
completedAt: '2026-04-06'
inputDocuments: ['prd.md', 'architecture.md']
---

# hf2q - Epic Breakdown

## Engineering Philosophy

> DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.

## Overview

This document provides the complete epic and story breakdown for hf2q, decomposing the requirements from the PRD and Architecture into implementable stories.

## Requirements Inventory

### Functional Requirements

- FR1: User can provide a local safetensors directory as conversion input
- FR2: User can provide a HuggingFace repo ID to download and convert in one command
- FR3: System downloads from HuggingFace Hub via `hf-hub` crate with automatic fallback to `hf` CLI
- FR4: System reads HuggingFace auth tokens from `~/.huggingface/token` or `HF_TOKEN` env var
- FR5: User can inspect a model's metadata before converting (`info` subcommand)
- FR6: System parses arbitrary HuggingFace model architectures via config.json without hardcoded model-family support
- FR7: User can convert models to MLX-compatible safetensors format
- FR8: User can convert models to CoreML format (`.mlpackage` / `.mlmodelc`)
- FR9: User can select a quantization method: f16, q8, q4, q2, q4-mxfp, mixed-2-6, mixed-3-6, mixed-4-6, dwq-mixed-4-6
- FR10: User can specify custom quantization parameters (bit width 2-8, group size 32/64/128)
- FR11: User can run DWQ calibration with configurable sample count
- FR12: User can designate sensitive layers to protect at higher precision (`--sensitive-layers`)
- FR13: System consolidates N input shards into 4 output shards
- FR14: System converts bf16 tensors to f16 during processing
- FR15: System falls back to f16 passthrough for unknown layer types with clear warnings (requires explicit `--unsupported-layers=passthrough`)
- FR16: System measures KL divergence between pre-quant and post-quant output distributions (per-layer and overall) by default
- FR17: System measures perplexity delta (pre-quant vs post-quant) by default
- FR18: System computes cosine similarity of layer activations pre/post quantization for debugging
- FR19: System generates `quantization_config.json` with per-layer bit allocation, group sizes, and method
- FR20: User can skip quality measurement with `--skip-quality`
- FR21: System reports quality metrics in terminal output and JSON report
- FR22: System detects and profiles hardware (chip model, unified memory, compute units)
- FR23: System fingerprints model architecture (type, param count, layer count, expert count, attention types)
- FR24: System determines optimal quantization settings based on hardware + model fingerprint
- FR25: User can run `--quant auto` (the default) to let the system choose optimal settings
- FR26: System stores conversion results in local RuVector database
- FR27: System queries RuVector for known-good configurations
- FR28: System improves recommendations over time as more conversions are performed
- FR29: System re-calibrates recommendations when hf2q version changes
- FR30: System produces format-specific output directory with weights, config, tokenizer, and quantization metadata
- FR31: User can specify a custom output directory
- FR32: User can generate structured JSON report (`--json-report`)
- FR33: System displays progress bars for all long-running phases
- FR34: System displays conversion summary with quality metrics on completion
- FR35: System processes large models (48GB+) via streaming/mmap I/O without loading all shards simultaneously
- FR36: System warns and requests confirmation for unknown layer types before fallback
- FR37: System respects memory bounds during parallel processing
- FR38: User can run in non-interactive mode (`--yes`)
- FR39: System returns structured exit codes: 0 success, 1 conversion error, 2 quality exceeded, 3 input error
- FR40: User can generate shell completions for bash, zsh, fish
- FR41: User can convert to NVFP4 format (NVIDIA) [future]
- FR42: User can convert to GPTQ format [future]
- FR43: User can convert to AWQ format [future]
- FR44: User can convert to GGUF format (llama.cpp) [future]
- FR45: User can convert to AMD ROCm-optimized format [future]
- FR46: User can convert to Intel OpenVINO format [future]

### Non-Functional Requirements

- NFR1: 26B model conversion completes without OOM on 128GB unified memory
- NFR2: Conversion completes within 2x the time of equivalent Python `mlx-lm` conversion
- NFR3: Shard loading saturates available disk bandwidth via parallel I/O (rayon)
- NFR4: Memory usage during conversion does not exceed 2x the size of a single shard
- NFR5: Progress reporting updates at least once per second
- NFR6: Conversion never produces silently corrupted output
- NFR7: Interrupted conversions (Ctrl+C) clean up partial output directories
- NFR8: RuVector database corruption does not prevent hf2q from functioning (falls back to heuristic auto)
- NFR9: MLX output loadable by `mlx-server` and `mlx-lm` without modification
- NFR10: CoreML output compilable by Apple's CoreML compiler and loadable by `coreml-native`
- NFR11: Compiles and runs on macOS 14+ with Apple Silicon (M1 through M5+)
- NFR12: MSRV: Rust 1.81.0
- NFR13: Single static binary — no runtime dependency on Python, Swift, or other runtimes
- NFR14: New output format backends do not require modifying core conversion pipeline
- NFR15: New HF architectures using standard layer types require no code changes

### Additional Requirements (from Architecture)

- AR1: Hybrid streaming pipeline — eager metadata, lazy tensor data via mmap, per-layer bounded memory
- AR2: Shared InferenceRunner wrapping `mlx-rs` — used by both DWQ calibration and quality measurement
- AR3: Pre-flight checks must run before any expensive work — validate all inputs, check RuVector, estimate memory
- AR4: RuVector is required infrastructure — unavailable = error, no data = warning + heuristics
- AR5: No implicit fallbacks — unknown layer types require explicit `--unsupported-layers=passthrough`
- AR6: `--dry-run` flag runs preflight + auto resolution + backend validation without converting
- AR7: `hf2q doctor` subcommand diagnoses RuVector, hardware detection, mlx-rs, disk space
- AR8: All traits are `Send + Sync` for future rayon parallelism
- AR9: `tracing` for structured logging, no `println!` for diagnostics
- AR10: Ctrl+C signal handler must clean up partial output directories
- AR11: `cargo init --name hf2q` — single crate, no workspace

### UX Design Requirements

N/A — CLI tool, no UX design document.

### FR Coverage Map

| FR | Epic | Description |
|---|---|---|
| FR1 | Epic 1 | Local safetensors input |
| FR2 | Epic 3 | HF repo download |
| FR3 | Epic 3 | hf-hub + hf CLI fallback |
| FR4 | Epic 3 | Auth token resolution |
| FR5 | Epic 1 | `info` subcommand |
| FR6 | Epic 1 | Arbitrary architecture parsing |
| FR7 | Epic 1 | MLX output |
| FR8 | Epic 8 | CoreML output |
| FR9 | Epic 1 (static) / Epic 5 (DWQ, mixed) | Quantization methods |
| FR10 | Epic 5 | Custom quant params |
| FR11 | Epic 5 | DWQ calibration |
| FR12 | Epic 5 | Sensitive layers |
| FR13 | Epic 1 | Shard consolidation |
| FR14 | Epic 1 | bf16→f16 conversion |
| FR15 | Epic 2 | Unknown layer handling |
| FR16 | Epic 4 | KL divergence |
| FR17 | Epic 4 | Perplexity delta |
| FR18 | Epic 4 | Cosine similarity |
| FR19 | Epic 4 | quantization_config.json |
| FR20 | Epic 4 | --skip-quality |
| FR21 | Epic 4 | Quality reporting |
| FR22 | Epic 6 | Hardware profiling |
| FR23 | Epic 6 | Model fingerprinting |
| FR24 | Epic 6 | Optimal settings |
| FR25 | Epic 6 | --quant auto |
| FR26 | Epic 7 | RuVector store |
| FR27 | Epic 7 | RuVector query |
| FR28 | Epic 7 | Learning improvement |
| FR29 | Epic 7 | Version re-calibration |
| FR30 | Epic 1 | Output directory |
| FR31 | Epic 1 | Custom output dir |
| FR32 | Epic 4 | JSON report |
| FR33 | Epic 1 | Progress bars |
| FR34 | Epic 1 | Conversion summary |
| FR35 | Epic 1 | Streaming mmap I/O |
| FR36 | Epic 2 | Unknown layer confirmation |
| FR37 | Epic 1 | Memory-bounded parallelism |
| FR38 | Epic 2 | --yes non-interactive |
| FR39 | Epic 2 | Exit codes |
| FR40 | Epic 9 | Shell completions |
| FR41-46 | Future | NVFP4, GPTQ, AWQ, GGUF, ROCm, OpenVINO |

## Epic List

### Epic 1: Core Conversion Pipeline
User can convert a local safetensors model to MLX format with static quantization (f16/q8/q4). The fundamental "it works" moment.
**FRs covered:** FR1, FR5, FR6, FR7, FR9 (static), FR13, FR14, FR30, FR31, FR33, FR34, FR35, FR37

### Epic 2: Fail-Early Validation & Error Excellence
User gets clear, actionable errors before any expensive work begins. No silent failures, no wasted time.
**FRs covered:** FR15, FR36, FR38, FR39

### Epic 3: HuggingFace Hub Integration
User can download and convert models directly from HuggingFace with one command.
**FRs covered:** FR2, FR3, FR4

### Epic 4: Quality Measurement
User gets KL divergence, perplexity delta, and quality reports proving conversion preserved model quality.
**FRs covered:** FR16, FR17, FR18, FR19, FR20, FR21, FR32

### Epic 5: Gold Mode — DWQ & Sensitive Layers
User can run calibrated DWQ quantization with per-layer bit allocation and protect modified layers.
**FRs covered:** FR9 (DWQ, mixed-bit), FR10, FR11, FR12

### Epic 6: Hardware Intelligence & Auto Mode
User runs `--quant auto` (the default) and the tool picks optimal settings for their hardware and model.
**FRs covered:** FR22, FR23, FR24, FR25

### Epic 7: Self-Learning via RuVector
Every conversion makes the tool smarter. Auto mode improves over time from stored results.
**FRs covered:** FR26, FR27, FR28, FR29

### Epic 8: CoreML Output Backend
User can convert dense models and vision encoders to CoreML format for ANE acceleration.
**FRs covered:** FR8

### Epic 9: CLI Polish & Automation
Shell completions, dry-run, and full scripting support for CI pipelines.
**FRs covered:** FR40

## Epic 1: Core Conversion Pipeline

User can convert a local safetensors model to MLX format with static quantization (f16/q8/q4). The fundamental "it works" moment.

### Story 1.1: Project Initialization & CLI Skeleton

As a developer,
I want to initialize the hf2q crate with clap-based CLI parsing,
So that the project has a working binary with `convert`, `info`, `doctor`, and `completions` subcommand stubs.

**Acceptance Criteria:**

**Given** a fresh checkout of the hf2q repo
**When** I run `cargo build`
**Then** the project compiles with zero warnings
**And** `hf2q --help` displays subcommands: convert, info, doctor, completions
**And** `hf2q convert --help` displays all flags from the PRD command structure
**And** `hf2q --version` displays the crate version
**And** all clap structs use the derive API
**And** `ConvertConfig` struct is defined with all fields matching PRD spec
**And** `tracing-subscriber` is initialized in main.rs

### Story 1.2: Config Parser & Model Metadata

As a user,
I want hf2q to read a HuggingFace model's config.json and extract architecture metadata,
So that I can inspect models and the tool understands what it's converting.

**Acceptance Criteria:**

**Given** a directory containing a valid HuggingFace config.json
**When** I run `hf2q info --input ./model`
**Then** the tool displays: architecture name, parameter count, hidden size, layer count, layer types, number of attention heads, vocab size, dtype, and shard count
**And** MoE models display expert count and top-k
**And** the parser handles arbitrary config.json fields without hardcoded model families (FR6)
**And** missing optional fields produce warnings, not errors
**And** the output is formatted for human readability in terminal

### Story 1.3: Streaming Safetensors Reader with Mmap

As a user,
I want hf2q to read safetensors shards via memory-mapped I/O,
So that large models (48GB+) can be processed without loading everything into memory.

**Acceptance Criteria:**

**Given** a directory containing N safetensors shards and a `model.safetensors.index.json`
**When** the reader opens the shards
**Then** it produces a `TensorMap` with lazy mmap'd `TensorRef` entries for every tensor
**And** memory usage does not exceed 2x the size of a single shard (NFR4)
**And** single-shard models (no index.json) are also supported
**And** progress is reported via `ProgressReporter` during shard discovery
**And** corrupted or unreadable shards produce clear errors with the shard filename

### Story 1.4: Intermediate Representation & bf16→f16 Conversion

As a developer,
I want a clean IR that represents model weights and metadata,
So that quantization and backends have a consistent data contract.

**Acceptance Criteria:**

**Given** a `TensorMap` from the shard reader
**When** the IR is constructed
**Then** `TensorRef` provides: name, shape, dtype, and lazy access to data bytes
**And** bf16 tensors can be converted to f16 on access (FR14)
**And** `ModelMetadata` includes architecture info from config parser
**And** `QuantizedModel` wraps the IR after quantization with per-tensor quant metadata
**And** all IR types implement `Send + Sync`

### Story 1.5: Static Quantizer (f16, q8, q4, q2)

As a user,
I want to quantize model weights using static round-to-nearest quantization,
So that I can reduce model size with fast, simple compression.

**Acceptance Criteria:**

**Given** a `TensorMap` with f16/bf16 tensors
**When** I run `hf2q convert --input ./model --format mlx --quant q4`
**Then** all weight tensors are quantized to 4-bit with group_size=64 (default)
**And** f16 mode performs bf16→f16 conversion only (lossless)
**And** q8 produces 8-bit quantized weights
**And** q2 produces 2-bit quantized weights
**And** non-weight tensors (layer norms, biases) are preserved at full precision
**And** the `Quantizer` trait is implemented with `name()` and `requires_calibration()` returning false

### Story 1.6: MLX Output Backend

As a user,
I want the converted model written as MLX-compatible safetensors with proper config,
So that I can load it in mlx-server or mlx-lm without modification.

**Acceptance Criteria:**

**Given** a `QuantizedModel` from the quantization engine
**When** the MLX backend writes output
**Then** the output directory contains consolidated safetensors shards (N input → 4 output) (FR13)
**And** an MLX-compatible `config.json` is generated with correct architecture fields
**And** `tokenizer.json` and `tokenizer_config.json` are copied from source
**And** `quantization_config.json` is written with per-tensor quant method and group size (FR19)
**And** `OutputBackend::validate()` runs before writing and catches format issues
**And** the output is loadable by `mlx-server` without modification (NFR9)
**And** `--output` flag controls output directory (FR31)

### Story 1.7: Progress Reporting & Conversion Summary

As a user,
I want progress bars during conversion and a summary when complete,
So that I know what's happening and can verify the result.

**Acceptance Criteria:**

**Given** a conversion in progress
**When** each phase runs (shard loading, quantization, writing)
**Then** a progress bar updates at least once per second (NFR5)
**And** phase names are clearly displayed (e.g., "Reading shards", "Quantizing", "Writing MLX output")
**And** on completion, a summary displays: input model info, quant method, output size, output path, elapsed time
**And** all progress uses `ProgressReporter` — no direct indicatif calls elsewhere
**And** rayon parallelism respects memory bounds during shard processing (FR37)

## Epic 2: Fail-Early Validation & Error Excellence

User gets clear, actionable errors before any expensive work begins. No silent failures, no wasted time.

### Story 2.1: Pre-flight Validation Engine

As a user,
I want hf2q to validate all inputs before starting conversion,
So that I never waste 20 minutes on a conversion that was doomed to fail.

**Acceptance Criteria:**

**Given** a `ConvertConfig` and `ModelMetadata`
**When** `preflight::validate()` runs
**Then** it checks: all layer types supported for chosen quant method, output format compatible with model architecture, `--sensitive-layers` range valid for model layer count, estimated output size fits available disk space
**And** any failure produces a clear error with actionable guidance
**And** preflight completes in under 5 seconds for any model
**And** all checks run before any shard reading or quantization begins

### Story 2.2: Unknown Layer Handling with Explicit Opt-In

As a user,
I want hf2q to refuse to silently downgrade unknown layers,
So that I always get exactly what I asked for or a clear explanation of why not.

**Acceptance Criteria:**

**Given** a model with layer types not supported by the chosen quantizer
**When** conversion is attempted without `--unsupported-layers=passthrough`
**Then** hf2q errors with: specific unknown layer names, which layer indices are affected, available options
**And** hf2q refuses to proceed
**When** conversion is attempted with `--unsupported-layers=passthrough`
**Then** unsupported layers are converted at f16, supported layers at requested quant
**And** a warning lists every layer that was passed through at f16
**And** the `quantization_config.json` reflects the actual per-layer quant applied

### Story 2.3: Exit Codes & Non-Interactive Mode

As a CI pipeline operator,
I want structured exit codes and a non-interactive mode,
So that I can script hf2q reliably in automated workflows.

**Acceptance Criteria:**

**Given** any hf2q invocation
**When** it completes
**Then** exit code 0 = success, 1 = conversion error, 2 = quality threshold exceeded, 3 = input error
**And** `--yes` flag skips all confirmation prompts
**And** without `--yes`, prompts timeout after 30 seconds in non-TTY environments with exit code 3
**And** stderr contains all warnings and progress, stdout is clean when `--json-report` is used

### Story 2.4: Ctrl+C Signal Handler & Cleanup

As a user,
I want interrupted conversions to clean up after themselves,
So that I don't have corrupted partial output directories.

**Acceptance Criteria:**

**Given** a conversion in progress
**When** the user sends SIGINT (Ctrl+C)
**Then** the tool catches the signal, stops work, removes the partial output directory, and prints "Conversion interrupted. Partial output cleaned up."
**And** exit code is 1
**And** the original input files are never modified
**And** if the output directory existed before conversion, it is NOT deleted — only directories created by this conversion are removed

## Epic 3: HuggingFace Hub Integration

User can download and convert models directly from HuggingFace with one command.

### Story 3.1: HF Hub Download via hf-hub Crate

As a user,
I want to specify a HuggingFace repo ID and have hf2q download the model,
So that I can convert models without manually downloading them first.

**Acceptance Criteria:**

**Given** `hf2q convert --repo meta-llama/Llama-3.1-8B-Instruct --format mlx --quant q4`
**When** the tool starts
**Then** it downloads all safetensors shards, config.json, tokenizer.json, and tokenizer_config.json from HuggingFace Hub via the `hf-hub` crate
**And** progress bars show download progress per file
**And** downloaded files are cached in the standard `hf-hub` cache directory
**And** subsequent runs with the same `--repo` use the cache (no re-download)
**And** `--input` and `--repo` are mutually exclusive — providing both is an error

### Story 3.2: Auth Token Resolution & hf CLI Fallback

As a user,
I want hf2q to handle gated models and fall back to the hf CLI if needed,
So that I can access private or gated repos seamlessly.

**Acceptance Criteria:**

**Given** a gated or private HuggingFace repo
**When** `--repo` is specified
**Then** the tool reads auth token from `~/.huggingface/token` or `HF_TOKEN` env var (FR4)
**And** if `hf-hub` crate fails (network error, auth error, unsupported feature)
**Then** the tool checks if `hf` CLI is on `$PATH` and retries via `hf download`
**And** if neither succeeds, error says exactly what failed and suggests: check token, check network, install hf CLI
**And** `hf2q info --repo org/model` also works for remote inspection

## Epic 4: Quality Measurement

User gets KL divergence, perplexity delta, and quality reports proving conversion preserved model quality.

### Story 4.1: Inference Runner (mlx-rs)

As a developer,
I want a shared InferenceRunner that wraps mlx-rs for forward passes,
So that both quality measurement and DWQ calibration use the same inference engine.

**Acceptance Criteria:**

**Given** a `TensorMap` representing model weights
**When** the `MlxRunner` is initialized
**Then** it loads tensors into mlx-rs arrays and can run forward passes on sample inputs
**And** the `InferenceRunner` trait defines: `load()`, `forward()`, `logits()` methods
**And** `StubRunner` returns `UnsupportedPlatform` error on non-Apple targets
**And** no module outside `inference/` imports `mlx_rs` directly
**And** the runner respects memory bounds — does not duplicate model weights in memory

### Story 4.2: KL Divergence Measurement

As a user,
I want per-layer and overall KL divergence between pre-quant and post-quant models,
So that I can quantify exactly how much information was lost.

**Acceptance Criteria:**

**Given** a completed quantization with both original and quantized weights available
**When** quality measurement runs (default, unless `--skip-quality`)
**Then** the tool computes KL divergence between original and quantized logit distributions on a calibration sample
**And** per-layer KL divergence is reported for every transformer layer
**And** overall KL divergence is reported as a single number
**And** results are included in terminal summary and JSON report
**And** `quantization_config.json` includes per-layer KL values

### Story 4.3: Perplexity Delta & Cosine Similarity

As a user,
I want perplexity delta and cosine similarity metrics,
So that I have multiple quality signals beyond just KL divergence.

**Acceptance Criteria:**

**Given** quality measurement is running
**When** perplexity is computed
**Then** it reports: pre-quant perplexity, post-quant perplexity, and delta
**And** cosine similarity is computed per-layer between original and quantized activations (FR18)
**And** both metrics are included in terminal summary and JSON report
**And** `--skip-quality` skips all quality measurement (FR20)
**And** progress bar shows quality measurement progress ("Measuring quality: layer 5/30")

### Story 4.4: JSON Report Generation

As a CI pipeline operator,
I want a structured JSON report of every conversion,
So that I can automate quality gates and catalog results.

**Acceptance Criteria:**

**Given** `--json-report` flag is set
**When** conversion completes
**Then** a JSON file is written with: input path/repo, output path, model metadata, hardware profile, quant config, per-layer bit allocation, KL divergence (per-layer + overall), perplexity delta, cosine similarity, output file sizes, elapsed time per phase
**And** JSON is written to stdout when combined with `--yes`
**And** the JSON schema is stable and documented in a comment at the top of `report.rs`

## Epic 5: Gold Mode — DWQ & Sensitive Layers

User can run calibrated DWQ quantization with per-layer bit allocation and protect modified layers.

### Story 5.1: Mixed-Bit Quantizer with Sensitive Layers

As a user,
I want to apply different bit widths to different layers and protect sensitive layers,
So that my uncensored/steered model modifications survive quantization.

**Acceptance Criteria:**

**Given** `--quant mixed-4-6 --sensitive-layers 13-24`
**When** quantization runs
**Then** layers 13-24 are quantized at 6-bit, all other layers at 4-bit
**And** `--quant mixed-2-6` and `--quant mixed-3-6` work similarly
**And** `--sensitive-layers` accepts ranges (13-24), comma-separated (1,5,13-24), or "first-last" syntax
**And** invalid layer ranges are caught by preflight
**And** `quantization_config.json` shows the exact bit width per layer
**And** custom `--bits N --group-size G` applies uniformly when no `--sensitive-layers`

### Story 5.2: DWQ Calibration Engine

As a user,
I want Distilled Weight Quantization that calibrates against the original model,
So that I get the best possible quantization quality — gold standard.

**Acceptance Criteria:**

**Given** `--quant dwq-mixed-4-6 --calibration-samples 1024 --sensitive-layers 13-24`
**When** DWQ runs
**Then** it loads calibration samples, runs forward passes through the original model via InferenceRunner, quantizes weights, runs forward passes through quantized model, measures KL divergence loss, and adjusts non-quantized parameters to minimize loss
**And** `--calibration-samples` controls sample count (default 1024)
**And** progress shows calibration phase with per-layer progress
**And** DWQ respects `--sensitive-layers` — protected layers get 6-bit, others get 4-bit
**And** `Quantizer::requires_calibration()` returns true for DWQ

## Epic 6: Hardware Intelligence & Auto Mode

User runs `--quant auto` (the default) and the tool picks optimal settings.

### Story 6.1: Hardware Profiler & Model Fingerprinting

As a developer,
I want the system to detect hardware specs and fingerprint model architectures,
So that auto mode can make informed decisions.

**Acceptance Criteria:**

**Given** any machine running hf2q
**When** `HardwareProfiler::detect()` runs
**Then** it returns: chip model (e.g., "Apple M5 Max"), total unified memory, available memory, core counts
**And** `ModelFingerprint::from_metadata()` returns: architecture name, total params, layer count, expert count, attention types, hidden size, dtype
**And** both produce stable, hashable identifiers for RuVector lookups
**And** hardware detection works on all macOS 14+ Apple Silicon

### Story 6.2: Auto Mode Resolver with Heuristics

As a user,
I want `--quant auto` to choose optimal settings for my hardware and model,
So that I never have to think about quantization settings unless I want to.

**Acceptance Criteria:**

**Given** `hf2q convert --input ./model --format mlx` (no --quant, defaults to auto)
**When** auto mode resolves
**Then** it queries RuVector for known-good configs (uses heuristics until RuVector integrated)
**And** heuristic rules: model fits at f16 → f16; fits at q8 → mixed-4-6; tight → q4; very tight → q2
**And** heuristic confidence is logged via tracing
**And** any explicit `--quant` flag overrides auto mode entirely
**And** the resolved config is displayed before conversion begins
**And** `--dry-run` shows the resolved config and exits without converting

## Epic 7: Self-Learning via RuVector

Every conversion makes the tool smarter. Auto mode improves over time.

### Story 7.1: RuVector Database Initialization & Doctor

As a user,
I want hf2q to maintain a local RuVector database and diagnose its health,
So that the learning system is always operational.

**Acceptance Criteria:**

**Given** first run of hf2q on a machine
**When** RuVector is initialized
**Then** a local database is created at `~/.hf2q/ruvector/`
**And** `hf2q doctor` reports: RuVector status, database size, stored conversion count, mlx-rs Metal availability, hf CLI availability, disk space
**And** if RuVector is unavailable, `hf2q convert` errors with: "RuVector not accessible. Required to store learnings. Run `hf2q doctor` to diagnose."
**And** doctor provides specific remediation steps for each issue found

### Story 7.2: Conversion Result Storage & Retrieval

As a user,
I want every conversion to store its results and auto mode to query past results,
So that the tool improves recommendations over time.

**Acceptance Criteria:**

**Given** a completed conversion with quality metrics
**When** the pipeline reaches the store phase
**Then** it writes to RuVector: hardware profile, model fingerprint, quant config, quality metrics, hf2q version, timestamp
**And** auto mode queries RuVector before heuristics: if exact match exists, use stored config
**And** if multiple results exist, prefer lowest KL divergence
**And** when hf2q version changes, stored results are flagged for re-calibration (FR29)
**And** RuVector operational but empty = warning + heuristics (not error)

## Epic 8: CoreML Output Backend

User can convert dense models and vision encoders to CoreML format.

### Story 8.1: CoreML Backend with Validation

As a user,
I want to convert models to CoreML format for Apple Neural Engine acceleration,
So that I can run dense models and vision encoders with maximum hardware utilization.

**Acceptance Criteria:**

**Given** `hf2q convert --input ./model --format coreml --quant f16`
**When** the CoreML backend runs
**Then** it produces a `.mlpackage` directory with the CoreML model spec
**And** `coreml-native::compile_model()` compiles to `.mlmodelc`
**And** `OutputBackend::validate()` checks: no MoE dynamic routing, all layers CoreML-compatible, estimated size fits ANE constraints
**And** output is loadable by `coreml-native` (NFR10)
**And** incompatible architectures produce a preflight error with clear explanation and suggestion to use `--format mlx`

## Epic 9: CLI Polish & Automation

Shell completions, dry-run, and full scripting polish.

### Story 9.1: Shell Completions & Dry-Run

As a user,
I want shell completions and a dry-run mode,
So that I can work efficiently and preview conversions before committing.

**Acceptance Criteria:**

**Given** `hf2q completions --shell zsh`
**When** the command runs
**Then** it outputs a zsh completion script to stdout
**And** bash and fish are also supported
**And** completions include all subcommands, flags, and enum values
**Given** `hf2q convert --input ./model --format mlx --dry-run`
**When** dry-run executes
**Then** it runs preflight, auto mode resolution, and backend validation
**And** prints: resolved quant config, estimated output size, estimated memory usage, any warnings
**And** exits with code 0 without writing any files
**And** `--dry-run` works with `--repo` (downloads metadata only, not full weights)
