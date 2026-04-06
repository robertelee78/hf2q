---
stepsCompleted: ['step-01-init', 'step-02-discovery', 'step-02b-vision', 'step-02c-executive-summary', 'step-03-success', 'step-04-journeys', 'step-05-domain', 'step-06-innovation', 'step-07-project-type']
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

## Executive Summary

**hf2q** is a pure Rust CLI tool that converts HuggingFace model weights to hardware-optimized formats — eliminating Python entirely from the model conversion pipeline. Initially targeting Apple Silicon (MLX, CoreML), with a roadmap to Linux/NVIDIA targets (NVFP4, GPTQ, AWQ, GGUF). It serves AI practitioners who work with open-weight models and want a single, Python-free tool for quantization and format conversion across platforms.

The tool operates at three tiers: a quick mode for fast static quantization, an expert gold mode with Distilled Weight Quantization (DWQ) and per-layer bit allocation, and a self-learning auto mode powered by RuVector that determines optimal conversion settings based on the user's hardware and model architecture. Auto mode improves with every conversion, building a knowledge base of `(hardware + model) → optimal config` mappings that can be shared across the community.

hf2q supports arbitrary HuggingFace model architectures (not just specific families), downloads models directly from HuggingFace Hub via the `hf-hub` crate (with `hf` CLI fallback), and produces consolidated, publication-quality output — including per-layer quantization configs and perplexity benchmarks.

### What Makes This Special

1. **First pure Rust HF model converter.** The entire ML ecosystem currently funnels through Python for model conversion. Even Rust-native inference tools like `mlx-server` require `pip install mlx-lm` to prepare models. hf2q closes that gap — zero Python, anywhere in the pipeline. macOS first, Linux next.

2. **Gold mode with `--sensitive-layers`.** DWQ calibration with per-layer bit allocation and explicit protection for ARA-steered or otherwise modified layers. This is purpose-built for practitioners who uncensor, fine-tune, or steer models and need quantization that respects their modifications. No other tool offers this.

3. **Self-learning auto mode via RuVector.** The tool profiles hardware, fingerprints the model architecture, and queries a self-optimizing vector database for the best conversion settings. Unknown combinations trigger a calibration sweep, learn, and store results. The tool gets smarter with every use — and community sharing means new model families bootstrap from collective knowledge.

## Project Classification

- **Project Type:** CLI Tool
- **Domain:** Scientific / ML Tooling
- **Complexity:** Medium — sophisticated quantization algorithms and hardware-aware optimization, but no regulatory or compliance overhead
- **Project Context:** Greenfield — first-of-its-kind tool, no existing codebase

## Success Criteria

### User Success

- **Zero-friction conversion:** A single `hf2q convert` command takes a HuggingFace model (local or remote) to a hardware-optimized format with no Python dependency, no environment setup, no version conflicts.
- **Confidence in output quality:** Every conversion produces a perplexity benchmark (pre/post delta) and optional inference sanity check so the user knows the output is correct — not just "it ran."
- **Auto mode delivers:** Running `--quant auto` produces results equal to or better than what the user would achieve manually tuning settings. The tool remembers what worked.
- **Gold mode respects modifications:** Practitioners who uncensor, steer, or fine-tune models can protect their modified layers during quantization with `--sensitive-layers`.

### Business Success

- **Genuinely useful to Robert:** The tool reliably handles every model conversion needed in an uncensoring/optimization workflow, on every model downloaded. If it solves this one user's real problem every time, it's successful.
- **Community validation (bonus, not goal):** If others discover it and find it useful — great. But the tool is built for utility, not vanity metrics.
- **Foundation for multi-platform:** Architecture supports adding Linux/NVIDIA output targets (NVFP4, GPTQ, AWQ, GGUF) without rewriting the core pipeline.

### Technical Success

- **Arbitrary HF architectures:** Handles any model with a config.json and safetensors weights — not hardcoded to specific model families.
- **Output parity:** Converted models produce equivalent inference quality to Python `mlx-lm` conversions (within measurable perplexity tolerance).
- **Performance:** Conversion speed is competitive with or faster than Python tooling, leveraging Rust + rayon parallelism on 48GB+ weight files.
- **RuVector integration:** Auto mode learns from conversions and improves recommendations over time. Knowledge persists across sessions.

### Measurable Outcomes

- Perplexity delta for DWQ mixed-4-6 gold mode: < 0.3 vs f16 baseline
- Conversion of a 26B model (32 shards): completes without OOM on 128GB M5 Max
- Auto mode: after 5+ conversions, recommendations match or beat manual expert settings
- Zero Python processes spawned, ever

## Product Scope

### MVP - Minimum Viable Product

- CLI binary with `convert` and `info` subcommands
- Input: local safetensors directory
- Output: MLX format (safetensors + MLX-compatible config)
- Quantization: f16, q8, q4 (static, no calibration)
- Arbitrary HF architecture support via config.json parsing
- Shard consolidation (32 → 4 shards)
- Basic progress reporting

### Growth Features (Post-MVP)

- HuggingFace Hub download (`--repo`) via `hf-hub` crate + `hf` CLI fallback
- Gold mode: DWQ calibration, mixed-bit quantization, `--sensitive-layers`
- Auto mode: RuVector integration for hardware profiling + learned optimal configs
- CoreML output target for dense models and vision encoders
- Perplexity benchmarking (pre/post conversion)
- All quantization tiers: q2, q4-mxfp, mixed-2-6, mixed-3-6, mixed-4-6, custom `--bits`/`--group-size`
- `quantization_config.json` output with per-layer bit map

### Vision (Future)

- Linux support with NVIDIA output targets (NVFP4, GPTQ, AWQ)
- AMD ROCm-optimized output formats
- Intel OpenVINO output formats
- GGUF output target for llama.cpp ecosystem (universal, runs everywhere)
- Community sharing of RuVector conversion knowledge (opt-in)
- Inference sanity check built into conversion pipeline
- Architecture: clean `input → IR → output backend` design so adding new GPU vendor targets is just adding a backend

## User Journeys

### Journey 1: Robert — The Expert Practitioner (Gold Mode)

**Opening Scene:** Robert has just finished a dual-pass ARA uncensoring run on a new Gemma 4 27B model. The output is 32 safetensors shards sitting in `/opt/gemma4/`. He wants to run inference on his M5 Max — but the weights are raw bf16, unoptimized, and no MLX-compatible config exists.

**Rising Action:** He runs `hf2q info --input ./gemma4` to inspect the model — confirms 26B params, MoE 128x8, identifies the ARA-steered layers 13-24. Then:
```bash
hf2q convert --input ./gemma4 --format mlx --quant dwq-mixed-4-6 \
  --sensitive-layers 13-24 --calibration-samples 1024
```
The tool profiles his hardware (M5 Max, 128GB), loads shards in parallel via rayon, runs calibration passes, and applies 6-bit precision to the steered layers while compressing the rest to 4-bit. Progress bars show each phase.

**Climax:** Conversion completes. The output directory contains 4 consolidated shards, an MLX config, a `quantization_config.json` showing the per-layer bit map, and a perplexity benchmark: delta of 0.11 vs f16 baseline. The steered layers are intact.

**Resolution:** Robert loads the model in `mlx-server` and it runs perfectly — fast inference, uncensored behavior preserved, no Python touched at any point. He runs `hf2q` on the next model without thinking twice.

### Journey 2: Sam — The Curious Newcomer (Quick Mode)

**Opening Scene:** Sam just got an M4 MacBook Pro with 48GB. He's seen people running local LLMs and wants to try Llama 3.1 8B. He found hf2q mentioned in a Reddit thread that said "just install it with cargo and run one command."

**Rising Action:** Sam installs hf2q (`cargo install hf2q`) and runs:
```bash
hf2q convert --repo meta-llama/Llama-3.1-8B-Instruct --format mlx --quant q4
```
The tool downloads the model from HuggingFace Hub, shows a progress bar for the download, then converts. Sam doesn't know what group sizes or bit allocations are — and he doesn't need to. The defaults are sane.

**Climax:** Two minutes later, the output directory appears with clean MLX-format safetensors. Sam points `mlx-server` at it and gets a working chat interface. His first local LLM, no Python, no conda, no "which version of torch do I need."

**Resolution:** Sam starts experimenting with other models. He tries `--quant auto` on a larger model and hf2q picks optimal settings for his 48GB hardware automatically. He never opens a Python REPL.

### Journey 3: Robert — Error Recovery (Edge Case)

**Opening Scene:** Robert tries to convert a brand-new architecture he downloaded — some experimental model with a custom attention mechanism that just dropped on HuggingFace.

**Rising Action:** He runs `hf2q convert --input ./experimental-model --format mlx --quant q4`. The tool reads config.json, parses the architecture, but hits an unsupported layer type in the weight map.

**Climax:** Instead of silently producing garbage, hf2q reports:
```
Warning: Unknown layer type 'cross_mamba_attention' in layers 4-8
  → Falling back to f16 passthrough for these layers
  → Remaining layers quantized to q4 as requested
  → Output may be larger than expected (est. 12GB vs 8GB)
Proceed? [Y/n]
```

**Resolution:** Robert proceeds. The model works — slightly larger than a full q4, but functional. He files an issue or adds support for the new layer type himself. The tool degraded gracefully instead of failing.

### Journey 4: CI Pipeline — Batch Automation

**Opening Scene:** A small open-source project maintains a collection of optimized models. They want to automate: when a new model appears on HuggingFace, convert it and publish the optimized version.

**Rising Action:** The CI script runs:
```bash
hf2q convert --repo $MODEL_REPO --format mlx --quant auto \
  --output ./converted/$MODEL_NAME --json-report
```
The `--json-report` flag outputs structured results (perplexity delta, quant config, timings) for automated quality checks. No interactive prompts — fully scriptable.

**Climax:** The pipeline catches a model where auto mode's perplexity delta exceeds the threshold (>0.5). It flags the conversion for manual review instead of publishing.

**Resolution:** The project maintains a catalog of quality-verified optimized models, all converted without Python, all with published quant configs and benchmarks.

### Journey Requirements Summary

| Journey | Capabilities Revealed |
|---|---|
| Robert (Gold) | DWQ calibration, `--sensitive-layers`, per-layer bit map, perplexity benchmark, shard consolidation, hardware profiling |
| Sam (Quick) | HF Hub download, sane defaults, progress reporting, zero-config conversion, `--quant auto` |
| Robert (Edge Case) | Graceful fallback for unknown architectures, clear warnings, user confirmation prompts |
| CI Pipeline | `--json-report`, non-interactive mode, scriptable exit codes, quality thresholds |

## Domain-Specific Requirements

### Numerical Correctness & Quality Measurement

- **KL divergence** between base (pre-quant) and quantized output distributions is the primary quality metric. Must be computed per-layer and overall during gold mode conversions.
- **Perplexity delta** as a secondary quick-check metric for all conversion modes.
- **Cosine similarity** of layer activations pre/post quant available for debugging which layers degraded most.
- Per-layer KL divergence report must be included in gold mode output, especially for `--sensitive-layers` to prove steered layers maintained fidelity.
- Auto mode optimizes for minimum KL divergence, not just perplexity.

### Weight Format Compatibility

- Each output target (MLX, CoreML, GGUF, NVFP4) has specific tensor layout requirements, endianness, dtype expectations, and metadata schemas. Output must be validated against format spec before writing.
- MLX expects safetensors with specific config.json fields. CoreML expects `.mlpackage` protobuf spec. GGUF has its own header/tensor layout. Getting any wrong produces models that load but output garbage.

### Memory Safety During Conversion

- 48GB+ weight files must be processed via streaming/memory-mapped I/O — never load all shards simultaneously.
- Conversion must complete without OOM on machines where the model barely fits (e.g., 27B model on 64GB machine).
- Rayon parallelism must respect memory bounds, not just CPU cores.

### Model Architecture Drift

- HuggingFace model architectures evolve weekly — new attention mechanisms, MoE routing schemes, and layer types appear constantly.
- Unknown layer types must fall back to f16 passthrough with clear warnings, not hard failures.
- Architecture support should be data-driven (config.json parsing) rather than hardcoded per-model-family where possible.

### Risk Mitigations

| Risk | Mitigation |
|---|---|
| Silent quality degradation | Mandatory KL divergence / perplexity reporting; auto mode rejects conversions above threshold |
| Incompatible output format | Format-specific validation before writing final output |
| OOM during large model conversion | Streaming shard processing, memory-mapped I/O, configurable parallelism |
| Unsupported architecture | Graceful fallback with warnings, not crashes |
| RuVector stale recommendations | Recommendations include the hf2q version and date; auto mode re-calibrates when tool version changes |

## Innovation & Novel Patterns

### Detected Innovation Areas

1. **First pure Rust model conversion pipeline.** The ML model conversion step has been a Python monopoly. Every existing tool — `mlx-lm`, `coremltools`, `auto-gptq`, `bitsandbytes` — is Python. hf2q is the first to break this dependency entirely, enabled by the maturation of the Rust ML crate ecosystem (`mlx-rs`, `safetensors`, `hf-hub`).

2. **Self-learning conversion optimizer.** CLI tools don't learn. hf2q's integration of RuVector (self-optimizing vector database with SONA engine) means conversion settings improve over time. Each `(hardware + model + target) → optimal config` mapping is stored, learned from, and refined. This turns a static tool into an adaptive system.

3. **Modification-aware quantization.** The `--sensitive-layers` flag acknowledges that model practitioners modify weights (ARA uncensoring, LoRA fine-tuning, DPO steering) and those modifications must survive quantization. This is a novel concept — existing quantization tools treat all layers uniformly. hf2q lets users declare which layers matter most and allocates precision accordingly.

4. **KL divergence as optimization objective.** Auto mode doesn't just pick a quant preset — it minimizes measured KL divergence between pre-quant and post-quant distributions, with per-layer reporting. This is research-grade quality assurance packaged into a CLI flag (`--quant auto`).

### Market Context & Competitive Landscape

- **Python tools (mlx-lm, coremltools, auto-gptq):** Dominant but require Python ecosystem. No self-learning, no modification-aware quantization.
- **llama.cpp quantize:** C++ based, GGUF-only, no MLX/CoreML output, no adaptive optimization.
- **mlx-rs / mlx-server:** Rust inference but defers to Python for conversion — hf2q fills exactly this gap.
- **No direct competitor** exists in the pure Rust model conversion space.

### Validation Approach

- **Output parity testing:** Convert identical models with hf2q and Python `mlx-lm`, compare KL divergence and perplexity — must be equivalent or better.
- **Auto mode convergence:** Track whether recommendations improve over 10+ conversions on the same hardware.
- **Community validation:** Publish converted models with full quality reports; let the community verify inference quality.
- **Sensitive-layers validation:** Convert ARA-steered models with and without `--sensitive-layers`; measure KL divergence on steered layers specifically.

### Risk Mitigation

| Innovation Risk | Fallback |
|---|---|
| RuVector auto mode recommendations are poor initially | Always allow manual override; auto is the default but any explicit `--quant` flag overrides it |
| KL divergence computation is too slow for large models | Quality measurement is on by default; users can skip with `--skip-quality` if speed is critical |
| `mlx-rs` doesn't support needed quantization ops | Implement quantization math directly in Rust using `half` crate; `mlx-rs` is for inference passes only |
| New HF architectures break the generic parser | Graceful f16 fallback per-layer; maintain a known-good architecture registry |

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
│   └── --yes        # Non-interactive, skip confirmation prompts
├── info             # Inspect model before converting
│   └── --input      # Local safetensors directory or HF repo
└── completions      # Generate shell completions
    └── --shell      # bash, zsh, fish
```

### Output Formats

**Terminal output (default):**
- Progress bars for download, shard loading, quantization, and quality measurement phases
- Conversion summary with quant config, output size, and quality metrics (KL divergence, perplexity delta)
- Warnings for unknown layer types or fallback decisions
- Confirmation prompts for destructive/surprising actions

**JSON report (`--json-report`):**
- Machine-readable conversion results: input/output paths, quant config, per-layer bit allocation, quality metrics, timings, hardware profile
- Suitable for CI pipelines, quality gates, and automated cataloging

**Model output directory:**
- Consolidated safetensors shards (target: 4 shards)
- Format-specific config (MLX `config.json`, CoreML `.mlpackage`, etc.)
- `quantization_config.json` — per-layer bit map, group sizes, method used
- `tokenizer.json` + `tokenizer_config.json` (copied from source)

### Config Schema

- CLI flags only (v0.1). No config file.
- `--quant auto` is the default — the tool picks optimal settings based on hardware + model.
- Quality measurement (KL divergence) is on by default.
- HuggingFace auth token read from `~/.huggingface/token` (standard location) or `HF_TOKEN` env var.

### Scripting Support

- Non-interactive mode via `--yes` flag (skip all prompts)
- `--json-report` for machine-readable output
- Exit codes: 0 (success), 1 (conversion error), 2 (quality threshold exceeded), 3 (input error)
- Stderr for warnings/progress, stdout clean for piping when `--json-report` is used
- Shell completions via `hf2q completions --shell {bash,zsh,fish}`

### Implementation Considerations

- Built with `clap` v4 (derive API) for argument parsing and help generation
- `clap_complete` for shell completion generation
- `indicatif` for progress bars
- All subcommands are synchronous by default; async download via `hf-hub` internals
- Binary size target: single static binary, no dynamic linking to Python or other runtimes
