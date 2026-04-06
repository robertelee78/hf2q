---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
status: 'complete'
completedAt: '2026-04-06'
inputDocuments: ['prd.md']
workflowType: 'architecture'
project_name: 'hf2q'
user_name: 'Robert'
date: '2026-04-06'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Engineering Philosophy

> DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.

## Project Context Analysis

### Requirements Overview

**Functional Requirements:**
46 FRs across 9 capability areas. The conversion pipeline (FR7-FR15) is the core — read safetensors, quantize, write output format. Quality measurement (FR16-FR21) requires inference passes and is compute-intensive. Self-learning (FR26-FR29) via RuVector adds a persistent state layer. Future output targets (FR41-FR46) demand a pluggable backend architecture.

**Non-Functional Requirements:**
- **NFR4** (memory < 2x single shard) is the hardest constraint — forces streaming architecture throughout
- **NFR6-7** (no silent corruption, clean interrupt) require transactional output writing
- **NFR9-10** (format compatibility) require validation against external specs
- **NFR14-15** (backend extensibility, architecture extensibility) mandate clean separation of concerns

**Scale & Complexity:**
- Primary domain: CLI tool / ML pipeline
- Complexity level: Medium-high
- Estimated architectural components: 8

### Technical Constraints & Dependencies

- **MSRV 1.81.0** (required by `mlx-rs`)
- **macOS-only for v1** — MLX and CoreML require Apple Silicon. Linux compiles but platform-specific backends return errors.
- **`mlx-rs` v0.25** for MLX array ops and safetensors integration
- **`ruvector-core` v2.1** for self-learning auto mode — must be optional (fallback to heuristics)
- **Single binary, no Python** — hard constraint, zero runtime deps on other languages

### Cross-Cutting Concerns

- **Memory management:** Streaming/mmap throughout — touches input, quantization, quality measurement, and output
- **Progress reporting:** Every long-running phase needs `indicatif` progress bars — download, shard loading, quantization, quality measurement
- **Error handling with graceful degradation:** Unknown layers → f16 fallback, RuVector failure → heuristic fallback, download failure → clear error
- **Format validation:** Every output backend must validate its output before writing — garbage-in-garbage-out prevention
- **Interrupt safety:** Ctrl+C must clean up partial output directories (NFR7)

## Project Foundation

### Technology Stack (Locked)

- **Language:** Rust (MSRV 1.81.0)
- **CLI:** `clap` v4 derive API + `clap_complete`
- **ML:** `mlx-rs` v0.25 (safetensors, metal, accelerate features)
- **Progress:** `indicatif` v0.17
- **Parallelism:** `rayon` v1.10
- **I/O:** `memmap2` v0.9 for memory-mapped shard access
- **Serialization:** `serde` + `serde_json`
- **Errors:** `anyhow` (application) + `thiserror` (library)

### Crate Structure: Single Crate

Single binary crate with clean internal module boundaries. No workspace overhead until the codebase warrants it.

```
hf2q/
├── Cargo.toml
└── src/
    ├── main.rs              # Entry point, clap dispatch
    ├── cli.rs               # Clap structs, arg validation
    ├── input/               # Safetensors reader, HF Hub downloader, config parser
    ├── ir.rs                # Intermediate representation (weights + metadata)
    ├── quantize/            # Static quant, mixed-bit, DWQ calibration
    ├── quality/             # KL divergence, perplexity, cosine similarity
    ├── backends/            # Output format writers (MLX, CoreML, future)
    ├── intelligence/        # Hardware profiler, RuVector integration, auto mode
    ├── progress.rs          # Indicatif progress bar helpers
    └── report.rs            # JSON report generation, terminal summary
```

**Rationale:** Chesterton's fence — don't add workspace complexity until we need it. Module boundaries provide the same separation of concerns. Workspace split is trivial later if Linux backends grow the codebase significantly.

**Updated module structure (from decisions):**

```
hf2q/
├── Cargo.toml
└── src/
    ├── main.rs              # Entry point, clap dispatch, preflight checks
    ├── cli.rs               # Clap structs, arg validation
    ├── input/
    │   ├── mod.rs           # Input orchestration
    │   ├── safetensors.rs   # Streaming shard reader via mmap
    │   ├── config_parser.rs # HF config.json → ModelMetadata
    │   └── hf_download.rs   # hf-hub crate (primary), hf CLI (fallback)
    ├── ir.rs                # TensorMap: name → TensorRef (lazy mmap'd data)
    ├── inference/
    │   ├── mod.rs           # InferenceRunner trait
    │   ├── mlx_runner.rs    # mlx-rs implementation (shared by DWQ + quality)
    │   └── stub_runner.rs   # Returns UnsupportedPlatform on Linux
    ├── quantize/
    │   ├── mod.rs           # Quantizer trait, dispatch
    │   ├── static_quant.rs  # f16, q8, q4, q2 round-to-nearest
    │   ├── mixed.rs         # Mixed-bit with --sensitive-layers
    │   └── dwq.rs           # DWQ calibration (uses InferenceRunner)
    ├── quality/
    │   ├── mod.rs           # QualityReport, orchestration
    │   ├── kl_divergence.rs # Per-layer and overall KL div
    │   ├── perplexity.rs    # Perplexity delta measurement
    │   └── cosine_sim.rs    # Layer activation similarity
    ├── backends/
    │   ├── mod.rs           # OutputBackend trait, registry
    │   ├── mlx.rs           # safetensors + MLX config.json
    │   ├── coreml.rs        # .mlpackage via coreml-native
    │   ├── gguf.rs          # Future
    │   └── nvfp4.rs         # Future
    ├── intelligence/
    │   ├── mod.rs           # AutoResolver orchestration
    │   ├── hardware.rs      # HardwareProfiler (sysinfo)
    │   ├── fingerprint.rs   # ModelFingerprint from config.json
    │   ├── heuristics.rs    # Rule-based fallback
    │   └── ruvector.rs      # RuVector query/store (required)
    ├── preflight.rs         # Pre-conversion validation checks
    ├── progress.rs          # Indicatif progress bar helpers
    ├── report.rs            # JSON report + terminal summary
    └── doctor.rs            # hf2q doctor subcommand diagnostics
```

**Initialization:**
```bash
cargo init --name hf2q
```

## Core Architectural Decisions

### Error Philosophy: Fail Early and Loud

> Never pretend to do what was asked and deliver less. If we can't do what the user asked, stop immediately, explain why, and tell them exactly how to fix it.

**Pre-flight checks** run before any expensive work:

```rust
fn preflight(input: &ModelMetadata, config: &ConvertConfig, hardware: &HardwareProfile) -> Result<()> {
    // All layer types supported for chosen quant?
    // Output format compatible with architecture?
    // Estimated memory fits in available RAM?
    // Calibration data available if DWQ requested?
    // --sensitive-layers range valid for this model?
    // RuVector accessible? (required — conversion must store learnings)
    // Any failure = clear error + actionable guidance
}
```

**No implicit fallbacks.** Explicit opt-in only:

| Situation | Behavior |
|---|---|
| Unknown layer type | ERROR + explain options. User must pass `--unsupported-layers=passthrough` to explicitly opt in to f16 passthrough. |
| RuVector operational, has data | Normal operation — use stored config. |
| RuVector operational, no data | WARNING: "No prior data for this combo. Using heuristics. Results will be stored." Proceed. |
| RuVector unavailable | ERROR: "RuVector not accessible. Required to store learnings. Run `hf2q doctor` to diagnose." Refuse to proceed. |
| Output format incompatible with model | ERROR before conversion + explain incompatibility. |
| Insufficient memory | Pre-flight ERROR + suggest lower quant or `--skip-quality`. |
| Quality threshold exceeded | Exit code 2 + clear report. |

### Critical Decisions Summary

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Pipeline data flow | Hybrid streaming: eager metadata, lazy tensor data via mmap, per-layer bounded memory | Best quality; satisfies NFR4; handles any model size |
| 2 | Quantization engine | Trait-based (`Quantizer`) with Static, MixedBit, DWQ implementations | Clean separation; DWQ shares InferenceRunner with quality |
| 3 | Inference runtime | Shared `InferenceRunner` wrapping `mlx-rs` | Avoids double model loading; contains mlx-rs dependency |
| 4 | Output backends | Trait-based (`OutputBackend`) with validate-before-write | Pluggable; adding format = adding a file |
| 5 | Auto mode | AutoResolver: RuVector (required) → heuristic supplement. No data = warning + heuristics. No RuVector = error. | Every conversion must leave system smarter |
| 6 | Error handling | Fail early and loud. Pre-flight checks. No implicit fallbacks. `thiserror` internal, `anyhow` at boundary. | Never deliver less than asked without explicit consent |

### Command Structure (Updated)

```
hf2q
├── convert         # Primary — convert model to target format
├── info            # Inspect model metadata
├── doctor          # Diagnose RuVector, hardware detection, mlx-rs
└── completions     # Shell completions (bash/zsh/fish)
```

### Implementation Sequence

1. IR + mmap shard reader (foundation)
2. Static quantizer (f16/q8/q4)
3. MLX output backend
4. CLI layer (clap + indicatif) + preflight checks
5. HF Hub downloader
6. InferenceRunner (mlx-rs)
7. Quality measurement (KL div, perplexity)
8. MixedBit quantizer + `--sensitive-layers`
9. DWQ quantizer (calibration)
10. Hardware profiler + heuristic auto mode
11. RuVector integration
12. CoreML backend
13. `hf2q doctor` subcommand

## Implementation Patterns & Consistency Rules

### Rust Naming Conventions (Standard — No Deviation)

- **Modules/files:** `snake_case` (`kl_divergence.rs`, `mlx_runner.rs`)
- **Types/traits:** `PascalCase` (`TensorRef`, `OutputBackend`, `QuantizeError`)
- **Functions/methods:** `snake_case` (`quantize_tensor`, `measure_kl_divergence`)
- **Constants:** `SCREAMING_SNAKE_CASE` (`DEFAULT_GROUP_SIZE`, `MAX_SHARD_COUNT`)
- **Enum variants:** `PascalCase` (`QuantMethod::DwqMixed46`)
- **Feature flags:** `kebab-case` (`coreml-backend`, `nvidia-backends`)

### Error Pattern

Every module defines its own error enum via `thiserror`. No `String` errors. No `.unwrap()` except in tests. `anyhow` only in `main.rs` and `cli.rs`.

### Trait Pattern

All extension points use traits. All traits are `Send + Sync` (future-proofs for rayon). Consistent shape:

```rust
trait Quantizer: Send + Sync {
    fn name(&self) -> &str;
    fn requires_calibration(&self) -> bool;
    fn quantize_tensor(&self, tensor: &TensorRef, config: &LayerQuantConfig) -> Result<QuantizedTensor>;
}

trait OutputBackend: Send + Sync {
    fn name(&self) -> &str;
    fn validate(&self, model: &QuantizedModel) -> Result<Vec<FormatWarning>>;
    fn write(&self, model: &QuantizedModel, out: &Path, progress: &ProgressReporter) -> Result<OutputManifest>;
}
```

### Progress Reporting Pattern

Every long-running function takes `&ProgressReporter`. No direct `indicatif` calls outside `progress.rs`.

### Testing Pattern

- Unit tests co-located (`#[cfg(test)] mod tests`)
- Integration tests in `tests/` directory
- Small safetensors fixtures in `tests/fixtures/`
- No mocks for core pipeline — real (small) tensors. Mock only external services.

### Logging Pattern

`tracing` for structured logging. All messages include context fields. No `println!` for diagnostics.

### Configuration Flow Pattern

CLI args → `ConvertConfig` struct → passed through pipeline. No global state. All env vars resolved in `cli.rs` at startup.

### Anti-Patterns (Forbidden)

- `.unwrap()` / `.expect()` outside tests
- `println!` for anything except user-facing output
- Global mutable state
- `String` as error type
- Direct `indicatif` outside `progress.rs`
- `todo!()` or `unimplemented!()` in shipped code
- Scattered `#[cfg]` — isolate in `inference/` and `backends/`

### Additional Dependency

```toml
tracing = "0.1"
tracing-subscriber = "0.3"  # Terminal log output

## Project Structure & Boundaries

### Complete Project Directory Structure

```
hf2q/
├── Cargo.toml
├── Cargo.lock
├── .gitignore
├── README.md
├── LICENSE-MIT
├── LICENSE-APACHE
│
├── src/
│   ├── main.rs                    # Entry point, clap dispatch, top-level anyhow
│   ├── cli.rs                     # Clap derive structs, arg validation, env var resolution
│   ├── preflight.rs               # Pre-conversion validation (fail early and loud)
│   ├── ir.rs                      # TensorMap, TensorRef, ModelMetadata, QuantizedModel
│   ├── progress.rs                # ProgressReporter wrapping indicatif
│   ├── report.rs                  # JSON report generation, terminal summary
│   ├── doctor.rs                  # hf2q doctor diagnostics
│   │
│   ├── input/
│   │   ├── mod.rs                 # InputSource resolution (local vs HF Hub)
│   │   ├── safetensors.rs         # Streaming mmap shard reader
│   │   ├── config_parser.rs       # HF config.json → ModelMetadata
│   │   └── hf_download.rs         # hf-hub crate + hf CLI fallback
│   │
│   ├── inference/
│   │   ├── mod.rs                 # InferenceRunner trait
│   │   ├── mlx_runner.rs          # mlx-rs implementation
│   │   └── stub_runner.rs         # UnsupportedPlatform on non-Apple
│   │
│   ├── quantize/
│   │   ├── mod.rs                 # Quantizer trait, dispatch by QuantMethod
│   │   ├── static_quant.rs        # f16, q8, q4, q2 round-to-nearest
│   │   ├── mixed.rs               # Mixed-bit with sensitive-layers
│   │   └── dwq.rs                 # DWQ calibration via InferenceRunner
│   │
│   ├── quality/
│   │   ├── mod.rs                 # QualityReport orchestration
│   │   ├── kl_divergence.rs       # Per-layer + overall KL div
│   │   ├── perplexity.rs          # Perplexity delta
│   │   └── cosine_sim.rs          # Layer activation similarity
│   │
│   ├── backends/
│   │   ├── mod.rs                 # OutputBackend trait, registry
│   │   ├── mlx.rs                 # safetensors + MLX config.json
│   │   ├── coreml.rs              # .mlpackage via coreml-native
│   │   ├── gguf.rs                # Future
│   │   └── nvfp4.rs               # Future
│   │
│   └── intelligence/
│       ├── mod.rs                 # AutoResolver orchestration
│       ├── hardware.rs            # HardwareProfiler via sysinfo
│       ├── fingerprint.rs         # ModelFingerprint from config.json
│       ├── heuristics.rs          # Rule-based quant selection
│       └── ruvector.rs            # RuVector query/store (required)
│
├── tests/
│   ├── fixtures/
│   │   ├── tiny_model/            # Small safetensors (< 1MB)
│   │   └── malformed/             # Invalid inputs for error paths
│   ├── convert_integration.rs
│   ├── info_integration.rs
│   ├── preflight_integration.rs
│   └── cli_integration.rs
│
└── benches/
    ├── quantize_bench.rs
    └── shard_read_bench.rs
```

### Architectural Boundaries

- **Input boundary:** `input/` owns all external format I/O. Nothing outside touches raw model files.
- **IR boundary:** `ir.rs` is the central data contract. Input produces it, quantize transforms it, backends consume it.
- **Inference boundary:** `inference/` wraps `mlx-rs` entirely. No other module imports `mlx_rs`. Platform `#[cfg]` lives only here and in `backends/`.
- **Backend boundary:** Each backend is self-contained. New format = new file + registry entry.
- **Intelligence boundary:** RuVector isolated in `ruvector.rs`. Codebase sees only `AutoResolver::resolve()`.

### FR to Module Mapping

| FR Range | Module | Responsibility |
|---|---|---|
| FR1-FR6 | `input/` | Model loading, HF download, config parsing |
| FR7-FR8 | `backends/` | Output format writing |
| FR9-FR15 | `quantize/` + `ir.rs` | Quantization, shard consolidation, dtype conversion |
| FR16-FR21 | `quality/` | KL divergence, perplexity, quality reporting |
| FR22-FR25 | `intelligence/` | Hardware profiling, auto mode |
| FR26-FR29 | `intelligence/ruvector.rs` | Self-learning storage and retrieval |
| FR30-FR34 | `backends/` + `report.rs` + `progress.rs` | Output, reporting, progress |
| FR35-FR37 | `input/safetensors.rs` + `preflight.rs` | Memory safety, validation |
| FR38-FR40 | `cli.rs` + `main.rs` | Scripting, exit codes, completions |

### Data Flow

```
CLI args → ConvertConfig
  → preflight::validate(config, metadata, hardware)?     ← FAIL EARLY
  → input::read_model(source) → ModelMetadata + TensorMap (lazy mmap)
  → intelligence::resolve(hardware, fingerprint) → QuantConfig  ← RuVector query
  → quantize::run(tensor_map, quant_config) → QuantizedModel
  → quality::measure(original, quantized, runner) → QualityReport  ← InferenceRunner
  → backend::validate(quantized_model)?                   ← FAIL EARLY
  → backend::write(quantized_model, output_dir) → OutputManifest
  → intelligence::store(hardware, fingerprint, config, quality)  ← RuVector store
  → report::summarize(manifest, quality_report)
```

## Architecture Validation Results

### Coherence Validation

- All technology choices compatible (mlx-rs 0.25, safetensors 0.7, clap 4.5, rayon 1.10, ruvector-core 2.1)
- MSRV 1.81.0 satisfies all dependencies
- Error handling pattern consistent: `thiserror` in modules, `anyhow` at boundary
- Trait pattern uniform: `Quantizer`, `OutputBackend`, `InferenceRunner` — all `Send + Sync`
- No global state — `ConvertConfig` flows through entire pipeline
- Module boundaries match decision boundaries with no cross-cutting imports

### Requirements Coverage

All 46 FRs and 15 NFRs map to specific modules. No orphan requirements. Future FRs (41-46) covered by pluggable backend design.

### Implementation Readiness

- Every module has clear input/output contracts via IR types
- Trait-based extension points documented with signatures
- Anti-patterns explicitly forbidden
- Testing strategy defined (real tensors, no core pipeline mocks)
- Data flow diagram shows exact call sequence

### Additional Features (from validation)

**`--dry-run` flag:** Runs preflight checks + auto mode resolution + backend validation, prints planned conversion config, then exits without converting. Useful for verifying what hf2q would do before committing to a long conversion.

**`hf2q doctor` verifies:**
1. RuVector database accessible and healthy
2. `mlx-rs` Metal backend available (Apple Silicon check)
3. `hf` CLI on `$PATH` (download fallback)
4. Available disk space
5. System memory vs typical model sizes

### Architecture Completeness Checklist

- [x] Project context analyzed, scale assessed, constraints identified
- [x] Technology stack fully specified with versions
- [x] Critical architectural decisions documented with rationale
- [x] Error philosophy defined (fail early and loud, no implicit fallbacks)
- [x] Implementation patterns and anti-patterns documented
- [x] Complete project structure with FR-to-module mapping
- [x] Data flow diagram with fail-early checkpoints
- [x] Validation passed: coherence, coverage, readiness

**Overall Status: READY FOR IMPLEMENTATION**
**Confidence Level: High**
```
