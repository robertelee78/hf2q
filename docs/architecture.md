# hf2q Architecture

## Overview

hf2q is a pure-Rust CLI tool that reads HuggingFace model weights in safetensors
format and writes optimally quantized output in GGUF (for llama.cpp/Ollama) or
quantized safetensors (for Candle/vLLM). It provides ten quantization methods
ranging from simple round-to-nearest (F16/Q8/Q4/Q2) through calibration-aware
mixed-bit (DWQ) and importance-matrix-driven per-tensor optimization (Apex). A
self-learning database (RuVector) stores past conversion outcomes so that
`--quant auto` improves over time. The tool is cross-platform (macOS + Linux),
requires no Python, and produces a single static binary.

Shaped by ADR-003 (2026-04-08), which stripped the original CoreML/MLX backends
and built-in inference server to focus entirely on quantization output for the
formats the ecosystem actually uses.


## System Architecture

```
                          +---------------------+
                          |   CLI (src/cli.rs)  |
                          |  clap derive parser |
                          +----------+----------+
                                     |
                                     v
                          +---------------------+
                          | main.rs dispatcher  |
                          | cmd_convert / info  |
                          | validate / doctor   |
                          +----------+----------+
                                     |
           +-------------------------+-------------------------+
           |                         |                         |
           v                         v                         v
   +--------------+        +------------------+       +-----------------+
   |  src/input/  |        | src/intelligence/ |       | src/preflight.rs|
   | safetensors  |        | auto_quant        |       | pre-conversion  |
   | config_parser|        | fingerprint       |       | validation      |
   | hf_download  |        | hardware          |       +-----------------+
   +-------+------+        | heuristics        |
           |               | ruvector          |
           v               +--------+----------+
   +---------------+                |
   | IR (src/ir.rs)|                |
   | TensorMap     |<---------------+  (fingerprint + hardware inform
   | ModelMetadata |                    auto-quant decisions)
   +-------+-------+
           |
           v
   +-----------------+     +-----------------------+
   | src/quantize/   |     | src/gpu/              |
   | static_quant    |<----| forward.rs (Candle)   |
   | mixed           |     | tokenizer.rs          |
   | dwq             |     | mod.rs (device select)|
   | dwq_activation  |     +-----------------------+
   | apex            |
   | sensitivity     |
   +-------+---------+
           |
           v
   +------------------+
   | QuantizedModel   |  (IR: packed data + per-tensor TensorQuantInfo)
   +-------+----------+
           |
     +-----+-----+
     |           |
     v           v
+----------+ +-----------------+     +------------------+
| GGUF     | | safetensors_out |     | src/quality/     |
| backend  | | backend         |     | cosine_sim       |
+----+-----+ +--------+--------+     | kl_divergence    |
     |                |              | perplexity       |
     v                v              | regression       |
  .gguf file    .safetensors +       +--------+---------+
                quant_config.json             |
                                              v
                                     +------------------+
                                     | src/report.rs    |
                                     | JSON report (CI) |
                                     +------------------+
```

Cross-cutting concerns:
- `src/progress.rs` -- indicatif progress bars used by every pipeline stage
- `src/doctor.rs` -- diagnostics subcommand (hardware, RuVector, disk)
- Signal handling -- SIGINT triggers cleanup of partial output directories


## Module Map

| Module | File | Lines | Purpose | Key Types |
|--------|------|------:|---------|-----------|
| **Entry point** | `src/main.rs` | 1136 | CLI dispatch, conversion pipeline orchestration, exit codes | `AppError`, `cmd_convert()` |
| **CLI** | `src/cli.rs` | 433 | clap derive parser, config resolution | `Cli`, `Command`, `ConvertArgs`, `ConvertConfig`, `OutputFormat`, `QuantMethod` |
| **IR** | `src/ir.rs` | 529 | Central data contract between stages | `TensorRef`, `TensorMap`, `ModelMetadata`, `QuantizedTensor`, `QuantizedModel`, `TensorQuantInfo`, `OutputManifest` |
| **Input (mod)** | `src/input/mod.rs` | 71 | Module root, `read_model()` entry point | `InputError` |
| **Input (safetensors)** | `src/input/safetensors.rs` | 440 | Mmap-based shard reader, safetensors binary format parser | `read_tensors()`, `discover_shards()` |
| **Input (config)** | `src/input/config_parser.rs` | 397 | HF config.json -> ModelMetadata, nested text_config support | `parse_config()`, `ConfigParseError` |
| **Input (download)** | `src/input/hf_download.rs` | 557 | HF Hub download via hf-hub crate + CLI fallback | `download_model()`, `DownloadError` |
| **Quantize (mod)** | `src/quantize/mod.rs` | 138 | `Quantizer` trait, `quantize_model()` pipeline | `Quantizer`, `LayerQuantConfig`, `QuantizeError` |
| **Quantize (static)** | `src/quantize/static_quant.rs` | 456 | F16/Q8/Q4/Q2 round-to-nearest with per-group scaling | `StaticQuantizer` |
| **Quantize (mixed)** | `src/quantize/mixed.rs` | 471 | Mixed-bit with sensitive layer protection | `MixedBitQuantizer`, `MixedBitPreset` |
| **Quantize (DWQ)** | `src/quantize/dwq.rs` | 534 | Weight-space calibrated quantization | `DwqQuantizer`, `DwqConfig`, `run_dwq_calibration()` |
| **Quantize (DWQ act.)** | `src/quantize/dwq_activation.rs` | 303 | Activation-based DWQ via Candle forward passes | `run_dwq_activation_calibration()` |
| **Quantize (Apex)** | `src/quantize/apex.rs` | 743 | Imatrix-calibrated per-tensor K-quant selection | `ApexConfig`, `run_apex_quantization()`, `compute_importance_matrix()` |
| **Quantize (sensitivity)** | `src/quantize/sensitivity.rs` | 245 | Per-layer sensitivity scoring from activations | `LayerSensitivity`, `compute_layer_sensitivity()` |
| **Backends (mod)** | `src/backends/mod.rs` | 86 | `OutputBackend` trait definition | `OutputBackend`, `BackendError` |
| **Backends (GGUF)** | `src/backends/gguf.rs` | 752 | GGUF v3 binary writer, HF-to-GGUF tensor naming | `GgufBackend` |
| **Backends (safetensors)** | `src/backends/safetensors_out.rs` | 595 | Multi-shard safetensors writer with quant_config.json | `SafetensorsBackend` |
| **GPU (mod)** | `src/gpu/mod.rs` | 242 | Candle device selection, IR-to-Candle tensor conversion | `GpuDevice`, `select_device()`, `tensor_from_ir()` |
| **GPU (forward)** | `src/gpu/forward.rs` | 507 | Decoder-only transformer forward pass for calibration | `ForwardOutput`, `TransformerLayer`, `RmsNorm` |
| **GPU (tokenizer)** | `src/gpu/tokenizer.rs` | 121 | Tokenizer loading and calibration text encoding | `load_tokenizer()`, `encode_calibration_text()` |
| **Intelligence (mod)** | `src/intelligence/mod.rs` | 363 | Auto mode resolver, format-aware dispatch | `AutoResolver`, `ResolvedConfig`, `ResolvedSource`, `OutputFormatHint` |
| **Intelligence (auto)** | `src/intelligence/auto_quant.rs` | 1018 | Bandwidth-model algorithm, per-component sensitivity | `AutoQuantPlan`, `ComponentOverride`, `AutoQuantConstraints` |
| **Intelligence (fingerprint)** | `src/intelligence/fingerprint.rs` | 285 | Stable model identity from config.json | `ModelFingerprint`, `stable_id()` |
| **Intelligence (hardware)** | `src/intelligence/hardware.rs` | 440 | Chip/memory/core detection via sysinfo | `HardwareProfile`, `stable_id()` |
| **Intelligence (heuristics)** | `src/intelligence/heuristics.rs` | 409 | Memory-fitting rules for quant selection | `HeuristicResult`, `select_quant_with_format()` |
| **Intelligence (RuVector)** | `src/intelligence/ruvector.rs` | 1193 | JSON-backed self-learning DB at ~/.hf2q/ruvector/ | `RuVectorDb`, `ConversionRecord`, `QualityMetrics` |
| **Quality (mod)** | `src/quality/mod.rs` | 718 | Orchestrates measurement, dequantize, threshold checks | `QualityReport`, `QualityThresholds`, `measure_quality()` |
| **Quality (cosine)** | `src/quality/cosine_sim.rs` | 252 | Per-layer weight cosine similarity | `compute_cosine_similarity()` |
| **Quality (KL)** | `src/quality/kl_divergence.rs` | 267 | KL divergence via Candle forward passes | `compute_kl_divergence()` |
| **Quality (perplexity)** | `src/quality/perplexity.rs` | 249 | Perplexity measurement via Candle | `compute_perplexity()` |
| **Quality (regression)** | `src/quality/regression.rs` | 401 | Baseline comparison, quality gate for CI | `QualityGate`, `RegressionWarning`, `detect_regression()` |
| **Preflight** | `src/preflight.rs` | 731 | Pre-conversion validation (layers, disk, format compat) | `PreflightReport`, `PreflightError`, `validate()` |
| **Report** | `src/report.rs` | 567 | Structured JSON report (schema v1) for CI pipelines | `ConversionReport`, `ReportBuilder` |
| **Doctor** | `src/doctor.rs` | 335 | System health diagnostics | `run_doctor()` |
| **Progress** | `src/progress.rs` | 199 | indicatif wrapper, byte/param formatting | `ProgressReporter`, `format_bytes()` |
| **Total** | 35 files | 16183 | | |


## Data Flow

What happens when you run:

    hf2q convert --repo google/gemma-3-27b --format gguf --quant apex

### Phase 0: Initialization

1. `Cli::parse()` produces `ConvertArgs`.
2. `resolve_convert_config()` resolves `--repo` into a local directory via
   `input::hf_download::download_model()`, which tries the `hf-hub` crate first
   then falls back to the `hf` CLI. Returns a `ConvertConfig` struct.
3. `config_parser::parse_config()` reads `config.json` to produce
   `ModelMetadata` (architecture, layers, MoE info, dtype).
4. `RuVectorDb::open_default()` opens `~/.hf2q/ruvector/` (required).
5. `HardwareProfiler::detect()` probes chip model, memory, cores via `sysinfo`.
6. `ModelFingerprint::from_metadata()` creates a stable hashable identity.
7. If `--quant auto`: `AutoResolver::resolve_with_format()` queries RuVector
   for stored results, falls back to `heuristics::select_quant_with_format()`.

### Phase 0.5: Preflight

8. `preflight::validate()` checks: input dir exists, safetensors present,
   layer types supported, sensitive layer ranges valid, output dir empty,
   disk space sufficient. Returns `PreflightReport` or actionable error.
9. If `--dry-run`: print the plan and exit(0).

### Phase 1: Read

10. `input::read_model()` calls `safetensors::read_tensors()`, which:
    - Discovers shards via `model.safetensors.index.json` or directory scan.
    - Memory-maps each shard via `memmap2`.
    - Parses the safetensors header (8-byte length + JSON + data offsets).
    - Copies tensor data out of each mmap into `TensorRef` structs.
    - Drops the mmap per shard to bound memory.
    - Returns `TensorMap` (HashMap<String, TensorRef>).

### Phase 2: Backend Selection and BF16 Conversion

11. Backend instantiated: `GgufBackend::new()` or `SafetensorsBackend::new()`.
12. Unless the backend requires native quantization, all BF16 tensors are
    converted to F16 via `TensorMap::convert_bf16_to_f16()`.

### Phase 3: Quantize

13. Dispatch by `QuantMethod`:
    - **F16/Q8/Q4/Q2**: `StaticQuantizer` -- per-group scale factors,
      round-to-nearest. Non-weight tensors (norms, biases, embeddings)
      preserved at original precision via `TensorRef::is_weight()`.
    - **Mixed-{2,3,4}-6**: `MixedBitQuantizer` -- routes each tensor through
      a `StaticQuantizer` at either `base_bits` or `sensitive_bits` depending
      on layer index vs `--sensitive-layers`.
    - **DWQ-Mixed-4-6**: `DwqQuantizer` -- mixed-bit quantization plus
      weight-space calibration that computes optimal scale per group:
      `scale = dot(W, Q) / dot(Q, Q)`. If `tokenizer.json` exists, uses
      `dwq_activation::run_dwq_activation_calibration()` with Candle forward
      passes for activation-based sensitivity scoring.
    - **Apex**: `run_apex_quantization()` -- two-pass pipeline:
      - Pass 1: `compute_importance_matrix()` runs Candle forward passes to
        capture activation magnitudes, then computes per-tensor importance as
        `mean(|activation| * |weight|)`. Falls back to weight-only magnitudes
        when GPU is unavailable.
      - Pass 2: Allocates K-quant types (Q2_K through Q6_K) per tensor to
        meet `--target-bpw` budget, higher-importance tensors get more bits.
        Stores the chosen GGML type name in `TensorQuantInfo::ggml_type`.

14. Output: `QuantizedModel` containing `HashMap<String, QuantizedTensor>`,
    each with packed data bytes, shape, and `TensorQuantInfo` (method, bits,
    group_size, scales, biases, optional ggml_type).

### Phase 4: Quality Measurement

15. Unless `--skip-quality`:
    - `dequantize_to_tensor_map()` reconstructs approximate f16 weights from
      quantized data for comparison.
    - `measure_quality()` computes weight-level cosine similarity per layer.
    - If tokenizer is available: KL divergence and perplexity via Candle
      forward passes on both original and quantized weights.
    - `regression::detect_regression()` compares against stored RuVector
      baseline (if any) and prints degradation warnings.

### Phase 5: Write Output

16. **GGUF backend** (`gguf.rs`):
    - Writes GGUF v3 binary: magic + version + header (tensor count, KV count).
    - Metadata KV pairs: architecture, layer count, vocab size, etc.
    - Per-tensor: maps HF names to GGUF names (`model.layers.N.self_attn.q_proj.weight`
      -> `blk.N.attn_q.weight`), selects GGML dtype from bit width or
      `ggml_type` override, writes data with 32-byte alignment padding.
    - Produces a single `.gguf` file.

17. **Safetensors backend** (`safetensors_out.rs`):
    - Builds safetensors header with `__metadata__` containing quant info.
    - Quantized weights stored as U8 (packed bits); preserved tensors keep
      original dtype. Scale tensors stored as separate entries.
    - Multi-shard output for models exceeding 4 GB per shard.
    - Writes `quantization_config.json` sidecar with per-tensor method,
      bits, group_size, and preserved flags.
    - Writes `model.safetensors.index.json` for sharded models.
    - Copies `config.json` and `tokenizer.json` from input.

### Phase 6: Post-conversion

18. Stores conversion result in RuVector for future `--quant auto` queries.
19. If `--quality-gate` and thresholds exceeded: exit(2).
20. If `--json-report`: writes `report.json` (schema v1) to output dir or stdout.
21. Prints terminal summary: model name, compression ratio, elapsed time.


## Intermediate Representation

The IR is the contract between input, quantize, and backend stages. All types
are in `src/ir.rs` and are `Send + Sync`.

### Input Stage Produces

```
TensorMap
  tensors: HashMap<String, TensorRef>

TensorRef
  name: String          -- e.g., "model.layers.0.self_attn.q_proj.weight"
  shape: Vec<usize>     -- e.g., [4096, 4096]
  dtype: DType           -- F32, F16, BF16, I32, I64, U8, U16, U32, Bool
  data: Vec<u8>          -- raw bytes (copied from mmap)

ModelMetadata
  architecture: String   -- from config.json "architectures[0]"
  model_type: String
  param_count: u64
  hidden_size: u64
  num_layers: u32
  layer_types: Vec<String>
  num_attention_heads: u32
  num_kv_heads: Option<u32>
  vocab_size: u64
  dtype: String
  shard_count: u32
  num_experts: Option<u32>
  top_k_experts: Option<u32>
  intermediate_size: Option<u64>
  raw_config: serde_json::Value
```

### Quantize Stage Produces

```
QuantizedModel
  metadata: ModelMetadata
  tensors: HashMap<String, QuantizedTensor>
  quant_method: String
  group_size: usize
  bits: u8

QuantizedTensor
  name: String
  shape: Vec<usize>
  original_dtype: DType
  data: Vec<u8>           -- packed quantized bytes
  quant_info: TensorQuantInfo

TensorQuantInfo
  method: String           -- "q4", "f16", "passthrough"
  bits: u8
  group_size: usize
  preserved: bool          -- true for norms, biases, scalars
  scales: Option<Vec<u8>>  -- per-group scale factors
  biases: Option<Vec<u8>>  -- per-group zero points
  ggml_type: Option<String> -- e.g., "Q4_K_M" (Apex only)
```

### Backend Produces

```
OutputManifest
  output_dir: String
  files: Vec<OutputFile>   -- filename + size_bytes
  total_size_bytes: u64
  shard_count: usize
```

### Weight Classification

`TensorRef::is_weight()` determines what gets quantized vs preserved:

- **Quantized**: 2D+ tensors with "weight", "proj", or "experts." in name
- **Preserved at f16**: layernorm, rmsnorm, bias, scalars, router scales,
  embed_tokens, embedding_projection


## Quantization Methods

| Method | CLI Flag | Type | Calibration | GPU | Implementation |
|--------|----------|------|-------------|-----|----------------|
| F16 | `--quant f16` | Passthrough (bf16->f16) | No | No | `StaticQuantizer` |
| Q8 | `--quant q8` | Round-to-nearest, 8-bit | No | No | `StaticQuantizer` |
| Q4 | `--quant q4` | Round-to-nearest, 4-bit | No | No | `StaticQuantizer` |
| Q2 | `--quant q2` | Round-to-nearest, 2-bit | No | No | `StaticQuantizer` |
| Mixed-2-6 | `--quant mixed-2-6` | 2-bit base, 6-bit sensitive | No | No | `MixedBitQuantizer` |
| Mixed-3-6 | `--quant mixed-3-6` | 3-bit base, 6-bit sensitive | No | No | `MixedBitQuantizer` |
| Mixed-4-6 | `--quant mixed-4-6` | 4-bit base, 6-bit sensitive | No | No | `MixedBitQuantizer` |
| DWQ-Mixed-4-6 | `--quant dwq-mixed-4-6` | Weight-space calibrated mixed | Weight (CPU) | Optional | `DwqQuantizer` |
| Apex | `--quant apex` | Imatrix per-tensor K-quant | Forward pass | Optional | `run_apex_quantization()` |
| Auto | `--quant auto` | Intelligence-selected | Varies | Varies | `AutoResolver` |

### Static Quantization Algorithm

For each group of `group_size` elements in a weight tensor:
1. Compute `absmax = max(|w|)` over the group.
2. Compute scale: `scale = absmax / ((1 << (bits-1)) - 1)`.
3. Quantize: `q = round(w / scale)`, clamped to `[-(1<<(bits-1)), (1<<(bits-1))-1]`.
4. Pack quantized integers and store scale factor.

### DWQ Calibration

After initial quantization, DWQ refines the scale per group:
`optimal_scale = dot(W_original, Q_int) / dot(Q_int, Q_int)`

This closed-form solution minimizes `||W - scale * Q||^2` without requiring
forward passes, providing strictly better quality than static quantization.

When a tokenizer is available, activation-based calibration runs Candle
forward passes and uses `sensitivity::compute_layer_sensitivity()` to assign
bits per layer based on `score = sqrt(variance) * log2(1 + max_magnitude)`.

### Apex Algorithm

1. Compute importance matrix: for each weight tensor,
   `importance = mean(|activation_input| * |weight|)`.
2. Rank tensors by importance score.
3. Solve a knapsack-style allocation: assign K-quant types (Q2_K through Q6_K)
   to meet the `--target-bpw` budget, giving more bits to higher-importance
   tensors. Each K-quant type has a known bits-per-weight cost:
   Q2_K=2.5, Q3_K_S=3.0, Q3_K_M=3.5, Q4_K_S=4.2, Q4_K_M=4.5,
   Q5_K_S=5.0, Q5_K_M=5.5, Q6_K=6.5.
4. Store the selected GGML type in `TensorQuantInfo::ggml_type` so the GGUF
   backend writes the correct type ID.


## Output Backends

### GGUF Backend (`src/backends/gguf.rs`)

Writes GGUF v3 binary format directly (no external crate dependency at runtime).

- **Magic**: `GGUF` (0x47475546)
- **Version**: 3
- **Alignment**: 32 bytes for tensor data
- **Metadata KV**: architecture, layer count, vocab size, head count, etc.
  encoded as GGUF string/uint32 types.
- **Tensor naming**: Maps HuggingFace convention
  (`model.layers.N.self_attn.q_proj.weight`) to GGUF convention
  (`blk.N.attn_q.weight`). Unmapped names are hashed to prevent collisions.
- **GGML dtypes**: F32 (0), F16 (1), Q4_0 (2), Q4_1 (3), Q5_0 (6), Q5_1 (7),
  Q8_0 (8), Q8_1 (9), Q2_K (10), Q3_K_S-L (11-13), Q4_K_S/M (14-15),
  Q5_K_S/M (16-17), Q6_K (18), IQ2_XXS/XS (19-20).
- **Type selection**: Uses `TensorQuantInfo::ggml_type` if set (Apex), otherwise
  maps from bit width: 2->Q4_0, 4->Q4_0, 8->Q8_0, 16->F16, preserved->F16/F32.
- Warns on models exceeding 20 GB (single-file limit considerations).

### Safetensors Backend (`src/backends/safetensors_out.rs`)

Writes HuggingFace-compatible quantized safetensors.

- **Sharding**: 4 GB per shard. Single file for small models.
- **Data storage**: Quantized weights stored as U8 (packed bits) with original
  shape recorded. Preserved tensors keep original dtype.
- **Scale tensors**: Stored as separate entries named `{tensor_name}.scales`.
- **Metadata header**: `__metadata__` with format, quant_method, bits, group_size.
- **Sidecar files**:
  - `quantization_config.json`: per-tensor method, bits, group_size, preserved.
  - `model.safetensors.index.json`: weight map for sharded output.
  - Copies `config.json`, `tokenizer.json`, `tokenizer_config.json` from input.

### OutputBackend Trait

```rust
pub trait OutputBackend: Send + Sync {
    fn name(&self) -> &str;
    fn validate(&self, model: &QuantizedModel) -> Result<Vec<FormatWarning>, BackendError>;
    fn write(&self, model: &QuantizedModel, input_dir: &Path,
             output_dir: &Path, progress: &ProgressReporter) -> Result<OutputManifest, BackendError>;
    fn quantize_and_write(...) -> Result<OutputManifest, BackendError>;  // native quant path
    fn requires_native_quantization(&self) -> bool;  // default: false
}
```


## GPU Compute

### Candle Integration (`src/gpu/`)

Candle (candle-core 0.10 + candle-nn 0.10) serves as the GPU compute library
for calibration and quality measurement. It is not used for inference serving.

- **Device selection** (`select_device()`): Prefers Metal (feature `metal`) on
  macOS, CUDA (feature `cuda`) on Linux, falls back to CPU. Both GPU features
  are optional at compile time.
- **IR conversion** (`tensor_from_ir()`): Converts `TensorRef` to `candle_core::Tensor`
  on the selected device. Validates data length matches shape * element_size.
- **Bulk loading** (`load_tensor_map()`): Converts entire `TensorMap` to
  `HashMap<String, Tensor>` on device. Unsupported dtypes are skipped.

### Forward Pass (`src/gpu/forward.rs`)

A minimal decoder-only transformer for calibration. Not a full inference engine.

- `RmsNorm`: Upcast to f32, normalize, multiply by weight, downcast.
- `TransformerLayer`: pre-norm attention (Q/K/V projections, scaled dot-product,
  O projection) + pre-norm FFN (gate/up/down projections with SiLU activation).
  No KV cache (calibration runs full sequences).
- Supports activation capture: returns per-layer hidden states when enabled.
- Loads weights by matching HuggingFace naming conventions in the TensorMap.

### Tokenizer (`src/gpu/tokenizer.rs`)

Wraps the `tokenizers` crate to load `tokenizer.json` and encode calibration
text. Provides `load_tokenizer()` and `encode_calibration_text()`. Used by
DWQ activation calibration and Apex imatrix computation.


## Intelligence System

### Auto-Quant Algorithm (`src/intelligence/auto_quant.rs`)

Selects optimal quantization when `--quant auto` is specified.

1. **Model classification**: Determines size class and MoE vs dense.
2. **Bandwidth model**: `tok/s = bandwidth_bytes/s / model_weight_bytes`.
   For MoE: `effective_bytes = shared + (K/N) * expert_bytes`.
3. **Memory fitting**: Finds the highest bit width that fits in available
   memory with 1.3x headroom (1.8x for "generous" fit).
4. **Per-component overrides**: Assigns higher bits to sensitive components
   ranked by measured impact: router proj > embed_tokens > lm_head > MLP >
   v_proj > k_proj > q_proj > o_proj > expert FFN.
5. **Output**: `AutoQuantPlan` with `base_bits`, `component_overrides`,
   estimated size, and estimated tok/s.

### Resolution Pipeline (`src/intelligence/mod.rs`)

`AutoResolver::resolve_with_format()`:
1. Query RuVector for exact match (hardware + model fingerprint).
2. Query RuVector for similar match.
3. Fall back to `heuristics::select_quant_with_format()`.

Returns `ResolvedConfig` with `quant_method`, `bits`, `group_size`,
`confidence`, `source` (RuVectorExact/Similar/Heuristic).

### Model Fingerprinting (`src/intelligence/fingerprint.rs`)

Creates a `ModelFingerprint` from `ModelMetadata`: architecture name, param
count, layer count, expert count, attention types, hidden size, dtype, FFN
size, head counts, vocab size. Produces a deterministic `stable_id()` via
`DefaultHasher` for RuVector key lookups.

### Hardware Profiling (`src/intelligence/hardware.rs`)

`HardwareProfiler::detect()` uses `sysinfo` to capture: chip model string,
total/available memory, performance/efficiency core counts. Estimates memory
bandwidth from chip model name (Apple M-series lookup table). Produces
`stable_id()` from chip + total memory + cores.

### RuVector Self-Learning (`src/intelligence/ruvector.rs`)

JSON-backed database at `~/.hf2q/ruvector/`. Stores `ConversionRecord` entries
containing hardware profile, model fingerprint, quant method, bits, group size,
quality metrics (KL, perplexity delta, cosine sim), hf2q version, timestamp.

Key operations:
- `store_conversion()`: Save a completed conversion result.
- `query_best_config()`: Find the best stored config for a hardware+model pair.
- `update_quality()`: Update quality metrics for an existing record.
- `flag_version_changes()`: Mark records from older hf2q versions for re-calibration.
- `check_status()`: Return DB health for `hf2q doctor`.

When the `ruvector` feature is enabled, wraps `ruvector-core` for vector
similarity search (similar model lookups). Without the feature, only exact
match by `stable_id()` is available.

### Heuristics (`src/intelligence/heuristics.rs`)

Fallback when RuVector has no data. Memory-fitting rules:
- Model fits at f16 with 1.8x headroom -> f16
- Fits at q8 with headroom -> mixed-4-6
- Fits tight -> q4
- Fits very tight -> q2
- Format-aware: GGUF prefers Apex for K-quant; safetensors prefers mixed-bit.


## Quality Pipeline

### Measurement (`src/quality/mod.rs`)

`measure_quality()` orchestrates all quality metrics:

1. **Weight cosine similarity** (`cosine_sim.rs`): Computes per-layer cosine
   similarity between original and dequantized weight tensors. Works without
   GPU or tokenizer. Reports average, minimum, and worst-layer index.

2. **KL divergence** (`kl_divergence.rs`): Runs Candle forward passes on
   calibration text through both original and quantized weight sets. Computes
   KL(P_original || P_quantized) from output logit distributions. Requires
   tokenizer and GPU (falls back gracefully).

3. **Perplexity** (`perplexity.rs`): Computes perplexity on calibration text
   for both original and quantized models via Candle. Reports pre/post values
   and delta.

### Thresholds and Gates

`QualityThresholds` (defaults):
- `max_kl_divergence`: 0.1
- `max_perplexity_delta`: 2.0
- `min_cosine_similarity`: 0.95

`check_thresholds()` returns a list of violation messages. When `--quality-gate`
is set and violations exist, the tool exits with code 2.

### Regression Detection (`src/quality/regression.rs`)

`detect_regression()` compares current metrics against a stored RuVector
baseline. Classifies degradation as Info (<50% tolerance), Warning (50-100%),
or Error (>100% tolerance). `build_quality_gate()` produces a serializable
`QualityGate` struct for CI JSON output.


## Configuration and CLI

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `convert` | Convert a HuggingFace model to GGUF or safetensors |
| `info` | Inspect model metadata (architecture, layers, params, MoE) |
| `validate` | Compare original vs quantized model quality metrics |
| `doctor` | Diagnose RuVector, hardware, disk space, hf CLI |
| `completions` | Generate shell completions (bash/zsh/fish) |

### Key Convert Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input <dir>` | -- | Local safetensors directory |
| `--repo <id>` | -- | HuggingFace repo ID (auto-downloads) |
| `--format <gguf\|safetensors>` | required | Output format |
| `--quant <method>` | `auto` | Quantization method |
| `--bits <2-8>` | method default | Override bit width |
| `--group-size <32\|64\|128>` | 64 | Quantization group size |
| `--sensitive-layers <spec>` | none | Layer ranges for higher precision |
| `--target-bpw <float>` | 4.5 | Target bits-per-weight (Apex) |
| `--calibration-samples <n>` | 1024 | Calibration sample count (DWQ) |
| `--output <dir>` | auto | Output directory |
| `--skip-quality` | false | Skip quality measurement |
| `--quality-gate` | false | Exit code 2 on threshold violation |
| `--json-report` | false | Emit structured JSON report |
| `--dry-run` | false | Print plan without converting |
| `--yes` | false | Non-interactive mode |
| `--unsupported-layers passthrough` | error | Pass unsupported layers at f16 |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Conversion error |
| 2 | Quality threshold exceeded |
| 3 | Input/validation error |


## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `safetensors` | 0.7 | Read/write safetensors format |
| `half` | 2.4 | bf16/f16 type conversions |
| `hf-hub` | 0.5 | HuggingFace Hub API for model downloads |
| `clap` | 4.5 | CLI argument parsing (derive API) |
| `clap_complete` | 4.5 | Shell completion generation |
| `rayon` | 1.10 | Data parallelism for quantization |
| `memmap2` | 0.9 | Memory-mapped I/O for safetensors reading |
| `indicatif` | 0.17 | Progress bars |
| `console` | 0.15 | Terminal styling |
| `serde` / `serde_json` | 1 | JSON serialization for config, reports, RuVector |
| `anyhow` | 1 | Error context propagation |
| `thiserror` | 2 | Typed error definitions |
| `candle-core` | 0.10 | GPU tensor operations (Metal/CUDA/CPU) |
| `candle-nn` | 0.10 | Neural network layers (Linear, Embedding) |
| `tokenizers` | 0.22 | HuggingFace tokenizer for calibration text |
| `sysinfo` | 0.32 | Hardware detection (CPU, memory, disk) |
| `tracing` | 0.1 | Structured logging |
| `ctrlc` | 3.4 | SIGINT handling for clean shutdown |
| `libc` | 0.2 | File locking (unix) |
| `ruvector-core` | 2.1 | Vector similarity search (optional, feature `ruvector`) |

### Feature Flags

| Feature | Effect |
|---------|--------|
| `metal` | Enable Metal GPU via candle-core |
| `cuda` | Enable CUDA GPU via candle-core |
| `ruvector` | Enable vector similarity in RuVector (adds ruvector-core dep) |


## Future Work

From ADR-003 phased plan:

- **Phase 3 (Intelligence)**: Full auto-quant with per-layer sensitivity profiling
  from forward passes, RuVector learning from quality outcomes over time,
  `--quant auto` achieving 90%+ user acceptance without manual tuning.

- **Phase 4 (Advanced Formats)**: GPTQ and AWQ output backends for vLLM
  optimized CUDA kernel paths. Conditional on benchmarks showing meaningful
  speed/accuracy improvement over generic quantized safetensors.

- **Ecosystem**: Quantization manifest sidecar for reproducible builds,
  HuggingFace model card generation, benchmark suite across reference models,
  plugin architecture for community backends.
