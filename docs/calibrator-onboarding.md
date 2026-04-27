# Calibrator Onboarding

Developer guide for adding a new `Calibrator` implementation to hf2q.
Targets contributors writing the next imatrix-class or DWQ-class
algorithm (AWQ, GPTQ, AQLM, …) on top of the orthogonal
`Calibrator × OutputFormat × Quantizer` split landed under ADR-014 P7.

If you are an end user picking between existing variants, you want
`docs/converting-a-model.md` and `docs/converting-qwen35.md` instead.

---

## Why the `Calibrator` trait exists

Pre-ADR-014, calibration logic was tangled with the static quantizer
hierarchy in `src/quantize/`. Each quantizer carried its own optional
calibration step bundled with its codebook search. ADR-014 P7
(Decision 9) splits these axes into three orthogonal traits:

- **`Calibrator`** (this trait, `src/calibrate/calibrator.rs`):
  produces per-tensor calibration data from a model + corpus. Examples:
  `NoneCalibrator`, `ImatrixCalibrator`, `DwqCalibrator`.
- **`OutputFormat`** (`src/quantize/output_format.rs` family — flat,
  K-quant, bit-pair, K-quant-adaptive): the on-disk codec.
- **`Quantizer`** (`src/quantize/`): the per-tensor codebook search
  that consumes `(weights, CalibrationData)` and produces packed bytes.

The split lets `(Calibrator, OutputFormat)` compose orthogonally — every
diagonal cell is exposed via a single `--quant` variant (the 17-variant
menu); off-diagonal cells are reachable via the orthogonal
`--calibration X --output-format Y` flag pair gated on
`HF2Q_UNSAFE_EXPERIMENTS=1`.

The win for new Calibrator implementations: an algorithm only needs to
emit `CalibrationData`. The downstream codec (`KQuantCodecQuantizer`
/ `VariantKQuantizer` / DWQ byte-emit / future codecs) consumes the
data through a typed enum surface; the new algorithm does not need
to touch the codec.

---

## Trait surface

The full surface lives in `src/calibrate/calibrator.rs`. Reproduced
here for context (read the source for the authoritative version).

```rust
pub trait Calibrator: Send + Sync {
    /// Human-readable calibrator name.
    /// Used by logs, the CLI variant resolver, and the imatrix
    /// sidecar's dataset metadata field.
    fn name(&self) -> &'static str;

    /// Whether this calibrator needs a forward pass through the model.
    /// `false` → no-op calibrator (`None`).
    /// `true`  → reads activations during forward (`Imatrix`, `Dwq`).
    fn requires_forward_pass(&self) -> bool;

    /// Run the calibration. Exact behaviour depends on the implementor.
    fn calibrate(
        &mut self,
        model: &crate::ir::lazy::LazyTensorMap,
        meta: &crate::ir::ModelMetadata,
        corpus: &CalibrationCorpus,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<CalibrationData, CalibrationError>;
}
```

**Bounds**: `Send + Sync`. Calibrators may be invoked from rayon worker
threads under future P3-style parallel calibration loops; the trait
bound documents the contract upfront.

**Object safety**: the trait is object-safe. `Box<dyn Calibrator>` is
the runtime type used by `cmd_convert::select_calibrator`. The
`calibrator_is_object_safe` and `calibrator_is_send_sync` tests in
`src/calibrate/calibrator.rs` lock both properties.

---

## `CalibrationData` enum

```rust
pub enum CalibrationData {
    /// No calibration was performed — the quantizer falls back to
    /// unweighted MSE / round-to-nearest.
    None,

    /// Per-tensor importance vectors (output of
    /// ImatrixCollector::finalise). Each Vec<f32> has length
    /// `row_size` (dense) or `row_size × n_experts` (MoE,
    /// expert-major).
    Imatrix(HashMap<String, Vec<f32>>),

    /// Per-tensor importance + raw stats (preserves Stats { values,
    /// counts } for round-trip through the GGUF format which encodes
    /// per-expert counts exactly, unlike the legacy format's lossy
    /// collapse).
    ImatrixWithStats(HashMap<String, ImatrixStats>),

    /// DWQ sensitivity scores — per-layer scalar importance that
    /// drives the bit-pair allocation in DwqQuantizer. Each entry's
    /// Vec<f32> is a single value (the layer's sensitivity score)
    /// wrapped for uniform shape with the imatrix variant.
    Dwq(HashMap<String, Vec<f32>>),
}
```

When designing a new variant, first check whether one of the four
existing ones fits:

- **Per-column importance vector** (length = `row_size` for dense or
  `row_size × n_experts` for MoE) → reuse `Imatrix` or
  `ImatrixWithStats`. The codec already routes these through
  `quantize_row_q*_k_imatrix` for K-quant, and the bridge
  `CalibrationData::from_imatrix_gguf` already exists.
- **Per-layer scalar importance** (one value per transformer block) →
  reuse `Dwq`. The downstream byte-emit consumes the per-layer scores
  via `dwq_calibration_to_sensitive_ranges` to produce the
  base-vs-sensitive split.
- **Anything else** → add a new variant. The match in
  `quantize_row_to_bytes` and the `CalibrationData::is_some` /
  `len` / `is_empty` predicates fail the build at any new variant
  addition (no `_` arms — Decision 12 lock); this is the intended
  forcing function so new variants get explicit downstream handling.

`ImatrixWithStats` exists because the GGUF v3 imatrix format (llama.cpp
PR #9400 / commit `90083283` / 2025-07-19) preserves per-expert counts
exactly, where the legacy format collapses them. New algorithms that
produce per-tensor data with per-MoE-expert resolution should follow
the same pattern (a `WithStats` variant carrying the raw
counts/auxiliary fields).

---

## `CalibrationError` variants

The 5 typed variants in `src/calibrate/calibrator.rs`:

```rust
pub enum CalibrationError {
    /// Architecture has no forward driver wired into the calibrator.
    /// E.g. ImatrixCalibrator on an arch where
    /// RealActivationCapture::run_calibration_prompt is not yet ported.
    /// Used by select_calibrator to refuse silent NoneCalibrator
    /// fallback when the user asked for imatrix-* / dwq-*.
    ForwardPassUnavailable { arch: String },

    /// CalibrationCorpus.is_empty() returned true.
    /// A real algorithm needs tokens to produce a real signal.
    EmptyCorpus,

    /// Wraps the imatrix-algorithm-specific errors (ImatrixError) under
    /// the uniform CalibrationError surface for the calibrator dispatch.
    Imatrix(ImatrixError),

    /// Wraps std::io::Error for cache I/O, intermediate-GGUF emit, etc.
    Io(std::io::Error),

    /// Free-form fallback for "shouldn't happen but I want a typed
    /// error". Use sparingly — prefer adding a new typed variant.
    Other { message: String },
}
```

When implementing a new calibrator, **never wrap a precondition
violation in `Other { message }` if a typed variant fits**. The
`ForwardPassUnavailable` and `EmptyCorpus` variants exist precisely so
callers can route on cause without string-matching. If your algorithm
introduces a new precondition (e.g. "needs a Llama-arch model"),
add a new typed variant in the same PR rather than smuggle it through
`Other`.

---

## Step-by-step: add a new Calibrator

### 1. Implement the `Calibrator` trait

Place the impl in `src/calibrate/<name>_calibrator.rs`. Reference impls
to study:

- `src/calibrate/calibrator.rs::NoneCalibrator` — minimal trivially-
  correct impl. Truly no-op `calibrate` returns `Ok(CalibrationData::None)`.
- `src/calibrate/imatrix_calibrator.rs::ImatrixCalibrator` — wraps
  `ImatrixCollector` + a forward-pass driver
  (`RealActivationCapture` for qwen35 / qwen35moe). Validates the
  per-tensor shapes against the model's `num_layers × hidden_size`
  upfront so a mismatch surfaces as a typed error, not silent
  corruption.
- `src/calibrate/dwq_calibrator.rs::DwqCalibrator` — wraps the DWQ
  activation capture + per-layer sensitivity scoring. Demonstrates the
  `with_activation_capture` deferred-build constructor pattern used
  when the forward driver needs a freshly-emitted intermediate GGUF.

### 2. Decide the `CalibrationData` variant

Reuse `Imatrix` or `Dwq` when the shape fits (see the enum section
above). If you must add a new variant:

- Add the variant to `pub enum CalibrationData` in
  `src/calibrate/calibrator.rs`.
- Update the four `match self` arms in `is_some`, `len`, `is_empty`,
  and any other predicate methods — the no-`_` lock will surface every
  call site at build time.
- Update `KQuantCodecQuantizer` / `VariantKQuantizer` /
  `DwqKQuantizer` — every codec that consumes `CalibrationData` matches
  exhaustively. This is intentional (Decision 12 forcing function); do
  not add `_ =>` arms to silence the build.
- Update `CalibrationData::from_imatrix_gguf` /
  `from_imatrix_collector` if your variant has a corresponding GGUF
  schema bridge.

### 3. Wire into `cmd_convert::select_calibrator`

`src/main.rs::select_calibrator` is the dispatcher. The match is
**exhaustive** (no `_` arm — adding a new `cli::QuantMethod` variant
without updating this dispatch fails the build, which is the intended
Decision-12 lock).

The signature:

```rust
fn select_calibrator(
    method: cli::QuantMethod,
    dwq_arch: calibrate::dwq::DwqArch,
    capture: CaptureSpec,
    base_bits: u8,
    sensitive_bits: u8,
    calibration_samples: u32,
    num_layers: u32,
    hidden_size: u32,
) -> Result<Box<dyn calibrate::calibrator::Calibrator>, CalibrationError>;
```

Two call sites: a **diagnostic preview** early in `cmd_convert` (logs
the calibrator selection without attaching capture) and the **live
dispatch** at the per-quantizer match arm. The `capture: CaptureSpec`
lifecycle is asymmetric:

- Preview call: passes `CaptureSpec::None`. Imatrix / DWQ variants
  surface `ForwardPassUnavailable` here — that's expected; the preview
  logs and continues.
- Live call: passes the freshly-constructed capture
  (`CaptureSpec::Eager(c)` for the in-memory path or
  `CaptureSpec::Lazy { tokenizer }` for the deferred lazy build path
  used by qwen35 DWQ).

A new calibrator that requires a forward pass adds its own arm to the
match and follows the existing `dwq_*` / `imatrix_*` pattern. A
no-forward-pass variant slots into the `Auto | F16 | Bf16 | Q2 | Q4 |
Q8 | Q4KM | Q5KM | Q6K` arm (today's `NoneCalibrator` cell).

### 4. Add a CLI variant (or piggyback on `--quant`)

Two paths:

**Add a `--quant` variant** (recommended for "validated" cells the
17-variant menu should expose):

- Add the variant to `pub enum QuantMethod` in `src/cli.rs`. Use the
  `#[value(name = "...")]` clap attribute for kebab-case names.
- Update the `Display` impl, `dwq_bit_pair`, `is_quantized`,
  `default_filename_suffix` methods.
- Add the resolution arm in
  `src/cli.rs::resolve_convert_config`.
- Add the dispatch arm in
  `src/main.rs::select_calibrator` and the per-quantizer match in
  `cmd_convert`.
- Update the `QuantMethod` doc-table at
  `src/cli.rs::QuantMethod` so the surface table is one-source-of-truth.

**Pair the existing orthogonal flags** (`--calibration X
--output-format Y`) — already wired for off-diagonal cells via the
`CalibrationFlag` and `OutputFormatFlag` enums. Add the variant to
`CalibrationFlag` and let `validate_off_diagonal_selector` route it.
Off-diagonal cells stay behind `HF2Q_UNSAFE_EXPERIMENTS=1` until the
"diagonal table" (`is_diagonal_cell` predicate) is widened to include
the new pair.

### 5. Add tests

Two layers, both required by the file-fence convention:

- **Always-on smoke** in `src/calibrate/<name>_calibrator.rs::tests`:
  trait properties (`name()`, `requires_forward_pass()`, object-safety,
  `Send + Sync`), corpus-emptiness rejection, error-discriminant
  routing. The pre-Iter-A tests for `NoneCalibrator` at
  `src/calibrate/calibrator.rs:386-407` are the minimal shape.
- **`#[ignore]`-gated real-model** in
  `tests/<calibrator>_integration.rs`: end-to-end against a real
  fixture (or a synthetic tiny-model fixture if a real one is
  impractical). Reference: `tests/imatrix_xvalidation.rs` runs the
  cross-validation gate against `llama-imatrix` on a Qwen3.5-0.6B
  fixture; the always-on portion uses fixture roundtrips, the
  `#[ignore]`-gated cell needs the external binary.

Ignore reasons should name the missing prerequisite explicitly so a
human reading `cargo test -- --ignored` understands which cells to
flip on. Pattern from P10:

```rust
#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk + Qwen35Model::load_from_gguf for Variant::Moe"]
```

---

## Cross-validation pattern

When porting an algorithm from an external reference (llama.cpp's
`llama-imatrix`, mlx-lm's DWQ, …), provide a **byte-equivalent (with
documented float tolerance) cross-validation gate** so the port can be
re-validated whenever the reference moves.

The pattern landed by P6 close iter-1 (commit `0920f7c`,
`src/calibrate/imatrix_xvalidate.rs`):

1. **Reuse the production loader.** `cross_validate_imatrix_gguf`
   reads both inputs via the same
   `ImatrixCollector::load_imatrix_gguf` reader the runtime quantize
   loop uses. This is the load-bearing property: a regression in the
   loader breaks both production and the gate, not just one.
2. **Compare under typed tolerances.** Element-wise
   `max(abs(a - b))` (absolute) and `max(abs(a - b) / max(abs(a),
   abs(b), 1e-12))` (relative). The OR (not AND) on the gate
   predicate is deliberate: small-magnitude tensors saturate the
   absolute bound trivially; large-magnitude tensors saturate the
   relative bound. Either is sufficient to declare numeric equivalence.
3. **Exact-match auxiliary fields.** Counts (token counts) have no
   float-precision leeway; any mismatch is a real-error semantic, not
   a precision artefact. Surface as a typed `Invariant` variant.
4. **Tensor-set diff first.** Tensors present in only one side are
   port bugs, never precision artefacts. Surface verbatim.
5. **`not_measured` sentinel.** Mark un-run gates with a recognisable
   sentinel constructor that returns `is_pass() == false` so an
   `#[ignore]`-gated cell that wasn't run cannot be mistaken for a
   passing one.

Tolerance defaults for imatrix (`abs = 1e-3, rel = 1e-2`) are
justified by the round-trip RMSE bounds locked in P7 iter-3x/3y
(Q4_K ≤ 0.05, Q5_K ≤ 0.025, Q6_K ≤ 0.012). New algorithms should
document their tolerance choice with similar grounding — a pulled-
out-of-thin-air number is a future debugging trap.

---

## Cache integration

DWQ's per-layer sensitivity scores are stable across bit-pair variants
of the same model on the same corpus (only the downstream bit-allocation
table changes). ADR-014 P5 added a sensitivity cache so the second
`hf2q convert ... --quant dwq-4-8` after a `dwq-4-6` run on the same
model + same corpus skips the forward pass entirely.

The cache lives at
`${XDG_CACHE_HOME:-$HOME/.cache}/hf2q/sensitivity/<sha>.json`. Key
construction:

```rust
SensitivityCacheKey {
    model_fingerprint: <sha256 of metadata + tensor shapes/dtypes>,
    corpus_sha:        <sha256 of token chunks>,
    algorithm_version: SENSITIVITY_ALGORITHM_VERSION, // pinned const
}
```

The hash of the canonicalised triple is the on-disk filename. **Bump
`SENSITIVITY_ALGORITHM_VERSION`** (currently `"1.0.variance-magnitude"`)
whenever the algorithm changes — the bump invalidates every previously
cached entry by changing the key derivation.

A new calibrator that produces stable-across-variants per-layer scores
should integrate with the cache. Pattern from `DwqCalibrator`:

1. Build the cache key before the forward pass.
2. Call `cache::load(&key)`. On hit, decode the cached
   `Vec<CachedLayerSensitivity>` into the algorithm's internal form
   and skip the forward pass.
3. On miss (`Ok(None)`) or version mismatch (warn-and-recompute),
   run the forward pass.
4. Call `cache::save(&key, &scores)` after the forward pass. Atomic
   write via temp + POSIX rename — safe under concurrent processes.

The cache is **opt-in**: callers explicitly `load` before forward pass
and `save` after. Corrupt or schema-mismatched entries warn and
recompute without failing calibration (the "broken-cache must never
break the convert" invariant).

If a new calibrator's per-tensor data is **not** stable across
variants, do **not** integrate with the cache — the cache hit/miss
predicate should never produce wrong-data.

---

## Examples

### `ImatrixCalibrator` — algorithm + GGUF I/O

`src/calibrate/imatrix_calibrator.rs` (806 LOC) wraps:

- **Algorithm**: `ImatrixCollector` (from `src/calibrate/imatrix.rs`,
  the pure-Rust port of llama.cpp's `GGML_OP_MUL_MAT(_ID)`
  accumulator). At each forward pass through a calibration corpus,
  the collector captures the input activation vector at every Linear
  layer and accumulates `x[col]² · 1.0` per column. The mean of
  squared activations becomes the per-column importance weight.
- **Forward driver**: `RealActivationCapture` (from ADR-013) for
  qwen35 / qwen35moe; `gemma4/forward_cpu.rs` for Gemma-4. The arch
  dispatch happens inside the calibrator impl; the orchestration shell
  (the trait) is arch-agnostic.
- **GGUF I/O**: `ImatrixCollector::save_imatrix_gguf` /
  `load_imatrix_gguf` write+read llama.cpp's PR #9400 GGUF v3
  imatrix-file schema (commit `90083283`, 2025-07-19), preserving
  MoE per-expert counts exactly.

The calibrator's `calibrate(...)` produces
`CalibrationData::ImatrixWithStats(HashMap<String, ImatrixStats>)`,
which the codec (`KQuantCodecQuantizer`) consumes via
`quantize_row_q*_k_imatrix_to_bytes` for the per-element-weighted
codebook search.

### `DwqCalibrator` — with optional activation capture deferred path

`src/calibrate/dwq_calibrator.rs` (841 LOC) wraps:

- **Algorithm**: per-layer sensitivity scoring from
  `src/calibrate/sensitivity.rs::compute_layer_sensitivity`. Today's
  formula: `sqrt(variance) * log2(1 + max_magnitude)` per layer
  (algorithm version `1.0.variance-magnitude`).
- **Activation capture**: `RealActivationCapture` for qwen35 /
  qwen35moe — same forward driver as ImatrixCalibrator. The
  **deferred-build path** (`with_activation_capture_lazy`) accepts a
  `LazyTensorMap` + tokenizer and constructs the capture inside
  `calibrate(...)` rather than upfront, so the `cmd_convert` Phase 2
  dispatch can drop the resident tensor map and rebuild it after
  capture (memory-pressure mitigation under the streaming pipeline).
- **Cache integration**: load before forward pass, save after. Cache
  keying via `model_fingerprint + corpus_sha + SENSITIVITY_ALGORITHM_VERSION`
  per the §"Cache integration" pattern above.
- **Output**: `CalibrationData::Dwq(HashMap<String, Vec<f32>>)` where
  each entry is `blk.<i>.sensitivity → vec![score]`. The downstream
  byte-emit consumes the per-layer flag map via
  `dwq_calibration_to_sensitive_ranges` (`src/main.rs`) which
  coalesces adjacent sensitive indices into the
  `Vec<RangeInclusive<usize>>` shape that `DwqKQuantizer` /
  `MixedBitQuantizer` expect for the base-vs-sensitive split.

The DWQ algorithm produces a **scalar per layer**, not a per-column
vector — the `Dwq` variant's `Vec<f32>` is single-element by design.
This shape is dimensionally incompatible with the K-quant codec's
imatrix-weighted search, which expects per-column importance; the
bridging decision (whether to broadcast the scalar uniformly,
degenerating to `_ref`, or to run a separate imatrix calibration
alongside DWQ) is documented in
`src/quantize/dwq_k_quantizer.rs` "Sensitivity → Imatrix" and is
deferred to a future iter — `DwqKQuantizer::new` accepts an
`Option<CalibrationData>` constructor parameter so the bridge can
land without churning the API.

---

## References

- `src/calibrate/calibrator.rs` — trait definition + `NoneCalibrator`
  + `CalibrationData` + `CalibrationError` + 7 unit tests covering
  trait properties + corpus introspection + GGUF round-trip bridge.
- `src/calibrate/imatrix_calibrator.rs` — reference impl (forward-pass
  required; GGUF I/O bridge).
- `src/calibrate/dwq_calibrator.rs` — reference impl (forward-pass
  required; cache integration; deferred-build constructor).
- `src/calibrate/imatrix.rs` — algorithm core (`ImatrixCollector`,
  `Stats`, GGUF v3 read/write per llama.cpp PR #9400).
- `src/calibrate/imatrix_xvalidate.rs` — cross-validation gate pattern
  (P6 close iter-1).
- `src/calibrate/cache.rs` — sensitivity-JSON cache (P5).
- `src/main.rs::select_calibrator` — runtime dispatcher (Decision-12
  exhaustive match).
- `src/cli.rs::QuantMethod` — 17-variant menu + `Display` +
  `dwq_bit_pair` + `default_filename_suffix`.
- `docs/converting-a-model.md` — end-user surface for the variants
  this trait emits.
- `docs/ADR-014-streaming-convert-pipeline.md` — the ADR that locked
  the orthogonal split (Decision 9) and the 17-variant menu
  (Decision 12).
