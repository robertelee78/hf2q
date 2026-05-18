# ADR-033: Unified Convert/Quant Pipeline — Single Quantizer, Single Tensor-Type Source-of-Truth, Incremental Writer

- **Status**: proposed
- **Date**: 2026-05-17
- **Deciders**: operator (robert@loveathome.us); claude (research + draft)
- **Tags**: convert, quantize, architecture, byte-parity, public-release, root-cause, no-fallback

## Context

### Why now

Operator directive 2026-05-17 (verbatim):
> "we're trying to get this repo ready for production use / public release —
> we keep going down different research/refactor rabbit holes because
> everything we test is radioactive dogshit"

Every model we test surfaces a new internal bug. Five fixes this session
alone (`5dd2189a`, `e549906a`, `753e87ff`, `77489aaa`, `2b9b5a42`) plugged
five distinct seams in the convert/quant pipeline. Each fix was correct in
isolation, but the rate at which new seams surface is not converging.
This ADR records the design that *stops* surfacing new seams.

### What the current pipeline is doing (Chesterton's-fence inventory)

`hf2q convert --quant <method>` is a streaming HF→GGUF pipeline. The
streaming property is deliberate (no ~50 GB F16 intermediate on disk) and
must be preserved. What's broken is everything *inside* the stream.

**Five overlapping quantizer impls** (`src/quantize/`):

| File | LOC | What it does | Used by |
|---|---|---|---|
| `k_quant_codec_quantizer.rs` | 953 | Fixed-target K-quant codec (target set at construction) | Production: every `--quant q*_k_*` path via `main.rs:2167` |
| `variant_quantizer.rs` | 604 | Per-tensor target picking for K-quant variants | Only `--quant imatrix-adaptive` (`main.rs:2225`) |
| `dwq_k_quantizer.rs` | 883 | Sensitivity-based mixed-bit (base+sensitive) on K-quant codec | `--quant dwq-{p46,p48,p68,p28}` and `mixed-{2,3,4}-6` legacy aliases |
| `mixed.rs` | 520 | Legacy Q4_0-family mixed-bit (deprecated for K-quant) | DWQ legacy path behind `HF2Q_USE_LEGACY_DWQ_Q4_0=1` |
| `static_quant.rs` | 468 | Static-bit Q4/Q5/Q6/Q8 (legacy IR format) | Calibration scratch only; not a CLI surface |

These are functionally overlapping. K-quant codec, variant quantizer, and
DWQ quantizer ALL call `quantize_tensor_2d_to_bytes` internally; they
differ only in how they pick the target. Each has its own copy of the
vision-pattern F16-passthrough check; each has its own misalignment
fallback path. **This is the root of Bug-B-class issues**: when one
fallback policy diverges from the others, behavior depends on which
quantizer was constructed.

**Two-pass GGUF writer** (`src/backends/gguf.rs:282-1259`):

- Pass 1: iterate all tensors, predict byte size via `ggml_tensor_size`,
  accumulate `tensor_data_offset` table.
- Pass 2: repack each tensor's bytes via `repack_to_ggml_blocks`, write
  to its pass-1-predicted offset, zero-pad if actual bytes shorter.

The pass-1 size predictor is a `match ggml_type` table with a default
`_ =>` arm. Any ggml_type missing from the table over-predicts → pass-2
zero-pads → silently inflates the file. This is the iter-99 bug class
(K-quants) and its 2026-05-17 sequel (legacy 32-aligned types
Q4_1/Q5_0/Q5_1, fixed in `2b9b5a42`). The bug pattern recurs every time a
new ggml_type lands in any emit path.

**Seven-field `TensorQuantInfo`** (`src/ir/mod.rs:603`):

```rust
pub struct TensorQuantInfo {
    pub method: String,           // "q4", "f16", "passthrough", "k-quant-codec-direct"
    pub bits: u8,                  // 0 (sentinel), 2, 4, 6, 8, 16
    pub group_size: usize,         // 0 for k-quants
    pub preserved: bool,           // true if F16 passthrough
    pub scales: Option<Vec<u8>>,   // legacy quantizers only
    pub biases: Option<Vec<u8>>,   // legacy quantizers only
    pub ggml_type: Option<String>, // codec-direct's target
}
```

Seven fields. Multiple combinations describe the SAME tensor:
- `method="k-quant-codec-direct" + bits=0 + ggml_type=Some("Q5_K")` → write as Q5_K
- `method="f16" + bits=16 + preserved=true + ggml_type=Some("F16")` → write as F16
- `method="q4" + bits=4 + scales=Some(...) + biases=Some(...)` → repack as Q4_0

The writer (`repack_to_ggml_blocks`) and the validator
(`fn validate`) read DIFFERENT fields to decide the same question.
Bug 753e87ff was exactly this: `validate` checked `bits` and printed
"will fall back to F16" warnings for codec-direct tensors that the
writer correctly emitted as Q5_K.

llama.cpp's equivalent: ONE field. `tensor->type: ggml_type`.

### llama.cpp's pipeline (the parity target)

llama.cpp does this in TWO stages, with a well-defined intermediate:

1. `convert_hf_to_gguf.py` (260 LOC Python). HF safetensors →
   F16/F32/BF16/Q8_0 GGUF. **No K-quants**. Output is a contract: a
   known-good GGUF with original-precision weights.
2. `llama-quantize` (`tools/quantize/quantize.cpp` + `src/llama-quant.cpp`,
   ~91 KB C++). F16 GGUF → target-quant GGUF.

The `llama-quantize` core is THREE functions:
- `tensor_get_category` (~50 LOC): classify tensor by name. 11 categories: `TOKEN_EMBD`, `ATTENTION_Q/K/V/QKV/KV_B/OUTPUT`, `FFN_UP/GATE/DOWN`, `OUTPUT`, `OTHER`.
- `llama_tensor_get_type_impl` (~250 LOC at `llama-quant.cpp:411-657`): pick target ggml_type given `(ftype, category, model_arch, n_layers, i_layer, n_expert, n_gqa, has_imatrix)`. Big switch.
- `tensor_type_fallback` (~50 LOC at `llama-quant.cpp:362-408`): shape-misalignment policy. Q2/3_K→Q4_0, Q4_K→Q5_0, Q5_K→Q5_1, Q6_K→Q8_0, IQ_X→IQ4_NL, F16 last resort.

Quant execution is `ggml_quantize_chunk(target_type, f32_data, new_data,
first_row*n_per_row, this_nrow, n_per_row, imatrix)` — ONE function,
ONE reference implementation (`ggml-quants.c`). The writer is
**incremental**: each tensor is `gguf_set_tensor_type` +
`gguf_set_tensor_data` + `fout.write(...)` + zero-pad to 32-byte
alignment as it's processed. No pre-allocated offset table.

### hf2q's variants and how they map

Operator-clarified naming (2026-05-17):

| hf2q variant | What it actually does | Current name | Day-1 name |
|---|---|---|---|
| llama.cpp-policy K-quants | per-tensor mix of Q*_K, Q6_K, Q8_0 bumps per `llama_tensor_get_type_impl` | `--quant q5_k_m` etc. (BUT current impl uses FIXED target, not per-tensor mix) | `--quant q5_k_m` (FIXED — must match llama.cpp output byte-for-byte) |
| APEX (hf2q value-add) | mostly base-Q + misalignment fallback. Existing APEX-ara: ALL Q6_K + Q8_0/Q5_1/IQ4_NL ffn_down. **ZERO Q5_K tensors** despite "Q5_K_M" filename. | `--quant apex` (REMOVED in ADR-014 P8); existing files are `*-APEX-Q5_K_M.gguf` (misleading suffix) | `--quant apex-q6k` / `apex-q5km` / `apex-q4km` family. Each picks a base-Q + misalignment downshift only. Name reflects what's underneath. |
| Imatrix-calibrated K-quants | llama.cpp policy + per-element importance weights from calibration corpus | `--quant imatrix-q5_k_m` etc. | `--quant q5_k_m --imatrix <path>` (calibration is a modifier, not a quant method) |
| Sensitivity-based mixed-bit | calibration data identifies "sensitive" layers → higher precision; others → base | `--quant dwq-p46/p48/p68/p28` (MISLABELED as DWQ) | `--variant sensitivity-mixed --quant <base> --sensitive <sensitive>`. Frees the `dwq` name. |
| Real DWQ (distillation) | Apple-MLX-style: quantize + fine-tune (distill) the quantized weights from FP teacher. **NOT YET IMPLEMENTED.** | n/a | `--variant dwq-distill` (aspirational, post-v1) |

The current `--quant q5_k_m` is wrong twice over: (a) it produces all-Q5_K
instead of llama.cpp's per-tensor mix (the public-release blocker), and
(b) it labels the APEX files as `Q5_K_M` even though they contain zero
Q5_K. Both fixed by this ADR.

## Decision

Replace the current 5-quantizer + 2-pass-writer + 7-field-quant_info
architecture with a unified design preserving the streaming property.

### 1. Single source of truth for tensor type

```rust
// src/ir/quantized_tensor.rs (new)
pub struct QuantizedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub original_dtype: DType,
    pub data: Arc<Vec<u8>>,
    pub ggml_type: GgmlType,  // ONE field. Bytes in `data` are encoded as this type.
}
```

`GgmlType` is a Rust enum mirroring `ggml_type` (C enum). Round-trips
through GGUF's `u32` wire format via `Into<u32>` / `TryFrom<u32>`.

Delete: `method`, `bits`, `group_size`, `preserved`, `scales`, `biases`,
`ggml_type: Option<String>`. No multi-field disambiguation.

### 2. Unified `Quantizer` trait

```rust
// src/quantize/quantizer.rs (new)
pub trait Quantizer: Send + Sync {
    /// Quantize one tensor row-chunk to its target GGML type.
    /// `f32_data` is the dequantized source (length `n_rows * n_per_row`).
    /// `target` is the ggml_type to encode to.
    /// `imatrix` is per-element importance weights (length `n_per_row`),
    /// or None for uncalibrated.
    /// Returns final GGUF block bytes.
    fn quantize_chunk(
        &self,
        f32_data: &[f32],
        n_rows: usize,
        n_per_row: usize,
        target: GgmlType,
        imatrix: Option<&[f32]>,
    ) -> Result<Vec<u8>, QuantizeError>;
}
```

Signature mirrors llama.cpp's `ggml_quantize_chunk` almost
byte-for-byte. Pure-Rust impl in `src/quantize/ggml_quants/` (port of
`ggml-quants.c`). **Per operator directive: no FFI, no shell-out. 100%
Rust full stop.** Byte-parity is a gate; pure-Rust kernels must
bit-match `ggml-quants.c` exactly.

### 3. Unified `QuantPolicy` trait — per-tensor target picker

```rust
// src/quantize/policy.rs (new)
pub trait QuantPolicy: Send {
    /// Pick the target ggml_type for one tensor.
    /// `category` is the tensor's role; `position` carries per-arch counters
    /// (i_attention_wv, i_ffn_down, n_expert, n_gqa, etc.) updated by the
    /// caller as tensors are streamed in deterministic order.
    fn target_for(
        &mut self,
        tensor: &TensorRef,
        category: TensorCategory,
        position: &mut TensorPosition,
        ftype: LlamaFtype,
        imatrix_available: bool,
    ) -> GgmlType;

    /// Apply the shape-misalignment downshift table.
    /// Default impl mirrors llama.cpp's tensor_type_fallback exactly.
    fn shape_fallback(&self, target: GgmlType, ncols: usize) -> GgmlType { /* ... */ }
}
```

Three concrete impls on day 1:

| Impl | What it does | Output |
|---|---|---|
| `StandardPolicy` | Byte-for-byte port of `llama_tensor_get_type_impl` from `llama-quant.cpp:411-657` | Same per-tensor target as `llama-quantize` |
| `ApexPolicy { base_q: GgmlType }` | All `attn_*`, `ffn_up`, `ffn_gate`, `tok_embd`, `output` → `base_q`. `ffn_down` → `base_q` if aligned, else `shape_fallback`. | `apex-q6k`, `apex-q5km`, `apex-q4km` reproduce existing APEX-ara mix when `base_q = Q6_K`, plus apex-Q5K and apex-Q4K variants |
| `SensitivityMixedPolicy` | Read calibration data; sensitive layers → `sensitive_q`, base layers → `base_q` | Replaces `DwqKQuantizer::P46/P48/P68/P28`. Renamed: `dwq` name is freed for the future distillation impl. |

Future aspirational: `DwqDistillPolicy` (Apple-MLX-style; quantize +
fine-tune). Not in scope for this ADR.

### 4. Tensor categorization

```rust
// src/quantize/category.rs (new)
pub enum TensorCategory {
    TokenEmbd, AttentionQ, AttentionV, AttentionK, AttentionQkv, AttentionKvB,
    AttentionOutput, FfnUp, FfnGate, FfnDown, Output, Other,
}

pub fn categorize(name: &str) -> TensorCategory { /* substring match per llama-quant.cpp:115-150 */ }
```

Direct port of `llama-quant.cpp::tensor_get_category`. 11 categories.

### 5. Incremental writer

```rust
// src/backends/gguf_writer.rs (new — replaces gguf.rs:282-1259 pass1/pass2 logic)
pub fn write_gguf_streaming(
    out: &mut (impl Write + Seek),
    metadata: &Metadata,
    tensor_stream: impl Iterator<Item = StreamingTensor>,
) -> Result<()>;

// Tensor metadata header is written incrementally as tensors arrive.
// No pre-computed offset table. Each tensor:
//   1. write its bytes
//   2. zero-pad to 32-byte alignment
//   3. update header (or, since GGUF spec needs offsets up front, use a
//      single-pass-with-seek pattern: reserve header space → write all
//      tensor data → seek back → fill in the header)
```

Two viable shapes for "incremental" given GGUF's spec (offsets are in
the header at the top):

| Variant | Implementation | Pros | Cons |
|---|---|---|---|
| **Seek-back** | Reserve fixed header region (size = N * (max_meta_row_size)), write tensor data sequentially, seek back at end to fill in metadata | Trivially correct; single pass over tensors; no size prediction | Requires the output stream to support seek (file does; stdout doesn't — we never write to stdout) |
| **Two-pass header** | First pass: collect tensor metadata only (no quantize). Second pass: stream tensors. Header is written first with correct offsets. | Output stream-only-writable | Requires knowing each tensor's actual bytes in advance — same size-prediction problem we have today |

**Decision: seek-back.** Output is always a file in the GGUF backend.
The size-prediction class of bugs is eliminated by construction. The
seek-back pattern matches llama.cpp's `llama-quantize` behavior
(`ofstream::seekp` for the magic/version/kv-count region updates).

### 6. CLI surface

```text
# Standard quants (llama.cpp parity)
hf2q convert --quant {q4_0|q4_1|q5_0|q5_1|q8_0|q2_k_s|q2_k|q3_k_s|q3_k_m|q3_k_l|q4_k_s|q4_k_m|q5_k_s|q5_k_m|q6_k|iq4_nl}

# Optional imatrix calibration (modifier, not its own quant method)
hf2q convert --quant q5_k_m --imatrix path/to/imatrix.dat

# APEX family (hf2q-only; all-base + misalignment fallback)
hf2q convert --quant apex-q6k    # current APEX behavior (mostly Q6_K)
hf2q convert --quant apex-q5km   # mostly Q5_K + Q6_K bumps via llama.cpp's use_more_bits
hf2q convert --quant apex-q4km   # mostly Q4_K + Q6_K bumps via use_more_bits

# Sensitivity-mixed (renamed from DWQ; calibration-driven)
hf2q convert --quant q4_k_m --variant sensitivity-mixed --sensitive q6_k --calibration <path>

# Aspirational, post-v1:
# hf2q convert --quant q4_k_m --variant dwq-distill --teacher <hf_dir>
```

Removed: `--quant apex` (was already removed in ADR-014 P8), `--quant
imatrix-q5_k_m` (becomes `--quant q5_k_m --imatrix <path>`), `--quant
dwq-p46/p48/p68/p28` (becomes `--variant sensitivity-mixed --base
... --sensitive ...`), `--quant mixed-2-6/3-6/4-6` (subsumed by
sensitivity-mixed), `--quant imatrix-adaptive` (was APEX-equivalent; now
`--quant apex-q5km` or `apex-q4km` depending on bpw budget).

### 7. What gets deleted

| Path | Reason |
|---|---|
| `src/quantize/k_quant_codec_quantizer.rs` | Replaced by `Quantizer` trait + `StandardPolicy` |
| `src/quantize/variant_quantizer.rs` | Replaced by `Quantizer` trait + `StandardPolicy` (variant policy folds into `StandardPolicy::target_for`) |
| `src/quantize/dwq_k_quantizer.rs` | Replaced by `SensitivityMixedPolicy` |
| `src/quantize/mixed.rs` | Legacy Q4_0 affine companion path; deprecated for K-quants; only used by `HF2Q_USE_LEGACY_DWQ_Q4_0=1` escape hatch; the escape hatch goes away |
| `src/quantize/static_quant.rs` | Calibration scratch only; not a CLI surface; replaced by direct `Quantizer::quantize_chunk` calls from the calibration module |
| `src/quantize/layer_mix.rs::should_emit_f16_for_kquant` | Subsumed by `QuantPolicy::shape_fallback` |
| `src/backends/gguf.rs:282-1259` pass1/pass2 logic | Replaced by seek-back writer |
| `src/backends/gguf.rs::repack_to_ggml_blocks` | No more repack; bytes are emitted in target format directly |
| `src/backends/gguf.rs::ggml_tensor_size` | No more size prediction |
| `src/backends/gguf.rs::validate` warning loop (the "0-bit will fall back to F16" generator) | No more 7-field ambiguity to validate |
| `TensorQuantInfo` (7 fields) | Replaced by `ggml_type: GgmlType` |

Rough LOC delta: ~6500 LOC deleted, ~2000 LOC added net. Drops the
convert/quant code volume from ~30k → ~26k while eliminating the
overlap.

### 8. No-fallback enforcement

Every place in the new code that currently silently falls back to F16
on an unsupported case becomes a typed error (per the standing
no-fallback rule):

- `Quantizer::quantize_chunk` returns `Err(QuantizeError::UnsupportedTarget(t))` if it can't encode to `target`.
- `QuantPolicy::shape_fallback` panics if no fallback target exists for the input ggml_type (= bug in caller).
- The seek-back writer never zero-pads to a wrong size (no size prediction).
- Vision-tensor F16 is the ONLY place F16 is written for a non-F16/BF16-input tensor, and it's an explicit policy decision (`StandardPolicy::target_for` returns `F16` for `is_vision_tensor_pattern` names). This is intentional, not a fallback.

## Plan (GOAP)

Goal state: production-ready `hf2q convert` with byte-parity to
`convert_hf_to_gguf.py | llama-quantize` on standard quants, across the
public-release test matrix. No internal-bug noise on first contact.

Test matrix (public-release gate):
- Gemma 4 26B-A4B (MoE) — google base + APEX-ara
- Qwen 3.6 35B-A3B (MoE) — APEX variant + a standard-quant variant
- Vision: Gemma 4 mmproj + Qwen3VL
- BERT (standard embedding architecture)

Cutover: per operator directive 2026-05-17, "Replace + parity gate per
commit". Each commit deletes old + adds new + proves equivalence on the
applicable test fixtures. **No env-flag transitions** (no users yet —
backward compatibility is not required).

### Step P0 — Pure-Rust `ggml_quantize_chunk` byte-parity port

- **Pre**: `/opt/llama.cpp/ggml/src/ggml-quants.c` available as reference.
- **Effect**: `src/quantize/ggml_quants/{q4_0,q4_1,q5_0,q5_1,q8_0,q2_k,q3_k,q4_k,q5_k,q6_k,iq4_nl}.rs` produce bit-identical bytes to `ggml-quants.c` for arbitrary F32 input. Unit tests: for each type, quantize a fixed-seed PRNG buffer in hf2q + run llama.cpp's `quantize-stats` on the same buffer; `cmp` results.
- **Cost**: high. Each kernel needs careful port; the existing `src/quantize/k_quant.rs` (5541 LOC) is most of the work but needs byte-cmp gates added per-type.
- **Risk**: BF16→F32→F16 rounding paths differ from llama.cpp's `convert_hf_to_gguf.py` cast pipeline → bytes diverge even though kernels are bit-correct. Mitigation: cast through the same intermediate F32 representation llama.cpp uses (no fast-math, no FMA reordering).
- **Rollback**: this step is purely additive (new module, new tests). Existing quantizers untouched.

### Step P1 — Unified `Quantizer` trait + `StandardPolicy` (llama.cpp parity)

- **Pre**: P0 complete; per-type byte-cmp tests green.
- **Effect**: `src/quantize/quantizer.rs` (trait), `src/quantize/policy.rs` (`StandardPolicy`, `tensor_get_category`, counters), `src/quantize/ggml_quants/mod.rs` (`Quantizer` impl that dispatches per target).
- **Gate**: produce a Q5_K_M of Gemma 4 26B-A4B via the new path; `cmp` against `convert_hf_to_gguf.py | llama-quantize -Q5_K_M` output. Must be byte-identical. Same for Q4_K_M, Q6_K, Q8_0, IQ4_NL.
- **Cost**: medium. Trait + policy ~500 LOC; per-tensor counters and edge cases (n_expert=8, falcon, 70B-attn-v) need careful port.
- **Risk**: subtle policy divergence (e.g., we miscount `i_attention_wv` because our tensor iteration order differs from llama.cpp's `weights_map`). Mitigation: dump (tensor_name, target_ggml_type) sequence from llama-quantize via `LLAMA_LOG_INFO`; assert identical sequence from `StandardPolicy`.
- **Rollback**: trait + policy + ggml_quants module are additive; old quantizers still exist.

### Step P2 — Seek-back incremental writer

- **Pre**: P1 produces tensors with `ggml_type: GgmlType` (post-`TensorQuantInfo` collapse, even if a thin compat layer translates old struct to new).
- **Effect**: `src/backends/gguf_writer.rs::write_gguf_streaming`. Reserve header region → write tensors sequentially → seek back to fill header offsets.
- **Gate**: write the same Gemma 4 Q5_K_M from P1 through the new writer; cmp against llama.cpp output (and against the old writer's output for hf2q-only variants).
- **Cost**: medium. ~800 LOC of writer code.
- **Risk**: GGUF spec corner cases (KV metadata with variable-length strings, split files via `--split-max-size`). Mitigation: incremental landing per metadata feature; test against a tiny model first (1B dense), then scale.
- **Rollback**: new writer behind no env flag, but in a new module path. Old `gguf.rs::write` stays until P2's gate is green, then deleted in same commit.

### Step P3 — Collapse `TensorQuantInfo` to single `ggml_type` field

- **Pre**: P1 + P2 green. New `Quantizer` and new writer can both round-trip through a single-field representation.
- **Effect**: drop the 7-field struct; replace with `ggml_type: GgmlType` on `QuantizedTensor`. Wire through every read-site (~40 call sites per grep).
- **Gate**: 358 quantize tests + 357 backend tests + parity matrix all green.
- **Cost**: medium. Mostly mechanical refactor, but each call site needs to be checked for "did I depend on the bits/method/preserved/scales/biases combination here?"
- **Risk**: hidden caller dependency on `quant_info.scales` for legacy companion-schema (DWQ-Q4_0 path). Mitigation: that path is the `HF2Q_USE_LEGACY_DWQ_Q4_0=1` escape hatch which goes away (operator: no backward compat needed).

### Step P4 — `ApexPolicy` family (apex-q6k, apex-q5km, apex-q4km)

- **Pre**: P1-P3 green.
- **Effect**: `ApexPolicy` implementing `QuantPolicy::target_for` per the table above. Three CLI options: `--quant apex-q6k|apex-q5km|apex-q4km`.
- **Gate**: produce `apex-q6k` of Gemma 4 26B-A4B; tensor-type-by-tensor-type diff against `gemma4-ara-2pass-APEX-Q5_K_M.gguf` (the existing reference). Must match category-by-category: all attn_* → Q6_K, all ffn_up/gate → Q6_K, ffn_down → {Q8_0, Q5_1, IQ4_NL} per misalignment, etc.
- **Cost**: low. ~150 LOC for the policy + 3 CLI strings.
- **Risk**: the existing APEX-ara files were produced with the legacy 2-pass pipeline; their exact byte-cmp against new pipeline output depends on whether bytes-per-tensor match. ppl-equivalence is the fallback gate.

### Step P5 — `SensitivityMixedPolicy` (renames current DWQ)

- **Pre**: P1-P4 green; calibration data plumbing reusable.
- **Effect**: rename `DwqKQuantizer` → `SensitivityMixedPolicy`. CLI moves from `--quant dwq-p46` to `--quant q4_k_m --variant sensitivity-mixed --sensitive q6_k --calibration <path>`.
- **Gate**: re-emit the four ADR-012 reference DWQ artifacts (`27B-dwq46/48`, `35BMOE-dwq46/48`); cmp against existing files. Functional-parity (ppl + coherent generation) is the gate per ADR-012's closure AC.
- **Cost**: medium. Calibration data ingestion needs to be re-shaped to feed `SensitivityMixedPolicy::target_for` instead of constructing a per-tensor codec.
- **Risk**: the existing DWQ artefacts are byte-frozen on disk; users (us) reference them by sha256. Rename in CLI ≠ rename in filename. Mitigation: the GGUF files keep their existing names; the CLI flag changes only.

### Step P6 — Delete old quantizers + 2-pass writer + 7-field struct

- **Pre**: all of P1-P5 green; no production path references the old code.
- **Effect**: delete `k_quant_codec_quantizer.rs`, `variant_quantizer.rs`, `dwq_k_quantizer.rs`, `mixed.rs`, `static_quant.rs`, the pass1/pass2/repack logic in `gguf.rs`, and the 7-field `TensorQuantInfo`. Wire `cmd_convert` to dispatch directly into the unified path.
- **Gate**: full test matrix green (Gemma 4 + Qwen 3.6 + Vision + BERT, standard + APEX + sensitivity-mixed variants).
- **Cost**: low (deletion is cheap; just verify nothing references the deleted code).
- **Risk**: a corner of the test harness silently depended on a deleted code path. Mitigation: incremental deletion (one file per commit) with full `cargo build --release` + `cargo test --release` between each.

### Step P7 — Public-release readiness sweep

- **Pre**: P0-P6 green.
- **Effect**: README, ADR-014 supersession entry pointing here, ADR-005 cross-link, docs/operator-env-vars.md cleanup (env flags removed: `HF2Q_USE_LEGACY_DWQ_Q4_0`, `HF2Q_STREAMING_PHASE3`, `HF2Q_STREAMING_PHASE3_MUT`, `HF2Q_DEBUG_GGUF_OFFSETS` if not still needed), CLI help strings, error messages aimed at first-time public users (not hf2q-internal terminology).
- **Gate**: a stranger can `hf2q convert --quant q5_k_m --input <hf dir> --output model.gguf` and get a working file without consulting source code.

### Risk factors (whole-ADR scope)

1. **Byte-parity may be impossible for some quants.** llama.cpp's
   `ggml-quants.c` has FMA-order-dependent rounding in some K-quant
   kernels; reproducing that in Rust without FFI is hard. Mitigation:
   record any bytes-differ tensor as a typed work item; ppl-parity is
   the fallback gate per operator directive ("Both — cmp is the gate,
   ppl is the fallback report"). The cmp report becomes the
   bug-list-to-fix.
2. **The streaming-with-seek-back writer fights GGUF v3's offset
   layout**, which expects all tensor info up front. Mitigation:
   reserve a generous header region (1 MB) and assert at end that
   actual header fits; if it doesn't, panic (typed error, not silent
   bloat).
3. **Imatrix calibration was special-cased in the old pipeline**
   (`ImatrixCalibrator` builds an in-memory imatrix from a corpus +
   eager-built `Qwen35Model`). Moving it to a `Quantizer` modifier
   means re-routing the calibration capture; not impossible but a
   discrete sub-project. Mitigation: P1's `Quantizer::quantize_chunk`
   signature already accepts `imatrix: Option<&[f32]>`; just need
   `cmd_convert` to pipe the calibration output into that argument.
4. **Test matrix doesn't yet include BERT** in our existing fixtures.
   Mitigation: add a small BERT fixture (e.g., `bge-large-en-v1.5`)
   pre-P1; treat its parity gate as a P7 deliverable.

### What we are explicitly NOT doing in this ADR

- Real DWQ (distillation-based, Apple-MLX-style). Aspirational; future ADR.
- libggml FFI or shell-out to `llama-quantize`. Operator: "100% rust full stop."
- F16 GGUF intermediate on disk. Operator: streaming is non-negotiable.
- Loosening the no-fallback rule. F16 stays for vision tensors (intentional policy) and as a typed-error sentinel only.

## Consequences

### Positive

- One source of truth for tensor type (`ggml_type: GgmlType`).
  Eliminates the 7-field ambiguity that produced Bug C (`753e87ff`).
- One per-tensor target policy. Adding a new variant means writing one
  `QuantPolicy` impl, not editing 3 quantizers + 1 writer + 1 size
  estimator.
- Incremental writer eliminates the iter-99 / Bug-B-sequel class of
  bugs by construction (no pre-allocated offset table, no size
  prediction).
- Public users get byte-parity Q5_K_M / Q4_K_M / Q6_K / Q8_0 / IQ4_NL.
  No more "hf2q's Q5_K_M is mostly Q5_K but bartowski's is mostly Q5_K
  + Q6_K bumps" surprises.
- APEX is correctly named (apex-q6k, apex-q5km, apex-q4km) — operators
  know exactly what they're getting.
- DWQ name is freed for the future real-distillation implementation.

### Negative

- ~6500 LOC deleted from a stable code path. Re-implementation must be
  bit-exact or we lose the existing DWQ-46/48/68 reference artifacts'
  byte-parity. Mitigation: P5's gate is ppl-equivalence (not byte) per
  operator, since the legacy DWQ pipeline isn't reproducible
  bit-for-bit anyway.
- Pure-Rust port of every `ggml_quantize_chunk` variant carries
  ongoing maintenance cost: every time ggml-quants.c adds a new
  type or fixes a kernel bug, we must port the change. Mitigation:
  formal byte-parity tests on a fixed-seed PRNG fixture as part of CI
  — any drift surfaces on the next `cargo test`.
- The seek-back writer pattern requires the output stream to support
  seek. Stdout, pipes, and HTTP uploads don't. We don't write convert
  output to any of those today, but we lose the option.

### Neutral

- The streaming convert property is preserved (no F16 intermediate on
  disk). The internal architecture is what changes, not the user-facing
  pipeline shape.
- The model-architecture-specific tensor mappers (`src/inference/models/<arch>/...`,
  `gguf-py/conversion/<arch>.py` analog) are unchanged. They feed
  tensors into the `Quantizer` rather than holding their own quant policy.
- ADR-014's "single-pass streaming" decision is preserved verbatim;
  this ADR refines the *internals* of that pipeline without disturbing
  the streaming property.

## Open questions for operator

1. **DWQ-distill reference.** Operator directed: "1 + 2 — plus search
   our code and memories more thoroughly" (1 = Apple MLX dwq.py;
   2 = original DWQ paper). The Apple MLX implementation needs to be
   pulled and read. Original paper needs to be located. These shape
   the FUTURE `DwqDistillPolicy` impl; not blocking for this ADR
   (which scopes only the rename of current DWQ → SensitivityMixed).
2. **BERT fixture choice.** What BERT model to add to the test matrix?
   `bge-large-en-v1.5` is the default community embedding model;
   alternative is `BAAI/bge-base-en-v1.5` (smaller, faster CI). Operator preference?
3. **Phase ordering preference.** Does operator want P0-P7 executed
   strictly in order (each phase fully complete before next), or
   parallel-on-paper with serial-on-commits (e.g., P1 and P2 are
   independent enough to be developed in parallel)?

## Links

- ADR-014 (streaming convert pipeline) — predecessor; this ADR refines its internals.
- ADR-005 (inference server) — convert output is the input to inference; parity gates must hold across both.
- ADR-012 (DWQ closure artifacts) — sensitivity-mixed gate uses these reference files.
- ADR-032 (root-cause Bug A and Bug B; ship best-outcome defaults) — companion ADR for inference-side fixes that surfaced the convert-side bugs this ADR addresses.
- `/opt/llama.cpp/convert_hf_to_gguf.py` (260 LOC; the convert reference).
- `/opt/llama.cpp/tools/quantize/quantize.cpp` + `/opt/llama.cpp/src/llama-quant.cpp` (the quantize reference).
- Auto-memory entries: `[[hf2q-convert-gemma4-f16-dispatch-2026-05-17]]`, `[[q5-k-q4-k-dequant-fixed-2026-05-17]]`, `[[feedback-codex-review-loop-2026-05-17]]`, `[[feedback-no-loop-suppression-2026-05-17]]`, `[[feedback-test-both-families-2026-05-17]]`.
