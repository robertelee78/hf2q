# ADR-033: Unified Convert/Quant Pipeline — Port llama.cpp + Real APEX, Single Source-of-Truth IR, Incremental Writer

- **Status**: SHIPPED 2026-05-19 — P-1..P6 Phase 1 + tokenizer + streaming + F32-keep + real-model validation all on main. Real Gemma 4 26B convert: 48GB safetensors → 18GB Q5_K_M GGUF in 8m 22s, loads in stock llama.cpp + decodes coherent reasoning at 111.5 t/s gen. Remaining: Pi (imatrix for I-tier APEX), §9 (per-model fingerprint manifest), Phase 2/3 retirement (operator decisions on B1/B2/B4)
- **Date**: 2026-05-18
- **Deciders**: operator (robert@loveathome.us); claude (interview + draft)
- **Tags**: convert, quantize, architecture, byte-parity, public-release, apex, mudler, imatrix
- **Supersedes**: ADR-014 (full supersession; streaming-convert property carried forward; the `--quant apex` CLI surface ADR-014 P8 D13 removed is reintroduced here with the correct semantics)
- **External pins** (load-bearing — byte-cmp gates assume these exact references):
  - `llama.cpp` @ `c779f6198` (operator's `/opt/llama.cpp` HEAD; the local branch with ADR-029 iter-57 instrumentation — NOT stock upstream)
  - `mudler/apex-quant` @ `63c5048b7dc9ff230f2397d7bc445ca28894b769` (GitHub main, 2026-05-17 14:42 UTC; the SHA we port from)
  - GGUF spec: v3 (matches `const GGUF_VERSION: u32 = 3` at `src/backends/gguf.rs:23`; matches `general.quantization_version = 2` in operator's existing APEX files)
  - rustc: pinned via `rust-toolchain.toml` (1.81.0 minimum per current `mlx-native` MSRV; P-1 verifies the project file pins a single version for byte-cmp determinism across developer machines)

## Context

### The problem

hf2q's convert/quant pipeline is currently a set of overlapping, partially-redundant subsystems that don't compose cleanly and don't reproduce any standard llama.cpp artifact byte-for-byte:

| Subsystem | LOC (HEAD) | Status |
|---|---|---|
| `quantize/k_quant_codec_quantizer.rs` | 953 | Production K-quant path |
| `quantize/variant_quantizer.rs` | 604 | Variant K-quant (imatrix-adaptive) |
| `quantize/dwq_k_quantizer.rs` | 883 | hf2q's homebrew DWQ |
| `quantize/mixed.rs` | 520 | Mixed-bit dispatcher |
| `quantize/static_quant.rs` | 468 | Static (non-K) quant |
| `quantize/mod.rs` / `k_quant.rs` / `k_quant_codec.rs` / `q_legacy.rs` / `layer_mix.rs` | 16,867 | Unclassified: kernels + policy + utility + dead all mixed |
| `backends/gguf.rs:282–1259` | ~977 | Two-pass GGUF writer (the iter-99 / Bug-B-sequel bug-class lives here) |
| `ir/mod.rs::TensorQuantInfo` | 7 fields | Carries `method | bits | preserved | scales | biases | ggml_type | …` simultaneously |
| 5 quantizer dispatch arms in `main.rs` (≈2160 / 2225 / 2400) | — | Each routes through a different policy with a different output shape |

The result of this fragmentation:

- **Every production model surfaces a new internal bug.** Five fixes shipped 2026-05-15..17 (`5dd2189a`, `e549906a`, `753e87ff`, `77489aaa`, `2b9b5a42`) plugged five distinct seams; rate is not converging.
- **No artifact hf2q produces can be byte-cmp'd against the canonical llama.cpp pipeline.** Our K-quant dispatch deviates from `llama-quantize`'s `llama_tensor_get_type_impl` in subtle ways the codebase doesn't enumerate.
- **The two "APEX" GGUF files the operator runs in production were produced externally, not by hf2q.** hf2q currently has no path to reproduce them from safetensors. The `gemma4-ara-2pass-APEX-Q5_K_M.gguf` and `qwen3.6/APEX-Q5_K_M.gguf` came from a separate toolchain on a separate machine.
- **`--quant apex` was removed in ADR-014 P8 Decision 13** because its semantics weren't well-defined. The 2026-05-17 ADR-033 draft (`ebecc21c`) reintroduced it but defined `ApexPolicy` as "pure base-Q + shape_fallback" — a definition that doesn't match any real APEX artifact and doesn't match `mudler/apex-quant`'s published behavior.

### What APEX actually is

Deep research 2026-05-17 (operator-led, with web/repo/HF reads) established APEX = `mudler/apex-quant`, an MoE-specific quantization toolkit on GitHub. Key facts (full reference in auto-memory `[[apex-quant-definition-2026-05-17]]`):

- **MoE-only.** Designed around 97%-sparse routed experts. Does not meaningfully apply to dense models.
- **Per-tensor-pattern overlay via stock `llama-quantize`'s `--tensor-type` / `--tensor-type-file` flags.** No custom llama.cpp patches.
- **Seven tiers:** I-Quality, Quality, I-Balanced, Balanced, I-Compact, Compact, Mini. The `I-` prefix variants use diverse-corpus imatrix calibration; the four non-I tiers do not.
- **Tensor classification by role:** routed-expert (tolerates Q4_K / IQ4_XS), shared-expert (needs Q8_0, kurtosis 13.10), attention (per-tier), token-embd / output (edge layers get Q6_K).
- **Layer-wise gradient:** edge layers (first / last 5 of 40-layer default; rescaled by `NUM_LAYERS` env var) get heavier quant; middle layers get lighter.
- **Quality tier matches F16 perplexity at ~⅓ the size** (Qwen3.5-35B-A3B: 6.527 vs 6.537 PPL at 21.3 GB vs 64.6 GB).

The 2026-05-17 ADR-033 draft conflated several unrelated concepts under the name "APEX." This rewrite separates them.

### What we want

Operator framing (2026-05-17 / 2026-05-18):

1. **"100% Rust full stop."** No FFI, no shell-out to llama-quantize, no Python subprocess. The whole convert+quant capability lives in hf2q.
2. **Reproduce the standard llama.cpp pipeline byte-for-byte.** `hf2q convert <hf-dir> --quant q5_k_m -o out.gguf` produces the same bytes as `convert_hf_to_gguf.py | llama-quantize -q5_k_m`.
3. **"Make our own APEX correctly."** Port `mudler/apex-quant`'s published recipe in pure Rust; reproduce the per-tier output for the supported MoE arches; couple with an in-tree imatrix generator for the I-tier variants.
4. **Streaming property preserved.** No intermediate F16 GGUF on disk (the ADR-014 invariant). safetensors → quantized GGUF in one in-memory pipeline.
5. **No-fallback rule.** F16 emit is allowed only for (a) vision-tensor patterns and (b) `--quant f16` explicit user request. Any other F16 emit is a typed error, not a silent demotion. (The 2026-05-17 draft promised this but had `shape_fallback` silently mirror llama.cpp's second-misalignment F16 path; this rewrite resolves the contradiction by making `shape_fallback` hard-error.)
6. **"Quality matters; mechanism doesn't."** The acceptance gates are byte-cmp against canonical references. How we enforce internal invariants (e.g., FMA ordering) is an implementation detail validated by the gates, not by the ADR.

## Decision

Collapse the five overlapping quantizer impls + two-pass writer + seven-field IR into:

1. **A single `QuantizedTensor` IR type** carrying only `{ ggml_type, data: Arc<Vec<u8>> }`. Fields beyond these two are added only when a proven need arises (safety-valve clause; not a license for re-bloat).
2. **A unified `Quantizer` trait** mirroring `ggml_quantize_chunk`'s signature; pure-Rust port of `ggml-quants.c`. No FFI.
3. **Two `QuantPolicy` impls** at v1 ship:
   - `StandardPolicy` — byte-for-byte port of `llama_tensor_get_type_impl` (covers `q4_0`/`q4_1`/`q5_0`/`q5_1`/`q4_k_m`/`q5_k_m`/`q6_k`/`q8_0`/`iq4_nl`/etc.; mirrors `tensor_type_fallback`'s first-downshift behavior).
   - `ApexPolicy` — pure-Rust port of `mudler/apex-quant`'s published recipe (7 tiers; MoE tensor classifier; `NUM_LAYERS`-aware layer gradient; per-tier regex → quant-type rules).
4. **`shape_fallback` contract:** every policy's `target_for(tensor)` returns `Result<GgmlType, QuantizeError>`. The first-downshift path succeeds; the second-misalignment case (where llama.cpp silently emits F16) returns a typed error. F16 is emitted only for vision-tensor patterns or `--quant f16`. No silent demotions anywhere. (The 2026-05-17 draft had a separate §8 "no-fallback enforcement" section; this rewrite rolls the enforcement into each policy's signature, so the contract is type-system-checked rather than narrative.)
5. **Seek-back incremental GGUF writer** — single pass; reserve a header region; stream tensor payloads to disk; seek back and fill the header. No pre-allocated offset table, no two-pass zero-padding. Eliminates the iter-99 / Bug-B-sequel bug class by construction. Single-file output only at v1 (`--split-max-size` is explicit non-goal; users can post-split with `llama-gguf-split`).
6. **MoE tensor classification for `ApexPolicy`:** combination of GGUF metadata introspection (`expert_count`, `expert_used_count`, arch-name) and `mudler/apex-quant`'s tensor-name regex tables ported verbatim. Per-arch classifier files live at `src/quantize/apex/<arch>.rs` for the supported set (qwen35moe, gemma4-MoE, MiniMax-M2.7's arch). Unsupported arches passed to `--quant apex-*` fail with a typed error naming the supported set.
7. **In-tree imatrix subsystem.** hf2q's existing forward-pass code (built on `mlx-native`'s Metal compute primitives) is reused: decode a calibration corpus, accumulate per-row importance, emit a `.imatrix.gguf` compatible with llama-imatrix's format. v1 ships both UX modes implicitly via flags: `--imatrix-corpus {cdv3,mudler,user-file}` auto-generates in-memory during convert; `--imatrix <file>` consumes a pre-made file. Default corpus when `--imatrix-corpus` is omitted on an I-tier: `cdv3` (bartowski's `calibration_datav3.txt`). Both corpora ship in `data/calibration/` alongside the binary (not embedded in the binary).
8. **CLI surface:**
   ```
   hf2q convert <hf-dir> --quant <name>
                          [--imatrix <file> | --imatrix-corpus {cdv3,mudler,user-file}]
                          [--tensor-type-file <file>]    # only for --quant apex-custom
                          [-o out.gguf]
   ```
   `<name>` ∈
   - StandardPolicy types: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q4_k_s`, `q4_k_m`, `q5_k_s`, `q5_k_m`, `q6_k`, `q8_0`, `iq4_nl`, `f16`, `f32`, `bf16`
   - ApexPolicy tiers (verified at mudler SHA `63c5048b`; mudler ships 12 algorithmic profile names — `quality, i-quality, balanced, i-balanced, compact, i-compact, mini, nano, i-nano, micro, i-micro, custom`; v1 drops the four experimental tiers `nano, i-nano, micro, i-micro`):
     - `apex-quality`, `apex-i-quality`
     - `apex-balanced`, `apex-i-balanced`
     - `apex-compact`, `apex-i-compact`
     - `apex-mini` (mudler "benefits from imatrix"; can be run with or without)
     - `apex-custom` — requires `--tensor-type-file <file>`; consumes operator-supplied per-tensor type overrides in mudler's `pattern=quant_type` line format
   - **Dropped from mudler's surface for v1:** `nano`, `i-nano`, `micro`, `i-micro` (all 4 labeled experimental upstream; target IQ2_XXS / IQ1_M / IQ2_S-class aggressive quants). Per-model configs for these tiers remain accessible via `--quant apex-custom --tensor-type-file data/apex-references/<model>_<nano|micro>.txt` (and similar for i-variants).
   - Reserved: `dwq` returns a typed `--quant dwq is reserved for the future real-DWQ ADR (Apple MLX dwq.py port)` error.
   - **No `apex` alias.** Tier must be spelled explicitly; ADR-014 P8 D13 removed the unqualified name because its meaning was ambiguous, and the same reason still applies.
   - **TQ1_0 / TQ2_0 (BitNet ternary) out of v1 scope.** Documented; tracked separately. `--quant tq1_0` returns a typed "out of v1 scope; see [tracking issue]" error.

### Per-model APEX config override (Decision §9 — silent auto-fingerprint)

Mudler ships `configs/<model>_<tier>.txt` per-model overrides alongside the algorithmic `scripts/generate_config.sh`. These vendored configs are hand-tuned for specific known models (e.g., `carnice_qwen36_mtp_quality.txt` matches the operator's qwen3.6 abliterix production model). v1 hf2q vendors them at `data/apex-references/<model>_<tier>.txt` and dispatches automatically:

- Compute a stable fingerprint over `(model_type, num_hidden_layers, hidden_size, num_experts, num_attention_heads, num_key_value_heads, intermediate_size, moe_intermediate_size)` from the source `config.json`.
- Check fingerprint against a `data/apex-references/manifest.json` table (built once at vendor time; maps fingerprint → reference config file).
- **If matched:** use the per-model config silently. The vendored config's rules win over the algorithmic generator's output.
- **If unmatched:** fall through to the algorithmic generator (`generate_config.sh` port).
- No CLI flag controls this; the fingerprint match is invisible to the user. (Trade-off acknowledged: surprising override risk; mitigated by `hf2q apex why <hf-dir>` debug subcommand that prints whether a fingerprint matched.)

The manifest is regenerated at vendor-time only (not at runtime); SHA-pinned to the mudler commit captured in `data/apex-references/MUDLER_SHA.txt`.

### FP8 source-dtype auto-detect (Decision §10 — silent)

When `config.json::quantization_config.quant_method == "fp8"` (per HuggingFace's standard quantization-config schema, used by MiniMax-M2.7 and others), hf2q convert auto-dequantizes the FP8 source to F32 in-memory before invoking the policy:

- Format: `float8_e4m3fn` (1-bit sign + 4-bit exponent + 3-bit mantissa, no inf, single NaN encoding).
- Layout: block-wise with `weight_block_size` field (e.g., `[128, 128]` for MiniMax-M2.7) — block-of-blocks scale factor stored alongside each tensor.
- Modules listed in `modules_to_not_convert` (e.g., `gate`, `e_score_correction_bias`, `lm_head`) are read as F32 / BF16 directly (no FP8 path).
- No CLI flag controls this; auto-detection is silent. (Trade-off: same as above; surprise risk mitigated by `hf2q convert --dry-run` flag that prints the resolved source dtype per tensor before quantizing.)

Source-dtype hard-error: if `quantization_config.quant_method` exists with a value other than `fp8` (e.g., `gptq`, `awq`), convert returns `ConvertError::UnsupportedSourceQuant { quant_method: String }`. No silent demotion to F32; supported source-quants are an explicit enum.

### Per-tensor IR (Decision §1 concrete)

Replaces `TensorQuantInfo`:

```rust
pub struct QuantizedTensor {
    pub ggml_type: GgmlType,    // enum mirroring llama.cpp's `ggml_type` for all wire values
    pub data: Arc<Vec<u8>>,     // packed block bytes
    // Add fields only if a proven need surfaces. Today, none.
}
```

`GgmlType` is a Rust enum spanning every `ggml_type` value llama.cpp writes to disk (block formats Q4_0..IQ4_NL plus F16/BF16/F32). Conversion to/from `u32` for header serialization is `From`/`TryFrom`.

### Quantizer trait (Decision §2 concrete)

```rust
pub trait Quantizer: Send + Sync {
    fn ggml_type(&self) -> GgmlType;
    fn quantize(
        &self,
        src: &[f32],
        n_per_row: usize,
        imatrix: Option<&[f32]>,    // length: n_per_row (per-column importance, per llama.cpp's convention)
    ) -> Result<Vec<u8>, QuantizeError>;
}
```

One impl per `GgmlType` whose disk format we emit. Lives at `src/quantize/ggml_quants/<type>.rs`. Each impl is a port of the corresponding `ggml_quants.c` function (`quantize_row_q4_K`, `quantize_row_q5_K`, …). Signature mirrors `ggml_quantize_chunk`'s row-major contract; behavior is byte-identical to llama.cpp's reference.

**Hard-error contract:** when convert encounters a tensor whose target `GgmlType` (per the active `QuantPolicy`) has no `Quantizer` impl, the pipeline returns a typed `QuantizeError::NoQuantizerForType { ggml_type: GgmlType }` — no silent fallback, no F16 escape. This implements the no-fallback rule at the trait-dispatch layer.

### LlamaFtype mapping (Decision §2 concrete)

Rust enum mirrors llama.cpp's `enum llama_ftype` (`/opt/llama.cpp/include/llama.h`) at the literal numeric values for byte-level header compatibility:

```rust
#[repr(u32)]
pub enum LlamaFtype {
    AllF32         =  0,
    MostlyF16      =  1,
    MostlyQ4_0     =  2,
    MostlyQ4_1     =  3,
    MostlyQ8_0     =  7,
    MostlyQ5_0     =  8,
    MostlyQ5_1     =  9,
    MostlyQ2_K     = 10,
    MostlyQ3_K_S   = 11,
    MostlyQ3_K_M   = 12,
    MostlyQ3_K_L   = 13,
    MostlyQ4_K_S   = 14,
    MostlyQ4_K_M   = 15,
    MostlyQ5_K_S   = 16,
    MostlyQ5_K_M   = 17,
    MostlyQ6_K     = 18,
    MostlyIQ4_NL   = 25,
    BF16           = 32,
    // Holes (4, 5, 6, 19-24, 26-31) are llama.cpp values out of v1 scope (TQ1_0/TQ2_0/IQ2_*/IQ3_*/IQ1_*).
    // Add only when the matching Quantizer impl ships.
}
```

v1 supported set: `AllF32 / MostlyF16 / BF16 / MostlyQ4_0 / MostlyQ4_1 / MostlyQ5_0 / MostlyQ5_1 / MostlyQ4_K_S / MostlyQ4_K_M / MostlyQ5_K_S / MostlyQ5_K_M / MostlyQ6_K / MostlyQ8_0 / MostlyIQ4_NL`.

### TensorRef (passed to QuantPolicy::target_for)

```rust
pub struct TensorRef<'a> {
    pub name: &'a str,            // canonical GGUF tensor name (e.g., "blk.0.attn_q.weight")
    pub shape: &'a [usize],       // dims, row-major
    pub source_dtype: SourceDtype, // F32 | F16 | BF16 (from safetensors header)
    pub arch: ArchName,            // Gemma4 | Qwen35Moe | MiniMaxM27 | Llama3 | Bert | NomicBert | Qwen3VlText | Gemma4Mmproj
    pub layer_index: Option<usize>, // None for global tensors (token_embd / output); Some(i) for per-block
}
```

`ArchName` is closed enum — adding a new arch is an explicit code change, NOT silent runtime detection.

### Vision / audio tensor patterns (canonical source)

The vision-tensor F16-emit gate is `crate::quantize::vision::is_vision_tensor_pattern(name)` and its sibling `is_audio_tensor_pattern(name)`. Together they decide whether a tensor is "modality-side" (Pa policy bypassed, F16 emitted directly) or "language-side" (policy decides).

`is_vision_tensor_pattern` returns `true` iff the name contains any of:
`model.visual.` | `vision_tower.` | `vision_model.` | `vit.` | (prefix) `visual.` | `.visual.`

`is_audio_tensor_pattern` (NEW per P-1 audit finding E) returns `true` iff the name contains any of:
`audio_tower.` | `audio_model.` | `whisper.`

These are the **only** places modality-pattern membership is decided. The convert dispatcher checks `is_vision_tensor_pattern(name) || is_audio_tensor_pattern(name)` BEFORE calling `QuantPolicy::target_for`. The current `layer_mix.rs::is_vision_tensor_pattern` (at `src/quantize/layer_mix.rs:366`, HEAD `85bee70e`) ports verbatim to `src/quantize/vision.rs`; the audio sibling is new code. The three inline duplicate vision checks at `backends/gguf.rs:322-333, 721-724, 905-909` (per P-1 audit) are deleted.

### QuantPolicy trait (Decision §3 concrete)

```rust
pub trait QuantPolicy {
    fn target_for(&self, tensor: &TensorRef) -> Result<GgmlType, QuantizeError>;
    // Optional: imatrix requirement check before quantize starts
    fn requires_imatrix(&self) -> bool;
}
```

`StandardPolicy { ftype: LlamaFtype }` is a port of `llama_tensor_get_type_impl`. `ApexPolicy { tier: ApexTier }` is the mudler port. The `target_for` return type makes the no-fallback rule type-system-checked: a policy CANNOT silently emit F16 — it must either succeed with a non-F16 type or return `Err`. Vision-tensor F16 is handled outside the policy at the dispatcher layer (the dispatcher checks vision-pattern membership before calling the policy at all).

## Plan

Phases run sequentially. Every phase has a binary acceptance gate; later phases do not start until the prior phase's gate passes.

### P-1 — Audit & classify + vendor external pins

**Why:** ADR-014's predecessor's delete-list was incomplete. Five files in `src/quantize/` (`mod.rs`, `k_quant.rs`, `k_quant_codec.rs`, `q_legacy.rs`, `layer_mix.rs`) totalling 16,867 LOC have no stated fate. Several contain kernels that may be the P0 port target; others are dead policy code that should be deleted. Separately: all of ADR-033's byte-cmp gates depend on external pins (llama.cpp @ `c779f6198`, mudler @ `63c5048b`); these need to be vendored locally so verification is reproducible offline (network restrictions or upstream changes can't invalidate the gate).

**What:**
- Function-by-function classification of every fn in those five files plus the existing dispatcher arms in `main.rs` and the GGUF writer slice at `backends/gguf.rs:282–1259`. Three buckets: `KEEP` (utility we still need), `MODIFY` (kernels P0 ports in place), `DELETE` (superseded policy / dead code).
- **Vendor mudler/apex-quant @ `63c5048b7dc9ff230f2397d7bc445ca28894b769` to `vendor/apex-quant/` (git submodule or `git archive` snapshot; either works; submodule preferred for auditability).** Generate `data/apex-references/manifest.json` (the fingerprint → config-file map per Decision §9) via a one-shot vendor script that walks `vendor/apex-quant/configs/*.txt` and computes fingerprints from each config's matching upstream HF model. Vendor script committed at `scripts/vendor_apex.sh`; rerun is a deliberate ADR-amendment event, not a CI step.
- **Confirm llama.cpp @ `c779f6198` is the operator's local `/opt/llama.cpp` HEAD AND remains so for the duration of the project.** If the operator's llama.cpp moves during execution, ADR-033 gates re-anchor to the new SHA (and the ADR documents the move with rationale). P-1 records the current SHA in `data/llama_cpp_pin.txt` for later re-verification.

**Acceptance criteria:**
- Markdown table inline in ADR-033 §"Audit results" (this section, populated after P-1 runs), one row per fn: `file.rs::fn_name | LOC | KEEP/MODIFY/DELETE | rationale`.
- Zero unclassified fns in those files at P-1 exit.
- The delete-list LOC total in §7 sums correctly (no hand-wave numbers).
- `vendor/apex-quant/` exists with the pinned SHA; `git -C vendor/apex-quant rev-parse HEAD` returns `63c5048b7dc9ff230f2397d7bc445ca28894b769`.
- `data/apex-references/manifest.json` exists with at least 5 entries (one each for gemma4-26B-A4B, qwen35moe-3.6, MiniMax-M2.7, plus 2 more from the mudler configs/ that match known model fingerprints).
- `data/llama_cpp_pin.txt` records the current local llama.cpp SHA.

**Deliverable:** updated §"Audit results" section in this ADR; vendored mudler ref; manifest; llama.cpp SHA record. No production code touched.

### P0 — Pure-Rust ggml-quants port + per-arch safetensors→F32 mapping (convert-side)

**Why:** Today's hf2q-side kernels are scattered across `k_quant.rs` (5541 LOC) and `static_quant.rs` (468 LOC); the per-arch safetensors→F32 mapping for inference lives in `src/inference/models/<arch>/` but ONLY covers `{bert, gemma4, nomic_bert, qwen35, qwen3vl_text}` (verified at HEAD `ebecc21c` via `ls src/inference/models/`). The convert path additionally needs Llama-3-8B (dense decoder test fixture) and MiniMax-M2.7 (3rd MoE for APEX validation), neither of which exists in inference. **Per operator decision 2026-05-18: P0 adds tensor-mapping for these arches in `src/convert/arch/` (a NEW dir; convert-side mapping is independent of inference-side forward-pass code) — inference support for these arches is deferred to a separate effort.**

**What:**
- Port `ggml-quants.c` quantize-side functions one per file under `src/quantize/ggml_quants/`. **v1 set (11 files)**: `{q2_k.rs, q3_k.rs, q4_0.rs, q4_1.rs, q5_0.rs, q5_1.rs, q4_k.rs, q5_k.rs, q6_k.rs, q8_0.rs, iq4_nl.rs}`. Maps 1:1 with `quantize_row_q2_K` / `quantize_row_q3_K` / `quantize_row_q4_0` / ... in `/opt/llama.cpp/ggml/src/ggml-quants.c` at the pinned SHA. (Q2_K + Q3_K added per P-1 audit finding A: their dequant is externally referenced by `src/quality/mod.rs:612` and `src/backends/gguf.rs` size estimator at L1275 / L1458 / L2207 / L2566 / L2819 / L3085; dropping them would break those call sites.) Pre-existing logic in `k_quant.rs` is either ported into these files (per P-1 classification) or deleted.
- **Imatrix-aware variants are NEW code for the 6 legacy types.** Per P-1 audit finding F, `src/quantize/q_legacy.rs` has zero `*_impl` (imatrix-aware) variants today. llama.cpp's `quantize_row_q4_0_impl` (ggml-quants.c:2008) accepts a `quant_weights` arg and dispatches on null. The new `Quantizer::quantize(src, n_per_row, imatrix: Option<&[f32]>)` requires imatrix-aware code for every legacy type. For `{q4_0, q4_1, q5_0, q5_1, q8_0, iq4_nl}` P0 ships BOTH the no-imatrix path (port of `quantize_row_<T>_ref`) AND the imatrix path (port of `quantize_row_<T>_impl`). The K-family files already had `_imatrix` variants in `k_quant.rs`, so this asymmetry only affects the 6 legacy files.
- Per-arch convert-side mapping at `src/convert/arch/<arch>.rs` for the full convert matrix: `{gemma4, qwen35moe, qwen3vl, gemma4_mmproj, bert, nomic_bert, llama3, minimax_m2}`. Each is a port of the corresponding `/opt/llama.cpp/conversion/*.py` module, restricted to tensor-name + shape mapping (no inference logic). For arches already in `src/inference/models/`, the convert-side mapper REUSES the inference-side tensor-name conventions; for new arches (`llama3`, `minimax_m2`), it's the only mapping that exists.
- A single `ArchName::detect(config_json: &Value) -> Result<ArchName>` reads `config.json::model_type` and `config.json::architectures` to dispatch. Failure is typed: `ConvertError::UnsupportedArch { detected: String, supported: Vec<&str> }`.
- **FP8 source-dtype support (NEW v1 scope per Decision §10):** add `src/convert/source_dtype/fp8.rs` implementing the block-wise `float8_e4m3fn` → F32 dequantize. Reads `quantization_config.weight_block_size` from `config.json` (typically `[128, 128]`); per-tensor, reads block scales stored alongside the FP8 payload (per HF convention: `<tensor>.weight` (FP8) + `<tensor>.weight_scale_inv` (F32 block scales)). Modules listed in `quantization_config.modules_to_not_convert` are read as F32 / BF16 directly. Required by MiniMax-M2.7 which ships in FP8.

**Acceptance criteria:**
- For every `GgmlType` in v1 scope (the 11-file list): a unit test takes a fixed-seed F32 input vector + fixed `n_per_row` (256 for K-quants, 32 for legacy) and produces output bytes that `cmp` byte-equal against the same input fed to llama.cpp's reference at the pinned SHA. Reference outputs generated once via a small C harness wrapping `ggml_quantize_chunk` and checked into `tests/fixtures/ggml_quants/<type>_<n>.bin`.
- **The C harness used to generate reference fixtures is built `aarch64-apple-darwin` (NEON enabled) on macOS Apple Silicon** (per P-1 audit finding I; `k_quant.rs` L9-18 module doc flags a NEON-vs-scalar argument-order divergence in `make_qkx2_quants`). The same harness rebuilt `x86_64-pc-linux-gnu` (no NEON) on x86 Linux must produce byte-identical fixtures; if it doesn't, hf2q ports are matched against the NEON variant explicitly and the divergence is documented in `tests/fixtures/ggml_quants/README.md`.
- For every arch in the convert matrix: a fixture test takes a real safetensors directory (or a tiny synthetic one for arches whose real model is multi-GB), runs hf2q's convert through to F32-tensor-emission only (no quantize), and `cmp`s the resulting F32 byte stream against what `convert_hf_to_gguf.py <hf-dir>` would emit at the pinned llama.cpp SHA. Fixtures stored at `tests/fixtures/convert_arch/<arch>.f32.bin`.
- Memory bound: peak RSS during convert of a 26B-param model (e.g., gemma4-26B-A4B) is bounded by `4 × largest_single_tensor_F32_size + 512 MiB` (tensor-by-tensor streaming with the source reader mmapping shards instead of loading them into the heap). Tightened from the original `2 × model_safetensors_size + 512 MiB` envelope 2026-05-18 after the real-model OOM finding — see §Open Issues / Real-Model Findings. Validated by `tests/convert_v2_integration.rs::convert_v2_streaming_rss_under_bound_2026_05_18` which runs convert under `/usr/bin/time` and asserts the OS-reported peak RSS stays under the bound.

### P1 — Quantizer trait + StandardPolicy

**Why:** Wire the kernels into a policy-driven pipeline that takes safetensors and produces a quantized GGUF.

**What:** Implement `Quantizer` trait per Decision §2; implement `StandardPolicy` per Decision §3 (byte-for-byte port of `llama_tensor_get_type_impl` at `/opt/llama.cpp/src/llama-quant.cpp:411-657` at the pinned SHA, with the `tensor_type_fallback` first-downshift behavior at `:362-408` and the second-misalignment hard-error per the no-fallback rule). Wire to a single `hf2q convert --quant <standard-type>` CLI path. Streaming property preserved (per P2's writer + ADR-014's no-disk-intermediate invariant).

**Acceptance criteria:** for every convert-matrix fixture and every StandardPolicy quant in `{q4_0, q4_k_s, q4_k_m, q5_k_s, q5_k_m, q6_k, q8_0, iq4_nl}`:
- `hf2q convert <hf-dir> --quant <type> -o hf2q.gguf` byte-equals `(convert_hf_to_gguf.py <hf-dir> --outtype f32 - | llama-quantize - <type> llama.gguf)` output, where `convert_hf_to_gguf.py` and `llama-quantize` come from llama.cpp @ `c779f6198`.
- `cmp hf2q.gguf llama.gguf` exits 0.
- If a tensor's policy resolves to a `GgmlType` that's not in v1's `Quantizer` impl set (the holes in `LlamaFtype`), convert returns `QuantizeError::NoQuantizerForType` — not a panic, not a silent F16 demotion.

### P2 — Seek-back incremental writer

**Why:** The two-pass writer at `backends/gguf.rs:282–1259` is the iter-99 / Bug-B-sequel bug-class home. A seek-back single-pass writer is structurally simpler and eliminates an entire class of "header / payload offset mismatch" bugs.

**What:** New writer in `src/backends/gguf/writer.rs`. Reserves a header region, streams tensor payloads to disk via the `Quantizer` trait, seeks back to fill the header. Single-file output only. Old two-pass writers deleted in P6. GGUF version 3 (matches `const GGUF_VERSION: u32 = 3` at HEAD).

- **`backends/gguf.rs:282-1259` contains TWO complete two-pass writers**, not one (per P-1 audit finding B): `Backend::write` (L282-738) for text GGUF and `write_mmproj_gguf` (L887-1189) for mmproj GGUF. Both must be replaced together; deleting only one leaves the bug-class half-alive in the other. The new writer is parametric on (text | mmproj) via the metadata builder, not two separate writers.
- **No zero-pad write site exists in the new writer** (per P-1 audit finding C). The four sites at `backends/gguf.rs:639-641, 659-661, 677-679, 1132-1134` (`if current_pos < target_pos { write zeros }`) are the literal iter-99 bug-class targets; the seek-back design has no pass-1 prediction and therefore no need to pad-correct.
- **F16 demotion logic is moved into `QuantPolicy::target_for`** as typed errors, not buried in the writer (per P-1 audit finding D). The two inline F16 fallback sites at `backends/gguf.rs:496-502` (K-quant row-misalignment → F16) and `:511-521` (block-32 misalignment → F16) are deleted; the policy's `target_for` returns `Err` instead, and the dispatcher routes vision-tensor F16 via `is_vision_tensor_pattern` / `is_audio_tensor_pattern` before calling the policy.
- **Three inline vision-pattern checks** at `backends/gguf.rs:322-333, 721-724, 905-909` (per P-1 audit finding E) consolidate to `is_vision_tensor_pattern` + the new sibling `is_audio_tensor_pattern` (the text-filter at L322-333 covers `audio_tower` substrings the mmproj-filter doesn't).

**Acceptance criteria:**
- All P1 byte-cmp gates pass under the new writer.
- A streaming-property test runs `hf2q convert <hf-dir> --quant q5_k_m -o out.gguf` while monitoring open file descriptors via `lsof` (or platform equivalent). The only output-side file descriptor that is open at any point during convert is `out.gguf` itself; no intermediate `.f16.gguf` or `.tmp.gguf` ever appears in the process's fd table or in the working directory. Test asserts `find . -name '*.gguf' -newer <start_marker>` returns only `out.gguf` after convert exits.
- **No zero-pad write site** in the new writer: `grep -nE 'write.*(zero|null|0u8\\s*;)' src/backends/gguf/writer.rs` returns no matches (no `write_zeros`-shaped call, no `vec![0u8; n]` write, no `seek_write_pad`). This is structurally enforced — the seek-back design has no pass-1 prediction.
- **No inline F16 demotion** in the new writer: `grep -nE 'F16|MOSTLY_F16|fallback' src/backends/gguf/writer.rs` returns no matches outside the dispatcher's explicit vision/audio path. Demotion to F16 lives in `QuantPolicy::target_for` as a typed `Err` and in the upstream dispatcher's vision/audio gate, never in the writer.
- Memory bound carries from P0 (`2 × model_safetensors_size + 512 MiB` peak RSS).

### P3 — Collapse `TensorQuantInfo` to `QuantizedTensor`

**Why:** Seven-field IR with simultaneous `method`/`bits`/`preserved`/`scales`/`biases`/`ggml_type` representations is the substrate for "field A says one thing, field B says another, code paths disagree" bugs.

**What:** Replace `TensorQuantInfo` with `QuantizedTensor { ggml_type, data }` (Decision §1). Walk every read site (estimated ~40 call sites per grep at HEAD; P-1 produces the exact list) and update.

**Acceptance criteria:**
- `grep -rn 'TensorQuantInfo' src/` returns zero hits.
- All P1 + P2 gates still pass.

### Pa — Mudler tier rules + MoE classifier

**Why:** `ApexPolicy::target_for` needs to know (for a given tier + tensor) what `GgmlType` to emit. Mudler encodes these rules in `generate_config.sh` (per-tier `--tensor-type-file` content) and tensor-name regex.

**What:**
- Clone `mudler/apex-quant` @ `63c5048b7dc9ff230f2397d7bc445ca28894b769` into a vendored read-only ref at `vendor/apex-quant/`. Document the SHA in `src/quantize/apex/rules.rs` as a top-of-file comment AND in `data/apex-references/MUDLER_SHA.txt`.
- **Port mudler's algorithmic per-tier rule tables** (`scripts/generate_config.sh`) to Rust constants in `src/quantize/apex/rules.rs`. v1 ships 7 named tiers + custom (per the Decision §6 CLI surface). For each tier, the rule table is the {EDGE_EXP, NEAR_EXP, MID_EXP, EDGE_SHARED, MID_SHARED, EDGE_ATTN, MID_ATTN} 7-tuple from `generate_config.sh` (verified at the pinned SHA), plus the layer-region boundaries (EDGE = L0..4 + (L_LAST-4)..L_LAST; NEAR = L5..9 + (L_LAST-9)..(L_LAST-5); MID = L10..(L_LAST-10), where L_LAST = NUM_LAYERS - 1). Layer count auto-detected from source `config.json::num_hidden_layers` (no `NUM_LAYERS` env var; the env-var override was mudler's CLI surface, not a model property).
- **Vendor mudler's per-model config files** to `data/apex-references/<original-name>.txt` (verbatim from `vendor/apex-quant/configs/<original-name>.txt`). v1 vendors at minimum: `gemma4_26b_*`, `qwen35_fernflower_*`, `carnice_qwen36_mtp_*`, `minimax_m27_*` (the configs that match v1 fixture models). Plus `data/apex-references/manifest.json` mapping (`(model_type, num_layers, hidden_size, num_experts, num_attention_heads, num_key_value_heads, intermediate_size, moe_intermediate_size)` fingerprint) → `<original-name>.txt`. Manifest regenerated at vendor-time by `scripts/vendor_apex_configs.sh` (computes fingerprints by reading each config's matching source model's config.json from HF).
- **MoE-arch detection** for `ApexPolicy::target_for`: via the source `config.json::model_type` field (not GGUF metadata — we're upstream of writing the GGUF). Per-arch tensor-name classifier files at `src/quantize/apex/<arch>.rs` (qwen35moe, gemma4_moe, minimax_m2) port mudler's tensor-name conventions for {routed expert / shared expert / attention / output / token_embd} classification.
- **`ApexPolicy::target_for` resolution order** (per Decisions §3 + §9):
  1. Vision-pattern check (handled upstream at the dispatcher; not by ApexPolicy).
  2. Fingerprint match against `data/apex-references/manifest.json`. If matched, look up `<tensor_name>` in the per-model config file → return that `GgmlType` (or `Err` if tensor not in the config).
  3. Otherwise: algorithmic `generate_config.sh`-equivalent — classify tensor by role + layer region, look up the {role × region} entry in the tier's rule table.
  4. If no rule matches (unexpected tensor name on a known arch): typed error.

**Acceptance criteria:**
- For each supported MoE arch (`qwen35moe`, `gemma4`, `minimax_m2`) and each algorithmic tier (`quality`, `i-quality`, `balanced`, `i-balanced`, `compact`, `i-compact`, `mini`): hf2q's `target_for` output (rendered as a `<tensor>=<quant_type>` line list, sorted by tensor name) matches `vendor/apex-quant/scripts/generate_config.sh --profile <tier> --layers <N>` output line-for-line for N = {40 (gemma4 default), 62 (MiniMax-M2.7)}. Validated by a fixture test that runs both and `diff`s the output.
- For each fingerprint-matched per-model config in the v1 vendor set: hf2q's `target_for` output `cmp 0` equals the literal vendored config file content.
- Unsupported arches return `ApexError::UnsupportedArch { arch: String, supported: &'static [&'static str] }`.
- `apex-custom` without `--tensor-type-file` returns `ApexError::CustomRequiresTensorTypeFile`.
- Pa exit gate: `cargo test -p hf2q --test apex_rules` is green.

### P4a — ApexPolicy non-I tiers ship

**Why:** First end-to-end APEX capability. Reproduces operator's `gemma4-ara-2pass-APEX-Q5_K_M.gguf` class of artifact (no imatrix).

**What:** Wire `--quant apex-mini / apex-compact / apex-balanced / apex-quality` through `ApexPolicy` + Pa's rules + P3's IR + P2's writer.

**Acceptance criteria (development-time gate; retires after stabilization):**
- For each test-matrix MoE fixture and each non-I tier: `hf2q convert <hf-dir> --quant apex-<tier> -o hf2q.gguf` byte-equals output from `mudler/apex-quant` running locally on the same `<hf-dir>` and tier.
- The gate is run by the developer / CI during the porting effort. Once stable across all matrix fixtures, the gate retires (we don't keep installing mudler in CI forever) — confidence is then maintained by the per-arch fixture tests in P0/P1 + structural mudler-rule tests in Pa.

### Pi — Imatrix subsystem

**Why:** I-tier APEX (I-Compact / I-Balanced / I-Quality) requires per-row activation-importance data. llama-imatrix's `.imatrix.gguf` format is the de facto reference; we need a hf2q-side generator that produces equivalent output.

**Reference format** (from `/opt/llama.cpp/tools/imatrix/imatrix.cpp` @ pinned SHA): the `.imatrix.gguf` carries:
- KV header: `general.type = "imatrix"` (string), `imatrix.datasets` (array of strings — calibration corpora names), `imatrix.chunk_count` (u32), `imatrix.chunk_size` (u32 = `n_ctx / n_parallel`)
- Per-source-tensor: a GGUF tensor named after the source weight (e.g., `blk.0.attn_q.weight`) whose payload encodes `Stats { values: Vec<f32>, counts: Vec<i64> }` — `values.len() == n_mat × row_size`; `counts.len() == n_mat`. n_mat is the number of "matrices" the source tensor was viewed as during inference (1 for most weights; expert_count for `*_exps.weight`).

**What:**
- Add `src/quantize/imatrix/` module: corpus loader (reads `data/calibration/<name>.txt`), forward-pass driver (reuses hf2q's existing decoder forward-pass on top of mlx-native), per-row importance accumulator (sum of squared activations per row, divided by counts at finalize), llama-imatrix-format writer producing exact-schema `.imatrix.gguf`.
- Bundle `data/calibration/calibration_datav3.txt` (cdv3, default; bartowski's canonical 750 KB corpus) and `data/calibration/mudler_v1.txt` (mudler-style; sampled from openassistant + the-stack-smol + math-instruct + ToolBench, version-pinned in `data/calibration/mudler_v1.README.md` with sampling seed + per-source token counts).
- Imatrix is computed in-memory during convert (no `.imatrix.gguf` written to disk) when `--imatrix-corpus <name>` is used; written to disk only as a side-effect when the user runs `hf2q convert --imatrix-out <path>` (separate flag, optional).
- Pi only runs against arches with hf2q inference support (`{bert, gemma4, nomic_bert, qwen35, qwen3vl_text}`). For convert-only arches (llama3, minimax_m27) added in P0, I-tier APEX is OUT of v1 scope; convert with `--quant apex-i-*` against those arches returns `ApexError::ImatrixRequiresInference { arch, supported_for_imatrix: &[ArchName] }`.

**Acceptance gate (with verify spike):** byte-cmp against llama-imatrix.
- **Verify spike (run before Pi proper begins, ~half-day):** feed identical corpus + model to llama-imatrix on CPU and on Metal (`-ngl 999`). `cmp` the two outputs. If bytes match: byte-cmp is achievable, gate stands. If bytes diverge: the strict gate is unsatisfiable due to FP accumulation order divergence between Metal kernels and CPU reference; ADR amends to numeric-cmp (per-entry within 1e-6 relative tolerance) before Pi continues.
- **CI gate (fast):** tiny synthetic corpus (~1k tokens of wikitext-2 valid); imatrix run completes in ~1 min; hf2q output `cmp`-equals llama-imatrix reference. Catches accumulation-order regressions every commit.
- **Pre-release gate (slow):** full cdv3 corpus; production-realistic; manual or weekly trigger. Catches divergence that surfaces only at scale.

### P4b — ApexPolicy I-tier variants

**Why:** Ship I-Compact / I-Balanced / I-Quality. Reproduces operator's `qwen3.6/APEX-Q5_K_M.gguf` class of artifact (imatrix-derived).

**What:** Wire `--quant apex-i-compact / apex-i-balanced / apex-i-quality` through `ApexPolicy` + Pa's rules + Pi's imatrix + P3's IR + P2's writer.

**Acceptance criteria:** byte-cmp against `mudler/apex-quant` running locally (development-time gate; same retirement story as P4a).

### P6 — Delete superseded code

**Why:** The new policy + writer + IR shipped in P1–P4b makes the old subsystems redundant.

**What:** Per P-1's audit, delete:
- The 5 superseded quantizer impls: `quantize/k_quant_codec_quantizer.rs`, `quantize/variant_quantizer.rs`, `quantize/dwq_k_quantizer.rs`, `quantize/mixed.rs`, `quantize/static_quant.rs` (3,428 LOC; see `docs/adr-033-audit/delete-listed.md`).
- `src/calibrate/dwq.rs` — hf2q's homebrew DWQ (operator: "current DWQ is fake DWQ; real DWQ = future Apple MLX `dwq.py` port; reserve `--quant dwq` for that ADR").
- `src/calibrate/apex.rs` — superseded by `ApexPolicy` + Pi imatrix subsystem (per P-1 audit finding H; not in original delete-list, surfaced during external-caller analysis).
- `src/quantize/k_quant_codec.rs` — pure dispatch shim (1,452 LOC, no kernels; per audit).
- `src/quantize/mod.rs` — orchestration only (~6,432 LOC delete; trait scaffolding reshaped in-place; per audit).
- The k_quant.rs test-mod (2,474 LOC) and the q_legacy.rs test-mod (896 LOC); kernel code in those files MOVES to the new `src/quantize/ggml_quants/<type>.rs` (per P0 ports).
- `backends/gguf.rs:282–1259` two-pass-writer slice AND its mmproj-writer sibling (`write_mmproj_gguf`, L887-1189; per P-1 audit finding B — two writers, not one).
- The three `backends/gguf.rs` branches at L1334, L2075, L4835 that switch on `METHOD_K_QUANT_CODEC_DIRECT` (per `docs/adr-033-audit/delete-listed.md` note 1; outside the writer slice but tendrils of the same delete chain).
- Per [[feedback-no-backwards-compat-2026-05-18]]: NO migration shims, NO env-var deprecation aliases, NO `cli::QuantMethod` legacy-name aliases. The 3 retired env vars (`HF2Q_STREAMING_PHASE3`, `HF2Q_STREAMING_PHASE3_MUT`, `HF2Q_USE_LEGACY_DWQ_Q4_0`) are deleted from `parse_env`; callers compile-fail and get fixed at the same commit.

The full per-file disposition lives in `docs/adr-033-audit/{synthesis.md, quantize-mod.md, k-quant.md, k-quant-codec.md, q-legacy.md, layer-mix.md, gguf-writer.md, main-dispatch.md, delete-listed.md}`.

`cargo build --release && cargo test --release` between each delete commit. **Pre-condition: the test suite is green at start-of-P6.** If it's not (today, this is unverified), P-1 includes a "green the suite first" sub-step.

**Acceptance criteria:**
- All earlier P-gates still pass after deletion.
- LOC delta in §7 sums correctly post-deletion.

### P7 — Public-release readiness

**Why:** ADR-033 was motivated by "we keep going down rabbit holes because everything we test is radioactive dogshit." P7 declares the rabbit-hole era over.

**What:** End-to-end smoke matrix (all matrix fixtures × all `<name>` quant types where `<name>` is in scope) runs green. README + `hf2q convert --help` document the supported set. Error messages for the deliberate non-goals (TQ1_0, split-file, raw `apex`, `dwq`) are typed and informative.

**Acceptance criteria (measurable):**
- `hf2q convert --help` enumerates every supported `--quant <name>` value with a one-line description; covers both StandardPolicy and ApexPolicy variants; lists reserved/out-of-scope names with their typed error.
- README has a "Quick start" section that's been executed end-to-end by someone other than the implementer; they produce a working GGUF for at least one of {gemma4, qwen35moe, bert} from a HuggingFace `<hf-dir>` using only the README's commands (no source-code reading). Verified by either (a) the implementer asking a teammate to run the README cold and report whether it worked, or (b) a fresh-checkout CI job that runs the exact README commands.
- Every typed-error code listed in Decision §6 (TQ1_0 out-of-scope, split-file out-of-scope, `apex` unqualified, `dwq` reserved, ApexUnsupportedArch, NoQuantizerForType, ImatrixRequiresInference) has a unit test asserting the error message contains an actionable hint (the supported alternative, the tracking issue, or the future ADR reference).

## Audit results

Populated 2026-05-18 by 7 parallel P-1 audit agents. Per-file detail at `docs/adr-033-audit/<name>.md`; full synthesis at `docs/adr-033-audit/synthesis.md`. Findings A–M surfaced during audit are folded into the relevant §Plan / §Decision sections above (e.g., P0's 11-file set, P2's two-writer replacement, vision/audio gate).

### Disposition totals

| File / scope | LOC | DELETE LOC | MODIFY LOC | KEEP LOC | Audit file |
|---|---|---|---|---|---|
| `src/quantize/mod.rs` | 6,440 | ~6,432 | 8 (trait reshape) | 0 | `docs/adr-033-audit/quantize-mod.md` |
| `src/quantize/k_quant.rs` | 5,541 | 2,474 (test mod) | 3,067 (5 K-quant files + common helpers) | 0 | `docs/adr-033-audit/k-quant.md` |
| `src/quantize/k_quant_codec.rs` | 1,452 | 1,452 | 0 | 0 | `docs/adr-033-audit/k-quant-codec.md` |
| `src/quantize/q_legacy.rs` | 2,130 | 0 (file is mv+split+cfg-rehome) | ~1,801 (6 legacy-quant files) | ~157 (dequant utils + QLegacyError) | `docs/adr-033-audit/q-legacy.md` |
| `src/quantize/layer_mix.rs` | 1,304 | ~1,107 | ~190 (standard_policy.rs) | 8 (vision.rs) | `docs/adr-033-audit/layer-mix.md` |
| **5-file subtotal** | **16,867** | **~11,465** | **~5,066** | **~165** | — |
| `src/backends/gguf.rs:282-1259` writer slice (text + mmproj writers) | ~977 | ~480 (9 regions; 4 zero-pad sites; size predictor; inline F16) | ~295 (8 regions; seek-back writer) | ~286 (KV-pair enc, tensor-name canon) | `docs/adr-033-audit/gguf-writer.md` |
| `src/main.rs` dispatch arms (L1043-3453) | ~3,445 | ~1,473 (17 regions; 5 dispatch arms + 3 DWQ subcmds + 11 stale CLI variants) | ~395 (6 regions; cli::QuantMethod rewrite, cmd_convert single-arm collapse) | ~1,577 (CLI bootstrap, serve, unrelated subcmds) | `docs/adr-033-audit/main-dispatch.md` |
| 5 ADR delete-listed files (`dwq_k_quantizer`, `k_quant_codec_quantizer`, `mixed`, `static_quant`, `variant_quantizer`) | 3,428 | 3,428 | 0 | 0 | `docs/adr-033-audit/delete-listed.md` |

**Grand totals: DELETE ~16,846 LOC | MODIFY ~5,756 LOC (kernel ports + policy port + writer rewrite + CLI reshape) | KEEP ~2,028 LOC (CLI bootstrap, quality-test utils, dequant round-trip helpers, KV-pair encoding).**

### Audit-driven amendments folded into the Plan

| # | Finding | ADR section amended |
|---|---|---|
| A | P0 v1 set is 11 files (added Q2_K + Q3_K) | §P0 "What" |
| B | gguf.rs has TWO two-pass writers (text + mmproj) | §P2 "What" |
| C | 4 zero-pad fallback sites at gguf.rs:639/659/677/1132 deleted under seek-back | §P2 "What" / "Acceptance criteria" |
| D | 2 inline F16 fallback sites (gguf.rs:496-502, :511-521) → typed errors in policy | §P2 "What" / "Acceptance criteria" |
| E | New `is_audio_tensor_pattern` sibling to vision gate; consolidate 3 inline gguf.rs duplicates | §"Vision / audio tensor patterns" |
| F | q_legacy gets imatrix-aware variants ADDED in P0 (none exist today) | §P0 "What" |
| G | StandardPolicy::target_for is COMPLETE port of llama_tensor_get_type_impl (no deferred branches) | §P1 "What" |
| H | `src/calibrate/apex.rs` added to P6 delete list (ADR orphan) | §P6 "What" |
| I | NEON-order caveat for C harness fixture generation | §P0 "Acceptance criteria" |
| K | `cli::QuantMethod` rewritten to Decision §6's surface (17 variants → ~20 new) | §P1 / §P6 "What" |
| M | 3 retired env vars deleted (no migration code) | §P6 "What" (and [[feedback-no-backwards-compat-2026-05-18]]) |

J (the `#[from]` edge on QLegacyError) and L (vision-gate move) are implementation details captured in `docs/adr-033-audit/synthesis.md` without separate ADR amendments.

## Acceptance criteria (overall)

The whole ADR ships when:

1. Every per-phase gate above passes.
2. **Convert matrix × StandardPolicy:** for each of `{gemma4-26B-A4B, qwen35moe-3.6-35B-A3B, qwen3vl_text, gemma4-mmproj, bert/bge-large-en, nomic_bert, llama3-8B, minimax-m27}` × each of `{q4_0, q4_k_s, q4_k_m, q5_k_s, q5_k_m, q6_k, q8_0, iq4_nl}` — `hf2q convert` output byte-cmps `cmp 0` against `(convert_hf_to_gguf.py | llama-quantize)` output at the pinned llama.cpp SHA.
3. **MoE matrix × ApexPolicy non-imatrix tiers:** for each of `{gemma4-26B-A4B, qwen35moe-3.6-35B-A3B, MiniMax-M2.7}` × each of `{apex-quality, apex-balanced, apex-compact, apex-mini}` — `hf2q convert` output byte-cmps `cmp 0` against `mudler/apex-quant --profile <tier>` @ pinned SHA output (development-time gate; retires after stabilization per P4a).
4. **Inference-supported MoE × ApexPolicy imatrix tiers:** for each of `{gemma4-26B-A4B, qwen35moe-3.6-35B-A3B}` (the subset with inference support; MiniMax-M2.7 is convert-only in v1) × each of `{apex-i-quality, apex-i-balanced, apex-i-compact}` — same byte-cmp gate. MiniMax-M2.7's I-tier variants in v1 return `ApexError::ImatrixRequiresInference { arch: minimax_m2, supported_for_imatrix: &[...] }`.
4a. **Per-model override matrix:** for each fingerprint-matched per-model config (at minimum `carnice_qwen36_mtp_quality.txt`, `gemma4_26b_quality.txt`, `minimax_m27_quality.txt`): `hf2q convert <matching-model> --quant apex-quality` byte-cmps `cmp 0` against the vendored config's literal rules. (Verifies the fingerprint-match dispatcher works end-to-end.)
5. **Streaming property:** no intermediate F16 GGUF on disk during convert (verified by P2's fd-monitoring test).
6. **No silent F16 fallbacks:** every F16-emitting code path is either the vision-pattern path or the explicit `--quant f16` path; `shape_fallback` returns `Err` on second-misalignment.
7. **Production APEX files are NOT a gate.** The operator's existing `gemma4-ara-2pass-APEX-Q5_K_M.gguf` and `qwen3.6/APEX-Q5_K_M.gguf` were produced externally with possibly-non-canonical recipes; we don't try to byte-reproduce them. The gate is "we byte-reproduce mudler/apex-quant @ pinned SHA's canonical recipe."

## Risks

### Risk 1 — Per-arch safetensors→F32 mapping divergence (P0 / P1)

**What:** llama.cpp ships 79 per-arch tensor-mapping modules in `/opt/llama.cpp/conversion/` (15,138 LOC). hf2q's `src/inference/models/<arch>/` covers only the test-matrix subset. For every supported arch, our mapping must produce byte-identical F32 values to llama.cpp's at the boundary, or P1 byte-cmp fails.

**Mitigation:** Mapping parity is an explicit P0 gate (not a P1 surprise). Each arch's parity check is a fixture test, generated once from llama.cpp's pipeline and checked in. Drift over time is caught by a per-release re-generation.

### Risk 2 — Metal-vs-CPU activation order in imatrix (Pi)

**What:** hf2q-imatrix runs on Metal kernels (via mlx-native); llama-imatrix's reference runs on its ggml CPU backend. FP accumulation order differs between architectures even when the algorithm is mathematically identical. Strict byte-cmp may be unsatisfiable.

**Mitigation:** Verify spike at Pi's start (run llama-imatrix on CPU vs Metal; cmp). If achievable, byte-cmp stands. If not, ADR amends to numeric-cmp (1e-6 relative tolerance) before Pi continues. Decision is empirically driven, not speculated.

### Risk 3 — `mudler/apex-quant` recipe drifts upstream

**What:** Mudler is an active GitHub repo. If they change a tier's tensor-type-file content after we port it, our `apex-quality` output drifts from theirs.

**Mitigation:** Pin to a specific mudler commit SHA in `src/quantize/apex/rules.rs`. Update is a deliberate ADR amendment, not a silent CI refresh. Tracker issue for "ported from mudler@<sha>; check upstream quarterly."

## Explicitly NOT doing (v1)

- **TQ1_0 / TQ2_0** (BitNet ternary). `--quant tq1_0 / tq2_0` returns "out of v1 scope" typed error. Tracked separately.
- **Split-file output** (`--split-max-size`, `--keep-split`). Single-file GGUF only. Users can post-split with `llama-gguf-split`.
- **PPL-parity fallback gate.** Byte-cmp is the only acceptance gate; tensors that don't byte-cmp are bugs, not "acceptable drift."
- **Apple MLX `dwq.py` distillation port.** `--quant dwq` is reserved with a typed-error stub. The full port is a future ADR ("real DWQ").
- **hf2q's existing homebrew DWQ** (`DwqKQuantizer` + `src/calibrate/dwq.rs`). Deleted in P6. No production artifact uses it; the name was misleading; the real DWQ lands separately.
- **`SensitivityMixedPolicy`** (the 2026-05-17 draft's rename of DwqKQuantizer). Doesn't exist in this rewrite. Two policies, not three.
- **`apex` unqualified** as a CLI value. Tier must be explicit; ADR-014 P8 D13's reasoning still applies.
- **Header reservation size as an ADR-level concern.** Implementation detail; writer picks an appropriate size.
- **Explicit FMA / fast-math enforcement plumbing.** P1 byte-cmp tests catch any drift empirically; that's the gate. Codifying a build-flag policy adds plumbing without adding signal.
- **Timeline.** Tracked separately; ADRs document decisions, not project plans.

## Open Issues / Real-Model Findings

### 2026-05-18 — Convert-v2 OOM on real 26B model (FIXED at the same commit)

After the Gemma 4 mapper rewrite shipped (mlx-native `93383cd`, hf2q `46c54876`) and all 4 integration tests + 33 unit tests passed on synthetic fixtures, **four** real-model convert attempts against `google/gemma-4-26b-a4b-it` (48 GB BF16 safetensors → ~18 GB Q5_K_M GGUF target) were SIGKILL'd by the macOS memory manager (exit 137) on a 64 GB Mac. The fourth attempt at Q8_0 also failed. Root cause: the buffered source-reader and orchestrator together allocated `~2 × model_safetensors_size` of F32 working buffers (BF16 → F32 doubles every element) **PLUS** the orchestrator's `Vec<StagedTensor>` held a second copy **PLUS** its `Vec<Prepared>` held a third copy of every quantized payload before any byte hit disk. Peak working set was on the order of `2 × 48 GB + 48 GB + 18 GB ≈ 162 GB` against a 64 GB physical memory budget.

**Fix landed at this commit:**

1. **`HfModelSource::open` replaces `HfModelSource::load`.** The source reader now mmaps each safetensors shard and records only `(name, shape, dtype, shard_idx, byte_offset, byte_len)` metadata up-front. No payload bytes resident in heap.
2. **`HfModelSource::iter_tensors() -> TensorStream<'_>` and `materialize_tensor(name)`.** One tensor's bytes are sliced out of its shard mmap, dequantized to F32 in a fresh `Vec<f32>`, and yielded; the previous tensor's buffer drops before the next allocation.
3. **`ConvertOrchestrator` switched to a two-phase streaming API.** `plan_tensors(Vec<PlanEntry>)` runs the policy pre-pass + per-tensor `target_for` on metadata-only entries. `begin_write(writer) -> StreamingWriter` emits the GGUF header + every KV + every tensor-info reservation. `StreamingWriter::stream_tensor(idx, &[f32])` quantizes inline and writes the payload, discarding both buffers within one call. `StreamingWriter::finalize()` seek-backs offsets.
4. **MoE expert fusion stays streaming.** The driver's plan-phase builds a `ConvertPlan` whose `PlanStep::Fused` entries list the HF expert-slice names in `expert_index` order. The stream phase loads N slices for ONE `(layer, kind)` group, concatenates their F32 buffers, streams the fused payload, and drops the temp before moving to the next group. Peak per-group memory ≈ `n_experts × per_expert_F32_bytes` (Gemma 4 26B: 128 × ~10 MB ≈ 1.3 GB per group — fits easily).

**Memory bound tightened in §P0** from `2 × model_safetensors_size + 512 MiB` to `4 × largest_single_tensor_F32_size + 512 MiB`. For Gemma 4 26B the largest tensor is `ffn_down` at `[2112, 2560]` BF16 → ~20 MB F32 + ~13 MB Q5_K_M payload, giving a bound around `~600 MB` instead of `~96 GB`. The original `2 × model_safetensors_size` envelope was always going to be infeasible on commodity hardware for 26B+ models even before the buffered-Vec antipattern compounded it; tensor-by-tensor is the correct shape of the bound.

**Validation:** the regression test `tests/convert_v2_integration.rs::convert_v2_streaming_rss_under_bound_2026_05_18` spawns convert-v2 under `/usr/bin/time -l` (macOS) / `time -f "%M"` (Linux), parses the OS-reported peak RSS, and asserts `peak < 4 × largest_F32_size + 512 MiB`. Pre-fix this test would have overshot by ~64 MB on its small fixture and by ~104 GB on Gemma 4 26B.

**Real-model re-run:** the operator should re-attempt `hf2q convert-v2 /opt/hf2q/models/google-gemma-4-26b-a4b-it --quant q5_k_m -o gemma4-26b-q5_k_m.gguf` on a 64 GB system after this commit lands. Expected peak RSS: well under 4 GB (mmap'd safetensors pages are anonymous-cache, not RSS-counted on macOS / Linux; the heap holds at most one F32 + one Q5_K_M payload at a time).

### 2026-05-18 — Tokenizer metadata missing from convert-v2 output (FIXED at the same commit)

After the streaming-OOM fix above produced a valid 18 GB Q5_K_M GGUF in 8m 22s (peak footprint 4.94 GB) from `google/gemma-4-26b-a4b-it`, `llama-cli -m <output>` rejected the file with `error loading model vocabulary: key not found in model: tokenizer.ggml.model`. Inspection: the convert-v2 output had 24 KV pairs, all `gemma4.*` or `general.*`, and **zero** `tokenizer.*` entries. The legacy `cmd_convert` pipeline emits the full tokenizer block from `src/backends/gguf.rs::load_tokenizer_metadata` (lines 2742-3200), but convert-v2 never wired in an equivalent — `run_convert_v2` jumped straight from `build_metadata_for_arch` to `begin_write` and dropped tokenizer-parse from its responsibilities.

**Fix landed at this commit — new `src/convert/tokenizer.rs` module + cli_driver integration.** Surface:

- **`build_tokenizer_metadata(model_dir: &Path, arch: ArchName) -> Result<Vec<(String, MetaValue)>, TokenizerError>`** ports the legacy emitter's logic into a focused, convert-v2-only module. Reads `tokenizer.json` + `tokenizer_config.json`, merges base BPE + `added_tokens`, cross-checks against `config.json::vocab_size` (or `text_config.vocab_size` for multimodal-wrapper configs), classifies token types via `LlamaHfVocab` rules + Gemma 4's USER_DEFINED `visible_tokens` set (gemma.py:630-642), resolves BOS/EOS/UNK/PAD ids in the merged vocab, and emits 11-13 GGUF KVs per arch.
- **`TokenizerError`** is a typed-error surface — per [[feedback-no-loop-suppression-2026-05-17]] no silent fallback. Variants: `TokenizerJsonMissing`, `TokenizerJsonMalformed`, `TokenizerJsonMissingModel`, `ConfigMissingVocabSize`, `SpecialTokenUnresolvable`, `AddedTokenIdOutOfRange`. Each variant matches one of the silent-corruption failure modes that produced the 2026-04-30 DWQ48/46 truncated-vocab regression.
- **Per-arch `tokenizer.ggml.model` dispatch (gemma.py:649 + legacy `determine_tokenizer_model_name`):** Gemma 4 → `"gemma4"` (unconditional, gemma.py:649); BPE + byte_fallback → `"llama"` (SentencePiece-style); BPE without byte_fallback → `"gpt2"`.
- **Per-arch `tokenizer.ggml.pre` dispatch (llama-vocab.cpp:1948-2061):** `Qwen35Moe → qwen35`, `Qwen3VlText → qwen2`, `Gemma4 / Gemma4Mmproj → gemma4`, `Llama3 / MiniMaxM2 → llama-bpe`, `Bert / NomicBert → default`.
- **Flags fixed per gemma.py:652-653:** `add_bos_token = true`, `add_space_prefix = false`. The legacy emitter applied these unconditionally; every convert-v2 arch wants both.
- **Chat-template priority chain (ADR-012 chat-template-auto-inject 2026-04-30):** `chat_template.jinja` sidecar → `tokenizer_config.json[chat_template]` → `chat_templates::arch_default_chat_template(arch.name())` → graceful skip.

**Driver wiring at `src/convert/cli_driver.rs::run_convert_v2`:** new step 4b between `build_metadata_for_arch` and `plan_tensors` (~10 LOC). `ConvertV2Error::Tokenizer(TokenizerError)` variant added; `main.rs` cmd_convert_v2 routes it to `AppError::Input` (input-side typed-error class).

**Validation:**

1. **Unit tests** (`src/convert/tokenizer.rs::tests`): 8 tests covering the Gemma 4 emit happy path (model="gemma4", pre="gemma4", BOS/EOS ids, NORMAL/CONTROL classification), Llama 3 emit (model="llama", pre="llama-bpe"), Qwen35MoE pre-tokenizer dispatch, plus the 4 typed-error paths (missing tokenizer.json, missing vocab_size, unresolvable eos, byte-token / look-special heuristic pin).
2. **Integration test** (`tests/convert_v2_integration.rs::convert_v2_gemma4_real_arch_round_trip`): now drops a 64-id synthetic tokenizer fixture into the gemma4 model dir and asserts `tokenizer.ggml.model == "gemma4"`, `tokenizer.ggml.tokens.len() == 64`, `tokenizer.ggml.bos_token_id == 60`, `tokenizer.ggml.eos_token_id == 61`. Llama3 round-trip metadata count assertion bumped from 11 to 22 (11 arch KVs + 11 tokenizer KVs). New shared helper `write_minimal_tokenizer_fixture(dir, vocab_size)` writes a deterministic fixture for every existing fixture builder.
3. **Real-model load test:** `llama-cli -m /opt/hf2q/tmp/byte-cmp/gemma4-hf2q-q5_k_m.gguf -p "hi" -n 4 --no-warmup -no-cnv -ngl 999` now loads without the `key not found in model: tokenizer.ggml.model` error.

Per [[feedback-no-backwards-compat-2026-05-18]]: the new `src/convert/tokenizer.rs` is the canonical convert-v2 path; the legacy block at `src/backends/gguf.rs:2742-3200` remains load-bearing only because P6 has not yet retired `cmd_convert`. When P6 lands, the legacy emitter is **deleted**, not aliased.

## Open questions for the operator

All major open questions resolved in the 2026-05-18 interview. Remaining minor checkpoints (not blocking ADR finalization; resolve at the indicated phase boundary):

1. **MiniMax-M2.7 first fetch confirmation:** verified at `MiniMaxAI/MiniMax-M2.7` on HuggingFace, not gated, FP8 source format. Operator should download once before Pa starts to confirm the safetensors directory layout matches what `src/convert/arch/minimax_m2.rs` expects. If the upload changes between ADR draft and Pa start, re-verify.
2. **`carnice/Qwen3.6-MoE-MTP-abliterated` source resolution:** the operator's qwen3.6 abliterix production GGUF was produced from this model. Confirm before Pa starts that the source safetensors are available (HF or otherwise) so the per-model fingerprint-match for `carnice_qwen36_mtp_quality.txt` can be tested end-to-end (acceptance criterion 4a). If only the GGUF artifact is available (no safetensors), the per-model match for this fingerprint becomes documentation-only.
3. **Mudler's `nano` / `micro` configs vendoring scope:** v1 drops these from the CLI but the vendored `configs/` dir contains per-model nano + micro files. Operator should decide before P7 whether `data/apex-references/` ships the nano + micro files (accessible via `--quant apex-custom --tensor-type-file`) or filters them out at vendor time. Default if no preference: ship all per-model configs; let `apex-custom` consume any of them.

## Links

- **Supersedes:** ADR-014 (full supersession; streaming-convert property carried forward; CLI namespace reclaimed)
- **Future:** ADR-NNN (real-DWQ; Apple MLX `dwq.py` port; `--quant dwq` reserved here for that ADR)
- ADR-005 — inference server (downstream consumer of GGUFs we emit)
- ADR-012 — Qwen3.5-MoE conversion (predecessor; closure AC informs our parity gates)
- ADR-032 — Bug A / Bug B root-cause shipping (parallel work; same operator-mantra)
- `mudler/apex-quant` — GitHub: the canonical APEX recipe we're porting
- Auto-memory:
  - `[[apex-quant-definition-2026-05-17]]` — deep-research synthesis on what APEX actually is
  - `[[hf2q-convert-gemma4-f16-dispatch-2026-05-17]]` — the convert-side fixes that motivated this ADR (Bug A + Bug B both at root layer)
  - `[[cfa-adr033-review-2026-05-17]]` — 46-finding review of the 2026-05-17 draft; this rewrite addresses every blocker and major finding
  - `[[codex-review-loop-rule-2026-05-17]]` — invoke codex post-rewrite to verify
  - `[[no-loop-suppression-2026-05-17]]` — same root-cause philosophy applied here (no silent F16 fallbacks)
