//! `StandardPolicy` — port of llama.cpp's standard quant-target picker.
//!
//! Per ADR-033 §"Plan" / P1 and Decision §"QuantPolicy trait (Decision
//! §3 concrete)". Implements two pieces:
//!
//! 1. **`tensor_type_fallback`** — port of
//!    `/opt/llama.cpp/src/llama-quant.cpp:362-408` (the row-misalignment
//!    first-downshift table). Per ADR §"shape_fallback contract" the
//!    second-misalignment case returns `Err` instead of silently
//!    demoting to F16. This file ships that port now.
//!
//! 2. **`StandardPolicy::target_for`** — port of
//!    `llama-quant.cpp:411-657` (`llama_tensor_get_type_impl`). The
//!    big ~247-LOC function with per-`(ftype, name, arch, category)`
//!    branching. Shipped over a follow-up commit; this file
//!    declares the struct + the trait stub now.

use super::error::QuantizeError;
use super::ggml_type::GgmlType;
use super::llama_ftype::LlamaFtype;
use super::tensor_ref::TensorRef;

/// Mirrors llama.cpp's `tensor_category` enum
/// (`/opt/llama.cpp/src/llama-quant.cpp` near line 99-150 per ADR-033
/// audit). Decides which `llama_tensor_get_type_impl` branch fires for
/// a given tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorCategory {
    /// `output.weight` / `output_norm.weight` — the LM head.
    Output,
    /// `token_embd.weight`.
    TokenEmbd,
    /// `blk.<i>.attn_v.weight`.
    AttnV,
    /// `blk.<i>.attn_q.weight`.
    AttnQ,
    /// `blk.<i>.attn_k.weight`.
    AttnK,
    /// `blk.<i>.attn_output.weight`.
    AttnOutput,
    /// `blk.<i>.ffn_gate(_inp|_exps)?.weight`.
    FfnGate,
    /// `blk.<i>.ffn_up(_exps)?.weight`.
    FfnUp,
    /// `blk.<i>.ffn_down(_exps)?.weight`.
    FfnDown,
    /// Catch-all (norms, biases, etc.) — F16/F32 pass-through.
    Other,
}

impl TensorCategory {
    /// Classify a tensor by name suffix patterns. Mirrors
    /// `tensor_category` at `llama-quant.cpp:99-150`.
    pub fn classify(name: &str) -> Self {
        // Output / token_embd are global (no `blk.` prefix).
        if name == "output.weight" || name == "output_norm.weight" {
            return TensorCategory::Output;
        }
        if name == "token_embd.weight" {
            return TensorCategory::TokenEmbd;
        }
        // Per-block patterns. Suffix-match in priority order.
        // (ffn_down comes before ffn_up before ffn_gate so the longer
        // names take precedence over shorter prefixes if they shared a
        // root — currently they don't, but order documents intent.)
        if name.ends_with(".attn_v.weight") {
            return TensorCategory::AttnV;
        }
        if name.ends_with(".attn_q.weight") {
            return TensorCategory::AttnQ;
        }
        if name.ends_with(".attn_k.weight") {
            return TensorCategory::AttnK;
        }
        if name.ends_with(".attn_output.weight") {
            return TensorCategory::AttnOutput;
        }
        if name.ends_with(".ffn_down.weight") || name.ends_with(".ffn_down_exps.weight") {
            return TensorCategory::FfnDown;
        }
        if name.ends_with(".ffn_up.weight") || name.ends_with(".ffn_up_exps.weight") {
            return TensorCategory::FfnUp;
        }
        if name.ends_with(".ffn_gate.weight")
            || name.ends_with(".ffn_gate_exps.weight")
            || name.ends_with(".ffn_gate_inp.weight")
        {
            return TensorCategory::FfnGate;
        }
        TensorCategory::Other
    }
}

/// `tensor_type_fallback` — port of
/// `/opt/llama.cpp/src/llama-quant.cpp:362-408`.
///
/// If `n_per_row % target.block_size() == 0` the target type is
/// returned unchanged. Otherwise the first-downshift table picks a
/// 32-aligned legacy variant of the same family:
///
/// | original  | first-downshift |
/// |-----------|-----------------|
/// | Q2_K, Q3_K, TQ1_0, TQ2_0 | Q4_0 |
/// | Q4_K      | Q5_0            |
/// | Q5_K      | Q5_1            |
/// | Q6_K      | Q8_0            |
/// | IQ*       | IQ4_NL          |
///
/// **Per ADR §"shape_fallback contract"**: if the downshift's
/// `block_size` still doesn't divide `n_per_row` (the C source's
/// "second misalignment" case where llama.cpp silently demotes to
/// F16), we return `Err(QuantizeError::NotBlockAligned)` instead.
/// F16 emission is reserved for vision/audio-pattern tensors and the
/// explicit `--quant f16` flag (handled by the upstream dispatcher,
/// not by this policy).
pub fn tensor_type_fallback(
    target: GgmlType,
    n_per_row: usize,
) -> Result<GgmlType, QuantizeError> {
    let target_block = target.block_size();
    if n_per_row % target_block == 0 {
        return Ok(target);
    }

    // First-downshift table per C:373-393.
    let downshift = match target {
        GgmlType::Q2_K | GgmlType::Q3_K => GgmlType::Q4_0,
        GgmlType::Q4_K => GgmlType::Q5_0,
        GgmlType::Q5_K => GgmlType::Q5_1,
        GgmlType::Q6_K => GgmlType::Q8_0,
        // IQ4_NL itself is already block-32; no downshift defined for it.
        // (C's switch lists IQ4_XS → IQ4_NL but we don't ship IQ4_XS in v1.)
        // F32/F16/BF16/Q8_0/Q8_K/legacy types — no fallback defined in C;
        // misalignment is a hard error here.
        _ => {
            return Err(QuantizeError::NotBlockAligned {
                ggml_type: target,
                n_per_row,
                block_size: target_block,
            });
        }
    };

    if n_per_row % downshift.block_size() != 0 {
        // C:394-403 falls back to F16 here. ADR §"shape_fallback
        // contract" makes this a typed error: F16 is reserved for
        // vision/audio tensors and explicit `--quant f16`.
        return Err(QuantizeError::NotBlockAligned {
            ggml_type: downshift,
            n_per_row,
            block_size: downshift.block_size(),
        });
    }

    Ok(downshift)
}

/// `StandardPolicy` — picks `GgmlType` per tensor at a given `LlamaFtype`.
/// Per ADR-033 Decision §"QuantPolicy trait (Decision §3 concrete)".
///
/// The body of `target_for` is the port of `llama_tensor_get_type_impl`
/// (`llama-quant.cpp:411-657`). Ships incrementally — this iteration
/// declares the struct + the trait method signature + an MVP path that
/// handles the simplest cases (primary type + tensor_type_fallback);
/// the per-arch / per-category override matrix lands in follow-up.
pub struct StandardPolicy {
    pub ftype: LlamaFtype,
}

impl StandardPolicy {
    pub const fn new(ftype: LlamaFtype) -> Self {
        Self { ftype }
    }

    /// Pick a `GgmlType` for `tensor` at this policy's `ftype`.
    ///
    /// MVP behavior (this commit):
    /// 1. Start from `ftype.primary_type()`.
    /// 2. Apply `tensor_type_fallback` for row misalignment.
    ///
    /// Follow-up commit ports the full per-(ftype, name, arch) override
    /// matrix from `llama_tensor_get_type_impl`. Until then, this
    /// policy is byte-equivalent to `(convert | llama-quantize)` only
    /// for the subset of tensors that don't hit category-override
    /// branches in the C source — i.e., not enough for the P1 byte-cmp
    /// gate. P1 acceptance waits for the full port.
    pub fn target_for(&self, tensor: &TensorRef) -> Result<GgmlType, QuantizeError> {
        let primary = self.ftype.primary_type();
        // F32/F16/BF16 pass-through ftypes — every tensor emits the same.
        if matches!(
            primary,
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
        ) {
            return Ok(primary);
        }
        // For quantizing ftypes, apply the shape-fallback contract.
        // (Per-category overrides land in the full port.)
        tensor_type_fallback(primary, tensor.n_per_row())
    }
}

#[cfg(test)]
mod tests {
    use super::super::tensor_ref::{ArchName, SourceDtype};
    use super::*;

    #[test]
    fn category_classify_basic() {
        assert_eq!(TensorCategory::classify("token_embd.weight"), TensorCategory::TokenEmbd);
        assert_eq!(TensorCategory::classify("output.weight"), TensorCategory::Output);
        assert_eq!(TensorCategory::classify("blk.0.attn_q.weight"), TensorCategory::AttnQ);
        assert_eq!(TensorCategory::classify("blk.10.attn_v.weight"), TensorCategory::AttnV);
        assert_eq!(TensorCategory::classify("blk.5.attn_k.weight"), TensorCategory::AttnK);
        assert_eq!(TensorCategory::classify("blk.3.attn_output.weight"), TensorCategory::AttnOutput);
        assert_eq!(TensorCategory::classify("blk.7.ffn_down.weight"), TensorCategory::FfnDown);
        assert_eq!(TensorCategory::classify("blk.7.ffn_down_exps.weight"), TensorCategory::FfnDown);
        assert_eq!(TensorCategory::classify("blk.7.ffn_up.weight"), TensorCategory::FfnUp);
        assert_eq!(TensorCategory::classify("blk.7.ffn_up_exps.weight"), TensorCategory::FfnUp);
        assert_eq!(TensorCategory::classify("blk.7.ffn_gate.weight"), TensorCategory::FfnGate);
        assert_eq!(TensorCategory::classify("blk.7.ffn_gate_exps.weight"), TensorCategory::FfnGate);
        assert_eq!(TensorCategory::classify("blk.7.ffn_gate_inp.weight"), TensorCategory::FfnGate);
        assert_eq!(TensorCategory::classify("blk.0.attn_norm.weight"), TensorCategory::Other);
    }

    #[test]
    fn fallback_passthrough_when_aligned() {
        // Q5_K block is 256; n_per_row=512 is aligned → no fallback.
        assert_eq!(tensor_type_fallback(GgmlType::Q5_K, 512).unwrap(), GgmlType::Q5_K);
        assert_eq!(tensor_type_fallback(GgmlType::Q4_K, 256).unwrap(), GgmlType::Q4_K);
        assert_eq!(tensor_type_fallback(GgmlType::Q4_0, 32).unwrap(), GgmlType::Q4_0);
    }

    #[test]
    fn fallback_q4_k_to_q5_0() {
        // Q4_K (256) misaligned at n_per_row=128 → Q5_0 (32) per C:387.
        assert_eq!(tensor_type_fallback(GgmlType::Q4_K, 128).unwrap(), GgmlType::Q5_0);
    }

    #[test]
    fn fallback_q5_k_to_q5_1() {
        // Q5_K (256) at n_per_row=160 → Q5_1 (32) per C:388.
        assert_eq!(tensor_type_fallback(GgmlType::Q5_K, 160).unwrap(), GgmlType::Q5_1);
    }

    #[test]
    fn fallback_q6_k_to_q8_0() {
        // Q6_K (256) at n_per_row=96 → Q8_0 (32) per C:389.
        assert_eq!(tensor_type_fallback(GgmlType::Q6_K, 96).unwrap(), GgmlType::Q8_0);
    }

    #[test]
    fn fallback_q2_q3_to_q4_0() {
        // Q2_K, Q3_K → Q4_0 per C:383-386.
        assert_eq!(tensor_type_fallback(GgmlType::Q2_K, 128).unwrap(), GgmlType::Q4_0);
        assert_eq!(tensor_type_fallback(GgmlType::Q3_K, 64).unwrap(), GgmlType::Q4_0);
    }

    #[test]
    fn fallback_second_misalignment_is_typed_error() {
        // Per ADR §"shape_fallback contract": if the downshift is ALSO
        // misaligned (n_per_row not div 32), return Err — NOT silent F16.
        // n_per_row=15 isn't aligned to 256 OR 32.
        let err = tensor_type_fallback(GgmlType::Q5_K, 15).unwrap_err();
        assert!(matches!(err, QuantizeError::NotBlockAligned { .. }));
    }

    #[test]
    fn fallback_no_path_for_unaligned_legacy() {
        // Q4_0 has no fallback in C's switch (block 32 is the floor).
        // n_per_row=17 is misaligned → typed error.
        let err = tensor_type_fallback(GgmlType::Q4_0, 17).unwrap_err();
        assert!(matches!(err, QuantizeError::NotBlockAligned { .. }));
    }

    #[test]
    fn standard_policy_passthrough_for_f32() {
        let shape: [usize; 2] = [4096, 1];
        let t = TensorRef {
            name: "blk.0.attn_q.weight",
            shape: &shape,
            source_dtype: SourceDtype::BF16,
            arch: ArchName::Llama3,
            layer_index: Some(0),
        };
        let p = StandardPolicy::new(LlamaFtype::AllF32);
        assert_eq!(p.target_for(&t).unwrap(), GgmlType::F32);
    }

    #[test]
    fn standard_policy_mvp_q5_k_m() {
        let shape: [usize; 2] = [4096, 1];
        let t = TensorRef {
            name: "blk.0.attn_q.weight",
            shape: &shape,
            source_dtype: SourceDtype::BF16,
            arch: ArchName::Llama3,
            layer_index: Some(0),
        };
        let p = StandardPolicy::new(LlamaFtype::MostlyQ5_K_M);
        // MVP path: primary_type() == Q5_K; n_per_row=4096 is aligned → Q5_K.
        // (Full target_for port will apply per-category overrides; this MVP
        // is intentionally byte-divergent from `llama-quantize` for tensors
        // that trigger override branches.)
        assert_eq!(p.target_for(&t).unwrap(), GgmlType::Q5_K);
    }
}
