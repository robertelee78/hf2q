//! Intermediate F16+Q8_0 quantizer for the convert-time activation-capture
//! reload (ADR-012 P9b — apex MoE OOM mitigation).
//!
//! # Why this exists
//!
//! The convert pipeline emits an intermediate GGUF that is reloaded as a
//! `Qwen35Model` for activation calibration. If MoE expert tensors land at
//! F16, `Qwen35Model::load_from_gguf` routes them to the F32-expanded
//! `weight_loader::load_moe_ffn` path — that explodes 256 experts to ~128 GB,
//! which exceeds the 128 GB RAM ceiling and pushes the convert into swap
//! thrashing (empirically: 175 GB virtual footprint, 0.3% CPU, jetsam-killed).
//!
//! Routing MoE expert weights to Q8_0 in the intermediate keeps the loader
//! on the `MoeQ` path (`load_moe_ffn_quantized`), which holds native ggml
//! blocks on Metal directly — peak ~33 GB instead of ~128 GB.
//!
//! Q8_0 vs F16 for the intermediate is near-lossless: Q8_0 is symmetric
//! 8-bit per group of 32 with f16 scale, so the worst-case rounding error is
//! `absmax/127` per group. For the calibration forward (not the final
//! quantization), this is well below F16's own ~3-4 decimal-digit precision
//! noise. The final DWQ output bits are unaffected — DWQ re-quantizes from
//! the original `tensor_map`, not from the intermediate.
//!
//! # What gets routed
//!
//! Tensors whose name ends with `.mlp.experts.gate_proj.weight`,
//! `.mlp.experts.up_proj.weight`, or `.mlp.experts.down_proj.weight`
//! (post-merge names from Phase 1.5; see `src/models/qwen35/moe.rs:417-419`)
//! go through the Q8 quantizer. Everything else — token_embd, attention
//! projections, norms, router weights, shared-expert weights, dense FFN —
//! goes through the F16 quantizer (existing behavior, byte-identical to
//! the prior intermediate format for non-MoE tensors).

use crate::ir::{QuantizedTensor, TensorRef};
use crate::quantize::static_quant::StaticQuantizer;
use crate::quantize::{LayerQuantConfig, QuantizeError, Quantizer};

/// Hybrid intermediate quantizer: F16 for everything except MoE expert
/// tensors, which go to Q8_0. Used by `emit_gguf_from_tensor_map` when the
/// arch requires activation capture and the model is a MoE variant.
pub struct IntermediateMoeQ8Quantizer {
    f16: StaticQuantizer,
    q8: StaticQuantizer,
}

impl IntermediateMoeQ8Quantizer {
    /// Construct the hybrid quantizer. Both inner quantizers are infallible
    /// for the canonical method strings.
    pub fn new() -> Self {
        Self {
            f16: StaticQuantizer::new("f16").expect("StaticQuantizer::new(\"f16\") infallible"),
            q8: StaticQuantizer::new("q8").expect("StaticQuantizer::new(\"q8\") infallible"),
        }
    }

    /// Detect the post-merge MoE expert tensor name pattern. Matches both
    /// `model.layers.N.mlp.experts.{gate,up,down}_proj.weight` and any
    /// alternate prefix (e.g. without `model.`) — only the suffix is
    /// authoritative because Phase 1.4 strips `language_model.` from the
    /// front. Restricted to the three projection names emitted by
    /// `merge_moe_experts_in_tensor_map` so non-merged or non-MoE tensors
    /// never accidentally route here.
    fn is_moe_expert(name: &str) -> bool {
        name.ends_with(".mlp.experts.gate_proj.weight")
            || name.ends_with(".mlp.experts.up_proj.weight")
            || name.ends_with(".mlp.experts.down_proj.weight")
    }
}

impl Default for IntermediateMoeQ8Quantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Quantizer for IntermediateMoeQ8Quantizer {
    fn name(&self) -> &str {
        "intermediate-f16+moe-q8"
    }

    fn requires_calibration(&self) -> bool {
        false
    }

    fn quantize_tensor(
        &self,
        tensor: &TensorRef,
        config: &LayerQuantConfig,
    ) -> Result<QuantizedTensor, QuantizeError> {
        // Only weight tensors are eligible for Q8 routing — preserve flag
        // (norms / biases / scalars) wins regardless of the name pattern.
        if !config.preserve && Self::is_moe_expert(&tensor.name) {
            // Q8_0 with group_size=32 (ggml block size). The Q8 quantizer
            // takes its bits from `config.bits`; force 8 here so the caller
            // can keep passing `bits=16` for the F16 default.
            let q8_config = LayerQuantConfig {
                bits: 8,
                group_size: 32,
                preserve: false,
            };
            return self.q8.quantize_tensor(tensor, &q8_config);
        }
        self.f16.quantize_tensor(tensor, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, TensorRef};

    fn dummy_bf16_tensor(name: &str, numel: usize) -> TensorRef {
        // Allocate `numel` BF16 zeros so the quantizer treats it as a real
        // weight tensor. is_weight() returns true when name ends in `.weight`
        // and numel >= 32.
        TensorRef {
            name: name.to_string(),
            shape: vec![numel],
            dtype: DType::BF16,
            data: vec![0u8; numel * 2],
        }
    }

    fn cfg(bits: u8, group_size: usize) -> LayerQuantConfig {
        LayerQuantConfig {
            bits,
            group_size,
            preserve: false,
        }
    }

    #[test]
    fn routes_moe_expert_to_q8() {
        let q = IntermediateMoeQ8Quantizer::new();
        let t = dummy_bf16_tensor(
            "model.layers.5.mlp.experts.gate_proj.weight",
            64,
        );
        let out = q.quantize_tensor(&t, &cfg(16, 32)).unwrap();
        assert_eq!(out.quant_info.bits, 8);
        assert_eq!(out.quant_info.group_size, 32);
        assert!(!out.quant_info.preserved);
        assert_eq!(out.quant_info.method, "q8");
    }

    #[test]
    fn routes_moe_up_to_q8() {
        let q = IntermediateMoeQ8Quantizer::new();
        let t =
            dummy_bf16_tensor("model.layers.0.mlp.experts.up_proj.weight", 64);
        let out = q.quantize_tensor(&t, &cfg(16, 32)).unwrap();
        assert_eq!(out.quant_info.bits, 8);
    }

    #[test]
    fn routes_moe_down_to_q8() {
        let q = IntermediateMoeQ8Quantizer::new();
        let t = dummy_bf16_tensor(
            "model.layers.10.mlp.experts.down_proj.weight",
            64,
        );
        let out = q.quantize_tensor(&t, &cfg(16, 32)).unwrap();
        assert_eq!(out.quant_info.bits, 8);
    }

    #[test]
    fn routes_attention_to_f16() {
        let q = IntermediateMoeQ8Quantizer::new();
        let t = dummy_bf16_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            64,
        );
        let out = q.quantize_tensor(&t, &cfg(16, 32)).unwrap();
        // F16 path: passthrough preserved=true (StaticQuantizer f16 mode
        // produces preserved=true for all weights via convert_to_f16's
        // wrapping in QuantizedTensor with bits=16).
        assert_eq!(out.quant_info.bits, 16);
    }

    #[test]
    fn routes_shared_expert_to_f16_not_q8() {
        // shared_expert (singular) is the always-on shared FFN, NOT a
        // routed expert. It must not match the experts pattern.
        let q = IntermediateMoeQ8Quantizer::new();
        let t = dummy_bf16_tensor(
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            64,
        );
        let out = q.quantize_tensor(&t, &cfg(16, 32)).unwrap();
        assert_eq!(out.quant_info.bits, 16);
    }

    #[test]
    fn routes_router_to_f16() {
        let q = IntermediateMoeQ8Quantizer::new();
        let t = dummy_bf16_tensor("model.layers.0.mlp.gate.weight", 64);
        let out = q.quantize_tensor(&t, &cfg(16, 32)).unwrap();
        assert_eq!(out.quant_info.bits, 16);
    }

    #[test]
    fn routes_token_embd_to_f16() {
        let q = IntermediateMoeQ8Quantizer::new();
        let t = dummy_bf16_tensor("model.embed_tokens.weight", 64);
        let out = q.quantize_tensor(&t, &cfg(16, 32)).unwrap();
        assert_eq!(out.quant_info.bits, 16);
    }

    #[test]
    fn preserve_flag_overrides_q8_route() {
        // A norm tensor that happens to share the experts suffix would
        // still be force-preserved via config.preserve=true. The pattern
        // alone shouldn't override an explicit preserve.
        let q = IntermediateMoeQ8Quantizer::new();
        let t = dummy_bf16_tensor(
            "model.layers.0.mlp.experts.gate_proj.weight",
            64,
        );
        let preserve_cfg = LayerQuantConfig {
            bits: 16,
            group_size: 32,
            preserve: true,
        };
        let out = q.quantize_tensor(&t, &preserve_cfg).unwrap();
        // preserve=true → F16 quantizer → passthrough/F16
        assert_eq!(out.quant_info.bits, 16);
        assert!(out.quant_info.preserved);
    }

    #[test]
    fn name_check_is_suffix_anchored() {
        // A tensor whose name CONTAINS the suffix mid-string must NOT match.
        let t = dummy_bf16_tensor(
            "some.weird.mlp.experts.gate_proj.weight.suffix",
            64,
        );
        assert!(!IntermediateMoeQ8Quantizer::is_moe_expert(&t.name));
    }
}
