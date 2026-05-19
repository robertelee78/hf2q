//! Per-arch MoE tensor classifier — given a `(ArchName, tensor_name)`
//! returns a `MoeTensorRole` so `ApexPolicy::target_for` can decide
//! which slot of the per-tier 7-tuple applies.
//!
//! Tensor-name conventions come from llama.cpp's GGUF naming (the
//! convert-side output names hf2q produces), cross-validated against
//! the vendored `vendor/apex-quant/configs/*.txt` files at pinned
//! mudler SHA `63c5048b7dc9ff230f2397d7bc445ca28894b769`. The configs
//! enumerate exactly these per-layer tensor suffixes (sans `.weight`):
//!
//! ```text
//! ffn_gate_exps     ffn_up_exps     ffn_down_exps     (routed)
//! ffn_gate_shexp    ffn_up_shexp    ffn_down_shexp    (shared)
//! attn_q  attn_k  attn_v  attn_output  attn_qkv  attn_gate
//! ssm_alpha  ssm_beta  ssm_out
//! ```
//!
//! v1 supports 3 MoE arches per ADR-033 Decision §6:
//!   - `Qwen35Moe`  — Qwen 3.5 / 3.6 MoE-A3B family
//!   - `Gemma4`     — but only when `n_expert > 1` (Gemma 4 is dense
//!                    by default; MoE variants exist)
//!   - `MiniMaxM2`  — MiniMax M2.7 (FP8 source)
//!
//! Other arches are typed-Err'd up front. Per
//! [[feedback-no-loop-suppression-2026-05-17]]: no silent demotion to
//! a dense quant path.

use super::super::tensor_ref::ArchName;

/// MoE-tensor role classification consumed by `ApexPolicy::target_for`.
/// Each role maps to one slot of the per-tier rule table (or to a
/// hard-coded `GgmlType` for global/structural tensors).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoeTensorRole {
    /// Routed sparse-expert tensor: `ffn_{gate,up,down}_exps.weight`.
    /// Picks `edge_exp / near_exp / mid_exp` from `TierRules`.
    RoutedExpert,
    /// Shared (always-active) expert tensor:
    /// `ffn_{gate,up,down}_shexp.weight`. Picks `edge_shared /
    /// mid_shared`.
    SharedExpert,
    /// Attention tensor: `attn_{q,k,v,output,qkv,gate}.weight` (and
    /// fused variants). Picks `edge_attn / mid_attn`.
    Attention,
    /// `token_embd.weight`. Mudler's standard is to keep token
    /// embedding at high precision (Q6_K) regardless of tier; this is
    /// implicit in the bash since `generate_config.sh` doesn't list
    /// `token_embd` (llama.cpp's quantize-tool default applies).
    TokenEmbd,
    /// `output.weight` / `lm_head.weight`. Same Q6_K-default
    /// rationale.
    Output,
    /// `blk.<i>.ffn_gate_inp.weight` — the per-token expert-router
    /// gate. Mudler standard is to leave it at Q5_0 (small,
    /// performance-critical, not in the bash config but llama.cpp's
    /// default applies).
    RouterGate,
    /// Norm tensors: `*_norm.weight`, `output_norm.weight`,
    /// `attn_norm.weight`, `ffn_norm.weight`, etc. Always emitted as
    /// F32 — norms are never quantized.
    Norm,
    /// SSM tensors (`ssm_alpha / ssm_beta / ssm_out`). Appear in
    /// `generate_config.sh:189-192` paired with attention's
    /// `attn_type`. Classified separately so the per-arch classifier
    /// can be explicit about which arch ships SSM (only some MoE
    /// arches use these).
    Ssm,
    /// Catch-all for tensors not in the explicit classifier set.
    /// `target_for` falls these back to the attention slot for the
    /// active region (mirroring mudler's convention of treating
    /// "everything else" as attention-class).
    Other,
}

/// v1 Apex supported arch CLI names. Used in error messages.
pub const SUPPORTED_APEX_ARCHES: &[&str] = &["qwen3moe", "gemma4", "minimax-m2"];

/// Classify a tensor by name + arch.
///
/// Per `vendor/apex-quant/configs/*.txt` the tensor-name suffixes
/// are arch-agnostic across the three supported arches (Qwen35Moe,
/// Gemma4-MoE, MiniMax-M2 all use llama.cpp's standard GGUF names).
/// The classifier therefore matches on **substring** patterns; the
/// `arch` parameter is reserved for future per-arch divergence
/// (and for the up-front "is this arch supported by Apex at all"
/// check at the policy layer).
///
/// Order matters: more-specific patterns (`shexp`, `exps`) checked
/// before less-specific (`ffn_*_exps` would match either if order
/// flipped). The classifier returns the first matching role.
pub fn classify_moe_tensor(arch: ArchName, name: &str) -> MoeTensorRole {
    // arch param reserved for future per-arch divergence; v1 uses a
    // unified suffix table for all 3 supported arches.
    let _ = arch;

    // --- Global tensors (no `blk.<i>.` prefix) ---
    // Check these first — they're exact-name matches at the root.
    if name == "token_embd.weight" || name == "per_layer_token_embd.weight" {
        return MoeTensorRole::TokenEmbd;
    }
    if name == "output.weight" || name == "lm_head.weight" {
        return MoeTensorRole::Output;
    }
    if name == "output_norm.weight" {
        return MoeTensorRole::Norm;
    }

    // --- Norms (any `*_norm.weight` substring) ---
    // Captures: attn_norm, ffn_norm, attn_q_norm, attn_k_norm,
    // ffn_gate_inp_norm (rare), output_norm (already caught above).
    if name.contains("_norm.weight") || name.ends_with(".norm.weight") {
        return MoeTensorRole::Norm;
    }

    // --- Router gate (per-token expert selector) — checked BEFORE
    //     `_shexp` to handle the `ffn_gate_inp_shexp` corner case ---
    // `blk.<i>.ffn_gate_inp.weight` selects which experts run for the
    // current token. `blk.<i>.ffn_gate_inp_shexp.weight` (QWEN3_NEXT
    // architecture) is the shared-expert router gate. Both are router
    // gates, not shared-expert weights. Codex 867dba20 review caught
    // the original ordering would misclassify the shexp variant.
    if name.contains("ffn_gate_inp") {
        return MoeTensorRole::RouterGate;
    }

    // --- MoE expert tensors ---
    // SHARED: mudler's `ffn_{gate,up,down}_shexp` tensors are always-
    // active per-token experts. The `shexp` substring is unique to
    // shared experts (after we've already filtered router gates above).
    if name.contains("_shexp.weight") || name.contains("_shexp_") {
        return MoeTensorRole::SharedExpert;
    }

    // ROUTED experts: `ffn_{gate,up,down}_exps.weight`. The `_exps`
    // suffix (note plural) distinguishes routed from shared. Be
    // careful NOT to match `shexp` here — we already caught those
    // above.
    if name.contains("_exps.weight") || name.contains("_exps_") {
        return MoeTensorRole::RoutedExpert;
    }

    // --- Attention ---
    // Substring matches mirror `standard_policy.rs::TensorCategory::classify`
    // priority order: fused QKV/KV_B before single Q/K/V.
    if name.contains("attn_qkv")
        || name.contains("attn_kv_b")
        || name.contains("attn_q.weight")
        || name.contains("attn_k.weight")
        || name.contains("attn_v.weight")
        || name.contains("attn_output")
        || name.contains("attn_gate.weight")
        || name.contains("attn_q_norm") // safety — caught by norm above anyway
        || name.contains("attn_k_norm")
    {
        // Re-route norm-suffix variants back to Norm in case the
        // first check missed them. Defense-in-depth.
        if name.contains("_norm.weight") {
            return MoeTensorRole::Norm;
        }
        return MoeTensorRole::Attention;
    }

    // --- SSM tensors (`ssm_alpha / ssm_beta / ssm_out`) ---
    // Appear in `generate_config.sh:189-192`. Currently no v1
    // production arch uses these, but the classifier honors them.
    if name.contains("ssm_alpha") || name.contains("ssm_beta") || name.contains("ssm_out") {
        return MoeTensorRole::Ssm;
    }

    // --- Catch-all ---
    // Anything not matched defaults to Other. `target_for` routes
    // Other through the attention slot (same as Ssm). This is
    // intentionally permissive — unknown tensors get a sane quant
    // rather than a hard error.
    MoeTensorRole::Other
}

/// Up-front "is this arch supported by Apex v1 at all" check.
/// Returns `true` for the supported MoE arches. Does NOT inspect
/// `n_expert` — that gate is enforced separately by `ApexPolicy::new`
/// at construction time (the dense-Gemma4 case errors with
/// `DenseModelNotSupported`).
pub const fn is_apex_supported_arch(arch: ArchName) -> bool {
    matches!(
        arch,
        ArchName::Qwen35Moe | ArchName::Gemma4 | ArchName::MiniMaxM2
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_routed_expert() {
        for suffix in ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"] {
            let name = format!("blk.5.{suffix}");
            assert_eq!(
                classify_moe_tensor(ArchName::Qwen35Moe, &name),
                MoeTensorRole::RoutedExpert,
                "{name}"
            );
        }
    }

    #[test]
    fn classify_shared_expert() {
        for suffix in [
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
        ] {
            let name = format!("blk.5.{suffix}");
            assert_eq!(
                classify_moe_tensor(ArchName::Qwen35Moe, &name),
                MoeTensorRole::SharedExpert,
                "{name}"
            );
        }
    }

    /// Codex 867dba20 review locked in: `ffn_gate_inp_shexp.weight`
    /// (QWEN3_NEXT shared-expert router gate) was misclassified as
    /// SharedExpert because `_shexp.weight` matched first. After fix,
    /// `ffn_gate_inp` is checked BEFORE shexp suffixes.
    #[test]
    fn classify_ffn_gate_inp_shexp_as_router_codex_867dba20() {
        assert_eq!(
            classify_moe_tensor(ArchName::Qwen35Moe, "blk.3.ffn_gate_inp_shexp.weight"),
            MoeTensorRole::RouterGate,
            "ffn_gate_inp_shexp.weight is a router gate (QWEN3_NEXT), not a shared expert"
        );
        // Plain ffn_gate_inp is still router (unchanged).
        assert_eq!(
            classify_moe_tensor(ArchName::Qwen35Moe, "blk.3.ffn_gate_inp.weight"),
            MoeTensorRole::RouterGate,
        );
    }

    #[test]
    fn classify_attention() {
        for suffix in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_qkv.weight",
        ] {
            let name = format!("blk.0.{suffix}");
            assert_eq!(
                classify_moe_tensor(ArchName::Qwen35Moe, &name),
                MoeTensorRole::Attention,
                "{name}"
            );
        }
    }

    #[test]
    fn classify_token_embd_and_output() {
        assert_eq!(
            classify_moe_tensor(ArchName::Qwen35Moe, "token_embd.weight"),
            MoeTensorRole::TokenEmbd
        );
        assert_eq!(
            classify_moe_tensor(ArchName::Qwen35Moe, "output.weight"),
            MoeTensorRole::Output
        );
    }

    #[test]
    fn classify_norms_as_norm() {
        for name in [
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.attn_q_norm.weight",
            "output_norm.weight",
        ] {
            assert_eq!(
                classify_moe_tensor(ArchName::Qwen35Moe, name),
                MoeTensorRole::Norm,
                "{name}"
            );
        }
    }

    #[test]
    fn classify_router_gate() {
        assert_eq!(
            classify_moe_tensor(ArchName::Qwen35Moe, "blk.0.ffn_gate_inp.weight"),
            MoeTensorRole::RouterGate
        );
    }

    #[test]
    fn supported_arches_set() {
        assert!(is_apex_supported_arch(ArchName::Qwen35Moe));
        assert!(is_apex_supported_arch(ArchName::Gemma4));
        assert!(is_apex_supported_arch(ArchName::MiniMaxM2));
        // Dense-only arches:
        assert!(!is_apex_supported_arch(ArchName::Llama3));
        assert!(!is_apex_supported_arch(ArchName::Bert));
        assert!(!is_apex_supported_arch(ArchName::NomicBert));
        assert!(!is_apex_supported_arch(ArchName::Qwen3VlText));
        assert!(!is_apex_supported_arch(ArchName::Falcon));
    }
}
