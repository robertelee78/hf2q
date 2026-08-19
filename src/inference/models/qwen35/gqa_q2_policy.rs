//! CPU-only policy for selecting the cooperative Qwen GQA-Q2 attention route.
//!
//! Keep policy parsing and geometry tests separate from `kv_cache`: every test
//! in that Metal-owning module is conservatively required to hold the shared
//! GPU test lock. This module contains no device, buffer, encoder, or kernel
//! state and its tests are therefore safe to run in parallel.

/// The isolated M5 Max kernel A/B first covered 8K and above. Keep shorter
/// requests on the scalar kernel until a separate short-context gate proves a
/// lower crossover. The specialized PSO is compiled during model warmup.
const GQA_Q2_MIN_KV_SEQ_LEN: u32 = 8_192;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GqaQ2Mode {
    Off,
    Auto,
    On,
}

fn parse_gqa_q2_mode(value: Option<&str>) -> GqaQ2Mode {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("0" | "off" | "false") => GqaQ2Mode::Off,
        Some("1" | "on" | "true") => GqaQ2Mode::On,
        Some("auto") | None => GqaQ2Mode::Auto,
        Some(other) => {
            tracing::warn!(
                value = other,
                "invalid HF2Q_QWEN_GQA_Q2 value; disabling cooperative attention"
            );
            GqaQ2Mode::Off
        }
    }
}

pub(super) fn gqa_q2_mode() -> GqaQ2Mode {
    static MODE: std::sync::OnceLock<GqaQ2Mode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| parse_gqa_q2_mode(std::env::var("HF2Q_QWEN_GQA_Q2").ok().as_deref()))
}

pub(super) fn use_gqa_q2_tq_sdpa(params: &Qwen35TqSdpaParams) -> bool {
    use_gqa_q2_tq_sdpa_for_mode(params, gqa_q2_mode())
}

fn use_gqa_q2_tq_sdpa_for_mode(params: &Qwen35TqSdpaParams, mode: GqaQ2Mode) -> bool {
    mode != GqaQ2Mode::Off
        && (mode == GqaQ2Mode::On || params.kv_seq_len >= GQA_Q2_MIN_KV_SEQ_LEN)
        && params.head_dim == 256
        && params.num_kv_heads > 0
        && params
            .num_attention_heads_per_kv()
            .is_some_and(|group| group % 2 == 0)
        && params.mask_type == 0
        && params.sliding_window == 0
        && params.ring_start == 0
        && params.softcap == 0.0
        && matches!(params.codebook_bits, 5 | 6 | 8)
}

/// Parameters for the qwen35 TQ SDPA dispatch.
///
/// Mirrors `mlx_native::ops::flash_attn_vec_tq_hb::FlashAttnVecTqHbParams`
/// but remains in the qwen35 namespace so engine code does not depend on the
/// mlx-native parameter type directly.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35TqSdpaParams {
    /// Q heads (e.g. 16 for qwen36 35B-A3B-APEX).
    pub num_heads: u32,
    /// K/V heads (e.g. 2 for qwen36).
    pub num_kv_heads: u32,
    /// head_dim (must be 256 or 512; production qwen35 = 256).
    pub head_dim: u32,
    /// Number of KV positions populated (cur_len at dispatch time).
    pub kv_seq_len: u32,
    /// Cache capacity (max_seq_len from `HybridKvCache` construction).
    pub kv_capacity: u32,
    /// Scale (typically `1 / sqrt(head_dim)`).
    pub scale: f32,
    /// Mask type: 0 = none, 1 = causal, 2 = sliding-window.
    pub mask_type: u32,
    /// Sliding window length (mask_type=2 only).
    pub sliding_window: u32,
    /// Softcap value (0 = disabled).
    pub softcap: f32,
    /// Ring buffer start slot for sliding-window cache (0 for global).
    pub ring_start: u32,
    /// D=512 per-block scale divisor (1.0 for d=256 = qwen35 production).
    pub scale_factor_d512: f32,
    /// Codebook bit-width (5, 6, or 8 — qwen35 default = 8).
    pub codebook_bits: u32,
}

impl Qwen35TqSdpaParams {
    fn num_attention_heads_per_kv(self) -> Option<u32> {
        (self.num_kv_heads > 0 && self.num_heads % self.num_kv_heads == 0)
            .then_some(self.num_heads / self.num_kv_heads.max(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Qwen35TqSdpaParams {
        Qwen35TqSdpaParams {
            num_heads: 24,
            num_kv_heads: 4,
            head_dim: 256,
            kv_seq_len: 8_192,
            kv_capacity: 262_144,
            scale: 1.0 / 16.0,
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: 8,
        }
    }

    #[test]
    fn qwen38_geometry_selects_q2_at_measured_crossover() {
        let mut p = params();
        p.kv_seq_len = GQA_Q2_MIN_KV_SEQ_LEN - 1;
        assert!(!use_gqa_q2_tq_sdpa_for_mode(&p, GqaQ2Mode::Auto));
        p.kv_seq_len = GQA_Q2_MIN_KV_SEQ_LEN;
        assert!(use_gqa_q2_tq_sdpa_for_mode(&p, GqaQ2Mode::Auto));
        assert!(!use_gqa_q2_tq_sdpa_for_mode(&p, GqaQ2Mode::Off));
        p.kv_seq_len = 1;
        assert!(use_gqa_q2_tq_sdpa_for_mode(&p, GqaQ2Mode::On));
    }

    #[test]
    fn unsupported_geometry_and_attention_modes_fall_back() {
        let mut p = params();
        p.head_dim = 512;
        assert!(!use_gqa_q2_tq_sdpa_for_mode(&p, GqaQ2Mode::On));
        p = params();
        p.num_heads = 20;
        p.num_kv_heads = 4;
        assert!(!use_gqa_q2_tq_sdpa_for_mode(&p, GqaQ2Mode::On));
        p = params();
        p.mask_type = 2;
        p.sliding_window = 4_096;
        assert!(!use_gqa_q2_tq_sdpa_for_mode(&p, GqaQ2Mode::On));
        p = params();
        p.codebook_bits = 4;
        assert!(!use_gqa_q2_tq_sdpa_for_mode(&p, GqaQ2Mode::On));
    }

    #[test]
    fn operator_mode_parser_is_explicit_and_fail_safe() {
        assert_eq!(parse_gqa_q2_mode(None), GqaQ2Mode::Auto);
        assert_eq!(parse_gqa_q2_mode(Some("auto")), GqaQ2Mode::Auto);
        assert_eq!(parse_gqa_q2_mode(Some("off")), GqaQ2Mode::Off);
        assert_eq!(parse_gqa_q2_mode(Some("0")), GqaQ2Mode::Off);
        assert_eq!(parse_gqa_q2_mode(Some("on")), GqaQ2Mode::On);
        assert_eq!(parse_gqa_q2_mode(Some("1")), GqaQ2Mode::On);
        assert_eq!(parse_gqa_q2_mode(Some("invalid")), GqaQ2Mode::Off);
    }
}
