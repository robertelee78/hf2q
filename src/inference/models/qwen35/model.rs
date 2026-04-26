//! Top-level `Qwen35Model` struct — the load target for Qwen3.5 / Qwen3.5-MoE
//! GGUFs.
//!
//! Ties together every component that preceding phases delivered:
//!
//! - [`Qwen35Config`](super::Qwen35Config) from the GGUF metadata parser.
//! - Per-layer weights: `Qwen35LayerWeights` enum (FullAttn or LinearAttn).
//! - FFN variant: `Qwen35FfnWeights` enum (dense SwiGLU or MoE).
//! - Optional [`MtpWeights`](super::mtp::MtpWeights) for MTP speculative decoding.
//! - Global tensors: `token_embd`, `output_weight`, `output_norm`.

use anyhow::{anyhow, Context, Result};

use super::delta_net::DeltaNetLayerWeights;
use super::ffn::{DenseFfnWeights, MoeFfnWeights};
use super::full_attn::FullAttnLayerWeights;
use super::mtp::{load_mtp_weights_if_present, MtpWeights};
use super::weight_loader::{DenseFfnWeightsQ, MoeFfnWeightsQ};
use super::{weight_loader, Qwen35Config, Qwen35LayerKind, Qwen35Variant};

use mlx_native::gguf::GgufFile;
use mlx_native::MlxDevice;

// ================================================================
// Layer weight enums
// ================================================================

/// FFN weights for a single Qwen3.5 layer. Qwen3.5 dense uses SwiGLU;
/// Qwen3.5-MoE uses a 256-expert router with a sigmoid-gated shared
/// expert. Exactly one variant is populated per layer per model.
pub enum Qwen35FfnWeights {
    Dense(DenseFfnWeights),
    /// Quantized dense SwiGLU weights loaded directly from GGUF (production path).
    /// Gate/up/down tensors stay in GGML block format on the Metal device;
    /// no F32 expansion occurs during load. Used for 27B dense DWQ GGUFs.
    DenseQ(DenseFfnWeightsQ),
    /// F32-dequantized MoE weights (unit tests / synthetic models only).
    Moe(MoeFfnWeights),
    /// Quantized MoE weights loaded directly from GGUF (production path).
    /// Expert tensors stay in GGML block format on the Metal device;
    /// no F32 expansion occurs during load.
    MoeQ(MoeFfnWeightsQ),
}

impl Qwen35FfnWeights {
    pub fn variant(&self) -> &'static str {
        match self {
            Qwen35FfnWeights::Dense(_) => "dense",
            Qwen35FfnWeights::DenseQ(_) => "dense-q",
            Qwen35FfnWeights::Moe(_) => "moe",
            Qwen35FfnWeights::MoeQ(_) => "moe-q",
        }
    }
}

/// Per-layer weight bundle. The attention variant is determined by the
/// `Qwen35LayerKind` at the corresponding position in `cfg.layer_types`;
/// the FFN variant is determined by `cfg.variant` (Dense or Moe) and is
/// the same across all layers.
pub enum Qwen35LayerWeights {
    /// Gated full-attention layer (every 4th layer for Qwen3.5).
    FullAttn {
        attn: FullAttnLayerWeights,
        ffn: Qwen35FfnWeights,
    },
    /// Linear-attention (Gated DeltaNet) layer.
    LinearAttn {
        attn: DeltaNetLayerWeights,
        ffn: Qwen35FfnWeights,
    },
}

impl Qwen35LayerWeights {
    pub fn kind(&self) -> Qwen35LayerKind {
        match self {
            Qwen35LayerWeights::FullAttn { .. } => Qwen35LayerKind::FullAttention,
            Qwen35LayerWeights::LinearAttn { .. } => Qwen35LayerKind::LinearAttention,
        }
    }

    pub fn ffn(&self) -> &Qwen35FfnWeights {
        match self {
            Qwen35LayerWeights::FullAttn { ffn, .. } => ffn,
            Qwen35LayerWeights::LinearAttn { ffn, .. } => ffn,
        }
    }
}

// ================================================================
// Top-level model
// ================================================================

/// Complete Qwen3.5 / Qwen3.5-MoE model: config + weights + MTP.
///
/// # Construction
///
/// Use [`Qwen35Model::load_from_gguf`] to populate from a GGUF file;
/// or [`Qwen35Model::empty_from_cfg`] for tests that construct zero-weight
/// models of a given shape.
pub struct Qwen35Model {
    pub cfg: Qwen35Config,
    /// Per-layer weight bundles. `len() == cfg.num_hidden_layers`.
    pub layers: Vec<Qwen35LayerWeights>,
    /// Token-embedding table (vocab_size × hidden_size, row-major).
    pub token_embd: Vec<f32>,
    /// LM-head output projection (hidden_size × vocab_size, row-major per
    /// GGUF's `[out_dim, in_dim]` convention).
    pub output_weight: Vec<f32>,
    /// Final-layer RMSNorm weight, shape `[hidden_size]`.
    pub output_norm: Vec<f32>,
    /// Optional MTP draft block. Executed only by speculative decoding, never
    /// by the verifier's main layer loop.
    pub mtp: Option<MtpWeights>,
}

impl Qwen35Model {
    /// Parse only the [`Qwen35Config`] from a GGUF without allocating
    /// weights. Cheap — useful for introspection (e.g. `hf2q info`).
    pub fn load_config_only(gguf: &GgufFile) -> Result<Qwen35Config> {
        Qwen35Config::from_gguf(gguf).context("Qwen35Config::from_gguf")
    }

    /// Construct a zero-weighted model of the shape prescribed by `cfg`.
    ///
    /// Useful for test harnesses that want to populate weights by hand
    /// (e.g. fuzzing per-layer forward with synthetic data). The MTP
    /// field is `None` (no MTP in an empty model).
    pub fn empty_from_cfg(cfg: Qwen35Config) -> Self {
        let h = cfg.hidden_size as usize;
        let vocab = cfg.vocab_size as usize;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for kind in &cfg.layer_types {
            let ffn = empty_ffn_for(&cfg);
            let layer = match kind {
                Qwen35LayerKind::FullAttention => Qwen35LayerWeights::FullAttn {
                    attn: empty_full_attn_weights(&cfg),
                    ffn,
                },
                Qwen35LayerKind::LinearAttention => Qwen35LayerWeights::LinearAttn {
                    attn: empty_delta_net_weights(&cfg),
                    ffn,
                },
            };
            layers.push(layer);
        }

        Self {
            layers,
            token_embd: vec![0.0f32; vocab * h],
            output_weight: vec![0.0f32; h * vocab],
            output_norm: vec![1.0f32; h],
            mtp: None,
            cfg,
        }
    }

    /// Load a complete model from a GGUF file.
    ///
    /// Acquires a Metal GPU device for dequantization, loads the three
    /// global tensors (`token_embd`, `output`, `output_norm`), then
    /// iterates over all layers.
    ///
    /// For MoE models the expert weight tensors (`ffn_{gate,up,down}_exps`)
    /// are kept in their native GGML block quantization via
    /// [`weight_loader::load_moe_ffn_quantized`].  This prevents the ~128 GB
    /// Metal working-set OOM that an F32-expanded load would cause for the
    /// 35B-A3B apex model.
    ///
    /// For Dense models the behaviour is unchanged: weights are dequantized
    /// to f32 via [`weight_loader::load_layer`].
    pub fn load_from_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut cfg = Self::load_config_only(gguf)?;

        let device = MlxDevice::new()
            .map_err(|e| anyhow!("MlxDevice::new for weight loading: {e}"))?;

        let (token_embd, output_weight, output_norm) =
            weight_loader::load_global_tensors(gguf, &cfg, &device)
                .context("load_global_tensors")?;

        // ADR-012 P9b real-model finding (Qwen3.6-27B): the embedding table is
        // physically padded for alignment (e.g. 248320 rows) while the metadata
        // vocab_size reports the logical vocab (e.g. 248044). When they
        // disagree, take the tensor shape as authoritative for cfg.vocab_size
        // — the io_heads.rs assertion `token_embd.len() == vocab * hidden`
        // and the LM head matmul both require the row-count match. The logical
        // vocab is still recoverable from `tokenizer.ggml.tokens` metadata if
        // ever needed (e.g. for sampler masking of pad rows).
        let h = cfg.hidden_size as usize;
        if h > 0 {
            let physical_vocab = token_embd.len() / h;
            if (physical_vocab as u32) != cfg.vocab_size {
                tracing::info!(
                    metadata_vocab = cfg.vocab_size,
                    physical_vocab = physical_vocab,
                    "qwen35 vocab pad: metadata reports {} but token_embd has {} rows; using physical for cfg.vocab_size",
                    cfg.vocab_size,
                    physical_vocab,
                );
                cfg.vocab_size = physical_vocab as u32;
            }
        }

        let mtp = load_mtp_weights_if_present(gguf, &cfg, &device)
            .context("load_mtp_weights_if_present")?;

        // ADR-012 item-2 architectural fix (2026-04-25): MoE experts MUST
        // be loaded as native ggml-quantized blocks (`MoeQ`). The previous
        // F16-detection / F32-expand fallback ("Moe" variant via
        // `weight_loader::load_moe_ffn`) was peer-misaligned — peers
        // (mlx-lm, llama.cpp, AutoAWQ) never F32-expand MoE experts at load
        // time. Apex 35B-A3B at F32 is ~128 GB which doesn't fit on a
        // 128 GB system; the convert pipeline now emits MoE experts at
        // Q8_0 in the intermediate (see `quantize::intermediate_moe_q8`)
        // so this branch is unreachable for production inputs. If a caller
        // ever supplies F16/F32 experts (e.g. legacy GGUFs), we fail loud
        // at load time rather than silently expanding.
        use mlx_native::ops::quantized_matmul_ggml::GgmlType;
        if cfg.variant == Qwen35Variant::Moe {
            if let Some(info) = gguf.tensor_info("blk.0.ffn_gate_exps.weight") {
                if matches!(info.ggml_type, GgmlType::F16 | GgmlType::F32) {
                    return Err(anyhow!(
                        "qwen35moe load: MoE expert tensor 'blk.0.ffn_gate_exps.weight' \
                         is dtype {:?}; native ggml-block quantization (Q4_0, Q5_K, Q6_K, Q8_0) \
                         is required. Re-emit the GGUF with quantized MoE experts — no \
                         F32-expansion fallback per ADR-012 item-2 (peer alignment).",
                        info.ggml_type
                    ));
                }
            }
        }

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            let layer = match cfg.variant {
                Qwen35Variant::Moe => {
                    let kind = cfg
                        .layer_types
                        .get(i as usize)
                        .copied()
                        .ok_or_else(|| anyhow!("layer_idx {i} out of range"))?;
                    // Production quantized experts (Q4_0/Q5_K/Q6_K/Q8_0) —
                    // keep native blocks on Metal; no F32 expansion.
                    let ffn_weights = {
                        let ffn = weight_loader::load_moe_ffn_quantized(gguf, i, &device)
                            .with_context(|| format!("load_moe_ffn_quantized layer {i}"))?;
                        Qwen35FfnWeights::MoeQ(ffn)
                    };
                    match kind {
                        Qwen35LayerKind::FullAttention => {
                            let attn = weight_loader::load_full_attn_layer(gguf, &cfg, i, &device)
                                .with_context(|| format!("load_full_attn layer {i}"))?;
                            Qwen35LayerWeights::FullAttn { attn, ffn: ffn_weights }
                        }
                        Qwen35LayerKind::LinearAttention => {
                            let attn = weight_loader::load_delta_net_layer(gguf, &cfg, i, &device)
                                .with_context(|| format!("load_delta_net layer {i}"))?;
                            Qwen35LayerWeights::LinearAttn { attn, ffn: ffn_weights }
                        }
                    }
                }
                Qwen35Variant::Dense => {
                    weight_loader::load_layer(gguf, &cfg, i, &device)
                        .with_context(|| format!("load_layer {i}"))?
                }
            };
            layers.push(layer);
        }

        Ok(Self {
            cfg,
            layers,
            token_embd,
            output_weight,
            output_norm,
            mtp,
        })
    }

    /// Report the active FFN variant (Dense or Moe) determined by config.
    pub fn ffn_variant(&self) -> Qwen35Variant {
        self.cfg.variant
    }

    /// Per-layer metadata helper: number of linear-attention layers.
    pub fn num_linear_attn_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l, Qwen35LayerWeights::LinearAttn { .. }))
            .count()
    }

    /// Per-layer metadata helper: number of full-attention layers.
    pub fn num_full_attn_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l, Qwen35LayerWeights::FullAttn { .. }))
            .count()
    }

    /// Per-layer kind lookup.
    pub fn layer_kind(&self, idx: u32) -> Option<Qwen35LayerKind> {
        self.layers.get(idx as usize).map(|l| l.kind())
    }
}

// ================================================================
// Empty-weight constructors (for tests)
// ================================================================

fn empty_full_attn_weights(cfg: &Qwen35Config) -> FullAttnLayerWeights {
    let h = cfg.hidden_size as usize;
    let nh = cfg.num_attention_heads as usize;
    let nkv = cfg.num_key_value_heads as usize;
    let d = cfg.head_dim as usize;
    let q_total = nh * d;
    let kv_total = nkv * d;
    FullAttnLayerWeights {
        attn_norm: vec![1.0f32; h],
        post_attn_norm: vec![1.0f32; h],
        wq: vec![0.0f32; q_total * h],
        wk: vec![0.0f32; kv_total * h],
        wv: vec![0.0f32; kv_total * h],
        w_gate: vec![0.0f32; q_total * h],
        attn_q_norm: vec![1.0f32; d],
        attn_k_norm: vec![1.0f32; d],
        wo: vec![0.0f32; h * q_total],
    }
}

fn empty_delta_net_weights(cfg: &Qwen35Config) -> DeltaNetLayerWeights {
    let h = cfg.hidden_size as usize;
    let nk = cfg.linear_num_key_heads as usize;
    let nv = cfg.linear_num_value_heads as usize;
    let dk = cfg.linear_key_head_dim as usize;
    let dv = cfg.linear_value_head_dim as usize;
    let k_width = cfg.linear_conv_kernel_dim as usize;
    let qkv_channels = 2 * nk * dk + nv * dv;
    let z_channels = nv * dv;
    DeltaNetLayerWeights {
        attn_norm: vec![1.0f32; h],
        post_attn_norm: vec![1.0f32; h],
        attn_qkv: vec![0.0f32; qkv_channels * h],
        attn_gate: vec![0.0f32; z_channels * h],
        ssm_conv1d: vec![0.0f32; k_width * qkv_channels],
        ssm_alpha: vec![0.0f32; nv * h],
        ssm_dt_bias: vec![0.0f32; nv],
        ssm_beta: vec![0.0f32; nv * h],
        ssm_a: vec![0.0f32; nv],
        // ssm_norm shape is [D_v], broadcast across n_v_heads per token.
        ssm_norm: vec![1.0f32; dv],
        ssm_out: vec![0.0f32; h * z_channels],
    }
}

fn empty_ffn_for(cfg: &Qwen35Config) -> Qwen35FfnWeights {
    match cfg.variant {
        Qwen35Variant::Dense => {
            let h = cfg.hidden_size as usize;
            let m = cfg
                .intermediate_size
                .expect("dense variant requires intermediate_size") as usize;
            Qwen35FfnWeights::Dense(DenseFfnWeights {
                gate: vec![0.0f32; m * h],
                up: vec![0.0f32; m * h],
                down: vec![0.0f32; h * m],
            })
        }
        Qwen35Variant::Moe => {
            let moe_cfg = cfg
                .moe
                .as_ref()
                .expect("moe variant requires moe config");
            let h = cfg.hidden_size as usize;
            let ne = moe_cfg.num_experts as usize;
            let m_moe = moe_cfg.moe_intermediate_size as usize;
            let m_sh = moe_cfg.shared_expert_intermediate_size as usize;
            Qwen35FfnWeights::Moe(MoeFfnWeights {
                router: vec![0.0f32; ne * h],
                expert_gate: vec![0.0f32; ne * m_moe * h],
                expert_up: vec![0.0f32; ne * m_moe * h],
                expert_down: vec![0.0f32; ne * h * m_moe],
                shared_gate_logit: vec![0.0f32; h],
                shared_gate: vec![0.0f32; m_sh * h],
                shared_up: vec![0.0f32; m_sh * h],
                shared_down: vec![0.0f32; h * m_sh],
            })
        }
    }
}


// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::{
        default_layer_types, Qwen35MoeConfig, Qwen35Variant,
    };

    fn moe_cfg_40() -> Qwen35Config {
        Qwen35Config {
            variant: Qwen35Variant::Moe,
            hidden_size: 64, // small for fast tests
            num_hidden_layers: 40,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            linear_num_key_heads: 4,
            linear_num_value_heads: 8,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            layer_types: default_layer_types(40, 4),
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            rotary_dim: 4,
            mrope_section: [1, 1, 0, 0],
            mrope_interleaved: true,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 1024,
            vocab_size: 256,
            attn_output_gate: true,
            mtp_num_hidden_layers: 0,
            intermediate_size: None,
            moe: Some(Qwen35MoeConfig {
                moe_intermediate_size: 16,
                num_experts: 4,
                num_experts_per_tok: 2,
                shared_expert_intermediate_size: 16,
            }),
        }
    }

    fn dense_cfg_12() -> Qwen35Config {
        let mut cfg = moe_cfg_40();
        cfg.variant = Qwen35Variant::Dense;
        cfg.num_hidden_layers = 12;
        cfg.layer_types = default_layer_types(12, 4);
        cfg.intermediate_size = Some(32);
        cfg.moe = None;
        cfg
    }

    #[test]
    fn empty_moe_40layer_has_correct_slot_counts() {
        let cfg = moe_cfg_40();
        let m = Qwen35Model::empty_from_cfg(cfg.clone());
        assert_eq!(m.layers.len(), 40);
        assert_eq!(m.num_full_attn_layers(), 10);
        assert_eq!(m.num_linear_attn_layers(), 30);
        assert!(m.mtp.is_none());
        assert_eq!(m.token_embd.len(), 256 * 64);
        assert_eq!(m.output_weight.len(), 64 * 256);
        assert_eq!(m.output_norm.len(), 64);
    }

    #[test]
    fn empty_dense_12layer_uses_swiglu_ffn() {
        let cfg = dense_cfg_12();
        let m = Qwen35Model::empty_from_cfg(cfg);
        for l in &m.layers {
            assert_eq!(l.ffn().variant(), "dense");
        }
    }

    #[test]
    fn empty_moe_12layer_uses_moe_ffn() {
        let cfg = moe_cfg_40();
        let m = Qwen35Model::empty_from_cfg(cfg);
        for l in &m.layers {
            assert_eq!(l.ffn().variant(), "moe");
        }
    }

    #[test]
    fn layer_kind_matches_config() {
        let cfg = moe_cfg_40();
        let m = Qwen35Model::empty_from_cfg(cfg.clone());
        for i in 0..40 {
            assert_eq!(
                m.layer_kind(i).unwrap(),
                cfg.layer_types[i as usize]
            );
        }
    }

    #[test]
    fn layer_kind_out_of_bounds_is_none() {
        let cfg = moe_cfg_40();
        let m = Qwen35Model::empty_from_cfg(cfg);
        assert_eq!(m.layer_kind(40), None);
        assert_eq!(m.layer_kind(9999), None);
    }

    #[test]
    fn full_attn_layer_has_q_and_kv_weights() {
        let cfg = moe_cfg_40();
        let m = Qwen35Model::empty_from_cfg(cfg.clone());
        // Layer 3 is full-attention (interval=4).
        let l3 = &m.layers[3];
        match l3 {
            Qwen35LayerWeights::FullAttn { attn, .. } => {
                let h = cfg.hidden_size as usize;
                let nh = cfg.num_attention_heads as usize;
                let d = cfg.head_dim as usize;
                assert_eq!(attn.wq.len(), nh * d * h);
                assert_eq!(attn.attn_q_norm.len(), d);
            }
            _ => panic!("expected layer 3 to be FullAttn"),
        }
    }

    #[test]
    fn linear_attn_layer_has_ssm_weights() {
        let cfg = moe_cfg_40();
        let m = Qwen35Model::empty_from_cfg(cfg.clone());
        // Layer 0 is linear-attention.
        let l0 = &m.layers[0];
        match l0 {
            Qwen35LayerWeights::LinearAttn { attn, .. } => {
                let nv = cfg.linear_num_value_heads as usize;
                assert_eq!(attn.ssm_a.len(), nv);
                assert_eq!(attn.ssm_dt_bias.len(), nv);
                let qkv_channels =
                    (2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
                        + cfg.linear_num_value_heads * cfg.linear_value_head_dim)
                        as usize;
                let k_width = cfg.linear_conv_kernel_dim as usize;
                assert_eq!(attn.ssm_conv1d.len(), k_width * qkv_channels);
            }
            _ => panic!("expected layer 0 to be LinearAttn"),
        }
    }

    #[test]
    fn ffn_variant_reported_via_config() {
        let cfg_moe = moe_cfg_40();
        let m_moe = Qwen35Model::empty_from_cfg(cfg_moe);
        assert_eq!(m_moe.ffn_variant(), Qwen35Variant::Moe);
        let cfg_dense = dense_cfg_12();
        let m_dense = Qwen35Model::empty_from_cfg(cfg_dense);
        assert_eq!(m_dense.ffn_variant(), Qwen35Variant::Dense);
    }

    /// Integration smoke: load_from_gguf on the real apex returns a fully-
    /// shaped model with 40 layers (10 full-attn + 30 linear-attn), no MTP.
    /// `#[ignore]`d so the 25 GB file isn't touched in regular test runs.
    #[test]
    #[ignore]
    fn load_from_real_apex_has_correct_shape() {
        let path = std::path::PathBuf::from(
            "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/\
             qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: apex GGUF not at expected path");
            return;
        }
        if let Err(e) = MlxDevice::new() {
            eprintln!("skipping: no Metal device: {e}");
            return;
        }
        let gguf = match GgufFile::open(&path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        let m = Qwen35Model::load_from_gguf(&gguf).expect("load");
        assert_eq!(m.cfg.variant, Qwen35Variant::Moe);
        assert_eq!(m.layers.len(), 40);
        assert_eq!(m.num_full_attn_layers(), 10);
        assert_eq!(m.num_linear_attn_layers(), 30);
        assert!(m.mtp.is_none(), "apex MTP stripped per 2026-04-23 dump");
    }
}
