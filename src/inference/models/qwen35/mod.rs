//! Qwen3.5 / Qwen3.5-MoE inference support.
//!
//! Entry point for **both** variants:
//!
//! * **Dense** (`general.architecture = "qwen35"`) — 27B dense Qwen3.5.
//! * **MoE** (`general.architecture = "qwen35moe"`) — 35B-A3B mixture of experts.
//!
//! Both variants share ≥90% of the forward surface (linear-attention,
//! gated full-attention, MROPE, MTP, tokenizer, hybrid KV cache, every
//! mlx-native kernel). Only the FFN block differs.
//!
//! Owned by **ADR-013**. Companion conversion spec is ADR-012.
//!
//! # Module layout
//!
//! * `mod.rs` (this file) — shared: [`Qwen35Config`], [`Qwen35LayerKind`],
//!   [`Qwen35Variant`], metadata parser, `ARCH_QWEN35*` constants.
//! * `dense.rs` — 27B dense-specific: SwiGLU FFN, tensor-name resolution,
//!   dense forward entry point.
//! * `moe.rs` — MoE-specific: 256-expert dispatch, shared-expert-gate,
//!   MoE tensor-name resolution, MoE forward entry point.
//! * `kernels.rs` — thin wrappers around mlx-native's new ops.
//! * `kv_cache.rs` — hybrid cache (full-attention KV + linear-attention SSM state).

use anyhow::{anyhow, bail, Result};
use mlx_native::gguf::{GgufFile, MetadataValue};

pub mod activation_capture;
pub mod delta_net;
pub mod dense;
pub mod ffn;
pub mod forward_cpu;
pub mod full_attn;
pub mod gpu_full_attn;
pub mod io_heads;
pub mod kernels;
pub mod kv_cache;
pub mod model;
pub mod moe;
pub mod mtp;

/// `general.architecture` value for the dense variant.
pub const ARCH_QWEN35: &str = "qwen35";
/// `general.architecture` value for the MoE variant.
pub const ARCH_QWEN35MOE: &str = "qwen35moe";

/// Dense vs MoE flavor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35Variant {
    Dense,
    Moe,
}

impl Qwen35Variant {
    /// Resolve from a `general.architecture` metadata string.
    pub fn from_arch(arch: &str) -> Option<Self> {
        match arch {
            ARCH_QWEN35 => Some(Qwen35Variant::Dense),
            ARCH_QWEN35MOE => Some(Qwen35Variant::Moe),
            _ => None,
        }
    }

    /// The metadata-key prefix this variant emits (e.g. `qwen35moe.`).
    pub fn key_prefix(&self) -> &'static str {
        match self {
            Qwen35Variant::Dense => ARCH_QWEN35,
            Qwen35Variant::Moe => ARCH_QWEN35MOE,
        }
    }
}

/// Per-layer kind for the Qwen3.5 interleaved hybrid stack.
///
/// **Distinct from Gemma-4's `LayerType::{Sliding, Full}`** (different
/// semantic axis). Per `project_model_class_split.md` and ADR-013
/// Decision 2, Qwen3.5 owns its own enum in its own module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35LayerKind {
    /// Gated DeltaNet linear-attention block.
    LinearAttention,
    /// Gated full-attention block (standard SDPA with output gate).
    FullAttention,
}

/// MoE-only config fields. `None` on the dense variant.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35MoeConfig {
    /// Per-expert FFN hidden size (512 for Qwen3.5-MoE).
    pub moe_intermediate_size: u32,
    /// Total number of experts (256 for Qwen3.5-MoE).
    pub num_experts: u32,
    /// Experts activated per token (8 for Qwen3.5-MoE).
    pub num_experts_per_tok: u32,
    /// Shared-expert FFN hidden size (512 for Qwen3.5-MoE). Shared experts
    /// are gated by a separate sigmoid — see ADR-013 Decision 13.
    pub shared_expert_intermediate_size: u32,
}

/// Full architecture config for a Qwen3.5 or Qwen3.5-MoE model.
///
/// Source of truth is the GGUF metadata. See [`Qwen35Config::from_gguf`] for
/// the exact key-to-field mapping — grounded in the apex GGUF dump on
/// 2026-04-23.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35Config {
    pub variant: Qwen35Variant,

    // --- Global dims / counts ---
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    /// Full-attention Q-head count.
    pub num_attention_heads: u32,
    /// Full-attention KV-head count (GQA).
    pub num_key_value_heads: u32,
    /// Full-attention per-head dim (= attention.key_length = attention.value_length; 256 for Qwen3.5).
    pub head_dim: u32,

    // --- Linear-attention (Gated DeltaNet) dims ---
    /// Number of K heads in the linear-attention branch.
    pub linear_num_key_heads: u32,
    /// Number of V heads in the linear-attention branch (>= linear_num_key_heads; GQA).
    pub linear_num_value_heads: u32,
    pub linear_key_head_dim: u32,
    pub linear_value_head_dim: u32,
    /// 1D conv kernel width for the SSM conv1d (4 for Qwen3.5).
    pub linear_conv_kernel_dim: u32,

    // --- Layer stack layout ---
    /// Full-attention layer period (4 for Qwen3.5: layers 3, 7, 11, ... are full attention).
    pub full_attention_interval: u32,
    /// Per-layer kind. Authoritative. Computed from `full_attention_interval`
    /// when `layer_types` is not emitted as an explicit GGUF array (current
    /// llama.cpp/apex convention; kept as a Vec so future metadata can
    /// override on a per-layer basis).
    pub layer_types: Vec<Qwen35LayerKind>,

    // --- RoPE / MROPE ---
    pub partial_rotary_factor: f32, // 0.25 for Qwen3.5 (rotary_dim / head_dim)
    pub rope_theta: f64,            // 1e7 for Qwen3.5
    pub rotary_dim: u32,            // 64 for Qwen3.5 (= partial_rotary_factor * head_dim)
    pub mrope_section: [u32; 4],    // [11, 11, 10, 0] for Qwen3.5
    /// IMROPE interleaved mode. Always `true` for Qwen3.5 (per the model family's
    /// runtime convention — `GGML_ROPE_TYPE_IMROPE == 40`).
    pub mrope_interleaved: bool,

    // --- Norm ---
    pub rms_norm_eps: f32, // 1e-6 for Qwen3.5

    // --- Misc runtime flags ---
    pub max_position_embeddings: u32,
    pub vocab_size: u32,
    pub attn_output_gate: bool, // true for Qwen3.5
    pub mtp_num_hidden_layers: u32, // 0 if MTP absent (apex GGUF case)

    // --- FFN variant-specific ---
    pub intermediate_size: Option<u32>, // dense: Some(17408); moe: None
    pub moe: Option<Qwen35MoeConfig>,   // dense: None; moe: Some(...)
}

// ---------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------

fn required_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    gguf.metadata_u32(key)
        .ok_or_else(|| anyhow!("qwen35 config: required key '{}' missing or wrong type", key))
}

fn required_f32(gguf: &GgufFile, key: &str) -> Result<f32> {
    gguf.metadata_f32(key)
        .ok_or_else(|| anyhow!("qwen35 config: required key '{}' missing or wrong type", key))
}

fn required_i32_array_4(gguf: &GgufFile, key: &str) -> Result<[u32; 4]> {
    let mv = gguf
        .metadata(key)
        .ok_or_else(|| anyhow!("qwen35 config: required key '{}' missing", key))?;
    let arr = match mv {
        MetadataValue::Array(a) => a,
        _ => bail!(
            "qwen35 config: key '{}' has type {:?}, expected Array",
            key,
            std::mem::discriminant(mv)
        ),
    };
    if arr.len() != 4 {
        bail!(
            "qwen35 config: key '{}' length {} != 4 (mrope sections)",
            key,
            arr.len()
        );
    }
    let mut out = [0u32; 4];
    for (i, v) in arr.iter().enumerate() {
        out[i] = match v {
            MetadataValue::Int32(x) if *x >= 0 => *x as u32,
            MetadataValue::Uint32(x) => *x,
            MetadataValue::Int8(x) if *x >= 0 => *x as u32,
            MetadataValue::Int16(x) if *x >= 0 => *x as u32,
            MetadataValue::Uint8(x) => *x as u32,
            MetadataValue::Uint16(x) => *x as u32,
            other => bail!(
                "qwen35 config: key '{}' element {} has unexpected variant {:?}",
                key,
                i,
                std::mem::discriminant(other)
            ),
        };
    }
    Ok(out)
}

/// Compute `layer_types` from `num_hidden_layers` and `full_attention_interval`.
///
/// Convention (verified against apex GGUF tensor dump on 2026-04-23 — layers
/// 0, 1, 2 are linear-attention and layer 3 is the first full-attention):
///
/// ```text
/// layer_types[i] = FullAttention   if (i + 1) % interval == 0
///                = LinearAttention otherwise
/// ```
///
/// For the default `full_attention_interval = 4`: layers {3, 7, 11, ...} are
/// FullAttention, all others are LinearAttention.
pub fn default_layer_types(num_hidden_layers: u32, interval: u32) -> Vec<Qwen35LayerKind> {
    if interval == 0 {
        return vec![Qwen35LayerKind::LinearAttention; num_hidden_layers as usize];
    }
    (0..num_hidden_layers)
        .map(|i| {
            if (i + 1) % interval == 0 {
                Qwen35LayerKind::FullAttention
            } else {
                Qwen35LayerKind::LinearAttention
            }
        })
        .collect()
}

impl Qwen35Config {
    /// Parse a Qwen3.5 / Qwen3.5-MoE configuration from loaded GGUF
    /// metadata.
    ///
    /// Required keys (missing → error):
    ///
    /// ```text
    /// {prefix}.block_count                            u32
    /// {prefix}.embedding_length                       u32
    /// {prefix}.attention.head_count                   u32
    /// {prefix}.attention.head_count_kv                u32
    /// {prefix}.attention.key_length                   u32 (= head_dim)
    /// {prefix}.attention.value_length                 u32 (= head_dim)
    /// {prefix}.attention.layer_norm_rms_epsilon       f32
    /// {prefix}.context_length                         u32
    /// {prefix}.rope.freq_base                         f32
    /// {prefix}.rope.dimension_count                   u32 (rotary_dim)
    /// {prefix}.rope.dimension_sections                i32[4]
    /// {prefix}.full_attention_interval                u32
    /// {prefix}.ssm.state_size                         u32 (linear head_dim)
    /// {prefix}.ssm.group_count                        u32 (linear_num_key_heads)
    /// {prefix}.ssm.inner_size                         u32 (= linear_num_value_heads * state_size)
    /// {prefix}.ssm.conv_kernel                        u32 (linear_conv_kernel_dim)
    /// ```
    ///
    /// MoE variant additionally requires:
    ///
    /// ```text
    /// qwen35moe.expert_count                          u32
    /// qwen35moe.expert_used_count                     u32
    /// qwen35moe.expert_feed_forward_length            u32
    /// qwen35moe.expert_shared_feed_forward_length     u32
    /// ```
    ///
    /// Dense variant requires instead:
    ///
    /// ```text
    /// qwen35.feed_forward_length                      u32  (intermediate_size)
    /// ```
    ///
    /// Optional keys (fallback to Qwen3.5 documented defaults if absent):
    ///
    /// ```text
    /// {prefix}.mtp.num_hidden_layers                  u32   default 0
    /// {prefix}.attention.output_gate                  bool  default true
    /// ```
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| anyhow!("GGUF missing required key 'general.architecture'"))?;

        let variant = Qwen35Variant::from_arch(arch).ok_or_else(|| {
            anyhow!(
                "general.architecture = {:?} is not a Qwen3.5 variant (expected {:?} or {:?})",
                arch,
                ARCH_QWEN35,
                ARCH_QWEN35MOE
            )
        })?;

        let p = variant.key_prefix();

        let num_hidden_layers = required_u32(gguf, &format!("{p}.block_count"))?;
        let hidden_size = required_u32(gguf, &format!("{p}.embedding_length"))?;
        let num_attention_heads = required_u32(gguf, &format!("{p}.attention.head_count"))?;
        let num_key_value_heads = required_u32(gguf, &format!("{p}.attention.head_count_kv"))?;

        // Qwen3.5 has key_length == value_length == head_dim.
        let key_length = required_u32(gguf, &format!("{p}.attention.key_length"))?;
        let value_length = required_u32(gguf, &format!("{p}.attention.value_length"))?;
        if key_length != value_length {
            bail!(
                "qwen35 config: attention.key_length ({}) != attention.value_length ({}); \
                 Qwen3.5 requires them equal",
                key_length,
                value_length
            );
        }
        let head_dim = key_length;

        let rms_norm_eps = required_f32(gguf, &format!("{p}.attention.layer_norm_rms_epsilon"))?;
        let max_position_embeddings = required_u32(gguf, &format!("{p}.context_length"))?;
        let rope_theta = required_f32(gguf, &format!("{p}.rope.freq_base"))? as f64;
        let rotary_dim = required_u32(gguf, &format!("{p}.rope.dimension_count"))?;
        let mrope_section = required_i32_array_4(gguf, &format!("{p}.rope.dimension_sections"))?;
        let full_attention_interval = required_u32(gguf, &format!("{p}.full_attention_interval"))?;

        // Linear attention (SSM keys).
        let ssm_state_size = required_u32(gguf, &format!("{p}.ssm.state_size"))?;
        let ssm_group_count = required_u32(gguf, &format!("{p}.ssm.group_count"))?;
        let ssm_inner_size = required_u32(gguf, &format!("{p}.ssm.inner_size"))?;
        let ssm_conv_kernel = required_u32(gguf, &format!("{p}.ssm.conv_kernel"))?;
        if ssm_state_size == 0 {
            bail!("qwen35 config: {p}.ssm.state_size must be > 0");
        }
        if ssm_inner_size % ssm_state_size != 0 {
            bail!(
                "qwen35 config: {p}.ssm.inner_size ({}) must be a multiple of \
                 {p}.ssm.state_size ({})",
                ssm_inner_size,
                ssm_state_size
            );
        }
        let linear_num_value_heads = ssm_inner_size / ssm_state_size;

        // Optional keys.
        let mtp_num_hidden_layers = gguf
            .metadata_u32(&format!("{p}.mtp.num_hidden_layers"))
            .unwrap_or(0);
        let attn_output_gate = gguf
            .metadata(&format!("{p}.attention.output_gate"))
            .and_then(|v| match v {
                MetadataValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(true);

        // Partial rotary factor is derived from rotary_dim / head_dim.
        if head_dim == 0 {
            bail!("qwen35 config: head_dim is 0");
        }
        let partial_rotary_factor = (rotary_dim as f32) / (head_dim as f32);

        // Vocab size: prefer explicit metadata, else use tokenizer tokens array length.
        let vocab_size = gguf
            .metadata_u32(&format!("{p}.vocab_size"))
            .or_else(|| {
                gguf.metadata("tokenizer.ggml.tokens")
                    .and_then(|v| match v {
                        MetadataValue::Array(a) => Some(a.len() as u32),
                        _ => None,
                    })
            })
            .ok_or_else(|| {
                anyhow!(
                    "qwen35 config: can't determine vocab_size \
                     (neither {p}.vocab_size nor tokenizer.ggml.tokens present)"
                )
            })?;

        // FFN variant-specific.
        let (intermediate_size, moe) = match variant {
            Qwen35Variant::Dense => {
                let fl = required_u32(gguf, &format!("{p}.feed_forward_length"))?;
                (Some(fl), None)
            }
            Qwen35Variant::Moe => {
                let num_experts = required_u32(gguf, &format!("{p}.expert_count"))?;
                let num_experts_per_tok =
                    required_u32(gguf, &format!("{p}.expert_used_count"))?;
                let moe_intermediate_size =
                    required_u32(gguf, &format!("{p}.expert_feed_forward_length"))?;
                let shared_expert_intermediate_size =
                    required_u32(gguf, &format!("{p}.expert_shared_feed_forward_length"))?;
                (
                    None,
                    Some(Qwen35MoeConfig {
                        moe_intermediate_size,
                        num_experts,
                        num_experts_per_tok,
                        shared_expert_intermediate_size,
                    }),
                )
            }
        };

        let layer_types = default_layer_types(num_hidden_layers, full_attention_interval);

        Ok(Qwen35Config {
            variant,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            linear_num_key_heads: ssm_group_count,
            linear_num_value_heads,
            linear_key_head_dim: ssm_state_size,
            linear_value_head_dim: ssm_state_size,
            linear_conv_kernel_dim: ssm_conv_kernel,
            full_attention_interval,
            layer_types,
            partial_rotary_factor,
            rope_theta,
            rotary_dim,
            mrope_section,
            mrope_interleaved: true, // Qwen3.5 convention (see ADR-013 Decision 10).
            rms_norm_eps,
            max_position_embeddings,
            vocab_size,
            attn_output_gate,
            mtp_num_hidden_layers,
            intermediate_size,
            moe,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_from_arch() {
        assert_eq!(Qwen35Variant::from_arch("qwen35"), Some(Qwen35Variant::Dense));
        assert_eq!(
            Qwen35Variant::from_arch("qwen35moe"),
            Some(Qwen35Variant::Moe)
        );
        assert_eq!(Qwen35Variant::from_arch("gemma4"), None);
        assert_eq!(Qwen35Variant::from_arch("qwen3"), None);
        assert_eq!(Qwen35Variant::from_arch(""), None);
    }

    #[test]
    fn layer_types_interval_4() {
        // 40 layers, every 4th is full-attention (layers 3, 7, ..., 39).
        let lt = default_layer_types(40, 4);
        assert_eq!(lt.len(), 40);
        for (i, kind) in lt.iter().enumerate() {
            let want = if (i + 1) % 4 == 0 {
                Qwen35LayerKind::FullAttention
            } else {
                Qwen35LayerKind::LinearAttention
            };
            assert_eq!(*kind, want, "layer {} kind mismatch", i);
        }
        // Expected pattern for first 8 layers: [L, L, L, F, L, L, L, F].
        use Qwen35LayerKind::*;
        assert_eq!(
            &lt[..8],
            &[
                LinearAttention, LinearAttention, LinearAttention, FullAttention,
                LinearAttention, LinearAttention, LinearAttention, FullAttention,
            ]
        );
    }

    #[test]
    fn layer_types_dense_27b_64layer() {
        let lt = default_layer_types(64, 4);
        let full_count = lt
            .iter()
            .filter(|k| **k == Qwen35LayerKind::FullAttention)
            .count();
        assert_eq!(full_count, 16); // 64 / 4 = 16 full-attention layers.
    }

    #[test]
    fn layer_types_interval_zero_all_linear() {
        let lt = default_layer_types(8, 0);
        assert!(lt.iter().all(|k| *k == Qwen35LayerKind::LinearAttention));
    }

    #[test]
    fn key_prefix_roundtrip() {
        assert_eq!(Qwen35Variant::Dense.key_prefix(), ARCH_QWEN35);
        assert_eq!(Qwen35Variant::Moe.key_prefix(), ARCH_QWEN35MOE);
    }

    /// Integration test against the real apex GGUF on disk. `#[ignore]` so
    /// `cargo test` stays fast; run explicitly with
    /// `cargo test -p hf2q -- --ignored parses_real_apex_gguf`.
    ///
    /// Verified values (dumped via `llama-gguf` + python parser on 2026-04-23):
    /// - num_hidden_layers = 40
    /// - hidden_size       = 2048
    /// - head_dim          = 256
    /// - rotary_dim        = 64
    /// - mrope_section     = [11, 11, 10, 0]
    /// - rope_theta        ≈ 1e7
    /// - num_experts       = 256
    /// - num_experts_per_tok = 8
    #[test]
    #[ignore]
    fn parses_real_apex_gguf() {
        let path = std::path::PathBuf::from(
            "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/\
             qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: apex GGUF not at expected path");
            return;
        }
        let gguf = match GgufFile::open(&path) {
            Ok(g) => g,
            Err(e) => {
                // Apex GGUF contains Q5_K tensors; mlx-native's GGUF loader
                // only supports F32/F16/Q4_0/Q8_0/Q4_K/Q6_K as of 2026-04-23.
                // Q5_K + I16 support lands in P5 (ADR-013 Decision 12).
                // Until then, this test cannot open the full apex file; skip.
                eprintln!("skipping: apex GGUF open failed ({e}) — waiting on P5 Q5_K/I16 loader support");
                return;
            }
        };
        let cfg = Qwen35Config::from_gguf(&gguf).expect("parse qwen35 config");

        assert_eq!(cfg.variant, Qwen35Variant::Moe);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_num_value_heads, 32); // inner_size / state_size = 4096 / 128
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert_eq!(cfg.linear_value_head_dim, 128);
        assert_eq!(cfg.linear_conv_kernel_dim, 4);
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(cfg.rotary_dim, 64);
        assert_eq!(cfg.mrope_section, [11, 11, 10, 0]);
        assert!(cfg.mrope_interleaved);
        assert!((cfg.partial_rotary_factor - 0.25).abs() < 1e-6);
        assert!((cfg.rope_theta - 1e7).abs() < 1.0);
        assert!(cfg.rms_norm_eps > 0.0 && cfg.rms_norm_eps < 1e-5);
        assert_eq!(cfg.layer_types.len(), 40);
        assert_eq!(cfg.layer_types[3], Qwen35LayerKind::FullAttention);
        assert_eq!(cfg.layer_types[0], Qwen35LayerKind::LinearAttention);

        let moe = cfg.moe.as_ref().expect("moe fields");
        assert_eq!(moe.num_experts, 256);
        assert_eq!(moe.num_experts_per_tok, 8);
        assert_eq!(moe.moe_intermediate_size, 512);
        assert_eq!(moe.shared_expert_intermediate_size, 512);
        assert!(cfg.intermediate_size.is_none());

        // MTP stripped from apex; must be 0.
        assert_eq!(cfg.mtp_num_hidden_layers, 0);
    }

    /// End-to-end Q5_K dequant against the real apex GGUF. Picks
    /// `blk.0.attn_gate.weight` (a Q5_K tensor of shape [2048, 4096] =
    /// 8,388,608 values = 32,768 Q5_K blocks) and verifies:
    /// - load_tensor_f32 returns the expected count.
    /// - All values are finite (no NaN / Inf from broken super-block arithmetic).
    /// - The value distribution is non-degenerate (std > 0).
    ///
    /// `#[ignore]`d because it opens the 25 GB file and dequantizes ~32K
    /// super-blocks; runs in ~100ms but we don't want every `cargo test`
    /// invocation touching disk. Run via `--ignored`.
    #[test]
    #[ignore]
    fn dequantizes_real_apex_q5k_tensor() {
        let path = std::path::PathBuf::from(
            "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/\
             qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: apex GGUF not at expected path");
            return;
        }
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let gguf = GgufFile::open(&path).expect("open apex gguf");
        let buf = gguf
            .load_tensor_f32("blk.0.attn_gate.weight", &device)
            .expect("load Q5_K tensor");

        let got: &[f32] = buf.as_slice().expect("slice");
        assert_eq!(got.len(), 2048 * 4096, "element count");

        // All finite.
        let mut n_nan = 0usize;
        let mut n_inf = 0usize;
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for v in got {
            if v.is_nan() {
                n_nan += 1;
            } else if !v.is_finite() {
                n_inf += 1;
            } else {
                sum += *v as f64;
                sum_sq += (*v as f64) * (*v as f64);
            }
        }
        assert_eq!(n_nan, 0, "Q5_K dequant produced NaN values");
        assert_eq!(n_inf, 0, "Q5_K dequant produced Inf values");

        let n = got.len() as f64;
        let mean = sum / n;
        let variance = (sum_sq / n) - mean * mean;
        let stddev = variance.max(0.0).sqrt();
        assert!(
            stddev > 1e-6,
            "Q5_K dequant produced degenerate (all-equal) tensor; stddev = {}",
            stddev
        );
        // Typical attention-gate weights have small magnitudes; sanity bound.
        assert!(stddev < 10.0, "Q5_K dequant stddev absurdly large: {}", stddev);

        eprintln!(
            "blk.0.attn_gate.weight (Q5_K → f32): count={}, mean={:.6}, stddev={:.6}",
            got.len(),
            mean,
            stddev
        );
    }
}
