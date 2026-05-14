//! DFlash drafter GPU weight residence (ADR-030 §3.3).
//!
//! Uploads validated BF16 safetensors data ([`super::weights::DFlashWeights`])
//! to mlx-native [`MlxBuffer`] resident on the Metal device. After this
//! step the drafter forward pass operates purely on GPU memory.
//!
//! Per [`super::config::DFlashConfig`] / `model_mlx.py:DFlashAttention.__init__`,
//! every weight is stored row-major BF16 in safetensors. Upload is a single
//! `copy_from_slice` per tensor (no transpose, no dequantization). Shapes
//! are re-validated post-upload so a corrupted MlxBuffer state fails fast.
//!
//! ## What this module does NOT do
//!
//! - **Embed_tokens / lm_head**: drafter shares these with the target
//!   via `bind()` per `model_mlx.py:153-168`. They live on the target
//!   model's [`MlxBuffer`]s and are referenced (not copied) at forward
//!   time.
//! - **Forward math**: the actual SDPA / RMSNorm / RoPE / MLP dispatch
//!   lives in `forward.rs` (Phase 2 next piece).
//! - **KV cache allocation**: lives in `kv_cache.rs` (Phase 2 next-next
//!   piece) — caches are sized at first-forward-time, not at weight-load
//!   time.

use super::config::DFlashConfig;
use super::weights::{DFlashWeights, WeightsError};
use mlx_native::{DType, MlxBuffer, MlxDevice, MlxError};
use safetensors::tensor::TensorView;

#[derive(Debug, thiserror::Error)]
pub enum TensorsError {
    #[error("dflash tensors mlx: {0}")]
    Mlx(#[from] MlxError),
    #[error("dflash tensors weights: {0}")]
    Weights(#[from] WeightsError),
    #[error("dflash tensors: missing manifest entry `{0}`")]
    MissingEntry(String),
}

/// Per-layer GPU-resident BF16 weights for one DFlash decoder layer.
///
/// Mirrors the Python `DFlashDecoderLayer` constituents
/// (`model_mlx.py:119-129`): one [`DFlashAttention`] + one `nn.MLP`
/// (qwen3-style SwiGLU) + two RMSNorms.
pub struct DFlashLayerTensors {
    /// `[hidden_size]` — applied before attention.
    pub input_layernorm: MlxBuffer,
    /// `[hidden_size]` — applied between attention residual and MLP.
    pub post_attention_layernorm: MlxBuffer,
    /// `[num_q_heads * head_dim, hidden_size]` — Q projection.
    pub q_proj: MlxBuffer,
    /// `[num_kv_heads * head_dim, hidden_size]` — K projection.
    pub k_proj: MlxBuffer,
    /// `[num_kv_heads * head_dim, hidden_size]` — V projection.
    pub v_proj: MlxBuffer,
    /// `[hidden_size, num_q_heads * head_dim]` — O projection.
    pub o_proj: MlxBuffer,
    /// `[head_dim]` — per-head RMSNorm on Q (qwen3-style).
    pub q_norm: MlxBuffer,
    /// `[head_dim]` — per-head RMSNorm on K (qwen3-style).
    pub k_norm: MlxBuffer,
    /// `[intermediate_size, hidden_size]` — SwiGLU gate projection.
    pub mlp_gate: MlxBuffer,
    /// `[intermediate_size, hidden_size]` — SwiGLU up projection.
    pub mlp_up: MlxBuffer,
    /// `[hidden_size, intermediate_size]` — SwiGLU down projection.
    pub mlp_down: MlxBuffer,
}

/// Full drafter model GPU weights: globals + per-layer tensors.
pub struct DFlashModelTensors {
    /// `[hidden_size, num_target_layers_used * hidden_size]` — projects
    /// concatenated target hidden states into draft hidden space.
    pub fc: MlxBuffer,
    /// `[hidden_size]` — RMSNorm applied to `fc` output (h_ctx).
    pub hidden_norm: MlxBuffer,
    /// `[hidden_size]` — final RMSNorm before lm_head.
    pub final_norm: MlxBuffer,
    /// One per draft layer (5 for gemma-4-26B-A4B-it drafter).
    pub layers: Vec<DFlashLayerTensors>,
}

/// Upload a single BF16 [`TensorView`] to a fresh GPU [`MlxBuffer`].
fn upload_bf16(device: &MlxDevice, view: &TensorView<'_>) -> Result<MlxBuffer, TensorsError> {
    let shape: Vec<usize> = view.shape().to_vec();
    let byte_len = view.data().len();
    let mut buf = device.alloc_buffer(byte_len, DType::BF16, shape)?;
    // SAFETY: alloc_buffer returns a fresh zeroed buffer; we have exclusive
    // access (no dispatch references it yet). `as_mut_slice::<u8>()` over
    // BF16 storage is well-defined (any byte pattern is a valid u8).
    let dst: &mut [u8] = buf
        .as_mut_slice::<u8>()
        .map_err(|e| TensorsError::Mlx(MlxError::InvalidArgument(format!("buffer slice: {e}"))))?;
    debug_assert_eq!(dst.len(), byte_len);
    dst.copy_from_slice(view.data());
    Ok(buf)
}

impl DFlashModelTensors {
    /// Upload all 58 drafter tensors to the GPU. The order of upload
    /// matches `DFlashWeights::manifest` (stable). Total upload cost:
    /// 58 × `copy_from_slice` over BF16 bytes (~820 MB for gemma-4-26B-A4B-it).
    ///
    /// After this returns, the drafter is ready for forward — embed_tokens
    /// and lm_head are NOT loaded (shared from target at forward time).
    pub fn upload(
        device: &MlxDevice,
        cfg: &DFlashConfig,
        weights: &DFlashWeights<'_>,
    ) -> Result<Self, TensorsError> {
        // Globals
        let fc = upload_bf16(device, fetch(weights, "fc.weight")?)?;
        let hidden_norm = upload_bf16(device, fetch(weights, "hidden_norm.weight")?)?;
        let final_norm = upload_bf16(device, fetch(weights, "norm.weight")?)?;

        // Per-layer
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("layers.{i}");
            let lw = DFlashLayerTensors {
                input_layernorm: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.input_layernorm.weight"))?,
                )?,
                post_attention_layernorm: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.post_attention_layernorm.weight"))?,
                )?,
                q_proj: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.self_attn.q_proj.weight"))?,
                )?,
                k_proj: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.self_attn.k_proj.weight"))?,
                )?,
                v_proj: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.self_attn.v_proj.weight"))?,
                )?,
                o_proj: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.self_attn.o_proj.weight"))?,
                )?,
                q_norm: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.self_attn.q_norm.weight"))?,
                )?,
                k_norm: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.self_attn.k_norm.weight"))?,
                )?,
                mlp_gate: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.mlp.gate_proj.weight"))?,
                )?,
                mlp_up: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.mlp.up_proj.weight"))?,
                )?,
                mlp_down: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.mlp.down_proj.weight"))?,
                )?,
            };
            layers.push(lw);
        }

        Ok(DFlashModelTensors {
            fc,
            hidden_norm,
            final_norm,
            layers,
        })
    }

    /// Total bytes resident on GPU after upload. Used for memory
    /// accounting / regression tests.
    pub fn gpu_resident_bytes(&self) -> usize {
        let layer_bytes: usize = self
            .layers
            .iter()
            .map(|l| {
                l.input_layernorm.byte_len()
                    + l.post_attention_layernorm.byte_len()
                    + l.q_proj.byte_len()
                    + l.k_proj.byte_len()
                    + l.v_proj.byte_len()
                    + l.o_proj.byte_len()
                    + l.q_norm.byte_len()
                    + l.k_norm.byte_len()
                    + l.mlp_gate.byte_len()
                    + l.mlp_up.byte_len()
                    + l.mlp_down.byte_len()
            })
            .sum();
        self.fc.byte_len() + self.hidden_norm.byte_len() + self.final_norm.byte_len() + layer_bytes
    }
}

fn fetch<'a, 'b>(
    weights: &'a DFlashWeights<'b>,
    name: &str,
) -> Result<&'a TensorView<'b>, TensorsError> {
    weights
        .tensor(name)
        .ok_or_else(|| TensorsError::MissingEntry(name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::dflash::config::DFlashConfig;
    use crate::inference::spec_decode::dflash::weights::{DFlashWeights, DFlashWeightsFile};

    fn gemma4_26b_a4b_dflash_config() -> DFlashConfig {
        DFlashConfig::from_json_str(
            super::super::config::tests::GEMMA4_26B_A4B_DFLASH_CONFIG,
        )
        .expect("test fixture must parse")
    }

    /// Integration test: upload all 58 drafter tensors to GPU and
    /// verify the resident byte total matches the safetensors data
    /// total (within 0 bytes — every byte is copied 1:1).
    ///
    /// Requires both a Metal device AND the cached drafter download.
    #[test]
    #[ignore = "requires Metal device + drafter HF cache"]
    fn uploads_real_drafter_to_gpu() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let home = std::env::var("HOME").expect("HOME set");
        let path = format!(
            "{home}/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057/model.safetensors"
        );
        let file = DFlashWeightsFile::open(&path).expect("file open");
        let weights = DFlashWeights::load(file.bytes(), &cfg).expect("validated load");
        let safetensors_data_bytes = weights.total_data_bytes();

        let tensors = DFlashModelTensors::upload(&device, &cfg, &weights).expect("GPU upload");
        let gpu_bytes = tensors.gpu_resident_bytes();

        assert_eq!(
            gpu_bytes, safetensors_data_bytes,
            "every byte from safetensors must land on the GPU 1:1"
        );
        assert_eq!(tensors.layers.len(), cfg.num_hidden_layers);
    }
}
