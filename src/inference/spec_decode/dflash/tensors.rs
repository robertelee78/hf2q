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

/// Per-layer GPU-resident weights for one DFlash decoder layer.
///
/// Mirrors the Python `DFlashDecoderLayer` constituents
/// (`model_mlx.py:119-129`): one [`DFlashAttention`] + one `nn.MLP`
/// (qwen3-style SwiGLU) + two RMSNorms.
///
/// ## DType policy
///
/// Most weights stay BF16 (the safetensors-on-disk dtype) for memory
/// efficiency — they are consumed by `dispatch_dense_mm_bf16`-class
/// matmul kernels that operate on BF16 weights + F32 activations.
///
/// **All RMSNorm weights are stored as F32** (ADR-030 iter-106): `q_norm`,
/// `k_norm`, `input_layernorm`, `post_attention_layernorm`, model-level
/// `hidden_norm`, and final `norm`. mlx-native's `dispatch_rms_norm`
/// kernel selects pipeline by INPUT dtype only; for F32 input it
/// dispatches `rms_norm_f32` which declares `device const float*
/// weight [[buffer(1)]]` (`shaders/rms_norm.metal:20`).  Passing a
/// BF16-backed weight buffer to that kernel causes the kernel to
/// read F32 words over the BF16 byte layout — values become bit-
/// misinterpreted combinations of adjacent BF16 elements AND reads
/// past the buffer end (BF16 buffer is dim*2 bytes; F32 kernel reads
/// dim*4 bytes).  Matching qwen35's convention (`forward_gpu.rs:332`
/// uploaded via `upload_f32_weight`).  Memory cost is small: all
/// drafter norms total 5 × (2 head + 2 layer) + 2 model = 14 RMSNorm
/// vectors, each at most `hidden_size = 2816` floats → ~158 KB.
pub struct DFlashLayerTensors {
    /// `[hidden_size]` **F32** — applied before attention.
    /// Cast from BF16 at upload time (ADR-030 iter-106).
    pub input_layernorm: MlxBuffer,
    /// `[hidden_size]` **F32** — applied between attention residual and MLP.
    /// Cast from BF16 at upload time (ADR-030 iter-106).
    pub post_attention_layernorm: MlxBuffer,
    /// `[num_q_heads * head_dim, hidden_size]` BF16 — Q projection.
    pub q_proj: MlxBuffer,
    /// `[num_kv_heads * head_dim, hidden_size]` BF16 — K projection.
    pub k_proj: MlxBuffer,
    /// `[num_kv_heads * head_dim, hidden_size]` BF16 — V projection.
    pub v_proj: MlxBuffer,
    /// `[hidden_size, num_q_heads * head_dim]` BF16 — O projection.
    pub o_proj: MlxBuffer,
    /// `[head_dim]` **F32** — per-head RMSNorm on Q (qwen3-style).
    /// Cast from BF16 at upload time for mlx-native rms_norm-f32 path.
    pub q_norm: MlxBuffer,
    /// `[head_dim]` **F32** — per-head RMSNorm on K (qwen3-style).
    /// Cast from BF16 at upload time.
    pub k_norm: MlxBuffer,
    /// `[intermediate_size, hidden_size]` BF16 — SwiGLU gate projection.
    pub mlp_gate: MlxBuffer,
    /// `[intermediate_size, hidden_size]` BF16 — SwiGLU up projection.
    pub mlp_up: MlxBuffer,
    /// `[hidden_size, intermediate_size]` BF16 — SwiGLU down projection.
    pub mlp_down: MlxBuffer,
}

/// Full drafter model GPU weights: globals + per-layer tensors.
pub struct DFlashModelTensors {
    /// `[hidden_size, num_target_layers_used * hidden_size]` — projects
    /// concatenated target hidden states into draft hidden space.
    pub fc: MlxBuffer,
    /// `[hidden_size]` **F32** — RMSNorm applied to `fc` output (h_ctx).
    /// Cast from BF16 at upload time (ADR-030 iter-106).
    pub hidden_norm: MlxBuffer,
    /// `[hidden_size]` **F32** — final RMSNorm before lm_head.
    /// Cast from BF16 at upload time (ADR-030 iter-106).
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

/// Convert a BF16 [`TensorView`] to F32 and upload to a fresh GPU
/// [`MlxBuffer`]. Used for q_norm/k_norm where mlx-native's rms_norm
/// kernel requires weight dtype to match the F32 input.
fn upload_bf16_as_f32(device: &MlxDevice, view: &TensorView<'_>) -> Result<MlxBuffer, TensorsError> {
    let bf16_bytes = view.data();
    let n_elem = bf16_bytes.len() / 2;
    if bf16_bytes.len() != n_elem * 2 {
        return Err(TensorsError::Mlx(MlxError::InvalidArgument(format!(
            "upload_bf16_as_f32: data len {} not even (not BF16-aligned)",
            bf16_bytes.len()
        ))));
    }
    let shape: Vec<usize> = view.shape().to_vec();
    let mut buf = device.alloc_buffer(n_elem * 4, DType::F32, shape)?;
    let dst: &mut [f32] = buf
        .as_mut_slice::<f32>()
        .map_err(|e| TensorsError::Mlx(MlxError::InvalidArgument(format!("f32 slice: {e}"))))?;
    debug_assert_eq!(dst.len(), n_elem);
    // BF16 = 16 MSBs of an F32. Reconstruct F32 by left-shifting BF16
    // bits into the top half of the F32 bits.
    for i in 0..n_elem {
        let lo = bf16_bytes[i * 2] as u32;
        let hi = bf16_bytes[i * 2 + 1] as u32;
        let bf16_bits = lo | (hi << 8);
        let f32_bits = bf16_bits << 16;
        dst[i] = f32::from_bits(f32_bits);
    }
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
        // ADR-030 iter-106 — hidden_norm + final_norm uploaded as F32.
        // dispatch_rms_norm selects pipeline by INPUT dtype (F32 here);
        // rms_norm_f32 declares weight as `device const float*` so a BF16
        // buffer would be bit-misinterpreted + out-of-bounds-read.
        let hidden_norm = upload_bf16_as_f32(device, fetch(weights, "hidden_norm.weight")?)?;
        let final_norm = upload_bf16_as_f32(device, fetch(weights, "norm.weight")?)?;

        // Per-layer
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("layers.{i}");
            let lw = DFlashLayerTensors {
                // ADR-030 iter-106 — see DFlashLayerTensors docstring.
                // F32 cast at upload mirrors qwen35 forward_gpu.rs:332.
                input_layernorm: upload_bf16_as_f32(
                    device,
                    fetch(weights, &format!("{p}.input_layernorm.weight"))?,
                )?,
                post_attention_layernorm: upload_bf16_as_f32(
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
                q_norm: upload_bf16_as_f32(
                    device,
                    fetch(weights, &format!("{p}.self_attn.q_norm.weight"))?,
                )?,
                k_norm: upload_bf16_as_f32(
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
    /// total plus the F32 expansion for the cast RMSNorm tensors
    /// (ADR-030 iter-106).
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

        // ADR-030 iter-106: RMSNorm weights are BF16 on disk but uploaded
        // as F32 (2× expansion).  Per-layer F32-cast norms: q_norm +
        // k_norm (head_dim each) + input_layernorm + post_attention_layernorm
        // (hidden_size each).  Model-level: hidden_norm + final_norm
        // (hidden_size each).  Each cast adds `n_elem * 2` bytes vs
        // straight BF16 copy.
        let per_layer_cast_elems = 2 * cfg.head_dim + 2 * cfg.hidden_size;
        let model_cast_elems = 2 * cfg.hidden_size;
        let cast_expansion_bytes =
            (cfg.num_hidden_layers * per_layer_cast_elems + model_cast_elems) * 2;
        assert_eq!(
            gpu_bytes,
            safetensors_data_bytes + cast_expansion_bytes,
            "GPU resident bytes must equal safetensors data + F32 cast expansion \
             (cfg.num_hidden_layers={}, head_dim={}, hidden_size={}, expansion={})",
            cfg.num_hidden_layers, cfg.head_dim, cfg.hidden_size, cast_expansion_bytes,
        );
        assert_eq!(tensors.layers.len(), cfg.num_hidden_layers);
    }
}
