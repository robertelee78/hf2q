//! GPU-resident DFlash2 weights shared by every target-family adapter.

use super::config::DFlash2Config;
use super::weights::{DFlash2Weights, OPTIONAL_MASK_EMBEDDING};
use mlx_native::{DType, MlxBuffer, MlxDevice, MlxError};
use safetensors::tensor::TensorView;

#[derive(Debug, thiserror::Error)]
pub enum DFlash2TensorsError {
    #[error("DFlash2 tensor missing after schema validation: {0}")]
    Missing(String),
    #[error("DFlash2 tensor upload: {0}")]
    Mlx(#[from] MlxError),
}

pub struct DFlash2LayerTensors {
    pub attention_dynamic: MlxBuffer,
    pub attention_convolution: MlxBuffer,
    pub input_norm: MlxBuffer,
    pub q_projection: MlxBuffer,
    pub k_projection: MlxBuffer,
    pub v_projection: MlxBuffer,
    pub output_projection: MlxBuffer,
    pub query_norm: MlxBuffer,
    pub key_norm: MlxBuffer,
    pub mlp_dynamic: MlxBuffer,
    pub post_attention_norm: MlxBuffer,
    pub gate_projection: MlxBuffer,
    pub up_projection: MlxBuffer,
    pub down_projection: MlxBuffer,
    pub mlp_convolution: MlxBuffer,
}

pub struct DFlash2ModelTensors {
    pub context_projection: MlxBuffer,
    pub hidden_norm: MlxBuffer,
    pub final_norm: MlxBuffer,
    pub selector_projection: MlxBuffer,
    pub predecessor_codebook: MlxBuffer,
    pub successor_codebook: MlxBuffer,
    pub mask_embedding: Option<MlxBuffer>,
    pub layers: Vec<DFlash2LayerTensors>,
}

fn tensor<'view, 'data>(
    weights: &'view DFlash2Weights<'data>,
    name: &str,
) -> Result<&'view TensorView<'data>, DFlash2TensorsError> {
    weights
        .tensor(name)
        .ok_or_else(|| DFlash2TensorsError::Missing(name.into()))
}

fn upload_bf16(
    device: &MlxDevice,
    view: &TensorView<'_>,
) -> Result<MlxBuffer, DFlash2TensorsError> {
    let mut buffer = device.alloc_buffer(view.data().len(), DType::BF16, view.shape().to_vec())?;
    buffer.as_mut_slice::<u8>()?.copy_from_slice(view.data());
    Ok(buffer)
}

fn upload_norm_f32(
    device: &MlxDevice,
    view: &TensorView<'_>,
) -> Result<MlxBuffer, DFlash2TensorsError> {
    let mut buffer =
        device.alloc_buffer(view.data().len() * 2, DType::F32, view.shape().to_vec())?;
    let output = buffer.as_mut_slice::<f32>()?;
    for (value, bytes) in output.iter_mut().zip(view.data().chunks_exact(2)) {
        *value = f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16);
    }
    Ok(buffer)
}

impl DFlash2ModelTensors {
    pub fn upload(
        device: &MlxDevice,
        config: &DFlash2Config,
        weights: &DFlash2Weights<'_>,
    ) -> Result<Self, DFlash2TensorsError> {
        let context_projection = upload_bf16(device, tensor(weights, "fc.weight")?)?;
        let hidden_norm = upload_norm_f32(device, tensor(weights, "hidden_norm.weight")?)?;
        let final_norm = upload_norm_f32(device, tensor(weights, "norm.weight")?)?;
        let selector_projection = upload_bf16(
            device,
            tensor(weights, "candidate_selector.hidden_projection.weight")?,
        )?;
        let predecessor_codebook = upload_bf16(
            device,
            tensor(weights, "candidate_selector.predecessor_codebook")?,
        )?;
        let successor_codebook = upload_bf16(
            device,
            tensor(weights, "candidate_selector.successor_codebook")?,
        )?;
        let mask_embedding = weights
            .tensor(OPTIONAL_MASK_EMBEDDING)
            .map(|view| upload_bf16(device, view))
            .transpose()?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("layers.{layer}");
            let bf16 =
                |suffix: &str| upload_bf16(device, tensor(weights, &format!("{prefix}.{suffix}"))?);
            let norm = |suffix: &str| {
                upload_norm_f32(device, tensor(weights, &format!("{prefix}.{suffix}"))?)
            };
            layers.push(DFlash2LayerTensors {
                attention_dynamic: bf16("attention_conv.kernel_projection.weight")?,
                attention_convolution: bf16("attention_conv.base_kernel")?,
                input_norm: norm("input_layernorm.weight")?,
                q_projection: bf16("self_attn.q_proj.weight")?,
                k_projection: bf16("self_attn.k_proj.weight")?,
                v_projection: bf16("self_attn.v_proj.weight")?,
                output_projection: bf16("self_attn.o_proj.weight")?,
                query_norm: norm("self_attn.q_norm.weight")?,
                key_norm: norm("self_attn.k_norm.weight")?,
                mlp_dynamic: bf16("mlp_conv.kernel_projection.weight")?,
                post_attention_norm: norm("post_attention_layernorm.weight")?,
                gate_projection: bf16("mlp.gate_proj.weight")?,
                up_projection: bf16("mlp.up_proj.weight")?,
                down_projection: bf16("mlp.down_proj.weight")?,
                mlp_convolution: bf16("mlp_conv.base_kernel")?,
            });
        }
        Ok(Self {
            context_projection,
            hidden_norm,
            final_norm,
            selector_projection,
            predecessor_codebook,
            successor_codebook,
            mask_embedding,
            layers,
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn bf16_norm_decode_is_exact() {
        let bytes = [0x80, 0x3f, 0x00, 0xc0];
        let values: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
            .collect();
        assert_eq!(values, [1.0, -2.0]);
    }
}
