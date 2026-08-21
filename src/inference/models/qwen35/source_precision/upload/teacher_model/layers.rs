use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};
use mlx_native::{DType, GgmlType, MlxBuffer};
use sha2::{Digest, Sha256};

use crate::inference::models::qwen35::gpu_delta_net::DeltaNetWeightsGpu;
use crate::inference::models::qwen35::gpu_full_attn::{
    FullAttnQGateWeightsGpu, FullAttnWeightsGpu,
};
use crate::inference::models::qwen35::source_precision::topology::Qwen35FutureDType;
use crate::inference::models::qwen35::Qwen35Config;

use super::super::QwenUploadedOutputRecordV1;
use super::PreparedWeightSlotV1;

pub(super) struct OutputEntry {
    pub(super) source_name: String,
    pub(super) output: QwenUploadedOutputRecordV1,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn full_attention(
    buffers: &mut BTreeMap<String, MlxBuffer>,
    outputs: &mut BTreeMap<String, OutputEntry>,
    slots: &mut Vec<PreparedWeightSlotV1>,
    layer: usize,
    config: &Qwen35Config,
    attn_norm: MlxBuffer,
    post_attn_norm: MlxBuffer,
    device_registry_id: u64,
) -> Result<FullAttnWeightsGpu> {
    let prefix = format!("blk.{layer}");
    let hidden = usize::try_from(config.hidden_size)?;
    let query_rows = usize::try_from(config.num_attention_heads)?
        .checked_mul(usize::try_from(config.head_dim)?)
        .context("prepared full-attention query rows overflow")?;
    let kv_rows = usize::try_from(config.num_key_value_heads)?
        .checked_mul(usize::try_from(config.head_dim)?)
        .context("prepared full-attention KV rows overflow")?;
    let mut take = |role: &str, suffix: &str, shape, dtype| {
        take_slot(
            buffers,
            outputs,
            slots,
            &format!("layer.{layer}.{role}"),
            &format!("{prefix}.{suffix}"),
            shape,
            dtype,
            device_registry_id,
        )
    };
    Ok(FullAttnWeightsGpu {
        attn_norm,
        post_attn_norm,
        q_gate: FullAttnQGateWeightsGpu::Split {
            wq: take(
                "attn_q",
                "attn_q.q",
                vec![query_rows, hidden],
                Qwen35FutureDType::Bf16,
            )?,
            wq_ggml_type: GgmlType::F32,
            w_gate: take(
                "attn_gate",
                "attn_q.gate",
                vec![query_rows, hidden],
                Qwen35FutureDType::Bf16,
            )?,
            w_gate_ggml_type: GgmlType::F32,
        },
        wk: take(
            "attn_k",
            "attn_k.weight",
            vec![kv_rows, hidden],
            Qwen35FutureDType::Bf16,
        )?,
        wk_ggml_type: GgmlType::F32,
        wv: take(
            "attn_v",
            "attn_v.weight",
            vec![kv_rows, hidden],
            Qwen35FutureDType::Bf16,
        )?,
        wv_ggml_type: GgmlType::F32,
        attn_q_norm: take(
            "attn_q_norm",
            "attn_q_norm.weight",
            vec![config.head_dim as usize],
            Qwen35FutureDType::F32,
        )?,
        attn_k_norm: take(
            "attn_k_norm",
            "attn_k_norm.weight",
            vec![config.head_dim as usize],
            Qwen35FutureDType::F32,
        )?,
        wo: take(
            "attn_output",
            "attn_output.weight",
            vec![hidden, query_rows],
            Qwen35FutureDType::Bf16,
        )?,
        wo_ggml_type: GgmlType::F32,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn linear_attention(
    buffers: &mut BTreeMap<String, MlxBuffer>,
    outputs: &mut BTreeMap<String, OutputEntry>,
    slots: &mut Vec<PreparedWeightSlotV1>,
    layer: usize,
    config: &Qwen35Config,
    attn_norm: MlxBuffer,
    post_attn_norm: MlxBuffer,
    device_registry_id: u64,
) -> Result<DeltaNetWeightsGpu> {
    let prefix = format!("blk.{layer}");
    let hidden = config.hidden_size as usize;
    let nk = config.linear_num_key_heads as usize;
    let nv = config.linear_num_value_heads as usize;
    let d = config.linear_key_head_dim as usize;
    let k = config.linear_conv_kernel_dim as usize;
    let qkv = 2_usize
        .checked_mul(nk)
        .and_then(|value| value.checked_mul(d))
        .and_then(|value| value.checked_add(nv.checked_mul(d)?))
        .context("prepared Delta channels overflow")?;
    let value_rows = nv
        .checked_mul(d)
        .context("prepared Delta value rows overflow")?;
    let mut take = |role: &str, suffix: &str, shape, dtype| {
        take_slot(
            buffers,
            outputs,
            slots,
            &format!("layer.{layer}.{role}"),
            &format!("{prefix}.{suffix}"),
            shape,
            dtype,
            device_registry_id,
        )
    };
    let attn_qkv = take(
        "attn_qkv",
        "attn_qkv.weight",
        vec![qkv, hidden],
        Qwen35FutureDType::Bf16,
    )?;
    let attn_gate = take(
        "attn_gate",
        "attn_gate.weight",
        vec![value_rows, hidden],
        Qwen35FutureDType::Bf16,
    )?;
    let ssm_conv1d = take(
        "ssm_conv1d",
        "ssm_conv1d.weight",
        vec![qkv, k],
        Qwen35FutureDType::F32,
    )?;
    let ssm_alpha = take(
        "ssm_alpha",
        "ssm_alpha.weight",
        vec![nv, hidden],
        Qwen35FutureDType::Bf16,
    )?;
    let ssm_dt_bias = take(
        "ssm_dt_bias",
        "ssm_dt.bias",
        vec![nv],
        Qwen35FutureDType::F32,
    )?;
    let ssm_dt_bias_cpu = ssm_dt_bias.as_slice::<f32>()?.to_vec();
    let ssm_beta = take(
        "ssm_beta",
        "ssm_beta.weight",
        vec![nv, hidden],
        Qwen35FutureDType::Bf16,
    )?;
    let ssm_a = take("ssm_a", "ssm_a", vec![nv], Qwen35FutureDType::F32)?;
    let ssm_a_cpu = ssm_a.as_slice::<f32>()?.to_vec();
    let ssm_norm = take(
        "ssm_norm",
        "ssm_norm.weight",
        vec![d],
        Qwen35FutureDType::F32,
    )?;
    let ssm_norm_cpu = ssm_norm.as_slice::<f32>()?.to_vec();
    let ssm_out = take(
        "ssm_out",
        "ssm_out.weight",
        vec![hidden, value_rows],
        Qwen35FutureDType::Bf16,
    )?;
    Ok(DeltaNetWeightsGpu {
        attn_norm,
        post_attn_norm,
        attn_qkv,
        attn_qkv_ggml_type: GgmlType::F32,
        attn_gate,
        attn_gate_ggml_type: GgmlType::F32,
        ssm_conv1d,
        ssm_alpha,
        ssm_alpha_ggml_type: GgmlType::F32,
        ssm_dt_bias,
        ssm_dt_bias_cpu,
        ssm_beta,
        ssm_beta_ggml_type: GgmlType::F32,
        ssm_a,
        ssm_a_cpu,
        ssm_norm,
        ssm_norm_cpu,
        ssm_out,
        ssm_out_ggml_type: GgmlType::F32,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn take_slot(
    buffers: &mut BTreeMap<String, MlxBuffer>,
    outputs: &mut BTreeMap<String, OutputEntry>,
    slots: &mut Vec<PreparedWeightSlotV1>,
    role: &str,
    node_id: &str,
    expected_shape: Vec<usize>,
    expected_dtype: Qwen35FutureDType,
    device_registry_id: u64,
) -> Result<MlxBuffer> {
    let entry = outputs
        .remove(node_id)
        .with_context(|| format!("prepared source teacher lacks output {node_id}"))?;
    ensure!(
        entry.output.shape == expected_shape && entry.output.dtype == expected_dtype,
        "prepared source teacher output {node_id} differs from role {role}"
    );
    let buffer = buffers
        .remove(node_id)
        .with_context(|| format!("prepared source teacher lacks buffer {node_id}"))?;
    let dtype = match expected_dtype {
        Qwen35FutureDType::Bf16 => DType::BF16,
        Qwen35FutureDType::F32 => DType::F32,
    };
    let bytes: &[u8] = match expected_dtype {
        Qwen35FutureDType::Bf16 => bytemuck::cast_slice(buffer.as_slice::<u16>()?),
        Qwen35FutureDType::F32 => bytemuck::cast_slice(buffer.as_slice::<f32>()?),
    };
    ensure!(
        buffer.dtype() == dtype
            && buffer.shape() == expected_shape
            && buffer.byte_len() == bytes.len()
            && buffer.data_byte_len() == bytes.len()
            && buffer.byte_offset() == 0
            && !buffer.is_file_backed()
            && buffer.is_cpu_writable()
            && buffer.metal_buffer().device().registry_id() == device_registry_id
            && u64::try_from(bytes.len())? == entry.output.byte_len
            && hex::encode(Sha256::digest(bytes)) == entry.output.buffer_byte_sha256,
        "prepared source teacher buffer {node_id} differs from B2b"
    );
    slots.push(PreparedWeightSlotV1 {
        role: role.to_owned(),
        source_name: entry.source_name,
        node_id: node_id.to_owned(),
        shape: expected_shape,
        dtype: expected_dtype,
        transform: entry.output.transform,
        byte_len: entry.output.byte_len,
        buffer_byte_sha256: entry.output.buffer_byte_sha256,
    });
    Ok(buffer)
}

pub(super) fn slot_totals(slots: &[PreparedWeightSlotV1]) -> Result<(usize, usize, u64, u64)> {
    let mut bf16_count = 0;
    let mut f32_count = 0;
    let mut bf16_bytes = 0_u64;
    let mut f32_bytes = 0_u64;
    for slot in slots {
        match slot.dtype {
            Qwen35FutureDType::Bf16 => {
                bf16_count += 1;
                bf16_bytes = bf16_bytes
                    .checked_add(slot.byte_len)
                    .context("prepared BF16 bytes overflow")?;
            }
            Qwen35FutureDType::F32 => {
                f32_count += 1;
                f32_bytes = f32_bytes
                    .checked_add(slot.byte_len)
                    .context("prepared F32 bytes overflow")?;
            }
        }
    }
    Ok((bf16_count, f32_count, bf16_bytes, f32_bytes))
}
