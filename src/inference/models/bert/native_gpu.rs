//! Native-storage matrix and embedding dispatch for encoder models.

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::ops::dense_gemm::{
    dispatch_dense_matvec_bf16w_f32io, dispatch_dense_matvec_f16w_f32io, dispatch_dense_matvec_f32,
    DenseGemmF16Params,
};
use mlx_native::ops::dense_mm_bf16::{dense_matmul_bf16_f32_tensor, DenseMmBf16F32Params};
use mlx_native::ops::dense_mm_f16::{dense_matmul_f16_f32_tensor, DenseMmF16F32Params};
use mlx_native::ops::dense_mm_f32_f32::{dense_matmul_f32_f32_tensor, DenseMmF32F32Params};
use mlx_native::ops::embedding_dense::{embedding_gather_dense, EmbeddingDenseParams};
use mlx_native::ops::embedding_kquant::{
    embedding_gather_q5_k, embedding_gather_q6_k, EmbeddingQ5KParams, EmbeddingQ6KParams,
};
use mlx_native::ops::embedding_q2_k::{embedding_gather_q2_k, EmbeddingQ2KParams};
use mlx_native::ops::embedding_q4_0::{embedding_gather_q4_0, EmbeddingQ4_0Params};
use mlx_native::ops::embedding_q4_k::{embedding_gather_q4_k, EmbeddingQ4KParams};
use mlx_native::ops::embedding_q8_0::{embedding_gather_q8_0, EmbeddingQ8_0Params};
use mlx_native::{
    quantized_matmul_ggml, CommandEncoder, DType, GgmlQuantizedMatmulParams, GgmlType,
    KernelRegistry, MlxBuffer, MlxDevice,
};

use crate::serve::forward_mlx_shared::MlxQWeight;

use super::bert_gpu::{
    bert_attention_gpu, bert_attention_with_mask_gpu, bert_bias_add_gpu, bert_gelu_gpu,
    bert_layer_norm_gpu, bert_residual_add_gpu, bert_residual_layer_norm_gpu,
};
use super::native_storage::native_embedding_codec_supported;

pub struct NativeBertEncoderBlock<'a> {
    pub q_w: &'a MlxQWeight,
    pub q_b: Option<&'a MlxBuffer>,
    pub k_w: &'a MlxQWeight,
    pub k_b: Option<&'a MlxBuffer>,
    pub v_w: &'a MlxQWeight,
    pub v_b: Option<&'a MlxBuffer>,
    pub o_w: &'a MlxQWeight,
    pub o_b: Option<&'a MlxBuffer>,
    pub attn_norm_weight: &'a MlxBuffer,
    pub attn_norm_bias: &'a MlxBuffer,
    pub up_w: &'a MlxQWeight,
    pub up_b: Option<&'a MlxBuffer>,
    pub down_w: &'a MlxQWeight,
    pub down_b: Option<&'a MlxBuffer>,
    pub output_norm_weight: &'a MlxBuffer,
    pub output_norm_bias: &'a MlxBuffer,
}

pub fn register_native_embedding_shaders(registry: &mut KernelRegistry) {
    mlx_native::ops::embedding_dense::register(registry);
    mlx_native::ops::embedding_q2_k::register(registry);
    mlx_native::ops::embedding_q4_0::register(registry);
    mlx_native::ops::embedding_q4_k::register(registry);
    mlx_native::ops::embedding_kquant::register(registry);
    mlx_native::ops::embedding_q8_0::register(registry);
}

#[allow(clippy::too_many_arguments)]
pub fn bert_linear_native_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxQWeight,
    bias: Option<&MlxBuffer>,
    rows: u32,
    in_features: u32,
    out_features: u32,
) -> Result<MlxBuffer> {
    ensure!(
        rows > 0 && in_features >= 32 && out_features > 0,
        "native linear dimensions must be nonzero and K >= 32"
    );
    ensure!(
        weight.affine.is_none(),
        "embedding native matrix cannot carry an affine overlay"
    );
    ensure!(
        weight.info.rows == out_features as usize && weight.info.cols == in_features as usize,
        "native matrix metadata [{}, {}] != requested [{out_features}, {in_features}]",
        weight.info.rows,
        weight.info.cols
    );
    ensure!(
        input.dtype() == DType::F32
            && input.element_count() >= rows as usize * in_features as usize,
        "native linear input must be F32 [{rows}, {in_features}]"
    );
    if let Some(bias) = bias {
        ensure!(
            bias.dtype() == DType::F32 && bias.element_count() == out_features as usize,
            "native linear bias must be F32 [{out_features}]"
        );
    }
    let output = device
        .alloc_buffer(
            rows as usize * out_features as usize * 4,
            DType::F32,
            vec![rows as usize, out_features as usize],
        )
        .map_err(|e| anyhow!("allocate native linear output: {e}"))?;

    let gemv = DenseGemmF16Params {
        m: rows,
        n: out_features,
        k: in_features,
    };
    match weight.info.ggml_dtype {
        GgmlType::F32 if rows == 1 => dispatch_dense_matvec_f32(
            encoder,
            registry,
            device.metal_device(),
            input,
            &weight.buffer,
            &output,
            &gemv,
        ),
        GgmlType::F16 if rows == 1 => dispatch_dense_matvec_f16w_f32io(
            encoder,
            registry,
            device.metal_device(),
            input,
            &weight.buffer,
            &output,
            &gemv,
        ),
        GgmlType::BF16 if rows == 1 => dispatch_dense_matvec_bf16w_f32io(
            encoder,
            registry,
            device.metal_device(),
            input,
            &weight.buffer,
            &output,
            &gemv,
        ),
        GgmlType::F32 => dense_matmul_f32_f32_tensor(
            encoder,
            registry,
            device,
            &weight.buffer,
            input,
            &output,
            &DenseMmF32F32Params {
                m: rows,
                n: out_features,
                k: in_features,
                src0_batch: 1,
                src1_batch: 1,
            },
        ),
        GgmlType::F16 => dense_matmul_f16_f32_tensor(
            encoder,
            registry,
            device,
            &weight.buffer,
            input,
            &output,
            &DenseMmF16F32Params {
                m: rows,
                n: out_features,
                k: in_features,
                src0_batch: 1,
                src1_batch: 1,
            },
        ),
        GgmlType::BF16 => dense_matmul_bf16_f32_tensor(
            encoder,
            registry,
            device,
            &weight.buffer,
            input,
            &output,
            &DenseMmBf16F32Params {
                m: rows,
                n: out_features,
                k: in_features,
                src0_batch: 1,
                src1_batch: 1,
            },
        ),
        ggml_type => quantized_matmul_ggml(
            encoder,
            registry,
            device,
            input,
            &weight.buffer,
            &output,
            &GgmlQuantizedMatmulParams {
                m: rows,
                n: out_features,
                k: in_features,
                ggml_type,
            },
        ),
    }
    .with_context(|| format!("native {:?} linear", weight.info.ggml_dtype))?;

    if let Some(bias) = bias {
        encoder.memory_barrier();
        return bert_bias_add_gpu(encoder, registry, device, &output, bias, rows, out_features)
            .context("native linear bias");
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn bert_embed_gather_native_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    table: &MlxQWeight,
    ids: &MlxBuffer,
    rows: u32,
    hidden: u32,
    n_ids: u32,
) -> Result<MlxBuffer> {
    ensure!(
        rows > 0 && hidden > 0 && n_ids > 0,
        "native embedding dimensions must be nonzero"
    );
    ensure!(
        table.affine.is_none(),
        "embedding native table cannot carry an affine overlay"
    );
    ensure!(
        native_embedding_codec_supported(table.info.ggml_dtype),
        "native embedding route unavailable for {:?}",
        table.info.ggml_dtype
    );
    ensure!(
        table.info.rows == rows as usize && table.info.cols == hidden as usize,
        "native embedding metadata [{}, {}] != requested [{rows}, {hidden}]",
        table.info.rows,
        table.info.cols
    );
    let output = device
        .alloc_buffer(
            n_ids as usize * hidden as usize * 4,
            DType::F32,
            vec![n_ids as usize, hidden as usize],
        )
        .map_err(|e| anyhow!("allocate native embedding output: {e}"))?;
    let vocab_size = rows as usize;
    let embed_dim = hidden as usize;
    let n_tokens = n_ids as usize;
    match table.info.ggml_dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => embedding_gather_dense(
            encoder,
            registry,
            device,
            &table.buffer,
            ids,
            &output,
            &EmbeddingDenseParams {
                vocab_size,
                embed_dim,
                n_tokens,
            },
        ),
        GgmlType::Q4_0 => embedding_gather_q4_0(
            encoder,
            registry,
            device,
            &table.buffer,
            ids,
            &output,
            &EmbeddingQ4_0Params {
                vocab_size,
                embed_dim,
                n_tokens,
            },
        ),
        GgmlType::Q8_0 => embedding_gather_q8_0(
            encoder,
            registry,
            device,
            &table.buffer,
            ids,
            &output,
            &EmbeddingQ8_0Params {
                vocab_size,
                embed_dim,
                n_tokens,
            },
        ),
        GgmlType::Q2_K => embedding_gather_q2_k(
            encoder,
            registry,
            device,
            &table.buffer,
            ids,
            &output,
            &EmbeddingQ2KParams {
                vocab_size,
                embed_dim,
                n_tokens,
            },
        ),
        GgmlType::Q4_K => embedding_gather_q4_k(
            encoder,
            registry,
            device,
            &table.buffer,
            ids,
            &output,
            &EmbeddingQ4KParams {
                vocab_size,
                embed_dim,
                n_tokens,
            },
        ),
        GgmlType::Q5_K => embedding_gather_q5_k(
            encoder,
            registry,
            device,
            &table.buffer,
            ids,
            &output,
            &EmbeddingQ5KParams {
                vocab_size,
                embed_dim,
                n_tokens,
            },
        ),
        GgmlType::Q6_K => embedding_gather_q6_k(
            encoder,
            registry,
            device,
            &table.buffer,
            ids,
            &output,
            &EmbeddingQ6KParams {
                vocab_size,
                embed_dim,
                n_tokens,
            },
        ),
        other => unreachable!("native embedding codec guard accepted {other:?}"),
    }
    .with_context(|| format!("native {:?} embedding gather", table.info.ggml_dtype))?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn bert_embeddings_native_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input_ids: &MlxBuffer,
    type_ids: Option<&MlxBuffer>,
    token_embd: &MlxQWeight,
    position_embd: &MlxQWeight,
    token_types: Option<&MlxQWeight>,
    norm_weight: &MlxBuffer,
    norm_bias: &MlxBuffer,
    eps: f32,
    seq_len: u32,
    hidden: u32,
    vocab: u32,
    max_pos: u32,
    type_vocab: u32,
) -> Result<MlxBuffer> {
    ensure!(
        seq_len > 0 && seq_len <= max_pos,
        "native BERT embedding sequence is out of range"
    );
    ensure!(
        type_ids.is_some() == token_types.is_some(),
        "native BERT type ids and table must both be present or absent"
    );
    let mut position_ids = device
        .alloc_buffer(seq_len as usize * 4, DType::U32, vec![seq_len as usize])
        .map_err(|e| anyhow!("allocate position ids: {e}"))?;
    position_ids
        .as_mut_slice::<u32>()?
        .iter_mut()
        .enumerate()
        .for_each(|(index, slot)| *slot = index as u32);
    let token = bert_embed_gather_native_gpu(
        encoder, registry, device, token_embd, input_ids, vocab, hidden, seq_len,
    )?;
    encoder.memory_barrier();
    let position = bert_embed_gather_native_gpu(
        encoder,
        registry,
        device,
        position_embd,
        &position_ids,
        max_pos,
        hidden,
        seq_len,
    )?;
    encoder.memory_barrier();
    let elements = seq_len
        .checked_mul(hidden)
        .ok_or_else(|| anyhow!("native BERT embedding element count overflow"))?;
    let token_position =
        bert_residual_add_gpu(encoder, registry, device, &token, &position, elements)?;
    encoder.memory_barrier();
    let summed = if let (Some(ids), Some(table)) = (type_ids, token_types) {
        let segment = bert_embed_gather_native_gpu(
            encoder, registry, device, table, ids, type_vocab, hidden, seq_len,
        )?;
        encoder.memory_barrier();
        let output = bert_residual_add_gpu(
            encoder,
            registry,
            device,
            &token_position,
            &segment,
            elements,
        )?;
        encoder.memory_barrier();
        output
    } else {
        token_position
    };
    bert_layer_norm_gpu(
        encoder,
        registry,
        device,
        &summed,
        norm_weight,
        norm_bias,
        eps,
        seq_len,
        hidden,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn apply_bert_encoder_block_native_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    tensors: &NativeBertEncoderBlock<'_>,
    attention_mask: Option<&MlxBuffer>,
    seq_len: u32,
    hidden: u32,
    num_heads: u32,
    intermediate: u32,
    eps: f32,
) -> Result<MlxBuffer> {
    ensure!(
        hidden > 0 && num_heads > 0 && hidden % num_heads == 0,
        "invalid native BERT head shape"
    );
    let head_dim = hidden / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q = bert_linear_native_gpu(
        encoder,
        registry,
        device,
        input,
        tensors.q_w,
        tensors.q_b,
        seq_len,
        hidden,
        hidden,
    )?;
    encoder.memory_barrier();
    let k = bert_linear_native_gpu(
        encoder,
        registry,
        device,
        input,
        tensors.k_w,
        tensors.k_b,
        seq_len,
        hidden,
        hidden,
    )?;
    encoder.memory_barrier();
    let v = bert_linear_native_gpu(
        encoder,
        registry,
        device,
        input,
        tensors.v_w,
        tensors.v_b,
        seq_len,
        hidden,
        hidden,
    )?;
    encoder.memory_barrier();
    let attention = match attention_mask {
        Some(mask) => bert_attention_with_mask_gpu(
            encoder, registry, device, &q, &k, &v, mask, seq_len, num_heads, head_dim, scale,
        ),
        None => bert_attention_gpu(
            encoder, registry, device, &q, &k, &v, seq_len, num_heads, head_dim, scale,
        ),
    }?;
    encoder.memory_barrier();
    let projected = bert_linear_native_gpu(
        encoder,
        registry,
        device,
        &attention,
        tensors.o_w,
        tensors.o_b,
        seq_len,
        hidden,
        hidden,
    )?;
    encoder.memory_barrier();
    let post_attention = bert_residual_layer_norm_gpu(
        encoder,
        registry,
        device,
        input,
        &projected,
        tensors.attn_norm_weight,
        tensors.attn_norm_bias,
        eps,
        seq_len,
        hidden,
    )?;
    encoder.memory_barrier();
    let up = bert_linear_native_gpu(
        encoder,
        registry,
        device,
        &post_attention,
        tensors.up_w,
        tensors.up_b,
        seq_len,
        hidden,
        intermediate,
    )?;
    encoder.memory_barrier();
    let activated = bert_gelu_gpu(encoder, registry, device, &up)?;
    encoder.memory_barrier();
    let down = bert_linear_native_gpu(
        encoder,
        registry,
        device,
        &activated,
        tensors.down_w,
        tensors.down_b,
        seq_len,
        intermediate,
        hidden,
    )?;
    encoder.memory_barrier();
    bert_residual_layer_norm_gpu(
        encoder,
        registry,
        device,
        &post_attention,
        &down,
        tensors.output_norm_weight,
        tensors.output_norm_bias,
        eps,
        seq_len,
        hidden,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::bert::native_storage::f32_qweight_for_test;

    #[test]
    fn native_linear_returns_the_post_bias_buffer() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        super::super::bert_gpu::register_bert_custom_shaders(&mut registry);
        let mut input = device
            .alloc_buffer(32 * 4, DType::F32, vec![1, 32])
            .expect("input");
        input.as_mut_slice::<f32>().expect("input slice").fill(1.0);
        let mut weight = device
            .alloc_buffer(32 * 32 * 4, DType::F32, vec![32, 32])
            .expect("weight");
        weight
            .as_mut_slice::<f32>()
            .expect("weight slice")
            .fill(0.0);
        let weight = f32_qweight_for_test(weight, 32, 32);
        let mut bias = device
            .alloc_buffer(32 * 4, DType::F32, vec![32])
            .expect("bias");
        for (index, value) in bias
            .as_mut_slice::<f32>()
            .expect("bias slice")
            .iter_mut()
            .enumerate()
        {
            *value = index as f32 + 0.25;
        }
        let mut encoder = device.command_encoder().expect("encoder");
        let output = bert_linear_native_gpu(
            &mut encoder,
            &mut registry,
            &device,
            &input,
            &weight,
            Some(&bias),
            1,
            32,
            32,
        )
        .expect("native biased linear");
        encoder.commit_and_wait().expect("commit");
        assert_eq!(
            output.as_slice::<f32>().expect("output"),
            bias.as_slice::<f32>().expect("bias"),
            "the returned buffer must include the bias dispatch result"
        );
    }
}
