//! Stored-format embedding gather for Gemma input activations.

use crate::serve::forward_mlx_shared::MlxQWeight;
use anyhow::{bail, Result};
use mlx_native::{GgmlType, MlxBuffer, MlxDevice};

/// Gather stored embedding rows and apply Gemma's sqrt(hidden) scale.
#[allow(clippy::too_many_arguments)]
pub fn encode_embedding_rows(
    session: &mut mlx_native::GraphSession<'_>,
    registry: &mut mlx_native::KernelRegistry,
    device: &MlxDevice,
    weight: &MlxQWeight,
    token_ids: &MlxBuffer,
    output: &MlxBuffer,
    n_tokens: usize,
) -> Result<()> {
    let params = (weight.info.rows, weight.info.cols, n_tokens);
    match weight.info.ggml_dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
            mlx_native::ops::embedding_dense::embedding_gather_dense(
                session.encoder_mut(),
                registry,
                device,
                &weight.buffer,
                token_ids,
                output,
                &mlx_native::ops::embedding_dense::EmbeddingDenseParams {
                    vocab_size: params.0,
                    embed_dim: params.1,
                    n_tokens: params.2,
                },
            )?
        }
        GgmlType::Q4_0 => session.embedding_gather_q4_0(
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::ops::embedding_q4_0::EmbeddingQ4_0Params {
                vocab_size: params.0,
                embed_dim: params.1,
                n_tokens: params.2,
            },
        )?,
        GgmlType::Q5_0 => session.embedding_gather_q5_0(
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::EmbeddingQ5_0Params {
                vocab_size: params.0,
                embed_dim: params.1,
                n_tokens: params.2,
            },
        )?,
        GgmlType::Q2_K => session.embedding_gather_q2_k(
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::ops::embedding_q2_k::EmbeddingQ2KParams {
                vocab_size: params.0,
                embed_dim: params.1,
                n_tokens: params.2,
            },
        )?,
        GgmlType::Q4_K => session.embedding_gather_q4_k(
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::ops::embedding_q4_k::EmbeddingQ4KParams {
                vocab_size: params.0,
                embed_dim: params.1,
                n_tokens: params.2,
            },
        )?,
        GgmlType::Q5_K => session.embedding_gather_q5_k(
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::ops::embedding_kquant::EmbeddingQ5KParams {
                vocab_size: params.0,
                embed_dim: params.1,
                n_tokens: params.2,
            },
        )?,
        GgmlType::Q6_K => session.embedding_gather_q6_k(
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::ops::embedding_kquant::EmbeddingQ6KParams {
                vocab_size: params.0,
                embed_dim: params.1,
                n_tokens: params.2,
            },
        )?,
        GgmlType::Q8_0 => session.embedding_gather_q8_0(
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::ops::embedding_q8_0::EmbeddingQ8_0Params {
                vocab_size: params.0,
                embed_dim: params.1,
                n_tokens: params.2,
            },
        )?,
        other => bail!("Gemma embedding dispatch reached unadmitted type {other:?}"),
    }
    session.track_dispatch(&[&weight.buffer, token_ids], &[output]);
    Ok(())
}

/// Gather stored embedding rows and apply Gemma's model-input scale.
#[allow(clippy::too_many_arguments)]
pub fn encode_embedding(
    session: &mut mlx_native::GraphSession<'_>,
    registry: &mut mlx_native::KernelRegistry,
    device: &MlxDevice,
    weight: &MlxQWeight,
    token_ids: &MlxBuffer,
    output: &MlxBuffer,
    n_tokens: usize,
) -> Result<()> {
    encode_embedding_rows(
        session, registry, device, weight, token_ids, output, n_tokens,
    )?;
    session.barrier_between(&[output], &[output]);
    mlx_native::ops::elementwise::scalar_mul_f32(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        output,
        output,
        n_tokens * weight.info.cols,
        (weight.info.cols as f32).sqrt(),
    )?;
    session.track_dispatch(&[output], &[output]);
    Ok(())
}
