//! Bounded source-BF16 input/output primitives for the private teacher runner.
//!
//! These helpers materialize only selected embedding rows and one vocabulary
//! logit row. They do not expose a model/session or mint execution authority.

use anyhow::{ensure, Context, Result};
use half::bf16;
use mlx_native::ops::rms_norm;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use crate::inference::models::qwen35::gpu_full_attn::apply_linear_projection_f32;

pub(super) fn gather_bf16_embedding_rows(
    device: &MlxDevice,
    embedding: &MlxBuffer,
    token_ids: &[u32],
    vocabulary_size: u32,
    hidden_size: u32,
) -> Result<MlxBuffer> {
    ensure!(
        !token_ids.is_empty(),
        "source teacher embedding input is empty"
    );
    ensure!(
        token_ids.iter().all(|token_id| *token_id < vocabulary_size),
        "source teacher embedding input contains a token outside vocabulary {vocabulary_size}"
    );
    ensure!(
        embedding.dtype() == DType::BF16
            && embedding.shape() == [vocabulary_size as usize, hidden_size as usize]
            && embedding.data_byte_len()
                == usize::try_from(vocabulary_size)?
                    .checked_mul(usize::try_from(hidden_size)?)
                    .and_then(|value| value.checked_mul(2))
                    .context("source teacher embedding size overflow")?,
        "source teacher embedding differs from projected config"
    );
    let rows = token_ids.len();
    let hidden = usize::try_from(hidden_size)?;
    let element_count = rows
        .checked_mul(hidden)
        .context("source teacher embedding activation overflow")?;
    let source = embedding
        .as_slice::<u16>()
        .context("view source teacher BF16 embedding")?;
    let mut output = device
        .alloc_buffer(
            element_count
                .checked_mul(4)
                .context("source teacher embedding output bytes overflow")?,
            DType::F32,
            vec![rows, hidden],
        )
        .context("allocate source teacher embedding activation")?;
    let destination = output
        .as_mut_slice::<f32>()
        .context("view source teacher embedding activation")?;
    for (row, token_id) in token_ids.iter().copied().enumerate() {
        let source_start = usize::try_from(token_id)?
            .checked_mul(hidden)
            .context("source teacher embedding row offset overflow")?;
        let destination_start = row
            .checked_mul(hidden)
            .context("source teacher embedding destination offset overflow")?;
        for feature in 0..hidden {
            destination[destination_start + feature] =
                bf16::from_bits(source[source_start + feature]).to_f32();
        }
    }
    Ok(output)
}

pub(super) fn text_positions(
    device: &MlxDevice,
    first_position: u32,
    token_count: u32,
) -> Result<MlxBuffer> {
    ensure!(token_count > 0, "source teacher position input is empty");
    let end = first_position
        .checked_add(token_count)
        .context("source teacher position range overflow")?;
    ensure!(
        end <= i32::MAX as u32,
        "source teacher position exceeds I32"
    );
    let len = usize::try_from(token_count)?;
    let mut output = device
        .alloc_buffer(
            len.checked_mul(4)
                .and_then(|value| value.checked_mul(4))
                .context("source teacher position bytes overflow")?,
            DType::I32,
            vec![4, len],
        )
        .context("allocate source teacher text positions")?;
    let destination = output
        .as_mut_slice::<i32>()
        .context("view source teacher text positions")?;
    for axis in 0..4_usize {
        for token in 0..len {
            destination[axis * len + token] = i32::try_from(
                first_position
                    .checked_add(u32::try_from(token)?)
                    .context("source teacher token position overflow")?,
            )?;
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn source_bf16_output_head_last(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    hidden: &MlxBuffer,
    output_norm: &MlxBuffer,
    output_weight: &MlxBuffer,
    sequence_length: u32,
    hidden_size: u32,
    vocabulary_size: u32,
    rms_norm_eps: f32,
) -> Result<Vec<f32>> {
    ensure!(
        sequence_length > 0,
        "source teacher output sequence is empty"
    );
    ensure!(
        vocabulary_size > 0 && vocabulary_size % 2 == 0,
        "source teacher v1 requires an even nonzero vocabulary for paired-row BF16 GEMV"
    );
    let hidden_elements = usize::try_from(sequence_length)?
        .checked_mul(usize::try_from(hidden_size)?)
        .context("source teacher hidden size overflow")?;
    let hidden_bytes = hidden_elements
        .checked_mul(4)
        .context("source teacher hidden byte size overflow")?;
    let norm_bytes = usize::try_from(hidden_size)?
        .checked_mul(4)
        .context("source teacher output norm byte size overflow")?;
    let output_bytes = usize::try_from(vocabulary_size)?
        .checked_mul(usize::try_from(hidden_size)?)
        .and_then(|elements| elements.checked_mul(2))
        .context("source teacher output weight byte size overflow")?;
    ensure!(
        hidden.dtype() == DType::F32
            && hidden.element_count() == hidden_elements
            && hidden.data_byte_len() == hidden_bytes
            && output_norm.dtype() == DType::F32
            && output_norm.shape() == [hidden_size as usize]
            && output_norm.data_byte_len() == norm_bytes
            && output_weight.dtype() == DType::BF16
            && output_weight.shape() == [vocabulary_size as usize, hidden_size as usize]
            && output_weight.data_byte_len() == output_bytes,
        "source teacher output buffers differ from projected config"
    );
    let last_row_offset = u64::from(sequence_length - 1)
        .checked_mul(u64::from(hidden_size))
        .and_then(|value| value.checked_mul(4))
        .context("source teacher last-row offset overflow")?;
    let last_hidden = hidden.slice_view(last_row_offset, hidden_size as usize);
    let normed = device
        .alloc_buffer(
            usize::try_from(hidden_size)?
                .checked_mul(4)
                .context("source teacher norm output bytes overflow")?,
            DType::F32,
            vec![1, hidden_size as usize],
        )
        .context("allocate source teacher normalized output")?;
    let mut params = device
        .alloc_buffer(8, DType::F32, vec![2])
        .context("allocate source teacher norm parameters")?;
    params
        .as_mut_slice::<f32>()
        .context("view source teacher norm parameters")?
        .copy_from_slice(&[rms_norm_eps, hidden_size as f32]);

    let mut encoder = device
        .command_encoder()
        .context("create source teacher output encoder")?;
    rms_norm::dispatch_rms_norm(
        &mut encoder,
        registry,
        device.metal_device(),
        &last_hidden,
        output_norm,
        &normed,
        &params,
        1,
        hidden_size,
    )
    .context("encode source teacher output RMSNorm")?;
    encoder.memory_barrier();
    let logits = apply_linear_projection_f32(
        &mut encoder,
        registry,
        device,
        &normed,
        output_weight,
        1,
        hidden_size,
        vocabulary_size,
    )
    .context("encode source teacher BF16 output projection")?;
    encoder
        .commit_and_wait_labeled("qwen35.source_teacher.output_head")
        .context("complete source teacher output head")?;
    let values = logits
        .as_slice::<f32>()
        .context("read completed source teacher logits")?
        .to_vec();
    ensure!(
        values.len() == vocabulary_size as usize && values.iter().all(|value| value.is_finite()),
        "source teacher output row is malformed or non-finite"
    );
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_output_head(
        hidden: &[f32],
        norm: &[f32],
        weight: &[u16],
        hidden_size: usize,
        vocabulary_size: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mean_square =
            hidden.iter().map(|value| value * value).sum::<f32>() / hidden_size as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        (0..vocabulary_size)
            .map(|token| {
                (0..hidden_size)
                    .map(|feature| {
                        hidden[feature]
                            * scale
                            * norm[feature]
                            * bf16::from_bits(weight[token * hidden_size + feature]).to_f32()
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn source_teacher_io_materializes_selected_rows_positions_and_one_completed_logit_row() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().unwrap();
        let vocabulary_size = 32_u32;
        let hidden_size = 64_u32;
        let mut embedding = device
            .alloc_buffer(
                vocabulary_size as usize * hidden_size as usize * 2,
                DType::BF16,
                vec![vocabulary_size as usize, hidden_size as usize],
            )
            .unwrap();
        let embedding_bits: Vec<u16> = (0..vocabulary_size as usize * hidden_size as usize)
            .map(|index| bf16::from_f32(((index * 17 % 101) as f32 - 50.0) / 128.0).to_bits())
            .collect();
        embedding
            .as_mut_slice::<u16>()
            .unwrap()
            .copy_from_slice(&embedding_bits);
        let gathered = gather_bf16_embedding_rows(
            &device,
            &embedding,
            &[3, 1, 31],
            vocabulary_size,
            hidden_size,
        )
        .unwrap();
        let gathered_values = gathered.as_slice::<f32>().unwrap();
        for (row, token) in [3_usize, 1, 31].into_iter().enumerate() {
            for feature in 0..hidden_size as usize {
                assert_eq!(
                    gathered_values[row * hidden_size as usize + feature],
                    bf16::from_bits(embedding_bits[token * hidden_size as usize + feature])
                        .to_f32()
                );
            }
        }

        let positions = text_positions(&device, 7, 3).unwrap();
        assert_eq!(
            positions.as_slice::<i32>().unwrap(),
            &[7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9]
        );

        let hidden_values: Vec<f32> = (0..2 * hidden_size as usize)
            .map(|index| ((index * 13 % 89) as f32 - 44.0) / 64.0)
            .collect();
        let mut hidden = device
            .alloc_buffer(
                hidden_values.len() * 4,
                DType::F32,
                vec![2, hidden_size as usize],
            )
            .unwrap();
        hidden
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&hidden_values);
        let norm_values: Vec<f32> = (0..hidden_size as usize)
            .map(|index| 0.75 + index as f32 / 512.0)
            .collect();
        let mut norm = device
            .alloc_buffer(
                hidden_size as usize * 4,
                DType::F32,
                vec![hidden_size as usize],
            )
            .unwrap();
        norm.as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&norm_values);
        let head_bits: Vec<u16> = (0..vocabulary_size as usize * hidden_size as usize)
            .map(|index| bf16::from_f32(((index * 29 % 127) as f32 - 63.0) / 256.0).to_bits())
            .collect();
        let mut head = device
            .alloc_buffer(
                head_bits.len() * 2,
                DType::BF16,
                vec![vocabulary_size as usize, hidden_size as usize],
            )
            .unwrap();
        head.as_mut_slice::<u16>()
            .unwrap()
            .copy_from_slice(&head_bits);
        let mut registry = KernelRegistry::new();
        let eps = 1.0e-6;
        let actual = source_bf16_output_head_last(
            &device,
            &mut registry,
            &hidden,
            &norm,
            &head,
            2,
            hidden_size,
            vocabulary_size,
            eps,
        )
        .unwrap();
        let expected = cpu_output_head(
            &hidden_values[hidden_size as usize..],
            &norm_values,
            &head_bits,
            hidden_size as usize,
            vocabulary_size as usize,
            eps,
        );
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs <= 2.0e-3, "output-head max_abs={max_abs}");
        assert!(actual.iter().any(|value| *value != 0.0));
        assert!(
            registry
                .pipeline_identity("hf2q_dense_gemv_bf16_f32_4")
                .is_ok(),
            "one-row source BF16 head did not resolve GEMV"
        );
    }

    #[test]
    fn source_teacher_io_rejects_oov_and_odd_vocabulary_before_dispatch() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().unwrap();
        let embedding = device
            .alloc_buffer(4 * 64 * 2, DType::BF16, vec![4, 64])
            .unwrap();
        assert!(gather_bf16_embedding_rows(&device, &embedding, &[4], 4, 64).is_err());

        let hidden = device
            .alloc_buffer(64 * 4, DType::F32, vec![1, 64])
            .unwrap();
        let norm = device.alloc_buffer(64 * 4, DType::F32, vec![64]).unwrap();
        let head = device
            .alloc_buffer(31 * 64 * 2, DType::BF16, vec![31, 64])
            .unwrap();
        let mut registry = KernelRegistry::new();
        assert!(source_bf16_output_head_last(
            &device,
            &mut registry,
            &hidden,
            &norm,
            &head,
            1,
            64,
            31,
            1.0e-6,
        )
        .is_err());
        assert!(
            registry
                .pipeline_identity("hf2q_dense_gemv_bf16_f32_4")
                .is_err(),
            "odd-vocabulary rejection reached GEMV"
        );
    }
}
