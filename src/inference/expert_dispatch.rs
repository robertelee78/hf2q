//! Storage-aware native expert-ID dispatch.
//!
//! GGUF scalar stacks use mlx-native's native F32/F16/BF16 primitive while
//! block-quantized stacks retain the established GGML routes.  Both paths
//! preserve the existing flattened `(n_tokens, top_k)` contract.

use anyhow::{Context, Result};
use mlx_native::{
    DType, DenseMatmulIdInputLayout, DenseMatmulIdMultiplicity, DenseMatmulIdParams,
    DenseMatmulIdRoute, GgmlQuantizedMatmulIdParams, GgmlType, GraphSession, KernelRegistry,
    MlxBuffer, MlxDevice,
};

fn scalar_dtype(ggml_type: GgmlType) -> Option<DType> {
    match ggml_type {
        GgmlType::F32 => Some(DType::F32),
        GgmlType::F16 => Some(DType::F16),
        GgmlType::BF16 => Some(DType::BF16),
        _ => None,
    }
}

fn dispatch_scalar(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    expert_ids: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulIdParams,
) -> Result<bool> {
    let Some(expected_dtype) = scalar_dtype(params.ggml_type) else {
        return Ok(false);
    };
    anyhow::ensure!(
        weight.dtype() == expected_dtype,
        "native scalar expert stack declares {:?} but its buffer dtype is {}",
        params.ggml_type,
        weight.dtype()
    );
    let scalar_params = DenseMatmulIdParams {
        m: params.n_tokens,
        n: params.n,
        k: params.k,
        top_k: params.top_k,
        n_experts: params.n_experts,
        expert_stride_bytes: params.expert_stride,
        input_layout: DenseMatmulIdInputLayout::SharedPerToken,
        id_multiplicity: DenseMatmulIdMultiplicity::MayRepeat,
        route: DenseMatmulIdRoute::Direct,
    };
    mlx_native::dense_matmul_id(
        session.encoder_mut(),
        registry,
        device,
        weight,
        input,
        expert_ids,
        output,
        None,
        &scalar_params,
    )
    .context("native scalar expert-ID matmul")?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_expert_matmul_id(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    expert_ids: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    if dispatch_scalar(
        session, registry, device, input, weight, expert_ids, output, params,
    )? {
        return Ok(());
    }
    session
        .quantized_matmul_id_ggml(registry, device, input, weight, expert_ids, output, params)
        .context("block-quantized expert-ID matmul")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_expert_matmul_id_mv(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    expert_ids: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    if dispatch_scalar(
        session, registry, device, input, weight, expert_ids, output, params,
    )? {
        return Ok(());
    }
    session
        .quantized_matmul_id_ggml_mv(registry, device, input, weight, expert_ids, output, params)
        .context("block-quantized expert-ID matvec")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_expert_matmul_id_pooled(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    expert_ids: &MlxBuffer,
    output: &MlxBuffer,
    scratch: &mut mlx_native::IdMmScratch,
    params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    if dispatch_scalar(
        session, registry, device, input, weight, expert_ids, output, params,
    )? {
        return Ok(());
    }
    session
        .quantized_matmul_id_ggml_pooled(
            registry, device, input, weight, expert_ids, output, scratch, params,
        )
        .context("pooled block-quantized expert-ID matmul")
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_native::{GraphExecutor, MlxDevice};

    fn f32_buffer(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
        let mut buffer = device
            .alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])
            .unwrap();
        buffer
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(values);
        buffer
    }

    fn bf16_buffer(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
        let mut buffer = device
            .alloc_buffer(values.len() * 2, DType::BF16, vec![values.len()])
            .unwrap();
        let encoded = buffer.as_mut_slice::<u16>().unwrap();
        for (destination, value) in encoded.iter_mut().zip(values) {
            *destination = half::bf16::from_f32(*value).to_bits();
        }
        buffer
    }

    #[test]
    fn scalar_expert_dispatch_supports_repeated_ids_and_flattened_down_rows() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Ok(device) = MlxDevice::new() else {
            eprintln!("skipping scalar expert dispatch test: no Metal device");
            return;
        };
        // Three 2x2 experts. Expert 1 is selected twice, proving MayRepeat;
        // n_tokens=4/top_k=1 is the flattened expert-down representation.
        let weights = bf16_buffer(
            &device,
            &[
                1.0, 0.0, 0.0, 1.0, // expert 0: identity
                2.0, 0.0, 0.0, 3.0, // expert 1
                4.0, 0.0, 0.0, 5.0, // expert 2
            ],
        );
        let stored_before = weights.as_slice::<u16>().unwrap().to_vec();
        assert_eq!(weights.dtype(), DType::BF16);
        let input = f32_buffer(&device, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut ids = device.alloc_buffer(16, DType::U32, vec![4]).unwrap();
        ids.as_mut_slice::<u32>()
            .unwrap()
            .copy_from_slice(&[1, 1, 0, 2]);
        let output = device.alloc_buffer(32, DType::F32, vec![4, 2]).unwrap();
        let params = GgmlQuantizedMatmulIdParams {
            n_tokens: 4,
            top_k: 1,
            n: 2,
            k: 2,
            n_experts: 3,
            expert_stride: 8,
            ggml_type: GgmlType::BF16,
        };
        let executor = GraphExecutor::new(device.clone());
        let mut registry = KernelRegistry::new();
        let mut session = executor.begin().unwrap();
        dispatch_expert_matmul_id_mv(
            &mut session,
            &mut registry,
            &device,
            &input,
            &weights,
            &ids,
            &output,
            &params,
        )
        .unwrap();
        session.finish().unwrap();
        assert_eq!(weights.dtype(), DType::BF16);
        assert_eq!(weights.as_slice::<u16>().unwrap(), stored_before);
        assert_eq!(
            output.as_slice::<f32>().unwrap(),
            &[2.0, 6.0, 6.0, 12.0, 5.0, 6.0, 28.0, 40.0]
        );
    }

    #[test]
    fn scalar_expert_dispatch_shares_each_input_across_selected_experts() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Ok(device) = MlxDevice::new() else {
            return;
        };
        let weights = bf16_buffer(&device, &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0]);
        let stored = weights.as_slice::<u16>().unwrap().to_vec();
        let input = f32_buffer(&device, &[1.0, 2.0, 3.0, 4.0]);
        let mut ids = device.alloc_buffer(16, DType::U32, vec![2, 2]).unwrap();
        ids.as_mut_slice::<u32>()
            .unwrap()
            .copy_from_slice(&[1, 0, 1, 1]);
        let output = device.alloc_buffer(32, DType::F32, vec![2, 2, 2]).unwrap();
        let params = GgmlQuantizedMatmulIdParams {
            n_tokens: 2,
            top_k: 2,
            n: 2,
            k: 2,
            n_experts: 2,
            expert_stride: 8,
            ggml_type: GgmlType::BF16,
        };
        let executor = GraphExecutor::new(device.clone());
        let mut registry = KernelRegistry::new();
        let mut session = executor.begin().unwrap();
        dispatch_expert_matmul_id(
            &mut session,
            &mut registry,
            &device,
            &input,
            &weights,
            &ids,
            &output,
            &params,
        )
        .unwrap();
        session.finish().unwrap();
        assert_eq!(
            output.as_slice::<f32>().unwrap(),
            &[2.0, 6.0, 1.0, 2.0, 6.0, 12.0, 6.0, 12.0]
        );
        assert_eq!(weights.dtype(), DType::BF16);
        assert_eq!(weights.as_slice::<u16>().unwrap(), stored);
    }

    #[test]
    fn scalar_expert_dispatch_rejects_dtype_and_extent_mismatches() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Ok(device) = MlxDevice::new() else {
            eprintln!("skipping scalar expert validation test: no Metal device");
            return;
        };
        let input = f32_buffer(&device, &[1.0, 2.0]);
        let wrong_dtype = device.alloc_buffer(16, DType::F16, vec![8]).unwrap();
        let short_weight = f32_buffer(&device, &[1.0, 0.0, 0.0, 1.0]);
        let mut ids = device.alloc_buffer(4, DType::U32, vec![1]).unwrap();
        ids.as_mut_slice::<u32>().unwrap()[0] = 1;
        let output = device.alloc_buffer(8, DType::F32, vec![1, 2]).unwrap();
        let params = GgmlQuantizedMatmulIdParams {
            n_tokens: 1,
            top_k: 1,
            n: 2,
            k: 2,
            n_experts: 2,
            expert_stride: 16,
            ggml_type: GgmlType::F32,
        };
        let executor = GraphExecutor::new(device.clone());
        let mut registry = KernelRegistry::new();
        let mut session = executor.begin().unwrap();
        assert!(dispatch_expert_matmul_id(
            &mut session,
            &mut registry,
            &device,
            &input,
            &wrong_dtype,
            &ids,
            &output,
            &params,
        )
        .unwrap_err()
        .to_string()
        .contains("buffer dtype"));
        let error = dispatch_expert_matmul_id(
            &mut session,
            &mut registry,
            &device,
            &input,
            &short_weight,
            &ids,
            &output,
            &params,
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("weight"), "{error:#}");
    }
}
