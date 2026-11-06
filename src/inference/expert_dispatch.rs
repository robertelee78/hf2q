//! Storage-aware native expert-ID dispatch.
//!
//! GGUF scalar stacks use mlx-native's native F32/F16/BF16 primitive while
//! block-quantized stacks retain the established GGML routes.  The plain
//! entry points keep the flattened `(n_tokens, top_k)` contract (one shared
//! activation row per token); the `_slotted` entry point reads one
//! activation row per `(token, slot)` pair.

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

#[allow(clippy::too_many_arguments)]
fn dispatch_scalar(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    expert_ids: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulIdParams,
    input_layout: DenseMatmulIdInputLayout,
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
        input_layout,
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
        session,
        registry,
        device,
        input,
        weight,
        expert_ids,
        output,
        params,
        DenseMatmulIdInputLayout::SharedPerToken,
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
        session,
        registry,
        device,
        input,
        weight,
        expert_ids,
        output,
        params,
        DenseMatmulIdInputLayout::SharedPerToken,
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
        session,
        registry,
        device,
        input,
        weight,
        expert_ids,
        output,
        params,
        DenseMatmulIdInputLayout::SharedPerToken,
    )? {
        return Ok(());
    }
    session
        .quantized_matmul_id_ggml_pooled(
            registry, device, input, weight, expert_ids, output, scratch, params,
        )
        .context("pooled block-quantized expert-ID matmul")
}

/// Slotted expert-ID matmul: `input` is `[n_tokens, top_k, K]`, one row per
/// `(token, slot)`, and the output is `[n_tokens, top_k, N]`. Bitwise equal
/// to the flattened call over `n_tokens·top_k` rows with `top_k = 1`
/// (ADR-063 Q4-GEMMA-PREFILL-MOE §3.2). Block-quantized stacks require the
/// `mm_id` route (`n_tokens > MM_ID_ROUTING_THRESHOLD`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_expert_matmul_id_pooled_slotted(
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
        session,
        registry,
        device,
        input,
        weight,
        expert_ids,
        output,
        params,
        DenseMatmulIdInputLayout::Slotted,
    )? {
        return Ok(());
    }
    session
        .quantized_matmul_id_ggml_pooled_slotted(
            registry, device, input, weight, expert_ids, output, scratch, params,
        )
        .context("pooled slotted block-quantized expert-ID matmul")
}

/// Slotted expert-ID matmul that reuses the routing schedule the previous
/// pooled `mm_id` call (same ids, `n_tokens`, `top_k`, `n_experts`, encoder)
/// left in `scratch` (ADR-063 Q4-GEMMA-PREFILL-MOE P2b). Block-quantized
/// stacks only: a scalar stack never builds a schedule. mlx-native refuses a
/// mismatched key before encoding; the error propagates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_expert_matmul_id_pooled_slotted_reusing_schedule(
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
    anyhow::ensure!(
        scalar_dtype(params.ggml_type).is_none(),
        "scalar {:?} expert stacks build no reusable mm_id schedule",
        params.ggml_type
    );
    session
        .quantized_matmul_id_ggml_pooled_slotted_reusing_schedule(
            registry, device, input, weight, expert_ids, output, scratch, params,
        )
        .context("pooled slotted block-quantized expert-ID matmul (reused schedule)")
}

/// True for block-quantized expert stacks (the GGML `mm_id` route, which
/// writes the pooled schedule), false for scalar GGUF stacks.
pub(crate) fn writes_pooled_schedule(ggml_type: GgmlType) -> bool {
    scalar_dtype(ggml_type).is_none()
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

    /// ADR-063 Q4-GEMMA-PREFILL-MOE §5.2: for every scalar expert storage
    /// type, the slotted down call is bitwise equal to the flattened call
    /// (`n_tokens·top_k` rows, `top_k = 1`) the Gemma prefill used before.
    #[test]
    fn scalar_slotted_down_is_bitwise_equal_to_flattened_rows() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Ok(device) = MlxDevice::new() else {
            eprintln!("skipping scalar slotted down parity test: no Metal device");
            return;
        };
        const K: usize = 96;
        const N: usize = 80;
        const EXPERTS: usize = 16;
        const TOP_K: usize = 8;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let weights_f32: Vec<f32> = (0..EXPERTS * N * K)
            .map(|_| ((next() % 2001) as f32 - 1000.0) / 997.0)
            .collect();
        let executor = GraphExecutor::new(device.clone());
        let mut registry = KernelRegistry::new();
        for (ggml_type, element_bytes) in
            [(GgmlType::F32, 4usize), (GgmlType::F16, 2), (GgmlType::BF16, 2)]
        {
            let weights = match ggml_type {
                GgmlType::F32 => f32_buffer(&device, &weights_f32),
                GgmlType::BF16 => bf16_buffer(&device, &weights_f32),
                _ => {
                    let mut buffer = device
                        .alloc_buffer(weights_f32.len() * 2, DType::F16, vec![weights_f32.len()])
                        .unwrap();
                    for (destination, value) in buffer
                        .as_mut_slice::<u16>()
                        .unwrap()
                        .iter_mut()
                        .zip(&weights_f32)
                    {
                        *destination = half::f16::from_f32(*value).to_bits();
                    }
                    buffer
                }
            };
            for seq_len in [33usize, 232] {
                let rows = seq_len * TOP_K;
                let input_values: Vec<f32> = (0..rows * K)
                    .map(|_| ((next() % 4001) as f32 - 2000.0) / 1999.0)
                    .collect();
                let input = f32_buffer(&device, &input_values);
                // Distinct ids per token, skewed toward low experts.
                let mut id_values = Vec::with_capacity(rows);
                for _ in 0..seq_len {
                    let mut chosen: Vec<u32> = Vec::with_capacity(TOP_K);
                    while chosen.len() < TOP_K {
                        let id = ((next() % 64) as u32).min((next() % EXPERTS as u64) as u32);
                        if !chosen.contains(&id) {
                            chosen.push(id);
                        }
                    }
                    id_values.extend(chosen);
                }
                let mut ids = device
                    .alloc_buffer(rows * 4, DType::U32, vec![seq_len, TOP_K])
                    .unwrap();
                ids.as_mut_slice::<u32>().unwrap().copy_from_slice(&id_values);
                let mut scratch =
                    mlx_native::IdMmScratch::alloc(&device, EXPERTS as u32, rows as u32).unwrap();
                let mut run = |slotted: bool| {
                    let output = device
                        .alloc_buffer(rows * N * 4, DType::F32, vec![rows, N])
                        .unwrap();
                    let params = GgmlQuantizedMatmulIdParams {
                        n_tokens: if slotted { seq_len } else { rows } as u32,
                        top_k: if slotted { TOP_K } else { 1 } as u32,
                        n: N as u32,
                        k: K as u32,
                        n_experts: EXPERTS as u32,
                        expert_stride: (N * K * element_bytes) as u64,
                        ggml_type,
                    };
                    let mut session = executor.begin().unwrap();
                    let dispatch = if slotted {
                        dispatch_expert_matmul_id_pooled_slotted
                    } else {
                        dispatch_expert_matmul_id_pooled
                    };
                    dispatch(
                        &mut session,
                        &mut registry,
                        &device,
                        &input,
                        &weights,
                        &ids,
                        &output,
                        &mut scratch,
                        &params,
                    )
                    .unwrap();
                    session.finish().unwrap();
                    output.as_slice::<u32>().unwrap().to_vec()
                };
                let flat = run(false);
                let slotted = run(true);
                assert!(
                    flat.iter().all(|bits| f32::from_bits(*bits).is_finite()),
                    "{ggml_type:?} seq_len {seq_len}: flattened output finite"
                );
                let first = flat.iter().zip(&slotted).position(|(a, b)| a != b);
                assert!(
                    first.is_none(),
                    "{ggml_type:?} seq_len {seq_len}: slotted differs from flattened at {first:?}"
                );
            }
        }
    }

    /// ADR-063 Q4-GEMMA-PREFILL-MOE §5.2 P2b: the down call reusing the
    /// gate/up schedule equals the self-scheduling slotted call bitwise, and
    /// a reuse without a matching schedule (fresh scratch, other ids, another
    /// session) is refused instead of reading a stale schedule.
    #[test]
    fn q8_0_slotted_down_reusing_gate_up_schedule_is_bitwise_and_keyed() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Ok(device) = MlxDevice::new() else {
            eprintln!("skipping schedule reuse test: no Metal device");
            return;
        };
        const K: usize = 64;
        const N: usize = 128;
        const EXPERTS: usize = 16;
        const TOP_K: usize = 8;
        const SEQ: usize = 40;
        const BLOCK_BYTES: usize = 34;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let q8_0 = |rows: usize, next: &mut dyn FnMut() -> u64| {
            let mut bytes = Vec::with_capacity(rows * K / 32 * BLOCK_BYTES);
            for _ in 0..rows * K / 32 {
                let scale = half::f16::from_f32(((next() % 64) as f32 + 1.0) / 4096.0);
                bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
                bytes.extend((0..32).map(|_| (next() % 255) as u8));
            }
            let mut buffer = device.alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()]).unwrap();
            buffer.as_mut_slice::<u8>().unwrap().copy_from_slice(&bytes);
            buffer
        };
        let gate_up_weight = q8_0(EXPERTS * N, &mut next);
        let down_weight = q8_0(EXPERTS * N, &mut next);
        let stride = (N * K / 32 * BLOCK_BYTES) as u64;
        let values = |len: usize, next: &mut dyn FnMut() -> u64| -> Vec<f32> {
            (0..len)
                .map(|_| ((next() % 2001) as f32 - 1000.0) / 1000.0)
                .collect()
        };
        let shared_input = f32_buffer(&device, &values(SEQ * K, &mut next));
        let slotted_input = f32_buffer(&device, &values(SEQ * TOP_K * K, &mut next));
        let ids_buffer = |next: &mut dyn FnMut() -> u64| {
            let mut ids = Vec::with_capacity(SEQ * TOP_K);
            for _ in 0..SEQ {
                let mut chosen: Vec<u32> = Vec::new();
                while chosen.len() < TOP_K {
                    let id = (next() % EXPERTS as u64) as u32;
                    if !chosen.contains(&id) {
                        chosen.push(id);
                    }
                }
                ids.extend(chosen);
            }
            let mut buffer = device
                .alloc_buffer(ids.len() * 4, DType::U32, vec![SEQ, TOP_K])
                .unwrap();
            buffer.as_mut_slice::<u32>().unwrap().copy_from_slice(&ids);
            buffer
        };
        let ids = ids_buffer(&mut next);
        let other_ids = ids_buffer(&mut next);
        let params = GgmlQuantizedMatmulIdParams {
            n_tokens: SEQ as u32,
            top_k: TOP_K as u32,
            n: N as u32,
            k: K as u32,
            n_experts: EXPERTS as u32,
            expert_stride: stride,
            ggml_type: GgmlType::Q8_0,
        };
        let output = || {
            device
                .alloc_buffer(SEQ * TOP_K * N * 4, DType::F32, vec![SEQ * TOP_K, N])
                .unwrap()
        };
        let executor = GraphExecutor::new(device.clone());
        let mut registry = KernelRegistry::new();
        let mut scratch =
            mlx_native::IdMmScratch::alloc(&device, EXPERTS as u32, (SEQ * TOP_K) as u32).unwrap();
        let gate_up = |session: &mut GraphSession<'_>,
                       registry: &mut KernelRegistry,
                       scratch: &mut mlx_native::IdMmScratch,
                       ids: &MlxBuffer| {
            dispatch_expert_matmul_id_pooled(
                session,
                registry,
                &device,
                &shared_input,
                &gate_up_weight,
                ids,
                &output(),
                scratch,
                &params,
            )
            .unwrap();
        };

        // Reference: the self-scheduling slotted call.
        let fresh = output();
        let mut session = executor.begin().unwrap();
        dispatch_expert_matmul_id_pooled_slotted(
            &mut session,
            &mut registry,
            &device,
            &slotted_input,
            &down_weight,
            &ids,
            &fresh,
            &mut scratch,
            &params,
        )
        .unwrap();
        session.finish().unwrap();

        // gate/up then the reusing down call, as in the Gemma body.
        let reused = output();
        let mut session = executor.begin().unwrap();
        gate_up(&mut session, &mut registry, &mut scratch, &ids);
        dispatch_expert_matmul_id_pooled_slotted_reusing_schedule(
            &mut session,
            &mut registry,
            &device,
            &slotted_input,
            &down_weight,
            &ids,
            &reused,
            &mut scratch,
            &params,
        )
        .unwrap();
        session.finish().unwrap();
        let fresh_bits = fresh.as_slice::<u32>().unwrap();
        assert!(fresh_bits.iter().all(|bits| f32::from_bits(*bits).is_finite()));
        assert!(
            fresh_bits == reused.as_slice::<u32>().unwrap(),
            "reused schedule differs from a fresh schedule"
        );

        let refused = |session: &mut GraphSession<'_>,
                       registry: &mut KernelRegistry,
                       scratch: &mut mlx_native::IdMmScratch,
                       ids: &MlxBuffer| {
            dispatch_expert_matmul_id_pooled_slotted_reusing_schedule(
                session,
                registry,
                &device,
                &slotted_input,
                &down_weight,
                ids,
                &output(),
                scratch,
                &params,
            )
            .is_err()
        };
        // A schedule built for other ids.
        let mut session = executor.begin().unwrap();
        gate_up(&mut session, &mut registry, &mut scratch, &other_ids);
        assert!(refused(&mut session, &mut registry, &mut scratch, &ids));
        session.finish().unwrap();
        // A schedule built in another session (encoder).
        let mut session = executor.begin().unwrap();
        gate_up(&mut session, &mut registry, &mut scratch, &ids);
        session.finish().unwrap();
        let mut session = executor.begin().unwrap();
        assert!(refused(&mut session, &mut registry, &mut scratch, &ids));
        session.finish().unwrap();
        // A fresh scratch holds no schedule.
        let mut empty =
            mlx_native::IdMmScratch::alloc(&device, EXPERTS as u32, (SEQ * TOP_K) as u32).unwrap();
        let mut session = executor.begin().unwrap();
        assert!(refused(&mut session, &mut registry, &mut empty, &ids));
        session.finish().unwrap();
    }
}
