//! Artifact-native scalar expert dispatch shared by every Gemma execution path.

use std::cell::RefCell;

use anyhow::{Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::{
    DType, DenseMatmulIdInputLayout, DenseMatmulIdMultiplicity, DenseMatmulIdParams,
    DenseMatmulIdRoute, DenseMatmulIdScratch, GgmlType, KernelRegistry, MlxBuffer, MlxDevice,
};

use crate::inference::dense_expert_activation::DenseExpertScratchCache;
use crate::serve::forward_mlx_shared::MlxAffineMoeStack;

thread_local! {
    static DENSE_ID_SCRATCH_GATE_UP: RefCell<DenseExpertScratchCache> = RefCell::new(DenseExpertScratchCache::default());
    static DENSE_ID_SCRATCH_DOWN: RefCell<DenseExpertScratchCache> = RefCell::new(DenseExpertScratchCache::default());
}

/// Report model-bound grouped-expert scratch at a drained worker boundary.
pub(crate) fn idle_runtime_owned_bytes() -> u64 {
    DENSE_ID_SCRATCH_GATE_UP
        .with(|cell| cell.borrow().owned_bytes())
        .saturating_add(DENSE_ID_SCRATCH_DOWN.with(|cell| cell.borrow().owned_bytes()))
}

/// Release model-bound grouped-expert scratch at a drained worker boundary.
pub(crate) fn release_idle_runtime_state() -> u64 {
    DENSE_ID_SCRATCH_GATE_UP
        .with(|cell| cell.borrow_mut().release_owned_bytes())
        .saturating_add(DENSE_ID_SCRATCH_DOWN.with(|cell| cell.borrow_mut().release_owned_bytes()))
}

#[derive(Clone, Copy)]
pub(crate) enum DenseExpertScratchSlot {
    GateUp,
    Down,
}

fn with_scratch<R>(
    slot: DenseExpertScratchSlot,
    activation_epoch: u64,
    device: &MlxDevice,
    n_experts: u32,
    max_tokens: u32,
    f: impl FnOnce(&DenseMatmulIdScratch) -> mlx_native::Result<R>,
) -> mlx_native::Result<R> {
    let cell = match slot {
        DenseExpertScratchSlot::GateUp => &DENSE_ID_SCRATCH_GATE_UP,
        DenseExpertScratchSlot::Down => &DENSE_ID_SCRATCH_DOWN,
    };
    cell.with(|cell| {
        cell.borrow_mut()
            .with(activation_epoch, device, n_experts, max_tokens, f)
    })
}

/// Dispatch an exact affine overlay or native F32/F16/BF16 expert stack,
/// returning `false` only when the artifact's block codec must retain its
/// existing route. Scalar-plan and scalar-buffer validation happen inside the
/// session API before it mutates the encoder; affine dispatch retains its
/// established GraphSession hazard contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_native_scalar_expert(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    activation_epoch: u64,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    expert_ids: &MlxBuffer,
    output: &MlxBuffer,
    affine: Option<&MlxAffineMoeStack>,
    ggml_type: GgmlType,
    source_rows: u32,
    top_k: u32,
    n: u32,
    k: u32,
    n_experts: u32,
    expert_stride_bytes: u64,
    input_layout: DenseMatmulIdInputLayout,
    scratch_slot: DenseExpertScratchSlot,
    label: &str,
) -> Result<bool> {
    if let Some(stack) = affine {
        anyhow::ensure!(
            stack.n == n as usize
                && stack.k == k as usize
                && stack.num_experts == n_experts as usize,
            "{label}: affine shape [{}, {}, {} experts] != expected [{n}, {k}, {n_experts} experts]",
            stack.n,
            stack.k,
            stack.num_experts
        );
        let (m, n_expert_used) = match input_layout {
            DenseMatmulIdInputLayout::SharedPerToken => (source_rows, top_k),
            DenseMatmulIdInputLayout::Slotted => (
                source_rows
                    .checked_mul(top_k)
                    .context("affine slotted row count overflow")?,
                1,
            ),
        };
        session.barrier_between(
            &[
                input,
                &stack.weight,
                &stack.scales,
                &stack.biases,
                expert_ids,
            ],
            &[output],
        );
        mlx_native::quantized_matmul_id_into(
            session.encoder_mut(),
            registry,
            device,
            input,
            &stack.weight,
            &stack.scales,
            &stack.biases,
            expert_ids,
            output,
            &mlx_native::QuantizedMatmulIdParams {
                m,
                k,
                n,
                group_size: stack.group_size,
                bits: stack.bits,
                n_expert_used,
                num_experts: n_experts,
            },
        )
        .with_context(|| format!("encode {label} affine expert"))?;
        return Ok(true);
    }
    if !matches!(ggml_type, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        return Ok(false);
    }
    let expected_dtype = match ggml_type {
        GgmlType::F32 => DType::F32,
        GgmlType::F16 => DType::F16,
        GgmlType::BF16 => DType::BF16,
        _ => unreachable!(),
    };
    anyhow::ensure!(
        weight.dtype() == expected_dtype,
        "{label}: declared {ggml_type:?} but maps as {}",
        weight.dtype()
    );
    let params = DenseMatmulIdParams {
        m: source_rows,
        n,
        k,
        top_k,
        n_experts,
        expert_stride_bytes,
        input_layout,
        id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
        route: DenseMatmulIdRoute::Direct,
    };
    with_scratch(
        scratch_slot,
        activation_epoch,
        device,
        n_experts,
        source_rows,
        |scratch| {
            session
                .dense_matmul_id_auto(
                    registry,
                    device,
                    activation_epoch,
                    weight,
                    input,
                    expert_ids,
                    output,
                    Some(scratch),
                    &params,
                )
                .map(|_| ())
        },
    )
    .with_context(|| format!("encode {label} scalar expert"))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_native::{GgmlQuantizedMatmulIdParams, GraphExecutor};

    #[test]
    fn malformed_affine_expert_stack_fails_before_native_dispatch() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("Metal device");
        let mut registry = KernelRegistry::new();
        let executor = GraphExecutor::new(device.clone());
        let alloc = |elements: usize, dtype: DType| {
            device
                .alloc_buffer(elements * dtype.size_of(), dtype, vec![elements])
                .expect("test buffer")
        };
        let valid_stack = MlxAffineMoeStack {
            weight: alloc(4 * 32 * (32 / 8), DType::U32),
            scales: alloc(4 * 32 * (32 / 32), DType::BF16),
            biases: alloc(4 * 32 * (32 / 32), DType::BF16),
            n: 32,
            k: 32,
            bits: 4,
            group_size: 32,
            num_experts: 4,
        };
        let shared_input = alloc(32, DType::F32);
        let slotted_input = alloc(64, DType::F32);
        let legacy_weight = alloc(4 * 32 * 18, DType::U8);
        let ids = alloc(2, DType::U32);
        let output = alloc(64, DType::F32);
        for (layout_name, input, layout, slot) in [
            (
                "shared",
                &shared_input,
                DenseMatmulIdInputLayout::SharedPerToken,
                DenseExpertScratchSlot::GateUp,
            ),
            (
                "slotted",
                &slotted_input,
                DenseMatmulIdInputLayout::Slotted,
                DenseExpertScratchSlot::Down,
            ),
        ] {
            for (field, stack) in [
                (
                    "n",
                    MlxAffineMoeStack {
                        n: 64,
                        ..valid_stack.clone()
                    },
                ),
                (
                    "k",
                    MlxAffineMoeStack {
                        k: 64,
                        ..valid_stack.clone()
                    },
                ),
                (
                    "num_experts",
                    MlxAffineMoeStack {
                        num_experts: 3,
                        ..valid_stack.clone()
                    },
                ),
            ] {
                let mut session = executor.begin_recorded().expect("recorded graph session");
                let error = dispatch_native_scalar_expert(
                    &mut session,
                    &mut registry,
                    &device,
                    1,
                    input,
                    &legacy_weight,
                    &ids,
                    &output,
                    Some(&stack),
                    GgmlType::Q4_0,
                    1,
                    2,
                    32,
                    32,
                    4,
                    18,
                    layout,
                    slot,
                    "malformed Gemma affine expert",
                )
                .unwrap_err();
                assert!(
                    error.to_string().contains("affine shape"),
                    "{layout_name}/{field}: {error}"
                );
                assert!(
                    session
                        .encoder_mut()
                        .take_capture()
                        .expect("active capture")
                        .is_empty(),
                    "{layout_name}/{field}: malformed overlay mutated the graph"
                );
            }
        }
    }

    #[test]
    fn gemma_q5_0_block_expert_fallback_executes_shared_and_slotted_routes() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("Metal device");
        let executor = GraphExecutor::new(device.clone());
        let (experts, n, k, top_k) = (3usize, 5usize, 32usize, 2usize);
        let source_weight = (0..experts * n * k)
            .map(|index| ((index * 29 % 97) as f32 - 48.0) / 53.0)
            .collect::<Vec<_>>();
        let packed = crate::quantize::ggml_quants::q5_0::quantize(&source_weight, k, None);
        let mut dequantized = vec![0.0_f32; source_weight.len()];
        mlx_native::gguf::test_only_dequantize(&packed, GgmlType::Q5_0, &mut dequantized)
            .expect("dequantize Gemma Q5_0 expert oracle");
        let expert_stride = packed.len() / experts;
        let mut weight = device
            .alloc_buffer(packed.len(), DType::U8, vec![packed.len()])
            .expect("allocate Gemma Q5_0 expert weight");
        weight
            .as_mut_slice::<u8>()
            .expect("map Gemma Q5_0 expert weight")
            .copy_from_slice(&packed);
        let mut ids = device
            .alloc_buffer(top_k * std::mem::size_of::<u32>(), DType::U32, vec![top_k])
            .expect("allocate Gemma expert IDs");
        ids.as_mut_slice::<u32>()
            .expect("map Gemma expert IDs")
            .copy_from_slice(&[0, 2]);

        for (label, layout, input_rows, params_top_k, scratch_slot) in [
            (
                "gate/up shared",
                DenseMatmulIdInputLayout::SharedPerToken,
                1usize,
                top_k,
                DenseExpertScratchSlot::GateUp,
            ),
            (
                "down slotted",
                DenseMatmulIdInputLayout::Slotted,
                top_k,
                1usize,
                DenseExpertScratchSlot::Down,
            ),
        ] {
            let input_values = (0..input_rows * k)
                .map(|index| ((index * 17 % 71) as f32 - 35.0) / 41.0)
                .collect::<Vec<_>>();
            let mut input = device
                .alloc_buffer(
                    input_values.len() * std::mem::size_of::<f32>(),
                    DType::F32,
                    vec![input_rows, k],
                )
                .expect("allocate Gemma Q5_0 expert input");
            input
                .as_mut_slice::<f32>()
                .expect("map Gemma Q5_0 expert input")
                .copy_from_slice(&input_values);
            let mut output = device
                .alloc_buffer(
                    top_k * n * std::mem::size_of::<f32>(),
                    DType::F32,
                    vec![top_k, n],
                )
                .expect("allocate Gemma Q5_0 expert output");
            output
                .as_mut_slice::<f32>()
                .expect("poison Gemma Q5_0 expert output")
                .fill(f32::NAN);

            let mut registry = KernelRegistry::new();
            let mut probe = executor.begin_recorded().expect("begin Gemma route probe");
            let scalar_selected = dispatch_native_scalar_expert(
                &mut probe,
                &mut registry,
                &device,
                7,
                &input,
                &weight,
                &ids,
                &output,
                None,
                GgmlType::Q5_0,
                1,
                top_k as u32,
                n as u32,
                k as u32,
                experts as u32,
                expert_stride as u64,
                layout,
                scratch_slot,
                label,
            )
            .unwrap_or_else(|error| panic!("Gemma Q5_0 {label} route probe: {error}"));
            assert!(!scalar_selected, "Q5_0 must retain Gemma's block route");
            assert!(
                probe
                    .encoder_mut()
                    .take_capture()
                    .expect("Gemma route probe capture")
                    .is_empty(),
                "Q5_0 scalar probe must not mutate the graph"
            );
            drop(probe);

            let mut session = executor.begin().expect("begin Gemma Q5_0 expert fallback");
            session.barrier_between(&[&input, &weight, &ids], &[&output]);
            session
                .quantized_matmul_id_ggml(
                    &mut registry,
                    &device,
                    &input,
                    &weight,
                    &ids,
                    &output,
                    &GgmlQuantizedMatmulIdParams {
                        n_tokens: input_rows as u32,
                        top_k: params_top_k as u32,
                        n: n as u32,
                        k: k as u32,
                        n_experts: experts as u32,
                        expert_stride: expert_stride as u64,
                        ggml_type: GgmlType::Q5_0,
                    },
                )
                .unwrap_or_else(|error| panic!("Gemma Q5_0 {label} dispatch: {error}"));
            session
                .finish()
                .unwrap_or_else(|error| panic!("Gemma Q5_0 {label} finish: {error}"));

            let actual = output
                .as_slice::<f32>()
                .expect("read Gemma Q5_0 expert output");
            assert!(actual.iter().all(|value| value.is_finite()), "{label}");
            assert!(
                actual.iter().any(|value| *value != 0.0),
                "{label} must execute nonzero Q5_0 expert math"
            );
            for (route, expert) in [0usize, 2].into_iter().enumerate() {
                let input_row = if layout == DenseMatmulIdInputLayout::SharedPerToken {
                    0
                } else {
                    route
                };
                for column in 0..n {
                    let expected = (0..k)
                        .map(|inner| {
                            input_values[input_row * k + inner]
                                * dequantized[(expert * n + column) * k + inner]
                        })
                        .sum::<f32>();
                    let got = actual[route * n + column];
                    assert!(
                        (got - expected).abs() <= 1e-2,
                        "Gemma Q5_0 {label} mismatch at route {route}, column {column}: {got} vs {expected}"
                    );
                }
            }
        }
    }
}
