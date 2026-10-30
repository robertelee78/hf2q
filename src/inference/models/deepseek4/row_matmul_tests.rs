//! Per-row exactness of the joint-decode matrix products on Metal.
//!
//! Joint decode is admissible only if every `m = rows` product it issues is
//! bitwise equal to the `m = 1` product the one-token graph issues for that
//! row. These tests pin that contract for the storage types DeepSeek-V4
//! serving uses (Q8_0 attention/head, dense F32 router) and for the grouped
//! output-A projection.

use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, MlxBuffer, MlxDevice};

use super::forward_support::{
    alloc, grouped_output_a, grouped_output_a_batched, raw_matmul, BatchedGroupedOutputArena,
};
use super::residency::RawMatrixRef;
use super::row_matmul::{row_matmul, row_view, splits_independent_rows, RowMatmul};
use crate::serve::gpu::GpuContext;

const Q8_0_BLOCK_VALUES: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;

fn q8_0_weight(device: &MlxDevice, n: usize, k: usize) -> MlxBuffer {
    let blocks = k / Q8_0_BLOCK_VALUES;
    let mut bytes = vec![0_u8; n * blocks * Q8_0_BLOCK_BYTES];
    for row in 0..n {
        for block in 0..blocks {
            let offset = (row * blocks + block) * Q8_0_BLOCK_BYTES;
            // Positive normal F16 scales with varied mantissas.
            let scale = 0x2000_u16 + ((row * 31 + block * 7) % 997) as u16;
            bytes[offset..offset + 2].copy_from_slice(&scale.to_le_bytes());
            for column in 0..Q8_0_BLOCK_VALUES {
                bytes[offset + 2 + column] =
                    (((row * 13 + block * 5 + column * 3) % 255) as i16 - 127) as i8 as u8;
            }
        }
    }
    let mut weight = device
        .alloc_buffer(bytes.len(), DType::U8, vec![n, k])
        .unwrap();
    weight.as_mut_slice::<u8>().unwrap().copy_from_slice(&bytes);
    weight
}

fn f32_input(device: &MlxDevice, shape: Vec<usize>, seed: usize) -> MlxBuffer {
    let mut buffer = alloc(device, DType::F32, shape, "row test input").unwrap();
    for (index, value) in buffer.as_mut_slice::<f32>().unwrap().iter_mut().enumerate() {
        *value = (((index + seed) * 37 % 101) as f32 - 50.0) * 0.0173;
    }
    buffer
}

fn bits(buffer: &MlxBuffer) -> Vec<u32> {
    buffer
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

#[test]
fn only_dense_float_storage_splits_independent_rows() {
    for ggml_type in [GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
        assert!(splits_independent_rows(ggml_type, 2));
        assert!(!splits_independent_rows(ggml_type, 1));
    }
    for ggml_type in [GgmlType::Q8_0, GgmlType::Q2_K, GgmlType::Q3_K] {
        assert!(!splits_independent_rows(ggml_type, 4));
    }
}

#[test]
fn row_views_are_bounded_and_offset() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    let buffer = f32_input(&device, vec![3, 8], 0);
    let view = row_view(&buffer, 2, vec![1, 8]).unwrap();
    assert_eq!(view.shape(), [1, 8]);
    assert_eq!(
        view.as_slice::<f32>().unwrap(),
        &buffer.as_slice::<f32>().unwrap()[16..24]
    );
    assert!(row_view(&buffer, 3, vec![1, 8]).is_err());
}

#[test]
fn q8_0_independent_rows_match_one_row_matvec_bitwise() {
    const ROWS: usize = 4;
    const N: usize = 63;
    const K: usize = 512;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    let weight = q8_0_weight(&device, N, K);
    let shape = [N, K];
    let weight = RawMatrixRef {
        buffer: &weight,
        ggml_type: GgmlType::Q8_0,
        shape: &shape,
    };
    let input = f32_input(&device, vec![ROWS, K], 3);
    let joint = alloc(&device, DType::F32, vec![ROWS, N], "joint output").unwrap();
    let serial = alloc(&device, DType::F32, vec![ROWS, N], "serial output").unwrap();
    let (executor, registry) = ctx.split();
    let mut session = executor.begin().unwrap();
    row_matmul(
        RowMatmul::IndependentRows,
        &mut session,
        registry,
        &device,
        &input,
        &weight,
        &joint,
        ROWS,
        N,
        K,
        "joint Q8_0",
    )
    .unwrap();
    for row in 0..ROWS {
        raw_matmul(
            &mut session,
            registry,
            &device,
            &row_view(&input, row, vec![1, K]).unwrap(),
            &weight,
            &row_view(&serial, row, vec![1, N]).unwrap(),
            1,
            N,
            K,
            "serial Q8_0",
        )
        .unwrap();
    }
    session.finish().unwrap();
    assert!(serial
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .any(|value| *value != 0.0));
    assert_eq!(bits(&joint), bits(&serial));
}

#[test]
fn dense_f32_independent_rows_match_one_row_matvec_bitwise() {
    const ROWS: usize = 3;
    const N: usize = 40;
    const K: usize = 64;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    let weight = f32_input(&device, vec![N, K], 11);
    let shape = [N, K];
    let weight = RawMatrixRef {
        buffer: &weight,
        ggml_type: GgmlType::F32,
        shape: &shape,
    };
    let input = f32_input(&device, vec![ROWS, K], 5);
    let joint = alloc(&device, DType::F32, vec![ROWS, N], "joint output").unwrap();
    let serial = alloc(&device, DType::F32, vec![ROWS, N], "serial output").unwrap();
    let (executor, registry) = ctx.split();
    let mut session = executor.begin().unwrap();
    row_matmul(
        RowMatmul::IndependentRows,
        &mut session,
        registry,
        &device,
        &input,
        &weight,
        &joint,
        ROWS,
        N,
        K,
        "joint router",
    )
    .unwrap();
    for row in 0..ROWS {
        raw_matmul(
            &mut session,
            registry,
            &device,
            &row_view(&input, row, vec![1, K]).unwrap(),
            &weight,
            &row_view(&serial, row, vec![1, N]).unwrap(),
            1,
            N,
            K,
            "serial router",
        )
        .unwrap();
    }
    session.finish().unwrap();
    assert_eq!(bits(&joint), bits(&serial));
}

#[test]
fn q8_0_grouped_output_rows_match_one_row_groups_bitwise() {
    const ROWS: usize = 4;
    const GROUPS: usize = 4;
    const RANK: usize = 32;
    const HEADS: usize = 4;
    const HEAD_DIM: usize = 64;
    const GROUP_WIDTH: usize = HEADS * HEAD_DIM / GROUPS;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    let weight = q8_0_weight(&device, GROUPS * RANK, GROUP_WIDTH);
    let shape = [GROUPS * RANK, GROUP_WIDTH];
    let weight = RawMatrixRef {
        buffer: &weight,
        ggml_type: GgmlType::Q8_0,
        shape: &shape,
    };
    let input = f32_input(&device, vec![ROWS, HEADS, HEAD_DIM], 7);
    let joint = alloc(
        &device,
        DType::F32,
        vec![ROWS, GROUPS * RANK],
        "joint output",
    )
    .unwrap();
    let serial = alloc(
        &device,
        DType::F32,
        vec![ROWS, GROUPS * RANK],
        "serial output",
    )
    .unwrap();
    let arena = BatchedGroupedOutputArena::new(&device, ROWS, GROUPS, GROUP_WIDTH, RANK).unwrap();
    let (executor, registry) = ctx.split();
    let mut session = executor.begin().unwrap();
    grouped_output_a_batched(
        &mut session,
        registry,
        &device,
        &input,
        &weight,
        &joint,
        &arena,
        ROWS,
        GROUPS,
        RANK,
        HEADS,
        HEAD_DIM,
    )
    .unwrap();
    for row in 0..ROWS {
        grouped_output_a(
            &mut session,
            registry,
            &device,
            &row_view(&input, row, vec![1, HEADS, HEAD_DIM]).unwrap(),
            &weight,
            &row_view(&serial, row, vec![1, GROUPS * RANK]).unwrap(),
            GROUPS,
            RANK,
            HEADS,
            HEAD_DIM,
        )
        .unwrap();
    }
    session.finish().unwrap();
    assert!(serial
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .any(|value| *value != 0.0));
    assert_eq!(bits(&joint), bits(&serial));
}
