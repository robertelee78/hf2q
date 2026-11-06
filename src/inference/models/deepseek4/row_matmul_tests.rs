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

fn f16_weight(device: &MlxDevice, n: usize, k: usize, seed: usize) -> (MlxBuffer, Vec<f32>) {
    let values: Vec<half::f16> = (0..n * k)
        .map(|index| half::f16::from_f32((((index + seed) * 53 % 199) as f32 - 99.0) * 0.0041))
        .collect();
    let mut weight = device
        .alloc_buffer(values.len() * 2, DType::F16, vec![n, k])
        .unwrap();
    weight.as_mut_slice::<u16>().unwrap().copy_from_slice(
        &values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
    );
    (weight, values.iter().map(|value| value.to_f32()).collect())
}

fn encode_one_row(
    ctx: &mut GpuContext,
    input: &MlxBuffer,
    weight: &RawMatrixRef<'_>,
    output: &MlxBuffer,
    n: usize,
    k: usize,
) {
    let device = ctx.device().clone();
    let (executor, registry) = ctx.split();
    let mut session = executor.begin().unwrap();
    raw_matmul(
        &mut session,
        registry,
        &device,
        input,
        weight,
        output,
        1,
        n,
        k,
        "F16 matvec",
    )
    .unwrap();
    session.finish().unwrap();
}

/// L9 (ADR-063 Q4-DS4-RACE): a one-row F16 product is the F32-input matvec,
/// within the design §3.7 matvec bound `2.5e-4 · Σ|wᵢxᵢ|` of an F64 dot, at
/// ds4's router (256 × 4096) and HC-function (24 × 4096) shapes.
#[test]
fn dense_f16_one_row_matvec_matches_f64_dot() {
    const K: usize = 4096;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    for n in [256_usize, 24] {
        let (weight, values) = f16_weight(&device, n, K, n);
        let shape = [n, K];
        let weight = RawMatrixRef {
            buffer: &weight,
            ggml_type: GgmlType::F16,
            shape: &shape,
        };
        let input = f32_input(&device, vec![1, K], 7);
        let output = alloc(&device, DType::F32, vec![1, n], "F16 matvec output").unwrap();
        encode_one_row(&mut ctx, &input, &weight, &output, n, K);
        let x = input.as_slice::<f32>().unwrap();
        let got = output.as_slice::<f32>().unwrap();
        for row in 0..n {
            let w = &values[row * K..(row + 1) * K];
            let want: f64 = w.iter().zip(x).map(|(w, x)| *w as f64 * *x as f64).sum();
            let magnitude: f64 = w
                .iter()
                .zip(x)
                .map(|(w, x)| (*w as f64 * *x as f64).abs())
                .sum();
            assert!(magnitude > 0.0);
            assert!(
                (got[row] as f64 - want).abs() <= 2.5e-4 * magnitude,
                "n={n} row {row}: got {} want {want} bound {}",
                got[row],
                2.5e-4 * magnitude
            );
        }
    }
}

/// The one-row F16 route keeps the input in F32: `1 + 2⁻¹²` is not a half
/// value (it rounds to 1.0), and a unit weight must return it exactly.
#[test]
fn dense_f16_one_row_matvec_does_not_round_the_input_to_half() {
    const N: usize = 8;
    const K: usize = 256;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    let one = half::f16::ONE.to_bits();
    let mut weight = device
        .alloc_buffer(N * K * 2, DType::F16, vec![N, K])
        .unwrap();
    weight.as_mut_slice::<u16>().unwrap().fill(one);
    let shape = [N, K];
    let weight = RawMatrixRef {
        buffer: &weight,
        ggml_type: GgmlType::F16,
        shape: &shape,
    };
    let value = 1.0_f32 + 2.0_f32.powi(-12);
    assert_eq!(half::f16::from_f32(value).to_f32(), 1.0);
    let mut input = alloc(&device, DType::F32, vec![1, K], "F16 precision input").unwrap();
    let x = input.as_mut_slice::<f32>().unwrap();
    x.fill(0.0);
    x[17] = value;
    let output = alloc(&device, DType::F32, vec![1, N], "F16 precision output").unwrap();
    encode_one_row(&mut ctx, &input, &weight, &output, N, K);
    for got in output.as_slice::<f32>().unwrap() {
        assert_eq!(got.to_bits(), value.to_bits());
    }
}

#[test]
fn dense_f16_independent_rows_match_one_row_matvec_bitwise() {
    const ROWS: usize = 2;
    const N: usize = 24;
    const K: usize = 4096;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    let (weight, _) = f16_weight(&device, N, K, 3);
    let shape = [N, K];
    let weight = RawMatrixRef {
        buffer: &weight,
        ggml_type: GgmlType::F16,
        shape: &shape,
    };
    let input = f32_input(&device, vec![ROWS, K], 9);
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
        "joint F16 HC function",
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
            "serial F16 HC function",
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
