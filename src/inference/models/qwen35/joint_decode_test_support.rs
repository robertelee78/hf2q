//! Synthetic GGML weights and inputs for the joint decode Metal tests.

use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, MlxBuffer, MlxDevice};

use super::gpu_full_attn::download_f32;

const F16_SCALE: [u8; 2] = 0x211F_u16.to_le_bytes(); // ≈ 0.01

pub(crate) fn next(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    *seed
}

pub(crate) fn random_f32(seed: &mut u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| (next(seed) as i32 as f32 / i32::MAX as f32) * 0.5)
        .collect()
}

/// Valid GGML blocks with finite scales and random quants.
pub(crate) fn synthetic_blocks(
    device: &MlxDevice,
    ggml_type: GgmlType,
    rows: u32,
    cols: u32,
    seed: &mut u32,
) -> MlxBuffer {
    let block_bytes = ggml_type.block_bytes() as usize;
    let blocks = (rows * cols / ggml_type.block_values()) as usize;
    let mut bytes = (0..blocks * block_bytes)
        .map(|_| (next(seed) >> 24) as u8)
        .collect::<Vec<_>>();
    let scale_offsets: &[usize] = match ggml_type {
        GgmlType::Q4_0 | GgmlType::Q8_0 => &[0],
        GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
        GgmlType::Q6_K => &[208],
        other => panic!("no synthetic layout for {other:?}"),
    };
    for block in bytes.chunks_exact_mut(block_bytes) {
        for &offset in scale_offsets {
            block[offset..offset + 2].copy_from_slice(&F16_SCALE);
        }
    }
    let mut buffer = device
        .alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()])
        .expect("allocate synthetic GGML blocks");
    buffer
        .as_mut_slice::<u8>()
        .expect("map synthetic GGML blocks")
        .copy_from_slice(&bytes);
    buffer
}

pub(crate) fn bits(buffer: &MlxBuffer, count: usize) -> Vec<u32> {
    download_f32(buffer).expect("download")[..count]
        .iter()
        .map(|value| value.to_bits())
        .collect()
}
