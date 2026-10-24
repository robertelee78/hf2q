//! Explicit synthetic packet; no model weights or production capacity policy.
use super::model::{compressor_tail::CompressorTail, rope::yarn_frequencies, Deepseek4Config};
use anyhow::{ensure, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};
use std::cell::Cell;

pub(super) const TEMP_BYTES: usize = 1_031_040;
pub(super) const CACHE_BYTES: u64 = 45_032_448;
pub(super) const TOTAL_BYTES: u64 = 46_063_488;
const SHAPES: &[(&[usize], DType)] = &[
    (&[1, 64, 1024], DType::F32),
    (&[1, 64, 1024], DType::F32),
    (&[1, 16, 512], DType::BF16),
    (&[1, 16, 1, 512], DType::BF16),
    (&[128, 512], DType::F32),
    (&[512], DType::F32),
    (&[1, 64, 256], DType::F32),
    (&[1, 64, 256], DType::F32),
    (&[1, 16, 128], DType::BF16),
    (&[1, 16, 1, 128], DType::BF16),
    (&[4, 256], DType::F32),
    (&[128], DType::F32),
    (&[64, 512], DType::BF16),
    (&[64], DType::U32),
    (&[32], DType::F32),
];
pub(super) struct Packet {
    pub roots: Vec<MlxBuffer>,
}
thread_local! {static SHAPE_ARGUMENT_BYTES: Cell<usize> = const { Cell::new(0) };}
fn record_shape(shape: &Vec<usize>) -> Result<()> {
    SHAPE_ARGUMENT_BYTES.with(|bytes| {
        let next = bytes.get() + shape.capacity() * std::mem::size_of::<usize>();
        ensure!(next <= 65536, "bounded shape argument inventory exceeded");
        bytes.set(next);
        Ok(())
    })
}
pub(super) fn shape_argument_bytes() -> usize {
    SHAPE_ARGUMENT_BYTES.with(Cell::get)
}
pub(super) struct TailViews {
    values: [MlxBuffer; 6],
}
impl TailViews {
    pub fn borrowed(&self) -> CompressorTail<'_> {
        let [kv, score, output, rope, ape, norm] = &self.values;
        CompressorTail {
            kv,
            score,
            output,
            rope,
            ape,
            norm,
        }
    }
}
fn view(buffer: &MlxBuffer, shape: Vec<usize>) -> Result<MlxBuffer> {
    record_shape(&shape)?;
    let count = shape.iter().product();
    Ok(buffer.slice_view(0, count).with_shape(shape)?)
}
pub(super) fn window_expected(row: usize, col: usize) -> u16 {
    ((((row * 3 + col) % 17) as i32 - 8) as f32)
        .to_bits()
        .wrapping_shr(16) as u16
}
pub(super) fn bf16_words(buffer: &MlxBuffer) -> Result<&[u16]> {
    ensure!(buffer.dtype() == DType::BF16);
    Ok(buffer.as_slice::<u16>()?)
}
pub(super) fn fill_bf16(buffer: &mut MlxBuffer, value: u16) -> Result<()> {
    ensure!(buffer.dtype() == DType::BF16);
    buffer.as_mut_slice::<u16>()?.fill(value);
    Ok(())
}
impl Packet {
    pub fn allocate(device: &MlxDevice, cfg: &Deepseek4Config) -> Result<Self> {
        SHAPE_ARGUMENT_BYTES.with(|bytes| bytes.set(0));
        let mut roots = Vec::with_capacity(15);
        let mut requested = 0;
        for &(shape, dtype) in SHAPES {
            let bytes = shape.iter().product::<usize>() * dtype.size_of();
            requested += bytes;
            ensure!(requested <= TEMP_BYTES && roots.len() < 15);
            let owned_shape = shape.to_vec();
            record_shape(&owned_shape)?;
            let root = device.alloc_buffer(bytes, dtype, owned_shape)?;
            ensure!(root.byte_len() == bytes && root.data_byte_len() == bytes);
            roots.push(root);
        }
        ensure!(requested == TEMP_BYTES && roots.len() == 15);
        // Pure HF builder, once only; never the distinct native keyed cache.
        let frequencies = yarn_frequencies(
            cfg.rope_head_dim as usize,
            cfg.original_context_length as usize,
            cfg.compress_rope_theta,
            cfg.rope_factor,
            cfg.yarn_beta_fast,
            cfg.yarn_beta_slow,
        )?;
        ensure!(frequencies.len() == 32 && frequencies.capacity() * 4 <= 256);
        roots[14]
            .as_mut_slice::<f32>()?
            .copy_from_slice(&frequencies);
        Ok(Self { roots })
    }
    pub fn prepare(&mut self) -> Result<()> {
        for &index in &[0usize, 1, 4, 5, 6, 7, 10, 11] {
            let values = self.roots[index].as_mut_slice::<f32>()?;
            for (i, value) in values.iter_mut().enumerate() {
                *value = match index {
                    5 | 11 => 1.0 + (i % 7) as f32 / 16.0,
                    1 | 7 => ((i % 19) as i32 - 9) as f32 / 32.0,
                    4 | 10 => ((i % 11) as i32 - 5) as f32 / 64.0,
                    _ => ((i % 29) as i32 - 14) as f32 / 8.0,
                };
            }
        }
        for &index in &[2usize, 3, 8, 9] {
            fill_bf16(&mut self.roots[index], 0x4480)?;
        }
        for (i, value) in self.roots[12].as_mut_slice::<u16>()?.iter_mut().enumerate() {
            *value = window_expected(i / 512, i % 512);
        }
        for (i, value) in self.roots[13].as_mut_slice::<u32>()?.iter_mut().enumerate() {
            *value = (i * 4) as u32;
        }
        Ok(())
    }
    pub fn main(&self, ratio: u32) -> Result<TailViews> {
        let projected = if ratio == 4 { 1024 } else { 512 };
        let count = if ratio == 4 { 16 } else { 1 };
        Ok(TailViews {
            values: [
                view(&self.roots[0], vec![1, 64, projected])?,
                view(&self.roots[1], vec![1, 64, projected])?,
                view(&self.roots[2], vec![1, count, 512])?,
                view(&self.roots[3], vec![1, count, 1, 512])?,
                view(
                    &self.roots[4],
                    if ratio == 4 {
                        vec![4, 1024]
                    } else {
                        vec![128, 512]
                    },
                )?,
                view(&self.roots[5], vec![512])?,
            ],
        })
    }
    pub fn indexer(&self) -> Result<TailViews> {
        Ok(TailViews {
            values: [
                view(&self.roots[6], vec![1, 64, 256])?,
                view(&self.roots[7], vec![1, 64, 256])?,
                view(&self.roots[8], vec![1, 16, 128])?,
                view(&self.roots[9], vec![1, 16, 1, 128])?,
                view(&self.roots[10], vec![4, 256])?,
                view(&self.roots[11], vec![128])?,
            ],
        })
    }
    pub fn positions(&self, count: usize) -> Result<MlxBuffer> {
        view(&self.roots[13], vec![count.max(1)])
    }
}
#[test]
fn explicit_packet_has_fifteen_roots_and_exact_requested_payload() {
    assert_eq!(SHAPES.len(), 15);
    assert_eq!(
        SHAPES
            .iter()
            .map(|(s, d)| s.iter().product::<usize>() * d.size_of())
            .sum::<usize>(),
        TEMP_BYTES
    );
    assert_eq!(CACHE_BYTES + TEMP_BYTES as u64, TOTAL_BYTES);
    assert_ne!(window_expected(0, 0), 0x4480);
    assert_eq!(window_expected(0, 0), 0xc100); // exact BF16(-8), independent of GPU
}
