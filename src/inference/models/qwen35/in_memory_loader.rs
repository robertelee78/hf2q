//! In-memory `Qwen35Model` loader (ADR-012 item-1 architectural fix).
//!
//! # Why this exists
//!
//! Peer-aligned convert pipeline: build the inference-time `Qwen35Model`
//! directly from the convert-time `TensorMap`, with **no intermediate GGUF
//! file** on disk. mlx-lm, AutoAWQ, and AutoGPTQ all keep tensors in memory
//! throughout convert; llama.cpp keeps F16 GGUF as a user artifact, not
//! throwaway. Our prior pipeline emitted a 33–66 GB intermediate, reloaded
//! it, then dropped it — a ~5-minute I/O tax and a peak-memory hazard.
//!
//! # Inputs
//!
//! * `tensor_map` — post-Phase-1.x transforms (language_model strip, fused
//!   gate_up split, MoE expert merge, RMS+1, linear-attn renames). Names
//!   are HF-derived (e.g. `model.layers.N.self_attn.q_proj.weight`,
//!   `model.layers.N.mlp.experts.gate_proj.weight`).
//! * `metadata` — drives the GGUF metadata side, also supplies `arch` and
//!   layer count.
//! * `device` — Metal device for buffer allocation.
//!
//! # MoE expert quantization
//!
//! The MoE expert tensors are the only weights that are quantized at load
//! time. They land at Q8_0 (32-element ggml blocks, fp16 scale per block,
//! int8 quants), giving the same on-GPU layout that `load_moe_ffn_quantized`
//! consumes when reading from disk. Q8_0 is near-lossless (~99.99% of F16
//! quality at 2× compression) and matches the production MoeQ path —
//! `quantized_matmul_ggml` dispatches Q8_0 blocks directly. No F32 expansion.
//!
//! # Out of scope
//!
//! This module does NOT translate names. It expects the caller to apply
//! `hf_name_to_gguf` first — see `src/backends/gguf.rs` for the canonical
//! mapping. Reusing the existing translation keeps the loader trivially
//! simple and avoids a second source of truth.

use anyhow::{anyhow, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

/// QK8_0 — the ggml Q8_0 block size (32 elements per block).
/// Mirrors `BLOCK_Q8_0_BYTES` in `src/backends/gguf.rs:848` (2 + 32 = 34
/// bytes per block; fp16 scale + 32 int8 quants).
const QK8_0: usize = 32;
const BLOCK_Q8_0_BYTES: usize = 2 + QK8_0;

/// Quantize an F32 weight slice to Q8_0 ggml blocks and upload as a U8
/// MlxBuffer with the supplied `shape`. The buffer layout matches what
/// `weight_loader::load_moe_ffn_quantized` produces when reading Q8_0
/// blocks from a GGUF (DType::U8 with `byte_len = num_blocks *
/// BLOCK_Q8_0_BYTES`), so the existing `quantized_matmul_ggml` Metal
/// kernels consume it without modification.
///
/// # Quantization math (matches ggml.c::quantize_row_q8_0)
///
/// For each 32-element block:
///   * `absmax = max(|x|)` over the block
///   * `d = absmax / 127`
///   * `q[i] = round(x[i] / d).clamp(-128, 127)` as i8
///   * `scale_fp16 = f16(d)`
///
/// Block bytes: `[scale_fp16 (2 bytes), q[0..32] (32 bytes)] = 34 bytes`.
///
/// # Errors
///
/// * `numel` must be divisible by 32 (Q8_0 block alignment). MoE expert
///   K-dims (2048 / 4096) and Qwen3.5/3.6 hidden sizes (5120) all
///   satisfy this.
/// * Buffer allocation must succeed.
pub fn quantize_f32_to_q8_0_buffer(
    f32_data: &[f32],
    shape: Vec<usize>,
    device: &MlxDevice,
) -> Result<MlxBuffer> {
    let numel = f32_data.len();
    if numel == 0 {
        return Err(anyhow!("quantize_f32_to_q8_0_buffer: empty input"));
    }
    if numel % QK8_0 != 0 {
        return Err(anyhow!(
            "quantize_f32_to_q8_0_buffer: numel={} not divisible by Q8_0 block size {}",
            numel,
            QK8_0
        ));
    }
    let shape_numel: usize = shape.iter().product();
    if shape_numel != numel {
        return Err(anyhow!(
            "quantize_f32_to_q8_0_buffer: shape product {} != numel {}",
            shape_numel,
            numel
        ));
    }

    let num_blocks = numel / QK8_0;
    let total_bytes = num_blocks * BLOCK_Q8_0_BYTES;

    // Allocate the device buffer first so we can write directly into it
    // (no intermediate Vec<u8> copy). MlxBuffer with shared storage maps
    // to host-addressable Metal memory on Apple Silicon.
    let mut buf = device
        .alloc_buffer(total_bytes, DType::U8, shape)
        .map_err(|e| anyhow!("quantize_f32_to_q8_0_buffer: alloc_buffer: {e}"))?;
    // W-5b.7 iter 2: register the Q8_0 weight buffer with the weight pool's
    // residency set so it joins MTLResidencySet for cold-page-fault avoidance
    // on the first forward pass.  No-op when HF2Q_NO_RESIDENCY=1.
    super::weight_pool::register_weight_buffer(device, &buf)
        .map_err(|e| anyhow!("quantize_f32_to_q8_0_buffer: register_weight_buffer: {e}"))?;

    {
        let dst: &mut [u8] = buf
            .as_mut_slice()
            .map_err(|e| anyhow!("quantize_f32_to_q8_0_buffer: as_mut_slice: {e}"))?;

        for block_idx in 0..num_blocks {
            let block = &f32_data[block_idx * QK8_0..(block_idx + 1) * QK8_0];
            let absmax = block.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
            let d = absmax / 127.0;
            let inv_d = if d == 0.0 { 0.0f32 } else { 1.0 / d };

            let block_off = block_idx * BLOCK_Q8_0_BYTES;
            // Scale (fp16, 2 bytes).
            let scale_f16 = half::f16::from_f32(d);
            let scale_bytes = scale_f16.to_le_bytes();
            dst[block_off] = scale_bytes[0];
            dst[block_off + 1] = scale_bytes[1];
            // Quants (int8, 32 bytes).
            for (i, &val) in block.iter().enumerate() {
                let q = (val * inv_d).round().clamp(-128.0, 127.0) as i32;
                dst[block_off + 2 + i] = (q as i8) as u8;
            }
        }
    }

    Ok(buf)
}

/// Convert a BF16 byte slice to F32 in-place into a destination Vec.
/// Allocation-free at the destination (preallocated by caller).
pub fn bf16_bytes_to_f32(src: &[u8], dst: &mut Vec<f32>) {
    debug_assert_eq!(src.len() % 2, 0, "BF16 byte length must be even");
    dst.clear();
    dst.reserve(src.len() / 2);
    for chunk in src.chunks_exact(2) {
        // BF16 is the upper 16 bits of an F32 (8 sign+exp + 7 mantissa);
        // shifting left by 16 reconstructs the F32 bit pattern.
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let f32_bits = (bits as u32) << 16;
        dst.push(f32::from_bits(f32_bits));
    }
}

/// Convert an F16 byte slice to F32 into a destination Vec.
pub fn f16_bytes_to_f32(src: &[u8], dst: &mut Vec<f32>) {
    debug_assert_eq!(src.len() % 2, 0, "F16 byte length must be even");
    dst.clear();
    dst.reserve(src.len() / 2);
    for chunk in src.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        dst.push(half::f16::from_bits(bits).to_f32());
    }
}

/// Convert an F32 byte slice to a Vec<f32> (the canonical case — just a
/// reinterpretation, but writing the bytes back gives a Vec we own).
pub fn f32_bytes_to_f32(src: &[u8], dst: &mut Vec<f32>) {
    debug_assert_eq!(src.len() % 4, 0, "F32 byte length must be divisible by 4");
    dst.clear();
    dst.reserve(src.len() / 4);
    for chunk in src.chunks_exact(4) {
        dst.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_native::MlxDevice;

    #[test]
    fn q8_0_round_trip_zero_input() {
        let device = MlxDevice::new().expect("device");
        let zeros = vec![0.0f32; 32];
        let buf =
            quantize_f32_to_q8_0_buffer(&zeros, vec![32], &device).expect("quantize");
        // 1 block × 34 bytes; check via slice byte length (element_count
        // returns shape product = 32, not byte count).
        let bytes = buf.as_slice::<u8>().expect("slice");
        assert_eq!(bytes.len(), 34);
        // scale=0 → fp16 zero (2 bytes of 0x00)
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[1], 0);
        // All quants are 0
        for i in 0..32 {
            assert_eq!(bytes[2 + i], 0);
        }
    }

    #[test]
    fn q8_0_block_round_trip_within_tolerance() {
        // Q8_0 has ~1/127 relative error per block; verify a uniform input
        // dequantizes back within tolerance.
        let device = MlxDevice::new().expect("device");
        let mut input = Vec::with_capacity(32);
        for i in 0..32 {
            input.push((i as f32) - 15.5);
        }
        let buf =
            quantize_f32_to_q8_0_buffer(&input, vec![32], &device).expect("quantize");
        let bytes = buf.as_slice::<u8>().expect("slice");

        // Decode scale (fp16 → f32)
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();

        // Reconstruct values from int8 + scale, compare to input
        let mut max_err = 0.0f32;
        for i in 0..32 {
            let q = bytes[2 + i] as i8;
            let recon = (q as f32) * scale;
            let err = (recon - input[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        // absmax = 15.5; d = 15.5/127 ≈ 0.122; per-element rounding < 0.5*d ≈ 0.061
        // Plus fp16 scale rounding (relative ~5e-4 of d ≈ 6e-5).
        assert!(max_err < 0.07, "max_err {} >= 0.07", max_err);
    }

    #[test]
    fn q8_0_rejects_unaligned_numel() {
        let device = MlxDevice::new().expect("device");
        let unaligned = vec![1.0f32; 31];
        let res = quantize_f32_to_q8_0_buffer(&unaligned, vec![31], &device);
        assert!(res.is_err());
    }

    #[test]
    fn q8_0_rejects_shape_mismatch() {
        let device = MlxDevice::new().expect("device");
        let v = vec![1.0f32; 32];
        let res = quantize_f32_to_q8_0_buffer(&v, vec![16, 4], &device);
        // 16 * 4 = 64 != 32 → reject
        assert!(res.is_err());
    }

    #[test]
    fn bf16_to_f32_round_trip() {
        // BF16 bit pattern for 1.0 = 0x3F80
        let bytes: Vec<u8> = vec![0x80, 0x3F, 0x00, 0x40]; // 1.0, 2.0
        let mut out = Vec::new();
        bf16_bytes_to_f32(&bytes, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 2.0);
    }

    #[test]
    fn f32_bytes_round_trip() {
        let f32s = [1.5f32, -2.25, 100.0, 0.0];
        let mut bytes = Vec::new();
        for v in &f32s {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut out = Vec::new();
        f32_bytes_to_f32(&bytes, &mut out);
        assert_eq!(out, f32s);
    }
}
