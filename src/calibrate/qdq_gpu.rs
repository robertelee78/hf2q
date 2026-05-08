//! GPU quantize-dequantize round-trip wrappers (Q4_0 and Q8_0).
//!
//! Higher-level surface over `mlx_native::ops::qdq_legacy` that takes
//! a host slice, runs the Metal kernel, and returns the round-tripped
//! values as a `Vec<f32>` (or a `GpuTensor` leaf for tape integration).
//!
//! These produce the `W_low` and `W_high` weight tensors that feed the
//! gradient-Taylor sensitivity formula `Σ grad · (W_low − W_high)` in
//! `estimate_sensitivities`.  Used by ADR-020 Track 1 (port of
//! mlx-lm `dynamic_quant.py`).
//!
//! Per ADR-020 §11 ("no external tools / no CPU fallback"): all
//! production-path qdq runs on GPU.  The CPU oracles in
//! [`crate::quantize::q_legacy`] are the byte-identical reference,
//! reachable only from `#[cfg(test)]` parity validators here.

use anyhow::{anyhow, Result};
use mlx_native::ops::qdq_legacy;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use crate::calibrate::autograd_gpu_tape::{GpuTape, GpuTensor};

/// GGUF block size — both Q4_0 and Q8_0 use 32-element blocks.
pub const QDQ_BLOCK_SIZE: usize = qdq_legacy::QDQ_BLOCK_SIZE as usize;

/// Run the GPU Q4_0 quantize-dequantize round-trip on a host slice.
///
/// Returns a `Vec<f32>` of the same length as `input` containing the
/// round-tripped values — byte-identical to
/// `quantize_row_q4_0 → dequantize_row_q4_0` from
/// [`crate::quantize::q_legacy`].
///
/// # Errors
/// - `input.len()` not divisible by [`QDQ_BLOCK_SIZE`] (32)
/// - Metal device / buffer allocation failures
pub fn qdq_q4_0_gpu(device: &MlxDevice, input: &[f32]) -> Result<Vec<f32>> {
    qdq_gpu_impl(device, input, "q4_0")
}

/// Run the GPU Q8_0 quantize-dequantize round-trip on a host slice.
pub fn qdq_q8_0_gpu(device: &MlxDevice, input: &[f32]) -> Result<Vec<f32>> {
    qdq_gpu_impl(device, input, "q8_0")
}

fn qdq_gpu_impl(device: &MlxDevice, input: &[f32], kind: &'static str) -> Result<Vec<f32>> {
    if !input.len().is_multiple_of(QDQ_BLOCK_SIZE) {
        return Err(anyhow!(
            "qdq_{kind}_gpu: input length {} not divisible by block size {QDQ_BLOCK_SIZE}",
            input.len()
        ));
    }
    let n_bytes = input.len() * std::mem::size_of::<f32>();
    let mut in_buf = device
        .alloc_buffer(n_bytes, DType::F32, vec![input.len()])
        .map_err(|e| anyhow!("qdq_{kind}_gpu alloc input: {e}"))?;
    let out_buf = device
        .alloc_buffer(n_bytes, DType::F32, vec![input.len()])
        .map_err(|e| anyhow!("qdq_{kind}_gpu alloc output: {e}"))?;
    {
        let slice: &mut [f32] = in_buf
            .as_mut_slice()
            .map_err(|e| anyhow!("qdq_{kind}_gpu input as_mut_slice: {e}"))?;
        slice.copy_from_slice(input);
    }
    let mut registry = KernelRegistry::new();
    qdq_legacy::register(&mut registry);
    let mut encoder = device
        .command_encoder()
        .map_err(|e| anyhow!("qdq_{kind}_gpu command_encoder: {e}"))?;
    match kind {
        "q4_0" => qdq_legacy::dispatch_qdq_q4_0_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &in_buf,
            &out_buf,
        ),
        "q8_0" => qdq_legacy::dispatch_qdq_q8_0_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &in_buf,
            &out_buf,
        ),
        _ => unreachable!("qdq kind must be q4_0 or q8_0"),
    }
    .map_err(|e| anyhow!("qdq_{kind}_gpu dispatch: {e}"))?;
    encoder
        .commit_and_wait()
        .map_err(|e| anyhow!("qdq_{kind}_gpu commit_and_wait: {e}"))?;
    let out_slice: &[f32] = out_buf
        .as_slice()
        .map_err(|e| anyhow!("qdq_{kind}_gpu output as_slice: {e}"))?;
    Ok(out_slice.to_vec())
}

/// Run the GPU Q4_0 round-trip and return a `GpuTensor` leaf on the
/// caller-provided tape.  Convenience for `estimate_sensitivities`
/// callers that need `W_low` / `W_high` as tape leaves.
pub fn qdq_q4_0_to_tensor(
    tape: &GpuTape,
    input: &[f32],
    shape: Vec<usize>,
) -> Result<GpuTensor> {
    let numel: usize = shape.iter().product();
    if input.len() != numel {
        return Err(anyhow!(
            "qdq_q4_0_to_tensor: input.len()={} != shape product {numel}",
            input.len()
        ));
    }
    let qdq_out = qdq_q4_0_gpu(tape.device(), input)?;
    GpuTensor::from_vec(tape, &qdq_out, shape)
}

/// Run the GPU Q8_0 round-trip and return a `GpuTensor` leaf.
pub fn qdq_q8_0_to_tensor(
    tape: &GpuTape,
    input: &[f32],
    shape: Vec<usize>,
) -> Result<GpuTensor> {
    let numel: usize = shape.iter().product();
    if input.len() != numel {
        return Err(anyhow!(
            "qdq_q8_0_to_tensor: input.len()={} != shape product {numel}",
            input.len()
        ));
    }
    let qdq_out = qdq_q8_0_gpu(tape.device(), input)?;
    GpuTensor::from_vec(tape, &qdq_out, shape)
}

/// Internal helper for tests — same as [`qdq_q4_0_gpu`] but constructs
/// its own one-off `MlxDevice`.
#[cfg(test)]
fn qdq_q4_0_gpu_oneshot(input: &[f32]) -> Result<Vec<f32>> {
    let device = MlxDevice::new().map_err(|e| anyhow!("MlxDevice::new: {e}"))?;
    qdq_q4_0_gpu(&device, input)
}

#[cfg(test)]
fn qdq_q8_0_gpu_oneshot(input: &[f32]) -> Result<Vec<f32>> {
    let device = MlxDevice::new().map_err(|e| anyhow!("MlxDevice::new: {e}"))?;
    qdq_q8_0_gpu(&device, input)
}

// Suppress dead-code lint on `MlxBuffer` import — referenced only in
// the `mlx_native::ops::qdq_legacy` re-export above as a transitive
// dep of the dispatch functions.  Kept to make the surface explicit.
const _: fn() -> Option<MlxBuffer> = || None;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::q_legacy::{
        dequantize_row_q4_0, dequantize_row_q8_0, quantize_row_q4_0, quantize_row_q8_0, BlockQ4_0,
        BlockQ8_0, QK4_0, QK8_0,
    };

    /// Canonical CPU oracle for Q4_0: round-trip via q_legacy.
    fn q4_0_canonical_cpu_oracle(input: &[f32]) -> Vec<f32> {
        assert!(input.len().is_multiple_of(QK4_0));
        let nb = input.len() / QK4_0;
        let mut blocks = vec![
            BlockQ4_0 {
                d_bits: 0,
                qs: [0u8; QK4_0 / 2],
            };
            nb
        ];
        quantize_row_q4_0(input, &mut blocks).expect("quantize_row_q4_0");
        let mut out = vec![0f32; input.len()];
        dequantize_row_q4_0(&blocks, &mut out).expect("dequantize_row_q4_0");
        out
    }

    /// Canonical CPU oracle for Q8_0: round-trip via q_legacy.
    fn q8_0_canonical_cpu_oracle(input: &[f32]) -> Vec<f32> {
        assert!(input.len().is_multiple_of(QK8_0));
        let nb = input.len() / QK8_0;
        let mut blocks = vec![
            BlockQ8_0 {
                d_bits: 0,
                qs: [0i8; QK8_0],
            };
            nb
        ];
        quantize_row_q8_0(input, &mut blocks).expect("quantize_row_q8_0");
        let mut out = vec![0f32; input.len()];
        dequantize_row_q8_0(&blocks, &mut out).expect("dequantize_row_q8_0");
        out
    }

    fn assert_byte_identical(label: &str, gpu: &[f32], cpu: &[f32]) {
        assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            if g.to_bits() != c.to_bits() {
                panic!(
                    "{label}: bit-mismatch at index {i}: gpu={} (0x{:08x}) cpu={} (0x{:08x})",
                    g,
                    g.to_bits(),
                    c,
                    c.to_bits()
                );
            }
        }
    }

    #[test]
    fn qdq_q4_0_gpu_byte_identical_to_q_legacy_canonical() {
        // Single-block linear ramp.
        let input: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.137).collect();
        let gpu = qdq_q4_0_gpu_oneshot(&input).expect("gpu");
        let cpu = q4_0_canonical_cpu_oracle(&input);
        assert_byte_identical("q4_0 canonical ramp", &gpu, &cpu);
    }

    #[test]
    fn qdq_q4_0_gpu_byte_identical_multi_block() {
        let mut input = Vec::with_capacity(256);
        for blk in 0..8 {
            let scale = (blk as f32 + 1.0) * 0.5;
            for i in 0..32 {
                let v = ((i as f32 * 17.0 + blk as f32 * 31.0).sin()) * scale;
                input.push(v);
            }
        }
        let gpu = qdq_q4_0_gpu_oneshot(&input).expect("gpu");
        let cpu = q4_0_canonical_cpu_oracle(&input);
        assert_byte_identical("q4_0 canonical multi-block sin", &gpu, &cpu);
    }

    #[test]
    fn qdq_q8_0_gpu_byte_identical_to_q_legacy_canonical() {
        let input: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.137).collect();
        let gpu = qdq_q8_0_gpu_oneshot(&input).expect("gpu");
        let cpu = q8_0_canonical_cpu_oracle(&input);
        assert_byte_identical("q8_0 canonical ramp", &gpu, &cpu);
    }

    #[test]
    fn qdq_q8_0_gpu_byte_identical_multi_block() {
        let mut input = Vec::with_capacity(256);
        for blk in 0..8 {
            let scale = (blk as f32 + 1.0) * 0.5;
            for i in 0..32 {
                let v = ((i as f32 * 17.0 + blk as f32 * 31.0).sin()) * scale;
                input.push(v);
            }
        }
        let gpu = qdq_q8_0_gpu_oneshot(&input).expect("gpu");
        let cpu = q8_0_canonical_cpu_oracle(&input);
        assert_byte_identical("q8_0 canonical multi-block sin", &gpu, &cpu);
    }

    #[test]
    fn qdq_q4_0_gpu_realistic_weight_range() {
        // Simulate a 32-row Linear weight column: 8 blocks × 32 elements
        // with values typical of fp16 weight scales (~ ±1.0).
        let mut input = Vec::with_capacity(256);
        for i in 0..256 {
            let v = (i as f32 / 256.0 - 0.5) * 2.0;
            // Add a few outliers like real weight distributions have.
            let v = if i % 31 == 0 { v * 3.0 } else { v };
            input.push(v);
        }
        let gpu = qdq_q4_0_gpu_oneshot(&input).expect("gpu");
        let cpu = q4_0_canonical_cpu_oracle(&input);
        assert_byte_identical("q4_0 realistic weight range", &gpu, &cpu);
    }

    #[test]
    fn qdq_q8_0_gpu_realistic_weight_range() {
        let mut input = Vec::with_capacity(256);
        for i in 0..256 {
            let v = (i as f32 / 256.0 - 0.5) * 2.0;
            let v = if i % 31 == 0 { v * 3.0 } else { v };
            input.push(v);
        }
        let gpu = qdq_q8_0_gpu_oneshot(&input).expect("gpu");
        let cpu = q8_0_canonical_cpu_oracle(&input);
        assert_byte_identical("q8_0 realistic weight range", &gpu, &cpu);
    }

    #[test]
    fn qdq_q4_0_to_tensor_round_trip() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) * 0.05).collect();
        let device = MlxDevice::new().expect("MlxDevice::new");
        let tape = GpuTape::new(device);
        let tensor = qdq_q4_0_to_tensor(&tape, &input, vec![64]).expect("qdq_q4_0_to_tensor");
        let buf = tensor.to_vec().expect("to_vec");
        assert_eq!(buf.len(), 64);
        let cpu = q4_0_canonical_cpu_oracle(&input);
        assert_byte_identical("q4_0 to_tensor", &buf, &cpu);
    }

    #[test]
    fn qdq_q8_0_to_tensor_round_trip() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) * 0.05).collect();
        let device = MlxDevice::new().expect("MlxDevice::new");
        let tape = GpuTape::new(device);
        let tensor = qdq_q8_0_to_tensor(&tape, &input, vec![64]).expect("qdq_q8_0_to_tensor");
        let buf = tensor.to_vec().expect("to_vec");
        assert_eq!(buf.len(), 64);
        let cpu = q8_0_canonical_cpu_oracle(&input);
        assert_byte_identical("q8_0 to_tensor", &buf, &cpu);
    }

    #[test]
    fn qdq_input_len_must_be_block_aligned() {
        let device = MlxDevice::new().expect("MlxDevice::new");
        // 33 elements → not block-aligned.
        let input: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let err = qdq_q4_0_gpu(&device, &input).expect_err("must reject non-block-aligned");
        assert!(format!("{err}").contains("not divisible by block size"));
    }

    #[test]
    fn qdq_q4_0_low_q8_0_high_diff_is_nonzero() {
        // Sanity: for the sensitivity formula `Σ grad · (W_low - W_high)`
        // to be non-trivial, qdq_q4_0(w) and qdq_q8_0(w) must differ on
        // realistic weights.  This is the gradient-Taylor anchor.
        let mut input = Vec::with_capacity(256);
        for i in 0..256 {
            input.push((i as f32 / 256.0 - 0.5) * 2.0);
        }
        let device = MlxDevice::new().expect("MlxDevice::new");
        let w_low = qdq_q4_0_gpu(&device, &input).expect("q4_0");
        let w_high = qdq_q8_0_gpu(&device, &input).expect("q8_0");
        let max_abs_diff = w_low
            .iter()
            .zip(w_high.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // Q4_0 is much coarser than Q8_0, so on a realistic weight
        // range the round-trips MUST differ by at least one Q4_0 step.
        assert!(
            max_abs_diff > 1e-4,
            "qdq_q4_0 and qdq_q8_0 must differ on realistic weights; \
             max_abs_diff={max_abs_diff}"
        );
    }
}
