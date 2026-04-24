//! GPU-side weight containers and per-step verification harness for the
//! Qwen3.5 gated full-attention forward pass (ADR-013 Decision 9, GPU path).
//!
//! This module is the bridge between [`super::full_attn`]'s pure-Rust scalar
//! reference (the authoritative spec + test oracle) and the mlx-native GPU
//! kernels. It carries the per-layer weights as `MlxBuffer` handles and
//! exposes the per-op dispatches in an orderable fashion so both
//! unit tests and the eventual `build_gated_attn_layer` can consume them.
//!
//! # Strategy
//!
//! The full GPU builder is ~7 mlx-native dispatches deep. Rather than write
//! all 7 at once and risk a hard-to-debug parity failure, each dispatch is
//! exposed as a small public function that takes and returns `MlxBuffer`
//! (or f32 `Vec` for test observability). The top-level builder composes
//! these. Each layer of the pipeline ships with a CPU-parity test that
//! runs the op alone and compares against an in-test f32 recomputation or
//! `super::full_attn`'s full-layer reference.
//!
//! This iter (iter 15 / P7b-prep) lands:
//! - `FullAttnWeightsGpu` — MlxBuffer container.
//! - `FullAttnWeightsGpu::from_cpu()` — uploads f32 weights to Metal.
//! - `apply_pre_attn_rms_norm()` — first op in the forward sequence; used
//!    as the pilot parity test proving the CPU→GPU bridge works on Qwen3.5
//!    shapes.
//!
//! Subsequent iters add `apply_qkv_projection()`, `apply_qk_per_head_norm()`,
//! `apply_imrope()`, `apply_sdpa_with_causal_mask()`, `apply_output_gate()`,
//! `apply_output_projection()`, then compose into `build_gated_attn_layer()`
//! with a full-layer parity test against `gated_full_attention_cpu_ref`.

use anyhow::{anyhow, Context, Result};
use mlx_native::ops::rms_norm;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::full_attn::FullAttnLayerWeights;

/// GPU-side weight handles for a single Qwen3.5 full-attention layer.
///
/// Uploaded from [`FullAttnLayerWeights`] once per layer at load time;
/// held by the model + read by the per-token forward.
pub struct FullAttnWeightsGpu {
    pub attn_norm: MlxBuffer,
    pub wq: MlxBuffer,
    pub wk: MlxBuffer,
    pub wv: MlxBuffer,
    pub w_gate: MlxBuffer,
    pub attn_q_norm: MlxBuffer,
    pub attn_k_norm: MlxBuffer,
    pub wo: MlxBuffer,
}

impl FullAttnWeightsGpu {
    /// Upload a [`FullAttnLayerWeights`] (pure-Rust f32) to Metal buffers.
    pub fn from_cpu(weights: &FullAttnLayerWeights, device: &MlxDevice) -> Result<Self> {
        Ok(Self {
            attn_norm: upload_f32(&weights.attn_norm, device)?,
            wq: upload_f32(&weights.wq, device)?,
            wk: upload_f32(&weights.wk, device)?,
            wv: upload_f32(&weights.wv, device)?,
            w_gate: upload_f32(&weights.w_gate, device)?,
            attn_q_norm: upload_f32(&weights.attn_q_norm, device)?,
            attn_k_norm: upload_f32(&weights.attn_k_norm, device)?,
            wo: upload_f32(&weights.wo, device)?,
        })
    }
}

/// Helper: copy an f32 `Vec` into a freshly-allocated `MlxBuffer` with shape
/// set to `[len]` (1-D). Callers can reshape the buffer later by constructing
/// a new buffer with the desired shape and copying — shape here is advisory
/// only (mlx-native kernels consult `element_count()` + dtype, not shape).
pub fn upload_f32(data: &[f32], device: &MlxDevice) -> Result<MlxBuffer> {
    let byte_len = data.len() * 4;
    let mut buf = device
        .alloc_buffer(byte_len, DType::F32, vec![data.len()])
        .map_err(|e| anyhow!("alloc f32 buffer len={}: {e}", data.len()))?;
    {
        let slice = buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("mut_slice: {e}"))?;
        slice.copy_from_slice(data);
    }
    Ok(buf)
}

/// Download an `MlxBuffer` of f32 values into a `Vec<f32>`.
pub fn download_f32(buf: &MlxBuffer) -> Result<Vec<f32>> {
    if buf.dtype() != DType::F32 {
        return Err(anyhow!(
            "download_f32: buffer dtype {} != f32",
            buf.dtype()
        ));
    }
    let slice: &[f32] = buf.as_slice().map_err(|e| anyhow!("as_slice: {e}"))?;
    Ok(slice.to_vec())
}

/// Apply pre-attention RMSNorm to a residual-stream input buffer.
///
/// Produces a new f32 buffer with the same shape. The output buffer is
/// allocated by this function; callers can reuse it downstream by passing
/// it as input to the next dispatch.
///
/// # Parity contract
///
/// Output must match [`super::full_attn::gated_full_attention_cpu_ref`]'s
/// step-1 output (RMSNorm row-wise with `attn_norm` weight, `rms_norm_eps`
/// from config) to ≤1e-5 per element for F32.
pub fn apply_pre_attn_rms_norm(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weights_gpu: &FullAttnWeightsGpu,
    seq_len: u32,
    hidden_size: u32,
    eps: f32,
) -> Result<MlxBuffer> {
    // Allocate output + params.
    let out = device
        .alloc_buffer(
            (seq_len * hidden_size) as usize * 4,
            DType::F32,
            vec![seq_len as usize, hidden_size as usize],
        )
        .map_err(|e| anyhow!("alloc out: {e}"))?;
    let mut params = device
        .alloc_buffer(8, DType::F32, vec![2])
        .map_err(|e| anyhow!("alloc params: {e}"))?;
    {
        let s = params
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("mut_slice: {e}"))?;
        s[0] = eps;
        s[1] = hidden_size as f32;
    }

    rms_norm::dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        input,
        &weights_gpu.attn_norm,
        &out,
        &params,
        seq_len,
        hidden_size,
    )
    .context("dispatch_rms_norm")?;

    Ok(out)
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::full_attn::{FullAttnLayerWeights, FullAttnShape};

    fn mk_rand(seed: &mut u32, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((*seed as i32 as f32) / (i32::MAX as f32)) * scale
            })
            .collect()
    }

    fn small_shape_and_weights() -> (FullAttnShape, FullAttnLayerWeights, u32) {
        let shape = FullAttnShape {
            hidden_size: 32,
            n_head: 4,
            n_kv: 2,
            head_dim: 16,
            rotary_dim: 8,
            rope_theta: 10000.0,
            mrope_section: [2, 2, 0, 0],
            rms_norm_eps: 1e-6,
        };
        let h = shape.hidden_size as usize;
        let nh = shape.n_head as usize;
        let nkv = shape.n_kv as usize;
        let d = shape.head_dim as usize;
        let q_total = nh * d;
        let kv_total = nkv * d;

        let mut seed = 0x1337_u32;
        let weights = FullAttnLayerWeights {
            attn_norm: {
                let mut v = vec![1.0f32; h];
                for (i, x) in v.iter_mut().enumerate() {
                    *x += 0.01 * (i as f32);
                }
                v
            },
            wq: mk_rand(&mut seed, q_total * h, 0.1),
            wk: mk_rand(&mut seed, kv_total * h, 0.1),
            wv: mk_rand(&mut seed, kv_total * h, 0.1),
            w_gate: mk_rand(&mut seed, q_total * h, 0.1),
            attn_q_norm: mk_rand(&mut seed, d, 0.05).into_iter().map(|v| 1.0 + v).collect(),
            attn_k_norm: mk_rand(&mut seed, d, 0.05).into_iter().map(|v| 1.0 + v).collect(),
            wo: mk_rand(&mut seed, h * q_total, 0.1),
        };
        let seq_len = 4u32;
        (shape, weights, seq_len)
    }

    /// Round-trip `upload_f32`/`download_f32` preserves contents.
    #[test]
    fn upload_download_roundtrip() {
        let device = MlxDevice::new().expect("device");
        let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.137 - 5.0).collect();
        let buf = upload_f32(&data, &device).expect("upload");
        let got = download_f32(&buf).expect("download");
        assert_eq!(got, data);
    }

    /// Weight upload into `FullAttnWeightsGpu` preserves all 8 tensors.
    #[test]
    fn from_cpu_uploads_all_weights() {
        let device = MlxDevice::new().expect("device");
        let (shape, weights_cpu, _) = small_shape_and_weights();
        let gpu = FullAttnWeightsGpu::from_cpu(&weights_cpu, &device).expect("upload");

        let h = shape.hidden_size as usize;
        let nh = shape.n_head as usize;
        let nkv = shape.n_kv as usize;
        let d = shape.head_dim as usize;
        let q_total = nh * d;
        let kv_total = nkv * d;

        // Verify every buffer was uploaded with correct contents.
        for (name, expected, buf) in [
            ("attn_norm", &weights_cpu.attn_norm, &gpu.attn_norm),
            ("wq", &weights_cpu.wq, &gpu.wq),
            ("wk", &weights_cpu.wk, &gpu.wk),
            ("wv", &weights_cpu.wv, &gpu.wv),
            ("w_gate", &weights_cpu.w_gate, &gpu.w_gate),
            ("attn_q_norm", &weights_cpu.attn_q_norm, &gpu.attn_q_norm),
            ("attn_k_norm", &weights_cpu.attn_k_norm, &gpu.attn_k_norm),
            ("wo", &weights_cpu.wo, &gpu.wo),
        ] {
            let got = download_f32(buf).expect("download");
            assert_eq!(
                got.len(),
                expected.len(),
                "{name}: length mismatch"
            );
            for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(g.to_bits(), e.to_bits(), "{name}[{i}]");
            }
        }

        // Suppress unused warnings for shape dims used in the fixture.
        let _ = (h, q_total, kv_total);
    }

    /// **Pilot parity test**: pre-attention RMSNorm on the GPU matches the
    /// scalar CPU reference to 1e-5. This is the first CPU→GPU bridge
    /// verified for the Qwen3.5 full-attention pipeline; proves the weight
    /// upload + dispatch + download plumbing works end-to-end.
    #[test]
    fn pre_attn_rms_norm_matches_cpu_ref() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let (shape, weights_cpu, seq_len) = small_shape_and_weights();
        let h = shape.hidden_size as usize;

        // Synthetic input.
        let mut seed = 0x4242_u32;
        let x_cpu: Vec<f32> = (0..(seq_len as usize * h))
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((seed as i32 as f32) / (i32::MAX as f32)) * 0.5
            })
            .collect();

        // CPU reference: run rms_norm_row per token.
        let mut expected = vec![0.0f32; seq_len as usize * h];
        for t in 0..seq_len as usize {
            let row = &x_cpu[t * h..(t + 1) * h];
            // Inline the same formula that full_attn::rms_norm_row uses:
            //   inv = 1 / sqrt(mean(row^2) + eps)
            //   out = row * inv * weight
            let sum_sq: f32 = row.iter().map(|v| v * v).sum();
            let inv = ((sum_sq / (h as f32)) + shape.rms_norm_eps).sqrt().recip();
            for j in 0..h {
                expected[t * h + j] = row[j] * inv * weights_cpu.attn_norm[j];
            }
        }

        // GPU path.
        let gpu = FullAttnWeightsGpu::from_cpu(&weights_cpu, &device).expect("upload");
        let input_gpu = upload_f32(&x_cpu, &device).expect("input");

        let mut encoder = device.command_encoder().expect("encoder");
        let out_gpu = apply_pre_attn_rms_norm(
            &mut encoder,
            &mut registry,
            &device,
            &input_gpu,
            &gpu,
            seq_len,
            shape.hidden_size,
            shape.rms_norm_eps,
        )
        .expect("apply rms_norm");
        encoder.commit_and_wait().expect("commit");

        let got = download_f32(&out_gpu).expect("download output");
        assert_eq!(got.len(), expected.len());
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let d = (g - e).abs();
            assert!(
                d < 1e-5,
                "pre_attn_rms_norm mismatch at {}: gpu={}, cpu={}, diff={}",
                i, g, e, d
            );
        }
    }

    /// Dtype correctness: `upload_f32` produces an F32 buffer.
    #[test]
    fn upload_f32_is_f32_dtype() {
        let device = MlxDevice::new().expect("device");
        let data = vec![1.0f32, 2.0, 3.0];
        let buf = upload_f32(&data, &device).expect("upload");
        assert_eq!(buf.dtype(), DType::F32);
        assert_eq!(buf.element_count(), 3);
    }

    /// download_f32 rejects non-F32 buffers with a clear error.
    #[test]
    fn download_rejects_wrong_dtype() {
        let device = MlxDevice::new().expect("device");
        let buf = device
            .alloc_buffer(4, DType::U32, vec![1])
            .expect("alloc u32");
        let res = download_f32(&buf);
        assert!(res.is_err(), "download_f32 should reject u32 buffer");
    }
}
