//! GPU-side GLP apply: a per-layer projection/additive dispatch on the
//! activation buffer, in the same encoder discipline as the family forward.
//!
//! - add:     `row += alpha * d`                (add_bias_row_2d shape)
//! - project: `row -= alpha * (row·d̂)d̂`        (custom glp_project_f32)
//!
//! The projection kernel's source lives in-tree (`shaders/glp_project.metal`)
//! and is registered with the family's KernelRegistry on first use, so we
//! never patch upstream crates. Alpha is a host-side scalar; the direction
//! arrives pre-uploaded and pre-normalized (the host divides by ‖d‖ once at
//! bind time, so the kernel's per-row dot is against the unit direction and
//! the scale is simply `alpha * dot`).

use anyhow::{Context, Result};
use mlx_native::metal;
use mlx_native::ops::encode_helpers::{as_bytes, encode_with_args, KernelArg};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::reader::GlpMode;

const PROJECT_SHADER: &str = include_str!("shaders/glp_project.metal");
const PROJECT_MHC_SHADER: &str = include_str!("shaders/glp_project_mhc.metal");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GlpProjectMhcParams {
    rows: u32,
    hc: u32,
    hidden: u32,
    per_stream_directions: u32,
    alpha: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GlpProjectParams {
    m: u32,
    n: u32,
    alpha: f32,
    d_norm_sq: f32,
}

/// Apply one layer's GLP direction to the activation buffer `hidden`
/// (`seq_len × h` F32), in place.
pub fn apply_layer_gpu(
    hidden: &mut MlxBuffer,
    direction: &MlxBuffer,
    mode: GlpMode,
    alpha: f32,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    seq_len: u32,
    h: u32,
) -> Result<()> {
    match mode {
        GlpMode::Add => apply_additive(hidden, direction, alpha, device, registry, seq_len, h),
        GlpMode::Project => {
            apply_projection(hidden, direction, alpha, device, registry, seq_len, h)
        }
    }
}

fn apply_additive(
    hidden: &mut MlxBuffer,
    direction: &MlxBuffer,
    alpha: f32,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    seq_len: u32,
    h: u32,
) -> Result<()> {
    // out[m, n] = hidden[m, n] + alpha * direction[n]  — row-broadcast add.
    // We encode alpha by pre-scaling the direction into a scratch buffer
    // (one tiny CPU upload per layer per request; the alternative is a
    // second params buffer in the shader — keep it simple for v1).
    let scaled: Vec<f32> = direction
        .as_slice::<f32>()
        .context("GLP direction must be CPU-readable for additive scaling")?
        .iter()
        .map(|x| x * alpha)
        .collect();
    let scaled_buf = {
        let byte_len = std::mem::size_of_val(scaled.as_slice());
        let raw = device.metal_device().new_buffer_with_data(
            scaled.as_ptr().cast(),
            byte_len as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        MlxBuffer::from_raw(raw, DType::F32, vec![scaled.len()])
    };
    let mut enc = device
        .command_encoder()
        .context("GLP additive encoder")?;
    mlx_native::ops::add_bias_row_2d::register(registry);
    mlx_native::ops::add_bias_row_2d::dispatch_add_bias_row_2d_f32(
        &mut enc,
        registry,
        device.metal_device(),
        hidden,
        &scaled_buf,
        hidden,
        seq_len,
        h,
    )
    .context("GLP additive dispatch")?;
    enc.commit_and_wait().context("GLP additive commit")?;
    Ok(())
}

fn apply_projection(
    hidden: &mut MlxBuffer,
    direction: &MlxBuffer,
    alpha: f32,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    seq_len: u32,
    h: u32,
) -> Result<()> {
    // d̂ normalization is the host's job at bind time; the kernel divides
    // the per-row dot by the (unit-normed) direction's squared norm, which
    // for a unit direction is 1.0 — we still pass it explicitly so the
    // kernel stays honest when a non-normalized vector is loaded.
    let d_norm_sq: f32 = direction
        .as_slice::<f32>()
        .context("GLP direction must be CPU-readable for norm computation")?
        .iter()
        .map(|x| x * x)
        .sum();
    if d_norm_sq <= f32::EPSILON {
        anyhow::bail!("GLP project direction has near-zero norm");
    }
    registry.register_source("glp_project_f32", PROJECT_SHADER);
    let params = GlpProjectParams {
        m: seq_len,
        n: h,
        alpha,
        d_norm_sq,
    };
    let pipeline = registry
        .get_pipeline("glp_project_f32", device.metal_device())
        .context("GLP projection pipeline")?;
    // Launch geometry (S1 fix): the shader's reduction is hard-coded to a
    // 256-lane threadgroup (`partial[256]`, stride-128 tree) and maps
    // threadgroup id → row, covering each row's columns strided. The host
    // must therefore dispatch EXACTLY 256 threads per threadgroup and one
    // threadgroup per row: grid = rows * 256. The previous `tg = h` launch
    // indexed `partial[256]` out of bounds for any hidden width > 256
    // (every Qwen family) and exceeded Metal's threadgroup limit for
    // widths > 1024.
    let grid = metal::MTLSize::new((seq_len as u64) * 256, 1, 1);
    let tg = metal::MTLSize::new(256, 1, 1);
    let mut enc = device
        .command_encoder()
        .context("GLP projection encoder")?;
    encode_with_args(
        &mut enc,
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Buffer(direction)),
            (2, KernelArg::Buffer(hidden)),
        ],
        grid,
        tg,
    );
    enc.commit_and_wait().context("GLP projection commit")?;
    Ok(())
}

/// DeepSeek mHC variant: `state` is `[rows, hc, hidden]` F32; the direction
/// is applied per (row, stream) slice of width `hidden`. If the bound
/// direction width is `hidden`, one shared direction steers every stream;
/// if it is `hc*hidden`, each stream gets its own slice (weightless mHC
/// discipline — never the flattened sum).
pub fn apply_layer_gpu_mhc(
    state: &mut MlxBuffer,
    direction: &MlxBuffer,
    mode: GlpMode,
    alpha: f32,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    rows: u32,
    hc: u32,
    hidden: u32,
) -> Result<()> {
    if mode != GlpMode::Project {
        anyhow::bail!("GLP mHC apply supports project mode only (add is untested on HC state)");
    }
    let dir_width = direction.shape().first().copied().unwrap_or(0) as u32;
    let per_stream = (dir_width == hc * hidden) as u32;
    anyhow::ensure!(
        dir_width == hidden || dir_width == hc * hidden,
        "GLP mHC direction width {dir_width} must equal hidden {hidden} or hc*hidden {}",
        hc * hidden
    );
    // S13: per-stream norms — the shared-direction form uses one norm; the
    // per-stream form (width == hc*hidden) computes EACH stream's own norm.
    // The previous code took stream 0's norm for every stream, mis-scaling
    // any stack with unequal per-stream norms. f64 accumulation (see bind).
    let dir_values: Vec<f32> = direction
        .as_slice::<f32>()
        .context("GLP direction must be CPU-readable for norm computation")?
        .to_vec();
    let mut norms = Vec::with_capacity(hc as usize);
    for stream in 0..hc as usize {
        let slice = if per_stream == 1 {
            &dir_values[stream * hidden as usize..(stream + 1) * hidden as usize]
        } else {
            &dir_values[..hidden as usize]
        };
        let norm_sq: f64 = slice.iter().map(|x| (*x as f64) * (*x as f64)).sum();
        if !norm_sq.is_finite() || norm_sq <= 0.0 {
            anyhow::bail!(
                "GLP mHC direction stream {stream} has an invalid norm ({norm_sq})"
            );
        }
        norms.push(norm_sq as f32);
    }
    let norms_buf = {
        let byte_len = std::mem::size_of_val(norms.as_slice());
        let raw = device.metal_device().new_buffer_with_data(
            norms.as_ptr().cast(),
            byte_len as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        MlxBuffer::from_raw(raw, DType::F32, vec![norms.len()])
    };
    registry.register_source("glp_project_mhc_f32", PROJECT_MHC_SHADER);
    let params = GlpProjectMhcParams {
        rows,
        hc,
        hidden,
        per_stream_directions: per_stream,
        alpha,
    };
    let pipeline = registry
        .get_pipeline("glp_project_mhc_f32", device.metal_device())
        .context("GLP mHC projection pipeline")?;
    // S13: canonical launch geometry — one FULL 256-lane threadgroup per
    // (row, stream) pair: grid = rows*hc*256 threads, tg = 256. The
    // previous grid (rows*hc TOTAL threads) launched rows*hc/256 groups —
    // for one row and four streams that is four threads in one partial
    // group, leaving the 256-lane reduction's partial[4..255] unwritten
    // and covering only stream 0.
    let grid = metal::MTLSize::new((rows as u64) * (hc as u64) * 256, 1, 1);
    let tg = metal::MTLSize::new(256, 1, 1);
    let mut enc = device
        .command_encoder()
        .context("GLP mHC projection encoder")?;
    encode_with_args(
        &mut enc,
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Buffer(direction)),
            (2, KernelArg::Buffer(state)),
            (3, KernelArg::Buffer(&norms_buf)),
        ],
        grid,
        tg,
    );
    enc.commit_and_wait().context("GLP mHC projection commit")?;
    Ok(())
}

/// In-session variant: encode the projection into an existing GraphSession
/// (used when the GLP hook runs inside a shared FFN session that the caller
/// commits later). No commit here — the session owner does that.
pub fn apply_layer_gpu_mhc_in_session(
    session: &mut mlx_native::graph::GraphSession<'_>,
    registry: &mut KernelRegistry,
    state: &MlxBuffer,
    direction: &MlxBuffer,
    alpha: f32,
    rows: u32,
    hc: u32,
    hidden: u32,
) -> Result<()> {
    let dir_width = direction.shape().first().copied().unwrap_or(0) as u32;
    let per_stream = (dir_width == hc * hidden) as u32;
    anyhow::ensure!(
        dir_width == hidden || dir_width == hc * hidden,
        "GLP mHC direction width {dir_width} must equal hidden {hidden} or hc*hidden {}",
        hc * hidden
    );
    // S13: per-stream norms (see the out-of-session helper) and the
    // canonical one-full-group-per-(row, stream) launch geometry.
    let dir_values: Vec<f32> = direction
        .as_slice::<f32>()
        .context("GLP direction must be CPU-readable for norm computation")?
        .to_vec();
    let mut norms = Vec::with_capacity(hc as usize);
    for stream in 0..hc as usize {
        let slice = if per_stream == 1 {
            &dir_values[stream * hidden as usize..(stream + 1) * hidden as usize]
        } else {
            &dir_values[..hidden as usize]
        };
        let norm_sq: f64 = slice.iter().map(|x| (*x as f64) * (*x as f64)).sum();
        if !norm_sq.is_finite() || norm_sq <= 0.0 {
            anyhow::bail!(
                "GLP mHC direction stream {stream} has an invalid norm ({norm_sq})"
            );
        }
        norms.push(norm_sq as f32);
    }
    let device = session.device().clone();
    let norms_buf = {
        let byte_len = std::mem::size_of_val(norms.as_slice());
        let raw = device.metal_device().new_buffer_with_data(
            norms.as_ptr().cast(),
            byte_len as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        MlxBuffer::from_raw(raw, DType::F32, vec![norms.len()])
    };
    registry.register_source("glp_project_mhc_f32", PROJECT_MHC_SHADER);
    let params = GlpProjectMhcParams {
        rows,
        hc,
        hidden,
        per_stream_directions: per_stream,
        alpha,
    };
    let pipeline = registry
        .get_pipeline("glp_project_mhc_f32", device.metal_device())
        .context("GLP mHC projection pipeline")?;
    let grid = metal::MTLSize::new((rows as u64) * (hc as u64) * 256, 1, 1);
    let tg = metal::MTLSize::new(256, 1, 1);
    encode_with_args(
        session.encoder_mut(),
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Buffer(direction)),
            (2, KernelArg::Buffer(state)),
            (3, KernelArg::Buffer(&norms_buf)),
        ],
        grid,
        tg,
    );
    Ok(())
}

/// In-session dense variant: project on a `[rows, hidden]` F32 buffer
/// (the FFN-writer site on DeepSeek; the post-layer residual on dense
/// families). No commit here — the session owner does that.
pub fn apply_layer_gpu_in_session(
    session: &mut mlx_native::graph::GraphSession<'_>,
    registry: &mut KernelRegistry,
    state: &MlxBuffer,
    direction: &MlxBuffer,
    mode: GlpMode,
    alpha: f32,
    rows: u32,
    hidden: u32,
) -> Result<()> {
    let dir_width = direction.shape().first().copied().unwrap_or(0) as u32;
    anyhow::ensure!(
        dir_width == hidden,
        "GLP dense direction width {dir_width} must equal hidden {hidden}"
    );
    match mode {
        GlpMode::Project => {
            let d_norm_sq: f32 = direction
                .as_slice::<f32>()
                .context("GLP direction must be CPU-readable for norm computation")?
                .iter()
                .map(|x| x * x)
                .sum();
            if d_norm_sq <= f32::EPSILON {
                anyhow::bail!("GLP project direction has near-zero norm");
            }
            registry.register_source("glp_project_f32", PROJECT_SHADER);
            let params = GlpProjectParams {
                m: rows,
                n: hidden,
                alpha,
                d_norm_sq,
            };
            let pipeline = registry
                .get_pipeline("glp_project_f32", session.device().metal_device())
                .context("GLP projection pipeline")?;
            // Canonical launch contract (see the shader header): 256-lane
            // threadgroups, one per row, columns covered strided. The
            // earlier oversized grid (rows*hidden threads) worked only
            // because surplus groups no-op on `row >= m`; dispatch exactly
            // one group per row instead.
            let grid = metal::MTLSize::new((rows as u64) * 256, 1, 1);
            let tg = metal::MTLSize::new(256, 1, 1);
            encode_with_args(
                session.encoder_mut(),
                pipeline,
                &[
                    (0, KernelArg::Bytes(as_bytes(&params))),
                    (1, KernelArg::Buffer(direction)),
                    (2, KernelArg::Buffer(state)),
                ],
                grid,
                tg,
            );
            Ok(())
        }
        GlpMode::Add => {
            // In-session additive apply: `state += alpha * direction`
            // (row-broadcast). The additive path folds strength into the
            // data (spec) — bind uploaded the RAW direction, so scale it
            // here. Encoded into the session's encoder (no commit; the
            // session owns execution) via the same row-bias kernel the
            // out-of-session path uses. The caller's barrier discipline
            // (read-modify-write of `state` declared to the conflict
            // tracker) covers this dispatch exactly as it covers the
            // projection.
            let device = session.device().clone();
            let scaled: Vec<f32> = direction
                .as_slice::<f32>()
                .context("GLP direction must be CPU-readable for additive scaling")?
                .iter()
                .map(|x| x * alpha)
                .collect();
            let scaled_buf = {
                let byte_len = std::mem::size_of_val(scaled.as_slice());
                let raw = device.metal_device().new_buffer_with_data(
                    scaled.as_ptr().cast(),
                    byte_len as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                MlxBuffer::from_raw(raw, DType::F32, vec![scaled.len()])
            };
            mlx_native::ops::add_bias_row_2d::register(registry);
            mlx_native::ops::add_bias_row_2d::dispatch_add_bias_row_2d_f32(
                session.encoder_mut(),
                registry,
                device.metal_device(),
                state,
                &scaled_buf,
                state,
                rows,
                hidden,
            )
            .context("GLP in-session additive dispatch")?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    // Device-binding tests need Metal; they run behind the hardware gate.
    // The pure-Rust arithmetic contract is covered by apply.rs tests.
}

#[cfg(test)]
mod kernel_hardware_tests {
    use super::*;
    use mlx_native::MlxDevice;

    /// Same-boot, in-kernel verification: run the projection kernel on
    /// synthetic state, then read back and compare. Proves the dispatch
    /// writes what the math says.
    #[test]
    fn mhc_kernel_subtracts_scale_dot_direction() {
        let device = MlxDevice::new().expect("MlxDevice");
        let mut registry = mlx_native::KernelRegistry::new();

        let rows: u32 = 1;
        let hc: u32 = 4;
        let hidden: u32 = 4096;
        let alpha: f32 = 1.0;

        // direction = all-ones normalized (unit norm)
        let dir_host: Vec<f32> = (0..hidden)
            .map(|_| 1.0f32 / (hidden as f32).sqrt())
            .collect();
        // state slice 0 = [1,0,0,...], others 0
        let total = (rows * hc * hidden) as usize;
        let mut state_host = vec![0.0f32; total];
        // put a spike at row 0, stream 0, col 0
        state_host[0] = 1.0;

        let mut st_buf = device
            .alloc_buffer(state_host.len() * 4, DType::F32, vec![state_host.len()])
            .expect("alloc test state");
        st_buf
            .as_logical_mut_slice::<f32>()
            .expect("write test state")
            .copy_from_slice(&state_host);
        let dir_buf = {
            let raw = device.metal_device().new_buffer_with_data(
                dir_host.as_ptr().cast(),
                (dir_host.len() * 4) as u64,
                mlx_native::metal::MTLResourceOptions::StorageModeShared,
            );
            let n = dir_host.len();
            MlxBuffer::from_raw(raw, DType::F32, vec![n])
        };


        apply_layer_gpu_mhc(
            &mut st_buf,
            &dir_buf,
            GlpMode::Project,
            alpha,
            &device,
            &mut registry,
            rows,
            hc,
            hidden,
        )
        .unwrap();

        // expected: dot = dir[0]*1.0 = 1/sqrt(hidden) ≈ 0.0156
        // scale = 1*dot/norm^2; norm^2 = 1.0
        // new val = 1.0 - scale*dir[0] ≈ 1 - 0.0156*0.0156 = 0.999756
        let out = st_buf.as_slice::<f32>().unwrap();
        let expected: f32 = 1.0 - (1.0 / (hidden as f32).sqrt().powi(2));
        assert!(
            (out[0] - expected).abs() < 1e-4,
            "kernel wrong: out[0]={} expected≈{}",
            out[0],
            expected
        );
    }

    /// Same proof for the DENSE (FFN-site) kernel: a real GraphSession, the
    /// same path the DeepSeek FFN hook uses. If this passes while the serve
    /// path no-ops, the bug is plumbing (buffer identity/order), not shader.
    #[test]
    fn dense_kernel_subtracts_scale_dot_direction() {
        let device = MlxDevice::new().expect("MlxDevice");
        let mut registry = mlx_native::KernelRegistry::new();

        let rows: u32 = 2;
        let hidden: u32 = 4096;
        let alpha: f32 = 1.0;

        let dir_host: Vec<f32> = (0..hidden)
            .map(|_| 1.0f32 / (hidden as f32).sqrt())
            .collect();
        let total = (rows * hidden) as usize;
        let mut state_host = vec![0.0f32; total];
        state_host[0] = 1.0; // spike at row 0, col 0

        let mut st_buf = device
            .alloc_buffer(total * 4, DType::F32, vec![total])
            .expect("alloc test state");
        st_buf
            .as_logical_mut_slice::<f32>()
            .expect("write test state")
            .copy_from_slice(&state_host);
        let dir_buf = {
            let raw = device.metal_device().new_buffer_with_data(
                dir_host.as_ptr().cast(),
                (dir_host.len() * 4) as u64,
                mlx_native::metal::MTLResourceOptions::StorageModeShared,
            );
            let n = dir_host.len();
            MlxBuffer::from_raw(raw, DType::F32, vec![n])
        };

        let executor = mlx_native::graph::GraphExecutor::new(device.clone());
        let mut session = executor.begin().expect("begin session");
        apply_layer_gpu_in_session(
            &mut session,
            &mut registry,
            &st_buf,
            &dir_buf,
            GlpMode::Project,
            alpha,
            rows,
            hidden,
        )
        .unwrap();
        session.finish().expect("commit session");

        let out = st_buf.as_slice::<f32>().unwrap();
        let d0 = 1.0f32 / (hidden as f32).sqrt();
        let expected: f32 = 1.0 - d0 * d0; // dot = d0; scale = dot/1
        assert!(
            (out[0] - expected).abs() < 1e-4,
            "dense kernel wrong: out[0]={} expected≈{} (untouched would be 1.0)",
            out[0],
            expected
        );
        // row 1 was all zeros and must stay zero (dot=0)
        assert!(
            out[hidden as usize].abs() < 1e-6,
            "row 1 changed despite zero dot: {}",
            out[hidden as usize]
        );
    }
}

#[cfg(test)]
mod projection_reference_tests {
    use super::*;

    /// S1 required proof: the out-of-session projection (the Qwen path)
    /// must agree with the numerically sound CPU reference at actual model
    /// widths and multiple row counts, with a nontrivial direction, at
    /// zero dose (identity) and a live dose. The previous `tg = h` launch
    /// corrupted the 256-lane reduction for any width > 256.
    #[test]
    fn qwen_path_projection_matches_cpu_reference_at_model_widths() {
        let device = MlxDevice::new().expect("MlxDevice");
        let mut registry = KernelRegistry::new();

        for rows in [1u32, 3, 17] {
            for width in [2048u32, 5120] {
                // Deterministic nontrivial direction and state.
                let direction: Vec<f32> = (0..width)
                    .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
                    .collect();
                let state: Vec<f32> = (0..(rows as usize * width as usize))
                    .map(|i| (((i * 7 + 3) % 31) as f32 - 15.0) * 0.1)
                    .collect();

                let dir_buf = {
                    let raw = device.metal_device().new_buffer_with_data(
                        direction.as_ptr().cast(),
                        std::mem::size_of_val(direction.as_slice()) as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    );
                    MlxBuffer::from_raw(raw, DType::F32, vec![direction.len()])
                };

                // Zero dose: alpha = 0 must be the exact identity.
                let mut hidden = {
                    let raw = device.metal_device().new_buffer_with_data(
                        state.as_ptr().cast(),
                        std::mem::size_of_val(state.as_slice()) as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    );
                    MlxBuffer::from_raw(raw, DType::F32, vec![state.len()])
                };
                apply_layer_gpu(
                    &mut hidden,
                    &dir_buf,
                    GlpMode::Project,
                    0.0,
                    &device,
                    &mut registry,
                    rows,
                    width,
                )
                .expect("zero-dose dispatch");
                let out = hidden.as_slice::<f32>().unwrap();
                for (a, b) in out.iter().zip(state.iter()) {
                    assert_eq!(a, b, "zero dose must be the exact identity");
                }

                // Live dose: compare against the CPU reference row by row.
                let alpha = 1.7f32;
                let mut hidden = {
                    let raw = device.metal_device().new_buffer_with_data(
                        state.as_ptr().cast(),
                        std::mem::size_of_val(state.as_slice()) as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    );
                    MlxBuffer::from_raw(raw, DType::F32, vec![state.len()])
                };
                apply_layer_gpu(
                    &mut hidden,
                    &dir_buf,
                    GlpMode::Project,
                    alpha,
                    &device,
                    &mut registry,
                    rows,
                    width,
                )
                .expect("live-dose dispatch");
                let gpu = hidden.as_slice::<f32>().unwrap().to_vec();

                let mut cpu = state.clone();
                for row in 0..rows as usize {
                    let slice =
                        &mut cpu[row * width as usize..(row + 1) * width as usize];
                    super::super::apply::apply_layer_project(slice, &direction, alpha)
                        .expect("cpu reference");
                }
                let mut max_rel: f32 = 0.0;
                for (g, c) in gpu.iter().zip(cpu.iter()) {
                    let denom = c.abs().max(1e-3);
                    max_rel = max_rel.max((g - c).abs() / denom);
                }
                assert!(
                    max_rel < 1e-4,
                    "GPU projection must match the CPU reference \
                     (rows={rows}, width={width}, max_rel={max_rel})"
                );
            }
        }
    }
}

#[cfg(test)]
mod mhc_reference_tests {
    use super::*;

    /// S13 required proof: the mHC helpers (currently unused in production
    /// — validated before any future use) must match a CPU reference across
    /// rows and streams, with per-stream directions whose norms DIFFER (the
    /// case the old stream-0-norm-for-every-stream code mis-scaled) and
    /// with the shared-direction form.
    #[test]
    fn mhc_projection_matches_cpu_reference_across_rows_and_streams() {
        let device = MlxDevice::new().expect("MlxDevice");
        let mut registry = KernelRegistry::new();

        let rows = 3u32;
        let hc = 4u32;
        let hidden = 1024u32;
        let alpha = 1.3f32;

        // Per-stream directions with deliberately unequal norms.
        let mut direction: Vec<f32> = Vec::with_capacity((hc * hidden) as usize);
        for stream in 0..hc {
            let scale = (stream + 1) as f32 * 0.25; // norms differ per stream
            for col in 0..hidden {
                let v = scale * (((col % 11) as f32) - 5.0) * 0.1;
                direction.push(v);
            }
        }
        let state: Vec<f32> = (0..(rows * hc * hidden))
            .map(|i| (((i * 13 + 5) % 29) as f32 - 14.0) * 0.1)
            .collect();

        let dir_buf = {
            let raw = device.metal_device().new_buffer_with_data(
                direction.as_ptr().cast(),
                std::mem::size_of_val(direction.as_slice()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            MlxBuffer::from_raw(raw, DType::F32, vec![direction.len()])
        };
        let mut state_buf = {
            let raw = device.metal_device().new_buffer_with_data(
                state.as_ptr().cast(),
                std::mem::size_of_val(state.as_slice()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            MlxBuffer::from_raw(raw, DType::F32, vec![state.len()])
        };

        apply_layer_gpu_mhc(
            &mut state_buf,
            &dir_buf,
            GlpMode::Project,
            alpha,
            &device,
            &mut registry,
            rows,
            hc,
            hidden,
        )
        .expect("mhc dispatch");
        let gpu = state_buf.as_slice::<f32>().unwrap().to_vec();

        // CPU reference: independent per-stream projection.
        let mut cpu = state.clone();
        for row in 0..rows as usize {
            for stream in 0..hc as usize {
                let base = (row * hc as usize + stream) * hidden as usize;
                let slice = &mut cpu[base..base + hidden as usize];
                let d = &direction[stream * hidden as usize..(stream + 1) * hidden as usize];
                super::super::apply::apply_layer_project(slice, d, alpha)
                    .expect("cpu reference");
            }
        }
        let mut max_rel: f32 = 0.0;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            let denom = c.abs().max(1e-3);
            max_rel = max_rel.max((g - c).abs() / denom);
        }
        assert!(
            max_rel < 1e-4,
            "mHC per-stream projection must match the CPU reference (max_rel={max_rel})"
        );

        // Shared-direction form: one direction steers every stream.
        let shared: Vec<f32> = direction[..hidden as usize].to_vec();
        let shared_buf = {
            let raw = device.metal_device().new_buffer_with_data(
                shared.as_ptr().cast(),
                std::mem::size_of_val(shared.as_slice()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            MlxBuffer::from_raw(raw, DType::F32, vec![shared.len()])
        };
        let mut state_buf = {
            let raw = device.metal_device().new_buffer_with_data(
                state.as_ptr().cast(),
                std::mem::size_of_val(state.as_slice()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            MlxBuffer::from_raw(raw, DType::F32, vec![state.len()])
        };
        apply_layer_gpu_mhc(
            &mut state_buf,
            &shared_buf,
            GlpMode::Project,
            alpha,
            &device,
            &mut registry,
            rows,
            hc,
            hidden,
        )
        .expect("mhc shared dispatch");
        let gpu = state_buf.as_slice::<f32>().unwrap().to_vec();
        let mut cpu = state.clone();
        for row in 0..rows as usize {
            for stream in 0..hc as usize {
                let base = (row * hc as usize + stream) * hidden as usize;
                let slice = &mut cpu[base..base + hidden as usize];
                super::super::apply::apply_layer_project(slice, &shared, alpha)
                    .expect("cpu reference");
            }
        }
        let mut max_rel: f32 = 0.0;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            let denom = c.abs().max(1e-3);
            max_rel = max_rel.max((g - c).abs() / denom);
        }
        assert!(
            max_rel < 1e-4,
            "mHC shared-direction projection must match the CPU reference (max_rel={max_rel})"
        );
    }
}
