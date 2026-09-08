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
    d_norm_sq: f32,
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
    // one threadgroup per row; tg width = n (hidden width, ≤ 2048 for our
    // families) so the threadgroup reduction spans exactly one row.
    let grid = metal::MTLSize::new((seq_len as u64) * (h as u64), 1, 1);
    let tg = metal::MTLSize::new(h as u64, 1, 1);
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
    let d_norm_sq: f32 = direction
        .as_slice::<f32>()
        .context("GLP direction must be CPU-readable for norm computation")?
        .iter()
        .take(hidden as usize)
        .map(|x| x * x)
        .sum();
    if d_norm_sq <= f32::EPSILON {
        anyhow::bail!("GLP mHC project direction has near-zero norm");
    }
    registry.register_source("glp_project_mhc_f32", PROJECT_MHC_SHADER);
    let params = GlpProjectMhcParams {
        rows,
        hc,
        hidden,
        per_stream_directions: per_stream,
        alpha,
        d_norm_sq,
    };
    let pipeline = registry
        .get_pipeline("glp_project_mhc_f32", device.metal_device())
        .context("GLP mHC projection pipeline")?;
    // One threadgroup per (row, stream) pair; threadgroup strided at the
    // Metal-safe 256 threads (the dense kernel's same pattern).
    let grid = metal::MTLSize::new((rows as u64) * (hc as u64), 1, 1);
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
    let d_norm_sq: f32 = direction
        .as_slice::<f32>()
        .context("GLP direction must be CPU-readable for norm computation")?
        .iter()
        .take(hidden as usize)
        .map(|x| x * x)
        .sum();
    if d_norm_sq <= f32::EPSILON {
        anyhow::bail!("GLP mHC project direction has near-zero norm");
    }
    registry.register_source("glp_project_mhc_f32", PROJECT_MHC_SHADER);
    let params = GlpProjectMhcParams {
        rows,
        hc,
        hidden,
        per_stream_directions: per_stream,
        alpha,
        d_norm_sq,
    };
    let pipeline = registry
        .get_pipeline("glp_project_mhc_f32", session.device().metal_device())
        .context("GLP mHC projection pipeline")?;
    let grid = metal::MTLSize::new((rows as u64) * (hc as u64), 1, 1);
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

/// In-session dense variant: project on a `[rows, hidden]` F32 buffer
/// (the FFN-writer site on DeepSeek; the post-layer residual on dense
/// families). No commit here — the session owner does that.
pub fn apply_layer_gpu_in_session(
    session: &mut mlx_native::graph::GraphSession<'_>,
    registry: &mut KernelRegistry,
    state: &MlxBuffer,
    direction: &MlxBuffer,
    alpha: f32,
    rows: u32,
    hidden: u32,
) -> Result<()> {
    let dir_width = direction.shape().first().copied().unwrap_or(0) as u32;
    anyhow::ensure!(
        dir_width == hidden,
        "GLP dense direction width {dir_width} must equal hidden {hidden}"
    );
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
    let grid = metal::MTLSize::new((rows as u64) * (hidden as u64), 1, 1);
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
}
