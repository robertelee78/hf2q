//! Shared timing and synthetic-data helpers for the DeepSeek-V4 gap-program microbenches
//! (M2a `ds4_op_microbench_tests.rs`, M2b `ds4_attn_microbench_tests.rs`).
//!
//! Timing: one trial encodes `reps` back-to-back copies of an op in one `GraphSession` and reads
//! the command buffer's GPU timestamps (`finish_with_gpu_time`, `GPUEndTime - GPUStartTime`); the
//! cell is the median over `TRIALS` trials of GPU time / `reps`. `reps` is calibrated per cell so
//! one session runs >= `SESSION_GPU_SECS` of GPU work (`MIN_REPS..=MAX_REPS`), which keeps the
//! GPU's per-command-buffer start-up out of the per-copy figure. Wall time (commit + wait) per copy
//! and host encode time per copy are recorded alongside. Untimed sessions run for >= `WARM_SECS`
//! before the timed ones so each cell sees the sustained GPU clock after host-side setup.
//! Cold cells rotate weights as `copies[rep % copies.len()]` with `copies.len() <= MIN_REPS`, so every
//! copy is read in every trial.

use std::io::Write;
use std::time::Instant;

use anyhow::Result;
use mlx_native::graph::GraphSession;
use mlx_native::ops::elementwise::scalar_mul_f32;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};
use serde_json::{json, Value};

use super::forward_support::{alloc, raw_matmul};
use super::joint_decode_rows::AttnCores;
use super::residency::{MatvecKernels, RawMatrixRef};
use crate::serve::gpu::GpuContext;

pub(super) const TRIALS: usize = 9;
/// Smallest copy count per session; cold-copy sets never exceed it.
pub(super) const MIN_REPS: usize = 64;
pub(super) const MAX_REPS: usize = 8192;
pub(super) const SESSION_GPU_SECS: f64 = 2e-3;
pub(super) const WARM_SECS: f64 = 0.05;
/// Cold cells rotate over enough weight copies to exceed this many bytes (capped at `MIN_REPS`).
pub(super) const COLD_BYTES: usize = 256 << 20;

pub(super) struct Rng(pub u64);
impl Rng {
    pub(super) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    pub(super) fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Few-row kernel family for every bench weight: `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC`, read the way
/// the model loader reads it, so one run times one family (each cell records it).
pub(super) fn bench_matvec() -> MatvecKernels {
    static MATVEC: std::sync::OnceLock<MatvecKernels> = std::sync::OnceLock::new();
    *MATVEC.get_or_init(|| {
        MatvecKernels::from_env()
            .expect("HF2Q_DEEPSEEK_ROW_EXACT_MATVEC")
            .value
    })
}

/// Joint-decode attention cores for the M2b cells: `HF2Q_DEEPSEEK_BATCHED_ATTN_CORES`, read the
/// way the model loader reads it (each cell records it).
pub(super) fn bench_cores() -> AttnCores {
    static CORES: std::sync::LazyLock<AttnCores> = std::sync::LazyLock::new(|| {
        AttnCores::from_env()
            .expect("HF2Q_DEEPSEEK_BATCHED_ATTN_CORES")
            .value
    });
    *CORES
}

/// Copies needed for `bytes` per copy to exceed `COLD_BYTES` (1 when warm), capped at `MIN_REPS`.
pub(super) fn copies_for(bytes: usize, warm: bool) -> usize {
    if warm {
        1
    } else {
        COLD_BYTES.div_ceil(bytes).clamp(1, MIN_REPS)
    }
}

pub(super) fn f16_matrix(device: &MlxDevice, n: usize, k: usize, seed: u64) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(n * k * 2, DType::F16, vec![n, k])
        .unwrap();
    let mut rng = Rng(seed);
    for v in buf.as_mut_slice::<u16>().unwrap() {
        *v = half::f16::from_f32(rng.unit() * 0.025).to_bits();
    }
    buf
}

pub(super) fn f32_buf(device: &MlxDevice, shape: Vec<usize>, seed: u64, scale: f32) -> MlxBuffer {
    let mut buf = alloc(device, DType::F32, shape, "bench f32").unwrap();
    let mut rng = Rng(seed);
    for v in buf.as_mut_slice::<f32>().unwrap() {
        *v = rng.unit() * scale;
    }
    buf
}

pub(super) fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

pub(super) struct Timing {
    /// Median GPU-timestamp µs per copy.
    pub us: f64,
    /// Median commit + wait wall µs per copy.
    pub wall_us: f64,
    /// Median host encode µs per copy.
    pub encode_us: f64,
    pub reps: usize,
}

impl Timing {
    pub(super) fn fields(&self) -> Value {
        json!({"us": self.us, "wall_us": self.wall_us, "encode_us": self.encode_us, "reps": self.reps,
               "trials": TRIALS, "timer": "gpu_timestamps"})
    }
}

struct Trial {
    gpu: f64,
    wall: f64,
    encode: f64,
}

fn run_trial(
    ctx: &mut GpuContext,
    device: &MlxDevice,
    reps: usize,
    encode: &mut impl FnMut(&mut GraphSession<'_>, &mut KernelRegistry, &MlxDevice, usize) -> Result<()>,
) -> Trial {
    let (executor, registry) = ctx.split();
    let mut session = executor.begin().unwrap();
    let t0 = Instant::now();
    for rep in 0..reps {
        encode(&mut session, registry, device, rep).unwrap();
    }
    let t1 = Instant::now();
    let gpu_ns = session.finish_with_gpu_time().unwrap();
    Trial {
        gpu: gpu_ns as f64 * 1e-9,
        wall: t1.elapsed().as_secs_f64(),
        encode: (t1 - t0).as_secs_f64(),
    }
}

/// Time `encode` (called with the copy index within the session) as described in the module doc.
pub(super) fn time_op(
    ctx: &mut GpuContext,
    encode: impl FnMut(&mut GraphSession<'_>, &mut KernelRegistry, &MlxDevice, usize) -> Result<()>,
) -> Timing {
    time_op_reps(ctx, MIN_REPS, encode)
}

/// `time_op` with an explicit minimum copy count per session (prefill cells whose single copy
/// already runs for milliseconds use a small minimum; sessions still run >= SESSION_GPU_SECS).
pub(super) fn time_op_reps(
    ctx: &mut GpuContext,
    min_reps: usize,
    mut encode: impl FnMut(&mut GraphSession<'_>, &mut KernelRegistry, &MlxDevice, usize) -> Result<()>,
) -> Timing {
    let device = ctx.device().clone();
    // Calibration session (also builds pipelines): size reps for >= SESSION_GPU_SECS per session.
    let probe = run_trial(ctx, &device, min_reps, &mut encode);
    let per_copy = probe.gpu / min_reps as f64;
    let reps = if per_copy > 0.0 {
        ((SESSION_GPU_SECS / per_copy).ceil() as usize).clamp(min_reps, MAX_REPS)
    } else {
        MAX_REPS
    };
    let warm_start = Instant::now();
    while warm_start.elapsed().as_secs_f64() < WARM_SECS {
        run_trial(ctx, &device, reps, &mut encode);
    }
    let trials: Vec<Trial> = (0..TRIALS)
        .map(|_| run_trial(ctx, &device, reps, &mut encode))
        .collect();
    let per =
        |f: fn(&Trial) -> f64| median(trials.iter().map(|t| f(t) / reps as f64).collect()) * 1e6;
    Timing {
        us: per(|t| t.gpu),
        wall_us: per(|t| t.wall),
        encode_us: per(|t| t.encode),
        reps,
    }
}

/// Encode once and wait (untimed setup, e.g. populating top-k indices).
pub(super) fn run_once(
    ctx: &mut GpuContext,
    encode: impl FnOnce(&mut GraphSession<'_>, &mut KernelRegistry, &MlxDevice) -> Result<()>,
) {
    let device = ctx.device().clone();
    let (executor, registry) = ctx.split();
    let mut session = executor.begin().unwrap();
    encode(&mut session, registry, &device).unwrap();
    session.finish().unwrap();
}

/// Bring the GPU to its sustained clock before the first cell: cold F16 matvecs for >= 2 s.
pub(super) fn gpu_warmup(ctx: &mut GpuContext) {
    let device = ctx.device().clone();
    let (n, k) = (8192usize, 4096usize);
    let weights: Vec<MlxBuffer> = (0..4).map(|c| f16_matrix(&device, n, k, 900 + c)).collect();
    let shape = [n, k];
    let x = f32_buf(&device, vec![1, k], 901, 1.0);
    let y = alloc(&device, DType::F32, vec![1, n], "warmup out").unwrap();
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < 2.0 {
        time_op(ctx, |s, r, d, rep| {
            let w = RawMatrixRef {
                buffer: &weights[rep % weights.len()],
                ggml_type: GgmlType::F16,
                shape: &shape,
                matvec: bench_matvec(),
            };
            raw_matmul(s, r, d, &x, &w, &y, 1, n, k, "warmup")
        });
    }
}

/// Floor of the method: 1 or 16 dependent 1-element dispatches (barrier between each) per copy.
/// `us` of the x1 cell is the smallest per-copy GPU time any cell can report; `(x16 - x1) / 15` is
/// the GPU cost of one dependent dispatch; `wall_us - us` is the commit + wait share per copy.
pub(super) fn floor_cells(ctx: &mut GpuContext, out: &mut Out) {
    let device = ctx.device().clone();
    let x = f32_buf(&device, vec![1], 902, 1.0);
    let y = alloc(&device, DType::F32, vec![1], "floor out").unwrap();
    for dispatches in [1usize, 16] {
        let t = time_op(ctx, |s, r, d, _| {
            for _ in 0..dispatches {
                s.barrier_between(&[&x], &[&y]);
                scalar_mul_f32(s.encoder_mut(), r, d.metal_device(), &x, &y, 1, 2.0)?;
            }
            Ok(())
        });
        out.cell(
            &json!({"family": "floor", "b": 1}),
            json!({"op": format!("scalar_mul_x{dispatches}"), "dispatches_per_copy": dispatches}),
            &t,
        );
    }
}

pub(super) struct Out(pub std::fs::File);
impl Out {
    /// Write `base` + `fields` + the timing fields as one JSON line.
    pub(super) fn cell(&mut self, base: &Value, fields: Value, timing: &Timing) {
        let mut cell = base.clone();
        let target = cell.as_object_mut().unwrap();
        for extra in [fields, timing.fields()] {
            for (key, value) in extra.as_object().unwrap() {
                target.insert(key.clone(), value.clone());
            }
        }
        target.insert("matvec".into(), json!(format!("{:?}", bench_matvec())));
        target.insert("cores".into(), json!(format!("{:?}", bench_cores())));
        eprintln!("{cell}");
        writeln!(self.0, "{cell}").unwrap();
    }
}
