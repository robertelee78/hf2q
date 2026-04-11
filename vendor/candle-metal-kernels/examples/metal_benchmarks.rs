use anyhow::Result;
use candle_metal_kernels::{
    metal::{create_command_buffer, CommandSemaphore, Device},
    source::Source,
    GemmDType, Kernels, RESOURCE_OPTIONS,
};
/// This example contains some simple benchmarks so that it's easy to run them in perf etc.
use clap::{Parser, Subcommand};
use half::f16;
use objc2_metal::MTLSize;
use std::sync::Arc;

fn run_gemm(f32: bool, n: usize) -> Result<()> {
    const WARMUP_ITERS: usize = 2;
    const MIN_DUR: f64 = 4.;

    let device = Device::system_default().unwrap();

    let (b, m, n, k) = (1, n, n, n);
    let kernels = candle_metal_kernels::Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let options = RESOURCE_OPTIONS;

    let (lhs, rhs) = if f32 {
        let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
        let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
        let lhs = device
            .new_buffer_with_data(
                lhs.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(&lhs),
                options,
            )
            .unwrap();
        let rhs = device
            .new_buffer_with_data(
                rhs.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(&rhs),
                options,
            )
            .unwrap();
        (lhs, rhs)
    } else {
        let lhs: Vec<f16> = (0..b * m * k).map(|f| f16::from_f32(f as f32)).collect();
        let rhs: Vec<f16> = (0..b * n * k).map(|f| f16::from_f32(f as f32)).collect();
        let lhs = device
            .new_buffer_with_data(
                lhs.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(&lhs),
                options,
            )
            .unwrap();
        let rhs = device
            .new_buffer_with_data(
                rhs.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(&rhs),
                options,
            )
            .unwrap();
        (lhs, rhs)
    };
    let (dtype, sizeof) = if f32 {
        (GemmDType::F32, core::mem::size_of::<f32>())
    } else {
        (GemmDType::F16, core::mem::size_of::<f16>())
    };
    let output = device.new_buffer(b * m * n * sizeof, options).unwrap();

    let mut sum_dt = 0f64;
    let mut iters = 0usize;
    for idx in 0.. {
        let semaphore = Arc::new(CommandSemaphore::new());
        let command_buffer = create_command_buffer(&command_queue, semaphore).unwrap();
        let start_time = std::time::Instant::now();
        candle_metal_kernels::call_mlx_gemm(
            &device,
            &command_buffer,
            &kernels,
            dtype,
            (b, m, n, k),
            &[m * k, k, 1],
            0,
            &lhs,
            &[n * k, n, 1],
            0,
            &rhs,
            &output,
        )?;
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let dt = start_time.elapsed().as_secs_f64();
        if idx < WARMUP_ITERS {
            continue;
        }
        sum_dt += dt;
        iters += 1;
        if sum_dt > MIN_DUR {
            break;
        }
    }
    let gflops = (2 * n * n * n * iters) as f64 / (1e9 * sum_dt);
    println!("{dtype:?},      {n:6}      gflops {gflops:.0}");

    Ok(())
}

// =============================================================================
// 1bNEW.29 NSG sweep microbenchmark
//
// Times the candle Metal Q-matmul kernels at the exact dispatch shapes hf2q
// runs in production for the Gemma 4 26B MoE forward, with N_SIMDGROUP swept
// across {1, 2, 4, 8} via dedicated kernel variants we added in
// vendor/candle-metal-kernels/src/metal_src/quantized.metal:
//   - kernel_mul_mv_q4_0_f32_nsg{1,2,4,8}
//   - kernel_mul_mv_q6_K_f32_nsg{1,2,4,8}
//
// nsg=2 corresponds to candle's production constant `#define N_SIMDGROUP 2`
// and serves as the apples-to-apples baseline for the sweep.
//
// We bypass `call_quantized_matmul_mv_t` (which is hardcoded for nsg=2 launch
// geometry) and dispatch the variant kernels manually with the matching
// threadgroup shape:
//   * Q4_0: N_DST=4 rows per simdgroup, so threadgroups along the row
//           dimension = ceil(n / (4*nsg)). Total threads/threadgroup = 32*nsg.
//   * Q6_K: 1 row per simdgroup, so threadgroups along the row dimension
//           = ceil(n / nsg). Total threads/threadgroup = 32*nsg.
//
// This is the "measure first" step for ADR-005 1bNEW.29 (see
// docs/spike-1bNEW29-nsg-sweep-data.md).
// =============================================================================

#[derive(Debug, Clone, Copy)]
enum QType {
    Q4_0,
    Q6K,
}

impl QType {
    fn block_bytes(&self) -> usize {
        match self {
            // sizeof(half) + QK4_0/2 = 2 + 16 = 18
            QType::Q4_0 => 18,
            // sizeof(half) + QK_K/16 + 3*QK_K/4 = 2 + 16 + 192 = 210
            QType::Q6K => 210,
        }
    }
    fn block_k(&self) -> usize {
        match self {
            QType::Q4_0 => 32,
            QType::Q6K => 256,
        }
    }
    fn n_dst(&self) -> usize {
        match self {
            // mul_vec_q_n_f32_impl uses N_DST=4 rows per simdgroup.
            QType::Q4_0 => 4,
            // kernel_mul_mv_q6_K_f32_impl uses 1 row per simdgroup.
            QType::Q6K => 1,
        }
    }
    fn label(&self) -> &'static str {
        match self {
            QType::Q4_0 => "Q4_0",
            QType::Q6K => "Q6_K",
        }
    }
    fn kernel_name(&self, nsg: usize) -> String {
        match self {
            QType::Q4_0 => format!("kernel_mul_mv_q4_0_f32_nsg{nsg}"),
            QType::Q6K => format!("kernel_mul_mv_q6_K_f32_nsg{nsg}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Shape {
    label: &'static str,
    qtype: QType,
    /// output rows of the weight matrix (n in (b, m, n, k))
    n: usize,
    /// input cols of the weight matrix (k in (b, m, n, k))
    k: usize,
}

const NSG_VALUES: [usize; 4] = [1, 2, 4, 8];

fn nsg_microbench_shapes() -> Vec<Shape> {
    // From docs/ADR-005-inference-server.md:906–913 — exact dispatch shapes
    // hf2q runs in production for Gemma 4 26B MoE.
    vec![
        Shape { label: "MLP gate_proj          [1,2816]@[2112,2816] Q4_0", qtype: QType::Q4_0, n: 2112, k: 2816 },
        Shape { label: "MLP down_proj          [1,2112]@[2816,2112] Q4_0", qtype: QType::Q4_0, n: 2816, k: 2112 },
        Shape { label: "Attn q_proj sliding    [1,2816]@[4096,2816] Q6_K", qtype: QType::Q6K,  n: 4096, k: 2816 },
        Shape { label: "Attn q_proj global     [1,2816]@[8192,2816] Q6_K", qtype: QType::Q6K,  n: 8192, k: 2816 },
        Shape { label: "Attn k_proj sliding    [1,2816]@[2048,2816] Q6_K", qtype: QType::Q6K,  n: 2048, k: 2816 },
        Shape { label: "Attn k_proj global     [1,2816]@[1024,2816] Q6_K", qtype: QType::Q6K,  n: 1024, k: 2816 },
        Shape { label: "Attn o_proj sliding    [1,4096]@[2816,4096] Q6_K", qtype: QType::Q6K,  n: 2816, k: 4096 },
        Shape { label: "Attn o_proj global     [1,8192]@[2816,8192] Q6_K", qtype: QType::Q6K,  n: 2816, k: 8192 },
    ]
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Run a single (shape, nsg) microbench. Returns (median_ns, mean_ns, samples).
/// Uses 1000 timed iterations after WARMUP iterations of warmup, with one
/// MTLCommandBuffer commit + waitUntilCompleted per dispatch (sync between
/// every call) so we are timing the per-dispatch end-to-end cost.
fn run_one_shape_nsg(
    device: &Device,
    kernels: &Kernels,
    queue: &candle_metal_kernels::metal::CommandQueue,
    shape: &Shape,
    nsg: usize,
    iters: usize,
    warmup: usize,
) -> Result<(u128, u128)> {
    let qtype = shape.qtype;
    let (n, k) = (shape.n, shape.k);
    let m = 1usize;
    let b = 1usize;

    // Validate alignment (should hold for all listed shapes; otherwise the
    // Q6_K nsg-templated impl would write OOB or the Q4_0 impl's per-thread
    // OOB guard would silently leave rows unwritten).
    let align = qtype.n_dst() * nsg;
    if n % align != 0 {
        anyhow::bail!(
            "shape '{}' n={} not divisible by N_DST*nsg={} (qtype={:?}, nsg={})",
            shape.label, n, align, qtype, nsg
        );
    }

    // Allocate buffers.
    assert_eq!(k % qtype.block_k(), 0, "k must be a multiple of block_k");
    let blocks_per_row = k / qtype.block_k();
    let weight_bytes = n * blocks_per_row * qtype.block_bytes();
    let lhs_bytes = m * k * std::mem::size_of::<f32>();
    let dst_bytes = m * n * std::mem::size_of::<f32>();

    let weight = device.new_buffer(weight_bytes, RESOURCE_OPTIONS)?;
    let lhs = device.new_buffer(lhs_bytes, RESOURCE_OPTIONS)?;
    let dst = device.new_buffer(dst_bytes, RESOURCE_OPTIONS)?;

    // Fill with deterministic small-magnitude data so the kernel does real
    // arithmetic on real bytes (not just zeros, which can be optimized away
    // by aggressive code motion in some Metal compilers). We treat the
    // weight buffer as raw bytes and populate it with a low-entropy pattern;
    // this isn't a valid GGUF block layout per se, but for the timing
    // microbench all that matters is that the kernel touches every byte.
    unsafe {
        let w_ptr = weight.contents();
        for i in 0..weight_bytes {
            *w_ptr.add(i) = ((i.wrapping_mul(2654435761)) & 0xFF) as u8;
        }
        let l_ptr = lhs.contents() as *mut f32;
        for i in 0..(m * k) {
            *l_ptr.add(i) = ((i & 0xFF) as f32) * 1e-3;
        }
    }

    // Resolve & cache the pipeline state once.
    let kernel_name = qtype.kernel_name(nsg);
    let pipeline = kernels
        .load_pipeline(device, Source::Quantized, kernel_name.clone())
        .map_err(|e| anyhow::anyhow!("load_pipeline({kernel_name}) failed: {e:?}"))?;

    // Pack scalar args.
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = 1;
    let r3: u32 = 1;

    // Threadgroup geometry.
    let threadgroup_count = MTLSize {
        width: ceil_div(n, qtype.n_dst() * nsg),
        height: m,
        depth: b,
    };
    let threads_per_threadgroup = MTLSize {
        width: 32 * nsg,
        height: 1,
        depth: 1,
    };

    // Closure that runs ONE timed iteration: build a fresh command buffer,
    // encode + dispatch + commit + waitUntilCompleted, return elapsed ns.
    let dispatch_once = || -> Result<u128> {
        let semaphore = Arc::new(CommandSemaphore::new());
        let cb = create_command_buffer(queue, semaphore)?;
        let encoder = cb.compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&weight), 0);
        encoder.set_buffer(1, Some(&lhs), 0);
        encoder.set_buffer(2, Some(&dst), 0);
        encoder.set_bytes(3, &ne00);
        encoder.set_bytes(4, &ne01);
        encoder.set_bytes(5, &ne02);
        encoder.set_bytes(6, &nb00);
        encoder.set_bytes(7, &nb01);
        encoder.set_bytes(8, &nb02);
        encoder.set_bytes(9, &ne10);
        encoder.set_bytes(10, &ne11);
        encoder.set_bytes(11, &ne12);
        encoder.set_bytes(12, &nb10);
        encoder.set_bytes(13, &nb11);
        encoder.set_bytes(14, &nb12);
        encoder.set_bytes(15, &ne0);
        encoder.set_bytes(16, &ne1);
        encoder.set_bytes(17, &r2);
        encoder.set_bytes(18, &r3);
        use objc2_metal::MTLResourceUsage;
        encoder.use_resource(&weight, MTLResourceUsage::Read);
        encoder.use_resource(&lhs, MTLResourceUsage::Read);
        encoder.use_resource(&dst, MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(threadgroup_count, threads_per_threadgroup);
        encoder.end_encoding();
        let t0 = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        Ok(t0.elapsed().as_nanos())
    };

    // Warmup.
    for _ in 0..warmup {
        let _ = dispatch_once()?;
    }

    // Timed iterations.
    let mut samples: Vec<u128> = Vec::with_capacity(iters);
    for _ in 0..iters {
        samples.push(dispatch_once()?);
    }
    samples.sort_unstable();
    let median = samples[iters / 2];
    let mean = samples.iter().copied().sum::<u128>() / (iters as u128);
    Ok((median, mean))
}

fn run_nsg_microbench(iters: usize, warmup: usize) -> Result<()> {
    let device = Device::system_default().expect("no Metal device");
    let kernels = Kernels::new();
    let queue = device.new_command_queue()?;

    println!("# 1bNEW.29 NSG sweep microbench");
    println!("# device:        {}", device.architecture_name());
    println!("# device_type:   {:?}", device.device_type());
    println!("# iters/sample:  {iters}");
    println!("# warmup:        {warmup}");
    println!("# methodology:   per-dispatch commit+waitUntilCompleted, fresh");
    println!("#                CommandBuffer per call, median across iters.");
    println!();
    println!(
        "{:<48} {:>9} {:>11} {:>11} {:>11} {:>11}",
        "shape", "type", "nsg=1 µs", "nsg=2 µs", "nsg=4 µs", "nsg=8 µs"
    );
    println!("{:-<104}", "");

    let shapes = nsg_microbench_shapes();
    let mut all_results: Vec<(Shape, [u128; 4])> = Vec::new();

    for shape in &shapes {
        let mut row_med: [u128; 4] = [0; 4];
        for (i, &nsg) in NSG_VALUES.iter().enumerate() {
            // Skip illegal alignments rather than abort the whole sweep.
            let align = shape.qtype.n_dst() * nsg;
            if shape.n % align != 0 {
                row_med[i] = u128::MAX;
                continue;
            }
            let (med, _mean) =
                run_one_shape_nsg(&device, &kernels, &queue, shape, nsg, iters, warmup)?;
            row_med[i] = med;
        }
        let fmt = |ns: u128| {
            if ns == u128::MAX {
                "  -- ".to_string()
            } else {
                format!("{:.2}", ns as f64 / 1000.0)
            }
        };
        println!(
            "{:<48} {:>9} {:>11} {:>11} {:>11} {:>11}",
            shape.label,
            shape.qtype.label(),
            fmt(row_med[0]),
            fmt(row_med[1]),
            fmt(row_med[2]),
            fmt(row_med[3]),
        );
        all_results.push((*shape, row_med));
    }

    // Per-shape verdict and overall summary.
    println!();
    println!("# per-shape verdict (smallest µs/call wins; baseline is nsg=2)");
    println!(
        "{:<48} {:>9} {:>10} {:>14}",
        "shape", "type", "best nsg", "speedup vs nsg=2"
    );
    println!("{:-<83}", "");
    let mut total_speedup = 0.0_f64;
    let mut counted = 0;
    for (shape, meds) in &all_results {
        let mut best_idx = 1usize; // start at nsg=2
        let mut best = meds[1];
        for (i, &m) in meds.iter().enumerate() {
            if m < best {
                best = m;
                best_idx = i;
            }
        }
        let baseline = meds[1] as f64;
        let speedup = if best == 0 { 1.0 } else { baseline / best as f64 };
        println!(
            "{:<48} {:>9} {:>10} {:>13.3}x",
            shape.label,
            shape.qtype.label(),
            NSG_VALUES[best_idx],
            speedup,
        );
        if best_idx != 1 {
            total_speedup += speedup;
            counted += 1;
        }
    }
    if counted == 0 {
        println!();
        println!("# verdict: nsg=2 optimal across ALL shapes — hypothesis falsified.");
    } else {
        println!();
        println!(
            "# verdict: {counted}/{} shapes have a non-default optimum; mean speedup {:.3}x.",
            all_results.len(),
            total_speedup / counted as f64,
        );
    }
    Ok(())
}

#[derive(Subcommand, Debug, Clone)]
enum Task {
    Gemm,
    /// 1bNEW.29 NSG sweep microbench at hf2q's exact dispatch shapes.
    NsgMicrobench {
        /// Number of timed iterations per (shape, nsg) cell.
        #[arg(long, default_value_t = 1000)]
        iters: usize,
        /// Number of untimed warmup iterations per cell.
        #[arg(long, default_value_t = 50)]
        warmup: usize,
    },
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The benchmark to be run.
    #[command(subcommand)]
    task: Task,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.task {
        Task::Gemm => {
            for f32 in [false, true] {
                for n in [512, 1024, 2048, 4096] {
                    run_gemm(f32, n)?;
                }
            }
        }
        Task::NsgMicrobench { iters, warmup } => {
            run_nsg_microbench(iters, warmup)?;
        }
    }
    Ok(())
}
