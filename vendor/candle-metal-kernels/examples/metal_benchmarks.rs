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

// =============================================================================
// 1bNEW.29 Option C — Q6_K NR0=2 microbench
//
// Times the candle Q6_K Metal matmul kernel (Variant A, production) vs a
// byte-for-byte port of llama.cpp's nr0=2 row-loop variant (Variant B, added
// to quantized.metal as `kernel_mul_mv_q6_K_f32_nr2`) at the exact dispatch
// shapes hf2q runs in production for Gemma 4 26B attention.
//
// Variant A is the existing `kernel_mul_mv_q6_K_f32` at
// vendor/candle-metal-kernels/src/metal_src/quantized.metal:5186-5294 —
// 1 row per simdgroup, dispatched as ne01/(1*NSG) = ne01/2 threadgroups.
//
// Variant B is the new `kernel_mul_mv_q6_K_f32_nr2` added after line 5294 —
// 2 rows per simdgroup (byte-for-byte port of llama.cpp's `nr0 = 2`), dispatched
// as ne01/(2*NSG) = ne01/4 threadgroups. Uses a real `nb01` byte stride to
// advance per-row src0 pointers inside the kernel; host must pass
// nb01 = sizeof(block_q6_K) * (k/QK_K).
//
// Decision rule: GO if Variant B is >=10% faster on any production shape AND
// correctness sanity check passes (max|Δ| <= 1e-5 relative). NO-GO otherwise.
//
// Methodology mirrors `run_one_shape_nsg` (NSG sweep from spike 1bNEW.29 Agent
// #1): per-dispatch commit+waitUntilCompleted, 1000 timed iters per cell,
// median across cross-run medians.
// =============================================================================

/// Byte-size of one Q6_K super-block (matches the Metal struct layout:
/// ql[128] + qh[64] + scales[16] + sizeof(half)=2 = 210 bytes).
const Q6K_BLOCK_BYTES: usize = 210;
/// Q6_K super-block element count (QK_K = 256).
const Q6K_QK_K: usize = 256;

#[derive(Debug, Clone, Copy)]
enum Q6KVariant {
    /// Existing production candle kernel: 1 row per simdgroup.
    ControlA,
    /// llama.cpp byte-for-byte port: 2 rows per simdgroup (nr0=2).
    Nr2VariantB,
}

impl Q6KVariant {
    fn kernel_name(self) -> &'static str {
        match self {
            Q6KVariant::ControlA => "kernel_mul_mv_q6_K_f32",
            Q6KVariant::Nr2VariantB => "kernel_mul_mv_q6_K_f32_nr2",
        }
    }
    fn label(self) -> &'static str {
        match self {
            Q6KVariant::ControlA => "A (nr0=1)",
            Q6KVariant::Nr2VariantB => "B (nr0=2)",
        }
    }
    /// Rows per simdgroup for this variant.
    fn nr0(self) -> usize {
        match self {
            Q6KVariant::ControlA => 1,
            Q6KVariant::Nr2VariantB => 2,
        }
    }
}

/// Shapes for the Q6_K microbench — reused from nsg_microbench_shapes, filtered
/// to Q6_K only. The 6 production shapes from docs/ADR-005-inference-server.md:
/// attention q_proj/k_proj/o_proj × {sliding, global}.
fn q6k_microbench_shapes() -> Vec<Shape> {
    nsg_microbench_shapes()
        .into_iter()
        .filter(|s| matches!(s.qtype, QType::Q6K))
        .collect()
}

/// Dispatch ONE timed iteration of a Q6_K variant at the given shape.
/// Returns elapsed wall-clock ns for a single commit + waitUntilCompleted.
#[allow(clippy::too_many_arguments)]
fn q6k_dispatch_once(
    queue: &candle_metal_kernels::metal::CommandQueue,
    pipeline: &candle_metal_kernels::metal::ComputePipeline,
    variant: Q6KVariant,
    weight: &candle_metal_kernels::metal::Buffer,
    lhs: &candle_metal_kernels::metal::Buffer,
    dst: &candle_metal_kernels::metal::Buffer,
    n: usize,
    k: usize,
) -> Result<u128> {
    let m = 1i64;
    let b = 1i64;

    // Real byte stride per row of Q6_K weights. For n×k with block_k=256, each
    // row has k/256 blocks × 210 bytes. Variant B requires this stride be
    // non-zero (it advances per-row pointers by it). Variant A ignores nb01
    // entirely, so passing the real stride to both is safe.
    let blocks_per_row = (k / Q6K_QK_K) as i64;
    let nb01_bytes = blocks_per_row * Q6K_BLOCK_BYTES as i64;

    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b;
    let nb00: i64 = 0;
    let nb02: i64 = 0;
    let ne10 = k as i64;
    let ne11 = m;
    let ne12 = b;
    let nb10: i64 = 0;
    let nb11: i64 = 0;
    let nb12: i64 = 0;
    let ne0 = n as i64;
    let ne1 = m;
    let r2: u32 = 1;
    let r3: u32 = 1;

    // Threadgroup grid: NSG=2, nr0=1 for control, nr0=2 for nr2 variant.
    // Rows per threadgroup = NSG * nr0 = 2 or 4.
    let rows_per_tg = 2 * variant.nr0();
    let threadgroup_count = MTLSize {
        width: ceil_div(n, rows_per_tg),
        height: m as usize,
        depth: b as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: 32 * 2, // NSG=2 → 64 threads/tg = 2 simdgroups × 32 lanes
        height: 1,
        depth: 1,
    };

    let semaphore = Arc::new(CommandSemaphore::new());
    let cb = create_command_buffer(queue, semaphore)?;
    let encoder = cb.compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(weight), 0);
    encoder.set_buffer(1, Some(lhs), 0);
    encoder.set_buffer(2, Some(dst), 0);
    encoder.set_bytes(3, &ne00);
    encoder.set_bytes(4, &ne01);
    encoder.set_bytes(5, &ne02);
    encoder.set_bytes(6, &nb00);
    encoder.set_bytes(7, &nb01_bytes);
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
    encoder.use_resource(weight, MTLResourceUsage::Read);
    encoder.use_resource(lhs, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(threadgroup_count, threads_per_threadgroup);
    encoder.end_encoding();
    let t0 = std::time::Instant::now();
    cb.commit();
    cb.wait_until_completed();
    Ok(t0.elapsed().as_nanos())
}

/// Allocate + deterministically fill a (weight, lhs, dst) triplet for a Q6_K
/// matmul of shape [1, k] @ [n, k] (weight in column-row layout, laid out as
/// `n` rows of `(k/QK_K)` Q6_K blocks). Output is a `[1, n]` f32 vector.
///
/// This isn't a semantically valid Q6_K tensor (quant scales are random bytes),
/// but the kernels don't validate — they compute a deterministic function of
/// the bytes, so as long as both variants see the SAME bytes they will produce
/// numerically-comparable results. This is the correctness-sanity-check
/// foundation: if Variant B's NR0=2 port is correct, both variants will sum
/// the same dot-product terms over the same bytes and output the same values.
#[allow(clippy::type_complexity)]
fn q6k_alloc_buffers(
    device: &Device,
    n: usize,
    k: usize,
) -> Result<(
    candle_metal_kernels::metal::Buffer,
    candle_metal_kernels::metal::Buffer,
    candle_metal_kernels::metal::Buffer,
)> {
    assert_eq!(k % Q6K_QK_K, 0, "k must be a multiple of QK_K=256");
    let blocks_per_row = k / Q6K_QK_K;
    let weight_bytes = n * blocks_per_row * Q6K_BLOCK_BYTES;
    let lhs_bytes = k * std::mem::size_of::<f32>();
    let dst_bytes = n * std::mem::size_of::<f32>();

    let weight = device.new_buffer(weight_bytes, RESOURCE_OPTIONS)?;
    let lhs = device.new_buffer(lhs_bytes, RESOURCE_OPTIONS)?;
    let dst = device.new_buffer(dst_bytes, RESOURCE_OPTIONS)?;

    // Deterministic fill with REALISTIC block_q6_K layout (not just garbage
    // bytes) so the kernels produce finite outputs we can meaningfully compare.
    //
    // block_q6_K = ql[128] + qh[64] + scales[16] + half d  = 210 bytes per
    // super-block of QK_K=256 weights. Layout (see quantized.metal:198-203):
    //   offset   0..128  : ql  (uint8_t[128]  — lower 4 bits of quants)
    //   offset 128..192  : qh  (uint8_t[64]   — upper 2 bits of quants)
    //   offset 192..208  : sc  ( int8_t[16]   — scales, i8)
    //   offset 208..210  : d   (half          — super-block scale)
    //
    // We fill ql/qh with a low-entropy Knuth-hash pattern, set scales to
    // small i8 values (±8 range), and force d = 0.0625 (half bit pattern
    // 0x2C00) so the per-block scale × dot-product products stay well within
    // f32 range. This guarantees finite outputs for both variants.
    unsafe {
        let w_ptr = weight.contents();
        let total_blocks = n * blocks_per_row;
        for block_idx in 0..total_blocks {
            let base = block_idx * Q6K_BLOCK_BYTES;
            // ql: 128 bytes
            for j in 0..128 {
                *w_ptr.add(base + j) =
                    ((block_idx.wrapping_mul(2654435761).wrapping_add(j)) & 0xFF) as u8;
            }
            // qh: 64 bytes
            for j in 0..64 {
                *w_ptr.add(base + 128 + j) =
                    ((block_idx.wrapping_mul(40503).wrapping_add(j)) & 0xFF) as u8;
            }
            // scales: 16 i8 values, pattern in -8..+8
            for j in 0..16 {
                let v = (((block_idx.wrapping_mul(7919) + j) & 0x0F) as i32) - 8;
                *(w_ptr.add(base + 192 + j) as *mut i8) = v as i8;
            }
            // d (half): bit pattern 0x2C00 = 0.0625. Little-endian = [0x00, 0x2C].
            *w_ptr.add(base + 208) = 0x00;
            *w_ptr.add(base + 209) = 0x2C;
        }

        let l_ptr = lhs.contents() as *mut f32;
        for i in 0..k {
            // Small, non-uniform lhs values. Range ~[-0.5, +0.5).
            *l_ptr.add(i) = (((i & 0xFF) as f32) / 256.0) - 0.5;
        }
        // Zero dst so partial writes by mis-behaving kernels are visible.
        let d_ptr = dst.contents() as *mut f32;
        for i in 0..n {
            *d_ptr.add(i) = 0.0;
        }
    }

    Ok((weight, lhs, dst))
}

/// Run one (shape, variant) cell. Returns (median_ns, mean_ns).
#[allow(clippy::too_many_arguments)]
fn q6k_run_one_cell(
    device: &Device,
    kernels: &Kernels,
    queue: &candle_metal_kernels::metal::CommandQueue,
    shape: &Shape,
    variant: Q6KVariant,
    iters: usize,
    warmup: usize,
) -> Result<(u128, u128)> {
    let (n, k) = (shape.n, shape.k);
    // Variant B requires n divisible by rows_per_tg = NSG*nr0 = 4. All 6
    // Q6_K production shapes (4096, 8192, 2048, 1024, 2816) are multiples of 4.
    let rows_per_tg = 2 * variant.nr0();
    if n % rows_per_tg != 0 {
        anyhow::bail!(
            "shape '{}' n={} not divisible by rows_per_tg={} (variant={:?})",
            shape.label, n, rows_per_tg, variant
        );
    }

    let (weight, lhs, dst) = q6k_alloc_buffers(device, n, k)?;
    let pipeline = kernels
        .load_pipeline(device, Source::Quantized, variant.kernel_name())
        .map_err(|e| anyhow::anyhow!("load_pipeline({}) failed: {e:?}", variant.kernel_name()))?;

    // Warmup.
    for _ in 0..warmup {
        let _ = q6k_dispatch_once(queue, &pipeline, variant, &weight, &lhs, &dst, n, k)?;
    }
    // Timed iterations.
    let mut samples: Vec<u128> = Vec::with_capacity(iters);
    for _ in 0..iters {
        samples.push(q6k_dispatch_once(queue, &pipeline, variant, &weight, &lhs, &dst, n, k)?);
    }
    samples.sort_unstable();
    let median = samples[iters / 2];
    let mean = samples.iter().copied().sum::<u128>() / (iters as u128);
    Ok((median, mean))
}

/// Correctness sanity check: dispatch both variants against IDENTICAL input
/// buffers, read out their dst vectors, compute max|Δ| and max relative |Δ|.
/// Returns (max_abs, max_rel, pass) — pass is true iff max_rel <= 1e-5 OR
/// max_abs is below a conservative epsilon (1e-3) for values that are
/// essentially zero.
fn q6k_correctness_check(
    device: &Device,
    kernels: &Kernels,
    queue: &candle_metal_kernels::metal::CommandQueue,
    shape: &Shape,
) -> Result<(f32, f32, bool, Vec<f32>, Vec<f32>)> {
    let (n, k) = (shape.n, shape.k);

    // Allocate TWO independent buffer triplets so the two variant runs don't
    // observe each other's dst writes. The weight + lhs buffers are filled
    // with the identical deterministic pattern (q6k_alloc_buffers is pure).
    let (weight_a, lhs_a, dst_a) = q6k_alloc_buffers(device, n, k)?;
    let (weight_b, lhs_b, dst_b) = q6k_alloc_buffers(device, n, k)?;

    let pipe_a = kernels
        .load_pipeline(device, Source::Quantized, Q6KVariant::ControlA.kernel_name())
        .map_err(|e| anyhow::anyhow!("load_pipeline(A) failed: {e:?}"))?;
    let pipe_b = kernels
        .load_pipeline(device, Source::Quantized, Q6KVariant::Nr2VariantB.kernel_name())
        .map_err(|e| anyhow::anyhow!("load_pipeline(B) failed: {e:?}"))?;

    // One dispatch each.
    let _ = q6k_dispatch_once(queue, &pipe_a, Q6KVariant::ControlA, &weight_a, &lhs_a, &dst_a, n, k)?;
    let _ = q6k_dispatch_once(queue, &pipe_b, Q6KVariant::Nr2VariantB, &weight_b, &lhs_b, &dst_b, n, k)?;

    // Read back.
    let mut out_a = vec![0f32; n];
    let mut out_b = vec![0f32; n];
    unsafe {
        let src_a = dst_a.contents() as *const f32;
        let src_b = dst_b.contents() as *const f32;
        for i in 0..n {
            out_a[i] = *src_a.add(i);
            out_b[i] = *src_b.add(i);
        }
    }

    // Max absolute and max relative delta. Also require no NaN or Inf — any
    // garbage output is a correctness failure, not a pass.
    let mut max_abs: f32 = 0.0;
    let mut max_rel: f32 = 0.0;
    let mut any_nan_or_inf = false;
    let mut nan_inf_count = 0;
    for i in 0..n {
        let a = out_a[i];
        let b = out_b[i];
        if !a.is_finite() || !b.is_finite() {
            any_nan_or_inf = true;
            nan_inf_count += 1;
            continue;
        }
        let d = (a - b).abs();
        if d > max_abs { max_abs = d; }
        let denom = a.abs().max(b.abs());
        if denom > 1e-6 {
            let r = d / denom;
            if r > max_rel { max_rel = r; }
        }
    }

    if any_nan_or_inf {
        println!("# correctness check: {nan_inf_count}/{n} elements are NaN/Inf — check lhs magnitude / quant bytes");
    }

    // Pass criteria: all finite, max relative delta under 1e-5 OR max absolute
    // delta under a small epsilon (covers the all-zero-output case).
    let pass = !any_nan_or_inf && (max_rel <= 1e-5 || max_abs <= 1e-6);
    Ok((max_abs, max_rel, pass, out_a, out_b))
}

fn run_q6k_microbench(iters: usize, warmup: usize, runs: usize) -> Result<()> {
    let device = Device::system_default().expect("no Metal device");
    let kernels = Kernels::new();
    let queue = device.new_command_queue()?;

    println!("# 1bNEW.29 Option C — Q6_K NR0=2 microbench");
    println!("# device:        {}", device.architecture_name());
    println!("# device_type:   {:?}", device.device_type());
    println!("# iters/sample:  {iters}");
    println!("# warmup:        {warmup}");
    println!("# runs:          {runs} (cross-run median)");
    println!("# methodology:   per-dispatch commit+waitUntilCompleted, fresh");
    println!("#                CommandBuffer per call, median across iters,");
    println!("#                then median across {runs} independent runs.");
    println!("# variants:      A = kernel_mul_mv_q6_K_f32 (candle production,");
    println!("#                    1 row/simdgroup)");
    println!("#                B = kernel_mul_mv_q6_K_f32_nr2 (llama.cpp port,");
    println!("#                    2 rows/simdgroup — nr0=2)");
    println!();

    let shapes = q6k_microbench_shapes();

    // -------------------------------------------------------------------
    // Phase 1: correctness sanity check on one representative shape.
    // -------------------------------------------------------------------
    println!("# Phase 1: correctness sanity check (Attn q_proj sliding)");
    let sanity_shape = shapes
        .iter()
        .find(|s| s.label.contains("q_proj sliding"))
        .expect("q_proj sliding shape missing")
        .clone();
    let (max_abs, max_rel, pass, out_a, out_b) =
        q6k_correctness_check(&device, &kernels, &queue, &sanity_shape)?;
    println!("#   shape:       {}", sanity_shape.label);
    println!("#   out_a[0..4]: {:?}", &out_a[..4.min(out_a.len())]);
    println!("#   out_b[0..4]: {:?}", &out_b[..4.min(out_b.len())]);
    println!("#   max|Δ|:      {max_abs:.6e}");
    println!("#   max rel|Δ|:  {max_rel:.6e}");
    println!("#   verdict:     {}", if pass { "PASS" } else { "FAIL" });
    if !pass {
        println!();
        println!("# FAIL: correctness check failed — Variant B port is WRONG.");
        println!("# NOT proceeding to timing data (would be meaningless).");
        anyhow::bail!("Q6_K nr2 correctness check failed: max|Δ|={max_abs}, max rel|Δ|={max_rel}");
    }
    println!();

    // -------------------------------------------------------------------
    // Phase 2: timing sweep — Variant A vs Variant B across 6 shapes.
    // -------------------------------------------------------------------
    println!("# Phase 2: timing sweep");
    println!(
        "{:<48} {:>12} {:>12} {:>10}",
        "shape", "A µs (med)", "B µs (med)", "A/B ratio"
    );
    println!("{:-<86}", "");

    let mut all_results: Vec<(Shape, u128, u128, f64)> = Vec::new();

    for shape in &shapes {
        // Collect medians from each run.
        let mut a_runs: Vec<u128> = Vec::with_capacity(runs);
        let mut b_runs: Vec<u128> = Vec::with_capacity(runs);
        for _ in 0..runs {
            let (med_a, _) = q6k_run_one_cell(
                &device, &kernels, &queue, shape, Q6KVariant::ControlA, iters, warmup,
            )?;
            let (med_b, _) = q6k_run_one_cell(
                &device, &kernels, &queue, shape, Q6KVariant::Nr2VariantB, iters, warmup,
            )?;
            a_runs.push(med_a);
            b_runs.push(med_b);
        }
        a_runs.sort_unstable();
        b_runs.sort_unstable();
        let a_med = a_runs[runs / 2];
        let b_med = b_runs[runs / 2];
        let ratio = if b_med == 0 {
            f64::INFINITY
        } else {
            (a_med as f64) / (b_med as f64)
        };
        println!(
            "{:<48} {:>11.2} {:>11.2} {:>9.3}x",
            shape.label,
            (a_med as f64) / 1000.0,
            (b_med as f64) / 1000.0,
            ratio,
        );
        all_results.push((*shape, a_med, b_med, ratio));
    }

    // -------------------------------------------------------------------
    // Phase 3: verdict — GO if any shape shows >=10% speedup, NO-GO otherwise.
    // -------------------------------------------------------------------
    println!();
    println!("# Phase 3: per-shape verdict (GO threshold: Variant B >=10% faster)");
    println!(
        "{:<48} {:>10} {:>10}",
        "shape", "speedup", "decision"
    );
    println!("{:-<72}", "");
    let mut any_go = false;
    let mut best_ratio = 0.0_f64;
    let mut best_shape = "";
    for (shape, _a, _b, ratio) in &all_results {
        let speedup_pct = (*ratio - 1.0) * 100.0;
        let decision = if *ratio >= 1.10 {
            any_go = true;
            if *ratio > best_ratio {
                best_ratio = *ratio;
                best_shape = shape.label;
            }
            "GO"
        } else {
            "NO-GO"
        };
        println!(
            "{:<48} {:>+9.1}% {:>10}",
            shape.label, speedup_pct, decision,
        );
    }

    println!();
    if any_go {
        println!(
            "# OVERALL VERDICT: GO — best shape is '{}' at {:.3}x ({:+.1}%)",
            best_shape,
            best_ratio,
            (best_ratio - 1.0) * 100.0
        );
    } else {
        println!("# OVERALL VERDICT: NO-GO — all shapes < 10% speedup.");
        println!("# Hypothesis (llama.cpp's nr0=2 port gains wall-clock on M5 Max");
        println!("# at hf2q Q6_K shapes) is EMPIRICALLY FALSIFIED.");
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
    /// 1bNEW.29 Option C — Q6_K nr0=2 row-loop port microbench.
    /// Compares production candle kernel vs llama.cpp byte-for-byte port.
    Q6kMicrobench {
        /// Number of timed iterations per (shape, variant) cell.
        #[arg(long, default_value_t = 1000)]
        iters: usize,
        /// Number of untimed warmup iterations per cell.
        #[arg(long, default_value_t = 100)]
        warmup: usize,
        /// Number of independent full sweeps; per-shape cross-run median.
        #[arg(long, default_value_t = 4)]
        runs: usize,
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
        Task::Q6kMicrobench { iters, warmup, runs } => {
            run_q6k_microbench(iters, warmup, runs)?;
        }
    }
    Ok(())
}
