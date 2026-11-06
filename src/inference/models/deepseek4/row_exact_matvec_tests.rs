//! `MatvecKernels::RowExact` (`HF2Q_DEEPSEEK_ROW_EXACT_MATVEC=1`) on the
//! hf2q wrappers, at the ds4 0731 file's decode shapes.
//!
//! - Joint rows equal serial rows bitwise: for B = 2..=4, every row of the
//!   joint product (`row_matmul(IndependentRows)` at `m = B`, `raw_matmul`
//!   at `m = B` for the head, `grouped_output_a_batched` at `rows = B`)
//!   equals the serial `m = 1` product of that row (`raw_matmul` at `m = 1`,
//!   `grouped_output_a`). The row-exact pipelines must be the ones that ran.
//! - F64 bound on one shape per family: `|y - ref| <= gamma_K * sum |w x|`,
//!   `gamma_K = K u / (1 - K u)`, `u = 2^-24`.
//! - `MatvecKernels::Stock` is the main-branch route: its wrapper outputs equal
//!   direct dispatches of the stock kernels bitwise and no row-exact pipeline
//!   is ever built.

use mlx_native::graph::GraphSession;
use mlx_native::ops::dense_gemm::{dispatch_dense_matvec_f16w_f32io, DenseGemmF16Params};
use mlx_native::ops::dense_mm_f16::{dense_matmul_f16_f32_tensor, DenseMmF16F32Params};
use mlx_native::ops::quantized_matmul_ggml::{
    quantized_matmul_ggml_batched_mv, GgmlBatchedQuantizedMatmulParams, GgmlQuantizedMatmulParams,
    GgmlType,
};
use mlx_native::ops::row_exact_matvec::row_exact_matvec_kernel_names;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::ds4_bench_support::Rng;
use super::forward_support::{
    alloc, grouped_output_a, grouped_output_a_batched, raw_matmul, BatchedGroupedOutputArena,
};
use super::residency::{MatvecKernels, RawMatrixRef};
use super::row_matmul::{row_matmul, row_view, RowMatmul};
use crate::serve::gpu::GpuContext;

const F16_SHAPES: [(&str, usize, usize); 7] = [
    ("hc_fn", 24, 16384),
    ("output_hc_fn", 4, 16384),
    ("router", 256, 4096),
    ("compressor_r4", 1024, 4096),
    ("compressor_r128", 512, 4096),
    ("indexer_q_b", 8192, 1024),
    ("indexer_proj", 64, 4096),
];

/// Q8_0 products joint decode issues through `row_matmul(IndependentRows)`.
const Q8_0_ROW_SHAPES: [(&str, usize, usize); 6] = [
    ("attn_q_a", 1024, 4096),
    ("attn_kv", 512, 4096),
    ("attn_q_b", 32768, 1024),
    ("attn_output_b", 4096, 8192),
    ("shexp_gate_up", 2048, 4096),
    ("shexp_down", 4096, 2048),
];

/// ds4 output head (`forward_logits` issues it through `raw_matmul` at m = B).
const HEAD: (usize, usize) = (129_280, 4096);
/// ds4 output-A: 8 groups of [1024, 4096] over 64 heads of 512.
const OUTPUT_A: (usize, usize, usize, usize) = (8, 1024, 64, 512);

fn f16_weight(device: &MlxDevice, n: usize, k: usize, seed: u64) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(n * k * 2, DType::F16, vec![n, k])
        .unwrap();
    let mut rng = Rng(seed);
    // Finite halves in ±[2^-8, 2^-7): random sign and mantissa, fixed exponent.
    for word in buffer.as_mut_slice::<u64>().unwrap() {
        *word = (rng.next() & 0x83ff_83ff_83ff_83ff) | 0x1c00_1c00_1c00_1c00;
    }
    buffer
}

fn q8_0_weight(device: &MlxDevice, n: usize, k: usize, seed: u64) -> MlxBuffer {
    let bytes = n * k / 32 * 34;
    let mut buffer = device.alloc_buffer(bytes, DType::U8, vec![n, k]).unwrap();
    let data = buffer.as_mut_slice::<u8>().unwrap();
    let mut rng = Rng(seed);
    for chunk in data.chunks_mut(8) {
        let r = rng.next().to_le_bytes();
        chunk.copy_from_slice(&r[..chunk.len()]);
    }
    for (index, block) in data.chunks_exact_mut(34).enumerate() {
        let scale = half::f16::from_f32(0.0005 + (index % 97) as f32 * 0.000_02);
        block[..2].copy_from_slice(&scale.to_bits().to_le_bytes());
    }
    buffer
}

fn input(device: &MlxDevice, rows: usize, k: usize, seed: u64) -> MlxBuffer {
    let mut buffer = alloc(device, DType::F32, vec![rows, k], "row-exact test input").unwrap();
    let mut rng = Rng(seed);
    for value in buffer.as_mut_slice::<f32>().unwrap() {
        *value = rng.unit();
    }
    buffer
}

fn output(device: &MlxDevice, rows: usize, n: usize) -> MlxBuffer {
    let mut buffer = alloc(device, DType::F32, vec![rows, n], "row-exact test output").unwrap();
    buffer.as_mut_slice::<u32>().unwrap().fill(0x7fc0_dead);
    buffer
}

fn bits(buffer: &MlxBuffer) -> Vec<u32> {
    buffer.as_slice::<u32>().unwrap().to_vec()
}

fn matrix<'a>(
    buffer: &'a MlxBuffer,
    ggml_type: GgmlType,
    shape: &'a [usize],
    matvec: MatvecKernels,
) -> RawMatrixRef<'a> {
    RawMatrixRef {
        buffer,
        ggml_type,
        shape,
        matvec,
    }
}

/// Encode `encode` in one session and wait; returns the registry for
/// pipeline inspection.
fn run<'c>(
    ctx: &'c mut GpuContext,
    encode: impl FnOnce(&mut GraphSession<'_>, &mut KernelRegistry, &MlxDevice),
) -> &'c mut KernelRegistry {
    let device = ctx.device().clone();
    let (executor, registry) = ctx.split();
    let mut session = executor.begin().unwrap();
    encode(&mut session, registry, &device);
    session.finish().unwrap();
    registry
}

fn row_exact_pipelines_built(registry: &KernelRegistry) -> Vec<&'static str> {
    row_exact_matvec_kernel_names()
        .filter(|name| registry.pipeline_identity(name).is_ok())
        .collect()
}

/// Joint `product(m = B)` rows equal serial `product(m = 1)` rows, B = 2..=4.
fn assert_joint_equals_serial(
    label: &str,
    ctx: &mut GpuContext,
    rows_in: usize,
    rows_out: usize,
    x: &MlxBuffer,
    joint: impl Fn(&mut GraphSession<'_>, &mut KernelRegistry, &MlxDevice, &MlxBuffer, usize),
    serial: impl Fn(&mut GraphSession<'_>, &mut KernelRegistry, &MlxDevice, &MlxBuffer, &MlxBuffer),
) {
    let device = ctx.device().clone();
    let serial_out = output(&device, 4, rows_out);
    run(ctx, |s, r, d| {
        for row in 0..4 {
            let xr = row_view(x, row, vec![1, rows_in]).unwrap();
            let yr = row_view(&serial_out, row, vec![1, rows_out]).unwrap();
            serial(s, r, d, &xr, &yr);
        }
    });
    let serial_bits = bits(&serial_out);
    assert!(
        serial_out
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
            && serial_out
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .any(|v| *v != 0.0),
        "{label}: serial output is not a finite non-zero result"
    );
    for b in 2..=4 {
        let joint_out = output(&device, b, rows_out);
        run(ctx, |s, r, d| joint(s, r, d, &joint_out, b));
        let got = bits(&joint_out);
        for row in 0..b {
            let span = row * rows_out..(row + 1) * rows_out;
            if got[span.clone()] != serial_bits[span.clone()] {
                let col = (0..rows_out)
                    .find(|&c| got[row * rows_out + c] != serial_bits[row * rows_out + c])
                    .unwrap();
                panic!(
                    "{label}: B={b} row {row} col {col}: joint {:#010x} serial {:#010x}",
                    got[row * rows_out + col],
                    serial_bits[row * rows_out + col]
                );
            }
        }
    }
}

fn independent_rows_case(
    ctx: &mut GpuContext,
    label: &str,
    w: &RawMatrixRef<'_>,
    n: usize,
    k: usize,
) {
    let device = ctx.device().clone();
    let x = input(&device, 4, k, 0x5eed ^ n as u64);
    assert_joint_equals_serial(
        label,
        ctx,
        k,
        n,
        &x,
        |s, r, d, y, b| {
            let xb = x.slice_view(0, b * k).with_shape(vec![b, k]).unwrap();
            row_matmul(
                RowMatmul::IndependentRows,
                s,
                r,
                d,
                &xb,
                w,
                y,
                b,
                n,
                k,
                label,
            )
            .unwrap();
        },
        |s, r, d, xr, yr| raw_matmul(s, r, d, xr, w, yr, 1, n, k, label).unwrap(),
    );
}

#[test]
fn row_exact_f16_joint_rows_equal_serial_rows_bitwise() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    for (index, (label, n, k)) in F16_SHAPES.into_iter().enumerate() {
        let buffer = f16_weight(&device, n, k, 0xf160 + index as u64);
        let shape = [n, k];
        let w = matrix(&buffer, GgmlType::F16, &shape, MatvecKernels::RowExact);
        independent_rows_case(&mut ctx, label, &w, n, k);
    }
    let (_, registry) = ctx.split();
    let built = row_exact_pipelines_built(registry);
    for m in 1..=4 {
        assert!(
            built
                .iter()
                .any(|name| name.starts_with("row_exact_mv_f16_")
                    && name.ends_with(&format!("_m{m}"))),
            "row-exact F16 m={m} pipeline did not run: {built:?}"
        );
    }
}

#[test]
fn row_exact_q8_0_joint_rows_equal_serial_rows_bitwise() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    for (index, (label, n, k)) in Q8_0_ROW_SHAPES.into_iter().enumerate() {
        let buffer = q8_0_weight(&device, n, k, 0x0800 + index as u64);
        let shape = [n, k];
        let w = matrix(&buffer, GgmlType::Q8_0, &shape, MatvecKernels::RowExact);
        independent_rows_case(&mut ctx, label, &w, n, k);
    }
    // The head is issued at m = B through raw_matmul, never split.
    let (n, k) = HEAD;
    let buffer = q8_0_weight(&device, n, k, 0x08ff);
    let shape = [n, k];
    let w = matrix(&buffer, GgmlType::Q8_0, &shape, MatvecKernels::RowExact);
    let x = input(&device, 4, k, 0x4ead);
    assert_joint_equals_serial(
        "output head",
        &mut ctx,
        k,
        n,
        &x,
        |s, r, d, y, b| {
            let xb = x.slice_view(0, b * k).with_shape(vec![b, k]).unwrap();
            raw_matmul(s, r, d, &xb, &w, y, b, n, k, "output head").unwrap();
        },
        |s, r, d, xr, yr| raw_matmul(s, r, d, xr, &w, yr, 1, n, k, "output head").unwrap(),
    );
    let (_, registry) = ctx.split();
    let built = row_exact_pipelines_built(registry);
    for m in 1..=4 {
        assert!(
            built
                .iter()
                .any(|name| name.starts_with("row_exact_mv_q8_0_")
                    && name.ends_with(&format!("_m{m}"))),
            "row-exact Q8_0 m={m} pipeline did not run: {built:?}"
        );
    }
}

#[test]
fn row_exact_grouped_output_a_joint_rows_equal_serial_rows_bitwise() {
    let (groups, rank, heads, head_dim) = OUTPUT_A;
    let gw = heads * head_dim / groups;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    let buffer = q8_0_weight(&device, groups * rank, gw, 0x0a0a);
    let shape = [groups * rank, gw];
    let w = matrix(&buffer, GgmlType::Q8_0, &shape, MatvecKernels::RowExact);
    let width = heads * head_dim;
    let x = input(&device, 4, width, 0x0a0b);
    assert_joint_equals_serial(
        "grouped output-A",
        &mut ctx,
        width,
        groups * rank,
        &x,
        |s, r, d, y, b| {
            let xb = x
                .slice_view(0, b * width)
                .with_shape(vec![b, heads, head_dim])
                .unwrap();
            let arena = BatchedGroupedOutputArena::new(d, b, groups, gw, rank).unwrap();
            grouped_output_a_batched(
                s, r, d, &xb, &w, y, &arena, b, groups, rank, heads, head_dim,
            )
            .unwrap();
        },
        |s, r, d, xr, yr| {
            let xr = xr.with_shape(vec![1, heads, head_dim]).unwrap();
            grouped_output_a(s, r, d, &xr, &w, yr, groups, rank, heads, head_dim).unwrap();
        },
    );
    let (_, registry) = ctx.split();
    let built = row_exact_pipelines_built(registry);
    assert!(
        built
            .iter()
            .any(|name| name.starts_with("row_exact_mv_q8_0_")),
        "row-exact grouped output-A did not run: {built:?}"
    );
}

/// Independent rows beyond one row-exact dispatch split into <= 8-row chunks
/// and still equal the serial rows.
#[test]
fn row_exact_independent_rows_above_eight_chunk_and_stay_exact() {
    const ROWS: usize = 11;
    let (n, k) = (512, 4096);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    for (ggml_type, buffer) in [
        (GgmlType::F16, f16_weight(&device, n, k, 0x11)),
        (GgmlType::Q8_0, q8_0_weight(&device, n, k, 0x12)),
    ] {
        let shape = [n, k];
        let w = matrix(&buffer, ggml_type, &shape, MatvecKernels::RowExact);
        let x = input(&device, ROWS, k, 0x13);
        let joint = output(&device, ROWS, n);
        let serial = output(&device, ROWS, n);
        run(&mut ctx, |s, r, d| {
            row_matmul(
                RowMatmul::IndependentRows,
                s,
                r,
                d,
                &x,
                &w,
                &joint,
                ROWS,
                n,
                k,
                "chunked",
            )
            .unwrap();
            for row in 0..ROWS {
                let xr = row_view(&x, row, vec![1, k]).unwrap();
                let yr = row_view(&serial, row, vec![1, n]).unwrap();
                raw_matmul(s, r, d, &xr, &w, &yr, 1, n, k, "chunked serial").unwrap();
            }
        });
        assert_eq!(bits(&joint), bits(&serial), "{ggml_type:?}");
    }
}

fn dequantize_row(ggml_type: GgmlType, buffer: &MlxBuffer, row: usize, k: usize) -> Vec<f64> {
    let bytes = buffer.as_slice::<u8>().unwrap();
    match ggml_type {
        GgmlType::F16 => bytes[row * k * 2..(row + 1) * k * 2]
            .chunks_exact(2)
            .map(|h| f64::from(half::f16::from_le_bytes([h[0], h[1]]).to_f32()))
            .collect(),
        GgmlType::Q8_0 => {
            let row_bytes = k / 32 * 34;
            bytes[row * row_bytes..(row + 1) * row_bytes]
                .chunks_exact(34)
                .flat_map(|block| {
                    let d = f64::from(half::f16::from_le_bytes([block[0], block[1]]).to_f32());
                    block[2..].iter().map(move |q| d * f64::from(*q as i8))
                })
                .collect()
        }
        other => panic!("no reference for {other:?}"),
    }
}

/// `|y - ref| <= gamma_K * sum |w x|` for every output of `y` (`[rows, n]`).
fn assert_f64_bound(
    label: &str,
    ggml_type: GgmlType,
    weight: &MlxBuffer,
    first_weight_row: usize,
    x: &[f32],
    y: &[f32],
    rows: usize,
    n: usize,
    k: usize,
) {
    let u = 2f64.powi(-24);
    let gamma = k as f64 * u / (1.0 - k as f64 * u);
    for col in 0..n {
        let w = dequantize_row(ggml_type, weight, first_weight_row + col, k);
        for row in 0..rows {
            let xr = &x[row * k..(row + 1) * k];
            let (exact, magnitude) = w.iter().zip(xr).fold((0.0, 0.0), |(e, m), (w, x)| {
                let p = w * f64::from(*x);
                (e + p, m + p.abs())
            });
            let got = f64::from(y[row * n + col]);
            assert!(
                (got - exact).abs() <= gamma * magnitude + 1e-30,
                "{label}: row {row} col {col}: got {got} F64 {exact} bound {}",
                gamma * magnitude
            );
        }
    }
}

#[test]
fn row_exact_products_are_within_the_f32_summation_bound() {
    const ROWS: usize = 4;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    // F16 router and Q8_0 attn_q_a through row_matmul(IndependentRows).
    for (label, ggml_type, n, k) in [
        ("router", GgmlType::F16, 256, 4096),
        ("attn_q_a", GgmlType::Q8_0, 1024, 4096),
    ] {
        let buffer = match ggml_type {
            GgmlType::F16 => f16_weight(&device, n, k, 0xb0),
            _ => q8_0_weight(&device, n, k, 0xb1),
        };
        let shape = [n, k];
        let w = matrix(&buffer, ggml_type, &shape, MatvecKernels::RowExact);
        let x = input(&device, ROWS, k, 0xb2);
        let y = output(&device, ROWS, n);
        run(&mut ctx, |s, r, d| {
            row_matmul(
                RowMatmul::IndependentRows,
                s,
                r,
                d,
                &x,
                &w,
                &y,
                ROWS,
                n,
                k,
                label,
            )
            .unwrap();
        });
        assert_f64_bound(
            label,
            ggml_type,
            &buffer,
            0,
            x.as_slice::<f32>().unwrap(),
            y.as_slice::<f32>().unwrap(),
            ROWS,
            n,
            k,
        );
    }
    // Grouped output-A: every group against its own weight rows and input slice.
    let (groups, rank, heads, head_dim) = OUTPUT_A;
    let gw = heads * head_dim / groups;
    let width = heads * head_dim;
    let buffer = q8_0_weight(&device, groups * rank, gw, 0xb3);
    let shape = [groups * rank, gw];
    let w = matrix(&buffer, GgmlType::Q8_0, &shape, MatvecKernels::RowExact);
    let x = input(&device, ROWS, width, 0xb4);
    let xg = x.with_shape(vec![ROWS, heads, head_dim]).unwrap();
    let y = output(&device, ROWS, groups * rank);
    let arena = BatchedGroupedOutputArena::new(&device, ROWS, groups, gw, rank).unwrap();
    run(&mut ctx, |s, r, d| {
        grouped_output_a_batched(
            s, r, d, &xg, &w, &y, &arena, ROWS, groups, rank, heads, head_dim,
        )
        .unwrap();
    });
    let xv = x.as_slice::<f32>().unwrap();
    let yv = y.as_slice::<f32>().unwrap();
    for g in 0..groups {
        let xs: Vec<f32> = (0..ROWS)
            .flat_map(|j| xv[j * width + g * gw..j * width + (g + 1) * gw].to_vec())
            .collect();
        let ys: Vec<f32> = (0..ROWS)
            .flat_map(|j| {
                yv[j * groups * rank + g * rank..j * groups * rank + (g + 1) * rank].to_vec()
            })
            .collect();
        assert_f64_bound(
            &format!("grouped output-A group {g}"),
            GgmlType::Q8_0,
            &buffer,
            g * rank,
            &xs,
            &ys,
            ROWS,
            rank,
            gw,
        );
    }
}

/// With the switch off the wrappers take main's routes: their outputs equal
/// direct dispatches of the stock kernels bitwise, and no row-exact pipeline
/// is built.
#[test]
fn stock_matvec_kernels_keep_the_main_routes_bitwise() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();

    // F16: m = 1 is the L9 F32-input matvec, independent rows split per row,
    // matrix rows take the tiled tensor kernel.
    let (n, k) = (256, 4096);
    let f16 = f16_weight(&device, n, k, 0xc0);
    let shape = [n, k];
    let w = matrix(&f16, GgmlType::F16, &shape, MatvecKernels::Stock);
    let x = input(&device, 4, k, 0xc1);
    let wrapper_rows = output(&device, 4, n);
    let wrapper_matrix = output(&device, 4, n);
    let direct_rows = output(&device, 4, n);
    let direct_matrix = output(&device, 4, n);
    run(&mut ctx, |s, r, d| {
        row_matmul(
            RowMatmul::IndependentRows,
            s,
            r,
            d,
            &x,
            &w,
            &wrapper_rows,
            4,
            n,
            k,
            "rows",
        )
        .unwrap();
        row_matmul(
            RowMatmul::Matrix,
            s,
            r,
            d,
            &x,
            &w,
            &wrapper_matrix,
            4,
            n,
            k,
            "matrix",
        )
        .unwrap();
        for row in 0..4 {
            let xr = row_view(&x, row, vec![1, k]).unwrap();
            let yr = row_view(&direct_rows, row, vec![1, n]).unwrap();
            s.barrier_between(&[&xr, &f16], &[&yr]);
            dispatch_dense_matvec_f16w_f32io(
                s.encoder_mut(),
                r,
                d.metal_device(),
                &xr,
                &f16,
                &yr,
                &DenseGemmF16Params {
                    m: 1,
                    n: n as u32,
                    k: k as u32,
                },
            )
            .unwrap();
        }
        s.barrier_between(&[&x, &f16], &[&direct_matrix]);
        dense_matmul_f16_f32_tensor(
            s.encoder_mut(),
            r,
            d,
            &f16,
            &x,
            &direct_matrix,
            &DenseMmF16F32Params {
                m: 4,
                n: n as u32,
                k: k as u32,
                src0_batch: 1,
                src1_batch: 1,
            },
        )
        .unwrap();
    });
    assert_eq!(
        bits(&wrapper_rows),
        bits(&direct_rows),
        "F16 independent rows"
    );
    assert_eq!(
        bits(&wrapper_matrix),
        bits(&direct_matrix),
        "F16 matrix rows"
    );

    // Q8_0: raw_matmul at m = 1..=4 is the canonical quantized matvec.
    let (n, k) = (1024, 4096);
    let q8 = q8_0_weight(&device, n, k, 0xc2);
    let shape = [n, k];
    let w = matrix(&q8, GgmlType::Q8_0, &shape, MatvecKernels::Stock);
    for m in 1..=4 {
        let xm = x.slice_view(0, m * k).with_shape(vec![m, k]).unwrap();
        let wrapper = output(&device, m, n);
        let direct = output(&device, m, n);
        run(&mut ctx, |s, r, d| {
            raw_matmul(s, r, d, &xm, &w, &wrapper, m, n, k, "Q8_0").unwrap();
            s.barrier_between(&[&xm, &q8], &[&direct]);
            s.quantized_matmul_ggml(
                r,
                d,
                &xm,
                &q8,
                &direct,
                &GgmlQuantizedMatmulParams {
                    m: m as u32,
                    n: n as u32,
                    k: k as u32,
                    ggml_type: GgmlType::Q8_0,
                },
            )
            .unwrap();
        });
        assert_eq!(bits(&wrapper), bits(&direct), "Q8_0 m={m}");
    }

    // Grouped output-A serial: the batched quantized matvec.
    let (groups, rank, heads, head_dim) = OUTPUT_A;
    let gw = heads * head_dim / groups;
    let width = heads * head_dim;
    let oa = q8_0_weight(&device, groups * rank, gw, 0xc3);
    let shape = [groups * rank, gw];
    let w = matrix(&oa, GgmlType::Q8_0, &shape, MatvecKernels::Stock);
    let xa = input(&device, 1, width, 0xc4);
    let xa3 = xa.with_shape(vec![1, heads, head_dim]).unwrap();
    let wrapper = output(&device, 1, groups * rank);
    let direct = output(&device, 1, groups * rank);
    let registry = run(&mut ctx, |s, r, d| {
        grouped_output_a(s, r, d, &xa3, &w, &wrapper, groups, rank, heads, head_dim).unwrap();
        s.barrier_between(&[&xa3, &oa], &[&direct]);
        quantized_matmul_ggml_batched_mv(
            s.encoder_mut(),
            r,
            d,
            &xa3,
            &oa,
            &direct,
            &GgmlBatchedQuantizedMatmulParams {
                batch: groups as u32,
                m: 1,
                n: rank as u32,
                k: gw as u32,
                ggml_type: GgmlType::Q8_0,
            },
        )
        .unwrap();
    });
    assert_eq!(bits(&wrapper), bits(&direct), "grouped output-A serial");
    assert_eq!(row_exact_pipelines_built(registry), Vec::<&str>::new());
}

/// Resolution order: the runtime switch when set (`0`/`1` only), else the
/// build-time default, which itself is Stock unless the build set it.
#[test]
fn matvec_switch_resolves_env_then_build_default_then_stock() {
    use std::ffi::OsStr;
    use MatvecKernels::{RowExact, Stock};
    for build_default in [Stock, RowExact] {
        // Unset: the build default.
        assert_eq!(
            MatvecKernels::resolve(None, build_default).unwrap(),
            build_default
        );
        // Set: the runtime value wins over either build default.
        assert_eq!(
            MatvecKernels::resolve(Some(OsStr::new("0")), build_default).unwrap(),
            Stock
        );
        assert_eq!(
            MatvecKernels::resolve(Some(OsStr::new("1")), build_default).unwrap(),
            RowExact
        );
        for bad in ["", "true", "2", "on"] {
            assert!(
                MatvecKernels::resolve(Some(OsStr::new(bad)), build_default).is_err(),
                "{bad}"
            );
        }
    }
    // The build default follows HF2Q_DEEPSEEK_ROW_EXACT_MATVEC_DEFAULT of this
    // build, and is Stock when the build did not set it.
    let expected = match option_env!("HF2Q_DEEPSEEK_ROW_EXACT_MATVEC_DEFAULT") {
        Some("1") => RowExact,
        _ => Stock,
    };
    assert_eq!(MatvecKernels::BUILD_DEFAULT, expected);
    // One line per binary for build receipts (two Q1 arms of one commit).
    eprintln!("MATVEC_BUILD_DEFAULT {:?}", MatvecKernels::BUILD_DEFAULT);
}
