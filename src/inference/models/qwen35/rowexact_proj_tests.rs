//! Model-free Metal gates for the row-exact projection selector
//! (`HF2Q_QWEN_ROWEXACT_PROJ`, [`super::rowexact_proj`]) at the
//! Qwen3.6-35B-A3B projection shapes. Weights are random bytes in the GGML
//! block layouts with finite f16 scales; inputs are uniform in [-1, 1).
//!
//! For every selected `(type, N, K)` with the switch forced on:
//! * batched == serial: row `j` of an `m = 2..=4` call (every order of three
//!   rows, several orders of two and four rows, rows not starting at 0) and of
//!   an `m = 8` call is bitwise the `m = 1` call of row `j`, through the Qwen
//!   projection boundary and through each projection helper family the decode
//!   paths use (DeltaNet `apply_proj`, the full-attention / MTP / lm-head
//!   `apply_linear_projection_f32*` helpers);
//! * the selector really routes: the boundary output at `m = 1, 4, 8` is
//!   bitwise `dispatch_row_exact_matvec`'s;
//! * F64 reference: `|y - ref| <= gamma_K * sum_i |w_i x_i|` with
//!   `gamma_K = K u / (1 - K u)`, `u = 2^-24` (the bound of mlx-native's
//!   `tests/row_exact_matvec.rs`).
//!
//! With the switch off, the boundary and every helper are bitwise the stock
//! `quantized_matmul_ggml` at `m = 1..=4`; with it on, an `m = 9` (prefill)
//! call and the excluded shapes (Q6_K 2048x512, Q6_K 512x2048) stay stock.
//!
//! Run: `cargo test --release --bin hf2q rowexact_proj_tests -- --test-threads=1 --nocapture`

use mlx_native::ops::quantized_matmul_ggml::{
    quantized_matmul_ggml as stock_quantized_matmul_ggml, GgmlQuantizedMatmulParams, GgmlType,
};
use mlx_native::ops::row_exact_matvec::{
    dispatch_row_exact_matvec, RowExactMatvecShape, RowExactWeight,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::execution_dispatch::quantized_matmul_ggml;
use super::gpu_delta_net::apply_proj_with_ggml_type;
use super::gpu_full_attn::{
    apply_linear_projection_f32_into_with_ggml_type, apply_linear_projection_f32_with_ggml_type,
    download_f32, upload_f32,
};
use super::rowexact_proj::{
    parse_qwen_rowexact_proj, row_exact_weight, select, with_rowexact_proj_for_test, ProjRoute,
    ProjRouteCount, ROW_EXACT_PROJ_SHAPES,
};

const ROWS: usize = 8;
/// F64 reference MACs checked per shape (columns are subsampled above this).
const REFERENCE_MACS: usize = 32 << 20;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    fn in_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.unit() * 0.5 + 0.5) * (hi - lo)
    }
}

fn f16_le(value: f32) -> [u8; 2] {
    half::f16::from_f32(value).to_bits().to_le_bytes()
}

fn f16_at(bytes: &[u8], offset: usize) -> f64 {
    f64::from(half::f16::from_le_bytes([bytes[offset], bytes[offset + 1]]).to_f32())
}

fn block_bytes(t: GgmlType) -> usize {
    match t {
        GgmlType::Q4_K => 144,
        GgmlType::Q6_K => 210,
        other => panic!("no layout for {other:?}"),
    }
}

/// Random `[n, k]` GGML blocks with finite, normal f16 scales, written
/// straight into a device buffer.
fn random_weight(device: &MlxDevice, t: GgmlType, n: usize, k: usize, rng: &mut Rng) -> MlxBuffer {
    let bb = block_bytes(t);
    let len = n * k / 256 * bb;
    let mut buffer = device
        .alloc_buffer(len, DType::U8, vec![len])
        .expect("allocate weight");
    let bytes = buffer.as_mut_slice::<u8>().expect("map weight");
    for chunk in bytes.chunks_mut(8) {
        let word = rng.next().to_le_bytes();
        chunk.copy_from_slice(&word[..chunk.len()]);
    }
    for block in bytes.chunks_exact_mut(bb) {
        match t {
            GgmlType::Q4_K => {
                block[0..2].copy_from_slice(&f16_le(rng.in_range(0.8e-4, 1.6e-4)));
                block[2..4].copy_from_slice(&f16_le(rng.in_range(0.8e-4, 1.6e-4)));
            }
            GgmlType::Q6_K => {
                block[208..210].copy_from_slice(&f16_le(rng.in_range(0.8e-4, 1.6e-4)));
            }
            other => panic!("no layout for {other:?}"),
        }
    }
    buffer
}

/// Q4_K: value `32j + l` of sub-block `j` is `d * sc_j * q - dmin * mn_j`.
fn dequant_q4_k(block: &[u8]) -> impl Iterator<Item = f64> + '_ {
    let (d, dmin) = (f16_at(block, 0), f16_at(block, 2));
    let s = &block[4..16];
    let qs = &block[16..144];
    (0..256).map(move |v| {
        let (j, l) = (v / 32, v % 32);
        let (sc, mn) = if j < 4 {
            (s[j] & 63, s[j + 4] & 63)
        } else {
            (
                (s[j + 4] & 0xf) | ((s[j - 4] >> 6) << 4),
                (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
            )
        };
        let byte = qs[32 * (j / 2) + l];
        let q = if j % 2 == 0 { byte & 0xf } else { byte >> 4 };
        d * f64::from(sc) * f64::from(q) - dmin * f64::from(mn)
    })
}

/// Q6_K: value `128n + 32c + l` is `d * sc[8n + 2c + l/16] * (q - 32)`.
fn dequant_q6_k(block: &[u8]) -> impl Iterator<Item = f64> + '_ {
    let (ql, qh, sc) = (&block[..128], &block[128..192], &block[192..208]);
    let d = f16_at(block, 208);
    (0..256).map(move |v| {
        let (n, c, l) = (v / 128, v % 128 / 32, v % 32);
        let low = (ql[64 * n + 32 * (c & 1) + l] >> (4 * (c >> 1))) & 0xf;
        let high = (qh[32 * n + l] >> (2 * c)) & 3;
        let q = i32::from(low | (high << 4)) - 32;
        d * f64::from(sc[8 * n + 2 * c + l / 16] as i8) * f64::from(q)
    })
}

/// Dequantized weight row `row` in F64.
fn weight_row(weight: &MlxBuffer, t: GgmlType, k: usize, row: usize) -> Vec<f64> {
    let bb = block_bytes(t);
    let row_bytes = k / 256 * bb;
    let bytes = weight.as_slice::<u8>().expect("weight bytes");
    let blocks = bytes[row * row_bytes..(row + 1) * row_bytes].chunks_exact(bb);
    match t {
        GgmlType::Q4_K => blocks.flat_map(dequant_q4_k).collect(),
        GgmlType::Q6_K => blocks.flat_map(dequant_q6_k).collect(),
        other => panic!("no layout for {other:?}"),
    }
}

/// Max of `|y - ref| / bound` over the checked cells of the solo rows;
/// panics past 1.
fn check_reference(
    label: &str,
    weight: &MlxBuffer,
    t: GgmlType,
    n: usize,
    k: usize,
    x: &[Vec<f32>],
    solo: &[Vec<u32>],
) -> f64 {
    let u = 2f64.powi(-24);
    let gamma = k as f64 * u / (1.0 - k as f64 * u);
    let stride = (n * k * x.len()).div_ceil(REFERENCE_MACS).max(1);
    let mut cols: Vec<usize> = (0..n).step_by(stride).collect();
    if cols.last() != Some(&(n - 1)) {
        cols.push(n - 1);
    }
    let mut worst = 0f64;
    for &col in &cols {
        let w = weight_row(weight, t, k, col);
        for (j, xr) in x.iter().enumerate() {
            let (mut exact, mut magnitude) = (0f64, 0f64);
            for (wi, xi) in w.iter().zip(xr) {
                let p = wi * f64::from(*xi);
                exact += p;
                magnitude += p.abs();
            }
            let got = f64::from(f32::from_bits(solo[j][col]));
            let bound = gamma * magnitude + 1e-30;
            let ratio = (got - exact).abs() / bound;
            assert!(
                ratio <= 1.0,
                "{label}: row {j} col {col}: got {got}, F64 {exact}, |err| {} > bound {bound}",
                (got - exact).abs()
            );
            worst = worst.max(ratio);
        }
    }
    worst
}

fn params(t: GgmlType, m: usize, n: usize, k: usize) -> GgmlQuantizedMatmulParams {
    GgmlQuantizedMatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        ggml_type: t,
    }
}

fn packed_input(device: &MlxDevice, x: &[Vec<f32>], order: &[usize]) -> MlxBuffer {
    let values: Vec<f32> = order
        .iter()
        .flat_map(|&row| x[row].iter().copied())
        .collect();
    upload_f32(&values, device).expect("upload input")
}

fn output(device: &MlxDevice, m: usize, n: usize) -> MlxBuffer {
    device
        .alloc_buffer(m * n * 4, DType::F32, vec![m, n])
        .expect("allocate output")
}

fn bits(buffer: &MlxBuffer, count: usize) -> Vec<u32> {
    download_f32(buffer).expect("download")[..count]
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

/// The ways a Qwen decode path reaches a dense GGML projection.
#[derive(Clone, Copy, Debug)]
enum Entry {
    /// `execution_dispatch::quantized_matmul_ggml` (every helper below).
    Boundary,
    /// DeltaNet `apply_proj_with_ggml_type` (serial and joint DeltaNet).
    DeltaNetProj,
    /// `apply_linear_projection_f32_with_ggml_type` (MTP, verify heads,
    /// full-attention projections of the `seq_len` forward).
    LinearProj,
    /// `apply_linear_projection_f32_into_with_ggml_type` (lm head, serial
    /// and joint).
    LinearInto,
}

const ENTRIES: [Entry; 4] = [
    Entry::Boundary,
    Entry::DeltaNetProj,
    Entry::LinearProj,
    Entry::LinearInto,
];

/// Arm a projection is run on.
#[derive(Clone, Copy, Debug)]
enum Arm {
    Entry(Entry),
    /// mlx-native's stock `quantized_matmul_ggml` (today's route).
    Stock,
    /// `dispatch_row_exact_matvec` called directly.
    RowExact(RowExactWeight),
}

#[allow(clippy::too_many_arguments)]
fn run(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    arm: Arm,
    weight: &MlxBuffer,
    t: GgmlType,
    x: &[Vec<f32>],
    order: &[usize],
    n: usize,
    k: usize,
) -> Vec<u32> {
    let m = order.len();
    let input = packed_input(device, x, order);
    let mut y = output(device, m, n);
    let mut enc = device.command_encoder().expect("encoder");
    let p = params(t, m, n, k);
    let (m32, n32, k32) = (m as u32, n as u32, k as u32);
    let result = match arm {
        Arm::Entry(Entry::Boundary) => {
            quantized_matmul_ggml(&mut enc, registry, device, &input, weight, &y, &p)
                .map_err(anyhow::Error::from)
        }
        Arm::Entry(Entry::DeltaNetProj) => {
            apply_proj_with_ggml_type(&mut enc, registry, device, &input, weight, t, m32, k32, n32)
                .map(|out| y = out)
        }
        Arm::Entry(Entry::LinearProj) => apply_linear_projection_f32_with_ggml_type(
            &mut enc, registry, device, &input, weight, t, m32, k32, n32,
        )
        .map(|out| y = out),
        Arm::Entry(Entry::LinearInto) => apply_linear_projection_f32_into_with_ggml_type(
            &mut enc, registry, device, &input, weight, t, &mut y, m32, k32, n32,
        ),
        Arm::Stock => {
            stock_quantized_matmul_ggml(&mut enc, registry, device, &input, weight, &y, &p)
                .map_err(anyhow::Error::from)
        }
        Arm::RowExact(kernel) => dispatch_row_exact_matvec(
            &mut enc,
            registry,
            device.metal_device(),
            kernel,
            weight,
            &input,
            &y,
            &RowExactMatvecShape::contiguous(m32, n32, k32),
        )
        .map_err(anyhow::Error::from),
    };
    result.unwrap_or_else(|e| panic!("{arm:?} {t:?} [{n}x{k}] m={m}: {e:#}"));
    enc.commit_and_wait_labeled("test.qwen.rowexact_proj")
        .expect("complete projection");
    bits(&y, m * n)
}

fn random_rows(rng: &mut Rng, k: usize) -> Vec<Vec<f32>> {
    (0..ROWS)
        .map(|_| (0..k).map(|_| rng.unit()).collect())
        .collect()
}

/// Joint row orders: widths 2..=4 (every order of three rows, rows not
/// starting at 0) and the full joint cap of 8.
fn joint_orders() -> Vec<Vec<usize>> {
    let mut orders = vec![vec![0, 1], vec![1, 0], vec![5, 2]];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                if a != b && b != c && a != c {
                    orders.push(vec![a, b, c]);
                }
            }
        }
    }
    orders.push(vec![7, 3, 6]);
    orders.extend([
        vec![0, 1, 2, 3],
        vec![3, 2, 1, 0],
        vec![1, 3, 0, 2],
        vec![6, 4, 7, 5],
    ]);
    orders.push((0..ROWS).collect());
    orders.push((0..ROWS).rev().collect());
    orders
}

/// Full gate for one selected shape.
fn gate_selected_shape(t: GgmlType, n: usize, k: usize, seed: u64) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let mut registry = KernelRegistry::new();
    let label = format!("{t:?} {n}x{k}");
    let mut rng = Rng(seed);
    let weight = random_weight(&device, t, n, k, &mut rng);
    let x = random_rows(&mut rng, k);
    let kernel = row_exact_weight(true, t, n as u32, k as u32, &weight)
        .unwrap_or_else(|| panic!("{label}: selected shape must be admitted"));

    // Switch on.
    with_rowexact_proj_for_test(true, || {
        let solo: Vec<Vec<u32>> = (0..ROWS)
            .map(|row| {
                run(
                    &device,
                    &mut registry,
                    Arm::Entry(Entry::Boundary),
                    &weight,
                    t,
                    &x,
                    &[row],
                    n,
                    k,
                )
            })
            .collect();
        for m in [1, 4, ROWS] {
            let order: Vec<usize> = (0..m).collect();
            let direct = run(
                &device,
                &mut registry,
                Arm::RowExact(kernel),
                &weight,
                t,
                &x,
                &order,
                n,
                k,
            );
            assert_eq!(
                direct,
                solo[..m].concat(),
                "{label}: boundary m=1 rows differ from dispatch_row_exact_matvec at m={m}"
            );
        }
        let orders = joint_orders();
        for entry in ENTRIES {
            for order in &orders {
                let joint = run(
                    &device,
                    &mut registry,
                    Arm::Entry(entry),
                    &weight,
                    t,
                    &x,
                    order,
                    n,
                    k,
                );
                for (slot, &row) in order.iter().enumerate() {
                    assert!(
                        joint[slot * n..(slot + 1) * n] == solo[row][..],
                        "{label}: {entry:?} order {order:?} slot {slot} (row {row}) differs from its m=1 call"
                    );
                }
            }
        }
        let worst = check_reference(&label, &weight, t, n, k, &x, &solo);

        // Prefill width stays stock.
        let wide: Vec<usize> = (0..ROWS).chain([0]).collect();
        let on_wide = run(
            &device,
            &mut registry,
            Arm::Entry(Entry::Boundary),
            &weight,
            t,
            &x,
            &wide,
            n,
            k,
        );
        let stock_wide = run(
            &device,
            &mut registry,
            Arm::Stock,
            &weight,
            t,
            &x,
            &wide,
            n,
            k,
        );
        assert_eq!(
            on_wide, stock_wide,
            "{label}: m=9 must keep the stock route"
        );

        let stock_solo: Vec<u32> = (0..ROWS)
            .flat_map(|row| {
                run(
                    &device,
                    &mut registry,
                    Arm::Stock,
                    &weight,
                    t,
                    &x,
                    &[row],
                    n,
                    k,
                )
            })
            .collect();
        let (mut differing, mut max_abs) = (0usize, 0f32);
        for (a, b) in solo.concat().iter().zip(&stock_solo) {
            if a != b {
                differing += 1;
                max_abs = max_abs.max((f32::from_bits(*a) - f32::from_bits(*b)).abs());
            }
        }
        eprintln!(
            "{label}: row-exact == m=1 over {} orders x {} entries; F64 max |err|/bound {worst:.3e}; vs stock m=1: {differing}/{} cells differ, max |diff| {max_abs:.3e}",
            orders.len(),
            ENTRIES.len(),
            ROWS * n
        );
    });

    // Switch off: every entry is today's stock dispatch, byte for byte.
    with_rowexact_proj_for_test(false, || {
        for order in [vec![0], vec![1, 0], vec![2, 0, 1], vec![3, 1, 2, 0]] {
            let stock = run(
                &device,
                &mut registry,
                Arm::Stock,
                &weight,
                t,
                &x,
                &order,
                n,
                k,
            );
            for entry in ENTRIES {
                let got = run(
                    &device,
                    &mut registry,
                    Arm::Entry(entry),
                    &weight,
                    t,
                    &x,
                    &order,
                    n,
                    k,
                );
                assert!(
                    got == stock,
                    "{label}: switch off, {entry:?} m={} differs from stock",
                    order.len()
                );
            }
        }
    });
}

#[test]
fn switch_parses_like_the_other_qwen_switches() {
    for raw in [
        None,
        Some(""),
        Some("  "),
        Some("0"),
        Some("off"),
        Some("FALSE"),
    ] {
        assert_eq!(parse_qwen_rowexact_proj(raw), Ok(false), "{raw:?}");
    }
    for raw in [Some("1"), Some("on"), Some("True"), Some(" 1 ")] {
        assert_eq!(parse_qwen_rowexact_proj(raw), Ok(true), "{raw:?}");
    }
    assert!(parse_qwen_rowexact_proj(Some("row_exact")).is_err());
}

/// Admission depends on the weight alone (type, N, K, storage, start
/// alignment, size) and on `m` only through the decode-width cap.
#[test]
fn admission_depends_on_the_weight_and_the_decode_width_only() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let bytes = |t: GgmlType, n: usize, k: usize| n * k / 256 * block_bytes(t);
    let u8_buffer = |len: usize| {
        device
            .alloc_buffer(len, DType::U8, vec![len])
            .expect("allocate")
    };
    for &(t, n, k) in &ROW_EXACT_PROJ_SHAPES {
        if n > 8192 {
            continue; // the lm head is admitted in its own gate
        }
        let expected = match t {
            GgmlType::Q4_K => RowExactWeight::Q4_K,
            _ => RowExactWeight::Q6_K,
        };
        let weight = u8_buffer(bytes(t, n as usize, k as usize));
        assert_eq!(row_exact_weight(true, t, n, k, &weight), Some(expected));
        assert_eq!(row_exact_weight(false, t, n, k, &weight), None);
        for m in 1..=8 {
            assert_eq!(
                select(true, &weight, &params(t, m, n as usize, k as usize)),
                Some(expected)
            );
            assert_eq!(
                select(false, &weight, &params(t, m, n as usize, k as usize)),
                None
            );
        }
        for m in [0, 9, 64] {
            assert_eq!(
                select(true, &weight, &params(t, m, n as usize, k as usize)),
                None
            );
        }
        let short = u8_buffer(bytes(t, n as usize, k as usize) - 1);
        assert_eq!(
            row_exact_weight(true, t, n, k, &short),
            None,
            "{t:?} {n}x{k} short"
        );
        let float = device
            .alloc_buffer(
                bytes(t, n as usize, k as usize),
                DType::F32,
                vec![bytes(t, n as usize, k as usize) / 4],
            )
            .expect("allocate");
        assert_eq!(
            row_exact_weight(true, t, n, k, &float),
            None,
            "{t:?} {n}x{k} F32"
        );
    }
    // Shapes outside the measured set keep the stock kernel.
    for (t, n, k) in [
        (GgmlType::Q6_K, 2048, 512),
        (GgmlType::Q6_K, 512, 2048),
        (GgmlType::Q4_K, 1024, 2048),
        (GgmlType::Q5_K, 8192, 2048),
        (GgmlType::Q4_K, 32, 2048),
    ] {
        let weight = u8_buffer(n * k / 256 * 210);
        assert_eq!(
            row_exact_weight(true, t, n as u32, k as u32, &weight),
            None,
            "{t:?} {n}x{k}"
        );
    }
    // Weight start alignment: 16 bytes for Q4_K, 2 for Q6_K.
    let q4 = bytes(GgmlType::Q4_K, 512, 2048);
    let base = u8_buffer(q4 + 16);
    assert!(row_exact_weight(true, GgmlType::Q4_K, 512, 2048, &base.slice_view(16, q4)).is_some());
    assert!(row_exact_weight(true, GgmlType::Q4_K, 512, 2048, &base.slice_view(8, q4)).is_none());
    let q6 = bytes(GgmlType::Q6_K, 8192, 2048);
    let base = u8_buffer(q6 + 2);
    assert!(row_exact_weight(true, GgmlType::Q6_K, 8192, 2048, &base.slice_view(2, q6)).is_some());
    assert!(row_exact_weight(true, GgmlType::Q6_K, 8192, 2048, &base.slice_view(1, q6)).is_none());
}

/// The worker-start `proj` label counts the loaded buffers: only weights in
/// the selected-shape set are eligible; `stock` when the switch is off or
/// none is admitted, `row_exact` when every eligible weight is, else `mixed`.
#[test]
fn route_label_counts_admitted_eligible_buffers() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let q4 = 512 * 2048 / 256 * block_bytes(GgmlType::Q4_K);
    let q6 = 8192 * 2048 / 256 * block_bytes(GgmlType::Q6_K);
    let base = device
        .alloc_buffer(q6 + 16, DType::U8, vec![q6 + 16])
        .expect("allocate");
    let aligned_q4 = base.slice_view(16, q4);
    let misaligned_q4 = base.slice_view(8, q4);
    let q6_weight = base.slice_view(0, q6);
    let excluded_q6 = base.slice_view(0, 512 * 2048 / 256 * 210);
    let count = |enabled: bool, weights: &[(GgmlType, u32, u32, &MlxBuffer)]| {
        let mut count = ProjRouteCount::default();
        for &(t, n, k, weight) in weights {
            count.add(enabled, t, n, k, weight);
        }
        count
    };
    // A Qwen3.6-like mix: two eligible weights plus shapes outside the set
    // (Q6_K attn_v 512x2048, Q4_K ssm_alpha 32x2048), which never count.
    let all_admitted = [
        (GgmlType::Q4_K, 512, 2048, &aligned_q4),
        (GgmlType::Q6_K, 8192, 2048, &q6_weight),
        (GgmlType::Q6_K, 512, 2048, &excluded_q6),
        (GgmlType::Q4_K, 32, 2048, &aligned_q4),
    ];
    let on = count(true, &all_admitted);
    assert_eq!(
        (on.label(), on.to_string()),
        ("row_exact", "2/2".to_string())
    );
    let off = count(false, &all_admitted);
    assert_eq!((off.label(), off.to_string()), ("stock", "0/2".to_string()));
    let partly = count(
        true,
        &[
            (GgmlType::Q4_K, 512, 2048, &misaligned_q4),
            (GgmlType::Q6_K, 8192, 2048, &q6_weight),
        ],
    );
    assert_eq!(
        (partly.label(), partly.to_string()),
        ("mixed", "1/2".to_string())
    );
    let none = count(true, &[(GgmlType::Q4_K, 512, 2048, &misaligned_q4)]);
    assert_eq!(
        (none.label(), none.to_string()),
        ("stock", "0/1".to_string())
    );
    // An artifact with no eligible shape (Qwen3.8 Q4_0/Q8_0) logs stock.
    let other = count(true, &[(GgmlType::Q6_K, 512, 2048, &excluded_q6)]);
    assert_eq!(
        (other.label(), other.to_string()),
        ("stock", "0/0".to_string())
    );
    // The MTP block joins the total.
    let route = ProjRoute {
        main: on,
        mtp: Some(count(true, &[(GgmlType::Q4_K, 512, 2048, &misaligned_q4)])),
    };
    assert_eq!(
        (route.total().label(), route.total().to_string()),
        ("mixed", "2/3".to_string())
    );
    assert_eq!(
        ProjRoute {
            main: on,
            mtp: None
        }
        .total(),
        on
    );
}

/// Q6_K 2048x512 and 512x2048 measured slower on the row-exact kernel and
/// keep the stock kernel at every width with the switch on.
#[test]
fn excluded_shapes_stay_stock_with_the_switch_on() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let mut registry = KernelRegistry::new();
    let mut rng = Rng(0xE7C1_0DED);
    for (n, k) in [(2048, 512), (512, 2048)] {
        let t = GgmlType::Q6_K;
        let weight = random_weight(&device, t, n, k, &mut rng);
        let x = random_rows(&mut rng, k);
        with_rowexact_proj_for_test(true, || {
            for order in [vec![0], vec![1, 0], vec![2, 0, 1], vec![3, 1, 2, 0]] {
                let stock = run(
                    &device,
                    &mut registry,
                    Arm::Stock,
                    &weight,
                    t,
                    &x,
                    &order,
                    n,
                    k,
                );
                for entry in ENTRIES {
                    let got = run(
                        &device,
                        &mut registry,
                        Arm::Entry(entry),
                        &weight,
                        t,
                        &x,
                        &order,
                        n,
                        k,
                    );
                    assert!(
                        got == stock,
                        "Q6_K {n}x{k}: {entry:?} m={} left stock",
                        order.len()
                    );
                }
            }
        });
    }
}

#[test]
fn rows_exact_q4_k_8192x2048() {
    gate_selected_shape(GgmlType::Q4_K, 8192, 2048, 0x0F3_0001);
}

#[test]
fn rows_exact_q4_k_4096x2048() {
    gate_selected_shape(GgmlType::Q4_K, 4096, 2048, 0x0F3_0002);
}

#[test]
fn rows_exact_q4_k_2048x4096() {
    gate_selected_shape(GgmlType::Q4_K, 2048, 4096, 0x0F3_0003);
}

#[test]
fn rows_exact_q6_k_8192x2048() {
    gate_selected_shape(GgmlType::Q6_K, 8192, 2048, 0x0F3_0004);
}

#[test]
fn rows_exact_q4_k_512x2048() {
    gate_selected_shape(GgmlType::Q4_K, 512, 2048, 0x0F3_0005);
}

#[test]
fn rows_exact_q4_k_2048x512() {
    gate_selected_shape(GgmlType::Q4_K, 2048, 512, 0x0F3_0006);
}

#[test]
fn rows_exact_q6_k_lm_head_248320x2048() {
    gate_selected_shape(GgmlType::Q6_K, 248320, 2048, 0x0F3_0007);
}

/// Every selected shape has a gate above.
#[test]
fn every_selected_shape_is_gated() {
    let gated = [
        (GgmlType::Q4_K, 8192, 2048),
        (GgmlType::Q4_K, 4096, 2048),
        (GgmlType::Q4_K, 2048, 4096),
        (GgmlType::Q6_K, 8192, 2048),
        (GgmlType::Q4_K, 512, 2048),
        (GgmlType::Q4_K, 2048, 512),
        (GgmlType::Q6_K, 248320, 2048),
    ];
    assert_eq!(ROW_EXACT_PROJ_SHAPES.len(), gated.len());
    for shape in ROW_EXACT_PROJ_SHAPES {
        assert!(gated.contains(&shape), "{shape:?} has no gate");
    }
}
