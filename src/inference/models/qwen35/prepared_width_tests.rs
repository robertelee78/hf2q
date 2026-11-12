//! Metal gate for the prepared encoders of a joint step at every width
//! (`HF2Q_REPACKED_GGUF_ANY_WIDTH=1`).
//!
//! With the switch on, the dense FFN, the attention and DeltaNet projections
//! and the output head of a DFlash2 target verify step run the prepared
//! kernels at every width up to 32 rows instead of only at 32. Against the
//! stock kernels of the narrower widths this changes bits (Class B), so for
//! a projection, the fused alpha/beta product and the whole dense FFN layer
//! this gate checks:
//! - a row's bits are the same at widths 1, 4, 8, 16, 24 and 32, wherever it
//!   is packed and whichever rows share its step;
//! - padding rows of NaN, infinity, a value that overflows half and a
//!   denormal beside the live rows change nothing;
//! - repeated runs give identical bits;
//! - every output stays within a derived bound of an F64 reference over the
//!   dequantized weights.
//!
//! The tests call the prepared encoders directly, so the process environment
//! does not select the route; `prepared_step_tests` covers the admission and
//! whole steps.

use mlx_native::ops::quantized_matmul_ggml::{GgmlQuantizedMatmulParams, GgmlType};
use mlx_native::{KernelRegistry, MlxBuffer, MlxDevice};

use super::gpu_ffn::{
    build_dense_ffn_layer_gpu_q_prepared_m32, repacked_rows_eligible, DenseFfnWeightsGpuQ,
};
use super::gpu_full_attn::{download_f32, upload_f32};
use super::joint_decode_test_support::{random_f32, synthetic_blocks};
use super::prepared_projection::{self, DeltaNetPlanes, Planes, SharedHead};

/// Joint widths: one row, the MoE target's 4, 8 and 16, three lanes, and 32.
const WIDTHS: [usize; 6] = [1, 4, 8, 16, 24, 32];
const POOL: usize = 32;

/// What a padding row may hold: NaN, infinity, a magnitude that overflows
/// half, and a denormal.
const GARBAGE: [f32; 4] = [f32::NAN, f32::INFINITY, -3.0e38, 1.0e-42];

/// `(width, live rows)` of the padded steps.
fn padded_steps() -> impl Iterator<Item = (usize, usize)> {
    [4_usize, 8, 16, 32]
        .into_iter()
        .flat_map(|width| [1, width / 2, width - 1].map(move |live| (width, live)))
}

/// The pool rows a padded step's live rows hold.
fn live_rows(live: usize) -> Vec<usize> {
    (0..live).map(|i| (i * 5 + 1) % POOL).collect()
}

/// Rows of the pool packed into a `width`-row step three ways: the first
/// rows, the last rows backwards, and a stride through the pool.
fn cohorts(width: usize) -> [Vec<usize>; 3] {
    [
        (0..width).collect(),
        (0..width).map(|i| POOL - 1 - i).collect(),
        (0..width).map(|i| (i * 7 + 3) % POOL).collect(),
    ]
}

/// Source rows whose values are exact in BF16, so the route's input cast
/// loses nothing and the F64 reference reads the same activations.
fn bf16_exact_rows(seed: &mut u32, rows: usize, width: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|_| {
            random_f32(seed, width)
                .into_iter()
                .map(|value| half::bf16::from_f32(value).to_f32())
                .collect()
        })
        .collect()
}

fn pack(device: &MlxDevice, rows: &[Vec<f32>], order: &[usize]) -> MlxBuffer {
    let width = rows[0].len();
    let values: Vec<f32> = order
        .iter()
        .flat_map(|&row| rows[row].iter().copied())
        .collect();
    upload_f32(&values, device)
        .expect("upload packed rows")
        .with_shape(vec![order.len(), width])
        .expect("shape packed rows")
}

/// `order`'s rows followed by rows of `garbage` up to `width` rows.
fn pack_padded(
    device: &MlxDevice,
    rows: &[Vec<f32>],
    order: &[usize],
    width: usize,
    garbage: f32,
) -> MlxBuffer {
    let k = rows[0].len();
    let mut values: Vec<f32> = order
        .iter()
        .flat_map(|&row| rows[row].iter().copied())
        .collect();
    values.resize(width * k, garbage);
    upload_f32(&values, device)
        .expect("upload padded rows")
        .with_shape(vec![width, k])
        .expect("shape padded rows")
}

fn row_bits(buffer: &MlxBuffer, rows: usize, width: usize) -> Vec<Vec<u32>> {
    download_f32(buffer).expect("download rows")[..rows * width]
        .chunks_exact(width)
        .map(|row| row.iter().map(|value| value.to_bits()).collect())
        .collect()
}

fn f16(bytes: &[u8]) -> f64 {
    f64::from(half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
}

/// Dequantize `[rows, cols]` Q4_K or Q6_K blocks to F64 (the GGML layouts).
/// Every weight is finite in half, as [`product_bound`] requires.
fn dequantize(weight: &MlxBuffer, ggml_type: GgmlType, rows: usize, cols: usize) -> Vec<f64> {
    let values = dequantize_blocks(weight, ggml_type, rows, cols);
    assert!(
        values.iter().all(|value| value.abs() < HALF_MAX),
        "a synthetic weight overflows half"
    );
    values
}

fn dequantize_blocks(
    weight: &MlxBuffer,
    ggml_type: GgmlType,
    rows: usize,
    cols: usize,
) -> Vec<f64> {
    let bytes = weight.as_slice::<u8>().expect("map weight blocks");
    let block_bytes = ggml_type.block_bytes() as usize;
    let mut values = Vec::with_capacity(rows * cols);
    for block in bytes[..rows * cols / 256 * block_bytes].chunks_exact(block_bytes) {
        match ggml_type {
            GgmlType::Q4_K => {
                let (d, dmin) = (f16(&block[0..2]), f16(&block[2..4]));
                let scales = &block[4..16];
                let quants = &block[16..144];
                for j in 0..8 {
                    let (scale, minimum) = if j < 4 {
                        (scales[j] & 63, scales[j + 4] & 63)
                    } else {
                        (
                            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
                            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
                        )
                    };
                    for element in 0..32 {
                        let quant = (quants[(j / 2) * 32 + element] >> (4 * (j % 2))) & 0xF;
                        values.push(
                            d * f64::from(scale) * f64::from(quant) - dmin * f64::from(minimum),
                        );
                    }
                }
            }
            GgmlType::Q6_K => {
                let d = f16(&block[208..210]);
                let mut chunk = [0.0_f64; 128];
                for half_block in 0..2 {
                    let low = &block[64 * half_block..64 * half_block + 64];
                    let high = &block[128 + 32 * half_block..128 + 32 * half_block + 32];
                    let scales = &block[192 + 8 * half_block..192 + 8 * half_block + 8];
                    for l in 0..32 {
                        let index = l / 16;
                        let quants = [
                            (low[l] & 0xF) | ((high[l] & 3) << 4),
                            (low[l + 32] & 0xF) | (((high[l] >> 2) & 3) << 4),
                            (low[l] >> 4) | (((high[l] >> 4) & 3) << 4),
                            (low[l + 32] >> 4) | (((high[l] >> 6) & 3) << 4),
                        ];
                        for (quarter, quant) in quants.into_iter().enumerate() {
                            let scale = f64::from(scales[index + 2 * quarter] as i8);
                            chunk[l + 32 * quarter] = d * scale * (f64::from(quant) - 32.0);
                        }
                    }
                    values.extend_from_slice(&chunk);
                }
            }
            other => panic!("no reference dequantization for {other:?}"),
        }
    }
    values
}

/// Largest finite half.
const HALF_MAX: f64 = 65_504.0;

/// An F64 dot product with what its error bounds need.
#[derive(Clone, Copy, Default)]
struct Dot {
    /// `sum x w`.
    value: f64,
    /// `sum |x w|`.
    magnitude: f64,
    /// `sum |x|`.
    abs_x: f64,
}

fn dot(x: impl IntoIterator<Item = f64>, weights: &[f64]) -> Dot {
    x.into_iter()
        .zip(weights)
        .fold(Dot::default(), |dot, (x, w)| Dot {
            value: dot.value + x * w,
            magnitude: dot.magnitude + (x * w).abs(),
            abs_x: dot.abs_x + x.abs(),
        })
}

/// `|output - reference| <= bound` for one prepared product of `k` inputs
/// whose activations the kernel reads exactly (BF16 values).
///
/// The kernel builds each weight in F32 from its block's scale, offset and
/// code (one rounding, relative 2^-24) and rounds it once to half: relative
/// 2^-11 for a weight in half's normal range, and at most 2^-25 absolute for
/// one below it (|w| < 2^-14, where half is subnormal). A weight beyond the
/// largest finite half, 65504, would overflow to infinity; the reference
/// asserts there is none. The products of a half weight and a BF16
/// activation are exact in F32, and the `k` of them are accumulated in F32
/// in an order the kernel fixes, K splits included: at most `k + 8`
/// additions, each relative 2^-24 of a partial sum that the sum of
/// magnitudes bounds. So
///
/// `|error| <= sum|x w| * (2^-11 + 2^-24 + (k + 8) * 2^-24) + sum|x| * 2^-25`.
fn product_bound(dot: &Dot, k: usize) -> f64 {
    let ulp = 2.0_f64.powi(-24);
    dot.magnitude * (2.0_f64.powi(-11) + ulp + (k as f64 + 8.0) * ulp)
        + dot.abs_x * 2.0_f64.powi(-25)
}

fn assert_within(got: u32, reference: f64, bound: f64, label: impl Fn() -> String) {
    let got = f64::from(f32::from_bits(got));
    assert!(
        (got - reference).abs() <= bound,
        "{}: {got} against {reference}, bound {bound}",
        label()
    );
}

#[test]
fn the_width_rule_is_32_rows_or_every_width_up_to_32_with_the_switch() {
    for rows in 0..=40 {
        assert_eq!(repacked_rows_eligible(rows, false), rows == 32, "{rows}");
        assert_eq!(
            repacked_rows_eligible(rows, true),
            (1..=32).contains(&rows),
            "{rows}"
        );
    }
}

/// The F32 -> BF16 cast in front of every prepared product rounds to
/// nearest, ties to even: the relative error 2^-8 the dense FFN bound uses.
#[test]
fn the_gpu_bf16_cast_rounds_to_nearest_even() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let mut seed = 0xCA57_u32;
    let mut values = random_f32(&mut seed, 4_096);
    for (index, value) in values.iter_mut().enumerate() {
        // Magnitudes from 2^-20 to 2^11, and exact ties in between.
        *value *= 2.0_f32.powi(index as i32 % 32 - 20);
        if index % 7 == 0 {
            *value = f32::from_bits((value.to_bits() & 0xFFFF_0000) | 0x8000);
        }
    }
    let input = upload_f32(&values, &device)
        .expect("upload")
        .with_shape(vec![1, values.len()])
        .expect("shape");
    let mut enc = device.command_encoder().expect("encoder");
    let rounded = prepared_projection::bf16_input(
        &mut enc,
        &mut registry,
        &device,
        &input,
        1,
        values.len() as u32,
    )
    .expect("cast");
    enc.commit_and_wait().expect("commit");
    let rounded = rounded.as_slice::<half::bf16>().expect("read BF16");
    for (value, rounded) in values.iter().zip(rounded) {
        assert!(
            rounded.to_bits() == half::bf16::from_f32(*value).to_bits(),
            "{value} cast to {rounded}"
        );
    }
}

/// One prepared projection (the attention and DeltaNet projections and the
/// output head all encode through `prepared_projection::project`).
#[test]
fn a_prepared_projection_row_has_one_value_at_every_width_and_cohort() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    // (type, outputs, inputs): one K run, split K, and a head-like wide one.
    for (ggml_type, n, k) in [
        (GgmlType::Q4_K, 192_usize, 512_usize),
        (GgmlType::Q6_K, 128, 2_048),
        (GgmlType::Q4_K, 64, 4_096),
        (GgmlType::Q6_K, 4_096, 512),
    ] {
        let mut seed = 0x51ED_u32 ^ (n * 31 + k) as u32;
        let weight = synthetic_blocks(&device, ggml_type, n as u32, k as u32, &mut seed);
        let rows = bf16_exact_rows(&mut seed, POOL, k);
        let planes = Planes::default();
        let mut run = |input: &MlxBuffer, live: usize| {
            super::decode_pool::reset_decode_pool();
            let width = input.shape()[0] as u32;
            let mut enc = device.command_encoder().expect("encoder");
            let input_bf16 = prepared_projection::bf16_input(
                &mut enc,
                &mut registry,
                &device,
                input,
                width,
                k as u32,
            )
            .expect("cast input");
            enc.memory_barrier();
            let output = prepared_projection::project(
                &mut enc,
                &mut registry,
                &device,
                &input_bf16,
                &weight,
                ggml_type,
                &planes,
                width,
                k as u32,
                n as u32,
            )
            .expect("prepared projection");
            enc.commit_and_wait().expect("commit");
            row_bits(&output, live, n)
        };
        let all: Vec<usize> = (0..POOL).collect();
        let reference = run(&pack(&device, &rows, &all), POOL);
        assert!(
            reference == run(&pack(&device, &rows, &all), POOL),
            "{ggml_type:?} {n}x{k}: repeat differs"
        );

        let weights = dequantize(&weight, ggml_type, n, k);
        for (row, bits) in reference.iter().enumerate() {
            for (column, &bit) in bits.iter().enumerate() {
                let exact = dot(
                    rows[row].iter().map(|x| f64::from(*x)),
                    &weights[column * k..(column + 1) * k],
                );
                assert_within(bit, exact.value, product_bound(&exact, k), || {
                    format!("{ggml_type:?} {n}x{k} row {row} column {column}")
                });
            }
        }

        for width in WIDTHS {
            for order in cohorts(width) {
                let bits = run(&pack(&device, &rows, &order), width);
                for (position, &row) in order.iter().enumerate() {
                    assert!(
                        bits[position] == reference[row],
                        "{ggml_type:?} {n}x{k} width {width}: row {row} at position {position} differs"
                    );
                }
            }
        }
        for (width, live) in padded_steps() {
            let order = live_rows(live);
            for garbage in GARBAGE {
                let bits = run(&pack_padded(&device, &rows, &order, width, garbage), live);
                for (position, &row) in order.iter().enumerate() {
                    assert!(
                        bits[position] == reference[row],
                        "{ggml_type:?} {n}x{k} width {width}, {live} live rows beside {garbage:e}: row {row} differs"
                    );
                }
            }
        }
    }
}

/// The fused alpha/beta projection of a DeltaNet layer and its split.
#[test]
fn fused_alpha_beta_rows_have_one_value_at_every_width_and_cohort() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let (heads, k) = (48_u32, 512_u32);
    let mut seed = 0xA1FA_u32;
    let alpha = synthetic_blocks(&device, GgmlType::Q4_K, heads, k, &mut seed);
    let beta = synthetic_blocks(&device, GgmlType::Q4_K, heads, k, &mut seed);
    let rows = bf16_exact_rows(&mut seed, POOL, k as usize);
    let planes = DeltaNetPlanes::default();
    let mut run = |input: &MlxBuffer, live: usize| {
        super::decode_pool::reset_decode_pool();
        let width = input.shape()[0] as u32;
        let mut enc = device.command_encoder().expect("encoder");
        let input_bf16 =
            prepared_projection::bf16_input(&mut enc, &mut registry, &device, input, width, k)
                .expect("cast input");
        enc.memory_barrier();
        let fused = prepared_projection::project_alpha_beta(
            &mut enc,
            &mut registry,
            &device,
            &input_bf16,
            &alpha,
            &beta,
            GgmlType::Q4_K,
            &planes,
            width,
            k,
            heads,
        )
        .expect("fused alpha/beta");
        enc.memory_barrier();
        let (alpha_out, beta_out) = prepared_projection::split_alpha_beta(
            &mut enc,
            &mut registry,
            &device,
            &fused,
            width,
            heads,
        )
        .expect("split alpha/beta");
        enc.commit_and_wait().expect("commit");
        (
            row_bits(&alpha_out, live, heads as usize),
            row_bits(&beta_out, live, heads as usize),
        )
    };
    let all: Vec<usize> = (0..POOL).collect();
    let reference = run(&pack(&device, &rows, &all), POOL);
    assert!(reference.0 != reference.1, "alpha and beta rows coincide");
    assert!(
        reference == run(&pack(&device, &rows, &all), POOL),
        "repeat differs"
    );

    let k = k as usize;
    for (name, weight, outputs) in [
        ("alpha", &alpha, &reference.0),
        ("beta", &beta, &reference.1),
    ] {
        let weights = dequantize(weight, GgmlType::Q4_K, heads as usize, k);
        for (row, bits) in outputs.iter().enumerate() {
            for (head, &bit) in bits.iter().enumerate() {
                let exact = dot(
                    rows[row].iter().map(|x| f64::from(*x)),
                    &weights[head * k..(head + 1) * k],
                );
                assert_within(bit, exact.value, product_bound(&exact, k), || {
                    format!("{name} row {row} head {head}")
                });
            }
        }
    }

    let same = |bits: &(Vec<Vec<u32>>, Vec<Vec<u32>>), order: &[usize], label: &str| {
        for (position, &row) in order.iter().enumerate() {
            assert!(
                bits.0[position] == reference.0[row] && bits.1[position] == reference.1[row],
                "{label}: row {row} at position {position} differs"
            );
        }
    };
    for width in WIDTHS {
        for order in cohorts(width) {
            let bits = run(&pack(&device, &rows, &order), width);
            same(&bits, &order, &format!("width {width}"));
        }
    }
    for (width, live) in padded_steps() {
        let order = live_rows(live);
        for garbage in GARBAGE {
            let bits = run(&pack_padded(&device, &rows, &order, width, garbage), live);
            same(
                &bits,
                &order,
                &format!("width {width}, {live} live rows beside {garbage:e}"),
            );
        }
    }
}

/// `|output - reference| <= bound` for every output of one activation row
/// through the prepared dense FFN layer, `residual + down(BF16(SiLU(gate x)
/// * (up x)))`, and the reference outputs.
///
/// Gate and up are prepared products of the exact BF16 input: errors `eg`,
/// `eu` from [`product_bound`]. The hidden activation `h = SiLU(g) u` is
/// computed in F32 from them. |SiLU'| is at most 1.1, so the input errors
/// move it by at most `1.1 eg (|u| + eu) + |SiLU(g)| eu`; the F32
/// exponential, sum, quotient and product add a relative 2^-20 (a few ulp
/// each, fast math included). The cast to BF16 rounds to nearest
/// (`the_gpu_bf16_cast_rounds_to_nearest_even`): relative 2^-8, the
/// dominant term. With `eh` the total per hidden element, the down product
/// of the rounded activations differs from the exact one by at most
/// `sum eh |w|` (times 1 + 2^-10 for the weight's own rounding) plus the
/// product's bound over the activations it actually reads, of magnitude at
/// most `|h| + eh`. The final F32 residual add rounds once more.
fn dense_ffn_reference(
    x: &[f32],
    residual: &[f32],
    gate: &[f64],
    up: &[f64],
    down: &[f64],
) -> Vec<(f64, f64)> {
    let (hidden, intermediate) = (x.len(), gate.len() / x.len());
    let ulp = 2.0_f64.powi(-24);
    let x = || x.iter().map(|value| f64::from(*value));
    // Per hidden element: |h| and its error bound eh.
    let (h, eh): (Vec<f64>, Vec<f64>) = (0..intermediate)
        .map(|i| {
            let g = dot(x(), &gate[i * hidden..(i + 1) * hidden]);
            let u = dot(x(), &up[i * hidden..(i + 1) * hidden]);
            let (eg, eu) = (product_bound(&g, hidden), product_bound(&u, hidden));
            let silu = g.value / (1.0 + (-g.value).exp());
            let h = silu * u.value;
            let moved = 1.1 * eg * (u.value.abs() + eu) + silu.abs() * eu;
            let computed = moved + (h.abs() + moved) * 2.0_f64.powi(-20);
            let rounded = computed + (h.abs() + computed) * 2.0_f64.powi(-8);
            (h, rounded + f64::from(f32::MIN_POSITIVE))
        })
        .unzip();
    (0..hidden)
        .map(|j| {
            let weights = &down[j * intermediate..(j + 1) * intermediate];
            let exact = dot(h.iter().copied(), weights);
            let carried: f64 = eh.iter().zip(weights).map(|(eh, w)| eh * w.abs()).sum();
            let read = dot(h.iter().zip(&eh).map(|(h, eh)| h.abs() + eh), weights);
            let product = carried * (1.0 + 2.0_f64.powi(-10)) + product_bound(&read, intermediate);
            let residual = f64::from(residual[j]);
            let value = exact.value + residual;
            (
                value,
                product + (exact.value.abs() + residual.abs() + product) * ulp,
            )
        })
        .collect()
}

/// The whole prepared dense FFN layer with its residual.
#[test]
fn a_prepared_dense_ffn_row_has_one_value_at_every_width_and_cohort() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let (hidden, intermediate) = (512_u32, 1_024_u32);
    for (gate_up_type, down_type) in [
        (GgmlType::Q4_K, GgmlType::Q6_K),
        (GgmlType::Q4_K, GgmlType::Q4_K),
    ] {
        let mut seed = 0xFF17_u32 ^ down_type.block_bytes();
        let weights = DenseFfnWeightsGpuQ {
            gate_q: synthetic_blocks(&device, gate_up_type, intermediate, hidden, &mut seed),
            up_q: synthetic_blocks(&device, gate_up_type, intermediate, hidden, &mut seed),
            down_q: synthetic_blocks(&device, down_type, hidden, intermediate, &mut seed),
            ggml_type_gate_up: gate_up_type,
            ggml_type_down: down_type,
            intermediate_size: intermediate,
            hidden_size: hidden,
            repacked_m32: Default::default(),
        };
        // Small activations keep the gate sums inside the SiLU's curved part.
        let rows: Vec<Vec<f32>> = bf16_exact_rows(&mut seed, POOL, hidden as usize)
            .into_iter()
            .map(|row| row.into_iter().map(|value| value / 64.0).collect())
            .collect();
        let residuals = bf16_exact_rows(&mut seed, POOL, hidden as usize);
        let mut run = |x: &MlxBuffer, residual: &MlxBuffer, live: usize| {
            super::decode_pool::reset_decode_pool();
            let width = x.shape()[0] as u32;
            let mut enc = device.command_encoder().expect("encoder");
            let output = build_dense_ffn_layer_gpu_q_prepared_m32(
                &mut enc,
                &device,
                &mut registry,
                x,
                &weights,
                Some(residual),
                &GgmlQuantizedMatmulParams {
                    m: width,
                    n: intermediate,
                    k: hidden,
                    ggml_type: gate_up_type,
                },
                &GgmlQuantizedMatmulParams {
                    m: width,
                    n: hidden,
                    k: intermediate,
                    ggml_type: down_type,
                },
            )
            .expect("prepared dense FFN");
            enc.commit_and_wait().expect("commit");
            row_bits(&output, live, hidden as usize)
        };
        let all: Vec<usize> = (0..POOL).collect();
        let packed = |order: &[usize]| {
            (
                pack(&device, &rows, order),
                pack(&device, &residuals, order),
            )
        };
        let (x, residual) = packed(&all);
        let reference = run(&x, &residual, POOL);
        assert!(
            reference == run(&x, &residual, POOL),
            "down {down_type:?}: repeat differs"
        );
        // The layer does something: its output is not the residual.
        let moved = reference
            .iter()
            .zip(&residuals)
            .flat_map(|(bits, residual)| bits.iter().zip(residual))
            .filter(|(bit, residual)| **bit != residual.to_bits())
            .count();
        assert!(
            moved > POOL * hidden as usize / 2,
            "down {down_type:?}: only {moved} outputs differ from the residual"
        );

        let (h, m) = (hidden as usize, intermediate as usize);
        let gate = dequantize(&weights.gate_q, gate_up_type, m, h);
        let up = dequantize(&weights.up_q, gate_up_type, m, h);
        let down = dequantize(&weights.down_q, down_type, h, m);
        // Eight rows: the reference is 2^21 products per row and tensor.
        for row in 0..8 {
            let exact = dense_ffn_reference(&rows[row], &residuals[row], &gate, &up, &down);
            for (column, (&bit, (value, bound))) in reference[row].iter().zip(exact).enumerate() {
                assert_within(bit, value, bound, || {
                    format!("down {down_type:?} row {row} column {column}")
                });
            }
        }

        for width in WIDTHS {
            for order in cohorts(width) {
                let (x, residual) = packed(&order);
                let bits = run(&x, &residual, width);
                for (position, &row) in order.iter().enumerate() {
                    assert!(
                        bits[position] == reference[row],
                        "down {down_type:?} width {width}: row {row} at position {position} differs"
                    );
                }
            }
        }
        for (width, live) in padded_steps() {
            let order = live_rows(live);
            for garbage in GARBAGE {
                let bits = run(
                    &pack_padded(&device, &rows, &order, width, garbage),
                    &pack_padded(&device, &residuals, &order, width, garbage),
                    live,
                );
                for (position, &row) in order.iter().enumerate() {
                    assert!(
                        bits[position] == reference[row],
                        "down {down_type:?} width {width}, {live} live rows beside {garbage:e}: row {row} differs"
                    );
                }
            }
        }
    }
}

/// Planes are published only after the command buffer that writes them has
/// completed: a failed preparation leaves the cell empty for a retry, and an
/// encoder abandoned after it read the planes leaves valid planes behind.
#[test]
fn planes_are_published_only_after_their_own_command_buffer_completed() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let (n, k, width) = (128_usize, 512_usize, 5_usize);
    let mut seed = 0xABA9_u32;
    let weight = synthetic_blocks(&device, GgmlType::Q4_K, n as u32, k as u32, &mut seed);
    let rows = bf16_exact_rows(&mut seed, POOL, k);
    let order: Vec<usize> = (0..width).collect();
    let input = pack(&device, &rows, &order);
    // Encode one projection; `commit` runs it, otherwise the encoder is
    // dropped as on an error return between encode and commit.
    let mut project =
        |planes: &Planes, weight: &MlxBuffer, commit: bool| -> anyhow::Result<Vec<Vec<u32>>> {
            super::decode_pool::reset_decode_pool();
            let mut enc = device.command_encoder().expect("encoder");
            let input_bf16 = prepared_projection::bf16_input(
                &mut enc,
                &mut registry,
                &device,
                &input,
                width as u32,
                k as u32,
            )
            .expect("cast input");
            enc.memory_barrier();
            let output = prepared_projection::project(
                &mut enc,
                &mut registry,
                &device,
                &input_bf16,
                weight,
                GgmlType::Q4_K,
                planes,
                width as u32,
                k as u32,
                n as u32,
            )?;
            if commit {
                enc.commit_and_wait().expect("commit");
                Ok(row_bits(&output, width, n))
            } else {
                Ok(Vec::new())
            }
        };
    let reference = project(&Planes::default(), &weight, true).expect("reference");
    assert!(reference.iter().flatten().any(|&bit| bit != 0));

    // A preparation that fails publishes nothing, and the retry prepares.
    let planes = Planes::default();
    let short = synthetic_blocks(&device, GgmlType::Q4_K, n as u32 / 2, k as u32, &mut seed);
    assert!(
        project(&planes, &short, true).is_err(),
        "a short weight was prepared"
    );
    assert!(planes.get().is_none(), "a failed preparation was published");
    assert!(
        project(&planes, &weight, true).expect("retry") == reference,
        "the retry after a failed preparation differs"
    );

    // An encoder abandoned after its first use of a weight: the planes it
    // published were written by their own command buffer, so the next use
    // reads the weight, not zeros.
    let planes = Planes::default();
    project(&planes, &weight, false).expect("abandoned encode");
    assert!(planes.get().is_some());
    assert!(
        project(&planes, &weight, true).expect("after an abandoned encoder") == reference,
        "planes published by an abandoned encoder are not the weight's"
    );
}

/// The same for the three cells of a dense FFN layer, prepared together,
/// and for the fused alpha/beta cell.
#[test]
fn ffn_and_alpha_beta_planes_survive_an_abandoned_encoder() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let (hidden, intermediate, width) = (256_u32, 512_u32, 5_usize);
    let mut seed = 0xABAA_u32;
    let blocks = |seed: &mut u32, ggml_type, n, k| synthetic_blocks(&device, ggml_type, n, k, seed);
    let (gate_q, up_q, down_q) = (
        blocks(&mut seed, GgmlType::Q4_K, intermediate, hidden),
        blocks(&mut seed, GgmlType::Q4_K, intermediate, hidden),
        blocks(&mut seed, GgmlType::Q6_K, hidden, intermediate),
    );
    let layer = || DenseFfnWeightsGpuQ {
        gate_q: gate_q.clone(),
        up_q: up_q.clone(),
        down_q: down_q.clone(),
        ggml_type_gate_up: GgmlType::Q4_K,
        ggml_type_down: GgmlType::Q6_K,
        intermediate_size: intermediate,
        hidden_size: hidden,
        repacked_m32: Default::default(),
    };
    let rows: Vec<Vec<f32>> = bf16_exact_rows(&mut seed, POOL, hidden as usize)
        .into_iter()
        .map(|row| row.into_iter().map(|value| value / 64.0).collect())
        .collect();
    let order: Vec<usize> = (0..width).collect();
    let x = pack(&device, &rows, &order);
    let mut ffn = |weights: &DenseFfnWeightsGpuQ, commit: bool| {
        super::decode_pool::reset_decode_pool();
        let mut enc = device.command_encoder().expect("encoder");
        let output = build_dense_ffn_layer_gpu_q_prepared_m32(
            &mut enc,
            &device,
            &mut registry,
            &x,
            weights,
            None,
            &GgmlQuantizedMatmulParams {
                m: width as u32,
                n: intermediate,
                k: hidden,
                ggml_type: GgmlType::Q4_K,
            },
            &GgmlQuantizedMatmulParams {
                m: width as u32,
                n: hidden,
                k: intermediate,
                ggml_type: GgmlType::Q6_K,
            },
        )
        .expect("prepared dense FFN");
        if commit {
            enc.commit_and_wait().expect("commit");
            row_bits(&output, width, hidden as usize)
        } else {
            Vec::new()
        }
    };
    let reference = ffn(&layer(), true);
    assert!(reference.iter().flatten().any(|&bit| bit != 0));
    let abandoned = layer();
    ffn(&abandoned, false);
    assert!(
        ffn(&abandoned, true) == reference,
        "FFN planes published by an abandoned encoder are not the weights'"
    );

    let (heads, k) = (48_u32, 512_u32);
    let alpha = blocks(&mut seed, GgmlType::Q4_K, heads, k);
    let beta = blocks(&mut seed, GgmlType::Q4_K, heads, k);
    let input = pack(
        &device,
        &bf16_exact_rows(&mut seed, width, k as usize),
        &order,
    );
    let mut alpha_beta = |planes: &DeltaNetPlanes, commit: bool| {
        super::decode_pool::reset_decode_pool();
        let mut enc = device.command_encoder().expect("encoder");
        let input_bf16 = prepared_projection::bf16_input(
            &mut enc,
            &mut registry,
            &device,
            &input,
            width as u32,
            k,
        )
        .expect("cast input");
        enc.memory_barrier();
        let fused = prepared_projection::project_alpha_beta(
            &mut enc,
            &mut registry,
            &device,
            &input_bf16,
            &alpha,
            &beta,
            GgmlType::Q4_K,
            planes,
            width as u32,
            k,
            heads,
        )
        .expect("fused alpha/beta");
        if commit {
            enc.commit_and_wait().expect("commit");
            row_bits(
                &fused,
                width,
                prepared_projection::alpha_beta_rows(heads) as usize,
            )
        } else {
            Vec::new()
        }
    };
    let reference = alpha_beta(&DeltaNetPlanes::default(), true);
    assert!(reference.iter().flatten().any(|&bit| bit != 0));
    let abandoned = DeltaNetPlanes::default();
    alpha_beta(&abandoned, false);
    assert!(
        alpha_beta(&abandoned, true) == reference,
        "alpha/beta planes published by an abandoned encoder are not the weights'"
    );
}

/// The shared head is prepared once, gives the bits of a consumer's own
/// planes, and refuses a head that was replaced after it was prepared.
#[test]
fn the_shared_head_is_prepared_once_and_matches_a_consumers_own_planes() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let (vocab, hidden) = (1_024_u32, 512_u32);
    let mut seed = 0x5EAD_u32;
    let head = synthetic_blocks(&device, GgmlType::Q6_K, vocab, hidden, &mut seed);
    let rows = bf16_exact_rows(&mut seed, POOL, hidden as usize);
    let shared = SharedHead::default();
    let own = Planes::default();
    for width in WIDTHS {
        let order: Vec<usize> = (0..width).collect();
        let input = pack(&device, &rows, &order);
        let rows_u32 = width as u32;
        let planes = shared
            .get_or_prepare(&mut registry, &device, &head, GgmlType::Q6_K, hidden, vocab)
            .expect("shared head");
        let first = shared
            .get_or_prepare(&mut registry, &device, &head, GgmlType::Q6_K, hidden, vocab)
            .expect("shared head again");
        assert!(
            std::ptr::eq(planes, first),
            "the shared head was prepared twice"
        );
        let mut enc = device.command_encoder().expect("encoder");
        let input_bf16 = prepared_projection::bf16_input(
            &mut enc,
            &mut registry,
            &device,
            &input,
            rows_u32,
            hidden,
        )
        .expect("cast input");
        enc.memory_barrier();
        let from_shared = prepared_projection::project_prepared(
            &mut enc,
            &mut registry,
            &device,
            &input_bf16,
            planes,
            GgmlType::Q6_K,
            rows_u32,
            hidden,
            vocab,
        )
        .expect("shared projection");
        let from_own = prepared_projection::project(
            &mut enc,
            &mut registry,
            &device,
            &input_bf16,
            &head,
            GgmlType::Q6_K,
            &own,
            rows_u32,
            hidden,
            vocab,
        )
        .expect("own projection");
        enc.commit_and_wait().expect("commit");
        let shared_bits = row_bits(&from_shared, width, vocab as usize);
        assert!(shared_bits.iter().flatten().any(|&bit| bit != 0));
        assert!(
            shared_bits == row_bits(&from_own, width, vocab as usize),
            "width {width}: shared planes differ from a consumer's own"
        );
    }
    let replacement = synthetic_blocks(&device, GgmlType::Q6_K, vocab, hidden, &mut seed);
    let error = shared
        .get_or_prepare(
            &mut registry,
            &device,
            &replacement,
            GgmlType::Q6_K,
            hidden,
            vocab,
        )
        .err()
        .expect("a replaced head must not read the old planes");
    assert!(error.to_string().contains("replaced"), "{error}");
}
