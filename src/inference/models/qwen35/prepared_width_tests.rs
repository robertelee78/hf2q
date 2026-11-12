//! Metal gate for the prepared joint routes at every joint width
//! (`HF2Q_REPACKED_GGUF_ANY_WIDTH=1`).
//!
//! With the switch on, the dense FFN, the attention and DeltaNet projections
//! and the output head of a joint step run the prepared kernels at every
//! width up to 32 rows instead of only at 32. Against the stock kernels of
//! the narrower widths this changes bits (Class B), so this gate checks:
//! - a row's bits are the same at widths 1, 4, 8, 16, 24 and 32, wherever it
//!   is packed and whichever rows share its step;
//! - repeated runs give identical bits;
//! - every output stays within a derived bound of an F64 reference over the
//!   dequantized weights.
//!
//! The tests call the prepared encoders directly, so the process environment
//! does not select the route.

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
fn dequantize(weight: &MlxBuffer, ggml_type: GgmlType, rows: usize, cols: usize) -> Vec<f64> {
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

/// `|output - reference| <= bound` for one prepared product of K inputs.
///
/// The kernel rounds each dequantized weight once to half (relative error at
/// most 2^-11), multiplies it by a BF16 activation that is exact here, and
/// accumulates the products in F32: at most K additions, each relative
/// 2^-24 of a partial sum bounded by the sum of magnitudes. The K splits are
/// added in F32 as well, which the same term covers. So the error is at most
/// `sum |x w| * (2^-11 + (K + 8) * 2^-24)`; the constant doubles it for the
/// F32 rounding of the scale and offset the weight is built from.
fn product_bound(magnitude: f64, k: usize) -> f64 {
    2.0 * magnitude * (2.0_f64.powi(-11) + (k as f64 + 8.0) * 2.0_f64.powi(-24)) + 1.0e-30
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
        let mut run = |order: &[usize]| {
            let width = order.len() as u32;
            let input = pack(&device, &rows, order);
            let mut enc = device.command_encoder().expect("encoder");
            let input_bf16 = prepared_projection::bf16_input(
                &mut enc,
                &mut registry,
                &device,
                &input,
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
            row_bits(&output, order.len(), n)
        };
        let all: Vec<usize> = (0..POOL).collect();
        let reference = run(&all);
        assert_eq!(reference, run(&all), "{ggml_type:?} {n}x{k}: repeat differs");

        let weights = dequantize(&weight, ggml_type, n, k);
        for (row, bits) in reference.iter().enumerate() {
            for (column, &bit) in bits.iter().enumerate() {
                let weight_row = &weights[column * k..(column + 1) * k];
                let (mut exact, mut magnitude) = (0.0_f64, 0.0_f64);
                for (x, w) in rows[row].iter().zip(weight_row) {
                    exact += f64::from(*x) * w;
                    magnitude += (f64::from(*x) * w).abs();
                }
                let got = f64::from(f32::from_bits(bit));
                assert!(
                    (got - exact).abs() <= product_bound(magnitude, k),
                    "{ggml_type:?} {n}x{k} row {row} column {column}: {got} vs {exact}, bound {}",
                    product_bound(magnitude, k)
                );
            }
        }

        for width in WIDTHS {
            for order in cohorts(width) {
                let bits = run(&order);
                for (position, &row) in order.iter().enumerate() {
                    assert!(
                        bits[position] == reference[row],
                        "{ggml_type:?} {n}x{k} width {width}: row {row} at position {position} differs"
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
    let mut run = |order: &[usize]| {
        let width = order.len() as u32;
        let input = pack(&device, &rows, order);
        let mut enc = device.command_encoder().expect("encoder");
        let input_bf16 =
            prepared_projection::bf16_input(&mut enc, &mut registry, &device, &input, width, k)
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
            row_bits(&alpha_out, order.len(), heads as usize),
            row_bits(&beta_out, order.len(), heads as usize),
        )
    };
    let all: Vec<usize> = (0..POOL).collect();
    let reference = run(&all);
    assert!(reference.0 != reference.1, "alpha and beta rows coincide");
    for width in WIDTHS {
        for order in cohorts(width) {
            let (alpha_bits, beta_bits) = run(&order);
            for (position, &row) in order.iter().enumerate() {
                assert!(
                    alpha_bits[position] == reference.0[row]
                        && beta_bits[position] == reference.1[row],
                    "width {width}: row {row} at position {position} differs"
                );
            }
        }
    }
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
        let mut run = |order: &[usize]| {
            let width = order.len() as u32;
            let x = pack(&device, &rows, order);
            let residual = pack(&device, &residuals, order);
            let mut enc = device.command_encoder().expect("encoder");
            let output = build_dense_ffn_layer_gpu_q_prepared_m32(
                &mut enc,
                &device,
                &mut registry,
                &x,
                &weights,
                Some(&residual),
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
            row_bits(&output, order.len(), hidden as usize)
        };
        let all: Vec<usize> = (0..POOL).collect();
        let reference = run(&all);
        assert_eq!(reference, run(&all), "down {down_type:?}: repeat differs");
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
        for width in WIDTHS {
            for order in cohorts(width) {
                let bits = run(&order);
                for (position, &row) in order.iter().enumerate() {
                    assert!(
                        bits[position] == reference[row],
                        "down {down_type:?} width {width}: row {row} at position {position} differs"
                    );
                }
            }
        }
    }
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
        assert!(std::ptr::eq(planes, first), "the shared head was prepared twice");
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
