//! Metal gate for the weight-shared joint dense FFN
//! (`JointDenseFfnRoute::SharedWeights`, i.e. `DenseGateUpDispatch::Separate`
//! at the padded width).
//!
//! The shared route is not bitwise equal to the serial builder's fused
//! gate/up/SiLU kernel (Q1 Class B), so this gate checks what it must still
//! guarantee:
//! - a row's bytes do not depend on which rows share the step or where it is
//!   packed (cohort invariance, Q1 §3.1 item 2);
//! - every row stays within F32 reordering error of the serial `m = 1`
//!   builder.
//!
//! The routing policy is frozen on the test registry (`mul_mv_ext` on, `mvN`
//! off as Qwen3.8 serving sets it, then `mvN` on), so the process
//! environment does not select the kernels.

use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{GgmlRoutingPolicy, KernelRegistry, MlxBuffer, MlxDevice};

use super::gpu_ffn::{
    build_dense_ffn_layer_gpu_q_into, build_dense_ffn_layer_gpu_q_into_with_gate_up,
    DenseFfnWeightsGpuQ, DenseGateUpDispatch,
};
use super::gpu_full_attn::{download_f32, upload_f32};
use super::joint_decode_routes::PADDED_MV_EXT_WIDTH;
use super::joint_decode_test_support::{bits, random_f32, synthetic_blocks};

const WIDTH: usize = PADDED_MV_EXT_WIDTH;

/// Packed orders over four source rows: identity, reversed, a padded step
/// (one real row replayed into every packed row), and a rotation.
const ORDERS: [[usize; WIDTH]; 4] = [[0, 1, 2, 3], [3, 2, 1, 0], [2, 2, 2, 2], [1, 3, 0, 2]];

fn registry_with_mv_ext(dense_decode_mvn: bool) -> KernelRegistry {
    let mut registry = KernelRegistry::new();
    registry
        .freeze_ggml_routing_policy(GgmlRoutingPolicy {
            dense_decode_mvn,
            dense_decode_mv_ext: true,
            ..GgmlRoutingPolicy::default()
        })
        .expect("freeze padded-mode routing policy");
    registry
}

fn pack(device: &MlxDevice, rows: &[Vec<f32>], order: &[usize], h: usize) -> MlxBuffer {
    let values = order
        .iter()
        .flat_map(|&row| rows[row].iter().copied())
        .collect::<Vec<_>>();
    upload_f32(&values, device)
        .expect("upload packed rows")
        .with_shape(vec![order.len(), h])
        .expect("shape packed rows")
}

#[test]
fn shared_gate_up_rows_are_cohort_invariant_and_track_the_serial_builder() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = match MlxDevice::new() {
        Ok(device) => device,
        Err(_) => return,
    };
    let (hidden, intermediate) = (256_u32, 512_u32);
    let h = hidden as usize;
    let mut seed = 0x51A7_0003_u32;
    for dense_decode_mvn in [false, true] {
        let mut registry = registry_with_mv_ext(dense_decode_mvn);
        for (gate_up, down) in [
            (GgmlType::Q4_K, GgmlType::Q6_K),
            (GgmlType::Q4_K, GgmlType::Q4_K),
            (GgmlType::Q5_K, GgmlType::Q6_K),
            (GgmlType::Q8_0, GgmlType::Q8_0),
            (GgmlType::Q6_K, GgmlType::Q4_K),
        ] {
            let at = format!("{gate_up:?}/{down:?} mvn={dense_decode_mvn}");
            let weights = DenseFfnWeightsGpuQ {
                gate_q: synthetic_blocks(&device, gate_up, intermediate, hidden, &mut seed),
                up_q: synthetic_blocks(&device, gate_up, intermediate, hidden, &mut seed),
                down_q: synthetic_blocks(&device, down, hidden, intermediate, &mut seed),
                ggml_type_gate_up: gate_up,
                ggml_type_down: down,
                intermediate_size: intermediate,
                hidden_size: hidden,
            };
            let x = (0..WIDTH)
                .map(|_| random_f32(&mut seed, h))
                .collect::<Vec<_>>();
            let residual = (0..WIDTH)
                .map(|_| random_f32(&mut seed, h))
                .collect::<Vec<_>>();
            let mut inputs = Vec::new();
            let mut shared = Vec::with_capacity(ORDERS.len());
            let mut serial = Vec::with_capacity(WIDTH);
            let mut enc = device.command_encoder().expect("encoder");
            for order in &ORDERS {
                let packed_x = pack(&device, &x, order, h);
                let packed_residual = pack(&device, &residual, order, h);
                enc.memory_barrier();
                shared.push(
                    build_dense_ffn_layer_gpu_q_into_with_gate_up(
                        &mut enc,
                        &device,
                        &mut registry,
                        &packed_x,
                        &weights,
                        Some(&packed_residual),
                        DenseGateUpDispatch::Separate,
                    )
                    .expect("shared dense FFN"),
                );
                inputs.extend([packed_x, packed_residual]);
            }
            for row in 0..WIDTH {
                let row_x = pack(&device, &x, &[row], h);
                let row_residual = pack(&device, &residual, &[row], h);
                enc.memory_barrier();
                serial.push(
                    build_dense_ffn_layer_gpu_q_into(
                        &mut enc,
                        &device,
                        &mut registry,
                        &row_x,
                        &weights,
                        Some(&row_residual),
                    )
                    .expect("serial m=1 dense FFN"),
                );
                inputs.extend([row_x, row_residual]);
            }
            enc.commit_and_wait_labeled("test.qwen.joint.shared_ffn")
                .expect("complete shared dense FFN");
            drop(inputs);

            let reference = bits(&shared[0], WIDTH * h);
            for (order, output) in ORDERS.iter().zip(&shared).skip(1) {
                let output = bits(output, WIDTH * h);
                for (packed_row, &source) in order.iter().enumerate() {
                    assert!(
                        output[packed_row * h..(packed_row + 1) * h]
                            == reference[source * h..(source + 1) * h],
                        "{at}: source row {source} packed at {packed_row} in {order:?} \
                         differs from the identity order"
                    );
                }
            }
            for (row, serial_output) in serial.iter().enumerate() {
                let expected = &download_f32(serial_output).expect("download serial row")[..h];
                let scale = expected
                    .iter()
                    .fold(1.0_f32, |scale, value| scale.max(value.abs()));
                let mut worst = 0.0_f32;
                for (&got, &want) in reference[row * h..(row + 1) * h].iter().zip(expected) {
                    let got = f32::from_bits(got);
                    assert!(
                        got.is_finite() && want.is_finite(),
                        "{at}: row {row} non-finite output (shared {got}, serial {want})"
                    );
                    worst = worst.max((got - want).abs());
                }
                assert!(
                    worst <= 1e-3 * scale,
                    "{at}: row {row} max |shared - serial| = {worst} exceeds 1e-3 x {scale}"
                );
            }
        }
    }
}
