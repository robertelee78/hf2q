use super::*;

use half::bf16;

const K: u32 = 64;

fn input_values(rows: u32) -> Vec<f32> {
    (0..rows as usize * K as usize)
        .map(|index| ((index * 19 % 97) as f32 - 48.0) / 64.0)
        .collect()
}

fn weight_bits(rows: u32) -> Vec<u16> {
    (0..rows as usize * K as usize)
        .map(|index| {
            let value = ((index * 23 % 113) as f32 - 56.0) / 128.0;
            bf16::from_f32(value).to_bits()
        })
        .collect()
}

fn upload_bf16_weight(device: &MlxDevice, rows: u32, bits: &[u16]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(bits.len() * 2, DType::BF16, vec![rows as usize, K as usize])
        .expect("allocate BF16 weight");
    buffer
        .as_mut_slice::<u16>()
        .expect("view BF16 weight")
        .copy_from_slice(bits);
    buffer
}

fn cpu_projection(input: &[f32], weight: &[u16], rows: u32, outputs: u32) -> Vec<f32> {
    let mut result = vec![0.0; rows as usize * outputs as usize];
    for row in 0..rows as usize {
        for output in 0..outputs as usize {
            let mut sum = 0.0f32;
            for feature in 0..K as usize {
                sum += input[row * K as usize + feature]
                    * bf16::from_bits(weight[output * K as usize + feature]).to_f32();
            }
            result[row * outputs as usize + output] = sum;
        }
    }
    result
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{label} produced non-finite output"
    );
    assert!(
        actual.iter().any(|value| *value != 0.0),
        "{label} produced only zeroes"
    );
    let (index, max_abs) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("nonempty projection");
    assert!(
        max_abs <= 2.0e-3,
        "{label} max_abs={max_abs} at {index}: actual={} expected={}",
        actual[index],
        expected[index]
    );
}

fn run_allocating_projection(
    device: &MlxDevice,
    rows: u32,
    outputs: u32,
) -> (Vec<f32>, Vec<f32>, bool) {
    let input = input_values(rows);
    let weights = weight_bits(outputs);
    let input_buffer = upload_f32(&input, device).expect("upload input");
    let weight_buffer = upload_bf16_weight(device, outputs, &weights);
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().expect("command encoder");
    let output = apply_proj(
        &mut encoder,
        &mut registry,
        device,
        &input_buffer,
        &weight_buffer,
        rows,
        K,
        outputs,
    )
    .expect("apply BF16 projection");
    encoder
        .commit_and_wait_labeled("qwen35.delta.bf16_projection.allocating")
        .expect("complete BF16 projection");
    let compiled_gemv = registry
        .pipeline_identity("hf2q_dense_gemv_bf16_f32_4")
        .is_ok();
    let mut downloaded = download_f32(&output).expect("download projection");
    downloaded.truncate(rows as usize * outputs as usize);
    (
        downloaded,
        cpu_projection(&input, &weights, rows, outputs),
        compiled_gemv,
    )
}

#[test]
fn delta_bf16_projection_route_is_explicit_and_odd_width_safe() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    assert_eq!(bf16_projection_route(1, 32), Bf16ProjectionRoute::Gemv);
    assert_eq!(bf16_projection_route(1, 48), Bf16ProjectionRoute::Gemv);
    assert_eq!(bf16_projection_route(1, 33), Bf16ProjectionRoute::TensorMm);
    assert_eq!(bf16_projection_route(2, 32), Bf16ProjectionRoute::TensorMm);
}

#[test]
fn delta_bf16_projection_helpers_use_gemv_only_for_safe_decode() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");

    let (decode, decode_expected, decode_compiled_gemv) = run_allocating_projection(&device, 1, 32);
    assert!(decode_compiled_gemv, "M=1 even-N did not compile GEMV");
    assert_close(&decode, &decode_expected, "allocating M=1 GEMV");

    let (odd, odd_expected, odd_compiled_gemv) = run_allocating_projection(&device, 1, 33);
    assert!(
        !odd_compiled_gemv,
        "odd-N safety fallback unexpectedly compiled GEMV"
    );
    assert_close(&odd, &odd_expected, "allocating odd-N MM fallback");

    let (prefill, prefill_expected, prefill_compiled_gemv) =
        run_allocating_projection(&device, 2, 32);
    assert!(
        !prefill_compiled_gemv,
        "M=2 prefill unexpectedly compiled GEMV"
    );
    assert_close(&prefill, &prefill_expected, "allocating M=2 MM");

    let input = input_values(1);
    let weights = weight_bits(48);
    let input_buffer = upload_f32(&input, &device).expect("upload input");
    let weight_buffer = upload_bf16_weight(&device, 48, &weights);
    let mut output = device
        .alloc_buffer(48 * 4, DType::F32, vec![1, 48])
        .expect("allocate output");
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().expect("command encoder");
    apply_proj_into(
        &mut encoder,
        &mut registry,
        &device,
        &input_buffer,
        &weight_buffer,
        &mut output,
        1,
        K,
        48,
    )
    .expect("apply BF16 projection into output");
    encoder
        .commit_and_wait_labeled("qwen35.delta.bf16_projection.into")
        .expect("complete BF16 projection into output");
    assert!(
        registry
            .pipeline_identity("hf2q_dense_gemv_bf16_f32_4")
            .is_ok(),
        "M=1 even-N arena helper did not compile GEMV"
    );
    assert_close(
        &download_f32(&output).expect("download into output"),
        &cpu_projection(&input, &weights, 1, 48),
        "arena M=1 GEMV",
    );

    let parity_weights = weight_bits(32);
    let parity_weight = upload_bf16_weight(&device, 32, &parity_weights);
    let mut gemv_output = device
        .alloc_buffer(32 * 4, DType::F32, vec![1, 32])
        .expect("allocate GEMV parity output");
    let mut mm_output = device
        .alloc_buffer(32 * 4, DType::F32, vec![1, 32])
        .expect("allocate MM parity output");
    let params = DenseMmBf16F32Params {
        m: 1,
        n: 32,
        k: K,
        src0_batch: 1,
        src1_batch: 1,
    };
    let mut parity_registry = KernelRegistry::new();
    let mut gemv_encoder = device.command_encoder().expect("GEMV parity encoder");
    dense_gemv_bf16_f32(
        &mut gemv_encoder,
        &mut parity_registry,
        &device,
        &parity_weight,
        &input_buffer,
        &mut gemv_output,
        &params,
    )
    .expect("encode GEMV parity");
    gemv_encoder
        .commit_and_wait_labeled("qwen35.delta.bf16_projection.parity_gemv")
        .expect("complete GEMV parity");
    let mut mm_encoder = device.command_encoder().expect("MM parity encoder");
    dense_matmul_bf16_f32_tensor(
        &mut mm_encoder,
        &mut parity_registry,
        &device,
        &parity_weight,
        &input_buffer,
        &mut mm_output,
        &params,
    )
    .expect("encode MM parity");
    mm_encoder
        .commit_and_wait_labeled("qwen35.delta.bf16_projection.parity_mm")
        .expect("complete MM parity");
    assert_close(
        &download_f32(&gemv_output).expect("download GEMV parity"),
        &download_f32(&mm_output).expect("download MM parity"),
        "native GEMV versus MM",
    );
}

#[test]
fn delta_bf16_projection_rejects_non_f32_activation_or_destination() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let weight_buffer = upload_bf16_weight(&device, 32, &weight_bits(32));
    let mut bf16_input = device
        .alloc_buffer(K as usize * 2, DType::BF16, vec![1, K as usize])
        .expect("allocate BF16 input");
    bf16_input
        .as_mut_slice::<u16>()
        .expect("view BF16 input")
        .fill(bf16::from_f32(1.0).to_bits());

    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().expect("command encoder");
    let error = apply_proj(
        &mut encoder,
        &mut registry,
        &device,
        &bf16_input,
        &weight_buffer,
        1,
        K,
        32,
    )
    .expect_err("BF16 activation must reject");
    assert!(format!("{error:#}").contains("requires F32 input/output"));

    let input = upload_f32(&input_values(1), &device).expect("upload F32 input");
    let mut bf16_output = device
        .alloc_buffer(32 * 2, DType::BF16, vec![1, 32])
        .expect("allocate BF16 output");
    let error = apply_proj_into(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight_buffer,
        &mut bf16_output,
        1,
        K,
        32,
    )
    .expect_err("BF16 destination must reject");
    assert!(format!("{error:#}").contains("requires F32 input/output"));
}
