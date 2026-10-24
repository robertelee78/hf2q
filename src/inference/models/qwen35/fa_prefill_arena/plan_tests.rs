//! Pure shape fixtures; no device, model, GPU lock or payload allocation.
use super::*;

fn check_ordered_shapes(seq: usize, nh: usize, nkv: usize, d: usize, total: usize) {
    let plan = FaPrefillPlan::new(seq as u32, nh as u32, nkv as u32, d as u32).unwrap();
    let expected = [
        ("q_bf16_seq", vec![seq, nh, d], seq * nh * d * 2),
        ("q_bf16_hm", vec![1, nh, seq, d], seq * nh * d * 2),
        ("k_bf16_seq", vec![seq, nkv, d], seq * nkv * d * 2),
        ("k_bf16_hm", vec![1, nkv, seq, d], seq * nkv * d * 2),
        ("v_bf16_seq", vec![seq, nkv, d], seq * nkv * d * 2),
        ("v_bf16_hm", vec![1, nkv, seq, d], seq * nkv * d * 2),
        ("out_bf16_hm", vec![1, nh, seq, d], seq * nh * d * 2),
    ];
    assert_eq!(plan.buffers.len(), 7);
    for (actual, (label, shape, bytes)) in plan.buffers.iter().zip(expected) {
        assert_eq!(actual.label, label);
        assert_eq!(actual.dtype, DType::BF16);
        assert_eq!(actual.shape, shape);
        assert_eq!(actual.requested_bytes, bytes);
    }
    assert_eq!(plan.requested_bytes, total);
}

#[test]
fn fa_prefill_plan_qwen38_f1_geometry_has_seven_exact_roots() {
    // Cached selected Qwen3.8 metadata: Q24/KV4/D256. Transaction rows
    // are2048, not the262144-token per-slot context.
    check_ordered_shapes(2048, 24, 4, 256, 92_274_688);
}

#[test]
fn fa_prefill_plan_qwen36_f1_geometry_has_seven_exact_roots() {
    // Cached selected Qwen3.6 metadata: Q16/KV2/D256.
    check_ordered_shapes(2048, 16, 2, 256, 58_720_256);
}

#[test]
fn fa_prefill_plan_one_and_irregular_rows_preserve_exact_direct_bytes() {
    check_ordered_shapes(1, 24, 4, 256, 45_056);
    check_ordered_shapes(101, 24, 4, 256, 4_550_656);
    check_ordered_shapes(101, 16, 2, 256, 2_895_872);
}

#[test]
fn fa_prefill_plan_zero_dimensions_precede_gqa_modulo() {
    for dims in [
        (0, 24, 4, 256),
        (1, 0, 4, 256),
        (1, 24, 0, 256),
        (1, 24, 4, 0),
    ] {
        let error = FaPrefillPlan::new(dims.0, dims.1, dims.2, dims.3)
            .err()
            .unwrap();
        assert_eq!(
            error.to_string(),
            format!(
            "FaPrefillArena::new: zero dim seq_capacity={} n_heads={} n_kv_heads={} head_dim={}",
            dims.0, dims.1, dims.2, dims.3)
        );
    }
}

#[test]
fn fa_prefill_plan_preserves_gqa_validation_without_new_width_policy() {
    let error = FaPrefillPlan::new(101, 15, 2, 256).err().unwrap();
    assert_eq!(
        error.to_string(),
        "FaPrefillArena::new: n_heads (15) must be divisible by n_kv_heads (2)"
    );
    check_ordered_shapes(3, 3, 1, 1, 78);
}

#[test]
fn fa_prefill_plan_checks_shape_and_byte_overflow_before_allocation() {
    let error = FaPrefillPlan::new(u32::MAX, u32::MAX, u32::MAX, u32::MAX)
        .err()
        .unwrap();
    assert!(error.to_string().contains("shape overflow"));
    // Element count fits64bit, but BF16 byte multiplication does not.
    let error = FaPrefillPlan::new(u32::MAX, u32::MAX, u32::MAX, 1)
        .err()
        .unwrap();
    if usize::BITS == 64 {
        assert!(error.to_string().contains("byte size overflow"));
    } else {
        assert!(error.to_string().contains("shape overflow"));
    }
}

#[test]
fn fa_prefill_plan_checks_sum_even_when_each_root_fits() {
    // On64bit each of the seven roots fits2^62bytes; the sum overflows.
    let error = FaPrefillPlan::new(1 << 31, 1 << 30, 1 << 30, 1)
        .err()
        .unwrap();
    if usize::BITS == 64 {
        assert!(error.to_string().contains("payload sum overflow"));
    } else {
        assert!(error.to_string().contains("shape overflow"));
    }
}
