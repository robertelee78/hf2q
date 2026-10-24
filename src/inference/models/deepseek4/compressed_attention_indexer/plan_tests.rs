//! Pure constructor contracts; no device, model, payload or kernel execution.
use super::*;

fn selected_indexer_config() -> Deepseek4Config {
    // Only these two fields are consumed by this constructor. Values match
    // the cached F1 Flash0731 metadata; no model fixture is opened here.
    let mut cfg = crate::inference::models::deepseek4::residency_tests::tiny_config();
    cfg.index_num_heads = 64;
    cfg.index_head_dim = 128;
    cfg
}

#[test]
fn indexer_plan_f1_262144_has_ten_exact_ordered_requests() {
    let plan = RatioFourIndexerPlan::new(&selected_indexer_config(), 2048, 512, 65536).unwrap();
    let expected = [
        (
            DType::F32,
            vec![1, 2048, 64, 128],
            "index query f32",
            67_108_864,
        ),
        (
            DType::BF16,
            vec![1, 2048, 64, 128],
            "index query bf16",
            33_554_432,
        ),
        (DType::F32, vec![1, 2048, 64], "index weights", 524_288),
        (
            DType::F32,
            vec![1, 2048, 64],
            "scaled index weights",
            524_288,
        ),
        (
            DType::F32,
            vec![1, 2048, 256],
            "index compressor KV",
            2_097_152,
        ),
        (
            DType::F32,
            vec![1, 2048, 256],
            "index compressor score",
            2_097_152,
        ),
        (
            DType::BF16,
            vec![1, 512, 128],
            "index compressor output",
            131_072,
        ),
        (
            DType::BF16,
            vec![1, 512, 1, 128],
            "index compressor rotated output",
            131_072,
        ),
        (
            DType::F32,
            vec![1, 2048, 65536],
            "index score scratch",
            536_870_912,
        ),
        (
            DType::I32,
            vec![1, 2048, 65536],
            "index top-k scratch",
            536_870_912,
        ),
    ];
    assert_eq!(plan.buffers.len(), expected.len());
    for (actual, (dtype, shape, label, bytes)) in plan.buffers.iter().zip(expected) {
        assert_eq!(actual.dtype, dtype);
        assert_eq!(actual.shape, shape);
        assert_eq!(actual.label, label);
        assert_eq!(actual.requested_bytes, bytes);
        assert_eq!(actual.pooled_bucket_bytes, bytes);
    }
    assert_eq!(plan.requested_bytes, 1_179_910_144);
    assert_eq!(plan.pooled_bucket_bytes, 1_179_910_144);
}

#[test]
fn indexer_plan_131072_is_a_distinct_diagnostic_input() {
    let plan = RatioFourIndexerPlan::new(&selected_indexer_config(), 2048, 512, 32768).unwrap();
    assert_eq!(plan.buffers[8].requested_bytes, 268_435_456);
    assert_eq!(plan.buffers[9].requested_bytes, 268_435_456);
    assert_eq!(plan.requested_bytes, 643_039_232);
}

#[test]
fn indexer_plan_zero_completed_groups_keep_original_minimum_storage() {
    let plan = RatioFourIndexerPlan::new(&selected_indexer_config(), 1, 0, 0).unwrap();
    assert_eq!(plan.buffers[6].shape, [1, 1, 128]);
    assert_eq!(plan.buffers[7].shape, [1, 1, 1, 128]);
    assert_eq!(plan.buffers[8].shape, [1, 1, 1]);
    assert_eq!(plan.buffers[9].shape, [1, 1, 1024]);
}

#[test]
fn indexer_plan_native_topk_block_boundary_and_irregular_bucket() {
    for (valid, row_stride) in [(1023, 1024), (1024, 1024), (1025, 2048)] {
        let plan = RatioFourIndexerPlan::new(&selected_indexer_config(), 3, 1, valid).unwrap();
        let topk = &plan.buffers[9];
        assert_eq!(topk.shape, [1, 3, row_stride]);
        assert_eq!(
            topk.requested_bytes,
            deepseek_indexer_topk_scratch_elements(1, 3, valid).unwrap() * 4
        );
        assert_eq!(plan.buffers[0].requested_bytes, 98_304);
        assert_eq!(plan.buffers[0].pooled_bucket_bytes, 131_072);
        assert!(plan.pooled_bucket_bytes > plan.requested_bytes);
    }
}

#[test]
fn indexer_plan_rejects_zero_dimensions_before_topk_division() {
    let mut cfg = selected_indexer_config();
    assert!(RatioFourIndexerPlan::new(&cfg, 0, 0, 0).is_err());
    cfg.index_num_heads = 0;
    assert!(RatioFourIndexerPlan::new(&cfg, 1, 0, 0).is_err());
    cfg.index_num_heads = 64;
    cfg.index_head_dim = 0;
    assert!(RatioFourIndexerPlan::new(&cfg, 1, 0, 0).is_err());
}

#[test]
fn indexer_plan_rejects_native_and_requested_overflow_without_payload_allocation() {
    assert!(RatioFourIndexerPlan::new(&selected_indexer_config(), usize::MAX, 1, 1).is_err());
    assert!(RatioFourIndexerPlan::new(&selected_indexer_config(), 1, usize::MAX, 1).is_err());
    assert!(IndexerBufferPlan::new(DType::F32, vec![usize::MAX, 2], "shape").is_err());
    assert!(IndexerBufferPlan::new(DType::F32, vec![usize::MAX], "bytes").is_err());
    assert!(IndexerBufferPlan::new(DType::U8, vec![usize::MAX], "bucket").is_err());
}

#[test]
fn indexer_plan_aggregate_overflow_is_checked_separately() {
    let large = 1usize << (usize::BITS - 1);
    let a = IndexerBufferPlan::new(DType::U8, vec![large], "a").unwrap();
    let b = IndexerBufferPlan::new(DType::U8, vec![large], "b").unwrap();
    assert!(totals(&[a, b]).is_err());
    // Requested sum fits but independently rounded bucket sum does not.
    let a = IndexerBufferPlan::new(DType::U8, vec![large / 2 + 1], "a").unwrap();
    let b = IndexerBufferPlan::new(DType::U8, vec![large / 2 + 1], "b").unwrap();
    assert!(totals(&[a, b])
        .unwrap_err()
        .to_string()
        .contains("bucket sum"));
}
