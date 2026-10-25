//! Pure constructor contracts; no device, model, payload or kernel execution.
use super::*;

fn selected_indexer_config() -> Deepseek4Config {
    // Only these two fields are consumed by this constructor. Values match
    // the cached F1 Flash0731 metadata; no model fixture is opened here.
    let mut cfg = crate::inference::models::deepseek4::residency_tests::tiny_config();
    cfg.index_num_heads = 64;
    cfg.index_head_dim = 128;
    cfg.index_top_k = 512;
    cfg
}

fn plan(
    cfg: &Deepseek4Config,
    rows: usize,
    completed: usize,
    valid: usize,
) -> Result<RatioFourIndexerPlan> {
    let request = SelectionRequest::new(cfg, rows, 0, valid, rows, 640, 128)?;
    RatioFourIndexerPlan::new(cfg, rows, completed, &request)
}
#[test]
fn indexer_plan_f1_262144_has_ten_exact_ordered_requests() {
    let plan = plan(&selected_indexer_config(), 2048, 512, 65536).unwrap();
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
            vec![1, 256, 65536],
            "index score scratch",
            67_108_864,
        ),
        (
            DType::I32,
            vec![1, 256, 65536],
            "index top-k scratch",
            67_108_864,
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
    assert_eq!(plan.requested_bytes, 240_386_048);
    assert_eq!(plan.pooled_bucket_bytes, 240_386_048);
}

#[test]
fn indexer_plan_131072_is_a_distinct_diagnostic_input() {
    let plan = plan(&selected_indexer_config(), 2048, 512, 32768).unwrap();
    assert_eq!(plan.buffers[8].requested_bytes, 33_554_432);
    assert_eq!(plan.buffers[9].requested_bytes, 33_554_432);
    assert_eq!(plan.requested_bytes, 173_277_184);
}

#[test]
fn indexer_plan_zero_completed_groups_keep_original_minimum_storage() {
    let plan = plan(&selected_indexer_config(), 1, 0, 0).unwrap();
    assert_eq!(plan.buffers[6].shape, [1, 1, 128]);
    assert_eq!(plan.buffers[7].shape, [1, 1, 1, 128]);
    assert_eq!(plan.buffers[8].shape, [1, 1, 1]);
    assert_eq!(plan.buffers[9].shape, [1, 1, 1024]);
}

#[test]
fn indexer_plan_native_topk_block_boundary_and_irregular_bucket() {
    for (valid, row_stride) in [(1023, 1024), (1024, 1024), (1025, 2048)] {
        let plan = plan(&selected_indexer_config(), 3, 1, valid).unwrap();
        let topk = &plan.buffers[9];
        assert_eq!(topk.shape, [1, 3, row_stride]);
        assert_eq!(
            topk.requested_bytes,
            mlx_native::ops::deepseek_indexer::deepseek_indexer_topk_scratch_elements(1, 3, valid)
                .unwrap()
                * 4
        );
        assert_eq!(plan.buffers[0].requested_bytes, 98_304);
        assert_eq!(plan.buffers[0].pooled_bucket_bytes, 131_072);
        assert!(plan.pooled_bucket_bytes > plan.requested_bytes);
    }
}

#[test]
fn indexer_plan_rejects_zero_dimensions_before_topk_division() {
    let mut cfg = selected_indexer_config();
    assert!(plan(&cfg, 0, 0, 0).is_err());
    cfg.index_num_heads = 0;
    assert!(plan(&cfg, 1, 0, 0).is_err());
    cfg.index_num_heads = 64;
    cfg.index_head_dim = 0;
    assert!(plan(&cfg, 1, 0, 0).is_err());
}

#[test]
fn indexer_plan_rejects_native_and_requested_overflow_without_payload_allocation() {
    assert!(plan(&selected_indexer_config(), usize::MAX, 1, 1).is_err());
    assert!(plan(&selected_indexer_config(), 1, usize::MAX, 1).is_err());
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

#[test]
fn indexer_plan_tiled_legal_endpoint_and_actual_traversal() {
    let cfg = selected_indexer_config();
    let layout =
        super::super::layout::prepare_layout(4, 2048, 251904, 0, 128, 63488, 512, true).unwrap();
    let request = SelectionRequest::new(
        &cfg,
        2048,
        251904,
        63488,
        layout.compressed_attention_offset,
        layout.storage_stride,
        layout.indexer_output_offset.unwrap(),
    )
    .unwrap();
    assert_eq!(
        (request.params.offset, request.stride, request.column),
        (2176, 640, 128)
    );
    let p = RatioFourIndexerPlan::new(&cfg, 2048, 512, &request).unwrap();
    assert_eq!(
        (p.requested_bytes, p.pooled_bucket_bytes),
        (236191744, 240386048)
    );
    let expected: Vec<_> = p
        .buffers
        .iter()
        .map(|b| (b.dtype, b.shape.clone(), b.label, b.requested_bytes))
        .collect();
    let mut actual = Vec::new();
    let roles = p
        .allocate_with(|b| {
            b.allocate_with(|dtype, shape, label| {
                let bytes = shape.iter().product::<usize>() * dtype.size_of();
                actual.push((dtype, shape, label, bytes));
                Ok(actual.len())
            })
        })
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(roles, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
}
#[test]
fn indexer_plan_tile_and_block_edges_use_native_plan() {
    let cfg = selected_indexer_config();
    for rows in [1, 255, 256, 257, 2048] {
        for keys in [1023, 1024, 1025] {
            let req = SelectionRequest::new(&cfg, rows, 0, keys, rows, 640, 128).unwrap();
            let p = RatioFourIndexerPlan::new(&cfg, rows, 1, &req).unwrap();
            let native = req.native.as_ref().unwrap();
            assert_eq!(p.buffers[8].requested_bytes, native.score_elements * 4);
            assert_eq!(p.buffers[9].requested_bytes, native.topk_elements * 4);
            assert_eq!(p.buffers[8].shape[1], rows.min(256));
        }
    }
}
#[test]
fn indexer_plan_request_rejects_bad_layout_and_geometry() {
    let cfg = selected_indexer_config();
    for (rows, start, valid, bias, stride, column) in [
        (0, 0, 1, 0, 640, 128),
        (1, usize::MAX, 1, 0, 640, 128),
        (1, 0, usize::MAX, 0, 640, 128),
        (1, 0, 1, usize::MAX, 640, 128),
        (1, 0, 1, 0, 511, 0),
        (1, 0, 1, 0, 640, 129),
        (4, 4, 1, 0, 640, 128),
    ] {
        assert!(SelectionRequest::new(&cfg, rows, start, valid, bias, stride, column).is_err());
    }
    let req = SelectionRequest::new(&cfg, 257, 0, 64, 257, 640, 128).unwrap();
    assert!(RatioFourIndexerPlan::new(&cfg, 256, 1, &req).is_err());
}

#[test]
fn indexer_plan_valid_zero_is_not_a_fabricated_native_call() {
    let cfg = selected_indexer_config();
    let req = SelectionRequest::new(&cfg, 257, 0, 0, 257, 640, 128).unwrap();
    assert!(req.native.is_none());
    assert_eq!(req.params.kv_len, 0);
    let p = RatioFourIndexerPlan::new(&cfg, 257, 0, &req).unwrap();
    assert_eq!(p.buffers[8].shape, [1, 257, 1]);
    assert_eq!(p.buffers[9].shape, [1, 257, 1024]);
}
#[test]
fn indexer_plan_real_traversal_stops_at_first_allocator_fault() {
    for fail_at in 0..10 {
        let p = plan(&selected_indexer_config(), 257, 64, 1025).unwrap();
        let mut calls = 0;
        let result = p.allocate_with(|_| {
            let index = calls;
            calls += 1;
            if index == fail_at {
                anyhow::bail!("allocation {index}");
            }
            Ok(index)
        });
        assert!(result.is_err());
        assert_eq!(calls, fail_at + 1);
    }
}
