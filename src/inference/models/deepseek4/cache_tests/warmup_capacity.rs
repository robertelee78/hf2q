//! Metadata-only necessary constraints for the proposed fresh 64-row probe.
//! Calls the actual production CachePlan, not a duplicate forward emulator.
//! No device, cache allocation, weights, native dispatch or parity claim.

use super::official_config;
use crate::inference::models::deepseek4::cache::{CacheBufferPlan, CacheError, Deepseek4CachePlan};

// Frozen F1 DeepSeek artifact metadata: official 0731 revision
// 7872f01b1d1fe23eabc4c98b48bffcef5a386062; GGUF SHA-256
// 50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6.
// The existing official_config fixture supplies these dimensions/ratios.
const SMALL: usize = 128;
const LARGE: usize = 131_072;

fn allow_row_capacity_only(small: &CacheBufferPlan, large: &CacheBufferPlan) {
    assert_eq!(small.dtype, large.dtype);
    assert_eq!(&small.shape[1..], &large.shape[1..]);
    let mut normalized = large.clone();
    normalized.shape[0] = small.shape[0];
    normalized.bytes = small.bytes;
    assert_eq!(*small, normalized);
}

fn allow_optional_row_capacity_only(
    small: &Option<CacheBufferPlan>,
    large: &Option<CacheBufferPlan>,
) {
    match (small, large) {
        (Some(small), Some(large)) => allow_row_capacity_only(small, large),
        (None, None) => {}
        _ => panic!("capacity change must not add/remove a required cache role"),
    }
}

#[test]
fn paired_actual_plans_change_only_explicit_physical_capacities() {
    let cfg = official_config();
    let small = Deepseek4CachePlan::for_context(&cfg, SMALL).unwrap();
    let large = Deepseek4CachePlan::for_context(&cfg, LARGE).unwrap();
    assert_eq!(small.layers.len(), 43);
    assert_eq!(large.layers.len(), 43);
    assert_eq!(small.resident_bytes, 18_722_816);
    assert_eq!(large.resident_bytes, 919_617_536);
    for (a, b) in small.layers.iter().zip(&large.layers) {
        assert_eq!(a.window_kv, b.window_kv);
        assert_eq!(a.window_kv.shape, [128, 512]);
        assert_eq!(a.main_kv_state, b.main_kv_state);
        assert_eq!(a.main_score_state, b.main_score_state);
        assert_eq!(a.indexer_kv_state, b.indexer_kv_state);
        assert_eq!(a.indexer_score_state, b.indexer_score_state);
        allow_row_capacity_only(&a.attention_kv, &b.attention_kv);
        allow_optional_row_capacity_only(&a.compressed_kv, &b.compressed_kv);
        allow_optional_row_capacity_only(&a.indexer_kv, &b.indexer_kv);
        let mut normalized = b.clone();
        normalized.attention_kv = a.attention_kv.clone();
        normalized.compressed_kv = a.compressed_kv.clone();
        normalized.indexer_kv = a.indexer_kv.clone();
        normalized.resident_bytes = a.resident_bytes;
        assert_eq!(*a, normalized, "unlisted layer-plan change");
        match a.compress_ratio {
            0 => {
                assert!(a.compressed_kv.is_none());
                assert!(a.indexer_kv.is_none());
                assert_eq!(a.attention_kv.shape, [128, 512]);
            }
            4 => {
                assert_eq!(a.compressed_kv.as_ref().unwrap().shape, [32, 512]);
                assert_eq!(b.compressed_kv.as_ref().unwrap().shape, [32_768, 512]);
                assert_eq!(a.indexer_kv.as_ref().unwrap().shape, [32, 128]);
                assert_eq!(b.indexer_kv.as_ref().unwrap().shape, [32_768, 128]);
            }
            128 => {
                assert_eq!(a.compressed_kv.as_ref().unwrap().shape, [1, 512]);
                assert_eq!(b.compressed_kv.as_ref().unwrap().shape, [1024, 512]);
                assert!(a.indexer_kv.is_none());
            }
            ratio => panic!("unexpected frozen ratio {ratio}"),
        }
    }
    let mut normalized = large;
    normalized.context_length = small.context_length;
    normalized.resident_bytes = small.resident_bytes;
    normalized.layers = small.layers.clone(); // all layer changes checked above
    assert_eq!(small, normalized, "unlisted top-level plan change");
    // Do not shrink model/YaRN/served-context configuration with the cache.
    assert_eq!(cfg.max_position_embeddings, 1_048_576);
    assert_eq!(cfg.original_context_length, 65_536);
}

#[test]
fn below_128_has_actual_missing_storage_and_window_constraints() {
    for capacity in [64, 126, 127] {
        let plan = Deepseek4CachePlan::for_context(&official_config(), capacity).unwrap();
        // Constructor acceptance alone is not forward-path compatibility.
        // compressed_attention.rs requires window==cfg.sliding_window and
        // present compressed storage even for zero completed ratio128 groups.
        assert_eq!(plan.layers[2].window_kv.shape, [capacity, 512]);
        assert_ne!(plan.layers[2].window_kv.shape[0], 128);
        assert!(plan.layers[3].compressed_kv.is_none());
    }
}

fn assert_bf16_write_range(buffer: &CacheBufferPlan, start: usize, count: usize, width: usize) {
    assert_eq!(buffer.dtype, mlx_native::DType::BF16);
    assert_eq!(buffer.shape[1], width);
    let end = start.checked_add(count).unwrap();
    assert!(end <= buffer.shape[0]);
    let end_bytes = end.checked_mul(width).unwrap().checked_mul(2).unwrap();
    assert!(end_bytes as u64 <= buffer.bytes);
}

#[test]
fn actual_production_span_has_absolute_64_row_and_16_or_zero_group_bounds() {
    let cfg = official_config();
    let small = Deepseek4CachePlan::for_context(&cfg, SMALL).unwrap();
    let large = Deepseek4CachePlan::for_context(&cfg, LARGE).unwrap();
    let a = small.plan_prefill_at(0, 64, 0).unwrap();
    let b = large.plan_prefill_at(0, 64, 0).unwrap();
    // These are actual production write/visibility descriptors. No semantic
    // difference is whitelisted; only physical plan capacities differ above.
    assert_eq!(a, b);
    assert_eq!(a.start_position, 0);
    assert_eq!(a.token_count, 64);
    assert_eq!(a.start_position + a.token_count, 64);
    let shifted = small.plan_prefill_at(1, 64, 0).unwrap();
    assert_ne!(
        a, shifted,
        "planner must not silently replace a nonzero position with fresh zero"
    );
    assert_eq!(shifted.start_position, 1);
    assert_eq!(shifted.layers[2].window_write_start, 1);
    assert_eq!(shifted.layers[2].window_valid_after, 65);
    for (index, span) in a.layers.iter().enumerate() {
        assert_eq!(span.layer_index, index);
        assert_eq!(span.window_source_start, 0);
        assert_eq!(span.window_write_start, 0);
        assert_eq!(span.window_write_count, 64);
        assert_eq!(span.window_valid_after, 64);
        assert_eq!(span.compressed_write_start, 0);
        assert_eq!(span.indexer_write_start, 0);
        let groups = if cfg.compress_ratios[index] == 4 {
            16
        } else {
            0
        };
        assert_eq!(span.compressed_count, groups);
        assert_eq!(span.compressed_valid_after, groups);
        assert_eq!(span.indexer_count, groups);
        assert_eq!(span.indexer_valid_after, groups);
        for plan in [&small, &large] {
            let layer = &plan.layers[index];
            assert_bf16_write_range(&layer.window_kv, 0, 64, 512);
            if let Some(compressed) = &layer.compressed_kv {
                // ratio128 storage exists, but no completed output row may
                // be included in this transaction's copy/visibility range.
                assert_bf16_write_range(compressed, 0, span.compressed_count, 512);
                let base = layer.window_kv.bytes;
                let end = base + span.compressed_count as u64 * 512 * 2;
                assert!(end <= layer.attention_kv.bytes);
            }
            if let Some(indexer) = &layer.indexer_kv {
                assert_bf16_write_range(indexer, 0, span.indexer_count, 128);
            }
        }
    }
    // Scope: CacheSpan does not expose every emitted attention/compressor
    // operation or its selected indices. This is not native encode acceptance,
    // no-uninitialized-read proof, numerical parity, or actual cursor commit.
}

#[test]
fn pure_production_planner_keeps_bound_errors_and_rejects_graft_overcapacity() {
    let cfg = official_config();
    let plan = Deepseek4CachePlan::for_context(&cfg, SMALL).unwrap();
    assert!(matches!(
        plan.plan_prefill_at(0, 0, 0),
        Err(CacheError::EmptyPrefill)
    ));
    assert!(matches!(
        plan.plan_prefill_at(65, 64, 0),
        Err(CacheError::ContextBound {
            requested: 129,
            maximum: 128
        })
    ));
    assert!(matches!(
        plan.plan_prefill_at(usize::MAX, 1, 0),
        Err(CacheError::ContextBound {
            requested: usize::MAX,
            maximum: 128
        })
    ));
    assert!(matches!(
        plan.plan_prefill_at(0, 64, 2),
        Err(CacheError::GraftCapacity {
            layer: 3,
            n_slots: 2,
            capacity: 1
        })
    ));
    // These plans must still reject invalid production configuration, not
    // acquire a warmup-only exception to the architecture contract.
    let mut invalid = cfg.clone();
    invalid.compress_ratios[2] = 3;
    assert!(matches!(
        Deepseek4CachePlan::for_context(&invalid, SMALL),
        Err(CacheError::CompressionRatio { layer: 2, ratio: 3 })
    ));
    invalid = cfg.clone();
    invalid.sliding_window = 64;
    assert!(matches!(
        Deepseek4CachePlan::for_context(&invalid, SMALL),
        Err(CacheError::SlidingWindow { actual: 64 })
    ));
    assert!(Deepseek4CachePlan::for_context(&cfg, 0).is_err());
    assert!(
        Deepseek4CachePlan::for_context(&cfg, cfg.max_position_embeddings as usize + 1).is_err()
    );
}
