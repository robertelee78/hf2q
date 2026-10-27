use super::*;
use crate::inference::models::qwen35::{default_layer_types, Qwen35MoeConfig, Qwen35Variant};

#[path = "c1_anchor_layout_tests.rs"]
mod c1_anchor_layout_tests;

// Scalar-only F1 header fixtures, no private paths/weights/tokenizer data.
// Q38: cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7
// Q36: 815a22cab21649948352c0eea6e9be0d9d15606e2024d86e7bdb71fb9df08220
// block_count includes nextn; ssm.inner_size/state_size gives value heads.
fn header_config(q38: bool) -> Qwen35Config {
    let (serialized_blocks, nextn, inner, kv, hidden, heads) = if q38 {
        (65, 1, 6144, 4, 5120, 24)
    } else {
        (41, 1, 4096, 2, 2048, 16)
    };
    let base = serialized_blocks - nextn;
    Qwen35Config {
        variant: if q38 {
            Qwen35Variant::Dense
        } else {
            Qwen35Variant::Moe
        },
        hidden_size: hidden,
        num_hidden_layers: base,
        num_attention_heads: heads,
        num_key_value_heads: kv,
        head_dim: 256,
        linear_num_key_heads: 16,
        linear_num_value_heads: inner / 128,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        layer_types: default_layer_types(base, 4),
        partial_rotary_factor: 0.25,
        rope_theta: 1e7,
        rotary_dim: 64,
        mrope_section: [11, 11, 10, 0],
        mrope_interleaved: true,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 262144,
        vocab_size: 248320,
        attn_output_gate: true,
        mtp_num_hidden_layers: nextn,
        mtp_use_dedicated_embeddings: false,
        intermediate_size: if q38 { Some(17408) } else { None },
        moe: if q38 {
            None
        } else {
            Some(Qwen35MoeConfig {
                moe_intermediate_size: 512,
                num_experts: 256,
                num_experts_per_tok: 8,
                shared_expert_intermediate_size: 512,
            })
        },
    }
}
#[test]
fn production_cache_plan_f1_header_totals_and_consumed_roots() {
    for (q38, full, linear, total, roots, cap_bytes, cap_roots) in [
        (
            true,
            37077647360usize,
            1255145472usize,
            38332792832usize,
            260,
            2510290944usize,
            96,
        ),
        (
            false,
            11995709440,
            526909440,
            12522618880,
            164,
            1053818880,
            60,
        ),
    ] {
        let cfg = header_config(q38);
        let plan = HybridCacheAllocationPlan::new(&cfg, 262144, 4, true, true).unwrap();
        assert_eq!(
            (
                plan.full_bytes,
                plan.linear_bytes,
                plan.total_bytes,
                plan.root_count
            ),
            (full, linear, total, roots)
        );
        assert_eq!(plan.conv_channels, if q38 { 10240 } else { 8192 });
        let mut consumed = Vec::new();
        let values = plan
            .allocate_with(|r| {
                consumed.push((r.label, r.dtype, r.shape.clone(), r.bytes, r.init));
                Ok(r.bytes)
            })
            .unwrap();
        assert_eq!(values.iter().flatten().sum::<usize>(), total);
        assert_eq!(consumed.len(), roots);
        assert_eq!(consumed[0].0, "alloc conv_state");
        assert_eq!(consumed[1].0, "alloc conv_state_scratch");
        assert_eq!(consumed[2].0, "alloc recurrent");
        assert_eq!(consumed[3].0, "alloc recurrent_scratch");
        assert_eq!(consumed[0].2, vec![if q38 { 10240 } else { 8192 }, 3, 4]);
        assert_eq!(consumed[2].2, vec![128, 128, if q38 { 48 } else { 32 }, 4]);
        assert!(consumed[..4]
            .iter()
            .all(|r| r.1 == DType::F32 && r.4 == Init::Zero));
        let end = &consumed[roots - 4..];
        assert_eq!(
            end.iter().map(|r| r.0).collect::<Vec<_>>(),
            vec![
                "alloc TQ full-attn K packed",
                "alloc TQ full-attn K norms",
                "alloc TQ full-attn V packed",
                "alloc TQ full-attn V norms"
            ]
        );
        assert!(end.iter().all(|r| r.4 == Init::Overwrite));
        assert_eq!(end[0].1, DType::U8);
        assert_eq!(end[1].1, DType::F32);
        let la = if q38 { 48 } else { 30 };
        let capture = CaptureAllocationPlan::new(&cfg, 4, 4, &vec![[None, None]; la]).unwrap();
        assert_eq!(
            (capture.new_bytes, capture.new_roots),
            (cap_bytes, cap_roots)
        );
        assert_eq!(
            (capture.old_plus_new_bytes, capture.upper_refs),
            (cap_bytes, cap_roots)
        );
    }
}
#[test]
fn production_cache_plan_f32_no_mtp_and_single_mtp_semantics() {
    for (q38, total, roots) in [(true, 147284033536usize, 226), (false, 47771549696, 142)] {
        let mut cfg = header_config(q38);
        cfg.mtp_num_hidden_layers = 7;
        let with = HybridCacheAllocationPlan::new(&cfg, 262144, 4, false, true).unwrap();
        assert_eq!((with.total_bytes, with.root_count), (total, roots));
        assert_eq!(with.slots.iter().filter(|s| s.mtp).count(), 1);
        let without = HybridCacheAllocationPlan::new(&cfg, 262144, 4, false, false).unwrap();
        assert_eq!(without.root_count, roots - 2);
        assert_eq!(without.slots.len(), cfg.layer_types.len());
        assert_eq!(
            with.slots.last().unwrap().roots[0].shape,
            vec![4, if q38 { 4 } else { 2 }, 262144, 256]
        );
        cfg.layer_types.clear();
        cfg.mtp_num_hidden_layers = 0;
        let empty = HybridCacheAllocationPlan::new(&cfg, 3, 2, true, true).unwrap();
        assert_eq!((empty.total_bytes, empty.root_count), (0, 0));
        assert!(empty
            .allocate_with(|_| -> Result<()> { panic!("empty schedule allocated") })
            .unwrap()
            .is_empty());
    }
}
#[test]
fn production_cache_plan_tq_byte_codes_norm_floor_and_standalone_traversal() {
    let mut cfg = header_config(true);
    for (dim, norms) in [(1, 1), (255, 1), (256, 1), (511, 1), (512, 2), (768, 3)] {
        cfg.head_dim = dim;
        let (roots, n) = tq_roots(&cfg, 7, 3).unwrap();
        assert_eq!(n, norms);
        let bytes = allocate_roots_with(&roots, |r| Ok(r.bytes)).unwrap();
        assert_eq!(
            bytes,
            vec![
                3 * 4 * 7 * dim as usize,
                3 * 4 * 7 * norms as usize * 4,
                3 * 4 * 7 * dim as usize,
                3 * 4 * 7 * norms as usize * 4
            ]
        );
        assert_eq!(roots[0].shape, vec![3, 4, 7, dim as usize]);
        assert_eq!(roots[1].shape, vec![3, 4, 7, norms as usize]);
    }
}
#[test]
fn production_cache_plan_schedule_order_and_allocator_fault_context() {
    let mut cfg = header_config(true);
    cfg.layer_types = vec![
        Qwen35LayerKind::FullAttention,
        Qwen35LayerKind::LinearAttention,
        Qwen35LayerKind::FullAttention,
    ];
    let plan = HybridCacheAllocationPlan::new(&cfg, 3, 2, true, true).unwrap();
    assert_eq!(
        plan.slots.iter().map(|s| s.slot).collect::<Vec<_>>(),
        vec![
            LayerSlot::Full(0),
            LayerSlot::Linear(0),
            LayerSlot::Full(1),
            LayerSlot::Full(2)
        ]
    );
    let mut calls = Vec::new();
    let error = plan
        .allocate_with(|r| {
            calls.push(r.label);
            if calls.len() == 6 {
                Err(anyhow!("injected"))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert_eq!(
        calls,
        vec![
            "alloc TQ full-attn K packed",
            "alloc TQ full-attn K norms",
            "alloc TQ full-attn V packed",
            "alloc TQ full-attn V norms",
            "alloc conv_state",
            "alloc conv_state_scratch"
        ]
    );
    assert_eq!(error.to_string(), "alloc linear-attn slot (layer 1)");
}
fn old_from(plan: &CaptureAllocationPlan) -> Vec<[Option<ExistingCapture>; 2]> {
    plan.actions
        .chunks_exact(2)
        .map(|pair| {
            std::array::from_fn(|i| {
                pair[i].allocation.as_ref().map(|r| ExistingCapture {
                    elements: r.elements,
                    backing_bytes: r.bytes,
                })
            })
        })
        .collect()
}
#[test]
fn production_cache_plan_capture_replacement_bounds_and_reuse_rules() {
    let cfg = header_config(true);
    let initial = CaptureAllocationPlan::new(&cfg, 4, 4, &vec![[None, None]; 48]).unwrap();
    let old = old_from(&initial);
    let grow = CaptureAllocationPlan::new(&cfg, 4, 5, &old).unwrap();
    assert_eq!(
        (
            grow.old_backing_bytes,
            grow.new_bytes,
            grow.new_roots,
            grow.upper_refs
        ),
        (2510290944, 3137863680, 96, 192)
    );
    assert_eq!(38332792832usize + grow.old_plus_new_bytes, 43980947456);
    let same = CaptureAllocationPlan::new(&cfg, 4, 4, &old).unwrap();
    assert_eq!((same.new_roots, same.new_bytes), (0, 0));
    let shrink = CaptureAllocationPlan::new(&cfg, 4, 3, &old).unwrap();
    assert_eq!((shrink.new_roots, shrink.new_bytes), (96, 1882718208));
    let single = CaptureAllocationPlan::new(&cfg, 1, 4, &[[None, None]]).unwrap();
    let single_old = old_from(&single);
    assert_eq!(
        CaptureAllocationPlan::new(&cfg, 1, 3, &single_old)
            .unwrap()
            .new_roots,
        0
    );
    let mut larger_backing = single_old;
    larger_backing[0][0].as_mut().unwrap().backing_bytes += 4096;
    let kept = CaptureAllocationPlan::new(&cfg, 1, 3, &larger_backing).unwrap();
    assert_eq!(kept.old_backing_bytes, single.new_bytes + 4096);
    assert_eq!(kept.retained_after_bytes, kept.old_backing_bytes);
}
#[test]
fn production_cache_plan_mixed_capture_preserves_kept_roots_and_applies_order() {
    let cfg = header_config(false);
    let fresh = CaptureAllocationPlan::new(&cfg, 4, 4, &[[None, None], [None, None]]).unwrap();
    let mut old = old_from(&fresh);
    old[0][1] = None;
    old[1][0] = None;
    let plan = CaptureAllocationPlan::new(&cfg, 4, 4, &old).unwrap();
    let mut installed = Vec::new();
    plan.apply_with(
        |r| Ok(r.label),
        |i, role, value| installed.push((i, role, value)),
    )
    .unwrap();
    assert_eq!(
        installed,
        vec![
            (0, CaptureRole::Conv, "alloc conv_capture_states"),
            (1, CaptureRole::Recurrent, "alloc capture_states")
        ]
    );
}
#[test]
fn production_cache_plan_capture_activity_zero_and_preparation_failure() {
    let mut active = Some(4);
    assert!(ensure_capture_with(&mut active, 0, || panic!("zero called operation")).is_err());
    assert_eq!(active, Some(4));
    let mut bad = header_config(true);
    bad.linear_num_key_heads = u32::MAX;
    assert!(ensure_capture_with(&mut active, 4, || {
        CaptureAllocationPlan::new(&bad, 4, 4, &[[None, None]])?;
        Ok(())
    })
    .is_err());
    assert_eq!(active, None);
    ensure_capture_with(&mut active, 3, || Ok(())).unwrap();
    assert_eq!(active, Some(3));
}
#[test]
fn production_cache_plan_capture_fault_retains_each_successful_assignment() {
    let cfg = header_config(true);
    let plan = CaptureAllocationPlan::new(&cfg, 4, 4, &[[None, None], [None, None]]).unwrap();
    for fail_at in [1, 3] {
        let mut active = Some(2);
        let mut values = [[Some("old recurrent"), Some("old conv")]; 2];
        let mut calls = 0;
        let error = ensure_capture_with(&mut active, 4, || {
            plan.apply_with(
                |r| {
                    let call = calls;
                    calls += 1;
                    if call == fail_at {
                        Err(anyhow!("injected capture alloc"))
                    } else {
                        Ok(r.label)
                    }
                },
                |i, role, v| values[i][usize::from(role == CaptureRole::Conv)] = Some(v),
            )
        })
        .unwrap_err();
        assert!(error.to_string().contains("injected"));
        assert_eq!(active, None);
        assert_eq!(values[0][0], Some("alloc capture_states"));
        if fail_at == 1 {
            assert_eq!(values[0][1], Some("old conv"));
            assert_eq!(values[1][0], Some("old recurrent"));
        } else {
            assert_eq!(values[0][1], Some("alloc conv_capture_states"));
            assert_eq!(values[1][0], Some("alloc capture_states"));
            assert_eq!(values[1][1], Some("old conv"));
        }
    }
}
#[test]
fn production_cache_plan_zero_conv_descriptor_keeps_partial_failure_behavior() {
    let mut cfg = header_config(true);
    cfg.layer_types = vec![Qwen35LayerKind::LinearAttention];
    cfg.linear_conv_kernel_dim = 1;
    let base = HybridCacheAllocationPlan::new(&cfg, 2, 1, false, false).unwrap();
    assert_eq!(base.slots[0].roots[0].shape, vec![10240, 1, 1]);
    let plan = CaptureAllocationPlan::new(&cfg, 1, 4, &[[None, None]]).unwrap();
    assert_eq!(plan.actions[1].allocation.as_ref().unwrap().bytes, 0);
    let mut active = Some(2);
    let mut assigned = Vec::new();
    let result = ensure_capture_with(&mut active, 4, || {
        plan.apply_with(
            |r| {
                if r.bytes == 0 {
                    Err(anyhow!("{}: zero native allocation", r.label))
                } else {
                    Ok(r.label)
                }
            },
            |_, role, v| assigned.push((role, v)),
        )
    });
    assert!(result
        .unwrap_err()
        .to_string()
        .starts_with("alloc conv_capture_states:"));
    assert_eq!(
        assigned,
        vec![(CaptureRole::Recurrent, "alloc capture_states")]
    );
    assert_eq!(active, None);
}
#[test]
fn production_cache_plan_checked_failures_and_zero_diagnostic_priority() {
    let cfg = header_config(true);
    assert_eq!(
        HybridCacheAllocationPlan::new(&cfg, 0, 0, true, true)
            .unwrap_err()
            .to_string(),
        "HybridKvCache: max_seq_len must be > 0"
    );
    assert_eq!(
        HybridCacheAllocationPlan::new(&cfg, 1, 0, true, true)
            .unwrap_err()
            .to_string(),
        "HybridKvCache: n_seqs must be > 0"
    );
    assert!(product(&[usize::MAX, 2]).is_err());
    assert!(add(usize::MAX, 1).is_err());
    assert!(BufferPlan::new("overflow", DType::F32, vec![usize::MAX / 4 + 1], Init::Zero).is_err());
    let mut bad = cfg.clone();
    bad.linear_num_key_heads = u32::MAX;
    assert!(HybridCacheAllocationPlan::new(&bad, 1, 1, true, true).is_err());
    let old = [[
        Some(ExistingCapture {
            elements: 0,
            backing_bytes: usize::MAX,
        }),
        None,
    ]];
    assert!(CaptureAllocationPlan::new(&cfg, 4, 4, &old).is_err());
}
