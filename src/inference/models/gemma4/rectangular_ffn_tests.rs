use super::*;

fn shape(lanes: usize) -> RectangularPrefillShape {
    RectangularPrefillShape {
        lanes,
        rows_per_lane: 32,
        start_position: 64,
    }
}

fn assert_disjoint_and_complete(spans: impl Iterator<Item = ElementSpan>, expected: usize) {
    let mut ranges = spans
        .map(|span| span.range().expect("valid span"))
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| range.start);
    assert_eq!(ranges.first().map(|range| range.start), Some(0));
    for pair in ranges.windows(2) {
        assert_eq!(
            pair[0].end, pair[1].start,
            "lane spans overlap or leave a gap"
        );
    }
    assert_eq!(ranges.last().map(|range| range.end), Some(expected));
}

#[test]
fn plans_exact_b2_and_b4_m32_body_spans() {
    for lanes in [2, 4] {
        let plan =
            plan_rectangular_ffn(shape(lanes), 4096, 1408, 8).expect("supported Gemma rectangle");
        assert_eq!(plan.lanes.len(), lanes);
        assert!(plan
            .lanes
            .iter()
            .all(|lane| lane.source_rows == 32 && lane.top_k == 8));
        assert!(!plan.lanes[0].scratch_reuse_barrier);
        assert!(plan.lanes[1..]
            .iter()
            .all(|lane| lane.scratch_reuse_barrier));

        assert_disjoint_and_complete(plan.lanes.iter().map(|lane| lane.input), lanes * 32 * 4096);
        assert_disjoint_and_complete(plan.lanes.iter().map(|lane| lane.routing), lanes * 32 * 8);
        assert_disjoint_and_complete(
            plan.lanes.iter().map(|lane| lane.expert_gate_up),
            lanes * 32 * 8 * 2 * 1408,
        );
    }

    let wide_shape = RectangularPrefillShape {
        rows_per_lane: 95,
        ..shape(4)
    };
    let wide = plan_rectangular_ffn(wide_shape, 4096, 1408, 8)
        .expect("full scalar width remains lane-local");
    assert!(wide
        .lanes
        .iter()
        .all(|lane| lane.source_rows == 95 && lane.top_k == 8));
}

#[test]
fn rejects_unproven_body_shapes_and_dimension_overflow() {
    for bad_shape in [
        RectangularPrefillShape {
            lanes: 3,
            ..shape(2)
        },
        RectangularPrefillShape {
            rows_per_lane: 31,
            ..shape(2)
        },
        RectangularPrefillShape {
            rows_per_lane: 1_025,
            ..shape(4)
        },
    ] {
        assert!(plan_rectangular_ffn(bad_shape, 4096, 1408, 8).is_err());
    }
    assert!(plan_rectangular_ffn(shape(2), 0, 1408, 8).is_err());
    assert!(plan_rectangular_ffn(shape(2), 4096, usize::MAX, 8).is_err());
}

#[test]
fn typed_span_views_alias_exact_parent_regions_and_fail_closed() {
    let Some(device) = mlx_native::MlxDevice::new().ok() else {
        eprintln!("skipping Metal-only rectangular FFN alias test");
        return;
    };
    let mut parent = device
        .alloc_buffer(24 * DType::U32.size_of(), DType::U32, vec![24])
        .expect("U32 parent");
    parent
        .as_mut_slice::<u32>()
        .expect("mapped parent")
        .iter_mut()
        .enumerate()
        .for_each(|(index, value)| *value = index as u32);
    let view = checked_span_view(
        &parent,
        ElementSpan {
            offset: 8,
            elements: 8,
        },
        DType::U32,
        "routing IDs",
    )
    .expect("exact alias");
    assert_eq!(view.contents_ptr(), parent.contents_ptr());
    assert_eq!(view.byte_offset(), 8 * DType::U32.size_of() as u64);
    assert_eq!(
        view.as_slice::<u32>().expect("typed alias"),
        &(8..16).collect::<Vec<_>>()
    );
    let nested_parent = parent.slice_view(4 * DType::U32.size_of() as u64, 16);
    let nested = checked_span_view(
        &nested_parent,
        ElementSpan {
            offset: 4,
            elements: 4,
        },
        DType::U32,
        "nested routing IDs",
    )
    .expect("nested exact alias");
    assert_eq!(nested.contents_ptr(), parent.contents_ptr());
    assert_eq!(nested.byte_offset(), 8 * DType::U32.size_of() as u64);
    assert_eq!(
        nested.as_slice::<u32>().expect("nested typed alias"),
        &(8..12).collect::<Vec<_>>()
    );
    assert!(checked_span_view(
        &parent,
        ElementSpan {
            offset: 20,
            elements: 8,
        },
        DType::U32,
        "oversized IDs",
    )
    .is_err());
    assert!(checked_span_view(
        &parent,
        ElementSpan {
            offset: 0,
            elements: 8,
        },
        DType::F32,
        "wrong dtype",
    )
    .is_err());
}

#[test]
fn production_gate_up_is_lane_local_while_expert_down_stays_aggregate() {
    let source = include_str!("../../../serve/forward_prefill_batched.rs");
    let gate_up_start = source
        .find("let rectangular_ffn_plan = rectangular_ffn_plans")
        .expect("production rectangular gate/up selector");
    let swiglu_start = source[gate_up_start..]
        .find("// Batched SwiGLU")
        .map(|offset| gate_up_start + offset)
        .expect("gate/up section end");
    let gate_up = &source[gate_up_start..swiglu_start];
    assert!(gate_up.contains("dispatch_rectangular_native_gate_up"));
    assert!(gate_up.contains("dispatch_rectangular_ggml_gate_up"));

    let helper_source = include_str!("rectangular_ffn.rs");
    let pooled_start = helper_source
        .find("fn dispatch_rectangular_ggml_gate_up")
        .expect("pooled rectangular gate/up helper");
    let tests_start = helper_source[pooled_start..]
        .find("#[cfg(test)]")
        .map(|offset| pooled_start + offset)
        .expect("helper section end");
    let pooled = &helper_source[pooled_start..tests_start];
    assert_eq!(
        pooled.matches("for lane in &plan.lanes").count(),
        1,
        "gate/up must dispatch exactly once per admitted lane"
    );
    assert!(pooled.contains("n_tokens: lane.source_rows"));
    assert!(pooled.contains("session.encoder_mut().memory_barrier()"));

    let down_start = source[swiglu_start..]
        .find("// MoE down experts")
        .map(|offset| swiglu_start + offset)
        .expect("expert-down section");
    let combine_start = source[down_start..]
        .find("// Wave P4.14")
        .map(|offset| down_start + offset)
        .expect("expert-down section end");
    let expert_down = &source[down_start..combine_start];
    assert!(!expert_down.contains("dispatch_rectangular"));
    assert!(expert_down.contains("seq_len as u32"));
    assert!(expert_down.contains("n_tokens: (seq_len * top_k) as u32"));
}
