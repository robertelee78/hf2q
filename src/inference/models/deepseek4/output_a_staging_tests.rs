//! Focused allocation-route controls; only the ignored test creates Metal resources.
use super::*;

#[test]
fn output_a_staging_actual_plan_matches_dispatch_threshold_and_type() {
    let threshold = MM_ROUTING_THRESHOLD as usize;
    for (ty, rows, input) in [
        (GgmlType::Q8_0, threshold, true),
        (GgmlType::Q8_0, threshold + 1, false),
        (GgmlType::Q4_0, threshold + 1, true),
    ] {
        let plan = BatchedGroupedOutputPlan::new(rows, 2, 64, 32, ty).unwrap();
        assert_eq!(plan.needs_input(), input);
        let mut shapes = Vec::new();
        let (i, o) = plan
            .allocate_with(|shape, _| {
                shapes.push(shape.clone());
                Ok(shape)
            })
            .unwrap();
        assert_eq!(i.is_some(), input);
        assert_eq!(o, vec![2, rows, 32]);
        assert_eq!(shapes.len(), 1 + usize::from(input));
        assert_eq!(plan.input_bytes, rows * 2 * 64 * 4);
        assert_eq!(plan.output_bytes, rows * 2 * 32 * 4);
    }
    let p = BatchedGroupedOutputPlan::new(2048, 8, 4096, 1024, GgmlType::Q8_0).unwrap();
    assert_eq!(p.input_bytes, 268435456);
    assert!(!p.needs_input());
    assert_eq!(
        BatchedGroupedOutputPlan::new(512, 8, 4096, 1024, GgmlType::Q8_0)
            .unwrap()
            .input_bytes,
        67108864
    );
}

#[test]
fn output_a_staging_bounds_route_drift_and_missing_fallback_fail_closed() {
    for (r, g, k, n) in [
        (0, 2, 64, 32),
        (33, 0, 64, 32),
        (33, 2, 63, 32),
        (33, 2, 64, 0),
        (usize::MAX, 2, 64, 32),
        (u32::MAX as usize, u32::MAX as usize, 64, 32),
    ] {
        assert!(BatchedGroupedOutputPlan::new(r, g, k, n, GgmlType::Q8_0).is_err());
    }
    let p = BatchedGroupedOutputPlan::new(33, 2, 64, 32, GgmlType::Q8_0).unwrap();
    let q = BatchedGroupedOutputPlan::new(33, 2, 64, 32, GgmlType::Q4_0).unwrap();
    assert!(p.matches(q, true).is_err());
    assert!(q.matches(q, false).is_err());
    assert!(p.matches(p, false).is_ok());
    assert!(q.matches(q, true).is_ok());
    let changed = BatchedGroupedOutputPlan::new(34, 2, 64, 32, GgmlType::Q8_0).unwrap();
    assert!(p.matches(changed, false).is_err());
}

#[test]
fn output_a_staging_allocation_failure_has_no_fallback_or_extra_calls() {
    let p = BatchedGroupedOutputPlan::new(33, 2, 64, 32, GgmlType::Q8_0).unwrap();
    let mut calls = 0;
    let result: Result<(Option<()>, ())> = p.allocate_with(|_, label| {
        calls += 1;
        assert_eq!(label, "batched output-A group-major output");
        anyhow::bail!("injected allocation refusal")
    });
    assert!(result.is_err());
    assert_eq!(calls, 1);
    let q = BatchedGroupedOutputPlan::new(MM_ROUTING_THRESHOLD as usize, 2, 64, 32, GgmlType::Q8_0)
        .unwrap();
    calls = 0;
    let result: Result<(Option<()>, ())> = q.allocate_with(|_, _| {
        calls += 1;
        anyhow::bail!("injected input refusal")
    });
    assert!(result.is_err());
    assert_eq!(calls, 1);
}

#[test]
#[ignore = "requires separately supervised tiny Metal control, no model; <=1 MiB requested test payload"]
fn output_a_staging_actual_q8_old_some_new_none_bits_and_fallback() {
    // Fixed fixture arithmetic is checked BEFORE any device/root creation.
    assert_eq!(MM_ROUTING_THRESHOLD, 8);
    let declared_root_payload =
        (33usize + 8) * (2 * 64 * 4 * 3 + 2 * 32 * 4 * 4) + 2 * (2 * 32 * 64 / 32 * 34);
    assert!(declared_root_payload < 1024 * 1024);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx =
        crate::serve::gpu::GpuContext::new().expect("actual Metal device required; never skip");
    let device = ctx.device().clone();
    let (groups, width, rank) = (2usize, 64usize, 32usize);
    let mut historical_requested = 0usize;
    for rows in [33usize, MM_ROUTING_THRESHOLD as usize] {
        println!("DS_OUTPUT_A_STAGING_BEGIN rows={rows} groups=2 width=64 rank=32");
        let mut input = alloc(
            &device,
            DType::F32,
            vec![rows, groups, width],
            "staging test input",
        )
        .unwrap();
        for (i, v) in input
            .as_logical_mut_slice::<f32>()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            *v = ((i % 31) as f32 - 15.0) * 0.03125;
        }
        // Valid deterministic Q8_0 fixture blocks: scale1 (F16), 32 signed bytes.
        // This is a fixed encoded test fixture, not another quantizer implementation.
        let byte_len = groups * rank * width / 32 * 34;
        let mut weight =
            alloc(&device, DType::U8, vec![byte_len], "staging test Q8 weight").unwrap();
        for (b, block) in weight
            .as_logical_mut_slice::<u8>()
            .unwrap()
            .chunks_exact_mut(34)
            .enumerate()
        {
            block[0] = 0;
            block[1] = 0x3c;
            for (i, v) in block[2..].iter_mut().enumerate() {
                *v = (((b + i) % 9) as i8 - 4) as u8;
            }
        }
        let shape = [groups * rank, width];
        let matrix = RawMatrixRef {
            buffer: &weight,
            ggml_type: GgmlType::Q8_0,
            shape: &shape,
        };
        let mut baseline =
            BatchedGroupedOutputArena::new(&device, rows, groups, width, rank, GgmlType::Q8_0)
                .unwrap();
        // Reproduce old unconditional input storage; both arms use the SAME production dispatch.
        if baseline.input_group_major.is_none() {
            baseline.input_group_major = Some(
                alloc(
                    &device,
                    DType::F32,
                    vec![groups, rows, width],
                    "old unused input",
                )
                .unwrap(),
            );
        }
        let candidate =
            BatchedGroupedOutputArena::new(&device, rows, groups, width, rank, GgmlType::Q8_0)
                .unwrap();
        assert_eq!(
            candidate.input_group_major.is_none(),
            rows > MM_ROUTING_THRESHOLD as usize
        );
        assert!(baseline.input_group_major.is_some());
        let expected = alloc(&device, DType::F32, vec![rows, groups * rank], "old output").unwrap();
        let actual = alloc(&device, DType::F32, vec![rows, groups * rank], "new output").unwrap();
        historical_requested += input.data_byte_len()
            + weight.data_byte_len()
            + baseline.output_group_major.data_byte_len()
            + baseline.input_group_major.as_ref().unwrap().data_byte_len()
            + candidate.output_group_major.data_byte_len()
            + candidate
                .input_group_major
                .as_ref()
                .map_or(0, MlxBuffer::data_byte_len)
            + expected.data_byte_len()
            + actual.data_byte_len();
        assert!(
            historical_requested <= 1024 * 1024,
            "declared test root payload bound"
        );
        for (arena, out) in [(&baseline, &expected), (&candidate, &actual)] {
            let (executor, registry) = ctx.split();
            let mut session = executor.begin().unwrap();
            grouped_output_a_batched(
                &mut session,
                registry,
                &device,
                &input,
                &matrix,
                out,
                arena,
                rows,
                groups,
                rank,
                groups,
                width,
            )
            .unwrap();
            session
                .finish()
                .expect("actual output-A command completion");
        }
        let old = expected.as_logical_slice::<f32>().unwrap();
        let new = actual.as_logical_slice::<f32>().unwrap();
        assert!(old.iter().any(|x| *x != 0.0));
        assert!(old.iter().all(|x| x.is_finite()));
        assert_eq!(
            old.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            new.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        // Route mismatch and missing fallback reject before any graph encoding.
        let (executor, registry) = ctx.split();
        let mut session = executor.begin_recorded().unwrap();
        assert!(grouped_output_a_batched(
            &mut session,
            registry,
            &device,
            &input,
            &matrix,
            &actual,
            &candidate,
            rows + 1,
            groups,
            rank,
            groups,
            width
        )
        .is_err());
        if rows <= MM_ROUTING_THRESHOLD as usize {
            let missing = BatchedGroupedOutputArena {
                plan: candidate.plan,
                input_group_major: None,
                output_group_major: candidate.output_group_major.clone(),
            };
            assert!(grouped_output_a_batched(
                &mut session,
                registry,
                &device,
                &input,
                &matrix,
                &actual,
                &missing,
                rows,
                groups,
                rank,
                groups,
                width
            )
            .is_err());
        }
        // No session submission follows either rejected boundary.
        drop(session);
        println!(
            "DS_OUTPUT_A_STAGING_CASE_COMPLETE rows={rows} exact_bits=true fallback_preserved=true"
        );
    }
    println!("DS_OUTPUT_A_STAGING_COMPLETE cases=2 exact_bits=true requested_payload_le_1mib=true model_loaded=false");
}
