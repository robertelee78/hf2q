use super::*;

#[test]
fn post_weight_capacity_binds_reserves_before_cache_allocation() -> Result<()> {
    let Some(device) = MlxDevice::new().ok() else {
        return Ok(());
    };
    let temp = tempfile::tempdir()?;
    let mut reserved_limits = upload_limits();
    reserved_limits.host_reserve_bytes = 4_096;
    reserved_limits.metal_reserve_bytes = 8_192;

    let host_output = temp.path().join("host-capacity.bin");
    let cache_allocations = Cell::new(0usize);
    let rejected = prepare_run_inputs_with_for_test(
        work()?,
        &host_output,
        preparation_policy(),
        |topology, limits| {
            prepare_with_capacity_for_test(
                topology,
                &device,
                reserved_limits,
                limits,
                capacity(),
                |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
            )
        },
        |teacher, _| {
            let accounted = teacher
                .receipt
                .runtime
                .accounted_runtime_payload_bytes
                .checked_add(teacher.receipt.runtime.unmeasured_runtime_reserve_bytes)
                .unwrap();
            let mut insufficient = capacity();
            insufficient.host_available_bytes = accounted
                .checked_add(teacher.upload_limits.host_reserve_bytes)
                .unwrap()
                - 1;
            let recheck = runtime_capacity_recheck(teacher, insufficient)?;
            cache_allocations.set(cache_allocations.get() + 1);
            let cache = prepare_qwen35_base_text_cache(
                &teacher.config,
                &teacher.device,
                teacher.receipt.runtime.max_sequence_tokens,
            )?;
            Ok((cache, recheck))
        },
    );
    assert!(rejected.is_err());
    assert_eq!(cache_allocations.get(), 0);
    assert_eq!(private_entry_count(temp.path()), 0);

    let metal_output = temp.path().join("metal-capacity.bin");
    let rejected = prepare_run_inputs_with_for_test(
        work()?,
        &metal_output,
        preparation_policy(),
        |topology, limits| {
            prepare_with_capacity_for_test(
                topology,
                &device,
                reserved_limits,
                limits,
                capacity(),
                |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
            )
        },
        |teacher, _| {
            let accounted = teacher
                .receipt
                .runtime
                .accounted_runtime_payload_bytes
                .checked_add(teacher.receipt.runtime.unmeasured_runtime_reserve_bytes)
                .unwrap();
            let metal_required = accounted
                .checked_add(teacher.upload_limits.metal_reserve_bytes)
                .unwrap();
            let mut insufficient = capacity();
            insufficient.metal_current_allocated_bytes = 1_024;
            insufficient.metal_recommended_working_set_bytes = 1_024 + metal_required - 1;
            let recheck = runtime_capacity_recheck(teacher, insufficient)?;
            cache_allocations.set(cache_allocations.get() + 1);
            let cache = prepare_qwen35_base_text_cache(
                &teacher.config,
                &teacher.device,
                teacher.receipt.runtime.max_sequence_tokens,
            )?;
            Ok((cache, recheck))
        },
    );
    assert!(rejected.is_err());
    assert_eq!(cache_allocations.get(), 0);
    assert_eq!(private_entry_count(temp.path()), 0);

    let exact_output = temp.path().join("exact-capacity.bin");
    let accepted = prepare_run_inputs_with_for_test(
        work()?,
        &exact_output,
        preparation_policy(),
        |topology, limits| {
            prepare_with_capacity_for_test(
                topology,
                &device,
                reserved_limits,
                limits,
                capacity(),
                |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
            )
        },
        |teacher, max_sequence_tokens| {
            let accounted = teacher
                .receipt
                .runtime
                .accounted_runtime_payload_bytes
                .checked_add(teacher.receipt.runtime.unmeasured_runtime_reserve_bytes)
                .unwrap();
            let mut exact = capacity();
            exact.host_available_bytes = accounted + teacher.upload_limits.host_reserve_bytes;
            exact.metal_current_allocated_bytes = 1_024;
            exact.metal_recommended_working_set_bytes =
                1_024 + accounted + teacher.upload_limits.metal_reserve_bytes;
            let recheck = runtime_capacity_recheck(teacher, exact)?;
            cache_allocations.set(cache_allocations.get() + 1);
            let cache = prepare_qwen35_base_text_cache(
                &teacher.config,
                &teacher.device,
                max_sequence_tokens,
            )?;
            Ok((cache, recheck))
        },
    )?;
    assert_eq!(cache_allocations.get(), 1);
    assert_eq!(
        accepted
            .receipt_for_test()
            .runtime_capacity_recheck
            .host_required_bytes,
        accepted
            .receipt_for_test()
            .runtime_capacity_recheck
            .capacity
            .host_available_bytes
    );
    assert_eq!(
        accepted
            .receipt_for_test()
            .runtime_capacity_recheck
            .metal_required_bytes,
        accepted
            .receipt_for_test()
            .runtime_capacity_recheck
            .metal_available_bytes
    );
    drop(accepted);
    assert_eq!(private_entry_count(temp.path()), 0);
    Ok(())
}
