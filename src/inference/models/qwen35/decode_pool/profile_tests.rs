use super::*;

#[test]
fn qwen_pool_policy_is_exact_finite_and_checked() {
    let q = quota();
    assert_eq!(
        (q.bytes, q.roots, q.attempts),
        (25_769_803_776, 16_384, 32_768)
    );
    let buckets = buckets().unwrap();
    assert_eq!(buckets.len(), 35);
    assert_eq!(buckets[0].bytes, 1);
    assert_eq!(buckets[34].bytes, 17_179_869_184);
    assert_eq!(
        buckets.iter().map(|b| b.quota.roots).sum::<usize>(),
        368_638
    );
    for (i, b) in buckets.iter().enumerate() {
        assert_eq!(b.bytes, 1usize << i);
        assert_eq!(b.quota.roots, ROOTS.min(BYTES / b.bytes));
        assert_eq!(b.quota.bytes, b.bytes * b.quota.roots);
        assert!(b.quota.bytes <= BYTES && b.quota.roots <= ROOTS);
        assert_eq!(b.quota.attempts, ATTEMPTS);
    }
}
#[test]
fn qwen_pool_legacy_first_prevents_installation_without_refund() {
    let state = AtomicU8::new(FRESH);
    mark_legacy(&state).unwrap();
    mark_legacy(&state).unwrap();
    assert!(start_install(&state).is_err());
    assert_eq!(state.load(Ordering::Acquire), LEGACY);
}
#[test]
fn qwen_pool_install_first_blocks_legacy_and_second_install() {
    let state = AtomicU8::new(FRESH);
    start_install(&state).unwrap();
    assert!(mark_legacy(&state).is_err());
    assert!(start_install(&state).is_err());
    assert_eq!(state.load(Ordering::Acquire), INSTALLING);
}
#[test]
fn qwen_pool_sticky_failure_survives_stringification_and_refuses_replay() {
    let state = AtomicU8::new(SELECTED);
    let result: Result<()> = record_result(&state, Err(error("injected native quota refusal")));
    let stringified = format!("helper context: {}", result.unwrap_err());
    assert!(stringified.contains("quota refusal"));
    assert!(require_health(&state, true).is_err());
    assert!(start_install(&state).is_err());
    assert!(mark_legacy(&state).is_err());
    // A later successful helper result cannot heal the failed generation.
    record_result(&state, Ok(())).unwrap();
    assert!(require_health(&state, true).is_err());
}
#[test]
fn qwen_pool_install_publish_cannot_heal_a_raced_failure() {
    let state = AtomicU8::new(FRESH);
    start_install(&state).unwrap();
    let failure: Result<()> = record_result(
        &state,
        Err(error("unregistered allocation during installation")),
    );
    assert!(failure.is_err());
    assert!(publish_install(&state).is_err());
    assert_eq!(state.load(Ordering::Acquire), UNHEALTHY);
    assert!(require_health(&state, true).is_err());
    let good = AtomicU8::new(FRESH);
    start_install(&good).unwrap();
    publish_install(&good).unwrap();
    require_health(&good, true).unwrap();
}
#[test]
fn qwen_pool_unregistered_worker_cannot_use_healthy_profile() {
    let state = AtomicU8::new(SELECTED);
    assert!(require_health(&state, false).is_err());
    require_health(&state, true).unwrap();
}
#[test]
fn qwen_pool_device_creation_refuses_any_existing_binding() {
    creation_permitted(false, false).unwrap();
    for (pool, device) in [(true, false), (false, true), (true, true)] {
        assert!(creation_permitted(pool, device).is_err());
    }
}
#[test]
fn qwen_pool_eligibility_refuses_other_modes_families_and_slot_counts() {
    for slots in [1, 2, 3, 4] {
        eligibility(true, Some(slots), true).unwrap();
    }
    for (qwen, slots, native) in [
        (false, Some(4), true),
        (true, None, true),
        (true, Some(0), true),
        (true, Some(5), true),
        (true, Some(4), false),
    ] {
        assert!(eligibility(qwen, slots, native).is_err());
    }
}

#[test]
fn qwen_pool_legacy_install_race_has_exactly_one_winner() {
    use std::sync::{Arc, Barrier};
    let mode = Arc::new(AtomicU8::new(FRESH));
    let gate = Arc::new(Barrier::new(2));
    let a_mode = mode.clone();
    let a_gate = gate.clone();
    let a = std::thread::spawn(move || {
        a_gate.wait();
        mark_legacy(&a_mode).is_ok()
    });
    let b_mode = mode.clone();
    let b = std::thread::spawn(move || {
        gate.wait();
        start_install(&b_mode).is_ok()
    });
    let first = a.join().unwrap();
    let second = b.join().unwrap();
    assert_ne!(first, second);
    assert!(matches!(mode.load(Ordering::Acquire), LEGACY | INSTALLING));
}

/// Public Native55 mechanism only, tiny allocation/readback, fresh test process.
/// This does not install the 24GiB policy or run a Qwen model/graph.
#[test]
#[ignore = "requires reviewed fresh-process tiny Metal packet; root execution only"]
fn qwen_pool_tiny_native_quota_reuse_and_queue_identity() {
    let device = MlxDevice::new().expect("Metal required; no skip");
    let authority = LegacyPoolGrowthAuthority::install(LegacyGrowthLimits {
        quota: GrowthQuota {
            bytes: 192,
            roots: 2,
            attempts: 4,
        },
        max_pools: 1,
        max_buckets_per_pool: 2,
        max_rank: 2,
        max_batch: 1,
    })
    .unwrap();
    let mut pool = authority
        .bind(
            &device,
            KEY,
            GrowthQuota {
                bytes: 192,
                roots: 2,
                attempts: 4,
            },
            &[
                GrowthBucket {
                    bytes: 64,
                    quota: GrowthQuota {
                        bytes: 64,
                        roots: 1,
                        attempts: 2,
                    },
                },
                GrowthBucket {
                    bytes: 128,
                    quota: GrowthQuota {
                        bytes: 128,
                        roots: 1,
                        attempts: 2,
                    },
                },
            ],
        )
        .unwrap();
    let mut a = pool.alloc(&device, 64, DType::U8, vec![64]).unwrap();
    let identity = a.metal_buffer().contents();
    assert!(a.as_slice::<u8>().unwrap().iter().all(|b| *b == 0));
    a.as_mut_slice::<u8>().unwrap().fill(0x5a);
    drop(a);
    // SAFETY: CPU only, issued owner dropped, no encoded/raw future access.
    unsafe {
        pool.reset(&device).unwrap();
    }
    let before = pool.snapshot().unwrap();
    let other = MlxDevice::new().unwrap();
    assert!(pool.alloc(&other, 64, DType::U8, vec![64]).is_err());
    assert_eq!(pool.snapshot().unwrap(), before);
    let a = pool
        .alloc(&device.clone(), 64, DType::U8, vec![64])
        .unwrap();
    assert_eq!(a.metal_buffer().contents(), identity);
    assert!(a.as_slice::<u8>().unwrap().iter().all(|b| *b == 0x5a));
    let b = pool.alloc(&device, 128, DType::U8, vec![128]).unwrap();
    assert!(b.as_slice::<u8>().unwrap().iter().all(|b| *b == 0));
    let charged = pool.snapshot().unwrap();
    assert_eq!((charged.process.bytes, charged.process.roots), (192, 2));
    assert!(pool.alloc(&device, 64, DType::U8, vec![64]).is_err());
    assert_eq!(pool.snapshot().unwrap(), charged);
    drop(a);
    drop(b);
    // SAFETY: no issued owners or GPU work remain.
    unsafe {
        pool.reset(&device).unwrap();
    }
    assert_eq!(pool.snapshot().unwrap(), charged);
    println!("QWEN_POOL_NATIVE_MECHANISM_COMPLETE requested_payload=192 gpu_submissions=0 model_loaded=false");
}

#[test]
#[ignore = "fresh process with exact profile+ack; tiny actual HF facade, no model/GPU"]
fn qwen_pool_tiny_actual_hf_route_sticky_failure() {
    assert!(
        intent().unwrap(),
        "launch with reviewed profile+ack; no skip"
    );
    // Inject only policy sizes at the same installation boundary used by
    // prepare. Production passes fixed v1; no runtime size override exists.
    install_policy(Policy {
        quota: GrowthQuota {
            bytes: 192,
            roots: 2,
            attempts: 4,
        },
        buckets: vec![
            GrowthBucket {
                bytes: 64,
                quota: GrowthQuota {
                    bytes: 64,
                    roots: 1,
                    attempts: 2,
                },
            },
            GrowthBucket {
                bytes: 128,
                quota: GrowthQuota {
                    bytes: 128,
                    roots: 1,
                    attempts: 2,
                },
            },
        ],
    })
    .unwrap();
    register_worker(123).unwrap();
    let device = MlxDevice::new().expect("Metal required; no skip");
    bind(123, &device).unwrap();
    let mut a = super::super::pooled_alloc_buffer(&device, 64, DType::U8, vec![64]).unwrap();
    let identity = a.metal_buffer().contents();
    assert!(a.as_slice::<u8>().unwrap().iter().all(|b| *b == 0));
    a.as_mut_slice::<u8>().unwrap().fill(0xa5);
    drop(a);
    super::super::checked_reset_decode_pool().unwrap();
    let a = super::super::pooled_alloc_buffer(&device, 64, DType::U8, vec![64]).unwrap();
    assert_eq!(a.metal_buffer().contents(), identity);
    assert!(a.as_slice::<u8>().unwrap().iter().all(|b| *b == 0xa5));
    let b = super::super::pooled_alloc_buffer(&device, 128, DType::U8, vec![128]).unwrap();
    let charged = LegacyPoolGrowthAuthority::get()
        .unwrap()
        .snapshot(KEY)
        .unwrap();
    assert_eq!(
        (
            charged.process.bytes,
            charged.process.roots,
            charged.process.attempts
        ),
        (192, 2, 2)
    );
    let error = super::super::pooled_alloc_buffer(&device, 64, DType::U8, vec![64])
        .err()
        .expect("quota refusal required");
    let stringified = anyhow::anyhow!("helper stringified failure: {error}");
    assert!(stringified.to_string().contains("quota"));
    assert!(unhealthy());
    assert!(permit_replay().is_err());
    assert!(super::super::checked_reset_decode_pool().is_err());
    assert!(super::super::pooled_alloc_buffer(&device, 64, DType::U8, vec![64]).is_err());
    assert_eq!(
        LegacyPoolGrowthAuthority::get()
            .unwrap()
            .snapshot(KEY)
            .unwrap()
            .process,
        charged.process
    );
    drop(a);
    drop(b);
    println!("QWEN_POOL_HF_ROUTE_COMPLETE requested_payload=192 sticky_failure=true replay_refused=true gpu_submissions=0 model_loaded=false");
}

fn run_drained_windows(bounded: bool) {
    use mlx_native::ops::elementwise::{elementwise_add, elementwise_mul};
    assert_eq!(
        intent().unwrap(),
        bounded,
        "exact arm environment required; no fallback"
    );
    if bounded {
        install_policy(Policy {
            quota: GrowthQuota {
                bytes: 64,
                roots: 1,
                attempts: 1,
            },
            buckets: vec![GrowthBucket {
                bytes: 64,
                quota: GrowthQuota {
                    bytes: 64,
                    roots: 1,
                    attempts: 1,
                },
            }],
        })
        .unwrap();
    }
    register_worker(456).unwrap();
    let device = MlxDevice::new().expect("Metal required; no skip");
    bind(456, &device).unwrap();
    let mut registry = mlx_native::KernelRegistry::new();
    let mut input = device.alloc_buffer(64, DType::F32, vec![16]).unwrap();
    let mut cpu_input = device.alloc_buffer(64, DType::F32, vec![16]).unwrap();
    let mut parent = device.alloc_buffer(64, DType::F32, vec![4, 4]).unwrap();
    let mut final_out = device.alloc_buffer(64, DType::F32, vec![16]).unwrap();
    for (i, x) in input.as_mut_slice::<f32>().unwrap().iter_mut().enumerate() {
        *x = i as f32 - 8.0;
    }
    cpu_input.as_mut_slice::<f32>().unwrap().fill(2.0);
    parent.as_mut_slice::<f32>().unwrap().fill(-999.0);
    final_out.as_mut_slice::<f32>().unwrap().fill(-777.0);
    let scratch = super::super::pooled_alloc_buffer(&device, 64, DType::F32, vec![16]).unwrap();
    let identity = scratch.metal_buffer().contents();
    let mut session = device
        .encoder_session()
        .unwrap()
        .expect("session mode required; no fallback");
    // All fallible encode work for each window precedes its only submission.
    elementwise_add(
        session.encoder(),
        &mut registry,
        device.metal_device(),
        &input,
        &cpu_input,
        &scratch,
        16,
        DType::F32,
    )
    .unwrap();
    session.encoder().memory_barrier();
    elementwise_mul(
        session.encoder(),
        &mut registry,
        device.metal_device(),
        &scratch,
        &cpu_input,
        &parent,
        16,
        DType::F32,
    )
    .unwrap();
    session.commit_and_wait().unwrap();
    for (i, x) in parent.as_slice::<f32>().unwrap().iter().enumerate() {
        assert_eq!(*x, (i as f32 - 6.0) * 2.0);
    }
    let returned_row = parent.slice_view(48, 4);
    drop(scratch);
    // SAFETY at actual checked reset: no future use of issued scratch; first
    // window drained. parent/returned_row and cpu_input have future uses but
    // are direct roots, NOT issued by this pool and cannot be recycled here.
    super::super::checked_reset_for_prefill_chunk().unwrap();
    let scratch = super::super::pooled_alloc_buffer(&device, 64, DType::F32, vec![16]).unwrap();
    assert_eq!(scratch.metal_buffer().contents(), identity);
    session.reset_for_next_stage().unwrap();
    elementwise_add(
        session.encoder(),
        &mut registry,
        device.metal_device(),
        &parent,
        &cpu_input,
        &scratch,
        16,
        DType::F32,
    )
    .unwrap();
    session.encoder().memory_barrier();
    elementwise_mul(
        session.encoder(),
        &mut registry,
        device.metal_device(),
        &scratch,
        &cpu_input,
        &final_out,
        16,
        DType::F32,
    )
    .unwrap();
    session.commit_and_wait().unwrap();
    for (i, x) in final_out.as_slice::<f32>().unwrap().iter().enumerate() {
        assert_eq!(*x, ((i as f32 - 6.0) * 2.0 + 2.0) * 2.0);
    }
    assert_eq!(
        returned_row.metal_buffer().contents(),
        parent.metal_buffer().contents()
    );
    assert_eq!(returned_row.data_byte_len(), 16);
    assert_eq!(returned_row.byte_offset(), 48);
    // Verify parent bytes (native CPU slice offset semantics are not assumed).
    for (i, x) in parent.as_slice::<f32>().unwrap().iter().enumerate() {
        assert_eq!(*x, (i as f32 - 6.0) * 2.0);
    }
    drop(scratch);
    super::super::checked_reset_decode_pool().unwrap();
    assert!(!unhealthy());
    if bounded {
        let count = LegacyPoolGrowthAuthority::get()
            .unwrap()
            .snapshot(KEY)
            .unwrap()
            .process;
        assert_eq!((count.bytes, count.roots, count.attempts), (64, 1, 1));
    } else {
        assert!(LegacyPoolGrowthAuthority::get().is_none());
    }
    println!("QWEN_POOL_DRAINED_WINDOWS_COMPLETE arm={} roots=5 requested_payload=320 windows=2 dispatches=4 direct_parent_preserved=true gpu_submissions=2 model_loaded=false", if bounded { "bounded" } else { "legacy" });
}
#[test]
#[ignore = "fresh process with exact profile+ack+session mode; reviewed tiny GPU packet"]
fn qwen_pool_tiny_drained_windows_preserve_direct_parent() {
    run_drained_windows(true);
}
#[test]
#[ignore = "fresh process with profile UNSET and session mode; tiny matched legacy reference"]
fn qwen_pool_tiny_legacy_drained_windows_reference() {
    run_drained_windows(false);
}

#[test]
fn qwen_pool_infallible_spawn_requires_legacy_before_any_construction() {
    direct_spawn_precondition(false);
    let failure = std::panic::catch_unwind(|| direct_spawn_precondition(true))
        .expect_err("requested direct spawn must panic");
    let text = failure
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| failure.downcast_ref::<String>().map(String::as_str))
        .unwrap();
    assert!(text.contains("requires fallible Engine::spawn_with_mode SlotAware"));
}
