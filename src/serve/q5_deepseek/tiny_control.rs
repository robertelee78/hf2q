//! Named-userdata <=32MiB, actual constructors, zero model/forward/commands.
//! This cannot create a selected-case admission permit.
use super::collector;
use crate::inference::{
    backing_observation::deepseek_control::tiny_config,
    models::deepseek4::cache::{Deepseek4Cache, Deepseek4CachePlan},
};

#[test]
fn five_cache_eight_anchor_named_userdata_is_bounded_before_device() {
    let plan = Deepseek4CachePlan::for_context(&tiny_config(), 256).unwrap();
    assert_eq!(plan.resident_bytes, 1_083_392);
    let requested = 5 * plan.resident_bytes + 8 * 999_424;
    assert_eq!(requested, 13_412_352);
    assert!(requested + 122 * 65_536 + 1_048_576 < 32 * 1024 * 1024);
}

#[test]
#[ignore = "root-supervised fresh-process tiny actual constructor control; no model or GPU commands"]
fn actual_five_retained_caches_eight_anchors_join_same_collector() {
    use mlx_native::{MlxDevice, NativeRootAuthority, RootCreationLimits};
    const BYTES: usize = 13_412_352;
    const ROOTS: usize = 122;
    assert!(BYTES + ROOTS * 65536 + 1024 * 1024 < 32 * 1024 * 1024);
    let plan = Deepseek4CachePlan::for_context(&tiny_config(), 256).unwrap();
    assert_eq!(plan.resident_bytes, 1_083_392);
    let authority = NativeRootAuthority::install_early(RootCreationLimits {
        max_attempted_requested_bytes: BYTES,
        max_root_attempts: ROOTS,
        max_raw_constructor_attempts: ROOTS,
        max_registered_devices: 1,
        max_rope_cache_entries: 1,
    })
    .unwrap();
    let before = (
        mlx_native::dispatch_count(),
        mlx_native::commit_count(),
        mlx_native::sync_count(),
    );
    let device = MlxDevice::new().expect("actual Metal required; no skip");
    authority.register_device(&device).unwrap();
    let _scope = collector::HeaderScope::enter().unwrap();
    let caches = (0..5)
        .map(|_| Deepseek4Cache::allocate(&plan, device.clone()).unwrap())
        .collect::<Vec<_>>();
    let mut anchors = Vec::new();
    for (index, cache) in caches.iter().enumerate() {
        cache
            .q5_visit(&mut |name, buffer| {
                collector::field(&format!("tiny/cache{index}/{name}"), buffer)
            })
            .unwrap();
        if index < 4 {
            anchors.push(cache.snapshot().unwrap());
            anchors.push(cache.snapshot().unwrap());
        }
    }
    for (index, anchor) in anchors.iter().enumerate() {
        anchor
            .q5_visit(&mut |name, buffer| {
                collector::field(&format!("tiny/anchor{index}/{name}"), buffer)
            })
            .unwrap();
    }
    collector::checked_health().unwrap();
    assert_eq!(collector::observed_root_totals().unwrap(), (ROOTS, BYTES));
    caches[0]
        .q5_visit(&mut |name, buffer| collector::field(&format!("tiny/cache0/{name}"), buffer))
        .unwrap();
    assert_eq!(collector::observed_root_totals().unwrap(), (ROOTS, BYTES)); // alias roles do not duplicate parents

    let snapshot = authority.snapshot().unwrap();
    assert_eq!(
        (
            snapshot.successful_roots,
            snapshot.attempted_requested_bytes
        ),
        (ROOTS, BYTES)
    );
    assert_eq!(
        before,
        (
            mlx_native::dispatch_count(),
            mlx_native::commit_count(),
            mlx_native::sync_count()
        )
    );
    println!("Q5_TINY_CONSTRUCTORS roots=122 requested_bytes=13412352 caches=5 anchors=8 model=false commands=false physical_credit=false");
}
