use super::*;
fn limits() -> (RootCreationLimits, LiveRootLimits) {
    (RootCreationLimits { max_attempted_requested_bytes: 4096, max_root_attempts: 2,
        max_raw_constructor_attempts: 2, max_registered_devices: 1, max_rope_cache_entries: 1 },
     LiveRootLimits { max_live_and_pending_requested_bytes: 4096, max_live_and_pending_roots: 2 })
}
fn snapshots() -> (RootCreationSnapshot, LiveRootSnapshot) {
    let (c,l)=limits();
    (RootCreationSnapshot { scope: "MOCK requested history", limits: Some(c), covered: true,
        legacy_used: false, unknown: None, attempted_requested_bytes: 0, root_attempts: 0,
        raw_attempt_reservations: 0, inflight: 0, successful_roots: 0, cache_slot_history: 0,
        published_cache_entries: 0, registered_devices: 0, denied: 0 },
     LiveRootSnapshot { scope: "MOCK requested live", limits: Some(l), covered: true,
        unknown: None, pending_requested_bytes: 0, live_requested_bytes: 0, pending_roots: 0,
        live_roots: 0, high_water_requested_bytes: 0, denied: 0, final_releases: 0 })
}
#[test]
fn actual_limit_validator_rejects_missing_capacity_or_multiple_devices() {
    let (c,l)=limits();validate_limits(c,l).unwrap();
    let mut bad=c;bad.max_registered_devices=2;assert!(validate_limits(bad,l).is_err());
    let mut bad=c;bad.max_raw_constructor_attempts=1;assert!(validate_limits(bad,l).is_err());
    let mut bad=l;bad.max_live_and_pending_requested_bytes=4097;assert!(validate_limits(c,bad).is_err());
    let mut bad=l;bad.max_live_and_pending_roots=0;assert!(validate_limits(c,bad).is_err());
}
#[test]
fn actual_checkpoint_validator_keeps_history_after_live_release() {
    let (mut c,mut l)=snapshots();validate_snapshots(&c,&l,true).unwrap();
    c.attempted_requested_bytes=64;c.root_attempts=1;c.successful_roots=1;
    c.raw_attempt_reservations=1;c.registered_devices=1;
    l.high_water_requested_bytes=64;l.final_releases=1;
    validate_snapshots(&c,&l,false).unwrap(); // live0 did not refund history64.
    assert!(validate_snapshots(&c,&l,true).is_err());
    c.attempted_requested_bytes=4097;assert!(validate_snapshots(&c,&l,false).is_err());
}
#[test]
fn actual_checkpoint_validator_refuses_unknown_denied_pending_and_overflow() {
    let (c,l)=snapshots();
    for index in 0..9 {
        let(mut c,mut l)=(c.clone(),l.clone());
        match index {
            0=>c.unknown=Some("unknown"),1=>l.covered=false,2=>c.denied=1,
            3=>l.denied=1,4=>c.inflight=1,5=>l.pending_roots=1,
            6=>{l.live_requested_bytes=usize::MAX;l.pending_requested_bytes=1;},
            7=>{l.live_roots=usize::MAX;l.pending_roots=1;},_=>c.legacy_used=true,
        }
        assert!(validate_snapshots(&c,&l,false).is_err(),"case {index}");
    }
}
#[test]
fn actual_context_binding_refuses_detached_or_another_owner() {
    use crate::serve::q5_deepseek::collector;
    let first=collector::HeaderScope::enter().unwrap();let original=first.handle();
    original.require_current_model(Some(&original)).unwrap();drop(first);
    assert!(original.require_current_model(Some(&original)).is_err());
    let second=collector::HeaderScope::enter().unwrap();let other=second.handle();
    assert!(other.require_current_model(Some(&original)).is_err());
    assert!(original.require_current_model(Some(&other)).is_err());
    assert!(other.require_current_model(None).is_err());
    other.require_current_model(Some(&other)).unwrap();
}
#[test]
#[ignore = "fresh root-supervised process; actual native handle/device binding and64B upload only; no commands/model/admission"]
fn actual_early_native_handle_same_device_and_worker_transfer() {
    use crate::serve::q5_deepseek::collector;
    // BEFORE native installation or Device:64B payload, <=4096B native quota,
    // two roots/two attempts, one actual device. No cache/model/helper imports.
    let(c,l)=limits();assert!(64<=l.max_live_and_pending_requested_bytes);
    let before=(mlx_native::dispatch_count(),mlx_native::commit_count(),mlx_native::sync_count());
    let scope=collector::HeaderScope::enter().unwrap();let context=scope.handle();
    collector::with_current(|cstate|install_reviewed(&mut cstate.native_domain,c,l)).unwrap();
    let device=MlxDevice::new().expect("actual device required; no skip");
    collector::register_model_device(&device).unwrap();
    let data=[7u8;64];let root=device.upload_shared_bytes(&data,mlx_native::DType::U8,vec![64]).unwrap();
    assert!(matches!(root.root_origin_snapshot(),mlx_native::RootOriginSnapshot::KnownNative{requested_bytes:64,..}));
    collector::checked_health().unwrap();
    drop(scope);
    let root=std::thread::spawn(move || {
        let attached=context.attach().unwrap();context.require_current_model(Some(&context)).unwrap();
        collector::checked_health().unwrap();
        collector::with_current(|state| {
            let h=state.native_domain.handle.context("same carried handle missing")?;
            assert_eq!(h.snapshot()?.attempted_requested_bytes,64);
            assert_eq!(h.snapshot()?.registered_devices,1);
            assert_eq!(h.live_snapshot()?.live_requested_bytes,64);
            Ok(())
        }).unwrap();
        // Duplicate installation refuses BEFORE invoking native install. Stop
        // after sticky error: no subsequent native allocation/GPU/replay.
        assert!(collector::with_current(|state|install_reviewed(&mut state.native_domain,c,l)).is_err());
        assert!(collector::checked_health().is_err());
        drop(attached);root
    }).join().unwrap();
    assert_eq!(before,(mlx_native::dispatch_count(),mlx_native::commit_count(),mlx_native::sync_count()));
    println!("Q5_EARLY_NATIVE_BINDING {}",serde_json::json!({"schema":"hf2q.q5.early-native-binding-tiny.v1","known_requested_bytes":64,"devices":1,"worker_transfer_checked":true,"duplicate_refused_sticky":true,"gpu_commands":false,"physical_credit_bytes":0,"runtime_permit":false}));
    drop(root);drop(device); // no immediate release/physical-refund assertion.
}
