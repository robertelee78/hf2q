#[test]
#[ignore = "fresh process with native root authority and active Metal residency sets"]
fn no_promotion_finalization_retires_replaced_root_on_loading_queue() {
    use mlx_native::{DType, LiveRootLimits, NativeRootAuthority, RootCreationLimits};
    use std::time::{Duration, Instant};

    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let authority = NativeRootAuthority::install_early_with_live(
        RootCreationLimits {
            max_attempted_requested_bytes: 8 << 20,
            max_root_attempts: 64,
            max_raw_constructor_attempts: 64,
            max_registered_devices: 4,
            max_rope_cache_entries: 4,
        },
        LiveRootLimits {
            max_live_and_pending_requested_bytes: 8 << 20,
            max_live_and_pending_roots: 64,
        },
    )
    .expect("fresh native root authority");
    let seed = MlxDevice::new().expect("Metal device");
    authority
        .register_device(&seed)
        .expect("register actual device");
    let file = zero_layer_native_gguf(Some(48));
    let gguf = GgufFile::open(file.path()).expect("open native fixture");
    let mut progress = crate::serve::header::LoadProgress::new(false, 1, 0);
    let mut model =
        Qwen35Model::load_from_gguf(&gguf, &mut progress).expect("load actual native weights");
    let device = model
        .weight_device
        .as_ref()
        .expect("loading device")
        .clone();
    assert!(
        device.residency_sets_enabled(),
        "active residency set required"
    );

    // A distinct shared root avoids attributing a coalesced mapped page
    // to one logical tensor. Restore the original weight before execution.
    let weight = model.token_embd_native.as_mut().expect("native embedding");
    assert_eq!(weight.buffer.dtype(), DType::U8);
    let bytes = weight.buffer.element_count();
    let temporary = device
        .alloc_buffer(bytes, DType::U8, vec![bytes])
        .expect("distinct replacement root");
    let original = std::mem::replace(&mut weight.buffer, temporary);
    crate::inference::device_settle::settle_or_record(&device)
        .expect("commit initial source residency");
    let before = authority.live_snapshot().expect("live source inventory");
    drop(std::mem::replace(&mut weight.buffer, original));
    let pending = authority
        .live_snapshot()
        .expect("staged retirement inventory");
    assert_eq!(
        pending.live_roots, before.live_roots,
        "source set must still own the staged removal before finalization"
    );

    // This dense fixture promotes no experts. Only finalization can flush
    // the dropped source while the model and its loading device stay alive.
    let model = model
        .finalize_canonical_experts()
        .expect("checked finalization");
    let deadline = Instant::now() + Duration::from_secs(2);
    let after = loop {
        let live = authority.live_snapshot().expect("final source inventory");
        if live.live_roots < before.live_roots || Instant::now() >= deadline {
            break live;
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    let history = authority.snapshot().expect("complete creation history");
    assert!(history.covered && !history.legacy_used && history.unknown.is_none());
    assert_eq!(history.denied, 0);
    assert!(after.covered && after.unknown.is_none());
    assert_eq!(after.pending_roots, 0);
    assert_eq!(
        after.live_roots,
        before.live_roots - 1,
        "zero-promotion finalization must retire the replaced native root"
    );
    assert!(after.live_requested_bytes < before.live_requested_bytes);
    drop((model, device, seed));
}
