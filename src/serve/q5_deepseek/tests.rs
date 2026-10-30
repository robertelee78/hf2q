use super::*;
use std::cell::Cell;

fn packet() -> Packet {
    let cells = [
        "same_fd_header",
        "source_native_binary",
        "weight_requested_capacity",
        "bootstrap_plus_four_sessions",
        "transient_and_pending_upper",
        "private_pool_domain_partition",
        "cpu_os_reaction_upper",
        "fresh_operating_review",
    ]
    .into_iter()
    .map(|key| {
        (
            key.to_string(),
            serde_json::json!({"state":"Planned", "evidence_sha256":"a".repeat(64)}),
        )
    })
    .collect::<serde_json::Map<_, _>>();
    serde_json::from_value(serde_json::json!({
        "schema":"hf2q.q5.deepseek.selected-case.preload.v1",
        "hf_source_sha256":"a".repeat(64), "native_package_sha256":"b".repeat(64),
        "binary_sha256":"c".repeat(64), "artifact_sha256":"d".repeat(64),
        "workload_sha256":"e".repeat(64), "policy_sha256":"f".repeat(64),
        "file":{"device":1,"inode":2,"bytes":3,"mtime_seconds":0,"mtime_nanoseconds":0,"ctime_seconds":0,"ctime_nanoseconds":0},
        "context":262144,"output_limit":8192,"logical_slots":4,
        "weight_policy":"FullCoalescedNoCopy","collector_upper_bytes":1048576,
        "phase":"preload", "cells":cells
    })).unwrap()
}

#[test]
fn actual_entry_refuses_missing_phase_before_load_callback() {
    let mut p = packet();
    p.cells.remove("private_pool_domain_partition");
    let called = Cell::new(false);
    let result = enter(&p, Path::new("not-opened-by-invalid-packet"), |_| {
        called.set(true);
        Ok(())
    });
    assert!(result.unwrap_err().to_string().contains("missing/extra"));
    assert!(!called.get());
}

#[test]
fn future_phase_never_borrows_preload_requirements() {
    for phase in [
        Phase::Load,
        Phase::Provision,
        Phase::Warmup,
        Phase::Forward,
        Phase::Drain,
        Phase::Shutdown,
    ] {
        let mut p = packet();
        p.phase = phase;
        assert!(p.validate().is_err());
        assert!(!p.requirements(phase).unwrap().contains(&"same_fd_header"));
    }
}

#[test]
fn planned_and_unknown_cannot_mint_permit_or_reduce_old_upper() {
    let mut p = packet();
    p.validate().unwrap();
    let message = reject_preload::<()>(
        &p,
        super::receipt::HeaderPlan::Unknown {
            reason: "HostControlWithoutActualHeader",
        },
    )
    .unwrap_err()
    .to_string();
    assert!(message.contains("134877044100"));
    assert!(message.contains("private_permit=NotMinted"));
    p.cells.insert(
        "fresh_operating_review".into(),
        packet::Cell::Unknown {
            reason: "stale actual observation".into(),
        },
    );
    assert!(reject_preload::<()>(
        &p,
        super::receipt::HeaderPlan::Unknown {
            reason: "HostControlWithoutActualHeader"
        }
    )
    .is_err());
    p.collector_upper_bytes = u64::MAX;
    assert!(reject_preload::<()>(
        &p,
        super::receipt::HeaderPlan::Unknown {
            reason: "HostControlWithoutActualHeader"
        }
    )
    .unwrap_err()
    .to_string()
    .contains("overflow"));
}

#[test]
fn serialized_permit_or_actual_claim_is_not_accepted() {
    let forged = r#"{"state":"ActualBound","bytes":0,"permit":true}"#;
    assert!(serde_json::from_str::<packet::Cell>(forged).is_err());
    assert!(activation(Some("packet.json".into()), None).is_err());
    assert!(activation(Some("".into()), Some("1".into())).is_err());
    activation(None, None).unwrap();
}

#[test]
fn unsupported_windows_and_zero_observer_charge_refuse_at_real_entry() {
    let mut p = packet();
    p.weight_policy = "ReviewedTensorWindows".into();
    let called = Cell::new(false);
    assert!(enter(&p, Path::new("not-opened"), |_| {
        called.set(true);
        Ok(())
    })
    .is_err());
    assert!(!called.get());
    let mut p = packet();
    p.collector_upper_bytes = 0;
    assert!(p.validate().is_err());
}

#[test]
fn actual_callback_chain_requires_every_own_result_and_rejects_failure() {
    let _scope = collector::HeaderScope::enter().unwrap();
    assert!(collector::HeaderScope::enter().is_err());
    let id = collector::chain();
    collector::command(id, 0, None);
    collector::command(id, 1, None);
    collector::command(id, 1, Some(true));
    collector::command(id, 0, Some(true));
    collector::checked_chain(id, 2).unwrap();
    collector::command(id, 2, None);
    collector::command(id, 2, Some(false));
    assert!(collector::checked_chain(id, 2).is_err());
}

#[test]
fn unsolicited_or_duplicate_result_poison_actual_callback_sink() {
    let _scope = collector::HeaderScope::enter().unwrap();
    let id = collector::chain();
    collector::command(id, 0, Some(true));
    assert!(collector::checked_chain(id, 2).is_err());
}

#[test]
fn actual_owner_callbacks_accept_request_zero_refuse_stale_and_overlap() {
    {
        let _scope = collector::HeaderScope::enter().unwrap();
        collector::owner_begin(0, 0, 9).unwrap();
        collector::owner_finish(0, 0, 9).unwrap();
    }
    for (request, slot, generation) in [(1, 0, 9), (0, 0, 8), (0, 1, 9)] {
        let _scope = collector::HeaderScope::enter().unwrap();
        collector::owner_begin(0, 0, 9).unwrap();
        assert!(collector::owner_finish(request, slot, generation).is_err());
        assert!(collector::owner_finish(0, 0, 9).is_err()); // sticky invalidation
    }
    let _scope = collector::HeaderScope::enter().unwrap();
    collector::owner_begin(0, 0, 9).unwrap();
    assert!(collector::owner_begin(1, 1, 1).is_err());
    assert!(collector::owner_finish(0, 0, 9).is_err());
}

#[test]
fn terminal_result_cannot_replace_earlier_own_checked_result() {
    let _scope = collector::HeaderScope::enter().unwrap();
    let id = collector::chain();
    collector::command(id, 0, None);
    collector::command(id, 1, None);
    collector::command(id, 1, Some(true));
    assert!(collector::checked_chain(id, 2).is_err());
    collector::command(id, 0, Some(true));
    assert!(collector::checked_chain(id, 2).is_err()); // no laundering after failure
}

#[test]
fn unpublished_page_observer_cannot_grant_runtime_support() {
    use super::page_observer::PublishedPageObserver;
    assert!(super::page_observer::Unavailable
        .require_runtime_support()
        .is_err());
}

#[test]
fn actual_metadata_context_transfers_only_after_owner_scope_detaches() {
    let scope = collector::HeaderScope::enter().unwrap();
    let context = scope.handle();
    assert!(context.attach().is_err());
    drop(scope);
    let counts = std::thread::spawn(move || {
        let scope = context.attach().unwrap();
        collector::owner_begin(0, 1, 7).unwrap();
        collector::owner_finish(0, 1, 7).unwrap();
        drop(scope);
        context.summary().unwrap()
    })
    .join()
    .unwrap();
    assert_eq!(counts, (0, 0, 0, false)); // metadata only; no native/FD observation
}

#[test]
fn pending_or_failed_metadata_context_cannot_transfer_to_another_worker() {
    let scope = collector::HeaderScope::enter().unwrap();
    let context = scope.handle();
    let id = collector::chain();
    collector::command(id, 0, None);
    drop(scope);
    assert!(context.attach().is_err());
    assert!(context.summary().is_err());
}

#[test]
fn actual_none_entry_calls_original_constructor_without_context_or_file_open() {
    let called = Cell::new(false);
    let value = load_selected(None, Path::new("does-not-exist"), |prepared| {
        assert!(prepared.is_none());
        called.set(true);
        Ok(17)
    })
    .unwrap();
    assert!(called.get());
    assert_eq!(value, 17);
}

#[test]
fn actual_closed_packet_rejects_duplicate_cells_and_uppercase_planned_hash() {
    let mut p = packet();
    p.cells.insert(
        "same_fd_header".into(),
        packet::Cell::Planned {
            evidence_sha256: "A".repeat(64),
        },
    );
    assert!(p.validate().is_err());
    let raw = serde_json::to_string(&packet()).unwrap();
    let duplicate = raw.replacen(
        "\"cells\":{",
        "\"cells\":{\"same_fd_header\":{\"state\":\"Unknown\",\"reason\":\"duplicate\"},",
        1,
    );
    assert_ne!(raw, duplicate);
    assert!(serde_json::from_str::<Packet>(&duplicate).is_err());
}

#[test]
fn actual_diagnostic_open_refuses_fifo_and_symlink_before_read() {
    use std::{
        ffi::CString,
        os::unix::{ffi::OsStrExt, fs::symlink},
    };
    let dir = tempfile::tempdir().unwrap();
    let fifo = dir.path().join("fifo");
    let name = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    // SAFETY: owned fresh test directory, live NUL-terminated CString.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    assert!(diagnostic_open(&fifo).is_err());
    let regular = dir.path().join("regular");
    std::fs::write(&regular, b"x").unwrap();
    let link = dir.path().join("alias");
    symlink(&regular, &link).unwrap();
    assert!(diagnostic_open(&link).is_err());
    assert!(diagnostic_open(&regular).is_ok());
}

#[test]
fn actual_same_fd_tiny_header_records_source_but_never_calls_model_constructor() {
    use std::os::unix::fs::MetadataExt;
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("header.gguf");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes()); // no tensors, header-only control
    bytes.extend_from_slice(&1u64.to_le_bytes());
    let key = b"general.architecture";
    let value = b"deepseek4";
    bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
    bytes.extend_from_slice(key);
    bytes.extend_from_slice(&8u32.to_le_bytes()); // GGUF string metadata
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    std::fs::write(&model, &bytes).unwrap();
    let m = std::fs::metadata(&model).unwrap();
    let mut p = packet();
    p.file = packet::FileIdentity {
        device: m.dev(),
        inode: m.ino(),
        bytes: m.len(),
        mtime_seconds: m.mtime(),
        mtime_nanoseconds: m.mtime_nsec(),
        ctime_seconds: m.ctime(),
        ctime_nanoseconds: m.ctime_nsec(),
    };
    let path = dir.path().join("packet.json");
    std::fs::write(&path, serde_json::to_vec(&p).unwrap()).unwrap();
    let called = Cell::new(false);
    let error = load_selected(Some(&path), &model, |_| {
        called.set(true);
        Ok(())
    })
    .unwrap_err();
    let report = format!("{error:#}");
    assert!(report.contains("RejectedPreload"));
    assert!(report.contains("(0, 0, 0, true)")); // actual retained FD, no native roots
    assert!(report.contains("NotMinted"));
    assert!(!called.get());
}

#[test]
fn actual_scheduler_auto_release_cleanup_does_not_authorize_stale_work() {
    use crate::serve::scheduler::{AdmitRequest, InflightBatchedScheduler, Scheduler};
    let mut scheduler = InflightBatchedScheduler::new(8, 4);
    let admitted = scheduler
        .admit(AdmitRequest {
            prompt_tokens: 1,
            max_tokens: 1,
            ..Default::default()
        })
        .unwrap();
    let handle = admitted.handle.unwrap();
    assert_eq!(scheduler.q5_request(handle).unwrap(), admitted.request_id.0);
    assert!(scheduler.q5_require_auto_released(handle).is_err());
    scheduler.advance_after_prefill(handle, 1);
    scheduler.advance_after_decode(handle); // actual automatic generation bump
    assert!(scheduler.q5_request(handle).is_err());
    scheduler.q5_require_auto_released(handle).unwrap();
    let next = scheduler
        .admit(AdmitRequest {
            prompt_tokens: 1,
            max_tokens: 1,
            preferred_slot: Some(handle.slot_id),
            ..Default::default()
        })
        .unwrap()
        .handle
        .unwrap();
    assert_eq!(next.slot_id, handle.slot_id);
    assert!(scheduler.q5_require_auto_released(handle).is_err());
    scheduler.release(next);
    assert!(scheduler.q5_require_auto_released(handle).is_err()); // empty but gen+2
}

#[test]
fn actual_completed_owner_witness_is_generation_specific_and_not_work_authority() {
    let _scope = collector::HeaderScope::enter().unwrap();
    assert!(collector::retained_cleanup_request(0, 4).is_err());
}

#[test]
fn actual_logical_owner_group_refuses_duplicate_slots_and_requests() {
    for keys in [
        vec![(0, 0, 1), (1, 0, 2)],
        vec![(0, 0, 1), (0, 1, 1)],
        vec![],
    ] {
        let _scope = collector::HeaderScope::enter().unwrap();
        assert!(collector::owner_begin_many(&keys).is_err());
        assert!(collector::checked_health().is_err());
    }
}

#[test]
fn actual_completed_owner_witness_retains_zero_id_and_refuses_generation_drift() {
    let _scope = collector::HeaderScope::enter().unwrap();
    collector::owner_begin(0, 0, 4).unwrap();
    collector::owner_finish(0, 0, 4).unwrap();
    assert_eq!(collector::retained_cleanup_request(0, 4).unwrap(), 0);
    assert!(collector::retained_cleanup_request(0, 5).is_err());
    assert!(collector::retained_cleanup_request(0, 4).is_err()); // sticky, not a retry
}

#[test]
fn actual_required_domain_registry_keeps_future_plans_out_of_observation_credit() {
    for phase in [
        Phase::Preload,
        Phase::Load,
        Phase::Provision,
        Phase::Warmup,
        Phase::Forward,
        Phase::Drain,
        Phase::Shutdown,
    ] {
        let domains = requirements::describe(phase);
        assert!(domains.len() <= 32);
        let keys = domains
            .iter()
            .map(|(key, _)| *key)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(keys.len(), domains.len());
        assert!(keys.contains("native/whole_current_live_pending_scope"));
        if matches!(phase, Phase::Provision | Phase::Forward | Phase::Drain) {
            for owner in [
                "loaded/current",
                "retained_slot/0",
                "retained_slot/1",
                "retained_slot/2",
                "retained_slot/3",
            ] {
                assert!(keys.contains(owner));
            }
            assert!(domains
                .iter()
                .all(|(_, proof)| !proof.contains("NotPlanned")));
        }
    }
}

#[test]
fn actual_owner_finish_refuses_unjoined_pending_result_before_publication() {
    let _scope = collector::HeaderScope::enter().unwrap();
    collector::owner_begin(0, 0, 1).unwrap();
    let id = collector::chain();
    collector::command(id, 0, None);
    assert!(collector::owner_finish(0, 0, 1).is_err());
    collector::command(id, 0, Some(true));
    assert!(collector::checked_chain(id, 1).is_err());
    assert!(collector::owner_finish(0, 0, 1).is_err());
}

#[test]
fn actual_all_results_aggregate_exceeds_old_event_cap_without_retaining_history() {
    let scope = collector::HeaderScope::enter().unwrap();
    for _ in 0..10_000 {
        let id = collector::chain();
        collector::command(id, 0, None);
        collector::command(id, 0, Some(true));
        collector::checked_chain(id, 1).unwrap();
    }
    assert_eq!(
        scope.handle().summary().unwrap(),
        (20_000, 10_000, 0, false)
    );
}

#[test]
fn actual_operation_context_rejects_another_model_even_with_identical_owner_keys() {
    let first_scope = collector::HeaderScope::enter().unwrap();
    let first = first_scope.handle();
    drop(first_scope);
    let second_scope = collector::HeaderScope::enter().unwrap();
    let second = second_scope.handle();
    collector::owner_begin(0, 0, 1).unwrap();
    assert!(first.require_current_model(Some(&second)).is_err());
    assert!(second.require_current_model(Some(&first)).is_err());
    second.require_current_model(Some(&second)).unwrap();
    collector::owner_finish(0, 0, 1).unwrap();
}

#[test]
fn actual_lifecycle_cannot_overlap_request_or_publish_pending_results() {
    {
        let _scope = collector::HeaderScope::enter().unwrap();
        collector::lifecycle_begin(collector::Lifecycle::Provision).unwrap();
        collector::require_lifecycle(collector::Lifecycle::Provision).unwrap();
        assert!(collector::owner_begin(0, 0, 1).is_err());
        assert!(collector::lifecycle_finish(collector::Lifecycle::Provision, true).is_err());
    }
    let _scope = collector::HeaderScope::enter().unwrap();
    collector::lifecycle_begin(collector::Lifecycle::Warmup).unwrap();
    let id = collector::chain();
    collector::command(id, 0, None);
    assert!(collector::lifecycle_finish(collector::Lifecycle::Warmup, true).is_err());
}

#[test]
fn actual_bounded_receipt_refuses_overflow_and_manifest_is_write_once() {
    assert_eq!(super::receipt::encode(&"x", 3).unwrap(), b"\"x\"");
    assert!(super::receipt::encode(&"xx", 3).is_err());
    let mut record = super::manifest::CurrentManifest::new(1, 1, &[(0, 0, 1)], None).unwrap();
    record.field("actual.optional", None).unwrap();
    assert!(record.field("actual.optional", None).is_err());
    record.close().unwrap();
    assert!(record.field("late", None).is_err());
    assert!(record.close().is_err());
}

#[test]
fn actual_unbalanced_swap_refuses_owner_publication() {
    let _scope = collector::HeaderScope::enter().unwrap();
    collector::owner_begin(0, 0, 1).unwrap();
    collector::mounted_begin(1).unwrap();
    collector::mounted_end().unwrap();
    assert!(collector::owner_finish(0, 0, 1).is_err());
}
