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
    let message = reject_preload::<()>(&p).unwrap_err().to_string();
    assert!(message.contains("134877044100"));
    assert!(message.contains("private_permit=NotMinted"));
    p.cells.insert(
        "fresh_operating_review".into(),
        packet::Cell::Unknown {
            reason: "stale actual observation".into(),
        },
    );
    assert!(reject_preload::<()>(&p).is_err());
    p.collector_upper_bytes = u64::MAX;
    assert!(reject_preload::<()>(&p)
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
