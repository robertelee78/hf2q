//! Real validator arithmetic; mocked OS values are not actual host admission.
use super::*;
use crate::inference::models::deepseek4::selected_budget_plan::Minimum;
const M: u64 = 1024 * 1024;
fn policy() -> Policy {
    Policy {
        page_bytes: 16384,
        max_buffer_bytes: 8 * 1024 * M,
        device_registry_id: 1,
        native_live_bytes: 4096 * M,
        native_live_roots: 1024,
        native_attempt_bytes: 8192 * M,
        native_attempt_roots: 2048,
        native_raw_attempts: 2048,
        rope_entries: 16,
        cpu_bytes: 256 * M,
        tools_bytes: 32 * M,
        client_bytes: 16 * M,
        driver_bytes: 32 * M,
        observer_bytes: 8 * M,
        inactive_file_residual_bytes: 32 * M,
        additional_noncase_bytes: 0,
        reaction_bytes: 32 * M,
        runtime_free_bytes: 32 * M,
        deadline_ms: 60000,
        sample_interval_ms: 500,
        max_http_inflight: 4,
        process_stop_bytes: 1024 * M,
        max_decoded_bytes: M,
        count_file_cache_as_operating_available: true,
        charge_creation_envelope: false,
        retired_native_residual_bytes: 32 * M,
    }
}
fn limits() -> Limits {
    Limits {
        queue_capacity: 4,
        max_enqueue_attempts: 28,
        max_prompt_tokens: 262144,
        max_output_tokens: 8192,
    }
}
fn minimum() -> Minimum {
    Minimum {
        weights: M,
        four_caches_stub: M,
        weight_roots: 2,
        cache_roots: 5,
    }
}
fn os() -> operating_os::Sample {
    operating_os::Sample {
        total: 16 * 1024 * M,
        free: 8192 * M,
        file_cache: 4096 * M,
        swap: 0,
        page: 16384,
        rss: 16 * M,
        footprint: 16 * M,
    }
}
#[test]
fn actual_validator_positive_retains_all_nonzero_cells_and_full_native_charge() {
    numeric(policy(), limits(), minimum(), os(), M).unwrap();
    let mut p = policy();
    p.count_file_cache_as_operating_available = false;
    numeric(p, limits(), minimum(), os(), M).unwrap();
    // Opaque roles are INCLUDED in4096MiB cap, never deducted from it.
    assert!(p.native_live_bytes > minimum().weights + minimum().four_caches_stub);
}
#[test]
fn actual_validator_refuses_zero_allowance_low_mandatory_and_limits() {
    for i in 0..11 {
        let mut p = policy();
        match i {
            0 => p.cpu_bytes = 0,
            1 => p.tools_bytes = 0,
            2 => p.client_bytes = 0,
            3 => p.driver_bytes = 0,
            4 => p.observer_bytes = 0,
            5 => p.inactive_file_residual_bytes = 0,
            6 => p.reaction_bytes = 0,
            7 => p.runtime_free_bytes = 0,
            8 => p.native_live_bytes = minimum().weights,
            9 => p.native_attempt_roots = 1,
            _ => p.native_raw_attempts = 1,
        };
        assert!(numeric(p, limits(), minimum(), os(), M).is_err(), "case{i}");
    }
}
#[test]
fn actual_validator_refuses_oversubscription_overflow_wrong_geometry_and_cpu_floor() {
    let mut p = policy();
    p.native_live_bytes = 15 * 1024 * M;
    p.native_attempt_bytes = p.native_live_bytes;
    assert!(numeric(p, limits(), minimum(), os(), M).is_err());
    let mut p = policy();
    p.additional_noncase_bytes = u64::MAX;
    assert!(numeric(p, limits(), minimum(), os(), M).is_err());
    let mut m = minimum();
    m.weights = u64::MAX;
    assert!(numeric(policy(), limits(), m, os(), M).is_err());
    let mut o = os();
    o.page = 4096;
    assert!(numeric(policy(), limits(), minimum(), o, M).is_err());
    let mut p = policy();
    p.cpu_bytes = M;
    assert!(numeric(p, limits(), minimum(), os(), M).is_err());
    let mut l = limits();
    l.max_output_tokens = 8193;
    assert!(numeric(policy(), l, minimum(), os(), M).is_err());
}
#[test]
fn actual_case_hash_and_evidence_bind_policy_and_real_regular_file() {
    let mut c = Case {
        policy: policy(),
        limits: limits(),
        review_path: "/pending".into(),
        review_sha256: "a".repeat(64),
        capability_receipt_path: "/pending".into(),
        capability_receipt_sha256: "b".repeat(64),
    };
    let before = case_digest(&c).unwrap();
    c.policy.reaction_bytes += 1;
    assert_ne!(before, case_digest(&c).unwrap());
    let f = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(f.path(), b"{\"actual\":1}").unwrap();
    let raw = std::fs::read(f.path()).unwrap();
    let v: serde_json::Value = evidence(f.path().to_str().unwrap(), &sha(&raw)).unwrap();
    assert_eq!(v["actual"], 1);
    assert!(evidence::<serde_json::Value>(f.path().to_str().unwrap(), &"0".repeat(64)).is_err());
    let d = tempfile::tempdir().unwrap();
    let link = d.path().join("alias");
    std::os::unix::fs::symlink(f.path(), &link).unwrap();
    assert!(evidence::<serde_json::Value>(link.to_str().unwrap(), &sha(&raw)).is_err());
}

#[test]
fn actual_review_window_and_closed_case_do_not_accept_a_serialized_permit() {
    review_fresh(10, 100, 50).unwrap();
    for (a, b, n) in [(51, 100, 50), (10, 49, 50), (10, 9, 10), (0, 86_400_001, 1)] {
        assert!(review_fresh(a, b, n).is_err());
    }
    let c = Case {
        policy: policy(),
        limits: limits(),
        review_path: "/pending".into(),
        review_sha256: "a".repeat(64),
        capability_receipt_path: "/pending".into(),
        capability_receipt_sha256: "b".repeat(64),
    };
    let mut v = serde_json::to_value(c).unwrap();
    v.as_object_mut()
        .unwrap()
        .insert("permit".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<Case>(v).is_err());
}
