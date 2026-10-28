//! Closed prospective serialized-trace profiles; never physical RAM admission.
use serde::{Deserialize, Serialize};
pub(super) const LEGACY_LIMIT: u64 = 256 * 1024 * 1024;
pub(super) const NATIVE_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
pub(super) const NATIVE_V3_LIMIT: u64 = 8 * 1024 * 1024 * 1024;
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Profile {
    NativeCommandCaptureV2,
    NativeCommandCaptureV3,
}
pub(super) fn deserialize_optional<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<Profile>, D::Error> {
    let name = String::deserialize(d)?;
    match name.as_str() {
        "native_command_capture_v2" => Ok(Some(Profile::NativeCommandCaptureV2)),
        "native_command_capture_v3" => Ok(Some(Profile::NativeCommandCaptureV3)),
        _ => Err(serde::de::Error::custom(
            "unsupported metadata budget profile",
        )),
    }
}
pub(super) fn limit(
    profile: Option<Profile>,
    native_diagnostic: Option<bool>,
) -> Result<u64, String> {
    match profile {
        None => Ok(LEGACY_LIMIT),
        Some(Profile::NativeCommandCaptureV2) if native_diagnostic == Some(true) => {
            Ok(NATIVE_LIMIT)
        }
        Some(Profile::NativeCommandCaptureV3) if native_diagnostic == Some(true) => {
            Ok(NATIVE_V3_LIMIT)
        }
        Some(_) => Err(
            "native metadata budget profile requires an explicit native diagnostic observer".into(),
        ),
    }
}
pub(super) fn event_limit(profile: Option<Profile>) -> u64 {
    match profile {
        Some(Profile::NativeCommandCaptureV3) => 16_000_000,
        _ => 1_000_000,
    }
}
#[cfg(test)]
mod tests {
    use super::super::{native, validate_config, Config, SCHEMA};
    use super::*;
    use serde_json::json;
    fn config(profile: Option<Profile>, native_enabled: bool) -> Config {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let output = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "hf2q-metadata-budget-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let mut identity = std::collections::BTreeMap::new();
        for key in [
            "server_source",
            "server_binary_sha256",
            "native_checksum",
            "model_sha256",
            "tokenizer_sha256",
            "template_sha256",
            "profile_sha256",
        ] {
            identity.insert(
                key.into(),
                "a".repeat(if key == "server_source" { 40 } else { 64 }),
            );
        }
        identity.insert("native_version".into(), "0.15.8".into());
        Config {
            schema: SCHEMA.into(),
            output,
            run_id: "metadata_budget_control".into(),
            server_budget_bytes: 128 * 1024 * 1024,
            client_budget_bytes: 128 * 1024 * 1024,
            metadata_budget_profile: profile,
            forward_demand_policy: None,
            native_observer: native_enabled.then(|| native::Config {
                event_limit: 1000,
                metadata_limit_bytes: 1024 * 1024,
                diagnostic_case: true,
                control_nonce: "n".repeat(32),
                witness_manifest: native::WitnessManifest {
                    schema: "hf2q.f2.forward-witness-policy.v1".into(),
                    operator: "target_input_rms_norm".into(),
                    input_sources: vec!["qwen35.forward.actual_tokens".into()],
                    coverage: "each_admitted_lane_in_first_fixed_window".into(),
                    batching_scope: "operator_only_full_body_unknown".into(),
                },
            }),
            identity,
        }
    }
    #[test]
    fn queued_demand_actual_config_gate_is_diagnostic_qwen_only() {
        let mut c = config(None, true);
        c.forward_demand_policy = Some(serde_json::from_value(serde_json::json!({
            "population_policy":"queued_forward_demand_v3","qwen_prefill_chunk_tokens":128
        })).unwrap());
        assert!(validate_config(&c).is_ok());
        c.native_observer.as_mut().unwrap().witness_manifest.input_sources =
            vec!["gemma4.scalar_forward".into()];
        assert!(validate_config(&c).unwrap_err().contains("Qwen"));
        c.native_observer = None;
        assert!(validate_config(&c).unwrap_err().contains("diagnostic"));
    }

    #[test]
    fn metadata_budget_legacy_exact_boundary_is_unchanged() {
        let mut c = config(None, false);
        assert!(validate_config(&c).is_ok());
        c.server_budget_bytes += 1;
        assert!(validate_config(&c).unwrap_err().contains("quota"));
        assert!(!c.output.exists());
    }
    #[test]
    fn metadata_budget_native_exact_boundary_and_client_minimum() {
        let mut c = config(Some(Profile::NativeCommandCaptureV2), true);
        c.server_budget_bytes = NATIVE_LIMIT - c.client_budget_bytes;
        assert!(validate_config(&c).is_ok());
        c.server_budget_bytes += 1;
        assert!(validate_config(&c).unwrap_err().contains("quota"));
        c.server_budget_bytes -= 1;
        c.client_budget_bytes -= 1;
        assert!(validate_config(&c).unwrap_err().contains("quota"));
        assert!(!c.output.exists());
    }
    #[test]
    fn metadata_budget_native_requires_real_opt_in_and_closed_enum() {
        assert_eq!(
            serde_json::to_value(Profile::NativeCommandCaptureV2).unwrap(),
            json!("native_command_capture_v2")
        );
        let mut c = config(Some(Profile::NativeCommandCaptureV2), false);
        assert!(validate_config(&c).is_err());
        c.native_observer = config(None, true).native_observer;
        c.native_observer.as_mut().unwrap().diagnostic_case = false;
        assert!(validate_config(&c).is_err());
        for value in [
            json!("other"),
            json!("native_command_capture_v1"),
            json!(null),
            json!(1),
            json!({"native_command_capture_v2":null}),
        ] {
            let v = json!({"schema":SCHEMA,"output":"/must-not-open","run_id":"control","server_budget_bytes":1,"client_budget_bytes":1,"identity":{},"metadata_budget_profile":value});
            assert!(serde_json::from_value::<Config>(v).is_err());
        }
    }
    #[test]
    fn metadata_budget_v3_serialized_and_event_boundaries_are_explicit() {
        for (profile, total, events) in [
            (None, LEGACY_LIMIT, 1_000_000),
            (
                Some(Profile::NativeCommandCaptureV2),
                NATIVE_LIMIT,
                1_000_000,
            ),
            (
                Some(Profile::NativeCommandCaptureV3),
                NATIVE_V3_LIMIT,
                16_000_000,
            ),
        ] {
            let mut c = config(profile, true);
            c.server_budget_bytes = total - c.client_budget_bytes;
            c.native_observer.as_mut().unwrap().event_limit = events;
            assert!(validate_config(&c).is_ok());
            c.native_observer.as_mut().unwrap().event_limit += 1;
            assert!(validate_config(&c).is_err());
            c.native_observer.as_mut().unwrap().event_limit = events;
            c.server_budget_bytes += 1;
            assert!(validate_config(&c).is_err());
            assert!(!c.output.exists());
        }
        let mut c = config(Some(Profile::NativeCommandCaptureV3), true);
        c.native_observer.as_mut().unwrap().metadata_limit_bytes = 16 * 1024 * 1024;
        assert!(validate_config(&c).is_ok());
        c.native_observer.as_mut().unwrap().metadata_limit_bytes += 1;
        assert!(validate_config(&c).is_err());
        assert!(validate_config(&config(Some(Profile::NativeCommandCaptureV3), false)).is_err());
        assert_eq!(
            serde_json::to_value(Profile::NativeCommandCaptureV3).unwrap(),
            json!("native_command_capture_v3")
        );
    }
    #[test]
    fn metadata_budget_writer_checked_arithmetic_refuses_past_each_bound() {
        for cap in [LEGACY_LIMIT, NATIVE_LIMIT, NATIVE_V3_LIMIT] {
            assert_eq!(super::super::checked_charge(cap - 1, 0, cap), Some(cap));
            assert_eq!(super::super::checked_charge(cap, 0, cap), None);
        }
        assert_eq!(
            super::super::checked_charge(u64::MAX, 0, NATIVE_LIMIT),
            None
        );
    }
}
