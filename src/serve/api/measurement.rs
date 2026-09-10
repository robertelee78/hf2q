//! Path-free runtime facts for a managed measurement process.
//!
//! This snapshot binds the active tokenizer, template, defaults and controls.
//! It does not infer model/binary hashes from names or hash a model pathname
//! after loading. The measurement launcher supplies artifact hashes for the
//! files it verified before starting its owned child process.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::engine::Engine;
use super::state::{AppState, ServerConfig};
use crate::serve::multi_model::LoadedEngine;

#[derive(Default)]
pub(super) struct FingerprintCache(Mutex<BTreeMap<u64, (String, String)>>);

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn canonical(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let ordered: BTreeMap<_, _> = map.into_iter().collect();
            Value::Object(
                ordered
                    .into_iter()
                    .map(|(k, v)| (k, canonical(v)))
                    .collect(),
            )
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonical).collect()),
        other => other,
    }
}

fn grammar_control(config: &ServerConfig) -> Value {
    // Follow request preparation's precedence: --gcd installs the embedded
    // grammar first; a schema default is considered only without that flag.
    let (kind, grammar) = if config.gcd {
        ("gbnf", Some(include_str!("grammar/gcd_w1.gbnf")))
    } else {
        ("compiled_json_schema", config.gcd_schema_grammar.as_deref())
    };
    json!({
        "active": grammar.is_some(),
        "kind": grammar.map(|_| kind),
        "sha256": grammar.map(|text| digest(text.as_bytes())),
        "locked": config.gcd_schema_locked,
    })
}

pub(super) fn snapshot(state: &AppState, engines: &[Arc<LoadedEngine<Engine>>]) -> Value {
    // A measurement uses one owned runtime/model. Refuse an ambiguous pool
    // instead of selecting the first model or silently binding another one.
    let [loaded] = engines else {
        return Value::Null;
    };
    let engine = &loaded.engine;
    let Ok(model_path) = engine.info().model_path.canonicalize() else {
        return Value::Null;
    };
    let Ok(mut cache) = state.measurement_cache.0.lock() else {
        return Value::Null;
    };
    cache.retain(|generation, _| *generation == loaded.generation);
    let fingerprints = match cache.get(&loaded.generation) {
        Some(pair) => pair.clone(),
        None => {
            let Ok(serialized) = engine.tokenizer().to_string(false) else {
                return Value::Null;
            };
            let Ok(tokenizer): Result<Value, _> = serde_json::from_str(&serialized) else {
                return Value::Null;
            };
            let tokenizer_hash = digest(canonical(tokenizer).to_string().as_bytes());
            // Native encoders can use an empty template string. Their named
            // encoder and the launcher's binary digest bind that behavior.
            let template = canonical(json!({
                "source": format!("{:?}", engine.info().chat_template_source),
                "template": engine.chat_template(),
            }));
            let pair = (tokenizer_hash, digest(template.to_string().as_bytes()));
            cache.insert(loaded.generation, pair.clone());
            pair
        }
    };
    drop(cache);
    canonical(json!({
        "schema_version": "hf2q.measurement-snapshot.v1",
        "process_pid": std::process::id(),
        "model_id": engine.model_id(),
        // Locator fingerprint only; the managed launcher separately hashes
        // the file bytes. Never represent a path digest as a model digest.
        "model_locator_sha256": digest(model_path.to_string_lossy().as_bytes()),
        "architecture": engine.info().arch_str,
        "engine_generation": loaded.generation,
        "tokenizer_sha256": fingerprints.0,
        "template_sha256": fingerprints.1,
        "engine_config": super::control::engine_config_identity_json(&loaded.config_identity),
        "admission": {
            "queue_capacity": state.config.queue_capacity,
            "max_concurrent_requests": state.config.max_concurrent_requests,
            "request_timeout_seconds": state.config.request_timeout_seconds,
        },
        "sampling_defaults": {
            "repetition_penalty": state.config.default_repetition_penalty,
            "thinking_token_budget": state.config.default_thinking_token_budget,
            "tool_thinking_token_budget": state.config.default_tool_thinking_token_budget,
            "overflow_policy": format!("{:?}", state.config.default_overflow_policy),
        },
        "active_controls": {
            "glp": {"active": loaded.config_identity.glp_active,
                    "alpha_override_bits": loaded.config_identity.glp_alpha_bits},
            "grammar": grammar_control(&state.config),
            "dwq_overlay": loaded.config_identity.dwq_overlay,
            "vision_projector": state.mmproj.is_some(),
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[test]
    fn measurement_grammar_identity_tracks_real_default_and_lock() {
        let mut config = ServerConfig::default();
        assert_eq!(grammar_control(&config)["active"], false);
        config.gcd_schema_grammar = Some("root ::= \"safe\"".into());
        config.gcd_schema_locked = true;
        let schema = grammar_control(&config);
        assert_eq!(schema["active"], true);
        assert_eq!(schema["locked"], true);
        assert_eq!(schema["sha256"], digest(b"root ::= \"safe\""));
        config.gcd = true;
        assert_ne!(grammar_control(&config)["sha256"], schema["sha256"]);
    }

    #[test]
    fn measurement_config_distinguishes_steering_from_plain_runtime() {
        use crate::serve::multi_model::{EngineConfig, EngineConfigIdentity};
        let mut config = EngineConfig::default();
        let plain = EngineConfigIdentity::from(&config);
        config.glp_path = Some("private-vector.gguf".into());
        config.glp_alpha = Some(0.0);
        let zero_dose = EngineConfigIdentity::from(&config);
        assert_ne!(plain, zero_dose);
        assert!(zero_dose.glp_active);
        assert_eq!(zero_dose.glp_alpha_bits, Some(0));
        config.glp_alpha = Some(1.0);
        assert_ne!(zero_dose, EngineConfigIdentity::from(&config));
        let wire = super::super::control::engine_config_identity_json(&zero_dose);
        assert!(!wire.to_string().contains("private-vector"));
    }

    #[tokio::test]
    async fn measurement_endpoint_cannot_bind_an_empty_runtime() {
        let app = super::super::build_router(AppState::new(ServerConfig::default()));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/hf2q/v1/runtime")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert!(json["measurement_snapshot"].is_null());
    }
}
