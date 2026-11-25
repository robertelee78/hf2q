//! Bound subject provenance for the separate, engine-free C2 executor.
use super::*;
use std::path::Path;

fn hex_id(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

pub(super) fn validate_subject_identity(
    source: &Value,
    images: &Value,
    calibration: &Value,
    expected_commit: &str,
) -> Result<(), &'static str> {
    if !hex_id(expected_commit, 40)
        || source["schema"] != "hf2q.private.c2-diagnostic-source-manifest.v1"
        || source["source_commit"] != expected_commit
        || images["source_commit"] != expected_commit
        || images["build_environment"]["HF2Q_BUILD_GIT_SHA"] != expected_commit
        || images["source_root"] != source["root"]
    {
        return Err("expected commit, source manifest and actual image build must agree");
    }
    if images["registry_native_version"] != "0.24.0" {
        return Err("reviewed published native pin required");
    }
    let production = &images["images"]["production"];
    if production["path"].as_str().is_none_or(|p| !Path::new(p).is_absolute())
        || production["sha256"].as_str().is_none_or(|s| !hex_id(s, 64))
        || calibration["binary"]["path"] != production["path"]
        || calibration["binary"]["sha256"] != production["sha256"]
    {
        return Err("calibration must bind the actual subject production image");
    }
    Ok(())
}

pub(super) fn rendering_closure(source: &Value) -> Value {
    let root = PathBuf::from(source["root"].as_str().unwrap());
    assert!(root.is_absolute());
    let files = source["file_sha256"].as_object().unwrap();
    let mut closure = serde_json::Map::new();
    macro_rules! component {
        ($relative:literal, $include:literal) => {{
            let digest = hash(include_bytes!($include));
            assert_eq!(files[root.join($relative).to_str().unwrap()], digest,
                "oracle production rendering component differs from subject: {}", $relative);
            closure.insert($relative.to_owned(), json!(digest));
        }};
    }
    component!("Cargo.toml", "../../../Cargo.toml");
    component!("Cargo.lock", "../../../Cargo.lock");
    component!("src/serve/mod.rs", "../mod.rs");
    component!("src/serve/api/handlers.rs", "handlers.rs");
    component!("src/serve/api/engine.rs", "engine.rs");
    component!("src/serve/api/schema.rs", "schema.rs");
    component!("src/serve/api/measurement.rs", "measurement.rs");
    component!("src/serve/api/registry.rs", "registry.rs");
    component!("src/serve/api/reasoning_controls.rs", "reasoning_controls.rs");
    component!("src/serve/api/grammar/request.rs", "grammar/request.rs");
    component!("src/serve/api/grammar/request_validation.rs", "grammar/request_validation.rs");
    component!("src/core/chat_templates.rs", "../../core/chat_templates.rs");
    component!("src/core/tokenizer_adapter.rs", "../../core/tokenizer_adapter.rs");
    component!("src/inference/models/qwen35/tokenizer.rs", "../../inference/models/qwen35/tokenizer.rs");
    json!(closure)
}

#[test]
fn c2_subject_identity_rejects_wrong_commit_image_and_native() {
    let commit = "a".repeat(40);
    let source = json!({"schema":"hf2q.private.c2-diagnostic-source-manifest.v1",
        "source_commit":commit,"root":"/subject"});
    let mut images = json!({"source_commit":commit,"source_root":"/subject",
        "build_environment":{"HF2Q_BUILD_GIT_SHA":commit},"registry_native_version":"0.24.0",
        "images":{"production":{"path":"/subject/hf2q","sha256":"b".repeat(64)}}});
    let calibration = json!({"binary":images["images"]["production"]});
    assert!(validate_subject_identity(&source, &images, &calibration, &commit).is_ok());
    assert!(validate_subject_identity(&source, &images, &calibration, &"c".repeat(40)).is_err());
    images["source_commit"] = json!("c".repeat(40));
    assert!(validate_subject_identity(&source, &images, &calibration, &commit).is_err());
    images["source_commit"] = json!(commit);
    images["images"]["production"]["sha256"] = json!("c".repeat(64));
    assert!(validate_subject_identity(&source, &images, &calibration, &commit).is_err());
    images["images"]["production"]["sha256"] = calibration["binary"]["sha256"].clone();
    images["registry_native_version"] = json!("0.23.1");
    assert!(validate_subject_identity(&source, &images, &calibration, &commit).is_err());
}
