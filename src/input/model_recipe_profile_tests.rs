use serde_json::Value;
use sha2::Digest;

use super::model_recipe::{
    ModelPreparationReceiptV2, PreparedModelProfileV1, SourceRetentionChoice,
    MAX_PREPARED_MODEL_PROFILE_BYTES,
};

const PREPARATION_BYTES: &[u8] =
    include_bytes!("../../data/model-recipes/qwen38-27b-preparation-receipt-v2.json");
const PROFILE_BYTES: &[u8] =
    include_bytes!("../../data/model-recipes/qwen38-27b-prepared-profile-v1.json");

#[test]
fn prepared_profile_golden_is_exact_and_cross_binds_the_pair_receipt() {
    let receipt = ModelPreparationReceiptV2::parse(PREPARATION_BYTES).unwrap();
    let profile = PreparedModelProfileV1::build_keep(&receipt, PREPARATION_BYTES).unwrap();
    assert_eq!(profile.to_deterministic_json().unwrap(), PROFILE_BYTES);
    assert_eq!(
        PreparedModelProfileV1::parse(PROFILE_BYTES).unwrap(),
        profile
    );
    assert_eq!(profile.repository_id(), "Qwen/Qwen3.8-27B");
    assert_eq!(
        profile.revision(),
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    );
    assert_eq!(profile.recipe_id(), "qwen38-27b-official-v1");
    assert_eq!(profile.source_retention(), SourceRetentionChoice::Keep);
    assert_eq!(PROFILE_BYTES.len(), 1_240);
    assert_eq!(
        hex::encode(sha2::Sha256::digest(PROFILE_BYTES)),
        "bc34868c014b2cd7de650bb2185c79d8a4389f71f68d7611ca6dd6de6c6dd9b0"
    );
    assert_eq!(
        profile.preparation_receipt_sha256(),
        "b947550f429b7035a97b75a01dc430ba13ff0e83cfbb561d1dab89e926649f36"
    );
    assert_eq!(
        profile
            .verify_preparation_receipt(PREPARATION_BYTES)
            .unwrap(),
        receipt
    );
}

#[test]
fn parser_rejects_size_duplicate_unknown_trailing_and_noncanonical_bytes() {
    let oversized = vec![b' '; MAX_PREPARED_MODEL_PROFILE_BYTES + 1];
    assert!(PreparedModelProfileV1::parse(&oversized).is_err());

    let text = std::str::from_utf8(PROFILE_BYTES).unwrap();
    let duplicate = text.replacen(
        "{\"kind\":\"hf2q.prepared-model-profile\",",
        "{\"kind\":\"hf2q.prepared-model-profile\",\"kind\":\"hf2q.prepared-model-profile\",",
        1,
    );
    assert!(PreparedModelProfileV1::parse(duplicate.as_bytes()).is_err());

    let unknown = text.replacen("{\"kind\":", "{\"unknown\":true,\"kind\":", 1);
    assert!(PreparedModelProfileV1::parse(unknown.as_bytes()).is_err());

    let mut trailing = PROFILE_BYTES.to_vec();
    trailing.extend_from_slice(b"{}");
    assert!(PreparedModelProfileV1::parse(&trailing).is_err());

    let value: Value = serde_json::from_slice(PROFILE_BYTES).unwrap();
    let mut pretty = serde_json::to_vec_pretty(&value).unwrap();
    pretty.push(b'\n');
    assert!(PreparedModelProfileV1::parse(&pretty).is_err());

    let deep = format!("{{\"deep\":{} }}", "[".repeat(65) + "0" + &"]".repeat(65));
    assert!(PreparedModelProfileV1::parse(deep.as_bytes()).is_err());
}

#[test]
fn parser_rejects_identity_retention_state_and_descriptor_mutations() {
    let cases: Vec<Box<dyn Fn(&mut Value)>> = vec![
        Box::new(|value| value["kind"] = Value::String("other".into())),
        Box::new(|value| value["schema_version"] = Value::from(2)),
        Box::new(|value| value["package"] = Value::String("other".into())),
        Box::new(|value| value["repository"]["id"] = Value::String("other/model".into())),
        Box::new(|value| value["recipe"]["sha256"] = Value::String("a".repeat(64))),
        Box::new(|value| value["preparation_receipt"]["schema_version"] = Value::from(1)),
        Box::new(|value| value["source_retention"] = Value::String("delete".into())),
        Box::new(|value| value["state"] = Value::String("ready".into())),
        Box::new(|value| {
            value["artifacts"]
                .as_array_mut()
                .unwrap()
                .pop()
                .map(|_| ())
                .unwrap()
        }),
        Box::new(|value| {
            value["artifacts"][0]["path"] = Value::String("artifacts/other.gguf".into())
        }),
        Box::new(|value| {
            value["artifacts"][0]["conversion_receipt_sha256"] = Value::String("A".repeat(64))
        }),
    ];
    for mutate in cases {
        let mut value: Value = serde_json::from_slice(PROFILE_BYTES).unwrap();
        mutate(&mut value);
        let mut bytes = serde_json::to_vec(&value).unwrap();
        bytes.push(b'\n');
        assert!(PreparedModelProfileV1::parse(&bytes).is_err());
    }
}

#[test]
fn profile_rejects_mutated_or_cross_pair_receipt_bytes() {
    let profile = PreparedModelProfileV1::parse(PROFILE_BYTES).unwrap();
    let mut value: Value = serde_json::from_slice(PREPARATION_BYTES).unwrap();
    value["artifacts"][0]["conversion_receipt_sha256"] = Value::String("b".repeat(64));
    let mut bytes = serde_json::to_vec(&value).unwrap();
    bytes.push(b'\n');
    assert!(ModelPreparationReceiptV2::parse(&bytes).is_ok());
    assert!(profile.verify_preparation_receipt(&bytes).is_err());
}
