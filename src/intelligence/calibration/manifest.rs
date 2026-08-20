use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::types::*;

pub(super) fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Hash JSON with hf2q's evidence ordering. Struct field order, message/tool
/// array order, and nested JSON object insertion order are significant because
/// the production renderer exposes those bytes/orders to model templates.
pub(super) fn ordered_evidence_json_sha256<T: Serialize>(
    value: &T,
) -> Result<String, CalibrationInputError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| CalibrationInputError::Serialization(error.to_string()))?;
    Ok(sha256_bytes(&bytes))
}

#[derive(Serialize)]
struct StructuredManifestHashView<'a> {
    schema_version: u32,
    dataset_id: &'a str,
    revision: &'a str,
    license: &'a str,
    split: DatasetSplit,
    seed: u64,
    example_order: &'a [String],
    examples: &'a [StructuredExample],
    source_record_sha256: &'a BTreeMap<String, String>,
    raw_example_sha256: &'a BTreeMap<String, String>,
}

/// Content-only view used for cross-split leakage detection. Dataset ids,
/// record ids, and stable ids must not let a duplicated example evade the raw
/// overlap gate.
#[derive(Serialize)]
struct RawExampleHashView<'a> {
    messages: &'a [crate::serve::api::schema::ChatMessage],
    tools: &'a [crate::serve::api::schema::Tool],
}

fn raw_example_sha256(example: &StructuredExample) -> Result<String, CalibrationInputError> {
    ordered_evidence_json_sha256(&RawExampleHashView {
        messages: &example.messages,
        tools: &example.tools,
    })
}

fn source_record_sha256(provenance: &ExampleProvenance) -> Result<String, CalibrationInputError> {
    ordered_evidence_json_sha256(&(
        provenance.dataset_id.as_str(),
        provenance.revision.as_str(),
        provenance.record_id.as_str(),
    ))
}

pub fn build_structured_dataset_manifest(
    dataset_id: String,
    revision: String,
    license: String,
    split: DatasetSplit,
    seed: u64,
    mut examples: Vec<StructuredExample>,
) -> Result<StructuredDatasetManifest, CalibrationInputError> {
    if dataset_id.is_empty() || revision.is_empty() || license.is_empty() || examples.is_empty() {
        return Err(CalibrationInputError::InvalidDataset(
            "dataset identity, license, and examples must be non-empty".into(),
        ));
    }
    for example in &mut examples {
        example.domains.sort();
        if example.domains.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(CalibrationInputError::InvalidDataset(format!(
                "example {} contains duplicate domains",
                example.stable_id
            )));
        }
    }
    let mut ids = BTreeSet::new();
    let mut source_record_hashes = BTreeMap::new();
    let mut raw_hashes = BTreeMap::new();
    for example in &examples {
        if example.stable_id.is_empty()
            || example.messages.is_empty()
            || example.domains.is_empty()
            || example.provenance.dataset_id != dataset_id
            || example.provenance.revision != revision
            || example.provenance.license != license
            || example.provenance.record_id.is_empty()
            || !ids.insert(example.stable_id.clone())
        {
            return Err(CalibrationInputError::InvalidDataset(format!(
                "example identity/provenance must match its dataset and every example needs messages/domains: {}",
                example.stable_id
            )));
        }
        if example.domains.iter().any(|domain| domain.is_empty()) {
            return Err(CalibrationInputError::InvalidDataset(format!(
                "example {} contains an empty domain",
                example.stable_id
            )));
        }
        source_record_hashes.insert(
            example.stable_id.clone(),
            source_record_sha256(&example.provenance)?,
        );
        raw_hashes.insert(example.stable_id.clone(), raw_example_sha256(example)?);
    }
    let example_order = examples
        .iter()
        .map(|example| example.stable_id.clone())
        .collect();
    let mut manifest = StructuredDatasetManifest {
        schema_version: CALIBRATION_INPUT_SCHEMA_VERSION,
        dataset_id,
        revision,
        license,
        split,
        seed,
        example_order,
        examples,
        source_record_sha256: source_record_hashes,
        raw_example_sha256: raw_hashes,
        manifest_sha256: String::new(),
    };
    manifest.manifest_sha256 = structured_manifest_sha256(&manifest)?;
    Ok(manifest)
}

fn structured_manifest_sha256(
    manifest: &StructuredDatasetManifest,
) -> Result<String, CalibrationInputError> {
    ordered_evidence_json_sha256(&StructuredManifestHashView {
        schema_version: manifest.schema_version,
        dataset_id: &manifest.dataset_id,
        revision: &manifest.revision,
        license: &manifest.license,
        split: manifest.split,
        seed: manifest.seed,
        example_order: &manifest.example_order,
        examples: &manifest.examples,
        source_record_sha256: &manifest.source_record_sha256,
        raw_example_sha256: &manifest.raw_example_sha256,
    })
}

pub fn validate_structured_dataset_manifest(
    manifest: &StructuredDatasetManifest,
) -> Result<(), CalibrationInputError> {
    if manifest.schema_version != CALIBRATION_INPUT_SCHEMA_VERSION
        || manifest.dataset_id.is_empty()
        || manifest.revision.is_empty()
        || manifest.license.is_empty()
        || manifest.examples.is_empty()
        || manifest.example_order.len() != manifest.examples.len()
        || manifest.source_record_sha256.len() != manifest.examples.len()
        || manifest.raw_example_sha256.len() != manifest.examples.len()
        || manifest
            .example_order
            .iter()
            .zip(&manifest.examples)
            .any(|(id, example)| id != &example.stable_id)
    {
        return Err(CalibrationInputError::InvalidDataset(
            "schema or exact example order is invalid".into(),
        ));
    }
    let mut ids = BTreeSet::new();
    for example in &manifest.examples {
        if example.stable_id.is_empty()
            || !ids.insert(example.stable_id.clone())
            || example.messages.is_empty()
            || example.domains.is_empty()
            || example.domains.iter().any(String::is_empty)
            || example.domains.windows(2).any(|pair| pair[0] >= pair[1])
            || example.provenance.dataset_id != manifest.dataset_id
            || example.provenance.revision != manifest.revision
            || example.provenance.license != manifest.license
            || example.provenance.record_id.is_empty()
        {
            return Err(CalibrationInputError::InvalidDataset(format!(
                "invalid example identity/provenance for {}",
                example.stable_id
            )));
        }
        let expected_raw = raw_example_sha256(example)?;
        let expected_record = source_record_sha256(&example.provenance)?;
        if manifest.raw_example_sha256.get(&example.stable_id) != Some(&expected_raw)
            || manifest.source_record_sha256.get(&example.stable_id) != Some(&expected_record)
        {
            return Err(CalibrationInputError::InvalidDataset(format!(
                "raw hash mismatch for {}",
                example.stable_id
            )));
        }
    }
    if structured_manifest_sha256(manifest)? != manifest.manifest_sha256 {
        return Err(CalibrationInputError::InvalidDataset(
            "manifest hash mismatch".into(),
        ));
    }
    Ok(())
}
