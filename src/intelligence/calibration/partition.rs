use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use super::manifest::ordered_evidence_json_sha256;
use super::render::validate_rendered_dataset;
use super::types::*;

#[derive(Serialize)]
struct OverlapReceiptHashView {
    source_record_overlap_count: usize,
    raw_overlap_count: usize,
    rendered_overlap_count: usize,
    token_window_overlap_count: usize,
    compared_example_count: usize,
}

#[derive(Serialize)]
struct PartitionManifestHashView<'a> {
    schema_version: u32,
    calibration_manifest_sha256: &'a str,
    policy_validation_manifest_sha256: &'a str,
    acceptance_holdout_manifest_sha256: &'a str,
    overlap_policy: &'a OverlapPolicy,
    overlap_receipt: &'a DatasetOverlapReceipt,
}

fn assert_common_render_identity(
    expected: &RenderedDatasetManifest,
    actual: &RenderedDatasetManifest,
) -> Result<(), CalibrationInputError> {
    if actual.source != expected.source
        || actual.verified_source_manifest_sha256 != expected.verified_source_manifest_sha256
        || actual.chat_template_source != expected.chat_template_source
        || actual.chat_template_sha256 != expected.chat_template_sha256
        || actual.tokenizer_json_sha256 != expected.tokenizer_json_sha256
        || actual.renderer_revision != expected.renderer_revision
        || actual.max_tokens_per_example != expected.max_tokens_per_example
        || actual.token_window_size != expected.token_window_size
    {
        return Err(CalibrationInputError::SplitMismatch);
    }
    Ok(())
}

fn count_cross_split_overlap<'a>(sets: impl IntoIterator<Item = &'a BTreeSet<String>>) -> usize {
    let mut first_split = BTreeMap::<&str, usize>::new();
    let mut overlap = BTreeSet::new();
    for (split_index, set) in sets.into_iter().enumerate() {
        for value in set {
            match first_split.get(value.as_str()) {
                Some(previous) if *previous != split_index => {
                    overlap.insert(value.as_str());
                }
                None => {
                    first_split.insert(value.as_str(), split_index);
                }
                _ => {}
            }
        }
    }
    overlap.len()
}

type ReceiptSets = (
    BTreeSet<String>,
    BTreeSet<String>,
    BTreeSet<String>,
    BTreeSet<String>,
);

fn receipt_sets(dataset: &RenderedDataset) -> ReceiptSets {
    let records = dataset
        .manifest
        .examples
        .iter()
        .map(|example| example.source_record_sha256.clone())
        .collect();
    let raw = dataset
        .manifest
        .examples
        .iter()
        .map(|example| example.raw_example_sha256.clone())
        .collect();
    let rendered = dataset
        .manifest
        .examples
        .iter()
        .map(|example| example.rendered_utf8_sha256.clone())
        .collect();
    let windows = dataset
        .manifest
        .examples
        .iter()
        .flat_map(|example| example.token_window_sha256.iter().cloned())
        .collect();
    (records, raw, rendered, windows)
}

/// Verify that selector calibration, policy validation, and final acceptance
/// are exact, source-compatible, and pairwise disjoint by source record,
/// content, rendered bytes, and fixed-width token windows.
pub fn verify_dataset_partition(
    calibration: &RenderedDataset,
    policy_validation: &RenderedDataset,
    acceptance_holdout: &RenderedDataset,
) -> Result<DatasetPartitionManifest, CalibrationInputError> {
    for dataset in [calibration, policy_validation, acceptance_holdout] {
        validate_rendered_dataset(dataset)?;
    }
    if calibration.manifest.split != DatasetSplit::Calibration
        || policy_validation.manifest.split != DatasetSplit::PolicyValidation
        || acceptance_holdout.manifest.split != DatasetSplit::AcceptanceHoldout
    {
        return Err(CalibrationInputError::SplitMismatch);
    }
    assert_common_render_identity(&calibration.manifest, &policy_validation.manifest)?;
    assert_common_render_identity(&calibration.manifest, &acceptance_holdout.manifest)?;

    let calibration_sets = receipt_sets(calibration);
    let validation_sets = receipt_sets(policy_validation);
    let holdout_sets = receipt_sets(acceptance_holdout);
    let source_record_overlap_count =
        count_cross_split_overlap([&calibration_sets.0, &validation_sets.0, &holdout_sets.0]);
    let raw_overlap_count =
        count_cross_split_overlap([&calibration_sets.1, &validation_sets.1, &holdout_sets.1]);
    let rendered_overlap_count =
        count_cross_split_overlap([&calibration_sets.2, &validation_sets.2, &holdout_sets.2]);
    let token_window_overlap_count =
        count_cross_split_overlap([&calibration_sets.3, &validation_sets.3, &holdout_sets.3]);
    if source_record_overlap_count != 0
        || raw_overlap_count != 0
        || rendered_overlap_count != 0
        || token_window_overlap_count != 0
    {
        return Err(CalibrationInputError::DatasetOverlap {
            source_records: source_record_overlap_count,
            raw: raw_overlap_count,
            rendered: rendered_overlap_count,
            token_windows: token_window_overlap_count,
        });
    }

    let compared_example_count = calibration
        .manifest
        .examples
        .len()
        .checked_add(policy_validation.manifest.examples.len())
        .and_then(|value| value.checked_add(acceptance_holdout.manifest.examples.len()))
        .ok_or_else(|| {
            CalibrationInputError::InvalidDataset("compared-example count overflow".into())
        })?;
    let mut overlap_receipt = DatasetOverlapReceipt {
        source_record_overlap_count,
        raw_overlap_count,
        rendered_overlap_count,
        token_window_overlap_count,
        compared_example_count,
        receipt_sha256: String::new(),
    };
    overlap_receipt.receipt_sha256 = ordered_evidence_json_sha256(&OverlapReceiptHashView {
        source_record_overlap_count,
        raw_overlap_count,
        rendered_overlap_count,
        token_window_overlap_count,
        compared_example_count,
    })?;

    let mut manifest = DatasetPartitionManifest {
        schema_version: CALIBRATION_INPUT_SCHEMA_VERSION,
        calibration_manifest_sha256: calibration.manifest.manifest_sha256.clone(),
        policy_validation_manifest_sha256: policy_validation.manifest.manifest_sha256.clone(),
        acceptance_holdout_manifest_sha256: acceptance_holdout.manifest.manifest_sha256.clone(),
        overlap_policy: OverlapPolicy::RejectSourceRecordRawRenderedOrTokenWindow,
        overlap_receipt,
        manifest_sha256: String::new(),
    };
    manifest.manifest_sha256 = ordered_evidence_json_sha256(&PartitionManifestHashView {
        schema_version: manifest.schema_version,
        calibration_manifest_sha256: &manifest.calibration_manifest_sha256,
        policy_validation_manifest_sha256: &manifest.policy_validation_manifest_sha256,
        acceptance_holdout_manifest_sha256: &manifest.acceptance_holdout_manifest_sha256,
        overlap_policy: &manifest.overlap_policy,
        overlap_receipt: &manifest.overlap_receipt,
    })?;
    Ok(manifest)
}
