use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::core::provenance::tensor_execution::ArtifactEvidence;
use crate::intelligence::calibration::VerifiedTeacherPredictionPlan;

use super::{
    is_git_commit, is_sha256, open_external_reference_target, open_native_reference_target,
    ExactTeacherExternalReferenceEvidenceV1, ExactTeacherReferenceInputV1,
    ExternalReferenceImplementationV1,
};
use crate::intelligence::exact_teacher::{
    ExactTeacherTargetError, ExactTeacherTargetReceipt, StructurallyVerifiedTeacherTargetArtifact,
};

const COMPARISON_SCHEMA_VERSION: u32 = 1;
const COMPARISON_PROFILE: &str = "qwen35_source_bf16_vs_external_reference_v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExactTeacherReferenceRowComparisonV1 {
    pub(crate) point_ordinal: usize,
    pub(crate) stable_id: String,
    pub(crate) prefix_token_count: usize,
    pub(crate) prefix_token_ids_sha256: String,
    pub(crate) max_abs: f64,
    pub(crate) max_abs_token_id: u32,
    pub(crate) kl_reference_to_native: f64,
    pub(crate) native_argmax_token_id: u32,
    pub(crate) reference_argmax_token_id: u32,
    pub(crate) top1_match: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExactTeacherReferenceAggregateV1 {
    pub(crate) row_count: usize,
    pub(crate) max_abs: f64,
    pub(crate) max_abs_point_ordinal: usize,
    pub(crate) max_abs_token_id: u32,
    pub(crate) mean_kl_reference_to_native: f64,
    pub(crate) max_kl_reference_to_native: f64,
    pub(crate) p50_kl_reference_to_native: f64,
    pub(crate) p95_kl_reference_to_native: f64,
    pub(crate) top1_match_count: usize,
    pub(crate) top1_match_rate: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExactTeacherReferenceTrajectoryComparisonV1 {
    pub(crate) stable_id: String,
    pub(crate) native_token_ids_sha256: String,
    pub(crate) reference_token_ids_sha256: String,
    pub(crate) exact_match: bool,
    pub(crate) first_divergence_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExactTeacherReferenceComparisonReceiptV1 {
    pub(crate) schema_version: u32,
    pub(crate) profile: String,
    pub(crate) comparator_git_commit: String,
    pub(crate) reference_input_sha256: String,
    pub(crate) prediction_plan_sha256: String,
    pub(crate) native_completion_receipt_sha256: String,
    pub(crate) native_target_artifact: ArtifactEvidence,
    pub(crate) external_evidence_sha256: String,
    pub(crate) external_implementation: ExternalReferenceImplementationV1,
    pub(crate) external_target_artifact: ArtifactEvidence,
    pub(crate) rows: Vec<ExactTeacherReferenceRowComparisonV1>,
    pub(crate) aggregate: ExactTeacherReferenceAggregateV1,
    pub(crate) trajectories: Vec<ExactTeacherReferenceTrajectoryComparisonV1>,
    pub(crate) thresholds_predeclared: bool,
    pub(crate) quality_gate_authority: bool,
    pub(crate) source_teacher_authority: bool,
    pub(crate) sensitivity_authority: bool,
    pub(crate) allocator_authority: bool,
    pub(crate) selector_authority: bool,
    pub(crate) autoquant_authority: bool,
    pub(crate) runtime_dependency: bool,
    pub(crate) dwq: bool,
    pub(crate) comparison_receipt_sha256: String,
}

#[derive(Serialize)]
struct ComparisonHashView<'a> {
    schema_version: u32,
    profile: &'a str,
    comparator_git_commit: &'a str,
    reference_input_sha256: &'a str,
    prediction_plan_sha256: &'a str,
    native_completion_receipt_sha256: &'a str,
    native_target_artifact: &'a ArtifactEvidence,
    external_evidence_sha256: &'a str,
    external_implementation: &'a ExternalReferenceImplementationV1,
    external_target_artifact: &'a ArtifactEvidence,
    rows: &'a [ExactTeacherReferenceRowComparisonV1],
    aggregate: &'a ExactTeacherReferenceAggregateV1,
    trajectories: &'a [ExactTeacherReferenceTrajectoryComparisonV1],
    thresholds_predeclared: bool,
    quality_gate_authority: bool,
    source_teacher_authority: bool,
    sensitivity_authority: bool,
    allocator_authority: bool,
    selector_authority: bool,
    autoquant_authority: bool,
    runtime_dependency: bool,
    dwq: bool,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compare_exact_teacher_reference_targets(
    plan: &VerifiedTeacherPredictionPlan,
    input: &ExactTeacherReferenceInputV1,
    native_target_path: &Path,
    native_receipt: ExactTeacherTargetReceipt,
    native_completion_receipt_sha256: String,
    comparator_git_commit: String,
    external_target_path: &Path,
    external_evidence: &ExactTeacherExternalReferenceEvidenceV1,
) -> Result<ExactTeacherReferenceComparisonReceiptV1, ExactTeacherTargetError> {
    if !is_sha256(&native_completion_receipt_sha256) || !is_git_commit(&comparator_git_commit) {
        return Err(ExactTeacherTargetError::Invalid(
            "native completion or comparator identity is invalid".into(),
        ));
    }
    let mut native = open_native_reference_target(native_target_path, input, plan, native_receipt)?;
    let mut external =
        open_external_reference_target(external_target_path, input, plan, external_evidence)?;
    if native.receipt().vocabulary_size != external.receipt().vocabulary_size
        || native.receipt().rows.len() != external.receipt().rows.len()
    {
        return Err(ExactTeacherTargetError::Invalid(
            "native and external target dimensions differ".into(),
        ));
    }

    let native_rows = native.receipt().rows.clone();
    let external_rows = external.receipt().rows.clone();
    let mut rows = Vec::with_capacity(native_rows.len());
    for (native_row, external_row) in native_rows.iter().zip(&external_rows) {
        if native_row.point_ordinal != external_row.point_ordinal
            || native_row.stable_id != external_row.stable_id
            || native_row.prefix_token_count != external_row.prefix_token_count
            || native_row.prefix_token_ids_sha256 != external_row.prefix_token_ids_sha256
        {
            return Err(ExactTeacherTargetError::Invalid(
                "native and external target row identities differ".into(),
            ));
        }
        let native_logits = read_row(&mut native, native_row)?;
        let external_logits = read_row(&mut external, external_row)?;
        let (max_abs, max_abs_token_id) = max_abs(&external_logits, &native_logits)?;
        let kl_reference_to_native = kl_reference_to_native(&external_logits, &native_logits)?;
        rows.push(ExactTeacherReferenceRowComparisonV1 {
            point_ordinal: native_row.point_ordinal,
            stable_id: native_row.stable_id.clone(),
            prefix_token_count: native_row.prefix_token_count,
            prefix_token_ids_sha256: native_row.prefix_token_ids_sha256.clone(),
            max_abs,
            max_abs_token_id,
            kl_reference_to_native,
            native_argmax_token_id: native_row.argmax_token_id,
            reference_argmax_token_id: external_row.argmax_token_id,
            top1_match: native_row.argmax_token_id == external_row.argmax_token_id,
        });
    }
    let aggregate = aggregate(&rows)?;
    let trajectories = native
        .receipt()
        .greedy_trajectories
        .iter()
        .zip(&external.receipt().greedy_trajectories)
        .map(|(native, reference)| {
            if native.stable_id != reference.stable_id
                || native.prompt_token_ids_sha256 != reference.prompt_token_ids_sha256
            {
                return Err(ExactTeacherTargetError::Invalid(
                    "native and external trajectory identities differ".into(),
                ));
            }
            Ok(ExactTeacherReferenceTrajectoryComparisonV1 {
                stable_id: native.stable_id.clone(),
                native_token_ids_sha256: native.token_ids_sha256.clone(),
                reference_token_ids_sha256: reference.token_ids_sha256.clone(),
                exact_match: native.token_ids == reference.token_ids,
                first_divergence_index: native
                    .token_ids
                    .iter()
                    .zip(&reference.token_ids)
                    .position(|(left, right)| left != right),
            })
        })
        .collect::<Result<Vec<_>, ExactTeacherTargetError>>()?;

    let mut receipt = ExactTeacherReferenceComparisonReceiptV1 {
        schema_version: COMPARISON_SCHEMA_VERSION,
        profile: COMPARISON_PROFILE.into(),
        comparator_git_commit,
        reference_input_sha256: input.reference_input_sha256.clone(),
        prediction_plan_sha256: input.prediction_plan.manifest_sha256.clone(),
        native_completion_receipt_sha256,
        native_target_artifact: native.receipt().target_artifact.clone(),
        external_evidence_sha256: external_evidence.evidence_sha256.clone(),
        external_implementation: external_evidence.implementation.clone(),
        external_target_artifact: external.receipt().target_artifact.clone(),
        rows,
        aggregate,
        trajectories,
        thresholds_predeclared: false,
        quality_gate_authority: false,
        source_teacher_authority: false,
        sensitivity_authority: false,
        allocator_authority: false,
        selector_authority: false,
        autoquant_authority: false,
        runtime_dependency: false,
        dwq: false,
        comparison_receipt_sha256: String::new(),
    };
    receipt.comparison_receipt_sha256 = comparison_sha256(&receipt)?;
    Ok(receipt)
}

/// Revalidate a serialized comparison receipt before it can participate in a
/// later threshold decision. This deliberately proves only canonical,
/// non-authoritative comparison evidence; it never upgrades the raw receipt.
pub(crate) fn validate_exact_teacher_reference_comparison_receipt(
    receipt: &ExactTeacherReferenceComparisonReceiptV1,
) -> Result<(), ExactTeacherTargetError> {
    let raw_flags_are_false = !receipt.thresholds_predeclared
        && !receipt.quality_gate_authority
        && !receipt.source_teacher_authority
        && !receipt.sensitivity_authority
        && !receipt.allocator_authority
        && !receipt.selector_authority
        && !receipt.autoquant_authority
        && !receipt.runtime_dependency
        && !receipt.dwq;
    let artifact_valid = |artifact: &ArtifactEvidence| {
        artifact.artifact_id == "exact_teacher_logits"
            && artifact.role == "structural_full_vocabulary_f32_target_rows"
            && artifact.byte_len > 0
            && is_sha256(&artifact.sha256)
    };
    let rows_are_canonical = !receipt.rows.is_empty()
        && receipt
            .rows
            .windows(2)
            .all(|rows| rows[0].point_ordinal < rows[1].point_ordinal)
        && receipt.rows.iter().all(|row| {
            !row.stable_id.is_empty()
                && row.prefix_token_count > 0
                && is_sha256(&row.prefix_token_ids_sha256)
                && row.max_abs.is_finite()
                && row.max_abs >= 0.0
                && row.kl_reference_to_native.is_finite()
                && row.kl_reference_to_native >= 0.0
                && row.top1_match == (row.native_argmax_token_id == row.reference_argmax_token_id)
        });
    let trajectories_are_canonical = receipt.trajectories.iter().all(|trajectory| {
        !trajectory.stable_id.is_empty()
            && is_sha256(&trajectory.native_token_ids_sha256)
            && is_sha256(&trajectory.reference_token_ids_sha256)
            && trajectory.exact_match == trajectory.first_divergence_index.is_none()
    });
    if receipt.schema_version != COMPARISON_SCHEMA_VERSION
        || receipt.profile != COMPARISON_PROFILE
        || !is_git_commit(&receipt.comparator_git_commit)
        || !is_sha256(&receipt.reference_input_sha256)
        || !is_sha256(&receipt.prediction_plan_sha256)
        || !is_sha256(&receipt.native_completion_receipt_sha256)
        || !is_sha256(&receipt.external_evidence_sha256)
        || !is_sha256(&receipt.external_implementation.producer_sha256)
        || !is_git_commit(&receipt.external_implementation.repository_commit)
        || !is_sha256(&receipt.external_implementation.dependency_lock_sha256)
    {
        return Err(ExactTeacherTargetError::Invalid(
            "exact-teacher comparison identity is invalid".into(),
        ));
    }
    if !artifact_valid(&receipt.native_target_artifact)
        || !artifact_valid(&receipt.external_target_artifact)
    {
        return Err(ExactTeacherTargetError::Invalid(
            "exact-teacher comparison artifact identity is invalid".into(),
        ));
    }
    if !rows_are_canonical || !trajectories_are_canonical {
        return Err(ExactTeacherTargetError::Invalid(
            "exact-teacher comparison rows or trajectories are invalid".into(),
        ));
    }
    if !raw_flags_are_false {
        return Err(ExactTeacherTargetError::Invalid(
            "raw exact-teacher comparison cannot grant authority".into(),
        ));
    }
    if !aggregate_equivalent(&aggregate(&receipt.rows)?, &receipt.aggregate) {
        return Err(ExactTeacherTargetError::Invalid(
            "exact-teacher comparison aggregate does not reproduce".into(),
        ));
    }
    if !is_sha256(&receipt.comparison_receipt_sha256) {
        return Err(ExactTeacherTargetError::Invalid(
            "exact-teacher comparison self-hash identity is invalid".into(),
        ));
    }
    Ok(())
}

/// Revalidate a comparison nested in a newly minted quality receipt.
///
/// Unlike imported historical artifacts, newly serialized comparisons use
/// serde_json's canonical float spelling, so their semantic hash view must
/// reproduce exactly after a parse/serialize round trip.
pub(crate) fn validate_canonical_exact_teacher_reference_comparison_receipt(
    receipt: &ExactTeacherReferenceComparisonReceiptV1,
) -> Result<(), ExactTeacherTargetError> {
    validate_exact_teacher_reference_comparison_receipt(receipt)?;
    if comparison_sha256(receipt)? != receipt.comparison_receipt_sha256 {
        return Err(ExactTeacherTargetError::Invalid(
            "canonical exact-teacher comparison self-hash differs".into(),
        ));
    }
    Ok(())
}

/// Authenticate the self-hash over exact recorded JSON bytes before parsing
/// floating-point metrics. Equivalent floats can reserialize with a shorter
/// spelling, so parse-and-reserialize is not a stable artifact check.
pub(crate) fn validate_exact_teacher_reference_comparison_artifact(
    bytes: &[u8],
) -> Result<ExactTeacherReferenceComparisonReceiptV1, ExactTeacherTargetError> {
    const MAX_COMPARISON_BYTES: usize = 16 * 1024 * 1024;
    const MARKER: &[u8] = b",\"comparison_receipt_sha256\":\"";
    if bytes.is_empty() || bytes.len() > MAX_COMPARISON_BYTES {
        return Err(ExactTeacherTargetError::Invalid(
            "exact-teacher comparison artifact is empty or too large".into(),
        ));
    }
    let serialized = bytes.strip_suffix(b"\n").unwrap_or(bytes);
    let marker_offset = serialized
        .windows(MARKER.len())
        .rposition(|window| window == MARKER)
        .ok_or_else(|| {
            ExactTeacherTargetError::Invalid(
                "exact-teacher comparison artifact lacks its terminal self-hash".into(),
            )
        })?;
    let receipt: ExactTeacherReferenceComparisonReceiptV1 = serde_json::from_slice(serialized)
        .map_err(|error| ExactTeacherTargetError::Serialization(error.to_string()))?;
    validate_exact_teacher_reference_comparison_receipt(&receipt)?;
    let expected_suffix = format!(
        ",\"comparison_receipt_sha256\":\"{}\"}}",
        receipt.comparison_receipt_sha256
    );
    if &serialized[marker_offset..] != expected_suffix.as_bytes() {
        return Err(ExactTeacherTargetError::Invalid(
            "exact-teacher comparison self-hash is not its terminal field".into(),
        ));
    }
    let mut hash_view = serialized[..marker_offset].to_vec();
    hash_view.push(b'}');
    if hex::encode(Sha256::digest(hash_view)) != receipt.comparison_receipt_sha256 {
        return Err(ExactTeacherTargetError::Invalid(
            "exact-teacher comparison artifact self-hash differs".into(),
        ));
    }
    Ok(receipt)
}

fn aggregate_equivalent(
    reproduced: &ExactTeacherReferenceAggregateV1,
    recorded: &ExactTeacherReferenceAggregateV1,
) -> bool {
    const TOLERANCE: f64 = 1.0e-12;
    reproduced.row_count == recorded.row_count
        && reproduced.max_abs_point_ordinal == recorded.max_abs_point_ordinal
        && reproduced.max_abs_token_id == recorded.max_abs_token_id
        && reproduced.top1_match_count == recorded.top1_match_count
        && (reproduced.max_abs - recorded.max_abs).abs() <= TOLERANCE
        && (reproduced.mean_kl_reference_to_native - recorded.mean_kl_reference_to_native).abs()
            <= TOLERANCE
        && (reproduced.max_kl_reference_to_native - recorded.max_kl_reference_to_native).abs()
            <= TOLERANCE
        && (reproduced.p50_kl_reference_to_native - recorded.p50_kl_reference_to_native).abs()
            <= TOLERANCE
        && (reproduced.p95_kl_reference_to_native - recorded.p95_kl_reference_to_native).abs()
            <= TOLERANCE
        && (reproduced.top1_match_rate - recorded.top1_match_rate).abs() <= TOLERANCE
}

fn read_row(
    artifact: &mut StructurallyVerifiedTeacherTargetArtifact,
    row: &crate::intelligence::exact_teacher::TeacherTargetRowReceipt,
) -> Result<Vec<f32>, ExactTeacherTargetError> {
    let path = artifact.path().to_owned();
    let file = artifact.retained_file_mut();
    file.seek(SeekFrom::Start(row.payload_offset))
        .map_err(|error| ExactTeacherTargetError::io(&path, error))?;
    let payload_len = usize::try_from(row.payload_bytes)
        .map_err(|_| ExactTeacherTargetError::Invalid("target row is too large".into()))?;
    let mut payload = vec![0_u8; payload_len];
    file.read_exact(&mut payload)
        .map_err(|error| ExactTeacherTargetError::io(&path, error))?;
    Ok(payload
        .chunks_exact(4)
        .map(|bytes| f32::from_bits(u32::from_le_bytes(bytes.try_into().unwrap())))
        .collect())
}

fn max_abs(reference: &[f32], native: &[f32]) -> Result<(f64, u32), ExactTeacherTargetError> {
    if reference.len() != native.len() || reference.is_empty() {
        return Err(ExactTeacherTargetError::Invalid(
            "reference row length differs from native".into(),
        ));
    }
    let mut maximum = 0.0_f64;
    let mut token_id = 0usize;
    for (index, (reference, native)) in reference.iter().zip(native).enumerate() {
        let delta = (f64::from(*reference) - f64::from(*native)).abs();
        if delta > maximum {
            maximum = delta;
            token_id = index;
        }
    }
    Ok((
        maximum,
        u32::try_from(token_id)
            .map_err(|_| ExactTeacherTargetError::Invalid("token id overflow".into()))?,
    ))
}

fn kl_reference_to_native(
    reference: &[f32],
    native: &[f32],
) -> Result<f64, ExactTeacherTargetError> {
    if reference.len() != native.len()
        || reference.is_empty()
        || reference
            .iter()
            .chain(native)
            .any(|value| !value.is_finite())
    {
        return Err(ExactTeacherTargetError::Invalid(
            "KL input rows are empty, non-finite, or mismatched".into(),
        ));
    }
    let reference_lse = logsumexp(reference);
    let native_lse = logsumexp(native);
    let mut kl = 0.0_f64;
    for (reference, native) in reference.iter().zip(native) {
        let log_p = f64::from(*reference) - reference_lse;
        let log_q = f64::from(*native) - native_lse;
        kl += log_p.exp() * (log_p - log_q);
    }
    if !kl.is_finite() {
        return Err(ExactTeacherTargetError::Invalid(
            "KL result is non-finite".into(),
        ));
    }
    Ok(kl.max(0.0))
}

fn logsumexp(logits: &[f32]) -> f64 {
    let maximum = logits
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    maximum
        + logits
            .iter()
            .map(|value| (f64::from(*value) - maximum).exp())
            .sum::<f64>()
            .ln()
}

fn aggregate(
    rows: &[ExactTeacherReferenceRowComparisonV1],
) -> Result<ExactTeacherReferenceAggregateV1, ExactTeacherTargetError> {
    if rows.is_empty() {
        return Err(ExactTeacherTargetError::Invalid(
            "reference comparison contains no rows".into(),
        ));
    }
    let max_abs = rows
        .iter()
        .max_by(|left, right| left.max_abs.total_cmp(&right.max_abs))
        .unwrap();
    let mut kl = rows
        .iter()
        .map(|row| row.kl_reference_to_native)
        .collect::<Vec<_>>();
    kl.sort_by(f64::total_cmp);
    let sum = kl.iter().sum::<f64>();
    let top1_match_count = rows.iter().filter(|row| row.top1_match).count();
    Ok(ExactTeacherReferenceAggregateV1 {
        row_count: rows.len(),
        max_abs: max_abs.max_abs,
        max_abs_point_ordinal: max_abs.point_ordinal,
        max_abs_token_id: max_abs.max_abs_token_id,
        mean_kl_reference_to_native: sum / rows.len() as f64,
        max_kl_reference_to_native: *kl.last().unwrap(),
        p50_kl_reference_to_native: nearest_rank(&kl, 50),
        p95_kl_reference_to_native: nearest_rank(&kl, 95),
        top1_match_count,
        top1_match_rate: top1_match_count as f64 / rows.len() as f64,
    })
}

fn nearest_rank(sorted: &[f64], percentile: usize) -> f64 {
    let rank = sorted
        .len()
        .checked_mul(percentile)
        .and_then(|value| value.checked_add(99))
        .map(|value| value / 100)
        .unwrap_or(sorted.len());
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn comparison_sha256(
    receipt: &ExactTeacherReferenceComparisonReceiptV1,
) -> Result<String, ExactTeacherTargetError> {
    let view = ComparisonHashView {
        schema_version: receipt.schema_version,
        profile: &receipt.profile,
        comparator_git_commit: &receipt.comparator_git_commit,
        reference_input_sha256: &receipt.reference_input_sha256,
        prediction_plan_sha256: &receipt.prediction_plan_sha256,
        native_completion_receipt_sha256: &receipt.native_completion_receipt_sha256,
        native_target_artifact: &receipt.native_target_artifact,
        external_evidence_sha256: &receipt.external_evidence_sha256,
        external_implementation: &receipt.external_implementation,
        external_target_artifact: &receipt.external_target_artifact,
        rows: &receipt.rows,
        aggregate: &receipt.aggregate,
        trajectories: &receipt.trajectories,
        thresholds_predeclared: receipt.thresholds_predeclared,
        quality_gate_authority: receipt.quality_gate_authority,
        source_teacher_authority: receipt.source_teacher_authority,
        sensitivity_authority: receipt.sensitivity_authority,
        allocator_authority: receipt.allocator_authority,
        selector_authority: receipt.selector_authority,
        autoquant_authority: receipt.autoquant_authority,
        runtime_dependency: receipt.runtime_dependency,
        dwq: receipt.dwq,
    };
    serde_json::to_vec(&view)
        .map(|bytes| hex::encode(Sha256::digest(bytes)))
        .map_err(|error| ExactTeacherTargetError::Serialization(error.to_string()))
}
