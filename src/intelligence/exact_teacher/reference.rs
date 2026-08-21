//! Non-authoritative matched-reference artifacts and comparisons.
//!
//! Exact token ids are exported from the opaque prediction plan so an external
//! validation program never re-renders prompts. Reopening either target proves
//! only bounded bytes and plan closure; it cannot recreate source-teacher,
//! sensitivity, allocator, selector, or auto-quant authority.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::core::provenance::tensor_execution::ArtifactEvidence;
use crate::intelligence::calibration::{
    validate_teacher_prediction_plan, RenderMode, TeacherPredictionPlanManifest,
    VerifiedTeacherPredictionPlan,
};

use super::target_artifact::{
    open_receipted_structural_teacher_target, reconstruct_structural_teacher_reference_target,
};
use super::{
    ExactTeacherTargetError, ExactTeacherTargetReceipt, TeacherGreedyTrajectoryReceipt,
    TeacherTargetArtifactLimits,
};

mod compare;

pub(crate) use compare::{
    compare_exact_teacher_reference_targets, ExactTeacherReferenceComparisonReceiptV1,
};

const REFERENCE_INPUT_SCHEMA_VERSION: u32 = 1;
const REFERENCE_INPUT_PROFILE: &str = "exact_teacher_reference_input_v1";
const EXTERNAL_EVIDENCE_SCHEMA_VERSION: u32 = 1;
const EXTERNAL_EVIDENCE_PROFILE: &str = "external_exact_teacher_reference_target_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExactTeacherReferenceExampleV1 {
    pub(crate) stable_id: String,
    pub(crate) render_mode: RenderMode,
    pub(crate) token_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExactTeacherReferenceInputV1 {
    pub(crate) schema_version: u32,
    pub(crate) profile: String,
    pub(crate) prediction_plan: TeacherPredictionPlanManifest,
    pub(crate) vocabulary_size: usize,
    pub(crate) target_limits: TeacherTargetArtifactLimits,
    pub(crate) examples: Vec<ExactTeacherReferenceExampleV1>,
    pub(crate) source_teacher_authority: bool,
    pub(crate) sensitivity_authority: bool,
    pub(crate) allocator_authority: bool,
    pub(crate) selector_authority: bool,
    pub(crate) autoquant_authority: bool,
    pub(crate) runtime_dependency: bool,
    pub(crate) reference_input_sha256: String,
}

#[derive(Serialize)]
struct ReferenceInputHashView<'a> {
    schema_version: u32,
    profile: &'a str,
    prediction_plan: &'a TeacherPredictionPlanManifest,
    vocabulary_size: usize,
    target_limits: TeacherTargetArtifactLimits,
    examples: &'a [ExactTeacherReferenceExampleV1],
    source_teacher_authority: bool,
    sensitivity_authority: bool,
    allocator_authority: bool,
    selector_authority: bool,
    autoquant_authority: bool,
    runtime_dependency: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExternalReferenceImplementationV1 {
    pub(crate) name: String,
    pub(crate) producer_sha256: String,
    pub(crate) repository_url: String,
    pub(crate) repository_commit: String,
    pub(crate) package_version: String,
    pub(crate) dependency_lock_sha256: String,
    pub(crate) python_version: String,
    pub(crate) framework_version: String,
    pub(crate) device: String,
    pub(crate) source_dtype: String,
    pub(crate) logit_dtype: String,
    pub(crate) attention_implementation: String,
    pub(crate) cache_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExactTeacherExternalReferenceEvidenceV1 {
    pub(crate) schema_version: u32,
    pub(crate) profile: String,
    pub(crate) reference_input_sha256: String,
    pub(crate) prediction_plan_sha256: String,
    pub(crate) target_artifact: ArtifactEvidence,
    pub(crate) greedy_trajectories: Vec<TeacherGreedyTrajectoryReceipt>,
    pub(crate) implementation: ExternalReferenceImplementationV1,
    pub(crate) external_reference: bool,
    pub(crate) runtime_dependency: bool,
    pub(crate) source_teacher_authority: bool,
    pub(crate) sensitivity_authority: bool,
    pub(crate) allocator_authority: bool,
    pub(crate) selector_authority: bool,
    pub(crate) autoquant_authority: bool,
    pub(crate) dwq: bool,
    pub(crate) evidence_sha256: String,
}

#[derive(Serialize)]
struct ExternalEvidenceHashView<'a> {
    schema_version: u32,
    profile: &'a str,
    reference_input_sha256: &'a str,
    prediction_plan_sha256: &'a str,
    target_artifact: &'a ArtifactEvidence,
    greedy_trajectories: &'a [TeacherGreedyTrajectoryReceipt],
    implementation: &'a ExternalReferenceImplementationV1,
    external_reference: bool,
    runtime_dependency: bool,
    source_teacher_authority: bool,
    sensitivity_authority: bool,
    allocator_authority: bool,
    selector_authority: bool,
    autoquant_authority: bool,
    dwq: bool,
}

pub(crate) fn build_exact_teacher_reference_input(
    plan: &VerifiedTeacherPredictionPlan,
    vocabulary_size: usize,
    target_limits: TeacherTargetArtifactLimits,
) -> Result<ExactTeacherReferenceInputV1, ExactTeacherTargetError> {
    let mut examples = Vec::with_capacity(plan.manifest().examples.len());
    plan.visit_examples(|receipt, token_ids, _points, _greedy| {
        examples.push(ExactTeacherReferenceExampleV1 {
            stable_id: receipt.stable_id.clone(),
            render_mode: receipt.render_mode,
            token_ids: token_ids.to_vec(),
        });
        Ok::<(), ExactTeacherTargetError>(())
    })?;
    let mut artifact = ExactTeacherReferenceInputV1 {
        schema_version: REFERENCE_INPUT_SCHEMA_VERSION,
        profile: REFERENCE_INPUT_PROFILE.into(),
        prediction_plan: plan.manifest().clone(),
        vocabulary_size,
        target_limits,
        examples,
        source_teacher_authority: false,
        sensitivity_authority: false,
        allocator_authority: false,
        selector_authority: false,
        autoquant_authority: false,
        runtime_dependency: false,
        reference_input_sha256: String::new(),
    };
    artifact.reference_input_sha256 = reference_input_sha256(&artifact)?;
    validate_exact_teacher_reference_input(&artifact)?;
    Ok(artifact)
}

pub(crate) fn validate_exact_teacher_reference_input(
    artifact: &ExactTeacherReferenceInputV1,
) -> Result<(), ExactTeacherTargetError> {
    validate_teacher_prediction_plan(&artifact.prediction_plan)
        .map_err(|error| ExactTeacherTargetError::Invalid(error.to_string()))?;
    if artifact.schema_version != REFERENCE_INPUT_SCHEMA_VERSION
        || artifact.profile != REFERENCE_INPUT_PROFILE
        || artifact.vocabulary_size == 0
        || artifact.vocabulary_size > artifact.target_limits.max_vocabulary_size
        || artifact.examples.len() != artifact.prediction_plan.examples.len()
        || artifact.source_teacher_authority
        || artifact.sensitivity_authority
        || artifact.allocator_authority
        || artifact.selector_authority
        || artifact.autoquant_authority
        || artifact.runtime_dependency
        || reference_input_sha256(artifact)? != artifact.reference_input_sha256
    {
        return Err(ExactTeacherTargetError::Invalid(
            "reference input identity or authority scope is invalid".into(),
        ));
    }
    for (example, receipt) in artifact
        .examples
        .iter()
        .zip(&artifact.prediction_plan.examples)
    {
        if example.stable_id != receipt.stable_id
            || example.render_mode != receipt.render_mode
            || example.token_ids.len() != receipt.token_count
            || rendered_token_ids_sha256(&example.stable_id, &example.token_ids)?
                != receipt.token_ids_sha256
            || example.token_ids.iter().any(|token| {
                usize::try_from(*token).unwrap_or(usize::MAX) >= artifact.vocabulary_size
            })
        {
            return Err(ExactTeacherTargetError::Invalid(
                "reference input tokens differ from the prediction-plan receipt".into(),
            ));
        }
    }
    for point in &artifact.prediction_plan.prediction_points {
        let example = artifact
            .examples
            .iter()
            .find(|example| example.stable_id == point.stable_id)
            .ok_or_else(|| {
                ExactTeacherTargetError::Invalid(
                    "reference input point lacks its retained example".into(),
                )
            })?;
        if point.prefix_token_count > example.token_ids.len()
            || prefix_token_sha256(&example.token_ids[..point.prefix_token_count])?
                != point.prefix_token_ids_sha256
            || matches!(
                point.kind,
                crate::intelligence::calibration::TeacherPredictionPointKind::TeacherForced {
                    target_token_index,
                    target_token_id,
                } if example.token_ids.get(target_token_index).copied() != Some(target_token_id)
            )
        {
            return Err(ExactTeacherTargetError::Invalid(
                "reference input point prefix or target token differs".into(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_external_reference_evidence(
    evidence: &ExactTeacherExternalReferenceEvidenceV1,
    input: &ExactTeacherReferenceInputV1,
) -> Result<(), ExactTeacherTargetError> {
    let implementation = &evidence.implementation;
    if evidence.schema_version != EXTERNAL_EVIDENCE_SCHEMA_VERSION
        || evidence.profile != EXTERNAL_EVIDENCE_PROFILE
        || evidence.reference_input_sha256 != input.reference_input_sha256
        || evidence.prediction_plan_sha256 != input.prediction_plan.manifest_sha256
        || evidence.target_artifact.artifact_id != "external_exact_teacher_logits"
        || evidence.target_artifact.role != "external_full_vocabulary_f32_target_rows"
        || !is_sha256(&evidence.target_artifact.sha256)
        || evidence.target_artifact.byte_len == 0
        || evidence.greedy_trajectories.len() != input.prediction_plan.greedy_prompts.len()
        || implementation.name.is_empty()
        || !is_sha256(&implementation.producer_sha256)
        || implementation.repository_url.is_empty()
        || !is_git_commit(&implementation.repository_commit)
        || implementation.package_version.is_empty()
        || !is_sha256(&implementation.dependency_lock_sha256)
        || implementation.python_version.is_empty()
        || implementation.framework_version.is_empty()
        || implementation.device.is_empty()
        || implementation.source_dtype.is_empty()
        || implementation.logit_dtype != "f32_le"
        || implementation.attention_implementation.is_empty()
        || !implementation.cache_enabled
        || !evidence.external_reference
        || evidence.runtime_dependency
        || evidence.source_teacher_authority
        || evidence.sensitivity_authority
        || evidence.allocator_authority
        || evidence.selector_authority
        || evidence.autoquant_authority
        || evidence.dwq
        || external_evidence_sha256(evidence)? != evidence.evidence_sha256
    {
        return Err(ExactTeacherTargetError::Invalid(
            "external reference evidence identity or scope is invalid".into(),
        ));
    }
    Ok(())
}

pub(crate) fn open_native_reference_target(
    path: &Path,
    input: &ExactTeacherReferenceInputV1,
    plan: &VerifiedTeacherPredictionPlan,
    receipt: ExactTeacherTargetReceipt,
) -> Result<super::StructurallyVerifiedTeacherTargetArtifact, ExactTeacherTargetError> {
    validate_exact_teacher_reference_input(input)?;
    ensure_input_matches_plan(input, plan)?;
    open_receipted_structural_teacher_target(path, plan, receipt)
}

pub(crate) fn open_external_reference_target(
    path: &Path,
    input: &ExactTeacherReferenceInputV1,
    plan: &VerifiedTeacherPredictionPlan,
    evidence: &ExactTeacherExternalReferenceEvidenceV1,
) -> Result<super::StructurallyVerifiedTeacherTargetArtifact, ExactTeacherTargetError> {
    validate_exact_teacher_reference_input(input)?;
    ensure_input_matches_plan(input, plan)?;
    validate_external_reference_evidence(evidence, input)?;
    let artifact = reconstruct_structural_teacher_reference_target(
        path,
        plan,
        input.vocabulary_size,
        input.target_limits,
        evidence.greedy_trajectories.clone(),
    )?;
    if artifact.receipt().target_artifact.byte_len != evidence.target_artifact.byte_len
        || artifact.receipt().target_artifact.sha256 != evidence.target_artifact.sha256
    {
        return Err(ExactTeacherTargetError::Invalid(
            "external reference target bytes differ from their evidence".into(),
        ));
    }
    Ok(artifact)
}

pub(crate) fn ensure_input_matches_plan(
    input: &ExactTeacherReferenceInputV1,
    plan: &VerifiedTeacherPredictionPlan,
) -> Result<(), ExactTeacherTargetError> {
    let rebuilt =
        build_exact_teacher_reference_input(plan, input.vocabulary_size, input.target_limits)?;
    if &rebuilt != input {
        return Err(ExactTeacherTargetError::Invalid(
            "reference input differs from the freshly authenticated plan".into(),
        ));
    }
    Ok(())
}

fn reference_input_sha256(
    artifact: &ExactTeacherReferenceInputV1,
) -> Result<String, ExactTeacherTargetError> {
    sha256_json(&ReferenceInputHashView {
        schema_version: artifact.schema_version,
        profile: &artifact.profile,
        prediction_plan: &artifact.prediction_plan,
        vocabulary_size: artifact.vocabulary_size,
        target_limits: artifact.target_limits,
        examples: &artifact.examples,
        source_teacher_authority: artifact.source_teacher_authority,
        sensitivity_authority: artifact.sensitivity_authority,
        allocator_authority: artifact.allocator_authority,
        selector_authority: artifact.selector_authority,
        autoquant_authority: artifact.autoquant_authority,
        runtime_dependency: artifact.runtime_dependency,
    })
}

pub(crate) fn external_evidence_sha256(
    evidence: &ExactTeacherExternalReferenceEvidenceV1,
) -> Result<String, ExactTeacherTargetError> {
    sha256_json(&ExternalEvidenceHashView {
        schema_version: evidence.schema_version,
        profile: &evidence.profile,
        reference_input_sha256: &evidence.reference_input_sha256,
        prediction_plan_sha256: &evidence.prediction_plan_sha256,
        target_artifact: &evidence.target_artifact,
        greedy_trajectories: &evidence.greedy_trajectories,
        implementation: &evidence.implementation,
        external_reference: evidence.external_reference,
        runtime_dependency: evidence.runtime_dependency,
        source_teacher_authority: evidence.source_teacher_authority,
        sensitivity_authority: evidence.sensitivity_authority,
        allocator_authority: evidence.allocator_authority,
        selector_authority: evidence.selector_authority,
        autoquant_authority: evidence.autoquant_authority,
        dwq: evidence.dwq,
    })
}

fn rendered_token_ids_sha256(
    stable_id: &str,
    token_ids: &[u32],
) -> Result<String, ExactTeacherTargetError> {
    let mut bytes = Vec::with_capacity(
        24usize
            .checked_add(stable_id.len())
            .and_then(|value| value.checked_add(token_ids.len().checked_mul(4)?))
            .ok_or_else(|| ExactTeacherTargetError::Invalid("token evidence overflow".into()))?,
    );
    bytes.extend_from_slice(b"hf2q-token-ids-v1");
    bytes.extend_from_slice(
        &u32::try_from(stable_id.len())
            .map_err(|_| ExactTeacherTargetError::Invalid("stable id is too long".into()))?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(stable_id.as_bytes());
    bytes.extend_from_slice(
        &u64::try_from(token_ids.len())
            .map_err(|_| ExactTeacherTargetError::Invalid("token count overflow".into()))?
            .to_le_bytes(),
    );
    for token in token_ids {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn prefix_token_sha256(token_ids: &[u32]) -> Result<String, ExactTeacherTargetError> {
    let mut hasher = Sha256::new();
    hasher.update(b"hf2q-teacher-prefix-token-ids-v1");
    hasher.update(
        u64::try_from(token_ids.len())
            .map_err(|_| ExactTeacherTargetError::Invalid("prefix token count overflow".into()))?
            .to_le_bytes(),
    );
    for token in token_ids {
        hasher.update(token.to_le_bytes());
    }
    Ok(hex::encode(hasher.finalize()))
}

fn sha256_json(value: &impl Serialize) -> Result<String, ExactTeacherTargetError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ExactTeacherTargetError::Serialization(error.to_string()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_git_commit(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
