//! Canonical schema and bounds verification for prediction-plan receipts.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use super::*;
use crate::intelligence::calibration::manifest::ordered_evidence_json_sha256;

const MAX_PLAN_EXAMPLES: usize = 4_096;
const MAX_PLAN_TOTAL_TOKENS: usize = 16 * 1024 * 1024;
const MAX_PLAN_RENDERED_UTF8_BYTES: u64 = 256 * 1024 * 1024;
const MAX_PLAN_PREDICTION_POINTS: usize = 262_144;
const MAX_PLAN_PREFIX_TOKENS: usize = 262_144;
const MAX_PLAN_GENERATION_PROMPTS: usize = 4_096;

#[derive(Serialize)]
struct PredictionPlanHashView<'a> {
    schema_version: u32,
    source: &'a crate::intelligence::measured_auto_quant::SourceIdentity,
    verified_source_manifest_sha256: &'a str,
    dataset_partition_manifest_sha256: &'a str,
    evaluation_split: DatasetSplit,
    evaluation_corpus_artifact_sha256: &'a str,
    evaluation_manifest_sha256: &'a str,
    rendered_token_stream_sha256: &'a str,
    limits: TeacherPredictionPlanLimits,
    total_example_count: usize,
    total_token_count: usize,
    total_rendered_utf8_bytes: u64,
    examples: &'a [TeacherPredictionExampleReceipt],
    prediction_points: &'a [TeacherPredictionPointReceipt],
    greedy_prompts: &'a [TeacherGreedyPromptReceipt],
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(super) fn prediction_plan_sha256(
    manifest: &TeacherPredictionPlanManifest,
) -> Result<String, CalibrationInputError> {
    ordered_evidence_json_sha256(&PredictionPlanHashView {
        schema_version: manifest.schema_version,
        source: &manifest.source,
        verified_source_manifest_sha256: &manifest.verified_source_manifest_sha256,
        dataset_partition_manifest_sha256: &manifest.dataset_partition_manifest_sha256,
        evaluation_split: manifest.evaluation_split,
        evaluation_corpus_artifact_sha256: &manifest.evaluation_corpus_artifact_sha256,
        evaluation_manifest_sha256: &manifest.evaluation_manifest_sha256,
        rendered_token_stream_sha256: &manifest.rendered_token_stream_sha256,
        limits: manifest.limits,
        total_example_count: manifest.total_example_count,
        total_token_count: manifest.total_token_count,
        total_rendered_utf8_bytes: manifest.total_rendered_utf8_bytes,
        examples: &manifest.examples,
        prediction_points: &manifest.prediction_points,
        greedy_prompts: &manifest.greedy_prompts,
    })
}

pub(in crate::intelligence::calibration) fn validate_prediction_plan_limits(
    limits: TeacherPredictionPlanLimits,
) -> Result<(), CalibrationInputError> {
    if limits.max_examples == 0
        || limits.max_total_tokens == 0
        || limits.max_rendered_utf8_bytes == 0
        || limits.max_prediction_points == 0
        || limits.max_prefix_tokens == 0
        || limits.max_generation_prompts == 0
        || limits.max_examples > MAX_PLAN_EXAMPLES
        || limits.max_total_tokens > MAX_PLAN_TOTAL_TOKENS
        || limits.max_rendered_utf8_bytes > MAX_PLAN_RENDERED_UTF8_BYTES
        || limits.max_prediction_points > MAX_PLAN_PREDICTION_POINTS
        || limits.max_prefix_tokens > MAX_PLAN_PREFIX_TOKENS
        || limits.max_generation_prompts > MAX_PLAN_GENERATION_PROMPTS
    {
        return Err(CalibrationInputError::InvalidDataset(
            "teacher prediction-plan bounds must be positive".into(),
        ));
    }
    Ok(())
}

pub fn validate_teacher_prediction_plan(
    manifest: &TeacherPredictionPlanManifest,
) -> Result<(), CalibrationInputError> {
    validate_prediction_plan_limits(manifest.limits)?;
    if manifest.schema_version != TEACHER_PREDICTION_PLAN_SCHEMA_VERSION
        || !super::super::render::source_valid(&manifest.source)
        || !is_lower_sha256(&manifest.verified_source_manifest_sha256)
        || !is_lower_sha256(&manifest.dataset_partition_manifest_sha256)
        || !is_lower_sha256(&manifest.evaluation_corpus_artifact_sha256)
        || !is_lower_sha256(&manifest.evaluation_manifest_sha256)
        || !is_lower_sha256(&manifest.rendered_token_stream_sha256)
        || !is_lower_sha256(&manifest.manifest_sha256)
        || manifest.total_example_count == 0
        || manifest.total_example_count > manifest.limits.max_examples
        || manifest.total_token_count == 0
        || manifest.total_token_count > manifest.limits.max_total_tokens
        || manifest.total_rendered_utf8_bytes == 0
        || manifest.total_rendered_utf8_bytes > manifest.limits.max_rendered_utf8_bytes
        || manifest.examples.len() != manifest.total_example_count
        || manifest.prediction_points.is_empty()
        || manifest.prediction_points.len() > manifest.limits.max_prediction_points
        || manifest.greedy_prompts.len() > manifest.limits.max_generation_prompts
    {
        return Err(CalibrationInputError::InvalidDataset(
            "teacher prediction plan identity or global bounds are invalid".into(),
        ));
    }
    let mut example_ids = BTreeSet::new();
    let mut recomputed_total_tokens = 0usize;
    for example in &manifest.examples {
        if example.stable_id.is_empty()
            || !example_ids.insert(example.stable_id.as_str())
            || example.token_count == 0
            || example.token_count > manifest.limits.max_prefix_tokens
            || !is_lower_sha256(&example.token_ids_sha256)
        {
            return Err(CalibrationInputError::InvalidDataset(
                "teacher prediction example identity is invalid".into(),
            ));
        }
        recomputed_total_tokens = recomputed_total_tokens
            .checked_add(example.token_count)
            .ok_or_else(|| {
                CalibrationInputError::InvalidDataset(
                    "teacher prediction total-token count overflow".into(),
                )
            })?;
    }
    if recomputed_total_tokens != manifest.total_token_count {
        return Err(CalibrationInputError::InvalidDataset(
            "teacher prediction total-token count does not reproduce".into(),
        ));
    }
    let mut generation_points = BTreeMap::new();
    let mut teacher_forced_examples = BTreeSet::new();
    let mut last_teacher_forced_index = BTreeMap::<&str, usize>::new();
    let example_order = manifest
        .examples
        .iter()
        .enumerate()
        .map(|(ordinal, example)| (example.stable_id.as_str(), ordinal))
        .collect::<BTreeMap<_, _>>();
    let mut last_point_example_ordinal = None;
    let mut has_teacher_forced = false;
    for (ordinal, point) in manifest.prediction_points.iter().enumerate() {
        if point.point_ordinal != ordinal
            || point.stable_id.is_empty()
            || point.prefix_token_count == 0
            || point.prefix_token_count > manifest.limits.max_prefix_tokens
            || !is_lower_sha256(&point.prefix_token_ids_sha256)
            || matches!(
                point.kind,
                TeacherPredictionPointKind::TeacherForced {
                    target_token_index,
                    ..
                } if target_token_index != point.prefix_token_count
            )
        {
            return Err(CalibrationInputError::InvalidDataset(
                "teacher prediction point identity or alignment is invalid".into(),
            ));
        }
        let example = manifest
            .examples
            .iter()
            .find(|example| example.stable_id == point.stable_id)
            .ok_or_else(|| {
                CalibrationInputError::InvalidDataset(
                    "teacher prediction point refers to an unknown example".into(),
                )
            })?;
        let example_ordinal = *example_order.get(point.stable_id.as_str()).ok_or_else(|| {
            CalibrationInputError::InvalidDataset(
                "teacher prediction point refers to an unknown example".into(),
            )
        })?;
        if last_point_example_ordinal.is_some_and(|previous| previous > example_ordinal) {
            return Err(CalibrationInputError::InvalidDataset(
                "teacher prediction points do not follow canonical example order".into(),
            ));
        }
        last_point_example_ordinal = Some(example_ordinal);
        match point.kind {
            TeacherPredictionPointKind::TeacherForced {
                target_token_index, ..
            } => {
                if example.render_mode != RenderMode::CompletedAssistantTranscript
                    || point.prefix_token_count >= example.token_count
                    || last_teacher_forced_index
                        .insert(point.stable_id.as_str(), target_token_index)
                        .is_some_and(|previous| previous >= target_token_index)
                {
                    return Err(CalibrationInputError::InvalidDataset(
                        "teacher-forced point disagrees with its example".into(),
                    ));
                }
                has_teacher_forced = true;
                teacher_forced_examples.insert(point.stable_id.as_str());
            }
            TeacherPredictionPointKind::GenerationNext => {
                if example.render_mode != RenderMode::GenerationPrompt
                    || point.prefix_token_count != example.token_count
                    || generation_points
                        .insert(
                            point.stable_id.as_str(),
                            (
                                point.prefix_token_count,
                                point.prefix_token_ids_sha256.as_str(),
                            ),
                        )
                        .is_some()
                {
                    return Err(CalibrationInputError::InvalidDataset(
                        "generation-next point disagrees with its example".into(),
                    ));
                }
            }
        }
    }
    if manifest
        .examples
        .iter()
        .any(|example| match example.render_mode {
            RenderMode::CompletedAssistantTranscript => {
                !teacher_forced_examples.contains(example.stable_id.as_str())
            }
            RenderMode::GenerationPrompt => {
                !generation_points.contains_key(example.stable_id.as_str())
            }
        })
    {
        return Err(CalibrationInputError::InvalidDataset(
            "teacher prediction plan omits an example".into(),
        ));
    }
    let mut greedy_ids = BTreeSet::new();
    let expected_greedy_order = manifest
        .examples
        .iter()
        .filter(|example| example.render_mode == RenderMode::GenerationPrompt)
        .map(|example| example.stable_id.as_str())
        .collect::<Vec<_>>();
    let observed_greedy_order = manifest
        .greedy_prompts
        .iter()
        .map(|prompt| prompt.stable_id.as_str())
        .collect::<Vec<_>>();
    let split_shape_valid = match manifest.evaluation_split {
        DatasetSplit::Calibration => has_teacher_forced && !manifest.greedy_prompts.is_empty(),
        DatasetSplit::PolicyValidation => has_teacher_forced,
        DatasetSplit::AcceptanceHoldout => !manifest.greedy_prompts.is_empty(),
    };
    if !split_shape_valid
        || manifest.greedy_prompts.iter().any(|prompt| {
            prompt.stable_id.is_empty()
                || !greedy_ids.insert(prompt.stable_id.as_str())
                || prompt.prefix_token_count == 0
                || prompt.prefix_token_count > manifest.limits.max_prefix_tokens
                || !is_lower_sha256(&prompt.prefix_token_ids_sha256)
                || generation_points.get(prompt.stable_id.as_str())
                    != Some(&(
                        prompt.prefix_token_count,
                        prompt.prefix_token_ids_sha256.as_str(),
                    ))
                || !manifest.examples.iter().any(|example| {
                    example.stable_id == prompt.stable_id
                        && example.render_mode == RenderMode::GenerationPrompt
                        && example.token_count == prompt.prefix_token_count
                })
        })
        || observed_greedy_order != expected_greedy_order
        || greedy_ids != generation_points.keys().copied().collect()
        || prediction_plan_sha256(manifest)? != manifest.manifest_sha256
    {
        return Err(CalibrationInputError::InvalidDataset(
            "teacher prediction plan hash or greedy prompt is invalid".into(),
        ));
    }
    Ok(())
}
