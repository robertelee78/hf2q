//! Opaque, bounded prediction plans derived from a verified evaluation split.

use sha2::{Digest, Sha256};

use super::render::validate_rendered_dataset;
use super::types::*;

mod verify;

use verify::prediction_plan_sha256;
pub(super) use verify::validate_prediction_plan_limits;
pub use verify::validate_teacher_prediction_plan;

fn prefix_token_sha256(tokens: &[u32]) -> Result<String, CalibrationInputError> {
    let count = u64::try_from(tokens.len()).map_err(|_| {
        CalibrationInputError::InvalidDataset("prediction prefix token count overflow".into())
    })?;
    let mut hasher = Sha256::new();
    hasher.update(b"hf2q-teacher-prefix-token-ids-v1");
    hasher.update(count.to_le_bytes());
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Reverify the three-way split, then derive every teacher-forced and
/// generation-next point from one explicitly selected characterization split.
/// Acceptance-holdout token ids are never retained by this constructor.
pub(crate) fn build_teacher_characterization_plan(
    expected_partition: &DatasetPartitionManifest,
    evaluation_split: DatasetSplit,
    evaluation_corpus: &VerifiedCalibrationCorpus,
    evaluation: &RenderedDataset,
    calibration: &RenderedDataset,
    policy_validation: &RenderedDataset,
    acceptance_holdout: &RenderedDataset,
    limits: TeacherPredictionPlanLimits,
) -> Result<VerifiedTeacherPredictionPlan, CalibrationInputError> {
    validate_prediction_plan_limits(limits)?;
    if !matches!(
        evaluation_split,
        DatasetSplit::Calibration | DatasetSplit::PolicyValidation
    ) {
        return Err(CalibrationInputError::InvalidDataset(
            "characterization plans cannot open AcceptanceHoldout".into(),
        ));
    }
    let actual_partition = super::partition::verify_dataset_partition(
        calibration,
        policy_validation,
        acceptance_holdout,
    )?;
    if &actual_partition != expected_partition {
        return Err(CalibrationInputError::SplitMismatch);
    }
    let rendered_structured = serde_json::to_vec(&evaluation.structured)
        .map_err(|error| CalibrationInputError::Serialization(error.to_string()))?;
    let corpus_structured = serde_json::to_vec(&evaluation_corpus.manifest)
        .map_err(|error| CalibrationInputError::Serialization(error.to_string()))?;
    if rendered_structured != corpus_structured {
        return Err(CalibrationInputError::InvalidDataset(
            "rendered evaluation split differs from its owned corpus artifact".into(),
        ));
    }
    validate_rendered_dataset(evaluation)?;
    if evaluation.manifest.split != evaluation_split
        || evaluation.manifest.examples.len() > limits.max_examples
    {
        return Err(CalibrationInputError::InvalidDataset(
            "teacher prediction plan evaluation split or bound is invalid".into(),
        ));
    }

    let mut total_token_count = 0usize;
    let mut total_rendered_utf8_bytes = 0u64;
    let mut points = Vec::new();
    let mut greedy_prompts = Vec::new();
    let mut example_receipts = Vec::with_capacity(evaluation.manifest.examples.len());
    let mut retained = Vec::with_capacity(evaluation.manifest.examples.len());
    for (receipt, structured) in evaluation
        .manifest
        .examples
        .iter()
        .zip(&evaluation.structured.examples)
    {
        let tokens = evaluation
            .token_ids
            .get(&receipt.stable_id)
            .ok_or_else(|| {
                CalibrationInputError::InvalidDataset(format!(
                    "prediction plan is missing tokens for {}",
                    receipt.stable_id
                ))
            })?;
        if tokens.is_empty() || tokens.len() > limits.max_prefix_tokens {
            return Err(CalibrationInputError::InvalidDataset(format!(
                "prediction prefix for {} is empty or exceeds its bound",
                receipt.stable_id
            )));
        }
        total_token_count = total_token_count.checked_add(tokens.len()).ok_or_else(|| {
            CalibrationInputError::InvalidDataset(
                "teacher prediction total-token count overflow".into(),
            )
        })?;
        if total_token_count > limits.max_total_tokens {
            return Err(CalibrationInputError::InvalidDataset(
                "teacher prediction total-token bound exceeded".into(),
            ));
        }
        let rendered_len = evaluation
            .rendered_utf8
            .get(&receipt.stable_id)
            .ok_or_else(|| {
                CalibrationInputError::InvalidDataset(format!(
                    "prediction plan is missing rendered bytes for {}",
                    receipt.stable_id
                ))
            })?
            .len();
        total_rendered_utf8_bytes = total_rendered_utf8_bytes
            .checked_add(u64::try_from(rendered_len).map_err(|_| {
                CalibrationInputError::InvalidDataset(
                    "rendered evaluation byte count is not representable".into(),
                )
            })?)
            .ok_or_else(|| {
                CalibrationInputError::InvalidDataset(
                    "rendered evaluation byte count overflow".into(),
                )
            })?;
        if total_rendered_utf8_bytes > limits.max_rendered_utf8_bytes {
            return Err(CalibrationInputError::InvalidDataset(
                "rendered evaluation byte bound exceeded".into(),
            ));
        }

        let mut point_ordinals = Vec::new();
        let mut greedy_prompt_ordinal = None;
        match structured.render_mode {
            RenderMode::CompletedAssistantTranscript => {
                for range in &receipt.scoring_ranges {
                    for target_token_index in range.start..range.end {
                        if target_token_index == 0 || target_token_index >= tokens.len() {
                            return Err(CalibrationInputError::InvalidDataset(format!(
                                "teacher-forced alignment is invalid for {}",
                                receipt.stable_id
                            )));
                        }
                        let point_ordinal = points.len();
                        if point_ordinal >= limits.max_prediction_points {
                            return Err(CalibrationInputError::InvalidDataset(
                                "teacher prediction point bound exceeded".into(),
                            ));
                        }
                        points.push(TeacherPredictionPointReceipt {
                            point_ordinal,
                            stable_id: receipt.stable_id.clone(),
                            kind: TeacherPredictionPointKind::TeacherForced {
                                target_token_index,
                                target_token_id: tokens[target_token_index],
                            },
                            prefix_token_count: target_token_index,
                            prefix_token_ids_sha256: prefix_token_sha256(
                                &tokens[..target_token_index],
                            )?,
                        });
                        point_ordinals.push(point_ordinal);
                    }
                }
            }
            RenderMode::GenerationPrompt => {
                let point_ordinal = points.len();
                if point_ordinal >= limits.max_prediction_points
                    || greedy_prompts.len() >= limits.max_generation_prompts
                {
                    return Err(CalibrationInputError::InvalidDataset(
                        "teacher prediction point or greedy-prompt bound exceeded".into(),
                    ));
                }
                let prefix_token_ids_sha256 = prefix_token_sha256(tokens)?;
                points.push(TeacherPredictionPointReceipt {
                    point_ordinal,
                    stable_id: receipt.stable_id.clone(),
                    kind: TeacherPredictionPointKind::GenerationNext,
                    prefix_token_count: tokens.len(),
                    prefix_token_ids_sha256: prefix_token_ids_sha256.clone(),
                });
                point_ordinals.push(point_ordinal);
                greedy_prompt_ordinal = Some(greedy_prompts.len());
                greedy_prompts.push(TeacherGreedyPromptReceipt {
                    stable_id: receipt.stable_id.clone(),
                    prefix_token_count: tokens.len(),
                    prefix_token_ids_sha256,
                });
            }
        }
        if points.len() > limits.max_prediction_points
            || greedy_prompts.len() > limits.max_generation_prompts
        {
            return Err(CalibrationInputError::InvalidDataset(
                "teacher prediction point or greedy-prompt bound exceeded".into(),
            ));
        }
        example_receipts.push(TeacherPredictionExampleReceipt {
            stable_id: receipt.stable_id.clone(),
            render_mode: structured.render_mode,
            token_count: tokens.len(),
            token_ids_sha256: receipt.token_ids_sha256.clone(),
        });
        retained.push(TeacherPredictionExample {
            token_ids: tokens.clone(),
            point_ordinals,
            greedy_prompt_ordinal,
        });
    }
    let has_teacher_forced = points
        .iter()
        .any(|point| matches!(point.kind, TeacherPredictionPointKind::TeacherForced { .. }));
    let split_shape_valid = match evaluation_split {
        DatasetSplit::Calibration => has_teacher_forced && !greedy_prompts.is_empty(),
        DatasetSplit::PolicyValidation => has_teacher_forced,
        DatasetSplit::AcceptanceHoldout => unreachable!("rejected above"),
    };
    if points.is_empty() || !split_shape_valid {
        return Err(CalibrationInputError::InvalidDataset(
            "teacher prediction plan does not satisfy its split-specific point contract".into(),
        ));
    }

    let mut manifest = TeacherPredictionPlanManifest {
        schema_version: TEACHER_PREDICTION_PLAN_SCHEMA_VERSION,
        source: evaluation.manifest.source.clone(),
        verified_source_manifest_sha256: evaluation
            .manifest
            .verified_source_manifest_sha256
            .clone(),
        dataset_partition_manifest_sha256: actual_partition.manifest_sha256,
        evaluation_split,
        evaluation_corpus_artifact_sha256: evaluation_corpus.artifact.sha256.clone(),
        evaluation_manifest_sha256: evaluation.manifest.manifest_sha256.clone(),
        rendered_token_stream_sha256: evaluation.manifest.token_id_stream_sha256.clone(),
        limits,
        total_example_count: retained.len(),
        total_token_count,
        total_rendered_utf8_bytes,
        examples: example_receipts,
        prediction_points: points,
        greedy_prompts,
        manifest_sha256: String::new(),
    };
    manifest.manifest_sha256 = prediction_plan_sha256(&manifest)?;
    validate_teacher_prediction_plan(&manifest)?;
    Ok(VerifiedTeacherPredictionPlan {
        manifest,
        examples: retained,
    })
}

#[cfg(test)]
pub(super) fn prefix_token_sha256_for_test(tokens: &[u32]) -> String {
    prefix_token_sha256(tokens).expect("small token prefix is representable")
}

#[cfg(test)]
pub(super) fn resign_prediction_plan_for_test(manifest: &mut TeacherPredictionPlanManifest) {
    manifest.manifest_sha256 = prediction_plan_sha256(manifest).unwrap();
}

#[cfg(test)]
pub(crate) fn prediction_plan_for_test() -> VerifiedTeacherPredictionPlan {
    let limits = TeacherPredictionPlanLimits {
        max_examples: 2,
        max_total_tokens: 64,
        max_rendered_utf8_bytes: 1_024,
        max_prediction_points: 3,
        max_prefix_tokens: 32,
        max_generation_prompts: 1,
    };
    let transcript_tokens = (0..18).map(|token| token % 4).collect::<Vec<_>>();
    let prompt_tokens = (0..16).map(|token| (token + 2) % 4).collect::<Vec<_>>();
    let points = vec![
        TeacherPredictionPointReceipt {
            point_ordinal: 0,
            stable_id: "completed".into(),
            kind: TeacherPredictionPointKind::TeacherForced {
                target_token_index: 16,
                target_token_id: transcript_tokens[16],
            },
            prefix_token_count: 16,
            prefix_token_ids_sha256: prefix_token_sha256(&transcript_tokens[..16]).unwrap(),
        },
        TeacherPredictionPointReceipt {
            point_ordinal: 1,
            stable_id: "completed".into(),
            kind: TeacherPredictionPointKind::TeacherForced {
                target_token_index: 17,
                target_token_id: transcript_tokens[17],
            },
            prefix_token_count: 17,
            prefix_token_ids_sha256: prefix_token_sha256(&transcript_tokens[..17]).unwrap(),
        },
        TeacherPredictionPointReceipt {
            point_ordinal: 2,
            stable_id: "generation".into(),
            kind: TeacherPredictionPointKind::GenerationNext,
            prefix_token_count: prompt_tokens.len(),
            prefix_token_ids_sha256: prefix_token_sha256(&prompt_tokens).unwrap(),
        },
    ];
    let greedy_prompts = vec![TeacherGreedyPromptReceipt {
        stable_id: "generation".into(),
        prefix_token_count: prompt_tokens.len(),
        prefix_token_ids_sha256: prefix_token_sha256(&prompt_tokens).unwrap(),
    }];
    let examples = vec![
        TeacherPredictionExampleReceipt {
            stable_id: "completed".into(),
            render_mode: RenderMode::CompletedAssistantTranscript,
            token_count: transcript_tokens.len(),
            token_ids_sha256: hex::encode(Sha256::digest(
                super::render::framed_token_bytes_for_test("completed", &transcript_tokens),
            )),
        },
        TeacherPredictionExampleReceipt {
            stable_id: "generation".into(),
            render_mode: RenderMode::GenerationPrompt,
            token_count: prompt_tokens.len(),
            token_ids_sha256: hex::encode(Sha256::digest(
                super::render::framed_token_bytes_for_test("generation", &prompt_tokens),
            )),
        },
    ];
    let mut manifest = TeacherPredictionPlanManifest {
        schema_version: TEACHER_PREDICTION_PLAN_SCHEMA_VERSION,
        source: crate::intelligence::measured_auto_quant::SourceIdentity {
            model_id: "Qwen/Qwen3.8-27B".into(),
            revision: "test-revision".into(),
            config_sha256: "1".repeat(64),
            tensor_bundle_sha256: "2".repeat(64),
            tokenizer_bundle_sha256: "3".repeat(64),
            chat_template_sha256: "4".repeat(64),
        },
        verified_source_manifest_sha256: "5".repeat(64),
        dataset_partition_manifest_sha256: "a".repeat(64),
        evaluation_split: DatasetSplit::Calibration,
        evaluation_corpus_artifact_sha256: "f".repeat(64),
        evaluation_manifest_sha256: "b".repeat(64),
        rendered_token_stream_sha256: "c".repeat(64),
        limits,
        total_example_count: 2,
        total_token_count: transcript_tokens.len() + prompt_tokens.len(),
        total_rendered_utf8_bytes: 32,
        examples,
        prediction_points: points,
        greedy_prompts,
        manifest_sha256: String::new(),
    };
    manifest.manifest_sha256 = prediction_plan_sha256(&manifest).unwrap();
    validate_teacher_prediction_plan(&manifest).unwrap();
    VerifiedTeacherPredictionPlan {
        manifest,
        examples: vec![
            TeacherPredictionExample {
                token_ids: transcript_tokens,
                point_ordinals: vec![0, 1],
                greedy_prompt_ordinal: None,
            },
            TeacherPredictionExample {
                token_ids: prompt_tokens,
                point_ordinals: vec![2],
                greedy_prompt_ordinal: Some(0),
            },
        ],
    }
}

#[cfg(test)]
pub(crate) fn policy_prediction_plan_for_test() -> VerifiedTeacherPredictionPlan {
    let mut plan = prediction_plan_for_test();
    plan.manifest.evaluation_split = DatasetSplit::PolicyValidation;
    plan.manifest.evaluation_corpus_artifact_sha256 = "e".repeat(64);
    plan.manifest.evaluation_manifest_sha256 = "d".repeat(64);
    plan.manifest.examples.truncate(1);
    plan.manifest.prediction_points.truncate(2);
    plan.manifest.greedy_prompts.clear();
    plan.examples.truncate(1);
    plan.manifest.total_example_count = 1;
    plan.manifest.total_token_count = plan.examples[0].token_ids.len();
    plan.manifest.manifest_sha256 = prediction_plan_sha256(&plan.manifest).unwrap();
    validate_teacher_prediction_plan(&plan.manifest).unwrap();
    plan
}

#[cfg(test)]
pub(crate) fn policy_prediction_plan_for_test_bound(
    source: crate::intelligence::measured_auto_quant::SourceIdentity,
    verified_source_manifest_sha256: String,
) -> VerifiedTeacherPredictionPlan {
    let mut plan = policy_prediction_plan_for_test();
    plan.manifest.source = source;
    plan.manifest.verified_source_manifest_sha256 = verified_source_manifest_sha256;
    plan.manifest.manifest_sha256 = prediction_plan_sha256(&plan.manifest).unwrap();
    validate_teacher_prediction_plan(&plan.manifest).unwrap();
    plan
}

#[cfg(test)]
pub(crate) fn prediction_plan_for_test_bound(
    source: crate::intelligence::measured_auto_quant::SourceIdentity,
    verified_source_manifest_sha256: String,
) -> VerifiedTeacherPredictionPlan {
    let mut plan = prediction_plan_for_test();
    plan.manifest.source = source;
    plan.manifest.verified_source_manifest_sha256 = verified_source_manifest_sha256;
    plan.manifest.manifest_sha256 = prediction_plan_sha256(&plan.manifest).unwrap();
    validate_teacher_prediction_plan(&plan.manifest).unwrap();
    plan
}

#[cfg(test)]
pub(crate) fn prediction_plan_for_test_bound_with_first_prefix(
    source: crate::intelligence::measured_auto_quant::SourceIdentity,
    verified_source_manifest_sha256: String,
    first_prefix_token_count: usize,
) -> VerifiedTeacherPredictionPlan {
    let mut plan = prediction_plan_for_test_bound(source, verified_source_manifest_sha256);
    let tokens = &plan.examples[0].token_ids;
    plan.manifest.prediction_points[0].kind = TeacherPredictionPointKind::TeacherForced {
        target_token_index: first_prefix_token_count,
        target_token_id: tokens[first_prefix_token_count],
    };
    plan.manifest.prediction_points[0].prefix_token_count = first_prefix_token_count;
    plan.manifest.prediction_points[0].prefix_token_ids_sha256 =
        prefix_token_sha256(&tokens[..first_prefix_token_count]).unwrap();
    plan.manifest.manifest_sha256 = prediction_plan_sha256(&plan.manifest).unwrap();
    validate_teacher_prediction_plan(&plan.manifest).unwrap();
    plan
}

#[cfg(test)]
pub(crate) fn prediction_plan_for_test_bound_with_gap(
    source: crate::intelligence::measured_auto_quant::SourceIdentity,
    verified_source_manifest_sha256: String,
) -> VerifiedTeacherPredictionPlan {
    let mut plan = prediction_plan_for_test_bound(source, verified_source_manifest_sha256);
    let tokens = &mut plan.examples[0].token_ids;
    tokens.push(2);
    let last_index = tokens.len() - 1;
    plan.manifest.examples[0].token_count = tokens.len();
    plan.manifest.total_token_count += 1;
    plan.manifest.prediction_points[1].kind = TeacherPredictionPointKind::TeacherForced {
        target_token_index: last_index,
        target_token_id: tokens[last_index],
    };
    plan.manifest.prediction_points[1].prefix_token_count = last_index;
    plan.manifest.prediction_points[1].prefix_token_ids_sha256 =
        prefix_token_sha256(&tokens[..last_index]).unwrap();
    plan.manifest.manifest_sha256 = prediction_plan_sha256(&plan.manifest).unwrap();
    validate_teacher_prediction_plan(&plan.manifest).unwrap();
    plan
}
