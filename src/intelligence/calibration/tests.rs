use std::collections::BTreeMap;

use tempfile::TempDir;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

use super::*;
use crate::serve::api::schema::{
    ChatMessage, ContentPart, ImageUrl, MessageContent, Tool, ToolCall, ToolCallFunction,
    ToolFunction,
};

const TEMPLATE: &str = "{% if tools %}{% for tool in tools %} {{ tool.function.name }}{% endfor %}{% endif %}{% for message in messages %} {{ message.role }} {{ message.content }}{% endfor %}{% if add_generation_prompt %} assistant{% endif %}";
const CONFIG: &[u8] = br#"{"architectures":["Qwen3_5ForCausalLM"],"model_type":"qwen3_5"}"#;

fn sha256(bytes: &[u8]) -> String {
    super::manifest::sha256_bytes(bytes)
}

fn source(
    template_sha256: String,
    tokenizer_bundle_sha256: String,
) -> crate::intelligence::measured_auto_quant::SourceIdentity {
    crate::intelligence::measured_auto_quant::SourceIdentity {
        model_id: "Qwen/Qwen3.8-27B".into(),
        revision: "source-revision".into(),
        config_sha256: sha256(CONFIG),
        tensor_bundle_sha256: sha256(b"weights"),
        tokenizer_bundle_sha256,
        chat_template_sha256: template_sha256,
    }
}

fn message(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: Some(MessageContent::Text(content.into())),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

fn example(id: &str, split_word: &str, completed: bool) -> StructuredExample {
    let mut messages = vec![message("user", split_word)];
    if completed {
        messages.push(message("assistant", &format!("answer{id}")));
    }
    StructuredExample {
        stable_id: id.into(),
        provenance: ExampleProvenance {
            dataset_id: "dataset".into(),
            revision: "revision".into(),
            record_id: format!("record-{id}"),
            license: "apache-2.0".into(),
        },
        domains: vec!["agentic-coding".into()],
        messages,
        tools: vec![Tool {
            tool_type: "function".into(),
            function: ToolFunction {
                name: format!("tool{id}"),
                description: Some("read a file".into()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                })),
            },
        }],
        render_mode: if completed {
            RenderMode::CompletedAssistantTranscript
        } else {
            RenderMode::GenerationPrompt
        },
        enable_thinking: false,
        chat_template_kwargs: BTreeMap::new(),
    }
}

fn dataset(split: DatasetSplit, example: StructuredExample) -> StructuredDatasetManifest {
    build_structured_dataset_manifest(
        "dataset".into(),
        "revision".into(),
        "apache-2.0".into(),
        split,
        7,
        vec![example],
    )
    .unwrap()
}

fn model_dir() -> (TempDir, RenderDatasetRequest) {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join("config.json"), CONFIG).unwrap();
    std::fs::write(temp.path().join("chat_template.jinja"), TEMPLATE).unwrap();

    let words = [
        "<unk>",
        "user",
        "assistant",
        "calibration",
        "validation",
        "holdout",
        "answercal",
        "answerval",
        "answerhold",
        "toolcal",
        "toolval",
        "toolhold",
    ];
    let vocab: serde_json::Map<String, serde_json::Value> = words
        .iter()
        .enumerate()
        .map(|(index, word)| ((*word).into(), serde_json::json!(index)))
        .collect();
    let vocab_path = temp.path().join("vocab.json");
    std::fs::write(&vocab_path, serde_json::to_vec(&vocab).unwrap()).unwrap();
    let model = WordLevel::from_file(vocab_path.to_str().unwrap(), "<unk>".into()).unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    tokenizer
        .save(temp.path().join("tokenizer.json"), false)
        .unwrap();

    let tokenizer_bytes = std::fs::read(temp.path().join("tokenizer.json")).unwrap();
    let resolved = crate::core::chat_template_resolver::resolve_chat_render_inputs(
        temp.path(),
        &tokenizer_bytes,
        "qwen35",
    )
    .unwrap();
    let sidecar_bytes = std::fs::read(temp.path().join("chat_template.jinja")).unwrap();
    let verified_source = crate::input::integrity::VerifiedSourceManifest::for_test_bound(
        "Qwen/Qwen3.8-27B",
        "source-revision",
        vec![
            crate::core::integrity::ShardIntegrity {
                filename: "config.json".into(),
                bytes: CONFIG.len() as u64,
                sha256: Some(sha256(CONFIG)),
                hf_etag: sha256(CONFIG),
                is_lfs: true,
            },
            crate::core::integrity::ShardIntegrity {
                filename: "tokenizer.json".into(),
                bytes: tokenizer_bytes.len() as u64,
                sha256: Some(sha256(&tokenizer_bytes)),
                hf_etag: sha256(&tokenizer_bytes),
                is_lfs: true,
            },
            crate::core::integrity::ShardIntegrity {
                filename: "chat_template.jinja".into(),
                bytes: sidecar_bytes.len() as u64,
                sha256: Some(sha256(&sidecar_bytes)),
                hf_etag: sha256(&sidecar_bytes),
                is_lfs: true,
            },
        ],
    );
    let template_sha256 = sha256(TEMPLATE.as_bytes());
    let tensor_bundle_sha256 = crate::core::provenance::compute_source_bundle_sha256(
        &verified_source
            .records()
            .iter()
            .map(crate::core::provenance::SourceShard::from_integrity)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut source = source(template_sha256, resolved.tokenizer_bundle_sha256);
    source.tensor_bundle_sha256 = tensor_bundle_sha256;
    let request = RenderDatasetRequest {
        model_dir: temp.path().into(),
        arch: "qwen35".into(),
        source,
        verified_source,
        renderer_revision: "production-renderer-v1".into(),
        max_tokens_per_example: 64,
        token_window_size: 3,
    };
    (temp, request)
}

#[test]
fn structured_dataset_hash_binds_order_tool_and_split() {
    let first = dataset(
        DatasetSplit::Calibration,
        example("cal", "calibration", false),
    );
    validate_structured_dataset_manifest(&first).unwrap();

    let mut tool_changed = example("cal", "calibration", false);
    tool_changed.tools[0].function.description = Some("different".into());
    let tool_changed = dataset(DatasetSplit::Calibration, tool_changed);
    assert_ne!(first.manifest_sha256, tool_changed.manifest_sha256);

    let split_changed = dataset(
        DatasetSplit::PolicyValidation,
        example("cal", "calibration", false),
    );
    assert_ne!(first.manifest_sha256, split_changed.manifest_sha256);

    let mut two_domains = example("cal", "calibration", false);
    two_domains.domains = vec!["multilingual".into(), "agentic-coding".into()];
    let ordered = dataset(DatasetSplit::Calibration, two_domains.clone());
    two_domains.domains.reverse();
    let reversed = dataset(DatasetSplit::Calibration, two_domains);
    assert_eq!(ordered.manifest_sha256, reversed.manifest_sha256);

    let bad_provenance = build_structured_dataset_manifest(
        "other".into(),
        "revision".into(),
        "apache-2.0".into(),
        DatasetSplit::Calibration,
        7,
        vec![example("cal", "calibration", false)],
    );
    assert!(matches!(
        bad_provenance,
        Err(CalibrationInputError::InvalidDataset(_))
    ));
}

#[test]
fn production_rendering_and_partition_are_source_bound_and_disjoint() {
    let (_temp, request) = model_dir();
    let calibration = render_and_tokenize_split(
        &dataset(
            DatasetSplit::Calibration,
            example("cal", "calibration", true),
        ),
        &request,
    )
    .unwrap();
    verify_rendered_dataset_from_source(&calibration, &request).unwrap();
    let validation = render_and_tokenize_split(
        &dataset(
            DatasetSplit::PolicyValidation,
            example("val", "validation", true),
        ),
        &request,
    )
    .unwrap();
    let holdout = render_and_tokenize_split(
        &dataset(
            DatasetSplit::AcceptanceHoldout,
            example("hold", "holdout", true),
        ),
        &request,
    )
    .unwrap();

    assert_eq!(calibration.manifest.examples[0].scoring_ranges.len(), 1);
    assert!(calibration.manifest.examples[0].scoring_ranges[0].start > 0);
    assert_eq!(
        calibration.manifest.chat_template_source,
        crate::core::chat_template_resolver::ChatTemplateSource::Sidecar
    );
    let partition = verify_dataset_partition(&calibration, &validation, &holdout).unwrap();
    assert_eq!(partition.overlap_receipt.compared_example_count, 3);
    assert_eq!(partition.overlap_receipt.raw_overlap_count, 0);
    assert_eq!(partition.overlap_receipt.source_record_overlap_count, 0);
    assert_eq!(partition.overlap_receipt.rendered_overlap_count, 0);
    assert_eq!(partition.overlap_receipt.token_window_overlap_count, 0);
    assert_ne!(
        partition.manifest_sha256,
        partition.overlap_receipt.receipt_sha256
    );
}

#[test]
fn characterization_plans_bind_the_selected_split_and_keep_holdout_closed() {
    let (temp, request) = model_dir();
    let calibration_manifest = build_structured_dataset_manifest(
        "dataset".into(),
        "revision".into(),
        "apache-2.0".into(),
        DatasetSplit::Calibration,
        7,
        vec![
            example("cal", "calibration", true),
            example("gen", "calibration", false),
        ],
    )
    .unwrap();
    let corpus_path = temp.path().join("calibration.json");
    let corpus_bytes = serde_json::to_vec(&calibration_manifest).unwrap();
    std::fs::write(&corpus_path, &corpus_bytes).unwrap();
    let corpus = verify_calibration_corpus_artifact(&VerifyCalibrationCorpusRequest {
        path: corpus_path,
        expected_sha256: sha256(&corpus_bytes),
        expected_dataset_id: "dataset".into(),
        expected_revision: "revision".into(),
        expected_declared_license: "apache-2.0".into(),
        expected_split: DatasetSplit::Calibration,
        limits: CalibrationCorpusArtifactLimits {
            max_artifact_bytes: 64 * 1024,
            max_examples: 4,
            max_messages: 8,
            max_tools: 4,
        },
    })
    .unwrap();
    assert_eq!(corpus.artifact().sha256, sha256(&corpus_bytes));
    let limits = TeacherPredictionPlanLimits {
        max_examples: 4,
        max_total_tokens: 128,
        max_rendered_utf8_bytes: 8 * 1024,
        max_prediction_points: 64,
        max_prefix_tokens: 64,
        max_generation_prompts: 2,
    };
    let mut tiny_render_limits = limits;
    tiny_render_limits.max_total_tokens = 1;
    assert!(render_and_tokenize_verified_split(&corpus, &request, tiny_render_limits).is_err());
    let calibration = render_and_tokenize_verified_split(&corpus, &request, limits).unwrap();
    let validation = render_and_tokenize_split(
        &dataset(
            DatasetSplit::PolicyValidation,
            example("val", "validation", true),
        ),
        &request,
    )
    .unwrap();
    let holdout = render_and_tokenize_split(
        &dataset(
            DatasetSplit::AcceptanceHoldout,
            example("hold", "holdout", false),
        ),
        &request,
    )
    .unwrap();
    let partition = verify_dataset_partition(&calibration, &validation, &holdout).unwrap();
    let plan = build_teacher_characterization_plan(
        &partition,
        DatasetSplit::Calibration,
        &corpus,
        &calibration,
        &calibration,
        &validation,
        &holdout,
        limits,
    )
    .unwrap();

    validate_teacher_prediction_plan(plan.manifest()).unwrap();
    assert_eq!(plan.manifest().evaluation_split, DatasetSplit::Calibration);
    assert_eq!(plan.manifest().source, calibration.manifest().source);
    assert_eq!(
        plan.manifest().verified_source_manifest_sha256,
        calibration.manifest().verified_source_manifest_sha256
    );
    assert_eq!(plan.manifest().greedy_prompts.len(), 1);
    assert_eq!(plan.manifest().greedy_prompts[0].stable_id, "gen");
    assert!(plan.prediction_point_count() >= 2);
    let mut seen_generation = false;
    let mut seen_teacher_forced = false;
    plan.visit_prediction_points::<std::convert::Infallible>(|point, prefix| {
        assert_eq!(point.prefix_token_count, prefix.len());
        assert_eq!(
            point.prefix_token_ids_sha256,
            super::prediction_plan::prefix_token_sha256_for_test(prefix)
        );
        assert_ne!(point.stable_id, "val");
        assert_ne!(point.stable_id, "hold");
        match point.kind {
            TeacherPredictionPointKind::TeacherForced {
                target_token_index,
                target_token_id,
            } => {
                seen_teacher_forced = true;
                assert_eq!(target_token_index, prefix.len());
                assert_eq!(
                    calibration.token_ids[&point.stable_id][target_token_index],
                    target_token_id
                );
            }
            TeacherPredictionPointKind::GenerationNext => {
                seen_generation = true;
                assert_eq!(point.stable_id, "gen");
            }
        }
        Ok(())
    })
    .unwrap();
    assert!(seen_generation && seen_teacher_forced);

    let mut grouped_ids = Vec::new();
    plan.visit_examples::<std::convert::Infallible>(
        |example_receipt, token_ids, points, greedy_prompt| {
            grouped_ids.push(example_receipt.stable_id.clone());
            assert_eq!(example_receipt.token_count, token_ids.len());
            assert!(!points.is_empty());
            assert!(points
                .iter()
                .all(|point| point.stable_id == example_receipt.stable_id));
            assert!(points
                .windows(2)
                .all(|pair| pair[1].point_ordinal == pair[0].point_ordinal + 1));
            match example_receipt.render_mode {
                RenderMode::CompletedAssistantTranscript => assert!(greedy_prompt.is_none()),
                RenderMode::GenerationPrompt => {
                    assert_eq!(points.len(), 1);
                    assert_eq!(greedy_prompt.unwrap().stable_id, example_receipt.stable_id);
                }
            }
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(grouped_ids, vec!["cal", "gen"]);

    let validation_corpus_path = temp.path().join("policy-validation.json");
    let validation_corpus_bytes = serde_json::to_vec(&validation.structured).unwrap();
    std::fs::write(&validation_corpus_path, &validation_corpus_bytes).unwrap();
    let validation_corpus = verify_calibration_corpus_artifact(&VerifyCalibrationCorpusRequest {
        path: validation_corpus_path,
        expected_sha256: sha256(&validation_corpus_bytes),
        expected_dataset_id: "dataset".into(),
        expected_revision: "revision".into(),
        expected_declared_license: "apache-2.0".into(),
        expected_split: DatasetSplit::PolicyValidation,
        limits: CalibrationCorpusArtifactLimits {
            max_artifact_bytes: 64 * 1024,
            max_examples: 4,
            max_messages: 8,
            max_tools: 4,
        },
    })
    .unwrap();
    let policy_plan = build_teacher_characterization_plan(
        &partition,
        DatasetSplit::PolicyValidation,
        &validation_corpus,
        &validation,
        &calibration,
        &validation,
        &holdout,
        limits,
    )
    .unwrap();
    assert_eq!(
        policy_plan.manifest().evaluation_split,
        DatasetSplit::PolicyValidation
    );
    assert!(policy_plan.manifest().greedy_prompts.is_empty());
    assert!(policy_plan
        .manifest()
        .prediction_points
        .iter()
        .all(|point| point.stable_id == "val"));
    validate_teacher_prediction_plan(policy_plan.manifest()).unwrap();

    let holdout_corpus_path = temp.path().join("acceptance-holdout.json");
    let holdout_corpus_bytes = serde_json::to_vec(&holdout.structured).unwrap();
    std::fs::write(&holdout_corpus_path, &holdout_corpus_bytes).unwrap();
    let holdout_corpus = verify_calibration_corpus_artifact(&VerifyCalibrationCorpusRequest {
        path: holdout_corpus_path,
        expected_sha256: sha256(&holdout_corpus_bytes),
        expected_dataset_id: "dataset".into(),
        expected_revision: "revision".into(),
        expected_declared_license: "apache-2.0".into(),
        expected_split: DatasetSplit::AcceptanceHoldout,
        limits: CalibrationCorpusArtifactLimits {
            max_artifact_bytes: 64 * 1024,
            max_examples: 4,
            max_messages: 8,
            max_tools: 4,
        },
    })
    .unwrap();
    let thresholds = bind_teacher_acceptance_thresholds(
        "1".repeat(64),
        "2".repeat(64),
        "3".repeat(64),
        holdout.manifest().source.clone(),
        holdout.manifest().verified_source_manifest_sha256.clone(),
        holdout_corpus.artifact().sha256.clone(),
    )
    .unwrap();
    let authorized_holdout = build_teacher_acceptance_holdout_plan(
        thresholds,
        &partition,
        &holdout_corpus,
        &holdout,
        &calibration,
        &validation,
        &holdout,
        limits,
    )
    .unwrap();
    assert_eq!(
        authorized_holdout.threshold_profile_sha256(),
        "1".repeat(64)
    );
    let holdout_plan = authorized_holdout.into_prediction_plan();
    assert_eq!(
        holdout_plan.manifest().evaluation_split,
        DatasetSplit::AcceptanceHoldout
    );
    assert_eq!(holdout_plan.manifest().greedy_prompts.len(), 1);
    assert!(holdout_plan
        .manifest()
        .prediction_points
        .iter()
        .all(|point| point.stable_id == "hold"));
    validate_teacher_prediction_plan(holdout_plan.manifest()).unwrap();

    let wrong_thresholds = bind_teacher_acceptance_thresholds(
        "1".repeat(64),
        "2".repeat(64),
        "3".repeat(64),
        holdout.manifest().source.clone(),
        holdout.manifest().verified_source_manifest_sha256.clone(),
        "4".repeat(64),
    )
    .unwrap();
    assert!(build_teacher_acceptance_holdout_plan(
        wrong_thresholds,
        &partition,
        &holdout_corpus,
        &holdout,
        &calibration,
        &validation,
        &holdout,
        limits,
    )
    .is_err());

    assert!(matches!(
        build_teacher_characterization_plan(
            &partition,
            DatasetSplit::AcceptanceHoldout,
            &holdout_corpus,
            &holdout,
            &calibration,
            &validation,
            &holdout,
            limits,
        ),
        Err(CalibrationInputError::InvalidDataset(_))
    ));

    let mut wrong_partition = partition.clone();
    wrong_partition.manifest_sha256 = "0".repeat(64);
    assert!(matches!(
        build_teacher_characterization_plan(
            &wrong_partition,
            DatasetSplit::Calibration,
            &corpus,
            &calibration,
            &calibration,
            &validation,
            &holdout,
            limits,
        ),
        Err(CalibrationInputError::SplitMismatch)
    ));

    let mut too_small = limits;
    too_small.max_prediction_points = 1;
    assert!(matches!(
        build_teacher_characterization_plan(
            &partition,
            DatasetSplit::Calibration,
            &corpus,
            &calibration,
            &calibration,
            &validation,
            &holdout,
            too_small,
        ),
        Err(CalibrationInputError::InvalidDataset(_))
    ));
}

#[test]
fn prediction_plan_validator_rejects_rehashed_noncanonical_example_and_greedy_order() {
    let plan = super::prediction_plan::prediction_plan_for_test();

    let mut reordered_examples = plan.manifest().clone();
    reordered_examples.examples.swap(0, 1);
    super::prediction_plan::resign_prediction_plan_for_test(&mut reordered_examples);
    assert!(validate_teacher_prediction_plan(&reordered_examples).is_err());

    let mut reordered_greedy = plan.manifest().clone();
    let original = reordered_greedy.greedy_prompts[0].clone();
    reordered_greedy.greedy_prompts.push(original);
    super::prediction_plan::resign_prediction_plan_for_test(&mut reordered_greedy);
    assert!(validate_teacher_prediction_plan(&reordered_greedy).is_err());

    let mut invalid_source = plan.manifest().clone();
    invalid_source.source.config_sha256 = "not-a-sha256".into();
    super::prediction_plan::resign_prediction_plan_for_test(&mut invalid_source);
    assert!(validate_teacher_prediction_plan(&invalid_source).is_err());

    let mut invalid_source_manifest = plan.manifest().clone();
    invalid_source_manifest.verified_source_manifest_sha256 = "A".repeat(64);
    super::prediction_plan::resign_prediction_plan_for_test(&mut invalid_source_manifest);
    assert!(validate_teacher_prediction_plan(&invalid_source_manifest).is_err());
}

#[test]
fn corpus_artifact_is_owned_bounded_and_hash_authenticated() {
    let temp = tempfile::tempdir().unwrap();
    let manifest = dataset(
        DatasetSplit::Calibration,
        example("cal", "calibration", true),
    );
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let path = temp.path().join("corpus.json");
    std::fs::write(&path, &bytes).unwrap();
    let base = VerifyCalibrationCorpusRequest {
        path: path.clone(),
        expected_sha256: sha256(&bytes),
        expected_dataset_id: "dataset".into(),
        expected_revision: "revision".into(),
        expected_declared_license: "apache-2.0".into(),
        expected_split: DatasetSplit::Calibration,
        limits: CalibrationCorpusArtifactLimits {
            max_artifact_bytes: 64 * 1024,
            max_examples: 2,
            max_messages: 4,
            max_tools: 2,
        },
    };
    let verified = verify_calibration_corpus_artifact(&base).unwrap();
    assert_eq!(
        verified.manifest().manifest_sha256,
        manifest.manifest_sha256
    );

    std::fs::write(&path, b"replaced").unwrap();
    assert_eq!(
        verified.manifest().manifest_sha256,
        manifest.manifest_sha256,
        "verified corpus must retain owned authenticated bytes"
    );
    assert!(verify_calibration_corpus_artifact(&base).is_err());

    std::fs::write(&path, &bytes).unwrap();
    let mut too_small = base.clone();
    too_small.limits.max_artifact_bytes = u64::try_from(bytes.len() - 1).unwrap();
    assert!(verify_calibration_corpus_artifact(&too_small).is_err());
    let mut wrong_license = base;
    wrong_license.expected_declared_license = "different".into();
    assert!(verify_calibration_corpus_artifact(&wrong_license).is_err());
}

#[test]
fn partition_rejects_rendered_and_token_window_overlap() {
    let (_temp, request) = model_dir();
    let calibration = render_and_tokenize_split(
        &dataset(
            DatasetSplit::Calibration,
            example("cal", "calibration", true),
        ),
        &request,
    )
    .unwrap();
    let mut duplicate = example("val", "calibration", true);
    duplicate.messages[1] = message("assistant", "answercal");
    duplicate.tools[0].function.name = "toolcal".into();
    let validation = render_and_tokenize_split(
        &dataset(DatasetSplit::PolicyValidation, duplicate),
        &request,
    )
    .unwrap();
    let holdout = render_and_tokenize_split(
        &dataset(
            DatasetSplit::AcceptanceHoldout,
            example("hold", "holdout", true),
        ),
        &request,
    )
    .unwrap();
    assert!(matches!(
        verify_dataset_partition(&calibration, &validation, &holdout),
        Err(CalibrationInputError::DatasetOverlap { raw, rendered, .. })
            if raw > 0 && rendered > 0
    ));
}

#[test]
fn rendering_rejects_media_reserved_kwargs_and_tampered_tokens() {
    let (_temp, request) = model_dir();
    let mut media = example("cal", "calibration", false);
    media.messages[0].content = Some(MessageContent::Parts(vec![ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: "file:///tmp/image.png".into(),
            detail: None,
        },
    }]));
    assert!(matches!(
        render_and_tokenize_split(&dataset(DatasetSplit::Calibration, media), &request),
        Err(CalibrationInputError::MediaUnsupported(_))
    ));

    let mut reserved = example("cal", "calibration", false);
    reserved
        .chat_template_kwargs
        .insert("messages".into(), serde_json::json!([]));
    assert!(matches!(
        render_and_tokenize_split(&dataset(DatasetSplit::Calibration, reserved), &request),
        Err(CalibrationInputError::Render { .. })
    ));

    let mut rendered = render_and_tokenize_split(
        &dataset(
            DatasetSplit::Calibration,
            example("cal", "calibration", false),
        ),
        &request,
    )
    .unwrap();
    rendered.token_ids.get_mut("cal").unwrap()[0] ^= 1;
    assert!(matches!(
        super::render::validate_rendered_dataset(&rendered),
        Err(CalibrationInputError::InvalidDataset(_))
    ));

    let mut rendered = render_and_tokenize_split(
        &dataset(
            DatasetSplit::Calibration,
            example("cal", "calibration", false),
        ),
        &request,
    )
    .unwrap();
    rendered.rendered_utf8.get_mut("cal").unwrap().push('x');
    assert!(matches!(
        super::render::validate_rendered_dataset(&rendered),
        Err(CalibrationInputError::InvalidDataset(_))
    ));
}

#[test]
fn token_hash_framing_distinguishes_ambiguous_integer_sequences() {
    let left = super::render::framed_token_bytes_for_test("example", &[1, 23]);
    let right = super::render::framed_token_bytes_for_test("example", &[12, 3]);
    assert_ne!(left, right);
    assert_ne!(sha256(&left), sha256(&right));
}

#[test]
fn source_record_identity_cannot_be_renamed_across_splits() {
    let (_temp, request) = model_dir();
    let calibration = render_and_tokenize_split(
        &dataset(
            DatasetSplit::Calibration,
            example("cal", "calibration", false),
        ),
        &request,
    )
    .unwrap();
    let mut renamed = example("val", "validation", false);
    renamed.provenance.record_id = "record-cal".into();
    let validation =
        render_and_tokenize_split(&dataset(DatasetSplit::PolicyValidation, renamed), &request)
            .unwrap();
    let holdout = render_and_tokenize_split(
        &dataset(
            DatasetSplit::AcceptanceHoldout,
            example("hold", "holdout", false),
        ),
        &request,
    )
    .unwrap();
    assert!(matches!(
        verify_dataset_partition(&calibration, &validation, &holdout),
        Err(CalibrationInputError::DatasetOverlap {
            source_records,
            ..
        }) if source_records > 0
    ));
}

#[test]
fn render_inputs_must_match_the_opaque_verified_source_snapshot() {
    let (_temp, mut request) = model_dir();
    request.source.tensor_bundle_sha256 = "0".repeat(64);
    assert!(matches!(
        render_and_tokenize_split(
            &dataset(
                DatasetSplit::Calibration,
                example("cal", "calibration", false),
            ),
            &request,
        ),
        Err(CalibrationInputError::InvalidDataset(_))
    ));

    let (temp, request) = model_dir();
    let tokenizer_path = temp.path().join("tokenizer.json");
    let mut bytes = std::fs::read(&tokenizer_path).unwrap();
    bytes.push(b' ');
    std::fs::write(tokenizer_path, bytes).unwrap();
    assert!(matches!(
        render_and_tokenize_split(
            &dataset(
                DatasetSplit::Calibration,
                example("cal", "calibration", false),
            ),
            &request,
        ),
        Err(CalibrationInputError::InvalidDataset(_))
            | Err(CalibrationInputError::SourceTokenizerBundleMismatch)
    ));

    let (temp, request) = model_dir();
    std::fs::remove_file(temp.path().join("chat_template.jinja")).unwrap();
    assert!(matches!(
        render_and_tokenize_split(
            &dataset(
                DatasetSplit::Calibration,
                example("cal", "calibration", false),
            ),
            &request,
        ),
        Err(CalibrationInputError::InvalidDataset(_))
    ));
}

#[test]
fn overlap_windows_are_fixed_width_and_detect_containment() {
    assert!(matches!(
        super::render::token_window_hashes_for_test("short", &[1, 2], 3),
        Err(CalibrationInputError::TokenWindowTooShort { .. })
    ));
    let exact = super::render::token_window_hashes_for_test("exact", &[1, 2, 3], 3).unwrap();
    let longer = super::render::token_window_hashes_for_test("long", &[9, 1, 2, 3, 8], 3).unwrap();
    assert!(longer.contains(&exact[0]));
}

#[test]
fn production_qwen_template_preserves_tool_call_and_result_transcript() {
    let mut assistant = message("assistant", "");
    assistant.content = None;
    assistant.tool_calls = Some(vec![ToolCall {
        id: "call-1".into(),
        call_type: "function".into(),
        function: ToolCallFunction {
            name: "read_file".into(),
            arguments: r#"{"path":"Cargo.toml"}"#.into(),
        },
    }]);
    let mut tool_result = message("tool", "package metadata");
    tool_result.tool_call_id = Some("call-1".into());
    let tools = vec![Tool {
        tool_type: "function".into(),
        function: ToolFunction {
            name: "read_file".into(),
            description: Some("read a file".into()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            })),
        },
    }];
    let rendered = crate::serve::api::engine::render_chat_prompt_with_tools_generation_prompt(
        crate::core::chat_templates::QWEN3_CHATML,
        &[
            message("user", "inspect the manifest"),
            assistant,
            tool_result,
        ],
        Some(&tools),
        false,
        None,
        false,
    )
    .unwrap();
    assert!(rendered.contains("<tool_call>\n<function=read_file>"));
    assert!(rendered.contains("<parameter=path>\nCargo.toml\n</parameter>"));
    assert!(rendered.contains("<tool_response>\npackage metadata\n</tool_response>"));
}
