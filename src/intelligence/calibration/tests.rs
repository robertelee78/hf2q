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
    let request = RenderDatasetRequest {
        model_dir: temp.path().into(),
        arch: "qwen35".into(),
        source: source(template_sha256, resolved.tokenizer_bundle_sha256),
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
