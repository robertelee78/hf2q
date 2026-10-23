//! Opt-in offline capture through production rendering/tokenizer functions.
//! Does not construct a device, engine, model graph or inference cache.
use super::{
    reasoning_controls_from_request, registry, resolve_api_enable_thinking,
    resolve_api_template_kwargs,
};
use crate::serve::api::schema::{ChatCompletionRequest, ToolChoiceValue};
use mlx_native::gguf::GgufFile;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Deserialize)]
struct CaptureConfig {
    output: PathBuf,
    models: Vec<ModelBinding>,
    gcd_enabled: bool,
}

#[derive(Deserialize)]
struct ModelBinding {
    id: String,
    gguf: PathBuf,
    declared_model_sha256: String,
    requests: PathBuf,
    context_tokens: usize,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    request: serde_json::Value,
}

fn safe_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn digest_file(path: &std::path::Path) -> String {
    use std::io::Read;
    let mut file = std::fs::File::open(path).unwrap();
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    hex::encode(hash.finalize())
}

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn capture_template<'a>(architecture: &str, embedded: Option<&'a str>) -> (&'a str, &'static str) {
    if architecture == "deepseek4" {
        (
            crate::serve::api::engine_deepseek4::Deepseek4LoadedModel::serving_chat_template(),
            "native_encoding",
        )
    } else {
        (
            embedded.expect("explicit embedded template required"),
            "gguf_metadata",
        )
    }
}

#[test]
#[ignore = "requires HF2Q_FROZEN_INPUT_CONFIG; offline real-artifact input capture, no model load"]
fn four_agent_frozen_inputs() {
    let path = std::env::var("HF2Q_FROZEN_INPUT_CONFIG").expect("explicit capture config required");
    let config_bytes = std::fs::read(path).unwrap();
    let config: CaptureConfig = serde_json::from_slice(&config_bytes).unwrap();
    assert!(
        !config.gcd_enabled,
        "capture only supports explicitly disabled GCD"
    );
    assert!(config.output.is_absolute(), "output must be absolute");
    let checkout = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .canonicalize()
        .unwrap();
    let parent = config.output.parent().unwrap().canonicalize().unwrap();
    assert!(
        parent.ancestors().all(|p| !p.join(".git").exists()),
        "output must be outside all Git checkout ancestors"
    );
    let git = |args: &[&str]| {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(&checkout)
            .output()
            .unwrap();
        assert!(output.status.success());
        String::from_utf8(output.stdout).unwrap().trim().to_string()
    };
    let source_commit = git(&["rev-parse", "HEAD"]);
    assert_eq!(
        Some(source_commit.as_str()),
        option_env!("HF2Q_BUILD_GIT_SHA"),
        "set GIT_COMMIT_SHA to the clean source commit and rebuild the capture binary"
    );
    assert!(
        git(&["status", "--porcelain"]).is_empty(),
        "capture requires a clean committed source tree"
    );
    assert!(
        !parent.starts_with(&checkout),
        "output must be outside the checkout"
    );
    let expected = ["qwen38", "qwen36", "gemma4", "deepseek4"];
    let ids: std::collections::BTreeSet<_> = config.models.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, expected.into_iter().collect());
    assert_eq!(config.models.len(), 4);
    std::fs::create_dir(&config.output).expect("output must not already exist");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&config.output, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let mut observations = Vec::new();
    for model in config.models {
        assert!(safe_id(&model.id));
        assert!(
            model.declared_model_sha256.len() == 64
                && model
                    .declared_model_sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        );
        let gguf = GgufFile::open(&model.gguf).unwrap();
        let architecture = gguf.metadata_string("general.architecture").unwrap();
        let expected_arch = match model.id.as_str() {
            "qwen38" => "qwen35",
            "qwen36" => "qwen35moe",
            "gemma4" => "gemma4",
            "deepseek4" => "deepseek4",
            _ => unreachable!(),
        };
        assert_eq!(architecture, expected_arch);
        let mut tokenizer = match architecture {
            "qwen35" | "qwen35moe" => {
                crate::inference::models::qwen35::tokenizer::build_tokenizer_from_gguf(&gguf)
            }
            "gemma4" => {
                crate::inference::models::gemma4::tokenizer::build_tokenizer_from_gguf(&gguf)
            }
            "deepseek4" => {
                crate::inference::models::deepseek4::tokenizer::build_tokenizer_from_gguf(&gguf)
            }
            _ => unreachable!(),
        }
        .unwrap();
        tokenizer.with_truncation(None).unwrap();
        let (template, template_source) = capture_template(
            architecture,
            gguf.metadata_string("tokenizer.chat_template"),
        );
        crate::core::chat_templates::validate_tool_chat_template(architecture, template).unwrap();
        let declared_max = gguf
            .metadata_u32(&format!("{architecture}.context_length"))
            .map(u64::from)
            .or_else(
                || match gguf.metadata(&format!("{architecture}.context_length")) {
                    Some(mlx_native::gguf::MetadataValue::Uint64(value)) => Some(*value),
                    _ => None,
                },
            )
            .unwrap();
        assert!(model.context_tokens > 0 && model.context_tokens as u64 <= declared_max);
        let registration = registry::find_for_architecture(architecture).unwrap();
        let template_path = config.output.join(format!("{}.template.jinja", model.id));
        std::fs::write(template_path, template.as_bytes()).unwrap();
        let tokenizer_json = tokenizer.to_string(false).unwrap();
        std::fs::write(
            config.output.join(format!("{}.tokenizer.json", model.id)),
            tokenizer_json.as_bytes(),
        )
        .unwrap();
        let request_bytes = std::fs::read(model.requests).unwrap();
        let cases: Vec<Case> = serde_json::from_slice(&request_bytes).unwrap();
        assert!(!cases.is_empty());
        let mut seen = std::collections::HashSet::new();
        for case in cases {
            assert!(
                safe_id(&case.id) && seen.insert(case.id.clone()),
                "unique safe case id required"
            );
            let supported = [
                "model",
                "messages",
                "tools",
                "tool_choice",
                "max_tokens",
                "max_completion_tokens",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "seed",
                "frequency_penalty",
                "presence_penalty",
                "repetition_penalty",
                "stream",
                "stream_options",
                "reasoning_effort",
                "reasoning",
                "thinking_token_budget",
                "chat_template_kwargs",
                "parallel_tool_calls",
            ];
            assert!(
                case.request
                    .as_object()
                    .unwrap()
                    .keys()
                    .all(|key| supported.contains(&key.as_str())),
                "unsupported request control; capture refuses rather than ignoring it"
            );
            let req: ChatCompletionRequest = serde_json::from_value(case.request).unwrap();
            assert!(!req.messages.is_empty());
            assert_ne!(req.max_completion_tokens.or(req.max_tokens), Some(0));
            assert!(
                req.messages
                    .iter()
                    .all(|m| m.content.as_ref().is_none_or(|c| c.image_urls().is_empty())),
                "text-only capture refuses image input"
            );
            let controls = reasoning_controls_from_request(&req).unwrap();
            let supports_thinking = crate::serve::template_supports_enable_thinking(template);
            let channel = crate::serve::api::reasoning_controls::reasoning_channel_for(
                Some(&registration),
                supports_thinking,
            );
            assert!(
                !matches!(
                    channel,
                    crate::serve::api::reasoning_controls::ReasoningChannel::None
                ) || !controls.d5_control_present()
                    || controls.requested_thinking_off(),
                "unsupported reasoning control for family"
            );
            let thinking = resolve_api_enable_thinking(
                controls.thinking_enabled_override(),
                crate::serve::template_supports_enable_thinking(template),
            );
            let kwargs = resolve_api_template_kwargs(
                Some(&registration),
                req.chat_template_kwargs.as_ref(),
                controls.deepseek_reasoning_effort_tier(),
            );
            let choice = crate::serve::api::grammar::request::validate_tool_request(&req).unwrap();
            assert!(
                matches!(choice, ToolChoiceValue::None | ToolChoiceValue::Auto),
                "capture supports auto/none tool choice only"
            );
            let tools = if matches!(choice, ToolChoiceValue::None) {
                None
            } else {
                req.tools.as_deref()
            };
            if registration.family == "qwen35" {
                if let Some(definitions) = tools {
                    registry::qwen35_tool_argument_wire_kinds(definitions).unwrap();
                }
            }
            let rendered = super::render_chat_prompt_or_400(
                template,
                &req.messages,
                tools,
                thinking,
                kwargs.as_ref(),
            )
            .unwrap();
            let tokens = super::tokenize_http_prompt(&tokenizer, &rendered).unwrap();
            assert!(
                tokens.len() < model.context_tokens,
                "capture refuses inputs requiring overflow rewriting"
            );
            assert!(!tokens.is_empty());
            let token_json = serde_json::to_vec(&tokens).unwrap();
            let prefix = format!("{}-{}", model.id, case.id);
            std::fs::write(
                config.output.join(format!("{prefix}.prompt.txt")),
                rendered.as_bytes(),
            )
            .unwrap();
            std::fs::write(
                config.output.join(format!("{prefix}.tokens.json")),
                &token_json,
            )
            .unwrap();
            observations.push(json!({"model":model.id,"case":case.id,"architecture":architecture,
                "declared_model_sha256":model.declared_model_sha256,"request_file_sha256":digest(&request_bytes),
                "template_sha256":digest(template.as_bytes()),"chat_template_source":template_source,"context_tokens":model.context_tokens,"tokenizer_sha256":digest(tokenizer_json.as_bytes()),
                "rendered_sha256":digest(rendered.as_bytes()),"tokens_sha256":digest(&token_json),
                "token_count":tokens.len(),"enable_thinking":thinking,"resolved_template_kwargs":kwargs}));
        }
    }
    let receipt = json!({"schema":"hf2q.offline-rendered-inputs.v1","model_inference_performed":false,
        "model_payload_hash_verified_here":false,"runtime_qualified":false,"gcd_enabled":false,"overflow_policy_applied":false,
        "source_commit":source_commit,"source_worktree_clean":true,
        "capture_config_sha256":digest(&config_bytes),
        "capture_recipe_source_sha256":digest(include_bytes!("handlers_four_agent_input_tests.rs")),
        "http_handler_source_sha256":digest(include_bytes!("handlers.rs")),
        "renderer_source_sha256":digest(include_bytes!("engine.rs")),
        "deepseek_loader_source_sha256":digest(include_bytes!("engine_deepseek4.rs")),
        "capture_binary_sha256":digest_file(&std::env::current_exe().unwrap()),
        "observations":observations});
    std::fs::write(
        config.output.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
    println!("offline rendered inputs: {}", config.output.display());
}

#[test]
fn http_prompt_tokens_do_not_add_bos_or_eos_postprocessor_tokens() {
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(
            [
                ("word".to_owned(), 0),
                ("[BOS]".to_owned(), 1),
                ("[EOS]".to_owned(), 2),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".to_owned())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    let processor = tokenizers::processors::template::TemplateProcessing::builder()
        .try_single("[BOS] $A [EOS]")
        .unwrap()
        .special_tokens(vec![("[BOS]", 1), ("[EOS]", 2)])
        .build()
        .unwrap();
    tokenizer.with_post_processor(Some(processor));
    assert_eq!(
        tokenizer.encode("word", true).unwrap().get_ids(),
        &[1, 0, 2]
    );
    assert_eq!(
        super::tokenize_http_prompt(&tokenizer, "word").unwrap(),
        vec![0]
    );
}

#[test]
fn deepseek_capture_cannot_be_redirected_by_a_foreign_embedded_template() {
    let (template, source) = capture_template("deepseek4", Some("foreign protocol"));
    assert_eq!(source, "native_encoding");
    assert_eq!(
        template,
        crate::core::chat_templates::DEEPSEEK_V4_FLASH_0731
    );
    assert_ne!(template, "foreign protocol");
}
