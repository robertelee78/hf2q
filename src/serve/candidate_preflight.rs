//! Model-allocation-free explicit-switch candidate validation.
//!
//! This path opens headers, builds tokenizers, and validates sidecar schemas,
//! but creates no Metal device or model tensor storage. It deliberately runs
//! before ADR-047 drains or shuts down a callable generation; the complete
//! loader repeats these checks while constructing the unpublished candidate.

use std::collections::HashSet;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use mlx_native::gguf::GgufFile;
use safetensors::tensor::Dtype;
use tokenizers::Tokenizer;

use super::api::engine::{Engine, EngineMode, LoadedModel, VisionConsumerContract};
use super::multi_model::EngineConfig;
use super::quant_select::QuantType;

pub(crate) fn preflight_engine_candidate(
    path: &Path,
    expected_quant: QuantType,
    config: &EngineConfig,
) -> Result<()> {
    anyhow::ensure!(path.is_file(), "candidate model is not a regular file");
    if let Some(expected) = config.expected_text_artifact_stamp.as_ref() {
        anyhow::ensure!(
            expected.matches_path(path),
            "candidate text artifact changed after model-local policy resolution"
        );
    }
    let gguf = GgufFile::open(path)
        .map_err(|error| anyhow::anyhow!("candidate GGUF header parse failed: {error}"))?;
    super::info_catalog::validate_tensor_headers(&gguf, path)
        .context("candidate GGUF tensor-directory extent preflight")?;

    let architecture = gguf
        .metadata_string("general.architecture")
        .unwrap_or_default();
    LoadedModel::validate_architecture_for_load(architecture, path)?;
    anyhow::ensure!(
        gguf.metadata_u32("general.file_type") == Some(expected_quant.gguf_file_type()),
        "candidate GGUF quant identity changed after switch planning"
    );
    let context =
        super::operator_settings::resolve_context_for_gguf(&gguf, config.requested_context)
            .map_err(anyhow::Error::msg)?;
    super::info_catalog::validate_family_context_floor(&gguf, context)
        .map_err(anyhow::Error::msg)?;

    if let EngineMode::SlotAware { max_slots } = config.engine_mode {
        anyhow::ensure!(max_slots > 0, "SlotAware max_slots must be nonzero");
        Engine::validate_slot_aware_capacity(
            max_slots,
            matches!(architecture, "qwen35" | "qwen35moe"),
        )
        .map_err(anyhow::Error::new)?;
    }

    let vision_contract = match architecture {
        "gemma4" => preflight_gemma(path, &gguf, config.tokenizer_path.as_deref())?,
        "qwen35" | "qwen35moe" => preflight_qwen(&gguf)?,
        "deepseek4" => preflight_deepseek(&gguf, config.tokenizer_path.as_deref())?,
        _ => unreachable!("architecture authority admitted an unknown family"),
    };

    if let Some(overlay) = config.dwq_overlay_path.as_deref() {
        anyhow::ensure!(
            architecture != "deepseek4",
            "DeepSeek-V4 does not consume a DWQ overlay sidecar"
        );
        preflight_dwq_overlay(overlay)?;
    }
    if let Some(projector) = config.projector.as_ref() {
        preflight_projector(path, &gguf, vision_contract, projector)?;
    }
    Ok(())
}

fn provenance_source(gguf: &GgufFile) -> Option<String> {
    match crate::core::provenance::detect(gguf) {
        crate::core::provenance::Provenance::Hf2q { source_sha256, .. } => Some(source_sha256),
        crate::core::provenance::Provenance::External => None,
    }
}

fn expected_projector_sha256(gguf: &GgufFile, family: &str) -> Result<Option<String>> {
    crate::core::provenance::projector_sha256(gguf)
        .map_err(|error| anyhow::anyhow!("{family} projector binding: {error}"))
}

fn preflight_gemma(
    path: &Path,
    gguf: &GgufFile,
    explicit_tokenizer: Option<&Path>,
) -> Result<Option<VisionConsumerContract>> {
    let cfg =
        super::config::Gemma4Config::from_gguf(gguf).context("Gemma4 metadata/config preflight")?;
    crate::inference::models::gemma4::native_matrix::preflight_io(
        gguf,
        cfg.vocab_size,
        cfg.hidden_size,
    )
    .context("Gemma4 IO tensor/native-route preflight")?;
    crate::inference::models::gemma4::native_matrix::preflight_projections(gguf, &cfg)
        .context("Gemma4 projection tensor/native-route preflight")?;
    crate::inference::models::gemma4::native_matrix::preflight_f32_state(gguf, &cfg)
        .context("Gemma4 exact-F32 state preflight")?;

    let template = gguf
        .metadata_string("tokenizer.chat_template")
        .unwrap_or(super::FALLBACK_GEMMA4_API_CHAT_TEMPLATE);
    crate::core::chat_templates::validate_tool_chat_template("gemma4", template)
        .map_err(|error| anyhow::anyhow!("Gemma4 chat template contract: {error}"))?;

    let force_embedded = std::env::var("HF2Q_TOKENIZER_GGUF_EMBEDDED")
        .ok()
        .map(|value| !matches!(value.to_ascii_lowercase().as_str(), "0" | "false" | "off"))
        .unwrap_or(true);
    let tokenizer_path =
        super::api::engine::resolve_tokenizer_path_optional(path, explicit_tokenizer);
    let mut tokenizer = if force_embedded {
        crate::inference::models::gemma4::tokenizer::build_tokenizer_from_gguf(gguf)
            .context("build Gemma4 tokenizer from GGUF metadata")?
    } else if let Some(path) = tokenizer_path {
        Tokenizer::from_file(&path)
            .map_err(|error| anyhow::anyhow!("load tokenizer {}: {error}", path.display()))?
    } else {
        crate::inference::models::gemma4::tokenizer::build_tokenizer_from_gguf(gguf)
            .context("build Gemma4 tokenizer from GGUF metadata")?
    };
    tokenizer
        .with_truncation(None)
        .map_err(|error| anyhow::anyhow!("disable Gemma4 tokenizer truncation: {error}"))?;

    Ok(Some(VisionConsumerContract {
        profile: crate::inference::vision::mmproj::ArchProfile::Gemma4Siglip,
        output_width: u32::try_from(cfg.hidden_size).context("Gemma4 hidden size exceeds u32")?,
        deepstack_output_count: Some(0),
        source_sha256: provenance_source(gguf),
        expected_projector_sha256: expected_projector_sha256(gguf, "Gemma4")?,
    }))
}

fn preflight_qwen(gguf: &GgufFile) -> Result<Option<VisionConsumerContract>> {
    let cfg = crate::inference::models::qwen35::model::Qwen35Model::load_config_only(gguf)
        .context("Qwen metadata/config preflight")?;
    crate::inference::models::qwen35::model::Qwen35Model::preflight_gguf(gguf, &cfg)?;
    let (_tokenizer, vision_special_tokens_present) =
        super::api::engine_qwen35::build_qwen35_serving_tokenizer(gguf)?;
    let template = gguf
        .metadata_string("tokenizer.chat_template")
        .unwrap_or(crate::core::chat_templates::QWEN3_CHATML);
    let template_arch = gguf
        .metadata_string("general.architecture")
        .unwrap_or("qwen35moe");
    crate::core::chat_templates::validate_tool_chat_template(template_arch, template)
        .map_err(|error| anyhow::anyhow!("Qwen chat template contract: {error}"))?;

    let profile = gguf.metadata_string("hf2q.vision.projector_profile");
    if profile == Some("qwen3vl_siglip") || vision_special_tokens_present {
        Ok(Some(VisionConsumerContract {
            profile: crate::inference::vision::mmproj::ArchProfile::QwenVisionSiglip,
            output_width: cfg.hidden_size,
            deepstack_output_count: gguf.metadata_u32("hf2q.vision.deepstack_output_count"),
            source_sha256: provenance_source(gguf),
            expected_projector_sha256: expected_projector_sha256(gguf, "Qwen")?,
        }))
    } else {
        Ok(None)
    }
}

fn preflight_deepseek(
    gguf: &GgufFile,
    tokenizer_path: Option<&Path>,
) -> Result<Option<VisionConsumerContract>> {
    let cfg = crate::inference::models::deepseek4::Deepseek4Model::load_config_only(gguf)
        .context("DeepSeek-V4 metadata/config preflight")?;
    crate::inference::models::deepseek4::Deepseek4Weights::preflight_gguf(gguf, &cfg)
        .context("DeepSeek-V4 tensor/native-route preflight")?;
    if let Some(path) = tokenizer_path {
        Tokenizer::from_file(path)
            .map_err(|error| anyhow::anyhow!("load tokenizer {}: {error}", path.display()))?;
    } else {
        crate::inference::models::deepseek4::tokenizer::build_tokenizer_from_gguf(gguf)
            .context("build DeepSeek-V4 tokenizer from GGUF metadata")?;
    }
    Ok(None)
}

fn preflight_dwq_overlay(path: &Path) -> Result<()> {
    let file = File::open(path).with_context(|| format!("open DWQ overlay {}", path.display()))?;
    // SAFETY: the map is immutable and remains owned until every borrowed
    // safetensors view has been validated and dropped in this function.
    let mapped = unsafe { memmap2::Mmap::map(&file) }
        .with_context(|| format!("map DWQ overlay {}", path.display()))?;
    let (_header_len, metadata) = safetensors::SafeTensors::read_metadata(&mapped)
        .map_err(|error| anyhow::anyhow!("read DWQ overlay metadata: {error:?}"))?;
    let (bits, group_size) =
        super::forward_mlx_shared::parse_dwq_overlay_metadata(metadata.metadata().as_ref())?;
    anyhow::ensure!(matches!(bits, 4 | 8), "DWQ overlay bits must be 4 or 8");
    anyhow::ensure!(group_size > 0, "DWQ overlay group_size must be nonzero");
    let tensors = safetensors::SafeTensors::deserialize(&mapped)
        .map_err(|error| anyhow::anyhow!("deserialize DWQ overlay: {error:?}"))?;
    let names = tensors.names().into_iter().collect::<HashSet<_>>();
    let mut stems = HashSet::new();
    for name in &names {
        for suffix in [".weight", ".scales", ".biases"] {
            if let Some(stem) = name.strip_suffix(suffix) {
                stems.insert(stem);
            }
        }
    }
    anyhow::ensure!(!stems.is_empty(), "DWQ overlay contains no affine tensors");
    for stem in stems {
        let weight_name = format!("{stem}.weight");
        let scales_name = format!("{stem}.scales");
        let biases_name = format!("{stem}.biases");
        let weight = tensors
            .tensor(&weight_name)
            .with_context(|| format!("DWQ overlay missing {weight_name}"))?;
        let scales = tensors
            .tensor(&scales_name)
            .with_context(|| format!("DWQ overlay missing {scales_name}"))?;
        let biases = tensors
            .tensor(&biases_name)
            .with_context(|| format!("DWQ overlay missing {biases_name}"))?;
        anyhow::ensure!(weight.dtype() == Dtype::U32, "{weight_name} must be U32");
        anyhow::ensure!(
            matches!(scales.dtype(), Dtype::F32 | Dtype::F16 | Dtype::BF16)
                && biases.dtype() == scales.dtype(),
            "{scales_name}/{biases_name} must use one matching floating dtype"
        );
        anyhow::ensure!(weight.shape().len() == 2, "{weight_name} must be rank two");
        let n = weight.shape()[0];
        let k = weight.shape()[1]
            .checked_mul(32 / bits as usize)
            .context("DWQ packed width overflow")?;
        anyhow::ensure!(
            k % group_size == 0,
            "{weight_name} unpacked width {k} is not divisible by group_size {group_size}"
        );
        let expected = [n, k / group_size];
        anyhow::ensure!(
            scales.shape() == expected && biases.shape() == expected,
            "{scales_name}/{biases_name} shapes must both be {expected:?}"
        );
    }
    Ok(())
}

fn preflight_projector(
    text_path: &Path,
    text_gguf: &GgufFile,
    contract: Option<VisionConsumerContract>,
    admission: &super::multi_model::ProjectorAdmission,
) -> Result<()> {
    admission.validate()?;
    let digest = crate::core::sha256::compute_file_sha256(&admission.path)
        .with_context(|| format!("hash projector {}", admission.path.display()))?;
    anyhow::ensure!(
        digest.eq_ignore_ascii_case(&admission.artifact_sha256),
        "projector digest changed after admission"
    );
    let pair_guard =
        crate::core::paired_artifact::PairReadGuard::acquire(text_path, &admission.path)
            .map_err(anyhow::Error::new)?;
    let projector_gguf = GgufFile::open(&admission.path)
        .map_err(|error| anyhow::anyhow!("projector GGUF header parse failed: {error}"))?;
    super::info_catalog::validate_tensor_headers(&projector_gguf, &admission.path)
        .context("projector tensor-directory extent preflight")?;
    pair_guard
        .validate(text_gguf, &projector_gguf, &digest)
        .map_err(anyhow::Error::new)?;
    let cfg = crate::inference::vision::mmproj::MmprojConfig::from_gguf(&projector_gguf)
        .context("projector config preflight")?;
    let names = projector_gguf.tensor_names();
    crate::inference::vision::mmproj::validate_tensor_set(&cfg, &names)
        .context("projector tensor-set preflight")?;
    crate::inference::vision::mmproj_weights::LoadedMmprojWeights::validate_native_storage(
        &projector_gguf,
    )
    .context("projector native-storage preflight")?;
    let profile = crate::inference::vision::mmproj::detect_arch_profile_with_projector(
        &cfg.projector,
        &names,
    );
    anyhow::ensure!(
        profile == admission.profile,
        "projector profile changed after admission"
    );
    let source = provenance_source(&projector_gguf);
    super::validate_mmproj_text_binding(
        contract,
        profile,
        cfg.projection_dim,
        cfg.deepstack_indexes.as_ref().map_or(0, Vec::len),
        source.as_deref(),
        Some(&digest),
    )
}
