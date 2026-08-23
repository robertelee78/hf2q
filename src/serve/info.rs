//! Static serving preflight for `hf2q info`.
//!
//! Model and projector compatibility is derived from GGUF metadata and tensor
//! directories without decoding tensors or initializing Metal. When the text
//! GGUF declares an exact projector digest, the explicitly supplied projector
//! is additionally streamed as bytes to verify that checksum.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use mlx_native::gguf::GgufFile;

use crate::cli;
use crate::core::provenance::Provenance;
use crate::inference::models::deepseek4::Deepseek4Model;
use crate::inference::models::qwen35::model::Qwen35Model;
use crate::inference::vision::mmproj::{self, ArchProfile, MmprojConfig};
use crate::setup::ServeDefaultsV2;

use super::api::engine::{EngineMode, VisionConsumerContract};
use super::config::Gemma4Config;
use super::info_catalog::{
    validate_family_context_floor, validate_gemma_tensors, validate_qwen35_tensors,
    validate_tensor_headers,
};
use super::operator_settings::{self, ResolvedContext, ResolvedKvBudget};

pub(super) struct StaticInspection {
    pub(super) model_path: PathBuf,
    pub(super) model_id: String,
    pub(super) architecture: String,
    pub(super) family: &'static str,
    pub(super) quant: String,
    pub(super) file_bytes: u64,
    pub(super) metadata_count: usize,
    pub(super) tensor_count: usize,
    pub(super) context: ResolvedContext,
    pub(super) engine_mode: EngineMode,
    pub(super) kv_budget: ResolvedKvBudget,
    pub(super) kv_persist_dir: Option<PathBuf>,
    pub(super) kv_persist_budget: ResolvedKvBudget,
    pub(super) kv_bytes_per_token: u64,
    pub(super) kv_fixed_bytes_per_slot: u64,
    pub(super) vision: String,
    pub(super) projector: Option<PathBuf>,
    pub(super) support: std::result::Result<&'static str, String>,
}

pub fn cmd_info(args: cli::InfoArgs, operator_defaults: Option<&ServeDefaultsV2>) -> Result<()> {
    match inspect(&args, operator_defaults) {
        Ok(report) => {
            super::info_report::print_report(&report);
            match report.support {
                Ok(_) => Ok(()),
                Err(reason) => Err(anyhow::anyhow!(reason)),
            }
        }
        Err(error) => {
            print_early_rejection(&args.model, &format!("{error:#}"));
            Err(error)
        }
    }
}

/// Preserve `info`'s final-line/exit-status contract for failures that occur
/// before GGUF inspection begins, including an invalid selected setup file.
pub(crate) fn print_early_rejection(model: &Path, reason: &str) {
    let reason = reason
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" | ");
    let reason = if reason.is_empty() {
        "unknown static preflight failure"
    } else {
        &reason
    };
    println!("Model: {}", model.display());
    println!(
        "Validation: static preflight (tensor payloads not decoded or uploaded; Metal not initialized)"
    );
    println!("Serve support: rejected — {reason}");
}

fn inspect(
    args: &cli::InfoArgs,
    operator_defaults: Option<&ServeDefaultsV2>,
) -> Result<StaticInspection> {
    anyhow::ensure!(
        args.model.is_file(),
        "model GGUF not found: {}",
        args.model.display()
    );
    let pair_guard = args
        .mmproj
        .as_deref()
        .map(|projector| {
            crate::core::paired_artifact::PairReadGuard::acquire_read_only(&args.model, projector)
                .map_err(|error| anyhow::anyhow!(error))
        })
        .transpose()?;
    let gguf = GgufFile::open(&args.model)
        .map_err(|error| anyhow::anyhow!("GGUF header parse failed: {error}"))?;
    let architecture = gguf
        .metadata_string("general.architecture")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("GGUF is missing required `general.architecture`"))?
        .to_owned();
    if !matches!(
        architecture.as_str(),
        "gemma4" | "qwen35" | "qwen35moe" | "deepseek4" | "qwen3_vl" | "qwen3vl" | "qwen3vlmoe"
    ) {
        anyhow::bail!(
            "unsupported GGUF general.architecture={architecture:?}; supported serve runtimes are gemma4, qwen35, qwen35moe, and deepseek4"
        );
    }
    let requested = operator_settings::requested_context(args.planning.ctx, operator_defaults)
        .map_err(anyhow::Error::msg)?;
    let context = operator_settings::resolve_context_for_gguf(&gguf, requested)
        .map_err(anyhow::Error::msg)?;
    validate_family_context_floor(&gguf, context).map_err(anyhow::Error::msg)?;
    let engine_mode = operator_settings::resolve_scheduler(&args.planning, operator_defaults)
        .map_err(anyhow::Error::msg)?;
    let scheduler_support = match engine_mode {
        EngineMode::SerialFifo => Ok("ready"),
        EngineMode::SlotAware { max_slots } => {
            super::api::engine::Engine::validate_slot_aware_capacity(max_slots)
                .map(|()| "ready")
                .map_err(|error| error.to_string())
        }
    };
    let kv_budget = operator_settings::resolve_kv_cache_budget(
        args.planning.kv_cache_budget.as_deref(),
        operator_defaults,
    )
    .map_err(anyhow::Error::msg)?;
    let kv_persist_budget = operator_settings::resolve_kv_persist_budget(
        args.planning.kv_persist_budget.as_deref(),
        operator_defaults,
    )
    .map_err(anyhow::Error::msg)?;
    operator_settings::validate_kv_persist_plan(
        args.planning.kv_persist_path.as_deref(),
        kv_persist_budget,
    )
    .map_err(anyhow::Error::msg)?;

    let file_bytes = validate_tensor_headers(&gguf, &args.model)?;

    let mut support = scheduler_support;
    let (family, kv_bytes_per_token, kv_fixed_bytes_per_slot, vision_contract) = match architecture
        .as_str()
    {
        "gemma4" => {
            let cfg = Gemma4Config::from_gguf(&gguf)
                .context("Gemma 4 metadata/config validation failed")?;
            validate_gemma_tensors(&gguf, &cfg)?;
            crate::inference::models::gemma4::tokenizer::build_tokenizer_from_gguf(&gguf)
                .context("Gemma 4 embedded tokenizer validation failed")?;
            let template = gguf
                .metadata_string("tokenizer.chat_template")
                .unwrap_or(crate::serve::FALLBACK_GEMMA4_API_CHAT_TEMPLATE);
            crate::core::chat_templates::validate_tool_chat_template("gemma4", template).map_err(
                |error| anyhow::anyhow!("Gemma 4 chat-template validation failed: {error}"),
            )?;
            let contract = vision_contract(
                &gguf,
                ArchProfile::Gemma4Siglip,
                cfg.hidden_size as u32,
                Some(0),
            )?;
            (
                "Gemma 4",
                super::load_info::gemma4_slot_kv_bytes_per_token(&cfg),
                super::load_info::gemma4_fixed_kv_bytes_per_slot(&cfg),
                Some(contract),
            )
        }
        "qwen35" | "qwen35moe" => {
            let cfg = Qwen35Model::load_config_only(&gguf)
                .context("Qwen3.5/3.6 metadata/config validation failed")?;
            validate_qwen35_tensors(&gguf, &cfg)?;
            let tokenizer =
                crate::inference::models::qwen35::tokenizer::build_tokenizer_from_gguf(&gguf)
                    .context("Qwen embedded tokenizer validation failed")?;
            let template = gguf
                .metadata_string("tokenizer.chat_template")
                .unwrap_or(crate::core::chat_templates::QWEN3_CHATML);
            crate::core::chat_templates::validate_tool_chat_template(&architecture, template)
                .map_err(|error| {
                    anyhow::anyhow!("Qwen chat-template validation failed: {error}")
                })?;
            let has_vision_markers = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
                .iter()
                .all(|token| tokenizer.token_to_id(token).is_some());
            let vision_contract = (gguf.metadata_string("hf2q.vision.projector_profile")
                == Some("qwen3vl_siglip")
                || has_vision_markers)
                .then(|| {
                    vision_contract(
                        &gguf,
                        ArchProfile::Qwen3VlSiglip,
                        cfg.hidden_size,
                        gguf.metadata_u32("hf2q.vision.deepstack_output_count"),
                    )
                })
                .transpose()?;
            let tq_active = super::api::tq_packed_descriptor::is_tq_active_mode();
            (
                "Qwen 3.5/3.6/3.8",
                super::load_info::qwen35_slot_kv_bytes_per_token(&cfg, tq_active),
                super::load_info::qwen35_fixed_kv_bytes_per_slot(&cfg),
                vision_contract,
            )
        }
        "deepseek4" => {
            let cfg = Deepseek4Model::load_config_only(&gguf)
                .context("DeepSeek-V4 metadata/config validation failed")?;
            crate::inference::models::deepseek4::weights::validate_tensor_catalog(&gguf, &cfg)
                .map_err(|error| anyhow::anyhow!("DeepSeek-V4 tensor catalog: {error}"))?;
            crate::inference::models::deepseek4::tokenizer::build_tokenizer_from_gguf(&gguf)
                .context("DeepSeek-V4 embedded tokenizer validation failed")?;
            (
                "DeepSeek V4 Flash",
                6_880,
                super::load_info::deepseek4_fixed_kv_bytes_per_slot(&cfg, 6_880),
                None,
            )
        }
        "qwen3_vl" | "qwen3vl" => {
            support = Err(format!(
                    "Qwen3-VL architecture {architecture:?} is recognized, but the serve engine seam is not wired (ADR-041 iter-9b); serving would reject this model before tensor loading"
                ));
            ("Qwen 3 VL", 0, 0, None)
        }
        "qwen3_vl_moe" | "qwen3vlmoe" => {
            support = Err(format!(
                    "Qwen3-VL MoE architecture {architecture:?} is recognized, but no supported hf2q serve pipeline can consume it"
                ));
            ("Qwen 3 VL MoE", 0, 0, None)
        }
        other => {
            support = Err(format!(
                "unsupported GGUF general.architecture={other:?}; supported serve runtimes are gemma4, qwen35, qwen35moe, and deepseek4"
            ));
            ("Unsupported", 0, 0, None)
        }
    };

    let projector = if let Some(path) = args.mmproj.as_deref() {
        if support.is_ok() {
            validate_projector(
                &gguf,
                path,
                vision_contract.clone(),
                pair_guard
                    .as_ref()
                    .expect("pair guard acquired whenever --mmproj is present"),
            )?;
        }
        Some(path.to_path_buf())
    } else {
        None
    };
    let vision = match (vision_contract.is_some(), projector.is_some()) {
        (true, true) => "supported; explicit compatible projector supplied".to_owned(),
        (true, false) => "supported by text model; image input requires --mmproj".to_owned(),
        (false, true) => "projector supplied but text model has no vision contract".to_owned(),
        (false, false) => "not supported by this text model".to_owned(),
    };
    if support.is_ok() && projector.is_some() && vision_contract.is_none() {
        support = Err(
            "capability_unsupported: the text artifact has no exact vision-consumer contract"
                .to_owned(),
        );
    }

    let model_id = gguf
        .metadata_string("general.name")
        // DeepSeek's accepted artifact stores the source digest as its
        // general.name. Mirror the serving loader: prefer that metadata only
        // when it is descriptive, otherwise expose the stable file stem.
        .filter(|name| {
            architecture != "deepseek4" || name.to_ascii_lowercase().contains("deepseek")
        })
        .map(str::to_owned)
        .or_else(|| {
            args.model
                .file_stem()
                .map(|value| value.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "unknown".to_owned());
    let quant =
        super::load_info::infer_quant_label(&gguf).unwrap_or_else(|| "unknown/mixed".to_owned());
    Ok(StaticInspection {
        model_path: args.model.clone(),
        model_id,
        architecture,
        family,
        quant,
        file_bytes,
        metadata_count: gguf.metadata_count(),
        tensor_count: gguf.tensor_count(),
        context,
        engine_mode,
        kv_budget,
        kv_persist_dir: args.planning.kv_persist_path.clone(),
        kv_persist_budget,
        kv_bytes_per_token,
        kv_fixed_bytes_per_slot,
        vision,
        projector,
        support,
    })
}

fn vision_contract(
    gguf: &GgufFile,
    profile: ArchProfile,
    output_width: u32,
    deepstack_output_count: Option<u32>,
) -> Result<VisionConsumerContract> {
    let source_sha256 = match crate::core::provenance::detect(gguf) {
        Provenance::Hf2q { source_sha256, .. } => Some(source_sha256),
        Provenance::External => None,
    };
    let expected_projector_sha256 =
        crate::core::provenance::projector_sha256(gguf).map_err(anyhow::Error::msg)?;
    Ok(VisionConsumerContract {
        profile,
        output_width,
        deepstack_output_count,
        source_sha256,
        expected_projector_sha256,
    })
}

fn validate_projector(
    text_gguf: &GgufFile,
    path: &Path,
    contract: Option<VisionConsumerContract>,
    pair_guard: &crate::core::paired_artifact::PairReadGuard,
) -> Result<()> {
    anyhow::ensure!(path.is_file(), "mmproj GGUF not found: {}", path.display());
    let gguf = GgufFile::open(path)
        .map_err(|error| anyhow::anyhow!("mmproj GGUF header parse failed: {error}"))?;
    validate_tensor_headers(&gguf, path).context("mmproj tensor directory validation failed")?;
    let cfg = MmprojConfig::from_gguf(&gguf).context("mmproj config validation failed")?;
    let names = gguf.tensor_names();
    mmproj::validate_tensor_set(&cfg, &names).context("mmproj tensor-set validation failed")?;
    let profile = mmproj::detect_arch_profile_with_projector(&cfg.projector, &names);
    anyhow::ensure!(
        profile.is_supported(),
        "mmproj architecture profile {:?} has no hf2q runtime",
        profile
    );
    let source_sha256 = match crate::core::provenance::detect(&gguf) {
        Provenance::Hf2q { source_sha256, .. } => Some(source_sha256),
        Provenance::External => None,
    };
    let expected_hash = contract
        .as_ref()
        .and_then(|contract| contract.expected_projector_sha256.as_deref());
    let artifact_hash = expected_hash
        .map(|_| {
            crate::core::sha256::compute_file_sha256(path)
                .with_context(|| format!("hash bound projector {}", path.display()))
        })
        .transpose()?;
    pair_guard
        .validate_static(text_gguf, &gguf, artifact_hash.as_deref())
        .map_err(|error| anyhow::anyhow!(error))?;
    super::validate_mmproj_text_binding(
        contract,
        profile,
        cfg.projection_dim,
        cfg.deepstack_indexes.as_ref().map_or(0, Vec::len),
        source_sha256.as_deref(),
        artifact_hash.as_deref(),
    )
}
