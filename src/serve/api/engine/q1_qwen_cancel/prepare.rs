//! Prospective packet metadata only. No device/model/context/inference/acquisition.
use crate::adr063_invariance::{
    acquisition::{self, FileBinding, FileIdentity},
    protocol::Workload,
    receipt::Family,
    store::hash,
};
use crate::inference::models::qwen35::{Qwen35Config, Qwen35Variant};
use anyhow::{ensure, Context, Result};
use serde::{de::DeserializeOwned, Deserialize};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::PathBuf,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelBinding {
    pub(super) path: PathBuf,
    pub(super) retained_full_sha256: String,
    pub(super) file_identity: FileIdentity,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    pub(super) model: ModelBinding,
    pub(super) full_content_receipt: FileBinding,
    pub(super) tokenizer: FileBinding,
    pub(super) template: FileBinding,
    pub(super) workload: FileBinding,
    pub(super) expected_exporter_source_commit: String,
    pub(super) expected_exporter_binary_sha256: String,
    pub(super) output: PathBuf,
}
fn exact_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn recipe_values(cfg: &Qwen35Config, work: &Workload) -> Result<(String, String, String, String)> {
    work.validate().map_err(anyhow::Error::msg)?;
    ensure!(
        matches!(
            (work.family, cfg.variant),
            (Family::Qwen38, Qwen35Variant::Dense) | (Family::Qwen36, Qwen35Variant::Moe)
        ) && cfg.vocab_size == work.vocab_size,
        "explicit Qwen vocabulary/family/variant"
    );
    ensure!(
        cfg.max_position_embeddings == 262144,
        "effective diagnostic context262144"
    );
    let debug = format!("{cfg:?}");
    ensure!(debug.len() <= 1024 * 1024, "config Debug bound");
    let config_debug_sha = hash(debug.as_bytes());
    let policy = super::super::super::qwen35_speculation::QwenSpeculationPolicy::Off;
    let cache_mode = format!("QwenTQ8:{policy:?}");
    let recipe =
        super::continuation::recipe(work.family, 262144, config_debug_sha.clone(), &cache_mode);
    let recipe_json = serde_json::to_string(&recipe)?;
    Ok((
        config_debug_sha,
        acquisition::config_digest(&recipe)?,
        acquisition::tokens_digest(work)?,
        recipe_json,
    ))
}
pub(super) fn output_preflight(output: &std::path::Path) -> Result<()> {
    ensure!(
        output.is_absolute() && !output.ancestors().any(|p| p.join(".git").exists()),
        "private output outside Git"
    );
    ensure!(
        output.parent().context("output parent")?.canonicalize()? == output.parent().unwrap(),
        "canonical output parent"
    );
    match std::fs::symlink_metadata(output) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        _ => anyhow::bail!("new output only"),
    }
}
#[test]
#[ignore = "root-reviewed bound header/JSON files only; no model/device execution"]
fn q1_qwen_continuation_metadata_packet_prepare() {
    run().unwrap();
}
/// Bounded config read from one explicit env var (1 MiB), raw bytes retained.
pub(super) fn read_config<T: DeserializeOwned>(var: &str) -> Result<(Vec<u8>, T)> {
    let path = PathBuf::from(std::env::var_os(var).context("explicit metadata config")?);
    let mut raw = Vec::new();
    File::open(path)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut raw)?;
    ensure!(raw.len() <= 1024 * 1024, "metadata config bound");
    let config = serde_json::from_slice(&raw)?;
    Ok((raw, config))
}
/// Every bound input of one Qwen metadata prepare, read once; header only.
pub(super) struct Bound {
    pub(super) source: &'static str,
    pub(super) binary: String,
    pub(super) full_receipt: Vec<u8>,
    pub(super) tokenizer: Vec<u8>,
    pub(super) template: Vec<u8>,
    pub(super) vectors: Vec<u8>,
    pub(super) work: Workload,
    pub(super) file: File,
    pub(super) before: FileIdentity,
    pub(super) header: mlx_native::gguf::GgufFile,
    /// `Qwen35Config::from_gguf` exactly as parsed; no context override.
    pub(super) header_cfg: Qwen35Config,
}
pub(super) fn bind(config: &Config) -> Result<Bound> {
    output_preflight(&config.output)?;
    ensure!(
        exact_hex(&config.expected_exporter_source_commit, 40)
            && exact_hex(&config.expected_exporter_binary_sha256, 64)
            && exact_hex(&config.model.retained_full_sha256, 64),
        "exact source/binary/model digests required"
    );
    let source = option_env!("HF2Q_BUILD_GIT_SHA").context("compiled source stamp required")?;
    ensure!(
        source == config.expected_exporter_source_commit,
        "exporter source differs"
    );
    let binary = acquisition::executable_digest()?;
    ensure!(
        binary == config.expected_exporter_binary_sha256,
        "actual exporter binary differs"
    );
    let full_receipt = acquisition::checked_file(&config.full_content_receipt, 16 * 1024 * 1024)?;
    acquisition::full_content_receipt_has_model(&full_receipt, &config.model.retained_full_sha256)?;
    let tokenizer = acquisition::checked_file(&config.tokenizer, 64 * 1024 * 1024)?;
    std::str::from_utf8(&tokenizer)
        .context("bound tokenizer snapshot must be UTF8; semantic attestation is external")?;
    let template = acquisition::checked_file(&config.template, 1024 * 1024)?;
    let template_text = std::str::from_utf8(&template)?;
    let vectors = acquisition::checked_file(&config.workload, 1024 * 1024)?;
    let work: Workload = serde_json::from_slice(&vectors)?;
    work.validate().map_err(anyhow::Error::msg)?;
    ensure!(
        config.model.path.is_absolute(),
        "absolute retained model path"
    );
    let file = File::open(&config.model.path)?;
    let before = FileIdentity::observed(&file)?;
    ensure!(
        before == config.model.file_identity,
        "actual model FD differs from retained file identity"
    );
    // ONLY header/metadata parser; this exact FD is cloned into the parser.
    // No tensor loading, weights, backend initialization, device or context.
    let header = mlx_native::gguf::GgufFile::from_file(file.try_clone()?)?;
    ensure!(
        header.tensor_data_offset() <= 32 * 1024 * 1024,
        "known selected GGUF header extent exceeds32MiB metadata packet"
    );
    let architecture = header
        .metadata_string("general.architecture")
        .context("GGUF architecture absent")?;
    ensure!(
        architecture
            == match work.family {
                Family::Qwen38 => "qwen35",
                Family::Qwen36 => "qwen35moe",
                _ => anyhow::bail!("unsupported family"),
            },
        "explicit architecture, no fallback"
    );
    ensure!(
        header.metadata_string("tokenizer.chat_template") == Some(template_text),
        "bound template is not exact embedded model template"
    );
    let header_cfg = Qwen35Config::from_gguf(&header)?;
    Ok(Bound {
        source,
        binary,
        full_receipt,
        tokenizer,
        template,
        vectors,
        work,
        file,
        before,
        header,
        header_cfg,
    })
}
impl Bound {
    pub(super) fn model_unchanged(&self, when: &str) -> Result<()> {
        ensure!(
            FileIdentity::observed(&self.file)? == self.before,
            "model FD changed {when}"
        );
        Ok(())
    }
    /// Recheck every bound small input after parse, then create the private
    /// output directory and write each file exclusively.
    pub(super) fn recheck_and_write(&self, config: &Config, files: &[(&str, &[u8])]) -> Result<()> {
        ensure!(
            acquisition::checked_file(&config.full_content_receipt, 16 * 1024 * 1024)?
                == self.full_receipt,
            "receipt changed"
        );
        ensure!(
            acquisition::checked_file(&config.tokenizer, 64 * 1024 * 1024)? == self.tokenizer,
            "tokenizer changed"
        );
        ensure!(
            acquisition::checked_file(&config.template, 1024 * 1024)? == self.template,
            "template changed"
        );
        ensure!(
            acquisition::checked_file(&config.workload, 1024 * 1024)? == self.vectors,
            "workload changed"
        );
        self.model_unchanged("before metadata output")?;
        output_preflight(&config.output)?;
        std::fs::create_dir(&config.output)?;
        std::fs::set_permissions(&config.output, std::fs::Permissions::from_mode(0o700))?;
        for (name, bytes) in files {
            let mut out = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(config.output.join(name))?;
            out.write_all(bytes)?;
        }
        Ok(())
    }
}
fn run() -> Result<()> {
    let (raw, config): (_, Config) = read_config("HF2Q_Q1_QWEN_PREP_CONFIG")?;
    let b = bind(&config)?;
    let (source, binary, work) = (b.source, &b.binary, &b.work);
    let (tokenizer, template, vectors, before, header) =
        (&b.tokenizer, &b.template, &b.vectors, &b.before, &b.header);
    let mut cfg = b.header_cfg.clone();
    cfg.max_position_embeddings = 262144;
    let (config_debug_sha256, config_sha256, token_vectors_sha256, recipe_json) =
        recipe_values(&cfg, work)?;
    b.model_unchanged("while reading metadata")?;
    // Retain bounded source-bound inputs for independent review. Hashes are
    // raw file identities, not a claimed HTTP semantic/tokenizer attestation.
    let value = serde_json::json!({"schema":"hf2q.q1.qwen-continuation-packet-preparation.v1","exporter_source_commit":source,"exporter_binary_sha256":binary,"native_version":env!("HF2Q_MLX_NATIVE_VERSION"),"native_checksum_verification":"external registry/build composition required","header_data_offset":header.tensor_data_offset(),"config_input_sha256":hash(&raw),"family":work.family,"model_path":config.model.path,"model_file_identity":before,"model_full_sha256_retained":config.model.retained_full_sha256,"model_payload_rehashed":false,"full_content_receipt_path":config.full_content_receipt.path,"full_content_receipt_sha256":config.full_content_receipt.sha256,"tokenizer_file_sha256":hash(tokenizer),"template_file_sha256":hash(template),"workload_file_sha256":hash(vectors),"token_vectors_sha256":token_vectors_sha256,"actual_qwen_config_debug":format!("{cfg:?}"),"actual_qwen_config_debug_sha256":config_debug_sha256,"config_sha256":config_sha256,"recipe":serde_json::from_str::<serde_json::Value>(&recipe_json)?,"configured_context":262144,"output_allowance":8192,"logical_lanes":4,"tokens_per_lane":144,"prospective_tail_scope":[64,144],"model_content_identity":"retained full-content receipt +actual FD stat join, not rehashed","tokenizer_semantic_attestation":false,"execution_approved":false,"composition_reviewed":false,"acquisition_attested":false,"model_loaded":false,"device_created":false,"context_created":false,"inference_performed":false,"whole_q1_qualified":false});
    let encoded = serde_json::to_vec_pretty(&value)?;
    ensure!(encoded.len() <= 2 * 1024 * 1024, "metadata output bound");
    b.recheck_and_write(&config, &[("prepared.json", encoded.as_slice())])?;
    println!("Q1_QWEN_CONTINUATION_METADATA_PREPARED family={:?} lanes=4 tokens=144 device=false acquisition=false execution_approved=false",work.family);
    Ok(())
}
#[test]
fn q1_metadata_preparation_uses_same_actual_recipe_and_le_vector_hash() {
    let mut cfg = crate::adr063_invariance::qwen_cache::selected_header_fixture(true);
    let work = Workload {
        family: Family::Qwen38,
        vocab_size: cfg.vocab_size,
        lanes: (0..4)
            .map(|lane| (0..144).map(|row| lane * 144 + row).collect())
            .collect(),
    };
    let (debug, recipe, tokens, json) = recipe_values(&cfg, &work).unwrap();
    let actual = super::continuation::recipe(
        work.family,
        262144,
        hash(format!("{cfg:?}").as_bytes()),
        "QwenTQ8:Off",
    );
    assert_eq!(debug, actual.actual_model_config_sha256);
    assert_eq!(recipe, acquisition::config_digest(&actual).unwrap());
    assert_eq!(json, serde_json::to_string(&actual).unwrap());
    assert_eq!(tokens, acquisition::tokens_digest(&work).unwrap());
    let mut wrong_family = work.clone();
    wrong_family.family = Family::Qwen36;
    assert!(recipe_values(&cfg, &wrong_family).is_err());
    cfg.vocab_size -= 1;
    assert!(recipe_values(&cfg, &work).is_err());
    cfg.vocab_size += 1;
    cfg.max_position_embeddings = 144;
    assert!(recipe_values(&cfg, &work).is_err());
}
#[test]
fn q1_metadata_preparation_refuses_existing_or_git_output_without_creation() {
    let owned = tempfile::tempdir().unwrap();
    let root = owned.path().canonicalize().unwrap();
    let out = root.join("prepared");
    assert!(output_preflight(&out).is_ok());
    assert!(!out.exists());
    std::fs::write(root.join(".git"), b"owned marker").unwrap();
    assert!(output_preflight(&out).is_err());
    std::fs::remove_file(root.join(".git")).unwrap();
    std::fs::create_dir(&out).unwrap();
    assert!(output_preflight(&out).is_err());
    assert!(output_preflight(std::path::Path::new("relative")).is_err());
}
