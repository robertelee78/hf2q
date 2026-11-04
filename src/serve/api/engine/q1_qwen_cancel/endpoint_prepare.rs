//! Qwen endpoint packet metadata for `qwen_actual` (Qwen3.8 and Qwen3.6).
//! Header/JSON only: no device, model load, context, inference or acquisition.
//! Emits the identity `config_sha256`, the complete `expected_cache` per
//! trajectory and one DISABLED packet body per trajectory. The supervisor
//! fills `acquisition`, `output` and `identity.native_checksum` and flips
//! `execution_approved`; until then every packet fails `qwen_actual` parsing.
use super::prepare::{self, Config};
use crate::adr063_invariance::{
    acquisition::{self, FileBinding},
    protocol::{Trajectory, Workload},
    qwen_checkpoint, qwen_localizer,
    receipt::{ExpectedTensor, Family},
    store::hash,
};
use crate::inference::models::qwen35::{Qwen35Config, Qwen35Variant};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use std::{
    io::Write,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::PathBuf,
};

/// Endpoint trajectories `qwen_actual` executes. T2 is a source refusal
/// (`matrix.rs`), T5/T6 are not endpoint adapters.
pub(super) const TRAJECTORIES: [Trajectory; 5] = [
    Trajectory::T0ColdSerial,
    Trajectory::T1Resume64,
    Trajectory::T3Chunk64,
    Trajectory::T4ReverseOrder,
    Trajectory::T7RepeatCold,
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EndpointConfig {
    common: Config,
    configured_capacity: u32,
    requested_payload_limit: u64,
    mtp_positions: [usize; 2],
    /// Optional `derivation.json` of `q1_qwen36_suffix144_workload_prepare`;
    /// when present it must name the bound workload file.
    workload_derivation: Option<FileBinding>,
    /// Optional stage-localizer packets; absent leaves every output unchanged.
    localizer: Option<LocalizerConfig>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalizerRoute {
    rerun_id: String,
    prefill_chunks: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalizerConfig {
    agent: usize,
    layers: Vec<usize>,
    operations: Vec<String>,
    normalize_rows: bool,
    routes: Vec<LocalizerRoute>,
}

pub(super) struct LocalizerPlan {
    pub(super) config_sha256: String,
    pub(super) recipe: serde_json::Value,
    /// `(rerun_id, prefill_chunks, manifest)` per route.
    pub(super) routes: Vec<(String, Vec<usize>, qwen_localizer::Manifest)>,
}

pub(super) struct EndpointPlan {
    pub(super) config_debug_sha256: String,
    pub(super) config_sha256: String,
    pub(super) token_vectors_sha256: String,
    pub(super) recipe: serde_json::Value,
    pub(super) packets: Vec<(Trajectory, Vec<ExpectedTensor>, u64)>,
}

/// Pure plan from the un-overridden header config: the same recipe and cache
/// manifest `qwen_actual` recomputes before model load.
pub(super) fn plan(
    header_cfg: &Qwen35Config,
    work: &Workload,
    capacity: u32,
    payload_limit: u64,
    mtp_positions: [usize; 2],
) -> Result<EndpointPlan> {
    work.validate().map_err(anyhow::Error::msg)?;
    ensure!(
        matches!(
            (work.family, header_cfg.variant),
            (Family::Qwen38, Qwen35Variant::Dense) | (Family::Qwen36, Qwen35Variant::Moe)
        ) && header_cfg.vocab_size == work.vocab_size,
        "explicit Qwen vocabulary/family/variant"
    );
    ensure!((144..=262144).contains(&capacity), "endpoint capacity 144..=262144");
    ensure!(payload_limit <= qwen_checkpoint::ENDPOINT_PAYLOAD_LIMIT_MAX, "endpoint payload limit <= 4 GiB");
    let recipe = qwen_checkpoint::endpoint_recipe(work.family, capacity, header_cfg);
    let mut packets = Vec::with_capacity(TRAJECTORIES.len());
    for trajectory in TRAJECTORIES {
        let cache = qwen_checkpoint::expected_cache(
            header_cfg,
            work.family,
            trajectory,
            capacity,
            mtp_positions,
        )?;
        ensure!(!cache.is_empty(), "complete active-cache manifest is mandatory");
        let planned = qwen_checkpoint::planned_payload(
            &cache,
            header_cfg.vocab_size as usize,
            payload_limit,
        )?;
        packets.push((trajectory, cache, planned));
    }
    Ok(EndpointPlan {
        config_debug_sha256: recipe.actual_model_config_sha256.clone(),
        config_sha256: acquisition::config_digest(&recipe)?,
        token_vectors_sha256: acquisition::tokens_digest(work)?,
        recipe: serde_json::to_value(&recipe)?,
        packets,
    })
}

/// Pure localizer plan from the un-overridden header config: the recipe
/// `qwen_localizer` checks and one manifest per route from its own builder.
/// Every route must freeze the identical `expected`, or `compare` refuses.
fn localizer_plan(
    header_cfg: &Qwen35Config,
    work: &Workload,
    capacity: u32,
    l: &LocalizerConfig,
) -> Result<LocalizerPlan> {
    ensure!((144..=262144).contains(&capacity), "localizer capacity 144..=262144");
    ensure!(!l.routes.is_empty() && l.routes.len() <= 8, "one to eight localizer routes");
    let mut routes: Vec<(String, Vec<usize>, qwen_localizer::Manifest)> = Vec::new();
    for r in &l.routes {
        ensure!(
            !r.rerun_id.is_empty()
                && r.rerun_id.len() <= 64
                && r.rerun_id.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
                && routes.iter().all(|(id, _, _)| *id != r.rerun_id),
            "distinct file-safe localizer rerun ids"
        );
        let manifest = qwen_localizer::localizer_manifest(
            header_cfg,
            l.agent,
            &r.prefill_chunks,
            &l.layers,
            &l.operations,
            l.normalize_rows,
        )
        .map_err(anyhow::Error::msg)?;
        if let Some((_, _, first)) = routes.first() {
            ensure!(
                first.1 == manifest.1,
                "localizer routes must freeze identical expected coordinates"
            );
        }
        routes.push((r.rerun_id.clone(), r.prefill_chunks.clone(), manifest));
    }
    let recipe = qwen_localizer::localizer_recipe(work.family, capacity, header_cfg);
    Ok(LocalizerPlan {
        config_sha256: acquisition::config_digest(&recipe)?,
        recipe: serde_json::to_value(&recipe)?,
        routes,
    })
}

/// `localizer-prepared.json` plus one DISABLED packet per route. The identity
/// is the endpoint identity with the stage-localizer `config_sha256`: the key
/// of a SEPARATE localizer composition, acquisition review and bindings file.
fn localizer_files(
    b: &prepare::Bound,
    common: &Config,
    raw: &[u8],
    capacity: u32,
    endpoint_identity: &serde_json::Value,
    l: &LocalizerConfig,
    lp: &LocalizerPlan,
) -> Result<Vec<(String, Vec<u8>)>> {
    let mut identity = endpoint_identity.clone();
    identity["config_sha256"] = serde_json::json!(lp.config_sha256);
    let mut files = Vec::new();
    let mut planned = serde_json::Map::new();
    for (rerun_id, chunks, (targets, expected, bytes)) in &lp.routes {
        let tokens = chunks.iter().sum::<usize>();
        planned.insert(rerun_id.clone(), serde_json::json!(bytes));
        let packet = serde_json::json!({
            "execution_approved": false,
            "acquisition": null,
            "identity": identity,
            "model": common.model.path,
            "output": null,
            "tokens": b.work.lanes[l.agent][..tokens],
            "workload": b.work,
            "agent": l.agent,
            "configured_capacity": capacity,
            "expected": expected,
            "targets": targets,
            "host_payload_limit": bytes,
            "rerun_id": rerun_id,
            "prefill_chunks": chunks,
        });
        let encoded = serde_json::to_vec_pretty(&packet)?;
        ensure!(
            encoded.len() + 65536 <= 8 * 1024 * 1024,
            "packet must fit the localizer 8 MiB read"
        );
        files.push((format!("localizer-packet-{rerun_id}.json"), encoded));
    }
    let value = serde_json::json!({
        "schema": "hf2q.q1.qwen-localizer-preparation.v1",
        "exporter_source_commit": b.source,
        "exporter_binary_sha256": b.binary,
        "native_version": env!("HF2Q_MLX_NATIVE_VERSION"),
        "config_input_sha256": hash(raw),
        "family": b.work.family,
        "model_path": common.model.path,
        "model_file_identity": b.before,
        "model_full_sha256_retained": common.model.retained_full_sha256,
        "header_data_offset": b.header.tensor_data_offset(),
        "header_qwen_config_debug": format!("{:?}", b.header_cfg),
        "header_config_context_override": false,
        "config_sha256": lp.config_sha256,
        "recipe": lp.recipe,
        "identity": identity,
        "comparison_key_scope": "SEPARATE stage-localizer composition/acquisition review/bindings; not the endpoint composition",
        "agent": l.agent,
        "layers": l.layers,
        "operations": l.operations,
        "normalize_rows": l.normalize_rows,
        "configured_capacity": capacity,
        "records_per_route": lp.routes[0].2 .1.len(),
        "planned_payload": planned,
        "host_payload_cap": qwen_localizer::HOST_PAYLOAD_CAP,
        "packets": files.iter().map(|(name, bytes)| serde_json::json!({
            "file": name, "sha256": hash(bytes)})).collect::<Vec<_>>(),
        "model_loaded": false,
        "device_created": false,
        "inference_performed": false,
        "acquisition_attested": false,
        "execution_approved": false,
        "composition_reviewed": false,
        "whole_q1_qualified": false,
    });
    let encoded = serde_json::to_vec_pretty(&value)?;
    ensure!(encoded.len() <= 2 * 1024 * 1024, "bounded localizer metadata");
    files.insert(0, ("localizer-prepared.json".into(), encoded));
    Ok(files)
}

#[test]
#[ignore = "root-reviewed bound header/JSON files only; no model/device execution"]
fn q1_qwen_endpoint_metadata_prepare() {
    run().unwrap();
}

fn run() -> Result<()> {
    let (raw, config): (_, EndpointConfig) =
        prepare::read_config("HF2Q_Q1_QWEN_ENDPOINT_PREP_CONFIG")?;
    let derivation = match &config.workload_derivation {
        Some(binding) => {
            let bytes = acquisition::checked_file(binding, 1024 * 1024)?;
            let value: serde_json::Value = serde_json::from_slice(&bytes)?;
            ensure!(
                value["schema"] == SUFFIX_SCHEMA
                    && value["workload_file_sha256"] == config.common.workload.sha256,
                "workload derivation must name the bound workload file"
            );
            Some((binding, value))
        }
        None => None,
    };
    let common = &config.common;
    let b = prepare::bind(common)?;
    let plan = plan(
        &b.header_cfg,
        &b.work,
        config.configured_capacity,
        config.requested_payload_limit,
        config.mtp_positions,
    )?;
    let localizer = match &config.localizer {
        Some(l) => {
            let lp = localizer_plan(&b.header_cfg, &b.work, config.configured_capacity, l)?;
            Some((l, lp))
        }
        None => None,
    };
    b.model_unchanged("while reading metadata")?;
    let identity = serde_json::json!({
        "family": b.work.family,
        "source_commit": b.source,
        "binary_sha256": b.binary,
        "native_version": env!("HF2Q_MLX_NATIVE_VERSION"),
        "native_checksum": null,
        "model_sha256": common.model.retained_full_sha256,
        "tokenizer_sha256": hash(&b.tokenizer),
        "template_sha256": hash(&b.template),
        "config_sha256": plan.config_sha256,
        "token_vectors_sha256": plan.token_vectors_sha256,
    });
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let mut planned = serde_json::Map::new();
    for (trajectory, cache, bytes) in &plan.packets {
        planned.insert(trajectory.label().into(), serde_json::json!(bytes));
        let packet = serde_json::json!({
            "execution_approved": false,
            "acquisition": null,
            "model": common.model.path,
            "output": null,
            "identity": identity,
            "workload": b.work,
            "trajectory": trajectory,
            "configured_capacity": config.configured_capacity,
            "requested_payload_limit": config.requested_payload_limit,
            "expected_cache": cache,
            "mtp_positions": config.mtp_positions,
        });
        let encoded = serde_json::to_vec_pretty(&packet)?;
        ensure!(encoded.len() + 65536 <= qwen_checkpoint::ENDPOINT_PACKET_READ_LIMIT,
            "packet must fit the qwen_actual packet read limit");
        files.push((format!("packet-{}.json", trajectory.label()), encoded));
    }
    let value = serde_json::json!({
        "schema": "hf2q.q1.qwen-endpoint-packet-preparation.v1",
        "exporter_source_commit": b.source,
        "exporter_binary_sha256": b.binary,
        "native_version": env!("HF2Q_MLX_NATIVE_VERSION"),
        "native_checksum_verification": "external registry/build composition required",
        "header_data_offset": b.header.tensor_data_offset(),
        "config_input_sha256": hash(&raw),
        "family": b.work.family,
        "model_path": common.model.path,
        "model_file_identity": b.before,
        "model_full_sha256_retained": common.model.retained_full_sha256,
        "model_payload_rehashed": false,
        "full_content_receipt_path": common.full_content_receipt.path,
        "full_content_receipt_sha256": common.full_content_receipt.sha256,
        "tokenizer_file_sha256": hash(&b.tokenizer),
        "template_file_sha256": hash(&b.template),
        "workload_file_sha256": hash(&b.vectors),
        "workload_derivation": derivation.as_ref().map(|(binding, value)| serde_json::json!({
            "path": binding.path, "sha256": binding.sha256,
            "label": value["label"], "decision": value["decision"]})),
        "token_vectors_sha256": plan.token_vectors_sha256,
        "header_qwen_config_debug": format!("{:?}", b.header_cfg),
        "header_qwen_config_debug_sha256": plan.config_debug_sha256,
        "header_config_context_override": false,
        "config_sha256": plan.config_sha256,
        "recipe": plan.recipe,
        "identity": identity,
        "configured_capacity": config.configured_capacity,
        "requested_payload_limit": config.requested_payload_limit,
        "mtp_positions": config.mtp_positions,
        "planned_payload": planned,
        "packets": files.iter().map(|(name, bytes)| serde_json::json!({
            "file": name, "sha256": hash(bytes)})).collect::<Vec<_>>(),
        "logical_lanes": 4,
        "tokens_per_lane": 144,
        "tokenizer_semantic_attestation": false,
        "execution_approved": false,
        "composition_reviewed": false,
        "acquisition_attested": false,
        "model_loaded": false,
        "device_created": false,
        "context_created": false,
        "inference_performed": false,
        "whole_q1_qualified": false,
    });
    let encoded = serde_json::to_vec_pretty(&value)?;
    ensure!(encoded.len() <= 2 * 1024 * 1024, "metadata output bound");
    let localizer_out = match &localizer {
        Some((l, lp)) => {
            localizer_files(&b, common, &raw, config.configured_capacity, &identity, l, lp)?
        }
        None => Vec::new(),
    };
    if let Some((binding, _)) = &derivation {
        // Re-read after parse like every other bound input.
        acquisition::checked_file(binding, 1024 * 1024)?;
    }
    let mut out: Vec<(&str, &[u8])> = vec![
        ("prepared.json", encoded.as_slice()),
        ("workload.json", b.vectors.as_slice()),
    ];
    out.extend(files.iter().map(|(name, bytes)| (name.as_str(), bytes.as_slice())));
    out.extend(localizer_out.iter().map(|(name, bytes)| (name.as_str(), bytes.as_slice())));
    b.recheck_and_write(common, &out)?;
    if let Some((l, lp)) = &localizer {
        println!(
            "Q1_QWEN_LOCALIZER_METADATA_PREPARED family={:?} routes={} agent={} records_per_route={} device=false execution_approved=false",
            b.work.family,
            lp.routes.len(),
            l.agent,
            lp.routes[0].2 .1.len()
        );
    }
    println!(
        "Q1_QWEN_ENDPOINT_METADATA_PREPARED family={:?} packets={} lanes=4 tokens=144 device=false execution_approved=false",
        b.work.family,
        plan.packets.len()
    );
    Ok(())
}

// ---- Qwen3.6 suffix-window workload (runbook decision D1, operator pending) ----

const SUFFIX_SCHEMA: &str = "hf2q.q1.qwen36-tool-round-suffix144-workload.v1";
const SUFFIX_LABEL: &str = "Qwen36ToolRoundSuffix144";
const SUFFIX_DECISION: &str = "D1 PENDING: operator/reviewer must accept the last-144-token windows \
of the four reviewed tool_round prompts as real agent prompts for Class B; otherwise Qwen3.6 (c) \
is diagnostic-only. Class A comparisons are valid either way.";

/// Last 144 ids of each reviewed rendered prompt, in round order. The full
/// prompts share their first 444-467 tokens, so whole-prompt prefixes cannot
/// form four distinct 144-token lanes; the suffix windows can.
pub(super) fn suffix144_workload(rounds: &[Vec<u32>], vocab_size: u32) -> Result<Workload> {
    ensure!(rounds.len() == 4, "exactly four tool_round prompts");
    let lanes = rounds
        .iter()
        .map(|tokens| {
            let start = tokens.len().checked_sub(144).context("tool_round prompt shorter than 144")?;
            Ok(tokens[start..].to_vec())
        })
        .collect::<Result<Vec<_>>>()?;
    let work = Workload {
        family: Family::Qwen36,
        lanes,
        vocab_size,
    };
    work.validate().map_err(anyhow::Error::msg)?;
    Ok(work)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SuffixConfig {
    /// `rendered-inputs-reviewed/qwen36-tool_round_{1,2,3,4}.tokens.json`.
    rounds: [FileBinding; 4],
    /// Header vocabulary; the endpoint prepare re-checks it against the GGUF.
    vocab_size: u32,
    output: PathBuf,
}

/// Explicit, labeled option for the Qwen3.6 endpoint workload. Writes
/// `workload.json` (bind it as `common.workload`) and `derivation.json`
/// (bind it as `workload_derivation`). No model file is opened.
#[test]
#[ignore = "root-reviewed bound token JSON only; operator decision D1 pending"]
fn q1_qwen36_suffix144_workload_prepare() {
    suffix_run().unwrap();
}

fn suffix_run() -> Result<()> {
    let (raw, config): (_, SuffixConfig) = prepare::read_config("HF2Q_Q1_QWEN36_SUFFIX_CONFIG")?;
    prepare::output_preflight(&config.output)?;
    let mut bound = Vec::with_capacity(4);
    let mut rounds = Vec::with_capacity(4);
    for binding in &config.rounds {
        let bytes = acquisition::checked_file(binding, 4 * 1024 * 1024)?;
        rounds.push(serde_json::from_slice::<Vec<u32>>(&bytes)?);
        bound.push(bytes);
    }
    let work = suffix144_workload(&rounds, config.vocab_size)?;
    let workload = serde_json::to_vec_pretty(&work)?;
    let derivation = serde_json::to_vec_pretty(&serde_json::json!({
        "schema": SUFFIX_SCHEMA,
        "label": SUFFIX_LABEL,
        "decision": SUFFIX_DECISION,
        "config_input_sha256": hash(&raw),
        "rounds": config.rounds.iter().zip(&rounds).map(|(b, tokens)| serde_json::json!({
            "path": b.path, "sha256": b.sha256, "prompt_tokens": tokens.len(),
            "window_start": tokens.len() - 144})).collect::<Vec<_>>(),
        "vocab_size": config.vocab_size,
        "workload_file_sha256": hash(&workload),
        "token_vectors_sha256": acquisition::tokens_digest(&work)?,
        "model_loaded": false,
    }))?;
    for (binding, old) in config.rounds.iter().zip(&bound) {
        ensure!(acquisition::checked_file(binding, 4 * 1024 * 1024)? == *old, "round changed");
    }
    prepare::output_preflight(&config.output)?;
    std::fs::create_dir(&config.output)?;
    std::fs::set_permissions(&config.output, std::fs::Permissions::from_mode(0o700))?;
    for (name, bytes) in [("workload.json", &workload), ("derivation.json", &derivation)] {
        std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(config.output.join(name))?
            .write_all(bytes)?;
    }
    println!("Q1_QWEN36_SUFFIX144_WORKLOAD_PREPARED lanes=4 tokens=144 label={SUFFIX_LABEL} decision=D1-pending");
    Ok(())
}

#[cfg(test)]
#[path = "endpoint_prepare_controls.rs"]
mod controls;
