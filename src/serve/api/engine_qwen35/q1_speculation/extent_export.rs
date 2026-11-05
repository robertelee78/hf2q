//! No-model Qwen3.6 (#653) extent export: the supervisor `manifest` files of
//! every Q36 multistore case and the T5-A2 pre-fork scope, generated from the
//! source manifest functions on the GGUF header config only. No tensor load,
//! device, context or inference.
//!
//! Manifest file (supervisor manifest form): exactly
//! `{"lanes":[..],"host_payload_limit":Σ planned store payload,"expected_stores":{name:[ExpectedTensor]}}`,
//! store name = the runner's `output/<name>` directory, list = the exact
//! `expected` the runner passes to `Store::new` (so its trace.json `expected`).
//! Scope file: a closed JSON list of `q1_decision_policy` keys
//! `[stage, agent, logical_row_start, layer, segment, logical_step]`.
use super::*;
use crate::adr063_invariance::{
    acquisition::{self, FileIdentity},
    qwen_continuation::{self as tail, payload},
    store::hash,
};
use crate::inference::models::qwen35::Qwen35Variant;
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};
const CAPACITY: u32 = 262144;
pub(super) const SCOPE_FILE: &str = "t5-prefork-scope.json";
/// Pre-registered Q36 extents (design §1.4/§1.5); the export refuses a
/// header whose source manifests differ.
const COMMITTED: usize = 1840;
const RAW: usize = 20;
const PREFORK: usize = 596;
const CONTINUATION: usize = 1440;
const SNAPSHOT: usize = 560;
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Model {
    path: PathBuf,
    file_identity: FileIdentity,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    schema: String,
    model: Model,
    output: PathBuf,
}
type Stores = Vec<(&'static str, Vec<ExpectedTensor>)>;
/// One supervisor manifest; `host_payload_limit` is the exact Σ planned payload.
fn manifest(lanes: &[usize], stores: Stores) -> Result<Value> {
    let mut total = 0u64;
    let mut map = serde_json::Map::new();
    for (name, expected) in stores {
        total = total.checked_add(payload(&expected)?).context("manifest payload overflow")?;
        ensure!(map.insert(name.into(), serde_json::to_value(&expected)?).is_none(), "duplicate store");
    }
    Ok(json!({"lanes":lanes,"host_payload_limit":total,"expected_stores":map}))
}
/// First-fork stores exactly as the runner (`committed`, `raw-verifier`) and
/// `first_fork::make` (`post-draft-discard129`, `target-before/after-mtp`) create them.
fn first_fork_stores(cfg: &Qwen35Config, forced: bool) -> Result<Stores> {
    let mut stores = vec![("committed", first_fork::committed(cfg)?)];
    if forced {
        stores.push(("raw-verifier", first_fork::raw(cfg)));
        stores.push(("post-draft-discard129", first_fork::draft_plan(cfg)?));
    }
    stores.push(("target-before-mtp", first_fork::output_plan(cfg)));
    stores.push(("target-after-mtp", first_fork::output_plan(cfg)));
    Ok(stores)
}
/// Every Q36 case manifest: file name → manifest value.
pub(super) fn manifests(cfg: &Qwen35Config) -> Result<Vec<(&'static str, Value)>> {
    let four = [0, 1, 2, 3];
    let committed = plan::committed(cfg, CAPACITY)?;
    let continuation = tail::expected(cfg, CAPACITY)?;
    let snapshot = tail::checkpoint_expected(cfg, CAPACITY)?;
    ensure!(committed.len() == COMMITTED && plan::raw(cfg).len() == RAW &&
        continuation.len() == CONTINUATION && snapshot.len() == SNAPSHOT,
        "Q36 header source manifests differ from the pre-registered 1840/20/1440/560 extents");
    Ok(vec![
        ("t5-preparation-manifest.json", manifest(&four, vec![])?),
        ("t5-baseline-manifest.json", manifest(&four, vec![("committed", committed.clone())])?),
        ("t5-forced-manifest.json", manifest(&four, vec![("committed", committed),
            ("raw-verifier", plan::raw(cfg))])?),
        ("t5-first-fork-baseline-manifest.json", manifest(&[0], first_fork_stores(cfg, false)?)?),
        ("t5-first-fork-forced-manifest.json", manifest(&[0], first_fork_stores(cfg, true)?)?),
        ("t6-cancelled-then-recovered-manifest.json", manifest(&four, vec![("continuation", continuation.clone()),
            ("before", snapshot.clone()), ("recovered", snapshot.clone())])?),
        ("t3cm-cold-chunk-matched64-manifest.json", manifest(&four, vec![("continuation", continuation),
            ("prefix64", snapshot)])?),
    ])
}
/// T5-A2 pre-fork scope: committed Logits row 128, or any non-Logits record at
/// logical_step 129 (both phases ran identical calls up to there).
pub(super) fn prefork_scope(cfg: &Qwen35Config) -> Result<Vec<Value>> {
    let scope: Vec<Value> = plan::committed(cfg, CAPACITY)?.into_iter()
        .filter(|e| if e.at.stage == Stage::Logits {e.at.logical_row_start == 128} else {e.at.logical_step == 129})
        .map(|e| json!([e.at.stage, e.at.agent, e.at.logical_row_start, e.at.layer, e.at.segment, e.at.logical_step]))
        .collect();
    ensure!(scope.len() == PREFORK, "T5-A2 scope differs from the pre-registered 596 keys");
    Ok(scope)
}
/// All output files (name, bytes): manifests, the scope and `extent-export.json`.
fn render(cfg: &Qwen35Config, binary: &str) -> Result<Vec<(String, Vec<u8>)>> {
    let mut files = Vec::new();
    let mut index = Vec::new();
    for (name, value) in manifests(cfg)? {
        let bytes = serde_json::to_vec_pretty(&value)?;
        ensure!(bytes.len() <= 64 * 1024 * 1024, "manifest file bound");
        let stores: serde_json::Map<String, Value> = value["expected_stores"].as_object().context("stores")?
            .iter().map(|(k, v)| (k.clone(), json!(v.as_array().map_or(0, Vec::len)))).collect();
        index.push(json!({"file":name,"sha256":hash(&bytes),"lanes":value["lanes"],
            "host_payload_limit":value["host_payload_limit"],"store_records":stores}));
        files.push((name.to_string(), bytes));
    }
    let scope = serde_json::to_vec_pretty(&prefork_scope(cfg)?)?;
    let debug = format!("{cfg:?}");
    let summary = json!({"schema":"hf2q.q1.qwen36-extent-export.v1","family":Family::Qwen36,
        "exporter_source_commit":option_env!("HF2Q_BUILD_GIT_SHA"),
        "exporter_binary_sha256":binary,
        "capacity":CAPACITY,"header_config_debug_sha256":hash(debug.as_bytes()),
        "header_config_equals_hosted_fixture":debug == format!("{:?}", qwen_cache::selected_header_fixture(false)),
        "manifests":index,"prefork_scope":{"file":SCOPE_FILE,"sha256":hash(&scope),"keys":PREFORK,
            "key_form":["stage","agent","logical_row_start","layer","segment","logical_step"],
            "predicate":"committed(cfg,262144): Logits and row 128, or not Logits and logical_step 129"},
        "model_loaded":false,"device_created":false,"inference_performed":false,
        "execution_approved":false,"whole_q1_qualified":false});
    files.push((SCOPE_FILE.into(), scope));
    files.push(("extent-export.json".into(), serde_json::to_vec_pretty(&summary)?));
    Ok(files)
}
fn output_preflight(path: &Path) -> Result<()> {
    ensure!(path.is_absolute()
        && path.parent().context("output parent")?.canonicalize()? == path.parent().unwrap()
        && !path.ancestors().any(|p| p.join(".git").exists()),
        "new canonical private output outside Git");
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        _ => anyhow::bail!("output exists/uncheckable"),
    }
}
#[test]
#[ignore = "root-reviewed Qwen3.6 GGUF header only; no model load/device/inference"]
fn q1_qwen36_extent_manifest_export() {
    run().unwrap();
}
fn run() -> Result<()> {
    let path = PathBuf::from(std::env::var_os("HF2Q_Q1_QWEN36_EXTENT_PACKET").context("explicit export packet")?);
    let mut raw = Vec::new();
    File::open(path)?.take(1024 * 1024 + 1).read_to_end(&mut raw)?;
    ensure!(raw.len() <= 1024 * 1024, "packet bound");
    let p: Packet = serde_json::from_slice(&raw)?;
    ensure!(p.schema == "hf2q.q1.qwen36-extent-export-packet.v1", "explicit Q36 extent export packet");
    output_preflight(&p.output)?;
    ensure!(p.model.path.is_absolute(), "absolute retained model path");
    let file = File::open(&p.model.path)?;
    ensure!(FileIdentity::observed(&file)? == p.model.file_identity, "model FD differs from retained file identity");
    // Header/metadata parser only, on this exact FD.
    let header = mlx_native::gguf::GgufFile::from_file(file.try_clone()?)?;
    ensure!(header.tensor_data_offset() <= 32 * 1024 * 1024, "selected GGUF header extent");
    ensure!(header.metadata_string("general.architecture") == Some("qwen35moe"), "explicit Qwen3.6 architecture");
    let mut cfg = Qwen35Config::from_gguf(&header)?;
    cfg.max_position_embeddings = CAPACITY;
    ensure!(matches!(cfg.variant, Qwen35Variant::Moe) && cfg.mtp_num_hidden_layers == 1, "Qwen3.6 header with one MTP layer");
    drop(header);
    let files = render(&cfg, &acquisition::executable_digest()?)?;
    ensure!(FileIdentity::observed(&file)? == p.model.file_identity, "model FD changed during export");
    output_preflight(&p.output)?;
    std::fs::create_dir(&p.output)?;
    std::fs::set_permissions(&p.output, std::fs::Permissions::from_mode(0o700))?;
    for (name, bytes) in &files {
        OpenOptions::new().create_new(true).write(true).mode(0o600).open(p.output.join(name))?.write_all(bytes)?;
    }
    println!("Q1_QWEN36_EXTENT_EXPORTED manifests=7 prefork_scope={PREFORK} model=false device=false execution_approved=false");
    Ok(())
}
#[test]
fn q1_qwen36_extent_manifests_match_source_counts_and_payload() {
    let cfg = qwen_cache::selected_header_fixture(false);
    let all: std::collections::BTreeMap<_, _> = manifests(&cfg).unwrap().into_iter().collect();
    let count = |file: &str, store: &str| all[file]["expected_stores"][store].as_array().map(Vec::len);
    let host = |file: &str| all[file]["host_payload_limit"].as_u64().unwrap();
    for file in all.keys() {
        let keys: std::collections::BTreeSet<_> = all[file].as_object().unwrap().keys().map(String::as_str).collect();
        let keys: Vec<_> = keys.into_iter().collect();
        assert_eq!(keys, ["expected_stores", "host_payload_limit", "lanes"], "closed manifest form");
    }
    assert_eq!(all["t5-preparation-manifest.json"], json!({"lanes":[0,1,2,3],"host_payload_limit":0,"expected_stores":{}}));
    assert_eq!(count("t5-baseline-manifest.json", "committed"), Some(1840));
    assert_eq!(count("t5-baseline-manifest.json", "raw-verifier"), None);
    assert_eq!(host("t5-baseline-manifest.json"), 872_421_120);
    assert_eq!(count("t5-forced-manifest.json", "committed"), Some(1840));
    assert_eq!(count("t5-forced-manifest.json", "raw-verifier"), Some(20));
    assert_eq!(host("t5-forced-manifest.json"), 872_421_120 + 16_023_680);
    assert_eq!(host("t5-baseline-manifest.json") + host("t5-forced-manifest.json"), 1_760_865_920);
    // The exact lists the runner hands Store::new (trace.json `expected`).
    let parsed: Vec<ExpectedTensor> = serde_json::from_value(all["t5-forced-manifest.json"]["expected_stores"]["raw-verifier"].clone()).unwrap();
    assert_eq!(parsed, plan::raw(&cfg));
    let parsed: Vec<ExpectedTensor> = serde_json::from_value(all["t5-baseline-manifest.json"]["expected_stores"]["committed"].clone()).unwrap();
    assert_eq!(parsed, plan::committed(&cfg, CAPACITY).unwrap());
    for (file, forced) in [("t5-first-fork-baseline-manifest.json", false), ("t5-first-fork-forced-manifest.json", true)] {
        assert_eq!(all[file]["lanes"], json!([0]));
        assert_eq!(host(file), first_fork::payload_plan(&cfg, forced).unwrap());
        assert_eq!(count(file, "committed"), Some(299));
        assert_eq!(count(file, "target-before-mtp"), Some(4));
        assert_eq!(count(file, "target-after-mtp"), Some(4));
        assert_eq!(count(file, "raw-verifier"), forced.then_some(5));
        assert_eq!(count(file, "post-draft-discard129"), forced.then_some(148));
        assert_eq!(all[file]["expected_stores"].as_object().unwrap().len(), if forced {5} else {3});
    }
    assert_eq!(host("t5-first-fork-baseline-manifest.json"), 141_687_488);
    assert_eq!(host("t5-first-fork-forced-manifest.json"), 211_029_904);
    let t6 = "t6-cancelled-then-recovered-manifest.json";
    assert_eq!((count(t6, "continuation"), count(t6, "before"), count(t6, "recovered")), (Some(1440), Some(560), Some(560)));
    assert_eq!(host(t6), 856_074_240 + 2 * 266_117_120);
    let t3cm = "t3cm-cold-chunk-matched64-manifest.json";
    assert_eq!((count(t3cm, "continuation"), count(t3cm, "prefix64"), count(t3cm, "before")), (Some(1440), Some(560), None));
    assert_eq!(host(t3cm), 856_074_240 + 266_117_120);
    for file in [t6, t3cm] {
        let parsed: Vec<ExpectedTensor> = serde_json::from_value(all[file]["expected_stores"]["continuation"].clone()).unwrap();
        assert_eq!(parsed, tail::expected(&cfg, CAPACITY).unwrap());
    }
    let rendered = render(&cfg, &"0".repeat(64)).unwrap();
    assert_eq!(rendered.len(), 9);
    assert!(rendered.iter().all(|(_, bytes)| bytes.len() <= 64 * 1024 * 1024));
}
#[test]
fn q1_qwen36_t5_prefork_scope_is_row128_logits_and_step129_caches() {
    let cfg = qwen_cache::selected_header_fixture(false);
    let scope = prefork_scope(&cfg).unwrap();
    assert_eq!(scope.len(), 596);
    let unique: std::collections::BTreeSet<String> = scope.iter().map(Value::to_string).collect();
    assert_eq!(unique.len(), 596, "duplicate scope key");
    let logits: Vec<_> = scope.iter().filter(|k| k[0] == "Logits").collect();
    assert_eq!(logits.len(), 4);
    assert!(logits.iter().all(|k| k[2] == 128 && k[3] == cfg.num_hidden_layers && k[4] == 0 && k[5] == 0));
    assert!(scope.iter().filter(|k| k[0] != "Logits").all(|k| k[5] == 129 && k.as_array().unwrap().len() == 6));
    for lane in 0..4 {
        let agent = format!("agent-{lane}");
        assert_eq!(scope.iter().filter(|k| k[1] == agent.as_str() && k[0] != "Logits").count(), 148);
    }
    // Every scope key names a committed record of both T5 phases.
    let committed: std::collections::BTreeSet<String> = plan::committed(&cfg, CAPACITY).unwrap().iter()
        .map(|e| json!([e.at.stage, e.at.agent, e.at.logical_row_start, e.at.layer, e.at.segment, e.at.logical_step]).to_string())
        .collect();
    assert!(unique.is_subset(&committed));
    assert!(prefork_scope(&qwen_cache::selected_header_fixture(true)).is_err(), "Q38 extent is not the Q36 scope");
}
