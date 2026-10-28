//! Artifact-only continuation comparison; never loads a model or creates a device.
use super::{acquisition::{self, FileBinding}, compare, qwen_cache, qwen_continuation,
    receipt::*, store::hash};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use std::{collections::BTreeMap, fs::OpenOptions, io::Write,
    os::unix::fs::OpenOptionsExt, path::{Path, PathBuf}};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    schema: String,
    identity: Identity,
    prepared: FileBinding,
    cold: FileBinding,
    recovered: FileBinding,
    before64: FileBinding,
    restored64: FileBinding,
    output: PathBuf,
}
fn load(binding: &FileBinding) -> Result<(PathBuf, Trace)> {
    let bytes = acquisition::checked_file(binding,128*1024*1024)?;
    let root = binding.path.parent().context("trace parent")?.to_path_buf();
    ensure!(root.canonicalize()? == root,"canonical trace root");
    Ok((root,serde_json::from_slice(&bytes)?))
}
fn output(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("output parent")?;
    ensure!(path.is_absolute() && parent.canonicalize()? == parent &&
        !path.ancestors().any(|p|p.join(".git").exists()),"private canonical output outside Git");
    ensure!(bytes.len() <= 1024*1024,"bounded comparison metadata");
    let mut file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(path)?;
    file.write_all(bytes)?;
    Ok(())
}
fn manifest(trace: &Trace, identity: &Identity, expected: &[ExpectedTensor], trajectory: &str) -> Result<()> {
    ensure!(trace.identity == *identity && trace.coverage.trajectory == trajectory &&
        trace.coverage.expected == expected && trace.coverage.captured.len() == expected.len() &&
        trace.coverage.errors.is_empty(),"complete exact selected-family trajectory manifest");
    Ok(())
}
fn run(packet: Packet) -> Result<()> {
    ensure!(packet.schema == "hf2q.q1.continuation-artifact-comparison.v1" &&
        packet.identity.family == Family::Qwen38,"explicit selected Qwen38 comparison");
    packet.identity.validate().map_err(anyhow::Error::msg)?;
    let prepared: serde_json::Value = serde_json::from_slice(
        &acquisition::checked_file(&packet.prepared,2*1024*1024)?)?;
    // Reuse the existing selected header layout fixture only after byte-binding
    // its FULL Debug representation to the actual retained GGUF header config.
    let cfg = qwen_cache::selected_header_fixture(true);
    ensure!(prepared["actual_qwen_config_debug"] == format!("{cfg:?}") &&
        prepared["actual_qwen_config_debug_sha256"] == hash(format!("{cfg:?}").as_bytes()) &&
        prepared["config_sha256"] == packet.identity.config_sha256 &&
        prepared["token_vectors_sha256"] == packet.identity.token_vectors_sha256 &&
        prepared["exporter_source_commit"] == packet.identity.source_commit &&
        prepared["exporter_binary_sha256"] == packet.identity.binary_sha256,
        "actual selected header/recipe/vector/source binding");
    let (cold_root,cold) = load(&packet.cold)?;
    let (recovered_root,recovered) = load(&packet.recovered)?;
    let (before_root,before) = load(&packet.before64)?;
    let (restored_root,restored) = load(&packet.restored64)?;
    let expected = qwen_continuation::expected(&cfg,262144)?;
    let checkpoint = qwen_continuation::checkpoint_expected(&cfg,262144)?;
    manifest(&cold,&packet.identity,&expected,"T0Tail64To143")?;
    manifest(&recovered,&packet.identity,&expected,"T6Tail64To143")?;
    manifest(&before,&packet.identity,&checkpoint,"T6Checkpoint64")?;
    manifest(&restored,&packet.identity,&checkpoint,"T6ActualEngineRecovered64")?;
    let exact: BTreeMap<_,_> = checkpoint.iter().map(|e|(e.at.stage,NumericPolicy::Exact)).collect();
    let recovery = compare::compare_recovery(&before_root,&before,&restored_root,&restored,&exact);
    ensure!(recovery.disposition == Disposition::ExactEqual,"actual64 restore is not exact: {:?}",recovery.disposition);
    let unspecified = expected.iter().map(|e|(e.at.stage,NumericPolicy::Unspecified)).collect();
    let continuation = compare::compare(&cold_root,&cold,&recovered_root,&recovered,&unspecified);
    ensure!(!matches!(continuation.disposition,Disposition::InvalidEvidence |
        Disposition::IdentityMismatch | Disposition::LocalizationIncomplete),
        "invalid/incomplete comparison: {:?}",continuation);
    // Unspecified floats remain inconclusive even when bit-identical. Discrete
    // invariant divergence, if any, is retained verbatim in the actual report.
    let report = serde_json::json!({"schema":"hf2q.q1.continuation-artifact-report.v1",
        "identity":packet.identity,"recovery64":recovery,"continuation":continuation,
        "float_policy":"Unspecified","float_disposition":"ExplicitInconclusiveNumericPolicy",
        "logical_lanes":4,"physical_slots":1,"rows_per_lane":80,"row_scope":[64,144],
        "active_cache_boundaries":[128,144],"first64_logits_captured":false,
        "whole_q1_qualified":false,"performance_qualified":false,"physical_admission":false});
    output(&packet.output,&serde_json::to_vec_pretty(&report)?)?;
    println!("Q1_CONTINUATION_ARTIFACT_COMPARE_COMPLETE lanes=4 rows=80 recovery64=exact float_policy=Unspecified whole_q1=false");
    Ok(())
}
#[test]
#[ignore="explicit frozen artifact packet only; no model/device; root supervised"]
fn q1_continuation_artifact_only_compare() {
    let path = std::env::var_os("HF2Q_Q1_ARTIFACT_COMPARE_PACKET").expect("explicit artifact packet");
    let binding = PathBuf::from(path);
    let metadata = std::fs::symlink_metadata(&binding).unwrap();
    assert!(binding.is_absolute() && metadata.is_file() && !metadata.file_type().is_symlink() && metadata.len()<=1024*1024);
    let packet:Packet = serde_json::from_slice(&std::fs::read(binding).unwrap()).unwrap();
    run(packet).unwrap();
}
#[test]
fn private_output_refuses_git_or_existing_file() {
    let tmp=tempfile::tempdir().unwrap();let root=tmp.path().canonicalize().unwrap();
    std::fs::write(root.join(".git"),b"marker").unwrap();
    assert!(output(&root.join("report.json"),b"{}").is_err());
    std::fs::remove_file(root.join(".git")).unwrap();
    output(&root.join("report.json"),b"{}").unwrap();
    assert!(output(&root.join("report.json"),b"{}").is_err());
}
