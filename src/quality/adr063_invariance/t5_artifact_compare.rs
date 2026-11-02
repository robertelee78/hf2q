//! Artifact-only T5 comparison: committed trajectory versus committed trajectory.
//! Rejected verifier tail has no ordinary-baseline counterpart and is audited separately.
use super::{acquisition::{self, FileBinding}, compare, protocol::Workload,
    qwen_cache, qwen_speculation, receipt::*, store::{hash, PAGE_BYTES}};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs::OpenOptions, io::{Read, Write},
    os::unix::fs::OpenOptionsExt, path::{Path, PathBuf}};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    schema: String,
    identity: Identity,
    prepared: FileBinding,
    frozen: FileBinding,
    baseline: FileBinding,
    forced: FileBinding,
    raw_verifier: FileBinding,
    baseline_execution: FileBinding,
    forced_execution: FileBinding,
    output: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Carry { token: u32, terminal: bool }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Frozen { workload: Workload, final_carry: [Carry; 4], forced_bad_ids: [u32; 4] }
fn checked_file(binding: &FileBinding, max: u64) -> Result<Vec<u8>> {
    let m = std::fs::symlink_metadata(&binding.path)?;
    ensure!(binding.path.is_absolute() && binding.sha256.len() == 64 && m.is_file() &&
        !m.file_type().is_symlink() && m.len() <= max, "bound regular file extent");
    let mut bytes = Vec::new();
    std::fs::File::open(&binding.path)?.take(max+1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= max && hash(&bytes) == binding.sha256, "bound file hash");
    Ok(bytes)
}
fn value(binding: &FileBinding) -> Result<Value> {
    Ok(serde_json::from_slice(&checked_file(binding, 2*1024*1024)?)?)
}
fn load(binding: &FileBinding) -> Result<(PathBuf, Trace)> {
    let root = binding.path.parent().context("trace parent")?.to_path_buf();
    ensure!(root.canonicalize()? == root, "canonical trace root");
    Ok((root, serde_json::from_slice(&checked_file(binding,128*1024*1024)?)?))
}
fn output(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("output parent")?;
    ensure!(path.is_absolute() && parent.canonicalize()? == parent &&
        !path.ancestors().any(|p|p.join(".git").exists()), "private canonical output outside Git");
    ensure!(bytes.len() <= 1024*1024, "bounded report");
    OpenOptions::new().write(true).create_new(true).mode(0o600).open(path)?.write_all(bytes)?;
    Ok(())
}
fn manifest(trace: &Trace, identity: &Identity, expected: &[ExpectedTensor], name: &str,
    root: &Path) -> Result<()> {
    ensure!(trace.schema == "hf2q.adr063.invariance.v2" && trace.identity == *identity &&
        trace.coverage.trajectory == name && trace.coverage.expected == expected &&
        trace.coverage.captured.len() == expected.len() && trace.coverage.errors.is_empty(),
        "complete selected-family manifest");
    let a = trace.acquisition.as_ref().context("actual acquisition missing")?;
    ensure!(a.required_model_opens == 2 && a.observed_model_opens >= 2 &&
        a.actual_binary_sha256 == identity.binary_sha256 && a.output_root == root,
        "direct Qwen actual two-open acquisition/root binding");
    Ok(())
}
fn execution(v: &Value, forced: bool, frozen: &Frozen, input_hash: &str) -> Result<()> {
    ensure!(v["schema"] == "hf2q.q1.qwen-t5-phase.v1" &&
        v["status"] == "CompleteScopedAcquisition" && v["error"].is_null() &&
        v["input_sha256"] == input_hash && v["forced_proposals"] == forced &&
        v["mode"] == if forced { "ForcedProposal" } else { "OrdinaryBaseline" },
        "actual completed phase/input/mode");
    let selected = v["canonical_decisions"].as_array().context("canonical lanes")?;
    let rows = v["ordinary_input_positions"].as_array().context("ordinary rows")?;
    ensure!(selected.len() == 4 && rows.len() == 4, "four complete lanes");
    for lane in 0..4 {
        let actual = selected[lane].as_array().context("17 actual decisions")?;
        ensure!(actual.len() == 17, "all17 decisions required");
        for (i, decision) in actual.iter().enumerate() {
            let (token, terminal) = if i < 16 {
                (frozen.workload.lanes[lane][128+i], false)
            } else { (frozen.final_carry[lane].token, frozen.final_carry[lane].terminal) };
            ensure!(*decision == json!({"token":token,"terminal":terminal}), "frozen decision lineage");
        }
        let expected: Vec<usize> = if forced { std::iter::once(128).chain(131..144).collect() }
            else { (128..144).collect() };
        ensure!(rows[lane] == json!(expected), "actual transaction positions");
    }
    let transitions = v["actual_transitions"].as_array().context("transitions")?;
    ensure!(transitions.len() == if forced {4} else {0}, "four actual K3 rounds only");
    for (lane, t) in transitions.iter().enumerate() {
        ensure!(*t == json!({"lane":lane,"entry":129,"matched":1,"rejected":1,
            "valid_inputs":2,"carry_row":1,"committed":131,
            "actual_output":frozen.workload.lanes[lane][130..132]}), "actual accept/reject/rollback");
    }
    Ok(())
}
// Small raw-only payload audit. Existing compare() remains the authoritative
// paired committed comparator. No synthetic baseline for rejected verifier rows.
fn raw_audit(root: &Path, trace: &Trace, frozen: &Frozen) -> Result<Value> {
    let mut seen = BTreeSet::new();
    let mut ids = std::collections::BTreeMap::new();
    let mut bytes = 0u64;
    for t in &trace.coverage.captured {
        ensure!(seen.insert(t.at.clone()) && trace.coverage.expected.iter().any(|e|
            e.at == t.at && e.scalar == t.scalar && e.shape == t.shape), "raw duplicate/shape drift");
        let size = t.shape.iter().try_fold(t.scalar.bytes(), |a,&b|a.checked_mul(b))
            .context("raw extent overflow")?;
        ensure!(size as u64 == t.logical_bytes && t.parent_identity != 0 &&
            t.parent_offset.checked_add(t.logical_bytes).is_some_and(|n|n <= t.parent_bytes) &&
            !t.checked_drain.is_empty() && Path::new(&t.file).components().count() == 1 &&
            t.file.starts_with("tensor-") && t.file.ends_with(".bin"), "raw extent/drain/path");
        let b = checked_file(&FileBinding {path:root.join(&t.file),sha256:t.sha256.clone()},
            4*1024*1024)?;
        ensure!(b.len() == size && t.pages.len() == b.len().div_ceil(PAGE_BYTES), "raw full payload/pages");
        for (i, (page, data)) in t.pages.iter().zip(b.chunks(PAGE_BYTES)).enumerate() {
            ensure!(page.offset == (i*PAGE_BYTES) as u64 && page.bytes == data.len() &&
                page.sha256 == hash(data), "raw contiguous full page hash");
        }
        match t.scalar {
            Scalar::F32 => ensure!(b.chunks_exact(4).all(|v|
                f32::from_le_bytes(v.try_into().unwrap()).is_finite()), "nonfinite raw verifier value"),
            Scalar::U32 => { ids.insert((t.at.agent.clone(),t.at.stage), b.chunks_exact(4)
                .map(|v|u32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>()); },
            _ => anyhow::bail!("unexpected raw scalar"),
        }
        bytes += b.len() as u64;
    }
    ensure!(seen.len() == 20, "all20 raw tensors");
    for lane in 0..4 {
        let get = |stage| ids.get(&(format!("agent-{lane}"),stage)).context("raw IDs missing");
        let original = get(Stage::SpecOriginalDraftIds)?;
        let forced = get(Stage::SpecForcedDraftIds)?;
        let canonical = get(Stage::SpecCanonicalIds)?;
        ensure!(original.len() == 3 && original.iter().all(|&id|id < frozen.workload.vocab_size) &&
            *forced == vec![frozen.workload.lanes[lane][130],frozen.forced_bad_ids[lane],original[2]] &&
            frozen.forced_bad_ids[lane] < frozen.workload.vocab_size &&
            frozen.forced_bad_ids[lane] != frozen.workload.lanes[lane][131] &&
            *canonical == frozen.workload.lanes[lane][130..132], "raw proposal/canonical identity");
    }
    Ok(json!({"records":20,"verifier_rows_per_lane":4,"payload_bytes":bytes,
        "all_payload_hashes_and_pages_verified":true,"all_f32_finite":true,
        "matching_baseline_claim":false,"rejected_tail_compared_to_baseline":false}))
}
fn run(packet: Packet) -> Result<()> {
    ensure!(packet.schema == "hf2q.q1.t5-artifact-comparison.v1" &&
        packet.identity.family == Family::Qwen38, "explicit selected Q38 T5 artifact packet");
    packet.identity.validate().map_err(anyhow::Error::msg)?;
    let prepared = value(&packet.prepared)?;
    let cfg = qwen_cache::selected_header_fixture(true);
    ensure!(prepared["actual_qwen_config_debug"] == format!("{cfg:?}") &&
        prepared["actual_qwen_config_debug_sha256"] == hash(format!("{cfg:?}").as_bytes()),
        "actual entire header config must match manifest fixture");
    let recipe = acquisition::ConfigRecipe {family: Family::Qwen38, capacity:262144,
        slots:1, output_allowance:8192, cache_mode:"QwenTQ8:MTP-CoherentAuto",
        route:"qwen-t5-explicit-MTP-K3-forced-proposal-v1",
        actual_model_config_sha256:hash(format!("{cfg:?}").as_bytes())};
    ensure!(acquisition::config_digest(&recipe)? == packet.identity.config_sha256,
        "actual source-defined T5 configuration recipe");
    let frozen: Frozen = serde_json::from_slice(&checked_file(&packet.frozen,1024*1024)?)?;
    frozen.workload.validate().map_err(anyhow::Error::msg)?;
    ensure!(frozen.workload.family == Family::Qwen38 && frozen.workload.vocab_size == cfg.vocab_size &&
        acquisition::tokens_digest(&frozen.workload)? == packet.identity.token_vectors_sha256,
        "actual frozen full144 token digest/family");
    let (br,b) = load(&packet.baseline)?;
    let (fr,f) = load(&packet.forced)?;
    let (rr,r) = load(&packet.raw_verifier)?;
    let expected = qwen_speculation::committed(&cfg,262144)?;
    manifest(&b,&packet.identity,&expected,"QwenT5:OrdinaryBaseline:committed",&br)?;
    manifest(&f,&packet.identity,&expected,"QwenT5:ForcedProposal:committed",&fr)?;
    manifest(&r,&packet.identity,&qwen_speculation::raw(&cfg),"QwenT5:ForcedProposal:raw-verifier",&rr)?;
    let be = value(&packet.baseline_execution)?;
    let fe = value(&packet.forced_execution)?;
    execution(&be,false,&frozen,&packet.frozen.sha256)?;
    execution(&fe,true,&frozen,&packet.frozen.sha256)?;
    ensure!(be["packet_sha256"] == b.acquisition.as_ref().unwrap().packet_sha256 &&
        fe["packet_sha256"] == f.acquisition.as_ref().unwrap().packet_sha256 &&
        fe["packet_sha256"] == r.acquisition.as_ref().unwrap().packet_sha256,
        "raw/committed/phase acquisition join");
    let mut ra = serde_json::to_value(r.acquisition.as_ref().unwrap())?;
    let mut fa = serde_json::to_value(f.acquisition.as_ref().unwrap())?;
    ra.as_object_mut().unwrap().remove("output_root");
    fa.as_object_mut().unwrap().remove("output_root");
    ensure!(ra == fa && r.comparison_key == f.comparison_key, "same actual forced acquisition");
    let raw = raw_audit(&rr,&r,&frozen)?;
    let policies = expected.iter().map(|e|(e.at.stage,NumericPolicy::Unspecified)).collect();
    let committed = compare::compare(&br,&b,&fr,&f,&policies);
    ensure!(!matches!(committed.disposition,Disposition::InvalidEvidence |
        Disposition::IdentityMismatch | Disposition::LocalizationIncomplete), "invalid committed evidence: {:?}",committed);
    output(&packet.output,&serde_json::to_vec_pretty(&json!({
        "schema":"hf2q.q1.t5-artifact-report.v1","identity":packet.identity,
        "committed":committed,"raw_verifier_audit":raw,"logical_lanes":4,"physical_slots":1,
        "committed_rows":[128,144],"cache_checkpoints":[129,131,144],
        "float_policy":"Unspecified","float_disposition":"ExplicitInconclusiveNumericPolicy",
        "packed_cache_byte_invariants_retained":true,"ordinary_AUTO_selection_qualified":false,
        "whole_q1_qualified":false,"performance_qualified":false,"physical_admission":false}))?)?;
    println!("Q1_T5_ARTIFACT_COMPARE_COMPLETE lanes=4 committed_rows=16 raw_records=20 float_policy=Unspecified whole_q1=false");
    Ok(())
}
#[test]
#[ignore="explicit frozen artifact packet only; no model/device; root supervised"]
fn q1_t5_artifact_only_compare() {
    let p = PathBuf::from(std::env::var_os("HF2Q_Q1_T5_ARTIFACT_COMPARE_PACKET").expect("explicit packet"));
    let m = std::fs::symlink_metadata(&p).unwrap();
    assert!(p.is_absolute() && m.is_file() && !m.file_type().is_symlink() && m.len() <= 1024*1024);
    run(serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()).unwrap();
}
#[test]
fn private_output_refuses_git_and_overwrite() {
    let temp = tempfile::tempdir().unwrap(); let root = temp.path().canonicalize().unwrap();
    std::fs::write(root.join(".git"),b"marker").unwrap();
    assert!(output(&root.join("report.json"),b"{}").is_err());
    std::fs::remove_file(root.join(".git")).unwrap();
    output(&root.join("report.json"),b"{}").unwrap();
    assert!(output(&root.join("report.json"),b"{}").is_err());
}

#[test]
fn phase_gate_rejects_wrong_cursor_or_missing_decision() {
    let frozen = Frozen { workload:Workload {family:Family::Qwen38,vocab_size:100,
        lanes:vec![vec![7;144];4]}, final_carry:std::array::from_fn(|_|Carry {token:8,terminal:false}),
        forced_bad_ids:[0;4] };
    let decisions:Vec<Value> = (0..17).map(|i|json!({"token":if i<16 {7}else{8},"terminal":false})).collect();
    let rows:Vec<usize> = std::iter::once(128).chain(131..144).collect();
    let transitions:Vec<Value> = (0..4).map(|lane|json!({"lane":lane,"entry":129,"matched":1,
        "rejected":1,"valid_inputs":2,"carry_row":1,"committed":131,"actual_output":[7,7]})).collect();
    let good=json!({"schema":"hf2q.q1.qwen-t5-phase.v1","mode":"ForcedProposal",
        "status":"CompleteScopedAcquisition","error":null,"input_sha256":"bound",
        "forced_proposals":true,"canonical_decisions":[decisions.clone(),decisions.clone(),decisions.clone(),decisions],
        "ordinary_input_positions":[rows.clone(),rows.clone(),rows.clone(),rows],"actual_transitions":transitions});
    assert!(execution(&good,true,&frozen,"bound").is_ok());
    let mut bad=good.clone();bad["actual_transitions"][0]["committed"]=json!(132);
    assert!(execution(&bad,true,&frozen,"bound").is_err());
    let mut bad=good;bad["canonical_decisions"][1].as_array_mut().unwrap().pop();
    assert!(execution(&bad,true,&frozen,"bound").is_err());
}

#[path = "t5_artifact_compare/raw_controls.rs"]
mod raw_controls;
