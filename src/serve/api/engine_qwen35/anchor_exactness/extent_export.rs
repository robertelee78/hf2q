//! Header-only expected extents. Calls and terminal cursors are an independent
//! root-reviewed transaction script, NEVER learned from either captured store.
//! This exporter does not load weights, allocate Metal or run inference.
use super::*;
use crate::inference::models::qwen35::{Qwen35Config, kv_cache::{LayerSlot, q1_capture_layout}};
use std::collections::BTreeMap;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Call {
    start: usize,
    input_rows: usize,
    logit_rows: usize,
    hidden_rows: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Point {
    target_cursor: usize,
    mtp_cursor: usize,
    retained_tokens: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    arm: String,
    calls: Vec<Call>,
    decision_count: usize,
    final_state: Point,
    transactions: Vec<Point>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportPacket {
    schema: String,
    model: PathBuf,
    file_identity: acquisition::FileIdentity,
    workload: Workload,
    capacity: u32,
    output: PathBuf,
    cases: Vec<Case>,
}
fn coordinate(agent: &str, segment: u32) -> Coordinate {
    Coordinate { agent: agent.into(), logical_step: 0, logical_row_start: 0,
        layer: 0, stage: Stage::CacheState, segment }
}
fn state(cfg: &Qwen35Config, capacity: u32, name: &str, point: &Point) -> Result<Vec<ExpectedTensor>> {
    let layouts = q1_capture_layout::layouts(cfg, capacity)?;
    let agent = format!("{name}/slot0");
    let full = layouts.iter().filter(|l| matches!(l.slot, LayerSlot::Full(_)) && !l.mtp).count();
    let linear = layouts.iter().filter(|l| matches!(l.slot, LayerSlot::Linear(_))).count();
    let has_mtp = layouts.iter().any(|l| l.mtp);
    ensure!(point.target_cursor <= capacity as usize && point.mtp_cursor <= capacity as usize
        && point.retained_tokens <= point.target_cursor,
        "terminal cursor/ledger declaration exceeds selected capacity");
    let mut expected = vec![ExpectedTensor { at: coordinate(&agent, 0), scalar: Scalar::U32,
        shape: vec![full + usize::from(has_mtp)] },
        ExpectedTensor { at: coordinate(&agent, 1), scalar: Scalar::U8, shape: vec![linear] }];
    let mut region = 2u32;
    // Same declared order as State::read, over the allocator's OWN geometry:
    // every target full-attention root, MTP roots, then current conv/recurrent.
    for mtp in [false, true] {
        for layout in layouts.iter().filter(|l| matches!(l.slot, LayerSlot::Full(_)) && l.mtp == mtp) {
            let valid = if mtp { point.mtp_cursor } else { point.target_cursor };
            for (dtype, shape) in &layout.roots {
                ensure!(shape.len() == 4 && shape[0] == 1, "allocator full-attention descriptor axes");
                let bytes = shape[1].checked_mul(valid).and_then(|n| n.checked_mul(shape[3]))
                    .and_then(|n| n.checked_mul(dtype.size_of())).context("cache extent overflow")?;
                if bytes > 0 { expected.push(ExpectedTensor { at: coordinate(&agent, region),
                    scalar: Scalar::U8, shape: vec![bytes] }); }
                region += 1;
            }
        }
    }
    for layout in layouts.iter().filter(|l| matches!(l.slot, LayerSlot::Linear(_))) {
        for index in [0, 2] {
            let (dtype, shape) = &layout.roots[index];
            let bytes = shape.iter().try_fold(dtype.size_of(), |n, d| n.checked_mul(*d))
                .context("recurrent extent overflow")?;
            expected.push(ExpectedTensor { at: coordinate(&agent, region), scalar: Scalar::U8,
                shape: vec![bytes] }); region += 1;
        }
    }
    if point.retained_tokens > 0 {
        expected.push(ExpectedTensor { at: coordinate(&agent, u32::MAX), scalar: Scalar::U32,
            shape: vec![point.retained_tokens] });
    }
    Ok(expected)
}
fn calls(cfg: &Qwen35Config, case: &Case) -> Result<Vec<ExpectedTensor>> {
    let agent = format!("{}/slot0", case.name);
    let mut expected = Vec::new();
    ensure!(!case.calls.is_empty() && case.decision_count > 0, "complete nonempty transaction script required");
    for (transaction, call) in case.calls.iter().enumerate() {
        ensure!(call.input_rows > 0 && call.logit_rows > 0 && call.logit_rows <= call.input_rows
            && call.hidden_rows <= call.input_rows, "independent call row extent malformed");
        let first = call.start + call.input_rows - call.logit_rows;
        for row in 0..call.logit_rows {
            expected.push(ExpectedTensor { at: trace::coordinate(&agent, transaction, first + row,
                Stage::Logits), scalar: Scalar::F32, shape: vec![1, cfg.vocab_size as usize] });
        }
        for row in 0..call.hidden_rows {
            expected.push(ExpectedTensor { at: trace::coordinate(&agent, transaction, call.start+call.input_rows-call.hidden_rows+row,
                Stage::LayerOutput), scalar: Scalar::F32, shape: vec![1, cfg.hidden_size as usize] });
        }
    }
    for index in 0..case.decision_count {
        expected.push(ExpectedTensor { at: trace::coordinate(&agent, index, index, Stage::SpecCanonicalIds),
            scalar: Scalar::U32, shape: vec![1] });
    }
    Ok(expected)
}
#[test]
#[ignore = "root-reviewed GGUF header/transaction-script export only; no model/device/inference"]
fn qwen_c2_independent_extent_manifest_export() { run().unwrap(); }
fn run() -> Result<()> {
    let path = PathBuf::from(std::env::var_os("HF2Q_C2_EXTENT_PACKET").context("HF2Q_C2_EXTENT_PACKET required")?);
    let mut raw = Vec::new();
    fs::File::open(path)?.take(4 * 1024 * 1024 + 1).read_to_end(&mut raw)?;
    ensure!(raw.len() <= 4 * 1024 * 1024, "extent packet exceeds4MiB");
    let packet: ExportPacket = serde_json::from_slice(&raw)?;
    ensure!(packet.schema == "hf2q.c2.extent-export-packet.v1", "extent export schema");
    packet.workload.validate().map_err(anyhow::Error::msg)?;
    output_preflight(&packet.output)?;
    let file = fs::File::open(&packet.model)?;
    ensure!(acquisition::FileIdentity::observed(&file)? == packet.file_identity,
        "header file identity differs from reviewed artifact");
    let header = mlx_native::gguf::GgufFile::from_file(file.try_clone()?)?;
    ensure!(header.tensor_data_offset() <= 32 * 1024 * 1024, "bounded metadata header required");
    let cfg = Qwen35Config::from_gguf(&header)?;
    ensure!(cfg.vocab_size == packet.workload.vocab_size, "workload/header vocabulary mismatch");
    let mut expected = BTreeMap::new();
    for case in &packet.cases {
        ensure!(!case.name.is_empty() && case.name.chars().all(|c| c.is_ascii_alphanumeric() || "-_".contains(c))
            && matches!(case.arm.as_str(), "actual" | "twin" | "single"), "case/arm binding malformed");
        let key = format!("{}-{}", case.name, case.arm);
        if !case.calls.is_empty() {
            ensure!(expected.insert(key.clone(), calls(&cfg, case)?).is_none(), "duplicate independent store");
        } else { ensure!(case.decision_count == 0, "empty call plan declares decisions"); }
        ensure!(expected.insert(format!("{key}-state"),
            state(&cfg, packet.capacity, &case.name, &case.final_state)?).is_none(), "duplicate state extent");
        if !case.transactions.is_empty() {
            let mut tensors = Vec::new();
            for (step, point) in case.transactions.iter().enumerate() {
                let mut items = state(&cfg, packet.capacity, &case.name, point)?;
                for item in &mut items { item.at.logical_step = step as u64; }
                tensors.extend(items);
            }
            expected.insert(format!("{key}-transactions"), tensors);
        }
    }
    ensure!(!expected.is_empty(), "no independently registered stores");
    ensure!(acquisition::FileIdentity::observed(&file)? == packet.file_identity, "model header changed");
    fs::create_dir(&packet.output)?;
    fs::set_permissions(&packet.output, fs::Permissions::from_mode(0o700))?;
    let mut bindings = BTreeMap::new();
    for (key, tensors) in &expected {
        let bytes = serde_json::to_vec_pretty(tensors)?;
        let path = packet.output.join(format!("{key}.expected.json"));
        fs::write(&path, &bytes)?;
        bindings.insert(key.clone(), serde_json::json!({"path":path,"sha256":hash(&bytes)}));
    }
    fs::write(packet.output.join("expected-bindings.json"), serde_json::to_vec_pretty(&bindings)?)?;
    fs::write(packet.output.join("extent-export.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "schema":"hf2q.c2.extent-export.v1", "model_loaded":false,"metal_allocated":false,
        "inference_executed":false,"script_sha256":hash(&raw),"stores":expected.keys().collect::<Vec<_>>(),
        "source_geometry":"HybridCacheAllocationPlan via q1_capture_layout",
        "original_refactor_gate":{"pre_c2_parent":"62a5326422e7ebb8e5621c978cfd6d38b23facd8",
            "first_c2":"51a3923fafbe46e4a18c22c5fdc24ce20a46593c",
            "both_historical_assemblies_native":"0.18.0",
            "historical_native_checksum":"2a804cf31840059e548c763be935be6b6b75563add645556311130a61ed525fd",
            "parent_anchor_semantics":"fixed single anchor; no later K1-environment claim",
            "candidate_native":"0.24.0","same_native_pin_and_case_required":true,
            "proof_scope":"historical original refactor only; NOT current0.24qualification",
            "assembled_or_executed":false}
    }))?)?;
    Ok(())
}
