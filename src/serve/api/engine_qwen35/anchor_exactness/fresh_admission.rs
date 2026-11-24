//! Join the owned diagnostic evidence, not bridge booleans. Q1 remains the
//! diagnostic operating policy; none of these records is test-binary Q5 admission.
use super::*;
use super::fresh_reference::Capture;
use serde_json::Value;
use std::collections::BTreeMap;

fn document(binding: &acquisition::FileBinding) -> Result<Value> {
    Ok(serde_json::from_slice(&acquisition::checked_file(binding, 32*1024*1024)?)?)
}
fn bound(value: &Value) -> Result<(acquisition::FileBinding, Value)> {
    let binding = serde_json::from_value(value.clone())?;
    let value = document(&binding)?;
    Ok((binding, value))
}
fn empty(value: &Value) -> bool { value.as_array().is_some_and(Vec::is_empty) }
fn bytes(value: &Value) -> Result<u64> { value.as_u64().context("measured byte/count field absent") }
fn native_cleanup(value: &Value) -> Result<()> {
    ensure!(value["schema"] == "hf2q.private.native-owned-lifetime.v1"
        && value["proven"] == true && value["direct_child_reaped"] == true
        && value["exit_codes"] == serde_json::json!([0])
        && empty(&value["remaining"]) && empty(&value["errors"]) && empty(&value["refused_pgids"])
        && bytes(&value["native_observations"])? > 0, "owned native cleanup is not proven");
    Ok(())
}
fn diagnostic_status(receipt: &Value, key: &str, mode: &str) -> Result<()> {
    ensure!(receipt["schema"] == "hf2q.private.c2-diagnostic-attempt.v1" && receipt["mode"] == mode
        && receipt["diagnostic_key_sha256"] == key && receipt["diagnostic_pass"] == true
        && receipt["model_launch_attempted"] == true && receipt["acquisition_complete"] == true
        && receipt["retain_lease"] == false && empty(&receipt["errors"])
        && receipt.get("sampler_error") == Some(&Value::Null)
        && receipt["physical_admission"] == false
        && receipt["calibration_kind"] == "Q1FootprintRecordNotQ5CalibrationPass",
        "failed/mismatched diagnostic cannot admit a reference");
    native_cleanup(&receipt["cleanup"])?;
    let pid = bytes(&receipt["pid"])?;
    ensure!(receipt["known_owned_lifetimes"].as_array().is_some_and(|rows|
        rows.iter().any(|row| row["pid"].as_u64() == Some(pid)
            && row["lifetime_id"].as_str().is_some_and(|id| id.starts_with("darwin-unique:")))),
        "diagnostic process has no recorded native lifetime");
    Ok(())
}
fn measured(receipt: &Value, root: &Path) -> Result<(u64,u64,u64)> {
    let (series_binding, series) = bound(&receipt["memory_series"])?;
    ensure!(series_binding.path == root.join("memory-series.json")
        && series["schema"] == "hf2q.q1.endpoint-memory-series.v1", "wrong measured memory series");
    let series = series["series"].as_array().context("memory series absent")?;
    let memory = &receipt["memory"];
    ensure!(!series.is_empty() && bytes(&memory["runtime_samples"])? == series.len() as u64,
        "complete measured memory samples required");
    let swap = bytes(&memory["baseline"]["swap_bytes"])?;
    let mut peak = 0; let mut rss = 0; let mut low = u64::MAX;
    for row in series {
        ensure!(row["pressure_raw"] == "1" && bytes(&row["swap"])? <= swap,
            "diagnostic normal-pressure/zero-positive-swap boundary failed");
        low = low.min(bytes(&row["free_plus_cached"])?);
        if let Some(value) = row["footprint"].as_u64() { peak = peak.max(value); }
        if let Some(value) = row["resident"].as_u64() { rss = rss.max(value); }
    }
    let charge = bytes(&memory["baseline"]["min_free_plus_cached_bytes"])? .saturating_sub(low);
    ensure!(peak > 0 && rss > 0 && peak == bytes(&memory["peak_phys_footprint_bytes"])?
        && rss == bytes(&memory["peak_resident_bytes"])? && charge == bytes(&memory["charged_bytes"])?,
        "measured calibration boundary does not match raw samples");
    for field in ["stdout", "stderr"] {
        let binding: acquisition::FileBinding = serde_json::from_value(receipt[field].clone())?;
        ensure!(binding.path.parent() == Some(root)
            && acquisition::file_digest(&binding.path)? == binding.sha256, "diagnostic stream binding changed");
    }
    Ok((peak,rss,charge))
}
fn prepared(binding: &acquisition::FileBinding, receipt: &Value, receipt_binding: &acquisition::FileBinding) -> Result<Value> {
    let prepared = document(binding)?;
    let root: PathBuf = serde_json::from_value(prepared["output"].clone())?;
    ensure!(prepared["schema"] == "hf2q.private.c2-diagnostic-prepared.v1"
        && receipt_binding.path == root.join("receipt.json")
        && binding.path == root.parent().context("attempt parent absent")?.join("prepared.json")
        && prepared["file_sha256"] == receipt["file_sha256"]
        && prepared["config"]["packet"] == receipt["packet"]
        && prepared["diagnostic_key_sha256"] == receipt["diagnostic_key_sha256"]
        && prepared["config"]["mode"] == receipt["mode"]
        && prepared["plan"]["kind"] == "fresh-http", "diagnostic does not belong to its actual prepared attempt");
    Ok(prepared)
}
fn growth_cap(value: u64) -> Result<u64> {
    value.checked_mul(5).and_then(|n| n.checked_add(3)).map(|n| n/4).context("measured growth cap overflow")
}

fn strict_outer_binding(binding:&acquisition::FileBinding,strict:&Value,outer:&Value)->Result<()> {
    let command=outer["argv"].as_array().context("native outer command absent")?;
    ensure!(command.iter().all(|arg|arg.is_string()),"native outer argument is not a string");
    let separator=command.iter().position(|arg|arg=="--").context("strict canonical command delimiter absent")?;
    let canonical=strict["argv"].as_array().context("strict canonical command absent")?;
    ensure!(!canonical.is_empty()&&canonical.as_slice()==&command[separator+1..]
        &&strict["canonical_binding"]["argv"].as_array()==Some(canonical),
        "native outer did not supervise this strict canonical command");
    let mut outputs=command[..separator].windows(2).filter(|pair|pair[0]=="--output");
    let output=outputs.next().context("strict outer output absent")?[1].as_str().context("strict output is not a path")?;
    ensure!(outputs.next().is_none()&&Path::new(output).is_absolute(),"ambiguous/nonabsolute strict outer output");
    ensure!(binding.path.file_name()==Some(std::ffi::OsStr::new("receipt.json"))
        &&binding.path.parent().context("strict receipt parent absent")?.canonicalize()?==Path::new(output).canonicalize()?,
        "selected strict receipt is not the native outer output");
    Ok(())
}

pub(super) fn validate(run: &Value, capture_binding: &acquisition::FileBinding, capture: &Capture, input_kind: &str) -> Result<()> {
    ensure!(run["schema"] == "hf2q.c2.fresh-fixed-input-run-receipt.v1"
        && run["capture_sha256"] == capture_binding.sha256 && run["acquisition_nonce"] == capture.acquisition_nonce
        && run["admitted"] == true && run["complete"] == true && run["cleanup"] == true
        && run["stopped"] == false && run["whole_c2_pass"] == false
        && run["physical_admission"] == false && run["admission_kind"] == "MatchedDiagnosticFootprint"
        && run["phase"] == "independent-reference-acquisition", "unsupported reference admission role");
    let (calibration_binding, calibration) = bound(&run["calibration_receipt"])?;
    let (diagnostic_binding, diagnostic) = bound(&run["diagnostic_receipt"])?;
    let (prepared_binding, _) = bound(&run["diagnostic_prepared"])?;
    let key = run["diagnostic_key_sha256"].as_str().context("diagnostic key absent")?;
    ensure!(key.len() == 64 && key.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "diagnostic key is not a canonical digest");
    diagnostic_status(&calibration,key,"footprint-calibration")?;
    diagnostic_status(&diagnostic,key,"matched-cap")?;
    let reference = prepared(&prepared_binding,&diagnostic,&diagnostic_binding)?;
    let calibration_root = calibration_binding.path.parent().context("calibration root absent")?;
    let calibration_prepared_path = calibration_root.parent().context("calibration attempt parent absent")?.join("prepared.json");
    let calibration_prepared = acquisition::FileBinding {
        sha256: acquisition::file_digest(&calibration_prepared_path)?, path: calibration_prepared_path,
    };
    let prior = prepared(&calibration_prepared,&calibration,&calibration_binding)?;
    let root: PathBuf = serde_json::from_value(reference["output"].clone())?;
    ensure!(calibration_binding.path != diagnostic_binding.path && calibration_binding.sha256 != diagnostic_binding.sha256
        && calibration_root != root.as_path() && !root.starts_with(calibration_root) && !calibration_root.starts_with(&root)
        && calibration["artifacts"]["acquisition_nonce"].as_str().is_some_and(|nonce|
            nonce.len()==64 && nonce != capture.acquisition_nonce)
        && diagnostic["artifacts"]["acquisition_nonce"] == capture.acquisition_nonce
        && diagnostic["artifacts"]["fresh_capture"] == serde_json::to_value(capture_binding)?
        && diagnostic["packet"]["sha256"] == capture.packet.sha256, "calibration/reference trajectories are not independent");
    let cfg = &reference["config"];
    ensure!(reference["plan"]["packet_env"] == "HF2Q_C2_FRESH_HTTP_PACKET"
        && cfg["footprint_record"] == serde_json::to_value(&calibration_binding)?
        && run["source_manifest"] == cfg["source_manifest"] && run["packet"] == cfg["packet"]
        && prior["config"]["source_manifest"] == cfg["source_manifest"]
        && cfg["test_binary"]["sha256"] == capture.identity.binary_sha256
        && prior["config"]["test_binary"] == cfg["test_binary"]
        && reference["matched_cap"] == diagnostic["matched_cap"], "prepared/source/executable admission join changed");
    let (peak,rss,charge) = measured(&calibration,calibration_root)?;
    let (reference_peak,reference_rss,_) = measured(&diagnostic,&root)?;
    let cap = &reference["matched_cap"];
    ensure!(cap["kind"] == "MatchedDiagnosticCapNotQ5Admission"
        && cap["record"] == serde_json::to_value(&calibration_binding)?
        && bytes(&cap["process_footprint_limit_bytes"])? == growth_cap(peak)?
        && bytes(&cap["process_rss_limit_bytes"])? == growth_cap(rss)?
        && bytes(&cap["charged_bytes"])? == charge, "cap was not derived from this actual measured calibration");
    ensure!(reference_peak <= bytes(&cap["process_footprint_limit_bytes"])?
        && reference_rss <= bytes(&cap["process_rss_limit_bytes"])?, "reference exceeded its measured process growth caps");
    let (_,mut cal_packet) = bound(&calibration["packet"])?;
    let (_,mut ref_packet) = bound(&diagnostic["packet"])?;
    cal_packet.as_object_mut().context("calibration packet object absent")?.remove("output");
    ref_packet.as_object_mut().context("reference packet object absent")?.remove("output");
    ensure!(cal_packet == ref_packet, "calibration/reference semantic packet drift");
    let mut files = BTreeMap::new();
    for receipt in [&calibration,&diagnostic] {
        for (path,digest) in receipt["file_sha256"].as_object().context("diagnostic sources absent")? {
            let digest = digest.as_str().context("source digest absent")?;
            if let Some(previous) = files.insert(path,digest) { ensure!(previous == digest,"shared diagnostic source changed"); }
        }
    }
    for (path,digest) in files { ensure!(acquisition::file_digest(Path::new(path))? == digest,"frozen diagnostic source changed"); }
    let (_,chain) = bound(&run["production_predecessor_chain"])?;
    ensure!(chain["schema"] == "hf2q.private.c2-diagnostic-production-predecessor.v1"
        && cfg["production_predecessor"] == run["production_predecessor_chain"]
        && chain["q5_receipt"] == run["production_q5_predecessor"], "production predecessor chain mismatch");
    let (production_binding,production) = bound(&chain["q5_receipt"])?;
    let (strict_binding,strict) = bound(&chain["strict_receipt"])?;
    let (_,outer) = bound(&chain["outer_receipt"])?;
    let (manifest_binding,manifest) = bound(&chain["outer_manifest"])?;
    let (_,case) = bound(&chain["case"])?;
    let family = match capture.identity.family { Family::Qwen38 => "qwen38", Family::Qwen36 => "qwen36",
        _ => anyhow::bail!("unsupported reference family") };
    ensure!(production["schema"] == "hf2q.q5.calibration-receipt.v1"
        && production["verdict"]["verdict"] == "CalibrationPass"
        && production["run_kind"] == "calibration" && production["cleanup_proven"] == true
        && production["cleanup"]["proven"] == true && production.get("sampler_error") == Some(&Value::Null)
        && production["binary"]["sha256"] == cfg["production_binary"]["sha256"]
        && production["binary"]["sha256"] != capture.identity.binary_sha256
        && production["artifact"]["sha256"] == capture.identity.model_sha256
        && production["host"] == chain["host"] && production["workload"] == case["workload"]
        && production["family"] == case["family"] && case["family"] == family
        && case["workload"]["slots"] == 4 && case["workload"]["context_per_slot"] == 262144
        && case["workload"]["anchors_per_slot"] == chain["depth"] && chain["depth"] == 4
        && chain["kind"] == input_kind && chain["route"] == "legacy-off",
        "not the matched full production Q5 calibration");
    let strict_q5 = PathBuf::from(strict["canonical_binding"]["receipt_path"].as_str().context("strict Q5 binding absent")?);
    ensure!(strict_q5.canonicalize()? == production_binding.path.canonicalize()?
        && acquisition::file_digest(&strict_q5)? == production_binding.sha256
        && strict["passed"] == true && strict["cleanup_proven"] == true && strict["cleanup"]["proven"] == true
        && empty(&strict["cleanup"]["remaining"])
        && outer["passed"] == true && outer["cleanup_proven"] == true && outer["retain_lease"] == false
        && empty(&outer["errors"]) && outer["manifest_sha256"] == manifest_binding.sha256
        && outer["argv"] == manifest["units"]["outer"]["argv"], "production strict/native chain did not supervise this Q5 receipt");
    native_cleanup(&outer["cleanup"])?;
    strict_outer_binding(&strict_binding,&strict,&outer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fresh_admission_refuses_failed_or_mismatched_diagnostic_receipts() {
        let receipt = serde_json::json!({"schema":"hf2q.private.c2-diagnostic-attempt.v1","mode":"matched-cap",
            "diagnostic_key_sha256":"key","diagnostic_pass":true,"model_launch_attempted":true,
            "acquisition_complete":true,"retain_lease":false,"errors":[],"sampler_error":null,"physical_admission":false,
            "calibration_kind":"Q1FootprintRecordNotQ5CalibrationPass","pid":17,
            "known_owned_lifetimes":[{"pid":17,"lifetime_id":"darwin-unique:19"}],
            "cleanup":{"schema":"hf2q.private.native-owned-lifetime.v1","proven":true,"direct_child_reaped":true,
                "exit_codes":[0],"remaining":[],"errors":[],"refused_pgids":[],"native_observations":3}});
        assert!(diagnostic_status(&receipt,"key","matched-cap").is_ok());
        for field in ["diagnostic_pass","acquisition_complete","model_launch_attempted"] {
            let mut failed = receipt.clone(); failed[field] = Value::Bool(false);
            assert!(diagnostic_status(&failed,"key","matched-cap").is_err());
        }
        let mut retained = receipt.clone(); retained["retain_lease"] = Value::Bool(true);
        assert!(diagnostic_status(&retained,"key","matched-cap").is_err());
        assert!(diagnostic_status(&receipt,"other-key","matched-cap").is_err());
        assert!(diagnostic_status(&receipt,"key","footprint-calibration").is_err());
        let mut unclean = receipt; unclean["cleanup"]["remaining"] = serde_json::json!([{"pid":17}]);
        assert!(diagnostic_status(&unclean,"key","matched-cap").is_err());
    }
    #[test]
    fn fresh_predecessor_refuses_cross_run_strict_and_native_evidence() {
        let directory=tempfile::tempdir().unwrap();
        let first=directory.path().join("strict-a");let second=directory.path().join("strict-b");
        fs::create_dir(&first).unwrap();fs::create_dir(&second).unwrap();
        let path=first.join("receipt.json");fs::write(&path,b"{}").unwrap();
        let binding=acquisition::FileBinding{sha256:acquisition::file_digest(&path).unwrap(),path};
        let strict=serde_json::json!({"argv":["q5","--case","case.json"],
            "canonical_binding":{"argv":["q5","--case","case.json"]}});
        let outer=serde_json::json!({"argv":["python","guard","--output",first,"--","q5","--case","case.json"]});
        assert!(strict_outer_binding(&binding,&strict,&outer).is_ok());
        let mut different_run=outer.clone();different_run["argv"][3]=serde_json::json!(second);
        assert!(strict_outer_binding(&binding,&strict,&different_run).is_err());
        let mut different_command=strict.clone();different_command["argv"][2]=serde_json::json!("other.json");
        assert!(strict_outer_binding(&binding,&different_command,&outer).is_err());
        let mut different_binding=strict;
        different_binding["canonical_binding"]["argv"][2]=serde_json::json!("other.json");
        assert!(strict_outer_binding(&binding,&different_binding,&outer).is_err());
    }
}
