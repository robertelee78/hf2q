//! Stage-by-stage element-wise diff between hf2q and llama.cpp ViT dumps.
//!
//! ADR-005 Phase 2c iter 124 (W55).
//!
//! Reads two dump directories produced by `HF2Q_VIT_DUMP=...` (hf2q) and
//! by the C++ `peer_dumper` (llama.cpp), pairs them by stage name, and
//! reports the max-abs-err per stage along with the first stage where
//! the divergence exceeds a tolerance.
//!
//! On-disk format (matches both producers):
//!   <stage>.bin   — raw F32 LE, contiguous
//!   <stage>.json  — { "name": ..., "dtype": "f32",
//!                      "shape": [...], "n_elements": ... }
//!
//! Invocation:
//!   diff <hf2q-dir> <peer-dir> [--tol 1e-4] [--print-all]
//!
//! Exit code 0 on identical-within-tolerance, 1 on any divergence, 2 on
//! I/O / shape errors.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Debug)]
struct Args {
    hf2q_dir: PathBuf,
    peer_dir: PathBuf,
    tol: f32,
    print_all: bool,
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = env::args().collect();
    if argv.len() < 3 {
        return Err(format!(
            "Usage: {} <hf2q-dir> <peer-dir> [--tol 1e-4] [--print-all]",
            argv[0]
        ));
    }
    let mut tol = 1e-4f32;
    let mut print_all = false;
    let mut i = 1;
    let positional = &mut Vec::new();
    while i < argv.len() {
        let a = &argv[i];
        match a.as_str() {
            "--tol" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--tol requires a value".to_string());
                }
                tol = argv[i]
                    .parse::<f32>()
                    .map_err(|e| format!("--tol: {e}"))?;
            }
            "--print-all" => print_all = true,
            "-h" | "--help" => {
                return Err(format!(
                    "Usage: {} <hf2q-dir> <peer-dir> [--tol 1e-4] [--print-all]",
                    argv[0]
                ));
            }
            other => positional.push(other.to_string()),
        }
        i += 1;
    }
    if positional.len() < 2 {
        return Err("Need <hf2q-dir> and <peer-dir>".to_string());
    }
    let hf2q_dir = PathBuf::from(&positional[0]);
    let peer_dir = PathBuf::from(&positional[1]);
    Ok(Args {
        hf2q_dir,
        peer_dir,
        tol,
        print_all,
    })
}

/// Read a `<stage>.bin` and return the F32 vector. Errors with shape
/// info derived from `<stage>.json` for diagnostics.
fn read_dump(dir: &Path, stage: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let bin_path = dir.join(format!("{stage}.bin"));
    let json_path = dir.join(format!("{stage}.json"));
    let bytes = fs::read(&bin_path)
        .map_err(|e| format!("read {}: {e}", bin_path.display()))?;
    if !bin_path.exists() {
        return Err(format!("missing {}", bin_path.display()));
    }
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{}: byte length {} not a multiple of 4",
            bin_path.display(),
            bytes.len()
        ));
    }
    let mut data = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let shape = if json_path.exists() {
        let j = fs::read_to_string(&json_path)
            .map_err(|e| format!("read {}: {e}", json_path.display()))?;
        parse_shape_from_json(&j).unwrap_or_else(|_| vec![data.len()])
    } else {
        vec![data.len()]
    };
    Ok((data, shape))
}

/// Hand-rolled shape extraction so we don't add a JSON dep. Looks for
/// `"shape":[a,b,c]` and parses the integers. Errors are coarse — the
/// JSON sidecar is well-formed-by-construction in both producers.
fn parse_shape_from_json(s: &str) -> Result<Vec<usize>, String> {
    let key = "\"shape\":[";
    let start = s
        .find(key)
        .ok_or_else(|| "shape key not found".to_string())?;
    let after = &s[start + key.len()..];
    let end = after
        .find(']')
        .ok_or_else(|| "shape close not found".to_string())?;
    let body = &after[..end];
    let mut out = Vec::new();
    for part in body.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        out.push(
            t.parse::<usize>()
                .map_err(|e| format!("shape parse '{t}': {e}"))?,
        );
    }
    Ok(out)
}

#[derive(Debug)]
struct StageDiff {
    stage: String,
    n_elements: usize,
    hf2q_shape: Vec<usize>,
    peer_shape: Vec<usize>,
    max_abs_err: f32,
    mean_abs_err: f32,
    max_rel_err: f32,
    /// True when both shapes match AND the data has the same length.
    shapes_match: bool,
    /// Position of the worst element (linear index) for diagnosis.
    worst_idx: usize,
    worst_hf2q: f32,
    worst_peer: f32,
}

fn diff_stage(stage: &str, hf2q: &Path, peer: &Path) -> Result<StageDiff, String> {
    let (h, hshape) = read_dump(hf2q, stage)?;
    let (p, pshape) = read_dump(peer, stage)?;
    let shapes_match = hshape == pshape;
    let n = h.len().min(p.len());
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut max_rel = 0.0f32;
    let mut worst_idx = 0usize;
    let mut worst_hf2q = 0.0f32;
    let mut worst_peer = 0.0f32;
    for i in 0..n {
        let d = (h[i] - p[i]).abs();
        sum_abs += d as f64;
        if d > max_abs {
            max_abs = d;
            worst_idx = i;
            worst_hf2q = h[i];
            worst_peer = p[i];
        }
        let denom = p[i].abs().max(1e-6);
        let r = d / denom;
        if r > max_rel {
            max_rel = r;
        }
    }
    let mean_abs = if n > 0 { (sum_abs / n as f64) as f32 } else { 0.0 };
    Ok(StageDiff {
        stage: stage.to_string(),
        n_elements: n,
        hf2q_shape: hshape,
        peer_shape: pshape,
        max_abs_err: max_abs,
        mean_abs_err: mean_abs,
        max_rel_err: max_rel,
        shapes_match,
        worst_idx,
        worst_hf2q,
        worst_peer,
    })
}

/// Discover stage names common to both directories. We use the `.bin`
/// files as the source of truth; orphans (only in one side) are
/// reported separately.
fn discover_stages(dir: &Path) -> Result<BTreeSet<String>, String> {
    let mut set = BTreeSet::new();
    for entry in fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("bin") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                set.insert(stem.to_string());
            }
        }
    }
    Ok(set)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    if !args.hf2q_dir.is_dir() {
        eprintln!("ERR: hf2q dir not found: {}", args.hf2q_dir.display());
        return ExitCode::from(2);
    }
    if !args.peer_dir.is_dir() {
        eprintln!("ERR: peer dir not found: {}", args.peer_dir.display());
        return ExitCode::from(2);
    }

    let h_stages = match discover_stages(&args.hf2q_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERR: {e}");
            return ExitCode::from(2);
        }
    };
    let p_stages = match discover_stages(&args.peer_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERR: {e}");
            return ExitCode::from(2);
        }
    };

    let common: Vec<String> = h_stages
        .iter()
        .filter(|s| p_stages.contains(*s))
        .cloned()
        .collect();
    let only_hf2q: Vec<String> =
        h_stages.iter().filter(|s| !p_stages.contains(*s)).cloned().collect();
    let only_peer: Vec<String> =
        p_stages.iter().filter(|s| !h_stages.contains(*s)).cloned().collect();

    println!(
        "vit_parity_probe diff — hf2q={} peer={} tol={}",
        args.hf2q_dir.display(),
        args.peer_dir.display(),
        args.tol
    );
    println!("=================================================================");
    println!(
        "{:<24} {:>10} {:>14} {:>14} {:>14} {:>9}",
        "stage", "n", "max_abs", "mean_abs", "max_rel", "shape_ok"
    );
    println!("-----------------------------------------------------------------");

    let mut first_divergent: Option<StageDiff> = None;
    let mut all_diffs: Vec<StageDiff> = Vec::with_capacity(common.len());
    let mut any_error = false;

    for stage in &common {
        match diff_stage(stage, &args.hf2q_dir, &args.peer_dir) {
            Ok(d) => {
                println!(
                    "{:<24} {:>10} {:>14.6e} {:>14.6e} {:>14.6e} {:>9}",
                    d.stage,
                    d.n_elements,
                    d.max_abs_err,
                    d.mean_abs_err,
                    d.max_rel_err,
                    if d.shapes_match { "yes" } else { "NO" }
                );
                let diverged = !d.shapes_match || d.max_abs_err > args.tol;
                if diverged && first_divergent.is_none() {
                    first_divergent = Some(StageDiff {
                        stage: d.stage.clone(),
                        n_elements: d.n_elements,
                        hf2q_shape: d.hf2q_shape.clone(),
                        peer_shape: d.peer_shape.clone(),
                        max_abs_err: d.max_abs_err,
                        mean_abs_err: d.mean_abs_err,
                        max_rel_err: d.max_rel_err,
                        shapes_match: d.shapes_match,
                        worst_idx: d.worst_idx,
                        worst_hf2q: d.worst_hf2q,
                        worst_peer: d.worst_peer,
                    });
                }
                all_diffs.push(d);
            }
            Err(e) => {
                eprintln!("ERR diff {stage}: {e}");
                any_error = true;
            }
        }
    }

    println!("=================================================================");
    if !only_hf2q.is_empty() {
        println!("Only in hf2q: {only_hf2q:?}");
    }
    if !only_peer.is_empty() {
        println!("Only in peer: {only_peer:?}");
    }

    if let Some(d) = &first_divergent {
        println!();
        println!("First divergent stage: {}", d.stage);
        println!("  shapes_match = {} (hf2q={:?}, peer={:?})",
                 d.shapes_match, d.hf2q_shape, d.peer_shape);
        println!("  max_abs_err  = {:.6e}", d.max_abs_err);
        println!("  mean_abs_err = {:.6e}", d.mean_abs_err);
        println!("  max_rel_err  = {:.6e}", d.max_rel_err);
        println!("  worst_idx    = {} (hf2q={:.6e}, peer={:.6e})",
                 d.worst_idx, d.worst_hf2q, d.worst_peer);
    } else if !any_error {
        println!();
        println!("All {} common stages within tolerance ({}).", common.len(), args.tol);
    }

    if args.print_all {
        println!();
        for d in &all_diffs {
            println!("DEBUG {}:", d.stage);
            println!("  hf2q_shape = {:?}", d.hf2q_shape);
            println!("  peer_shape = {:?}", d.peer_shape);
            println!("  n_elements = {}", d.n_elements);
            println!("  max_abs    = {:.6e}", d.max_abs_err);
            println!("  worst_idx  = {} (hf2q={:.6e}, peer={:.6e})",
                     d.worst_idx, d.worst_hf2q, d.worst_peer);
        }
    }

    if any_error {
        return ExitCode::from(2);
    }
    if first_divergent.is_some() {
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_shape_basic() {
        let s = r#"{"name":"x","dtype":"f32","shape":[36,1152],"n_elements":41472}"#;
        assert_eq!(parse_shape_from_json(s).unwrap(), vec![36, 1152]);
    }

    #[test]
    fn parse_shape_one_dim() {
        let s = r#"{"shape":[2816]}"#;
        assert_eq!(parse_shape_from_json(s).unwrap(), vec![2816]);
    }

    #[test]
    fn parse_shape_missing() {
        let s = r#"{"foo":"bar"}"#;
        assert!(parse_shape_from_json(s).is_err());
    }
}
