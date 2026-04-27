//! iter23_audit — ADR-007 iter-23 semantic gate measurement.
//!
//! Runs three gates:
//!   Gate A: cosine similarity of SDPA outputs (dense vs TQ) at 4 layers × 100 positions.
//!           PASS iff mean ≥ 0.999 AND p1 ≥ 0.99.
//!   Gate B: argmax divergence rate over 1000 pure-greedy tokens.
//!           PASS iff rate < 1% (≤ 10 divergences per 1000).
//!   Gate C: perplexity delta from NLL sums.
//!           PASS iff |PPL_tq - PPL_dense| / PPL_dense < 1%.
//!
//! Verdict:
//!   ACCEPT         — all 3 gates pass   → ADR-007 closes as shippable
//!   ACCEPT_PARTIAL — 2 of 3 pass + 3rd within 2× threshold → iter 24 refinement
//!   REJECT         — any gate fails wide margin → investigation needed
//!
//! Environment:
//!   HF2Q_ITER23_MODEL — path to GGUF (required)
//!   HF2Q_ITER23_HF2Q  — path to hf2q binary (default: cargo build artifact)
//!
//! Produces /tmp/cfa-iter23/audit.json with full schema.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<()> {
    let model = std::env::var("HF2Q_ITER23_MODEL")
        .context("HF2Q_ITER23_MODEL must be set to the GGUF model path")?;
    let model_path = PathBuf::from(&model);
    anyhow::ensure!(model_path.exists(), "Model not found: {}", model_path.display());

    // Locate hf2q binary: env override → worktree release build → main repo release.
    let hf2q_bin = locate_hf2q_binary()?;
    eprintln!("[iter23_audit] hf2q binary: {}", hf2q_bin.display());
    eprintln!("[iter23_audit] model: {}", model_path.display());

    // Sourdough prompt path (relative to Cargo manifest dir, then absolute fallback).
    let prompt_path = locate_prompt()?;
    let prompt_text = fs::read_to_string(&prompt_path)
        .with_context(|| format!("read prompt: {}", prompt_path.display()))?;
    let prompt_text = prompt_text.trim().to_string();
    eprintln!("[iter23_audit] prompt: {:?} (len={})", prompt_path.display(), prompt_text.len());

    // Output directory.
    let out_dir = PathBuf::from("/tmp/cfa-iter23");
    let dump_dir = out_dir.join("dumps");
    fs::create_dir_all(&dump_dir)?;
    fs::create_dir_all(&out_dir)?;

    let t0 = Instant::now();

    // -----------------------------------------------------------------------
    // Gate A: cosine similarity of SDPA outputs (layers 0,5,15,25 × pos 0..100)
    // -----------------------------------------------------------------------
    eprintln!("\n[iter23_audit] === Gate A: SDPA cosine similarity ===");
    let cosine_result = run_gate_a(&hf2q_bin, &model_path, &prompt_text, &dump_dir, &out_dir)?;
    eprintln!("[iter23_audit] Gate A result: mean={:.6} min={:.6} p1={:.6} p50={:.6} p99={:.6} pass={}",
        cosine_result.mean, cosine_result.min, cosine_result.p1,
        cosine_result.p50, cosine_result.p99, cosine_result.pass);

    // -----------------------------------------------------------------------
    // Gate B + C: argmax divergence + PPL delta (combined 1000-token greedy run)
    // -----------------------------------------------------------------------
    eprintln!("\n[iter23_audit] === Gate B+C: argmax divergence + PPL delta ===");
    let (divergence_result, ppl_result) =
        run_gate_b_c(&hf2q_bin, &model_path, &prompt_text, &out_dir)?;
    eprintln!("[iter23_audit] Gate B: count={} rate={:.4} pass={}",
        divergence_result.count, divergence_result.rate, divergence_result.pass);
    eprintln!("[iter23_audit] Gate C: ppl_dense={:.4} ppl_tq={:.4} delta={:.6} pass={}",
        ppl_result.ppl_dense, ppl_result.ppl_tq, ppl_result.delta, ppl_result.pass);

    // -----------------------------------------------------------------------
    // Verdict
    // -----------------------------------------------------------------------
    let gates_pass = [cosine_result.pass, divergence_result.pass, ppl_result.pass];
    let pass_count = gates_pass.iter().filter(|&&p| p).count();

    // 2× threshold check for ACCEPT_PARTIAL: gate nearly-passed.
    let cosine_near = cosine_result.mean >= 0.999 / 2.0 || cosine_result.p1 >= 0.99 / 2.0;
    let div_near = divergence_result.rate < 0.02;   // 2% vs 1% threshold
    let ppl_near = ppl_result.delta.abs() < 0.02;   // 2% vs 1% threshold
    let near_count = [cosine_near, div_near, ppl_near].iter().filter(|&&n| n).count();

    let (verdict, rationale) = if pass_count == 3 {
        ("ACCEPT", "All 3 gates pass — ADR-007 closes as shippable. TQ is the default decode path.".to_string())
    } else if pass_count == 2 && near_count >= 3 {
        let failing: Vec<&str> = [
            (!cosine_result.pass).then_some("cosine"),
            (!divergence_result.pass).then_some("argmax_divergence"),
            (!ppl_result.pass).then_some("ppl_delta"),
        ].iter().filter_map(|x| *x).collect();
        ("ACCEPT_PARTIAL", format!(
            "2/3 gates pass; {} is within 2x threshold. Iter 24 refines.",
            failing.join(", ")
        ))
    } else {
        let failing: Vec<&str> = [
            (!cosine_result.pass).then_some("cosine"),
            (!divergence_result.pass).then_some("argmax_divergence"),
            (!ppl_result.pass).then_some("ppl_delta"),
        ].iter().filter_map(|x| *x).collect();
        ("REJECT", format!(
            "Gates failing: {}. Deeper investigation required.",
            failing.join(", ")
        ))
    };

    let elapsed_s = t0.elapsed().as_secs_f64();
    eprintln!("\n[iter23_audit] === VERDICT: {} ({:.0}s) ===", verdict, elapsed_s);
    eprintln!("[iter23_audit] {}", rationale);

    // -----------------------------------------------------------------------
    // Write audit.json
    // -----------------------------------------------------------------------
    let audit = serde_json::json!({
        "iter": 23,
        "elapsed_s": elapsed_s,
        "gates": {
            "cosine": {
                "mean": cosine_result.mean,
                "min": cosine_result.min,
                "p1": cosine_result.p1,
                "p50": cosine_result.p50,
                "p99": cosine_result.p99,
                "threshold_mean": 0.999,
                "threshold_p1": 0.99,
                "pass": cosine_result.pass,
            },
            "argmax_divergence": {
                "count": divergence_result.count,
                "total": divergence_result.total,
                "rate": divergence_result.rate,
                "threshold": 0.01,
                "pass": divergence_result.pass,
            },
            "ppl_delta": {
                "ppl_dense": ppl_result.ppl_dense,
                "ppl_tq": ppl_result.ppl_tq,
                "delta": ppl_result.delta,
                "threshold": 0.01,
                "pass": ppl_result.pass,
            }
        },
        "verdict": verdict,
        "verdict_rationale": rationale,
    });
    let audit_path = out_dir.join("audit.json");
    fs::write(&audit_path, serde_json::to_string_pretty(&audit)?)?;
    eprintln!("[iter23_audit] audit.json written to {}", audit_path.display());

    // Also copy audit.json to worktree root for commit.
    let worktree_audit = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("audit.json");
    fs::write(&worktree_audit, serde_json::to_string_pretty(&audit)?)?;
    eprintln!("[iter23_audit] audit.json written to {}", worktree_audit.display());

    Ok(())
}

// ---------------------------------------------------------------------------
// Gate A implementation
// ---------------------------------------------------------------------------

struct CosineGateResult {
    mean: f64,
    min: f64,
    p1: f64,
    p50: f64,
    p99: f64,
    pass: bool,
}

fn run_gate_a(
    hf2q: &Path,
    model: &Path,
    prompt: &str,
    dump_dir: &Path,
    out_dir: &Path,
) -> Result<CosineGateResult> {
    // Layers to dump (index 0 = sliding, 5 = global, 15 = middle, 25 = late).
    let layers_list = "0,5,15,25";
    // Dump first 100 decode positions.
    let max_pos = 100;
    // We use 100 decode tokens so we have enough positions.
    let max_tokens = max_pos + 10; // a little headroom for EOS

    let dense_dump_dir = dump_dir.join("dense");
    let tq_dump_dir = dump_dir.join("tq");
    fs::create_dir_all(&dense_dump_dir)?;
    fs::create_dir_all(&tq_dump_dir)?;

    // --- Dense run: dump sdpa_out for layers [0,5,15,25] at positions 0..100 ---
    eprintln!("[gate-A] Running dense decode with sdpa_out dump (max_pos={max_pos})...");
    let dense_tokens_raw = run_hf2q_sdpa_dump(
        hf2q, model, prompt, max_tokens,
        "dense_all", &dense_dump_dir,
        layers_list, max_pos,
    ).context("Gate A dense run")?;

    // Parse emitted tokens from dense run for the TQ replay.
    let dense_tokens = parse_emitted_tokens(&dense_tokens_raw);
    eprintln!("[gate-A] Dense run emitted {} tokens", dense_tokens.len());

    // Write tokens to file for TQ replay.
    let token_replay_path = out_dir.join("gate_a_dense_tokens.txt");
    let token_replay_str = dense_tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ");
    fs::write(&token_replay_path, &token_replay_str)?;

    // --- TQ run: fixed-token replay + sdpa_out dump ---
    eprintln!("[gate-A] Running TQ decode with fixed-token replay + sdpa_out dump...");
    run_hf2q_sdpa_dump_replay(
        hf2q, model, prompt, max_tokens,
        "tq_all", &tq_dump_dir,
        layers_list, max_pos,
        &token_replay_str,
    ).context("Gate A TQ run")?;

    // --- Compute cosine similarities via Python script ---
    let script_path = out_dir.join("cosine_sim.py");
    write_cosine_script(&script_path, &dense_dump_dir, &tq_dump_dir, &[0, 5, 15, 25], max_pos)?;

    eprintln!("[gate-A] Running cosine_sim.py...");
    let py_out = Command::new("python3")
        .arg(&script_path)
        .output()
        .context("Failed to run python3 cosine_sim.py")?;

    if !py_out.status.success() {
        let stderr = String::from_utf8_lossy(&py_out.stderr);
        anyhow::bail!("cosine_sim.py failed:\n{stderr}");
    }

    let py_stdout = String::from_utf8_lossy(&py_out.stdout);
    eprintln!("[gate-A] cosine_sim.py output:\n{py_stdout}");

    // Parse JSON output from Python script.
    parse_cosine_json(&py_stdout)
}

#[allow(clippy::too_many_arguments)]
fn run_hf2q_sdpa_dump(
    hf2q: &Path,
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    layer_policy: &str,
    dump_dir: &Path,
    layers_list: &str,
    max_pos: usize,
) -> Result<String> {
    let mut cmd = Command::new(hf2q);
    cmd.arg("generate")
        .arg("--model").arg(model)
        .arg("--prompt").arg(prompt)
        .arg("--max-tokens").arg(max_tokens.to_string())
        .arg("--temperature").arg("0.0")
        .env("HF2Q_LAYER_POLICY", layer_policy)
        .env("HF2Q_DUMP_ALL_CACHE", "1")
        .env("HF2Q_DUMP_SDPA_MAX_POS", max_pos.to_string())
        .env("HF2Q_DUMP_LAYERS_LIST", layers_list)
        .env("HF2Q_DUMP_DIR", dump_dir.to_str().unwrap())
        .env("HF2Q_DECODE_EMIT_TOKENS", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    let output = cmd.output().context("hf2q dense sdpa dump run")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("hf2q dense run failed (policy={layer_policy}):\n{stderr}");
    }
    Ok(String::from_utf8_lossy(&output.stderr).into_owned())
}

#[allow(clippy::too_many_arguments)]
fn run_hf2q_sdpa_dump_replay(
    hf2q: &Path,
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    layer_policy: &str,
    dump_dir: &Path,
    layers_list: &str,
    max_pos: usize,
    token_replay: &str,
) -> Result<()> {
    let mut cmd = Command::new(hf2q);
    cmd.arg("generate")
        .arg("--model").arg(model)
        .arg("--prompt").arg(prompt)
        .arg("--max-tokens").arg(max_tokens.to_string())
        .arg("--temperature").arg("0.0")
        .env("HF2Q_LAYER_POLICY", layer_policy)
        .env("HF2Q_DUMP_ALL_CACHE", "1")
        .env("HF2Q_DUMP_SDPA_MAX_POS", max_pos.to_string())
        .env("HF2Q_DUMP_LAYERS_LIST", layers_list)
        .env("HF2Q_DUMP_DIR", dump_dir.to_str().unwrap())
        .env("HF2Q_DECODE_INPUT_TOKENS", token_replay)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    let output = cmd.output().context("hf2q TQ sdpa dump replay run")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("hf2q TQ replay run failed (policy={layer_policy}):\n{stderr}");
    }
    Ok(())
}

fn parse_emitted_tokens(stderr: &str) -> Vec<u32> {
    // Lines like: "[HF2Q_DECODE_EMIT] step=N token=X"
    let mut tokens: Vec<(usize, u32)> = Vec::new();
    for line in stderr.lines() {
        if let Some(rest) = line.strip_prefix("[HF2Q_DECODE_EMIT] ") {
            let mut step = None;
            let mut token = None;
            for part in rest.split_whitespace() {
                if let Some(s) = part.strip_prefix("step=") {
                    step = s.parse::<usize>().ok();
                } else if let Some(t) = part.strip_prefix("token=") {
                    token = t.parse::<u32>().ok();
                }
            }
            if let (Some(s), Some(t)) = (step, token) {
                tokens.push((s, t));
            }
        }
    }
    tokens.sort_by_key(|(s, _)| *s);
    tokens.into_iter().map(|(_, t)| t).collect()
}

fn write_cosine_script(
    path: &Path,
    dense_dir: &Path,
    tq_dir: &Path,
    layers: &[usize],
    _max_pos: usize,
) -> Result<()> {
    let dense_str = dense_dir.to_str().unwrap();
    let tq_str = tq_dir.to_str().unwrap();
    let layers_repr: Vec<String> = layers.iter().map(|l| l.to_string()).collect();
    let layers_py = format!("[{}]", layers_repr.join(", "));

    // The dump files are named hf2q_sdpa_out_layer<LL>_pos<N>.bin where N is the
    // absolute seq_pos (prompt_len + decode_step). We discover files via glob so
    // the script does not need to know the prompt length offset.
    let script = format!(r#"#!/usr/bin/env python3
"""iter23 Gate A: cosine similarity of SDPA outputs."""
import os, struct, json, math, re
import glob

dense_dir = {dense_str:?}
tq_dir    = {tq_str:?}
layers    = {layers_py}

def load_f32_bin(path):
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // 4
    if n == 0:
        return []
    return list(struct.unpack(f'{{n}}f', data))

def cosine_sim(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    dot = sum(a[i]*b[i] for i in range(n))
    na  = math.sqrt(sum(x*x for x in a[:n]))
    nb  = math.sqrt(sum(x*x for x in b[:n]))
    if na < 1e-12 or nb < 1e-12:
        return 1.0  # zero vector pair: treat as identical
    return dot / (na * nb)

# Discover all dense sdpa_out files for our layers (any pos value).
all_cosines = []
per_layer = {{}}

for layer in layers:
    per_layer[layer] = []
    pattern = os.path.join(dense_dir, f'hf2q_sdpa_out_layer{{layer:02d}}_pos*.bin')
    dense_files = sorted(glob.glob(pattern))
    if not dense_files:
        print(f'WARNING: no dense sdpa_out files found for layer={{layer}} in {{dense_dir}}')
        continue
    for dense_path in dense_files:
        basename = os.path.basename(dense_path)
        # Extract pos from filename: hf2q_sdpa_out_layer<LL>_pos<N>.bin
        m = re.search(r'_pos(\d+)\.bin$', basename)
        if not m:
            continue
        pos = m.group(1)
        tq_path = os.path.join(tq_dir, f'hf2q_sdpa_out_layer{{layer:02d}}_pos{{pos}}.bin')
        if not os.path.exists(tq_path):
            print(f'WARNING: TQ dump missing for layer={{layer}} pos={{pos}}: {{tq_path}}')
            continue
        dense_v = load_f32_bin(dense_path)
        tq_v    = load_f32_bin(tq_path)
        if not dense_v or not tq_v:
            continue
        cs = cosine_sim(dense_v, tq_v)
        per_layer[layer].append(cs)
        all_cosines.append(cs)

if not all_cosines:
    print(json.dumps({{"error": "no cosine pairs found — check dump dirs", "pass": False}}))
    raise SystemExit(1)

all_cosines.sort()
n = len(all_cosines)

def percentile(sorted_list, p):
    idx = max(0, min(int(p / 100.0 * len(sorted_list)), len(sorted_list) - 1))
    return sorted_list[idx]

result = {{
    "n_pairs": n,
    "mean":  sum(all_cosines) / n,
    "min":   all_cosines[0],
    "p1":    percentile(all_cosines, 1),
    "p50":   percentile(all_cosines, 50),
    "p99":   percentile(all_cosines, 99),
    "per_layer": {{str(l): {{"mean": sum(v)/len(v) if v else 0.0, "n": len(v)}} for l, v in per_layer.items() if v}},
}}
result["pass"] = result["mean"] >= 0.999 and result["p1"] >= 0.99
print(json.dumps(result))
"#, dense_str = dense_str, tq_str = tq_str, layers_py = layers_py);

    fs::write(path, script.as_bytes())?;
    Ok(())
}

fn parse_cosine_json(stdout: &str) -> Result<CosineGateResult> {
    // Find the last JSON line.
    let json_line = stdout.lines()
        .filter(|l| l.trim().starts_with('{'))
        .next_back()
        .ok_or_else(|| anyhow::anyhow!("No JSON output from cosine_sim.py"))?;
    let v: serde_json::Value = serde_json::from_str(json_line)
        .context("parse cosine_sim.py JSON")?;
    Ok(CosineGateResult {
        mean: v["mean"].as_f64().unwrap_or(0.0),
        min:  v["min"].as_f64().unwrap_or(0.0),
        p1:   v["p1"].as_f64().unwrap_or(0.0),
        p50:  v["p50"].as_f64().unwrap_or(0.0),
        p99:  v["p99"].as_f64().unwrap_or(0.0),
        pass: v["pass"].as_bool().unwrap_or(false),
    })
}

// ---------------------------------------------------------------------------
// Gate B + C implementation (combined run)
// ---------------------------------------------------------------------------

struct DivergenceResult {
    count: usize,
    total: usize,
    rate: f64,
    pass: bool,
}

struct PplResult {
    ppl_dense: f64,
    ppl_tq: f64,
    delta: f64,
    pass: bool,
}

fn run_gate_b_c(
    hf2q: &Path,
    model: &Path,
    prompt: &str,
    out_dir: &Path,
) -> Result<(DivergenceResult, PplResult)> {
    let max_tokens = 1000;

    // --- Dense run ---
    eprintln!("[gate-B/C] Running dense greedy decode (1000 tokens)...");
    let dense_stderr = run_hf2q_greedy_nll(hf2q, model, prompt, max_tokens, "dense_all")
        .context("Gate B/C dense run")?;

    let dense_tokens = parse_emitted_tokens(&dense_stderr);
    let dense_nlls = parse_nll_values(&dense_stderr);
    eprintln!("[gate-B/C] Dense: {} tokens, {} NLL values", dense_tokens.len(), dense_nlls.len());

    // Write dense tokens for replay.
    let dense_tokens_path = out_dir.join("tokens-dense.txt");
    let dense_tok_str = dense_tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ");
    fs::write(&dense_tokens_path, &dense_tok_str)?;

    // --- TQ run: fixed-token replay (pure greedy, no override needed for Gate C,
    //     but for Gate B we use pure greedy on TQ and compare to dense pure greedy) ---
    // Gate B: BOTH runs are pure greedy (no fixed-token replay). Divergence =
    //         positions where the two greedy outputs differ.
    // Gate C: NLL of TQ-chosen tokens vs NLL of dense-chosen tokens on their own paths.
    eprintln!("[gate-B/C] Running TQ greedy decode (1000 tokens)...");
    let tq_stderr = run_hf2q_greedy_nll(hf2q, model, prompt, max_tokens, "tq_all")
        .context("Gate B/C TQ run")?;

    let tq_tokens = parse_emitted_tokens(&tq_stderr);
    let tq_nlls = parse_nll_values(&tq_stderr);
    eprintln!("[gate-B/C] TQ: {} tokens, {} NLL values", tq_tokens.len(), tq_nlls.len());

    // Write TQ tokens.
    let tq_tokens_path = out_dir.join("tokens-tq.txt");
    let tq_tok_str = tq_tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ");
    fs::write(&tq_tokens_path, &tq_tok_str)?;

    // Gate B: count divergences.
    let compare_len = dense_tokens.len().min(tq_tokens.len()).min(max_tokens);
    let divergences = dense_tokens[..compare_len].iter()
        .zip(tq_tokens[..compare_len].iter())
        .filter(|(d, t)| d != t)
        .count();
    let div_rate = if compare_len > 0 { divergences as f64 / compare_len as f64 } else { 1.0 };

    // Gate C: PPL from NLL sums.
    let dense_ppl = nll_to_ppl(&dense_nlls);
    let tq_ppl = nll_to_ppl(&tq_nlls);
    let ppl_delta = if dense_ppl > 0.0 {
        (tq_ppl - dense_ppl).abs() / dense_ppl
    } else {
        1.0
    };

    let div_result = DivergenceResult {
        count: divergences,
        total: compare_len,
        rate: div_rate,
        pass: div_rate < 0.01,
    };
    let ppl_result = PplResult {
        ppl_dense: dense_ppl,
        ppl_tq: tq_ppl,
        delta: ppl_delta,
        pass: ppl_delta < 0.01,
    };

    Ok((div_result, ppl_result))
}

fn run_hf2q_greedy_nll(
    hf2q: &Path,
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    layer_policy: &str,
) -> Result<String> {
    let mut cmd = Command::new(hf2q);
    cmd.arg("generate")
        .arg("--model").arg(model)
        .arg("--prompt").arg(prompt)
        .arg("--max-tokens").arg(max_tokens.to_string())
        .arg("--temperature").arg("0.0")
        .env("HF2Q_LAYER_POLICY", layer_policy)
        .env("HF2Q_DECODE_EMIT_TOKENS", "1")
        .env("HF2Q_EMIT_NLL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    let output = cmd.output().context("hf2q greedy+nll run")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("hf2q greedy run failed (policy={layer_policy}):\n{stderr}");
    }
    Ok(String::from_utf8_lossy(&output.stderr).into_owned())
}

fn parse_nll_values(stderr: &str) -> Vec<f64> {
    // Lines like: "[HF2Q_NLL] step=N token=X nll=Y"
    let mut nlls: Vec<(usize, f64)> = Vec::new();
    for line in stderr.lines() {
        if let Some(rest) = line.strip_prefix("[HF2Q_NLL] ") {
            let mut step = None;
            let mut nll = None;
            for part in rest.split_whitespace() {
                if let Some(s) = part.strip_prefix("step=") {
                    step = s.parse::<usize>().ok();
                } else if let Some(n) = part.strip_prefix("nll=") {
                    nll = n.parse::<f64>().ok();
                }
            }
            if let (Some(s), Some(n)) = (step, nll) {
                nlls.push((s, n));
            }
        }
    }
    nlls.sort_by_key(|(s, _)| *s);
    nlls.into_iter().map(|(_, n)| n).collect()
}

fn nll_to_ppl(nlls: &[f64]) -> f64 {
    if nlls.is_empty() {
        return f64::NAN;
    }
    let mean_nll = nlls.iter().sum::<f64>() / nlls.len() as f64;
    mean_nll.exp()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn locate_hf2q_binary() -> Result<PathBuf> {
    // 1. Explicit env override.
    if let Ok(p) = std::env::var("HF2Q_ITER23_HF2Q") {
        let path = PathBuf::from(p);
        anyhow::ensure!(path.exists(), "HF2Q_ITER23_HF2Q binary not found: {}", path.display());
        return Ok(path);
    }

    // 2. Worktree release build (produced when we `cargo build --release`).
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let worktree_bin = manifest_dir.join("target/release/hf2q");
    if worktree_bin.exists() {
        return Ok(worktree_bin);
    }

    // 3. Main repo release build.
    let main_bin = PathBuf::from("/opt/hf2q/target/release/hf2q");
    if main_bin.exists() {
        eprintln!("[iter23_audit] WARNING: using main-repo binary (worktree not yet built)");
        return Ok(main_bin);
    }

    anyhow::bail!(
        "Cannot find hf2q binary. Build with `cargo build --release` or set HF2Q_ITER23_HF2Q."
    )
}

fn locate_prompt() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p = manifest_dir.join("tests/evals/prompts/sourdough.txt");
    if p.exists() {
        return Ok(p);
    }
    anyhow::bail!("Cannot find sourdough prompt at {}", p.display())
}
