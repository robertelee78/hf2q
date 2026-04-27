//! iter24_audit — ADR-007 iter-24 semantic gate measurement at 4/5/6/8-bit TQ.
//!
//! Measures all three gates at each bit-width and returns a `bit_width_matrix`:
//!   4 bit-widths × 3 gates = 12 PASS/FAIL cells.
//!
//! Gates (industry-standard thresholds from 2026-04-24 standing directive):
//!   Gate A: cosine sim of SDPA output (dense vs TQ).
//!           PASS iff mean ≥ 0.999 AND p1 ≥ 0.99
//!           Measured at layers [0,5,15,25], 100 decode positions each.
//!   Gate B: argmax divergence under fixed-token replay.
//!           PASS iff rate < 1% (≤ 10 per 1000 tokens).
//!           Fixed-token replay: TQ KV is built from the DENSE decode sequence,
//!           then we ask "would TQ have picked the same token?".
//!   Gate C: perplexity delta from NLL sums.
//!           PASS iff |PPL_tq - PPL_dense| / PPL_dense < 1%.
//!
//! Verdicts:
//!   ACCEPT              — smallest bit-width with all 3 gates pass is shippable
//!   ACCEPT_WITH_BIT_WIDTH — smallest B where all 3 gates pass (B > 4)
//!   REJECT              — no bit-width passes all 3 gates
//!
//! Environment:
//!   HF2Q_ITER24_MODEL   — path to GGUF (required)
//!   HF2Q_ITER24_HF2Q    — path to hf2q binary (default: worktree release build)
//!
//! Produces /tmp/cfa-iter24/audit.json with full bit_width_matrix schema.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

// Gate thresholds (industry-standard, 2026-04-24).
const GATE_A_MEAN: f64 = 0.999;
const GATE_A_P1: f64 = 0.99;
const GATE_B_RATE: f64 = 0.01;
const GATE_C_DELTA: f64 = 0.01;

const LAYERS: &[usize] = &[0, 5, 15, 25];
const GATE_A_POSITIONS: usize = 100;
const GATE_B_C_TOKENS: usize = 1000;

fn main() -> Result<()> {
    let model = std::env::var("HF2Q_ITER24_MODEL")
        .context("HF2Q_ITER24_MODEL must be set to the GGUF model path")?;
    let model_path = PathBuf::from(&model);
    anyhow::ensure!(model_path.exists(), "Model not found: {}", model_path.display());

    let hf2q_bin = locate_hf2q_binary()?;
    eprintln!("[iter24_audit] hf2q binary: {}", hf2q_bin.display());
    eprintln!("[iter24_audit] model: {}", model_path.display());

    let prompt_path = locate_prompt()?;
    let prompt_text = fs::read_to_string(&prompt_path)
        .with_context(|| format!("read prompt: {}", prompt_path.display()))?;
    let prompt_text = prompt_text.trim().to_string();
    eprintln!(
        "[iter24_audit] prompt: {:?} (len={})",
        prompt_path.display(),
        prompt_text.len()
    );

    let out_dir = PathBuf::from("/tmp/cfa-iter24");
    fs::create_dir_all(&out_dir)?;

    let t0 = Instant::now();

    // -----------------------------------------------------------------------
    // Dense baseline — run once, reuse tokens for all TQ bit-widths.
    // -----------------------------------------------------------------------
    eprintln!("\n[iter24_audit] === Dense baseline runs ===");

    let dense_dump_dir = out_dir.join("dumps/dense");
    fs::create_dir_all(&dense_dump_dir)?;

    eprintln!("[iter24_audit] Dense Gate A dump ({} positions)...", GATE_A_POSITIONS);
    let dense_gate_a_stderr = run_hf2q_sdpa_dump(
        &hf2q_bin,
        &model_path,
        &prompt_text,
        GATE_A_POSITIONS + 10,
        "dense_all",
        None, // no codebook_bits for dense
        &dense_dump_dir,
        LAYERS,
        GATE_A_POSITIONS,
    )
    .context("Dense Gate A dump")?;

    let dense_gate_a_tokens = parse_emitted_tokens(&dense_gate_a_stderr);
    eprintln!(
        "[iter24_audit] Dense Gate A emitted {} tokens",
        dense_gate_a_tokens.len()
    );
    let dense_gate_a_tok_str = tokens_to_str(&dense_gate_a_tokens);

    eprintln!(
        "[iter24_audit] Dense Gate B/C greedy ({} tokens)...",
        GATE_B_C_TOKENS
    );
    let dense_bc_stderr = run_hf2q_greedy_nll(
        &hf2q_bin,
        &model_path,
        &prompt_text,
        GATE_B_C_TOKENS,
        "dense_all",
        None,
    )
    .context("Dense Gate B/C greedy")?;

    let dense_bc_tokens = parse_emitted_tokens(&dense_bc_stderr);
    let dense_bc_nlls = parse_nll_values(&dense_bc_stderr);
    let dense_bc_tok_str = tokens_to_str(&dense_bc_tokens);
    eprintln!(
        "[iter24_audit] Dense B/C: {} tokens, {} NLL values",
        dense_bc_tokens.len(),
        dense_bc_nlls.len()
    );

    // Write dense token files for inspection.
    fs::write(out_dir.join("dense_gate_a_tokens.txt"), &dense_gate_a_tok_str)?;
    fs::write(out_dir.join("dense_gate_bc_tokens.txt"), &dense_bc_tok_str)?;

    let dense_ppl = nll_to_ppl(&dense_bc_nlls);
    eprintln!("[iter24_audit] Dense PPL: {:.4}", dense_ppl);

    // -----------------------------------------------------------------------
    // Per-bit-width measurements.
    // -----------------------------------------------------------------------
    let bit_widths: &[Option<u32>] = &[
        None,    // 4-bit native TQ (default)
        Some(5),
        Some(6),
        Some(8),
    ];

    let mut matrix: Vec<serde_json::Value> = Vec::new();

    for &codebook_bits in bit_widths {
        let bw_label = codebook_bits.map(|b| b.to_string()).unwrap_or_else(|| "4".to_string());
        let bw_u32 = codebook_bits.unwrap_or(4);
        eprintln!(
            "\n[iter24_audit] ===== {}-bit TQ measurements =====",
            bw_label
        );

        // --- Gate A ---
        let tq_dump_dir = out_dir.join(format!("dumps/tq{}bit", bw_label));
        fs::create_dir_all(&tq_dump_dir)?;

        eprintln!("[iter24_audit] {}-bit Gate A (fixed-token replay)...", bw_label);
        let gate_a_result = measure_gate_a(
            &hf2q_bin,
            &model_path,
            &prompt_text,
            codebook_bits,
            &dense_dump_dir,
            &tq_dump_dir,
            &dense_gate_a_tok_str,
            &out_dir,
            bw_u32,
        )
        .context(format!("{}-bit Gate A", bw_label))?;

        eprintln!(
            "[iter24_audit] {}-bit Gate A: mean={:.6} p1={:.6} pass={}",
            bw_label, gate_a_result.mean, gate_a_result.p1, gate_a_result.pass
        );

        // --- Gate B (fixed-token replay) ---
        eprintln!(
            "[iter24_audit] {}-bit Gate B (fixed-token replay)...",
            bw_label
        );
        let gate_b_result = measure_gate_b(
            &hf2q_bin,
            &model_path,
            &prompt_text,
            codebook_bits,
            &dense_bc_tokens,
            &dense_bc_tok_str,
        )
        .context(format!("{}-bit Gate B", bw_label))?;

        eprintln!(
            "[iter24_audit] {}-bit Gate B: divergences={}/{} rate={:.4} pass={}",
            bw_label,
            gate_b_result.count,
            gate_b_result.total,
            gate_b_result.rate,
            gate_b_result.pass
        );

        // --- Gate C ---
        eprintln!("[iter24_audit] {}-bit Gate C (PPL delta)...", bw_label);
        let gate_c_result = measure_gate_c(
            &hf2q_bin,
            &model_path,
            &prompt_text,
            codebook_bits,
            dense_ppl,
        )
        .context(format!("{}-bit Gate C", bw_label))?;

        eprintln!(
            "[iter24_audit] {}-bit Gate C: ppl_dense={:.4} ppl_tq={:.4} delta={:.6} pass={}",
            bw_label, dense_ppl, gate_c_result.ppl_tq, gate_c_result.delta, gate_c_result.pass
        );

        // Memory savings (bits per element vs F32 baseline).
        let mem_savings_vs_f32 = 1.0 - (bw_u32 as f64 / 32.0);

        matrix.push(serde_json::json!({
            "bit_width": bw_u32,
            "label": format!("{}-bit", bw_label),
            "mem_savings_vs_f32": mem_savings_vs_f32,
            "gate_a": {
                "mean": gate_a_result.mean,
                "min": gate_a_result.min,
                "p1": gate_a_result.p1,
                "p50": gate_a_result.p50,
                "p99": gate_a_result.p99,
                "n_pairs": gate_a_result.n_pairs,
                "threshold_mean": GATE_A_MEAN,
                "threshold_p1": GATE_A_P1,
                "pass": gate_a_result.pass,
            },
            "gate_b": {
                "count": gate_b_result.count,
                "total": gate_b_result.total,
                "rate": gate_b_result.rate,
                "threshold": GATE_B_RATE,
                "pass": gate_b_result.pass,
                "method": "fixed_token_replay",
            },
            "gate_c": {
                "ppl_dense": dense_ppl,
                "ppl_tq": gate_c_result.ppl_tq,
                "delta": gate_c_result.delta,
                "threshold": GATE_C_DELTA,
                "pass": gate_c_result.pass,
            },
            "all_pass": gate_a_result.pass && gate_b_result.pass && gate_c_result.pass,
        }));
    }

    let elapsed_s = t0.elapsed().as_secs_f64();

    // -----------------------------------------------------------------------
    // Verdict
    // -----------------------------------------------------------------------
    // Find smallest bit-width where all 3 gates pass.
    let shippable = matrix
        .iter()
        .find(|row| row["all_pass"].as_bool().unwrap_or(false))
        .cloned();

    let (verdict, shippable_bits, rationale) = match &shippable {
        Some(row) => {
            let bw = row["bit_width"].as_u64().unwrap_or(0);
            if bw == 4 {
                (
                    "ACCEPT",
                    Some(bw),
                    "4-bit TQ passes all 3 gates — ADR-007 closes as shippable at 4-bit.".to_string(),
                )
            } else {
                (
                    "ACCEPT_WITH_BIT_WIDTH",
                    Some(bw),
                    format!(
                        "{}-bit is the smallest shippable bit-width; \
                         4-bit fails at least one gate. Increase bit-width to {}-bit.",
                        bw, bw
                    ),
                )
            }
        }
        None => (
            "REJECT",
            None,
            "No bit-width (4/5/6/8) passes all 3 gates. Deeper investigation required.".to_string(),
        ),
    };

    eprintln!(
        "\n[iter24_audit] === VERDICT: {} ({:.0}s) ===",
        verdict, elapsed_s
    );
    eprintln!("[iter24_audit] {}", rationale);
    if let Some(bw) = shippable_bits {
        eprintln!("[iter24_audit] Shippable bit-width: {}", bw);
        // Memory savings for shippable bit-width.
        let savings = 1.0 - (bw as f64 / 32.0);
        eprintln!(
            "[iter24_audit] KV cache memory savings vs F32: {:.1}%",
            savings * 100.0
        );
    }

    // -----------------------------------------------------------------------
    // Write audit.json
    // -----------------------------------------------------------------------
    let audit = serde_json::json!({
        "schema_version": "iter24-v1",
        "iter": 24,
        "elapsed_s": elapsed_s,
        "gate_thresholds": {
            "gate_a_mean": GATE_A_MEAN,
            "gate_a_p1": GATE_A_P1,
            "gate_b_rate": GATE_B_RATE,
            "gate_c_delta": GATE_C_DELTA,
        },
        "bit_width_matrix": matrix,
        "verdict": verdict,
        "shippable_bit_width": shippable_bits,
        "verdict_rationale": rationale,
    });

    let audit_json = serde_json::to_string_pretty(&audit)?;
    let audit_path = out_dir.join("audit.json");
    fs::write(&audit_path, &audit_json)?;
    eprintln!("[iter24_audit] audit.json written to {}", audit_path.display());

    // Also write to worktree root for commit.
    let worktree_audit =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("audit.json");
    fs::write(&worktree_audit, &audit_json)?;
    eprintln!(
        "[iter24_audit] audit.json also written to {}",
        worktree_audit.display()
    );

    // Print summary table.
    eprintln!("\n[iter24_audit] === Summary ===");
    eprintln!(
        "{:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "bits", "A:mean", "A:p1", "B:rate", "C:delta", "all_pass"
    );
    for row in &matrix {
        eprintln!(
            "{:>8} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8}",
            row["label"].as_str().unwrap_or("?"),
            row["gate_a"]["mean"].as_f64().unwrap_or(0.0),
            row["gate_a"]["p1"].as_f64().unwrap_or(0.0),
            row["gate_b"]["rate"].as_f64().unwrap_or(1.0),
            row["gate_c"]["delta"].as_f64().unwrap_or(1.0),
            if row["all_pass"].as_bool().unwrap_or(false) { "PASS" } else { "FAIL" },
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Gate A — cosine similarity of SDPA outputs
// ---------------------------------------------------------------------------

struct GateAResult {
    mean: f64,
    min: f64,
    p1: f64,
    p50: f64,
    p99: f64,
    n_pairs: usize,
    pass: bool,
}

#[allow(clippy::too_many_arguments)]
fn measure_gate_a(
    hf2q: &Path,
    model: &Path,
    prompt: &str,
    codebook_bits: Option<u32>,
    dense_dump_dir: &Path,
    tq_dump_dir: &Path,
    dense_tokens_str: &str,
    out_dir: &Path,
    bw_u32: u32,
) -> Result<GateAResult> {
    // TQ run: fixed-token replay with SDPA dump.
    let tq_stderr = run_hf2q_sdpa_dump(
        hf2q,
        model,
        prompt,
        GATE_A_POSITIONS + 10,
        "tq_all",
        codebook_bits,
        tq_dump_dir,
        LAYERS,
        GATE_A_POSITIONS,
    )
    .context("Gate A TQ sdpa dump")?;
    let _ = tq_stderr; // not needed after dump

    // Fixed-token replay: also run with HF2Q_DECODE_INPUT_TOKENS so the TQ path
    // processes the exact same decode sequence as dense.
    let tq_replay_dir = out_dir.join(format!("dumps/tq{}bit_replay", bw_u32));
    fs::create_dir_all(&tq_replay_dir)?;
    run_hf2q_sdpa_dump_replay(
        hf2q,
        model,
        prompt,
        GATE_A_POSITIONS + 10,
        "tq_all",
        codebook_bits,
        &tq_replay_dir,
        LAYERS,
        GATE_A_POSITIONS,
        dense_tokens_str,
    )
    .context("Gate A TQ replay sdpa dump")?;

    // Use the replay dump for cosine comparison.
    let script_path = out_dir.join(format!("cosine_sim_{}bit.py", bw_u32));
    write_cosine_script(&script_path, dense_dump_dir, &tq_replay_dir, LAYERS, GATE_A_POSITIONS)?;

    let py_out = Command::new("python3")
        .arg(&script_path)
        .output()
        .context("python3 cosine_sim script")?;

    if !py_out.status.success() {
        let stderr = String::from_utf8_lossy(&py_out.stderr);
        anyhow::bail!("cosine_sim.py failed:\n{stderr}");
    }

    let py_stdout = String::from_utf8_lossy(&py_out.stdout);
    parse_cosine_json(&py_stdout)
}

// ---------------------------------------------------------------------------
// Gate B — argmax divergence (fixed-token replay)
// ---------------------------------------------------------------------------

struct GateBResult {
    count: usize,
    total: usize,
    rate: f64,
    pass: bool,
}

/// Fixed-token replay Gate B:
///   1. Dense greedy run → dense_tokens[0..N].
///   2. Feed dense_tokens as HF2Q_DECODE_INPUT_TOKENS to TQ.
///   3. For each step, TQ builds KV from the forced input token, runs SDPA,
///      then emits the argmax it would have picked.
///   4. Divergence = positions where TQ argmax != dense token.
fn measure_gate_b(
    hf2q: &Path,
    model: &Path,
    prompt: &str,
    codebook_bits: Option<u32>,
    dense_tokens: &[u32],
    dense_tokens_str: &str,
) -> Result<GateBResult> {
    // Run TQ with forced token inputs, collecting what token it *would* emit.
    let n = dense_tokens.len().min(GATE_B_C_TOKENS);

    let tq_stderr = run_hf2q_replay_emit(
        hf2q,
        model,
        prompt,
        n + 10,
        "tq_all",
        codebook_bits,
        dense_tokens_str,
    )
    .context("Gate B TQ replay emit")?;

    let tq_would_emit = parse_emitted_tokens(&tq_stderr);
    let compare_len = n.min(tq_would_emit.len());

    let divergences = dense_tokens[..compare_len]
        .iter()
        .zip(tq_would_emit[..compare_len].iter())
        .filter(|(d, t)| d != t)
        .count();
    let rate = if compare_len > 0 {
        divergences as f64 / compare_len as f64
    } else {
        1.0
    };

    Ok(GateBResult {
        count: divergences,
        total: compare_len,
        rate,
        pass: rate < GATE_B_RATE,
    })
}

// ---------------------------------------------------------------------------
// Gate C — PPL delta
// ---------------------------------------------------------------------------

struct GateCResult {
    ppl_tq: f64,
    delta: f64,
    pass: bool,
}

fn measure_gate_c(
    hf2q: &Path,
    model: &Path,
    prompt: &str,
    codebook_bits: Option<u32>,
    dense_ppl: f64,
) -> Result<GateCResult> {
    let tq_stderr = run_hf2q_greedy_nll(
        hf2q,
        model,
        prompt,
        GATE_B_C_TOKENS,
        "tq_all",
        codebook_bits,
    )
    .context("Gate C TQ greedy nll")?;

    let tq_nlls = parse_nll_values(&tq_stderr);
    let tq_ppl = nll_to_ppl(&tq_nlls);
    let delta = if dense_ppl > 0.0 && dense_ppl.is_finite() {
        (tq_ppl - dense_ppl).abs() / dense_ppl
    } else {
        1.0
    };

    Ok(GateCResult {
        ppl_tq: tq_ppl,
        delta,
        pass: delta < GATE_C_DELTA,
    })
}

// ---------------------------------------------------------------------------
// hf2q invocation helpers
// ---------------------------------------------------------------------------

/// Dense or TQ SDPA dump run. Collects sdpa_out binary files to dump_dir.
#[allow(clippy::too_many_arguments)]
fn run_hf2q_sdpa_dump(
    hf2q: &Path,
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    layer_policy: &str,
    codebook_bits: Option<u32>,
    dump_dir: &Path,
    layers: &[usize],
    max_pos: usize,
) -> Result<String> {
    let layers_str: Vec<String> = layers.iter().map(|l| l.to_string()).collect();
    let layers_list = layers_str.join(",");

    let mut cmd = Command::new(hf2q);
    cmd.arg("generate")
        .arg("--model")
        .arg(model)
        .arg("--prompt")
        .arg(prompt)
        .arg("--max-tokens")
        .arg(max_tokens.to_string())
        .arg("--temperature")
        .arg("0.0")
        .env("HF2Q_LAYER_POLICY", layer_policy)
        .env("HF2Q_DUMP_ALL_CACHE", "1")
        .env("HF2Q_DUMP_SDPA_MAX_POS", max_pos.to_string())
        .env("HF2Q_DUMP_LAYERS_LIST", &layers_list)
        .env("HF2Q_DUMP_DIR", dump_dir.to_str().unwrap())
        .env("HF2Q_DECODE_EMIT_TOKENS", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    if let Some(bits) = codebook_bits {
        cmd.env("HF2Q_TQ_CODEBOOK_BITS", bits.to_string());
    }

    let output = cmd.output().context("hf2q sdpa dump run")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "hf2q sdpa dump failed (policy={layer_policy}, bits={codebook_bits:?}):\n{stderr}"
        );
    }
    Ok(String::from_utf8_lossy(&output.stderr).into_owned())
}

/// TQ SDPA dump run with fixed-token replay.
#[allow(clippy::too_many_arguments)]
fn run_hf2q_sdpa_dump_replay(
    hf2q: &Path,
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    layer_policy: &str,
    codebook_bits: Option<u32>,
    dump_dir: &Path,
    layers: &[usize],
    max_pos: usize,
    token_replay: &str,
) -> Result<()> {
    let layers_str: Vec<String> = layers.iter().map(|l| l.to_string()).collect();
    let layers_list = layers_str.join(",");

    let mut cmd = Command::new(hf2q);
    cmd.arg("generate")
        .arg("--model")
        .arg(model)
        .arg("--prompt")
        .arg(prompt)
        .arg("--max-tokens")
        .arg(max_tokens.to_string())
        .arg("--temperature")
        .arg("0.0")
        .env("HF2Q_LAYER_POLICY", layer_policy)
        .env("HF2Q_DUMP_ALL_CACHE", "1")
        .env("HF2Q_DUMP_SDPA_MAX_POS", max_pos.to_string())
        .env("HF2Q_DUMP_LAYERS_LIST", &layers_list)
        .env("HF2Q_DUMP_DIR", dump_dir.to_str().unwrap())
        .env("HF2Q_DECODE_INPUT_TOKENS", token_replay)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    if let Some(bits) = codebook_bits {
        cmd.env("HF2Q_TQ_CODEBOOK_BITS", bits.to_string());
    }

    let output = cmd.output().context("hf2q sdpa dump replay run")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "hf2q sdpa dump replay failed (policy={layer_policy}, bits={codebook_bits:?}):\n{stderr}"
        );
    }
    Ok(())
}

/// TQ run with forced input tokens; collects what argmax the model *would* pick.
fn run_hf2q_replay_emit(
    hf2q: &Path,
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    layer_policy: &str,
    codebook_bits: Option<u32>,
    token_replay: &str,
) -> Result<String> {
    let mut cmd = Command::new(hf2q);
    cmd.arg("generate")
        .arg("--model")
        .arg(model)
        .arg("--prompt")
        .arg(prompt)
        .arg("--max-tokens")
        .arg(max_tokens.to_string())
        .arg("--temperature")
        .arg("0.0")
        .env("HF2Q_LAYER_POLICY", layer_policy)
        .env("HF2Q_DECODE_INPUT_TOKENS", token_replay)
        .env("HF2Q_DECODE_EMIT_TOKENS", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    if let Some(bits) = codebook_bits {
        cmd.env("HF2Q_TQ_CODEBOOK_BITS", bits.to_string());
    }

    let output = cmd.output().context("hf2q replay emit run")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "hf2q replay emit failed (policy={layer_policy}, bits={codebook_bits:?}):\n{stderr}"
        );
    }
    Ok(String::from_utf8_lossy(&output.stderr).into_owned())
}

/// Greedy decode with NLL output.
fn run_hf2q_greedy_nll(
    hf2q: &Path,
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    layer_policy: &str,
    codebook_bits: Option<u32>,
) -> Result<String> {
    let mut cmd = Command::new(hf2q);
    cmd.arg("generate")
        .arg("--model")
        .arg(model)
        .arg("--prompt")
        .arg(prompt)
        .arg("--max-tokens")
        .arg(max_tokens.to_string())
        .arg("--temperature")
        .arg("0.0")
        .env("HF2Q_LAYER_POLICY", layer_policy)
        .env("HF2Q_DECODE_EMIT_TOKENS", "1")
        .env("HF2Q_EMIT_NLL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    if let Some(bits) = codebook_bits {
        cmd.env("HF2Q_TQ_CODEBOOK_BITS", bits.to_string());
    }

    let output = cmd.output().context("hf2q greedy nll run")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "hf2q greedy nll failed (policy={layer_policy}, bits={codebook_bits:?}):\n{stderr}"
        );
    }
    Ok(String::from_utf8_lossy(&output.stderr).into_owned())
}

// ---------------------------------------------------------------------------
// Python cosine script
// ---------------------------------------------------------------------------

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

    let script = format!(
        r#"#!/usr/bin/env python3
"""iter24 Gate A: cosine similarity of SDPA outputs."""
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
        return 1.0
    return dot / (na * nb)

all_cosines = []
per_layer = {{}}

for layer in layers:
    per_layer[layer] = []
    pattern = os.path.join(dense_dir, f'hf2q_sdpa_out_layer{{layer:02d}}_pos*.bin')
    dense_files = sorted(glob.glob(pattern))
    if not dense_files:
        print(f'WARNING: no dense sdpa_out files found for layer={{layer}} in {{dense_dir}}', flush=True)
        continue
    for dense_path in dense_files:
        basename = os.path.basename(dense_path)
        m = re.search(r'_pos(\d+)\.bin$', basename)
        if not m:
            continue
        pos = m.group(1)
        tq_path = os.path.join(tq_dir, f'hf2q_sdpa_out_layer{{layer:02d}}_pos{{pos}}.bin')
        if not os.path.exists(tq_path):
            continue
        dense_v = load_f32_bin(dense_path)
        tq_v    = load_f32_bin(tq_path)
        if not dense_v or not tq_v:
            continue
        cs = cosine_sim(dense_v, tq_v)
        per_layer[layer].append(cs)
        all_cosines.append(cs)

if not all_cosines:
    print(json.dumps({{"error": "no cosine pairs found", "pass": False}}))
    raise SystemExit(1)

all_cosines.sort()
n = len(all_cosines)

def percentile(sl, p):
    idx = max(0, min(int(p / 100.0 * len(sl)), len(sl) - 1))
    return sl[idx]

result = {{
    "n_pairs": n,
    "mean":  sum(all_cosines) / n,
    "min":   all_cosines[0],
    "p1":    percentile(all_cosines, 1),
    "p50":   percentile(all_cosines, 50),
    "p99":   percentile(all_cosines, 99),
    "per_layer": {{str(l): {{"mean": sum(v)/len(v) if v else 0.0, "n": len(v)}} for l, v in per_layer.items() if v}},
}}
result["pass"] = result["mean"] >= {gate_a_mean} and result["p1"] >= {gate_a_p1}
print(json.dumps(result))
"#,
        dense_str = dense_str,
        tq_str = tq_str,
        layers_py = layers_py,
        gate_a_mean = GATE_A_MEAN,
        gate_a_p1 = GATE_A_P1,
    );

    fs::write(path, script.as_bytes())?;
    Ok(())
}

fn parse_cosine_json(stdout: &str) -> Result<GateAResult> {
    let json_line = stdout
        .lines()
        .filter(|l| l.trim().starts_with('{'))
        .next_back()
        .ok_or_else(|| anyhow::anyhow!("No JSON output from cosine_sim.py"))?;
    let v: serde_json::Value =
        serde_json::from_str(json_line).context("parse cosine_sim.py JSON")?;
    Ok(GateAResult {
        mean: v["mean"].as_f64().unwrap_or(0.0),
        min: v["min"].as_f64().unwrap_or(0.0),
        p1: v["p1"].as_f64().unwrap_or(0.0),
        p50: v["p50"].as_f64().unwrap_or(0.0),
        p99: v["p99"].as_f64().unwrap_or(0.0),
        n_pairs: v["n_pairs"].as_u64().unwrap_or(0) as usize,
        pass: v["pass"].as_bool().unwrap_or(false),
    })
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

fn parse_emitted_tokens(stderr: &str) -> Vec<u32> {
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

fn parse_nll_values(stderr: &str) -> Vec<f64> {
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

fn tokens_to_str(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Binary / prompt location
// ---------------------------------------------------------------------------

fn locate_hf2q_binary() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("HF2Q_ITER24_HF2Q") {
        let path = PathBuf::from(p);
        anyhow::ensure!(
            path.exists(),
            "HF2Q_ITER24_HF2Q binary not found: {}",
            path.display()
        );
        return Ok(path);
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let worktree_bin = manifest_dir.join("target/release/hf2q");
    if worktree_bin.exists() {
        return Ok(worktree_bin);
    }

    let main_bin = PathBuf::from("/opt/hf2q/target/release/hf2q");
    if main_bin.exists() {
        eprintln!("[iter24_audit] WARNING: using main-repo binary (worktree not yet built)");
        return Ok(main_bin);
    }

    anyhow::bail!(
        "Cannot find hf2q binary. Build with `cargo build --release` or set HF2Q_ITER24_HF2Q."
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
