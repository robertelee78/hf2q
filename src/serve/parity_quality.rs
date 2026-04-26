//! ADR-007 §853-866 Gate H — TQ-active quality envelope check.
//!
//! Companion to the byte-prefix Gates (C/D/E/F).  Runs an in-process
//! two-regime decode loop on a single model load:
//!
//!   Pass 1 (`DecodeRegime::ForceDense`) — capture the dense token
//!   sequence (greedy argmax) + dense per-step NLL + per-(layer,position)
//!   dense `sdpa_out` dumps.
//!
//!   Pass 2 (`DecodeRegime::ForceTq`) — replay the dense token sequence
//!   via [`MlxModelWeights::set_replay_tokens`] so each TQ-step's logits
//!   are scored against the SAME token id dense produced (the ADR-007
//!   §853-866 PPL input shape), capture per-step TQ NLL + TQ-side
//!   `sdpa_out` dumps, and record argmax-flip events (TQ pre-replay
//!   argmax != dense argmax).
//!
//! Synthesis: per-(layer,pos) cosine_pairwise_f32(dense_sdpa, tq_sdpa)
//! aggregated to mean / min / p1 / p50 / p99; PPL Δ = |PPL_tq -
//! PPL_dense| / PPL_dense; argmax flip rate = mismatches / steps.
//!
//! This module exposes two CLI entry points:
//!
//!   - [`cmd_parity_capture_tq_quality`] — produces
//!     `<output_dir>/<prompt>_tq_quality.json` with the day-of-capture
//!     envelope baked in (the frozen reference).
//!   - [`cmd_parity_check_tq_quality`] — loads the frozen fixture and
//!     verifies a fresh TQ-active replay run still meets the floors
//!     (cosine_mean, cosine_p1, argmax_max, ppl_delta_max).
//!
//! W12 design choices preserved:
//!   - Cosine values are kept in full (~30k f32 = ~120 KB) so quantiles
//!     come from a deterministic sort, not streaming approximation.
//!   - SDPA dumps go to disk and are renamed between passes
//!     (`<dump_dir>/dense/` vs `<dump_dir>/tq/`).  We don't hold raw
//!     SDPA outputs across passes (would be ~1 GB at 30 layers × 1000
//!     positions × 8192 elements × 4B).
//!   - INVESTIGATION_ENV is a `LazyLock` that freezes at first access,
//!     so we set the dump-related env vars BEFORE constructing
//!     `MlxModelWeights` and switch regimes via the per-instance
//!     `set_decode_regime` / `set_replay_tokens` setters (which were
//!     designed in iter-108a precisely for this caller).
//!
//! Wired into release-check.sh's Gate H block (single invocation; thermal-
//! aware ordering not critical since Gate H is comparison-based across
//! one process).  Pure Rust — no Python, no shell-out for synthesis.

use anyhow::{Context, Result};
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::Path;
use std::time::SystemTime;

use crate::cli;
use crate::serve::config::Gemma4Config;
use crate::serve::forward_mlx::{
    cosine_pairwise_f32, DecodeRegime, MlxModelWeights,
};
use crate::serve::gpu;
use crate::serve::header;

/// Aggregate cosine-similarity statistics over all (layer, position) pairs.
#[derive(Debug, Clone)]
pub struct CosineStats {
    pub mean: f32,
    pub min: f32,
    pub p1: f32,
    pub p50: f32,
    pub p99: f32,
    pub n_pairs: usize,
}

/// Full Gate H envelope synthesized from a two-regime decode run.
#[derive(Debug, Clone)]
pub struct GateHEnvelope {
    pub cosine: CosineStats,
    pub argmax_flip_rate: f32,
    pub ppl_dense: f64,
    pub ppl_tq: f64,
    pub ppl_delta_pct: f64, // |PPL_tq − PPL_dense| / PPL_dense (fraction)
    pub n_steps: usize,
}

/// Per-step capture record from a single decode pass.
#[derive(Debug, Clone)]
struct PassCapture {
    /// Tokens picked by the model's argmax at each step (PRE-replay).
    /// In a replay pass these are the live-argmax picks; the *returned*
    /// token id from `forward_decode` is the replay token.
    pre_replay_argmax: Vec<u32>,
    /// Tokens actually returned by `forward_decode` (post-replay).  In a
    /// dense pass with no replay, equals `pre_replay_argmax`.
    final_tokens: Vec<u32>,
    /// Per-step NLL of `final_tokens[i]` under this pass's logits.
    nll_per_step: Vec<f32>,
}

/// `cmd_parity capture --tq-quality` — produce the frozen Gate H fixture.
///
/// Spec:
/// ```json
/// {
///   "git_head": "<sha>",
///   "model_sha256": "<sha>",
///   "prompt": "<name>",
///   "max_tokens": <usize>,
///   "captured_at": "<ISO-8601>",
///   "dense_tokens": [<u32>...],
///   "dense_nll_per_step": [<f32>...],
///   "envelope": {
///     "cosine_mean": <f32>,
///     "cosine_p1": <f32>,
///     "argmax_div": <f32>,
///     "ppl_delta_pct": <f64>
///   }
/// }
/// ```
pub fn cmd_parity_capture_tq_quality(
    model_path: &Path,
    output_dir: &Path,
    prompt_name: &str,
    max_tokens: Option<usize>,
) -> Result<()> {
    if prompt_name == "all" {
        anyhow::bail!(
            "parity capture --tq-quality requires a single prompt; \
             `--prompt all` is not supported (Gate H fixtures are per-prompt)"
        );
    }

    let evals_dir = Path::new("tests/evals");
    let prompt_file = evals_dir
        .join("prompts")
        .join(format!("{prompt_name}.txt"));
    anyhow::ensure!(
        prompt_file.exists(),
        "Prompt file not found: {}",
        prompt_file.display()
    );
    let prompt_text = fs::read_to_string(&prompt_file)?.trim().to_string();

    // Default to 1000 tokens for sourdough; smaller prompts inherit the
    // existing manifest's bound when present, else fall back to 1000.
    let tokens = max_tokens.unwrap_or(1000);

    let dump_root = std::env::temp_dir().join(format!(
        "hf2q_gate_h_capture_{}",
        std::process::id()
    ));
    fs::create_dir_all(&dump_root)?;
    let dense_dump_dir = dump_root.join("dense");
    let tq_dump_dir = dump_root.join("tq");
    fs::create_dir_all(&dense_dump_dir)?;
    fs::create_dir_all(&tq_dump_dir)?;

    eprintln!("=== Gate H Capture: {} ===", prompt_name);
    eprintln!("Model:        {}", model_path.display());
    eprintln!("Prompt:       {} ({} chars)", prompt_name, prompt_text.len());
    eprintln!("Tokens:       {}", tokens);
    eprintln!("Dump root:    {}", dump_root.display());
    eprintln!();

    let (envelope, dense_capture) = run_two_regime_decode(
        model_path,
        &prompt_text,
        tokens,
        &dump_root,
        &dense_dump_dir,
        &tq_dump_dir,
    )?;

    let model_sha = sha256_file(model_path)
        .with_context(|| format!("hash model: {}", model_path.display()))?;
    let git_head = git_head_sha();
    let captured_at = iso8601_utc_now();

    let fixture = serde_json::json!({
        "git_head": git_head,
        "model_sha256": model_sha,
        "prompt": prompt_name,
        "max_tokens": tokens,
        "captured_at": captured_at,
        "dense_tokens": dense_capture.final_tokens,
        "dense_nll_per_step": dense_capture
            .nll_per_step
            .iter()
            .map(|n| *n as f64)
            .collect::<Vec<_>>(),
        "envelope": {
            "cosine_mean": envelope.cosine.mean as f64,
            "cosine_p1": envelope.cosine.p1 as f64,
            "argmax_div": envelope.argmax_flip_rate as f64,
            "ppl_delta_pct": envelope.ppl_delta_pct,
        }
    });

    fs::create_dir_all(output_dir)?;
    let out_path = output_dir.join(format!("{prompt_name}_tq_quality.json"));
    let pretty = serde_json::to_string_pretty(&fixture)
        .context("serialize fixture")?;
    fs::write(&out_path, pretty)?;
    eprintln!();
    eprintln!("Wrote fixture: {}", out_path.display());
    eprintln!(
        "  cosine_mean={:.6}  cosine_p1={:.6}  argmax_div={:.4}  ppl_delta={:.4}",
        envelope.cosine.mean,
        envelope.cosine.p1,
        envelope.argmax_flip_rate,
        envelope.ppl_delta_pct
    );

    // Best-effort cleanup of dump scratch dir; ignore errors so we don't
    // mask a successful capture on a transient FS hiccup.
    let _ = fs::remove_dir_all(&dump_root);

    Ok(())
}

/// `cmd_parity check --tq-quality` — verify a fresh TQ-active run still
/// meets the frozen Gate H envelope floors.
#[allow(clippy::too_many_arguments)]
pub fn cmd_parity_check_tq_quality(
    model_path: &Path,
    prompt_name: &str,
    fixture_path: &Path,
    cosine_mean_floor: f32,
    cosine_p1_floor: f32,
    argmax_max: f32,
    ppl_delta_max: f32,
    max_tokens: Option<usize>,
) -> Result<()> {
    anyhow::ensure!(
        fixture_path.exists(),
        "Gate H fixture not found: {}.\n\
         Hint: run `hf2q parity capture --tq-quality --model <gguf> \
         --prompt {prompt_name}` first (iter-112 closes this loop).",
        fixture_path.display()
    );
    let fixture_str = fs::read_to_string(fixture_path)
        .with_context(|| format!("read fixture: {}", fixture_path.display()))?;
    let fixture: JsonValue =
        serde_json::from_str(&fixture_str).context("parse fixture JSON")?;

    let fixture_prompt = fixture["prompt"].as_str().unwrap_or("");
    anyhow::ensure!(
        fixture_prompt == prompt_name,
        "Fixture prompt mismatch: fixture has {:?}, --prompt is {:?}",
        fixture_prompt,
        prompt_name
    );
    let fixture_tokens = fixture["max_tokens"].as_u64().unwrap_or(0) as usize;
    let tokens = max_tokens.unwrap_or(fixture_tokens.max(1));
    anyhow::ensure!(
        tokens == fixture_tokens,
        "Token count mismatch: fixture max_tokens={fixture_tokens}, \
         --max-tokens={tokens}.  Re-capture or pass --max-tokens={fixture_tokens}."
    );

    // Optional model-sha guard: warn but don't block — running against a
    // different quant of the same logical model is a legitimate use case
    // for the operator gate.
    let cur_sha = sha256_file(model_path).unwrap_or_else(|_| "unknown".into());
    if let Some(fixture_sha) = fixture["model_sha256"].as_str() {
        if fixture_sha != cur_sha {
            eprintln!(
                "[gate-h] warning: model_sha256 mismatch (fixture={}, current={})",
                fixture_sha, cur_sha
            );
        }
    }

    let evals_dir = Path::new("tests/evals");
    let prompt_file = evals_dir
        .join("prompts")
        .join(format!("{prompt_name}.txt"));
    anyhow::ensure!(
        prompt_file.exists(),
        "Prompt file not found: {}",
        prompt_file.display()
    );
    let prompt_text = fs::read_to_string(&prompt_file)?.trim().to_string();

    let dump_root = std::env::temp_dir().join(format!(
        "hf2q_gate_h_check_{}",
        std::process::id()
    ));
    fs::create_dir_all(&dump_root)?;
    let dense_dump_dir = dump_root.join("dense");
    let tq_dump_dir = dump_root.join("tq");
    fs::create_dir_all(&dense_dump_dir)?;
    fs::create_dir_all(&tq_dump_dir)?;

    eprintln!("=== Gate H Check: {} ===", prompt_name);
    eprintln!("Model:        {}", model_path.display());
    eprintln!("Fixture:      {}", fixture_path.display());
    eprintln!("Tokens:       {}", tokens);
    eprintln!(
        "Floors:       cosine_mean>={cosine_mean_floor:.6}  \
         cosine_p1>={cosine_p1_floor:.6}  \
         argmax<={argmax_max:.4}  \
         ppl_delta<={ppl_delta_max:.4}"
    );
    eprintln!();

    let (envelope, _dense_capture) = run_two_regime_decode(
        model_path,
        &prompt_text,
        tokens,
        &dump_root,
        &dense_dump_dir,
        &tq_dump_dir,
    )?;

    let _ = fs::remove_dir_all(&dump_root);

    eprintln!();
    eprintln!("=== Gate H Envelope (this run) ===");
    eprintln!(
        "  cosine: mean={:.6}  min={:.6}  p1={:.6}  p50={:.6}  p99={:.6}  n_pairs={}",
        envelope.cosine.mean,
        envelope.cosine.min,
        envelope.cosine.p1,
        envelope.cosine.p50,
        envelope.cosine.p99,
        envelope.cosine.n_pairs,
    );
    eprintln!(
        "  argmax_flip_rate: {:.4}  ({} flips / {} steps)",
        envelope.argmax_flip_rate,
        (envelope.argmax_flip_rate * envelope.n_steps as f32).round() as usize,
        envelope.n_steps,
    );
    eprintln!(
        "  PPL: dense={:.4}  tq={:.4}  delta={:.4}",
        envelope.ppl_dense, envelope.ppl_tq, envelope.ppl_delta_pct,
    );

    // Compare against the frozen fixture envelope as a sanity context line.
    if let Some(fixture_env) = fixture.get("envelope") {
        eprintln!();
        eprintln!("=== Fixture Envelope (frozen reference) ===");
        eprintln!(
            "  cosine_mean={:.6}  cosine_p1={:.6}  argmax_div={:.4}  ppl_delta={:.4}",
            fixture_env["cosine_mean"].as_f64().unwrap_or(f64::NAN),
            fixture_env["cosine_p1"].as_f64().unwrap_or(f64::NAN),
            fixture_env["argmax_div"].as_f64().unwrap_or(f64::NAN),
            fixture_env["ppl_delta_pct"].as_f64().unwrap_or(f64::NAN),
        );
    }

    let mut failures: Vec<String> = Vec::new();
    if envelope.cosine.mean < cosine_mean_floor {
        failures.push(format!(
            "cosine_mean {:.6} < floor {:.6}",
            envelope.cosine.mean, cosine_mean_floor
        ));
    }
    if envelope.cosine.p1 < cosine_p1_floor {
        failures.push(format!(
            "cosine_p1 {:.6} < floor {:.6}",
            envelope.cosine.p1, cosine_p1_floor
        ));
    }
    if envelope.argmax_flip_rate > argmax_max {
        failures.push(format!(
            "argmax_flip_rate {:.4} > max {:.4}",
            envelope.argmax_flip_rate, argmax_max
        ));
    }
    if (envelope.ppl_delta_pct as f32) > ppl_delta_max {
        failures.push(format!(
            "ppl_delta {:.4} > max {:.4}",
            envelope.ppl_delta_pct, ppl_delta_max
        ));
    }

    eprintln!();
    if failures.is_empty() {
        println!("PASS: Gate H envelope holds (ADR-007 §853-866)");
        Ok(())
    } else {
        for f in &failures {
            eprintln!("FAIL: {}", f);
        }
        anyhow::bail!("Gate H failed: {} threshold(s) tripped", failures.len())
    }
}

/// Two-regime in-process decode loop — the heart of Gate H.
///
/// Steps:
///  1. Set dump-related env vars BEFORE the first `INVESTIGATION_ENV`
///     access (the `LazyLock` freezes on first read).
///  2. Load model once.
///  3. Pass 1 — `ForceDense`: prefill + decode `tokens` steps, capturing
///     argmax + NLL.  After each `forward_decode` move the per-position
///     SDPA dump file from `<dump_root>/hf2q_sdpa_out_layer*_pos*.bin`
///     into `<dense_dump_dir>/`.
///  4. Pass 2 — `ForceTq` + `set_replay_tokens(dense_tokens)`: replay
///     prefill + decode, capturing pre-replay argmax + NLL on the
///     replayed token at each step.  Move SDPA dumps into
///     `<tq_dump_dir>/`.
///  5. Synthesize cosine / argmax-flip / PPL Δ.
fn run_two_regime_decode(
    model_path: &Path,
    prompt_text: &str,
    tokens: usize,
    dump_root: &Path,
    dense_dump_dir: &Path,
    tq_dump_dir: &Path,
) -> Result<(GateHEnvelope, PassCapture)> {
    // STEP 1 — env-var setup.  Must happen before any code path triggers
    // INVESTIGATION_ENV's LazyLock (model load reads it; so does the dump
    // helper).  All three vars are READ-ONLY diagnostics; none affects
    // forward-pass math.
    //
    // Safety: setting env vars from the calling process is sound when no
    // other thread is concurrently reading them; cmd_parity is a single-
    // threaded entry point, so there's no race here.
    unsafe {
        std::env::set_var("HF2Q_DUMP_DIR", dump_root.as_os_str());
        std::env::set_var("HF2Q_DUMP_ALL_CACHE", "1");
        std::env::set_var(
            "HF2Q_DUMP_SDPA_MAX_POS",
            tokens.to_string(),
        );
    }

    // STEP 2 — load model.
    let tokenizer_path = super::find_tokenizer(model_path, None)?;
    let config_path = super::find_config(model_path, None)?;
    let cfg = Gemma4Config::from_config_json(&config_path)?;
    let mut ctx = gpu::GpuContext::new()
        .map_err(|e| anyhow::anyhow!("GPU init: {e}"))?;
    let gguf = mlx_native::gguf::GgufFile::open(model_path)
        .map_err(|e| anyhow::anyhow!("GGUF open: {e}"))?;
    let mut progress = header::LoadProgress::new(false, 1, 0);
    let mut mlx_w =
        MlxModelWeights::load_from_gguf(&gguf, &cfg, &mut ctx, &mut progress)?;

    let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer: {e}"))?;
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow::anyhow!("Tokenizer truncation: {e}"))?;

    let rendered = super::render_chat_template(
        &gguf,
        &cli::GenerateArgs {
            model: model_path.to_path_buf(),
            prompt: Some(prompt_text.to_string()),
            prompt_file: None,
            tokenizer: None,
            config: None,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            max_tokens: tokens,
            chat_template: None,
            chat_template_file: None,
            benchmark: false,
            speculative: false,
        },
        prompt_text,
    )?;
    let encoding = tokenizer
        .encode(rendered.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenize: {e}"))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = prompt_tokens.len();

    // Per-pass per-position decode-step → seq_pos mapping.  forward_prefill
    // returns the FIRST decoded token (at seq_pos = prompt_len), then we
    // call forward_decode for steps 1..tokens at seq_pos = prompt_len + i.
    //
    // The SDPA dump file for layer `L` at decode step `i` is named
    // `hf2q_sdpa_out_layer{L:02}_pos{seq_pos}.bin`.  We translate seq_pos
    // → step index when iterating later.

    // STEP 3 — Pass 1: dense.
    eprintln!("[gate-h] Pass 1/2: dense forced regime ({} tokens)...", tokens);
    mlx_w.set_decode_regime(DecodeRegime::ForceDense);
    let dense_capture = run_one_pass(
        &mut mlx_w,
        &mut ctx,
        &prompt_tokens,
        tokens,
        /*replay=*/ None,
    )?;
    move_sdpa_dumps(dump_root, dense_dump_dir, prompt_len, tokens)?;

    // STEP 4 — Pass 2: TQ + replay.
    eprintln!(
        "[gate-h] Pass 2/2: TQ forced regime + dense-token replay ({} tokens)...",
        tokens
    );
    mlx_w.set_decode_regime(DecodeRegime::ForceTq);
    mlx_w.set_replay_tokens(dense_capture.final_tokens.clone());
    let tq_capture = run_one_pass(
        &mut mlx_w,
        &mut ctx,
        &prompt_tokens,
        tokens,
        Some(&dense_capture.final_tokens),
    )?;
    move_sdpa_dumps(dump_root, tq_dump_dir, prompt_len, tokens)?;

    // STEP 5 — synthesize.
    let num_layers = mlx_w.layers.len();
    let cosine = synthesize_cosine(
        dense_dump_dir,
        tq_dump_dir,
        num_layers,
        prompt_len,
        tokens,
        &mlx_w,
    )?;

    // PPL delta from per-step NLLs.
    let ppl_dense = nll_to_ppl(&dense_capture.nll_per_step);
    let ppl_tq = nll_to_ppl(&tq_capture.nll_per_step);
    let ppl_delta_pct = if ppl_dense > 0.0 && ppl_dense.is_finite() {
        ((ppl_tq - ppl_dense).abs() / ppl_dense) as f64
    } else {
        f64::NAN
    };

    // Argmax flip rate: TQ pre-replay argmax vs dense final tokens.
    let n = dense_capture
        .final_tokens
        .len()
        .min(tq_capture.pre_replay_argmax.len());
    let mut flips = 0usize;
    for i in 0..n {
        if tq_capture.pre_replay_argmax[i] != dense_capture.final_tokens[i] {
            flips += 1;
        }
    }
    let argmax_flip_rate = if n == 0 {
        0.0
    } else {
        flips as f32 / n as f32
    };

    let envelope = GateHEnvelope {
        cosine,
        argmax_flip_rate,
        ppl_dense: ppl_dense as f64,
        ppl_tq: ppl_tq as f64,
        ppl_delta_pct,
        n_steps: n,
    };

    Ok((envelope, dense_capture))
}

/// Run one prefill+decode pass capturing per-step argmax + NLL.
///
/// `replay` is informational (used for assertion + clarity); the actual
/// override is wired through `MlxModelWeights::set_replay_tokens` which
/// the caller has already configured.  When `replay` is provided the
/// returned `pre_replay_argmax` records the LIVE argmax (i.e., what the
/// model would have picked) so the caller can compute argmax-flip rate.
fn run_one_pass(
    mlx_w: &mut MlxModelWeights,
    ctx: &mut gpu::GpuContext,
    prompt_tokens: &[u32],
    tokens: usize,
    replay: Option<&[u32]>,
) -> Result<PassCapture> {
    let first_returned = mlx_w.forward_prefill(prompt_tokens, tokens, ctx)?;
    // After prefill the per-instance state has been reset and decode_step
    // is back to 0; `forward_decode` will replay from index 0 going
    // forward.  Compute the prefill's argmax-equivalent: when `replay` is
    // Some, the replay vector overrides what `forward_prefill` returns —
    // but `forward_prefill` is NOT replay-aware (only `forward_decode` is),
    // so `first_returned` IS the live argmax.  When replay is provided,
    // the *reported* dense-token at this step is replay[0]; record both.
    //
    // NLL: read the live logits buffer left behind by forward_prefill /
    // forward_decode and compute -log P(scored_token).
    let scored_first = match replay {
        Some(r) if !r.is_empty() => r[0],
        _ => first_returned,
    };
    let first_nll = mlx_w.token_nll_from_logits(scored_first)?;

    let mut pre_replay_argmax: Vec<u32> = Vec::with_capacity(tokens);
    let mut final_tokens: Vec<u32> = Vec::with_capacity(tokens);
    let mut nll_per_step: Vec<f32> = Vec::with_capacity(tokens);

    pre_replay_argmax.push(first_returned);
    final_tokens.push(scored_first);
    nll_per_step.push(first_nll);

    // Decode loop.  For step i (i >= 1), the input token to forward_decode
    // is the previous step's `final` token (so the model sees the same
    // token sequence in both passes).
    for step in 1..tokens {
        let prev_token = final_tokens[step - 1];
        let pos = prompt_tokens.len() + step - 1;

        // To capture pre-replay argmax we need the live argmax separately
        // from the replay-substituted return value.  forward_decode's
        // post-argmax tail substitutes the replay token AFTER the live
        // argmax has been computed AND after the logits buffer has been
        // populated.  When a replay is active we still get the live
        // argmax INSIDE forward_decode but it's not exposed.
        //
        // Cleanest: temporarily disable replay for this call by clearing
        // `set_replay_tokens` at step boundaries.  But that resets
        // decode_step and breaks the Gate H plumbing.
        //
        // Alternative: read the argmax from the logits buffer ourselves
        // AFTER forward_decode returns.  forward_decode already published
        // the live logits before the replay substitution; logits_view()
        // returns them.  We compute argmax in Rust and treat the
        // forward_decode return value as the (post-replay) "final" token.
        let mut profile = None;
        let returned = mlx_w.forward_decode(prev_token, pos, ctx, &mut profile)?;
        let live_argmax = argmax_from_logits(mlx_w.logits_view()?);
        let scored = match replay {
            Some(r) if step < r.len() => r[step],
            _ => returned,
        };
        let nll = mlx_w.token_nll_from_logits(scored)?;
        pre_replay_argmax.push(live_argmax);
        final_tokens.push(scored);
        nll_per_step.push(nll);
    }

    Ok(PassCapture {
        pre_replay_argmax,
        final_tokens,
        nll_per_step,
    })
}

fn argmax_from_logits(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Move per-(layer, pos) `hf2q_sdpa_out_layer*_pos*.bin` files from
/// `<src>` into `<dst>` for the seq_pos range that this pass produced.
///
/// We only move files within the expected range so any leftover dumps
/// from a previous run / previous pass don't contaminate.
fn move_sdpa_dumps(
    src: &Path,
    dst: &Path,
    prompt_len: usize,
    tokens: usize,
) -> Result<()> {
    if !src.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if !name.starts_with("hf2q_sdpa_out_layer") {
            continue;
        }
        // Only consider files whose pos is in [prompt_len, prompt_len+tokens).
        // Any other dumps are from out-of-window writes (e.g. the prefill
        // dumps `pos=prompt_len-1` for the warmup token).  Move them too
        // but they won't be read in synthesis, so it's harmless.
        let dest = dst.join(name);
        // Move (rename) if same FS, copy+delete otherwise.
        if let Err(e) = fs::rename(&path, &dest) {
            // Cross-FS or in-use; fall back to copy+delete.
            if let Err(copy_err) = fs::copy(&path, &dest) {
                anyhow::bail!(
                    "move sdpa dump {} -> {}: rename={e}; copy={copy_err}",
                    path.display(),
                    dest.display()
                );
            }
            let _ = fs::remove_file(&path);
        }
    }
    let _ = (prompt_len, tokens); // reserved for future range filtering
    Ok(())
}

/// Iterate (layer, pos) pairs, read both pass dumps, compute cosine.
///
/// Storage strategy: we keep all cosine values (≤ 30 layers × 1000
/// positions = 30k f32 ≈ 120 KB), then sort once for deterministic
/// quantiles.  Streaming p-quantile (TDigest) would be necessary at
/// >100k pairs; at our scale a Vec<f32> + sort is simpler and exact.
fn synthesize_cosine(
    dense_dir: &Path,
    tq_dir: &Path,
    num_layers: usize,
    prompt_len: usize,
    tokens: usize,
    mlx_w: &MlxModelWeights,
) -> Result<CosineStats> {
    let mut cosines: Vec<f32> = Vec::with_capacity(num_layers * tokens);

    for layer_idx in 0..num_layers {
        let nh = mlx_w.num_attention_heads;
        let hd = mlx_w.layers[layer_idx].head_dim;
        let n_elems = nh * hd;
        for step in 0..tokens {
            // The first decoded token's seq_pos is prompt_len (forward_prefill
            // writes that slot).  forward_decode for step >= 1 uses
            // seq_pos = prompt_len + step - 1.  So the seq_pos of the
            // SDPA dump for decode step `step` is prompt_len + step.
            //
            // BUT — the `HF2Q_DUMP_SDPA_MAX_POS` gate uses the decode-step
            // counter, not seq_pos, and dumps fire for decode-step <
            // max_pos.  Files are named by seq_pos.  So step 0
            // corresponds to the prefill's seq_pos; decode steps 1..N
            // correspond to seq_pos prompt_len..prompt_len+N-1.
            //
            // For a portable, defensive read: try the expected seq_pos
            // first (prompt_len + step - 1 for step >= 1, prompt_len for
            // step 0), and skip silently if absent (the file may not
            // have been written if the SDPA dump path is not on the
            // active SDPA branch for this layer).
            let seq_pos = if step == 0 {
                prompt_len
            } else {
                prompt_len + step - 1
            };
            let fname = format!(
                "hf2q_sdpa_out_layer{:02}_pos{}.bin",
                layer_idx, seq_pos
            );
            let dense_path = dense_dir.join(&fname);
            let tq_path = tq_dir.join(&fname);
            if !dense_path.exists() || !tq_path.exists() {
                continue;
            }
            let dense_vec = read_f32_bin(&dense_path, n_elems)?;
            let tq_vec = read_f32_bin(&tq_path, n_elems)?;
            let cs = cosine_pairwise_f32(&dense_vec, &tq_vec);
            if cs.is_finite() {
                cosines.push(cs);
            }
        }
    }

    if cosines.is_empty() {
        anyhow::bail!(
            "Gate H synthesis: no cosine pairs found.  Check that \
             HF2Q_DUMP_SDPA_MAX_POS plumbing actually fired (decode dump \
             path requires HF2Q_DUMP_ALL_CACHE=1 + dense-or-TQ SDPA branch \
             reached at sdpa_out write time)."
        );
    }

    // Sort ascending for quantiles.
    cosines.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = cosines.len();
    let mean = cosines.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
    let min = cosines[0];
    let p1 = cosines[((n as f64 * 0.01) as usize).min(n - 1)];
    let p50 = cosines[n / 2];
    let p99 = cosines[((n as f64 * 0.99) as usize).min(n - 1)];

    Ok(CosineStats {
        mean: mean as f32,
        min,
        p1,
        p50,
        p99,
        n_pairs: n,
    })
}

fn read_f32_bin(path: &Path, n_elems: usize) -> Result<Vec<f32>> {
    let mut f = fs::File::open(path)
        .with_context(|| format!("open dump: {}", path.display()))?;
    let mut bytes = Vec::with_capacity(n_elems * 4);
    f.read_to_end(&mut bytes)
        .with_context(|| format!("read dump: {}", path.display()))?;
    let bytes_f32_count = bytes.len() / 4;
    anyhow::ensure!(
        bytes_f32_count >= n_elems,
        "dump {} has {} f32 elems, expected >= {}",
        path.display(),
        bytes_f32_count,
        n_elems
    );
    let mut out = Vec::with_capacity(n_elems);
    for i in 0..n_elems {
        let off = i * 4;
        let arr = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

fn nll_to_ppl(nlls: &[f32]) -> f64 {
    if nlls.is_empty() {
        return f64::NAN;
    }
    let sum: f64 = nlls.iter().map(|x| *x as f64).sum();
    (sum / nlls.len() as f64).exp()
}

/// SHA-256 of a file as lowercase hex.
fn sha256_file(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)
        .with_context(|| format!("open: {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(hex::encode(digest))
}

fn git_head_sha() -> String {
    use std::process::Command;
    Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// ISO-8601 UTC ("YYYY-MM-DDTHH:MM:SSZ") from std::time::SystemTime.
///
/// Pure-Rust, no chrono/time dep.  The implementation is the standard
/// civil-date algorithm by Howard Hinnant (`days_from_civil` inverse,
/// public-domain).
fn iso8601_utc_now() -> String {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;

    // Days since 1970-01-01.
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400) as u32;
    let h = secs_of_day / 3600;
    let m = (secs_of_day / 60) % 60;
    let s = secs_of_day % 60;

    // Howard Hinnant civil_from_days (public domain).
    let z = days + 719_468; // shift epoch to 0000-03-01
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if mo <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, mo, d, h, m, s
    )
}

