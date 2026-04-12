//! Inference engine for GGUF models — load, generate, and serve.

pub mod config;
pub mod gemma4;
pub mod gguf_loader;
pub mod lm_head_kernel;
#[cfg(feature = "metal")]
pub mod moe_kernel;
pub mod rms_norm_kernel;
pub mod rope_kernel;
pub mod sampler;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use std::path::Path;

use crate::cli;
use config::Gemma4Config;
use gemma4::{DispatchSnapshot, Gemma4Model, PerTokenMetrics};
use gguf_loader::GgufModel;
use sampler::{SamplingParams, SAMPLING_EPS};
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// ADR-005 1bNEW.3 — greedy fast-path window size.
///
/// At T=0 / rep-penalty=1.0 decode, we chain `GREEDY_WINDOW` forward
/// passes into candle's Metal command pool using the still-lazy argmax
/// of each forward as the `input_ids` of the next, then `cat + to_vec1`
/// the whole window in a single forced GPU→CPU drain. With N=4, the
/// per-token sampler sync of 1/tok collapses to 1/4 tok (32 drains for
/// 128 decode tokens), matching the ADR line 411 suggestion and the MLX
/// `mx.async_eval` pattern.
///
/// N=1 is a valid degenerate mode (drain every token; same cardinality
/// as the baseline but through the batched code path) and is kept as a
/// bisect-safe knob. Larger N saturates candle's 5-buffer × 50-op
/// command pool and causes backpressure inside `select_entry` (see
/// `candle-metal-kernels/src/metal/commands.rs:118-147`), so N should
/// stay in the `[1, 4]` range.
const GREEDY_WINDOW: usize = 4;

/// Resolve the tokenizer path: explicit flag, or look next to GGUF / in parent dirs.
fn find_tokenizer(model_path: &Path, explicit: Option<&Path>) -> Result<std::path::PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    // Look next to GGUF
    let dir = model_path.parent().unwrap_or(Path::new("."));
    let candidate = dir.join("tokenizer.json");
    if candidate.exists() {
        return Ok(candidate);
    }
    // Look in models/{model_name}/ directory
    let _stem = model_path.file_stem().unwrap_or_default().to_string_lossy();
    // Try common patterns
    for subdir in &["gemma4", "gemma-4"] {
        let candidate = Path::new("models").join(subdir).join("tokenizer.json");
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    // Try to match model name prefix
    let models_dir = Path::new("models");
    if models_dir.is_dir() {
        for entry in std::fs::read_dir(models_dir)? {
            let entry = entry?;
            if entry.path().is_dir() {
                let tok = entry.path().join("tokenizer.json");
                if tok.exists() {
                    return Ok(tok);
                }
            }
        }
    }
    anyhow::bail!(
        "Cannot find tokenizer.json. Tried next to GGUF and in models/. \
         Use --tokenizer to specify the path explicitly."
    )
}

/// Resolve config.json path.
fn find_config(model_path: &Path, explicit: Option<&Path>) -> Result<std::path::PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    let dir = model_path.parent().unwrap_or(Path::new("."));
    let candidate = dir.join("config.json");
    if candidate.exists() {
        return Ok(candidate);
    }
    let models_dir = Path::new("models");
    if models_dir.is_dir() {
        for entry in std::fs::read_dir(models_dir)? {
            let entry = entry?;
            if entry.path().is_dir() {
                let cfg = entry.path().join("config.json");
                if cfg.exists() {
                    return Ok(cfg);
                }
            }
        }
    }
    anyhow::bail!(
        "Cannot find config.json. Use --config to specify the path explicitly."
    )
}

/// Resolve the prompt text from either `--prompt` or `--prompt-file`.
fn resolve_prompt(args: &cli::GenerateArgs) -> Result<String> {
    match (&args.prompt, &args.prompt_file) {
        (Some(text), _) => Ok(text.clone()),
        (None, Some(path)) => {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read prompt file: {}", path.display()))?;
            let trimmed = content.trim().to_string();
            anyhow::ensure!(!trimmed.is_empty(), "Prompt file is empty: {}", path.display());
            Ok(trimmed)
        }
        (None, None) => anyhow::bail!("Either --prompt or --prompt-file must be specified"),
    }
}

/// Run a single generation pass. Returns (generated_token_count, decode_elapsed).
/// If `silent` is true, suppresses token-by-token stdout output.
fn run_single_generation(
    model: &mut Gemma4Model,
    tokenizer: &tokenizers::Tokenizer,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    device: &Device,
    silent: bool,
) -> Result<(usize, std::time::Duration)> {
    use std::io::Write;

    let eos_token_ids: Vec<u32> = vec![1, 106]; // Gemma EOS tokens
    let mut all_tokens = prompt_tokens.to_vec();
    let counters = model.counters();

    // Prefill (not counted in benchmark tok/s, but timed + reported so
    // TTFT deltas from warmup coverage are visible; see ADR-005 1bNEW.12).
    if !silent {
        eprintln!("Prefilling {} tokens...", prompt_tokens.len());
    }
    let input = Tensor::new(prompt_tokens, device)?
        .unsqueeze(0)?;  // [1, seq_len]
    let prefill_start = std::time::Instant::now();
    let mut logits = model.forward(&input, 0)?;
    let prefill_elapsed = prefill_start.elapsed();
    if !silent {
        eprintln!(
            "Prefill complete in {:.1} ms ({} tokens, {:.1} tok/s).",
            prefill_elapsed.as_secs_f64() * 1000.0,
            prompt_tokens.len(),
            prompt_tokens.len() as f64 / prefill_elapsed.as_secs_f64(),
        );
    }

    // ADR-005 1bNEW.0 (metrics instrumentation):
    // Reset counters AFTER prefill so per-token averages describe only the
    // decode-loop region — matching what the `--benchmark` tok/s number
    // measures. Prefill has a different dispatch profile (seq_len > 1 path
    // through Attention, mask construction, etc.).
    counters.reset();

    // Debug: dump first decode-step logits to a file for cross-tool comparison.
    // Set HF2Q_DUMP_LOGITS=path.bin to write 262144 f32 LE bytes (vocab_size).
    if let Ok(dump_path) = std::env::var("HF2Q_DUMP_LOGITS") {
        let logits_f32 = logits.to_dtype(candle_core::DType::F32)?;
        let flat = logits_f32.flatten_all()?;
        let v: Vec<f32> = flat.to_vec1::<f32>()?;
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for f in &v { bytes.extend_from_slice(&f.to_le_bytes()); }
        std::fs::write(&dump_path, &bytes)?;
        eprintln!("HF2Q_DUMP_LOGITS: wrote {} f32 values ({} bytes) to {}",
            v.len(), bytes.len(), dump_path);
        // Also print top-10 by logit
        let mut idx_val: Vec<(usize, f32)> = v.iter().enumerate().map(|(i, &x)| (i, x)).collect();
        idx_val.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("HF2Q top-10 logits: {:?}", &idx_val[..10]);
    }

    // ADR-005 1bNEW.3 — greedy fast path (windowed async argmax drain).
    //
    // Gated on `temperature < SAMPLING_EPS && repetition_penalty == 1.0`.
    // Decode is deterministic under those conditions, so we chain
    // `GREEDY_WINDOW` forward passes together using the lazy argmax of
    // each forward as the next forward's `input_ids`, and `cat + to_vec1`
    // them all in a single forced GPU→CPU drain per window — collapsing
    // the baseline 1-sync-per-token pattern to 1-sync-per-N-tokens. See
    // `run_decode_greedy_batched` below for the full loop.
    //
    // Any other sampling configuration routes through the per-token sync
    // path (needed by the RNG / rep-penalty gather); that path is
    // byte-identical to 1bNEW.1.
    let fast_path = params.temperature < SAMPLING_EPS
        && params.repetition_penalty == 1.0;

    let (generated, elapsed) = if fast_path {
        run_decode_greedy_batched(
            model,
            tokenizer,
            logits,
            &mut all_tokens,
            &eos_token_ids,
            params.max_tokens,
            silent,
        )?
    } else {
        // === Per-token sync path (non-greedy or rep_penalty != 1.0) ===
        let mut next_token = sampler::sample_token(&logits, params, &[], &counters)?;
        all_tokens.push(next_token);

        if !silent {
            let token_str = tokenizer.decode(&[next_token], false)
                .unwrap_or_default();
            print!("{}", token_str);
            std::io::stdout().flush()?;
        }

        // Decode loop — timed separately from prefill
        let start = std::time::Instant::now();
        let mut generated = 1usize;
        for _ in 1..params.max_tokens {
            if eos_token_ids.contains(&next_token) {
                break;
            }

            let input = Tensor::new(&[next_token], device)?
                .unsqueeze(0)?;  // [1, 1]
            let seqlen_offset = all_tokens.len() - 1;
            logits = model.forward(&input, seqlen_offset)?;
            next_token = sampler::sample_token(&logits, params, &all_tokens, &counters)?;
            all_tokens.push(next_token);
            generated += 1;

            if !silent {
                let token_str = tokenizer.decode(&[next_token], false)
                    .unwrap_or_default();
                print!("{}", token_str);
                std::io::stdout().flush()?;
            }
        }

        (generated, start.elapsed())
    };

    // Debug: dump the full generated token sequence (prompt + decoded) as
    // u32 LE for offline diffing against llama.cpp. Set
    // `HF2Q_DUMP_TOKEN_SEQUENCE=path.bin` to enable. Prompt length is
    // reported via stderr so the caller can slice out the decoded tail.
    if let Ok(dump_path) = std::env::var("HF2Q_DUMP_TOKEN_SEQUENCE") {
        let mut bytes = Vec::with_capacity(all_tokens.len() * 4);
        for &t in &all_tokens {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        std::fs::write(&dump_path, &bytes)?;
        eprintln!(
            "HF2Q_DUMP_TOKEN_SEQUENCE: wrote {} u32 values ({} bytes) to {} (prompt_len={}, decode_len={})",
            all_tokens.len(),
            bytes.len(),
            dump_path,
            prompt_tokens.len(),
            all_tokens.len() - prompt_tokens.len(),
        );
    }

    Ok((generated, elapsed))
}

/// ADR-005 1bNEW.3 — windowed async-drain greedy decode.
///
/// Runs `max_tokens` total generated tokens by chaining `GREEDY_WINDOW`
/// forward passes between each GPU→CPU drain. Each forward's lazy argmax
/// (a `[1, 1]` u32 tensor) feeds directly into the next forward's
/// `input_ids`, all while still inside candle's Metal command pool. When
/// the window fills up — or EOS / max_tokens forces an earlier stop —
/// the window's lazy argmax tensors are concatenated and drained with a
/// single `to_vec1::<u32>()` call, which is the ONE forced sync per
/// window.
///
/// **Correctness invariants:**
/// - Token sequence is bitwise identical to the per-token sync path. The
///   forward ops, argmax op, KV-cache mutations, and lm_head softcapping
///   are the same ops in the same order — this routine only changes
///   *when* the CPU observes the result. Validated by a byte-identical
///   stdout diff against the 1bNEW.1 baseline (commit 13b5536) on the
///   187-token canonical bench prompt, 128 decode tokens.
/// - EOS detection is delayed by at most `GREEDY_WINDOW - 1` forward
///   passes: if token `k` mid-window is EOS, the `k+1..window_end`
///   forwards still execute (and their KV-cache writes still land in
///   the cache) before the drain notices. Those tokens are dropped from
///   output via `all_tokens.truncate(...)` inside `drain_window`. This
///   is faithful to the ADR "sync every N tokens" wording and
///   unavoidable given that we cannot inspect a still-lazy tensor.
/// - The final partial window is drained before returning, so candle's
///   command pool has no unread work on exit.
///
/// **Counter accounting:**
/// - `sampler_sync_count` increments once per `to_vec1` drain, not once
///   per token. For `max_tokens = 128` with `GREEDY_WINDOW = 4` the
///   layout is:
///     1 pre-timing drain (the first post-prefill token)
///     + ceil((max_tokens - 1) / GREEDY_WINDOW) loop drains
///     = 1 + 32 = 33 sampler syncs for a 128-token run,
///   down from 128 at the 1bNEW.1 baseline (one per token). Pre-timing
///   drains before the decode timer starts, matching the per-token
///   path's timer discipline; its wall-clock cost is excluded from
///   tok/s exactly as the per-token path excludes its first sample.
/// - `dispatches_per_token` is incremented by `greedy_argmax_lazy`
///   (1 per sampled token for the argmax) and by `drain_window`
///   (2 per drain for the cat + flatten_all).
///
/// **Why this doesn't match the ADR's 3-6 tok/s gain estimate.** Q3's
/// 7.51 ms sampler-sync figure was measured on the pre-1bNEW.1 baseline
/// (24.30 tok/s, 60 MoE to_vec2 syncs per token absorbing most of the
/// GPU wait). Post-1bNEW.1 the sampler sync consolidates the full
/// forward-pass tail (lm_head + softcap + argmax + all layer-N remnants)
/// and measures ~19 ms/call (bracketed with Instant::now around the old
/// `argmax().to_scalar()` on the 1bNEW.1 baseline, 2026-04-10). That
/// 19 ms is unavoidable GPU compute time, not sync overhead — candle's
/// `to_cpu` path goes through `flush_and_wait` on the entire command
/// pool (`candle-metal-kernels/src/metal/commands.rs:176`), which drains
/// every in-flight buffer regardless of which tensor is being read back.
/// Queuing forward N+1 before syncing on forward N does NOT overlap the
/// sync with N+1's head — it just makes the sync wait for N+N+1 instead
/// of just N. The one-ahead pipeline idiom that works in MLX (which has
/// a multi-stream async graph executor) actively regresses in candle.
/// The item lands as a structural refactor: `sampler_sync_count` drops
/// from 128 to 33, tok/s is at parity (+0.2 ± noise), and the primitive
/// is in place for a future candle-side patch that exposes per-buffer
/// wait semantics.
#[allow(clippy::too_many_arguments)]
fn run_decode_greedy_batched(
    model: &mut Gemma4Model,
    tokenizer: &tokenizers::Tokenizer,
    prefill_logits: Tensor,
    all_tokens: &mut Vec<u32>,
    eos_token_ids: &[u32],
    max_tokens: usize,
    silent: bool,
) -> Result<(usize, std::time::Duration)> {
    use std::io::Write;

    if max_tokens == 0 {
        return Ok((0, std::time::Duration::ZERO));
    }

    let counters = model.counters();
    let seqlen_base = all_tokens.len();
    let device = prefill_logits.device().clone();

    // === Pre-timing: resolve the first post-prefill token ===
    //
    // Matches the per-token path: the first sample runs BEFORE the
    // decode timer starts. This is where Metal shader-compilation
    // warmup sits (~200 ms on a cold process) — excluding it from the
    // timer gives numbers directly comparable to the 1bNEW.1 baseline's
    // tok/s figure.
    //
    // We use a concrete `Tensor::new(&[first_u32], ...)` for the very
    // first forward's `input_ids` (the same shape the per-token path
    // constructs). The subsequent forwards chain the lazy argmax
    // directly.
    let first_u32: u32 = sampler::greedy_argmax_lazy(&prefill_logits, &counters)?
        .flatten_all()?
        .to_vec1::<u32>()?[0];
    counters.sampler_sync_count.fetch_add(1, Ordering::Relaxed);
    counters.dispatches_per_token.fetch_add(1, Ordering::Relaxed); // flatten_all
    all_tokens.push(first_u32);
    let mut generated: usize = 1;

    if !silent {
        let token_str = tokenizer.decode(&[first_u32], false).unwrap_or_default();
        print!("{}", token_str);
        let _ = std::io::stdout().flush();
    }

    // Early-out when only one token was requested, or the first token
    // itself is EOS.
    if max_tokens == 1 || eos_token_ids.contains(&first_u32) {
        return Ok((generated, std::time::Duration::ZERO));
    }

    // === Decode timer starts here (matches per-token path) ===
    let start = std::time::Instant::now();

    // Window of pending lazy argmax tensors waiting for a drain. Each
    // element is a `[1, 1]` u32 tensor produced by `greedy_argmax_lazy`
    // that will also be used as `input_ids` for the next forward pass
    // via candle `Embedding::forward` → `index_select` (no host copy).
    let mut window: Vec<Tensor> = Vec::with_capacity(GREEDY_WINDOW);

    // Seed the loop with a concrete `Tensor::new(&[first_u32])` for the
    // first forward. From forward(2) onward, `pending_input` is the
    // lazy argmax of forward(k-1)'s logits — the chain entry point.
    let mut pending_input = Tensor::new(&[first_u32], &device)?.unsqueeze(0)?; // [1, 1]

    // Main decode loop — produces tokens 2..max_tokens. Each iteration:
    //   1. If the window is full, drain it (1 forced sync per
    //      `GREEDY_WINDOW` tokens).
    //   2. Enqueue forward(K) using `pending_input`, which is the
    //      fresh `Tensor::new(&[first_u32])` on the first iteration
    //      and the still-lazy argmax of forward(K-1) on subsequent
    //      iterations — candle's `Embedding::forward → index_select`
    //      accepts either because both are contiguous u32 `[1, 1]`.
    //   3. Sample the new token lazily via `greedy_argmax_lazy`; push
    //      the lazy result into the window and wire it as the next
    //      iteration's `pending_input`.
    //
    // Forward pass K writes KV cache position `seqlen_base + K - 1`
    // (K counted from 1). With `generated == K` on entry, the offset
    // formula is `seqlen_base + generated - 1`, equivalent to
    // `all_tokens.len() - 1 - window.len()`.
    let mut eos_hit = false;
    while generated < max_tokens {
        if window.len() >= GREEDY_WINDOW {
            match drain_window(
                &mut window,
                all_tokens,
                eos_token_ids,
                seqlen_base,
                generated,
                tokenizer,
                silent,
                &counters,
            )? {
                DrainOutcome::Continue => {}
                DrainOutcome::EosAt => {
                    eos_hit = true;
                    break;
                }
            }
        }

        let seqlen_offset = seqlen_base + generated - 1;
        let logits = model.forward(&pending_input, seqlen_offset)?;
        pending_input = sampler::greedy_argmax_lazy(&logits, &counters)?;
        window.push(pending_input.clone());
        all_tokens.push(0u32);
        generated += 1;
    }

    // Final drain of any pending window content. Skipped if we already
    // broke out on EOS mid-loop (in which case `drain_window` has
    // already truncated `all_tokens` to the EOS boundary).
    if !eos_hit && !window.is_empty() {
        let _ = drain_window(
            &mut window,
            all_tokens,
            eos_token_ids,
            seqlen_base,
            generated,
            tokenizer,
            silent,
            &counters,
        )?;
    }

    if !silent {
        let _ = std::io::stdout().flush();
    }

    let elapsed = start.elapsed();

    // `generated_emitted` = actually-emitted tokens, which equals
    // `all_tokens.len() - seqlen_base` after `drain_window` has truncated
    // any mid-window EOS tail. Returning this value keeps the tok/s
    // numerator honest — we don't credit speculative forwards that EOS
    // discarded.
    let generated_emitted = all_tokens.len() - seqlen_base;
    Ok((generated_emitted, elapsed))
}

/// Result of a single window drain.
enum DrainOutcome {
    /// The drained window contained no EOS token; keep decoding.
    Continue,
    /// EOS hit inside the drained window — `all_tokens` has been
    /// truncated to include the EOS token and discard any speculative
    /// forwards queued past it. The caller should stop emitting.
    EosAt,
}

/// Drain the pending window of lazy argmax tensors with ONE forced
/// GPU→CPU sync, then walk the materialized u32s: overwrite the
/// placeholders in `all_tokens`, print each to stdout if non-silent, and
/// stop early on EOS (truncating `all_tokens` so the caller's
/// `generated_emitted` count matches the actually-printed tokens).
#[allow(clippy::too_many_arguments)]
fn drain_window(
    window: &mut Vec<Tensor>,
    all_tokens: &mut Vec<u32>,
    eos_token_ids: &[u32],
    seqlen_base: usize,
    generated_so_far: usize,
    tokenizer: &tokenizers::Tokenizer,
    silent: bool,
    counters: &Arc<gemma4::DispatchCounters>,
) -> Result<DrainOutcome> {
    use std::io::Write;

    if window.is_empty() {
        return Ok(DrainOutcome::Continue);
    }
    let window_len = window.len();

    // ONE forced sync for the whole window. Each element is a `[1, 1]`
    // u32 tensor; `cat` on dim 0 yields `[window_len, 1]`, `flatten_all`
    // gives `[window_len]`, and `to_vec1::<u32>` is the single drain.
    let stacked = Tensor::cat(window.as_slice(), 0)?.flatten_all()?;
    let tokens: Vec<u32> = stacked.to_vec1::<u32>()?;
    counters.sampler_sync_count.fetch_add(1, Ordering::Relaxed);
    counters.dispatches_per_token.fetch_add(2, Ordering::Relaxed); // cat + flatten_all
    window.clear();
    debug_assert_eq!(tokens.len(), window_len);

    // The placeholders for these tokens are at the tail of `all_tokens`,
    // starting at `placeholder_start`.
    let placeholder_start = seqlen_base + generated_so_far - window_len;

    let mut eos_idx: Option<usize> = None;
    for (k, &tok) in tokens.iter().enumerate() {
        all_tokens[placeholder_start + k] = tok;
        if !silent {
            let token_str = tokenizer.decode(&[tok], false).unwrap_or_default();
            print!("{}", token_str);
            let _ = std::io::stdout().flush();
        }
        if eos_token_ids.contains(&tok) {
            eos_idx = Some(k);
            break;
        }
    }

    if let Some(k) = eos_idx {
        // Truncate `all_tokens` so its length reflects only tokens that
        // were actually emitted (EOS inclusive, matching the per-token
        // path which pushes the EOS token and then breaks).
        all_tokens.truncate(placeholder_start + k + 1);
        return Ok(DrainOutcome::EosAt);
    }

    Ok(DrainOutcome::Continue)
}

/// Detect hardware info for benchmark reporting.
fn detect_hardware_info() -> (String, u64) {
    use crate::intelligence::hardware::HardwareProfiler;

    match HardwareProfiler::detect() {
        Ok(profile) => {
            let mem_gb = profile.total_memory_bytes / (1024 * 1024 * 1024);
            (profile.chip_model, mem_gb)
        }
        Err(_) => ("Unknown".to_string(), 0),
    }
}

/// Compute the median of a sorted slice.
fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Compute p95 from a sorted slice (using nearest-rank method).
fn p95(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    // Nearest-rank: ceil(0.95 * n) - 1, clamped
    let rank = ((0.95 * n as f64).ceil() as usize).saturating_sub(1).min(n - 1);
    sorted[rank]
}

/// Hardcoded fallback chat template used ONLY when no GGUF-embedded template
/// exists and the user has not passed `--chat-template` / `--chat-template-file`.
///
/// This exists as a last-resort compatibility path for older/incomplete GGUFs
/// that predate the Phase 1 chat-template fix. For the primary Gemma 4 path,
/// the template comes from GGUF metadata (`tokenizer.chat_template`), matching
/// llama.cpp behavior.
const FALLBACK_GEMMA4_CHAT_TEMPLATE: &str =
    "<bos><|turn>system\n<|think|><turn|>\n<|turn>user\n{{PROMPT}}<turn|>\n<|turn>model\n";

/// Resolve the chat template per ADR-005 Phase 1 priority order:
///
///   1. CLI `--chat-template STRING`
///   2. CLI `--chat-template-file FILE`
///   3. GGUF `tokenizer.chat_template` metadata
///   4. Hardcoded fallback string (last resort)
///
/// Renders the resolved template with minijinja using the HuggingFace chat
/// format (single-turn user message). On any render error from an embedded or
/// CLI-supplied template, returns the error; the hardcoded fallback path does
/// NOT go through jinja (it uses simple placeholder substitution) so it cannot
/// itself fail.
fn render_chat_template(
    gguf: &GgufModel,
    args: &cli::GenerateArgs,
    user_prompt: &str,
) -> Result<String> {
    // Priority 1: CLI --chat-template string
    if let Some(tmpl) = args.chat_template.as_deref() {
        tracing::info!("Chat template: using CLI --chat-template override");
        return render_jinja_template(tmpl, user_prompt);
    }

    // Priority 2: CLI --chat-template-file
    if let Some(path) = args.chat_template_file.as_deref() {
        tracing::info!("Chat template: loading from --chat-template-file {}", path.display());
        let tmpl = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read --chat-template-file {}", path.display()))?;
        return render_jinja_template(&tmpl, user_prompt);
    }

    // Priority 3: GGUF metadata `tokenizer.chat_template`
    if let Some(tmpl) = gguf.get_metadata_string("tokenizer.chat_template") {
        tracing::info!(
            "Chat template: using GGUF metadata tokenizer.chat_template ({} chars)",
            tmpl.len()
        );
        return render_jinja_template(&tmpl, user_prompt);
    }

    // Priority 4: hardcoded fallback — last resort, simple substitution
    tracing::warn!(
        "Chat template: no GGUF metadata tokenizer.chat_template and no CLI override; \
         falling back to hardcoded Gemma4 template"
    );
    Ok(FALLBACK_GEMMA4_CHAT_TEMPLATE.replace("{{PROMPT}}", user_prompt))
}

/// Render a Jinja2 chat template using minijinja.
///
/// Passes HuggingFace-standard variables: `messages`, `add_generation_prompt`,
/// `bos_token`, `eos_token`. Gemma 4's template should only reference these.
fn render_jinja_template(template_str: &str, user_prompt: &str) -> Result<String> {
    let mut env = minijinja::Environment::new();
    env.add_template("chat", template_str)
        .context("Failed to parse chat template as Jinja2")?;
    let tmpl = env
        .get_template("chat")
        .context("Failed to load parsed chat template")?;
    let rendered = tmpl
        .render(minijinja::context! {
            messages => vec![
                minijinja::context! { role => "user", content => user_prompt }
            ],
            add_generation_prompt => true,
            bos_token => "<bos>",
            eos_token => "<eos>",
        })
        .context("Failed to render chat template")?;
    Ok(rendered)
}

/// Run the `generate` subcommand.
pub fn cmd_generate(args: cli::GenerateArgs) -> Result<()> {
    let model_path = &args.model;
    anyhow::ensure!(model_path.exists(), "Model not found: {}", model_path.display());

    let tokenizer_path = find_tokenizer(model_path, args.tokenizer.as_deref())?;
    let config_path = find_config(model_path, args.config.as_deref())?;

    tracing::info!("Model:     {}", model_path.display());
    tracing::info!("Tokenizer: {}", tokenizer_path.display());
    tracing::info!("Config:    {}", config_path.display());

    // Parse model config
    let cfg = Gemma4Config::from_config_json(&config_path)
        .context("Failed to parse config.json")?;
    tracing::info!(
        "Gemma4 A4B: {} layers, {} heads, hidden={}, {} experts (top-{})",
        cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size,
        cfg.num_experts, cfg.top_k_experts,
    );

    // Select device
    let device = select_device()?;
    tracing::info!("Device: {:?}", device);

    // Load GGUF
    eprintln!("Loading GGUF model...");
    let gguf = GgufModel::load(model_path, &device)?;

    // Load model weights from GGUF. ADR-005 1bNEW.1 Phase B: pass the
    // `--moe-kernel` mode down so `Gemma4Model::load_with_modes` can
    // wire the fused path on every layer (Phase C widened from layer 0
    // only). ADR-005 1bNEW.4 Phase B: `--rms-norm-kernel` plumbs the
    // same way; `fused` compiles the downstream MSL library once at
    // load time and clones the `Arc<RmsNormPipelines>` into every
    // RmsNorm call site. Default is `loop` (Phase B bisect-safety);
    // Phase C flips the default to `fused`.
    eprintln!("Loading model weights from GGUF (quantized QMatMul)...");
    let moe_mode: gemma4::MoeKernelMode = args.moe_kernel.into();
    let rms_mode: rms_norm_kernel::RmsNormKernelMode = args.rms_norm_kernel.into();
    // ADR-005 1bNEW.6 Phase B: `--rope-kernel` plumbs the same way
    // as `--moe-kernel` and `--rms-norm-kernel`. Default is `loop`
    // in Phase B (bisect-safety); Phase C flips the default to
    // `fused` after the 5-run benchmark gate validates it.
    let rope_mode: rope_kernel::RopeKernelMode = args.rope_kernel.into();
    // ADR-005 1bNEW.17 Phase B: `--lm-head-kernel` plumbs the same way.
    // Default is `loop` in Phase B (bisect-safety); Phase C flips the
    // default to `fused` after the coherence + recall gates pass.
    let lm_head_mode: lm_head_kernel::LmHeadKernelMode = args.lm_head_kernel.into();
    // ADR-005 1bNEW.20 Phase A/B: `--kv-cache-kernel` plumbs the same
    // way. Default is `slice_scatter` in Phase A/B (bisect-safety);
    // Phase C flips the default to `in_place` after the 5-run canonical
    // bench gate, 128-token coherence gate, and adversarial-recall gate
    // all pass.
    let kv_cache_mode: gemma4::KvCacheKernelMode = args.kv_cache_kernel.into();
    tracing::info!("MoE dispatch mode: {:?}", moe_mode);
    let mut model = Gemma4Model::load_with_modes(
        &cfg, &gguf, &device, moe_mode, rms_mode, rope_mode, lm_head_mode, kv_cache_mode,
    )?;

    // Warmup: run two dummy forwards to force Metal shader compilation for
    // both the decode and prefill code paths at model-load time.
    //
    // ADR-005 1bNEW.14 (Phase 1 baseline): the single-token warmup below
    // triggers PSO compilation for the decode-time vector SDPA kernel
    // (`q_len == 1` branch in `Attention::forward`), eliminating a ~37 ms
    // cold-compile spike on the first decode token.
    //
    // ADR-005 1bNEW.12 (this change, 2026-04-10): the decode warmup alone
    // does NOT cover the prefill code path, which has its own kernel
    // variants:
    //   - Global layers (head_dim=512) now route through candle's fused
    //     BF16 SDPA full kernel at `bd=512, bq=8, bk=8` (1bNEW.10,
    //     commit 29b84ef). First dispatch cold-compiles
    //     `steel_attention_bfloat16_bq8_bk8_bd512_wm1_wn1_maskbfloat16_t`.
    //   - Prefill MoE `kernel_mul_mv_id_*` batched matmul (1bNEW.1):
    //     first multi-token dispatch pre-compiles the per-token offset
    //     code path that decode `n_tokens=1` never exercises.
    //   - Prefill softmax over a multi-row `attn_weights` matrix on the
    //     bd=256 manual path — first call cold-compiles the F32
    //     `softmax_last_dim` kernel shape that decode never sees.
    //
    // Running a short multi-token forward pass at load time pre-compiles
    // all prefill PSOs so the first real request pays only the model's
    // actual compute cost, not the shader-compile wait. Matches the
    // spirit of llama.cpp's pre-built `.metallib` at build time; hf2q
    // cannot ship a pre-compiled metallib from Rust, so runtime warmup
    // is the nearest equivalent.
    //
    // Warmup length of 8 tokens is chosen as a comfortable multiple of
    // `bq=8` (the bd=512 tile row count) that exercises the full bd=512
    // tiled path including an aligned last row batch. Sliding-layer
    // (bd=256) prefill uses the manual F32 path after 1bNEW.10's
    // head_dim split, so any q_len ≥ 2 covers its PSOs identically.
    //
    // Cleanup via `clear_kv_cache()` after each warmup pass so prefill
    // for the first real request starts from a clean cache state.
    eprintln!("Warming up model (decode path)...");
    let warmup_decode = Tensor::new(&[2u32], &device)?.unsqueeze(0)?; // BOS token
    let _ = model.forward(&warmup_decode, 0)?;
    model.clear_kv_cache();

    eprintln!("Warming up model (prefill path)...");
    // 10-token dummy input — deliberately NOT a multiple of `bq=8`.
    //
    // Candle's SDPA full kernel selects its compute pipeline via
    // `load_pipeline_with_constants` at
    // `candle-metal-kernels/src/kernels/sdpa.rs:126-133` with four boolean
    // function_constants: `align_Q` (id 200), `align_K` (id 201),
    // `has_mask` (id 300), `do_causal` (id 301). Distinct constant
    // combinations compile to distinct PSOs — so a bq=8-aligned warmup
    // (e.g. `q_seq=8`) would compile `align_Q=true, align_K=true`, which
    // is NOT the PSO a real 14-token chat-templated prompt needs
    // (`align_Q=false, align_K=false`). `q_seq=10` exercises the
    // non-aligned variant: `10 % 8 = 2 → align_Q=false`,
    // `10 % 8 = 2 → align_K=false`. Most real chat-template-rendered
    // prompt lengths land in this common case, so this is the PSO
    // worth pre-compiling. `has_mask=false, do_causal=true` are fixed by
    // the call site at `gemma4.rs::Attention::forward` prefill branch
    // and thus shared with every real request.
    let warmup_prefill_ids: [u32; 10] = [2, 105, 2364, 107, 10979, 106, 107, 2, 2, 2];
    let warmup_prefill = Tensor::new(&warmup_prefill_ids, &device)?.unsqueeze(0)?;
    let warmup_prefill_logits = model.forward(&warmup_prefill, 0)?;
    // Force a GPU sync so the warmup pass fully completes before the first
    // real request begins. Without this, candle's lazy command pool keeps
    // the warmup dispatches pending and the first real prefill forward
    // `flush_and_wait`s behind them, double-dipping on latency: empirically
    // measured at ~−5.5 ms → +5 ms delta (regression) without the sync on
    // Apple M5 Max, 2026-04-10. Reading a single scalar from the warmup
    // logits is the lightest available sync primitive — anything that
    // reaches `Tensor::to_vec*` or `to_scalar` triggers
    // `candle_metal_kernels::commands::flush_and_wait`
    // (`candle-metal-kernels/src/metal/commands.rs:176-202`).
    let _ = warmup_prefill_logits
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    model.clear_kv_cache();
    eprintln!("Warmup complete.");

    // Load tokenizer. Disable the truncation policy baked into some tokenizer.json
    // files (Gemma 4's ships with `truncation.max_length: 256`), which would otherwise
    // silently truncate every prompt past 256 tokens and make bench/llama.cpp
    // comparisons meaningless. We always want the full prompt; if a caller needs
    // truncation, they can enforce it at a higher layer.
    let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow::anyhow!("Failed to disable tokenizer truncation: {}", e))?;
    let tokenizer = tokenizer;

    // Resolve prompt from --prompt or --prompt-file
    let prompt_text_raw = resolve_prompt(&args)?;

    // Resolve the chat template per ADR-005 Phase 1 priority:
    //   CLI --chat-template > CLI --chat-template-file > GGUF metadata > fallback
    // The final fallback matches llama.cpp's behavior only when no template was
    // embedded in the GGUF and the user has not overridden it.
    let prompt_text = render_chat_template(&gguf, &args, &prompt_text_raw)?;

    // ADR-005 1bNEW.0c: write the fully-rendered prompt to a file and exit,
    // so scripts/crawl_verify.sh can feed the byte-identical rendered text to
    // llama-completion without `--jinja` (which routes through a different
    // prompt path than hf2q — see ADR line 198). Presence-gated, no runtime
    // cost when unset. Uses the same pattern as HF2Q_DUMP_LOGITS /
    // HF2Q_DUMP_PROMPT_TOKENS.
    if let Ok(dump_path) = std::env::var("HF2Q_DUMP_RENDERED_PROMPT") {
        std::fs::write(&dump_path, prompt_text.as_bytes())
            .with_context(|| format!("HF2Q_DUMP_RENDERED_PROMPT: failed to write {dump_path}"))?;
        eprintln!(
            "HF2Q_DUMP_RENDERED_PROMPT: wrote {} bytes to {}",
            prompt_text.len(),
            dump_path
        );
        return Ok(());
    }

    let encoding = tokenizer.encode(prompt_text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    tracing::info!("Prompt: {} tokens", prompt_tokens.len());
    if std::env::var("HF2Q_DUMP_PROMPT_TOKENS").is_ok() {
        eprintln!("HF2Q_DUMP_PROMPT_TOKENS: first10={:?} last10={:?} total={}",
            &prompt_tokens[..prompt_tokens.len().min(10)],
            &prompt_tokens[prompt_tokens.len().saturating_sub(10)..],
            prompt_tokens.len());
    }

    let params = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        max_tokens: args.max_tokens,
    };

    if args.benchmark {
        // === Benchmark mode: 5 consecutive runs ===
        const NUM_RUNS: usize = 5;
        let (chip, mem_gb) = detect_hardware_info();
        let model_filename = model_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        eprintln!("Benchmark mode: {} runs, {} max tokens, temperature={}",
            NUM_RUNS, args.max_tokens, args.temperature);

        let mut tok_per_sec_runs: Vec<f64> = Vec::with_capacity(NUM_RUNS);
        // ADR-005 1bNEW.0: snapshot counters from the final run (each run's
        // prefill-then-decode loop resets counters at the start of decode in
        // `run_single_generation`, so snapshot-after-run reflects that run's
        // decode region).
        let mut last_snapshot: DispatchSnapshot = DispatchSnapshot::default();

        for run in 1..=NUM_RUNS {
            model.clear_kv_cache();

            let (generated, elapsed) = run_single_generation(
                &mut model, &tokenizer, &prompt_tokens, &params, &device, true,
            )?;
            last_snapshot = model.counters().snapshot();

            let tps = generated as f64 / elapsed.as_secs_f64();
            tok_per_sec_runs.push(tps);
            eprintln!("  Run {}/{}: {} tokens in {:.2}s ({:.1} tok/s)",
                run, NUM_RUNS, generated, elapsed.as_secs_f64(), tps);
        }

        // Sort for median / p95 computation
        let mut sorted = tok_per_sec_runs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let med = median(&sorted);
        let p95_val = p95(&sorted);

        println!();
        println!("=== Benchmark Results ===");
        println!("Hardware: {}, {} GB", chip, mem_gb);
        println!("Model: {}", model_filename);
        println!("Prompt tokens: {}", prompt_tokens.len());
        println!("Generated tokens: {} (per run)", args.max_tokens);
        println!("Runs: {}", NUM_RUNS);
        println!();
        for (i, tps) in tok_per_sec_runs.iter().enumerate() {
            println!("Run {}: {:.1} tok/s", i + 1, tps);
        }
        println!();
        println!("Median: {:.1} tok/s", med);
        println!("P95:    {:.1} tok/s", p95_val);

        // ADR-005 1bNEW.0 — dispatch counter report.
        //
        // Written to `metrics.txt` in the current working directory. The
        // ADR phrasing is "alongside `bench.log`", but no `bench.log` exists
        // in the current code path — the benchmark harness prints to stdout
        // only. Emitting next to `bench.log` therefore means "in the CWD
        // where `bench.log` would land if/when it is added," which is the
        // same directory the user invoked hf2q from.
        //
        // The per-token averages divide by `forward_count` (the number of
        // decode forward calls that happened after the counters were reset).
        // At T=0 greedy with max-tokens=128 and a stop-at-EOS that typically
        // does not trigger in-prompt, forward_count ≈ 127 (first token from
        // prefill, then 127 decode steps). Counters are per-decode-step.
        let per_token: PerTokenMetrics = last_snapshot.per_token();
        let metrics_path = std::path::PathBuf::from("metrics.txt");
        let metrics_body = format!(
            "# ADR-005 Phase 1b 1bNEW.0 — dispatch counter report\n\
             # Emitted by hf2q `generate --benchmark`\n\
             # Counters cover the decode loop only (reset after prefill).\n\
             # Last benchmark run of {} is reported; at T=0 greedy each run\n\
             # is deterministic, so per-token averages are stable.\n\
             model: {}\n\
             hardware: {}, {} GB\n\
             prompt_tokens: {}\n\
             max_tokens: {}\n\
             runs: {}\n\
             median_tok_per_sec: {:.2}\n\
             p95_tok_per_sec: {:.2}\n\
             \n\
             # Raw totals over the decode loop ({} forward calls)\n\
             forward_count: {}\n\
             total_dispatches: {}\n\
             total_moe_to_vec2: {}\n\
             total_moe_dispatches: {}\n\
             moe_layer_invocations: {}\n\
             total_sampler_sync: {}\n\
             total_norm_dispatches: {}\n\
             \n\
             # Per-token averages (= total / forward_count)\n\
             dispatches_per_token: {:.2}\n\
             moe_to_vec2_count: {:.2}\n\
             moe_dispatches_per_layer: {:.2}\n\
             sampler_sync_count: {:.2}\n\
             norm_dispatches_per_token: {:.2}\n",
            NUM_RUNS,
            model_filename,
            chip, mem_gb,
            prompt_tokens.len(),
            args.max_tokens,
            NUM_RUNS,
            med, p95_val,
            last_snapshot.forward_count,
            last_snapshot.forward_count,
            last_snapshot.dispatches_per_token,
            last_snapshot.moe_to_vec2_count,
            last_snapshot.moe_dispatches,
            last_snapshot.moe_layer_invocations,
            last_snapshot.sampler_sync_count,
            last_snapshot.norm_dispatches_per_token,
            per_token.dispatches_per_token,
            per_token.moe_to_vec2_count,
            per_token.moe_dispatches_per_layer,
            per_token.sampler_sync_count,
            per_token.norm_dispatches_per_token,
        );
        std::fs::write(&metrics_path, metrics_body)
            .with_context(|| format!("Failed to write {}", metrics_path.display()))?;
        eprintln!("Wrote dispatch counter report → {}", metrics_path.display());
    } else {
        // === Normal single-run generation with streaming output ===
        let (generated, elapsed) = run_single_generation(
            &mut model, &tokenizer, &prompt_tokens, &params, &device, false,
        )?;

        let tok_per_sec = generated as f64 / elapsed.as_secs_f64();
        eprintln!("\n\n--- {} tokens in {:.2}s ({:.1} tok/s) ---",
            generated, elapsed.as_secs_f64(), tok_per_sec);
    }

    Ok(())
}

/// Select the best available compute device.
fn select_device() -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        tracing::info!("Using Metal GPU");
        return Ok(Device::new_metal(0)?);
    }
    #[cfg(feature = "cuda")]
    {
        tracing::info!("Using CUDA GPU");
        return Ok(Device::new_cuda(0)?);
    }
    #[allow(unreachable_code)]
    {
        tracing::info!("Using CPU (no GPU features enabled)");
        Ok(Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::render_jinja_template;

    /// Minimal Gemma-like template: verifies minijinja rendering of a single
    /// user message with `add_generation_prompt`.
    #[test]
    fn jinja_template_renders_single_user_turn() {
        let tmpl = "{{ bos_token }}{% for m in messages %}<|turn|>{{ m.role }}\n{{ m.content }}<|end|>\n{% endfor %}{% if add_generation_prompt %}<|turn|>model\n{% endif %}";
        let out = render_jinja_template(tmpl, "hello").expect("render ok");
        assert!(out.starts_with("<bos>"), "output should start with bos_token: {out}");
        assert!(out.contains("<|turn|>user\nhello<|end|>"), "user turn missing: {out}");
        assert!(out.ends_with("<|turn|>model\n"), "generation prompt missing: {out}");
    }

    /// Parse failure on an invalid Jinja template should surface as an error.
    #[test]
    fn jinja_template_parse_error_is_reported() {
        let tmpl = "{% unclosed"; // invalid
        let err = render_jinja_template(tmpl, "x").unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("parse") || msg.contains("Jinja") || msg.contains("template"),
            "expected parse error, got: {msg}"
        );
    }
}
