//! Inference engine for GGUF models — load, generate, and serve.

pub mod config;
pub mod gemma4;
pub mod gguf_loader;
pub mod sampler;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use std::path::Path;

use crate::cli;
use config::Gemma4Config;
use gemma4::{DispatchSnapshot, Gemma4Model, PerTokenMetrics};
use gguf_loader::GgufModel;
use sampler::SamplingParams;

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

    // Prefill (not timed — benchmark measures decode only)
    if !silent {
        eprintln!("Prefilling {} tokens...", prompt_tokens.len());
    }
    let input = Tensor::new(prompt_tokens, device)?
        .unsqueeze(0)?;  // [1, seq_len]
    let mut logits = model.forward(&input, 0)?;

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

    let elapsed = start.elapsed();
    Ok((generated, elapsed))
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

    // Load model weights from GGUF
    eprintln!("Loading model weights from GGUF (quantized QMatMul)...");
    let mut model = Gemma4Model::load(&cfg, &gguf, &device)?;

    // Warmup: run dummy token to force Metal shader compilation
    eprintln!("Warming up model...");
    let warmup_input = Tensor::new(&[2u32], &device)?.unsqueeze(0)?; // BOS token
    let _ = model.forward(&warmup_input, 0)?;
    model.clear_kv_cache();
    eprintln!("Warmup complete.");

    // Load tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

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
