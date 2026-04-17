//! Inference engine for GGUF models — load, generate, and serve.
//!
//! ADR-008: single backend (mlx-native).  All candle code has been removed.

pub mod config;
pub mod forward_mlx;
pub mod forward_prefill;
pub mod forward_prefill_batched;
pub mod gpu;
pub mod header;
#[allow(dead_code)]
pub mod sampler_pure;

use anyhow::{Context, Result};
use std::path::Path;

use crate::cli;
use crate::debug::INVESTIGATION_ENV;
use config::Gemma4Config;

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

/// Hardcoded fallback chat template used ONLY when no GGUF-embedded template
/// exists and the user has not passed `--chat-template` / `--chat-template-file`.
const FALLBACK_GEMMA4_CHAT_TEMPLATE: &str =
    "<bos><|turn>system\n<|think|><turn|>\n<|turn>user\n{{PROMPT}}<turn|>\n<|turn>model\n";

/// Resolve the chat template per ADR-005 Phase 1 priority order:
///
///   1. CLI `--chat-template STRING`
///   2. CLI `--chat-template-file FILE`
///   3. GGUF `tokenizer.chat_template` metadata
///   4. Hardcoded fallback string (last resort)
fn render_chat_template(
    gguf: &mlx_native::gguf::GgufFile,
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
    if let Some(tmpl) = gguf.metadata_string("tokenizer.chat_template") {
        tracing::info!(
            "Chat template: using GGUF metadata tokenizer.chat_template ({} chars)",
            tmpl.len()
        );
        return render_jinja_template(tmpl, user_prompt);
    }

    // Priority 4: hardcoded fallback
    tracing::warn!(
        "Chat template: no GGUF metadata tokenizer.chat_template and no CLI override; \
         falling back to hardcoded Gemma4 template"
    );
    Ok(FALLBACK_GEMMA4_CHAT_TEMPLATE.replace("{{PROMPT}}", user_prompt))
}

/// Render a Jinja2 chat template using minijinja.
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
///
/// ADR-008: single backend path — loads directly from GGUF into mlx-native.
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

    // Initialize mlx-native GPU context. Timing starts here; ends once
    // weights are resident. The elapsed duration feeds Step 5's header
    // line 2 ("loaded in Xs").
    let load_start = std::time::Instant::now();
    tracing::info!("Initializing mlx-native GPU context");
    let mut ctx = gpu::GpuContext::new()
        .map_err(|e| anyhow::anyhow!("mlx-native init failed: {e}"))?;
    let backend_chip = ctx.gpu_name().to_string();
    tracing::info!("mlx-native backend: {}", backend_chip);

    // Load weights directly from GGUF (ADR-008: no candle)
    tracing::info!("Loading GGUF model");
    let gguf = mlx_native::gguf::GgufFile::open(model_path)
        .map_err(|e| anyhow::anyhow!("GGUF open: {e}"))?;
    tracing::debug!("GGUF loaded: {} tensors, {} metadata keys",
        gguf.tensor_count(), gguf.metadata_count());

    // Extract human-readable model name from GGUF metadata, with fallback
    // to the file stem. Consumed by the header printer in Step 5.
    let model_name = gguf
        .metadata_string("general.name")
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            model_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".to_string())
        });
    tracing::debug!("Model name (GGUF general.name or file stem): {}", model_name);

    tracing::info!("Loading model weights from GGUF into mlx-native buffers");
    let stderr_is_tty = std::io::IsTerminal::is_terminal(&std::io::stderr());
    let stdout_is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());
    // Suppress progress line when tracing events at info+ are enabled
    // (verbosity >= 1) — the tracing debug/info stream already gives
    // per-layer visibility, and mixing \r overwrites with log lines is
    // garbled. Equivalent to "show progress only at default verbosity".
    let verbosity = if tracing::enabled!(tracing::Level::INFO) { 1 } else { 0 };
    let mut load_progress = header::LoadProgress::new(
        stderr_is_tty,
        verbosity,
        cfg.num_hidden_layers,
    );
    let mut mlx_w = forward_mlx::MlxModelWeights::load_from_gguf(
        &gguf, &cfg, &mut ctx, &mut load_progress,
    )?;
    let n_layers = mlx_w.layers.len();
    let load_elapsed = load_start.elapsed();
    tracing::info!("mlx-native weights loaded ({} layers) in {:.1}s",
        n_layers, load_elapsed.as_secs_f64());

    // Default-mode header lines 1 and 2 — product output on stdout,
    // dimmed on TTY. Line 3 (prefill stats) renders after prefill completes.
    let total_gb = std::fs::metadata(model_path)
        .map(|m| m.len() as f64 / 1e9)
        .unwrap_or(0.0);
    let header_top = header::HeaderInfoTop {
        chip: header::short_chip_label(&backend_chip),
        backend: "mlx-native",
        model: model_name.clone(),
        load_s: load_elapsed.as_secs_f64(),
        n_layers,
        total_gb,
    };
    let mut stdout = std::io::stdout();
    header::print_header_top(&mut stdout, &header_top, stdout_is_tty)
        .context("print header top")?;

    // Load tokenizer
    let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow::anyhow!("Failed to disable tokenizer truncation: {}", e))?;
    let tokenizer = tokenizer;

    // Resolve prompt
    let prompt_text_raw = resolve_prompt(&args)?;
    let prompt_text = render_chat_template(&gguf, &args, &prompt_text_raw)?;

    // ADR-005 1bNEW.0c: dump rendered prompt and exit if requested
    if let Some(dump_path) = INVESTIGATION_ENV.dump_rendered_prompt.as_deref() {
        std::fs::write(dump_path, prompt_text.as_bytes())
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
    if INVESTIGATION_ENV.dump_prompt_tokens {
        eprintln!("HF2Q_DUMP_PROMPT_TOKENS: first10={:?} last10={:?} total={}",
            &prompt_tokens[..prompt_tokens.len().min(10)],
            &prompt_tokens[prompt_tokens.len().saturating_sub(10)..],
            prompt_tokens.len());
    }

    let params = sampler_pure::SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        max_tokens: args.max_tokens,
    };

    // --- mlx-native forward pass ---
    use std::io::Write;

    tracing::info!("Running mlx-native forward pass");
    let eos_token_ids: Vec<u32> = vec![1, 106];

    // Profiling support
    let mut profiler = forward_mlx::ProfileAccumulator::new(2);
    let kernel_profile_mode = INVESTIGATION_ENV.mlx_kernel_profile;

    // Prefill: true batched prefill with dense SDPA (ADR-009 Track 1).
    // Uses dense F32 attention instead of TQ-packed attention during prompt
    // ingestion to eliminate compounding quantization noise.
    // ADR-009 Phase 3A: HF2Q_BATCHED_PREFILL=1 uses the new batched prefill
    // path (matches llama.cpp default). Per-token remains default until
    // parity is validated.
    let use_batched = INVESTIGATION_ENV.batched_prefill;
    let prefill_start = std::time::Instant::now();
    let last_token = if use_batched {
        mlx_w.forward_prefill_batched(&prompt_tokens, args.max_tokens, &mut ctx)?
    } else {
        mlx_w.forward_prefill(&prompt_tokens, args.max_tokens, &mut ctx)?
    };
    let prefill_elapsed = prefill_start.elapsed();

    // Default-mode header line 3 — prefill stats + blank line framing
    // the generation stream. Stdout, dimmed on TTY.
    let prefill_n = prompt_tokens.len();
    let prefill_ms = prefill_elapsed.as_secs_f64() * 1000.0;
    let prefill_tok_s = if prefill_elapsed.as_secs_f64() > 0.0 {
        prefill_n as f64 / prefill_elapsed.as_secs_f64()
    } else {
        0.0
    };
    header::print_header_prefill(
        &mut stdout,
        &header::HeaderInfoPrefill { prefill_n, prefill_ms, prefill_tok_s },
        stdout_is_tty,
    ).context("print header prefill")?;

    // Decode
    let mut all_tokens = prompt_tokens.to_vec();
    let mut next_token = last_token;
    all_tokens.push(next_token);
    {
        let token_str = tokenizer.decode(&[next_token], false).unwrap_or_default();
        print!("{}", token_str);
        std::io::stdout().flush()?;
    }

    let decode_start = std::time::Instant::now();
    let mut generated = 1usize;
    let mut kernel_profiles: Vec<forward_mlx::KernelTypeProfile> = Vec::new();
    let kernel_profile_warmup = 2usize;
    let kernel_profile_measure = 3usize;
    for _ in 1..params.max_tokens {
        if eos_token_ids.contains(&next_token) {
            break;
        }
        let pos = all_tokens.len() - 1;

        if kernel_profile_mode {
            let (tok, kp) = mlx_w.forward_decode_kernel_profile(
                next_token, pos, &mut ctx)?;
            next_token = tok;
            if generated > kernel_profile_warmup {
                kernel_profiles.push(kp);
            }
            if kernel_profiles.len() >= kernel_profile_measure {
                all_tokens.push(next_token);
                generated += 1;
                let token_str = tokenizer.decode(&[next_token], false).unwrap_or_default();
                print!("{}", token_str);
                std::io::stdout().flush()?;
                break;
            }
        } else {
            let mut p = profiler.start_token();
            next_token = mlx_w.forward_decode(next_token, pos, &mut ctx, &mut p)?;
            profiler.finish_token(p);
        }
        all_tokens.push(next_token);
        generated += 1;
        {
            let token_str = tokenizer.decode(&[next_token], false).unwrap_or_default();
            print!("{}", token_str);
            std::io::stdout().flush()?;
        }
    }
    let decode_elapsed = decode_start.elapsed();
    let tok_per_sec = generated as f64 / decode_elapsed.as_secs_f64();
    let (td, tr) = if stderr_is_tty { ("\x1b[2m", "\x1b[0m") } else { ("", "") };
    eprintln!(
        "\n\n{td}--- mlx-native: {} tokens in {:.2}s ({:.1} tok/s) ---{tr}",
        generated, decode_elapsed.as_secs_f64(), tok_per_sec,
    );

    // Print profiling summary if enabled
    profiler.print_summary();

    // Print kernel-type profiling report if enabled
    if kernel_profile_mode && !kernel_profiles.is_empty() {
        forward_mlx::MlxModelWeights::print_kernel_profile_report(&kernel_profiles);
    }

    if args.benchmark {
        let (chip, mem_gb) = detect_hardware_info();
        let model_filename = model_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        println!();
        println!("=== Benchmark Results ===");
        println!("Hardware: {}, {} GB", chip, mem_gb);
        println!("Model: {}", model_filename);
        println!("Prompt tokens: {}", prompt_tokens.len());
        println!("Generated tokens: {}", generated);
        println!("Decode tok/s: {:.1}", tok_per_sec);
    }

    // ADR-005 Gate G: mlx-native dispatch + sync counter emission.
    // Gated by HF2Q_DUMP_COUNTERS=1. Emits totals, per-prompt-token, and
    // per-decode-token rates on stderr so release-check.sh can threshold
    // them. The counters are atomic globals in mlx-native (not per-
    // invocation); for fresh numbers, run hf2q in a fresh process.
    if std::env::var("HF2Q_DUMP_COUNTERS").ok().as_deref() == Some("1") {
        let dispatches = mlx_native::dispatch_count();
        let syncs = mlx_native::sync_count();
        let prompt_n = prompt_tokens.len() as u64;
        let decode_n = generated as u64;
        let dispatches_per_prompt_tok = if prompt_n > 0 { dispatches as f64 / prompt_n as f64 } else { 0.0 };
        let syncs_per_decode_tok = if decode_n > 0 { syncs as f64 / decode_n as f64 } else { 0.0 };
        eprintln!(
            "[MLX_COUNTERS] dispatches={} syncs={} prompt_tokens={} decode_tokens={} \
             dispatches_per_prompt_tok={:.2} syncs_per_decode_tok={:.2}",
            dispatches, syncs, prompt_n, decode_n,
            dispatches_per_prompt_tok, syncs_per_decode_tok,
        );
    }

    Ok(())
}

/// Run the `parity` subcommand (ADR-009 Phase 2).
///
/// `parity check` — compare hf2q output against locked reference fixtures.
/// `parity capture` — generate fresh reference outputs from hf2q.
pub fn cmd_parity(args: cli::ParityArgs) -> Result<()> {
    use cli::ParityCommand;

    match args.command {
        ParityCommand::Check { model, prompt, min_prefix, max_tokens, self_baseline } => {
            cmd_parity_check(&model, &prompt, min_prefix, max_tokens, self_baseline)
        }
        ParityCommand::Capture { model, output, prompt, max_tokens } => {
            cmd_parity_capture(&model, &output, &prompt, max_tokens)
        }
    }
}

/// Parity check: run hf2q on a prompt and compare against locked reference.
fn cmd_parity_check(
    model_path: &Path,
    prompt_name: &str,
    min_prefix: Option<usize>,
    max_tokens: Option<usize>,
    self_baseline: bool,
) -> Result<()> {
    let evals_dir = Path::new("tests/evals");
    let ref_dir = evals_dir.join("reference");

    // Load prompt
    let prompt_file = evals_dir.join("prompts").join(format!("{prompt_name}.txt"));
    anyhow::ensure!(prompt_file.exists(), "Prompt file not found: {}", prompt_file.display());
    let prompt_text = std::fs::read_to_string(&prompt_file)?.trim().to_string();

    // Load reference. Default: llama.cpp-anchored parity (*_llama.txt).
    // Gate D (--self-baseline): hf2q frozen self-baseline (*_hf2q.txt),
    // bisect-safe when math deliberately changes and llama.cpp drift is
    // expected.
    let ref_suffix = if self_baseline { "_hf2q" } else { "_llama" };
    let ref_file = ref_dir.join(format!("{prompt_name}{ref_suffix}.txt"));
    anyhow::ensure!(ref_file.exists(), "Reference file not found: {}", ref_file.display());
    let ref_bytes = std::fs::read(&ref_file)?;

    // Determine settings
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(ref_dir.join("MANIFEST.json"))?
    )?;
    let prompt_meta = &manifest["prompts"][prompt_name];
    let tokens = max_tokens.unwrap_or_else(||
        prompt_meta["max_tokens"].as_u64().unwrap_or(1000) as usize
    );
    let threshold = min_prefix.unwrap_or_else(||
        // Parse from gate field like "common_prefix >= 3094"
        prompt_meta["parity_gate"].as_str()
            .and_then(|s| s.split(">=").nth(1))
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(0)
    );

    // Run hf2q
    eprintln!("=== Parity Check: {} ===", prompt_name);
    eprintln!("Model:     {}", model_path.display());
    eprintln!("Prompt:    {} ({} chars)", prompt_name, prompt_text.len());
    eprintln!("Tokens:    {}", tokens);
    eprintln!("Threshold: {} bytes", threshold);
    eprintln!();

    let tokenizer_path = find_tokenizer(model_path, None)?;
    let config_path = find_config(model_path, None)?;
    let cfg = config::Gemma4Config::from_config_json(&config_path)?;
    let mut ctx = gpu::GpuContext::new()
        .map_err(|e| anyhow::anyhow!("GPU init: {e}"))?;
    let gguf = mlx_native::gguf::GgufFile::open(model_path)
        .map_err(|e| anyhow::anyhow!("GGUF open: {e}"))?;
    // cmd_parity has its own output contract — no progress line.
    let mut parity_progress = header::LoadProgress::new(false, 1, 0);
    let mut mlx_w = forward_mlx::MlxModelWeights::load_from_gguf(
        &gguf, &cfg, &mut ctx, &mut parity_progress,
    )?;

    let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer: {e}"))?;
    tokenizer.with_truncation(None)
        .map_err(|e| anyhow::anyhow!("Tokenizer truncation: {e}"))?;

    let rendered = render_chat_template(&gguf, &cli::GenerateArgs {
        model: model_path.to_path_buf(),
        prompt: Some(prompt_text.clone()),
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
    }, &prompt_text)?;

    let encoding = tokenizer.encode(rendered.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenize: {e}"))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    // Prefill + decode
    let eos_token_ids: Vec<u32> = vec![1, 106];
    let first_token = mlx_w.forward_prefill(&prompt_tokens, tokens, &mut ctx)?;
    let mut all_tokens = prompt_tokens.to_vec();
    let mut next_token = first_token;
    all_tokens.push(next_token);

    for _ in 1..tokens {
        if eos_token_ids.contains(&next_token) {
            break;
        }
        let pos = all_tokens.len() - 1;
        let mut p = None;
        next_token = mlx_w.forward_decode(next_token, pos, &mut ctx, &mut p)?;
        all_tokens.push(next_token);
    }

    // Decode generated tokens to text
    let gen_tokens = &all_tokens[prompt_tokens.len()..];
    let hf2q_text = tokenizer.decode(gen_tokens, false).unwrap_or_default();
    let hf2q_bytes = hf2q_text.as_bytes();

    // Compare
    let n = ref_bytes.len().min(hf2q_bytes.len());
    let mut common = 0;
    while common < n && ref_bytes[common] == hf2q_bytes[common] {
        common += 1;
    }

    eprintln!();
    let ref_label = if self_baseline { "frozen hf2q" } else { "llama.cpp" };
    println!("Reference: {} bytes ({})", ref_bytes.len(), ref_label);
    println!("hf2q:      {} bytes", hf2q_bytes.len());
    println!("Common:    {} bytes", common);
    if self_baseline {
        // Gate D contract: byte-identical. Length must match AND every
        // byte in the common-prefix comparison was equal.
        let identical = hf2q_bytes.len() == ref_bytes.len() && common == ref_bytes.len();
        if identical {
            println!("PASS: byte-identical to frozen hf2q baseline ({} bytes)", common);
        } else {
            println!("FAIL: not byte-identical to frozen hf2q baseline");
            if common < n {
                let ctx_start = common;
                let ctx_end = (common + 80).min(n);
                let ref_snip = String::from_utf8_lossy(&ref_bytes[ctx_start..ctx_end]);
                let hf2q_snip = String::from_utf8_lossy(&hf2q_bytes[ctx_start..ctx_end.min(hf2q_bytes.len())]);
                println!();
                println!("Divergence at byte {}:", common);
                println!("  frozen: {:?}", ref_snip);
                println!("  hf2q:   {:?}", hf2q_snip);
            }
            anyhow::bail!("Self-baseline check failed: hf2q differs from frozen baseline");
        }
    } else {
        println!("Threshold: {} bytes", threshold);
        if common >= threshold {
            println!("PASS: {} >= {}", common, threshold);
            if common > threshold {
                println!("      ({} bytes above threshold)", common - threshold);
            }
        } else {
            println!("FAIL: {} < {}", common, threshold);
            // Show divergence context
            if common < n {
                let ctx_start = common;
                let ctx_end = (common + 80).min(n);
                let ref_snip = String::from_utf8_lossy(&ref_bytes[ctx_start..ctx_end]);
                let hf2q_snip = String::from_utf8_lossy(&hf2q_bytes[ctx_start..ctx_end.min(hf2q_bytes.len())]);
                println!();
                println!("Divergence at byte {}:", common);
                println!("  llama: {:?}", ref_snip);
                println!("  hf2q:  {:?}", hf2q_snip);
            }
            anyhow::bail!("Parity check failed: {} < {}", common, threshold);
        }
    }

    Ok(())
}

/// Parity capture: generate fresh hf2q output and save to reference dir.
fn cmd_parity_capture(
    model_path: &Path,
    output_dir: &Path,
    prompt_name: &str,
    max_tokens: Option<usize>,
) -> Result<()> {
    let evals_dir = Path::new("tests/evals");

    let prompts: Vec<String> = if prompt_name == "all" {
        vec!["sourdough".into(), "short_hello".into(), "sliding_wrap".into()]
    } else {
        vec![prompt_name.to_string()]
    };

    // Load model once
    let tokenizer_path = find_tokenizer(model_path, None)?;
    let config_path = find_config(model_path, None)?;
    let cfg = config::Gemma4Config::from_config_json(&config_path)?;
    let mut ctx = gpu::GpuContext::new()
        .map_err(|e| anyhow::anyhow!("GPU init: {e}"))?;
    let gguf = mlx_native::gguf::GgufFile::open(model_path)
        .map_err(|e| anyhow::anyhow!("GGUF open: {e}"))?;
    // Model loaded once here; individual prompts re-create weights below to reset KV state
    let _gguf_preload = &gguf; // keep gguf alive
    let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer: {e}"))?;
    tokenizer.with_truncation(None)
        .map_err(|e| anyhow::anyhow!("Tokenizer truncation: {e}"))?;

    std::fs::create_dir_all(output_dir)?;

    for pname in &prompts {
        let prompt_file = evals_dir.join("prompts").join(format!("{pname}.txt"));
        anyhow::ensure!(prompt_file.exists(), "Prompt not found: {}", prompt_file.display());
        let prompt_text = std::fs::read_to_string(&prompt_file)?.trim().to_string();

        let tokens = max_tokens.unwrap_or(match pname.as_str() {
            "sourdough" => 1000,
            "short_hello" => 50,
            "sliding_wrap" => 500,
            _ => 200,
        });

        eprintln!("Capturing: {} ({} tokens)", pname, tokens);

        // Need to reload model for each prompt since KV cache state persists
        // Re-create model weights (reset KV caches).
        // cmd_parity has its own output contract — no progress line.
        let mut parity_progress = header::LoadProgress::new(false, 1, 0);
        let mut mlx_w_fresh = forward_mlx::MlxModelWeights::load_from_gguf(
            &gguf, &cfg, &mut ctx, &mut parity_progress,
        )?;

        let rendered = render_chat_template(&gguf, &cli::GenerateArgs {
            model: model_path.to_path_buf(),
            prompt: Some(prompt_text.clone()),
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
        }, &prompt_text)?;

        let encoding = tokenizer.encode(rendered.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenize: {e}"))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

        let eos_token_ids: Vec<u32> = vec![1, 106];
        let first_token = mlx_w_fresh.forward_prefill(&prompt_tokens, tokens, &mut ctx)?;
        let mut all_tokens = prompt_tokens.to_vec();
        let mut next_token = first_token;
        all_tokens.push(next_token);

        for _ in 1..tokens {
            if eos_token_ids.contains(&next_token) {
                break;
            }
            let pos = all_tokens.len() - 1;
            let mut p = None;
            next_token = mlx_w_fresh.forward_decode(next_token, pos, &mut ctx, &mut p)?;
            all_tokens.push(next_token);
        }

        let gen_tokens = &all_tokens[prompt_tokens.len()..];
        let text = tokenizer.decode(gen_tokens, false).unwrap_or_default();

        let out_path = output_dir.join(format!("{pname}_hf2q.txt"));
        std::fs::write(&out_path, &text)?;
        eprintln!("  Wrote {} bytes to {}", text.len(), out_path.display());
    }

    Ok(())
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
