//! Inference engine for GGUF models — load, generate, and serve.
//!
//! ADR-008: single backend (mlx-native).  All candle code has been removed.

pub mod config;
pub mod forward_mlx;
pub mod forward_prefill;
pub mod gpu;
#[allow(dead_code)]
pub mod sampler_pure;

use anyhow::{Context, Result};
use std::path::Path;

use crate::cli;
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

    // Initialize mlx-native GPU context
    eprintln!("Initializing mlx-native GPU context...");
    let mut ctx = gpu::GpuContext::new()
        .map_err(|e| anyhow::anyhow!("mlx-native init failed: {e}"))?;
    eprintln!("mlx-native backend: {}", ctx.gpu_name());

    // Load weights directly from GGUF (ADR-008: no candle)
    eprintln!("Loading GGUF model...");
    let gguf = mlx_native::gguf::GgufFile::open(model_path)
        .map_err(|e| anyhow::anyhow!("GGUF open: {e}"))?;
    eprintln!("GGUF loaded: {} tensors, {} metadata keys",
        gguf.tensor_count(), gguf.metadata_count());

    eprintln!("Loading model weights from GGUF into mlx-native buffers...");
    let mut mlx_w = forward_mlx::MlxModelWeights::load_from_gguf(&gguf, &cfg, &mut ctx)?;
    eprintln!("mlx-native weights loaded ({} layers).", mlx_w.layers.len());

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

    let params = sampler_pure::SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        max_tokens: args.max_tokens,
    };

    // --- mlx-native forward pass ---
    use std::io::Write;

    eprintln!("Running mlx-native forward pass...");
    let eos_token_ids: Vec<u32> = vec![1, 106];

    // Profiling support
    let mut profiler = forward_mlx::ProfileAccumulator::new(2);
    let kernel_profile_mode = std::env::var("HF2Q_MLX_KERNEL_PROFILE")
        .map_or(false, |v| v == "1");

    // Prefill: true batched prefill with dense SDPA (ADR-009 Track 1).
    // Uses dense F32 attention instead of TQ-packed attention during prompt
    // ingestion to eliminate compounding quantization noise.
    let last_token = mlx_w.forward_prefill(&prompt_tokens, &mut ctx)?;

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
    eprintln!(
        "\n\n--- mlx-native: {} tokens in {:.2}s ({:.1} tok/s) ---",
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
