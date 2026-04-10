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
use gemma4::Gemma4Model;
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

    // Prefill (not timed — benchmark measures decode only)
    if !silent {
        eprintln!("Prefilling {} tokens...", prompt_tokens.len());
    }
    let input = Tensor::new(prompt_tokens, device)?
        .unsqueeze(0)?;  // [1, seq_len]
    let mut logits = model.forward(&input, 0)?;

    let mut next_token = sampler::sample_token(&logits, params, &[])?;
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
        next_token = sampler::sample_token(&logits, params, &all_tokens)?;
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

    // Encode prompt using GGUF-style control tokens for Gemma4
    let prompt_text = format!(
        "<bos><|turn>system\n<|think|><turn|>\n<|turn>user\n{}<turn|>\n<|turn>model\n",
        prompt_text_raw
    );
    let encoding = tokenizer.encode(prompt_text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    tracing::info!("Prompt: {} tokens", prompt_tokens.len());

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

        for run in 1..=NUM_RUNS {
            model.clear_kv_cache();

            let (generated, elapsed) = run_single_generation(
                &mut model, &tokenizer, &prompt_tokens, &params, &device, true,
            )?;

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
