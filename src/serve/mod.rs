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

/// Run the `generate` subcommand.
pub fn cmd_generate(args: cli::GenerateArgs) -> Result<()> {
    use std::io::Write;

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
    eprintln!("Loading model weights from GGUF (this dequantizes ~13GB)...");
    let mut model = Gemma4Model::load(&cfg, &gguf, &device)?;

    // Load tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Encode prompt with chat template
    let prompt_text = format!(
        "<bos><start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
        args.prompt
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

    // Generate
    let eos_token_ids: Vec<u32> = vec![1, 106]; // Gemma EOS tokens
    let mut all_tokens = prompt_tokens.clone();

    // Prefill
    eprintln!("Prefilling {} tokens...", prompt_tokens.len());
    let input = Tensor::new(prompt_tokens.as_slice(), &device)?
        .unsqueeze(0)?;  // [1, seq_len]
    let mut logits = model.forward(&input, 0)?;
    // Debug: inspect logits
    {
        let l = logits.to_dtype(candle_core::DType::F32)?.squeeze(0)?.squeeze(0)?;
        let l_vec: Vec<f32> = l.to_vec1()?;
        let max_val = l_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = l_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let nan_count = l_vec.iter().filter(|v| v.is_nan()).count();
        let inf_count = l_vec.iter().filter(|v| v.is_infinite()).count();
        let argmax = l_vec.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i,_)| i).unwrap_or(0);
        eprintln!("[DEBUG] logits: min={:.4}, max={:.4}, nan={}, inf={}, argmax={}, vocab_size={}",
            min_val, max_val, nan_count, inf_count, argmax, l_vec.len());
        // Show top-5 tokens
        let mut indexed: Vec<(usize, f32)> = l_vec.iter().copied().enumerate().collect();
        indexed.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("[DEBUG] top-5: {:?}", &indexed[..5.min(indexed.len())]);
    }

    let mut next_token = sampler::sample_token(&logits, &params, &[])?;
    all_tokens.push(next_token);

    let token_str = tokenizer.decode(&[next_token], false)
        .unwrap_or_default();
    eprintln!("[DEBUG] first token: id={}, str={:?}", next_token, token_str);
    print!("{}", token_str);
    std::io::stdout().flush()?;

    // Decode loop
    let start = std::time::Instant::now();
    let mut generated = 1usize;
    for _ in 1..args.max_tokens {
        if eos_token_ids.contains(&next_token) {
            break;
        }

        let input = Tensor::new(&[next_token], &device)?
            .unsqueeze(0)?;  // [1, 1]
        let seqlen_offset = all_tokens.len() - 1;
        logits = model.forward(&input, seqlen_offset)?;
        next_token = sampler::sample_token(&logits, &params, &all_tokens)?;
        all_tokens.push(next_token);
        generated += 1;

        let token_str = tokenizer.decode(&[next_token], false)
            .unwrap_or_default();
        if generated <= 5 {
            eprintln!("[DEBUG] token {}: id={}, str={:?}", generated, next_token, token_str);
        }
        print!("{}", token_str);
        std::io::stdout().flush()?;
    }

    let elapsed = start.elapsed();
    let tok_per_sec = generated as f64 / elapsed.as_secs_f64();
    eprintln!("\n\n--- {} tokens in {:.2}s ({:.1} tok/s) ---", generated, elapsed.as_secs_f64(), tok_per_sec);

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
