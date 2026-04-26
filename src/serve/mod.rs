//! Inference engine for GGUF models — load, generate, and serve.
//!
//! ADR-008: single backend (mlx-native).  All candle code has been removed.

pub mod api;
#[allow(dead_code)]
pub mod auto_pipeline;
#[allow(dead_code)]
pub mod cache;
pub mod config;
pub mod forward_mlx;
pub mod forward_prefill;
pub mod forward_prefill_batched;
pub mod gpu;
pub mod header;
pub mod parity_quality;
#[allow(dead_code)]
pub mod multi_model;
#[allow(dead_code)]
pub mod provenance;
#[allow(dead_code)]
pub mod quant_select;
#[allow(dead_code)]
pub mod sampler_pure;

use anyhow::{Context, Result};
use std::path::Path;

use crate::cli;
use crate::debug::INVESTIGATION_ENV;
use config::Gemma4Config;

/// Build a `KernelRegistry` with every shader the embedding forward
/// path needs registered AND compiled. One warmup forward is run
/// against the loaded weights so every `get_pipeline()` call hits the
/// cache thereafter. Returns the warmed registry; caller wraps it in
/// `Arc<Mutex<>>` and stashes in `AppState::embedding_registry` so
/// per-request handlers reuse the cached pipelines instead of paying
/// ~150 ms of shader-compile cost on every `/v1/embeddings` call.
fn build_warmed_embedding_registry(
    em: &api::state::EmbeddingModel,
) -> Result<mlx_native::KernelRegistry> {
    use crate::inference::models::bert::bert_gpu::{
        apply_bert_full_forward_gpu, register_bert_custom_shaders,
    };
    use crate::inference::models::nomic_bert::{
        apply_nomic_bert_full_forward_gpu, register_nomic_bert_kernels,
    };
    use api::state::EmbeddingArch;
    use mlx_native::{DType, KernelRegistry, MlxDevice};

    let arch = em
        .arch
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("registry warmup: EmbeddingModel has no arch"))?;

    let device = MlxDevice::new()
        .map_err(|e| anyhow::anyhow!("registry warmup: MlxDevice::new: {e}"))?;
    let mut registry = KernelRegistry::new();

    // Synthetic warmup input: a single padded-to-32 sequence of [PAD]
    // tokens. The exact ids don't matter — we just need to drive every
    // kernel through one full forward so all pipelines compile. Using
    // pad_id (which is in-vocab) keeps gather happy.
    let seq_len: u32 = 32;
    let pad_id = em.tokenizer.specials().pad;
    let ids: Vec<u32> = vec![pad_id; seq_len as usize];
    let ids_buf = device
        .alloc_buffer((seq_len as usize) * 4, DType::U32, vec![seq_len as usize])
        .map_err(|e| anyhow::anyhow!("registry warmup: alloc ids: {e}"))?;
    // SAFETY: just-allocated u32 buffer; exclusive access.
    unsafe {
        let s: &mut [u32] = std::slice::from_raw_parts_mut(
            ids_buf.contents_ptr() as *mut u32,
            seq_len as usize,
        );
        s.copy_from_slice(&ids);
    }

    let mut encoder = device
        .command_encoder()
        .map_err(|e| anyhow::anyhow!("registry warmup: command_encoder: {e}"))?;
    let valid_token_count: u32 = 1; // any value ≥ 1 ≤ seq_len works for warmup.

    let _out = match arch {
        EmbeddingArch::Bert { config, weights } => {
            register_bert_custom_shaders(&mut registry);
            apply_bert_full_forward_gpu(
                &mut encoder,
                &mut registry,
                &device,
                &ids_buf,
                None,
                weights,
                config,
                seq_len,
                valid_token_count,
            )?
        }
        EmbeddingArch::NomicBert { config, weights } => {
            register_nomic_bert_kernels(&mut registry);
            apply_nomic_bert_full_forward_gpu(
                &mut encoder,
                &mut registry,
                &device,
                &ids_buf,
                None,
                weights,
                config,
                seq_len,
                valid_token_count,
            )?
        }
    };
    encoder
        .commit_and_wait()
        .map_err(|e| anyhow::anyhow!("registry warmup: commit_and_wait: {e}"))?;

    tracing::info!(
        arch = arch.arch_name(),
        cached_pipelines = registry.cached_count(),
        "Warmed embedding kernel registry"
    );

    Ok(registry)
}

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
pub(crate) const FALLBACK_GEMMA4_CHAT_TEMPLATE: &str =
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
///
/// Extends the minijinja environment with Python string methods used by
/// Qwen3.5/3.6 and other HuggingFace chat templates:
///
/// - `str.startswith(prefix)` / `str.endswith(suffix)` — Qwen3.6 multi-step
///   tool-call detection path.
/// - `tojson` filter — common HF template helper.
/// - `raise_exception(msg)` function — Qwen3.6 guard for missing user query
///   (we always supply a user message, so this code path is unreachable; we
///   log a warning and continue rather than aborting template rendering).
fn render_jinja_template(template_str: &str, user_prompt: &str) -> Result<String> {
    let mut env = minijinja::Environment::new();

    // `tojson` filter — converts any value to its JSON representation.
    env.add_filter("tojson", |v: minijinja::Value| {
        serde_json::to_string(&v).unwrap_or_else(|_| "null".to_string())
    });

    // `raise_exception(msg)` — log-and-continue instead of aborting.
    // Qwen3.6 calls this when the message list has no user turn; we always
    // inject a user message so the guard should not fire.
    env.add_function("raise_exception", |msg: String| -> minijinja::Value {
        tracing::warn!("chat template raise_exception: {}", msg);
        minijinja::Value::UNDEFINED
    });

    // Python string method shims via `set_unknown_method_callback`.
    // Handles: `.startswith(prefix)`, `.endswith(suffix)` called as attribute
    // method invocations on string values (Qwen3.6 template line 72).
    env.set_unknown_method_callback(|_state, value, method, args| {
        let s = value.as_str().unwrap_or("");
        match method {
            "startswith" => {
                let prefix = args.first()
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                Ok(minijinja::Value::from(s.starts_with(prefix)))
            }
            "endswith" => {
                let suffix = args.first()
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                Ok(minijinja::Value::from(s.ends_with(suffix)))
            }
            other => Err(minijinja::Error::from(
                minijinja::ErrorKind::UnknownMethod,
            ).with_source(std::io::Error::other(format!(
                "string has no method named {other}"
            ))))
        }
    });

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
/// ADR-013 P11: routes `qwen35` / `qwen35moe` GGUF architectures to the
/// dedicated Qwen3.5 forward path (`cmd_generate_qwen35`) before attempting
/// the Gemma4 path, so the two model families share the same CLI surface.
pub fn cmd_generate(args: cli::GenerateArgs) -> Result<()> {
    let model_path = &args.model;
    anyhow::ensure!(model_path.exists(), "Model not found: {}", model_path.display());

    // --- Architecture detection (fast: metadata-only GGUF open) ---
    {
        let gguf_peek = mlx_native::gguf::GgufFile::open(model_path)
            .map_err(|e| anyhow::anyhow!("GGUF open (arch peek): {e}"))?;
        if let Some(arch) = gguf_peek.metadata_string("general.architecture") {
            use crate::inference::models::qwen35::{ARCH_QWEN35, ARCH_QWEN35MOE};
            if arch == ARCH_QWEN35 || arch == ARCH_QWEN35MOE {
                tracing::info!("Detected architecture '{}' → routing to Qwen3.5 path", arch);
                return cmd_generate_qwen35(args, gguf_peek);
            }
        }
    }

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

// ================================================================
// Qwen3.5 generate path (ADR-013 P13.1)
// ================================================================

/// Generate subcommand dispatch for `qwen35` / `qwen35moe` GGUF architectures.
///
/// Full end-to-end generate loop with stateful KV / SSM state threading:
///
/// 1. Load model + tokenizer from GGUF.
/// 2. Render + tokenize the prompt via GGUF's `tokenizer.chat_template`.
/// 3. Allocate [`HybridKvCache`] for the session (max_seq_len = prompt_len +
///    max_tokens, capped to `max_position_embeddings`).
/// 4. **Prefill**: call `forward_gpu(prompt_tokens, positions, kv_cache)`.
///    DeltaNet SSM state is threaded through `kv_cache.linear_attn` slots.
/// 5. **Decode loop** for `max_tokens` steps:
///    - Sample next token (argmax, temp=0 greedy).
///    - Check EOS (GGUF `tokenizer.ggml.eos_token_id`; default 151645 for
///      Qwen3.5 / Qwen3.6 per HF tokenizer_config.json).
///    - Call `forward_gpu([token], [pos], kv_cache)` — kv_cache carries the
///      DeltaNet conv+recurrent state from the previous step.
/// 6. Print the canonical hf2q 4-line header then the generated text.
///
/// # State threading
///
/// DeltaNet (Gated DeltaNet) layers maintain a recurrent state and a conv
/// ring-buffer that must persist across decode steps. `forward_gpu` now reads
/// from and writes back to the `kv_cache.linear_attn` slots on every call.
/// For full-attention layers the SDPA is re-run from scratch on each decode
/// token (the KV-append incremental path is a future optimisation).
fn cmd_generate_qwen35(
    args: cli::GenerateArgs,
    gguf: mlx_native::gguf::GgufFile,
) -> Result<()> {
    use crate::inference::models::qwen35::kv_cache::HybridKvCache;
    use crate::inference::models::qwen35::io_heads::greedy_argmax_last_token;
    use crate::inference::models::qwen35::model::Qwen35Model;
    use mlx_native::MlxDevice;
    use std::io::Write;

    let model_path = &args.model;
    let tokenizer_path = find_tokenizer(model_path, args.tokenizer.as_deref())?;

    tracing::info!("Qwen3.5 path: {}", model_path.display());
    tracing::info!("Tokenizer:    {}", tokenizer_path.display());

    // ---- Load model ----
    let load_start = std::time::Instant::now();
    tracing::info!("Loading Qwen3.5 model from GGUF");
    let model = Qwen35Model::load_from_gguf(&gguf)
        .context("Qwen35Model::load_from_gguf")?;
    let load_elapsed = load_start.elapsed();
    tracing::info!(
        "Qwen3.5 model loaded ({} layers, variant={:?}) in {:.2}s",
        model.layers.len(),
        model.cfg.variant,
        load_elapsed.as_secs_f64()
    );

    // ---- Resolve EOS from GGUF metadata ----
    // Qwen3.5 / Qwen3.6: tokenizer.ggml.eos_token_id is typically 151645 or 151643.
    // We prefer the GGUF-declared value (the authoritative source for this GGUF)
    // over any hard-coded default, per project_qwen36_architecture.md.
    let eos_token_id: u32 = gguf
        .metadata_u32("tokenizer.ggml.eos_token_id")
        .unwrap_or(151645); // HF Qwen3.5 default EOS
    tracing::info!("Qwen3.5 EOS token id: {}", eos_token_id);

    // ---- Load tokenizer ----
    let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer load failed: {e}"))?;
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow::anyhow!("Tokenizer truncation: {e}"))?;

    // ---- Resolve + render prompt ----
    let prompt_text_raw = resolve_prompt(&args)?;
    let prompt_text = render_chat_template(&gguf, &args, &prompt_text_raw)?;

    // ---- Tokenize ----
    let encoding = tokenizer
        .encode(prompt_text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = prompt_tokens.len();
    tracing::info!("Qwen3.5: {} prompt tokens", prompt_len);

    let max_seq = (prompt_len + args.max_tokens + 64)
        .max(128)
        .min(model.cfg.max_position_embeddings as usize);
    let spec_env = std::env::var("HF2Q_SPEC_DECODE").ok();
    let mut use_spec_decode = match spec_env.as_deref() {
        Some("0") => false,
        Some("1") => true,
        _ => args.speculative || model.mtp.is_some(),
    };
    if use_spec_decode && model.mtp.is_none() {
        tracing::warn!(
            "Speculative decoding requested but this GGUF has no MTP weights; using greedy decode"
        );
        use_spec_decode = false;
    }

    // ---- Build header info ----
    let backend_chip = {
        use crate::serve::gpu::GpuContext;
        GpuContext::new()
            .ok()
            .map(|c| c.gpu_name().to_string())
            .unwrap_or_else(|| "Metal".to_string())
    };
    let model_name = gguf
        .metadata_string("general.name")
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            model_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "qwen3.5".to_string())
        });
    let total_gb = std::fs::metadata(model_path)
        .map(|m| m.len() as f64 / 1e9)
        .unwrap_or(0.0);
    let n_layers = model.layers.len();
    let stdout_is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());

    let header_top = header::HeaderInfoTop {
        chip: header::short_chip_label(&backend_chip),
        backend: "mlx-native",
        model: model_name,
        load_s: load_elapsed.as_secs_f64(),
        n_layers,
        total_gb,
    };
    let mut stdout = std::io::stdout();
    header::print_header_top(&mut stdout, &header_top, stdout_is_tty)
        .context("print header top")?;

    if use_spec_decode {
        use crate::inference::models::qwen35::spec_decode::SpecDecode;
        tracing::info!("Qwen3.5 speculative decode enabled");
        let result = SpecDecode::run_with_eos(
            &model,
            &prompt_tokens,
            args.max_tokens,
            Some(eos_token_id),
            max_seq as u32,
        )
        .context("Qwen3.5 SpecDecode::run_with_eos")?;

        let prefill_tok_s = if result.stats.prefill_elapsed.as_secs_f64() > 0.0 {
            prompt_len as f64 / result.stats.prefill_elapsed.as_secs_f64()
        } else {
            0.0
        };
        header::print_header_prefill(
            &mut stdout,
            &header::HeaderInfoPrefill {
                prefill_n: prompt_len,
                prefill_ms: result.stats.prefill_elapsed.as_secs_f64() * 1000.0,
                prefill_tok_s,
            },
            stdout_is_tty,
        )
        .context("print header prefill")?;

        let decoded = tokenizer.decode(&result.tokens, false).unwrap_or_default();
        print!("{}", decoded);
        stdout.flush()?;

        let generated = result.tokens.len();
        let tok_per_sec = if result.stats.decode_elapsed.as_secs_f64() > 0.0 {
            generated as f64 / result.stats.decode_elapsed.as_secs_f64()
        } else {
            0.0
        };
        let (td, tr) = if std::io::IsTerminal::is_terminal(&std::io::stderr()) {
            ("\x1b[2m", "\x1b[0m")
        } else {
            ("", "")
        };
        eprintln!(
            "\n\n{td}--- mlx-native (Qwen3.5 spec): {} tokens in {:.2}s ({:.1} tok/s, accept {:.1}%) ---{tr}",
            generated,
            result.stats.decode_elapsed.as_secs_f64(),
            tok_per_sec,
            result.stats.acceptance_rate_pct(),
        );

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
            println!("Prompt tokens: {}", prompt_len);
            println!("Generated tokens: {}", generated);
            println!("Prefill tok/s: {:.1}", prefill_tok_s);
            println!("Decode tok/s: {:.1}", tok_per_sec);
            println!("Spec accept %: {:.1}", result.stats.acceptance_rate_pct());
        }
        return Ok(());
    }

    // ---- Allocate HybridKvCache ----
    let device = MlxDevice::new()
        .map_err(|e| anyhow::anyhow!("MlxDevice::new: {e}"))?;
    let mut kv_cache = HybridKvCache::new(&model.cfg, &device, max_seq as u32, 1)
        .context("HybridKvCache::new")?;
    tracing::info!(
        "Qwen3.5 KV cache allocated: max_seq={}, {} MB",
        max_seq,
        kv_cache.total_bytes() / (1024 * 1024)
    );

    // ---- Prefill ----
    // Build flat positions [4 * prompt_len]: axis-major, all axes = token index.
    let prefill_positions: Vec<i32> = {
        let mut flat = vec![0i32; 4 * prompt_len];
        for axis in 0..4 {
            for t in 0..prompt_len {
                flat[axis * prompt_len + t] = t as i32;
            }
        }
        flat
    };

    tracing::info!("Qwen3.5 prefill: seq_len={}", prompt_len);
    let prefill_start = std::time::Instant::now();
    let prefill_logits = model
        .forward_gpu(&prompt_tokens, &prefill_positions, &mut kv_cache)
        .context("Qwen35Model::forward_gpu (prefill)")?;
    let prefill_elapsed = prefill_start.elapsed();

    // Sanity-check logits shape.
    let vocab_size = model.cfg.vocab_size;
    anyhow::ensure!(
        prefill_logits.len() == prompt_len * vocab_size as usize,
        "forward_gpu (prefill) returned logits.len()={} != prompt_len({}) * vocab({}) = {}",
        prefill_logits.len(),
        prompt_len,
        vocab_size,
        prompt_len * vocab_size as usize,
    );

    let prefill_tok_s = prompt_len as f64 / prefill_elapsed.as_secs_f64();
    header::print_header_prefill(
        &mut stdout,
        &header::HeaderInfoPrefill {
            prefill_n: prompt_len,
            prefill_ms: prefill_elapsed.as_secs_f64() * 1000.0,
            prefill_tok_s,
        },
        stdout_is_tty,
    )
    .context("print header prefill")?;

    // HF2Q_DUMP_LOGITS=1: write the last-token logit vector to /tmp/hf2q_logits_t0.bin
    // and exit immediately. Used for first-token logit comparison vs llama.cpp.
    if std::env::var("HF2Q_DUMP_LOGITS").as_deref() == Ok("1") {
        let last_logits = &prefill_logits[prefill_logits.len() - vocab_size as usize..];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                last_logits.as_ptr() as *const u8,
                last_logits.len() * 4,
            )
        };
        std::fs::write("/tmp/hf2q_logits_t0.bin", bytes)
            .context("HF2Q_DUMP_LOGITS: write /tmp/hf2q_logits_t0.bin")?;
        // Top-3 to stderr for quick sanity check.
        let mut indexed: Vec<(usize, f32)> = last_logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("HF2Q_DUMP_LOGITS: wrote {} f32 values to /tmp/hf2q_logits_t0.bin", last_logits.len());
        eprintln!("  top-3: {:?}", &indexed[..3.min(indexed.len())]);
        return Ok(());
    }

    // Sample the first token from prefill logits (last token's row).
    let last_prefill_logits = &prefill_logits[prefill_logits.len() - vocab_size as usize..];
    let mut next_token = greedy_argmax_last_token(last_prefill_logits, vocab_size);
    tracing::info!("Qwen3.5 first decoded token: {}", next_token);

    // Print first token immediately.
    {
        let s = tokenizer.decode(&[next_token], false).unwrap_or_default();
        print!("{}", s);
        stdout.flush()?;
    }

    // ---- Decode loop ----
    let decode_start = std::time::Instant::now();
    let mut generated = 1usize;

    for step in 1..args.max_tokens {
        if next_token == eos_token_id {
            break;
        }

        // Absolute position of the decode token: prompt_len + (step - 1) since
        // step=1 is the second decode token, positioned at prompt_len.
        // (step=0 was the first decode token sampled from prefill; it already
        // "consumed" position prompt_len implicitly because we ran prefill at
        // positions 0..prompt_len-1, so the next position is prompt_len.)
        let pos = (prompt_len + step - 1) as i32;

        // Check we haven't overrun the KV cache.
        if pos as usize >= max_seq {
            tracing::warn!(
                "Qwen3.5 decode: reached max_seq {} at step {}; stopping",
                max_seq,
                step
            );
            break;
        }

        // Build single-token positions buffer: flat [4 * 1] all set to `pos`.
        let decode_positions = vec![pos; 4];

        // forward_gpu_greedy: GPU argmax → 4-byte download (vs 600KB full logits).
        // Eliminates ~5ms/token vocabulary download for greedy decode.
        let _t_step = if std::env::var("HF2Q_STEP_PROFILE").is_ok() {
            Some(std::time::Instant::now())
        } else { None };
        next_token = model
            .forward_gpu_greedy(&[next_token], &decode_positions, &mut kv_cache)
            .with_context(|| format!("forward_gpu_greedy decode step {step}"))?;
        if let Some(t) = _t_step {
            eprintln!("[STEP_PROFILE] step={step} total={:.2}ms", t.elapsed().as_micros() as f64 / 1000.0);
        }
        generated += 1;

        let s = tokenizer.decode(&[next_token], false).unwrap_or_default();
        print!("{}", s);
        stdout.flush()?;
    }

    let decode_elapsed = decode_start.elapsed();
    let tok_per_sec = generated as f64 / decode_elapsed.as_secs_f64();

    let (td, tr) = if std::io::IsTerminal::is_terminal(&std::io::stderr()) {
        ("\x1b[2m", "\x1b[0m")
    } else {
        ("", "")
    };
    eprintln!(
        "\n\n{td}--- mlx-native (Qwen3.5): {} tokens in {:.2}s ({:.1} tok/s) ---{tr}",
        generated,
        decode_elapsed.as_secs_f64(),
        tok_per_sec,
    );

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
        println!("Prompt tokens: {}", prompt_len);
        println!("Generated tokens: {}", generated);
        println!("Prefill tok/s: {:.1}", prefill_tok_s);
        println!("Decode tok/s: {:.1}", tok_per_sec);
    }

    Ok(())
}

/// ADR-005 Phase 4 iter-208 (W76): callable engine loader extracted from
/// `cmd_serve` so the [`multi_model::HotSwapManager`] can dispatch against
/// the same load path the single-model startup uses.
///
/// **Pure refactor.**  The body of this function is the byte-identical
/// sequence that previously lived inline in `cmd_serve` (header validate
/// → `LoadedModel::load` → `Engine::spawn` → optional synchronous warmup
/// on a one-shot tokio runtime).  `cmd_serve` now calls this for the
/// existing single-model startup path; iter-209 will plumb it through
/// `HotSwapManager::load_or_get` so a second model load mid-process is
/// possible without re-implementing the load.
///
/// **Why synchronous warmup.**  Iter-103 (`8109954`) fixed the
/// chat-warmup-logits-go-NaN bug surfaced when `--mmproj` is supplied:
/// running the chat warmup BEFORE any other Metal device activity (mmproj
/// load, embedding-model load, ViT warmup) keeps the chat-model GPU
/// state stable.  The hot-swap orchestrator preserves that ordering by
/// running each engine's warmup synchronously inside `load_engine` so
/// the engine returned to the caller is fully primed.
///
/// **Errors** propagate from header parse, weights load, tokenizer parse,
/// chat-template resolution, or warmup — every failure is fatal to the
/// load attempt.  Caller decides how to surface it (cmd_serve aborts the
/// boot; HotSwapManager will return the error to the request handler).
/// Derive a stable pool key for a filesystem-path passthrough — used
/// when `auto_pipeline::resolve_or_prepare_model` returns
/// `repo_id: None` (the operator passed `--model /path/to.gguf`
/// instead of an HF repo-id).  The file stem matches what the engine
/// itself uses for `model_id()` when GGUF metadata lacks `general.name`,
/// so the pool key matches the surface every other code path sees.
///
/// Deterministic per identical input: two requests resolving to the
/// same on-disk file yield the same key, the pool reuses the engine.
pub fn pool_key_for_path(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

pub fn load_engine(
    path: &Path,
    config: &multi_model::EngineConfig,
) -> Result<api::engine::Engine> {
    anyhow::ensure!(
        path.exists(),
        "Model not found: {}",
        path.display()
    );
    // Header-only parse surfaces bad magic immediately without loading
    // any tensor data.
    {
        let gguf = mlx_native::gguf::GgufFile::open(path)
            .map_err(|e| anyhow::anyhow!("GGUF header parse failed: {e}"))?;
        tracing::info!(
            path = %path.display(),
            tensors = gguf.tensor_count(),
            metadata = gguf.metadata_count(),
            "Validated GGUF header"
        );
    }

    let load_opts = api::engine::LoadOptions {
        model_path: path.to_path_buf(),
        tokenizer_path: config.tokenizer_path.clone(),
        config_path: config.config_path.clone(),
    };
    let loaded = api::engine::LoadedModel::load(&load_opts)?;
    let engine = api::engine::Engine::spawn(loaded, config.queue_capacity);

    if config.warmup_synchronously {
        // Warm up the engine SYNCHRONOUSLY here, BEFORE any other Metal
        // device activity (mmproj load, embedding-model load, ViT
        // warmup) happens.  This is iter-103's fix for the
        // chat-warmup-logits-go-NaN bug surfaced when `--mmproj` is
        // supplied: bisection showed loading the mmproj weights
        // (~1 GB of fresh F16-→F32 dequant'd Metal buffers) corrupts
        // the chat-model's pre-warmup state somehow (likely Metal-
        // driver buffer interleaving on Apple Silicon unified memory),
        // making every chat-model logit NaN at first decode.  Running
        // the chat warmup BEFORE any other Metal work is a structural
        // fix: the chat-model's GPU state is fully exercised + stable
        // by the time the mmproj load adds buffers.  As a happy side
        // effect, /readyz returns 200 immediately upon serving instead
        // of after the previously-async warmup completed.
        //
        // Uses a temp tokio runtime to drive the async API.  The
        // serve-time runtime is built later and is independent of
        // this one.
        let warmup_rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("build tokio runtime for synchronous engine warmup")?;
        let warmup_started = std::time::Instant::now();
        warmup_rt
            .block_on(engine.warmup())
            .context("synchronous engine warmup")?;
        tracing::info!(
            elapsed_ms = warmup_started.elapsed().as_millis() as u64,
            "Engine warmed up synchronously (pre-mmproj order — iter-103 fix)"
        );
        drop(warmup_rt);
    }

    Ok(engine)
}

/// Run the `serve` subcommand — start the OpenAI-compatible HTTP API server.
///
/// ADR-005 Phase 2a iter-2 backbone: exposes `/health`, `/readyz`,
/// `/v1/models`, `/v1/models/:id`. Chat completions + embeddings land in
/// the next iter with the engine wiring.
///
/// Behavior:
///   1. Build `ServerConfig` from CLI args + env vars.
///   2. If `--model` is supplied, validate the GGUF header opens cleanly
///      (fail-fast on bad weights, per Decision #15). No tensor data is
///      read — that happens when the engine loads in iter 3.
///   3. Build the axum router and bind the listener.
///   4. Serve until SIGINT / SIGTERM (graceful shutdown per Decision #17).
pub fn cmd_serve(args: cli::ServeArgs) -> Result<()> {
    use api::state::ServerConfig;
    use api::schema::OverflowPolicy;

    // --- Resolve config ---
    let auth_token = args
        .auth_token
        .clone()
        .or_else(|| std::env::var("HF2Q_AUTH_TOKEN").ok().filter(|s| !s.is_empty()));

    let overflow_policy = match args.overflow_policy {
        cli::OverflowPolicyArg::Reject => OverflowPolicy::Reject,
        cli::OverflowPolicyArg::TruncateLeft => OverflowPolicy::TruncateLeft,
        cli::OverflowPolicyArg::Summarize => OverflowPolicy::Summarize,
    };

    let cache_dir = args.cache_dir.clone().or_else(api::state::default_cache_dir);

    let config = ServerConfig {
        host: args.host.clone(),
        port: args.port,
        auth_token,
        cors_allowed_origins: args.cors_origins.clone(),
        queue_capacity: args.queue_capacity,
        max_concurrent_requests: 0,
        request_timeout_seconds: 0,
        default_overflow_policy: overflow_policy,
        cache_dir,
        system_fingerprint: Some(system_fingerprint()),
    };

    // Warn when exposing beyond localhost. Decision #7 + #13 — public-internet
    // is NOT a supported deployment target.
    if args.host == "0.0.0.0" {
        tracing::warn!(
            "Server bound to 0.0.0.0 — exposes API on all interfaces. \
             Public-internet exposure is NOT a supported deployment target \
             (see ADR-005 Decision #13: reverse-proxy assumption). \
             For LAN-only Open WebUI, this is the intended usage."
        );
    }

    // --- AppState construction (iter-209) ---
    // Build the pool-backed AppState before any model resolution: opens
    // the on-disk cache once + detects hardware once + constructs an
    // empty `HotSwapManager<Engine>` sized off the unified-memory budget
    // per ADR-005 line 929 (80% default).  The same `cache` + `hardware`
    // are shared with request-time auto_pipeline resolution.
    let default_model_arg = args
        .model
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned());
    let mut state = api::AppState::new_for_serve(
        config.clone(),
        args.no_integrity,
        config.queue_capacity,
        default_model_arg.clone(),
    )?;

    // --- Optional pre-warm (--model supplied) ---
    // ADR-005 Phase 4 iter-208 (W76): the load path flows through the
    // shared `load_engine` callable via `HotSwapManager::load_or_get`,
    // which dispatches against `DefaultModelLoader`.  iter-209 unifies
    // the startup path with the request-time auto-swap path: both call
    // `pool.load_or_get(...)` against the same manager.  Pre-warming at
    // startup keeps Decision #15 (fail-fast on bad weights) intact and
    // guarantees /readyz returns 200 with a usable pooled engine.
    //
    // Filesystem-path passthrough uses the file stem as the pool's
    // `repo` key + a synthetic `Q4_K_M` quant (the on-disk quant is
    // baked into the file; the pool key just needs determinism per
    // identical input).
    if let Some(model_arg) = default_model_arg.as_ref() {
        let mut cache_guard = state
            .cache
            .lock()
            .map_err(|e| anyhow::anyhow!("cache mutex poisoned at startup: {e}"))?;
        let resolved = auto_pipeline::resolve_or_prepare_model(
            model_arg,
            &mut cache_guard,
            state.hardware.as_ref(),
            state.no_integrity,
        )
        .context("auto-pipeline: resolve --model into a GGUF path")?;
        // Drop the cache lock before the long-running engine load so
        // request-time resolutions can proceed concurrently if they
        // somehow arrive (test paths) — production startup is
        // single-threaded but the lock-discipline is right anyway.
        drop(cache_guard);

        if let Some(repo) = resolved.repo_id.as_deref() {
            let quant_str: &str = resolved
                .quant
                .map(quant_select::QuantType::as_str)
                .unwrap_or("");
            tracing::info!(
                repo,
                quant = quant_str,
                from_cache = resolved.from_cache,
                gguf = %resolved.gguf_path.display(),
                "auto-pipeline: --model resolved"
            );
        }

        let pool_repo = resolved
            .repo_id
            .clone()
            .unwrap_or_else(|| pool_key_for_path(&resolved.gguf_path));
        let pool_quant = resolved
            .quant
            .unwrap_or(quant_select::QuantType::Q4_K_M);
        let engine_config = multi_model::EngineConfig {
            tokenizer_path: args.tokenizer.clone(),
            config_path: args.config.clone(),
            queue_capacity: config.queue_capacity,
            warmup_synchronously: true,
        };
        let mut pool_guard = state
            .pool
            .write()
            .map_err(|e| anyhow::anyhow!("pool rwlock poisoned at startup: {e}"))?;
        pool_guard
            .load_or_get(&pool_repo, pool_quant, &resolved.gguf_path, &engine_config)
            .map_err(|e| anyhow::anyhow!("startup pre-warm: {e}"))?;
        drop(pool_guard);
        tracing::info!(
            repo = %pool_repo,
            quant = %pool_quant.as_str(),
            "hf2q startup pre-warm: model admitted to pool"
        );
    }

    // --- Optionally validate + load the BERT embedding model config ---
    // Decision: load config only (header parse), NOT weights. Per
    // ADR-005 Phase 2b iter 16: the forward pass that consumes weights
    // lands when live-model validation is possible (OOM-blocked today).
    // The startup failure path still works: a bad GGUF at this path fails
    // the server boot cleanly.
    let embedding_model = if let Some(emb_path) = args.embedding_model.as_ref() {
        anyhow::ensure!(
            emb_path.exists(),
            "Embedding model not found: {}",
            emb_path.display()
        );
        let gguf = mlx_native::gguf::GgufFile::open(emb_path)
            .map_err(|e| anyhow::anyhow!("Embedding GGUF header parse failed: {e}"))?;

        // Sniff the architecture so we dispatch the correct loader +
        // forward path. Per ADR-005 Phase 2b: bge/mxbai are arch="bert"
        // (separate Q/K/V, position_embd, GeLU MLP, optionally CLS pool);
        // nomic-embed-text-v1.5 is arch="nomic-bert" (fused QKV, RoPE,
        // SwiGLU, Mean pool — see `inference::models::nomic_bert`).
        let arch_str = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| anyhow::anyhow!("Embedding GGUF missing general.architecture"))?
            .to_string();

        // Vocab + tokenizer are shared across the BERT family — both
        // archs serialize their WPM vocab the same way per llama.cpp's
        // `llm_tokenizer_wpm_session::tokenize`. Iter-79 cross-lane
        // edit added bos→cls / eos→sep fallbacks so nomic GGUFs parse
        // through the BertVocab path unchanged.
        let vocab = crate::inference::models::bert::BertVocab::from_gguf(&gguf)
            .map_err(|e| anyhow::anyhow!("Embedding GGUF vocab parse failed: {e}"))?;
        let tokenizer = crate::inference::models::bert::BertWpmTokenizer::new(&vocab);
        let model_id = emb_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "embedding-model".into());

        let device = mlx_native::MlxDevice::new()
            .map_err(|e| anyhow::anyhow!("create MlxDevice for embedding load: {e}"))?;

        let arch = match arch_str.as_str() {
            "bert" => {
                let cfg = crate::inference::models::bert::BertConfig::from_gguf(&gguf)
                    .map_err(|e| anyhow::anyhow!("BERT GGUF config parse failed: {e}"))?;
                crate::inference::models::bert::weights::validate_tensor_set(&gguf, &cfg)
                    .map_err(|e| anyhow::anyhow!("BERT GGUF tensor validation: {e}"))?;
                let weights =
                    crate::inference::models::bert::weights::LoadedBertWeights::load(
                        &gguf, &cfg, device,
                    )
                    .map_err(|e| anyhow::anyhow!("BERT weights load failed: {e}"))?;
                tracing::info!(
                    path = %emb_path.display(),
                    arch = "bert",
                    hidden = cfg.hidden_size,
                    layers = cfg.num_hidden_layers,
                    pooling = ?cfg.pooling_type,
                    vocab_size = vocab.len(),
                    tensor_count = weights.len(),
                    "Validated embedding GGUF + loaded weights onto device"
                );
                api::state::EmbeddingArch::Bert {
                    config: cfg,
                    weights: std::sync::Arc::new(weights),
                }
            }
            "nomic-bert" => {
                let cfg = crate::inference::models::nomic_bert::NomicBertConfig::from_gguf(&gguf)
                    .map_err(|e| anyhow::anyhow!("nomic-bert GGUF config parse failed: {e}"))?;
                crate::inference::models::nomic_bert::validate_tensor_set(&gguf, &cfg)
                    .map_err(|e| anyhow::anyhow!("nomic-bert GGUF tensor validation: {e}"))?;
                let weights =
                    crate::inference::models::nomic_bert::LoadedNomicBertWeights::load(
                        &gguf, &cfg, device,
                    )
                    .map_err(|e| anyhow::anyhow!("nomic-bert weights load failed: {e}"))?;
                tracing::info!(
                    path = %emb_path.display(),
                    arch = "nomic-bert",
                    hidden = cfg.hidden_size,
                    layers = cfg.num_hidden_layers,
                    pooling = ?cfg.pooling_type,
                    rope_freq_base = cfg.rope_freq_base,
                    vocab_size = vocab.len(),
                    tensor_count = weights.len(),
                    "Validated embedding GGUF + loaded weights onto device"
                );
                api::state::EmbeddingArch::NomicBert {
                    config: cfg,
                    weights: std::sync::Arc::new(weights),
                }
            }
            other => {
                anyhow::bail!(
                    "embedding GGUF general.architecture='{other}' is not supported. \
                     Phase 2b day-one models: 'bert' (bge / mxbai) and 'nomic-bert' \
                     (nomic-embed-text-v1.5). File: {}",
                    emb_path.display()
                );
            }
        };

        Some(api::state::EmbeddingModel {
            gguf_path: emb_path.clone(),
            vocab: std::sync::Arc::new(vocab),
            tokenizer: std::sync::Arc::new(tokenizer),
            model_id,
            arch: Some(arch),
        })
    } else {
        None
    };

    // --- Optionally validate + load the mmproj (multimodal projector) ---
    // Header parse only; weight loading lands alongside the ViT forward
    // pass (ADR-005 Phase 2c Task #15). Fail fast if the file is absent
    // or malformed so the server never advertises multimodal capability
    // it can't back.
    let mmproj = if let Some(mmp_path) = args.mmproj.as_ref() {
        anyhow::ensure!(
            mmp_path.exists(),
            "mmproj not found: {}",
            mmp_path.display()
        );
        let gguf = mlx_native::gguf::GgufFile::open(mmp_path)
            .map_err(|e| anyhow::anyhow!("mmproj GGUF header parse failed: {e}"))?;
        let mmp_config = crate::inference::vision::mmproj::MmprojConfig::from_gguf(&gguf)
            .map_err(|e| anyhow::anyhow!("mmproj GGUF config parse failed: {e}"))?;
        // Walk the GGUF's tensor list against the arch-agnostic
        // required set (iter 30 + iter 31). Fails fast on an incomplete
        // producer rather than hitting NotFound mid-forward-pass.
        let actual_names: Vec<&str> = gguf.tensor_names();
        crate::inference::vision::mmproj::validate_tensor_set(&mmp_config, &actual_names)
            .map_err(|e| anyhow::anyhow!("mmproj GGUF tensor-set validation: {e}"))?;
        // Detect the arch profile so forward-pass dispatch knows
        // which per-block-norm shape to expect (Gemma 4 SigLIP vs
        // classic CLIP vs Unknown).
        let arch = crate::inference::vision::mmproj::detect_arch_profile(&actual_names);
        if !arch.is_supported() {
            anyhow::bail!(
                "mmproj arch profile is Unknown — neither Gemma 4 \
                 SigLIP markers (ln1/ln2/post_ffw_norm) nor CLIP marker \
                 (attn_norm) found in block 0. hf2q's ViT forward pass \
                 cannot dispatch on this file."
            );
        }
        // Load every tensor onto the Metal device. For Gemma 4 this is
        // ~400MB / 356 tensors / ~10s cold-cache on M5 Max.
        //
        // Iter-103 added the `HF2Q_SKIP_MMPROJ_LOAD=1` escape hatch
        // for bisecting the chat-warmup-logits-go-NaN bug: if
        // skipping the mmproj weight load (just keep the config +
        // arch detection) makes chat warmup produce valid logits,
        // the bug is in `LoadedMmprojWeights::load`'s buffer-alloc
        // / dequant path; if NaN persists, the bug is somewhere
        // earlier (the GGUF mmap itself).
        let skip_mmproj_load =
            std::env::var("HF2Q_SKIP_MMPROJ_LOAD").as_deref() == Ok("1");
        let device = mlx_native::MlxDevice::new()
            .map_err(|e| anyhow::anyhow!("create MlxDevice for mmproj load: {e}"))?;
        let mmp_weights = if skip_mmproj_load {
            tracing::warn!(
                "HF2Q_SKIP_MMPROJ_LOAD=1 — using empty mmproj weights; \
                 vision requests will 500 on first forward attempt"
            );
            crate::inference::vision::mmproj_weights::LoadedMmprojWeights::empty(device)
        } else {
            crate::inference::vision::mmproj_weights::LoadedMmprojWeights::load(
                &gguf, &mmp_config, device,
            )
            .map_err(|e| anyhow::anyhow!("mmproj weight load: {e}"))?
        };
        let model_id = mmp_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "mmproj".into());
        tracing::info!(
            path = %mmp_path.display(),
            image_size = mmp_config.image_size,
            patch_size = mmp_config.patch_size,
            hidden = mmp_config.hidden_size,
            layers = mmp_config.num_hidden_layers,
            projector = mmp_config.projector.as_str(),
            arch = arch.as_str(),
            tensors_loaded = mmp_weights.len(),
            "Loaded mmproj GGUF header + tensor set + weights"
        );
        // Iter 53: ViT GPU warmup — runs one synthetic full forward
        // to trigger Metal kernel pipeline compilation. Drops first
        // user-visible multimodal request from ~5–10s (cold compile)
        // to ~1.3s (steady-state) on M5 Max.
        //
        // Iter-103 added the `HF2Q_SKIP_VIT_WARMUP=1` escape hatch
        // for bisecting the chat-warmup-logits-go-NaN bug: if
        // skipping the ViT warmup makes chat warmup produce valid
        // logits, the bug lives in `warmup_vit_gpu`'s leftover GPU
        // state; if NaN persists, the bug lives in
        // `LoadedMmprojWeights::load`.
        let skip_vit_warmup =
            std::env::var("HF2Q_SKIP_VIT_WARMUP").as_deref() == Ok("1");
        if skip_vit_warmup {
            tracing::warn!(
                "HF2Q_SKIP_VIT_WARMUP=1 — skipping ViT GPU warmup; first \
                 multimodal request will pay kernel-compile cost"
            );
        } else {
            let warmup_t0 = std::time::Instant::now();
            match crate::inference::vision::vit_gpu::warmup_vit_gpu(&mmp_weights, &mmp_config) {
                Ok(()) => tracing::info!(
                    elapsed_ms = warmup_t0.elapsed().as_millis() as u64,
                    "ViT GPU warmup complete"
                ),
                Err(e) => tracing::warn!(
                    error = %e,
                    "ViT GPU warmup failed; first multimodal request will pay kernel-compile cost"
                ),
            }
        }
        Some(api::state::LoadedMmproj {
            gguf_path: mmp_path.clone(),
            config: mmp_config,
            arch,
            weights: std::sync::Arc::new(mmp_weights),
            model_id,
        })
    } else {
        None
    };

    // --- Build router ---
    // `state` was constructed above (iter-209) with the cache + hardware
    // + empty pool; pre-warm has already admitted the `--model` engine
    // (when supplied).  Here we attach the embedding model + mmproj
    // descriptors before the router takes ownership.
    if let Some(em) = embedding_model {
        // Pre-warm a persistent kernel registry: register all kernels
        // the arch needs + run one warmup forward against the loaded
        // weights so every Metal pipeline compiles and caches before
        // the first /v1/embeddings request. Eliminates the ~150 ms
        // per-request shader-compile cost surfaced by iter-82
        // benchmarking. Stashes the registry behind an Arc<Mutex<>>
        // for handler dispatch.
        let registry = build_warmed_embedding_registry(&em).context("warm embedding registry")?;
        state = state
            .with_embedding_model(em)
            .with_embedding_registry(std::sync::Arc::new(std::sync::Mutex::new(registry)));
    }
    if let Some(m) = mmproj {
        state = state.with_mmproj(m);
    }
    let state_for_warmup = state.clone();
    let router = api::build_router(state);

    // --- Async runtime + serve ---
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;

    rt.block_on(async move {
        let bind = format!("{}:{}", config.host, config.port);
        let listener = tokio::net::TcpListener::bind(&bind)
            .await
            .with_context(|| format!("binding to {bind}"))?;
        let local_addr = listener.local_addr().ok();
        tracing::info!(
            addr = %local_addr.map(|a| a.to_string()).unwrap_or_else(|| bind.clone()),
            "hf2q HTTP server listening"
        );
        eprintln!("hf2q serving on http://{}", bind);

        // Iter-209: warmup ran SYNCHRONOUSLY at pre-warm time (when
        // `--model` was supplied) inside `pool.load_or_get` →
        // `DefaultModelLoader::load` → `load_engine` (warmup_synchronously
        // = true).  ready_for_gen is already initialized to `true` by
        // `AppState::new_for_serve`; this block remains as a no-op
        // observability anchor — the previous engine.is_some() guard is
        // replaced with a pool-state log so operators see the boot
        // ordering signal in the logs.  The iter-103 ordering invariant
        // (chat warmup BEFORE mmproj load) is preserved by the call
        // ordering above (pre-warm runs before mmproj load).
        {
            let pool_state_log = state_for_warmup
                .pool
                .read()
                .ok()
                .map(|m| m.pool_stats());
            if let Some(stats) = pool_state_log {
                tracing::info!(
                    loaded = stats.loaded_count,
                    capacity = stats.capacity_models,
                    bytes_resident = stats.total_resident_bytes,
                    bytes_budget = stats.memory_budget_bytes,
                    "hf2q ready (pool-backed; pre-warm complete if --model supplied)"
                );
            }
        }

        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .context("axum::serve")?;

        // Axum has stopped accepting + drained in-flight HTTP responses.
        // Each in-flight handler that called `engine.generate*` has already
        // received its reply.  Iter-209: with the pool replacing the
        // single-slot Option<Engine>, we shut down EVERY pooled engine in
        // parallel (each owns a separate worker thread + its own GPU
        // resources).  Without this, the tokio runtime drops at the
        // bottom of `block_on` and the `mpsc::Sender` to each worker is
        // closed implicitly; the worker exits its loop on the next
        // `blocking_recv`, but a mid-decode generation gets cut off
        // rather than running to its natural finish_reason.
        //
        // We snapshot the engine handles under the read lock, then drop
        // the lock before awaiting (await-while-holding-RwLock would
        // deadlock if any handler held it).  The pool itself is then
        // cleared so refcounts drop deterministically.
        let shutdown_engines: Vec<_> = state_for_warmup
            .pool
            .read()
            .ok()
            .map(|mgr| {
                mgr.snapshot_engines()
                    .into_iter()
                    .map(|le| le.engine.clone())
                    .collect()
            })
            .unwrap_or_default();
        for engine in shutdown_engines {
            match engine.shutdown().await {
                Ok(()) => tracing::info!("hf2q-engine worker joined"),
                Err(e) => tracing::warn!(error = %e, "hf2q-engine worker join failed"),
            }
        }

        tracing::info!("hf2q HTTP server shut down cleanly");
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}

/// Build the server's `system_fingerprint` — `hf2q-<short-git-sha-or-ver>-mlx-native`.
fn system_fingerprint() -> String {
    // Prefer the CARGO_PKG_VERSION (baked at build time). Git sha could be
    // added via a build.rs; for now the pkg version + backend is sufficient
    // identity for OpenAI's `system_fingerprint` contract.
    format!("hf2q-{}-mlx-native", env!("CARGO_PKG_VERSION"))
}

/// Graceful-shutdown signal handler: wait for Ctrl-C or SIGTERM (Decision #17).
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut s) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => tracing::info!("received SIGINT, shutting down"),
        _ = terminate => tracing::info!("received SIGTERM, shutting down"),
    }
}

/// Run the `cache` subcommand (ADR-005 Phase 3 iter-205, AC line 5351).
///
/// Three actions:
/// - `cache list`  — enumerate cached models + quants + on-disk size.
/// - `cache size`  — total bytes used by the cache.
/// - `cache clear` — invalidate entries (single quant, whole repo, or all).
///
/// Output is human-readable text on stdout; errors go through stderr
/// via the standard `Result` plumbing.  Bytes-freed is reported back
/// to the operator from the on-disk walk performed before removal.
///
/// Safety:
/// - `--all` requires `--yes` (refuses with a named error otherwise).
/// - Without `--model` and without `--all`, prints the usage and
///   exits with `Err` (not a silent no-op — the operator likely
///   intended one of the two).
pub fn cmd_cache(args: cli::CacheArgs) -> Result<()> {
    use cli::CacheAction;

    let mut cache = cache::ModelCache::open()
        .context("open model cache")?;
    match args.action {
        CacheAction::List => cmd_cache_list(&cache),
        CacheAction::Size => cmd_cache_size(&cache),
        CacheAction::Clear {
            model,
            quant,
            all,
            yes,
        } => cmd_cache_clear(&mut cache, model, quant, all, yes),
    }
}

fn cmd_cache_list(cache: &cache::ModelCache) -> Result<()> {
    let entries: Vec<_> = cache.iter_entries().collect();
    if entries.is_empty() {
        println!("(cache empty — root: {})", cache.root().display());
        return Ok(());
    }
    println!("hf2q cache @ {}", cache.root().display());
    println!(
        "{:<48} {:<10} {:>12} {:>20}",
        "MODEL", "QUANT", "BYTES", "LAST_ACCESSED"
    );
    for view in &entries {
        if view.model.quantizations.is_empty() {
            // Source recorded but no quants yet (an in-flight or
            // failed quantize); render the model row with `(none)`
            // so it's visible to `cache list`.
            println!(
                "{:<48} {:<10} {:>12} {:>20}",
                view.repo_id,
                "(none)",
                "-",
                view.model.last_accessed_secs,
            );
            continue;
        }
        for (quant, qe) in &view.model.quantizations {
            println!(
                "{:<48} {:<10} {:>12} {:>20}",
                view.repo_id,
                quant,
                qe.bytes,
                view.model.last_accessed_secs,
            );
        }
    }
    Ok(())
}

fn cmd_cache_size(cache: &cache::ModelCache) -> Result<()> {
    let total = cache.total_bytes_on_disk();
    println!(
        "hf2q cache @ {} — {} bytes ({:.2} GiB)",
        cache.root().display(),
        total,
        total as f64 / (1u64 << 30) as f64,
    );
    Ok(())
}

fn cmd_cache_clear(
    cache: &mut cache::ModelCache,
    model: Option<String>,
    quant: Option<String>,
    all: bool,
    yes: bool,
) -> Result<()> {
    use crate::serve::quant_select::QuantType;

    // Sanity: cannot mix --all with --model / --quant.  Refusing here
    // (rather than silently picking one path) prevents an operator
    // from running `cache clear --model x --all --yes` and being
    // surprised which one won.
    if all && (model.is_some() || quant.is_some()) {
        return Err(anyhow::anyhow!(
            "hf2q cache clear: --all is mutually exclusive with --model / --quant"
        ));
    }
    if !all && model.is_none() {
        return Err(anyhow::anyhow!(
            "hf2q cache clear: must specify --model <repo-id> [--quant <type>] \
             OR --all --yes (the latter purges every cached model)"
        ));
    }

    if all {
        if !yes {
            return Err(anyhow::anyhow!(
                "hf2q cache clear --all: refused without --yes \
                 (this would remove every cached model under {})",
                cache.root().display()
            ));
        }
        let freed = cache.purge().context("purge cache")?;
        println!(
            "hf2q cache: purged ({} bytes / {:.2} GiB freed)",
            freed,
            freed as f64 / (1u64 << 30) as f64,
        );
        return Ok(());
    }

    // --model is set (validated above).  --quant optional.
    let repo = model.expect("validated above");
    if let Some(q_str) = quant {
        let q = QuantType::from_canonical_str(&q_str)
            .map_err(|e| anyhow::anyhow!("--quant: {}", e))?;
        let freed = cache
            .invalidate(&repo, q)
            .with_context(|| format!("clear {}@{}", repo, q.as_str()))?;
        println!(
            "hf2q cache: cleared {}@{} ({} bytes freed)",
            repo,
            q.as_str(),
            freed
        );
    } else {
        let freed = cache
            .invalidate_repo(&repo)
            .with_context(|| format!("clear {} (all quants)", repo))?;
        println!(
            "hf2q cache: cleared {} (all quants — {} bytes freed)",
            repo, freed
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
        ParityCommand::Check {
            model,
            prompt,
            min_prefix,
            max_tokens,
            self_baseline,
            tq_quality,
            fixture,
            cosine_mean_floor,
            cosine_p1_floor,
            argmax_max,
            ppl_delta_max,
        } => {
            if tq_quality {
                // ADR-007 §853-866 Gate H — TQ-active envelope check.
                // The fixture is required: Gate H is a comparison gate
                // (cosine / argmax / PPL Δ vs the frozen
                // <prompt>_tq_quality.json), not absolute.  Erroring
                // early without --fixture beats running a 1000-token
                // two-regime decode and then discovering nothing to
                // compare against.
                if self_baseline {
                    anyhow::bail!(
                        "parity check --tq-quality is incompatible with \
                         --self-baseline (Gate D vs Gate H — different gates)"
                    );
                }
                let fixture = fixture.ok_or_else(|| {
                    anyhow::anyhow!(
                        "parity check --tq-quality requires --fixture \
                         <path/to/<prompt>_tq_quality.json>.\n\
                         Hint: the fixture is produced by `hf2q parity \
                         capture --tq-quality --model <gguf> --prompt \
                         {prompt}` (iter-112)."
                    )
                })?;
                parity_quality::cmd_parity_check_tq_quality(
                    &model,
                    &prompt,
                    &fixture,
                    cosine_mean_floor,
                    cosine_p1_floor,
                    argmax_max,
                    ppl_delta_max,
                    max_tokens,
                )
            } else {
                // Suppress unused-warnings for the Gate H args on the
                // byte-prefix path.
                let _ = (
                    fixture,
                    cosine_mean_floor,
                    cosine_p1_floor,
                    argmax_max,
                    ppl_delta_max,
                );
                cmd_parity_check(&model, &prompt, min_prefix, max_tokens, self_baseline)
            }
        }
        ParityCommand::Capture {
            model,
            output,
            prompt,
            max_tokens,
            tq_quality,
        } => {
            if tq_quality {
                parity_quality::cmd_parity_capture_tq_quality(
                    &model, &output, &prompt, max_tokens,
                )
            } else {
                cmd_parity_capture(&model, &output, &prompt, max_tokens)
            }
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
        speculative: false,
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
            speculative: false,
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
