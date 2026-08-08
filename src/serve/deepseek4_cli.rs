//! Native one-shot DeepSeek-V4 generation for `hf2q generate`.

use std::io::Write;
use std::time::Instant;

use anyhow::{Context, Result};
use mlx_native::gguf::GgufFile;
use tokenizers::Tokenizer;

use crate::cli;
use crate::core::deepseek_v4_encoding::{
    encode_json, EncodeOptions, ReasoningEffort, ThinkingMode,
};
use crate::inference::models::deepseek4::{cache::Deepseek4Cache, tokenizer, Deepseek4Model};

use super::resolve_prompt;

pub(super) fn cmd_generate(args: cli::GenerateArgs, gguf: GgufFile) -> Result<()> {
    anyhow::ensure!(
        args.temperature == 0.0,
        "DeepSeek-V4 native generation currently supports greedy temperature=0 only"
    );
    anyhow::ensure!(
        args.repetition_penalty == 1.0,
        "DeepSeek-V4 native generation does not yet apply repetition penalties"
    );
    anyhow::ensure!(
        args.mmproj.is_none() && args.image.is_none(),
        "DeepSeek-V4-Flash-0731 is a text-only runtime"
    );

    let tokenizer = match args.tokenizer.as_deref() {
        Some(path) => Tokenizer::from_file(path)
            .map_err(|error| anyhow::anyhow!("load tokenizer {}: {error}", path.display()))?,
        None => tokenizer::build_tokenizer_from_gguf(&gguf)
            .context("build DeepSeek-V4 tokenizer from GGUF metadata")?,
    };
    let raw_prompt = resolve_prompt(&args)?;
    let rendered = if args.chat_template.is_some() || args.chat_template_file.is_some() {
        super::render_chat_template(&gguf, &args, Some(&tokenizer), &raw_prompt)?
    } else {
        let messages = serde_json::to_string(&vec![
            serde_json::json!({"role": "user", "content": raw_prompt}),
        ])?;
        encode_json(
            &messages,
            EncodeOptions {
                thinking_mode: if args.no_thinking {
                    ThinkingMode::Chat
                } else {
                    ThinkingMode::Thinking
                },
                drop_thinking: true,
                add_bos: true,
                reasoning_effort: ReasoningEffort::Low,
            },
        )?
    };
    if let Some(path) = crate::debug::INVESTIGATION_ENV
        .dump_rendered_prompt
        .as_deref()
    {
        std::fs::write(path, rendered.as_bytes())
            .with_context(|| format!("write rendered DeepSeek-V4 prompt to {path}"))?;
        eprintln!(
            "HF2Q_DUMP_RENDERED_PROMPT: wrote {} bytes to {path}",
            rendered.len()
        );
        return Ok(());
    }
    let prompt = tokenizer
        .encode(rendered, false)
        .map_err(|error| anyhow::anyhow!("tokenize DeepSeek-V4 prompt: {error}"))?;
    let prompt_tokens = prompt.get_ids();
    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "DeepSeek-V4 prompt encoded to zero tokens"
    );
    if std::env::var("HF2Q_DEBUG_TOKENIZE_ONLY").as_deref() == Ok("1") {
        let ids = prompt_tokens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        println!("TOKENIZE_DEBUG_IDS: {ids}");
        return Ok(());
    }

    let load_started = Instant::now();
    let mut model = Deepseek4Model::load_from_gguf(&gguf).context("load DeepSeek-V4 model")?;
    anyhow::ensure!(
        prompt_tokens
            .iter()
            .all(|token| *token < model.cfg.vocab_size),
        "DeepSeek-V4 prompt contains a token outside the model vocabulary"
    );
    let requested_context = prompt_tokens
        .len()
        .checked_add(args.max_tokens)
        .context("DeepSeek-V4 requested context overflow")?;
    anyhow::ensure!(
        requested_context <= model.cfg.max_position_embeddings as usize,
        "DeepSeek-V4 prompt + generation length {} exceeds context {}",
        requested_context,
        model.cfg.max_position_embeddings
    );
    let cache_context = requested_context.max(model.cfg.sliding_window as usize);
    let mut cache = model
        .allocate_cache(cache_context)
        .context("allocate DeepSeek-V4 request cache")?;
    eprintln!(
        "DeepSeek-V4 loaded in {:.2}s; {} prompt tokens; {} resident weight bytes",
        load_started.elapsed().as_secs_f64(),
        prompt_tokens.len(),
        model.weights.resident_bytes()
    );

    if args.benchmark {
        return run_benchmark(&args, &gguf, prompt_tokens, &mut model, &mut cache);
    }

    let prefill_started = Instant::now();
    if stage_profile_enabled() {
        mlx_native::kernel_profile::reset();
    }
    if pipeline_bucket_enabled() {
        mlx_native::reset_pipeline_dispatch_buckets();
    }
    let prefill_gpu_started = mlx_native::gpu_busy_ns();
    let prefill_dispatch_started = mlx_native::dispatch_count();
    let prefill_cb_started = mlx_native::cmd_buf_count();
    let final_state = if std::env::var("HF2Q_DEEPSEEK_SEQUENTIAL_PREFILL").as_deref() == Ok("1") {
        let mut state = None;
        for &token in prompt_tokens {
            state = Some(
                model
                    .forward_verifier_one(token, &mut cache)
                    .context("execute DeepSeek-V4 sequential prompt token")?,
            );
        }
        state.context("DeepSeek-V4 sequential prompt produced no state")?
    } else {
        model
            .forward_verifier_prompt(prompt_tokens, &mut cache)
            .context("execute DeepSeek-V4 batched prompt prefill")?
    };
    let prefill_elapsed = prefill_started.elapsed();
    eprintln!(
        "DeepSeek-V4 prefill: {} tokens in {:.2}s ({:.3} tok/s)",
        prompt_tokens.len(),
        prefill_elapsed.as_secs_f64(),
        prompt_tokens.len() as f64 / prefill_elapsed.as_secs_f64()
    );
    print_gpu_profile(
        "prefill",
        prefill_gpu_started,
        prefill_dispatch_started,
        prefill_cb_started,
    );
    print_stage_profile("prefill", prefill_gpu_started);
    print_pipeline_profile("prefill");
    // Match the existing Qwen diagnostic contract: capture the exact
    // post-prefill state and first-token logits, then exit before sampling.
    // Keeping this path read-only lets matrix and sequential prefill runs be
    // compared without generation policy becoming a confounder.
    if std::env::var("HF2Q_DUMP_LOGITS").as_deref() == Ok("1") {
        let prompt_last_state = model
            .last_token_state(&final_state)
            .context("view DeepSeek-V4 diagnostic final prompt state")?;
        let logits = model
            .forward_logits(&prompt_last_state)
            .context("execute DeepSeek-V4 diagnostic output head")?;
        write_f32_dump(
            "/tmp/hf2q_deepseek_state_t0.bin",
            prompt_last_state.as_logical_slice::<f32>()?,
        )?;
        let logit_values = logits.as_slice::<f32>()?;
        write_f32_dump("/tmp/hf2q_logits_t0.bin", logit_values)?;
        let mut indexed = logit_values.iter().copied().enumerate().collect::<Vec<_>>();
        indexed.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        eprintln!(
            "HF2Q_DUMP_LOGITS: wrote {} state and {} logit f32 values",
            prompt_last_state.shape().iter().product::<usize>(),
            logit_values.len()
        );
        eprintln!("  top-3: {:?}", &indexed[..3.min(indexed.len())]);
        return Ok(());
    }
    if args.max_tokens == 0 {
        return Ok(());
    }

    let decode_started = Instant::now();
    if stage_profile_enabled() {
        mlx_native::kernel_profile::reset();
    }
    if pipeline_bucket_enabled() {
        mlx_native::reset_pipeline_dispatch_buckets();
    }
    let decode_gpu_started = mlx_native::gpu_busy_ns();
    let decode_dispatch_started = mlx_native::dispatch_count();
    let decode_cb_started = mlx_native::cmd_buf_count();
    let prompt_last_state = model
        .last_token_state(&final_state)
        .context("view DeepSeek-V4 final prompt state")?;
    let mut logits = model
        .forward_logits(&prompt_last_state)
        .context("execute DeepSeek-V4 prompt output head")?;
    let eos = gguf
        .metadata_u32("tokenizer.ggml.eos_token_id")
        .unwrap_or(1);
    let mut generated = 0usize;
    let mut stream = tokenizer.decode_stream(false);
    let mut stdout = std::io::stdout().lock();
    for step in 0..args.max_tokens {
        let token = model
            .greedy_token(&logits)
            .context("select DeepSeek-V4 greedy token")?;
        if !args.ignore_eos && token == eos {
            break;
        }
        generated += 1;
        if let Some(fragment) = stream
            .step(token)
            .map_err(|error| anyhow::anyhow!("decode DeepSeek-V4 token {token}: {error}"))?
        {
            write!(stdout, "{fragment}")?;
            stdout.flush()?;
        }
        if step + 1 < args.max_tokens {
            let state = model
                .forward_verifier_one(token, &mut cache)
                .with_context(|| {
                    format!("DeepSeek-V4 decode token at position {}", cache.position())
                })?;
            logits = model
                .forward_logits(&state)
                .context("execute DeepSeek-V4 decode output head")?;
        }
    }
    writeln!(stdout)?;
    let decode_elapsed = decode_started.elapsed();
    eprintln!(
        "DeepSeek-V4 decode: {generated} tokens in {:.2}s ({:.3} tok/s)",
        decode_elapsed.as_secs_f64(),
        generated as f64 / decode_elapsed.as_secs_f64()
    );
    print_gpu_profile(
        "decode",
        decode_gpu_started,
        decode_dispatch_started,
        decode_cb_started,
    );
    print_stage_profile("decode", decode_gpu_started);
    print_pipeline_profile("decode");
    Ok(())
}

fn write_f32_dump(path: &str, values: &[f32]) -> Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    std::fs::write(path, bytes).with_context(|| format!("write diagnostic dump {path}"))
}

fn run_benchmark(
    args: &cli::GenerateArgs,
    gguf: &GgufFile,
    prompt_tokens: &[u32],
    model: &mut Deepseek4Model,
    cache: &mut Deepseek4Cache,
) -> Result<()> {
    const RUNS: usize = 5;

    let eos = gguf
        .metadata_u32("tokenizer.ggml.eos_token_id")
        .unwrap_or(1);
    let mut prefill_tps = Vec::with_capacity(RUNS);
    let mut decode_tps = Vec::with_capacity(RUNS);
    let mut generated_per_run = Vec::with_capacity(RUNS);
    let mut decode_steps_per_run = Vec::with_capacity(RUNS);

    for run in 0..RUNS {
        cache.reset().context("reset DeepSeek-V4 benchmark cache")?;

        // Match llama.cpp's timing boundary: prompt evaluation includes the
        // output head that produces the first generated token. Decode timing
        // then contains one verifier + output-head evaluation per subsequent
        // token (llama.cpp reports these as its eval runs).
        let prefill_started = Instant::now();
        let final_state = model
            .forward_verifier_prompt(prompt_tokens, cache)
            .context("execute DeepSeek-V4 benchmark prefill")?;
        let prompt_last_state = model
            .last_token_state(&final_state)
            .context("view DeepSeek-V4 benchmark final prompt state")?;
        let mut logits = model
            .forward_logits(&prompt_last_state)
            .context("execute DeepSeek-V4 benchmark prompt output head")?;
        let prefill_elapsed = prefill_started.elapsed();

        let mut generated = 0usize;
        let mut decode_steps = 0usize;
        let mut token = None;
        let mut stopped = false;
        if args.max_tokens > 0 {
            let first = model
                .greedy_token(&logits)
                .context("select DeepSeek-V4 benchmark first token")?;
            if args.ignore_eos || first != eos {
                generated = 1;
                token = Some(first);
            } else {
                stopped = true;
            }
        }

        let decode_started = Instant::now();
        while !stopped && generated < args.max_tokens {
            let previous = token.context("DeepSeek-V4 benchmark decode token is missing")?;
            let state = model
                .forward_verifier_one(previous, cache)
                .with_context(|| {
                    format!(
                        "execute DeepSeek-V4 benchmark token at position {}",
                        cache.position()
                    )
                })?;
            logits = model
                .forward_logits(&state)
                .context("execute DeepSeek-V4 benchmark decode output head")?;
            decode_steps += 1;
            let next = model
                .greedy_token(&logits)
                .context("select DeepSeek-V4 benchmark decode token")?;
            if !args.ignore_eos && next == eos {
                break;
            }
            generated += 1;
            token = Some(next);
        }
        let decode_elapsed = decode_started.elapsed();
        let prefill_rate = prompt_tokens.len() as f64 / prefill_elapsed.as_secs_f64();
        let decode_rate = if decode_steps == 0 {
            0.0
        } else {
            decode_steps as f64 / decode_elapsed.as_secs_f64()
        };
        eprintln!(
            "  Run {}/{}: prefill {} tok in {:.3}s ({:.1} tok/s); decode {} evals / {} generated tok in {:.3}s ({:.1} tok/s)",
            run + 1,
            RUNS,
            prompt_tokens.len(),
            prefill_elapsed.as_secs_f64(),
            prefill_rate,
            decode_steps,
            generated,
            decode_elapsed.as_secs_f64(),
            decode_rate,
        );
        prefill_tps.push(prefill_rate);
        decode_tps.push(decode_rate);
        generated_per_run.push(generated);
        decode_steps_per_run.push(decode_steps);
    }

    let decode_steps = decode_steps_per_run
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    super::print_benchmark_summary(
        &args.model,
        prompt_tokens.len(),
        &generated_per_run,
        Some(&prefill_tps),
        &decode_tps,
        &[("Decode evals".to_string(), decode_steps)],
    );
    Ok(())
}

fn print_gpu_profile(label: &str, gpu_started: u64, dispatch_started: u64, cb_started: u64) {
    if std::env::var("HF2Q_GPU_BUSY").as_deref() != Ok("1") {
        return;
    }
    eprintln!(
        "DeepSeek-V4 {label} GPU profile: {:.3}s busy; {} dispatches; {} command buffers",
        mlx_native::gpu_busy_ns().saturating_sub(gpu_started) as f64 / 1e9,
        mlx_native::dispatch_count().saturating_sub(dispatch_started),
        mlx_native::cmd_buf_count().saturating_sub(cb_started),
    );
}

fn stage_profile_enabled() -> bool {
    std::env::var("HF2Q_DEEPSEEK_STAGE_PROFILE").as_deref() == Ok("1")
}

fn pipeline_bucket_enabled() -> bool {
    std::env::var("MLX_DISP_BUCKET").as_deref() == Ok("1")
}

fn print_pipeline_profile(phase: &str) {
    if !pipeline_bucket_enabled() {
        return;
    }
    let entries = mlx_native::pipeline_dispatch_buckets();
    let total: u64 = entries.iter().map(|(_, count)| count).sum();
    eprintln!(
        "DeepSeek-V4 {phase} dispatch profile: {total} dispatches across {} pipelines",
        entries.len(),
    );
    for (label, count) in entries.into_iter().take(20) {
        eprintln!("DeepSeek-V4 {phase} dispatch: {label}: {count}");
    }
}

fn print_stage_profile(phase: &str, gpu_started: u64) {
    if !stage_profile_enabled() {
        return;
    }
    let entries = mlx_native::kernel_profile::dump();
    let mut categories = [
        ("FFN", 0_u64, 0_u64),
        ("uncompressed attention", 0, 0),
        ("compressed attention", 0, 0),
    ];
    let mut staged_ns = 0_u64;
    for (label, entry) in &entries {
        staged_ns = staged_ns.saturating_add(entry.total_ns);
        if let Some(category) = categories
            .iter_mut()
            .find(|(name, _, _)| label.contains(name))
        {
            category.1 = category.1.saturating_add(entry.total_ns);
            category.2 = category.2.saturating_add(entry.count);
        }
    }
    for (category, total_ns, count) in categories {
        eprintln!(
            "DeepSeek-V4 {phase} category: {category}: {:.3}ms / {count} calls",
            total_ns as f64 / 1e6,
        );
    }
    let gpu_ns = mlx_native::gpu_busy_ns().saturating_sub(gpu_started);
    eprintln!(
        "DeepSeek-V4 {phase} category: outside verifier stages: {:.3}ms",
        gpu_ns.saturating_sub(staged_ns) as f64 / 1e6,
    );
    for (label, entry) in entries.into_iter().take(12) {
        eprintln!(
            "DeepSeek-V4 {phase} stage: {label}: {:.3}ms / {} calls",
            entry.total_ns as f64 / 1e6,
            entry.count,
        );
    }
}
