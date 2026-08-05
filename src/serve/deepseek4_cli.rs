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
use crate::inference::models::deepseek4::{tokenizer, Deepseek4Model};

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
        !args.benchmark,
        "DeepSeek-V4 benchmark mode requires the pending optimized prefill path"
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

    let prefill_started = Instant::now();
    let mut final_state = None;
    for &token in prompt_tokens {
        final_state = Some(
            model
                .forward_verifier_one(token, &mut cache)
                .with_context(|| {
                    format!("DeepSeek-V4 prompt token at position {}", cache.position())
                })?,
        );
    }
    let prefill_elapsed = prefill_started.elapsed();
    eprintln!(
        "DeepSeek-V4 prefill: {} tokens in {:.2}s ({:.3} tok/s)",
        prompt_tokens.len(),
        prefill_elapsed.as_secs_f64(),
        prompt_tokens.len() as f64 / prefill_elapsed.as_secs_f64()
    );
    if args.max_tokens == 0 {
        return Ok(());
    }

    let decode_started = Instant::now();
    let mut logits = model
        .forward_logits(
            final_state
                .as_ref()
                .expect("nonempty prompt has final state"),
        )
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
    Ok(())
}
