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
#[allow(dead_code)]
pub mod kv_persist;
#[allow(dead_code)]
pub mod load_info;
#[allow(dead_code)]
pub mod multi_model;
pub mod parity_quality;
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

    let device =
        MlxDevice::new().map_err(|e| anyhow::anyhow!("registry warmup: MlxDevice::new: {e}"))?;
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
        let s: &mut [u32] =
            std::slice::from_raw_parts_mut(ids_buf.contents_ptr() as *mut u32, seq_len as usize);
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
///
/// Iter-215 Wedge-2: visibility raised to `pub(crate)` so the new
/// `serve::api::engine_qwen35::Qwen35LoadedModel::load` constructor
/// can reuse the same tokenizer-resolution logic `cmd_generate_qwen35`
/// uses (parity with the working CLI chat path).
pub(crate) fn find_tokenizer(model_path: &Path, explicit: Option<&Path>) -> Result<std::path::PathBuf> {
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
    anyhow::bail!("Cannot find config.json. Use --config to specify the path explicitly.")
}

/// Resolve the prompt text from either `--prompt` or `--prompt-file`.
fn resolve_prompt(args: &cli::GenerateArgs) -> Result<String> {
    match (&args.prompt, &args.prompt_file) {
        (Some(text), _) => Ok(text.clone()),
        (None, Some(path)) => {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read prompt file: {}", path.display()))?;
            let trimmed = content.trim().to_string();
            anyhow::ensure!(
                !trimmed.is_empty(),
                "Prompt file is empty: {}",
                path.display()
            );
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
///
/// **CLI `generate` path only.** This template uses a literal `{{PROMPT}}`
/// placeholder consumed by `String::replace` (see `render_chat_template`
/// below). It is NOT compatible with the API path's minijinja rendering
/// (which iterates a `messages` array). The API path uses
/// [`FALLBACK_GEMMA4_API_CHAT_TEMPLATE`] instead.
///
/// **iter-219b parity-gate fix (2026-05-01):** sibling fix to iter-217's
/// API-path correction (commit `5a1b999`). The pre-fix version had a
/// `<|turn>system\n<|think|><turn|>\n` system block that activates
/// Gemma 4's thinking-mode — the model emits a full `<|channel>thought\n
/// ...<channel|>` reasoning block before any answer content. That
/// triggered the parity-gate divergence the iter-219b release-check
/// caught: llama.cpp reference says `"The answer to 2 + 2 is **4**."`
/// while hf2q produced `"<|channel>thought\n* Question: \"Hello, what
/// i..."`. Common-prefix bytes = 0; all 6 parity checks failed (Gates
/// C/D/E + F).
///
/// The fix mirrors `FALLBACK_GEMMA4_API_CHAT_TEMPLATE` (line 271+):
/// drop the `<|think|>` system marker and append an empty channel block
/// `<|channel>thought\n<channel|>` after `<|turn>model\n`. The empty
/// block closes the channel before content begins, so the model emits
/// the answer directly (matching llama.cpp's behavior with no thinking
/// hint).
///
/// Probe (built binary, daily-driver Gemma 4 GGUF):
///   pre-fix: hf2q output begins `<|channel>thought\n* Question: ...`
///            (168 bytes; common-prefix=0 vs llama ref).
///   post-fix: hf2q output begins `The answer to 2 + 2 is **4**.`
///            (matches llama ref through 29-byte threshold).
pub(crate) const FALLBACK_GEMMA4_CHAT_TEMPLATE: &str =
    "<bos><|turn>user\n{{PROMPT}}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";

/// Hardcoded fallback chat template for the **API path** (consumed by
/// `src/serve/api/engine.rs::render_chat_prompt_with_tools` via minijinja).
///
/// Iterates the `messages` array so `{{m.content}}` actually renders the
/// caller's user content. The CLI fallback (`FALLBACK_GEMMA4_CHAT_TEMPLATE`)
/// uses a literal `{{PROMPT}}` placeholder consumed by `String::replace`,
/// which minijinja treats as an undefined variable — multi-thousand-token
/// user prompts collapsed to ~14 tokens of boilerplate when the GGUF
/// shipped no embedded template (observed in ADR-017 Phase A0.2b matrix
/// run, all SwapBackInSameCtx cells; the chat-template miss is the root
/// cause of the flat no_cache_ttft sweep that broke ship-gate ratios).
///
/// The role mapping mirrors `render_chat_prompt_with_tools`'s
/// `assistant→model` remap (`<|turn>model` is the assistant marker for
/// Gemma 4); the legacy CLI template embeds a system + `<|think|>`
/// preamble which is tokenization-equivalent across single-turn rounds.
///
/// **iter-217 fix (2026-04-30):** the assistant prefix now appends an
/// **empty channel block** `<|channel>thought\n<channel|>` after
/// `<|turn>model\n`, mirroring the upstream Gemma 4 chat template's
/// behavior when `enable_thinking=false` (the default; see
/// `vllm/examples/tool_chat_template_gemma4.jinja:326-330`). Without this
/// empty-block prime, the model emits a stray `<channel|>` close marker
/// in its first decoded tokens (training expects the channel to be
/// closed before content begins). The `ReasoningSplitter` requires both
/// open + close in output to extract reasoning, so a lone close marker
/// leaked verbatim into `delta.content`. Reproducer pre-fix:
/// `curl /v1/chat/completions {"messages":[{"role":"user","content":"What is the capital of France? Answer in English."}]}`
/// → `"content":"<channel|>The capital of France is **Paris**."`.
/// Post-fix: `"content":"The capital of France is **Paris**."` (close
/// marker absorbed by the prompt-side empty-block).
pub(crate) const FALLBACK_GEMMA4_API_CHAT_TEMPLATE: &str = concat!(
    "<bos>",
    "{%- for m in messages -%}",
    "<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n",
    "{%- endfor -%}",
    "<|turn>model\n",
    "<|channel>thought\n<channel|>",
);

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
        tracing::info!(
            "Chat template: loading from --chat-template-file {}",
            path.display()
        );
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
                let prefix = args.first().and_then(|v| v.as_str()).unwrap_or("");
                Ok(minijinja::Value::from(s.starts_with(prefix)))
            }
            "endswith" => {
                let suffix = args.first().and_then(|v| v.as_str()).unwrap_or("");
                Ok(minijinja::Value::from(s.ends_with(suffix)))
            }
            other => Err(
                minijinja::Error::from(minijinja::ErrorKind::UnknownMethod).with_source(
                    std::io::Error::other(format!("string has no method named {other}")),
                ),
            ),
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
    anyhow::ensure!(
        model_path.exists(),
        "Model not found: {}",
        model_path.display()
    );

    // --- Architecture detection (fast: metadata-only GGUF open) ---
    {
        let gguf_peek = mlx_native::gguf::GgufFile::open(model_path)
            .map_err(|e| anyhow::anyhow!("GGUF open (arch peek): {e}"))?;
        if let Some(arch) = gguf_peek.metadata_string("general.architecture") {
            use crate::inference::models::qwen35::{is_qwen36_gguf, ARCH_QWEN35, ARCH_QWEN35MOE};
            if arch == ARCH_QWEN35 || arch == ARCH_QWEN35MOE {
                // Wave 5a (ADR-005 Phase 4 ACs 5468/5470 partial): the
                // Qwen3.5 forward path is autoregressive-only (per-token
                // DeltaNet state update; correct at short prefill, slow at
                // long prefill until the W-5b chunk-scan kernel lands).
                // Qwen3.6 GGUFs reuse the same `general.architecture`
                // strings as Qwen3.5, so we must distinguish here via
                // `general.name` and require explicit opt-in to avoid
                // silently shipping a slow long-prefill path.
                if is_qwen36_gguf(&gguf_peek) && !INVESTIGATION_ENV.qwen36_autoreg {
                    anyhow::bail!(
                        "Qwen3.6 GGUF detected (general.name contains 'qwen3.6'), but \
                         autoregressive forward-path support is opt-in. Set \
                         HF2Q_QWEN36_AUTOREG=1 to dispatch through the existing \
                         autoregressive Qwen3.5 path (correct at short prefill; long-prefill \
                         SOTA via chunk-scan kernel deferred to Wave 5b). Model: {}",
                        model_path.display(),
                    );
                }
                tracing::info!("Detected architecture '{}' → routing to Qwen3.5 path", arch);
                return cmd_generate_qwen35(args, gguf_peek);
            }
        }
    }

    // ADR-018 C3 — unified load path.
    //
    // cmd_generate previously did manual GGUF-open + weights-load +
    // tokenizer-load (~70 LOC of duplicated logic against
    // `GemmaLoadedModel::load`). Refactor: route through the
    // `*LoadedModel::load` constructor so CLI and SERVE share one load
    // surface. Behaviour-equivalent: the constructor opens the GGUF,
    // parses config, builds GpuContext, resolves tokenizer + chat
    // template + EOS + provenance, and emits the same TTY-aware
    // `\r loading i/n layers` progress line via `header::LoadProgress`
    // (mirrors the previously-inline TTY+verbosity dance at mod.rs:519-531).
    //
    // After the load completes we re-open the GGUF (cheap mmap header
    // parse — microsecond-scale, no tensor reads) for the prompt-time
    // call sites that still need it: `render_chat_template`, BOS-token
    // detection, and `build_load_info`.
    let stderr_is_tty = std::io::IsTerminal::is_terminal(&std::io::stderr());
    let stdout_is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());

    let load_opts = api::engine::LoadOptions {
        model_path: model_path.clone(),
        tokenizer_path: args.tokenizer.clone(),
        config_path: args.config.clone(),
    };
    let load_start = std::time::Instant::now();
    let loaded = api::engine::GemmaLoadedModel::load(&load_opts)
        .context("GemmaLoadedModel::load")?;
    let load_elapsed = load_start.elapsed();

    // ADR-018 C3: emit the unified 13-line load banner on stdout (dim
    // on TTY) BEFORE any prompt rendering or prefill begins, preserving
    // the legacy ordering `print_header_top → render_prompt → prefill →
    // print_header_prefill → decode`. The banner replaces the old 2-line
    // `print_header_top` shape; its content is sourced from the
    // `LoadInfo` snapshot built by `loaded.build_load_info`.
    //
    // Re-open the GGUF for `build_load_info` (which needs the metadata
    // for arch_str + bpw + bos_token_id + chat_template_source) and for
    // the downstream `render_chat_template` + BOS detection. The GGUF
    // header parse is mmap-backed and the OS page cache is already warm
    // from the load above.
    let gguf = mlx_native::gguf::GgufFile::open(model_path)
        .map_err(|e| anyhow::anyhow!("GGUF re-open (post-load, banner+prompt): {e}"))?;

    let info = <api::engine::GemmaLoadedModel as load_info::LoadInfoBuilder>::build_load_info(
        &loaded, &gguf, load_elapsed, None, false,
    );
    load_info::emit_tracing(&info);
    let mut stdout = std::io::stdout();
    load_info::print_banner(&info, &mut stdout, stdout_is_tty)
        .context("print load banner")?;

    // Partial-move the loaded artifacts into local mutable bindings so
    // the rest of cmd_generate (prefill + decode + profiler) keeps the
    // same `&mut ctx` syntax it had pre-refactor. Rust permits this
    // because `GemmaLoadedModel` has no manual `Drop` impl: each field
    // moves independently. After this point `loaded` is no longer
    // usable as a whole — that's intentional, because none of the
    // remaining cmd_generate body wants the aggregate.
    let mut ctx = loaded.ctx;
    let mut mlx_w = loaded.weights;
    let tokenizer = loaded.tokenizer;

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

    let encoding = tokenizer
        .encode(prompt_text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    // ADR-015 iter42 — Gemma forward_mlx prefill coherence fix.
    //
    // ROOT CAUSE: hf2q's `tokenizer.encode(text, add_special_tokens=false)` does
    // not honour GGUF's `tokenizer.ggml.add_bos_token` flag.  llama.cpp's
    // `common_tokenize` *does* honour it (see ggml-llama.cpp `common.cpp`
    // → `llama_vocab::tokenize` with `add_special=true`); for Gemma4 it
    // additionally force-overrides the flag to `true` regardless of metadata
    // (`load: override 'tokenizer.ggml.add_bos_token' to 'true' for Gemma4`),
    // because Gemma4 weight matrices were trained with `<bos>` always present
    // at sequence start and produce gibberish without it (deterministic
    // single-pattern repetition like `the, my name is the, my name is …`).
    //
    // The bug was hidden through iter41 because:
    //   * The Gemma chat template's first directive is `{{ bos_token }}` —
    //     when the default GGUF chat template is rendered, BOS already lands
    //     in the prompt as literal `<bos>` and the tokenizer maps it to
    //     `bos_token_id`.  Any default-CLI invocation (e.g. iter34's bench)
    //     produced coherent output.
    //   * iter41's `tests/coherence_smoke.rs` passes `--chat-template
    //     "{% for message in messages %}{{ message.content }}{% endfor %}"`
    //     to test raw-prompt coherence — this strips the GGUF chat template
    //     and `bos_token` is never emitted.  Without BOS, Gemma decode
    //     produces gibberish; the test correctly fails.
    //
    // FIX: mirror llama.cpp's `common_tokenize` semantics.  When the GGUF
    // declares `tokenizer.ggml.add_bos_token=true` AND the first token of
    // the rendered prompt is not already `bos_token_id`, prepend the BOS
    // token.  This makes raw-prompt coherence_smoke produce the same token
    // stream that llama-completion produced when capturing the goldens.
    //
    // Why qwen35 paths don't need this: cmd_generate_qwen35 (the Qwen3.5/3.6
    // generate path) is structurally separate; this fix lives in the gemma
    // forward_mlx path only.  Verified at HEAD that qwen35 fixtures produce
    // coherent output without BOS — adding BOS for qwen35 would shift its
    // golden token stream without coherence benefit.
    let add_bos = matches!(
        gguf.metadata("tokenizer.ggml.add_bos_token"),
        Some(mlx_native::gguf::MetadataValue::Bool(true))
    );
    let bos_token_id = gguf.metadata_u32("tokenizer.ggml.bos_token_id");
    if add_bos {
        if let Some(bos) = bos_token_id {
            let already_has_bos = prompt_tokens.first() == Some(&bos);
            if !already_has_bos {
                prompt_tokens.insert(0, bos);
                tracing::info!(
                    "Prepended BOS token {} (GGUF tokenizer.ggml.add_bos_token=true)",
                    bos
                );
            }
        }
    }
    let prompt_tokens = prompt_tokens; // freeze
    tracing::info!("Prompt: {} tokens", prompt_tokens.len());
    if INVESTIGATION_ENV.dump_prompt_tokens {
        eprintln!(
            "HF2Q_DUMP_PROMPT_TOKENS: first10={:?} last10={:?} total={}",
            &prompt_tokens[..prompt_tokens.len().min(10)],
            &prompt_tokens[prompt_tokens.len().saturating_sub(10)..],
            prompt_tokens.len()
        );
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
        &header::HeaderInfoPrefill {
            prefill_n,
            prefill_ms,
            prefill_tok_s,
        },
        stdout_is_tty,
    )
    .context("print header prefill")?;

    // Decode
    //
    // 2026-05-02: mirror cmd_generate_qwen35 cumulative-decode pattern so
    // multi-byte UTF-8 codepoints (emoji, CJK) don't get split at token
    // boundaries → garble like `���`. Also reuse the n-gram repetition
    // guard so Gemma's greedy decode can't run to --max-tokens on a stuck
    // loop. Special-token-string check skipped here — Gemma's marker set
    // (`<end_of_turn>`, `<start_of_turn>`, `<eos>`, `<bos>`) differs from
    // Qwen's; integer-id stop via eos_token_ids.contains() catches `<end_of_turn>`
    // (token id 106) which is the dominant turn-end marker.
    let mut all_tokens = prompt_tokens.to_vec();
    let mut next_token = last_token;
    all_tokens.push(next_token);
    let mut decoded_tokens: Vec<u32> = vec![next_token];
    let mut printed_text = tokenizer
        .decode(&decoded_tokens, false)
        .unwrap_or_default();
    print!("{}", printed_text);
    std::io::stdout().flush()?;

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

        let kernel_profile_break = if kernel_profile_mode {
            let (tok, kp) = mlx_w.forward_decode_kernel_profile(next_token, pos, &mut ctx)?;
            next_token = tok;
            if generated > kernel_profile_warmup {
                kernel_profiles.push(kp);
            }
            kernel_profiles.len() >= kernel_profile_measure
        } else {
            let mut p = profiler.start_token();
            next_token = mlx_w.forward_decode(next_token, pos, &mut ctx, &mut p)?;
            profiler.finish_token(p);
            false
        };
        all_tokens.push(next_token);
        generated += 1;
        decoded_tokens.push(next_token);

        // Cumulative decode + delta print (llama.cpp tok_str_pos pattern).
        let new_full = tokenizer
            .decode(&decoded_tokens, false)
            .unwrap_or_default();
        if new_full.len() > printed_text.len() && new_full.starts_with(&printed_text) {
            print!("{}", &new_full[printed_text.len()..]);
            std::io::stdout().flush()?;
        }
        printed_text = new_full;

        if kernel_profile_break {
            break;
        }

        // N-gram repetition guard — same shared detector as the qwen35 path.
        if let Some((ngram, repeats)) = detect_greedy_repetition_loop(&decoded_tokens) {
            tracing::info!(
                "Gemma decode: greedy n-gram repetition detected (last {} tokens \
                 repeated {} times); stopping. Sampling \
                 (temperature/top_k/top_p/repetition_penalty) is wired in \
                 forward_decode but the CLI does not pass them through; \
                 use the chat-completion API for non-deterministic decoding.",
                ngram, repeats
            );
            eprintln!(
                "\n[hf2q] Gemma greedy decode entered a {}-token repetition loop \
                 — stopping. Use the chat-completion API for non-deterministic \
                 decoding.",
                ngram
            );
            break;
        }
    }
    let decode_elapsed = decode_start.elapsed();
    let tok_per_sec = generated as f64 / decode_elapsed.as_secs_f64();
    let (td, tr) = if stderr_is_tty {
        ("\x1b[2m", "\x1b[0m")
    } else {
        ("", "")
    };
    eprintln!(
        "\n\n{td}--- mlx-native: {} tokens in {:.2}s ({:.1} tok/s) ---{tr}",
        generated,
        decode_elapsed.as_secs_f64(),
        tok_per_sec,
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
        let dispatches_per_prompt_tok = if prompt_n > 0 {
            dispatches as f64 / prompt_n as f64
        } else {
            0.0
        };
        let syncs_per_decode_tok = if decode_n > 0 {
            syncs as f64 / decode_n as f64
        } else {
            0.0
        };
        eprintln!(
            "[MLX_COUNTERS] dispatches={} syncs={} prompt_tokens={} decode_tokens={} \
             dispatches_per_prompt_tok={:.2} syncs_per_decode_tok={:.2}",
            dispatches, syncs, prompt_n, decode_n, dispatches_per_prompt_tok, syncs_per_decode_tok,
        );
    }

    Ok(())
}

// ================================================================
// Qwen3.5 generate path (ADR-013 P13.1)
// ================================================================

const QWEN35_PREFILL_SWEEP_TEXT: &str =
    "The benchmark prompt describes a careful engineering investigation. \
     It repeats neutral facts about measuring model speed, preserving token \
     probabilities, checking coherence, and comparing current code against \
     peer implementations. ";

fn qwen35_sweep_token_count(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<usize> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("qwen35 prefill sweep tokenization failed: {e}"))?;
    Ok(encoding.get_ids().len())
}

fn qwen35_sweep_prompt(
    tokenizer: &tokenizers::Tokenizer,
    target_tokens: usize,
) -> Result<(String, Vec<u32>)> {
    let repeated = QWEN35_PREFILL_SWEEP_TEXT.repeat((target_tokens / 20) + 500);
    let ids = tokenizer
        .encode(repeated.as_str(), false)
        .map_err(|e| anyhow::anyhow!("qwen35 prefill sweep base tokenization failed: {e}"))?
        .get_ids()
        .to_vec();
    anyhow::ensure!(
        ids.len() >= target_tokens,
        "qwen35 prefill sweep base prompt produced only {} tokens for target {}",
        ids.len(),
        target_tokens,
    );

    let mut best_text = tokenizer
        .decode(&ids[..target_tokens], false)
        .map_err(|e| anyhow::anyhow!("qwen35 prefill sweep decode failed: {e}"))?;
    let mut best_n = qwen35_sweep_token_count(tokenizer, &best_text)?;
    if best_n == target_tokens {
        let enc = tokenizer
            .encode(best_text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("qwen35 prefill sweep final tokenization failed: {e}"))?;
        return Ok((best_text, enc.get_ids().to_vec()));
    }

    let lo = target_tokens.saturating_sub(256).max(1);
    let hi = (target_tokens + 256).min(ids.len());
    for n_ids in lo..=hi {
        let text = tokenizer
            .decode(&ids[..n_ids], false)
            .map_err(|e| anyhow::anyhow!("qwen35 prefill sweep decode failed: {e}"))?;
        let n = qwen35_sweep_token_count(tokenizer, &text)?;
        if n == target_tokens {
            let enc = tokenizer.encode(text.as_str(), false).map_err(|e| {
                anyhow::anyhow!("qwen35 prefill sweep final tokenization failed: {e}")
            })?;
            return Ok((text, enc.get_ids().to_vec()));
        }
        if n.abs_diff(target_tokens) < best_n.abs_diff(target_tokens) {
            best_text = text;
            best_n = n;
        }
    }

    let enc = tokenizer
        .encode(best_text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("qwen35 prefill sweep final tokenization failed: {e}"))?;
    Ok((best_text, enc.get_ids().to_vec()))
}

fn qwen35_positions(seq_len: usize) -> Vec<i32> {
    let mut flat = vec![0i32; 4 * seq_len];
    for axis in 0..4 {
        for t in 0..seq_len {
            flat[axis * seq_len + t] = t as i32;
        }
    }
    flat
}

fn maybe_run_qwen35_prefill_sweep(
    model: &crate::inference::models::qwen35::model::Qwen35Model,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<bool> {
    let Ok(lengths_raw) = std::env::var("HF2Q_QWEN35_PREFILL_SWEEP") else {
        return Ok(false);
    };
    let lengths: Vec<usize> = lengths_raw
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim().parse::<usize>().with_context(|| {
                format!("HF2Q_QWEN35_PREFILL_SWEEP contains non-integer length {s:?}")
            })
        })
        .collect::<Result<_>>()?;
    anyhow::ensure!(
        !lengths.is_empty(),
        "HF2Q_QWEN35_PREFILL_SWEEP must contain at least one length"
    );
    let trials = std::env::var("HF2Q_QWEN35_PREFILL_SWEEP_TRIALS")
        .ok()
        .map(|s| {
            s.parse::<usize>()
                .context("parse HF2Q_QWEN35_PREFILL_SWEEP_TRIALS")
        })
        .transpose()?
        .unwrap_or(3)
        .max(1);
    let warmups = std::env::var("HF2Q_QWEN35_PREFILL_SWEEP_WARMUPS")
        .ok()
        .map(|s| {
            s.parse::<usize>()
                .context("parse HF2Q_QWEN35_PREFILL_SWEEP_WARMUPS")
        })
        .transpose()?
        .unwrap_or(1);

    use crate::inference::models::qwen35::io_heads::greedy_argmax_last_token;
    use crate::inference::models::qwen35::kv_cache::HybridKvCache;
    use mlx_native::MlxDevice;

    fn top_n_indices(values: &[f32], n: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        indexed.into_iter().take(n).map(|(idx, _)| idx).collect()
    }

    fn logit_row_metrics(a: &[f32], b: &[f32]) -> (f32, f64, usize) {
        let mut max_abs = 0.0f32;
        let mut dot = 0.0f64;
        let mut a2 = 0.0f64;
        let mut b2 = 0.0f64;
        for (&av, &bv) in a.iter().zip(b) {
            max_abs = max_abs.max((av - bv).abs());
            let af = av as f64;
            let bf = bv as f64;
            dot += af * bf;
            a2 += af * af;
            b2 += bf * bf;
        }
        let cosine = if a2 > 0.0 && b2 > 0.0 {
            dot / (a2.sqrt() * b2.sqrt())
        } else {
            f64::NAN
        };
        let top_a = top_n_indices(a, 10);
        let top_b = top_n_indices(b, 10);
        let overlap = top_a.iter().filter(|idx| top_b.contains(idx)).count();
        (max_abs, cosine, overlap)
    }

    let device = MlxDevice::new()
        .map_err(|e| anyhow::anyhow!("qwen35 prefill sweep MlxDevice::new: {e}"))?;
    println!(
        "{{\"event\":\"qwen35_prefill_sweep_start\",\"lengths\":{:?},\"warmups\":{},\"trials\":{}}}",
        lengths, warmups, trials
    );

    for target in lengths {
        let (_prompt, prompt_tokens) = qwen35_sweep_prompt(tokenizer, target)?;
        let prompt_len = prompt_tokens.len();
        let positions = qwen35_positions(prompt_len);
        let max_seq = (prompt_len + 65)
            .max(128)
            .min(model.cfg.max_position_embeddings as usize);

        for iteration in 0..(warmups + trials) {
            let phase = if iteration < warmups {
                "warmup"
            } else {
                "measure"
            };
            let trial = if phase == "warmup" {
                iteration
            } else {
                iteration - warmups
            };
            let mut kv_cache = HybridKvCache::new(&model.cfg, &device, max_seq as u32, 1)
                .context("qwen35 prefill sweep HybridKvCache::new")?;
            let t0 = std::time::Instant::now();
            let full_logits =
                std::env::var("HF2Q_QWEN35_PREFILL_SWEEP_FULL_LOGITS").as_deref() == Ok("1");
            let compare_full_last =
                std::env::var("HF2Q_QWEN35_PREFILL_SWEEP_COMPARE_FULL_LAST").as_deref() == Ok("1");
            let logits = if compare_full_last {
                let mut full_kv = HybridKvCache::new(&model.cfg, &device, max_seq as u32, 1)
                    .context("qwen35 prefill sweep compare full HybridKvCache::new")?;
                let full = model
                    .forward_gpu(&prompt_tokens, &positions, &mut full_kv)
                    .context("qwen35 prefill sweep compare forward_gpu")?;
                let mut last_kv = HybridKvCache::new(&model.cfg, &device, max_seq as u32, 1)
                    .context("qwen35 prefill sweep compare last HybridKvCache::new")?;
                let last = model
                    .forward_gpu_last_logits(&prompt_tokens, &positions, &mut last_kv)
                    .context("qwen35 prefill sweep compare forward_gpu_last_logits")?;
                let vocab = model.cfg.vocab_size as usize;
                anyhow::ensure!(
                    full.len() == prompt_len * vocab,
                    "qwen35 prefill sweep compare full logits.len()={} != expected {}",
                    full.len(),
                    prompt_len * vocab,
                );
                anyhow::ensure!(
                    last.len() == vocab,
                    "qwen35 prefill sweep compare last logits.len()={} != expected {}",
                    last.len(),
                    vocab,
                );
                let full_last = &full[full.len() - vocab..];
                let full_token = greedy_argmax_last_token(full_last, model.cfg.vocab_size);
                let last_token = greedy_argmax_last_token(&last, model.cfg.vocab_size);
                let (max_abs, cosine, top10_overlap) = logit_row_metrics(full_last, &last);
                println!(
                    "{{\"event\":\"qwen35_prefill_sweep_compare\",\"target_tokens\":{},\"actual_tokens\":{},\"phase\":\"{}\",\"iteration\":{},\"trial\":{},\"full_token\":{},\"last_token\":{},\"max_abs\":{:.9},\"cosine\":{:.12},\"top10_overlap\":{}}}",
                    target,
                    prompt_len,
                    phase,
                    iteration,
                    trial,
                    full_token,
                    last_token,
                    max_abs,
                    cosine,
                    top10_overlap,
                );
                last
            } else if full_logits {
                model
                    .forward_gpu(&prompt_tokens, &positions, &mut kv_cache)
                    .context("qwen35 prefill sweep forward_gpu")?
            } else {
                model
                    .forward_gpu_last_logits(&prompt_tokens, &positions, &mut kv_cache)
                    .context("qwen35 prefill sweep forward_gpu_last_logits")?
            };
            let elapsed = t0.elapsed();
            let vocab = model.cfg.vocab_size as usize;
            let expected_logits = if full_logits && !compare_full_last {
                prompt_len * vocab
            } else {
                vocab
            };
            anyhow::ensure!(
                logits.len() == expected_logits,
                "qwen35 prefill sweep logits.len()={} != expected {} (full_logits={})",
                logits.len(),
                expected_logits,
                full_logits,
            );
            let last_logits = &logits[logits.len() - vocab..];
            let first_token = greedy_argmax_last_token(last_logits, model.cfg.vocab_size);
            let ms = elapsed.as_secs_f64() * 1000.0;
            let tps = prompt_len as f64 / elapsed.as_secs_f64();
            println!(
                "{{\"event\":\"qwen35_prefill_sweep\",\"target_tokens\":{},\"actual_tokens\":{},\"phase\":\"{}\",\"iteration\":{},\"trial\":{},\"prefill_ms\":{:.3},\"prefill_tps\":{:.3},\"first_token\":{},\"output_head\":\"{}\"}}",
                target,
                prompt_len,
                phase,
                iteration,
                trial,
                ms,
                tps,
                first_token,
                if full_logits { "all" } else { "last" },
            );
        }
    }
    println!("{{\"event\":\"qwen35_prefill_sweep_end\"}}");
    Ok(true)
}

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
/// Detect a greedy-decode repetition loop in `decoded_tokens`.
///
/// Returns `Some((ngram_size, occurrences))` if the last 128 tokens contain
/// any 8/12/16/20/24-token sequence that appears ≥4 times (not necessarily
/// consecutively — greedy loops can have minor drift, e.g., one outlier
/// token mid-cycle).
///
/// Why this is a real fix and not a fallback (per mantra `feedback_no_shortcuts`):
/// pure greedy on thinking-style Qwen3.x checkpoints can enter a deterministic
/// loop with no exit (verified 2026-05-02 user report on French-Toast prompt:
/// "I should make sure the response is accurate and up-to-date." × 30+ before
/// `<|im_end|>` finally fired at token 1014). Without sampling
/// (temperature/top_k/top_p/repetition_penalty — TODO follow-up to extend the
/// qwen35 generate path), running to `--max-tokens` produces 1000+ tokens of
/// garbage. Stopping on detected repetition with a diagnostic stderr message
/// IS the correct behavior for greedy mode — it's not a workaround for
/// missing sampling, it's the deterministic-mode escape hatch.
///
/// Algorithm: take the LAST `ngram_size` tokens as a key, scan the 128-token
/// window for non-overlapping occurrences of that key. If ≥4 occurrences are
/// found at any of the candidate sizes, report the first match. The
/// non-overlapping increment (`i += ngram` on hit) avoids over-counting at
/// short n-gram sizes (e.g. all-same-token would report `ngram=8` with
/// `occurrences=16` from a 128-token window).
fn detect_greedy_repetition_loop(decoded_tokens: &[u32]) -> Option<(usize, usize)> {
    const REPEAT_NGRAM_SIZES: &[usize] = &[8, 12, 16, 20, 24];
    const REPEAT_SCAN_WINDOW: usize = 128;
    const REPEAT_MIN_OCCURRENCES: usize = 4;

    if decoded_tokens.len() < REPEAT_SCAN_WINDOW {
        return None;
    }
    let n = decoded_tokens.len();
    let window = &decoded_tokens[n - REPEAT_SCAN_WINDOW..];
    for &ngram in REPEAT_NGRAM_SIZES {
        if window.len() < ngram * REPEAT_MIN_OCCURRENCES {
            continue;
        }
        let key = &window[window.len() - ngram..];
        let mut occurrences = 0usize;
        let mut i = 0;
        while i + ngram <= window.len() {
            if &window[i..i + ngram] == key {
                occurrences += 1;
                i += ngram;
            } else {
                i += 1;
            }
        }
        if occurrences >= REPEAT_MIN_OCCURRENCES {
            return Some((ngram, occurrences));
        }
    }
    None
}

fn cmd_generate_qwen35(args: cli::GenerateArgs, gguf: mlx_native::gguf::GgufFile) -> Result<()> {
    use crate::inference::models::qwen35::io_heads::greedy_argmax_last_token;
    use crate::inference::models::qwen35::kv_cache::HybridKvCache;
    use crate::serve::api::engine_qwen35::Qwen35LoadedModel;
    use mlx_native::MlxDevice;
    use std::io::Write;

    let model_path = &args.model;

    // ADR-018 C3 — unified load path.
    //
    // cmd_generate_qwen35 previously did manual GGUF + tokenizer + EOS
    // resolution (~60 LOC of duplicated logic against
    // `Qwen35LoadedModel::load`). Refactor: route through the
    // `*LoadedModel::load` constructor so CLI and SERVE share one load
    // surface. Behaviour-equivalent: the constructor opens the GGUF
    // (re-open from the dispatcher's metadata-only handle is fine —
    // page cache is warm), loads weights via mlx-native, resolves
    // tokenizer + chat template + EOS + provenance, and emits the same
    // TTY-aware `\r loading i/n layers` progress line via
    // `header::LoadProgress`.
    //
    // The post-load cmd_generate_qwen35 body still uses the original
    // `gguf` handle (passed in by `cmd_generate`'s arch detect) for
    // chat-template render and EOS-from-metadata fallback — those are
    // mmap reads against the same on-disk file and don't need a second
    // open. `LoadInfo`'s build path takes its own `&GgufFile`; we pass
    // the parameter handle.
    let stdout_is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());

    let load_opts = api::engine::LoadOptions {
        model_path: model_path.clone(),
        tokenizer_path: args.tokenizer.clone(),
        config_path: args.config.clone(),
    };
    let load_start = std::time::Instant::now();
    let loaded = Qwen35LoadedModel::load(&load_opts).context("Qwen35LoadedModel::load")?;
    let load_elapsed = load_start.elapsed();

    // ADR-018 C3: emit the unified 13-line load banner BEFORE prompt
    // rendering / prefill. Order matches cmd_generate (Gemma) and the
    // legacy `print_header_top` site.
    let info = <Qwen35LoadedModel as load_info::LoadInfoBuilder>::build_load_info(
        &loaded, &gguf, load_elapsed, None, false,
    );
    load_info::emit_tracing(&info);
    let mut stdout = std::io::stdout();
    load_info::print_banner(&info, &mut stdout, stdout_is_tty)
        .context("print load banner")?;

    // Partial-move the loaded artifacts into local mutable bindings so
    // the rest of cmd_generate_qwen35 (prefill + decode) keeps the same
    // shape. `Qwen35LoadedModel` has no manual `Drop`; partial moves
    // are sound.
    let model = loaded.model;
    let tokenizer = loaded.tokenizer;
    // The legacy code path resolved EOS via `gguf.metadata_u32(...)
    // .unwrap_or(151645)` AND emitted a `tracing::info!("Qwen3.5 EOS
    // token id: {}", eos_token_id)`. The unified surface stores
    // `eos_token_ids: Vec<u32>` in the loaded struct (already lifted
    // from GGUF or defaulted) — read it directly. The trailing
    // `tracing::info!` is replaced by emit_tracing's `eos_token_ids`
    // structured field.
    // 2026-05-02: cmd_generate_qwen35 had a single-eos collapse here that let
    // `<|im_end|>` (151645) leak through whenever `<|endoftext|>` (151643) was
    // first in the Vec — caused decode loops to run past 1st turn-end (user
    // report: French Toast prompt produced ~30-line repetition + multiple
    // `<|im_end|>`/`<|im_start|>assistant` leaks). Use the full Vec.
    // The legacy `eos_token_id` single value is kept for the SpecDecode path
    // below which still takes `Option<u32>` (TODO: extend SpecDecode to take
    // a slice; tracked in a follow-up to avoid scope creep here).
    let eos_token_id: u32 = loaded
        .eos_token_ids
        .first()
        .copied()
        .unwrap_or(151_645);
    let eos_token_ids: Vec<u32> = if loaded.eos_token_ids.is_empty() {
        vec![151_645]
    } else {
        loaded.eos_token_ids.clone()
    };

    if maybe_run_qwen35_prefill_sweep(&model, &tokenizer)? {
        return Ok(());
    }

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

    // Gate-time tokenizer parity tooling. When `HF2Q_DEBUG_TOKENIZE_ONLY=1`,
    // print the encoded token IDs on stdout (one space-separated line
    // prefixed `TOKENIZE_DEBUG_IDS:`) and exit `Ok(())` before model
    // load. Used by `scripts/qwen35_tokenizer_parity.sh` to compare
    // against `llama-tokenize` on the same GGUF — the parity contract
    // for the GGUF-driven tokenizer per ADR-013 §"Sovereignty".
    if std::env::var("HF2Q_DEBUG_TOKENIZE_ONLY").as_deref() == Ok("1") {
        let id_str: Vec<String> = prompt_tokens.iter().map(|i| i.to_string()).collect();
        println!("TOKENIZE_DEBUG_IDS: {}", id_str.join(" "));
        return Ok(());
    }

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

    // ADR-018 C3: the legacy 2-line `print_header_top` site that lived
    // here is gone — the unified 13-line load banner already rendered
    // at the top of `cmd_generate_qwen35` (the byte-equivalent earlier
    // call to `load_info::print_banner`). The local variables that
    // backed the legacy header (`backend_chip`, `model_name`,
    // `total_gb`, `n_layers`, `header_top`) carried no value past the
    // print site and were deleted to keep the load-fact surface in one
    // place. `stdout` and `stdout_is_tty` were the only locals used
    // downstream — both are bound up top now.

    if use_spec_decode {
        use crate::inference::models::qwen35::spec_decode::SpecDecode;
        tracing::info!("Qwen3.5 speculative decode enabled");
        // ADR-013 P19 H12: warm GPU cache before SpecDecode's internal prefill
        // timer starts (matches the greedy branch). Idempotent if SpecDecode
        // already calls `ensure_gpu_cache_primed` internally.
        model
            .ensure_gpu_cache_primed()
            .context("Qwen35Model::ensure_gpu_cache_primed (P19 H12 spec-decode warmup)")?;
        let result = SpecDecode::run_with_eos(
            &model,
            &prompt_tokens,
            args.max_tokens,
            Some(eos_token_id),
            max_seq as u32,
        )
        .context("qwen35 SpecDecode::run_with_eos")?;

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
            "\n\n{td}--- mlx-native (qwen35 spec): {} tokens in {:.2}s ({:.1} tok/s, accept {:.1}%) ---{tr}",
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
    let device = MlxDevice::new().map_err(|e| anyhow::anyhow!("MlxDevice::new: {e}"))?;
    let mut kv_cache =
        HybridKvCache::new(&model.cfg, &device, max_seq as u32, 1).context("HybridKvCache::new")?;
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

    // ADR-013 P19 H12 (2026-05-01): warm the per-thread GPU cache (one-shot
    // ~17 GB Q4 weight materialization onto Metal heap + lm_head BF16/Q4_0
    // pre-quant + flash_attn_prefill kernel registration) BEFORE the prefill
    // timer starts. Without this, the ~984 ms upload was charged to
    // `prefill_tok_s`, producing a 28× apparent gap vs llama.cpp's
    // `prompt eval time` (which excludes model load by construction).
    // Compute is unchanged; only the timer-span moves. `wave5b8_profile`'s
    // `UploadWeights` section still tracks the cost — it just lives here now.
    let warmup_start = std::time::Instant::now();
    model
        .ensure_gpu_cache_primed()
        .context("Qwen35Model::ensure_gpu_cache_primed (P19 H12 warmup)")?;
    tracing::info!(
        "Qwen3.5 GPU warmup (P19 H12): {:.2}s",
        warmup_start.elapsed().as_secs_f64()
    );

    tracing::info!("Qwen3.5 prefill: seq_len={}", prompt_len);
    // ADR-013 P19 H9 (2026-05-01): empirical CB-sync attribution.  Reset
    // mlx-native's atomic counters before prefill, read after, and print
    // when HF2Q_PROFILE_SYNC=1 is set.  This is a measurement-only diff:
    // SYNC_COUNT counts commit_and_wait calls/prefill, DISPATCH_COUNT
    // counts kernel dispatches, BARRIER_COUNT counts memory_barriers.
    // The hypothesis ("hf2q does ~120-160 commit_and_wait/prefill while
    // llama.cpp does ~1") is TESTABLE by comparing these numbers against
    // ggml-metal's per-graph submit pattern.  Zero overhead when env
    // unset (RAII guard does no work).
    let profile_sync = std::env::var("HF2Q_PROFILE_SYNC").is_ok();
    if profile_sync {
        mlx_native::reset_counters();
    }
    let prefill_start = std::time::Instant::now();
    let prefill_logits = model
        .forward_gpu_last_logits(&prompt_tokens, &prefill_positions, &mut kv_cache)
        .context("Qwen35Model::forward_gpu_last_logits (prefill)")?;
    let prefill_elapsed = prefill_start.elapsed();
    if profile_sync {
        eprintln!(
            "[P19 H9] prefill seq_len={} elapsed_ms={:.1} sync_count={} dispatch_count={} barrier_count={} cmd_buf_count={}",
            prompt_len,
            prefill_elapsed.as_secs_f64() * 1000.0,
            mlx_native::sync_count(),
            mlx_native::dispatch_count(),
            mlx_native::barrier_count(),
            mlx_native::cmd_buf_count(),
        );
    }

    // Sanity-check logits shape.
    let vocab_size = model.cfg.vocab_size;
    anyhow::ensure!(
        prefill_logits.len() == vocab_size as usize,
        "forward_gpu_last_logits (prefill) returned logits.len()={} != vocab({})",
        prefill_logits.len(),
        vocab_size,
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
        let last_logits = &prefill_logits;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(last_logits.as_ptr() as *const u8, last_logits.len() * 4)
        };
        std::fs::write("/tmp/hf2q_logits_t0.bin", bytes)
            .context("HF2Q_DUMP_LOGITS: write /tmp/hf2q_logits_t0.bin")?;
        // Top-3 to stderr for quick sanity check.
        let mut indexed: Vec<(usize, f32)> = last_logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!(
            "HF2Q_DUMP_LOGITS: wrote {} f32 values to /tmp/hf2q_logits_t0.bin",
            last_logits.len()
        );
        eprintln!("  top-3: {:?}", &indexed[..3.min(indexed.len())]);
        return Ok(());
    }

    // Sample the first token from prefill logits (last token's row).
    let last_prefill_logits = &prefill_logits;
    let mut next_token = greedy_argmax_last_token(last_prefill_logits, vocab_size);
    tracing::info!("Qwen3.5 first decoded token: {}", next_token);

    // 2026-05-02: cumulative-decode + delta-print pattern (mirrors llama.cpp's
    // `tok_str_pos` discipline). Per-token `tokenizer.decode(&[t], ...)` breaks
    // multi-byte UTF-8 codepoints (emoji, CJK) at token boundaries → garble
    // like `���` in the user's French-Toast output. Decode the full so-far
    // sequence each step and print only the byte-delta. Costs O(generated)
    // bytes of allocation per step but produces correct UTF-8 every time.
    // Also catches special-token-string leaks (`<|im_end|>` etc. emitted as
    // text rather than as the eos_token_ids vocab id) — see is_special_token
    // check below.
    let mut decoded_tokens: Vec<u32> = vec![next_token];
    let mut printed_text = tokenizer
        .decode(&decoded_tokens, false)
        .unwrap_or_default();
    print!("{}", printed_text);
    stdout.flush()?;

    // ---- Decode loop ----
    let decode_start = std::time::Instant::now();
    let mut generated = 1usize;

    // Special-token strings that some Qwen3.x checkpoints emit as TEXT instead
    // of as their reserved vocab id (when the model has been distilled / merged
    // and the `<|im_end|>` token id 151645 isn't always emitted at turn-end).
    // If any of these substrings appear after a decode step, treat as stop.
    // This complements the `eos_token_ids.contains(&next_token)` integer check
    // — covers both cases without a fallback.
    // 2026-05-02 (follow-up): `<|im_start|>` REMOVED from this list. Qwen3
    // thinking-style checkpoints emit a literal `<|im_start|>` text fragment
    // mid-turn during the thinking → answer transition (right after the
    // `<|end|>` end-of-thinking marker), and stopping there cuts the response
    // before any actual answer is produced. The original French-Toast leak
    // sequence (`<|im_end|>...<|im_start|>assistant`) is still caught by the
    // `<|im_end|>` substring entry below — `<|im_start|>` was redundant
    // defense for that case and is hostile to thinking-mode output.
    const SPECIAL_TOKEN_STOPS: &[&str] = &[
        "<|im_end|>",
        "<|endoftext|>",
    ];

    for step in 1..args.max_tokens {
        if eos_token_ids.contains(&next_token) {
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
        } else {
            None
        };
        next_token = model
            .forward_gpu_greedy(&[next_token], &decode_positions, &mut kv_cache)
            .with_context(|| format!("forward_gpu_greedy decode step {step}"))?;
        if let Some(t) = _t_step {
            eprintln!(
                "[STEP_PROFILE] step={step} total={:.2}ms",
                t.elapsed().as_micros() as f64 / 1000.0
            );
        }
        generated += 1;
        decoded_tokens.push(next_token);

        // Cumulative decode + delta print. UTF-8 garble = solved.
        let new_full = tokenizer
            .decode(&decoded_tokens, false)
            .unwrap_or_default();
        if new_full.len() > printed_text.len() && new_full.starts_with(&printed_text) {
            print!("{}", &new_full[printed_text.len()..]);
            stdout.flush()?;
        }
        // Special-token-string leak: stop loud rather than emit garbage like
        // `<|im_end|>...<|im_start|>assistant\n<|im_start|>assistant` which the
        // user observed on their French-Toast run when `<|im_end|>` slipped
        // past the integer-id check (e.g. tokenizer emitted it as a multi-byte
        // text fragment composed of non-special token ids).
        if SPECIAL_TOKEN_STOPS.iter().any(|m| new_full.contains(m)) {
            tracing::info!(
                "Qwen3.5 decode: special-token string detected in cumulative \
                 decode at step {step}; stopping."
            );
            break;
        }
        // 2026-05-02: greedy-decode repetition-stop guard. Pure greedy on
        // thinking-style Qwen3.x checkpoints can enter a deterministic loop
        // (user's French-Toast run produced "I should make sure the response
        // is accurate and up-to-date." × ~30). Without sampling (temp/top_k/
        // top_p/repetition_penalty — TODO follow-up), running to --max-tokens
        // produces 1000+ tokens of garbage. Detect: if the last 32 tokens
        // contain a 16-token n-gram that repeats ≥3 times consecutively,
        // we're stuck — break with a diagnostic. ~24 LOC vs the proper
        // sampling fix (~80-100 LOC plumb through forward_gpu_last_logits +
        // CPU sampling). Both should land; this is the minimum to keep the
        // CLI honest. NOT a fallback — this is correct behavior on a stuck
        // greedy loop (mantra: "no fallback" means don't substitute a lesser
        // option for the right answer; here, breaking on detected repetition
        // IS the right answer for greedy-mode stuck loops).
        // Try multiple n-gram window sizes so we match cycles of various
        // lengths. The user's French-Toast repetition was a 17-token cycle
        // ("I should make sure the response is accurate and up-to-date.")
        // which a single fixed 16-token window misses. Iterate the typical
        // greedy-loop cycle range.
        let detected_repeat = detect_greedy_repetition_loop(&decoded_tokens);
        if let Some((ngram, repeats)) = detected_repeat {
            tracing::info!(
                "Qwen3.5 decode: greedy n-gram repetition detected at step \
                 {step} (last {} tokens repeated {} times); stopping. \
                 Sampling (temperature/top_k/top_p/repetition_penalty) is \
                 not yet wired in this CLI path — TODO follow-up.",
                ngram, repeats
            );
            eprintln!(
                "\n[hf2q] greedy decode entered a {}-token repetition loop \
                 at step {} — stopping. Use the chat-completion API \
                 (which has sampling wired) for non-deterministic decoding.",
                ngram, step
            );
            break;
        }
        printed_text = new_full;
    }

    let decode_elapsed = decode_start.elapsed();
    let tok_per_sec = generated as f64 / decode_elapsed.as_secs_f64();

    let (td, tr) = if std::io::IsTerminal::is_terminal(&std::io::stderr()) {
        ("\x1b[2m", "\x1b[0m")
    } else {
        ("", "")
    };
    eprintln!(
        "\n\n{td}--- mlx-native (qwen35): {} tokens in {:.2}s ({:.1} tok/s) ---{tr}",
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

pub fn load_engine(path: &Path, config: &multi_model::EngineConfig) -> Result<api::engine::Engine> {
    anyhow::ensure!(path.exists(), "Model not found: {}", path.display());
    // Header-only parse surfaces bad magic + populates the diagnostic
    // log line.  Cheap (memory-mapped header, no tensor read).
    //
    // ADR-005 Phase 4 reopen iter-215 Wedge-2: the pre-iter-215
    // arch-detect bail for qwen35 / qwen35moe (commit 064797e) is
    // replaced by actual SERVE-side dispatch.  `LoadedModel::load`
    // (api/engine.rs) reads `general.architecture` and routes to
    // `Qwen35LoadedModel::load` for Qwen3.5/3.6 or
    // `GemmaLoadedModel::load` otherwise; the worker thread arm for
    // the Qwen35 variant returns HTTP 501 with an operator-actionable
    // message (Wedge-3 deferred follow-up wires the live forward
    // pass).  See `LoadedModel::load` for the dispatch surface.
    {
        let gguf = mlx_native::gguf::GgufFile::open(path)
            .map_err(|e| anyhow::anyhow!("GGUF header parse failed: {e}"))?;
        let arch = gguf
            .metadata_string("general.architecture")
            .map(|s| s.to_string())
            .unwrap_or_default();
        tracing::info!(
            path = %path.display(),
            tensors = gguf.tensor_count(),
            metadata = gguf.metadata_count(),
            arch = %arch,
            "Validated GGUF header"
        );
    }

    let load_opts = api::engine::LoadOptions {
        model_path: path.to_path_buf(),
        tokenizer_path: config.tokenizer_path.clone(),
        config_path: config.config_path.clone(),
    };
    let loaded = api::engine::LoadedModel::load(&load_opts)?;
    let engine = api::engine::Engine::spawn(loaded, config.queue_capacity, None);
    load_info::emit_tracing(engine.info());

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
/// ADR-017 §R-F1 startup-disable predicate: returns `true` iff the
/// kv-persist substrate should be constructed at startup.
///
/// Contract:
///   * `flag` — `--kv-persist=PATH` argument as a borrowed string slice
///     (`None` = flag absent).
///   * `env` — value of the `HF2Q_KV_PERSIST` env var (`None` = unset).
///
/// Rules:
///   * No flag → never enable (matches pre-ADR-017 behavior).
///   * Flag present + env unset / empty / any value other than `"0"`
///     (after trim) → enable.
///   * Flag present + env trims to exactly `"0"` → disable
///     (operator emergency-override per `docs/operating-kv-cache.md` §10).
///
/// Env-driven semantics are process-scoped: operators restart
/// `cmd_serve` to re-enable. The pure-function form keeps the policy
/// trivially unit-testable without spinning up the binary.
pub(crate) fn should_enable_kv_persist(flag: Option<&str>, env: Option<&str>) -> bool {
    flag.is_some() && env.map(|v| v.trim()).unwrap_or("") != "0"
}

pub(crate) fn maybe_print_serve_banner<W: std::io::Write>(
    info: &load_info::LoadInfo,
    w: &mut W,
    tty: bool,
    quiet: bool,
) -> std::io::Result<()> {
    if tty && !quiet {
        load_info::print_banner(info, w, tty)?;
    }
    Ok(())
}

pub fn cmd_serve(args: cli::ServeArgs) -> Result<()> {
    use api::schema::OverflowPolicy;
    use api::state::ServerConfig;

    // --- Resolve config ---
    let auth_token = args.auth_token.clone().or_else(|| {
        std::env::var("HF2Q_AUTH_TOKEN")
            .ok()
            .filter(|s| !s.is_empty())
    });

    let overflow_policy = match args.overflow_policy {
        cli::OverflowPolicyArg::Reject => OverflowPolicy::Reject,
        cli::OverflowPolicyArg::TruncateLeft => OverflowPolicy::TruncateLeft,
        cli::OverflowPolicyArg::Summarize => OverflowPolicy::Summarize,
    };

    let cache_dir = args
        .cache_dir
        .clone()
        .or_else(api::state::default_cache_dir);

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

    // --- ADR-017 Phase C.1 — optional persistent block-prefix KV cache ---
    // When `--kv-persist=PATH` is set, replace the AppState's
    // NoopKvSpiller-backed HotSwapManager with one wired to the
    // Phase A-substrate (DiskBlockStore + AsyncWriterHandle +
    // BlockPrefixCacheSpiller) and a per-loaded-family
    // EngineBindable registry. Off-path (flag absent) is byte-
    // identical to pre-C.1 — the existing AppState manager stays
    // wired with NoopKvSpiller.
    //
    // The substrate variables outlive the cmd_serve scope via Arc
    // captures inside the new HotSwapManager. The recovery scan
    // runs synchronously here so a previously-written cache becomes
    // available before the optional pre-warm load_or_get fires.
    //
    // ADR-017 §R-F1 startup-disable override: `HF2Q_KV_PERSIST=0`
    // (exact match, after trim) overrides `--kv-persist=PATH` and
    // leaves the NoopKvSpiller-backed AppState manager wired. This
    // closes the operator-runbook §10 gap (`docs/operating-kv-cache.md`)
    // — operators emergency-disable by setting the env var and
    // restarting (env vars are process-scoped, so true mid-flight
    // disable is restart-driven; that's the standard operator
    // pattern for env-driven config). Restart without
    // `HF2Q_KV_PERSIST=0` to re-enable. The decision predicate is
    // factored into `should_enable_kv_persist` below for unit
    // testing.
    let kv_persist_env = std::env::var("HF2Q_KV_PERSIST").ok();
    let kv_persist_flag_path = args.kv_persist_path.as_ref();
    let kv_persist_enabled = should_enable_kv_persist(
        kv_persist_flag_path.map(|p| p.to_string_lossy()).as_deref(),
        kv_persist_env.as_deref(),
    );
    if !kv_persist_enabled && kv_persist_flag_path.is_some() {
        // Flag was supplied but env override forced disable — surface
        // a single warn line so operators see the override took
        // effect without grepping logs for absent counters.
        let path_for_log = kv_persist_flag_path
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        tracing::warn!(
            kv_persist_path = %path_for_log,
            "ADR-017 R-F1 override: HF2Q_KV_PERSIST=0 — disabling kv-persist \
             despite --kv-persist={path_for_log}. Operator emergency-disable per \
             operating-kv-cache.md §10. Restart without HF2Q_KV_PERSIST=0 to \
             re-enable.",
            path_for_log = path_for_log,
        );
    }
    let kv_persist_loader_wrapper: Option<std::sync::Arc<
        crate::serve::kv_persist::LoaderWrapper<api::engine::Engine>,
    >> = if let Some(cache_dir) = args.kv_persist_path.as_ref().filter(|_| kv_persist_enabled) {
        use std::path::PathBuf;
        use std::sync::{Arc, Mutex};
        use crate::serve::kv_persist::families::gemma4_dense::{
            Gemma4DenseConfig, Gemma4DenseSpillFactory,
        };
        use crate::serve::kv_persist::registry::FamilyHookFactory;
        use crate::serve::kv_persist::{
            AsyncWriterHandle, BlockPrefixCacheSpiller, DiskBlockStore, KvPersistRegistry,
            LoaderWrapper, StubGemma4Spill, DEFAULT_CHANNEL_CAPACITY,
        };
        use crate::serve::multi_model::{DefaultModelLoader, HotSwapManager, LoadedPool};

        // 1. Ensure cache_dir exists.
        std::fs::create_dir_all(cache_dir).with_context(|| {
            format!(
                "ADR-017 C.1: create kv-persist cache dir at {}",
                cache_dir.display()
            )
        })?;

        // 2. Recovery scan — rebuild BlockIndex from on-disk
        //    envelopes left by a prior run. Returns
        //    `(BlockIndex, RecoveryReport)` per recovery.rs:110.
        //
        //    ADR-017 §R-F7: route through `recover_from_disk_with_counters`
        //    so the scan-time quarantine moves bump
        //    `hf2q_kv_quarantined_total{reason=...}` against the same
        //    AppState-owned counters Arc the /metrics handler reads.
        //    Upcast `Arc<KvSpillCounters>` → `Arc<dyn KvCacheMetricsSink>`
        //    here because the substrate's seam is trait-objected (per
        //    `kv_persist::metrics` module docs — keeps the lib facade
        //    decoupled from `serve::api`).
        let metrics_sink: Arc<dyn crate::serve::kv_persist::metrics::KvCacheMetricsSink> =
            Arc::clone(&state.kv_spill_counters)
                as Arc<dyn crate::serve::kv_persist::metrics::KvCacheMetricsSink>;
        let (recovered_index, recovery_report) =
            crate::serve::kv_persist::recover_from_disk_with_counters(
                cache_dir,
                Some(&metrics_sink),
            )
            .with_context(|| {
                format!(
                    "ADR-017 C.1: recover_from_disk({})",
                    cache_dir.display()
                )
            })?;
        tracing::info!(
            cache_dir = %cache_dir.display(),
            blocks_indexed = recovery_report.blocks_indexed,
            blocks_quarantined = recovery_report.blocks_quarantined,
            bytes_indexed = recovery_report.bytes_indexed,
            elapsed_ms = recovery_report.elapsed_ms,
            "ADR-017 C.1: kv-persist recovery scan complete"
        );

        // 3. DiskBlockStore — owns the read-path file I/O.
        //
        //    ADR-017 P1-3 (adversarial review fix): wire the on-disk
        //    LRU budget from `HF2Q_KV_PERSIST_BUDGET_BYTES` (u64,
        //    bytes). Default `0` = unlimited (matches the pre-P1-3
        //    behavior so the env var is purely additive). Per
        //    ADR-017 §R7 the intended future default is "10% of
        //    unified RAM" (~12.8 GiB on a 128-GiB M5 Max); that
        //    requires a `mlx_native::sysinfo::physical_ram_bytes()`
        //    helper which does not yet exist, and per the standing
        //    directive we do NOT add a new mlx-native dep just for
        //    this. Operators set the env var explicitly until §R7's
        //    helper lands.
        //
        //    Construct via `new_with_index` (kept at 0 for
        //    source-compat with existing callers), then override
        //    through `set_budget_bytes` so the AtomicU64 holds the
        //    parsed value before any eviction call fires.
        let kv_persist_budget_bytes: u64 =
            match std::env::var("HF2Q_KV_PERSIST_BUDGET_BYTES") {
                Ok(raw) => match raw.trim().parse::<u64>() {
                    Ok(parsed) => parsed,
                    Err(err) => {
                        tracing::warn!(
                            raw = %raw,
                            error = %err,
                            "ADR-017 P1-3: HF2Q_KV_PERSIST_BUDGET_BYTES \
                             parse failed; defaulting to 0 (unlimited)"
                        );
                        0
                    }
                },
                Err(_) => 0,
            };
        let store = Arc::new(
            DiskBlockStore::new_with_index(cache_dir.clone(), recovered_index, 0)
                .with_context(|| {
                    format!(
                        "ADR-017 C.1: DiskBlockStore::new_with_index({})",
                        cache_dir.display()
                    )
                })?,
        );
        store.set_budget_bytes(kv_persist_budget_bytes);
        // ADR-017 §R-F7: wire the DiskBlockStore to the AppState's
        // kv_spill_counters so eviction bumps
        // `hf2q_kv_cache_evictions_total{trigger="budget_overflow"}`
        // against the same Arc the /metrics handler reads. AppState
        // also gets a clone of the store handle so the gauges
        // `hf2q_kv_cache_bytes_on_disk` + `hf2q_kv_cache_blocks_total`
        // resolve at scrape time against the live BlockIndex.
        store.set_kv_counters(Arc::clone(&metrics_sink));
        state.kv_disk_store = Some(Arc::clone(&store));
        tracing::info!(
            budget_bytes = kv_persist_budget_bytes,
            "ADR-017 P1-3: HF2Q_KV_PERSIST_BUDGET_BYTES wired (0 = unlimited)"
        );

        // 4. AsyncWriterHandle — owns the write-path background
        //    worker. DEFAULT_CHANNEL_CAPACITY is the iter-212
        //    contract default; a future Phase D iter exposes it via
        //    a CLI knob.
        let writer = Arc::new(AsyncWriterHandle::spawn(
            Arc::clone(&store),
            DEFAULT_CHANNEL_CAPACITY,
        ));

        // 5. BlockPrefixCacheSpiller — owns the lifecycle. Per-
        //    family hooks register below.
        let spiller: Arc<BlockPrefixCacheSpiller<api::engine::Engine>> =
            Arc::new(BlockPrefixCacheSpiller::new(
                Arc::clone(&store),
                Arc::clone(&writer),
            ));

        // 6. KvPersistRegistry — the Phase C.1 EngineBindable
        //    registry. Populated once per known family below.
        let registry = Arc::new(KvPersistRegistry::new());

        // 7. Stub-family registration: register `StubGemma4Spill`
        //    for the operator-supplied --model (if any). The stub
        //    returns Skipped on every snapshot/restore call (per
        //    spiller.rs:506-528) and a no-op on bind/unbind (per
        //    the spiller.rs EngineBindable impl), so the on-path
        //    is observable but functionally inert.
        //
        //    B-dense.2 swaps `StubGemma4Spill` for the real
        //    `Gemma4DenseSpill` once the GGUF metadata extraction
        //    + EngineHandle construction lands.
        if let Some(model_arg) = default_model_arg.as_deref() {
            let stub = Arc::new(StubGemma4Spill);
            let stub_for_spiller: Arc<
                Mutex<dyn crate::serve::kv_persist::KvCacheSpill>,
            > = Arc::new(Mutex::new(StubGemma4Spill));
            let stub_for_registry: Arc<
                dyn crate::serve::kv_persist::EngineBindable,
            > = stub.clone();
            // Keys: derive a synthetic (repo, quant) the same way
            // the pre-warm path will (so the spiller's lookup hits
            // when load_or_get fires). Phase D's GGUF-derived
            // (repo, quant) lands later.
            let pool_repo = pool_key_for_path(&PathBuf::from(model_arg));
            let pool_quant = quant_select::QuantType::Q4_K_M;
            spiller.register_family(pool_repo.clone(), pool_quant, stub_for_spiller);
            registry.register(pool_repo.clone(), pool_quant, stub_for_registry);
            tracing::info!(
                repo = %pool_repo,
                quant = %pool_quant.as_str(),
                "ADR-017 C.1: registered StubGemma4Spill for operator --model"
            );

            // ADR-017 B-dense.2 — register a Gemma4DenseSpillFactory
            // alongside the C.1 stub. On the first successful engine
            // load delivering an `Arc<EngineHandle>` (the production
            // path drives this from a separate post-load bind in
            // cmd_serve, not the auto-LoaderWrapper-bind path), the
            // registry's `try_substitute_on_load` will materialize a
            // real Gemma4DenseSpill and atomically substitute BOTH
            // the spiller's family hook AND the registry's bindable
            // hook. Until that fires, the C.1 stub remains the
            // active hook (Skipped on every snapshot/restore).
            //
            // The shape config wired here is a *placeholder* for
            // B-dense.2's harness scope: the real GGUF-derived shape
            // landing extracts from the loaded `MlxModelWeights`
            // (LayerSpec / DenseKvBuffers / sliding_window) and
            // requires plumbing through the `cmd_serve` post-load
            // path. The factory itself is fully shape-parametric;
            // only the data fed to its constructor is placeholder.
            //
            // The matrix harness in `tests/kv_persist_gemma4_roundtrip.rs`
            // exercises the seam end-to-end via subprocess + real
            // GGUF.
            let placeholder_cfg = Gemma4DenseConfig {
                // Two layers — one Sliding, one Full — minimum that
                // exercises both layer-type branches in the spill's
                // capacity_for_layer / alloc_layer paths. The real
                // Gemma 4 26B has 64 layers (48 sliding + 16
                // full-attention); this placeholder is *correct
                // shape* but not *correct length* — the real-shape
                // wiring is a follow-up plumbing task that pulls
                // from MlxModelWeights post-load.
                layer_types: vec![
                    crate::serve::config::LayerType::Sliding,
                    crate::serve::config::LayerType::Full,
                ],
                nkv_heads: vec![8, 2],
                head_dim: vec![256, 512],
                kv_dtype: mlx_native::DType::F32,
                sliding_window: 4096,
                max_decode_tokens: 8192,
            };
            let factory: Arc<dyn FamilyHookFactory> =
                Arc::new(Gemma4DenseSpillFactory::new(placeholder_cfg));
            registry.register_factory(pool_repo.clone(), pool_quant, factory);
            tracing::info!(
                repo = %pool_repo,
                quant = %pool_quant.as_str(),
                "ADR-017 B-dense.2: registered Gemma4DenseSpillFactory \
                 (lazy real-hook construction at first engine load)"
            );
        }

        // 8. LoaderWrapper — decorates DefaultModelLoader. The
        //    wrapper's pending_bind slot is armed below, immediately
        //    before the pre-warm load_or_get call (per the
        //    synchronous-contract rationale in
        //    loader_wrapper.rs's module docs).
        let real_loader: Arc<
            dyn crate::serve::multi_model::ModelLoader<api::engine::Engine>,
        > = Arc::new(DefaultModelLoader);
        let wrapper = Arc::new(LoaderWrapper::new(real_loader, Arc::clone(&registry)));
        // ADR-017 B-dense.2 — wire the spiller into the wrapper so a
        // successful factory substitution updates both the registry
        // (handled inside try_substitute_on_load) AND the spiller's
        // register_family table (handled in
        // LoaderWrapper::update_spiller_registration). Order
        // matters: set_spiller MUST be called before any load_or_get
        // arms a pending_bind.
        wrapper.set_spiller(Arc::clone(&spiller));
        let loader_for_manager: Arc<
            dyn crate::serve::multi_model::ModelLoader<api::engine::Engine>,
        > = wrapper.clone();

        // 9. Build a fresh HotSwapManager using new_with_spiller and
        //    swap state.pool to point at it. Re-derive the pool +
        //    counters the same way AppState::new_for_serve did so
        //    behavior diverges from the off-path ONLY in the
        //    spiller wiring.
        let pool = LoadedPool::from_hardware(state.hardware.as_ref());
        let mut manager: HotSwapManager<api::engine::Engine> =
            HotSwapManager::new_with_spiller(
                pool,
                loader_for_manager,
                spiller as Arc<dyn crate::serve::multi_model::KvSpiller<api::engine::Engine>>,
            );
        manager.set_kv_counters(Arc::clone(&state.kv_spill_counters));
        state.pool = Arc::new(std::sync::RwLock::new(manager));

        tracing::info!(
            cache_dir = %cache_dir.display(),
            "ADR-017 C.1: kv-persist spiller substrate wired into HotSwapManager"
        );

        Some(wrapper)
    } else {
        None
    };

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
    let mut startup_engine_for_banner: Option<api::engine::Engine> = None;
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
        let pool_quant = resolved.quant.unwrap_or(quant_select::QuantType::Q4_K_M);
        let engine_config = multi_model::EngineConfig {
            tokenizer_path: args.tokenizer.clone(),
            config_path: args.config.clone(),
            queue_capacity: config.queue_capacity,
            warmup_synchronously: true,
        };
        // ADR-017 C.1: arm the LoaderWrapper's pending_bind slot for
        // the about-to-fire load_or_get. Synchronous contract — see
        // loader_wrapper.rs's module docs.
        if let Some(wrapper) = kv_persist_loader_wrapper.as_ref() {
            wrapper.set_pending_bind(pool_repo.clone(), pool_quant);
        }

        let mut pool_guard = state
            .pool
            .write()
            .map_err(|e| anyhow::anyhow!("pool rwlock poisoned at startup: {e}"))?;
        let loaded_engine = pool_guard
            .load_or_get(&pool_repo, pool_quant, &resolved.gguf_path, &engine_config)
            .map_err(|e| anyhow::anyhow!("startup pre-warm: {e}"))?;
        startup_engine_for_banner = Some(loaded_engine.engine.clone());
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
                let weights = crate::inference::models::bert::weights::LoadedBertWeights::load(
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
                let cfg =
                    crate::inference::models::nomic_bert::NomicBertConfig::from_gguf(&gguf)
                        .map_err(|e| anyhow::anyhow!("nomic-bert GGUF config parse failed: {e}"))?;
                crate::inference::models::nomic_bert::validate_tensor_set(&gguf, &cfg)
                    .map_err(|e| anyhow::anyhow!("nomic-bert GGUF tensor validation: {e}"))?;
                let weights = crate::inference::models::nomic_bert::LoadedNomicBertWeights::load(
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
        let skip_mmproj_load = std::env::var("HF2Q_SKIP_MMPROJ_LOAD").as_deref() == Ok("1");
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
                &gguf,
                &mmp_config,
                device,
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
        let skip_vit_warmup = std::env::var("HF2Q_SKIP_VIT_WARMUP").as_deref() == Ok("1");
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

    let stdout_is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let quiet = args.quiet;

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
        if let Some(engine) = startup_engine_for_banner.as_ref() {
            let mut stdout = std::io::stdout();
            maybe_print_serve_banner(engine.info(), &mut stdout, stdout_is_tty, quiet)
                .context("print serve load banner")?;
        }
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
            let pool_state_log = state_for_warmup.pool.read().ok().map(|m| m.pool_stats());
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
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
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
/// ADR-017 §R-F6: each action accepts `--kv-namespace` to retarget
/// from the model-weights cache to the persistent KV cache subtree
/// (`<kv-persist>/models/<fp_short>/`). Path discovery uses
/// `--kv-path PATH` then `HF2Q_KV_PERSIST_PATH`; no implicit default.
/// See `docs/operating-kv-cache.md` §11 #4.
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
/// - `--kv-namespace clear` refuses without `--model` and refuses
///   `--all` entirely (per-repo scope only — see runbook §11 #4).
/// - `--kv-namespace clear` consults a `<kv_root>/.cache_lock`
///   sentinel via `flock(LOCK_EX | LOCK_NB)`; pass `--force` to
///   override.
pub fn cmd_cache(args: cli::CacheArgs) -> Result<()> {
    use cli::CacheAction;

    // ADR-017 §R-F6: when `--kv-namespace` is set on any action, the
    // operation targets `<kv-persist>/models/` instead of the
    // weights-side `cache::ModelCache`. We dispatch BEFORE opening
    // the weights cache so a kv-namespace op never touches
    // `~/.cache/hf2q/cache_index.json`.
    match args.action {
        CacheAction::List {
            kv_namespace: true,
            kv_path,
        } => {
            return cmd_cache_kv_list(kv_path.as_deref());
        }
        CacheAction::Size {
            kv_namespace: true,
            kv_path,
        } => {
            return cmd_cache_kv_size(kv_path.as_deref());
        }
        CacheAction::Clear {
            kv_namespace: true,
            ref model,
            ref quant,
            all,
            yes: _,
            ref kv_path,
            force,
        } => {
            return cmd_cache_kv_clear(
                kv_path.as_deref(),
                model.as_deref(),
                quant.as_deref(),
                all,
                force,
            );
        }
        _ => { /* fall through to weights-side */ }
    }

    let mut cache = cache::ModelCache::open().context("open model cache")?;
    match args.action {
        CacheAction::List { .. } => cmd_cache_list(&cache),
        CacheAction::Size { .. } => cmd_cache_size(&cache),
        CacheAction::Clear {
            model,
            quant,
            all,
            yes,
            kv_namespace: _,
            kv_path: _,
            force: _,
        } => cmd_cache_clear(&mut cache, model, quant, all, yes),
    }
}

/// `hf2q cache list --kv-namespace [--kv-path PATH]` — ADR-017 §R-F6.
///
/// Walks `<kv_root>/models/` and emits one row per `<fp_short>` dir
/// with `bytes_on_disk` + `block_count`. Tolerates missing
/// `<kv_root>` / missing `models/` (operator may invoke before the
/// first `cmd_serve --kv-persist` run).
fn cmd_cache_kv_list(kv_path: Option<&Path>) -> Result<()> {
    use crate::serve::kv_persist::cache_ops;

    let kv_root = cache_ops::resolve_kv_root(kv_path)?;
    let entries = cache_ops::list_namespaces(&kv_root)?;
    if entries.is_empty() {
        println!(
            "(kv-cache empty — root: {})",
            kv_root.display()
        );
        return Ok(());
    }
    println!("hf2q kv-cache @ {}", kv_root.display());
    println!(
        "{:<20} {:>14} {:>10}",
        "FP_SHORT", "BYTES_ON_DISK", "BLOCKS"
    );
    for e in &entries {
        println!(
            "{:<20} {:>14} {:>10}",
            e.fp_short, e.bytes_on_disk, e.block_count
        );
    }
    Ok(())
}

/// `hf2q cache size --kv-namespace [--kv-path PATH]` — ADR-017 §R-F6.
fn cmd_cache_kv_size(kv_path: Option<&Path>) -> Result<()> {
    use crate::serve::kv_persist::cache_ops;

    let kv_root = cache_ops::resolve_kv_root(kv_path)?;
    let total = cache_ops::total_bytes(&kv_root);
    println!(
        "hf2q kv-cache @ {} — {} bytes ({:.2} GiB)",
        kv_root.display(),
        total,
        total as f64 / (1u64 << 30) as f64,
    );
    Ok(())
}

/// `hf2q cache clear --kv-namespace --model <repo> [--quant <q>]
///   [--kv-path PATH] [--force]` — ADR-017 §R-F6.
///
/// Refuses without `--model` (no whole-cache wipe — operator runbook
/// §11 #4 deliberately scopes this command to per-repo); refuses
/// under `--all` (likewise). Without `--force` the active-serve
/// sentinel-flock guard is consulted; the diagnostic is explicit
/// about its meaning.
fn cmd_cache_kv_clear(
    kv_path: Option<&Path>,
    model: Option<&str>,
    quant: Option<&str>,
    all: bool,
    force: bool,
) -> Result<()> {
    use crate::serve::kv_persist::cache_ops;
    use crate::serve::quant_select::QuantType;

    if all {
        return Err(anyhow::anyhow!(
            "hf2q cache clear --kv-namespace: --all is not supported. \
             Per-repo scope only (operator runbook §11 #4). To wipe \
             the entire kv-cache, stop `hf2q serve` and `rm -rf \
             <kv-persist>/models <kv-persist>/locks` directly."
        ));
    }
    let Some(repo) = model else {
        return Err(anyhow::anyhow!(
            "hf2q cache clear --kv-namespace: --model <repo-id> required \
             (no whole-cache wipe via this command — see operator runbook §11 #4)"
        ));
    };

    let kv_root = cache_ops::resolve_kv_root(kv_path)?;

    if let Some(q_str) = quant {
        let q = QuantType::from_canonical_str(q_str)
            .map_err(|e| anyhow::anyhow!("--quant: {}", e))?;
        let outcome = cache_ops::clear_namespace(&kv_root, repo, q, force).map_err(|e| {
            anyhow::anyhow!(
                "hf2q cache clear --kv-namespace --model {} --quant {}: {}",
                repo,
                q.as_str(),
                e
            )
        })?;
        if outcome.existed {
            println!(
                "hf2q kv-cache: cleared {}@{} (fp_short={}, {} bytes freed)",
                repo, q.as_str(), outcome.fp_short, outcome.bytes_freed
            );
        } else {
            println!(
                "hf2q kv-cache: nothing to clear for {}@{} (fp_short={} not present)",
                repo, q.as_str(), outcome.fp_short
            );
        }
    } else {
        let outcomes = cache_ops::clear_namespace_all_quants(&kv_root, repo, force)
            .map_err(|e| {
                anyhow::anyhow!(
                    "hf2q cache clear --kv-namespace --model {} (all quants): {}",
                    repo, e
                )
            })?;
        let total_bytes: u64 = outcomes.iter().map(|o| o.bytes_freed).sum();
        let removed: Vec<&str> = outcomes
            .iter()
            .filter(|o| o.existed)
            .map(|o| o.fp_short.as_str())
            .collect();
        if removed.is_empty() {
            println!(
                "hf2q kv-cache: nothing to clear for {} (no quant variants present)",
                repo
            );
        } else {
            println!(
                "hf2q kv-cache: cleared {} (all quants) — {} fp_short dirs, {} bytes freed: [{}]",
                repo,
                removed.len(),
                total_bytes,
                removed.join(", ")
            );
        }
    }
    Ok(())
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
                view.repo_id, "(none)", "-", view.model.last_accessed_secs,
            );
            continue;
        }
        for (quant, qe) in &view.model.quantizations {
            println!(
                "{:<48} {:<10} {:>12} {:>20}",
                view.repo_id, quant, qe.bytes, view.model.last_accessed_secs,
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
        let q =
            QuantType::from_canonical_str(&q_str).map_err(|e| anyhow::anyhow!("--quant: {}", e))?;
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
                parity_quality::cmd_parity_capture_tq_quality(&model, &output, &prompt, max_tokens)
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
    anyhow::ensure!(
        prompt_file.exists(),
        "Prompt file not found: {}",
        prompt_file.display()
    );
    let prompt_text = std::fs::read_to_string(&prompt_file)?.trim().to_string();

    // Load reference. Default: llama.cpp-anchored parity (*_llama.txt).
    // Gate D (--self-baseline): hf2q frozen self-baseline (*_hf2q.txt),
    // bisect-safe when math deliberately changes and llama.cpp drift is
    // expected.
    let ref_suffix = if self_baseline { "_hf2q" } else { "_llama" };
    let ref_file = ref_dir.join(format!("{prompt_name}{ref_suffix}.txt"));
    anyhow::ensure!(
        ref_file.exists(),
        "Reference file not found: {}",
        ref_file.display()
    );
    let ref_bytes = std::fs::read(&ref_file)?;

    // Determine settings
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(ref_dir.join("MANIFEST.json"))?)?;
    let prompt_meta = &manifest["prompts"][prompt_name];
    let tokens =
        max_tokens.unwrap_or_else(|| prompt_meta["max_tokens"].as_u64().unwrap_or(1000) as usize);
    let threshold = min_prefix.unwrap_or_else(||
        // Parse from gate field like "common_prefix >= 3094"
        prompt_meta["parity_gate"].as_str()
            .and_then(|s| s.split(">=").nth(1))
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(0));

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
    let mut ctx = gpu::GpuContext::new().map_err(|e| anyhow::anyhow!("GPU init: {e}"))?;
    let gguf = mlx_native::gguf::GgufFile::open(model_path)
        .map_err(|e| anyhow::anyhow!("GGUF open: {e}"))?;
    // cmd_parity has its own output contract — no progress line.
    let mut parity_progress = header::LoadProgress::new(false, 1, 0);
    let mut mlx_w =
        forward_mlx::MlxModelWeights::load_from_gguf(&gguf, &cfg, &mut ctx, &mut parity_progress)?;

    let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer: {e}"))?;
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow::anyhow!("Tokenizer truncation: {e}"))?;

    let rendered = render_chat_template(
        &gguf,
        &cli::GenerateArgs {
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
        },
        &prompt_text,
    )?;

    let encoding = tokenizer
        .encode(rendered.as_str(), false)
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
    let ref_label = if self_baseline {
        "frozen hf2q"
    } else {
        "llama.cpp"
    };
    println!("Reference: {} bytes ({})", ref_bytes.len(), ref_label);
    println!("hf2q:      {} bytes", hf2q_bytes.len());
    println!("Common:    {} bytes", common);
    if self_baseline {
        // Gate D contract: byte-identical. Length must match AND every
        // byte in the common-prefix comparison was equal.
        let identical = hf2q_bytes.len() == ref_bytes.len() && common == ref_bytes.len();
        if identical {
            println!(
                "PASS: byte-identical to frozen hf2q baseline ({} bytes)",
                common
            );
        } else {
            println!("FAIL: not byte-identical to frozen hf2q baseline");
            if common < n {
                let ctx_start = common;
                let ctx_end = (common + 80).min(n);
                let ref_snip = String::from_utf8_lossy(&ref_bytes[ctx_start..ctx_end]);
                let hf2q_snip =
                    String::from_utf8_lossy(&hf2q_bytes[ctx_start..ctx_end.min(hf2q_bytes.len())]);
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
                let hf2q_snip =
                    String::from_utf8_lossy(&hf2q_bytes[ctx_start..ctx_end.min(hf2q_bytes.len())]);
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
        vec![
            "sourdough".into(),
            "short_hello".into(),
            "sliding_wrap".into(),
        ]
    } else {
        vec![prompt_name.to_string()]
    };

    // Load model once
    let tokenizer_path = find_tokenizer(model_path, None)?;
    let config_path = find_config(model_path, None)?;
    let cfg = config::Gemma4Config::from_config_json(&config_path)?;
    let mut ctx = gpu::GpuContext::new().map_err(|e| anyhow::anyhow!("GPU init: {e}"))?;
    let gguf = mlx_native::gguf::GgufFile::open(model_path)
        .map_err(|e| anyhow::anyhow!("GGUF open: {e}"))?;
    // Model loaded once here; individual prompts re-create weights below to reset KV state
    let _gguf_preload = &gguf; // keep gguf alive
    let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer: {e}"))?;
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow::anyhow!("Tokenizer truncation: {e}"))?;

    std::fs::create_dir_all(output_dir)?;

    for pname in &prompts {
        let prompt_file = evals_dir.join("prompts").join(format!("{pname}.txt"));
        anyhow::ensure!(
            prompt_file.exists(),
            "Prompt not found: {}",
            prompt_file.display()
        );
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
            &gguf,
            &cfg,
            &mut ctx,
            &mut parity_progress,
        )?;

        let rendered = render_chat_template(
            &gguf,
            &cli::GenerateArgs {
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
            },
            &prompt_text,
        )?;

        let encoding = tokenizer
            .encode(rendered.as_str(), false)
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
    use super::{
        detect_greedy_repetition_loop, maybe_print_serve_banner, render_jinja_template,
        should_enable_kv_persist, FALLBACK_GEMMA4_API_CHAT_TEMPLATE,
        FALLBACK_GEMMA4_CHAT_TEMPLATE,
    };
    use crate::serve::load_info::{
        ArchFamily, ChatTemplateSource, LoadInfo, TokenizerSource,
    };
    use crate::serve::provenance::Provenance;
    use std::path::PathBuf;
    use std::time::Duration;

    fn synthetic_serve_banner_info() -> LoadInfo {
        LoadInfo {
            model_id: "serve-test-model".to_string(),
            arch_str: "gemma4".to_string(),
            arch_family: ArchFamily::Gemma4,
            model_path: PathBuf::from("/tmp/serve-test-model.gguf"),
            on_disk_bytes: 1024,
            backend_chip: "Apple M5 Max".to_string(),
            backend: "mlx-native",
            n_layers: 2,
            hidden_size: 32,
            vocab_size: 128,
            n_attention_heads: 4,
            n_key_value_heads: 2,
            head_dim: 8,
            sliding_window: Some(16),
            full_attention_interval: None,
            max_context_length: Some(128),
            moe: None,
            quant_label: Some("Q4_K".to_string()),
            quant_bpw: Some(4.5),
            tokenizer_source: TokenizerSource::GgufEmbedded,
            eos_token_ids: vec![1],
            bos_token_id: Some(2),
            chat_template_source: ChatTemplateSource::GgufEmbedded,
            provenance: Provenance::External,
            vision_projector: None,
            load_wall_clock: Duration::from_millis(25),
            resident_weight_bytes: None,
            kv_cache_budget_bytes: None,
            kv_spill_active: false,
        }
    }

    #[test]
    fn cmd_serve_banner_emits_on_tty() {
        let info = synthetic_serve_banner_info();
        let mut buf = Vec::new();
        maybe_print_serve_banner(&info, &mut buf, true, false).expect("print serve banner");
        let got = String::from_utf8(buf).expect("utf8");
        assert_eq!(got.lines().count(), 13);
        assert!(got.contains("hf2q load: model = serve-test-model"));
        assert!(got.contains("\x1b[2m"));
    }

    #[test]
    fn cmd_serve_banner_silent_when_non_tty() {
        let info = synthetic_serve_banner_info();
        let mut buf = Vec::new();
        maybe_print_serve_banner(&info, &mut buf, false, false).expect("skip serve banner");
        assert!(buf.is_empty());
    }

    #[test]
    fn cmd_serve_banner_silent_when_quiet() {
        let info = synthetic_serve_banner_info();
        let mut buf = Vec::new();
        maybe_print_serve_banner(&info, &mut buf, true, true).expect("skip quiet serve banner");
        assert!(buf.is_empty());
    }

    // ADR-017 §R-F1 startup-disable override regression guards.
    // Closes operator-runbook §10 gap by pinning the predicate's
    // truth table. Pure-function tests; no binary / env mutation.

    #[test]
    fn hf2q_kv_persist_zero_overrides_flag() {
        assert!(!should_enable_kv_persist(Some("/path"), Some("0")));
    }

    #[test]
    fn hf2q_kv_persist_unset_respects_flag_present() {
        assert!(should_enable_kv_persist(Some("/path"), None));
    }

    #[test]
    fn hf2q_kv_persist_one_respects_flag() {
        assert!(should_enable_kv_persist(Some("/path"), Some("1")));
    }

    #[test]
    fn hf2q_kv_persist_no_flag_means_disabled_regardless() {
        assert!(!should_enable_kv_persist(None, None));
        assert!(!should_enable_kv_persist(None, Some("0")));
        assert!(!should_enable_kv_persist(None, Some("1")));
    }

    #[test]
    fn hf2q_kv_persist_zero_with_whitespace_still_disables() {
        // Trim semantics — operators occasionally export with
        // trailing whitespace from copy-paste; `"  0  "` MUST still
        // trigger the override.
        assert!(!should_enable_kv_persist(Some("/path"), Some("  0  ")));
        assert!(!should_enable_kv_persist(Some("/path"), Some("0\n")));
    }

    #[test]
    fn hf2q_kv_persist_empty_or_other_values_respect_flag() {
        // Empty / `"true"` / `"yes"` / arbitrary strings all
        // respect the flag (only exact `"0"` after trim disables).
        assert!(should_enable_kv_persist(Some("/path"), Some("")));
        assert!(should_enable_kv_persist(Some("/path"), Some("true")));
        assert!(should_enable_kv_persist(Some("/path"), Some("yes")));
        assert!(should_enable_kv_persist(Some("/path"), Some("00")));
    }


    /// iter-219b parity-gate fix (2026-05-01) regression guard. The CLI
    /// fallback chat template MUST NOT activate Gemma 4's thinking-mode
    /// via `<|think|>` system marker — that diverged from llama.cpp's
    /// behavior and broke all 6 parity-suite checks (Gates C/D/E + F)
    /// on HEAD `3cd6ea5`. iter-217 fixed the API path's fallback the
    /// same way; this test ensures the CLI fallback stays aligned.
    ///
    /// The template MUST end with the empty `<|channel>thought\n<channel|>`
    /// block (closes thinking-mode pre-content) and MUST NOT contain
    /// `<|think|>` anywhere.
    #[test]
    fn iter219b_cli_fallback_chat_template_matches_iter217_contract() {
        assert!(
            !FALLBACK_GEMMA4_CHAT_TEMPLATE.contains("<|think|>"),
            "CLI fallback MUST NOT contain `<|think|>` (activates thinking-mode \
             that diverges from llama.cpp parity gate). Got: {FALLBACK_GEMMA4_CHAT_TEMPLATE:?}"
        );
        assert!(
            FALLBACK_GEMMA4_CHAT_TEMPLATE.contains("<|channel>thought\n<channel|>"),
            "CLI fallback MUST end with empty `<|channel>thought\\n<channel|>` block \
             (closes thinking-mode pre-content; mirrors iter-217 API-path fix). \
             Got: {FALLBACK_GEMMA4_CHAT_TEMPLATE:?}"
        );
        assert!(
            FALLBACK_GEMMA4_CHAT_TEMPLATE.contains("{{PROMPT}}"),
            "CLI fallback MUST keep `{{{{PROMPT}}}}` placeholder for `String::replace` \
             rendering. Got: {FALLBACK_GEMMA4_CHAT_TEMPLATE:?}"
        );
        // API fallback must also retain its iter-217 invariant (sibling check).
        assert!(
            !FALLBACK_GEMMA4_API_CHAT_TEMPLATE.contains("<|think|>"),
            "API fallback regression — `<|think|>` reintroduced after iter-217 fix"
        );
        assert!(
            FALLBACK_GEMMA4_API_CHAT_TEMPLATE.contains("<|channel>thought\n<channel|>"),
            "API fallback regression — empty channel block missing"
        );
    }

    /// Minimal Gemma-like template: verifies minijinja rendering of a single
    /// user message with `add_generation_prompt`.
    #[test]
    fn jinja_template_renders_single_user_turn() {
        let tmpl = "{{ bos_token }}{% for m in messages %}<|turn|>{{ m.role }}\n{{ m.content }}<|end|>\n{% endfor %}{% if add_generation_prompt %}<|turn|>model\n{% endif %}";
        let out = render_jinja_template(tmpl, "hello").expect("render ok");
        assert!(
            out.starts_with("<bos>"),
            "output should start with bos_token: {out}"
        );
        assert!(
            out.contains("<|turn|>user\nhello<|end|>"),
            "user turn missing: {out}"
        );
        assert!(
            out.ends_with("<|turn|>model\n"),
            "generation prompt missing: {out}"
        );
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

    // ── ADR-005 Phase 4 reopen iter-215 Wedge-2 — load_engine arch dispatch ──
    //
    // Iter-215 replaces the iter-214 wedge-1 bail (commit 064797e) with
    // actual SERVE-side dispatch.  The pre-iter-215 tests
    // (load_engine_rejects_qwen35_arch_with_operator_actionable_error,
    //  load_engine_rejects_qwen35moe_arch_with_operator_actionable_error,
    //  load_engine_does_not_reject_unknown_arch_at_wedge)
    // tested the bail behavior; they are replaced below by tests that
    // assert the new dispatch routes to the correct LoadedModel variant.

    /// Write a minimal valid GGUF (magic + version + 0 tensors + N
    /// string KVs) to a tempfile. Just enough structure for the
    /// header-only parse to succeed and `metadata_string` to find
    /// the keys.
    fn write_minimal_gguf_with_arch(arch: &str) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::Builder::new()
            .suffix(".gguf")
            .tempfile()
            .expect("tempfile");
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count = 1
        // KV: key="general.architecture" value=<arch>
        let key = b"general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING = 8
        let val = arch.as_bytes();
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);
        f.write_all(&buf).expect("write");
        f.flush().expect("flush");
        f
    }

    /// Synthetic GGUF with arch=qwen35 routes through
    /// `Qwen35LoadedModel::load`.  The minimal GGUF has 0 tensors so
    /// the full weights load fails downstream — but the failure
    /// surface MUST be from the qwen35 path, NOT the pre-iter-215
    /// bail or the Gemma loader's `attn_q.weight not found` panic.
    #[test]
    fn load_engine_routes_qwen35_to_qwen35_loaded_model() {
        let tmp = write_minimal_gguf_with_arch("qwen35");
        let cfg = super::multi_model::EngineConfig {
            tokenizer_path: None,
            config_path: None,
            queue_capacity: 4,
            warmup_synchronously: false,
        };
        let result = super::load_engine(tmp.path(), &cfg);
        // 0-tensor GGUF can't fully load, but the FAILURE shape proves
        // dispatch reached the qwen35 path.
        assert!(result.is_err(), "0-tensor synthetic GGUF must fail load");
        let msg = format!("{:#}", result.err().unwrap());
        // Wedge-1 bail message is GONE.
        assert!(
            !msg.contains("Phase 4 follow-up: SERVE-side load for general.architecture"),
            "wedge-1 bail must not fire post iter-215; got: {msg}"
        );
        // Wedge-1 message body referenced cmd_generate_qwen35 — the
        // SERVE-side load must NOT name that as a workaround anymore
        // (the SERVE path handles qwen35 now; chat returns 501 from
        // the worker arm, not load_engine).  The downstream
        // Qwen35Model::load_from_gguf failure does not name it.
        assert!(
            !msg.contains("hf2q generate") || !msg.contains("cmd_generate_qwen35"),
            "load_engine error must not include the iter-214 wedge-1 workaround \
             pointer; got: {msg}"
        );
    }

    /// Symmetric for qwen35moe (MoE variant routes through the same
    /// arm because the dispatcher's match covers both).
    #[test]
    fn load_engine_routes_qwen35moe_to_qwen35_loaded_model() {
        let tmp = write_minimal_gguf_with_arch("qwen35moe");
        let cfg = super::multi_model::EngineConfig {
            tokenizer_path: None,
            config_path: None,
            queue_capacity: 4,
            warmup_synchronously: false,
        };
        let result = super::load_engine(tmp.path(), &cfg);
        assert!(result.is_err(), "0-tensor synthetic GGUF must fail load");
        let msg = format!("{:#}", result.err().unwrap());
        assert!(
            !msg.contains("Phase 4 follow-up: SERVE-side load for general.architecture"),
            "wedge-1 bail must not fire post iter-215; got: {msg}"
        );
    }

    /// Negative-path / regression guard: unknown architectures still
    /// route through the Gemma default arm of `LoadedModel::load`.
    /// The Gemma loader fails for missing config/tensors but the
    /// failure is the Gemma path's failure, NOT a wedge bail or a
    /// qwen35 dispatch.  Iter-215 dispatch must not over-route on the
    /// match arm.
    #[test]
    fn load_engine_routes_unknown_arch_to_gemma_default_unchanged() {
        let tmp = write_minimal_gguf_with_arch("totally-fake-arch-name");
        let cfg = super::multi_model::EngineConfig {
            tokenizer_path: None,
            config_path: None,
            queue_capacity: 4,
            warmup_synchronously: false,
        };
        let result = super::load_engine(tmp.path(), &cfg);
        assert!(result.is_err(), "minimal GGUF must fail downstream load");
        let msg = format!("{:#}", result.err().unwrap());
        assert!(
            !msg.contains("Phase 4 follow-up: SERVE-side load for general.architecture"),
            "wedge-1 bail must not fire on unknown arch; got: {msg}"
        );
        // Iter-215 sentinel must NOT fire for unknown arch — that
        // would mean the dispatcher's match arm widened beyond the
        // intended set.
        assert!(
            !msg.contains("qwen35_not_implemented"),
            "qwen35 sentinel must not fire on unknown arch; got: {msg}"
        );
    }

    // ---- Greedy repetition-loop detector tests ----
    //
    // 2026-05-02 user reported the qwen35 generate CLI looping for 1014
    // tokens of "I should make sure the response is accurate and up-to-date."
    // before `<|im_end|>` finally fired. Pure greedy on thinking-style
    // checkpoints is brittle. The detector below is the deterministic-mode
    // escape hatch — sampling (temp/top_k/top_p) lands as a follow-up.

    #[test]
    fn detect_repetition_returns_none_below_window_size() {
        let toks: Vec<u32> = (0..50).collect();
        assert_eq!(detect_greedy_repetition_loop(&toks), None);
    }

    #[test]
    fn detect_repetition_returns_none_for_diverse_tokens() {
        let toks: Vec<u32> = (0..200).collect();
        assert_eq!(detect_greedy_repetition_loop(&toks), None);
    }

    #[test]
    fn detect_repetition_finds_8_token_cycle() {
        // 8-token cycle repeated 16 times → 128-token window full of one cycle.
        let cycle: [u32; 8] = [101, 102, 103, 104, 105, 106, 107, 108];
        let mut toks: Vec<u32> = (0..50).collect();
        for _ in 0..20 {
            toks.extend_from_slice(&cycle);
        }
        let result = detect_greedy_repetition_loop(&toks);
        assert!(result.is_some(), "8-token cycle should be detected");
        let (ngram, occ) = result.unwrap();
        assert_eq!(ngram, 8, "should detect at the smallest matching size");
        assert!(
            occ >= 4,
            "should report ≥4 occurrences in a saturated window, got {occ}"
        );
    }

    #[test]
    fn detect_repetition_finds_16_token_cycle() {
        // 16-token cycle won't be detected at ngram=8 (key not periodic at
        // smaller granularity for arbitrary tokens), but at ngram=16 should
        // fire.
        let cycle: [u32; 16] = [
            201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216,
        ];
        let mut toks: Vec<u32> = (0..50).collect();
        for _ in 0..10 {
            toks.extend_from_slice(&cycle);
        }
        let result = detect_greedy_repetition_loop(&toks);
        assert!(result.is_some(), "16-token cycle should be detected");
    }

    #[test]
    fn detect_repetition_tolerates_minor_drift() {
        // User's actual case: cycle is mostly identical with one outlier
        // mid-stream. Build 128-token window where last 16 tokens repeat
        // 5+ times non-consecutively (one outlier replaces a single token
        // in the middle).
        let cycle: [u32; 12] = [301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312];
        let outlier: u32 = 999;
        let mut toks: Vec<u32> = (0..30).collect();
        for i in 0..10 {
            if i == 4 {
                // Outlier replaces one token mid-cycle.
                let mut drifted = cycle.to_vec();
                drifted[6] = outlier;
                toks.extend_from_slice(&drifted);
            } else {
                toks.extend_from_slice(&cycle);
            }
        }
        let result = detect_greedy_repetition_loop(&toks);
        assert!(
            result.is_some(),
            "minor-drift cycle should still detect (non-consecutive matching), got None"
        );
    }

    #[test]
    fn detect_repetition_does_not_overcount_with_overlapping_increment() {
        // Pin the non-overlap invariant: an all-same-token sequence MUST NOT
        // report more than `window/ngram` occurrences (otherwise short n-grams
        // would over-count and produce false positives at the threshold). For
        // a 128-token all-zero window, ngram=8 should report ≤16 occurrences.
        let toks: Vec<u32> = vec![0; 200];
        let result = detect_greedy_repetition_loop(&toks);
        assert!(result.is_some());
        let (ngram, occ) = result.unwrap();
        assert!(
            occ <= 128 / ngram,
            "non-overlap invariant violated: ngram={ngram} occ={occ} > {}",
            128 / ngram
        );
    }
}
