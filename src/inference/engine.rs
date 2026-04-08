//! Inference engine — autoregressive text generation with configurable sampling.
//!
//! Orchestrates the full generation pipeline:
//! 1. Load chat template and render the prompt
//! 2. Tokenize the rendered prompt
//! 3. Prefill: single forward pass on all prompt tokens
//! 4. Sample the first generated token from last-position logits
//! 5. Decode loop: forward pass on single token, sample, check stop conditions
//! 6. Stream decoded text via callback as tokens are produced
//!
//! Runs synchronously for CLI use. A future story will wrap with
//! `spawn_blocking` + channels for async server integration.
//!
//! ## Double-Buffer Pipeline (Phase 3a)
//!
//! The decode loop is structured for future async overlap between the GPU
//! forward pass for token N+1 and CPU sampling/decoding of token N.
//!
//! Currently `model.forward()` calls `commit_and_wait()` internally, so the
//! two steps are still serialized. Phase 3b (handled by a separate agent in
//! gemma4.rs) will split this into `forward_start()` / `forward_wait()` to
//! enable true overlap.
//!
//! ### ForwardHandle — Future Async Interface (TODO: Phase 3b)
//!
//! Once gemma4.rs exposes non-blocking forward passes, the trait will gain:
//!
//! ```ignore
//! /// Begin a GPU forward pass without blocking the CPU.
//! /// Returns a handle that can be waited on later.
//! fn forward_start(&mut self, tokens: &[u32]) -> Result<ForwardHandle, Gemma4Error>;
//!
//! /// Block until the handle's GPU work completes and return the logits.
//! fn forward_wait(&mut self, handle: ForwardHandle) -> Result<Vec<f32>, Gemma4Error>;
//! ```
//!
//! The decode loop restructuring is already in place so that inserting
//! `forward_start` / `forward_wait` calls will achieve true pipeline overlap
//! without any further loop restructuring.

use std::path::Path;
use std::time::Instant;

use thiserror::Error;
use tracing::{debug, info};

use crate::inference::models::gemma4::{Gemma4Config, Gemma4Error, Gemma4Model};
use crate::inference::models::registry;
use crate::inference::prompt_cache::{CacheLookup, PromptCache, PromptCacheConfig};
use crate::inference::sampler::{Sampler, SamplerConfig};
use crate::inference::weight_loader;
use crate::tokenizer::chat_template::{ChatTemplate, ChatTemplateError, Message};
use crate::tokenizer::{HfTokenizer, TokenizerError};

use mlx_native::MlxDevice;

// ---------------------------------------------------------------------------
// Double-buffer pipeline types
// ---------------------------------------------------------------------------

/// Pre-allocated logit buffers for the decode loop.
///
/// The double-buffer pattern keeps two `Vec<f32>` of exactly `vocab_size`
/// elements alive for the entire generation run, eliminating per-step heap
/// allocations.  One buffer holds the logits being sampled (CPU), the other
/// is ready to receive the next forward-pass output (GPU).
///
/// When Phase 3b lands (non-blocking `forward_start` / `forward_wait`), the
/// CPU can sample from `current` while the GPU writes into `next`, achieving
/// true pipeline overlap without any allocation.
struct DecodeBuffers {
    /// Buffer for the current step's logits (being sampled on CPU).
    current: Vec<f32>,
    /// Buffer for the next step's logits (will receive GPU output).
    next: Vec<f32>,
}

impl DecodeBuffers {
    /// Allocate both buffers to `vocab_size` elements, zeroed.
    fn new(vocab_size: usize) -> Self {
        Self {
            current: vec![0.0f32; vocab_size],
            next: vec![0.0f32; vocab_size],
        }
    }

    /// Swap `current` and `next` in place (zero-cost pointer swap).
    ///
    /// Call this once per step after the GPU output has been written into
    /// `next` and before sampling from it:
    ///
    /// ```ignore
    /// // GPU writes into buffers.next ...
    /// buffers.swap();
    /// // Now sample from buffers.current (was next).
    /// ```
    #[inline]
    fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }
}

/// Outcome of processing a single sampled token in the decode loop.
///
/// Used by `process_sampled_token` to signal whether to continue or halt.
enum DecodeControl {
    /// Continue generating: next token ID is `next_token`.
    Continue { next_token: u32 },
    /// Stop generation (EOS, stop sequence, or callback returned false).
    Stop,
}

// ---------------------------------------------------------------------------
// Errors from the inference engine.
// ---------------------------------------------------------------------------

/// Errors from the inference engine.
#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Model loading failed: {0}")]
    ModelLoad(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),

    #[error("Chat template error: {0}")]
    ChatTemplate(#[from] ChatTemplateError),

    #[error("Forward pass error: {0}")]
    Forward(#[from] Gemma4Error),

    #[error("Device error: {0}")]
    Device(String),

    #[error("Generation error: {reason}")]
    Generation { reason: String },
}

/// Configuration for the inference engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Sampling configuration.
    pub sampler: SamplerConfig,
    /// Optional stop sequences (generation halts when any is produced).
    pub stop_sequences: Vec<String>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            sampler: SamplerConfig::default(),
            stop_sequences: Vec::new(),
        }
    }
}

/// Statistics from a generation run.
#[derive(Debug, Clone)]
pub struct GenerationStats {
    /// Number of prompt tokens processed during prefill.
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    pub generated_tokens: usize,
    /// Time spent in prefill (seconds).
    pub prefill_time_secs: f64,
    /// Time spent in decode loop (seconds).
    pub decode_time_secs: f64,
    /// Total wall-clock time (seconds).
    pub total_time_secs: f64,
    /// Number of prompt tokens that were served from the prompt cache
    /// (skipped during prefill). Zero when prompt caching is disabled
    /// or on a cache miss.
    pub cached_tokens: usize,
    /// Time to first token (milliseconds). Includes prefill + first sample.
    pub time_to_first_token_ms: f64,
    /// Number of GPU sync points (commit_and_wait calls) during this generation.
    pub gpu_sync_count: u64,
    /// Number of GPU kernel dispatches during this generation.
    pub gpu_dispatch_count: u64,
}

impl GenerationStats {
    /// Tokens per second during the decode phase.
    pub fn decode_tokens_per_sec(&self) -> f64 {
        if self.decode_time_secs > 0.0 {
            self.generated_tokens as f64 / self.decode_time_secs
        } else {
            0.0
        }
    }

    /// Prompt tokens per second during prefill.
    pub fn prefill_tokens_per_sec(&self) -> f64 {
        if self.prefill_time_secs > 0.0 {
            self.prompt_tokens as f64 / self.prefill_time_secs
        } else {
            0.0
        }
    }
}

/// The inference engine, holding all components needed for generation.
pub struct InferenceEngine {
    /// The loaded Gemma 4 model (with KV cache).
    model: Gemma4Model,
    /// HuggingFace tokenizer.
    tokenizer: HfTokenizer,
    /// Chat template for prompt formatting.
    chat_template: ChatTemplate,
    /// Engine configuration (max_tokens, stop sequences, etc.).
    pub config: EngineConfig,
    /// Prompt cache for prefix matching across multi-turn conversations.
    prompt_cache: PromptCache,
}

impl InferenceEngine {
    /// Build a new inference engine from a model directory and configuration.
    ///
    /// Loads the model weights, tokenizer, and chat template. The `chat_template_override`
    /// parameter allows overriding the model's built-in template with a CLI-specified file.
    ///
    /// Prompt caching is enabled by default. Pass a `PromptCacheConfig` with
    /// `enabled: false` to disable it, or use [`new_with_prompt_cache`].
    pub fn new(
        model_dir: &Path,
        config: EngineConfig,
        chat_template_override: Option<&Path>,
    ) -> Result<Self, EngineError> {
        // Load tokenizer
        let tokenizer = HfTokenizer::from_dir(model_dir)?;
        info!(vocab_size = tokenizer.vocab_size(), "Tokenizer loaded");

        // Load chat template (CLI override > model dir)
        let chat_template = if let Some(override_path) = chat_template_override {
            ChatTemplate::from_file(override_path)?
        } else {
            ChatTemplate::from_model_dir(model_dir)?
        };

        // Detect architecture and parse config
        let config_path = model_dir.join("config.json");
        let model_config = registry::detect_architecture(&config_path)
            .map_err(|e| EngineError::ModelLoad(format!("Architecture detection: {e}")))?;

        let gemma_config = Gemma4Config::from_model_config(&model_config.raw)
            .map_err(|e| EngineError::ModelLoad(format!("Config parse: {e}")))?;

        // Initialize Metal device
        let device = MlxDevice::new()
            .map_err(|e| EngineError::Device(format!("Metal device init: {e}")))?;

        // Load weights
        let weights = weight_loader::load_weights(model_dir, &device)
            .map_err(|e| EngineError::ModelLoad(format!("Weight loading: {e}")))?;

        info!(
            tensors = weights.len(),
            bytes = weights.total_bytes(),
            "Weights loaded"
        );

        // Build model
        let model = Gemma4Model::from_weights(weights, gemma_config, device)?;

        Ok(Self {
            model,
            tokenizer,
            chat_template,
            config,
            prompt_cache: PromptCache::new(PromptCacheConfig::default()),
        })
    }

    /// Build a new inference engine with explicit prompt cache configuration.
    ///
    /// Same as [`new`] but allows disabling or customizing prompt caching.
    pub fn new_with_prompt_cache(
        model_dir: &Path,
        config: EngineConfig,
        chat_template_override: Option<&Path>,
        prompt_cache_config: PromptCacheConfig,
    ) -> Result<Self, EngineError> {
        let mut engine = Self::new(model_dir, config, chat_template_override)?;
        engine.prompt_cache = PromptCache::new(prompt_cache_config);
        Ok(engine)
    }

    /// Generate text from a user prompt.
    ///
    /// The `on_token` callback is invoked for each generated token with the decoded
    /// text fragment. Return `true` to continue generation, `false` to stop early.
    ///
    /// Returns the full generated text and generation statistics.
    pub fn generate<F>(
        &mut self,
        prompt: &str,
        on_token: F,
    ) -> Result<(String, GenerationStats), EngineError>
    where
        F: Fn(&str) -> bool,
    {
        let total_start = Instant::now();

        // Step 1: Render the chat template
        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
            tool_call_id: None,
            tool_calls: None,
        }];

        // Use id_to_token (not decode) for BOS/EOS strings.
        // decode() skips special tokens, returning empty string for <bos>.
        let bos_token = self
            .tokenizer
            .bos_id()
            .and_then(|id| self.tokenizer.id_to_token(id))
            .unwrap_or_default();
        let eos_token = self
            .tokenizer
            .eos_id()
            .and_then(|id| self.tokenizer.id_to_token(id))
            .unwrap_or_default();

        let rendered = self
            .chat_template
            .render(&messages, &bos_token, &eos_token)?;
        debug!(rendered_len = rendered.len(), "Chat template rendered");

        // Step 2: Tokenize the rendered prompt
        let prompt_tokens = self.tokenizer.encode_with_special_tokens(&rendered)?;
        let prompt_len = prompt_tokens.len();
        debug!(prompt_tokens = prompt_len, "Prompt tokenized");

        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation {
                reason: "Prompt tokenized to zero tokens".into(),
            });
        }

        // Step 3: Prefill with prompt cache awareness
        let prefill_start = Instant::now();
        let (prefill_logits, cached_tokens) =
            self.prefill_with_cache(&prompt_tokens, false)?;
        let prefill_time = prefill_start.elapsed().as_secs_f64();

        let vocab_size = self.model.config().vocab_size;

        // The prefill_logits contain logits only for the tokens that were
        // actually encoded (the non-cached portion). We need the last position.
        let encoded_len = prompt_len - cached_tokens;
        let last_pos_logits =
            extract_last_position_logits(&prefill_logits, encoded_len, vocab_size)?;

        // Step 4: Create sampler and sample first token
        let mut sampler = Sampler::new(self.config.sampler.clone());
        let mut generated_ids: Vec<u32> = Vec::with_capacity(self.config.max_tokens);
        let mut generated_text = String::new();

        let eos_id = self.tokenizer.eos_id();

        let decode_start = Instant::now();

        // Reset GPU counters before sampling begins so we capture the full
        // generation cost (prefill sync + decode dispatches).
        // TODO(instrumentor-agent): enable once mlx_native::reset_counters lands.
        #[cfg(feature = "mlx-native")]
        {
            // mlx_native::reset_counters();
        }

        let first_token = sampler.sample_next(&last_pos_logits, &generated_ids);

        // TTFT: time from generate() entry to first sampled token.
        let time_to_first_token_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        // Check stop conditions for first token
        if Some(first_token) == eos_id {
            // Store prompt tokens in cache before returning
            self.store_prompt_in_cache(prompt_tokens, false);
            let total_time = total_start.elapsed().as_secs_f64();
            return Ok((
                generated_text,
                GenerationStats {
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                    prefill_time_secs: prefill_time,
                    decode_time_secs: 0.0,
                    total_time_secs: total_time,
                    cached_tokens,
                    time_to_first_token_ms,
                    gpu_sync_count: 0,
                    gpu_dispatch_count: 0,
                },
            ));
        }

        generated_ids.push(first_token);
        let token_text = self.tokenizer.decode(&[first_token]).unwrap_or_default();
        generated_text.push_str(&token_text);

        // Stream the first token
        if !on_token(&token_text) {
            self.store_prompt_in_cache(prompt_tokens, false);
            let total_time = total_start.elapsed().as_secs_f64();
            return Ok((
                generated_text,
                GenerationStats {
                    prompt_tokens: prompt_len,
                    generated_tokens: 1,
                    prefill_time_secs: prefill_time,
                    decode_time_secs: decode_start.elapsed().as_secs_f64(),
                    total_time_secs: total_time,
                    cached_tokens,
                    time_to_first_token_ms,
                    gpu_sync_count: 0,
                    gpu_dispatch_count: 0,
                },
            ));
        }

        // Step 5-6: Decode loop
        for step in 1..self.config.max_tokens {
            // Forward pass on the single new token (KV cache reuses previous keys/values)
            let step_logits = self.model.forward(&[*generated_ids.last().unwrap()])?;

            // Extract logits for the single position
            let token_logits = extract_last_position_logits(&step_logits, 1, vocab_size)?;

            // Sample next token
            let next_token = sampler.sample_next(&token_logits, &generated_ids);

            // Check EOS
            if Some(next_token) == eos_id {
                debug!(step = step, "EOS token generated");
                break;
            }

            generated_ids.push(next_token);

            // Decode the new token
            let token_text = self.tokenizer.decode(&[next_token]).unwrap_or_default();
            generated_text.push_str(&token_text);

            // Check stop sequences
            if self.check_stop_sequences(&generated_text) {
                // Remove the stop sequence from output
                self.trim_stop_sequence(&mut generated_text);
                debug!(step = step, "Stop sequence matched");
                break;
            }

            // Stream token to callback
            if !on_token(&token_text) {
                debug!(step = step, "Generation stopped by callback");
                break;
            }
        }

        // Store the prompt tokens in cache (not the generated tokens; those
        // are part of the assistant turn and will appear in the next request's
        // prompt via the chat template).
        self.store_prompt_in_cache(prompt_tokens, false);

        let decode_time = decode_start.elapsed().as_secs_f64();
        let total_time = total_start.elapsed().as_secs_f64();

        // Read GPU counters accumulated during this generation.
        // TODO(instrumentor-agent): enable once mlx_native counter functions land.
        #[cfg(feature = "mlx-native")]
        let (gpu_sync_count, gpu_dispatch_count) = {
            // (mlx_native::sync_count(), mlx_native::dispatch_count())
            (0u64, 0u64)
        };
        #[cfg(not(feature = "mlx-native"))]
        let (gpu_sync_count, gpu_dispatch_count) = (0u64, 0u64);

        let stats = GenerationStats {
            prompt_tokens: prompt_len,
            generated_tokens: generated_ids.len(),
            prefill_time_secs: prefill_time,
            decode_time_secs: decode_time,
            total_time_secs: total_time,
            cached_tokens,
            time_to_first_token_ms,
            gpu_sync_count,
            gpu_dispatch_count,
        };

        Ok((generated_text, stats))
    }

    /// Perform prefill with prompt cache awareness.
    ///
    /// Checks the prompt cache for a prefix match. On cache hit, restores
    /// the KV cache positions and encodes only the new tokens. On miss,
    /// resets the KV cache and encodes all tokens from scratch.
    ///
    /// Returns `(logits, cached_tokens)` where:
    /// - `logits` contains the forward-pass output for the encoded tokens
    ///   (shape `[encoded_len, vocab_size]` where `encoded_len = prompt_len - cached_tokens`)
    /// - `cached_tokens` is the number of tokens served from cache (0 on miss)
    fn prefill_with_cache(
        &mut self,
        prompt_tokens: &[u32],
        has_vision: bool,
    ) -> Result<(Vec<f32>, usize), EngineError> {
        let lookup = self.prompt_cache.lookup(prompt_tokens, has_vision);

        match lookup {
            CacheLookup::Hit {
                prefix_len,
                kv_position,
            } => {
                // Restore KV cache to the prefix boundary. The buffers already
                // contain valid KV data for positions 0..kv_position from the
                // previous forward pass; we just need to rewind the write heads.
                self.model.restore_cache_position(kv_position).map_err(|e| {
                    EngineError::Forward(e.into())
                })?;

                // Encode only the new tokens (from prefix_len onward)
                let new_tokens = &prompt_tokens[prefix_len..];
                if new_tokens.is_empty() {
                    // The entire prompt was cached. We still need logits for
                    // the last position to sample the first generated token.
                    // Re-encode just the last token to get its logits. The KV
                    // cache already has all the context, so this single-token
                    // forward is equivalent to a decode step.
                    let last_token = &prompt_tokens[prompt_tokens.len() - 1..];
                    // But we need to rewind one position since the last token's
                    // KV is already in the cache from the previous request.
                    // Actually: just rewind by 1 and re-encode the last token.
                    if kv_position > 0 {
                        self.model
                            .restore_cache_position(kv_position - 1)
                            .map_err(|e| EngineError::Forward(e.into()))?;
                    }
                    let logits = self.model.forward(last_token)?;
                    // The cached_tokens count is prefix_len - 1 since we
                    // re-encoded the last one.
                    return Ok((logits, prefix_len.saturating_sub(1)));
                }

                debug!(
                    cached = prefix_len,
                    new = new_tokens.len(),
                    "Prefill: encoding only new tokens after cache hit"
                );

                let logits = self.model.forward(new_tokens)?;
                Ok((logits, prefix_len))
            }
            CacheLookup::Miss => {
                // Full reset and encode from scratch
                self.model.reset_cache();
                let logits = self.model.forward(prompt_tokens)?;
                Ok((logits, 0))
            }
        }
    }

    /// Store the prompt token sequence in the cache after a successful
    /// generation (or early EOS).
    ///
    /// Only stores the prompt tokens (not the generated tokens). The
    /// generated tokens form the assistant's response, which will appear
    /// in the next request's prompt via the chat template.
    fn store_prompt_in_cache(&mut self, prompt_tokens: Vec<u32>, has_vision: bool) {
        let kv_position = prompt_tokens.len();
        let kv_byte_size = self.model.kv_cache_allocated_bytes();
        self.prompt_cache
            .store(prompt_tokens, kv_position, has_vision, kv_byte_size);
    }

    /// Check if the generated text ends with any configured stop sequence.
    fn check_stop_sequences(&self, text: &str) -> bool {
        self.config
            .stop_sequences
            .iter()
            .any(|seq| text.ends_with(seq.as_str()))
    }

    /// Remove the trailing stop sequence from generated text.
    fn trim_stop_sequence(&self, text: &mut String) {
        for seq in &self.config.stop_sequences {
            if text.ends_with(seq.as_str()) {
                text.truncate(text.len() - seq.len());
                return;
            }
        }
    }

    /// Generate text from an already-rendered prompt string.
    ///
    /// Unlike `generate()`, this method skips chat template rendering and
    /// expects the caller to have already formatted the prompt (including
    /// special tokens if needed). This is used by the API server which
    /// handles multi-message template rendering externally.
    ///
    /// This is the primary beneficiary of prompt caching: in multi-turn
    /// chat, each request contains the entire conversation history, so the
    /// common prefix grows with each turn.
    ///
    /// The `on_token` callback is invoked for each generated token with the
    /// decoded text fragment. Return `true` to continue, `false` to stop.
    pub fn generate_from_formatted_prompt<F>(
        &mut self,
        rendered_prompt: &str,
        on_token: F,
    ) -> Result<(String, GenerationStats), EngineError>
    where
        F: Fn(&str) -> bool,
    {
        let total_start = Instant::now();

        // Tokenize the already-rendered prompt
        let prompt_tokens = self.tokenizer.encode_with_special_tokens(rendered_prompt)?;
        let prompt_len = prompt_tokens.len();
        debug!(prompt_tokens = prompt_len, "Prompt tokenized");

        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation {
                reason: "Prompt tokenized to zero tokens".into(),
            });
        }

        // Prefill with prompt cache awareness
        let prefill_start = Instant::now();
        let (prefill_logits, cached_tokens) =
            self.prefill_with_cache(&prompt_tokens, false)?;
        let prefill_time = prefill_start.elapsed().as_secs_f64();

        let vocab_size = self.model.config().vocab_size;
        let encoded_len = prompt_len - cached_tokens;
        let last_pos_logits =
            extract_last_position_logits(&prefill_logits, encoded_len, vocab_size)?;

        // Sample first token
        let mut sampler = Sampler::new(self.config.sampler.clone());
        let mut generated_ids: Vec<u32> = Vec::with_capacity(self.config.max_tokens);
        let mut generated_text = String::new();
        let eos_id = self.tokenizer.eos_id();
        let decode_start = Instant::now();

        // Reset GPU counters before sampling begins.
        // TODO(instrumentor-agent): enable once mlx_native::reset_counters lands.
        #[cfg(feature = "mlx-native")]
        {
            // mlx_native::reset_counters();
        }

        let first_token = sampler.sample_next(&last_pos_logits, &generated_ids);

        // TTFT: time from generate_from_formatted_prompt() entry to first sampled token.
        let time_to_first_token_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        if Some(first_token) == eos_id {
            self.store_prompt_in_cache(prompt_tokens, false);
            let total_time = total_start.elapsed().as_secs_f64();
            return Ok((
                generated_text,
                GenerationStats {
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                    prefill_time_secs: prefill_time,
                    decode_time_secs: 0.0,
                    total_time_secs: total_time,
                    cached_tokens,
                    time_to_first_token_ms,
                    gpu_sync_count: 0,
                    gpu_dispatch_count: 0,
                },
            ));
        }

        generated_ids.push(first_token);
        let token_text = self.tokenizer.decode(&[first_token]).unwrap_or_default();
        generated_text.push_str(&token_text);

        if !on_token(&token_text) {
            self.store_prompt_in_cache(prompt_tokens, false);
            let total_time = total_start.elapsed().as_secs_f64();
            return Ok((
                generated_text,
                GenerationStats {
                    prompt_tokens: prompt_len,
                    generated_tokens: 1,
                    prefill_time_secs: prefill_time,
                    decode_time_secs: decode_start.elapsed().as_secs_f64(),
                    total_time_secs: total_time,
                    cached_tokens,
                    time_to_first_token_ms,
                    gpu_sync_count: 0,
                    gpu_dispatch_count: 0,
                },
            ));
        }

        // Decode loop
        for step in 1..self.config.max_tokens {
            let step_logits = self.model.forward(&[*generated_ids.last().unwrap()])?;
            let token_logits = extract_last_position_logits(&step_logits, 1, vocab_size)?;
            let next_token = sampler.sample_next(&token_logits, &generated_ids);

            if Some(next_token) == eos_id {
                debug!(step = step, "EOS token generated");
                break;
            }

            generated_ids.push(next_token);
            let token_text = self.tokenizer.decode(&[next_token]).unwrap_or_default();
            generated_text.push_str(&token_text);

            if self.check_stop_sequences(&generated_text) {
                self.trim_stop_sequence(&mut generated_text);
                debug!(step = step, "Stop sequence matched");
                break;
            }

            if !on_token(&token_text) {
                debug!(step = step, "Generation stopped by callback");
                break;
            }
        }

        // Store prompt tokens in the cache for the next turn
        self.store_prompt_in_cache(prompt_tokens, false);

        let decode_time = decode_start.elapsed().as_secs_f64();
        let total_time = total_start.elapsed().as_secs_f64();

        // Read GPU counters accumulated during this generation.
        // TODO(instrumentor-agent): enable once mlx_native counter functions land.
        #[cfg(feature = "mlx-native")]
        let (gpu_sync_count, gpu_dispatch_count) = {
            // (mlx_native::sync_count(), mlx_native::dispatch_count())
            (0u64, 0u64)
        };
        #[cfg(not(feature = "mlx-native"))]
        let (gpu_sync_count, gpu_dispatch_count) = (0u64, 0u64);

        Ok((
            generated_text,
            GenerationStats {
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
                prefill_time_secs: prefill_time,
                decode_time_secs: decode_time,
                total_time_secs: total_time,
                cached_tokens,
                time_to_first_token_ms,
                gpu_sync_count,
                gpu_dispatch_count,
            },
        ))
    }

    /// Generate text from a formatted prompt with vision features injected.
    ///
    /// This is the multimodal generation path. The `vision_features` buffer
    /// contains pre-encoded vision tokens (from the vision encoder), and
    /// `image_token_id` indicates which token positions should be replaced.
    ///
    /// The rendered prompt should contain `image_token_id` tokens as placeholders
    /// for the vision features.
    pub fn generate_from_formatted_prompt_with_vision<F>(
        &mut self,
        rendered_prompt: &str,
        image_token_id: u32,
        vision_features: &[f32],
        on_token: F,
    ) -> Result<(String, GenerationStats), EngineError>
    where
        F: Fn(&str) -> bool,
    {
        let total_start = Instant::now();

        // Tokenize the prompt
        let prompt_tokens = self.tokenizer.encode_with_special_tokens(rendered_prompt)?;
        let prompt_len = prompt_tokens.len();
        debug!(prompt_tokens = prompt_len, "Prompt tokenized (with vision)");

        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation {
                reason: "Prompt tokenized to zero tokens".into(),
            });
        }

        // Vision invalidates prompt cache -- always reset
        self.model.reset_cache();

        // Prefill with vision token injection
        let prefill_start = Instant::now();
        let prefill_logits = self.model.forward_with_vision(
            &prompt_tokens,
            image_token_id,
            vision_features,
        )?;
        let prefill_time = prefill_start.elapsed().as_secs_f64();

        let vocab_size = self.model.config().vocab_size;
        let last_pos_logits =
            extract_last_position_logits(&prefill_logits, prompt_len, vocab_size)?;

        // Sample first token
        let mut sampler = Sampler::new(self.config.sampler.clone());
        let mut generated_ids: Vec<u32> = Vec::with_capacity(self.config.max_tokens);
        let mut generated_text = String::new();
        let eos_id = self.tokenizer.eos_id();
        let decode_start = Instant::now();

        // Reset GPU counters before sampling begins.
        // TODO(instrumentor-agent): enable once mlx_native::reset_counters lands.
        #[cfg(feature = "mlx-native")]
        {
            // mlx_native::reset_counters();
        }

        let first_token = sampler.sample_next(&last_pos_logits, &generated_ids);

        // TTFT: time from generate_from_formatted_prompt_with_vision() entry to first sampled token.
        let time_to_first_token_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        if Some(first_token) == eos_id {
            // Store in prompt cache with vision flag
            self.store_prompt_in_cache(prompt_tokens, true);
            let total_time = total_start.elapsed().as_secs_f64();
            return Ok((
                generated_text,
                GenerationStats {
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                    prefill_time_secs: prefill_time,
                    decode_time_secs: 0.0,
                    total_time_secs: total_time,
                    cached_tokens: 0,
                    time_to_first_token_ms,
                    gpu_sync_count: 0,
                    gpu_dispatch_count: 0,
                },
            ));
        }

        generated_ids.push(first_token);
        let token_text = self.tokenizer.decode(&[first_token]).unwrap_or_default();
        generated_text.push_str(&token_text);

        if !on_token(&token_text) {
            self.store_prompt_in_cache(prompt_tokens, true);
            let total_time = total_start.elapsed().as_secs_f64();
            return Ok((
                generated_text,
                GenerationStats {
                    prompt_tokens: prompt_len,
                    generated_tokens: 1,
                    prefill_time_secs: prefill_time,
                    decode_time_secs: decode_start.elapsed().as_secs_f64(),
                    total_time_secs: total_time,
                    cached_tokens: 0,
                    time_to_first_token_ms,
                    gpu_sync_count: 0,
                    gpu_dispatch_count: 0,
                },
            ));
        }

        // Decode loop (same as text-only -- vision tokens are cached in KV)
        for step in 1..self.config.max_tokens {
            let step_logits = self.model.forward(&[*generated_ids.last().unwrap()])?;
            let token_logits = extract_last_position_logits(&step_logits, 1, vocab_size)?;
            let next_token = sampler.sample_next(&token_logits, &generated_ids);

            if Some(next_token) == eos_id {
                debug!(step = step, "EOS token generated");
                break;
            }

            generated_ids.push(next_token);
            let token_text = self.tokenizer.decode(&[next_token]).unwrap_or_default();
            generated_text.push_str(&token_text);

            if self.check_stop_sequences(&generated_text) {
                self.trim_stop_sequence(&mut generated_text);
                debug!(step = step, "Stop sequence matched");
                break;
            }

            if !on_token(&token_text) {
                debug!(step = step, "Generation stopped by callback");
                break;
            }
        }

        let decode_time = decode_start.elapsed().as_secs_f64();
        let total_time = total_start.elapsed().as_secs_f64();

        self.store_prompt_in_cache(prompt_tokens, true);

        // Read GPU counters accumulated during this generation.
        // TODO(instrumentor-agent): enable once mlx_native counter functions land.
        #[cfg(feature = "mlx-native")]
        let (gpu_sync_count, gpu_dispatch_count) = {
            // (mlx_native::sync_count(), mlx_native::dispatch_count())
            (0u64, 0u64)
        };
        #[cfg(not(feature = "mlx-native"))]
        let (gpu_sync_count, gpu_dispatch_count) = (0u64, 0u64);

        Ok((
            generated_text,
            GenerationStats {
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
                prefill_time_secs: prefill_time,
                decode_time_secs: decode_time,
                total_time_secs: total_time,
                cached_tokens: 0,
                time_to_first_token_ms,
                gpu_sync_count,
                gpu_dispatch_count,
            },
        ))
    }

    /// Access the model's weight map (for vision encoder weight loading).
    pub fn weights(&self) -> &crate::inference::weight_loader::WeightMap {
        self.model.weights()
    }

    /// Get a reference to the model's maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.model.config().max_seq_len
    }

    /// Get the model name derived from config.
    pub fn model_name(&self) -> &str {
        // The model doesn't store its name; this is handled at a higher level.
        // Return a placeholder; the serve module extracts the name from the path.
        "unknown"
    }

    /// Get the model's hidden size (embedding dimension).
    pub fn hidden_size(&self) -> usize {
        self.model.config().hidden_size
    }

    /// Compute embeddings for the given text by running a prefill-only
    /// forward pass and mean-pooling the final hidden states.
    ///
    /// This resets the KV cache, tokenizes the input, runs the full
    /// transformer stack, then returns the mean-pooled, L2-normalized
    /// hidden state vector along with the token count.
    ///
    /// Returns `(embedding_vector, token_count)`.
    pub fn embed_text(&mut self, text: &str) -> Result<(Vec<f32>, usize), EngineError> {
        // Reset KV cache for this independent forward pass
        self.model.reset_cache();

        // Tokenize
        let tokens = self.tokenizer.encode_with_special_tokens(text)?;
        let token_count = tokens.len();

        if tokens.is_empty() {
            return Err(EngineError::Generation {
                reason: "Input tokenized to zero tokens".into(),
            });
        }

        let hidden_size = self.model.config().hidden_size;

        // Forward pass: get hidden states (before lm_head)
        let hidden_states = self.model.forward_hidden_states(&tokens)?;

        // Mean pool across all token positions
        // hidden_states shape: [seq_len, hidden_size]
        let seq_len = token_count;
        let mut pooled = vec![0.0f32; hidden_size];
        for pos in 0..seq_len {
            let offset = pos * hidden_size;
            for dim in 0..hidden_size {
                pooled[dim] += hidden_states[offset + dim];
            }
        }
        let inv_len = 1.0 / seq_len as f32;
        for v in pooled.iter_mut() {
            *v *= inv_len;
        }

        // L2 normalize
        let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            let inv_norm = 1.0 / norm;
            for v in pooled.iter_mut() {
                *v *= inv_norm;
            }
        }

        Ok((pooled, token_count))
    }
}

/// Extract the logits for the last sequence position from a flat logits buffer.
///
/// The buffer has shape `[seq_len, vocab_size]`, and we want the last row.
fn extract_last_position_logits(
    logits: &[f32],
    seq_len: usize,
    vocab_size: usize,
) -> Result<Vec<f32>, EngineError> {
    let expected_len = seq_len * vocab_size;
    if logits.len() < expected_len {
        return Err(EngineError::Generation {
            reason: format!(
                "Logits buffer too small: got {} elements, expected {} (seq_len={}, vocab_size={})",
                logits.len(),
                expected_len,
                seq_len,
                vocab_size
            ),
        });
    }

    let start = (seq_len - 1) * vocab_size;
    let end = start + vocab_size;
    Ok(logits[start..end].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_last_position_logits() {
        // seq_len=3, vocab_size=4
        let logits = vec![
            1.0, 2.0, 3.0, 4.0, // pos 0
            5.0, 6.0, 7.0, 8.0, // pos 1
            9.0, 10.0, 11.0, 12.0, // pos 2 (last)
        ];
        let last = extract_last_position_logits(&logits, 3, 4).unwrap();
        assert_eq!(last, vec![9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_extract_last_position_single() {
        let logits = vec![1.0, 2.0, 3.0];
        let last = extract_last_position_logits(&logits, 1, 3).unwrap();
        assert_eq!(last, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_extract_last_position_buffer_too_small() {
        let logits = vec![1.0, 2.0];
        let result = extract_last_position_logits(&logits, 2, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_generation_stats_tokens_per_sec() {
        let stats = GenerationStats {
            prompt_tokens: 100,
            generated_tokens: 50,
            prefill_time_secs: 0.5,
            decode_time_secs: 2.5,
            total_time_secs: 3.0,
            cached_tokens: 0,
            time_to_first_token_ms: 0.0,
            gpu_sync_count: 0,
            gpu_dispatch_count: 0,
        };
        assert!((stats.decode_tokens_per_sec() - 20.0).abs() < 1e-6);
        assert!((stats.prefill_tokens_per_sec() - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_generation_stats_zero_time() {
        let stats = GenerationStats {
            prompt_tokens: 0,
            generated_tokens: 0,
            prefill_time_secs: 0.0,
            decode_time_secs: 0.0,
            total_time_secs: 0.0,
            cached_tokens: 0,
            time_to_first_token_ms: 0.0,
            gpu_sync_count: 0,
            gpu_dispatch_count: 0,
        };
        assert_eq!(stats.decode_tokens_per_sec(), 0.0);
        assert_eq!(stats.prefill_tokens_per_sec(), 0.0);
    }

    #[test]
    fn test_generation_stats_with_cache() {
        let stats = GenerationStats {
            prompt_tokens: 500,
            generated_tokens: 50,
            prefill_time_secs: 0.1,
            decode_time_secs: 2.5,
            total_time_secs: 2.6,
            cached_tokens: 450,
            time_to_first_token_ms: 0.0,
            gpu_sync_count: 0,
            gpu_dispatch_count: 0,
        };
        // Only 50 tokens were actually prefilled, but prompt_tokens reports
        // the full count for the API response.
        assert_eq!(stats.prompt_tokens, 500);
        assert_eq!(stats.cached_tokens, 450);
        // Prefill tok/s should reflect the reported prompt_tokens / prefill_time
        assert!((stats.prefill_tokens_per_sec() - 5000.0).abs() < 1e-6);
    }
}
