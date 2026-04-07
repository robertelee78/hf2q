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

use std::path::Path;
use std::time::Instant;

use thiserror::Error;
use tracing::{debug, info};

use crate::inference::models::gemma4::{Gemma4Config, Gemma4Error, Gemma4Model};
use crate::inference::models::registry;
use crate::inference::sampler::{Sampler, SamplerConfig};
use crate::inference::weight_loader;
use crate::tokenizer::chat_template::{ChatTemplate, ChatTemplateError, Message};
use crate::tokenizer::{HfTokenizer, TokenizerError};

use mlx_native::MlxDevice;

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
}

impl InferenceEngine {
    /// Build a new inference engine from a model directory and configuration.
    ///
    /// Loads the model weights, tokenizer, and chat template. The `chat_template_override`
    /// parameter allows overriding the model's built-in template with a CLI-specified file.
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
        })
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

        // Reset KV cache for the new sequence
        self.model.reset_cache();

        // Step 1: Render the chat template
        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        let bos_token = self
            .tokenizer
            .bos_id()
            .and_then(|id| self.tokenizer.decode(&[id]).ok())
            .unwrap_or_default();
        let eos_token = self
            .tokenizer
            .eos_id()
            .and_then(|id| self.tokenizer.decode(&[id]).ok())
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

        // Step 3: Prefill — forward pass on all prompt tokens
        let prefill_start = Instant::now();
        let prefill_logits = self.model.forward(&prompt_tokens)?;
        let prefill_time = prefill_start.elapsed().as_secs_f64();

        let vocab_size = self.model.config().vocab_size;

        // Extract logits for the last prompt position
        let last_pos_logits = extract_last_position_logits(&prefill_logits, prompt_len, vocab_size)?;

        // Step 4: Create sampler and sample first token
        let mut sampler = Sampler::new(self.config.sampler.clone());
        let mut generated_ids: Vec<u32> = Vec::with_capacity(self.config.max_tokens);
        let mut generated_text = String::new();

        let eos_id = self.tokenizer.eos_id();

        let decode_start = Instant::now();

        let first_token = sampler.sample_next(&last_pos_logits, &generated_ids);

        // Check stop conditions for first token
        if Some(first_token) == eos_id {
            let total_time = total_start.elapsed().as_secs_f64();
            return Ok((
                generated_text,
                GenerationStats {
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                    prefill_time_secs: prefill_time,
                    decode_time_secs: 0.0,
                    total_time_secs: total_time,
                },
            ));
        }

        generated_ids.push(first_token);
        let token_text = self.tokenizer.decode(&[first_token]).unwrap_or_default();
        generated_text.push_str(&token_text);

        // Stream the first token
        if !on_token(&token_text) {
            let total_time = total_start.elapsed().as_secs_f64();
            return Ok((
                generated_text,
                GenerationStats {
                    prompt_tokens: prompt_len,
                    generated_tokens: 1,
                    prefill_time_secs: prefill_time,
                    decode_time_secs: decode_start.elapsed().as_secs_f64(),
                    total_time_secs: total_time,
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

        let decode_time = decode_start.elapsed().as_secs_f64();
        let total_time = total_start.elapsed().as_secs_f64();

        let stats = GenerationStats {
            prompt_tokens: prompt_len,
            generated_tokens: generated_ids.len(),
            prefill_time_secs: prefill_time,
            decode_time_secs: decode_time,
            total_time_secs: total_time,
        };

        Ok((generated_text, stats))
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

        // Reset KV cache for the new sequence
        self.model.reset_cache();

        // Tokenize the already-rendered prompt
        let prompt_tokens = self.tokenizer.encode_with_special_tokens(rendered_prompt)?;
        let prompt_len = prompt_tokens.len();
        debug!(prompt_tokens = prompt_len, "Prompt tokenized");

        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation {
                reason: "Prompt tokenized to zero tokens".into(),
            });
        }

        // Prefill
        let prefill_start = Instant::now();
        let prefill_logits = self.model.forward(&prompt_tokens)?;
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

        let first_token = sampler.sample_next(&last_pos_logits, &generated_ids);

        if Some(first_token) == eos_id {
            let total_time = total_start.elapsed().as_secs_f64();
            return Ok((
                generated_text,
                GenerationStats {
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                    prefill_time_secs: prefill_time,
                    decode_time_secs: 0.0,
                    total_time_secs: total_time,
                },
            ));
        }

        generated_ids.push(first_token);
        let token_text = self.tokenizer.decode(&[first_token]).unwrap_or_default();
        generated_text.push_str(&token_text);

        if !on_token(&token_text) {
            let total_time = total_start.elapsed().as_secs_f64();
            return Ok((
                generated_text,
                GenerationStats {
                    prompt_tokens: prompt_len,
                    generated_tokens: 1,
                    prefill_time_secs: prefill_time,
                    decode_time_secs: decode_start.elapsed().as_secs_f64(),
                    total_time_secs: total_time,
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

        let decode_time = decode_start.elapsed().as_secs_f64();
        let total_time = total_start.elapsed().as_secs_f64();

        Ok((
            generated_text,
            GenerationStats {
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
                prefill_time_secs: prefill_time,
                decode_time_secs: decode_time,
                total_time_secs: total_time,
            },
        ))
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
        };
        assert_eq!(stats.decode_tokens_per_sec(), 0.0);
        assert_eq!(stats.prefill_tokens_per_sec(), 0.0);
    }
}
