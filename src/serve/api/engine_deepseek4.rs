//! Native DeepSeek-V4-Flash serving adapter.
//!
//! The worker owns one fixed-capacity cache and the exact token sequence
//! committed into it. Growing OpenAI transcripts reuse that live prefix and
//! prefill only the rendered suffix; divergent/compacted transcripts reset.

mod sampling;
mod stream;

pub use sampling::generate_once;
pub use stream::generate_stream;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use mlx_native::gguf::GgufFile;
use mlx_native::MlxBuffer;
use tokenizers::Tokenizer;

use crate::inference::models::deepseek4::{
    cache::{Deepseek4Cache, Deepseek4CacheSnapshot},
    tokenizer as deepseek_tokenizer, Deepseek4Model, FRESH_MATRIX_PREFILL_WINDOW_MULTIPLIER,
};
use crate::serve::load_info::{
    self, ArchFamily, ChatTemplateSource, LoadInfo, LoadInfoBuilder, MoeShape, TokenizerSource,
};

use super::engine::LoadOptions;

/// First practical OpenCode target: enough headroom for the ~100K-token
/// sessions already used by the repository's Qwen serving harness. Operators
/// can lower/raise it with `HF2Q_DEEPSEEK_MAX_SEQ_LEN`; model metadata remains
/// authoritative.
pub const DEFAULT_CONTEXT_LENGTH: usize = 131_072;
const RECOVERY_TAIL_TOKENS: usize = 8;
const MAX_CHEAP_INCREMENTAL_SUFFIX_TOKENS: usize = 32;
const MIN_MATRIX_REUSE_PREFIX_TOKENS: usize = 128;

pub struct Deepseek4LoadedModel {
    pub model: Deepseek4Model,
    pub cache: Deepseek4Cache,
    pub model_id: String,
    pub model_path: PathBuf,
    pub context_length: Option<usize>,
    pub quant_type: Option<String>,
    pub tokenizer: Tokenizer,
    pub chat_template: String,
    pub eos_token_ids: Vec<u32>,
    pub load_duration: Duration,
    pub provenance: crate::core::provenance::Provenance,
    /// Exact tokens currently committed into `cache`.
    committed_tokens: Vec<u32>,
    /// Logits following the last committed token, retained for an exact hit.
    live_logits: Option<MlxBuffer>,
    /// Cache state a small fixed tail before the end of a recent prompt. The
    /// official encoder can rewrite that tail when it drops old reasoning.
    turn_anchor: Option<Deepseek4CacheSnapshot>,
    turn_anchor_tokens: Vec<u32>,
}

impl Deepseek4LoadedModel {
    pub fn load(opts: &LoadOptions) -> Result<Self> {
        let started = Instant::now();
        let gguf = GgufFile::open(&opts.model_path)
            .map_err(|error| anyhow::anyhow!("GGUF open: {error}"))?;
        let tokenizer = if let Some(path) = opts.tokenizer_path.as_deref() {
            Tokenizer::from_file(path)
                .map_err(|error| anyhow::anyhow!("load tokenizer {}: {error}", path.display()))?
        } else {
            deepseek_tokenizer::build_tokenizer_from_gguf(&gguf)
                .context("build DeepSeek-V4 tokenizer from GGUF metadata")?
        };
        let path_model_id = || {
            opts.model_path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| "DeepSeek-V4-Flash-0731".to_string())
        };
        let model_id = gguf
            .metadata_string("general.name")
            // The accepted artifact's name is its source SHA. Prefer a
            // descriptive value or the stable GGUF stem exposed by models.
            .filter(|name| name.to_ascii_lowercase().contains("deepseek"))
            .map(ToOwned::to_owned)
            .unwrap_or_else(path_model_id);
        let quant_type = load_info::infer_quant_label(&gguf);
        let provenance = crate::core::provenance::detect(&gguf);
        let eos = gguf
            .metadata_u32("tokenizer.ggml.eos_token_id")
            .unwrap_or(1);
        let model =
            Deepseek4Model::load_from_gguf(&gguf).context("load native DeepSeek-V4 model")?;
        let requested_context = std::env::var("HF2Q_DEEPSEEK_MAX_SEQ_LEN")
            .ok()
            .map(|value| {
                value.parse::<usize>().with_context(|| {
                    format!("HF2Q_DEEPSEEK_MAX_SEQ_LEN must be a positive integer, got {value:?}")
                })
            })
            .transpose()?
            .unwrap_or(DEFAULT_CONTEXT_LENGTH);
        anyhow::ensure!(requested_context > 0, "DeepSeek-V4 context must be nonzero");
        let context_length = requested_context.min(model.cfg.max_position_embeddings as usize);
        anyhow::ensure!(
            context_length >= model.cfg.sliding_window as usize,
            "DeepSeek-V4 serving context {context_length} must be at least the {}-token native window",
            model.cfg.sliding_window
        );
        let cache = model
            .allocate_cache(context_length)
            .with_context(|| format!("allocate {context_length}-token DeepSeek-V4 cache"))?;

        Ok(Self {
            model,
            cache,
            model_id,
            model_path: opts.model_path.clone(),
            context_length: Some(context_length),
            quant_type,
            tokenizer,
            // Routes the shared renderer through the stateful native encoder.
            chat_template: crate::core::chat_templates::DEEPSEEK_V4_FLASH_0731.to_string(),
            eos_token_ids: vec![eos],
            load_duration: started.elapsed(),
            provenance,
            committed_tokens: Vec::new(),
            live_logits: None,
            turn_anchor: None,
            turn_anchor_tokens: Vec::new(),
        })
    }

    fn context_limit(&self) -> usize {
        self.context_length
            .unwrap_or(self.model.cfg.max_position_embeddings as usize)
    }

    fn reset_live_cache(&mut self) -> Result<()> {
        self.cache
            .reset()
            .context("reset DeepSeek-V4 serving cache")?;
        self.committed_tokens.clear();
        self.live_logits = None;
        self.turn_anchor = None;
        self.turn_anchor_tokens.clear();
        Ok(())
    }

    fn capture_turn_anchor(&mut self, prompt_prefix: &[u32]) -> Result<()> {
        // Release the old fixed-capacity copy before allocating its replacement.
        self.turn_anchor = None;
        self.turn_anchor_tokens.clear();
        let snapshot = self
            .cache
            .snapshot()
            .context("snapshot DeepSeek-V4 prompt-boundary cache")?;
        anyhow::ensure!(
            snapshot.position() == prompt_prefix.len(),
            "DeepSeek-V4 prompt anchor position {} does not match {} rendered tokens",
            snapshot.position(),
            prompt_prefix.len()
        );
        self.turn_anchor = Some(snapshot);
        self.turn_anchor_tokens.extend_from_slice(prompt_prefix);
        Ok(())
    }

    fn append_prompt_tokens(
        &mut self,
        tokens: &[u32],
        cancelled: &impl Fn() -> bool,
    ) -> Result<Option<MlxBuffer>> {
        if tokens.is_empty() {
            return Ok(None);
        }
        let mut state = None;
        let mut offset = 0;
        if self.cache.position() == 0 {
            let batch = tokens.len().min(
                self.model.cfg.sliding_window as usize * FRESH_MATRIX_PREFILL_WINDOW_MULTIPLIER,
            );
            anyhow::ensure!(
                !cancelled(),
                "DeepSeek-V4 generation cancelled during prefill"
            );
            state = Some(
                self.model
                    .forward_verifier_prefill(&tokens[..batch], &mut self.cache)
                    .context("execute DeepSeek-V4 fresh-prompt matrix prefill")?,
            );
            self.committed_tokens.extend_from_slice(&tokens[..batch]);
            offset = batch;
        }
        for &token in &tokens[offset..] {
            anyhow::ensure!(
                !cancelled(),
                "DeepSeek-V4 generation cancelled during prefill"
            );
            state = Some(
                self.model
                    .forward_verifier_one(token, &mut self.cache)
                    .context("execute DeepSeek-V4 cached-prefix append token")?,
            );
            self.committed_tokens.push(token);
        }
        Ok(state)
    }

    fn prefill_suffix(
        &mut self,
        prompt_tokens: &[u32],
        cancelled: impl Fn() -> bool,
    ) -> Result<(MlxBuffer, usize)> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "DeepSeek-V4 prompt is empty");
        anyhow::ensure!(
            prompt_tokens.len() <= self.context_limit(),
            "DeepSeek-V4 prompt has {} tokens, exceeding serving context {}",
            prompt_tokens.len(),
            self.context_limit()
        );

        let reuse = prefer_matrix_prefill(
            prompt_tokens.len(),
            select_prefix_reuse(
                prompt_tokens,
                &self.committed_tokens,
                if self.cache.is_poisoned() {
                    usize::MAX
                } else {
                    self.cache.position()
                },
                &self.turn_anchor_tokens,
                self.turn_anchor
                    .as_ref()
                    .map_or(0, Deepseek4CacheSnapshot::position),
            ),
        );
        let reset_from_scratch = matches!(reuse, PrefixReuse::Reset);
        let cached_tokens = match reuse {
            PrefixReuse::Live(count) => count,
            PrefixReuse::RecoveryAnchor(count) => {
                let snapshot = self
                    .turn_anchor
                    .as_ref()
                    .context("DeepSeek-V4 turn anchor disappeared before restore")?;
                self.cache
                    .restore(snapshot)
                    .context("restore DeepSeek-V4 prompt-boundary cache")?;
                self.committed_tokens.clone_from(&self.turn_anchor_tokens);
                self.live_logits = None;
                count
            }
            PrefixReuse::Reset => {
                self.reset_live_cache()?;
                0
            }
        };
        if prompt_tokens.len() == cached_tokens {
            return self
                .live_logits
                .clone()
                .map(|logits| (logits, cached_tokens))
                .context("DeepSeek-V4 exact-prefix cache hit has no aligned live logits");
        }

        let recovery_position = prompt_tokens.len().saturating_sub(RECOVERY_TAIL_TOKENS);
        let native_prefill_rows = self.model.cfg.sliding_window as usize;

        // Short position-zero prompts must execute as one verifier batch.
        // Build their small recovery checkpoint in a separate prepass.
        if reset_from_scratch && recovery_position > 0 && recovery_position < native_prefill_rows {
            self.append_prompt_tokens(&prompt_tokens[..recovery_position], &cancelled)?;
            self.capture_turn_anchor(&prompt_tokens[..recovery_position])?;
            self.cache
                .reset()
                .context("reset DeepSeek-V4 live cache after recovery prepass")?;
            self.committed_tokens.clear();
            self.live_logits = None;

            let final_state = self
                .append_prompt_tokens(prompt_tokens, &cancelled)?
                .context("DeepSeek-V4 prompt produced no verifier state")?;
            return self.finish_prefill(final_state, 0);
        }

        let mut cursor = cached_tokens;
        let mut final_state = None;
        if recovery_position > cursor {
            final_state =
                self.append_prompt_tokens(&prompt_tokens[cursor..recovery_position], &cancelled)?;
            cursor = recovery_position;
            self.capture_turn_anchor(&prompt_tokens[..recovery_position])?;
        }
        final_state = self
            .append_prompt_tokens(&prompt_tokens[cursor..], &cancelled)?
            .or(final_state);
        self.finish_prefill(
            final_state.context("DeepSeek-V4 suffix produced no verifier state")?,
            cached_tokens,
        )
    }

    fn finish_prefill(
        &mut self,
        state: MlxBuffer,
        cached_tokens: usize,
    ) -> Result<(MlxBuffer, usize)> {
        let last = self
            .model
            .last_token_state(&state)
            .context("view DeepSeek-V4 last prompt state")?;
        let logits = self
            .model
            .forward_logits(&last)
            .context("execute DeepSeek-V4 prompt output head")?;
        self.live_logits = Some(logits.clone());
        Ok((logits, cached_tokens))
    }

    fn commit_generated_token(&mut self, token: u32) -> Result<MlxBuffer> {
        let state = self
            .model
            .forward_verifier_one(token, &mut self.cache)
            .with_context(|| {
                format!(
                    "execute DeepSeek-V4 decode token at cache position {}",
                    self.cache.position()
                )
            })?;
        self.committed_tokens.push(token);
        let logits = self
            .model
            .forward_logits(&state)
            .context("execute DeepSeek-V4 decode output head")?;
        self.live_logits = Some(logits.clone());
        Ok(logits)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefixReuse {
    Live(usize),
    RecoveryAnchor(usize),
    Reset,
}

fn prefer_matrix_prefill(prompt_len: usize, reuse: PrefixReuse) -> PrefixReuse {
    let cached = match reuse {
        PrefixReuse::Live(count) | PrefixReuse::RecoveryAnchor(count) => count,
        PrefixReuse::Reset => return PrefixReuse::Reset,
    };
    let suffix = prompt_len.saturating_sub(cached);
    if suffix > MAX_CHEAP_INCREMENTAL_SUFFIX_TOKENS && cached < MIN_MATRIX_REUSE_PREFIX_TOKENS {
        PrefixReuse::Reset
    } else {
        reuse
    }
}

fn select_prefix_reuse(
    prompt: &[u32],
    live: &[u32],
    live_position: usize,
    anchor: &[u32],
    anchor_position: usize,
) -> PrefixReuse {
    if !live.is_empty() && live_position == live.len() && prompt.starts_with(live) {
        PrefixReuse::Live(live.len())
    } else if !anchor.is_empty()
        && anchor_position == anchor.len()
        && prompt.len() > anchor.len()
        && prompt.starts_with(anchor)
    {
        PrefixReuse::RecoveryAnchor(anchor.len())
    } else {
        PrefixReuse::Reset
    }
}

impl LoadInfoBuilder for Deepseek4LoadedModel {
    fn build_load_info(
        &self,
        gguf: &GgufFile,
        load_wall_clock: Duration,
        kv_cache_budget_bytes: Option<u64>,
        kv_spill_active: bool,
    ) -> LoadInfo {
        LoadInfo {
            model_id: self.model_id.clone(),
            arch_str: load_info::arch_str_from_gguf(gguf),
            arch_family: ArchFamily::Deepseek4,
            model_path: self.model_path.clone(),
            on_disk_bytes: load_info::on_disk_bytes(&self.model_path),
            backend_chip: self.model.ctx.gpu_name(),
            backend: "mlx-native",
            n_layers: self.model.cfg.num_hidden_layers,
            hidden_size: self.model.cfg.hidden_size,
            vocab_size: self.model.cfg.vocab_size,
            n_attention_heads: self.model.cfg.num_attention_heads,
            n_key_value_heads: self.model.cfg.num_key_value_heads,
            head_dim: self.model.cfg.head_dim,
            sliding_window: Some(self.model.cfg.sliding_window),
            full_attention_interval: None,
            max_context_length: self.context_length.map(|value| value as u32),
            moe: Some(MoeShape {
                n_experts: self.model.cfg.num_experts,
                n_experts_per_tok: self.model.cfg.num_experts_per_tok,
            }),
            quant_label: self.quant_type.clone(),
            quant_bpw: load_info::compute_bpw(gguf),
            tokenizer_source: TokenizerSource::GgufEmbedded,
            eos_token_ids: self.eos_token_ids.clone(),
            bos_token_id: gguf.metadata_u32("tokenizer.ggml.bos_token_id"),
            chat_template_source: ChatTemplateSource::NativeEncoding {
                name: "DEEPSEEK_V4_FLASH_0731",
            },
            provenance: self.provenance.clone(),
            vision_projector: None,
            load_wall_clock,
            resident_weight_bytes: Some(self.model.weights.resident_bytes()),
            kv_cache_budget_bytes,
            kv_spill_active,
            tq_kv_active: false,
            // Live cache + one equally sized recovery snapshot.
            kv_bytes_per_token_override: Some(13_760),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{prefer_matrix_prefill, select_prefix_reuse, PrefixReuse};

    #[test]
    fn growing_transcript_prefers_longest_live_prefix() {
        assert_eq!(
            select_prefix_reuse(&[1, 2, 3, 4, 5], &[1, 2, 3, 4], 4, &[1, 2], 2),
            PrefixReuse::Live(4)
        );
    }

    #[test]
    fn canonical_reasoning_drop_restores_pre_answer_turn_anchor() {
        assert_eq!(
            select_prefix_reuse(&[1, 2, 7, 8], &[1, 2, 9, 7], 4, &[1, 2], 2),
            PrefixReuse::RecoveryAnchor(2)
        );
    }

    #[test]
    fn compaction_or_edit_resets_when_neither_exact_prefix_matches() {
        assert_eq!(
            select_prefix_reuse(&[4, 5, 6], &[1, 2, 3], 3, &[1, 2], 2),
            PrefixReuse::Reset
        );
    }

    #[test]
    fn trivial_prefix_reuse_yields_to_full_matrix_prefill() {
        assert_eq!(
            prefer_matrix_prefill(306, PrefixReuse::RecoveryAnchor(1)),
            PrefixReuse::Reset
        );
        assert_eq!(
            prefer_matrix_prefill(516, PrefixReuse::Live(16)),
            PrefixReuse::Reset
        );
    }

    #[test]
    fn short_suffix_and_meaningful_prefix_still_reuse_cache() {
        assert_eq!(
            prefer_matrix_prefill(48, PrefixReuse::Live(37)),
            PrefixReuse::Live(37)
        );
        assert_eq!(
            prefer_matrix_prefill(180, PrefixReuse::RecoveryAnchor(140)),
            PrefixReuse::RecoveryAnchor(140)
        );
    }
}
