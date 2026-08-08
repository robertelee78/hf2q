//! Native DeepSeek-V4-Flash serving adapter.
//!
//! The worker owns one demand-grown cache and the exact token sequence
//! committed into it. Growing OpenAI transcripts reuse that live prefix across
//! capacity changes and prefill only the rendered suffix;
//! divergent/compacted transcripts reset.

mod progress;
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
    decode_scratch_stats, matrix_prefill_chunk_len, prefill_scratch_stats, release_decode_scratch,
    release_prefill_scratch, tokenizer as deepseek_tokenizer, Deepseek4Model,
    TransientScratchStats,
};
use crate::serve::load_info::{
    self, ArchFamily, ChatTemplateSource, LoadInfo, LoadInfoBuilder, MoeShape, TokenizerSource,
};

use self::progress::RequestProgress;
use super::engine::LoadOptions;

/// First practical OpenCode target: enough headroom for the ~100K-token
/// sessions already used by the repository's Qwen serving harness. Operators
/// can lower/raise it with `HF2Q_DEEPSEEK_MAX_SEQ_LEN`; model metadata remains
/// authoritative.
pub const DEFAULT_CONTEXT_LENGTH: usize = 131_072;
const INITIAL_CACHE_LENGTH: usize = 131_072;
const RECOVERY_TAIL_TOKENS: usize = 8;
const MAX_CHEAP_INCREMENTAL_SUFFIX_TOKENS: usize = 32;
const MIN_MATRIX_REUSE_PREFIX_TOKENS: usize = 128;
/// Preserve a modest hot decode arena across successful agent turns, but do
/// not leave a multi-gigabyte working set idle beside a 100 GiB model.
const MAX_RETAINED_DECODE_SCRATCH_BYTES: usize = 256 * 1024 * 1024;

fn balance_fresh_three_chunk_prefill(
    cache_position: usize,
    token_count: usize,
    sliding_window: usize,
    default_windows: usize,
) -> usize {
    if cache_position != 0 || sliding_window == 0 || default_windows == 0 {
        return default_windows;
    }
    let default_chunk = sliding_window.saturating_mul(default_windows);
    if token_count <= default_chunk.saturating_mul(2)
        || token_count >= default_chunk.saturating_mul(3)
    {
        return default_windows;
    }
    token_count
        .div_ceil(3)
        .div_ceil(sliding_window)
        .clamp(1, default_windows)
}

fn should_retain_decode_scratch(stats: TransientScratchStats) -> bool {
    stats.free_bytes <= MAX_RETAINED_DECODE_SCRATCH_BYTES
}

fn log_scratch(action: &'static str, phase: &'static str, stats: TransientScratchStats) {
    if stats.free_buffers == 0 && stats.free_bytes == 0 {
        return;
    }
    tracing::info!(
        action,
        phase,
        buffers = stats.free_buffers,
        bytes = stats.free_bytes,
        "DeepSeek-V4 transient scratch"
    );
}

pub(super) fn release_completed_prefill_scratch() {
    let stats = prefill_scratch_stats();
    if stats.free_bytes > 0 {
        log_scratch("release", "prefill-complete", release_prefill_scratch());
    }
}

/// Fail-safe cleanup for early returns. Successful requests keep a bounded
/// decode arena hot; failed/cancelled requests release both arenas.
pub(super) struct RequestScratchGuard {
    completed: bool,
}

impl RequestScratchGuard {
    pub(super) fn new() -> Self {
        Self { completed: false }
    }

    pub(super) fn complete(mut self) {
        release_completed_prefill_scratch();
        let decode = decode_scratch_stats();
        if !should_retain_decode_scratch(decode) {
            log_scratch("release", "decode-retention-cap", release_decode_scratch());
        } else {
            log_scratch("retain", "decode-hot", decode);
        }
        self.completed = true;
    }
}

impl Drop for RequestScratchGuard {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        log_scratch(
            "release",
            "request-abort-prefill",
            release_prefill_scratch(),
        );
        log_scratch("release", "request-abort-decode", release_decode_scratch());
    }
}

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
        let mut model =
            Deepseek4Model::load_from_gguf(&gguf).context("load native DeepSeek-V4 model")?;
        tracing::info!(
            logical_weight_bytes = model.weights.resident_bytes(),
            file_backed_weight_bytes = model.weights.file_backed_bytes(),
            anonymous_weight_bytes = model.weights.anonymous_bytes(),
            mapped_weight_segments = model.weights.mapped_segment_count(),
            "DeepSeek-V4 weight residency established"
        );
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
        let initial_cache_length = context_length.min(INITIAL_CACHE_LENGTH);
        let mut cache = model
            .allocate_cache(initial_cache_length)
            .with_context(|| {
                format!("allocate initial {initial_cache_length}-token DeepSeek-V4 cache")
            })?;
        tracing::info!(
            serving_context = context_length,
            allocated_cache_context = initial_cache_length,
            "DeepSeek-V4 cache admitted with demand-grown capacity"
        );

        // Metal pipeline creation and the first page-in of the 43 verifier
        // layers otherwise happen after the server has announced readiness,
        // adding several silent seconds to the first OpenCode turn. Exercise
        // an aligned matrix prefill and the output head before becoming ready,
        // then discard every byte of logical conversation state.
        if std::env::var("HF2Q_DEEPSEEK_SKIP_WARMUP").as_deref() != Ok("1") {
            let warm_tokens = vec![eos; 64];
            let warm_state = model
                .forward_verifier_prefill(&warm_tokens, &mut cache)
                .context("warm DeepSeek-V4 matrix prefill")?;
            let warm_last = model
                .last_token_state(&warm_state)
                .context("view DeepSeek-V4 warmup state")?;
            model
                .forward_logits(&warm_last)
                .context("warm DeepSeek-V4 output head")?;
            cache
                .reset()
                .context("reset DeepSeek-V4 cache after warmup")?;
            log_scratch("release", "startup-warmup", release_prefill_scratch());
        }

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

    fn ensure_cache_capacity(&mut self, prompt_tokens: usize, max_tokens: usize) -> Result<bool> {
        let target = cache_capacity_for_request(
            self.context_limit(),
            self.cache.capacity(),
            prompt_tokens,
            max_tokens,
        );
        if target <= self.cache.capacity() {
            return Ok(false);
        }

        let previous = self.cache.capacity();
        let migration_started = Instant::now();
        let source_cache_bytes = self.cache.resident_bytes();
        let migrated_tokens = self.cache.position();
        let mut grown = self.model.allocate_cache(target).with_context(|| {
            format!("grow DeepSeek-V4 cache from {previous} to {target} tokens")
        })?;
        grown
            .migrate_from(&self.cache, self.turn_anchor.as_mut())
            .with_context(|| {
                format!("migrate DeepSeek-V4 live cache from {previous} to {target} tokens")
            })?;
        let target_cache_bytes = grown.resident_bytes();
        let old = std::mem::replace(&mut self.cache, grown);
        drop(old);
        tracing::info!(
            previous_cache_context = previous,
            allocated_cache_context = target,
            serving_context = self.context_limit(),
            migrated_tokens,
            source_cache_bytes,
            target_cache_bytes,
            migration_elapsed_ms = migration_started.elapsed().as_secs_f64() * 1000.0,
            "DeepSeek-V4 cache capacity grown without prefix reset"
        );
        Ok(true)
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
        progress: &mut RequestProgress,
    ) -> Result<Option<MlxBuffer>> {
        if tokens.is_empty() {
            return Ok(None);
        }
        let mut state = None;
        let mut offset = 0;
        let explicit_prefill_windows = std::env::var_os("HF2Q_DEEPSEEK_PREFILL_WINDOWS").is_some();
        let mut window_multiplier = self.model.matrix_prefill_window_multiplier()?;
        if !explicit_prefill_windows && self.cache.capacity() > INITIAL_CACHE_LENGTH {
            // A grown context-linear cache leaves less unified-memory scratch
            // than the initial 131K allocation. Preserve the measured 2K
            // transaction for ordinary agent sessions, but halve it after a
            // real request crosses the first capacity boundary.
            window_multiplier = window_multiplier.min(8);
        } else if !explicit_prefill_windows {
            // A fresh prompt just above two full transactions otherwise leaves
            // a severely underfilled third matrix (4,987 -> 2,048/2,048/891).
            // Keep the same three transactions but balance their row counts.
            // Longer OpenCode contexts retain the measured 2,048-token path.
            window_multiplier = balance_fresh_three_chunk_prefill(
                self.cache.position(),
                tokens.len(),
                self.model.cfg.sliding_window as usize,
                window_multiplier,
            );
        }
        while offset < tokens.len() {
            let batch = matrix_prefill_chunk_len(
                self.cache.position(),
                tokens.len() - offset,
                self.model.cfg.sliding_window as usize,
                window_multiplier,
            );
            if batch == 0 {
                break;
            }
            anyhow::ensure!(
                !cancelled(),
                "DeepSeek-V4 generation cancelled during prefill"
            );
            state = Some(
                self.model
                    .forward_verifier_prefill(&tokens[offset..offset + batch], &mut self.cache)
                    .context("execute DeepSeek-V4 matrix prefill chunk")?,
            );
            self.committed_tokens
                .extend_from_slice(&tokens[offset..offset + batch]);
            offset += batch;
            progress.advance_prefill(batch);
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
            progress.advance_prefill(1);
        }
        Ok(state)
    }

    fn prefill_suffix(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        cancelled: impl Fn() -> bool,
        progress: &mut RequestProgress,
    ) -> Result<(MlxBuffer, usize)> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "DeepSeek-V4 prompt is empty");
        anyhow::ensure!(
            prompt_tokens.len() <= self.context_limit(),
            "DeepSeek-V4 prompt has {} tokens, exceeding serving context {}",
            prompt_tokens.len(),
            self.context_limit()
        );
        let cache_grew = self.ensure_cache_capacity(prompt_tokens.len(), max_tokens)?;

        let cache_poisoned = self.cache.is_poisoned();
        let live_position = self.cache.position();
        let selectable_live_position = if cache_poisoned {
            usize::MAX
        } else {
            live_position
        };
        let anchor_position = self
            .turn_anchor
            .as_ref()
            .map_or(0, Deepseek4CacheSnapshot::position);
        let selected_reuse = select_prefix_reuse(
            prompt_tokens,
            &self.committed_tokens,
            selectable_live_position,
            &self.turn_anchor_tokens,
            anchor_position,
        );
        let reuse = prefer_matrix_prefill(prompt_tokens.len(), selected_reuse);
        if matches!(reuse, PrefixReuse::Reset) {
            progress.cache_reset_diagnostic(
                self.committed_tokens.len(),
                live_position,
                common_prefix_len(prompt_tokens, &self.committed_tokens),
                self.turn_anchor_tokens.len(),
                anchor_position,
                common_prefix_len(prompt_tokens, &self.turn_anchor_tokens),
                cache_poisoned,
                selected_reuse != reuse,
                cache_grew,
            );
        }
        let reset_from_scratch = matches!(reuse, PrefixReuse::Reset);
        let (cached_tokens, cache_action) = match reuse {
            PrefixReuse::Live(count) => (count, if cache_grew { "grow-live" } else { "live" }),
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
                (
                    count,
                    if cache_grew {
                        "grow-recovery-anchor"
                    } else {
                        "recovery-anchor"
                    },
                )
            }
            PrefixReuse::Reset => {
                self.reset_live_cache()?;
                (0, if cache_grew { "grow-reset" } else { "reset" })
            }
        };
        let recovery_position = prompt_tokens.len().saturating_sub(RECOVERY_TAIL_TOKENS);
        let native_prefill_rows = self.model.cfg.sliding_window as usize;
        let short_recovery_prepass =
            reset_from_scratch && recovery_position > 0 && recovery_position < native_prefill_rows;
        let work_tokens = prompt_tokens
            .len()
            .saturating_sub(cached_tokens)
            .saturating_add(
                short_recovery_prepass
                    .then_some(recovery_position)
                    .unwrap_or(0),
            );
        progress.plan_prefill(cached_tokens, work_tokens, cache_action);
        if prompt_tokens.len() == cached_tokens {
            return self
                .live_logits
                .clone()
                .map(|logits| (logits, cached_tokens))
                .context("DeepSeek-V4 exact-prefix cache hit has no aligned live logits");
        }

        // Short position-zero prompts must execute as one verifier batch.
        // Build their small recovery checkpoint in a separate prepass.
        if short_recovery_prepass {
            self.append_prompt_tokens(&prompt_tokens[..recovery_position], &cancelled, progress)?;
            self.capture_turn_anchor(&prompt_tokens[..recovery_position])?;
            self.cache
                .reset()
                .context("reset DeepSeek-V4 live cache after recovery prepass")?;
            self.committed_tokens.clear();
            self.live_logits = None;

            let final_state = self
                .append_prompt_tokens(prompt_tokens, &cancelled, progress)?
                .context("DeepSeek-V4 prompt produced no verifier state")?;
            return self.finish_prefill(final_state, 0);
        }

        let mut cursor = cached_tokens;
        let mut final_state = None;
        if recovery_position > cursor {
            final_state = self.append_prompt_tokens(
                &prompt_tokens[cursor..recovery_position],
                &cancelled,
                progress,
            )?;
            cursor = recovery_position;
            self.capture_turn_anchor(&prompt_tokens[..recovery_position])?;
        }
        final_state = self
            .append_prompt_tokens(&prompt_tokens[cursor..], &cancelled, progress)?
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

fn cache_capacity_for_request(
    serving_context: usize,
    current_capacity: usize,
    prompt_tokens: usize,
    max_tokens: usize,
) -> usize {
    let required = prompt_tokens
        .saturating_add(max_tokens.max(1).saturating_sub(1))
        .min(serving_context);
    if required <= current_capacity {
        return current_capacity;
    }
    required
        .div_ceil(INITIAL_CACHE_LENGTH)
        .saturating_mul(INITIAL_CACHE_LENGTH)
        .min(serving_context)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefixReuse {
    Live(usize),
    RecoveryAnchor(usize),
    Reset,
}

fn common_prefix_len(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(left, right)| left == right)
        .count()
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
            // Context-linear live cache. Recovery rollback is a fixed ~17 MiB.
            kv_bytes_per_token_override: Some(6_880),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        balance_fresh_three_chunk_prefill, cache_capacity_for_request, common_prefix_len,
        prefer_matrix_prefill, select_prefix_reuse, should_retain_decode_scratch, PrefixReuse,
        TransientScratchStats,
    };

    #[test]
    fn fresh_three_chunk_prefill_balances_only_the_underfilled_band() {
        assert_eq!(balance_fresh_three_chunk_prefill(0, 4_987, 128, 16), 13);
        assert_eq!(balance_fresh_three_chunk_prefill(0, 4_096, 128, 16), 16);
        assert_eq!(balance_fresh_three_chunk_prefill(0, 6_144, 128, 16), 16);
        assert_eq!(balance_fresh_three_chunk_prefill(0, 119_821, 128, 16), 16);
        assert_eq!(balance_fresh_three_chunk_prefill(1, 4_987, 128, 16), 16);
    }

    #[test]
    fn decode_scratch_retention_is_bounded() {
        assert!(should_retain_decode_scratch(TransientScratchStats {
            free_buffers: 1,
            free_bytes: 256 * 1024 * 1024,
        }));
        assert!(!should_retain_decode_scratch(TransientScratchStats {
            free_buffers: 1,
            free_bytes: 256 * 1024 * 1024 + 1,
        }));
    }

    #[test]
    fn advertised_context_does_not_eagerly_allocate_unused_kv() {
        assert_eq!(
            cache_capacity_for_request(524_288, 131_072, 98_000, 8_192),
            131_072
        );
        assert_eq!(
            cache_capacity_for_request(524_288, 131_072, 131_000, 8_192),
            262_144
        );
        assert_eq!(
            cache_capacity_for_request(524_288, 262_144, 510_000, 32_768),
            524_288
        );
    }

    #[test]
    fn growing_transcript_prefers_longest_live_prefix() {
        assert_eq!(
            select_prefix_reuse(&[1, 2, 3, 4, 5], &[1, 2, 3, 4], 4, &[1, 2], 2),
            PrefixReuse::Live(4)
        );
    }

    #[test]
    fn reset_diagnostics_measure_the_exact_divergence_point() {
        assert_eq!(common_prefix_len(&[1, 2, 3, 4], &[1, 2, 9, 4]), 2);
        assert_eq!(common_prefix_len(&[1, 2], &[1, 2, 3]), 2);
        assert_eq!(common_prefix_len(&[1], &[2]), 0);
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
