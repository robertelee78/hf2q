//! Native DeepSeek-V4-Flash serving adapter.
//!
//! The worker owns one demand-grown cache and the exact token sequence
//! committed into it. Growing OpenAI transcripts reuse that live prefix across
//! capacity changes and prefill only the rendered suffix;
//! divergent/compacted transcripts reset.

mod progress;
mod sampling;
mod slots;
mod stream;

pub(super) use sampling::generate_once;
pub(super) use slots::{
    Deepseek4CooperativePrefillPlan, Deepseek4PrefillAdvance, Deepseek4PrefillState,
    Deepseek4SlotCompletion, Deepseek4SlotState,
};
pub(super) use stream::generate_stream;

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
    TransientScratchStats, MIN_MATRIX_APPEND_TOKENS,
};
use crate::serve::load_info::{
    self, ArchFamily, ChatTemplateSource, LoadInfo, LoadInfoBuilder, MoeShape, TokenizerSource,
};

use self::progress::RequestProgress;
use super::engine::LoadOptions;
use super::engine_supervisor::EngineSupervisor;

/// First practical OpenCode target: enough headroom for the ~100K-token
/// sessions already used by the repository's Qwen serving harness. Operators
/// can lower/raise it with `HF2Q_DEEPSEEK_MAX_SEQ_LEN`; model metadata remains
/// authoritative.
pub const DEFAULT_CONTEXT_LENGTH: usize = 131_072;
const INITIAL_CACHE_LENGTH: usize = 131_072;
const RECOVERY_TAIL_TOKENS: usize = 8;

fn resumable_matrix_prefill_chunk_len(
    cache_position: usize,
    remaining: usize,
    sliding_window: usize,
    window_multiplier: usize,
) -> usize {
    let mut chunk =
        matrix_prefill_chunk_len(cache_position, remaining, sliding_window, window_multiplier);
    if remaining > chunk {
        let tail = remaining - chunk;
        if tail > RECOVERY_TAIL_TOKENS && tail < MIN_MATRIX_APPEND_TOKENS {
            chunk = chunk.saturating_sub(MIN_MATRIX_APPEND_TOKENS - tail);
        }
    }
    chunk
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResumablePrefillChunk {
    Matrix(usize),
    Incremental(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResumablePrefillSlice {
    chunk: ResumablePrefillChunk,
    capture_anchor_after: bool,
}

fn plan_resumable_prefill_chunk(
    cache_position: usize,
    remaining: usize,
    sliding_window: usize,
    window_multiplier: usize,
    recovery_position: usize,
) -> Result<ResumablePrefillSlice> {
    anyhow::ensure!(
        remaining > 0,
        "DeepSeek-V4 resumable prefill made no progress"
    );
    anyhow::ensure!(
        sliding_window > 0 && window_multiplier > 0,
        "DeepSeek-V4 resumable prefill has an invalid window plan (window={}, multiplier={})",
        sliding_window,
        window_multiplier
    );
    let batch = resumable_matrix_prefill_chunk_len(
        cache_position,
        remaining,
        sliding_window,
        window_multiplier,
    );
    let chunk = if batch > 0 {
        ResumablePrefillChunk::Matrix(batch)
    } else {
        anyhow::ensure!(
            remaining < MIN_MATRIX_APPEND_TOKENS,
            "DeepSeek-V4 resumable prefill selected an empty chunk at cursor {} with {} tokens remaining (window={}, multiplier={})",
            cache_position,
            remaining,
            sliding_window,
            window_multiplier
        );
        ResumablePrefillChunk::Incremental(remaining)
    };
    let tokens = match chunk {
        ResumablePrefillChunk::Matrix(tokens) | ResumablePrefillChunk::Incremental(tokens) => {
            tokens
        }
    };
    let end = cache_position
        .checked_add(tokens)
        .context("DeepSeek-V4 resumable prefill chunk end overflowed")?;
    Ok(ResumablePrefillSlice {
        chunk,
        capture_anchor_after: end == recovery_position,
    })
}
const MAX_CHEAP_INCREMENTAL_SUFFIX_TOKENS: usize = 32;
const MIN_MATRIX_REUSE_PREFIX_TOKENS: usize = 128;
const GPU_TRANSACTION_TIMEOUT: Duration = Duration::from_secs(30);
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

struct Deepseek4ResumablePrefill {
    cursor: usize,
    recovery_position: usize,
    window_multiplier: usize,
    cached_tokens: usize,
    origin: Deepseek4PrefillOrigin,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Deepseek4PrefillOrigin {
    Cold,
    Cached,
}

impl Deepseek4ResumablePrefill {
    fn initial_cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    fn is_cold_wave(&self) -> bool {
        self.origin == Deepseek4PrefillOrigin::Cold
    }
}

enum Deepseek4ResumablePrefillAdvance {
    Pending {
        advanced_tokens: usize,
    },
    Ready {
        logits: MlxBuffer,
        cached_tokens: usize,
        advanced_tokens: usize,
    },
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
    /// Prompt-boundary checkpoint captured by the current request. It is not
    /// promoted to reusable affinity until that request completes.
    pending_turn_anchor: Option<Deepseek4CacheSnapshot>,
    pending_turn_anchor_tokens: Vec<u32>,
    keep_turn_anchor_on_success: bool,
    request_anchor_transaction_active: bool,
    /// Full-logical-context agent sessions provisioned only for SlotAware.
    slot_sessions: Option<Vec<Deepseek4Session>>,
}

/// One retained DeepSeek-V4 agent context. The model/Metal execution state is
/// shared, but each slot owns independent KV, compressor state, recovery
/// anchor, and exact rendered-token identity.
pub(super) struct Deepseek4Session {
    cache: Deepseek4Cache,
    committed_tokens: Vec<u32>,
    live_logits: Option<MlxBuffer>,
    turn_anchor: Option<Deepseek4CacheSnapshot>,
    turn_anchor_tokens: Vec<u32>,
    pending_turn_anchor: Option<Deepseek4CacheSnapshot>,
    pending_turn_anchor_tokens: Vec<u32>,
    keep_turn_anchor_on_success: bool,
    request_anchor_transaction_active: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Deepseek4CancellationRecovery {
    TurnAnchor,
    Reset,
}

fn deepseek4_cancellation_recovery(
    cache_poisoned: bool,
    anchor_position: Option<usize>,
    anchor_tokens: usize,
) -> Deepseek4CancellationRecovery {
    if !cache_poisoned && anchor_position == Some(anchor_tokens) && anchor_tokens > 0 {
        Deepseek4CancellationRecovery::TurnAnchor
    } else {
        Deepseek4CancellationRecovery::Reset
    }
}

impl Deepseek4Session {
    /// Start with the same bounded cache shape as the proven SerialFifo path,
    /// while keeping append-only KV lazy so four idle slots do not register
    /// their complete initial arenas as resident.
    /// `Deepseek4LoadedModel::ensure_cache_capacity` grows this session in
    /// 131K-token steps when its request needs more space, up to the complete
    /// per-slot logical context. Logical context is therefore independent of
    /// slot count without making short turns pay 524K-shaped Metal strides.
    pub(super) fn new(loaded: &Deepseek4LoadedModel) -> Result<Self> {
        let initial_capacity = loaded.context_limit().min(INITIAL_CACHE_LENGTH);
        let cache = loaded
            .model
            .allocate_logical_cache(initial_capacity)
            .with_context(|| {
                format!("allocate initial {initial_capacity}-token DeepSeek-V4 agent slot cache")
            })?;
        Ok(Self {
            cache,
            committed_tokens: Vec::new(),
            live_logits: None,
            turn_anchor: None,
            turn_anchor_tokens: Vec::new(),
            pending_turn_anchor: None,
            pending_turn_anchor_tokens: Vec::new(),
            keep_turn_anchor_on_success: false,
            request_anchor_transaction_active: false,
        })
    }

    pub(super) fn committed_tokens(&self) -> &[u32] {
        &self.committed_tokens
    }

    pub(super) fn cache_position(&self) -> usize {
        self.cache.position()
    }

    pub(super) fn prepare_cooperative_prefill(
        &mut self,
        start: usize,
        token_count: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            self.cache.position() == start
                && self.committed_tokens.len() == start
                && self.live_logits.is_none(),
            "DeepSeek-V4 cooperative prefill start {start} disagrees with cache {} / token ledger {} / live logits {}",
            self.cache.position(),
            self.committed_tokens.len(),
            self.live_logits.is_some()
        );
        self.committed_tokens
            .try_reserve(token_count)
            .context("reserve DeepSeek-V4 cooperative token ledger")?;
        Ok(())
    }

    pub(super) fn validate_cooperative_prefill_result(
        &self,
        start: usize,
        token_count: usize,
    ) -> Result<()> {
        let end = start
            .checked_add(token_count)
            .context("DeepSeek-V4 cooperative token ledger overflow")?;
        anyhow::ensure!(
            self.cache.position() == end
                && self.committed_tokens.len() == start
                && self.live_logits.is_none(),
            "DeepSeek-V4 cooperative result end {end} disagrees with cache {} / token ledger {} / live logits {}",
            self.cache.position(),
            self.committed_tokens.len(),
            self.live_logits.is_some()
        );
        Ok(())
    }

    pub(super) fn publish_cooperative_tokens(&mut self, tokens: &[u32]) {
        debug_assert_eq!(
            self.cache.position(),
            self.committed_tokens.len() + tokens.len()
        );
        self.committed_tokens.extend_from_slice(tokens);
    }

    /// Return the longest prefix this retained slot can safely resume for the
    /// rendered prompt. DeepSeek's native encoder may rewrite the generated
    /// reasoning/tool tail on the next turn, so scheduler affinity must
    /// consider the prompt-boundary recovery anchor as well as the exact live
    /// decode ledger. Selecting an empty slot when the anchor matches would
    /// discard the entire agent context before capacity growth can migrate it.
    pub(super) fn reusable_prefix_len(&self, prompt_tokens: &[u32]) -> usize {
        let live = (!self.cache.is_poisoned()
            && self.cache.position() == self.committed_tokens.len()
            && !self.committed_tokens.is_empty()
            && prompt_tokens.starts_with(&self.committed_tokens))
        .then_some(self.committed_tokens.len())
        .unwrap_or(0);
        let recovery = self
            .turn_anchor
            .as_ref()
            .filter(|anchor| {
                anchor.position() == self.turn_anchor_tokens.len()
                    && !self.turn_anchor_tokens.is_empty()
                    && prompt_tokens.len() > self.turn_anchor_tokens.len()
                    && prompt_tokens.starts_with(&self.turn_anchor_tokens)
            })
            .map_or(0, |_| self.turn_anchor_tokens.len());
        let selected = if live >= recovery && live > 0 {
            PrefixReuse::Live(live)
        } else if recovery > 0 {
            PrefixReuse::RecoveryAnchor(recovery)
        } else {
            PrefixReuse::Reset
        };
        match prefer_matrix_prefill(prompt_tokens.len(), selected) {
            PrefixReuse::Live(count) | PrefixReuse::RecoveryAnchor(count) => count,
            PrefixReuse::Reset => 0,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.committed_tokens.is_empty() && self.turn_anchor_tokens.is_empty()
    }

    pub(super) fn reset(&mut self) -> Result<()> {
        self.cache
            .reset()
            .context("reset DeepSeek-V4 agent slot cache")?;
        self.committed_tokens.clear();
        self.live_logits = None;
        self.turn_anchor = None;
        self.turn_anchor_tokens.clear();
        self.pending_turn_anchor = None;
        self.pending_turn_anchor_tokens.clear();
        self.keep_turn_anchor_on_success = false;
        self.request_anchor_transaction_active = false;
        Ok(())
    }

    /// Cancelled work must not publish its partially extended live cursor, but
    /// a valid pre-request turn anchor remains safe to reuse. Restore that
    /// compact checkpoint when possible; cold/no-anchor cancellation retains
    /// the conservative full-reset fallback.
    pub(super) fn recover_after_cancellation(&mut self) -> Result<()> {
        self.pending_turn_anchor = None;
        self.pending_turn_anchor_tokens.clear();
        self.keep_turn_anchor_on_success = false;
        self.request_anchor_transaction_active = false;
        let recovery = deepseek4_cancellation_recovery(
            self.cache.is_poisoned(),
            self.turn_anchor
                .as_ref()
                .map(Deepseek4CacheSnapshot::position),
            self.turn_anchor_tokens.len(),
        );
        if recovery == Deepseek4CancellationRecovery::Reset {
            return self.reset();
        }
        let anchor = self
            .turn_anchor
            .as_ref()
            .context("DeepSeek-V4 cancellation turn anchor disappeared")?;
        self.cache
            .restore(anchor)
            .context("restore DeepSeek-V4 turn anchor after cancellation")?;
        self.committed_tokens.clone_from(&self.turn_anchor_tokens);
        self.live_logits = None;
        anyhow::ensure!(
            self.cache.position() == self.committed_tokens.len(),
            "DeepSeek-V4 cancellation rollback cache position {} disagrees with token ledger {}",
            self.cache.position(),
            self.committed_tokens.len()
        );
        Ok(())
    }

    /// Publish the current request's candidate checkpoint only after its
    /// terminal result is known to be successful.
    pub(super) fn commit_request_anchor(&mut self) {
        if !self.request_anchor_transaction_active {
            return;
        }
        self.request_anchor_transaction_active = false;
        if let Some(anchor) = self.pending_turn_anchor.take() {
            self.turn_anchor = Some(anchor);
            self.turn_anchor_tokens = std::mem::take(&mut self.pending_turn_anchor_tokens);
        } else if !self.keep_turn_anchor_on_success {
            self.turn_anchor = None;
            self.turn_anchor_tokens.clear();
        }
        self.pending_turn_anchor_tokens.clear();
        self.keep_turn_anchor_on_success = false;
    }

    /// Swap this slot into the single model execution surface. The staging
    /// state in `loaded` is swapped back after the request/tick, so no KV bytes
    /// are copied and every slot retains its own compressor state.
    pub(super) fn swap_with_loaded(&mut self, loaded: &mut Deepseek4LoadedModel) {
        std::mem::swap(&mut self.cache, &mut loaded.cache);
        std::mem::swap(&mut self.committed_tokens, &mut loaded.committed_tokens);
        std::mem::swap(&mut self.live_logits, &mut loaded.live_logits);
        std::mem::swap(&mut self.turn_anchor, &mut loaded.turn_anchor);
        std::mem::swap(&mut self.turn_anchor_tokens, &mut loaded.turn_anchor_tokens);
        std::mem::swap(
            &mut self.pending_turn_anchor,
            &mut loaded.pending_turn_anchor,
        );
        std::mem::swap(
            &mut self.pending_turn_anchor_tokens,
            &mut loaded.pending_turn_anchor_tokens,
        );
        std::mem::swap(
            &mut self.keep_turn_anchor_on_success,
            &mut loaded.keep_turn_anchor_on_success,
        );
        std::mem::swap(
            &mut self.request_anchor_transaction_active,
            &mut loaded.request_anchor_transaction_active,
        );
    }
}

impl Deepseek4LoadedModel {
    pub(super) fn supervised_verifier_prefill_cohort(
        &mut self,
        token_batches: &[&[u32]],
        sessions: &mut [Deepseek4Session],
        sorted_slot_indices: &[usize],
        supervisor: &EngineSupervisor,
    ) -> Result<Vec<MlxBuffer>> {
        anyhow::ensure!(
            (2..=4).contains(&sorted_slot_indices.len())
                && token_batches.len() == sorted_slot_indices.len(),
            "DeepSeek-V4 cooperative serving requires 2..=4 aligned lanes"
        );
        anyhow::ensure!(
            sorted_slot_indices.windows(2).all(|pair| pair[0] < pair[1]),
            "DeepSeek-V4 cooperative slot indices must be unique and sorted"
        );
        let mut wanted = sorted_slot_indices.iter().copied().peekable();
        let mut caches = Vec::with_capacity(sorted_slot_indices.len());
        for (index, session) in sessions.iter_mut().enumerate() {
            if wanted.peek().copied() == Some(index) {
                caches.push(&mut session.cache);
                wanted.next();
            }
        }
        anyhow::ensure!(
            wanted.next().is_none() && caches.len() == token_batches.len(),
            "DeepSeek-V4 cooperative slot index is out of range"
        );
        let lease = supervisor.arm("deepseek4_cooperative_prefill", GPU_TRANSACTION_TIMEOUT)?;
        self.model.forward_verifier_prefill_cohort_with_commit_gate(
            token_batches,
            &mut caches,
            || lease.finish(),
        )
    }

    fn supervised_verifier_prefill(
        &mut self,
        token_ids: &[u32],
        supervisor: &EngineSupervisor,
        kind: &'static str,
    ) -> Result<MlxBuffer> {
        let lease = supervisor.arm(kind, GPU_TRANSACTION_TIMEOUT)?;
        self.model
            .forward_verifier_prefill_with_commit_gate(token_ids, &mut self.cache, || {
                lease.finish()
            })
    }

    fn supervised_verifier_one(
        &mut self,
        token_id: u32,
        supervisor: &EngineSupervisor,
        kind: &'static str,
    ) -> Result<MlxBuffer> {
        let lease = supervisor.arm(kind, GPU_TRANSACTION_TIMEOUT)?;
        self.model
            .forward_verifier_one_with_commit_gate(token_id, &mut self.cache, || lease.finish())
    }

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
        let cache = model
            .allocate_cache(initial_cache_length)
            .with_context(|| {
                format!("allocate initial {initial_cache_length}-token DeepSeek-V4 cache")
            })?;
        tracing::info!(
            serving_context = context_length,
            allocated_cache_context = initial_cache_length,
            "DeepSeek-V4 cache admitted with demand-grown capacity"
        );

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
            pending_turn_anchor: None,
            pending_turn_anchor_tokens: Vec::new(),
            keep_turn_anchor_on_success: false,
            request_anchor_transaction_active: false,
            slot_sessions: None,
        })
    }

    /// Exercise the verifier and output head only after the worker supervisor
    /// exists. Loading used to submit these Metal transactions before any
    /// readiness poison/timeout authority was available, so a startup wedge
    /// could leave the server permanently warming with no bounded failure.
    pub(super) fn warmup(&mut self, supervisor: &EngineSupervisor) -> Result<()> {
        if std::env::var("HF2Q_DEEPSEEK_SKIP_WARMUP").as_deref() == Ok("1") {
            return Ok(());
        }
        let eos = self.eos_token_ids.first().copied().unwrap_or(1);
        let warm_tokens = vec![eos; 64];
        let warm_state = self
            .supervised_verifier_prefill(&warm_tokens, supervisor, "deepseek4_warmup_prefill")
            .context("warm DeepSeek-V4 matrix prefill")?;
        let warm_last = self
            .model
            .last_token_state(&warm_state)
            .context("view DeepSeek-V4 warmup state")?;
        supervisor.run("deepseek4_warmup_head", GPU_TRANSACTION_TIMEOUT, || {
            self.model
                .forward_logits(&warm_last)
                .context("warm DeepSeek-V4 output head")
        })?;
        self.cache
            .reset()
            .context("reset DeepSeek-V4 cache after warmup")?;
        log_scratch("release", "startup-warmup", release_prefill_scratch());
        Ok(())
    }

    pub(super) fn provision_slot_sessions(&mut self, max_slots: u32) -> Result<()> {
        anyhow::ensure!(max_slots > 0, "DeepSeek-V4 max_slots must be nonzero");
        let mut sessions = Vec::with_capacity(max_slots as usize);
        for slot in 0..max_slots {
            sessions.push(
                Deepseek4Session::new(self)
                    .with_context(|| format!("reserve DeepSeek-V4 logical slot {slot}"))?,
            );
        }
        tracing::info!(
            max_slots,
            logical_context_per_slot = self.context_limit(),
            "DeepSeek-V4 full-context agent slots reserved"
        );
        self.slot_sessions = Some(sessions);
        Ok(())
    }

    pub(super) fn take_slot_sessions(&mut self) -> Result<Vec<Deepseek4Session>> {
        self.slot_sessions
            .take()
            .context("DeepSeek-V4 SlotAware sessions were not provisioned")
    }

    fn context_limit(&self) -> usize {
        self.context_length
            .unwrap_or(self.model.cfg.max_position_embeddings as usize)
    }

    pub(super) fn reset_live_cache(&mut self) -> Result<()> {
        self.cache
            .reset()
            .context("reset DeepSeek-V4 serving cache")?;
        self.committed_tokens.clear();
        self.live_logits = None;
        self.turn_anchor = None;
        self.turn_anchor_tokens.clear();
        self.pending_turn_anchor = None;
        self.pending_turn_anchor_tokens.clear();
        self.keep_turn_anchor_on_success = false;
        self.request_anchor_transaction_active = false;
        Ok(())
    }

    fn reset_live_cache_preserving_turn_anchor(&mut self) -> Result<()> {
        self.cache
            .reset()
            .context("reset DeepSeek-V4 serving cache")?;
        self.committed_tokens.clear();
        self.live_logits = None;
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
        let mut grown = self.model.allocate_logical_cache(target).with_context(|| {
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

    fn begin_request_anchor_transaction(&mut self, prompt_tokens: &[u32]) {
        self.pending_turn_anchor = None;
        self.pending_turn_anchor_tokens.clear();
        self.request_anchor_transaction_active = true;
        self.keep_turn_anchor_on_success = self.turn_anchor.as_ref().is_some_and(|anchor| {
            anchor.position() == self.turn_anchor_tokens.len()
                && !self.turn_anchor_tokens.is_empty()
                && prompt_tokens.starts_with(&self.turn_anchor_tokens)
        });
    }

    pub(super) fn commit_request_anchor(&mut self) {
        if !self.request_anchor_transaction_active {
            return;
        }
        self.request_anchor_transaction_active = false;
        if let Some(anchor) = self.pending_turn_anchor.take() {
            self.turn_anchor = Some(anchor);
            self.turn_anchor_tokens = std::mem::take(&mut self.pending_turn_anchor_tokens);
        } else if !self.keep_turn_anchor_on_success {
            self.turn_anchor = None;
            self.turn_anchor_tokens.clear();
        }
        self.pending_turn_anchor_tokens.clear();
        self.keep_turn_anchor_on_success = false;
    }

    pub(super) fn recover_after_cancellation(&mut self) -> Result<()> {
        self.pending_turn_anchor = None;
        self.pending_turn_anchor_tokens.clear();
        self.keep_turn_anchor_on_success = false;
        self.request_anchor_transaction_active = false;
        let recovery = deepseek4_cancellation_recovery(
            self.cache.is_poisoned(),
            self.turn_anchor
                .as_ref()
                .map(Deepseek4CacheSnapshot::position),
            self.turn_anchor_tokens.len(),
        );
        if recovery == Deepseek4CancellationRecovery::Reset {
            return self.reset_live_cache();
        }
        let anchor = self
            .turn_anchor
            .as_ref()
            .context("DeepSeek-V4 cancellation turn anchor disappeared")?;
        self.cache
            .restore(anchor)
            .context("restore DeepSeek-V4 turn anchor after cancellation")?;
        self.committed_tokens.clone_from(&self.turn_anchor_tokens);
        self.live_logits = None;
        anyhow::ensure!(
            self.cache.position() == self.committed_tokens.len(),
            "DeepSeek-V4 cancellation rollback cache position {} disagrees with token ledger {}",
            self.cache.position(),
            self.committed_tokens.len()
        );
        Ok(())
    }

    fn capture_turn_anchor(&mut self, prompt_prefix: &[u32]) -> Result<()> {
        // Release only this request's prior candidate before allocating its
        // replacement. The committed pre-request checkpoint remains private
        // until success promotes the new candidate.
        self.pending_turn_anchor = None;
        self.pending_turn_anchor_tokens.clear();
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
        self.pending_turn_anchor = Some(snapshot);
        self.pending_turn_anchor_tokens
            .extend_from_slice(prompt_prefix);
        Ok(())
    }

    fn prefill_window_multiplier(&self, token_count: usize) -> Result<usize> {
        let explicit_prefill_windows = std::env::var_os("HF2Q_DEEPSEEK_PREFILL_WINDOWS").is_some();
        let mut window_multiplier = self.model.matrix_prefill_window_multiplier()?;
        if !explicit_prefill_windows && self.cache.capacity() > INITIAL_CACHE_LENGTH {
            window_multiplier = window_multiplier.min(8);
        } else if !explicit_prefill_windows {
            window_multiplier = balance_fresh_three_chunk_prefill(
                self.cache.position(),
                token_count,
                self.model.cfg.sliding_window as usize,
                window_multiplier,
            );
        }
        Ok(window_multiplier)
    }

    fn begin_resumable_cold_prefill(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        progress: &mut RequestProgress,
    ) -> Result<Deepseek4ResumablePrefill> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "DeepSeek-V4 prompt is empty");
        anyhow::ensure!(
            prompt_tokens.len() >= self.model.cfg.sliding_window as usize,
            "DeepSeek-V4 resumable prefill requires at least one native window"
        );
        anyhow::ensure!(
            prompt_tokens.len() <= self.context_limit(),
            "DeepSeek-V4 prompt has {} tokens, exceeding serving context {}",
            prompt_tokens.len(),
            self.context_limit()
        );
        self.begin_request_anchor_transaction(prompt_tokens);
        let cache_grew = self.ensure_cache_capacity(prompt_tokens.len(), max_tokens)?;
        progress.cache_reset_diagnostic(
            self.committed_tokens.len(),
            self.cache.position(),
            common_prefix_len(prompt_tokens, &self.committed_tokens),
            self.turn_anchor_tokens.len(),
            self.turn_anchor
                .as_ref()
                .map_or(0, Deepseek4CacheSnapshot::position),
            common_prefix_len(prompt_tokens, &self.turn_anchor_tokens),
            self.cache.is_poisoned(),
            false,
            cache_grew,
        );
        self.reset_live_cache_preserving_turn_anchor()?;
        progress.plan_prefill(
            0,
            prompt_tokens.len(),
            if cache_grew { "grow-reset" } else { "reset" },
        );
        let window_multiplier = self.prefill_window_multiplier(prompt_tokens.len())?;
        Ok(Deepseek4ResumablePrefill {
            cursor: 0,
            recovery_position: prompt_tokens.len().saturating_sub(RECOVERY_TAIL_TOKENS),
            window_multiplier,
            cached_tokens: 0,
            origin: Deepseek4PrefillOrigin::Cold,
        })
    }

    fn begin_resumable_cached_prefill(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        progress: &mut RequestProgress,
    ) -> Result<Deepseek4ResumablePrefill> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "DeepSeek-V4 prompt is empty");
        anyhow::ensure!(
            prompt_tokens.len() <= self.context_limit(),
            "DeepSeek-V4 prompt has {} tokens, exceeding serving context {}",
            prompt_tokens.len(),
            self.context_limit()
        );
        self.begin_request_anchor_transaction(prompt_tokens);
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
        let (cached_tokens, cache_action) = match reuse {
            PrefixReuse::Live(count) => (count, if cache_grew { "grow-live" } else { "live" }),
            PrefixReuse::RecoveryAnchor(count) => {
                let snapshot = self
                    .turn_anchor
                    .as_ref()
                    .context("DeepSeek-V4 turn anchor disappeared before cached resume")?;
                self.cache
                    .restore(snapshot)
                    .context("restore DeepSeek-V4 prompt-boundary cache for cached resume")?;
                self.committed_tokens.clone_from(&self.turn_anchor_tokens);
                (
                    count,
                    if cache_grew {
                        "grow-recovery-anchor"
                    } else {
                        "recovery-anchor"
                    },
                )
            }
            PrefixReuse::Reset => anyhow::bail!(
                "DeepSeek-V4 cached resumable prefill lost its effective reusable prefix"
            ),
        };
        anyhow::ensure!(
            cached_tokens > 0 && cached_tokens < prompt_tokens.len(),
            "DeepSeek-V4 cached resumable prefill requires a nonempty suffix (cached={}, prompt={})",
            cached_tokens,
            prompt_tokens.len()
        );
        anyhow::ensure!(
            self.cache.position() == cached_tokens
                && self.committed_tokens.len() == cached_tokens
                && prompt_tokens.starts_with(&self.committed_tokens),
            "DeepSeek-V4 cached resume cursor {} disagrees with cache {} / token ledger {}",
            cached_tokens,
            self.cache.position(),
            self.committed_tokens.len()
        );
        self.live_logits = None;
        progress.plan_prefill(
            cached_tokens,
            prompt_tokens.len().saturating_sub(cached_tokens),
            cache_action,
        );
        let window_multiplier =
            self.prefill_window_multiplier(prompt_tokens.len().saturating_sub(cached_tokens))?;
        Ok(Deepseek4ResumablePrefill {
            cursor: cached_tokens,
            recovery_position: prompt_tokens.len().saturating_sub(RECOVERY_TAIL_TOKENS),
            window_multiplier,
            cached_tokens,
            origin: Deepseek4PrefillOrigin::Cached,
        })
    }

    fn advance_resumable_prefill(
        &mut self,
        prompt_tokens: &[u32],
        plan: &mut Deepseek4ResumablePrefill,
        cancelled: &impl Fn() -> bool,
        progress: &mut RequestProgress,
        max_matrix_prefill_windows: Option<usize>,
        supervisor: &EngineSupervisor,
    ) -> Result<Deepseek4ResumablePrefillAdvance> {
        anyhow::ensure!(
            self.cache.position() == plan.cursor && self.committed_tokens.len() == plan.cursor,
            "DeepSeek-V4 paused prefill cursor {} disagrees with cache {} / token ledger {}",
            plan.cursor,
            self.cache.position(),
            self.committed_tokens.len()
        );
        anyhow::ensure!(
            self.live_logits.is_none(),
            "DeepSeek-V4 paused prefill published live logits before completion"
        );
        anyhow::ensure!(
            !cancelled(),
            "DeepSeek-V4 generation cancelled during prefill"
        );

        let segment_end = if plan.cursor < plan.recovery_position {
            plan.recovery_position
        } else {
            prompt_tokens.len()
        };
        let remaining = segment_end.saturating_sub(plan.cursor);
        let window_multiplier = max_matrix_prefill_windows
            .map(|limit| plan.window_multiplier.min(limit))
            .unwrap_or(plan.window_multiplier);
        let slice = plan_resumable_prefill_chunk(
            self.cache.position(),
            remaining,
            self.model.cfg.sliding_window as usize,
            window_multiplier,
            plan.recovery_position,
        )?;
        let (batch, incremental) = match slice.chunk {
            ResumablePrefillChunk::Matrix(batch) => (batch, false),
            ResumablePrefillChunk::Incremental(batch) => (batch, true),
        };
        if let Some(window_cap) = max_matrix_prefill_windows {
            progress.mixed_prefill_slice(batch, window_cap);
        }
        if incremental {
            let start = plan.cursor;
            let mut state = None;
            for &token in &prompt_tokens[start..segment_end] {
                anyhow::ensure!(
                    !cancelled(),
                    "DeepSeek-V4 generation cancelled during incremental prefill"
                );
                state = Some(
                    self.supervised_verifier_one(
                        token,
                        supervisor,
                        "deepseek4_incremental_prefill",
                    )
                    .context("execute DeepSeek-V4 resumable incremental token")?,
                );
                self.committed_tokens.push(token);
                plan.cursor += 1;
                progress.advance_prefill(1);
            }
            let state = state.context("DeepSeek-V4 recovery-tail prefill produced no state")?;
            anyhow::ensure!(
                !cancelled(),
                "DeepSeek-V4 generation cancelled after committed incremental prefill"
            );
            if slice.capture_anchor_after {
                self.capture_turn_anchor(&prompt_tokens[..plan.recovery_position])?;
                progress.recovery_anchor_captured(plan.recovery_position);
            }
            if plan.cursor < prompt_tokens.len() {
                drop(state);
                return Ok(Deepseek4ResumablePrefillAdvance::Pending {
                    advanced_tokens: batch,
                });
            }
            let (logits, cached_tokens) =
                self.finish_prefill(state, plan.cached_tokens, supervisor)?;
            return Ok(Deepseek4ResumablePrefillAdvance::Ready {
                logits,
                cached_tokens,
                advanced_tokens: batch,
            });
        }
        let end = plan.cursor.saturating_add(batch);
        let state = self
            .supervised_verifier_prefill(
                &prompt_tokens[plan.cursor..end],
                supervisor,
                "deepseek4_matrix_prefill",
            )
            .context("execute DeepSeek-V4 resumable matrix prefill chunk")?;
        self.committed_tokens
            .extend_from_slice(&prompt_tokens[plan.cursor..end]);
        plan.cursor = end;
        progress.advance_prefill(batch);
        anyhow::ensure!(
            !cancelled(),
            "DeepSeek-V4 generation cancelled after committed prefill chunk"
        );

        if slice.capture_anchor_after {
            self.capture_turn_anchor(&prompt_tokens[..plan.recovery_position])?;
            progress.recovery_anchor_captured(plan.recovery_position);
        }
        if plan.cursor < prompt_tokens.len() {
            drop(state);
            return Ok(Deepseek4ResumablePrefillAdvance::Pending {
                advanced_tokens: batch,
            });
        }
        let (logits, cached_tokens) = self.finish_prefill(state, plan.cached_tokens, supervisor)?;
        Ok(Deepseek4ResumablePrefillAdvance::Ready {
            logits,
            cached_tokens,
            advanced_tokens: batch,
        })
    }

    fn append_prompt_tokens(
        &mut self,
        tokens: &[u32],
        cancelled: &impl Fn() -> bool,
        progress: &mut RequestProgress,
        supervisor: &EngineSupervisor,
    ) -> Result<Option<MlxBuffer>> {
        if tokens.is_empty() {
            return Ok(None);
        }
        let mut state = None;
        let mut offset = 0;
        let window_multiplier = self.prefill_window_multiplier(tokens.len())?;
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
                self.supervised_verifier_prefill(
                    &tokens[offset..offset + batch],
                    supervisor,
                    "deepseek4_matrix_prefill",
                )
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
                self.supervised_verifier_one(token, supervisor, "deepseek4_tail_prefill")
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
        supervisor: &EngineSupervisor,
    ) -> Result<(MlxBuffer, usize)> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "DeepSeek-V4 prompt is empty");
        anyhow::ensure!(
            prompt_tokens.len() <= self.context_limit(),
            "DeepSeek-V4 prompt has {} tokens, exceeding serving context {}",
            prompt_tokens.len(),
            self.context_limit()
        );
        self.begin_request_anchor_transaction(prompt_tokens);
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
                self.reset_live_cache_preserving_turn_anchor()?;
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
            self.append_prompt_tokens(
                &prompt_tokens[..recovery_position],
                &cancelled,
                progress,
                supervisor,
            )?;
            self.capture_turn_anchor(&prompt_tokens[..recovery_position])?;
            progress.recovery_anchor_captured(recovery_position);
            self.cache
                .reset()
                .context("reset DeepSeek-V4 live cache after recovery prepass")?;
            self.committed_tokens.clear();
            self.live_logits = None;

            let final_state = self
                .append_prompt_tokens(prompt_tokens, &cancelled, progress, supervisor)?
                .context("DeepSeek-V4 prompt produced no verifier state")?;
            return self.finish_prefill(final_state, 0, supervisor);
        }

        let mut cursor = cached_tokens;
        let mut final_state = None;
        if recovery_position > cursor {
            final_state = self.append_prompt_tokens(
                &prompt_tokens[cursor..recovery_position],
                &cancelled,
                progress,
                supervisor,
            )?;
            cursor = recovery_position;
            self.capture_turn_anchor(&prompt_tokens[..recovery_position])?;
            progress.recovery_anchor_captured(recovery_position);
        }
        final_state = self
            .append_prompt_tokens(&prompt_tokens[cursor..], &cancelled, progress, supervisor)?
            .or(final_state);
        self.finish_prefill(
            final_state.context("DeepSeek-V4 suffix produced no verifier state")?,
            cached_tokens,
            supervisor,
        )
    }

    fn finish_prefill(
        &mut self,
        state: MlxBuffer,
        cached_tokens: usize,
        supervisor: &EngineSupervisor,
    ) -> Result<(MlxBuffer, usize)> {
        let last = self
            .model
            .last_token_state(&state)
            .context("view DeepSeek-V4 last prompt state")?;
        let logits = supervisor.run("deepseek4_prefill_head", GPU_TRANSACTION_TIMEOUT, || {
            self.model
                .forward_logits(&last)
                .context("execute DeepSeek-V4 prompt output head")
        })?;
        self.live_logits = Some(logits.clone());
        Ok((logits, cached_tokens))
    }

    fn commit_generated_token(
        &mut self,
        token: u32,
        supervisor: &EngineSupervisor,
    ) -> Result<MlxBuffer> {
        let cache_position = self.cache.position();
        let state = self
            .supervised_verifier_one(token, supervisor, "deepseek4_decode_verifier")
            .with_context(|| {
                format!("execute DeepSeek-V4 decode token at cache position {cache_position}")
            })?;
        self.committed_tokens.push(token);
        let logits = supervisor.run("deepseek4_decode_head", GPU_TRANSACTION_TIMEOUT, || {
            self.model
                .forward_logits(&state)
                .context("execute DeepSeek-V4 decode output head")
        })?;
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
            // Context-linear compressed/indexer rows. Circular window,
            // compressor state, and compact recovery rollback are fixed.
            kv_bytes_per_token_override: Some(6_880),
            kv_fixed_bytes_per_slot_override: Some(load_info::deepseek4_fixed_kv_bytes_per_slot(
                &self.model.cfg,
                6_880,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        balance_fresh_three_chunk_prefill, cache_capacity_for_request, common_prefix_len,
        deepseek4_cancellation_recovery, plan_resumable_prefill_chunk, prefer_matrix_prefill,
        resumable_matrix_prefill_chunk_len, select_prefix_reuse, should_retain_decode_scratch,
        Deepseek4CancellationRecovery, PrefixReuse, ResumablePrefillChunk, ResumablePrefillSlice,
        TransientScratchStats, MIN_MATRIX_APPEND_TOKENS, RECOVERY_TAIL_TOKENS,
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
    fn interactive_prefill_cap_leaves_a_legal_matrix_or_recovery_tail() {
        assert_eq!(resumable_matrix_prefill_chunk_len(1, 285, 128, 2), 252);
        assert_eq!(285 - 252, 33);
        assert_eq!(resumable_matrix_prefill_chunk_len(1, 264, 128, 2), 256);
        assert_eq!(264 - 256, 8);
    }

    #[test]
    fn cached_suffix_below_matrix_minimum_replays_without_an_empty_chunk() {
        let cached_tokens = 107_066;
        let prompt_tokens = 107_090;
        let recovery_position = prompt_tokens - RECOVERY_TAIL_TOKENS;

        for remaining in 1..MIN_MATRIX_APPEND_TOKENS {
            assert_eq!(
                plan_resumable_prefill_chunk(cached_tokens, remaining, 128, 16, usize::MAX,)
                    .unwrap(),
                ResumablePrefillSlice {
                    chunk: ResumablePrefillChunk::Incremental(remaining),
                    capture_anchor_after: false,
                }
            );
        }

        let before_anchor = plan_resumable_prefill_chunk(
            cached_tokens,
            recovery_position - cached_tokens,
            128,
            16,
            recovery_position,
        )
        .unwrap();
        assert_eq!(
            before_anchor,
            ResumablePrefillSlice {
                chunk: ResumablePrefillChunk::Incremental(16),
                capture_anchor_after: true,
            }
        );
        let ResumablePrefillChunk::Incremental(before_anchor_tokens) = before_anchor.chunk else {
            panic!("sub-matrix suffix before the recovery anchor must replay incrementally");
        };
        assert_eq!(cached_tokens + before_anchor_tokens, recovery_position);

        let recovery_tail = plan_resumable_prefill_chunk(
            recovery_position,
            prompt_tokens - recovery_position,
            128,
            16,
            recovery_position,
        )
        .unwrap();
        assert_eq!(
            recovery_tail,
            ResumablePrefillSlice {
                chunk: ResumablePrefillChunk::Incremental(8),
                capture_anchor_after: false,
            }
        );
        let ResumablePrefillChunk::Incremental(recovery_tail_tokens) = recovery_tail.chunk else {
            panic!("the recovery tail must replay incrementally");
        };
        assert_eq!(recovery_position + recovery_tail_tokens, prompt_tokens);
        assert_eq!(
            plan_resumable_prefill_chunk(
                recovery_position,
                MIN_MATRIX_APPEND_TOKENS,
                128,
                16,
                usize::MAX,
            )
            .unwrap(),
            ResumablePrefillSlice {
                chunk: ResumablePrefillChunk::Matrix(MIN_MATRIX_APPEND_TOKENS),
                capture_anchor_after: false,
            }
        );
        assert!(
            plan_resumable_prefill_chunk(cached_tokens, 16, 128, 0, recovery_position).is_err()
        );
        assert!(
            plan_resumable_prefill_chunk(cached_tokens, 0, 128, 16, recovery_position).is_err()
        );
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
    fn cancellation_preserves_only_a_valid_unpoisoned_turn_anchor() {
        assert_eq!(
            deepseek4_cancellation_recovery(false, Some(7_174), 7_174),
            Deepseek4CancellationRecovery::TurnAnchor
        );
        for (poisoned, position, tokens) in [
            (true, Some(7_174), 7_174),
            (false, Some(7_173), 7_174),
            (false, None, 7_174),
            (false, Some(0), 0),
        ] {
            assert_eq!(
                deepseek4_cancellation_recovery(poisoned, position, tokens),
                Deepseek4CancellationRecovery::Reset
            );
        }
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
