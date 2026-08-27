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
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use mlx_native::gguf::GgufFile;
use mlx_native::MlxBuffer;
use tokenizers::Tokenizer;

use crate::inference::models::deepseek4::{
    cache::Deepseek4Cache, decode_scratch_stats, matrix_prefill_chunk_len, prefill_scratch_stats,
    release_decode_scratch, release_prefill_scratch, tokenizer as deepseek_tokenizer,
    Deepseek4Model, TransientScratchStats, MIN_MATRIX_APPEND_TOKENS,
};
use crate::serve::load_info::{
    self, ArchFamily, ChatTemplateSource, LoadInfo, LoadInfoBuilder, MoeShape, TokenizerSource,
};

use self::progress::RequestProgress;
use super::anchor_store::{
    maybe_inject_anchor_restore_failure, AnchorDivergence, AnchorEntry, AnchorRestoreEvent,
    AnchorRestoreFaultFamily, AnchorRestoreOutcome, StagePending,
};
use super::deepseek4_anchor_store::{
    self as anchor_store_policy, Deepseek4AnchorBudget, Deepseek4AnchorStore,
    Deepseek4PromptAnchor, DEFAULT_MAX_COMMITTED_ANCHORS,
};
use super::engine::LoadOptions;
use super::engine_supervisor::EngineSupervisor;

const INITIAL_CACHE_LENGTH: usize = 131_072;
const RECOVERY_TAIL_TOKENS: usize = 8;

fn anchor_aggregate_budget_bytes() -> u64 {
    crate::serve::kv_persist::lcp_registry::default_lcp_byte_budget()
}

fn anchor_reprovision_budget_bytes(budget: &Deepseek4AnchorBudget) -> u64 {
    budget.aggregate_budget_bytes()
}

fn anchor_control_capacity(configured_stores: usize, aggregate_budget_bytes: u64) -> usize {
    let full_control_fits = (std::mem::size_of::<Deepseek4PromptAnchor>() as u64)
        .checked_mul(DEFAULT_MAX_COMMITTED_ANCHORS as u64)
        .and_then(|bytes| bytes.checked_mul(configured_stores as u64))
        .is_some_and(|bytes| bytes <= aggregate_budget_bytes);
    if full_control_fits {
        DEFAULT_MAX_COMMITTED_ANCHORS
    } else {
        0
    }
}

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
enum ShortRecoveryExecution {
    Standard,
    FullPromptOnly,
    CapturePrepassThenFullPrompt,
}

fn plan_short_recovery_execution(
    candidate: bool,
    admission: Option<StagePending>,
) -> Result<ShortRecoveryExecution> {
    if !candidate {
        return Ok(ShortRecoveryExecution::Standard);
    }
    match admission.context("DeepSeek-V4 short-recovery candidate is missing admission")? {
        StagePending::Staged => Ok(ShortRecoveryExecution::CapturePrepassThenFullPrompt),
        StagePending::NoCommittedCapacity | StagePending::BudgetExceeded { .. } => {
            Ok(ShortRecoveryExecution::FullPromptOnly)
        }
        StagePending::PendingOccupied => {
            anyhow::bail!(
                "DeepSeek-V4 short-recovery anchor preflight found an occupied pending payload"
            )
        }
    }
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
    /// Accumulated recovery-tail checkpoints over this mutable cache lineage.
    anchor_store: Deepseek4AnchorStore,
    anchor_budget: Arc<Deepseek4AnchorBudget>,
    request_anchor_transaction_active: bool,
    /// The logical SlotAware slot currently swapped into this execution
    /// surface. SerialFifo keeps this `None`.
    telemetry_slot: Option<u32>,
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
    anchor_store: Deepseek4AnchorStore,
    anchor_budget: Arc<Deepseek4AnchorBudget>,
    request_anchor_transaction_active: bool,
    telemetry_slot: Option<u32>,
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

fn deepest_recovery_anchor(store: &Deepseek4AnchorStore, prompt_tokens: &[u32]) -> Option<usize> {
    store.deepest_matching_index(|anchor| {
        anchor.snapshot.position() == anchor.prompt_tokens.len()
            && !anchor.prompt_tokens.is_empty()
            // DeepSeek recovery anchors intentionally carry no output logits.
            // Equality must stay on the live-ledger path; an anchor requires a
            // nonempty suffix whose first forward produces aligned logits.
            && prompt_tokens.len() > anchor.prompt_tokens.len()
            && prompt_tokens.starts_with(&anchor.prompt_tokens)
    })
}

fn deepest_cancellation_anchor(store: &Deepseek4AnchorStore, live_cursor: usize) -> Option<usize> {
    store.deepest_matching_index(|anchor| {
        anchor.snapshot.position() == anchor.prompt_tokens.len()
            && !anchor.prompt_tokens.is_empty()
            && anchor.prompt_tokens.len() <= live_cursor
    })
}

fn capture_anchor_candidate(
    cache: &Deepseek4Cache,
    prompt_prefix: &[u32],
    context: &'static str,
) -> Result<Deepseek4PromptAnchor> {
    let started = Instant::now();
    let snapshot = cache.snapshot().context(context)?;
    anyhow::ensure!(
        snapshot.position() == prompt_prefix.len(),
        "DeepSeek-V4 prompt anchor position {} does not match {} rendered tokens",
        snapshot.position(),
        prompt_prefix.len()
    );
    Ok(Deepseek4PromptAnchor::new(
        prompt_prefix,
        snapshot,
        started.elapsed(),
    ))
}

fn prospective_anchor_candidate_owned_bytes(
    cache: &Deepseek4Cache,
    prompt_tokens: usize,
) -> Result<u64> {
    cache
        .prospective_snapshot_owned_bytes()
        .context("preflight DeepSeek-V4 snapshot ownership")?
        .checked_add(
            (prompt_tokens as u64)
                .checked_mul(std::mem::size_of::<u32>() as u64)
                .context("DeepSeek-V4 anchor token-byte overflow")?,
        )
        .context("DeepSeek-V4 prospective anchor byte overflow")
}

fn preflight_anchor_capture(
    cache: &Deepseek4Cache,
    store: &Deepseek4AnchorStore,
    budget: &Deepseek4AnchorBudget,
    prompt_tokens: usize,
) -> Result<(StagePending, u64)> {
    let anchor_bytes = prospective_anchor_candidate_owned_bytes(cache, prompt_tokens)?;
    Ok((budget.preflight_capture(store, anchor_bytes), anchor_bytes))
}

fn capture_turn_anchor_with_source(
    cache: &Deepseek4Cache,
    store: &mut Deepseek4AnchorStore,
    budget: &Deepseek4AnchorBudget,
    prompt_prefix: &[u32],
    snapshot_context: &'static str,
    capture_source: &'static str,
) -> Result<bool> {
    let (admission, anchor_bytes) =
        preflight_anchor_capture(cache, store, budget, prompt_prefix.len())?;
    let anchor = super::anchor_store::capture_if_anchor_admitted(admission, || {
        capture_anchor_candidate(cache, prompt_prefix, snapshot_context)
    })?;
    let Some(anchor) = anchor else {
        anchor_store_policy::record_preflight_budget_skip(
            store,
            budget,
            anchor_bytes,
            admission,
            capture_source,
        );
        return Ok(false);
    };
    let outcome = anchor_store_policy::stage_pending(store, budget, anchor, capture_source)?;
    anyhow::ensure!(
        outcome == StagePending::Staged,
        "DeepSeek-V4 anchor admission changed between preflight and staging: {outcome:?}"
    );
    Ok(true)
}

fn preflight_anchor_store_migration(
    destination: &Deepseek4Cache,
    source: &Deepseek4Cache,
    store: &Deepseek4AnchorStore,
    previous: usize,
    target: usize,
) -> Result<()> {
    for index in 0..store.committed_len() {
        let anchor = store
            .committed(index)
            .context("DeepSeek-V4 committed anchor disappeared during growth preflight")?;
        destination
            .preflight_snapshot_migration(source, &anchor.snapshot)
            .with_context(|| {
                format!(
                    "preflight DeepSeek-V4 committed anchor {index} across {previous}->{target} growth"
                )
            })?;
    }
    if let Some(anchor) = store.pending() {
        destination
            .preflight_snapshot_migration(source, &anchor.snapshot)
            .with_context(|| {
                format!("preflight DeepSeek-V4 pending anchor across {previous}->{target} growth")
            })?;
    }
    Ok(())
}

impl Deepseek4Session {
    /// Start with the same bounded cache shape as the proven SerialFifo path,
    /// while keeping append-only KV lazy so four idle slots do not register
    /// their complete initial arenas as resident.
    /// `Deepseek4LoadedModel::ensure_cache_capacity` grows this session in
    /// 131K-token steps when its request needs more space, up to the complete
    /// per-slot logical context. Logical context is therefore independent of
    /// slot count without making short turns pay 524K-shaped Metal strides.
    pub(super) fn new(
        loaded: &Deepseek4LoadedModel,
        anchor_budget: Arc<Deepseek4AnchorBudget>,
        anchor_control_capacity: usize,
        slot: u32,
    ) -> Result<Self> {
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
            anchor_store: Deepseek4AnchorStore::with_committed_capacity(anchor_control_capacity),
            anchor_budget,
            request_anchor_transaction_active: false,
            telemetry_slot: Some(slot),
        })
    }

    pub(super) fn committed_tokens(&self) -> &[u32] {
        &self.committed_tokens
    }

    pub(super) fn cache_position(&self) -> usize {
        self.cache.position()
    }

    pub(super) fn idle_owned_bytes(&self) -> Result<(u64, u64)> {
        let kv_bytes = self.cache.owned_bytes();
        let prefix_bytes = (self.committed_tokens.capacity() as u64)
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .and_then(|bytes| bytes.checked_add(self.anchor_store.owned_bytes()))
            .context("DeepSeek-V4 session-owned byte total overflow")?;
        Ok((kv_bytes, prefix_bytes))
    }

    pub(super) fn live_logits_allocation(&self) -> Option<(usize, u64)> {
        self.live_logits
            .as_ref()
            .map(|logits| (logits.contents_ptr() as usize, logits.byte_len() as u64))
    }

    pub(super) fn decode_cohort_compatible_with(&self, other: &Self) -> bool {
        self.cache.position() == self.committed_tokens.len()
            && other.cache.position() == other.committed_tokens.len()
            && self.live_logits.is_some()
            && other.live_logits.is_some()
            && self.cache.decode_cohort_compatible_with(&other.cache)
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

    pub(super) fn capture_cooperative_turn_anchor(
        &mut self,
        prompt_prefix: &[u32],
    ) -> Result<bool> {
        capture_turn_anchor_with_source(
            &self.cache,
            &mut self.anchor_store,
            &self.anchor_budget,
            prompt_prefix,
            "snapshot DeepSeek-V4 cooperative prompt-boundary cache",
            "cooperative-prefill",
        )
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
        let recovery = deepest_recovery_anchor(&self.anchor_store, prompt_tokens)
            .and_then(|index| self.anchor_store.committed(index))
            .map_or(0, |anchor| anchor.prompt_tokens.len());
        let selected = if live >= recovery && live > 0 {
            PrefixReuse::Live(live)
        } else if recovery > 0 {
            PrefixReuse::RecoveryAnchor {
                count: recovery,
                index: deepest_recovery_anchor(&self.anchor_store, prompt_tokens)
                    .expect("positive recovery score has an anchor"),
            }
        } else {
            PrefixReuse::Reset
        };
        match prefer_matrix_prefill(prompt_tokens.len(), selected) {
            PrefixReuse::Live(count) | PrefixReuse::RecoveryAnchor { count, .. } => count,
            PrefixReuse::Reset => 0,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.committed_tokens.is_empty() && self.anchor_store.committed_len() == 0
    }

    pub(super) fn reset(&mut self) -> Result<()> {
        let clear = anchor_store_policy::clear_all(
            &mut self.anchor_store,
            &self.anchor_budget,
            "agent-slot-reset",
        );
        self.committed_tokens.clear();
        self.live_logits = None;
        self.request_anchor_transaction_active = false;
        let reset = self
            .cache
            .reset()
            .context("reset DeepSeek-V4 agent slot cache");
        match (clear, reset) {
            (Ok(()), Ok(())) => Ok(()),
            (clear, reset) => anyhow::bail!(
                "DeepSeek-V4 agent slot reset did not complete fail-closed cleanup (clear={clear:?}, reset={reset:?})"
            ),
        }
    }

    /// Cancelled work must not publish its partially extended live cursor, but
    /// a valid pre-request turn anchor remains safe to reuse. Restore that
    /// compact checkpoint when possible; cold/no-anchor cancellation retains
    /// the conservative full-reset fallback.
    pub(super) fn recover_after_cancellation(&mut self) -> Result<()> {
        self.request_anchor_transaction_active = false;
        let pending_discarded =
            anchor_store_policy::discard_pending(&mut self.anchor_store, &self.anchor_budget)?;
        let live_cursor = self.cache.position();
        let recovery_index = deepest_cancellation_anchor(&self.anchor_store, self.cache.position());
        let recovery = deepseek4_cancellation_recovery(
            self.cache.is_poisoned(),
            recovery_index.and_then(|index| {
                self.anchor_store
                    .committed(index)
                    .map(|anchor| anchor.snapshot.position())
            }),
            recovery_index
                .and_then(|index| self.anchor_store.committed(index))
                .map_or(0, |anchor| anchor.prompt_tokens.len()),
        );
        if recovery == Deepseek4CancellationRecovery::Reset {
            let divergence = AnchorDivergence::rewind(
                live_cursor,
                recovery_index
                    .and_then(|index| self.anchor_store.committed(index))
                    .map_or(0, |anchor| anchor.prompt_tokens.len()),
            );
            let reset = self.reset();
            anchor_store_policy::record_restore(AnchorRestoreEvent {
                family: "deepseek4",
                slot: self.telemetry_slot,
                cause: "cancellation rollback",
                outcome: if reset.is_ok() {
                    AnchorRestoreOutcome::MissNoMatch
                } else {
                    AnchorRestoreOutcome::FailedCleanup
                },
                attempted_hit_depth: 0,
                hit_depth: 0,
                divergence,
                tokens_saved: 0,
                descendant_prune_count: 0,
                pending_discarded,
                publication_disposition: None,
                capture_duration: Duration::ZERO,
                peak_committed_pending_bytes: self.anchor_store.peak_owned_bytes(),
            });
            return reset;
        }
        let recovery_index =
            recovery_index.context("DeepSeek-V4 cancellation anchor disappeared")?;
        let anchor_tokens = self
            .anchor_store
            .committed(recovery_index)
            .map_or(0, |anchor| anchor.prompt_tokens.len());
        self.restore_committed_anchor(
            recovery_index,
            "cancellation rollback",
            AnchorDivergence::rewind(self.cache.position(), anchor_tokens),
            pending_discarded,
            None,
        )?;
        anchor_store_policy::cancel_at_cursor(
            &mut self.anchor_store,
            &self.anchor_budget,
            self.cache.position(),
        )?;
        Ok(())
    }

    /// Publish the current request's candidate checkpoint only after its
    /// terminal result is known to be successful.
    pub(super) fn commit_request_anchor(&mut self) -> Result<()> {
        if !self.request_anchor_transaction_active {
            return Ok(());
        }
        self.request_anchor_transaction_active = false;
        if let Some(publication) =
            anchor_store_policy::publish_pending(&mut self.anchor_store, &self.anchor_budget)?
        {
            tracing::info!(
                target: "hf2q::serve::api::deepseek4_anchor",
                committed_anchors = self.anchor_store.committed_len(),
                anchor_owned_bytes = self.anchor_store.owned_bytes(),
                evicted = publication.evicted,
                replaced_equal_depth = publication.replaced_equal_depth,
                "DeepSeek-V4 pending anchor published after terminal ledger commit"
            );
        }
        Ok(())
    }

    /// Swap this slot into the single model execution surface. The staging
    /// state in `loaded` is swapped back after the request/tick, so no KV bytes
    /// are copied and every slot retains its own compressor state.
    pub(super) fn swap_with_loaded(&mut self, loaded: &mut Deepseek4LoadedModel) {
        std::mem::swap(&mut self.cache, &mut loaded.cache);
        std::mem::swap(&mut self.committed_tokens, &mut loaded.committed_tokens);
        std::mem::swap(&mut self.live_logits, &mut loaded.live_logits);
        std::mem::swap(&mut self.anchor_store, &mut loaded.anchor_store);
        std::mem::swap(
            &mut self.request_anchor_transaction_active,
            &mut loaded.request_anchor_transaction_active,
        );
        std::mem::swap(&mut self.telemetry_slot, &mut loaded.telemetry_slot);
    }

    fn restore_committed_anchor(
        &mut self,
        index: usize,
        cause: &'static str,
        divergence: AnchorDivergence,
        pending_discarded_before_restore: bool,
        request_max_tokens: Option<usize>,
    ) -> Result<()> {
        let (tokens, token_count, capture_duration, publication_disposition) = {
            let anchor = self
                .anchor_store
                .committed(index)
                .context("DeepSeek-V4 committed anchor disappeared before restore")?;
            (
                anchor.prompt_tokens.to_vec(),
                anchor.prompt_tokens.len(),
                anchor.capture_duration,
                anchor.publication_disposition(),
            )
        };
        let pending_discarded = pending_discarded_before_restore || self.anchor_store.has_pending();
        let observation = AnchorRestoreEvent {
            family: "deepseek4",
            slot: self.telemetry_slot,
            cause,
            outcome: AnchorRestoreOutcome::Hit,
            attempted_hit_depth: index + 1,
            hit_depth: index + 1,
            divergence,
            tokens_saved: token_count,
            descendant_prune_count: 0,
            pending_discarded,
            publication_disposition: Some(publication_disposition),
            capture_duration,
            peak_committed_pending_bytes: self.anchor_store.peak_owned_bytes(),
        };
        let restore = (|| -> Result<()> {
            maybe_inject_anchor_restore_failure(
                AnchorRestoreFaultFamily::Deepseek4,
                request_max_tokens,
            )?;
            let anchor = self
                .anchor_store
                .committed(index)
                .context("DeepSeek-V4 committed anchor disappeared during restore")?;
            self.cache.restore(&anchor.snapshot)?;
            Ok(())
        })();
        if let Err(error) = restore {
            return self.fail_closed_restore(error, observation);
        }
        let prune = match anchor_store_policy::prune_after_restore(
            &mut self.anchor_store,
            &self.anchor_budget,
            index,
        ) {
            Ok(prune) => prune,
            Err(error) => return self.fail_closed_restore(error, observation),
        };
        if self.cache.position() != token_count {
            return self.fail_closed_restore(
                anyhow::anyhow!(
                    "restored cache position {} disagrees with token ledger {}",
                    self.cache.position(),
                    token_count
                ),
                observation,
            );
        }
        self.committed_tokens = tokens;
        self.live_logits = None;
        let mut observation = observation;
        observation.descendant_prune_count = prune.pruned;
        observation.pending_discarded |= prune.pending_discarded;
        anchor_store_policy::record_restore(observation);
        Ok(())
    }

    fn fail_closed_restore(
        &mut self,
        restore_error: impl std::fmt::Display,
        mut observation: AnchorRestoreEvent,
    ) -> Result<()> {
        let cause = observation.cause;
        let restore_error = restore_error.to_string();
        let clear = anchor_store_policy::clear_all(
            &mut self.anchor_store,
            &self.anchor_budget,
            "failed-restore",
        );
        self.committed_tokens.clear();
        self.live_logits = None;
        self.request_anchor_transaction_active = false;
        let reset = self.cache.reset();
        observation.outcome = if clear.is_ok() && reset.is_ok() {
            AnchorRestoreOutcome::RestoreFailedResetSucceeded
        } else {
            AnchorRestoreOutcome::FailedCleanup
        };
        observation.hit_depth = 0;
        observation.tokens_saved = 0;
        observation.peak_committed_pending_bytes = self.anchor_store.peak_owned_bytes();
        anchor_store_policy::record_restore(observation);
        match (clear, reset) {
            (Ok(()), Ok(())) => anyhow::bail!(
                "DeepSeek-V4 {cause} failed; cache hard-reset and anchor lineage cleared: {restore_error}"
            ),
            (clear, reset) => anyhow::bail!(
                "DeepSeek-V4 {cause} failed ({restore_error}); fail-closed cleanup also failed (clear={clear:?}, reset={reset:?})"
            ),
        }
    }
}

impl Deepseek4LoadedModel {
    pub(super) fn idle_runtime_release_encoder(&self) -> Result<mlx_native::CommandEncoder> {
        self.model
            .ctx
            .device()
            .command_encoder()
            .context("create DeepSeek-V4 idle-release residency commit")
    }

    pub(super) fn idle_slot_staging_owned_bytes(&self) -> Result<(u64, u64)> {
        anyhow::ensure!(
            self.cache.position() == 0
                && self.committed_tokens.is_empty()
                && self.live_logits.is_none()
                && self.anchor_store.committed_len() == 0
                && !self.anchor_store.has_pending()
                && !self.request_anchor_transaction_active
                && self.telemetry_slot.is_none(),
            "DeepSeek-V4 SlotAware staging cache is not cold at idle park"
        );
        let prefix_bytes = (self.committed_tokens.capacity() as u64)
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .and_then(|bytes| bytes.checked_add(self.anchor_store.owned_bytes()))
            .context("DeepSeek-V4 staging-owned byte total overflow")?;
        Ok((self.cache.owned_bytes(), prefix_bytes))
    }

    pub(super) fn commit_generated_tokens_cohort(
        &mut self,
        token_ids: [u32; 4],
        sessions: &mut [Deepseek4Session],
        sorted_slot_indices: [usize; 4],
        supervisor: &EngineSupervisor,
    ) -> Result<[MlxBuffer; 4]> {
        anyhow::ensure!(
            sorted_slot_indices.windows(2).all(|pair| pair[0] < pair[1]),
            "DeepSeek-V4 decode cohort slot indices must be unique and sorted"
        );
        let mut wanted = sorted_slot_indices.into_iter().peekable();
        let mut selected = Vec::with_capacity(4);
        for (index, session) in sessions.iter_mut().enumerate() {
            if wanted.peek().copied() == Some(index) {
                anyhow::ensure!(
                    session.cache.position() == session.committed_tokens.len(),
                    "DeepSeek-V4 decode cohort slot {index} cache position {} disagrees with token ledger {}",
                    session.cache.position(),
                    session.committed_tokens.len()
                );
                session
                    .committed_tokens
                    .try_reserve(1)
                    .context("reserve DeepSeek-V4 decode cohort token ledger")?;
                selected.push(session);
                wanted.next();
            }
        }
        anyhow::ensure!(
            wanted.next().is_none() && selected.len() == 4,
            "DeepSeek-V4 decode cohort slot index is out of range"
        );
        let [lane0, lane1, lane2, lane3] = selected.as_mut_slice() else {
            unreachable!("validated four DeepSeek-V4 decode cohort sessions")
        };
        let mut caches = [
            &mut lane0.cache,
            &mut lane1.cache,
            &mut lane2.cache,
            &mut lane3.cache,
        ];
        let lease = supervisor.arm("deepseek4_decode_cohort", GPU_TRANSACTION_TIMEOUT)?;
        let state = self
            .model
            .forward_verifier_decode_cohort_with_commit_gate(token_ids, &mut caches, || {
                lease.finish()
            })
            .context("execute DeepSeek-V4 decode cohort verifier")?;
        for (session, token) in selected.iter_mut().zip(token_ids) {
            session.committed_tokens.push(token);
        }
        let logits = supervisor.run(
            "deepseek4_decode_cohort_head",
            GPU_TRANSACTION_TIMEOUT,
            || {
                self.model
                    .forward_logits(&state)
                    .context("execute DeepSeek-V4 decode cohort output head")
            },
        )?;
        let vocab = self.model.cfg.vocab_size as usize;
        anyhow::ensure!(
            logits.dtype() == mlx_native::DType::F32 && logits.shape() == [4, vocab],
            "DeepSeek-V4 decode cohort logits must be F32 [4, {vocab}], got {} {:?}",
            logits.dtype(),
            logits.shape()
        );
        let row_bytes = vocab
            .checked_mul(mlx_native::DType::F32.size_of())
            .context("DeepSeek-V4 decode cohort logits row overflow")?;
        let rows = std::array::from_fn(|lane| {
            logits
                .slice_view((lane * row_bytes) as u64, vocab)
                .with_shape(vec![1, vocab])
                .expect("validated DeepSeek-V4 decode cohort logits row")
        });
        for (session, row) in selected.iter_mut().zip(&rows) {
            session.live_logits = Some(row.clone());
        }
        Ok(rows)
    }

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
        Self::load_with_context(opts, None)
    }

    pub(crate) fn load_with_context(
        opts: &LoadOptions,
        effective_context: Option<u32>,
    ) -> Result<Self> {
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
        let context_length =
            effective_context.unwrap_or(model.cfg.max_position_embeddings) as usize;
        anyhow::ensure!(context_length > 0, "DeepSeek-V4 context must be nonzero");
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
        let aggregate_anchor_budget = anchor_aggregate_budget_bytes();
        let anchor_control_capacity = anchor_control_capacity(1, aggregate_anchor_budget);
        let anchor_store = Deepseek4AnchorStore::with_committed_capacity(anchor_control_capacity);
        let anchor_budget =
            Deepseek4AnchorBudget::new(1, aggregate_anchor_budget, anchor_store.owned_bytes())?;

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
            anchor_store,
            anchor_budget,
            request_anchor_transaction_active: false,
            telemetry_slot: None,
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
        anyhow::ensure!(
            self.anchor_store.committed_len() == 0 && !self.anchor_store.has_pending(),
            "DeepSeek-V4 SlotAware sessions must be provisioned before anchor publication"
        );
        let configured_stores = max_slots as usize + 1;
        let aggregate_anchor_budget = anchor_reprovision_budget_bytes(&self.anchor_budget);
        let anchor_control_capacity =
            anchor_control_capacity(configured_stores, aggregate_anchor_budget);
        self.anchor_store = Deepseek4AnchorStore::with_committed_capacity(anchor_control_capacity);
        let initial_anchor_owned_bytes = self
            .anchor_store
            .owned_bytes()
            .checked_mul(configured_stores as u64)
            .context("DeepSeek-V4 initial aggregate anchor control bytes overflow")?;
        self.anchor_budget = Deepseek4AnchorBudget::new(
            max_slots as usize,
            aggregate_anchor_budget,
            initial_anchor_owned_bytes,
        )?;
        let mut sessions = Vec::with_capacity(max_slots as usize);
        for slot in 0..max_slots {
            sessions.push(
                Deepseek4Session::new(
                    self,
                    Arc::clone(&self.anchor_budget),
                    anchor_control_capacity,
                    slot,
                )
                .with_context(|| format!("reserve DeepSeek-V4 logical slot {slot}"))?,
            );
        }
        tracing::info!(
            max_slots,
            logical_context_per_slot = self.context_limit(),
            anchor_aggregate_budget_bytes = aggregate_anchor_budget,
            anchor_control_capacity,
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
        let clear = anchor_store_policy::clear_all(
            &mut self.anchor_store,
            &self.anchor_budget,
            "serving-cache-reset",
        );
        self.committed_tokens.clear();
        self.live_logits = None;
        self.request_anchor_transaction_active = false;
        let reset = self
            .cache
            .reset()
            .context("reset DeepSeek-V4 serving cache");
        match (clear, reset) {
            (Ok(()), Ok(())) => Ok(()),
            (clear, reset) => anyhow::bail!(
                "DeepSeek-V4 serving reset did not complete fail-closed cleanup (clear={clear:?}, reset={reset:?})"
            ),
        }
    }

    fn reset_live_cache_and_clear_anchors(&mut self, cause: &'static str) -> Result<()> {
        let clear =
            anchor_store_policy::clear_all(&mut self.anchor_store, &self.anchor_budget, cause);
        self.committed_tokens.clear();
        self.live_logits = None;
        let reset = self
            .cache
            .reset()
            .context("reset DeepSeek-V4 serving cache");
        match (clear, reset) {
            (Ok(()), Ok(())) => Ok(()),
            (clear, reset) => anyhow::bail!(
                "DeepSeek-V4 {cause} did not complete fail-closed cleanup (clear={clear:?}, reset={reset:?})"
            ),
        }
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
        let anchor_preflight = preflight_anchor_store_migration(
            &grown,
            &self.cache,
            &self.anchor_store,
            previous,
            target,
        );
        if let Err(error) = anchor_preflight {
            tracing::warn!(
                previous_cache_context = previous,
                allocated_cache_context = target,
                error = %error,
                "DeepSeek-V4 cache growth invalidated an incompatible anchor lineage"
            );
            anchor_store_policy::clear_all(
                &mut self.anchor_store,
                &self.anchor_budget,
                "cache-growth-anchor-preflight",
            )?;
        }
        grown.migrate_from(&self.cache).with_context(|| {
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

    fn begin_request_anchor_transaction(&mut self, _prompt_tokens: &[u32]) -> Result<()> {
        anchor_store_policy::discard_pending(&mut self.anchor_store, &self.anchor_budget)?;
        self.request_anchor_transaction_active = true;
        Ok(())
    }

    pub(super) fn commit_request_anchor(&mut self) -> Result<()> {
        if !self.request_anchor_transaction_active {
            return Ok(());
        }
        self.request_anchor_transaction_active = false;
        if let Some(publication) =
            anchor_store_policy::publish_pending(&mut self.anchor_store, &self.anchor_budget)?
        {
            tracing::info!(
                target: "hf2q::serve::api::deepseek4_anchor",
                committed_anchors = self.anchor_store.committed_len(),
                anchor_owned_bytes = self.anchor_store.owned_bytes(),
                evicted = publication.evicted,
                replaced_equal_depth = publication.replaced_equal_depth,
                "DeepSeek-V4 pending anchor published after terminal ledger commit"
            );
        }
        Ok(())
    }

    pub(super) fn recover_after_cancellation(&mut self) -> Result<()> {
        self.request_anchor_transaction_active = false;
        let pending_discarded =
            anchor_store_policy::discard_pending(&mut self.anchor_store, &self.anchor_budget)?;
        let live_cursor = self.cache.position();
        let recovery_index = deepest_cancellation_anchor(&self.anchor_store, self.cache.position());
        let recovery = deepseek4_cancellation_recovery(
            self.cache.is_poisoned(),
            recovery_index.and_then(|index| {
                self.anchor_store
                    .committed(index)
                    .map(|anchor| anchor.snapshot.position())
            }),
            recovery_index
                .and_then(|index| self.anchor_store.committed(index))
                .map_or(0, |anchor| anchor.prompt_tokens.len()),
        );
        if recovery == Deepseek4CancellationRecovery::Reset {
            let divergence = AnchorDivergence::rewind(
                live_cursor,
                recovery_index
                    .and_then(|index| self.anchor_store.committed(index))
                    .map_or(0, |anchor| anchor.prompt_tokens.len()),
            );
            let reset = self.reset_live_cache();
            anchor_store_policy::record_restore(AnchorRestoreEvent {
                family: "deepseek4",
                slot: self.telemetry_slot,
                cause: "cancellation rollback",
                outcome: if reset.is_ok() {
                    AnchorRestoreOutcome::MissNoMatch
                } else {
                    AnchorRestoreOutcome::FailedCleanup
                },
                attempted_hit_depth: 0,
                hit_depth: 0,
                divergence,
                tokens_saved: 0,
                descendant_prune_count: 0,
                pending_discarded,
                publication_disposition: None,
                capture_duration: Duration::ZERO,
                peak_committed_pending_bytes: self.anchor_store.peak_owned_bytes(),
            });
            return reset;
        }
        let recovery_index =
            recovery_index.context("DeepSeek-V4 cancellation anchor disappeared")?;
        let anchor_tokens = self
            .anchor_store
            .committed(recovery_index)
            .map_or(0, |anchor| anchor.prompt_tokens.len());
        self.restore_committed_anchor(
            recovery_index,
            "cancellation rollback",
            AnchorDivergence::rewind(self.cache.position(), anchor_tokens),
            pending_discarded,
            None,
        )?;
        anchor_store_policy::cancel_at_cursor(
            &mut self.anchor_store,
            &self.anchor_budget,
            self.cache.position(),
        )?;
        Ok(())
    }

    fn capture_turn_anchor(&mut self, prompt_prefix: &[u32]) -> Result<bool> {
        capture_turn_anchor_with_source(
            &self.cache,
            &mut self.anchor_store,
            &self.anchor_budget,
            prompt_prefix,
            "snapshot DeepSeek-V4 prompt-boundary cache",
            "serial-or-swapped-prefill",
        )
    }

    fn restore_committed_anchor(
        &mut self,
        index: usize,
        cause: &'static str,
        divergence: AnchorDivergence,
        pending_discarded_before_restore: bool,
        request_max_tokens: Option<usize>,
    ) -> Result<()> {
        let (tokens, token_count, capture_duration, publication_disposition) = {
            let anchor = self
                .anchor_store
                .committed(index)
                .context("DeepSeek-V4 committed anchor disappeared before restore")?;
            (
                anchor.prompt_tokens.to_vec(),
                anchor.prompt_tokens.len(),
                anchor.capture_duration,
                anchor.publication_disposition(),
            )
        };
        let pending_discarded = pending_discarded_before_restore || self.anchor_store.has_pending();
        let observation = AnchorRestoreEvent {
            family: "deepseek4",
            slot: self.telemetry_slot,
            cause,
            outcome: AnchorRestoreOutcome::Hit,
            attempted_hit_depth: index + 1,
            hit_depth: index + 1,
            divergence,
            tokens_saved: token_count,
            descendant_prune_count: 0,
            pending_discarded,
            publication_disposition: Some(publication_disposition),
            capture_duration,
            peak_committed_pending_bytes: self.anchor_store.peak_owned_bytes(),
        };
        let restore = (|| -> Result<()> {
            maybe_inject_anchor_restore_failure(
                AnchorRestoreFaultFamily::Deepseek4,
                request_max_tokens,
            )?;
            let anchor = self
                .anchor_store
                .committed(index)
                .context("DeepSeek-V4 committed anchor disappeared during restore")?;
            self.cache.restore(&anchor.snapshot)?;
            Ok(())
        })();
        if let Err(error) = restore {
            return self.fail_closed_restore(error, observation);
        }
        let prune = match anchor_store_policy::prune_after_restore(
            &mut self.anchor_store,
            &self.anchor_budget,
            index,
        ) {
            Ok(prune) => prune,
            Err(error) => return self.fail_closed_restore(error, observation),
        };
        if self.cache.position() != token_count {
            return self.fail_closed_restore(
                anyhow::anyhow!(
                    "restored cache position {} disagrees with token ledger {}",
                    self.cache.position(),
                    token_count
                ),
                observation,
            );
        }
        self.committed_tokens = tokens;
        self.live_logits = None;
        let mut observation = observation;
        observation.descendant_prune_count = prune.pruned;
        observation.pending_discarded |= prune.pending_discarded;
        anchor_store_policy::record_restore(observation);
        Ok(())
    }

    fn fail_closed_restore(
        &mut self,
        restore_error: impl std::fmt::Display,
        mut observation: AnchorRestoreEvent,
    ) -> Result<()> {
        let cause = observation.cause;
        let restore_error = restore_error.to_string();
        let clear = anchor_store_policy::clear_all(
            &mut self.anchor_store,
            &self.anchor_budget,
            "failed-restore",
        );
        self.committed_tokens.clear();
        self.live_logits = None;
        self.request_anchor_transaction_active = false;
        let reset = self.cache.reset();
        observation.outcome = if clear.is_ok() && reset.is_ok() {
            AnchorRestoreOutcome::RestoreFailedResetSucceeded
        } else {
            AnchorRestoreOutcome::FailedCleanup
        };
        observation.hit_depth = 0;
        observation.tokens_saved = 0;
        observation.peak_committed_pending_bytes = self.anchor_store.peak_owned_bytes();
        anchor_store_policy::record_restore(observation);
        match (clear, reset) {
            (Ok(()), Ok(())) => anyhow::bail!(
                "DeepSeek-V4 {cause} failed; cache hard-reset and anchor lineage cleared: {restore_error}"
            ),
            (clear, reset) => anyhow::bail!(
                "DeepSeek-V4 {cause} failed ({restore_error}); fail-closed cleanup also failed (clear={clear:?}, reset={reset:?})"
            ),
        }
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
        self.begin_request_anchor_transaction(prompt_tokens)?;
        let cache_grew = self.ensure_cache_capacity(prompt_tokens.len(), max_tokens)?;
        let diagnostic_anchor_index = self.anchor_store.newest_committed_at_or_before(usize::MAX);
        let diagnostic_anchor =
            diagnostic_anchor_index.and_then(|index| self.anchor_store.committed(index));
        let diagnostic_anchor_tokens = diagnostic_anchor
            .map(|anchor| anchor.prompt_tokens.as_ref())
            .unwrap_or(&[]);
        progress.cache_reset_diagnostic(
            self.committed_tokens.len(),
            self.cache.position(),
            common_prefix_len(prompt_tokens, &self.committed_tokens),
            diagnostic_anchor_tokens.len(),
            diagnostic_anchor.map_or(0, |anchor| anchor.snapshot.position()),
            common_prefix_len(prompt_tokens, diagnostic_anchor_tokens),
            self.cache.is_poisoned(),
            false,
            cache_grew,
        );
        let divergence = AnchorDivergence::between(&self.committed_tokens, prompt_tokens);
        let mut observation = AnchorRestoreEvent {
            family: "deepseek4",
            slot: self.telemetry_slot,
            cause: "cold-resumable-prefill",
            outcome: AnchorRestoreOutcome::MissNoMatch,
            attempted_hit_depth: 0,
            hit_depth: 0,
            divergence,
            tokens_saved: 0,
            descendant_prune_count: 0,
            pending_discarded: self.anchor_store.has_pending(),
            publication_disposition: None,
            capture_duration: Duration::ZERO,
            peak_committed_pending_bytes: self.anchor_store.peak_owned_bytes(),
        };
        let reset = self.reset_live_cache_and_clear_anchors("cold-resumable-prefill");
        if reset.is_err() {
            observation.outcome = AnchorRestoreOutcome::FailedCleanup;
        }
        anchor_store_policy::record_restore(observation);
        reset?;
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
        self.begin_request_anchor_transaction(prompt_tokens)?;
        let cache_grew = self.ensure_cache_capacity(prompt_tokens.len(), max_tokens)?;
        let cache_poisoned = self.cache.is_poisoned();
        let live_position = self.cache.position();
        let selectable_live_position = if cache_poisoned {
            usize::MAX
        } else {
            live_position
        };
        let selected_reuse = select_prefix_reuse_from_store(
            prompt_tokens,
            &self.committed_tokens,
            selectable_live_position,
            &self.anchor_store,
        );
        let reuse = prefer_matrix_prefill(prompt_tokens.len(), selected_reuse);
        let (cached_tokens, cache_action) = match reuse {
            PrefixReuse::Live(count) => (count, if cache_grew { "grow-live" } else { "live" }),
            PrefixReuse::RecoveryAnchor { count, index } => {
                let divergence = AnchorDivergence::between(&self.committed_tokens, prompt_tokens);
                self.restore_committed_anchor(
                    index,
                    "cached resumable restore",
                    divergence,
                    false,
                    Some(max_tokens),
                )?;
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
                if self.capture_turn_anchor(&prompt_tokens[..plan.recovery_position])? {
                    progress.recovery_anchor_captured(plan.recovery_position);
                }
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
            if self.capture_turn_anchor(&prompt_tokens[..plan.recovery_position])? {
                progress.recovery_anchor_captured(plan.recovery_position);
            }
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
        self.begin_request_anchor_transaction(prompt_tokens)?;
        let cache_grew = self.ensure_cache_capacity(prompt_tokens.len(), max_tokens)?;

        let cache_poisoned = self.cache.is_poisoned();
        let live_position = self.cache.position();
        let selectable_live_position = if cache_poisoned {
            usize::MAX
        } else {
            live_position
        };
        let selected_reuse = select_prefix_reuse_from_store(
            prompt_tokens,
            &self.committed_tokens,
            selectable_live_position,
            &self.anchor_store,
        );
        let reuse = prefer_matrix_prefill(prompt_tokens.len(), selected_reuse);
        if matches!(reuse, PrefixReuse::Reset) {
            let diagnostic_anchor_index =
                self.anchor_store.newest_committed_at_or_before(usize::MAX);
            let diagnostic_anchor =
                diagnostic_anchor_index.and_then(|index| self.anchor_store.committed(index));
            let diagnostic_anchor_tokens = diagnostic_anchor
                .map(|anchor| anchor.prompt_tokens.as_ref())
                .unwrap_or(&[]);
            progress.cache_reset_diagnostic(
                self.committed_tokens.len(),
                live_position,
                common_prefix_len(prompt_tokens, &self.committed_tokens),
                diagnostic_anchor_tokens.len(),
                diagnostic_anchor.map_or(0, |anchor| anchor.snapshot.position()),
                common_prefix_len(prompt_tokens, diagnostic_anchor_tokens),
                cache_poisoned,
                selected_reuse != reuse,
                cache_grew,
            );
        }
        let reset_from_scratch = matches!(reuse, PrefixReuse::Reset);
        let (cached_tokens, cache_action) = match reuse {
            PrefixReuse::Live(count) => (count, if cache_grew { "grow-live" } else { "live" }),
            PrefixReuse::RecoveryAnchor { count, index } => {
                let divergence = AnchorDivergence::between(&self.committed_tokens, prompt_tokens);
                self.restore_committed_anchor(
                    index,
                    "prompt-boundary restore",
                    divergence,
                    false,
                    Some(max_tokens),
                )?;
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
                let divergence = AnchorDivergence::between(&self.committed_tokens, prompt_tokens);
                let mut observation = AnchorRestoreEvent {
                    family: "deepseek4",
                    slot: self.telemetry_slot,
                    cause: "cold-prefill",
                    outcome: AnchorRestoreOutcome::MissNoMatch,
                    attempted_hit_depth: 0,
                    hit_depth: 0,
                    divergence,
                    tokens_saved: 0,
                    descendant_prune_count: 0,
                    pending_discarded: self.anchor_store.has_pending(),
                    publication_disposition: None,
                    capture_duration: Duration::ZERO,
                    peak_committed_pending_bytes: self.anchor_store.peak_owned_bytes(),
                };
                let reset = self.reset_live_cache_and_clear_anchors("cold-prefill");
                if reset.is_err() {
                    observation.outcome = AnchorRestoreOutcome::FailedCleanup;
                }
                anchor_store_policy::record_restore(observation);
                reset?;
                (0, if cache_grew { "grow-reset" } else { "reset" })
            }
        };
        let recovery_position = prompt_tokens.len().saturating_sub(RECOVERY_TAIL_TOKENS);
        let native_prefill_rows = self.model.cfg.sliding_window as usize;
        let short_recovery_candidate =
            reset_from_scratch && recovery_position > 0 && recovery_position < native_prefill_rows;
        let short_recovery_admission = short_recovery_candidate
            .then(|| {
                preflight_anchor_capture(
                    &self.cache,
                    &self.anchor_store,
                    &self.anchor_budget,
                    recovery_position,
                )
            })
            .transpose()?;
        let short_recovery_execution = plan_short_recovery_execution(
            short_recovery_candidate,
            short_recovery_admission.map(|(admission, _)| admission),
        )?;
        let short_recovery_prepass =
            short_recovery_execution == ShortRecoveryExecution::CapturePrepassThenFullPrompt;
        if let Some((admission, anchor_bytes)) = short_recovery_admission.filter(|(admission, _)| {
            matches!(
                admission,
                StagePending::NoCommittedCapacity | StagePending::BudgetExceeded { .. }
            )
        }) {
            anchor_store_policy::record_preflight_budget_skip(
                &self.anchor_store,
                &self.anchor_budget,
                anchor_bytes,
                admission,
                "short-recovery-prepass",
            );
        }
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

        if short_recovery_execution == ShortRecoveryExecution::FullPromptOnly {
            let final_state = self
                .append_prompt_tokens(prompt_tokens, &cancelled, progress, supervisor)?
                .context("DeepSeek-V4 prompt produced no verifier state")?;
            return self.finish_prefill(final_state, 0, supervisor);
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
            // Keep the prepass checkpoint request-local and outside the store
            // while the identical full prompt replay resets the mutable log.
            // Publication remains impossible until replay and terminal request
            // success both complete.
            let recovery_anchor = capture_anchor_candidate(
                &self.cache,
                &prompt_tokens[..recovery_position],
                "snapshot DeepSeek-V4 short recovery prepass",
            )?;
            self.cache
                .reset()
                .context("reset DeepSeek-V4 live cache after recovery prepass")?;
            self.committed_tokens.clear();
            self.live_logits = None;

            let final_state = self
                .append_prompt_tokens(prompt_tokens, &cancelled, progress, supervisor)?
                .context("DeepSeek-V4 prompt produced no verifier state")?;
            let outcome = anchor_store_policy::stage_pending(
                &mut self.anchor_store,
                &self.anchor_budget,
                recovery_anchor,
                "short-recovery-prepass",
            )?;
            anyhow::ensure!(
                outcome == StagePending::Staged,
                "DeepSeek-V4 short-recovery admission changed before staging: {outcome:?}"
            );
            progress.recovery_anchor_captured(recovery_position);
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
            if self.capture_turn_anchor(&prompt_tokens[..recovery_position])? {
                progress.recovery_anchor_captured(recovery_position);
            }
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

    pub(super) fn commit_generated_token(
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
    RecoveryAnchor { count: usize, index: usize },
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
        PrefixReuse::Live(count) | PrefixReuse::RecoveryAnchor { count, .. } => count,
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
        PrefixReuse::RecoveryAnchor {
            count: anchor.len(),
            index: 0,
        }
    } else {
        PrefixReuse::Reset
    }
}

fn select_prefix_reuse_from_store(
    prompt: &[u32],
    live: &[u32],
    live_position: usize,
    store: &Deepseek4AnchorStore,
) -> PrefixReuse {
    let live = (!live.is_empty() && live_position == live.len() && prompt.starts_with(live))
        .then_some(live.len());
    let recovery = deepest_recovery_anchor(store, prompt).and_then(|index| {
        store
            .committed(index)
            .map(|anchor| (anchor.prompt_tokens.len(), index))
    });
    match (live, recovery) {
        (Some(live), Some((recovery, _))) if live >= recovery => PrefixReuse::Live(live),
        (_, Some((count, index))) => PrefixReuse::RecoveryAnchor { count, index },
        (Some(live), None) => PrefixReuse::Live(live),
        (None, None) => PrefixReuse::Reset,
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
            max_context_length: crate::serve::operator_settings::declared_context_length(gguf)
                .ok()
                .or_else(|| self.context_length.map(|value| value as u32)),
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
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use mlx_native::MlxDevice;
    use tracing_subscriber::prelude::*;

    use crate::inference::models::deepseek4::{
        cache::{Deepseek4Cache, Deepseek4CachePlan},
        Deepseek4Config,
    };

    use super::super::anchor_store::StagePending;
    use super::super::deepseek4_anchor_store::{
        assert_budget_conservation, publish_pending, stage_pending, Deepseek4AnchorBudget,
        Deepseek4AnchorStore, Deepseek4PromptAnchor, DEFAULT_MAX_COMMITTED_ANCHORS,
    };
    use super::{
        anchor_reprovision_budget_bytes, balance_fresh_three_chunk_prefill,
        cache_capacity_for_request, common_prefix_len, deepseek4_cancellation_recovery,
        plan_resumable_prefill_chunk, plan_short_recovery_execution, prefer_matrix_prefill,
        preflight_anchor_store_migration,
        resumable_matrix_prefill_chunk_len, select_prefix_reuse, select_prefix_reuse_from_store,
        should_retain_decode_scratch, AnchorDivergence, Deepseek4CancellationRecovery,
        Deepseek4Session, PrefixReuse, ResumablePrefillChunk, ResumablePrefillSlice,
        ShortRecoveryExecution, TransientScratchStats, MIN_MATRIX_APPEND_TOKENS,
        RECOVERY_TAIL_TOKENS,
    };

    #[derive(Clone, Default)]
    struct AnchorEventLayer {
        events: Arc<Mutex<Vec<BTreeMap<String, String>>>>,
    }

    struct AnchorFieldVisitor<'a> {
        fields: &'a mut BTreeMap<String, String>,
    }

    impl tracing::field::Visit for AnchorFieldVisitor<'_> {
        fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }

        fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }

        fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }

        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }

        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.fields
                .insert(field.name().to_string(), format!("{value:?}"));
        }
    }

    impl<S> tracing_subscriber::Layer<S> for AnchorEventLayer
    where
        S: tracing::Subscriber,
    {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut fields = BTreeMap::new();
            event.record(&mut AnchorFieldVisitor {
                fields: &mut fields,
            });
            if fields
                .get("family")
                .is_some_and(|family| family == "deepseek4")
            {
                self.events.lock().expect("events lock").push(fields);
            }
        }
    }

    fn capture_anchor_events<T>(f: impl FnOnce() -> T) -> (T, Vec<BTreeMap<String, String>>) {
        let layer = AnchorEventLayer::default();
        let events = Arc::clone(&layer.events);
        let subscriber = tracing_subscriber::registry().with(layer);
        let result = tracing::subscriber::with_default(subscriber, f);
        let captured = events.lock().expect("events lock").clone();
        (result, captured)
    }

    fn test_config() -> Deepseek4Config {
        Deepseek4Config {
            num_hidden_layers: 2,
            hidden_size: 64,
            hidden_size_out: 256,
            max_position_embeddings: 256,
            vocab_size: 128,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            head_dim: 8,
            rope_head_dim: 4,
            rope_theta: 10_000.0,
            rope_factor: 1.0,
            original_context_length: 256,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            q_lora_rank: 16,
            o_lora_rank: 16,
            output_groups: 2,
            sliding_window: 128,
            compress_ratios: vec![4, 128],
            compress_rope_theta: 160_000.0,
            index_num_heads: 8,
            index_head_dim: 4,
            index_top_k: 8,
            rms_norm_eps: 1e-6,
            num_experts: 4,
            num_experts_per_tok: 2,
            num_shared_experts: 1,
            expert_intermediate_size: 32,
            route_scale: 1.0,
            normalize_topk: true,
            swiglu_clamp_experts: vec![10.0; 2],
            swiglu_clamp_shared: vec![10.0; 2],
            hyper_connection_count: 2,
            hyper_connection_sinkhorn_iterations: 2,
            hyper_connection_epsilon: 1e-6,
            hash_layer_count: 2,
        }
    }

    fn test_cache() -> Deepseek4Cache {
        let plan = Deepseek4CachePlan::for_context(&test_config(), 128).unwrap();
        Deepseek4Cache::allocate(&plan, MlxDevice::new().unwrap()).unwrap()
    }

    fn new_test_session() -> Deepseek4Session {
        let store = Deepseek4AnchorStore::with_committed_capacity(DEFAULT_MAX_COMMITTED_ANCHORS);
        let budget = Deepseek4AnchorBudget::new(1, 64 * 1024 * 1024, store.owned_bytes()).unwrap();
        Deepseek4Session {
            cache: test_cache(),
            committed_tokens: Vec::new(),
            live_logits: None,
            anchor_store: store,
            anchor_budget: budget,
            request_anchor_transaction_active: false,
            telemetry_slot: Some(7),
        }
    }

    #[test]
    fn zero_depth_anchor_grant_skips_deepseek_snapshot_before_copy() {
        let store = Deepseek4AnchorStore::with_committed_capacity(0);
        let budget = Deepseek4AnchorBudget::new(4, 1_024, store.owned_bytes()).unwrap();
        let admission = budget.preflight_capture(&store, 2_048);
        assert_eq!(admission, StagePending::NoCommittedCapacity);

        let mut called = false;
        let captured = super::super::anchor_store::capture_if_anchor_admitted::<u64>(
            admission,
            || {
                called = true;
                Ok(99)
            },
        )
        .unwrap();
        assert_eq!(captured, None);
        assert!(!called, "DeepSeek snapshot must not run at effective K=0");
    }

    #[test]
    fn slot_reprovision_keeps_the_worker_lifetime_anchor_grant() {
        let store = Deepseek4AnchorStore::with_committed_capacity(0);
        let frozen_grant = 987_654_u64;
        let budget = Deepseek4AnchorBudget::new(4, frozen_grant, store.owned_bytes()).unwrap();
        assert_eq!(anchor_reprovision_budget_bytes(&budget), frozen_grant);
    }

    #[test]
    fn short_recovery_capacity_only_changes_checkpoint_work() {
        assert_eq!(
            plan_short_recovery_execution(false, None).unwrap(),
            ShortRecoveryExecution::Standard
        );
        assert_eq!(
            plan_short_recovery_execution(
                true,
                Some(StagePending::NoCommittedCapacity),
            )
            .unwrap(),
            ShortRecoveryExecution::FullPromptOnly,
            "K=0 must retain one authoritative full-prompt append without a snapshot prepass"
        );
        assert_eq!(
            plan_short_recovery_execution(true, Some(StagePending::Staged)).unwrap(),
            ShortRecoveryExecution::CapturePrepassThenFullPrompt,
            "admitted capture adds only the request-local prefix prepass"
        );
        assert!(plan_short_recovery_execution(
            true,
            Some(StagePending::PendingOccupied),
        )
        .is_err());
    }

    fn publish_current_anchor(session: &mut Deepseek4Session, tokens: &[u32]) {
        assert_eq!(session.cache.position(), tokens.len());
        let snapshot = session.cache.snapshot().unwrap();
        let anchor = Deepseek4PromptAnchor::new(tokens, snapshot, Duration::ZERO);
        assert_eq!(
            stage_pending(
                &mut session.anchor_store,
                &session.anchor_budget,
                anchor,
                "test",
            )
            .unwrap(),
            StagePending::Staged
        );
        assert!(
            publish_pending(&mut session.anchor_store, &session.anchor_budget)
                .unwrap()
                .is_some()
        );
    }

    fn advance_to(cache: &mut Deepseek4Cache, depth: usize) {
        while cache.position() < depth {
            cache.commit_step(cache.position()).unwrap();
        }
    }

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
            PrefixReuse::RecoveryAnchor { count: 2, index: 0 }
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
            prefer_matrix_prefill(306, PrefixReuse::RecoveryAnchor { count: 1, index: 0 }),
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
            prefer_matrix_prefill(
                180,
                PrefixReuse::RecoveryAnchor {
                    count: 140,
                    index: 0,
                }
            ),
            PrefixReuse::RecoveryAnchor {
                count: 140,
                index: 0,
            }
        );
    }

    #[test]
    fn deepseek_anchor_lineage_rewind_rejects_old_descendants() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let mut session = new_test_session();
        let lineage = [10, 11, 12, 13, 14, 15];
        for depth in [2, 4, 6] {
            advance_to(&mut session.cache, depth);
            session.committed_tokens = lineage[..depth].to_vec();
            publish_current_anchor(&mut session, &lineage[..depth]);
        }
        assert_eq!(session.anchor_store.committed_token_counts(), vec![2, 4, 6]);

        let branch = [10, 11, 99];
        let PrefixReuse::RecoveryAnchor { count: 2, index } = select_prefix_reuse_from_store(
            &branch,
            &session.committed_tokens,
            session.cache.position(),
            &session.anchor_store,
        ) else {
            panic!("branch must select the oldest surviving ancestor");
        };
        let divergence = AnchorDivergence::between(&session.committed_tokens, &branch);
        session
            .restore_committed_anchor(index, "lineage regression", divergence, false, None)
            .unwrap();
        assert_eq!(session.anchor_store.committed_token_counts(), vec![2]);

        session.cache.commit_step(2).unwrap();
        session.committed_tokens.push(99);
        let old_c = [10, 11, 12, 13, 14, 15, 16];
        assert_eq!(
            select_prefix_reuse_from_store(
                &old_c,
                &session.committed_tokens,
                session.cache.position(),
                &session.anchor_store,
            ),
            PrefixReuse::RecoveryAnchor { count: 2, index: 0 },
            "old B/C checkpoints must not regain authority after branch X writes"
        );
    }

    #[test]
    fn pending_anchor_is_affinity_invisible_and_cancellation_preserves_committed_set() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let mut session = new_test_session();
        advance_to(&mut session.cache, 2);
        publish_current_anchor(&mut session, &[1, 2]);
        advance_to(&mut session.cache, 6);
        session.committed_tokens = vec![9; 6];

        let snapshot = session.cache.snapshot().unwrap();
        let pending = Deepseek4PromptAnchor::new(&[1, 2, 3, 4, 5, 6], snapshot, Duration::ZERO);
        assert_eq!(
            stage_pending(
                &mut session.anchor_store,
                &session.anchor_budget,
                pending,
                "test-cancellation",
            )
            .unwrap(),
            StagePending::Staged
        );
        assert_eq!(
            session.reusable_prefix_len(&[1, 2, 3, 4, 5, 6, 7]),
            2,
            "request-local pending checkpoint leaked into affinity"
        );

        let committed_before = session.anchor_store.committed_token_counts();
        let expected_peak = session.anchor_store.peak_owned_bytes();
        let (recovery, events) = capture_anchor_events(|| session.recover_after_cancellation());
        recovery.unwrap();
        assert_eq!(
            events.len(),
            1,
            "cancellation restore must emit exactly once"
        );
        assert_eq!(events[0]["slot"], "Some(7)");
        assert_eq!(events[0]["outcome"], "hit");
        assert_eq!(events[0]["pending_discarded"], "true");
        assert_eq!(
            events[0]["peak_committed_pending_bytes"],
            expected_peak.to_string()
        );
        assert_eq!(
            session.anchor_store.committed_token_counts(),
            committed_before
        );
        assert!(!session.anchor_store.has_pending());
        assert_eq!(session.committed_tokens, vec![1, 2]);
    }

    #[test]
    fn growth_preflights_every_committed_and_pending_store_anchor() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let mut session = new_test_session();
        for depth in [2, 4] {
            advance_to(&mut session.cache, depth);
            let tokens = (0..depth as u32).collect::<Vec<_>>();
            session.committed_tokens.clone_from(&tokens);
            publish_current_anchor(&mut session, &tokens);
        }
        advance_to(&mut session.cache, 6);
        session.committed_tokens = (0..6).collect();
        let pending = Deepseek4PromptAnchor::new(
            &session.committed_tokens,
            session.cache.snapshot().unwrap(),
            Duration::ZERO,
        );
        assert_eq!(
            stage_pending(
                &mut session.anchor_store,
                &session.anchor_budget,
                pending,
                "test-growth",
            )
            .unwrap(),
            StagePending::Staged
        );

        let target_plan = Deepseek4CachePlan::for_context(&test_config(), 256).unwrap();
        let mut grown = Deepseek4Cache::allocate(&target_plan, MlxDevice::new().unwrap()).unwrap();
        preflight_anchor_store_migration(&grown, &session.cache, &session.anchor_store, 128, 256)
            .unwrap();
        grown.migrate_from(&session.cache).unwrap();

        for (index, expected) in [2, 4].into_iter().enumerate() {
            grown
                .restore(&session.anchor_store.committed(index).unwrap().snapshot)
                .unwrap();
            assert_eq!(grown.position(), expected);
        }
        grown
            .restore(&session.anchor_store.pending().unwrap().snapshot)
            .unwrap();
        assert_eq!(grown.position(), 6);
        assert_budget_conservation(&session.anchor_budget, [session.anchor_store.owned_bytes()]);
    }

    #[test]
    fn late_restore_failure_hard_resets_and_clears_the_whole_store() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let mut session = new_test_session();
        advance_to(&mut session.cache, 7);
        session.committed_tokens = (0..7).collect();
        let mut snapshot = session.cache.snapshot().unwrap();
        snapshot.corrupt_last_layer_window_shape_for_test();
        let anchor =
            Deepseek4PromptAnchor::new(&session.committed_tokens, snapshot, Duration::ZERO);
        assert_eq!(
            stage_pending(
                &mut session.anchor_store,
                &session.anchor_budget,
                anchor,
                "test-failed-restore",
            )
            .unwrap(),
            StagePending::Staged
        );
        publish_pending(&mut session.anchor_store, &session.anchor_budget)
            .unwrap()
            .unwrap();
        advance_to(&mut session.cache, 8);
        session.committed_tokens.push(7);

        let divergence = AnchorDivergence::rewind(session.cache.position(), 7);
        let (restore, events) = capture_anchor_events(|| {
            session.restore_committed_anchor(
                0,
                "injected late-layer restore",
                divergence,
                false,
                None,
            )
        });
        let error = restore.unwrap_err();
        assert_eq!(events.len(), 1, "failed restore must emit exactly once");
        assert_eq!(events[0]["slot"], "Some(7)");
        assert_eq!(events[0]["outcome"], "restore_failed_reset_succeeded");
        assert!(error.to_string().contains("hard-reset"));
        assert_eq!(session.cache.position(), 0);
        assert!(session.committed_tokens.is_empty());
        assert_eq!(session.anchor_store.committed_len(), 0);
        assert!(!session.anchor_store.has_pending());
        assert_budget_conservation(&session.anchor_budget, [session.anchor_store.owned_bytes()]);
    }

    #[test]
    fn cold_reset_and_poison_both_invalidate_anchor_authority() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let mut cold = new_test_session();
        advance_to(&mut cold.cache, 2);
        cold.committed_tokens = vec![1, 2];
        publish_current_anchor(&mut cold, &[1, 2]);
        cold.reset().unwrap();
        assert_eq!(cold.anchor_store.committed_len(), 0);
        assert_eq!(cold.cache.position(), 0);

        let mut no_anchor = new_test_session();
        advance_to(&mut no_anchor.cache, 1);
        let (reset, no_anchor_events) =
            capture_anchor_events(|| no_anchor.recover_after_cancellation());
        reset.unwrap();
        assert_eq!(
            no_anchor_events.len(),
            1,
            "no-anchor cancellation reset must emit exactly once"
        );
        assert_eq!(no_anchor_events[0]["slot"], "Some(7)");
        assert_eq!(no_anchor_events[0]["outcome"], "miss_no_match");

        let mut poisoned = new_test_session();
        advance_to(&mut poisoned.cache, 2);
        poisoned.committed_tokens = vec![3, 4];
        publish_current_anchor(&mut poisoned, &[3, 4]);
        poisoned.cache.poison_for_test();
        let (reset, poisoned_events) =
            capture_anchor_events(|| poisoned.recover_after_cancellation());
        reset.unwrap();
        assert_eq!(
            poisoned_events.len(),
            1,
            "poison cancellation reset must emit exactly once"
        );
        assert_eq!(poisoned_events[0]["slot"], "Some(7)");
        assert_eq!(poisoned_events[0]["outcome"], "miss_no_match");
        assert_eq!(poisoned.anchor_store.committed_len(), 0);
        assert_eq!(poisoned.cache.position(), 0);
        assert!(!poisoned.cache.is_poisoned());
    }

    #[test]
    fn slot_anchor_stores_are_family_isolated_and_aggregate_bytes_conserve() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let store_a = Deepseek4AnchorStore::with_committed_capacity(DEFAULT_MAX_COMMITTED_ANCHORS);
        let store_b = Deepseek4AnchorStore::with_committed_capacity(DEFAULT_MAX_COMMITTED_ANCHORS);
        let initial_owned = store_a
            .owned_bytes()
            .checked_add(store_b.owned_bytes())
            .unwrap();
        let budget = Deepseek4AnchorBudget::new(2, 64 * 1024 * 1024, initial_owned).unwrap();
        let mut a = Deepseek4Session {
            cache: test_cache(),
            committed_tokens: Vec::new(),
            live_logits: None,
            anchor_store: store_a,
            anchor_budget: Arc::clone(&budget),
            request_anchor_transaction_active: false,
            telemetry_slot: Some(0),
        };
        let mut b = Deepseek4Session {
            cache: test_cache(),
            committed_tokens: Vec::new(),
            live_logits: None,
            anchor_store: store_b,
            anchor_budget: Arc::clone(&budget),
            request_anchor_transaction_active: false,
            telemetry_slot: Some(1),
        };
        advance_to(&mut a.cache, 2);
        a.committed_tokens = vec![1, 2];
        publish_current_anchor(&mut a, &[1, 2]);
        advance_to(&mut b.cache, 3);
        b.committed_tokens = vec![7, 8, 9];
        publish_current_anchor(&mut b, &[7, 8, 9]);
        assert_budget_conservation(
            &budget,
            [a.anchor_store.owned_bytes(), b.anchor_store.owned_bytes()],
        );

        a.reset().unwrap();
        assert_eq!(a.anchor_store.committed_len(), 0);
        assert_eq!(b.anchor_store.committed_token_counts(), vec![3]);
        assert_eq!(b.reusable_prefix_len(&[7, 8, 9, 10]), 3);
        assert_budget_conservation(
            &budget,
            [a.anchor_store.owned_bytes(), b.anchor_store.owned_bytes()],
        );
    }

    #[test]
    fn anchor_equality_never_impersonates_a_live_logit_hit() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let mut session = new_test_session();
        advance_to(&mut session.cache, 2);
        session.committed_tokens = vec![9, 9];
        publish_current_anchor(&mut session, &[1, 2]);
        assert_eq!(session.reusable_prefix_len(&[1, 2]), 0);
        session.committed_tokens = vec![1, 2];
        assert_eq!(session.reusable_prefix_len(&[1, 2]), 2);
    }
}
