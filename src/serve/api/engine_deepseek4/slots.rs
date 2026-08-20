//! Interleavable DeepSeek-V4 request state for full-context agent slots.
//!
//! The verifier model is shared, while the caller swaps the selected slot's
//! cache into `Deepseek4LoadedModel` for one tick. Everything that used to
//! live on the stack of `generate_stream`/`generate_once` is retained here so
//! another agent can run between decode tokens without changing tokenization,
//! sampling, reasoning splitting, or DSML tool routing.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use mlx_native::MlxBuffer;
use tokio::sync::mpsc;

use crate::serve::api::engine::{GenerationResult, SamplingParams};
use crate::serve::api::engine_supervisor::EngineSupervisor;
use crate::serve::api::grammar::GrammarRuntime;
use crate::serve::api::registry::ModelRegistration;
use crate::serve::api::sse::GenerationEvent;
use crate::serve::sampler_pure;

use super::progress::RequestProgress;
use super::sampling::{
    accept_forced_token, accepted_single_tool_call_is_terminal, decode_token_limit,
    grammar_runtime, sample, sampler_config, split_reasoning,
};
use super::stream::StreamRouter;
use super::{
    plan_resumable_prefill_chunk, release_completed_prefill_scratch, Deepseek4LoadedModel,
    Deepseek4ResumablePrefill, Deepseek4ResumablePrefillAdvance, RequestScratchGuard,
    ResumablePrefillChunk,
};

pub(crate) struct Deepseek4SlotCompletion {
    pub(crate) result: GenerationResult,
    pub(crate) semantic_ttft: Duration,
}

pub(crate) enum Deepseek4PrefillAdvance {
    Pending {
        state: Deepseek4PrefillState,
        advanced_tokens: usize,
    },
    Ready {
        state: Deepseek4SlotState,
        advanced_tokens: usize,
        cold_wave: bool,
    },
}

pub(crate) struct Deepseek4PrefillState {
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
    stream: bool,
    request_started: Instant,
    prefill_started: Instant,
    progress: RequestProgress,
    scratch_guard: RequestScratchGuard,
    plan: Deepseek4ResumablePrefill,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Deepseek4CooperativePrefillPlan {
    start: usize,
    token_count: usize,
    capture_anchor_after: bool,
}

impl Deepseek4CooperativePrefillPlan {
    pub(crate) fn start(self) -> usize {
        self.start
    }

    pub(crate) fn token_count(self) -> usize {
        self.token_count
    }

    pub(crate) fn end(self) -> usize {
        self.start + self.token_count
    }

    pub(crate) fn captures_anchor(self) -> bool {
        self.capture_anchor_after
    }
}

impl Deepseek4PrefillState {
    pub(crate) fn begin(
        loaded: &mut Deepseek4LoadedModel,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        stream: bool,
    ) -> Result<Self> {
        let scratch_guard = RequestScratchGuard::new();
        let request_started = Instant::now();
        let mut progress = RequestProgress::start(
            if stream {
                "slot-stream-yielding-prefill"
            } else {
                "slot-unary-yielding-prefill"
            },
            prompt_tokens.len(),
            params.max_tokens,
        );
        let plan = loaded.begin_resumable_cold_prefill(
            &prompt_tokens,
            params.max_tokens,
            &mut progress,
        )?;
        Ok(Self {
            prompt_tokens,
            params,
            stream,
            request_started,
            prefill_started: Instant::now(),
            progress,
            scratch_guard,
            plan,
        })
    }

    pub(crate) fn begin_cached(
        loaded: &mut Deepseek4LoadedModel,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        stream: bool,
    ) -> Result<Self> {
        let scratch_guard = RequestScratchGuard::new();
        let request_started = Instant::now();
        let mut progress = RequestProgress::start(
            if stream {
                "slot-stream-yielding-cached-prefill"
            } else {
                "slot-unary-yielding-cached-prefill"
            },
            prompt_tokens.len(),
            params.max_tokens,
        );
        let plan = loaded.begin_resumable_cached_prefill(
            &prompt_tokens,
            params.max_tokens,
            &mut progress,
        )?;
        Ok(Self {
            prompt_tokens,
            params,
            stream,
            request_started,
            prefill_started: Instant::now(),
            progress,
            scratch_guard,
            plan,
        })
    }

    pub(crate) fn initial_cached_tokens(&self) -> usize {
        self.plan.initial_cached_tokens()
    }

    pub(crate) fn is_cold_wave(&self) -> bool {
        self.plan.is_cold_wave()
    }

    pub(crate) fn uncached_tokens(&self) -> usize {
        self.prompt_tokens
            .len()
            .saturating_sub(self.plan.initial_cached_tokens())
    }

    /// True only while this request owns the bounded incremental suffix after
    /// its rollback anchor. The tail is at most `RECOVERY_TAIL_TOKENS`; a
    /// completed prefill is no longer active tail work.
    pub(crate) fn is_recovery_tail(&self) -> bool {
        self.plan.cursor >= self.plan.recovery_position
            && self.plan.cursor < self.prompt_tokens.len()
    }

    /// Cursor metadata for timer-free alignment of staggered warm matrix
    /// prefills. Cold waves retain their four-lane admission barrier, and a
    /// recovery tail is incremental work rather than an alignable segment.
    pub(crate) fn warm_matrix_alignment_state(&self) -> Option<(usize, usize, usize)> {
        (!self.plan.is_cold_wave() && self.plan.cursor < self.plan.recovery_position).then_some((
            self.plan.cursor,
            self.plan.recovery_position,
            self.plan.window_multiplier,
        ))
    }

    /// Validate that the existing serial prefill planner will land exactly on
    /// `target_cursor`. Near a recovery boundary it may intentionally shorten
    /// a nominal window-aligned chunk to preserve a legal incremental tail;
    /// such a chunk is not an alignment catch-up.
    pub(crate) fn exact_warm_matrix_alignment_window_cap(
        &self,
        target_cursor: usize,
        sliding_window: usize,
    ) -> Option<usize> {
        let (cursor, recovery_position, window_multiplier) = self.warm_matrix_alignment_state()?;
        let delta = target_cursor.checked_sub(cursor)?;
        if delta == 0 || sliding_window == 0 || delta % sliding_window != 0 {
            return None;
        }
        let window_cap = (delta / sliding_window).min(window_multiplier);
        if window_cap == 0 {
            return None;
        }
        let slice = plan_resumable_prefill_chunk(
            cursor,
            recovery_position.checked_sub(cursor)?,
            sliding_window,
            window_cap,
            recovery_position,
        )
        .ok()?;
        matches!(slice.chunk, ResumablePrefillChunk::Matrix(tokens) if tokens == delta)
            .then_some(window_cap)
    }

    /// Plan a cooperative FFN chunk without changing request, cache, or
    /// scheduler state. Aligned matrix work through the recovery anchor is
    /// eligible; incremental replay and final head publication stay on the
    /// established serial path.
    pub(crate) fn plan_cooperative_prefill(
        &self,
        cache_position: usize,
        committed_tokens: usize,
        sliding_window: usize,
        max_rows: usize,
    ) -> Result<Option<Deepseek4CooperativePrefillPlan>> {
        anyhow::ensure!(
            cache_position == self.plan.cursor && committed_tokens == self.plan.cursor,
            "DeepSeek-V4 cooperative candidate cursor {} disagrees with cache {} / token ledger {}",
            self.plan.cursor,
            cache_position,
            committed_tokens
        );
        if self.plan.cursor >= self.plan.recovery_position || max_rows < sliding_window {
            return Ok(None);
        }
        let remaining = self.plan.recovery_position - self.plan.cursor;
        let window_multiplier = self.plan.window_multiplier.min(max_rows / sliding_window);
        let slice = plan_resumable_prefill_chunk(
            cache_position,
            remaining,
            sliding_window,
            window_multiplier,
            self.plan.recovery_position,
        )?;
        let ResumablePrefillChunk::Matrix(token_count) = slice.chunk else {
            return Ok(None);
        };
        let plan = Deepseek4CooperativePrefillPlan {
            start: self.plan.cursor,
            token_count,
            capture_anchor_after: slice.capture_anchor_after,
        };
        if token_count > max_rows || plan.end() > self.plan.recovery_position {
            return Ok(None);
        }
        Ok(Some(plan))
    }

    pub(crate) fn cooperative_tokens(
        &self,
        plan: Deepseek4CooperativePrefillPlan,
    ) -> Result<&[u32]> {
        anyhow::ensure!(
            plan.start == self.plan.cursor && plan.end() <= self.prompt_tokens.len(),
            "DeepSeek-V4 cooperative prefill plan no longer matches request cursor"
        );
        Ok(&self.prompt_tokens[plan.start..plan.end()])
    }

    pub(crate) fn validate_cooperative_prefill_commit(
        &self,
        plan: Deepseek4CooperativePrefillPlan,
    ) -> Result<()> {
        anyhow::ensure!(
            plan.start == self.plan.cursor
                && plan.end() <= self.plan.recovery_position
                && plan.captures_anchor() == (plan.end() == self.plan.recovery_position),
            "DeepSeek-V4 cooperative publication crossed a recovery checkpoint"
        );
        Ok(())
    }

    pub(crate) fn cooperative_anchor_tokens(
        &self,
        plan: Deepseek4CooperativePrefillPlan,
    ) -> Result<Option<&[u32]>> {
        if !plan.captures_anchor() {
            return Ok(None);
        }
        anyhow::ensure!(
            plan.start == self.plan.cursor && plan.end() == self.plan.recovery_position,
            "DeepSeek-V4 cooperative anchor plan no longer matches the recovery checkpoint"
        );
        Ok(Some(&self.prompt_tokens[..self.plan.recovery_position]))
    }

    pub(crate) fn publish_cooperative_prefill(&mut self, plan: Deepseek4CooperativePrefillPlan) {
        debug_assert_eq!(plan.start, self.plan.cursor);
        debug_assert!(plan.end() <= self.plan.recovery_position);
        self.plan.cursor = plan.end();
        self.progress.advance_prefill(plan.token_count);
        if plan.captures_anchor() {
            self.progress
                .recovery_anchor_captured(self.plan.recovery_position);
        }
    }

    pub(crate) fn advance(
        mut self,
        loaded: &mut Deepseek4LoadedModel,
        registration: Option<&ModelRegistration>,
        cancelled: impl Fn() -> bool,
        max_matrix_prefill_windows: Option<usize>,
        supervisor: &EngineSupervisor,
    ) -> Result<Deepseek4PrefillAdvance> {
        let cold_wave = self.plan.is_cold_wave();
        let advance = match loaded.advance_resumable_prefill(
            &self.prompt_tokens,
            &mut self.plan,
            &cancelled,
            &mut self.progress,
            max_matrix_prefill_windows,
            supervisor,
        ) {
            Ok(advance) => advance,
            Err(error) => {
                if cancelled() {
                    self.progress.cancelled();
                }
                return Err(error);
            }
        };
        match advance {
            Deepseek4ResumablePrefillAdvance::Pending { advanced_tokens } => {
                Ok(Deepseek4PrefillAdvance::Pending {
                    state: self,
                    advanced_tokens,
                })
            }
            Deepseek4ResumablePrefillAdvance::Ready {
                logits,
                cached_tokens,
                advanced_tokens,
            } => {
                let prefill_duration = self.prefill_started.elapsed();
                self.progress.finish_prefill(prefill_duration);
                release_completed_prefill_scratch();
                self.progress.start_decode();
                let state = Deepseek4SlotState::from_prefill(
                    self.prompt_tokens,
                    self.params,
                    logits,
                    cached_tokens,
                    self.stream,
                    self.request_started,
                    prefill_duration,
                    self.progress,
                    self.scratch_guard,
                    loaded,
                    registration,
                )?;
                Ok(Deepseek4PrefillAdvance::Ready {
                    state,
                    advanced_tokens,
                    cold_wave,
                })
            }
        }
    }
}

pub(crate) struct Deepseek4SlotState {
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
    logits: MlxBuffer,
    sampler: sampler_pure::SamplingParams,
    runtime: Option<GrammarRuntime>,
    thinking_budget: Option<Deepseek4ThinkingBudgetState>,
    max_tokens: usize,
    generated: Vec<u32>,
    logprobs: Option<Vec<f32>>,
    decode_ids: Vec<u32>,
    decode_prefix: String,
    decode_prefix_index: usize,
    decoded_running: String,
    finish_reason: &'static str,
    finished: bool,
    request_started: Instant,
    prefill_duration: Duration,
    decode_started: Instant,
    cached_tokens: usize,
    stream_router: Option<StreamRouter>,
    progress: RequestProgress,
    scratch_guard: Option<RequestScratchGuard>,
}

#[derive(Debug, Clone)]
struct Deepseek4ThinkingBudgetState {
    limit: usize,
    reasoning_tokens: usize,
    forced_tokens: Arc<Vec<u32>>,
    close_tokens: Arc<Vec<u32>>,
    forced_cursor: Option<usize>,
    closed: bool,
}

impl Deepseek4ThinkingBudgetState {
    fn from_params(params: &SamplingParams) -> Option<Self> {
        let limit = params.thinking_token_budget?;
        let forced_tokens = params.reasoning_end_tokens.clone()?;
        let close_tokens = params.reasoning_close_tokens.clone()?;
        (params.reasoning_forced_open
            && limit > 0
            && !forced_tokens.is_empty()
            && !close_tokens.is_empty())
        .then_some(Self {
            limit,
            reasoning_tokens: 0,
            forced_tokens,
            close_tokens,
            forced_cursor: None,
            closed: false,
        })
    }

    fn next_forced_token(&mut self) -> Option<(u32, bool)> {
        if self.closed {
            return None;
        }
        let started = self.forced_cursor.is_none() && self.reasoning_tokens >= self.limit;
        if started {
            self.forced_cursor = Some(0);
        }
        let cursor = self.forced_cursor?;
        let token = self.forced_tokens.get(cursor).copied()?;
        self.forced_cursor = Some(cursor + 1);
        Some((token, started))
    }

    fn observe_generated(&mut self, generated_tokens: &[u32]) {
        if self.closed {
            return;
        }
        if generated_tokens.ends_with(self.close_tokens.as_slice()) {
            self.closed = true;
            return;
        }
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(1);
    }
}

impl Deepseek4SlotState {
    pub(crate) fn prefill_seed(
        loaded: &mut Deepseek4LoadedModel,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        registration: Option<&ModelRegistration>,
        stream: bool,
        cancelled: impl Fn() -> bool,
        supervisor: &EngineSupervisor,
    ) -> Result<Self> {
        let scratch_guard = RequestScratchGuard::new();
        let request_started = Instant::now();
        let mut progress = RequestProgress::start(
            if stream { "slot-stream" } else { "slot-unary" },
            prompt_tokens.len(),
            params.max_tokens,
        );
        let prefill_started = Instant::now();
        let (logits, cached_tokens) = loaded.prefill_suffix(
            &prompt_tokens,
            params.max_tokens,
            cancelled,
            &mut progress,
            supervisor,
        )?;
        let prefill_duration = prefill_started.elapsed();
        progress.finish_prefill(prefill_duration);
        release_completed_prefill_scratch();
        progress.start_decode();
        Self::from_prefill(
            prompt_tokens,
            params,
            logits,
            cached_tokens,
            stream,
            request_started,
            prefill_duration,
            progress,
            scratch_guard,
            loaded,
            registration,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_prefill(
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        logits: MlxBuffer,
        cached_tokens: usize,
        stream: bool,
        request_started: Instant,
        prefill_duration: Duration,
        progress: RequestProgress,
        scratch_guard: RequestScratchGuard,
        loaded: &Deepseek4LoadedModel,
        registration: Option<&ModelRegistration>,
    ) -> Result<Self> {
        let max_tokens = decode_token_limit(
            params.max_tokens,
            prompt_tokens.len(),
            loaded.context_limit(),
        );
        let sampler = sampler_config(&params);
        let runtime = grammar_runtime(&params, registration)?;
        let thinking_budget = Deepseek4ThinkingBudgetState::from_params(&params);
        let logprobs = params.logprobs.then(|| Vec::with_capacity(max_tokens));
        let stream_router = stream.then(|| {
            StreamRouter::new(registration, params.reasoning_forced_open, request_started)
        });
        Ok(Self {
            prompt_tokens,
            params,
            logits,
            sampler,
            runtime,
            thinking_budget,
            max_tokens,
            generated: Vec::with_capacity(max_tokens),
            logprobs,
            decode_ids: Vec::new(),
            decode_prefix: String::new(),
            decode_prefix_index: 0,
            decoded_running: String::new(),
            finish_reason: "length",
            finished: false,
            request_started,
            prefill_duration,
            decode_started: Instant::now(),
            cached_tokens,
            stream_router,
            progress,
            scratch_guard: Some(scratch_guard),
        })
    }

    /// Sample and route one token without advancing the model cache.
    ///
    /// `Some(token)` means the caller must commit that token to the lane's
    /// cache and then call [`Self::install_next_logits`]. `None` means this
    /// request reached a terminal condition and no cache transaction is
    /// required. Keeping sampling outside cache advancement lets the strict
    /// four-lane scheduler path submit one shared GPU transaction without
    /// changing the established serial token semantics.
    pub(crate) fn prepare_tick(
        &mut self,
        loaded: &mut Deepseek4LoadedModel,
        registration: Option<&ModelRegistration>,
        events: Option<&mpsc::Sender<GenerationEvent>>,
        supervisor: &EngineSupervisor,
    ) -> Result<Option<u32>> {
        if self.finished {
            return Ok(None);
        }
        let forced_token = self
            .thinking_budget
            .as_mut()
            .and_then(Deepseek4ThinkingBudgetState::next_forced_token);
        if forced_token.is_some_and(|(_, started)| started) {
            tracing::warn!(
                request_id = self.progress.id(),
                budget = self.params.thinking_token_budget,
                generated_tokens = self.generated.len(),
                "DeepSeek-V4 required-tool thinking budget reached; forcing reasoning close"
            );
        }
        let (token, logprob) = if let Some((token, _)) = forced_token {
            accept_forced_token(&mut self.runtime, &self.params, token)?;
            (token, self.params.logprobs.then_some(0.0))
        } else {
            sample(
                loaded,
                &self.logits,
                &self.params,
                &self.sampler,
                &self.generated,
                &mut self.runtime,
                supervisor,
            )?
        };
        if loaded.eos_token_ids.contains(&token)
            || self.runtime.as_ref().is_some_and(GrammarRuntime::is_dead)
        {
            self.finish_reason = "stop";
            self.finished = true;
            return Ok(None);
        }

        self.generated.push(token);
        if let Some(budget) = self.thinking_budget.as_mut() {
            budget.observe_generated(&self.generated);
        }
        if let (Some(values), Some(value)) = (self.logprobs.as_mut(), logprob) {
            values.push(value);
        }
        if let Some(fragment) = tokenizers::tokenizer::step_decode_stream(
            &loaded.tokenizer,
            vec![token],
            false,
            &mut self.decode_ids,
            &mut self.decode_prefix,
            &mut self.decode_prefix_index,
        )
        .map_err(|error| anyhow::anyhow!("decode DeepSeek-V4 token {token}: {error}"))?
        {
            self.decoded_running.push_str(&fragment);
            if let Some(router) = self.stream_router.as_mut() {
                let events = events.context("DeepSeek-V4 slot stream has no event channel")?;
                router.push(
                    &fragment,
                    &self.decoded_running,
                    &mut self.runtime,
                    registration,
                    events,
                )?;
                if let Some(ttft) = router.first_visible_at() {
                    self.progress.first_semantic_token(ttft);
                }
            }
        }

        if self
            .params
            .stop_strings
            .iter()
            .any(|stop| !stop.is_empty() && self.decoded_running.contains(stop))
        {
            self.finish_reason = "stop";
            self.finished = true;
        } else if self.generated.len() >= self.max_tokens {
            self.finished = true;
        } else if accepted_single_tool_call_is_terminal(&self.params, self.runtime.as_ref()) {
            self.finish_reason = "stop";
            self.finished = true;
            tracing::info!(
                target: "hf2q::serve::api::engine_deepseek4::progress",
                request_id = self.progress.id(),
                "DeepSeek-V4 single-tool terminal commit skipped"
            );
        }

        if self.finished {
            self.progress.advance_decode(self.generated.len());
            Ok(None)
        } else {
            Ok(Some(token))
        }
    }

    pub(crate) fn install_next_logits(&mut self, logits: MlxBuffer) {
        debug_assert!(!self.finished);
        self.logits = logits;
        self.progress.advance_decode(self.generated.len());
    }

    pub(crate) fn tick(
        &mut self,
        loaded: &mut Deepseek4LoadedModel,
        registration: Option<&ModelRegistration>,
        events: Option<&mpsc::Sender<GenerationEvent>>,
        supervisor: &EngineSupervisor,
    ) -> Result<()> {
        if let Some(token) = self.prepare_tick(loaded, registration, events, supervisor)? {
            let logits = loaded.commit_generated_token(token, supervisor)?;
            self.install_next_logits(logits);
        }
        Ok(())
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.finished
    }

    pub(crate) fn request_id(&self) -> u64 {
        self.progress.id()
    }

    pub(crate) fn finish(
        mut self,
        registration: Option<&ModelRegistration>,
        events: Option<&mpsc::Sender<GenerationEvent>>,
    ) -> Result<Deepseek4SlotCompletion> {
        if let Some(router) = self.stream_router.as_mut() {
            let events = events.context("DeepSeek-V4 slot stream has no event channel")?;
            if router.finish(&mut self.runtime, registration, events)? {
                self.finish_reason = "tool_calls";
            }
        }
        let semantic_ttft = self
            .stream_router
            .as_ref()
            .and_then(StreamRouter::first_visible_at)
            .unwrap_or_else(|| self.request_started.elapsed());
        let (text, reasoning_text) = split_reasoning(
            &self.decoded_running,
            registration,
            self.params.reasoning_forced_open,
        );
        let decode_duration = self.decode_started.elapsed();
        self.progress.complete(
            self.finish_reason,
            self.generated.len(),
            self.stream_router.as_ref().map(|_| semantic_ttft),
        );
        if let Some(guard) = self.scratch_guard.take() {
            guard.complete();
        }
        Ok(Deepseek4SlotCompletion {
            result: GenerationResult {
                text,
                reasoning_text,
                prompt_tokens: self.prompt_tokens.len(),
                completion_tokens: self.generated.len(),
                reasoning_tokens: None,
                finish_reason: self.finish_reason,
                prefill_duration: self.prefill_duration,
                decode_duration,
                cached_tokens: self.cached_tokens,
                logprobs: self.logprobs,
            },
            semantic_ttft,
        })
    }

    pub(crate) fn cancelled(&self) {
        self.progress.cancelled();
    }

    pub(crate) fn failed(&self, error: &anyhow::Error) {
        self.progress.failed(error);
    }
}

#[cfg(test)]
mod tests {
    use super::super::Deepseek4PrefillOrigin;
    use super::*;

    fn cached_prefill_state(cursor: usize, recovery_position: usize) -> Deepseek4PrefillState {
        Deepseek4PrefillState {
            prompt_tokens: (0..recovery_position + 64)
                .map(|token| (token % 120_000) as u32)
                .collect(),
            params: SamplingParams::default(),
            stream: false,
            request_started: Instant::now(),
            prefill_started: Instant::now(),
            progress: RequestProgress::start("cooperative-plan-test", recovery_position + 64, 1),
            scratch_guard: RequestScratchGuard::new(),
            plan: Deepseek4ResumablePrefill {
                cursor,
                recovery_position,
                window_multiplier: 16,
                cached_tokens: cursor,
                origin: Deepseek4PrefillOrigin::Cached,
            },
        }
    }

    fn thinking_budget_params(limit: usize) -> SamplingParams {
        SamplingParams {
            reasoning_forced_open: true,
            thinking_token_budget: Some(limit),
            reasoning_end_tokens: Some(Arc::new(vec![90, 91])),
            reasoning_close_tokens: Some(Arc::new(vec![90, 91])),
            ..SamplingParams::default()
        }
    }

    #[test]
    fn required_tool_thinking_budget_forces_close_then_releases_decode() {
        let mut budget =
            Deepseek4ThinkingBudgetState::from_params(&thinking_budget_params(2)).unwrap();
        assert_eq!(budget.next_forced_token(), None);
        budget.observe_generated(&[11]);
        assert_eq!(budget.next_forced_token(), None);
        budget.observe_generated(&[11, 12]);
        assert_eq!(budget.next_forced_token(), Some((90, true)));
        budget.observe_generated(&[11, 12, 90]);
        assert_eq!(budget.next_forced_token(), Some((91, false)));
        budget.observe_generated(&[11, 12, 90, 91]);
        assert_eq!(budget.next_forced_token(), None);
    }

    #[test]
    fn natural_reasoning_close_disarms_required_tool_budget() {
        let mut budget =
            Deepseek4ThinkingBudgetState::from_params(&thinking_budget_params(8)).unwrap();
        budget.observe_generated(&[1, 90, 91]);
        assert_eq!(budget.next_forced_token(), None);
    }

    #[test]
    fn cooperative_prefill_plan_matches_two_three_and_four_lane_serving_widths() {
        let state = cached_prefill_state(6_676, 8_000);
        for (lanes, expected_rows) in [(2, 1_024), (3, 640), (4, 512)] {
            let lane_budget =
                crate::inference::models::deepseek4::MAX_COOPERATIVE_PREFILL_ROWS / lanes / 128
                    * 128;
            let plan = state
                .plan_cooperative_prefill(6_676, 6_676, 128, lane_budget)
                .unwrap()
                .expect("warm matrix suffix should be cooperative");
            assert_eq!(plan.start(), 6_676);
            assert_eq!(plan.token_count(), expected_rows);
            assert!(plan.end() < 8_000);
            assert_eq!(state.cooperative_tokens(plan).unwrap().len(), expected_rows);
        }
    }

    #[test]
    fn warm_matrix_alignment_cap_lands_exactly_or_falls_back() {
        let state = cached_prefill_state(6_676, 9_466);
        assert_eq!(
            state.warm_matrix_alignment_state(),
            Some((6_676, 9_466, 16))
        );
        assert_eq!(
            state.exact_warm_matrix_alignment_window_cap(7_316, 128),
            Some(5)
        );

        let shrink_band = cached_prefill_state(852, 1_000);
        assert_eq!(
            shrink_band.exact_warm_matrix_alignment_window_cap(980, 128),
            None,
            "the prefill planner shortens this nominal 128-token catch-up to preserve a legal tail"
        );
    }

    #[test]
    fn uncached_tokens_reports_the_exact_recovery_suffix() {
        let mut state = cached_prefill_state(6_676, 6_676);
        state.prompt_tokens.truncate(6_684);
        assert_eq!(state.uncached_tokens(), 8);
        assert!(state.is_recovery_tail());

        let mut cold = cached_prefill_state(0, 6_676);
        cold.plan.origin = super::super::Deepseek4PrefillOrigin::Cold;
        cold.prompt_tokens.truncate(6_684);
        assert!(!cold.is_recovery_tail());
        assert_eq!(cold.warm_matrix_alignment_state(), None);
        cold.plan.cursor = cold.plan.recovery_position;
        assert!(cold.is_recovery_tail());
        assert_eq!(cold.warm_matrix_alignment_state(), None);
        cold.plan.cursor = cold.prompt_tokens.len();
        assert!(!cold.is_recovery_tail());
    }

    #[test]
    fn cooperative_prefill_accepts_aligned_cold_matrix_and_anchor_but_excludes_tail() {
        let mut cold = cached_prefill_state(0, 6_676);
        cold.plan.origin = super::super::Deepseek4PrefillOrigin::Cold;
        cold.prompt_tokens.truncate(6_684);
        let mut transactions = 0;
        let mut cooperative_rows = 0;
        let mut anchors = 0;
        while let Some(plan) = cold
            .plan_cooperative_prefill(cold.plan.cursor, cold.plan.cursor, 128, 512)
            .unwrap()
        {
            assert!((1..=512).contains(&plan.token_count()));
            assert!(plan.end() <= cold.plan.recovery_position);
            anchors += plan.captures_anchor() as usize;
            if plan.captures_anchor() {
                assert_eq!(
                    cold.cooperative_anchor_tokens(plan).unwrap().unwrap().len(),
                    cold.plan.recovery_position
                );
            }
            cold.validate_cooperative_prefill_commit(plan).unwrap();
            cold.publish_cooperative_prefill(plan);
            transactions += 1;
            cooperative_rows += plan.token_count();
        }
        assert_eq!(transactions, 14);
        assert_eq!(anchors, 1);
        assert_eq!(cooperative_rows, cold.plan.cursor);
        assert_eq!(cold.plan.cursor, cold.plan.recovery_position);
        assert_eq!(cold.prompt_tokens.len() - cold.plan.recovery_position, 8);

        let anchor = cached_prefill_state(7_488, 8_000);
        let anchor_plan = anchor
            .plan_cooperative_prefill(7_488, 7_488, 128, 512)
            .unwrap()
            .expect("aligned final matrix slice should capture the recovery anchor");
        assert!(anchor_plan.captures_anchor());
        assert_eq!(anchor_plan.end(), 8_000);

        let incremental = cached_prefill_state(7_968, 8_000);
        assert!(incremental
            .plan_cooperative_prefill(7_968, 7_968, 128, 512)
            .unwrap()
            .is_none());
        assert!(incremental
            .plan_cooperative_prefill(7_967, 7_968, 128, 512)
            .is_err());
    }

    #[test]
    fn cooperative_prefill_state_publication_is_prevalidated_and_infallible() {
        let mut state = cached_prefill_state(6_676, 8_000);
        let plan = state
            .plan_cooperative_prefill(6_676, 6_676, 128, 512)
            .unwrap()
            .unwrap();
        state.validate_cooperative_prefill_commit(plan).unwrap();
        state.publish_cooperative_prefill(plan);
        assert_eq!(state.plan.cursor, 7_188);
        assert!(state.validate_cooperative_prefill_commit(plan).is_err());
    }
}
