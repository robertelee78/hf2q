//! Interleavable DeepSeek-V4 request state for full-context agent slots.
//!
//! The verifier model is shared, while the caller swaps the selected slot's
//! cache into `Deepseek4LoadedModel` for one tick. Everything that used to
//! live on the stack of `generate_stream`/`generate_once` is retained here so
//! another agent can run between decode tokens without changing tokenization,
//! sampling, reasoning splitting, or DSML tool routing.

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
    decode_token_limit, grammar_runtime, sample, sampler_config, split_reasoning,
};
use super::stream::StreamRouter;
use super::{
    release_completed_prefill_scratch, Deepseek4LoadedModel, Deepseek4ResumablePrefill,
    Deepseek4ResumablePrefillAdvance, RequestScratchGuard,
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

    pub(crate) fn advance(
        mut self,
        loaded: &mut Deepseek4LoadedModel,
        registration: Option<&ModelRegistration>,
        cancelled: impl Fn() -> bool,
        supervisor: &EngineSupervisor,
    ) -> Result<Deepseek4PrefillAdvance> {
        let cold_wave = self.plan.is_cold_wave();
        let advance = match loaded.advance_resumable_prefill(
            &self.prompt_tokens,
            &mut self.plan,
            &cancelled,
            &mut self.progress,
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

    pub(crate) fn tick(
        &mut self,
        loaded: &mut Deepseek4LoadedModel,
        registration: Option<&ModelRegistration>,
        events: Option<&mpsc::Sender<GenerationEvent>>,
        supervisor: &EngineSupervisor,
    ) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        let (token, logprob) = sample(
            loaded,
            &self.logits,
            &self.params,
            &self.sampler,
            &self.generated,
            &mut self.runtime,
            supervisor,
        )?;
        if loaded.eos_token_ids.contains(&token)
            || self.runtime.as_ref().is_some_and(GrammarRuntime::is_dead)
        {
            self.finish_reason = "stop";
            self.finished = true;
            return Ok(());
        }

        self.generated.push(token);
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
        }

        if !self.finished {
            self.logits = loaded.commit_generated_token(token, supervisor)?;
        }
        self.progress.advance_decode(self.generated.len());
        Ok(())
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.finished
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
