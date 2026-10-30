//! ADR-063 C5 — opt-in warm-request frontend timing.
//!
//! `HF2Q_FRONTEND_TIMING=1` makes every dispatched chat completion emit one
//! `frontend timing` line (target `hf2q::serve::frontend_timing`, INFO) with
//! microsecond spans for the handler-owned work: pool admission, vision
//! preflight, constraint compilation, primary render and tokenization, the
//! stable-prefix render/tokenize/match, the reasoning probe render and the
//! thinking-budget marker encoding. The same line carries the time from
//! request receipt to the first semantic stream event (or to unary
//! completion), so frontend work can be compared against warm TTFT without
//! timestamped logs.
//!
//! The variable is read once per process. When it is unset every span is a
//! branch on `None`: no clock reads, no allocation, no extra task and no
//! extra stream hop.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use super::sse::{DeltaKind, GenerationEvent};

pub(super) const FRONTEND_TIMING_ENV: &str = "HF2Q_FRONTEND_TIMING";

/// Handler-owned phases. Each accumulates, so a phase entered twice
/// (constraint compilation) reports its total.
#[derive(Clone, Copy, Debug)]
pub(super) enum Span {
    /// Pool resolution and model lease (`resolve_engine_for_request`).
    Admission,
    /// Image fetch/preprocess/ViT preflight; zero for text-only requests.
    Vision,
    /// Request output constraint and tool grammar compilation.
    Constraint,
    /// Primary chat-template render (with generation prompt).
    Render,
    /// Primary prompt tokenization.
    Tokenize,
    /// Truncate/summarize rewrite; zero unless the prompt overflowed.
    OverflowRewrite,
    /// Stable-prefix render (no generation prompt; Gemma 4 and Qwen 3.5+).
    StableRender,
    /// Stable-prefix tokenization.
    StableTokenize,
    /// Stable-prefix token match and Gemma full-input checkpoint tag.
    PrefixMatch,
    /// Minimal-transcript render probing for an open reasoning block.
    ProbeRender,
    /// Thinking-budget resolution, including marker tokenization.
    Budget,
    /// Stream budget admission and worker enqueue; zero for unary.
    Enqueue,
}

const SPAN_COUNT: usize = Span::Enqueue as usize + 1;

static NEXT_SEQ: AtomicU64 = AtomicU64::new(0);

static ENABLED: LazyLock<bool> =
    LazyLock::new(|| env_enables(std::env::var(FRONTEND_TIMING_ENV).ok().as_deref()));

fn env_enables(value: Option<&str>) -> bool {
    value == Some("1")
}

fn micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

struct Record {
    seq: u64,
    started: Instant,
    stream: bool,
    messages: usize,
    tools: usize,
    family: String,
    rendered_bytes: usize,
    prompt_tokens: usize,
    stable_prefix_tokens: Option<usize>,
    span_us: [u64; SPAN_COUNT],
    prepare_us: u64,
}

impl Record {
    fn emit(
        &self,
        outcome: &'static str,
        first_output_us: Option<u64>,
        completion_us: Option<u64>,
    ) {
        let s = &self.span_us;
        tracing::info!(
            target: "hf2q::serve::frontend_timing",
            seq = self.seq,
            family = %self.family,
            stream = self.stream,
            outcome,
            messages = self.messages,
            tools = self.tools,
            rendered_bytes = self.rendered_bytes,
            prompt_tokens = self.prompt_tokens,
            stable_prefix_tokens = self.stable_prefix_tokens,
            admission_us = s[Span::Admission as usize],
            vision_us = s[Span::Vision as usize],
            constraint_us = s[Span::Constraint as usize],
            render_us = s[Span::Render as usize],
            tokenize_us = s[Span::Tokenize as usize],
            overflow_rewrite_us = s[Span::OverflowRewrite as usize],
            stable_render_us = s[Span::StableRender as usize],
            stable_tokenize_us = s[Span::StableTokenize as usize],
            prefix_match_us = s[Span::PrefixMatch as usize],
            probe_render_us = s[Span::ProbeRender as usize],
            budget_us = s[Span::Budget as usize],
            enqueue_us = s[Span::Enqueue as usize],
            prepare_us = self.prepare_us,
            first_output_us,
            completion_us,
            "frontend timing"
        );
    }
}

/// Per-request timing record; `None` inside when the env gate is off.
pub(super) struct FrontendTiming(Option<Box<Record>>);

impl FrontendTiming {
    /// Gate on `HF2Q_FRONTEND_TIMING=1` (read once per process).
    pub(super) fn from_env(stream: bool, messages: usize, tools: usize) -> Self {
        Self::new(*ENABLED, stream, messages, tools)
    }

    pub(super) fn new(enabled: bool, stream: bool, messages: usize, tools: usize) -> Self {
        if !enabled {
            return Self(None);
        }
        Self(Some(Box::new(Record {
            seq: NEXT_SEQ.fetch_add(1, Ordering::Relaxed),
            started: Instant::now(),
            stream,
            messages,
            tools,
            family: String::new(),
            rendered_bytes: 0,
            prompt_tokens: 0,
            stable_prefix_tokens: None,
            span_us: [0; SPAN_COUNT],
            prepare_us: 0,
        })))
    }

    #[cfg(test)]
    pub(super) fn disabled() -> Self {
        Self(None)
    }

    /// Clock read only when enabled; pass the result to [`Self::record`].
    #[inline]
    pub(super) fn start(&self) -> Option<Instant> {
        self.0.as_ref().map(|_| Instant::now())
    }

    #[inline]
    pub(super) fn record(&mut self, span: Span, started: Option<Instant>) {
        if let (Some(record), Some(started)) = (self.0.as_mut(), started) {
            record.span_us[span as usize] =
                record.span_us[span as usize].saturating_add(micros(started.elapsed()));
        }
    }

    pub(super) fn set_family(&mut self, family: impl FnOnce() -> String) {
        if let Some(record) = self.0.as_mut() {
            record.family = family();
        }
    }

    pub(super) fn set_rendered_bytes(&mut self, bytes: usize) {
        if let Some(record) = self.0.as_mut() {
            record.rendered_bytes = bytes;
        }
    }

    /// Final prompt shape handed to the worker; closes the prepare window.
    pub(super) fn mark_prepared(
        &mut self,
        prompt_tokens: usize,
        stable_prefix_tokens: Option<usize>,
    ) {
        if let Some(record) = self.0.as_mut() {
            record.prompt_tokens = prompt_tokens;
            record.stable_prefix_tokens = stable_prefix_tokens;
            record.prepare_us = micros(record.started.elapsed());
        }
    }

    /// Unary path: one line once the worker result (or error) is back.
    pub(super) fn finish_unary(self, succeeded: bool) {
        if let Some(record) = self.0 {
            let completion_us = micros(record.started.elapsed());
            record.emit(
                if succeeded { "ok" } else { "error" },
                None,
                Some(completion_us),
            );
        }
    }

    /// Streaming path: when enabled, interpose a relay that emits the line
    /// at the first semantic event and forwards every event unchanged and
    /// in order. Disabled returns `upstream` itself (no relay).
    ///
    /// Dropping the returned receiver (client disconnect) ends the relay,
    /// which drops `upstream`, so the worker still observes cancellation
    /// on its next send.
    pub(super) fn relay_stream(
        self,
        mut upstream: mpsc::Receiver<GenerationEvent>,
    ) -> mpsc::Receiver<GenerationEvent> {
        let Some(record) = self.0 else {
            return upstream;
        };
        let (tx, rx) = mpsc::channel(64);
        tokio::spawn(async move {
            let mut pending = Some(record);
            while let Some(event) = upstream.recv().await {
                if let Some(outcome) = first_output_outcome(&event) {
                    if let Some(record) = pending.take() {
                        let elapsed = micros(record.started.elapsed());
                        record.emit(outcome, Some(elapsed), None);
                    }
                }
                if tx.send(event).await.is_err() {
                    break;
                }
            }
            if let Some(record) = pending.take() {
                record.emit("closed_before_output", None, None);
            }
        });
        rx
    }
}

/// Classify the event that ends the wait for semantic output: any
/// reasoning/content/tool-call delta, or a terminal event. Logprobs alone
/// do not count.
fn first_output_outcome(event: &GenerationEvent) -> Option<&'static str> {
    match event {
        GenerationEvent::Delta {
            kind: DeltaKind::Reasoning,
            ..
        } => Some("first_reasoning"),
        GenerationEvent::Delta {
            kind: DeltaKind::Content,
            ..
        } => Some("first_content"),
        GenerationEvent::ToolCallDelta { .. } => Some("first_tool_call"),
        GenerationEvent::Done { .. } => Some("done_without_output"),
        GenerationEvent::Error(_) => Some("error"),
        GenerationEvent::Logprobs(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::api::schema::ChoiceLogprobs;
    use crate::serve::api::sse::StreamStats;

    fn delta(kind: DeltaKind, text: &str) -> GenerationEvent {
        GenerationEvent::Delta {
            kind,
            text: text.to_string(),
        }
    }

    fn done() -> GenerationEvent {
        GenerationEvent::Done {
            finish_reason: "stop",
            prompt_tokens: 3,
            completion_tokens: 2,
            stats: StreamStats::default(),
        }
    }

    fn describe(event: &GenerationEvent) -> String {
        match event {
            GenerationEvent::Delta { kind, text } => format!("{kind:?}:{text}"),
            GenerationEvent::ToolCallDelta {
                index, arguments, ..
            } => format!("tool{index}:{}", arguments.as_deref().unwrap_or("")),
            GenerationEvent::Logprobs(_) => "logprobs".into(),
            GenerationEvent::Done {
                finish_reason,
                completion_tokens,
                ..
            } => format!("done:{finish_reason}:{completion_tokens}"),
            GenerationEvent::Error(error) => format!("error:{error}"),
        }
    }

    #[test]
    fn only_the_exact_value_one_enables_timing() {
        assert!(env_enables(Some("1")));
        for value in [None, Some(""), Some("0"), Some("true"), Some(" 1")] {
            assert!(!env_enables(value), "{value:?} must leave timing off");
        }
    }

    #[test]
    fn disabled_timing_takes_no_clock_reads_and_records_nothing() {
        let mut timing = FrontendTiming::disabled();
        assert!(timing.start().is_none());
        timing.record(Span::Render, Some(Instant::now()));
        timing.mark_prepared(10, Some(4));
        assert!(timing.0.is_none());
    }

    #[test]
    fn enabled_spans_accumulate_per_phase() {
        let mut timing = FrontendTiming::new(true, true, 4, 2);
        let earlier = Instant::now() - Duration::from_millis(3);
        timing.record(Span::Constraint, Some(earlier));
        timing.record(Span::Constraint, Some(earlier));
        timing.record(Span::Render, None);
        let record = timing.0.as_ref().expect("enabled");
        assert!(record.span_us[Span::Constraint as usize] >= 6_000);
        assert_eq!(record.span_us[Span::Render as usize], 0);
        assert_eq!(record.span_us[Span::Tokenize as usize], 0);
    }

    #[test]
    fn first_output_ignores_logprobs_and_accepts_every_semantic_or_terminal_event() {
        assert_eq!(
            first_output_outcome(&delta(DeltaKind::Reasoning, "r")),
            Some("first_reasoning")
        );
        assert_eq!(
            first_output_outcome(&delta(DeltaKind::Content, "c")),
            Some("first_content")
        );
        assert_eq!(first_output_outcome(&done()), Some("done_without_output"));
        assert_eq!(
            first_output_outcome(&GenerationEvent::Error("x".into())),
            Some("error")
        );
        assert_eq!(
            first_output_outcome(&GenerationEvent::Logprobs(ChoiceLogprobs {
                content: Vec::new()
            })),
            None
        );
    }

    #[tokio::test]
    async fn disabled_stream_is_returned_without_a_relay() {
        let (tx, rx) = mpsc::channel(4);
        let mut out = FrontendTiming::disabled().relay_stream(rx);
        tx.send(delta(DeltaKind::Content, "a")).await.unwrap();
        drop(tx);
        assert_eq!(describe(&out.recv().await.unwrap()), "Content:a");
        assert!(out.recv().await.is_none());
    }

    #[tokio::test]
    async fn enabled_relay_forwards_every_event_in_order_then_closes() {
        let (tx, rx) = mpsc::channel(4);
        let mut out = FrontendTiming::new(true, true, 2, 0).relay_stream(rx);
        let producer = tokio::spawn(async move {
            for i in 0..100 {
                tx.send(delta(DeltaKind::Reasoning, &i.to_string()))
                    .await
                    .unwrap();
            }
            tx.send(GenerationEvent::ToolCallDelta {
                index: 0,
                id: Some("call_0".into()),
                call_type: Some("function".into()),
                name: Some("read".into()),
                arguments: Some("{}".into()),
            })
            .await
            .unwrap();
            tx.send(done()).await.unwrap();
        });
        let mut seen = Vec::new();
        while let Some(event) = out.recv().await {
            seen.push(describe(&event));
        }
        producer.await.unwrap();
        let mut expected: Vec<String> = (0..100).map(|i| format!("Reasoning:{i}")).collect();
        expected.push("tool0:{}".into());
        expected.push("done:stop:2".into());
        assert_eq!(seen, expected);
    }

    #[tokio::test]
    async fn dropping_the_relayed_stream_closes_the_worker_sender() {
        let (tx, rx) = mpsc::channel(1);
        let out = FrontendTiming::new(true, true, 2, 0).relay_stream(rx);
        tx.send(delta(DeltaKind::Content, "a")).await.unwrap();
        drop(out);
        let closed = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if tx.send(delta(DeltaKind::Content, "b")).await.is_err() {
                    break;
                }
            }
        })
        .await;
        assert!(
            closed.is_ok(),
            "client disconnect must still reach the worker"
        );
    }
}
