//! Server-Sent Events (SSE) stream encoding for OpenAI-compatible streaming
//! chat completions.
//!
//! Restored from `fe54bc2~1:src/serve/sse.rs` (the commit preceding the MLX
//! divorce) after engine-agnostic review. Two intentional deletions from the
//! restore:
//!   1. The `super::tool_parser::{...}` import path — tool-call parsing is
//!      obsoleted by grammar-constrained decoding (ADR-005 Decision #6);
//!      model output is well-formed JSON by construction, so the
//!      `generation_events_to_sse_with_tools` function in the original file
//!      is not restored. It is superseded by a per-model **boundary-marker**
//!      splitter landing alongside the per-model tool-call registration
//!      (Decision #21; task tracked separately) — whose job is classification
//!      (reasoning vs content vs tool-call), not parsing.
//!   2. `crate::inference::engine::GenerationStats` — candle-era struct that
//!      no longer exists under ADR-008 (mlx-native). Replaced by
//!      `StreamStats` (a neutral, engine-agnostic timing/usage handoff).
//!
//! Decision #20 — SSE keepalive comment every 15s. Implemented via axum's
//! `KeepAlive` helper. Keepalive text is `""` so the line `: \n\n` functions
//! as a comment frame, which proxies and OpenAI SDK clients tolerate.
//!
//! Decision #2 — serialized FIFO queue with silent wait + SSE keepalive. A
//! queue-wait keepalive is emitted by the HTTP handler BEFORE the generation
//! starts (while the request sits in the queue); once generation starts, the
//! axum `KeepAlive` layered on the returned Sse stream takes over.
//!
//! Decision #21 — reasoning_content split. Callers route tokens into the
//! appropriate `ChunkDelta` slot (`content` vs `reasoning_content`) via
//! `GenerationEvent::Delta{kind,text}`, classified upstream by the
//! boundary-marker state machine (lands with per-model registration). This
//! file treats the classification as pre-computed and just encodes.
//!
//! # ADR-040 §6 Phase C C3 — per-slot keepalive seam
//!
//! **cfa-iter-A5b MINOR #1 wording fix**: this section was previously
//! titled "per-slot keepalive accounting", which overclaimed —
//! `slot_id` is traced ONCE at stream construction, not per
//! keepalive frame. The 15-second keepalive itself IS wired through
//! axum's `KeepAlive` (see [`SSE_KEEPALIVE_INTERVAL_SECS`]); what
//! C3 added is a typed SEAM ([`SseStreamOptions::slot_id`] +
//! [`generation_events_to_sse_with_slot`]) so the Phase C2c+
//! SlotAware runtime can attribute streams to the responsible
//! physical slot via the construction-time trace event. Per-frame
//! attribution is a future iter (not in scope for C3 / iter-A5b).
//!
//! Under `SchedulerPolicy::FifoSerial` (today's default, ADR-040 §3.2 +
//! §3.6) `max_slots = 1` and there is at most one in-flight streaming
//! request per engine — keepalive is "per-slot" trivially because per-stream
//! ≡ per-slot ≡ per-connection at that bound. Each HTTP handler call at
//! `handlers.rs::chat_completions_stream` constructs its OWN
//! `mpsc::channel(64)` + its OWN `Sse<...>` wrapper at
//! [`generation_events_to_sse`]; the axum `KeepAlive` layered on that
//! `Sse` lives entirely inside the per-request future, so its 15s
//! "last-emission" timer is already scoped to that single stream's state.
//!
//! Under [`crate::serve::scheduler::SchedulerPolicy::InflightBatched`]
//! (the real scheduler-policy enum variant; gated behind
//! `EngineMode::SlotAware { max_slots = N }` at Phase C2c future per
//! ADR-040 §6 Phase C), the same per-request construction shape
//! preserves the per-slot association because each concurrent slot is
//! serviced by its own handler future + its own
//! `generation_events_to_sse` call + its own `KeepAlive` layer — the
//! N streams share zero keepalive state. This module therefore needs NO
//! cross-stream aggregation refactor for the per-slot promise; the
//! structural shape is already correct.
//!
//! What C3 DOES add: an optional [`SseStreamOptions::slot_id`] field so
//! the stream's per-slot association is *typed* at the boundary (rather
//! than implicit-per-task) for diagnostics + future per-slot
//! observability hooks. Under FifoSerial the field is `None` (legacy
//! shape, byte-identical to pre-C3); under InflightBatched the handler
//! will populate it from the `SlotHandle` returned by `Scheduler::admit`
//! so the construction-time trace event attributes the stream to the
//! responsible physical slot. The keepalive INTERVAL + TEXT are
//! unchanged (15s + `""`), satisfying ADR-040 §1.4's "continuous
//! batching changes WHEN the request executes, not the request/response
//! shape" contract for clients at N=1.

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::Stream;
use futures::StreamExt;
use tokio::sync::mpsc;

use super::schema::{
    ChatCompletionChunk, ChoiceLogprobs, ChunkChoice, ChunkDelta, CompletionTokensDetails,
    PromptTokensDetails, StreamingTimingInfo, UsageStats,
};
use super::state::ServerMetrics;

/// Which delta slot a token fragment belongs to (per Decision #21).
///
/// The classification is performed upstream by the per-model boundary-marker
/// state machine — this file just emits into the corresponding JSON slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaKind {
    /// Goes into `delta.content` — the final answer text.
    Content,
    /// Goes into `delta.reasoning_content` — the model's pre-answer reasoning.
    Reasoning,
}

/// Neutral, engine-agnostic timing + usage handoff carried on the final
/// `GenerationEvent::Done`. Replaces the candle-era `GenerationStats` import.
///
/// All fields are `Option` so producers that don't yet instrument a given
/// metric can populate `None`. The handler serializes into the
/// `x_hf2q_timing` field and, when `stream_options.include_usage = true`,
/// into the final chunk's `usage` field.
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    pub prefill_time_secs: Option<f64>,
    pub decode_time_secs: Option<f64>,
    pub total_time_secs: Option<f64>,
    pub time_to_first_token_ms: Option<f64>,
    pub prefill_tokens_per_sec: Option<f64>,
    pub decode_tokens_per_sec: Option<f64>,
    pub gpu_sync_count: Option<u64>,
    pub gpu_dispatch_count: Option<u64>,
    /// Prompt tokens served from prompt cache (Decision #24).
    /// Guarantees tune-up item 5a (2026-08-20): producers report a
    /// known-zero count as `Some(0)`; `None` means "not measured" and
    /// the usage frame reports it as 0 — the emitted
    /// `prompt_tokens_details.cached_tokens` field is never omitted
    /// when usage is emitted.
    pub cached_prompt_tokens: Option<usize>,
    /// Tokens spent on pre-answer reasoning (Decision #21).
    pub reasoning_tokens: Option<usize>,
}

/// Events sent from the blocking generation thread to the async SSE stream.
///
/// Token deltas carry a `DeltaKind` so the encoder can route them to
/// `content` vs `reasoning_content` without re-classifying. Tool-call deltas
/// are emitted directly by the upstream grammar-aware pipeline as
/// `ToolCallDeltaEvent` — they are well-formed by construction (Decision #6)
/// and carry their own index/id/name/arguments fragments.
#[derive(Debug)]
pub enum GenerationEvent {
    /// A generated token fragment in a specific delta slot.
    Delta { kind: DeltaKind, text: String },
    /// A tool-call delta emitted by the grammar-aware sampler.
    ToolCallDelta {
        index: usize,
        /// Present only on the first delta for this `index`.
        id: Option<String>,
        /// Always `"function"`; present only on the first delta for this `index`.
        call_type: Option<String>,
        /// Function name fragment (present only on the first delta for `index`).
        name: Option<String>,
        /// Arguments JSON fragment (appended across deltas).
        arguments: Option<String>,
    },
    /// Per-token logprobs for the current position, if `logprobs: true` was set.
    Logprobs(ChoiceLogprobs),
    /// Generation completed successfully. `finish_reason` is typically
    /// `"stop"` (saw a stop sequence) or `"length"` (hit `max_tokens`).
    /// `"tool_calls"` is emitted when the model finishes inside a tool call.
    Done {
        finish_reason: &'static str,
        prompt_tokens: usize,
        completion_tokens: usize,
        stats: StreamStats,
    },
    /// Generation failed mid-stream. The error is logged + an `"error"`
    /// finish_reason is emitted so the client sees a clean termination.
    Error(String),
}

/// Options controlling the SSE stream's final chunk composition.
///
/// These are populated from the `ChatCompletionRequest` by the handler:
///   - `include_usage` ← `stream_options.include_usage`
///   - `logprobs`      ← request's `logprobs`
///
/// **ADR-040 C3 note**: per-slot association is NOT carried on this
/// struct. It is passed as a separate parameter to
/// [`generation_events_to_sse_with_slot`] so the legacy struct-literal
/// constructions in `handlers.rs` remain compile-stable (no additive
/// public-field requirement at the call site). The fields here are
/// the OpenAI-surface options only; the slot id is a Phase C
/// scheduler concept that belongs to the function signature, not the
/// wire-format options.
#[derive(Debug, Clone, Default)]
pub struct SseStreamOptions {
    /// If true, the final SSE chunk includes a `usage` field.
    pub include_usage: bool,
    /// If true, each content/reasoning delta chunk carries a `logprobs` slot.
    pub logprobs: bool,
    /// Optional system fingerprint to tag every chunk with.
    pub system_fingerprint: Option<String>,
}

/// Build the inner SSE event stream WITHOUT the outer `Sse`/keepalive
/// wrapping. Tests use this directly so they can assert on the raw frames
/// without dragging in `http_body_util` / `hyper` body-collection plumbing.
/// The public entrypoint `generation_events_to_sse` wraps this with
/// `Sse::new(...).keep_alive(...)`.
pub fn generation_events_stream(
    rx: mpsc::Receiver<GenerationEvent>,
    request_id: String,
    model_name: String,
    created: i64,
    opts: SseStreamOptions,
) -> impl Stream<Item = Result<Event, Infallible>> {
    generation_events_stream_with_metrics(rx, request_id, model_name, created, opts, None)
}

fn generation_events_stream_with_metrics(
    mut rx: mpsc::Receiver<GenerationEvent>,
    request_id: String,
    model_name: String,
    created: i64,
    opts: SseStreamOptions,
    metrics: Option<Arc<ServerMetrics>>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let sfp = opts.system_fingerprint.clone();
        let include_usage = opts.include_usage;

        let role_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_name.clone(),
            system_fingerprint: sfp.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".into()),
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
            x_hf2q_timing: None,
        };
        yield Ok(Event::default().data(serde_json::to_string(&role_chunk).unwrap_or_default()));

        // Carried across token events so we can attach logprobs to the next
        // delta chunk (OpenAI pairs logprobs with the chunk whose content
        // delta they correspond to).
        let mut pending_logprobs: Option<ChoiceLogprobs> = None;

        while let Some(event) = rx.recv().await {
            match event {
                GenerationEvent::Delta { kind, text } => {
                    let (content, reasoning) = match kind {
                        DeltaKind::Content => (Some(text), None),
                        DeltaKind::Reasoning => (None, Some(text)),
                    };
                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_name.clone(),
                        system_fingerprint: sfp.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content,
                                reasoning_content: reasoning,
                                tool_calls: None,
                            },
                            finish_reason: None,
                            logprobs: pending_logprobs.take(),
                        }],
                        usage: None,
                        x_hf2q_timing: None,
                    };
                    yield Ok(Event::default()
                        .data(serde_json::to_string(&chunk).unwrap_or_default()));
                }
                GenerationEvent::ToolCallDelta {
                    index,
                    id,
                    call_type,
                    name,
                    arguments,
                } => {
                    use super::schema::{ToolCallDelta, ToolCallFunctionDelta};
                    let function = if name.is_some() || arguments.is_some() {
                        Some(ToolCallFunctionDelta { name, arguments })
                    } else {
                        None
                    };
                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_name.clone(),
                        system_fingerprint: sfp.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: None,
                                reasoning_content: None,
                                tool_calls: Some(vec![ToolCallDelta {
                                    index,
                                    id,
                                    call_type,
                                    function,
                                }]),
                            },
                            finish_reason: None,
                            logprobs: None,
                        }],
                        usage: None,
                        x_hf2q_timing: None,
                    };
                    yield Ok(Event::default()
                        .data(serde_json::to_string(&chunk).unwrap_or_default()));
                }
                GenerationEvent::Logprobs(lp) => {
                    if opts.logprobs {
                        pending_logprobs = Some(lp);
                    }
                }
                GenerationEvent::Done {
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                    stats,
                } => {
                    if let Some(metrics) = metrics.as_ref() {
                        metrics.record_chat_completion_success(prompt_tokens, completion_tokens);
                    }
                    let usage = if include_usage {
                        Some(UsageStats {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens: prompt_tokens + completion_tokens,
                            // Guarantees tune-up item 5a (2026-08-20):
                            // when usage is emitted, cached_tokens is
                            // ALWAYS reported explicitly — a cache miss
                            // is `cached_tokens: 0`, never an omitted
                            // field (omission was indistinguishable
                            // from "server doesn't report caching",
                            // while unary always reported it).
                            prompt_tokens_details: Some(PromptTokensDetails {
                                cached_tokens: stats.cached_prompt_tokens.unwrap_or(0),
                            }),
                            completion_tokens_details: stats
                                .reasoning_tokens
                                .map(|reasoning_tokens| CompletionTokensDetails { reasoning_tokens }),
                        })
                    } else {
                        None
                    };
                    let timing = StreamingTimingInfo {
                        prefill_time_secs: stats.prefill_time_secs,
                        decode_time_secs: stats.decode_time_secs,
                        total_time_secs: stats.total_time_secs,
                        time_to_first_token_ms: stats.time_to_first_token_ms,
                        prefill_tokens_per_sec: stats.prefill_tokens_per_sec,
                        decode_tokens_per_sec: stats.decode_tokens_per_sec,
                        gpu_sync_count: stats.gpu_sync_count,
                        gpu_dispatch_count: stats.gpu_dispatch_count,
                    };
                    let timing = (!timing.is_empty()).then_some(timing);
                    let final_chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_name.clone(),
                        system_fingerprint: sfp.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: None,
                                reasoning_content: None,
                                tool_calls: None,
                            },
                            finish_reason: Some(finish_reason.into()),
                            logprobs: pending_logprobs.take(),
                        }],
                        usage,
                        x_hf2q_timing: timing,
                    };
                    yield Ok(Event::default()
                        .data(serde_json::to_string(&final_chunk).unwrap_or_default()));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                GenerationEvent::Error(msg) => {
                    tracing::error!(error = %msg, "Generation error during streaming");
                    // iter-228a Phase-2c (Codex review of 3811f5d, finding #3
                    // medium): emit the error message as a content delta
                    // BEFORE the finish_reason chunk. Pre-Phase-2c the
                    // streaming path discarded the diagnostic — clients
                    // only saw a chunk with finish_reason='error' followed
                    // by [DONE], with no actionable message (e.g. the
                    // qwen3vl_text_forward_pending sentinel from iter-228a's
                    // engine seam was effectively invisible). Emitting the
                    // message as content matches what real OpenAI clients
                    // surface to users (Continue, OpenWebUI, etc. render
                    // delta.content even when finish_reason='error').
                    let message_chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_name.clone(),
                        system_fingerprint: sfp.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: Some(msg.clone()),
                                reasoning_content: None,
                                tool_calls: None,
                            },
                            finish_reason: None,
                            logprobs: None,
                        }],
                        usage: None,
                        x_hf2q_timing: None,
                    };
                    yield Ok(Event::default()
                        .data(serde_json::to_string(&message_chunk).unwrap_or_default()));
                    let error_chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_name.clone(),
                        system_fingerprint: sfp.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: None,
                                reasoning_content: None,
                                tool_calls: None,
                            },
                            finish_reason: Some("error".into()),
                            logprobs: None,
                        }],
                        usage: None,
                        x_hf2q_timing: None,
                    };
                    yield Ok(Event::default()
                        .data(serde_json::to_string(&error_chunk).unwrap_or_default()));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            }
        }

        // Channel closed without Done/Error — sender dropped. This is
        // abnormal; emit an `error` finish_reason so clients see a clean
        // termination instead of hanging.
        tracing::warn!("Generation channel closed unexpectedly");
        let error_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_name.clone(),
            system_fingerprint: sfp.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: Some("error".into()),
                logprobs: None,
            }],
            usage: None,
            x_hf2q_timing: None,
        };
        yield Ok(Event::default().data(serde_json::to_string(&error_chunk).unwrap_or_default()));
        yield Ok(Event::default().data("[DONE]"));
    }
}

/// 15-second SSE keepalive interval (Decision #20).
///
/// Exposed as a `pub const` so tests can pin the value without
/// constructing an `Sse<...>` + introspecting axum internals. Changing
/// this constant changes the wire-level keepalive cadence — see
/// ADR-040 §1.4 (client-invisibility contract): under
/// `SchedulerPolicy::FifoSerial` (today's default), the byte output at
/// N=1 must remain byte-identical to pre-ADR-040, which includes
/// keeping this interval at 15s.
pub const SSE_KEEPALIVE_INTERVAL_SECS: u64 = 15;

/// Empty comment text for keepalive frames (proxies + OpenAI SDK clients
/// tolerate the bare `:\n\n` frame).
pub const SSE_KEEPALIVE_TEXT: &str = "";

/// Public entrypoint: build an `Sse` response from a `GenerationEvent` stream.
///
/// Follows the OpenAI SSE format:
///   1. First chunk:   `delta: {"role": "assistant"}`
///   2. Token chunks:  `delta: {"content": "..."}` or
///                     `delta: {"reasoning_content": "..."}`
///   3. Tool chunks:   `delta: {"tool_calls": [...]}`
///   4. Final chunk:   `finish_reason: "stop" | "length" | "tool_calls" | "error"`
///                     (with `usage` when `include_usage`)
///   5. Terminator:    `data: [DONE]`
///
/// The returned `Sse<...>` has a 15-second keepalive layer (Decision #20) —
/// an empty SSE comment (`:\n\n`) is sent if no chunks have been written for
/// 15s. This prevents reverse-proxy and client idle-timeout disconnects.
///
/// # ADR-040 §6 Phase C C3 — per-slot keepalive seam
///
/// **cfa-iter-A5b MINOR #1 wording fix**: this section was previously
/// titled "per-slot keepalive accounting", which overclaimed —
/// `slot_id` is traced ONCE at stream construction, not per
/// keepalive frame. The 15-second keepalive itself IS wired through
/// axum's `KeepAlive`; what C3 added is a typed SEAM for future
/// per-slot attribution (per-frame accounting is a future iter).
///
/// The keepalive timer is implemented by axum's `KeepAlive`, whose
/// "last-emission" clock lives entirely inside the returned `Sse<...>`
/// future — i.e. per-`generation_events_to_sse`-call. Under
/// [`crate::serve::scheduler::SchedulerPolicy::FifoSerial`] (today,
/// `max_slots=1`) there is at most one such call in flight per engine,
/// so per-stream ≡ per-slot. Under
/// [`crate::serve::scheduler::SchedulerPolicy::InflightBatched`] (gated
/// behind `EngineMode::SlotAware { max_slots = N }` at Phase C2c
/// future), N concurrent handler futures each invoke this function
/// once, each gets its own `Sse<...>` + its own `KeepAlive` timer —
/// per-stream remains ≡ per-slot. **No cross-stream state aggregation**
/// is required to satisfy the "per-slot keepalive seam" promise in
/// ADR-040 §6 Phase C row C3; the structural shape is correct by
/// construction.
///
/// This entrypoint preserves the pre-C3 4-arg signature for the
/// existing `handlers.rs` call sites — it delegates to
/// [`generation_events_to_sse_with_slot`] with `slot_id = None`, which
/// is the FifoSerial path. Phase C2c+ will switch the handler call
/// site to [`generation_events_to_sse_with_slot`] directly so the
/// scheduler-allocated `SlotId.0` can be threaded through.
pub fn generation_events_to_sse(
    rx: mpsc::Receiver<GenerationEvent>,
    request_id: String,
    model_name: String,
    created: i64,
    opts: SseStreamOptions,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    generation_events_to_sse_with_slot(rx, request_id, model_name, created, opts, None)
}

/// Build an SSE response whose successful terminal event contributes to the
/// process-wide chat-completion and token counters. The legacy public encoder
/// remains metrics-neutral for callers and focused wire-format tests that do
/// not own server state.
pub(crate) fn generation_events_to_sse_with_metrics(
    rx: mpsc::Receiver<GenerationEvent>,
    request_id: String,
    model_name: String,
    created: i64,
    opts: SseStreamOptions,
    metrics: Arc<ServerMetrics>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    Sse::new(generation_events_stream_with_metrics(
        rx,
        request_id,
        model_name,
        created,
        opts,
        Some(metrics),
    ))
    .keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS))
            .text(SSE_KEEPALIVE_TEXT),
    )
}

/// Metrics-aware SSE whose stream owns an arbitrary lifetime guard. The
/// guard is dropped only when the stream terminates or the HTTP body is
/// dropped, which lets ADR-047 bind a model lease to the complete SSE body.
pub(crate) fn generation_events_to_sse_with_metrics_and_guard<G>(
    rx: mpsc::Receiver<GenerationEvent>,
    request_id: String,
    model_name: String,
    created: i64,
    opts: SseStreamOptions,
    metrics: Arc<ServerMetrics>,
    guard: G,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
where
    G: Send + 'static,
{
    let inner = generation_events_stream_with_metrics(
        rx,
        request_id,
        model_name,
        created,
        opts,
        Some(metrics),
    );
    let guarded = async_stream::stream! {
        let _guard = guard;
        futures::pin_mut!(inner);
        while let Some(item) = inner.next().await {
            yield item;
        }
    };
    Sse::new(guarded).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS))
            .text(SSE_KEEPALIVE_TEXT),
    )
}

/// **ADR-040 §6 Phase C C3** — slot-aware variant of
/// [`generation_events_to_sse`]. Accepts an optional `slot_id: Option<u32>`
/// carrying the scheduler-allocated `SlotId.0` for this stream.
///
/// The argument is `Option<u32>` (rather than
/// `Option<crate::serve::multi_seq_kv::SlotId>`) so this module remains
/// decoupled from `multi_seq_kv` — `sse.rs` is a wire-format encoder and
/// intentionally does not import the scheduler/KV types.
///
/// Semantics:
/// - **`slot_id = None` (FifoSerial today)**: legacy shape; the
///   per-stream `Sse<...>` + `KeepAlive` is the per-slot scope by
///   construction (`max_slots = 1`). Byte-identical to pre-C3 behaviour
///   (ADR-040 §1.4 client-invisibility contract). No tracing emitted on
///   the hot path.
/// - **`slot_id = Some(_)` (SlotAware future, C2c+)**: the handler
///   populates this from `SlotHandle::slot_id().0` so per-slot tracing
///   + `/metrics` can attribute keepalive frames to the physical slot.
///   The 15s interval ([`SSE_KEEPALIVE_INTERVAL_SECS`]) +
///   empty-comment-text ([`SSE_KEEPALIVE_TEXT`]) contract is unchanged
///   regardless of `slot_id`, preserving ADR-040 §1.4's "continuous
///   batching changes WHEN the request executes, not the
///   request/response shape" contract.
///
/// # Why a separate function (not an additional field on
/// [`SseStreamOptions`])
///
/// Adding a public field to `SseStreamOptions` would break the
/// existing struct-literal construction at
/// `src/serve/api/handlers.rs::chat_completions_stream` (which
/// constructs `SseStreamOptions { include_usage, logprobs,
/// system_fingerprint }` without a `..Default::default()` tail). The
/// brief for iter-C3 constrains edits to `sse.rs` + `schema.rs` +
/// the ADR doc, so a field addition would require an edit to
/// `handlers.rs` outside scope. The function-signature approach
/// preserves the wire-format struct's surface AND keeps the slot id
/// as a scheduler concept that lives on the function call, not the
/// per-chunk options.
pub fn generation_events_to_sse_with_slot(
    rx: mpsc::Receiver<GenerationEvent>,
    request_id: String,
    model_name: String,
    created: i64,
    opts: SseStreamOptions,
    slot_id: Option<u32>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // ADR-040 C3: record the per-slot association at stream
    // construction so /metrics + diagnostics can correlate the stream
    // with the responsible physical slot when SlotAware lands
    // (C2c+).  Under FifoSerial slot_id is None and we emit no trace
    // — the hot path remains byte-identical to pre-C3.
    //
    // cfa-iter-A5b MINOR #1 wording fix: this trace fires ONCE at
    // construction, not per keepalive frame; previous "per-slot
    // accounting" wording overclaimed. Per-frame attribution is a
    // future iter.
    if let Some(slot_id_value) = slot_id {
        tracing::trace!(
            request_id = %request_id,
            slot_id = slot_id_value,
            keepalive_interval_secs = SSE_KEEPALIVE_INTERVAL_SECS,
            "sse stream constructed (ADR-040 C3 per-slot seam)"
        );
    }
    Sse::new(generation_events_stream(
        rx, request_id, model_name, created, opts,
    ))
    .keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS))
            .text(SSE_KEEPALIVE_TEXT),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// These tests drive a real `tokio::sync::mpsc` into `generation_events_to_sse`
// and assert on the serialized SSE body. They do not mock the mpsc or the
// stream — the goal is to exercise the exact encoding path the handler uses.

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::response::IntoResponse;
    use futures::StreamExt;

    #[tokio::test]
    async fn sse_response_body_owns_lifetime_guard_until_drop() {
        #[derive(Clone)]
        struct DropProbe(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::Release);
            }
        }

        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let probe = DropProbe(Arc::clone(&dropped));
        let (_tx, rx) = mpsc::channel(1);
        let response = generation_events_to_sse_with_metrics_and_guard(
            rx,
            "chatcmpl-lease".into(),
            "model".into(),
            0,
            SseStreamOptions::default(),
            Arc::new(ServerMetrics::default()),
            probe,
        )
        .into_response();
        assert!(!dropped.load(std::sync::atomic::Ordering::Acquire));
        drop(response);
        assert!(dropped.load(std::sync::atomic::Ordering::Acquire));
    }
    use std::sync::atomic::Ordering;
    use tokio::sync::mpsc::Sender;

    /// Drain the Sse response body into a vector of SSE data-payload strings
    /// (one per event), with comment/keepalive frames skipped.
    async fn drain_sse<S>(sse: Sse<S>) -> Vec<String>
    where
        S: Stream<Item = Result<Event, Infallible>> + Send + 'static,
    {
        let resp = sse.into_response();
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = std::str::from_utf8(&bytes).unwrap().to_string();

        let mut out = Vec::new();
        for frame in text.split("\n\n") {
            let trimmed = frame.trim_end();
            if trimmed.is_empty() {
                continue;
            }
            let data_lines: Vec<&str> = trimmed
                .lines()
                .filter_map(|l| l.strip_prefix("data: "))
                .collect();
            if !data_lines.is_empty() {
                out.push(data_lines.join("\n"));
            }
        }
        out
    }

    async fn spawn_feeder(tx: Sender<GenerationEvent>, events: Vec<GenerationEvent>) {
        for ev in events {
            tx.send(ev).await.unwrap();
        }
        drop(tx);
    }

    fn make_sse(
        rx: mpsc::Receiver<GenerationEvent>,
        opts: SseStreamOptions,
    ) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
        generation_events_to_sse(
            rx,
            "req-test".into(),
            "gemma4-test".into(),
            1700000000,
            opts,
        )
    }

    fn assert_success_metrics(
        metrics: &ServerMetrics,
        completed: u64,
        prompt_tokens: u64,
        completion_tokens: u64,
    ) {
        assert_eq!(
            metrics.chat_completions_completed.load(Ordering::Relaxed),
            completed
        );
        assert_eq!(
            metrics.prompt_tokens_total.load(Ordering::Relaxed),
            prompt_tokens
        );
        assert_eq!(
            metrics.decode_tokens_total.load(Ordering::Relaxed),
            completion_tokens
        );
    }

    #[tokio::test]
    async fn successful_done_records_stream_metrics_once() {
        let metrics = Arc::new(ServerMetrics::default());
        let (tx, rx) = mpsc::channel(4);
        tx.send(GenerationEvent::Done {
            finish_reason: "stop",
            prompt_tokens: 17,
            completion_tokens: 5,
            stats: StreamStats::default(),
        })
        .await
        .unwrap();
        drop(tx);

        let sse = generation_events_to_sse_with_metrics(
            rx,
            "req-metrics".into(),
            "test-model".into(),
            1700000000,
            SseStreamOptions::default(),
            Arc::clone(&metrics),
        );
        let payloads = drain_sse(sse).await;
        assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));
        assert_success_metrics(&metrics, 1, 17, 5);
    }

    #[tokio::test]
    async fn duplicate_done_records_only_first_terminal_event() {
        let metrics = Arc::new(ServerMetrics::default());
        let (tx, rx) = mpsc::channel(4);
        tx.try_send(GenerationEvent::Done {
            finish_reason: "stop",
            prompt_tokens: 11,
            completion_tokens: 3,
            stats: StreamStats::default(),
        })
        .unwrap();
        tx.try_send(GenerationEvent::Done {
            finish_reason: "length",
            prompt_tokens: 999,
            completion_tokens: 999,
            stats: StreamStats::default(),
        })
        .unwrap();
        drop(tx);

        let sse = generation_events_to_sse_with_metrics(
            rx,
            "req-duplicate-done".into(),
            "test-model".into(),
            1700000000,
            SseStreamOptions::default(),
            Arc::clone(&metrics),
        );
        let _ = drain_sse(sse).await;
        assert_success_metrics(&metrics, 1, 11, 3);
    }

    #[tokio::test]
    async fn stream_error_does_not_record_success_metrics() {
        let metrics = Arc::new(ServerMetrics::default());
        let (tx, rx) = mpsc::channel(2);
        tx.send(GenerationEvent::Error("synthetic failure".into()))
            .await
            .unwrap();
        drop(tx);

        let sse = generation_events_to_sse_with_metrics(
            rx,
            "req-error".into(),
            "test-model".into(),
            1700000000,
            SseStreamOptions::default(),
            Arc::clone(&metrics),
        );
        let _ = drain_sse(sse).await;
        assert_success_metrics(&metrics, 0, 0, 0);
    }

    #[tokio::test]
    async fn sender_close_without_terminal_does_not_record_success_metrics() {
        let metrics = Arc::new(ServerMetrics::default());
        let (tx, rx) = mpsc::channel(1);
        drop(tx);

        let sse = generation_events_to_sse_with_metrics(
            rx,
            "req-close".into(),
            "test-model".into(),
            1700000000,
            SseStreamOptions::default(),
            Arc::clone(&metrics),
        );
        let _ = drain_sse(sse).await;
        assert_success_metrics(&metrics, 0, 0, 0);
    }

    #[tokio::test]
    async fn consumer_drop_before_done_does_not_record_success_metrics() {
        let metrics = Arc::new(ServerMetrics::default());
        let (tx, rx) = mpsc::channel(4);
        tx.try_send(GenerationEvent::Delta {
            kind: DeltaKind::Content,
            text: "partial".into(),
        })
        .unwrap();
        tx.try_send(GenerationEvent::Done {
            finish_reason: "stop",
            prompt_tokens: 13,
            completion_tokens: 2,
            stats: StreamStats::default(),
        })
        .unwrap();
        drop(tx);

        let mut stream = Box::pin(generation_events_stream_with_metrics(
            rx,
            "req-drop".into(),
            "test-model".into(),
            1700000000,
            SseStreamOptions::default(),
            Some(Arc::clone(&metrics)),
        ));
        assert!(stream.as_mut().next().await.is_some(), "role event");
        assert!(stream.as_mut().next().await.is_some(), "content event");
        drop(stream);
        assert_success_metrics(&metrics, 0, 0, 0);
    }

    #[tokio::test]
    async fn emits_role_chunk_first_then_content_then_done() {
        let (tx, rx) = mpsc::channel(8);
        let events = vec![
            GenerationEvent::Delta {
                kind: DeltaKind::Content,
                text: "Hello".into(),
            },
            GenerationEvent::Delta {
                kind: DeltaKind::Content,
                text: ", world!".into(),
            },
            GenerationEvent::Done {
                finish_reason: "stop",
                prompt_tokens: 5,
                completion_tokens: 3,
                stats: StreamStats::default(),
            },
        ];
        let sse = make_sse(rx, SseStreamOptions::default());
        tokio::spawn(spawn_feeder(tx, events));
        let payloads = drain_sse(sse).await;
        assert!(
            payloads.len() >= 4,
            "got {} payloads: {:?}",
            payloads.len(),
            payloads
        );
        // Role chunk
        let role: serde_json::Value = serde_json::from_str(&payloads[0]).unwrap();
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        // Content chunks
        let c0: serde_json::Value = serde_json::from_str(&payloads[1]).unwrap();
        assert_eq!(c0["choices"][0]["delta"]["content"], "Hello");
        let c1: serde_json::Value = serde_json::from_str(&payloads[2]).unwrap();
        assert_eq!(c1["choices"][0]["delta"]["content"], ", world!");
        // Final chunk
        let done: serde_json::Value = serde_json::from_str(&payloads[3]).unwrap();
        assert_eq!(done["choices"][0]["finish_reason"], "stop");
        // By default, usage is not included (Decision: opt-in via stream_options).
        assert!(done.get("usage").is_none() || done["usage"].is_null());
        // Terminator
        assert_eq!(payloads.last().unwrap(), "[DONE]");
    }

    #[tokio::test]
    async fn reasoning_delta_routes_to_reasoning_content_slot() {
        let (tx, rx) = mpsc::channel(8);
        let events = vec![
            GenerationEvent::Delta {
                kind: DeltaKind::Reasoning,
                text: "let me think...".into(),
            },
            GenerationEvent::Delta {
                kind: DeltaKind::Content,
                text: "42".into(),
            },
            GenerationEvent::Done {
                finish_reason: "stop",
                prompt_tokens: 3,
                completion_tokens: 4,
                stats: StreamStats::default(),
            },
        ];
        let sse = make_sse(rx, SseStreamOptions::default());
        tokio::spawn(spawn_feeder(tx, events));
        let payloads = drain_sse(sse).await;
        // role, reasoning, content, done, [DONE]
        assert_eq!(payloads.len(), 5);
        let reasoning: serde_json::Value = serde_json::from_str(&payloads[1]).unwrap();
        assert_eq!(
            reasoning["choices"][0]["delta"]["reasoning_content"],
            "let me think..."
        );
        assert!(reasoning["choices"][0]["delta"].get("content").is_none());
        let content: serde_json::Value = serde_json::from_str(&payloads[2]).unwrap();
        assert_eq!(content["choices"][0]["delta"]["content"], "42");
        assert!(content["choices"][0]["delta"]
            .get("reasoning_content")
            .is_none());
    }

    #[tokio::test]
    async fn include_usage_true_yields_usage_in_final_chunk() {
        let (tx, rx) = mpsc::channel(8);
        let stats = StreamStats {
            cached_prompt_tokens: Some(2),
            reasoning_tokens: Some(1),
            ..Default::default()
        };
        let events = vec![
            GenerationEvent::Delta {
                kind: DeltaKind::Content,
                text: "ok".into(),
            },
            GenerationEvent::Done {
                finish_reason: "stop",
                prompt_tokens: 7,
                completion_tokens: 5,
                stats,
            },
        ];
        let opts = SseStreamOptions {
            include_usage: true,
            system_fingerprint: Some("hf2q-test-mlx-native".into()),
            ..Default::default()
        };
        let sse = make_sse(rx, opts);
        tokio::spawn(spawn_feeder(tx, events));
        let payloads = drain_sse(sse).await;
        let done: serde_json::Value = serde_json::from_str(&payloads[payloads.len() - 2]).unwrap();
        assert_eq!(done["usage"]["prompt_tokens"], 7);
        assert_eq!(done["usage"]["completion_tokens"], 5);
        assert_eq!(done["usage"]["total_tokens"], 12);
        assert_eq!(done["usage"]["prompt_tokens_details"]["cached_tokens"], 2);
        assert_eq!(
            done["usage"]["completion_tokens_details"]["reasoning_tokens"],
            1
        );
        assert_eq!(done["system_fingerprint"], "hf2q-test-mlx-native");
        assert!(done.get("x_hf2q_timing").is_none());
    }

    /// Guarantees tune-up item 5a (2026-08-20): a cache MISS (producer
    /// reported no cached tokens) still yields an explicit
    /// `prompt_tokens_details.cached_tokens: 0` in the usage frame —
    /// the field is never omitted when usage is emitted, so
    /// orchestrators can distinguish "no cache hit" from "server does
    /// not report caching".
    #[tokio::test]
    async fn include_usage_cache_miss_reports_explicit_zero_cached_tokens() {
        let (tx, rx) = mpsc::channel(8);
        let events = vec![GenerationEvent::Done {
            finish_reason: "stop",
            prompt_tokens: 7,
            completion_tokens: 5,
            // Default stats: cached_prompt_tokens = None (miss / not
            // measured).
            stats: StreamStats::default(),
        }];
        let opts = SseStreamOptions {
            include_usage: true,
            ..Default::default()
        };
        let sse = make_sse(rx, opts);
        tokio::spawn(spawn_feeder(tx, events));
        let payloads = drain_sse(sse).await;
        let done: serde_json::Value = serde_json::from_str(&payloads[payloads.len() - 2]).unwrap();
        assert_eq!(
            done["usage"]["prompt_tokens_details"]["cached_tokens"], 0,
            "cache miss MUST serialize cached_tokens: 0 explicitly, not omit \
             prompt_tokens_details"
        );
    }

    #[tokio::test]
    async fn final_chunk_serializes_only_available_timing_fields() {
        let (tx, rx) = mpsc::channel(4);
        let events = vec![GenerationEvent::Done {
            finish_reason: "stop",
            prompt_tokens: 7,
            completion_tokens: 5,
            stats: StreamStats {
                time_to_first_token_ms: Some(42.5),
                decode_tokens_per_sec: Some(18.25),
                ..Default::default()
            },
        }];
        let sse = make_sse(rx, SseStreamOptions::default());
        tokio::spawn(spawn_feeder(tx, events));
        let payloads = drain_sse(sse).await;
        let done: serde_json::Value = serde_json::from_str(&payloads[payloads.len() - 2]).unwrap();
        assert_eq!(done["x_hf2q_timing"]["time_to_first_token_ms"], 42.5);
        assert_eq!(done["x_hf2q_timing"]["decode_tokens_per_sec"], 18.25);
        assert!(done["x_hf2q_timing"].get("gpu_sync_count").is_none());
        assert!(done.get("usage").is_none());
    }

    #[tokio::test]
    async fn error_event_emits_error_message_then_finish_reason_then_done() {
        let (tx, rx) = mpsc::channel(4);
        let events = vec![
            GenerationEvent::Delta {
                kind: DeltaKind::Content,
                text: "partial".into(),
            },
            GenerationEvent::Error("metal panic".into()),
        ];
        let sse = make_sse(rx, SseStreamOptions::default());
        tokio::spawn(spawn_feeder(tx, events));
        let payloads = drain_sse(sse).await;
        // iter-228a Phase-2c (Codex finding #3): emission shape is now
        // role, content, ERROR-MESSAGE-AS-CONTENT, error-final-with-
        // finish_reason, [DONE] = 5 payloads (was 4 pre-Phase-2c).
        // Carrying the error message as a content delta lets streaming
        // clients surface the diagnostic to users instead of dropping
        // it on the floor.
        assert_eq!(payloads.len(), 5);
        let msg_chunk: serde_json::Value = serde_json::from_str(&payloads[2]).unwrap();
        assert_eq!(
            msg_chunk["choices"][0]["delta"]["content"], "metal panic",
            "Phase-2c: error message MUST be emitted as a content delta \
             before the finish_reason chunk so streaming clients see \
             the diagnostic"
        );
        assert!(
            msg_chunk["choices"][0]["finish_reason"].is_null(),
            "message chunk must NOT carry finish_reason (the next chunk does)"
        );
        let err: serde_json::Value = serde_json::from_str(&payloads[3]).unwrap();
        assert_eq!(err["choices"][0]["finish_reason"], "error");
        assert_eq!(payloads.last().unwrap(), "[DONE]");
    }

    #[tokio::test]
    async fn channel_closed_without_done_emits_error_and_terminator() {
        let (tx, rx) = mpsc::channel(4);
        let events = vec![GenerationEvent::Delta {
            kind: DeltaKind::Content,
            text: "fragment".into(),
        }];
        let sse = make_sse(rx, SseStreamOptions::default());
        tokio::spawn(spawn_feeder(tx, events));
        let payloads = drain_sse(sse).await;
        assert_eq!(payloads.last().unwrap(), "[DONE]");
        // Second-to-last chunk should be an error finish_reason
        let err: serde_json::Value = serde_json::from_str(&payloads[payloads.len() - 2]).unwrap();
        assert_eq!(err["choices"][0]["finish_reason"], "error");
    }

    #[tokio::test]
    async fn tool_call_delta_round_trips_through_sse() {
        let (tx, rx) = mpsc::channel(4);
        let events = vec![
            GenerationEvent::ToolCallDelta {
                index: 0,
                id: Some("call_abc".into()),
                call_type: Some("function".into()),
                name: Some("get_weather".into()),
                arguments: None,
            },
            GenerationEvent::ToolCallDelta {
                index: 0,
                id: None,
                call_type: None,
                name: None,
                arguments: Some("{\"city\":".into()),
            },
            GenerationEvent::ToolCallDelta {
                index: 0,
                id: None,
                call_type: None,
                name: None,
                arguments: Some("\"NYC\"}".into()),
            },
            GenerationEvent::Done {
                finish_reason: "tool_calls",
                prompt_tokens: 9,
                completion_tokens: 7,
                stats: StreamStats::default(),
            },
        ];
        let sse = make_sse(rx, SseStreamOptions::default());
        tokio::spawn(spawn_feeder(tx, events));
        let payloads = drain_sse(sse).await;
        // role, 3× tool_call delta, final, [DONE]
        assert_eq!(payloads.len(), 6);
        let first_tc: serde_json::Value = serde_json::from_str(&payloads[1]).unwrap();
        let tc = &first_tc["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["index"], 0);
        assert_eq!(tc["id"], "call_abc");
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "get_weather");
        let second_tc: serde_json::Value = serde_json::from_str(&payloads[2]).unwrap();
        let tc2 = &second_tc["choices"][0]["delta"]["tool_calls"][0];
        assert!(tc2.get("id").is_none() || tc2["id"].is_null());
        assert_eq!(tc2["function"]["arguments"], "{\"city\":");
        let done: serde_json::Value = serde_json::from_str(&payloads[4]).unwrap();
        assert_eq!(done["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn logprobs_attach_to_next_content_chunk_when_enabled() {
        use super::super::schema::{ChoiceLogprobs, TokenLogprob};
        let (tx, rx) = mpsc::channel(8);
        let lp = ChoiceLogprobs {
            content: vec![TokenLogprob {
                token: "Hello".into(),
                logprob: -0.1,
                bytes: None,
                top_logprobs: Vec::new(),
            }],
        };
        let events = vec![
            GenerationEvent::Logprobs(lp),
            GenerationEvent::Delta {
                kind: DeltaKind::Content,
                text: "Hello".into(),
            },
            GenerationEvent::Done {
                finish_reason: "stop",
                prompt_tokens: 2,
                completion_tokens: 1,
                stats: StreamStats::default(),
            },
        ];
        let opts = SseStreamOptions {
            logprobs: true,
            ..Default::default()
        };
        let sse = make_sse(rx, opts);
        tokio::spawn(spawn_feeder(tx, events));
        let payloads = drain_sse(sse).await;
        // role, content (with logprobs), done, [DONE]
        let content: serde_json::Value = serde_json::from_str(&payloads[1]).unwrap();
        assert_eq!(
            content["choices"][0]["logprobs"]["content"][0]["token"],
            "Hello"
        );
    }

    #[tokio::test]
    async fn logprobs_ignored_when_disabled_in_opts() {
        use super::super::schema::{ChoiceLogprobs, TokenLogprob};
        let (tx, rx) = mpsc::channel(8);
        let lp = ChoiceLogprobs {
            content: vec![TokenLogprob {
                token: "Hi".into(),
                logprob: -0.3,
                bytes: None,
                top_logprobs: Vec::new(),
            }],
        };
        let events = vec![
            GenerationEvent::Logprobs(lp),
            GenerationEvent::Delta {
                kind: DeltaKind::Content,
                text: "Hi".into(),
            },
            GenerationEvent::Done {
                finish_reason: "stop",
                prompt_tokens: 1,
                completion_tokens: 1,
                stats: StreamStats::default(),
            },
        ];
        // opts.logprobs = false (default)
        let sse = make_sse(rx, SseStreamOptions::default());
        tokio::spawn(spawn_feeder(tx, events));
        let payloads = drain_sse(sse).await;
        let content: serde_json::Value = serde_json::from_str(&payloads[1]).unwrap();
        assert!(
            content["choices"][0].get("logprobs").is_none()
                || content["choices"][0]["logprobs"].is_null()
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // ADR-040 §6 Phase C C3 — per-slot keepalive seam tests
    // (cfa-iter-A5b MINOR #1 wording fix: "seam" not "accounting" —
    // slot_id is traced once at construction, not per-keepalive-frame).
    // ───────────────────────────────────────────────────────────────────
    //
    // These pin the C3 structural shape: each `generation_events_to_sse`
    // call constructs its OWN `Sse<...>` + its OWN `KeepAlive`, so the
    // per-stream timer state is per-slot by construction (under SlotAware
    // each slot ⇔ one handler future ⇔ one call to this function). The
    // tests also pin the 15s interval (regression guard against §1.4
    // byte-invariance violation) + the N=1 byte-identical-to-pre-C3
    // contract under FifoSerial.

    /// **ADR-040 C3** — independent slot-aware
    /// [`generation_events_to_sse_with_slot`] invocations do NOT share
    /// state. The output payloads for slot 0 vs slot 1 reflect only the
    /// events fed to each respective stream; the slot association is
    /// per-`Sse<...>`-instance, never cross-stream-aggregated. This is
    /// the load-bearing pin for the "per-slot keepalive seam"
    /// contract under SlotAware (C2c+).
    #[tokio::test]
    async fn c3_sse_keepalive_per_slot_state_is_isolated() {
        // Stream A: slot 0, emits "alpha" then Done.
        let (tx_a, rx_a) = mpsc::channel(4);
        let sse_a = generation_events_to_sse_with_slot(
            rx_a,
            "req-slot0".into(),
            "test-model".into(),
            1700000000,
            SseStreamOptions::default(),
            Some(0),
        );
        let events_a = vec![
            GenerationEvent::Delta {
                kind: DeltaKind::Content,
                text: "alpha".into(),
            },
            GenerationEvent::Done {
                finish_reason: "stop",
                prompt_tokens: 1,
                completion_tokens: 1,
                stats: StreamStats::default(),
            },
        ];

        // Stream B: slot 1, emits "beta" then Done. Constructed
        // concurrently to prove there is no shared per-engine
        // keepalive state to leak across.
        let (tx_b, rx_b) = mpsc::channel(4);
        let sse_b = generation_events_to_sse_with_slot(
            rx_b,
            "req-slot1".into(),
            "test-model".into(),
            1700000000,
            SseStreamOptions::default(),
            Some(1),
        );
        let events_b = vec![
            GenerationEvent::Delta {
                kind: DeltaKind::Content,
                text: "beta".into(),
            },
            GenerationEvent::Done {
                finish_reason: "stop",
                prompt_tokens: 1,
                completion_tokens: 1,
                stats: StreamStats::default(),
            },
        ];

        tokio::spawn(spawn_feeder(tx_a, events_a));
        tokio::spawn(spawn_feeder(tx_b, events_b));

        let payloads_a = drain_sse(sse_a).await;
        let payloads_b = drain_sse(sse_b).await;

        // Each stream's content reflects ONLY its own feed (no
        // cross-stream contamination of the per-stream encoder state).
        let content_a: serde_json::Value = serde_json::from_str(&payloads_a[1]).unwrap();
        let content_b: serde_json::Value = serde_json::from_str(&payloads_b[1]).unwrap();
        assert_eq!(
            content_a["choices"][0]["delta"]["content"], "alpha",
            "ADR-040 C3: slot 0 stream MUST surface only its own \
             feed; per-slot keepalive seam requires per-stream \
             encoder state isolation"
        );
        assert_eq!(
            content_b["choices"][0]["delta"]["content"], "beta",
            "ADR-040 C3: slot 1 stream MUST surface only its own \
             feed; per-slot keepalive seam requires per-stream \
             encoder state isolation"
        );
        // request_id is also per-stream — would catch a regression
        // that accidentally globalised stream metadata.
        let role_a: serde_json::Value = serde_json::from_str(&payloads_a[0]).unwrap();
        let role_b: serde_json::Value = serde_json::from_str(&payloads_b[0]).unwrap();
        assert_eq!(role_a["id"], "req-slot0");
        assert_eq!(role_b["id"], "req-slot1");
        assert_ne!(
            role_a["id"], role_b["id"],
            "ADR-040 C3: each per-slot stream carries its own \
             request_id (vacuous-test guard)"
        );
    }

    /// **ADR-040 C3** — the 15s keepalive interval is byte-pinned at
    /// the [`SSE_KEEPALIVE_INTERVAL_SECS`] constant. Regression guard
    /// against ADR-040 §1.4 client-invisibility violation: changing the
    /// interval under FifoSerial would alter the client-observable
    /// keepalive cadence and break Decision #20.
    #[test]
    fn c3_sse_keepalive_15s_interval_unchanged_under_fifo_serial() {
        assert_eq!(
            SSE_KEEPALIVE_INTERVAL_SECS, 15,
            "ADR-040 §1.4 + Decision #20: SSE keepalive interval MUST \
             remain 15s under SchedulerPolicy::FifoSerial. Changing \
             this constant breaks the byte-invariance contract for \
             clients at N=1."
        );
        assert_eq!(
            SSE_KEEPALIVE_TEXT, "",
            "ADR-040 §1.4 + Decision #20: SSE keepalive frame text \
             MUST remain empty (`:\\n\\n` comment frame). Non-empty \
             keepalive text would emit a `data:` line which SDK \
             clients would parse as a generation chunk."
        );
    }

    /// **ADR-040 C3** — under FifoSerial (the legacy entrypoint
    /// [`generation_events_to_sse`]), the byte stream is identical to
    /// the same options + same events fed through the C3-aware
    /// [`generation_events_to_sse_with_slot`] with `slot_id = None`.
    /// This pins the §1.4 client-invisibility contract: the C3
    /// helper's `slot_id = None` path MUST be byte-equivalent to the
    /// pre-C3 legacy entrypoint at N=1 under FifoSerial.
    #[tokio::test]
    async fn c3_sse_keepalive_no_byte_change_at_n1_under_serialfifo() {
        // Stream 1: legacy 4-arg entrypoint (handlers.rs path).
        let (tx1, rx1) = mpsc::channel(4);
        let sse1 = generation_events_to_sse(
            rx1,
            "req-legacy".into(),
            "test-model".into(),
            1700000000,
            SseStreamOptions::default(),
        );

        // Stream 2: C3-aware entrypoint with slot_id=None — must be
        // byte-equivalent to (1) since both go through the same
        // 15s keepalive + same encoder + same options.
        let (tx2, rx2) = mpsc::channel(4);
        let sse2 = generation_events_to_sse_with_slot(
            rx2,
            "req-legacy".into(),
            "test-model".into(),
            1700000000,
            SseStreamOptions::default(),
            None,
        );

        let events_factory = || {
            vec![
                GenerationEvent::Delta {
                    kind: DeltaKind::Content,
                    text: "Hello".into(),
                },
                GenerationEvent::Delta {
                    kind: DeltaKind::Content,
                    text: ", world!".into(),
                },
                GenerationEvent::Done {
                    finish_reason: "stop",
                    prompt_tokens: 5,
                    completion_tokens: 3,
                    stats: StreamStats::default(),
                },
            ]
        };
        tokio::spawn(spawn_feeder(tx1, events_factory()));
        tokio::spawn(spawn_feeder(tx2, events_factory()));

        let payloads1 = drain_sse(sse1).await;
        let payloads2 = drain_sse(sse2).await;

        assert_eq!(
            payloads1, payloads2,
            "ADR-040 §1.4: generation_events_to_sse (legacy 4-arg) \
             must be byte-identical to \
             generation_events_to_sse_with_slot(.., slot_id=None) — \
             the C3 helper's None branch IS the FifoSerial path and \
             MUST NOT change the wire output at N=1"
        );

        // Sanity: the byte-stream is the same shape as the
        // emits_role_chunk_first_then_content_then_done test (5
        // frames: role, content, content, done, [DONE]).
        assert_eq!(payloads1.len(), 5);
        assert_eq!(payloads1.last().unwrap(), "[DONE]");
    }
}
