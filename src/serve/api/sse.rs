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

use std::convert::Infallible;

use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::Stream;
use tokio::sync::mpsc;

use super::schema::{
    ChatCompletionChunk, ChoiceLogprobs, ChunkChoice, ChunkDelta, CompletionTokensDetails,
    PromptTokensDetails, UsageStats,
};

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
    Delta {
        kind: DeltaKind,
        text: String,
    },
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
    mut rx: mpsc::Receiver<GenerationEvent>,
    request_id: String,
    model_name: String,
    created: i64,
    opts: SseStreamOptions,
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
                    let usage = if include_usage {
                        Some(UsageStats {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens: prompt_tokens + completion_tokens,
                            prompt_tokens_details: stats
                                .cached_prompt_tokens
                                .map(|cached_tokens| PromptTokensDetails { cached_tokens }),
                            completion_tokens_details: stats
                                .reasoning_tokens
                                .map(|reasoning_tokens| CompletionTokensDetails { reasoning_tokens }),
                        })
                    } else {
                        None
                    };
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
                    };
                    yield Ok(Event::default()
                        .data(serde_json::to_string(&final_chunk).unwrap_or_default()));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                GenerationEvent::Error(msg) => {
                    tracing::error!(error = %msg, "Generation error during streaming");
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
        };
        yield Ok(Event::default().data(serde_json::to_string(&error_chunk).unwrap_or_default()));
        yield Ok(Event::default().data("[DONE]"));
    }
}

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
pub fn generation_events_to_sse(
    rx: mpsc::Receiver<GenerationEvent>,
    request_id: String,
    model_name: String,
    created: i64,
    opts: SseStreamOptions,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    Sse::new(generation_events_stream(rx, request_id, model_name, created, opts)).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text(""),
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
        generation_events_to_sse(rx, "req-test".into(), "gemma4-test".into(), 1700000000, opts)
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
        assert!(payloads.len() >= 4, "got {} payloads: {:?}", payloads.len(), payloads);
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
        assert!(
            content["choices"][0]["delta"].get("reasoning_content").is_none()
        );
    }

    #[tokio::test]
    async fn include_usage_true_yields_usage_in_final_chunk() {
        let (tx, rx) = mpsc::channel(8);
        let mut stats = StreamStats::default();
        stats.cached_prompt_tokens = Some(2);
        stats.reasoning_tokens = Some(1);
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
        let mut opts = SseStreamOptions::default();
        opts.include_usage = true;
        opts.system_fingerprint = Some("hf2q-test-mlx-native".into());
        let sse = make_sse(rx, opts);
        tokio::spawn(spawn_feeder(tx, events));
        let payloads = drain_sse(sse).await;
        let done: serde_json::Value =
            serde_json::from_str(&payloads[payloads.len() - 2]).unwrap();
        assert_eq!(done["usage"]["prompt_tokens"], 7);
        assert_eq!(done["usage"]["completion_tokens"], 5);
        assert_eq!(done["usage"]["total_tokens"], 12);
        assert_eq!(done["usage"]["prompt_tokens_details"]["cached_tokens"], 2);
        assert_eq!(done["usage"]["completion_tokens_details"]["reasoning_tokens"], 1);
        assert_eq!(done["system_fingerprint"], "hf2q-test-mlx-native");
    }

    #[tokio::test]
    async fn error_event_emits_error_finish_reason_then_done() {
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
        // role, content, error-final, [DONE]
        assert_eq!(payloads.len(), 4);
        let err: serde_json::Value = serde_json::from_str(&payloads[2]).unwrap();
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
        let err: serde_json::Value =
            serde_json::from_str(&payloads[payloads.len() - 2]).unwrap();
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
        let mut opts = SseStreamOptions::default();
        opts.logprobs = true;
        let sse = make_sse(rx, opts);
        tokio::spawn(spawn_feeder(tx, events));
        let payloads = drain_sse(sse).await;
        // role, content (with logprobs), done, [DONE]
        let content: serde_json::Value = serde_json::from_str(&payloads[1]).unwrap();
        assert_eq!(content["choices"][0]["logprobs"]["content"][0]["token"], "Hello");
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
}
