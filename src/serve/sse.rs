//! SSE stream encoding for OpenAI-compatible streaming responses.
//!
//! Converts a stream of `GenerationEvent` messages from the sync-to-async
//! bridge into properly formatted Server-Sent Events matching the OpenAI
//! chat completion chunk format.

use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use tokio::sync::mpsc;

use super::schema::{
    ChatCompletionChunk, ChunkChoice, ChunkDelta, ToolCallDelta, ToolCallFunctionDelta, UsageStats,
};
use super::tool_parser::{generate_tool_call_id, ToolCallParser, ToolParserEvent};

/// Events sent from the blocking generation thread to the async SSE stream.
#[derive(Debug)]
pub enum GenerationEvent {
    /// A generated token fragment.
    Token(String),
    /// Generation completed successfully.
    Done {
        prompt_tokens: usize,
        completion_tokens: usize,
        /// Full generation statistics for timing/GPU counter reporting.
        /// Carried through to the non-streaming handler to populate
        /// `x_hf2q_timing` in the response.
        stats: crate::inference::engine::GenerationStats,
    },
    /// Generation failed mid-stream.
    Error(String),
}

/// Convert an mpsc receiver of generation events into an SSE stream.
///
/// The stream follows the OpenAI SSE format:
/// 1. First chunk: `delta: {"role": "assistant"}`
/// 2. Token chunks: `delta: {"content": "<token>"}`
/// 3. Final chunk: `delta: {}`, `finish_reason: "stop"` (or "error")
/// 4. Terminator: `data: [DONE]`
/// Convert an mpsc receiver of generation events into an SSE stream.
///
/// This is the basic version without tool call parsing. The tool-aware
/// version is `generation_events_to_sse_with_tools`.
#[allow(dead_code)]
pub fn generation_events_to_sse(
    mut rx: mpsc::Receiver<GenerationEvent>,
    request_id: String,
    model_name: String,
    created: i64,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        // Emit the initial role chunk
        let role_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".into(),
            created,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".into()),
                    content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let json = serde_json::to_string(&role_chunk).unwrap_or_default();
        yield Ok(Event::default().data(json));

        // Process events from the generation thread
        while let Some(event) = rx.recv().await {
            match event {
                GenerationEvent::Token(text) => {
                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: Some(text),
                                tool_calls: None,
                            },
                            finish_reason: None,
                        }],
                        usage: None,
                    };
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    yield Ok(Event::default().data(json));
                }
                GenerationEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                    stats: _,
                } => {
                    // Emit final chunk with finish_reason "stop"
                    let final_chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: None,
                                tool_calls: None,
                            },
                            finish_reason: Some("stop".into()),
                        }],
                        usage: Some(UsageStats {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens: prompt_tokens + completion_tokens,
                            prompt_tokens_details: None,
                        }),
                    };
                    let json = serde_json::to_string(&final_chunk).unwrap_or_default();
                    yield Ok(Event::default().data(json));
                    // Emit [DONE] terminator
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                GenerationEvent::Error(msg) => {
                    tracing::error!(error = %msg, "Generation error during streaming");
                    // Emit error chunk with finish_reason "error"
                    let error_chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: None,
                                tool_calls: None,
                            },
                            finish_reason: Some("error".into()),
                        }],
                        usage: None,
                    };
                    let json = serde_json::to_string(&error_chunk).unwrap_or_default();
                    yield Ok(Event::default().data(json));
                    // Always terminate with [DONE]
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            }
        }

        // Channel closed without Done/Error -- sender dropped unexpectedly.
        // Emit an error chunk and terminate cleanly so clients don't hang.
        tracing::warn!("Generation channel closed unexpectedly");
        let error_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".into(),
            created,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                },
                finish_reason: Some("error".into()),
            }],
            usage: None,
        };
        let json = serde_json::to_string(&error_chunk).unwrap_or_default();
        yield Ok(Event::default().data(json));
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text(""),
    )
}

/// Convert generation events into an SSE stream with tool call parsing.
///
/// When `tools_active` is true, each token is fed through the `ToolCallParser`
/// which detects tool call patterns and emits `tool_calls` deltas instead of
/// (or in addition to) `content` deltas.
pub fn generation_events_to_sse_with_tools(
    mut rx: mpsc::Receiver<GenerationEvent>,
    request_id: String,
    model_name: String,
    created: i64,
    tools_active: bool,
    forced_function: Option<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let mut parser = ToolCallParser::new(tools_active, forced_function);
        // Track which tool call IDs have been assigned (indexed by tool call index)
        let mut tool_call_ids: Vec<String> = Vec::new();
        // Track which tool calls have had their first delta emitted
        let mut first_delta_emitted: Vec<bool> = Vec::new();

        // Emit the initial role chunk
        let role_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".into(),
            created,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".into()),
                    content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let json = serde_json::to_string(&role_chunk).unwrap_or_default();
        yield Ok(Event::default().data(json));

        // Process events from the generation thread
        while let Some(event) = rx.recv().await {
            match event {
                GenerationEvent::Token(text) => {
                    let parser_events = parser.feed(&text);
                    for pe in parser_events {
                        match pe {
                            ToolParserEvent::ContentDelta(content_text) => {
                                let chunk = ChatCompletionChunk {
                                    id: request_id.clone(),
                                    object: "chat.completion.chunk".into(),
                                    created,
                                    model: model_name.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: ChunkDelta {
                                            role: None,
                                            content: Some(content_text),
                                            tool_calls: None,
                                        },
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                };
                                let json = serde_json::to_string(&chunk).unwrap_or_default();
                                yield Ok(Event::default().data(json));
                            }
                            ToolParserEvent::ToolCallStart { index } => {
                                // Assign a unique ID for this tool call
                                while tool_call_ids.len() <= index {
                                    tool_call_ids.push(generate_tool_call_id());
                                    first_delta_emitted.push(false);
                                }
                            }
                            ToolParserEvent::NameDelta { index, text: name_text } => {
                                let is_first = !first_delta_emitted.get(index).copied().unwrap_or(true);
                                if let Some(flag) = first_delta_emitted.get_mut(index) {
                                    *flag = true;
                                }

                                let tc_delta = ToolCallDelta {
                                    index,
                                    id: if is_first { tool_call_ids.get(index).cloned() } else { None },
                                    call_type: if is_first { Some("function".to_string()) } else { None },
                                    function: Some(ToolCallFunctionDelta {
                                        name: Some(name_text),
                                        arguments: None,
                                    }),
                                };

                                let chunk = ChatCompletionChunk {
                                    id: request_id.clone(),
                                    object: "chat.completion.chunk".into(),
                                    created,
                                    model: model_name.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: ChunkDelta {
                                            role: None,
                                            content: None,
                                            tool_calls: Some(vec![tc_delta]),
                                        },
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                };
                                let json = serde_json::to_string(&chunk).unwrap_or_default();
                                yield Ok(Event::default().data(json));
                            }
                            ToolParserEvent::ArgumentsDelta { index, text: args_text } => {
                                let tc_delta = ToolCallDelta {
                                    index,
                                    id: None,
                                    call_type: None,
                                    function: Some(ToolCallFunctionDelta {
                                        name: None,
                                        arguments: Some(args_text),
                                    }),
                                };

                                let chunk = ChatCompletionChunk {
                                    id: request_id.clone(),
                                    object: "chat.completion.chunk".into(),
                                    created,
                                    model: model_name.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: ChunkDelta {
                                            role: None,
                                            content: None,
                                            tool_calls: Some(vec![tc_delta]),
                                        },
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                };
                                let json = serde_json::to_string(&chunk).unwrap_or_default();
                                yield Ok(Event::default().data(json));
                            }
                            ToolParserEvent::ToolCallEnd { .. } => {
                                // Tool call complete -- no specific SSE event needed;
                                // finish_reason is set on the final chunk.
                            }
                        }
                    }
                }
                GenerationEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                    stats: _,
                } => {
                    // Finalize the parser to flush any in-progress state
                    let final_events = parser.finalize();
                    for pe in final_events {
                        match pe {
                            ToolParserEvent::ContentDelta(text) if !text.is_empty() => {
                                let chunk = ChatCompletionChunk {
                                    id: request_id.clone(),
                                    object: "chat.completion.chunk".into(),
                                    created,
                                    model: model_name.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: ChunkDelta {
                                            role: None,
                                            content: Some(text),
                                            tool_calls: None,
                                        },
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                };
                                let json = serde_json::to_string(&chunk).unwrap_or_default();
                                yield Ok(Event::default().data(json));
                            }
                            ToolParserEvent::ArgumentsDelta { index, text } if !text.is_empty() => {
                                let tc_delta = ToolCallDelta {
                                    index,
                                    id: None,
                                    call_type: None,
                                    function: Some(ToolCallFunctionDelta {
                                        name: None,
                                        arguments: Some(text),
                                    }),
                                };
                                let chunk = ChatCompletionChunk {
                                    id: request_id.clone(),
                                    object: "chat.completion.chunk".into(),
                                    created,
                                    model: model_name.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: ChunkDelta {
                                            role: None,
                                            content: None,
                                            tool_calls: Some(vec![tc_delta]),
                                        },
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                };
                                let json = serde_json::to_string(&chunk).unwrap_or_default();
                                yield Ok(Event::default().data(json));
                            }
                            _ => {}
                        }
                    }

                    // Determine finish_reason based on whether tool calls were detected
                    let finish_reason = if parser.has_tool_calls() {
                        "tool_calls"
                    } else {
                        "stop"
                    };

                    let final_chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: None,
                                tool_calls: None,
                            },
                            finish_reason: Some(finish_reason.into()),
                        }],
                        usage: Some(UsageStats {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens: prompt_tokens + completion_tokens,
                            prompt_tokens_details: None,
                        }),
                    };
                    let json = serde_json::to_string(&final_chunk).unwrap_or_default();
                    yield Ok(Event::default().data(json));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                GenerationEvent::Error(msg) => {
                    tracing::error!(error = %msg, "Generation error during streaming");
                    let error_chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: None,
                                tool_calls: None,
                            },
                            finish_reason: Some("error".into()),
                        }],
                        usage: None,
                    };
                    let json = serde_json::to_string(&error_chunk).unwrap_or_default();
                    yield Ok(Event::default().data(json));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            }
        }

        // Channel closed unexpectedly
        tracing::warn!("Generation channel closed unexpectedly");
        let error_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".into(),
            created,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                },
                finish_reason: Some("error".into()),
            }],
            usage: None,
        };
        let json = serde_json::to_string(&error_chunk).unwrap_or_default();
        yield Ok(Event::default().data(json));
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text(""),
    )
}

/// Helper to get the current unix timestamp.
pub fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;

    /// Helper to collect the SSE response body as a string.
    async fn collect_sse_body(
        events: Vec<GenerationEvent>,
    ) -> String {
        let (tx, rx) = mpsc::channel(32);

        // Send all events
        for event in events {
            tx.send(event).await.unwrap();
        }
        drop(tx);

        let sse = generation_events_to_sse(
            rx,
            "chatcmpl-test123".into(),
            "test-model".into(),
            1700000000,
        );

        let response = sse.into_response();
        let body = response.into_body();
        let bytes = body.collect().await.unwrap().to_bytes();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn test_sse_happy_path_tokens_then_done() {
        let body = collect_sse_body(vec![
            GenerationEvent::Token("Hello".into()),
            GenerationEvent::Token(" world".into()),
            GenerationEvent::Done {
                prompt_tokens: 5,
                completion_tokens: 2,
                stats: crate::inference::engine::GenerationStats {
                    prompt_tokens: 5,
                    generated_tokens: 2,
                    prefill_time_secs: 0.0,
                    decode_time_secs: 0.0,
                    total_time_secs: 0.0,
                    cached_tokens: 0,
                    time_to_first_token_ms: 0.0,
                    gpu_sync_count: 0,
                    gpu_dispatch_count: 0,
                },
            },
        ])
        .await;

        // Should contain the role chunk, two token chunks, a stop chunk, and [DONE]
        assert!(body.contains(r#""role":"assistant""#), "Missing role chunk");
        assert!(body.contains(r#""content":"Hello""#), "Missing Hello token");
        assert!(body.contains(r#""content":" world""#), "Missing world token");
        assert!(
            body.contains(r#""finish_reason":"stop""#),
            "Missing stop finish_reason"
        );
        assert!(body.contains("data: [DONE]"), "Missing [DONE] terminator");

        // Verify object type
        assert!(
            body.contains(r#""object":"chat.completion.chunk""#),
            "Missing chunk object type"
        );

        // Verify request ID consistency
        assert!(
            body.contains(r#""id":"chatcmpl-test123""#),
            "Missing consistent request ID"
        );
    }

    #[tokio::test]
    async fn test_sse_error_mid_stream() {
        let body = collect_sse_body(vec![
            GenerationEvent::Token("partial".into()),
            GenerationEvent::Error("Metal command buffer error".into()),
        ])
        .await;

        assert!(
            body.contains(r#""content":"partial""#),
            "Missing partial token"
        );
        assert!(
            body.contains(r#""finish_reason":"error""#),
            "Missing error finish_reason"
        );
        assert!(body.contains("data: [DONE]"), "Missing [DONE] after error");
    }

    #[tokio::test]
    async fn test_sse_channel_closed_unexpectedly() {
        let (tx, rx) = mpsc::channel(32);
        tx.send(GenerationEvent::Token("partial".into()))
            .await
            .unwrap();
        // Drop sender without sending Done or Error
        drop(tx);

        let sse = generation_events_to_sse(
            rx,
            "chatcmpl-dropped".into(),
            "test-model".into(),
            1700000000,
        );

        let response = sse.into_response();
        let body = response.into_body();
        let bytes = body.collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            body_str.contains(r#""finish_reason":"error""#),
            "Missing error chunk on unexpected close"
        );
        assert!(
            body_str.contains("data: [DONE]"),
            "Missing [DONE] on unexpected close"
        );
    }

    #[tokio::test]
    async fn test_sse_empty_generation() {
        let body = collect_sse_body(vec![GenerationEvent::Done {
            prompt_tokens: 10,
            completion_tokens: 0,
            stats: crate::inference::engine::GenerationStats {
                prompt_tokens: 10,
                generated_tokens: 0,
                prefill_time_secs: 0.0,
                decode_time_secs: 0.0,
                total_time_secs: 0.0,
                cached_tokens: 0,
                time_to_first_token_ms: 0.0,
                gpu_sync_count: 0,
                gpu_dispatch_count: 0,
            },
        }])
        .await;

        // Should have role chunk, stop chunk, and [DONE]
        assert!(body.contains(r#""role":"assistant""#));
        assert!(body.contains(r#""finish_reason":"stop""#));
        assert!(body.contains("data: [DONE]"));
        // Should NOT have any content tokens
        assert!(!body.contains(r#""content":"#));
    }

    #[tokio::test]
    async fn test_sse_usage_in_final_chunk() {
        let body = collect_sse_body(vec![GenerationEvent::Done {
            prompt_tokens: 10,
            completion_tokens: 20,
            stats: crate::inference::engine::GenerationStats {
                prompt_tokens: 10,
                generated_tokens: 20,
                prefill_time_secs: 0.0,
                decode_time_secs: 0.0,
                total_time_secs: 0.0,
                cached_tokens: 0,
                time_to_first_token_ms: 0.0,
                gpu_sync_count: 0,
                gpu_dispatch_count: 0,
            },
        }])
        .await;

        assert!(body.contains(r#""prompt_tokens":10"#));
        assert!(body.contains(r#""completion_tokens":20"#));
        assert!(body.contains(r#""total_tokens":30"#));
    }

    #[test]
    fn test_unix_timestamp_is_reasonable() {
        let ts = unix_timestamp();
        // Should be after 2020-01-01 and before 2100-01-01
        assert!(ts > 1577836800, "Timestamp too small: {}", ts);
        assert!(ts < 4102444800, "Timestamp too large: {}", ts);
    }
}
