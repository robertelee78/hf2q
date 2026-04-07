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

use super::schema::{ChatCompletionChunk, ChunkChoice, ChunkDelta, UsageStats};

/// Events sent from the blocking generation thread to the async SSE stream.
#[derive(Debug)]
pub enum GenerationEvent {
    /// A generated token fragment.
    Token(String),
    /// Generation completed successfully.
    Done {
        prompt_tokens: usize,
        completion_tokens: usize,
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
                            },
                            finish_reason: Some("stop".into()),
                        }],
                        usage: Some(UsageStats {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens: prompt_tokens + completion_tokens,
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
