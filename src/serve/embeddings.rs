//! Embeddings endpoint handler for the OpenAI-compatible API.
//!
//! Implements `POST /v1/embeddings` which runs a prefill-only forward pass
//! through the model and mean-pools the final hidden states to produce a
//! fixed-size embedding vector.
//!
//! Key design: embeddings bypass the generation queue entirely, using a
//! separate `tokio::Semaphore` so that embedding requests never block
//! behind active text generation (and vice versa).

use std::sync::Arc;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use tokio::sync::Semaphore;
use tracing::{error, warn};

use super::schema::*;
use super::AppState;

// ---------------------------------------------------------------------------
// Embedding lane semaphore
// ---------------------------------------------------------------------------

/// Semaphore controlling concurrent embedding requests.
///
/// This is separate from the generation queue, ensuring embeddings never
/// contend with text generation for queue slots.
#[derive(Debug)]
pub struct EmbeddingLane {
    semaphore: Arc<Semaphore>,
}

impl EmbeddingLane {
    /// Create a new embedding lane with the given concurrency limit.
    pub fn new(concurrency: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(concurrency)),
        }
    }

    /// Acquire a permit (blocks if at limit, never rejects).
    pub async fn acquire(&self) -> tokio::sync::OwnedSemaphorePermit {
        self.semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("Embedding semaphore closed unexpectedly")
    }
}

// ---------------------------------------------------------------------------
// POST /v1/embeddings handler
// ---------------------------------------------------------------------------

/// Handle a POST /v1/embeddings request.
pub async fn embeddings(
    State(state): State<AppState>,
    body: Result<Json<EmbeddingRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<impl IntoResponse, ApiError> {
    // Parse request body
    let Json(req) = body.map_err(|rejection| {
        warn!(error = %rejection, "Invalid JSON in embedding request");
        ApiError::invalid_request(
            format!("Invalid request body: {}", rejection),
            None,
        )
    })?;

    // Validate model name
    if !req.model.is_empty() && req.model != state.model_name {
        return Err(ApiError::model_not_found(&req.model));
    }

    // Parse input into a list of strings
    let inputs = req.input.into_vec();
    if inputs.is_empty() {
        return Err(ApiError::invalid_request(
            "Input must be a non-empty string or array of strings",
            Some("input".into()),
        ));
    }

    // Validate no empty strings
    for (i, input) in inputs.iter().enumerate() {
        if input.is_empty() {
            return Err(ApiError::invalid_request(
                format!("Input string at index {} is empty", i),
                Some("input".into()),
            ));
        }
    }

    // Acquire embedding lane permit (blocks if at capacity, never rejects)
    let _permit = state.embedding_lane.acquire().await;

    // Run the embedding forward pass in a blocking thread.
    // This is independent of the generation queue.
    let engine = state.engine.clone();
    let model_name = state.model_name.clone();
    let max_seq_len = state.max_seq_len;

    let handle = tokio::task::spawn_blocking(move || {
        compute_embeddings(engine, &inputs, max_seq_len)
    });

    let result = handle.await.map_err(|e| {
        error!(error = %e, "Embedding task panicked");
        ApiError::internal_error()
    })?;

    let (embeddings_data, total_prompt_tokens) = result?;

    let response = EmbeddingResponse {
        object: "list".to_string(),
        data: embeddings_data,
        model: model_name,
        usage: EmbeddingUsage {
            prompt_tokens: total_prompt_tokens,
            total_tokens: total_prompt_tokens,
        },
    };

    Ok(Json(response))
}

/// Compute embeddings for a list of input strings.
///
/// Runs synchronously inside `spawn_blocking`. Each input is processed
/// sequentially since they share the model's KV cache. Returns the
/// embedding objects and total token count.
fn compute_embeddings(
    engine: Arc<std::sync::Mutex<crate::inference::engine::InferenceEngine>>,
    inputs: &[String],
    max_seq_len: usize,
) -> Result<(Vec<EmbeddingObject>, usize), ApiError> {
    let mut engine_guard = engine.lock().unwrap();
    let mut data = Vec::with_capacity(inputs.len());
    let mut total_tokens = 0usize;

    for (index, input) in inputs.iter().enumerate() {
        // Quick token count check before running the forward pass
        let token_count = {
            // embed_text handles tokenization internally, but we need to
            // validate context length. We'll do a rough check: the engine
            // will fail if it exceeds the limit, but we can pre-check.
            // Actually, embed_text does its own tokenization, so let it
            // handle errors naturally.
            0usize // placeholder -- actual count comes from embed_text
        };
        let _ = token_count; // suppress unused warning

        let (embedding, token_count) = engine_guard.embed_text(input).map_err(|e| {
            let msg = e.to_string();
            if msg.contains("context") || msg.contains("too many tokens") {
                ApiError::context_length_exceeded(max_seq_len, 0)
            } else {
                ApiError::generation_error(format!("Embedding failed for input {}: {}", index, msg))
            }
        })?;

        total_tokens += token_count;

        data.push(EmbeddingObject {
            object: "embedding".to_string(),
            embedding,
            index,
        });
    }

    Ok((data, total_tokens))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_lane_creation() {
        let lane = EmbeddingLane::new(4);
        // Should have 4 permits
        assert_eq!(lane.semaphore.available_permits(), 4);
    }

    #[tokio::test]
    async fn test_embedding_lane_acquire_and_release() {
        let lane = EmbeddingLane::new(2);

        let permit1 = lane.acquire().await;
        assert_eq!(lane.semaphore.available_permits(), 1);

        let permit2 = lane.acquire().await;
        assert_eq!(lane.semaphore.available_permits(), 0);

        drop(permit1);
        assert_eq!(lane.semaphore.available_permits(), 1);

        drop(permit2);
        assert_eq!(lane.semaphore.available_permits(), 2);
    }

    #[test]
    fn test_embedding_request_single_string_deserialize() {
        let json = r#"{"model": "test", "input": "hello world"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        let inputs = req.input.into_vec();
        assert_eq!(inputs, vec!["hello world"]);
    }

    #[test]
    fn test_embedding_request_array_deserialize() {
        let json = r#"{"model": "test", "input": ["hello", "world"]}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        let inputs = req.input.into_vec();
        assert_eq!(inputs, vec!["hello", "world"]);
    }

    #[test]
    fn test_embedding_response_serialization() {
        let resp = EmbeddingResponse {
            object: "list".to_string(),
            data: vec![EmbeddingObject {
                object: "embedding".to_string(),
                embedding: vec![0.1, 0.2, 0.3],
                index: 0,
            }],
            model: "test-model".to_string(),
            usage: EmbeddingUsage {
                prompt_tokens: 5,
                total_tokens: 5,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["object"], "embedding");
        assert_eq!(json["data"][0]["index"], 0);
        // f32 precision: check the array length and approximate values
        let emb = json["data"][0]["embedding"].as_array().unwrap();
        assert_eq!(emb.len(), 3);
        assert!((emb[0].as_f64().unwrap() - 0.1).abs() < 1e-5);
        assert!((emb[1].as_f64().unwrap() - 0.2).abs() < 1e-5);
        assert!((emb[2].as_f64().unwrap() - 0.3).abs() < 1e-5);
        assert_eq!(json["model"], "test-model");
        assert_eq!(json["usage"]["prompt_tokens"], 5);
        assert_eq!(json["usage"]["total_tokens"], 5);
    }

    #[test]
    fn test_mean_pooling_and_l2_norm() {
        // Simulate what embed_text does: mean pool and L2 normalize
        let hidden_size = 3;
        let seq_len = 2;
        // Hidden states: [[1, 2, 3], [3, 4, 5]]
        let hidden_states = vec![1.0f32, 2.0, 3.0, 3.0, 4.0, 5.0];

        // Mean pool
        let mut pooled = vec![0.0f32; hidden_size];
        for pos in 0..seq_len {
            let offset = pos * hidden_size;
            for dim in 0..hidden_size {
                pooled[dim] += hidden_states[offset + dim];
            }
        }
        let inv_len = 1.0 / seq_len as f32;
        for v in pooled.iter_mut() {
            *v *= inv_len;
        }
        assert_eq!(pooled, vec![2.0, 3.0, 4.0]);

        // L2 normalize
        let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        let inv_norm = 1.0 / norm;
        for v in pooled.iter_mut() {
            *v *= inv_norm;
        }

        // Verify unit vector
        let l2: f32 = pooled.iter().map(|x| x * x).sum();
        assert!((l2 - 1.0).abs() < 1e-5, "L2 norm should be ~1.0, got {}", l2);
    }
}
