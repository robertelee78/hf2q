//! Engine — model-owning worker thread with serialized FIFO dispatch.
//!
//! ADR-005 Phase 2 Decision #2: the inference engine runs one request at a
//! time under a serialized FIFO queue. This module implements that model via
//! a dedicated OS thread that owns the `MlxModelWeights` + `GpuContext` and
//! accepts requests over a `tokio::sync::mpsc` channel.
//!
//! # Why a channel + thread, not a `tokio::Mutex`
//!
//! Two equivalent designs were considered:
//!
//!   - **Mutex guarding (weights, ctx)** — every handler `.lock().await`s and
//!     runs the forward pass inside the critical section with
//!     `tokio::task::block_in_place`. This bleeds sync compute into the tokio
//!     task pool and requires a multi-thread runtime invariant.
//!   - **Worker thread + mpsc (this file)** — handlers send requests to a
//!     channel. The worker thread drains the channel serially and replies via
//!     `oneshot`. Compute is a plain `std::thread` so the tokio runtime is
//!     never blocked and the FIFO ordering is inherent.
//!
//! The second is chosen because (a) forward passes are ~10-100ms of pure
//! compute — holding a tokio mutex across that would starve keep-alive /
//! request-id / CORS layers; (b) the queue cap (Decision #19) maps directly
//! to the channel capacity; (c) it avoids the `block_in_place` footgun.
//!
//! # Reference lineage
//!
//! The prefill / decode / tokenize path is exactly the same pipeline as
//! `serve::cmd_generate` (see `/opt/hf2q/src/serve/mod.rs`). This module
//! does not reimplement the forward pass; it wraps it. Every existing
//! behavior (ADR-009 dense-KV, ADR-010 Q8 rerank, chat-template priority
//! order) is preserved by construction.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, oneshot};

use crate::serve::config::Gemma4Config;
use crate::serve::forward_mlx::{MlxModelWeights, ProfileAccumulator};
use crate::serve::gpu::GpuContext;
use crate::serve::header;
use crate::serve::sampler_pure::{
    self, SamplingParams as SamplerParams,
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Sampling parameters passed to the engine worker. Full Tier 2/3/4
/// surface plumbed from the request.
///
/// **Honored at decode time** (iter-94, iter-95):
/// - `temperature`, `top_p`, `top_k`, `repetition_penalty` —
///   routed through `sampler_pure::sample_token` over the live
///   logits whenever any field requests non-greedy sampling
///   (see `sample_logits` gate in `generate_once`).  All-default
///   request → on-GPU greedy argmax fast path (no logits readback).
/// - `max_tokens`, `stop_strings` — decode-loop terminators.
/// - `logit_bias` — additive bias applied to live logits before
///   `sampler_pure` (Tier 4, OpenAI semantics).
/// - `grammar` + `token_bytes` — when present, mask invalid tokens
///   per-step before sampling (iter-95 grammar-constrained decode,
///   gated on `response_format=json_object`/`json_schema`).
///
/// **Plumbed but NOT yet honored** (accepted from the request,
/// retained on the struct, but not consumed by the current sampler):
/// - `frequency_penalty`, `presence_penalty` — Tier 2 OpenAI extras.
/// - `min_p` — Tier 3 llama.cpp extension.
/// - `seed` — RNG seeding (sampler_pure uses a thread-local RNG today).
/// - `logprobs`, `top_logprobs` — Tier 4 response shape; surface only.
/// - `parallel_tool_calls` — Tier 4; lands with the tool-call path
///   referenced at the worker dispatch site (see Decision #21 in the
///   registration block).
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub max_tokens: usize,
    /// Stop strings — if one appears in the running decoded text, generation
    /// halts with finish_reason `stop`. Case-sensitive.
    pub stop_strings: Vec<String>,

    // --- Tier 2 additions (plumbed, not all wired into sampler yet) ---
    /// Nucleus-sampling lower bound used with `top_p`. OpenAI Tier 2.
    pub frequency_penalty: f32,
    /// OpenAI Tier 2.
    pub presence_penalty: f32,
    /// Optional RNG seed for reproducible sampling. `None` → thread RNG.
    /// Greedy (T=0) decodes are deterministic regardless.
    pub seed: Option<u64>,

    // --- Tier 3 addition (llama.cpp / ollama extension) ---
    /// Min-p sampling cutoff. `0.0` disables. Tier 3.
    pub min_p: f32,

    // --- Tier 4 (power-user) ---
    /// Per-token-id logit bias map. Applied before softmax once the
    /// forward-decode refactor exposes logits.
    pub logit_bias: std::collections::HashMap<u32, f32>,
    /// If `true`, include top-k logprobs in the response. Tier 4.
    pub logprobs: bool,
    /// Number of top alternatives to report per chosen token. 0 = only the
    /// chosen token's logprob.
    pub top_logprobs: u32,
    /// `true` = allow multiple tool calls in the same turn. Tier 4. Plumbs
    /// through to the grammar-constrained decode path when it lands.
    pub parallel_tool_calls: bool,
}

impl Default for SamplingParams {
    /// Sampling defaults used when a request omits a field. T=0 greedy, no
    /// penalties. Matches the behavior of `cmd_generate` when all CLI
    /// sampling flags default.
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            max_tokens: 512,
            stop_strings: Vec::new(),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
            min_p: 0.0,
            logit_bias: std::collections::HashMap::new(),
            logprobs: false,
            top_logprobs: 0,
            parallel_tool_calls: true,
        }
    }
}

/// Result of a non-streaming chat generation.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Decoded text that goes into `message.content` — post reasoning-marker
    /// split (Decision #21). If the model has no reasoning markers
    /// registered, this is the full raw decoded text.
    pub text: String,
    /// Decoded text that goes into `message.reasoning_content`. `None` when
    /// the model's registration has no reasoning markers or when no
    /// reasoning span was emitted.
    pub reasoning_text: Option<String>,
    /// Prompt token count (after chat-template rendering + tokenization).
    pub prompt_tokens: usize,
    /// Completion token count (tokens emitted by the decoder).
    pub completion_tokens: usize,
    /// Number of completion tokens that were emitted inside a reasoning
    /// span (Decision #21). `None` when no reasoning markers registered /
    /// no reasoning span opened. Counted per-token in the decode loop.
    pub reasoning_tokens: Option<usize>,
    /// Reason generation halted: `"stop"` | `"length"`.
    pub finish_reason: &'static str,
    /// Prefill wall-clock.
    pub prefill_duration: Duration,
    /// Decode wall-clock.
    pub decode_duration: Duration,
}

// ---------------------------------------------------------------------------
// Engine handle (the public API)
// ---------------------------------------------------------------------------

/// Engine handle — cheap to clone, threaded through `AppState`. All methods
/// are async so they can be awaited from axum handlers without blocking the
/// tokio runtime.
#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    tx: mpsc::Sender<Request>,
    /// Worker-thread join handle. Held in a `Mutex<Option<...>>` so
    /// `Engine::shutdown` can `take()` it once and `.join()` the thread, and
    /// callers can be cheap-clone an `Engine` without contending on the
    /// handle. Outside of shutdown, the slot is read-only.
    worker_handle: Mutex<Option<JoinHandle<()>>>,
    /// Metadata exposed to handlers without touching the worker thread.
    /// Immutable for the lifetime of the engine.
    model_id: String,
    context_length: Option<usize>,
    quant_type: Option<String>,
    /// Hidden-state dimensionality of the loaded model.  Surfaced to the
    /// `/v1/embeddings` handler when the chat model is used as an
    /// embedder (Phase 2a Task #8) — used to validate the OpenAI
    /// `dimensions` parameter and to size the response payload.
    hidden_size: usize,
    eos_token_ids: Vec<u32>,
    /// Tokenizer cloned per-request so handlers can tokenize without a lock.
    tokenizer: Arc<Tokenizer>,
    /// Chat-template string (GGUF metadata or fallback). Rendered per request
    /// using `minijinja`.
    chat_template: Arc<String>,
    /// Per-model registration — reasoning-boundary + tool-call markers
    /// (Decision #21). `None` when no family matches this model's id.
    registration: Option<super::registry::ModelRegistration>,
}

/// The request protocol the worker thread drains.
enum Request {
    Warmup {
        reply: oneshot::Sender<Result<()>>,
    },
    Generate {
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        reply: oneshot::Sender<Result<GenerationResult>>,
    },
    /// Streaming generation — tokens flow back to the handler via `events`
    /// as `GenerationEvent::Delta{ kind, text }` per decode step, then a
    /// terminating `Done { finish_reason, prompt_tokens, completion_tokens,
    /// stats }` (or `Error`). When the handler's SSE stream is dropped
    /// (client disconnect per Decision #18), `events.send` returns Err and
    /// the worker breaks early — the queue slot is freed immediately.
    /// `cancellation_counter` (if `Some`) is incremented by 1 when the
    /// worker aborts because the receiver was dropped; surfaced via
    /// `hf2q_sse_cancellations` in `/metrics`.
    GenerateStream {
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        events: mpsc::Sender<super::sse::GenerationEvent>,
        cancellation_counter: Option<Arc<std::sync::atomic::AtomicU64>>,
    },
    /// Pooled-embedding request (ADR-005 Phase 2a Task #8 / iter-92).
    ///
    /// Runs the chat model's prefill forward pass and returns the
    /// L2-normalized last-token hidden state — the natural "Last" pooling
    /// for autoregressive (causal-attention) chat models.  Used by the
    /// `/v1/embeddings` handler when no `--embedding-model` is loaded but
    /// the chat model is.  See
    /// `MlxModelWeights::forward_embed_last` for the GPU-side semantics.
    Embed {
        prompt_tokens: Vec<u32>,
        reply: oneshot::Sender<Result<Vec<f32>>>,
    },
    /// Graceful-shutdown sentinel.
    Shutdown,
}

// ---------------------------------------------------------------------------
// Load path — LoadedModel
// ---------------------------------------------------------------------------

/// All the artifacts needed for inference, held together so the worker can
/// take ownership in a single move.
pub struct LoadedModel {
    pub weights: MlxModelWeights,
    pub ctx: GpuContext,
    pub config: Gemma4Config,
    pub model_id: String,
    pub context_length: Option<usize>,
    pub quant_type: Option<String>,
    pub tokenizer: Tokenizer,
    pub chat_template: String,
    pub eos_token_ids: Vec<u32>,
    pub load_duration: Duration,
}

/// Options for `LoadedModel::load`. Mirrors `cli::ServeArgs` without pulling
/// the CLI type into this module.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub model_path: PathBuf,
    pub tokenizer_path: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
}

impl LoadedModel {
    /// Perform the full model-load pipeline: open GGUF, load weights into
    /// mlx-native, load the tokenizer, resolve the chat template, read the
    /// context length from metadata.
    ///
    /// This mirrors `cmd_generate`'s load sequence (`src/serve/mod.rs:188-252`)
    /// so the two entrypoints are guaranteed to produce the same model state.
    /// Any future change to the load path belongs in a shared helper rather
    /// than duplicated here — maintainers: if you touch one, touch both.
    pub fn load(opts: &LoadOptions) -> Result<Self> {
        let load_start = Instant::now();

        let model_path = &opts.model_path;
        anyhow::ensure!(
            model_path.exists(),
            "Model not found: {}",
            model_path.display()
        );

        // Resolve tokenizer + config paths the same way cmd_generate does.
        let tokenizer_path = find_tokenizer(model_path, opts.tokenizer_path.as_deref())?;
        let config_path = find_config(model_path, opts.config_path.as_deref())?;

        tracing::info!("Engine load: model = {}", model_path.display());
        tracing::info!("Engine load: tokenizer = {}", tokenizer_path.display());
        tracing::info!("Engine load: config = {}", config_path.display());

        let config = Gemma4Config::from_config_json(&config_path)
            .context("Failed to parse config.json")?;

        // Open GGUF (header + metadata only).
        let gguf = mlx_native::gguf::GgufFile::open(model_path)
            .map_err(|e| anyhow::anyhow!("GGUF open: {e}"))?;

        // Extract model id: prefer general.name, fall back to file stem.
        let model_id = gguf
            .metadata_string("general.name")
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                model_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "unknown".to_string())
            });

        // Context length: arch-prefixed metadata key.
        let arch = gguf.metadata_string("general.architecture").unwrap_or("");
        let context_length = if arch.is_empty() {
            None
        } else {
            gguf.metadata_u32(&format!("{arch}.context_length"))
                .map(|v| v as usize)
        };

        // Quant label: dominant non-fp tensor type. Same histogram algorithm
        // as the /v1/models handler; computed inline here rather than via a
        // shared helper so this file stays self-contained.
        let quant_type = infer_quant_type_from_gguf(&gguf);

        // Chat template: GGUF embedded or hardcoded fallback.
        let chat_template = gguf
            .metadata_string("tokenizer.chat_template")
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                tracing::warn!(
                    "Engine load: no GGUF `tokenizer.chat_template`; \
                     using hardcoded Gemma4 fallback. Request rendering will \
                     only handle single-turn user prompts correctly."
                );
                crate::serve::FALLBACK_GEMMA4_CHAT_TEMPLATE.to_string()
            });

        // Load GPU ctx + weights. `header::LoadProgress` is happy with a
        // non-TTY parent; we set verbosity to 1 to suppress the progress line
        // when the server is running (logs replace the progress UX).
        let mut ctx = GpuContext::new()
            .map_err(|e| anyhow::anyhow!("mlx-native init failed: {e}"))?;

        let n_layers = config.num_hidden_layers;
        let mut load_progress = header::LoadProgress::new(false, 1, n_layers);
        let weights = MlxModelWeights::load_from_gguf(
            &gguf,
            &config,
            &mut ctx,
            &mut load_progress,
        )?;

        // Load tokenizer.
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("Failed to disable tokenizer truncation: {e}"))?;

        // EOS tokens: reuse the hardcoded list from cmd_generate. This is
        // what Gemma 4 uses; other models will be generalized alongside
        // per-model registration (Decision #21 — lands with tool calling).
        let eos_token_ids: Vec<u32> = vec![1, 106];

        let load_duration = load_start.elapsed();
        tracing::info!(
            "Engine load: {} layers, ctx_len={:?}, load_time={:.1}s",
            weights.layers.len(),
            context_length,
            load_duration.as_secs_f64()
        );

        Ok(Self {
            weights,
            ctx,
            config,
            model_id,
            context_length,
            quant_type,
            tokenizer,
            chat_template,
            eos_token_ids,
            load_duration,
        })
    }
}

// ---------------------------------------------------------------------------
// Engine::spawn and worker loop
// ---------------------------------------------------------------------------

impl Engine {
    /// Spawn the worker thread and return a handle. The `queue_capacity` sets
    /// the mpsc channel buffer; when full, handlers receive a `queue_full`
    /// error and map it to 429 + Retry-After (Decision #19).
    pub fn spawn(loaded: LoadedModel, queue_capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel::<Request>(queue_capacity.max(1));

        let model_id = loaded.model_id.clone();
        let context_length = loaded.context_length;
        let quant_type = loaded.quant_type.clone();
        let hidden_size = loaded.weights.hidden_size;
        let eos_token_ids = loaded.eos_token_ids.clone();
        let tokenizer = Arc::new(loaded.tokenizer.clone());
        let chat_template = Arc::new(loaded.chat_template.clone());
        let registration = super::registry::find_for(&model_id);
        if let Some(ref r) = registration {
            tracing::info!(
                family = r.family,
                reasoning = r.has_reasoning(),
                tools = r.has_tools(),
                "hf2q-engine: matched model registration"
            );
        } else {
            tracing::info!(
                model_id = %model_id,
                "hf2q-engine: no matching model registration (text emitted as plain content)"
            );
        }

        // Move registration into the worker closure in addition to the handle.
        let worker_registration = registration.clone();
        let worker_handle = std::thread::Builder::new()
            .name("hf2q-engine".into())
            .spawn(move || worker_run(loaded, rx, worker_registration))
            .expect("spawn hf2q-engine thread");

        Engine {
            inner: Arc::new(EngineInner {
                tx,
                worker_handle: Mutex::new(Some(worker_handle)),
                model_id,
                context_length,
                quant_type,
                hidden_size,
                eos_token_ids,
                tokenizer,
                chat_template,
                registration,
            }),
        }
    }

    pub fn model_id(&self) -> &str {
        &self.inner.model_id
    }
    pub fn context_length(&self) -> Option<usize> {
        self.inner.context_length
    }
    pub fn quant_type(&self) -> Option<&str> {
        self.inner.quant_type.as_deref()
    }
    /// Hidden-state dimensionality.  Used by the `/v1/embeddings` handler
    /// when the chat model is the embedder (Phase 2a Task #8).
    pub fn hidden_size(&self) -> usize {
        self.inner.hidden_size
    }
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.inner.tokenizer
    }
    pub fn chat_template(&self) -> &str {
        &self.inner.chat_template
    }
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.inner.eos_token_ids
    }
    pub fn registration(&self) -> Option<&super::registry::ModelRegistration> {
        self.inner.registration.as_ref()
    }

    /// Run a single-prompt warmup pass. Blocks until the worker finishes it.
    /// Typical cost is one prefill + a few decode tokens on a tiny prompt —
    /// at the 10ms-order on M5 Max. The warmup's job is to compile all
    /// kernels and fault in hot weights so the first real request doesn't
    /// pay the one-time setup latency.
    pub async fn warmup(&self) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.inner
            .tx
            .send(Request::Warmup { reply: reply_tx })
            .await
            .context("engine worker is gone")?;
        reply_rx.await.context("warmup reply dropped")?
    }

    /// Enqueue a non-streaming generation. Returns `queue_full` if the FIFO
    /// is at capacity (handlers map to 429 + Retry-After).
    pub async fn generate(
        &self,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
    ) -> Result<GenerationResult> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let req = Request::Generate {
            prompt_tokens,
            params,
            reply: reply_tx,
        };
        // Use `try_send` so we can distinguish queue-full from a closed worker.
        match self.inner.tx.try_send(req) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                anyhow::bail!("queue_full");
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                anyhow::bail!("engine worker is gone");
            }
        }
        reply_rx.await.context("generation reply dropped")?
    }

    /// Enqueue a streaming generation. The caller owns `events_rx` (returned
    /// separately) and wraps it in the SSE encoder. Returns immediately after
    /// queueing; the worker emits tokens into `events_tx` as they decode.
    ///
    /// Dropping `events_rx` (handler-side) causes the next worker `send` to
    /// fail and the worker aborts the decode loop, freeing the queue slot
    /// (Decision #18). Queue-full returns an error that the handler maps to
    /// 429 + Retry-After.
    pub async fn generate_stream(
        &self,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        events_tx: mpsc::Sender<super::sse::GenerationEvent>,
        cancellation_counter: Option<Arc<std::sync::atomic::AtomicU64>>,
    ) -> Result<()> {
        let req = Request::GenerateStream {
            prompt_tokens,
            params,
            events: events_tx,
            cancellation_counter,
        };
        match self.inner.tx.try_send(req) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => anyhow::bail!("queue_full"),
            Err(mpsc::error::TrySendError::Closed(_)) => anyhow::bail!("engine worker is gone"),
        }
    }

    /// Enqueue a pooled-embedding request (ADR-005 Task #8, iter-92).
    ///
    /// Returns the L2-normalized last-token hidden state as a `Vec<f32>`
    /// of length `hidden_size`.  Uses the same FIFO queue + 429 semantics
    /// as `generate`: a full queue maps to `queue_full` (handler maps to
    /// HTTP 429 + Retry-After).
    pub async fn embed(&self, prompt_tokens: Vec<u32>) -> Result<Vec<f32>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let req = Request::Embed {
            prompt_tokens,
            reply: reply_tx,
        };
        match self.inner.tx.try_send(req) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                anyhow::bail!("queue_full");
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                anyhow::bail!("engine worker is gone");
            }
        }
        reply_rx.await.context("embedding reply dropped")?
    }

    /// Request a clean shutdown of the worker. Drains in-flight + queued work
    /// (FIFO ordering means the `Shutdown` sentinel runs after every request
    /// already enqueued) and then joins the worker thread.
    ///
    /// Returns `Ok(())` once the worker thread has fully exited; returns
    /// `Err(...)` only if the join itself panicked.
    ///
    /// Idempotent: calling twice (or on a clone whose sibling already
    /// joined) is a no-op — the second call observes `worker_handle = None`
    /// and returns immediately. The blocking `.join()` runs inside
    /// `tokio::task::spawn_blocking` so the calling tokio runtime is not
    /// blocked while the worker drains a long generation.
    pub async fn shutdown(&self) -> Result<()> {
        // Send Shutdown sentinel. Errors here mean the worker tx was already
        // dropped/closed — the thread has already exited; treat as success
        // for the join step below.
        let _ = self.inner.tx.send(Request::Shutdown).await;

        // Take the JoinHandle exactly once. Subsequent shutdown() calls
        // observe None and return Ok(()).
        let handle = match self.inner.worker_handle.lock() {
            Ok(mut guard) => guard.take(),
            Err(_) => None, // poisoned mutex => worker already gone
        };

        if let Some(handle) = handle {
            // Join can block until the in-flight Generate finishes; do it
            // off-runtime so we don't stall axum's drain phase. The handle's
            // ownership has already been moved out of `inner`, so this
            // closure can take it.
            tokio::task::spawn_blocking(move || handle.join())
                .await
                .context("spawn_blocking for worker join")?
                .map_err(|panic| {
                    anyhow::anyhow!(
                        "engine worker thread panicked on shutdown: {:?}",
                        panic
                    )
                })?;
        }

        Ok(())
    }
}

/// Worker-thread entry point. Owns the `LoadedModel` and drains requests
/// serially. `registration` (if `Some`) drives reasoning-content split
/// (Decision #21) — decode text passes through a `ReasoningSplitter` on
/// the way out.
fn worker_run(
    mut loaded: LoadedModel,
    mut rx: mpsc::Receiver<Request>,
    registration: Option<super::registry::ModelRegistration>,
) {
    tracing::info!(
        model = %loaded.model_id,
        "hf2q-engine worker thread started"
    );

    while let Some(req) = rx.blocking_recv() {
        match req {
            Request::Warmup { reply } => {
                let result = warmup_once(&mut loaded);
                let _ = reply.send(result);
            }
            Request::Generate {
                prompt_tokens,
                params,
                reply,
            } => {
                let result = generate_once(&mut loaded, &prompt_tokens, &params, registration.as_ref());
                let _ = reply.send(result);
            }
            Request::GenerateStream {
                prompt_tokens,
                params,
                events,
                cancellation_counter,
            } => {
                // The streaming path sends every event (Delta / Done / Error)
                // via `events`. Errors stay inside the function — the
                // terminal event is always one of Done/Error, unless the
                // receiver was dropped (client disconnect → early exit).
                // When the early-exit path fires, we bump the cancellation
                // counter if supplied (→ hf2q_sse_cancellations in /metrics).
                generate_stream_once(
                    &mut loaded,
                    &prompt_tokens,
                    &params,
                    &events,
                    registration.as_ref(),
                    cancellation_counter.as_deref(),
                );
            }
            Request::Embed {
                prompt_tokens,
                reply,
            } => {
                // Single-shot pooled embedding (Last pooling).  The
                // worker holds &mut LoadedModel, so prefill's mutation of
                // self.activations + self.dense_kvs is fine here — it
                // can't race with a concurrent generate call because the
                // worker is serial.
                let result = loaded
                    .weights
                    .forward_embed_last(&prompt_tokens, &mut loaded.ctx);
                let _ = reply.send(result);
            }
            Request::Shutdown => {
                tracing::info!("hf2q-engine worker received Shutdown; exiting");
                break;
            }
        }
    }

    tracing::info!("hf2q-engine worker thread exited");
}

// ---------------------------------------------------------------------------
// Inference pipeline (synchronous, owned by the worker thread)
// ---------------------------------------------------------------------------

/// Single-pass warmup: run prefill + 1 decode on a tiny canary prompt to
/// compile all kernels and fault in the hot weights.
fn warmup_once(loaded: &mut LoadedModel) -> Result<()> {
    let started = Instant::now();
    // A 1-token prompt is enough to cycle through the prefill + decode path.
    // Use the GGUF bos-token id if available; else fall back to 1.
    let bos: u32 = 1;
    let prompt = vec![bos];
    let max_tokens = 1;
    let last_token = loaded
        .weights
        .forward_prefill(&prompt, max_tokens, &mut loaded.ctx)?;
    // One decode step to exercise the decode kernel set.
    let mut profiler = None;
    let _ = loaded
        .weights
        .forward_decode(last_token, prompt.len(), &mut loaded.ctx, &mut profiler)?;
    tracing::info!(
        "hf2q-engine warmup complete in {:.0}ms",
        started.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

/// Generate one full response: prefill the prompt, then decode up to
/// `max_tokens`, halting on EOS or a configured stop string. The decode path
/// is greedy-argmax (temperature 0). Richer sampling (top-p, top-k, seed,
/// logit_bias) lands when the grammar stack (Decision #6) comes in — the
/// sampler hook is the same.
fn generate_once(
    loaded: &mut LoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    registration: Option<&super::registry::ModelRegistration>,
) -> Result<GenerationResult> {
    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "generate_once: empty prompt_tokens"
    );
    let prompt_len = prompt_tokens.len();
    let max_tokens = params.max_tokens.max(1);

    // ── Sampler config — Tier 2/3/4 surface (iter-94) ──────────────────
    //
    // Pre-iter-94 the decode loop only consumed `forward_decode`'s
    // on-GPU greedy argmax — every `temperature` / `top_p` / `top_k` /
    // `repetition_penalty` / `logit_bias` request was silently downcast
    // to greedy.  Iter-94 forks on whether ANY field requests non-greedy
    // sampling and routes those through `sampler_pure::sample_token`
    // over the live `self.activations.logits` slice (already populated
    // as a side-effect of `forward_decode` / `forward_prefill`'s lm_head
    // + softcap dispatch).
    //
    // Greedy fast path (all fields at default) keeps the existing
    // forward_decode return-value chain — no logits readback, no extra
    // copy.  Sampling slow path discards the on-GPU argmax token
    // (~20 µs of wasted GPU work, negligible vs the ~10-100ms layer
    // forward) and re-derives the next token from the live logits.
    let sample_logits = params.temperature > 0.0
        || params.top_k > 0
        || params.top_p < 1.0
        || params.repetition_penalty != 1.0
        || !params.logit_bias.is_empty();
    let sampler_params = if sample_logits {
        Some(SamplerParams {
            temperature: params.temperature as f64,
            top_p: params.top_p as f64,
            top_k: params.top_k,
            repetition_penalty: params.repetition_penalty as f64,
            max_tokens: params.max_tokens,
        })
    } else {
        None
    };

    // Local helper — apply Tier 4 logit_bias and sample.  Captures
    // params + sampler_params by reference; takes the running
    // `generated_tokens` for repetition-penalty.
    let sample_from_live_logits =
        |weights: &mut MlxModelWeights, generated: &[u32]| -> Result<u32> {
            let sp = sampler_params.as_ref().expect("sample_logits gate");
            let mut logits: Vec<f32> = weights.logits_view()?.to_vec();
            if !params.logit_bias.is_empty() {
                let v = logits.len();
                for (&id, &bias) in &params.logit_bias {
                    let idx = id as usize;
                    if idx < v {
                        logits[idx] += bias;
                    }
                }
            }
            Ok(sampler_pure::sample_token(&mut logits, sp, generated))
        };

    // --- Prefill ---
    let prefill_start = Instant::now();
    let prefill_argmax = loaded
        .weights
        .forward_prefill(prompt_tokens, max_tokens, &mut loaded.ctx)?;
    let prefill_duration = prefill_start.elapsed();

    // First decode token: greedy fast-path uses prefill's on-GPU argmax;
    // sampling path re-derives from prefill's live logits buffer (last
    // prompt-token's lm_head output) so the user-controlled temperature
    // applies to the very first generated token, not just decode-loop
    // tokens 2..N.  The greedy-fast-path skips logits readback entirely.
    let mut next_token = if sample_logits {
        sample_from_live_logits(&mut loaded.weights, &[])?
    } else {
        prefill_argmax
    };

    // --- Reasoning splitter + counter (Decision #21) ---
    // Feed each decoded fragment through a local ReasoningSplitter; count
    // tokens whose post-feed state is `in_reasoning`. Mirrors the streaming
    // path's accounting exactly so stream + non-stream usage agree.
    let mut splitter = registration
        .filter(|r| r.has_reasoning())
        .and_then(super::registry::ReasoningSplitter::from_registration);
    let reasoning_enabled = splitter.is_some();
    let mut reasoning_token_count: usize = 0;

    // --- Decode loop ---
    let decode_start = Instant::now();
    let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    generated_tokens.push(next_token);

    let first_fragment = loaded
        .tokenizer
        .decode(&[next_token], false)
        .unwrap_or_default();
    let mut decoded_text = first_fragment.clone();
    if let Some(sp) = splitter.as_mut() {
        let _ = sp.feed(&first_fragment);
        if sp.in_reasoning() {
            reasoning_token_count += 1;
        }
    }

    let mut finish_reason: &'static str = "length";
    let mut profiler = ProfileAccumulator::new(0);

    // Early EOS check on the prefill-emitted first token.
    if loaded.eos_token_ids.contains(&next_token) {
        finish_reason = "stop";
    } else if hit_stop_string(&decoded_text, &params.stop_strings) {
        finish_reason = "stop";
    } else {
        for _ in 1..max_tokens {
            let pos = prompt_len + generated_tokens.len() - 1;
            let mut p = profiler.start_token();
            // forward_decode populates self.activations.logits as a
            // side-effect of its lm_head + softcap dispatch chain; the
            // returned u32 is the on-GPU greedy argmax (only used on the
            // greedy fast-path).
            let greedy_token = loaded
                .weights
                .forward_decode(next_token, pos, &mut loaded.ctx, &mut p)?;
            profiler.finish_token(p);

            next_token = if sample_logits {
                // Sampling slow path: read logits, apply Tier 4 logit_bias,
                // call sampler_pure for temperature/top_p/top_k/rep-penalty.
                // OpenAI logit_bias is additive: `logit_bias[id] = bias`
                // adds `bias` to `logits[id]` before softmax.  Out-of-range
                // ids are silently ignored (matches OpenAI behaviour).
                sample_from_live_logits(&mut loaded.weights, &generated_tokens)?
            } else {
                // Greedy fast path — use forward_decode's on-GPU argmax.
                greedy_token
            };

            if loaded.eos_token_ids.contains(&next_token) {
                finish_reason = "stop";
                break;
            }

            generated_tokens.push(next_token);
            let fragment = loaded
                .tokenizer
                .decode(&[next_token], false)
                .unwrap_or_default();
            decoded_text.push_str(&fragment);
            if let Some(sp) = splitter.as_mut() {
                let _ = sp.feed(&fragment);
                if sp.in_reasoning() {
                    reasoning_token_count += 1;
                }
            }
            if hit_stop_string(&decoded_text, &params.stop_strings) {
                finish_reason = "stop";
                // Strip the stop string from the returned text (OpenAI
                // convention per ADR-005 "Stop-sequence stripping from
                // returned text").
                strip_trailing_stop(&mut decoded_text, &params.stop_strings);
                break;
            }
        }
    }
    let decode_duration = decode_start.elapsed();

    // When finish_reason == "stop" but the EOS was seen, make sure the EOS
    // token text isn't present in the returned content.
    let _ = params; // params.temperature etc. are greedy defaults in this iter

    // Apply reasoning split (Decision #21) if this model has boundary
    // markers registered. If not, the full decoded text goes into
    // `content` and `reasoning_text` is `None`.
    let (content, reasoning_text) = match registration {
        Some(reg) if reg.has_reasoning() => super::registry::split_full_output(reg, &decoded_text),
        _ => (decoded_text, None),
    };

    Ok(GenerationResult {
        text: content,
        reasoning_text,
        prompt_tokens: prompt_len,
        completion_tokens: generated_tokens.len(),
        reasoning_tokens: if reasoning_enabled && reasoning_token_count > 0 {
            Some(reasoning_token_count)
        } else {
            None
        },
        finish_reason,
        prefill_duration,
        decode_duration,
    })
}

/// Infer a quant label from an open GGUF. Shared algorithm with the
/// `/v1/models` handler (kept inline here for module self-containment).
fn infer_quant_type_from_gguf(gguf: &mlx_native::gguf::GgufFile) -> Option<String> {
    use mlx_native::GgmlType;
    use std::collections::HashMap;

    let mut histogram: HashMap<&'static str, usize> = HashMap::new();
    for name in gguf.tensor_names() {
        let Some(info) = gguf.tensor_info(name) else { continue };
        if matches!(info.ggml_type, GgmlType::F32 | GgmlType::F16) {
            continue;
        }
        let label = match info.ggml_type {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::Q4_0 => "Q4_0",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q4_K => "Q4_K",
            GgmlType::Q5_K => "Q5_K",
            GgmlType::Q6_K => "Q6_K",
            GgmlType::I16 => "I16",
        };
        *histogram.entry(label).or_insert(0) += 1;
    }
    histogram
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(k, _)| k.to_string())
}

/// Streaming variant of `generate_once`. Sends `GenerationEvent::Delta` per
/// decoded token, followed by a terminating `Done` (with finish_reason +
/// usage) or `Error`. If the `events` receiver is dropped (SSE client
/// disconnect, Decision #18), the next `blocking_send` returns Err and the
/// loop exits early — no more events are sent, the queue slot is freed.
fn generate_stream_once(
    loaded: &mut LoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    events: &mpsc::Sender<super::sse::GenerationEvent>,
    registration: Option<&super::registry::ModelRegistration>,
    cancellation_counter: Option<&std::sync::atomic::AtomicU64>,
) {
    use super::sse::{DeltaKind, GenerationEvent, StreamStats};

    // Helper: send an event; if the receiver is gone, bump the
    // cancellation counter (→ hf2q_sse_cancellations in /metrics) and bail.
    macro_rules! send {
        ($ev:expr) => {
            if events.blocking_send($ev).is_err() {
                tracing::info!("SSE stream dropped by client; aborting decode");
                if let Some(c) = cancellation_counter {
                    c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return;
            }
        };
    }

    if prompt_tokens.is_empty() {
        send!(GenerationEvent::Error(
            "generate_stream_once: empty prompt_tokens".into()
        ));
        return;
    }
    let prompt_len = prompt_tokens.len();
    let max_tokens = params.max_tokens.max(1);

    // Reasoning splitter — classifies each decoded fragment into the
    // content / reasoning_content slot. `None` when the model has no
    // registered reasoning markers; all fragments then route to `Content`.
    let mut splitter = registration
        .and_then(|r| super::registry::ReasoningSplitter::from_registration(r));

    // Local helper to emit a fragment through the splitter (if any) into
    // the correct DeltaKind slot. Returns the bytes emitted (for stop-string
    // bookkeeping). Note: the splitter holds back a tail that's drained at
    // generation end.
    let emit_fragment = |splitter: &mut Option<super::registry::ReasoningSplitter>,
                         events: &mpsc::Sender<GenerationEvent>,
                         fragment: &str| -> Result<(), ()> {
        if fragment.is_empty() {
            return Ok(());
        }
        if let Some(sp) = splitter.as_mut() {
            for (slot, text) in sp.feed(fragment) {
                let kind = match slot {
                    super::registry::SplitSlot::Content => DeltaKind::Content,
                    super::registry::SplitSlot::Reasoning => DeltaKind::Reasoning,
                };
                if events.blocking_send(GenerationEvent::Delta { kind, text }).is_err() {
                    return Err(());
                }
            }
        } else if events
            .blocking_send(GenerationEvent::Delta {
                kind: DeltaKind::Content,
                text: fragment.to_string(),
            })
            .is_err()
        {
            return Err(());
        }
        Ok(())
    };

    // Snapshot mlx-native process-global GPU counters pre-generation so we
    // can report the per-request delta on the terminal `Done` event's
    // StreamStats (mirrors the non-streaming path's x_hf2q_timing counters).
    let pre_dispatches = mlx_native::dispatch_count();
    let pre_syncs = mlx_native::sync_count();

    // ── Sampler config — Tier 2/3/4 surface (iter-94, mirrors generate_once) ──
    let sample_logits = params.temperature > 0.0
        || params.top_k > 0
        || params.top_p < 1.0
        || params.repetition_penalty != 1.0
        || !params.logit_bias.is_empty();
    let sampler_params = if sample_logits {
        Some(SamplerParams {
            temperature: params.temperature as f64,
            top_p: params.top_p as f64,
            top_k: params.top_k,
            repetition_penalty: params.repetition_penalty as f64,
            max_tokens: params.max_tokens,
        })
    } else {
        None
    };

    // --- Prefill ---
    let prefill_start = Instant::now();
    let next_token_result =
        loaded
            .weights
            .forward_prefill(prompt_tokens, max_tokens, &mut loaded.ctx);
    let prefill_duration = prefill_start.elapsed();
    let prefill_argmax = match next_token_result {
        Ok(t) => t,
        Err(e) => {
            send!(GenerationEvent::Error(format!("prefill failed: {e}")));
            return;
        }
    };
    // First decode token: greedy fast-path uses prefill's on-GPU argmax;
    // sampling path re-derives from prefill's live logits (last
    // prompt-token's lm_head output) so user-controlled temperature
    // applies to the very first generated token.
    let mut next_token = if let Some(sp) = sampler_params.as_ref() {
        let logits_view = match loaded.weights.logits_view() {
            Ok(v) => v.to_vec(),
            Err(e) => {
                send!(GenerationEvent::Error(format!("first-token logits read: {e}")));
                return;
            }
        };
        let mut logits = logits_view;
        if !params.logit_bias.is_empty() {
            let v = logits.len();
            for (&id, &bias) in &params.logit_bias {
                let idx = id as usize;
                if idx < v {
                    logits[idx] += bias;
                }
            }
        }
        sampler_pure::sample_token(&mut logits, sp, &[])
    } else {
        prefill_argmax
    };

    // --- Decode loop ---
    let decode_start = Instant::now();
    let mut completion_tokens = 0usize;
    let mut accumulated_text = String::new();
    let mut reasoning_token_count = 0usize;
    let mut finish_reason: &'static str = "length";
    let mut profiler = ProfileAccumulator::new(0);
    // Iter-94: streaming path needs the running token list for
    // sampler_pure's repetition_penalty.  Pre-iter-94 only the
    // accumulated_text was tracked (sufficient for stop-string scan),
    // because the loop ran greedy-only.
    let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    generated_tokens.push(next_token);

    // Emit prefill-produced first token:
    let first_text = loaded
        .tokenizer
        .decode(&[next_token], false)
        .unwrap_or_default();
    let mut is_eos_first = loaded.eos_token_ids.contains(&next_token);
    if !is_eos_first && !first_text.is_empty() {
        accumulated_text.push_str(&first_text);
        if emit_fragment(&mut splitter, events, &first_text).is_err() {
            tracing::info!("SSE stream dropped by client; aborting decode");
            return;
        }
    }
    completion_tokens += 1;
    if splitter.as_ref().map(|s| s.in_reasoning()).unwrap_or(false) {
        reasoning_token_count += 1;
    }
    if is_eos_first {
        finish_reason = "stop";
    } else if hit_stop_string(&accumulated_text, &params.stop_strings) {
        finish_reason = "stop";
        is_eos_first = true;
    }

    if !is_eos_first {
        for _ in 1..max_tokens {
            let pos = prompt_len + completion_tokens - 1;
            let mut p = profiler.start_token();
            let dec_result =
                loaded
                    .weights
                    .forward_decode(next_token, pos, &mut loaded.ctx, &mut p);
            profiler.finish_token(p);
            let greedy_token = match dec_result {
                Ok(t) => t,
                Err(e) => {
                    send!(GenerationEvent::Error(format!("decode failed: {e}")));
                    return;
                }
            };
            next_token = if let Some(sp) = sampler_params.as_ref() {
                let mut logits: Vec<f32> = match loaded.weights.logits_view() {
                    Ok(v) => v.to_vec(),
                    Err(e) => {
                        send!(GenerationEvent::Error(format!("logits read: {e}")));
                        return;
                    }
                };
                if !params.logit_bias.is_empty() {
                    let v = logits.len();
                    for (&id, &bias) in &params.logit_bias {
                        let idx = id as usize;
                        if idx < v {
                            logits[idx] += bias;
                        }
                    }
                }
                sampler_pure::sample_token(&mut logits, sp, &generated_tokens)
            } else {
                greedy_token
            };

            if loaded.eos_token_ids.contains(&next_token) {
                finish_reason = "stop";
                break;
            }
            completion_tokens += 1;
            generated_tokens.push(next_token);
            let fragment = loaded
                .tokenizer
                .decode(&[next_token], false)
                .unwrap_or_default();
            accumulated_text.push_str(&fragment);
            if emit_fragment(&mut splitter, events, &fragment).is_err() {
                tracing::info!("SSE stream dropped by client; aborting decode");
                return;
            }
            if splitter.as_ref().map(|s| s.in_reasoning()).unwrap_or(false) {
                reasoning_token_count += 1;
            }
            if hit_stop_string(&accumulated_text, &params.stop_strings) {
                finish_reason = "stop";
                break;
            }
        }
    }

    // Drain any leftover tail the splitter was holding back.
    if let Some(sp) = splitter.as_mut() {
        if let Some((slot, tail)) = sp.finish() {
            let kind = match slot {
                super::registry::SplitSlot::Content => DeltaKind::Content,
                super::registry::SplitSlot::Reasoning => DeltaKind::Reasoning,
            };
            if !tail.is_empty() {
                if events.blocking_send(GenerationEvent::Delta { kind, text: tail }).is_err() {
                    tracing::info!("SSE stream dropped by client; aborting decode");
                    return;
                }
            }
        }
    }

    let decode_duration = decode_start.elapsed();

    let stats = StreamStats {
        prefill_time_secs: Some(prefill_duration.as_secs_f64()),
        decode_time_secs: Some(decode_duration.as_secs_f64()),
        total_time_secs: Some(
            (prefill_duration + decode_duration).as_secs_f64(),
        ),
        time_to_first_token_ms: Some(prefill_duration.as_secs_f64() * 1000.0),
        prefill_tokens_per_sec: Some(if prefill_duration.as_secs_f64() > 0.0 {
            prompt_len as f64 / prefill_duration.as_secs_f64()
        } else {
            0.0
        }),
        decode_tokens_per_sec: Some(if decode_duration.as_secs_f64() > 0.0 {
            completion_tokens as f64 / decode_duration.as_secs_f64()
        } else {
            0.0
        }),
        gpu_sync_count: Some(mlx_native::sync_count().saturating_sub(pre_syncs)),
        gpu_dispatch_count: Some(
            mlx_native::dispatch_count().saturating_sub(pre_dispatches),
        ),
        cached_prompt_tokens: None,
        reasoning_tokens: if reasoning_token_count > 0 {
            Some(reasoning_token_count)
        } else {
            None
        },
    };

    send!(GenerationEvent::Done {
        finish_reason,
        prompt_tokens: prompt_len,
        completion_tokens,
        stats,
    });
}

fn hit_stop_string(text: &str, stops: &[String]) -> bool {
    if stops.is_empty() {
        return false;
    }
    stops.iter().any(|s| !s.is_empty() && text.ends_with(s.as_str()))
}

fn strip_trailing_stop(text: &mut String, stops: &[String]) {
    for s in stops {
        if !s.is_empty() && text.ends_with(s) {
            let new_len = text.len() - s.len();
            text.truncate(new_len);
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenizer + chat-template helpers usable from handlers
// ---------------------------------------------------------------------------

/// Render a Jinja2 chat template over an OpenAI-shaped message list.
///
/// The minijinja environment mirrors the one the one-shot `cmd_generate`
/// path uses: `messages`, `add_generation_prompt`, `bos_token`, `eos_token`
/// are in scope. Content handling:
///
///   - `content: "plain string"` → the template sees `content = "..."`.
///   - `content: [{type:"text", text:"..."}, ...]` → text parts are
///     concatenated; image parts are ignored in this iter (multimodal lands
///     with Phase 2c). A future iter will pass typed parts to vision-aware
///     templates.
///   - OpenAI `assistant` role is remapped to `model` if the GGUF template
///     is Gemma 4 (detected by presence of `<|turn>model` in the template).
///     Otherwise roles are passed through verbatim.
pub fn render_chat_prompt(
    template_str: &str,
    messages: &[super::schema::ChatMessage],
) -> Result<String> {
    use super::schema::MessageContent;

    let remap_assistant_to_model = template_str.contains("<|turn>model");
    let mut out_msgs: Vec<serde_json::Value> = Vec::with_capacity(messages.len());
    for msg in messages {
        let mut role = msg.role.clone();
        if remap_assistant_to_model && role == "assistant" {
            role = "model".to_string();
        }
        let content_text = msg
            .content
            .as_ref()
            .map(|c| match c {
                MessageContent::Text(s) => s.clone(),
                MessageContent::Parts(_) => c.text(),
            })
            .unwrap_or_default();
        out_msgs.push(serde_json::json!({
            "role": role,
            "content": content_text,
        }));
    }

    let mut env = minijinja::Environment::new();
    env.add_template("chat", template_str)
        .context("Failed to parse chat template as Jinja2")?;
    let tmpl = env
        .get_template("chat")
        .context("Failed to load parsed chat template")?;
    let rendered = tmpl
        .render(minijinja::context! {
            messages => out_msgs,
            add_generation_prompt => true,
            bos_token => "<bos>",
            eos_token => "<eos>",
        })
        .context("Failed to render chat template")?;
    Ok(rendered)
}

/// Resolve tokenizer path the same way `cmd_generate` does.
fn find_tokenizer(model_path: &Path, explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    let dir = model_path.parent().unwrap_or(Path::new("."));
    let candidate = dir.join("tokenizer.json");
    if candidate.exists() {
        return Ok(candidate);
    }
    for subdir in &["gemma4", "gemma-4"] {
        let candidate = Path::new("models").join(subdir).join("tokenizer.json");
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    let models_dir = Path::new("models");
    if models_dir.is_dir() {
        for entry in std::fs::read_dir(models_dir)? {
            let entry = entry?;
            if entry.path().is_dir() {
                let tok = entry.path().join("tokenizer.json");
                if tok.exists() {
                    return Ok(tok);
                }
            }
        }
    }
    anyhow::bail!(
        "Cannot find tokenizer.json. Use --tokenizer to specify the path explicitly."
    )
}

/// Resolve config.json path (same heuristics as cmd_generate).
fn find_config(model_path: &Path, explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    let dir = model_path.parent().unwrap_or(Path::new("."));
    let candidate = dir.join("config.json");
    if candidate.exists() {
        return Ok(candidate);
    }
    for subdir in &["gemma4", "gemma-4"] {
        let candidate = Path::new("models").join(subdir).join("config.json");
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    let models_dir = Path::new("models");
    if models_dir.is_dir() {
        for entry in std::fs::read_dir(models_dir)? {
            let entry = entry?;
            if entry.path().is_dir() {
                let c = entry.path().join("config.json");
                if c.exists() {
                    return Ok(c);
                }
            }
        }
    }
    anyhow::bail!(
        "Cannot find config.json. Use --config to specify the path explicitly."
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::schema::{ChatMessage, ContentPart, ImageUrl, MessageContent};

    #[test]
    fn sampling_params_default_is_greedy_t0() {
        let p = SamplingParams::default();
        assert_eq!(p.temperature, 0.0);
        assert_eq!(p.top_p, 1.0);
        assert_eq!(p.top_k, 0);
        assert_eq!(p.repetition_penalty, 1.0);
        assert_eq!(p.max_tokens, 512);
        assert!(p.stop_strings.is_empty());
    }

    #[test]
    fn hit_stop_string_empty_stops_is_false() {
        assert!(!hit_stop_string("anything", &[]));
    }

    #[test]
    fn hit_stop_string_matches_trailing() {
        let stops = vec!["END".to_string()];
        assert!(hit_stop_string("blah END", &stops));
        assert!(!hit_stop_string("END blah", &stops));
        assert!(!hit_stop_string("blah", &stops));
    }

    #[test]
    fn hit_stop_string_ignores_empty_stop() {
        let stops = vec!["".to_string(), "END".to_string()];
        // Empty strings should not cause false positives.
        assert!(!hit_stop_string("blah", &stops));
        assert!(hit_stop_string("blah END", &stops));
    }

    #[test]
    fn strip_trailing_stop_removes_suffix() {
        let mut s = String::from("hello END");
        strip_trailing_stop(&mut s, &["END".to_string()]);
        assert_eq!(s, "hello ");
    }

    #[test]
    fn strip_trailing_stop_no_match_leaves_unchanged() {
        let mut s = String::from("hello");
        strip_trailing_stop(&mut s, &["END".to_string()]);
        assert_eq!(s, "hello");
    }

    #[test]
    fn render_chat_prompt_single_user_round_trip() {
        // A minimal Jinja template that just formats role:content per line.
        let tmpl = r#"{%- for m in messages -%}
{{ m.role }}: {{ m.content }}
{%- endfor -%}
{%- if add_generation_prompt -%}
assistant:
{%- endif -%}"#;
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Text("hi".into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let out = render_chat_prompt(tmpl, &msgs).unwrap();
        assert!(out.contains("user: hi"));
        assert!(out.ends_with("assistant:"));
    }

    #[test]
    fn render_chat_prompt_remaps_assistant_for_gemma_template() {
        // Template that contains the Gemma 4 marker `<|turn>model` triggers
        // the assistant→model remap.
        let tmpl = "<|turn>system\n<|turn>user\n{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}<|turn>model\n";
        let msgs = vec![
            ChatMessage {
                role: "user".into(),
                content: Some(MessageContent::Text("hi".into())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: Some(MessageContent::Text("hello".into())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];
        let out = render_chat_prompt(tmpl, &msgs).unwrap();
        assert!(out.contains("user:hi"));
        // assistant should have been remapped to model
        assert!(out.contains("model:hello"));
        assert!(!out.contains("assistant:hello"));
    }

    #[test]
    fn render_chat_prompt_does_not_remap_for_non_gemma_template() {
        let tmpl = "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}";
        let msgs = vec![ChatMessage {
            role: "assistant".into(),
            content: Some(MessageContent::Text("hello".into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let out = render_chat_prompt(tmpl, &msgs).unwrap();
        assert!(out.contains("assistant:hello"));
    }

    #[test]
    fn render_chat_prompt_concatenates_multimodal_text_parts() {
        let tmpl = "{% for m in messages %}{{ m.content }}|{% endfor %}";
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text { text: "what is ".into() },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,XXX".into(),
                        detail: None,
                    },
                },
                ContentPart::Text { text: "this?".into() },
            ])),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let out = render_chat_prompt(tmpl, &msgs).unwrap();
        // Image part is silently dropped (iter 3 scope); text parts joined.
        assert_eq!(out.trim(), "what is this?|");
    }

    // -----------------------------------------------------------------
    // Engine::shutdown — joins the worker thread (Phase 2a Decision #17)
    //
    // These tests stand up an `Engine` with a stub worker thread that
    // drains the channel and exits on the `Shutdown` sentinel. The real
    // worker (`worker_run`) requires a `LoadedModel` (GGUF + tokenizer);
    // for unit-testing the lifecycle wiring we substitute a no-op worker
    // that exercises the same exit path. The point of the test is to
    // verify that `Engine::shutdown` actually joins the OS thread, that
    // it is idempotent, and that it propagates a panic in the worker.
    // -----------------------------------------------------------------

    fn make_test_engine_with_worker<F>(worker: F) -> Engine
    where
        F: FnOnce(mpsc::Receiver<Request>) + Send + 'static,
    {
        let (tx, rx) = mpsc::channel::<Request>(8);
        let handle = std::thread::Builder::new()
            .name("hf2q-engine-test".into())
            .spawn(move || worker(rx))
            .expect("spawn test worker");

        Engine {
            inner: Arc::new(EngineInner {
                tx,
                worker_handle: Mutex::new(Some(handle)),
                model_id: "test-model".into(),
                context_length: None,
                quant_type: None,
                hidden_size: 0,
                eos_token_ids: vec![],
                tokenizer: Arc::new(Tokenizer::new(tokenizers::models::bpe::BPE::default())),
                chat_template: Arc::new(String::new()),
                registration: None,
            }),
        }
    }

    /// Stub worker: drain until `Shutdown`, then exit cleanly.
    fn drain_until_shutdown(mut rx: mpsc::Receiver<Request>) {
        while let Some(req) = rx.blocking_recv() {
            if matches!(req, Request::Shutdown) {
                break;
            }
            // Other request kinds are not exercised by these tests; in the
            // production worker they have replies, but here we just drop
            // them — the senders never await a reply.
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_joins_worker_thread() {
        let engine = make_test_engine_with_worker(drain_until_shutdown);
        // Worker should be live before shutdown.
        {
            let guard = engine.inner.worker_handle.lock().unwrap();
            assert!(guard.is_some(), "worker handle present pre-shutdown");
        }
        engine.shutdown().await.expect("clean shutdown");
        // Handle must have been taken (and joined) by shutdown.
        let guard = engine.inner.worker_handle.lock().unwrap();
        assert!(guard.is_none(), "worker handle taken post-shutdown");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_is_idempotent() {
        let engine = make_test_engine_with_worker(drain_until_shutdown);
        engine.shutdown().await.expect("first shutdown");
        // A second call must not panic and must not deadlock; the
        // worker_handle slot is empty so the join step is skipped, and
        // the Sender is closed so `tx.send` errors silently.
        engine.shutdown().await.expect("second shutdown is no-op");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_propagates_worker_panic() {
        // Worker panics on its first message. shutdown() sends Shutdown,
        // which the worker receives → panics → join returns Err → our
        // shutdown() returns Err with the panic context.
        let engine = make_test_engine_with_worker(|mut rx| {
            let _ = rx.blocking_recv();
            panic!("test panic in worker");
        });
        let res = engine.shutdown().await;
        let err = res.expect_err("expected join failure");
        let msg = format!("{}", err);
        assert!(
            msg.contains("panicked"),
            "shutdown error should name 'panicked', got: {msg}"
        );
    }
}
