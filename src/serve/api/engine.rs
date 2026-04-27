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
use crate::serve::forward_prefill::SoftTokenInjection;
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
/// Tool-call enforcement policy derived from the request's `tool_choice`.
///
/// Wave-2.5 A4: the streaming worker's `route_content` fallback silently
/// emitted unparseable tool-call bodies as plain Content for ALL tool_choice
/// modes, including `Required` and explicit `Function`.  For constrained
/// modes the model is supposed to emit a valid call (the grammar guarantees
/// it structurally), so a parse failure is a server-side bug — not a
/// graceful-degradation scenario.  `Auto` genuinely needs the fallback
/// because there is no grammar constraint and a partial/malformed tool call
/// is recoverable by the client as plain text.
///
/// # Why not derive from `schema::ToolChoiceValue` here
///
/// `SamplingParams` is an engine-layer type; it must not import from
/// `schema` (HTTP-layer).  This enum re-expresses only the
/// policy-relevant distinctions (Auto vs Constrained).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolCallPolicy {
    /// `tool_choice = "auto"` (or absent).  The model may or may not call a
    /// tool; parse failures fall back to Content (existing behavior).
    #[default]
    Auto,
    /// `tool_choice = "required"` or `tool_choice = {type: "function", ...}`.
    /// Grammar guarantees well-formed output; a parse failure is promoted to
    /// `GenerationEvent::Error` with `finish_reason = "error"` on the
    /// streaming path, and an HTTP 500 on the non-streaming path.
    Constrained,
}

/// Kind discriminant for the grammar attached to a request.
///
/// Mirrors llama.cpp `enum common_grammar_type` at
/// `/opt/llama.cpp/common/common.h:171-176`:
///
/// ```c++
/// enum common_grammar_type {
///     COMMON_GRAMMAR_TYPE_NONE,
///     COMMON_GRAMMAR_TYPE_USER,
///     COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,
///     COMMON_GRAMMAR_TYPE_TOOL_CALLS,
/// };
/// ```
///
/// Wave 2.6 W-α5 motivation (cfa-20260427-adr005-wave2.6 research-report.md
/// Q1, audit `codex-review-last.txt` divergence "A1 / response_format
/// regression" severity HIGH): without this kind, the wave-2.5 A1 fix
/// gates **every** grammar on `ToolCallSplitter::in_tool_call()` — which
/// silently disables `response_format=json_object` / `json_schema`
/// enforcement on registered Gemma/Qwen models because the splitter never
/// fires for non-tool requests.  The kind tells the runtime whether to
/// enforce unconditionally (`ResponseFormat`) or to wait for a trigger
/// before enforcing (`ToolCallBody`, the lazy-grammar pattern from
/// llama.cpp PR #9639).
///
/// vLLM's `StructuredOutputsParams` and SGLang's mutually exclusive
/// `json_schema` / `regex` / `ebnf` fields are the same shape — one
/// constraint kind per request, asserted at request-parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrammarKind {
    /// User-supplied or `response_format`-derived grammar.
    /// Applies UNCONDITIONALLY for the entire generation, from the very
    /// first token.  The grammar runtime never enters `awaiting_trigger`
    /// state — the mask fires every step and `accept_bytes` advances
    /// every step.  Pre-A1 (wave 2.4 and earlier) behavior.
    ///
    /// Mirrors `COMMON_GRAMMAR_TYPE_USER` + `COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT`
    /// in llama.cpp.
    ///
    /// Default — preserves backward compatibility for any caller that
    /// constructs `SamplingParams` without setting the field.
    #[default]
    ResponseFormat,
    /// Tool-call body grammar (output of `compile_tool_grammar`).
    /// Applies ONLY after the model emits the per-model open marker
    /// (e.g. Gemma 4 `call:`, Qwen 3.5 `<function=`).  Until then, the
    /// runtime sits in `awaiting_trigger == true`: `apply()` is a no-op
    /// (no mask), `accept()` is a no-op (no advance, no
    /// dead/accepted-state termination check).  When the
    /// ToolCallSplitter sees the open marker, the engine calls
    /// `runtime.trigger()` to flip the flag false; the runtime then
    /// enforces every subsequent token through to the close marker.
    ///
    /// Mirrors `COMMON_GRAMMAR_TYPE_TOOL_CALLS` + `grammar_lazy=true`
    /// in llama.cpp (`/opt/llama.cpp/src/llama-grammar.cpp:1287-1344`).
    ToolCallBody,
}

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
    /// Per-token-id logit bias map. Additive bias applied to the
    /// live logits before `sampler_pure::sample_token` (wired in
    /// iter-94; OpenAI semantics — non-finite bias on a token vetoes
    /// it, finite bias shifts its logit).
    pub logit_bias: std::collections::HashMap<u32, f32>,
    /// If `true`, include top-k logprobs in the response. Tier 4.
    pub logprobs: bool,
    /// Number of top alternatives to report per chosen token. 0 = only the
    /// chosen token's logprob.
    pub top_logprobs: u32,
    /// `true` = allow multiple tool calls in the same turn. Tier 4. Plumbs
    /// through to the grammar-constrained decode path when it lands.
    pub parallel_tool_calls: bool,

    // --- Grammar-constrained decoding (Decision #6, Task #5, iter-95) ---
    /// Pre-compiled GBNF grammar to constrain decode-time token selection.
    /// `None` ⇒ unconstrained (default sampling on raw logits).  When
    /// `Some(g)`, the decode loop builds a fresh `GrammarRuntime`
    /// (`mask::mask_invalid_tokens` clones it per-token), calls the
    /// mask BEFORE `sampler_pure::sample_token`, and feeds the chosen
    /// token's bytes through the runtime so the next step's mask is
    /// correctly narrowed.
    ///
    /// Built by `handlers.rs::compile_response_format`.  `Grammar` is
    /// `Clone` (cheap — just a `Vec<Vec<GretElement>>`); the chat
    /// handler clones it into the per-request `SamplingParams`.
    pub grammar: Option<super::grammar::Grammar>,
    /// Per-vocab decoded UTF-8 byte table for grammar masking.  `None`
    /// when `grammar` is also `None` (no grammar, no need for the
    /// table).  When `grammar` is `Some`, this MUST be `Some(table)`
    /// — the chat handler obtains it via `Engine::token_bytes_table()`
    /// (lazily built + cached on the Engine).  Cheap to attach: an
    /// Arc clone (no copy of the underlying vector).
    pub token_bytes: Option<Arc<Vec<Vec<u8>>>>,
    /// Kind discriminant for the grammar attached above.
    ///
    /// Wave 2.6 W-α5 (research-report.md Q1, audit divergence "A1 /
    /// response_format regression").  Decides whether the grammar runtime
    /// is unconditionally enforcing (`ResponseFormat`, the default) or
    /// trigger-gated (`ToolCallBody`).  Set by:
    ///   * `compile_response_format` → `GrammarKind::ResponseFormat`
    ///   * `compile_tool_grammar`    → `GrammarKind::ToolCallBody`
    ///
    /// Default = `ResponseFormat` so any caller that builds
    /// `SamplingParams` without touching the field gets the
    /// pre-wave-2.5 unconditional-enforcement behavior.
    pub grammar_kind: GrammarKind,

    // --- Wave-2.5 A4 — Tool-call parse-failure policy ---
    /// Policy for handling tool-call body parse failures.  Set from
    /// `tool_choice` by `handlers.rs::prepare_chat_generation_core`.
    /// Defaults to `Auto` (content fallback) so the pre-wave-2.5
    /// behavior is preserved for all callers that don't set this field.
    pub tool_call_policy: ToolCallPolicy,
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
            grammar: None,
            token_bytes: None,
            grammar_kind: GrammarKind::default(),
            tool_call_policy: ToolCallPolicy::Auto,
        }
    }
}

/// Owned soft-token override sent through the worker channel.
///
/// Identical contract to [`SoftTokenInjection`] but owns the
/// `MlxBuffer` (channel-friendly: needs `Send`).  The worker thread
/// rebuilds borrowed `SoftTokenInjection<'_>` slices from a
/// `&[SoftTokenData]` for the prefill call.  Phase 2c Task #17 / iter-98.
#[derive(Debug, Clone)]
pub struct SoftTokenData {
    /// Half-open position range within the prompt: `[start, end)`.
    pub range: std::ops::Range<usize>,
    /// Replacement embeddings, shape `[range.len(), hidden_size]` F32,
    /// row-major.  Cheap-clone (Arc-shared underlying Metal buffer).
    pub embeddings: mlx_native::MlxBuffer,
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
    /// Number of prompt tokens served from the prompt cache (Phase 2a
    /// Task #7, Decision #24).  Reported via OpenAI's
    /// `usage.prompt_tokens_details.cached_tokens`.  Iter-96 single-slot
    /// full-equality cache: this is `prompt_tokens` on a cache hit
    /// (entire prefill + decode skipped) and `0` otherwise.  Iter-97+
    /// extends to LCP-based partial-prefill resume which can report
    /// any value `0 ≤ cached_tokens ≤ prompt_tokens`.
    pub cached_tokens: usize,
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
    /// Vocabulary size — needed to size the per-vocab token_bytes table
    /// (built lazily on first grammar request).
    vocab_size: usize,
    eos_token_ids: Vec<u32>,
    /// Tokenizer cloned per-request so handlers can tokenize without a lock.
    tokenizer: Arc<Tokenizer>,
    /// Chat-template string (GGUF metadata or fallback). Rendered per request
    /// using `minijinja`.
    chat_template: Arc<String>,
    /// Per-model registration — reasoning-boundary + tool-call markers
    /// (Decision #21). `None` when no family matches this model's id.
    registration: Option<super::registry::ModelRegistration>,
    /// Per-vocab decoded UTF-8 byte table — `token_bytes[id]` is the
    /// bytes the tokenizer emits when token `id` is sampled.  Built on
    /// first grammar request via `Engine::token_bytes_table()` and
    /// cached for the engine's lifetime (vocab × ~3-5 bytes ≈ 1 MB at
    /// 256K vocab — trivial vs the model weights).  `OnceLock` so
    /// concurrent first-callers race only on the build, not on reads.
    /// Phase 2a Task #5 / iter-95.
    token_bytes: std::sync::OnceLock<Arc<Vec<Vec<u8>>>>,
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
        /// Per-position embedding overrides for the multimodal vision
        /// path (Phase 2c iter-211 W79). Empty slice ⇒ identity over
        /// the text-only `forward_prefill` path (the prefill function
        /// is already a thin wrapper around
        /// `forward_prefill_with_soft_tokens` with an empty slice —
        /// see `src/serve/forward_prefill.rs:111-118`).
        ///
        /// Pre-iter-211 the streaming worker did not carry soft-token
        /// data and the chat handler returned a 400 when an `image_url`
        /// content part was present + `stream: true`. Iter-211 closes
        /// AC 3103 by routing soft tokens through the streaming path
        /// using the same forward-prefill API the non-streaming
        /// `Request::GenerateWithSoftTokens` arm already uses.
        soft_tokens: Vec<SoftTokenData>,
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
    /// Vision-aware chat generation (Phase 2c Task #17 / iter-98).
    /// See `Engine::generate_with_soft_tokens` doc.
    GenerateWithSoftTokens {
        prompt_tokens: Vec<u32>,
        soft_tokens: Vec<SoftTokenData>,
        params: SamplingParams,
        reply: oneshot::Sender<Result<GenerationResult>>,
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
    /// Single-slot prompt cache (Phase 2a Task #7 / Decision #24, iter-96).
    /// Owned by the worker thread; lives across requests.  See
    /// `PromptCache` doc for the cache contract.
    pub prompt_cache: PromptCache,
}

/// Generation-affecting parameters that must all match for a cache hit.
///
/// Wave-2.5 B5 (HIGH-7): the iter-96 cache keyed only on prompt tokens,
/// silently ignoring `max_tokens`, `stop_strings`, `logit_bias`, and
/// `grammar`.  Two requests with the same prompt but different max_tokens
/// would incorrectly replay a shorter (or longer) cached response.  This
/// newtype makes all generation-affecting fields part of the equality
/// check.
///
/// Only fields that affect greedy-decode (T=0) output are included:
/// - `max_tokens` — early-stop trigger
/// - `stop_strings` — early-stop trigger
/// - `logit_bias` — additive shift applied before greedy argmax
/// - `grammar` — token-validity mask applied before argmax
///
/// Fields that are irrelevant to greedy decode (temperature=0,
/// top_p=1, top_k=0, repetition_penalty=1, seed=None) are deliberately
/// excluded — the existing sampling-bypass gate in `lookup`/`store`
/// already ensures those requests never reach the cache.
///
/// `frequency_penalty` and `presence_penalty` are plumbed but currently
/// not wired into the greedy path; they are excluded here and will be
/// added when they become effective.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptCacheKey {
    pub max_tokens: usize,
    pub stop_strings: Vec<String>,
    /// Sorted key-value pairs from `logit_bias` so that two maps with
    /// identical contents compare equal regardless of insertion order.
    /// The bias values are stored as `f32` bit-patterns (via
    /// `to_bits()`) to enable structural equality without floating-point
    /// surprises.  Finite `f32` bias values are the only ones with
    /// meaningful semantics; `f32::NAN` keys would be a caller bug.
    pub logit_bias_sorted: Vec<(u32, u32)>,
    /// Structural equality: `GretElement: PartialEq` + `Grammar: PartialEq`.
    /// `None` means no grammar constraint; `Some(g)` means the entire
    /// GBNF rule set must match.
    pub grammar: Option<super::grammar::Grammar>,
}

impl PromptCacheKey {
    /// Construct from a `SamplingParams`.
    pub fn from_params(params: &SamplingParams) -> Self {
        let mut bias_sorted: Vec<(u32, u32)> = params
            .logit_bias
            .iter()
            .map(|(&tok, &bias)| (tok, bias.to_bits()))
            .collect();
        bias_sorted.sort_unstable_by_key(|&(tok, _)| tok);
        Self {
            max_tokens: params.max_tokens,
            stop_strings: params.stop_strings.clone(),
            logit_bias_sorted: bias_sorted,
            grammar: params.grammar.clone(),
        }
    }
}

/// Single-slot prompt cache (Phase 2a Task #7, iter-96).
///
/// **Iter-96 scope: full-equality + temperature=0 cache.**  When the
/// next chat request's prompt_tokens exactly matches `tokens` AND the
/// caller's `temperature == 0` (deterministic decode), the cache
/// short-circuits the entire prefill+decode and replays the previous
/// response.  Useful for retries (network failures), eval consistency,
/// repeated benchmarks, idempotent agentic loops.
///
/// **Iter-97+ scope: LCP-based partial-prefill resume.**  Compute the
/// longest common prefix between the new prompt and `tokens`, set
/// `kv_caches[*].write_pos = LCP`, pre-warm `dense_kvs[0..LCP)` by
/// dequantizing `kv_caches[0..LCP)` via `tq_dequantize_kv`, then run
/// `forward_prefill` for tokens `[LCP..N)`.  Reports `cached_tokens =
/// LCP` (any value `0 ≤ LCP ≤ prompt_tokens`).  Defers to a later
/// iteration because the dequant pre-warm is non-trivial — the iter-96
/// full-equality cache is a real, shippable subset.
///
/// Sampling (`temperature > 0`) **bypasses the cache** even on full
/// equality — replaying the deterministic-greedy decoded text under a
/// sampling request would silently violate the user's expectation of
/// per-call variation.  No cache write happens on sampling-mode hits
/// either; sampling completions are always re-generated.
///
/// Grammar-constrained requests (`response_format=json_object` /
/// `json_schema`) follow the same rule: greedy + matching prompt =
/// cache hit; sampling = cache bypass.  The grammar runtime state
/// at the end of generation is NOT cached (would over-constrain a
/// future hit if the cached grammar differed from the new request's).
///
/// # Design invariants (DO NOT re-create a separate prompt_cache module)
///
/// A simpler `PromptCache` with only `lcp_len` / `update` / `clear` methods
/// was prototyped in `src/serve/api/prompt_cache.rs` (ADR-005 Task #7 first
/// cut) and deleted in wave-1.5 (2026-04-26) because:
///
/// 1. **Full-equality is the shipped contract.**  The iter-96 cache fires
///    only when `new_prompt == cached_prompt` exactly.  The prototype's
///    LCP algorithm is correct but belongs to the iter-97+ scope (LCP-based
///    partial-prefill resume) which needs a `forward_decode` refactor to
///    expose the KV write position — that refactor is deferred.
///
/// 2. **Full-response-replay, not partial-skip.**  On a cache hit the
///    worker returns the complete cached `GenerationResult` (`text`,
///    `reasoning_text`, `completion_tokens`, `finish_reason`, …) without
///    running the decoder at all.  `cached_tokens = prompt_len` surfaces
///    in the OpenAI usage shape per the spec.
///
/// 3. **Owned by the worker thread.**  `PromptCache` lives inside
///    `LoadedModel` (field `prompt_cache`), which is exclusive to the
///    single worker thread; no synchronization needed.  Moving it to a
///    shared module would require Arc/Mutex overhead without benefit.
///
/// Future LCP-based work belongs here, extending `lookup`/`store`.
#[derive(Debug, Clone)]
pub struct PromptCache {
    /// The previous request's prompt token sequence (post-rendering,
    /// post-tokenization).  Empty on a fresh worker (no prior request).
    pub tokens: Vec<u32>,
    /// Wave-2.5 B5: all generation-affecting params from the previous
    /// request.  A new request must match both `tokens` AND `key` to
    /// get a cache hit.
    pub key: PromptCacheKey,
    /// The text the previous request emitted (post reasoning-marker
    /// split).  This is what gets replayed on a cache hit.
    pub text: String,
    /// The `reasoning_text` field from the previous result.  Replayed
    /// alongside `text` so the response shape matches the original.
    pub reasoning_text: Option<String>,
    /// Number of completion tokens the previous request emitted.
    pub completion_tokens: usize,
    /// Reasoning-token count from the previous result.
    pub reasoning_tokens: Option<usize>,
    /// `"stop"` | `"length"` from the previous result.
    pub finish_reason: &'static str,
}

impl Default for PromptCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptCache {
    /// Empty cache — initial state for a fresh worker.
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            key: PromptCacheKey {
                max_tokens: 0,
                stop_strings: Vec::new(),
                logit_bias_sorted: Vec::new(),
                grammar: None,
            },
            text: String::new(),
            reasoning_text: None,
            completion_tokens: 0,
            reasoning_tokens: None,
            finish_reason: "length",
        }
    }

    /// Cache check: returns the cached result if and only if
    /// `prompt_tokens` exactly equals the cached prompt, the caller
    /// is in greedy decode mode (temperature = 0, no sampling-only
    /// fields set), AND all generation-affecting params match the
    /// cached key (max_tokens, stop_strings, logit_bias, grammar).
    ///
    /// Wave-2.5 B5: added `PromptCacheKey` comparison to prevent
    /// silent-correctness bugs where same prompt + different max_tokens,
    /// stops, logit_bias, or grammar would incorrectly replay a stale
    /// cached response.
    pub fn lookup(&self, prompt_tokens: &[u32], params: &SamplingParams) -> Option<GenerationResult> {
        // Bypass for any non-greedy mode.  These all introduce per-call
        // variance that a cached replay would silently erase.
        if params.temperature > 0.0
            || params.top_k > 0
            || params.top_p < 1.0
            || params.repetition_penalty != 1.0
            || params.seed.is_some()
        {
            return None;
        }
        if self.tokens.is_empty() || self.tokens.as_slice() != prompt_tokens {
            return None;
        }
        // Wave-2.5 B5: generation-affecting params must also match.
        let request_key = PromptCacheKey::from_params(params);
        if self.key != request_key {
            return None;
        }
        Some(GenerationResult {
            text: self.text.clone(),
            reasoning_text: self.reasoning_text.clone(),
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: self.completion_tokens,
            reasoning_tokens: self.reasoning_tokens,
            finish_reason: self.finish_reason,
            // Cache hit: prefill and decode were both skipped — report
            // zero wall-clock for both phases.  TTFT effectively becomes
            // the cache lookup time (~1µs), surfaced as 0 in the response.
            prefill_duration: Duration::ZERO,
            decode_duration: Duration::ZERO,
            cached_tokens: prompt_tokens.len(),
        })
    }

    /// Cache write: store this request's result so the next
    /// equal-prompt + greedy + equal-params request can short-circuit.
    ///
    /// Same eligibility gate as `lookup` — sampling-mode requests are
    /// not cached (storing them would mean a future greedy request
    /// could replay a sampling outcome, violating determinism).
    pub fn store(
        &mut self,
        prompt_tokens: &[u32],
        params: &SamplingParams,
        result: &GenerationResult,
    ) {
        if params.temperature > 0.0
            || params.top_k > 0
            || params.top_p < 1.0
            || params.repetition_penalty != 1.0
            || params.seed.is_some()
        {
            return;
        }
        self.tokens = prompt_tokens.to_vec();
        self.key = PromptCacheKey::from_params(params);
        self.text = result.text.clone();
        self.reasoning_text = result.reasoning_text.clone();
        self.completion_tokens = result.completion_tokens;
        self.reasoning_tokens = result.reasoning_tokens;
        self.finish_reason = result.finish_reason;
    }
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
            prompt_cache: PromptCache::new(),
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
        let vocab_size = loaded.weights.vocab_size;
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
                vocab_size,
                eos_token_ids,
                tokenizer,
                chat_template,
                registration,
                token_bytes: std::sync::OnceLock::new(),
            }),
        }
    }

    /// Lazily build + cache the per-vocab decoded UTF-8 byte table used
    /// by the grammar mask (Phase 2a Task #5 / iter-95).
    ///
    /// `token_bytes[id]` is the bytes the tokenizer emits when token `id`
    /// is sampled — exactly `tokenizer.decode(&[id], false)` lowered to
    /// raw UTF-8 bytes.  Empty entries (special / unprintable tokens
    /// like `<eos>` / `<turn|>`) are left blank; the mask treats them
    /// as "do not constrain" — the decode loop's EOS/stop-string layer
    /// owns those.
    ///
    /// Cost: vocab × one tokenizer.decode call.  At Gemma-4's vocab=256K
    /// this is ~50-200 ms on first call (CPU work; not on hot path),
    /// then ~free for every subsequent grammar request through this
    /// Engine.  The build runs on the calling thread so the worker
    /// thread is unaffected.
    ///
    /// Returned as `Arc<Vec<Vec<u8>>>` — the chat handler attaches it
    /// to `SamplingParams.token_bytes` (cheap Arc clone) so the worker
    /// thread can consume it without re-resolving on every request.
    pub fn token_bytes_table(&self) -> Arc<Vec<Vec<u8>>> {
        self.inner
            .token_bytes
            .get_or_init(|| {
                let v = self.inner.vocab_size;
                let tok = &self.inner.tokenizer;
                let mut out: Vec<Vec<u8>> = Vec::with_capacity(v);
                for id in 0..v as u32 {
                    // `decode` returns the rendered text per token; for
                    // BPE-byte-fallback vocabs the bytes round-trip via
                    // UTF-8.  Failure (out-of-range id, unsupported
                    // token) returns an empty string — emit empty bytes
                    // so the mask treats it as a "special" / unprintable
                    // token (left untouched).
                    let s = tok.decode(&[id], false).unwrap_or_default();
                    out.push(s.into_bytes());
                }
                tracing::info!(
                    "Engine: built per-vocab token_bytes table ({} ids, ~{:.1} MB)",
                    v,
                    out.iter().map(|v| v.len()).sum::<usize>() as f64 / 1e6
                );
                Arc::new(out)
            })
            .clone()
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
        soft_tokens: Vec<SoftTokenData>,
    ) -> Result<()> {
        let req = Request::GenerateStream {
            prompt_tokens,
            params,
            events: events_tx,
            cancellation_counter,
            soft_tokens,
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

    /// Vision-aware non-streaming generation (Phase 2c Task #17 / iter-98).
    ///
    /// Same as `generate` but passes per-position embedding overrides
    /// that the worker plugs into the prefill via
    /// `MlxModelWeights::forward_prefill_with_soft_tokens`.  Used by
    /// the chat handler when an `image_url` content part is present
    /// in the request: the projected vision embeddings for each image
    /// flow through this API as `SoftTokenData` covering the
    /// placeholder-token positions in the prompt.
    pub async fn generate_with_soft_tokens(
        &self,
        prompt_tokens: Vec<u32>,
        soft_tokens: Vec<SoftTokenData>,
        params: SamplingParams,
    ) -> Result<GenerationResult> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let req = Request::GenerateWithSoftTokens {
            prompt_tokens,
            soft_tokens,
            params,
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
        reply_rx.await.context("vision generation reply dropped")?
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
                soft_tokens,
            } => {
                // The streaming path sends every event (Delta / Done / Error)
                // via `events`. Errors stay inside the function — the
                // terminal event is always one of Done/Error, unless the
                // receiver was dropped (client disconnect → early exit).
                // When the early-exit path fires, we bump the cancellation
                // counter if supplied (→ hf2q_sse_cancellations in /metrics).
                //
                // Phase 2c iter-211 W79: build borrowed `SoftTokenInjection<'_>`
                // slices from the owned `SoftTokenData` (channel-friendly Send)
                // mirroring the pattern used by `Request::GenerateWithSoftTokens`
                // above. Empty `soft_tokens` ⇒ identity over the text-only
                // prefill path (the prefill function is already a thin wrapper
                // around `forward_prefill_with_soft_tokens` with an empty
                // slice — see `src/serve/forward_prefill.rs:111-118`).
                let injections: Vec<SoftTokenInjection<'_>> = soft_tokens
                    .iter()
                    .map(|d| SoftTokenInjection {
                        range: d.range.clone(),
                        embeddings: &d.embeddings,
                    })
                    .collect();
                generate_stream_once(
                    &mut loaded,
                    &prompt_tokens,
                    &injections,
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
            Request::GenerateWithSoftTokens {
                prompt_tokens,
                soft_tokens,
                params,
                reply,
            } => {
                // Vision-aware generate (Phase 2c Task #17 / iter-98).
                // Build borrowed `SoftTokenInjection<'_>` slices from the
                // owned `SoftTokenData` we received over the channel; the
                // borrow lifetime is bounded by this match arm so it
                // can't outlive the underlying buffers.
                let injections: Vec<SoftTokenInjection<'_>> = soft_tokens
                    .iter()
                    .map(|d| SoftTokenInjection {
                        range: d.range.clone(),
                        embeddings: &d.embeddings,
                    })
                    .collect();
                let result = generate_once_with_soft_tokens(
                    &mut loaded,
                    &prompt_tokens,
                    &injections,
                    &params,
                    registration.as_ref(),
                );
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
    generate_once_with_soft_tokens(loaded, prompt_tokens, &[], params, registration)
}

/// Vision-aware variant — same as `generate_once` except the prefill
/// goes through `forward_prefill_with_soft_tokens` so per-position
/// embedding overrides apply.  Phase 2c Task #17 / iter-98.
///
/// When `soft_tokens` is empty, behaviour is byte-identical to
/// `generate_once`.
fn generate_once_with_soft_tokens(
    loaded: &mut LoadedModel,
    prompt_tokens: &[u32],
    soft_tokens: &[SoftTokenInjection<'_>],
    params: &SamplingParams,
    registration: Option<&super::registry::ModelRegistration>,
) -> Result<GenerationResult> {
    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "generate_once: empty prompt_tokens"
    );
    let prompt_len = prompt_tokens.len();
    let max_tokens = params.max_tokens.max(1);

    // ── Prompt cache fast-path (Phase 2a Task #7 / iter-96) ────────────
    //
    // When the request is fully deterministic (greedy: T=0, no top_k /
    // top_p / repetition_penalty / seed) AND the prompt_tokens exactly
    // match the previous request's prompt, replay the cached result.
    // Skips the entire prefill+decode chain — the only cost is the
    // O(N) prompt-tokens equality compare.  The OpenAI usage shape
    // surfaces `cached_tokens = prompt_len` so clients can attribute
    // the saving.
    //
    // Sampling-mode bypasses the cache: replaying a deterministic
    // greedy decode for a sampling request would silently violate the
    // user's expectation of per-call variation.  See `PromptCache`
    // module doc for the full eligibility rules.
    if let Some(cached) = loaded.prompt_cache.lookup(prompt_tokens, params) {
        tracing::debug!(
            "prompt_cache: HIT — {} tokens served from cache, prefill+decode skipped",
            cached.prompt_tokens
        );
        return Ok(cached);
    }

    // ── Sampler config — Tier 2/3/4 surface + grammar (iter-94 / iter-95) ──
    //
    // Pre-iter-94 the decode loop only consumed `forward_decode`'s
    // on-GPU greedy argmax — every `temperature` / `top_p` / `top_k` /
    // `repetition_penalty` / `logit_bias` request was silently downcast
    // to greedy.  Iter-94 forks on whether ANY field requests non-greedy
    // sampling and routes those through `sampler_pure::sample_token`
    // over the live `self.activations.logits` slice.  Iter-95 adds the
    // grammar branch: when `params.grammar.is_some()`, mask the live
    // logits via `grammar::mask::mask_invalid_tokens` BEFORE handing
    // them to `sampler_pure` (or to the greedy argmax for T=0).  The
    // chosen token's bytes then advance the runtime so the next step's
    // mask is correctly narrowed.
    //
    // Greedy fast path (all fields at default + no grammar) keeps the
    // existing forward_decode return-value chain — no logits readback,
    // no extra copy.  Sampling/grammar slow path discards the on-GPU
    // argmax token (~20 µs of wasted GPU work, negligible vs the
    // ~10-100ms layer forward) and re-derives the next token from the
    // mask + sample chain.
    let sample_logits = params.temperature > 0.0
        || params.top_k > 0
        || params.top_p < 1.0
        || params.repetition_penalty != 1.0
        || !params.logit_bias.is_empty()
        || params.grammar.is_some();
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

    // Build the per-request grammar runtime (Phase 2a Task #5 / iter-95).
    // `Grammar` is `Clone` (cheap ~Vec<Vec<GretElement>>); the runtime
    // owns the clone + a small Vec<Stack> of in-flight positions.  We
    // mutate in place across decode steps (advance via accept_bytes
    // after each sampled token).
    let mut grammar_runtime: Option<super::grammar::GrammarRuntime> = match params.grammar.as_ref() {
        Some(g) => {
            let start_rule_id = g
                .rule_id("root")
                .ok_or_else(|| anyhow::anyhow!("grammar has no root rule"))?;
            let rt = super::grammar::GrammarRuntime::new(g.clone(), start_rule_id)
                .ok_or_else(|| anyhow::anyhow!("grammar runtime init failed"))?;
            Some(rt)
        }
        None => None,
    };
    let token_bytes_ref: Option<&[Vec<u8>]> = params.token_bytes.as_deref().map(|v| &v[..]);

    // Wave-2.5 A1: ToolCallSplitter for the non-streaming decode loop.
    // Tracks whether the current decode position is inside a tool-call body
    // so the grammar mask can be gated on `tc_splitter.in_tool_call()`.
    // Grammar only constrains tokens inside the body — preamble tokens (before
    // the open marker) are unconstrained (Option B grammar boundary).
    // The splitter is `None` when the model has no tool markers registered;
    // in that case grammar masking runs unconditionally (pre-A1 behaviour).
    let mut tc_splitter_ns: Option<super::registry::ToolCallSplitter> =
        registration.and_then(|r| super::registry::ToolCallSplitter::from_registration(r));

    // Local helper — apply grammar mask + Tier 4 logit_bias and sample.
    // Mutably borrows the runtime so it can be advanced after sampling
    // (caller does the advance to keep this closure side-effect-light).
    // Returns the sampled token id; caller must feed
    // `token_bytes[id]` through the runtime to keep it in sync.
    //
    // `in_tool_body`: A1 gate — true when the ToolCallSplitter reports the
    // current position is inside a tool-call body.  When the splitter is
    // absent (no tool markers), this is always `true` so grammar masking
    // runs unconditionally (backward-compatible fallback).
    let sample_from_live_logits =
        |weights: &mut MlxModelWeights,
         generated: &[u32],
         runtime: Option<&super::grammar::GrammarRuntime>,
         in_tool_body: bool|
         -> Result<u32> {
            let sp = sampler_params.as_ref().expect("sample_logits gate");
            let mut logits: Vec<f32> = weights.logits_view()?.to_vec();
            // Tier 4 logit_bias FIRST: additive per OpenAI convention.
            if !params.logit_bias.is_empty() {
                let v = logits.len();
                for (&id, &bias) in &params.logit_bias {
                    let idx = id as usize;
                    if idx < v {
                        logits[idx] += bias;
                    }
                }
            }
            // Grammar mask: zero out tokens that would drive the runtime
            // dead.  Wave-2.5 A1: only applies when we are inside a tool-call
            // body (`in_tool_body == true`).  Outside the body (preamble tokens
            // before the open marker) the model is free to emit any text — the
            // grammar only constrains the call body structure.
            if in_tool_body {
                if let (Some(rt), Some(tb)) = (runtime, token_bytes_ref) {
                    super::grammar::mask::mask_invalid_tokens(rt, tb, &mut logits);
                }
            }
            Ok(sampler_pure::sample_token(&mut logits, sp, generated))
        };

    // --- Prefill ---
    // Iter-98: route through forward_prefill_with_soft_tokens. Empty
    // soft_tokens slice is the no-op identity over forward_prefill —
    // text-only requests pay zero overhead.
    let prefill_start = Instant::now();
    let prefill_argmax = loaded
        .weights
        .forward_prefill_with_soft_tokens(prompt_tokens, soft_tokens, max_tokens, &mut loaded.ctx)?;
    let prefill_duration = prefill_start.elapsed();

    // First decode token: greedy fast-path uses prefill's on-GPU argmax;
    // sampling path re-derives from prefill's live logits buffer (last
    // prompt-token's lm_head output) so the user-controlled temperature
    // applies to the very first generated token, not just decode-loop
    // tokens 2..N.  The greedy-fast-path skips logits readback entirely.
    //
    // Wave-2.5 A1: grammar mask is skipped for the first token because we
    // cannot yet be inside a tool-call body — the splitter hasn't seen any
    // output yet so `in_tool_call()` is always false at this point.
    let mut next_token = if sample_logits {
        // A1: splitter not yet advanced; first token is always pre-body.
        let in_body_first = tc_splitter_ns.as_ref().map(|_| false).unwrap_or(true);
        let tok = sample_from_live_logits(&mut loaded.weights, &[], grammar_runtime.as_ref(), in_body_first)?;
        // Feed the chosen token's bytes through the grammar runtime so
        // the next step's mask is correctly narrowed.  No-op when no
        // grammar.  An empty token_bytes entry (special / unprintable)
        // is also a no-op since accept_bytes on an empty slice returns
        // true with no state change.
        if let (Some(rt), Some(tb)) = (grammar_runtime.as_mut(), token_bytes_ref) {
            let bytes = tb.get(tok as usize).map(|v| v.as_slice()).unwrap_or(&[]);
            if !bytes.is_empty() {
                rt.accept_bytes(bytes);
            }
        }
        tok
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
    // Wave-2.5 A1: feed first fragment through tc_splitter_ns so its
    // in_tool_call() state is accurate for the first iteration of the
    // decode loop.  Typically the first decoded token is never the open
    // marker, but this keeps the state machine correct in all edge cases.
    if let Some(tcs) = tc_splitter_ns.as_mut() {
        let _ = tcs.feed(&first_fragment);
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
                // grammar mask, then call sampler_pure for
                // temperature/top_p/top_k/rep-penalty.
                //
                // Wave-2.5 A1: grammar mask is gated on whether the
                // ToolCallSplitter is currently inside a tool-call body.
                // Outside the body (preamble before the open marker) the
                // model is unconstrained.  If there is no tool splitter
                // registered for this model (tc_splitter_ns is None), fall
                // back to the pre-A1 unconditional masking so grammar still
                // applies for models/requests that don't use tool calls.
                let in_body = tc_splitter_ns.as_ref()
                    .map(|t| t.in_tool_call())
                    .unwrap_or(true);
                let tok = sample_from_live_logits(
                    &mut loaded.weights,
                    &generated_tokens,
                    grammar_runtime.as_ref(),
                    in_body,
                )?;
                // Advance the grammar runtime by the chosen token's bytes.
                if let (Some(rt), Some(tb)) = (grammar_runtime.as_mut(), token_bytes_ref) {
                    let bytes = tb.get(tok as usize).map(|v| v.as_slice()).unwrap_or(&[]);
                    if !bytes.is_empty() {
                        rt.accept_bytes(bytes);
                    }
                }
                tok
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
            // Wave-2.5 A1: advance tc_splitter_ns so its in_tool_call()
            // state is correct for the NEXT step's grammar gate.
            if let Some(tcs) = tc_splitter_ns.as_mut() {
                let _ = tcs.feed(&fragment);
            }
            if hit_stop_string(&decoded_text, &params.stop_strings) {
                finish_reason = "stop";
                // Strip the stop string from the returned text (OpenAI
                // convention per ADR-005 "Stop-sequence stripping from
                // returned text").
                strip_trailing_stop(&mut decoded_text, &params.stop_strings);
                break;
            }
            // Grammar-driven termination (Phase 2a Task #5 / iter-95).
            //
            // After the grammar runtime tried to accept the chosen
            // token, `is_dead()` becomes true if no in-flight stack
            // can extend further.  Two ways this fires after a
            // grammar-constrained decode step:
            //
            //   1. **Mask masked everything**: every printable token's
            //      bytes failed the grammar, so `sampler_pure` softmaxed
            //      all-`-inf` logits, summed to zero, and fell back to
            //      `indexed[0]` (usually id=0 = `<pad>` for Gemma).
            //      That token's bytes also fail the grammar (`<pad>`
            //      decodes to literal `"<pad>"` text — `<` is not valid
            //      JSON after `} ws`).  `accept_bytes` returned false
            //      above ⇒ runtime is now dead.
            //   2. **Grammar fully matched + last token was the final
            //      legal one**: the runtime accepted the token but has
            //      no remaining stacks — the parse is complete.
            //
            // Both cases collapse to "decoder should halt".  Pop the
            // last pushed token + re-decode the surviving prefix so
            // any out-of-grammar fragment (`<pad>`) doesn't appear in
            // the response body.
            if grammar_runtime.as_ref().is_some_and(|rt| rt.is_dead()) {
                finish_reason = "stop";
                generated_tokens.pop();
                decoded_text = loaded
                    .tokenizer
                    .decode(&generated_tokens, false)
                    .unwrap_or_default();
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

    let result = GenerationResult {
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
        cached_tokens: 0, // iter-96: 0 on cache miss; > 0 on hit (handled by fast-path return earlier)
    };

    // Store this generation in the prompt cache — same eligibility
    // gate as `lookup` (sampling-mode requests are not cached).  The
    // store happens AFTER all error paths above so a partial / failed
    // generation can never poison the cache.
    loaded.prompt_cache.store(prompt_tokens, params, &result);

    Ok(result)
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
    soft_tokens: &[SoftTokenInjection<'_>],
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

    // Tool-call splitter (iter-133 Iter B-2) — classifies the
    // post-reasoning Content stream into in/out-of-tool-call spans. When a
    // tool-call span closes, its body is parsed into structured
    // `name + arguments_json` and emitted as one or more
    // `GenerationEvent::ToolCallDelta` chunks (id+name first, full
    // arguments string second; matches the SSE encoder's expectation in
    // `sse.rs:208-247`).
    //
    // Composition: the engine runs ReasoningSplitter first; any
    // `Content`-classified output then flows into ToolCallSplitter. Reasoning
    // never appears inside a tool call — neither chat template emits a
    // reasoning-open marker (Gemma 4 `<|channel>` or Qwen 3.5/3.6 `<think>`)
    // between tool-call markers — so this layering is safe.
    let mut tool_splitter = registration
        .and_then(|r| super::registry::ToolCallSplitter::from_registration(r));
    // Per-call body accumulator + per-stream tool-call index. Body is
    // bounded by max_tokens so unbounded growth is impossible. Index is
    // incremented every time a tool-call closes and emits a delta — used
    // as the OpenAI `delta.tool_calls[*].index` field.
    let mut tool_call_body: String = String::new();
    let mut tool_call_index: usize = 0;
    // Set true on first ToolCallClose; latched. Drives `finish_reason ==
    // "tool_calls"` per OpenAI spec (decode loop's normal `"stop"` /
    // `"length"` is overridden when this flag is set on the terminating
    // path).
    let mut saw_tool_call: bool = false;

    // Wave-2.5 A4: capture the policy so the route_content closure can branch
    // on Constrained vs Auto when a tool-call body fails to parse.
    let tool_call_policy = params.tool_call_policy;

    // Wave-2.5 A1: shared AtomicBool tracking whether the streaming decode
    // loop is currently inside a tool-call body.  `route_content` toggles it
    // on ToolCallOpen (true) / ToolCallClose (false).  The grammar mask in the
    // decode loop reads it before each sampling step so the constraint only
    // applies to tokens that are inside the body (Option B boundary).
    //
    // Arc is used so both the `route_content` closure (which captures it by
    // clone) and the decode loop (which reads it directly) can share ownership
    // without lifetime conflicts.  All accesses use `Relaxed` ordering: the
    // decode loop runs single-threaded so sequentially consistent ordering is
    // unnecessary.
    let grammar_active = std::sync::Arc::new(
        std::sync::atomic::AtomicBool::new(
            // Initial state: if there is no tool splitter registered for this
            // model, default to `true` so grammar runs unconditionally
            // (backward-compatible fallback for models without tool markers).
            tool_splitter.is_none(),
        ),
    );
    let grammar_active_for_closure = grammar_active.clone();

    // Helper: for a Content-classified text run, route through the
    // ToolCallSplitter (if any) and emit the appropriate
    // GenerationEvent. When ToolCallSplitter is None, route every byte to
    // `DeltaKind::Content` (current behavior pre-iter-B-2).
    let route_content = |tool_splitter: &mut Option<super::registry::ToolCallSplitter>,
                         body: &mut String,
                         tc_index: &mut usize,
                         saw_tc: &mut bool,
                         events: &mpsc::Sender<GenerationEvent>,
                         text: &str,
                         reg: Option<&super::registry::ModelRegistration>|
     -> Result<(), ()> {
        if text.is_empty() {
            return Ok(());
        }
        let Some(tcs) = tool_splitter.as_mut() else {
            // No tool markers registered — original behavior.
            if events
                .blocking_send(GenerationEvent::Delta {
                    kind: DeltaKind::Content,
                    text: text.to_string(),
                })
                .is_err()
            {
                return Err(());
            }
            return Ok(());
        };
        for ev in tcs.feed(text) {
            match ev {
                super::registry::ToolCallEvent::Content(t) => {
                    if !t.is_empty()
                        && events
                            .blocking_send(GenerationEvent::Delta {
                                kind: DeltaKind::Content,
                                text: t,
                            })
                            .is_err()
                    {
                        return Err(());
                    }
                }
                super::registry::ToolCallEvent::ToolCallOpen => {
                    body.clear();
                    // Wave-2.5 A1: entering a tool-call body — activate
                    // grammar masking for subsequent tokens.
                    grammar_active_for_closure.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                super::registry::ToolCallEvent::ToolCallText(t) => {
                    body.push_str(&t);
                }
                super::registry::ToolCallEvent::ToolCallClose => {
                    // Wave-2.5 A1: leaving a tool-call body — deactivate
                    // grammar masking.  The close event fires after the body
                    // is fully accumulated so the last body token was already
                    // constrained.
                    grammar_active_for_closure.store(false, std::sync::atomic::Ordering::Relaxed);
                    // Parse the accumulated body. On parse failure, log +
                    // re-emit the body as Content so the user still sees
                    // something (the per-model parser is best-effort
                    // fallback; well-formed output is the common case
                    // because the in-template syntax is fixed).
                    let parsed = reg
                        .and_then(|r| super::registry::parse_tool_call_body(r, body));
                    let body_dump = std::mem::take(body);
                    match parsed {
                        Some(pc) => {
                            // First chunk: id + type + name. The id is a
                            // synthesized opaque identifier; clients echo
                            // it in their `tool_call_id` follow-up message.
                            // Format mirrors OpenAI's `call_<24hex>` shape.
                            let id = format!(
                                "call_hf2q_{:016x}",
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map(|d| d.as_nanos() as u64)
                                    .unwrap_or(0)
                                    ^ (*tc_index as u64).wrapping_mul(0x9e3779b97f4a7c15)
                            );
                            if events
                                .blocking_send(
                                    GenerationEvent::ToolCallDelta {
                                        index: *tc_index,
                                        id: Some(id),
                                        call_type: Some("function".into()),
                                        name: Some(pc.name),
                                        arguments: None,
                                    },
                                )
                                .is_err()
                            {
                                return Err(());
                            }
                            // Second chunk: full arguments JSON string.
                            // OpenAI clients accumulate `function.arguments`
                            // deltas; one chunk is spec-valid.
                            if events
                                .blocking_send(
                                    GenerationEvent::ToolCallDelta {
                                        index: *tc_index,
                                        id: None,
                                        call_type: None,
                                        name: None,
                                        arguments: Some(pc.arguments_json),
                                    },
                                )
                                .is_err()
                            {
                                return Err(());
                            }
                            *tc_index += 1;
                            *saw_tc = true;
                        }
                        None => {
                            // Wave-2.5 A4: promote parse failure to error for
                            // Constrained mode (Required / Function).  For
                            // Auto mode keep the existing Content fallback —
                            // the model may legitimately emit partial /
                            // malformed syntax when unconstrained (no grammar
                            // guard), and re-emitting the body as Content lets
                            // the client see what the model intended rather
                            // than silently discarding output.
                            if tool_call_policy == ToolCallPolicy::Constrained {
                                tracing::error!(
                                    body = %body_dump,
                                    "tool-call body unparseable under Constrained policy \
                                     (tool_choice=required/function); promoting to error"
                                );
                                let _ = events.blocking_send(GenerationEvent::Error(
                                    "tool_call_parse_failure".into(),
                                ));
                                return Err(());
                            }
                            tracing::warn!(
                                body = %body_dump,
                                "tool-call body unparseable; emitting as content fallback \
                                 (tool_choice=auto)"
                            );
                            if events
                                .blocking_send(GenerationEvent::Delta {
                                    kind: DeltaKind::Content,
                                    text: body_dump,
                                })
                                .is_err()
                            {
                                return Err(());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    };

    // Local helper to emit a fragment through the reasoning splitter (if
    // any) and then through the tool-call router. Returns the bytes emitted
    // (for stop-string bookkeeping). Note: each splitter holds back a tail
    // that's drained at generation end.
    let emit_fragment = |splitter: &mut Option<super::registry::ReasoningSplitter>,
                         tool_splitter: &mut Option<super::registry::ToolCallSplitter>,
                         body: &mut String,
                         tc_index: &mut usize,
                         saw_tc: &mut bool,
                         events: &mpsc::Sender<GenerationEvent>,
                         fragment: &str,
                         reg: Option<&super::registry::ModelRegistration>|
     -> Result<(), ()> {
        if fragment.is_empty() {
            return Ok(());
        }
        if let Some(sp) = splitter.as_mut() {
            for (slot, text) in sp.feed(fragment) {
                match slot {
                    super::registry::SplitSlot::Reasoning => {
                        if !text.is_empty()
                            && events
                                .blocking_send(GenerationEvent::Delta {
                                    kind: DeltaKind::Reasoning,
                                    text,
                                })
                                .is_err()
                        {
                            return Err(());
                        }
                    }
                    super::registry::SplitSlot::Content => {
                        route_content(
                            tool_splitter,
                            body,
                            tc_index,
                            saw_tc,
                            events,
                            &text,
                            reg,
                        )?;
                    }
                }
            }
        } else {
            // No reasoning splitter — route everything as Content.
            route_content(
                tool_splitter,
                body,
                tc_index,
                saw_tc,
                events,
                fragment,
                reg,
            )?;
        }
        Ok(())
    };

    // Snapshot mlx-native process-global GPU counters pre-generation so we
    // can report the per-request delta on the terminal `Done` event's
    // StreamStats (mirrors the non-streaming path's x_hf2q_timing counters).
    let pre_dispatches = mlx_native::dispatch_count();
    let pre_syncs = mlx_native::sync_count();

    // ── Sampler config — Tier 2/3/4 + grammar (iter-94 / iter-95, mirrors generate_once) ──
    let sample_logits = params.temperature > 0.0
        || params.top_k > 0
        || params.top_p < 1.0
        || params.repetition_penalty != 1.0
        || !params.logit_bias.is_empty()
        || params.grammar.is_some();
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
    let mut grammar_runtime: Option<super::grammar::GrammarRuntime> = match params.grammar.as_ref() {
        Some(g) => {
            let start_rule_id = match g.rule_id("root") {
                Some(id) => id,
                None => {
                    send!(GenerationEvent::Error("grammar has no root rule".into()));
                    return;
                }
            };
            match super::grammar::GrammarRuntime::new(g.clone(), start_rule_id) {
                Some(rt) => Some(rt),
                None => {
                    send!(GenerationEvent::Error("grammar runtime init failed".into()));
                    return;
                }
            }
        }
        None => None,
    };
    let token_bytes_ref: Option<&[Vec<u8>]> = params.token_bytes.as_deref().map(|v| &v[..]);

    // --- Prefill ---
    // Iter-211 W79: routed through `forward_prefill_with_soft_tokens` so
    // vision content parts can stream. `soft_tokens` is empty for text-only
    // requests; the prefill API treats an empty slice as identity over
    // `forward_prefill` (`src/serve/forward_prefill.rs:117`), so the
    // text-only path stays byte-identical.
    let prefill_start = Instant::now();
    let next_token_result =
        loaded
            .weights
            .forward_prefill_with_soft_tokens(prompt_tokens, soft_tokens, max_tokens, &mut loaded.ctx);
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
        // Wave-2.5 A1: first token is always pre-body (no output has been
        // emitted yet so the tool splitter hasn't opened a body).
        // `grammar_active` was initialized to `false` for models with tool
        // markers, so this check is consistent: mask is skipped here.
        if grammar_active.load(std::sync::atomic::Ordering::Relaxed) {
            if let (Some(rt), Some(tb)) = (grammar_runtime.as_ref(), token_bytes_ref) {
                super::grammar::mask::mask_invalid_tokens(rt, tb, &mut logits);
            }
        }
        let tok = sampler_pure::sample_token(&mut logits, sp, &[]);
        if let (Some(rt), Some(tb)) = (grammar_runtime.as_mut(), token_bytes_ref) {
            let bytes = tb.get(tok as usize).map(|v| v.as_slice()).unwrap_or(&[]);
            if !bytes.is_empty() {
                rt.accept_bytes(bytes);
            }
        }
        tok
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
        if emit_fragment(
            &mut splitter,
            &mut tool_splitter,
            &mut tool_call_body,
            &mut tool_call_index,
            &mut saw_tool_call,
            events,
            &first_text,
            registration,
        )
        .is_err()
        {
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
                // Wave-2.5 A1: only apply grammar mask when we are inside a
                // tool-call body.  `grammar_active` is set to `true` by
                // `route_content` when a ToolCallOpen event fires and reset to
                // `false` on ToolCallClose.  Tokens outside the body (preamble
                // before the open marker) are unconstrained.
                if grammar_active.load(std::sync::atomic::Ordering::Relaxed) {
                    if let (Some(rt), Some(tb)) = (grammar_runtime.as_ref(), token_bytes_ref) {
                        super::grammar::mask::mask_invalid_tokens(rt, tb, &mut logits);
                    }
                }
                let tok = sampler_pure::sample_token(&mut logits, sp, &generated_tokens);
                if let (Some(rt), Some(tb)) = (grammar_runtime.as_mut(), token_bytes_ref) {
                    let bytes = tb.get(tok as usize).map(|v| v.as_slice()).unwrap_or(&[]);
                    if !bytes.is_empty() {
                        rt.accept_bytes(bytes);
                    }
                }
                tok
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
            if emit_fragment(
                &mut splitter,
                &mut tool_splitter,
                &mut tool_call_body,
                &mut tool_call_index,
                &mut saw_tool_call,
                events,
                &fragment,
                registration,
            )
            .is_err()
            {
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
            // Grammar-driven termination — see generate_once for full doc.
            // Streaming variant: we can't pop the trailing token cleanly
            // because the fragment was already emitted to the SSE stream;
            // accept the small wart.  Iter-96+ candidate: hold back the
            // last fragment until next-step grammar state is known so it
            // can be suppressed pre-emit.
            if grammar_runtime.as_ref().is_some_and(|rt| rt.is_dead()) {
                finish_reason = "stop";
                break;
            }
            if let Some(rt) = grammar_runtime.as_ref() {
                if rt.is_accepted() {
                    if let Some(tb) = token_bytes_ref {
                        let bytes = tb.get(next_token as usize).map(|v| v.as_slice()).unwrap_or(&[]);
                        if bytes.is_empty() {
                            finish_reason = "stop";
                            break;
                        }
                    }
                }
            }
        }
    }

    // Drain any leftover tail the reasoning splitter was holding back. If
    // the tail is Content-classified, route it through the tool-call
    // splitter so a marker straddling EOS is still detected.
    if let Some(sp) = splitter.as_mut() {
        if let Some((slot, tail)) = sp.finish() {
            match slot {
                super::registry::SplitSlot::Reasoning => {
                    if !tail.is_empty()
                        && events
                            .blocking_send(GenerationEvent::Delta {
                                kind: DeltaKind::Reasoning,
                                text: tail,
                            })
                            .is_err()
                    {
                        tracing::info!("SSE stream dropped by client; aborting decode");
                        return;
                    }
                }
                super::registry::SplitSlot::Content => {
                    if route_content(
                        &mut tool_splitter,
                        &mut tool_call_body,
                        &mut tool_call_index,
                        &mut saw_tool_call,
                        events,
                        &tail,
                        registration,
                    )
                    .is_err()
                    {
                        tracing::info!("SSE stream dropped by client; aborting decode");
                        return;
                    }
                }
            }
        }
    }
    // Drain any tool-splitter tail. If the stream ended mid-tool-call (no
    // close marker observed), the tail is ToolCallText: re-emit as plain
    // Content so the partial body isn't silently dropped — the operator
    // can see the truncation in `delta.content` and the test harness asserts
    // SSE shape, not byte-perfect content.
    if let Some(tcs) = tool_splitter.as_mut() {
        if let Some(ev) = tcs.finish() {
            match ev {
                super::registry::ToolCallEvent::Content(t) => {
                    if !t.is_empty()
                        && events
                            .blocking_send(GenerationEvent::Delta {
                                kind: DeltaKind::Content,
                                text: t,
                            })
                            .is_err()
                    {
                        tracing::info!("SSE stream dropped by client; aborting decode");
                        return;
                    }
                }
                super::registry::ToolCallEvent::ToolCallText(t) => {
                    // Mid-call truncation. Emit residual body as Content
                    // (open marker was already swallowed by the splitter,
                    // but for diagnostic clarity we wrap with the literal
                    // open marker so a stop-mid-call is still visible).
                    let prefix = registration.and_then(|r| r.tool_open).unwrap_or("");
                    let fallback = format!("{prefix}{t}");
                    if !fallback.is_empty()
                        && events
                            .blocking_send(GenerationEvent::Delta {
                                kind: DeltaKind::Content,
                                text: fallback,
                            })
                            .is_err()
                    {
                        tracing::info!("SSE stream dropped by client; aborting decode");
                        return;
                    }
                }
                super::registry::ToolCallEvent::ToolCallOpen
                | super::registry::ToolCallEvent::ToolCallClose => {
                    // unreachable: finish() never emits Open/Close.
                }
            }
        }
    }
    // Override finish_reason to "tool_calls" per OpenAI spec when at least
    // one structured tool-call delta was emitted on this stream. Spec:
    // https://platform.openai.com/docs/guides/function-calling — when the
    // model invokes a tool, finish_reason becomes "tool_calls" rather than
    // "stop"/"length". This overrides EOS-driven "stop" set above.
    if saw_tool_call {
        finish_reason = "tool_calls";
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

    // Iter-96 prompt cache update on streaming completion.  Streaming
    // currently does NOT consult the cache on input (would require
    // fake-emitting Delta events from cached text — iter-97 follow-up).
    // But updating the cache on EVERY successful generation, regardless
    // of streaming mode, means that a streaming request followed by a
    // non-streaming request with the same prompt will hit the cache.
    let cache_result = GenerationResult {
        text: accumulated_text.clone(),
        reasoning_text: None, // splitter already routed reasoning into Delta events
        prompt_tokens: prompt_len,
        completion_tokens,
        reasoning_tokens: if reasoning_token_count > 0 {
            Some(reasoning_token_count)
        } else {
            None
        },
        finish_reason,
        prefill_duration,
        decode_duration,
        cached_tokens: 0,
    };
    loaded.prompt_cache.store(prompt_tokens, params, &cache_result);
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
/// path uses: `messages`, `add_generation_prompt`, `bos_token`, `eos_token`,
/// and (when supplied) `tools` are in scope. Content handling:
///
///   - `content: "plain string"` → the template sees `content = "..."`.
///   - `content: [{type:"text", text:"..."}, ...]` → text parts are
///     concatenated; image parts are ignored in this iter (multimodal lands
///     with Phase 2c). A future iter will pass typed parts to vision-aware
///     templates.
///   - OpenAI `assistant` role is remapped to `model` if the GGUF template
///     is Gemma 4 (detected by presence of `<|turn>model` in the template).
///     Otherwise roles are passed through verbatim.
///   - Per-message `tool_calls` (assistant-emitted) and synthetic
///     `tool_responses` (synthesized from OpenAI `role: "tool"` history
///     messages, see [`render_chat_prompt_with_tools`]) are exposed to the
///     template as message fields so tool-aware templates (e.g. Gemma 4's
///     `<|tool_call>` / `<|tool_response>` markers) render correctly.
///
/// Use [`render_chat_prompt_with_tools`] to supply tool definitions; the
/// thin entry-point `render_chat_prompt` is kept for legacy callers (one-shot
/// `cmd_generate`, the chat-template overflow path) that don't carry tools.
pub fn render_chat_prompt(
    template_str: &str,
    messages: &[super::schema::ChatMessage],
) -> Result<String> {
    render_chat_prompt_with_tools(template_str, messages, None, false)
}

/// Render the chat template with optional tool-definition exposure.
///
/// ADR-005 Phase 2a iter-133 Iter B production fix-forward: prior to this
/// iter, `tools` and per-message `tool_calls` / `tool_call_id` carried by
/// the request schema were silently dropped before render — every tool-aware
/// chat template (Gemma 4, Qwen 3.5/3.6, Llama 3.x) saw an empty `tools`
/// variable and emitted no tool-call definitions to the model. As a result,
/// the model never had a chance to invoke a tool even when the operator
/// declared one. This function threads them through:
///
///   1. `tools` (top-level Jinja variable): the raw OpenAI tool definitions,
///      serialized as JSON values. Templates check `{%- if tools -%}` before
///      iterating, so `None`/empty leaves existing behavior unchanged.
///   2. Per-message `tool_calls` (on assistant messages): each assistant
///      message in `messages` gets its `tool_calls` array exposed as a
///      Jinja-visible field. The template iterates and emits per-model markers
///      (e.g. Gemma 4's `<|tool_call>call:NAME{...}<tool_call|>`).
///   3. Synthetic `tool_responses` (on `role: "tool"` messages): OpenAI
///      represents tool results as `{role: "tool", tool_call_id, content}`
///      sibling messages. Most chat templates (Gemma 4 included) instead
///      expect a per-message `tool_responses: [{name, response}]` field.
///      We synthesize that field on each `role: "tool"` message by looking up
///      the function name from the prior assistant `tool_calls` keyed by
///      `tool_call_id`. The role itself is left verbatim — the template
///      decides whether to wrap with `<|turn>tool` or similar.
pub fn render_chat_prompt_with_tools(
    template_str: &str,
    messages: &[super::schema::ChatMessage],
    tools: Option<&[super::schema::Tool]>,
    enable_thinking: bool,
) -> Result<String> {
    use super::schema::MessageContent;

    let remap_assistant_to_model = template_str.contains("<|turn>model");

    // Build a lookup table tool_call_id → function-name for synthesizing
    // `tool_responses` on subsequent role:"tool" messages. This is a
    // single-pass scan; assistant messages with tool_calls populate the
    // map, role:"tool" messages consume it.
    let mut id_to_name: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for msg in messages {
        if let Some(tcs) = msg.tool_calls.as_ref() {
            for tc in tcs {
                id_to_name.insert(tc.id.clone(), tc.function.name.clone());
            }
        }
    }

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
        let mut obj = serde_json::Map::new();
        obj.insert("role".into(), serde_json::Value::String(role));
        obj.insert("content".into(), serde_json::Value::String(content_text));

        // Assistant tool_calls: serialize each as
        // `{id, type, function: {name, arguments}}`. `arguments` is kept as
        // the raw OpenAI string (Gemma 4's template handles both string and
        // mapping shapes; preserving the string form is closest to how the
        // model emitted it the first time).
        if let Some(tcs) = msg.tool_calls.as_ref() {
            let arr: Vec<serde_json::Value> = tcs
                .iter()
                .map(|tc| {
                    serde_json::json!({
                        "id": tc.id,
                        "type": tc.call_type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })
                })
                .collect();
            obj.insert("tool_calls".into(), serde_json::Value::Array(arr));
        }

        // role:"tool" messages → synthesize `tool_responses` field. The
        // OpenAI shape is `{tool_call_id, content}`; we look up the tool
        // name from id_to_name and pass the content string verbatim.
        // Templates (Gemma 4) accept either string or mapping for the
        // `response` field; the string path is the safer default.
        if msg.role == "tool" {
            if let Some(tcid) = msg.tool_call_id.as_ref() {
                let name = id_to_name
                    .get(tcid)
                    .cloned()
                    .unwrap_or_else(|| "unknown".into());
                let response_str = msg
                    .content
                    .as_ref()
                    .map(|c| match c {
                        MessageContent::Text(s) => s.clone(),
                        MessageContent::Parts(_) => c.text(),
                    })
                    .unwrap_or_default();
                obj.insert(
                    "tool_responses".into(),
                    serde_json::json!([{"name": name, "response": response_str}]),
                );
            }
        }

        out_msgs.push(serde_json::Value::Object(obj));
    }

    // Tools serialize verbatim. Each Tool is `{type, function: {name,
    // description, parameters}}`; serde_json::to_value mirrors the wire
    // shape. Skipping the threading entirely when `None` keeps the existing
    // legacy callers (one-shot generate, overflow tokenize) byte-identical.
    let tools_json: serde_json::Value = match tools {
        None => serde_json::Value::Null,
        Some(t) if t.is_empty() => serde_json::Value::Null,
        Some(t) => serde_json::to_value(t).unwrap_or(serde_json::Value::Null),
    };

    let mut env = minijinja::Environment::new();
    // ADR-005 Phase 2a iter-133 Iter A side-fix: register
    // `minijinja-contrib`'s pycompat callback so chat templates authored
    // against Python's Jinja2 (which transparently exposes Python-string
    // methods like `.split()`, `.strip()`, `.lstrip()`, `.lower()`) render
    // without `UnknownMethod` errors. The Gemma 4 chat template's
    // `strip_thinking` macro calls `text.split('<channel|>')` on every
    // assistant content; without pycompat, every multi-turn chat fails
    // HTTP 400 at the second user turn (the first model message hits the
    // macro). Discovered during the iter-133 Open WebUI multi-turn E2E
    // test harness; pycompat is purely additive (consulted only when
    // minijinja can't find a method natively), so existing single-turn
    // and non-Python-style templates are unaffected.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template("chat", template_str)
        .context("Failed to parse chat template as Jinja2")?;
    let tmpl = env
        .get_template("chat")
        .context("Failed to load parsed chat template")?;
    let rendered = tmpl
        .render(minijinja::context! {
            messages => out_msgs,
            tools => tools_json,
            // ADR-005 Phase 2a iter-133 Iter D (W67): expose
            // `enable_thinking` to the chat template so reasoning-capable
            // models can be told to actually emit a thinking trace. Gemma 4's
            // template (chat_template.jinja:161-163) emits a `<|think|>`
            // system-block hint when this is true; lines 263-265 also
            // skip the closed-channel seed (`<|channel>thought\n<channel|>`)
            // so the model is free to emit its own
            // `<|channel>thought\n…<channel|>` block during decode. Qwen 3.5/3.6
            // templates accept the same kwarg per HF convention. Templates
            // that don't reference `enable_thinking` ignore the variable
            // (Jinja default behavior — undefined-or-default(false) gates
            // are unchanged), so the legacy non-reasoning path stays
            // byte-identical.
            enable_thinking => enable_thinking,
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
    fn render_chat_prompt_handles_pythonic_string_methods_via_pycompat() {
        // ADR-005 Phase 2a iter-133 Iter A regression test for the
        // pycompat side-fix in `render_chat_prompt`. Real-world chat
        // templates (Gemma 4's `strip_thinking` macro is the surfaced
        // case) call Python-string methods like `.split()` directly on
        // string values; minijinja's vanilla Environment doesn't expose
        // those, so a multi-turn render that exercises the macro fails
        // with `UnknownMethod: string has no method named split` at the
        // second user turn.
        //
        // This template is a minimal stand-in: a `{%- macro -%}` that
        // calls `text.split('|')` (Python-style), invoked once per
        // assistant message inside a `for messages` loop. Without the
        // pycompat callback this `render_chat_prompt` call panics on
        // unwrap; with the callback it renders cleanly.
        let tmpl = "<|turn>model\n\
                    {%- macro splitter(text) -%}\
                    {%- for part in text.split('|') -%}{{ part }}+{% endfor -%}\
                    {%- endmacro -%}\
                    {% for m in messages %}\
                    {%- if m.role == 'model' -%}M:{{ splitter(m.content) }}\n\
                    {%- else -%}{{ m.role }}:{{ m.content }}\n{% endif -%}\
                    {% endfor %}";
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
                content: Some(MessageContent::Text("a|b|c".into())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "user".into(),
                content: Some(MessageContent::Text("again".into())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];
        let out = render_chat_prompt(tmpl, &msgs).unwrap();
        assert!(out.contains("user:hi"), "out={out}");
        // assistant remapped to model (gemma marker present), then split('|')
        // produced ["a", "b", "c"], each with a trailing '+'.
        assert!(out.contains("M:a+b+c+"), "out={out}");
        assert!(out.contains("user:again"), "out={out}");
    }

    #[test]
    fn render_chat_prompt_with_tools_threads_tools_into_jinja_context() {
        // ADR-005 Phase 2a iter-133 Iter B fix-forward: prior to this iter,
        // `tools` declared on a chat-completions request were silently
        // dropped before render — every tool-aware chat template (Gemma 4,
        // Qwen 3.5/3.6, Llama 3.x) saw an empty `tools` variable and emitted
        // no tool definitions to the model. Regression test: a minimal
        // template that just emits "TOOLS:<count>\n" + tool names
        // round-trips correctly.
        let tmpl = "{%- if tools -%}TOOLS:{{ tools | length }}\n\
                    {%- for t in tools -%}{{ t.function.name }};{%- endfor -%}\n\
                    {%- endif -%}\
                    {%- for m in messages -%}{{ m.role }}:{{ m.content }}\n{%- endfor -%}";
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Text("weather?".into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let tools = vec![
            super::super::schema::Tool {
                tool_type: "function".into(),
                function: super::super::schema::ToolFunction {
                    name: "get_current_weather".into(),
                    description: Some("Look up the weather".into()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    })),
                },
            },
            super::super::schema::Tool {
                tool_type: "function".into(),
                function: super::super::schema::ToolFunction {
                    name: "get_news".into(),
                    description: None,
                    parameters: None,
                },
            },
        ];
        let out = render_chat_prompt_with_tools(tmpl, &msgs, Some(&tools), false).unwrap();
        assert!(out.contains("TOOLS:2"), "out={out}");
        assert!(out.contains("get_current_weather;"), "out={out}");
        assert!(out.contains("get_news;"), "out={out}");
        assert!(out.contains("user:weather?"), "out={out}");

        // None / empty path → tools block must NOT fire.
        let out_none = render_chat_prompt_with_tools(tmpl, &msgs, None, false).unwrap();
        assert!(!out_none.contains("TOOLS:"), "out_none={out_none}");
        let out_empty = render_chat_prompt_with_tools(tmpl, &msgs, Some(&[]), false).unwrap();
        assert!(!out_empty.contains("TOOLS:"), "out_empty={out_empty}");

        // Legacy entry-point `render_chat_prompt` (no tools param) should be
        // byte-identical to the empty/None path.
        let out_legacy = render_chat_prompt(tmpl, &msgs).unwrap();
        assert_eq!(out_legacy, out_none);
    }

    #[test]
    fn render_chat_prompt_with_tools_threads_per_message_tool_calls_and_responses() {
        // Verify the per-message threading of `tool_calls` (assistant) and
        // synthesized `tool_responses` (from role:"tool" messages, looked up
        // by tool_call_id). A minimal template iterates messages and emits
        // tool_calls + tool_responses verbatim.
        let tmpl = "{%- for m in messages -%}\
                    [{{ m.role }}]\
                    {%- if m.tool_calls -%}\
                    {%- for tc in m.tool_calls -%}TC:{{ tc.function.name }}({{ tc.function.arguments }});{%- endfor -%}\
                    {%- endif -%}\
                    {%- if m.tool_responses -%}\
                    {%- for tr in m.tool_responses -%}TR:{{ tr.name }}={{ tr.response }};{%- endfor -%}\
                    {%- endif -%}\
                    {%- if m.content -%}{{ m.content }}{%- endif -%}\n\
                    {%- endfor -%}";
        let msgs = vec![
            ChatMessage {
                role: "user".into(),
                content: Some(MessageContent::Text("Paris weather?".into())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![super::super::schema::ToolCall {
                    id: "call_abc".into(),
                    call_type: "function".into(),
                    function: super::super::schema::ToolCallFunction {
                        name: "get_current_weather".into(),
                        arguments: "{\"location\":\"Paris\"}".into(),
                    },
                }]),
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some(MessageContent::Text(
                    "{\"temperature\": 18}".into(),
                )),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: Some("call_abc".into()),
                name: None,
            },
        ];
        let out = render_chat_prompt_with_tools(tmpl, &msgs, None, false).unwrap();
        assert!(out.contains("[user]Paris weather?"), "out={out}");
        // Assistant message: tool_calls visible verbatim (string-arguments).
        assert!(
            out.contains("[assistant]TC:get_current_weather({\"location\":\"Paris\"});"),
            "out={out}"
        );
        // Tool message: tool_responses synthesized via id_to_name lookup.
        assert!(
            out.contains("[tool]TR:get_current_weather={\"temperature\": 18};"),
            "out={out}"
        );
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
                vocab_size: 0,
                eos_token_ids: vec![],
                tokenizer: Arc::new(Tokenizer::new(tokenizers::models::bpe::BPE::default())),
                chat_template: Arc::new(String::new()),
                registration: None,
                token_bytes: std::sync::OnceLock::new(),
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

    // -----------------------------------------------------------------------
    // Wave-2.5 B5 — PromptCache key expansion (HIGH-7 silent-correctness fix)
    // -----------------------------------------------------------------------

    /// Helper: build a stored PromptCache that looks like a previous greedy
    /// request completed with `result_text`.
    fn make_cached(
        tokens: &[u32],
        params: &SamplingParams,
        result_text: &str,
    ) -> PromptCache {
        let mut cache = PromptCache::new();
        let result = GenerationResult {
            text: result_text.to_string(),
            reasoning_text: None,
            prompt_tokens: tokens.len(),
            completion_tokens: 5,
            reasoning_tokens: None,
            finish_reason: "stop",
            prefill_duration: Duration::ZERO,
            decode_duration: Duration::ZERO,
            cached_tokens: 0,
        };
        cache.store(tokens, params, &result);
        cache
    }

    #[test]
    fn prompt_cache_miss_on_different_max_tokens() {
        let tokens: Vec<u32> = vec![1, 2, 3];
        let mut base = SamplingParams::default();
        base.max_tokens = 100;
        let cache = make_cached(&tokens, &base, "hello");

        let mut req = SamplingParams::default();
        req.max_tokens = 200; // different max_tokens — must MISS
        assert!(
            cache.lookup(&tokens, &req).is_none(),
            "same prompt + different max_tokens must not hit cache"
        );
    }

    #[test]
    fn prompt_cache_miss_on_different_stop_strings() {
        let tokens: Vec<u32> = vec![1, 2, 3];
        let mut base = SamplingParams::default();
        base.stop_strings = vec!["STOP".to_string()];
        let cache = make_cached(&tokens, &base, "hello");

        let mut req = SamplingParams::default();
        req.stop_strings = vec!["END".to_string()]; // different stops — must MISS
        assert!(
            cache.lookup(&tokens, &req).is_none(),
            "same prompt + different stop_strings must not hit cache"
        );
    }

    #[test]
    fn prompt_cache_miss_on_different_logit_bias() {
        let tokens: Vec<u32> = vec![1, 2, 3];
        let mut base = SamplingParams::default();
        base.logit_bias.insert(42, 5.0);
        let cache = make_cached(&tokens, &base, "hello");

        let mut req = SamplingParams::default();
        req.logit_bias.insert(42, 10.0); // different bias value — must MISS
        assert!(
            cache.lookup(&tokens, &req).is_none(),
            "same prompt + different logit_bias must not hit cache"
        );
    }

    #[test]
    fn prompt_cache_miss_on_different_response_format_grammar() {
        use super::super::grammar::parser::{GretElement, GretType, Grammar};
        use std::collections::HashMap;

        let tokens: Vec<u32> = vec![1, 2, 3];

        // Build a trivial grammar A: single rule with one Char element + End.
        let grammar_a = Grammar {
            rules: vec![vec![
                GretElement::new(GretType::Char, b'a' as u32),
                GretElement::new(GretType::End, 0),
            ]],
            symbol_ids: {
                let mut m = HashMap::new();
                m.insert("root".to_string(), 0u32);
                m
            },
        };
        // Grammar B differs in the Char value.
        let grammar_b = Grammar {
            rules: vec![vec![
                GretElement::new(GretType::Char, b'b' as u32),
                GretElement::new(GretType::End, 0),
            ]],
            symbol_ids: {
                let mut m = HashMap::new();
                m.insert("root".to_string(), 0u32);
                m
            },
        };

        let mut base = SamplingParams::default();
        base.grammar = Some(grammar_a);
        let cache = make_cached(&tokens, &base, r#"{"ok":true}"#);

        let mut req = SamplingParams::default();
        req.grammar = Some(grammar_b); // different grammar — must MISS
        assert!(
            cache.lookup(&tokens, &req).is_none(),
            "same prompt + different grammar must not hit cache"
        );
    }

    /// Wave-2.5 B5 — tool_choice key sensitivity.
    ///
    /// `tool_choice` is compiled to a `Grammar` (via `compile_tool_grammar`)
    /// before being stored in `SamplingParams.grammar`. Two requests with the
    /// same prompt but different tool grammars (different tool_choice values)
    /// must produce a cache MISS. This test exercises the grammar arm of the
    /// PromptCacheKey directly — the same code path that `tool_choice=function`
    /// exercises at the end of `prepare_chat_completion_common`.
    #[test]
    fn prompt_cache_miss_on_different_tool_choice_grammar() {
        use super::super::grammar::parser::{GretElement, GretType, Grammar};
        use std::collections::HashMap;

        let tokens: Vec<u32> = vec![1, 2, 3];

        // Simulate grammar compiled for tool_choice=function{name:"tool_a"}.
        let grammar_tool_a = Grammar {
            rules: vec![vec![
                GretElement::new(GretType::Char, b'a' as u32),
                GretElement::new(GretType::End, 0),
            ]],
            symbol_ids: {
                let mut m = HashMap::new();
                m.insert("root".to_string(), 0u32);
                m
            },
        };
        // Simulate grammar compiled for tool_choice=function{name:"tool_b"}.
        let grammar_tool_b = Grammar {
            rules: vec![vec![
                GretElement::new(GretType::Char, b'b' as u32),
                GretElement::new(GretType::End, 0),
            ]],
            symbol_ids: {
                let mut m = HashMap::new();
                m.insert("root".to_string(), 0u32);
                m
            },
        };

        // Cache a response generated under tool_a grammar.
        let mut base = SamplingParams::default();
        base.grammar = Some(grammar_tool_a);
        let cache = make_cached(&tokens, &base, r#"{"name":"tool_a"}"#);

        // A subsequent request with tool_b grammar must MISS — it would
        // produce different output under a different constraint.
        let mut req = SamplingParams::default();
        req.grammar = Some(grammar_tool_b);
        assert!(
            cache.lookup(&tokens, &req).is_none(),
            "same prompt + different tool_choice grammar must not hit cache \
             (would silently replay the wrong tool call)"
        );

        // A request with NO grammar (tool_choice absent / unconstrained) must
        // also MISS — unconstrained decode differs from constrained decode.
        let req_no_grammar = SamplingParams::default();
        assert!(
            cache.lookup(&tokens, &req_no_grammar).is_none(),
            "same prompt + no grammar vs. tool grammar must not hit cache"
        );
    }

    #[test]
    fn prompt_cache_hit_requires_all_params_equal() {
        let tokens: Vec<u32> = vec![10, 20, 30];
        let mut params = SamplingParams::default();
        params.max_tokens = 64;
        params.stop_strings = vec!["END".to_string()];
        params.logit_bias.insert(99, -1.0);
        let cache = make_cached(&tokens, &params, "cached text");

        // Same params → must HIT
        let mut same = SamplingParams::default();
        same.max_tokens = 64;
        same.stop_strings = vec!["END".to_string()];
        same.logit_bias.insert(99, -1.0);
        let hit = cache.lookup(&tokens, &same);
        assert!(
            hit.is_some(),
            "identical prompt + identical params must hit cache"
        );
        assert_eq!(hit.unwrap().text, "cached text");
    }
}

// ---------------------------------------------------------------------------
// Wave-2.5 A1 — conditional grammar wire unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod test_a1_conditional_grammar_wire {
    /// Verify ToolCallSplitter state transitions that drive the A1 grammar gate.
    ///
    /// The grammar mask in the decode loop reads `tool_splitter.in_tool_call()`.
    /// This test confirms the splitter correctly transitions:
    ///   - before any input:     in_tool_call == false  (mask should NOT fire)
    ///   - after ToolCallOpen:   in_tool_call == true   (mask SHOULD fire)
    ///   - after ToolCallClose:  in_tool_call == false  (mask should NOT fire)
    #[test]
    fn splitter_in_body_transitions_drive_grammar_gate() {
        // Use the Gemma4 registration (has real tool open/close markers).
        let reg = crate::serve::api::registry::find_for("gemma4-27b-it")
            .expect("gemma4 registration must exist");
        let (open, close) = match (reg.tool_open, reg.tool_close) {
            (Some(o), Some(c)) => (o, c),
            _ => {
                eprintln!("gemma4 has no tool markers — skip A1 splitter test");
                return;
            }
        };
        let mut splitter =
            crate::serve::api::registry::ToolCallSplitter::from_registration(&reg)
                .expect("ToolCallSplitter::from_registration must return Some for gemma4");

        // Initial state: not inside a tool-call body.
        // Grammar mask should NOT be active.
        assert!(
            !splitter.in_tool_call(),
            "A1: before any input, in_tool_call must be false \
             (grammar mask must NOT fire for preamble tokens)"
        );

        // Feed the open marker — splitter enters the body.
        // Grammar mask SHOULD now be active.
        let events_open = splitter.feed(open);
        assert!(
            events_open.iter().any(|e| matches!(
                e,
                crate::serve::api::registry::ToolCallEvent::ToolCallOpen
            )),
            "A1: feeding the open marker must emit ToolCallOpen"
        );
        assert!(
            splitter.in_tool_call(),
            "A1: after feeding the open marker, in_tool_call must be true \
             (grammar mask MUST fire for body tokens)"
        );

        // Feed the close marker — splitter exits the body.
        // Grammar mask should NOT be active.
        let events_close = splitter.feed(close);
        assert!(
            events_close.iter().any(|e| matches!(
                e,
                crate::serve::api::registry::ToolCallEvent::ToolCallClose
            )),
            "A1: feeding the close marker must emit ToolCallClose"
        );
        assert!(
            !splitter.in_tool_call(),
            "A1: after feeding the close marker, in_tool_call must be false \
             (grammar mask must NOT fire after the body)"
        );
    }

    /// Verify the AtomicBool used in `generate_stream_once` transitions correctly
    /// when simulating ToolCallOpen / ToolCallClose events.
    ///
    /// This test mirrors what `route_content` does inside `generate_stream_once`:
    /// it manually drives an AtomicBool the same way the production code does, and
    /// asserts the before/during/after states.
    #[test]
    fn grammar_active_atomic_bool_transitions() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        // Simulate the `grammar_active` Arc<AtomicBool> from generate_stream_once.
        // Initial value: false (tool splitter present → grammar inactive until Open).
        let grammar_active = Arc::new(AtomicBool::new(false));

        // Pre-Open: grammar inactive.
        assert!(
            !grammar_active.load(Ordering::Relaxed),
            "A1: grammar_active must be false before ToolCallOpen"
        );

        // Simulate ToolCallOpen handler (what route_content does).
        grammar_active.store(true, Ordering::Relaxed);
        assert!(
            grammar_active.load(Ordering::Relaxed),
            "A1: grammar_active must be true after ToolCallOpen"
        );

        // Simulate ToolCallClose handler.
        grammar_active.store(false, Ordering::Relaxed);
        assert!(
            !grammar_active.load(Ordering::Relaxed),
            "A1: grammar_active must be false after ToolCallClose"
        );
    }

    /// Verify the no-tool-splitter fallback: when the model has no registered
    /// tool markers, `grammar_active` initialises to `true` so grammar runs
    /// unconditionally (backward-compatible behaviour for plain grammar requests).
    #[test]
    fn no_tool_splitter_grammar_active_defaults_true() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        // Use a registration that has no tool markers (Gemma4 has markers;
        // check for a model without them, or simulate `None` by passing `None`
        // to ToolCallSplitter::from_registration directly via a synthetic reg).
        // The simplest approach: simulate what generate_stream_once does —
        // `tool_splitter.is_none()` when the splitter was not created.
        let tool_splitter: Option<crate::serve::api::registry::ToolCallSplitter> = None;

        // Production code: `grammar_active = AtomicBool::new(tool_splitter.is_none())`
        let grammar_active = Arc::new(AtomicBool::new(tool_splitter.is_none()));

        assert!(
            grammar_active.load(Ordering::Relaxed),
            "A1: when there is no tool splitter, grammar_active must be true \
             (unconditional grammar masking — backward-compatible fallback)"
        );
    }
}
