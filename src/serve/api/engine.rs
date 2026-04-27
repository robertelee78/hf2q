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
/// Wave 3 W-B2 — `AutoLazyGrammar` variant added.  Once the W-B2 lazy
/// grammar wiring lands (`compile_tool_grammar` returns `Some(grammar)`
/// for `tool_choice=auto` with tools[] non-empty AND a registered
/// family), Auto becomes a constrained mode FROM `ToolCallOpen` ONWARDS.
/// The model is free to emit preamble content; once the open marker
/// fires the grammar enforces every body byte.  In that scenario a
/// body-parse-failure is structurally impossible — exactly the same
/// invariant that promotes Constrained's body-parse failure to a loud
/// error.  `AutoLazyGrammar` carries that contract through to the
/// streaming `emit_streaming_tool_call_close` and the non-streaming
/// `extract_tool_calls_from_text` so the loud-error promotion fires
/// equally on Required/Function AND Auto-with-grammar paths.
///
/// `Auto` (no grammar) keeps the content-fallback semantics — only fires
/// when the request is Auto AND tools[] is empty OR the model family is
/// unknown OR `tool_choice` was Auto and `compile_tool_grammar` returned
/// `Ok(None)`.  Under that branch there is no grammar enforcement, the
/// model may legitimately emit malformed syntax, and re-emitting the
/// raw bytes as Content is the right semantics.
///
/// # Why not derive from `schema::ToolChoiceValue` here
///
/// `SamplingParams` is an engine-layer type; it must not import from
/// `schema` (HTTP-layer).  This enum re-expresses only the
/// policy-relevant distinctions (Auto / AutoLazyGrammar / Constrained).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolCallPolicy {
    /// `tool_choice = "auto"` (or absent) AND no compiled grammar.  Grammar
    /// is optional under Auto, so this branch fires when:
    ///   * `tools[]` is empty / missing, OR
    ///   * the model family has no registered tool-call emitter, OR
    ///   * `tool_choice` was explicitly `"auto"` and the request never
    ///     declared tools.
    ///
    /// Body-parse failures fall back to Content (existing wave-2.5
    /// behaviour). Mirrors llama.cpp's unconstrained tool-call path.
    #[default]
    Auto,
    /// Wave 3 W-B2 — `tool_choice = "auto"` AND a lazy grammar IS active.
    /// `compile_tool_grammar` produced a `GrammarKind::ToolCallBodyAuto`
    /// runtime that is suspended (`awaiting_trigger=true`) until the
    /// `ToolCallSplitter` reports `ToolCallOpen`; from that point onwards
    /// the grammar enforces every body byte exactly like the Constrained
    /// path. A parse failure under this policy means the lazy grammar
    /// engine produced structurally invalid output — same regression
    /// signature as Constrained, same loud-error promotion required.
    ///
    /// Mirrors llama.cpp `grammar_lazy=true` at common/chat.cpp:898-913,
    /// 1177-1200, 1399-1416, 1626-1628.
    AutoLazyGrammar,
    /// `tool_choice = "required"` or `tool_choice = {type: "function", ...}`.
    /// Grammar guarantees well-formed output FROM BYTE 0; a parse failure is
    /// promoted to `GenerationEvent::Error` with `finish_reason = "error"`
    /// on the streaming path, and an HTTP 500 on the non-streaming path.
    Constrained,
}

impl ToolCallPolicy {
    /// `true` when the policy carries an active grammar that physically
    /// constrains the tool-call body (Constrained from byte 0, or
    /// AutoLazyGrammar from `ToolCallOpen` onwards).  Body-parse failures
    /// under these policies are unreachable in correct operation and
    /// promote to loud errors. Wave 3 W-B2 — single source of truth for
    /// the loud-error decision used by both streaming
    /// (`emit_streaming_tool_call_close`) and non-streaming
    /// (`extract_tool_calls_from_text`) paths.
    pub fn enforces_body_grammar(&self) -> bool {
        matches!(self, ToolCallPolicy::Constrained | ToolCallPolicy::AutoLazyGrammar)
    }
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
    /// Tool-call body grammar under `tool_choice = auto`.
    ///
    /// Applies ONLY after the model emits the per-model open marker
    /// (e.g. Gemma 4 `<|tool_call>`, Qwen 3.5/3.6 `<tool_call>`).  Until
    /// then, the runtime sits in `awaiting_trigger == true`: `apply()`
    /// is a no-op (no mask), `accept()` is a no-op (no advance, no
    /// dead/accepted-state termination check).  When the
    /// ToolCallSplitter sees the open marker, the engine calls
    /// `runtime.trigger()` to flip the flag false; the runtime then
    /// enforces every subsequent token through to the close marker.
    ///
    /// Mirrors llama.cpp `grammar_lazy = true` for `tool_choice == AUTO`
    /// at `/opt/llama.cpp/common/chat.cpp:913, 1200, 1416`.
    ///
    /// NOTE — wave 2.7 W-η has not yet wired AUTO to a marker-aware lazy
    /// grammar (`compile_tool_grammar` only fires for Required/Function).
    /// This variant is forward-looking; today it is unreachable from the
    /// chat-completions path.
    ToolCallBodyAuto,
    /// Tool-call body grammar under `tool_choice = required` or
    /// `tool_choice = function(name)`.
    ///
    /// EAGER from token 0 — the runtime never enters `awaiting_trigger`
    /// state; the mask fires every step starting at the very first
    /// decode token.  The grammar root is shape `OneOrMoreCalls` with the
    /// per-model open marker, body, and close marker all wired into the
    /// root rule (see `registry::GrammarShape`).  The model is
    /// structurally unable to emit non-call output: byte 0 must be the
    /// first byte of the open marker (e.g. `<` for Gemma 4 `<|tool_call>`)
    /// or the request rejects every other token via the mask.
    ///
    /// Mirrors llama.cpp `grammar_lazy = false` for
    /// `tool_choice == REQUIRED` at `/opt/llama.cpp/common/chat.cpp:898-913,
    /// 1177-1200, 1399-1416`.  Wave 2.7 W-η HIGH-1.
    ToolCallBodyRequired,
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
    ///   * `compile_tool_grammar`    → `GrammarKind::ToolCallBodyRequired`
    ///                                 (or `ToolCallBodyAuto` when wave 2.7+
    ///                                  wires AUTO to a marker-aware lazy
    ///                                  grammar — not yet reachable)
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
/// Wave-2.6 W-ε (B5 honest closure): wave-2.5 commit overstated B5
/// closure.  The key still excluded `frequency_penalty`, `presence_penalty`,
/// `min_p`, `grammar_kind`, `tool_call_policy`, `logprobs`,
/// `top_logprobs`, and `parallel_tool_calls`.  Option A (mantra): every
/// generation-affecting parameter is included in the key — even parameters
/// not yet wired into the sampler — so that future wiring never introduces
/// a silent stale-replay bug.
///
/// Inventory of ALL `SamplingParams` fields and their cache treatment:
///
/// **Excluded (bypass gate already handles these):**
/// - `temperature` — non-zero bypasses cache; never reaches key check
/// - `top_p` — < 1.0 bypasses cache
/// - `top_k` — > 0 bypasses cache
/// - `repetition_penalty` — ≠ 1.0 bypasses cache
/// - `seed` — Some(_) bypasses cache
/// - `token_bytes` — derived from `grammar`; identical iff `grammar` is identical
///
/// **Included (affect model output or response shape):**
/// - `max_tokens` — early-stop trigger
/// - `stop_strings` — early-stop trigger
/// - `logit_bias` — additive shift applied before argmax (wired)
/// - `grammar` — token-validity mask (wired)
/// - `grammar_kind` — ResponseFormat vs ToolCallBody changes enforcement
///   timing; wired in wave-2.6 W-α5 (same grammar, different kind →
///   completely different output for tool-call vs. unconditional paths)
/// - `frequency_penalty` — penalty applied to sampler (plumbed, not yet wired
///   into greedy path; included now so future wiring is safe)
/// - `presence_penalty` — same as frequency_penalty
/// - `min_p` — min-p sampling cutoff (plumbed, not yet wired; included for
///   forward-compatibility)
/// - `tool_call_policy` — Auto vs Constrained changes error-promotion on
///   parse failure; a cached Auto replay served to a Constrained caller
///   would silently suppress error signalling
/// - `logprobs` — changes response shape (logprob data in choices)
/// - `top_logprobs` — changes response shape (number of top alternatives)
/// - `parallel_tool_calls` — plumbed, not yet wired; included for
///   forward-compatibility
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
    /// ResponseFormat vs ToolCallBody — same Grammar but different kind
    /// produces different enforcement timing and therefore different output.
    /// Wired in wave-2.6 W-α5.
    pub grammar_kind: GrammarKind,
    /// Stored as bit-pattern to allow structural equality without f32 surprises.
    /// Default 0.0 → 0u32. Plumbed but not yet wired into greedy path;
    /// included now for forward-safe wiring.
    pub frequency_penalty_bits: u32,
    /// Same treatment as `frequency_penalty_bits`.
    pub presence_penalty_bits: u32,
    /// Min-p sampling cutoff bit-pattern. 0.0 → 0u32.
    pub min_p_bits: u32,
    /// Auto vs Constrained — affects error-promotion on parse failure.
    pub tool_call_policy: ToolCallPolicy,
    /// `true` = include per-token logprob data in response. Changes response
    /// shape; different callers expect different response structures.
    pub logprobs: bool,
    /// Number of top-alternatives to report per token. Changes response shape.
    pub top_logprobs: u32,
    /// Multi-tool-call flag. Plumbed, not yet wired; included for
    /// forward-safe wiring.
    pub parallel_tool_calls: bool,
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
            grammar_kind: params.grammar_kind,
            frequency_penalty_bits: params.frequency_penalty.to_bits(),
            presence_penalty_bits: params.presence_penalty.to_bits(),
            min_p_bits: params.min_p.to_bits(),
            tool_call_policy: params.tool_call_policy,
            logprobs: params.logprobs,
            top_logprobs: params.top_logprobs,
            parallel_tool_calls: params.parallel_tool_calls,
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
                grammar_kind: GrammarKind::default(),
                frequency_penalty_bits: 0u32,
                presence_penalty_bits: 0u32,
                min_p_bits: 0u32,
                tool_call_policy: ToolCallPolicy::Auto,
                logprobs: false,
                top_logprobs: 0,
                parallel_tool_calls: true,
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
    /// cached key.
    ///
    /// Wave-2.5 B5 / Wave-2.6 W-ε: `PromptCacheKey` now covers the
    /// complete inventory of generation-affecting params:
    /// `max_tokens`, `stop_strings`, `logit_bias`, `grammar`,
    /// `grammar_kind`, `frequency_penalty`, `presence_penalty`,
    /// `min_p`, `tool_call_policy`, `logprobs`, `top_logprobs`,
    /// `parallel_tool_calls`.  See `PromptCacheKey` doc for rationale.
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
    //
    // Wave 2.6 W-α5 Q2: when `params.grammar_kind == ToolCallBody`, the
    // runtime starts SUSPENDED via `set_awaiting_trigger(true)`.  The
    // mask + accept calls below are unconditional — the runtime
    // self-gates internally (mirrors llama.cpp lazy-grammar pattern at
    // /opt/llama.cpp/src/llama-grammar.cpp:1287-1344, citation in
    // research-report.md Q2).  The trigger flips when the
    // `ToolCallSplitter` sees the per-model open marker (handler
    // below).  For `GrammarKind::ResponseFormat` the runtime starts
    // EAGER — enforcement from token 0, byte-identical to pre-A1
    // behavior.  This is the wave-2.5 audit divergence A1 fix.
    let mut grammar_runtime: Option<super::grammar::GrammarRuntime> = match params.grammar.as_ref() {
        Some(g) => {
            let start_rule_id = g
                .rule_id("root")
                .ok_or_else(|| anyhow::anyhow!("grammar has no root rule"))?;
            let mut rt = super::grammar::GrammarRuntime::new(g.clone(), start_rule_id)
                .ok_or_else(|| anyhow::anyhow!("grammar runtime init failed"))?;
            // Wave 2.7 W-η Q-A: only `ToolCallBodyAuto` arms the lazy
            // (awaiting_trigger) gate.  `ToolCallBodyRequired` is EAGER
            // from token 0 — the grammar root already wraps the body in
            // open/close markers, so the mask must fire at byte 0 and
            // reject any token whose decoded bytes don't prefix the
            // open marker.  Mirrors llama.cpp `grammar_lazy = false` for
            // `tool_choice == REQUIRED` at common/chat.cpp:898-913,
            // 1177-1200, 1399-1416.
            if matches!(params.grammar_kind, GrammarKind::ToolCallBodyAuto) {
                rt.set_awaiting_trigger(true);
            }
            Some(rt)
        }
        None => None,
    };
    let token_bytes_ref: Option<&[Vec<u8>]> = params.token_bytes.as_deref().map(|v| &v[..]);

    // Wave-2.5 A1 / Wave 2.6 W-α5 Q2: ToolCallSplitter for the
    // non-streaming decode loop.  Used here ONLY to detect the per-model
    // open marker so we can call `runtime.trigger()` on the grammar — the
    // runtime then self-gates (no separate `in_body` boolean needed).
    // For `GrammarKind::ResponseFormat` runtimes the trigger is a no-op
    // because the runtime was constructed eager.  The splitter is `None`
    // when the model has no tool markers registered; in that case the
    // runtime never gets a trigger event but is also never suspended
    // (ResponseFormat default, or ToolCallBody on an unregistered model
    // which compile_tool_grammar refuses upstream).
    let mut tc_splitter_ns: Option<super::registry::ToolCallSplitter> =
        registration.and_then(|r| super::registry::ToolCallSplitter::from_registration(r));

    // Local helper — apply grammar mask + Tier 4 logit_bias and sample.
    // Mutably borrows the runtime so it can be advanced after sampling
    // (caller does the advance to keep this closure side-effect-light).
    // Returns the sampled token id; caller must feed
    // `token_bytes[id]` through the runtime to keep it in sync.
    //
    // Wave 2.6 W-α5 Q2: the mask call is UNCONDITIONAL.  The runtime
    // self-gates via `is_awaiting_trigger()` inside
    // `mask::mask_invalid_tokens` — when suspended (ToolCallBody
    // pre-trigger), the function early-returns 0 and leaves logits
    // untouched.  This removes the wave-2.5 `if in_tool_body { mask }`
    // wrapper and the sibling `Arc<AtomicBool>` it implied — exactly
    // the architecture the audit caught at engine.rs:1401, 1489, etc.
    let sample_from_live_logits =
        |weights: &mut MlxModelWeights,
         generated: &[u32],
         runtime: Option<&super::grammar::GrammarRuntime>|
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
            // dead.  Self-gates on `runtime.is_awaiting_trigger()` —
            // suspended runtimes mask zero tokens (preamble freedom for
            // ToolCallBody-kind grammars before the open marker fires).
            if let (Some(rt), Some(tb)) = (runtime, token_bytes_ref) {
                super::grammar::mask::mask_invalid_tokens(rt, tb, &mut logits);
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
    // Wave 2.6 W-α5 Q2: the mask + accept calls are UNCONDITIONAL.  For
    // `GrammarKind::ToolCallBodyAuto` the runtime is suspended
    // (`is_awaiting_trigger() == true`) so both calls are no-ops and the
    // first token is naturally unconstrained — the same behavior the
    // wave-2.5 explicit `in_body_first = false` short-circuit
    // produced, but achieved structurally via the runtime self-gate.
    // For `GrammarKind::ResponseFormat` and Wave 2.7 W-η Q-A's
    // `ToolCallBodyRequired` the runtime enforces from token 0
    // (response_format fixes audit divergence A1; Required eagerly
    // constrains the model to emit a tool call from byte 0).
    let mut next_token = if sample_logits {
        let tok = sample_from_live_logits(&mut loaded.weights, &[], grammar_runtime.as_ref())?;
        // Feed the chosen token's bytes through the grammar runtime so
        // the next step's mask is correctly narrowed.  No-op when no
        // grammar OR when the runtime is awaiting trigger (suspended
        // runtime self-gates).  Empty token_bytes (special/unprintable)
        // is also skipped — accept_bytes on empty is a true no-op.
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
    // Wave 2.6 W-α5 Q2: feed first fragment through the tool-call
    // splitter; if it emits a `ToolCallOpen` event, trigger the grammar
    // runtime so subsequent tokens are constrained by the body grammar.
    // (Typically the first decoded token is never the open marker, but
    // this keeps the state machine correct for any edge case where the
    // chat template ends mid-marker.)  llama.cpp does NOT reset the
    // trigger on close — multi-call support comes from the grammar
    // shape `(call)+`.  See research-report.md Q2 anti-finding +
    // /opt/llama.cpp/docs/function-calling.md.
    if let Some(tcs) = tc_splitter_ns.as_mut() {
        let events = tcs.feed(&first_fragment);
        if let Some(rt) = grammar_runtime.as_mut() {
            if events.iter().any(|e| matches!(e, super::registry::ToolCallEvent::ToolCallOpen)) {
                rt.trigger();
            }
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
                // grammar mask, then call sampler_pure for
                // temperature/top_p/top_k/rep-penalty.
                //
                // Wave 2.6 W-α5 Q2: mask + accept calls are
                // UNCONDITIONAL.  The runtime self-gates via
                // `is_awaiting_trigger()`; suspended runtimes (lazy
                // tool-call body grammar pre-trigger) mask zero tokens
                // and ignore advance, so preamble emission is naturally
                // unconstrained.  Eager runtimes (ResponseFormat)
                // enforce every step.  This removes the wave-2.5 sibling
                // `Arc<AtomicBool>` and the `if in_body { mask }` /
                // `if in_body { accept }` split that the audit caught at
                // engine.rs:1401, 1489.
                let tok = sample_from_live_logits(
                    &mut loaded.weights,
                    &generated_tokens,
                    grammar_runtime.as_ref(),
                )?;
                // Advance the grammar runtime by the chosen token's bytes.
                // Self-gates internally — see GrammarRuntime::accept_bytes.
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
            // Wave 2.6 W-α5 Q2: feed the splitter; if it emits a
            // ToolCallOpen on this fragment, trigger the grammar runtime
            // so subsequent decode steps enforce the body grammar.
            // ToolCallClose does NOT reset the trigger — multi-call
            // grammars handle re-entry via `(call)+` shape (research-
            // report.md Q2; /opt/llama.cpp/docs/function-calling.md).
            if let Some(tcs) = tc_splitter_ns.as_mut() {
                let events = tcs.feed(&fragment);
                if let Some(rt) = grammar_runtime.as_mut() {
                    if events.iter().any(|e| matches!(e, super::registry::ToolCallEvent::ToolCallOpen)) {
                        rt.trigger();
                    }
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

/// Outcome of `finalize_streaming_tool_state` — tells the streaming
/// driver whether to proceed to `Done`, abort silently (client gone),
/// or skip `Done` because a structured `Error` event has already been
/// emitted by the helper. Wave 2.8 W-θ HIGH-1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinalizeStreamingAction {
    /// Drain succeeded (or was a no-op). Caller proceeds to emit Done.
    Continue,
    /// `events.blocking_send` returned Err while emitting a tail Content
    /// delta. SSE receiver is gone; caller should abort silently (no Done).
    ClientDropped,
    /// Helper emitted a `GenerationEvent::Error` (Constrained mid-call
    /// truncation or no-call). Caller MUST skip Done — the SSE encoder
    /// produces the final error chunk on receipt of Error.
    ErrorEmitted,
}

/// Drain the tool-call splitter tail at end-of-stream and then enforce the
/// grammar-active-policy safety nets before the streaming `Done` event.
///
/// Wave 2.8 W-θ HIGH-1 — streaming companion to the non-streaming
/// defensive 500 in `handlers.rs:410-444` (commit da545d5).
/// Wave 3 W-B2 — `AutoLazyGrammar` joins `Constrained` in the loud-error
/// branch (mid-call truncation only; no-call check stays Constrained-only
/// because Auto-lazy explicitly allows the model to emit zero calls).
///
/// Behaviour matrix (`policy` = `tool_call_policy`):
/// ```text
///                       │ Auto (no grammar)              │ AutoLazyGrammar / Constrained
/// ─────────────────────┼─────────────────────────────────┼────────────────────────────────
/// finish() = Content   │ emit Content                    │ emit Content
/// finish() = TC-text   │ emit Content (open-marker re-   │ emit GenerationEvent::Error
///                       │   prepended for clarity)        │   "tool_call_truncated_under_constrained"
///                       │                                 │   → ErrorEmitted
/// post-drain no call   │ no action                       │ Constrained ONLY: emit Error
///                       │                                 │   "tool_call_no_call_under_constrained"
///                       │                                 │ AutoLazyGrammar: no action
///                       │                                 │   (Auto explicitly allows no-call)
/// ```
///
/// **Why mid-call truncation errors under both Constrained AND
/// AutoLazyGrammar**: in either case the grammar is active inside the
/// tool-call body. A truncation past the open marker but before the
/// close marker means decoding stopped mid-grammar — the runtime is
/// neither `is_accepted()` nor `is_dead()` and the body bytes captured
/// so far cannot be parsed into a tool call. Same regression signature
/// as Constrained truncation; same loud-error promotion.
///
/// **Why no-call check stays Constrained-only**: Auto explicitly permits
/// the model to emit zero tool calls (preamble freedom — the whole
/// point of lazy grammar). A streaming run that ended without ever
/// firing `ToolCallOpen` is the legitimate Auto-no-call path, NOT a
/// regression. Required/Function on the other hand mandate at least
/// one call (the eager grammar root accepts only `OneOrMoreCalls`),
/// so a no-call run there means max_tokens cut the call mid-emission
/// or the grammar emitter has a bug.
///
/// Extracted from the inline finalize block so the audit-driver test for
/// HIGH-1 can exercise this exact code path. See
/// `finalize_streaming_tool_state_tests` below.
fn finalize_streaming_tool_state(
    tool_splitter: Option<&mut super::registry::ToolCallSplitter>,
    policy: ToolCallPolicy,
    saw_tool_call: bool,
    registration: Option<&super::registry::ModelRegistration>,
    completion_tokens: usize,
    accumulated_text_len: usize,
    events: &mpsc::Sender<super::sse::GenerationEvent>,
) -> FinalizeStreamingAction {
    use super::sse::{DeltaKind, GenerationEvent};

    // Wave 3 W-B2: mid-call truncation fires for any policy carrying an
    // active body grammar (Constrained from byte 0, AutoLazyGrammar from
    // ToolCallOpen onwards). The single source of truth lives in
    // `ToolCallPolicy::enforces_body_grammar`.
    let body_grammar_active = policy.enforces_body_grammar();
    // Wave 3 W-B2: no-call check stays Constrained-only; AutoLazyGrammar
    // explicitly allows the model to emit zero calls (preamble freedom).
    let policy_constrained = matches!(policy, ToolCallPolicy::Constrained);

    if let Some(tcs) = tool_splitter {
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
                        return FinalizeStreamingAction::ClientDropped;
                    }
                }
                super::registry::ToolCallEvent::ToolCallText(t) => {
                    if body_grammar_active {
                        // HIGH-1 streaming companion to da545d5 + Wave 3
                        // W-B2: emit a structured error event INSTEAD of
                        // silently re-emitting the residual as Content.
                        // Fires for Constrained AND AutoLazyGrammar.
                        let policy_label = match policy {
                            ToolCallPolicy::Constrained => "required/function",
                            ToolCallPolicy::AutoLazyGrammar => {
                                "auto (lazy grammar active)"
                            }
                            ToolCallPolicy::Auto => {
                                unreachable!("body_grammar_active gate")
                            }
                        };
                        tracing::error!(
                            residual_len = t.len(),
                            policy = policy_label,
                            "tool_call_truncated_under_constrained: streaming \
                             ended mid-tool-call (open marker observed, no close \
                             marker) under tool_choice={}; per-model body \
                             grammar should have prevented this",
                            policy_label
                        );
                        let _ = events.blocking_send(GenerationEvent::Error(
                            "tool_call_truncated_under_constrained".into(),
                        ));
                        return FinalizeStreamingAction::ErrorEmitted;
                    }
                    // Auto (no grammar): legacy behaviour — emit residual
                    // body as Content with the literal open marker
                    // re-prepended for diagnostic clarity (the splitter
                    // swallowed the open marker when it flipped state).
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
                        return FinalizeStreamingAction::ClientDropped;
                    }
                }
                super::registry::ToolCallEvent::ToolCallOpen
                | super::registry::ToolCallEvent::ToolCallClose => {
                    // unreachable: finish() never emits Open/Close.
                }
            }
        }
    }

    // Post-drain no-call check — streaming companion to handlers.rs:410-444.
    // Constrained ONLY (Required/Function): grammar root mandates
    // OneOrMoreCalls; a no-call run is a regression. AutoLazyGrammar
    // explicitly permits no-call (the whole point of lazy grammar is
    // preamble freedom + optional emission).
    if policy_constrained && !saw_tool_call {
        tracing::error!(
            completion_tokens = completion_tokens,
            text_len = accumulated_text_len,
            "tool_call_no_call_under_constrained: streaming ended with zero \
             tool calls under tool_choice=required/function; eager grammar \
             should have prevented this — either max_tokens cut a call \
             mid-emission or the grammar emitter has a bug"
        );
        let _ = events.blocking_send(GenerationEvent::Error(
            "tool_call_no_call_under_constrained".into(),
        ));
        return FinalizeStreamingAction::ErrorEmitted;
    }

    FinalizeStreamingAction::Continue
}

/// Wave 3 W-B3 — T2.3 incremental tool-call argument streaming.
///
/// Per OpenAI Chat Completions streaming spec, `delta.tool_calls[N].function.arguments`
/// is a *string accumulator* on the client side — clients append each arg-delta to
/// the previous, then `JSON.parse(accumulated)` once the chunk with `finish_reason ==
/// "tool_calls"` arrives. Pre-W-B3, the streaming engine accumulated the entire
/// per-family body (Gemma `<|tool_call>...<tool_call|>`, Qwen `<tool_call>...</tool_call>`)
/// into `tool_call_body`, then on `ToolCallClose` parsed it and emitted a SINGLE
/// arguments delta carrying the full JSON. Spec-valid, but a UI cannot show
/// progressive tool-call args while the model is still emitting them.
///
/// The emitter wires in three places inside the `route_content` closure:
///
///   - **`ToolCallOpen`**: construct a fresh `ToolCallStreamEmitter` for this call.
///   - **`ToolCallText(t)`**: after appending `t` to `tool_call_body`, call
///     `emitter.advance(body, events)` which:
///       1. Emits the **first chunk** as soon as the function name is parseable
///          from the body prefix (`{index, id, type:"function", function:{name}}`).
///       2. After the first chunk fires, emits the JSON args opening `{` as the
///          first `arguments` delta — clients begin accumulating the JSON string.
///       3. For each newly-closed kv pair (Gemma: top-level `,` or `}`; Qwen:
///          `</parameter>` block boundary), emits `,"key":<jsonval>` as a fresh
///          `arguments` delta (no leading `,` for the first kv).
///   - **`ToolCallClose`**: call `emitter.finalize(body, events, ...)` which:
///       - On the happy path (incremental emission started + body re-parses),
///         emits any tail kvs that the streaming scanner missed (last one before
///         the closer) and the closing `}`. `tc_index` increments here.
///       - On the fallback path (incremental emission never started — body parsed
///         OK but arrived in one fragment short of name extraction; OR the family
///         has no streaming converter; OR partial extraction failed mid-stream),
///         delegates to `emit_streaming_tool_call_close` which preserves the
///         pre-W-B3 close-buffered shape AND the policy-enforced loud-error
///         branches.
///
/// # Tail-parser design
///
/// The streaming scanner walks `body[scan_cursor..]` and extracts kv pairs at
/// JSON-syntactically-meaningful boundaries — closed string values, terminated
/// bare numerics, closed `<parameter>` blocks. It NEVER emits mid-string or
/// mid-key. The grammar runtime (eager from W-η for Constrained / lazy from
/// W-B2 for AutoLazyGrammar) physically guarantees the body bytes are
/// well-formed at every prefix the scanner inspects, so partial-prefix parse
/// failures inside the scanner are a grammar-engine bug — surfaced by leaving
/// `kvs_emitted` short, which forces the close-time `finalize` to fall through
/// to the close-buffered path and trigger the existing loud-error branch.
///
/// # Why NOT add a "speculative close + diff" approach
///
/// Considered: append the family's expected closer to `body`, re-parse with
/// `parse_tool_call_body`, diff against the last successful args-JSON, emit
/// the new tail. Rejected because:
///   - String values would emit STALE partials. `body = "call:f{loc:<|\"|>San Fra"`
///     speculatively closes to `{"loc":"San Fra"}`; clients would see `"San Fra"`
///     which then disagrees with the final `"San Francisco"`. OpenAI clients
///     concatenate without dedup — they would receive `"San FraSan Francisco"`.
///   - The closed-kv scanner is structurally simpler AND emits only at boundaries
///     that JSON treats as values committed (the previous kv + comma terminator).
///
/// # Backward compatibility
///
/// Single-chunk emission still works: clients that don't care about progressive
/// UI updates simply concatenate any number of `arguments` deltas and JSON-parse
/// the result. The spec does not bound how many deltas a server emits per call.
/// The first-chunk-has-name + finish_reason="tool_calls"-on-terminal contract
/// is preserved on both shapes.
struct ToolCallStreamEmitter {
    /// `gemma4` / `qwen35` / unknown. Unknown families skip the streaming path
    /// (the close-time fallback handles them).
    family: Option<&'static str>,
    /// `delta.tool_calls[N].index` — pre-incremented from the per-stream counter
    /// at construction. Stable across all chunks for THIS call.
    index: usize,
    /// Synthesized opaque identifier emitted in the first chunk. Cached so
    /// `finalize` can reuse it on the close-buffered fallback if the streaming
    /// path never fired (we never emitted the first chunk under that branch
    /// either, so the cached id stays unused — kept for symmetry).
    id: String,
    /// Whether the first chunk (id+type+name) has been emitted. Latched true.
    name_emitted: bool,
    /// Whether the args opening `{` has been emitted as the first arguments
    /// delta. Latched true after `name_emitted` flips and the first kv-emit
    /// or `finalize` runs (clients need the `{` before any kv content).
    args_open_emitted: bool,
    /// Number of top-level kv pairs already emitted to the client. Drives the
    /// leading-comma decision (no comma for the first kv).
    kvs_emitted: usize,
    /// Byte cursor into `tool_call_body` — bytes < cursor have been scanned
    /// for kv-emission. Bytes >= cursor are unscanned (may contain a
    /// completed-but-not-yet-emitted kv OR a partial kv in progress).
    scan_cursor: usize,
}

impl ToolCallStreamEmitter {
    /// Construct an emitter for a tool-call span starting at `tc_index`. The
    /// caller is responsible for passing the SAME `tc_index` to `finalize`'s
    /// fallback path so the close-buffered shape stays aligned when the
    /// streaming converter declines.
    fn new(family: Option<&'static str>, tc_index: usize) -> Self {
        let id = format!(
            "call_hf2q_{:016x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
                ^ (tc_index as u64).wrapping_mul(0x9e3779b97f4a7c15)
        );
        Self {
            family,
            index: tc_index,
            id,
            name_emitted: false,
            args_open_emitted: false,
            kvs_emitted: 0,
            scan_cursor: 0,
        }
    }

    /// Advance the emitter against the current `body` after a `ToolCallText`
    /// fragment has been appended. Emits any newly-extractable name / kv
    /// fragments. Idempotent on repeated calls with the same body — safe to
    /// invoke even when no new bytes arrived.
    fn advance(
        &mut self,
        body: &str,
        events: &tokio::sync::mpsc::Sender<super::sse::GenerationEvent>,
    ) -> Result<(), ()> {
        use super::sse::GenerationEvent;

        // Step 1: emit the name + opening chunk if not yet done.
        if !self.name_emitted {
            let name = match self.family {
                Some("gemma4") => extract_gemma4_name_prefix(body),
                Some("qwen35") => extract_qwen35_name_prefix(body),
                _ => None,
            };
            let Some((name, header_end)) = name else {
                return Ok(());
            };
            // First chunk: index + id + type + name. arguments omitted —
            // clients see `function.name` complete on chunk 1 per spec.
            if events
                .blocking_send(GenerationEvent::ToolCallDelta {
                    index: self.index,
                    id: Some(self.id.clone()),
                    call_type: Some("function".into()),
                    name: Some(name),
                    arguments: None,
                })
                .is_err()
            {
                return Err(());
            }
            self.name_emitted = true;
            self.scan_cursor = header_end;
        }

        // Step 2: emit the opening `{` of the args object (once).
        if !self.args_open_emitted {
            if events
                .blocking_send(GenerationEvent::ToolCallDelta {
                    index: self.index,
                    id: None,
                    call_type: None,
                    name: None,
                    arguments: Some("{".into()),
                })
                .is_err()
            {
                return Err(());
            }
            self.args_open_emitted = true;
        }

        // Step 3: scan body[scan_cursor..] for newly-closed kv pairs.
        match self.family {
            Some("gemma4") => self.scan_emit_gemma4_kvs(body, events)?,
            Some("qwen35") => self.scan_emit_qwen35_kvs(body, events)?,
            _ => {}
        }
        Ok(())
    }

    /// Scan-and-emit closed Gemma 4 kvs from `body[scan_cursor..]`. A kv is
    /// "closed" when we observe its terminator at top level — `,` for
    /// non-final kvs, `}` for the final one. We emit only on `,` boundaries
    /// during streaming; the trailing `}` is finalize's job (the last kv
    /// before `}` may not yet be in the body, so we can't speculate).
    fn scan_emit_gemma4_kvs(
        &mut self,
        body: &str,
        events: &tokio::sync::mpsc::Sender<super::sse::GenerationEvent>,
    ) -> Result<(), ()> {
        use super::sse::GenerationEvent;
        // Walk from scan_cursor, tracking `<|"|>` string state. On a top-level
        // `,` after `scan_cursor`, parse the kv span [scan_cursor..comma] and
        // emit. Advance scan_cursor past the comma.
        let bytes = body.as_bytes();
        let mut in_str = false;
        let mut i = self.scan_cursor;
        let kv_start = self.scan_cursor;
        let mut last_kv_start = kv_start;
        while i < bytes.len() {
            if !in_str && bytes[i..].starts_with(b"<|\"|>") {
                in_str = true;
                i += 5;
                continue;
            }
            if in_str && bytes[i..].starts_with(b"<|\"|>") {
                in_str = false;
                i += 5;
                continue;
            }
            if !in_str && bytes[i] == b',' {
                let kv = &body[last_kv_start..i];
                if let Some(json) = gemma4_kv_to_json(kv) {
                    let prefix = if self.kvs_emitted == 0 { "" } else { "," };
                    let frag = format!("{prefix}{json}");
                    if events
                        .blocking_send(GenerationEvent::ToolCallDelta {
                            index: self.index,
                            id: None,
                            call_type: None,
                            name: None,
                            arguments: Some(frag),
                        })
                        .is_err()
                    {
                        return Err(());
                    }
                    self.kvs_emitted += 1;
                    last_kv_start = i + 1;
                    self.scan_cursor = i + 1;
                }
                i += 1;
                continue;
            }
            // `}` at top level marks the args object's close. Stop scanning —
            // finalize will emit the trailing kv (if any) + `}`. Don't
            // speculatively emit on `}` because the body may still gain bytes
            // (sticky tool_close marker streamed separately).
            if !in_str && bytes[i] == b'}' {
                break;
            }
            i += 1;
        }
        Ok(())
    }

    /// Scan-and-emit closed Qwen 3.5/3.6 `<parameter=KEY>VAL</parameter>`
    /// blocks from `body[scan_cursor..]`. A block is "closed" when we observe
    /// `</parameter>` after its opening tag. Emits one delta per closed block.
    fn scan_emit_qwen35_kvs(
        &mut self,
        body: &str,
        events: &tokio::sync::mpsc::Sender<super::sse::GenerationEvent>,
    ) -> Result<(), ()> {
        use super::sse::GenerationEvent;
        loop {
            // Locate the next `<parameter=` in body[scan_cursor..].
            let rest = &body[self.scan_cursor..];
            let Some(rel_open) = rest.find("<parameter=") else { break };
            let p_open = self.scan_cursor + rel_open;
            let key_start = p_open + "<parameter=".len();
            let Some(rel_gt) = body[key_start..].find('>') else { break };
            let key_end = key_start + rel_gt;
            let val_start = key_end + 1;
            let Some(rel_close) = body[val_start..].find("</parameter>") else {
                break;
            };
            let val_end = val_start + rel_close;
            let after_close = val_end + "</parameter>".len();
            let key = body[key_start..key_end].trim();
            let val_raw = body[val_start..val_end].trim();
            if key.is_empty() {
                // Malformed — leave scan_cursor where it is so finalize can
                // exercise the close-buffered loud-error branch under
                // policy.enforces_body_grammar(). Stop streaming this block.
                break;
            }
            let json_val: serde_json::Value = match serde_json::from_str(val_raw) {
                Ok(v) => v,
                Err(_) => serde_json::Value::String(val_raw.to_string()),
            };
            let key_json = serde_json::to_string(key).unwrap_or_else(|_| format!("\"{key}\""));
            let val_json = serde_json::to_string(&json_val)
                .unwrap_or_else(|_| "null".to_string());
            let prefix = if self.kvs_emitted == 0 { "" } else { "," };
            let frag = format!("{prefix}{key_json}:{val_json}");
            if events
                .blocking_send(GenerationEvent::ToolCallDelta {
                    index: self.index,
                    id: None,
                    call_type: None,
                    name: None,
                    arguments: Some(frag),
                })
                .is_err()
            {
                return Err(());
            }
            self.kvs_emitted += 1;
            self.scan_cursor = after_close;
        }
        Ok(())
    }

    /// Emit the close-time tail. Two routes:
    ///
    ///   - **Streaming path was active** (`name_emitted == true`): re-parse
    ///     the full body to recover the last (un-streamed) kv plus the args
    ///     close. Emit the residual JSON tail and the closing `}` as one
    ///     final arguments delta. Increment `tc_index` and set `saw_tc`.
    ///     Returns `Ok(())`.
    ///
    ///     Wave 3.5 MED honesty note: this branch fires for any
    ///     well-formed body whose first `advance` call could extract
    ///     the function name from a prefix — INCLUDING single-fragment
    ///     bodies where the entire `call:NAME{...}` (Gemma 4) or
    ///     `<function=NAME>...</function>` (Qwen 3.5/3.6) arrived in
    ///     one fragment.  The audit at
    ///     `/tmp/cfa-cfa-20260427-adr005-wave3/codex-review-last.txt`
    ///     (divergence "W-B3 single-fragment fallback" severity MED)
    ///     correctly observed that no "single-fragment legacy
    ///     fallback" exists for well-formed bodies — `advance` always
    ///     emits chunk 1 (id+name) + chunk 2 (`{`) immediately on
    ///     well-formed input.  The incremental shape IS the canonical
    ///     OpenAI streaming contract; there is no client-visible
    ///     "two-chunk close-buffered" shape for well-formed
    ///     single-fragment bodies under Wave 3 W-B3 + later.
    ///
    ///   - **Streaming path never fired** (`name_emitted == false`): the
    ///     emitter declined the body (unknown family, OR name didn't appear
    ///     in any prefix). Delegate to `emit_streaming_tool_call_close` so
    ///     the close-buffered shape AND the policy-enforced loud-error
    ///     branches stay byte-for-byte identical to pre-W-B3 behaviour.
    ///     This branch is exercised by
    ///     `streaming_unknown_family_falls_back_to_legacy`.
    fn finalize(
        &mut self,
        body: String,
        registration: Option<&super::registry::ModelRegistration>,
        policy: ToolCallPolicy,
        tc_index: &mut usize,
        saw_tc: &mut bool,
        events: &tokio::sync::mpsc::Sender<super::sse::GenerationEvent>,
    ) -> Result<(), ()> {
        use super::sse::GenerationEvent;

        if !self.name_emitted {
            // Fallback: streaming path never fired. Use the legacy close-
            // buffered emit so policy-enforced loud-error branches and the
            // single-chunk shape both stay intact.
            let parsed = registration
                .and_then(|r| super::registry::parse_tool_call_body(r, &body));
            return emit_streaming_tool_call_close(
                parsed, body, policy, tc_index, saw_tc, events,
            );
        }

        // Streaming path was active. Re-parse the now-complete body and emit
        // the tail (last kv we couldn't stream because we couldn't
        // distinguish "final kv" from "next kv arriving later") + the
        // closing `}`.
        let parsed = registration
            .and_then(|r| super::registry::parse_tool_call_body(r, &body));
        let Some(pc) = parsed else {
            // Body failed parse despite streaming having extracted the name.
            // Under policy.enforces_body_grammar(), this means the grammar
            // engine produced bytes the per-family parser can't reassemble —
            // an unreachable-fallback regression. Promote to loud Error.
            // Under Auto (no grammar), preserve content fallback semantics
            // by closing the streaming JSON args we already emitted with `}`
            // (so the client's accumulator is at least valid JSON for the
            // partial it received) and then NOT emitting the residue as a
            // re-content delta — the partial args we streamed are the
            // semantically-faithful slice we managed to extract.
            if policy.enforces_body_grammar() {
                tracing::error!(
                    body = %body,
                    "tool_call_unreachable_fallback_required: body unparseable \
                     after streaming name extraction; per-family grammar bug"
                );
                let _ = events.blocking_send(GenerationEvent::Error(
                    "tool_call_unreachable_fallback_required".into(),
                ));
                return Err(());
            }
            // Auto-no-grammar: close the streaming JSON args we already
            // committed to so client accumulators land on valid JSON.
            if events
                .blocking_send(GenerationEvent::ToolCallDelta {
                    index: self.index,
                    id: None,
                    call_type: None,
                    name: None,
                    arguments: Some("}".into()),
                })
                .is_err()
            {
                return Err(());
            }
            *tc_index += 1;
            *saw_tc = true;
            return Ok(());
        };

        // Reconstruct the exact JSON args string emitted so far ( = `{` plus
        // each kv-comma-separated ) and compute the residual tail by
        // diffing against `pc.arguments_json`. This is robust to:
        //   - emitter scanned 0 kvs (whole args arrived in the close fragment)
        //   - emitter scanned all-but-last kv (typical streaming case)
        //   - emitter scanned all kvs (rare: comma after final kv would have
        //     to appear in body, which Gemma's template doesn't emit; Qwen
        //     trailing `</parameter>` followed by `</function>` does mean
        //     scan_cursor is past the last kv before finalize)
        let so_far = self.reconstruct_emitted_args_prefix(&pc.arguments_json);
        let tail = pc.arguments_json[so_far.len()..].to_string();
        if !tail.is_empty() {
            if events
                .blocking_send(GenerationEvent::ToolCallDelta {
                    index: self.index,
                    id: None,
                    call_type: None,
                    name: None,
                    arguments: Some(tail),
                })
                .is_err()
            {
                return Err(());
            }
        }
        *tc_index += 1;
        *saw_tc = true;
        Ok(())
    }

    /// Reconstruct the JSON-args string the streaming emitter has *already*
    /// sent to the client, so `finalize` can compute the residual tail by
    /// suffix-diff against the full `arguments_json` returned by
    /// `parse_tool_call_body`.
    ///
    /// Strategy: walk `full_args_json` (which is well-formed `{...}`) and
    /// take the longest prefix that contains exactly `kvs_emitted` top-level
    /// kvs. The streaming emitter always emits `{`, then for kv #1 just the
    /// raw kv JSON, then for kv #2..N a leading `,`. Therefore the prefix
    /// we already emitted ends RIGHT BEFORE the start of kv #(kvs_emitted+1)
    /// — i.e. before the comma preceding it (or before the `}` if all kvs
    /// were streamed).
    fn reconstruct_emitted_args_prefix<'a>(&self, full_args_json: &'a str) -> &'a str {
        // Walk the JSON object counting kv-pairs at depth 1. We can use a
        // simple state machine: track `{}` depth and `"` string state, count
        // commas at depth 1 (each comma = boundary between two kvs).
        let bytes = full_args_json.as_bytes();
        let mut depth: i32 = 0;
        let mut in_str = false;
        let mut esc = false;
        let mut commas_at_depth_1 = 0usize;
        // Number of kvs in the prefix we've sent = self.kvs_emitted.
        // Number of commas in that prefix = max(0, kvs_emitted - 1) + (1 if
        // kvs_emitted > 0 we've emitted up to and including kv #N, NOT
        // beyond it). So we want the longest prefix ending RIGHT BEFORE
        // `,` #(kvs_emitted) or, if we've emitted 0 kvs, right after the
        // opening `{`.
        if self.kvs_emitted == 0 {
            // Emitted only `{`. Prefix is `{`.
            // Find the first `{` (well-formed JSON starts with it).
            for (i, &b) in bytes.iter().enumerate() {
                if b == b'{' {
                    return &full_args_json[..=i];
                }
            }
            return "";
        }
        // We need to find the position of the (kvs_emitted)th comma at
        // depth 1, OR the closing `}` at depth 1 if no further comma exists
        // — and return the prefix ending just before it.
        for (i, &b) in bytes.iter().enumerate() {
            if in_str {
                if esc {
                    esc = false;
                } else if b == b'\\' {
                    esc = true;
                } else if b == b'"' {
                    in_str = false;
                }
                continue;
            }
            match b {
                b'"' => in_str = true,
                b'{' | b'[' => depth += 1,
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 && commas_at_depth_1 + 1 == self.kvs_emitted {
                        // No further comma — we've streamed every kv. The
                        // prefix is everything up to (not including) `}`.
                        return &full_args_json[..i];
                    }
                }
                b',' => {
                    if depth == 1 {
                        commas_at_depth_1 += 1;
                        if commas_at_depth_1 == self.kvs_emitted {
                            // The Nth comma at depth 1 separates kv #N from
                            // kv #(N+1). The prefix we've emitted ends
                            // RIGHT BEFORE this comma (since kv #(N+1) hasn't
                            // been streamed; finalize will emit `,kv#(N+1)`
                            // — so the residue must include the comma).
                            return &full_args_json[..i];
                        }
                    }
                }
                _ => {}
            }
        }
        // Defensive: malformed input — return full string so tail is empty.
        full_args_json
    }
}

/// Extract the function name from a Gemma 4 body prefix `call:NAME{`. Returns
/// `Some((name, header_end))` where `header_end` is the byte offset of the
/// `{` (so `body[header_end+1..]` is the kv-list region the streaming
/// scanner walks). Returns `None` if the `{` hasn't arrived yet.
fn extract_gemma4_name_prefix(body: &str) -> Option<(String, usize)> {
    let trimmed_offset = body.len() - body.trim_start().len();
    let after_ws = &body[trimmed_offset..];
    let after_call = after_ws.strip_prefix("call:")?;
    let brace_rel = after_call.find('{')?;
    let name = after_call[..brace_rel].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let absolute_brace = trimmed_offset + "call:".len() + brace_rel;
    Some((name, absolute_brace + 1))
}

/// Extract the function name from a Qwen 3.5/3.6 body prefix
/// `<function=NAME>`. Returns `Some((name, header_end))` where `header_end`
/// is the byte offset just past `>` (so the streaming scanner walks
/// `body[header_end..]` for `<parameter>` blocks). Returns `None` if the
/// closing `>` hasn't arrived yet.
fn extract_qwen35_name_prefix(body: &str) -> Option<(String, usize)> {
    let trimmed_offset = body.len() - body.trim_start().len();
    let after_ws = &body[trimmed_offset..];
    let after_open = after_ws.strip_prefix("<function=")?;
    let gt_rel = after_open.find('>')?;
    let name = after_open[..gt_rel].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let absolute_gt = trimmed_offset + "<function=".len() + gt_rel;
    Some((name, absolute_gt + 1))
}

/// Convert one Gemma 4 kv span `key:<jsonval>` into a JSON `"key":<json>`
/// fragment. Mirrors the value-coercion logic in `parse_gemma4_tool_call`
/// at registry.rs:737-768. Returns `None` on malformed kv (caller leaves
/// scan_cursor untouched so finalize falls into the close-buffered path).
fn gemma4_kv_to_json(kv: &str) -> Option<String> {
    let (k, v) = kv.split_once(':')?;
    let key = k.trim();
    if key.is_empty() {
        return None;
    }
    let v = v.trim();
    let json_val = if let Some(stripped) = v
        .strip_prefix("<|\"|>")
        .and_then(|s| s.strip_suffix("<|\"|>"))
    {
        serde_json::Value::String(stripped.to_string())
    } else if let Ok(num) = v.parse::<i64>() {
        serde_json::Value::from(num)
    } else if let Ok(num) = v.parse::<f64>() {
        serde_json::Value::from(num)
    } else if v == "true" {
        serde_json::Value::Bool(true)
    } else if v == "false" {
        serde_json::Value::Bool(false)
    } else if v == "null" {
        serde_json::Value::Null
    } else {
        serde_json::Value::String(v.to_string())
    };
    let key_json = serde_json::to_string(key).ok()?;
    let val_json = serde_json::to_string(&json_val).ok()?;
    Some(format!("{key_json}:{val_json}"))
}

/// Dispatch the close-time tool-call body after `ToolCallClose` fires in the
/// streaming `route_content` closure.
///
/// Wave 3 W-A3 — T2.4 partial removal (Constrained body-parse-failure case).
/// Wave 3 W-B2 — T2.4 final closure: `AutoLazyGrammar` joins `Constrained`
/// in the loud-error branch.
///
/// ## Behaviour matrix
///
/// ```text
///                       │ Constrained / AutoLazyGrammar   │ Auto (no grammar)
/// ─────────────────────┼─────────────────────────────────┼────────────────────────────────
/// parse OK              │ emit ToolCallDelta ×2           │ emit ToolCallDelta ×2
/// parse FAILURE         │ emit GenerationEvent::Error     │ emit delta.content fallback
///                       │   "tool_call_unreachable_       │   (tracing::warn)
///                       │    fallback_required"           │
///                       │ + tracing::error! (grammar bug) │
///                       │ → Err(())                       │ → Ok(())  [or Err if send fails]
/// ```
///
/// **Why Auto (no grammar) preserves the content fallback**: when the
/// request is `tool_choice=auto` AND no grammar is active (no tools[]
/// declared, OR an unregistered model family) there is no enforcement on
/// body shape. A model can legitimately emit malformed / partial
/// tool-call syntax and the caller should still see the raw bytes rather
/// than losing them silently. The content fallback is the defined
/// behaviour for this unconstrained branch.
///
/// **Why Constrained AND AutoLazyGrammar both error loudly**: under
/// `Constrained` the wave-2.7 W-η da545d5 eager grammar constrains every
/// token from byte 0. Under `AutoLazyGrammar` (wave 3 W-B2) the same
/// per-model body grammar is active from `ToolCallOpen` onwards via the
/// `awaiting_trigger` gate flip. In both cases the grammar physically
/// constrains the body bytes, so a body-parse failure means the grammar
/// engine produced structurally invalid output — a server-side
/// regression, not a model quality issue. Surfacing it as a loud error
/// (rather than a silent content fallback) makes the regression
/// immediately visible to operators.
///
/// The unified gate is `policy.enforces_body_grammar()` — see
/// `ToolCallPolicy::enforces_body_grammar` for the single source of
/// truth.
///
/// Extracted from the `route_content` closure so audit-driver tests can
/// exercise this exact code path directly (same extraction pattern as
/// `finalize_streaming_tool_state` for HIGH-1).
fn emit_streaming_tool_call_close(
    parsed: Option<super::registry::ParsedToolCall>,
    body_dump: String,
    policy: ToolCallPolicy,
    tc_index: &mut usize,
    saw_tc: &mut bool,
    events: &tokio::sync::mpsc::Sender<super::sse::GenerationEvent>,
) -> Result<(), ()> {
    use super::sse::{DeltaKind, GenerationEvent};

    match parsed {
        Some(pc) => {
            // First chunk: id + type + name. The id is a synthesized opaque
            // identifier; clients echo it in their `tool_call_id` follow-up
            // message. Format mirrors OpenAI's `call_<24hex>` shape.
            let id = format!(
                "call_hf2q_{:016x}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0)
                    ^ (*tc_index as u64).wrapping_mul(0x9e3779b97f4a7c15)
            );
            if events
                .blocking_send(GenerationEvent::ToolCallDelta {
                    index: *tc_index,
                    id: Some(id),
                    call_type: Some("function".into()),
                    name: Some(pc.name),
                    arguments: None,
                })
                .is_err()
            {
                return Err(());
            }
            // Second chunk: full arguments JSON string. OpenAI clients
            // accumulate `function.arguments` deltas; one chunk is spec-valid.
            if events
                .blocking_send(GenerationEvent::ToolCallDelta {
                    index: *tc_index,
                    id: None,
                    call_type: None,
                    name: None,
                    arguments: Some(pc.arguments_json),
                })
                .is_err()
            {
                return Err(());
            }
            *tc_index += 1;
            *saw_tc = true;
            Ok(())
        }
        None => {
            // Wave 3 W-A3 + W-B2 — T2.4 final closure on registered families.
            //
            // Both Constrained AND AutoLazyGrammar carry an active grammar
            // that physically constrains the body — Constrained from byte 0
            // (eager), AutoLazyGrammar from `ToolCallOpen` onwards (lazy).
            // A parse failure under either policy means the grammar engine
            // produced structurally invalid output. Promote to a loud
            // structured error so the regression surfaces immediately
            // rather than being silently swallowed by the content fallback.
            //
            // Auto (no grammar): legitimate parse-failure path. The model
            // emitted malformed syntax with no grammar to constrain it;
            // re-emit as content so the caller sees what the model intended.
            if policy.enforces_body_grammar() {
                let policy_label = match policy {
                    ToolCallPolicy::Constrained => "constrained (required/function)",
                    ToolCallPolicy::AutoLazyGrammar => "auto-lazy-grammar (wave-3 W-B2)",
                    ToolCallPolicy::Auto => unreachable!("enforces_body_grammar gate"),
                };
                tracing::error!(
                    body = %body_dump,
                    policy = policy_label,
                    "tool_call_unreachable_fallback_required: tool-call body \
                     unparseable under {} policy; the per-model body grammar \
                     should have prevented this — grammar engine bug",
                    policy_label
                );
                let _ = events.blocking_send(GenerationEvent::Error(
                    "tool_call_unreachable_fallback_required".into(),
                ));
                return Err(());
            }
            // Auto (no grammar) path: preserve the pre-wave-2.5 content
            // fallback. (See function doc above for the rationale.)
            tracing::warn!(
                body = %body_dump,
                "tool-call body unparseable; emitting as content fallback \
                 (tool_choice=auto with no active grammar — no enforcement \
                 on body shape; either tools[] empty or unregistered family)"
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
            Ok(())
        }
    }
}

/// Replay a cached `GenerationResult` as a sequence of SSE events.
///
/// Wave 3 W-A2 — closes the asymmetry documented at the iter-96 streaming
/// store-only callsite.  Pre-W-A2 the streaming path stored to
/// `PromptCache` on every successful completion but never consulted the
/// cache on input, so most clients (which use streaming) never benefited
/// from cache hits.  This helper lets `generate_stream_once` short-circuit
/// when `PromptCache::lookup` returns `Some(...)`: emit the cached text as
/// SSE deltas (re-classified through the same Reasoning + ToolCall splitter
/// pipeline the live decode uses), then emit `Done` with zero timings and
/// `cached_prompt_tokens = Some(prompt_len)` so the client surfaces
/// `usage.prompt_tokens_details.cached_tokens` exactly like a non-streaming
/// hit.
///
/// # Cache-shape decision (deliberate, surfaced in commit body)
///
/// `PromptCache` stores text only — no token sequence — so the replay
/// CANNOT preserve original token boundaries (one delta per cached token).
/// Two designs were considered:
///
/// 1. **Single big content delta.** Simple. Loses tool-call shape if the
///    cached response was a tool call: the open/close markers would arrive
///    inside `delta.content` instead of producing structured
///    `delta.tool_calls[*]` events.  Spec-violating for tool-call replays.
///
/// 2. **Re-route through the live splitter pipeline (chosen).** Build a
///    fresh `ReasoningSplitter` + `ToolCallSplitter` from the model
///    registration (same factory the live decode uses), feed the cached
///    `text` in once, and dispatch the resulting events through the same
///    `route_content` / `emit_streaming_tool_call_close` helpers.  When
///    the cached text contains tool-call markers, the splitter emits
///    structured `ToolCallDelta` events identical to a fresh decode.
///    When the text is plain content, the splitter emits a single
///    `Content` delta.  When `reasoning_text` is `Some(...)` (cached from
///    a non-streaming completion that already split reasoning out), it is
///    emitted first as a `Reasoning` delta so the SSE response shape
///    matches the original.
///
/// Per-cached-token replay (preserve TTFT-like incremental UX) requires
/// extending `PromptCache` to store the per-token Delta sequence rather
/// than the assembled text.  That is a real shape extension; documented as
/// a follow-up rather than shoehorned into this iter.
///
/// # Why pass `tool_call_policy`
///
/// Tool-call body parse failures branch on policy (Constrained → loud
/// `GenerationEvent::Error`; Auto → silent content fallback).  The cache
/// key includes `tool_call_policy`, so a hit guarantees the policy
/// matches the original request — but `emit_streaming_tool_call_close`
/// still requires it as an argument to make the branch explicit.
///
/// # Why `grammar_runtime: None`
///
/// `route_content`'s `ToolCallOpen` branch flips
/// `runtime.is_awaiting_trigger()` so subsequent decode-loop mask calls
/// fire.  In replay there is NO decode loop — the runtime is irrelevant.
/// Pass `None` so the branch is a no-op (matches the no-grammar live
/// path).  This is sound because the trigger is purely a live-decode
/// gating mechanism, not part of the SSE event shape.
///
/// Returns `Ok(())` if the full cached response (Reasoning? + content
/// events + Done) was emitted; `Err(())` if any send failed (client
/// disconnected mid-replay) — caller bumps the cancellation counter and
/// returns, mirroring the live-decode disconnect path.
///
/// # End-of-stream splitter drain (Wave 3.5 HIGH-2)
///
/// Both `ReasoningSplitter` and `ToolCallSplitter` hold back a sliding
/// tail (`tail_buf`) up to `tail_cap` bytes long in case the next
/// fragment continues a marker boundary.  Pre-Wave-3.5 the replay fed
/// `cached.text` once and emitted Done — never calling `finish()` on
/// either splitter — so the held-back tail bytes were silently dropped.
/// This caused truncated content on cache hits whose tails happened to
/// look like partial markers.
///
/// The drain order mirrors the live-decode path
/// (engine.rs:3691-3757): `reasoning_splitter.finish()` first, routing
/// any Content tail through `tool_splitter`; then
/// `tool_splitter.finish()` to emit the final residual.  Unlike the
/// live path, replay does NOT promote ToolCallText residuals to
/// structured Errors — a cached entry was already validated when
/// stored, so any residual tail is plain content not a mid-decode
/// truncation.  Audit citation:
/// `/tmp/cfa-cfa-20260427-adr005-wave3/codex-review-last.txt`
/// divergence "W-A2 streaming cache replay" severity HIGH.
fn replay_cached_streaming_response(
    cached: &GenerationResult,
    registration: Option<&super::registry::ModelRegistration>,
    tool_call_policy: ToolCallPolicy,
    events: &mpsc::Sender<super::sse::GenerationEvent>,
) -> Result<(), ()> {
    use super::sse::{DeltaKind, GenerationEvent, StreamStats};

    // ── 1. Reasoning replay ─────────────────────────────────────────────
    //
    // Non-streaming-origin cache entries store reasoning_text separately
    // (split out of the assembled text via `split_full_output` in
    // `generate_once_with_soft_tokens`).  Streaming-origin entries store
    // reasoning_text == None because the live splitter routed reasoning
    // fragments into Reasoning deltas as decoded; the assembled text
    // contains the full pre-split stream.  Either way: emit the
    // explicit reasoning_text first (if any), then route the text through
    // the splitter to handle the streaming-origin embedded-marker case.
    if let Some(reasoning) = cached.reasoning_text.as_deref() {
        if !reasoning.is_empty()
            && events
                .blocking_send(GenerationEvent::Delta {
                    kind: DeltaKind::Reasoning,
                    text: reasoning.to_string(),
                })
                .is_err()
        {
            return Err(());
        }
    }

    // ── 2. Content / tool-call replay ───────────────────────────────────
    //
    // Build fresh splitters mirroring the `generate_stream_once` setup
    // (lines below the lookup).  The ReasoningSplitter handles any
    // embedded reasoning markers (streaming-origin entries).  The
    // ToolCallSplitter handles embedded tool-call markers regardless of
    // origin (neither streaming nor non-streaming strips them).
    let mut reasoning_splitter = registration
        .and_then(|r| super::registry::ReasoningSplitter::from_registration(r));
    let mut tool_splitter = registration
        .and_then(|r| super::registry::ToolCallSplitter::from_registration(r));

    let mut tool_call_body: String = String::new();
    let mut tool_call_index: usize = 0;
    let mut saw_tool_call: bool = false;

    // Replay-side fragment routing.  Mirrors `route_content` minus
    // grammar plumbing (no decode loop ⇒ no runtime to trigger).  Inline
    // here rather than reusing the closure because the closure captures
    // generate_stream_once-local state we don't have in this free fn —
    // duplicating the ~30-line dispatch is cheaper than threading a
    // borrow web across a closure parameter list.  Architecturally this
    // is the same pattern as `emit_streaming_tool_call_close` (extracted
    // for the close-branch) and `finalize_streaming_tool_state` (extracted
    // for the end-of-stream drain).
    let route_replay_fragment =
        |tcs: &mut Option<super::registry::ToolCallSplitter>,
         body: &mut String,
         tc_index: &mut usize,
         saw_tc: &mut bool,
         events: &mpsc::Sender<GenerationEvent>,
         text: &str|
         -> Result<(), ()> {
            if text.is_empty() {
                return Ok(());
            }
            let Some(tcs) = tcs.as_mut() else {
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
                        // Replay: no grammar_runtime to trigger.
                    }
                    super::registry::ToolCallEvent::ToolCallText(t) => {
                        body.push_str(&t);
                    }
                    super::registry::ToolCallEvent::ToolCallClose => {
                        let parsed = registration
                            .and_then(|r| super::registry::parse_tool_call_body(r, body));
                        let body_dump = std::mem::take(body);
                        emit_streaming_tool_call_close(
                            parsed,
                            body_dump,
                            tool_call_policy,
                            tc_index,
                            saw_tc,
                            events,
                        )?;
                    }
                }
            }
            Ok(())
        };

    // Feed the cached text through the reasoning splitter first, then
    // route Content-classified spans through the tool-call splitter.
    // Mirrors `emit_fragment` in generate_stream_once.
    if let Some(rs) = reasoning_splitter.as_mut() {
        for (slot, text) in rs.feed(&cached.text) {
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
                    route_replay_fragment(
                        &mut tool_splitter,
                        &mut tool_call_body,
                        &mut tool_call_index,
                        &mut saw_tool_call,
                        events,
                        &text,
                    )?;
                }
            }
        }
    } else {
        route_replay_fragment(
            &mut tool_splitter,
            &mut tool_call_body,
            &mut tool_call_index,
            &mut saw_tool_call,
            events,
            &cached.text,
        )?;
    }

    // ── 2b. End-of-stream splitter drain ────────────────────────────────
    //
    // Wave 3.5 HIGH-2: live decode drains BOTH the ReasoningSplitter and
    // the ToolCallSplitter at end-of-stream because each holds back a
    // tail (`tail_buf` of size up to `tail_cap` bytes) in case the next
    // fragment continues a marker boundary.  Pre-Wave-3.5 replay fed
    // `cached.text` through `feed()` ONCE then jumped straight to Done,
    // never calling `finish()` on either splitter — so the held-back
    // tail bytes (typically a few characters) were silently dropped.
    //
    // Concrete failure mode the audit caught
    // (/tmp/cfa-cfa-20260427-adr005-wave3/codex-review-last.txt
    // divergence "W-A2 streaming cache replay" severity HIGH): a cached
    // plain-text response shorter than `tail_cap` bytes (or whose tail
    // looks like a partial marker prefix) had its terminal characters
    // truncated.  A cached response with a tool-call marker followed by
    // postscript content had the postscript truncated.
    //
    // Mirrors the live-decode drain at engine.rs:3691-3757
    // (reasoning_splitter.finish → tool_splitter Content route →
    // tool_splitter.finish via finalize_streaming_tool_state).  The
    // replay equivalent uses `route_replay_fragment` (no grammar
    // runtime) and inlines the tool_splitter.finish() drain because we
    // don't enforce mid-call truncation Errors on a cache hit (the
    // policy-loud-error contract is for live decode; cached entries
    // were already validated when stored).
    if let Some(rs) = reasoning_splitter.as_mut() {
        if let Some((slot, tail)) = rs.finish() {
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
                        return Err(());
                    }
                }
                super::registry::SplitSlot::Content => {
                    // Route through the tool-call splitter so a Content
                    // tail straddling a marker boundary still classifies
                    // correctly (mirrors live drain at engine.rs:3710-3727).
                    route_replay_fragment(
                        &mut tool_splitter,
                        &mut tool_call_body,
                        &mut tool_call_index,
                        &mut saw_tool_call,
                        events,
                        &tail,
                    )?;
                }
            }
        }
    }
    // tool_splitter.finish() drain: route any held-back tail to the
    // appropriate slot.  Unlike the live path (which fires a structured
    // Error on mid-call truncation under Constrained/AutoLazyGrammar),
    // replay always treats the tail as plain content for the same
    // reason: the cache stored a verified-complete response, so any
    // residual tail is by definition not a mid-decode truncation.
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
                        return Err(());
                    }
                }
                super::registry::ToolCallEvent::ToolCallText(t) => {
                    // Cached entry ended mid-tool-call (open marker
                    // observed but no close marker reached).  Re-emit
                    // as Content with the open marker re-prepended for
                    // diagnostic clarity — same shape as the live
                    // Auto-no-grammar drain at engine.rs:2024-2035.
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
                        return Err(());
                    }
                }
                super::registry::ToolCallEvent::ToolCallOpen
                | super::registry::ToolCallEvent::ToolCallClose => {
                    // unreachable — finish() never emits Open/Close.
                }
            }
        }
    }

    // ── 3. Done event ───────────────────────────────────────────────────
    //
    // `cached_prompt_tokens = Some(prompt_len)` so the SSE final-chunk
    // usage surfaces `prompt_tokens_details.cached_tokens` identically to
    // the non-streaming hit path (engine.rs:752-754).  Zero timings
    // because prefill+decode were skipped — same convention as the
    // non-streaming GenerationResult on hit.
    let stats = StreamStats {
        prefill_time_secs: Some(0.0),
        decode_time_secs: Some(0.0),
        total_time_secs: Some(0.0),
        time_to_first_token_ms: Some(0.0),
        prefill_tokens_per_sec: None,
        decode_tokens_per_sec: None,
        gpu_sync_count: None,
        gpu_dispatch_count: None,
        cached_prompt_tokens: Some(cached.cached_tokens),
        reasoning_tokens: cached.reasoning_tokens,
    };

    if events
        .blocking_send(GenerationEvent::Done {
            // Tool-call replays must override finish_reason so clients see
            // `tool_calls` (OpenAI spec). The splitter sets `saw_tool_call`
            // when a ToolCallClose fires.
            finish_reason: if saw_tool_call { "tool_calls" } else { cached.finish_reason },
            prompt_tokens: cached.prompt_tokens,
            completion_tokens: cached.completion_tokens,
            stats,
        })
        .is_err()
    {
        return Err(());
    }
    Ok(())
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

    // ── Prompt cache fast-path (Wave 3 W-A2) ──────────────────────────────
    //
    // Mirrors the non-streaming preroll lookup at engine.rs:1419 (iter-96
    // / wave-2.5 B5).  Pre-W-A2 the streaming path stored to PromptCache on
    // every successful completion (`store` call below the decode loop) but
    // never consulted it on input — most clients use streaming, so cache
    // hits were essentially impossible on the production path.  W-A2
    // closes the documented gap (former engine.rs:2109 "iter-97 follow-up"
    // comment).
    //
    // Eligibility: `PromptCache::lookup` self-gates on temperature/top_k/
    // top_p/repetition_penalty/seed (sampling-mode bypass), prompt-token
    // equality, AND PromptCacheKey full-inventory equality (wave-2.5 B5
    // expansion: max_tokens, stop_strings, logit_bias, grammar,
    // grammar_kind, frequency/presence/min_p penalties, tool_call_policy,
    // logprobs/top_logprobs, parallel_tool_calls).  See PromptCacheKey
    // doc at engine.rs:489-535 for the full inventory.
    //
    // On hit, `replay_cached_streaming_response` re-routes the cached
    // text through the same Reasoning + ToolCall splitter pipeline the
    // live decode uses, so the SSE shape (Content / Reasoning /
    // ToolCallDelta) matches what the original request produced.  See
    // that helper's doc for the cache-shape decision (text-only ⇒ single
    // splitter pass, NOT per-token replay; per-token replay would require
    // extending PromptCache to store the Delta sequence).
    if let Some(cached) = loaded.prompt_cache.lookup(prompt_tokens, params) {
        tracing::debug!(
            "prompt_cache: STREAMING HIT — {} tokens served from cache, prefill+decode skipped",
            cached.prompt_tokens
        );
        if replay_cached_streaming_response(
            &cached,
            registration,
            params.tool_call_policy,
            events,
        )
        .is_err()
        {
            tracing::info!("SSE stream dropped by client during cache replay; aborting");
            if let Some(c) = cancellation_counter {
                c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        return;
    }

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

    // Wave 3 W-B3 — T2.3 incremental tool-call argument streaming.
    //
    // Per-call streaming-emit state. `Some(...)` between `ToolCallOpen` and
    // `ToolCallClose`; `None` otherwise. `ToolCallText` fragments call
    // `emitter.advance(body, events)` to flush newly-extractable name + kv
    // fragments immediately rather than waiting for `ToolCallClose` to
    // emit one big arguments delta. `ToolCallClose` calls
    // `emitter.finalize(...)` which either:
    //   - emits the closing `}` + any tail kvs (streaming path was active), or
    //   - delegates to `emit_streaming_tool_call_close` (streaming path
    //     declined, falling back to pre-W-B3 close-buffered shape).
    //
    // Cache replay keeps the close-buffered path — see `route_replay_fragment`.
    // Incremental emission has zero benefit for cache hits (the full text is
    // available synchronously) and would force two divergent SSE shapes for
    // identical `cached.text` content.
    let mut tool_call_emitter: Option<ToolCallStreamEmitter> = None;

    // Wave-2.5 A4: capture the policy so the route_content closure can branch
    // on Constrained vs Auto when a tool-call body fails to parse.
    let tool_call_policy = params.tool_call_policy;

    // Wave 2.6 W-α5 Q2: the wave-2.5 `Arc<AtomicBool> grammar_active`
    // sibling-state pattern is REMOVED.  The trigger gate now lives
    // inside `GrammarRuntime` itself (`is_awaiting_trigger()`); the
    // `route_content` closure flips it via `runtime.trigger()` on
    // ToolCallOpen, and the decode loop calls `mask_invalid_tokens` /
    // `accept_bytes` / `is_dead` UNCONDITIONALLY — all three self-gate
    // on the SAME boolean, eliminating the split-state condition the
    // wave-2.5 audit caught at engine.rs:1401, 1489, 1554, 2041, 2145,
    // 2195.  See cfa-20260427-adr005-wave2.6 research-report.md Q2 +
    // /opt/llama.cpp/src/llama-grammar.cpp:1287-1439 for the canonical
    // pattern.
    //
    // To call `runtime.trigger()` from the `route_content` closure, the
    // closure needs mutable access to the runtime.  The runtime is
    // owned by `generate_stream_once`, so we share it via
    // `Rc<RefCell<...>>` — single-threaded, no atomics needed (the
    // streaming worker is one OS thread).  When grammar is None,
    // `grammar_runtime` is None and the trigger plumbing is a no-op.
    // (We use `Rc<RefCell<...>>` instead of an `&mut` borrow because
    // the closure outlives the borrow checker's view of the runtime
    // through the decode loop — same shape used elsewhere in this
    // function for shared mutable state.)

    // Helper: for a Content-classified text run, route through the
    // ToolCallSplitter (if any) and emit the appropriate
    // GenerationEvent. When ToolCallSplitter is None, route every byte to
    // `DeltaKind::Content` (current behavior pre-iter-B-2).
    //
    // Wave 2.6 W-α5 Q2: takes `grammar_runtime: &mut Option<GrammarRuntime>`
    // so the ToolCallOpen branch can call `runtime.trigger()` — this is
    // the splice point where the lazy-grammar awakens.  ToolCallClose
    // does NOT reset (multi-call grammars rely on the grammar shape
    // accepting `(call)+`; see research-report.md Q2 + llama.cpp PR
    // #9639).
    let route_content = |tool_splitter: &mut Option<super::registry::ToolCallSplitter>,
                         body: &mut String,
                         tc_index: &mut usize,
                         saw_tc: &mut bool,
                         emitter: &mut Option<ToolCallStreamEmitter>,
                         grammar_runtime: &mut Option<super::grammar::GrammarRuntime>,
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
                    // Wave 3 W-B3 — T2.3 incremental: construct a fresh
                    // emitter for this call. Family from registration; an
                    // unregistered family yields an emitter that declines
                    // (its `advance` is a no-op and `finalize` falls back
                    // to `emit_streaming_tool_call_close`).
                    *emitter = Some(ToolCallStreamEmitter::new(
                        reg.map(|r| r.family),
                        *tc_index,
                    ));
                    // Wave 2.6 W-α5 Q2: entering a tool-call body — flip
                    // the grammar runtime's trigger so subsequent decode
                    // steps enforce the body grammar.  No-op when the
                    // runtime is None (no grammar request) or already
                    // post-trigger (re-entry on a grammar without
                    // explicit reset support — llama.cpp behavior).
                    if let Some(rt) = grammar_runtime.as_mut() {
                        rt.trigger();
                    }
                }
                super::registry::ToolCallEvent::ToolCallText(t) => {
                    body.push_str(&t);
                    // Wave 3 W-B3 — T2.3 incremental: drive the per-call
                    // emitter to flush newly-extractable name + kv
                    // fragments. No-op when the family is unregistered
                    // (`advance` returns Ok(()) without sending anything,
                    // leaving finalize to use the close-buffered fallback).
                    if let Some(em) = emitter.as_mut() {
                        em.advance(body, events)?;
                    }
                }
                super::registry::ToolCallEvent::ToolCallClose => {
                    // Wave 2.6 W-α5 Q2: leaving a tool-call body.  The
                    // grammar runtime is NOT reset — multi-call support
                    // relies on the grammar root accepting `(call)+`
                    // directly (Hermes 2 Pro template; research-report.md
                    // Q2 anti-finding).  If the model attempts a second
                    // call against a single-call grammar, the runtime
                    // will simply die on the next mask step (graceful
                    // termination via the unconditional is_dead check
                    // in the decode loop).
                    //
                    // Wave 3 W-A3: close-time dispatch delegated to
                    // `emit_streaming_tool_call_close` so the parse-failure
                    // branch can be audit-driver tested independently.
                    //
                    // Wave 3 W-B3: if the per-call emitter activated mid-
                    // stream (name was emitted), finalize emits the
                    // closing `}` + tail kvs and increments `tc_index`.
                    // Otherwise finalize delegates to the legacy
                    // `emit_streaming_tool_call_close` so the close-
                    // buffered shape AND policy-enforced loud-error
                    // branches stay byte-for-byte identical.
                    let body_dump = std::mem::take(body);
                    let mut em = emitter.take().unwrap_or_else(|| {
                        // Defensive — ToolCallOpen always precedes Close,
                        // but if a buggy splitter ever emits Close-without-
                        // Open we still want the legacy close-buffered
                        // semantics to fire.
                        ToolCallStreamEmitter::new(reg.map(|r| r.family), *tc_index)
                    });
                    em.finalize(body_dump, reg, tool_call_policy, tc_index, saw_tc, events)?;
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
                         emitter: &mut Option<ToolCallStreamEmitter>,
                         grammar_runtime: &mut Option<super::grammar::GrammarRuntime>,
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
                            emitter,
                            grammar_runtime,
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
                emitter,
                grammar_runtime,
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
    // Wave 2.6 W-α5 Q2 + Wave 2.7 W-η Q-A: arm the trigger gate ONLY when
    // the request carries an AUTO-mode tool-call body grammar
    // (`GrammarKind::ToolCallBodyAuto`).  `ResponseFormat` and
    // `ToolCallBodyRequired` runtimes leave the gate disarmed for eager
    // enforcement from token 0 — fixes audit divergence A1 /
    // response_format regression for ResponseFormat, and forces tool-call
    // emission for Required/Function (mirrors llama.cpp grammar_lazy=false
    // in common/chat.cpp:898-913, 1177-1200, 1399-1416).
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
                Some(mut rt) => {
                    // Wave 2.7 W-η Q-A: see non-streaming arming above —
                    // `ToolCallBodyRequired` keeps the eager gate
                    // (`awaiting_trigger=false`); only `ToolCallBodyAuto`
                    // suspends until ToolCallSplitter sees the open marker.
                    if matches!(params.grammar_kind, GrammarKind::ToolCallBodyAuto) {
                        rt.set_awaiting_trigger(true);
                    }
                    Some(rt)
                }
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
        // Wave 2.6 W-α5 Q2: mask + accept are UNCONDITIONAL.  For
        // `ToolCallBody`-kind runtimes the gate is armed (no-op);
        // for `ResponseFormat`-kind it enforces from token 0 — the
        // wave-2.5 audit fix for the response_format regression.
        if let (Some(rt), Some(tb)) = (grammar_runtime.as_ref(), token_bytes_ref) {
            super::grammar::mask::mask_invalid_tokens(rt, tb, &mut logits);
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
            &mut tool_call_emitter,
            &mut grammar_runtime,
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
                // Wave 2.6 W-α5 Q2: mask + accept UNCONDITIONAL.  The
                // runtime self-gates on `is_awaiting_trigger()`.  When
                // suspended (ToolCallBody pre-trigger), mask is a no-op
                // and the model emits preamble freely; once
                // `route_content` sees the open marker and calls
                // `runtime.trigger()`, every subsequent step enforces.
                // ResponseFormat-kind runtimes were never suspended and
                // enforce from the first token.  This collapses the
                // wave-2.5 4-line `if grammar_active.load { mask }` /
                // separate `accept` paired pattern into 2 lines that are
                // structurally correct.
                if let (Some(rt), Some(tb)) = (grammar_runtime.as_ref(), token_bytes_ref) {
                    super::grammar::mask::mask_invalid_tokens(rt, tb, &mut logits);
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
                &mut tool_call_emitter,
                &mut grammar_runtime,
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
                        &mut tool_call_emitter,
                        &mut grammar_runtime,
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
    // Drain any tool-splitter tail and then enforce Constrained-policy
    // safety nets before Done. The drain + check logic is factored into
    // `finalize_streaming_tool_state` so the test harness can drive the
    // exact production code path (audit-driver test for HIGH-1; do not
    // duplicate this logic in a test stand-in).
    match finalize_streaming_tool_state(
        tool_splitter.as_mut(),
        tool_call_policy,
        saw_tool_call,
        registration,
        completion_tokens,
        accumulated_text.len(),
        events,
    ) {
        FinalizeStreamingAction::Continue => {}
        FinalizeStreamingAction::ClientDropped => {
            tracing::info!("SSE stream dropped by client; aborting decode");
            return;
        }
        FinalizeStreamingAction::ErrorEmitted => {
            // Mid-call truncation or no-call under Constrained already
            // emitted GenerationEvent::Error — do NOT also emit Done. The
            // SSE encoder closes the stream with a finish_reason="error"
            // final chunk on receipt of Error (sse.rs:298-322).
            return;
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

    // Iter-96 prompt cache update on streaming completion.
    //
    // Wave 3 W-A2: the streaming path now ALSO consults the cache on
    // input via `replay_cached_streaming_response` (see lookup at the
    // top of this function).  The store here is unchanged; updating on
    // every successful completion lets BOTH a subsequent streaming AND a
    // subsequent non-streaming request with the same prompt+params hit
    // the cache (the cache slot is mode-agnostic; only the lookup +
    // replay path differs).
    //
    // Cache-shape note: `text` is set to the full pre-split
    // `accumulated_text` (markers and all).  The replay helper re-routes
    // through fresh ReasoningSplitter + ToolCallSplitter to re-emit the
    // proper SSE shape, so embedded markers re-classify correctly.
    // `reasoning_text: None` because the live splitter already routed
    // reasoning fragments into Reasoning deltas during decode — there is
    // no separately-tracked reasoning string to replay (the assembled
    // text alone, fed back through a fresh splitter, reproduces the
    // same event sequence on hit).
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

    // -----------------------------------------------------------------------
    // Wave-2.6 W-ε — honest B5 closure: tests for newly-keyed params
    // -----------------------------------------------------------------------

    #[test]
    fn prompt_cache_miss_on_different_grammar_kind() {
        let tokens: Vec<u32> = vec![1, 2, 3];
        use super::super::grammar::parser::{GretElement, GretType, Grammar};
        use std::collections::HashMap;

        // Build a trivial grammar used by both base and req.
        let grammar = Grammar {
            rules: vec![vec![
                GretElement::new(GretType::Char, b'x' as u32),
                GretElement::new(GretType::End, 0),
            ]],
            symbol_ids: { let mut m = HashMap::new(); m.insert("root".to_string(), 0u32); m },
        };

        let mut base = SamplingParams::default();
        base.grammar = Some(grammar.clone());
        base.grammar_kind = GrammarKind::ResponseFormat;
        let cache = make_cached(&tokens, &base, "x");

        // Same grammar, but ToolCallBodyAuto kind — enforcement timing differs → MISS.
        let mut req = SamplingParams::default();
        req.grammar = Some(grammar);
        req.grammar_kind = GrammarKind::ToolCallBodyAuto;
        assert!(
            cache.lookup(&tokens, &req).is_none(),
            "same grammar + different grammar_kind must not hit cache \
             (ResponseFormat enforces unconditionally; ToolCallBodyAuto is trigger-gated)"
        );
    }

    #[test]
    fn prompt_cache_miss_on_different_frequency_penalty() {
        let tokens: Vec<u32> = vec![1, 2, 3];
        let mut base = SamplingParams::default();
        base.frequency_penalty = 0.0;
        let cache = make_cached(&tokens, &base, "hello");

        let mut req = SamplingParams::default();
        req.frequency_penalty = 0.5; // non-default — must MISS
        assert!(
            cache.lookup(&tokens, &req).is_none(),
            "same prompt + different frequency_penalty must not hit cache"
        );
    }

    #[test]
    fn prompt_cache_miss_on_different_presence_penalty() {
        let tokens: Vec<u32> = vec![1, 2, 3];
        let mut base = SamplingParams::default();
        base.presence_penalty = 0.0;
        let cache = make_cached(&tokens, &base, "hello");

        let mut req = SamplingParams::default();
        req.presence_penalty = 0.3; // non-default — must MISS
        assert!(
            cache.lookup(&tokens, &req).is_none(),
            "same prompt + different presence_penalty must not hit cache"
        );
    }

    #[test]
    fn prompt_cache_miss_on_different_min_p() {
        let tokens: Vec<u32> = vec![1, 2, 3];
        let mut base = SamplingParams::default();
        base.min_p = 0.0;
        let cache = make_cached(&tokens, &base, "hello");

        let mut req = SamplingParams::default();
        req.min_p = 0.1; // non-default — must MISS
        assert!(
            cache.lookup(&tokens, &req).is_none(),
            "same prompt + different min_p must not hit cache"
        );
    }

    #[test]
    fn prompt_cache_miss_on_different_tool_call_policy() {
        let tokens: Vec<u32> = vec![1, 2, 3];
        let mut base = SamplingParams::default();
        base.tool_call_policy = ToolCallPolicy::Auto;
        let cache = make_cached(&tokens, &base, r#"{"name":"fn"}"#);

        let mut req = SamplingParams::default();
        req.tool_call_policy = ToolCallPolicy::Constrained; // different policy — must MISS
        assert!(
            cache.lookup(&tokens, &req).is_none(),
            "same prompt + different tool_call_policy must not hit cache \
             (Constrained promotes parse failures; Auto falls back to content)"
        );
    }

    #[test]
    fn prompt_cache_miss_on_different_logprobs() {
        let tokens: Vec<u32> = vec![1, 2, 3];
        let mut base = SamplingParams::default();
        base.logprobs = false;
        let cache = make_cached(&tokens, &base, "hello");

        let mut req = SamplingParams::default();
        req.logprobs = true; // logprob data requested — different response shape → MISS
        assert!(
            cache.lookup(&tokens, &req).is_none(),
            "same prompt + logprobs=true vs false must not hit cache \
             (response shape differs: logprob entries present vs absent)"
        );
    }

    #[test]
    fn prompt_cache_miss_on_different_top_logprobs() {
        let tokens: Vec<u32> = vec![1, 2, 3];
        let mut base = SamplingParams::default();
        base.logprobs = true;
        base.top_logprobs = 2;
        let cache = make_cached(&tokens, &base, "hello");

        let mut req = SamplingParams::default();
        req.logprobs = true;
        req.top_logprobs = 5; // different number of alternatives — must MISS
        assert!(
            cache.lookup(&tokens, &req).is_none(),
            "same prompt + different top_logprobs must not hit cache \
             (response shape differs: number of top alternatives)"
        );
    }

    #[test]
    fn prompt_cache_miss_on_different_parallel_tool_calls() {
        let tokens: Vec<u32> = vec![1, 2, 3];
        let mut base = SamplingParams::default();
        base.parallel_tool_calls = true;
        let cache = make_cached(&tokens, &base, "hello");

        let mut req = SamplingParams::default();
        req.parallel_tool_calls = false; // non-default — must MISS
        assert!(
            cache.lookup(&tokens, &req).is_none(),
            "same prompt + different parallel_tool_calls must not hit cache"
        );
    }

    /// Full-inventory hit test: all generation-affecting params identical
    /// including the wave-2.6 W-ε additions.
    #[test]
    fn prompt_cache_hit_full_inventory_equal() {
        use super::super::grammar::parser::{GretElement, GretType, Grammar};
        use std::collections::HashMap;

        let tokens: Vec<u32> = vec![10, 20, 30];

        let grammar = Grammar {
            rules: vec![vec![
                GretElement::new(GretType::Char, b'z' as u32),
                GretElement::new(GretType::End, 0),
            ]],
            symbol_ids: { let mut m = HashMap::new(); m.insert("root".to_string(), 0u32); m },
        };

        let mut params = SamplingParams::default();
        params.max_tokens = 64;
        params.stop_strings = vec!["END".to_string()];
        params.logit_bias.insert(99, -1.0);
        params.grammar = Some(grammar.clone());
        params.grammar_kind = GrammarKind::ResponseFormat;
        params.frequency_penalty = 0.1;
        params.presence_penalty = 0.2;
        params.min_p = 0.05;
        params.tool_call_policy = ToolCallPolicy::Auto;
        params.logprobs = true;
        params.top_logprobs = 3;
        params.parallel_tool_calls = false;
        let cache = make_cached(&tokens, &params, "full-inventory-hit");

        // Identical params → must HIT.
        let mut same = SamplingParams::default();
        same.max_tokens = 64;
        same.stop_strings = vec!["END".to_string()];
        same.logit_bias.insert(99, -1.0);
        same.grammar = Some(grammar);
        same.grammar_kind = GrammarKind::ResponseFormat;
        same.frequency_penalty = 0.1;
        same.presence_penalty = 0.2;
        same.min_p = 0.05;
        same.tool_call_policy = ToolCallPolicy::Auto;
        same.logprobs = true;
        same.top_logprobs = 3;
        same.parallel_tool_calls = false;

        let hit = cache.lookup(&tokens, &same);
        assert!(
            hit.is_some(),
            "identical full-inventory params must hit cache"
        );
        assert_eq!(hit.unwrap().text, "full-inventory-hit");
    }
}

// ---------------------------------------------------------------------------
// Wave 3 W-A2 — streaming PromptCache replay tests
//
// Drive `replay_cached_streaming_response` directly through a real
// `mpsc::channel`, drain the receiver, and assert SSE event shape.
// Single-shot per test — no full engine, no live model load.  Mirrors the
// same direct-helper pattern wave-2.8 finalize_streaming_tool_state_tests
// used (no sham reconstruction of the production codepath).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod streaming_prompt_cache_replay_tests {
    use super::*;
    use super::super::sse::{DeltaKind, GenerationEvent};

    /// Build a `GenerationResult` that looks like a non-streaming-origin
    /// cache entry (post-reasoning-split text + explicit reasoning_text).
    fn cached_non_streaming(text: &str, reasoning: Option<&str>) -> GenerationResult {
        GenerationResult {
            text: text.to_string(),
            reasoning_text: reasoning.map(|s| s.to_string()),
            prompt_tokens: 7,
            completion_tokens: 5,
            reasoning_tokens: reasoning.map(|_| 3),
            finish_reason: "stop",
            prefill_duration: Duration::ZERO,
            decode_duration: Duration::ZERO,
            cached_tokens: 7,
        }
    }

    /// Drain all events the helper produces synchronously.  Helper writes
    /// to a tokio mpsc via `blocking_send`, which works against a tokio
    /// receiver from a non-async context if the channel has capacity (we
    /// use 32, well over what any single replay needs).
    fn drain(rx: &mut mpsc::Receiver<GenerationEvent>) -> Vec<GenerationEvent> {
        let mut out = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            out.push(ev);
        }
        out
    }

    #[test]
    fn replay_emits_content_then_done_for_plain_text() {
        let (tx, mut rx) = mpsc::channel(32);
        let cached = cached_non_streaming("Hello, world!", None);

        let res = replay_cached_streaming_response(
            &cached,
            None, // no registration ⇒ everything routes as Content
            ToolCallPolicy::Auto,
            &tx,
        );
        assert!(res.is_ok(), "replay must succeed when no client disconnect");

        // Drop the sender so try_recv finds events without blocking.
        drop(tx);
        let events = drain(&mut rx);

        // Expected: 1 Delta(Content) + 1 Done.
        assert_eq!(events.len(), 2, "got events: {events:?}");
        match &events[0] {
            GenerationEvent::Delta { kind: DeltaKind::Content, text } => {
                assert_eq!(text, "Hello, world!");
            }
            other => panic!("expected Delta(Content); got {other:?}"),
        }
        match &events[1] {
            GenerationEvent::Done {
                finish_reason,
                prompt_tokens,
                completion_tokens,
                stats,
            } => {
                assert_eq!(*finish_reason, "stop");
                assert_eq!(*prompt_tokens, 7);
                assert_eq!(*completion_tokens, 5);
                // Cache-hit signal: cached_prompt_tokens populated, timings zeroed.
                assert_eq!(stats.cached_prompt_tokens, Some(7));
                assert_eq!(stats.prefill_time_secs, Some(0.0));
                assert_eq!(stats.decode_time_secs, Some(0.0));
            }
            other => panic!("expected Done; got {other:?}"),
        }
    }

    #[test]
    fn replay_emits_reasoning_then_content_when_reasoning_text_set() {
        let (tx, mut rx) = mpsc::channel(32);
        // Non-streaming-origin entry: reasoning was split out into its own
        // field; the assembled `text` is post-split content only.
        let cached = cached_non_streaming("the answer", Some("let me think..."));

        let res = replay_cached_streaming_response(
            &cached,
            None,
            ToolCallPolicy::Auto,
            &tx,
        );
        assert!(res.is_ok());
        drop(tx);
        let events = drain(&mut rx);

        // Expected: Reasoning, Content, Done.
        assert_eq!(events.len(), 3, "got events: {events:?}");
        match &events[0] {
            GenerationEvent::Delta { kind: DeltaKind::Reasoning, text } => {
                assert_eq!(text, "let me think...");
            }
            other => panic!("expected Delta(Reasoning); got {other:?}"),
        }
        match &events[1] {
            GenerationEvent::Delta { kind: DeltaKind::Content, text } => {
                assert_eq!(text, "the answer");
            }
            other => panic!("expected Delta(Content); got {other:?}"),
        }
        assert!(matches!(events[2], GenerationEvent::Done { .. }));
    }

    /// Replay a cache entry whose `text` contains tool-call markers
    /// (mirrors a streaming-origin cache entry where `accumulated_text`
    /// captures the raw pre-split stream).  The replay must re-route
    /// through the live-decode tool-call splitter so the SSE shape is
    /// `ToolCallDelta` events, not raw content text.
    #[test]
    fn replay_routes_tool_call_markers_to_tool_call_delta_events() {
        let (tx, mut rx) = mpsc::channel(32);

        // Use the gemma4 registration so we have real tool open/close
        // markers + a body-parser registered.
        let reg = match super::super::registry::find_for("gemma4-27b-it") {
            Some(r) => r,
            None => {
                eprintln!("gemma4 registration absent; skipping tool-call replay test");
                return;
            }
        };
        let (open, close) = match (reg.tool_open, reg.tool_close) {
            (Some(o), Some(c)) => (o, c),
            _ => {
                eprintln!("gemma4 has no tool markers; skipping tool-call replay test");
                return;
            }
        };

        // Construct a cached `text` shaped like Gemma 4's tool-call output:
        //     "preamble<open>{"name":"foo","arguments":{}}<close>postscript"
        // The body uses the per-model parser-friendly shape; since we don't
        // know gemma4's exact body grammar offline, the assertion focuses
        // on event-class-shape (Content + ToolCallDelta + Content) — NOT
        // on whether parse succeeds.  Both Some(parsed) → ToolCallDelta×2
        // and None (under Auto) → Content fallback are valid replay
        // shapes per `emit_streaming_tool_call_close`'s policy matrix.
        let cached_text = format!(
            "preamble {open}{{\"name\":\"foo\",\"arguments\":{{}}}}{close} postscript"
        );
        let cached = GenerationResult {
            text: cached_text,
            reasoning_text: None,
            prompt_tokens: 4,
            completion_tokens: 9,
            reasoning_tokens: None,
            finish_reason: "stop",
            prefill_duration: Duration::ZERO,
            decode_duration: Duration::ZERO,
            cached_tokens: 4,
        };

        let res = replay_cached_streaming_response(
            &cached,
            Some(&reg),
            ToolCallPolicy::Auto,
            &tx,
        );
        assert!(res.is_ok(), "replay must succeed");
        drop(tx);
        let events = drain(&mut rx);

        // Must contain at least one Delta (preamble) and a terminal Done.
        assert!(
            events.iter().any(|e| matches!(
                e,
                GenerationEvent::Delta { kind: DeltaKind::Content, text } if text.contains("preamble")
            )),
            "preamble must be emitted as Content delta; got {events:?}"
        );
        let done_idx = events.iter().position(|e| matches!(e, GenerationEvent::Done { .. }))
            .expect("Done event missing");
        assert_eq!(done_idx, events.len() - 1, "Done must be last event");

        // Extract the Done and assert cached_tokens surfaces.
        if let GenerationEvent::Done { stats, finish_reason, .. } = &events[done_idx] {
            assert_eq!(stats.cached_prompt_tokens, Some(4));
            // finish_reason: if the splitter drove ToolCallOpen+Close to
            // completion AND parser succeeded, we expect "tool_calls"; if
            // parser failed under Auto the body re-emits as content and
            // saw_tool_call stays false (cached.finish_reason="stop"
            // wins).  Both are valid here — the test asserts the BRANCH
            // wires correctly, not the per-model parser outcome.
            assert!(
                *finish_reason == "tool_calls" || *finish_reason == "stop",
                "finish_reason should be tool_calls or stop; got {finish_reason:?}"
            );
        }
    }

    /// Sanity: the streaming preroll lookup is gated on the same
    /// eligibility predicate as the non-streaming preroll.  An empty cache
    /// with a default-greedy request returns None — the lookup short-
    /// circuits and the live decode runs.  This is the miss path probe.
    #[test]
    fn empty_cache_lookup_returns_none() {
        let cache = PromptCache::new();
        let params = SamplingParams::default();
        let prompt = vec![1u32, 2, 3];
        assert!(
            cache.lookup(&prompt, &params).is_none(),
            "fresh cache must miss on first request"
        );
    }

    /// After `store`, the SAME prompt + params hits and produces a
    /// `GenerationResult` with `cached_tokens == prompt.len()`.  This is
    /// the same contract the non-streaming preroll relies on — the
    /// streaming replay just consumes that result.
    #[test]
    fn store_then_lookup_round_trips_for_replay() {
        let prompt = vec![10u32, 20, 30, 40];
        let params = SamplingParams::default();
        let result = GenerationResult {
            text: "cached body".into(),
            reasoning_text: None,
            prompt_tokens: prompt.len(),
            completion_tokens: 11,
            reasoning_tokens: None,
            finish_reason: "stop",
            prefill_duration: Duration::ZERO,
            decode_duration: Duration::ZERO,
            cached_tokens: 0,
        };
        let mut cache = PromptCache::new();
        cache.store(&prompt, &params, &result);

        let hit = cache
            .lookup(&prompt, &params)
            .expect("must hit after store");
        assert_eq!(hit.text, "cached body");
        assert_eq!(hit.cached_tokens, prompt.len());
        assert_eq!(hit.completion_tokens, 11);
        assert_eq!(hit.finish_reason, "stop");
    }

    /// Replay returns Err(()) when the receiver is dropped mid-replay —
    /// the production callsite then bumps the cancellation counter.  This
    /// exercises the disconnect path which mirrors the live decode's
    /// `events.blocking_send(...).is_err()` checks.
    #[test]
    fn replay_returns_err_when_receiver_dropped() {
        let (tx, rx) = mpsc::channel(1); // tiny buffer
        // Drop receiver so all sends fail.
        drop(rx);

        let cached = cached_non_streaming("anything", None);
        let res = replay_cached_streaming_response(
            &cached,
            None,
            ToolCallPolicy::Auto,
            &tx,
        );
        assert!(
            res.is_err(),
            "replay must return Err when receiver was dropped"
        );
    }

    // ---------------------------------------------------------------------
    // Wave 3.5 HIGH-2 — audit-driven splitter-drain tests
    //
    // Audit divergence
    // /tmp/cfa-cfa-20260427-adr005-wave3/codex-review-last.txt
    // "W-A2 streaming cache replay" severity HIGH:
    //
    //   "replay_cached_streaming_response feeds cached.text once at
    //    src/serve/api/engine.rs:2977-3012 and immediately emits Done
    //    at src/serve/api/engine.rs:3015-3048. It never calls
    //    ReasoningSplitter::finish() or ToolCallSplitter::finish(),
    //    even though those splitters hold back tail bytes until
    //    finish at src/serve/api/registry.rs:397-463 and
    //    src/serve/api/registry.rs:587-658. Registered plain-text
    //    cache hits can therefore emit empty/truncated content."
    //
    // The two missed-test gaps the audit cited:
    //   1. "No unit test replays short plain content with a registered
    //      model; replay_emits_content_then_done_for_plain_text passes
    //      registration=None ... bypassing both tail-holding splitters."
    //   2. "No replay test asserts final postscript/tail content after
    //      tool-call markers."
    //
    // The tests below close both gaps and would fail on a regression
    // that removes the new finish() drain calls.
    // ---------------------------------------------------------------------

    /// Wave 3.5 HIGH-2 missed-test #1: a registered-model plain-text
    /// replay must drain the splitter tail before Done.
    ///
    /// The Gemma 4 ToolCallSplitter has `tail_cap = max(open_marker.len,
    /// close_marker.len) = max(12, 12) = 12 bytes` (registry.rs:565).
    /// Cached text shorter than `tail_cap` ends up entirely in the
    /// splitter's tail_buf — `feed()` emits zero events, `finish()` is
    /// the only way to recover the bytes.  Pre-Wave-3.5 replay never
    /// called `finish()`, so the entire response was lost.
    #[test]
    fn replay_emits_tail_content_after_splitter_drain() {
        let reg = match super::super::registry::find_for("gemma4-27b-it") {
            Some(r) => r,
            None => {
                eprintln!("gemma4 registration absent; skipping HIGH-2 drain test");
                return;
            }
        };

        let (tx, mut rx) = mpsc::channel(8);
        // Short plain-text cache entry (< Gemma's 12-byte marker
        // tail_cap).  No marker, no reasoning — pure content.  This
        // is the exact "registered plain-text cache hit" shape the
        // audit cited.
        let cached = cached_non_streaming("hi", None);

        let res = replay_cached_streaming_response(
            &cached,
            Some(&reg), // <-- KEY: registration enables splitter (the bug only fires when splitter is built)
            ToolCallPolicy::Auto,
            &tx,
        );
        assert!(res.is_ok(), "replay must succeed");
        drop(tx);
        let events = drain(&mut rx);

        // The cached "hi" MUST appear as a Content delta before Done.
        // Pre-Wave-3.5: the splitter's tail_buf swallowed "hi" entirely
        // because feed() held back the last `tail_cap` bytes and
        // finish() was never called → zero Content deltas → silent
        // data loss on the cache hit.
        let content_text: String = events
            .iter()
            .filter_map(|e| match e {
                GenerationEvent::Delta { kind: DeltaKind::Content, text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            content_text, "hi",
            "registered plain-text cache replay MUST emit the full \
             cached content as Content delta(s) before Done.  \
             Pre-Wave-3.5 the splitter's tail_buf silently swallowed \
             content shorter than tail_cap (12 bytes for Gemma 4) \
             because finish() was never called.  Audit citation: \
             /tmp/cfa-cfa-20260427-adr005-wave3/codex-review-last.txt \
             'W-A2 streaming cache replay' severity HIGH.\n\
             events: {events:?}"
        );

        // Done must follow.
        assert!(
            matches!(events.last(), Some(GenerationEvent::Done { .. })),
            "Done must be the last event"
        );
    }

    /// Wave 3.5 HIGH-2 missed-test #2: a registered-model replay whose
    /// cached text contains a tool-call marker block PLUS trailing
    /// postscript content must emit BOTH the structured tool-call AND
    /// the postscript content.
    ///
    /// Pre-Wave-3.5 the postscript portion shorter than the splitter's
    /// `tail_cap` bytes would be silently dropped, OR a postscript
    /// whose tail looked like a partial open-marker prefix would be
    /// held back forever.
    #[test]
    fn replay_with_registered_model_emits_tool_call_then_postscript() {
        let reg = match super::super::registry::find_for("gemma4-27b-it") {
            Some(r) => r,
            None => {
                eprintln!("gemma4 registration absent; skipping HIGH-2 postscript test");
                return;
            }
        };
        let (open, close) = match (reg.tool_open, reg.tool_close) {
            (Some(o), Some(c)) => (o, c),
            _ => {
                eprintln!("gemma4 has no tool markers; skipping HIGH-2 postscript test");
                return;
            }
        };

        // Cached text: tool-call block + trailing postscript shorter
        // than `tail_cap` (Gemma's max marker length is 12 bytes; a
        // 5-byte postscript "after" sits entirely in the splitter's
        // tail_buf after feed() returns and is only recoverable via
        // finish()).
        let cached_text = format!(
            "{open}call:foo{{x:1}}{close}after"
        );
        let cached = GenerationResult {
            text: cached_text,
            reasoning_text: None,
            prompt_tokens: 4,
            completion_tokens: 9,
            reasoning_tokens: None,
            finish_reason: "stop",
            prefill_duration: Duration::ZERO,
            decode_duration: Duration::ZERO,
            cached_tokens: 4,
        };

        let (tx, mut rx) = mpsc::channel(16);
        let res = replay_cached_streaming_response(
            &cached,
            Some(&reg),
            ToolCallPolicy::Auto,
            &tx,
        );
        assert!(res.is_ok(), "replay must succeed");
        drop(tx);
        let events = drain(&mut rx);

        // Concatenate ALL Content deltas — the postscript "after" MUST
        // appear somewhere.  Pre-Wave-3.5 the splitter's finish() was
        // never called and "after" was silently dropped.
        let content_concat: String = events
            .iter()
            .filter_map(|e| match e {
                GenerationEvent::Delta { kind: DeltaKind::Content, text } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert!(
            content_concat.contains("after"),
            "postscript content 'after' MUST appear in a Content delta \
             after the tool-call block.  Pre-Wave-3.5 the ToolCallSplitter \
             held the postscript in its tail_buf and finish() was never \
             called → silent postscript loss.  Audit citation: \
             /tmp/cfa-cfa-20260427-adr005-wave3/codex-review-last.txt \
             missed-test 'No replay test asserts final postscript/tail \
             content after tool-call markers'.\n\
             content concat: {content_concat:?}\nevents: {events:?}"
        );

        // Wave 3.6 W-4 strengthening (audit gap from
        // /tmp/cfa-cfa-20260427-adr005-wave3.5/codex-review-last.txt):
        //
        // "asserts postscript content and Done, but does not assert
        //  ToolCallDelta or finish_reason=tool_calls for the parsed
        //  marker block."
        //
        // The cached text is `{open}call:foo{{x:1}}{close}after`.
        // The body `call:foo{{x:1}}` is parseable by parse_gemma4_tool_call
        // → parse_tool_call_body returns Some(ParsedToolCall{name:"foo",
        //   args:{"x":1}}).  emit_streaming_tool_call_close then emits
        // two ToolCallDelta events (name chunk + args chunk) and sets
        // saw_tool_call=true → Done gets finish_reason="tool_calls".
        //
        // The splitter chain must re-classify the marker block into
        // structured ToolCallDelta events identical to a fresh decode.
        let tool_call_deltas: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, GenerationEvent::ToolCallDelta { .. }))
            .collect();
        assert!(
            !tool_call_deltas.is_empty(),
            "Wave 3.6 W-4: MUST emit at least one ToolCallDelta for the \
             parsed `call:foo{{x:1}}` body — the splitter chain re-classifies \
             the marker block into structured ToolCall deltas.  \
             events: {events:?}"
        );

        // The first ToolCallDelta MUST carry the function name (name chunk).
        let has_name_delta = events.iter().any(|e| matches!(
            e,
            GenerationEvent::ToolCallDelta { name: Some(_), .. }
        ));
        assert!(
            has_name_delta,
            "Wave 3.6 W-4: the first ToolCallDelta MUST carry `name: Some(...)` \
             (function name = \"foo\"); subsequent deltas carry arguments. \
             tool_call_deltas: {tool_call_deltas:?}"
        );

        // The Done event MUST report finish_reason="tool_calls" because
        // saw_tool_call is set by emit_streaming_tool_call_close when parse
        // succeeds (engine.rs:3184: `if saw_tool_call { "tool_calls" } else ...`).
        // Cached text has finish_reason="stop" but the replay overrides it.
        let done_finish_reason = events
            .iter()
            .find_map(|e| match e {
                GenerationEvent::Done { finish_reason, .. } => Some(*finish_reason),
                _ => None,
            })
            .expect("Done event must be present");
        assert_eq!(
            done_finish_reason,
            "tool_calls",
            "Wave 3.6 W-4: finish_reason MUST be 'tool_calls' (not '{}') when \
             a tool call was extracted from the cached text during replay. \
             The replay overrides cached.finish_reason (='stop') with \
             'tool_calls' when saw_tool_call=true (engine.rs:3184). \
             events: {events:?}",
            done_finish_reason
        );

        // Sanity: Done must terminate the stream.
        assert!(
            matches!(events.last(), Some(GenerationEvent::Done { .. })),
            "Done must be the last event"
        );
    }

    /// Wave 3.5 HIGH-2 — drain the ReasoningSplitter tail too.
    ///
    /// Cached text contains reasoning markers + a short tail of
    /// content.  Pre-Wave-3.5 the ReasoningSplitter's `finish()` was
    /// never called and the residual tail was lost.
    #[test]
    fn replay_drains_reasoning_splitter_tail() {
        let reg = match super::super::registry::find_for("gemma4-27b-it") {
            Some(r) => r,
            None => {
                eprintln!("gemma4 registration absent; skipping HIGH-2 reasoning drain test");
                return;
            }
        };

        // Build a cached text whose final bytes are a content tail
        // shorter than the reasoning_splitter's tail_cap.  We don't
        // know the exact reasoning markers offline; we just probe the
        // drain semantics by feeding short content with no reasoning.
        // Combined with the tool-call splitter the reasoning drain
        // path is exercised through the registered registration.
        let cached = cached_non_streaming("ok", None);

        let (tx, mut rx) = mpsc::channel(8);
        let res = replay_cached_streaming_response(
            &cached,
            Some(&reg),
            ToolCallPolicy::Auto,
            &tx,
        );
        assert!(res.is_ok(), "replay must succeed");
        drop(tx);
        let events = drain(&mut rx);

        let content_concat: String = events
            .iter()
            .filter_map(|e| match e {
                GenerationEvent::Delta { kind: DeltaKind::Content, text } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            content_concat, "ok",
            "short content 'ok' MUST traverse both reasoning_splitter \
             AND tool_splitter via the new finish() drain calls; a \
             regression that drops EITHER drain would lose the bytes \
             because both splitters' tail_buf can swallow 2 bytes \
             entirely.\nevents: {events:?}"
        );
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

    /// Wave 2.6 W-α5 Q2 — replacement for the wave-2.5
    /// `grammar_active_atomic_bool_transitions` test.
    ///
    /// The wave-2.5 architecture used a sibling `Arc<AtomicBool>
    /// grammar_active` toggled by ToolCallOpen/Close.  The audit caught
    /// it as architecturally wrong (mask + advance + dead-check could
    /// disagree because they read different state).  Wave 2.6 moves the
    /// gate INSIDE GrammarRuntime via `awaiting_trigger` — the production
    /// streaming worker now wires `route_content`'s ToolCallOpen handler
    /// to call `runtime.trigger()` directly.  This test exercises that
    /// exact production path: a real registered tool-call splitter, a
    /// real GrammarRuntime, and the same trigger-on-open pattern
    /// `route_content` uses.
    #[test]
    fn tool_call_open_triggers_grammar_runtime() {
        use crate::serve::api::grammar::GrammarRuntime;
        use crate::serve::api::grammar::parser::parse;

        // Real Gemma4 registration with real open/close markers.
        let reg = crate::serve::api::registry::find_for("gemma4-27b-it")
            .expect("gemma4 registration must exist");
        let (open, close) = match (reg.tool_open, reg.tool_close) {
            (Some(o), Some(c)) => (o, c),
            _ => {
                eprintln!("gemma4 has no tool markers — skip Q2 trigger test");
                return;
            }
        };
        let mut splitter =
            crate::serve::api::registry::ToolCallSplitter::from_registration(&reg)
                .expect("ToolCallSplitter::from_registration must return Some for gemma4");

        // Build a real GrammarRuntime in the lazy state, the way
        // generate_stream_once does for `GrammarKind::ToolCallBodyAuto`.
        let g = parse("root ::= \"x\"\n").expect("parse");
        let rid = g.rule_id("root").expect("root rule");
        let mut runtime = GrammarRuntime::new(g, rid).expect("runtime");
        runtime.set_awaiting_trigger(true);
        assert!(
            runtime.is_awaiting_trigger(),
            "lazy-grammar runtime starts in awaiting_trigger=true (production setup for ToolCallBody-kind requests)"
        );

        // Pre-open: feeding splitter with content that doesn't include
        // the open marker emits no ToolCallOpen and so the production
        // code does NOT call runtime.trigger().  The runtime stays
        // suspended.
        let _events = splitter.feed("plain preamble text ");
        assert!(
            runtime.is_awaiting_trigger(),
            "preamble fragments MUST NOT trigger the runtime"
        );

        // Production trigger pattern (mirrors route_content's
        // ToolCallOpen branch in generate_stream_once):
        let events_open = splitter.feed(open);
        if events_open.iter().any(|e| matches!(
            e,
            crate::serve::api::registry::ToolCallEvent::ToolCallOpen
        )) {
            runtime.trigger();
        }
        assert!(
            !runtime.is_awaiting_trigger(),
            "after ToolCallOpen the production code MUST flip the runtime trigger"
        );

        // Post-open: feeding the close marker through the splitter
        // does NOT reset the runtime — multi-call grammars handle
        // re-entry via `(call)+` shape (research-report.md Q2 anti-
        // finding; /opt/llama.cpp/docs/function-calling.md).
        let _events_close = splitter.feed(close);
        assert!(
            !runtime.is_awaiting_trigger(),
            "ToolCallClose MUST NOT re-arm the runtime trigger \
             (llama.cpp parity: PR #9639 lazy grammar is one-shot per request)"
        );
    }
}

// ---------------------------------------------------------------------------
// Wave 2.8 W-θ HIGH-1 — finalize_streaming_tool_state
//
// Audit-driver tests: exercise the EXACT streaming SSE event chain through
// `tool_splitter.finish()` drain + post-drain Constrained no-call check.
// Tests drive the production helper directly with a real `mpsc::channel`,
// drain the receiver, and assert SSE event shape — NOT a stand-in
// reconstruction (the wave-2 sham-test pattern Codex caught).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod finalize_streaming_tool_state_tests {
    use super::*;
    use crate::serve::api::registry::{self, ToolCallSplitter};
    use crate::serve::api::sse::{DeltaKind, GenerationEvent};
    use tokio::sync::mpsc;

    fn gemma4_reg() -> registry::ModelRegistration {
        registry::find_for("gemma4-27b-it").expect("gemma4 registration must exist")
    }

    /// Drain the receiver synchronously (we are inside a single-threaded
    /// helper that uses `blocking_send`; the matching consumer is
    /// `try_recv` after the helper returns).
    fn drain_recv(rx: &mut mpsc::Receiver<GenerationEvent>) -> Vec<GenerationEvent> {
        let mut out = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            out.push(ev);
        }
        out
    }

    /// Streaming Constrained mid-call truncation: the splitter has seen the
    /// open marker but not the close marker, so `finish()` returns
    /// `ToolCallText(residual)`. Under Constrained policy the helper MUST
    /// emit `GenerationEvent::Error("tool_call_truncated_under_constrained")`
    /// and return `ErrorEmitted` (so the streaming driver skips Done). It
    /// MUST NOT emit Content (the silent-fallback wave-2.6 audit divergence).
    #[test]
    fn streaming_constrained_mid_call_truncation_yields_error_event() {
        let reg = gemma4_reg();
        let mut splitter = ToolCallSplitter::from_registration(&reg)
            .expect("gemma4 has tool markers, splitter must build");

        // Drive the splitter the same way the engine does: feed bytes that
        // include the open marker and a partial body (no close marker).
        // After this, splitter.in_tool_call() == true and tail_buf holds
        // the partial body.
        let open = reg.tool_open.expect("gemma4 has tool_open");
        let _ = splitter.feed(&format!("{open}call:get_weather{{"));
        assert!(
            splitter.in_tool_call(),
            "splitter must be in_tool_call after open marker; finish() will \
             then return ToolCallText (mid-call truncation)"
        );

        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(8);

        let action = finalize_streaming_tool_state(
            Some(&mut splitter),
            ToolCallPolicy::Constrained,
            /* saw_tool_call */ false,
            Some(&reg),
            /* completion_tokens */ 7,
            /* accumulated_text_len */ 18,
            &tx,
        );

        assert_eq!(
            action,
            FinalizeStreamingAction::ErrorEmitted,
            "Constrained + ToolCallText residual MUST return ErrorEmitted"
        );

        // Close the sender so try_recv terminates cleanly.
        drop(tx);
        let events = drain_recv(&mut rx);

        assert_eq!(
            events.len(),
            1,
            "exactly one error event expected, got: {:?}",
            events
        );
        match &events[0] {
            GenerationEvent::Error(code) => {
                assert_eq!(
                    code, "tool_call_truncated_under_constrained",
                    "structured error code must match defensive 500 vocabulary"
                );
            }
            other => panic!(
                "expected GenerationEvent::Error, got: {:?} \
                 (silent Content fallback would be the wave-2.6 audit divergence)",
                other
            ),
        }
    }

    /// Streaming Constrained no-call: `saw_tool_call == false` and the
    /// splitter has nothing buffered (decode finished without ever entering
    /// a tool-call span). Under Constrained policy the helper MUST emit
    /// `GenerationEvent::Error("tool_call_no_call_under_constrained")` and
    /// return `ErrorEmitted` BEFORE Done. Mirrors the non-streaming check
    /// in handlers.rs:410-444 (commit da545d5).
    #[test]
    fn streaming_constrained_no_call_yields_error_event() {
        let reg = gemma4_reg();
        let mut splitter = ToolCallSplitter::from_registration(&reg)
            .expect("gemma4 has tool markers, splitter must build");
        // No feed → splitter idle, finish() returns None.
        assert!(
            !splitter.in_tool_call(),
            "splitter must be idle for the no-call test"
        );

        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(8);

        let action = finalize_streaming_tool_state(
            Some(&mut splitter),
            ToolCallPolicy::Constrained,
            /* saw_tool_call */ false,
            Some(&reg),
            /* completion_tokens */ 64,
            /* accumulated_text_len */ 0,
            &tx,
        );

        assert_eq!(
            action,
            FinalizeStreamingAction::ErrorEmitted,
            "Constrained + saw_tool_call=false MUST return ErrorEmitted"
        );

        drop(tx);
        let events = drain_recv(&mut rx);

        assert_eq!(events.len(), 1, "exactly one error event expected");
        match &events[0] {
            GenerationEvent::Error(code) => {
                assert_eq!(
                    code, "tool_call_no_call_under_constrained",
                    "structured error code must match defensive 500 vocabulary"
                );
            }
            other => panic!("expected GenerationEvent::Error, got: {:?}", other),
        }
    }

    /// Auto policy mid-call truncation: Auto allows partial / malformed
    /// tool-call syntax. The helper MUST emit the residual as `Content`
    /// (with the literal open marker re-prepended for diagnostic clarity)
    /// and return `Continue` (caller emits Done normally). This is the
    /// pre-2.8 behaviour we MUST preserve.
    #[test]
    fn streaming_auto_mid_call_truncation_emits_content_fallback() {
        let reg = gemma4_reg();
        let mut splitter = ToolCallSplitter::from_registration(&reg).unwrap();
        let open = reg.tool_open.expect("gemma4 has tool_open");
        let _ = splitter.feed(&format!("{open}call:get_weather{{"));
        assert!(splitter.in_tool_call());

        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(8);

        let action = finalize_streaming_tool_state(
            Some(&mut splitter),
            ToolCallPolicy::Auto,
            /* saw_tool_call */ false,
            Some(&reg),
            /* completion_tokens */ 7,
            /* accumulated_text_len */ 18,
            &tx,
        );
        assert_eq!(
            action,
            FinalizeStreamingAction::Continue,
            "Auto policy MUST preserve pre-2.8 Content-fallback behaviour"
        );
        drop(tx);
        let events = drain_recv(&mut rx);
        assert_eq!(events.len(), 1, "exactly one Content delta expected");
        match &events[0] {
            GenerationEvent::Delta { kind, text } => {
                assert!(matches!(kind, DeltaKind::Content));
                assert!(
                    text.contains(open),
                    "Auto fallback re-prepends the literal open marker for \
                     diagnostic clarity (so the operator sees the truncation \
                     in delta.content); got: {text:?}"
                );
            }
            other => panic!("expected Content delta, got: {:?}", other),
        }
    }

    /// Auto policy no-call: a turn that never produced a tool call is the
    /// normal Auto outcome. The helper MUST return `Continue` and emit no
    /// events.
    #[test]
    fn streaming_auto_no_call_emits_no_events() {
        let reg = gemma4_reg();
        let mut splitter = ToolCallSplitter::from_registration(&reg).unwrap();
        // Idle splitter, no feed.

        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(8);

        let action = finalize_streaming_tool_state(
            Some(&mut splitter),
            ToolCallPolicy::Auto,
            /* saw_tool_call */ false,
            Some(&reg),
            64,
            0,
            &tx,
        );
        assert_eq!(action, FinalizeStreamingAction::Continue);
        drop(tx);
        let events = drain_recv(&mut rx);
        assert!(
            events.is_empty(),
            "Auto + idle splitter must emit no finalize events; got {:?}",
            events
        );
    }

    /// Wave 3 W-B2 — AutoLazyGrammar mid-call truncation MUST yield the
    /// SAME loud-error event as Constrained.
    ///
    /// Under AutoLazyGrammar the per-model body grammar is active inside
    /// the tool-call span (post-trigger). A truncation past ToolCallOpen
    /// without ToolCallClose means decoding stopped mid-grammar — the
    /// runtime is neither accepted nor dead. Same regression signature
    /// as Constrained truncation; same `tool_call_truncated_under_constrained`
    /// error code (preserves the structured-vocabulary single source of
    /// truth so log/metrics matchers continue to work unchanged).
    #[test]
    fn streaming_auto_lazy_grammar_mid_call_truncation_yields_error_event() {
        let reg = gemma4_reg();
        let mut splitter = ToolCallSplitter::from_registration(&reg)
            .expect("gemma4 has tool markers, splitter must build");

        let open = reg.tool_open.expect("gemma4 has tool_open");
        let _ = splitter.feed(&format!("{open}call:get_weather{{"));
        assert!(
            splitter.in_tool_call(),
            "splitter must be in_tool_call after open marker; finish() will \
             then return ToolCallText (mid-call truncation)"
        );

        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(8);

        let action = finalize_streaming_tool_state(
            Some(&mut splitter),
            ToolCallPolicy::AutoLazyGrammar,
            /* saw_tool_call */ false,
            Some(&reg),
            /* completion_tokens */ 7,
            /* accumulated_text_len */ 18,
            &tx,
        );

        assert_eq!(
            action,
            FinalizeStreamingAction::ErrorEmitted,
            "AutoLazyGrammar + ToolCallText residual MUST return ErrorEmitted \
             identically to Constrained — the lazy grammar IS active inside \
             the body so a truncation is a regression"
        );

        drop(tx);
        let events = drain_recv(&mut rx);

        assert_eq!(
            events.len(),
            1,
            "exactly one error event expected, got: {:?}",
            events
        );
        match &events[0] {
            GenerationEvent::Error(code) => {
                assert_eq!(
                    code, "tool_call_truncated_under_constrained",
                    "structured error code MUST match the unified vocabulary; \
                     AutoLazyGrammar reuses the Constrained code so log/metrics \
                     matchers continue to work unchanged"
                );
            }
            GenerationEvent::Delta { kind: DeltaKind::Content, text } => {
                panic!(
                    "REGRESSION: AutoLazyGrammar mid-call truncation emitted \
                     Content fallback (text={text:?}); the wave-3 W-B2 T2.4 \
                     final closure MUST promote this to Error"
                );
            }
            other => panic!(
                "expected GenerationEvent::Error, got: {:?} \
                 (silent Content fallback would be the wave-2.6 audit divergence)",
                other
            ),
        }
    }

    /// Wave 3 W-B2 — AutoLazyGrammar with NO call must NOT trigger the
    /// no-call check.
    ///
    /// Auto explicitly permits the model to emit zero tool calls
    /// (preamble freedom — the whole point of lazy grammar). A streaming
    /// run that ended without ever firing `ToolCallOpen` is the
    /// legitimate Auto-no-call path under AutoLazyGrammar, NOT a
    /// regression.  This is the key semantic distinction from
    /// `ToolCallPolicy::Constrained` (where the eager grammar's
    /// OneOrMoreCalls root mandates >= 1 call).
    #[test]
    fn streaming_auto_lazy_grammar_no_call_emits_no_events() {
        let reg = gemma4_reg();
        let mut splitter = ToolCallSplitter::from_registration(&reg).unwrap();
        // Idle splitter, no feed.
        assert!(!splitter.in_tool_call());

        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(8);

        let action = finalize_streaming_tool_state(
            Some(&mut splitter),
            ToolCallPolicy::AutoLazyGrammar,
            /* saw_tool_call */ false,
            Some(&reg),
            64,
            0,
            &tx,
        );
        assert_eq!(
            action,
            FinalizeStreamingAction::Continue,
            "AutoLazyGrammar + idle splitter MUST return Continue — Auto \
             explicitly allows the model to emit zero tool calls (preamble \
             freedom is the whole point of lazy grammar). The no-call check \
             stays Constrained-only."
        );
        drop(tx);
        let events = drain_recv(&mut rx);
        assert!(
            events.is_empty(),
            "AutoLazyGrammar + idle splitter must emit no finalize events; \
             got {:?}",
            events
        );
    }

    /// Constrained policy with `saw_tool_call == true` (the model produced
    /// at least one full call): no-call check MUST NOT fire even though
    /// policy is Constrained. The helper returns `Continue`.
    #[test]
    fn streaming_constrained_saw_tool_call_continues_to_done() {
        let reg = gemma4_reg();
        let mut splitter = ToolCallSplitter::from_registration(&reg).unwrap();
        // Drive a full call so splitter is idle and would have emitted a
        // ToolCallClose during streaming. We only care that
        // splitter.finish() returns None (no residual) and that the
        // post-drain no-call check sees saw_tool_call=true.
        let open = reg.tool_open.expect("open");
        let close = reg.tool_close.expect("close");
        let _ = splitter.feed(&format!("{open}call:foo{{}}{close}"));
        assert!(!splitter.in_tool_call());

        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(8);

        let action = finalize_streaming_tool_state(
            Some(&mut splitter),
            ToolCallPolicy::Constrained,
            /* saw_tool_call */ true,
            Some(&reg),
            12,
            18,
            &tx,
        );
        assert_eq!(action, FinalizeStreamingAction::Continue);
        drop(tx);
        let events = drain_recv(&mut rx);
        assert!(
            events.is_empty(),
            "Constrained + saw_tool_call=true must emit no finalize events; \
             got {:?}",
            events
        );
    }
}

// ---------------------------------------------------------------------------
// Wave 3 W-A3 — emit_streaming_tool_call_close (T2.4 partial removal)
// Wave 3 W-B2 — emit_streaming_tool_call_close (T2.4 final closure on
//               registered Auto-with-tools path)
//
// Audit-driver tests for the body-parse-failure branches of
// `emit_streaming_tool_call_close`.  Three scenarios:
//
//   1. `required_body_parse_failure_yields_error_not_content` (W-A3) —
//      Constrained policy + parse failure → GenerationEvent::Error
//      ("tool_call_unreachable_fallback_required"). MUST NOT emit Content.
//
//   2. `auto_lazy_grammar_body_parse_failure_yields_error_not_content`
//      (W-B2) — AutoLazyGrammar policy + parse failure →
//      GenerationEvent::Error("tool_call_unreachable_fallback_required").
//      MUST NOT emit Content.  Same loud-error promotion as Constrained
//      because the lazy grammar IS active inside the body.
//
//   3. `auto_body_parse_failure_preserves_content_fallback` (W-A3) —
//      Auto (no grammar) policy + parse failure →
//      GenerationEvent::Delta{Content, body_dump}.  Regression-preserve:
//      the content fallback for unconstrained Auto (no tools[] / unknown
//      family) is the defined behaviour and MUST NOT regress.
//
// All tests drive `emit_streaming_tool_call_close` directly with
// `parsed = None` (simulating a malformed body after ToolCallClose fires).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod emit_streaming_tool_call_close_tests {
    use super::*;
    use crate::serve::api::sse::{DeltaKind, GenerationEvent};
    use tokio::sync::mpsc;

    fn drain_recv(rx: &mut mpsc::Receiver<GenerationEvent>) -> Vec<GenerationEvent> {
        let mut out = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            out.push(ev);
        }
        out
    }

    /// T2.4 partial removal — Required path.
    ///
    /// Simulates: ToolCallText has accumulated malformed JSON ("garbage{{}}")
    /// into `body`, then ToolCallClose fires. `parse_tool_call_body` returns
    /// None.  Under Constrained policy `emit_streaming_tool_call_close` MUST:
    ///   - return `Err(())`
    ///   - emit exactly one `GenerationEvent::Error` with code
    ///     `"tool_call_unreachable_fallback_required"`
    ///   - NOT emit any `GenerationEvent::Delta { kind: Content, … }`
    ///
    /// This branch should be unreachable in correct operation (the eager
    /// grammar from wave-2.7 W-η da545d5 physically prevents a bad body).
    /// If it fires, it is a grammar-engine regression and must be loud.
    #[test]
    fn required_body_parse_failure_yields_error_not_content() {
        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(8);
        let mut tc_index: usize = 0;
        let mut saw_tc: bool = false;

        let result = emit_streaming_tool_call_close(
            None, // parsed = None: simulates a body that failed parse_tool_call_body
            "garbage{{}}".to_string(),
            ToolCallPolicy::Constrained,
            &mut tc_index,
            &mut saw_tc,
            &tx,
        );

        assert!(
            result.is_err(),
            "Constrained + parse failure MUST return Err(()) \
             (streaming driver aborts decode loop)"
        );
        assert_eq!(
            tc_index, 0,
            "tc_index must not be incremented on parse failure"
        );
        assert!(!saw_tc, "saw_tc must remain false on parse failure");

        drop(tx);
        let events = drain_recv(&mut rx);

        assert_eq!(
            events.len(),
            1,
            "exactly one GenerationEvent::Error expected; got: {:?}",
            events
        );
        match &events[0] {
            GenerationEvent::Error(code) => {
                assert_eq!(
                    code, "tool_call_unreachable_fallback_required",
                    "error code MUST be 'tool_call_unreachable_fallback_required' \
                     (wave 3 W-A3 T2.4 partial removal); old 'tool_call_parse_failure' \
                     code would indicate a regression to wave-2.5 A4 vocabulary"
                );
            }
            GenerationEvent::Delta { kind: DeltaKind::Content, text } => {
                panic!(
                    "REGRESSION: Constrained parse failure emitted Content fallback \
                     (text={text:?}); this is the T2.4 silent-fallback that W-A3 removes"
                );
            }
            other => panic!(
                "expected GenerationEvent::Error, got: {:?}",
                other
            ),
        }
    }

    /// Wave 3 W-B2 — T2.4 FINAL closure for the registered-Auto path.
    ///
    /// Simulates the same malformed body under
    /// `ToolCallPolicy::AutoLazyGrammar` — the policy the handler sets
    /// when `tool_choice=auto` AND the W-B2 lazy grammar IS active
    /// (tools[] non-empty AND model family registered AND
    /// `effective_grammar_kind == ToolCallBodyAuto`).
    /// `emit_streaming_tool_call_close` MUST treat this branch
    /// identically to Constrained:
    ///   - return `Err(())`
    ///   - emit exactly one `GenerationEvent::Error` with code
    ///     `"tool_call_unreachable_fallback_required"`
    ///   - NOT emit any `GenerationEvent::Delta { kind: Content, … }`
    ///
    /// Rationale: under AutoLazyGrammar the per-model body grammar is
    /// active inside the tool-call span (the `awaiting_trigger` flag is
    /// flipped by `route_content`'s ToolCallOpen handler before the
    /// body bytes are accepted by the runtime). A parse failure
    /// therefore means the lazy grammar engine produced structurally
    /// invalid output — same regression signature as Constrained.
    #[test]
    fn auto_lazy_grammar_body_parse_failure_yields_error_not_content() {
        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(8);
        let mut tc_index: usize = 0;
        let mut saw_tc: bool = false;

        let result = emit_streaming_tool_call_close(
            None,
            "garbage{{}}".to_string(),
            ToolCallPolicy::AutoLazyGrammar,
            &mut tc_index,
            &mut saw_tc,
            &tx,
        );

        assert!(
            result.is_err(),
            "AutoLazyGrammar + parse failure MUST return Err(()) \
             (streaming driver aborts decode loop, identical to Constrained)"
        );
        assert_eq!(
            tc_index, 0,
            "tc_index must not be incremented on parse failure"
        );
        assert!(!saw_tc, "saw_tc must remain false on parse failure");

        drop(tx);
        let events = drain_recv(&mut rx);

        assert_eq!(
            events.len(),
            1,
            "exactly one GenerationEvent::Error expected; got: {:?}",
            events
        );
        match &events[0] {
            GenerationEvent::Error(code) => {
                assert_eq!(
                    code, "tool_call_unreachable_fallback_required",
                    "AutoLazyGrammar must emit the SAME error code as Constrained \
                     (the unified loud-error vocabulary)"
                );
            }
            GenerationEvent::Delta { kind: DeltaKind::Content, text } => {
                panic!(
                    "REGRESSION: AutoLazyGrammar parse failure emitted Content fallback \
                     (text={text:?}); the wave-3 W-B2 T2.4 final closure MUST promote \
                     this branch to Error identically to Constrained"
                );
            }
            other => panic!(
                "expected GenerationEvent::Error, got: {:?}",
                other
            ),
        }
    }

    /// T2.4 regression-preserve — Auto (no grammar) path.
    ///
    /// Simulates the same malformed body under
    /// `ToolCallPolicy::Auto` — the policy the handler sets when
    /// `tool_choice=auto` AND no grammar is active (no tools[] declared,
    /// OR an unregistered model family). This branch MUST:
    ///   - return `Ok(())`
    ///   - emit exactly one `GenerationEvent::Delta { kind: Content, text: body_dump }`
    ///   - NOT emit `GenerationEvent::Error`
    ///
    /// Under Auto-no-grammar there is no enforcement on body shape. The
    /// model may legitimately emit partial / malformed tool-call syntax;
    /// preserving the content fallback lets the client see the raw bytes
    /// rather than losing them. Wave 3 W-B2 narrowed this branch (the
    /// registered-family+tools path now uses `AutoLazyGrammar`), but the
    /// remaining Auto-no-grammar slice still keeps the fallback — this
    /// test pins it.
    #[test]
    fn auto_body_parse_failure_preserves_content_fallback() {
        let body = "some malformed body text".to_string();
        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(8);
        let mut tc_index: usize = 0;
        let mut saw_tc: bool = false;

        let result = emit_streaming_tool_call_close(
            None, // parsed = None: simulates a body that failed parse_tool_call_body
            body.clone(),
            ToolCallPolicy::Auto,
            &mut tc_index,
            &mut saw_tc,
            &tx,
        );

        assert!(
            result.is_ok(),
            "Auto + parse failure MUST return Ok(()) \
             (content fallback is the defined Auto behaviour, not an error)"
        );
        assert_eq!(tc_index, 0, "tc_index must not be incremented on parse failure");
        assert!(!saw_tc, "saw_tc must remain false on parse failure");

        drop(tx);
        let events = drain_recv(&mut rx);

        assert_eq!(
            events.len(),
            1,
            "exactly one Content delta expected for Auto fallback; got: {:?}",
            events
        );
        match &events[0] {
            GenerationEvent::Delta { kind, text } => {
                assert!(
                    matches!(kind, DeltaKind::Content),
                    "Auto parse-failure delta MUST be DeltaKind::Content; got: {:?}",
                    kind
                );
                assert_eq!(
                    text, &body,
                    "Auto fallback MUST re-emit the original body_dump verbatim; \
                     got: {text:?}"
                );
            }
            GenerationEvent::Error(code) => {
                panic!(
                    "REGRESSION: Auto parse failure promoted to GenerationEvent::Error \
                     (code={code:?}); Auto MUST preserve content fallback until \
                     Wave 3 Phase B lazy grammar lands"
                );
            }
            other => panic!("expected Content delta, got: {:?}", other),
        }
    }
}

// ---------------------------------------------------------------------------
// Wave 3 W-B3 — T2.3 incremental tool-call argument streaming.
//
// Audit-driver tests for `ToolCallStreamEmitter`: they feed body fragments
// through `advance` and `finalize` directly, then assert the emitted SSE
// shape exactly matches the OpenAI Chat Completions streaming spec:
//
//   - Chunk 1: function.name complete, no arguments.
//   - Chunks 2..N-1: arguments fragments that concatenate to valid JSON.
//   - Final chunk: closing `}` (and any kv tail the streaming scanner
//     deferred). On the streaming-driver side `finish_reason="tool_calls"`
//     fires from the terminating `Done` event after `saw_tool_call` is
//     latched true by `finalize`.
//
// All tests drive the emitter directly with hand-crafted body fragments
// (rather than through the full `ToolCallSplitter` + `route_content`
// pipeline) so the tail-parser + chunk-emit logic is isolated from
// splitter / grammar / sampler concerns.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tool_call_stream_emitter_tests {
    use super::*;
    use crate::serve::api::sse::GenerationEvent;
    use tokio::sync::mpsc;

    fn drain(rx: &mut mpsc::Receiver<GenerationEvent>) -> Vec<GenerationEvent> {
        let mut out = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            out.push(ev);
        }
        out
    }

    /// Collapse a sequence of `ToolCallDelta` events into the (name, args_string)
    /// pair the OpenAI client would reconstruct: name from the first chunk that
    /// carries it, args from the concatenation of every chunk's `arguments`.
    fn rebuild_call(events: &[GenerationEvent], expect_index: usize) -> (Option<String>, String) {
        let mut name: Option<String> = None;
        let mut args = String::new();
        for ev in events {
            if let GenerationEvent::ToolCallDelta {
                index, name: n, arguments, ..
            } = ev
            {
                if *index != expect_index {
                    continue;
                }
                if let Some(nm) = n {
                    name = Some(nm.clone());
                }
                if let Some(a) = arguments {
                    args.push_str(a);
                }
            }
        }
        (name, args)
    }

    /// `streaming_tool_call_emits_name_in_first_chunk` — chunk 1 carries
    /// `function.name` complete, with `arguments=None`. Subsequent chunks
    /// stream `arguments` only (no `name` retransmission).
    #[test]
    fn streaming_tool_call_emits_name_in_first_chunk() {
        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(32);
        let mut emitter = ToolCallStreamEmitter::new(Some("gemma4"), 0);
        // Feed a body prefix that contains the name + opening brace, plus
        // a started kv. The first `advance` call MUST emit chunk 1
        // (id+type+name) and chunk 2 (the args opening `{`).
        let body = "call:get_weather{location:<|\"|>San Fra".to_string();
        emitter.advance(&body, &tx).expect("advance ok");
        drop(tx);
        let events = drain(&mut rx);
        // Chunk 1: name+id+type, no args.
        match &events[0] {
            GenerationEvent::ToolCallDelta {
                index,
                id,
                call_type,
                name,
                arguments,
            } => {
                assert_eq!(*index, 0);
                assert!(id.is_some(), "first chunk MUST carry id");
                assert_eq!(call_type.as_deref(), Some("function"));
                assert_eq!(name.as_deref(), Some("get_weather"));
                assert!(
                    arguments.is_none(),
                    "first chunk MUST NOT carry arguments (name-only per spec)"
                );
            }
            other => panic!("expected first ToolCallDelta with name; got {other:?}"),
        }
        // Chunk 2: args opening `{`, no name.
        match &events[1] {
            GenerationEvent::ToolCallDelta {
                name, arguments, id, ..
            } => {
                assert!(id.is_none(), "subsequent chunks MUST NOT retransmit id");
                assert!(
                    name.is_none(),
                    "subsequent chunks MUST NOT retransmit name"
                );
                assert_eq!(
                    arguments.as_deref(),
                    Some("{"),
                    "second chunk MUST be the args opening `{{`"
                );
            }
            other => panic!("expected ToolCallDelta with `{{` arg; got {other:?}"),
        }
    }

    /// `streaming_tool_call_emits_arguments_incrementally` — feed a 3-fragment
    /// body and assert at least 3 distinct `arguments` deltas fire (one per
    /// closed-kv boundary).
    #[test]
    fn streaming_tool_call_emits_arguments_incrementally() {
        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(32);
        let mut emitter = ToolCallStreamEmitter::new(Some("gemma4"), 0);
        let mut body = String::new();
        // Fragment 1: header + first kv started, no closer.
        body.push_str("call:get_weather{location:<|\"|>");
        emitter.advance(&body, &tx).expect("frag1");
        // Fragment 2: close first kv with `,` and start second kv.
        body.push_str("San Francisco<|\"|>,");
        emitter.advance(&body, &tx).expect("frag2");
        // Fragment 3: second kv complete + closer.
        body.push_str("units:<|\"|>celsius<|\"|>}");
        emitter.advance(&body, &tx).expect("frag3");
        // Close finalizes the last kv + `}`.
        let mut tc_index: usize = 0;
        let mut saw_tc: bool = false;
        emitter
            .finalize(
                body,
                Some(&super::super::registry::GEMMA4),
                ToolCallPolicy::Constrained,
                &mut tc_index,
                &mut saw_tc,
                &tx,
            )
            .expect("finalize");
        drop(tx);
        let events = drain(&mut rx);
        // Count `arguments`-bearing deltas. We expect at least:
        //   chunk: `{`   (opening, from advance frag1)
        //   chunk: `"location":"San Francisco"`  (frag2 closes first kv)
        //   chunk: `,"units":"celsius"`  +  closing `}` (finalize)
        //     OR finalize emits both as a single tail.
        let arg_chunks: Vec<&str> = events
            .iter()
            .filter_map(|ev| {
                if let GenerationEvent::ToolCallDelta {
                    arguments: Some(a), ..
                } = ev
                {
                    Some(a.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            arg_chunks.len() >= 3,
            "expected >=3 arguments deltas (incremental shape); got {arg_chunks:?}"
        );
        assert_eq!(tc_index, 1, "tc_index MUST be incremented on finalize");
        assert!(saw_tc, "saw_tc MUST be latched true on finalize");
    }

    /// `streaming_tool_call_arguments_concatenate_to_valid_json` — collect
    /// every `arguments` delta in stream order, concatenate them, JSON-parse
    /// the result, and assert it equals the canonical `parse_tool_call_body`
    /// args output.
    #[test]
    fn streaming_tool_call_arguments_concatenate_to_valid_json() {
        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(64);
        let mut emitter = ToolCallStreamEmitter::new(Some("gemma4"), 0);
        // Feed body in 4 fragments that bisect the kv structure at
        // non-boundary points.
        let mut body = String::new();
        let chunks = [
            "call:get_weather{location:<|\"|>",
            "San Francis",
            "co<|\"|>,units:<|\"|>celsius<|\"|>",
            "}",
        ];
        for c in &chunks {
            body.push_str(c);
            emitter.advance(&body, &tx).expect("advance");
        }
        let mut tc_index: usize = 0;
        let mut saw_tc: bool = false;
        emitter
            .finalize(
                body.clone(),
                Some(&super::super::registry::GEMMA4),
                ToolCallPolicy::Constrained,
                &mut tc_index,
                &mut saw_tc,
                &tx,
            )
            .expect("finalize");
        drop(tx);
        let events = drain(&mut rx);
        let (name, args) = rebuild_call(&events, 0);
        assert_eq!(name.as_deref(), Some("get_weather"));
        let parsed: serde_json::Value =
            serde_json::from_str(&args).expect("args MUST be valid JSON");
        // Canonical args from the existing parser:
        let canonical = super::super::registry::parse_tool_call_body(
            &super::super::registry::GEMMA4,
            &body,
        )
        .expect("canonical parse");
        let canonical_json: serde_json::Value =
            serde_json::from_str(&canonical.arguments_json).expect("canonical json");
        assert_eq!(
            parsed, canonical_json,
            "concatenated streaming args MUST equal canonical parse"
        );
    }

    /// `streaming_tool_call_emits_finish_reason_tool_calls_terminal` — verify
    /// `finalize` latches `saw_tc=true` so the Done event downstream picks
    /// `finish_reason="tool_calls"`. The terminating `Done` is a downstream
    /// concern (driven by the decode loop and `replay_cached_streaming_response`
    /// branch); here we pin the contract that finalize-on-success MUST set
    /// `saw_tc` so the Done-emit logic at engine.rs:2513 + replay.rs:2538 can
    /// override the default `"stop"` to `"tool_calls"`.
    #[test]
    fn streaming_tool_call_emits_finish_reason_tool_calls_terminal() {
        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(16);
        let mut emitter = ToolCallStreamEmitter::new(Some("gemma4"), 0);
        let body = "call:f{x:1}".to_string();
        emitter.advance(&body, &tx).expect("advance");
        let mut tc_index: usize = 0;
        let mut saw_tc: bool = false;
        emitter
            .finalize(
                body,
                Some(&super::super::registry::GEMMA4),
                ToolCallPolicy::Constrained,
                &mut tc_index,
                &mut saw_tc,
                &tx,
            )
            .expect("finalize");
        drop(tx);
        assert!(
            saw_tc,
            "finalize on success MUST latch saw_tc=true so the Done event \
             picks finish_reason=\"tool_calls\""
        );
        assert_eq!(tc_index, 1, "tc_index MUST advance to 1");
        let events = drain(&mut rx);
        let last = events
            .iter()
            .rev()
            .find_map(|ev| {
                if let GenerationEvent::ToolCallDelta {
                    arguments: Some(a), ..
                } = ev
                {
                    Some(a.as_str())
                } else {
                    None
                }
            })
            .expect("at least one arguments delta");
        assert!(
            last.ends_with('}'),
            "the final arguments delta MUST close the JSON object with `}}`; \
             got tail={last:?}"
        );
    }

    /// `streaming_multiple_tool_calls_with_distinct_indices` — drive two
    /// emitters in sequence (mirrors `parallel_tool_calls=true` where
    /// the model emits two consecutive `<|tool_call>...<tool_call|>` spans),
    /// and assert each call's deltas carry distinct `index` values.
    #[test]
    fn streaming_multiple_tool_calls_with_distinct_indices() {
        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(64);
        let mut tc_index: usize = 0;
        let mut saw_tc: bool = false;

        // Call 0.
        let mut em0 = ToolCallStreamEmitter::new(Some("gemma4"), tc_index);
        let body0 = "call:f0{a:1}".to_string();
        em0.advance(&body0, &tx).expect("advance0");
        em0.finalize(
            body0,
            Some(&super::super::registry::GEMMA4),
            ToolCallPolicy::Constrained,
            &mut tc_index,
            &mut saw_tc,
            &tx,
        )
        .expect("finalize0");
        assert_eq!(tc_index, 1, "tc_index advances after call 0");

        // Call 1.
        let mut em1 = ToolCallStreamEmitter::new(Some("gemma4"), tc_index);
        let body1 = "call:f1{b:2}".to_string();
        em1.advance(&body1, &tx).expect("advance1");
        em1.finalize(
            body1,
            Some(&super::super::registry::GEMMA4),
            ToolCallPolicy::Constrained,
            &mut tc_index,
            &mut saw_tc,
            &tx,
        )
        .expect("finalize1");
        assert_eq!(tc_index, 2, "tc_index advances after call 1");

        drop(tx);
        let events = drain(&mut rx);
        let mut indices = std::collections::BTreeSet::new();
        for ev in &events {
            if let GenerationEvent::ToolCallDelta { index, .. } = ev {
                indices.insert(*index);
            }
        }
        assert!(
            indices.contains(&0) && indices.contains(&1),
            "both index=0 and index=1 MUST appear in the delta stream; got {indices:?}"
        );
    }

    /// Wave 3.5 MED — `streaming_single_fragment_emits_incremental_shape`.
    ///
    /// Honest replacement for the misnamed
    /// `streaming_single_fragment_falls_back_to_close_buffered_shape`
    /// test (Wave 3 W-B3).  The previous name promised a "legacy
    /// fallback" to the pre-W-B3 two-chunk close-buffered shape, but
    /// the actual `advance` + `finalize` flow emits MORE than two
    /// chunks even for a single-fragment body:
    ///
    ///   * `advance(body)` sees `call:f{` complete in the FIRST call
    ///     (engine.rs:2196-2239) and immediately emits chunk 1
    ///     (id+name, no arguments) and chunk 2 (`{` opening).
    ///   * `finalize` then emits the residual tail
    ///     (`"x":1` + closing `}`) as additional chunks.
    ///
    /// `finalize` only delegates to the legacy
    /// `emit_streaming_tool_call_close` when `name_emitted == false`
    /// (engine.rs:2398-2406) — i.e. when `advance` couldn't extract
    /// the name from any prefix (unknown family OR the single
    /// fragment didn't contain enough to find the name).  For a
    /// well-formed Gemma 4 single-fragment body like `call:f{x:1}`,
    /// `advance` extracts `f` immediately, sets `name_emitted=true`,
    /// and the legacy fallback is NEVER taken.
    ///
    /// Wave 3 audit divergence "W-B3 single-fragment fallback"
    /// severity MED at
    /// `/tmp/cfa-cfa-20260427-adr005-wave3/codex-review-last.txt`:
    ///
    ///   "advance emits name and the arguments opening as soon as it
    ///    sees call:f{ at engine.rs:2196-2239; finalize delegates to
    ///    legacy only if name_emitted is false at engine.rs:2398-2406.
    ///    The test named streaming_single_fragment_falls_back_to_
    ///    close_buffered_shape only checks concatenated JSON, not
    ///    event count or legacy shape."
    ///
    /// Resolution per audit recommendation (ii) + worker prompt
    /// directive: the incremental shape IS the canonical OpenAI
    /// spec; the "single-fragment legacy fallback" was an unnecessary
    /// backwards-compat hack that was never actually wired up for
    /// well-formed bodies.  Update test to assert the true shape:
    /// chunk 1 has id+name, chunk 2 has `{` opening, finalize emits
    /// the tail (multiple kv chunks possible if the kv-scanner ran;
    /// or one tail chunk if it didn't).  Concatenated arguments MUST
    /// be valid JSON.  No legacy two-chunk shape is preserved or
    /// expected.
    ///
    /// The TRUE legacy fallback (delegating to
    /// `emit_streaming_tool_call_close`) is exercised by
    /// `streaming_unknown_family_falls_back_to_legacy` (unknown
    /// family → `advance` is a no-op → `finalize` delegates).
    #[test]
    fn streaming_single_fragment_emits_incremental_shape() {
        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(16);
        let mut emitter = ToolCallStreamEmitter::new(Some("gemma4"), 0);
        // A single fragment containing the FULL body.  The emitter's
        // first `advance` extracts the name `f` and emits:
        //   chunk 1: id + type + name (no arguments)
        //   chunk 2: arguments=`{`
        // Then finalize emits the residual tail.
        let body = "call:f{x:1}".to_string();
        emitter.advance(&body, &tx).expect("advance");
        let mut tc_index: usize = 0;
        let mut saw_tc: bool = false;
        emitter
            .finalize(
                body,
                Some(&super::super::registry::GEMMA4),
                ToolCallPolicy::Constrained,
                &mut tc_index,
                &mut saw_tc,
                &tx,
            )
            .expect("finalize");
        drop(tx);
        let events = drain(&mut rx);

        // Honest event-shape assertion (audit-driven).  All emitted
        // events MUST be ToolCallDelta with the canonical incremental
        // shape — NOT the legacy two-chunk close-buffered shape.
        assert!(
            events.iter().all(|e| matches!(e, GenerationEvent::ToolCallDelta { .. })),
            "all events MUST be ToolCallDelta (no Content fallback for \
             well-formed Gemma 4 body); got {events:?}"
        );

        // Chunk 1 MUST carry id+type+name (no arguments).  This is the
        // canonical OpenAI streaming first-chunk shape.
        let chunk1 = events.first().expect("at least one event");
        match chunk1 {
            GenerationEvent::ToolCallDelta {
                index, id, call_type, name, arguments,
            } => {
                assert_eq!(*index, 0, "chunk 1 index MUST be 0");
                assert!(id.is_some(), "chunk 1 MUST carry id (canonical OpenAI shape)");
                assert_eq!(call_type.as_deref(), Some("function"));
                assert_eq!(name.as_deref(), Some("f"), "chunk 1 MUST carry function name");
                assert!(arguments.is_none(), "chunk 1 MUST NOT carry arguments");
            }
            other => panic!("chunk 1 must be ToolCallDelta with id+name; got {other:?}"),
        }

        // Chunk 2 MUST be the `{` opening (advance step 2).  No id, no
        // name retransmission.
        let chunk2 = events.get(1).expect("at least two events");
        match chunk2 {
            GenerationEvent::ToolCallDelta {
                index, id, call_type, name, arguments,
            } => {
                assert_eq!(*index, 0);
                assert!(id.is_none(), "chunk 2 MUST NOT retransmit id");
                assert!(call_type.is_none(), "chunk 2 MUST NOT retransmit type");
                assert!(name.is_none(), "chunk 2 MUST NOT retransmit name");
                assert_eq!(
                    arguments.as_deref(),
                    Some("{"),
                    "chunk 2 MUST be the args opening `{{`"
                );
            }
            other => panic!("chunk 2 must be ToolCallDelta with `{{`; got {other:?}"),
        }

        // Event count MUST be at least 2 (chunks 1 and 2 from advance).
        // The pre-W-B3 legacy two-chunk close-buffered shape would have
        // been: chunk 1 (id+name+full args), chunk 2 (close).  The
        // Wave 3 W-B3 incremental shape is strictly different and
        // typically emits more chunks (one per closed kv + a tail).
        assert!(
            events.len() >= 2,
            "incremental shape emits at least 2 chunks (id+name then `{{`); \
             got {} events: {events:?}",
            events.len()
        );

        // Concatenated args across all chunks MUST be valid JSON
        // matching the input body.  This is the canonical OpenAI
        // accumulator-on-the-client contract.
        let (name, args) = rebuild_call(&events, 0);
        assert_eq!(name.as_deref(), Some("f"));
        let v: serde_json::Value =
            serde_json::from_str(&args).expect("args concatenate to valid JSON");
        assert_eq!(v, serde_json::json!({"x": 1}));

        // tc_index MUST advance and saw_tc latch — these are the
        // contracts the live decode loop relies on.
        assert_eq!(tc_index, 1, "tc_index MUST advance to 1 after finalize");
        assert!(saw_tc, "saw_tc MUST latch true after finalize");
    }

    /// Qwen 3.5/3.6 streaming — `<function=NAME>...<parameter=KEY>VAL</parameter>...</function>`
    /// emits one delta per closed `<parameter>` block.
    #[test]
    fn streaming_qwen35_emits_per_parameter_block() {
        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(32);
        let mut emitter = ToolCallStreamEmitter::new(Some("qwen35"), 0);
        let mut body = String::new();
        body.push_str("<function=lookup>");
        emitter.advance(&body, &tx).expect("frag1");
        body.push_str("\n<parameter=q>\n\"hello\"\n</parameter>");
        emitter.advance(&body, &tx).expect("frag2");
        body.push_str("\n<parameter=k>\n5\n</parameter>\n</function>");
        emitter.advance(&body, &tx).expect("frag3");
        let mut tc_index: usize = 0;
        let mut saw_tc: bool = false;
        emitter
            .finalize(
                body.clone(),
                Some(&super::super::registry::QWEN35),
                ToolCallPolicy::Constrained,
                &mut tc_index,
                &mut saw_tc,
                &tx,
            )
            .expect("finalize");
        drop(tx);
        let events = drain(&mut rx);
        let (name, args) = rebuild_call(&events, 0);
        assert_eq!(name.as_deref(), Some("lookup"));
        let v: serde_json::Value =
            serde_json::from_str(&args).expect("args concatenate to valid JSON");
        let canonical = super::super::registry::parse_tool_call_body(
            &super::super::registry::QWEN35,
            &body,
        )
        .expect("canonical");
        let cv: serde_json::Value = serde_json::from_str(&canonical.arguments_json).unwrap();
        assert_eq!(
            v, cv,
            "Qwen 3.5/3.6 streaming args MUST equal canonical parse"
        );
    }

    /// Unknown family — `advance` is a no-op (no `name_emitted`), and
    /// `finalize` delegates to the legacy close-buffered path. Verify the
    /// emitter never emits anything before finalize when the family lacks a
    /// streaming converter, AND that the legacy `Auto` content fallback
    /// fires when the body fails to parse.
    #[test]
    fn streaming_unknown_family_falls_back_to_legacy() {
        let (tx, mut rx) = mpsc::channel::<GenerationEvent>(8);
        let mut emitter = ToolCallStreamEmitter::new(None, 0);
        emitter
            .advance("anything goes here", &tx)
            .expect("advance no-op");
        // No emissions yet — unknown family declined the streaming path.
        let mid_events: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        assert!(
            mid_events.is_empty(),
            "unknown family MUST NOT emit deltas during advance; got {mid_events:?}"
        );
        // Finalize under Auto policy with no registration — body is treated
        // as malformed (no parser), legacy emit fires the content fallback.
        let mut tc_index: usize = 0;
        let mut saw_tc: bool = false;
        let result = emitter.finalize(
            "anything goes here".to_string(),
            None,
            ToolCallPolicy::Auto,
            &mut tc_index,
            &mut saw_tc,
            &tx,
        );
        assert!(result.is_ok(), "Auto + unparseable MUST be Ok (content fallback)");
        drop(tx);
        let events = drain(&mut rx);
        assert_eq!(events.len(), 1, "exactly one Content delta expected");
        match &events[0] {
            GenerationEvent::Delta {
                kind: super::super::sse::DeltaKind::Content,
                text,
            } => {
                assert_eq!(text, "anything goes here");
            }
            other => panic!("expected Content delta, got {other:?}"),
        }
    }
}
