//! OpenAI-compatible request/response types for the hf2q API server.
//!
//! Restored from git `fe54bc2~1:src/serve/schema.rs` (the commit preceding the
//! MLX divorce at `fe54bc2`) after engine-agnostic review confirmed the file
//! contains only wire-format types with no inference-engine dependencies. The
//! restore was extended in-place to cover the OpenAI parameter surface agreed
//! in ADR-005 Phase 2 party-mode session `adr_005_phase_2` (2026-04-23):
//! Tiers 1+2+3+4, `response_format`, `stream_options`, `logprobs` +
//! `top_logprobs`, `logit_bias`, `parallel_tool_calls`, reasoning-content
//! split, and the `hf2q_overflow_policy` per-request extension.
//!
//! All types here match the OpenAI API specification so that OpenAI SDKs,
//! Open WebUI, Continue, Cursor, and other clients can speak to hf2q without
//! modification. Fields outside the OpenAI surface that hf2q needs (timings,
//! overflow policy) use the `x_hf2q_*` / `hf2q_*` naming prefix so they
//! round-trip cleanly through strict clients.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};

// ---------------------------------------------------------------------------
// Error types (Decision #24 — OpenAI-compliant `{error: {...}}` envelope)
// ---------------------------------------------------------------------------

/// Top-level error wrapper matching the OpenAI `{"error": {...}}` format.
#[derive(Debug, Clone, Serialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
    /// HTTP status code (not serialized in the JSON body).
    #[serde(skip)]
    pub status: StatusCode,
    /// Optional `Retry-After` header value (seconds). Populated on 429 / 503.
    #[serde(skip)]
    pub retry_after_seconds: Option<u64>,
}

/// The inner error object within the OpenAI error envelope.
///
/// Matches OpenAI's documented schema: `{message, type, param, code}`.
#[derive(Debug, Clone, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ApiError {
    fn bare(
        status: StatusCode,
        message: impl Into<String>,
        error_type: &str,
        code: Option<&str>,
        param: Option<String>,
    ) -> Self {
        Self {
            status,
            retry_after_seconds: None,
            error: ApiErrorBody {
                message: message.into(),
                error_type: error_type.into(),
                param,
                code: code.map(String::from),
            },
        }
    }

    /// Generic invalid request error (HTTP 400).
    pub fn invalid_request(message: impl Into<String>, param: Option<String>) -> Self {
        Self::bare(
            StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
            None,
            param,
        )
    }

    /// Model not found error (HTTP 404).
    pub fn model_not_found(model_name: &str) -> Self {
        Self::bare(
            StatusCode::NOT_FOUND,
            format!("The model '{}' does not exist", model_name),
            "invalid_request_error",
            Some("model_not_found"),
            Some("model".into()),
        )
    }

    /// Model not loaded — used when a `/v1/models` entry is cached on disk but
    /// not the currently-loaded model. Phase 4 hot-swap replaces this with an
    /// auto-swap without a contract change (Decision #26).
    pub fn model_not_loaded(model_name: &str) -> Self {
        Self::bare(
            StatusCode::BAD_REQUEST,
            format!("The model '{}' is cached but not currently loaded. Start the server with `--model <path>` for this model.", model_name),
            "invalid_request_error",
            Some("model_not_loaded"),
            Some("model".into()),
        )
    }

    /// Context length exceeded error (HTTP 400). Used only when the overflow
    /// policy is `reject`; `truncate_left` and `summarize` handle it silently.
    pub fn context_length_exceeded(max_tokens: usize, actual_tokens: usize) -> Self {
        Self::bare(
            StatusCode::BAD_REQUEST,
            format!(
                "This model's maximum context length is {} tokens. However, your messages resulted in {} tokens.",
                max_tokens, actual_tokens
            ),
            "invalid_request_error",
            Some("context_length_exceeded"),
            Some("messages".into()),
        )
    }

    /// Queue full (HTTP 429) — serialized FIFO queue at hard cap.
    ///
    /// **ADR-005 Phase 2 Decision #2** — serialized FIFO queue under
    /// [`crate::serve::api::engine::EngineMode::SerialFifo`] (= the
    /// [`crate::serve::scheduler::SchedulerPolicy::FifoSerial`] scheduler).
    /// **ADR-005 Phase 2 Decision #19** — under FifoSerial (default at
    /// engine spawn unless overridden by `--scheduler` or setup config),
    /// `queue_full` fires when the bounded mpsc channel
    /// (`Engine::spawn(queue_capacity)`) is at hard cap.
    ///
    /// **ADR-040 Phase C C4** (SHIPPED 2026-05-23, cf. ADR-040 §6.1.9)
    /// added explicit `SchedulerPolicy` selection. It is now configured by
    /// `--scheduler` or `[serve].scheduler`. There are TWO
    /// distinct enums at play (and the docstring + test below pin
    /// both — cfa-iter-A5b MAJOR #1 fixed a pre-iter-A5b docstring
    /// bug that referenced a nonexistent variant on the wrong enum;
    /// the test below enforces the correct enum + variant pairing):
    /// - [`crate::serve::scheduler::SchedulerPolicy`] = `{ FifoSerial,
    ///   InflightBatched }` — the SCHEDULER POLICY enum, picks the
    ///   admission FSM.
    /// - [`crate::serve::api::engine::EngineMode`] = `{ SerialFifo,
    ///   SlotAware { max_slots } }` — the ENGINE MODE enum, picks
    ///   the worker_run runtime + per-model SlotAware seam.
    ///
    /// The per-policy semantics for this 429 are:
    /// - Under [`crate::serve::scheduler::SchedulerPolicy::FifoSerial`]
    ///   (default), `queue_full` fires at `queue_capacity` overflow per
    ///   Decision #19 — the legacy single-slot serial path.
    /// - Under [`crate::serve::scheduler::SchedulerPolicy::InflightBatched`]
    ///   (gated behind [`crate::serve::api::engine::EngineMode::SlotAware { max_slots }`]
    ///   at Phase C2c+ future), `queue_full` will fire when
    ///   `total_admissible` (= `queue_capacity` + `max_slots`) is
    ///   exhausted; admission carries a typed
    ///   [`crate::serve::scheduler::AdmitError::QueueFull`] with the
    ///   `queue_capacity` + `total_admissible` field pair (iter-1.5 F6).
    ///   The HTTP-layer mapping (this method) is unchanged at the wire
    ///   level — same status + same body shape + same Retry-After — so
    ///   ADR-040 §1.4 "client-invisibility" is preserved.
    ///
    /// `Retry-After` is populated with a conservative 1-second
    /// suggestion (Decision #19; preserved verbatim under both
    /// policies).
    pub fn queue_full() -> Self {
        let mut e = Self::bare(
            StatusCode::TOO_MANY_REQUESTS,
            "Server is at capacity. Too many pending requests.",
            "server_error",
            Some("queue_full"),
            None,
        );
        e.retry_after_seconds = Some(1);
        e
    }

    /// **ADR-040 full-context slots** — shared physical KV budget exceeded
    /// (HTTP 429
    /// + `Retry-After: 1`).
    ///
    /// Fires when the scheduler's
    /// [`crate::serve::scheduler::AdmitError::SlotBudgetExceeded`]
    /// rejects an admit because the request's
    /// `AdmitRequest::kv_bytes_needed`, together with retained idle-slot
    /// high-water, exceeds the shared KV byte budget. The logical context
    /// advertised by each slot is never divided by `max_slots`. Distinct from
    /// [`Self::queue_full`] (transient — capacity will free as
    /// in-flight requests complete) because this is operator-actionable
    /// on physical residency: another retained slot may need to finish or be
    /// recycled, or the operator may raise the shared budget.
    ///
    /// The wire-level shape mirrors `queue_full` (429 + Retry-After: 1)
    /// per ADR-040 §3.5 ("per-slot OOM returns 429 to the admitting
    /// handler — Decision #19 contract preserved") so SDK clients
    /// treat both the same.  The `code` field is `"slot_budget_exceeded"`
    /// (distinct from `"queue_full"`) so observability + alerting can
    /// differentiate the two 429 emitters.
    ///
    /// The message embeds `needed_bytes` + `budget_bytes` from the
    /// upstream `AdmitError::SlotBudgetExceeded` so the operator-facing
    /// 429 names what was attempted vs what was permitted — same
    /// pattern as `MultiSeqError::SlotOom`'s
    /// `needed_bytes` / `budget_bytes` pair
    /// (`src/serve/multi_seq_kv.rs:265`).
    pub fn slot_budget_exceeded(needed_bytes: u64, budget_bytes: u64) -> Self {
        let mut e = Self::bare(
            StatusCode::TOO_MANY_REQUESTS,
            format!(
                "Shared physical KV cache budget exceeded \
                 (needed_bytes={}, budget_bytes={}). Each agent slot still \
                 has the model's full logical context; wait for or recycle \
                 another slot, reduce max_tokens or shorten the prompt, or \
                 raise `kv_cache_budget_bytes` (ADR-040).",
                needed_bytes, budget_bytes
            ),
            "server_error",
            Some("slot_budget_exceeded"),
            None,
        );
        e.retry_after_seconds = Some(1);
        e
    }

    /// **Guarantees tune-up item 4 (2026-08-20)** — the request's
    /// prompt + max_tokens KV demand exceeds a hard ceiling it can
    /// NEVER satisfy (per-slot budget, or the entire shared budget in
    /// isolation). HTTP **400**, code `"kv_budget_unsatisfiable"`,
    /// **no Retry-After**.
    ///
    /// Split out of [`Self::slot_budget_exceeded`], which used to cover
    /// this permanent case too: "a 429 only ever means the box is busy
    /// right now" is the published contract, and an agent honoring
    /// `Retry-After: 1` on a request that can never fit would loop
    /// forever. `slot_budget_exceeded` (429 + Retry-After) now fires
    /// only for transient aggregate pressure that another request's
    /// completion or slot recycling can relieve.
    ///
    /// `error_type` is `invalid_request_error` (400 class — the
    /// request itself is unservable as specified), matching how
    /// OpenAI-family SDKs treat context-length-style terminal errors.
    pub fn kv_budget_unsatisfiable(needed_bytes: u64, budget_bytes: u64) -> Self {
        Self::bare(
            StatusCode::BAD_REQUEST,
            format!(
                "Request can never fit the physical KV cache budget \
                 (needed_bytes={}, budget_bytes={}): prompt + max_tokens \
                 exceeds what a slot can ever hold. Non-retryable — reduce \
                 max_tokens or shorten the prompt, or raise \
                 `kv_cache_budget_bytes` (ADR-040).",
                needed_bytes, budget_bytes
            ),
            "invalid_request_error",
            Some("kv_budget_unsatisfiable"),
            None,
        )
    }

    /// Server is still warming up (HTTP 503). Emitted before `/readyz` flips to
    /// 200 (Decision #15, #16). Includes `Retry-After: 1`.
    pub fn not_ready() -> Self {
        let mut e = Self::bare(
            StatusCode::SERVICE_UNAVAILABLE,
            "Model is still warming up; please retry shortly.",
            "server_error",
            Some("not_ready"),
            None,
        );
        e.retry_after_seconds = Some(1);
        e
    }

    /// The selected inference worker stopped after a fatal device or worker
    /// failure (HTTP 503). Unlike a normal generation error, retrying this
    /// engine generation in place is unsafe; an operator/supervisor must
    /// recreate the model process first.
    pub fn engine_unhealthy(detail: impl Into<String>) -> Self {
        let mut e = Self::bare(
            StatusCode::SERVICE_UNAVAILABLE,
            detail.into(),
            "server_error",
            Some("engine_unhealthy"),
            None,
        );
        e.retry_after_seconds = Some(1);
        e
    }

    /// Generation error (HTTP 500) — Metal failure, decoder panic caught, etc.
    pub fn generation_error(detail: impl Into<String>) -> Self {
        Self::bare(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Generation failed: {}", detail.into()),
            "server_error",
            Some("generation_error"),
            None,
        )
    }

    /// Generic internal server error (HTTP 500).
    pub fn internal_error() -> Self {
        Self::bare(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal server error",
            "server_error",
            Some("internal_error"),
            None,
        )
    }

    /// Unauthorized (HTTP 401) — missing or invalid bearer token when auth is
    /// configured (Decision #8).
    pub fn unauthorized() -> Self {
        Self::bare(
            StatusCode::UNAUTHORIZED,
            "Missing or invalid authorization header.",
            "authentication_error",
            Some("invalid_api_key"),
            None,
        )
    }

    /// No mmproj configured (HTTP 400) — the request contains `image_url`
    /// content parts but the server was started without `--mmproj`.
    /// Lands in the 400 class (not 501) because the request is malformed
    /// against THIS server configuration: the client needs to either omit
    /// images or use a server instance that has a mmproj loaded.
    pub fn no_mmproj_loaded() -> Self {
        Self::bare(
            StatusCode::BAD_REQUEST,
            "Request includes image_url content parts but this server \
             was started without a multimodal projector. Start with \
             `--mmproj <path>` or send a text-only request.",
            "invalid_request_error",
            Some("no_mmproj_loaded"),
            Some("messages".into()),
        )
    }

    /// Grammar-rejection (HTTP 400) — a malformed JSON schema or GBNF grammar
    /// was supplied in `response_format` or `tools` (Decision #6).
    pub fn grammar_error(detail: impl Into<String>) -> Self {
        Self::grammar_error_for("response_format", detail)
    }

    /// Grammar compilation error attributed to the exact request surface.
    pub fn grammar_error_for(param: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::bare(
            StatusCode::BAD_REQUEST,
            format!("Grammar compilation failed: {}", detail.into()),
            "invalid_request_error",
            Some("grammar_error"),
            Some(param.into()),
        )
    }

    /// Not found error (HTTP 404) — used for unmatched routes.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::bare(
            StatusCode::NOT_FOUND,
            message,
            "invalid_request_error",
            None,
            None,
        )
    }

    /// Not implemented (HTTP 501) — the request is structurally valid
    /// but the SERVER cannot fulfil it because the underlying inference
    /// path is not yet implemented.  ADR-005 Phase 4 reopen iter-215
    /// Wedge-2: Qwen3.5/3.6 chat completions land here today
    /// (model loaded, /readyz / /v1/models / /metrics work, but the
    /// SERVE-side forward pass is Wedge-3 deferred follow-up).  The
    /// caller's request is well-formed; the SERVER's capability
    /// surface is the bottleneck — 501 is the correct HTTP class per
    /// RFC 7231 §6.6.2.
    ///
    /// **ADR-040 Phase C C3 mapping** (iter-2.5 M1 + iter-A3a closure):
    /// [`crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported`]
    /// is the canonical multi-seq KV cache "not yet implemented in
    /// this per-model impl" sentinel (e.g. `fork_seq` cross-slot copy
    /// under Phase A2c/A3c deferral). It is the upstream source of
    /// HTTP 501 emitted via this method — distinct from
    /// [`crate::serve::multi_seq_kv::MultiSeqError::SlotOom`] (429 +
    /// Retry-After) and
    /// [`crate::serve::multi_seq_kv::MultiSeqError::SlotOutOfRange`]
    /// (500 internal-defect). Helper [`Self::capability_unsupported`]
    /// is the typed seam for that upstream mapping; per-handler error
    /// converters call it directly so the operator-facing message
    /// names the unsupported capability.
    pub fn not_implemented(message: impl Into<String>) -> Self {
        Self::bare(
            StatusCode::NOT_IMPLEMENTED,
            message,
            "server_error",
            Some("not_implemented"),
            None,
        )
    }

    /// **ADR-040 Phase C C3** (iter-2.5 M1 + iter-A3a closure):
    /// HTTP 501 for the multi-seq KV cache
    /// [`crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported`]
    /// variant. Thin wrapper over [`Self::not_implemented`] that
    /// embeds the unsupported-capability label so operator-facing
    /// messages name exactly which trait method is the bottleneck
    /// (e.g. `"fork_seq cross-slot copy"`).
    ///
    /// Distinct from [`Self::queue_full`] (429 — capacity exhausted,
    /// transient) and [`Self::generation_error`] (500 — runtime
    /// fault). 501 is the correct HTTP class per RFC 7231 §6.6.2:
    /// the caller's request is well-formed; the SERVER's capability
    /// surface (the per-model `MultiSeqKvCache` impl in this case)
    /// is the bottleneck.
    ///
    /// The `code` field is `"capability_unsupported"` (distinct from
    /// `"not_implemented"`) so observability + alerting can
    /// differentiate "trait-method-not-yet-impled" from other 501
    /// emitters (e.g. iter-215 Wedge-2 Qwen3.5/3.6 wedge); the wire
    /// `status` + `error_type` are identical so OpenAI SDK clients
    /// treat both the same.
    pub fn capability_unsupported(capability: &str) -> Self {
        Self::bare(
            StatusCode::NOT_IMPLEMENTED,
            format!(
                "Capability not yet implemented: {capability} \
                 (ADR-040 §6 Phase C C3 — MultiSeqKvCache::* unimplemented per-model)"
            ),
            "server_error",
            Some("capability_unsupported"),
            None,
        )
    }

    /// Same wire shape as [`Self::capability_unsupported`] (501 + code
    /// `"capability_unsupported"`) but with a caller-authored message,
    /// for capability refusals that are not `MultiSeqKvCache` trait gaps
    /// and where the ADR-040 suffix baked into the sibling constructor
    /// would mislead the operator. First consumer: the guarantees
    /// tune-up item-2 tool-calling gate (2026-08-20) — tools[] declared
    /// under tool_choice=auto on a family with no registered tool-call
    /// emitter refuses at request time instead of serving calls that
    /// can be neither enforced nor parsed.
    pub fn capability_unsupported_message(message: impl Into<String>) -> Self {
        Self::bare(
            StatusCode::NOT_IMPLEMENTED,
            message.into(),
            "server_error",
            Some("capability_unsupported"),
            None,
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        use axum::http::{header, HeaderValue};

        let status = self.status;
        let retry_after = self.retry_after_seconds;
        let body = serde_json::to_string(&self).unwrap_or_else(|_| {
            r#"{"error":{"message":"Internal serialization error","type":"server_error","param":null,"code":null}}"#.into()
        });

        let mut response =
            (status, [(header::CONTENT_TYPE, "application/json")], body).into_response();

        if let Some(secs) = retry_after {
            if let Ok(val) = HeaderValue::from_str(&secs.to_string()) {
                response.headers_mut().insert(header::RETRY_AFTER, val);
            }
        }

        response
    }
}

// ---------------------------------------------------------------------------
// Health / Readyz
// ---------------------------------------------------------------------------

/// Response for `GET /health` — JSON liveness with model info (Decision #12).
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    /// "ok" when the process is alive; "error" if a core component has failed.
    pub status: String,
    /// Currently-loaded model id (path basename or user-supplied alias).
    pub model: Option<String>,
    /// Backend name (`mlx-native` under ADR-008).
    pub backend: &'static str,
    /// Model context length in tokens.
    pub context_length: Option<usize>,
    /// Process uptime in seconds.
    pub uptime_seconds: u64,
}

/// Response for `GET /readyz` — k8s-style readiness (Decision #12, #16).
#[derive(Debug, Clone, Serialize)]
pub struct ReadyzResponse {
    pub ready: bool,
    pub detail: &'static str,
}

// ---------------------------------------------------------------------------
// Models (Decision #26)
// ---------------------------------------------------------------------------

/// A single model object in the OpenAI format.
///
/// Extended with hf2q-specific fields: `quant_type`, `context_length`,
/// `backend`, `loaded`. These survive round-tripping through OpenAI SDKs
/// because SDKs preserve unknown fields in the deserialized object.
///
/// ADR-018 C5: extended again with the unified `LoadInfo` snapshot fields
/// (`arch`, `max_context_length`, `provenance`, `moe_*`, `sliding_window`,
/// `kv_spill_active`, `quant_bpw`). Each new field is `Option<_>` and
/// serde-skip-if-none, so externally-produced cache-scanned entries that
/// have no live engine still serialize to a strict subset of the pre-C5
/// shape — downstream OpenAI-API-compatible clients keep working.
#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: &'static str,
    /// Effective context length in tokens for this loaded engine
    /// (non-standard, widely supported).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<usize>,
    /// GGUF quant type (`Q4_K_M`, `Q6_K`, `Q8_0`, `F16`, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant_type: Option<String>,
    /// Inference backend. Always `mlx-native` today.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<&'static str>,
    /// Whether this model is the currently-loaded one. Phase 4 hot-swap will
    /// allow multiple `loaded: true` entries; Phase 2 is exactly one.
    pub loaded: bool,

    // ── ADR-018 C5 — `LoadInfo`-sourced fields (live-engine path only) ──
    //
    // Each field below is populated by the live-engine path
    // (`handlers.rs::list_models` reading `engine.info()`) and left `None`
    // by the cache-scan / embedding / mmproj paths that have no LoadInfo
    // snapshot. `serde(skip_serializing_if = "Option::is_none")` keeps the
    // wire format byte-identical to the pre-C5 shape for those callers.
    /// Raw GGUF `general.architecture` string, e.g. `"gemma4"`,
    /// `"qwen35"`, `"qwen35moe"`. Mirrors `LoadInfo::arch_str`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    /// Maximum context length declared by the GGUF
    /// (`{arch}.context_length`). Distinct from `context_length`, which is
    /// the effective operator-capped value for a live engine. This field is
    /// `Some` only on engine-backed entries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context_length: Option<u64>,
    /// Provenance label — `"hf2q"` for hf2q-emitted GGUFs, `"external"`
    /// otherwise. Derived from the `Provenance` enum on `LoadInfo`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<&'static str>,
    /// Total expert count for MoE models; `None` for dense and for
    /// non-engine-backed entries. Mirrors `LoadInfo.moe.n_experts`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moe_experts: Option<u32>,
    /// Routed experts per token; `None` for dense and for non-engine-
    /// backed entries. Mirrors `LoadInfo.moe.n_experts_per_tok`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moe_experts_per_tok: Option<u32>,
    /// Sliding-window size in tokens, when applicable. Gemma4 sets this;
    /// Qwen35 leaves it `None`. Mirrors `LoadInfo::sliding_window`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sliding_window: Option<u32>,
    /// `true` iff the engine has a KV-spill hook bound for this load.
    /// Mirrors `LoadInfo::kv_spill_active`. `None` for non-engine-backed
    /// entries (cache-scanner has no engine to ask).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_spill_active: Option<bool>,
    /// Parameter-weighted bits-per-weight, averaged across non-fp tensors.
    /// Mirrors `LoadInfo::quant_bpw`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant_bpw: Option<f32>,
    /// Input modalities accepted by this live inference model. Unloaded
    /// cache entries omit the field because hf2q has not validated their
    /// runtime attachments. A loaded model with a bound projector advertises
    /// `["text", "image"]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_modalities: Option<Vec<&'static str>>,
    /// Output modalities produced by this inference model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_modalities: Option<Vec<&'static str>>,
    /// Projector attached to this chat model after exact runtime binding.
    /// Projectors are implementation components, not independently selectable
    /// language models, so they are linked here instead of listed separately.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision_projector: Option<String>,
}

/// Response for `GET /v1/models`.
#[derive(Debug, Clone, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

// ---------------------------------------------------------------------------
// Chat Completions — request
// ---------------------------------------------------------------------------

/// A single message in the chat conversation.
///
/// `content` supports both the simple string format and the OpenAI Vision API
/// array format. `reasoning_content` (Decision #21) is the OpenAI-o1-style
/// split for thinking-model reasoning traces; it is separate from `content`
/// both on input (history echo-back) and output (model response).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    /// Message content, either a plain string or an array of content parts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    /// Reasoning-content (OpenAI-o1-style split; Decision #21). On request
    /// echo-back, clients send it as a sibling field to `content`. On
    /// response it carries the model's pre-answer reasoning trace,
    /// delimited by per-model boundary markers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Tool calls made by an assistant message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID for a tool-role message (references the prior tool call).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional name (for `system` / `user` messages in OpenAI).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Message content: either a plain string or an array of content parts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain string content.
    Text(String),
    /// Array of content parts (for multimodal messages).
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Concatenate all text parts into a single string.
    pub fn text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Extract image URLs from multimodal content parts.
    pub fn image_urls(&self) -> Vec<&str> {
        match self {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ImageUrl { image_url } => Some(image_url.url.as_str()),
                    _ => None,
                })
                .collect(),
        }
    }

    /// True if the message contains at least one image content part.
    pub fn has_images(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. })),
        }
    }

    /// `Some(text)` if non-empty, else `None`.
    pub fn as_text_opt(&self) -> Option<String> {
        let text = self.text();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }
}

/// A single content part within a multimodal message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content.
    #[serde(rename = "text")]
    Text { text: String },
    /// Image URL content (base64 data URL, file path, or HTTP URL).
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

/// Image URL within a content part.
///
/// Supported formats:
/// - `data:image/{format};base64,{data}` — inline base64 (Open WebUI default)
/// - `file:///path/to/image.jpg` — local file
/// - `/path/to/image.jpg` — local file (shorthand)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageUrl {
    pub url: String,
    /// Optional detail level (`auto` / `low` / `high`). hf2q accepts for
    /// compatibility but does not currently branch on it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A tool call object (assistant-generated; grammar-constrained so the
/// `arguments` string is guaranteed well-formed JSON by construction).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

/// Function details within a tool call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallFunction {
    pub name: String,
    /// Arguments as a JSON string (per OpenAI; the string contains a JSON
    /// document, not the parsed object).
    pub arguments: String,
}

/// A tool definition in the request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

/// Function definition within a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolFunction {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema describing the function's parameters. hf2q converts this
    /// to GBNF via the ported `json-schema-to-grammar` (Decision #6).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Stop sequence: OpenAI supports either a single string or array of strings.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

impl StopSequence {
    /// Convert to a `Vec<String>` regardless of variant.
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopSequence::Single(s) => vec![s],
            StopSequence::Multiple(v) => v,
        }
    }
}

/// A JSON Schema supplied through vLLM's `structured_outputs.json` field.
///
/// vLLM accepts either the schema object itself or a string containing the
/// serialized schema. Keeping those forms distinct lets the compiler reject a
/// malformed JSON string with a request-local diagnostic instead of silently
/// treating it as an unconstrained string schema.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum StructuredOutputJson {
    Object(serde_json::Map<String, serde_json::Value>),
    String(String),
}

/// A JSON Schema accepted by llama.cpp's top-level `json_schema` surface.
/// Draft 2020-12 permits both object schemas and the boolean `true`/`false`
/// schemas. Other JSON types are not schemas and fail during deserialization.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum JsonSchemaValue {
    Boolean(bool),
    Object(serde_json::Map<String, serde_json::Value>),
}

impl JsonSchemaValue {
    pub fn as_value(&self) -> serde_json::Value {
        match self {
            Self::Boolean(value) => serde_json::Value::Bool(*value),
            Self::Object(value) => serde_json::Value::Object(value.clone()),
        }
    }

    pub fn into_value(self) -> serde_json::Value {
        match self {
            Self::Boolean(value) => serde_json::Value::Bool(value),
            Self::Object(value) => serde_json::Value::Object(value),
        }
    }
}

/// vLLM-compatible per-request structured-output controls.
///
/// Exactly one of the six constraint fields must be present. Backend options
/// do not count as constraints and may accompany the selected constraint.
/// Unknown fields are rejected so misspelled constraints cannot degrade to
/// unconstrained generation.
#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct StructuredOutputs {
    #[serde(default)]
    pub choice: Option<Vec<String>>,
    #[serde(default)]
    pub regex: Option<String>,
    #[serde(default)]
    pub json: Option<StructuredOutputJson>,
    #[serde(default)]
    pub json_object: Option<bool>,
    #[serde(default)]
    pub grammar: Option<String>,
    /// XGrammar structural-tag input is intentionally retained as raw JSON.
    /// Its evolving legacy/current shapes require compiler-side semantic
    /// validation, but the enclosing object and mutual exclusivity remain
    /// fail-closed here.
    #[serde(default)]
    pub structural_tag: Option<serde_json::Value>,
    #[serde(default)]
    pub disable_any_whitespace: Option<bool>,
    #[serde(default)]
    pub disable_additional_properties: Option<bool>,
    #[serde(default)]
    pub whitespace_pattern: Option<String>,
}

/// Why a [`StructuredOutputs`] request is not a single usable constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredOutputsValidationError {
    NoConstraint,
    MultipleConstraints,
    JsonObjectMustBeTrue,
}

impl std::fmt::Display for StructuredOutputsValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoConstraint => f.write_str(
                "structured_outputs must specify exactly one of choice, regex, json, \
                 json_object, grammar, or structural_tag",
            ),
            Self::MultipleConstraints => f.write_str(
                "structured_outputs may specify only one of choice, regex, json, \
                 json_object, grammar, or structural_tag",
            ),
            Self::JsonObjectMustBeTrue => {
                f.write_str("structured_outputs.json_object must be true when specified")
            }
        }
    }
}

impl std::error::Error for StructuredOutputsValidationError {}

impl StructuredOutputs {
    /// Enforce vLLM's one-constraint invariant before grammar compilation.
    pub fn validate_exactly_one_constraint(&self) -> Result<(), StructuredOutputsValidationError> {
        let count = [
            self.choice.is_some(),
            self.regex.is_some(),
            self.json.is_some(),
            self.json_object.is_some(),
            self.grammar.is_some(),
            self.structural_tag.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count();

        match count {
            0 => Err(StructuredOutputsValidationError::NoConstraint),
            1 if self.json_object == Some(false) => {
                Err(StructuredOutputsValidationError::JsonObjectMustBeTrue)
            }
            1 => Ok(()),
            _ => Err(StructuredOutputsValidationError::MultipleConstraints),
        }
    }
}

/// Numeric trigger kinds used by llama.cpp's server request schema.
///
/// These discriminants are wire values, not implementation-local ordinals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaGrammarTriggerType {
    Token,
    Word,
    Pattern,
    PatternFull,
}

impl<'de> Deserialize<'de> for LlamaGrammarTriggerType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match u8::deserialize(deserializer)? {
            0 => Ok(Self::Token),
            1 => Ok(Self::Word),
            2 => Ok(Self::Pattern),
            3 => Ok(Self::PatternFull),
            other => Err(D::Error::custom(format!(
                "grammar trigger type must be an integer from 0 through 3, got {other}"
            ))),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlamaGrammarTriggerWire {
    #[serde(rename = "type")]
    trigger_type: LlamaGrammarTriggerType,
    value: String,
    #[serde(default)]
    token: Option<i32>,
}

/// A single llama.cpp lazy-grammar trigger.
///
/// Token triggers (`type: 0`) require the resolved token id. Word and regex
/// triggers must not carry one, preventing ambiguous or ignored input.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(try_from = "LlamaGrammarTriggerWire")]
pub struct LlamaGrammarTrigger {
    pub trigger_type: LlamaGrammarTriggerType,
    pub value: String,
    pub token: Option<i32>,
}

impl TryFrom<LlamaGrammarTriggerWire> for LlamaGrammarTrigger {
    type Error = String;

    fn try_from(wire: LlamaGrammarTriggerWire) -> Result<Self, Self::Error> {
        match (wire.trigger_type, wire.token) {
            (LlamaGrammarTriggerType::Token, None) => {
                return Err("grammar trigger type 0 requires a token field".into())
            }
            (LlamaGrammarTriggerType::Word, Some(_))
            | (LlamaGrammarTriggerType::Pattern, Some(_))
            | (LlamaGrammarTriggerType::PatternFull, Some(_)) => {
                return Err("only grammar trigger type 0 may contain a token field".into())
            }
            _ => {}
        }
        Ok(Self {
            trigger_type: wire.trigger_type,
            value: wire.value,
            token: wire.token,
        })
    }
}

/// `response_format` parameter (Decision #6; Tier 1 surface).
///
/// Four shapes are accepted:
///   `{"type": "text"}`         — unconstrained (default).
///   `{"type": "json_object"}`  — legacy "any valid JSON" constraint.
///   `{"type": "json_schema",
///     "json_schema": {"name": "...", "schema": {...}, "strict": true}}`
///                              — schema-constrained JSON via the ported
///                                 `json-schema-to-grammar` + GBNF sampler.
///   `{"type": "structural_tag", ...}`
///                              — vLLM/XGrammar tagged structured content.
///
/// All three compile down to a grammar the sampler applies token-by-token.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: JsonSchemaSpec },
    /// vLLM structural-tag formats are validated by the structured-output
    /// compiler. Preserve all fields after the `type` discriminator so both
    /// the current `format` shape and the legacy `structures`/`triggers`
    /// shape survive request parsing without premature interpretation.
    #[serde(rename = "structural_tag")]
    StructuralTag {
        #[serde(flatten)]
        spec: serde_json::Map<String, serde_json::Value>,
    },
}

impl<'de> Deserialize<'de> for ResponseFormat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut object = serde_json::Map::<String, serde_json::Value>::deserialize(deserializer)?;
        let format_type = object
            .remove("type")
            .and_then(|value| value.as_str().map(str::to_owned))
            .ok_or_else(|| D::Error::custom("response_format.type must be a string"))?;
        match format_type.as_str() {
            "text" => {
                if !object.is_empty() {
                    return Err(D::Error::custom(
                        "response_format type=text does not accept additional fields",
                    ));
                }
                Ok(Self::Text)
            }
            "json_object" => {
                if !object.is_empty() {
                    return Err(D::Error::custom(
                        "response_format type=json_object does not accept additional fields",
                    ));
                }
                Ok(Self::JsonObject)
            }
            "json_schema" => {
                let json_schema = object.remove("json_schema").ok_or_else(|| {
                    D::Error::custom("response_format type=json_schema requires json_schema")
                })?;
                if !object.is_empty() {
                    return Err(D::Error::custom(
                        "response_format type=json_schema does not accept additional fields",
                    ));
                }
                let json_schema = serde_json::from_value(json_schema).map_err(D::Error::custom)?;
                Ok(Self::JsonSchema { json_schema })
            }
            "structural_tag" => Ok(Self::StructuralTag { spec: object }),
            other => Err(D::Error::custom(format!(
                "unsupported response_format.type {other:?}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct JsonSchemaSpec {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub schema: serde_json::Value,
    /// If `true`, the schema must match exactly (OpenAI `strict: true`).
    /// hf2q treats `null`/`false`/absent as the same "not strict" mode.
    #[serde(default)]
    pub strict: Option<bool>,
}

/// `stream_options` parameter (Tier 2 surface).
///
/// Currently only `include_usage` is specified by OpenAI.
#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

/// `logit_bias` parameter (Tier 4 surface) — raw OpenAI shape is
/// `{token_id_string: bias_float}`; we parse into a typed map.
pub type LogitBiasMap = std::collections::HashMap<String, f32>;

/// Per-request overflow-policy override (hf2q extension, Decision #23).
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum OverflowPolicy {
    /// Return HTTP 400 `context_length_exceeded` — classic behavior.
    Reject,
    /// Drop oldest non-system messages until the prompt fits.
    TruncateLeft,
    /// Summarize oldest non-system messages; replace in-place with a
    /// synthetic `system` message "[Summary of prior conversation]: ...".
    #[default]
    Summarize,
}

/// Request body for `POST /v1/chat/completions` — the full Phase 2a surface
/// (Tiers 1+2+3+4 per Decision #22).
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    // --- Tier 1: core ---
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// OpenAI's newer replacement for `max_tokens`. When both are set,
    /// `max_completion_tokens` wins.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// vLLM structured-output surface. The handler must call
    /// [`StructuredOutputs::validate_exactly_one_constraint`] before
    /// compiling the selected constraint.
    #[serde(default)]
    pub structured_outputs: Option<StructuredOutputs>,
    /// llama.cpp-compatible top-level GBNF grammar.
    #[serde(default)]
    pub grammar: Option<String>,
    /// llama.cpp-compatible top-level JSON Schema. JSON Schema Draft 2020-12
    /// permits an object or a boolean schema; all other JSON types reject at
    /// the request boundary.
    #[serde(default)]
    pub json_schema: Option<JsonSchemaValue>,
    /// Apply the explicit grammar only after a configured trigger fires.
    #[serde(default)]
    pub grammar_lazy: Option<bool>,
    /// Token strings that must remain atomic for lazy token triggers.
    #[serde(default)]
    pub preserved_tokens: Option<Vec<String>>,
    /// llama.cpp-compatible lazy trigger objects.
    #[serde(default)]
    pub grammar_triggers: Option<Vec<LlamaGrammarTrigger>>,

    // --- Tier 2: important ---
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// DeepSeek-V4 reasoning budget. Accepted at the OpenAI-compatible
    /// top level so clients need not know hf2q's `chat_template_kwargs`
    /// extension. Native values are `low`, `high`, and `max`; the top-level
    /// stock-client sentinel `none` is normalized to the `low` baseline.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Optional per-request cap on tokens emitted while the model remains in
    /// its reasoning span. When reached, reasoning-capable families force
    /// their tokenizer-derived close sequence and continue with the remaining
    /// completion budget so callers receive an answer rather than a truncated
    /// reasoning-only response. Compatible with vLLM's extension name.
    #[serde(default)]
    pub thinking_token_budget: Option<usize>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,

    // --- Tier 3: peer extensions ---
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub min_p: Option<f32>,

    // --- Tier 4: power-user ---
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    #[serde(default)]
    pub logit_bias: Option<LogitBiasMap>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,

    // --- hf2q extensions ---
    /// Per-request overflow policy override (Decision #23).
    #[serde(default)]
    pub hf2q_overflow_policy: Option<OverflowPolicy>,

    /// Per-request reasoning-mode override (ADR-005 Phase 2a iter-133
    /// Iter D, W67). When `Some(true)`, the chat-template render passes
    /// `enable_thinking=true` so reasoning-capable models actually emit a
    /// thinking trace (e.g. Gemma 4 emits `<|channel>thought\n…<channel|>`,
    /// Qwen 3.5/3.6 emits `<think>…</think>`). `Some(false)` explicitly
    /// disables reasoning. When omitted, hf2q derives the native default from
    /// the loaded chat template so stock OpenAI-compatible clients do not
    /// need a private extension to use a reasoning-capable checkpoint
    /// correctly. Clients with a reasoning toggle may still send an explicit
    /// value and override that automatic choice.
    #[serde(default)]
    pub hf2q_enable_thinking: Option<bool>,

    /// Extra variables merged into the chat-template Jinja context
    /// (ADR-005 iter-229 Decision 4; peer-compatible name/shape).
    /// Merged AFTER the renderer's own values, so a kwarg wins every
    /// collision that survives validation — renderer-owned keys
    /// (`messages`, `tools`, `add_generation_prompt`, `bos_token`,
    /// `eos_token`, `raise_exception`) are rejected 400 naming the key;
    /// `enable_thinking` is deliberately overridable. Values reach Jinja
    /// verbatim (no type coercion). Canonical use:
    /// `{"preserve_thinking": false}` explicitly opts out of the Qwen API
    /// path's append-stable default. Keeping it enabled makes prior assistant
    /// render bytes stable when a later user turn is appended, which allows
    /// the saved KV prefix to remain reusable after tool-bearing turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
}

// ---------------------------------------------------------------------------
// Chat Completions — response (non-streaming)
// ---------------------------------------------------------------------------

/// Extended timing information returned alongside chat completions (hf2q).
///
/// Serialized into `x_hf2q_timing`. Omitted when not populated.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimingInfo {
    pub prefill_time_secs: f64,
    pub decode_time_secs: f64,
    pub total_time_secs: f64,
    /// Time from request start to the first sampled token (milliseconds).
    pub time_to_first_token_ms: f64,
    pub prefill_tokens_per_sec: f64,
    pub decode_tokens_per_sec: f64,
    /// Number of GPU command-buffer commits during the request. Useful for
    /// cross-run perf regression detection.
    pub gpu_sync_count: u64,
    /// Number of GPU dispatches (kernel launches) during the request.
    pub gpu_dispatch_count: u64,
}

/// Timing attached to the final streaming chunk. Streaming producers are
/// incrementally instrumented, so every counter is independently optional.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct StreamingTimingInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_time_secs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_time_secs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_time_secs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_first_token_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_tokens_per_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_tokens_per_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_sync_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_dispatch_count: Option<u64>,
}

impl StreamingTimingInfo {
    pub fn is_empty(&self) -> bool {
        self.prefill_time_secs.is_none()
            && self.decode_time_secs.is_none()
            && self.total_time_secs.is_none()
            && self.time_to_first_token_ms.is_none()
            && self.prefill_tokens_per_sec.is_none()
            && self.decode_tokens_per_sec.is_none()
            && self.gpu_sync_count.is_none()
            && self.gpu_dispatch_count.is_none()
    }
}

/// Full chat completion response (non-streaming).
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    /// Optional OpenAI system fingerprint (sampler+engine identity); hf2q
    /// sets `hf2q-<short-git-sha>-<mlx-native>` or omits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: UsageStats,
    /// Extended timing information (hf2q-specific). Omitted when unavailable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_hf2q_timing: Option<TimingInfo>,
}

/// A single choice in a non-streaming response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
    /// Per-token logprobs; populated only when the request set `logprobs: true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// Details about prompt token processing (OpenAI-compatible).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Breakdown of prompt-token processing (OpenAI-compatible).
#[derive(Debug, Clone, Serialize)]
pub struct PromptTokensDetails {
    /// Number of prompt tokens served from the prompt cache (Decision #24).
    pub cached_tokens: usize,
}

/// Breakdown of completion-token processing (OpenAI-compatible).
#[derive(Debug, Clone, Serialize)]
pub struct CompletionTokensDetails {
    /// Number of reasoning tokens (the portion of the completion between the
    /// model's reasoning-open and reasoning-close markers; Decision #21).
    pub reasoning_tokens: usize,
}

// ---------------------------------------------------------------------------
// Logprobs — Tier 4
// ---------------------------------------------------------------------------

/// Top-level `logprobs` object on a chat-completion choice.
#[derive(Debug, Clone, Serialize)]
pub struct ChoiceLogprobs {
    pub content: Vec<TokenLogprob>,
}

/// Per-token logprob entry.
#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f32,
    /// Raw token bytes (UTF-8 byte values). Useful for tokens that straddle
    /// UTF-8 boundaries and thus don't cleanly fit `token`.
    pub bytes: Option<Vec<u8>>,
    /// Top-K alternatives at this position. Empty vec if `top_logprobs` was 0.
    pub top_logprobs: Vec<TopLogprobEntry>,
}

/// A single top-K alternative logprob entry.
#[derive(Debug, Clone, Serialize)]
pub struct TopLogprobEntry {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

// ---------------------------------------------------------------------------
// Chat Completions — streaming (SSE chunks)
// ---------------------------------------------------------------------------

/// A streaming chunk for SSE responses.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChunkChoice>,
    /// Usage stats. Included only in the final chunk when the request set
    /// `stream_options.include_usage: true` (Tier 2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageStats>,
    /// hf2q diagnostic timing. Included only on the final chunk and only
    /// when the generation producer populated at least one timing counter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_hf2q_timing: Option<StreamingTimingInfo>,
}

/// A single choice in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
    /// Per-token logprobs for this chunk's delta. Populated only when the
    /// request set `logprobs: true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// The delta content in a streaming chunk (Decision #21 — reasoning split).
#[derive(Debug, Clone, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// OpenAI-o1-style reasoning delta, streamed separately from `content`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Tool call deltas for streaming tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// A single tool call delta in a streaming chunk.
///
/// The first delta for a given index includes `id`, `type`, and
/// `function.name`. Subsequent deltas for the same index append to
/// `function.arguments`.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallDelta {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunctionDelta>,
}

/// Partial function details within a streaming tool call delta.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// Tool choice (parsed enum)
// ---------------------------------------------------------------------------

/// Parsed `tool_choice` value from the request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoiceValue {
    /// "auto" or absent — the model decides.
    Auto,
    /// "none" — skip tool calling entirely.
    None,
    /// "required" — the model must emit a tool call.
    Required,
    /// Force a specific function by name.
    Function(String),
}

impl ToolChoiceValue {
    /// Parse the OpenAI `tool_choice` union without a permissive fallback.
    /// Unknown strings and malformed named-function objects are request
    /// errors; they must never turn into `auto` generation.
    pub fn try_parse(value: Option<&serde_json::Value>) -> Result<Self, String> {
        match value {
            None => Ok(ToolChoiceValue::Auto),
            Some(serde_json::Value::String(s)) => match s.as_str() {
                "auto" => Ok(ToolChoiceValue::Auto),
                "none" => Ok(ToolChoiceValue::None),
                "required" => Ok(ToolChoiceValue::Required),
                _ => Err(format!(
                    "tool_choice must be 'auto', 'none', 'required', or a named function object; got {s:?}"
                )),
            },
            Some(serde_json::Value::Object(obj)) => {
                if obj.len() != 2
                    || obj.get("type").and_then(serde_json::Value::as_str) != Some("function")
                {
                    return Err(
                        "named tool_choice must contain exactly type='function' and function"
                            .into(),
                    );
                }
                let function = obj
                    .get("function")
                    .and_then(serde_json::Value::as_object)
                    .ok_or_else(|| "tool_choice.function must be an object".to_string())?;
                if function.len() != 1 {
                    return Err(
                        "tool_choice.function must contain exactly one name field".into(),
                    );
                }
                let name = function
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| "tool_choice.function.name must be a string".to_string())?;
                if name.is_empty() {
                    return Err("tool_choice.function.name must not be empty".into());
                }
                Ok(ToolChoiceValue::Function(name.to_string()))
            }
            Some(_) => Err(
                "tool_choice must be a string or a named function object".to_string(),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Embeddings (Decision #4)
// ---------------------------------------------------------------------------

/// Input for the embeddings endpoint — accepts a single string or an array.
///
/// OpenAI also accepts `Vec<Vec<u32>>` (pre-tokenized inputs); hf2q rejects
/// that path with `invalid_request_error` until a concrete client asks for it.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

impl EmbeddingInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s],
            EmbeddingInput::Multiple(v) => v,
        }
    }
}

/// Request body for `POST /v1/embeddings`.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    /// Encoding format. hf2q supports `"float"` (default); `"base64"` returns
    /// 400 `invalid_request_error` until a concrete client needs it.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// OpenAI dimensions cap. hf2q treats as advisory — the returned vector
    /// is whatever the model produces; truncation is not performed silently.
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// Optional user identifier (accepted, ignored — matches OpenAI behavior).
    #[serde(default)]
    pub user: Option<String>,
}

/// A single embedding object in the response. The `embedding` field
/// can be either a list of floats (`encoding_format="float"`) or a
/// base64-encoded string of little-endian F32 bytes (the OpenAI Python
/// SDK's *default* encoding — chosen to reduce JSON payload size).
/// Serialized as an untagged union so JSON output looks like OpenAI's:
///   { "object": "embedding", "embedding": [0.1, 0.2, ...], "index": 0 }
///   { "object": "embedding", "embedding": "base64string==", "index": 0 }
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum EmbeddingPayload {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub embedding: EmbeddingPayload,
    pub index: usize,
}

/// Response for `POST /v1/embeddings`.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// Token usage stats specific to the embeddings endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_error_serialization() {
        let err = ApiError::invalid_request("Something went wrong", None);
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["message"], "Something went wrong");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert!(json["error"]["param"].is_null());
        assert!(json["error"]["code"].is_null());
    }

    #[test]
    fn test_api_error_with_param() {
        let err = ApiError::invalid_request("Bad field", Some("messages".into()));
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["param"], "messages");
    }

    #[test]
    fn test_model_not_found_error() {
        let err = ApiError::model_not_found("gpt-5");
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["code"], "model_not_found");
        assert!(json["error"]["message"].as_str().unwrap().contains("gpt-5"));
        assert_eq!(err.status, StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_model_not_loaded_error() {
        let err = ApiError::model_not_loaded("qwen3.6-27b");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.error.code.as_deref(), Some("model_not_loaded"));
        assert!(err.error.message.contains("qwen3.6-27b"));
        assert_eq!(err.error.param.as_deref(), Some("model"));
    }

    #[test]
    fn test_context_length_exceeded_error() {
        let err = ApiError::context_length_exceeded(8192, 9000);
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["code"], "context_length_exceeded");
        let msg = json["error"]["message"].as_str().unwrap();
        assert!(msg.contains("8192"));
        assert!(msg.contains("9000"));
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_queue_full_error_is_429_with_retry_after() {
        let err = ApiError::queue_full();
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
            Some("1")
        );
    }

    /// Guarantees tune-up item 4 (2026-08-20): transient budget
    /// pressure stays 429 + Retry-After: 1 (retryable — "a 429 only
    /// ever means the box is busy right now").
    #[test]
    fn test_slot_budget_exceeded_is_429_with_retry_after() {
        let err = ApiError::slot_budget_exceeded(5_000, 4_000);
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["code"], "slot_budget_exceeded");
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
            Some("1")
        );
    }

    /// Guarantees tune-up item 4 (2026-08-20): a request that can NEVER
    /// fit (prompt + max_tokens exceeds a hard KV ceiling) is a
    /// non-retryable 400 with NO Retry-After — an agent honoring
    /// Retry-After on this rejection would loop forever. Mirrors
    /// `test_queue_full_error_is_429_with_retry_after` for the
    /// non-retryable path.
    #[test]
    fn test_kv_budget_unsatisfiable_is_400_without_retry_after() {
        let err = ApiError::kv_budget_unsatisfiable(2_048_000, 1_048_576);
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["code"], "kv_budget_unsatisfiable");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response.headers().get("retry-after"),
            None,
            "never-fits rejection MUST NOT carry Retry-After"
        );
    }

    #[test]
    fn test_not_ready_is_503_with_retry_after() {
        let err = ApiError::not_ready();
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
            Some("1")
        );
    }

    #[test]
    fn test_unauthorized_error() {
        let err = ApiError::unauthorized();
        assert_eq!(err.status, StatusCode::UNAUTHORIZED);
        assert_eq!(err.error.error_type, "authentication_error");
    }

    #[test]
    fn test_grammar_error() {
        let err = ApiError::grammar_error("unclosed brace at pos 42");
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "grammar_error");
        assert_eq!(json["error"]["param"], "response_format");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unclosed brace at pos 42"));
    }

    #[test]
    fn test_generation_error() {
        let err = ApiError::generation_error("Metal command buffer error");
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["code"], "generation_error");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Metal command buffer error"));
        assert_eq!(err.status, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_internal_error() {
        let err = ApiError::internal_error();
        assert_eq!(err.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(err.error.code, Some("internal_error".into()));
    }

    #[test]
    fn test_model_list_response_serialization() {
        // ADR-018 C5: every new `LoadInfo`-sourced field is `None` here so
        // the serialized shape mirrors the pre-C5 cache-scanned entry shape
        // exactly (the optional extension keys are skipped via
        // `#[serde(skip_serializing_if = "Option::is_none")]`).
        let resp = ModelListResponse {
            object: "list",
            data: vec![ModelObject {
                id: "test-model".into(),
                object: "model",
                created: 1234567890,
                owned_by: "hf2q",
                context_length: Some(262144),
                quant_type: Some("Q4_K_M".into()),
                backend: Some("mlx-native"),
                loaded: true,
                arch: None,
                max_context_length: None,
                provenance: None,
                moe_experts: None,
                moe_experts_per_tok: None,
                sliding_window: None,
                kv_spill_active: None,
                quant_bpw: None,
                input_modalities: None,
                output_modalities: None,
                vision_projector: None,
            }],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["id"], "test-model");
        assert_eq!(json["data"][0]["object"], "model");
        assert_eq!(json["data"][0]["created"], 1234567890);
        assert_eq!(json["data"][0]["owned_by"], "hf2q");
        assert_eq!(json["data"][0]["context_length"], 262144);
        assert_eq!(json["data"][0]["quant_type"], "Q4_K_M");
        assert_eq!(json["data"][0]["backend"], "mlx-native");
        assert_eq!(json["data"][0]["loaded"], true);

        // Backward-compat pin: with every C5 field `None`, the serialized
        // wire shape MUST NOT include any of the new keys.
        let entry = &json["data"][0];
        assert!(entry.get("arch").is_none(), "arch must be skipped");
        assert!(
            entry.get("max_context_length").is_none(),
            "max_context_length must be skipped"
        );
        assert!(
            entry.get("provenance").is_none(),
            "provenance must be skipped"
        );
        assert!(
            entry.get("moe_experts").is_none(),
            "moe_experts must be skipped"
        );
        assert!(
            entry.get("moe_experts_per_tok").is_none(),
            "moe_experts_per_tok must be skipped"
        );
        assert!(
            entry.get("sliding_window").is_none(),
            "sliding_window must be skipped"
        );
        assert!(
            entry.get("kv_spill_active").is_none(),
            "kv_spill_active must be skipped"
        );
        assert!(
            entry.get("quant_bpw").is_none(),
            "quant_bpw must be skipped"
        );
        assert!(entry.get("input_modalities").is_none());
        assert!(entry.get("output_modalities").is_none());
        assert!(entry.get("vision_projector").is_none());
    }

    #[test]
    fn test_model_object_with_load_info_fields() {
        // ADR-018 C5: when the live-engine path populates the new fields,
        // they appear in the wire format alongside the legacy fields.
        let obj = ModelObject {
            id: "Qwen3.6-27B-A3B-DWQ46-MoE".into(),
            object: "model",
            created: 1700000000,
            owned_by: "hf2q",
            context_length: Some(262_144),
            quant_type: Some("Q4_K".into()),
            backend: Some("mlx-native"),
            loaded: true,
            arch: Some("qwen35moe".into()),
            max_context_length: Some(262_144),
            provenance: Some("hf2q"),
            moe_experts: Some(128),
            moe_experts_per_tok: Some(8),
            sliding_window: None,
            kv_spill_active: Some(false),
            quant_bpw: Some(4.55),
            input_modalities: Some(vec!["text"]),
            output_modalities: Some(vec!["text"]),
            vision_projector: None,
        };
        let json = serde_json::to_value(&obj).unwrap();
        assert_eq!(json["arch"], "qwen35moe");
        assert_eq!(json["max_context_length"], 262_144);
        assert_eq!(json["provenance"], "hf2q");
        assert_eq!(json["moe_experts"], 128);
        assert_eq!(json["moe_experts_per_tok"], 8);
        assert!(
            json.get("sliding_window").is_none(),
            "sliding_window=None must be skipped"
        );
        assert_eq!(json["kv_spill_active"], false);
        // Float comparison via approximate equality on the JSON number's f64.
        let bpw = json["quant_bpw"].as_f64().expect("quant_bpw f64");
        assert!(
            (bpw - 4.55_f64).abs() < 1e-3,
            "quant_bpw expected ≈4.55, got {bpw}"
        );
        assert_eq!(json["input_modalities"], serde_json::json!(["text"]));
        assert_eq!(json["output_modalities"], serde_json::json!(["text"]));
        assert!(json.get("vision_projector").is_none());
    }

    #[test]
    fn test_health_response_serialization() {
        let resp = HealthResponse {
            status: "ok".into(),
            model: Some("gemma4-26b".into()),
            backend: "mlx-native",
            context_length: Some(262144),
            uptime_seconds: 42,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["status"], "ok");
        assert_eq!(json["model"], "gemma4-26b");
        assert_eq!(json["backend"], "mlx-native");
        assert_eq!(json["uptime_seconds"], 42);
    }

    #[test]
    fn test_readyz_response_serialization() {
        let resp = ReadyzResponse {
            ready: false,
            detail: "warming up",
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["ready"], false);
        assert_eq!(json["detail"], "warming up");
    }

    #[test]
    fn test_chat_completion_response_serialization() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-123".into(),
            object: "chat.completion",
            created: 1700000000,
            model: "test-model".into(),
            system_fingerprint: Some("hf2q-deadbeef-mlx-native".into()),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: Some(MessageContent::Text("Hello!".into())),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                finish_reason: "stop".into(),
                logprobs: None,
            }],
            usage: UsageStats {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            },
            x_hf2q_timing: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["id"], "chatcmpl-123");
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["system_fingerprint"], "hf2q-deadbeef-mlx-native");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        assert_eq!(json["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["prompt_tokens"], 10);
        assert_eq!(json["usage"]["completion_tokens"], 5);
        assert_eq!(json["usage"]["total_tokens"], 15);
    }

    #[test]
    fn test_chat_completion_request_all_tiers_deserialize() {
        let json = r#"{
            "model": "gemma4-26b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "max_tokens": 100,
            "max_completion_tokens": 200,
            "temperature": 0.7,
            "stop": "END",
            "response_format": {"type": "json_object"},
            "top_p": 0.9,
            "seed": 42,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "stream_options": {"include_usage": true},
            "top_k": 40,
            "repetition_penalty": 1.05,
            "min_p": 0.05,
            "logprobs": true,
            "top_logprobs": 5,
            "logit_bias": {"1234": -100.0, "5678": 100.0},
            "parallel_tool_calls": false,
            "hf2q_overflow_policy": "summarize"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "gemma4-26b");
        assert_eq!(req.max_tokens, Some(100));
        assert_eq!(req.max_completion_tokens, Some(200));
        assert_eq!(req.temperature, Some(0.7));
        assert!(matches!(
            req.response_format,
            Some(ResponseFormat::JsonObject)
        ));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.seed, Some(42));
        assert_eq!(req.frequency_penalty, Some(0.1));
        assert_eq!(req.presence_penalty, Some(0.2));
        assert_eq!(
            req.stream_options.as_ref().unwrap().include_usage,
            Some(true)
        );
        assert_eq!(req.top_k, Some(40));
        assert_eq!(req.repetition_penalty, Some(1.05));
        assert_eq!(req.min_p, Some(0.05));
        assert_eq!(req.logprobs, Some(true));
        assert_eq!(req.top_logprobs, Some(5));
        assert_eq!(req.logit_bias.as_ref().unwrap().len(), 2);
        assert_eq!(req.parallel_tool_calls, Some(false));
        assert_eq!(req.hf2q_overflow_policy, Some(OverflowPolicy::Summarize));
    }

    #[test]
    fn test_response_format_json_schema_deserialize() {
        let json = r#"{
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "description": "A typed answer",
                    "schema": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
                    "strict": true
                }
            }
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        match req.response_format {
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                assert_eq!(json_schema.name, "answer");
                assert_eq!(json_schema.description.as_deref(), Some("A typed answer"));
                assert_eq!(json_schema.strict, Some(true));
                assert!(json_schema.schema.is_object());
            }
            other => panic!("expected JsonSchema, got {:?}", other),
        }
    }

    #[test]
    fn test_structured_outputs_accepts_each_vllm_constraint_shape() {
        let cases = [
            serde_json::json!({"choice": ["yes", "no"]}),
            serde_json::json!({"regex": "[a-z]+"}),
            serde_json::json!({"json": {"type": "object"}}),
            serde_json::json!({"json": "{\"type\":\"object\"}"}),
            serde_json::json!({"json_object": true}),
            serde_json::json!({"grammar": "root ::= \"ok\""}),
            serde_json::json!({"structural_tag": {"format": {"type": "object"}}}),
        ];

        for value in cases {
            let parsed: StructuredOutputs = serde_json::from_value(value.clone())
                .unwrap_or_else(|error| panic!("{value} must deserialize: {error}"));
            assert_eq!(parsed.validate_exactly_one_constraint(), Ok(()));
        }

        let object: StructuredOutputs =
            serde_json::from_value(serde_json::json!({"json": {"type": "array"}})).unwrap();
        assert!(matches!(object.json, Some(StructuredOutputJson::Object(_))));
        let string: StructuredOutputs = serde_json::from_value(serde_json::json!({
            "json": "{\"type\":\"array\"}"
        }))
        .unwrap();
        assert!(matches!(string.json, Some(StructuredOutputJson::String(_))));
    }

    #[test]
    fn test_structured_outputs_options_do_not_count_as_constraints() {
        let parsed: StructuredOutputs = serde_json::from_value(serde_json::json!({
            "grammar": "root ::= \"ok\"",
            "disable_any_whitespace": true,
            "disable_additional_properties": true,
            "whitespace_pattern": "[ ]?"
        }))
        .unwrap();
        assert_eq!(parsed.validate_exactly_one_constraint(), Ok(()));
    }

    #[test]
    fn test_structured_outputs_validation_is_fail_closed() {
        let empty: StructuredOutputs = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(
            empty.validate_exactly_one_constraint(),
            Err(StructuredOutputsValidationError::NoConstraint)
        );

        let multiple: StructuredOutputs = serde_json::from_value(serde_json::json!({
            "choice": ["yes", "no"],
            "regex": "yes|no"
        }))
        .unwrap();
        assert_eq!(
            multiple.validate_exactly_one_constraint(),
            Err(StructuredOutputsValidationError::MultipleConstraints)
        );

        let false_json_object: StructuredOutputs =
            serde_json::from_value(serde_json::json!({"json_object": false})).unwrap();
        assert_eq!(
            false_json_object.validate_exactly_one_constraint(),
            Err(StructuredOutputsValidationError::JsonObjectMustBeTrue)
        );

        assert!(
            serde_json::from_value::<StructuredOutputs>(serde_json::json!({
                "guided_regex": "[a-z]+"
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<StructuredOutputs>(serde_json::json!({
                "choice": "yes"
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<StructuredOutputs>(serde_json::json!({
                "json": ["not", "a", "schema"]
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<StructuredOutputs>(serde_json::json!({
                "grammar": {"root": "ok"}
            }))
            .is_err()
        );
    }

    #[test]
    fn test_chat_request_deserializes_vllm_and_llama_grammar_surfaces() {
        let json = r#"{
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "structured_outputs": {"choice": ["allow", "deny"]},
            "grammar": "root ::= \"ok\"",
            "json_schema": {"type": "object"},
            "grammar_lazy": true,
            "preserved_tokens": ["<tool>", "</tool>"],
            "grammar_triggers": [
                {"type": 0, "value": "<tool>", "token": 32000},
                {"type": 1, "value": "<tool>"},
                {"type": 2, "value": "^tool:"},
                {"type": 3, "value": "tool:(.*)"}
            ]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            req.structured_outputs
                .as_ref()
                .unwrap()
                .validate_exactly_one_constraint(),
            Ok(())
        );
        assert_eq!(req.grammar.as_deref(), Some("root ::= \"ok\""));
        assert_eq!(
            req.json_schema.as_ref().unwrap().as_value()["type"],
            "object"
        );
        assert_eq!(req.grammar_lazy, Some(true));
        assert_eq!(
            req.preserved_tokens.as_deref(),
            Some(["<tool>".to_string(), "</tool>".to_string()].as_slice())
        );
        let triggers = req.grammar_triggers.unwrap();
        assert_eq!(triggers.len(), 4);
        assert_eq!(triggers[0].trigger_type, LlamaGrammarTriggerType::Token);
        assert_eq!(triggers[0].token, Some(32000));
        assert_eq!(triggers[1].trigger_type, LlamaGrammarTriggerType::Word);
        assert_eq!(triggers[2].trigger_type, LlamaGrammarTriggerType::Pattern);
        assert_eq!(
            triggers[3].trigger_type,
            LlamaGrammarTriggerType::PatternFull
        );
    }

    #[test]
    fn test_llama_grammar_surface_rejects_type_and_trigger_drift() {
        let request_with = |fields: &str| {
            format!(r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],{fields}}}"#)
        };

        for fields in [
            r#""grammar": {"root": "ok"}"#,
            r#""json_schema": "{\"type\":\"object\"}""#,
            r#""json_schema": [true]"#,
            r#""json_schema": 7"#,
            r#""grammar_lazy": "true""#,
            r#""preserved_tokens": [1]"#,
            r#""grammar_triggers": [{"type": 4, "value": "x"}]"#,
            r#""grammar_triggers": [{"type": 0, "value": "x"}]"#,
            r#""grammar_triggers": [{"type": 1, "value": "x", "token": 7}]"#,
            r#""grammar_triggers": [{"type": 1, "value": "x", "extra": true}]"#,
        ] {
            let json = request_with(fields);
            assert!(
                serde_json::from_str::<ChatCompletionRequest>(&json).is_err(),
                "invalid surface must fail: {json}"
            );
        }
    }

    #[test]
    fn test_llama_top_level_json_schema_accepts_object_and_boolean_schemas() {
        for (wire, expected) in [
            (r#"{"type":"object"}"#, serde_json::json!({"type":"object"})),
            ("true", serde_json::Value::Bool(true)),
            ("false", serde_json::Value::Bool(false)),
        ] {
            let json = format!(
                r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],"json_schema":{wire}}}"#
            );
            let request: ChatCompletionRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(request.json_schema.unwrap().into_value(), expected);
        }

        // llama.cpp treats explicit null like omission; serde's Option wire
        // shape preserves that compatibility.
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"json_schema":null}"#,
        )
        .unwrap();
        assert!(request.json_schema.is_none());
    }

    #[test]
    fn test_tool_definition_wire_shape_is_strict() {
        let valid = serde_json::json!({
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Look up weather",
                "parameters": {"type": "object"}
            }
        });
        serde_json::from_value::<Tool>(valid).unwrap();

        for invalid in [
            serde_json::json!({
                "type": "function",
                "function": {"name": "lookup", "unknown": true}
            }),
            serde_json::json!({
                "type": "function",
                "function": {"name": "lookup"},
                "unknown": true
            }),
        ] {
            assert!(
                serde_json::from_value::<Tool>(invalid.clone()).is_err(),
                "unknown tool fields must reject: {invalid}"
            );
        }
    }

    #[test]
    fn test_response_format_structural_tag_preserves_current_and_legacy_shapes() {
        for value in [
            serde_json::json!({
                "type": "structural_tag",
                "format": {"type": "object", "properties": {}}
            }),
            serde_json::json!({
                "type": "structural_tag",
                "structures": [{"begin": "<json>", "schema": {}, "end": "</json>"}],
                "triggers": ["<json>"]
            }),
        ] {
            let parsed: ResponseFormat = serde_json::from_value(value.clone()).unwrap();
            match &parsed {
                ResponseFormat::StructuralTag { spec } => assert!(!spec.is_empty()),
                other => panic!("expected StructuralTag, got {other:?}"),
            }
            assert_eq!(serde_json::to_value(parsed).unwrap(), value);
        }
    }

    #[test]
    fn test_json_schema_response_wrapper_rejects_unknown_fields() {
        let value = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
                "strcit": true
            }
        });
        assert!(serde_json::from_value::<ResponseFormat>(value).is_err());

        for value in [
            serde_json::json!({"type":"text", "extra":true}),
            serde_json::json!({"type":"json_object", "extra":true}),
            serde_json::json!({
                "type":"json_schema",
                "json_schema":{"name":"answer", "schema":{}},
                "extra":true
            }),
        ] {
            assert!(
                serde_json::from_value::<ResponseFormat>(value.clone()).is_err(),
                "unknown response_format fields must reject: {value}"
            );
        }
    }

    #[test]
    fn test_chat_completion_request_minimal() {
        let json = r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream.is_none());
        assert!(req.temperature.is_none());
        assert!(req.top_p.is_none());
        assert!(req.max_tokens.is_none());
        assert!(req.max_completion_tokens.is_none());
        assert!(req.stop.is_none());
        assert!(req.response_format.is_none());
        assert!(req.structured_outputs.is_none());
        assert!(req.grammar.is_none());
        assert!(req.json_schema.is_none());
        assert!(req.grammar_lazy.is_none());
        assert!(req.preserved_tokens.is_none());
        assert!(req.grammar_triggers.is_none());
        assert!(req.seed.is_none());
        assert!(req.top_k.is_none());
        assert!(req.logprobs.is_none());
        assert!(req.hf2q_overflow_policy.is_none());
    }

    #[test]
    fn test_stop_sequence_single() {
        let json = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stop":"END"}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let stops = req.stop.unwrap().into_vec();
        assert_eq!(stops, vec!["END"]);
    }

    #[test]
    fn test_stop_sequence_multiple() {
        let json = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stop":["A","B"]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let stops = req.stop.unwrap().into_vec();
        assert_eq!(stops, vec!["A", "B"]);
    }

    #[test]
    fn test_final_chunk_with_usage() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-789".into(),
            object: "chat.completion.chunk",
            created: 1700000000,
            model: "test-model".into(),
            system_fingerprint: None,
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: Some("stop".into()),
                logprobs: None,
            }],
            usage: Some(UsageStats {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            x_hf2q_timing: None,
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["total_tokens"], 30);
    }

    #[test]
    fn test_tool_choice_try_parse_auto() {
        assert_eq!(
            ToolChoiceValue::try_parse(None).unwrap(),
            ToolChoiceValue::Auto
        );
        let val = serde_json::json!("auto");
        assert_eq!(
            ToolChoiceValue::try_parse(Some(&val)).unwrap(),
            ToolChoiceValue::Auto
        );
    }

    #[test]
    fn test_tool_choice_try_parse_none() {
        let val = serde_json::json!("none");
        assert_eq!(
            ToolChoiceValue::try_parse(Some(&val)).unwrap(),
            ToolChoiceValue::None
        );
    }

    #[test]
    fn test_tool_choice_try_parse_required() {
        let val = serde_json::json!("required");
        assert_eq!(
            ToolChoiceValue::try_parse(Some(&val)).unwrap(),
            ToolChoiceValue::Required
        );
    }

    #[test]
    fn test_tool_choice_try_parse_forced_function() {
        let val = serde_json::json!({"type": "function", "function": {"name": "get_weather"}});
        match ToolChoiceValue::try_parse(Some(&val)).unwrap() {
            ToolChoiceValue::Function(name) => assert_eq!(name, "get_weather"),
            other => panic!("Expected Function, got {:?}", other),
        }
    }

    #[test]
    fn test_tool_choice_try_parse_rejects_malformed_values() {
        for value in [
            serde_json::json!("sometimes"),
            serde_json::json!(true),
            serde_json::json!({}),
            serde_json::json!({"type":"function"}),
            serde_json::json!({"type":"other","function":{"name":"lookup"}}),
            serde_json::json!({"type":"function","function":{}}),
            serde_json::json!({"type":"function","function":{"name":""}}),
            serde_json::json!({"type":"function","function":{"name":"lookup","extra":true}}),
            serde_json::json!({"type":"function","function":{"name":"lookup"},"extra":true}),
        ] {
            assert!(
                ToolChoiceValue::try_parse(Some(&value)).is_err(),
                "malformed tool_choice must fail closed: {value}"
            );
        }
    }

    #[test]
    fn test_tool_call_delta_serialization() {
        let delta = ToolCallDelta {
            index: 0,
            id: Some("call_abc123".to_string()),
            call_type: Some("function".to_string()),
            function: Some(ToolCallFunctionDelta {
                name: Some("get_weather".to_string()),
                arguments: None,
            }),
        };
        let json = serde_json::to_value(&delta).unwrap();
        assert_eq!(json["index"], 0);
        assert_eq!(json["id"], "call_abc123");
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "get_weather");
        assert!(json["function"].get("arguments").is_none());
    }

    #[test]
    fn test_chat_message_with_tool_call_id() {
        let json = r#"{"role":"tool","content":"sunny","tool_call_id":"call_123"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "tool");
        assert_eq!(
            msg.content.as_ref().map(|c| c.text()).as_deref(),
            Some("sunny")
        );
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_123"));
    }

    #[test]
    fn test_chat_message_reasoning_content_round_trip() {
        let msg = ChatMessage {
            role: "assistant".into(),
            content: Some(MessageContent::Text("final answer".into())),
            reasoning_content: Some("let me think step by step...".into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["reasoning_content"], "let me think step by step...");
        assert_eq!(json["content"], "final answer");
        let round_trip: ChatMessage = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip, msg);
    }

    #[test]
    fn test_chunk_delta_with_reasoning_only() {
        let delta = ChunkDelta {
            role: None,
            content: None,
            reasoning_content: Some("wait...".into()),
            tool_calls: None,
        };
        let json = serde_json::to_value(&delta).unwrap();
        assert!(json.get("content").is_none());
        assert_eq!(json["reasoning_content"], "wait...");
    }

    #[test]
    fn test_embedding_input_single_string() {
        let json = r#""hello world""#;
        let input: EmbeddingInput = serde_json::from_str(json).unwrap();
        assert_eq!(input.into_vec(), vec!["hello world"]);
    }

    #[test]
    fn test_embedding_input_array() {
        let json = r#"["hello", "world"]"#;
        let input: EmbeddingInput = serde_json::from_str(json).unwrap();
        assert_eq!(input.into_vec(), vec!["hello", "world"]);
    }

    #[test]
    fn test_embedding_request_deserialize() {
        let json = r#"{"model": "gemma4", "input": "test input"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "gemma4");
        assert!(matches!(req.input, EmbeddingInput::Single(_)));
        assert!(req.encoding_format.is_none());
        assert!(req.dimensions.is_none());
    }

    #[test]
    fn test_embedding_response_schema() {
        let resp = EmbeddingResponse {
            object: "list",
            data: vec![
                EmbeddingObject {
                    object: "embedding",
                    embedding: EmbeddingPayload::Float(vec![0.1, 0.2]),
                    index: 0,
                },
                EmbeddingObject {
                    object: "embedding",
                    embedding: EmbeddingPayload::Float(vec![0.3, 0.4]),
                    index: 1,
                },
            ],
            model: "test".to_string(),
            usage: EmbeddingUsage {
                prompt_tokens: 10,
                total_tokens: 10,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"].as_array().unwrap().len(), 2);
        assert_eq!(json["data"][0]["object"], "embedding");
        assert_eq!(json["data"][0]["index"], 0);
        assert_eq!(json["data"][1]["index"], 1);
        assert_eq!(json["usage"]["prompt_tokens"], 10);
        assert_eq!(json["usage"]["total_tokens"], 10);
    }

    #[test]
    fn test_chat_request_with_tools_deserialize() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                    }
                }
            ],
            "tool_choice": "auto"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.tools.is_some());
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
        assert!(req.tool_choice.is_some());
    }

    #[test]
    fn test_overflow_policy_deserialize_each_variant() {
        for (raw, expected) in [
            ("\"reject\"", OverflowPolicy::Reject),
            ("\"truncate_left\"", OverflowPolicy::TruncateLeft),
            ("\"summarize\"", OverflowPolicy::Summarize),
        ] {
            let p: OverflowPolicy = serde_json::from_str(raw).unwrap();
            assert_eq!(p, expected);
        }
    }

    #[test]
    fn test_overflow_policy_default_is_summarize() {
        // Decision #23 — `summarize` is the default. If this test breaks on a
        // future refactor, the docs + ADR must change in lockstep.
        assert_eq!(OverflowPolicy::default(), OverflowPolicy::Summarize);
    }

    #[test]
    fn test_message_content_text_string() {
        let json = r#"{"role":"user","content":"Hello"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content.as_ref().unwrap().text(), "Hello");
        assert!(!msg.content.as_ref().unwrap().has_images());
        assert!(msg.content.as_ref().unwrap().image_urls().is_empty());
    }

    #[test]
    fn test_message_content_null() {
        let json = r#"{"role":"assistant","content":null}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert!(msg.content.is_none());
    }

    #[test]
    fn test_message_content_vision_array() {
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
            ]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        let content = msg.content.as_ref().unwrap();
        assert_eq!(content.text(), "What's in this image?");
        assert!(content.has_images());
        let urls = content.image_urls();
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0], "data:image/png;base64,abc123");
    }

    #[test]
    fn test_message_content_multiple_images() {
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,img1"}},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,img2"}}
            ]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        let content = msg.content.as_ref().unwrap();
        assert_eq!(content.text(), "Compare these:");
        let urls = content.image_urls();
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "data:image/png;base64,img1");
        assert_eq!(urls[1], "data:image/jpeg;base64,img2");
    }

    #[test]
    fn test_message_content_image_url_with_detail() {
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "file:///tmp/test.png", "detail": "high"}}
            ]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        let content = msg.content.as_ref().unwrap();
        assert!(content.has_images());
        assert_eq!(content.image_urls()[0], "file:///tmp/test.png");
    }

    #[test]
    fn test_message_content_text_only_array() {
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "First part"},
                {"type": "text", "text": " second part"}
            ]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        let content = msg.content.as_ref().unwrap();
        assert_eq!(content.text(), "First part second part");
        assert!(!content.has_images());
    }

    #[test]
    fn test_message_content_as_text_opt() {
        let content = MessageContent::Text("hello".to_string());
        assert_eq!(content.as_text_opt(), Some("hello".to_string()));

        let content = MessageContent::Text("".to_string());
        assert_eq!(content.as_text_opt(), None);
    }

    #[test]
    fn test_message_content_serialization_round_trip_text() {
        let msg = ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Text("Hello".into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["content"], "Hello");
    }

    #[test]
    fn test_message_content_serialization_round_trip_parts() {
        let msg = ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Look at this:".into(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,abc".into(),
                        detail: None,
                    },
                },
            ])),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert!(json["content"].is_array());
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "Look at this:");
        assert_eq!(json["content"][1]["type"], "image_url");
        assert_eq!(
            json["content"][1]["image_url"]["url"],
            "data:image/png;base64,abc"
        );
    }

    #[test]
    fn test_logprobs_serialization() {
        let lp = ChoiceLogprobs {
            content: vec![TokenLogprob {
                token: "Hello".into(),
                logprob: -0.5,
                bytes: Some(vec![72, 101, 108, 108, 111]),
                top_logprobs: vec![TopLogprobEntry {
                    token: "Hi".into(),
                    logprob: -1.2,
                    bytes: Some(vec![72, 105]),
                }],
            }],
        };
        let json = serde_json::to_value(&lp).unwrap();
        assert_eq!(json["content"][0]["token"], "Hello");
        assert!((json["content"][0]["logprob"].as_f64().unwrap() - -0.5).abs() < 1e-6);
        assert_eq!(json["content"][0]["top_logprobs"][0]["token"], "Hi");
    }

    // ───────────────────────────────────────────────────────────────────
    // ADR-040 §6 Phase C C3 — Decision #2 docstring + CapabilityUnsupported
    // → HTTP 501 mapping tests
    // ───────────────────────────────────────────────────────────────────

    /// **ADR-040 C3** — the `queue_full()` docstring now names
    /// `SchedulerPolicy` (the C4 SHIPPED operator surface from
    /// ADR-040 §6.1.9) alongside Decision #2. Pinning the docstring
    /// content as a test catches future drift — a `pub fn queue_full`
    /// without `SchedulerPolicy` in the surrounding doc-block is a
    /// regression on the C3 documentation goal.
    #[test]
    fn c3_schema_queue_full_docstring_names_scheduler_policy() {
        // We pin the docstring at compile-time via the source file —
        // the std::env! var CARGO_MANIFEST_DIR + the known relative
        // path is the load-bearing identity here. include_str! pulls
        // the schema.rs source into this test binary as a string
        // constant so the test is self-contained (no fs I/O at test
        // time).
        let source = include_str!("schema.rs");

        // Find the queue_full doc block + body. The doc block is the
        // run of `///` lines immediately preceding `pub fn queue_full`.
        let queue_full_pos = source
            .find("pub fn queue_full() -> Self")
            .expect("source must contain `pub fn queue_full() -> Self`");

        // Walk backwards collecting lines until we hit a non-`///` /
        // non-blank line — that's the doc block.
        let preamble = &source[..queue_full_pos];
        let docblock_lines: Vec<&str> = preamble
            .lines()
            .rev()
            .take_while(|line| {
                let trimmed = line.trim_start();
                trimmed.starts_with("///") || trimmed.is_empty()
            })
            .collect();
        let docblock = docblock_lines
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            docblock.contains("SchedulerPolicy"),
            "ADR-040 C3: the queue_full() docstring MUST name \
             `SchedulerPolicy` (per ADR-040 §6.1.9 C4 SHIPPED). \
             Current doc block:\n{docblock}"
        );
        assert!(
            docblock.contains("Decision #2"),
            "ADR-040 C3: the queue_full() docstring MUST cite \
             ADR-005 Decision #2 (the carve-out this scheduler \
             selection sits alongside). Current doc block:\n{docblock}"
        );
        assert!(
            docblock.contains("FifoSerial") && docblock.contains("InflightBatched"),
            "ADR-040 C3 (iter-A5b MAJOR #1 fix): the queue_full() \
             docstring MUST name BOTH real `SchedulerPolicy` variants \
             — `FifoSerial` (default under Decision #19) and \
             `InflightBatched` (the real Phase C2c+ scheduler-policy \
             enum variant). The doc must NOT name a nonexistent \
             `SchedulerPolicy::SlotAware` variant. Current doc \
             block:\n{docblock}"
        );
        // cfa-iter-A5b MAJOR #1 regression pin: the docstring MUST
        // also name the SEPARATE `EngineMode::SlotAware` enum (engine
        // mode + max_slots, distinct from the scheduler policy
        // surface) so operators don't confuse the two enums.
        assert!(
            docblock.contains("EngineMode::SlotAware"),
            "ADR-040 C3 (iter-A5b MAJOR #1 fix): the queue_full() \
             docstring MUST name `EngineMode::SlotAware` as the \
             engine-mode enum variant gating the InflightBatched \
             runtime (distinct from the SchedulerPolicy variant). \
             Current doc block:\n{docblock}"
        );
        // cfa-iter-A5b MAJOR #1 regression pin: assert the docstring
        // does NOT erroneously refer to a `SchedulerPolicy::SlotAware`
        // variant — there is no such variant; that string is the
        // pre-iter-A5b vaporware the codex review surfaced.
        assert!(
            !docblock.contains("SchedulerPolicy::SlotAware"),
            "ADR-040 C3 (iter-A5b MAJOR #1 fix): the queue_full() \
             docstring MUST NOT reference `SchedulerPolicy::SlotAware` \
             — that variant does not exist on the SchedulerPolicy enum \
             (variants are `FifoSerial` + `InflightBatched`). \
             SlotAware lives on the SEPARATE `EngineMode` enum. \
             Current doc block:\n{docblock}"
        );
        assert!(
            docblock.contains("ADR-040"),
            "ADR-040 C3: the queue_full() docstring MUST reference \
             ADR-040 so operators searching for the scheduler-policy \
             surface land on this method. Current doc block:\n{docblock}"
        );
    }

    /// **ADR-040 C3** — the
    /// [`crate::serve::multi_seq_kv::MultiSeqError::CapabilityUnsupported`]
    /// variant maps to HTTP 501 Not Implemented via
    /// [`ApiError::capability_unsupported`]. Pins both the status code
    /// + the `code` field + the response shape (status code on the
    /// rendered Response, not just on the struct).
    /// **ADR-040 §3.5 iter-A5** — the shared physical KV budget exceeded
    /// helper maps to HTTP 429 + `Retry-After: 1`, mirrors the
    /// `queue_full` wire shape, embeds the needed/budget byte pair in
    /// the body, and surfaces a distinct `slot_budget_exceeded` code
    /// for observability.
    #[test]
    fn c3_schema_slot_budget_exceeded_returns_429_with_retry_after() {
        let needed = 5 * 1024 * 1024u64;
        let budget = 4 * 1024 * 1024u64;
        let err = ApiError::slot_budget_exceeded(needed, budget);

        // Struct-level assertions.
        assert_eq!(
            err.status,
            StatusCode::TOO_MANY_REQUESTS,
            "ADR-040 §3.5 A5: SlotBudgetExceeded MUST map to HTTP 429 \
             (per Decision #19, parallel to queue_full)"
        );
        assert_eq!(
            err.error.error_type, "server_error",
            "ADR-040 §3.5 A5: error_type follows queue_full convention \
             (server_error class)"
        );
        assert_eq!(
            err.error.code.as_deref(),
            Some("slot_budget_exceeded"),
            "ADR-040 §3.5 A5: code MUST be `slot_budget_exceeded` \
             (distinct from queue_full so observability + alerting \
             can differentiate the two 429 emitters)"
        );
        assert_eq!(
            err.retry_after_seconds,
            Some(1),
            "ADR-040 §3.5 A5: Retry-After: 1 mirrors queue_full \
             (Decision #19 wire-level contract preserved)"
        );

        // Message MUST name the actionable diagnostic (max_tokens or
        // prompt) + cite ADR-040.
        assert!(
            err.error.message.contains(&needed.to_string()),
            "ADR-040 §3.5 A5: message MUST embed needed_bytes verbatim. \
             Got: {}",
            err.error.message
        );
        assert!(
            err.error.message.contains(&budget.to_string()),
            "ADR-040 §3.5 A5: message MUST embed budget_bytes verbatim. \
             Got: {}",
            err.error.message
        );
        assert!(
            err.error.message.contains("max_tokens"),
            "ADR-040 §3.5 A5: message MUST cite max_tokens as a \
             remediation lever so the operator knows what to change. \
             Got: {}",
            err.error.message
        );
        assert!(
            err.error.message.contains("prompt"),
            "ADR-040 §3.5 A5: message MUST cite the prompt-shortening \
             remediation. Got: {}",
            err.error.message
        );
        assert!(
            err.error.message.contains("ADR-040"),
            "ADR-040 §3.5 A5: message MUST cite ADR-040 §3.5 so \
             operators can find the canonical documentation. Got: {}",
            err.error.message
        );

        // Wire-level Response: this is the load-bearing contract for
        // SDK clients (must be byte-shape-compatible with queue_full).
        let response = err.into_response();
        assert_eq!(
            response.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "ADR-040 §3.5 A5: rendered Response status MUST be 429"
        );
        assert_eq!(
            response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
            Some("1"),
            "ADR-040 §3.5 A5: rendered Response MUST carry \
             Retry-After: 1 (Decision #19)"
        );
    }

    #[test]
    fn c3_schema_capability_unsupported_maps_to_501() {
        let err = ApiError::capability_unsupported(
            "fork_seq cross-slot copy (Qwen35 HybridKvCache; deferred to Phase A2c)",
        );
        // Struct-level assertions
        assert_eq!(
            err.status,
            StatusCode::NOT_IMPLEMENTED,
            "ADR-040 C3: MultiSeqError::CapabilityUnsupported MUST \
             map to HTTP 501 Not Implemented (distinct from \
             SlotOom→429 and SlotOutOfRange→500)"
        );
        assert_eq!(
            err.error.error_type, "server_error",
            "ADR-040 C3: error_type follows the iter-215 Wedge-2 \
             `not_implemented` convention (server_error class)"
        );
        assert_eq!(
            err.error.code.as_deref(),
            Some("capability_unsupported"),
            "ADR-040 C3: the `code` field MUST be \
             `capability_unsupported` so observability + alerting \
             can differentiate from other 501 emitters"
        );
        // Message MUST name the unsupported capability + cite ADR-040.
        assert!(
            err.error.message.contains("fork_seq cross-slot copy"),
            "ADR-040 C3: the rendered message MUST name the \
             unsupported capability so operators know which trait \
             method is the bottleneck. Got: {}",
            err.error.message
        );
        assert!(
            err.error.message.contains("ADR-040"),
            "ADR-040 C3: the rendered message MUST cite ADR-040 \
             §6 Phase C C3 so the operator can find the canonical \
             documentation. Got: {}",
            err.error.message
        );
        // Wire-level Response assertion: this is the load-bearing
        // contract for SDK clients.
        let response = err.into_response();
        assert_eq!(
            response.status(),
            StatusCode::NOT_IMPLEMENTED,
            "ADR-040 C3: the rendered HTTP Response status MUST be \
             501 Not Implemented (RFC 7231 §6.6.2 — caller's request \
             is well-formed; the server's capability surface is the \
             bottleneck)"
        );
        // 501 is NOT a transient error — Retry-After must NOT be set
        // (unlike 429 queue_full which carries Retry-After: 1).
        assert!(
            response.headers().get("retry-after").is_none(),
            "ADR-040 C3: 501 Not Implemented is NOT transient — the \
             unsupported capability requires a future iter to ship; \
             no Retry-After should be emitted (unlike 429 queue_full)"
        );
    }
}
