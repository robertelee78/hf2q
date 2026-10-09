//! Typed engine-boundary request-error classification (ADR-062 D2).
//!
//! Engine workers reject some requests for reasons the client can fix or
//! must know are permanent: a prompt that can never fit this server
//! configuration, a family with no embedding path, a capability this build
//! has not implemented. Those rejections used to travel as
//! substring-sentinel `anyhow` messages the HTTP layer matched with
//! `contains`; ADR-062 D2 replaces that with a typed root cause. The engine
//! attaches one of these variants as the root of the `anyhow::Error` it
//! delivers to the handler, and `handlers::common_engine_error_response`
//! maps it to the typed `ApiError` catalog in `schema.rs`.
//!
//! Wire mapping (kind → HTTP status + stable `code`):
//! - [`EngineRequestError::PromptTooLong`] → 400 `prompt_exceeds_server_limit`
//!   (D0 contract; raised only under an explicitly chosen `fifo-serial`
//!   scheduler, and the message names the fix)
//! - [`EngineRequestError::ContextOverflow`] → 400 `context_length_exceeded`
//!   (the code stock OpenCode maps to its `context_overflow`/compaction
//!   path; never a 500/501 from an engine-level cap)
//! - [`EngineRequestError::EmbeddingsUnsupported`] → 400 `embeddings_unsupported`
//! - [`EngineRequestError::UnsupportedCapability`] → 501
//!   `capability_unsupported` (reserved for genuinely unimplemented
//!   capabilities — never length/capacity limits)
//! - [`EngineRequestError::InvalidRequest`] → 400 `invalid_request`
//!
//! `queue_full`/`slot_budget_exceeded` (429), `kv_budget_unsatisfiable`
//! (400, typed `EngineAdmitError` at pre-dispatch), and
//! `engine_failed`/`engine_unhealthy` (503) keep their existing channels and
//! are intentionally outside this enum.
//!
//! `Display` keeps the legacy prefixes ("serial_prompt_limit:",
//! "invalid_request:") so an error that loses its typed root (for example a
//! path that re-formats it into a string event) still classifies through the
//! handler's interim substring fallback with the same wire result.

use std::fmt;

use super::engine::{serial_prompt_limit_message, SERIAL_PROMPT_LIMIT_SENTINEL};

/// A client-actionable request rejection raised inside the engine boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineRequestError {
    /// The prompt exceeds the single-transaction prefill cap of an
    /// explicitly chosen `fifo-serial` scheduler. The request, not the
    /// server, is what cannot be served under the operator's chosen
    /// scheduler. Maps to 400 `prompt_exceeds_server_limit`.
    PromptTooLong { prompt_tokens: usize, limit: usize },
    /// The prompt (or prompt + `max_tokens`) exceeds the model/slot context
    /// or a server-side single-transaction cap, so the request can never be
    /// served as specified. Maps to 400 `context_length_exceeded`. Includes
    /// the operator-greppable engine detail (surface, slot, lengths).
    ContextOverflow {
        context_limit: usize,
        needed: usize,
        detail: String,
    },
    /// The loaded family has no embedding path at all. Maps to 400
    /// `embeddings_unsupported` (ADR-062 D2: DeepSeek-V4 until supported).
    EmbeddingsUnsupported { family: &'static str },
    /// A genuinely unimplemented capability (no path under any scheduler
    /// this build offers). Maps to 501 `capability_unsupported`. The message
    /// keeps the legacy `capability_unsupported:` prefix for operator-grep
    /// continuity.
    UnsupportedCapability { message: String },
    /// Engine-boundary request validation failure that is not a
    /// length/capacity rejection (empty prompt, `max_tokens == 0`, a count
    /// that does not fit u32). Maps to 400 `invalid_request`.
    InvalidRequest { message: String },
}

impl fmt::Display for EngineRequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PromptTooLong {
                prompt_tokens,
                limit,
            } => write!(
                f,
                "{SERIAL_PROMPT_LIMIT_SENTINEL}: {}",
                serial_prompt_limit_message(*prompt_tokens, *limit)
            ),
            Self::ContextOverflow { detail, .. } => write!(f, "{detail}"),
            Self::EmbeddingsUnsupported { family } => write!(
                f,
                "embeddings_unsupported: the loaded {family} runtime has no \
                 embedding path; /v1/embeddings cannot be served with this model"
            ),
            Self::UnsupportedCapability { message } => write!(f, "{message}"),
            Self::InvalidRequest { message } => write!(f, "invalid_request: {message}"),
        }
    }
}

impl std::error::Error for EngineRequestError {}

impl EngineRequestError {
    /// Wrap as the root cause of an `anyhow::Error` so the handler layer
    /// classifies by type (`downcast_ref`) instead of substring matching.
    pub fn into_anyhow(self) -> anyhow::Error {
        anyhow::Error::new(self)
    }

    /// Build a typed context-overflow rejection. `detail` is the
    /// operator-greppable engine message (surface, slot, lengths, fix).
    pub fn context_overflow(context_limit: usize, needed: usize, detail: String) -> Self {
        Self::ContextOverflow {
            context_limit,
            needed,
            detail,
        }
    }

    /// Build a typed invalid-request rejection with the legacy
    /// engine-boundary detail message.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }
}
