//! Shared application state (`AppState`) for the hf2q HTTP API server.
//!
//! The axum router threads a single `AppState` through every handler. This
//! type is deliberately thin: it holds immutable per-server config, the
//! startup timestamp (for `/health` uptime), a warmup-ready atomic (for
//! `/readyz`), and a monotonic request counter (for request-id generation
//! and `/metrics`).
//!
//! In this iteration the inference engine itself is NOT part of `AppState` —
//! the engine + model load + warmup pipeline lands in the next iteration
//! alongside `/v1/chat/completions` and `/v1/embeddings`. The `ready_for_gen`
//! flag is currently always `true` because no generation endpoint is routed
//! yet; when the engine is introduced, the flag will start `false`, flip to
//! `true` after warmup completes, and the chat/embedding handlers will gate
//! on it per ADR-005 Decision #16 (`503 + Retry-After` during warmup).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use super::engine::Engine;
use super::schema::OverflowPolicy;
use crate::inference::models::bert::BertConfig;
use crate::inference::vision::mmproj::{ArchProfile, MmprojConfig};
use crate::inference::vision::mmproj_weights::LoadedMmprojWeights;

/// Server-level configuration, captured at startup from CLI flags + defaults.
///
/// All fields are immutable for the lifetime of the server. A restart is
/// required to change any of them (matching ollama / llama.cpp conventions).
///
/// Decision numbers reference ADR-005 Phase 2 refinement (2026-04-23).
#[derive(Debug, Clone)]
pub struct ServerConfig {
    // --- Networking ---
    /// Bind address. Defaults to `127.0.0.1` (Decision #7).
    pub host: String,
    /// TCP port. Defaults to `8080`.
    pub port: u16,

    // --- Auth (Decision #8) ---
    /// Optional Bearer token. When `Some(token)`, every request must carry
    /// `Authorization: Bearer <token>` or receive 401. When `None`, no auth.
    pub auth_token: Option<String>,

    // --- CORS (Decision #9) ---
    /// Allowed origins for CORS. Empty = wide-open `*` (localhost dev
    /// default); populated = restrictive allowlist.
    pub cors_allowed_origins: Vec<String>,

    // --- Queue (Decision #19) ---
    /// Hard cap on the FIFO generation queue. Overflow returns 429 +
    /// `Retry-After`. Applies only to generation endpoints.
    pub queue_capacity: usize,

    // --- Rate limits (Decision #10) ---
    /// Max concurrent in-flight HTTP requests per bind. 0 = unlimited
    /// (bounded only by the queue cap + OS).
    pub max_concurrent_requests: usize,

    // --- Timeouts ---
    /// Per-request timeout (applies to the whole request including queue wait
    /// and generation). 0 = no timeout.
    pub request_timeout_seconds: u64,

    // --- Overflow policy (Decision #23) ---
    /// Default context-overflow policy. Per-request `hf2q_overflow_policy`
    /// overrides this.
    pub default_overflow_policy: OverflowPolicy,

    // --- Model catalog ---
    /// Directory to scan for `/v1/models` listing. Per Decision #26 this is
    /// `~/.cache/hf2q/`; overridable for tests / bring-your-own-cache.
    pub cache_dir: Option<PathBuf>,

    // --- Server identity ---
    /// Optional system fingerprint advertised via `ChatCompletionResponse.
    /// system_fingerprint`. Defaults to `None`; production can set to
    /// `"hf2q-<short-git-sha>-mlx-native"`.
    pub system_fingerprint: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        // Defaults are aligned with Decision #7 (localhost bind) + conservative
        // queue + no auth. Tests construct with defaults + per-test overrides.
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            auth_token: None,
            cors_allowed_origins: Vec::new(),
            queue_capacity: 32,
            max_concurrent_requests: 0,
            request_timeout_seconds: 0,
            default_overflow_policy: OverflowPolicy::Summarize,
            cache_dir: default_cache_dir(),
            system_fingerprint: None,
        }
    }
}

/// Resolve the default HF2Q cache directory (`$HOME/.cache/hf2q`).
///
/// Returns `None` if `$HOME` is unset (test / hermetic CI envs).
pub fn default_cache_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|h| h.join(".cache").join("hf2q"))
}

/// Process-wide metric counters surfaced via `/metrics` in Prometheus text
/// format (Decision #11). Cheap atomics; handler code bumps them inline.
///
/// `sse_cancellations` is wrapped in `Arc<AtomicU64>` because it needs to
/// be shared with the engine worker thread (the worker detects the
/// receiver drop and bumps the counter directly). The other counters are
/// bumped from the handler thread and don't need the extra Arc hop.
#[derive(Debug, Default)]
pub struct ServerMetrics {
    /// Total number of HTTP requests that reached a handler (post-auth).
    pub requests_total: AtomicU64,
    /// Total number of chat-completion generations started.
    pub chat_completions_started: AtomicU64,
    /// Total number of chat-completion generations that completed
    /// successfully.
    pub chat_completions_completed: AtomicU64,
    /// Total number of chat-completion generations that hit
    /// `queue_full` (FIFO at capacity → 429).
    pub chat_completions_queue_full: AtomicU64,
    /// Total number of SSE stream cancellations (client dropped the
    /// connection mid-generation; Decision #18). Shared Arc so the
    /// engine worker thread can bump directly.
    pub sse_cancellations: Arc<AtomicU64>,
    /// Total tokens decoded across all completions (cumulative counter).
    pub decode_tokens_total: AtomicU64,
    /// Total prompt tokens ingested across all completions.
    pub prompt_tokens_total: AtomicU64,
    /// Total requests rejected at handler boundary (auth, malformed, etc.).
    pub requests_rejected_total: AtomicU64,
}

impl ServerMetrics {
    /// Clone the shared `sse_cancellations` Arc so the engine worker thread
    /// can bump it from outside the handler.
    pub fn sse_cancellations_counter_arc(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.sse_cancellations)
    }
}

/// Shared runtime state threaded through axum handlers.
///
/// Cheap to clone (every field is behind `Arc` or is a plain atomic wrapper).
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ServerConfig>,
    pub started_at: Arc<Instant>,
    /// `true` once the server is ready to serve generation. When `engine` is
    /// `None`, generation endpoints return 400 `model_not_loaded`. When
    /// `engine` is `Some` but `ready_for_gen` is `false`, they return 503
    /// `not_ready` + `Retry-After: 1` (Decision #16, warmup-in-progress).
    pub ready_for_gen: Arc<AtomicBool>,
    /// Monotonic counter for request-id generation + metrics.
    pub request_counter: Arc<AtomicU64>,
    /// Inference engine. `None` means no `--model` was supplied at startup
    /// (the HTTP backbone still serves `/health`, `/readyz`, `/v1/models`).
    /// `Some(engine)` means a model is loaded and the worker thread is up;
    /// generation endpoints gate on `ready_for_gen` for warmup status.
    pub engine: Option<Engine>,
    /// BERT embedding model config (from `--embedding-model <path>`).
    /// `None` when no embedding model was supplied. Validated at startup
    /// via `BertConfig::from_gguf` — the forward pass that consumes this
    /// lands in a later iter (ADR-005 Phase 2b).
    pub embedding_config: Option<EmbeddingModel>,
    /// Multimodal projector (mmproj GGUF) loaded at startup from
    /// `--mmproj <path>`. When `Some`, the chat handler accepts
    /// `image_url` content parts and routes them through the vision
    /// preprocessor + ViT forward pass that this mmproj describes.
    /// `None` means the server is text-only.
    pub mmproj: Option<LoadedMmproj>,
    /// Process-wide metric counters surfaced via `/metrics`.
    pub metrics: Arc<ServerMetrics>,
}

/// BERT embedding model, discovered from `--embedding-model <path>` at
/// startup. Holds the config + the GGUF path so later iters can load
/// weights on demand.
/// Loaded BERT embedding model, discovered from `--embedding-model <path>`
/// at startup. Holds the config + vocab + a ready-to-use WordPiece
/// tokenizer so embedding requests tokenize without re-parsing the GGUF
/// metadata. Weights load on-demand in the forward-pass iter.
///
/// Shared via `Arc` so multiple handler calls can tokenize concurrently
/// against the same immutable tokenizer.
#[derive(Clone)]
pub struct EmbeddingModel {
    pub gguf_path: PathBuf,
    pub config: BertConfig,
    pub vocab: Arc<crate::inference::models::bert::BertVocab>,
    /// Ready-to-use WordPiece tokenizer. Wrapped in `Arc` because
    /// `tokenizers::Tokenizer` is not trivially `Clone` across its
    /// generic-type-erased form, but `Arc<Tokenizer>` is cheap to share.
    pub tokenizer: Arc<tokenizers::Tokenizer>,
    /// Model id (file stem) — surfaced via `/v1/models`.
    pub model_id: String,
}

impl std::fmt::Debug for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingModel")
            .field("gguf_path", &self.gguf_path)
            .field("config", &self.config)
            .field("vocab_len", &self.vocab.len())
            .field("model_id", &self.model_id)
            .finish()
    }
}

impl EmbeddingModel {
    /// Convenience: tokenize a single input string using the embedded
    /// WordPiece tokenizer. Returns the token-id vector.
    pub fn encode(&self, input: &str) -> anyhow::Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(input, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }
}

/// Loaded mmproj (multimodal projector) descriptor. Captures the GGUF
/// path, the parsed `MmprojConfig` header, the detected `ArchProfile`
/// (iter 31), the loaded-on-GPU weights wrapped in `Arc` for cheap
/// clone, and a stable `model_id` (file stem).
///
/// Weights are loaded eagerly at server startup so the first
/// multimodal request doesn't pay the ~10s mmap/dequant cost. The
/// `Arc` makes `LoadedMmproj` cheap to clone across handler calls
/// while keeping the GPU buffers singly-owned behind the Arc.
#[derive(Debug, Clone)]
pub struct LoadedMmproj {
    pub gguf_path: PathBuf,
    pub config: MmprojConfig,
    pub arch: ArchProfile,
    pub weights: Arc<LoadedMmprojWeights>,
    pub model_id: String,
}

impl AppState {
    /// Construct `AppState` without an engine. Only `/health`, `/readyz`,
    /// `/v1/models` will serve real data; generation endpoints return
    /// `model_not_loaded`. Used by the iter-2 backbone path and by tests.
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config: Arc::new(config),
            started_at: Arc::new(Instant::now()),
            // No engine → no warmup needed → ready.
            ready_for_gen: Arc::new(AtomicBool::new(true)),
            request_counter: Arc::new(AtomicU64::new(0)),
            engine: None,
            embedding_config: None,
            mmproj: None,
            metrics: Arc::new(ServerMetrics::default()),
        }
    }

    /// Construct `AppState` with an engine. The engine worker thread is
    /// already running at this point; `ready_for_gen` starts `false` until
    /// the warmup task flips it.
    pub fn with_engine(config: ServerConfig, engine: Engine) -> Self {
        Self {
            config: Arc::new(config),
            started_at: Arc::new(Instant::now()),
            ready_for_gen: Arc::new(AtomicBool::new(false)),
            request_counter: Arc::new(AtomicU64::new(0)),
            engine: Some(engine),
            embedding_config: None,
            mmproj: None,
            metrics: Arc::new(ServerMetrics::default()),
        }
    }

    /// Attach a BERT embedding model config. Cheap (clones internal
    /// references). Called by `cmd_serve` after validating the supplied
    /// GGUF header.
    pub fn with_embedding_model(mut self, em: EmbeddingModel) -> Self {
        self.embedding_config = Some(em);
        self
    }

    /// Attach an mmproj descriptor. Called by `cmd_serve` after validating
    /// the supplied mmproj GGUF header. The ViT forward pass that consumes
    /// this lands in ADR-005 Phase 2c Task #15.
    pub fn with_mmproj(mut self, m: LoadedMmproj) -> Self {
        self.mmproj = Some(m);
        self
    }

    /// Seconds since the server started.
    pub fn uptime_seconds(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }

    /// Mark the server ready for generation (called after warmup).
    pub fn mark_ready_for_gen(&self) {
        self.ready_for_gen.store(true, Ordering::Release);
    }

    /// Mark the server NOT ready (e.g. during graceful shutdown drain).
    pub fn mark_not_ready(&self) {
        self.ready_for_gen.store(false, Ordering::Release);
    }

    pub fn is_ready_for_gen(&self) -> bool {
        self.ready_for_gen.load(Ordering::Acquire)
    }

    /// Allocate the next request counter value.
    pub fn next_request_seq(&self) -> u64 {
        self.request_counter.fetch_add(1, Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_localhost() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.port, 8080);
        assert!(cfg.auth_token.is_none());
        assert!(cfg.cors_allowed_origins.is_empty());
        assert_eq!(cfg.queue_capacity, 32);
        assert_eq!(cfg.default_overflow_policy, OverflowPolicy::Summarize);
    }

    #[test]
    fn app_state_starts_ready_in_iter_2() {
        let state = AppState::new(ServerConfig::default());
        assert!(state.is_ready_for_gen());
        assert_eq!(state.uptime_seconds(), 0);
    }

    #[test]
    fn mark_not_ready_flips_to_false_then_back() {
        let state = AppState::new(ServerConfig::default());
        assert!(state.is_ready_for_gen());
        state.mark_not_ready();
        assert!(!state.is_ready_for_gen());
        state.mark_ready_for_gen();
        assert!(state.is_ready_for_gen());
    }

    #[test]
    fn request_seq_is_monotonic() {
        let state = AppState::new(ServerConfig::default());
        let a = state.next_request_seq();
        let b = state.next_request_seq();
        let c = state.next_request_seq();
        assert_eq!(a + 1, b);
        assert_eq!(b + 1, c);
    }

    #[test]
    fn embedding_model_encode_round_trips_hello() {
        // Build a minimal 6-token synthetic BERT vocab + tokenizer.
        // Verifies the EmbeddingModel::encode wrapper wires through the
        // tokenizers crate correctly (integration of iter-20 tokenizer
        // builder with iter-21 state struct).
        use crate::inference::models::bert::{
            build_wordpiece_tokenizer, BertConfig, BertSpecialTokens, BertVocab, PoolingType,
        };
        let vocab = BertVocab {
            tokens: vec![
                "[UNK]".into(),
                "[CLS]".into(),
                "[SEP]".into(),
                "[PAD]".into(),
                "hello".into(),
                "world".into(),
            ],
            specials: BertSpecialTokens {
                cls: 1,
                sep: 2,
                pad: 3,
                unk: 0,
                mask: 0,
            },
        };
        let tokenizer = build_wordpiece_tokenizer(&vocab).expect("build");
        let em = EmbeddingModel {
            gguf_path: "/tmp/synthetic.gguf".into(),
            config: BertConfig {
                hidden_size: 384,
                num_attention_heads: 12,
                num_hidden_layers: 12,
                intermediate_size: 1536,
                max_position_embeddings: 512,
                vocab_size: 6,
                type_vocab_size: 2,
                layer_norm_eps: 1e-12,
                hidden_act: "gelu".into(),
                pooling_type: PoolingType::Mean,
                causal_attention: false,
            },
            vocab: Arc::new(vocab),
            tokenizer: Arc::new(tokenizer),
            model_id: "synthetic-embed".into(),
        };
        let ids = em.encode("hello world").expect("encode");
        assert!(ids.contains(&4), "expected 'hello'=4 in {:?}", ids);
        assert!(ids.contains(&5), "expected 'world'=5 in {:?}", ids);
    }

    #[test]
    fn with_mmproj_attaches_descriptor_to_state() {
        // Verifies the `with_mmproj` builder — iter 25 multimodal wiring.
        // Exercises the typed plumbing (field presence, model_id, path
        // round-trip) without touching a real GGUF; parsing is covered by
        // `inference::vision::mmproj::tests`.
        use crate::inference::vision::mmproj::{MmprojConfig, ProjectorType};
        let cfg = MmprojConfig {
            image_size: 896,
            patch_size: 14,
            num_patches_side: 64,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            layer_norm_eps: 1e-6,
            projector: ProjectorType::Mlp,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        };
        let device = mlx_native::MlxDevice::new().expect("create device");
        let m = LoadedMmproj {
            gguf_path: "/tmp/synthetic-mmproj.gguf".into(),
            config: cfg.clone(),
            arch: ArchProfile::Gemma4Siglip,
            weights: Arc::new(LoadedMmprojWeights::empty(device)),
            model_id: "synthetic-mmproj".into(),
        };
        let state = AppState::new(ServerConfig::default()).with_mmproj(m);
        let attached = state.mmproj.as_ref().expect("mmproj should be Some");
        assert_eq!(attached.model_id, "synthetic-mmproj");
        assert_eq!(attached.gguf_path.file_name().unwrap(), "synthetic-mmproj.gguf");
        assert_eq!(attached.config, cfg);
        assert_eq!(attached.arch, ArchProfile::Gemma4Siglip);
        assert!(attached.config.projector.is_supported());
    }
}
