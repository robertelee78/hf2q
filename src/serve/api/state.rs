//! Shared application state (`AppState`) for the hf2q HTTP API server.
//!
//! The axum router threads a single `AppState` through every handler.  In
//! ADR-005 Phase 4 iter-209 (W77) the single-slot `engine: Option<Engine>`
//! field was replaced with `pool: Arc<RwLock<HotSwapManager<Engine>>>`:
//! request-time auto-swap (Decision #26) routes the OpenAI `model:` field
//! through the pool, evicting LRU entries under capacity / memory-budget
//! pressure (W74 `LoadedPool` + W76 `HotSwapManager`).  The pool starts
//! empty when `--model` is not supplied; the first request specifying a
//! model triggers an auto-load via [`crate::serve::auto_pipeline`].
//!
//! Decision #26 surface stays compatible: `400 model_not_loaded` is
//! returned when a request names a model that auto_pipeline cannot
//! resolve (not on disk + not a valid HF repo-id) — i.e., a genuinely
//! un-loadable input — while previously-cached or repo-id models
//! auto-swap transparently.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use super::engine::Engine;
use super::schema::OverflowPolicy;
use crate::inference::models::bert::config::PoolingType;
use crate::inference::models::bert::BertConfig;
use crate::inference::models::bert::weights::LoadedBertWeights;
use crate::inference::models::nomic_bert::{LoadedNomicBertWeights, NomicBertConfig};
use crate::inference::vision::mmproj::{ArchProfile, MmprojConfig};
use crate::inference::vision::mmproj_weights::LoadedMmprojWeights;
use crate::intelligence::hardware::HardwareProfile;
use crate::serve::cache::ModelCache;
use crate::serve::multi_model::{
    DefaultModelLoader, HotSwapManager, LoadedPool,
};

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

/// Construct a unique-per-process tempdir cache root for the test path
/// (`AppState::new`).  Each `AppState::new` call yields a fresh root so
/// concurrent test threads never share manifest state.  The directory
/// is left on disk after the test — `std::env::temp_dir()` is platform-
/// specific and the OS reaps it; tests that care set their own root
/// via [`AppState::new_for_serve`] / `cli::ServeArgs.cache_dir`.
fn synthetic_cache_root() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let mut p = std::env::temp_dir();
    p.push(format!("hf2q-test-cache-{pid}-{id}"));
    p
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
///
/// ADR-005 Phase 4 iter-209 (W77) replaced the single-slot
/// `engine: Option<Engine>` field with a [`HotSwapManager<Engine>`] pool
/// behind `Arc<RwLock<...>>`.  `load_or_get` is mutating (LRU touch + insert)
/// so request handlers acquire the write-lock briefly to admit a new model
/// or promote a cached one; `try_get` is non-mutating and could be served
/// under a read-lock for diagnostic / metrics endpoints.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ServerConfig>,
    pub started_at: Arc<Instant>,
    /// `true` once the server is ready to serve generation.  After Phase 4
    /// iter-209 the pool is empty at process start when `--model` is not
    /// supplied; the flag still gates the first request through warmup
    /// (and ADR-005 Decision #16 applies on auto-swap reloads — the
    /// re-load path runs synchronous warmup before returning).
    pub ready_for_gen: Arc<AtomicBool>,
    /// Monotonic counter for request-id generation + metrics.
    pub request_counter: Arc<AtomicU64>,
    /// Multi-model engine pool.  Replaces the pre-Phase-4 `Option<Engine>`.
    /// Empty pool + no [`Self::default_model`] means the server is HTTP-only
    /// (every generation request returns 400 `model_not_loaded` — the
    /// auto-pipeline cannot resolve an empty / unspecified `req.model`).
    pub pool: Arc<std::sync::RwLock<HotSwapManager<Engine>>>,
    /// On-disk cache (`~/.cache/hf2q/`).  Held behind `Arc<Mutex<_>>` so
    /// concurrent handlers that resolve a `req.model` through the
    /// auto-pipeline (which may mutate the manifest on download /
    /// quantize / touch) serialize on the same cache instance.
    pub cache: Arc<std::sync::Mutex<ModelCache>>,
    /// Hardware profile detected once at startup.  Used by the
    /// auto-pipeline's quant selector + the pool's memory-budget
    /// adapter; immutable for the lifetime of the process.
    pub hardware: Arc<HardwareProfile>,
    /// `--no-integrity` operator opt-out (off by default).  When `true`,
    /// the cache integrity re-check on every load is skipped (with a
    /// stern warning logged at request time).  Mirrors the
    /// `cli::ServeArgs.no_integrity` field.
    pub no_integrity: bool,
    /// FIFO queue capacity for newly-loaded engines.  Mirrors
    /// `cli::ServeArgs.queue_capacity` (Decision #19 surface).  Captured
    /// at startup and threaded into every `EngineConfig` the loader
    /// dispatches with.
    pub engine_queue_capacity: usize,
    /// `--model` argument from CLI startup, if any.  Used as the fallback
    /// "default model" when a request omits the OpenAI `model:` field
    /// (or sends an empty string).  Stored as the original argument
    /// string so the auto-pipeline classifies it the same way the
    /// startup pre-warm did.
    pub default_model: Option<String>,
    /// BERT embedding model config (from `--embedding-model <path>`).
    /// `None` when no embedding model was supplied. Validated at startup
    /// via `BertConfig::from_gguf` — the forward pass that consumes this
    /// lands in a later iter (ADR-005 Phase 2b).
    pub embedding_config: Option<EmbeddingModel>,
    /// Persistent kernel registry for embedding forwards. Pre-warmed at
    /// server boot via one warmup forward so all needed pipelines are
    /// compiled and cached. Per-request handlers lock briefly, dispatch
    /// against the cached registry, release. Eliminates the ~150 ms
    /// per-request shader-compile cost the iter-82 benchmark surfaced
    /// (kept registry per-request → recompiled every shader; HTTP-path
    /// hit ~190 ms vs in-process ~8 ms forward floor).
    pub embedding_registry: Option<Arc<std::sync::Mutex<mlx_native::KernelRegistry>>>,
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
    pub vocab: Arc<crate::inference::models::bert::BertVocab>,
    /// llama.cpp-compatible WordPiece tokenizer (uses ▁-prefix word
    /// starters, matches the bge / nomic / mxbai GGUF format byte-for-
    /// byte). Shared across BERT-family architectures because all of
    /// them use the same WPM vocab convention in GGUF (per
    /// `llm_tokenizer_wpm_session::tokenize` in llama.cpp).
    pub tokenizer: Arc<crate::inference::models::bert::BertWpmTokenizer>,
    /// Model id (file stem) — surfaced via `/v1/models`.
    pub model_id: String,
    /// Architecture variant. Carries the per-arch config + weights so
    /// the handler dispatches the correct forward pass. Optional only
    /// in the test-scaffolding path that bypasses real weight loading;
    /// production always populates this via `cmd_serve`.
    pub arch: Option<EmbeddingArch>,
}

/// Per-arch config + weights bundle. The handler matches on this enum
/// to dispatch the correct forward pass:
///   - `Bert` → `apply_bert_full_forward_gpu` (separate Q/K/V, GeLU MLP,
///     position_embd lookup, CLS/Mean pool per `bert.pooling_type`).
///   - `NomicBert` → `apply_nomic_bert_full_forward_gpu` (fused QKV,
///     SwiGLU MLP, RoPE on Q/K, Mean pool per `nomic-bert.pooling_type`).
///
/// Common properties (hidden_size, max_position_embeddings, pooling_type,
/// layer count) are exposed via accessor methods so the handler can
/// share validation logic across both variants.
#[derive(Debug, Clone)]
pub enum EmbeddingArch {
    Bert {
        config: BertConfig,
        weights: Arc<LoadedBertWeights>,
    },
    NomicBert {
        config: NomicBertConfig,
        weights: Arc<LoadedNomicBertWeights>,
    },
}

impl EmbeddingArch {
    /// Output embedding dimension (a.k.a. `hidden_size` in HF / GGUF).
    /// Used for the `dimensions` parameter validation in `/v1/embeddings`.
    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Bert { config, .. } => config.hidden_size,
            Self::NomicBert { config, .. } => config.hidden_size,
        }
    }

    /// Maximum sequence length the model was trained for. Used to
    /// truncate over-long inputs before the forward pass.
    pub fn max_position_embeddings(&self) -> usize {
        match self {
            Self::Bert { config, .. } => config.max_position_embeddings,
            Self::NomicBert { config, .. } => config.max_position_embeddings,
        }
    }

    /// Pooling reduction (Mean / CLS / Last) read from the GGUF
    /// metadata. Surfaced via `/v1/models` extension fields.
    pub fn pooling_type(&self) -> PoolingType {
        match self {
            Self::Bert { config, .. } => config.pooling_type,
            Self::NomicBert { config, .. } => config.pooling_type,
        }
    }

    /// Architecture name as it appears in GGUF `general.architecture`.
    pub fn arch_name(&self) -> &'static str {
        match self {
            Self::Bert { .. } => "bert",
            Self::NomicBert { .. } => "nomic-bert",
        }
    }
}

impl std::fmt::Debug for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingModel")
            .field("gguf_path", &self.gguf_path)
            .field("arch", &self.arch.as_ref().map(|a| a.arch_name()))
            .field("hidden", &self.arch.as_ref().map(|a| a.hidden_size()))
            .field("vocab_len", &self.vocab.len())
            .field("model_id", &self.model_id)
            .finish()
    }
}

impl EmbeddingModel {
    /// Convenience: tokenize a single input string using the embedded
    /// WordPiece tokenizer. Returns the token-id vector. Matches
    /// llama.cpp's WPM tokenizer; pass `add_special_tokens=true` to
    /// wrap the output in `[CLS] ... [SEP]`.
    pub fn encode(&self, input: &str, add_special_tokens: bool) -> Vec<u32> {
        self.tokenizer.encode(input, add_special_tokens)
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
    /// Construct `AppState` for production use — opens (or creates) the
    /// on-disk cache, detects hardware once, and constructs an empty
    /// [`HotSwapManager`] sized off the unified-memory budget per
    /// ADR-005 line 929 (80% default).  `cmd_serve` calls this then
    /// optionally pre-warms the pool with the `--model` argument before
    /// passing to the router.
    ///
    /// Errors propagate from `ModelCache::open` (filesystem permissions
    /// on `~/.cache/hf2q/`) and `HardwareProfiler::detect` (sysinfo
    /// unavailable).  Tests use [`Self::new`] (synthetic-fixture path)
    /// to avoid real filesystem + sysinfo dependencies.
    pub fn new_for_serve(
        config: ServerConfig,
        no_integrity: bool,
        engine_queue_capacity: usize,
        default_model: Option<String>,
    ) -> anyhow::Result<Self> {
        let hardware = crate::intelligence::hardware::HardwareProfiler::detect()
            .map_err(|e| anyhow::anyhow!("hardware detection: {e}"))?;
        let cache = ModelCache::open()?;
        let pool = LoadedPool::from_hardware(&hardware);
        let manager = HotSwapManager::new(pool, Arc::new(DefaultModelLoader));
        Ok(Self {
            config: Arc::new(config),
            started_at: Arc::new(Instant::now()),
            // ready_for_gen starts true: the pool is the gating surface for
            // generation; warmup is per-load (synchronous) inside the
            // loader.  /readyz reports server liveness, not a per-model
            // warmup status (which is now an auto-swap concept).
            ready_for_gen: Arc::new(AtomicBool::new(true)),
            request_counter: Arc::new(AtomicU64::new(0)),
            pool: Arc::new(std::sync::RwLock::new(manager)),
            cache: Arc::new(std::sync::Mutex::new(cache)),
            hardware: Arc::new(hardware),
            no_integrity,
            engine_queue_capacity,
            default_model,
            embedding_config: None,
            embedding_registry: None,
            mmproj: None,
            metrics: Arc::new(ServerMetrics::default()),
        })
    }

    /// Construct `AppState` for tests / router unit tests — uses a
    /// synthetic empty pool with a 1 GiB memory budget and a tempdir
    /// cache so no real filesystem/sysinfo work runs.
    ///
    /// The pool starts empty; without `default_model` set, every
    /// generation request will return 400 `model_not_loaded` (the
    /// auto-pipeline cannot resolve a missing model name).  Tests
    /// asserting that 400-shape behaviour use this constructor.
    pub fn new(config: ServerConfig) -> Self {
        // Synthetic 1 GiB budget — tests never load a real engine; the
        // budget exists only so PoolError::ZeroCapacity / OversizedHandle
        // paths can be exercised under unit tests.
        let pool = LoadedPool::with_capacity_and_budget(3, 1u64 << 30);
        let manager = HotSwapManager::new(pool, Arc::new(DefaultModelLoader));
        // Synthetic cache root in a per-process tempdir.  Tests that need
        // a specific cache state should construct via `new_for_serve` or
        // hand-build an `AppState` (every field is `pub`).
        let cache = ModelCache::open_at(synthetic_cache_root())
            .expect("open synthetic cache for AppState::new (test path)");
        // Synthetic hardware (16 GiB) — tests don't depend on the value;
        // the auto-pipeline path is mocked out at the caller in test
        // contexts.
        let hardware = HardwareProfile {
            chip_model: "Synthetic-Test".into(),
            total_memory_bytes: 16u64 << 30,
            available_memory_bytes: 16u64 << 30,
            performance_cores: 8,
            efficiency_cores: 4,
            total_cores: 12,
            memory_bandwidth_gbs: 400.0,
        };
        Self {
            config: Arc::new(config),
            started_at: Arc::new(Instant::now()),
            ready_for_gen: Arc::new(AtomicBool::new(true)),
            request_counter: Arc::new(AtomicU64::new(0)),
            pool: Arc::new(std::sync::RwLock::new(manager)),
            cache: Arc::new(std::sync::Mutex::new(cache)),
            hardware: Arc::new(hardware),
            no_integrity: false,
            engine_queue_capacity: 32,
            default_model: None,
            embedding_config: None,
            embedding_registry: None,
            mmproj: None,
            metrics: Arc::new(ServerMetrics::default()),
        }
    }

    /// Set the default model lookup key (the original `--model` CLI
    /// argument).  Returned by-value for builder chaining.
    pub fn with_default_model(mut self, default_model: Option<String>) -> Self {
        self.default_model = default_model;
        self
    }

    /// Attach a BERT embedding model config. Cheap (clones internal
    /// references). Called by `cmd_serve` after validating the supplied
    /// GGUF header.
    pub fn with_embedding_model(mut self, em: EmbeddingModel) -> Self {
        self.embedding_config = Some(em);
        self
    }

    /// Attach a pre-warmed kernel registry for embedding forwards.
    /// Caller is responsible for registering the right kernels for the
    /// loaded arch (BERT custom shaders + mlx-native rope + silu_mul as
    /// appropriate) and running one warmup forward to compile every
    /// pipeline before stashing. Per-request handlers lock the inner
    /// `Mutex` briefly, dispatch against cached pipelines, release.
    pub fn with_embedding_registry(
        mut self,
        registry: Arc<std::sync::Mutex<mlx_native::KernelRegistry>>,
    ) -> Self {
        self.embedding_registry = Some(registry);
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
        // Synthetic vocab using llama.cpp's BERT-WPM convention:
        // word-starter tokens are prefixed with ▁ (U+2581). The
        // BertWpmTokenizer prepends ▁ to every input word before
        // greedy lookup, so the vocab MUST store the ▁-prefixed form.
        let vocab = BertVocab {
            tokens: vec![
                "[UNK]".into(),
                "[CLS]".into(),
                "[SEP]".into(),
                "[PAD]".into(),
                "\u{2581}hello".into(),
                "\u{2581}world".into(),
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
            vocab: Arc::new(vocab.clone()),
            tokenizer: Arc::new(crate::inference::models::bert::BertWpmTokenizer::new(&vocab)),
            model_id: "synthetic-embed".into(),
            arch: None,
        };
        let _ = tokenizer; // legacy HF tokenizer no longer used; kept for shape only
        let ids = em.encode("hello world", false);
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
