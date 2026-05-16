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

use std::collections::HashMap;
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
use crate::core::hardware::HardwareProfile;
use crate::serve::cache::ModelCache;
use crate::serve::multi_model::{
    DefaultModelLoader, HotSwapManager, LoadedPool, RestoreErrorKind, RestoreOutcome,
    SpillErrorKind, SpillOutcome,
};
use crate::serve::quant_select::QuantType;

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

// ---------------------------------------------------------------------------
// ADR-005 Phase 4 reopen iter-213 (AC 5472) — KV-spill telemetry counters.
// ---------------------------------------------------------------------------

/// Counter outcome label cardinality.  CLOSED ENUM.  ADR-017 Phase C MUST
/// satisfy this set; if a fifth outcome is genuinely needed (e.g.
/// `version_mismatch` for D10 cache-version envelope), amend ADR-005
/// Phase 4 with a `5473` telemetry-extension AC.  Cardinality additions
/// are non-breaking (Prometheus tolerates new label values), but the
/// closed-enum guard is the contract: the iter-212 [`SpillOutcome`] /
/// [`RestoreOutcome`] enums MUST map onto exactly these four labels.
///
/// Order is load-bearing: it matches `record_*` index lookup in
/// [`KvSpillCounters`] and the emit order in `metrics_handler`, so
/// scrape diffs stay stable across iters.  Adding a new label MUST
/// append to the end (not insert mid-array) to preserve scrape-line
/// ordering.
pub const KV_SPILL_OUTCOMES: &[&str] = &["success", "codec_err", "io_err", "parity_fail"];

/// Outcome enum-as-usize index helper — keeps [`KvSpillCounters`] table
/// lookup branchless.  Must stay in sync with [`KV_SPILL_OUTCOMES`].
const KV_OUTCOME_SUCCESS: usize = 0;
const KV_OUTCOME_CODEC_ERR: usize = 1;
const KV_OUTCOME_IO_ERR: usize = 2;
const KV_OUTCOME_PARITY_FAIL: usize = 3;
const KV_OUTCOME_COUNT: usize = 4;

/// Process-wide KV-spill / KV-restore telemetry counters surfaced via
/// `/metrics` (ADR-005 Phase 4 reopen iter-213, AC 5472).  Counters
/// are keyed by `(repo, quant, outcome)` per the AC's emit format:
///
/// ```text
/// hf2q_pool_kv_spills_total{repo="...",quant="Q4_0",outcome="success"} 0
/// hf2q_pool_kv_spills_total{repo="...",quant="Q4_0",outcome="codec_err"} 0
/// hf2q_pool_kv_spills_total{repo="...",quant="Q4_0",outcome="io_err"} 0
/// hf2q_pool_kv_spills_total{repo="...",quant="Q4_0",outcome="parity_fail"} 0
/// ```
///
/// Key choice: `(repo, quant)` is a `(String, String)` tuple — the
/// `repo` arrives from `HotSwapManager::load_or_get` (request-time) and
/// `quant` is the canonical [`QuantType::as_str`] form (matches the
/// `LoadedHandle.quant` shape).  Behind a `Mutex<HashMap<...>>` because
/// each `(repo, quant)` slot stores a `[AtomicU64; KV_OUTCOME_COUNT]`
/// row that is bumped atomically once the slot exists; the Mutex is
/// taken only on first observation of a new key (lazy init) and on
/// scrape-time iteration.
///
/// Per-call (NOT per-block) increment semantics: when `MockSpiller`
/// returns `EnqueuedBlocks(N)`, `outcome="success"` increments by 1.
/// Block count rides on a separate gauge in ADR-017 Phase C if needed.
///
/// `Skipped` outcomes do NOT increment any counter — they signal "no
/// work was done" (the noop spiller's default).  Counting Skipped
/// would conflate the noop fast path with successful spills, which is
/// the AC 5472 closed-enum guard's whole point.
///
/// ADR-017 §R-F7 (adversarial review 2026-05-01): the same telemetry
/// surface also owns the four cache-side counters that ADR-017
/// specifies but that Phase 4 left unwired:
///
///   * `hf2q_kv_quarantined_total{reason}` — bumped on every move to
///     `kv-quarantine/`. Reason label = `QuarantineReason::as_str()`
///     (`trunc` / `verbump` / `bodyhash` / `parity`).
///   * `hf2q_kv_cache_evictions_total{trigger}` — bumped per evicted
///     block in `evict_lru_until_under_budget`. Trigger label =
///     `"budget_overflow"` today; future labels (e.g. `"manual"`) MUST
///     append, not replace, to preserve scrape-line ordering.
///   * `hf2q_kv_cache_bytes_on_disk` (gauge) — sourced lazily on
///     `/metrics` scrape from the live `BlockIndex.total_bytes_on_disk()`
///     via `AppState.kv_disk_store`. Held outside this struct because
///     the gauge has no per-event-bump model — it's a pure read.
///   * `hf2q_kv_cache_blocks_total` (gauge) — same pattern, sourced
///     from `BlockIndex.block_count()`.
#[derive(Debug, Default)]
pub struct KvSpillCounters {
    /// `hf2q_pool_kv_spills_total` storage.  Inner row is indexed by
    /// `KV_OUTCOME_*` constants.
    spills: std::sync::Mutex<HashMap<(String, String), [AtomicU64; KV_OUTCOME_COUNT]>>,
    /// `hf2q_pool_kv_restores_total` storage.  Same shape.
    restores: std::sync::Mutex<HashMap<(String, String), [AtomicU64; KV_OUTCOME_COUNT]>>,
    /// `Server-Timing` response-header toggle.  Default `false`
    /// (iter-213 default-OFF per AC 5472).  ADR-017 Phase C
    /// `cmd_serve --kv-persist` flag flips this to `true` so spill /
    /// restore wall-clock surfaces on auto-swap reload responses.
    server_timing_enabled: AtomicBool,
    /// ADR-017 §R-F7: `hf2q_kv_quarantined_total{reason}` — one
    /// AtomicU64 per `QuarantineReason` variant. Index order MUST match
    /// [`KV_QUARANTINE_REASONS`] so the `/metrics` emit order is stable
    /// across scrapes.
    quarantines: [AtomicU64; KV_QUARANTINE_REASON_COUNT],
    /// ADR-017 §R-F7: `hf2q_kv_cache_evictions_total{trigger}` — one
    /// AtomicU64 per trigger label. Today only `"budget_overflow"`
    /// fires; the array is sized for that single trigger and grows on
    /// future-trigger admission per [`KV_EVICTION_TRIGGERS`].
    evictions: [AtomicU64; KV_EVICTION_TRIGGER_COUNT],
    /// ADR-017 Phase E option (a) iter-2: total per-request LCP probes
    /// after `PromptCache` full-equality miss. Increments once per
    /// post-miss probe regardless of detection outcome — denominator
    /// for the detection-rate gauge.
    lcp_lookups_total: AtomicU64,
    /// ADR-017 Phase E.a iter-2: subset of `lcp_lookups_total` where a
    /// non-trivial partial-prefix opportunity was detected
    /// (`0 < K < new_tokens.len()`). Numerator for the detection-rate
    /// gauge.  Iter-2 reports only — partial-prefill resume path stays
    /// OFF until iter-3 (env-gated `HF2Q_KV_LCP_RESUME=1`, default-OFF,
    /// Codex Phase-2b audit gated).
    lcp_detected_total: AtomicU64,
}

// ADR-017 §R-F7: re-export the trait + label constant set from the
// `kv_persist::metrics` seam. The labels live at the seam (not here)
// so the bump sites in `block_store` / `recovery` and the scrape sites
// in `api::handlers` can both reach them without `api::state` becoming
// a dependency of the narrow `kv_persist` lib facade. See the
// `kv_persist::metrics` module docs for the why.
pub use crate::serve::kv_persist::metrics::{
    KvCacheMetricsSink, KvQuarantineReason, KV_EVICTION_TRIGGERS, KV_EVICTION_TRIGGER_COUNT,
    KV_QUARANTINE_REASONS, KV_QUARANTINE_REASON_COUNT,
};
const KV_EVICTION_TRIGGER_BUDGET_OVERFLOW: usize = 0;

impl KvSpillCounters {
    /// Construct an empty counters table with `Server-Timing`
    /// response-header toggle DEFAULT-OFF (iter-213 invariant).
    pub fn new() -> Self {
        Self::default()
    }

    /// Map a [`SpillOutcome`] enum variant to its `KV_OUTCOME_*` index
    /// position.  Returns `None` for [`SpillOutcome::Skipped`] — the
    /// noop spiller's default outcome — because Skipped is not a
    /// counted operation per AC 5472 closed-enum semantics.
    fn spill_outcome_index(outcome: SpillOutcome) -> Option<usize> {
        match outcome {
            SpillOutcome::Skipped => None,
            SpillOutcome::EnqueuedBlocks(_) => Some(KV_OUTCOME_SUCCESS),
            SpillOutcome::Error(SpillErrorKind::CodecErr) => Some(KV_OUTCOME_CODEC_ERR),
            SpillOutcome::Error(SpillErrorKind::IoErr) => Some(KV_OUTCOME_IO_ERR),
            SpillOutcome::Error(SpillErrorKind::ParityFail) => Some(KV_OUTCOME_PARITY_FAIL),
        }
    }

    /// Map a [`RestoreOutcome`] variant to its `KV_OUTCOME_*` index.
    /// Symmetric to [`Self::spill_outcome_index`].
    fn restore_outcome_index(outcome: RestoreOutcome) -> Option<usize> {
        match outcome {
            RestoreOutcome::Skipped => None,
            RestoreOutcome::RestoredBlocks(_) => Some(KV_OUTCOME_SUCCESS),
            RestoreOutcome::Error(RestoreErrorKind::CodecErr) => Some(KV_OUTCOME_CODEC_ERR),
            RestoreOutcome::Error(RestoreErrorKind::IoErr) => Some(KV_OUTCOME_IO_ERR),
            RestoreOutcome::Error(RestoreErrorKind::ParityFail) => Some(KV_OUTCOME_PARITY_FAIL),
        }
    }

    /// Allocate a fresh four-AtomicU64 row for a new `(repo, quant)`
    /// key.  Inline so the lazy-init path in `record_*` stays terse.
    fn new_row() -> [AtomicU64; KV_OUTCOME_COUNT] {
        [
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
        ]
    }

    /// Record a spill outcome (per-call, NOT per-block).  Lazily
    /// initializes the `(repo, quant)` row to all-zeros on first
    /// observation so the scrape-time emit shows the full four-outcome
    /// cardinality from the moment any spill activity occurs against a
    /// `(repo, quant)` pair.  Skipped outcomes are a no-op (per AC
    /// 5472: Skipped does NOT increment).
    pub fn record_spill(&self, repo: &str, quant: QuantType, outcome: SpillOutcome) {
        let Some(idx) = Self::spill_outcome_index(outcome) else {
            return; // Skipped — noop spiller; do NOT increment.
        };
        let key = (repo.to_string(), quant.as_str().to_string());
        let mut guard = self.spills.lock().expect("kv_spill_counters poisoned");
        let row = guard.entry(key).or_insert_with(Self::new_row);
        row[idx].fetch_add(1, Ordering::Relaxed);
    }

    /// Record a restore outcome.  Symmetric to [`Self::record_spill`].
    pub fn record_restore(&self, repo: &str, quant: QuantType, outcome: RestoreOutcome) {
        let Some(idx) = Self::restore_outcome_index(outcome) else {
            return; // Skipped.
        };
        let key = (repo.to_string(), quant.as_str().to_string());
        let mut guard = self.restores.lock().expect("kv_spill_counters poisoned");
        let row = guard.entry(key).or_insert_with(Self::new_row);
        row[idx].fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot every observed `(repo, quant)` spill row as
    /// `((repo, quant), [success, codec_err, io_err, parity_fail])`.
    /// Used by the `/metrics` handler to emit the four-outcome
    /// cardinality lines.  Order is sorted lexicographically on
    /// `(repo, quant)` so successive scrapes are stable for diff
    /// tooling.
    pub fn snapshot_spills(&self) -> Vec<((String, String), [u64; KV_OUTCOME_COUNT])> {
        let guard = self.spills.lock().expect("kv_spill_counters poisoned");
        let mut out: Vec<_> = guard
            .iter()
            .map(|(k, row)| {
                (
                    k.clone(),
                    [
                        row[0].load(Ordering::Relaxed),
                        row[1].load(Ordering::Relaxed),
                        row[2].load(Ordering::Relaxed),
                        row[3].load(Ordering::Relaxed),
                    ],
                )
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Snapshot every observed `(repo, quant)` restore row.  Symmetric
    /// to [`Self::snapshot_spills`].
    pub fn snapshot_restores(&self) -> Vec<((String, String), [u64; KV_OUTCOME_COUNT])> {
        let guard = self.restores.lock().expect("kv_spill_counters poisoned");
        let mut out: Vec<_> = guard
            .iter()
            .map(|(k, row)| {
                (
                    k.clone(),
                    [
                        row[0].load(Ordering::Relaxed),
                        row[1].load(Ordering::Relaxed),
                        row[2].load(Ordering::Relaxed),
                        row[3].load(Ordering::Relaxed),
                    ],
                )
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// `Server-Timing` response-header gate.  Default `false`
    /// (iter-213 default-OFF).  ADR-017 Phase C `cmd_serve
    /// --kv-persist` flag flips this to `true`.
    pub fn server_timing_enabled(&self) -> bool {
        self.server_timing_enabled.load(Ordering::Acquire)
    }

    /// Set the Server-Timing toggle.  Wired by ADR-017 Phase C
    /// `cmd_serve --kv-persist`; iter-213 keeps it default-OFF and
    /// asserts the default via `iter213_server_timing_header_default_off`.
    pub fn set_server_timing_enabled(&self, enabled: bool) {
        self.server_timing_enabled.store(enabled, Ordering::Release);
    }

    // -----------------------------------------------------------------
    // ADR-017 §R-F7 — cache-side counters (quarantine / eviction).
    // -----------------------------------------------------------------

    /// Snapshot the four quarantine-reason rows in
    /// [`KV_QUARANTINE_REASONS`] order. Used by `/metrics` to emit the
    /// closed-enum cardinality block (all four lines emit
    /// unconditionally — Prometheus convention; absent counter ⇒ no
    /// histogram, missed alerts). The bump method
    /// [`KvCacheMetricsSink::record_quarantine`] is implemented for
    /// this struct via the trait impl below.
    pub fn snapshot_quarantines(&self) -> [u64; KV_QUARANTINE_REASON_COUNT] {
        let mut out = [0u64; KV_QUARANTINE_REASON_COUNT];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.quarantines[i].load(Ordering::Relaxed);
        }
        out
    }

    /// Snapshot the eviction-trigger row(s) in [`KV_EVICTION_TRIGGERS`]
    /// order. One element per trigger label (today: just
    /// `"budget_overflow"`). The bump method
    /// [`KvCacheMetricsSink::record_eviction_budget_overflow`] is
    /// implemented for this struct via the trait impl below.
    pub fn snapshot_evictions(&self) -> [u64; KV_EVICTION_TRIGGER_COUNT] {
        [self.evictions[KV_EVICTION_TRIGGER_BUDGET_OVERFLOW].load(Ordering::Relaxed)]
    }

    /// ADR-017 Phase E.a iter-2: snapshot the (lookups, detected) tuple
    /// for `/metrics` emission. Both counters are emitted unconditionally
    /// — Prometheus convention; absent counter ⇒ no histogram, missed
    /// alerts (mirrors the quarantine-array discipline at
    /// [`Self::snapshot_quarantines`]).
    pub fn snapshot_lcp(&self) -> (u64, u64) {
        (
            self.lcp_lookups_total.load(Ordering::Relaxed),
            self.lcp_detected_total.load(Ordering::Relaxed),
        )
    }
}

/// ADR-017 §R-F7: production [`KvCacheMetricsSink`] impl. The substrate
/// (`kv_persist::block_store`, `kv_persist::recovery`) holds an
/// `Arc<dyn KvCacheMetricsSink>` and bumps via this impl; the
/// `/metrics` handler reads via [`KvSpillCounters::snapshot_quarantines`]
/// + [`KvSpillCounters::snapshot_evictions`] on the same Arc.
impl KvCacheMetricsSink for KvSpillCounters {
    fn record_quarantine(&self, reason: KvQuarantineReason) {
        self.quarantines[reason.index()].fetch_add(1, Ordering::Relaxed);
    }

    fn record_eviction_budget_overflow(&self) {
        self.evictions[KV_EVICTION_TRIGGER_BUDGET_OVERFLOW].fetch_add(1, Ordering::Relaxed);
    }

    /// ADR-017 Phase E.a iter-2: record one LCP probe outcome.
    /// `detected_k.is_some()` ⇒ both `lookups_total` and
    /// `detected_total` increment; `None` ⇒ only `lookups_total`. The
    /// `_k_value` discard is intentional — iter-2 keeps the cardinality
    /// at 2 counters; iter-3+ may swap to a histogram if K-bucketed
    /// distribution ever becomes the load-bearing observability shape.
    fn record_lcp_probe(&self, detected_k: Option<usize>) {
        self.lcp_lookups_total.fetch_add(1, Ordering::Relaxed);
        if detected_k.is_some() {
            self.lcp_detected_total.fetch_add(1, Ordering::Relaxed);
        }
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
    /// KV-spill / KV-restore telemetry counters surfaced via `/metrics`
    /// (ADR-005 Phase 4 reopen iter-213, AC 5472).  Shared between the
    /// metrics handler (read path; emits Prometheus text) and the
    /// HotSwapManager trigger sites in `multi_model.rs::load_or_get` /
    /// `multi_model.rs::evict` (write path; bumps the per-outcome row
    /// once per `pre_evict` / `post_admit` call).  The Arc is cloned
    /// into the manager via [`HotSwapManager::with_kv_counters`] at
    /// `AppState` construction so both paths see the same counters
    /// without an extra hop through the pool RwLock.
    pub kv_spill_counters: Arc<KvSpillCounters>,
    /// ADR-017 §R-F7 (adversarial review 2026-05-01): handle to the live
    /// on-disk block store, exposed so the `/metrics` handler can source
    /// the `hf2q_kv_cache_bytes_on_disk` and `hf2q_kv_cache_blocks_total`
    /// gauges directly from the authoritative `BlockIndex` at scrape
    /// time — no separate running counter, no consistency-window race
    /// between insert/remove paths and the gauge read.
    ///
    /// `None` when `--kv-persist` is absent (the entire kv-persist
    /// substrate is opt-in per ADR-017 Phase C.1); both gauges report
    /// `0` in that mode so the surface stays parseable.
    ///
    /// Wired by `cmd_serve` after `DiskBlockStore::new_with_index`
    /// returns, via [`Self::with_kv_disk_store`].
    pub kv_disk_store: Option<Arc<crate::serve::kv_persist::DiskBlockStore>>,
    /// ADR-017 Closure iter-2 (2026-05-04): the live
    /// `BlockPrefixCacheSpiller` constructed in `cmd_serve` when
    /// `--kv-persist` is enabled. Held here (concrete type, not
    /// trait object) so `drain_loaded_models_to_disk` (graceful-
    /// shutdown spill) can poll
    /// [`BlockPrefixCacheSpiller::pending_writer_queue_depth`] to
    /// know when the async writer queue has drained. The same `Arc`
    /// is also wired into `HotSwapManager::new_with_spiller` (as a
    /// trait object) so the eviction trigger sites at
    /// `multi_model.rs:1090, 1189` keep working unchanged.
    ///
    /// `None` when `--kv-persist` is absent (kv-persist substrate is
    /// opt-in per Phase C.1); the graceful-shutdown drain becomes a
    /// no-op in that mode.
    pub kv_spiller: Option<
        Arc<
            crate::serve::kv_persist::BlockPrefixCacheSpiller<
                crate::serve::api::engine::Engine,
            >,
        >,
    >,
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
        let hardware = crate::core::hardware::HardwareProfiler::detect()
            .map_err(|e| anyhow::anyhow!("hardware detection: {e}"))?;
        let cache = ModelCache::open()?;
        let pool = LoadedPool::from_hardware(&hardware);
        // ADR-005 Phase 4 reopen iter-213 (AC 5472): KV-spill counters
        // are owned by AppState and Arc-cloned into HotSwapManager so the
        // trigger sites bump the same table the /metrics handler reads.
        let kv_spill_counters = Arc::new(KvSpillCounters::new());
        let mut manager = HotSwapManager::new(pool, Arc::new(DefaultModelLoader));
        manager.set_kv_counters(Arc::clone(&kv_spill_counters));
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
            kv_spill_counters,
            // ADR-017 §R-F7: gauge source wired post-construction by
            // `cmd_serve` once `--kv-persist` builds the substrate.
            kv_disk_store: None,
            // ADR-017 Closure iter-2: graceful-shutdown spiller handle
            // wired post-construction by `cmd_serve` when `--kv-persist`
            // builds the substrate; otherwise None and the drain is a
            // no-op.
            kv_spiller: None,
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
        // ADR-005 Phase 4 reopen iter-213 (AC 5472): KV-spill counters
        // wired in the test path identically to `new_for_serve` so router
        // tests can scrape `/metrics` and observe the four-outcome
        // cardinality lines.  Counters start at zero per Prometheus
        // convention.
        let kv_spill_counters_test = Arc::new(KvSpillCounters::new());
        let mut manager = HotSwapManager::new(pool, Arc::new(DefaultModelLoader));
        manager.set_kv_counters(Arc::clone(&kv_spill_counters_test));
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
            kv_spill_counters: kv_spill_counters_test,
            // ADR-017 §R-F7: tests hand-wire this when they want to
            // exercise the gauge surface; default `None` keeps router
            // unit tests independent of the kv-persist substrate.
            kv_disk_store: None,
            // ADR-017 Closure iter-2: tests start without a spiller; the
            // shutdown drain is a no-op for the test path.
            kv_spiller: None,
        }
    }

    /// Set the default model lookup key (the original `--model` CLI
    /// argument).  Returned by-value for builder chaining.
    pub fn with_default_model(mut self, default_model: Option<String>) -> Self {
        self.default_model = default_model;
        self
    }

    /// ADR-017 §R-F7: attach the live `DiskBlockStore` so the
    /// `/metrics` handler can source `hf2q_kv_cache_bytes_on_disk` and
    /// `hf2q_kv_cache_blocks_total` from the authoritative `BlockIndex`
    /// at scrape time. Called by `cmd_serve` after the kv-persist
    /// substrate is constructed; off-path (no `--kv-persist`) leaves
    /// this `None` and both gauges report `0`.
    pub fn with_kv_disk_store(
        mut self,
        store: Arc<crate::serve::kv_persist::DiskBlockStore>,
    ) -> Self {
        self.kv_disk_store = Some(store);
        self
    }

    /// ADR-017 Closure iter-2 (2026-05-04): attach the live
    /// `BlockPrefixCacheSpiller` so the graceful-shutdown drain
    /// (`drain_loaded_models_to_disk`) can poll
    /// [`BlockPrefixCacheSpiller::pending_writer_queue_depth`] until
    /// the async writer queue has drained. Called by `cmd_serve`
    /// after the spiller substrate is constructed; off-path (no
    /// `--kv-persist`) leaves this `None` and the drain becomes a
    /// no-op (logs `kv-persist not enabled; skipping drain`).
    pub fn with_kv_spiller(
        mut self,
        spiller: Arc<
            crate::serve::kv_persist::BlockPrefixCacheSpiller<
                crate::serve::api::engine::Engine,
            >,
        >,
    ) -> Self {
        self.kv_spiller = Some(spiller);
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
            build_wordpiece_tokenizer, BertSpecialTokens, BertVocab,
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
            // iter-224 Wedge-4b: Qwen3-VL-only fields default to None on
            // non-Qwen3-VL fixtures.
            spatial_merge_size: None,
            projection_dim: None,
            deepstack_indexes: None,
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

    // ─────────────────────────────────────────────────────────────────────
    // ADR-017 Phase E option (a) iter-2 — `record_lcp_probe` counter tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn kv_spill_counters_lcp_probe_records_lookups_unconditionally() {
        // ADR-017 Phase E.a iter-2: every probe (even when no
        // partial-prefix opportunity exists) bumps `lookups_total`.
        // Operators read both counters from /metrics and compute the
        // detection-rate gauge `(detected/lookups)` in dashboards;
        // requiring `lookups_total` to be the denominator is the
        // load-bearing invariant tested here.
        let counters = KvSpillCounters::new();

        // Probe outcomes: miss (None) → only lookups_total bumps.
        counters.record_lcp_probe(None);
        counters.record_lcp_probe(None);
        counters.record_lcp_probe(None);

        let (lookups, detected) = counters.snapshot_lcp();
        assert_eq!(lookups, 3, "every probe must bump lookups_total");
        assert_eq!(detected, 0, "None outcome must NOT bump detected_total");
    }

    #[test]
    fn kv_spill_counters_lcp_probe_increments_detected_on_some() {
        // ADR-017 Phase E.a iter-2: a `Some(K)` outcome bumps both
        // counters. `K` value is intentionally NOT recorded in iter-2
        // — the histogram-by-K shape would change cardinality and
        // break dashboard-stable scrape lines.
        let counters = KvSpillCounters::new();

        counters.record_lcp_probe(Some(5));
        counters.record_lcp_probe(Some(127));
        counters.record_lcp_probe(None); // miss between hits
        counters.record_lcp_probe(Some(42));

        let (lookups, detected) = counters.snapshot_lcp();
        assert_eq!(lookups, 4, "all 4 probes must bump lookups_total");
        assert_eq!(
            detected, 3,
            "3 Some outcomes must bump detected_total; 1 None must not"
        );
    }

    #[test]
    fn kv_spill_counters_lcp_probe_starts_at_zero() {
        // Defensive: a fresh counters table reports (0, 0) for LCP.
        // The `/metrics` handler emits these unconditionally so a pre-
        // first-request scrape sees `hf2q_kv_lcp_lookups_total 0`
        // (Prometheus convention; absent counter ⇒ no histogram).
        let counters = KvSpillCounters::new();
        assert_eq!(counters.snapshot_lcp(), (0, 0));
    }
}
