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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::engine::Engine;
use super::schema::OverflowPolicy;
use crate::core::hardware::HardwareProfile;
use crate::inference::models::bert::config::PoolingType;
use crate::inference::models::bert::weights::LoadedBertWeights;
use crate::inference::models::bert::BertConfig;
use crate::inference::models::nomic_bert::{LoadedNomicBertWeights, NomicBertConfig};
use crate::inference::vision::mmproj::{ArchProfile, MmprojConfig};
use crate::inference::vision::mmproj_weights::LoadedMmprojWeights;
use crate::serve::cache::ModelCache;
use crate::serve::multi_model::{
    DefaultModelLoader, EngineConfig, HotSwapManager, LoadedPool, RestoreErrorKind, RestoreOutcome,
    SpillErrorKind, SpillOutcome,
};
use crate::serve::quant_select::QuantType;

/// Server-level configuration, captured at startup from CLI flags + defaults.
///
/// All fields are immutable for the lifetime of the server. A restart is
/// required to change any of them (matching peer-server conventions).
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
    /// default); populated = restrictive whitelist.
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
    /// Record one successfully completed chat generation and its token usage.
    ///
    /// Unary and SSE handlers share this seam so the Prometheus counters have
    /// identical semantics. Callers must invoke it only after the generation
    /// has reached its successful terminal state; errors and cancellations do
    /// not count as completions.
    pub fn record_chat_completion_success(&self, prompt_tokens: usize, completion_tokens: usize) {
        self.chat_completions_completed
            .fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens_total
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.decode_tokens_total
            .fetch_add(completion_tokens as u64, Ordering::Relaxed);
    }

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
    /// ADR-047 model-admission gate plus generation-bound request leases.
    /// Ordinary OpenAI resolution takes a shared guard; explicit diagnostic
    /// switching takes the exclusive guard and drains exact generations.
    pub model_lifecycle: Arc<super::lifecycle::ModelLifecycleCoordinator>,
    /// Owns request-time metadata/transfer helpers until they have exited and
    /// been reaped. Server shutdown cancels this root before HTTP draining.
    pub preparations: super::cancellation::PreparationSupervisor,
    /// Bounded, ephemeral server authority for exact hosted artifact choices.
    pub artifact_catalog: super::artifact_catalog::ArtifactCatalogCoordinator,
    /// Bounded server-local roots eligible for receipt-backed diagnostic
    /// artifact discovery. Paths never cross the HTTP boundary.
    pub local_artifacts: super::local_artifacts::LocalArtifactInventory,
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
    /// Process-wide defaults for every dynamically loaded engine. Model-local
    /// paths are deliberately absent here; scheduler, queue, KV budget, and
    /// metrics wiring must survive model swaps unchanged.
    pub engine_config_template: EngineConfig,
    /// Exact canonical GGUF path -> model-local load configuration. Startup
    /// sidecars and explicit overlays belong only to the artifact they were
    /// supplied for and are restored when that same artifact is reloaded.
    pub engine_config_overrides: Arc<std::sync::RwLock<HashMap<PathBuf, EngineConfig>>>,
    /// `--model` argument from CLI startup, if any.  Used as the fallback
    /// "default model" when a request omits the OpenAI `model:` field
    /// (or sends an empty string).  Stored as the original argument
    /// string so the auto-pipeline classifies it the same way the
    /// startup pre-warm did.
    pub default_model: Option<String>,
    /// Dedicated encoder-model lifecycle.  A lease owns the model, tokenizer,
    /// and warmed kernel registry as one generation so a request can never
    /// combine state from two model loads.  Replacement fails while a lease is
    /// live and drops the old generation before invoking the next loader.
    pub embedding_slot: Arc<std::sync::RwLock<EmbeddingModelSlot>>,
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
        Arc<crate::serve::kv_persist::BlockPrefixCacheSpiller<crate::serve::api::engine::Engine>>,
    >,
}

/// Loaded BERT-family embedding generation. Config, native mapped weights,
/// tokenizer, vocabulary, and warmed registry are admitted atomically so a
/// request cannot observe a mixture of two model loads.
///
/// Shared via `Arc` so multiple handler calls can tokenize concurrently
/// against the same immutable tokenizer.
pub struct EmbeddingModel {
    pub gguf_path: PathBuf,
    pub vocab: Arc<crate::inference::models::bert::BertVocab>,
    /// WordPiece tokenizer using the GGUF BERT convention (`▁`-prefixed word
    /// starters). Shared across explicit BERT-family architectures.
    pub tokenizer: Arc<crate::inference::models::bert::BertWpmTokenizer>,
    /// Generation identity. Startup uses the configured artifact identity;
    /// public activation replaces it with the exact catalog-bound selection.
    pub model_id: String,
    /// Registry compiled for exactly this model generation. Keeping it inside
    /// the model closes the former independently-swappable model/registry pair.
    pub registry: Arc<std::sync::Mutex<mlx_native::KernelRegistry>>,
    /// Load-phase timing captured by the production loader. This stays with
    /// the generation it describes, so a swap receipt cannot accidentally
    /// report timings from the previous model.
    pub load_timing: EmbeddingLoadTiming,
    /// Architecture variant. Carries the per-arch config + weights so
    /// the handler dispatches the correct forward pass. Optional only
    /// in the test-scaffolding path that bypasses real weight loading;
    /// production always populates this via `cmd_serve`.
    pub arch: Option<EmbeddingArch>,
}

/// Production embedding admission timing. `weight_load_elapsed` includes
/// header/config parsing plus mapped native-weight construction;
/// `registry_warm_elapsed` is the first forward used to compile the exact
/// generation's kernels; `total_elapsed` covers both plus bookkeeping.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmbeddingLoadTiming {
    pub weight_load_elapsed: std::time::Duration,
    pub registry_warm_elapsed: std::time::Duration,
    pub total_elapsed: std::time::Duration,
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

    pub fn storage_stats(
        &self,
    ) -> crate::inference::models::bert::native_storage::NativeStorageStats {
        match self {
            Self::Bert { weights, .. } => weights.storage_stats(),
            Self::NomicBert { weights, .. } => weights.storage_stats(),
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
    /// Tokenize one input from this generation's embedded vocabulary. Pass
    /// `add_special_tokens=true` to wrap it in `[CLS] ... [SEP]`.
    pub fn encode(&self, input: &str, add_special_tokens: bool) -> Vec<u32> {
        self.tokenizer.encode(input, add_special_tokens)
    }

    pub fn resident_bytes(&self) -> u64 {
        self.arch
            .as_ref()
            .map(EmbeddingArch::storage_stats)
            .map(|stats| stats.resident_bytes)
            .unwrap_or(0)
    }
}

#[cfg(test)]
pub(crate) fn load_synthetic_native_embedding_model(
    path: &std::path::Path,
    model_id: &str,
    hello_id: usize,
) -> anyhow::Result<EmbeddingModel> {
    use crate::inference::models::bert::bert_gpu::{
        apply_bert_full_forward_gpu, register_bert_custom_shaders,
    };
    use crate::inference::models::bert::native_storage::test_support::bert_cfg;
    use crate::inference::models::bert::{BertSpecialTokens, BertVocab};
    use mlx_native::{DType, KernelRegistry, MlxDevice};

    let config = bert_cfg(6);
    let weights = Arc::new(LoadedBertWeights::load_from_path(path, &config)?);
    let mut tokens = vec![
        "[UNK]".into(),
        "[CLS]".into(),
        "[SEP]".into(),
        "[PAD]".into(),
        "unused-a".into(),
        "unused-b".into(),
    ];
    tokens[hello_id] = "\u{2581}hello".into();
    let vocab = BertVocab {
        tokens,
        specials: BertSpecialTokens {
            cls: 1,
            sep: 2,
            pad: 3,
            unk: 0,
            mask: 0,
        },
    };
    let tokenizer = Arc::new(crate::inference::models::bert::BertWpmTokenizer::new(
        &vocab,
    ));

    // Match production admission: the registry belongs to this generation
    // and is warmed before the model becomes acquirable.
    let device = MlxDevice::new().map_err(|error| anyhow::anyhow!("test device: {error}"))?;
    let mut registry = KernelRegistry::new();
    register_bert_custom_shaders(&mut registry);
    let mut ids = device
        .alloc_buffer(32 * 4, DType::U32, vec![32])
        .map_err(|error| anyhow::anyhow!("test ids: {error}"))?;
    ids.as_mut_slice::<u32>()?.fill(3);
    ids.as_mut_slice::<u32>()?[..3].copy_from_slice(&[1, hello_id as u32, 2]);
    let mut encoder = device.command_encoder()?;
    let _ = apply_bert_full_forward_gpu(
        &mut encoder,
        &mut registry,
        &device,
        &ids,
        None,
        &weights,
        &config,
        32,
        3,
    )?;
    encoder.commit_and_wait()?;

    Ok(EmbeddingModel {
        gguf_path: path.to_path_buf(),
        vocab: Arc::new(vocab),
        tokenizer,
        model_id: model_id.into(),
        registry: Arc::new(std::sync::Mutex::new(registry)),
        load_timing: EmbeddingLoadTiming::default(),
        arch: Some(EmbeddingArch::Bert { config, weights }),
    })
}

/// Request-bound view of one dedicated embedding generation.
pub struct EmbeddingModelLease {
    pub model: Arc<EmbeddingModel>,
    pub generation: u64,
}

/// Successful no-double-residency replacement receipt.
#[derive(Debug, Clone)]
pub struct EmbeddingSwapReceipt {
    pub generation: u64,
    pub old_model_id: Option<String>,
    pub new_model_id: String,
    pub new_arch: Option<&'static str>,
    pub load_timing: EmbeddingLoadTiming,
    pub reclaimed_bytes: u64,
    pub resident_bytes: u64,
    pub unload_elapsed: std::time::Duration,
    pub load_elapsed: std::time::Duration,
}

/// Read-only lifecycle view used by the authenticated runtime and activation
/// endpoints.  Model-local tokenizer and registry handles never escape this
/// snapshot; they remain owned by the same [`EmbeddingModel`] generation.
#[derive(Debug, Clone)]
pub struct EmbeddingModelSnapshot {
    /// `None` only while the replacement lock is held; the successful or
    /// failed transaction publishes the exact generation immediately after.
    pub generation: Option<u64>,
    pub configured: bool,
    pub loading: bool,
    pub model_id: Option<String>,
    pub gguf_path: Option<PathBuf>,
    pub arch: Option<&'static str>,
    pub resident_bytes: u64,
    pub last_load_error: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingSwapError {
    #[error("embedding generation changed: expected {expected}, current generation is {current}")]
    StaleGeneration { expected: u64, current: u64 },
    #[error(
        "embedding model '{model_id}' has {active_leases} active request lease(s); refusing a double-resident swap"
    )]
    ActiveLeases {
        model_id: String,
        active_leases: usize,
    },
    #[error("embedding replacement load failed: {0}")]
    Load(#[source] anyhow::Error),
}

/// Single-slot lifecycle for a dedicated BERT-family encoder.
///
/// The next loader is called only after the prior model has been uniquely
/// owned and dropped. This deliberately trades temporary unavailability on a
/// failed load for bounded unified-memory use during model swaps.
#[derive(Default)]
pub struct EmbeddingModelSlot {
    active: Option<Arc<EmbeddingModel>>,
    generation: u64,
    configured: bool,
    last_load_error: Option<String>,
}

impl EmbeddingModelSlot {
    pub fn acquire(&self) -> Option<EmbeddingModelLease> {
        self.active.as_ref().map(|model| EmbeddingModelLease {
            model: Arc::clone(model),
            generation: self.generation,
        })
    }

    pub fn install_initial(&mut self, model: EmbeddingModel) -> anyhow::Result<u64> {
        anyhow::ensure!(
            self.active.is_none(),
            "embedding slot already has an active model"
        );
        self.generation = self.generation.saturating_add(1);
        self.configured = true;
        self.last_load_error = None;
        self.active = Some(Arc::new(model));
        Ok(self.generation)
    }

    pub fn was_configured(&self) -> bool {
        self.configured
    }

    pub fn last_load_error(&self) -> Option<&str> {
        self.last_load_error.as_deref()
    }

    pub fn logical_resident_bytes(&self) -> u64 {
        self.active
            .as_ref()
            .map(|model| model.resident_bytes())
            .unwrap_or(0)
    }

    pub fn snapshot(&self) -> EmbeddingModelSnapshot {
        EmbeddingModelSnapshot {
            generation: Some(self.generation),
            configured: self.configured,
            loading: false,
            model_id: self.active.as_ref().map(|model| model.model_id.clone()),
            gguf_path: self.active.as_ref().map(|model| model.gguf_path.clone()),
            arch: self
                .active
                .as_ref()
                .and_then(|model| model.arch.as_ref())
                .map(EmbeddingArch::arch_name),
            resident_bytes: self.logical_resident_bytes(),
            last_load_error: self.last_load_error.clone(),
        }
    }

    pub fn try_replace_after_evict<F>(&mut self, loader: F) -> anyhow::Result<EmbeddingSwapReceipt>
    where
        F: FnOnce() -> anyhow::Result<EmbeddingModel>,
    {
        self.try_replace_after_evict_at_generation(self.generation, loader)
            .map_err(anyhow::Error::new)
    }

    pub fn try_replace_after_evict_at_generation<F>(
        &mut self,
        expected_generation: u64,
        loader: F,
    ) -> Result<EmbeddingSwapReceipt, EmbeddingSwapError>
    where
        F: FnOnce() -> anyhow::Result<EmbeddingModel>,
    {
        if self.generation != expected_generation {
            return Err(EmbeddingSwapError::StaleGeneration {
                expected: expected_generation,
                current: self.generation,
            });
        }
        if let Some(active) = self.active.as_ref() {
            let owners = Arc::strong_count(active);
            if owners != 1 {
                return Err(EmbeddingSwapError::ActiveLeases {
                    model_id: active.model_id.clone(),
                    active_leases: owners - 1,
                });
            }
        }

        self.configured = true;
        self.last_load_error = None;
        let old = self.active.take();
        let old_model_id = old.as_ref().map(|model| model.model_id.clone());
        let reclaimed_bytes = old
            .as_ref()
            .map(|model| model.resident_bytes())
            .unwrap_or(0);
        let unload_started = Instant::now();
        drop(old);
        let unload_elapsed = unload_started.elapsed();

        let load_started = Instant::now();
        let next = match loader() {
            Ok(next) => next,
            Err(error) => {
                self.last_load_error = Some(error.to_string());
                return Err(EmbeddingSwapError::Load(error));
            }
        };
        let load_elapsed = load_started.elapsed();
        let new_model_id = next.model_id.clone();
        let new_arch = next.arch.as_ref().map(EmbeddingArch::arch_name);
        let load_timing = next.load_timing;
        let resident_bytes = next.resident_bytes();
        self.generation = self.generation.saturating_add(1);
        self.active = Some(Arc::new(next));
        Ok(EmbeddingSwapReceipt {
            generation: self.generation,
            old_model_id,
            new_model_id,
            new_arch,
            load_timing,
            reclaimed_bytes,
            resident_bytes,
            unload_elapsed,
            load_elapsed,
        })
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
    pub artifact_sha256: String,
    pub source_sha256: Option<String>,
    /// Bounded projector-local embedding cache and single-flight compute
    /// gate. The payload is Arc-owned so hits do not duplicate large tensors.
    pub vision_cache: Arc<crate::inference::vision::pipeline::VisionEmbeddingCache>,
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
        let cache = match config.cache_dir.as_ref() {
            Some(root) => ModelCache::open_at(root)?,
            None => ModelCache::open()?,
        };
        let pool = LoadedPool::from_hardware(&hardware);
        // ADR-005 Phase 4 reopen iter-213 (AC 5472): KV-spill counters
        // are owned by AppState and Arc-cloned into HotSwapManager so the
        // trigger sites bump the same table the /metrics handler reads.
        let kv_spill_counters = Arc::new(KvSpillCounters::new());
        let mut manager = HotSwapManager::new(pool, Arc::new(DefaultModelLoader));
        manager.set_kv_counters(Arc::clone(&kv_spill_counters));
        let engine_config_template = EngineConfig {
            tokenizer_path: None,
            config_path: None,
            queue_capacity: engine_queue_capacity,
            warmup_synchronously: true,
            kv_metrics_sink: Some(Arc::clone(&kv_spill_counters)
                as Arc<dyn crate::serve::kv_persist::metrics::KvCacheMetricsSink>),
            dwq_overlay_path: None,
            engine_mode: super::engine::EngineMode::SerialFifo,
            kv_cache_budget_bytes: None,
        };
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
            model_lifecycle: Arc::new(super::lifecycle::ModelLifecycleCoordinator::default()),
            preparations: super::cancellation::PreparationSupervisor::default(),
            artifact_catalog: super::artifact_catalog::ArtifactCatalogCoordinator::default(),
            local_artifacts: super::local_artifacts::LocalArtifactInventory::default(),
            cache: Arc::new(std::sync::Mutex::new(cache)),
            hardware: Arc::new(hardware),
            no_integrity,
            engine_queue_capacity,
            engine_config_template,
            engine_config_overrides: Arc::new(std::sync::RwLock::new(HashMap::new())),
            default_model,
            embedding_slot: Arc::new(std::sync::RwLock::new(EmbeddingModelSlot::default())),
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
        let engine_config_template = EngineConfig {
            tokenizer_path: None,
            config_path: None,
            queue_capacity: 32,
            warmup_synchronously: true,
            kv_metrics_sink: Some(Arc::clone(&kv_spill_counters_test)
                as Arc<dyn crate::serve::kv_persist::metrics::KvCacheMetricsSink>),
            dwq_overlay_path: None,
            engine_mode: super::engine::EngineMode::SerialFifo,
            kv_cache_budget_bytes: None,
        };
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
            model_lifecycle: Arc::new(super::lifecycle::ModelLifecycleCoordinator::default()),
            preparations: super::cancellation::PreparationSupervisor::default(),
            artifact_catalog: super::artifact_catalog::ArtifactCatalogCoordinator::default(),
            local_artifacts: super::local_artifacts::LocalArtifactInventory::default(),
            cache: Arc::new(std::sync::Mutex::new(cache)),
            hardware: Arc::new(hardware),
            no_integrity: false,
            engine_queue_capacity: 32,
            engine_config_template,
            engine_config_overrides: Arc::new(std::sync::RwLock::new(HashMap::new())),
            default_model: None,
            embedding_slot: Arc::new(std::sync::RwLock::new(EmbeddingModelSlot::default())),
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

    /// Replace process-wide dynamic-load defaults after CLI scheduler and KV
    /// policy resolution. Model-local sidecars/overlays must remain `None` in
    /// this template and are registered separately by exact artifact path.
    pub fn with_engine_config_template(mut self, config: EngineConfig) -> Self {
        assert!(
            config.tokenizer_path.is_none()
                && config.config_path.is_none()
                && config.dwq_overlay_path.is_none(),
            "process-wide engine template cannot contain model-local sidecar or overlay paths"
        );
        self.engine_queue_capacity = config.queue_capacity;
        self.engine_config_template = config;
        self
    }

    /// Bind startup-only sidecars/overlay to the exact physical artifact.
    pub fn register_engine_config_for_path(
        &self,
        path: &std::path::Path,
        config: EngineConfig,
    ) -> anyhow::Result<()> {
        let key = std::fs::canonicalize(path).map_err(|error| {
            anyhow::anyhow!(
                "cannot canonicalize engine-config artifact {}: {error}",
                path.display()
            )
        })?;
        self.engine_config_overrides
            .write()
            .map_err(|_| anyhow::anyhow!("engine-config registry poisoned"))?
            .insert(key, config);
        Ok(())
    }

    /// Resolve the exact configuration for one physical artifact. Unknown
    /// artifacts inherit only the process-wide template; a previously loaded
    /// startup artifact recovers its model-local sidecars and overlay.
    pub fn engine_config_for_path(&self, path: &std::path::Path) -> anyhow::Result<EngineConfig> {
        let key = std::fs::canonicalize(path).map_err(|error| {
            anyhow::anyhow!(
                "cannot canonicalize engine-config artifact {}: {error}",
                path.display()
            )
        })?;
        Ok(self
            .engine_config_overrides
            .read()
            .map_err(|_| anyhow::anyhow!("engine-config registry poisoned"))?
            .get(&key)
            .cloned()
            .unwrap_or_else(|| self.engine_config_template.clone()))
    }

    pub fn with_local_artifacts(
        mut self,
        local_artifacts: super::local_artifacts::LocalArtifactInventory,
    ) -> Self {
        self.local_artifacts = local_artifacts;
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
            crate::serve::kv_persist::BlockPrefixCacheSpiller<crate::serve::api::engine::Engine>,
        >,
    ) -> Self {
        self.kv_spiller = Some(spiller);
        self
    }

    /// Attach a BERT embedding model config. Cheap (clones internal
    /// references). Called by `cmd_serve` after validating the supplied
    /// GGUF header.
    pub fn with_embedding_model(self, em: EmbeddingModel) -> Self {
        self.embedding_slot
            .write()
            .expect("embedding slot poisoned during construction")
            .install_initial(em)
            .expect("embedding model installed twice during construction");
        self
    }

    pub fn acquire_embedding_model(&self) -> Option<EmbeddingModelLease> {
        // A public activation holds the write lock while it drops the old
        // generation and warms the replacement.  Request handlers must fail
        // closed during that interval rather than block an async worker on a
        // model load.
        self.embedding_slot.try_read().ok()?.acquire()
    }

    pub fn embedding_model_was_configured(&self) -> bool {
        self.embedding_slot
            .try_read()
            .map(|slot| slot.was_configured())
            // A contended lock means a configured replacement is in flight.
            // Treat it as unavailable dedicated state, never as permission to
            // fall back to a chat-model embedding path.
            .unwrap_or(true)
    }

    pub fn embedding_model_snapshot(&self) -> EmbeddingModelSnapshot {
        match self.embedding_slot.try_read() {
            Ok(slot) => slot.snapshot(),
            Err(_) => EmbeddingModelSnapshot {
                generation: None,
                configured: true,
                loading: true,
                model_id: None,
                gguf_path: None,
                arch: None,
                resident_bytes: 0,
                last_load_error: None,
            },
        }
    }

    pub fn try_swap_embedding_model<F>(&self, loader: F) -> anyhow::Result<EmbeddingSwapReceipt>
    where
        F: FnOnce() -> anyhow::Result<EmbeddingModel>,
    {
        self.embedding_slot
            .write()
            .map_err(|_| anyhow::anyhow!("embedding lifecycle lock poisoned"))?
            .try_replace_after_evict(loader)
    }

    pub fn try_swap_embedding_model_at_generation<F>(
        &self,
        expected_generation: u64,
        loader: F,
    ) -> Result<EmbeddingSwapReceipt, EmbeddingSwapError>
    where
        F: FnOnce() -> anyhow::Result<EmbeddingModel>,
    {
        self.embedding_slot
            .write()
            .map_err(|_| {
                EmbeddingSwapError::Load(anyhow::anyhow!("embedding lifecycle lock poisoned"))
            })?
            .try_replace_after_evict_at_generation(expected_generation, loader)
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
        if !self.ready_for_gen.load(Ordering::Acquire) {
            return false;
        }
        self.pool
            .try_read()
            .map(|pool| {
                pool.snapshot_engines()
                    .into_iter()
                    .all(|loaded| loaded.engine.is_worker_healthy())
            })
            .unwrap_or(false)
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
    fn readiness_fails_fast_while_model_pool_is_write_locked() {
        let state = AppState::new(ServerConfig::default());
        let _pool_write = state.pool.write().expect("lock synthetic model pool");
        assert!(
            !state.is_ready_for_gen(),
            "pool mutation must report not-ready instead of blocking readiness"
        );
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
        // Synthetic vocab using the peer's BERT-WPM convention:
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
            tokenizer: Arc::new(crate::inference::models::bert::BertWpmTokenizer::new(
                &vocab,
            )),
            model_id: "synthetic-embed".into(),
            registry: Arc::new(std::sync::Mutex::new(mlx_native::KernelRegistry::new())),
            load_timing: EmbeddingLoadTiming::default(),
            arch: None,
        };
        let _ = tokenizer; // legacy HF tokenizer no longer used; kept for shape only
        let ids = em.encode("hello world", false);
        assert!(ids.contains(&4), "expected 'hello'=4 in {:?}", ids);
        assert!(ids.contains(&5), "expected 'world'=5 in {:?}", ids);
    }

    fn synthetic_embedding_model(model_id: &str, hello_id: usize) -> EmbeddingModel {
        use crate::inference::models::bert::{BertSpecialTokens, BertVocab};
        let mut tokens = vec![
            "[UNK]".into(),
            "[CLS]".into(),
            "[SEP]".into(),
            "[PAD]".into(),
            "unused-a".into(),
            "unused-b".into(),
        ];
        tokens[hello_id] = "\u{2581}hello".into();
        let vocab = BertVocab {
            tokens,
            specials: BertSpecialTokens {
                cls: 1,
                sep: 2,
                pad: 3,
                unk: 0,
                mask: 0,
            },
        };
        EmbeddingModel {
            gguf_path: format!("/tmp/{model_id}.gguf").into(),
            tokenizer: Arc::new(crate::inference::models::bert::BertWpmTokenizer::new(
                &vocab,
            )),
            vocab: Arc::new(vocab),
            model_id: model_id.into(),
            registry: Arc::new(std::sync::Mutex::new(mlx_native::KernelRegistry::new())),
            load_timing: EmbeddingLoadTiming::default(),
            arch: None,
        }
    }

    #[test]
    fn embedding_slot_a_b_a_is_isolated_and_never_double_resident() {
        let mut slot = EmbeddingModelSlot::default();
        assert_eq!(
            slot.install_initial(synthetic_embedding_model("a-1", 4))
                .expect("install A"),
            1
        );
        let lease = slot.acquire().expect("A lease");
        assert_eq!(lease.generation, 1);
        assert_eq!(lease.model.encode("hello", false), vec![4]);
        let a_registry = Arc::downgrade(&lease.model.registry);
        let attempted = Arc::new(AtomicBool::new(false));
        let attempted_in_loader = Arc::clone(&attempted);
        let error = slot
            .try_replace_after_evict(move || {
                attempted_in_loader.store(true, Ordering::SeqCst);
                Ok(synthetic_embedding_model("b", 5))
            })
            .expect_err("live lease must make swap busy");
        assert!(error.to_string().contains("1 active request lease"));
        assert!(
            !attempted.load(Ordering::SeqCst),
            "busy swap must not load B"
        );

        let weak_a = Arc::downgrade(&lease.model);
        drop(lease);
        let receipt_b = slot
            .try_replace_after_evict(|| {
                assert!(
                    weak_a.upgrade().is_none(),
                    "A must be dropped before B loads"
                );
                Ok(synthetic_embedding_model("b", 5))
            })
            .expect("swap A to B");
        assert_eq!(receipt_b.generation, 2);
        assert_eq!(receipt_b.old_model_id.as_deref(), Some("a-1"));
        assert_eq!(receipt_b.new_model_id, "b");
        let lease_b = slot.acquire().expect("B lease");
        assert_eq!(lease_b.model.encode("hello", false), vec![5]);
        assert!(a_registry.upgrade().is_none());

        let weak_b = Arc::downgrade(&lease_b.model);
        drop(lease_b);
        let receipt_a2 = slot
            .try_replace_after_evict(|| {
                assert!(
                    weak_b.upgrade().is_none(),
                    "B must be dropped before A reloads"
                );
                Ok(synthetic_embedding_model("a-2", 4))
            })
            .expect("swap B to fresh A");
        assert_eq!(receipt_a2.generation, 3);
        let lease_a2 = slot.acquire().expect("fresh A lease");
        assert_eq!(lease_a2.model.encode("hello", false), vec![4]);
        assert!(a_registry.upgrade().is_none());
        assert_eq!(slot.logical_resident_bytes(), 0);
    }

    #[test]
    fn stale_embedding_generation_never_evicts_or_calls_loader() {
        let mut slot = EmbeddingModelSlot::default();
        slot.install_initial(synthetic_embedding_model("a", 4))
            .expect("install A");
        let attempted = Arc::new(AtomicBool::new(false));
        let attempted_in_loader = Arc::clone(&attempted);
        let error = slot
            .try_replace_after_evict_at_generation(0, move || {
                attempted_in_loader.store(true, Ordering::SeqCst);
                Ok(synthetic_embedding_model("b", 5))
            })
            .expect_err("stale generation must fail closed");
        assert!(matches!(
            error,
            EmbeddingSwapError::StaleGeneration {
                expected: 0,
                current: 1
            }
        ));
        assert!(!attempted.load(Ordering::SeqCst));
        let lease = slot.acquire().expect("A remains resident");
        assert_eq!(lease.generation, 1);
        assert_eq!(lease.model.model_id, "a");
    }

    #[test]
    fn failed_embedding_replacement_stays_configured_and_unavailable() {
        let mut slot = EmbeddingModelSlot::default();
        slot.install_initial(synthetic_embedding_model("a", 4))
            .expect("install A");
        let weak_a = Arc::downgrade(&slot.acquire().expect("A lease").model);
        // The temporary lease above is dropped at the end of the statement,
        // leaving the slot as the sole owner before replacement.
        let error = slot
            .try_replace_after_evict(|| {
                assert!(
                    weak_a.upgrade().is_none(),
                    "A must be gone before loading B"
                );
                Err(anyhow::anyhow!("synthetic B load failed"))
            })
            .expect_err("replacement must surface loader failure");
        assert!(error.to_string().contains("synthetic B load failed"));
        assert!(slot.was_configured());
        assert!(
            slot.acquire().is_none(),
            "failed replacement leaves no stale model"
        );
        assert_eq!(slot.last_load_error(), Some("synthetic B load failed"));

        let receipt = slot
            .try_replace_after_evict(|| Ok(synthetic_embedding_model("b", 5)))
            .expect("a later explicit replacement may recover the slot");
        assert_eq!(receipt.generation, 2);
        assert_eq!(receipt.old_model_id, None);
        assert_eq!(slot.last_load_error(), None);
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
            image_min_pixels: None,
            image_max_pixels: None,
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
            artifact_sha256: "0".repeat(64),
            source_sha256: None,
            vision_cache: Arc::new(
                crate::inference::vision::pipeline::VisionEmbeddingCache::new(
                    crate::inference::vision::pipeline::DEFAULT_VISION_EMBEDDING_CACHE_BYTES,
                ),
            ),
        };
        let state = AppState::new(ServerConfig::default()).with_mmproj(m);
        let attached = state.mmproj.as_ref().expect("mmproj should be Some");
        assert_eq!(attached.model_id, "synthetic-mmproj");
        assert_eq!(
            attached.gguf_path.file_name().unwrap(),
            "synthetic-mmproj.gguf"
        );
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

    #[test]
    fn engine_config_registry_preserves_process_policy_without_leaking_model_sidecars() {
        let mut base = AppState::new(ServerConfig::default()).engine_config_template;
        base.engine_mode = crate::serve::api::engine::EngineMode::SlotAware { max_slots: 8 };
        base.kv_cache_budget_bytes = Some(4 * 1024 * 1024);
        let state =
            AppState::new(ServerConfig::default()).with_engine_config_template(base.clone());
        let model_a = tempfile::NamedTempFile::new().unwrap();
        let model_b = tempfile::NamedTempFile::new().unwrap();
        let mut model_a_config = base.clone();
        model_a_config.tokenizer_path = Some("/private/model-a-tokenizer.json".into());
        model_a_config.config_path = Some("/private/model-a-config.json".into());
        model_a_config.dwq_overlay_path = Some("/private/model-a-overlay.safetensors".into());
        state
            .register_engine_config_for_path(model_a.path(), model_a_config.clone())
            .unwrap();

        let first_a = state.engine_config_for_path(model_a.path()).unwrap();
        let unrelated_b = state.engine_config_for_path(model_b.path()).unwrap();
        let reloaded_a = state.engine_config_for_path(model_a.path()).unwrap();
        assert_eq!(
            crate::serve::multi_model::EngineConfigIdentity::from(&first_a),
            crate::serve::multi_model::EngineConfigIdentity::from(&reloaded_a)
        );
        assert_eq!(first_a.tokenizer_path, model_a_config.tokenizer_path);
        assert_eq!(first_a.config_path, model_a_config.config_path);
        assert_eq!(first_a.dwq_overlay_path, model_a_config.dwq_overlay_path);
        assert_eq!(unrelated_b.engine_mode, base.engine_mode);
        assert_eq!(
            unrelated_b.kv_cache_budget_bytes,
            base.kv_cache_budget_bytes
        );
        assert!(unrelated_b.tokenizer_path.is_none());
        assert!(unrelated_b.config_path.is_none());
        assert!(unrelated_b.dwq_overlay_path.is_none());
    }
}
