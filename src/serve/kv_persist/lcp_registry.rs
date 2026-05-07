//! ADR-017 Phase E option (a) — `LcpRegistry` standalone module.
//!
//! Per the research dossier at
//! `docs/research/adr017-phase-e-option-a-2026-05-05.md` §9 + §10, this
//! module ships the per-`(model_fingerprint, tenant, params)` registry
//! that maps cached prompts to per-layer KV-state Arc handles. It is
//! the load-bearing substrate for iter-3's high-risk
//! `forward_prefill.rs` LCP partial-prefill resume modification.
//!
//! ## What this module does
//!
//! - Stores `(LcpKey, prompt_token_ids, payload, sliding_window,
//!   linear_capacity)` tuples.
//! - On `lookup(key, new_prompt_token_ids)`:
//!   - returns `None` on key mismatch (different fingerprint, tenant, params)
//!   - returns `None` on full equality (`K == new_tokens.len()`) so
//!     Phase E option (b) PromptCache full-equality replay isn't masked
//!   - returns `None` on zero overlap (`K == 0`) so the caller's
//!     "no LCP" path detection is unambiguous
//!   - returns `Some(LcpPrefix { k, dense_kvs, ... })` on partial prefix
//!     match, where `dense_kvs` is a `Vec<Arc<T>>` clone (Arc-pinned
//!     handle that survives concurrent registry eviction — §10.3
//!     Strategy A; §6 R3).
//! - **Byte-budget LRU eviction** (iter E.a default-on): entries are
//!   evicted LRU-first until `current_bytes + new_bytes <= byte_budget`.
//!   The budget is computed from sysinfo `available_memory() × 5%`
//!   (clamped to `[1 GiB, 16 GiB]`) or overridden via
//!   `HF2Q_KV_LCP_RESUME_CAPACITY`. `lookup` promotes hit entries to MRU.
//!   Entry count is secondarily bounded by a `capacity` ceiling (default
//!   `usize::MAX` in byte-budget mode; set explicitly by `new(n)` shim).
//!
//! ## ByteSized trait
//!
//! The registry requires `T: ByteSized` to compute exact byte counts at
//! store time. Production payload types implement this via their existing
//! byte-reporting APIs (`MlxBuffer::byte_len` for `DenseKvBuffers`;
//! `HybridKvCacheSnapshot::total_bytes` for Qwen35). No estimators.
//!
//! ## Generic over the payload type `T`
//!
//! Production wires `T = DenseKvBuffers` (per dossier §10.3 Strategy A,
//! the registry stores `Vec<Arc<DenseKvBuffers>>` per entry — one
//! `Arc<DenseKvBuffers>` per layer). Unit tests at
//! `tests/lcp_registry_unit.rs` use `T = Vec<u8>` as a marker payload
//! since `MlxBuffer` requires a Metal device that unit tests don't have.
//! The registry is generic over `T: Send + Sync + 'static + ByteSized`.
//!
//! ## LRU implementation
//!
//! Mirrors the `LoadedPool` LRU pattern at `multi_model.rs:225-510`:
//! `Vec<LcpKey>` for `lru_order` + `HashMap<LcpKey, LcpEntry<T>>` for
//! `entries`. LRU = front, MRU = back. Per the project's "no new deps"
//! standing rule (`multi_model.rs:61`), we don't pull in the `lru`
//! crate.
//!
//! ## Concurrency model
//!
//! `LcpRegistry<T>` is **NOT** `Sync`-by-default — `&mut self` on
//! `store` and `lookup` (lookup needs &mut for LRU promotion). The
//! intended concurrency model is: hold a `Mutex<LcpRegistry<T>>` at
//! the engine-level call site (iter-2 wiring); take the lock briefly
//! to do the lookup, clone the Arc handles out, drop the lock, then
//! the in-flight prefill reads the Arc handles without further lock
//! contention. The Arc-pinning invariant (§6 R3) means a concurrent
//! registry eviction during prefill is safe — the in-flight Arc clones
//! keep the payload alive.

use std::collections::HashMap;
use std::sync::{Arc, Once};

use crate::serve::kv_persist::format::ModelFingerprint;

// ─────────────────────────────────────────────────────────────────────────────
// ByteSized trait
// ─────────────────────────────────────────────────────────────────────────────

/// Exact byte size of a registry payload at store time.
///
/// Implementors MUST return the true byte count from existing byte-reporting
/// APIs (e.g. `MlxBuffer::byte_len`, `HybridKvCacheSnapshot::total_bytes`).
/// No estimators. The registry uses this to enforce byte-budget eviction.
pub trait ByteSized {
    /// Returns the exact byte count of this value.
    fn byte_len(&self) -> u64;
}

// ─────────────────────────────────────────────────────────────────────────────
// Dynamic byte-budget probe
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the byte budget for `LcpRegistry` at engine-init time.
///
/// Priority order:
///
/// 1. **`HF2Q_KV_LCP_RESUME_CAPACITY` env override** — accepts:
///    - Bare integer with a suffix (`b`/`B` = bytes; `k`/`K` = ×1024;
///      `m`/`M` = ×1024²; `g`/`G` = ×1024³). E.g. `2g` = 2 GiB.
///    - Bare integer ≥ 4096 → treated as a raw byte count.
///    - Bare integer < 4096 → **legacy entry-count** (backward-compat):
///      converts via `n × 300 MB` heuristic (300 MB was the measured
///      per-entry cost on Qwen 3.6 35B-A3B at cap=8 time; see
///      `c04d5d2` motivation). Emits a one-time deprecation
///      `eprintln!` directing operators to the byte-suffix form.
///
/// 2. **sysinfo probe** (env unset): `available_memory() × 5%` clamped
///    to `[1 GiB, 16 GiB]`. The 5% / floor / ceiling triple ensures we
///    claim a defensible slice of free RAM (≈5 GB on a fresh 128 GB Mac,
///    ~16 Gemma-26B entries or ~16 Qwen35-35B entries) without OOM-
///    class budgets that cap=64 (19 GB) would cause.
///
///    If `available_memory()` returns 0 (shouldn't happen, but paranoia
///    rule: no magic number fallback), the floor (1 GiB) is returned
///    with a one-time warning.
///
/// Always emits one info-level `eprintln!` naming the chosen budget and
/// source on the first call (guards the invariant: no silent feature flip).
pub fn default_lcp_byte_budget() -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    const FLOOR: u64 = GIB;          // 1 GiB minimum
    const CEILING: u64 = 16 * GIB;  // 16 GiB maximum

    static LOG_ONCE: Once = Once::new();

    let env_val = std::env::var("HF2Q_KV_LCP_RESUME_CAPACITY").ok();

    if let Some(raw) = env_val {
        // Try to parse as integer + optional suffix.
        let trimmed = raw.trim();
        let (digits, suffix) = if let Some(last) = trimmed.chars().last() {
            match last {
                'b' | 'B' | 'k' | 'K' | 'm' | 'M' | 'g' | 'G' => {
                    (&trimmed[..trimmed.len() - 1], Some(last))
                }
                _ => (trimmed, None),
            }
        } else {
            (trimmed, None)
        };

        if let Ok(n) = digits.parse::<u64>() {
            let budget = if let Some(suffix_char) = suffix {
                // Suffix form: always byte-budget.
                let multiplier: u64 = match suffix_char {
                    'b' | 'B' => 1,
                    'k' | 'K' => 1024,
                    'm' | 'M' => 1024 * 1024,
                    'g' | 'G' => 1024 * 1024 * 1024,
                    _ => unreachable!("suffix already validated above"),
                };
                n.saturating_mul(multiplier)
            } else if n < 4096 {
                // Legacy entry-count path: n × 300 MB heuristic.
                let bytes = n.saturating_mul(300 * 1024 * 1024);
                // Emit deprecation warning exactly once.
                static LEGACY_ONCE: Once = Once::new();
                LEGACY_ONCE.call_once(|| {
                    eprintln!(
                        "[hf2q lcp] HF2Q_KV_LCP_RESUME_CAPACITY={n} interpreted as \
                         legacy entry-count (={bytes} bytes); use {n}g for byte-budget"
                    );
                });
                bytes
            } else {
                // Bare integer ≥ 4096: raw byte count.
                n
            };

            LOG_ONCE.call_once(|| {
                eprintln!(
                    "[hf2q lcp] byte_budget={} MB (source=env HF2Q_KV_LCP_RESUME_CAPACITY={})",
                    budget / (1024 * 1024),
                    raw.trim()
                );
            });
            return budget;
        }
        // Unparsable env value: fall through to sysinfo probe.
        static PARSE_WARN_ONCE: Once = Once::new();
        PARSE_WARN_ONCE.call_once(|| {
            eprintln!(
                "[hf2q lcp] WARNING: HF2Q_KV_LCP_RESUME_CAPACITY={:?} is not a \
                 parsable integer (with optional b/k/m/g suffix); falling back to \
                 sysinfo probe",
                raw
            );
        });
    }

    // sysinfo probe path.
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let avail = sys.available_memory();

    let budget = if avail == 0 {
        // Paranoia: sysinfo returned 0 — use floor and warn.
        static ZERO_AVAIL_ONCE: Once = Once::new();
        ZERO_AVAIL_ONCE.call_once(|| {
            eprintln!(
                "[hf2q lcp] WARNING: sysinfo available_memory() returned 0 bytes; \
                 using floor budget of 1 GiB"
            );
        });
        FLOOR
    } else {
        let computed = (avail as f64 * 0.05) as u64;
        computed.clamp(FLOOR, CEILING)
    };

    LOG_ONCE.call_once(|| {
        eprintln!(
            "[hf2q lcp] byte_budget={} MB (source=sysinfo available_memory={} GB × 5%)",
            budget / (1024 * 1024),
            avail / (1024 * 1024 * 1024)
        );
    });

    budget
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Composite key for `LcpRegistry` lookups. Three-axis isolation:
///
/// - `model_fingerprint`: KV state is bound to model weights + RoPE
///   config + dtype (per dossier §3 + §6 R5). Different model →
///   incompatible KV.
/// - `tenant_id`: multi-tenant isolation (§6 R7). Two requests with
///   identical token IDs from different tenants must not cross-hit.
///   Single-tenant deployments use a constant string (e.g. `"default"`).
/// - `params_hash`: opaque hash over sampling params (temperature,
///   top_p, top_k, seed, repetition_penalty, etc.). Different sampling
///   doesn't actually invalidate KV reuse for the prefill phase, BUT
///   sets that affect tokenization (e.g. chat-template variants) DO.
///   Conservative default: include all sampling params.
///
/// `Clone` so callers can build the key once + reuse for `store` then
/// `lookup`. `Hash + Eq` for the underlying `HashMap`.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct LcpKey {
    pub model_fingerprint: ModelFingerprint,
    pub tenant_id: String,
    pub params_hash: u64,
}

/// Result of a successful `LcpRegistry::lookup`.
///
/// `dense_kvs` is the per-layer Arc-pinned payload clone; the caller
/// (iter-3 forward_prefill) reads from this directly and the Arc keeps
/// the payload alive through any concurrent registry eviction (§6 R3).
#[derive(Clone, Debug)]
pub struct LcpPrefix<T> {
    /// Length of the longest common prefix (in tokens). Guaranteed:
    /// `0 < k < new_tokens.len()` and `k <= cached_prompt_len`.
    pub k: usize,
    /// Per-layer payload Arc clones. Length matches the cached
    /// payload's per-layer length (production: `num_layers`).
    pub dense_kvs: Vec<Arc<T>>,
    /// Cached entry's sliding-window size. Used by iter-3 to compute
    /// `min(k, sliding_window)` for the sliding-layer `seq_len`.
    pub sliding_window: usize,
    /// Cached entry's linear-layer capacity. Used by iter-3's three-
    /// case capacity protocol (§10.2): zero-copy if `cached_capacity
    /// >= new_seq_len + max_decode_tokens`; resize-and-copy otherwise.
    pub linear_capacity: usize,
    /// Cached entry's full prompt length (= `cached_prompt.len()`).
    /// Caller can assert `k <= cached_prompt_len` as a defensive
    /// invariant.
    pub cached_prompt_len: usize,
}

/// Errors from `LcpRegistry::store`. `lookup` returns `Option`, not
/// `Result`, because cache misses are normal (not errors).
#[derive(Debug, PartialEq, Eq)]
pub enum LcpStoreError {
    /// Empty prompt (`prompt_tokens.is_empty()`) has no LCP semantics.
    EmptyPrompt,
    /// Empty payload (`dense_kvs.is_empty()`) — every model has ≥1
    /// layer; an empty payload is a misconfiguration.
    EmptyPayload,
    /// The entry's byte size exceeds the entire byte budget. Even after
    /// evicting all existing entries, this entry cannot fit. The caller
    /// must decide whether to skip or abort — silently dropping is NOT
    /// an option (mantra: no fallback / no stub).
    EntryExceedsBudget {
        /// Byte size of the entry that could not be stored.
        entry_bytes: u64,
        /// The configured byte budget.
        budget_bytes: u64,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// LcpRegistry<T>
// ─────────────────────────────────────────────────────────────────────────────

/// One entry per `(model_fingerprint, tenant, params)` triple. The
/// caller's prompt is cached in full so subsequent lookups can compute
/// the LCP exactly.
struct LcpEntry<T> {
    /// The full prompt (token IDs) that produced this cached payload.
    /// Lookup compares the new prompt's prefix against this byte-for-
    /// byte (well, token-ID-for-token-ID — see dossier §2.2 for why
    /// token-ID granularity is mandatory).
    prompt: Vec<u32>,
    /// Per-layer Arc-wrapped payload. Length is the model's `num_layers`.
    dense_kvs: Vec<Arc<T>>,
    sliding_window: usize,
    linear_capacity: usize,
    /// Exact byte count of the payload, computed at store time via
    /// `T: ByteSized`. Used for byte-budget accounting: subtracted on
    /// eviction, added on successful insert.
    bytes: u64,
}

/// Generic over the cached payload type `T: Send + Sync + 'static + ByteSized`.
/// Production wires `T = DenseKvBuffers`; tests use `T = Vec<u8>`.
pub struct LcpRegistry<T>
where
    T: Send + Sync + 'static + ByteSized,
{
    /// Entry-count safety ceiling. In byte-budget mode this is
    /// `usize::MAX` (no entry-count limit beyond physical memory).
    /// In shim-`new(n)` mode this is `n` (preserves old test behavior).
    capacity: usize,
    /// Insertion-order vector tracking LRU → MRU. `lru_order[0]` is
    /// the LRU candidate; `lru_order.last()` is the MRU. `lookup` hits
    /// promote the entry to MRU.
    lru_order: Vec<LcpKey>,
    /// Keyed by `LcpKey`; storage for the actual cached state.
    entries: HashMap<LcpKey, LcpEntry<T>>,
    /// Maximum bytes the registry may hold. Evict LRU entries until
    /// `current_bytes + new_bytes <= byte_budget` before each new-key
    /// insert. Set to `u64::MAX` by the `new(n)` back-compat shim.
    byte_budget: u64,
    /// Running total of bytes currently stored across all entries.
    /// Invariant: `current_bytes == sum(entry.bytes for entry in entries)`.
    current_bytes: u64,
}

impl<T> LcpRegistry<T>
where
    T: Send + Sync + 'static + ByteSized,
{
    /// Construct a registry with a **byte-budget** eviction policy.
    ///
    /// Entries are evicted LRU-first until `current_bytes + new_bytes <=
    /// byte_budget` before each new-key insert. The entry-count ceiling is
    /// `usize::MAX` (effectively unbounded by count; memory is the limit).
    ///
    /// **Panics** when `byte_budget == 0` — a zero-budget registry would
    /// immediately evict every store, which is always a misconfiguration.
    pub fn with_byte_budget(byte_budget: u64) -> Self {
        assert!(
            byte_budget > 0,
            "LcpRegistry::with_byte_budget(0) is a misconfiguration — every \
             store would immediately evict itself; refusing to construct"
        );
        Self {
            capacity: usize::MAX,
            lru_order: Vec::new(),
            entries: HashMap::new(),
            byte_budget,
            current_bytes: 0,
        }
    }

    /// Back-compat shim: construct a registry with an **entry-count** cap,
    /// mirroring the original `new(capacity)` API.
    ///
    /// The byte budget is set to `u64::MAX` (effectively unbounded by bytes;
    /// entry count is the limit). All 5 existing call sites in unit test
    /// fixtures and the `new(1)` test helpers stay unchanged.
    ///
    /// **Panics** when `capacity == 0` — a zero-capacity registry would
    /// immediately evict every store, which is always a misconfiguration.
    /// Mirrors `LoadedPool` at `multi_model.rs:208-211`.
    pub fn new(capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "LcpRegistry::new(0) is a misconfiguration — every store would \
             immediately evict itself; refusing to construct"
        );
        Self {
            capacity,
            lru_order: Vec::with_capacity(capacity),
            entries: HashMap::with_capacity(capacity),
            byte_budget: u64::MAX,
            current_bytes: 0,
        }
    }

    /// Number of currently-cached entries (≤ `capacity`).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when the registry has zero entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total bytes currently stored across all entries.
    ///
    /// Invariant: equals `sum(entry.bytes for all entries)`. Updated on
    /// every store, eviction, and clear.
    pub fn current_bytes(&self) -> u64 {
        self.current_bytes
    }

    /// Configured byte budget. Eviction runs until `current_bytes +
    /// new_bytes <= byte_budget` before a new-key insert.
    pub fn byte_budget(&self) -> u64 {
        self.byte_budget
    }

    /// Store a prompt + payload tuple under `key`.
    ///
    /// **Byte-budget eviction**: for new-key inserts, evicts LRU entries
    /// until `current_bytes + new_bytes <= byte_budget`. If even after
    /// evicting all entries the single entry still exceeds the budget,
    /// returns `Err(LcpStoreError::EntryExceedsBudget)` — never silently
    /// no-ops.
    ///
    /// **Same-key reinsert**: subtracts the old entry's bytes, adds the
    /// new entry's bytes; no eviction of OTHER entries (total entry count
    /// is unchanged). Promotes to MRU.
    ///
    /// **Entry-count ceiling**: secondarily bounded by `capacity`; evicts
    /// LRU if `entries.len() >= capacity` (back-compat shim behavior).
    ///
    /// `prompt_tokens` is the full prompt as token-IDs (NOT the
    /// prompt + decoded-output concatenation; see dossier §10.6 R12).
    /// `dense_kvs` is the per-layer payload (one `Arc<T>` per layer,
    /// non-empty).
    ///
    /// # Errors
    ///
    /// - [`LcpStoreError::EmptyPrompt`] when `prompt_tokens.is_empty()`.
    /// - [`LcpStoreError::EmptyPayload`] when `dense_kvs.is_empty()`.
    /// - [`LcpStoreError::EntryExceedsBudget`] when `new_bytes > byte_budget`.
    pub fn store(
        &mut self,
        key: LcpKey,
        prompt_tokens: Vec<u32>,
        dense_kvs: Vec<Arc<T>>,
        sliding_window: usize,
        linear_capacity: usize,
    ) -> Result<(), LcpStoreError> {
        if prompt_tokens.is_empty() {
            return Err(LcpStoreError::EmptyPrompt);
        }
        if dense_kvs.is_empty() {
            return Err(LcpStoreError::EmptyPayload);
        }

        // Compute the exact byte count of the new payload via ByteSized.
        let new_bytes: u64 = dense_kvs.iter().map(|a| a.byte_len()).sum();

        // Same-key reinsert path: overwrite + promote to MRU.
        // Codex Phase-2b finding 2026-05-06 (HIGH): the same-key path must
        // honour EntryExceedsBudget identically to the new-key path. Letting
        // an oversize payload through here would leave current_bytes above
        // byte_budget and silently violate the store invariant the falsifier
        // (tests/lcp_registry_byte_budget_falsifier.rs) asserts after every
        // store. Mantra: no fallback, no silent-disable.
        if self.entries.contains_key(&key) {
            if new_bytes > self.byte_budget {
                return Err(LcpStoreError::EntryExceedsBudget {
                    entry_bytes: new_bytes,
                    budget_bytes: self.byte_budget,
                });
            }
            let old_bytes = self.entries[&key].bytes;
            self.current_bytes = self.current_bytes.saturating_sub(old_bytes);
            self.entries.insert(
                key.clone(),
                LcpEntry {
                    prompt: prompt_tokens,
                    dense_kvs,
                    sliding_window,
                    linear_capacity,
                    bytes: new_bytes,
                },
            );
            self.current_bytes += new_bytes;
            // Promote to MRU.
            if let Some(pos) = self.lru_order.iter().position(|k| k == &key) {
                let k = self.lru_order.remove(pos);
                self.lru_order.push(k);
            }
            return Ok(());
        }

        // New-key insert path.
        // Check: single entry larger than entire budget → hard error (no fallback).
        if new_bytes > self.byte_budget {
            return Err(LcpStoreError::EntryExceedsBudget {
                entry_bytes: new_bytes,
                budget_bytes: self.byte_budget,
            });
        }

        // Byte-budget eviction: evict LRU entries until within budget.
        while self.current_bytes + new_bytes > self.byte_budget && !self.lru_order.is_empty() {
            if let Some(victim) = self.lru_order.first().cloned() {
                if let Some(evicted) = self.entries.remove(&victim) {
                    self.current_bytes = self.current_bytes.saturating_sub(evicted.bytes);
                }
                self.lru_order.remove(0);
            }
        }

        // Entry-count ceiling eviction (back-compat shim: capacity < usize::MAX).
        while self.entries.len() >= self.capacity {
            if let Some(victim) = self.lru_order.first().cloned() {
                if let Some(evicted) = self.entries.remove(&victim) {
                    self.current_bytes = self.current_bytes.saturating_sub(evicted.bytes);
                }
                self.lru_order.remove(0);
            } else {
                // Defensive: lru_order out-of-sync with entries.
                break;
            }
        }

        self.entries.insert(
            key.clone(),
            LcpEntry {
                prompt: prompt_tokens,
                dense_kvs,
                sliding_window,
                linear_capacity,
                bytes: new_bytes,
            },
        );
        self.current_bytes += new_bytes;
        self.lru_order.push(key);

        // Q2 empirical sizing log: emit ONCE on the first successful new-key
        // insert so operators can verify the budget admits a reasonable
        // number of entries for their model + hardware.
        {
            static EMPIRICAL_ONCE: Once = Once::new();
            EMPIRICAL_ONCE.call_once(|| {
                let budget_mb = if self.byte_budget == u64::MAX {
                    // back-compat shim (entry-count mode): no meaningful budget to report
                    return;
                } else {
                    self.byte_budget / (1024 * 1024)
                };
                let entry_mb = new_bytes / (1024 * 1024);
                let admits = if new_bytes > 0 {
                    self.byte_budget / new_bytes
                } else {
                    0
                };
                eprintln!(
                    "[hf2q lcp] empirical: budget={budget_mb} MB / \
                     first_entry={entry_mb} MB → admits ≈{admits} entries"
                );
            });
        }

        Ok(())
    }

    /// Look up the longest common prefix between `new_tokens` and the
    /// cached prompt under `key`.
    ///
    /// Returns `Some(LcpPrefix)` iff:
    /// - An entry exists for `key`.
    /// - The token-ID prefix length `k` satisfies `0 < k < new_tokens.len()`
    ///   AND `k <= cached.prompt.len()`. (Strictly partial prefix; full
    ///   equality returns None so Phase E option (b) PromptCache replay
    ///   isn't masked. Zero overlap returns None so the caller's "no
    ///   LCP" path detection is unambiguous.)
    ///
    /// On a hit, the entry is promoted to MRU.
    ///
    /// **The returned `LcpPrefix.dense_kvs` is a `Vec<Arc<T>>` clone
    /// — `Arc::clone` per layer.** This means a concurrent
    /// `clear()` / `store()` that evicts this entry from the registry
    /// will NOT free the underlying payload until the caller drops
    /// the returned `Arc<T>` clones. This is the load-bearing safety
    /// guarantee for iter-3's in-flight prefill (§6 R3).
    pub fn lookup(&mut self, key: &LcpKey, new_tokens: &[u32]) -> Option<LcpPrefix<T>> {
        let entry = self.entries.get(key)?;

        // Compute the longest common token-ID prefix.
        let cached = &entry.prompt[..];
        let max_compare = cached.len().min(new_tokens.len());
        let mut k = 0usize;
        while k < max_compare && cached[k] == new_tokens[k] {
            k += 1;
        }

        // §2.3 line 172 — gate out 0-overlap and full-equality.
        if k == 0 || k == new_tokens.len() {
            return None;
        }

        // Build the result (Arc-clones per layer — the load-bearing
        // pinning invariant).
        let result = LcpPrefix {
            k,
            dense_kvs: entry.dense_kvs.iter().map(Arc::clone).collect(),
            sliding_window: entry.sliding_window,
            linear_capacity: entry.linear_capacity,
            cached_prompt_len: entry.prompt.len(),
        };

        // Promote to MRU.
        if let Some(pos) = self.lru_order.iter().position(|kk| kk == key) {
            let touched = self.lru_order.remove(pos);
            self.lru_order.push(touched);
        }

        Some(result)
    }

    /// ADR-017 Phase E option (a) iter-3 — **consuming variant of
    /// [`Self::lookup`]**. Returns the same `Option<LcpPrefix<T>>` as
    /// `lookup` AND removes the entry from the registry on hit so the
    /// caller becomes the sole holder of the per-layer Arc clones
    /// (post-take strong_count = 1, ignoring incidental clones).
    ///
    /// **Why this exists:** iter-3's partial-prefill resume mutates the
    /// cached `dense_kvs[*]` buffers in place during the new prefill
    /// (positions `[K..N')`). Sharing the Arc with the registry while
    /// the engine is mutating would corrupt future cache hits. The
    /// consuming take + post-prefill re-store dance gives the engine
    /// exclusive ownership during prefill and re-publishes a fresh
    /// snapshot at end-of-decode.
    ///
    /// **Distinction from lookup:** iter-2's observability probe MUST
    /// keep the entry in the registry (multiple sequential probes on
    /// the same key are valid and load-bearing for `/metrics`'s
    /// detection-rate gauge). Iter-3's actual restore-and-mutate path
    /// MUST consume. Two methods, two semantics.
    ///
    /// On miss (no entry, zero overlap, full equality, k==N) the
    /// registry is unchanged.
    pub fn take_prefix(&mut self, key: &LcpKey, new_tokens: &[u32]) -> Option<LcpPrefix<T>> {
        // Re-use lookup's hit logic. lookup also promotes to MRU on
        // hit; that's harmless because we're about to remove the entry
        // entirely.
        let prefix = self.lookup(key, new_tokens)?;
        // Remove from HashMap + lru_order. After this, the only Arc
        // clones outstanding are the ones we just handed back via
        // `prefix.dense_kvs`. Subtract the entry's bytes from
        // current_bytes to keep the accounting invariant exact.
        if let Some(removed) = self.entries.remove(key) {
            self.current_bytes = self.current_bytes.saturating_sub(removed.bytes);
        }
        if let Some(pos) = self.lru_order.iter().position(|kk| kk == key) {
            self.lru_order.remove(pos);
        }
        Some(prefix)
    }

    /// Drop all entries. Used by tests and by future iter-2 wiring on
    /// model reload (when the registry's cached payloads become
    /// invalid — e.g. weights file changed). The underlying payloads
    /// remain alive as long as in-flight callers hold their `Arc<T>`
    /// clones — see §6 R3. Resets `current_bytes` to 0.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_order.clear();
        self.current_bytes = 0;
    }

    /// Configured entry-count safety ceiling. In byte-budget mode this is
    /// `usize::MAX`. In shim-`new(n)` mode this is `n`. Diagnostic
    /// accessor; the registry enforces the bound automatically on `store`.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> std::fmt::Debug for LcpRegistry<T>
where
    T: Send + Sync + 'static + ByteSized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LcpRegistry")
            .field("capacity", &self.capacity)
            .field("byte_budget", &self.byte_budget)
            .field("current_bytes", &self.current_bytes)
            .field("len", &self.entries.len())
            .field("lru_order_len", &self.lru_order.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// iter-2 — engine-side detection helper.
// ---------------------------------------------------------------------------

/// ADR-017 Phase E.a iter-2 — engine-side LCP detection probe.
///
/// Given a registry, key, new prompt tokens, and a multimodal flag,
/// returns `Some(K)` where `0 < K < new_tokens.len()` if a partial-
/// prefix resume opportunity exists, else `None`. Used by
/// `engine.rs::generate_once_with_soft_tokens` immediately AFTER the
/// `PromptCache` full-equality miss (so E.b's win path is preserved)
/// and BEFORE `forward_prefill_with_soft_tokens` so the iter-3
/// partial-prefill modification has a single, named decision point.
///
/// ## Multimodal gate (dossier §10.5)
///
/// `has_soft_tokens == true` ⇒ always `None`. The cached KV state was
/// generated under text-only embedding-lookup; replaying its prefix
/// into a request whose first-N tokens have per-position soft-token
/// overrides (vision / audio / deepstack) would silently corrupt
/// outputs because the cached `[0..K)` KV state would not reflect the
/// new request's per-position soft-token deltas. The bail is a hard
/// precondition, not a hint.
///
/// ## Why a separate function (vs a method on `LcpRegistry`)
///
/// 1. **Engine call site has the multimodal flag, registry doesn't.**
///    `LcpRegistry` is payload-agnostic; threading `has_soft_tokens`
///    into `lookup` would couple the registry to the engine's
///    multimodal scope. Keeping the gate in a free function lets the
///    registry stay generic.
/// 2. **Iter-2 ships an observability layer.** The probe returns
///    `Option<usize>` (just K) — it never hands out the full
///    `LcpPrefix<T>` payload, because iter-2 doesn't yet pre-warm
///    `dense_kvs`. Iter-3 will swap callers to `LcpRegistry::lookup`
///    directly to get the Arc-cloned payload.
///
/// Returns the LCP length `K` (1-indexed token count) on hit, `None`
/// on any of: zero overlap (`K == 0`), full equality (`K == N`),
/// fingerprint/tenant/params mismatch, or `has_soft_tokens == true`.
///
/// **Why `&mut`** — `LcpRegistry::lookup` is `&mut self` (it promotes
/// the hit entry to MRU on each access). The probe inherits that.
/// Engine sites already hold `loaded.lcp_registry` mutably (the worker
/// thread is the sole owner per the `LoadedModel` ownership model
/// mirrored from `prompt_cache`).
pub fn probe_lcp_opportunity<T>(
    registry: &mut LcpRegistry<T>,
    key: &LcpKey,
    new_tokens: &[u32],
    has_soft_tokens: bool,
) -> Option<usize>
where
    T: Send + Sync + 'static + ByteSized,
{
    if has_soft_tokens {
        // Multimodal request — never probe LCP. The cached KV state
        // was text-only; per-position soft-token overrides would
        // diverge under partial-prefill resume.
        return None;
    }
    registry.lookup(key, new_tokens).map(|prefix| prefix.k)
}
