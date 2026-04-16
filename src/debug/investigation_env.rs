//! Central loader for hf2q's investigation-only environment variables.
//!
//! Every category-4 env var in `docs/shipping-contract.md` is parsed here
//! exactly once, at first access to [`INVESTIGATION_ENV`]. Hot-path code
//! reads fields off the cached struct instead of calling `std::env::var`
//! directly — so the decode loop does zero env lookups after init.
//!
//! This module intentionally does not perform the S-4 startup warning or
//! the `HF2Q_UNSAFE_EXPERIMENTS=1` acknowledgment gate yet; those land as
//! a separate story and will fill in [`InvestigationEnv::activate`].
//!
//! **Not in scope here:**
//! - `HF2Q_LMHEAD_Q8` is a category-2 operator knob (user-facing,
//!   documented in `docs/operator-env-vars.md`) and is read at load time
//!   inside the lm_head init path. It does not belong in this struct.
//!
//! Classification rule (see shipping contract): a toggle requires the
//! unsafe-ack only when it is **known to risk correctness or runtime
//! reliability**, not merely experimental or inert.
//!
//! Parse semantics are preserved *exactly* from the original inline
//! reads — `is_ok()` vs `map_or(false, |v| v == "1")` are distinct
//! signals in the existing code (`is_ok()` means "set to anything,
//! including empty"; `== "1"` means strictly the literal "1"), and each
//! field below documents which shape the call sites expect.
//!
//! When adding a new investigation env var, wire it here and register
//! its classification in `docs/shipping-contract.md`.

use std::env;
use std::sync::LazyLock;

/// Process-wide cache of investigation-only environment variables.
///
/// Parses on first access, then returns cached values as simple field
/// reads. Access via `crate::debug::INVESTIGATION_ENV.<field>`.
pub static INVESTIGATION_ENV: LazyLock<InvestigationEnv> =
    LazyLock::new(InvestigationEnv::from_env);

/// Parsed snapshot of every investigation-only env var at the point of
/// first access. Each field's doc comment notes its category and its
/// exact original parse semantics.
#[derive(Debug, Clone)]
pub struct InvestigationEnv {
    // ========================================================================
    // Category 4 — ack-required (known to risk correctness or reliability).
    // S-4 will refuse to activate these unless HF2Q_UNSAFE_EXPERIMENTS=1.
    // ========================================================================
    /// `HF2Q_F16_KV=1` — allocate dense KV cache as F16. Known-worse
    /// output vs the F32 default; documented in ADR-009.
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub f16_kv: bool,

    /// `HF2Q_BATCHED_PREFILL=1` — use the experimental batched prefill
    /// path instead of per-token. Bails on `seq_len > sliding_window`.
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub batched_prefill: bool,

    /// `HF2Q_SKIP_TQ_ENCODE=1` — skip TQ encode for timing bisection.
    /// Produces garbage output; only used to attribute TQ cost.
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub skip_tq_encode: bool,

    /// `HF2Q_SKIP_TQ_SDPA=1` — skip TQ SDPA path for timing bisection.
    /// Produces garbage output.
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub skip_tq_sdpa: bool,

    // ========================================================================
    // Category 4 — warn-only (ineffective but safe).
    // ========================================================================
    /// `HF2Q_GRAPH_OPT=1` — use `begin_recorded` + `finish_optimized`.
    /// Shows no measured win on the default path (reorder aborts on
    /// unannotated dispatches).
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub graph_opt: bool,

    /// `HF2Q_LMHEAD_COMPARE=1` — keep both F16 and Q8 lm_head resident
    /// for future A/B diagnostics. Not wired into live decode today.
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub lmhead_compare: bool,

    // ========================================================================
    // Category 4 — internal perf tuning (not part of product surface).
    // ========================================================================
    /// `HF2Q_DUAL_BUFFER=N` — split decode session after layer N.
    /// Default split after layer 3 (applied by [`dual_buffer_split`])
    /// is part of the category-1 shipped path, so this field stores
    /// the raw env intent; the call site reconciles with `num_layers`.
    dual_buffer_raw: Option<String>,

    // ========================================================================
    // Category 4 — read-only diagnostics (silent; cannot affect output).
    // ========================================================================
    /// `HF2Q_DUMP_DIR` — output directory for dump files.
    /// Original parse: `unwrap_or_else(|_| "/tmp".into())`.
    pub dump_dir: String,

    /// `HF2Q_PREFILL_DUMP="L,T"` — dump Q/K/V norm chain at (layer L,
    /// token T) during per-token prefill.
    pub prefill_dump: Option<(usize, usize)>,

    /// `HF2Q_BATCHED_DUMP="L,T"` — same as above for batched prefill.
    pub batched_dump: Option<(usize, usize)>,

    /// `HF2Q_BATCHED_LAYER_SCAN=T` — dump `pf_hidden` row T at the start
    /// of every layer (cross-layer drift bisection).
    pub batched_layer_scan: Option<usize>,

    /// `HF2Q_DUMP_LAYERS=<seq_pos>` — enable per-layer hidden-state
    /// dumps at this decode position. Parsed to Option<usize>; call
    /// sites compare with their local `seq_pos`.
    pub dump_layers: Option<usize>,

    /// `HF2Q_DUMP_BOUNDARY=<seq_pos>` — dump pre-lm_head hidden +
    /// logits + top-10 argmax for a specific decode position.
    pub dump_boundary: Option<usize>,

    /// `HF2Q_DUMP_LAYER_DETAIL=<layer>` — sub-layer detail dump target.
    pub dump_layer_detail: Option<usize>,

    /// `HF2Q_DUMP_NORM_WEIGHT=<layer>` — one-shot dump of
    /// `input_layernorm.weight` at this layer.
    pub dump_norm_weight: Option<usize>,

    /// `HF2Q_DUMP_ALL_CACHE=1` — when dumping, include all cached K,V.
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub dump_all_cache: bool,

    /// `HF2Q_DUMP_RENDERED_PROMPT=<path>` — write rendered chat-template
    /// prompt to `<path>` and exit. Raw string; `None` if unset.
    pub dump_rendered_prompt: Option<String>,

    /// `HF2Q_DUMP_PROMPT_TOKENS` — log tokenized prompt head/tail.
    /// Original parse: `is_ok()` — true when set to ANY value,
    /// including empty. Not `== "1"`.
    pub dump_prompt_tokens: bool,

    // ========================================================================
    // Category 4 — timing/profiling attribution (no effect on output).
    // ========================================================================
    /// `HF2Q_MLX_TIMING` — log per-token encode/gpu_wait times etc.
    /// Original parse: `is_ok()` — true when set to ANY value.
    pub mlx_timing: bool,

    /// `HF2Q_SPLIT_TIMING=1` — insert an extra commit between body and
    /// head to measure them separately (~50 μs overhead).
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub split_timing: bool,

    /// `HF2Q_MLX_KERNEL_PROFILE=1` — per-kernel profile mode.
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub mlx_kernel_profile: bool,

    /// `HF2Q_MLX_PROFILE=1` — general MLX profiling.
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub mlx_profile: bool,

    // ========================================================================
    // Category 3 — benchmarking-only; ack-required.
    // Documented as an operator knob but unsafe to flip casually.
    // ========================================================================
    /// `HF2Q_LMHEAD_RERANK=0` — disable the exact-F32 rerank of top
    /// Q8 candidates. Reintroduces the rare near-tiebreak flip.
    /// `true` iff the env var is literally `"0"` (matching the original
    /// `matches!(var, Ok("0"))` check).
    pub lmhead_rerank_disabled: bool,

    // ========================================================================
    // Unsafe-ack gate input (read, not itself a category-4 toggle).
    // Consumed by S-4 to decide whether ack-required toggles activate.
    // ========================================================================
    /// `HF2Q_UNSAFE_EXPERIMENTS=1` — explicit acknowledgment that the
    /// user is intentionally flipping an ack-required investigation
    /// toggle.
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub unsafe_experiments_acked: bool,
}

impl InvestigationEnv {
    /// Parse every investigation env var from the current process
    /// environment. Called exactly once via [`INVESTIGATION_ENV`]'s
    /// `LazyLock`.
    pub fn from_env() -> Self {
        Self {
            // Ack-required.
            f16_kv: env_eq_one("HF2Q_F16_KV"),
            batched_prefill: env_eq_one("HF2Q_BATCHED_PREFILL"),
            skip_tq_encode: env_eq_one("HF2Q_SKIP_TQ_ENCODE"),
            skip_tq_sdpa: env_eq_one("HF2Q_SKIP_TQ_SDPA"),

            // Warn-only.
            graph_opt: env_eq_one("HF2Q_GRAPH_OPT"),
            lmhead_compare: env_eq_one("HF2Q_LMHEAD_COMPARE"),

            // Dual buffer: store raw for call-site resolution with num_layers.
            dual_buffer_raw: env::var("HF2Q_DUAL_BUFFER").ok(),

            // Silent diagnostics.
            dump_dir: env::var("HF2Q_DUMP_DIR").unwrap_or_else(|_| "/tmp".into()),
            prefill_dump: env_pair("HF2Q_PREFILL_DUMP"),
            batched_dump: env_pair("HF2Q_BATCHED_DUMP"),
            batched_layer_scan: env_usize("HF2Q_BATCHED_LAYER_SCAN"),
            dump_layers: env_usize("HF2Q_DUMP_LAYERS"),
            dump_boundary: env_usize("HF2Q_DUMP_BOUNDARY"),
            dump_layer_detail: env_usize("HF2Q_DUMP_LAYER_DETAIL"),
            dump_norm_weight: env_usize("HF2Q_DUMP_NORM_WEIGHT"),
            dump_all_cache: env_eq_one("HF2Q_DUMP_ALL_CACHE"),
            dump_rendered_prompt: env::var("HF2Q_DUMP_RENDERED_PROMPT").ok(),
            dump_prompt_tokens: env::var("HF2Q_DUMP_PROMPT_TOKENS").is_ok(),

            // Profiling / timing.
            mlx_timing: env::var("HF2Q_MLX_TIMING").is_ok(),
            split_timing: env_eq_one("HF2Q_SPLIT_TIMING"),
            mlx_kernel_profile: env_eq_one("HF2Q_MLX_KERNEL_PROFILE"),
            mlx_profile: env_eq_one("HF2Q_MLX_PROFILE"),

            // Ack-required benchmarking.
            lmhead_rerank_disabled: matches!(
                env::var("HF2Q_LMHEAD_RERANK").as_deref(),
                Ok("0")
            ),

            // Ack itself.
            unsafe_experiments_acked: env_eq_one("HF2Q_UNSAFE_EXPERIMENTS"),
        }
    }

    /// Resolve the dual-buffer split point against the current model's
    /// layer count. Preserves the original inline behavior exactly:
    ///
    /// - Env unset → default to `Some(3)` (split after layer 3).
    /// - Env set to a parsable `usize` → `Some(n)` only if
    ///   `n > 0 && n < num_layers`, else `None`.
    /// - Env set but not a `usize` → `None`.
    pub fn dual_buffer_split(&self, num_layers: usize) -> Option<usize> {
        match self.dual_buffer_raw.as_deref() {
            None => Some(3),
            Some(v) => v
                .parse::<usize>()
                .ok()
                .filter(|&n| n > 0 && n < num_layers),
        }
    }

    /// Whether any ack-required toggle is currently active. Used by S-4
    /// to decide when to enforce the `HF2Q_UNSAFE_EXPERIMENTS=1` gate.
    pub fn any_ack_required_active(&self) -> bool {
        self.f16_kv
            || self.batched_prefill
            || self.skip_tq_encode
            || self.skip_tq_sdpa
            || self.lmhead_rerank_disabled
    }

    /// Startup hook for the S-4 warning + ack gate. Currently a no-op;
    /// S-4 will fill this in to:
    ///   1. print one warning block listing every active investigation
    ///      toggle, marking ack-required ones;
    ///   2. abort (or refuse to activate) any ack-required toggle when
    ///      `HF2Q_UNSAFE_EXPERIMENTS=1` is not set.
    ///
    /// Kept as an explicit method — rather than baking it into
    /// `from_env` — so the call site (startup, pre-decode) is visible
    /// in `serve/mod.rs` rather than implicit at first field access.
    pub fn activate(&self) {
        // Intentionally empty. S-4 fills in.
    }
}

// ----------------------------------------------------------------------------
// Parse helpers — each mirrors one of the shapes used inline in the old code.
// ----------------------------------------------------------------------------

/// Mirrors `std::env::var(name).map_or(false, |v| v == "1")`.
fn env_eq_one(name: &str) -> bool {
    env::var(name).map_or(false, |v| v == "1")
}

/// Mirrors `std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok())`.
fn env_usize(name: &str) -> Option<usize> {
    env::var(name).ok().and_then(|v| v.parse::<usize>().ok())
}

/// Mirrors the inline `HF2Q_*_DUMP="L,T"` parsers: exactly one comma,
/// both sides parse as `usize`.
fn env_pair(name: &str) -> Option<(usize, usize)> {
    let v = env::var(name).ok()?;
    let parts: Vec<&str> = v.split(',').collect();
    if parts.len() != 2 {
        return None;
    }
    Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
}
