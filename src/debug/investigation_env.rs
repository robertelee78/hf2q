//! Central loader for hf2q's investigation-only environment variables.
//!
//! Every category-4 env var in `docs/shipping-contract.md` is parsed here
//! exactly once, at first access to [`INVESTIGATION_ENV`]. Hot-path code
//! reads fields off the cached struct instead of calling `std::env::var`
//! directly — so the decode loop does zero env lookups after init.
//!
//! # Unsafe-ack gate
//!
//! Toggles classified as *known to risk correctness or runtime
//! reliability* are ack-required: they only take effect when the user
//! also sets `HF2Q_UNSAFE_EXPERIMENTS=1`. If an ack-required toggle is
//! set without the ack, the public field reads `false` (i.e. the toggle
//! is disabled) and [`InvestigationEnv::activate`] prints a REFUSED
//! line so the user notices. This preserves prod-binary debuggability
//! while preventing accidental bug reports.
//!
//! # Not in scope here
//!
//! - `HF2Q_LMHEAD_Q8` is a category-2 operator knob (user-facing,
//!   documented in `docs/operator-env-vars.md`) and is read at load
//!   time inside the lm_head init path. It does not belong in this
//!   struct.
//!
//! # Parse semantics
//!
//! Each field's doc comment notes the original inline parse shape —
//! `is_ok()` vs `map_or(false, |v| v == "1")` are distinct signals in
//! the existing code and are preserved exactly.
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

/// Parsed-and-gated snapshot of every investigation-only env var. Each
/// ack-required field holds the *effective* value (already gated by
/// `HF2Q_UNSAFE_EXPERIMENTS=1`); [`activate`] is what actually surfaces
/// the refusal to the operator.
#[derive(Debug, Clone)]
pub struct InvestigationEnv {
    // ========================================================================
    // Category 4 — ack-required (known to risk correctness or reliability).
    // These fields hold the EFFECTIVE value: `true` only when the env var
    // was set AND `HF2Q_UNSAFE_EXPERIMENTS=1` was also set. If the user
    // set an ack-required toggle without the ack, the field is `false`
    // and `activate()` prints a REFUSED line.
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
    // Category 4 — warn-only (ineffective but safe). No gate; raw intent.
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

    /// `HF2Q_DUMP_ALL_CACHE=1` — at Phase 3A dump sites (attn Q/K/V, sdpa_out,
    /// dense cached K/V), fire for all 30 layers instead of the single
    /// detail layer, and include the full cached K/V history rather than
    /// the current write slot. Enables single-run full-coverage audits.
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub dump_all_cache: bool,

    // ========================================================================
    // Category 4 — C-0b localization: TQ packed-cache state dump.
    // ========================================================================
    /// `HF2Q_DUMP_TQ_STATE=1` — at end-of-prefill (after the final TQ-seq
    /// encode dispatch) dump k_packed + k_norms + v_packed + v_norms + a
    /// meta JSON sidecar for each layer in `dump_tq_layers_list`. Safe;
    /// purely read-only of live GPU buffers (requires a finish/begin pair
    /// at the call site, which is the caller's responsibility).
    pub dump_tq_state: bool,

    /// `HF2Q_DUMP_PRE_QUANT=1` — dump pre-hadamard-quantize K/V tensors
    /// to `{dump_dir}/pre_quant/` when layer_idx=0 and kv_seq_len=23.
    /// Fires BEFORE `dispatch_hadamard_quantize_kv` (line ~1226 in
    /// forward_mlx.rs), capturing the raw F32 K (attn_k_normed) and V
    /// (attn_v or moe_expert_out) before TQ encode. These pre-quant dumps
    /// serve as the independent-floor oracle inputs for ADR-007 C-2 multi-step
    /// audit. Category-4 read-only diagnostic; no `HF2Q_UNSAFE_EXPERIMENTS`
    /// ack required.
    pub dump_pre_quant: bool,

    /// `HF2Q_DUMP_LAYERS_LIST=0,5` — comma-separated layer indices to
    /// include in the TQ state dump. Empty list (default) means ALL layers
    /// when `dump_tq_state` is set. Parsed as `Vec<usize>`.
    pub dump_tq_layers_list: Vec<usize>,

    /// `HF2Q_DUMP_RENDERED_PROMPT=<path>` — write rendered chat-template
    /// prompt to `<path>` and exit. Raw string; `None` if unset.
    pub dump_rendered_prompt: Option<String>,

    /// `HF2Q_DUMP_PROMPT_TOKENS` — log tokenized prompt head/tail.
    /// Original parse: `is_ok()` — true when set to ANY value,
    /// including empty. Not `== "1"`.
    pub dump_prompt_tokens: bool,

    // ========================================================================
    // Category 4 — TurboQuant codebook selection (ADR-007 / iter-21 Track B).
    // Read-once; hot-path alloc gate and SDPA dispatch use the cached value.
    // ========================================================================
    /// `HF2Q_TQ_CODEBOOK_BITS` — KV codebook width selector.
    ///   - unset / "8": 8-bit native HB SDPA (DEFAULT, shippable).
    ///   - "4": legacy 4-bit `flash_attn_vec_tq` path (127-byte sourdough
    ///     ceiling; not shippable as default; opt-in only).
    ///   - "5" / "6": intermediate higher-bit HB SDPA (Lloyd-Max native).
    ///
    /// Stored as `u32` (0 = 4-bit legacy, 5/6/8 = the literal bit width).
    /// Original parse:
    ///   `match std::env::var("HF2Q_TQ_CODEBOOK_BITS").as_deref() { Ok("4")=>0, Ok("5")=>5, ... _ => 8 }`.
    pub tq_codebook_bits: u32,

    // ========================================================================
    // Category 4 — iter-18 S2C sliding-layer-0 first-divergence dump.
    // Gate + run-name for diagnostic decode dumps at layer 0 (hd=256,
    // sliding), positions 1..=10. Silent; no effect on forward-pass math.
    // ========================================================================
    /// `HF2Q_DUMP_SLIDING_LAYER_0=1` — enable first-divergence dump at
    /// layer 0 (sliding, hd=256) for decode positions 1..=10.
    /// Original parse: `std::env::var(...).ok().as_deref() == Some("1")`.
    pub dump_sliding_layer_0: bool,

    /// `HF2Q_DUMP_RUN_NAME` — run identifier string written into dump
    /// filenames so dense vs TQ passes can be distinguished.
    /// Original parse: `std::env::var(...).ok()` → `Option<String>`.
    pub dump_run_name: Option<String>,

    // ========================================================================
    // Category 4 — iter-18 S2A post-scale RMS probe (DEBUG_TQ_RMS).
    // Commits the encode command buffer and reads back block norms for
    // empirical RMS verification. Read-only; never alters forward math.
    // ========================================================================
    /// `HF2Q_DEBUG_TQ_RMS=1` — enable post-scale RMS probe for TQ encode.
    /// Commits the encode CB and reads back k_norms/v_norms for empirical
    /// verification of the [0.8, 1.2] RMS band (iter-19 A2 fix).
    /// Original parse: `std::env::var(...).ok().as_deref() == Some("1")`.
    pub debug_tq_rms: bool,

    // ========================================================================
    // Category 4 — SDPA regime selector (HF2Q_USE_DENSE / HF2Q_LAYER_POLICY).
    // These two vars select per-layer dense vs TQ SDPA dispatch.
    // Read per-token per-layer in the decode loop when gate_h_inactive and
    // when DecodeRegime::Default is active. LazyLock is the correct home.
    // ========================================================================
    /// `HF2Q_USE_DENSE=1` — force all layers to dense SDPA (ADR-009 Track 3).
    /// Original parse: `std::env::var("HF2Q_USE_DENSE").as_deref() == Ok("1")`.
    pub use_dense: bool,

    /// `HF2Q_LAYER_POLICY` — per-layer SDPA policy selector.
    ///   - "dense_all": all layers dense.
    ///   - "tq_all" / unset: all layers TQ (default).
    ///   - "tq_slide_dense_global": TQ for sliding, dense for global.
    ///   - "dense_slide_tq_global": dense for sliding, TQ for global.
    ///   - other: logs warning, defaults to `tq_all`.
    /// Original parse: `std::env::var("HF2Q_LAYER_POLICY").as_deref()` match.
    pub layer_policy: Option<String>,

    // ========================================================================
    // Category 4 — Gate H release-check companion plumbing (ADR-007 §853-866).
    // Three env vars that the audit binaries (iter23/24/25_audit.rs) set on
    // the hf2q child process; iter-108a wires them through the production
    // decode loop in `forward_mlx::forward_decode` so iter-108b can replace
    // the audit-binary harness with a release-check.sh-driven Gate 5 run.
    // All three are diagnostic-only — emit-only or token-replay; none touch
    // model weights or alter forward-pass math beyond replacing the *picked*
    // token (logits stay live for cosine/NLL capture).
    // ========================================================================
    /// `HF2Q_EMIT_NLL=1` — after each decoded-token's logits are computed,
    /// compute and emit the per-token NLL on stderr in the format
    /// `[HF2Q_NLL] step=<N> token=<X> nll=<Y>`. The format is the contract
    /// consumed by `iter25_audit.rs::parse_nll_values` for PPL aggregation.
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub emit_nll: bool,

    /// `HF2Q_DECODE_EMIT_TOKENS=1` — after each decode iteration, emit the
    /// picked token on stderr in the format
    /// `[HF2Q_DECODE_EMIT] step=<N> token=<X>`. The format is the contract
    /// consumed by `iter23/24/25_audit.rs::parse_emitted_tokens`.
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub decode_emit_tokens: bool,

    /// `HF2Q_DECODE_INPUT_TOKENS=<space-separated u32 list>` — replay fixed
    /// tokens overriding the on-GPU argmax (and any rerank). When set, for
    /// step `i < replay.len()` the decode loop returns `replay[i]` instead
    /// of the sampler's pick. After the replay buffer is exhausted, control
    /// falls through to the normal sampler. The argmax/rerank still runs
    /// (so cosine/NLL captures see live logits) — only the *picked* token
    /// is overridden.  Format mirrors the audit-binary contract:
    /// `iter23_audit.rs:206-216` writes the env var as
    /// `dense_tokens.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")`.
    /// Empty / unparsable entries are silently skipped.
    pub decode_input_tokens: Vec<u32>,

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
    // EFFECTIVE value (post-gate).
    // ========================================================================
    /// `HF2Q_LMHEAD_RERANK=0` — disable the exact-F32 rerank of top
    /// Q8 candidates. Reintroduces the rare near-tiebreak flip.
    /// Effective: `true` only when the env var is literally `"0"` AND
    /// `HF2Q_UNSAFE_EXPERIMENTS=1` is set.
    pub lmhead_rerank_disabled: bool,

    // ========================================================================
    // Ack gate state (consumed by `activate` to print startup summary).
    // ========================================================================
    /// `HF2Q_UNSAFE_EXPERIMENTS=1` — explicit acknowledgment that the
    /// user is intentionally flipping an ack-required investigation
    /// toggle.
    /// Original parse: `map_or(false, |v| v == "1")`.
    ///
    /// Exposed for introspection (tests, future diagnostics); the gate
    /// itself is applied inside [`from_env`] so hot-path readers don't
    /// need to recheck the ack.
    #[allow(dead_code)]
    pub unsafe_experiments_acked: bool,

    /// Raw (pre-gate) intents for ack-required toggles. Private; used
    /// only by [`activate`] to print REFUSED lines when the user set a
    /// toggle but omitted the ack.
    raw: RawAckIntent,
}

/// What the user *asked* for on the ack-required toggles, before the
/// `HF2Q_UNSAFE_EXPERIMENTS=1` gate is applied.
#[derive(Debug, Clone, Default)]
struct RawAckIntent {
    f16_kv: bool,
    batched_prefill: bool,
    skip_tq_encode: bool,
    skip_tq_sdpa: bool,
    lmhead_rerank_disabled: bool,
}

impl InvestigationEnv {
    /// Parse every investigation env var from the current process
    /// environment and apply the ack gate to ack-required toggles.
    /// Called exactly once via [`INVESTIGATION_ENV`]'s `LazyLock`.
    pub fn from_env() -> Self {
        let raw = RawAckIntent {
            f16_kv: env_eq_one("HF2Q_F16_KV"),
            batched_prefill: env_eq_one("HF2Q_BATCHED_PREFILL"),
            skip_tq_encode: env_eq_one("HF2Q_SKIP_TQ_ENCODE"),
            skip_tq_sdpa: env_eq_one("HF2Q_SKIP_TQ_SDPA"),
            lmhead_rerank_disabled: matches!(
                env::var("HF2Q_LMHEAD_RERANK").as_deref(),
                Ok("0")
            ),
        };
        let ack = env_eq_one("HF2Q_UNSAFE_EXPERIMENTS");

        Self {
            // Ack-required — effective value is raw AND ack.
            f16_kv: raw.f16_kv && ack,
            batched_prefill: raw.batched_prefill && ack,
            skip_tq_encode: raw.skip_tq_encode && ack,
            skip_tq_sdpa: raw.skip_tq_sdpa && ack,
            lmhead_rerank_disabled: raw.lmhead_rerank_disabled && ack,

            // Warn-only — no gate.
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
            dump_tq_state: env_eq_one("HF2Q_DUMP_TQ_STATE"),
            dump_pre_quant: env_eq_one("HF2Q_DUMP_PRE_QUANT"),
            dump_tq_layers_list: env_usize_list("HF2Q_DUMP_LAYERS_LIST"),
            dump_rendered_prompt: env::var("HF2Q_DUMP_RENDERED_PROMPT").ok(),
            dump_prompt_tokens: env::var("HF2Q_DUMP_PROMPT_TOKENS").is_ok(),

            // TurboQuant codebook width (ADR-007 / iter-21 Track B).
            tq_codebook_bits: match env::var("HF2Q_TQ_CODEBOOK_BITS").as_deref() {
                Ok("4") => 0u32,
                Ok("5") => 5u32,
                Ok("6") => 6u32,
                Ok("8") | Err(_) => 8u32,
                Ok(_other) => 8u32,
            },

            // iter-18 S2C sliding-layer-0 dump gate + run name.
            dump_sliding_layer_0: matches!(
                env::var("HF2Q_DUMP_SLIDING_LAYER_0").as_deref(),
                Ok("1")
            ),
            dump_run_name: env::var("HF2Q_DUMP_RUN_NAME").ok(),

            // iter-18 S2A post-scale RMS probe.
            debug_tq_rms: matches!(
                env::var("HF2Q_DEBUG_TQ_RMS").as_deref(),
                Ok("1")
            ),

            // SDPA regime selectors.
            use_dense: matches!(env::var("HF2Q_USE_DENSE").as_deref(), Ok("1")),
            layer_policy: env::var("HF2Q_LAYER_POLICY").ok(),

            // Gate H release-check plumbing (ADR-007 §853-866; iter-108a).
            emit_nll: env_eq_one("HF2Q_EMIT_NLL"),
            decode_emit_tokens: env_eq_one("HF2Q_DECODE_EMIT_TOKENS"),
            decode_input_tokens: env_u32_list_space("HF2Q_DECODE_INPUT_TOKENS"),

            // Profiling / timing.
            mlx_timing: env::var("HF2Q_MLX_TIMING").is_ok(),
            split_timing: env_eq_one("HF2Q_SPLIT_TIMING"),
            mlx_kernel_profile: env_eq_one("HF2Q_MLX_KERNEL_PROFILE"),
            mlx_profile: env_eq_one("HF2Q_MLX_PROFILE"),

            unsafe_experiments_acked: ack,
            raw,
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

    /// Print one-shot startup summary of active investigation toggles
    /// and any ack-required refusals. No output when nothing is set —
    /// safe to call unconditionally at process startup.
    ///
    /// Sections (only emitted when non-empty):
    ///
    /// - **UNSAFE (ack-required, activated):** ack-required toggles
    ///   that both the user asked for AND `HF2Q_UNSAFE_EXPERIMENTS=1`
    ///   was set. These are genuinely live for this run.
    /// - **REFUSED (ack-required, `HF2Q_UNSAFE_EXPERIMENTS=1` missing):**
    ///   ack-required toggles the user asked for but the ack was
    ///   absent; the toggles are disabled.
    /// - **ACTIVE (investigation, safe):** warn-only category-4 toggles
    ///   — they take effect but carry known caveats.
    /// - **DIAGNOSTICS:** read-only / timing toggles — listed for
    ///   visibility, no behavioral implication.
    pub fn activate(&self) {
        // Active ack-required (user set toggle AND ack was present).
        let mut active_unsafe: Vec<(&str, &str)> = Vec::new();
        if self.f16_kv {
            active_unsafe.push((
                "HF2Q_F16_KV=1",
                "known-worse KV cache representation (ADR-009)",
            ));
        }
        if self.batched_prefill {
            active_unsafe.push((
                "HF2Q_BATCHED_PREFILL=1",
                "experimental; errors when seq_len > sliding_window",
            ));
        }
        if self.skip_tq_encode {
            active_unsafe.push((
                "HF2Q_SKIP_TQ_ENCODE=1",
                "timing bisection; PRODUCES GARBAGE OUTPUT",
            ));
        }
        if self.skip_tq_sdpa {
            active_unsafe.push((
                "HF2Q_SKIP_TQ_SDPA=1",
                "timing bisection; PRODUCES GARBAGE OUTPUT",
            ));
        }
        if self.lmhead_rerank_disabled {
            active_unsafe.push((
                "HF2Q_LMHEAD_RERANK=0",
                "raw Q8 argmax; rare near-tiebreak flips",
            ));
        }

        // Refused (user set ack-required toggle but HF2Q_UNSAFE_EXPERIMENTS=1 missing).
        let mut refused: Vec<(&str, &str)> = Vec::new();
        if self.raw.f16_kv && !self.f16_kv {
            refused.push((
                "HF2Q_F16_KV=1",
                "ack required: also set HF2Q_UNSAFE_EXPERIMENTS=1",
            ));
        }
        if self.raw.batched_prefill && !self.batched_prefill {
            refused.push((
                "HF2Q_BATCHED_PREFILL=1",
                "ack required: also set HF2Q_UNSAFE_EXPERIMENTS=1",
            ));
        }
        if self.raw.skip_tq_encode && !self.skip_tq_encode {
            refused.push((
                "HF2Q_SKIP_TQ_ENCODE=1",
                "ack required: also set HF2Q_UNSAFE_EXPERIMENTS=1",
            ));
        }
        if self.raw.skip_tq_sdpa && !self.skip_tq_sdpa {
            refused.push((
                "HF2Q_SKIP_TQ_SDPA=1",
                "ack required: also set HF2Q_UNSAFE_EXPERIMENTS=1",
            ));
        }
        if self.raw.lmhead_rerank_disabled && !self.lmhead_rerank_disabled {
            refused.push((
                "HF2Q_LMHEAD_RERANK=0",
                "ack required: also set HF2Q_UNSAFE_EXPERIMENTS=1",
            ));
        }

        // Active warn-only (safe but noteworthy).
        let mut active_safe: Vec<(&str, &str)> = Vec::new();
        if self.graph_opt {
            active_safe.push((
                "HF2Q_GRAPH_OPT=1",
                "no measured win; reorder aborts on unannotated dispatches",
            ));
        }
        if self.lmhead_compare {
            active_safe.push((
                "HF2Q_LMHEAD_COMPARE=1",
                "inert today (not wired into live decode)",
            ));
        }

        // Read-only / timing diagnostics.
        let mut diagnostics: Vec<String> = Vec::new();
        if let Some((l, t)) = self.prefill_dump {
            diagnostics.push(format!("HF2Q_PREFILL_DUMP={l},{t}"));
        }
        if let Some((l, t)) = self.batched_dump {
            diagnostics.push(format!("HF2Q_BATCHED_DUMP={l},{t}"));
        }
        if let Some(t) = self.batched_layer_scan {
            diagnostics.push(format!("HF2Q_BATCHED_LAYER_SCAN={t}"));
        }
        if let Some(p) = self.dump_layers {
            diagnostics.push(format!("HF2Q_DUMP_LAYERS={p}"));
        }
        if let Some(p) = self.dump_boundary {
            diagnostics.push(format!("HF2Q_DUMP_BOUNDARY={p}"));
        }
        if let Some(l) = self.dump_layer_detail {
            diagnostics.push(format!("HF2Q_DUMP_LAYER_DETAIL={l}"));
        }
        if let Some(l) = self.dump_norm_weight {
            diagnostics.push(format!("HF2Q_DUMP_NORM_WEIGHT={l}"));
        }
        if self.dump_all_cache {
            diagnostics.push("HF2Q_DUMP_ALL_CACHE=1".into());
        }
        if self.dump_tq_state {
            let layers_str = if self.dump_tq_layers_list.is_empty() {
                "all".to_string()
            } else {
                self.dump_tq_layers_list.iter().map(|l| l.to_string())
                    .collect::<Vec<_>>().join(",")
            };
            diagnostics.push(format!("HF2Q_DUMP_TQ_STATE=1 (layers: {layers_str})"));
        }
        if self.dump_pre_quant {
            diagnostics.push("HF2Q_DUMP_PRE_QUANT=1 (pre-quant K/V dump before TQ encode)".into());
        }
        if self.dump_rendered_prompt.is_some() {
            diagnostics.push("HF2Q_DUMP_RENDERED_PROMPT=<path>".into());
        }
        if self.dump_prompt_tokens {
            diagnostics.push("HF2Q_DUMP_PROMPT_TOKENS".into());
        }
        if self.tq_codebook_bits != 8 {
            diagnostics.push(format!(
                "HF2Q_TQ_CODEBOOK_BITS={} ({})",
                if self.tq_codebook_bits == 0 { 4 } else { self.tq_codebook_bits },
                if self.tq_codebook_bits == 0 { "legacy 4-bit TQ" } else { "HB SDPA" }
            ));
        }
        if self.dump_sliding_layer_0 {
            diagnostics.push("HF2Q_DUMP_SLIDING_LAYER_0=1".into());
        }
        if let Some(ref name) = self.dump_run_name {
            diagnostics.push(format!("HF2Q_DUMP_RUN_NAME={name}"));
        }
        if self.debug_tq_rms {
            diagnostics.push("HF2Q_DEBUG_TQ_RMS=1".into());
        }
        if self.use_dense {
            diagnostics.push("HF2Q_USE_DENSE=1".into());
        }
        if let Some(ref policy) = self.layer_policy {
            diagnostics.push(format!("HF2Q_LAYER_POLICY={policy}"));
        }
        if self.mlx_timing {
            diagnostics.push("HF2Q_MLX_TIMING".into());
        }
        if self.split_timing {
            diagnostics.push("HF2Q_SPLIT_TIMING=1".into());
        }
        if self.mlx_kernel_profile {
            diagnostics.push("HF2Q_MLX_KERNEL_PROFILE=1".into());
        }
        if self.mlx_profile {
            diagnostics.push("HF2Q_MLX_PROFILE=1".into());
        }

        let nothing_to_report = active_unsafe.is_empty()
            && refused.is_empty()
            && active_safe.is_empty()
            && diagnostics.is_empty();
        if nothing_to_report {
            return;
        }

        eprintln!();
        eprintln!("hf2q: investigation-only environment variables detected");
        eprintln!("      (not part of the shipping contract — see docs/shipping-contract.md)");

        if !active_unsafe.is_empty() {
            eprintln!();
            eprintln!("  UNSAFE (ack-required, activated):");
            for (name, note) in &active_unsafe {
                eprintln!("    {name:<30}  {note}");
            }
        }

        if !refused.is_empty() {
            eprintln!();
            eprintln!("  REFUSED (ack-required, HF2Q_UNSAFE_EXPERIMENTS=1 not set):");
            for (name, note) in &refused {
                eprintln!("    {name:<30}  {note}");
            }
            eprintln!();
            eprintln!("  The REFUSED toggles above are DISABLED for this run.");
        }

        if !active_safe.is_empty() {
            eprintln!();
            eprintln!("  ACTIVE (investigation, safe):");
            for (name, note) in &active_safe {
                eprintln!("    {name:<30}  {note}");
            }
        }

        if !diagnostics.is_empty() {
            eprintln!();
            eprintln!("  DIAGNOSTICS (read-only / timing):");
            for name in &diagnostics {
                eprintln!("    {name}");
            }
        }

        eprintln!();
    }
}

// ----------------------------------------------------------------------------
// Parse helpers — each mirrors one of the shapes used inline in the old code.
// ----------------------------------------------------------------------------

/// Mirrors `std::env::var(name).map_or(false, |v| v == "1")`.
fn env_eq_one(name: &str) -> bool {
    env::var(name).is_ok_and(|v| v == "1")
}

/// Mirrors `std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok())`.
fn env_usize(name: &str) -> Option<usize> {
    env::var(name).ok().and_then(|v| v.parse::<usize>().ok())
}

/// Parses `HF2Q_DUMP_LAYERS_LIST=0,5` as a `Vec<usize>`.
/// Returns an empty Vec if the env var is unset or empty.
fn env_usize_list(name: &str) -> Vec<usize> {
    env::var(name).ok()
        .filter(|v| !v.is_empty())
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_default()
}

/// Parses `HF2Q_DECODE_INPUT_TOKENS="123 456 789"` as a `Vec<u32>`.
/// Returns an empty Vec if the env var is unset or empty.  Whitespace
/// is the separator (matching `iter23_audit.rs:207`'s `.join(" ")`);
/// unparsable entries are silently skipped so the replay simply ends
/// at the first malformed token (rather than blowing up the whole run).
fn env_u32_list_space(name: &str) -> Vec<u32> {
    env::var(name).ok()
        .filter(|v| !v.is_empty())
        .map(|v| v.split_whitespace().filter_map(|s| s.parse::<u32>().ok()).collect())
        .unwrap_or_default()
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

// ────────────────────────────────────────────────────────────────────
// Tests — 6 new InvestigationEnv parse fields (wave-1.5 T1.2)
// ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Env-var mutation is process-wide; serialize all tests that touch it.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Save+restore env vars around a test that mutates them.
    struct EnvGuard {
        snapshots: Vec<(String, Option<String>)>,
    }
    impl EnvGuard {
        fn new(keys: &[&str]) -> Self {
            let snapshots = keys
                .iter()
                .map(|k| (k.to_string(), std::env::var(k).ok()))
                .collect();
            for k in keys {
                std::env::remove_var(k);
            }
            Self { snapshots }
        }
        fn set(&self, k: &str, v: &str) {
            std::env::set_var(k, v);
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.snapshots {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    // ── tq_codebook_bits ─────────────────────────────────────────────

    #[test]
    fn tq_codebook_bits_default_when_unset() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_TQ_CODEBOOK_BITS"]);
        drop(guard); // var removed by constructor
        let env = InvestigationEnv::from_env();
        assert_eq!(env.tq_codebook_bits, 8u32, "unset => default 8");
    }

    #[test]
    fn tq_codebook_bits_parse_success_all_variants() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_TQ_CODEBOOK_BITS"]);

        // "4" → sentinel 0 (legacy path).
        guard.set("HF2Q_TQ_CODEBOOK_BITS", "4");
        assert_eq!(InvestigationEnv::from_env().tq_codebook_bits, 0u32);

        // "5" → 5.
        guard.set("HF2Q_TQ_CODEBOOK_BITS", "5");
        assert_eq!(InvestigationEnv::from_env().tq_codebook_bits, 5u32);

        // "6" → 6.
        guard.set("HF2Q_TQ_CODEBOOK_BITS", "6");
        assert_eq!(InvestigationEnv::from_env().tq_codebook_bits, 6u32);

        // "8" → 8.
        guard.set("HF2Q_TQ_CODEBOOK_BITS", "8");
        assert_eq!(InvestigationEnv::from_env().tq_codebook_bits, 8u32);
    }

    #[test]
    fn tq_codebook_bits_unknown_value_falls_back_to_8() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_TQ_CODEBOOK_BITS"]);
        guard.set("HF2Q_TQ_CODEBOOK_BITS", "99");
        assert_eq!(
            InvestigationEnv::from_env().tq_codebook_bits,
            8u32,
            "unrecognised value => 8"
        );
    }

    // ── dump_sliding_layer_0 ─────────────────────────────────────────

    #[test]
    fn dump_sliding_layer_0_default_when_unset() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::new(&["HF2Q_DUMP_SLIDING_LAYER_0"]);
        assert!(!InvestigationEnv::from_env().dump_sliding_layer_0);
    }

    #[test]
    fn dump_sliding_layer_0_enabled_by_one() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_DUMP_SLIDING_LAYER_0"]);
        guard.set("HF2Q_DUMP_SLIDING_LAYER_0", "1");
        assert!(InvestigationEnv::from_env().dump_sliding_layer_0);
    }

    #[test]
    fn dump_sliding_layer_0_not_enabled_by_other_values() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_DUMP_SLIDING_LAYER_0"]);
        for bad in &["0", "true", "yes", "2"] {
            guard.set("HF2Q_DUMP_SLIDING_LAYER_0", bad);
            assert!(
                !InvestigationEnv::from_env().dump_sliding_layer_0,
                "value {:?} must not enable the flag",
                bad
            );
        }
    }

    // ── dump_run_name ────────────────────────────────────────────────

    #[test]
    fn dump_run_name_default_when_unset() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::new(&["HF2Q_DUMP_RUN_NAME"]);
        assert_eq!(InvestigationEnv::from_env().dump_run_name, None);
    }

    #[test]
    fn dump_run_name_captures_arbitrary_string() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_DUMP_RUN_NAME"]);
        guard.set("HF2Q_DUMP_RUN_NAME", "dense-vs-tq-pass-01");
        assert_eq!(
            InvestigationEnv::from_env().dump_run_name.as_deref(),
            Some("dense-vs-tq-pass-01")
        );
    }

    #[test]
    fn dump_run_name_empty_string_is_some() {
        // env::var returns Ok("") for an empty var; .ok() => Some("").
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_DUMP_RUN_NAME"]);
        guard.set("HF2Q_DUMP_RUN_NAME", "");
        assert_eq!(
            InvestigationEnv::from_env().dump_run_name.as_deref(),
            Some("")
        );
    }

    // ── debug_tq_rms ─────────────────────────────────────────────────

    #[test]
    fn debug_tq_rms_default_when_unset() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::new(&["HF2Q_DEBUG_TQ_RMS"]);
        assert!(!InvestigationEnv::from_env().debug_tq_rms);
    }

    #[test]
    fn debug_tq_rms_enabled_by_one() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_DEBUG_TQ_RMS"]);
        guard.set("HF2Q_DEBUG_TQ_RMS", "1");
        assert!(InvestigationEnv::from_env().debug_tq_rms);
    }

    #[test]
    fn debug_tq_rms_not_enabled_by_other_values() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_DEBUG_TQ_RMS"]);
        for bad in &["0", "true", "on"] {
            guard.set("HF2Q_DEBUG_TQ_RMS", bad);
            assert!(
                !InvestigationEnv::from_env().debug_tq_rms,
                "value {:?} must not enable debug_tq_rms",
                bad
            );
        }
    }

    // ── use_dense ────────────────────────────────────────────────────

    #[test]
    fn use_dense_default_when_unset() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::new(&["HF2Q_USE_DENSE"]);
        assert!(!InvestigationEnv::from_env().use_dense);
    }

    #[test]
    fn use_dense_enabled_by_one() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_USE_DENSE"]);
        guard.set("HF2Q_USE_DENSE", "1");
        assert!(InvestigationEnv::from_env().use_dense);
    }

    #[test]
    fn use_dense_not_enabled_by_other_values() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_USE_DENSE"]);
        for bad in &["0", "true", "yes", "dense"] {
            guard.set("HF2Q_USE_DENSE", bad);
            assert!(
                !InvestigationEnv::from_env().use_dense,
                "value {:?} must not enable use_dense",
                bad
            );
        }
    }

    // ── layer_policy ─────────────────────────────────────────────────

    #[test]
    fn layer_policy_default_when_unset() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::new(&["HF2Q_LAYER_POLICY"]);
        assert_eq!(InvestigationEnv::from_env().layer_policy, None);
    }

    #[test]
    fn layer_policy_captures_known_policy_strings() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_LAYER_POLICY"]);

        for policy in &[
            "dense_all",
            "tq_all",
            "tq_slide_dense_global",
            "dense_slide_tq_global",
        ] {
            guard.set("HF2Q_LAYER_POLICY", policy);
            assert_eq!(
                InvestigationEnv::from_env().layer_policy.as_deref(),
                Some(*policy),
                "policy {:?} should be captured verbatim",
                policy
            );
        }
    }

    #[test]
    fn layer_policy_captures_unknown_string_verbatim() {
        // The policy selector deliberately accepts unknown strings (logs a
        // warning at runtime; defaulting happens in the dispatch caller).
        // The parse step must not filter them out.
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_LAYER_POLICY"]);
        guard.set("HF2Q_LAYER_POLICY", "some_future_policy");
        assert_eq!(
            InvestigationEnv::from_env().layer_policy.as_deref(),
            Some("some_future_policy")
        );
    }
}
