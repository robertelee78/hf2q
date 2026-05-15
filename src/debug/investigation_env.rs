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
    /// `HF2Q_F16_KV=1` — allocate dense KV cache as F16. Halves KV
    /// read bandwidth at attention.
    ///
    /// Historical: ADR-009 (2026-04-16) classified this as "known-
    /// worse output" based on a measured 19× cache_k drift + 45×
    /// sdpa_out drift vs llama.cpp. **REFUTED at HEAD** by ADR-028
    /// iter-168 byte-identity test (`mlx-native` commit `a325827`,
    /// `tests/test_flash_attn_vec_f16_byte_identity.rs`):
    ///
    ///   F32 baseline ↔ F32-with-F16-rounded-inputs:  rel_rms 2.57e-5
    ///   F32 baseline ↔ F16-kernel:                   rel_rms 2.57e-5
    ///   F32-with-F16-inputs ↔ F16-kernel:            rel_rms 0.0
    ///   amplification:                                1.00×
    ///
    /// The F16 kernel is byte-identical to the F32 kernel fed
    /// F16-rounded inputs — **F16 storage precision is the only
    /// source of difference**. ADR-009's reported amplification
    /// has been fixed somewhere in the iter-101..149 FA-vec work
    /// (NSG axis + FWHT-pre fusion, ADR-028).
    ///
    /// Production effect at gemma4 26B-A4B (sliding=1024):
    ///   - +15.7% wall-clock when combined with `HF2Q_USE_DENSE=1`
    ///     at 200-token benches (62.5 → 72.3 tok/s)
    ///   - 251 MiB/slot KV memory (vs 502 F32-dense, 191 TQ-HB)
    ///   - 25 ppm rel_rms output drift (F16 precision tradeoff)
    ///
    /// **DEPRECATED at ADR-028 iter-234 (2026-05-09)**: long-context
    /// stress testing (1000 tok) revealed gemma4 produces random
    /// `<pad>` emission at non-deterministic output lengths (sweep:
    /// N=200 ✓, N=400 ✗, N=600 ✓, N=800 ✓, N=1000 ✗).  hf2q's
    /// non-greedy default sampler means F16 logit-noise occasionally
    /// pushes `<pad>` to argmax-rank early, killing generation.
    /// The iter-189 "+8.5%" win was sampling luck at 200-tok.
    ///
    /// Cross-model check at qwen3.6 35B-A3B APEX-Q5_K_M, 1000-tok:
    /// coherent BUT zero perf gain (126.2 vs 125.7 tok/s — MoE
    /// sparse activation makes KV bandwidth not the bottleneck).
    ///
    /// → **No remaining safe use case**.  Use Path E+G (USE_DENSE +
    /// LMHEAD_Q6K) for gemma4 perf instead — F32 KV preserved,
    /// +3.7% over default at 1000-tok, coherent at long context.
    /// Activation banner now warns DEPRECATED.
    ///
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub f16_kv: bool,

    /// `HF2Q_BATCHED_PREFILL` — use the batched prefill path instead of
    /// per-token.  ADR-028 iter-344 default-flipped to ON: per-token
    /// prefill at default was 14-45× SLOWER than peer (70 tok/s vs
    /// 3130 tok/s pp512); batched gives ~34× speedup at pp4096 (2366
    /// tok/s = 0.80× peer) with coherence intact at every tested length
    /// up to pp3813 (4× sliding_window) per iter-343.  Operator iter-76
    /// signed off on the L6 MoE sliding_wrap deferral (still ack-able
    /// via opt-out).  Opt out via `HF2Q_BATCHED_PREFILL=0` / `=false` /
    /// `=off`.  Decoupled from `HF2Q_UNSAFE_EXPERIMENTS` ack (iter-344).
    pub batched_prefill: bool,

    /// `HF2Q_SKIP_TQ_ENCODE=1` — skip TQ encode for timing bisection.
    /// Produces garbage output; only used to attribute TQ cost.
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub skip_tq_encode: bool,

    /// `HF2Q_SKIP_TQ_SDPA=1` — skip TQ SDPA path for timing bisection.
    /// Produces garbage output.
    /// Original parse: `map_or(false, |v| v == "1")`.
    pub skip_tq_sdpa: bool,

    /// `HF2Q_SKIP_DENSE_MLP=1` — skip dense MLP dispatches (gate, up,
    /// fused_gelu_mul, down) per layer for timing bisection.  ADR-028
    /// iter-200 — measure dense MLP cost as candidate for further
    /// optimization.  Produces garbage output (mlp_down stale buffer).
    pub skip_dense_mlp: bool,

    /// `HF2Q_SKIP_MOE_EXPERTS=1` — skip MoE expert dispatches per layer
    /// (gate_up_id + swiglu + down_id) for timing bisection.  ADR-028
    /// iter-201 — measure MoE expert cost (already partially measured
    /// at iter-181 = 1.84 ms matmul; this captures full chain).
    /// Produces garbage output (moe_down_id_out stale).
    pub skip_moe_experts: bool,

    /// `HF2Q_SKIP_MOE_SWIGLU=1` — skip just the moe_swiglu_batch_encode
    /// dispatch per layer (keep gate_up_id and down_id).  ADR-028 iter-202
    /// — bisect swiglu's exact cost before deciding whether to build a
    /// fused Q6_K _swiglu kernel.  Produces garbage (down_id reads stale
    /// moe_swiglu_id_out).
    pub skip_moe_swiglu: bool,

    /// `HF2Q_SKIP_HEAD_NORM_ROPE=1` — skip the 2 fused_head_norm_rope
    /// dispatches per layer (Q-norm-rope + K-norm-rope).  ADR-028 iter-204
    /// — bisect attention head-prep cost.  Produces garbage SDPA
    /// (attn_q_normed/attn_k_normed stale).
    pub skip_head_norm_rope: bool,

    /// `HF2Q_SKIP_POST_ATTN_NORM=1` — skip the post-attention
    /// fused_norm_add dispatch per layer (sequential between O-proj and
    /// B8).  ADR-028 iter-205 — bisect a sequential-critical-path op.
    /// Produces garbage residual stream.
    pub skip_post_attn_norm: bool,

    /// `HF2Q_SKIP_WEIGHTED_SUM=1` — skip B14 moe_weighted_sum_encode
    /// dispatch per layer (sequential after down_id, combines top_k
    /// expert outputs).  ADR-028 iter-206 — bisect.  Produces garbage
    /// (moe_accum stale).
    pub skip_weighted_sum: bool,

    /// `HF2Q_SKIP_END_OF_LAYER=1` — skip the 2 fused_norm_add
    /// dispatches at end-of-layer:
    ///   (a) post-FF norm 2 + combine MLP+MoE (writes mlp_down)
    ///   (b) end-of-layer residual + scalar mul (writes hidden)
    /// Both sequential, both use fused_norm_add (same kernel as the
    /// 0.55 ms post-attn-norm-add measured in iter-205).
    /// ADR-028 iter-207.  Produces garbage (hidden stale).
    pub skip_end_of_layer: bool,

    /// `HF2Q_SKIP_END_OF_LAYER_FINAL=1` — skip ONLY the FINAL
    /// fused_norm_add_scalar at end-of-layer (writes hidden).  Keeps
    /// post-FF norm 2 (writes mlp_down).  ADR-028 iter-208 sub-bisect
    /// to isolate the final residual update cost vs the post-FF norm 2.
    pub skip_end_of_layer_final: bool,

    /// `HF2Q_SKIP_ATTN_QKV=1` — skip the 3 attention QKV qmatmul
    /// dispatches per layer (Q proj + K proj + V proj, all concurrent).
    /// ADR-028 iter-210 — measure production cost of attention QKV
    /// (vs iter-180 batched-bench estimate).  Produces garbage
    /// downstream attention.
    pub skip_attn_qkv: bool,

    /// `HF2Q_SKIP_O_PROJ=1` — skip the attention O proj qmatmul per
    /// layer (sequential after SDPA).  ADR-028 iter-211 — bisect.
    /// Produces garbage attn_out.
    pub skip_o_proj: bool,

    /// `HF2Q_SKIP_ROUTING=1` — skip B9 router_proj qmatmul +
    /// B10 fused_moe_routing dispatches per layer (2 dispatches/layer).
    /// ADR-028 iter-213 — bisect routing scaffold cost.
    /// ⚠ INVALID BISECT (iter-213 lesson): produces garbage expert IDs
    /// which collapse MoE matmul to single-expert reads (cache hit
    /// artifact).  Real cost is ~0.5-1 ms; SKIP measures 4.24 ms
    /// (3 ms is cache-hit artifact).  Kept for future SKIP_ROUTING_WITH_VALID_IDS work.
    pub skip_routing: bool,

    /// `HF2Q_SKIP_V_NORM=1` — skip the per-head V-norm RMS norm
    /// dispatch (sequential after V proj, before KV cache copy).
    /// ADR-028 iter-214.  Produces garbage V cache (SDPA reads bad V).
    /// VALID bisect: V-norm output is consumed by KV-copy + SDPA as
    /// data, not control signals — no cache-pattern confound.
    pub skip_v_norm: bool,

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

    /// `HF2Q_DUMP_PRE_QUANT_LAYERS=0,15,30,45,60` — comma-separated layer
    /// indices to include in `dump_pre_quant`. Empty (default) means
    /// `[0]` only (legacy behavior preserved). When set, the
    /// `kv_seq_len == 23` gate is also relaxed to fire at every position
    /// in `dump_pre_quant_positions` (or every position if that's empty).
    /// Path C F-0.3 distribution measurement.
    pub dump_pre_quant_layers: Vec<usize>,

    /// `HF2Q_DUMP_PRE_QUANT_POSITIONS=23,50,100,200,500` — comma-separated
    /// `kv_seq_len` values at which to fire the pre-quant dump. Empty
    /// (default) means `[23]` only (legacy behavior preserved). Path C
    /// F-0.3 distribution measurement.
    pub dump_pre_quant_positions: Vec<usize>,

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

    /// `HF2Q_HYBRID_KV` — ADR-028 Phase 10 (iter-347) / ADR-029 iter-13 default-flip.
    ///
    /// At lazy KV alloc in `forward_decode`, build `HybridKvBuffers` instead
    /// of `HbKvBuffers`: F16 K + TQ-HB-packed V. The K side stays dense F16
    /// (peer-equivalent simdgroup-matmul K throughput); V stays 1-byte-per-
    /// element TQ-HB packed. Memory cost: 158 MB at gemma4 32K vs 128 MB
    /// pure TQ-HB (3.19× saving vs 3.94×, preserving 81% of the TQ-HB
    /// memory advantage). Routes attention through `flash_attn_vec_hybrid`
    /// (mlx-native Phase 10d) and `dispatch_kv_copy_kf16_quantize_v_no_fwht`
    /// (Phase 10c.5 fused write).
    ///
    /// **Default ON** since ADR-029 iter-13 (2026-05-11). H12 confirmed in
    /// iter-12 via 3-trial fresh-process bench at HEAD `0808e4e9`: median
    /// +9.5% gemma4 throughput (78.5 vs 71.6 t/s) with byte-class-coherent
    /// output. Brings gemma4-APEX-Q5_K_M peer ratio from 0.756× → 0.805×
    /// (+4.9 pp). Opt-out via `=0` / `=false` / `=off` (legacy TQ-HB path).
    pub hybrid_kv: bool,

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
    // Category 4 — Wave 5a: Qwen3.6 autoregressive forward-path opt-in.
    // Qwen3.6 GGUFs are detected via `general.name` substring match. The
    // existing Qwen3.5 forward path (`inference::models::qwen35::*`) is the
    // autoregressive (per-token state-update) DeltaNet kernel; correctness
    // is established at short prefill lengths but the SOTA chunk-scan kernel
    // for long-prefill perf is deferred to W-5b. Until W-5b lands, Qwen3.6
    // GGUFs require explicit opt-in via this env var to avoid silently
    // shipping a slow long-prefill path.
    // ========================================================================
    /// `HF2Q_QWEN36_AUTOREG=1` — opt in to running Qwen3.6 GGUFs through the
    /// existing autoregressive Qwen3.5 forward path. When unset and a
    /// Qwen3.6 GGUF is detected, `cmd_generate` errors out with an
    /// operator-actionable message rather than silently routing through
    /// the slow autoregressive path. Wave 5a (ADR-005 Phase 4 ACs
    /// 5468/5470 partial closure). Wave 5b will replace this gate with
    /// a chunk-scan kernel for long-prefill SOTA perf.
    /// Original parse (none — new field): `env_eq_one("HF2Q_QWEN36_AUTOREG")`.
    pub qwen36_autoreg: bool,

    // ========================================================================
    // Category 3 (ack-required) — Wave 5b iter 5: chunk-scan prefill opt-in.
    //
    // Routes Qwen3.6 prefills at `seq_len > 64` through the mlx-native
    // chunk-parallel delta-rule pipeline (`mlx_native::ops::
    // chunk_gated_delta_rule::dispatch_chunk_gated_delta_rule_fwd`)
    // instead of the autoregressive per-token path.  Closes the long-
    // prefill SOTA perf path on ADR-005 ACs 5468/5470 (currently
    // only-partial via Wave 5a's autoregressive opt-in).
    //
    // Classified Category 3 (benchmarking-only, ack-required) because it
    // changes the forward-pass dispatch — sourdough byte-prefix gates and
    // walk-bar parity at pp4096+ are validated as separate iters before
    // this becomes Category 1. Effective only when
    // `HF2Q_UNSAFE_EXPERIMENTS=1` is also set.
    // ========================================================================
    /// `HF2Q_CHUNK_SCAN_PREFILL=1` — opt in to dispatching Qwen3.6 prefills
    /// at `seq_len > 64` through the chunk-parallel delta-rule pipeline.
    /// Effective: `true` only when the env var is `"1"` AND
    /// `HF2Q_UNSAFE_EXPERIMENTS=1` is set.  Wave 5b iter 5.
    /// Original parse (none — new field): `env_eq_one("HF2Q_CHUNK_SCAN_PREFILL")`
    /// gated by `env_eq_one("HF2Q_UNSAFE_EXPERIMENTS")`.
    pub chunk_scan_prefill: bool,

    // ========================================================================
    // Wave 5b.20 `gqa_expand_legacy` field + `HF2Q_GQA_EXPAND_LEGACY` env gate
    // removed in W-5b.21 after a 30/30 cross-path determinism audit at PP4106
    // (id 11 on every cell) confirmed parity. The GPU
    // `dispatch_repeat_tiled_f32` path is now unconditional in
    // `apply_gated_delta_net_chunk`. Standing parity bar:
    // `mlx-native/tests/test_repeat_tiled.rs::test_repeat_tiled_qwen36_27b_shape_seq128`.
    // ========================================================================

    // ========================================================================
    // ADR-031 Phase B — parallel-encode gate + interaction guard.
    //
    // HF2Q_PARALLEL_ENCODE=1 opts in to encoding layer chunks concurrently
    // on the global encoder worker + main thread.  Default OFF.
    //
    // Interaction guard: HF2Q_PER_LAYER_DISP=1 uses a global mlx-native
    // dispatch counter; two CPU encode threads racing on it produce
    // meaningless per-layer attribution.  When PER_LAYER_DISP is set,
    // parallel_encode_enabled() returns false and emits a once-per-process
    // warning via tracing::warn.
    //
    // parallel_encode_raw: raw intent (pre-guard); true when
    //   HF2Q_PARALLEL_ENCODE=1 was set at process start.
    // per_layer_disp_raw: true when HF2Q_PER_LAYER_DISP=1 was set at
    //   process start.  Snapshotted here so both are read once from the
    //   environment at LazyLock init time.
    // ========================================================================

    /// Raw (pre-guard) intent for `HF2Q_PARALLEL_ENCODE=1`.
    /// Use `parallel_encode_enabled()` for the guarded effective value.
    pub parallel_encode_raw: bool,

    /// Raw snapshot of `HF2Q_PER_LAYER_DISP=1`.  Also used by forward_decode
    /// to decide whether to print per-layer dispatch counts; snapshotted here
    /// so both interaction-guard checks see the same value.
    pub per_layer_disp_raw: bool,

    /// Minimum `seq_pos` at which the parallel-encode path engages.
    /// Below this depth the serial path is used even when
    /// `HF2Q_PARALLEL_ENCODE=1` (worker overhead > benefit at shallow KV
    /// depth).  Default 512.  Override via
    /// `HF2Q_PARALLEL_ENCODE_KV_THRESHOLD=N`.
    pub parallel_encode_kv_threshold: usize,

    // ========================================================================
    // Category 4 — SDPA regime selector (HF2Q_USE_DENSE / HF2Q_LAYER_POLICY).
    // These two vars select per-layer dense vs TQ SDPA dispatch.
    // Read per-token per-layer in the decode loop when gate_h_inactive and
    // when DecodeRegime::Default is active. LazyLock is the correct home.
    // ========================================================================
    /// `HF2Q_USE_DENSE=1` — force all layers to dense SDPA (ADR-009 Track 3).
    /// Original parse: `std::env::var("HF2Q_USE_DENSE").as_deref() == Ok("1")`.
    pub use_dense: bool,

    /// `HF2Q_KV_LCP_RESUME` — ADR-017 Phase E option (a) LCP partial-prefill
    /// resume. **Default ON**; opt-out via `HF2Q_KV_LCP_RESUME=0` / `=false`
    /// / `=off`. When ON + the engine's LcpRegistry lookup returns `Some(k)`
    /// + multimodal bail passes (`soft_tokens.is_empty()`) + capacity
    /// precondition holds (cached linear_capacity ≥ new request's
    /// seq_len + max_decode_tokens) + `HF2Q_USE_DENSE=1` is also set
    /// (TQ-packed kv_caches not safely resumable without separate
    /// restoration), the request bypasses the wholesale
    /// `cache.write_pos = 0` reset at `forward_prefill.rs:445-448`
    /// and resumes from token K — reusing the cached
    /// `dense_kvs[*][0..K)` in place.
    ///
    /// Auto-disable: when this flag is true via default-on but
    /// `HF2Q_USE_DENSE=0`, the engine gate auto-disables LCP and logs
    /// exactly one warning per process (see `engine.rs::warn_lcp_resume_without_dense`).
    /// Operators who explicitly set `HF2Q_KV_LCP_RESUME=1` override
    /// the auto-disable (explicit opt-in always wins).
    ///
    /// R-C4-LCP byte-identity at 5 K fractions (iter-5) and R-P7
    /// multi-turn-chat speedup (iter-6) gates both passed; default-ON
    /// promotion landed in the E.a default-on iter.
    pub kv_lcp_resume: bool,

    /// `HF2Q_KV_LCP_LONG_RESUME` — enables LCP partial-prefill resume
    /// for prompts where `prompt_len > sliding_window` on Gemma 4 (and
    /// other sliding-window models). Default OFF.
    ///
    /// ADR-017 Phase E.a iter-3.6 — opt-in extension to iter-3.5c's
    /// prefill-wrap restriction. When ON (and `HF2Q_KV_LCP_RESUME=1`
    /// and `HF2Q_USE_DENSE=1`), sliding layers allocate LINEAR buffers
    /// (`cap = max(sw, prompt_len + max_decode_tokens)`) instead of
    /// ring buffers (`cap = sw`); the per-token KV write uses
    /// `slot = tok_i` (no `% sw` wrap); the flash_attn_vec dispatch
    /// uses `mask_type=2 + sliding_window=sw` instead of
    /// `mask_type=1 + ring`. The kernel applies sliding-window
    /// masking based on slot index (which now equals logical
    /// position because the buffer is linear) — semantically
    /// equivalent to the ring path but without wrap.
    ///
    /// Also lifts the `prompt_len <= sliding_window` skip in the
    /// engine's prefill-wrap guards (`engine.rs:4516` non-streaming +
    /// `engine.rs:7027` streaming) and the `seq_len <= sw` predicate
    /// in `forward_prefill.rs:~1820` (snapshot creation guard).
    ///
    /// Memory cost: per cached entry, sliding-layer K+V grows from
    /// `8 × sw × 256 × 2 × 2 = 8 MB` per layer to
    /// `8 × N × 256 × 2 × 2 = 64 MB` per layer at N=8K (verified
    /// estimate on Gemma 4 26B). 30 layers ⇒ ~1.9 GB extra resident
    /// per cached entry. With registry capacity=1, total extra
    /// resident is bounded.
    ///
    /// References:
    ///   - flash_attn_vec.metal:166-170 (kernel mask_type=2 impl)
    ///   - forward_mlx.rs:2632-2635 (Chesterton's fence: ring vs
    ///     linear masking semantics)
    ///   - docs/research/adr017-iter36-phaseB-architecture-2026-05-05.md
    pub kv_lcp_long_resume: bool,

    /// `HF2Q_KV_LCP_CHUNKED_PREFILL` — when ON, Qwen 3.5/3.6 prefill
    /// runs in fixed-size chunks of `kv_lcp_deltanet_checkpoint_stride`
    /// tokens instead of one monolithic call. **Default ON**; opt-out via
    /// `HF2Q_KV_LCP_CHUNKED_PREFILL=0` / `=false` / `=off`.
    ///
    /// Both `kv_lcp_resume` and `kv_lcp_chunked_prefill` must be ON
    /// together for Qwen 3.5/3.6 to get any LCP benefit: the engine
    /// gate at `engine_qwen35.rs:1083-1088` computes
    /// `chunked_eligible = lcp_resume_enabled && kv_lcp_chunked_prefill`.
    /// Per decisions.json Q6: both flags flip together.
    ///
    /// ADR-017 Phase B-hybrid.2a — chunked prefill is the foundation
    /// for SSM-state checkpointing + partial-prefill resume. Each
    /// chunk's call propagates the recurrent state via the `kv_cache`
    /// (DeltaNet conv_state + recurrent state pingpong; full-attn
    /// `current_len[0]` cursor). The cumulative effect of multiple
    /// chunked calls MUST be byte-identical to a single monolithic
    /// call — that's the falsifier test
    /// `tests/lcp_qwen35_chunked_prefill.rs::
    ///  phase_b2a_chunked_vs_monolithic_byte_identity`.
    ///
    /// Cost trade-off: chunked dispatch adds ~5-15 % wall (extra
    /// kernel launches per chunk; extra `lm_head` matmul per chunk's
    /// last token). Phase B.2 lifts this trade by actually engaging
    /// LCP resume (skipping [0..K_aligned) chunks entirely on the
    /// shared-prefix path), netting a positive speedup overall.
    pub kv_lcp_chunked_prefill: bool,

    /// `HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE` — default 1024 — the
    /// stride between SSM-state checkpoints during chunked prefill.
    /// MUST be a positive multiple of `FIXED_BT = 64` (per
    /// `chunk_gated_delta_rule` precondition at gpu_delta_net.rs:1078).
    /// Default 1024 = 16 internal chunks per stride.
    ///
    /// Memory cost: per cached entry, `ceil(N / stride)` checkpoints
    /// at ~96 MB each (Qwen 3.6 27B 48 DeltaNet layers × ~2 MB
    /// recurrent state). For N=8192, stride=1024: 8 checkpoints =
    /// ~768 MB per cached entry. Capacity=1 registry ⇒ bounded.
    pub kv_lcp_deltanet_checkpoint_stride: usize,

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

    /// `HF2Q_FUSED_END_OF_LAYER=1` — replace the 2 sequential
    /// fused_norm_add dispatches at end-of-layer (post-FF norm 2 +
    /// end-of-layer FINAL) with the iter-217 fused single-dispatch
    /// kernel `fused_post_ff_norm2_endlayer_f32`.  Bisect-confirmed
    /// (iter-208) +2.7% throughput target.  Parity test PASS at
    /// rel_error < 1e-5 (iter-218).  Default-OFF until production
    /// bench validates.
    pub fused_end_of_layer: bool,

    /// `HF2Q_FUSED_MOE_WSUM_END_LAYER_V2=1` — fuse `moe_weighted_sum` INTO
    /// `fused_post_ff_norm2_endlayer_v2` (Path A end-of-layer), eliminating
    /// 1 dispatch + moe_accum round-trip per layer (30 dispatches/decode-token
    /// on gemma4).  Requires `HF2Q_FUSED_END_OF_LAYER=1` AND `dim % 4 == 0`.
    /// Parity byte-identical at gemma4 prod shape (dim=2816, top_k=8) — see
    /// test_fused_moe_wsum_endlayer_v2_parity.rs.
    /// **Default-OFF**: ADR-029 iter-3 re-test on adr-029 HEAD with full
    /// default-flag stack (LMHEAD_Q6K + Q6K_MV_NR2 + Q6K_ID_MV_NR2 all on)
    /// produces byte-identical 50-tok haiku output on gemma4-APEX-Q5_K_M
    /// — coherence regresses are no longer reproducible at HEAD (the iter-367
    /// claim is stale).  Throughput at HEAD: 74.4 t/s median (σ-pct 0.11%)
    /// vs 75.0 baseline → **-0.8% throughput regression**, mirroring the
    /// iter-2 H6 fused-triple-norm pattern (fewer-larger Metal kernels
    /// regress over more-smaller at gemma4 decode shape on M5 Max).
    /// Standing: leave default-off; mantra "code + test == truth" — the
    /// kernel is functionally correct but loses on launch-overhead-vs-
    /// per-call-cost balance at this shape.
    pub fused_moe_wsum_end_layer_v2: bool,

    /// `HF2Q_FUSED_TRIPLE_NORM=1` — replace the per-layer pair
    /// `fused_norm_add(hidden, attn_out, post_attn_w → residual)` +
    /// 3× `rms_norm(residual, w_a/b/c → out_a/b/c)` with the single
    /// `fused_post_attn_triple_norm_f32` kernel.  Saves 3
    /// dispatches/layer × 30 layers = 90 dispatches/token on gemma4
    /// decode.  Kernel already exists in mlx-native (used by batched
    /// prefill) and is byte-identical with the unfused path on prefill
    /// fixtures.
    /// **Default-OFF**: ADR-029 iter-1 H6 test on gemma4-APEX-Q5_K_M
    /// at HEAD with default-flag stack: coherence byte-identical
    /// (50-tok haiku), throughput **72.9 t/s** median (σ-pct 0.05%, n=5)
    /// vs 75.0 baseline = **-2.8% regression**.  The fused single-dispatch
    /// kernel is correct but its per-call cost exceeds the savings from
    /// dropping 4 unfused launches at gemma4's decode shape on M5 Max.
    /// Sibling falsification: HF2Q_FUSED_MOE_WSUM_END_LAYER_V2 above.
    /// Standing decision: leave default-off; the dispatch-fusion lever
    /// class appears to lose on Apple Metal at hidden_size=2816, top_k=8.
    ///
    /// ADR-029 iter-175 Step 1o RE-BENCH at HEAD (post H-E precompile +
    /// FC-promote + q6_K_nr2 + many other landed levers): 2-cycle alt-pair
    /// tg100 with 60s cool-downs:
    ///   A (default): 96.1, 96.1 → mean 96.10 t/s
    ///   B (HF2Q_FUSED_TRIPLE_NORM=1): 89.6, 92.7 → mean 91.15 t/s
    ///   Delta: -5.15% (BIGGER regression than original -2.8%)
    /// The unfused path benefited more from the accumulated levers than
    /// the fused path; the gap WIDENED.  Doubly-falsified at HEAD.
    pub fused_triple_norm: bool,

    /// `HF2Q_KV_DUAL_LEGACY=1` — force the legacy 2-dispatch K+V cache
    /// copy path (one for K, one for V) instead of the iter-145 fused
    /// single-dispatch dual kernel. ADR-028 forensic A/B switch; both
    /// paths are bit-identical by mlx-native unit tests
    /// (`test_kv_cache_copy_batch_f32_kv_dual_byte_identity` +
    /// `test_kv_cache_copy_batch_f32_to_f16_kv_dual_byte_identity`).
    pub kv_dual_legacy: bool,

    /// `HF2Q_HB_DUAL_LEGACY=1` — force the legacy 2-dispatch
    /// `dispatch_hadamard_quantize_kv_hb` path (one for K, one for V)
    /// instead of the iter-148 fused
    /// `dispatch_hadamard_quantize_kv_hb_dual`. ADR-028 forensic A/B
    /// switch; both paths byte-identical by mlx-native unit test
    /// (`test_hadamard_quantize_kv_hb_dual_byte_identity_d256`).
    pub hb_dual_legacy: bool,

    /// `HF2Q_TQ_FAST_FUSED_KV=1` — enable the ADR-028 iter-485 (Phase 7d
    /// / H4) fused 4-bit K+V single-position TQ encoder. When set, the
    /// gemma4 decode path at `forward_mlx::run_decode_step_layer` swaps
    /// the two consecutive `dispatch_hadamard_quantize_kv` calls (K then
    /// V) for a single `dispatch_hadamard_quantize_kv_fast_dual` launch.
    /// Default OFF (opt-in until decode-bench ≥+3% gate clears); both
    /// paths byte-identical by mlx-native unit test
    /// (`test_hadamard_quantize_kv_fast_dual_byte_identity_d256`).
    pub tq_fast_fused_kv: bool,

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
    skip_dense_mlp: bool,
    skip_moe_experts: bool,
    skip_moe_swiglu: bool,
    skip_head_norm_rope: bool,
    skip_post_attn_norm: bool,
    skip_weighted_sum: bool,
    skip_end_of_layer: bool,
    skip_end_of_layer_final: bool,
    skip_attn_qkv: bool,
    skip_o_proj: bool,
    skip_routing: bool,
    skip_v_norm: bool,
    lmhead_rerank_disabled: bool,
    chunk_scan_prefill: bool,
}

impl InvestigationEnv {
    /// Parse every investigation env var from the current process
    /// environment and apply the ack gate to ack-required toggles.
    /// Called exactly once via [`INVESTIGATION_ENV`]'s `LazyLock`.
    pub fn from_env() -> Self {
        let raw = RawAckIntent {
            f16_kv: env_eq_one("HF2Q_F16_KV"),
            // ADR-028 iter-344: default-ON (was env_eq_one).  Decoupled
            // from UNSAFE ack at the activation site below.
            batched_prefill: env_default_true("HF2Q_BATCHED_PREFILL"),
            skip_tq_encode: env_eq_one("HF2Q_SKIP_TQ_ENCODE"),
            skip_tq_sdpa: env_eq_one("HF2Q_SKIP_TQ_SDPA"),
            skip_dense_mlp: env_eq_one("HF2Q_SKIP_DENSE_MLP"),
            skip_moe_experts: env_eq_one("HF2Q_SKIP_MOE_EXPERTS"),
            skip_moe_swiglu: env_eq_one("HF2Q_SKIP_MOE_SWIGLU"),
            skip_head_norm_rope: env_eq_one("HF2Q_SKIP_HEAD_NORM_ROPE"),
            skip_post_attn_norm: env_eq_one("HF2Q_SKIP_POST_ATTN_NORM"),
            skip_weighted_sum: env_eq_one("HF2Q_SKIP_WEIGHTED_SUM"),
            skip_end_of_layer: env_eq_one("HF2Q_SKIP_END_OF_LAYER"),
            skip_end_of_layer_final: env_eq_one("HF2Q_SKIP_END_OF_LAYER_FINAL"),
            skip_attn_qkv: env_eq_one("HF2Q_SKIP_ATTN_QKV"),
            skip_o_proj: env_eq_one("HF2Q_SKIP_O_PROJ"),
            skip_routing: env_eq_one("HF2Q_SKIP_ROUTING"),
            skip_v_norm: env_eq_one("HF2Q_SKIP_V_NORM"),
            lmhead_rerank_disabled: matches!(
                env::var("HF2Q_LMHEAD_RERANK").as_deref(),
                Ok("0")
            ),
            chunk_scan_prefill: env_eq_one("HF2Q_CHUNK_SCAN_PREFILL"),
        };
        let ack = env_eq_one("HF2Q_UNSAFE_EXPERIMENTS");

        Self {
            // Ack-required — effective value is raw AND ack.
            f16_kv: raw.f16_kv && ack,
            // ADR-028 iter-344: batched_prefill DECOUPLED from ack.
            // Operator iter-76 signed off on the L6 MoE sliding_wrap
            // deferral; iter-343 falsifier-tested at pp3813 (5/5 short
            // coherence + 1000-tok + 4K long-context all coherent at
            // HEAD with iter-326+331+337+338 stack).  Promoting to
            // first-class default-ON; opt-out via env=0/false/off.
            batched_prefill: raw.batched_prefill,
            skip_tq_encode: raw.skip_tq_encode && ack,
            skip_tq_sdpa: raw.skip_tq_sdpa && ack,
            skip_dense_mlp: raw.skip_dense_mlp && ack,
            skip_moe_experts: raw.skip_moe_experts && ack,
            skip_moe_swiglu: raw.skip_moe_swiglu && ack,
            skip_head_norm_rope: raw.skip_head_norm_rope && ack,
            skip_post_attn_norm: raw.skip_post_attn_norm && ack,
            skip_weighted_sum: raw.skip_weighted_sum && ack,
            skip_end_of_layer: raw.skip_end_of_layer && ack,
            skip_end_of_layer_final: raw.skip_end_of_layer_final && ack,
            skip_attn_qkv: raw.skip_attn_qkv && ack,
            skip_o_proj: raw.skip_o_proj && ack,
            skip_routing: raw.skip_routing && ack,
            skip_v_norm: raw.skip_v_norm && ack,
            lmhead_rerank_disabled: raw.lmhead_rerank_disabled && ack,
            chunk_scan_prefill: raw.chunk_scan_prefill && ack,

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
            dump_pre_quant_layers: env_usize_list("HF2Q_DUMP_PRE_QUANT_LAYERS"),
            dump_pre_quant_positions: env_usize_list("HF2Q_DUMP_PRE_QUANT_POSITIONS"),
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

            // ADR-028 Phase 10 (iter-347) / ADR-029 iter-13: hybrid F16-K + TQ-HB-V.
            // **Default ON** after ADR-029 iter-12 confirmation (3-trial fresh
            // bench: +9.5% gemma4 throughput, byte-class-coherent output,
            // 0.756× → 0.805× peer ratio). Opt-out via =0 / =false / =off
            // (returns legacy TQ-HB path); coherence parity preserved either way.
            hybrid_kv: env_default_true("HF2Q_HYBRID_KV"),

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

            // Wave 5a Qwen3.6 autoregressive opt-in (no ack gate — does not
            // alter forward-pass math, just unblocks dispatch).
            qwen36_autoreg: env_eq_one("HF2Q_QWEN36_AUTOREG"),

            // Wave 5b.20 `HF2Q_GQA_EXPAND_LEGACY` env parser deleted in
            // W-5b.21 (30/30 PASS at PP4106 — gate removed; production now
            // unconditionally runs the GPU `dispatch_repeat_tiled_f32` path).

            // SDPA regime selectors.
            use_dense: matches!(env::var("HF2Q_USE_DENSE").as_deref(), Ok("1")),
            layer_policy: env::var("HF2Q_LAYER_POLICY").ok(),

            // ADR-017 Phase E.a default-on — LCP partial-prefill resume.
            // Default ON; opt-out via HF2Q_KV_LCP_RESUME=0 / =false / =off.
            // See struct field doc for full contract.
            kv_lcp_resume: env_default_true("HF2Q_KV_LCP_RESUME"),
            // ADR-017 Phase E.a iter-3.6 — long-prompt LCP resume (lifts
            // iter-3.5c sliding-ring prefill-wrap restriction). Default
            // OFF; opt-in via `HF2Q_KV_LCP_LONG_RESUME=1`. See struct
            // field doc for full contract.
            kv_lcp_long_resume: env_eq_one("HF2Q_KV_LCP_LONG_RESUME"),
            // ADR-017 Phase E.a default-on — chunked prefill toggle.
            // Default ON; opt-out via HF2Q_KV_LCP_CHUNKED_PREFILL=0 / =false / =off.
            kv_lcp_chunked_prefill: env_default_true("HF2Q_KV_LCP_CHUNKED_PREFILL"),
            // ADR-017 Phase B-hybrid.2a — checkpoint stride (default
            // 1024). Constrained to a positive multiple of 64 by the
            // chunk_gated_delta_rule precondition. We don't enforce the
            // multiple-of-64 here at parse time; the chunked-prefill
            // call site validates and rounds up if necessary.
            kv_lcp_deltanet_checkpoint_stride: env::var(
                "HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE",
            )
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(1024),

            // Gate H release-check plumbing (ADR-007 §853-866; iter-108a).
            emit_nll: env_eq_one("HF2Q_EMIT_NLL"),
            decode_emit_tokens: env_eq_one("HF2Q_DECODE_EMIT_TOKENS"),
            decode_input_tokens: env_u32_list_space("HF2Q_DECODE_INPUT_TOKENS"),

            // Profiling / timing.
            mlx_timing: env::var("HF2Q_MLX_TIMING").is_ok(),
            split_timing: env_eq_one("HF2Q_SPLIT_TIMING"),
            fused_triple_norm: env_eq_one("HF2Q_FUSED_TRIPLE_NORM"),
            // ADR-028 iter-326: default-flipped to ON (operator REFRAME #2).
            // Opt out with `HF2Q_FUSED_END_OF_LAYER=0` / `=false` / `=off`.
            fused_end_of_layer: env_default_true("HF2Q_FUSED_END_OF_LAYER"),
            // ADR-028 iter-367: default-OFF until coherence-with-stack debug.
            fused_moe_wsum_end_layer_v2: env_eq_one("HF2Q_FUSED_MOE_WSUM_END_LAYER_V2"),
            kv_dual_legacy: env_eq_one("HF2Q_KV_DUAL_LEGACY"),
            hb_dual_legacy: env_eq_one("HF2Q_HB_DUAL_LEGACY"),
            // ADR-028 iter-485 (Phase 7d / H4): default-OFF opt-in.
            tq_fast_fused_kv: env_eq_one("HF2Q_TQ_FAST_FUSED_KV"),
            mlx_kernel_profile: env_eq_one("HF2Q_MLX_KERNEL_PROFILE"),
            mlx_profile: env_eq_one("HF2Q_MLX_PROFILE"),

            // ADR-031 Phase B: parallel-encode gate + interaction guard.
            // Both vars are snapshotted once at LazyLock init so the
            // parallel_encode_enabled() interaction guard always sees the
            // same values as the GpuContext::new() call site.
            parallel_encode_raw: env_eq_one("HF2Q_PARALLEL_ENCODE"),
            per_layer_disp_raw: env_eq_one("HF2Q_PER_LAYER_DISP"),
            parallel_encode_kv_threshold: env::var("HF2Q_PARALLEL_ENCODE_KV_THRESHOLD")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(512),

            unsafe_experiments_acked: ack,
            raw,
        }
    }

    /// Returns true when parallel-encode is active for this process:
    ///   - `HF2Q_PARALLEL_ENCODE=1` was set at process start, AND
    ///   - `HF2Q_PER_LAYER_DISP=1` was NOT set (interaction guard).
    ///
    /// When `HF2Q_PER_LAYER_DISP=1` overrides the parallel request,
    /// a once-per-process `tracing::warn!` is emitted via a `Once` guard so
    /// the operator immediately sees why PARALLEL=ON had no effect.
    pub fn parallel_encode_enabled(&self) -> bool {
        if !self.parallel_encode_raw {
            return false;
        }
        if self.per_layer_disp_raw {
            // Interaction guard: HF2Q_PER_LAYER_DISP=1 races on the global
            // mlx_native::dispatch_count() counter with two CPU encode threads.
            // Force parallel OFF and warn once.
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    "HF2Q_PARALLEL_ENCODE=1 is set but HF2Q_PER_LAYER_DISP=1 is also set — \
                     parallel encode DISABLED (per-layer dispatch counter is process-global; \
                     two encode threads would race on it). Unset HF2Q_PER_LAYER_DISP to enable \
                     parallel encode."
                );
            });
            return false;
        }
        true
    }

    /// Resolve the dual-buffer split point against the current model's
    /// layer count. Preserves the original inline behavior exactly:
    ///
    /// - Env unset → default to `Some(2)` (split after layer 2).
    /// - Env set to a parsable `usize` → `Some(n)` only if
    ///   `n > 0 && n < num_layers`, else `None`.
    /// - Env set but not a `usize` → `None`.
    ///
    /// ADR-028 iter-373: default flipped 3 → 2.  3-cycle sweep on gemma4
    /// at iter-321 stack: split=1 73.87 (-1.1%), split=2 74.83 (+0.18%),
    /// split=3 74.70 (baseline), split=5 74.47 (-0.31%).  split=2 hits
    /// the sweet spot — buf0 has 2 layers' worth of GPU work to overlap
    /// with CPU encoding the remaining 28 layers.  split=1 has too little
    /// work in buf0; split=5+ delays GPU start unnecessarily.
    ///
    /// ADR-028 iter-374 NOTE: this method returns the FIRST split point
    /// for backward compatibility.  For multi-split support (3+ buffers),
    /// use [`Self::dual_buffer_splits`] which returns `Vec<usize>` parsed
    /// from comma-separated env values like `HF2Q_DUAL_BUFFER=2,10,20`.
    pub fn dual_buffer_split(&self, num_layers: usize) -> Option<usize> {
        self.dual_buffer_splits(num_layers).first().copied()
    }

    /// Resolve all dual-buffer split points (sorted, unique, in-range).
    ///
    /// ADR-028 iter-374: multi-split support.  Each split point causes a
    /// `commit()` and re-`begin()` of the encoder mid-forward.  Returns
    /// empty Vec when fully disabled.
    ///
    /// Env parsing:
    /// - Env unset → `vec![2]` (single default split, matches iter-373).
    /// - Env set to a single int "N" → `vec![N]` if `0 < N < num_layers`.
    /// - Env set to comma-separated "N1,N2,..." → sorted unique in-range.
    /// - Env set to "0" or invalid → empty Vec (disabled).
    pub fn dual_buffer_splits(&self, num_layers: usize) -> Vec<usize> {
        match self.dual_buffer_raw.as_deref() {
            None => vec![2],
            Some(v) => {
                let mut splits: Vec<usize> = v
                    .split(',')
                    .filter_map(|tok| tok.trim().parse::<usize>().ok())
                    .filter(|&n| n > 0 && n < num_layers)
                    .collect();
                splits.sort_unstable();
                splits.dedup();
                splits
            }
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
                "DEPRECATED: gemma4-incoherent at random N (ADR-028 iter-234, \
                 random `<pad>` emission); qwen3.6 no-op (no perf gain). \
                 Path E+G recommended instead",
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
        if self.chunk_scan_prefill {
            active_unsafe.push((
                "HF2Q_CHUNK_SCAN_PREFILL=1",
                "Wave 5b iter 5 chunk-pipeline prefill at seq_len > 64; \
                 sourdough/walk-bar parity validation pending",
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
        if self.raw.chunk_scan_prefill && !self.chunk_scan_prefill {
            refused.push((
                "HF2Q_CHUNK_SCAN_PREFILL=1",
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
        if self.qwen36_autoreg {
            diagnostics.push(
                "HF2Q_QWEN36_AUTOREG=1 (Wave 5a opt-in: autoregressive only; long-prefill SOTA \
                 deferred to W-5b chunk-scan kernel)".into(),
            );
        }
        // Wave 5b.20 `HF2Q_GQA_EXPAND_LEGACY` activate-diagnostic deleted in
        // W-5b.21 alongside the field + env parser (30/30 cross-path
        // determinism PASS at PP4106).
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

/// Default-ON boolean: returns `true` when the env var is unset OR set to
/// a truthy value (`"1"`, `"true"`, `"on"`, case-insensitive); returns
/// `false` only when explicitly set to a falsy value (`"0"`, `"false"`,
/// `"off"`, case-insensitive). Any other non-empty unrecognized value is
/// treated as `true` (permissive default-on: if someone sets a var they
/// probably want it on). Used for feature flags that are default-ON and
/// opt-out via `=0` / `=false` / `=off`.
fn env_default_true(name: &str) -> bool {
    match env::var(name).ok().as_deref() {
        // Env unset → default ON.
        None => true,
        // Truthy: "1", "true", "on" (case-insensitive) → ON.
        Some(v) if v.eq_ignore_ascii_case("1")
            || v.eq_ignore_ascii_case("true")
            || v.eq_ignore_ascii_case("on") => true,
        // Falsy: "0", "false", "off" (case-insensitive) → OFF.
        Some(v) if v.eq_ignore_ascii_case("0")
            || v.eq_ignore_ascii_case("false")
            || v.eq_ignore_ascii_case("off") => false,
        // Non-empty unrecognized value → permissive default-on.
        Some(_) => true,
    }
}

/// Returns `true` iff `HF2Q_KV_LCP_RESUME` is set to exactly `"1"`.
///
/// Used in the auto-disable logic at the engine gate: when `kv_lcp_resume`
/// is `true` from `env_default_true` but `HF2Q_USE_DENSE=0`, we need to
/// distinguish "user explicitly requested LCP" (env == "1") from "user
/// never touched the env" (default-on). Only the explicit-`"1"` path
/// overrides the auto-disable; default-on is silently disabled on dense=0.
pub fn is_kv_lcp_resume_explicitly_one() -> bool {
    std::env::var("HF2Q_KV_LCP_RESUME").as_deref() == Ok("1")
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

    // ── qwen36_autoreg (Wave 5a) ─────────────────────────────────────

    #[test]
    fn qwen36_autoreg_default_when_unset() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::new(&["HF2Q_QWEN36_AUTOREG"]);
        assert!(
            !InvestigationEnv::from_env().qwen36_autoreg,
            "unset => default false (Qwen3.6 GGUF dispatch must error out without explicit opt-in)"
        );
    }

    #[test]
    fn qwen36_autoreg_enabled_by_one() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_QWEN36_AUTOREG"]);
        guard.set("HF2Q_QWEN36_AUTOREG", "1");
        assert!(InvestigationEnv::from_env().qwen36_autoreg);
    }

    #[test]
    fn qwen36_autoreg_not_enabled_by_other_values() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_QWEN36_AUTOREG"]);
        for bad in &["0", "true", "yes", "autoreg", ""] {
            guard.set("HF2Q_QWEN36_AUTOREG", bad);
            assert!(
                !InvestigationEnv::from_env().qwen36_autoreg,
                "value {:?} must not enable qwen36_autoreg",
                bad
            );
        }
    }

    #[test]
    fn qwen36_autoreg_does_not_require_unsafe_ack() {
        // Wave 5a: this is a category-4 dispatch gate, not a forward-pass-math
        // toggle. Setting `HF2Q_QWEN36_AUTOREG=1` alone (no UNSAFE_EXPERIMENTS
        // ack) MUST take effect. If the future moves it under the ack umbrella
        // this test will catch the regression.
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_QWEN36_AUTOREG", "HF2Q_UNSAFE_EXPERIMENTS"]);
        guard.set("HF2Q_QWEN36_AUTOREG", "1");
        // Note: HF2Q_UNSAFE_EXPERIMENTS deliberately not set.
        let env = InvestigationEnv::from_env();
        assert!(env.qwen36_autoreg);
        assert!(!env.unsafe_experiments_acked);
    }

    // ── hybrid_kv (ADR-028 Phase 10 / iter-347) ──────────────────────

    #[test]
    fn hybrid_kv_default_when_unset() {
        // ADR-029 iter-13 default-flip: unset => ON (was OFF pre-iter-13).
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::new(&["HF2Q_HYBRID_KV"]);
        assert!(
            InvestigationEnv::from_env().hybrid_kv,
            "unset => default true after ADR-029 iter-13 default-flip"
        );
    }

    #[test]
    fn hybrid_kv_enabled_by_one() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_HYBRID_KV"]);
        guard.set("HF2Q_HYBRID_KV", "1");
        assert!(InvestigationEnv::from_env().hybrid_kv);
    }

    #[test]
    fn hybrid_kv_disabled_by_zero_false_off() {
        // ADR-029 iter-13: env_default_true semantics. Explicit falsy values
        // turn HYBRID off (legacy TQ-HB path); other values keep it on.
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_HYBRID_KV"]);
        for falsy in &["0", "false", "off", "FALSE", "Off"] {
            guard.set("HF2Q_HYBRID_KV", falsy);
            assert!(
                !InvestigationEnv::from_env().hybrid_kv,
                "value {:?} must disable hybrid_kv (env_default_true falsy)",
                falsy
            );
        }
    }

    #[test]
    fn hybrid_kv_permissive_truthy() {
        // ADR-029 iter-13: env_default_true semantics. "1"/"true"/"on" and
        // unrecognized non-empty values all leave HYBRID on.
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_HYBRID_KV"]);
        for truthy in &["1", "true", "on", "yes", "hybrid", ""] {
            guard.set("HF2Q_HYBRID_KV", truthy);
            assert!(
                InvestigationEnv::from_env().hybrid_kv,
                "value {:?} keeps hybrid_kv on (permissive default-true)",
                truthy
            );
        }
    }

    #[test]
    fn hybrid_kv_does_not_require_unsafe_ack() {
        // ADR-028 Phase 10: this is a memory-layout selector, not a math
        // skip. Setting `HF2Q_HYBRID_KV=1` (or leaving unset under ADR-029
        // iter-13 default-true) takes effect — the SDPA dispatcher routes
        // to the wired `flash_attn_vec_hybrid` kernel without needing the
        // UNSAFE experiment ack.
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_HYBRID_KV", "HF2Q_UNSAFE_EXPERIMENTS"]);
        guard.set("HF2Q_HYBRID_KV", "1");
        // Note: HF2Q_UNSAFE_EXPERIMENTS deliberately not set.
        let env = InvestigationEnv::from_env();
        assert!(env.hybrid_kv);
        assert!(!env.unsafe_experiments_acked);
    }

    // ── chunk_scan_prefill (Wave 5b iter 5) ──────────────────────────

    #[test]
    fn chunk_scan_prefill_default_when_unset() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::new(&[
            "HF2Q_CHUNK_SCAN_PREFILL",
            "HF2Q_UNSAFE_EXPERIMENTS",
        ]);
        assert!(
            !InvestigationEnv::from_env().chunk_scan_prefill,
            "unset => default false (chunk-pipeline prefill must be opt-in)"
        );
    }

    #[test]
    fn chunk_scan_prefill_requires_unsafe_ack() {
        // Ack-required gate: HF2Q_CHUNK_SCAN_PREFILL=1 alone does NOT take
        // effect. Both vars must be set for the field to read true.
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&[
            "HF2Q_CHUNK_SCAN_PREFILL",
            "HF2Q_UNSAFE_EXPERIMENTS",
        ]);
        guard.set("HF2Q_CHUNK_SCAN_PREFILL", "1");
        // Note: HF2Q_UNSAFE_EXPERIMENTS deliberately not set.
        let env = InvestigationEnv::from_env();
        assert!(
            !env.chunk_scan_prefill,
            "raw intent without ack must be REFUSED (effective false)"
        );
        assert!(env.raw.chunk_scan_prefill, "raw intent must be captured for REFUSED reporting");
        assert!(!env.unsafe_experiments_acked);
    }

    #[test]
    fn chunk_scan_prefill_enabled_with_ack() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&[
            "HF2Q_CHUNK_SCAN_PREFILL",
            "HF2Q_UNSAFE_EXPERIMENTS",
        ]);
        guard.set("HF2Q_CHUNK_SCAN_PREFILL", "1");
        guard.set("HF2Q_UNSAFE_EXPERIMENTS", "1");
        let env = InvestigationEnv::from_env();
        assert!(env.chunk_scan_prefill, "1 + ack => effective true");
        assert!(env.unsafe_experiments_acked);
    }

    #[test]
    fn chunk_scan_prefill_not_enabled_by_other_values() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&[
            "HF2Q_CHUNK_SCAN_PREFILL",
            "HF2Q_UNSAFE_EXPERIMENTS",
        ]);
        guard.set("HF2Q_UNSAFE_EXPERIMENTS", "1");
        // env_eq_one accepts only "1" — these must all be rejected.
        for bad in &["0", "true", "yes", "TRUE", "on", ""] {
            guard.set("HF2Q_CHUNK_SCAN_PREFILL", bad);
            assert!(
                !InvestigationEnv::from_env().chunk_scan_prefill,
                "value {:?} must not enable chunk_scan_prefill",
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

    // ── env_default_true ────────────────────────────────────────────────────

    /// When the env var is unset, `env_default_true` must return `true`
    /// (default-on semantics).
    #[test]
    fn env_default_true_unset_returns_true() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::new(&["HF2Q_KV_LCP_RESUME"]);
        assert!(
            env_default_true("HF2Q_KV_LCP_RESUME"),
            "unset env var must return true (default-on)"
        );
    }

    /// `=1` must return `true`.
    #[test]
    fn env_default_true_eq_one_returns_true() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_KV_LCP_RESUME"]);
        guard.set("HF2Q_KV_LCP_RESUME", "1");
        assert!(
            env_default_true("HF2Q_KV_LCP_RESUME"),
            "=1 must return true"
        );
    }

    /// `=0` must return `false` (explicit opt-out).
    #[test]
    fn env_default_true_eq_zero_returns_false() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_KV_LCP_RESUME"]);
        guard.set("HF2Q_KV_LCP_RESUME", "0");
        assert!(
            !env_default_true("HF2Q_KV_LCP_RESUME"),
            "=0 must return false (opt-out)"
        );
    }

    /// `=true` (case-insensitive) must return `true`.
    #[test]
    fn env_default_true_eq_true_string_returns_true() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_KV_LCP_RESUME"]);
        for v in &["true", "True", "TRUE"] {
            guard.set("HF2Q_KV_LCP_RESUME", v);
            assert!(
                env_default_true("HF2Q_KV_LCP_RESUME"),
                "={v} must return true"
            );
        }
    }

    /// `=off` (case-insensitive) must return `false`.
    #[test]
    fn env_default_true_eq_off_returns_false() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_KV_LCP_RESUME"]);
        for v in &["off", "OFF", "Off"] {
            guard.set("HF2Q_KV_LCP_RESUME", v);
            assert!(
                !env_default_true("HF2Q_KV_LCP_RESUME"),
                "={v} must return false (opt-out)"
            );
        }
    }

    /// `=false` (case-insensitive) must return `false`.
    #[test]
    fn env_default_true_eq_false_string_returns_false() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_KV_LCP_RESUME"]);
        for v in &["false", "False", "FALSE"] {
            guard.set("HF2Q_KV_LCP_RESUME", v);
            assert!(
                !env_default_true("HF2Q_KV_LCP_RESUME"),
                "={v} must return false (opt-out)"
            );
        }
    }

    /// `kv_lcp_resume` field is `true` when env is unset (default-on via
    /// `env_default_true`).
    #[test]
    fn kv_lcp_resume_defaults_true_when_unset() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::new(&["HF2Q_KV_LCP_RESUME"]);
        assert!(
            InvestigationEnv::from_env().kv_lcp_resume,
            "kv_lcp_resume must default to true (default-on)"
        );
    }

    /// `kv_lcp_resume` field is `false` when `HF2Q_KV_LCP_RESUME=0`.
    #[test]
    fn kv_lcp_resume_false_when_zero() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_KV_LCP_RESUME"]);
        guard.set("HF2Q_KV_LCP_RESUME", "0");
        assert!(
            !InvestigationEnv::from_env().kv_lcp_resume,
            "kv_lcp_resume must be false when HF2Q_KV_LCP_RESUME=0"
        );
    }

    /// `kv_lcp_chunked_prefill` field is `true` when env is unset (default-on).
    #[test]
    fn kv_lcp_chunked_prefill_defaults_true_when_unset() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::new(&["HF2Q_KV_LCP_CHUNKED_PREFILL"]);
        assert!(
            InvestigationEnv::from_env().kv_lcp_chunked_prefill,
            "kv_lcp_chunked_prefill must default to true (default-on)"
        );
    }

    /// `is_kv_lcp_resume_explicitly_one` returns true only when the env is
    /// exactly `"1"`.
    #[test]
    fn is_kv_lcp_resume_explicitly_one_contract() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_KV_LCP_RESUME"]);

        // Unset → false (not explicitly set).
        assert!(!is_kv_lcp_resume_explicitly_one(), "unset must return false");

        // "1" → true.
        guard.set("HF2Q_KV_LCP_RESUME", "1");
        assert!(is_kv_lcp_resume_explicitly_one(), r#""1" must return true"#);

        // "true" → false (not the literal "1").
        guard.set("HF2Q_KV_LCP_RESUME", "true");
        assert!(!is_kv_lcp_resume_explicitly_one(), r#""true" must return false (only "1" qualifies)"#);

        // "0" → false.
        guard.set("HF2Q_KV_LCP_RESUME", "0");
        assert!(!is_kv_lcp_resume_explicitly_one(), r#""0" must return false"#);
    }

    // ── parallel_encode_enabled (ADR-031 Phase B) ────────────────────────────

    /// a) Both env vars unset → enabled() returns false.
    #[test]
    fn parallel_encode_enabled_both_unset_returns_false() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::new(&["HF2Q_PARALLEL_ENCODE", "HF2Q_PER_LAYER_DISP"]);
        let env = InvestigationEnv::from_env();
        assert!(!env.parallel_encode_raw, "raw field should be false when unset");
        assert!(!env.per_layer_disp_raw, "per_layer_disp_raw should be false when unset");
        assert!(
            !env.parallel_encode_enabled(),
            "both unset => enabled() must return false (default OFF)"
        );
    }

    /// b) PARALLEL=1, DISP unset → enabled() returns true.
    #[test]
    fn parallel_encode_enabled_parallel_one_disp_unset_returns_true() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_PARALLEL_ENCODE", "HF2Q_PER_LAYER_DISP"]);
        guard.set("HF2Q_PARALLEL_ENCODE", "1");
        let env = InvestigationEnv::from_env();
        assert!(env.parallel_encode_raw, "raw field should be true with =1");
        assert!(!env.per_layer_disp_raw, "per_layer_disp_raw should be false when unset");
        assert!(
            env.parallel_encode_enabled(),
            "PARALLEL=1, DISP unset => enabled() must return true"
        );
    }

    /// c) PARALLEL=1, DISP=1 → enabled() returns false (interaction guard).
    #[test]
    fn parallel_encode_enabled_interaction_guard_blocks_when_disp_set() {
        let _lock = ENV_LOCK.lock().unwrap();
        let guard = EnvGuard::new(&["HF2Q_PARALLEL_ENCODE", "HF2Q_PER_LAYER_DISP"]);
        guard.set("HF2Q_PARALLEL_ENCODE", "1");
        guard.set("HF2Q_PER_LAYER_DISP", "1");
        let env = InvestigationEnv::from_env();
        assert!(env.parallel_encode_raw, "raw field should be true");
        assert!(env.per_layer_disp_raw, "per_layer_disp_raw should be true");
        assert!(
            !env.parallel_encode_enabled(),
            "PARALLEL=1 + DISP=1 => enabled() must return false (interaction guard)"
        );
    }
}
