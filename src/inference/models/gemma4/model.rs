//! Gemma 4 model weights, loader, and core inference utilities.
//!
//! Owns `MlxModelWeights` — the top-level weight container for the Gemma 4
//! mlx-native forward path. Also contains the GGUF loader, DWQ overlay,
//! embed_tokens, and all per-instance setters.
//!
//! Moved from `src/serve/forward_mlx.rs` by ADR-038 Step 3.

use anyhow::{Context, Result};
use mlx_native::{DType, GgmlType, MlxBuffer, MlxDevice};

use crate::debug::INVESTIGATION_ENV;
use crate::inference::dense_bf16_activation::NativeBf16Matrix;
use crate::inference::dense_expert_activation::NativeScalarExpertMatrix;
use crate::inference::models::gemma4::kv_cache::{
    DecodeRegime, DenseKvBuffers, HbKvBuffers, HybridKvBuffers, MlxKvCache,
};
use crate::inference::models::gemma4::native_matrix::{
    load_mapped_projection, map_admitted_tensor_data, preflight_f32_state, preflight_io,
    preflight_projections, resolve_tied_or_explicit, NativeF32StatePlan,
};
use crate::serve::config::{Gemma4Config, LayerType};
use crate::serve::forward_mlx_shared::{
    map_native_gguf_tensor_view, parse_dwq_moe_expert_role, parse_dwq_overlay_metadata,
    parse_dwq_overlay_role, DwqOverlayRole, MlxAffineMoeStack, MlxQWeight, MoeBaseRole,
};
use crate::serve::gpu::{GpuContext, QuantWeightInfo};

// ---------------------------------------------------------------------------
// Weight storage for the mlx-native forward path
// ---------------------------------------------------------------------------

// MlxAffineExtra, MlxQWeight, MlxAffineMoeStack moved to forward_mlx_shared.rs
// (ADR-038 Step 1). Re-exported via the pub use shim above.

/// Per-layer attention weights for the mlx-native forward path.
pub struct MlxAttentionWeights {
    pub q_proj: MlxQWeight,
    pub k_proj: MlxQWeight,
    pub v_proj: Option<MlxQWeight>, // None when k_eq_v
    pub o_proj: MlxQWeight,
    pub q_norm_weight: MlxBuffer,
    pub k_norm_weight: MlxBuffer,
}

/// Per-layer dense MLP weights for the mlx-native forward path.
pub struct MlxMlpWeights {
    pub gate_proj: MlxQWeight,
    pub up_proj: MlxQWeight,
    pub down_proj: MlxQWeight,
}

/// 1-element placeholder allocator helper.
///
/// Wedge-4 / iter-227: produces a tiny MlxBuffer that dense-FFN layers can
/// stash into the `MlxMoeWeights` slot without paying for the GBs of MoE
/// expert tensors that do not exist on disk. Dtype + shape are arbitrary
/// (the dense forward path never reads them); F32 with shape `[1]` is the
/// cheapest valid combination.
fn alloc_one_f32_placeholder(
    mlx_device: &mlx_native::MlxDevice,
    label: &'static str,
) -> Result<MlxBuffer> {
    mlx_device
        .alloc_buffer(std::mem::size_of::<f32>(), mlx_native::DType::F32, vec![1])
        .map_err(|e| anyhow::anyhow!("dense MoE placeholder alloc ({label}): {e}"))
}

/// G4-CFA-5b — load an OPTIONAL per-layer norm tensor, falling back to a
/// 1-element F32 placeholder when the GGUF doesn't carry the tensor.
///
/// Dense Gemma 4 31B GGUFs (e.g. `google_gemma-4-31B-it-Q4_K_M.gguf` from
/// bartowski / unsloth) carry only 4 FFN-related norms per block (`ffn_norm`,
/// `post_ffw_norm`, `attn_q_norm`, `attn_k_norm`). The MoE variants
/// (`pre_ffw_norm_2`, `post_ffw_norm_1`, `post_ffw_norm_2`) are present only
/// in 26B-A4B MoE GGUFs because those tensors back the second parallel FFN
/// branch + per-branch post-norms used by the MoE forward path
/// (`gpu_full_attn.rs:1641-2374`).
///
/// The dense tree-verify path (`gemma4_tree_verify_full_layer_q` at
/// `gpu_full_attn.rs:3099`) reads exactly 2 FFN norms — `pre_feedforward_layernorm`
/// (step B) and `post_feedforward_layernorm` (step G) — never the `_1` / `_2`
/// MoE-only siblings. Returning a 1-element placeholder for the absent norms
/// keeps `MlxLayerNorms` uniform across dense + MoE GGUFs without a
/// `Vec<Option<MlxBuffer>>` ripple change; an accidental dense-vs-MoE misroute
/// would falsify the runtime MoE `stacked_*.is_some()` gate (same iter-227
/// pattern as `MlxMoeWeights::dense_placeholder`).
///
/// Detection rule: the preflight plan records deterministic GGUF-metadata
/// presence (never filename-based, per the iter-227 correctness pin). A
/// present optional norm is an exact mapped F32 view; only absence allocates
/// the documented one-element placeholder.
fn load_optional_norm_or_placeholder(
    state_plan: &NativeF32StatePlan,
    mapped: &mlx_native::gguf::GgufMappedTensorSet<'_>,
    name: &str,
    mlx_device: &mlx_native::MlxDevice,
    label: &'static str,
) -> Result<MlxBuffer> {
    if state_plan.is_present(name)? {
        state_plan
            .load_mapped(mapped, name)
            .map_err(|e| anyhow::anyhow!("{label} ({name}): {e}"))
    } else {
        alloc_one_f32_placeholder(mlx_device, label)
    }
}

// MlxAffineMoeStack moved to forward_mlx_shared.rs (ADR-038 Step 1).

/// Per-expert MoE weights for one layer (quantized, GGML block format).
pub struct MlxMoeWeights {
    /// Stacked gate_up weights: all experts concatenated into `[n_experts, N, packed_K]`.
    /// Used for the fused `quantized_matmul_id_ggml` dispatch.
    pub stacked_gate_up: Option<MlxBuffer>,
    /// Stacked down weights: all experts concatenated into `[n_experts, N, packed_K]`.
    pub stacked_down: Option<MlxBuffer>,
    /// Byte stride between expert slices in the stacked gate_up buffer.
    pub gate_up_expert_stride: u64,
    /// Byte stride between expert slices in the stacked down buffer.
    pub down_expert_stride: u64,
    /// Router projection weight (quantized).
    pub router_proj: MlxQWeight,
    /// Per-expert scale `[num_experts]` F32.
    pub per_expert_scale: MlxBuffer,
    /// GGML quant type for gate_up experts (stored separately so we can
    /// drop the individual expert Vec after stacking).
    pub gate_up_ggml_dtype: mlx_native::GgmlType,
    /// GGML quant type for down experts.
    pub down_ggml_dtype: mlx_native::GgmlType,
    /// Number of experts to select per token.
    pub top_k: usize,
    /// MoE intermediate size per expert.
    pub moe_intermediate_size: usize,
    /// Pre-computed router combined weight: `router_scale[i] * (hidden_size^-0.5)`.
    /// Used by GPU `rms_norm` to compute the router input in one dispatch:
    ///   `output = unit_norm(residual) * router_combined_weight`
    /// This replaces the 3-step CPU sequence: unit_norm → scale → mul.
    pub router_combined_weight: MlxBuffer,
    /// ADR-020 AC#5 Iter C2.2 — optional DWQ-overlay-applied affine
    /// stacks, replacing `stacked_gate_up` + `stacked_down` for the
    /// qwen35moe MoE dispatch path (Iter C2.3 wires the routing).
    /// `gate_up_affine` covers the FUSED gate+up case (qwen3.5 GGUF
    /// `ffn_gate_up_exps`); `gate_affine` + `up_affine` cover the
    /// SEPARATE case (uncommon — added for completeness, not yet
    /// produced by hf2q dwq-train).
    pub gate_up_affine: Option<MlxAffineMoeStack>,
    pub down_affine: Option<MlxAffineMoeStack>,
    /// ADR-029 iter-175 Step 1e — lazy-baked Q6_K_ID NR2 `m=1` decode
    /// dispatch record for the gate_up MoE call.  Populated on the
    /// first dispatch via `OnceLock::get_or_init` calling
    /// `mlx_native::ops::quantized_matmul_id_ggml::build_q6k_id_nr2_m1_record`.
    ///
    /// Three states encode the bake outcome:
    ///   - `OnceLock::new()` — not yet attempted.  Try to bake on first call.
    ///   - `Some(record)` — bake succeeded; fast-path eligible.
    ///   - `None` (inside the OnceLock) — bake skipped (`HF2Q_Q6K_ID_MV_NR2`
    ///     off, or non-Q6_K dtype).  Permanently fall through to unbaked.
    ///
    /// Per-layer slot: gemma4 APEX-Q5_K_M has 5 MoE layers × 1 dispatch =
    /// ~30 calls per decode-token through this path (n_tokens=1; top_k
    /// rows folded into `threadgroups.y`).
    pub decode_record_q6k_id_m1_gateup: std::sync::OnceLock<Option<mlx_native::DispatchRecord>>,
    /// ADR-029 iter-175 Step 1e2 — lazy-baked Q8_0_ID regular decode
    /// dispatch record for the MoE down call.  Populated on first
    /// dispatch via `OnceLock::get_or_init` calling
    /// `mlx_native::ops::quantized_matmul_id_ggml::build_q8_0_id_decode_record`.
    ///
    /// Distinct from `decode_record_q6k_id_m1_gateup`:
    ///   - Uses regular `kernel_mul_mv_id_q8_0_f32` (not NR2)
    ///   - Different geometry: threads=(8, 8, 1), align=8, no shmem
    ///   - Down call site passes `n_tokens=real_top_k, top_k=1` (vs gate_up
    ///     which uses `n_tokens=1, top_k=real_top_k`) — distinct params bake
    ///
    /// Per-layer slot: ~30 dispatches/decode-tok through the
    /// down path on gemma4 APEX-Q5_K_M (1 down × 30 layers).
    pub decode_record_q8_0_id_m1_down: std::sync::OnceLock<Option<mlx_native::DispatchRecord>>,
}

impl MlxMoeWeights {
    /// Affine expert overlays are one representation replacement, never two
    /// independently optional role overrides. Every execution path calls this
    /// before choosing a route so partial overlays fail closed consistently.
    pub(crate) fn affine_pair(&self) -> Result<Option<(&MlxAffineMoeStack, &MlxAffineMoeStack)>> {
        match (self.gate_up_affine.as_ref(), self.down_affine.as_ref()) {
            (None, None) => Ok(None),
            (Some(gate_up), Some(down)) => Ok(Some((gate_up, down))),
            _ => anyhow::bail!("Gemma MoE affine overlays must replace gate/up and down together"),
        }
    }
}

impl MlxMoeWeights {
    /// Construct a placeholder MoE bundle for **dense** layers.
    ///
    /// Dense GGUF layers carry zero MoE expert tensors. To keep the
    /// per-layer struct (`MlxDecoderLayerWeights`)
    /// uniform across dense + MoE layers without rippling
    /// `Vec<Option<MlxMoeWeights>>` through the forward path, we expose
    /// this constructor: it returns a bundle with `stacked_gate_up: None`
    /// and `stacked_down: None` plus 1-element placeholder buffers for
    /// every required field. The dense forward dispatch
    /// (`MlxModelWeights::forward_decode` / `forward_prefill`) consumes
    /// `MlxMlpWeights`, never the MoE bundle, so these placeholders are
    /// inert. The fused-id MoE dispatch already gates on
    /// `stacked_gate_up.is_some() && stacked_down.is_some()` (see
    /// `forward_decode` lines ~2863 / ~3922), so a misrouted MoE call
    /// against a dense-placeholder layer would falsify the `is_some()`
    /// gate at runtime rather than silently consuming garbage.
    ///
    /// Allocation cost is ~16 bytes per layer (vs. GBs of real expert
    /// tensors), and `top_k` / `moe_intermediate_size` are zeroed so
    /// any accidental read of them is also visibly wrong.
    pub fn dense_placeholder(mlx_device: &mlx_native::MlxDevice) -> Result<Self> {
        let router_proj_buf = alloc_one_f32_placeholder(mlx_device, "router_proj_buf")?;
        Ok(MlxMoeWeights {
            stacked_gate_up: None,
            stacked_down: None,
            gate_up_expert_stride: 0,
            down_expert_stride: 0,
            router_proj: MlxQWeight {
                buffer: router_proj_buf,
                info: QuantWeightInfo {
                    ggml_dtype: mlx_native::GgmlType::F32,
                    rows: 1,
                    cols: 1,
                },
                affine: None,
                decode_record_q6k_m1: std::sync::OnceLock::new(),
            },
            per_expert_scale: alloc_one_f32_placeholder(mlx_device, "per_expert_scale")?,
            gate_up_ggml_dtype: mlx_native::GgmlType::F32,
            down_ggml_dtype: mlx_native::GgmlType::F32,
            top_k: 0,
            moe_intermediate_size: 0,
            router_combined_weight: alloc_one_f32_placeholder(
                mlx_device,
                "router_combined_weight",
            )?,
            gate_up_affine: None,
            down_affine: None,
            decode_record_q6k_id_m1_gateup: std::sync::OnceLock::new(),
            decode_record_q8_0_id_m1_down: std::sync::OnceLock::new(),
        })
    }
}

/// Per-layer norm weights (7 RmsNorm per layer).
pub struct MlxLayerNorms {
    pub input_layernorm: MlxBuffer,
    pub post_attention_layernorm: MlxBuffer,
    pub pre_feedforward_layernorm: MlxBuffer,
    pub post_feedforward_layernorm: MlxBuffer,
    pub pre_feedforward_layernorm_2: MlxBuffer,
    pub post_feedforward_layernorm_1: MlxBuffer,
    pub post_feedforward_layernorm_2: MlxBuffer,
}

/// All mlx-native weights for one decoder layer, plus per-layer config
/// that used to live as parallel Vecs on `MlxModelWeights`.
pub struct MlxDecoderLayerWeights {
    pub attn: MlxAttentionWeights,
    pub mlp: MlxMlpWeights,
    pub moe: MlxMoeWeights,
    pub norms: MlxLayerNorms,
    pub layer_scalar: MlxBuffer,
    /// Head dim for this layer (Gemma-4: 256 for sliding, 512 for global).
    pub head_dim: usize,
    /// KV heads for this layer (Gemma-4: 8 for sliding, 2 for global).
    pub num_kv_heads: usize,
    /// Sliding vs Full attention — drives SDPA dispatch and KV cache layout.
    pub layer_type: LayerType,
}

/// One deliberately anonymous model-state allocation. Ordinary GGUF matrix
/// and state bytes are excluded from this ledger and must remain file-backed.
#[derive(Debug, Clone, PartialEq, Eq)]
struct NamedAnonymousModelBuffer {
    name: String,
    byte_len: u64,
}

/// Residency audit for ordinary GGUF model weights and state. The named list
/// contains only the router-derived value and documented one-F32 placeholders.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GemmaOrdinaryStorageSummary {
    ordinary: crate::serve::forward_mlx_shared::NativeMatrixStorageSummary,
    named_anonymous: Vec<NamedAnonymousModelBuffer>,
}

fn verify_ordinary_gguf_storage(storage: &GemmaOrdinaryStorageSummary) -> Result<u64> {
    anyhow::ensure!(
        storage.ordinary.unique_matrix_views > 0,
        "Gemma production load retained no ordinary GGUF matrix/state views"
    );
    anyhow::ensure!(
        storage.ordinary.anonymous_bytes == 0,
        "Gemma production load created {} anonymous ordinary matrix/state bytes",
        storage.ordinary.anonymous_bytes
    );
    storage
        .named_anonymous
        .iter()
        .try_fold(0_u64, |total, entry| {
            anyhow::ensure!(
                entry.byte_len > 0,
                "Gemma named anonymous model buffer '{}' is empty",
                entry.name
            );
            total
                .checked_add(entry.byte_len)
                .ok_or_else(|| anyhow::anyhow!("Gemma named anonymous storage byte count overflow"))
        })
}

fn push_qweight_storage<'a>(weight: &'a MlxQWeight, buffers: &mut Vec<&'a MlxBuffer>) {
    buffers.push(&weight.buffer);
    if let Some(affine) = weight.affine.as_ref() {
        buffers.extend([&affine.scales, &affine.biases]);
    }
}

fn is_documented_one_f32_placeholder(buffer: &MlxBuffer) -> bool {
    !buffer.is_file_backed()
        && buffer.dtype() == mlx_native::DType::F32
        && buffer.shape() == [1]
        && buffer.data_byte_len() == std::mem::size_of::<f32>()
}

// MlxKvCache, HbKvBuffers, DenseKvBuffers, HybridKvBuffers, alloc_hybrid_kv_for_layer,
// and DecodeRegime moved to crate::inference::models::gemma4::kv_cache
// (ADR-038 Step 2). Imported above via the use statement.

/// Reusable activation buffers for one forward pass.
pub struct MlxActivationBuffers {
    /// Reusable token-id input for one-row native embedding gather.
    pub embedding_token_id: MlxBuffer,
    /// Hidden state `[1, hidden_size]` F32.
    pub hidden: MlxBuffer,
    /// Scratch buffer for attention Q output `[1, num_heads * head_dim]` F32.
    pub attn_q: MlxBuffer,
    /// Scratch buffer for attention K output `[1, num_kv_heads * head_dim]` F32
    /// (sized for the largest layer — global with num_kv_heads=2, head_dim=512).
    pub attn_k: MlxBuffer,
    /// Scratch buffer for attention output after O projection `[1, hidden_size]` F32.
    pub attn_out: MlxBuffer,
    /// Scratch buffer for RMS norm output `[1, hidden_size]` F32.
    pub norm_out: MlxBuffer,
    /// Scratch buffer for residual `[1, hidden_size]` F32.
    pub residual: MlxBuffer,
    /// Scratch buffer for MLP gate output `[1, intermediate_size]` F32.
    pub mlp_gate: MlxBuffer,
    /// Scratch buffer for MLP up output `[1, intermediate_size]` F32.
    pub mlp_up: MlxBuffer,
    /// Scratch buffer for MLP fused output `[1, intermediate_size]` F32.
    pub mlp_fused: MlxBuffer,
    /// Scratch buffer for MLP down output `[1, hidden_size]` F32.
    pub mlp_down: MlxBuffer,
    /// Scratch buffer for SDPA output `[1, num_heads, 1, head_dim]` F32.
    /// Sized for largest head config (16 heads * 512 head_dim for global).
    pub sdpa_out: MlxBuffer,
    /// Temporary buffer for SDPA NWG>1 partial results (reduce kernel input).
    pub sdpa_tmp: MlxBuffer,
    /// RMS norm params buffer `[eps, dim]` as F32.
    pub norm_params: MlxBuffer,
    /// Position buffer `[pos]` as U32 — single element for decode.
    pub position: MlxBuffer,
    /// Softcap params buffer (used if softcapping is configured).
    pub softcap_params: MlxBuffer,
    /// Argmax output index buffer `[1]` U32.
    pub argmax_index: MlxBuffer,
    /// Argmax output value buffer `[1]` F32.
    pub argmax_value: MlxBuffer,
    /// Argmax params buffer `[vocab_size]` as U32.
    pub argmax_params: MlxBuffer,
    /// Logits output buffer `[1, vocab_size]` F32.
    pub logits: MlxBuffer,
    /// MoE scratch: router logits `[1, num_experts]` F32.
    pub moe_router_logits: MlxBuffer,
    /// MoE scratch: expert down output `[1, hidden_size]` F32.
    pub moe_expert_out: MlxBuffer,
    /// MoE scratch: accumulated output `[1, hidden_size]` F32.
    pub moe_accum: MlxBuffer,
    /// MoE scratch: norm output for router `[1, hidden_size]` F32.
    pub moe_norm_out: MlxBuffer,
    /// Router norm output `[1, hidden_size]` F32 — separate from `norm_out` to
    /// allow router norm to run concurrent with pre-FF norm 1 (which writes norm_out).
    pub router_norm_out: MlxBuffer,
    /// MoE scratch: expert ids buffer for _id kernel `[top_k]` U32.
    pub moe_expert_ids: MlxBuffer,
    /// MoE scratch: gate_up _id output `[top_k, 2*moe_intermediate]` F32.
    pub moe_gate_up_id_out: MlxBuffer,
    /// MoE scratch: down _id output `[top_k, hidden_size]` F32.
    pub moe_down_id_out: MlxBuffer,
    /// MoE scratch: swiglu output for _id path `[top_k, moe_intermediate]` F32.
    pub moe_swiglu_id_out: MlxBuffer,
    // --- Session merge buffers (S1+S2 collapse) ---
    /// Per-head norm params for sliding layers: `[eps, sliding_head_dim]` F32.
    pub norm_params_sliding_hd: MlxBuffer,
    /// Per-head norm params for global layers: `[eps, global_head_dim]` F32.
    pub norm_params_global_hd: MlxBuffer,
    /// GPU buffer holding global-layer freq_factors `[global_head_dim/2]` F32.
    pub rope_freq_factors_gpu: MlxBuffer,
    /// Dedicated V projection output buffer `[max_kv_heads * max_hd]` F32.
    /// Separates V from moe_expert_out to avoid aliasing in merged session.
    pub attn_v: MlxBuffer,
    /// Scratch buffer for Q after per-head norm `[num_heads * max_hd]` F32.
    pub attn_q_normed: MlxBuffer,
    /// Scratch buffer for K after per-head norm `[max_kv_heads * max_hd]` F32.
    pub attn_k_normed: MlxBuffer,
    /// MoE scratch: pre-scaled routing weights for weighted_sum kernel `[top_k]` F32.
    pub moe_routing_weights_gpu: MlxBuffer,
}

/// All mlx-native weights for the full Gemma 4 model.
pub struct MlxModelWeights {
    /// Model-lifetime authority for artifact-native route plans. Zero means
    /// the loaded weights have not completed activation against a GPU context.
    pub(super) native_activation_epoch: std::sync::atomic::AtomicU64,
    /// Artifact-native token table.  When `lm_head` is `None`, this exact
    /// buffer is also the tied output head.
    pub embed_weight: MlxQWeight,
    pub layers: Vec<MlxDecoderLayerWeights>,
    pub final_norm: MlxBuffer,
    /// Explicit untied `output.weight`, if the artifact declares one.
    /// Absence means tied storage; no duplicate buffer is synthesized.
    pub lm_head: Option<MlxQWeight>,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub rms_norm_eps: f32,
    pub final_logit_softcapping: Option<f32>,
    /// Per-layer KV caches.
    pub kv_caches: Vec<MlxKvCache>,
    /// Reusable activation buffers.
    pub activations: MlxActivationBuffers,
    /// Sliding window size.
    pub sliding_window: usize,
    /// RoPE theta for sliding layers.
    pub rope_theta_sliding: f32,
    /// RoPE theta for global layers.
    pub rope_theta_global: f32,
    /// Number of MoE experts.
    pub num_experts: usize,
    /// Intermediate size for dense MLP.
    pub intermediate_size: usize,
    /// Dense F32 KV buffers per layer for decode (ADR-009 Track 3).
    ///
    /// When set (by `forward_prefill`), `forward_decode` uses dense SDPA
    /// instead of TQ-packed SDPA. Each layer has K and V in head-major
    /// layout `[nkv_heads, capacity, head_dim]`.
    ///
    /// Per-layer capacity: sliding layers use ring-buffer mode sized to
    /// `sliding_window` (writes wrap at `seq_pos % sliding_window`);
    /// global layers use a linear buffer sized to `seq_len + max_tokens`.
    /// Attention is permutation-invariant over cached K,V (RoPE is baked
    /// in before caching), so the ring's slot order doesn't matter for
    /// correctness — the kernel just attends to all populated slots.
    /// ADR-017 Phase E.a iter-2.5 (Strategy A): per-layer Arc-wrapped
    /// owned KV buffers. The Arc tier is structural — at iter-2.5 the
    /// worker thread is still the sole holder of every Arc (strong
    /// count == 1 for every entry), so `Arc::get_mut` always succeeds
    /// at the kv-restore mutation site (engine.rs ~3479). Iter-3 will
    /// hand out Arc-clones to the LcpRegistry; at that point the
    /// mutation discipline tightens (registry-cloned Arcs become
    /// read-only via Arc::deref auto-coercion, and any in-place rewrite
    /// must consume + re-store the Arc to bring strong_count back to 1).
    ///
    /// **Read-path consumers UNCHANGED.** `dense_kvs[i].k`,
    /// `dense_kvs[i].v`, `dense_kvs[i].capacity`, `dense_kvs[i].is_sliding`
    /// all auto-deref through `Arc::deref` so existing forward_mlx.rs
    /// reader sites at lines 2432-2588 + 2794 compile without per-site
    /// edits. Field-access syntax `(&Arc<T>).k` resolves through the
    /// auto-deref chain `&Arc<T>` → `&T` → `&T.k` at zero cost.
    ///
    /// See dossier `docs/research/adr017-phase-e-option-a-2026-05-05.md`
    /// §10.3 Strategy A for the full rationale (~25 LOC additive in
    /// this file vs Strategy B's outer-Arc shape, which conflates
    /// per-layer eviction with whole-Vec rebuilds).
    pub dense_kvs: Option<Vec<std::sync::Arc<DenseKvBuffers>>>,
    /// ADR-017 Phase E.a iter-3.5b — end-of-prefill snapshot for the
    /// LcpRegistry. Populated by `forward_prefill_with_soft_tokens_resume`
    /// AT THE END of the per-token prefill loop (after all prompt
    /// positions written, BEFORE the function returns). Consumed by
    /// `engine.rs::generate_*` at the post-decode LCP store site, then
    /// cleared. `None` when the iter-3 env-gates (`HF2Q_KV_LCP_RESUME=1`
    /// + `HF2Q_USE_DENSE=1`) are off (no snapshot needed; LCP path
    /// inactive).
    ///
    /// **Why a separate field, not a return value:** changing
    /// `forward_prefill_with_soft_tokens_resume`'s return type from
    /// `Result<u32>` to `Result<(u32, Option<Vec<Arc<DenseKvBuffers>>>)>`
    /// would touch every call site (warmup, generate, embed_last,
    /// generate_stream_once, ...). A side-channel field on
    /// `MlxModelWeights` minimizes the surface — only the post-decode
    /// LCP store site reads it.
    ///
    /// **Why a snapshot, not a live-Arc clone:** decode mutates
    /// `dense_kvs[*][slot=p%capacity]` for sliding layers. The
    /// LcpRegistry must hold a SNAPSHOT taken at end-of-prefill
    /// (decode hasn't run yet) so future LCP hits read pure
    /// prompt-prefix state, not decode-corrupted ring slots. Lifts
    /// the iter-3 v1 wrap-guard restriction (which previously
    /// skipped store when `prompt_len + decode_writes > sliding_window`)
    /// at the cost of one extra per-layer KV allocation + memcpy per
    /// resume-eligible request (~50ms on Gemma 4 26B).
    pub dense_kvs_snapshot_for_lcp: Option<Vec<std::sync::Arc<DenseKvBuffers>>>,
    /// ADR-017 Phase E.a "gemma-hybrid-lcp" (2026-08-03) — end-of-prefill
    /// snapshot of the hybrid leg (F16 K + TQ-HB V) for the LCP registry,
    /// mirroring `dense_kvs_snapshot_for_lcp`. Populated only when
    /// `kv_lcp_resume` is on AND the hybrid regime allocated
    /// `self.hybrid_kv` this prefill (production default). Decode under
    /// the hybrid regime reads `hybrid_kv`, so an LCP resume must restore
    /// this leg alongside the dense one — see `GemmaLcpLayerKv`.
    pub hybrid_kv_snapshot_for_lcp:
        Option<Vec<std::sync::Arc<crate::inference::models::gemma4::kv_cache::HybridKvBuffers>>>,
    /// Tmp buffer for flash_attn_vec when using dense decode.
    pub dense_sdpa_tmp: Option<MlxBuffer>,
    // iter-20 Leg F `leg_f_kvs` + `leg_f_sdpa_tmp` shadow-cache fields deleted
    // iter-222 (2026-05-01) along with the iter-34 dense-on-shadow Leg F decode
    // branch and `dense_sdpa_on_tq_kv_enabled()` helper. See the file-level
    // iter-222 closure note above the deleted helper site for the rationale
    // (Gate H regression plus the "no fallback" contract). The
    // inline-fused TQ-native kernels (`flash_attn_vec_tq` / `flash_attn_vec_tq_hb`)
    // read directly from the TQ-packed `kv_caches[layer].{k,v}_packed` and
    // `leg_hb_encoded` buffers respectively — no F32 shadow cache required.
    /// iter-21 Track B: byte-packed higher-bit (5/6/8-bit) KV encoded cache.
    ///
    /// When `HF2Q_TQ_CODEBOOK_BITS=5|6|8` (default 8), K/V are encoded to
    /// byte-packed 5/6/8-bit Lloyd-Max indices via `hadamard_quantize_kv_hb`,
    /// stored here, and consumed inline by the `flash_attn_vec_tq_hb` kernel
    /// (no shadow-cache dequant round-trip).
    ///
    /// Layout: `[nkv_heads, capacity, head_dim]` U8 (1 byte per element).
    /// Norms: same layout as 4-bit caches (D=256: 1 norm/pos, D=512: 2/pos).
    pub leg_hb_encoded: Option<Vec<HbKvBuffers>>,
    /// ADR-028 Phase 10 (iter-347): hybrid K storage, F16 K + TQ-HB-packed V.
    ///
    /// Mutually exclusive with `leg_hb_encoded` at allocation time — exactly
    /// one of the two is `Some(…)` for any given model instance, governed by
    /// the `HF2Q_HYBRID_KV` env-gate (parsed in `investigation_env.rs`,
    /// default OFF until Phase 10f parity + 10g coherence gates pass).
    ///
    /// Why an `Option` field rather than a wrapping enum: the existing
    /// SDPA-dispatch site (`forward_decode`, ~line 3567) keys on the variant
    /// of `leg_hb_encoded` today; making the hybrid path additive (a sibling
    /// `Option` checked first) keeps the legacy TQ-HB path bit-identical when
    /// the gate is OFF (regression-safety mantra).
    #[allow(dead_code)] // Read in Phase 10c K-encode skip + 10e SDPA dispatcher (next iters).
    pub hybrid_kv: Option<Vec<HybridKvBuffers>>,
    /// Per-instance decode-step counter for the Gate H stderr emit lines.
    ///
    /// Increments on every successful `forward_decode`.  The audit-binary
    /// contract (`iter25_audit.rs::parse_nll_values`) sorts by `step=`,
    /// so a monotonic 0-based counter is the right shape.  Reset to 0
    /// at construction; [`MlxModelWeights::set_decode_regime`] also resets
    /// it between regimes for Gate H two-regime-one-process runs.
    pub decode_step: u64,
    /// ADR-007 Gate H per-call regime override (W12 iter-108a blocker #3).
    ///
    /// Default value [`DecodeRegime::Default`] preserves today's env-var-only
    /// path bit-exactly — the SDPA-mode gate reads `HF2Q_USE_DENSE` and
    /// `HF2Q_LAYER_POLICY` exactly as it does on the iter-108a base
    /// commit.  Set via [`MlxModelWeights::set_decode_regime`] to flip
    /// between TQ-active and dense-active SDPA within a single process
    /// (Gate H two-regime run); the setter also resets [`Self::decode_step`]
    /// so each regime's stderr `[HF2Q_NLL]` / `[HF2Q_DECODE_EMIT]` lines
    /// start at `step=0`.
    pub decode_regime: DecodeRegime,
    /// Cached startup-time flag: true iff none of the iter-108a Gate H
    /// runtime hooks are active. When true, the decode hot path skips
    /// per-token NLL emit, decode-emit, decode-replay, the `decode_step`
    /// counter mutation, AND the per-layer `decode_regime` enum match —
    /// keeping pre-iter-108a per-token cost bit-for-bit (W14b 5.6%
    /// regression: 95.0 → 100.6 tok/s baseline, 2026-04-25).
    ///
    /// Computed at construction from `INVESTIGATION_ENV` (a `LazyLock`
    /// that is populated exactly once per process via `from_env`) and
    /// the `decode_regime` field.  Re-evaluated only inside
    /// [`MlxModelWeights::set_decode_regime`] (since a non-Default
    /// regime requires the per-layer SDPA-gate match path to run).
    /// Never read or written on the per-token hot path beyond the
    /// initial single load — LLVM hoists the bool check above the
    /// per-layer loop and the per-token tail, so the entire
    /// instrumentation block becomes dead code under `if !gate_h_inactive`.
    pub gate_h_inactive: bool,
    /// ADR-007 Gate H per-instance replay-token override (W21 iter-108b).
    ///
    /// When non-empty, takes precedence over
    /// [`InvestigationEnv::decode_input_tokens`] in the post-argmax tail
    /// of [`Self::forward_decode`].  This is the in-process replay surface
    /// used by `cmd_parity_check_tq_quality` / `cmd_parity_capture_tq_quality`:
    /// pass 1 (dense) records the picked tokens directly via
    /// `forward_decode`'s return value, then pass 2 (TQ) sets this field
    /// before the decode loop so each TQ-step's logits are scored against
    /// the same token sequence dense produced (the ADR-007 §853-866 PPL
    /// input shape).  See [`Self::set_replay_tokens`].
    ///
    /// `LazyLock` makes [`InvestigationEnv::decode_input_tokens`] frozen
    /// at first access, so the env-var path can't switch mid-process —
    /// hence the per-instance override.  Empty by default; emit/NLL/decode-
    /// step bookkeeping in `forward_decode` continues to gate on
    /// [`Self::gate_h_inactive`], which is set to `false` by
    /// [`Self::set_replay_tokens`] whenever the replay vector is non-empty.
    ///
    /// (Wired by iter-108b's `parity_quality::run_two_regime_decode`;
    /// no in-tree caller as of iter-108a.)
    pub replay_tokens: Vec<u32>,
    /// ADR-007 Gate H per-instance dump-config override (W39 iter-112b).
    ///
    /// `INVESTIGATION_ENV.dump_dir` and `.dump_all_cache` are populated from
    /// process-start env via a `LazyLock` triggered in `main.rs::main` *before*
    /// `Cli::parse`, so the env vars W21's `parity_quality::run_two_regime_decode`
    /// sets at run time (after the LazyLock has frozen) never reach the SDPA
    /// dump gate at `forward_decode` lines 1268-1271 nor the dump-path
    /// formatter inside `dumps::dump_f32`.  W39 splits the two readers so the
    /// in-process Gate H harness can supply per-instance values instead:
    ///   - `Some(dir)` overrides `INVESTIGATION_ENV.dump_dir` for SDPA-out
    ///     dump file paths (consulted by the call site in `forward_decode`,
    ///     which routes through a path that picks up the override).
    ///   - `Some(true)` forces `dump_all_cache=true` for SDPA-out gating.
    ///   - `None` falls back to `INVESTIGATION_ENV` (i.e. the default
    ///     env-var-only path is byte-identical to pre-iter-112b).
    ///
    /// `set_dump_overrides` exposes the setter and is the only mutator.
    /// Like `set_replay_tokens`, the gate-H instrumentation flag isn't
    /// affected — these knobs are purely diagnostic plumbing for the
    /// SDPA-out file gate, not for the per-token NLL/replay block.
    ///
    /// (Wired by iter-112b's `parity_quality::run_two_regime_decode`; no
    /// other in-tree caller.)
    pub dump_dir_override: Option<std::path::PathBuf>,
    pub dump_all_cache_override: Option<bool>,
    /// ADR-007 Gate H per-instance decode-step dump counter (W39 iter-112b).
    ///
    /// Replaces the process-static `AtomicUsize` previously declared inside
    /// `forward_decode` at lines 1262-1267.  That static accumulated across
    /// the dense and TQ passes of the same Gate H run — pass 1 left it at
    /// `tokens`, so pass 2's `decode_step_for_dump < max_pos` was false at
    /// every step.  Per-instance + reset-between-passes restores the
    /// per-pass `[0, max_pos)` window.
    ///
    /// Reset to 0 by `set_decode_regime` and `set_replay_tokens` (matching
    /// `decode_step` semantics) and by the explicit
    /// `reset_decode_step_dump_counter` for the rare caller that wants to
    /// reset without touching regime / replay state.
    pub decode_step_dump_counter: usize,

    /// ADR-030 Phase 4 — optional DFlash spec-decode hidden-state
    /// capture session. When `Some`, `forward_prefill_batched`
    /// populates `dflash_capture.hidden_output` at indices matching
    /// `dflash_capture.target_layer_ids` during the layer loop.
    /// Default `None` preserves byte-identical legacy behavior — no
    /// production-path caller installs this; only the spec-decode
    /// orchestrator's `install_dflash_capture`/`take_dflash_capture`
    /// pair touches it.
    pub dflash_capture:
        Option<crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession>,
    /// ADR-029 iter-175 Step 1f — model-wide pre-baked `DispatchRecord`
    /// for hidden-size F32 `rms_norm` dispatches.  Populated on first call
    /// via `OnceLock::get_or_init` calling
    /// `mlx_native::ops::rms_norm::build_rms_norm_decode_record(F32, 1, hs)`.
    ///
    /// Hot-path coverage on gemma4 APEX-Q5_K_M decode: ~120 dispatches/tok
    /// (pre-FF norm + pre-FF norm 2 + router norm + post-FF norm 1) × 30
    /// layers.  All call sites share the same `(F32, rows=1, dim=hs)` bake.
    ///
    /// Three states encode the bake outcome:
    ///   - `OnceLock::new()` — not yet attempted; try to bake on first call.
    ///   - `Some(record)` — bake succeeded; fast-path eligible.
    ///   - `None` (inside the OnceLock) — bake skipped (unsupported dtype,
    ///     `HF2Q_RMS_NORM_V2=off` with mismatched bake, etc.).  Permanently
    ///     fall through to unbaked `dispatch_rms_norm`.
    ///
    /// Per-MODEL slot (not per-layer): all 30 layers' hs-norms share the
    /// same record since the bake key is `(dtype, rows, dim)` which is
    /// identical across them.  Total memory: 1 OnceLock × ~150 B.
    pub decode_record_rms_norm_f32_hs: std::sync::OnceLock<Option<mlx_native::DispatchRecord>>,

    /// ADR-040 iter-G(a) — cross-slot batched prefill descriptor. When
    /// `Some`, `forward_prefill_batched` runs in MULTI-SEQ mode: N prompts are
    /// concatenated into one T-token stream (`T = Σ seq_lens`, passed as
    /// `prompt_tokens`/`seq_len`) and processed in ONE forward pass with
    /// per-seq isolation. Four deltas gate on `.is_some()`:
    ///   1. positions — per-seq RoPE reset `[0..L0, 0..L1, …]` (not `0..T`),
    ///   2. mask — host-built block-diagonal causal mask
    ///      ([`super::super::super::serve::forward_prefill_batched::build_block_diagonal_mask_bf16`])
    ///      replaces the single-seq GPU `build_sdpa_mask_bf16`,
    ///   3. KV write — each seq's `[O_i, L_i)` slice of the T-stream K/V is
    ///      scattered into its own slot via `slot_views_hybrid[i]`,
    ///   4. head — gather each seq's last row → N first tokens → `out_first_tokens`.
    /// `None` (default) preserves BYTE-IDENTICAL single-seq behavior. Set and
    /// consumed by the `forward_prefill_batched_multi_seq` wrapper; never
    /// persists across calls. Hybrid-KV regime only (the production default);
    /// other regimes admit serially. Gated behind `HF2Q_PREFILL_CROSS_SLOT=1`.
    pub multi_seq_prefill: Option<MultiSeqPrefillState>,
}

/// ADR-040 iter-G(a) — per-call descriptor for cross-slot batched prefill.
/// Built and installed on `MlxModelWeights::multi_seq_prefill` by the
/// `forward_prefill_batched_multi_seq` wrapper, read by the four gated deltas
/// inside `forward_prefill_batched`, then taken back out for `out_first_tokens`.
pub struct MultiSeqPrefillState {
    /// Per-seq prompt lengths `L_i`. `Σ seq_lens == seq_len` (the T-token
    /// stream length passed to `forward_prefill_batched`).
    pub seq_lens: Vec<usize>,
    /// Per-seq start offset `O_i` into the concatenated T-token stream
    /// (exclusive prefix-sum of `seq_lens`; `O_0 == 0`).
    pub seq_offsets: Vec<usize>,
    /// Absolute logical position of the first token in each sequence's
    /// contribution. Cold multi-sequence prefill uses zero for every entry;
    /// resumed prefill uses the exact cached-prefix length for that slot.
    pub start_positions: Vec<usize>,
    /// Physical KV slot selected by each sequence. This is redundant with
    /// `slot_views_hybrid` for writes, but is required by mlx-native's
    /// batched hybrid-attention kernel when it reads the full shared slab.
    pub slot_ids: Vec<crate::serve::multi_seq_kv::SlotId>,
    /// Per-seq × per-layer hybrid-KV slot-views. Outer index = sequence,
    /// inner index = layer. Each `HybridKvBuffers` is a `slice_view` bundle
    /// sharing the `multi_seq_kv_hybrid` scaffold's Metal buffers at that
    /// seq's `slot_id` byte offset — so a kernel write through the view lands
    /// in the slot's scaffold region. Built by the wrapper via
    /// `build_slot_view_hybrid` (the iter-G(b) slot-view primitive, per-seq).
    pub slot_views_hybrid: Vec<Vec<HybridKvBuffers>>,
    /// Per-layer views over the complete multi-slot hybrid scaffold. Used
    /// only by resumed multi-sequence attention; per-query `slot_ids` select
    /// an isolated slot region inside these shared buffers.
    pub full_views_hybrid: Vec<HybridKvBuffers>,
    /// Filled by the head delta: each seq's first decoded (greedy argmax)
    /// token. Length `== seq_lens.len()` on return. Empty until the head runs.
    pub out_first_tokens: Vec<u32>,
    /// Final post-softcap logits for every sequence head row. These are kept
    /// separate because tool grammars, logit bias, logprobs, and probabilistic
    /// sampling must be applied independently per request after the shared
    /// transformer-body pass. Returning only argmax tokens would silently
    /// weaken native tool semantics for batched agent requests.
    pub out_logits: Vec<Vec<f32>>,
}

/// Per-request head output from one shared multi-sequence Gemma prefill.
/// `first_tokens[i]` is the exact GPU argmax of `logits[i]`; callers may use
/// the logits instead when request-local sampling or grammar constraints are
/// active.
pub struct MultiSeqPrefillOutput {
    pub first_tokens: Vec<u32>,
    pub logits: Vec<Vec<f32>>,
}
// ADR-031 Phase B foundation — compile-time Send+Sync assertion.
//
// Phase B needs to share `&MlxModelWeights` across a main thread and a
// worker thread during parallel-encode (HF2Q_PARALLEL_ENCODE=1).  That
// requires Self: Sync.  This assertion fails the build at this site if a
// future field violates the contract, surfacing the regression long
// before runtime.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MlxModelWeights>();
};

// HybridKvBuffers, alloc_hybrid_kv_for_layer, DecodeRegime moved to
// crate::inference::models::gemma4::kv_cache (ADR-038 Step 2).

// f32_slice_to_le_bytes moved to kv_persist.rs (only used by tq_v2_snapshot_block).

// DwqOverlayRole, parse_dwq_overlay_metadata, parse_dwq_overlay_role,
// MoeBaseRole, parse_dwq_moe_expert_role moved to forward_mlx_shared.rs
// (ADR-038 Step 1). Re-exported via the pub use shim above.

fn gemma_native_bf16_matrix<'a>(
    label: &'a str,
    weight: &'a MlxQWeight,
) -> Result<Option<NativeBf16Matrix<'a>>> {
    if weight.affine.is_some() {
        return Ok(None);
    }
    anyhow::ensure!(
        (weight.buffer.dtype() == DType::BF16) == (weight.info.ggml_dtype == GgmlType::BF16),
        "{label}: Gemma buffer storage {} disagrees with declared type {:?}",
        weight.buffer.dtype(),
        weight.info.ggml_dtype
    );
    if weight.info.ggml_dtype != GgmlType::BF16 {
        return Ok(None);
    }
    Ok(Some(NativeBf16Matrix::unbatched(
        label,
        &weight.buffer,
        u32::try_from(weight.info.rows)?,
        u32::try_from(weight.info.cols)?,
    )))
}

impl MlxModelWeights {
    pub(crate) fn native_bf16_matrices(&self) -> Result<Vec<NativeBf16Matrix<'_>>> {
        fn push<'a>(
            out: &mut Vec<NativeBf16Matrix<'a>>,
            label: &'static str,
            weight: &'a MlxQWeight,
        ) -> Result<()> {
            if let Some(matrix) = gemma_native_bf16_matrix(label, weight)? {
                out.push(matrix);
            }
            Ok(())
        }

        let mut out = Vec::new();
        for layer in &self.layers {
            push(&mut out, "Gemma Q projection", &layer.attn.q_proj)?;
            push(&mut out, "Gemma K projection", &layer.attn.k_proj)?;
            if let Some(v_proj) = layer.attn.v_proj.as_ref() {
                push(&mut out, "Gemma V projection", v_proj)?;
            }
            push(&mut out, "Gemma attention output", &layer.attn.o_proj)?;
            push(&mut out, "Gemma FFN gate", &layer.mlp.gate_proj)?;
            push(&mut out, "Gemma FFN up", &layer.mlp.up_proj)?;
            push(&mut out, "Gemma FFN down", &layer.mlp.down_proj)?;
            push(&mut out, "Gemma MoE router", &layer.moe.router_proj)?;
        }
        push(
            &mut out,
            "Gemma output head",
            self.lm_head.as_ref().unwrap_or(&self.embed_weight),
        )?;
        Ok(out)
    }

    pub(crate) fn native_scalar_expert_matrices(
        &self,
    ) -> Result<Vec<NativeScalarExpertMatrix<'_>>> {
        use mlx_native::{DenseMatmulIdInputLayout, DenseMatmulIdMultiplicity};

        let calibrated_m = super::native_expert_activation_widths();
        let mut out = Vec::new();
        for layer in &self.layers {
            let moe = &layer.moe;
            if let Some((gate_up, down)) = moe.affine_pair()? {
                gate_up.validate_geometry(
                    "Gemma expert gate/up",
                    moe.moe_intermediate_size
                        .checked_mul(2)
                        .context("Gemma affine gate/up width overflow")?,
                    self.hidden_size,
                    self.num_experts,
                )?;
                down.validate_geometry(
                    "Gemma expert down",
                    self.hidden_size,
                    moe.moe_intermediate_size,
                    self.num_experts,
                )?;
                continue;
            }
            let Some(gate_up) = moe.stacked_gate_up.as_ref() else {
                continue;
            };
            let down = moe
                .stacked_down
                .as_ref()
                .context("Gemma MoE layer has gate/up experts but no down experts")?;
            for (label, weight, ggml_type, n, k, stride, input_layout) in [
                (
                    "Gemma expert gate/up",
                    gate_up,
                    moe.gate_up_ggml_dtype,
                    moe.moe_intermediate_size
                        .checked_mul(2)
                        .context("Gemma gate/up expert width overflow")?,
                    self.hidden_size,
                    moe.gate_up_expert_stride,
                    DenseMatmulIdInputLayout::SharedPerToken,
                ),
                (
                    "Gemma expert down",
                    down,
                    moe.down_ggml_dtype,
                    self.hidden_size,
                    moe.moe_intermediate_size,
                    moe.down_expert_stride,
                    DenseMatmulIdInputLayout::Slotted,
                ),
            ] {
                if !matches!(ggml_type, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
                    continue;
                }
                let expected_dtype = match ggml_type {
                    GgmlType::F32 => DType::F32,
                    GgmlType::F16 => DType::F16,
                    GgmlType::BF16 => DType::BF16,
                    _ => unreachable!(),
                };
                anyhow::ensure!(
                    weight.dtype() == expected_dtype,
                    "{label}: declared {ggml_type:?} but maps as {}",
                    weight.dtype()
                );
                out.push(NativeScalarExpertMatrix {
                    label,
                    weight,
                    n: u32::try_from(n)?,
                    k: u32::try_from(k)?,
                    top_k: u32::try_from(moe.top_k)?,
                    n_experts: u32::try_from(self.num_experts)?,
                    expert_stride_bytes: stride,
                    input_layout,
                    id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
                    calibrated_m: calibrated_m.clone(),
                });
            }
        }
        Ok(out)
    }

    /// Freeze the complete target representation union before any Gemma
    /// projection can execute. Drafter matrices extend only the dense BF16
    /// portion; target scalar expert identities are always included.
    pub(crate) fn activate_native_routes(
        &self,
        ctx: &mut GpuContext,
        extra_bf16: &[NativeBf16Matrix<'_>],
    ) -> Result<()> {
        let mut dense = self.native_bf16_matrices()?;
        dense.extend_from_slice(extra_bf16);
        // Inventory and validate both representation families before either
        // helper can freeze immutable registry state.
        let experts = self.native_scalar_expert_matrices()?;
        ctx.activate_native_bf16_dense(&dense)?;
        ctx.activate_native_scalar_experts(&experts)?;
        let activation_epoch = ctx.activation_epoch();
        self.native_activation_epoch
            .compare_exchange(
                0,
                activation_epoch,
                std::sync::atomic::Ordering::Release,
                std::sync::atomic::Ordering::Acquire,
            )
            .map_err(|bound| {
                anyhow::anyhow!(
                    "Gemma weights are already bound to activation epoch {bound}; cannot bind epoch {activation_epoch}"
                )
            })?;
        Ok(())
    }

    pub(crate) fn native_activation_epoch(&self) -> Result<u64> {
        let epoch = self
            .native_activation_epoch
            .load(std::sync::atomic::Ordering::Acquire);
        anyhow::ensure!(epoch != 0, "Gemma native routes were not activated");
        Ok(epoch)
    }

    /// Resolve the artifact-declared output head.  A missing `output.weight`
    /// is the tied case and intentionally returns the embedding allocation.
    #[inline]
    pub fn resolved_lm_head(&self) -> &MlxQWeight {
        resolve_tied_or_explicit(&self.embed_weight, self.lm_head.as_ref())
    }

    /// Account every ordinary GGUF matrix/state view retained by the model.
    /// Any anonymous byte in `ordinary` is an implicit transform and therefore
    /// a load-contract violation. The separate named ledger is limited to the
    /// precomputed router weight and exact one-F32 structural placeholders.
    fn ordinary_gguf_storage_summary(&self) -> Result<GemmaOrdinaryStorageSummary> {
        let mut ordinary = Vec::new();
        let mut named_anonymous = Vec::new();
        push_qweight_storage(&self.embed_weight, &mut ordinary);
        if let Some(head) = self.lm_head.as_ref() {
            push_qweight_storage(head, &mut ordinary);
        }
        ordinary.push(&self.final_norm);

        let mut record_named = |name: String, buffer: &MlxBuffer| -> Result<()> {
            if buffer.is_file_backed() {
                anyhow::bail!(
                    "Gemma named anonymous model buffer '{name}' unexpectedly has file backing"
                );
            }
            named_anonymous.push(NamedAnonymousModelBuffer {
                name,
                byte_len: u64::try_from(buffer.data_byte_len())?,
            });
            Ok(())
        };
        for (layer_index, layer) in self.layers.iter().enumerate() {
            for projection in [
                &layer.attn.q_proj,
                &layer.attn.k_proj,
                &layer.attn.o_proj,
                &layer.mlp.gate_proj,
                &layer.mlp.up_proj,
                &layer.mlp.down_proj,
            ] {
                push_qweight_storage(projection, &mut ordinary);
            }
            if let Some(projection) = layer.attn.v_proj.as_ref() {
                push_qweight_storage(projection, &mut ordinary);
            }
            ordinary.extend([
                &layer.attn.q_norm_weight,
                &layer.attn.k_norm_weight,
                &layer.norms.input_layernorm,
                &layer.norms.post_attention_layernorm,
                &layer.norms.pre_feedforward_layernorm,
                &layer.norms.post_feedforward_layernorm,
                &layer.layer_scalar,
            ]);

            for (suffix, buffer) in [
                (
                    "pre_ffw_norm_2.weight",
                    &layer.norms.pre_feedforward_layernorm_2,
                ),
                (
                    "post_ffw_norm_1.weight",
                    &layer.norms.post_feedforward_layernorm_1,
                ),
                (
                    "post_ffw_norm_2.weight",
                    &layer.norms.post_feedforward_layernorm_2,
                ),
            ] {
                if is_documented_one_f32_placeholder(buffer) {
                    record_named(format!("blk.{layer_index}.{suffix}:placeholder"), buffer)?;
                } else {
                    ordinary.push(buffer);
                }
            }

            match (&layer.moe.stacked_gate_up, &layer.moe.stacked_down) {
                (Some(gate_up), Some(down)) => {
                    ordinary.extend([gate_up, down, &layer.moe.per_expert_scale]);
                    push_qweight_storage(&layer.moe.router_proj, &mut ordinary);
                    record_named(
                        format!("blk.{layer_index}.router_combined_weight:derived"),
                        &layer.moe.router_combined_weight,
                    )?;
                }
                (None, None) => {
                    for (role, buffer) in [
                        ("router_proj", &layer.moe.router_proj.buffer),
                        ("per_expert_scale", &layer.moe.per_expert_scale),
                        ("router_combined_weight", &layer.moe.router_combined_weight),
                    ] {
                        if is_documented_one_f32_placeholder(buffer) {
                            record_named(
                                format!("blk.{layer_index}.{role}:dense-placeholder"),
                                buffer,
                            )?;
                        } else {
                            ordinary.push(buffer);
                        }
                    }
                }
                _ => anyhow::bail!(
                    "Gemma layer {layer_index} retained only one of its two expert stacks"
                ),
            }
        }
        if self.activations.rope_freq_factors_gpu.is_file_backed() {
            ordinary.push(&self.activations.rope_freq_factors_gpu);
        } else if is_documented_one_f32_placeholder(&self.activations.rope_freq_factors_gpu) {
            record_named(
                "rope_freqs.weight:placeholder".to_owned(),
                &self.activations.rope_freq_factors_gpu,
            )?;
        } else {
            ordinary.push(&self.activations.rope_freq_factors_gpu);
        }

        Ok(GemmaOrdinaryStorageSummary {
            ordinary: crate::serve::forward_mlx_shared::summarize_native_matrix_storage(ordinary)?,
            named_anonymous,
        })
    }

    /// ADR-030 Phase 4 — install a DFlash hidden-state capture session.
    ///
    /// While installed, `forward_prefill_batched` will populate the
    /// session's `hidden_output` buffer with `pf_hidden` contents at
    /// layer indices matching `session.target_layer_ids`. Reset the
    /// session via `take_dflash_capture` after the forward returns.
    ///
    /// Default state (no install): byte-identical to legacy behavior.
    pub fn install_dflash_capture(
        &mut self,
        session: crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession,
    ) {
        self.dflash_capture = Some(session);
    }

    /// Take back the installed DFlash capture session, returning its
    /// populated buffers. Returns `None` if no session was installed.
    /// After this call, subsequent `forward_prefill_batched` calls
    /// revert to legacy non-capturing behavior.
    pub fn take_dflash_capture(
        &mut self,
    ) -> Option<crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession> {
        self.dflash_capture.take()
    }

    /// True if a DFlash capture session is currently installed.
    pub fn has_dflash_capture(&self) -> bool {
        self.dflash_capture.is_some()
    }

    /// ADR-030 Phase 4 — public embed_tokens lookup.
    ///
    /// Mirrors the gather+scale embedding inside `forward_prefill_batched`
    /// (lines 618-641): for each token in `tokens`, copy
    /// `embed_weight[token_id * hidden_size..]` into the output buffer,
    /// then scale by `sqrt(hidden_size)` (gemma's embed_scale convention).
    ///
    /// Returns a fresh MlxBuffer of shape `[tokens.len(), hidden_size]`
    /// F32, ready to feed into `dispatch_dflash_model_forward` as `h`.
    ///
    /// Used by the DFlash spec-decode orchestrator to embed the
    /// "block" `[last_committed_token, mask, mask, ..., mask]` before
    /// the drafter forward.
    pub fn embed_tokens(
        &self,
        tokens: &[u32],
        gpu: &mut crate::serve::gpu::GpuContext,
    ) -> anyhow::Result<MlxBuffer> {
        let hs = self.hidden_size;
        let n_tokens = tokens.len();
        if n_tokens == 0 {
            anyhow::bail!("embed_tokens: empty tokens");
        }
        for &token in tokens {
            if token as usize >= self.embed_weight.info.rows {
                anyhow::bail!(
                    "embed_tokens: token id {} out of vocab range {}",
                    token,
                    self.embed_weight.info.rows
                );
            }
        }
        let (exec, reg) = gpu.split();
        let dev = exec.device();
        let out = dev
            .alloc_buffer(
                n_tokens * hs * 4,
                mlx_native::DType::F32,
                vec![n_tokens, hs],
            )
            .map_err(|e| anyhow::anyhow!("alloc embed output: {e}"))?;
        let mut token_ids = dev
            .alloc_buffer(
                n_tokens * std::mem::size_of::<u32>(),
                mlx_native::DType::U32,
                vec![n_tokens],
            )
            .map_err(|e| anyhow::anyhow!("alloc embed token ids: {e}"))?;
        token_ids
            .as_mut_slice::<u32>()
            .map_err(|e| anyhow::anyhow!("write embed token ids: {e}"))?
            .copy_from_slice(tokens);
        let mut session = exec
            .begin()
            .map_err(|e| anyhow::anyhow!("begin embed session: {e}"))?;
        crate::inference::models::gemma4::native_matrix::encode_embedding(
            &mut session,
            reg,
            dev,
            &self.embed_weight,
            &token_ids,
            &out,
            n_tokens,
        )?;
        session
            .finish()
            .map_err(|e| anyhow::anyhow!("finish embed session: {e}"))?;
        Ok(out)
    }

    /// Load all model weights directly from a GGUF file into mlx-native
    /// MlxBuffers.
    ///
    /// `progress` drives the default-mode in-place `\r`-overwrite progress
    /// line on stderr; it is a no-op when stderr isn't a TTY or verbosity > 0
    /// (tracing debug events then cover per-layer detail).
    ///
    /// ADR-008 Phase 2: replaces `load_from_candle()` — weights go
    /// GGUF → MlxBuffer with zero candle involvement.
    pub fn load_from_gguf(
        gguf: &mlx_native::gguf::GgufFile,
        cfg: &Gemma4Config,
        gpu: &mut GpuContext,
        progress: &mut crate::serve::header::LoadProgress,
    ) -> Result<Self> {
        let mlx_device = gpu.device();
        tracing::debug!("Loading mlx-native weights directly from GGUF...");

        // Preflight the tied/untied IO contract before any Metal allocation.
        // Malformed or unsupported storage therefore cannot leave a partially
        // loaded multi-gigabyte model behind.
        let load_timing = std::env::var("HF2Q_LOAD_TIMING").as_deref() == Ok("1");
        let t_pre = std::time::Instant::now();
        let io_plan = preflight_io(gguf, cfg.vocab_size, cfg.hidden_size)?;
        preflight_projections(gguf, cfg)?;
        let state_plan = preflight_f32_state(gguf, cfg)?;
        let mapped = map_admitted_tensor_data(gguf, mlx_device)?;
        let embed_weight = MlxQWeight::from_mapped_gguf_tensor(
            &mapped,
            gguf.tensor_info(&io_plan.embedding.name)
                .expect("embedding passed preflight"),
        )?;
        let lm_head = io_plan
            .output
            .as_ref()
            .map(|spec| {
                MlxQWeight::from_mapped_gguf_tensor(
                    &mapped,
                    gguf.tensor_info(&spec.name)
                        .expect("output head passed preflight"),
                )
            })
            .transpose()?;
        tracing::info!(
            "Gemma IO matrices retain artifact storage: embedding={:?}, head={}, tied={}, file_backed=true",
            embed_weight.info.ggml_dtype,
            lm_head
                .as_ref()
                .map(|weight| format!("{:?}", weight.info.ggml_dtype))
                .unwrap_or_else(|| "embedding".to_owned()),
            lm_head.is_none(),
        );
        if load_timing {
            tracing::info!(
                "[LOAD_TIMING] embed_weight_load={:.0}ms",
                t_pre.elapsed().as_secs_f64() * 1000.0
            );
        }
        let t_fn = std::time::Instant::now();

        // --- Final norm (F32) ---
        tracing::debug!("Loading final_norm");
        let final_norm = state_plan
            .load_mapped(&mapped, "output_norm.weight")
            .map_err(|e| anyhow::anyhow!("final_norm: {e}"))?;
        if load_timing {
            tracing::info!(
                "[LOAD_TIMING] final_norm_load={:.0}ms",
                t_fn.elapsed().as_secs_f64() * 1000.0
            );
        }

        // --- Per-layer weights ---
        let num_layers = cfg.num_hidden_layers;
        let mut layers = Vec::with_capacity(num_layers);
        let mut kv_caches = Vec::with_capacity(num_layers);

        // ADR-028 iter-462: bucket timing inside layer loop, opt-in via
        // HF2Q_LOAD_TIMING=1.  Bisects mlx_weights_load (88% of startup
        // per iter-461) into attn/mlp/moe/misc.
        let load_timing = std::env::var("HF2Q_LOAD_TIMING").as_deref() == Ok("1");
        let mut cum_attn_ns = 0u128;
        let mut cum_mlp_ns = 0u128;
        let mut cum_moe_ns = 0u128;
        let mut cum_misc_ns = 0u128;
        // ADR-028 iter-463: MoE sub-buckets.  `cum_moe_other_ns` is derived
        // from the siblings at end-of-loop inside the `if load_timing` block
        // below — it has no per-layer accumulation, so it lives there.
        let mut cum_moe_gate_up_ns = 0u128;
        let mut cum_moe_down_ns = 0u128;
        let mut cum_moe_router_cpu_ns = 0u128;

        for i in 0..num_layers {
            tracing::debug!("GGUF layer {}/{}: loading weights", i + 1, num_layers);

            // -- Attention quantized weights --
            let t_attn = std::time::Instant::now();
            let q_proj = load_mapped_projection(gguf, &mapped, &format!("blk.{i}.attn_q.weight"))?;
            let k_proj = load_mapped_projection(gguf, &mapped, &format!("blk.{i}.attn_k.weight"))?;
            let v_proj = if cfg.is_full_attention(i) && cfg.attention_k_eq_v {
                None
            } else {
                Some(load_mapped_projection(
                    gguf,
                    &mapped,
                    &format!("blk.{i}.attn_v.weight"),
                )?)
            };
            let o_proj =
                load_mapped_projection(gguf, &mapped, &format!("blk.{i}.attn_output.weight"))?;

            // -- Attention head norms (F32) --
            let q_norm_weight = state_plan
                .load_mapped(&mapped, &format!("blk.{i}.attn_q_norm.weight"))
                .map_err(|e| anyhow::anyhow!("layer {i} q_norm: {e}"))?;
            let k_norm_weight = state_plan
                .load_mapped(&mapped, &format!("blk.{i}.attn_k_norm.weight"))
                .map_err(|e| anyhow::anyhow!("layer {i} k_norm: {e}"))?;
            cum_attn_ns += t_attn.elapsed().as_nanos();

            let attn = MlxAttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm_weight,
                k_norm_weight,
            };

            // -- Dense MLP (quantized) --
            let t_mlp = std::time::Instant::now();
            let gate_proj =
                load_mapped_projection(gguf, &mapped, &format!("blk.{i}.ffn_gate.weight"))?;
            let up_proj = load_mapped_projection(gguf, &mapped, &format!("blk.{i}.ffn_up.weight"))?;
            let down_proj =
                load_mapped_projection(gguf, &mapped, &format!("blk.{i}.ffn_down.weight"))?;

            let mlp = MlxMlpWeights {
                gate_proj,
                up_proj,
                down_proj,
            };
            cum_mlp_ns += t_mlp.elapsed().as_nanos();

            // -- MoE expert weights (3D tensors, already stacked in GGUF) --
            //
            // Keep the MoE expert load conditional on tensor presence. The
            // earlier loader
            // unconditionally required `blk.{i}.ffn_gate_up_exps.weight`
            // and bailed with `missing blk.0.ffn_gate_up_exps.weight`
            // when handed a structurally dense GGUF with
            // `ffn_{gate,up,down}.weight` per layer and no expert tensors.
            //
            // Detection rule (deterministic, GGUF-metadata-only — never
            // filename-based per the iter-227 correctness pin): a layer
            // is MoE iff BOTH `ffn_gate_up_exps.weight` AND
            // `ffn_down_exps.weight` tensors are present in the GGUF.
            // When either is absent we treat the layer as dense and
            // populate `MlxMoeWeights` with `stacked_{gate_up,down}: None`
            // plus 1-element placeholder buffers for the `router_*` /
            // `per_expert_scale` / `router_combined_weight` fields. The
            // forward dispatch (`forward_decode` lines 2863 / 3922)
            // already gates fused-id MoE on `stacked_gate_up.is_some()
            // && stacked_down.is_some()`; the dense MLP path consumes
            // `MlxMlpWeights` (loaded above unconditionally at lines
            // 962-971) and never reads the placeholder MoE fields.
            // Layer mixing (some layers MoE, some dense) is supported
            // structurally: expert presence is a per-block decision.
            let gu_name = format!("blk.{i}.ffn_gate_up_exps.weight");
            let dn_name = format!("blk.{i}.ffn_down_exps.weight");
            let gu_info_opt = gguf.tensor_info(&gu_name);
            let dn_info_opt = gguf.tensor_info(&dn_name);
            let layer_has_moe_experts = gu_info_opt.is_some() && dn_info_opt.is_some();

            let t_moe = std::time::Instant::now();
            let moe = if layer_has_moe_experts {
                // MoE layer — preserve pre-iter-227 load behavior byte-
                // identically. The two-clone of `gguf.tensor_info` is
                // safe; we already established both are Some above.
                let t_gu = std::time::Instant::now();
                let gu_info = gu_info_opt.unwrap();
                let stacked_gate_up_buf = map_native_gguf_tensor_view(&mapped, gu_info)?;
                let gate_up_expert_stride = stacked_gate_up_buf.data_byte_len() / cfg.num_experts;
                let gate_up_ggml_dtype = gu_info.ggml_type;
                cum_moe_gate_up_ns += t_gu.elapsed().as_nanos();

                let t_dn = std::time::Instant::now();
                let dn_info = dn_info_opt.unwrap();
                let stacked_down_buf = map_native_gguf_tensor_view(&mapped, dn_info)?;
                let down_expert_stride = stacked_down_buf.data_byte_len() / cfg.num_experts;
                let down_ggml_dtype = dn_info.ggml_type;
                cum_moe_down_ns += t_dn.elapsed().as_nanos();

                if (i + 1) % 5 == 0 || i == 0 {
                    tracing::debug!(
                        "GGUF layer {}/{}: MoE experts loaded (stacked, {:.1} MB + {:.1} MB)",
                        i + 1,
                        num_layers,
                        stacked_gate_up_buf.data_byte_len() as f64 / 1e6,
                        stacked_down_buf.data_byte_len() as f64 / 1e6
                    );
                }

                // -- Router and scales (F32) --
                let router_proj =
                    load_mapped_projection(gguf, &mapped, &format!("blk.{i}.ffn_gate_inp.weight"))?;
                let router_scale = state_plan
                    .load_mapped(&mapped, &format!("blk.{i}.ffn_gate_inp.scale"))
                    .map_err(|e| anyhow::anyhow!("layer {i} router_scale: {e}"))?;
                let per_expert_scale = state_plan
                    .load_mapped(&mapped, &format!("blk.{i}.ffn_down_exps.scale"))
                    .map_err(|e| anyhow::anyhow!("layer {i} per_expert_scale: {e}"))?;

                // Pre-compute router combined weight:
                //   router_combined_weight[j] = router_scale[j] * (hidden_size ^ -0.5)
                let t_rcw = std::time::Instant::now();
                let router_combined_weight = {
                    let scale_factor = (cfg.hidden_size as f32).powf(-0.5);
                    let rs: &[f32] = router_scale.as_slice().map_err(|e| {
                        anyhow::anyhow!("router_scale read for combined weight: {e}")
                    })?;
                    let mut combined = mlx_device
                        .alloc_buffer(
                            cfg.hidden_size * std::mem::size_of::<f32>(),
                            mlx_native::DType::F32,
                            vec![cfg.hidden_size],
                        )
                        .map_err(|e| anyhow::anyhow!("router_combined_weight alloc: {e}"))?;
                    let dst: &mut [f32] = combined
                        .as_mut_slice()
                        .map_err(|e| anyhow::anyhow!("router_combined_weight write: {e}"))?;
                    for j in 0..cfg.hidden_size {
                        dst[j] = rs[j] * scale_factor;
                    }
                    combined
                };
                cum_moe_router_cpu_ns += t_rcw.elapsed().as_nanos();

                MlxMoeWeights {
                    stacked_gate_up: Some(stacked_gate_up_buf),
                    stacked_down: Some(stacked_down_buf),
                    gate_up_expert_stride: gate_up_expert_stride as u64,
                    down_expert_stride: down_expert_stride as u64,
                    router_proj,
                    per_expert_scale,
                    gate_up_ggml_dtype,
                    down_ggml_dtype,
                    top_k: cfg.top_k_experts,
                    moe_intermediate_size: cfg.moe_intermediate_size,
                    router_combined_weight,
                    gate_up_affine: None,
                    down_affine: None,
                    decode_record_q6k_id_m1_gateup: std::sync::OnceLock::new(),
                    decode_record_q8_0_id_m1_down: std::sync::OnceLock::new(),
                }
            } else {
                // Dense layer — produce a placeholder MoE bundle so the
                // existing per-layer struct (`MlxDecoderLayerWeights`)
                // stays uniform without a Vec<Option<MoeWeights>>
                // ripple change. The dense forward path uses
                // `MlxMlpWeights` (loaded at lines 962-971) and never
                // reads these placeholder buffers; the fused-id MoE
                // dispatch already gates on `stacked_gate_up.is_some()
                // && stacked_down.is_some()` (forward_decode lines
                // 2863 / 3922) so the placeholders are consulted only
                // by metadata fields like `top_k` (read but unused on
                // the dense path). Buffer sizes are 1 element to
                // minimize wasted allocation; on a 28-layer dense model
                // this adds 28 × ~16 bytes = ~448 bytes
                // overhead vs. the pre-iter-227 unconditional path
                // (which would have OOM-allocated GBs of expert
                // tensors that don't exist on disk).
                if i == 0 {
                    tracing::debug!(
                        "GGUF layer {}/{}: dense FFN detected (no {gu_name} / {dn_name}); \
                         skipping MoE expert load — using placeholder MoE bundle",
                        i + 1,
                        num_layers,
                    );
                }
                MlxMoeWeights::dense_placeholder(mlx_device)
                    .map_err(|e| anyhow::anyhow!("layer {i} MoE placeholder alloc: {e}"))?
            };

            // -- Norm weights (F32) --
            //
            // G4-CFA-5b (2026-05-23): the 3 MoE-only FFN norms
            // (`pre_ffw_norm_2`, `post_ffw_norm_1`, `post_ffw_norm_2`) are
            // present in 26B-A4B MoE GGUFs but ABSENT in dense Gemma 4 31B
            // GGUFs (bartowski / unsloth `google_gemma-4-31B-it-Q4_K_M.gguf`
            // carries only 4 FFN-related norms: `attn_norm`, `ffn_norm`,
            // `post_attention_norm`, `post_ffw_norm`, plus the always-present
            // `attn_q_norm` / `attn_k_norm`). The dense tree-verify path
            // (`gemma4_tree_verify_full_layer_q`, gpu_full_attn.rs:3099) reads
            // only `pre_feedforward_layernorm` (step B) and
            // `post_feedforward_layernorm` (step G) — never the `_1` / `_2`
            // MoE-only siblings. Use `load_optional_norm_or_placeholder` to
            // fall back to 1-element F32 placeholders when those tensors are
            // absent (same `iter-227` placeholder-bundle pattern as
            // `MlxMoeWeights::dense_placeholder`). The dense FFN path
            // (`MlxMlpWeights`, loaded above unconditionally) and the MoE
            // forward path (which reads the `_1` / `_2` norms) are mutually
            // exclusive at runtime — `forward_decode` gates MoE on
            // `stacked_*.is_some()` and `forward_tree_verify_gpu` is the
            // dense-only entry point. A misroute would falsify the
            // `stacked_*.is_some()` gate, not silently consume garbage.
            let norms = MlxLayerNorms {
                input_layernorm: state_plan
                    .load_mapped(&mapped, &format!("blk.{i}.attn_norm.weight"))
                    .map_err(|e| anyhow::anyhow!("layer {i} attn_norm: {e}"))?,
                post_attention_layernorm: state_plan
                    .load_mapped(&mapped, &format!("blk.{i}.post_attention_norm.weight"))
                    .map_err(|e| anyhow::anyhow!("layer {i} post_attn_norm: {e}"))?,
                pre_feedforward_layernorm: state_plan
                    .load_mapped(&mapped, &format!("blk.{i}.ffn_norm.weight"))
                    .map_err(|e| anyhow::anyhow!("layer {i} ffn_norm: {e}"))?,
                post_feedforward_layernorm: state_plan
                    .load_mapped(&mapped, &format!("blk.{i}.post_ffw_norm.weight"))
                    .map_err(|e| anyhow::anyhow!("layer {i} post_ffw_norm: {e}"))?,
                pre_feedforward_layernorm_2: load_optional_norm_or_placeholder(
                    &state_plan,
                    &mapped,
                    &format!("blk.{i}.pre_ffw_norm_2.weight"),
                    mlx_device,
                    "pre_ffw_norm_2",
                )?,
                post_feedforward_layernorm_1: load_optional_norm_or_placeholder(
                    &state_plan,
                    &mapped,
                    &format!("blk.{i}.post_ffw_norm_1.weight"),
                    mlx_device,
                    "post_ffw_norm_1",
                )?,
                post_feedforward_layernorm_2: load_optional_norm_or_placeholder(
                    &state_plan,
                    &mapped,
                    &format!("blk.{i}.post_ffw_norm_2.weight"),
                    mlx_device,
                    "post_ffw_norm_2",
                )?,
            };

            // -- Layer scalar (F32) --
            let layer_scalar = state_plan
                .load_mapped(&mapped, &format!("blk.{i}.layer_output_scale.weight"))
                .map_err(|e| anyhow::anyhow!("layer {i} layer_scalar: {e}"))?;

            // -- Per-layer config --
            let hd = cfg.head_dim_for_layer(i);
            let nkv = cfg.num_kv_heads_for_layer(i);
            let is_full = cfg.is_full_attention(i);
            let layer_type = if is_full {
                LayerType::Full
            } else {
                LayerType::Sliding
            };

            // -- KV cache allocation (identical to the old load_from_candle) --
            // ADR-040 §3.5 iter-A5c (cfa-A5b MAJOR #3): route the
            // per-layer-type → (is_ring, capacity) mapping through the
            // extracted `layer_type_to_alloc_params` helper so the
            // iter-A5c regression test can exercise the SAME mapping
            // production runs. `is_ring` is not consumed by this allocator
            // (this path constructs the legacy `MlxKvCache` with explicit
            // `is_sliding: !is_full`), but capacity wiring matches the
            // helper so a future branch-swap of Full/Sliding in the helper
            // would also break this production alloc path.
            let (_is_ring, capacity) =
                crate::inference::models::gemma4::kv_cache::layer_type_to_alloc_params(
                    layer_type,
                    cfg.sliding_window,
                    cfg.max_position_embeddings,
                );
            // TurboQuant 4-bit nibble-packed indices + F32 norms (ADR-007 Phase 1.2).
            // D=256: 1 norm per position (norms_per_pos=1).
            // D=512: 2 per-block norms per position (norms_per_pos=2),
            //   per AmesianX cpy-utils.cuh:241-269 (ADR-007 iter-15 per-block norm).
            let packed_bytes = nkv * capacity * (hd / 2);
            let norms_per_pos = (hd / 256).max(1);
            let norms_elements = nkv * capacity * norms_per_pos;
            let norms_bytes = norms_elements * 4; // f32 = 4 bytes

            let k_packed = mlx_device
                .alloc_buffer(
                    packed_bytes,
                    mlx_native::DType::U8,
                    vec![nkv, capacity, hd / 2],
                )
                .map_err(|e| anyhow::anyhow!("KV cache K packed alloc: {e}"))?;
            let k_norms = mlx_device
                .alloc_buffer(
                    norms_bytes,
                    mlx_native::DType::F32,
                    if norms_per_pos == 1 {
                        vec![nkv, capacity]
                    } else {
                        vec![nkv, capacity, norms_per_pos]
                    },
                )
                .map_err(|e| anyhow::anyhow!("KV cache K norms alloc: {e}"))?;
            let v_packed = mlx_device
                .alloc_buffer(
                    packed_bytes,
                    mlx_native::DType::U8,
                    vec![nkv, capacity, hd / 2],
                )
                .map_err(|e| anyhow::anyhow!("KV cache V packed alloc: {e}"))?;
            let v_norms = mlx_device
                .alloc_buffer(
                    norms_bytes,
                    mlx_native::DType::F32,
                    if norms_per_pos == 1 {
                        vec![nkv, capacity]
                    } else {
                        vec![nkv, capacity, norms_per_pos]
                    },
                )
                .map_err(|e| anyhow::anyhow!("KV cache V norms alloc: {e}"))?;

            kv_caches.push(MlxKvCache {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
                capacity,
                is_sliding: !is_full,
                write_pos: 0,
                seq_len: 0,
            });

            cum_moe_ns += t_moe.elapsed().as_nanos();
            let t_misc = std::time::Instant::now();
            layers.push(MlxDecoderLayerWeights {
                attn,
                mlp,
                moe,
                norms,
                layer_scalar,
                head_dim: hd,
                num_kv_heads: nkv,
                layer_type,
            });
            progress.on_layer(i + 1);
            cum_misc_ns += t_misc.elapsed().as_nanos();
        }
        progress.finish();
        if load_timing {
            tracing::info!(
                "[LOAD_TIMING] layer_loop_buckets attn={:.0}ms mlp={:.0}ms moe={:.0}ms misc(norms+push+progress)={:.0}ms n_layers={}",
                cum_attn_ns as f64 / 1e6,
                cum_mlp_ns as f64 / 1e6,
                cum_moe_ns as f64 / 1e6,
                cum_misc_ns as f64 / 1e6,
                num_layers,
            );
            // ADR-028 iter-463: MoE sub-buckets
            let cum_moe_other_ns = cum_moe_ns
                .saturating_sub(cum_moe_gate_up_ns + cum_moe_down_ns + cum_moe_router_cpu_ns);
            tracing::info!(
                "[LOAD_TIMING] moe_sub_buckets gate_up={:.0}ms down={:.0}ms router_cpu={:.0}ms other(router_proj+scales+placeholder)={:.0}ms",
                cum_moe_gate_up_ns as f64 / 1e6,
                cum_moe_down_ns as f64 / 1e6,
                cum_moe_router_cpu_ns as f64 / 1e6,
                cum_moe_other_ns as f64 / 1e6,
            );
        }
        tracing::info!(
            "Loaded {}/{} mlx-native layer weights from GGUF (including MoE)",
            num_layers,
            num_layers
        );

        // -- Allocate activation buffers --
        let mut activations = alloc_activation_buffers(mlx_device, cfg)?;

        // -- RoPE freq_factors from GGUF --
        if state_plan.is_present("rope_freqs.weight")? {
            let ff_buf = state_plan
                .load_mapped(&mapped, "rope_freqs.weight")
                .map_err(|e| anyhow::anyhow!("rope_freqs: {e}"))?;
            activations.rope_freq_factors_gpu = ff_buf;
        }

        // -- Build result --
        let mut result = Ok(Self {
            native_activation_epoch: std::sync::atomic::AtomicU64::new(0),
            embed_weight,
            layers,
            final_norm,
            lm_head,
            hidden_size: cfg.hidden_size,
            vocab_size: cfg.vocab_size,
            num_attention_heads: cfg.num_attention_heads,
            rms_norm_eps: cfg.rms_norm_eps as f32,
            final_logit_softcapping: cfg.final_logit_softcapping.map(|v| v as f32),
            kv_caches,
            activations,
            sliding_window: cfg.sliding_window,
            rope_theta_sliding: cfg.rope_theta_sliding as f32,
            rope_theta_global: cfg.rope_theta_global as f32,
            num_experts: cfg.num_experts,
            intermediate_size: cfg.intermediate_size,
            dense_kvs: None,
            dense_kvs_snapshot_for_lcp: None,
            hybrid_kv_snapshot_for_lcp: None,
            dense_sdpa_tmp: None,
            // iter-222 (2026-05-01): leg_f_kvs / leg_f_sdpa_tmp shadow-cache
            // fields deleted along with iter-34 dense-on-shadow Leg F branch.
            leg_hb_encoded: None,
            // ADR-028 Phase 10 (iter-347): hybrid F16-K + TQ-HB-V — Option
            // sibling, default `None` until lazy-allocated by the env-gated
            // path in `forward_decode` (Phase 10c).
            hybrid_kv: None,
            // ADR-007 Gate H release-check counter — increments per
            // forward_decode call, used by the `[HF2Q_NLL]` / `[HF2Q_DECODE_EMIT]`
            // stderr lines (W12 iter-108a blocker #1).
            decode_step: 0,
            // ADR-007 Gate H per-call regime override (W12 iter-108a blocker #3).
            // Default == today's env-var-only path; setter flips it for two-
            // regime-one-process release-check runs.
            decode_regime: DecodeRegime::Default,
            // iter-108a-fix (W15, 2026-04-25): cache the "Gate H inactive"
            // predicate so the decode hot path can elide the per-token
            // NLL/emit/replay block AND the per-layer regime-match site
            // when no Gate H hooks are armed. INVESTIGATION_ENV is a
            // process-lifetime LazyLock so this snapshot stays valid
            // until set_decode_regime is called (which refreshes it).
            gate_h_inactive: {
                let env = &*INVESTIGATION_ENV;
                !env.emit_nll && !env.decode_emit_tokens && env.decode_input_tokens.is_empty()
                // decode_regime is Default at construction, so the regime
                // arm is true here without an extra read.
                // replay_tokens is empty at construction (default Vec::new()),
                // so it does not flip gate_h_inactive here either.
            },
            // W21 iter-108b: per-instance replay vector for the in-process
            // two-regime Gate H run.  Empty by default → no behavior change
            // from iter-108a; populated only via [`set_replay_tokens`].
            replay_tokens: Vec::new(),
            // W39 iter-112b: per-instance dump-config overrides.  None by
            // default → SDPA-out dump gate falls back to INVESTIGATION_ENV,
            // bit-identical to pre-iter-112b.  parity_quality sets these
            // before each Gate H pass so the dumps land in the per-pass
            // dir even though INVESTIGATION_ENV's LazyLock is frozen.
            dump_dir_override: None,
            dump_all_cache_override: None,
            // W39 iter-112b: per-instance decode-step dump counter.  Replaces
            // the old process-static AtomicUsize so the SDPA dump gate's
            // [0, max_pos) window resets at the start of each Gate H pass.
            decode_step_dump_counter: 0,
            // ADR-030 Phase 4 — capture session NOT installed by default.
            // Spec-decode orchestrator installs via install_dflash_capture()
            // before calling forward_prefill_batched.
            dflash_capture: None,
            // ADR-029 iter-175 Step 1f — lazy-baked per-(F32, 1, hs) rms_norm
            // record.  Shared across all hs-norm call sites in decode.
            decode_record_rms_norm_f32_hs: std::sync::OnceLock::new(),
            // ADR-040 iter-G(a) — multi-seq cross-slot prefill descriptor not
            // installed by default. The forward_prefill_batched_multi_seq
            // wrapper sets it per-call; None preserves byte-identical single-seq.
            multi_seq_prefill: None,
        });

        // Pre-initialize constant param buffers so we never write them
        // inside the hot forward_decode path.
        if let Ok(ref mut w) = result {
            // Softcap params: [cap, n_elements_as_f32_bits]
            if let Some(cap) = w.final_logit_softcapping {
                let p: &mut [f32] = w
                    .activations
                    .softcap_params
                    .as_mut_slice()
                    .map_err(|e| anyhow::anyhow!("softcap_params init: {e}"))?;
                p[0] = cap;
                p[1] = f32::from_bits(w.vocab_size as u32);
            }
            // Argmax params: [vocab_size]
            {
                let p: &mut [u32] = w
                    .activations
                    .argmax_params
                    .as_mut_slice()
                    .map_err(|e| anyhow::anyhow!("argmax_params init: {e}"))?;
                p[0] = w.vocab_size as u32;
            }
        }

        if let Ok(ref weights) = result {
            let storage = weights
                .ordinary_gguf_storage_summary()
                .context("account Gemma ordinary GGUF storage")?;
            let named_anonymous_bytes = verify_ordinary_gguf_storage(&storage)?;
            tracing::info!(
                ordinary_views = storage.ordinary.unique_matrix_views,
                ordinary_file_backed_bytes = storage.ordinary.file_backed_bytes,
                ordinary_anonymous_bytes = storage.ordinary.anonymous_bytes,
                named_anonymous_buffers = storage.named_anonymous.len(),
                named_anonymous_bytes,
                "Gemma matrices and ordinary state retain exact scoped GGUF storage"
            );
        }

        result
    }

    /// Set the decode regime for the next prefill+decode trajectory.
    ///
    /// ADR-007 Gate H (W12 iter-108a blocker #3) — flips the SDPA-mode
    /// gate at `forward_mlx.rs::forward_decode` between TQ-active and
    /// dense-active without re-loading the model from GGUF.  The four
    /// codebook-bits gates (`forward_mlx.rs:1100/1234`, `forward_prefill
    /// .rs:330`) are *not* affected by this setter — the codebook width
    /// is a representation choice that stays consistent across both
    /// regimes (Gate H runs both regimes on the same KV format).  Only
    /// the SDPA-reader (the `use_dense_sdpa` gate) consults the override.
    ///
    /// Calling this also resets the per-instance step counter so each
    /// regime's stderr `[HF2Q_NLL]` / `[HF2Q_DECODE_EMIT]` lines start
    /// at `step=0`, matching the audit-binary contract (every audit
    /// invocation is a fresh process today).
    ///
    /// Default-mode behavior (i.e. `regime == DecodeRegime::Default`)
    /// is *byte-identical* to today's env-var-only path — see the
    /// `forward_mlx.rs::forward_decode` use_dense_sdpa gate for the
    /// invariant.  Non-Default regimes ignore `HF2Q_USE_DENSE` and
    /// `HF2Q_LAYER_POLICY` for the duration of the next decode loop.
    ///
    /// (Wired by iter-108b's release-check.sh Gate 5 harness; no in-tree
    /// caller as of iter-108a — the surface is designed for the
    /// iter-108b two-regime release-check entry point.)
    /// ADR-020 AC#5 Iter D — overlay a DWQ-trained mlx-affine safetensors
    /// file on top of an already-GGUF-loaded model.  For each Linear
    /// stem present in the safetensors file (`<stem>.weight`/.scales/.biases`
    /// triplet), the matching slot in `MlxModelWeights` is replaced by
    /// an affine-mode `MlxQWeight` (per Iter B + Iter C dispatch routing).
    ///
    /// The safetensors `bits` + `group_size` are read from the file's
    /// metadata (embedded by `train_all_linears_dwq` since AC#5 Iter D).
    /// Older DWQ safetensors without metadata fall back to bits=4,
    /// group_size=32 (the production default).
    ///
    /// Stem mapping (dense layers only — MoE expert tensors are skipped
    /// with a warning, tracked as Iter C2):
    ///
    /// | Stem | Slot |
    /// |---|---|
    /// | `blk.{i}.attn_q` | `layers[i].attn.q_proj` |
    /// | `blk.{i}.attn_k` | `layers[i].attn.k_proj` |
    /// | `blk.{i}.attn_v` | `layers[i].attn.v_proj` (if Some) |
    /// | `blk.{i}.attn_output` | `layers[i].attn.o_proj` |
    /// | `blk.{i}.ffn_gate` | `layers[i].mlp.gate_proj` |
    /// | `blk.{i}.ffn_up` | `layers[i].mlp.up_proj` |
    /// | `blk.{i}.ffn_down` | `layers[i].mlp.down_proj` |
    ///
    /// Returns the count of overridden Linears.  Logs each unmatched
    /// stem at `tracing::warn!` so operators can audit which trained
    /// tensors were ignored.
    pub fn apply_dwq_overlay(
        &mut self,
        device: &MlxDevice,
        path: &std::path::Path,
    ) -> Result<usize> {
        use crate::core::mlx_safetensors_loader::MlxAffineLinear;
        use anyhow::Context;

        anyhow::ensure!(
            self.native_activation_epoch
                .load(std::sync::atomic::Ordering::Acquire)
                == 0,
            "Gemma DWQ overlay cannot mutate weights after native routes are activated; reload the model with the overlay"
        );

        let bytes = std::fs::read(path)
            .with_context(|| format!("apply_dwq_overlay: read {}", path.display()))?;

        // Pull metadata via `read_metadata` (the deserialized SafeTensors
        // hides its `Metadata` field; `read_metadata` is the public path).
        let (_n, metadata_obj) = safetensors::SafeTensors::read_metadata(&bytes)
            .map_err(|e| anyhow::anyhow!("apply_dwq_overlay: read_metadata: {e:?}"))?;

        let (bits, group_size) = parse_dwq_overlay_metadata(metadata_obj.metadata().as_ref())
            .with_context(|| format!("apply_dwq_overlay: parse metadata of {}", path.display()))?;

        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| anyhow::anyhow!("apply_dwq_overlay: deserialize safetensors: {e:?}"))?;

        // Walk all `<stem>.weight` keys; require `<stem>.scales` +
        // `<stem>.biases` to be present too.
        let mut stems: Vec<String> = Vec::new();
        for name in st.names() {
            if let Some(stem) = name.strip_suffix(".weight") {
                if st.tensor(&format!("{stem}.scales")).is_err() {
                    continue;
                }
                if st.tensor(&format!("{stem}.biases")).is_err() {
                    continue;
                }
                stems.push(stem.to_string());
            }
        }

        let mut overridden: usize = 0;
        let mut unknown_skipped: usize = 0;

        // ADR-020 AC#5 Iter C2.2 — stage MoE per-expert linears for
        // post-pass aggregation.  Key: `(layer_idx, MoeBaseRole)`,
        // value: Vec<(expert_idx, MlxAffineLinear)>.
        type MoeBucket =
            std::collections::HashMap<(usize, MoeBaseRole), Vec<(usize, MlxAffineLinear)>>;
        let mut moe_buckets: MoeBucket = std::collections::HashMap::new();

        for stem in &stems {
            let linear = MlxAffineLinear::from_safetensors(&st, stem, bits, group_size)
                .with_context(|| format!("apply_dwq_overlay: parse {stem}"))?;

            // Match `blk.{i}.<role>` patterns.
            let after_blk = match stem.strip_prefix("blk.") {
                Some(s) => s,
                None => {
                    tracing::warn!(stem = %stem, "DWQ overlay: stem does not start with 'blk.'; skipping");
                    unknown_skipped += 1;
                    continue;
                }
            };
            let dot = match after_blk.find('.') {
                Some(d) => d,
                None => {
                    tracing::warn!(stem = %stem, "DWQ overlay: stem missing '.<role>'; skipping");
                    unknown_skipped += 1;
                    continue;
                }
            };
            let layer_idx: usize = match after_blk[..dot].parse() {
                Ok(v) => v,
                Err(_) => {
                    tracing::warn!(stem = %stem, "DWQ overlay: layer idx not numeric; skipping");
                    unknown_skipped += 1;
                    continue;
                }
            };
            if layer_idx >= self.layers.len() {
                tracing::warn!(stem = %stem, layer = layer_idx, "DWQ overlay: layer idx out of range; skipping");
                unknown_skipped += 1;
                continue;
            }
            let role = &after_blk[(dot + 1)..];
            match parse_dwq_overlay_role(role) {
                DwqOverlayRole::AttnQ => {
                    self.layers[layer_idx].attn.q_proj =
                        MlxQWeight::from_mlx_affine_linear(device, &linear).with_context(|| {
                            format!("apply_dwq_overlay: build qweight for {stem}")
                        })?;
                }
                DwqOverlayRole::AttnK => {
                    self.layers[layer_idx].attn.k_proj =
                        MlxQWeight::from_mlx_affine_linear(device, &linear).with_context(|| {
                            format!("apply_dwq_overlay: build qweight for {stem}")
                        })?;
                }
                DwqOverlayRole::AttnV => {
                    if self.layers[layer_idx].attn.v_proj.is_some() {
                        self.layers[layer_idx].attn.v_proj = Some(
                            MlxQWeight::from_mlx_affine_linear(device, &linear).with_context(
                                || format!("apply_dwq_overlay: build qweight for {stem}"),
                            )?,
                        );
                    } else {
                        tracing::warn!(stem = %stem, "DWQ overlay: attn_v but slot is None (k_eq_v); skipping");
                        unknown_skipped += 1;
                        continue;
                    }
                }
                DwqOverlayRole::AttnOutput => {
                    self.layers[layer_idx].attn.o_proj =
                        MlxQWeight::from_mlx_affine_linear(device, &linear).with_context(|| {
                            format!("apply_dwq_overlay: build qweight for {stem}")
                        })?;
                }
                DwqOverlayRole::FfnGate => {
                    self.layers[layer_idx].mlp.gate_proj =
                        MlxQWeight::from_mlx_affine_linear(device, &linear).with_context(|| {
                            format!("apply_dwq_overlay: build qweight for {stem}")
                        })?;
                }
                DwqOverlayRole::FfnUp => {
                    self.layers[layer_idx].mlp.up_proj =
                        MlxQWeight::from_mlx_affine_linear(device, &linear).with_context(|| {
                            format!("apply_dwq_overlay: build qweight for {stem}")
                        })?;
                }
                DwqOverlayRole::FfnDown => {
                    self.layers[layer_idx].mlp.down_proj =
                        MlxQWeight::from_mlx_affine_linear(device, &linear).with_context(|| {
                            format!("apply_dwq_overlay: build qweight for {stem}")
                        })?;
                }
                DwqOverlayRole::MoeExpert => {
                    if let Some((base, expert_idx)) = parse_dwq_moe_expert_role(role) {
                        moe_buckets
                            .entry((layer_idx, base))
                            .or_default()
                            .push((expert_idx, linear));
                    } else {
                        tracing::warn!(stem = %stem, role = %role, "DWQ overlay: malformed MoE expert stem; skipping");
                        unknown_skipped += 1;
                    }
                    continue;
                }
                DwqOverlayRole::Unknown => {
                    tracing::warn!(stem = %stem, role = %role, "DWQ overlay: unknown role; skipping");
                    unknown_skipped += 1;
                    continue;
                }
            }
            overridden += 1;
            tracing::debug!(stem = %stem, "DWQ overlay applied (dense)");
        }

        // ADR-020 AC#5 Iter C2.2 — second pass: aggregate per-expert
        // bucketed Linears into MlxAffineMoeStack and assign to the
        // matching MoE slot.  Verifies expert indices form a contiguous
        // 0..n_experts range with consistent shape across experts.
        let mut moe_stacked: usize = 0;
        for ((layer_idx, base), mut linears) in moe_buckets.into_iter() {
            // Sort by expert idx, dedup, validate contiguous 0..n_experts.
            linears.sort_by_key(|(e, _)| *e);
            let n_experts = linears.len();
            for (i, (e, _)) in linears.iter().enumerate() {
                if *e != i {
                    anyhow::bail!(
                        "DWQ overlay MoE bucket (layer={layer_idx}, base={:?}) has non-contiguous expert idx (got {} at slot {})",
                        base,
                        e,
                        i,
                    );
                }
            }
            // All experts share shape — validate from the first.
            let n = linears[0].1.n;
            let k = linears[0].1.k;
            let bits_per = linears[0].1.bits;
            let gs_per = linears[0].1.group_size;
            for (e, l) in &linears[1..] {
                if l.n != n || l.k != k || l.bits != bits_per || l.group_size != gs_per {
                    anyhow::bail!(
                        "DWQ overlay MoE bucket (layer={layer_idx}, base={:?}) expert {} shape ({},{},bits={},gs={}) ≠ expert 0 ({},{},bits={},gs={})",
                        base,
                        e,
                        l.n,
                        l.k,
                        l.bits,
                        l.group_size,
                        n,
                        k,
                        bits_per,
                        gs_per,
                    );
                }
            }
            if bits_per != 4 || gs_per != 32 {
                anyhow::bail!(
                    "DWQ overlay MoE bucket (layer={layer_idx}, base={:?}): only bits=4 group_size=32 supported in Iter C2.2 (got bits={}, gs={})",
                    base,
                    bits_per,
                    gs_per,
                );
            }
            let pack_factor = 32 / bits_per as usize;
            let k_packed = k / pack_factor;
            let groups_per_row = k / (gs_per as usize);

            // Pack each expert's q_int → U32 and convert F32 → BF16
            // for scales/biases (the `quantized_matmul_id` kernel's
            // native dtype, mirroring mlx-lm's BF16 on-disk convention).
            let stack_words = n_experts * n * k_packed;
            let mut packed_stack: Vec<u32> = vec![0u32; stack_words];
            let mut scales_stack_bf16: Vec<u16> = vec![0u16; n_experts * n * groups_per_row];
            let mut biases_stack_bf16: Vec<u16> = vec![0u16; n_experts * n * groups_per_row];
            for (e, lin) in &linears {
                for row in 0..n {
                    for kp in 0..k_packed {
                        let mut word: u32 = 0;
                        for j in 0..pack_factor {
                            let code = lin.q_int[row * k + kp * pack_factor + j] as u32;
                            debug_assert!(code <= 0xF);
                            word |= (code & 0xF) << (j * 4);
                        }
                        packed_stack[((*e * n) + row) * k_packed + kp] = word;
                    }
                }
                let s_offset = e * n * groups_per_row;
                for (i, v) in lin.scales.iter().enumerate() {
                    scales_stack_bf16[s_offset + i] = half::bf16::from_f32(*v).to_bits();
                }
                for (i, v) in lin.biases.iter().enumerate() {
                    biases_stack_bf16[s_offset + i] = half::bf16::from_f32(*v).to_bits();
                }
            }

            // Allocate GPU buffers + upload.
            let mut weight_buf = device
                .alloc_buffer(
                    stack_words * std::mem::size_of::<u32>(),
                    mlx_native::DType::U32,
                    vec![n_experts, n, k_packed],
                )
                .map_err(|e| anyhow::anyhow!("MoE stack weight alloc: {e}"))?;
            weight_buf
                .as_mut_slice::<u32>()
                .map_err(|e| anyhow::anyhow!("MoE stack weight slice: {e}"))?
                .copy_from_slice(&packed_stack);

            let mut scales_buf = device
                .alloc_buffer(
                    scales_stack_bf16.len() * std::mem::size_of::<u16>(),
                    mlx_native::DType::BF16,
                    vec![n_experts, n, groups_per_row],
                )
                .map_err(|e| anyhow::anyhow!("MoE stack scales alloc: {e}"))?;
            scales_buf
                .as_mut_slice::<u16>()
                .map_err(|e| anyhow::anyhow!("MoE stack scales slice: {e}"))?
                .copy_from_slice(&scales_stack_bf16);

            let mut biases_buf = device
                .alloc_buffer(
                    biases_stack_bf16.len() * std::mem::size_of::<u16>(),
                    mlx_native::DType::BF16,
                    vec![n_experts, n, groups_per_row],
                )
                .map_err(|e| anyhow::anyhow!("MoE stack biases alloc: {e}"))?;
            biases_buf
                .as_mut_slice::<u16>()
                .map_err(|e| anyhow::anyhow!("MoE stack biases slice: {e}"))?
                .copy_from_slice(&biases_stack_bf16);

            let stack = MlxAffineMoeStack {
                weight: weight_buf,
                scales: scales_buf,
                biases: biases_buf,
                n,
                k,
                bits: bits_per,
                group_size: gs_per as u32,
                num_experts: n_experts,
            };
            let layer = &mut self.layers[layer_idx];
            match base {
                MoeBaseRole::GateUp => {
                    layer.moe.gate_up_affine = Some(stack);
                }
                MoeBaseRole::Down => {
                    layer.moe.down_affine = Some(stack);
                }
                MoeBaseRole::Gate | MoeBaseRole::Up => {
                    // Separate gate / up case — not yet wired into a
                    // dispatch path (Iter C2.3 only routes the FUSED
                    // gate_up case for qwen3.5).  Surface as warning;
                    // operator can revisit when a non-fused MoE GGUF
                    // arch shows up.
                    tracing::warn!(
                        layer_idx,
                        ?base,
                        n_experts,
                        "DWQ overlay: separate gate/up MoE case not wired to dispatch yet (qwen3.5 uses fused gate_up); stack constructed but unused"
                    );
                    let _ = stack;
                }
            }
            moe_stacked += 1;
            tracing::debug!(
                layer_idx,
                ?base,
                n_experts,
                n,
                k,
                "DWQ overlay applied (MoE stack)"
            );
        }

        tracing::info!(
            overridden,
            moe_stacked,
            unknown_skipped,
            bits,
            group_size,
            "DWQ overlay applied: {overridden} dense Linears + {moe_stacked} MoE stacks"
        );
        Ok(overridden + moe_stacked)
    }

    #[allow(dead_code)]
    pub fn set_decode_regime(&mut self, regime: DecodeRegime) {
        self.decode_regime = regime;
        self.decode_step = 0;
        // W39 iter-112b: also reset the per-instance SDPA-dump step
        // counter so each Gate H regime's [0, max_pos) dump window
        // restarts at 0 (the old process-static AtomicUsize accumulated
        // across passes and silently dropped pass 2's dumps).
        self.decode_step_dump_counter = 0;
        // iter-108a-fix (W15): re-evaluate the Gate H elision flag.
        // Non-Default regime forces the per-layer SDPA-gate match path,
        // so we can't elide it; the env-var hooks could still be off,
        // but we only treat the path as "inactive" when ALL Gate H
        // surfaces are quiet (env hooks unset AND regime is Default).
        let env = &*INVESTIGATION_ENV;
        self.gate_h_inactive = matches!(regime, DecodeRegime::Default)
            && !env.emit_nll
            && !env.decode_emit_tokens
            && env.decode_input_tokens.is_empty()
            && self.replay_tokens.is_empty();
    }

    /// Set the per-instance replay vector for the next decode trajectory
    /// (ADR-007 Gate H, W21 iter-108b two-regime in-process harness).
    ///
    /// When non-empty, the post-argmax tail of [`Self::forward_decode`]
    /// substitutes `replay[step]` for the model's argmax pick — same
    /// contract as `HF2Q_DECODE_INPUT_TOKENS` but bypassing the
    /// `INVESTIGATION_ENV` `LazyLock` (which is frozen at first access
    /// and so cannot be flipped between the dense and TQ passes of a
    /// single Gate H run).  After the replay buffer is exhausted, the
    /// loop falls through to the live argmax pick — identical fall-back
    /// to the env-var path.
    ///
    /// Pass an empty `Vec` to clear the override.  Also resets
    /// [`Self::decode_step`] (matching `set_decode_regime` semantics) so
    /// each replay run's `step` counter starts at 0, and refreshes
    /// [`Self::gate_h_inactive`] so the per-token instrumentation block
    /// runs whenever a replay is active even when env hooks are silent.
    ///
    /// (Wired by iter-108b's `parity_quality::run_two_regime_decode`;
    /// no other in-tree caller.)
    #[allow(dead_code)]
    pub fn set_replay_tokens(&mut self, replay: Vec<u32>) {
        self.replay_tokens = replay;
        self.decode_step = 0;
        // W39 iter-112b: see `set_decode_regime` — reset the SDPA-dump
        // step counter for the upcoming pass.
        self.decode_step_dump_counter = 0;
        let env = &*INVESTIGATION_ENV;
        self.gate_h_inactive = matches!(self.decode_regime, DecodeRegime::Default)
            && !env.emit_nll
            && !env.decode_emit_tokens
            && env.decode_input_tokens.is_empty()
            && self.replay_tokens.is_empty();
    }

    /// Set per-instance SDPA-dump overrides for the next decode trajectory
    /// (ADR-007 Gate H, W39 iter-112b two-regime in-process harness).
    ///
    /// `INVESTIGATION_ENV` is a `LazyLock` populated by
    /// `INVESTIGATION_ENV.activate()` at `main.rs::main` *before* `Cli::parse`.
    /// W21's `parity_quality::run_two_regime_decode` sets `HF2Q_DUMP_DIR` and
    /// `HF2Q_DUMP_ALL_CACHE` at run time, but those `set_var` calls reach
    /// `std::env` *after* the LazyLock has frozen, so the SDPA-out dump gate
    /// at `forward_decode` and the dump-path formatter inside `dumps::dump_f32`
    /// keep reading the pre-launch (default) values — `dump_all_cache=false`
    /// and `dump_dir=/tmp` — silently dropping every Gate H dump.
    ///
    /// This setter exposes the per-instance override surface.  Both
    /// arguments are `Option`: `Some(_)` overrides the corresponding
    /// `INVESTIGATION_ENV` field for SDPA-out gating + path formation;
    /// `None` falls back to `INVESTIGATION_ENV` (i.e. the default
    /// env-var-only path is bit-identical to pre-iter-112b).
    ///
    /// Also resets [`Self::decode_step_dump_counter`] so the upcoming pass's
    /// `[0, max_pos)` dump window starts at step 0.
    ///
    /// (Wired by iter-112b's `parity_quality::run_two_regime_decode`; no
    /// other in-tree caller.)
    #[allow(dead_code)]
    pub fn set_dump_overrides(&mut self, dir: Option<std::path::PathBuf>, all_cache: Option<bool>) {
        self.dump_dir_override = dir;
        self.dump_all_cache_override = all_cache;
        self.decode_step_dump_counter = 0;
    }

    /// Reset the per-instance SDPA-dump step counter without touching
    /// regime / replay / override state.  W39 iter-112b: most Gate H call
    /// sites reset via `set_decode_regime` / `set_replay_tokens` /
    /// `set_dump_overrides`; this is the explicit setter for callers that
    /// only want to roll the counter back (e.g. between sub-passes within
    /// the same regime).
    #[allow(dead_code)]
    pub fn reset_decode_step_dump_counter(&mut self) {
        self.decode_step_dump_counter = 0;
    }
}

/// Wedge-4 / iter-227 — `MlxMoeWeights::dense_placeholder` invariants.
///
/// These tests pin the placeholder constructor's contract at the Rust
/// type-system level so a future refactor cannot silently re-introduce
/// the iter-227 dispatch crash. They do NOT exercise GPU kernels — the
/// constructor is pure CPU + tiny MlxBuffer allocations.
///
/// Live-load coverage of the conditional MoE-expert load itself
/// (skipping `blk.0.ffn_gate_up_exps.weight` when a dense GGUF lacks it)
/// is covered by the `iter227_*` architecture-dispatch tests in
/// `serve::tests`.
#[cfg(test)]
mod native_bf16_inventory_tests {
    use super::*;

    fn weight(device: &MlxDevice, storage: DType, declared: GgmlType) -> MlxQWeight {
        let bytes_per_element = match storage {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            other => panic!("unsupported scalar fixture dtype {other}"),
        };
        let buffer = device
            .alloc_buffer(32 * 64 * bytes_per_element, storage, vec![32, 64])
            .expect("scalar matrix fixture");
        MlxQWeight::from_test_buffer(buffer, declared, 32, 64)
    }

    #[test]
    fn native_bf16_inventory_admits_only_matching_artifact_storage() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("Metal device");
        let native = weight(&device, DType::BF16, GgmlType::BF16);
        let matrix = gemma_native_bf16_matrix("native", &native)
            .expect("matching BF16 metadata")
            .expect("native BF16 matrix");
        assert_eq!((matrix.n, matrix.k), (32, 64));
        assert_eq!(matrix.reachable_row_mask, u16::MAX);

        let f16 = weight(&device, DType::F16, GgmlType::F16);
        assert!(
            gemma_native_bf16_matrix("f16", &f16)
                .expect("matching F16 metadata")
                .is_none(),
            "F16 must remain on its artifact-native F16 route"
        );
        let f32 = weight(&device, DType::F32, GgmlType::F32);
        assert!(
            gemma_native_bf16_matrix("f32", &f32)
                .expect("matching F32 metadata")
                .is_none(),
            "F32 must remain on its artifact-native F32 route"
        );
    }

    #[test]
    fn native_bf16_inventory_rejects_declared_storage_mismatch() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("Metal device");
        let mismatch = weight(&device, DType::F16, GgmlType::BF16);
        let error = match gemma_native_bf16_matrix("mismatch", &mismatch) {
            Err(error) => error,
            Ok(_) => panic!("declared BF16 over F16 bytes must fail closed"),
        };
        assert!(
            error.to_string().contains("disagrees with declared type"),
            "unexpected mismatch error: {error:#}"
        );
    }
}

#[cfg(test)]
mod dense_placeholder_tests {
    use super::*;

    /// The dense placeholder bundle MUST report `stacked_gate_up: None`
    /// AND `stacked_down: None` so the fused-id MoE dispatch's
    /// `is_some() && is_some()` gate falsifies cleanly. If a future
    /// refactor accidentally allocates Some(empty_buffer) here, the
    /// MoE dispatch would walk into 1-element buffers and produce
    /// garbage logits without panicking — far worse than today's
    /// "missing tensor" load-time bail.
    #[test]
    fn iter227_dense_placeholder_has_no_stacked_expert_buffers() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                // No Metal device available (e.g. CI without GPU);
                // skip — the live load path on M5 Max exercises this.
                eprintln!(
                    "skipping iter227_dense_placeholder_has_no_stacked_expert_buffers: no MlxDevice"
                );
                return;
            }
        };
        let moe = MlxMoeWeights::dense_placeholder(&device)
            .expect("dense_placeholder allocation must succeed on Metal device");
        assert!(
            moe.stacked_gate_up.is_none(),
            "dense placeholder MUST have stacked_gate_up = None to falsify fused-id MoE gate"
        );
        assert!(
            moe.stacked_down.is_none(),
            "dense placeholder MUST have stacked_down = None to falsify fused-id MoE gate"
        );
    }

    /// Sentinel scalars (`top_k = 0`, `moe_intermediate_size = 0`,
    /// strides = 0) make any accidental read of the placeholder fields
    /// visibly wrong instead of producing plausible-looking garbage.
    #[test]
    fn iter227_dense_placeholder_zeros_scalar_metadata() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping iter227_dense_placeholder_zeros_scalar_metadata: no MlxDevice");
                return;
            }
        };
        let moe = MlxMoeWeights::dense_placeholder(&device)
            .expect("dense_placeholder allocation must succeed on Metal device");
        assert_eq!(moe.top_k, 0, "dense placeholder must zero top_k");
        assert_eq!(
            moe.moe_intermediate_size, 0,
            "dense placeholder must zero moe_intermediate_size"
        );
        assert_eq!(moe.gate_up_expert_stride, 0);
        assert_eq!(moe.down_expert_stride, 0);
    }

    /// Allocation cost regression guard: the placeholder bundle must
    /// stay tiny so a 28-layer dense load adds <1 KB total
    /// MoE-bookkeeping overhead vs. the previous unconditional path
    /// (which would have OOM-allocated GBs of expert tensors that
    /// don't exist on disk).
    #[test]
    fn iter227_dense_placeholder_buffers_are_one_element_each() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "skipping iter227_dense_placeholder_buffers_are_one_element_each: no MlxDevice"
                );
                return;
            }
        };
        let moe = MlxMoeWeights::dense_placeholder(&device)
            .expect("dense_placeholder allocation must succeed on Metal device");
        // Each placeholder buffer is 1 F32 element = 4 bytes.
        assert_eq!(
            moe.per_expert_scale.byte_len(),
            std::mem::size_of::<f32>(),
            "per_expert_scale placeholder must be 1 F32 element"
        );
        assert_eq!(
            moe.router_combined_weight.byte_len(),
            std::mem::size_of::<f32>(),
            "router_combined_weight placeholder must be 1 F32 element"
        );
        assert_eq!(
            moe.router_proj.buffer.byte_len(),
            std::mem::size_of::<f32>(),
            "router_proj placeholder buffer must be 1 F32 element"
        );
    }
}

#[cfg(test)]
mod ordinary_gguf_state_tests {
    use super::*;
    use crate::backends::gguf::writer::GgufWriter;
    use crate::quantize::ggml_quants::GgmlType as WriterGgmlType;

    #[derive(Clone, Copy)]
    enum StateMutation {
        None,
        RequiredOutputNormQ8,
        OptionalRopeF16,
        AbsentRope,
    }

    struct SyntheticGemmaGguf {
        file: tempfile::NamedTempFile,
        cfg: Gemma4Config,
        tensor_count: usize,
        payload_bytes: u64,
    }

    struct FixtureTensor {
        name: String,
        dims: Vec<u64>,
        ggml_type: WriterGgmlType,
        payload: Vec<u8>,
    }

    fn q8_matrix(name: &str, rows: usize, cols: usize) -> FixtureTensor {
        assert_eq!(cols % 32, 0);
        FixtureTensor {
            name: name.to_owned(),
            dims: vec![cols as u64, rows as u64],
            ggml_type: WriterGgmlType::Q8_0,
            payload: vec![0; rows * (cols / 32) * 34],
        }
    }

    fn f32_matrix(name: &str, rows: usize, cols: usize) -> FixtureTensor {
        FixtureTensor {
            name: name.to_owned(),
            dims: vec![cols as u64, rows as u64],
            ggml_type: WriterGgmlType::F32,
            payload: vec![0; rows * cols * std::mem::size_of::<f32>()],
        }
    }

    fn q8_expert_stack(name: &str, experts: usize, rows: usize, cols: usize) -> FixtureTensor {
        assert_eq!(cols % 32, 0);
        FixtureTensor {
            name: name.to_owned(),
            dims: vec![cols as u64, rows as u64, experts as u64],
            ggml_type: WriterGgmlType::Q8_0,
            payload: vec![0; experts * rows * (cols / 32) * 34],
        }
    }

    fn state_tensor(name: &str, elements: usize, ggml_type: WriterGgmlType) -> FixtureTensor {
        let bytes = match ggml_type {
            WriterGgmlType::F32 => elements * std::mem::size_of::<f32>(),
            WriterGgmlType::F16 => elements * std::mem::size_of::<u16>(),
            WriterGgmlType::Q8_0 => {
                assert_eq!(elements % 32, 0);
                (elements / 32) * 34
            }
            other => panic!("unsupported synthetic state type {other:?}"),
        };
        FixtureTensor {
            name: name.to_owned(),
            dims: vec![elements as u64],
            ggml_type,
            payload: vec![0; bytes],
        }
    }

    fn synthetic_gemma(mutation: StateMutation, with_experts: bool) -> SyntheticGemmaGguf {
        let hidden = 256usize;
        let vocab = 32usize;
        let num_experts = if with_experts { 2 } else { 0 };
        let top_k_experts = if with_experts { 1 } else { 0 };
        let moe_intermediate_size = if with_experts { 32 } else { 0 };
        let mut tensors = vec![
            q8_matrix("token_embd.weight", vocab, hidden),
            q8_matrix("blk.0.attn_q.weight", hidden, hidden),
            q8_matrix("blk.0.attn_k.weight", hidden, hidden),
            q8_matrix("blk.0.attn_v.weight", hidden, hidden),
            q8_matrix("blk.0.attn_output.weight", hidden, hidden),
            q8_matrix("blk.0.ffn_gate.weight", hidden, hidden),
            q8_matrix("blk.0.ffn_up.weight", hidden, hidden),
            q8_matrix("blk.0.ffn_down.weight", hidden, hidden),
        ];
        if with_experts {
            tensors.extend([
                q8_expert_stack(
                    "blk.0.ffn_gate_up_exps.weight",
                    num_experts,
                    2 * moe_intermediate_size,
                    hidden,
                ),
                q8_expert_stack(
                    "blk.0.ffn_down_exps.weight",
                    num_experts,
                    hidden,
                    moe_intermediate_size,
                ),
                f32_matrix("blk.0.ffn_gate_inp.weight", num_experts, hidden),
            ]);
        }
        for name in [
            "output_norm.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.post_attention_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.post_ffw_norm.weight",
        ] {
            let ggml_type = if name == "output_norm.weight"
                && matches!(mutation, StateMutation::RequiredOutputNormQ8)
            {
                WriterGgmlType::Q8_0
            } else {
                WriterGgmlType::F32
            };
            tensors.push(state_tensor(name, hidden, ggml_type));
        }
        tensors.push(state_tensor(
            "blk.0.layer_output_scale.weight",
            1,
            WriterGgmlType::F32,
        ));
        if with_experts {
            for name in [
                "blk.0.pre_ffw_norm_2.weight",
                "blk.0.post_ffw_norm_1.weight",
                "blk.0.post_ffw_norm_2.weight",
            ] {
                tensors.push(state_tensor(name, hidden, WriterGgmlType::F32));
            }
            tensors.extend([
                state_tensor("blk.0.ffn_gate_inp.scale", hidden, WriterGgmlType::F32),
                state_tensor(
                    "blk.0.ffn_down_exps.scale",
                    num_experts,
                    WriterGgmlType::F32,
                ),
            ]);
        }
        if !matches!(mutation, StateMutation::AbsentRope) {
            tensors.push(state_tensor(
                "rope_freqs.weight",
                hidden / 2,
                if matches!(mutation, StateMutation::OptionalRopeF16) {
                    WriterGgmlType::F16
                } else {
                    WriterGgmlType::F32
                },
            ));
        }

        let tensor_count = tensors.len();
        let payload_bytes = tensors
            .iter()
            .map(|tensor| tensor.payload.len() as u64)
            .sum();
        let file = tempfile::NamedTempFile::new().expect("temporary Gemma GGUF");
        let sink = std::fs::File::create(file.path()).expect("create Gemma GGUF");
        let mut writer = GgufWriter::new(sink);
        writer
            .write_header(tensor_count as u64, 0)
            .expect("write Gemma GGUF header");
        let mut indices = Vec::with_capacity(tensor_count);
        for tensor in &tensors {
            indices.push(
                writer
                    .reserve_tensor_info(&tensor.name, &tensor.dims, tensor.ggml_type)
                    .unwrap_or_else(|error| {
                        panic!("reserve synthetic tensor '{}': {error}", tensor.name)
                    }),
            );
        }
        writer.pad_to_alignment().expect("align Gemma GGUF");
        for (tensor, index) in tensors.iter().zip(indices) {
            writer
                .stream_tensor_payload(index, &tensor.payload)
                .unwrap_or_else(|error| {
                    panic!("write synthetic tensor '{}': {error}", tensor.name)
                });
        }
        writer.finalize().expect("finalize Gemma GGUF");

        SyntheticGemmaGguf {
            file,
            cfg: Gemma4Config {
                vocab_size: vocab,
                hidden_size: hidden,
                intermediate_size: hidden,
                moe_intermediate_size,
                num_hidden_layers: 1,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                num_global_key_value_heads: 1,
                head_dim: hidden,
                global_head_dim: hidden,
                rms_norm_eps: 1e-6,
                rope_theta_sliding: 10_000.0,
                rope_theta_global: 1_000_000.0,
                sliding_window: 8,
                max_position_embeddings: 16,
                final_logit_softcapping: None,
                attention_bias: false,
                attention_k_eq_v: false,
                tie_word_embeddings: true,
                num_experts,
                top_k_experts,
                layer_types: vec![LayerType::Sliding],
            },
            tensor_count,
            payload_bytes,
        }
    }

    fn preflight_state_error(mutation: StateMutation) -> String {
        let fixture = synthetic_gemma(mutation, false);
        let gguf = mlx_native::gguf::GgufFile::open(fixture.file.path())
            .expect("open synthetic Gemma GGUF");
        preflight_f32_state(&gguf, &fixture.cfg)
            .expect_err("wrong F32 state storage must fail admission")
            .to_string()
    }

    #[test]
    fn activation_argmax_params_match_native_operator_contract() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(device) => device,
            Err(_) => {
                eprintln!(
                    "skipping activation_argmax_params_match_native_operator_contract: no MlxDevice"
                );
                return;
            }
        };
        let fixture = synthetic_gemma(StateMutation::None, false);
        let buffers = alloc_activation_buffers(&device, &fixture.cfg)
            .expect("allocate synthetic Gemma activation buffers");
        assert_eq!(buffers.argmax_params.dtype(), DType::U32);
        assert_eq!(buffers.argmax_params.shape(), &[1]);
        assert_eq!(buffers.argmax_params.byte_len(), std::mem::size_of::<u32>());
    }

    fn real_loader_error_before_map(mutation: StateMutation) -> Option<String> {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let fixture = synthetic_gemma(mutation, false);
        let gguf = mlx_native::gguf::GgufFile::open(fixture.file.path())
            .expect("open synthetic Gemma GGUF");
        let mut gpu = crate::serve::gpu::GpuContext::new().ok()?;
        crate::inference::models::gemma4::native_matrix::reset_map_attempts_for_test();
        let mut progress = crate::serve::header::LoadProgress::new(false, 1, 1);
        let error =
            match MlxModelWeights::load_from_gguf(&gguf, &fixture.cfg, &mut gpu, &mut progress) {
                Ok(_) => panic!("wrong F32 state storage must fail the real loader"),
                Err(error) => error,
            };
        assert_eq!(
            crate::inference::models::gemma4::native_matrix::map_attempts_for_test(),
            0,
            "state admission must fail before GGUF tensor-data mapping"
        );
        Some(error.to_string())
    }

    #[test]
    fn f32_state_catalog_is_complete_for_dense_and_moe_artifacts() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use crate::inference::models::gemma4::native_matrix::NativeF32StateRequirement::{
            Optional, Required,
        };

        for (with_experts, expert_requirement) in [(false, Optional), (true, Required)] {
            let fixture = synthetic_gemma(StateMutation::None, with_experts);
            let gguf = mlx_native::gguf::GgufFile::open(fixture.file.path())
                .expect("open synthetic Gemma GGUF");
            let plan =
                preflight_f32_state(&gguf, &fixture.cfg).expect("catalog exact Gemma F32 state");
            let actual: std::collections::BTreeMap<_, _> = plan
                .states()
                .iter()
                .map(|state| {
                    (
                        state.name.as_str(),
                        (state.requirement, state.present, state.shape.as_slice()),
                    )
                })
                .collect();
            assert_eq!(actual.len(), if with_experts { 14 } else { 12 });
            for state in plan.states() {
                let expected_bytes =
                    state.shape.iter().product::<usize>() * std::mem::size_of::<f32>();
                assert_eq!(state.byte_len, expected_bytes, "{}", state.name);
            }
            for (name, shape) in [
                ("output_norm.weight", &[256][..]),
                ("blk.0.attn_q_norm.weight", &[256][..]),
                ("blk.0.attn_k_norm.weight", &[256][..]),
                ("blk.0.attn_norm.weight", &[256][..]),
                ("blk.0.post_attention_norm.weight", &[256][..]),
                ("blk.0.ffn_norm.weight", &[256][..]),
                ("blk.0.post_ffw_norm.weight", &[256][..]),
                ("blk.0.layer_output_scale.weight", &[1][..]),
            ] {
                assert_eq!(actual[name], (Required, true, shape), "{name}");
            }
            assert_eq!(actual["rope_freqs.weight"], (Optional, true, &[128][..]));
            for name in [
                "blk.0.pre_ffw_norm_2.weight",
                "blk.0.post_ffw_norm_1.weight",
                "blk.0.post_ffw_norm_2.weight",
            ] {
                assert_eq!(
                    actual[name],
                    (expert_requirement, with_experts, &[256][..]),
                    "{name}"
                );
            }
            for (name, shape) in [
                ("blk.0.ffn_gate_inp.scale", &[256][..]),
                ("blk.0.ffn_down_exps.scale", &[2][..]),
            ] {
                if with_experts {
                    assert_eq!(actual[name], (Required, true, shape), "{name}");
                } else {
                    assert!(!actual.contains_key(name), "{name}");
                }
            }
        }
    }

    #[test]
    fn required_quantized_state_fails_hosted_admission() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let error = preflight_state_error(StateMutation::RequiredOutputNormQ8);
        assert!(error.contains("output_norm.weight"), "{error}");
        assert!(error.contains("exact F32 storage"), "{error}");
        assert!(error.contains("Q8_0"), "{error}");
    }

    #[test]
    fn optional_present_non_f32_state_fails_hosted_admission() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let error = preflight_state_error(StateMutation::OptionalRopeF16);
        assert!(error.contains("rope_freqs.weight"), "{error}");
        assert!(error.contains("exact F32 storage"), "{error}");
        assert!(error.contains("F16"), "{error}");
    }

    #[test]
    fn real_loader_rejects_state_mutations_before_mapping() {
        for mutation in [
            StateMutation::RequiredOutputNormQ8,
            StateMutation::OptionalRopeF16,
        ] {
            let Some(error) = real_loader_error_before_map(mutation) else {
                return;
            };
            assert!(error.contains("exact F32 storage"), "{error}");
        }
    }

    #[test]
    fn ordinary_real_loader_retains_all_artifact_model_bytes_file_backed() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let fixture = synthetic_gemma(StateMutation::AbsentRope, false);
        let gguf = mlx_native::gguf::GgufFile::open(fixture.file.path())
            .expect("open synthetic Gemma GGUF");
        let Some(mut gpu) = crate::serve::gpu::GpuContext::new().ok() else {
            return;
        };
        crate::inference::models::gemma4::native_matrix::reset_map_attempts_for_test();
        let mut progress = crate::serve::header::LoadProgress::new(false, 1, 1);
        let model = MlxModelWeights::load_from_gguf(&gguf, &fixture.cfg, &mut gpu, &mut progress)
            .expect("load synthetic Gemma GGUF");
        assert_eq!(
            crate::inference::models::gemma4::native_matrix::map_attempts_for_test(),
            1,
            "valid ordinary load must create exactly one scoped GGUF mapping"
        );
        let storage = model
            .ordinary_gguf_storage_summary()
            .expect("summarize ordinary Gemma model storage");
        assert_eq!(storage.ordinary.unique_matrix_views, fixture.tensor_count);
        assert_eq!(storage.ordinary.file_backed_bytes, fixture.payload_bytes);
        assert_eq!(storage.ordinary.anonymous_bytes, 0);

        let names: std::collections::BTreeSet<_> = storage
            .named_anonymous
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        assert_eq!(
            names,
            std::collections::BTreeSet::from([
                "blk.0.per_expert_scale:dense-placeholder",
                "blk.0.post_ffw_norm_1.weight:placeholder",
                "blk.0.post_ffw_norm_2.weight:placeholder",
                "blk.0.pre_ffw_norm_2.weight:placeholder",
                "blk.0.router_combined_weight:dense-placeholder",
                "blk.0.router_proj:dense-placeholder",
                "rope_freqs.weight:placeholder",
            ])
        );
        assert!(
            storage
                .named_anonymous
                .iter()
                .all(|entry| entry.byte_len == std::mem::size_of::<f32>() as u64),
            "only documented one-F32 dense placeholders may be anonymous"
        );
    }

    #[test]
    fn production_storage_gate_rejects_anonymous_or_empty_ordinary_state() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let valid = GemmaOrdinaryStorageSummary {
            ordinary: crate::serve::forward_mlx_shared::NativeMatrixStorageSummary {
                unique_matrix_views: 1,
                file_backed_bytes: 128,
                anonymous_bytes: 0,
            },
            named_anonymous: vec![NamedAnonymousModelBuffer {
                name: "documented-placeholder".to_owned(),
                byte_len: 4,
            }],
        };
        assert_eq!(verify_ordinary_gguf_storage(&valid).unwrap(), 4);

        let mut anonymous = valid.clone();
        anonymous.ordinary.anonymous_bytes = 4;
        assert!(verify_ordinary_gguf_storage(&anonymous)
            .unwrap_err()
            .to_string()
            .contains("anonymous ordinary matrix/state bytes"));

        let mut empty = valid.clone();
        empty.ordinary.unique_matrix_views = 0;
        assert!(verify_ordinary_gguf_storage(&empty)
            .unwrap_err()
            .to_string()
            .contains("retained no ordinary GGUF"));

        let mut empty_named = valid;
        empty_named.named_anonymous[0].byte_len = 0;
        assert!(verify_ordinary_gguf_storage(&empty_named)
            .unwrap_err()
            .to_string()
            .contains("is empty"));
    }

    #[test]
    fn ordinary_moe_real_loader_names_only_router_derived_storage() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let fixture = synthetic_gemma(StateMutation::None, true);
        let gguf = mlx_native::gguf::GgufFile::open(fixture.file.path())
            .expect("open synthetic Gemma MoE GGUF");
        let Some(mut gpu) = crate::serve::gpu::GpuContext::new().ok() else {
            return;
        };
        crate::inference::models::gemma4::native_matrix::reset_map_attempts_for_test();
        let mut progress = crate::serve::header::LoadProgress::new(false, 1, 1);
        let model = MlxModelWeights::load_from_gguf(&gguf, &fixture.cfg, &mut gpu, &mut progress)
            .expect("load synthetic Gemma MoE GGUF");
        assert_eq!(
            crate::inference::models::gemma4::native_matrix::map_attempts_for_test(),
            1,
            "valid ordinary MoE load must create exactly one scoped GGUF mapping"
        );
        let storage = model
            .ordinary_gguf_storage_summary()
            .expect("summarize ordinary Gemma MoE model storage");
        let transient_router_scale_bytes = fixture.cfg.hidden_size * std::mem::size_of::<f32>();
        assert_eq!(
            storage.ordinary.unique_matrix_views,
            fixture.tensor_count - 1
        );
        assert_eq!(
            storage.ordinary.file_backed_bytes,
            fixture.payload_bytes - transient_router_scale_bytes as u64
        );
        assert_eq!(storage.ordinary.anonymous_bytes, 0);
        assert_eq!(
            storage.named_anonymous,
            vec![NamedAnonymousModelBuffer {
                name: "blk.0.router_combined_weight:derived".to_owned(),
                byte_len: (fixture.cfg.hidden_size * std::mem::size_of::<f32>()) as u64,
            }]
        );
    }
}

// ---------------------------------------------------------------------------
// Activation buffer allocation
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Activation buffer allocation
// ---------------------------------------------------------------------------

/// Allocate all reusable activation buffers for the forward pass.
fn alloc_activation_buffers(
    device: &MlxDevice,
    cfg: &Gemma4Config,
) -> Result<MlxActivationBuffers> {
    let hs = cfg.hidden_size;
    let max_hd = cfg.global_head_dim; // 512
    let num_heads = cfg.num_attention_heads; // 16
    let max_kv_heads = cfg.num_key_value_heads.max(cfg.num_global_key_value_heads);
    let vocab = cfg.vocab_size;
    let interm = cfg.intermediate_size;
    let moe_interm = cfg.moe_intermediate_size;
    let num_experts = cfg.num_experts;

    let f32_sz = std::mem::size_of::<f32>();
    let u32_sz = std::mem::size_of::<u32>();

    let alloc_f32 = |n: usize, name: &str| -> Result<MlxBuffer> {
        device
            .alloc_buffer(n * f32_sz, mlx_native::DType::F32, vec![n])
            .map_err(|e| anyhow::anyhow!("alloc {name} ({n} f32): {e}"))
    };
    let alloc_u32 = |n: usize, name: &str| -> Result<MlxBuffer> {
        device
            .alloc_buffer(n * u32_sz, mlx_native::DType::U32, vec![n])
            .map_err(|e| anyhow::anyhow!("alloc {name} ({n} u32): {e}"))
    };

    // RMS norm params: [eps, dim] as f32
    let mut norm_params = alloc_f32(2, "norm_params")?;
    {
        let p: &mut [f32] = norm_params
            .as_mut_slice()
            .map_err(|e| anyhow::anyhow!("norm_params init: {e}"))?;
        p[0] = cfg.rms_norm_eps as f32;
        p[1] = hs as f32;
    }

    // Softcap params
    let softcap_params = alloc_f32(2, "softcap_params")?;

    // Native argmax consumes one U32 parameter: the row width (vocab size).
    let argmax_params = alloc_u32(1, "argmax_params")?;

    Ok(MlxActivationBuffers {
        embedding_token_id: alloc_u32(1, "embedding_token_id")?,
        hidden: alloc_f32(hs, "hidden")?,
        attn_q: alloc_f32(num_heads * max_hd, "attn_q")?,
        attn_k: alloc_f32(max_kv_heads * max_hd, "attn_k")?,
        attn_out: alloc_f32(hs, "attn_out")?,
        norm_out: alloc_f32(hs, "norm_out")?,
        residual: alloc_f32(hs, "residual")?,
        mlp_gate: alloc_f32(interm, "mlp_gate")?,
        mlp_up: alloc_f32(interm, "mlp_up")?,
        mlp_fused: alloc_f32(interm.max(moe_interm), "mlp_fused")?,
        mlp_down: alloc_f32(hs, "mlp_down")?,
        sdpa_out: alloc_f32(num_heads * max_hd, "sdpa_out")?,
        sdpa_tmp: {
            let tmp_bytes = mlx_native::ops::flash_attn_vec_tq::tmp_buffer_bytes(
                num_heads as u32,
                max_hd as u32,
            );
            device
                .alloc_buffer(tmp_bytes, mlx_native::DType::F32, vec![tmp_bytes / 4])
                .map_err(|e| anyhow::anyhow!("sdpa_tmp alloc: {e}"))?
        },
        norm_params,
        position: alloc_u32(1, "position")?,
        softcap_params,
        argmax_index: alloc_u32(1, "argmax_index")?,
        argmax_value: alloc_f32(1, "argmax_value")?,
        argmax_params,
        logits: alloc_f32(vocab, "logits")?,
        // G4-CFA-5b (2026-05-23): MoE activation buffers fall back to
        // 1-element placeholders on dense GGUFs (num_experts == 0 ⇒
        // alloc_buffer with 0 bytes errors with "Buffer byte length must
        // be > 0"). These buffers are read EXCLUSIVELY by the MoE forward
        // path; the dense `forward_tree_verify_gpu` entry never touches
        // them. Same `iter-227` placeholder-bundle pattern as
        // `MlxMoeWeights::dense_placeholder` + the optional norms above.
        moe_router_logits: alloc_f32(num_experts.max(1), "moe_router_logits")?,
        moe_expert_out: alloc_f32(hs.max(max_kv_heads * max_hd), "moe_expert_out")?,
        moe_accum: alloc_f32(hs, "moe_accum")?,
        moe_norm_out: alloc_f32(hs, "moe_norm_out")?,
        router_norm_out: alloc_f32(hs, "router_norm_out")?,
        // Fused _id dispatch buffers (sized for top_k = cfg.top_k_experts).
        // G4-CFA-5b: clamp to ≥1 element for dense GGUFs (see above).
        moe_expert_ids: alloc_u32(cfg.top_k_experts.max(1), "moe_expert_ids")?,
        moe_gate_up_id_out: alloc_f32(
            (cfg.top_k_experts * 2 * moe_interm).max(1),
            "moe_gate_up_id_out",
        )?,
        moe_down_id_out: alloc_f32((cfg.top_k_experts * hs).max(1), "moe_down_id_out")?,
        moe_swiglu_id_out: alloc_f32((cfg.top_k_experts * moe_interm).max(1), "moe_swiglu_id_out")?,
        // --- Session merge buffers (S1+S2 collapse) ---
        norm_params_sliding_hd: {
            let sliding_hd = cfg.head_dim;
            let mut buf = alloc_f32(2, "norm_params_sliding_hd")?;
            let p: &mut [f32] = buf
                .as_mut_slice()
                .map_err(|e| anyhow::anyhow!("norm_params_sliding_hd init: {e}"))?;
            p[0] = cfg.rms_norm_eps as f32;
            p[1] = sliding_hd as f32;
            buf
        },
        norm_params_global_hd: {
            let global_hd = cfg.global_head_dim;
            let mut buf = alloc_f32(2, "norm_params_global_hd")?;
            let p: &mut [f32] = buf
                .as_mut_slice()
                .map_err(|e| anyhow::anyhow!("norm_params_global_hd init: {e}"))?;
            p[0] = cfg.rms_norm_eps as f32;
            p[1] = global_hd as f32;
            buf
        },
        rope_freq_factors_gpu: alloc_f32(1, "rope_freq_factors_gpu_placeholder")?,
        attn_v: alloc_f32(max_kv_heads * max_hd, "attn_v")?,
        attn_q_normed: alloc_f32(num_heads * max_hd, "attn_q_normed")?,
        attn_k_normed: alloc_f32(max_kv_heads * max_hd, "attn_k_normed")?,
        // G4-CFA-5b: clamp to ≥1 element for dense GGUFs (top_k_experts==0).
        moe_routing_weights_gpu: alloc_f32(cfg.top_k_experts.max(1), "moe_routing_weights_gpu")?,
    })
}
