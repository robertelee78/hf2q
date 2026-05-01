//! ADR-017 §B-dense.2 follow-up — `KvSpillDescriptor`: cached shape config
//! for the KV-persist hook bridge between `Engine` and the
//! `Gemma4DenseSpillFactory`.
//!
//! ## Why this lives in `serve::api`
//!
//! The `Engine` worker thread owns the `MlxModelWeights` (with its
//! `dense_kvs`, per-layer `head_dim` / `num_kv_heads` / `layer_type`,
//! and the `sliding_window` model parameter). The `KvCacheSpill` hook
//! cannot see those internals across the worker-thread boundary —
//! `Engine::spawn` moves the `LoadedModel` into the worker before any
//! external code can read it.
//!
//! `KvSpillDescriptor` snapshots the immutable KV-cache shape at spawn
//! time (synchronous, before the move into the worker thread) so that
//! the hook factory can read it without round-tripping through the
//! request channel. This mirrors the pattern already used for
//! `hidden_size`, `vocab_size`, and `eos_token_ids` on `EngineInner`
//! (see `engine.rs` lines 506-510, populated at `Engine::spawn` lines
//! 1200-1202).
//!
//! ## Scope
//!
//! This descriptor describes ONLY Gemma 4 dense F32/F16 K/V layouts.
//! Qwen3.5/3.6 hybrid (DeltaNet + full-attn) state is a separate
//! shape and lives in a future B-hybrid descriptor.
//!
//! ## Read-only on `MlxModelWeights`
//!
//! `from_gemma_loaded_model` reads `weights.layers[i]` shape fields and
//! the model-level `sliding_window`. It does NOT mutate the weights and
//! does NOT touch `forward_mlx.rs` — the only references it keeps are
//! the read-only shape fields (`head_dim: usize`, `num_kv_heads: usize`,
//! `layer_type: LayerType`) plus the model-level `sliding_window`.

use crate::serve::config::LayerType;

/// K/V element dtype tag — wraps `mlx_native::DType` to keep this
/// module's surface independent of the mlx-native crate's full DType
/// enum (we only ever store F32 or F16 in the dense KV path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDType {
    /// 4-byte float per element (default).
    F32,
    /// 2-byte float per element (`HF2Q_F16_KV=1`).
    F16,
}

impl KvDType {
    /// Number of bytes per element.
    pub fn elem_bytes(self) -> usize {
        match self {
            KvDType::F32 => 4,
            KvDType::F16 => 2,
        }
    }

    /// Convert to the mlx-native DType. The two valid dense KV dtypes
    /// round-trip cleanly; other DTypes would not.
    pub fn to_mlx_dtype(self) -> mlx_native::DType {
        match self {
            KvDType::F32 => mlx_native::DType::F32,
            KvDType::F16 => mlx_native::DType::F16,
        }
    }
}

/// Cached KV-cache shape config for one Gemma 4 model. Captured by
/// `Engine::spawn` from the `MlxModelWeights` before the move into the
/// worker thread; read-only thereafter.
///
/// The fields here are sufficient to drive
/// `crate::serve::kv_persist::families::gemma4_dense::Gemma4DenseConfig`
/// construction without round-tripping through the worker channel.
#[derive(Debug, Clone)]
pub struct KvSpillDescriptor {
    /// Sliding-window size (drives ring-buffer capacity for
    /// `LayerType::Sliding` layers). Equals `MlxModelWeights.sliding_window`.
    pub sliding_window: usize,
    /// Default budget for full-attention layer linear capacity at
    /// restore-before-prefill time. Used by the factory to seed the
    /// `Gemma4DenseConfig.max_decode_tokens` field. Mirrors the
    /// engine's per-request `params.max_tokens` upper bound (today
    /// 512 by default; tracked here as the spill-side static budget).
    pub max_decode_tokens: usize,
    /// Number of attention layers — equals `weights.layers.len()`.
    pub num_layers: usize,
    /// Per-layer attention type. Length == `num_layers`.
    pub layer_types: Vec<LayerType>,
    /// Per-layer KV-head count. Length == `num_layers`. For Gemma 4
    /// sliding layers this is `num_key_value_heads`; for full-attn
    /// layers it's `num_global_key_value_heads`.
    pub nkv_heads: Vec<usize>,
    /// Per-layer head dim. Length == `num_layers`. For Gemma 4 sliding
    /// layers this is `head_dim` (256); for full-attn layers it's
    /// `global_head_dim` (512).
    pub head_dim: Vec<usize>,
    /// Shared dtype across all K/V layers (Gemma 4 uses one dtype for
    /// the whole cache: F32 by default, F16 under `HF2Q_F16_KV=1`).
    pub kv_dtype: KvDType,
}

impl KvSpillDescriptor {
    /// Construct a descriptor from a Gemma 4 `MlxModelWeights` snapshot.
    ///
    /// `max_decode_tokens` is the default per-request decode budget the
    /// engine plans for; the factory uses this to size the linear-
    /// capacity for full-attention layers when a restore fires before
    /// the first prefill (see `gemma4_dense.rs:367-372`).
    ///
    /// `kv_dtype` reflects the operator-time `HF2Q_F16_KV` env at
    /// spawn time. The descriptor is captured ONCE at spawn (not per-
    /// request) so toggling the env mid-process doesn't re-shape the
    /// descriptor. This matches `forward_prefill.rs:259-260` which
    /// reads the env once into `INVESTIGATION_ENV.f16_kv` at module
    /// init via `LazyLock`.
    pub fn from_gemma_loaded_model(
        weights: &crate::serve::forward_mlx::MlxModelWeights,
        max_decode_tokens: usize,
        kv_dtype: KvDType,
    ) -> Self {
        let num_layers = weights.layers.len();
        let mut layer_types = Vec::with_capacity(num_layers);
        let mut nkv_heads = Vec::with_capacity(num_layers);
        let mut head_dim = Vec::with_capacity(num_layers);
        for layer in &weights.layers {
            layer_types.push(layer.layer_type);
            nkv_heads.push(layer.num_kv_heads);
            head_dim.push(layer.head_dim);
        }
        Self {
            sliding_window: weights.sliding_window,
            max_decode_tokens,
            num_layers,
            layer_types,
            nkv_heads,
            head_dim,
            kv_dtype,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod kv_spill_descriptor_tests {
    use super::*;

    /// Test 1: KvDType::elem_bytes returns the correct byte count for
    /// F32 (4 bytes) and F16 (2 bytes). Drives the `expected_kv_bytes`
    /// math inside `Gemma4DenseSpill::snapshot_block`.
    #[test]
    fn kv_dtype_elem_bytes_match_byte_widths() {
        assert_eq!(KvDType::F32.elem_bytes(), 4);
        assert_eq!(KvDType::F16.elem_bytes(), 2);
    }

    /// Test 2: KvDType round-trips through mlx_native::DType.
    /// Falsifier: `to_mlx_dtype` collapses both arms or returns BF16.
    #[test]
    fn kv_dtype_round_trips_through_mlx_native_dtype() {
        assert_eq!(KvDType::F32.to_mlx_dtype(), mlx_native::DType::F32);
        assert_eq!(KvDType::F16.to_mlx_dtype(), mlx_native::DType::F16);
        // Distinct outputs for distinct inputs.
        assert_ne!(KvDType::F32.to_mlx_dtype(), KvDType::F16.to_mlx_dtype());
    }

    /// Test 3: A hand-crafted descriptor preserves its field values
    /// across a clone. The factory clones the descriptor out of the
    /// engine; this verifies cloning is byte-identical.
    #[test]
    fn descriptor_clone_preserves_all_fields() {
        let d = KvSpillDescriptor {
            sliding_window: 1024,
            max_decode_tokens: 512,
            num_layers: 4,
            layer_types: vec![
                LayerType::Sliding,
                LayerType::Sliding,
                LayerType::Full,
                LayerType::Sliding,
            ],
            nkv_heads: vec![2, 2, 1, 2],
            head_dim: vec![256, 256, 512, 256],
            kv_dtype: KvDType::F32,
        };
        let c = d.clone();
        assert_eq!(c.sliding_window, 1024);
        assert_eq!(c.max_decode_tokens, 512);
        assert_eq!(c.num_layers, 4);
        assert_eq!(c.layer_types.len(), 4);
        assert_eq!(c.layer_types[2], LayerType::Full);
        assert_eq!(c.nkv_heads, vec![2, 2, 1, 2]);
        assert_eq!(c.head_dim, vec![256, 256, 512, 256]);
        assert_eq!(c.kv_dtype, KvDType::F32);
    }

    /// Test 4: A descriptor with mismatched per-layer vector lengths
    /// is observable — the consumer (`Gemma4DenseSpill::new`) rejects
    /// it. The descriptor itself is plain-old-data and does not enforce
    /// length invariants; that's the consumer's job.
    ///
    /// This test documents the invariant: `from_gemma_loaded_model`
    /// always produces vectors of equal length because they're built
    /// from a single iteration over `weights.layers`.
    #[test]
    fn descriptor_vector_lengths_must_be_equal_to_num_layers() {
        // Hand-build one to assert the invariant the constructor
        // promises.
        let d = KvSpillDescriptor {
            sliding_window: 16,
            max_decode_tokens: 32,
            num_layers: 3,
            layer_types: vec![LayerType::Sliding, LayerType::Full, LayerType::Sliding],
            nkv_heads: vec![2, 1, 2],
            head_dim: vec![8, 16, 8],
            kv_dtype: KvDType::F16,
        };
        assert_eq!(d.layer_types.len(), d.num_layers);
        assert_eq!(d.nkv_heads.len(), d.num_layers);
        assert_eq!(d.head_dim.len(), d.num_layers);
    }

    /// Test 5: Descriptor's F16 dtype halves the per-element byte cost.
    /// Falsifier: F16 returns 4 bytes (would over-count payload by 2x
    /// and the factory would over-allocate buffers).
    #[test]
    fn f16_descriptor_halves_per_element_bytes() {
        let d_f32 = KvSpillDescriptor {
            sliding_window: 16,
            max_decode_tokens: 32,
            num_layers: 1,
            layer_types: vec![LayerType::Sliding],
            nkv_heads: vec![2],
            head_dim: vec![8],
            kv_dtype: KvDType::F32,
        };
        let d_f16 = KvSpillDescriptor {
            kv_dtype: KvDType::F16,
            ..d_f32.clone()
        };
        // Per-token byte cost for one head: 8 elements * elem_bytes.
        let cost_f32 = d_f32.head_dim[0] * d_f32.kv_dtype.elem_bytes();
        let cost_f16 = d_f16.head_dim[0] * d_f16.kv_dtype.elem_bytes();
        assert_eq!(cost_f32, 32, "8 * 4 bytes per head/token (F32)");
        assert_eq!(cost_f16, 16, "8 * 2 bytes per head/token (F16)");
        assert_eq!(
            cost_f32, cost_f16 * 2,
            "F32 is exactly twice the byte cost of F16 per element"
        );
    }
}
