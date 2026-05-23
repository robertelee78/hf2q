//! Gemma 4 KV-cache buffer types and decode-regime control.
//!
//! Extracted from `src/serve/forward_mlx.rs` by ADR-038 Step 2.
//! Mirrors the `qwen35/kv_cache.rs` pattern.

use anyhow::{anyhow, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

/// Per-layer KV cache buffers for the mlx-native path (TurboQuant compressed).
///
/// ADR-007 Phase 1.2: KV cache is stored as 4-bit nibble-packed indices
/// with per-position F32 norms, replacing F16 dense buffers.  This halves
/// KV memory bandwidth during SDPA and enables 262K context.
pub struct MlxKvCache {
    /// K packed indices `[num_kv_heads, capacity, head_dim/2]` U8 (nibble-packed).
    pub k_packed: MlxBuffer,
    /// K per-position norms `[num_kv_heads, capacity]` F32.
    pub k_norms: MlxBuffer,
    /// V packed indices `[num_kv_heads, capacity, head_dim/2]` U8 (nibble-packed).
    pub v_packed: MlxBuffer,
    /// V per-position norms `[num_kv_heads, capacity]` F32.
    pub v_norms: MlxBuffer,
    /// Cache capacity (max_seq_len for global, sliding_window for sliding).
    pub capacity: usize,
    /// Whether this is a sliding window cache.
    pub is_sliding: bool,
    /// Current write position (next position to write).
    pub write_pos: usize,
    /// Number of valid positions in the cache.
    pub seq_len: usize,
}

impl MlxKvCache {
    /// ADR-028 iter-229 / ADR-028 §iter-227 work item D: ds4-style counter
    /// rollback for speculative decode infrastructure.
    ///
    /// Logically discards the most-recent `n_back` positions from the cache.
    /// Following ds4's pattern (`DS4_MTP_KEEP_ACCEPTED` macro, ds4.c:16246),
    /// no actual cache bytes are cleared — `seq_len` is decremented to make
    /// the trailing positions invisible to subsequent SDPA reads.  Future
    /// writes (via `write_pos`) will overwrite those slots naturally.
    ///
    /// Returns the new `seq_len` for caller assertion.
    ///
    /// # Semantics
    /// - **Linear cache** (`is_sliding=false`): `write_pos == seq_len`, both
    ///   decrement by `n_back`.  Subsequent writes resume at the new
    ///   `write_pos`.
    /// - **Sliding cache** (`is_sliding=true`): more complex — `write_pos`
    ///   wraps modulo `capacity`.  Implemented for the linear case here;
    ///   sliding-aware rollback requires position-tracking metadata and is
    ///   deferred until iter-227 work item C (SD state machine).
    ///
    /// # Errors
    /// Returns `Err` if `n_back > seq_len` or sliding cache (not yet supported).
    pub fn trim(&mut self, n_back: usize) -> Result<usize, &'static str> {
        if self.is_sliding {
            // Sliding cache rollback requires logical-position tracking
            // (the slot index ≠ logical position when wrapped).  For ds4-style
            // SD on gemma4, sliding layers may need a counter parallel to
            // `write_pos` that tracks logical end-position separately.
            // Deferred to iter-227 work item C.
            return Err("trim() not yet supported on sliding cache");
        }
        if n_back > self.seq_len {
            return Err("trim n_back exceeds seq_len");
        }
        // Linear cache: write_pos == seq_len. Both decrement.
        self.seq_len -= n_back;
        self.write_pos = self.seq_len;
        Ok(self.seq_len)
    }

    /// Returns the count of valid (visible) positions.  Equivalent to
    /// `seq_len` post-iter-229 but exposed as named API for the SD
    /// state machine (matches ds4's `s->graph.mtp_n_raw` semantic).
    #[inline]
    pub fn visible_len(&self) -> usize {
        self.seq_len
    }
}

/// Per-layer byte-packed higher-bit (5/6-bit) KV buffers (iter-21 Track B).
pub struct HbKvBuffers {
    /// Byte-packed K indices `[nkv_heads, capacity, head_dim]` U8.
    pub k_packed: MlxBuffer,
    /// K per-position norms (same layout as 4-bit: D=256 → 1/pos, D=512 → 2/pos).
    pub k_norms: MlxBuffer,
    /// Byte-packed V indices `[nkv_heads, capacity, head_dim]` U8.
    pub v_packed: MlxBuffer,
    /// V per-position norms.
    pub v_norms: MlxBuffer,
    /// Cache capacity in positions.
    pub capacity: usize,
    /// True if ring-buffer (sliding) semantics.
    pub is_sliding: bool,
    /// Norms per position (1 for D=256, 2 for D=512).
    #[allow(dead_code)]
    pub norms_per_pos: usize,
}

/// Per-layer dense F32/F16 KV buffers for dense attention path (ADR-009).
pub struct DenseKvBuffers {
    pub k: MlxBuffer,
    pub v: MlxBuffer,
    /// Capacity (positions) in this layer's cache. Sliding layers use
    /// ring-buffer mode: capacity == sliding_window, writes wrap.
    /// Global layers use linear mode: capacity >= seq_len + max_tokens.
    pub capacity: usize,
    /// True if this is a sliding layer (ring-buffer semantics).
    pub is_sliding: bool,
    /// ADR-017 Phase E.a iter-3.5a (Codex round-1 LOW #4) — KV element
    /// dtype invariant. Today every engine in a process uses one
    /// `HF2Q_F16_KV` setting (parsed at LazyLock init), so all
    /// `DenseKvBuffers` in the live LcpRegistry necessarily share
    /// dtype. Recording it explicitly:
    ///
    ///   * makes the invariant a load-bearing struct field (not a
    ///     process-implicit assumption that could regress under any
    ///     future hot-reload / multi-engine path),
    ///   * lets the engine's `take_prefix` capacity-check site assert
    ///     `cached.dtype == model.kv_dtype` per layer alongside the
    ///     existing capacity + is_sliding checks, closing a class of
    ///     silent-corruption bugs at the type level.
    ///
    /// Populated at every construction site: `forward_prefill.rs` per-
    /// layer alloc, `forward_prefill_batched.rs` per-layer alloc, and
    /// `engine.rs::kv_restore_gemma` per-layer alloc.
    pub dtype: DType,
}

impl crate::serve::kv_persist::lcp_registry::ByteSized for DenseKvBuffers {
    /// Exact byte count of the K + V buffers for this layer.
    /// Uses `MlxBuffer::byte_len()` — the same API used at forward_mlx.rs:1167+
    /// and 5158+. No estimation.
    fn byte_len(&self) -> u64 {
        (self.k.byte_len() + self.v.byte_len()) as u64
    }
}

/// ADR-028 Phase 10 (iter-347): hybrid K storage — F16 K alongside TQ-HB-packed V.
///
/// Motivation (ADR-028 §iter-346 audit):
///   * Pure TQ-HB (today's default): 504 MB raw F32 → 128 MB packed (3.94× saving).
///   * Pure F16 K + F16 V (peer): 504 MB → 252 MB (2× saving).
///   * Hybrid (F16 K + TQ-HB V): 504 MB → 158 MB (3.19× saving — 81% of TQ-HB).
///
/// The structural decode-side gap vs llama.cpp peer (1.81× per-dispatch wall on
/// our TQ-HB SDPA, formally measured iter-326..342) is owned by the K-side
/// scalar dequant loop inside `flash_attn_vec_tq_hb`: peer's K is F16 and consumed
/// by `simdgroup_matrix` matmul, ours is byte-packed and consumed by per-thread
/// scalar lookup against the codebook. Storing K as F16 (this struct) and using a
/// new `flash_attn_vec_hybrid_dk256` SDPA kernel (Phase 10d) brings the K-side
/// throughput up to peer-equivalent simdgroup math while the V-side stays in
/// 1-byte-per-element TQ-HB packing.
///
/// Field layout mirrors the union of `DenseKvBuffers` (K) and `HbKvBuffers` (V) so
/// existing snapshot / restore / KV-persist code that walks the K and V buffers
/// can pattern-match by field name without learning a new shape.
///
/// Allocation gate: env `HF2Q_HYBRID_KV` (parsed in `investigation_env.rs`).
/// Default OFF until parity + bench gates pass (Phase 10f/g). When ON, the
/// per-layer alloc site at `forward_decode` (currently building `HbKvBuffers`)
/// instead builds `HybridKvBuffers`, the K-encode dispatch is skipped, and the
/// SDPA dispatcher routes to the hybrid kernel.
pub struct HybridKvBuffers {
    /// Dense F16 K cache `[nkv_heads, capacity, head_dim]`. dtype is always
    /// `mlx_native::DType::F16` for this struct (the whole point — F16 K → peer
    /// SDPA-equivalent simdgroup matmul). No F32 variant: F32 K would erase the
    /// memory advantage of the hybrid design (158 MB → 284 MB at gemma4 32K
    /// context, defeating the purpose of mixing in TQ-HB on V at all).
    pub k: MlxBuffer,
    /// Byte-packed V indices `[nkv_heads, capacity, head_dim]` U8 — same layout
    /// and codec as `HbKvBuffers::v_packed`. The V-encode dispatch
    /// (`hadamard_quantize_kv_*`) writes here unchanged.
    pub v_packed: MlxBuffer,
    /// V per-position norms — same layout as `HbKvBuffers::v_norms`.
    /// (D=256 → 1/pos, D=512 → 2/pos.)
    pub v_norms: MlxBuffer,
    /// Cache capacity in positions (matches `DenseKvBuffers::capacity` and
    /// `HbKvBuffers::capacity` — populated identically at alloc site).
    pub capacity: usize,
    /// True if ring-buffer (sliding) semantics — same as the sibling structs.
    pub is_sliding: bool,
    /// Norms per position (1 for D=256, 2 for D=512). Mirrors
    /// `HbKvBuffers::norms_per_pos`.
    #[allow(dead_code)]
    pub norms_per_pos: usize,
    /// ADR-030 iter-96: BF16 K cache for DFlash spec-decode xlen verify
    /// path. Same layout `[nkv_heads, capacity, head_dim]` as `k` but BF16
    /// dtype. Populated from `pf_k_perm` BF16 (head_norm_rope's bf16 output)
    /// via `dispatch_kv_cache_copy_seq_bf16_to_bf16_head_major`. Used by
    /// xlen branch's SDPA to read BF16 K bit-identical to what Option C
    /// reads from pf_k_perm — avoids the F16-roundtrip precision drift
    /// root-caused at iter-92/93.  Lazy-alloc'd on first xlen-mode call.
    pub bf16_xlen_k: Option<MlxBuffer>,
    /// ADR-030 iter-96: BF16 V cache, same semantics as `bf16_xlen_k`.
    pub bf16_xlen_v: Option<MlxBuffer>,
}

impl crate::serve::kv_persist::lcp_registry::ByteSized for HybridKvBuffers {
    /// Exact byte count: F16 K + U8 V + F32 V-norms. Used by the LcpRegistry
    /// byte budget the same way `DenseKvBuffers::byte_len` is.
    fn byte_len(&self) -> u64 {
        (self.k.byte_len() + self.v_packed.byte_len() + self.v_norms.byte_len()) as u64
    }
}

/// ADR-028 Phase 10c (iter-348): per-layer F16-K + TQ-HB-V buffer allocator.
///
/// Single-source-of-truth for the hybrid allocation shape — called from
/// 3 sites (decode lazy-alloc, per-token prefill alloc, batched-prefill
/// alloc) so all three stay in lockstep.
///
/// F16 K: 2 bytes/elem, shape `[nkv, cap, hd]`.
/// V layout identical to legacy `HbKvBuffers` V-side (1 byte/elem packed +
/// per-pos F32 norms with `norms_per_pos = max(1, hd / 256)`).
pub(crate) fn alloc_hybrid_kv_for_layer(
    dev: &MlxDevice,
    layer_idx: usize,
    nkv: usize,
    hd: usize,
    cap: usize,
    is_ring: bool,
) -> anyhow::Result<HybridKvBuffers> {
    let norms_per_pos = (hd / 256).max(1);
    let norms_n = nkv * cap * norms_per_pos;
    // F16 K: byte_count = elements * 2.
    let k = dev.alloc_buffer(nkv * cap * hd * 2, DType::F16,
        vec![nkv, cap, hd])
        .map_err(|e| anyhow!("hybrid F16 K L{layer_idx}: {e}"))?;
    // ADR-029 iter-20 H27: when HF2Q_FULL_F16_KV is set, V is F16 (2 bytes/elem)
    // and v_norms is a small dummy buffer (kernel ignores it when v_is_f16=1).
    // Otherwise: legacy TQ-HB packed V (1 byte/elem) + per-position F32 norms.
    let full_f16_v = std::env::var("HF2Q_FULL_F16_KV")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "on"))
        .unwrap_or(false);
    let (v_packed, v_norms) = if full_f16_v {
        let v_f16 = dev.alloc_buffer(nkv * cap * hd * 2, DType::F16,
            vec![nkv, cap, hd])
            .map_err(|e| anyhow!("hybrid F16 V L{layer_idx}: {e}"))?;
        // Dummy norms buffer (unused but kept for ABI compat with hybrid SDPA
        // signature; kernel's v_is_f16 FC=1 skips the read).
        let v_norms_dummy = dev.alloc_buffer(4, DType::F32, vec![1])
            .map_err(|e| anyhow!("hybrid V norms (dummy) L{layer_idx}: {e}"))?;
        (v_f16, v_norms_dummy)
    } else {
        let v_p = dev.alloc_buffer(nkv * cap * hd, DType::U8,
            vec![nkv, cap, hd])
            .map_err(|e| anyhow!("hybrid V packed L{layer_idx}: {e}"))?;
        let v_n = dev.alloc_buffer(norms_n * 4, DType::F32,
            if norms_per_pos == 1 { vec![nkv, cap] } else { vec![nkv, cap, norms_per_pos] })
            .map_err(|e| anyhow!("hybrid V norms L{layer_idx}: {e}"))?;
        (v_p, v_n)
    };
    // ADR-030 iter-96: lazy-alloc the BF16 xlen cache only when env opted-in.
    // Saves ~55MB at gemma-4 when xlen mode disabled.
    let xlen_mode = std::env::var("HF2Q_DFLASH_XLEN_SDPA").as_deref() == Ok("1");
    let (bf16_xlen_k, bf16_xlen_v) = if xlen_mode {
        let bk = dev.alloc_buffer(nkv * cap * hd * 2, DType::BF16,
            vec![nkv, cap, hd])
            .map_err(|e| anyhow!("bf16 xlen K L{layer_idx}: {e}"))?;
        let bv = dev.alloc_buffer(nkv * cap * hd * 2, DType::BF16,
            vec![nkv, cap, hd])
            .map_err(|e| anyhow!("bf16 xlen V L{layer_idx}: {e}"))?;
        (Some(bk), Some(bv))
    } else {
        (None, None)
    };
    Ok(HybridKvBuffers { k, v_packed, v_norms, capacity: cap, is_sliding: is_ring, norms_per_pos, bf16_xlen_k, bf16_xlen_v })
}

/// Per-call decode regime override for ADR-007 Gate H two-regime-one-process
/// runs (W12 iter-108a blocker #3).
///
/// Set on `MlxModelWeights` before each prefill+decode trajectory via
/// [`crate::serve::forward_mlx::MlxModelWeights::set_decode_regime`].
/// Consulted at the SDPA-mode gate inside `forward_decode` (the
/// `use_dense_sdpa` check); the four codebook-bits gates stay env-var-driven
/// because the codebook width is consistent across both regimes.
///
/// - `Default` (the zero value): preserve today's env-var behavior.
/// - `ForceTq`: ignore env, behave as if `HF2Q_USE_DENSE` were unset and
///   `HF2Q_LAYER_POLICY=tq_all` — TQ-active SDPA on every layer.
/// - `ForceDense`: ignore env, behave as if `HF2Q_USE_DENSE=1` were set —
///   dense-active SDPA on every layer.
///
/// Gate H uses one `MlxModelWeights` instance and runs (a)
/// `set_decode_regime(ForceDense)` → fresh prefill + decode loop → capture
/// tokens + per-token NLL + SDPA-output dumps; then (b)
/// `set_decode_regime(ForceTq)` → fresh prefill + decode loop with the same
/// prompt → cosine the SDPA outputs and PPL the NLLs. The per-instance step
/// counter is reset by `set_decode_regime` so each regime's stderr lines
/// start at `step=0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeRegime {
    /// Honor `HF2Q_USE_DENSE` / `HF2Q_LAYER_POLICY` env vars (today's path).
    #[default]
    Default,
    /// Force TQ-active SDPA regardless of env.
    /// (Wired by iter-108b's release-check.sh Gate 5 harness; no in-tree
    /// caller as of iter-108a.)
    #[allow(dead_code)]
    ForceTq,
    /// Force dense-active SDPA regardless of env.
    /// (Wired by iter-108b's release-check.sh Gate 5 harness.)
    #[allow(dead_code)]
    ForceDense,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip_dev() -> Option<MlxDevice> {
        match MlxDevice::new() {
            Ok(d) => Some(d),
            Err(_) => {
                eprintln!("skip: no MlxDevice");
                None
            }
        }
    }

    #[test]
    fn mlx_kv_cache_trim_linear_decrements_seq_len() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let buf = || dev.alloc_buffer(4, DType::F32, vec![1]).unwrap();
        let mut cache = MlxKvCache {
            k_packed: buf(), k_norms: buf(), v_packed: buf(), v_norms: buf(),
            capacity: 16, is_sliding: false, write_pos: 8, seq_len: 8,
        };
        let new_len = cache.trim(3).unwrap();
        assert_eq!(new_len, 5);
        assert_eq!(cache.seq_len, 5);
        assert_eq!(cache.write_pos, 5);
    }

    #[test]
    fn mlx_kv_cache_trim_sliding_errors() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let buf = || dev.alloc_buffer(4, DType::F32, vec![1]).unwrap();
        let mut cache = MlxKvCache {
            k_packed: buf(), k_norms: buf(), v_packed: buf(), v_norms: buf(),
            capacity: 16, is_sliding: true, write_pos: 4, seq_len: 4,
        };
        assert!(cache.trim(1).is_err());
    }

    #[test]
    fn mlx_kv_cache_trim_overflow_errors() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let buf = || dev.alloc_buffer(4, DType::F32, vec![1]).unwrap();
        let mut cache = MlxKvCache {
            k_packed: buf(), k_norms: buf(), v_packed: buf(), v_norms: buf(),
            capacity: 16, is_sliding: false, write_pos: 3, seq_len: 3,
        };
        assert!(cache.trim(10).is_err());
    }

    #[test]
    fn mlx_kv_cache_visible_len_eq_seq_len() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let buf = || dev.alloc_buffer(4, DType::F32, vec![1]).unwrap();
        let cache = MlxKvCache {
            k_packed: buf(), k_norms: buf(), v_packed: buf(), v_norms: buf(),
            capacity: 32, is_sliding: false, write_pos: 7, seq_len: 7,
        };
        assert_eq!(cache.visible_len(), cache.seq_len);
    }

    #[test]
    fn decode_regime_default_via_default_trait() {
        let r: DecodeRegime = Default::default();
        assert_eq!(r, DecodeRegime::Default);
    }

    #[test]
    fn decode_regime_variants_distinct() {
        assert_ne!(DecodeRegime::Default, DecodeRegime::ForceTq);
        assert_ne!(DecodeRegime::Default, DecodeRegime::ForceDense);
        assert_ne!(DecodeRegime::ForceTq, DecodeRegime::ForceDense);
    }

    #[test]
    fn hybrid_kv_buffers_byte_len_sums_fields() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let nkv = 2; let cap = 4; let hd = 256;
        let k = dev.alloc_buffer(nkv * cap * hd * 2, DType::F16, vec![nkv, cap, hd]).unwrap();
        let v_packed = dev.alloc_buffer(nkv * cap * hd, DType::U8, vec![nkv, cap, hd]).unwrap();
        let v_norms = dev.alloc_buffer(nkv * cap * 4, DType::F32, vec![nkv, cap]).unwrap();
        let k_bytes = k.byte_len();
        let vp_bytes = v_packed.byte_len();
        let vn_bytes = v_norms.byte_len();
        let buf = HybridKvBuffers {
            k, v_packed, v_norms,
            capacity: cap, is_sliding: false, norms_per_pos: 1,
            bf16_xlen_k: None, bf16_xlen_v: None,
        };
        use crate::serve::kv_persist::lcp_registry::ByteSized;
        assert_eq!(buf.byte_len(), (k_bytes + vp_bytes + vn_bytes) as u64);
    }

    #[test]
    fn dense_kv_buffers_byte_len_sums_k_plus_v() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        let nkv = 2; let cap = 8; let hd = 256;
        let k = dev.alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd]).unwrap();
        let v = dev.alloc_buffer(nkv * cap * hd * 4, DType::F32, vec![nkv, cap, hd]).unwrap();
        let kb = k.byte_len(); let vb = v.byte_len();
        let buf = DenseKvBuffers { k, v, capacity: cap, is_sliding: false, dtype: DType::F32 };
        use crate::serve::kv_persist::lcp_registry::ByteSized;
        assert_eq!(buf.byte_len(), (kb + vb) as u64);
    }

    #[test]
    fn alloc_hybrid_kv_for_layer_no_xlen_no_full_f16() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        // Ensure env gates are off for this test.
        std::env::remove_var("HF2Q_FULL_F16_KV");
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
        let buf = alloc_hybrid_kv_for_layer(&dev, 0, 2, 256, 8, false).unwrap();
        assert!(buf.bf16_xlen_k.is_none());
        assert!(buf.bf16_xlen_v.is_none());
        assert_eq!(buf.capacity, 8);
        assert!(!buf.is_sliding);
        assert_eq!(buf.norms_per_pos, 1);
    }

    #[test]
    fn alloc_hybrid_kv_for_layer_full_f16_v_allocates_f16() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        std::env::set_var("HF2Q_FULL_F16_KV", "1");
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
        let buf = alloc_hybrid_kv_for_layer(&dev, 1, 2, 256, 4, true).unwrap();
        // v_norms is the dummy 4-byte buffer when full_f16_v=true.
        assert_eq!(buf.v_norms.byte_len(), 4);
        assert!(buf.is_sliding);
        std::env::remove_var("HF2Q_FULL_F16_KV");
    }

    #[test]
    fn alloc_hybrid_kv_for_layer_xlen_allocates_bf16_buffers() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        std::env::remove_var("HF2Q_FULL_F16_KV");
        std::env::set_var("HF2Q_DFLASH_XLEN_SDPA", "1");
        let buf = alloc_hybrid_kv_for_layer(&dev, 2, 2, 256, 4, false).unwrap();
        assert!(buf.bf16_xlen_k.is_some());
        assert!(buf.bf16_xlen_v.is_some());
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
    }

    #[test]
    fn alloc_hybrid_kv_for_layer_norms_per_pos_d256_d512() {
        let dev = match skip_dev() { Some(d) => d, None => return };
        std::env::remove_var("HF2Q_FULL_F16_KV");
        std::env::remove_var("HF2Q_DFLASH_XLEN_SDPA");
        // D=256: norms_per_pos == 1.
        let buf256 = alloc_hybrid_kv_for_layer(&dev, 0, 2, 256, 4, false).unwrap();
        assert_eq!(buf256.norms_per_pos, 1);
        // D=512: norms_per_pos == 2.
        let buf512 = alloc_hybrid_kv_for_layer(&dev, 0, 2, 512, 4, false).unwrap();
        assert_eq!(buf512.norms_per_pos, 2);
    }
}
