//! ADR-017 TQ-packed KV snapshot/restore for the Gemma 4 forward pass.
//!
//! Moved from `src/serve/forward_mlx.rs` by ADR-038 Step 3.

use anyhow::Result;

use super::model::MlxModelWeights;

/// ADR-017 B-tq.3 helper: convert a `&[f32]` slice into a `Vec<u8>` of
/// little-endian bytes via per-element `to_le_bytes`. Used by the
/// `tq_v2_snapshot_block` capture path.
#[allow(dead_code)]
fn f32_slice_to_le_bytes(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 4);
    for &x in src {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

impl MlxModelWeights {
    // =========================================================================
    // ADR-017 Phase B-tq.3 — engine-side TQ-packed snapshot/restore hooks
    // =========================================================================
    //
    // These bridge the runtime `MlxKvCache` byte buffers to the
    // `tq_packed_v2` envelope codec at
    // `serve::kv_persist::families::tq_packed`.  They run AFTER
    // `dispatch_hadamard_quantize_kv` has committed (caller is responsible
    // for issuing `s.finish()` first — there's no implicit barrier here)
    // so the live K/V packed buffers carry the post-quantize Lloyd-Max
    // indices + per-token-per-head FWHT magnitudes.
    //
    // Snapshot path:
    //   `tq_v2_snapshot_block(layer, range, bits, flags, scale)`
    //     → reads `kv_caches[layer].{k_packed, k_norms, v_packed, v_norms}`
    //     → packs two `tq_packed_v2` envelopes (one for K, one for V)
    //     → returns `(k_payload, v_payload)` ready for
    //       `TqPackedSpill::insert_block(layer, range.start, ..)` × 2
    //       (the spiller stores K + V under different (layer, range)
    //       keys; convention: K at `range.start`, V at `range.start +
    //       0x80_00_00_00` — see callers).
    //
    // Restore path is the inverse: takes the two payloads + writes back
    // into the live MlxKvCache buffers at the correct (head, position,
    // hd_packed) offsets.
    //
    // BOTH operations require a prior `commit_and_wait` on the encoder
    // session if the caller has issued any GPU work targeting these
    // buffers — this method does NOT issue its own barrier (callers
    // already control session boundaries via `exec.begin/finish`).

    /// Capture (K, V) `tq_packed_v2` envelope payloads from a token range
    /// of `kv_caches[layer_rank]`.  See module-level B-tq.3 doc for the
    /// barrier preconditions.
    ///
    /// `bits_per_coord` MUST match the active codec at quantize time —
    /// production default is 4 (nibble-packed) per ADR-007 §3 default
    /// configuration.  `flags` should set `HADAMARD_ROTATED` whenever
    /// the runtime applied FWHT before quantizing (the production path
    /// always does).  `scale` is the per-block multiplicative scale
    /// (typically 1.0 since the magnitude lives in the per-token norms).
    ///
    /// `#[allow(dead_code)]` because activation lives behind the
    /// `TqPackedSpillFactory` registration in `cmd_serve` (operator-
    /// controlled, deferred per ADR-007 reopen Path C clearance).  The
    /// method's correctness is exercised via the byte-level helpers'
    /// unit tests at `serve::kv_persist::families::tq_packed::tests::
    /// tq_v2_capture_restore_byte_identity`.
    #[allow(dead_code)]
    pub fn tq_v2_snapshot_block(
        &self,
        layer_rank: usize,
        range: std::ops::Range<u32>,
        bits_per_coord: crate::serve::kv_persist::families::tq_packed::TqBitsPerCoord,
        flags: u32,
        scale: f64,
    ) -> Result<(Vec<u8>, Vec<u8>), crate::serve::multi_model::SpillErrorKind> {
        use crate::serve::kv_persist::families::tq_packed;
        use crate::serve::multi_model::SpillErrorKind;

        // **B-tq.7** — at bits >= 5 the runtime stores K/V in
        // `leg_hb_encoded[layer_rank]` (1 byte per coord, shape
        // `[nkv, capacity, head_dim]`); at bits == 4 it stores in
        // `kv_caches[layer_rank].k_packed` (nibble-packed, shape
        // `[nkv, capacity, head_dim/2]`).  The active SDPA reads
        // from the matching buffer; snapshot must do the same.
        //
        // Branch up-front so the seq_len gate, shape derivation,
        // and byte reads all use the same buffer.
        let use_hb = bits_per_coord.0 >= 5;

        let (k_packed_bytes, k_norms_f32, v_packed_bytes, v_norms_f32, capacity_runtime, hd_packed_runtime, n_kv_heads_runtime, seq_len_live):
            (&[u8], &[f32], &[u8], &[f32], usize, usize, usize, usize) = if use_hb {
            let hb = self
                .leg_hb_encoded
                .as_ref()
                .ok_or(SpillErrorKind::CodecErr)?;
            let lay = hb.get(layer_rank).ok_or(SpillErrorKind::CodecErr)?;
            let cache = self
                .kv_caches
                .get(layer_rank)
                .ok_or(SpillErrorKind::CodecErr)?;
            // HB shares the kv_caches' seq_len bookkeeping (same
            // forward_decode increments both); read from kv_caches.
            (
                lay.k_packed.as_slice::<u8>().map_err(|_| SpillErrorKind::IoErr)?,
                lay.k_norms.as_slice::<f32>().map_err(|_| SpillErrorKind::IoErr)?,
                lay.v_packed.as_slice::<u8>().map_err(|_| SpillErrorKind::IoErr)?,
                lay.v_norms.as_slice::<f32>().map_err(|_| SpillErrorKind::IoErr)?,
                lay.capacity,
                lay.k_packed.shape().get(2).copied().unwrap_or(0),
                lay.k_packed.shape().first().copied().unwrap_or(0),
                cache.seq_len,
            )
        } else {
            let cache = self
                .kv_caches
                .get(layer_rank)
                .ok_or(SpillErrorKind::CodecErr)?;
            (
                cache.k_packed.as_slice::<u8>().map_err(|_| SpillErrorKind::IoErr)?,
                cache.k_norms.as_slice::<f32>().map_err(|_| SpillErrorKind::IoErr)?,
                cache.v_packed.as_slice::<u8>().map_err(|_| SpillErrorKind::IoErr)?,
                cache.v_norms.as_slice::<f32>().map_err(|_| SpillErrorKind::IoErr)?,
                cache.capacity,
                cache.k_packed.shape().get(2).copied().unwrap_or(0),
                cache.k_packed.shape().first().copied().unwrap_or(0),
                cache.seq_len,
            )
        };

        // Gate snapshot on live state (B-tq.4 iter-5 fix; same logic
        // for both buffer paths).
        if (range.start as usize) >= seq_len_live {
            return Err(SpillErrorKind::CodecErr);
        }

        let n_kv_heads = n_kv_heads_runtime as u32;
        let capacity = capacity_runtime as u32;
        // head_dim derives from the runtime packed-buffer row stride.
        // 4-bit (kv_caches): `hd_packed = head_dim/2`, hd = hd_packed*8/4 = hd_packed*2.
        // 8-bit (leg_hb_encoded): `hd_packed = head_dim`, hd = hd_packed*8/8 = hd_packed.
        // Generalised: `hd = hd_packed * 8 / bits`.
        let head_dim_bits = (hd_packed_runtime as u64) * 8;
        if head_dim_bits % (bits_per_coord.0 as u64) != 0 {
            return Err(SpillErrorKind::CodecErr);
        }
        let head_dim = (head_dim_bits / (bits_per_coord.0 as u64)) as u32;
        // F32 → LE bytes via per-element `to_le_bytes` to avoid an extra
        // dep.  Hot path is amortised — snapshot fires per block, not
        // per token.
        let k_norms_le: Vec<u8> = f32_slice_to_le_bytes(k_norms_f32);
        let v_norms_le: Vec<u8> = f32_slice_to_le_bytes(v_norms_f32);

        let k_payload = tq_packed::capture_tq_v2_payload_from_buffers(
            k_packed_bytes,
            &k_norms_le,
            capacity,
            n_kv_heads,
            head_dim,
            bits_per_coord,
            range.clone(),
            flags,
            scale,
        )?;
        let v_payload = tq_packed::capture_tq_v2_payload_from_buffers(
            v_packed_bytes,
            &v_norms_le,
            capacity,
            n_kv_heads,
            head_dim,
            bits_per_coord,
            range,
            flags,
            scale,
        )?;
        Ok((k_payload, v_payload))
    }

    /// Restore (K, V) `tq_packed_v2` envelope payloads into a token
    /// range of `kv_caches[layer_rank]`.  Inverse of
    /// [`Self::tq_v2_snapshot_block`].  Writes through `as_mut_slice` —
    /// callers MUST hold exclusive access to the live KV cache (the
    /// engine's per-session mutex).
    ///
    /// `#[allow(dead_code)]` for the same reason as
    /// [`Self::tq_v2_snapshot_block`].
    #[allow(dead_code)]
    pub fn tq_v2_restore_block(
        &mut self,
        layer_rank: usize,
        range: std::ops::Range<u32>,
        bits_per_coord: crate::serve::kv_persist::families::tq_packed::TqBitsPerCoord,
        k_payload: &[u8],
        v_payload: &[u8],
    ) -> Result<(), crate::serve::multi_model::SpillErrorKind> {
        use crate::serve::kv_persist::families::tq_packed;
        use crate::serve::multi_model::SpillErrorKind;

        // **B-tq.7** — branch on bits like the snapshot path does.
        // Restore writes back to `leg_hb_encoded[layer_rank]` at
        // bits >= 5; otherwise to `kv_caches[layer_rank]`.
        let use_hb = bits_per_coord.0 >= 5;

        // Borrow the layer's K/V buffers from the appropriate field.
        // Branch separately for K then V to keep borrow lifetimes
        // tight (each `as_mut_slice` borrows the buffer).
        let (capacity, n_kv_heads, head_dim) = if use_hb {
            let hb = self
                .leg_hb_encoded
                .as_ref()
                .ok_or(SpillErrorKind::CodecErr)?;
            let lay = hb.get(layer_rank).ok_or(SpillErrorKind::CodecErr)?;
            let cap = lay.capacity as u32;
            let nkv = lay.k_packed.shape().first().copied().unwrap_or(0) as u32;
            let hd_packed = lay.k_packed.shape().get(2).copied().unwrap_or(0);
            let head_dim_bits = (hd_packed as u64) * 8;
            if head_dim_bits % (bits_per_coord.0 as u64) != 0 {
                return Err(SpillErrorKind::CodecErr);
            }
            let hd = (head_dim_bits / (bits_per_coord.0 as u64)) as u32;
            (cap, nkv, hd)
        } else {
            let cache = self
                .kv_caches
                .get(layer_rank)
                .ok_or(SpillErrorKind::CodecErr)?;
            let cap = cache.capacity as u32;
            let nkv = cache.k_packed.shape().first().copied().unwrap_or(0) as u32;
            let hd_packed = cache.k_packed.shape().get(2).copied().unwrap_or(0);
            let head_dim_bits = (hd_packed as u64) * 8;
            if head_dim_bits % (bits_per_coord.0 as u64) != 0 {
                return Err(SpillErrorKind::CodecErr);
            }
            let hd = (head_dim_bits / (bits_per_coord.0 as u64)) as u32;
            (cap, nkv, hd)
        };

        // Two passes (K, V) × two operations (packed indices, F32 norms),
        // each requiring a separate `&mut` borrow.  Inner closure
        // `with_layer` factors out the source-of-truth selection.

        macro_rules! borrow_k_packed {
            () => {{
                if use_hb {
                    self.leg_hb_encoded
                        .as_mut()
                        .ok_or(SpillErrorKind::CodecErr)?[layer_rank]
                        .k_packed
                        .as_mut_slice::<u8>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                } else {
                    self.kv_caches[layer_rank]
                        .k_packed
                        .as_mut_slice::<u8>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                }
            }};
        }
        macro_rules! borrow_k_norms {
            () => {{
                if use_hb {
                    self.leg_hb_encoded
                        .as_mut()
                        .ok_or(SpillErrorKind::CodecErr)?[layer_rank]
                        .k_norms
                        .as_mut_slice::<f32>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                } else {
                    self.kv_caches[layer_rank]
                        .k_norms
                        .as_mut_slice::<f32>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                }
            }};
        }
        macro_rules! borrow_v_packed {
            () => {{
                if use_hb {
                    self.leg_hb_encoded
                        .as_mut()
                        .ok_or(SpillErrorKind::CodecErr)?[layer_rank]
                        .v_packed
                        .as_mut_slice::<u8>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                } else {
                    self.kv_caches[layer_rank]
                        .v_packed
                        .as_mut_slice::<u8>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                }
            }};
        }
        macro_rules! borrow_v_norms {
            () => {{
                if use_hb {
                    self.leg_hb_encoded
                        .as_mut()
                        .ok_or(SpillErrorKind::CodecErr)?[layer_rank]
                        .v_norms
                        .as_mut_slice::<f32>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                } else {
                    self.kv_caches[layer_rank]
                        .v_norms
                        .as_mut_slice::<f32>()
                        .map_err(|_| SpillErrorKind::IoErr)?
                }
            }};
        }

        // --- K ---
        {
            let k_packed_mut: &mut [u8] = borrow_k_packed!();
            let _ = restore_packed_only(
                k_packed_mut,
                capacity,
                n_kv_heads,
                head_dim,
                bits_per_coord,
                range.clone(),
                k_payload,
            )?;
        }
        {
            let k_norms_f32: &mut [f32] = borrow_k_norms!();
            let _ = restore_norms_only_f32(
                k_norms_f32,
                capacity,
                n_kv_heads,
                head_dim,
                range.clone(),
                k_payload,
            )?;
        }

        // --- V ---
        {
            let v_packed_mut: &mut [u8] = borrow_v_packed!();
            let _ = restore_packed_only(
                v_packed_mut,
                capacity,
                n_kv_heads,
                head_dim,
                bits_per_coord,
                range.clone(),
                v_payload,
            )?;
        }
        {
            let v_norms_f32: &mut [f32] = borrow_v_norms!();
            let _ = restore_norms_only_f32(
                v_norms_f32,
                capacity,
                n_kv_heads,
                head_dim,
                range.clone(),
                v_payload,
            )?;
        }

        // Helper closures defined here as fn items to avoid double-borrow.
        // (Defined as fn so they don't capture the surrounding scope.)
        fn restore_packed_only(
            packed_bytes_mut: &mut [u8],
            capacity: u32,
            n_kv_heads: u32,
            head_dim: u32,
            bits_per_coord: tq_packed::TqBitsPerCoord,
            range: std::ops::Range<u32>,
            payload: &[u8],
        ) -> Result<(), SpillErrorKind> {
            let (header, idx, _norms) =
                tq_packed::unpack_tq_v2_payload(payload).map_err(|_| SpillErrorKind::CodecErr)?;
            if header.bits_per_coord != bits_per_coord
                || header.head_dim != head_dim
                || header.n_kv_heads != n_kv_heads
                || header.n_tokens != (range.end - range.start)
            {
                return Err(SpillErrorKind::CodecErr);
            }
            let bits = bits_per_coord.0 as u64;
            if (head_dim as u64) * bits % 8 != 0 {
                return Err(SpillErrorKind::CodecErr);
            }
            let hd_packed = ((head_dim as u64) * bits / 8) as usize;
            let nkv_us = n_kv_heads as usize;
            let cap_us = capacity as usize;
            let n_tokens = (range.end - range.start) as usize;
            // **B-tq.7**: bounds-check write target.  Global layers
            // in Gemma 4 use dynamic capacity sizing — at server-B
            // post_admit time, the layer's buffer may be too small to
            // hold the snapshot's range (e.g. global layer cap=2 at
            // warmup vs snapshot range 0..256).  Writing OOB would
            // panic the worker thread.  Return CodecErr instead so
            // the spiller bails on this layer cleanly; the
            // prompt_cache replay path (R-P5) still gives the warm
            // benefit since it short-circuits prefill before the
            // cache needs to grow.
            let expected_buf_len = nkv_us
                .checked_mul(cap_us)
                .and_then(|v| v.checked_mul(hd_packed))
                .ok_or(SpillErrorKind::CodecErr)?;
            if packed_bytes_mut.len() != expected_buf_len {
                return Err(SpillErrorKind::CodecErr);
            }
            if (range.end as usize) > cap_us {
                return Err(SpillErrorKind::CodecErr);
            }
            for h in 0..nkv_us {
                let head_base = h * cap_us * hd_packed;
                let row_start = head_base + (range.start as usize) * hd_packed;
                let row_end = head_base + (range.end as usize) * hd_packed;
                let src_off = h * n_tokens * hd_packed;
                let src_end = src_off + n_tokens * hd_packed;
                packed_bytes_mut[row_start..row_end].copy_from_slice(&idx[src_off..src_end]);
            }
            Ok(())
        }
        fn restore_norms_only_f32(
            norms_f32_mut: &mut [f32],
            capacity: u32,
            n_kv_heads: u32,
            head_dim: u32,
            range: std::ops::Range<u32>,
            payload: &[u8],
        ) -> Result<(), SpillErrorKind> {
            let (header, _idx, norms) =
                tq_packed::unpack_tq_v2_payload(payload).map_err(|_| SpillErrorKind::CodecErr)?;
            if header.n_kv_heads != n_kv_heads
                || header.head_dim != head_dim
                || header.n_tokens != (range.end - range.start)
            {
                return Err(SpillErrorKind::CodecErr);
            }
            // **B-tq.7**: norms_per_pos derived from head_dim.  D=256
            // sliding layers → 1 norm/pos; D=512 global layers → 2.
            let norms_per_pos = ((head_dim as usize) / 256).max(1);
            let nkv_us = n_kv_heads as usize;
            let cap_us = capacity as usize;
            let n_tokens = (range.end - range.start) as usize;
            // **B-tq.7**: bounds-check write target (matches
            // restore_packed_only).
            let expected_norms_len = nkv_us
                .checked_mul(cap_us)
                .and_then(|v| v.checked_mul(norms_per_pos))
                .ok_or(SpillErrorKind::CodecErr)?;
            if norms_f32_mut.len() != expected_norms_len {
                return Err(SpillErrorKind::CodecErr);
            }
            if (range.end as usize) > cap_us {
                return Err(SpillErrorKind::CodecErr);
            }
            // norms F32 LE -> per-element decode into typed slice.
            // Layout: dst is `[nkv, capacity, norms_per_pos]` flat F32;
            // src is `[nkv, n_tokens, norms_per_pos]` packed F32 LE.
            for h in 0..nkv_us {
                let head_base = h * cap_us * norms_per_pos;
                for t in 0..n_tokens {
                    for k in 0..norms_per_pos {
                        let dst_idx = head_base + (range.start as usize + t) * norms_per_pos + k;
                        let src_off =
                            ((h * n_tokens + t) * norms_per_pos + k) * 4;
                        let bytes = [
                            norms[src_off],
                            norms[src_off + 1],
                            norms[src_off + 2],
                            norms[src_off + 3],
                        ];
                        norms_f32_mut[dst_idx] = f32::from_le_bytes(bytes);
                    }
                }
            }
            Ok(())
        }

        Ok(())
    }

}
