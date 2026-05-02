//! Caller-owned scratch arena for the 7 chunk-internal allocations of
//! [`super::gpu_delta_net::apply_gated_delta_net_chunk`] (ADR-015 iter78).
//!
//! Mirrors [`super::DnPrefillArena`] (iter74),
//! [`super::DenseFfnArena`] / [`super::MoeFfnArena`] (iter72), and
//! [`super::FaPrefillArena`] (ADR-013 P21 stage-1) for the chunk-pipeline
//! wrapper body in [`super::gpu_delta_net::apply_gated_delta_net_chunk`].
//!
//! # Why
//!
//! `wave5b8_profile` measurements at apex pp4096 (qwen3.6-35b-a3b q4_0-flat,
//! 30 DN layers + 10 FA layers) show two chunk-pipeline encoder Sections
//! with `ffn.alloc_scratch`-shaped quantile skew:
//!
//! - `chunk.allocs` — 7 fresh [`MlxDevice::alloc_buffer`] calls at
//!   [`super::gpu_delta_net`] lines 1153–1184 (pre-encoder).
//! - `chunk.commit_wait` — the apply_gated_delta_net_chunk encoder's
//!   `commit_and_wait_labeled("layer.gdn.chunk_attn")` at line 1430.
//!
//! Per the iter77 critical-path audit at
//! `/tmp/cfa-iter77/research/CRITICAL-PATH-AUDIT.md` (HIGH confidence):
//!
//! - The 7 allocs sit IMMEDIATELY after the chunk-prep encoder's
//!   `enc.commit_and_wait()` (gpu_delta_net.rs:1972 non-arena;
//!   gpu_delta_net.rs:2476 arena). The GPU is GUARANTEED idle when the
//!   alloc block runs.
//! - This is structurally identical to iter72/iter74 winning topology
//!   (alloc happens in a GPU-idle window), NOT iter76's
//!   absorbed topology (alloc happens during in-flight prior-layer GPU
//!   work).
//! - At apex pp4096 with 30 DN layers, this lifts ~6 GB of fresh
//!   `device.alloc_buffer` per prefill (~200 MB × 30 layers).
//!
//! # Mechanism mirror to iter74
//!
//! The 7 chunk-internal scratches are written + read within ONE chunk
//! pipeline encoder (the `enc` opened at gpu_delta_net.rs:1237 and
//! committed at line 1430) and never read by the next layer (the only
//! inter-layer values are `output_buf` and `final_state`, both
//! caller-provided).
//!
//! Lifting them to caller scope keeps them alive for the full prefill,
//! eliminating the per-layer alloc churn. Allocated ONCE at the top of
//! `forward_gpu_impl`, reused across all 30 DN layers, dropped at the end
//! of `forward_gpu_impl` AFTER the final `commit_and_wait_labeled` at the
//! output head — same iter58b residency-rescission protection contract as
//! [`super::FaPrefillArena`] / [`super::DenseFfnArena`] /
//! [`super::MoeFfnArena`] / [`super::DnPrefillArena`].
//!
//! # NOT included in the arena
//!
//! `output_buf` and `final_state` are **caller-provided** parameters of
//! `apply_gated_delta_net_chunk` and live OUTSIDE this arena. They are
//! the iter58b refactor's caller-owned ping-pong slots threaded straight
//! into the chunk pipeline. Conflating them with this arena would alias
//! recurrent-state slots, corrupting cross-layer residual flow silently.
//! This arena holds ONLY the 7 chunk-internal scratches.
//!
//! # Memory footprint (35B-A3B q4_0-flat pp4096)
//!
//! With seq=4096, n_v_heads=32, d_k=128, d_v=128 (canonical 35B-A3B):
//!
//! | Buffer | DType | Bytes |
//! |---|---|---:|
//! | `q_expanded` | F32 | seq*nv*dk*4 = 67.1 MB |
//! | `k_expanded` | F32 | same = 67.1 MB |
//! | `q_bf16` | BF16 | seq*nv*dk*2 = 33.6 MB |
//! | `k_bf16` | BF16 | same = 33.6 MB |
//! | `v_bf16` | BF16 | seq*nv*dv*2 = 33.6 MB |
//! | `g_log_decay` | F32 | seq*nv*4 = 0.5 MB |
//! | `o_bf16` | BF16 | seq*nv*dv*2 = 33.6 MB |
//! | **Total** | | **~268 MB** |
//!
//! Live for the entire prefill duration (replaces 30× per-layer
//! allocations of the same total at the iter77 baseline). M5 Max 128 GB
//! unified memory keeps this well within budget.

use anyhow::{anyhow, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

/// Caller-owned scratch arena for the 7 chunk-internal allocations of
/// [`super::gpu_delta_net::apply_gated_delta_net_chunk`].
///
/// Contains the 7 transient F32/BF16 scratches that the chunk-pipeline
/// wrapper currently allocates per-call via `device.alloc_buffer`. Lifting
/// them to caller scope keeps them alive for the full prefill,
/// eliminating the per-layer alloc churn captured by the W-5b.8
/// `chunk.allocs` skew bucket (iter77 audit predicted ~30× outlier ratio,
/// matching iter72's `ffn.alloc_scratch` precedent).
///
/// All scratches are sized for the actual prefill `(seq_len, …)` shape;
/// per-call `validate_fits` guards against accidental shape drift on a
/// future model that mixes DN layers of different shapes (Qwen3.6 35B-A3B
/// is uniform — every DN layer has the same n_v_heads, d_k, d_v).
pub struct ChunkAllocsArena {
    /// `[seq * n_v_heads * d_k]` F32 — GQA-tiled-expanded Q (input to
    /// F32→BF16 cast).
    pub q_expanded_buf: MlxBuffer,
    /// `[seq * n_v_heads * d_k]` F32 — GQA-tiled-expanded K (input to
    /// F32→BF16 cast).
    pub k_expanded_buf: MlxBuffer,
    /// `[seq * n_v_heads * d_k]` BF16 — Q after F32→BF16 cast (chunk
    /// pipeline input dtype).
    pub q_bf16_buf: MlxBuffer,
    /// `[seq * n_v_heads * d_k]` BF16 — K after F32→BF16 cast.
    pub k_bf16_buf: MlxBuffer,
    /// `[seq * n_v_heads * d_v]` BF16 — V after F32→BF16 cast.
    pub v_bf16_buf: MlxBuffer,
    /// `[seq * n_v_heads]` F32 — sign-flipped g for FLA's
    /// `g_log_decay = log(alpha)` convention.
    pub g_log_decay_buf: MlxBuffer,
    /// `[seq * n_v_heads * d_v]` BF16 — chunk-pipeline output (cast back
    /// to F32 before returning to caller-provided `output_buf`).
    pub o_bf16_buf: MlxBuffer,

    // ── Capacity bookkeeping ───────────────────────────────────────────────
    /// The `seq_capacity` value the arena was allocated for.
    pub seq_capacity: u32,
    /// The `n_v_heads` value the arena was allocated for.
    pub n_v_heads: u32,
    /// The `d_k` value the arena was allocated for.
    pub d_k: u32,
    /// The `d_v` value the arena was allocated for.
    pub d_v: u32,
}

impl ChunkAllocsArena {
    /// Allocate all 7 chunk-internal scratches sized for a single prefill
    /// pass with the given shape parameters.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any dimension is zero or any
    /// [`MlxDevice::alloc_buffer`] call fails.
    pub fn new(
        device: &MlxDevice,
        seq_capacity: u32,
        n_v_heads: u32,
        d_k: u32,
        d_v: u32,
    ) -> Result<Self> {
        if seq_capacity == 0 || n_v_heads == 0 || d_k == 0 || d_v == 0 {
            return Err(anyhow!(
                "ChunkAllocsArena::new: zero dim \
                 seq_capacity={} n_v_heads={} d_k={} d_v={}",
                seq_capacity,
                n_v_heads,
                d_k,
                d_v,
            ));
        }

        let seq = seq_capacity as usize;
        let nv = n_v_heads as usize;
        let dk = d_k as usize;
        let dv = d_v as usize;

        // Match the element-count expressions in
        // `apply_gated_delta_net_chunk` exactly (gpu_delta_net.rs:1121-1126).
        let q_elems_exp = seq * nv * dk; // [T, H, K] expanded
        let v_elems = seq * nv * dv; // [T, H, V]
        let g_elems = seq * nv; // [T, H]
        let out_elems_bf16 = v_elems; // [T, H, V] bf16

        let q_expanded_buf = device
            .alloc_buffer(q_elems_exp * 4, DType::F32, vec![q_elems_exp])
            .map_err(|e| anyhow!("ChunkAllocsArena alloc q_expanded_buf: {e}"))?;
        let k_expanded_buf = device
            .alloc_buffer(q_elems_exp * 4, DType::F32, vec![q_elems_exp])
            .map_err(|e| anyhow!("ChunkAllocsArena alloc k_expanded_buf: {e}"))?;
        let q_bf16_buf = device
            .alloc_buffer(q_elems_exp * 2, DType::BF16, vec![q_elems_exp])
            .map_err(|e| anyhow!("ChunkAllocsArena alloc q_bf16_buf: {e}"))?;
        let k_bf16_buf = device
            .alloc_buffer(q_elems_exp * 2, DType::BF16, vec![q_elems_exp])
            .map_err(|e| anyhow!("ChunkAllocsArena alloc k_bf16_buf: {e}"))?;
        let v_bf16_buf = device
            .alloc_buffer(v_elems * 2, DType::BF16, vec![v_elems])
            .map_err(|e| anyhow!("ChunkAllocsArena alloc v_bf16_buf: {e}"))?;
        let g_log_decay_buf = device
            .alloc_buffer(g_elems * 4, DType::F32, vec![g_elems])
            .map_err(|e| anyhow!("ChunkAllocsArena alloc g_log_decay_buf: {e}"))?;
        let o_bf16_buf = device
            .alloc_buffer(out_elems_bf16 * 2, DType::BF16, vec![out_elems_bf16])
            .map_err(|e| anyhow!("ChunkAllocsArena alloc o_bf16_buf: {e}"))?;

        Ok(Self {
            q_expanded_buf,
            k_expanded_buf,
            q_bf16_buf,
            k_bf16_buf,
            v_bf16_buf,
            g_log_decay_buf,
            o_bf16_buf,
            seq_capacity,
            n_v_heads,
            d_k,
            d_v,
        })
    }

    /// Validate that a per-call shape fits inside the arena's capacity.
    /// The arena is sized for the actual prefill `seq_len`, so equality
    /// is the common case.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `seq_len > self.seq_capacity` or any shape field
    /// differs from the recorded values.
    pub fn validate_fits(
        &self,
        seq_len: u32,
        n_v_heads: u32,
        d_k: u32,
        d_v: u32,
    ) -> Result<()> {
        if seq_len > self.seq_capacity {
            return Err(anyhow!(
                "ChunkAllocsArena::validate_fits: seq_len {} exceeds capacity {}",
                seq_len,
                self.seq_capacity
            ));
        }
        if n_v_heads != self.n_v_heads || d_k != self.d_k || d_v != self.d_v {
            return Err(anyhow!(
                "ChunkAllocsArena::validate_fits: shape mismatch — \
                 arena (nv={}, dk={}, dv={}) vs \
                 call (nv={}, dk={}, dv={})",
                self.n_v_heads,
                self.d_k,
                self.d_v,
                n_v_heads,
                d_k,
                d_v,
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device_or_skip() -> Option<MlxDevice> {
        MlxDevice::new().ok()
    }

    /// Qwen3.6 35B-A3B canonical chunk shape at pp=4096:
    /// n_v_heads=32, d_k=128, d_v=128.
    /// (Shape values consistent with the apex DWQ q4_0-flat fixture.)
    #[test]
    fn test_chunk_allocs_arena_new_qwen36_35b_pp4096() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!(
                    "test_chunk_allocs_arena_new_qwen36_35b_pp4096: \
                     skipping — no Metal device"
                );
                return;
            }
        };
        let (seq, nv, dk, dv) = (4096u32, 32u32, 128u32, 128u32);
        let arena =
            ChunkAllocsArena::new(&device, seq, nv, dk, dv).expect("chunk allocs arena pp4096");

        assert_eq!(arena.seq_capacity, seq);
        assert_eq!(arena.n_v_heads, nv);
        assert_eq!(arena.d_k, dk);
        assert_eq!(arena.d_v, dv);

        let q_elems_exp = (seq as usize) * (nv as usize) * (dk as usize);
        let v_elems = (seq as usize) * (nv as usize) * (dv as usize);
        let g_elems = (seq as usize) * (nv as usize);

        assert_eq!(arena.q_expanded_buf.byte_len(), q_elems_exp * 4, "q_expanded_buf");
        assert_eq!(arena.k_expanded_buf.byte_len(), q_elems_exp * 4, "k_expanded_buf");
        assert_eq!(arena.q_bf16_buf.byte_len(), q_elems_exp * 2, "q_bf16_buf");
        assert_eq!(arena.k_bf16_buf.byte_len(), q_elems_exp * 2, "k_bf16_buf");
        assert_eq!(arena.v_bf16_buf.byte_len(), v_elems * 2, "v_bf16_buf");
        assert_eq!(arena.g_log_decay_buf.byte_len(), g_elems * 4, "g_log_decay_buf");
        assert_eq!(arena.o_bf16_buf.byte_len(), v_elems * 2, "o_bf16_buf");
    }

    /// Smaller shape sanity-check.
    #[test]
    fn test_chunk_allocs_arena_new_small_shape() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!(
                    "test_chunk_allocs_arena_new_small_shape: skipping — no Metal device"
                );
                return;
            }
        };
        let arena =
            ChunkAllocsArena::new(&device, 128, 8, 32, 32).expect("chunk allocs arena small");
        assert_eq!(arena.seq_capacity, 128);
        assert_eq!(arena.n_v_heads, 8);
        assert_eq!(arena.d_k, 32);
        assert_eq!(arena.d_v, 32);
    }

    /// Zero-dim rejection (each parameter individually).
    #[test]
    fn test_chunk_allocs_arena_new_zero_dim_rejected() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!(
                    "test_chunk_allocs_arena_new_zero_dim_rejected: \
                     skipping — no Metal device"
                );
                return;
            }
        };
        assert!(ChunkAllocsArena::new(&device, 0, 8, 32, 32).is_err());
        assert!(ChunkAllocsArena::new(&device, 128, 0, 32, 32).is_err());
        assert!(ChunkAllocsArena::new(&device, 128, 8, 0, 32).is_err());
        assert!(ChunkAllocsArena::new(&device, 128, 8, 32, 0).is_err());
    }

    /// validate_fits exact match returns Ok.
    #[test]
    fn test_chunk_allocs_validate_fits_exact_match() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!(
                    "test_chunk_allocs_validate_fits_exact_match: skipping — no Metal device"
                );
                return;
            }
        };
        let arena =
            ChunkAllocsArena::new(&device, 256, 8, 32, 32).expect("chunk allocs arena new");
        assert!(arena.validate_fits(256, 8, 32, 32).is_ok());
        // seq_len < capacity also Ok.
        assert!(arena.validate_fits(128, 8, 32, 32).is_ok());
    }

    /// validate_fits seq overrun returns Err.
    #[test]
    fn test_chunk_allocs_validate_fits_seq_overrun() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!(
                    "test_chunk_allocs_validate_fits_seq_overrun: skipping — no Metal device"
                );
                return;
            }
        };
        let arena =
            ChunkAllocsArena::new(&device, 256, 8, 32, 32).expect("chunk allocs arena new");
        assert!(arena.validate_fits(512, 8, 32, 32).is_err());
    }

    /// validate_fits shape mismatch returns Err for each axis individually.
    #[test]
    fn test_chunk_allocs_validate_fits_shape_mismatch() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!(
                    "test_chunk_allocs_validate_fits_shape_mismatch: skipping — no Metal device"
                );
                return;
            }
        };
        let arena =
            ChunkAllocsArena::new(&device, 256, 8, 32, 32).expect("chunk allocs arena new");
        assert!(arena.validate_fits(256, 4, 32, 32).is_err()); // nv
        assert!(arena.validate_fits(256, 8, 16, 32).is_err()); // dk
        assert!(arena.validate_fits(256, 8, 32, 16).is_err()); // dv
    }

    /// Verifies that device.alloc_buffer zero-initialises all buffers —
    /// same iter61a guarantee FaPrefillArena/DenseFfnArena/DnPrefillArena
    /// rely on for safe re-use across DN layers in a prefill.
    #[test]
    fn test_chunk_allocs_arena_buffers_zero_initialised() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!(
                    "test_chunk_allocs_arena_buffers_zero_initialised: \
                     skipping — no Metal device"
                );
                return;
            }
        };
        let arena =
            ChunkAllocsArena::new(&device, 64, 4, 16, 16).expect("chunk allocs arena new");

        // F32 buffers
        let f32_bufs: [(&MlxBuffer, &str); 3] = [
            (&arena.q_expanded_buf, "q_expanded_buf"),
            (&arena.k_expanded_buf, "k_expanded_buf"),
            (&arena.g_log_decay_buf, "g_log_decay_buf"),
        ];
        for (buf, name) in &f32_bufs {
            let slice = buf
                .as_slice::<f32>()
                .unwrap_or_else(|e| panic!("{name} as_slice::<f32> failed: {e}"));
            let check_len = 16.min(slice.len());
            for (i, &v) in slice[..check_len].iter().enumerate() {
                assert_eq!(
                    v, 0.0f32,
                    "{name}[{i}] = {v} (expected zero from device.alloc_buffer)"
                );
            }
        }

        // BF16 buffers — read as raw u16 and check zero (BF16 0.0 == 0u16)
        let bf16_bufs: [(&MlxBuffer, &str); 4] = [
            (&arena.q_bf16_buf, "q_bf16_buf"),
            (&arena.k_bf16_buf, "k_bf16_buf"),
            (&arena.v_bf16_buf, "v_bf16_buf"),
            (&arena.o_bf16_buf, "o_bf16_buf"),
        ];
        for (buf, name) in &bf16_bufs {
            let slice = buf
                .as_slice::<u16>()
                .unwrap_or_else(|e| panic!("{name} as_slice::<u16> failed: {e}"));
            let check_len = 16.min(slice.len());
            for (i, &v) in slice[..check_len].iter().enumerate() {
                assert_eq!(
                    v, 0u16,
                    "{name}[{i}] = {v} (expected zero from device.alloc_buffer)"
                );
            }
        }
    }

    /// q_expanded and k_expanded must have identical sizes (both
    /// `[T, H, K]` after GQA-tiled expansion). This contract is relied on
    /// by the Stage A0 `dispatch_repeat_tiled_f32` writes and the Stage A
    /// F32→BF16 casts.
    #[test]
    fn test_chunk_allocs_q_k_expanded_same_size() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!(
                    "test_chunk_allocs_q_k_expanded_same_size: skipping — no Metal device"
                );
                return;
            }
        };
        let arena =
            ChunkAllocsArena::new(&device, 256, 8, 32, 32).expect("chunk allocs arena new");
        assert_eq!(
            arena.q_expanded_buf.byte_len(),
            arena.k_expanded_buf.byte_len(),
            "q_expanded and k_expanded must be the same size (both [T, H, K] F32)"
        );
        assert_eq!(
            arena.q_bf16_buf.byte_len(),
            arena.k_bf16_buf.byte_len(),
            "q_bf16 and k_bf16 must be the same size (both [T, H, K] BF16)"
        );
        assert_eq!(
            arena.v_bf16_buf.byte_len(),
            arena.o_bf16_buf.byte_len(),
            "v_bf16 and o_bf16 must be the same size (both [T, H, V] BF16)"
        );
    }
}
