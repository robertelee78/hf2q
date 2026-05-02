//! Caller-owned scratch arena for the FA prefill bridge across all FA
//! layers in a single prefill.
//!
//! # Lifetime contract
//!
//! The arena is allocated ONCE at the top of `forward_gpu_impl` (after
//! `ensure_gpu_cache_primed`, before the per-layer loop) when `seq_len > 1`
//! AND the model has at least one `FullAttn` layer. It is reused across all
//! FA layers in the loop, then dropped at the end of `forward_gpu_impl`
//! AFTER the final encoder `commit_and_wait_labeled` at the output-head.
//!
//! **Why this prevents the iter58b residency-rescission failure mode:**
//! The arena buffers are owned by `forward_gpu_impl` for the entire prefill.
//! They do NOT drop at wrapper return. Therefore:
//!
//! 1. The wrapper's `enc.commit()` returns immediately; the wrapper returns
//!    immediately.
//! 2. Nothing inside the wrapper drops a `device.alloc_buffer` `MlxBuffer`
//!    that is still referenced by an in-flight command buffer.
//! 3. No deferred `removeAllocation:` is staged on the residency set when
//!    the wrapper returns.
//! 4. The next encoder's `commit*` does NOT flush a stale
//!    residency-rescission for buffers still referenced by the wrapper's CB.
//! 5. The iter58b failure mode — where a wrapper-local buffer drop staged a
//!    `removeAllocation:` that was flushed mid-flight by the next CB's
//!    `commit*`, demoting still-referenced pages and producing garbage values
//!    — is structurally unreachable.
//!
//! Caller-owned arena outlives all encoders that reference it; the iter58b
//! residency-rescission failure mode cannot fire because no `MlxBuffer::Drop`
//! runs in the wrapper frame.
//!
//! See ADR-013 P21 Stage 1 for full rationale.

use anyhow::{anyhow, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

/// Caller-owned scratch arena for the FA prefill bridge (ops1-4 + ops6-7)
/// across all FA layers in a single prefill.
///
/// Contains the 7 BF16 scratch buffers that `apply_flash_attn_prefill_seq_major`
/// currently allocates function-locally. Lifting them to caller scope keeps
/// them alive for the full prefill, making wrapper-terminal `commit()` safe
/// (no residency-set rescission race — see module-level doc for the full
/// iter58b analysis).
///
/// The 8th scratch (`out_seq`, the F32 return value of
/// `apply_flash_attn_prefill_seq_major`) is intentionally NOT included in
/// the arena — it is the function return value, consumed by the caller chain
/// within the same prefill scope, and does not require arena lifetime.
pub struct FaPrefillArena {
    /// `[seq, nh, d]` BF16 — Q in seq-major layout.
    pub q_bf16_seq: MlxBuffer,
    /// `[1, nh, seq, d]` BF16 — Q in head-major layout.
    pub q_bf16_hm: MlxBuffer,
    /// `[seq, nkv, d]` BF16 — K in seq-major layout.
    pub k_bf16_seq: MlxBuffer,
    /// `[1, nkv, seq, d]` BF16 — K in head-major layout.
    pub k_bf16_hm: MlxBuffer,
    /// `[seq, nkv, d]` BF16 — V in seq-major layout.
    pub v_bf16_seq: MlxBuffer,
    /// `[1, nkv, seq, d]` BF16 — V in head-major layout.
    pub v_bf16_hm: MlxBuffer,
    /// `[1, nh, seq, d]` BF16 — attention output in head-major layout.
    pub out_bf16_hm: MlxBuffer,

    // ── Capacity bookkeeping ─────────────────────────────────────────────
    // The arena is sized for the actual prefill seq_len. Per-layer calls
    // may have the same or shorter seq_len; validate_fits() guards against
    // accidental overrun.

    /// The `seq_capacity` value the arena was allocated for.
    pub seq_capacity: u32,
    /// The `n_heads` value the arena was allocated for.
    pub n_heads: u32,
    /// The `n_kv_heads` value the arena was allocated for.
    pub n_kv_heads: u32,
    /// The `head_dim` value the arena was allocated for.
    pub head_dim: u32,
}

impl FaPrefillArena {
    /// Allocate all 7 BF16 scratch buffers sized for a single prefill pass
    /// with the given shape parameters.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - Any dimension is zero (`seq_capacity`, `n_heads`, `n_kv_heads`, or
    ///   `head_dim` equals 0).
    /// - `n_heads % n_kv_heads != 0` (GQA divisibility violated).
    /// - Any `device.alloc_buffer` call fails (out of GPU memory).
    pub fn new(
        device: &MlxDevice,
        seq_capacity: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) -> Result<Self> {
        if seq_capacity == 0 || n_heads == 0 || n_kv_heads == 0 || head_dim == 0 {
            return Err(anyhow!(
                "FaPrefillArena::new: zero dim \
                 seq_capacity={} n_heads={} n_kv_heads={} head_dim={}",
                seq_capacity,
                n_heads,
                n_kv_heads,
                head_dim
            ));
        }
        if n_heads % n_kv_heads != 0 {
            return Err(anyhow!(
                "FaPrefillArena::new: n_heads ({}) must be divisible by n_kv_heads ({})",
                n_heads,
                n_kv_heads
            ));
        }

        let seq = seq_capacity as usize;
        let nh = n_heads as usize;
        let nkv = n_kv_heads as usize;
        let d = head_dim as usize;

        // Element counts match exactly what apply_flash_attn_prefill_seq_major
        // currently allocates at lines 1046-1049 of gpu_full_attn.rs.
        let q_elems = seq * nh * d;
        let k_elems = seq * nkv * d;
        let v_elems = seq * nkv * d; // v_elems == k_elems
        let out_elems = seq * nh * d; // out_elems == q_elems

        // Each field's byte_len = element_count * 2 (BF16 = 2 bytes/element).
        // Shapes match A.2 verbatim.
        let q_bf16_seq = device
            .alloc_buffer(q_elems * 2, DType::BF16, vec![seq, nh, d])
            .map_err(|e| anyhow!("FaPrefillArena::new alloc q_bf16_seq: {e}"))?;
        let q_bf16_hm = device
            .alloc_buffer(q_elems * 2, DType::BF16, vec![1, nh, seq, d])
            .map_err(|e| anyhow!("FaPrefillArena::new alloc q_bf16_hm: {e}"))?;
        let k_bf16_seq = device
            .alloc_buffer(k_elems * 2, DType::BF16, vec![seq, nkv, d])
            .map_err(|e| anyhow!("FaPrefillArena::new alloc k_bf16_seq: {e}"))?;
        let k_bf16_hm = device
            .alloc_buffer(k_elems * 2, DType::BF16, vec![1, nkv, seq, d])
            .map_err(|e| anyhow!("FaPrefillArena::new alloc k_bf16_hm: {e}"))?;
        let v_bf16_seq = device
            .alloc_buffer(v_elems * 2, DType::BF16, vec![seq, nkv, d])
            .map_err(|e| anyhow!("FaPrefillArena::new alloc v_bf16_seq: {e}"))?;
        let v_bf16_hm = device
            .alloc_buffer(v_elems * 2, DType::BF16, vec![1, nkv, seq, d])
            .map_err(|e| anyhow!("FaPrefillArena::new alloc v_bf16_hm: {e}"))?;
        let out_bf16_hm = device
            .alloc_buffer(out_elems * 2, DType::BF16, vec![1, nh, seq, d])
            .map_err(|e| anyhow!("FaPrefillArena::new alloc out_bf16_hm: {e}"))?;

        Ok(Self {
            q_bf16_seq,
            q_bf16_hm,
            k_bf16_seq,
            k_bf16_hm,
            v_bf16_seq,
            v_bf16_hm,
            out_bf16_hm,
            seq_capacity,
            n_heads,
            n_kv_heads,
            head_dim,
        })
    }

    /// Validate that a per-layer call's shape fits inside the arena's
    /// capacity. The arena is typically sized for the actual prefill
    /// `seq_len`, so equality is the common case.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - `seq_len > self.seq_capacity` (would overrun the allocated buffer).
    /// - `n_heads`, `n_kv_heads`, or `head_dim` differ from the recorded
    ///   values (buffers were sized for a different shape).
    pub fn validate_fits(
        &self,
        seq_len: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        if seq_len > self.seq_capacity {
            return Err(anyhow!(
                "FaPrefillArena::validate_fits: seq_len {} exceeds capacity {}",
                seq_len,
                self.seq_capacity
            ));
        }
        if n_heads != self.n_heads
            || n_kv_heads != self.n_kv_heads
            || head_dim != self.head_dim
        {
            return Err(anyhow!(
                "FaPrefillArena::validate_fits: shape mismatch — \
                 arena (n_heads={}, n_kv_heads={}, head_dim={}) vs \
                 call (n_heads={}, n_kv_heads={}, head_dim={})",
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                n_heads,
                n_kv_heads,
                head_dim,
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: acquire an MlxDevice or skip if no Metal device is available
    /// (e.g. headless CI without GPU).
    fn device_or_skip() -> Option<MlxDevice> {
        MlxDevice::new().ok()
    }

    /// Qwen3.5/3.6 canonical shape at pp=101: seq=101, nh=16, nkv=2, d=256.
    /// Verifies all 7 fields' byte_len() match the A.2 formula and
    /// seq_capacity is recorded correctly.
    #[test]
    fn test_arena_new_qwen35_pp101() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_arena_new_qwen35_pp101: skipping — no Metal device");
                return;
            }
        };
        let (seq, nh, nkv, d) = (101usize, 16usize, 2usize, 256usize);
        let arena = FaPrefillArena::new(&device, seq as u32, nh as u32, nkv as u32, d as u32)
            .expect("arena new pp101");

        assert_eq!(arena.seq_capacity, seq as u32);
        assert_eq!(arena.n_heads, nh as u32);
        assert_eq!(arena.n_kv_heads, nkv as u32);
        assert_eq!(arena.head_dim, d as u32);

        let q_bytes = seq * nh * d * 2;
        let k_bytes = seq * nkv * d * 2;
        assert_eq!(arena.q_bf16_seq.byte_len(), q_bytes, "q_bf16_seq byte_len");
        assert_eq!(arena.q_bf16_hm.byte_len(), q_bytes, "q_bf16_hm byte_len");
        assert_eq!(arena.k_bf16_seq.byte_len(), k_bytes, "k_bf16_seq byte_len");
        assert_eq!(arena.k_bf16_hm.byte_len(), k_bytes, "k_bf16_hm byte_len");
        assert_eq!(arena.v_bf16_seq.byte_len(), k_bytes, "v_bf16_seq byte_len");
        assert_eq!(arena.v_bf16_hm.byte_len(), k_bytes, "v_bf16_hm byte_len");
        assert_eq!(arena.out_bf16_hm.byte_len(), q_bytes, "out_bf16_hm byte_len");
    }

    /// Qwen3.5/3.6 canonical shape at pp=4096: verifies no allocation failure
    /// and capacity recorded.
    #[test]
    fn test_arena_new_qwen35_pp4096() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_arena_new_qwen35_pp4096: skipping — no Metal device");
                return;
            }
        };
        let arena =
            FaPrefillArena::new(&device, 4096, 16, 2, 256).expect("arena new pp4096");
        assert_eq!(arena.seq_capacity, 4096);
    }

    /// Passing seq=0, nh=0, nkv=0, or d=0 each individually must return Err.
    #[test]
    fn test_arena_new_zero_dim_rejected() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_arena_new_zero_dim_rejected: skipping — no Metal device");
                return;
            }
        };
        // seq = 0
        assert!(
            FaPrefillArena::new(&device, 0, 16, 2, 256).is_err(),
            "seq=0 should be rejected"
        );
        // n_heads = 0
        assert!(
            FaPrefillArena::new(&device, 101, 0, 2, 256).is_err(),
            "n_heads=0 should be rejected"
        );
        // n_kv_heads = 0
        assert!(
            FaPrefillArena::new(&device, 101, 16, 0, 256).is_err(),
            "n_kv_heads=0 should be rejected"
        );
        // head_dim = 0
        assert!(
            FaPrefillArena::new(&device, 101, 16, 2, 0).is_err(),
            "head_dim=0 should be rejected"
        );
    }

    /// n_heads=15, n_kv_heads=2 (15 % 2 != 0) must return Err.
    #[test]
    fn test_arena_new_gqa_divisibility_rejected() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_arena_new_gqa_divisibility_rejected: skipping — no Metal device");
                return;
            }
        };
        assert!(
            FaPrefillArena::new(&device, 101, 15, 2, 256).is_err(),
            "n_heads=15, n_kv_heads=2 (15 % 2 != 0) should be rejected"
        );
    }

    /// Arena sized for seq_capacity=128; validate_fits(seq_len=256) returns Err.
    #[test]
    fn test_validate_fits_seq_overrun() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_validate_fits_seq_overrun: skipping — no Metal device");
                return;
            }
        };
        let arena =
            FaPrefillArena::new(&device, 128, 16, 2, 256).expect("arena new seq128");
        assert!(
            arena.validate_fits(256, 16, 2, 256).is_err(),
            "seq_len=256 > capacity=128 should be rejected"
        );
    }

    /// Arena recorded n_heads=16; validate_fits(n_heads=8) returns Err.
    #[test]
    fn test_validate_fits_shape_mismatch() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_validate_fits_shape_mismatch: skipping — no Metal device");
                return;
            }
        };
        let arena =
            FaPrefillArena::new(&device, 128, 16, 2, 256).expect("arena new nh16");
        assert!(
            arena.validate_fits(128, 8, 2, 256).is_err(),
            "n_heads=8 vs arena n_heads=16 should be rejected"
        );
        assert!(
            arena.validate_fits(128, 16, 4, 256).is_err(),
            "n_kv_heads=4 vs arena n_kv_heads=2 should be rejected"
        );
        assert!(
            arena.validate_fits(128, 16, 2, 128).is_err(),
            "head_dim=128 vs arena head_dim=256 should be rejected"
        );
    }

    /// Arena recorded (seq=128, nh=16, nkv=2, d=256); validate_fits with
    /// the exact same values returns Ok.
    #[test]
    fn test_validate_fits_exact_match() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_validate_fits_exact_match: skipping — no Metal device");
                return;
            }
        };
        let arena =
            FaPrefillArena::new(&device, 128, 16, 2, 256).expect("arena new seq128");
        assert!(
            arena.validate_fits(128, 16, 2, 256).is_ok(),
            "exact-match validate_fits should return Ok"
        );
        // seq_len < capacity also Ok
        assert!(
            arena.validate_fits(64, 16, 2, 256).is_ok(),
            "seq_len=64 <= capacity=128 should be Ok"
        );
    }

    /// Verifies that device.alloc_buffer zero-initializes all buffers.
    /// Read the first 16 BF16 (u16) elements of each of the 7 buffers and
    /// assert all are zero. This exercises the ADR-015 iter61a zero-init
    /// guarantee documented in mlx_native/src/device.rs:120-146.
    #[test]
    fn test_arena_buffers_zero_initialized() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_arena_buffers_zero_initialized: skipping — no Metal device");
                return;
            }
        };
        let arena =
            FaPrefillArena::new(&device, 64, 16, 2, 256).expect("arena new seq64");

        let bufs: [(&MlxBuffer, &str); 7] = [
            (&arena.q_bf16_seq, "q_bf16_seq"),
            (&arena.q_bf16_hm, "q_bf16_hm"),
            (&arena.k_bf16_seq, "k_bf16_seq"),
            (&arena.k_bf16_hm, "k_bf16_hm"),
            (&arena.v_bf16_seq, "v_bf16_seq"),
            (&arena.v_bf16_hm, "v_bf16_hm"),
            (&arena.out_bf16_hm, "out_bf16_hm"),
        ];
        for (buf, name) in &bufs {
            let slice = buf
                .as_slice::<u16>()
                .unwrap_or_else(|e| panic!("{name} as_slice::<u16> failed: {e}"));
            let check_len = 16.min(slice.len());
            for (i, &v) in slice[..check_len].iter().enumerate() {
                assert_eq!(
                    v,
                    0u16,
                    "{name}[{i}] = {v:#06x} (BF16 0x0000 = +0.0), expected zero"
                );
            }
        }
    }
}
