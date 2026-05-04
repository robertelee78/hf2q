//! Image-token-positioned residual add (ADR-005 iter-224 row 3 — Wedge-4c.5).
//!
//! Implements the LM-side counterpart of /opt/llama.cpp/src/models/qwen3vl.cpp:96-100:
//!
//! ```text
//! if (il < (int) n_deepstack_layers) {
//!     ggml_tensor * ds = ggml_view_2d(ctx0, res->t_inp_embd, n_embd, n_tokens,
//!         res->t_inp_embd->nb[1], (il + 1) * n_embd * sizeof(float));
//!     cur = ggml_add(ctx0, cur, ds);
//! }
//! ```
//!
//! In peer's path, `ds` has `n_tokens` columns and non-image positions
//! carry zeros (filled by the token-embedding path's `ggml_pad`); the
//! add is therefore full-tensor but a no-op at non-image positions.
//!
//! Hf2q's path does NOT pre-zero the deepstack chunks across all
//! positions — the `compute_vision_embeddings_gpu_qwen3vl` augmented
//! buffer is dense over `n_image_tokens` rows, not the full LM
//! sequence. We could pre-zero a `[n_tokens, hidden]` scatter at the
//! engine seam (simpler, one-time CPU cost) BUT processing all
//! `n_tokens * hidden` floats per LM layer scales O(seq_len) when only
//! the image-token rows ever change. The position-gated kernel here
//! processes exactly `n_image_tokens * hidden` floats per call —
//! O(image-tokens) scaling regardless of prompt length.
//!
//! ## Mathematical contract
//!
//! Inputs:
//!   - `cur`: post-FFN-residual tensor, shape `[n_tokens, hidden]`
//!     row-major F32. Mutated in-place.
//!   - `chunk_il_plus_1`: deepstack chunk for the current LM layer,
//!     shape `[n_image_tokens, hidden]` row-major F32.
//!   - `image_token_positions`: `&[u32]` of length `n_image_tokens`
//!     listing the positions in `cur`'s n_tokens axis where the image
//!     tokens reside. Must satisfy `0 <= positions[k] < n_tokens` and
//!     all positions distinct (caller's contract — handler-side
//!     placeholder expansion ensures this; we don't re-validate
//!     uniqueness at GPU dispatch time).
//!
//! Operation:
//!   ```
//!   for k in 0..n_image_tokens:
//!     for h in 0..hidden:
//!       cur[positions[k]][h] += chunk[k][h]
//!   ```
//!
//! Other positions in `cur` are unchanged.
//!
//! ## Why a fresh module instead of extending vit_gpu.rs
//!
//! - `vit_gpu.rs` is the ViT-side toolbox — its primitives target the
//!   `[n_patches, hidden]` flow, not the LM's `[n_tokens, hidden]`
//!   stream. The LM-side hook lives at the engine seam where the
//!   per-layer kv_cache + soft-tokens already mix; pulling the kernel
//!   into a vision module would smear concerns.
//! - The kernel is tightly scoped: one Metal compute pipeline + one
//!   Rust dispatch wrapper + position-gated tests. A standalone module
//!   keeps the diff readable and the test surface compact.

use anyhow::{anyhow, Result};

use mlx_native::metal::MTLSize;
use mlx_native::ops::encode_helpers::KernelArg;
use mlx_native::{CommandEncoder, KernelRegistry, MlxBuffer, MlxDevice};

/// Per-dispatch shape parameters consumed by the Metal kernel.
///
/// Kept small (4×u32) so the encoder passes the struct via
/// `setBytes:length:atIndex:` rather than allocating a parameter
/// buffer.
#[repr(C)]
#[derive(Clone, Copy)]
struct ImageTokenResidualAddParams {
    /// `n_image_tokens` — number of image-token rows in `chunk` AND
    /// length of the `image_token_positions` array.
    n_image_tokens: u32,
    /// `hidden` — feature dim per row in both `cur` and `chunk`.
    hidden: u32,
    /// `n_tokens` — total token count along `cur`'s leading axis. Used
    /// only for a defensive in-kernel bounds check (positions must be
    /// `< n_tokens`).
    n_tokens: u32,
    /// Padding to align the struct to 16 bytes.
    _pad: u32,
}

/// Metal source for the in-place position-gated residual add. Each
/// thread handles one `(token, hidden_idx)` pair.
const IMAGE_TOKEN_RESIDUAL_ADD_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ImageTokenResidualAddParams {
    uint n_image_tokens;
    uint hidden;
    uint n_tokens;
    uint _pad;
};

// In-place: cur[positions[k], h] += chunk[k, h]  for k in [0, n_image_tokens),
//                                                h in [0, hidden).
// One thread per (k, h) pair; threads outside that grid early-return.
//
// Position bounds-check: a position >= n_tokens silently clamps to a
// no-op rather than corrupting memory. The caller (LM forward
// `embed_tokens_gpu_with_soft_tokens` + image-token expansion path)
// already validates positions before dispatch; this in-kernel guard
// is defense-in-depth so a future refactor can't turn a position
// off-by-one into a heap-overrun.
kernel void image_token_residual_add_f32(
    device       float* cur               [[buffer(0)]],
    device const float* chunk             [[buffer(1)]],
    device const uint*  positions         [[buffer(2)]],
    constant ImageTokenResidualAddParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint h = gid.x;
    uint k = gid.y;
    if (h >= params.hidden || k >= params.n_image_tokens) return;
    uint pos = positions[k];
    if (pos >= params.n_tokens) return;
    uint cur_idx   = pos * params.hidden + h;
    uint chunk_idx = k   * params.hidden + h;
    cur[cur_idx] = cur[cur_idx] + chunk[chunk_idx];
}
"#;

/// Register the residual-add shader source with `registry`. Idempotent;
/// safe to call multiple times against the same registry. Caller invokes
/// once per fresh `KernelRegistry` (typically at engine warm-up).
pub fn register_image_token_residual_add_shader(registry: &mut KernelRegistry) {
    registry.register_source(
        "image_token_residual_add_f32",
        IMAGE_TOKEN_RESIDUAL_ADD_SHADER,
    );
}

/// View any `Copy + repr(C)` POD as a byte slice (matches the
/// `pod_as_bytes` helper in vit_gpu.rs).
fn pod_as_bytes<T: Copy>(p: &T) -> &[u8] {
    // SAFETY: `T: Copy + repr(C)` with primitive fields means the
    // in-memory representation is contiguous and exactly
    // `size_of::<T>()` bytes.
    unsafe {
        std::slice::from_raw_parts(p as *const T as *const u8, std::mem::size_of::<T>())
    }
}

/// Position-gated residual add for image-token rows (in-place).
///
/// See module docstring for the mathematical contract. Caller must:
///   1. Register the shader source first via
///      `register_image_token_residual_add_shader(&mut registry)`.
///   2. Encode any preceding writes to `cur` and `chunk` BEFORE this
///      dispatch — the kernel reads `chunk` and reads-modifies-writes
///      `cur`. A `memory_barrier()` before/after the dispatch is the
///      caller's responsibility.
///
/// `image_token_positions` must be a non-empty `&[u32]` of length
/// equal to `n_image_tokens`. We upload it to a fresh F32-typed
/// MlxBuffer (the dtype is irrelevant for binding; the kernel reads it
/// as `device const uint*`).
///
/// # Errors
///
/// - `n_image_tokens == 0` (no work — caller should skip this dispatch).
/// - `hidden == 0` or `n_tokens == 0`.
/// - `image_token_positions.len() != n_image_tokens`.
/// - `cur.byte_len() < n_tokens * hidden * 4` (cur's storage too small).
/// - `chunk.byte_len() < n_image_tokens * hidden * 4` (chunk too small).
/// - propagated from kernel pipeline compile failures.
pub fn image_token_residual_add_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cur: &MlxBuffer,
    chunk: &MlxBuffer,
    image_token_positions: &[u32],
    n_tokens: u32,
    n_image_tokens: u32,
    hidden: u32,
) -> Result<()> {
    if n_image_tokens == 0 {
        return Err(anyhow!(
            "image_token_residual_add_gpu: n_image_tokens must be > 0 \
             (caller should skip this dispatch when no image tokens are present)"
        ));
    }
    if hidden == 0 || n_tokens == 0 {
        return Err(anyhow!(
            "image_token_residual_add_gpu: hidden ({}) and n_tokens ({}) must be > 0",
            hidden,
            n_tokens
        ));
    }
    if image_token_positions.len() != n_image_tokens as usize {
        return Err(anyhow!(
            "image_token_residual_add_gpu: image_token_positions.len()={} != \
             n_image_tokens={}",
            image_token_positions.len(),
            n_image_tokens
        ));
    }
    let cur_required = (n_tokens as usize) * (hidden as usize) * 4;
    let chunk_required = (n_image_tokens as usize) * (hidden as usize) * 4;
    // We compare against the slice's logical span (byte_len - byte_offset)
    // so a buffer view backed by a larger storage still validates.
    let cur_span = cur.byte_len().saturating_sub(cur.byte_offset() as usize);
    let chunk_span = chunk.byte_len().saturating_sub(chunk.byte_offset() as usize);
    if cur_span < cur_required {
        return Err(anyhow!(
            "image_token_residual_add_gpu: cur span {} < required {} \
             (n_tokens={} * hidden={} * 4)",
            cur_span,
            cur_required,
            n_tokens,
            hidden
        ));
    }
    if chunk_span < chunk_required {
        return Err(anyhow!(
            "image_token_residual_add_gpu: chunk span {} < required {} \
             (n_image_tokens={} * hidden={} * 4)",
            chunk_span,
            chunk_required,
            n_image_tokens,
            hidden
        ));
    }

    // Upload the positions array as a fresh u32-shaped F32-dtype
    // MlxBuffer (the kernel reads `device const uint*` regardless of
    // the dtype label; binding goes through metal_buffer + byte_offset
    // which are dtype-agnostic).
    let positions_bytes = (n_image_tokens as usize) * std::mem::size_of::<u32>();
    let mut positions_buf = device
        .alloc_buffer(positions_bytes, mlx_native::DType::F32, vec![n_image_tokens as usize])
        .map_err(|e| anyhow!("image_token_residual_add_gpu: alloc positions buffer: {e}"))?;
    {
        // SAFETY: just-allocated F32 buffer of n_image_tokens elements;
        // we view it as u32 (same byte width). Apple Silicon shared
        // memory is CPU-writable; no GPU work touches this buffer
        // before our dispatch.
        let dst: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(
                positions_buf.contents_ptr() as *mut u32,
                n_image_tokens as usize,
            )
        };
        dst.copy_from_slice(image_token_positions);
        // Silence unused_mut warning when the compiler can't see the
        // mutation through unsafe. The contents_ptr write above is the
        // mutation; positions_buf must be `mut` so we can pass &mut later
        // if the API ever requires it (today it doesn't).
        let _ = &mut positions_buf;
    }

    let pipeline = registry
        .get_pipeline("image_token_residual_add_f32", device.metal_device())
        .map_err(|e| anyhow!("image_token_residual_add_gpu: get_pipeline: {e}"))?;

    let params = ImageTokenResidualAddParams {
        n_image_tokens,
        hidden,
        n_tokens,
        _pad: 0,
    };
    let bytes = pod_as_bytes(&params);
    // Grid: x = hidden (row-axis parallelism), y = n_image_tokens.
    let grid = MTLSize::new(hidden as u64, n_image_tokens as u64, 1);
    // Threadgroup: keep small along y to fit common image-token counts
    // (256-1024 typical); along x clamp to 64 (Metal SIMD-group width
    // multiple) or `hidden` whichever is smaller.
    let tg_x = std::cmp::min(64u64, hidden as u64);
    let tg_y = std::cmp::min(8u64, n_image_tokens as u64);
    let tg = MTLSize::new(tg_x, tg_y.max(1), 1);
    encoder.encode_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(cur)),
            (1, KernelArg::Buffer(chunk)),
            (2, KernelArg::Buffer(&positions_buf)),
            (3, KernelArg::Bytes(bytes)),
        ],
        grid,
        tg,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_native::DType;

    /// Allocate an F32 MlxBuffer populated with `init(i) -> f32` for i
    /// in 0..n_elements. Helper mirrors the pattern in
    /// `vit_gpu.rs::tests` so the dispatch tests stay self-contained.
    fn alloc_f32(
        device: &MlxDevice,
        n_elements: usize,
        shape: Vec<usize>,
        init: impl Fn(usize) -> f32,
    ) -> MlxBuffer {
        let mut buf = device
            .alloc_buffer(n_elements * 4, DType::F32, shape)
            .expect("alloc_buffer F32");
        {
            let dst: &mut [f32] = buf.as_mut_slice::<f32>().expect("as_mut_slice F32");
            for (i, slot) in dst.iter_mut().enumerate().take(n_elements) {
                *slot = init(i);
            }
        }
        buf
    }

    /// Run a single `image_token_residual_add_gpu` dispatch on `(cur,
    /// chunk, positions)` and return the resulting `cur` rows.
    fn dispatch_once(
        cur_init: impl Fn(usize) -> f32,
        chunk_init: impl Fn(usize) -> f32,
        positions: &[u32],
        n_tokens: u32,
        hidden: u32,
    ) -> Vec<f32> {
        let n_image_tokens = positions.len() as u32;
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        register_image_token_residual_add_shader(&mut registry);
        let cur = alloc_f32(
            &device,
            (n_tokens as usize) * (hidden as usize),
            vec![n_tokens as usize, hidden as usize],
            cur_init,
        );
        let chunk = alloc_f32(
            &device,
            (n_image_tokens as usize) * (hidden as usize),
            vec![n_image_tokens as usize, hidden as usize],
            chunk_init,
        );
        let mut encoder = device.command_encoder().expect("encoder");
        image_token_residual_add_gpu(
            &mut encoder,
            &mut registry,
            &device,
            &cur,
            &chunk,
            positions,
            n_tokens,
            n_image_tokens,
            hidden,
        )
        .expect("dispatch");
        encoder.commit_and_wait().expect("commit_and_wait");
        cur.as_slice::<f32>()
            .expect("cur readback")
            .to_vec()
    }

    #[test]
    fn identity_zero_chunk_leaves_cur_unchanged() {
        let n_tokens = 6u32;
        let hidden = 4u32;
        let positions = [1u32, 3, 5];
        // chunk is all zeros → no add at any position.
        let result = dispatch_once(
            |i| (i as f32) * 0.5 + 1.0,
            |_| 0.0,
            &positions,
            n_tokens,
            hidden,
        );
        // Identity: every cur value is the original `cur_init` output.
        for i in 0..(n_tokens as usize * hidden as usize) {
            let expected = (i as f32) * 0.5 + 1.0;
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "cur[{i}] = {} != expected {}",
                result[i],
                expected
            );
        }
    }

    #[test]
    fn single_token_add_writes_one_row_only() {
        let n_tokens = 4u32;
        let hidden = 3u32;
        let positions = [2u32];
        // chunk row 0 = [10, 20, 30].
        let chunk_vals = [10.0f32, 20.0, 30.0];
        let result = dispatch_once(
            |_| 0.0,             // cur all zeros
            |i| chunk_vals[i],  // chunk[0] = [10,20,30]
            &positions,
            n_tokens,
            hidden,
        );
        // Row 2 must equal chunk_vals; every other row stays 0.
        for t in 0..n_tokens as usize {
            for h in 0..hidden as usize {
                let i = t * (hidden as usize) + h;
                if t == 2 {
                    assert!(
                        (result[i] - chunk_vals[h]).abs() < 1e-6,
                        "cur[2][{h}] = {}, expected {}",
                        result[i],
                        chunk_vals[h]
                    );
                } else {
                    assert!(
                        result[i] == 0.0,
                        "cur[{t}][{h}] should stay 0; got {}",
                        result[i]
                    );
                }
            }
        }
    }

    #[test]
    fn multi_token_add_at_distinct_positions_handles_each_independently() {
        let n_tokens = 5u32;
        let hidden = 2u32;
        // Three image tokens at non-contiguous positions.
        let positions = [0u32, 2, 4];
        // chunk[0] = [1,2], chunk[1] = [10,20], chunk[2] = [100,200].
        let chunk_vals = [1.0f32, 2.0, 10.0, 20.0, 100.0, 200.0];
        // cur starts at: row r, col c → r*10 + c (so we can spot
        // residual contamination loud).
        let cur_init = |i: usize| {
            let r = i / hidden as usize;
            let c = i % hidden as usize;
            (r * 10 + c) as f32
        };
        let result = dispatch_once(
            cur_init,
            |i| chunk_vals[i],
            &positions,
            n_tokens,
            hidden,
        );
        // row 0: cur 0,1 + chunk 1,2 → 1,3
        // row 1: untouched → 10,11
        // row 2: cur 20,21 + chunk 10,20 → 30,41
        // row 3: untouched → 30,31
        // row 4: cur 40,41 + chunk 100,200 → 140,241
        let expected: Vec<f32> = vec![
            1.0, 3.0,
            10.0, 11.0,
            30.0, 41.0,
            30.0, 31.0,
            140.0, 241.0,
        ];
        assert_eq!(result.len(), expected.len());
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "result[{i}] = {got} != expected {want}"
            );
        }
    }

    #[test]
    fn position_gated_non_image_rows_unchanged() {
        // Strong test: every image-token row gets a known shift; every
        // non-image row must equal cur_init exactly. Equivalent to the
        // peer's "non-image-token positions are zero in t_inp_embd" pin
        // for the gather-style kernel.
        let n_tokens = 16u32;
        let hidden = 8u32;
        let positions = [3u32, 7, 11];
        let cur_init = |i: usize| (i as f32).sin();
        let chunk_init = |i: usize| -> f32 {
            // unique positive shift per (k, h).
            ((i + 1) as f32) * 0.25
        };
        let result = dispatch_once(
            cur_init,
            chunk_init,
            &positions,
            n_tokens,
            hidden,
        );
        let h = hidden as usize;
        let pos_set: std::collections::HashSet<u32> = positions.iter().copied().collect();
        for t in 0..n_tokens as usize {
            for hh in 0..h {
                let i = t * h + hh;
                if pos_set.contains(&(t as u32)) {
                    // Image-token row: result = cur_init + chunk_at_k
                    let k = positions
                        .iter()
                        .position(|&p| p == t as u32)
                        .unwrap();
                    let chunk_idx = k * h + hh;
                    let expected = cur_init(i) + chunk_init(chunk_idx);
                    assert!(
                        (result[i] - expected).abs() < 1e-5,
                        "image-token row {t}[{hh}]: got {} expected {}",
                        result[i],
                        expected
                    );
                } else {
                    // Non-image-token row: must equal cur_init exactly.
                    let expected = cur_init(i);
                    assert!(
                        (result[i] - expected).abs() < 1e-7,
                        "non-image row {t}[{hh}] should be unchanged; got {} expected {}",
                        result[i],
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_mismatched_positions_length() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        register_image_token_residual_add_shader(&mut registry);
        let cur = alloc_f32(&device, 8, vec![4, 2], |_| 0.0);
        let chunk = alloc_f32(&device, 4, vec![2, 2], |_| 0.0);
        // n_image_tokens declared as 2 but positions is length 3.
        let positions = [0u32, 1, 2];
        let mut encoder = device.command_encoder().expect("encoder");
        let err = image_token_residual_add_gpu(
            &mut encoder,
            &mut registry,
            &device,
            &cur,
            &chunk,
            &positions,
            4, // n_tokens
            2, // n_image_tokens
            2, // hidden
        )
        .expect_err("length mismatch must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("image_token_positions.len()") && msg.contains("n_image_tokens"),
            "error must call out the length disagreement; got: {msg}"
        );
    }

    #[test]
    fn rejects_zero_n_image_tokens() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        register_image_token_residual_add_shader(&mut registry);
        let cur = alloc_f32(&device, 8, vec![4, 2], |_| 0.0);
        // Allocate a single-element placeholder chunk — Metal rejects a
        // zero-byte buffer at allocation time, but the dispatch
        // wrapper's n_image_tokens=0 guard fires before any buffer-size
        // validation. The chunk argument is never read at the
        // wrapper layer when n_image_tokens=0 (early return).
        let chunk = alloc_f32(&device, 1, vec![1], |_| 0.0);
        let positions: [u32; 0] = [];
        let mut encoder = device.command_encoder().expect("encoder");
        let err = image_token_residual_add_gpu(
            &mut encoder,
            &mut registry,
            &device,
            &cur,
            &chunk,
            &positions,
            4,
            0,
            2,
        )
        .expect_err("zero n_image_tokens must fail loud (caller should skip)");
        let msg = format!("{err}");
        assert!(
            msg.contains("n_image_tokens must be > 0"),
            "error message should ask caller to skip; got: {msg}"
        );
    }

    #[test]
    fn rejects_undersized_cur_buffer() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        register_image_token_residual_add_shader(&mut registry);
        // declared n_tokens=10 hidden=4 ⇒ cur needs 160 bytes; we pass
        // only 64 bytes.
        let cur = alloc_f32(&device, 16, vec![16], |_| 0.0);
        let chunk = alloc_f32(&device, 4, vec![1, 4], |_| 1.0);
        let positions = [3u32];
        let mut encoder = device.command_encoder().expect("encoder");
        let err = image_token_residual_add_gpu(
            &mut encoder,
            &mut registry,
            &device,
            &cur,
            &chunk,
            &positions,
            10,
            1,
            4,
        )
        .expect_err("undersized cur must fail loud");
        let msg = format!("{err}");
        assert!(
            msg.contains("cur span") && msg.contains("required"),
            "error must call out cur size mismatch; got: {msg}"
        );
    }

    #[test]
    fn out_of_bounds_position_silently_skips() {
        // Defense-in-depth in-kernel guard: a position >= n_tokens must
        // turn into a no-op (clamp) rather than a heap-overrun.
        // Caller should never pass such a value (handler-side
        // expand_image_placeholders constructs positions from the
        // post-expansion prompt and validates against n_tokens), but
        // we want the kernel to be robust if it does.
        let n_tokens = 4u32;
        let hidden = 3u32;
        // Position 99 is out-of-bounds; the second position 1 is valid.
        let positions = [99u32, 1];
        let chunk_vals = [7.0f32, 8.0, 9.0,  // chunk[0] — should be skipped
                          11.0, 12.0, 13.0]; // chunk[1] — applied at row 1
        let result = dispatch_once(
            |_| 0.0,
            |i| chunk_vals[i],
            &positions,
            n_tokens,
            hidden,
        );
        // Only row 1 should carry chunk[1]; rows 0,2,3 stay zero.
        for t in 0..n_tokens as usize {
            for h in 0..hidden as usize {
                let i = t * (hidden as usize) + h;
                if t == 1 {
                    assert!(
                        (result[i] - chunk_vals[3 + h]).abs() < 1e-6,
                        "row 1[{h}] = {} expected {}",
                        result[i],
                        chunk_vals[3 + h]
                    );
                } else {
                    assert_eq!(result[i], 0.0, "row {t}[{h}] should stay zero");
                }
            }
        }
    }
}
