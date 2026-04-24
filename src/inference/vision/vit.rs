//! ViT forward-pass primitives (ADR-005 Phase 2c, Task #15 begins).
//!
//! The mmproj-hosted vision tower is a standard CLIP/SigLIP-style ViT:
//!
//!   patch_embd   ─ Conv2d(in=3, out=hidden, k=patch, s=patch, p=0)
//!   + pos_embd   ─ [N_patches, hidden]
//!   blk[0..L] ── LN → QKV → attn → proj → LN → MLP(gelu) → residual
//!   post_ln      ─ final LayerNorm
//!   projector    ─ MLP: hidden → text_hidden
//!
//! This module ports each stage as a pure-f32 CPU function first, testable
//! against synthetic weights (identity kernels, delta inputs). The GPU
//! port follows once CPU parity is locked against an mlx-lm Gemma 4
//! vision reference (live-model-gated).
//!
//! # Tensor-layout conventions (match llama.cpp's mmproj writer)
//!
//!   - `pixel_values`  — CHW, `[3, H, W]` f32, row-major within each channel,
//!     channels concatenated. Produced by `preprocess::preprocess_rgb_chw`.
//!   - `patch_embd_weight`  — `[hidden, 3, patch, patch]` in
//!     out-channel-major layout. Element index:
//!       `oc*(3*p*p) + ic*(p*p) + dy*p + dx`.
//!   - `patch_embd_bias`    — `[hidden]`, optional (None when absent).
//!   - `patch_embeddings`   — `[N_patches, hidden]` row-major, with patches
//!     enumerated row-by-row top-to-bottom (patch index =
//!     `py*num_patches_side + px`).
//!
//! # Not done in this iter
//!
//!   - GGUF weight loader (iter 28) — reads tensors out of the mmproj
//!     file and wraps them in typed slices. This module consumes `&[f32]`
//!     so it's decoupled from the loader.
//!   - Position embeddings, LN, attention, MLP — iter 29+.
//!   - GPU dispatch — deferred until CPU parity is locked.

#![allow(dead_code)]

use anyhow::{anyhow, Result};

/// Apply the patch-embedding Conv2d to a CHW pixel tensor.
///
/// Mathematically equivalent to:
///
/// ```text
/// for py in 0..num_patches_side:
///   for px in 0..num_patches_side:
///     for oc in 0..hidden:
///       acc = bias[oc] if bias is Some else 0
///       for ic in 0..3:
///         for dy in 0..patch_size:
///           for dx in 0..patch_size:
///             y = py*patch_size + dy
///             x = px*patch_size + dx
///             acc += pixels[ic*H*W + y*W + x] * weight[oc*3*p*p + ic*p*p + dy*p + dx]
///       out[(py*num_patches_side + px)*hidden + oc] = acc
/// ```
///
/// Returned `Vec<f32>` has length `num_patches² × hidden`.
///
/// # Errors
///
/// - `pixel_values.len() != 3 * image_size²`
/// - `patch_embd_weight.len() != hidden * 3 * patch_size²`
/// - `patch_embd_bias.map(|b| b.len() != hidden).unwrap_or(false)`
/// - `image_size % patch_size != 0`
///
/// Algorithm cost: O(N_patches × hidden × 3 × patch² ) — for Gemma 4
/// (64×64 patches × 1152 hidden × 3 × 196) ~= 2.7 GFLOP per image. This
/// is the CPU-correctness reference; the GPU port is where performance
/// lives.
pub fn patch_embed_forward(
    pixel_values: &[f32],
    patch_embd_weight: &[f32],
    patch_embd_bias: Option<&[f32]>,
    image_size: u32,
    patch_size: u32,
    hidden: u32,
) -> Result<Vec<f32>> {
    // --- Shape validation ---
    if patch_size == 0 {
        return Err(anyhow!("patch_size must be > 0"));
    }
    if image_size % patch_size != 0 {
        return Err(anyhow!(
            "image_size ({}) must be divisible by patch_size ({})",
            image_size,
            patch_size
        ));
    }
    let num_patches_side = image_size / patch_size;
    let p = patch_size as usize;
    let h = hidden as usize;
    let hw = (image_size as usize) * (image_size as usize);
    let expected_pixels = 3 * hw;
    if pixel_values.len() != expected_pixels {
        return Err(anyhow!(
            "pixel_values len {} != expected 3*{}*{} = {}",
            pixel_values.len(),
            image_size,
            image_size,
            expected_pixels
        ));
    }
    let expected_w = h * 3 * p * p;
    if patch_embd_weight.len() != expected_w {
        return Err(anyhow!(
            "patch_embd_weight len {} != expected {}*3*{}*{} = {}",
            patch_embd_weight.len(),
            hidden,
            patch_size,
            patch_size,
            expected_w
        ));
    }
    if let Some(b) = patch_embd_bias {
        if b.len() != h {
            return Err(anyhow!(
                "patch_embd_bias len {} != hidden {}",
                b.len(),
                hidden
            ));
        }
    }

    let w = image_size as usize;
    let nps = num_patches_side as usize;
    let num_patches = nps * nps;
    let mut out = vec![0f32; num_patches * h];

    // Stride constants for weight indexing.
    let ws_oc = 3 * p * p; // stride between out-channels
    let ws_ic = p * p; //     stride between in-channels within one out-channel
    let ws_y = p; //          stride between rows within one in-channel slice

    // Stride constants for pixel_values (CHW).
    let ps_c = hw; // stride between channels
    let ps_y = w; //  stride between rows within one channel

    for py in 0..nps {
        let y0 = py * p;
        for px in 0..nps {
            let x0 = px * p;
            let patch_idx = py * nps + px;
            let out_base = patch_idx * h;
            for oc in 0..h {
                let mut acc: f32 = match patch_embd_bias {
                    Some(b) => b[oc],
                    None => 0.0,
                };
                let w_base = oc * ws_oc;
                for ic in 0..3usize {
                    let w_ic_base = w_base + ic * ws_ic;
                    let p_ic_base = ic * ps_c;
                    for dy in 0..p {
                        let w_row_base = w_ic_base + dy * ws_y;
                        let p_row_base = p_ic_base + (y0 + dy) * ps_y + x0;
                        for dx in 0..p {
                            acc += pixel_values[p_row_base + dx] * patch_embd_weight[w_row_base + dx];
                        }
                    }
                }
                out[out_base + oc] = acc;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a patch_embd weight tensor where `weight[oc, ic, dy, dx]`
    /// equals the supplied closure result. Layout matches
    /// `patch_embed_forward`'s expected shape.
    fn make_weight<F: FnMut(usize, usize, usize, usize) -> f32>(
        hidden: usize,
        patch_size: usize,
        mut f: F,
    ) -> Vec<f32> {
        let mut w = vec![0f32; hidden * 3 * patch_size * patch_size];
        for oc in 0..hidden {
            for ic in 0..3 {
                for dy in 0..patch_size {
                    for dx in 0..patch_size {
                        let idx = oc * 3 * patch_size * patch_size
                            + ic * patch_size * patch_size
                            + dy * patch_size
                            + dx;
                        w[idx] = f(oc, ic, dy, dx);
                    }
                }
            }
        }
        w
    }

    /// Build a CHW pixel tensor from a closure `pixel(c, y, x) -> f32`.
    fn make_pixels<F: FnMut(usize, usize, usize) -> f32>(
        image_size: usize,
        mut f: F,
    ) -> Vec<f32> {
        let mut p = vec![0f32; 3 * image_size * image_size];
        for c in 0..3 {
            for y in 0..image_size {
                for x in 0..image_size {
                    let idx = c * image_size * image_size + y * image_size + x;
                    p[idx] = f(c, y, x);
                }
            }
        }
        p
    }

    #[test]
    fn delta_kernel_copies_top_left_pixel_per_patch() {
        // hidden=3. weight[oc, ic, dy, dx] = 1 iff (oc == ic && dy == 0 && dx == 0), else 0.
        // No bias. Image 8×8, patch 4 → 2×2 patches. Each patch output
        // should equal the top-left pixel of that patch in the matching
        // channel.
        let img = 8;
        let patch = 4;
        let hidden = 3;
        let weight = make_weight(hidden, patch, |oc, ic, dy, dx| {
            if oc == ic && dy == 0 && dx == 0 {
                1.0
            } else {
                0.0
            }
        });
        // Put a unique value at each pixel so we can assert exactly.
        let pixels = make_pixels(img, |c, y, x| (c * 100 + y * 10 + x) as f32);
        let out = patch_embed_forward(
            &pixels,
            &weight,
            None,
            img as u32,
            patch as u32,
            hidden as u32,
        )
        .unwrap();
        assert_eq!(out.len(), 2 * 2 * hidden);
        // Patch (0,0) — top-left of each channel at (0,0).
        assert_eq!(out[0 * hidden + 0], 0.0); // c=0 pixel (0,0) = 0
        assert_eq!(out[0 * hidden + 1], 100.0); // c=1 pixel (0,0) = 100
        assert_eq!(out[0 * hidden + 2], 200.0); // c=2 pixel (0,0) = 200
        // Patch (0,1) — top-left at (0,4).
        assert_eq!(out[1 * hidden + 0], 4.0); // c=0 pixel (0,4) = 4
        assert_eq!(out[1 * hidden + 1], 104.0);
        assert_eq!(out[1 * hidden + 2], 204.0);
        // Patch (1,0) — top-left at (4,0).
        assert_eq!(out[2 * hidden + 0], 40.0);
        assert_eq!(out[2 * hidden + 1], 140.0);
        assert_eq!(out[2 * hidden + 2], 240.0);
        // Patch (1,1) — top-left at (4,4).
        assert_eq!(out[3 * hidden + 0], 44.0);
        assert_eq!(out[3 * hidden + 1], 144.0);
        assert_eq!(out[3 * hidden + 2], 244.0);
    }

    #[test]
    fn all_ones_kernel_produces_patch_sum() {
        // hidden=1. All weights = 1 → output = sum of all pixels in patch
        // window across all 3 channels.
        let img: usize = 4;
        let patch: usize = 2;
        let hidden: usize = 1;
        let weight = vec![1f32; hidden * 3 * patch * patch];
        // Each channel is uniform: c=0 → 1, c=1 → 2, c=2 → 3. One 2×2
        // patch sums 4 pixels per channel × 3 channels = 4*(1+2+3) = 24.
        let pixels = make_pixels(img, |c, _y, _x| (c as f32) + 1.0);
        let out = patch_embed_forward(
            &pixels,
            &weight,
            None,
            img as u32,
            patch as u32,
            hidden as u32,
        )
        .unwrap();
        assert_eq!(out.len(), 4 * 1); // 2×2 patches × 1 hidden
        for (i, v) in out.iter().enumerate() {
            assert!(
                (*v - 24.0).abs() < 1e-5,
                "patch {}: expected 24, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn bias_is_added_once_per_output_element() {
        // hidden=2, all weights = 0, bias = [10, 20]. Output should be
        // bias repeated for every patch.
        let img: usize = 4;
        let patch: usize = 2;
        let hidden: usize = 2;
        let weight = vec![0f32; hidden * 3 * patch * patch];
        let bias = vec![10f32, 20.0];
        let pixels = vec![999f32; 3 * img * img]; // garbage input, weights zero them
        let out = patch_embed_forward(
            &pixels,
            &weight,
            Some(&bias),
            img as u32,
            patch as u32,
            hidden as u32,
        )
        .unwrap();
        assert_eq!(out.len(), 4 * 2);
        for i in 0..4 {
            assert!((out[i * 2 + 0] - 10.0).abs() < 1e-5);
            assert!((out[i * 2 + 1] - 20.0).abs() < 1e-5);
        }
    }

    #[test]
    fn single_patch_covers_whole_image() {
        // image_size == patch_size → 1 patch covering everything.
        let img: usize = 4;
        let patch: usize = 4;
        let hidden: usize = 1;
        // Weight: all ones on channel 0 top-left pixel only.
        let weight = make_weight(hidden, patch, |_oc, ic, dy, dx| {
            if ic == 0 && dy == 0 && dx == 0 {
                1.0
            } else {
                0.0
            }
        });
        let pixels = make_pixels(img, |c, _, _| (c + 1) as f32);
        let out = patch_embed_forward(
            &pixels,
            &weight,
            None,
            img as u32,
            patch as u32,
            hidden as u32,
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], 1.0); // c=0 value everywhere = 1
    }

    #[test]
    fn rejects_mismatched_pixel_len() {
        let weight = vec![0f32; 1 * 3 * 2 * 2];
        let pixels = vec![0f32; 10]; // wrong size
        let err = patch_embed_forward(&pixels, &weight, None, 4, 2, 1).unwrap_err();
        assert!(format!("{err}").contains("pixel_values len"));
    }

    #[test]
    fn rejects_mismatched_weight_len() {
        let pixels = vec![0f32; 3 * 4 * 4];
        let weight = vec![0f32; 5]; // wrong size
        let err = patch_embed_forward(&pixels, &weight, None, 4, 2, 1).unwrap_err();
        assert!(format!("{err}").contains("patch_embd_weight len"));
    }

    #[test]
    fn rejects_bias_length_mismatch() {
        let weight = vec![0f32; 2 * 3 * 2 * 2];
        let pixels = vec![0f32; 3 * 4 * 4];
        let bias = vec![0f32; 99];
        let err = patch_embed_forward(&pixels, &weight, Some(&bias), 4, 2, 2).unwrap_err();
        assert!(format!("{err}").contains("patch_embd_bias len"));
    }

    #[test]
    fn rejects_non_divisible_image_size() {
        let weight = vec![0f32; 1 * 3 * 3 * 3];
        let pixels = vec![0f32; 3 * 5 * 5];
        let err = patch_embed_forward(&pixels, &weight, None, 5, 3, 1).unwrap_err();
        assert!(format!("{err}").contains("divisible"));
    }

    #[test]
    fn rejects_zero_patch_size() {
        let weight = vec![0f32; 0];
        let pixels = vec![0f32; 3 * 4 * 4];
        let err = patch_embed_forward(&pixels, &weight, None, 4, 0, 1).unwrap_err();
        assert!(format!("{err}").contains("patch_size"));
    }

    #[test]
    fn gemma4_shape_does_not_panic() {
        // Smoke: exercise the function at the Gemma 4 shape to ensure
        // index arithmetic is sound at the production dims. 896 is too
        // slow for a unit test; scale the image + hidden to sqrt(Gemma)
        // while keeping the patch shape identical (14×14). Target: image
        // 56 = 4 patches per side = 16 patches total; hidden 32 (was 1152).
        let img: usize = 56;
        let patch: usize = 14;
        let hidden: usize = 32;
        let weight = vec![0.01f32; hidden * 3 * patch * patch];
        let bias = vec![0.1f32; hidden];
        let pixels = vec![0.5f32; 3 * img * img];
        let out = patch_embed_forward(
            &pixels,
            &weight,
            Some(&bias),
            img as u32,
            patch as u32,
            hidden as u32,
        )
        .unwrap();
        let num_patches = (img / patch) * (img / patch);
        assert_eq!(out.len(), num_patches * hidden);
        // Sanity: each output element = 0.5 * 0.01 * 3 * 14 * 14 + 0.1
        //                              = 0.5 * 0.01 * 588 + 0.1 = 2.94 + 0.1 = 3.04
        for v in &out {
            assert!((*v - 3.04).abs() < 1e-3, "expected 3.04, got {v}");
        }
    }
}
