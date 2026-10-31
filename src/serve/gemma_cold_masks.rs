//! Gemma 4 batched-prefill attention masks and their tile classifiers, with
//! every mask fill declared to the session's conflict tracker.
//!
//! `GraphSession` encodes into a concurrent Metal encoder and orders
//! dispatches only through what `barrier_between` / `track_dispatch` are
//! told. mlx-native's mask builders (`build_sdpa_mask_bf16`,
//! `build_sdpa_mask_f16`, `build_block_diagonal_sdpa_mask_bf16`) allocate
//! their output inside the call and fill it on the GPU, so the output cannot
//! be named before the dispatch. Left undeclared, a later
//! `barrier_between(&[&mask], ..)` finds no conflict and the reader (the F16
//! cast, the `blk` tile classifier) runs concurrently with the fill: it can
//! read a partly written mask, and the classifier can mark a tile that should
//! be fully masked (a future key, an out-of-window key, or another sequence's
//! keys in the block-diagonal mask) as attended. [`encode_tracked_mask`]
//! declares the fill right after it is encoded.
use anyhow::{anyhow, Result};
use mlx_native::ops::elementwise::{cast, CastDirection};
use mlx_native::ops::flash_attn_prefill_blk::{dispatch_flash_attn_prefill_blk, BlkParams};
use mlx_native::ops::flash_attn_prefill_mask::{
    build_block_diagonal_sdpa_mask_bf16, build_sdpa_mask_bf16, SdpaMaskParams,
};
use mlx_native::{CommandEncoder, DType, GraphSession, KernelRegistry, MlxBuffer, MlxDevice};

/// Encode a mask builder that allocates its output internally, then declare
/// its GPU write to the session's conflict tracker so the first
/// `barrier_between` naming the mask as a read emits a barrier.
///
/// `reads` are the buffers the builder reads (ordered against earlier writers
/// in the current concurrent group before the fill is encoded).
pub(crate) fn encode_tracked_mask(
    s: &mut GraphSession<'_>,
    reads: &[&MlxBuffer],
    build: impl FnOnce(&mut CommandEncoder) -> mlx_native::Result<MlxBuffer>,
) -> mlx_native::Result<MlxBuffer> {
    s.barrier_between(reads, &[]);
    let mask = build(s.encoder_mut())?;
    s.track_dispatch(&[], &[&mask]);
    Ok(mask)
}

/// Shape of a cold (non-resumed) batched-prefill mask.
pub(crate) enum ColdMaskLayout<'a> {
    /// One sequence: a `[len, len]` causal mask.
    Square { len: usize },
    /// Concatenated sequences: a `[len, len]` block-diagonal mask from each
    /// token's sequence index and per-sequence position.
    BlockDiagonal {
        seq_id: &'a MlxBuffer,
        local_pos: &'a MlxBuffer,
        len: usize,
    },
}

impl ColdMaskLayout<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Square { len } | Self::BlockDiagonal { len, .. } => *len,
        }
    }
}

/// One cold-prefill mask: the BF16 mask and, for the F16 flash-attention
/// route, its F16 copy.
pub(crate) struct ColdMask {
    pub(crate) bf16: MlxBuffer,
    pub(crate) f16: Option<MlxBuffer>,
}

/// Build a causal BF16 mask (sliding when `window` is set) and, when
/// `with_f16`, cast it to a rank-2 F16 mask (the F16 flash-attention
/// dispatcher detects the broadcast plane by rank 2). `name` labels errors.
pub(crate) fn encode_cold_mask(
    s: &mut GraphSession<'_>,
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    layout: ColdMaskLayout<'_>,
    window: Option<u32>,
    with_f16: bool,
    name: &str,
) -> Result<ColdMask> {
    let len = layout.len();
    let bf16 = match layout {
        ColdMaskLayout::Square { len } => encode_tracked_mask(s, &[], |encoder| {
            build_sdpa_mask_bf16(
                dev,
                reg,
                encoder,
                &SdpaMaskParams {
                    seq_len_q: len as u32,
                    seq_len_k: len as u32,
                    window_size: window,
                    causal: true,
                    q_abs_offset: 0,
                },
            )
        })
        .map_err(|e| anyhow!("build {name}: {e}"))?,
        ColdMaskLayout::BlockDiagonal {
            seq_id,
            local_pos,
            len,
        } => encode_tracked_mask(s, &[seq_id, local_pos], |encoder| {
            build_block_diagonal_sdpa_mask_bf16(
                dev, reg, encoder, seq_id, local_pos, len as u32, window, true,
            )
        })
        .map_err(|e| anyhow!("build multi-seq {name}: {e}"))?,
    };
    let f16 = if with_f16 {
        let elems = len * len;
        let f16 = dev
            .alloc_buffer(elems * DType::F16.size_of(), DType::F16, vec![len, len])
            .map_err(|e| anyhow!("alloc {name}_f16: {e}"))?;
        s.barrier_between(&[&bf16], &[&f16]);
        cast(
            s.encoder_mut(),
            reg,
            dev.metal_device(),
            &bf16,
            &f16,
            elems,
            CastDirection::BF16ToF16,
        )
        .map_err(|e| anyhow!("cast {name} BF16→F16: {e}"))?;
        Some(f16)
    } else {
        None
    };
    Ok(ColdMask { bf16, f16 })
}

/// Classify `mask` into the tile-skip bytes `blk` (one byte per
/// `(qtile, ktile)` at the main kernel's tile shape).
pub(crate) fn encode_mask_tiles(
    s: &mut GraphSession<'_>,
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    mask: &MlxBuffer,
    blk: &MlxBuffer,
    params: &BlkParams,
    name: &str,
) -> Result<()> {
    s.barrier_between(&[mask], &[blk]);
    dispatch_flash_attn_prefill_blk(s.encoder_mut(), dev, reg, mask, blk, params)
        .map_err(|e| anyhow!("dispatch {name}: {e}"))
}

#[cfg(test)]
#[path = "gemma_cold_masks_tests.rs"]
mod tests;
