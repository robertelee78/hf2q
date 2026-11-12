//! Tiled full attention of a joint DFlash2 verify step over the TQ cache
//! (`HF2Q_QWEN_VERIFY_ATTN_TILE=1`).
//!
//! A verify step holds, per lane, a window of up to eight rows at consecutive
//! positions of the lane's slot. The stock joint layer runs the serial chain
//! once per row behind barriers (K/V encode, Q rotation, split-K attention,
//! reduce, un-rotation, row copy): about 225 dispatches in 160 serialized
//! stages per layer at four lanes. This route encodes one layer as six
//! dispatches in four stages:
//!
//! 1. one batched TQ encode each for K and V of every row at its position
//!    (the serial encode's bytes),
//! 2. `mlx_native::ops::tq_hb_verify_tile`: one query rotation + pack, one
//!    split dispatch that computes a lane's rows and a KV head's query heads
//!    together over pages of the cache with Metal tensor matmuls, one reduce,
//!    and one fused un-rotation + sigmoid gate over all packed rows,
//!
//! the structure of Splash's paged verify attention on hf2q's unchanged
//! TQ-HB cache format. The cache bytes and cursors are those of the serial
//! chain; the attention output differs in bits (Class B): centroids enter the
//! matmuls as half and the online softmax runs per page and history split.
//! A row's output depends on its lane's cursor, its offset in the window, its
//! query and the cache before its causal end only, never on the other lanes,
//! the window's later rows, its packed row or the padding (mlx-native
//! `test_tq_hb_verify_tile`, `verify_attn_tile_tests`). The same token
//! verified at another offset of another window (another split partition)
//! may differ in its last bits.
//!
//! The kernel carries probability times value norm in half, which is exact
//! to half precision for stored value norms from about 1 to 1e6 (below 1,
//! tokens next to a dominant one start to drop out). The stored norms of the
//! served models are 2.0 to 168 (Qwen3.8-27B, 16 layers) and 2.8 to 36
//! (Qwen3.6-35B-A3B, 10 layers) over a four-agent coding run.

use std::cell::OnceCell;
use std::sync::LazyLock;

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::ops::tq_hb_verify_tile::{
    tq_hb_verify_tile_attention, tq_hb_verify_tile_available, verify_tile_scratch_bytes,
    verify_tile_slot_splits, TqHbVerifyLane, TqHbVerifyTileCache, TqHbVerifyTileScratch,
    TqHbVerifyTileShape, VERIFY_TILE_ROWS,
};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::decode_pool::pooled_alloc_buffer;
use super::joint_decode_core::JointWindow;
use super::kv_cache::FullAttnKvSlot;
use crate::serve::multi_seq_kv::SlotId;

/// Opt-in switch of the tiled DFlash2 verify attention; exact value `1`.
pub(crate) const QWEN_VERIFY_ATTN_TILE_ENV: &str = "HF2Q_QWEN_VERIFY_ATTN_TILE";

/// Whether the serving worker requests the tiled route (read once).
pub(crate) fn enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var(QWEN_VERIFY_ATTN_TILE_ENV)
            .map(|value| value == "1")
            .unwrap_or(false)
    });
    *ENABLED
}

/// The split scratch of one verify step, shared by its full-attention
/// layers: every layer's slots sit at the same cursors, so the first layer's
/// sizes serve them all, and the layers are ordered by the chain's barriers.
#[derive(Default)]
pub(crate) struct VerifyTileScratch {
    buffers: OnceCell<ScratchBuffers>,
}

struct ScratchBuffers {
    queries: MlxBuffer,
    partials: MlxBuffer,
    statistics: MlxBuffer,
    attention: MlxBuffer,
}

fn tile_shape(
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    width: usize,
) -> TqHbVerifyTileShape {
    TqHbVerifyTileShape {
        n_heads,
        n_kv_heads,
        head_dim,
        kv_capacity: max_seq_len,
        codebook_bits: crate::debug::INVESTIGATION_ENV.tq_codebook_bits,
        scale: 1.0_f32 / (head_dim as f32).sqrt(),
        width: width as u32,
    }
}

/// Whether [`apply_tq_verify_tile_gated_into`] serves this step of a
/// full-attention layer: a TQ-only cache with 8-bit codes, a compiled
/// geometry (head dim 256, six or eight query heads per KV head, capacity a
/// page multiple), every window of at most eight rows with room in its slot,
/// and the Metal tensor pipelines within this device's threadgroup limits.
/// Otherwise the caller runs the stock chain; the first refusal is logged.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tq_verify_tile_route(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    slot: &FullAttnKvSlot,
    windows: &[JointWindow],
    width: usize,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
) -> Result<bool> {
    let shape = tile_shape(n_heads, n_kv_heads, head_dim, max_seq_len, width);
    let refusal = if slot.tq.is_none() || slot.k.is_some() {
        Some("the cache is not TQ-only")
    } else if !shape.supported() {
        Some(
            "no kernel for this geometry (needs 8-bit codes, head dim 256, six or eight \
             query heads per KV head and a capacity that is a multiple of 32)",
        )
    } else if !windows.iter().all(|window| {
        (1..=VERIFY_TILE_ROWS as usize).contains(&window.rows)
            && slot
                .current_len
                .get(window.slot.0 as usize)
                .is_some_and(|&len| len as usize + window.rows <= max_seq_len as usize)
    }) {
        Some("a window has more than eight rows or no room in its slot")
    } else if !tq_hb_verify_tile_available(registry, device, &shape)
        .map_err(|e| anyhow!("probe tiled verify attention: {e}"))?
    {
        Some("the device lacks the Metal tensor API or the kernels' threadgroup limits")
    } else {
        None
    };
    if let Some(reason) = refusal {
        // The switch asked for the route; say once why a step keeps the
        // per-row chain.
        static FIRST_REFUSAL: std::sync::Once = std::sync::Once::new();
        FIRST_REFUSAL.call_once(|| {
            tracing::info!(
                reason,
                num_heads = n_heads,
                num_kv_heads = n_kv_heads,
                head_dim,
                codebook_bits = shape.codebook_bits,
                capacity = max_seq_len,
                "Qwen DFlash2 verify keeps the per-row attention chain: tiled TQ attention refused"
            );
        });
    }
    Ok(refusal.is_none())
}

/// Tiled TQ attention and output gate of one joint full-attention layer.
///
/// The rows are packed as for
/// [`super::gpu_full_attn::apply_tq_sdpa_rows_gated_into`]: `q_rows`
/// `[width, n_heads, head_dim]` (after IMROPE, unrotated; only read),
/// `k_rows` / `v_rows` `[width, n_kv_heads * head_dim]`, `gate_rows`
/// `[width, n_heads * head_dim]`. Window `w` holds packed rows `w.start..
/// w.start + w.rows` at positions `cursor..cursor + w.rows` of slot `w.slot`;
/// padding rows (past the windows) receive packed row 0's output. Advances
/// every window's cursor by its rows and returns the gated `[width, n_heads *
/// head_dim]` rows; the caller orders their reader with a barrier. Requires
/// [`tq_verify_tile_route`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_tq_verify_tile_gated_into(
    enc: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_rows: &MlxBuffer,
    k_rows: &MlxBuffer,
    v_rows: &MlxBuffer,
    gate_rows: &MlxBuffer,
    slot: &mut FullAttnKvSlot,
    windows: &[JointWindow],
    scratch: &VerifyTileScratch,
    width: usize,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
) -> Result<MlxBuffer> {
    ensure!(
        tq_verify_tile_route(
            registry, device, slot, windows, width, n_heads, n_kv_heads, head_dim, max_seq_len
        )?,
        "tiled verify attention needs a TQ-only 8-bit slot, a compiled geometry and windows of at most {VERIFY_TILE_ROWS} rows with room"
    );
    ensure!(
        windows.first().is_some_and(|window| window.start == 0),
        "tiled verify attention: the first window must start at packed row 0"
    );
    let shape = tile_shape(n_heads, n_kv_heads, head_dim, max_seq_len, width);
    let q_total = (n_heads * head_dim) as usize;
    let real: usize = windows.iter().map(|window| window.rows).sum();
    ensure!(
        q_rows.element_count() >= width * q_total && gate_rows.element_count() >= width * q_total,
        "tiled verify attention: Q/gate rows hold fewer than {width} rows of {q_total}"
    );

    // (1) K/V of every row at its position, packed row order.
    let mut writes: Vec<(SlotId, u32)> = Vec::with_capacity(real);
    let mut lanes = Vec::with_capacity(windows.len());
    for window in windows {
        ensure!(
            window.start == writes.len(),
            "tiled verify attention: windows must cover the real rows in packed order"
        );
        let cursor = slot.current_len[window.slot.0 as usize];
        writes.extend((0..window.rows as u32).map(|row| (window.slot, cursor + row)));
        lanes.push(TqHbVerifyLane {
            slot: window.slot.0,
            committed: cursor,
            rows: window.rows as u32,
            first_row: window.start as u32,
        });
    }
    slot.encode_rows_to_tq(
        k_rows,
        v_rows,
        &writes,
        n_kv_heads,
        head_dim,
        max_seq_len,
        shape.codebook_bits,
        enc,
        registry,
        device,
    )
    .context("tiled verify attention K/V encode")?;
    crate::inference::stage_profile::boundary(enc, "adr065.kv_write")
        .context("profile Qwen joint KV write")?;

    // (2) Every window's attention, un-rotated and gated, into the packed rows.
    let slot_splits = verify_tile_slot_splits(&lanes, max_seq_len);
    let sizes = verify_tile_scratch_bytes(&shape, lanes.len(), slot_splits);
    let buffers = match scratch.buffers.get() {
        Some(buffers) => buffers,
        None => {
            let alloc = |bytes: usize, dtype: DType, label: &str| {
                pooled_alloc_buffer(device, bytes, dtype, vec![bytes / dtype.size_of()])
                    .map_err(|e| anyhow!("alloc tiled verify attention {label}: {e}"))
            };
            let buffers = ScratchBuffers {
                queries: alloc(sizes.queries, DType::F16, "queries")?,
                partials: alloc(sizes.partials, DType::F32, "partials")?,
                statistics: alloc(sizes.statistics, DType::F32, "statistics")?,
                attention: alloc(sizes.attention, DType::F32, "rotated rows")?,
            };
            scratch.buffers.get_or_init(|| buffers)
        }
    };
    let gated = pooled_alloc_buffer(
        device,
        width * q_total * 4,
        DType::F32,
        vec![width * q_total],
    )
    .map_err(|e| anyhow!("alloc tiled verify gated attention: {e}"))?;
    let tq = slot
        .tq
        .as_ref()
        .ok_or_else(|| anyhow!("tiled verify attention: slot.tq is None"))?;
    tq_hb_verify_tile_attention(
        enc,
        registry,
        device,
        q_rows,
        gate_rows,
        &TqHbVerifyTileCache {
            k_packed: &tq.k_packed,
            k_norms: &tq.k_norms,
            v_packed: &tq.v_packed,
            v_norms: &tq.v_norms,
        },
        &gated,
        &TqHbVerifyTileScratch {
            queries: &buffers.queries,
            partials: &buffers.partials,
            statistics: &buffers.statistics,
            attention: &buffers.attention,
        },
        &shape,
        &lanes,
    )
    .map_err(|e| anyhow!("tiled verify attention: {e}"))?;

    for window in windows {
        slot.current_len[window.slot.0 as usize] += window.rows as u32;
    }
    static FIRST_TILE: std::sync::Once = std::sync::Once::new();
    FIRST_TILE.call_once(|| {
        tracing::info!(
            lanes = lanes.len(),
            rows = real,
            width,
            num_heads = n_heads,
            num_kv_heads = n_kv_heads,
            "Qwen DFlash2 verify selected tiled TQ attention"
        );
    });
    Ok(gated)
}
