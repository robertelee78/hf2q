//! Requested-byte closure for the prepared prompt-prefill forward.
//!
//! Roots, not views or command references, are counted. This is an admission
//! bound, not Metal allocatedSize, driver residency or a measured footprint.
use super::{prefill_prepared, Qwen35Config, Qwen35LayerKind};

/// Capabilities of the route that can actually run, not just a profile name.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefillMemoryRoute {
    pub(crate) joint: bool,
    pub(crate) wait_layers: usize,
    pub(crate) short_wait_layers: usize,
    pub(crate) prepared: bool,
    pub(crate) tq_only: bool,
    pub(crate) resume_fast: bool,
}

/// Disjoint requested-byte owners. Prepared weights (including their retained
/// split partials and fused source copies) are charged by model admission.
/// Normalized nextn parents and retained carry roots are excluded here;
/// carry_memory composes their actual target-forward/catch-up overlap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PrefillMemoryBound {
    pub(crate) pool: u64,
    pub(crate) attention: u64,
    pub(crate) history: u64,
    pub(crate) segments: u64,
    pub(crate) ffn: u64,
    pub(crate) delta_states: u64,
    pub(crate) common: u64,
    pub(crate) output_head: u64,
    pub(crate) deferred_outputs: u64,
    pub(crate) mm_id_tls: u64,
    pub(crate) allocation_allowance: u64,
}

impl PrefillMemoryBound {
    pub(crate) fn total(self) -> u64 {
        [self.pool, self.attention, self.history, self.segments, self.ffn,
            self.delta_states, self.common, self.deferred_outputs,
            self.mm_id_tls, self.output_head, self.allocation_allowance]
            .into_iter().fold(0, u64::saturating_add)
    }
}

/// One worker can execute an ordinary single-slot chunk, a single-slot cue
/// fold, or an aggregate joint transaction. Their forward-local roots do not
/// coexist: take the largest complete lifetime, never a componentwise sum or
/// an invented `(content_rows + cue_rows) * slots` joint transaction.
///
/// `content_rows` is this WORKER's scheduler quantum, not the quantum of a
/// separate one-slot worker. plan_chunk may append MAX_CUE_ROWS to its solo
/// content chunk; the joint scheduler charges the cue inside its aggregate
/// quantum. forward_gpu_last_logits_folded executes the fold as a one-lane
/// joint plan, so it has shared attention roots rather than the ordinary
/// single-slot dispatch's plain results.
pub(crate) fn envelope(
    cfg: &Qwen35Config,
    content_rows: usize,
    slots: usize,
    kv_rows: usize,
    route: PrefillMemoryRoute,
    fold_cue: bool,
) -> Result<PrefillMemoryBound, &'static str> {
    if !route.joint && !fold_cue && route.wait_layers == 1 {
        return bound(cfg, content_rows, slots, kv_rows, route);
    }
    if slots == 0 || slots > u32::MAX as usize {
        return Err("invalid prefill worker slot capacity");
    }
    // The ordinary solo fallback remains legal even when joint/fold is on.
    // At wait=1 it runs without the capped pipelined pool. Its unbounded
    // retained free set has no closure under the bounded-pool formula.
    if route.wait_layers == 1 {
        return Err("ordinary solo wait-one fallback has no bounded retained-pool closure");
    }
    let folded_rows = if fold_cue {
        Some(content_rows.checked_add(super::prefill_fold::MAX_CUE_ROWS)
            .ok_or("folded solo row capacity overflow")?)
    } else {
        None
    };
    let retained_tls = mm_id_tls_bound_bytes(cfg, folded_rows.unwrap_or(content_rows))?;
    let mut peak = bound(cfg, content_rows, 1, kv_rows,
        PrefillMemoryRoute { joint: false, ..route })?;
    // TLS survives a completed solo forward into a later joint call. Each
    // candidate must retain that legal high-water, not just its current rows.
    peak.mm_id_tls = retained_tls;
    if route.joint {
        let mut joint = bound(cfg, content_rows, slots, kv_rows, route)?;
        joint.mm_id_tls = retained_tls;
        if joint.total() > peak.total() {
            peak = joint;
        }
    }
    if let Some(rows) = folded_rows {
        let mut folded = bound(cfg, rows, 1, kv_rows,
            PrefillMemoryRoute { joint: true, ..route })?;
        folded.mm_id_tls = retained_tls;
        if folded.total() > peak.total() {
            peak = folded;
        }
    }
    if peak.total() == u64::MAX {
        return Err("prefill route envelope byte overflow");
    }
    Ok(peak)
}

/// Complete bound while either bounded joint execution or deferred waits is
/// active. Invalid/unclosed routes fail closed; callers must charge MAX, never
/// zero, on error. The untouched ordinary route has no incremental commitment.
pub(crate) fn bound(
    cfg: &Qwen35Config,
    forward_rows: usize,
    slots: usize,
    kv_rows: usize,
    route: PrefillMemoryRoute,
) -> Result<PrefillMemoryBound, &'static str> {
    if !route.joint && route.wait_layers == 1 {
        return Ok(PrefillMemoryBound::default());
    }
    if !route.prepared {
        return Err("the bounded prefill pool requires prepared projection products");
    }
    if forward_rows == 0 || slots == 0 || kv_rows == 0 || route.short_wait_layers == 0
        || route.wait_layers == 0 || route.wait_layers > prefill_prepared::MAX_WAIT_LAYERS
    {
        return Err("invalid prefill admission geometry or wait cadence");
    }
    if forward_rows > u32::MAX as usize || slots > u32::MAX as usize
        || kv_rows > u32::MAX as usize
    {
        return Err("prefill capacities exceed native u32 indexing");
    }
    // Non-D256 falls through CPU permutations/legacy SDPA. No closed bound
    // for that route is supplied by the D256 prepared bridge proof.
    if cfg.head_dim != 256 {
        return Err("legacy non-D256 attention has no closed prepared-prefill bound");
    }
    let r = forward_rows as u128;
    let lanes = if route.joint { slots as u128 } else { 1 };
    let h = u128::from(cfg.hidden_size);
    let q = u128::from(cfg.num_attention_heads) * u128::from(cfg.head_dim);
    let kv = u128::from(cfg.num_key_value_heads) * u128::from(cfg.head_dim);
    let nk = u128::from(cfg.linear_num_key_heads);
    let nv = u128::from(cfg.linear_num_value_heads);
    let dk = u128::from(cfg.linear_key_head_dim);
    let dv = u128::from(cfg.linear_value_head_dim);
    let a = nk * dk;
    let z = nv * dv;
    let channels = 2 * a + z;
    if h == 0 || q == 0 || kv == 0 || nv == 0 || dk == 0 || dv == 0
        || cfg.moe.as_ref().is_some_and(|moe| {
            moe.num_experts == 0 || moe.num_experts_per_tok == 0
        })
    {
        return Err("zero prefill allocation dimension");
    }
    if [r * h, r * q, r * kv, r * nv * dk, r * nv * dv,
        kv * kv_rows as u128, dk * dv * nv, channels]
        .into_iter().any(|elements| elements > u32::MAX as u128)
    {
        return Err("prefill allocation shape exceeds native u32 indexing");
    }
    let full = cfg.layer_types.contains(&Qwen35LayerKind::FullAttention);
    let linear = cfg.layer_types.contains(&Qwen35LayerKind::LinearAttention);
    let bytes = |n| u64::try_from(n).map_err(|_| "prefill allocation byte overflow");
    let full_layers = cfg.layer_types.iter()
        .filter(|kind| **kind == Qwen35LayerKind::FullAttention).count() as u128;
    let layers = full_layers.min(route.wait_layers.max(route.short_wait_layers) as u128);

    // forward_gpu's cap/reset protocol: one retained free set, at most one
    // outstanding threshold before the crossing layer, and that layer's
    // bucketed requests. These are one pool, not one pool per switch.
    let shared_result = route.joint && route.tq_only && route.resume_fast;
    let layer = prefill_prepared::prefill_layer_scratch_bound_bytes(
        cfg, forward_rows, lanes as usize, shared_result,
    );
    let mut result = PrefillMemoryBound {
        pool: (prefill_prepared::PREFILL_RETAINED_SCRATCH_BYTES as u64)
            .saturating_add(if route.wait_layers > 1 {
                prefill_prepared::WAIT_LAYERS_WINDOW_SCRATCH_BYTES as u64
            } else { 0 })
            .saturating_add(layer),
        ..PrefillMemoryBound::default()
    };
    if route.wait_layers == 1 {
        // <=32-row forwards retain HF2Q_FFN_TERMINAL_K_BATCH's cadence;
        // without pipelining their issued bytes do not trigger the window
        // threshold. Charge that whole possible short window, not one layer.
        let short_layer = prefill_prepared::prefill_layer_scratch_bound_bytes(
            cfg, forward_rows.min(32), lanes as usize, shared_result,
        );
        let short_window = u128::from(short_layer)
            * (route.short_wait_layers.min(cfg.num_hidden_layers as usize) as u128);
        result.pool = result.pool.max(
            (prefill_prepared::PREFILL_RETAINED_SCRATCH_BYTES as u64)
                .saturating_add(bytes(short_window)?),
        );
    }
    if full {
        // FaProjectionsArena's ten F32 arrays; FaPrefillArena's seven BF16
        // arrays; one F32 attention result. Joint segment tight views share
        // these roots. The single-lane route owns the same sized set.
        result.attention = attention_arena_bytes(cfg, forward_rows)
            .ok_or("attention arena byte overflow")?;
        let shared_history = route.tq_only && route.resume_fast;
        result.history = if shared_history {
            // ResumeTqWorkspace: ONE F32 staging, BF16 K and V. The F32
            // output aliases the attention result for joint execution.
            bytes(8 * kv * kv_rows as u128)?
        } else {
            // Disabled fast bridge: tight F32 K/V plus BF16 K/V. F32-cache
            // resume needs only the BF16 pair, but may use the full capacity
            // stride. Charging the larger legacy TQ bridge covers both.
            // Include its three query-sized BF16 temporaries + F32 output.
            bytes(lanes * (12 * kv * kv_rows as u128 + 10 * r * q))?
        };
        // ResumeTqScratch's private result on non-joint execution is distinct
        // from cold-prefill results. Cold/non-TQ/disabled-fast helpers can
        // still return plain per-layer outputs held through the checked wait.
        // Joint TQ fast/cold paths use the shared result, not one per layer.
        result.deferred_outputs = if shared_result {
            // Direct short TQ attention allocates a plain F32 result, held
            // across deferred layers. At most two <=32-row segments/lane.
            bytes(4 * q * (32 * 2 * lanes) * layers)?
        } else {
            bytes(4 * q * r * lanes * layers)?
        };
        // segment_positions: four U32s/row; gathered full-attention inputs
        // survive the forward. Shared-prefix duplication is <= R per lane.
        result.segments = bytes(if route.joint { r * lanes * (16 + 4 * h) } else { 16 * r })?;
    }
    // Dense/MoE arenas survive the whole forward. The row-exact MoE call
    // covers R rows; at most two short segments/lane add <=32 rows each.
    result.ffn = ffn_arena_bytes(cfg, forward_rows, if route.joint { 2 * slots } else { 0 })
        .ok_or("FFN arena byte overflow")?;
    if linear {
        // DnPrefillArena::new's 16 row arrays and seven parameter roots.
        let dn = 4 * r * (2 * channels + 5 * a + 4 * z + h + 4 * nv) + 92;
        // ChunkAllocsArena::new's seven arrays are eagerly allocated even
        // when this exact transaction does not engage the chunk kernel.
        let wrapper = r * (12 * nv * dk + 4 * nv * dv + 4 * nv);
        // Native constructor's authoritative thirteen-root plan. Any chunk
        // transaction <=R has at most floor(R/64)*64 rows. Unsupported
        // dimensions do not engage the chunk path; no invented arena bytes.
        let chunk_rows = forward_rows / 64 * 64;
        let native = if chunk_rows > 0 && cfg.linear_key_head_dim == 128
            && cfg.linear_value_head_dim <= 128
        {
            let t = u32::try_from(chunk_rows).map_err(|_| "chunk row capacity exceeds u32")?;
            mlx_native::ops::chunk_gated_delta_rule::ChunkInternalArenaPlan::for_shape(
                1, t, cfg.linear_num_value_heads, cfg.linear_num_value_heads,
                cfg.linear_key_head_dim, cfg.linear_value_head_dim, 64,
            ).map_err(|_| "native chunk arena allocation plan rejected geometry")?.requested_bytes() as u128
        } else { 0 };
        result.common = bytes(dn + wrapper + native)?;
        if route.joint {
            // DnRunScratch: one prefix recurrent state, prefix convolution
            // state and one private boundary convolution state per lane.
            let conv = 4 * channels * u128::from(cfg.linear_conv_kernel_dim.saturating_sub(1));
            result.delta_states = bytes((1 + lanes) * conv + 4 * dk * dv * nv)?;
        }
    }
    // LayerBoundaryArena input/residual pair + two-slot FFN output ring.
    // hidden_holds only clones the ring/embedding roots, not new allocations.
    result.common = result.common.saturating_add(bytes(16 * r * h)?);
    // Serving prefill projects only final/boundary rows, not [R,vocab].
    // LaneLogits retains at most two CPU vectors/lane while one plain device
    // logits buffer is downloaded. Include one CPU embedding staging copy.
    // The durable normalized nextn parent is excluded here. CatchupPlan must
    // charge it during BOTH target-forward and catch-up lifetimes.
    result.output_head = bytes(4 * u128::from(cfg.vocab_size) * (2 * lanes + 1) + 4 * r * h)?;
    result.mm_id_tls = mm_id_tls_bound_bytes(cfg, forward_rows)?;
    // Existing allowance for parameter roots/page rounding. Not a claim that
    // allocator metadata, pipeline state or driver memory is measured here.
    result.allocation_allowance = bytes(lanes * 64 * 16 * 1024)?;
    if result.total() == u64::MAX {
        return Err("prefill allocation sum overflow");
    }
    Ok(result)
}

fn mm_id_tls_bound_bytes(cfg: &Qwen35Config, row_high_water: usize) -> Result<u64, &'static str> {
    let Some(moe) = cfg.moe.as_ref() else { return Ok(0) };
    if row_high_water == 0 || row_high_water > u32::MAX as usize
        || moe.num_experts == 0 || moe.num_experts_per_tok == 0
    {
        return Err("invalid MM-ID TLS high-water geometry");
    }
    let rows = row_high_water as u128;
    // Only <=MOE_BATCHED_MAX_ROWS calls can use the live-tile shared-expert
    // schedule. Its expert ids are blocks of DENSE_MATVEC_F32_ROWS_MAX_ROWS
    // rows (moe_live_tiles.rs), not blocks of the entire 2080-row prefill.
    let shared_rows = row_high_water.min(super::moe_batched::MOE_BATCHED_MAX_ROWS);
    let shared_experts = shared_rows.div_ceil(
        mlx_native::ops::dense_matvec_f32_rows::DENSE_MATVEC_F32_ROWS_MAX_ROWS as usize,
    );
    let experts = u128::from(moe.num_experts).max(shared_experts as u128);
    // Actual callers pass R*model_top_k; IdMmScratch::alloc multiplies THAT
    // by eight again, independently of model_top_k.
    let tokens = rows * u128::from(moe.num_experts_per_tok);
    if tokens > u32::MAX as u128 || tokens * 8 > i32::MAX as u128 {
        return Err("MM-ID TLS capacity exceeds signed native row indexing");
    }
    // Gate/Up/Down can retain prior constructor capacities independently.
    // `*guard = Some(alloc(...))` evaluates its RHS before dropping the old
    // slot: a fourth slot-sized allocation can overlap the three old owners.
    u64::try_from(4 * id_mm_slot_bytes(experts, tokens))
        .map_err(|_| "MM-ID TLS high-water byte overflow")
}

fn id_mm_slot_bytes(experts: u128, tokens: u128) -> u128 {
    // IdMmScratch::alloc_with_top_k(..., 8): max each buffer independently
    // across UniformGrid and CompactTiles, not max of the two totals.
    let routed = tokens * 8;
    let tiles = routed.min(routed.div_ceil(32) + experts.min(routed) - 1);
    let htpe = (4 * experts).max(4 * (1 + 3 * tiles));
    let hids = (4 * experts * tokens).max(4 * (routed + experts));
    htpe + hids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tls_sizes_both_native_schedules_and_constructor_top_k() {
        // R=2048, model top_k=8: native constructor gets 16384 tokens,
        // not 2048; UniformGrid's list dominates the compact list.
        assert_eq!(id_mm_slot_bytes(256, 16_384), 16_829_432);
        assert!(id_mm_slot_bytes(256, 16_384) > id_mm_slot_bytes(256, 2_048));
        assert_eq!(id_mm_slot_bytes(1, 1), 52);
    }
}

/// All valid <=32-row MoE FFN routes, including the live-tile fallback and
/// staged gather. The long call uses its forward-owned arena; short calls
/// use pooled roots even when a segment also owns an unused arena.
pub(crate) fn short_moe_scratch_bytes(cfg: &Qwen35Config) -> u64 {
    let Some(moe) = cfg.moe.as_ref() else { return 0 };
    let (r, k, h, m, ms, ne) = (
        32u128, u128::from(moe.num_experts_per_tok), u128::from(cfg.hidden_size),
        u128::from(moe.moe_intermediate_size),
        u128::from(moe.shared_expert_intermediate_size), u128::from(moe.num_experts),
    );
    let Ok(grouped) = mlx_native::grouped_m8_gathered_rows(
        32, moe.num_experts_per_tok, moe.num_experts,
    ) else { return u64::MAX };
    let Ok(shared) = mlx_native::moe_tiles_grouped_rows(32, 1, 4)
        else { return u64::MAX };
    let grouped = grouped as u128;
    let batched = 4 * r * (ne + 1 + 3 * ms + h) + 8 * r * k
        + 12 * r * k * m + 4 * r * k * h
        + 2 * grouped * h.max(m) + 12;
    let tiles = 4 * r * (ne + 1 + h) + 8 * r * k + 4 * r * k * h
        + 2 * grouped * (h + m) + 2 * shared as u128 * (h + ms)
        + 4 * r + 8 * mlx_native::MOE_TILES_GRID_WORDS as u128 + 4;
    let params = mlx_native::MoeTilesFusedParams {
        n_tokens: 32, top_k: moe.num_experts_per_tok,
        n_experts: moe.num_experts, hidden: cfg.hidden_size,
        intermediate: moe.moe_intermediate_size, column_tile: false,
    };
    let fused = match (
        mlx_native::moe_tiles_fused_grouped_rows(&params),
        mlx_native::moe_tiles_fused_schedule_words(&params),
        mlx_native::moe_tiles_fused_output_rows(&params),
    ) {
        (Ok(rows), Ok(schedule), Ok(output)) => {
            4 * r * (ne + 1) + 8 * r * k + 4 * output as u128 * h
                + 4 * (schedule as u128 + rows as u128
                    + mlx_native::MOE_TILES_FUSED_GRID_WORDS as u128)
                + 2 * rows as u128 * (h + m) + 4
        }
        // The fused path falls back to the separately staged path when its
        // dimensions do not admit. It has allocated no scratch at that point.
        _ => 0,
    };
    // Each requested root occupies at most twice its payload bucket bytes.
    u64::try_from(2 * batched.max(tiles).max(fused)).unwrap_or(u64::MAX)
}

/// Large shared attention roots, independent of the dispatch route. Small
/// immutable parameters are covered by the separate allocation allowance.
pub(crate) fn attention_arena_bytes(cfg: &Qwen35Config, rows: usize) -> Option<u64> {
    let h = u128::from(cfg.hidden_size);
    if rows > u32::MAX as usize {
        return None;
    }
    let q = u128::from(cfg.num_attention_heads) * u128::from(cfg.head_dim);
    let kv = u128::from(cfg.num_key_value_heads) * u128::from(cfg.head_dim);
    u64::try_from(rows as u128 * (4 * (h + 5 * q + 4 * kv) + 6 * q + 8 * kv + 4 * q)).ok()
}

/// DenseFfnArena/MoeFfnArena payloads, including their small parameter roots.
pub(crate) fn ffn_arena_bytes(cfg: &Qwen35Config, rows: usize, short_segments: usize) -> Option<u64> {
    if rows > u32::MAX as usize || short_segments > u32::MAX as usize {
        return None;
    }
    let (r, extra, h) = (rows as u128, short_segments as u128, u128::from(cfg.hidden_size));
    let bytes = match (cfg.intermediate_size, cfg.moe.as_ref()) {
        (Some(m), _) => r * (12 * u128::from(m) + 4 * h) + 4,
        (None, Some(moe)) => {
            let k = u128::from(moe.num_experts_per_tok);
            let per_row = 8 * k + 12 * k * u128::from(moe.moe_intermediate_size)
                + 4 * k * h + 12 * u128::from(moe.shared_expert_intermediate_size)
                + 4 * u128::from(moe.num_experts) + 4;
            (r + 32 * extra) * per_row + 12 * (1 + extra)
        }
        _ => 0,
    };
    u64::try_from(bytes).ok()
}

/// Matches the short-forward cadence resolved by forward_gpu_impl. It is
/// independent of PREFILL_WAIT_LAYERS and accepts every positive usize.
pub(crate) fn short_wait_layers() -> usize {
    std::env::var("HF2Q_FFN_TERMINAL_K_BATCH").ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&layers| layers >= 1).unwrap_or(8)
}

/// Prior pool owner bound across all legal worker transactions, not the pool
/// component of whichever transaction has the largest whole-forward peak.
/// Ordinary wait-one has no closed static pool bound; unknown is never zero.
pub(crate) fn retained_pool_bound_bytes(
    cfg: &Qwen35Config,
    content_rows: usize,
    slots: usize,
    kv_rows: usize,
    route: PrefillMemoryRoute,
    fold_cue: bool,
) -> Result<u64, &'static str> {
    if slots == 0 || slots > u32::MAX as usize {
        return Err("invalid prefill slot count");
    }
    if route.wait_layers == 1 {
        return Err("ordinary wait-one pool has no static retained bound");
    }
    let solo = PrefillMemoryRoute { joint: false, ..route };
    let mut pool = bound(cfg, content_rows, 1, kv_rows, solo)?.pool;
    if route.joint && slots > 1 {
        pool = pool.max(bound(cfg, content_rows, slots, kv_rows, route)?.pool);
    }
    if fold_cue {
        let folded_rows = content_rows.checked_add(super::prefill_fold::MAX_CUE_ROWS)
            .ok_or("folded prefill row count overflow")?;
        let folded = PrefillMemoryRoute { joint: true, ..route };
        pool = pool.max(bound(cfg, folded_rows, 1, kv_rows, folded)?.pool);
    }
    Ok(pool)
}
