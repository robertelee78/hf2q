//! Per-token cross-layer context for `encode_one_layer` (ADR-028 iter-390).
//!
//! Bundles the local variables that forward_decode's pre-loop section
//! computes once and the layer-encoding loop reads many times.  This is
//! the prerequisite for iter-391's extraction of the layer body into a
//! `&self` method, which in turn enables iter-392's parallel encoding
//! across worker threads.
//!
//! # Why this struct exists
//!
//! Before iter-388: forward_decode's layer loop body had `&mut self.X`
//! borrows everywhere — extraction would require massive refactoring of
//! borrow patterns.
//!
//! After iter-388: layer body is `&self`-only on `self` accesses.
//! Remaining cross-loop state lives in 59 local `let` bindings inside
//! forward_decode's pre-loop section.  `LayerCtx` bundles these so
//! `encode_one_layer(&self, layer_idx, ctx, session, gpu)` can be a
//! clean signature instead of taking 30+ parameters.
//!
//! # Field categories
//!
//! 1. **Token state**: `seq_pos`, `input_token` — passed in to
//!    forward_decode.
//! 2. **Model constants**: `num_layers`, `hs`, `top_k`, `num_experts`,
//!    `moe_int`, `eps` — derived from self at fn entry.
//! 3. **KV-cache state**: `kv_info` (Vec of per-layer (is_sliding,
//!    write_pos, capacity, seq_len) tuples) — computed pre-loop.
//! 4. **Debug/dump flags**: `dump_layers`, `dump_detail_layer`,
//!    `dump_after_post_attn`, `dump_sliding_l0`, `dump_run_name` —
//!    INVESTIGATION_ENV-derived.
//! 5. **Buffer-split config**: `dual_buffer_splits` (per iter-374
//!    multi-split support).
//! 6. **Profiling state**: `per_layer_disp_enabled`,
//!    `per_layer_disp_log` (mutable shared via &mut).

// LayerType import deferred to iter-391 where it's actually needed.

/// Per-token, cross-layer encoding context.  Built once in
/// forward_decode's pre-loop section, then passed (`&LayerCtx`) to
/// each invocation of `encode_one_layer(&self, layer_idx, ctx, ...)`.
///
/// **iter-391 SCOPE NOTE**: this struct is defined now but NOT yet
/// populated/used.  iter-391 will:
/// 1. Add fields here as discovered during the extraction.
/// 2. Build the struct in forward_decode's pre-loop section.
/// 3. Replace each layer-body local-var reference with `ctx.X`.
/// 4. Call `self.encode_one_layer(layer_idx, &ctx, &mut session, gpu)`
///    in the loop.
///
/// Until iter-391, this struct is a documented placeholder — present in
/// the source tree to guide the upcoming refactor without yet altering
/// production behavior.
#[allow(dead_code)] // Populated + used in iter-391+
pub(crate) struct LayerCtx<'a> {
    // --- Token state ---
    pub seq_pos: usize,
    pub input_token: u32,

    // --- Model constants (cached from self) ---
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub eps: f32,

    // --- KV-cache pre-loop snapshot ---
    /// Per-layer (is_sliding, write_pos, capacity, seq_len) at decode entry.
    /// Layer-encoding loop reads these via `ctx.kv_info[layer_idx]`.
    pub kv_info: &'a [(bool, usize, usize, usize)],

    // --- Debug/dump flags (INVESTIGATION_ENV-derived, evaluated once) ---
    pub dump_layers: bool,
    pub dump_detail_layer: Option<usize>,
    pub dump_sliding_l0: bool,
    pub dump_run_name: Option<&'a str>,

    // --- Multi-split commit points (iter-374) ---
    pub dual_buffer_splits: &'a [usize],

    // --- Per-layer dispatch profiling ---
    pub per_layer_disp_enabled: bool,

    // --- TQ / HB-SDPA parameters (process-static, cached in forward_decode) ---
    pub tq_scale_factor_d512: f32,
    pub tq_codebook_bits: u32,
    pub use_native_hb_sdpa: bool,

    // --- Cache-efficiency dump flag ---
    pub dump_all_cache_eff: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stub test — confirms the struct compiles + is constructible.
    /// Real usage validation happens in iter-391+ when the extraction
    /// actually wires this struct into encode_one_layer.
    #[test]
    fn layer_ctx_struct_compiles() {
        let kv_info = vec![(false, 0usize, 1024usize, 0usize); 30];
        let dual_buffer_splits = vec![2usize];
        let _ctx = LayerCtx {
            seq_pos: 0,
            input_token: 1,
            num_layers: 30,
            hidden_size: 2816,
            num_attention_heads: 16,
            eps: 1e-6,
            kv_info: &kv_info,
            dump_layers: false,
            dump_detail_layer: None,
            dump_sliding_l0: false,
            dump_run_name: None,
            dual_buffer_splits: &dual_buffer_splits,
            per_layer_disp_enabled: false,
            tq_scale_factor_d512: 1.0,
            tq_codebook_bits: 8,
            use_native_hb_sdpa: true,
            dump_all_cache_eff: false,
        };
    }
}
