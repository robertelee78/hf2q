//! Thin wrappers around mlx-native's new Qwen3.5 ops.
//!
//! Keeps the forward-pass files (`dense.rs`, `moe.rs`, `mod.rs::build_*_layer`)
//! readable: no math, just dispatch. The math lives in the mlx-native kernel;
//! the wrappers here bind the GGUF-resident weights to mlx-native buffers and
//! encode the op into the current `CommandEncoder`.
//!
//! mlx-native ops consumed (all landed in iters 1–5):
//!
//! * `l2_norm` — DeltaNet Q/K normalization (Decision 3)
//! * `cumsum` — DeltaNet chunked-path decay mask (Decision 4)
//! * `tri_solve` — DeltaNet chunked-path debug solve (Decision 5)
//! * `gated_delta_net` — fused DeltaNet recurrence (Decision 6)
//! * `ssm_conv` — DeltaNet 1D causal conv + SiLU (Decision 7)
//! * `rope_multi` — IMROPE for full-attention Q/K (Decision 10)
//!
//! Wrappers land in P7 (full-attn) and P8 (DeltaNet) per ADR-013.
