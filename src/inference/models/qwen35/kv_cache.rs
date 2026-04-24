//! Hybrid KV cache for Qwen3.5 (full-attn KV + linear-attn SSM state).
//!
//! Full-attention layers need a standard token-indexed KV cache.
//! Linear-attention layers need TWO caches per seq:
//!
//! * `conv_state`: `[K-1=3, channels, n_seqs]` — DeltaNet 1D conv ring buffer.
//! * `recurrent`:  `[D_k, D_v, num_v_heads, n_seqs]` — state matrix.
//!
//! Cache allocation strategy lands in P6. See ADR-013 Decision 11.
//!
//! This file also hosts the scalar CPU reference implementation used as the
//! per-layer parity oracle during P7/P8 validation. mlx-native already
//! provides `mlx_native::ops::gated_delta_net::cpu_reference_f32` — hf2q
//! re-exports rather than duplicating.
