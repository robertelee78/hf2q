//! Gemma 4 per-layer FFN encoding — Path A stub.
//!
//! Under Path A, `encode_one_layer` in `gpu_full_attn.rs` keeps attention and
//! FFN interleaved.  This module is a placeholder for Path B follow-up, which
//! will extract `encode_attention_block` and `encode_ffn_block` into separate
//! functions to mirror the `qwen35/gpu_ffn.rs` layout.
//!
//! ADR-038 Step 3.  ADR-038 §5 "open follow-ups" tracks Path B.
