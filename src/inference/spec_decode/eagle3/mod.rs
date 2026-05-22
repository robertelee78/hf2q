//! EAGLE-3 + dynamic tree speculative decoding (ADR-037).
//!
//! Phase tracker (status at hf2q HEAD `3b78d2bc` + codex /cfa fixes):
//!
//! - **Phase E1** (mlx-native): tree_attention Metal kernel — CLOSED.
//!   Codex /cfa Critical 0 + Major 0. 21 parity tests PASS across 5
//!   topology classes (tree=1, chain, fixed-square, dynamic
//!   asymmetric, prefix+tree combined).
//! - **Phase E2**: drafter training — multi-week training compute.
//! - **Phase E3** (hf2q, this module): drafter loader + multi-layer
//!   hidden state plumbing. **E3a + E3b SHIPPED**:
//!   * `multi_layer_hidden.rs` defines `Eagle3HiddenCollector` and the
//!     `[seq_len, num_aux * hidden_size]` concatenation contract
//!     per vLLM `model_executor/models/llama_eagle3.py:186`.
//!   * `config.rs` defines `Eagle3DrafterConfig` with architectural
//!     gates (norm_before_fc, fc_norm, use_qk_norm, attention_bias,
//!     tie_lm_head, include_draft_id_mapping, has_own_embed_tokens).
//!   * `weights.rs` defines `Eagle3Weights` strict manifest loader
//!     with vLLM d2t/t2d name normalization (per `llama_eagle3.py:415-419`).
//! - **Phase E4-E8**: drafter forward + dynamic tree + tree-walk +
//!   orchestrator + empirical validation + final closure.
//!
//! Peer reference: `/opt/vllm/vllm/model_executor/models/llama_eagle3.py`
//! (MIT). Production-deployed EAGLE-3 in vLLM consumes
//! `num_aux_hidden_states` (default 3) hidden states from the target
//! model, concatenated along the last dim, projected by an FC layer
//! to the drafter's hidden_size, then fed into a 1-layer transformer
//! drafter alongside the input embeddings.

pub mod config;
pub mod drafter;
pub mod drafter_gpu;
pub mod dynamic_tree;
pub mod forward;
pub mod multi_layer_hidden;
pub mod tensors;
pub mod weights;
