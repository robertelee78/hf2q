//! Exact selected bf65 GGUF scalar configuration; MUST match full prepared Debug.
use crate::serve::config::{Gemma4Config,LayerType};
pub(super) fn cfg() -> Gemma4Config {
    // Exact scalar mapping from frozen GGUF bf65b7e1...: block_count30,
    // head_count_kv[8,8,8,8,8,2] repeated; swa/global key_length256/512,
    // window1024/context262144. No tensor/tokenizer payload fixture.
    Gemma4Config {
        vocab_size: 262144,
        hidden_size: 2816,
        intermediate_size: 2112,
        moe_intermediate_size: 704,
        num_hidden_layers: 30,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        num_global_key_value_heads: 2,
        head_dim: 256,
        global_head_dim: 512,
        rms_norm_eps: 9.999999974752427e-7,
        rope_theta_sliding: 10000.,
        rope_theta_global: 1000000.,
        sliding_window: 1024,
        max_position_embeddings: 262144,
        final_logit_softcapping: Some(30.),
        attention_bias: false,
        attention_k_eq_v: true,
        tie_word_embeddings: true,
        num_experts: 128,
        top_k_experts: 8,
        layer_types: (0..30)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    LayerType::Full
                } else {
                    LayerType::Sliding
                }
            })
            .collect(),
    }
}
