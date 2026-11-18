//! Immutable config.json payloads for the two supported Qwen checkpoints.
//!
//! Preserve every byte, including the final newline: `allocation_config` uses
//! the descriptor's normal SHA256, config-schema and target-shape validation.

// incoai/Qwen3.6-35B-A3B-DFlash2@51ef7b6923ad6c14cb1bb41c37a9041446496ab6
// 1291 bytes; SHA256 86d2e8e57e7c70bb867240708d00c0535ae9ed3c828dbdcb21584471c7bbce59.
pub(super) const QWEN36_CONFIG_JSON: &[u8] = br#"{
  "architectures": [
    "DFlash2DraftModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": null,
  "dflash_config": {
    "block_size": 8,
    "mask_token_id": 248077,
    "target_layer_ids": [
      1,
      6,
      11,
      16,
      22,
      27,
      32,
      37
    ],
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "selector_rank": 256,
    "selector_top_k": 16
  },
  "dtype": "bfloat16",
  "eos_token_id": 248044,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 6144,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention"
  ],
  "max_position_embeddings": 262144,
  "max_window_layers": 6,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 6,
  "num_key_value_heads": 8,
  "num_target_layers": 40,
  "pad_token_id": 248044,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
    "rope_theta": 10000000,
    "rope_type": "default"
  },
  "sliding_window": 2048,
  "tie_word_embeddings": false,
  "transformers_version": "5.7.0",
  "use_cache": true,
  "use_sliding_window": true,
  "vocab_size": 248320,
  "is_causal": false
}
"#;

// z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4
// 1239 bytes; SHA256 873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980.
pub(super) const QWEN38_CONFIG_JSON: &[u8] = br#"{
  "architectures": [
    "DFlash2DraftModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": null,
  "is_causal": false,
  "dflash_config": {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248070,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [
      5,
      19,
      33,
      47,
      61
    ]
  },
  "dtype": "bfloat16",
  "eos_token_id": 248044,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 17408,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention"
  ],
  "max_position_embeddings": 262144,
  "max_window_layers": 5,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 5,
  "num_key_value_heads": 8,
  "num_target_layers": 64,
  "pad_token_id": 248044,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
    "rope_theta": 10000000,
    "rope_type": "default"
  },
  "sliding_window": 2048,
  "tie_word_embeddings": false,
  "transformers_version": "5.15.0",
  "use_cache": true,
  "use_sliding_window": true,
  "vocab_size": 248320
}
"#;
