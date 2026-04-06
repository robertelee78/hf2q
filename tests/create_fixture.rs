//! Helper to create test fixture safetensors file.
//! Run with: cargo run --example create_fixture
//! (This is actually invoked as a test helper)

use std::collections::BTreeMap;

/// Create a minimal valid safetensors file with tiny tensors.
pub fn create_tiny_safetensors() -> Vec<u8> {
    // Define small tensors for a tiny 2-layer model
    let tensors: Vec<(&str, Vec<usize>, &str, Vec<u8>)> = vec![
        // Layer 0 weight: 8x8 F16 = 128 bytes
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![8, 8],
            "F16",
            vec![0u8; 128],
        ),
        // Layer 0 norm: 8 F16 = 16 bytes
        (
            "model.layers.0.input_layernorm.weight",
            vec![8],
            "F16",
            vec![0u8; 16],
        ),
        // Layer 1 weight: 8x8 F16 = 128 bytes
        (
            "model.layers.1.self_attn.q_proj.weight",
            vec![8, 8],
            "F16",
            vec![0u8; 128],
        ),
        // Layer 1 norm: 8 F16 = 16 bytes
        (
            "model.layers.1.input_layernorm.weight",
            vec![8],
            "F16",
            vec![0u8; 16],
        ),
        // Embedding: 32x8 F16 = 512 bytes
        (
            "model.embed_tokens.weight",
            vec![32, 8],
            "F16",
            vec![0u8; 512],
        ),
    ];

    let mut header_map = BTreeMap::new();
    let mut current_offset = 0usize;
    let mut all_data = Vec::new();

    for (name, shape, dtype, data) in &tensors {
        let end_offset = current_offset + data.len();

        let tensor_info = serde_json::json!({
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [current_offset, end_offset]
        });

        header_map.insert(name.to_string(), tensor_info);
        all_data.extend_from_slice(data);
        current_offset = end_offset;
    }

    let header_json = serde_json::to_string(&header_map).unwrap();
    let header_bytes = header_json.as_bytes();
    let header_size = header_bytes.len() as u64;

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header_size.to_le_bytes());
    file_data.extend_from_slice(header_bytes);
    file_data.extend_from_slice(&all_data);

    file_data
}
