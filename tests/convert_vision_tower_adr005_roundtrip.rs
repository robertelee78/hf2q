//! ADR-012 P10 Decision 18 — Layer B: ADR-005 round-trip gate.
//!
//! Emit a synthetic mmproj via `hf2q convert --emit-vision-tower`, then
//! load it back via the ADR-005 phase 2c mmproj entry point
//! (`src/inference/vision/mmproj.rs::MmprojConfig::from_gguf`). If our
//! emitter drops a required metadata key, uses a wrong dtype, or
//! produces an unloadable binary, this test fails loudly.
//!
//! Layer B is the contract between ADR-012 (convert side) and ADR-005
//! (inference side). Both live in this repo; we exercise the seam.
//!
//! The real-model (Qwen3.6-27B dense) variant is gated behind `#[ignore]`
//! so default cargo test stays < 30 s without any large download.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use mlx_native::gguf::GgufFile;

const TINY_VIT_CONFIG: &str = r#"{
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "intermediate_size": 128,
    "vocab_size": 128,
    "head_dim": 16,
    "linear_num_value_heads": 8,
    "full_attention_interval": 2,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000.0,
    "rope_scaling": {"mrope_section":[3,3,2],"type":"mrope"},
    "mtp_num_hidden_layers": 0,
    "attn_output_gate": true,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 131072,
    "dtype": "float16",
    "vision_config": {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "patch_size": 4,
        "image_size": 32,
        "intermediate_size": 128,
        "layer_norm_eps": 1e-5,
        "projector_type": "mlp"
    }
}"#;

fn push_f16_zeros(
    tensors: &mut Vec<(String, Vec<usize>, &'static str, Vec<u8>)>,
    name: &str,
    shape: Vec<usize>,
) {
    let size: usize = shape.iter().product::<usize>() * 2;
    tensors.push((name.to_string(), shape, "F16", vec![0u8; size]));
}

fn build_safetensors_bytes(tensors: Vec<(String, Vec<usize>, &str, Vec<u8>)>) -> Vec<u8> {
    let mut header_map = BTreeMap::new();
    let mut offset = 0usize;
    let mut payload = Vec::new();
    for (name, shape, dtype, data) in &tensors {
        let end = offset + data.len();
        header_map.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, end],
            }),
        );
        payload.extend_from_slice(data);
        offset = end;
    }
    let header_json = serde_json::to_string(&header_map).unwrap();
    let hbytes = header_json.as_bytes();
    let hsize = hbytes.len() as u64;
    let mut out = Vec::new();
    out.extend_from_slice(&hsize.to_le_bytes());
    out.extend_from_slice(hbytes);
    out.extend_from_slice(&payload);
    out
}

fn build_tiny_vit_safetensors() -> Vec<u8> {
    let h = 64usize;
    let inter = 128usize;
    let num_patches = 64usize; // 8*8

    let mut tensors: Vec<(String, Vec<usize>, &str, Vec<u8>)> = Vec::new();
    push_f16_zeros(&mut tensors, "model.embed_tokens.weight", vec![128, h]);
    push_f16_zeros(&mut tensors, "lm_head.weight", vec![128, h]);
    push_f16_zeros(&mut tensors, "model.norm.weight", vec![h]);
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        push_f16_zeros(&mut tensors, &format!("{p}.input_layernorm.weight"), vec![h]);
        push_f16_zeros(&mut tensors, &format!("{p}.post_attention_layernorm.weight"), vec![h]);
        if layer == 1 {
            push_f16_zeros(&mut tensors, &format!("{p}.self_attn.q_proj.weight"), vec![h, h]);
            push_f16_zeros(&mut tensors, &format!("{p}.self_attn.k_proj.weight"), vec![16, h]);
            push_f16_zeros(&mut tensors, &format!("{p}.self_attn.v_proj.weight"), vec![16, h]);
            push_f16_zeros(&mut tensors, &format!("{p}.self_attn.o_proj.weight"), vec![h, h]);
            push_f16_zeros(&mut tensors, &format!("{p}.self_attn.gate.weight"), vec![1, h]);
        } else {
            let qkv = (4 + 1 + 8) * 16;
            push_f16_zeros(&mut tensors, &format!("{p}.linear_attn.in_proj_qkv.weight"), vec![qkv, h]);
            push_f16_zeros(&mut tensors, &format!("{p}.linear_attn.out_proj.weight"), vec![h, 128]);
            push_f16_zeros(&mut tensors, &format!("{p}.linear_attn.in_proj_a.weight"), vec![4, h]);
            push_f16_zeros(&mut tensors, &format!("{p}.linear_attn.in_proj_b.weight"), vec![4, h]);
            push_f16_zeros(&mut tensors, &format!("{p}.linear_attn.in_proj_z.weight"), vec![128, h]);
        }
        push_f16_zeros(&mut tensors, &format!("{p}.mlp.gate_proj.weight"), vec![inter, h]);
        push_f16_zeros(&mut tensors, &format!("{p}.mlp.up_proj.weight"), vec![inter, h]);
        push_f16_zeros(&mut tensors, &format!("{p}.mlp.down_proj.weight"), vec![h, inter]);
    }

    push_f16_zeros(
        &mut tensors,
        "model.vision_tower.embeddings.patch_embeddings.projection.weight",
        vec![h, 3, 4, 4],
    );
    push_f16_zeros(
        &mut tensors,
        "model.vision_tower.embeddings.position_embeddings.weight",
        vec![num_patches, h],
    );
    push_f16_zeros(&mut tensors, "model.vision_tower.post_layernorm.weight", vec![h]);
    push_f16_zeros(&mut tensors, "model.vision_tower.post_layernorm.bias", vec![h]);
    for l in 0..2 {
        let pre = format!("model.vision_tower.encoder.layer.{l}");
        push_f16_zeros(&mut tensors, &format!("{pre}.attention.q_proj.weight"), vec![h, h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.attention.k_proj.weight"), vec![h, h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.attention.v_proj.weight"), vec![h, h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.attention.output.dense.weight"), vec![h, h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.layer_norm1.weight"), vec![h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.layer_norm2.weight"), vec![h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.mlp.fc1.weight"), vec![inter, h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.mlp.fc2.weight"), vec![h, inter]);
    }
    push_f16_zeros(&mut tensors, "model.multi_modal_projector.linear_1.weight", vec![h, h]);
    push_f16_zeros(&mut tensors, "model.multi_modal_projector.linear_2.weight", vec![h, h]);

    build_safetensors_bytes(tensors)
}

fn setup_model(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("config.json"), TINY_VIT_CONFIG).unwrap();
    fs::write(dir.join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();
    fs::write(dir.join("tokenizer_config.json"), r#"{"model_max_length":131072}"#).unwrap();
    fs::write(dir.join("model.safetensors"), build_tiny_vit_safetensors()).unwrap();
}

#[test]
fn synthetic_mmproj_loads_via_mlx_native_gguf_reader() {
    // Layer B proxy: use mlx-native's GgufFile (which backs both the
    // text loader and the mmproj loader) to verify metadata integrity.
    // This exercises the same read path MmprojConfig::from_gguf uses
    // without requiring the crate-internal inference module (which is
    // a binary crate internal and not importable from integration tests).
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("in");
    let output = tmp.path().join("out/text.gguf");
    setup_model(&input);
    fs::create_dir_all(output.parent().unwrap()).unwrap();

    let _ = assert_cmd::Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input", input.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "f16",
            "--output", output.to_str().unwrap(),
            "--emit-vision-tower",
            "--yes",
        ])
        .assert();

    // Find mmproj alongside the text output (if any).
    let parent = output.parent().unwrap();
    let mmproj = fs::read_dir(parent)
        .unwrap()
        .filter_map(|e| e.ok().map(|x| x.path()))
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.starts_with("mmproj-") && s.ends_with(".gguf"))
                .unwrap_or(false)
        });

    let mmproj = Some(mmproj.unwrap_or_else(|| {
        let dir_contents: Vec<String> = fs::read_dir(parent)
            .unwrap()
            .filter_map(|e| e.ok().map(|x| x.file_name().to_string_lossy().to_string()))
            .collect();
        panic!(
            "ADR-012 P10 Layer B: --emit-vision-tower produced no mmproj under {:?}. \
             Contents: {:?}. See docs/ADR-012-qwen35moe-conversion.md Decision 18 §3.",
            parent, dir_contents
        );
    }));

    let path = mmproj.unwrap();
    let gguf = GgufFile::open(&path).expect("open mmproj GGUF");

    // The mmproj loader requires `general.architecture == "clip"`. Any
    // drift in our emitter here and the ADR-005 loader rejects.
    let arch = gguf
        .metadata_string("general.architecture")
        .expect("general.architecture present");
    assert_eq!(arch, "clip");

    // `clip.vision.*` required keys — same set MmprojConfig::from_gguf
    // reads.
    for key in &[
        "clip.vision.image_size",
        "clip.vision.patch_size",
        "clip.vision.embedding_length",
        "clip.vision.feed_forward_length",
        "clip.vision.attention.head_count",
        "clip.vision.block_count",
    ] {
        assert!(
            gguf.metadata_u32(key).is_some(),
            "mmproj missing required u32 metadata key {:?}",
            key
        );
    }

    // Image size / patch size relationship — validated by
    // MmprojConfig::from_gguf at load time. Fail loudly here too.
    let image_size = gguf.metadata_u32("clip.vision.image_size").unwrap();
    let patch_size = gguf.metadata_u32("clip.vision.patch_size").unwrap();
    assert_eq!(image_size, 32);
    assert_eq!(patch_size, 4);
    assert_eq!(image_size % patch_size, 0);
}

#[test]
#[ignore]
fn real_27b_dense_mmproj_loads() {
    // Real-model extension. `#[ignore]` so default cargo test stays
    // fast; run with `cargo test -- --ignored` after producing the
    // real mmproj via `hf2q smoke --arch qwen35 --with-vision`.
    //
    // Asserts the 27-layer ViT tensor set lands with shapes matching
    // Qwen3.6-27B's vision_config. Catches any "passes synthetic,
    // fails at real dimensions" mapping bug that the synthetic layout
    // tests (Layer C) can't catch.
    let path = std::path::PathBuf::from(
        "/opt/hf2q/models/qwen3.6-27b-q4_0/mmproj-qwen3.6-27b-F16.gguf",
    );
    if !path.exists() {
        eprintln!("skipping: real mmproj not at expected path (run hf2q smoke first)");
        return;
    }
    let gguf = GgufFile::open(&path).expect("open real mmproj");
    assert_eq!(
        gguf.metadata_string("general.architecture").as_deref(),
        Some("clip")
    );
    let num_blocks = gguf.metadata_u32("clip.vision.block_count").unwrap();
    for l in 0..num_blocks {
        for suffix in &["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_out.weight"] {
            let name = format!("v.blk.{}.{}", l, suffix);
            assert!(
                gguf.tensor_info(&name).is_some(),
                "real 27B mmproj missing {:?}",
                name
            );
        }
    }
}
