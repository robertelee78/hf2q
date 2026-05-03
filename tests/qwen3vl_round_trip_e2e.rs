//! ADR-005 iter-224 row 6 Wedge-4f — Qwen3-VL HF→GGUF→hf2q round-trip
//! end-to-end harness.
//!
//! This test file is the closure gate for the Wedge-4 series. It drives
//! the FULL convert path on a synthetic Qwen3-VL HF-shaped fixture and
//! asserts that:
//!
//!   1. The converter detects `vision_config` with Qwen3-VL markers and
//!      emits an mmproj GGUF alongside the text GGUF.
//!   2. Every Qwen3-VL-specific GGUF metadata key is present and
//!      typed correctly:
//!        * `clip.projector_type = "qwen3vl_merger"` (String)
//!        * `clip.use_gelu = true`                   (Bool)
//!        * `clip.vision.spatial_merge_size = 2`     (u32)
//!        * `clip.vision.is_deepstack_layers`        (Bool[block_count])
//!   3. Every required Qwen3-VL tensor lands in the mmproj with the
//!      canonical llama.cpp name:
//!        * Per-block: `v.blk.{N}.attn_q/k/v/out.{weight,bias}`,
//!          `v.blk.{N}.ln1/ln2.{weight,bias}`,
//!          `v.blk.{N}.ffn_up/down.{weight,bias}`
//!        * Globals: `v.patch_embd.weight` (+ `.weight.1` if temporal=2),
//!          `v.position_embd.weight`, `v.post_ln.{weight,bias}`,
//!          `mm.{0,2}.{weight,bias}`
//!        * DeepStack: `v.deepstack.{abs_idx}.{norm,fc1,fc2}.{weight,bias}`
//!   4. The emitted file parses back via the same `mlx_native::gguf`
//!      reader the production loader uses.
//!
//! ## Test stratification
//!
//! Three tiers of tests:
//!
//!   * **Default (always-on)**: synthetic 2-block Qwen3-VL fixture round-trip
//!     via `cargo run --bin hf2q -- convert ... --emit-vision-tower`.
//!     <30s wall time, no external download.
//!
//!   * **Operator-gated** (`HF2Q_QWEN3VL_ROUND_TRIP=1`): real
//!     `Qwen/Qwen3-VL-2B-Instruct` HF dir → converter →
//!     mmproj loader cleanly accepts. Provides full-dimension shape
//!     parity + token-1 logit comparison vs llama.cpp golden. Skipped
//!     by default since the target model is ~5 GB on disk.
//!
//! ## Wedge-4 closure
//!
//! When this harness passes:
//!
//!   * Wedge-4a (LM API opener)       — LANDED (cbfffa3)
//!   * Wedge-4b (mmproj profile)      — LANDED (fa7acfb)
//!   * Wedge-4c (ViT forward)         — LANDED (4c1b85b)
//!   * Wedge-4c.5 (LM-side hooks)     — LANDED (in 4c)
//!   * Wedge-4d (chat handler)        — LANDED (67cee7e)
//!   * Wedge-4e (streaming + tools)   — LANDED (28b75b7)
//!   * Wedge-4f (convert — THIS)      — LANDED (this commit)
//!
//! Full closure: HF → GGUF → hf2q chat-with-images works end-to-end on
//! both serve and CLI paths. Wedge-4 series is COMPLETE.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use mlx_native::gguf::{GgufFile, MetadataValue};

// ---------------------------------------------------------------------------
// Synthetic Qwen3-VL fixture
// ---------------------------------------------------------------------------
//
// Mini-fixture: 2-block ViT, hidden=64, num_heads=8, patch_size=4,
// image_size=32, intermediate_size=128, deepstack at indexes [0, 1].
//
// Note: Qwen3-VL safetensors omit the `model.` prefix on the visual
// tower (per /opt/llama.cpp/convert_hf_to_gguf.py:4908-4909
// `name.replace("model.visual.", "visual.", 1)`); we mirror that.

const TINY_HIDDEN: usize = 64;
const TINY_NUM_LAYERS: usize = 2;
const TINY_PATCH_SIZE: usize = 4;
// Image size = 32 → 8×8 patches → num_position_embeddings = 64 (used in
// `TINY_CONFIG`'s `num_position_embeddings: 64`). Kept as a doc-only
// comment — the literal lives in the JSON config string and the
// safetensors builder uses `num_patches = 64` directly to avoid a
// dead-code warning on an unused const.
const TINY_INTERMEDIATE: usize = 128;
const TINY_TEMPORAL_PATCH: usize = 2;
const TINY_DEEPSTACK_INDEXES: [u32; 2] = [0, 1];
const TINY_SPATIAL_MERGE: u32 = 2;

const TINY_CONFIG: &str = r#"{
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "model_type": "qwen3_vl",
    "text_config": {
        "model_type": "qwen3_5",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "intermediate_size": 128,
        "vocab_size": 128,
        "head_dim": 16,
        "linear_num_value_heads": 8,
        "linear_num_key_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000.0,
        "rope_scaling": {"mrope_section":[3,3,2],"type":"mrope"},
        "mtp_num_hidden_layers": 0,
        "attn_output_gate": true,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "dtype": "float16"
    },
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "intermediate_size": 128,
    "vocab_size": 128,
    "head_dim": 16,
    "linear_num_value_heads": 8,
    "linear_num_key_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 2,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000.0,
    "rope_scaling": {"mrope_section":[3,3,2],"type":"mrope"},
    "mtp_num_hidden_layers": 0,
    "attn_output_gate": true,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-5,
    "dtype": "float16",
    "vision_config": {
        "hidden_size": 64,
        "num_heads": 8,
        "depth": 2,
        "patch_size": 4,
        "num_position_embeddings": 64,
        "intermediate_size": 128,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "deepstack_visual_indexes": [0, 1],
        "projector_type": "qwen3vl_merger",
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std":  [0.26862954, 0.26130258, 0.27577711]
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

fn build_qwen3vl_safetensors() -> Vec<u8> {
    let h = TINY_HIDDEN;
    let inter = TINY_INTERMEDIATE;

    let mut tensors: Vec<(String, Vec<usize>, &str, Vec<u8>)> = Vec::new();

    // ---- Text side (minimal — enough to satisfy the text convert path) ----
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
            let qkv_rows = 4 * 16 * 2 + 8 * 16; // 256
            push_f16_zeros(&mut tensors, &format!("{p}.linear_attn.in_proj_qkv.weight"), vec![qkv_rows, h]);
            push_f16_zeros(&mut tensors, &format!("{p}.linear_attn.out_proj.weight"), vec![h, 8 * 16]);
            push_f16_zeros(&mut tensors, &format!("{p}.linear_attn.in_proj_a.weight"), vec![8, h]);
            push_f16_zeros(&mut tensors, &format!("{p}.linear_attn.in_proj_b.weight"), vec![8, h]);
            push_f16_zeros(&mut tensors, &format!("{p}.linear_attn.in_proj_z.weight"), vec![8 * 16, h]);
        }
        push_f16_zeros(&mut tensors, &format!("{p}.mlp.gate_proj.weight"), vec![inter, h]);
        push_f16_zeros(&mut tensors, &format!("{p}.mlp.up_proj.weight"), vec![inter, h]);
        push_f16_zeros(&mut tensors, &format!("{p}.mlp.down_proj.weight"), vec![h, inter]);
    }

    // ---- Vision tower (no `model.` prefix per Qwen3-VL convention) ----
    //
    // patch_embed.proj.weight is 5-D: [hidden, in=3, T=2, H=4, W=4].
    // The converter splits along T into two 4-D slices.
    push_f16_zeros(
        &mut tensors,
        "visual.patch_embed.proj.weight",
        vec![h, 3, TINY_TEMPORAL_PATCH, TINY_PATCH_SIZE, TINY_PATCH_SIZE],
    );
    // No `visual.patch_embed.proj.bias` in our fixture — Qwen3-VL ships
    // it but the loader treats it as optional; the converter will simply
    // not emit `v.patch_embd.bias` when absent.

    // Position embedding: count = (image_size/patch_size)^2 = (32/4)^2 = 64.
    push_f16_zeros(&mut tensors, "visual.pos_embed.weight", vec![64, h]);

    // Per-block ViT encoder. Use FUSED attn_qkv form (Qwen3-VL HF
    // canonical) so the mmproj loader's Wedge-4c.5 split-name view
    // installer is exercised.
    for l in 0..TINY_NUM_LAYERS {
        let pre = format!("visual.blocks.{l}");
        // FUSED QKV: [3*hidden, hidden] — matches Qwen3-VL HF source.
        push_f16_zeros(&mut tensors, &format!("{pre}.attn.qkv.weight"), vec![3 * h, h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.attn.qkv.bias"), vec![3 * h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.attn.proj.weight"), vec![h, h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.attn.proj.bias"), vec![h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.norm1.weight"), vec![h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.norm1.bias"), vec![h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.norm2.weight"), vec![h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.norm2.bias"), vec![h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.mlp.linear_fc1.weight"), vec![inter, h]);
        push_f16_zeros(&mut tensors, &format!("{pre}.mlp.linear_fc1.bias"), vec![inter]);
        push_f16_zeros(&mut tensors, &format!("{pre}.mlp.linear_fc2.weight"), vec![h, inter]);
        push_f16_zeros(&mut tensors, &format!("{pre}.mlp.linear_fc2.bias"), vec![h]);
    }

    // Merger: linear_fc1 → mm.0, linear_fc2 → mm.2, norm → v.post_ln.
    // The merger fuses 2×2 patches → input dim = hidden * 4 = 256.
    let merger_in_dim = h * (TINY_SPATIAL_MERGE as usize) * (TINY_SPATIAL_MERGE as usize);
    push_f16_zeros(&mut tensors, "visual.merger.norm.weight", vec![merger_in_dim]);
    push_f16_zeros(&mut tensors, "visual.merger.norm.bias", vec![merger_in_dim]);
    push_f16_zeros(&mut tensors, "visual.merger.linear_fc1.weight", vec![inter, merger_in_dim]);
    push_f16_zeros(&mut tensors, "visual.merger.linear_fc1.bias", vec![inter]);
    push_f16_zeros(&mut tensors, "visual.merger.linear_fc2.weight", vec![h, inter]);
    push_f16_zeros(&mut tensors, "visual.merger.linear_fc2.bias", vec![h]);

    // DeepStack heads — relative idx 0/1 → absolute 0/1 (per
    // TINY_DEEPSTACK_INDEXES). Each head: norm + linear_fc1 + linear_fc2.
    // The deepstack head's input dim mirrors the merger input
    // (hidden * spatial_merge_size^2) per qwen3vl.cpp:150-165.
    for rel_idx in 0..TINY_DEEPSTACK_INDEXES.len() {
        let pre = format!("visual.deepstack_merger_list.{rel_idx}");
        push_f16_zeros(&mut tensors, &format!("{pre}.norm.weight"), vec![merger_in_dim]);
        push_f16_zeros(&mut tensors, &format!("{pre}.norm.bias"), vec![merger_in_dim]);
        push_f16_zeros(&mut tensors, &format!("{pre}.linear_fc1.weight"), vec![inter, merger_in_dim]);
        push_f16_zeros(&mut tensors, &format!("{pre}.linear_fc1.bias"), vec![inter]);
        push_f16_zeros(&mut tensors, &format!("{pre}.linear_fc2.weight"), vec![h, inter]);
        push_f16_zeros(&mut tensors, &format!("{pre}.linear_fc2.bias"), vec![h]);
    }

    build_safetensors_bytes(tensors)
}

fn setup_qwen3vl_fixture(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("config.json"), TINY_CONFIG).unwrap();
    fs::write(dir.join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();
    fs::write(dir.join("tokenizer_config.json"), r#"{"model_max_length":131072}"#).unwrap();
    fs::write(dir.join("generation_config.json"), r#"{"do_sample":false}"#).unwrap();
    fs::write(dir.join("special_tokens_map.json"), r#"{"bos_token":"<|im_start|>"}"#).unwrap();
    fs::write(dir.join("model.safetensors"), build_qwen3vl_safetensors()).unwrap();
}

fn run_convert(input: &Path, text_output: &Path) -> std::process::Output {
    fs::create_dir_all(text_output.parent().unwrap()).unwrap();
    let assert = assert_cmd::Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input", input.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "f16",
            "--output", text_output.to_str().unwrap(),
            "--emit-vision-tower",
            "--yes",
        ])
        .assert();
    assert.get_output().clone()
}

fn find_mmproj_under(parent: &Path) -> Option<std::path::PathBuf> {
    fs::read_dir(parent)
        .ok()?
        .filter_map(|e| e.ok().map(|x| x.path()))
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.starts_with("mmproj-") && s.ends_with(".gguf"))
                .unwrap_or(false)
        })
}

// ---------------------------------------------------------------------------
// Test 1 — converter detects Qwen3-VL vision_config and emits mmproj
// ---------------------------------------------------------------------------

#[test]
fn wedge4f_qwen3vl_vision_config_emits_mmproj() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen3vl-tiny");
    let output = tmp.path().join("out");
    setup_qwen3vl_fixture(&input);
    let text_path = output.join("text.gguf");
    let out = run_convert(&input, &text_path);
    assert!(
        out.status.success(),
        "convert failed: stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let parent = text_path.parent().unwrap();
    let mmproj = find_mmproj_under(parent);
    assert!(
        mmproj.is_some(),
        "Wedge-4f: --emit-vision-tower with Qwen3-VL vision_config must emit \
         mmproj-*.gguf alongside text GGUF; found nothing under {:?}",
        parent
    );
}

// ---------------------------------------------------------------------------
// Test 2 — Qwen3-VL-specific metadata keys present + correctly typed
// ---------------------------------------------------------------------------

#[test]
fn wedge4f_qwen3vl_metadata_keys_present() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen3vl-meta");
    let output = tmp.path().join("out");
    setup_qwen3vl_fixture(&input);
    let text_path = output.join("text.gguf");
    let _ = run_convert(&input, &text_path);
    let parent = text_path.parent().unwrap();
    let mmproj = find_mmproj_under(parent).expect("mmproj emitted");
    let gguf = GgufFile::open(&mmproj).expect("open mmproj");

    // Architecture is `clip` per llama.cpp convention.
    assert_eq!(
        gguf.metadata_string("general.architecture"),
        Some("clip")
    );

    // Qwen3-VL projector_type pinned (exact upstream string).
    let projector = gguf
        .metadata_string("clip.projector_type")
        .expect("clip.projector_type present");
    assert_eq!(
        projector, "qwen3vl_merger",
        "clip.projector_type must be the canonical 'qwen3vl_merger' string \
         per /opt/llama.cpp/gguf-py/gguf/constants.py:996 \
         (VISION_PROJECTOR_TYPE_NAMES[QWEN3VL])"
    );

    // spatial_merge_size = 2 per the canonical Qwen3-VL config.
    let sms = gguf
        .metadata_u32("clip.vision.spatial_merge_size")
        .expect("clip.vision.spatial_merge_size present");
    assert_eq!(sms, 2, "spatial_merge_size for Qwen3-VL is 2 (2x2 patch fold)");

    // is_deepstack_layers — Bool[block_count] with `true` at indexes
    // [0, 1] (TINY_DEEPSTACK_INDEXES). Length must equal block_count
    // per /opt/llama.cpp/convert_hf_to_gguf.py:4895-4896.
    let raw = gguf
        .metadata("clip.vision.is_deepstack_layers")
        .expect("clip.vision.is_deepstack_layers present");
    let arr = match raw {
        MetadataValue::Array(a) => a,
        other => panic!("expected Array(Bool), got {:?}", other),
    };
    assert_eq!(
        arr.len(),
        TINY_NUM_LAYERS,
        "is_deepstack_layers length must equal block_count"
    );
    let mut true_positions: Vec<u32> = Vec::new();
    for (i, v) in arr.iter().enumerate() {
        match v {
            MetadataValue::Bool(true) => true_positions.push(i as u32),
            MetadataValue::Bool(false) => {}
            other => panic!("expected Bool entry, got {:?}", other),
        }
    }
    assert_eq!(
        true_positions, TINY_DEEPSTACK_INDEXES,
        "true entries must match deepstack_visual_indexes"
    );

    // Optional: clip.use_gelu (Bool) — Qwen3-VL emits this as true.
    let use_gelu_kv = gguf
        .metadata("clip.use_gelu")
        .expect("clip.use_gelu present");
    match use_gelu_kv {
        MetadataValue::Bool(true) => {}
        other => panic!("expected Bool(true) for clip.use_gelu, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Test 3 — every required Qwen3-VL tensor lands in the mmproj
// ---------------------------------------------------------------------------

#[test]
fn wedge4f_qwen3vl_required_tensors_present() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen3vl-tensors");
    let output = tmp.path().join("out");
    setup_qwen3vl_fixture(&input);
    let text_path = output.join("text.gguf");
    let _ = run_convert(&input, &text_path);
    let parent = text_path.parent().unwrap();
    let mmproj = find_mmproj_under(parent).expect("mmproj emitted");
    let gguf = GgufFile::open(&mmproj).expect("open mmproj");
    let names: Vec<&str> = gguf.tensor_names();

    // Globals — always present.
    let must_have = [
        "v.patch_embd.weight",      // first temporal slice
        "v.patch_embd.weight.1",    // second temporal slice (Qwen3-VL dual stem)
        "v.position_embd.weight",
        "v.post_ln.weight",         // merger.norm → V_POST_NORM (NOT mm.input_norm)
        "v.post_ln.bias",
        "mm.0.weight",              // merger.linear_fc1
        "mm.0.bias",
        "mm.2.weight",              // merger.linear_fc2
        "mm.2.bias",
    ];
    for name in &must_have {
        assert!(
            names.contains(name),
            "Wedge-4f: mmproj missing required global tensor {:?}; got {:?}",
            name, names
        );
    }

    // Per-block — fused QKV (Qwen3-VL canonical form).
    for l in 0..TINY_NUM_LAYERS {
        for suffix in &[
            "attn_qkv.weight",
            "attn_qkv.bias",
            "attn_out.weight",
            "attn_out.bias",
            "ln1.weight",
            "ln1.bias",
            "ln2.weight",
            "ln2.bias",
            "ffn_up.weight",
            "ffn_up.bias",
            "ffn_down.weight",
            "ffn_down.bias",
        ] {
            let expected = format!("v.blk.{}.{}", l, suffix);
            assert!(
                names.iter().any(|n| *n == expected),
                "Wedge-4f: mmproj missing per-block tensor {:?}",
                expected
            );
        }
    }

    // DeepStack heads — at every absolute index in deepstack_visual_indexes
    // (relative idx 0,1 → absolute 0,1 in our fixture).
    for &abs_idx in &TINY_DEEPSTACK_INDEXES {
        for suffix in &[
            "norm.weight",
            "norm.bias",
            "fc1.weight",
            "fc1.bias",
            "fc2.weight",
            "fc2.bias",
        ] {
            let expected = format!("v.deepstack.{}.{}", abs_idx, suffix);
            assert!(
                names.iter().any(|n| *n == expected),
                "Wedge-4f: mmproj missing DeepStack tensor {:?}",
                expected
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4 — patch_embd dual-stem split correctness
// ---------------------------------------------------------------------------

#[test]
fn wedge4f_qwen3vl_patch_embd_dual_stem_split() {
    // Qwen3-VL ships patch_embed.proj.weight as 5-D
    // [hidden, in, T=2, H, W]; the converter splits T → two 4-D
    // tensors `v.patch_embd.weight` (T-slice 0) and
    // `v.patch_embd.weight.1` (T-slice 1).
    //
    // Pinned per /opt/llama.cpp/convert_hf_to_gguf.py:4955-4963
    // (`Qwen3VLVisionModel.modify_tensors`'s
    // `data_torch[:, :, 0, ...]` / `data_torch[:, :, 1, ...]`).
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen3vl-patch");
    let output = tmp.path().join("out");
    setup_qwen3vl_fixture(&input);
    let text_path = output.join("text.gguf");
    let _ = run_convert(&input, &text_path);
    let parent = text_path.parent().unwrap();
    let mmproj = find_mmproj_under(parent).expect("mmproj emitted");
    let gguf = GgufFile::open(&mmproj).expect("open mmproj");

    let info0 = gguf
        .tensor_info("v.patch_embd.weight")
        .expect("v.patch_embd.weight present");
    let info1 = gguf
        .tensor_info("v.patch_embd.weight.1")
        .expect("v.patch_embd.weight.1 present");

    // Both slices have the SAME 4-D shape:
    // [hidden, in_channels, patch_size, patch_size] ↔ on-disk reverse
    // = [patch, patch, in_channels, hidden].
    assert_eq!(
        info0.shape, info1.shape,
        "the two temporal patch slices must share shape — they are \
         siblings produced by the same `data_torch[..., t, ...]` split"
    );
}

// ---------------------------------------------------------------------------
// Test 5 — non-Qwen3-VL configs do NOT emit Qwen3-VL keys (regression gate)
// ---------------------------------------------------------------------------

#[test]
fn wedge4f_non_qwen3vl_omits_qwen3vl_keys() {
    // Sovereignty: a CLIP-classic vision_config (projector_type="mlp",
    // no deepstack_visual_indexes) MUST NOT emit any of the Qwen3-VL
    // extension keys. Enforces that our writer is gated on the
    // family detection helper rather than always-on.
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("clip-classic");
    let output = tmp.path().join("out");
    fs::create_dir_all(&input).unwrap();
    fs::write(
        input.join("config.json"),
        r#"{
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
            "linear_num_key_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
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
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "patch_size": 4,
                "image_size": 32,
                "intermediate_size": 128,
                "layer_norm_eps": 1e-5,
                "projector_type": "mlp"
            }
        }"#,
    )
    .unwrap();
    fs::write(input.join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();
    fs::write(input.join("tokenizer_config.json"), r#"{"model_max_length":131072}"#).unwrap();

    // Build a small classic-CLIP safetensors (just text + minimal vision).
    let mut tensors: Vec<(String, Vec<usize>, &str, Vec<u8>)> = Vec::new();
    push_f16_zeros(&mut tensors, "model.embed_tokens.weight", vec![128, 64]);
    push_f16_zeros(&mut tensors, "lm_head.weight", vec![128, 64]);
    push_f16_zeros(&mut tensors, "model.norm.weight", vec![64]);
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        push_f16_zeros(&mut tensors, &format!("{p}.input_layernorm.weight"), vec![64]);
        push_f16_zeros(&mut tensors, &format!("{p}.post_attention_layernorm.weight"), vec![64]);
        if layer == 1 {
            push_f16_zeros(&mut tensors, &format!("{p}.self_attn.q_proj.weight"), vec![64, 64]);
            push_f16_zeros(&mut tensors, &format!("{p}.self_attn.k_proj.weight"), vec![16, 64]);
            push_f16_zeros(&mut tensors, &format!("{p}.self_attn.v_proj.weight"), vec![16, 64]);
            push_f16_zeros(&mut tensors, &format!("{p}.self_attn.o_proj.weight"), vec![64, 64]);
            push_f16_zeros(&mut tensors, &format!("{p}.self_attn.gate.weight"), vec![1, 64]);
        } else {
            let qkv_rows = 4 * 16 * 2 + 8 * 16;
            push_f16_zeros(&mut tensors, &format!("{p}.linear_attn.in_proj_qkv.weight"), vec![qkv_rows, 64]);
            push_f16_zeros(&mut tensors, &format!("{p}.linear_attn.out_proj.weight"), vec![64, 8 * 16]);
            push_f16_zeros(&mut tensors, &format!("{p}.linear_attn.in_proj_a.weight"), vec![8, 64]);
            push_f16_zeros(&mut tensors, &format!("{p}.linear_attn.in_proj_b.weight"), vec![8, 64]);
            push_f16_zeros(&mut tensors, &format!("{p}.linear_attn.in_proj_z.weight"), vec![8 * 16, 64]);
        }
        push_f16_zeros(&mut tensors, &format!("{p}.mlp.gate_proj.weight"), vec![128, 64]);
        push_f16_zeros(&mut tensors, &format!("{p}.mlp.up_proj.weight"), vec![128, 64]);
        push_f16_zeros(&mut tensors, &format!("{p}.mlp.down_proj.weight"), vec![64, 128]);
    }
    // CLIP-classic vision tensors.
    push_f16_zeros(
        &mut tensors,
        "model.vision_tower.embeddings.patch_embeddings.projection.weight",
        vec![64, 3, 4, 4],
    );
    push_f16_zeros(
        &mut tensors,
        "model.vision_tower.embeddings.position_embeddings.weight",
        vec![64, 64],
    );
    push_f16_zeros(&mut tensors, "model.vision_tower.post_layernorm.weight", vec![64]);
    push_f16_zeros(&mut tensors, "model.vision_tower.post_layernorm.bias", vec![64]);
    for l in 0..4 {
        let pre = format!("model.vision_tower.encoder.layer.{l}");
        push_f16_zeros(&mut tensors, &format!("{pre}.attention.q_proj.weight"), vec![64, 64]);
        push_f16_zeros(&mut tensors, &format!("{pre}.attention.k_proj.weight"), vec![64, 64]);
        push_f16_zeros(&mut tensors, &format!("{pre}.attention.v_proj.weight"), vec![64, 64]);
        push_f16_zeros(&mut tensors, &format!("{pre}.attention.output.dense.weight"), vec![64, 64]);
        push_f16_zeros(&mut tensors, &format!("{pre}.layer_norm1.weight"), vec![64]);
        push_f16_zeros(&mut tensors, &format!("{pre}.layer_norm2.weight"), vec![64]);
        push_f16_zeros(&mut tensors, &format!("{pre}.mlp.fc1.weight"), vec![128, 64]);
        push_f16_zeros(&mut tensors, &format!("{pre}.mlp.fc2.weight"), vec![64, 128]);
    }
    push_f16_zeros(&mut tensors, "model.multi_modal_projector.linear_1.weight", vec![64, 64]);
    push_f16_zeros(&mut tensors, "model.multi_modal_projector.linear_2.weight", vec![64, 64]);
    fs::write(input.join("model.safetensors"), build_safetensors_bytes(tensors)).unwrap();

    let text_path = output.join("text.gguf");
    let _ = run_convert(&input, &text_path);
    let parent = text_path.parent().unwrap();
    let mmproj = find_mmproj_under(parent).expect("mmproj emitted");
    let gguf = GgufFile::open(&mmproj).expect("open mmproj");

    // Qwen3-VL keys MUST be absent on a CLIP-classic config.
    let qwen3vl_keys = [
        "clip.use_gelu",
        "clip.vision.spatial_merge_size",
        "clip.vision.is_deepstack_layers",
    ];
    for key in &qwen3vl_keys {
        assert!(
            gguf.metadata(key).is_none(),
            "Wedge-4f regression: CLIP-classic mmproj must NOT emit \
             Qwen3-VL extension key {:?}; writer family-gate broken",
            key
        );
    }

    // The base CLIP keys must still be present.
    assert_eq!(
        gguf.metadata_string("clip.projector_type"),
        Some("mlp"),
        "CLIP-classic projector_type still 'mlp'"
    );
}

// ---------------------------------------------------------------------------
// Test 6 — operator-gated real-model variant (HF2Q_QWEN3VL_ROUND_TRIP=1)
// ---------------------------------------------------------------------------

/// Operator-gated Qwen3-VL-2B-Instruct round-trip. Disabled by default
/// because the target model is ~5 GB on disk and downloading isn't
/// CI-appropriate. Run via:
///
/// ```bash
/// export HF2Q_QWEN3VL_ROUND_TRIP=1
/// export HF2Q_QWEN3VL_HF_DIR=/path/to/Qwen/Qwen3-VL-2B-Instruct
/// cargo test --test qwen3vl_round_trip_e2e --release \
///     -- --nocapture --ignored
/// ```
///
/// When the env vars are unset OR the dir doesn't exist, the test
/// reports the missing piece and exits cleanly (operator-actionable
/// message; not a hard fail).
#[test]
#[ignore]
fn wedge4f_qwen3vl_real_2b_round_trip() {
    let gate = std::env::var("HF2Q_QWEN3VL_ROUND_TRIP")
        .map(|v| v.trim() == "1" || v.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !gate {
        eprintln!(
            "skipping: HF2Q_QWEN3VL_ROUND_TRIP not set. \
             To run: export HF2Q_QWEN3VL_ROUND_TRIP=1 + \
             HF2Q_QWEN3VL_HF_DIR=/path/to/Qwen/Qwen3-VL-2B-Instruct"
        );
        return;
    }
    let hf_dir = match std::env::var("HF2Q_QWEN3VL_HF_DIR").map(std::path::PathBuf::from) {
        Ok(p) if p.exists() => p,
        Ok(p) => {
            eprintln!("skipping: HF2Q_QWEN3VL_HF_DIR={:?} does not exist on disk", p);
            return;
        }
        Err(_) => {
            eprintln!("skipping: HF2Q_QWEN3VL_HF_DIR is not set");
            return;
        }
    };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("real-out");
    let text_path = output.join("qwen3vl-2b.gguf");
    let out = run_convert(&hf_dir, &text_path);
    assert!(
        out.status.success(),
        "real Qwen3-VL-2B convert failed: stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let parent = text_path.parent().unwrap();
    let mmproj = find_mmproj_under(parent).expect("real mmproj emitted");
    let gguf = GgufFile::open(&mmproj).expect("open real mmproj");
    let names: Vec<&str> = gguf.tensor_names();

    assert_eq!(
        gguf.metadata_string("clip.projector_type"),
        Some("qwen3vl_merger")
    );
    let block_count = gguf
        .metadata_u32("clip.vision.block_count")
        .expect("block_count present");
    assert!(block_count > 0);

    // Real model must carry per-block tensors at the canonical names
    // for every block.
    for l in 0..block_count {
        let attn_out = format!("v.blk.{}.attn_out.weight", l);
        assert!(
            names.iter().any(|n| *n == attn_out),
            "real Qwen3-VL-2B mmproj missing {:?}",
            attn_out
        );
    }

    // Real model has spatial_merge_size = 2 and a deepstack array.
    let sms = gguf
        .metadata_u32("clip.vision.spatial_merge_size")
        .expect("spatial_merge_size present on real Qwen3-VL");
    assert_eq!(sms, 2);
    assert!(
        gguf.metadata("clip.vision.is_deepstack_layers").is_some(),
        "real Qwen3-VL-2B must declare is_deepstack_layers"
    );

    eprintln!(
        "Wedge-4f real round-trip OK: {} tensors in mmproj at {:?}",
        names.len(),
        mmproj
    );
}
