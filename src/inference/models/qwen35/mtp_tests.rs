use super::load_mtp_weights_if_present;
use super::super::gpu_full_attn::upload_f32;
use super::super::kv_cache::HybridKvCache;
use super::super::mtp_weights_load::mtp_tensor_names;
use super::super::{default_layer_types, Qwen35Config, Qwen35LayerKind, Qwen35Variant};
use mlx_native::gguf::GgufFile;
use mlx_native::{KernelRegistry, MlxDevice};
use std::io::Write;

struct TestTensor {
    name: &'static str,
    dims: Vec<u64>,
    data: Vec<f32>,
}

fn tiny_cfg(mtp_layers: u32) -> Qwen35Config {
    Qwen35Config {
        variant: Qwen35Variant::Dense,
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 32,
        linear_num_key_heads: 1,
        linear_num_value_heads: 1,
        linear_key_head_dim: 32,
        linear_value_head_dim: 32,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 2,
        layer_types: default_layer_types(2, 2),
        partial_rotary_factor: 1.0,
        rope_theta: 1_000_000.0,
        rotary_dim: 32,
        mrope_section: [8, 8, 8, 8],
        mrope_interleaved: true,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 128,
        vocab_size: 64,
        attn_output_gate: true,
        mtp_num_hidden_layers: mtp_layers,
        intermediate_size: Some(32),
        moe: None,
    }
}

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

fn zeros(n: usize) -> Vec<f32> {
    vec![0.0; n]
}

fn tiny_tensors() -> Vec<TestTensor> {
    let h = 32usize;
    let v = 64usize;
    let m = 32usize;
    vec![
        TestTensor { name: "blk.2.nextn.enorm.weight", dims: vec![h as u64], data: ones(h) },
        TestTensor { name: "blk.2.nextn.hnorm.weight", dims: vec![h as u64], data: ones(h) },
        TestTensor { name: "blk.2.nextn.eh_proj.weight", dims: vec![(2 * h) as u64, h as u64], data: ones(2 * h * h) },
        TestTensor { name: "blk.2.nextn.embed_tokens.weight", dims: vec![h as u64, v as u64], data: zeros(v * h) },
        TestTensor { name: "blk.2.nextn.shared_head_norm.weight", dims: vec![h as u64], data: ones(h) },
        TestTensor { name: "blk.2.nextn.shared_head_head.weight", dims: vec![h as u64, v as u64], data: zeros(v * h) },
        TestTensor { name: "blk.2.attn_norm.weight", dims: vec![h as u64], data: ones(h) },
        TestTensor { name: "blk.2.post_attention_norm.weight", dims: vec![h as u64], data: ones(h) },
        TestTensor { name: "blk.2.attn_q.weight", dims: vec![h as u64, h as u64], data: zeros(h * h) },
        TestTensor { name: "blk.2.attn_k.weight", dims: vec![h as u64, h as u64], data: zeros(h * h) },
        TestTensor { name: "blk.2.attn_v.weight", dims: vec![h as u64, h as u64], data: zeros(h * h) },
        TestTensor { name: "blk.2.attn_output.weight", dims: vec![h as u64, h as u64], data: zeros(h * h) },
        TestTensor { name: "blk.2.attn_q_norm.weight", dims: vec![h as u64], data: ones(h) },
        TestTensor { name: "blk.2.attn_k_norm.weight", dims: vec![h as u64], data: ones(h) },
        TestTensor { name: "blk.2.ffn_gate.weight", dims: vec![h as u64, m as u64], data: zeros(m * h) },
        TestTensor { name: "blk.2.ffn_up.weight", dims: vec![h as u64, m as u64], data: zeros(m * h) },
        TestTensor { name: "blk.2.ffn_down.weight", dims: vec![m as u64, h as u64], data: zeros(h * m) },
    ]
}

fn write_gguf(path: &std::path::Path, tensors: &[TestTensor]) {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());

    let mut offset = 0u64;
    let mut offsets = Vec::with_capacity(tensors.len());
    for t in tensors {
        while offset % 32 != 0 {
            offset += 1;
        }
        offsets.push(offset);
        offset += (t.data.len() * 4) as u64;
    }

    for (t, off) in tensors.iter().zip(offsets.iter()) {
        buf.extend_from_slice(&(t.name.len() as u64).to_le_bytes());
        buf.extend_from_slice(t.name.as_bytes());
        buf.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&off.to_le_bytes());
    }
    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    let data_start = buf.len();
    for (t, off) in tensors.iter().zip(offsets.iter()) {
        while (buf.len() - data_start) < *off as usize {
            buf.push(0);
        }
        for f in &t.data {
            buf.extend_from_slice(&f.to_le_bytes());
        }
    }

    let mut f = std::fs::File::create(path).expect("create gguf");
    f.write_all(&buf).expect("write gguf");
    f.flush().expect("flush gguf");
}

fn try_device() -> Option<MlxDevice> {
    match MlxDevice::new() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping MTP GPU test: {e}");
            None
        }
    }
}

#[test]
fn mtp_absent_scan_returns_empty() {
    let tmp = std::env::temp_dir().join(format!("mtp_absent_{}.gguf", std::process::id()));
    write_gguf(
        &tmp,
        &[TestTensor { name: "blk.0.attn_norm.weight", dims: vec![32], data: ones(32) }],
    );
    let gguf = GgufFile::open(&tmp).expect("open");
    assert!(mtp_tensor_names(&gguf, 2).is_empty());
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn mtp_loads_gpu_weights_from_synthetic_gguf() {
    let Some(device) = try_device() else { return };
    let tmp = std::env::temp_dir().join(format!("mtp_present_{}.gguf", std::process::id()));
    write_gguf(&tmp, &tiny_tensors());
    let gguf = GgufFile::open(&tmp).expect("open");
    let mtp = load_mtp_weights_if_present(&gguf, &tiny_cfg(1), &device)
        .expect("load")
        .expect("some");
    assert_eq!(mtp.layer_index, 2);
    assert_eq!(mtp.hidden_size, 32);
    assert_eq!(mtp.vocab_size, 64);
    assert!(!mtp.is_empty());
    assert!(mtp.has_tensor_suffix("enorm.weight"));
    assert!(mtp.has_tensor_suffix("attn_q.weight"));
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn mtp_forward_draft_returns_logits() {
    let Some(device) = try_device() else { return };
    let tmp = std::env::temp_dir().join(format!("mtp_forward_{}.gguf", std::process::id()));
    write_gguf(&tmp, &tiny_tensors());
    let gguf = GgufFile::open(&tmp).expect("open");
    let cfg = tiny_cfg(1);
    let mtp = load_mtp_weights_if_present(&gguf, &cfg, &device)
        .expect("load")
        .expect("some");
    let mut registry = KernelRegistry::new();
    let mut kv = HybridKvCache::new(&cfg, &device, 16, 1).expect("cache");
    assert!(kv.mtp_slot.is_some());
    let prev = upload_f32(&vec![0.0; 32], &device).expect("prev");
    let embed = upload_f32(&vec![0.0; 32], &device).expect("embed");
    let logits = mtp
        .forward_draft(&prev, &embed, &mut kv, &[0, 0, 0, 0], &device, &mut registry, &cfg)
        .expect("forward");
    assert_eq!(logits.element_count(), 64);
    std::fs::remove_file(&tmp).ok();
}

#[test]
#[ignore]
fn mtp_on_real_apex_returns_none() {
    let Some(device) = try_device() else { return };
    let path = std::path::PathBuf::from(
        "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/\
         qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf",
    );
    if !path.exists() {
        eprintln!("skipping: apex GGUF not at expected path");
        return;
    }
    let gguf = match GgufFile::open(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let cfg = Qwen35Config::from_gguf(&gguf).expect("cfg");
    let result = load_mtp_weights_if_present(&gguf, &cfg, &device).expect("load_mtp");
    assert!(result.is_none(), "apex GGUF should have MTP stripped");
}

#[test]
fn test_cfg_layer_types_not_all_full() {
    let cfg = tiny_cfg(1);
    assert_eq!(cfg.layer_types[0], Qwen35LayerKind::LinearAttention);
    assert_eq!(cfg.layer_types[1], Qwen35LayerKind::FullAttention);
}
