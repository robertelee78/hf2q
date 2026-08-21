use super::super::gpu_full_attn::{download_f32, upload_f32};
use super::super::kv_cache::HybridKvCache;
use super::super::mtp::MtpFfnKind;
use super::super::mtp_weights_load::mtp_tensor_names;
use super::super::{default_layer_types, Qwen35Config, Qwen35LayerKind, Qwen35Variant};
use super::{load_mtp_weights_if_present, shifted_nextn_copy_plan, upload_i32};
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
        mtp_use_dedicated_embeddings: true,
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
        TestTensor {
            name: "blk.2.nextn.enorm.weight",
            dims: vec![h as u64],
            data: ones(h),
        },
        TestTensor {
            name: "blk.2.nextn.hnorm.weight",
            dims: vec![h as u64],
            data: ones(h),
        },
        TestTensor {
            name: "blk.2.nextn.eh_proj.weight",
            dims: vec![(2 * h) as u64, h as u64],
            data: ones(2 * h * h),
        },
        TestTensor {
            name: "blk.2.nextn.embed_tokens.weight",
            dims: vec![h as u64, v as u64],
            data: zeros(v * h),
        },
        TestTensor {
            name: "blk.2.nextn.shared_head_norm.weight",
            dims: vec![h as u64],
            data: ones(h),
        },
        TestTensor {
            name: "blk.2.nextn.shared_head_head.weight",
            dims: vec![h as u64, v as u64],
            data: zeros(v * h),
        },
        TestTensor {
            name: "blk.2.attn_norm.weight",
            dims: vec![h as u64],
            data: ones(h),
        },
        TestTensor {
            name: "blk.2.post_attention_norm.weight",
            dims: vec![h as u64],
            data: ones(h),
        },
        TestTensor {
            name: "blk.2.attn_q.weight",
            dims: vec![h as u64, h as u64],
            data: zeros(h * h),
        },
        TestTensor {
            name: "blk.2.attn_k.weight",
            dims: vec![h as u64, h as u64],
            data: zeros(h * h),
        },
        TestTensor {
            name: "blk.2.attn_v.weight",
            dims: vec![h as u64, h as u64],
            data: zeros(h * h),
        },
        TestTensor {
            name: "blk.2.attn_output.weight",
            dims: vec![h as u64, h as u64],
            data: zeros(h * h),
        },
        TestTensor {
            name: "blk.2.attn_q_norm.weight",
            dims: vec![h as u64],
            data: ones(h),
        },
        TestTensor {
            name: "blk.2.attn_k_norm.weight",
            dims: vec![h as u64],
            data: ones(h),
        },
        TestTensor {
            name: "blk.2.ffn_gate.weight",
            dims: vec![h as u64, m as u64],
            data: zeros(m * h),
        },
        TestTensor {
            name: "blk.2.ffn_up.weight",
            dims: vec![h as u64, m as u64],
            data: zeros(m * h),
        },
        TestTensor {
            name: "blk.2.ffn_down.weight",
            dims: vec![m as u64, h as u64],
            data: zeros(h * m),
        },
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
fn prompt_catchup_shifts_target_hidden_right() {
    let cold = shifted_nextn_copy_plan(4, 8, false).expect("cold plan");
    assert_eq!(cold.pending, None, "cold row zero remains zero-initialized");
    let prefix = cold.target_prefix.expect("rows 1..3 copy target rows 0..2");
    assert_eq!(prefix.src_offset, 0);
    assert_eq!(prefix.dst_offset, 8);
    assert_eq!(prefix.count, 24);

    let resumed = shifted_nextn_copy_plan(4, 8, true).expect("resumed plan");
    let pending = resumed.pending.expect("saved prior target row");
    assert_eq!(pending.src_offset, 0);
    assert_eq!(pending.dst_offset, 0);
    assert_eq!(pending.count, 8);
    assert_eq!(resumed.target_prefix, cold.target_prefix);
}

#[test]
fn mtp_absent_scan_returns_empty() {
    let tmp = std::env::temp_dir().join(format!("mtp_absent_{}.gguf", std::process::id()));
    write_gguf(
        &tmp,
        &[TestTensor {
            name: "blk.0.attn_norm.weight",
            dims: vec![32],
            data: ones(32),
        }],
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
    // ADR-013 P14 follow-up: dedicated path populates the buffer.
    assert!(
        mtp.embed_tokens.is_some(),
        "dedicated_embeddings=true must yield Some(embed_tokens) buffer"
    );
    std::fs::remove_file(&tmp).ok();
}

/// ADR-013 P14 follow-up (2026-04-30): Qwen3.6 27B + 35B-A3B share the main
/// model's `token_embd.weight`; convert correctly skips emitting
/// `blk.{N}.nextn.embed_tokens.weight`. The loader must succeed (not bail) when
/// `mtp_use_dedicated_embeddings=false` AND the dedicated tensor is absent, and
/// must populate `embed_tokens = None` to signal the shared path.
#[test]
fn mtp_loads_with_shared_embeddings_when_flag_false_and_tensor_absent() {
    let Some(device) = try_device() else { return };
    // tiny_tensors() includes the dedicated embed_tokens tensor; strip it for
    // the shared-embeddings scenario.
    let tensors: Vec<TestTensor> = tiny_tensors()
        .into_iter()
        .filter(|t| t.name != "blk.2.nextn.embed_tokens.weight")
        .collect();
    let tmp = std::env::temp_dir().join(format!("mtp_shared_{}.gguf", std::process::id()));
    write_gguf(&tmp, &tensors);
    let gguf = GgufFile::open(&tmp).expect("open");
    let mut cfg = tiny_cfg(1);
    cfg.mtp_use_dedicated_embeddings = false;
    let mtp = load_mtp_weights_if_present(&gguf, &cfg, &device)
        .expect("loader must succeed when flag=false and tensor absent")
        .expect("MtpWeights present (mtp_num_hidden_layers=1)");
    assert_eq!(mtp.layer_index, 2);
    assert_eq!(mtp.hidden_size, 32);
    // Shared-embeddings path: no buffer materialised, main model's token_embd
    // is reused via Qwen35Model::token_embd at draft time.
    assert!(
        mtp.embed_tokens.is_none(),
        "mtp_use_dedicated_embeddings=false must yield None (shared with main)"
    );
    // Other MTP tensors still load correctly.
    assert!(mtp.has_tensor_suffix("enorm.weight"));
    assert!(mtp.has_tensor_suffix("attn_q.weight"));
    // The dedicated-embeddings tensor name is NOT in the discovered names.
    assert!(
        !mtp.loaded_tensor_names
            .iter()
            .any(|n| n == "blk.2.nextn.embed_tokens.weight"),
        "dedicated tensor must not appear in loaded_tensor_names"
    );
    std::fs::remove_file(&tmp).ok();
}

/// ADR-013 P14 follow-up: belt-and-suspenders for convert-side inconsistency.
/// If a GGUF carries both `mtp_use_dedicated_embeddings=False` AND the dedicated
/// tensor is present, refuse rather than silently dropping the tensor.
#[test]
fn mtp_rejects_inconsistent_shared_flag_with_dedicated_tensor_present() {
    let Some(device) = try_device() else { return };
    let tmp = std::env::temp_dir().join(format!("mtp_inconsistent_{}.gguf", std::process::id()));
    write_gguf(&tmp, &tiny_tensors()); // dedicated tensor present
    let gguf = GgufFile::open(&tmp).expect("open");
    let mut cfg = tiny_cfg(1);
    cfg.mtp_use_dedicated_embeddings = false; // inconsistent with tensor presence
    let result = load_mtp_weights_if_present(&gguf, &cfg, &device);
    assert!(
        result.is_err(),
        "loader must refuse when flag=false but dedicated tensor present"
    );
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
    let prev_values: Vec<f32> = (1..=32).map(|value| value as f32).collect();
    let prev = upload_f32(&prev_values, &device).expect("prev");
    let embed = upload_f32(&vec![0.0; 32], &device).expect("embed");
    let (logits, nextn_hidden) = mtp
        .forward_draft_for_token(
            &prev,
            0,
            &embed,
            &mut kv,
            crate::serve::multi_seq_kv::SlotId(0),
            &[0, 0, 0, 0],
            &device,
            &mut registry,
            &cfg,
        )
        .expect("forward");
    assert_eq!(logits.element_count(), 64);
    let nextn = download_f32(&nextn_hidden).expect("download normalized MTP hidden");
    let mean_square = nextn.iter().map(|value| value * value).sum::<f32>() / nextn.len() as f32;
    assert!(
        (mean_square - 1.0).abs() < 1e-3,
        "MTP chained hidden must be post-shared-head RMSNorm; mean_square={mean_square}"
    );
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn mtp_fused_greedy_head_matches_logits_argmax() {
    let Some(device) = try_device() else { return };
    let tmp = std::env::temp_dir().join(format!("mtp_fused_greedy_{}.gguf", std::process::id()));
    write_gguf(&tmp, &tiny_tensors());
    let gguf = GgufFile::open(&tmp).expect("open");
    let cfg = tiny_cfg(1);
    let mtp = load_mtp_weights_if_present(&gguf, &cfg, &device)
        .expect("load")
        .expect("some");
    let mut registry = KernelRegistry::new();
    let mut logits_kv = HybridKvCache::new(&cfg, &device, 16, 1).expect("logits cache");
    let mut fused_kv = HybridKvCache::new(&cfg, &device, 16, 1).expect("fused cache");
    let prev_values: Vec<f32> = (1..=32).map(|value| value as f32).collect();
    let prev = upload_f32(&prev_values, &device).expect("prev");
    let embed = upload_f32(&vec![0.0; 32], &device).expect("embed");

    let (logits, reference_hidden) = mtp
        .forward_draft_for_token(
            &prev,
            0,
            &embed,
            &mut logits_kv,
            crate::serve::multi_seq_kv::SlotId(0),
            &[0, 0, 0, 0],
            &device,
            &mut registry,
            &cfg,
        )
        .expect("logits forward");
    let logits = download_f32(&logits).expect("download logits");
    let expected = logits
        .iter()
        .enumerate()
        .fold(
            (0u32, f32::NEG_INFINITY),
            |(best_i, best_v), (i, &value)| {
                if value > best_v {
                    (i as u32, value)
                } else {
                    (best_i, best_v)
                }
            },
        )
        .0;

    let (actual, fused_hidden) = mtp
        .forward_draft_greedy_for_token(
            &prev,
            0,
            &embed,
            &mut fused_kv,
            crate::serve::multi_seq_kv::SlotId(0),
            &[0, 0, 0, 0],
            &device,
            &mut registry,
            &cfg,
        )
        .expect("fused greedy forward");
    assert_eq!(actual, expected);
    assert_eq!(
        download_f32(&fused_hidden).expect("download fused hidden"),
        download_f32(&reference_hidden).expect("download reference hidden"),
        "fusing argmax must not change the normalized hidden carried into the next draft"
    );
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn prompt_catchup_aligns_target_and_mtp_cursors() {
    let Some(device) = try_device() else { return };
    let tmp = std::env::temp_dir().join(format!("mtp_catchup_{}.gguf", std::process::id()));
    write_gguf(&tmp, &tiny_tensors());
    let gguf = GgufFile::open(&tmp).expect("open");
    let cfg = tiny_cfg(1);
    let mtp = load_mtp_weights_if_present(&gguf, &cfg, &device)
        .expect("load")
        .expect("some");
    let mut registry = KernelRegistry::new();
    let mut kv = HybridKvCache::new(&cfg, &device, 16, 1).expect("cache");
    let target_nextn = upload_f32(&vec![0.25; 3 * 32], &device).expect("target nextn");
    let shared_embed = upload_f32(&vec![0.0; 3 * 32], &device).expect("shared embed");

    mtp.process_target_batch(
        &[0, 1, 2],
        None,
        &target_nextn,
        &shared_embed,
        &mut kv,
        crate::serve::multi_seq_kv::SlotId(0),
        &[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        &device,
        &mut registry,
        &cfg,
    )
    .expect("MTP full-prompt catch-up");
    for full in &mut kv.full_attn {
        full.current_len[0] = 3;
    }
    kv.validate_speculative_cursors_for_slot(crate::serve::multi_seq_kv::SlotId(0), 3)
        .expect("target/MTP cursor equality after prompt catch-up");
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn mtp_kv_only_append_matches_full_attention_cache_prefix() {
    let Some(device) = try_device() else { return };
    let tmp = std::env::temp_dir().join(format!("mtp_kv_only_{}.gguf", std::process::id()));
    write_gguf(&tmp, &tiny_tensors());
    let gguf = GgufFile::open(&tmp).expect("open");
    let cfg = tiny_cfg(1);
    let mtp = load_mtp_weights_if_present(&gguf, &cfg, &device)
        .expect("load")
        .expect("some");
    let projected_values: Vec<f32> = (0..3 * 32)
        .map(|index| ((index as f32) * 0.037).sin())
        .collect();
    let projected = upload_f32(&projected_values, &device).expect("projected");
    let positions = upload_i32(&[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], &device).expect("positions");
    let slot = crate::serve::multi_seq_kv::SlotId(0);
    let mut full_registry = KernelRegistry::new();
    let mut kv_only_registry = KernelRegistry::new();
    let mut full_cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("full cache");
    let mut kv_only_cache = HybridKvCache::new(&cfg, &device, 16, 1).expect("KV-only cache");

    let _ = mtp
        .forward_full_attention(
            &projected,
            &positions,
            &mut full_cache,
            slot,
            3,
            &device,
            &mut full_registry,
            &cfg,
        )
        .expect("full attention reference");
    let mut full_drain = device.command_encoder().expect("full drain encoder");
    full_drain.commit_and_wait().expect("full drain");

    mtp.append_attention_kv(
        &projected,
        &positions,
        &mut kv_only_cache,
        slot,
        3,
        &device,
        &mut kv_only_registry,
        &cfg,
    )
    .expect("KV-only append");
    let mut kv_only_drain = device.command_encoder().expect("KV-only drain encoder");
    kv_only_drain.commit_and_wait().expect("KV-only drain");

    let full_slot = full_cache.mtp_slot.as_ref().expect("full MTP slot");
    let kv_only_slot = kv_only_cache.mtp_slot.as_ref().expect("KV-only MTP slot");
    assert_eq!(full_slot.current_len, kv_only_slot.current_len);
    assert_eq!(
        full_slot
            .k
            .as_ref()
            .expect("full K")
            .as_slice::<f32>()
            .expect("full K bytes"),
        kv_only_slot
            .k
            .as_ref()
            .expect("KV-only K")
            .as_slice::<f32>()
            .expect("KV-only K bytes")
    );
    assert_eq!(
        full_slot
            .v
            .as_ref()
            .expect("full V")
            .as_slice::<f32>()
            .expect("full V bytes"),
        kv_only_slot
            .v
            .as_ref()
            .expect("KV-only V")
            .as_slice::<f32>()
            .expect("KV-only V bytes")
    );
    std::fs::remove_file(&tmp).ok();
}

// ADR-017 iter-3.6 follow-up (2026-05-05): #[ignore] removed; path
// fixed to actual fixture name (APEX-Q5_K_M.gguf). Test runtime-skips
// when the artefact is absent.
#[test]
fn mtp_on_real_apex_returns_none() {
    let Some(device) = try_device() else { return };
    let path = std::path::PathBuf::from(
        "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/\
         APEX-Q5_K_M.gguf",
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

/// ADR-034 P3.1 regression gate (2026-05-21 at HEAD `1cfefea0`):
///
/// Canonical Qwen 3.5 35B-A3B MoE-MTP GGUFs emit 8 MoE-style tensors at the
/// MTP block (`blk.40.ffn_gate_inp/ffn_{gate,up,down}_exps/ffn_{gate,up,down,gate_inp}_shexp`).
/// Before HEAD `afbf5684` the MTP loader hardcoded the dense FFN tensor
/// names and crashed at load time. This test asserts the structural
/// outcome of the fix:
///
///   1. `load_mtp_weights_if_present` succeeds against a canonical MoE-MTP GGUF
///   2. The resulting `MtpWeights::ffn_kind()` returns `MtpFfnKind::Moe`
///   3. `loaded_tensor_names` contains MoE expert tensor suffixes (so
///      `has_tensor_suffix("ffn_gate_exps.weight")` reports them)
///   4. `layer_index` matches the model's MTP block index (40 for 35B-A3B)
///
/// Skips when the canonical fixture isn't present locally (CI / contributor
/// environments without the 20 GB GGUF on disk). The empirical 8/8 quant
/// sweep in the cover commits validates the runtime path; this test locks
/// in the LOAD-side structural contract as a CI-side gate.
#[test]
fn mtp_loads_canonical_moe_mtp_q4_k_m_with_moe_variant_2026_05_21() {
    let Some(device) = try_device() else { return };
    let path = std::path::PathBuf::from(
        "/opt/hf2q/cache/byte_cmp/Qwen-Qwen3.5-35B-A3B_canonical_q4_k_m.gguf",
    );
    if !path.exists() {
        eprintln!(
            "skipping: canonical MoE-MTP fixture not at {}; this test \
             requires the 20 GB Qwen 3.5 35B-A3B Q4_K_M GGUF to exercise \
             the MoE inner-FFN load path",
            path.display()
        );
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
    let result = load_mtp_weights_if_present(&gguf, &cfg, &device)
        .expect("MoE-MTP loader must succeed on canonical Q4_K_M GGUF");
    let mtp = result.expect("canonical Q4_K_M ships MTP weights");

    // Structural assertions on the MoE-MTP branch.
    assert_eq!(
        mtp.ffn_kind(),
        MtpFfnKind::Moe,
        "canonical Qwen 3.5 35B-A3B MoE-MTP must load via MtpFfnWeightsGpu::Moe branch"
    );
    assert_eq!(
        mtp.layer_index, cfg.num_hidden_layers,
        "MTP block sits at blk.{{num_hidden_layers}}"
    );

    // MoE-specific tensor names must be reflected in loaded_tensor_names
    // (the mtp_tensor_names membership set was extended in the cover fix).
    assert!(
        mtp.has_tensor_suffix("ffn_gate_exps.weight"),
        "MoE expert gate tensor must be tracked"
    );
    assert!(
        mtp.has_tensor_suffix("ffn_up_exps.weight"),
        "MoE expert up tensor must be tracked"
    );
    assert!(
        mtp.has_tensor_suffix("ffn_down_exps.weight"),
        "MoE expert down tensor must be tracked"
    );
    assert!(
        mtp.has_tensor_suffix("ffn_gate_inp.weight"),
        "MoE router must be tracked"
    );
    assert!(
        mtp.has_tensor_suffix("ffn_gate_inp_shexp.weight"),
        "MoE shared-expert sigmoid gate must be tracked"
    );

    // The dense suffix MUST NOT appear in a MoE-MTP GGUF — guard against
    // silent fall-through if a future loader change accidentally pulls
    // dense names back in.
    assert!(
        !mtp.has_tensor_suffix("ffn_gate.weight"),
        "MoE-MTP GGUF must not advertise dense ffn_gate.weight at the MTP block"
    );
}

/// Companion to the MoE-MTP gate: locks in the dense-MTP path against a
/// regression. Qwen 3.6 27B emits a dense SwiGLU FFN at the MTP block
/// (`blk.64.ffn_{gate,up,down}.weight`). The loader must take the Dense
/// branch and `ffn_kind()` must report Dense.
///
/// Skips when the 27 GB Qwen 3.6 27B MTP GGUF isn't present locally.
#[test]
fn mtp_loads_canonical_dense_mtp_q8_0_with_dense_variant_2026_05_21() {
    let Some(device) = try_device() else { return };
    let path =
        std::path::PathBuf::from("/opt/hf2q/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q8_0-mtp.gguf");
    if !path.exists() {
        eprintln!(
            "skipping: canonical dense MTP fixture not at {}",
            path.display()
        );
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
    let result = load_mtp_weights_if_present(&gguf, &cfg, &device)
        .expect("dense MTP loader must succeed on canonical Qwen 3.6 27B GGUF");
    let mtp = result.expect("canonical Qwen 3.6 27B Q8_0 ships MTP weights");

    assert_eq!(
        mtp.ffn_kind(),
        MtpFfnKind::Dense,
        "canonical Qwen 3.6 27B dense-MTP must load via native DenseQ branch"
    );
    assert_eq!(mtp.layer_index, cfg.num_hidden_layers);
    assert!(mtp.has_tensor_suffix("ffn_gate.weight"));
    assert!(mtp.has_tensor_suffix("ffn_up.weight"));
    assert!(mtp.has_tensor_suffix("ffn_down.weight"));
    // MoE suffixes MUST NOT appear in a dense MTP GGUF.
    assert!(!mtp.has_tensor_suffix("ffn_gate_exps.weight"));
    assert!(!mtp.has_tensor_suffix("ffn_gate_inp.weight"));
}
