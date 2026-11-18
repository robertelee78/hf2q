use super::super::gpu_full_attn::{download_f32, upload_f32};
use super::super::kv_cache::HybridKvCache;
use super::super::mtp::MtpFfnKind;
use super::super::mtp_weights_load::mtp_tensor_names;
use super::super::{default_layer_types, Qwen35Config, Qwen35LayerKind, Qwen35Variant};
use super::{
    load_mtp_weights_if_present, load_mtp_weights_if_present_with_shared_head,
    shifted_nextn_copy_plan, upload_i32,
};
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
    let mut tensors: Vec<TestTensor> = tiny_tensors()
        .into_iter()
        .filter(|t| t.name != "blk.2.nextn.embed_tokens.weight")
        .collect();
    tensors.push(TestTensor {
        name: "token_embd.weight",
        dims: vec![32, 64],
        data: zeros(64 * 32),
    });
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

#[test]
fn mtp_shared_head_falls_back_to_tied_main_token_embedding() {
    let Some(device) = try_device() else { return };
    let h = 32usize;
    let vocab = 64usize;
    let mut tensors: Vec<TestTensor> = tiny_tensors()
        .into_iter()
        .filter(|tensor| {
            tensor.name != "blk.2.nextn.embed_tokens.weight"
                && tensor.name != "blk.2.nextn.shared_head_head.weight"
        })
        .collect();
    tensors.push(TestTensor {
        name: "token_embd.weight",
        dims: vec![h as u64, vocab as u64],
        data: zeros(vocab * h),
    });
    let tmp = std::env::temp_dir().join(format!("mtp_tied_head_{}.gguf", std::process::id()));
    write_gguf(&tmp, &tensors);
    let gguf = GgufFile::open(&tmp).expect("open");
    let mut cfg = tiny_cfg(1);
    cfg.mtp_use_dedicated_embeddings = false;
    let mtp = load_mtp_weights_if_present(&gguf, &cfg, &device)
        .expect("tied main token embedding is the shared output head")
        .expect("MTP weights");
    assert_eq!(mtp.vocab_size, 64);
    assert!(mtp.embed_tokens.is_none());
    assert_eq!(mtp.shared_head_head.element_count(), vocab * h);
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn mtp_shared_head_borrows_the_supplied_main_buffer_without_a_second_allocation() {
    use crate::serve::forward_mlx_shared::MlxQWeight;
    use crate::serve::gpu::QuantWeightInfo;
    use mlx_native::metal::foreign_types::ForeignType;
    use mlx_native::ops::quantized_matmul_ggml::GgmlType;
    use mlx_native::DType;

    let Some(device) = try_device() else { return };
    let h = 32usize;
    let vocab = 64usize;
    let mut tensors: Vec<TestTensor> = tiny_tensors()
        .into_iter()
        .filter(|tensor| {
            tensor.name != "blk.2.nextn.embed_tokens.weight"
                && tensor.name != "blk.2.nextn.shared_head_head.weight"
        })
        .collect();
    tensors.push(TestTensor {
        name: "token_embd.weight",
        dims: vec![h as u64, vocab as u64],
        data: zeros(vocab * h),
    });
    let tmp = std::env::temp_dir().join(format!("mtp_supplied_head_{}.gguf", std::process::id()));
    write_gguf(&tmp, &tensors);
    let gguf = GgufFile::open(&tmp).expect("open");
    let mut cfg = tiny_cfg(1);
    cfg.mtp_use_dedicated_embeddings = false;

    let bytes = vocab * h / 32 * 34;
    let supplied = MlxQWeight {
        buffer: device
            .alloc_buffer(bytes, DType::U8, vec![bytes])
            .expect("allocate supplied Q8_0 head"),
        info: QuantWeightInfo {
            ggml_dtype: GgmlType::Q8_0,
            rows: vocab,
            cols: h,
        },
        affine: None,
        f16_shadow: None,
        decode_record_q6k_m1: std::sync::OnceLock::new(),
    };
    let supplied_ptr = supplied.buffer.metal_buffer().as_ptr();
    let mtp = load_mtp_weights_if_present_with_shared_head(&gguf, &cfg, &device, Some(&supplied))
        .expect("supplied main head is valid")
        .expect("MTP weights");

    assert_eq!(mtp.shared_head_head.metal_buffer().as_ptr(), supplied_ptr);
    assert_eq!(mtp.shared_head_head_ggml_type, GgmlType::Q8_0);
    assert_eq!(mtp.vocab_size, vocab as u32);
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
fn catchup_real_roots_pool_retention_and_every_submitted_command_are_checked() {
    use super::super::carry_memory::{self, CarryOwners, CatchupPlan, WorkerScope};
    use super::super::decode_pool;
    use super::super::encoder_stage::FailSubmitted;
    use mlx_native::DType;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Some(device) = try_device() else { return };
    for dedicated in [false, true] {
        for tq in [false, true] {
            let mut cfg = tiny_cfg(1);
            cfg.mtp_use_dedicated_embeddings = dedicated;
            cfg.head_dim = 256;
            cfg.rotary_dim = 256;
            cfg.mrope_section = [64; 4];
            let mut tensors = tiny_tensors();
            for tensor in &mut tensors {
                match tensor.name {
                    "blk.2.attn_q.weight" | "blk.2.attn_k.weight" | "blk.2.attn_v.weight" => {
                        tensor.dims = vec![32, 256];
                        tensor.data = zeros(32 * 256);
                    }
                    "blk.2.attn_output.weight" => {
                        tensor.dims = vec![256, 32];
                        tensor.data = zeros(256 * 32);
                    }
                    "blk.2.attn_q_norm.weight" | "blk.2.attn_k_norm.weight" => {
                        tensor.dims = vec![256];
                        tensor.data = ones(256);
                    }
                    _ => {}
                }
            }
            if !dedicated {
                tensors.retain(|tensor| tensor.name != "blk.2.nextn.embed_tokens.weight");
                tensors.push(TestTensor {
                    name: "token_embd.weight", dims: vec![32, 64], data: zeros(32 * 64),
                });
            }
            let path = std::env::temp_dir().join(format!(
                "mtp_carry_{}_{}_{}.gguf", std::process::id(), dedicated, tq,
            ));
            write_gguf(&path, &tensors);
            let gguf = GgufFile::open(&path).unwrap();
            let mtp = load_mtp_weights_if_present(&gguf, &cfg, &device).unwrap().unwrap();
            let old_parent = device.alloc_buffer(8 * 32 * 4, DType::F32, vec![8, 32]).unwrap();
            let old_row = old_parent.slice_view(7 * 32 * 4, 32);
            drop(old_parent);
            let mut owners = CarryOwners::default();
            owners.include(&old_row);
            let retained = owners.bytes();
            let plan = CatchupPlan::new(&cfg, 3, dedicated);
            // The real serving pre-allocation check must fail one byte below
            // the promise, without issuing a target allocation/command.
            {
                let _worker = WorkerScope::open(
                    owners, retained, plan.compose(0, retained, 0) - 1,
                );
                assert!(carry_memory::commit_target(&cfg, 3).is_err());
            }
            // Each submitted stage can fail independently; the final drain
            // still checks all four, including stages behind that failure.
            for fail in 0..=4 {
                let _pooled_outputs = decode_pool::StepPooledOutputs::enter(fail % 2 == 0);
                decode_pool::checked_reset_decode_pool().unwrap();
                let mut owners = CarryOwners::default();
                owners.include(&old_row);
                let _worker = WorkerScope::open(
                    owners, retained, plan.compose(0, retained, 0),
                );
                carry_memory::commit_target(&cfg, 3).expect("exact phase promise");
                let target = upload_f32(&vec![0.25; 3 * 32], &device).unwrap();
                let shared = upload_f32(&vec![0.0; 3 * 32], &device).unwrap();
                let mut cache = HybridKvCache::new_with_options(&cfg, &device, 16, 1, tq).unwrap();
                let mut registry = KernelRegistry::new();
                let injected = FailSubmitted::nth("mtp.", fail);
                let result = mtp.process_target_batch(
                    &[0, 1, 2], Some(&old_row), &target, &shared, &mut cache,
                    crate::serve::multi_seq_kv::SlotId(0),
                    &[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                    &device, &mut registry, &cfg,
                );
                assert_eq!(injected.seen(), (fail < 4, 4));
                if fail < 4 {
                    assert!(result.is_err(), "submitted command {fail} must surface");
                } else {
                    result.expect("all submitted stages checked");
                    let (actual, promised) = carry_memory::last_phase().unwrap();
                    assert!(actual <= promised);
                    assert!(actual > retained + target.byte_len() as u64 + shared.byte_len() as u64);
                    assert_eq!(cache.mtp_slot.as_ref().unwrap().current_len[0], 3);
                }
                let pool = decode_pool::worker_memory_snapshot().unwrap().1;
                assert!(pool.retained.bytes > 0, "pooled norm/rotation roots remain owned");
            }
            decode_pool::checked_reset_decode_pool().unwrap();
            assert!(decode_pool::worker_memory_snapshot().unwrap().1.retained.bytes > 0);
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn quantized_dedicated_catchup_keeps_token_ids_at_power_of_two_geometry() {
    use super::super::carry_memory::{self, CarryOwners, CatchupPlan, WorkerScope};
    use super::super::decode_pool::{checked_reset_decode_pool, StepPooledOutputs};
    use super::super::encoder_stage::FailSubmitted;
    use super::super::prefill_prepared::ScratchCapsForTest;
    use super::super::in_memory_loader::quantize_f32_to_q8_0_buffer;
    use crate::serve::forward_mlx_shared::MlxQWeight;
    use crate::serve::gpu::QuantWeightInfo;
    use mlx_native::ops::quantized_matmul_ggml::GgmlType;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Some(device) = try_device() else { return };
    assert!(!matches!(std::env::var("MLX_UNRETAINED_REFS").as_deref(), Ok("1")),
        "this regression exercises the normal asynchronous retained-reference route");
    const ROWS: usize = 64;
    const HIDDEN: usize = 256;
    let mut cfg = tiny_cfg(1);
    cfg.hidden_size = HIDDEN as u32;
    cfg.head_dim = HIDDEN as u32;
    cfg.rotary_dim = HIDDEN as u32;
    cfg.mrope_section = [64; 4];
    cfg.intermediate_size = Some(HIDDEN as u32);
    let mut tensors = tiny_tensors();
    for tensor in &mut tensors {
        tensor.dims = match tensor.name {
            "blk.2.nextn.eh_proj.weight" => vec![2 * HIDDEN as u64, HIDDEN as u64],
            "blk.2.nextn.embed_tokens.weight" | "blk.2.nextn.shared_head_head.weight" => {
                vec![HIDDEN as u64, cfg.vocab_size as u64]
            }
            _ if tensor.dims.len() == 1 => vec![HIDDEN as u64],
            _ => vec![HIDDEN as u64, HIDDEN as u64],
        };
        let elements = tensor.dims.iter().map(|dim| *dim as usize).product();
        tensor.data = vec![tensor.data[0]; elements];
    }
    let path = std::env::temp_dir().join(format!(
        "mtp_quantized_dedicated_carry_{}.gguf", std::process::id(),
    ));
    write_gguf(&path, &tensors);
    let gguf = GgufFile::open(&path).unwrap();
    let mut mtp = load_mtp_weights_if_present(&gguf, &cfg, &device).unwrap().unwrap();
    let table = (0..cfg.vocab_size as usize * HIDDEN)
        .map(|index| 0.125 + (index % HIDDEN) as f32 / 1024.0)
        .collect::<Vec<_>>();
    // A real native Q8_0 table: the consumer must dispatch the asynchronous
    // gather, not the existing F32 host-row upload fixture.
    mtp.embed_tokens = Some(MlxQWeight {
        buffer: quantize_f32_to_q8_0_buffer(
            &table, vec![cfg.vocab_size as usize, HIDDEN], &device,
        ).unwrap(),
        info: QuantWeightInfo {
            ggml_dtype: GgmlType::Q8_0, rows: cfg.vocab_size as usize, cols: HIDDEN,
        },
        affine: None,
        f16_shadow: None,
        decode_record_q6k_m1: std::sync::OnceLock::new(),
    });
    let _pooled_outputs = StepPooledOutputs::enter(true);
    let row_bytes = ROWS * HIDDEN * 4;
    assert!(row_bytes.is_power_of_two(), "no projection bucket-rounding slack");
    assert_eq!(cfg.num_key_value_heads * cfg.head_dim, HIDDEN as u32);
    let plan = CatchupPlan::new(&cfg, ROWS, true);
    let tokens = (0..ROWS as u32).collect::<Vec<_>>();
    let positions = super::super::spec_decode::positions_for_range(0, ROWS);
    let _caps = ScratchCapsForTest::set(1 << 20, 1 << 10, 1 << 20);
    for case in 0..4 {
        checked_reset_decode_pool().unwrap();
        // Leave a genuinely issued warm root, not merely free capacity.
        // The serving boundary must reclaim it before the first gather.
        super::super::decode_pool::cap_pool(None);
        let warm = super::super::decode_pool::pooled_alloc_buffer(
            &device, 1 << 20, mlx_native::DType::U8, vec![1 << 20],
        ).unwrap();
        drop(warm);
        assert!(super::super::decode_pool::worker_carry_pool_bytes().unwrap() >= 1 << 20);
        let _worker = WorkerScope::open(
            CarryOwners::default(), 0, plan.compose(0, 0, 0),
        );
        carry_memory::commit_target(&cfg, ROWS).expect("prospective quantized catch-up promise");
        let target = upload_f32(&vec![0.25; ROWS * HIDDEN], &device).unwrap();
        assert_eq!(target.byte_len(), row_bytes);
        let mut cache = HybridKvCache::new_with_options(&cfg, &device, 96, 1, true).unwrap();
        let mut registry = KernelRegistry::new();
        let transaction = carry_memory::CatchupTransaction::open(
            &cfg, ROWS, &target, None,
        ).unwrap();
        assert!(super::super::decode_pool::worker_carry_pool_bytes().unwrap() <= 1 << 10,
            "checked target boundary trims issued warm roots to the enforced prior-pool limit");
        let checked = FailSubmitted::nth("", if case == 0 { usize::MAX } else { 0 });
        let gathered = transaction.gather(|| {
            let mut shared = device.alloc_buffer(
                row_bytes, mlx_native::DType::F32, vec![ROWS, HIDDEN],
            )?;
            // The production shared-embedding producer, with a real native
            // Q8_0 table and real asynchronous gather command, not an upload.
            super::super::forward_gpu::embed_tokens_gpu_into(
                &tokens, &[], mtp.embed_tokens.as_ref(), cfg.vocab_size,
                cfg.hidden_size, &device, &mut registry, &mut shared, None,
            )?;
            if case == 3 {
                anyhow::bail!("injected host failure after shared gather submission");
            }
            Ok(shared)
        });
        if case == 3 {
            let error = gathered.err().expect("producer error must check its submitted gather");
            assert_eq!(checked.seen(), (true, 1));
            assert!(format!("{error:#}").contains("injected failure of qwen35.embedding.native"));
            drop(transaction);
            continue;
        }
        let shared = gathered.expect("shared quantized gather submitted asynchronously");
        assert_eq!(shared.byte_len(), row_bytes);
        if case == 2 {
            let error = transaction.finish::<()>(Err(anyhow::anyhow!(
                "injected admission failure before MTP catch-up",
            ))).expect_err("early host failure must surface the submitted gather failure");
            assert_eq!(checked.seen(), (true, 1));
            assert!(format!("{error:#}").contains("injected failure of qwen35.embedding.native"));
            drop(transaction);
            continue;
        }
        let encoded = mtp.process_target_batch(
            &tokens, None, &target, &shared, &mut cache,
            crate::serve::multi_seq_kv::SlotId(0), &positions,
            &device, &mut registry, &cfg,
        );
        assert_eq!(checked.seen(), (false, 0),
            "joining MTP must not replace or prematurely settle shared commands");
        let result = transaction.finish(encoded);
        assert_eq!(checked.seen(), (case == 1, 6),
            "one checked drain covers shared/dedicated gathers and all four MTP stages");
        if case == 1 {
            let error = result.expect_err("the first shared gather failure must propagate");
            assert!(format!("{error:#}").contains("injected failure of qwen35.embedding.native"));
        } else {
            result.expect("shared and dedicated quantized gathers plus KV-only catch-up");
            assert_eq!(cache.mtp_slot.as_ref().unwrap().current_len[0], ROWS as u32);
        }
        drop(transaction);
        // Requires the parent-owned hook after native route selection (the
        // F32 early-return path must not retain an inactive ID root):
        // super::carry_memory::keep_operand(&ids).
        let ids = (ROWS * std::mem::size_of::<u32>()) as u64;
        assert_eq!(carry_memory::last_phase_token_id_bytes(), Some(2 * ids));
        let (actual, promised) = carry_memory::last_phase().unwrap();
        assert!(actual <= promised);
        assert!(actual > promised - ids,
            "exact buckets cannot hide even one omitted command-owned token-ID root");
    }
    std::fs::remove_file(path).unwrap();
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
