//! Row-batched joint full-attention core against the per-row chain, on hf2q's
//! own TQ cache (ADR-063 Q4-QWEN36, per-row small ops).
//!
//! `apply_tq_sdpa_rows_gated_into` must give every row of every cohort
//! (widths 1..=4, several orders, a padded width) the bits the per-row chain
//! gives it alone — `apply_sdpa_with_kv_cache_decode_into` followed by
//! `apply_sigmoid_gate_multiply`, the serial single-token layer's core — and
//! leave the cache bytes and cursors exactly where that chain leaves them.
//! Cursors span every split-K schedule and both attention kernels (GQA-Q2
//! from kv 8192).

use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::gpu_full_attn::{
    apply_sdpa_with_kv_cache_decode_into, apply_sigmoid_gate_multiply,
    apply_tq_sdpa_rows_gated_into, tq_sdpa_rows_route,
};
use super::joint_decode_test_support::{next, random_f32};
use super::kv_cache::HybridKvCache;
use super::{default_layer_types, Qwen35Config, Qwen35MoeConfig, Qwen35Variant};
use crate::serve::multi_seq_kv::SlotId;

const NH: u32 = 16;
const NKV: u32 = 2;
const HD: u32 = 256;
const Q_TOTAL: usize = (NH * HD) as usize;
const KV_TOTAL: usize = (NKV * HD) as usize;
const CAPACITY: u32 = 8320;
const N_SEQS: u32 = 4;

fn cfg() -> Qwen35Config {
    Qwen35Config {
        variant: Qwen35Variant::Moe,
        hidden_size: 2048,
        num_hidden_layers: 4,
        num_attention_heads: NH,
        num_key_value_heads: NKV,
        head_dim: HD,
        linear_num_key_heads: 16,
        linear_num_value_heads: 32,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        layer_types: default_layer_types(4, 4),
        partial_rotary_factor: 0.25,
        rope_theta: 1e7,
        rotary_dim: 64,
        mrope_section: [11, 11, 10, 0],
        mrope_interleaved: true,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 262_144,
        vocab_size: 248_320,
        attn_output_gate: true,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: true,
        intermediate_size: None,
        moe: Some(Qwen35MoeConfig {
            moe_intermediate_size: 512,
            num_experts: 256,
            num_experts_per_tok: 8,
            shared_expert_intermediate_size: 512,
        }),
    }
}

/// Initial full-attention cache bytes and cursors shared by every scenario.
struct CacheInit {
    k_packed: Vec<u8>,
    v_packed: Vec<u8>,
    k_norms: Vec<f32>,
    v_norms: Vec<f32>,
    cursors: Vec<u32>,
}

fn cache_init() -> CacheInit {
    let mut seed = 0xA77E_5EED_u32;
    let codes = (N_SEQS * NKV * CAPACITY * HD) as usize;
    let norms = (N_SEQS * NKV * CAPACITY) as usize;
    let mut bytes = |n: usize| (0..n).map(|_| (next(&mut seed) >> 24) as u8).collect();
    let k_packed = bytes(codes);
    let v_packed = bytes(codes);
    let mut norm_values = |n: usize| -> Vec<f32> {
        random_f32(&mut seed, n).into_iter().map(|x| 0.5 + x.abs()).collect()
    };
    let k_norms = norm_values(norms);
    let v_norms = norm_values(norms);
    // Slot s holds cursors[s] tokens: nwg 16, GQA-Q2 (kv 8192), nsg 4, nwg 32 / nsg 1.
    let cursors = vec![299, 8191, 2999, 799];
    CacheInit { k_packed, v_packed, k_norms, v_norms, cursors }
}

fn new_cache(device: &MlxDevice, init: &CacheInit) -> HybridKvCache {
    let mut cache = HybridKvCache::new_with_options(&cfg(), device, CAPACITY, N_SEQS, true)
        .expect("TQ cache");
    let slot = &mut cache.full_attn[0];
    slot.current_len.copy_from_slice(&init.cursors);
    let tq = slot.tq.as_mut().expect("TQ buffers");
    tq.k_packed.as_mut_slice::<u8>().unwrap().copy_from_slice(&init.k_packed);
    tq.v_packed.as_mut_slice::<u8>().unwrap().copy_from_slice(&init.v_packed);
    tq.k_norms.as_mut_slice::<f32>().unwrap().copy_from_slice(&init.k_norms);
    tq.v_norms.as_mut_slice::<f32>().unwrap().copy_from_slice(&init.v_norms);
    cache
}

fn cache_state(cache: &HybridKvCache) -> (Vec<u8>, Vec<u8>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let slot = &cache.full_attn[0];
    let tq = slot.tq.as_ref().unwrap();
    let bits = |b: &MlxBuffer| b.as_slice::<f32>().unwrap().iter().map(|x| x.to_bits()).collect();
    (
        tq.k_packed.as_slice::<u8>().unwrap().to_vec(),
        tq.v_packed.as_slice::<u8>().unwrap().to_vec(),
        bits(&tq.k_norms),
        bits(&tq.v_norms),
        slot.current_len.clone(),
    )
}

fn f32_buffer(device: &MlxDevice, values: &[f32], shape: Vec<usize>) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, shape)
        .expect("allocate f32");
    buffer.as_mut_slice::<f32>().unwrap().copy_from_slice(values);
    buffer
}

/// One decode row: IMROPE'd Q, K and V, and the attention gate.
struct Row {
    slot: SlotId,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    gate: Vec<f32>,
}

fn rows() -> Vec<Row> {
    let mut seed = 0x0DDB_A11_u32;
    (0..N_SEQS)
        .map(|s| Row {
            slot: SlotId(s),
            q: random_f32(&mut seed, Q_TOTAL).into_iter().map(|x| 4.0 * x).collect(),
            k: random_f32(&mut seed, KV_TOTAL).into_iter().map(|x| 4.0 * x).collect(),
            v: random_f32(&mut seed, KV_TOTAL).into_iter().map(|x| 4.0 * x).collect(),
            gate: random_f32(&mut seed, Q_TOTAL).into_iter().map(|x| 16.0 * x).collect(),
        })
        .collect()
}

/// The per-row chain for one row alone: its attention output (un-rotated)
/// and its gated output.
fn per_row(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    cache: &mut HybridKvCache,
    row: &Row,
) -> (Vec<f32>, Vec<u32>) {
    let q = f32_buffer(device, &row.q, vec![1, NH as usize, HD as usize]);
    let k = f32_buffer(device, &row.k, vec![1, NKV as usize, HD as usize]);
    let v = f32_buffer(device, &row.v, vec![1, KV_TOTAL]);
    let gate = f32_buffer(device, &row.gate, vec![Q_TOTAL]);
    let mut enc = device.command_encoder().expect("encoder");
    let out = apply_sdpa_with_kv_cache_decode_into(
        &mut enc,
        device,
        registry,
        &q,
        &k,
        &v,
        &mut cache.full_attn[0],
        1,
        NH,
        NKV,
        HD,
        CAPACITY,
        row.slot,
    )
    .expect("per-row attention");
    enc.memory_barrier();
    let gated = apply_sigmoid_gate_multiply(&mut enc, registry, device, &out, &gate, Q_TOTAL as u32)
        .expect("per-row gate");
    enc.commit_and_wait().expect("per-row completion");
    let attention = out.as_slice::<f32>().unwrap()[..Q_TOTAL].to_vec();
    let gated = gated.as_slice::<f32>().unwrap()[..Q_TOTAL].iter().map(|x| x.to_bits()).collect();
    (attention, gated)
}

fn assert_bits(label: &str, expected: &[u32], actual: &[u32]) {
    assert_eq!(expected.len(), actual.len(), "{label}: length");
    if let Some(i) = expected.iter().zip(actual).position(|(a, b)| a != b) {
        panic!(
            "{label}: element {i}: expected {:?}, got {:?}",
            f32::from_bits(expected[i]),
            f32::from_bits(actual[i])
        );
    }
}

#[test]
fn joint_attention_rows_match_the_per_row_chain_bitwise() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = match MlxDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping: no Metal device: {error}");
            return;
        }
    };
    let mut registry = KernelRegistry::new();
    let init = cache_init();
    let inputs = rows();

    // Per-row references, each row alone; then the cache after every row.
    let expected: Vec<(Vec<f32>, Vec<u32>)> = inputs
        .iter()
        .map(|row| per_row(&device, &mut registry, &mut new_cache(&device, &init), row))
        .collect();
    let mut all_rows = new_cache(&device, &init);
    for row in &inputs {
        per_row(&device, &mut registry, &mut all_rows, row);
    }
    let expected_cache = cache_state(&all_rows);

    // (cohort in packed order, packed width)
    let cohorts: &[(&[usize], usize)] = &[
        (&[0], 1),
        (&[1], 1),
        (&[2, 3], 2),
        (&[3, 1], 2),
        (&[0, 1, 2], 3),
        (&[2, 0, 1], 3),
        (&[0, 1, 2, 3], 4),
        (&[3, 2, 1, 0], 4),
        (&[1, 3, 0, 2], 4),
        (&[1, 2, 0], 4),
    ];
    for &(cohort, width) in cohorts {
        let mut cache = new_cache(&device, &init);
        let slots: Vec<SlotId> = cohort.iter().map(|&i| inputs[i].slot).collect();
        assert!(tq_sdpa_rows_route(&cache.full_attn[0], &slots, HD, CAPACITY));
        // Padding rows carry row 0's inputs, as the joint step packs them.
        let packed = |field: fn(&Row) -> &Vec<f32>| -> Vec<f32> {
            (0..width)
                .flat_map(|p| field(&inputs[cohort[if p < cohort.len() { p } else { 0 }]]).clone())
                .collect()
        };
        let mut pad_seed = 0x9AD_u32;
        let mut gate = packed(|r| &r.gate);
        gate[cohort.len() * Q_TOTAL..]
            .copy_from_slice(&random_f32(&mut pad_seed, (width - cohort.len()) * Q_TOTAL));
        let q = f32_buffer(&device, &packed(|r| &r.q), vec![width, NH as usize, HD as usize]);
        let k = f32_buffer(&device, &packed(|r| &r.k), vec![width, NKV as usize, HD as usize]);
        let v = f32_buffer(&device, &packed(|r| &r.v), vec![width, KV_TOTAL]);
        let gate_buf = f32_buffer(&device, &gate, vec![width, Q_TOTAL]);
        let mut enc = device.command_encoder().expect("encoder");
        let gated = apply_tq_sdpa_rows_gated_into(
            &mut enc,
            &device,
            &mut registry,
            &q,
            &k,
            &v,
            &gate_buf,
            &mut cache.full_attn[0],
            &slots,
            width,
            NH,
            NKV,
            HD,
            CAPACITY,
        )
        .expect("row-batched attention");
        enc.commit_and_wait().expect("row-batched completion");
        let gated: Vec<u32> = gated.as_slice::<f32>().unwrap()[..width * Q_TOTAL]
            .iter()
            .map(|x| x.to_bits())
            .collect();
        for (p, &i) in cohort.iter().enumerate() {
            assert_bits(
                &format!("cohort {cohort:?} width {width} row {i}"),
                &expected[i].1,
                &gated[p * Q_TOTAL..(p + 1) * Q_TOTAL],
            );
        }
        for p in cohort.len()..width {
            // Padding: row 0's attention under the padding row's gate.
            let attention = f32_buffer(&device, &expected[cohort[0]].0, vec![Q_TOTAL]);
            let pad_gate = f32_buffer(&device, &gate[p * Q_TOTAL..(p + 1) * Q_TOTAL], vec![Q_TOTAL]);
            let mut enc = device.command_encoder().expect("encoder");
            let want = apply_sigmoid_gate_multiply(
                &mut enc,
                &mut registry,
                &device,
                &attention,
                &pad_gate,
                Q_TOTAL as u32,
            )
            .expect("padding reference");
            enc.commit_and_wait().expect("padding reference completion");
            let want: Vec<u32> =
                want.as_slice::<f32>().unwrap()[..Q_TOTAL].iter().map(|x| x.to_bits()).collect();
            assert_bits(
                &format!("cohort {cohort:?} width {width} padding row {p}"),
                &want,
                &gated[p * Q_TOTAL..(p + 1) * Q_TOTAL],
            );
        }
        if cohort.len() == inputs.len() {
            assert!(
                cache_state(&cache) == expected_cache,
                "cohort {cohort:?}: cache bytes or cursors differ from the per-row chain"
            );
        }
    }
}

#[test]
fn joint_attention_rows_route_needs_distinct_slots_with_room() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = match MlxDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping: no Metal device: {error}");
            return;
        }
    };
    let mut cache = HybridKvCache::new_with_options(&cfg(), &device, 64, 2, true).expect("cache");
    let slot = &mut cache.full_attn[0];
    assert!(tq_sdpa_rows_route(slot, &[SlotId(0), SlotId(1)], HD, 64));
    assert!(!tq_sdpa_rows_route(slot, &[SlotId(1), SlotId(1)], HD, 64));
    assert!(!tq_sdpa_rows_route(slot, &[SlotId(0), SlotId(2)], HD, 64));
    slot.current_len[1] = 64;
    assert!(!tq_sdpa_rows_route(slot, &[SlotId(0), SlotId(1)], HD, 64));
    assert!(!tq_sdpa_rows_route(slot, &[SlotId(0)], 512, 64));
    let f32_cache = HybridKvCache::new_with_options(&cfg(), &device, 64, 2, false).expect("cache");
    assert!(!tq_sdpa_rows_route(&f32_cache.full_attn[0], &[SlotId(0)], HD, 64));
}
