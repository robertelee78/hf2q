//! Tiled DFlash2 verify attention against the serial per-row chain, on
//! hf2q's own TQ cache, at the served geometries (24 query / 4 KV heads and
//! 16 / 2, head dim 256).
//!
//! `apply_tq_verify_tile_gated_into` must
//! - leave the cache bytes and cursors exactly where the serial chain leaves
//!   them (the K/V encode is the serial one, Class A),
//! - give every window row an output inside a small envelope of the serial
//!   chain's (`apply_sdpa_with_kv_cache_decode_into` then
//!   `apply_sigmoid_gate_multiply`, row after row): the route is Class B,
//!   its F64 bound is mlx-native's `test_tq_hb_verify_tile`,
//! - be deterministic, and give a lane the same bits alone and in any
//!   cohort, packed position or padding,
//! - reuse one step scratch across layers,
//! - keep all of that across steps with partial acceptance, and never read
//!   a rejected or never-written cache token.

use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::gpu_full_attn::{apply_sdpa_with_kv_cache_decode_into, apply_sigmoid_gate_multiply};
use super::joint_decode_core::JointWindow;
use super::joint_decode_test_support::{next, random_f32};
use super::kv_cache::HybridKvCache;
use super::verify_attn_tile::{
    apply_tq_verify_tile_gated_into, tq_verify_tile_route, VerifyTileScratch,
};
use super::{default_layer_types, Qwen35Config, Qwen35MoeConfig, Qwen35Variant};
use crate::serve::multi_seq_kv::SlotId;

const HD: u32 = 256;
const CAPACITY: u32 = 2_112;
const N_SEQS: u32 = 4;
/// Cursor of each slot: one page, a split boundary region, an almost empty
/// slot, several pages per split.
const CURSORS: [u32; 4] = [299, 2_040, 5, 799];

#[derive(Clone, Copy)]
struct Geometry {
    nh: u32,
    nkv: u32,
}

const QWEN38: Geometry = Geometry { nh: 24, nkv: 4 };
const QWEN36: Geometry = Geometry { nh: 16, nkv: 2 };

impl Geometry {
    fn q_total(self) -> usize {
        (self.nh * HD) as usize
    }

    fn kv_total(self) -> usize {
        (self.nkv * HD) as usize
    }

    fn cfg(self) -> Qwen35Config {
        Qwen35Config {
            variant: Qwen35Variant::Moe,
            hidden_size: 2048,
            num_hidden_layers: 4,
            num_attention_heads: self.nh,
            num_key_value_heads: self.nkv,
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
}

/// Initial full-attention cache bytes of every scenario.
struct CacheInit {
    k_packed: Vec<u8>,
    v_packed: Vec<u8>,
    k_norms: Vec<f32>,
    v_norms: Vec<f32>,
}

fn cache_init(geometry: Geometry) -> CacheInit {
    let mut seed = 0xA77E_711E_u32;
    let codes = (N_SEQS * geometry.nkv * CAPACITY * HD) as usize;
    let norms = (N_SEQS * geometry.nkv * CAPACITY) as usize;
    let mut bytes = |n: usize| (0..n).map(|_| (next(&mut seed) >> 24) as u8).collect();
    let k_packed = bytes(codes);
    let v_packed = bytes(codes);
    let mut norm_values = |n: usize, scale: f32| -> Vec<f32> {
        random_f32(&mut seed, n)
            .into_iter()
            .map(|x| scale * (0.5 + x.abs()))
            .collect()
    };
    // Key norms of the size the rows' own encodes produce, so history and
    // window tokens compete in the softmax.
    let k_norms = norm_values(norms, 24.0);
    let v_norms = norm_values(norms, 24.0);
    CacheInit {
        k_packed,
        v_packed,
        k_norms,
        v_norms,
    }
}

fn new_cache(device: &MlxDevice, geometry: Geometry, init: &CacheInit) -> HybridKvCache {
    let mut cache =
        HybridKvCache::new_with_options(&geometry.cfg(), device, CAPACITY, N_SEQS, true)
            .expect("TQ cache");
    let slot = &mut cache.full_attn[0];
    slot.current_len.copy_from_slice(&CURSORS);
    let tq = slot.tq.as_mut().expect("TQ buffers");
    tq.k_packed
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&init.k_packed);
    tq.v_packed
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&init.v_packed);
    tq.k_norms
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&init.k_norms);
    tq.v_norms
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&init.v_norms);
    cache
}

type CacheState = (Vec<u8>, Vec<u8>, Vec<u32>, Vec<u32>, Vec<u32>);

fn cache_state(cache: &HybridKvCache) -> CacheState {
    let slot = &cache.full_attn[0];
    let tq = slot.tq.as_ref().unwrap();
    let bits = |b: &MlxBuffer| {
        b.as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect()
    };
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
    buffer
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(values);
    buffer
}

/// One window row: IMROPE'd Q, K and V, and the attention gate.
#[derive(Clone)]
struct Row {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    gate: Vec<f32>,
}

/// Eight rows for each slot.
fn rows(geometry: Geometry) -> Vec<Vec<Row>> {
    rows_seeded(geometry, 0x0DDB_711E)
}

fn rows_seeded(geometry: Geometry, mut seed: u32) -> Vec<Vec<Row>> {
    (0..N_SEQS)
        .map(|_| {
            (0..8)
                .map(|_| Row {
                    q: random_f32(&mut seed, geometry.q_total())
                        .into_iter()
                        .map(|x| 4.0 * x)
                        .collect(),
                    k: random_f32(&mut seed, geometry.kv_total())
                        .into_iter()
                        .map(|x| 4.0 * x)
                        .collect(),
                    v: random_f32(&mut seed, geometry.kv_total())
                        .into_iter()
                        .map(|x| 4.0 * x)
                        .collect(),
                    gate: random_f32(&mut seed, geometry.q_total())
                        .into_iter()
                        .map(|x| 16.0 * x)
                        .collect(),
                })
                .collect()
        })
        .collect()
}

/// The serial chain for one row at its slot's cursor: its gated output.
fn per_row(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    geometry: Geometry,
    cache: &mut HybridKvCache,
    slot: SlotId,
    row: &Row,
) -> Vec<f32> {
    let (nh, nkv) = (geometry.nh as usize, geometry.nkv as usize);
    let q = f32_buffer(device, &row.q, vec![1, nh, HD as usize]);
    let k = f32_buffer(device, &row.k, vec![1, nkv, HD as usize]);
    let v = f32_buffer(device, &row.v, vec![1, geometry.kv_total()]);
    let gate = f32_buffer(device, &row.gate, vec![geometry.q_total()]);
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
        geometry.nh,
        geometry.nkv,
        HD,
        CAPACITY,
        slot,
    )
    .expect("per-row attention");
    enc.memory_barrier();
    let gated = apply_sigmoid_gate_multiply(
        &mut enc,
        registry,
        device,
        &out,
        &gate,
        geometry.q_total() as u32,
    )
    .expect("per-row gate");
    enc.commit_and_wait().expect("per-row completion");
    gated.as_slice::<f32>().unwrap()[..geometry.q_total()].to_vec()
}

/// A verify step: `(slot, window rows)` per lane in packed order.
type Step<'a> = &'a [(u32, usize)];

/// The tiled route for `step` at `width` on `cache`: the gated packed rows.
fn tile_step(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    geometry: Geometry,
    cache: &mut HybridKvCache,
    inputs: &[Vec<Row>],
    step: Step<'_>,
    width: usize,
    scratch: &VerifyTileScratch,
) -> Vec<f32> {
    let mut windows = Vec::new();
    let mut packed: Vec<&Row> = Vec::new();
    for &(slot, rows) in step {
        windows.push(JointWindow {
            slot: SlotId(slot),
            start: packed.len(),
            rows,
        });
        packed.extend(inputs[slot as usize][..rows].iter());
    }
    // Padding rows carry row 0's inputs, as the joint step packs them.
    while packed.len() < width {
        packed.push(packed[0]);
    }
    let pack = |field: fn(&Row) -> &Vec<f32>| -> Vec<f32> {
        packed
            .iter()
            .flat_map(|row| field(row).iter().copied())
            .collect()
    };
    let (nh, nkv) = (geometry.nh as usize, geometry.nkv as usize);
    let q = f32_buffer(device, &pack(|r| &r.q), vec![width, nh, HD as usize]);
    let k = f32_buffer(device, &pack(|r| &r.k), vec![width, nkv, HD as usize]);
    let v = f32_buffer(device, &pack(|r| &r.v), vec![width, geometry.kv_total()]);
    let gate = f32_buffer(device, &pack(|r| &r.gate), vec![width, geometry.q_total()]);
    assert!(tq_verify_tile_route(
        registry,
        device,
        &cache.full_attn[0],
        &windows,
        width,
        geometry.nh,
        geometry.nkv,
        HD,
        CAPACITY
    )
    .expect("route probe"));
    let mut enc = device.command_encoder().expect("encoder");
    let gated = apply_tq_verify_tile_gated_into(
        &mut enc,
        device,
        registry,
        &q,
        &k,
        &v,
        &gate,
        &mut cache.full_attn[0],
        &windows,
        scratch,
        width,
        geometry.nh,
        geometry.nkv,
        HD,
        CAPACITY,
    )
    .expect("tiled verify attention");
    enc.commit_and_wait().expect("tiled verify completion");
    gated.as_slice::<f32>().unwrap()[..width * geometry.q_total()].to_vec()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}

fn tile_available(device: &MlxDevice, registry: &mut KernelRegistry, geometry: Geometry) -> bool {
    let cache =
        HybridKvCache::new_with_options(&geometry.cfg(), device, 64, 1, true).expect("cache");
    let window = [JointWindow {
        slot: SlotId(0),
        start: 0,
        rows: 1,
    }];
    tq_verify_tile_route(
        registry,
        device,
        &cache.full_attn[0],
        &window,
        1,
        geometry.nh,
        geometry.nkv,
        HD,
        64,
    )
    .expect("route probe")
}

/// Envelope of the tiled rows around the serial chain's, over the rows of a
/// step: the largest `|tile - serial|` of a row relative to the serial row's
/// RMS (measured 0.78e-2 to 1.01e-2), and the worst row's relative RMS error
/// (measured 6.2e-4 to 7.9e-4; the limit catches a doubling).
const ENVELOPE_MAX: f64 = 2e-2;
const ENVELOPE_RMS: f64 = 2e-3;

/// `(largest |tile - serial| / rms(serial), worst relative RMS)` over the
/// rows of `serial`, which `tiled` holds packed in the same order.
fn envelope(serial: &[Vec<f32>], tiled: &[f32], q_total: usize) -> (f64, f64) {
    let (mut worst_max, mut worst_rms) = (0.0_f64, 0.0_f64);
    for (row, expected) in serial.iter().enumerate() {
        let got = &tiled[row * q_total..(row + 1) * q_total];
        let signal: f64 = expected.iter().map(|x| f64::from(*x).powi(2)).sum();
        let (mut error, mut largest) = (0.0_f64, 0.0_f64);
        for (a, b) in got.iter().zip(expected) {
            let delta = (f64::from(*a) - f64::from(*b)).abs();
            error += delta * delta;
            largest = largest.max(delta);
        }
        assert!(
            error.is_finite() && signal > 0.0,
            "row {row}: degenerate output"
        );
        worst_max = worst_max.max(largest / (signal / q_total as f64).sqrt());
        worst_rms = worst_rms.max((error / signal).sqrt());
    }
    (worst_max, worst_rms)
}

fn assert_envelope(label: &str, serial: &[Vec<f32>], tiled: &[f32], q_total: usize) {
    let (worst_max, worst_rms) = envelope(serial, tiled, q_total);
    eprintln!(
        "{label}: worst row max |tile - serial| / rms(serial) {worst_max:.3e}, relative RMS {worst_rms:.3e}"
    );
    assert!(
        worst_max <= ENVELOPE_MAX && worst_rms <= ENVELOPE_RMS,
        "{label}: tile leaves the serial envelope: max {worst_max:.3e}, relative RMS {worst_rms:.3e}"
    );
}

#[test]
fn tiled_verify_attention_keeps_the_serial_cache_and_envelope() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = match MlxDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping: no Metal device: {error}");
            return;
        }
    };
    let mut registry = KernelRegistry::new();
    for (name, geometry) in [("qwen38", QWEN38), ("qwen36", QWEN36)] {
        if !tile_available(&device, &mut registry, geometry) {
            eprintln!("RECORDED SKIP: tiled verify attention unavailable (no Metal tensor API)");
            return;
        }
        let q_total = geometry.q_total();
        let init = cache_init(geometry);
        let inputs = rows(geometry);
        let step: Step<'_> = &[(0, 8), (1, 8), (2, 3), (3, 1)];

        // Serial chain, row after row, lane after lane.
        let mut serial_cache = new_cache(&device, geometry, &init);
        let mut serial = Vec::new();
        for &(slot, rows) in step {
            for row in &inputs[slot as usize][..rows] {
                serial.push(per_row(
                    &device,
                    &mut registry,
                    geometry,
                    &mut serial_cache,
                    SlotId(slot),
                    row,
                ));
            }
        }
        let serial_state = cache_state(&serial_cache);

        let mut cache = new_cache(&device, geometry, &init);
        let scratch = VerifyTileScratch::default();
        let tiled = tile_step(
            &device,
            &mut registry,
            geometry,
            &mut cache,
            &inputs,
            step,
            32,
            &scratch,
        );
        assert!(
            cache_state(&cache) == serial_state,
            "{name}: cache bytes or cursors differ from the serial chain"
        );
        assert_envelope(name, &serial, &tiled, q_total);
        // Padding rows carry row 0's inputs and so its bits.
        let first = bits(&tiled[..q_total]);
        for row in 20..32 {
            assert_eq!(
                bits(&tiled[row * q_total..(row + 1) * q_total]),
                first,
                "{name} padding {row}"
            );
        }

        // Deterministic, and the step scratch serves a second layer.
        let mut again_cache = new_cache(&device, geometry, &init);
        let again = tile_step(
            &device,
            &mut registry,
            geometry,
            &mut again_cache,
            &inputs,
            step,
            32,
            &scratch,
        );
        assert_eq!(bits(&again), bits(&tiled), "{name}: repeated step differs");

        // A lane alone, at another width and packed position, in another
        // order: the same bits and the same cache region.
        for (cohort, width) in [
            (&[(1u32, 8usize)][..], 8usize),
            (&[(1, 8)][..], 32),
            (&[(3, 1), (2, 3), (1, 8), (0, 8)][..], 32),
            (&[(2, 3), (1, 8)][..], 16),
        ] {
            let mut cohort_cache = new_cache(&device, geometry, &init);
            let out = tile_step(
                &device,
                &mut registry,
                geometry,
                &mut cohort_cache,
                &inputs,
                cohort,
                width,
                &VerifyTileScratch::default(),
            );
            let start: usize = cohort
                .iter()
                .take_while(|(slot, _)| *slot != 1)
                .map(|(_, rows)| rows)
                .sum();
            assert_eq!(
                bits(&out[start * q_total..(start + 8) * q_total]),
                bits(&tiled[8 * q_total..16 * q_total]),
                "{name}: cohort {cohort:?} width {width} changed lane 1"
            );
            assert_eq!(cohort_cache.full_attn[0].current_len[1], CURSORS[1] + 8);
        }
    }
}

#[test]
fn tiled_verify_attention_route_needs_a_compiled_tq_step() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = match MlxDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping: no Metal device: {error}");
            return;
        }
    };
    let mut registry = KernelRegistry::new();
    let geometry = QWEN36;
    if !tile_available(&device, &mut registry, geometry) {
        eprintln!("RECORDED SKIP: tiled verify attention unavailable (no Metal tensor API)");
        return;
    }
    let window = |slot: u32, start: usize, rows: usize| JointWindow {
        slot: SlotId(slot),
        start,
        rows,
    };
    let mut route =
        |cache: &HybridKvCache, windows: &[JointWindow], nh: u32, hd: u32, capacity: u32| {
            tq_verify_tile_route(
                &mut registry,
                &device,
                &cache.full_attn[0],
                windows,
                32,
                nh,
                geometry.nkv,
                hd,
                capacity,
            )
            .expect("route probe")
        };
    let mut cache =
        HybridKvCache::new_with_options(&geometry.cfg(), &device, 64, 2, true).expect("cache");
    assert!(route(
        &cache,
        &[window(0, 0, 8), window(1, 8, 1)],
        16,
        HD,
        64
    ));
    // Nine rows, a slot outside the cache, no room for the window.
    assert!(!route(&cache, &[window(0, 0, 9)], 16, HD, 64));
    assert!(!route(&cache, &[window(2, 0, 1)], 16, HD, 64));
    cache.full_attn[0].current_len[1] = 60;
    assert!(!route(
        &cache,
        &[window(0, 0, 8), window(1, 8, 5)],
        16,
        HD,
        64
    ));
    assert!(route(
        &cache,
        &[window(0, 0, 8), window(1, 8, 4)],
        16,
        HD,
        64
    ));
    // No kernel: four query heads per KV head, head dim 512, a capacity that
    // is not a page multiple.
    assert!(!route(&cache, &[window(0, 0, 8)], 8, HD, 64));
    assert!(!route(&cache, &[window(0, 0, 8)], 16, 512, 64));
    assert!(!route(&cache, &[window(0, 0, 8)], 16, HD, 48));
    // The F32 control cache keeps the stock chain.
    let f32_cache =
        HybridKvCache::new_with_options(&geometry.cfg(), &device, 64, 2, false).expect("cache");
    assert!(!route(&f32_cache, &[window(0, 0, 8)], 16, HD, 64));
}

/// A copy of `source`'s layer cache and cursors in a fresh cache.
fn clone_cache(device: &MlxDevice, geometry: Geometry, source: &HybridKvCache) -> HybridKvCache {
    let (k_packed, v_packed, k_norms, v_norms, cursors) = cache_state(source);
    let mut cache =
        HybridKvCache::new_with_options(&geometry.cfg(), device, CAPACITY, N_SEQS, true)
            .expect("TQ cache");
    let slot = &mut cache.full_attn[0];
    slot.current_len.copy_from_slice(&cursors);
    let tq = slot.tq.as_mut().expect("TQ buffers");
    tq.k_packed
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&k_packed);
    tq.v_packed
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&v_packed);
    for (buffer, bits) in [(&mut tq.k_norms, &k_norms), (&mut tq.v_norms, &v_norms)] {
        for (dst, &src) in buffer.as_mut_slice::<f32>().unwrap().iter_mut().zip(bits) {
            *dst = f32::from_bits(src);
        }
    }
    cache
}

/// Every token at or after each slot's cursor (rejected or never written):
/// norms set to `value`, codes flipped.
fn poison_from_cursor(cache: &mut HybridKvCache, geometry: Geometry, value: f32) {
    let slot = &mut cache.full_attn[0];
    let cursors = slot.current_len.clone();
    let (nkv, capacity, hd) = (geometry.nkv as usize, CAPACITY as usize, HD as usize);
    let tq = slot.tq.as_mut().expect("TQ buffers");
    let tokens = |s: usize, h: usize| {
        let base = (s * nkv + h) * capacity;
        base + cursors[s] as usize..base + capacity
    };
    for buffer in [&mut tq.k_norms, &mut tq.v_norms] {
        let norms = buffer.as_mut_slice::<f32>().unwrap();
        for s in 0..N_SEQS as usize {
            for h in 0..nkv {
                norms[tokens(s, h)].fill(value);
            }
        }
    }
    for buffer in [&mut tq.k_packed, &mut tq.v_packed] {
        let codes = buffer.as_mut_slice::<u8>().unwrap();
        for s in 0..N_SEQS as usize {
            for h in 0..nkv {
                let range = tokens(s, h);
                for byte in &mut codes[range.start * hd..range.end * hd] {
                    *byte ^= 0xA5;
                }
            }
        }
    }
}

/// Whether two caches agree on the cursors and on every token before them.
fn same_committed(a: &HybridKvCache, b: &HybridKvCache, geometry: Geometry) -> bool {
    let (a, b) = (cache_state(a), cache_state(b));
    let (nkv, capacity, hd) = (geometry.nkv as usize, CAPACITY as usize, HD as usize);
    a.4 == b.4
        && (0..N_SEQS as usize).all(|s| {
            (0..nkv).all(|h| {
                let first = (s * nkv + h) * capacity;
                let tokens = first..first + a.4[s] as usize;
                let codes = tokens.start * hd..tokens.end * hd;
                a.2[tokens.clone()] == b.2[tokens.clone()]
                    && a.3[tokens.clone()] == b.3[tokens]
                    && a.0[codes.clone()] == b.0[codes.clone()]
                    && a.1[codes.clone()] == b.1[codes]
            })
        })
}

/// Five consecutive verify steps with partial acceptance, as the DFlash2
/// commit truncates the cursors: after every step the tiled route's cache
/// bytes and cursors are the serial chain's, its rows stay in the envelope,
/// and a twin cache whose rejected and never-written tokens hold NaN,
/// infinity or -1e30 norms and flipped codes gives the same output bits and
/// writes the same windows, so no row reads a rejected or stale token.
#[test]
fn tiled_verify_attention_survives_rollbacks_and_ignores_rejected_tokens() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = match MlxDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping: no Metal device: {error}");
            return;
        }
    };
    let mut registry = KernelRegistry::new();
    // (lanes as (slot, window rows), rows each lane keeps)
    let steps: [(Step<'_>, &[usize]); 5] = [
        (&[(0, 8), (1, 8), (2, 3), (3, 1)], &[3, 0, 3, 1]),
        (&[(0, 8), (1, 8), (2, 8), (3, 8)], &[8, 5, 0, 2]),
        (&[(3, 2), (1, 8), (0, 5)], &[1, 8, 0]),
        (&[(2, 8), (0, 1), (1, 4), (3, 7)], &[7, 1, 0, 6]),
        (&[(1, 8)], &[2]),
    ];
    for (name, geometry) in [("qwen38", QWEN38), ("qwen36", QWEN36)] {
        if !tile_available(&device, &mut registry, geometry) {
            eprintln!("RECORDED SKIP: tiled verify attention unavailable (no Metal tensor API)");
            return;
        }
        let q_total = geometry.q_total();
        let init = cache_init(geometry);
        let mut serial_cache = new_cache(&device, geometry, &init);
        let mut tile_cache = new_cache(&device, geometry, &init);
        for (index, (step, accepted)) in steps.iter().enumerate() {
            let label = format!("{name} step {index}");
            let inputs = rows_seeded(geometry, 0x5EED_0000 + index as u32);
            let before = tile_cache.full_attn[0].current_len.clone();
            assert_eq!(before, serial_cache.full_attn[0].current_len, "{label}");
            let mut serial = Vec::new();
            for &(slot, rows) in step.iter() {
                for row in &inputs[slot as usize][..rows] {
                    serial.push(per_row(
                        &device,
                        &mut registry,
                        geometry,
                        &mut serial_cache,
                        SlotId(slot),
                        row,
                    ));
                }
            }
            let twins: Vec<(f32, HybridKvCache)> = [f32::NAN, f32::INFINITY, -1e30]
                .into_iter()
                .map(|value| {
                    let mut twin = clone_cache(&device, geometry, &tile_cache);
                    poison_from_cursor(&mut twin, geometry, value);
                    (value, twin)
                })
                .collect();
            let tiled = tile_step(
                &device,
                &mut registry,
                geometry,
                &mut tile_cache,
                &inputs,
                step,
                32,
                &VerifyTileScratch::default(),
            );
            assert!(
                cache_state(&tile_cache) == cache_state(&serial_cache),
                "{label}: cache bytes or cursors differ from the serial chain"
            );
            assert_envelope(&label, &serial, &tiled, q_total);
            for (value, mut twin) in twins {
                let out = tile_step(
                    &device,
                    &mut registry,
                    geometry,
                    &mut twin,
                    &inputs,
                    step,
                    32,
                    &VerifyTileScratch::default(),
                );
                assert_eq!(
                    bits(&out),
                    bits(&tiled),
                    "{label}: a row read a rejected or stale cache token ({value})"
                );
                assert!(
                    same_committed(&twin, &tile_cache, geometry),
                    "{label}: the twin with {value} wrote other windows"
                );
            }
            for (&(slot, _), &kept) in step.iter().zip(accepted.iter()) {
                let new_len = before[slot as usize] + kept as u32;
                for cache in [&mut serial_cache, &mut tile_cache] {
                    cache
                        .truncate_full_attn_to_for_slot(SlotId(slot), new_len)
                        .expect("truncate to the accepted rows");
                }
            }
        }
    }
}
