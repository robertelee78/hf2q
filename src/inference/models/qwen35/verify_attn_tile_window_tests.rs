//! Window independence of the tiled DFlash2 verify attention on hf2q's own
//! TQ cache.
//!
//! A token's gated attention output must be a function of its position, its
//! inputs and the cache before it, whichever verify window carries it: every
//! row index 0-7, windows of every length, steps of one to four lanes at
//! widths 8, 16 and 32, and windows re-verified after a partial acceptance.
//! Window boundaries come from the scheduler's verify budget, which follows
//! wall-clock time, so anything else makes the served text vary from run to
//! run.
//!
//! Each lane verifies the same 24 positions under every plan below, from the
//! same cache; a window's own K/V rows are written by the route and are
//! committed history of the next window. Every observation of a token must
//! have the same bits and every plan must leave the same cache bytes. The
//! lanes start around every page boundary up to 9,000 tokens of history at
//! which the split length of the kernel steps (every 768 tokens from 1,536),
//! around plain page boundaries, inside a page and at an empty slot, at both
//! served geometries.

use std::collections::HashMap;

use mlx_native::{KernelRegistry, MlxDevice};

use super::joint_decode_test_support::{next, random_f32};
use super::kv_cache::HybridKvCache;
use super::verify_attn_tile::VerifyTileScratch;
use super::verify_attn_tile_tests::{
    bits, tile_available, tile_step_rows, Geometry, Row, HD, QWEN36, QWEN38,
};
use crate::serve::multi_seq_kv::SlotId;

const CAPACITY: u32 = 9_216;
const SLOTS: usize = 4;
/// Positions every lane verifies and keeps under every plan.
const SPAN: u32 = 24;
/// Position `p` of a slot has the inputs of position `p % PERIOD`.
const PERIOD: usize = 13;

/// Where the four slots' cursors start: 16 tokens before a page boundary,
/// so the tokens on both sides of it are verified at every row index.
/// Between them the sets cross every page boundary below 9,000 tokens where
/// the split length steps (1,536 to 8,448 in steps of 768), plain page
/// boundaries (32, 128, 768, 5,024, 9,184) and start at an empty slot.
const BASES: [[u32; SLOTS]; 4] = [
    [0, 1_520, 3_056, 8_432],
    [16, 2_288, 6_128, 5_008],
    [752, 3_824, 4_592, 9_168],
    [5_360, 6_896, 7_664, 112],
];

/// One way to verify a lane's span: `(rows verified, rows kept)` per step.
/// The kept rows of every plan add up to [`SPAN`].
type Plan = &'static [(usize, usize)];

const PLANS: [Plan; 11] = [
    &[(8, 8), (8, 8), (8, 8)],
    // A first window of r rows puts every later token at another row index.
    &[(1, 1), (8, 8), (8, 8), (7, 7)],
    &[(2, 2), (8, 8), (8, 8), (6, 6)],
    &[(3, 3), (8, 8), (8, 8), (5, 5)],
    &[(4, 4), (8, 8), (8, 8), (4, 4)],
    &[(5, 5), (8, 8), (8, 8), (3, 3)],
    &[(6, 6), (8, 8), (8, 8), (2, 2)],
    &[(7, 7), (8, 8), (8, 8), (1, 1)],
    // Short windows.
    &[(5, 5), (3, 3), (2, 2), (6, 6), (8, 8)],
    // Partial acceptance: rejected rows are verified again, from another
    // window start.
    &[(8, 3), (8, 5), (8, 0), (8, 8), (8, 7), (1, 1)],
    &[(8, 1), (8, 1), (8, 6), (8, 2), (8, 8), (6, 6)],
];

/// The lanes of a step in packed order, and its width: every plan runs with
/// four lanes at width 32, and three of them also with fewer lanes, another
/// lane order and the narrower tiles of the MoE target.
const COHORTS: [(&[usize], usize, &[usize]); 4] = [
    (&[0, 1, 2, 3], 32, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    (&[2], 8, &[0, 4, 9]),
    (&[3, 1], 16, &[0, 5, 10]),
    (&[1, 3, 0], 32, &[0, 2, 8]),
];

/// The slots' initial cache content: random codes and norms everywhere.
struct Initial {
    k_packed: Vec<u8>,
    v_packed: Vec<u8>,
    k_norms: Vec<f32>,
    v_norms: Vec<f32>,
}

fn initial(geometry: Geometry) -> Initial {
    let mut seed = 0xA11C_0DE5_u32;
    let tokens = SLOTS * geometry.nkv as usize * CAPACITY as usize;
    let mut bytes = |n: usize| -> Vec<u8> {
        let mut out = Vec::with_capacity(n + 4);
        while out.len() < n {
            out.extend_from_slice(&next(&mut seed).to_le_bytes());
        }
        out.truncate(n);
        out
    };
    let k_packed = bytes(tokens * HD as usize);
    let v_packed = bytes(tokens * HD as usize);
    let mut norms = |n: usize, lo: f32, hi: f32| -> Vec<f32> {
        random_f32(&mut seed, n)
            .into_iter()
            .map(|x| lo + (hi - lo) * x.abs().min(1.0))
            .collect()
    };
    Initial {
        k_packed,
        v_packed,
        // The ranges stored by the served models.
        k_norms: norms(tokens, 16.0, 29.0),
        v_norms: norms(tokens, 2.0, 40.0),
    }
}

/// Token range `tokens` of `(slot, kv head)` in the norm arrays.
fn norm_range(
    geometry: Geometry,
    slot: usize,
    head: usize,
    tokens: std::ops::Range<u32>,
) -> std::ops::Range<usize> {
    let first = (slot * geometry.nkv as usize + head) * CAPACITY as usize;
    first + tokens.start as usize..first + tokens.end as usize
}

/// Reset the cursors to `bases` and the tokens a plan may have written
/// (`base..base + SPAN + 8` of every slot) to their initial content.
fn reset(cache: &mut HybridKvCache, geometry: Geometry, initial: &Initial, bases: &[u32; SLOTS]) {
    let hd = HD as usize;
    let slot_cache = &mut cache.full_attn[0];
    slot_cache.current_len.copy_from_slice(bases);
    let tq = slot_cache.tq.as_mut().expect("TQ buffers");
    for (slot, &base) in bases.iter().enumerate() {
        for head in 0..geometry.nkv as usize {
            let norms = norm_range(geometry, slot, head, base..(base + SPAN + 8).min(CAPACITY));
            let codes = norms.start * hd..norms.end * hd;
            tq.k_packed.as_mut_slice::<u8>().unwrap()[codes.clone()]
                .copy_from_slice(&initial.k_packed[codes.clone()]);
            tq.v_packed.as_mut_slice::<u8>().unwrap()[codes.clone()]
                .copy_from_slice(&initial.v_packed[codes]);
            tq.k_norms.as_mut_slice::<f32>().unwrap()[norms.clone()]
                .copy_from_slice(&initial.k_norms[norms.clone()]);
            tq.v_norms.as_mut_slice::<f32>().unwrap()[norms.clone()]
                .copy_from_slice(&initial.v_norms[norms]);
        }
    }
}

fn new_cache(device: &MlxDevice, geometry: Geometry, initial: &Initial) -> HybridKvCache {
    let mut cache =
        HybridKvCache::new_with_options(&geometry.cfg(), device, CAPACITY, SLOTS as u32, true)
            .expect("TQ cache");
    let tq = cache.full_attn[0].tq.as_mut().expect("TQ buffers");
    tq.k_packed
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&initial.k_packed);
    tq.v_packed
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&initial.v_packed);
    tq.k_norms
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&initial.k_norms);
    tq.v_norms
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&initial.v_norms);
    cache
}

/// The cache bytes of the span of `slot`: codes and norm bits of K and V.
fn span_bytes(cache: &HybridKvCache, geometry: Geometry, slot: usize, base: u32) -> Vec<u8> {
    let hd = HD as usize;
    let tq = cache.full_attn[0].tq.as_ref().expect("TQ buffers");
    let mut out = Vec::new();
    for head in 0..geometry.nkv as usize {
        let norms = norm_range(geometry, slot, head, base..base + SPAN);
        let codes = norms.start * hd..norms.end * hd;
        out.extend_from_slice(&tq.k_packed.as_slice::<u8>().unwrap()[codes.clone()]);
        out.extend_from_slice(&tq.v_packed.as_slice::<u8>().unwrap()[codes]);
        for buffer in [&tq.k_norms, &tq.v_norms] {
            for norm in &buffer.as_slice::<f32>().unwrap()[norms.clone()] {
                out.extend_from_slice(&norm.to_bits().to_le_bytes());
            }
        }
    }
    out
}

/// Inputs of positions `0..PERIOD` of every slot.
fn inputs(geometry: Geometry) -> Vec<Vec<Row>> {
    let mut seed = 0x0DDB_A5E5_u32;
    let mut scaled = |n: usize, scale: f32| -> Vec<f32> {
        random_f32(&mut seed, n)
            .into_iter()
            .map(|x| scale * x)
            .collect()
    };
    (0..SLOTS)
        .map(|_| {
            (0..PERIOD)
                .map(|_| Row {
                    q: scaled(geometry.q_total(), 4.0),
                    k: scaled(geometry.kv_total(), 4.0),
                    v: scaled(geometry.kv_total(), 4.0),
                    gate: scaled(geometry.q_total(), 16.0),
                })
                .collect()
        })
        .collect()
}

/// What every observation of a token must agree on, and where it was seen.
#[derive(Default)]
struct Seen {
    bits: Vec<u32>,
    /// Row indices of the windows that verified the token.
    rows: u8,
    observations: u32,
}

#[test]
fn tiled_verify_attention_gives_a_token_the_same_bits_in_every_window() {
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
        let initial = initial(geometry);
        let rows = inputs(geometry);
        let mut cache = new_cache(&device, geometry, &initial);
        let (mut tokens, mut observations, mut steps) = (0usize, 0u64, 0usize);
        for bases in &BASES {
            // (slot, position) -> the token's bits and where it was seen.
            let mut seen: HashMap<(usize, u32), Seen> = HashMap::new();
            let mut span: [Option<Vec<u8>>; SLOTS] = Default::default();
            for (lanes, width, plans) in COHORTS {
                for &plan in plans.iter().map(|&index| &PLANS[index]) {
                    assert_eq!(plan.iter().map(|step| step.1).sum::<usize>(), SPAN as usize);
                    reset(&mut cache, geometry, &initial, bases);
                    for &(verified, kept) in plan {
                        let starts: Vec<u32> = lanes
                            .iter()
                            .map(|&slot| cache.full_attn[0].current_len[slot])
                            .collect();
                        let step: Vec<(u32, Vec<&Row>)> = lanes
                            .iter()
                            .zip(&starts)
                            .map(|(&slot, &start)| {
                                let window = (0..verified)
                                    .map(|row| &rows[slot][(start as usize + row) % PERIOD])
                                    .collect();
                                (slot as u32, window)
                            })
                            .collect();
                        let gated = tile_step_rows(
                            &device,
                            &mut registry,
                            geometry,
                            &mut cache,
                            &step,
                            width,
                            CAPACITY,
                            &VerifyTileScratch::default(),
                        );
                        steps += 1;
                        for (lane, (&slot, &start)) in lanes.iter().zip(&starts).enumerate() {
                            for row in 0..verified {
                                let packed = (lane * verified + row) * q_total;
                                let got = bits(&gated[packed..packed + q_total]);
                                let position = start + row as u32;
                                let entry = seen.entry((slot, position)).or_default();
                                if entry.observations == 0 {
                                    entry.bits = got;
                                } else {
                                    assert!(
                                        entry.bits == got,
                                        "{name} bases {bases:?}: slot {slot} position {position} \
                                         as row {row} of a {verified}-row window at {start} in a \
                                         {}-lane step of width {width} differs from its other \
                                         windows",
                                        lanes.len()
                                    );
                                }
                                entry.rows |= 1 << row;
                                entry.observations += 1;
                                observations += 1;
                            }
                            cache
                                .truncate_full_attn_to_for_slot(
                                    SlotId(slot as u32),
                                    start + kept as u32,
                                )
                                .expect("keep the accepted rows");
                        }
                    }
                    for &slot in lanes {
                        assert_eq!(
                            cache.full_attn[0].current_len[slot],
                            bases[slot] + SPAN,
                            "{name}: slot {slot} cursor after a plan"
                        );
                        let bytes = span_bytes(&cache, geometry, slot, bases[slot]);
                        match &span[slot] {
                            None => span[slot] = Some(bytes),
                            Some(expected) => assert!(
                                *expected == bytes,
                                "{name} bases {bases:?}: slot {slot} cache bytes depend on the windows"
                            ),
                        }
                    }
                }
            }
            for (slot, &base) in bases.iter().enumerate() {
                for position in base..base + SPAN {
                    // A token `j` positions into the span can be row 0..=j of
                    // a window; from the eighth on, every row index.
                    let reachable = 0xffu8 >> 7u32.saturating_sub(position - base);
                    assert_eq!(
                        seen[&(slot, position)].rows,
                        reachable,
                        "{name}: slot {slot} position {position} was not verified at every row index"
                    );
                }
            }
            tokens += seen.len();
        }
        eprintln!(
            "{name}: {tokens} tokens, {observations} observations in {steps} verify steps: \
             every token bitwise equal in every window"
        );
    }
}
