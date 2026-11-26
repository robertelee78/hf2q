//! Q4-GEMMA-DECODE A1, adapted from the unmerged #699 real-artifact gate.
//! One loaded weight set, two independent hybrid KV scaffolds, identical
//! tokens/positions, and the real stock/F2 batched body. No policy/env mutation.
//! Run with HF2Q_GEMMA_DECODE_KERNELS=1 and HF2Q_BYTE_EQUIV_E2E_GGUF.

use super::*;
use crate::inference::models::gemma4::batched_body::BatchedKvRegime;
use crate::inference::models::gemma4::decode_router_capture::{self, Step};
use crate::inference::models::gemma4::kv_cache::{MultiSeqHbKvBuffers, MultiSeqHybridKvBuffers};
use crate::inference::models::gemma4::routing_overlap::{kv_len, same_bucket};

#[path = "engine_gemma4_decode_timing_tests.rs"]
mod timing;

const SLOTS: usize = 4;

struct Arm {
    fused: bool,
    hb: Vec<MultiSeqHbKvBuffers>,
    hybrid: Vec<MultiSeqHybridKvBuffers>,
}

struct Lane {
    slot: u32,
    prompt: Vec<u32>,
    budget: usize,
    decoded: usize,
    token: u32,
    position: usize,
}

struct StepOut {
    logits: Vec<f32>,
    router: Option<Step>,
    body_gpu_ns: u64,
    head_gpu_ns: u64,
    fused_head: bool,
}

fn load_model() -> LoadedModel {
    let path = PathBuf::from(std::env::var("HF2Q_BYTE_EQUIV_E2E_GGUF")
        .expect("HF2Q_BYTE_EQUIV_E2E_GGUF must name the approved Gemma GGUF"));
    let options = LoadOptions {
        model_path: path, tokenizer_path: None, config_path: None,
        dwq_overlay_path: None, glp_path: None, glp_alpha: None,
        kv_graft_path: None, kv_persist_dir: None, kv_persist_budget_bytes: 0,
        lcp_resume_budget: crate::serve::kv_persist::lcp_registry::LcpResumeBudget::DEFAULT,
    };
    // Every lane retains 2048 positions, enough for all declared trajectories.
    // This is the A1/M1 fixture, not a production Q5 context qualification.
    let loaded = LoadedModel::load_with_context(&options, Some(2048)).expect("load Gemma fixture");
    let LoadedModel::Gemma(model) = &loaded else { panic!("Gemma artifact required") };
    assert!(model.weights.decode_grouped_geglu,
        "launch with HF2Q_GEMMA_DECODE_KERNELS=1 so load validates the reference policy");
    assert!(model.weights.layers.iter().any(|layer|
        layer.moe.gate_up_ggml_dtype == mlx_native::GgmlType::Q4_K && layer.moe.top_k <= 8),
        "artifact must actually exercise F2 Q4_K experts");
    loaded
}

fn prompt(len: usize, seed: usize) -> Vec<u32> {
    (0..len).map(|i| 1000 + ((i * 7919 + seed * 104729) % 20000) as u32).collect()
}

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().fold((0, f32::NEG_INFINITY), |best, (i, &value)|
        if value > best.1 { (i as u32, value) } else { best }).0
}

fn provision(model: &mut GemmaLoadedModel, fused: bool) -> Arm {
    model.provision_multi_seq_kv_for_slot_aware(SLOTS as u32).expect("provision independent KV");
    let hb = model.multi_seq_kv.take().expect("HB scaffold");
    let hybrid = model.multi_seq_kv_hybrid.take().expect("production hybrid KV required");
    assert!(model.multi_seq_kv_dense.take().is_none() && model.multi_seq_kv_mlx.take().is_none());
    Arm { fused, hb, hybrid }
}

fn assert_cache_equal(model: &GemmaLoadedModel, a: &Arm, b: &Arm,
    slot: usize, positions: std::ops::Range<usize>) {
    assert_eq!(a.hybrid.len(), b.hybrid.len());
    for (layer, (left, right)) in a.hybrid.iter().zip(&b.hybrid).enumerate() {
        let heads = model.weights.layers[layer].num_kv_heads;
        let capacity = left.capacity;
        assert_eq!(capacity, right.capacity);
        let first = if left.is_sliding { positions.start.max(positions.end.saturating_sub(capacity)) }
            else { positions.start };
        for (what, x, y) in [("K", &left.k, &right.k),
            ("V packed", &left.v_packed, &right.v_packed),
            ("V norms", &left.v_norms, &right.v_norms)] {
            let x = x.as_logical_slice::<u8>().expect("read completed cache");
            let y = y.as_logical_slice::<u8>().expect("read completed twin cache");
            assert_ne!(x.as_ptr(), y.as_ptr(), "A1 caches must not alias");
            assert_eq!(x.len(), y.len());
            let rows = left.n_seqs as usize * heads * capacity;
            assert_eq!(x.len() % rows, 0);
            let row_bytes = x.len() / rows;
            for position in first..positions.end {
                let p = if left.is_sliding { position % capacity } else { position };
                assert!(p < capacity);
                for head in 0..heads {
                    let offset = ((slot * heads + head) * capacity + p) * row_bytes;
                    assert_eq!(&x[offset..offset + row_bytes], &y[offset..offset + row_bytes],
                        "{what} differs at layer {layer}, slot {slot}, position {position}");
                }
            }
        }
    }
}

fn prefill(model: &mut GemmaLoadedModel, arm: &mut Arm, slot: SlotId,
    tokens: &[u32], output_allowance: usize) -> u32 {
    for layer in &mut arm.hb { layer.reset_for_slot(slot).expect("reset HB slot"); }
    for layer in &mut arm.hybrid { layer.reset_for_slot(slot).expect("reset hybrid slot"); }
    model.weights.decode_grouped_geglu = arm.fused;
    clear_gemma4_self_mounts(model);
    let first = model.weights.forward_prefill_with_soft_tokens_slot_aware(
        tokens, &[], output_allowance, &mut model.ctx, slot, &mut arm.hb,
        Some(&mut arm.hybrid), None, None,
    ).expect("prefill actual arm");
    clear_gemma4_self_mounts(model);
    first
}

fn decode_step(model: &mut GemmaLoadedModel, arm: &mut Arm, slots: &[SlotId],
    tokens: &[u32], positions: &[usize], capture: bool) -> StepOut {
    model.weights.decode_grouped_geglu = arm.fused;
    for (slot, &position) in slots.iter().zip(positions) {
        for layer in &mut arm.hb { layer.seq_lens[slot.0 as usize] = position as u32; }
        for layer in &mut arm.hybrid { layer.seq_lens[slot.0 as usize] = position as u32; }
    }
    decode_router_capture::arm(capture);
    let before = mlx_native::gpu_busy_ns();
    let mut regime = BatchedKvRegime::Hybrid(&mut arm.hybrid);
    let mut fused = None;
    let hidden = model.weights.forward_decode_body_batched(
        tokens, slots, positions, &mut regime, &mut fused, &mut model.ctx,
    ).expect("actual batched decode body");
    let after_body = mlx_native::gpu_busy_ns();
    let router = decode_router_capture::take();
    decode_router_capture::arm(false);
    assert_eq!(router.is_some(), capture, "actual router readback missing");
    let fused_head = fused.is_some();
    let logits = match fused {
        Some(head) => head.logits,
        None => model.weights.lm_head_batched(&hidden, tokens.len(), &mut model.ctx)
            .expect("actual batched head").logits,
    };
    StepOut { logits, router, fused_head, body_gpu_ns: after_body - before,
        head_gpu_ns: mlx_native::gpu_busy_ns() - after_body }
}

fn assert_logits_equal(a: &StepOut, b: &StepOut) {
    assert_eq!(a.logits.len(), b.logits.len());
    if let Some(i) = a.logits.iter().zip(&b.logits).position(|(x, y)| x.to_bits() != y.to_bits()) {
        panic!("logits differ at {i}: {:#010x} vs {:#010x}",
            a.logits[i].to_bits(), b.logits[i].to_bits());
    }
}

#[test]
#[ignore = "requires the approved Gemma GGUF, explicit F2 switch and exclusive GPU lane"]
fn gemma_decode_kernels_match_stock_bitwise() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut loaded = load_model();
    let LoadedModel::Gemma(model) = &mut loaded else { unreachable!() };
    let mut stock = provision(model, false);
    let mut candidate = provision(model, true);
    assert!(stock.hybrid.iter().any(|layer| layer.is_sliding));
    assert!(stock.hybrid.iter().any(|layer| !layer.is_sliding));
    let mut lanes = vec![
        Lane { slot: 0, prompt: prompt(1000, 0), budget: 300, decoded: 0, token: 0, position: 0 },
        Lane { slot: 1, prompt: prompt(490, 1), budget: 140, decoded: 0, token: 0, position: 0 },
        Lane { slot: 2, prompt: prompt(1010, 2), budget: 200, decoded: 0, token: 0, position: 0 },
        Lane { slot: 3, prompt: prompt(37, 3), budget: 60, decoded: 0, token: 0, position: 0 },
    ];
    let mut pending = Some(Lane { slot: 3, prompt: prompt(300, 4), budget: 100,
        decoded: 0, token: 0, position: 0 });
    let admit = |model: &mut GemmaLoadedModel, a: &mut Arm, b: &mut Arm, lane: &mut Lane| {
        let x = prefill(model, a, SlotId(lane.slot), &lane.prompt, lane.budget);
        let y = prefill(model, b, SlotId(lane.slot), &lane.prompt, lane.budget);
        assert_eq!(x, y, "prefill first token");
        assert_cache_equal(model, a, b, lane.slot as usize, 0..lane.prompt.len());
        lane.token = x;
        lane.position = lane.prompt.len();
    };
    for lane in &mut lanes { admit(model, &mut stock, &mut candidate, lane); }
    let (mut steps, mut mixed, mut same) = (0, 0, 0);
    let mut widths = std::collections::BTreeSet::new();
    let mut admitted = false;
    while !lanes.is_empty() {
        lanes.sort_by_key(|lane| lane.slot);
        let width = lanes.len();
        widths.insert(width);
        let slots: Vec<_> = lanes.iter().map(|lane| SlotId(lane.slot)).collect();
        let tokens: Vec<_> = lanes.iter().map(|lane| lane.token).collect();
        let positions: Vec<_> = lanes.iter().map(|lane| lane.position).collect();
        if width >= 2 {
            let is_mixed = stock.hybrid.iter().any(|layer| {
                let lengths: Vec<_> = positions.iter().map(|&position|
                    kv_len(position, layer.capacity, layer.is_sliding, 0)).collect();
                !same_bucket(&lengths)
            });
            if is_mixed { mixed += 1; } else { same += 1; }
        }
        let a = decode_step(model, &mut stock, &slots, &tokens, &positions, true);
        let b = decode_step(model, &mut candidate, &slots, &tokens, &positions, true);
        assert_logits_equal(&a, &b);
        let ra = a.router.as_ref().expect("stock routing");
        let rb = b.router.as_ref().expect("candidate routing");
        assert_eq!(ra.n, width);
        assert_eq!(ra.stride, rb.stride);
        assert_eq!(ra.ids, rb.ids, "expert IDs at step {steps}");
        assert_eq!(ra.weights.len(), rb.weights.len());
        assert!(ra.weights.iter().zip(&rb.weights).all(|(x, y)| x.to_bits() == y.to_bits()),
            "expert weights at step {steps}");
        let vocab = a.logits.len() / width;
        for (row, lane) in lanes.iter_mut().enumerate() {
            let token = argmax(&a.logits[row * vocab..(row + 1) * vocab]);
            assert_eq!(token, argmax(&b.logits[row * vocab..(row + 1) * vocab]));
            assert_cache_equal(model, &stock, &candidate, lane.slot as usize,
                lane.position..lane.position + 1);
            lane.token = token;
            lane.position += 1;
            lane.decoded += 1;
        }
        lanes.retain(|lane| lane.decoded < lane.budget);
        if !admitted && !lanes.iter().any(|lane| lane.slot == 3) {
            let mut lane = pending.take().expect("one mid-run admission");
            admit(model, &mut stock, &mut candidate, &mut lane);
            lanes.push(lane);
            admitted = true;
        }
        steps += 1;
    }
    assert!(steps >= 256 && admitted && mixed > 0 && same > 0);
    for width in [1, 2, 3, 4] { assert!(widths.contains(&width)); }
    eprintln!("GEMMA_DECODE_A1_EXACT steps={steps} widths={widths:?} mixed={mixed} same={same} logits_tokens_K_V_norms_router_ids_weights=true");
}
