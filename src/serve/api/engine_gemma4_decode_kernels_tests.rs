//! ADR-063 Q4-GEMMA-DECODE A1 real-artifact Class A gate (design §6.3).
//!
//! One process, one GGUF: the same four ragged lanes (plus one mid-run
//! admission) decode in lockstep under the stock route
//! (`decode_kernels = None`) and under the candidate (the adopted F2 set,
//! `GemmaDecodeKernels::adopted()`: Q4_K expert gate_up with the GEGLU
//! epilogue at widths <= 4), each arm on its own hybrid KV scaffold. Every step must
//! match bitwise: logits (u32 bits), sampled token, the F16 K / TQ-HB V
//! packed / V norm bytes each layer wrote for each active slot, and every
//! layer's routed expert ids and routing weights.
//!
//! Run (model required; one process; quiet host):
//!   HF2Q_BYTE_EQUIV_E2E_GGUF=<gemma4.gguf> cargo test --release --bin hf2q -- \
//!     --exact serve::api::engine::gemma4_decode_kernels_tests::gemma_decode_kernels_match_stock_bitwise \
//!     --ignored --test-threads=1
use super::*;
use crate::inference::models::gemma4::batched_body::{router_capture, BatchedKvRegime};
use crate::inference::models::gemma4::decode_kernels::GemmaDecodeKernels;
use crate::inference::models::gemma4::kv_cache::{MultiSeqHbKvBuffers, MultiSeqHybridKvBuffers};
use crate::inference::models::gemma4::routing_overlap::{kv_len, same_bucket};
use mlx_native::MlxBuffer;

const GGUF_ENV: &str = "HF2Q_BYTE_EQUIV_E2E_GGUF";
const SLOTS: usize = 4;

/// One decode arm: its policy and its own KV scaffold.
struct Arm {
    name: &'static str,
    kernels: Option<GemmaDecodeKernels>,
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
    router: router_capture::Step,
}

fn prompt(len: usize, seed: usize) -> Vec<u32> {
    (0..len)
        .map(|i| 1_000 + ((i * 7_919 + seed * 104_729) % 20_000) as u32)
        .collect()
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .copied()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |best, (i, v)| {
            if v > best.1 {
                (i as u32, v)
            } else {
                best
            }
        })
        .0
}

fn provision(g: &mut GemmaLoadedModel, name: &'static str, kernels: Option<GemmaDecodeKernels>) -> Arm {
    g.provision_multi_seq_kv_for_slot_aware(SLOTS as u32)
        .unwrap_or_else(|e| panic!("provision {name} KV: {e:#}"));
    let hb = g.multi_seq_kv.take().expect("HB scaffold");
    let hybrid = g
        .multi_seq_kv_hybrid
        .take()
        .expect("A1 requires the production hybrid F16-K + TQ-HB-V regime (HF2Q_HYBRID_KV unset)");
    assert!(
        g.multi_seq_kv_dense.take().is_none() && g.multi_seq_kv_mlx.take().is_none(),
        "A1 requires HF2Q_USE_DENSE unset and the default TQ codebook"
    );
    Arm { name, kernels, hb, hybrid }
}

/// Bytes of one cache position of `slot` across all KV heads of a
/// `[n_seqs, nkv, capacity, row]` buffer.
fn position_bytes(
    buf: &MlxBuffer,
    n_seqs: usize,
    nkv: usize,
    capacity: usize,
    slot: usize,
    cache_pos: usize,
) -> Vec<u8> {
    let bytes: &[u8] = buf.as_slice().expect("KV buffer is CPU-readable");
    let rows = n_seqs * nkv * capacity;
    assert_eq!(bytes.len() % rows, 0, "KV buffer is not [n_seqs, nkv, capacity, row]");
    let row = bytes.len() / rows;
    (0..nkv)
        .flat_map(|h| {
            let start = ((slot * nkv + h) * capacity + cache_pos) * row;
            bytes[start..start + row].iter().copied()
        })
        .collect()
}

/// Assert both arms hold identical K, V-packed and V-norm bytes at the given
/// positions of `slot` on every layer.
fn assert_cache_equal(
    g: &GemmaLoadedModel,
    a: &Arm,
    b: &Arm,
    slot: usize,
    positions: std::ops::Range<usize>,
    context: &str,
) {
    for (layer, (la, lb)) in a.hybrid.iter().zip(&b.hybrid).enumerate() {
        let nkv = g.weights.layers[layer].num_kv_heads;
        let n_seqs = la.n_seqs as usize;
        let cap = la.capacity;
        let first = if la.is_sliding {
            positions.start.max(positions.end.saturating_sub(cap))
        } else {
            positions.start
        };
        for p in first..positions.end {
            let cp = if la.is_sliding { p % cap } else { p };
            assert!(cp < cap, "{context}: L{layer} position {p} exceeds capacity {cap}");
            for (what, ba, bb) in [
                ("F16 K", &la.k, &lb.k),
                ("TQ-HB V packed", &la.v_packed, &lb.v_packed),
                ("TQ-HB V norms", &la.v_norms, &lb.v_norms),
            ] {
                assert!(
                    position_bytes(ba, n_seqs, nkv, cap, slot, cp)
                        == position_bytes(bb, n_seqs, nkv, cap, slot, cp),
                    "{context}: {what} bytes differ at L{layer} slot {slot} position {p} \
                     ({} vs {})",
                    a.name,
                    b.name
                );
            }
        }
    }
}

fn prefill(g: &mut GemmaLoadedModel, arm: &mut Arm, lane: &Lane) -> u32 {
    let slot = SlotId(lane.slot);
    for buf in &mut arm.hb {
        buf.reset_for_slot(slot).expect("reset HB slot");
    }
    for buf in &mut arm.hybrid {
        buf.reset_for_slot(slot).expect("reset hybrid slot");
    }
    g.weights.decode_kernels = arm.kernels.clone();
    clear_gemma4_self_mounts(g);
    let first = g
        .weights
        .forward_prefill_with_soft_tokens_slot_aware(
            &lane.prompt,
            &[],
            lane.budget,
            &mut g.ctx,
            slot,
            &mut arm.hb,
            Some(&mut arm.hybrid),
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("{} prefill slot {}: {e:#}", arm.name, lane.slot));
    clear_gemma4_self_mounts(g);
    first
}

fn decode_step(
    g: &mut GemmaLoadedModel,
    arm: &mut Arm,
    slots: &[SlotId],
    tokens: &[u32],
    positions: &[usize],
) -> StepOut {
    g.weights.decode_kernels = arm.kernels.clone();
    for (slot, &p) in slots.iter().zip(positions) {
        for layer in &mut arm.hb {
            layer.seq_lens[slot.0 as usize] = p as u32;
        }
        for layer in &mut arm.hybrid {
            layer.seq_lens[slot.0 as usize] = p as u32;
        }
    }
    router_capture::arm(true);
    let mut regime = BatchedKvRegime::Hybrid(&mut arm.hybrid);
    let mut fused = None;
    let hidden = g
        .weights
        .forward_decode_body_batched(tokens, slots, positions, &mut regime, &mut fused, &mut g.ctx)
        .unwrap_or_else(|e| panic!("{} batched decode body: {e:#}", arm.name));
    let router = router_capture::take().expect("router capture published");
    router_capture::arm(false);
    let logits = match fused {
        Some(head) => head.logits,
        None => {
            g.weights
                .lm_head_batched(&hidden, tokens.len(), &mut g.ctx)
                .unwrap_or_else(|e| panic!("{} batched head: {e:#}", arm.name))
                .logits
        }
    };
    StepOut { logits, router }
}

fn assert_step_equal(a: &StepOut, b: &StepOut, width: usize, context: &str) {
    assert_eq!(a.logits.len(), b.logits.len(), "{context}: logits length");
    if let Some(i) = a
        .logits
        .iter()
        .zip(&b.logits)
        .position(|(x, y)| x.to_bits() != y.to_bits())
    {
        panic!(
            "{context}: logits differ at flat index {i} (row {}): {:#010x} vs {:#010x}",
            i / (a.logits.len() / width),
            a.logits[i].to_bits(),
            b.logits[i].to_bits()
        );
    }
    assert_eq!(a.router.n, width, "{context}: router capture width");
    assert_eq!(a.router.ids, b.router.ids, "{context}: routed expert ids differ");
    let wa: Vec<u32> = a.router.weights.iter().map(|w| w.to_bits()).collect();
    let wb: Vec<u32> = b.router.weights.iter().map(|w| w.to_bits()).collect();
    assert_eq!(wa, wb, "{context}: routing weights differ");
}

#[test]
#[ignore = "requires HF2Q_BYTE_EQUIV_E2E_GGUF (Gemma 4 GGUF) and the GPU"]
fn gemma_decode_kernels_match_stock_bitwise() {
    let gguf_path: PathBuf = std::env::var(GGUF_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| panic!("{GGUF_ENV} must name a Gemma 4 GGUF"));
    assert!(gguf_path.exists(), "GGUF missing: {}", gguf_path.display());
    let load_opts = LoadOptions {
        model_path: gguf_path,
        tokenizer_path: None,
        config_path: None,
        dwq_overlay_path: None,
        glp_path: None,
        glp_alpha: None,
        kv_graft_path: None,
        kv_persist_dir: None,
        kv_persist_budget_bytes: 0,
    };
    let mut loaded = LoadedModel::load(&load_opts).expect("load Gemma GGUF");
    let LoadedModel::Gemma(g) = &mut loaded else {
        panic!("expected a Gemma GGUF")
    };
    let candidate = GemmaDecodeKernels::adopted();
    assert!(
        g.weights
            .layers
            .iter()
            .any(|l| candidate.f2_gate_up_geglu(l.moe.gate_up_ggml_dtype, 1)),
        "no Q4_K expert gate_up: F2 never engages on this artifact"
    );
    let mut stock = provision(g, "stock", None);
    let mut cand = provision(g, "candidate", Some(candidate));
    assert!(stock.hybrid.iter().any(|l| l.is_sliding), "no sliding layer");
    assert!(stock.hybrid.iter().any(|l| !l.is_sliding), "no global layer");

    // Prompt lengths straddle the (nwg, nsg) bucket edges: kv 512 (nwg 16 ->
    // 32) for lane 1, kv 1024 (global nsg) for lanes 0/2, a short lane 3, and
    // sliding rings wrapping for lanes 0/2. Lane 4 is admitted into slot 3
    // when lane 3 finishes. Widths run 4 -> 3 -> 2 -> 1.
    let mut lanes = vec![
        Lane { slot: 0, prompt: prompt(1_000, 0), budget: 300, decoded: 0, token: 0, position: 0 },
        Lane { slot: 1, prompt: prompt(490, 1), budget: 140, decoded: 0, token: 0, position: 0 },
        Lane { slot: 2, prompt: prompt(1_010, 2), budget: 200, decoded: 0, token: 0, position: 0 },
        Lane { slot: 3, prompt: prompt(37, 3), budget: 60, decoded: 0, token: 0, position: 0 },
    ];
    let mut pending = Some(Lane { slot: 3, prompt: prompt(300, 4), budget: 100, decoded: 0, token: 0, position: 0 });
    let global_cap = stock
        .hybrid
        .iter()
        .filter(|l| !l.is_sliding)
        .map(|l| l.capacity)
        .min()
        .unwrap();
    assert!(1_010 + 200 < global_cap, "global capacity {global_cap} too small for A1 lanes");

    let admit = |g: &mut GemmaLoadedModel, stock: &mut Arm, cand: &mut Arm, lane: &mut Lane| {
        let a = prefill(g, stock, lane);
        let b = prefill(g, cand, lane);
        assert_eq!(a, b, "prefill first token differs on slot {}", lane.slot);
        assert_cache_equal(g, stock, cand, lane.slot as usize, 0..lane.prompt.len(), "prefill");
        lane.token = a;
        lane.position = lane.prompt.len();
    };
    for lane in &mut lanes {
        admit(g, &mut stock, &mut cand, lane);
    }

    let (mut steps, mut mixed_steps, mut same_steps) = (0usize, 0usize, 0usize);
    let mut widths = std::collections::BTreeSet::new();
    let mut admitted = false;
    while !lanes.is_empty() {
        lanes.sort_by_key(|l| l.slot);
        let width = lanes.len();
        widths.insert(width);
        let slots: Vec<SlotId> = lanes.iter().map(|l| SlotId(l.slot)).collect();
        let tokens: Vec<u32> = lanes.iter().map(|l| l.token).collect();
        let positions: Vec<usize> = lanes.iter().map(|l| l.position).collect();
        if width >= 2 {
            let mixed = stock.hybrid.iter().any(|l| {
                let lens: Vec<u32> = positions
                    .iter()
                    .map(|&p| kv_len(p, l.capacity, l.is_sliding, 0))
                    .collect();
                !same_bucket(&lens)
            });
            if mixed {
                mixed_steps += 1;
            } else {
                same_steps += 1;
            }
        }
        let context = format!("step {steps} width {width} slots {slots:?} positions {positions:?}");
        let a = decode_step(g, &mut stock, &slots, &tokens, &positions);
        let b = decode_step(g, &mut cand, &slots, &tokens, &positions);
        assert_step_equal(&a, &b, width, &context);
        let vocab = a.logits.len() / width;
        for (row, lane) in lanes.iter_mut().enumerate() {
            let token = argmax(&a.logits[row * vocab..(row + 1) * vocab]);
            assert_eq!(
                token,
                argmax(&b.logits[row * vocab..(row + 1) * vocab]),
                "{context}: sampled token differs on slot {}",
                lane.slot
            );
            assert_cache_equal(
                g,
                &stock,
                &cand,
                lane.slot as usize,
                lane.position..lane.position + 1,
                &context,
            );
            lane.token = token;
            lane.position += 1;
            lane.decoded += 1;
        }
        lanes.retain(|l| l.decoded < l.budget);
        if !admitted && !lanes.iter().any(|l| l.slot == 3) {
            let mut lane = pending.take().expect("pending admission");
            admit(g, &mut stock, &mut cand, &mut lane);
            lanes.push(lane);
            admitted = true;
        }
        steps += 1;
    }
    eprintln!(
        "[gemma-decode-kernels A1] steps={steps} widths={widths:?} mixed_steps={mixed_steps} \
         same_bucket_steps={same_steps}"
    );
    assert!(steps >= 256, "A1 needs >= 256 steps, ran {steps}");
    assert!(admitted, "mid-run admission did not happen");
    for w in [4, 3, 1] {
        assert!(widths.contains(&w), "width {w} not exercised: {widths:?}");
    }
    assert!(mixed_steps > 0, "no mixed-bucket step at width >= 2");
    assert!(same_steps > 0, "no same-bucket step at width >= 2");
}
