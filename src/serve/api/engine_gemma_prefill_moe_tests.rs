//! ADR-063 Q4-GEMMA-PREFILL-MOE (#690) §5.3 item 5: real-artifact Class A
//! gate of the prefill MoE down route.
//!
//! Ignored; `HF2Q_GEMMA4_GGUF` names the official Gemma 4 26B-A4B GGUF. Run
//! alone with `--exact --ignored --nocapture --test-threads=1`.
//! * `gemma_prefill_moe_route_matches_previous_route_bitwise`: in one
//!   process, the previous flattened down call (`Flat`, arm A) and the
//!   slotted call (`Slotted`, arm B) in sequence, each over four cold
//!   prefills of 458 rows and one four-lane joint warm forward of 4 × 58
//!   rows over those histories (the canonical workload's turn-1 shapes).
//!   Arm A must equal arm B in every byte: every layer's routed-expert down
//!   output (`pf_moe_down`) and MLP output (`pf_mlp_down`, where the fused
//!   weighted sum lands; `pf_moe_accum` is not written on that path) and
//!   residual, each cold prefill's first token and logits, the cold KV of
//!   every slot, and each joint lane's first token, logits, KV and cursors.
//!   Both arms run with the per-layer probe armed and again without it.
use super::gemma_warm_batch_tests::{
    assert_lane_equal, load_gate_model, reserve_for, slot_kv_hashes, LaneImage, WarmFixture,
};
use super::*;
use crate::serve::forward_prefill_batched::live_layer_probe::{self, LayerCapture};
use crate::serve::gemma_prefill_moe::GemmaPrefillMoeDown;

const COLD_ROWS: usize = 458;
const WARM_ROWS: usize = 58;

struct ArmImage {
    cold: Vec<(u32, Vec<u32>)>,
    cold_kv: Vec<Vec<[u8; 32]>>,
    warm: Vec<LaneImage>,
    layers: Vec<LayerCapture>,
}

fn run_arm(guard: &mut Gemma4KvGuard<'_>, down: GemmaPrefillMoeDown, probe: bool) -> ArmImage {
    guard.model.weights.prefill_moe_down = down;
    if probe {
        live_layer_probe::arm();
    }
    let fixture =
        WarmFixture::build_shape(guard, &[COLD_ROWS; 4], &[WARM_ROWS; 4]);
    let cold_kv = (0..4)
        .map(|slot| slot_kv_hashes(guard, slot, COLD_ROWS, COLD_ROWS, false))
        .collect();
    let lanes: Vec<_> = (0..4).map(|slot| fixture.lane(slot)).collect();
    assert!(lanes.iter().all(|&(_, prompt, history, _)| {
        gemma_warm_batch::lane_eligible(prompt.len(), history)
    }));
    let warm = fixture
        .joint(guard, &lanes, reserve_for(&lanes))
        .expect("joint warm forward");
    let layers = if probe {
        live_layer_probe::take()
    } else {
        Vec::new()
    };
    ArmImage {
        cold: fixture.cold.clone(),
        cold_kv,
        warm,
        layers,
    }
}

fn first_difference(a: &[f32], b: &[f32]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
}

fn assert_arms_equal(label: &str, flat: &ArmImage, slotted: &ArmImage) {
    for (slot, (a, b)) in flat.cold.iter().zip(&slotted.cold).enumerate() {
        assert_eq!(a.0, b.0, "{label}: cold slot {slot} first token");
        assert!(a.1 == b.1, "{label}: cold slot {slot} logits bits differ");
    }
    assert_eq!(flat.cold_kv, slotted.cold_kv, "{label}: cold KV bytes");
    for (slot, (a, b)) in flat.warm.iter().zip(&slotted.warm).enumerate() {
        assert_lane_equal(&format!("{label}: joint warm lane {slot}"), b, a);
    }
    assert_eq!(flat.layers.len(), slotted.layers.len(), "{label}: captured layers");
    for (index, (a, b)) in flat.layers.iter().zip(&slotted.layers).enumerate() {
        assert_eq!(a.layer, b.layer);
        for (name, x, y) in [
            ("pf_moe_down", &a.moe_down, &b.moe_down),
            ("pf_mlp_down", &a.mlp_out, &b.mlp_out),
            ("hidden", &a.hidden, &b.hidden),
        ] {
            let at = first_difference(x, y);
            assert!(
                at.is_none(),
                "{label}: capture {index} (layer {}) {name} differs at element {at:?}",
                a.layer
            );
        }
    }
}

#[test]
#[ignore = "official Gemma 4 GGUF: slotted prefill MoE down equals the flattened route bit for bit"]
fn gemma_prefill_moe_route_matches_previous_route_bitwise() {
    let mut loaded = load_gate_model();
    let LoadedModel::Gemma(g) = &mut loaded else {
        unreachable!("load_gate_model returns Gemma")
    };
    let mut guard = Gemma4KvGuard::take(g).unwrap();
    for probe in [true, false] {
        let flat = run_arm(&mut guard, GemmaPrefillMoeDown::Flat, probe);
        let slotted = run_arm(&mut guard, GemmaPrefillMoeDown::Slotted, probe);
        if probe {
            // 4 cold prefills + 1 joint forward, 30 layers each.
            let per_forward = guard.model.weights.layers.len();
            assert_eq!(flat.layers.len(), 5 * per_forward, "probe captured every layer");
        }
        let label = if probe { "probed" } else { "unprobed" };
        assert_arms_equal(label, &flat, &slotted);
        println!(
            "GEMMA_PREFILL_MOE_ROUTE_AB {label}: flat == slotted over {} cold slots, {} warm lanes, {} layer captures",
            flat.cold.len(),
            flat.warm.len(),
            flat.layers.len()
        );
    }
}
