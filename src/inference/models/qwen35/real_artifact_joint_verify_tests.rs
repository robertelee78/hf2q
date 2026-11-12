//! Official-artifact gate for the joint MTP verify window (H1) and replay
//! commit (H2) on Qwen3.8: the model-free gate's checks
//! (`joint_verify_tests`) on the real model, after a short ragged prefill
//! of four slots. Arm M of the spec-off reference: padded `mul_mv_ext`
//! mode, shared-weights dense FFN, column-invariant MMA layer products and
//! pooled step outputs. `HF2Q_QWEN38_GGUF` must name a Qwen3.8 path so the
//! serve routing policy (mvN off, `mul_mv_ext` on) binds; no routing
//! variable needs to be exported:
//!
//! ```text
//! HF2Q_QWEN38_GGUF=/path/qwen3.8.gguf cargo test --release --bin hf2q \
//!   qwen38_joint_verify_matches_spec_off_bitwise -- --ignored --exact
//! ```

use super::joint_decode_routes::{JointDenseFfnRoute, JointRouteMode};
use super::joint_projection::JointProjectionKernel;
use super::joint_step_options::JointStepOptions;
use super::joint_verify_tests::{check_invariance, lane, run_case, Arm, Fixture, LaneCase, SLOTS};
use super::kv_cache::HybridKvCache;
use super::real_artifact_joint_decode_invariance_tests::cache_with_capacity;
use super::real_artifact_joint_decode_tests::{load, prompt, GateRouting};
use super::spec_decode::positions_for_range;
use crate::serve::multi_seq_kv::SlotId;

const CAPACITY: u32 = 96;

const CASES: &[(&str, &[LaneCase])] = &[
    (
        "4x2 full/pending/full/discard",
        &[lane(0, 1, 2), lane(1, 1, 1), lane(2, 1, 2), lane(3, 1, 0)],
    ),
    (
        "3 lanes 1/2/2 permuted",
        &[lane(3, 1, 2), lane(0, 2, 1), lane(2, 2, 3)],
    ),
    ("2x4 full/partial", &[lane(1, 3, 4), lane(2, 3, 2)]),
    ("1x4 partial", &[lane(3, 3, 2)]),
];

#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF (a Qwen3.8-named path) and Apple Metal"]
fn qwen38_joint_verify_matches_spec_off_bitwise() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let model = load("HF2Q_QWEN38_GGUF", GateRouting::Serve);
    let arm = Arm {
        mode: JointRouteMode::PaddedMvExt,
        dense_ffn: JointDenseFfnRoute::SharedWeights,
        options: JointStepOptions {
            projection: JointProjectionKernel::ColumnInvariantMma,
            pooled_outputs: true,
            verify_attention_tile: false,
        },
    };
    assert_eq!(model.joint_decode_mode(), Ok(arm.mode));
    let new_cache = || -> HybridKvCache {
        let mut cache = cache_with_capacity(&model, CAPACITY);
        for slot in 0..SLOTS {
            let tokens = prompt(slot, 0);
            model
                .forward_gpu_last_logits(
                    &tokens,
                    &positions_for_range(0, tokens.len()),
                    &mut cache,
                    SlotId(slot),
                )
                .expect("prefill");
        }
        cache
    };
    let start = std::array::from_fn(|slot| prompt(slot as u32, 0).len() as i32);
    let token = |slot: u32, position: i32| {
        let stream = prompt(slot, 1);
        stream[position as usize % stream.len()]
    };
    let fixture = Fixture {
        model: &model,
        arm,
        new_cache: &new_cache,
        start,
        token: &token,
    };
    for (label, case) in CASES {
        run_case(&fixture, case, &format!("qwen38 {label}"));
    }
    check_invariance(&fixture, "qwen38");
}
