//! Gates of the fused mixer stage of a DFlash2 verify step
//! ([`super::joint_mixer_fused`], `HF2Q_QWEN_VERIFY_MIXER_FUSED`).
//!
//! The stage is Class A: the whole DFlash2 verify step with the option must
//! give the bits of the step without it. The kernel's own gates (bitwise
//! against the stock kernel chain at the served geometries, containment,
//! lane independence) are mlx-native `tests/test_gdn_window_fused.rs`.
//!
//! The prepared projections are process-wide switches, so run this module
//! twice to cover both projection routes under the fused stage:
//! `cargo test --locked --bin hf2q joint_mixer_fused` and the same with
//! `HF2Q_REPACKED_GGUF_M32=1 HF2Q_REPACKED_GGUF_M32_PROJ=1` (then the layer's
//! prepared input products run as one wave).

use super::joint_verify_tests::{
    dflash_arm, lane, padded_policy, run_dflash_case, with_synthetic_fixture, Arm, Fixture,
    LaneCase,
};
use super::{Qwen35JointOutput, Qwen35JointVerifyLane};
use crate::serve::multi_seq_kv::SlotId;

/// The DFlash2 arm with the fused mixer stage requested.
fn fused_arm() -> Arm {
    let mut arm = dflash_arm();
    arm.options.verify_mixer_fused = true;
    arm
}

/// The DFlash2 verify step with the fused mixer stage against the stock
/// step of the same model and caches, bit for bit: every row's logits and
/// hidden state, the target captures, and each lane's retained cache after
/// the commit (attention K/V and cursors, DeltaNet convolution and recurrent
/// state) for lanes that keep all, some or none of their rows. The synthetic
/// model's six DeltaNet layers have 128-wide heads and four taps, the
/// geometry the fused kernel serves, so the step must take it: it issues at
/// least 100 fewer dispatches per DeltaNet layer at 32 rows.
#[test]
fn dflash2_fused_mixer_is_bitwise_the_stock_route() {
    let cases: [(&str, &[LaneCase], u64); 5] = [
        (
            "four full lanes",
            &[lane(0, 7, 0), lane(1, 7, 2), lane(2, 7, 5), lane(3, 7, 8)],
            600,
        ),
        (
            "short windows",
            &[lane(0, 2, 1), lane(2, 0, 1), lane(3, 6, 4)],
            150,
        ),
        ("one lane", &[lane(2, 7, 5)], 150),
        (
            "permuted lanes",
            &[lane(3, 7, 8), lane(2, 7, 5), lane(1, 7, 2), lane(0, 7, 0)],
            600,
        ),
        ("single rows", &[lane(1, 0, 1), lane(3, 0, 0)], 0),
    ];
    with_synthetic_fixture(padded_policy(), fused_arm(), |fixture| {
        let stock_fixture = Fixture {
            model: fixture.model,
            arm: dflash_arm(),
            new_cache: fixture.new_cache,
            start: fixture.start,
            token: fixture.token,
        };
        for (name, lanes, fewer) in cases {
            let before = mlx_native::dispatch_count();
            let stock = run_dflash_case(&stock_fixture, lanes, name);
            let stock_dispatches = mlx_native::dispatch_count() - before;
            let before = mlx_native::dispatch_count();
            let fused = run_dflash_case(fixture, lanes, name);
            let fused_dispatches = mlx_native::dispatch_count() - before;
            eprintln!(
                "fused mixer, {name}: {stock_dispatches} stock and {fused_dispatches} fused dispatches"
            );
            assert!(
                stock_dispatches >= fused_dispatches + fewer,
                "{name}: the step did not take the fused mixer stage: {stock_dispatches} stock \
                 and {fused_dispatches} fused dispatches"
            );
            assert!(
                fused.rows == stock.rows,
                "{name}: a row's logits or hidden state differ from the stock route"
            );
            assert!(
                fused.captured == stock.captured,
                "{name}: a target capture differs from the stock route"
            );
            assert_eq!(
                fused.committed, stock.committed,
                "{name}: the retained cache differs from the stock route"
            );
            let again = run_dflash_case(fixture, lanes, name);
            assert!(
                again.rows == fused.rows
                    && again.captured == fused.captured
                    && again.committed == fused.committed,
                "{name}: a repeated fused step differs"
            );
        }

        // MTP verify rows are the joint decode step's: it refuses the stage.
        let mut cache = (fixture.new_cache)();
        let tokens = [(fixture.token)(0, fixture.start[0])];
        let refused = fixture.model.forward_gpu_joint_verify(
            &[Qwen35JointVerifyLane {
                slot_id: SlotId(0),
                position: fixture.start[0],
                tokens: &tokens,
                output: Qwen35JointOutput::LogitsAndHidden,
            }],
            &mut cache,
            fixture.arm.mode,
            fixture.arm.dense_ffn,
            fixture.arm.options,
        );
        assert!(
            refused.is_err(),
            "MTP verify accepted the fused mixer stage"
        );
    });
}
