//! Gates of the retained verify replay storage (`replay_arena`).
//!
//! The retained arena is handed to a step with whatever the previous step
//! left in it. These tests fill it with NaN before every step: a verify
//! forward or replay commit that read an element it had not written first
//! would carry the NaN into a row's logits or hidden, or into the committed
//! DeltaNet state, and fail the bitwise comparisons below.
//!
//! All tests need Apple Metal (`hf2q_gpu_test_lock`) and skip without a
//! device:
//!
//! ```text
//! cargo test --release --bin hf2q replay_arena_tests
//! ```

use super::joint_verify_tests::{
    dflash_arm, lane, mma_arm, padded_policy, run_case, run_dflash_case, with_synthetic_fixture,
    LaneCase, M1_CASES,
};
use super::replay_arena;

/// Restores the environment's switch value and drops the retained arena
/// when a test ends, also by panic.
struct Retained;

impl Retained {
    fn on() -> Self {
        replay_arena::release_retained();
        replay_arena::set_test_override(Some(true));
        Self
    }
}

impl Drop for Retained {
    fn drop(&mut self) {
        replay_arena::set_test_override(None);
        replay_arena::release_retained();
    }
}

/// Joint MTP verify and its replay commit over a NaN-filled retained arena
/// are the spec-off steps' bits: every cap-8 layout, with full, partial,
/// pending-only and discarded windows (`run_case` panics on any difference
/// in a row's logits or hidden, in the committed cache state, or in the
/// spec-off continuation after the commit).
#[test]
fn retained_replay_arena_keeps_verify_and_commit_bitwise() {
    with_synthetic_fixture(padded_policy(), mma_arm(), |fixture| {
        let _retained = Retained::on();
        // One pass over every layout first, so the arena it leaves behind
        // fits each of them.
        for (label, case) in M1_CASES {
            run_case(fixture, case, &format!("sizing {label}"));
        }
        let fresh = replay_arena::fresh_allocations();
        for (label, case) in M1_CASES {
            assert!(
                replay_arena::fill_retained(f32::NAN),
                "{label}: no arena was retained"
            );
            run_case(fixture, case, &format!("retained {label}"));
        }
        assert_eq!(
            replay_arena::fresh_allocations(),
            fresh,
            "a step allocated replay storage although a large enough arena was retained"
        );
    });
}

/// A DFlash2 width-32 verify step and its commit give the same rows,
/// committed state and target captures with a fresh zero-filled arena per
/// step (switch off) and with a retained arena filled with NaN.
#[test]
fn retained_replay_arena_keeps_dflash2_verify_bitwise() {
    const CASES: &[(&str, &[LaneCase])] = &[
        (
            "4x8 discard/partial/partial/full",
            &[lane(0, 7, 0), lane(1, 7, 2), lane(2, 7, 5), lane(3, 7, 8)],
        ),
        (
            "4x8 permuted",
            &[lane(3, 7, 8), lane(2, 7, 5), lane(1, 7, 2), lane(0, 7, 0)],
        ),
        (
            "3x8 partial",
            &[lane(0, 7, 3), lane(2, 7, 1), lane(3, 7, 7)],
        ),
        ("1x8 partial", &[lane(2, 7, 5)]),
    ];
    with_synthetic_fixture(padded_policy(), dflash_arm(), |fixture| {
        replay_arena::set_test_override(Some(false));
        let stock = CASES
            .iter()
            .map(|(label, case)| run_dflash_case(fixture, case, &format!("stock {label}")))
            .collect::<Vec<_>>();
        assert!(
            !replay_arena::fill_retained(f32::NAN),
            "the stock route retained an arena"
        );

        let _retained = Retained::on();
        // Twice: the second pass starts from an arena that already served
        // every layout.
        for pass in 0..2 {
            for ((label, case), stock) in CASES.iter().zip(&stock) {
                let retained = replay_arena::fill_retained(f32::NAN);
                assert!(
                    retained || (pass == 0 && *label == CASES[0].0),
                    "{label}: no arena was retained"
                );
                let observed = run_dflash_case(fixture, case, &format!("retained {label}"));
                assert!(
                    observed == *stock,
                    "{label} pass {pass}: retained replay storage changed a row, a committed \
                     state or a target capture"
                );
            }
        }
    });
}

/// A step that needs more than the retained arena holds allocates its own,
/// and the larger one is the one kept.
#[test]
fn retained_replay_arena_grows_to_the_largest_step() {
    with_synthetic_fixture(padded_policy(), dflash_arm(), |fixture| {
        let _retained = Retained::on();
        let before = replay_arena::fresh_allocations();
        run_dflash_case(fixture, &[lane(2, 7, 5)], "one lane");
        assert_eq!(replay_arena::fresh_allocations(), before + 1);
        run_dflash_case(
            fixture,
            &[lane(0, 7, 0), lane(1, 7, 2), lane(2, 7, 5), lane(3, 7, 8)],
            "four lanes",
        );
        assert_eq!(
            replay_arena::fresh_allocations(),
            before + 2,
            "four lanes fitted the one-lane arena"
        );
        run_dflash_case(fixture, &[lane(2, 7, 5)], "one lane again");
        run_dflash_case(
            fixture,
            &[lane(3, 7, 8), lane(2, 7, 5), lane(1, 7, 2), lane(0, 7, 0)],
            "four lanes again",
        );
        assert_eq!(
            replay_arena::fresh_allocations(),
            before + 2,
            "the four-lane arena was not kept"
        );
    });
}
