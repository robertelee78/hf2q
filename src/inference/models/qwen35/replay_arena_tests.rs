//! Gates of the retained verify replay storage (`replay_arena`).
//!
//! The retained arena is handed to a step with whatever the previous step
//! left in it. These tests poison it before every step ([`POISONS`]: NaN, a
//! value near `f32::MAX`, a large negative value, +infinity, and the previous
//! step's own contents left in place): a verify forward or replay commit
//! that read an element it had not written first would carry the poison
//! into a row's logits or hidden, into a target capture, or into the
//! committed DeltaNet state, and fail the bitwise comparisons below.
//!
//! Covered: DFlash2 width-32 steps of one to four lanes at every accept
//! count 0..=8, on the stock and the tiled attention arm; windows of mixed
//! lengths 1..=8, single-row windows (which retain nothing) beside multi-row
//! ones; joint MTP verify on the MMA, `mul_mv_ext` and exact arms; a step
//! after a verify that was abandoned without a commit; two steps alive at
//! once; and the arena's growth.
//!
//! Whoever changes a dispatch that writes the replay storage
//! (`joint_decode_core::encode_joint_delta_net_layer`,
//! `encode_delta_net_replay`) must keep these passing: they are the proof
//! that the storage needs no clearing.
//!
//! All tests need Apple Metal (`hf2q_gpu_test_lock`) and skip without a
//! device:
//!
//! ```text
//! cargo test --release --bin hf2q replay_arena_tests
//! ```

use super::joint_decode_routes::{JointDenseFfnRoute, JointRouteMode};
use super::joint_step_options::JointStepOptions;
use super::joint_verify_tests::{
    dflash_arm, dflash_commit, dflash_forward, dflash_tile_arm, exact_policy, lane, mma_arm,
    padded_policy, run_case, run_dflash_case, tile_cfg, with_synthetic_fixture,
    with_synthetic_fixture_cfg, Arm, Fixture, LaneCase, M1_CASES,
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

/// What a step finds in the retained arena.
#[derive(Clone, Copy, Debug)]
enum Poison {
    Fill(f32),
    /// Whatever the previous step left.
    Stale,
}

const POISONS: [Poison; 5] = [
    Poison::Fill(f32::NAN),
    Poison::Fill(3.0e38),
    Poison::Fill(-1.0e30),
    Poison::Fill(f32::INFINITY),
    Poison::Stale,
];

fn poison(poison: Poison, label: &str) {
    let retained = match poison {
        Poison::Fill(value) => replay_arena::fill_retained(value),
        Poison::Stale => replay_arena::has_retained(),
    };
    assert!(retained, "{label}: no arena was retained");
}

/// Four lanes of eight rows: the largest DFlash2 step.
const FULL: [LaneCase; 4] = [lane(0, 7, 3), lane(1, 7, 8), lane(2, 7, 0), lane(3, 7, 5)];

/// Every case on the stock route (a fresh zero-filled arena per step), then
/// once per poison over the retained arena: rows, committed state and
/// target captures must be the stock route's bits, and no step may allocate.
fn check_dflash_cases(fixture: &Fixture<'_>, cases: &[Vec<LaneCase>], label: &str) {
    replay_arena::set_test_override(Some(false));
    let stock = cases
        .iter()
        .map(|case| run_dflash_case(fixture, case, &format!("{label} stock {case:?}")))
        .collect::<Vec<_>>();
    assert!(
        !replay_arena::has_retained(),
        "{label}: the stock route retained an arena"
    );

    let _retained = Retained::on();
    // The largest step first, so its arena fits every case.
    run_dflash_case(fixture, &FULL, &format!("{label} sizing"));
    let fresh = replay_arena::fresh_allocations();
    for (case, stock) in cases.iter().zip(&stock) {
        for kind in POISONS {
            let label = format!("{label} {kind:?} {case:?}");
            poison(kind, &label);
            let observed = run_dflash_case(fixture, case, &label);
            assert!(
                observed == *stock,
                "{label}: retained replay storage changed a row, a committed state or a \
                 target capture"
            );
        }
    }
    assert_eq!(
        replay_arena::fresh_allocations(),
        fresh,
        "{label}: a step allocated replay storage although a large enough arena was retained"
    );
}

/// One to four lanes of eight rows at every accept count 0..=8 (the lanes
/// of a case take consecutive counts and rotated slots).
fn sweep_cases() -> Vec<Vec<LaneCase>> {
    let mut cases = Vec::new();
    for lanes in 1..=4_usize {
        for accept in 0..=8_usize {
            cases.push(
                (0..lanes)
                    .map(|index| lane(((index + accept) % 4) as u32, 7, (accept + index) % 9))
                    .collect(),
            );
        }
    }
    cases
}

#[test]
fn retained_arena_dflash2_lane_and_accept_sweep_is_bitwise() {
    with_synthetic_fixture(padded_policy(), dflash_arm(), |fixture| {
        check_dflash_cases(fixture, &sweep_cases(), "stock arm");
    });
}

#[test]
fn retained_arena_dflash2_lane_and_accept_sweep_is_bitwise_on_the_tiled_arm() {
    with_synthetic_fixture_cfg(padded_policy(), dflash_tile_arm(), tile_cfg(), |fixture| {
        check_dflash_cases(fixture, &sweep_cases(), "tiled arm");
    });
}

/// Windows of one to eight rows in one step. A single-row window retains
/// nothing and takes no replay; the last two cases hold only such windows.
fn mixed_window_cases() -> Vec<Vec<LaneCase>> {
    vec![
        vec![lane(0, 0, 1), lane(1, 7, 3), lane(2, 3, 2), lane(3, 0, 0)],
        vec![lane(0, 1, 1), lane(1, 0, 1), lane(2, 5, 6), lane(3, 2, 0)],
        vec![lane(1, 2, 3), lane(3, 0, 1), lane(0, 7, 8), lane(2, 1, 1)],
        vec![lane(3, 6, 4), lane(0, 0, 0), lane(1, 4, 5)],
        vec![lane(3, 3, 4), lane(2, 5, 1)],
        vec![lane(2, 4, 0)],
        vec![lane(2, 0, 1), lane(1, 0, 0)],
        vec![lane(0, 0, 1)],
    ]
}

// One fixture per test: a test thread's decode pool belongs to the first
// device it allocates from.
#[test]
fn retained_arena_dflash2_mixed_window_lengths_are_bitwise() {
    with_synthetic_fixture(padded_policy(), dflash_arm(), |fixture| {
        check_dflash_cases(fixture, &mixed_window_cases(), "mixed windows");
    });
}

#[test]
fn retained_arena_dflash2_mixed_window_lengths_are_bitwise_on_the_tiled_arm() {
    with_synthetic_fixture_cfg(padded_policy(), dflash_tile_arm(), tile_cfg(), |fixture| {
        check_dflash_cases(fixture, &mixed_window_cases(), "mixed windows, tiled arm");
    });
}

/// Joint MTP verify and its replay commit over a poisoned retained arena
/// are the spec-off steps' bits: every cap-8 layout, with full, partial,
/// pending-only and discarded windows (`run_case` panics on any difference
/// in a row's logits or hidden, in the committed cache state, or in the
/// spec-off continuation after the commit).
fn check_joint_mtp(fixture: &Fixture<'_>, label: &str) {
    let _retained = Retained::on();
    // One pass over every layout first, so the arena it leaves behind fits
    // each of them.
    for (case_label, case) in M1_CASES {
        run_case(fixture, case, &format!("{label} sizing {case_label}"));
    }
    let fresh = replay_arena::fresh_allocations();
    for (case_label, case) in M1_CASES {
        for kind in POISONS {
            let label = format!("{label} {kind:?} {case_label}");
            poison(kind, &label);
            run_case(fixture, case, &label);
        }
    }
    assert_eq!(
        replay_arena::fresh_allocations(),
        fresh,
        "{label}: a step allocated replay storage although a large enough arena was retained"
    );
}

#[test]
fn retained_arena_joint_mtp_is_bitwise_on_the_mma_arm() {
    with_synthetic_fixture(padded_policy(), mma_arm(), |fixture| {
        check_joint_mtp(fixture, "mma");
    });
}

#[test]
fn retained_arena_joint_mtp_is_bitwise_on_the_mv_ext_arm() {
    let arm = Arm {
        mode: JointRouteMode::PaddedMvExt,
        dense_ffn: JointDenseFfnRoute::SharedWeights,
        options: JointStepOptions::default(),
    };
    with_synthetic_fixture(padded_policy(), arm, |fixture| {
        check_joint_mtp(fixture, "mv_ext");
    });
}

#[test]
fn retained_arena_joint_mtp_is_bitwise_on_the_exact_arm() {
    let arm = Arm {
        mode: JointRouteMode::Exact,
        dense_ffn: JointDenseFfnRoute::SerialBuilder,
        options: JointStepOptions::default(),
    };
    with_synthetic_fixture(exact_policy(), arm, |fixture| {
        check_joint_mtp(fixture, "exact");
    });
}

/// A verify whose forward ran and whose result was dropped without a commit
/// (a failed or cancelled step) leaves its arena behind with that step's
/// rows in it. The next step, of another layout, is still the stock bits.
#[test]
fn retained_arena_step_after_an_abandoned_verify_is_bitwise() {
    let cases: [&[LaneCase]; 4] = [
        &[lane(1, 7, 2), lane(3, 7, 6)],
        &[lane(0, 3, 2), lane(2, 0, 1), lane(3, 5, 0)],
        &[lane(2, 7, 5)],
        &FULL,
    ];
    with_synthetic_fixture(padded_policy(), dflash_arm(), |fixture| {
        replay_arena::set_test_override(Some(false));
        let stock = cases
            .iter()
            .map(|case| run_dflash_case(fixture, case, "abandoned: stock"))
            .collect::<Vec<_>>();

        let _retained = Retained::on();
        for kind in POISONS {
            for (case, stock) in cases.iter().zip(&stock) {
                let label = format!("after an abandoned verify, {kind:?} {case:?}");
                drop(dflash_forward(fixture, &FULL, "abandoned forward"));
                poison(kind, &label);
                let observed = run_dflash_case(fixture, case, &label);
                assert!(
                    observed == *stock,
                    "{label}: the step after an abandoned verify differs from the stock route"
                );
            }
        }
    });
}

/// Two verify steps between forward and commit at the same time: the second
/// must not get the first one's storage. Both commit orders.
#[test]
fn retained_arena_two_steps_alive_at_once_do_not_share_storage() {
    let small: [LaneCase; 2] = [lane(2, 5, 2), lane(0, 3, 4)];
    with_synthetic_fixture(padded_policy(), dflash_arm(), |fixture| {
        replay_arena::set_test_override(Some(false));
        let stock_full = run_dflash_case(fixture, &FULL, "two alive: stock full");
        let stock_small = run_dflash_case(fixture, &small, "two alive: stock small");

        let _retained = Retained::on();
        run_dflash_case(fixture, &FULL, "two alive: sizing");
        for (round, kind) in POISONS.into_iter().enumerate() {
            let label = format!("two alive, round {round} {kind:?}");
            poison(kind, &label);
            let before = replay_arena::fresh_allocations();
            // The first forward takes the retained arena; the second finds
            // none and allocates its own.
            let (first_case, second_case, first_stock, second_stock): (
                &[LaneCase],
                &[LaneCase],
                _,
                _,
            ) = if round % 2 == 0 {
                (&FULL, &small, &stock_full, &stock_small)
            } else {
                (&small, &FULL, &stock_small, &stock_full)
            };
            let first = dflash_forward(fixture, first_case, &label);
            assert!(
                !replay_arena::has_retained(),
                "{label}: the arena stayed in the cell while a step used it"
            );
            let second = dflash_forward(fixture, second_case, &label);
            assert_eq!(
                replay_arena::fresh_allocations(),
                before + 1,
                "{label}: the second live step did not allocate its own storage"
            );
            // Commit in forward order on even rounds past the first pair,
            // in reverse order otherwise.
            let (observed_first, observed_second) = if round >= 2 && round % 2 == 0 {
                let observed_first = dflash_commit(fixture, first_case, first, &label);
                let observed_second = dflash_commit(fixture, second_case, second, &label);
                (observed_first, observed_second)
            } else {
                let observed_second = dflash_commit(fixture, second_case, second, &label);
                let observed_first = dflash_commit(fixture, first_case, first, &label);
                (observed_first, observed_second)
            };
            assert!(
                observed_first == *first_stock,
                "{label}: the first live step differs from the stock route"
            );
            assert!(
                observed_second == *second_stock,
                "{label}: the second live step differs from the stock route"
            );
            // The larger of the two arenas is the one kept: the full step
            // fits it without a new allocation.
            let before = replay_arena::fresh_allocations();
            poison(Poison::Fill(f32::NAN), &label);
            let again = run_dflash_case(fixture, &FULL, &label);
            assert!(
                again == stock_full,
                "{label}: the step after two live steps differs"
            );
            assert_eq!(
                replay_arena::fresh_allocations(),
                before,
                "{label}: the larger arena was not the one kept"
            );
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
        run_dflash_case(fixture, &FULL, "four lanes");
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
