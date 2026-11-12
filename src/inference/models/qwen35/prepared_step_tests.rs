//! Whole-step Metal gate for the prepared routes of a Qwen joint step and
//! their admission (`HF2Q_REPACKED_GGUF_M32`, `_M32_PROJ`, `_M32_HEAD`,
//! `_ANY_WIDTH`, `_SHARED_HEAD`), on the synthetic model of
//! `joint_verify_tests`. The switches are set per test through
//! [`prepared_switch_override`](super::prepared_switch_override), so the
//! process environment does not select the routes.
//!
//! What it holds:
//! - a DFlash2 target verify step takes the prepared dense FFN, projections
//!   and head, each under its own switch;
//! - with the any-width switch a lane's rows, committed cache and target
//!   captures are bitwise the same at every tile width (8, 16, 32), in every
//!   cohort and packing order, with padding rows beside them, on the
//!   host-token and the GPU-token paths (whose padding rows hold another
//!   token), and they are the rows of the width-32 step without the switch;
//! - plain joint decode and joint MTP verify steps keep their routes under
//!   every switch, so an exact-mode joint row is still bitwise its serial
//!   one-token forward;
//! - the shared head is prepared only when a consumer reads it.
//!
//! Two layouts. The dense layout (Qwen3.8) verifies in a fixed 32-row tile.
//! The MoE layout (Qwen3.6) sizes the tile by its rows; the synthetic model
//! declares the MoE config, which selects that layout and the exact mode,
//! but keeps quantized dense FFN layers, so its steps exercise all three
//! prepared families at every width. Real MoE layers are covered by the
//! ignored official-artifact gate in `real_artifact_dflash2_width_tests`.

use mlx_native::DType;

use super::forward_gpu::Qwen35JointVerifyLane;
use super::joint_decode_routes::{JointDenseFfnRoute, JointRouteMode};
use super::joint_projection::JointProjectionKernel;
use super::joint_step_options::JointStepOptions;
use super::joint_verify_tests::{
    dflash_arm, exact_policy, lane_digest, padded_policy, row_bits, synthetic_cfg,
    with_synthetic_fixture_cfg, with_synthetic_fixture_embedding, Arm, Fixture, LaneDigest,
    RowBits, SLOTS,
};
use super::prepared_switch_override::{Override, PreparedSwitches};
use super::{Qwen35Config, Qwen35JointOutput, Qwen35JointRow, Qwen35MoeConfig};
use crate::inference::spec_decode::dflash2::adapters::DFlash2TargetCapture;
use crate::serve::multi_seq_kv::SlotId;

/// The lane every case observes.
pub(super) const TARGET: u32 = 2;
/// A valid token no lane's stream holds at the verified positions: what the
/// GPU-token path's padding rows carry instead of a copy of row 0.
const PADDING_TOKEN: u32 = 509;

/// The MoE layout: the config that makes the DFlash2 tile follow its rows
/// and selects the exact mode. FFN layers stay the synthetic dense ones.
pub(super) fn moe_layout_cfg(query_heads: u32) -> Qwen35Config {
    Qwen35Config {
        num_attention_heads: query_heads,
        moe: Some(Qwen35MoeConfig {
            moe_intermediate_size: 512,
            num_experts: 8,
            num_experts_per_tok: 2,
            shared_expert_intermediate_size: 512,
        }),
        ..synthetic_cfg()
    }
}

/// The exact mode as the Qwen3.6 DFlash2 target runs it.
pub(super) fn exact_arm(attention_tile: bool) -> Arm {
    Arm {
        mode: JointRouteMode::Exact,
        dense_ffn: JointDenseFfnRoute::SerialBuilder,
        options: JointStepOptions {
            projection: JointProjectionKernel::Canonical,
            pooled_outputs: true,
            verify_attention_tile: attention_tile,
            verify_mixer_fused: false,
        },
    }
}

#[derive(Clone, Copy)]
pub(super) enum Tokens {
    /// Host tokens: padding rows replay row 0.
    Host,
    /// The GPU token tile, its padding rows holding [`PADDING_TOKEN`].
    GpuPoisonedPadding,
}

/// The target lane's observable result of one DFlash2 verify step.
#[derive(Clone, PartialEq)]
pub(super) struct LaneTrace {
    pub(super) rows: Vec<RowBits>,
    pub(super) committed: LaneDigest,
    /// Captured layer outputs of the lane's rows.
    pub(super) captured: Vec<u32>,
}

/// One DFlash2 target verify step over `cohort` (`(slot, rows)` in packed
/// order), every lane accepted in full.
pub(super) fn dflash2_verify(
    fixture: &Fixture<'_>,
    cohort: &[(u32, usize)],
    tokens: Tokens,
) -> LaneTrace {
    let model = fixture.model;
    let mut cache = (fixture.new_cache)();
    let lane_tokens = cohort
        .iter()
        .map(|&(slot, rows)| {
            (0..rows as i32)
                .map(|row| (fixture.token)(slot, fixture.start[slot as usize] + row))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let lanes = cohort
        .iter()
        .zip(&lane_tokens)
        .map(|(&(slot, _), tokens)| Qwen35JointVerifyLane {
            slot_id: SlotId(slot),
            position: fixture.start[slot as usize],
            tokens,
            output: Qwen35JointOutput::LogitsAndHidden,
        })
        .collect::<Vec<_>>();
    let layers = [0, model.cfg.num_hidden_layers as usize - 1];
    let feature_width = model.cfg.hidden_size as usize * layers.len();
    let (captured, verify_rows, token_tile) = model
        .with_gpu_cache_mut(|device, _registry| {
            let captured = device.alloc_buffer(
                SLOTS as usize * 8 * feature_width * 4,
                DType::F32,
                vec![SLOTS as usize * 8, feature_width],
            )?;
            let mut verify_rows =
                device.alloc_buffer(SLOTS as usize * 4, DType::U32, vec![SLOTS as usize])?;
            verify_rows.as_mut_slice::<u32>()?.fill(8);
            let mut token_tile = device.alloc_buffer(32 * 4, DType::U32, vec![32])?;
            let tile = token_tile.as_mut_slice::<u32>()?;
            tile.fill(PADDING_TOKEN);
            for (slot, token) in tile.iter_mut().zip(lane_tokens.iter().flatten()) {
                *slot = *token;
            }
            Ok((captured, verify_rows, token_tile))
        })
        .expect("allocate DFlash2 capture");
    let capture_lanes = cohort
        .iter()
        .map(|&(slot, _)| slot as usize)
        .collect::<Vec<_>>();
    let capture = DFlash2TargetCapture {
        layer_ids: &layers,
        output: &captured,
        lane_indices: &capture_lanes,
        verify_rows: &verify_rows,
        lanes: SLOTS as usize,
        rows_per_lane: 8,
        feature_width,
    };
    let arm = fixture.arm;
    let verify = match tokens {
        Tokens::Host => model.forward_gpu_dflash2_verify(
            &lanes,
            &mut cache,
            arm.mode,
            arm.dense_ffn,
            arm.options,
            Some(capture),
        ),
        Tokens::GpuPoisonedPadding => model.forward_gpu_dflash2_verify_from_gpu_tokens(
            &lanes,
            &token_tile,
            &mut cache,
            arm.mode,
            arm.dense_ffn,
            arm.options,
            Some(capture),
            None,
        ),
    }
    .unwrap_or_else(|error| panic!("DFlash2 verify of {cohort:?}: {error:#}"));
    let index = cohort
        .iter()
        .position(|&(slot, _)| slot == TARGET)
        .expect("the cohort holds the target lane");
    let rows = verify.rows[index]
        .iter()
        .map(|row| row_bits(model, row))
        .collect::<Vec<_>>();
    let start = TARGET as usize * 8 * feature_width;
    let captured = captured.as_slice::<f32>().expect("read capture")
        [start..start + cohort[index].1 * feature_width]
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let accepted = cohort.iter().map(|&(_, rows)| rows).collect::<Vec<_>>();
    model
        .commit_joint_verify(&mut cache, verify.pending, &accepted)
        .expect("commit DFlash2 verify");
    LaneTrace {
        rows,
        committed: lane_digest(&cache, &model.cfg, TARGET),
        captured,
    }
}

/// The target lane's rows and committed cache from a joint MTP verify step.
fn mtp_verify(fixture: &Fixture<'_>, cohort: &[(u32, usize)]) -> (Vec<RowBits>, LaneDigest) {
    let model = fixture.model;
    let mut cache = (fixture.new_cache)();
    let lane_tokens = cohort
        .iter()
        .map(|&(slot, rows)| {
            (0..rows as i32)
                .map(|row| (fixture.token)(slot, fixture.start[slot as usize] + row))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let lanes = cohort
        .iter()
        .zip(&lane_tokens)
        .map(|(&(slot, _), tokens)| Qwen35JointVerifyLane {
            slot_id: SlotId(slot),
            position: fixture.start[slot as usize],
            tokens,
            output: Qwen35JointOutput::LogitsAndHidden,
        })
        .collect::<Vec<_>>();
    let arm = fixture.arm;
    let verify = model
        .forward_gpu_joint_verify(&lanes, &mut cache, arm.mode, arm.dense_ffn, arm.options)
        .unwrap_or_else(|error| panic!("joint MTP verify of {cohort:?}: {error:#}"));
    let index = cohort
        .iter()
        .position(|&(slot, _)| slot == TARGET)
        .expect("the cohort holds the target lane");
    let rows = verify.rows[index]
        .iter()
        .map(|row| row_bits(model, row))
        .collect();
    let accepted = cohort.iter().map(|&(_, rows)| rows).collect::<Vec<_>>();
    model
        .commit_joint_verify(&mut cache, verify.pending, &accepted)
        .expect("commit joint MTP verify");
    (rows, lane_digest(&cache, &model.cfg, TARGET))
}

/// The target slot's rows and cache from `steps` plain joint decode steps
/// shared with `slots`.
fn plain_joint(fixture: &Fixture<'_>, slots: &[u32], steps: usize) -> (Vec<RowBits>, LaneDigest) {
    let model = fixture.model;
    let mut cache = (fixture.new_cache)();
    let arm = fixture.arm;
    let index = slots
        .iter()
        .position(|&slot| slot == TARGET)
        .expect("the step holds the target slot");
    let rows = (0..steps as i32)
        .map(|step| {
            let step_rows = slots
                .iter()
                .map(|&slot| {
                    let position = fixture.start[slot as usize] + step;
                    Qwen35JointRow {
                        slot_id: SlotId(slot),
                        token: (fixture.token)(slot, position),
                        position,
                        output: Qwen35JointOutput::LogitsAndHidden,
                    }
                })
                .collect::<Vec<_>>();
            let outputs = model
                .forward_gpu_decode_joint_with_options(
                    &step_rows,
                    &mut cache,
                    arm.mode,
                    arm.dense_ffn,
                    arm.options,
                )
                .unwrap_or_else(|error| panic!("plain joint step of {slots:?}: {error:#}"));
            row_bits(model, &outputs[index])
        })
        .collect();
    (rows, lane_digest(&cache, &model.cfg, TARGET))
}

/// The target slot's rows and cache from `steps` serial one-token forwards.
fn serial(fixture: &Fixture<'_>, steps: usize) -> (Vec<RowBits>, LaneDigest) {
    let model = fixture.model;
    let hidden = model.cfg.hidden_size as usize;
    let mut cache = (fixture.new_cache)();
    let rows = (0..steps as i32)
        .map(|step| {
            let position = fixture.start[TARGET as usize] + step;
            let (logits, nextn) = model
                .forward_gpu_with_nextn_hidden(
                    &[(fixture.token)(TARGET, position)],
                    &[position; 4],
                    &mut cache,
                    SlotId(TARGET),
                )
                .unwrap_or_else(|error| panic!("serial step {step}: {error:#}"));
            (
                logits.iter().map(|value| value.to_bits()).collect(),
                nextn.as_slice::<f32>().expect("read hidden")[..hidden]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect(),
            )
        })
        .collect();
    (rows, lane_digest(&cache, &model.cfg, TARGET))
}

/// Whether to run: whole steps are compared bit for bit, and under Metal
/// shader validation a whole joint step of the synthetic model does not
/// always repeat its bits (three of four validated runs of this file failed,
/// each in another comparison: two stock steps of one input, a lane's
/// committed cache beside equal rows, two prepared steps of one input;
/// every plain run and twelve plain or validated repeats of one step agree).
/// The kernels this gate is about are checked under validation where they
/// can be isolated: `prepared_width_tests` and mlx-native's
/// `repacked_gguf_mm` tests.
pub(super) fn whole_steps_repeat() -> bool {
    let repeat = std::env::var_os("MTL_SHADER_VALIDATION").is_none();
    if !repeat {
        eprintln!("skipped: whole joint steps do not repeat under Metal shader validation");
    }
    repeat
}

fn logits_of(trace: &LaneTrace) -> Vec<&Vec<u32>> {
    trace.rows.iter().map(|row| &row.0).collect()
}

fn hidden_of(trace: &LaneTrace) -> Vec<&Vec<u32>> {
    trace.rows.iter().map(|row| &row.1).collect()
}

pub(super) fn assert_same_lane(actual: &LaneTrace, expected: &LaneTrace, label: &str) {
    assert!(
        actual.rows == expected.rows,
        "{label}: the lane's rows differ"
    );
    assert!(
        actual.committed == expected.committed,
        "{label}: the lane's committed cache differs"
    );
    assert!(
        actual.captured == expected.captured,
        "{label}: the lane's target captures differ"
    );
}

/// Each prepared family is wired to its switch: turning one on changes the
/// rows it feeds and nothing upstream of it.
fn assert_each_switch_selects_its_route(fixture: &Fixture<'_>, cohort: &[(u32, usize)]) {
    let run = |switches| {
        let _switches = Override::set(switches);
        dflash2_verify(fixture, cohort, Tokens::Host)
    };
    let any_width = PreparedSwitches {
        any_width: true,
        ..PreparedSwitches::OFF
    };
    let stock = run(PreparedSwitches::OFF);
    assert!(
        run(any_width) == stock,
        "the any-width switch alone changed a step"
    );
    let ffn = run(PreparedSwitches {
        ffn: true,
        ..any_width
    });
    assert!(
        ffn.rows != stock.rows && ffn.captured != stock.captured,
        "the dense FFN kept the stock route"
    );
    let projections = run(PreparedSwitches {
        ffn: true,
        projections: true,
        ..any_width
    });
    assert!(
        projections.rows != ffn.rows && projections.captured != ffn.captured,
        "the projections kept the stock route"
    );
    let head = run(PreparedSwitches {
        ffn: true,
        projections: true,
        head: true,
        ..any_width
    });
    assert!(
        logits_of(&head) != logits_of(&projections),
        "the head kept the stock route"
    );
    assert!(
        hidden_of(&head) == hidden_of(&projections) && head.captured == projections.captured,
        "the head switch changed rows upstream of the head"
    );
    // The shared planes are the head's own planes.
    assert!(
        run(PreparedSwitches::ALL) == head,
        "the shared head changed a step"
    );
}

/// Dense layout (Qwen3.8): a fixed 32-row tile holding 8 to 32 live rows.
#[test]
fn dense_layout_dflash2_step_takes_the_prepared_routes_and_ignores_cohort_and_padding() {
    if !whole_steps_repeat() {
        return;
    }
    let (policy, arm, cfg) = (padded_policy(), dflash_arm(), synthetic_cfg());
    with_synthetic_fixture_embedding(policy, arm, cfg, true, |fixture| {
        let alone: &[(u32, usize)] = &[(TARGET, 8)];
        // The shared head is prepared only for a consumer that reads it: the
        // prepared target head. Without it nothing asks for the planes.
        {
            let _switches = Override::set(PreparedSwitches {
                head: false,
                ..PreparedSwitches::ALL
            });
            dflash2_verify(fixture, alone, Tokens::Host);
            assert!(
                !fixture.model.prepared_output_head.is_prepared(),
                "the shared head was prepared for no consumer"
            );
        }
        assert_each_switch_selects_its_route(fixture, alone);
        assert!(fixture.model.prepared_output_head.is_prepared());

        let _switches = Override::set(PreparedSwitches::ALL);
        let reference = dflash2_verify(fixture, alone, Tokens::Host);
        assert_same_lane(
            &dflash2_verify(fixture, alone, Tokens::Host),
            &reference,
            "repeat",
        );
        let cohorts: [(&str, &[(u32, usize)]); 6] = [
            ("16 live rows, target last", &[(0, 8), (TARGET, 8)]),
            ("16 live rows, target first", &[(TARGET, 8), (3, 8)]),
            ("24 live rows", &[(0, 8), (1, 8), (TARGET, 8)]),
            ("32 live rows", &[(0, 8), (1, 8), (TARGET, 8), (3, 8)]),
            (
                "32 live rows, permuted",
                &[(3, 8), (TARGET, 8), (1, 8), (0, 8)],
            ),
            ("11 live rows", &[(0, 3), (TARGET, 8)]),
        ];
        for (label, cohort) in cohorts {
            for tokens in [Tokens::Host, Tokens::GpuPoisonedPadding] {
                assert_same_lane(&dflash2_verify(fixture, cohort, tokens), &reference, label);
            }
        }
        assert_same_lane(
            &dflash2_verify(fixture, alone, Tokens::GpuPoisonedPadding),
            &reference,
            "8 live rows beside 24 poisoned padding rows",
        );
        // The dense tile is always 32 rows: the any-width switch is inert.
        let _width_32 = Override::set(PreparedSwitches::WIDTH_32);
        assert_same_lane(
            &dflash2_verify(fixture, alone, Tokens::Host),
            &reference,
            "any-width off",
        );
    });
}

/// MoE layout (Qwen3.6): the tile is 4, 8, 16 or 32 rows.
fn moe_layout_rows_have_one_value(attention_tile: bool) {
    if !whole_steps_repeat() {
        return;
    }
    let cfg = moe_layout_cfg(if attention_tile { 8 } else { 2 });
    let (policy, arm) = (exact_policy(), exact_arm(attention_tile));
    with_synthetic_fixture_embedding(policy, arm, cfg, true, |fixture| {
        let alone: &[(u32, usize)] = &[(TARGET, 8)];
        let full: &[(u32, usize)] = &[(0, 8), (1, 8), (TARGET, 8), (3, 8)];
        // At 8 rows each family is wired to its switch.
        assert_each_switch_selects_its_route(fixture, alone);

        let (stock_8, width_32) = {
            let _off = Override::set(PreparedSwitches::OFF);
            let stock_8 = dflash2_verify(fixture, alone, Tokens::Host);
            let _width_32 = Override::set(PreparedSwitches::WIDTH_32);
            // Without the any-width switch an 8-row tile keeps the stock
            // kernels and only the 32-row tile is prepared, so the lane's
            // bits follow the tile width.
            assert_same_lane(
                &dflash2_verify(fixture, alone, Tokens::Host),
                &stock_8,
                "8-row tile without any-width",
            );
            let width_32 = dflash2_verify(fixture, full, Tokens::Host);
            assert!(
                width_32.rows != stock_8.rows,
                "the 32-row tile kept the stock route"
            );
            (stock_8, width_32)
        };

        let _switches = Override::set(PreparedSwitches::ALL);
        let reference = dflash2_verify(fixture, alone, Tokens::Host);
        assert!(reference.rows != stock_8.rows);
        // The 8-row tile now computes what the 32-row tile always did.
        assert_same_lane(&reference, &width_32, "8-row tile against the 32-row tile");
        assert_same_lane(
            &dflash2_verify(fixture, alone, Tokens::Host),
            &reference,
            "repeat",
        );
        let cohorts: [(&str, &[(u32, usize)]); 8] = [
            ("width 16, target last", &[(0, 8), (TARGET, 8)]),
            ("width 16, target first", &[(TARGET, 8), (3, 8)]),
            ("width 16, 11 live rows", &[(0, 3), (TARGET, 8)]),
            ("width 16, 9 live rows", &[(TARGET, 8), (1, 1)]),
            ("width 32, 24 live rows", &[(0, 8), (1, 8), (TARGET, 8)]),
            ("width 32, 17 live rows", &[(TARGET, 8), (0, 8), (3, 1)]),
            ("width 32, full", full),
            (
                "width 32, full, permuted",
                &[(3, 8), (TARGET, 8), (1, 8), (0, 8)],
            ),
        ];
        for (label, cohort) in cohorts {
            for tokens in [Tokens::Host, Tokens::GpuPoisonedPadding] {
                assert_same_lane(&dflash2_verify(fixture, cohort, tokens), &reference, label);
            }
        }
        // A shorter window is the prefix of the longer one at every width.
        let windows: [(&str, &[(u32, usize)]); 3] = [
            ("width 4", &[(TARGET, 4)]),
            ("width 8, 5 live rows", &[(TARGET, 4), (1, 1)]),
            ("width 16", &[(0, 8), (TARGET, 4)]),
        ];
        for (label, cohort) in windows {
            for tokens in [Tokens::Host, Tokens::GpuPoisonedPadding] {
                let window = dflash2_verify(fixture, cohort, tokens);
                assert!(
                    window.rows[..] == reference.rows[..4],
                    "{label}: a 4-row window differs from the 8-row window's first rows"
                );
            }
        }
    });
}

#[test]
fn moe_layout_dflash2_rows_have_one_value_at_every_tile_width_cohort_and_padding() {
    moe_layout_rows_have_one_value(false);
}

#[test]
fn moe_layout_dflash2_rows_have_one_value_with_the_tiled_verify_attention() {
    moe_layout_rows_have_one_value(true);
}

/// Plain joint decode and joint MTP verify steps keep their routes under
/// every switch; in the exact mode a joint row stays bitwise its serial
/// one-token forward.
#[test]
fn plain_joint_and_mtp_steps_keep_their_routes_under_every_switch() {
    if !whole_steps_repeat() {
        return;
    }
    let stock_and_switched = |fixture: &Fixture<'_>, exact: bool| {
        let run = |switches| {
            let _switches = Override::set(switches);
            let mut cases = vec![
                ("MTP verify, 4 rows", mtp_verify(fixture, &[(TARGET, 4)])),
                (
                    "MTP verify, 8 rows",
                    mtp_verify(fixture, &[(0, 4), (TARGET, 4)]),
                ),
                (
                    "MTP verify, 16 rows",
                    mtp_verify(fixture, &[(0, 4), (1, 4), (TARGET, 4), (3, 4)]),
                ),
                ("plain joint, 1 slot", plain_joint(fixture, &[TARGET], 4)),
                (
                    "plain joint, 4 slots",
                    plain_joint(fixture, &[0, 1, TARGET, 3], 4),
                ),
                ("serial", serial(fixture, 4)),
            ];
            if exact {
                cases.push((
                    "MTP verify, 12 rows",
                    mtp_verify(fixture, &[(1, 3), (TARGET, 4), (3, 5)]),
                ));
            }
            cases
        };
        let stock = run(PreparedSwitches::OFF);
        let switched = run(PreparedSwitches::ALL);
        for ((label, stock), (_, switched)) in stock.iter().zip(&switched) {
            assert!(
                stock == switched,
                "{label}: the prepared-route switches changed the step"
            );
        }
        switched
    };
    with_synthetic_fixture_cfg(
        exact_policy(),
        exact_arm(false),
        moe_layout_cfg(2),
        |fixture| {
            let switched = stock_and_switched(fixture, true);
            let of = |label: &str| {
                &switched
                    .iter()
                    .find(|(name, _)| *name == label)
                    .expect("case")
                    .1
            };
            // Exact mode with every switch on: joint is still serial.
            let serial = of("serial");
            for label in ["plain joint, 1 slot", "plain joint, 4 slots"] {
                assert!(
                    of(label) == serial,
                    "{label} differs from the serial forward"
                );
            }
        },
    );
    with_synthetic_fixture_cfg(padded_policy(), dflash_arm(), synthetic_cfg(), |fixture| {
        stock_and_switched(fixture, false);
    });
}
