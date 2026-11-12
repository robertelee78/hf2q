//! Gates of the fused mixer stage of a DFlash2 verify step
//! ([`super::joint_mixer_fused`], `HF2Q_QWEN_VERIFY_MIXER_FUSED`).
//!
//! The stage is Class A: the whole DFlash2 verify step with the option must
//! give the bits of the step without it. The kernel's own gates (bitwise
//! against the stock kernel chain at the served geometries, containment,
//! lane independence) are mlx-native `tests/test_gdn_window_fused.rs`.
//!
//! Besides the single-step comparison, consecutive steps on carried caches
//! (full, partial and zero acceptance between them, then spec-off decode)
//! are compared on the admitted geometry, on the geometries the fused core
//! refuses (head dimension 64, ten convolution taps: the layer must keep
//! the stock chain) and at the served head counts with and without the tiled
//! verify attention. These cases come from the independent review of the
//! stage (Review829).
//!
//! The prepared projections are process-wide switches, so run this module
//! twice to cover both projection routes under the fused stage:
//! `cargo test --locked --bin hf2q joint_mixer_fused` and the same with
//! `HF2Q_REPACKED_GGUF_M32=1 HF2Q_REPACKED_GGUF_M32_PROJ=1` (then the layer's
//! prepared input products run as one wave behind a norm that writes BF16).

use super::joint_verify_tests::{
    dflash_arm, digests, lane, padded_policy, run_dflash_case, run_dflash_step, spec_off_step,
    synthetic_cfg, with_synthetic_fixture, with_synthetic_fixture_cfg, Arm, Fixture, LaneCase,
};
use super::{Qwen35Config, Qwen35JointOutput, Qwen35JointVerifyLane};
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

/// The geometry and row conditions of the fused core and the reason each
/// refusal logs (no GPU).
#[test]
fn fused_core_refusals_name_their_reason() {
    use super::delta_net::DeltaNetLayerShape;
    use super::joint_decode_core::{JointRows, JointWindow};
    use super::joint_mixer_fused::geometry_refusal;

    let slots = [SlotId(0), SlotId(1)];
    let windows = |rows: usize| {
        [
            JointWindow {
                slot: SlotId(0),
                start: 0,
                rows,
            },
            JointWindow {
                slot: SlotId(1),
                start: rows,
                rows: 1,
            },
        ]
    };
    let refusal = |cfg: &Qwen35Config, rows: usize| {
        let windows = windows(rows);
        geometry_refusal(
            &DeltaNetLayerShape::from_config(cfg),
            &JointRows {
                slots: &slots,
                windows: &windows,
                width: 32,
                attention_tile: None,
                mixer_fused: true,
            },
        )
    };
    let base = synthetic_cfg();
    assert_eq!(refusal(&base, 8), None);
    assert_eq!(refusal(&base, 1), None);
    for taps in [2, 9] {
        let cfg = Qwen35Config {
            linear_conv_kernel_dim: taps,
            ..synthetic_cfg()
        };
        assert_eq!(refusal(&cfg, 8), None, "{taps} taps");
    }
    let reason = |cfg: Qwen35Config, rows: usize| refusal(&cfg, rows).unwrap_or("admitted");
    assert!(reason(
        Qwen35Config {
            linear_key_head_dim: 64,
            linear_value_head_dim: 64,
            ..synthetic_cfg()
        },
        8
    )
    .contains("head dimension"));
    assert!(reason(
        Qwen35Config {
            linear_num_key_heads: 3,
            ..synthetic_cfg()
        },
        8
    )
    .contains("multiple of the key heads"));
    for taps in [1, 10] {
        assert!(
            reason(
                Qwen35Config {
                    linear_conv_kernel_dim: taps,
                    ..synthetic_cfg()
                },
                8
            )
            .contains("taps"),
            "{taps} taps"
        );
    }
    assert!(reason(synthetic_cfg(), 9).contains("more than eight rows"));
}

/// The DFlash2 arm with the tiled verify attention and the fused mixer
/// stage as given.
fn arm_with(tile: bool, fused: bool) -> Arm {
    let mut arm = dflash_arm();
    arm.options.verify_attention_tile = tile;
    arm.options.verify_mixer_fused = fused;
    arm
}

/// Whether this process runs the prepared projections (process-wide
/// switches, read once by the routes).
fn prepared_projections() -> bool {
    let on = |name: &str| std::env::var(name).as_deref() == Ok("1");
    on("HF2Q_REPACKED_GGUF_M32") && on("HF2Q_REPACKED_GGUF_M32_PROJ")
}

/// What the fused core does with a geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Core {
    /// It serves every DeltaNet layer of every step.
    Admitted,
    /// It refuses the geometry: the layers keep the stock chain.
    Refused,
}

/// Five consecutive verify steps on carried caches, fused against stock,
/// with full, partial and zero acceptance between them, then two spec-off
/// decode steps from both caches. Rows, captures and every slot's cache
/// digest must be bitwise equal after each step.
fn consecutive_steps(fixture: &Fixture<'_>, tile: bool, core: Core, label: &str) {
    let plan: [&[LaneCase]; 5] = [
        &[lane(0, 7, 3), lane(1, 7, 0), lane(2, 7, 8), lane(3, 4, 1)],
        &[lane(3, 7, 8), lane(0, 5, 0), lane(1, 2, 2)],
        &[lane(2, 0, 1), lane(1, 7, 4), lane(0, 7, 7), lane(3, 0, 0)],
        &[lane(1, 6, 6), lane(2, 7, 2)],
        &[lane(0, 7, 5), lane(1, 7, 0), lane(2, 3, 4), lane(3, 7, 1)],
    ];
    let stock_arm = arm_with(tile, false);
    let fused_arm = arm_with(tile, true);
    let mut stock_cache = (fixture.new_cache)();
    let mut fused_cache = (fixture.new_cache)();
    let mut position = fixture.start;
    for (index, lanes) in plan.iter().enumerate() {
        let name = format!("{label}, step {index}");
        let before = mlx_native::dispatch_count();
        let (stock_rows, stock_captured) = run_dflash_step(
            fixture,
            stock_arm,
            &mut stock_cache,
            &position,
            lanes,
            &name,
        );
        let stock_dispatches = mlx_native::dispatch_count() - before;
        let before = mlx_native::dispatch_count();
        let (fused_rows, fused_captured) = run_dflash_step(
            fixture,
            fused_arm,
            &mut fused_cache,
            &position,
            lanes,
            &name,
        );
        let fused_dispatches = mlx_native::dispatch_count() - before;
        eprintln!(
            "fused mixer, {name}: {stock_dispatches} stock and {fused_dispatches} fused dispatches"
        );
        match core {
            Core::Admitted => assert!(
                stock_dispatches >= fused_dispatches + 100,
                "{name}: the step did not take the fused core: {stock_dispatches} stock and \
                 {fused_dispatches} fused dispatches"
            ),
            // With stock projections nothing of the stage applies; with
            // prepared ones the wave and the BF16 norm still do.
            Core::Refused if prepared_projections() => assert!(
                fused_dispatches <= stock_dispatches,
                "{name}: {stock_dispatches} stock and {fused_dispatches} fused dispatches"
            ),
            Core::Refused => assert_eq!(
                fused_dispatches, stock_dispatches,
                "{name}: a refused geometry changed the dispatches"
            ),
        }
        assert!(fused_rows == stock_rows, "{name}: rows differ");
        assert!(fused_captured == stock_captured, "{name}: captures differ");
        assert_eq!(
            digests(fixture, &fused_cache),
            digests(fixture, &stock_cache),
            "{name}: caches differ"
        );
        for lane in lanes.iter() {
            position[lane.slot as usize] += lane.accepted as i32;
        }
    }
    // The spec-off decode continues from both caches identically.
    for step in 0..2 {
        let slots = [0_u32, 1, 2, 3];
        let positions = slots.map(|slot| position[slot as usize] + step);
        let stock = spec_off_step(fixture, &mut stock_cache, &slots, &positions);
        let fused = spec_off_step(fixture, &mut fused_cache, &slots, &positions);
        assert!(
            stock == fused,
            "{label}: spec-off continuation {step} differs"
        );
    }
    assert_eq!(
        digests(fixture, &stock_cache),
        digests(fixture, &fused_cache),
        "{label}: caches differ after the spec-off continuation"
    );
}

/// Review829 `review829_consecutive_steps`.
#[test]
fn dflash2_fused_mixer_consecutive_steps_are_bitwise_the_stock_route() {
    with_synthetic_fixture(padded_policy(), dflash_arm(), |fixture| {
        consecutive_steps(fixture, false, Core::Admitted, "128-wide heads, four taps");
    });
}

/// Review829 `review829_fallback_head_dim_64`: the fused core is not
/// admitted, so every DeltaNet layer keeps the stock chain (behind the wave
/// and the BF16 norm when the projections are prepared).
#[test]
fn dflash2_fused_mixer_keeps_the_stock_chain_at_head_dim_64() {
    let cfg = Qwen35Config {
        linear_key_head_dim: 64,
        linear_value_head_dim: 64,
        ..synthetic_cfg()
    };
    with_synthetic_fixture_cfg(padded_policy(), dflash_arm(), cfg, |fixture| {
        consecutive_steps(fixture, false, Core::Refused, "head dimension 64");
    });
}

/// Review829 `review829_conv_taps`, admitted half: five convolution taps.
#[test]
fn dflash2_fused_mixer_serves_five_conv_taps() {
    let cfg = Qwen35Config {
        linear_conv_kernel_dim: 5,
        ..synthetic_cfg()
    };
    with_synthetic_fixture_cfg(padded_policy(), dflash_arm(), cfg, |fixture| {
        consecutive_steps(fixture, false, Core::Admitted, "five taps");
    });
}

/// Review829 `review829_conv_taps`, refused half: ten convolution taps.
#[test]
fn dflash2_fused_mixer_keeps_the_stock_chain_at_ten_conv_taps() {
    let cfg = Qwen35Config {
        linear_conv_kernel_dim: 10,
        ..synthetic_cfg()
    };
    with_synthetic_fixture_cfg(padded_policy(), dflash_arm(), cfg, |fixture| {
        consecutive_steps(fixture, false, Core::Refused, "ten taps");
    });
}

/// The served DeltaNet head counts (16 key heads; `value_heads` value heads)
/// with an attention geometry the tiled verify attention admits
/// (`query_heads` query heads on one KV head), with the serial and with the
/// tiled attention in both arms.
fn served_heads(value_heads: u32, query_heads: u32) {
    let cfg = Qwen35Config {
        linear_num_key_heads: 16,
        linear_num_value_heads: value_heads,
        num_attention_heads: query_heads,
        num_key_value_heads: 1,
        ..synthetic_cfg()
    };
    with_synthetic_fixture_cfg(padded_policy(), dflash_arm(), cfg, |fixture| {
        for (tile, attention) in [(false, "serial"), (true, "tiled")] {
            consecutive_steps(
                fixture,
                tile,
                Core::Admitted,
                &format!(
                    "16/{value_heads} heads, {query_heads} query heads, {attention} attention"
                ),
            );
        }
    });
}

/// Review829 `review829_real_heads_36`: the Qwen3.6 head counts.
#[test]
fn dflash2_fused_mixer_at_qwen36_head_counts_with_tiled_attention() {
    served_heads(32, 8);
}

/// Review829 `review829_real_heads_38`: the Qwen3.8 head counts.
#[test]
fn dflash2_fused_mixer_at_qwen38_head_counts_with_tiled_attention() {
    served_heads(48, 6);
}
