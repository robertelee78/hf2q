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
//! Every whole-step case runs with the stock projections and with every
//! prepared-route switch on, set per test through
//! [`prepared_switch_override`](super::prepared_switch_override); with the
//! prepared routes a layer's input products run as one wave behind a norm
//! that writes BF16.
//!
//! Two layouts, as in `prepared_step_tests`. The dense layout (Qwen3.8)
//! verifies in a fixed 32-row tile. The MoE layout (Qwen3.6) sizes the tile
//! by its rows (4, 8, 16 or 32) and runs the exact mode; the synthetic model
//! declares the MoE config but keeps quantized dense FFN layers. There the
//! wave and the BF16 norm must follow the prepared routes' width rule:
//! taken below 32 rows only with `HF2Q_REPACKED_GGUF_ANY_WIDTH`. Real MoE
//! layers are covered by the ignored official-artifact gate
//! `real_artifact_dflash2_width_tests::
//! qwen36_dflash2_fused_mixer_is_bitwise_the_stock_route_at_every_tile_width`.
//!
//! Whole steps are compared bit for bit, so, like `prepared_step_tests`,
//! these tests skip under Metal shader validation, where a whole joint step
//! of the synthetic model does not always repeat its bits whatever its
//! route. The kernels are gated under validation in mlx-native.

use super::joint_verify_tests::{
    dflash_arm, digests, exact_policy, lane, padded_policy, run_dflash_case, run_dflash_step,
    spec_off_step, synthetic_cfg, with_synthetic_fixture, with_synthetic_fixture_cfg,
    with_synthetic_fixture_embedding, Arm, Fixture, LaneCase,
};
use super::prepared_step_tests::{
    assert_same_lane, dflash2_verify, exact_arm, moe_layout_cfg, whole_steps_repeat, Tokens, TARGET,
};
use super::prepared_switch_override::{Override, PreparedSwitches};
use super::{Qwen35Config, Qwen35JointOutput, Qwen35JointVerifyLane, Qwen35LayerKind};
use crate::serve::multi_seq_kv::SlotId;
use mlx_native::metal::MTLGPUFamily;
use mlx_native::{KernelRegistry, MlxDevice};

/// The projection routes every whole-step case runs under.
const ROUTES: [(&str, PreparedSwitches); 2] = [
    ("stock projections", PreparedSwitches::OFF),
    ("prepared routes", PreparedSwitches::ALL),
];

/// DeltaNet and full-attention layer counts of a config.
fn layer_counts(cfg: &Qwen35Config) -> (u64, u64) {
    let full = cfg
        .layer_types
        .iter()
        .filter(|kind| **kind == Qwen35LayerKind::FullAttention)
        .count() as u64;
    (cfg.layer_types.len() as u64 - full, full)
}

/// Fewest dispatches the fused core saves per DeltaNet layer: the stock
/// chain of one single-row window is eight dispatches (two for the
/// convolution, the Q and K norms, the query scale, the gates, the
/// recurrence, the gated norm) and every further row, window or padding row
/// adds to it; the fused core is one.
const CORE_SAVES_AT_LEAST: u64 = 7;

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
    if !whole_steps_repeat() {
        return;
    }
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
        for ((routes, switches), (name, lanes, fewer)) in ROUTES
            .into_iter()
            .flat_map(|routes| cases.into_iter().map(move |case| (routes, case)))
        {
            let _switches = Override::set(switches);
            let name = &format!("{routes}, {name}");
            // The first step under a switch prepares its planes.
            run_dflash_case(&stock_fixture, lanes, name);
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

/// `base` with the fused mixer stage as given.
fn with_mixer(base: Arm, fused: bool) -> Arm {
    let mut arm = base;
    arm.options.verify_mixer_fused = fused;
    arm
}

/// The dense-layout DFlash2 arm with the tiled verify attention as given.
fn dense_arm(tile: bool) -> Arm {
    let mut arm = dflash_arm();
    arm.options.verify_attention_tile = tile;
    arm
}

/// What the fused core does with a geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Core {
    /// It serves every DeltaNet layer of every step.
    Admitted,
    /// It refuses the geometry: the layers keep the stock chain.
    Refused,
}

/// Verify steps of the dense layout: every step is a 32-row tile.
const DENSE_PLAN: [&[LaneCase]; 5] = [
    &[lane(0, 7, 3), lane(1, 7, 0), lane(2, 7, 8), lane(3, 4, 1)],
    &[lane(3, 7, 8), lane(0, 5, 0), lane(1, 2, 2)],
    &[lane(2, 0, 1), lane(1, 7, 4), lane(0, 7, 7), lane(3, 0, 0)],
    &[lane(1, 6, 6), lane(2, 7, 2)],
    &[lane(0, 7, 5), lane(1, 7, 0), lane(2, 3, 4), lane(3, 7, 1)],
];

/// Verify steps of the MoE layout, whose tile follows its rows: 8 rows
/// (one lane, then two short ones), 16, 4 (a single row), 16 with padding
/// rows, 32.
const MOE_PLAN: [&[LaneCase]; 6] = [
    &[lane(2, 7, 3)],
    &[lane(0, 2, 3), lane(2, 4, 0)],
    &[lane(1, 7, 8), lane(3, 7, 2)],
    &[lane(0, 0, 1)],
    &[lane(1, 3, 2), lane(2, 7, 8), lane(3, 0, 0)],
    &[lane(0, 7, 5), lane(1, 7, 0), lane(2, 7, 8), lane(3, 7, 1)],
];

/// Consecutive verify steps of `plan` on carried caches, `base` with the
/// fused mixer against `base` without it, with full, partial and zero
/// acceptance between them, then two spec-off decode steps from both
/// caches, under each of [`ROUTES`]. Rows, captures and every slot's cache
/// digest must be bitwise equal after each step.
fn consecutive_steps(
    fixture: &Fixture<'_>,
    base: Arm,
    core: Core,
    plan: &[&[LaneCase]],
    label: &str,
) {
    let (delta_layers, _) = layer_counts(&fixture.model.cfg);
    for (routes, switches) in ROUTES {
        let _switches = Override::set(switches);
        let label = &format!("{label}, {routes}");
        let stock_arm = with_mixer(base, false);
        let fused_arm = with_mixer(base, true);
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
                    stock_dispatches >= fused_dispatches + CORE_SAVES_AT_LEAST * delta_layers,
                    "{name}: the step did not take the fused core: {stock_dispatches} stock and \
                     {fused_dispatches} fused dispatches"
                ),
                // With stock projections nothing of the stage applies; with
                // prepared ones the wave and the BF16 norm still do (and the
                // stock arm's first step prepares the planes).
                Core::Refused if switches.projections => assert!(
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
}

/// Review829 `review829_consecutive_steps`.
#[test]
fn dflash2_fused_mixer_consecutive_steps_are_bitwise_the_stock_route() {
    if !whole_steps_repeat() {
        return;
    }
    with_synthetic_fixture(padded_policy(), dflash_arm(), |fixture| {
        consecutive_steps(
            fixture,
            dense_arm(false),
            Core::Admitted,
            &DENSE_PLAN,
            "128-wide heads, four taps",
        );
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
    if !whole_steps_repeat() {
        return;
    }
    with_synthetic_fixture_cfg(padded_policy(), dflash_arm(), cfg, |fixture| {
        consecutive_steps(
            fixture,
            dense_arm(false),
            Core::Refused,
            &DENSE_PLAN,
            "head dimension 64",
        );
    });
}

/// Review829 `review829_conv_taps`, admitted half: five convolution taps.
#[test]
fn dflash2_fused_mixer_serves_five_conv_taps() {
    let cfg = Qwen35Config {
        linear_conv_kernel_dim: 5,
        ..synthetic_cfg()
    };
    if !whole_steps_repeat() {
        return;
    }
    with_synthetic_fixture_cfg(padded_policy(), dflash_arm(), cfg, |fixture| {
        consecutive_steps(
            fixture,
            dense_arm(false),
            Core::Admitted,
            &DENSE_PLAN,
            "five taps",
        );
    });
}

/// Review829 `review829_conv_taps`, refused half: ten convolution taps.
#[test]
fn dflash2_fused_mixer_keeps_the_stock_chain_at_ten_conv_taps() {
    let cfg = Qwen35Config {
        linear_conv_kernel_dim: 10,
        ..synthetic_cfg()
    };
    if !whole_steps_repeat() {
        return;
    }
    with_synthetic_fixture_cfg(padded_policy(), dflash_arm(), cfg, |fixture| {
        consecutive_steps(
            fixture,
            dense_arm(false),
            Core::Refused,
            &DENSE_PLAN,
            "ten taps",
        );
    });
}

/// Whether mlx-native can route this host's dense matmuls of more than eight
/// rows, the 32-row products every step here is made of. Its route first
/// asks whether the tensor pipeline `kernel_mul_mm_q4_0_tensor_f32` builds on
/// the device (the probe is that kernel whatever the weight's type), and
/// takes the simdgroup kernels when the runtime compiler has no
/// `<metal_tensor>` header. The same build is asked here.
///
/// Native defect: mlx-native 0.21.9 treats every other failure of that build
/// as an error of the dispatch instead of taking the simdgroup kernels. A
/// device outside the Metal 4 GPU family under macOS 26 has one: GitHub's
/// hosted runner ("Apple Paravirtual device") compiles the shader and fails
/// pipeline creation with `unsupported deferred-static-alloca-size function
/// body in ...cooperative_tensor...`, so no dense matmul of more than eight
/// rows runs there, at any shape. The fix is mlx-native branch
/// `fix/tensor-route-availability` (such a device has no tensor API: the
/// route falls back). This skip waits for the release that carries it and
/// goes away with that pin.
///
/// A device of the Metal 4 family is never skipped: a failed build there is a
/// defect and the step reports it.
fn dense_matmul_route_resolves() -> bool {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return true;
    };
    let metal = device.metal_device();
    let mut registry = KernelRegistry::new();
    let Err(error) = registry
        .get_pipeline_with_constants(
            "kernel_mul_mm_q4_0_tensor_f32",
            metal,
            &[],
            &[(700, 1), (701, 1), (702, 1)],
        )
        .map(|_| ())
    else {
        return true;
    };
    let error = error.to_string();
    if error.contains("'metal_tensor' file not found")
        || metal.supports_family(MTLGPUFamily::Metal4)
    {
        return true;
    }
    eprintln!(
        "RECORDED SKIP: mlx-native 0.21.9 cannot route a dense matmul of more than eight rows \
         on {} (no Metal 4 GPU family; the tensor probe fails: {error})",
        device.name()
    );
    false
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
    if !whole_steps_repeat() || !dense_matmul_route_resolves() {
        return;
    }
    with_synthetic_fixture_cfg(padded_policy(), dflash_arm(), cfg, |fixture| {
        for (tile, attention) in [(false, "serial"), (true, "tiled")] {
            consecutive_steps(
                fixture,
                dense_arm(tile),
                Core::Admitted,
                &DENSE_PLAN,
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

/// MoE layout (Qwen3.6), one verify step per case: the step with the fused
/// mixer is bitwise the step without it at every tile width (4, 8, 16, 32),
/// on the host-token and the GPU-token paths (whose padding rows hold
/// another token), with stock projections, with the prepared routes at 32
/// rows only and with them at every width. The target lane's rows,
/// committed cache and captures are compared.
///
/// The wave and the BF16 norm follow the prepared routes' width rule. Where
/// a layer's input products are prepared, the fused step saves four more
/// dispatches per DeltaNet layer than it does with stock projections (the
/// two F32 to BF16 casts and the two copies that split alpha from beta) and
/// one more per full-attention layer (its cast): at 32 rows whenever the
/// projections are prepared, at 4, 8 and 16 rows only with the any-width
/// switch.
fn moe_layout_fused_mixer(attention_tile: bool) {
    if !whole_steps_repeat() {
        return;
    }
    let cfg = moe_layout_cfg(if attention_tile { 8 } else { 2 });
    let (delta_layers, full_layers) = layer_counts(&cfg);
    let prepared_saves = 4 * delta_layers + full_layers;
    let (policy, arm) = (exact_policy(), exact_arm(attention_tile));
    with_synthetic_fixture_embedding(policy, arm, cfg, true, |fixture| {
        let fused_fixture = Fixture {
            model: fixture.model,
            arm: with_mixer(fixture.arm, true),
            new_cache: fixture.new_cache,
            start: fixture.start,
            token: fixture.token,
        };
        let cohorts: [(&str, &[(u32, usize)], usize); 8] = [
            ("width 4", &[(TARGET, 4)], 4),
            ("width 8", &[(TARGET, 8)], 8),
            ("width 8, two short lanes", &[(0, 3), (TARGET, 5)], 8),
            ("width 16", &[(0, 8), (TARGET, 8)], 16),
            ("width 16, 9 live rows", &[(TARGET, 8), (1, 1)], 16),
            ("width 32, 24 live rows", &[(0, 8), (1, 8), (TARGET, 8)], 32),
            ("width 32, full", &[(0, 8), (1, 8), (TARGET, 8), (3, 8)], 32),
            (
                "width 32, full, permuted",
                &[(3, 8), (TARGET, 8), (1, 8), (0, 8)],
                32,
            ),
        ];
        // Dispatches the fused mixer saves per (routes, cohort), host tokens.
        let mut saved = Vec::new();
        for (routes, switches) in [
            ("stock projections", PreparedSwitches::OFF),
            ("prepared routes at 32 rows", PreparedSwitches::WIDTH_32),
            ("prepared routes at every width", PreparedSwitches::ALL),
        ] {
            let _switches = Override::set(switches);
            let mut row = Vec::new();
            for (label, cohort, width) in cohorts {
                let name = format!("{routes}, {label}");
                for tokens in [Tokens::GpuPoisonedPadding, Tokens::Host] {
                    let stock = dflash2_verify(fixture, cohort, tokens);
                    let fused = dflash2_verify(&fused_fixture, cohort, tokens);
                    assert_same_lane(&fused, &stock, &name);
                }
                // Both arms have prepared their planes by now.
                let before = mlx_native::dispatch_count();
                let stock = dflash2_verify(fixture, cohort, Tokens::Host);
                let stock_dispatches = mlx_native::dispatch_count() - before;
                let before = mlx_native::dispatch_count();
                let fused = dflash2_verify(&fused_fixture, cohort, Tokens::Host);
                let fused_dispatches = mlx_native::dispatch_count() - before;
                assert_same_lane(&fused, &stock, &format!("{name}, repeat"));
                eprintln!(
                    "fused mixer, MoE layout, {name}: {stock_dispatches} stock and \
                     {fused_dispatches} fused dispatches"
                );
                assert!(
                    stock_dispatches >= fused_dispatches + CORE_SAVES_AT_LEAST * delta_layers,
                    "{name}: the step did not take the fused core: {stock_dispatches} stock \
                     and {fused_dispatches} fused dispatches"
                );
                row.push((label, width, stock_dispatches - fused_dispatches));
            }
            saved.push(row);
        }
        let [stock, width_32, every_width] = &saved[..] else {
            panic!("three route settings");
        };
        for (index, &(label, width, stock_saved)) in stock.iter().enumerate() {
            assert_eq!(
                every_width[index].2 - stock_saved,
                prepared_saves,
                "{label}: with the any-width switch the wave and the BF16 norm run at every \
                 tile width"
            );
            assert_eq!(
                width_32[index].2 - stock_saved,
                if width == 32 { prepared_saves } else { 0 },
                "{label}: without the any-width switch the wave and the BF16 norm run at 32 \
                 rows only"
            );
        }
    });
}

#[test]
fn moe_layout_fused_mixer_is_bitwise_the_stock_route_at_every_tile_width() {
    moe_layout_fused_mixer(false);
}

#[test]
fn moe_layout_fused_mixer_is_bitwise_the_stock_route_with_the_tiled_verify_attention() {
    moe_layout_fused_mixer(true);
}

/// MoE layout (Qwen3.6), consecutive verify steps on carried caches at tile
/// widths 8, 8, 16, 4, 16 and 32 with full, partial and zero acceptance
/// between them (a partial commit replays the DeltaNet state from the rows
/// the fused kernel retained), then spec-off decode.
fn moe_layout_consecutive_steps(attention_tile: bool) {
    if !whole_steps_repeat() {
        return;
    }
    let cfg = moe_layout_cfg(if attention_tile { 8 } else { 2 });
    let arm = exact_arm(attention_tile);
    with_synthetic_fixture_cfg(exact_policy(), arm, cfg, |fixture| {
        consecutive_steps(fixture, arm, Core::Admitted, &MOE_PLAN, "MoE layout");
    });
}

#[test]
fn moe_layout_fused_mixer_consecutive_steps_are_bitwise_the_stock_route() {
    moe_layout_consecutive_steps(false);
}

#[test]
fn moe_layout_fused_mixer_consecutive_steps_with_the_tiled_verify_attention() {
    moe_layout_consecutive_steps(true);
}
