//! A DFlash2 verify step on a host without the Metal tensor API
//! ([`crate::inference::route_host`]).
//!
//! The prepared routes exist only on that API. With every prepared-route
//! switch on, the tiled verify attention and the fused mixer requested, a
//! step on such a host must still run, and with the prepared switches alone
//! it must be the stock step: the same dispatches and the same bits.
//!
//! The host is the thread's ([`HostForTest::without_tensor_api`]): the
//! admission points answer as on a host without the API, and the model's
//! registry compiles as that host's runtime compiler does, so on any test
//! host the stock kernels are the portable ones. On CI's hosted runners that
//! is also the device's own answer.

use super::joint_verify_tests::{
    dflash_arm, lane, padded_policy, run_dflash_case, synthetic_cfg, with_synthetic_fixture_cfg,
    Arm, Fixture, LaneCase,
};
use super::prepared_step_tests::whole_steps_repeat;
use super::prepared_switch_override::{Override, PreparedSwitches};
use super::Qwen35Config;
use crate::inference::route_host::HostForTest;

const LANES: [LaneCase; 4] = [lane(0, 7, 0), lane(1, 7, 2), lane(2, 7, 5), lane(3, 7, 8)];

fn with_arm<'a>(fixture: &Fixture<'a>, arm: Arm) -> Fixture<'a> {
    Fixture {
        model: fixture.model,
        arm,
        new_cache: fixture.new_cache,
        start: fixture.start,
        token: fixture.token,
    }
}

#[test]
fn route_host_without_the_tensor_api_keeps_a_dflash2_verify_step_on_the_stock_route() {
    if !whole_steps_repeat() {
        return;
    }
    let _host = HostForTest::without_tensor_api();
    // The Qwen3.8 DeltaNet head counts, with an attention geometry the tiled
    // verify attention has a kernel for (six query heads on one KV head).
    let cfg = Qwen35Config {
        linear_num_key_heads: 16,
        linear_num_value_heads: 48,
        num_attention_heads: 6,
        num_key_value_heads: 1,
        ..synthetic_cfg()
    };
    with_synthetic_fixture_cfg(padded_policy(), dflash_arm(), cfg, |fixture| {
        let step = |switches: PreparedSwitches, arm: Arm, label: &str| {
            let _switches = Override::set(switches);
            let fixture = with_arm(fixture, arm);
            // The first step under a switch would prepare its planes.
            run_dflash_case(&fixture, &LANES, label);
            let before = mlx_native::dispatch_count();
            let trace = run_dflash_case(&fixture, &LANES, label);
            (trace, mlx_native::dispatch_count() - before)
        };
        let (stock, stock_dispatches) = step(PreparedSwitches::OFF, dflash_arm(), "stock");

        // Every prepared-route switch on: the host cannot run the prepared
        // kernels, so each product keeps its stock kernel.
        let (prepared, prepared_dispatches) =
            step(PreparedSwitches::ALL, dflash_arm(), "prepared switches");
        assert_eq!(
            prepared_dispatches, stock_dispatches,
            "the step with the prepared switches did not take the stock route"
        );
        assert!(
            prepared == stock,
            "the step with the prepared switches differs from the stock step"
        );

        // The tiled verify attention probes its pipelines per layer and
        // keeps the per-row chain where they do not build.
        let mut tiled_arm = dflash_arm();
        tiled_arm.options.verify_attention_tile = true;
        let (tiled, tiled_dispatches) = step(PreparedSwitches::ALL, tiled_arm, "tiled attention");
        assert_eq!(
            tiled_dispatches, stock_dispatches,
            "the tiled verify attention did not keep the per-row chain"
        );
        assert!(
            tiled == stock,
            "the step with the tiled attention requested differs from the stock step"
        );

        // The whole profile of a verify step: the fused mixer core is
        // portable and runs; its rows are the stock step's (Class A).
        let mut profile_arm = tiled_arm;
        profile_arm.options.verify_mixer_fused = true;
        let (profile, _) = step(PreparedSwitches::ALL, profile_arm, "route profile");
        assert!(
            profile.rows == stock.rows && profile.committed == stock.committed,
            "the step with the route profile's switches differs from the stock step"
        );
    });
}
