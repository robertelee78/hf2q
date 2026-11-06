//! ADR-063 Q4-TRACE-GPU-TIMING G2: the GPU-timestamp stream is bitwise neutral.
//!
//! This is Q4-TRACE G1 with the native observer installed with the per-command GPU
//! timing stream. For each padded joint width, two identically prefilled caches advance
//! the same cohort unarmed and armed; every row's logits and hidden and every slot's
//! cache bytes must be bitwise equal. Each armed forward must emit exactly one timing
//! event per observed command, all `Available`: every armed joint command (embed, the
//! layer chains, the output head) is submitted and checked. Run alone:
//!
//! ```text
//! HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1 HF2Q_QWEN38_GGUF=/path/qwen38.gguf \
//!   cargo test --release --bin hf2q \
//!   inference::models::qwen35::real_artifact_joint_decode_gpu_timing_tests::qwen38_joint_decode_gpu_timing_armed_rows_are_bitwise_unarmed \
//!   -- --ignored --exact
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use mlx_native::submission_observer::{
    CompletedGpuTimingEvent, ObservationKind, ObserverSinkError, SubmissionEvent,
    SubmissionObserver,
};
use mlx_native::CompletedGpuTimestamps;

use super::joint_decode_routes::JointRouteMode;
use super::real_artifact_joint_decode_tests::load;
use super::real_artifact_joint_decode_witness_tests::{run_arm, COHORTS, STEPS};

/// Qwen3.8 joint forward: embed, 16 four-layer chains, output head.
const COMMANDS_PER_FORWARD: u64 = 18;

static CREATED: AtomicU64 = AtomicU64::new(0);
static AVAILABLE: AtomicU64 = AtomicU64::new(0);
static UNKNOWN: AtomicU64 = AtomicU64::new(0);
struct CountingSink;
impl SubmissionObserver for CountingSink {
    fn record(&self, event: &SubmissionEvent) -> Result<(), ObserverSinkError> {
        if event.kind == ObservationKind::Created {
            CREATED.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
    fn record_gpu_timing(&self, event: &CompletedGpuTimingEvent) -> Result<(), ObserverSinkError> {
        match event.timestamps {
            CompletedGpuTimestamps::Available {
                start_seconds,
                end_seconds,
                ..
            } if start_seconds < end_seconds => AVAILABLE.fetch_add(1, Ordering::Relaxed),
            _ => UNKNOWN.fetch_add(1, Ordering::Relaxed),
        };
        Ok(())
    }
}

fn counts() -> [u64; 3] {
    [&CREATED, &AVAILABLE, &UNKNOWN].map(|counter| counter.load(Ordering::Relaxed))
}

#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF, HF2Q_DECODE_MVN=0, HF2Q_DECODE_MV_EXT=1, Apple Metal; run alone with --exact"]
fn qwen38_joint_decode_gpu_timing_armed_rows_are_bitwise_unarmed() {
    for (name, value) in [("HF2Q_DECODE_MVN", "0"), ("HF2Q_DECODE_MV_EXT", "1")] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(value),
            "the padded arm requires externally-set {name}={value}"
        );
    }
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    mlx_native::submission_observer::install_submission_observer_with_gpu_timestamps(
        Arc::new(CountingSink),
        16_000_000,
        16 * 1024 * 1024,
    )
    .expect("install the native observer with GPU timestamps (run this test alone)");
    let model = load("HF2Q_QWEN38_GGUF");
    assert_eq!(
        model.joint_decode_mode(),
        Ok(JointRouteMode::PaddedMvExt),
        "official artifact must be admitted in padded mul_mv_ext mode"
    );
    for (label, cohort) in COHORTS {
        let before = counts();
        let unarmed = run_arm(&model, label, cohort, false);
        assert_eq!(
            counts(),
            before,
            "{label}: an unarmed forward is not observed"
        );
        let armed = run_arm(&model, label, cohort, true);
        let after = counts();
        let [created, available, unknown] = [0, 1, 2].map(|index| after[index] - before[index]);
        assert_eq!(
            created,
            STEPS as u64 * COMMANDS_PER_FORWARD,
            "{label}: observed commands per armed run"
        );
        assert_eq!(
            available, created,
            "{label}: one Available timing per command"
        );
        assert_eq!(
            unknown, 0,
            "{label}: no Unknown timing for submitted joint commands"
        );
        assert!(
            armed == unarmed,
            "{label}: rows or cache state differ armed vs unarmed"
        );
    }
}
