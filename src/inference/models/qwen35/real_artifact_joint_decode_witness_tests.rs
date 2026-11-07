//! ADR-063 Q4-TRACE G1: the joint-decode forward witness is bitwise neutral.
//!
//! For each logical width of the padded `mul_mv_ext` mode (packed width
//! always 4), two identically prefilled caches advance the same cohort for a
//! few steps: once with the observer unarmed and once armed, with a native
//! submission observer installed for both arms. Every row's logits and
//! post-norm hidden, and every slot's cursor, parity, KV codes/scales and
//! conv/recurrent bytes, must be bitwise equal. The armed arm must actually
//! take the hook: one input manifest, one witness, no unmapped pre-norm and
//! one end marker per forward. The model binds the padded policy `hf2q serve`
//! resolves for a Qwen3.8-named path (#729). The test arms process-global
//! state, so run it alone:
//!
//! ```text
//! HF2Q_QWEN38_GGUF=/path/qwen3.8.gguf \
//!   cargo test --release --bin hf2q \
//!   inference::models::qwen35::real_artifact_joint_decode_witness_tests::qwen38_joint_decode_witness_armed_rows_are_bitwise_unarmed \
//!   -- --ignored --exact
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use mlx_native::submission_observer::{
    ObservationKind, ObserverSinkError, SubmissionEvent, SubmissionObserver,
};

use super::joint_decode_routes::{JointDenseFfnRoute, JointRouteMode};
use super::model::Qwen35Model;
use super::real_artifact_joint_decode_invariance_tests::{
    cache_with_capacity, slot_state, SlotState,
};
use super::real_artifact_joint_decode_tests::{
    argmax, bits, hidden_bits, load, prompt, GateRouting, SLOTS,
};
use super::spec_decode::positions_for_range;
use super::{Qwen35JointOutput, Qwen35JointRow};
use crate::inference::native_observation::{armed_transaction_for_test, take_hook_counts};
use crate::serve::multi_seq_kv::SlotId;

pub(super) const STEPS: usize = 3;
const CAPACITY: u32 = 160;
pub(super) const COHORTS: [(&str, &[u32]); 4] = [
    ("logical4_pad0", &[0, 1, 2, 3]),
    ("logical3_pad1", &[0, 1, 3]),
    ("logical2_pad2", &[1, 2]),
    ("logical1_pad3", &[2]),
];

/// Observed dispatches, and those tagged with a current scope.
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
static SCOPED_DISPATCHES: AtomicU64 = AtomicU64::new(0);
struct CountingSink;
impl SubmissionObserver for CountingSink {
    fn record(&self, event: &SubmissionEvent) -> Result<(), ObserverSinkError> {
        if event.kind == ObservationKind::DispatchEncoded {
            DISPATCHES.fetch_add(1, Ordering::Relaxed);
            if event.scope.is_some() {
                SCOPED_DISPATCHES.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(())
    }
}

/// Every row's per-step logits and hidden bits, then every slot's state.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct ArmTrace {
    logits: Vec<Vec<Vec<u32>>>,
    hidden: Vec<Vec<Vec<u32>>>,
    states: Vec<SlotState>,
}

fn owners(cohort: &[u32]) -> Vec<crate::serve::phase_trace::SlotState> {
    cohort
        .iter()
        .map(|&slot| crate::serve::phase_trace::SlotState {
            request_id: 100 + u64::from(slot),
            slot_id: slot,
            generation: 0,
            phase: "decode",
            active: true,
            forward_eligibility: Some(true),
            runnable_forward: None,
            runnable_emit: None,
            inflight: None,
            blocked: None,
        })
        .collect()
}

pub(super) fn run_arm(model: &Qwen35Model, label: &str, cohort: &[u32], armed: bool) -> ArmTrace {
    let hidden_size = model.cfg.hidden_size as usize;
    let mut cache = cache_with_capacity(model, CAPACITY);
    let mut next = Vec::with_capacity(SLOTS as usize);
    let mut position = Vec::with_capacity(SLOTS as usize);
    for slot in 0..SLOTS {
        let tokens = prompt(slot, 0);
        let logits = model
            .forward_gpu_last_logits(
                &tokens,
                &positions_for_range(0, tokens.len()),
                &mut cache,
                SlotId(slot),
            )
            .expect("prefill");
        next.push(argmax(&logits));
        position.push(tokens.len() as i32);
    }
    let mut logits = Vec::with_capacity(STEPS);
    let mut hidden = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let rows = cohort
            .iter()
            .map(|&slot| Qwen35JointRow {
                slot_id: SlotId(slot),
                token: next[slot as usize],
                position: position[slot as usize],
                output: Qwen35JointOutput::LogitsAndHidden,
            })
            .collect::<Vec<_>>();
        take_hook_counts();
        let outputs = {
            let _transaction = armed.then(|| armed_transaction_for_test(owners(cohort)));
            model.forward_gpu_decode_joint(
                &rows,
                &mut cache,
                JointRouteMode::PaddedMvExt,
                JointDenseFfnRoute::SharedWeights,
            )
        }
        .unwrap_or_else(|error| panic!("{label} armed={armed} step {step}: {error:#}"));
        // [inputs, witnesses, unmapped, end markers] for this one forward.
        let expected = if armed { [1, 1, 0, 1] } else { [0; 4] };
        assert_eq!(
            take_hook_counts(),
            expected,
            "{label} armed={armed} step {step}: hook decisions"
        );
        let mut step_logits = Vec::with_capacity(rows.len());
        let mut step_hidden = Vec::with_capacity(rows.len());
        for (row, output) in rows.iter().zip(outputs) {
            let row_logits = output.logits.expect("row logits");
            step_hidden.push(hidden_bits(
                output.nextn_hidden.as_ref().expect("row hidden"),
                hidden_size,
            ));
            next[row.slot_id.0 as usize] = argmax(&row_logits);
            position[row.slot_id.0 as usize] += 1;
            step_logits.push(bits(&row_logits));
        }
        logits.push(step_logits);
        hidden.push(step_hidden);
    }
    ArmTrace {
        logits,
        hidden,
        states: (0..SLOTS).map(|slot| slot_state(&cache, slot)).collect(),
    }
}

#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF (a Qwen3.8-named path), Apple Metal; run alone with --exact"]
fn qwen38_joint_decode_witness_armed_rows_are_bitwise_unarmed() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    mlx_native::submission_observer::install_submission_observer(
        Arc::new(CountingSink),
        16_000_000,
        16 * 1024 * 1024,
    )
    .expect("install the native submission observer (run this test alone)");
    let model = load("HF2Q_QWEN38_GGUF", GateRouting::Serve);
    assert_eq!(
        model.joint_decode_mode(),
        Ok(JointRouteMode::PaddedMvExt),
        "official artifact must be admitted in padded mul_mv_ext mode"
    );
    for (label, cohort) in COHORTS {
        let before = DISPATCHES.load(Ordering::Relaxed);
        let unarmed = run_arm(&model, label, cohort, false);
        assert_eq!(
            DISPATCHES.load(Ordering::Relaxed),
            before,
            "{label}: an unarmed forward is not observed"
        );
        let scoped = SCOPED_DISPATCHES.load(Ordering::Relaxed);
        let armed = run_arm(&model, label, cohort, true);
        assert!(
            SCOPED_DISPATCHES.load(Ordering::Relaxed) > scoped,
            "{label}: the armed forward was not observed"
        );
        assert_eq!(unarmed.logits.len(), STEPS);
        assert!(
            armed.logits == unarmed.logits,
            "{label}: logits differ armed vs unarmed"
        );
        assert!(
            armed.hidden == unarmed.hidden,
            "{label}: post-norm hidden differs armed vs unarmed"
        );
        assert!(
            armed.states == unarmed.states,
            "{label}: cache state differs armed vs unarmed"
        );
    }
}
